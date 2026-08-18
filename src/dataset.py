from pathlib import Path
import random
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from video_processor import VideoProcessor
except ImportError:
    from src.video_processor import VideoProcessor


class LIGVideoDataset(Dataset):
    """
    Dataset для MP4-видео лазерного процесса.

    Один элемент Dataset = один кадр из одного видео на одной относительной
    позиции + параметры лазера + целевое сопротивление.

    ВАЖНО:
    Dataset сначала строится целиком, а train/validation/test затем делятся
    функцией split_dataset_by_video(). Деление выполняется ПО ЦЕЛЫМ ВИДЕО,
    поэтому соседние кадры одного MP4 не могут попасть в разные выборки.
    """

    def __init__(
        self,
        metadata_path,
        video_root,
        frame_size=(128, 128),
        position_step=0.02,
    ):
        self.metadata_path = Path(metadata_path)
        self.video_root = Path(video_root)

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"CSV-файл не найден: {self.metadata_path}")
        if not self.video_root.exists():
            raise FileNotFoundError(f"Папка с видео не найдена: {self.video_root}")

        self.processor = VideoProcessor(
            frame_size=frame_size,
            position_step=position_step,
        )

        # Названия колонок в новом metadata.csv.
        self.sample_column = "sample"
        self.video_id_column = "video_id"
        self.filename_column = "filename"
        self.target_column = "resistance_kOhm_sq"
        self.power_column = "power_mW"
        self.speed_column = "speed_mm_s"
        self.distance_column = "distance_um"

        # Дополнительные поля CSV (не подаются в модель, но сохраняются для контроля).
        self.replicate_column = "replicate"
        self.csv_fps_column = "fps"
        self.resolution_column = "resolution"

        # Параметры нормализации задаются ТОЛЬКО после train/val/test split.
        self.laser_mean = None
        self.laser_std = None

        self.metadata = pd.read_csv(self.metadata_path)
        self.metadata = self.metadata.dropna(how="all").copy()
        self._check_columns(self.metadata)

        # Приводим ключевые строковые поля к единому виду.
        self.metadata[self.sample_column] = (
            self.metadata[self.sample_column]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
        self.metadata[self.video_id_column] = (
            self.metadata[self.video_id_column].astype(str).str.strip()
        )
        self.metadata[self.filename_column] = (
            self.metadata[self.filename_column].astype(str).str.strip()
        )

        self.samples = []
        self.video_count = 0
        self.skipped_videos = []

        self._build_samples()

        if not self.samples:
            raise ValueError(
                "Dataset пуст. Не удалось сформировать ни одного sample."
            )

    def _check_columns(self, df):
        """Проверить наличие обязательных столбцов нового CSV."""
        required_columns = [
            self.sample_column,
            self.video_id_column,
            self.filename_column,
            self.target_column,
            self.power_column,
            self.speed_column,
            self.distance_column,
        ]

        missing = [column for column in required_columns if column not in df.columns]
        if missing:
            raise ValueError(f"В CSV отсутствуют столбцы: {missing}")

        duplicate_keys = df.duplicated(
            subset=[self.sample_column, self.filename_column], keep=False
        )
        if duplicate_keys.any():
            duplicates = df.loc[
                duplicate_keys, [self.sample_column, self.filename_column]
            ].to_dict("records")
            raise ValueError(
                "В CSV есть повторяющиеся пары sample + filename: "
                f"{duplicates[:10]}"
            )

    @staticmethod
    def _extract_video_id(filename):
        """Извлечь S001, S002, ... из имени MP4."""
        match = re.search(r"_S(\d{3})_", filename)
        if match is None:
            raise ValueError(f"Не удалось определить video_id из имени: {filename}")
        return "S" + match.group(1)

    def _find_metadata_row(self, sample_name, video_path):
        """Найти единственную строку CSV по папке Sample и точному имени MP4."""
        sample_mask = (
            self.metadata[self.sample_column].str.casefold()
            == str(sample_name).strip().casefold()
        )
        filename_mask = (
            self.metadata[self.filename_column].str.casefold()
            == video_path.name.strip().casefold()
        )
        rows = self.metadata[sample_mask & filename_mask]

        if len(rows) == 0:
            raise ValueError(
                f"Файл '{video_path.name}' из папки '{sample_name}' не найден в CSV"
            )
        if len(rows) > 1:
            raise ValueError(
                f"Для '{sample_name}/{video_path.name}' найдено несколько строк CSV."
            )

        return rows.iloc[0]

    @staticmethod
    def _to_valid_float(value):
        """
        Преобразовать значение в конечное float.

        Не принимаются:
        NaN, +inf, -inf, строки 'inf', пустые/нечисловые строки.
        """
        if pd.isna(value):
            return None

        try:
            number = float(str(value).strip()) if isinstance(value, str) else float(value)
        except (TypeError, ValueError):
            return None

        if not np.isfinite(number):
            return None

        return number

    def _skip_video(self, sample_name, video_path, video_id, reason):
        """Запомнить причину пропуска видео и вывести понятное сообщение."""
        record = {
            "sample_name": sample_name,
            "video_id": video_id,
            "video_path": str(video_path),
            "reason": str(reason),
        }
        self.skipped_videos.append(record)
        print(
            f"[SKIP] {sample_name} / {video_id or video_path.name}: {reason}"
        )

    def _build_samples(self):
        """Связать MP4 с CSV и сформировать примеры по позициям видео."""
        sample_dirs = sorted(
            path
            for path in self.video_root.iterdir()
            if path.is_dir() and path.name.lower().startswith("sample")
        )

        if not sample_dirs:
            raise ValueError("Не найдено папок Sample 1, Sample 2, ...")

        positions = self.processor.get_position_grid()
        n_checked = 0

        for sample_dir in sample_dirs:
            sample_name = sample_dir.name

            sample_exists = (
                self.metadata[self.sample_column].str.casefold()
                == sample_name.strip().casefold()
            ).any()
            if not sample_exists:
                print(
                    f"[SKIP FOLDER] '{sample_name}': "
                    "нет соответствующих строк в CSV."
                )
                continue

            video_dir = sample_dir / "mp4"
            if not video_dir.is_dir():
                print(f"[INFO] В '{sample_name}' нет подпапки mp4/.")
                continue

            video_files = sorted(video_dir.glob("*.mp4"))
            if not video_files:
                print(f"[INFO] В '{sample_name}/mp4' нет MP4-файлов.")
                continue

            for video_path in video_files:
                video_id = None

                try:
                    video_id = self._extract_video_id(video_path.name)
                    row = self._find_metadata_row(sample_name, video_path)
                    csv_video_id = str(row[self.video_id_column]).strip()
                    if csv_video_id != video_id:
                        raise ValueError(
                            f"video_id в имени ({video_id}) не совпадает с CSV ({csv_video_id})"
                        )

                    target = self._to_valid_float(row[self.target_column])
                    power = self._to_valid_float(row[self.power_column])
                    speed = self._to_valid_float(row[self.speed_column])
                    distance = self._to_valid_float(row[self.distance_column])

                    if target is None:
                        self._skip_video(
                            sample_name,
                            video_path,
                            video_id,
                            "некорректный target (NaN/inf/не число)",
                        )
                        continue

                    if power is None or speed is None or distance is None:
                        self._skip_video(
                            sample_name,
                            video_path,
                            video_id,
                            "некорректные параметры лазера (NaN/inf/не число)",
                        )
                        continue

                    # Эта проверка одновременно подтверждает, что MP4 существует,
                    # открывается и содержит корректное число кадров/FPS.
                    video_info = self.processor.get_video_info(video_path)
                    n_checked += 1
                    if n_checked == 1 or n_checked % 20 == 0:
                        print(f"[INDEX] opened {n_checked} videos...", flush=True)

                    for position in positions:
                        self.samples.append(
                            {
                                "video_path": video_path,
                                "sample_name": sample_name,
                                "video_id": video_id,
                                "position": float(position),
                                "power": power,
                                "speed": speed,
                                "distance": distance,
                                "target": target,
                                "replicate": row.get(self.replicate_column, None),
                                "csv_fps": row.get(self.csv_fps_column, None),
                                "resolution": row.get(self.resolution_column, None),
                                "fps": video_info["fps"],
                                "total_frames": video_info["total_frames"],
                                "duration": video_info["duration"],
                            }
                        )

                    self.video_count += 1

                except Exception as error:
                    self._skip_video(
                        sample_name,
                        video_path,
                        video_id,
                        error,
                    )

    def set_laser_normalization(self, mean, std):
        """
        Задать mean/std, рассчитанные ТОЛЬКО по train-видео.
        После этого power/speed/distance возвращаются в стандартизованном виде.
        """
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)

        if mean.shape != (3,) or std.shape != (3,):
            raise ValueError("mean и std должны содержать ровно 3 значения.")
        if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(std)):
            raise ValueError("mean/std должны быть конечными числами.")
        if np.any(std <= 0):
            raise ValueError("Все значения std должны быть больше нуля.")

        self.laser_mean = mean
        self.laser_std = std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Кадр читается лениво: только когда DataLoader запросил этот элемент.
        frame = self.processor.get_frame(
            sample["video_path"],
            sample["position"],
        )

        laser_values = np.array(
            [sample["power"], sample["speed"], sample["distance"]],
            dtype=np.float32,
        )

        if self.laser_mean is not None and self.laser_std is not None:
            laser_values = (laser_values - self.laser_mean) / self.laser_std

        laser_params = torch.tensor(laser_values, dtype=torch.float32)
        position = torch.tensor([sample["position"]], dtype=torch.float32)
        target = torch.tensor([sample["target"]], dtype=torch.float32)

        return {
            "frame": frame,
            "laser_params": laser_params,
            "target": target,
            "position": position,
            "video_id": sample["video_id"],
            "sample_name": sample["sample_name"],
            "video_path": str(sample["video_path"]),
        }


def _unique_video_records(dataset):
    """Взять по одной записи на каждое уникальное MP4."""
    records = {}

    for sample in dataset.samples:
        key = str(Path(sample["video_path"]).resolve())
        if key not in records:
            records[key] = {
                "video_path": key,
                "sample_name": sample["sample_name"],
                "video_id": sample["video_id"],
                "power": sample["power"],
                "speed": sample["speed"],
                "distance": sample["distance"],
                "target": sample["target"],
            }

    return list(records.values())


def _split_counts(total, train_ratio, val_ratio, test_ratio):
    """Получить ненулевые размеры трех частей, если видео достаточно."""
    if total < 3:
        raise ValueError("Для train/val/test необходимо минимум 3 корректных видео.")

    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=float)

    if np.any(ratios <= 0):
        raise ValueError("Все доли train/val/test должны быть больше нуля.")
    if not np.isclose(ratios.sum(), 1.0):
        raise ValueError("Сумма train_ratio + val_ratio + test_ratio должна быть 1.")

    raw = ratios * total
    counts = np.floor(raw).astype(int)

    # Если данных достаточно, гарантируем хотя бы одно видео в каждой части.
    for i in range(3):
        if counts[i] == 0:
            counts[i] = 1

    # Доводим сумму до total, отдавая/забирая видео по дробным остаткам.
    while counts.sum() < total:
        fractional = raw - np.floor(raw)
        order = np.argsort(-fractional)
        for i in order:
            counts[i] += 1
            if counts.sum() == total:
                break

    while counts.sum() > total:
        # Уменьшаем только части, где останется хотя бы одно видео.
        candidates = [i for i in range(3) if counts[i] > 1]
        if not candidates:
            break
        i = max(candidates, key=lambda j: counts[j] - raw[j])
        counts[i] -= 1

    return tuple(int(x) for x in counts)


def split_dataset_by_video(
    dataset,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
    normalize_laser_params=True,
):
    """
    Разделить Dataset на train/validation/test БЕЗ УТЕЧКИ КАДРОВ.

    Сначала перемешиваются уникальные MP4, затем целое видео назначается
    ровно в одну часть. После этого туда попадают ВСЕ позиции этого видео.

    Если normalize_laser_params=True, mean/std для power/speed/distance
    вычисляются только по train-видео и затем применяются ко всем трем частям.
    """
    video_records = _unique_video_records(dataset)

    rng = random.Random(seed)
    rng.shuffle(video_records)

    train_count, val_count, test_count = _split_counts(
        len(video_records),
        train_ratio,
        val_ratio,
        test_ratio,
    )

    train_records = video_records[:train_count]
    val_records = video_records[train_count:train_count + val_count]
    test_records = video_records[train_count + val_count:]

    train_videos = {record["video_path"] for record in train_records}
    val_videos = {record["video_path"] for record in val_records}
    test_videos = {record["video_path"] for record in test_records}

    # Главная защита от data leakage.
    assert train_videos.isdisjoint(val_videos)
    assert train_videos.isdisjoint(test_videos)
    assert val_videos.isdisjoint(test_videos)

    train_indices = []
    val_indices = []
    test_indices = []

    for idx, sample in enumerate(dataset.samples):
        path = str(Path(sample["video_path"]).resolve())

        if path in train_videos:
            train_indices.append(idx)
        elif path in val_videos:
            val_indices.append(idx)
        elif path in test_videos:
            test_indices.append(idx)
        else:
            raise RuntimeError(f"Видео не попало ни в одну выборку: {path}")

    if normalize_laser_params:
        train_params = np.array(
            [
                [record["power"], record["speed"], record["distance"]]
                for record in train_records
            ],
            dtype=np.float32,
        )

        mean = train_params.mean(axis=0)
        std = train_params.std(axis=0)

        # Если один параметр оказался константой в train, не делим на ноль.
        std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
        dataset.set_laser_normalization(mean, std)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    split_info = {
        "seed": seed,
        "train_videos": train_records,
        "val_videos": val_records,
        "test_videos": test_records,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "laser_mean": None if dataset.laser_mean is None else dataset.laser_mean.copy(),
        "laser_std": None if dataset.laser_std is None else dataset.laser_std.copy(),
    }

    print_split_summary(dataset, split_info)

    return train_dataset, val_dataset, test_dataset, split_info


def _range_text(records, key):
    values = [float(record[key]) for record in records]
    return f"{min(values):g} .. {max(values):g}" if values else "-"


def print_split_summary(dataset, split_info):
    """Показать, что именно попало в train/validation/test."""
    print("\n" + "=" * 68)
    print("DATASET SUMMARY")
    print("=" * 68)
    print(f"Использовано корректных видео: {dataset.video_count}")
    print(f"Пропущено видео:             {len(dataset.skipped_videos)}")
    print(f"Всего кадровых samples:      {len(dataset)}")

    for title, key in [
        ("TRAIN", "train_videos"),
        ("VALIDATION", "val_videos"),
        ("TEST", "test_videos"),
    ]:
        records = split_info[key]
        sample_key = {
            "train_videos": "train_samples",
            "val_videos": "val_samples",
            "test_videos": "test_samples",
        }[key]

        print("\n" + title)
        print(f"  видео:       {len(records)}")
        print(f"  samples:     {split_info[sample_key]}")
        print(f"  power:       {_range_text(records, 'power')}")
        print(f"  speed:       {_range_text(records, 'speed')}")
        print(f"  distance:    {_range_text(records, 'distance')}")
        print(f"  target:      {_range_text(records, 'target')}")

    if split_info["laser_mean"] is not None:
        print("\nНормализация параметров лазера рассчитана ТОЛЬКО по TRAIN:")
        print("  mean [power, speed, distance] =", split_info["laser_mean"])
        print("  std  [power, speed, distance] =", split_info["laser_std"])

    if dataset.skipped_videos:
        print("\nПРОПУЩЕННЫЕ ВИДЕО:")
        for item in dataset.skipped_videos:
            print(
                f"  {item['sample_name']} / "
                f"{item['video_id'] or Path(item['video_path']).name}: "
                f"{item['reason']}"
            )

    print("=" * 68 + "\n")


def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size=32,
    num_workers=0,
    seed=42,
):
    """Создать DataLoader для обучения, валидации и финального теста."""
    if batch_size <= 0:
        raise ValueError("batch_size должен быть больше нуля.")
    if num_workers < 0:
        raise ValueError("num_workers не может быть отрицательным.")

    generator = torch.Generator()
    generator.manual_seed(seed)

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    # На Windows безопасный стартовый вариант — num_workers=0.
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common,
    )

    return train_loader, val_loader, test_loader


def build_datasets_and_loaders(
    metadata_path,
    video_root,
    frame_size=(128, 128),
    position_step=0.02,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    batch_size=32,
    num_workers=0,
    seed=42,
    normalize_laser_params=True,
):
    """
    Удобная функция полного конвейера подготовки данных.

    Возвращает:
        dataset,
        train_dataset, val_dataset, test_dataset,
        train_loader, val_loader, test_loader,
        split_info
    """
    dataset = LIGVideoDataset(
        metadata_path=metadata_path,
        video_root=video_root,
        frame_size=frame_size,
        position_step=position_step,
    )

    train_dataset, val_dataset, test_dataset, split_info = split_dataset_by_video(
        dataset=dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        normalize_laser_params=normalize_laser_params,
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )

    return (
        dataset,
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
        split_info,
    )


LIGDataset = LIGVideoDataset
