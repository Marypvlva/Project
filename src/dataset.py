from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from video_processor import VideoProcessor


class LIGVideoDataset(Dataset):
    """
    Dataset для MP4-видео лазерного процесса.

    Ожидаемая структура:

    data2/
        table.xlsx

        Sample 1/
            *.mp4

        Sample 2/
            *.mp4

        Sample 3/
            *.mp4

        ...

    Один элемент Dataset соответствует:
        одному видео
        +
        одной относительной позиции внутри видео.

    Например:
        position = 0.10 -> 10% видео
        position = 0.50 -> середина видео
        position = 0.90 -> 90% видео

    Возвращает:
        frame:
            Tensor [3, 128, 128]

        laser_params:
            Tensor [3]
            [power, speed, distance_between_lines]

        target:
            Tensor [1]
            итоговое сопротивление

        position:
            Tensor [1]
            положение внутри видео от 0 до 1

        video_id:
            S001, S002, ...

        sample_name:
            Sample 1, Sample 2, ...

        video_path:
            путь к MP4
    """

    def __init__(
        self,
        metadata_path,
        video_root,
        frame_size=(128, 128),
        position_step=0.02
    ):
        """
        Parameters
        ----------
        metadata_path:
            путь к table.xlsx

        video_root:
            корневая папка с Sample 1, Sample 2, ...

        frame_size:
            размер кадра для сети

        position_step:
            шаг по относительной позиции видео.

            Например:
                0.02 -> 2%, 4%, 6%, ..., 98%
                0.05 -> 5%, 10%, 15%, ..., 95%
                0.10 -> 10%, 20%, ..., 90%
        """

        self.metadata_path = Path(
            metadata_path
        )

        self.video_root = Path(
            video_root
        )

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Excel-файл не найден: "
                f"{self.metadata_path}"
            )

        if not self.video_root.exists():
            raise FileNotFoundError(
                f"Папка с видео не найдена: "
                f"{self.video_root}"
            )

        # --------------------------------------------------
        # VideoProcessor
        # --------------------------------------------------

        self.processor = VideoProcessor(
            frame_size=frame_size,
            position_step=position_step
        )

        # --------------------------------------------------
        # Названия колонок Excel
        # --------------------------------------------------

        self.video_id_column = (
            "Название видео"
        )

        self.target_column = (
            "Сопротивление ДО ультразвуковой обработки, кОм"
        )

        self.power_column = (
            "Мощность, mW"
        )

        self.speed_column = (
            "Скорость обработки лазером образцов, мм/мин"
        )

        self.distance_column = (
            "Расстояние между линиями, мкм"
        )

        # --------------------------------------------------
        # Читаем Excel
        # --------------------------------------------------

        self.excel_file = pd.ExcelFile(
            self.metadata_path
        )

        self.metadata = {}

        for sheet_name in self.excel_file.sheet_names:

            # Работаем только с листами Sample ...
            if not sheet_name.lower().startswith(
                "sample"
            ):
                continue

            df = pd.read_excel(
                self.metadata_path,
                sheet_name=sheet_name
            )

            # Удаляем полностью пустые строки
            df = df.dropna(
                how="all"
            )

            self._check_columns(
                df,
                sheet_name
            )

            self.metadata[
                sheet_name
            ] = df

        if len(self.metadata) == 0:
            raise ValueError(
                "В Excel не найдено "
                "ни одного листа Sample."
            )

        # --------------------------------------------------
        # Список samples
        # --------------------------------------------------

        self.samples = []

        # Количество реально использованных видео
        self.video_count = 0

        # Информация о видео,
        # которые пришлось пропустить
        self.skipped_videos = []

        self._build_samples()

        if len(self.samples) == 0:
            raise ValueError(
                "Dataset пуст. "
                "Не удалось сформировать "
                "ни одного sample."
            )

    def _check_columns(
        self,
        df,
        sheet_name
    ):
        """
        Проверить наличие нужных
        столбцов в Excel.
        """

        required_columns = [
            self.video_id_column,
            self.target_column,
            self.power_column,
            self.speed_column,
            self.distance_column
        ]

        for column in required_columns:

            if column not in df.columns:
                raise ValueError(
                    f"На листе '{sheet_name}' "
                    f"отсутствует столбец:\n"
                    f"{column}"
                )

    @staticmethod
    def _extract_video_id(filename):
        """
        Извлечь S001, S002, ...

        Сейчас ожидается, что имя MP4
        всё ещё содержит подстроку вида:

            _S001_
            _S004_
            ...

        Например:

            MiroC110_S004_R00_C00_....mp4

        ->

            S004
        """

        match = re.search(
            r"_S(\d{3})_",
            filename
        )

        if match is None:
            raise ValueError(
                f"Не удалось определить "
                f"video_id из имени:\n"
                f"{filename}"
            )

        return (
            "S"
            + match.group(1)
        )

    def _find_metadata_row(
        self,
        sample_name,
        video_id
    ):
        """
        Найти строку Excel
        для конкретного видео.
        """

        if sample_name not in self.metadata:
            raise ValueError(
                f"В Excel отсутствует лист "
                f"'{sample_name}'"
            )

        df = self.metadata[
            sample_name
        ]

        rows = df[
            df[self.video_id_column]
            .astype(str)
            .str.strip()
            == video_id
        ]

        if len(rows) == 0:
            raise ValueError(
                f"{video_id} не найден "
                f"на листе '{sample_name}'"
            )

        if len(rows) > 1:
            raise ValueError(
                f"Для {video_id} на листе "
                f"'{sample_name}' найдено "
                f"несколько строк."
            )

        return rows.iloc[0]

    @staticmethod
    def _is_valid_number(value):
        """
        Проверить числовое значение.

        Пропускаем:
            NaN
            inf
            "inf"
            нечисловые строки
        """

        if pd.isna(value):
            return False

        if isinstance(
            value,
            str
        ):
            value = (
                value
                .strip()
            )

            if value.lower() == "inf":
                return False

            try:
                value = float(
                    value
                )

            except ValueError:
                return False

        return np.isfinite(
            float(value)
        )

    def _build_samples(self):
        """
        Найти MP4-файлы и сформировать
        samples по относительным позициям.
        """

        # --------------------------------------------------
        # Ищем папки Sample *
        # --------------------------------------------------

        sample_dirs = sorted(
            [
                path
                for path in self.video_root.iterdir()
                if (
                    path.is_dir()
                    and
                    path.name.lower().startswith(
                        "sample"
                    )
                )
            ]
        )

        if len(sample_dirs) == 0:
            raise ValueError(
                "Не найдено папок "
                "Sample 1, Sample 2, ..."
            )

        # --------------------------------------------------
        # Одинаковая сетка позиций
        # для всех видео
        # --------------------------------------------------

        positions = (
            self.processor
            .get_position_grid()
        )

        # --------------------------------------------------
        # Обходим все Sample
        # --------------------------------------------------

        for sample_dir in sample_dirs:

            sample_name = (
                sample_dir.name
            )

            # Если папка есть,
            # а листа Excel нет
            if sample_name not in self.metadata:

                print(
                    f"Папка '{sample_name}' "
                    f"пропущена: "
                    f"нет соответствующего "
                    f"листа Excel."
                )

                continue

            # --------------------------------------------------
            # Ищем MP4
            # --------------------------------------------------

            video_files = sorted(
                sample_dir.glob(
                    "*.mp4"
                )
            )

            if len(video_files) == 0:
                continue

            # --------------------------------------------------
            # Обрабатываем каждое видео
            # --------------------------------------------------

            for video_path in video_files:

                try:

                    # ------------------------------------------
                    # Получаем S001, S004, ...
                    # ------------------------------------------

                    video_id = (
                        self._extract_video_id(
                            video_path.name
                        )
                    )

                    # ------------------------------------------
                    # Ищем строку Excel
                    # ------------------------------------------

                    row = (
                        self._find_metadata_row(
                            sample_name,
                            video_id
                        )
                    )

                    # ------------------------------------------
                    # Target
                    # ------------------------------------------

                    target = row[
                        self.target_column
                    ]

                    if not self._is_valid_number(
                        target
                    ):

                        self.skipped_videos.append(
                            {
                                "sample_name":
                                    sample_name,

                                "video_id":
                                    video_id,

                                "reason":
                                    f"target={target}"
                            }
                        )

                        print(
                            f"Пропущено: "
                            f"{sample_name} / "
                            f"{video_id} "
                            f"(target={target})"
                        )

                        continue

                    # ------------------------------------------
                    # Параметры лазера
                    # ------------------------------------------

                    power = row[
                        self.power_column
                    ]

                    speed = row[
                        self.speed_column
                    ]

                    distance = row[
                        self.distance_column
                    ]

                    params = [
                        power,
                        speed,
                        distance
                    ]

                    if not all(
                        self._is_valid_number(x)
                        for x in params
                    ):

                        self.skipped_videos.append(
                            {
                                "sample_name":
                                    sample_name,

                                "video_id":
                                    video_id,

                                "reason":
                                    "Некорректные "
                                    "laser_params"
                            }
                        )

                        continue

                    power = float(
                        power
                    )

                    speed = float(
                        speed
                    )

                    distance = float(
                        distance
                    )

                    target = float(
                        target
                    )

                    # ------------------------------------------
                    # Проверяем MP4
                    # ------------------------------------------

                    video_info = (
                        self.processor
                        .get_video_info(
                            video_path
                        )
                    )

                    fps = (
                        video_info[
                            "fps"
                        ]
                    )

                    total_frames = (
                        video_info[
                            "total_frames"
                        ]
                    )

                    duration = (
                        video_info[
                            "duration"
                        ]
                    )

                    # ------------------------------------------
                    # Один sample =
                    # одно видео + одна position
                    # ------------------------------------------

                    for position in positions:

                        self.samples.append(
                            {
                                "video_path":
                                    video_path,

                                "sample_name":
                                    sample_name,

                                "video_id":
                                    video_id,

                                "position":
                                    float(
                                        position
                                    ),

                                "power":
                                    power,

                                "speed":
                                    speed,

                                "distance":
                                    distance,

                                "target":
                                    target,

                                # Эти поля не нужны
                                # непосредственно модели,
                                # но полезны для диагностики.
                                "fps":
                                    fps,

                                "total_frames":
                                    total_frames,

                                "duration":
                                    duration
                            }
                        )

                    self.video_count += 1

                except Exception as error:

                    self.skipped_videos.append(
                        {
                            "sample_name":
                                sample_name,

                            "video_path":
                                str(
                                    video_path
                                ),

                            "reason":
                                str(
                                    error
                                )
                        }
                    )

                    print(
                        f"Ошибка при обработке "
                        f"{video_path.name}:\n"
                        f"{error}"
                    )

    def __len__(self):
        """
        Количество samples Dataset.
        """

        return len(
            self.samples
        )

    def __getitem__(
        self,
        idx
    ):
        """
        Получить один sample.

        Изображения заранее в памяти
        не хранятся.

        Только при обращении к dataset[idx]
        читается нужный кадр MP4.
        """

        sample = self.samples[
            idx
        ]

        # --------------------------------------------------
        # Кадр
        # --------------------------------------------------

        frame = (
            self.processor
            .get_frame(
                sample[
                    "video_path"
                ],
                sample[
                    "position"
                ]
            )
        )

        # --------------------------------------------------
        # Параметры лазера
        # --------------------------------------------------

        laser_params = torch.tensor(
            [
                sample[
                    "power"
                ],
                sample[
                    "speed"
                ],
                sample[
                    "distance"
                ]
            ],
            dtype=torch.float32
        )

        # --------------------------------------------------
        # Position
        #
        # Уже находится в диапазоне 0..1,
        # дополнительная нормализация не нужна.
        # --------------------------------------------------

        position = torch.tensor(
            [
                sample[
                    "position"
                ]
            ],
            dtype=torch.float32
        )

        # --------------------------------------------------
        # Target
        # --------------------------------------------------

        target = torch.tensor(
            [
                sample[
                    "target"
                ]
            ],
            dtype=torch.float32
        )

        # --------------------------------------------------
        # Возвращаем sample
        # --------------------------------------------------

        return {
            "frame":
                frame,

            "laser_params":
                laser_params,

            "target":
                target,

            "position":
                position,

            "video_id":
                sample[
                    "video_id"
                ],

            "sample_name":
                sample[
                    "sample_name"
                ],

            "video_path":
                str(
                    sample[
                        "video_path"
                    ]
                )
        }


LIGDataset = LIGVideoDataset
