"""
Обучение регрессора сопротивления поверх предобученного CDAE.

Данные — LIGVideoDataset + split_dataset_by_video (код коллеги, без правок):
  один CSV, деление ПО ЦЕЛЫМ ВИДЕО, z-score params только по train.

Ключи батча:
  frame                         — [B, 3, 128, 128], float [0, 1]
  laser_params / params         — [B, 3] power, speed, distance (z-scored)
  position / time               — [B, 1] относительная позиция в видео [0, 1]
  target                        — [B, 1] сопротивление в кОм

Один сэмпл = один кадр. Один mp4 → много кадров с одним R.
Метрики: logs/regression_metrics.csv. Best по val MAE.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler, Subset

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from cdae_model import (
    ResistanceRegressor,
    build_regressor_from_cdae,
    load_cdae,
    load_regressor,
    save_regressor,
)
from dataset import LIGVideoDataset, split_dataset_by_video

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_ROOT = Path("/home/jupyter/filestore/dataset")
DEFAULT_METADATA = DEFAULT_VIDEO_ROOT / "metadata.csv"
DEFAULT_CDAE_CKPT = PROJECT_ROOT / "checkpoints" / "cdae_weights.pth"
DEFAULT_OUT_CKPT = PROJECT_ROOT / "checkpoints" / "regressor.pt"
DEFAULT_METRICS_CSV = PROJECT_ROOT / "logs" / "regression_metrics.csv"
IMAGE_SIZE = 128

METRICS_FIELDS = [
    "epoch",
    "train_huber",
    "train_mae",
    "train_mse",
    "train_r2",
    "val_huber",
    "val_mae",
    "val_mse",
    "val_r2",
    "is_best",
]


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    """frame + laser_params/params + position/time + target."""
    if not isinstance(batch, dict):
        raise TypeError(f"Ожидался dict-батч, получено {type(batch).__name__}")

    frame = batch["frame"].to(device)
    if "laser_params" in batch:
        params = batch["laser_params"].to(device)
    elif "params" in batch:
        params = batch["params"].to(device)
    elif "process_params" in batch:
        params = batch["process_params"].to(device)
    else:
        raise KeyError("В батче нет 'laser_params' / 'params' / 'process_params'")

    if "position" in batch:
        time = batch["position"].to(device)
    elif "time" in batch:
        time = batch["time"].to(device)
    else:
        raise KeyError("В батче нет 'position' / 'time'")

    target = batch["target"].to(device)
    if target.ndim == 1:
        target = target.unsqueeze(1)
    if time.ndim == 1:
        time = time.unsqueeze(1)

    return {"frame": frame, "params": params, "time": time, "target": target}


def _metrics_from_sums(
    abs_err: float,
    sq_err: float,
    sum_y: float,
    sum_y2: float,
    n: int,
    huber_sum: float,
) -> dict[str, float]:
    if n <= 0:
        raise ValueError("пустой loader: нет сэмплов для метрик")
    mse = sq_err / n
    ss_tot = max(0.0, sum_y2 - (sum_y ** 2) / n)
    r2 = 0.0 if ss_tot < 1e-12 else 1.0 - (sq_err / ss_tot)
    return {
        "huber": huber_sum / n,
        "mae": abs_err / n,
        "mse": mse,
        "r2": r2,
    }


@torch.no_grad()
def evaluate(
    model: ResistanceRegressor,
    loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
) -> dict[str, float]:
    """Huber / MAE / MSE / R² по сопротивлению (кОм)."""
    was_training = model.training
    encoder_was_training = model.encoder.training
    model.eval()
    if criterion is None:
        criterion = nn.HuberLoss(reduction="sum")

    abs_err = sq_err = sum_y = sum_y2 = huber_sum = 0.0
    n = 0

    for batch in loader:
        b = _move_batch(batch, device)
        pred = model(b["frame"], b["params"], b["time"])
        target = b["target"]
        diff = pred - target
        abs_err += diff.abs().sum().item()
        sq_err += (diff ** 2).sum().item()
        sum_y += target.sum().item()
        sum_y2 += (target ** 2).sum().item()
        huber_sum += criterion(pred, target).item()
        n += target.numel()

    if was_training:
        model.train()
        model.encoder.train(encoder_was_training)

    return _metrics_from_sums(abs_err, sq_err, sum_y, sum_y2, n, huber_sum)


def _set_train_modes(model: ResistanceRegressor) -> None:
    model.train()
    if not model.encoder_trainable():
        model.encoder.eval()


def _init_metrics_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=METRICS_FIELDS).writeheader()


def _append_metrics_csv(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=METRICS_FIELDS).writerow(row)


def _format_epoch_line(row: dict[str, Any]) -> str:
    parts = [
        f"epoch {int(row['epoch']):03d}",
        f"train_huber={row['train_huber']:.4f}",
        f"train_mae={row['train_mae']:.4f}",
        f"train_mse={row['train_mse']:.4f}",
        f"train_r2={row['train_r2']:.4f}",
        f"val_huber={row['val_huber']:.4f}",
        f"val_mae={row['val_mae']:.4f}",
        f"val_mse={row['val_mse']:.4f}",
        f"val_r2={row['val_r2']:.4f}",
    ]
    if row.get("is_best"):
        parts.append("[best saved]")
    return "  ".join(parts)


def _to_float_list(values: Any) -> Optional[list[float]]:
    if values is None:
        return None
    return [float(x) for x in list(values)]


class VideoGroupedSubsetSampler(Sampler[int]):
    """
    Кадры одного mp4 идут подряд (по position), порядок видео перемешивается.

    Работает по Subset от split_dataset_by_video: индексы локальные 0..len(subset)-1.
    """

    def __init__(self, subset: Subset, *, seed: int = 42) -> None:
        if not isinstance(subset, Subset):
            raise TypeError("VideoGroupedSubsetSampler ждёт torch Subset")
        base = subset.dataset
        if not hasattr(base, "samples"):
            raise TypeError("Базовый датасет должен иметь .samples")

        groups: dict[str, list[int]] = {}
        for local_idx, global_idx in enumerate(subset.indices):
            sample = base.samples[int(global_idx)]
            key = str(sample["video_path"])
            groups.setdefault(key, []).append(local_idx)
        for local_idxs in groups.values():
            local_idxs.sort(
                key=lambda i: float(base.samples[int(subset.indices[i])]["position"])
            )
        self._groups = list(groups.values())
        self._seed = seed
        self._epoch = 0

    def __iter__(self):
        rng = random.Random(self._seed + self._epoch)
        order = list(self._groups)
        rng.shuffle(order)
        self._epoch += 1
        for group in order:
            yield from group

    def __len__(self) -> int:
        return sum(len(group) for group in self._groups)


def _make_loader(
    subset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    grouped: bool,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if grouped:
        kwargs["sampler"] = VideoGroupedSubsetSampler(subset, seed=seed)
    else:
        kwargs["shuffle"] = False
    return DataLoader(subset, **kwargs)


def train_regression(
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    cdae_ckpt: str | Path = DEFAULT_CDAE_CKPT,
    out_ckpt: str | Path = DEFAULT_OUT_CKPT,
    metrics_csv: str | Path = DEFAULT_METRICS_CSV,
    epochs: int = 50,
    lr: float = 1e-4,
    device: str | torch.device = "cpu",
    unfreeze_encoder_after: Optional[int] = None,
    encoder_lr: float = 1e-5,
    extra_config: Optional[dict[str, Any]] = None,
    test_loader: Optional[DataLoader] = None,
) -> ResistanceRegressor:
    """
    load CDAE → freeze encoder → Huber на R[кОм] →
    evaluate train/val → best по val MAE → вернуть best-веса.
    """
    if len(train_loader) == 0:
        raise ValueError("train_loader пустой")
    if len(val_loader) == 0:
        raise ValueError("val_loader пустой")

    device = torch.device(device)
    out_ckpt = Path(out_ckpt)
    metrics_csv = Path(metrics_csv)
    extra_config = extra_config or {}
    _init_metrics_csv(metrics_csv)

    cdae = load_cdae(cdae_ckpt, map_location=device)
    model = build_regressor_from_cdae(cdae, freeze_encoder=True, use_time=True)
    model.to(device)
    _set_train_modes(model)

    criterion = nn.HuberLoss()
    criterion_sum = nn.HuberLoss(reduction="sum")
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )

    best_mae = float("inf")
    saved_once = False

    print(f"метрики по эпохам → {metrics_csv}", flush=True)
    print(f"cdae_ckpt={cdae_ckpt}", flush=True)
    print(f"epochs={epochs}  device={device}", flush=True)

    for epoch in range(epochs):
        if unfreeze_encoder_after is not None and epoch == unfreeze_encoder_after:
            model.unfreeze_encoder()
            optimizer = torch.optim.Adam(
                [
                    {"params": list(model.encoder.parameters()), "lr": encoder_lr},
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if p.requires_grad and not n.startswith("encoder.")
                        ],
                        "lr": lr,
                    },
                ]
            )
            print(f"[epoch {epoch}] encoder разморожен, lr={encoder_lr}", flush=True)

        _set_train_modes(model)
        n_batches = len(train_loader)

        for batch_index, batch in enumerate(train_loader):
            b = _move_batch(batch, device)
            pred = model(b["frame"], b["params"], b["time"])
            loss = criterion(pred, b["target"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if (
                batch_index == 0
                or (batch_index + 1) % 50 == 0
                or (batch_index + 1) == n_batches
            ):
                print(
                    f"  batch {batch_index + 1}/{n_batches}  huber={loss.item():.4f}",
                    flush=True,
                )

        train_m = evaluate(model, train_loader, device, criterion=criterion_sum)
        val_m = evaluate(model, val_loader, device, criterion=criterion_sum)

        is_best = 0
        if val_m["mae"] < best_mae:
            best_mae = val_m["mae"]
            save_regressor(model, out_ckpt, cdae=cdae, extra_config=extra_config)
            saved_once = True
            is_best = 1

        row: dict[str, Any] = {
            "epoch": epoch,
            "train_huber": train_m["huber"],
            "train_mae": train_m["mae"],
            "train_mse": train_m["mse"],
            "train_r2": train_m["r2"],
            "val_huber": val_m["huber"],
            "val_mae": val_m["mae"],
            "val_mse": val_m["mse"],
            "val_r2": val_m["r2"],
            "is_best": is_best,
        }
        _append_metrics_csv(metrics_csv, row)
        print(_format_epoch_line(row), flush=True)

    if not saved_once:
        save_regressor(model, out_ckpt, cdae=cdae, extra_config=extra_config)
        print(f"сохранено (fallback): {out_ckpt}", flush=True)
    else:
        model = load_regressor(out_ckpt, cdae=cdae, map_location=device)
        model.to(device)
        print(
            f"лучший чекпоинт (по val MAE): {out_ckpt}  mae={best_mae:.4f}",
            flush=True,
        )

    if test_loader is not None and len(test_loader) > 0:
        test_m = evaluate(model, test_loader, device, criterion=criterion_sum)
        print(
            f"test  huber={test_m['huber']:.4f}  "
            f"mae={test_m['mae']:.4f}  mse={test_m['mse']:.4f}  r2={test_m['r2']:.4f}",
            flush=True,
        )

    print(f"таблица метрик: {metrics_csv}", flush=True)
    return model


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Обучение регрессора сопротивления поверх CDAE"
    )
    p.add_argument("--cdae-ckpt", type=Path, default=DEFAULT_CDAE_CKPT)
    p.add_argument("--out-ckpt", type=Path, default=DEFAULT_OUT_CKPT)
    p.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    p.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_ROOT)
    p.add_argument(
        "--position-step",
        type=float,
        default=0.02,
        help="Шаг позиции кадра: 0.02 → 2%, 4%, ..., 98%.",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--encoder-lr", type=float, default=1e-5)
    p.add_argument("--unfreeze-after", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not Path(args.cdae_ckpt).exists():
        raise SystemExit(
            f"нет чекпоинта CDAE: {args.cdae_ckpt}. "
            "Сначала train_cdae.py → checkpoints/cdae_weights.pth"
        )
    if not Path(args.metadata).exists():
        raise SystemExit(f"нет metadata: {args.metadata}")
    if not Path(args.video_dir).exists():
        raise SystemExit(f"нет video-dir: {args.video_dir}")
    if not 0.0 < args.position_step < 1.0:
        raise SystemExit("--position-step must be in (0, 1).")

    print("=" * 60, flush=True)
    print("REGRESSION TRAINING", flush=True)
    print("=" * 60, flush=True)
    print(f"Device:       {args.device}", flush=True)
    print(f"Metadata:     {args.metadata}", flush=True)
    print(f"Video dir:    {args.video_dir}", flush=True)
    print(f"CDAE ckpt:    {args.cdae_ckpt}", flush=True)
    print(f"Pos. step:    {args.position_step}", flush=True)
    print(f"Epochs:       {args.epochs}", flush=True)
    print(f"Batch size:   {args.batch_size}", flush=True)
    print("=" * 60, flush=True)
    print("Indexing videos (open each mp4 once)...", flush=True)

    dataset = LIGVideoDataset(
        metadata_path=args.metadata,
        video_root=args.video_dir,
        frame_size=(IMAGE_SIZE, IMAGE_SIZE),
        position_step=args.position_step,
    )
    train_dataset, val_dataset, test_dataset, split_info = split_dataset_by_video(
        dataset,
        seed=args.seed,
        normalize_laser_params=True,
    )

    laser_mean = _to_float_list(split_info.get("laser_mean"))
    laser_std = _to_float_list(split_info.get("laser_std"))
    extra_config = {
        "laser_mean": laser_mean,
        "laser_std": laser_std,
        "position_step": float(args.position_step),
        "split_seed": int(args.seed),
    }

    norm_path = Path(args.metrics_csv).with_name("regression_norm.json")
    norm_path.parent.mkdir(parents=True, exist_ok=True)
    with norm_path.open("w", encoding="utf-8") as f:
        json.dump(extra_config, f, indent=2, ensure_ascii=False)
    print(f"laser mean/std → {norm_path}", flush=True)

    pin_memory = str(args.device).startswith("cuda")
    train_loader = _make_loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        seed=args.seed,
        grouped=True,
    )
    val_loader = _make_loader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        seed=args.seed,
        grouped=False,
    )
    test_loader = _make_loader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        seed=args.seed,
        grouped=False,
    )

    train_regression(
        train_loader,
        val_loader,
        cdae_ckpt=args.cdae_ckpt,
        out_ckpt=args.out_ckpt,
        metrics_csv=args.metrics_csv,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        unfreeze_encoder_after=args.unfreeze_after,
        encoder_lr=args.encoder_lr,
        extra_config=extra_config,
        test_loader=test_loader,
    )


if __name__ == "__main__":
    main()
