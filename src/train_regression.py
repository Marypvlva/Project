"""
Обучение регрессора сопротивления поверх предобученного CDAE.

Ключи батча от LIGVideoDataset:
  frame                         — [B, 3, 128, 128], float [0, 1]
  laser_params / params         — [B, 3] power, speed, distance
  position / time               — [B, 1] относительная позиция в видео [0, 1]
  target                        — [B, 1] сопротивление в кОм

Метрики: logs/regression_metrics.csv + консоль. Best по val MAE.
"""

from __future__ import annotations

import argparse
import csv
import inspect
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from cdae_model import (
    ResistanceRegressor,
    build_regressor_from_cdae,
    load_cdae,
    load_regressor,
    save_regressor,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CDAE_CKPT = PROJECT_ROOT / "checkpoints" / "cdae_weights.pth"
DEFAULT_OUT_CKPT = PROJECT_ROOT / "checkpoints" / "regressor.pt"
DEFAULT_METRICS_CSV = PROJECT_ROOT / "logs" / "regression_metrics.csv"

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


def _load_dataset_class():
    """LIGVideoDataset коллеги; запасной алиас LIGDataset."""
    try:
        from dataset import LIGVideoDataset

        return LIGVideoDataset
    except ImportError:
        pass
    try:
        from dataset import LIGDataset

        return LIGDataset
    except ImportError as exc:
        raise SystemExit(
            "Не найден dataset.LIGVideoDataset / LIGDataset."
        ) from exc


def _make_dataset(
    dataset_cls: type,
    metadata_path: Path,
    video_dir: Path,
) -> Dataset:
    kwargs: dict[str, Any] = {"metadata_path": metadata_path}
    params = inspect.signature(dataset_cls.__init__).parameters
    if "video_root" in params:
        kwargs["video_root"] = video_dir
    elif "video_dir" in params:
        kwargs["video_dir"] = video_dir
    else:
        raise TypeError(
            f"{dataset_cls.__name__} не принимает video_root / video_dir"
        )
    return dataset_cls(**kwargs)


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

    print(f"метрики по эпохам → {metrics_csv}")
    print(f"cdae_ckpt={cdae_ckpt}")
    print(f"epochs={epochs}  device={device}")

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
            print(f"[epoch {epoch}] encoder разморожен, lr={encoder_lr}")

        _set_train_modes(model)

        for batch in train_loader:
            b = _move_batch(batch, device)
            pred = model(b["frame"], b["params"], b["time"])
            loss = criterion(pred, b["target"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        train_m = evaluate(model, train_loader, device, criterion=criterion_sum)
        val_m = evaluate(model, val_loader, device, criterion=criterion_sum)

        is_best = 0
        if val_m["mae"] < best_mae:
            best_mae = val_m["mae"]
            save_regressor(model, out_ckpt, cdae=cdae)
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
        print(_format_epoch_line(row))

    if not saved_once:
        save_regressor(model, out_ckpt, cdae=cdae)
        print(f"сохранено (fallback): {out_ckpt}")
    else:
        # Вернуть best-веса
        model = load_regressor(out_ckpt, cdae=cdae, map_location=device)
        model.to(device)
        print(f"лучший чекпоинт (по val MAE): {out_ckpt}  mae={best_mae:.4f}")

    print(f"таблица метрик: {metrics_csv}")
    return model


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Обучение регрессора сопротивления поверх CDAE")
    p.add_argument("--cdae-ckpt", type=Path, default=DEFAULT_CDAE_CKPT)
    p.add_argument("--out-ckpt", type=Path, default=DEFAULT_OUT_CKPT)
    p.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    p.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data2" / "table.xlsx")
    p.add_argument("--val-metadata", type=Path, default=PROJECT_ROOT / "data2" / "table_val.xlsx")
    p.add_argument("--video-dir", type=Path, default=PROJECT_ROOT / "data2")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--encoder-lr", type=float, default=1e-5)
    p.add_argument("--unfreeze-after", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    dataset_cls = _load_dataset_class()

    if not Path(args.cdae_ckpt).exists():
        raise SystemExit(
            f"нет чекпоинта CDAE: {args.cdae_ckpt}. "
            "Сначала train_cdae.py → checkpoints/cdae_weights.pth"
        )
    if not Path(args.metadata).exists():
        raise SystemExit(f"нет train metadata: {args.metadata}")
    if not Path(args.val_metadata).exists():
        raise SystemExit(f"нет val metadata: {args.val_metadata}")

    train_dataset: Dataset = _make_dataset(
        dataset_cls,
        metadata_path=args.metadata,
        video_dir=args.video_dir,
    )
    val_dataset: Dataset = _make_dataset(
        dataset_cls,
        metadata_path=args.val_metadata,
        video_dir=args.video_dir,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
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
    )

if __name__ == "__main__":
    main()
