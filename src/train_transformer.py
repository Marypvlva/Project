"""
Обучение temporal Transformer-регрессора поверх CDAE.

Альтернатива train_regression.py (один кадр + MLP). 

Ожидаемые ключи батча (датасет коллеги, temporal-режим):
  frames                        — [B, T, 3, 128, 128], float [0, 1]; последний = текущий
  params / process_params       — [B, 3], z-scored
  time                          — [B, 1], t/duration для последнего кадра
  target                        — [B, 1], R в кОм (inf/nan → 86)
  frame_mask (optional)         — [B, T] bool, False = padding в начале видео
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

from cdae_model import load_cdae
from transformer_model import (
    DEFAULT_WINDOW_SIZE,
    TemporalResistanceRegressor,
    build_temporal_regressor_from_cdae,
    load_temporal_regressor,
    save_temporal_regressor,
)

MAX_RESISTANCE_KOHM = 86.0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CDAE_CKPT = PROJECT_ROOT / "checkpoints" / "cdae_weights.pth"
DEFAULT_OUT_CKPT = PROJECT_ROOT / "checkpoints" / "temporal_regressor.pt"
DEFAULT_METRICS_CSV = PROJECT_ROOT / "logs" / "transformer_metrics.csv"

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
    if not isinstance(batch, dict):
        raise TypeError(f"Ожидался dict-батч, получено {type(batch).__name__}")

    if "frames" not in batch:
        raise KeyError(
            "В батче нет 'frames' [B, T, 3, H, W]. "
            "Нужен temporal-датасет (LIGTemporalDataset или window_size=...)."
        )

    frames = batch["frames"].to(device)
    if "params" in batch:
        params = batch["params"].to(device)
    elif "process_params" in batch:
        params = batch["process_params"].to(device)
    else:
        raise KeyError("В батче нет 'params' / 'process_params'")

    time = batch["time"].to(device)
    target = batch["target"].to(device).float()
    target = torch.nan_to_num(
        target,
        nan=MAX_RESISTANCE_KOHM,
        posinf=MAX_RESISTANCE_KOHM,
        neginf=MAX_RESISTANCE_KOHM,
    ).clamp(min=0.0, max=MAX_RESISTANCE_KOHM)

    if target.ndim == 1:
        target = target.unsqueeze(1)
    if time.ndim == 1:
        time = time.unsqueeze(1)

    out: dict[str, torch.Tensor] = {
        "frames": frames,
        "params": params,
        "time": time,
        "target": target,
    }
    if "frame_mask" in batch:
        out["frame_mask"] = batch["frame_mask"].to(device)
    return out


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
    model: TemporalResistanceRegressor,
    loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
) -> dict[str, float]:
    was_training = model.training
    encoder_was_training = model.encoder.training
    model.eval()

    if criterion is None:
        criterion = nn.HuberLoss(reduction="sum")

    abs_err = sq_err = sum_y = sum_y2 = huber_sum = 0.0
    n = 0

    for batch in loader:
        b = _move_batch(batch, device)
        pred = model(
            b["frames"],
            b["params"],
            b["time"],
            frame_mask=b.get("frame_mask"),
        )
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


def _set_train_modes(model: TemporalResistanceRegressor) -> None:
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
    try:
        from dataset import LIGTemporalDataset

        return LIGTemporalDataset
    except ImportError:
        pass
    try:
        from dataset import LIGDataset

        return LIGDataset
    except ImportError:
        pass
    try:
        from dataset import LIGVideoDataset

        return LIGVideoDataset
    except ImportError as exc:
        raise SystemExit(
            "Не найден dataset.LIGTemporalDataset / LIGDataset — нужен код от #2."
        ) from exc


def _build_dataset(
    dataset_cls: type,
    *,
    metadata_path: Path,
    video_dir: Path,
    window_size: int,
    param_mean: Optional[Any] = None,
    param_std: Optional[Any] = None,
) -> Dataset:
    kwargs: dict[str, Any] = {
        "metadata_path": metadata_path,
        "video_dir": video_dir,
    }
    sig = inspect.signature(dataset_cls.__init__)
    params = sig.parameters
    if "window_size" in params:
        kwargs["window_size"] = window_size
    if "temporal" in params:
        kwargs["temporal"] = True
    if param_mean is not None and "param_mean" in params:
        kwargs["param_mean"] = param_mean
    if param_std is not None and "param_std" in params:
        kwargs["param_std"] = param_std
    return dataset_cls(**kwargs)


def train_transformer(
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    cdae_ckpt: str | Path = DEFAULT_CDAE_CKPT,
    out_ckpt: str | Path = DEFAULT_OUT_CKPT,
    metrics_csv: str | Path = DEFAULT_METRICS_CSV,
    window_size: int = DEFAULT_WINDOW_SIZE,
    epochs: int = 50,
    lr: float = 1e-4,
    device: str | torch.device = "cpu",
    unfreeze_encoder_after: Optional[int] = None,
    encoder_lr: float = 1e-5,
) -> TemporalResistanceRegressor:
    if len(train_loader) == 0:
        raise ValueError("train_loader пустой")
    if len(val_loader) == 0:
        raise ValueError("val_loader пустой")

    device = torch.device(device)
    out_ckpt = Path(out_ckpt)
    metrics_csv = Path(metrics_csv)
    _init_metrics_csv(metrics_csv)

    cdae = load_cdae(cdae_ckpt, map_location=device)
    model = build_temporal_regressor_from_cdae(
        cdae,
        freeze_encoder=True,
        use_time=True,
        window_size=window_size,
    )
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

    print(f"метрики → {metrics_csv}")
    print(f"cdae_ckpt={cdae_ckpt}")
    print(f"window_size={window_size}  epochs={epochs}  device={device}")

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
            pred = model(
                b["frames"],
                b["params"],
                b["time"],
                frame_mask=b.get("frame_mask"),
            )
            loss = criterion(pred, b["target"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        train_m = evaluate(model, train_loader, device, criterion=criterion_sum)
        val_m = evaluate(model, val_loader, device, criterion=criterion_sum)

        is_best = 0
        if val_m["mae"] < best_mae:
            best_mae = val_m["mae"]
            save_temporal_regressor(model, out_ckpt, cdae=cdae)
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
        save_temporal_regressor(model, out_ckpt, cdae=cdae)
        print(f"сохранено (fallback): {out_ckpt}")
    else:
        model = load_temporal_regressor(out_ckpt, cdae=cdae, map_location=device)
        model.to(device)
        print(f"лучший чекпоинт (val MAE): {out_ckpt}  mae={best_mae:.4f}")

    print(f"таблица метрик: {metrics_csv}")
    return model


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Temporal Transformer-регрессор поверх CDAE (буфер последних кадров)"
    )
    p.add_argument("--cdae-ckpt", type=Path, default=DEFAULT_CDAE_CKPT)
    p.add_argument("--out-ckpt", type=Path, default=DEFAULT_OUT_CKPT)
    p.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    p.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data" / "metadata_train.csv")
    p.add_argument("--val-metadata", type=Path, default=PROJECT_ROOT / "data" / "metadata_val.csv")
    p.add_argument("--video-dir", type=Path, default=PROJECT_ROOT / "data" / "raw_videos")
    p.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
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
    if args.window_size <= 0:
        raise SystemExit("--window-size must be > 0")

    train_dataset = _build_dataset(
        dataset_cls,
        metadata_path=args.metadata,
        video_dir=args.video_dir,
        window_size=args.window_size,
    )
    val_kwargs: dict[str, Any] = {}
    if hasattr(train_dataset, "param_mean"):
        val_kwargs["param_mean"] = train_dataset.param_mean
    if hasattr(train_dataset, "param_std"):
        val_kwargs["param_std"] = train_dataset.param_std
    val_dataset = _build_dataset(
        dataset_cls,
        metadata_path=args.val_metadata,
        video_dir=args.video_dir,
        window_size=args.window_size,
        **val_kwargs,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    train_transformer(
        train_loader,
        val_loader,
        cdae_ckpt=args.cdae_ckpt,
        out_ckpt=args.out_ckpt,
        metrics_csv=args.metrics_csv,
        window_size=args.window_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        unfreeze_encoder_after=args.unfreeze_after,
        encoder_lr=args.encoder_lr,
    )


if __name__ == "__main__":
    main()
