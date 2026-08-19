"""
Обучение temporal Transformer-регрессора поверх CDAE.

Данные — LIGTemporalDataset + split_dataset_by_video:
  окно последних T кадров, один CSV, split по видео, z-score params по train.

Ключи батча:
  frames                        — [B, T, 3, 128, 128]; последний = текущий
  frame_mask                    — [B, T] bool; False = padding слева
  laser_params                  — [B, 3] z-scored
  position                      — [B, 1] прогресс последнего кадра
  target                        — [B, 1] R в кОм
  target_censored               — [B, 1] bool; True если R >= 86 («-» в Excel)

Лосс: one-sided censored MSE. Best по val censored MSE.
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
from torch.utils.data import DataLoader

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from censored_loss import (
    MAX_RESISTANCE_KOHM,
    OneSidedCensoredMSELoss,
    one_sided_censored_abs_error,
    one_sided_censored_squared_error,
)
from cdae_model import load_cdae
from dataset import LIGTemporalDataset, split_dataset_by_video
from train_regression import VideoGroupedSubsetSampler, _to_float_list
from transformer_model import (
    DEFAULT_WINDOW_SIZE,
    TemporalResistanceRegressor,
    build_temporal_regressor_from_cdae,
    load_temporal_regressor,
    save_temporal_regressor,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_ROOT = Path("/home/jupyter/filestore/dataset")
DEFAULT_METADATA = DEFAULT_VIDEO_ROOT / "metadata.csv"
DEFAULT_CDAE_CKPT = PROJECT_ROOT / "checkpoints" / "cdae_weights.pth"
DEFAULT_OUT_CKPT = PROJECT_ROOT / "checkpoints" / "temporal_regressor.pt"
DEFAULT_METRICS_CSV = PROJECT_ROOT / "logs" / "transformer_metrics.csv"
IMAGE_SIZE = 128

METRICS_FIELDS = [
    "epoch",
    "train_censored_mse",
    "train_mae",
    "train_mse_unc",
    "train_r2_unc",
    "val_censored_mse",
    "val_mae",
    "val_mse_unc",
    "val_r2_unc",
    "is_best",
]


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    if not isinstance(batch, dict):
        raise TypeError(f"Ожидался dict-батч, получено {type(batch).__name__}")

    if "frames" not in batch:
        raise KeyError(
            "В батче нет 'frames' [B, T, 3, H, W]. "
            "Нужен LIGTemporalDataset."
        )

    frames = batch["frames"].to(device)
    if "laser_params" in batch:
        params = batch["laser_params"].to(device)
    elif "params" in batch:
        params = batch["params"].to(device)
    elif "process_params" in batch:
        params = batch["process_params"].to(device)
    else:
        raise KeyError("В батче нет 'laser_params' / 'params' / 'process_params'")

    if "position" in batch:
        position = batch["position"].to(device)
    elif "time" in batch:
        position = batch["time"].to(device)
    else:
        raise KeyError("В батче нет 'position'")

    target = batch["target"].to(device)
    if "target_censored" in batch:
        target_censored = batch["target_censored"].to(device)
    else:
        target_censored = torch.zeros_like(target, dtype=torch.bool, device=device)

    if target.ndim == 1:
        target = target.unsqueeze(1)
    if target_censored.ndim == 1:
        target_censored = target_censored.unsqueeze(1)
    if position.ndim == 1:
        position = position.unsqueeze(1)

    out: dict[str, torch.Tensor] = {
        "frames": frames,
        "params": params,
        "position": position,
        "target": target,
        "target_censored": target_censored,
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
    censored_mse_sum: float,
    n_all: int,
) -> dict[str, float]:
    if n_all <= 0:
        raise ValueError("пустой loader: нет сэмплов для метрик")
    mse_unc = sq_err / n if n > 0 else float("nan")
    ss_tot = max(0.0, sum_y2 - (sum_y ** 2) / n) if n > 0 else 0.0
    r2_unc = float("nan") if n <= 0 else (
        0.0 if ss_tot < 1e-12 else 1.0 - (sq_err / ss_tot)
    )
    return {
        "censored_mse": censored_mse_sum / n_all,
        "mae": abs_err / n_all,
        "mse_unc": mse_unc,
        "r2_unc": r2_unc,
    }


@torch.no_grad()
def evaluate(
    model: TemporalResistanceRegressor,
    loader: DataLoader,
    device: torch.device,
    *,
    censor_threshold: float = MAX_RESISTANCE_KOHM,
) -> dict[str, float]:
    """One-sided censored MSE + MAE; MSE/R² — только на нецензурированных."""
    was_training = model.training
    encoder_was_training = model.encoder.training
    model.eval()

    abs_err = sq_err = sum_y = sum_y2 = censored_mse_sum = 0.0
    n_uncensored = 0
    n_all = 0

    for batch in loader:
        b = _move_batch(batch, device)
        pred = model(
            b["frames"],
            b["params"],
            b["position"],
            frame_mask=b.get("frame_mask"),
        )
        target = b["target"]
        censored = b["target_censored"]

        sq = one_sided_censored_squared_error(
            pred, target, censored, censor_threshold=censor_threshold
        )
        ae = one_sided_censored_abs_error(
            pred, target, censored, censor_threshold=censor_threshold
        )
        censored_mse_sum += sq.sum().item()
        abs_err += ae.sum().item()
        n_all += target.numel()

        uncensored = ~censored.bool()
        if uncensored.any():
            diff = pred[uncensored] - target[uncensored]
            sq_err += (diff ** 2).sum().item()
            sum_y += target[uncensored].sum().item()
            sum_y2 += (target[uncensored] ** 2).sum().item()
            n_uncensored += int(uncensored.sum().item())

    if was_training:
        model.train()
        model.encoder.train(encoder_was_training)

    return _metrics_from_sums(
        abs_err, sq_err, sum_y, sum_y2, n_uncensored, censored_mse_sum, n_all
    )


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
        f"train_censored_mse={row['train_censored_mse']:.4f}",
        f"train_mae={row['train_mae']:.4f}",
        f"train_mse_unc={row['train_mse_unc']:.4f}",
        f"train_r2_unc={row['train_r2_unc']:.4f}",
        f"val_censored_mse={row['val_censored_mse']:.4f}",
        f"val_mae={row['val_mae']:.4f}",
        f"val_mse_unc={row['val_mse_unc']:.4f}",
        f"val_r2_unc={row['val_r2_unc']:.4f}",
    ]
    if row.get("is_best"):
        parts.append("[best saved]")
    return "  ".join(parts)


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


def train_transformer(
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    cdae_ckpt: str | Path = DEFAULT_CDAE_CKPT,
    out_ckpt: str | Path = DEFAULT_OUT_CKPT,
    metrics_csv: str | Path = DEFAULT_METRICS_CSV,
    window_size: int = DEFAULT_WINDOW_SIZE,
    epochs: int = 20,
    lr: float = 1e-4,
    device: str | torch.device = "cpu",
    unfreeze_encoder_after: Optional[int] = None,
    encoder_lr: float = 1e-5,
    extra_config: Optional[dict[str, Any]] = None,
    test_loader: Optional[DataLoader] = None,
) -> TemporalResistanceRegressor:
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
    model = build_temporal_regressor_from_cdae(
        cdae,
        freeze_encoder=True,
        use_position=True,
        window_size=window_size,
    )
    model.to(device)
    _set_train_modes(model)

    criterion = OneSidedCensoredMSELoss(censor_threshold=MAX_RESISTANCE_KOHM)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )

    best_censored_mse = float("inf")
    saved_once = False

    print(f"метрики → {metrics_csv}", flush=True)
    print(f"cdae_ckpt={cdae_ckpt}", flush=True)
    print(f"window_size={window_size}", flush=True)
    print(f"censor_threshold={MAX_RESISTANCE_KOHM} kOhm", flush=True)
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
            pred = model(
                b["frames"],
                b["params"],
                b["position"],
                frame_mask=b.get("frame_mask"),
            )
            loss = criterion(pred, b["target"], b["target_censored"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if (
                batch_index == 0
                or (batch_index + 1) % 25 == 0
                or (batch_index + 1) == n_batches
            ):
                print(
                    f"  batch {batch_index + 1}/{n_batches}  "
                    f"censored_mse={loss.item():.4f}",
                    flush=True,
                )

        train_m = evaluate(model, train_loader, device)
        val_m = evaluate(model, val_loader, device)

        is_best = 0
        if val_m["censored_mse"] < best_censored_mse:
            best_censored_mse = val_m["censored_mse"]
            save_temporal_regressor(model, out_ckpt, cdae=cdae, extra_config=extra_config)
            saved_once = True
            is_best = 1

        row: dict[str, Any] = {
            "epoch": epoch,
            "train_censored_mse": train_m["censored_mse"],
            "train_mae": train_m["mae"],
            "train_mse_unc": train_m["mse_unc"],
            "train_r2_unc": train_m["r2_unc"],
            "val_censored_mse": val_m["censored_mse"],
            "val_mae": val_m["mae"],
            "val_mse_unc": val_m["mse_unc"],
            "val_r2_unc": val_m["r2_unc"],
            "is_best": is_best,
        }
        _append_metrics_csv(metrics_csv, row)
        print(_format_epoch_line(row), flush=True)

    if not saved_once:
        save_temporal_regressor(model, out_ckpt, cdae=cdae, extra_config=extra_config)
        print(f"сохранено (fallback): {out_ckpt}", flush=True)
    else:
        model = load_temporal_regressor(out_ckpt, cdae=cdae, map_location=device)
        model.to(device)
        print(
            f"лучший чекпоинт (val censored MSE): {out_ckpt}  "
            f"censored_mse={best_censored_mse:.4f}",
            flush=True,
        )

    if test_loader is not None and len(test_loader) > 0:
        test_m = evaluate(model, test_loader, device)
        print(
            f"test  censored_mse={test_m['censored_mse']:.4f}  "
            f"mae={test_m['mae']:.4f}  "
            f"mse_unc={test_m['mse_unc']:.4f}  r2_unc={test_m['r2_unc']:.4f}",
            flush=True,
        )

    print(f"таблица метрик: {metrics_csv}", flush=True)
    return model


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Temporal Transformer-регрессор поверх CDAE (окно последних кадров)"
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
    p.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
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
    if args.window_size <= 0:
        raise SystemExit("--window-size must be > 0")
    if not 0.0 < args.position_step < 1.0:
        raise SystemExit("--position-step must be in (0, 1).")

    print("=" * 60, flush=True)
    print("TRANSFORMER TRAINING", flush=True)
    print("=" * 60, flush=True)
    print(f"Device:       {args.device}", flush=True)
    print(f"Metadata:     {args.metadata}", flush=True)
    print(f"Video dir:    {args.video_dir}", flush=True)
    print(f"CDAE ckpt:    {args.cdae_ckpt}", flush=True)
    print(f"Window size:  {args.window_size}", flush=True)
    print(f"Pos. step:    {args.position_step}", flush=True)
    print(f"Epochs:       {args.epochs}", flush=True)
    print(f"Batch size:   {args.batch_size}", flush=True)
    print("=" * 60, flush=True)
    print("Indexing videos (open each mp4 once)...", flush=True)

    dataset = LIGTemporalDataset(
        metadata_path=args.metadata,
        video_root=args.video_dir,
        frame_size=(IMAGE_SIZE, IMAGE_SIZE),
        position_step=args.position_step,
        window_size=args.window_size,
    )
    if "target_censored" not in dataset.metadata.columns:
        print(
            "[WARN] В CSV нет колонки target_censored — все таргеты считаются "
            "нецензурированными. Пересоберите metadata через parser.py.",
            flush=True,
        )
    else:
        n_cens = int(dataset.metadata["target_censored"].fillna(0).astype(int).sum())
        print(
            f"Censored videos in CSV: {n_cens} / {len(dataset.metadata)}",
            flush=True,
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
        "censor_threshold_kOhm": float(MAX_RESISTANCE_KOHM),
        "window_size": int(args.window_size),
    }

    norm_path = Path(args.metrics_csv).with_name("transformer_norm.json")
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
        extra_config=extra_config,
        test_loader=test_loader,
    )


if __name__ == "__main__":
    main()
