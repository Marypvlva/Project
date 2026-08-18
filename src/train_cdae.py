"""
Обучение CDAE.

Кадры — LIGVideoDataset (on-the-fly из MP4).
Здесь только: batch["frame"] [B, 3, 128, 128] в [0, 1] → шум → CDAE → MSE.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from cdae_model import CDAE
from dataset import LIGVideoDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
IMAGE_SIZE = 128
DEFAULT_VIDEO_ROOT = Path("/home/jupyter/filestore/dataset")
DEFAULT_METADATA = DEFAULT_VIDEO_ROOT / "metadata.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Convolutional Denoising Autoencoder (CDAE)."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="CSV metadata. По умолчанию /home/jupyter/filestore/dataset/metadata.csv",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=DEFAULT_VIDEO_ROOT,
        help="Корень с Sample 1/mp4, Sample 2/mp4, ... "
        "По умолчанию /home/jupyter/filestore/dataset",
    )
    parser.add_argument(
        "--position-step",
        type=float,
        default=0.02,
        help="Шаг относительной позиции кадра: 0.02 → 2%, 4%, ..., 98%.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of frames in one training batch.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam optimizer learning rate.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.10,
        help="Standard deviation of Gaussian noise added to clean frames.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory where model weights will be saved.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be greater than 0.")
    if args.noise_std < 0:
        raise ValueError("--noise-std cannot be negative.")
    if not 0.0 < args.position_step < 1.0:
        raise ValueError("--position-step must be in (0, 1).")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if not args.metadata.exists():
        raise FileNotFoundError(
            f"Metadata file does not exist: {args.metadata}"
        )
    if not args.metadata.is_file():
        raise ValueError(
            f"Metadata path is not a file: {args.metadata}"
        )
    if not args.video_dir.exists():
        raise FileNotFoundError(
            f"Video directory does not exist: {args.video_dir}"
        )
    if not args.video_dir.is_dir():
        raise ValueError(
            f"Video path is not a directory: {args.video_dir}"
        )


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataset(
    metadata_path: Path,
    video_dir: Path,
    position_step: float = 0.02,
) -> LIGVideoDataset:
    return LIGVideoDataset(
        metadata_path=metadata_path,
        video_root=video_dir,
        frame_size=(IMAGE_SIZE, IMAGE_SIZE),
        position_step=position_step,
    )


class VideoGroupedSampler(Sampler[int]):
    """
    Кадры одного mp4 идут подряд (по position), порядок видео перемешивается.

    Так VideoCapture не прыгает по 246 файлам и кэш каптуры работает.
    """

    def __init__(self, dataset: LIGVideoDataset, *, seed: int = 42) -> None:
        groups: dict[str, list[int]] = {}
        for idx, sample in enumerate(dataset.samples):
            key = str(sample["video_path"])
            groups.setdefault(key, []).append(idx)
        for idxs in groups.values():
            idxs.sort(key=lambda i: float(dataset.samples[i]["position"]))
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


def extract_frames(batch) -> torch.Tensor:
    if torch.is_tensor(batch):
        frames = batch

    elif isinstance(batch, (tuple, list)):
        if len(batch) == 0:
            raise ValueError("Received an empty batch.")
        frames = batch[0]

    elif isinstance(batch, dict):
        if "frame" not in batch:
            raise KeyError(
                "Dataset returned a dictionary, "
                "but it does not contain the key 'frame'."
            )
        frames = batch["frame"]

    else:
        raise TypeError(
            "Unsupported batch type: "
            f"{type(batch).__name__}. "
            "Expected Tensor, tuple/list, or dict."
        )

    if not torch.is_tensor(frames):
        raise TypeError(
            "Frames returned by Dataset must be torch.Tensor, "
            f"but received {type(frames).__name__}."
        )
    return frames


def validate_frames(frames: torch.Tensor) -> None:
    if frames.ndim != 4:
        raise ValueError(
            "Frames must have 4 dimensions "
            "[batch, channels, height, width], "
            f"but received shape {tuple(frames.shape)}."
        )

    if frames.shape[1] != 3:
        raise ValueError(
            "Expected 3 color channels (RGB), "
            f"but received shape {tuple(frames.shape)}."
        )

    if frames.shape[2:] != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"Expected frame size {IMAGE_SIZE}x{IMAGE_SIZE}, "
            f"but received shape {tuple(frames.shape)}."
        )

    frames_min = frames.min().item()
    frames_max = frames.max().item()
    tolerance = 1e-6

    if frames_min < -tolerance or frames_max > 1.0 + tolerance:
        raise ValueError(
            "Frames must be normalized to the range [0, 1]. "
            f"Received min={frames_min:.6f}, "
            f"max={frames_max:.6f}."
        )


def add_gaussian_noise(
    clean_frames: torch.Tensor,
    noise_std: float,
) -> torch.Tensor:
    noise = torch.randn_like(clean_frames) * noise_std
    noisy_frames = clean_frames + noise
    noisy_frames = torch.clamp(
        noisy_frames,
        min=0.0,
        max=1.0,
    )
    return noisy_frames


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    noise_std: float,
) -> tuple[float, float, float]:
    model.train()
    sum_squared_error = 0.0
    sum_absolute_error = 0.0

    # для расчёта R² нам нужны статистики target.
    target_sum = 0.0
    target_squared_sum = 0.0

    total_elements = 0
    n_batches = len(dataloader)
    for batch_index, batch in enumerate(dataloader):
        clean_frames = extract_frames(batch)
        clean_frames = clean_frames.float()
        if batch_index == 0:
            validate_frames(clean_frames)
        clean_frames = clean_frames.to(device)
        noisy_frames = add_gaussian_noise(
            clean_frames=clean_frames,
            noise_std=noise_std,
        )
        optimizer.zero_grad(set_to_none=True)
        reconstructed_frames = model(noisy_frames)
        if reconstructed_frames.shape != clean_frames.shape:
            raise ValueError(
                "CDAE output shape does not match target shape. "
                f"Model output: {tuple(reconstructed_frames.shape)}, "
                f"target: {tuple(clean_frames.shape)}."
            )
        loss = loss_fn(
            reconstructed_frames,
            clean_frames,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss detected: {loss.item()}"
            )
        loss.backward()
        optimizer.step()

        if batch_index == 0 or (batch_index + 1) % 50 == 0 or (batch_index + 1) == n_batches:
            print(
                f"  batch {batch_index + 1}/{n_batches}  loss={loss.item():.6f}",
                flush=True,
            )


        # Метрики
        # detach() нужен, чтобы расчёт метрик не участвовал в графе вычисления градиентов
        with torch.no_grad():
            predictions = reconstructed_frames.detach()
            targets = clean_frames.detach()
            errors = predictions - targets
            sum_squared_error += (
                errors.pow(2).sum().item()
            )
            sum_absolute_error += (
                errors.abs().sum().item()
            )
            target_sum += targets.sum().item()
            target_squared_sum += (
                targets.pow(2).sum().item()
            )
            total_elements += targets.numel()

    if total_elements == 0:
        raise RuntimeError(
            "DataLoader produced zero elements. "
            "Check Dataset and input data."
        )


    mse = sum_squared_error / total_elements
    mae = sum_absolute_error / total_elements

    total_sum_of_squares = (
        target_squared_sum - (target_sum ** 2) / total_elements
    )
    if total_sum_of_squares <= 1e-12:
        r2 = float("nan")
    else:
        r2 = (
            1.0
            - sum_squared_error / total_sum_of_squares
        )
    return mse, mae, r2


def save_model_weights(
    model: nn.Module,
    checkpoint_dir: Path,
) -> Path:
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    weights_path = checkpoint_dir / "cdae_weights.pth"
    torch.save(
        model.state_dict(),
        weights_path,
    )
    return weights_path


def save_training_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: Path,
) -> Path:
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    checkpoint_path = checkpoint_dir / "cdae_training_checkpoint.pth"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        checkpoint_path,
    )
    return checkpoint_path


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_random_seed(args.seed)
    device = get_device()
    print("=" * 60, flush=True)
    print("CDAE TRAINING", flush=True)
    print("=" * 60, flush=True)
    print(f"Device:       {device}", flush=True)
    print(f"Metadata:     {args.metadata}", flush=True)
    print(f"Video dir:    {args.video_dir}", flush=True)
    print(f"Pos. step:    {args.position_step}", flush=True)
    print(f"Epochs:       {args.epochs}", flush=True)
    print(f"Batch size:   {args.batch_size}", flush=True)
    print(f"Learning rate:{args.learning_rate}", flush=True)
    print(f"Noise std:    {args.noise_std}", flush=True)
    print("=" * 60, flush=True)
    print("Indexing videos (open each mp4 once)...", flush=True)

    dataset = build_dataset(
        metadata_path=args.metadata,
        video_dir=args.video_dir,
        position_step=args.position_step,
    )
    if len(dataset) == 0:
        raise RuntimeError("LIGVideoDataset contains zero samples.")
    n_videos = getattr(dataset, "video_count", None)
    n_skipped = len(getattr(dataset, "skipped_videos", []) or [])
    extra = ""
    if n_videos is not None:
        extra = f"  videos={n_videos}  skipped={n_skipped}"
    print(f"Dataset samples: {len(dataset)}{extra}", flush=True)

    sampler = VideoGroupedSampler(dataset, seed=args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    model = CDAE(img_size=IMAGE_SIZE)
    model = model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    for epoch in range(1, args.epochs + 1):
        mse, mae, r2 = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            noise_std=args.noise_std,
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"| MSE: {mse:.6f} "
            f"| MAE: {mae:.6f} "
            f"| R²: {r2:.6f}",
            flush=True,
        )

        save_model_weights(
            model=model,
            checkpoint_dir=args.checkpoint_dir,
        )
        save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=mse,
            checkpoint_dir=args.checkpoint_dir,
        )
    print("=" * 60)
    print("Training completed.")
    final_weights_path = (
        args.checkpoint_dir / "cdae_weights.pth"
    )
    print(f"CDAE weights saved to: {final_weights_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()