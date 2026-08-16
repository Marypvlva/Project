"""
CDAE-претрейн + регрессор сопротивления для on-the-fly мониторинга LIG.

------------------------------
Входные кадры
  - RGB, float в [0, 1]
  - форма [B, 3, H, W], по умолчанию H=W=128 (ресайз из 1280x1024 — задача #2)
  - модель однокадровая; временные окна / шаг по кадрам — на стороне датасета

Параметры процесса
  - [мощность, скорость, толщина], форма [B, 3]
  - ОБЯЗАТЕЛЬНО стандартизовать заранее (mean/std по train); сырые единицы ломают MLP

Таргет
  - сопротивление в кОм, форма [B, 1] (без логарифма)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Union
import torch
import torch.nn as nn

DEFAULT_IMG_SIZE = 128
IN_CHANNELS = 3
NUM_PROCESS_PARAMS = 3  # мощность, скорость, толщина

PathLike = Union[str, Path]


def _conv_block(in_ch: int, out_ch: int, kernel_size: int = 5) -> nn.Sequential:
    """Свёртка + BatchNorm + ReLU без изменения H×W."""
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _up_block(in_ch: int, out_ch: int, kernel_size: int = 5) -> nn.Sequential:
    """Увеличение карты признаков ×2 + свёртка."""
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CDAE(nn.Module):
    """
    Свёрточный denoising-автоэнкодер.

    Шум подаётся снаружи (датасет / цикл обучения). Модуль только
    восстанавливает (возможно зашумлённый) кадр в диапазоне [0, 1].

    Геометрия по умолчанию при img_size=128:
      128 -> 64 -> 32 -> 16  (encoder)
      16  -> 32 -> 64 -> 128 (decoder)
    """

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        img_size: int = DEFAULT_IMG_SIZE,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        if img_size % 8 != 0:
            raise ValueError(f"img_size должен делиться на 8, получено {img_size}")

        self.in_channels = in_channels
        self.img_size = img_size
        self.latent_channels = base_channels * 4  # 128 при base=32

        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        self.encoder = nn.Sequential(
            _conv_block(in_channels, c1),
            nn.MaxPool2d(2),
            _conv_block(c1, c2),
            nn.MaxPool2d(2),
            _conv_block(c2, c3),
            nn.MaxPool2d(2),
        )

        self.decoder = nn.Sequential(
            _up_block(c3, c2),
            _up_block(c2, c1),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c1, in_channels, kernel_size=5, padding=2),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Пространственная карта признаков [B, C_lat, H/8, W/8]."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Реконструкция кадра из latent-карты."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Полный проход: encode → decode."""
        return self.decode(self.encode(x))


class ResistanceRegressor(nn.Module):
    """
    Encoder из CDAE (freeze/finetune) + параметры процесса (+ опционально прогресс)
    → сопротивление R [кОм].

    forward(frame, params, time=None) -> [B, 1]
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        latent_channels: int = 128,
        num_params: int = NUM_PROCESS_PARAMS,
        img_emb_dim: int = 128,
        param_emb_dim: int = 32,
        hidden_dim: int = 256,
        use_time: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.use_time = use_time
        self.latent_channels = latent_channels

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.img_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_channels, img_emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        param_in = num_params + (1 if use_time else 0)
        self.param_proj = nn.Sequential(
            nn.Linear(param_in, param_emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.Linear(img_emb_dim + param_emb_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Эмбеддинг одного кадра [B, img_emb_dim]."""
        feats = self.encoder(frame)
        return self.img_proj(self.pool(feats))

    def forward(
        self,
        frame: torch.Tensor,
        params: torch.Tensor,
        time: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Аргументы:
            frame:  [B, 3, H, W] в [0, 1]
            params: [B, 3] стандартизованные параметры процесса
            time:   [B, 1] прогресс процесса в [0, 1] (elapsed / duration).
                    Обязателен при use_time=True; иначе игнорируется.
        """
        img_emb = self.encode_frame(frame)

        if self.use_time:
            if time is None:
                raise ValueError("time [B, 1] обязателен при use_time=True")
            cond = torch.cat([params, time], dim=1)
        else:
            cond = params

        param_emb = self.param_proj(cond)
        return self.head(torch.cat([img_emb, param_emb], dim=1))

    def freeze_encoder(self) -> None:
        """Заморозить веса encoder (обучение только головы)."""
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Разморозить encoder для дообучения."""
        for p in self.encoder.parameters():
            p.requires_grad = True

    def encoder_trainable(self) -> bool:
        """True, если у encoder есть параметры с requires_grad=True."""
        return any(p.requires_grad for p in self.encoder.parameters())


def build_regressor_from_cdae(
    cdae: CDAE,
    *,
    freeze_encoder: bool = True,
    use_time: bool = True,
    **regressor_kwargs,
) -> ResistanceRegressor:
    """Повесить регрессионную голову на (предобученный) encoder CDAE."""
    regressor_kwargs.setdefault("latent_channels", cdae.latent_channels)
    model = ResistanceRegressor(
        encoder=cdae.encoder,
        use_time=use_time,
        **regressor_kwargs,
    )
    if freeze_encoder:
        model.freeze_encoder()
    return model


def load_cdae(path: PathLike, map_location: str | torch.device = "cpu") -> CDAE:
    """Загрузить CDAE: save_cdae / cdae_weights.pth / training checkpoint."""
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        cfg = ckpt.get("config", {})
        model = CDAE(
            in_channels=cfg.get("in_channels", IN_CHANNELS),
            img_size=cfg.get("img_size", DEFAULT_IMG_SIZE),
            base_channels=cfg.get("base_channels", 32),
        )
        model.load_state_dict(ckpt["state_dict"])
        return model
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model = CDAE()
        model.load_state_dict(ckpt["model_state_dict"])
        return model
    model = CDAE()
    model.load_state_dict(ckpt)
    return model


def save_cdae(model: CDAE, path: PathLike) -> None:
    """Сохранить CDAE (state_dict + config)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "in_channels": model.in_channels,
                "img_size": model.img_size,
                "base_channels": model.latent_channels // 4,
            },
        },
        path,
    )


def load_regressor(
    path: PathLike,
    cdae: Optional[CDAE] = None,
    map_location: str | torch.device = "cpu",
) -> ResistanceRegressor:
    """
    Загрузить чекпоинт регрессора.

    Предпочтительный формат: веса encoder уже внутри state_dict регрессора.
    Если сохранили только голову — передай загруженный `cdae`, его `.encoder` будет использован.
    """
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        cfg = ckpt.get("config", {})
        if cdae is None:
            cdae = CDAE(
                in_channels=cfg.get("in_channels", IN_CHANNELS),
                img_size=cfg.get("img_size", DEFAULT_IMG_SIZE),
                base_channels=cfg.get("base_channels", 32),
            )
        model = build_regressor_from_cdae(
            cdae,
            freeze_encoder=cfg.get("freeze_encoder", True),
            use_time=cfg.get("use_time", True),
            latent_channels=cfg.get("latent_channels", cdae.latent_channels),
            img_emb_dim=cfg.get("img_emb_dim", 128),
            param_emb_dim=cfg.get("param_emb_dim", 32),
            hidden_dim=cfg.get("hidden_dim", 256),
            dropout=cfg.get("dropout", 0.2),
        )
        model.load_state_dict(ckpt["state_dict"])
        return model

    if cdae is None:
        raise ValueError("Для загрузки сырого state_dict нужен уже собранный `cdae`.")
    model = build_regressor_from_cdae(cdae)
    model.load_state_dict(ckpt)
    return model


def save_regressor(model: ResistanceRegressor, path: PathLike, cdae: Optional[CDAE] = None) -> None:
    """Сохранить регрессор (state_dict + config)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base_channels = (
        cdae.latent_channels // 4 if cdae is not None else model.latent_channels // 4
    )
    cfg = {
        "use_time": model.use_time,
        "latent_channels": model.latent_channels,
        "freeze_encoder": not model.encoder_trainable(),
        "in_channels": cdae.in_channels if cdae is not None else IN_CHANNELS,
        "img_size": cdae.img_size if cdae is not None else DEFAULT_IMG_SIZE,
        "base_channels": base_channels,
    }
    # Размеры головы вытаскиваем из модулей
    img_linear = model.img_proj[1]
    param_linear = model.param_proj[0]
    head_linear = model.head[0]
    cfg.update(
        {
            "img_emb_dim": img_linear.out_features,
            "param_emb_dim": param_linear.out_features,
            "hidden_dim": head_linear.out_features,
            "dropout": float(model.img_proj[3].p),
        }
    )
    torch.save({"state_dict": model.state_dict(), "config": cfg}, path)
