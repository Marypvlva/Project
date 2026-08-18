"""
Temporal-регрессор: буфер последних T кадров → Transformer → R [кОм].

Encoder берётся из предобученного CDAE (freeze/finetune).
Отдельная альтернатива однокадровому ResistanceRegressor — не замена.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from cdae_model import (
    CDAE,
    DEFAULT_IMG_SIZE,
    IN_CHANNELS,
    NUM_PROCESS_PARAMS,
    PathLike,
)

DEFAULT_WINDOW_SIZE = 20


class TemporalResistanceRegressor(nn.Module):
    """
    CDAE encoder на каждый кадр окна → Transformer → голова → R.

    forward(frames, params, time=None, frame_mask=None) -> [B, 1]

    frames:      [B, T, 3, H, W]  последние T кадров, последний = текущий
    params:      [B, 3]             z-scored мощность, скорость, толщина
    time:        [B, 1]             t / duration (момент последнего кадра)
    frame_mask:  [B, T] bool, True = реальный кадр; False = padding в начале видео
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        window_size: int = DEFAULT_WINDOW_SIZE,
        latent_channels: int = 128,
        frame_emb_dim: int = 128,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        num_params: int = NUM_PROCESS_PARAMS,
        param_emb_dim: int = 32,
        hidden_dim: int = 256,
        use_time: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.encoder = encoder
        self.window_size = window_size
        self.use_time = use_time
        self.latent_channels = latent_channels
        self.d_model = d_model

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.frame_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_channels, frame_emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.token_proj = (
            nn.Linear(frame_emb_dim, d_model)
            if frame_emb_dim != d_model
            else nn.Identity()
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, window_size, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        param_in = num_params + (1 if use_time else 0)
        self.param_proj = nn.Sequential(
            nn.Linear(param_in, param_emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.Linear(d_model + param_emb_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """[B, T, 3, H, W] -> [B, T, frame_emb_dim]."""
        b, t, c, h, w = frames.shape
        flat = frames.reshape(b * t, c, h, w)
        feats = self.encoder(flat)
        emb = self.frame_proj(self.pool(feats))
        return emb.view(b, t, -1)

    def forward(
        self,
        frames: torch.Tensor,
        params: torch.Tensor,
        time: Optional[torch.Tensor] = None,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if frames.ndim != 5:
            raise ValueError(f"frames must be [B, T, 3, H, W], got {tuple(frames.shape)}")

        b, t, _, _, _ = frames.shape
        if t != self.window_size:
            raise ValueError(
                f"expected T={self.window_size}, got T={t}. "
                "Padding/truncation — на стороне датасета."
            )

        tokens = self.token_proj(self.encode_frames(frames))
        tokens = tokens + self.pos_embed[:, :t, :]

        if frame_mask is not None:
            if frame_mask.shape != (b, t):
                raise ValueError(f"frame_mask must be [B, T], got {tuple(frame_mask.shape)}")
            key_padding = ~frame_mask.bool()
        else:
            key_padding = None

        temporal = self.transformer(tokens, src_key_padding_mask=key_padding)
        if frame_mask is not None:
            valid = frame_mask.float().unsqueeze(-1)
            denom = valid.sum(dim=1).clamp_min(1.0)
            pooled = (temporal * valid).sum(dim=1) / denom
        else:
            pooled = temporal[:, -1, :]

        if self.use_time:
            if time is None:
                raise ValueError("time [B, 1] обязателен при use_time=True")
            cond = torch.cat([params, time], dim=1)
        else:
            cond = params

        param_emb = self.param_proj(cond)
        return self.head(torch.cat([pooled, param_emb], dim=1))

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True

    def encoder_trainable(self) -> bool:
        return any(p.requires_grad for p in self.encoder.parameters())


def build_temporal_regressor_from_cdae(
    cdae: CDAE,
    *,
    freeze_encoder: bool = True,
    window_size: int = DEFAULT_WINDOW_SIZE,
    use_time: bool = True,
    **kwargs,
) -> TemporalResistanceRegressor:
    kwargs.setdefault("latent_channels", cdae.latent_channels)
    model = TemporalResistanceRegressor(
        encoder=cdae.encoder,
        window_size=window_size,
        use_time=use_time,
        **kwargs,
    )
    if freeze_encoder:
        model.freeze_encoder()
    return model


def save_temporal_regressor(
    model: TemporalResistanceRegressor,
    path: PathLike,
    cdae: Optional[CDAE] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base_channels = (
        cdae.latent_channels // 4 if cdae is not None else model.latent_channels // 4
    )
    cfg = {
        "model_type": "temporal_transformer",
        "window_size": model.window_size,
        "use_time": model.use_time,
        "latent_channels": model.latent_channels,
        "d_model": model.d_model,
        "freeze_encoder": not model.encoder_trainable(),
        "in_channels": cdae.in_channels if cdae is not None else IN_CHANNELS,
        "img_size": cdae.img_size if cdae is not None else DEFAULT_IMG_SIZE,
        "base_channels": base_channels,
        "frame_emb_dim": model.frame_proj[1].out_features,
        "nhead": model.transformer.layers[0].self_attn.num_heads,
        "num_layers": len(model.transformer.layers),
        "dim_feedforward": model.transformer.layers[0].linear1.out_features,
        "param_emb_dim": model.param_proj[0].out_features,
        "hidden_dim": model.head[0].out_features,
        "dropout": float(model.frame_proj[3].p),
    }
    torch.save({"state_dict": model.state_dict(), "config": cfg}, path)


def load_temporal_regressor(
    path: PathLike,
    cdae: Optional[CDAE] = None,
    map_location: str | torch.device = "cpu",
) -> TemporalResistanceRegressor:
    ckpt = torch.load(path, map_location=map_location)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError("Expected checkpoint with 'state_dict' and 'config'.")

    cfg = ckpt.get("config", {})
    if cdae is None:
        cdae = CDAE(
            in_channels=cfg.get("in_channels", IN_CHANNELS),
            img_size=cfg.get("img_size", DEFAULT_IMG_SIZE),
            base_channels=cfg.get("base_channels", 32),
        )

    model = build_temporal_regressor_from_cdae(
        cdae,
        freeze_encoder=cfg.get("freeze_encoder", True),
        use_time=cfg.get("use_time", True),
        window_size=cfg.get("window_size", DEFAULT_WINDOW_SIZE),
        latent_channels=cfg.get("latent_channels", cdae.latent_channels),
        frame_emb_dim=cfg.get("frame_emb_dim", 128),
        d_model=cfg.get("d_model", 128),
        nhead=cfg.get("nhead", 4),
        num_layers=cfg.get("num_layers", 2),
        dim_feedforward=cfg.get("dim_feedforward", 256),
        param_emb_dim=cfg.get("param_emb_dim", 32),
        hidden_dim=cfg.get("hidden_dim", 256),
        dropout=cfg.get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["state_dict"])
    return model
