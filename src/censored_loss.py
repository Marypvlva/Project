"""
One-sided censored MSE для сопротивления с верхним потолком прибора (кОм).

Цензурированный таргет: истинное R известно только как R >= C (в Excel было "-").
Нецензурированный: обычный квадрат ошибки (pred - y)^2.
Цензурированный: штраф только за занижение, max(0, C - pred)^2.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Потолок измерителя (кОм); то же значение, что MAX_RESISTANCE_KOHM в parser.py.
MAX_RESISTANCE_KOHM = 86.0


def _as_bool_mask(censored: torch.Tensor) -> torch.Tensor:
    if censored.dtype == torch.bool:
        return censored
    return censored != 0


def one_sided_censored_squared_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    censored: torch.Tensor,
    *,
    censor_threshold: float = MAX_RESISTANCE_KOHM,
) -> torch.Tensor:
    """Поэлементная one-sided censored SE, форма как у pred [B, 1]."""
    if pred.shape != target.shape or pred.shape != censored.shape:
        raise ValueError(
            f"Формы pred/target/censored должны совпадать, получено "
            f"{tuple(pred.shape)}, {tuple(target.shape)}, {tuple(censored.shape)}"
        )

    mask = _as_bool_mask(censored)
    err = torch.zeros_like(pred)

    if (~mask).any():
        diff = pred[~mask] - target[~mask]
        err[~mask] = diff * diff

    if mask.any():
        under = censor_threshold - pred[mask]
        err[mask] = torch.clamp(under, min=0.0) ** 2

    return err


def one_sided_censored_abs_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    censored: torch.Tensor,
    *,
    censor_threshold: float = MAX_RESISTANCE_KOHM,
) -> torch.Tensor:
    """Поэлементная one-sided абсолютная ошибка для метрик."""
    mask = _as_bool_mask(censored)
    err = torch.zeros_like(pred)

    if (~mask).any():
        err[~mask] = (pred[~mask] - target[~mask]).abs()

    if mask.any():
        err[mask] = torch.clamp(censor_threshold - pred[mask], min=0.0)

    return err


class OneSidedCensoredMSELoss(nn.Module):
    """Средний one-sided censored MSE по батчу."""

    def __init__(self, censor_threshold: float = MAX_RESISTANCE_KOHM) -> None:
        super().__init__()
        self.censor_threshold = float(censor_threshold)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        censored: torch.Tensor,
    ) -> torch.Tensor:
        err = one_sided_censored_squared_error(
            pred,
            target,
            censored,
            censor_threshold=self.censor_threshold,
        )
        return err.mean()
