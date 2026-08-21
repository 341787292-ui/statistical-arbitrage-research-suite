from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActiveFeatureBatch:
    features: np.ndarray
    valid_mask: np.ndarray
    current_returns: np.ndarray
    start: int
    end: int


def cumulative_residual_windows(
    residual_returns: np.ndarray,
    lookback: int,
    *,
    zero_is_missing: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build lagged cumulative residual-price windows.

    Output index ``i`` corresponds to trading day ``t = i + lookback`` and uses
    residual returns from ``t-lookback`` through ``t-1`` only.
    """

    residuals = np.asarray(residual_returns, dtype=np.float64)
    if residuals.ndim != 2:
        raise ValueError("residual_returns must have shape (time, assets).")
    if lookback < 2 or residuals.shape[0] <= lookback:
        raise ValueError("lookback must leave at least one evaluation day.")

    time_count, asset_count = residuals.shape
    windows = np.zeros((time_count - lookback, asset_count, lookback), dtype=np.float64)
    valid = np.zeros((time_count - lookback, asset_count), dtype=bool)

    for output_index, t in enumerate(range(lookback, time_count)):
        history = residuals[t - lookback : t]
        selected = np.all(np.isfinite(history), axis=0)
        if zero_is_missing:
            selected &= ~np.any(history == 0, axis=0)
        if np.any(selected):
            windows[output_index, selected] = np.cumsum(
                history[:, selected],
                axis=0,
            ).T
            valid[output_index, selected] = True
    return windows, valid


def fourier_features(cumulative_windows: np.ndarray) -> np.ndarray:
    """Encode a real cumulative window with its non-redundant FFT components."""

    windows = np.asarray(cumulative_windows, dtype=np.float64)
    if windows.ndim < 2:
        raise ValueError("cumulative_windows must include a lookback dimension.")
    lookback = windows.shape[-1]
    transformed = np.fft.rfft(windows, axis=-1)
    real = transformed.real
    imag = transformed.imag[..., 1:-1] if lookback % 2 == 0 else transformed.imag[..., 1:]
    features = np.concatenate([real, imag], axis=-1)
    if features.shape[-1] != lookback:
        raise AssertionError("Fourier representation must preserve the input dimension.")
    return features


def build_active_feature_batch(
    residual_returns: np.ndarray,
    *,
    start: int,
    end: int,
    lookback: int = 30,
    feature_type: str = "cumsum",
    zero_is_missing: bool = True,
) -> ActiveFeatureBatch:
    """Build only active residual windows for a temporal training batch."""

    residuals = np.asarray(residual_returns, dtype=np.float32)
    if residuals.ndim != 2:
        raise ValueError("residual_returns must have shape (time, assets).")
    if start < lookback or end <= start or end > residuals.shape[0]:
        raise ValueError("Batch bounds must be within the data and after lookback.")

    day_count = end - start
    asset_count = residuals.shape[1]
    valid_mask = np.zeros((day_count, asset_count), dtype=bool)
    feature_blocks: list[np.ndarray] = []
    for row, t in enumerate(range(start, end)):
        history = residuals[t - lookback : t]
        selected = np.all(np.isfinite(history), axis=0)
        if zero_is_missing:
            selected &= ~np.any(history == 0, axis=0)
        valid_mask[row] = selected
        if not np.any(selected):
            continue
        cumulative = np.cumsum(history[:, selected], axis=0).T
        if feature_type == "fourier":
            cumulative = fourier_features(cumulative)
        elif feature_type != "cumsum":
            raise ValueError("feature_type must be 'cumsum' or 'fourier'.")
        feature_blocks.append(np.asarray(cumulative, dtype=np.float32))

    features = (
        np.concatenate(feature_blocks, axis=0)
        if feature_blocks
        else np.empty((0, lookback), dtype=np.float32)
    )
    if features.shape[0] != int(valid_mask.sum()):
        raise AssertionError("Active features must follow valid-mask row-major order.")
    return ActiveFeatureBatch(
        features=features,
        valid_mask=valid_mask,
        current_returns=np.nan_to_num(residuals[start:end], nan=0.0),
        start=start,
        end=end,
    )
