from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from paper_reproduction.dlsa.objectives import build_portfolio_tensors, negative_sharpe
from paper_reproduction.dlsa.preprocessing import (
    build_active_feature_batch,
    cumulative_residual_windows,
    fourier_features,
)


@dataclass(frozen=True)
class TrainingResult:
    losses: list[float]
    daily_returns: np.ndarray
    annualized_sharpe: float


@dataclass(frozen=True)
class RollingTestResult:
    daily_returns: np.ndarray
    turnover: np.ndarray
    short_proportion: np.ndarray
    retrain_origins: list[int]
    training_losses: list[list[float]]
    annualized_mean: float
    annualized_volatility: float
    annualized_sharpe: float


@dataclass(frozen=True)
class StreamingTrainingResult:
    losses: list[float]
    daily_returns: np.ndarray
    active_asset_count: int


@dataclass(frozen=True)
class StreamingEvaluationResult:
    daily_returns: np.ndarray
    turnover: np.ndarray
    short_proportion: np.ndarray


def fit_single_training_window(
    model: nn.Module,
    residual_returns: np.ndarray,
    *,
    composition_matrices: np.ndarray | None = None,
    lookback: int = 30,
    epochs: int = 5,
    learning_rate: float = 0.001,
    transaction_cost: float = 0.0,
    short_holding_cost: float = 0.0,
    feature_type: str = "cumsum",
    temporal_batch_size: int | None = None,
    device: str = "cpu",
) -> TrainingResult:
    """Train one fixed window with the paper's economic Sharpe objective.

    This function validates the model/objective path. The 1,000/125 rolling
    re-estimation driver is a later milestone and is not implied here.
    """

    residuals = np.asarray(residual_returns, dtype=np.float32)
    windows, valid = cumulative_residual_windows(residuals, lookback)
    windows = _transform_features(windows, feature_type)
    evaluation_residuals = residuals[lookback:]
    finite_current = np.isfinite(evaluation_residuals)
    valid &= finite_current
    windows_tensor = torch.tensor(windows, dtype=torch.float32, device=device)
    returns_tensor = torch.tensor(
        np.nan_to_num(evaluation_residuals, nan=0.0),
        dtype=torch.float32,
        device=device,
    )
    valid_tensor = torch.tensor(valid, dtype=torch.bool, device=device)
    composition_tensor = (
        torch.tensor(composition_matrices[lookback:], dtype=torch.float32, device=device)
        if composition_matrices is not None
        else None
    )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    latest_returns = torch.zeros(windows.shape[0], device=device)
    evaluation_count = windows.shape[0]
    batch_size = temporal_batch_size or evaluation_count
    for _ in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        epoch_returns: list[torch.Tensor] = []
        for start in range(0, evaluation_count, batch_size):
            end = min(start + batch_size, evaluation_count)
            if end - start < 2 or not torch.any(valid_tensor[start:end]):
                continue
            scores = torch.zeros((end - start, valid.shape[1]), dtype=torch.float32, device=device)
            batch_valid = valid_tensor[start:end]
            scores[batch_valid] = model(windows_tensor[start:end][batch_valid])
            portfolio = build_portfolio_tensors(
                scores,
                returns_tensor[start:end],
                composition_matrices=(
                    composition_tensor[start:end] if composition_tensor is not None else None
                ),
                transaction_cost=transaction_cost,
                short_holding_cost=short_holding_cost,
            )
            loss = negative_sharpe(portfolio.returns)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            epoch_returns.append(portfolio.returns.detach())
        if not epoch_losses:
            raise ValueError("Training window contains no valid temporal batch.")
        losses.append(float(np.mean(epoch_losses)))
        latest_returns = torch.cat(epoch_returns)

    daily_returns = latest_returns.cpu().numpy()
    daily_std = float(np.std(daily_returns, ddof=0))
    annualized_sharpe = (
        float(np.mean(daily_returns) / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0
    )
    return TrainingResult(
        losses=losses,
        daily_returns=daily_returns,
        annualized_sharpe=annualized_sharpe,
    )


def evaluate_single_window(
    model: nn.Module,
    residual_returns: np.ndarray,
    *,
    composition_matrices: np.ndarray | None = None,
    lookback: int = 30,
    feature_type: str = "cumsum",
    transaction_cost: float = 0.0,
    short_holding_cost: float = 0.0,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residuals = np.asarray(residual_returns, dtype=np.float32)
    windows, valid = cumulative_residual_windows(residuals, lookback)
    windows = _transform_features(windows, feature_type)
    current_returns = torch.tensor(
        np.nan_to_num(residuals[lookback:], nan=0.0),
        dtype=torch.float32,
        device=device,
    )
    valid &= np.isfinite(residuals[lookback:])
    valid_tensor = torch.tensor(valid, dtype=torch.bool, device=device)
    windows_tensor = torch.tensor(windows, dtype=torch.float32, device=device)
    composition_tensor = (
        torch.tensor(composition_matrices[lookback:], dtype=torch.float32, device=device)
        if composition_matrices is not None
        else None
    )
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        scores = torch.zeros(valid.shape, dtype=torch.float32, device=device)
        if torch.any(valid_tensor):
            scores[valid_tensor] = model(windows_tensor[valid_tensor])
        portfolio = build_portfolio_tensors(
            scores,
            current_returns,
            composition_matrices=composition_tensor,
            transaction_cost=transaction_cost,
            short_holding_cost=short_holding_cost,
        )
    return (
        portfolio.returns.cpu().numpy(),
        portfolio.turnover.cpu().numpy(),
        portfolio.short_proportion.cpu().numpy(),
    )


def rolling_train_test(
    model_factory,
    residual_returns: np.ndarray,
    *,
    composition_matrices: np.ndarray | None = None,
    lookback: int = 30,
    training_length: int = 1000,
    retrain_frequency: int = 125,
    temporal_batch_size: int = 125,
    epochs: int = 100,
    learning_rate: float = 0.001,
    feature_type: str = "cumsum",
    transaction_cost: float = 0.0,
    short_holding_cost: float = 0.0,
    device: str = "cpu",
) -> RollingTestResult:
    """Run strict rolling train/test windows and stitch only OOS returns."""

    residuals = np.asarray(residual_returns, dtype=np.float32)
    if residuals.ndim != 2:
        raise ValueError("residual_returns must have shape (time, assets).")
    if residuals.shape[0] <= training_length:
        raise ValueError("Residual series must be longer than training_length.")
    if composition_matrices is not None and composition_matrices.shape[0] != residuals.shape[0]:
        raise ValueError("Composition matrices must align with residual returns.")

    stitched_returns: list[np.ndarray] = []
    stitched_turnover: list[np.ndarray] = []
    stitched_short: list[np.ndarray] = []
    retrain_origins: list[int] = []
    all_losses: list[list[float]] = []

    for origin in range(training_length, residuals.shape[0], retrain_frequency):
        test_end = min(origin + retrain_frequency, residuals.shape[0])
        train_start = origin - training_length
        model = model_factory()
        fit = fit_single_training_window(
            model,
            residuals[train_start:origin],
            composition_matrices=(
                composition_matrices[train_start:origin]
                if composition_matrices is not None
                else None
            ),
            lookback=lookback,
            epochs=epochs,
            learning_rate=learning_rate,
            transaction_cost=transaction_cost,
            short_holding_cost=short_holding_cost,
            feature_type=feature_type,
            temporal_batch_size=temporal_batch_size,
            device=device,
        )
        evaluation_start = origin - lookback
        test_returns, test_turnover, test_short = evaluate_single_window(
            model,
            residuals[evaluation_start:test_end],
            composition_matrices=(
                composition_matrices[evaluation_start:test_end]
                if composition_matrices is not None
                else None
            ),
            lookback=lookback,
            feature_type=feature_type,
            transaction_cost=transaction_cost,
            short_holding_cost=short_holding_cost,
            device=device,
        )
        expected = test_end - origin
        if len(test_returns) != expected:
            raise AssertionError("OOS test slice length does not match its calendar span.")
        stitched_returns.append(test_returns)
        stitched_turnover.append(test_turnover)
        stitched_short.append(test_short)
        retrain_origins.append(origin)
        all_losses.append(fit.losses)

    daily_returns = np.concatenate(stitched_returns)
    turnover = np.concatenate(stitched_turnover)
    short_proportion = np.concatenate(stitched_short)
    daily_mean = float(np.mean(daily_returns))
    daily_volatility = float(np.std(daily_returns, ddof=0))
    annualized_mean = daily_mean * 252
    annualized_volatility = daily_volatility * np.sqrt(252)
    annualized_sharpe = (
        annualized_mean / annualized_volatility if annualized_volatility > 0 else 0.0
    )
    return RollingTestResult(
        daily_returns=daily_returns,
        turnover=turnover,
        short_proportion=short_proportion,
        retrain_origins=retrain_origins,
        training_losses=all_losses,
        annualized_mean=annualized_mean,
        annualized_volatility=annualized_volatility,
        annualized_sharpe=annualized_sharpe,
    )


def fit_training_window_streaming(
    model: nn.Module,
    residual_returns: np.ndarray,
    *,
    composition_matrices: np.ndarray | None = None,
    lookback: int = 30,
    epochs: int = 100,
    learning_rate: float = 0.001,
    temporal_batch_size: int = 125,
    model_chunk_size: int = 4096,
    gradient_checkpointing: bool = False,
    feature_type: str = "cumsum",
    transaction_cost: float = 0.0,
    short_holding_cost: float = 0.0,
    zero_is_missing: bool = True,
    device: str = "cpu",
) -> StreamingTrainingResult:
    """Train a full-universe window without materializing every rolling window."""

    residuals = np.asarray(residual_returns, dtype=np.float32)
    if residuals.ndim != 2 or residuals.shape[0] <= lookback:
        raise ValueError("Training data must have shape (time, assets) beyond lookback.")
    if composition_matrices is not None and composition_matrices.shape[:2] != residuals.shape:
        raise ValueError("Composition matrices must align with residual returns.")

    batches = [
        build_active_feature_batch(
            residuals,
            start=start,
            end=min(start + temporal_batch_size, residuals.shape[0]),
            lookback=lookback,
            feature_type=feature_type,
            zero_is_missing=zero_is_missing,
        )
        for start in range(lookback, residuals.shape[0], temporal_batch_size)
    ]
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    latest_returns: list[np.ndarray] = []

    for _ in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        epoch_returns: list[np.ndarray] = []
        for batch in batches:
            if batch.features.shape[0] == 0 or batch.end - batch.start < 2:
                continue
            features = torch.tensor(batch.features, dtype=torch.float32, device=device)
            valid = torch.tensor(batch.valid_mask, dtype=torch.bool, device=device)
            current_returns = torch.tensor(
                batch.current_returns,
                dtype=torch.float32,
                device=device,
            )
            active_scores = _forward_in_chunks(
                model,
                features,
                model_chunk_size,
                use_checkpoint=gradient_checkpointing,
            )
            scores = torch.zeros(valid.shape, dtype=torch.float32, device=device)
            scores = scores.masked_scatter(valid, active_scores)
            compositions = (
                torch.tensor(
                    composition_matrices[batch.start : batch.end],
                    dtype=torch.float32,
                    device=device,
                )
                if composition_matrices is not None
                else None
            )
            portfolio = build_portfolio_tensors(
                scores,
                current_returns,
                composition_matrices=compositions,
                transaction_cost=transaction_cost,
                short_holding_cost=short_holding_cost,
            )
            loss = negative_sharpe(portfolio.returns)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            epoch_returns.append(portfolio.returns.detach().cpu().numpy())
        if not epoch_losses:
            raise ValueError("Training window contains no valid temporal batch.")
        losses.append(float(np.mean(epoch_losses)))
        latest_returns = epoch_returns

    return StreamingTrainingResult(
        losses=losses,
        daily_returns=np.concatenate(latest_returns),
        active_asset_count=residuals.shape[1],
    )


def evaluate_model_streaming(
    model: nn.Module,
    residual_returns: np.ndarray,
    *,
    composition_matrices: np.ndarray | None = None,
    lookback: int = 30,
    temporal_batch_size: int = 125,
    model_chunk_size: int = 4096,
    feature_type: str = "cumsum",
    transaction_cost: float = 0.0,
    short_holding_cost: float = 0.0,
    zero_is_missing: bool = True,
    device: str = "cpu",
) -> StreamingEvaluationResult:
    residuals = np.asarray(residual_returns, dtype=np.float32)
    model = model.to(device)
    model.eval()
    returns_blocks: list[np.ndarray] = []
    turnover_blocks: list[np.ndarray] = []
    short_blocks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(lookback, residuals.shape[0], temporal_batch_size):
            end = min(start + temporal_batch_size, residuals.shape[0])
            batch = build_active_feature_batch(
                residuals,
                start=start,
                end=end,
                lookback=lookback,
                feature_type=feature_type,
                zero_is_missing=zero_is_missing,
            )
            features = torch.tensor(batch.features, dtype=torch.float32, device=device)
            valid = torch.tensor(batch.valid_mask, dtype=torch.bool, device=device)
            scores = torch.zeros(valid.shape, dtype=torch.float32, device=device)
            if features.shape[0]:
                scores = scores.masked_scatter(
                    valid,
                    _forward_in_chunks(model, features, model_chunk_size),
                )
            current_returns = torch.tensor(
                batch.current_returns,
                dtype=torch.float32,
                device=device,
            )
            compositions = (
                torch.tensor(
                    composition_matrices[start:end],
                    dtype=torch.float32,
                    device=device,
                )
                if composition_matrices is not None
                else None
            )
            portfolio = build_portfolio_tensors(
                scores,
                current_returns,
                composition_matrices=compositions,
                transaction_cost=transaction_cost,
                short_holding_cost=short_holding_cost,
            )
            returns_blocks.append(portfolio.returns.cpu().numpy())
            turnover_blocks.append(portfolio.turnover.cpu().numpy())
            short_blocks.append(portfolio.short_proportion.cpu().numpy())
    return StreamingEvaluationResult(
        daily_returns=np.concatenate(returns_blocks),
        turnover=np.concatenate(turnover_blocks),
        short_proportion=np.concatenate(short_blocks),
    )


def rolling_train_test_streaming(
    model_factory,
    residual_returns: np.ndarray,
    *,
    composition_matrices: np.ndarray | None = None,
    lookback: int = 30,
    training_length: int = 1000,
    retrain_frequency: int = 125,
    temporal_batch_size: int = 125,
    model_chunk_size: int = 4096,
    gradient_checkpointing: bool = False,
    epochs: int = 100,
    learning_rate: float = 0.001,
    feature_type: str = "cumsum",
    transaction_cost: float = 0.0,
    short_holding_cost: float = 0.0,
    zero_is_missing: bool = True,
    max_retrains: int | None = None,
    device: str = "cpu",
) -> RollingTestResult:
    """Paper-style rolling neural experiment with bounded working memory."""

    residuals = np.asarray(residual_returns, dtype=np.float32)
    if residuals.ndim != 2 or residuals.shape[0] <= training_length:
        raise ValueError("Residual data must extend beyond the training window.")
    if composition_matrices is not None and composition_matrices.shape[:2] != residuals.shape:
        raise ValueError("Composition matrices must align with residual returns.")
    stitched_returns: list[np.ndarray] = []
    stitched_turnover: list[np.ndarray] = []
    stitched_short: list[np.ndarray] = []
    retrain_origins: list[int] = []
    all_losses: list[list[float]] = []

    origins = list(range(training_length, residuals.shape[0], retrain_frequency))
    if max_retrains is not None:
        origins = origins[:max_retrains]
    for origin in origins:
        test_end = min(origin + retrain_frequency, residuals.shape[0])
        train_start = origin - training_length
        training_slice = residuals[train_start:origin]
        if zero_is_missing:
            assets_to_trade = np.count_nonzero(training_slice, axis=0) >= lookback
        else:
            assets_to_trade = np.sum(np.isfinite(training_slice), axis=0) >= lookback
        if not np.any(assets_to_trade):
            raise ValueError(f"No active residuals at rolling origin {origin}.")
        train_data = training_slice[:, assets_to_trade]
        train_compositions = (
            composition_matrices[train_start:origin, assets_to_trade]
            if composition_matrices is not None
            else None
        )
        model = model_factory()
        fit = fit_training_window_streaming(
            model,
            train_data,
            composition_matrices=train_compositions,
            lookback=lookback,
            epochs=epochs,
            learning_rate=learning_rate,
            temporal_batch_size=temporal_batch_size,
            model_chunk_size=model_chunk_size,
            gradient_checkpointing=gradient_checkpointing,
            feature_type=feature_type,
            transaction_cost=transaction_cost,
            short_holding_cost=short_holding_cost,
            zero_is_missing=zero_is_missing,
            device=device,
        )
        evaluation_start = origin - lookback
        test_data = residuals[evaluation_start:test_end, assets_to_trade]
        test_compositions = (
            composition_matrices[evaluation_start:test_end, assets_to_trade]
            if composition_matrices is not None
            else None
        )
        evaluated = evaluate_model_streaming(
            model,
            test_data,
            composition_matrices=test_compositions,
            lookback=lookback,
            temporal_batch_size=temporal_batch_size,
            model_chunk_size=model_chunk_size,
            feature_type=feature_type,
            transaction_cost=transaction_cost,
            short_holding_cost=short_holding_cost,
            zero_is_missing=zero_is_missing,
            device=device,
        )
        stitched_returns.append(evaluated.daily_returns)
        stitched_turnover.append(evaluated.turnover)
        stitched_short.append(evaluated.short_proportion)
        retrain_origins.append(origin)
        all_losses.append(fit.losses)

    daily_returns = np.concatenate(stitched_returns)
    turnover = np.concatenate(stitched_turnover)
    short_proportion = np.concatenate(stitched_short)
    annualized_mean = float(daily_returns.mean() * 252)
    annualized_volatility = float(daily_returns.std(ddof=0) * np.sqrt(252))
    annualized_sharpe = (
        annualized_mean / annualized_volatility if annualized_volatility > 0 else 0.0
    )
    return RollingTestResult(
        daily_returns=daily_returns,
        turnover=turnover,
        short_proportion=short_proportion,
        retrain_origins=retrain_origins,
        training_losses=all_losses,
        annualized_mean=annualized_mean,
        annualized_volatility=annualized_volatility,
        annualized_sharpe=annualized_sharpe,
    )


def _transform_features(windows: np.ndarray, feature_type: str) -> np.ndarray:
    if feature_type == "cumsum":
        return windows
    if feature_type == "fourier":
        return fourier_features(windows)
    raise ValueError("feature_type must be 'cumsum' or 'fourier'.")


def _forward_in_chunks(
    model: nn.Module,
    features: torch.Tensor,
    chunk_size: int,
    *,
    use_checkpoint: bool = False,
) -> torch.Tensor:
    if chunk_size <= 0:
        raise ValueError("model_chunk_size must be positive.")
    outputs: list[torch.Tensor] = []
    for start in range(0, features.shape[0], chunk_size):
        chunk = features[start : start + chunk_size]
        if use_checkpoint:
            outputs.append(checkpoint(model, chunk, use_reentrant=False))
        else:
            outputs.append(model(chunk))
    return torch.cat(outputs)
