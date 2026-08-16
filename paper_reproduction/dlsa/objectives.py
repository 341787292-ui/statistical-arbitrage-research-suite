from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PortfolioTensors:
    residual_weights: torch.Tensor
    stock_weights: torch.Tensor
    returns: torch.Tensor
    turnover: torch.Tensor
    short_proportion: torch.Tensor


def build_portfolio_tensors(
    raw_residual_scores: torch.Tensor,
    residual_returns: torch.Tensor,
    *,
    composition_matrices: torch.Tensor | None = None,
    transaction_cost: float = 0.0,
    short_holding_cost: float = 0.0,
) -> PortfolioTensors:
    """Map residual scores to L1-normalized stock portfolios and returns."""

    if raw_residual_scores.shape != residual_returns.shape:
        raise ValueError("Scores and residual returns must have the same shape.")
    if composition_matrices is None:
        raw_stock_weights = raw_residual_scores
    else:
        if composition_matrices.shape[:2] != raw_residual_scores.shape:
            raise ValueError("Composition matrices must have shape (time, residuals, stocks).")
        raw_stock_weights = torch.bmm(
            raw_residual_scores.unsqueeze(1),
            composition_matrices,
        ).squeeze(1)

    gross = raw_stock_weights.abs().sum(dim=1, keepdim=True)
    safe_gross = torch.where(gross > 0, gross, torch.ones_like(gross))
    residual_weights = raw_residual_scores / safe_gross
    stock_weights = raw_stock_weights / safe_gross
    residual_weights = torch.where(gross > 0, residual_weights, torch.zeros_like(residual_weights))
    stock_weights = torch.where(gross > 0, stock_weights, torch.zeros_like(stock_weights))

    previous = torch.cat([torch.zeros_like(stock_weights[:1]), stock_weights[:-1]], dim=0)
    turnover = (stock_weights - previous).abs().sum(dim=1)
    short_proportion = torch.minimum(stock_weights, torch.zeros_like(stock_weights)).abs().sum(dim=1)
    gross_returns = (residual_weights * residual_returns).sum(dim=1)
    net_returns = (
        gross_returns
        - transaction_cost * turnover
        - short_holding_cost * short_proportion
    )
    return PortfolioTensors(
        residual_weights=residual_weights,
        stock_weights=stock_weights,
        returns=net_returns,
        turnover=turnover,
        short_proportion=short_proportion,
    )


def negative_sharpe(returns: torch.Tensor) -> torch.Tensor:
    if returns.ndim != 1 or returns.numel() < 2:
        raise ValueError("At least two one-dimensional returns are required.")
    return -returns.mean() / returns.std(unbiased=True).clamp_min(1e-8)

