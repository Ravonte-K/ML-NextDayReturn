"""Evaluation scaffolding for time-aware ML and trading diagnostics."""

from __future__ import annotations


def regression_metrics_plan() -> list[str]:
    """Core regression metrics for next-day return prediction."""
    return ["mae", "rmse"]


def direction_metrics_plan() -> list[str]:
    """Direction metrics for sign/label prediction."""
    return ["accuracy"]


def trading_metrics_plan() -> list[str]:
    """Trading-aware metrics for threshold strategy."""
    return ["total_return", "cagr", "sharpe", "max_drawdown", "hit_rate", "num_trades"]


def print_evaluation_plan() -> None:
    print("[evaluate] Regression metrics:", regression_metrics_plan())
    print("[evaluate] Direction metrics:", direction_metrics_plan())
    print("[evaluate] Trading metrics:", trading_metrics_plan())
    print("[evaluate] Next step: add split + walk-forward implementations.")


if __name__ == "__main__":
    print_evaluation_plan()
