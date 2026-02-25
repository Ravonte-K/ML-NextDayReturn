"""Feature engineering scaffolding for causal, leakage-safe signals."""

from __future__ import annotations


FEATURE_BACKLOG = [
    "ret_1",
    "ret_2",
    "ret_5",
    "vol_10",
    "mean_10",
    "mom_sma_10",
    "mom_sma_20",
    "vol_ratio_20",
    "range_pct",
    "rsi_14 (optional)",
    "atr_proxy (optional)",
    "day_of_week (optional)",
]


def print_feature_plan() -> None:
    """Print planned feature roadmap for implementation phases."""
    print("[features] Planned causal features:")
    for feat in FEATURE_BACKLOG:
        print(f"  - {feat}")


if __name__ == "__main__":
    print_feature_plan()
