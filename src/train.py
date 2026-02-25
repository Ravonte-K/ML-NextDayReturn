"""Training entrypoint for ML Next-Day Return project.

Phase 0 goal: reproducible command-line entrypoint.
Run with:
    python -m src.train --config configs/baseline_config.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluate import print_evaluation_plan
from src.features import print_feature_plan


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_config(config: dict) -> None:
    required_top_level = ["data", "target", "splits", "model", "trading"]
    missing = [k for k in required_top_level if k not in config]
    if missing:
        raise ValueError(f"Missing config sections: {missing}")


def ensure_output_dirs() -> None:
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train next-day return model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_config.json",
        help="Path to experiment config JSON",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    validate_config(config)
    ensure_output_dirs()

    print("[train] Phase 0 skeleton run successful.")
    print(f"[train] Loaded config: {args.config}")
    print(f"[train] Task: {config['model']['task']}, family: {config['model']['family']}")
    print("[train] Next actions: implement data loading, baseline models, and walk-forward evaluation.")

    print_feature_plan()
    print_evaluation_plan()


if __name__ == "__main__":
    main()
