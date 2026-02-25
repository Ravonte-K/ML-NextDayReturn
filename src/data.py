"""Data ingestion and target-construction scaffolding.

This module is intentionally lightweight for Phase 0 skeleton setup.
Fill in download logic and target generation in Phase 1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_PROCESSED_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "y_return",
    "y_dir",
]


def load_config(path: str) -> dict:
    """Load a JSON config file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_directories(config: dict) -> None:
    """Create data directories from config if missing."""
    raw_dir = Path(config["data"]["raw_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Data pipeline entrypoint")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_config.json",
        help="Path to config JSON",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_directories(config)

    print("[data] Skeleton ready.")
    print("[data] Next step: implement OHLCV download/cache and processed dataset output.")
    print(f"[data] Expected processed columns: {REQUIRED_PROCESSED_COLUMNS}")


if __name__ == "__main__":
    main()
