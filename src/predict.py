"""Prediction mode scaffolding (Phase 7)."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Load model and emit next-day predictions")
    parser.add_argument("--model-path", type=str, default="models/model.pkl")
    parser.add_argument("--as-of-date", type=str, default=None)
    args = parser.parse_args()

    print("[predict] Skeleton ready.")
    print(f"[predict] Model path: {args.model_path}")
    print(f"[predict] As-of date: {args.as_of_date}")


if __name__ == "__main__":
    main()
