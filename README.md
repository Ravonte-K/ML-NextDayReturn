# ML Next-Day Return

A leakage-aware machine learning project template for predicting next-day stock returns and/or direction using daily OHLCV data.

## Project Objective

Build an end-to-end, reproducible pipeline that:
- Predicts **next-day return** (`y_return`) and optionally **next-day direction** (`y_dir`).
- Uses only information available at time `t`.
- Evaluates models with time-aware validation (single split first, then walk-forward).
- Reports both prediction metrics and simple trading-aware outcomes.

---

## Repository Structure

```text
ML-NextDayReturn/
├── configs/                # Tickers, dates, model params, experiment settings
│   ├── baseline_config.json
│   └── tickers.txt
├── data/
│   ├── raw/                # Downloaded market data cache
│   └── processed/          # Feature-ready parquet/csv outputs
├── models/                 # Saved model artifacts
├── notebooks/              # Exploration only (no production logic)
├── reports/                # Evaluation markdown + plots
├── src/
│   ├── __init__.py
│   ├── data.py             # Download/cache + target construction
│   ├── features.py         # Causal feature engineering
│   ├── train.py            # Train entrypoint (`python -m src.train`)
│   ├── evaluate.py         # Metrics + walk-forward evaluation scaffolding
│   └── predict.py          # Batch "tomorrow" prediction scaffolding
└── requirements.txt
```

---

## Quickstart

1. Create environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the training entrypoint (skeleton):

```bash
python -m src.train --config configs/baseline_config.json
```

3. Optional data step:

```bash
python -m src.data --config configs/baseline_config.json
```

---

## Planned Build Phases

This repo is set up to follow your progressive roadmap:

- **Phase 0**: Skeleton + reproducible entrypoints.
- **Phase 1**: OHLCV ingestion + clean target (`y_return`, `y_dir`).
- **Phase 2**: Honest baselines (zero + rolling mean).
- **Phase 3**: Simple causal features + linear models.
- **Phase 4**: Walk-forward cross-validation.
- **Phase 5**: Stronger tabular model (e.g., XGBoost/LightGBM).
- **Phase 6**: Trading-aware backtest metrics and equity curve.
- **Phase 7**: Packaging + daily prediction mode.

---

## Rules of Engagement

- No notebook-only logic.
- No random shuffling across time.
- No feature scaling fit on full data.
- No future leakage in rolling/lagged features.
- Keep all production workflows runnable from `src/` modules.

---

## Baseline Config Convention

`configs/baseline_config.json` includes:
- universe (tickers)
- date range
- split dates
- feature toggles
- model family placeholders
- trading assumptions (cost, threshold)

Update this config first whenever you start a new experiment.

---

## Definition of Done (Minimum Strong Project)

A strong stopping point is reached once the project has:
- Clean target construction.
- Causal feature set.
- Walk-forward evaluation.
- Baseline comparisons.
- Trading-aware backtest report.

