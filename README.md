# Hydro Power Cost Prediction (Elastic Net)

Predict one‑step‑ahead **marginal cost ($/MWh)** of hydro units using an **Elastic Net** regression (Ridge + Lasso). The mixed penalty controls multicollinearity among SCADA features (head, flow, gate, efficiency) while removing weak signals for a **stable, sparse, interpretable** model.

> Dataset: [https://www.kaggle.com/datasets/hemantk/hydropower-plant-dataset](https://www.kaggle.com/datasets/hemantk/hydropower-plant-dataset)

## Features

* `src/` layout with CLI entry points: `hydro-train`, `hydro-eval`, `hydro-predict`
* `ColumnTransformer` + `StandardScaler` + `OneHotEncoder` → robust preprocessing
* Elastic Net with `GridSearchCV` over `alpha` and `l1_ratio`
* Time features (hour/dayofweek/month), configurable lags & interactions
* Reproducible config in `configs/config.yaml`
* Plots utilities (residual & parity)
* CI: Ruff + Black + PyTest

## Quickstart

```bash
# 1) Create venv & install
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\\Scripts\\activate
pip install -e .[dev]

# 2) Download the Kaggle dataset (CSV) → place at data/hydropower.csv (or update path in configs/config.yaml)

# 3) Inspect columns to choose the target
hydro-train --csv data/hydropower.csv --inspect

# 4) Train (uses configs/config.yaml)
hydro-train --csv data/hydropower.csv --target marginal_cost

# 5) Evaluate on a held‑out file
hydro-eval --csv data/hydropower.csv --target marginal_cost

# 6) Predict on new data (with or without the target column)
hydro-predict --csv data/new_hours.csv --out artifacts/predictions.csv
```

### Notes on Target Column

Different hydropower datasets label the cost differently (e.g., `marginal_cost`, `cost_usd_mwh`, `price`). Use `--inspect` to list columns and suggested targets.

## Configuration

See `configs/config.yaml` for:

* paths (`csv_path`, artifacts)
* target & timestamp column names
* feature engineering toggles (time features, lags, interactions)
* ElasticNet search space (alpha, l1_ratio)
* split strategy

## Troubleshooting

* **Target not found** → run `--inspect`, then pass `--target <name>`.
* **All NaNs after lagging** → ensure your data is time‑ordered and that lag values are appropriate.
* **Poor generalization** → reduce lags/interactions, widen `alpha` grid, try robust time split.

## Development

```bash
ruff check . --fix
black .
pytest
```

## License

MIT
