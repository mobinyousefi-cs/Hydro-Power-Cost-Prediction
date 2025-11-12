#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=
Project: Hydro Power Cost Prediction with Elastic Net
File: test_pipeline_smoke.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-12
Updated: 2025-11-12
License: MIT License (see LICENSE file for details)
==========================================================================================================================================================================
"""
import numpy as np
import pandas as pd
from hydro_cost.model import build_pipeline, default_param_grid
from sklearn.model_selection import train_test_split, GridSearchCV


def synthetic_df(n=200, seed=42):
    rng = np.random.default_rng(seed)
    time = pd.date_range("2024-01-01", periods=n, freq="H")
    head = rng.normal(50, 5, size=n)
    flow = rng.normal(200, 20, size=n)
    gate = np.clip(rng.normal(0.6, 0.1, size=n), 0, 1)
    eff = np.clip(0.9 - 0.001 * (flow - 200) ** 2 / 400, 0.7, 0.95)
    y = 8 + 0.05 * flow + 0.03 * head + 4 * gate + 10 * (1 - eff) + rng.normal(0, 1, n)
    return pd.DataFrame({
        "timestamp": time,
        "head_m": head,
        "flow_m3s": flow,
        "gate_opening": gate,
        "efficiency": eff,
        "marginal_cost": y,
    })


def test_gridsearch_runs():
    df = synthetic_df()
    X = df.drop(columns=["marginal_cost"])  # pipeline will auto-handle timestamp
    y = df["marginal_cost"]

    pipe = build_pipeline()
    grid = GridSearchCV(pipe, default_param_grid(), cv=3, n_jobs=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    grid.fit(X_train, y_train)

    score = grid.best_estimator_.score(X_test, y_test)
    assert np.isfinite(score)
