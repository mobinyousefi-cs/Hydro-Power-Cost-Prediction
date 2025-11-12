#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=
Project: Hydro Power Cost Prediction with Elastic Net
File: train.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-12
Updated: 2025-11-12
License: MIT License (see LICENSE file for details)
==========================================================================================================================================================================
"""
from __future__ import annotations
import argparse
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

from .config import load_config
from .utils import ensure_dir, save_json, set_seed, infer_target
from .data import load_dataframe, prepare_dataset
from .model import build_pipeline, default_param_grid


def parse_args():
    p = argparse.ArgumentParser(description="Train Elastic Net model for hydro cost prediction")
    p.add_argument("--config", type=str, default=None, help="Path to YAML config")
    p.add_argument("--csv", type=str, default=None, help="Path to dataset CSV (overrides config)")
    p.add_argument("--target", type=str, default=None, help="Target column (overrides config)")
    p.add_argument("--inspect", action="store_true", help="Only inspect columns & exit")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    csv_path = args.csv or cfg.csv_path
    df = load_dataframe(csv_path)

    if args.inspect:
        print("Columns:\n", df.dtypes)
        guesses = infer_target(df)
        if guesses:
            print("\nLikely target candidates:", guesses)
        return

    target = args.target or cfg.target
    X, y = prepare_dataset(df, target=target, timestamp_col=cfg.timestamp, feature_cfg=cfg.features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.split.get("test_size", 0.2), shuffle=cfg.split.get("shuffle", True), random_state=cfg.seed
    )

    pipe = build_pipeline(
        max_iter=cfg.model.get("elasticnet", {}).get("max_iter", 2000),
        tol=cfg.model.get("elasticnet", {}).get("tol", 1e-4),
    )
    grid = GridSearchCV(pipe, param_grid=default_param_grid(cfg), cv=5, n_jobs=-1, scoring="neg_mean_squared_error")
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    preds = best.predict(X_test)

    metrics = {
        "rmse": float(mean_squared_error(y_test, preds, squared=False)),
        "r2": float(r2_score(y_test, preds)),
        "best_params": grid.best_params_,
        "n_features_in_": int(getattr(best.named_steps["model"], "n_features_in_", len(X_train.columns))),
    }

    ensure_dir(cfg.artifacts_dir)
    joblib.dump(best, cfg.model_path)
    save_json(metrics, cfg.metrics_path)

    print("Saved model to:", cfg.model_path)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
