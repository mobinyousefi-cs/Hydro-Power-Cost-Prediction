#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=
Project: Hydro Power Cost Prediction with Elastic Net
File: evaluate.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-12
Updated: 2025-11-12
License: MIT License (see LICENSE file for details)
==========================================================================================================================================================================
"""
from __future__ import annotations
import argparse
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from .config import load_config
from .data import load_dataframe, prepare_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate saved model on a dataset")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--target", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    model = joblib.load(cfg.model_path)

    df = load_dataframe(args.csv)
    X, y = prepare_dataset(df, target=args.target, timestamp_col=cfg.timestamp, feature_cfg=cfg.features)
    preds = model.predict(X)
    rmse = mean_squared_error(y, preds, squared=False)
    r2 = r2_score(y, preds)
    print({"rmse": float(rmse), "r2": float(r2)})


if __name__ == "__main__":
    main()
