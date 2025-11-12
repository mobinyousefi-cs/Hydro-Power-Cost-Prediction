#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=
Project: Hydro Power Cost Prediction with Elastic Net
File: predict.py
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
from .config import load_config
from .data import load_dataframe, prepare_dataset
from .utils import ensure_dir


def parse_args():
    p = argparse.ArgumentParser(description="Run predictions with saved model")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--target", type=str, default=None, help="If provided and present, it will be dropped before predicting")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    model = joblib.load(cfg.model_path)

    df = load_dataframe(args.csv)

    target = args.target or cfg.target
    if target in df.columns:
        X, _ = prepare_dataset(df, target=target, timestamp_col=cfg.timestamp, feature_cfg=cfg.features)
    else:
        tmp_target = df.columns[0]
        df[tmp_target] = 0.0
        X, _ = prepare_dataset(df, target=tmp_target, timestamp_col=cfg.timestamp, feature_cfg=cfg.features)

    preds = model.predict(X)

    out_path = args.out or cfg.predictions_path
    ensure_dir(cfg.artifacts_dir)
    pd.DataFrame({"prediction": preds}).to_csv(out_path, index=False)
    print("Saved predictions to:", out_path)


if __name__ == "__main__":
    main()
