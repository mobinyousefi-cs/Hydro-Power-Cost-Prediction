#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=
Project: Hydro Power Cost Prediction with Elastic Net
File: data.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-12
Updated: 2025-11-12
License: MIT License (see LICENSE file for details)
==========================================================================================================================================================================
"""
from __future__ import annotations
import pandas as pd
from typing import Tuple
from .features import add_time_features, add_lags, add_interactions, finalize_frame


def load_dataframe(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def prepare_dataset(
    df: pd.DataFrame,
    target: str,
    timestamp_col: str | None,
    feature_cfg: dict,
) -> Tuple[pd.DataFrame, pd.Series]:
    if feature_cfg.get("make_time_features", False) and timestamp_col:
        df = add_time_features(df, timestamp_col, feature_cfg.get("time_features", []))

    if feature_cfg.get("lags", {}).get("enable", False):
        lag_cfg = feature_cfg["lags"]
        df = add_lags(df, lag_cfg.get("columns", []), lag_cfg.get("lags", []))

    if feature_cfg.get("interactions", {}).get("enable", False):
        inter_cfg = feature_cfg["interactions"]
        df = add_interactions(df, inter_cfg.get("pairs", []))

    df = finalize_frame(df)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available: {list(df.columns)[:20]} ...")

    y = df[target]
    X = df.drop(columns=[target])
    return X, y
