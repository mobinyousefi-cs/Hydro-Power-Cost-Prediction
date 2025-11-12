#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=
Project: Hydro Power Cost Prediction with Elastic Net
File: features.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-12
Updated: 2025-11-12
License: MIT License (see LICENSE file for details)
==========================================================================================================================================================================
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Iterable


def add_time_features(df: pd.DataFrame, timestamp_col: str, features: list[str]) -> pd.DataFrame:
    if timestamp_col not in df.columns:
        return df
    ts = pd.to_datetime(df[timestamp_col], errors="coerce")
    out = df.copy()
    if "hour" in features:
        out["hour"] = ts.dt.hour
    if "dayofweek" in features:
        out["dayofweek"] = ts.dt.dayofweek
    if "month" in features:
        out["month"] = ts.dt.month
    return out


def add_lags(df: pd.DataFrame, columns: Iterable[str], lags: Iterable[int]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        for l in lags:
            out[f"{col}_lag{l}"] = out[col].shift(l)
    return out


def add_interactions(df: pd.DataFrame, pairs: list[list[str]]) -> pd.DataFrame:
    out = df.copy()
    for a, b in pairs:
        if a in out.columns and b in out.columns:
            out[f"{a}__x__{b}"] = out[a] * out[b]
    return out


def finalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna().reset_index(drop=True)
