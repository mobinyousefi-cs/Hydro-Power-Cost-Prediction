#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=
Project: Hydro Power Cost Prediction with Elastic Net
File: utils.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-12
Updated: 2025-11-12
License: MIT License (see LICENSE file for details)
==========================================================================================================================================================================
"""
from __future__ import annotations
import os
import json
import random
import numpy as np
import pandas as pd
from typing import Iterable


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def infer_target(df: pd.DataFrame) -> list[str]:
    candidates = [c for c in df.columns if any(k in c.lower() for k in ["cost", "price", "marginal", "$/", "usd"])]
    return candidates


def select_columns(df: pd.DataFrame, exclude: Iterable[str]) -> list[str]:
    ex = set(exclude)
    return [c for c in df.columns if c not in ex]
