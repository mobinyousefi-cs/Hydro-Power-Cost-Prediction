#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=
Project: Hydro Power Cost Prediction with Elastic Net
File: model.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-12
Updated: 2025-11-12
License: MIT License (see LICENSE file for details)
==========================================================================================================================================================================
"""
from __future__ import annotations
from typing import Dict, Any
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet


NUMERIC_SELECTOR = "numeric"
CATEGORICAL_SELECTOR = "categorical"


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(steps=[("scaler", StandardScaler(with_mean=True, with_std=True))])
    categorical = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    pre = ColumnTransformer(
        transformers=[
            (NUMERIC_SELECTOR, numeric, make_numeric_selector()),
            (CATEGORICAL_SELECTOR, categorical, make_categorical_selector()),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre


def make_numeric_selector():
    def _selector(X):
        return X.select_dtypes(include=["number"]).columns
    return _selector


def make_categorical_selector():
    def _selector(X):
        return X.select_dtypes(exclude=["number"]).columns
    return _selector


def build_pipeline(alpha: float | None = None, l1_ratio: float | None = None, max_iter: int = 2000, tol: float = 1e-4) -> Pipeline:
    pre = build_preprocessor()
    model = ElasticNet(alpha=alpha or 1.0, l1_ratio=l1_ratio or 0.5, max_iter=max_iter, tol=tol, random_state=0)
    pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])
    return pipe


def default_param_grid(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if cfg is None:
        alphas = np.logspace(-3, 1, 9)
        l1 = [0.1, 0.3, 0.5, 0.7, 0.9]
        return {
            "model__alpha": list(alphas),
            "model__l1_ratio": l1,
        }
    m = cfg.get("model", {}).get("elasticnet", {})
    return {
        "model__alpha": m.get("alpha", [0.001, 0.01, 0.1, 1.0, 10.0]),
        "model__l1_ratio": m.get("l1_ratio", [0.1, 0.3, 0.5, 0.7, 0.9]),
    }
