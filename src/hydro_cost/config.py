#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=
Project: Hydro Power Cost Prediction with Elastic Net
File: config.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-12
Updated: 2025-11-12
License: MIT License (see LICENSE file for details)
==========================================================================================================================================================================
"""
from __future__ import annotations
import os
import yaml
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Config:
    seed: int
    artifacts_dir: str
    model_filename: str
    metrics_filename: str
    predictions_filename: str
    csv_path: str
    target: str
    timestamp: str | None
    autotype: bool
    features: Dict[str, Any]
    model: Dict[str, Any]
    split: Dict[str, Any]

    @property
    def model_path(self) -> str:
        return os.path.join(self.artifacts_dir, self.model_filename)

    @property
    def metrics_path(self) -> str:
        return os.path.join(self.artifacts_dir, self.metrics_filename)

    @property
    def predictions_path(self) -> str:
        return os.path.join(self.artifacts_dir, self.predictions_filename)


DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "configs", "config.yaml")
DEFAULT_CONFIG_PATH = os.path.normpath(DEFAULT_CONFIG_PATH)


def load_config(path: str | None = None) -> Config:
    cfg_path = path or os.environ.get("HYDRO_CONFIG", DEFAULT_CONFIG_PATH)
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)
