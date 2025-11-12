#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=
Project: Hydro Power Cost Prediction with Elastic Net
File: plots.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-12
Updated: 2025-11-12
License: MIT License (see LICENSE file for details)
==========================================================================================================================================================================
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def residual_plot(y_true, y_pred, title: str = "Residuals"):
    resid = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=y_pred, y=resid, ax=ax, s=12)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (y - ŷ)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def parity_plot(y_true, y_pred, title: str = "Parity Plot"):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=y_true, y=y_pred, ax=ax, s=12)
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    ax.plot([mn, mx], [mn, mx], color="black", lw=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    fig.tight_layout()
    return fig
