"""Normalization utilities used in the DSMP operator."""
from __future__ import annotations
import numpy as np
import pandas as pd


def minmax(series: pd.Series, eps: float = 1e-9) -> pd.Series:
    lo = float(series.min())
    hi = float(series.max())
    return ((series - lo) / (hi - lo + eps)).clip(0.0, 1.0)


def reference_normalize(series: pd.Series, lower: float, upper: float) -> pd.Series:
    if upper <= lower:
        raise ValueError("upper reference bound must exceed lower bound")
    return ((series - lower) / (upper - lower)).clip(0.0, 1.0)


def add_reference_and_portfolio_normalizations(df: pd.DataFrame) -> pd.DataFrame:
    """Add portfolio-relative and reference-based normalizations.

    Reference bounds are fixed and should be calibrated from design, regulation, or historical
    engineering envelopes. Values here are illustrative for the synthetic study.
    """
    out = df.copy()
    out["RTO_norm_pf"] = minmax(out["delta_RTO_hours"])
    out["RTO_norm_ref"] = reference_normalize(out["delta_RTO_hours"], lower=0.0, upper=96.0)
    out["P_var_norm"] = minmax(out["P_var_delta"])
    out["P_gap_norm"] = minmax(out["P_gap_plugin_predictive"].clip(lower=0.0))
    return out


def compute_consequence(df: pd.DataFrame, weights: tuple[float, float, float]) -> pd.Series:
    ws, wp, we = weights
    return (ws * df["C_safety"] + wp * df["C_operational"] + we * df["C_environmental"]).clip(0.0, 1.0)


def compute_uncertainty(df: pd.DataFrame, weights: tuple[float, float, float]) -> pd.Series:
    wv, wq, wm = weights
    return (wv * df["P_var_norm"] + wq * df["data_quality_penalty"] + wm * df["model_mismatch"]).clip(0.0, 1.0)
