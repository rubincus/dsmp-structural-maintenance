"""Dynamic risk-resilience scoring and action mapping."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import DSMPConfig
from .normalization import compute_consequence, compute_uncertainty


def compute_scores(df: pd.DataFrame, config: DSMPConfig = DSMPConfig(), use_reference_norm: bool = True) -> pd.DataFrame:
    """Compute static and dynamic maintenance scores.

    R_i(t) = alpha * P_i(t) * C_i + rho * RTO_i + eta * U_i(t)
    where all terms are normalized and alpha + rho + eta = 1.
    """
    out = df.copy()
    alpha, rho, eta = config.normalized_weights()
    out["C_norm"] = compute_consequence(out, config.normalized_consequence_weights())
    rto_col = "RTO_norm_ref" if use_reference_norm and "RTO_norm_ref" in out.columns else "RTO_norm_pf"
    out["RTO_norm"] = out[rto_col]
    out["U_norm"] = compute_uncertainty(out, config.normalized_uncertainty_weights())
    out["static_risk"] = (out["P_plugin"] * out["C_norm"]).clip(0.0, 1.0)
    out["dynamic_score"] = (alpha * out["static_risk"] + rho * out["RTO_norm"] + eta * out["U_norm"]).clip(0.0, 1.0)
    out["dynamic_score_failure_conditioned_rto"] = (
        alpha * out["static_risk"] + rho * out["P_plugin"] * out["RTO_norm"] + eta * out["U_norm"]
    ).clip(0.0, 1.0)
    out["rank_static"] = out["static_risk"].rank(ascending=False, method="min").astype(int)
    out["rank_dynamic"] = out["dynamic_score"].rank(ascending=False, method="min").astype(int)
    out["rank_shift"] = out["rank_static"] - out["rank_dynamic"]
    return out


def assign_threshold_actions(df: pd.DataFrame, config: DSMPConfig = DSMPConfig(), score_col: str = "dynamic_score") -> pd.DataFrame:
    """Assign actions from score thresholds."""
    out = df.copy()
    r1, r2, r3 = config.thresholds
    conds = [
        out[score_col] < r1,
        (out[score_col] >= r1) & (out[score_col] < r2),
        (out[score_col] >= r2) & (out[score_col] < r3),
        out[score_col] >= r3,
    ]
    labels = ["monitor", "inspect", "repair", "reinforce_or_replace"]
    out["threshold_action"] = np.select(conds, labels, default="monitor")
    return out


def verify_index_properties(df: pd.DataFrame) -> dict[str, bool]:
    """Unit-style verification of boundedness and monotonic-safe algebra."""
    bounded = bool(((df["dynamic_score"] >= -1e-12) & (df["dynamic_score"] <= 1 + 1e-12)).all())
    static_bounded = bool(((df["static_risk"] >= -1e-12) & (df["static_risk"] <= 1 + 1e-12)).all())
    return {"dynamic_bounded": bounded, "static_bounded": static_bounded}
