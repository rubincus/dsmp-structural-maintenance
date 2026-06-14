"""Sensitivity, weight robustness, and rank-stability experiments."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .config import DSMPConfig


def lambda_eta_surface(asset_row: pd.Series, grid_n: int = 50) -> pd.DataFrame:
    """Score surface over rho and eta with alpha = 1 - rho - eta when feasible."""
    records = []
    static = float(asset_row["static_risk"])
    rto = float(asset_row["RTO_norm"])
    u = float(asset_row["U_norm"])
    for rho in np.linspace(0, 0.65, grid_n):
        for eta in np.linspace(0, 0.65, grid_n):
            alpha = 1.0 - rho - eta
            if alpha < 0:
                continue
            score = alpha * static + rho * rto + eta * u
            records.append({"alpha": alpha, "rho": rho, "eta": eta, "score": score})
    return pd.DataFrame(records)


def rank_stability(df: pd.DataFrame, score_col: str = "dynamic_score", perturbation_levels=None, simulations: int = 250, seed: int = 0) -> pd.DataFrame:
    if perturbation_levels is None:
        perturbation_levels = np.linspace(0, 0.25, 11)
    rng = np.random.default_rng(seed)
    base_scores = df[score_col].to_numpy(dtype=float)
    base_rank = pd.Series((-base_scores).argsort().argsort()).to_numpy(dtype=float)
    records = []
    for sigma in perturbation_levels:
        cors = []
        for _ in range(simulations):
            perturbed = base_scores + rng.normal(0, sigma, size=len(base_scores))
            rank = pd.Series((-perturbed).argsort().argsort()).to_numpy(dtype=float)
            corr = spearmanr(base_rank, rank).correlation
            cors.append(corr if np.isfinite(corr) else 1.0)
        records.append({"sigma": float(sigma), "mean_rank_correlation": float(np.mean(cors)), "std_rank_correlation": float(np.std(cors))})
    return pd.DataFrame(records)


def weight_simplex_robustness(df: pd.DataFrame, config: DSMPConfig = DSMPConfig(), samples: int = 1500) -> pd.DataFrame:
    """Sample the weight simplex and report rank variability for each asset."""
    rng = np.random.default_rng(config.seed + 77)
    center = np.array(config.normalized_weights())
    concentration = 60.0 * center
    weights = rng.dirichlet(concentration, size=samples)
    static = df["static_risk"].to_numpy(dtype=float)
    rto = df["RTO_norm"].to_numpy(dtype=float)
    u = df["U_norm"].to_numpy(dtype=float)
    rank_matrix = np.empty((samples, len(df)), dtype=int)
    for k, (alpha, rho, eta) in enumerate(weights):
        score = alpha * static + rho * rto + eta * u
        rank_matrix[k] = (-score).argsort().argsort() + 1
    out = df[["asset_id"]].copy()
    out["rank_mean"] = rank_matrix.mean(axis=0)
    out["rank_std"] = rank_matrix.std(axis=0)
    out["top10_frequency"] = (rank_matrix <= 10).mean(axis=0)
    return out.sort_values("rank_std", ascending=False)
