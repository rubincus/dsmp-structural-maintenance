"""Expanded DSMP validation experiments from the Bayesian bridge design protocol."""
from __future__ import annotations
import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .bayesian import gamma_poisson_update
from .config import DSMPConfig
from .epistemic import data_quality_penalty, model_discrepancy_penalty
from .graph_augmented import cascade_susceptibility, graph_centrality
from .normalization import add_reference_and_portfolio_normalizations, reference_normalize
from .scoring import assign_threshold_actions, compute_scores
from .validation_metrics import (
    brier_score,
    calibration_table,
    cross_fitted_recalibration,
    expected_calibration_error,
    holm_bonferroni_adjust,
    nondominated_mask,
    paired_wilcoxon_summary,
    rank_reversal_frequency,
    ranking_metrics,
    summarize_metric,
)
from .voi_gate import apply_voi_gate, approximate_evsi, best_terminal_action


STRUCTURAL_CLASSES = [
    "Primary frame",
    "Secondary frame",
    "Support node",
    "Connection cluster",
    "Foundation interface",
    "Service access",
]

FAILURE_MODES = [
    "corrosion",
    "fatigue",
    "cracking",
    "connection loss",
    "settlement",
    "impact damage",
]

DISPLAY_METHOD_LABELS = {
    "proposed_full": "DSMP + VoI, uncalibrated",
    "proposed_recalibrated": "DSMP + VoI, Platt-recalibrated",
    "proposed_true_probability": "DSMP + VoI with true $P_f$",
    "dsmp_direct": "DSMP direct, uncalibrated",
    "dsmp_direct_recalibrated": "DSMP direct, Platt-recalibrated",
    "dsmp_direct_true_probability": "DSMP direct with true $P_f$",
    "no_voi": "No VoI gate",
    "conditional_rto": "Conditional recovery",
    "risk_only": "Probability--consequence",
    "severity_only": "Severity-only",
    "consequence_only": "Consequence-only",
    "static_rcm": "Static RCM-like",
    "voi_only": "VoI-only inspection",
    "greedy_cost_risk": "Greedy cost--risk",
    "robust_topsis": "Robust TOPSIS",
    "true_probability_upper_bound": "True-probability upper bound",
    "no_rto": "No recovery term",
    "no_uncertainty": "No uncertainty term",
    "no_graph": "No graph adjustment",
    "minmax_norm": "Portfolio min--max normalization",
}


def display_method_label(method: str) -> str:
    return DISPLAY_METHOD_LABELS.get(method, method.replace("_", " "))

CLASS_PARAMS = {
    "Primary frame": {
        "delta": 0.024,
        "lambda0": 0.045,
        "safety": 0.86,
        "production": 0.78,
        "environment": 0.34,
        "demand": 0.72,
        "r0": 38.0,
        "r1": 42.0,
        "r2": 16.0,
        "r3": 22.0,
        "criticality": 0.92,
    },
    "Secondary frame": {
        "delta": 0.019,
        "lambda0": 0.035,
        "safety": 0.60,
        "production": 0.58,
        "environment": 0.24,
        "demand": 0.62,
        "r0": 26.0,
        "r1": 30.0,
        "r2": 9.0,
        "r3": 13.0,
        "criticality": 0.58,
    },
    "Support node": {
        "delta": 0.027,
        "lambda0": 0.052,
        "safety": 0.82,
        "production": 0.84,
        "environment": 0.40,
        "demand": 0.75,
        "r0": 44.0,
        "r1": 46.0,
        "r2": 20.0,
        "r3": 28.0,
        "criticality": 0.88,
    },
    "Connection cluster": {
        "delta": 0.032,
        "lambda0": 0.060,
        "safety": 0.70,
        "production": 0.68,
        "environment": 0.30,
        "demand": 0.68,
        "r0": 30.0,
        "r1": 36.0,
        "r2": 13.0,
        "r3": 18.0,
        "criticality": 0.70,
    },
    "Foundation interface": {
        "delta": 0.017,
        "lambda0": 0.030,
        "safety": 0.92,
        "production": 0.88,
        "environment": 0.52,
        "demand": 0.70,
        "r0": 56.0,
        "r1": 58.0,
        "r2": 24.0,
        "r3": 34.0,
        "criticality": 0.96,
    },
    "Service access": {
        "delta": 0.014,
        "lambda0": 0.026,
        "safety": 0.42,
        "production": 0.34,
        "environment": 0.18,
        "demand": 0.54,
        "r0": 16.0,
        "r1": 24.0,
        "r2": 7.0,
        "r3": 9.0,
        "criticality": 0.36,
    },
}

MODE_PARAMS = {
    "corrosion": {"kappa": 0.020, "phi": 0.004, "a": 1.15, "b": 0.16, "d": 0.55},
    "fatigue": {"kappa": 0.014, "phi": 0.013, "a": 1.45, "b": 0.34, "d": 0.25},
    "cracking": {"kappa": 0.012, "phi": 0.010, "a": 1.30, "b": 0.26, "d": 0.20},
    "connection loss": {"kappa": 0.010, "phi": 0.008, "a": 1.20, "b": 0.22, "d": 0.18},
    "settlement": {"kappa": 0.018, "phi": 0.003, "a": 1.05, "b": 0.12, "d": 0.46},
    "impact damage": {"kappa": 0.008, "phi": 0.015, "a": 1.55, "b": 0.40, "d": 0.12},
}

ACTION_EFFECTS = {
    "defer": {"cost": 0.00, "crew": 0.00, "beta_x": 0.00, "beta_u": 0.00, "beta_rto": 0.00},
    "inspect": {"cost": 0.05, "crew": 0.25, "beta_x": 0.00, "beta_u": 0.35, "beta_rto": 0.00},
    "monitor": {"cost": 0.08, "crew": 0.20, "beta_x": 0.00, "beta_u": 0.45, "beta_rto": 0.00},
    "repair": {"cost": 0.30, "crew": 0.60, "beta_x": 0.35, "beta_u": 0.10, "beta_rto": 0.15},
    "reinforce": {"cost": 0.60, "crew": 0.90, "beta_x": 0.55, "beta_u": 0.15, "beta_rto": 0.35},
    "replace": {"cost": 1.00, "crew": 1.20, "beta_x": 0.90, "beta_u": 0.25, "beta_rto": 0.60},
}

DIRECT_INTERVENTIONS = ("repair", "reinforce", "replace")
NSGA2_ACTIONS = ("defer", "inspect", "monitor", "repair", "reinforce", "replace")


@dataclass(frozen=True)
class ValidationConfig:
    seed: int = 42
    n_assets: int = 300
    horizon_years: int = 10
    mc_paths: int = 256
    n_seeds: int = 50
    budget_grid: tuple[float, ...] = (8.0, 12.0, 16.0, 20.0)
    capacity_per_budget: float = 1.45
    top_k: int = 30
    prior_alpha: float = 2.0
    prior_beta: float = 4.0
    t_ref_recency: float = 4.0
    quality_weights: tuple[float, float, float, float] = (0.30, 0.25, 0.25, 0.20)
    dsmp_weights: tuple[float, float, float] = (0.55, 0.25, 0.20)
    consequence_weights: tuple[float, float, float] = (0.45, 0.40, 0.15)
    uncertainty_weights: tuple[float, float, float] = (0.40, 0.35, 0.25)
    graph_gamma: float = 0.40
    graph_zeta: float = 0.35
    graph_zeta_r: float = 0.35
    # Revision-stage robustness parameter: multiplies the initial demand of every asset.
    # The default of 1.0 reproduces the reported protocol exactly; values below 1.0
    # generate lower-hazard regimes (Section 6.6 of the revised manuscript).
    demand_scale: float = 1.0
    # Multiplies the yearly degradation increments (history and Monte Carlo step); 1.0 = reported protocol.
    degradation_scale: float = 1.0


def _beta_from_mean(mean: float, concentration: float, rng: np.random.Generator) -> float:
    mean = float(np.clip(mean, 0.01, 0.99))
    return float(rng.beta(mean * concentration, (1.0 - mean) * concentration))


def _class_values(classes: np.ndarray, key: str) -> np.ndarray:
    return np.asarray([CLASS_PARAMS[c][key] for c in classes], dtype=float)


def _mode_values(modes: np.ndarray, key: str) -> np.ndarray:
    return np.asarray([MODE_PARAMS[m][key] for m in modes], dtype=float)


def validation_config_to_dsmp(config: ValidationConfig, seed: int) -> DSMPConfig:
    alpha, rho, eta = config.dsmp_weights
    return DSMPConfig(
        seed=seed,
        n_assets=config.n_assets,
        horizon_years=1.0,
        prior_alpha=config.prior_alpha,
        prior_beta=config.prior_beta,
        alpha=alpha,
        rho=rho,
        eta=eta,
        consequence_weights=config.consequence_weights,
        uncertainty_weights=config.uncertainty_weights,
        graph_gamma=config.graph_gamma,
        graph_zeta=config.graph_zeta,
        graph_zeta_r=config.graph_zeta_r,
    )


def robust_topsis_score(df: pd.DataFrame) -> pd.Series:
    """Robust MCDA baseline score averaged over plausible TOPSIS weights."""
    criteria = df[["P_predictive", "C_norm", "RTO_graph", "U_norm"]].to_numpy(dtype=float)
    criteria = np.clip(criteria, 0.0, None)
    norm = np.linalg.norm(criteria, axis=0)
    norm[norm == 0.0] = 1.0
    criteria = criteria / norm
    weight_sets = np.asarray(
        [
            [0.40, 0.30, 0.20, 0.10],
            [0.30, 0.35, 0.25, 0.10],
            [0.30, 0.25, 0.25, 0.20],
            [0.25, 0.30, 0.30, 0.15],
        ],
        dtype=float,
    )
    scores = []
    for weights in weight_sets:
        weights = weights / weights.sum()
        weighted = criteria * weights
        ideal = weighted.max(axis=0)
        anti = weighted.min(axis=0)
        d_ideal = np.linalg.norm(weighted - ideal, axis=1)
        d_anti = np.linalg.norm(weighted - anti, axis=1)
        scores.append(d_anti / (d_ideal + d_anti + 1e-12))
    return pd.Series(np.mean(scores, axis=0), index=df.index).clip(0.0, 1.0)


def add_probability_recalibration(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Add out-of-fold probability recalibration columns for validation."""
    out = df.copy()
    out["P_platt"] = cross_fitted_recalibration(out["P_predictive"], out["F_true"], method="platt", n_splits=5, seed=seed + 1203)
    out["P_isotonic"] = cross_fitted_recalibration(out["P_predictive"], out["F_true"], method="isotonic", n_splits=5, seed=seed + 2407)
    out["P_recalibrated"] = out["P_platt"]
    return out


def build_validation_dependency_graph(asset_ids: list[str], classes: np.ndarray, rng: np.random.Generator) -> nx.DiGraph:
    graph = nx.DiGraph()
    for asset_id, cls in zip(asset_ids, classes):
        graph.add_node(asset_id, structural_class=cls)
    couplings = {
        ("Primary frame", "Support node"),
        ("Support node", "Primary frame"),
        ("Support node", "Connection cluster"),
        ("Connection cluster", "Support node"),
        ("Foundation interface", "Primary frame"),
        ("Primary frame", "Foundation interface"),
    }
    n = len(asset_ids)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            same = classes[i] == classes[j]
            coupled = (classes[i], classes[j]) in couplings
            p = 0.035 if same else (0.018 if coupled else 0.006)
            if rng.random() < p:
                graph.add_edge(asset_ids[i], asset_ids[j], weight=float(rng.uniform(0.10, 1.0)))
    return graph


def generate_ground_truth_portfolio(seed: int, config: ValidationConfig) -> tuple[pd.DataFrame, nx.DiGraph]:
    """Generate Experiment-1 validation data with known latent ground truth."""
    rng = np.random.default_rng(seed)
    n = config.n_assets
    asset_ids = [f"V{seed % 10000:04d}_{i + 1:03d}" for i in range(n)]
    classes = rng.choice(STRUCTURAL_CLASSES, size=n, p=[0.18, 0.20, 0.17, 0.20, 0.12, 0.13])
    modes = rng.choice(FAILURE_MODES, size=n, p=[0.20, 0.20, 0.22, 0.15, 0.12, 0.11])
    graph = build_validation_dependency_graph(asset_ids, classes, rng)
    pi = graph_centrality(graph)
    sigma = cascade_susceptibility(graph, zeta=config.graph_zeta)
    pi_arr = np.asarray([pi[a] for a in asset_ids], dtype=float)
    sigma_arr = np.asarray([sigma[a] for a in asset_ids], dtype=float)

    x = rng.beta(2.0, 8.0, size=n)
    environmental_exposure = rng.uniform(0.0, 1.0, size=n)
    random_effect = rng.normal(0.0, 0.030, size=n)
    xi = np.clip(rng.normal(0.010, 0.004, size=n), 0.001, 0.030)
    total_events = np.zeros(n, dtype=int)
    load = np.ones(n, dtype=float)

    class_delta = _class_values(classes, "delta")
    class_lambda0 = _class_values(classes, "lambda0")
    mode_kappa = _mode_values(modes, "kappa")
    mode_phi = _mode_values(modes, "phi")
    mode_a = _mode_values(modes, "a")
    mode_b = _mode_values(modes, "b")
    mode_d = _mode_values(modes, "d")

    for _ in range(config.horizon_years):
        load = rng.lognormal(mean=0.0, sigma=0.28, size=n)
        process_noise = rng.normal(0.0, 0.018, size=n)
        increment = class_delta + mode_kappa * environmental_exposure + mode_phi * np.log1p(load) + random_effect + process_noise
        x = np.clip(x + float(config.degradation_scale) * increment, 0.0, 1.0)
        rate = class_lambda0 * np.exp(mode_a * x + mode_b * load + mode_d * environmental_exposure + 2.5 * random_effect)
        total_events += rng.poisson(np.clip(rate, 0.001, 5.0))

    resistance0 = rng.lognormal(mean=0.0, sigma=0.08, size=n)
    demand0 = resistance0 * np.clip(rng.normal(_class_values(classes, "demand"), 0.06), 0.45, 0.95) * float(config.demand_scale)
    x_mc = np.empty((n, config.mc_paths), dtype=float)
    load_mc = rng.lognormal(mean=0.0, sigma=0.30, size=(n, config.mc_paths))
    for j in range(config.mc_paths):
        noise = rng.normal(0.0, 0.020, size=n)
        x_mc[:, j] = np.clip(
            x + float(config.degradation_scale) * (class_delta + mode_kappa * environmental_exposure + mode_phi * np.log1p(load_mc[:, j]) + random_effect + noise),
            0.0,
            1.0,
        )
    demand_perturbation = rng.normal(0.0, 0.09, size=(n, config.mc_paths))
    resistance = resistance0[:, None] * (1.0 - x_mc) * np.exp(-xi[:, None])
    demand = demand0[:, None] * (1.0 + demand_perturbation)
    p_true = np.clip((resistance <= demand).mean(axis=1), 0.0, 1.0)
    f_true = rng.binomial(1, p_true)

    p_missing = np.clip(rng.beta(2.0, 6.0, size=n) + 0.12 * (classes == "Service access"), 0.0, 0.92)
    p_det = 0.15 + (0.95 - 0.15) / (1.0 + np.exp(-8.0 * (x - 0.45)))
    n_required = np.full(n, config.horizon_years, dtype=float)
    n_available = rng.binomial(config.horizon_years, np.clip(p_det * (1.0 - p_missing), 0.0, 1.0))
    dt_last = np.where(n_available > 0, rng.exponential(scale=1.4, size=n), config.horizon_years + 1.0)
    n_records = np.maximum(n_available + rng.poisson(2.0, size=n) + 1, 1)
    conflict_prob = np.clip(0.04 + 0.22 * p_missing + 0.10 * x, 0.0, 0.60)
    n_conflict = rng.binomial(n_records, conflict_prob)
    q_penalty, q_components = data_quality_penalty(
        n_available,
        n_required,
        dt_last,
        config.t_ref_recency,
        p_missing,
        n_conflict,
        n_records,
        weights=config.quality_weights,
    )

    m_phys = np.clip(x + 0.12 * (demand0 / np.maximum(resistance0, 1e-9)), 0.0, 1.0)
    m_data = np.clip(x + rng.normal(0.0, 0.045 + 0.12 * p_missing, size=n), 0.0, 1.0)
    m_penalty = model_discrepancy_penalty(m_phys, m_data, m_ref=1.0)

    c_safety = np.asarray([_beta_from_mean(CLASS_PARAMS[c]["safety"], 22.0, rng) for c in classes])
    c_operational = np.asarray([_beta_from_mean(CLASS_PARAMS[c]["production"], 22.0, rng) for c in classes])
    c_environmental = np.asarray([_beta_from_mean(CLASS_PARAMS[c]["environment"], 22.0, rng) for c in classes])
    cw = np.asarray(config.consequence_weights, dtype=float)
    cw = cw / cw.sum()
    c_true = np.clip(cw[0] * c_safety + cw[1] * c_operational + cw[2] * c_environmental, 0.0, 1.0)

    rto = (
        _class_values(classes, "r0")
        + _class_values(classes, "r1") * x
        + _class_values(classes, "r2") * pi_arr
        + _class_values(classes, "r3") * sigma_arr
        + rng.lognormal(mean=np.log(4.0), sigma=0.45, size=n)
    )
    rto = np.clip(rto, 2.0, 160.0)
    severity = np.clip(np.round(1.0 + 5.0 * x + rng.normal(0.0, 0.45, size=n)), 1, 6).astype(int)

    raw = pd.DataFrame(
        {
            "asset_id": asset_ids,
            "structural_class": classes,
            "failure_mode": modes,
            "severity": severity,
            "condition_state": np.clip(np.ceil(1.0 + 4.0 * x), 1, 5).astype(int),
            "age_years": rng.uniform(2.0, 35.0, size=n),
            "events": total_events,
            "inspections": n_available,
            "exposure_years": np.full(n, config.horizon_years, dtype=float),
            "C_safety": c_safety,
            "C_operational": c_operational,
            "C_environmental": c_environmental,
            "delta_RTO_hours": rto,
            "data_quality_penalty": q_penalty.to_numpy(dtype=float),
            "model_mismatch": m_penalty.to_numpy(dtype=float),
            "X_current": x,
            "environmental_exposure": environmental_exposure,
            "load_current": load,
            "p_true": p_true,
            "F_true": f_true,
            "C_true": c_true,
            "class_criticality": _class_values(classes, "criticality"),
            "n_required": n_required,
            "n_available": n_available,
            "dt_last": dt_last,
            "p_missing": p_missing,
            "n_conflict": n_conflict,
            "n_records": n_records,
            "m_phys": m_phys,
            "m_data": m_data,
            "graph_centrality": pi_arr,
            "cascade_susceptibility": sigma_arr,
        }
    )
    raw = pd.concat([raw, q_components.reset_index(drop=True)], axis=1)

    dsmp_cfg = validation_config_to_dsmp(config, seed)
    df = gamma_poisson_update(raw, dsmp_cfg)
    df = add_reference_and_portfolio_normalizations(df)
    df = compute_scores(df, dsmp_cfg, use_reference_norm=True)
    df = assign_threshold_actions(df, dsmp_cfg)
    df = apply_voi_gate(df, dsmp_cfg)
    df = add_probability_recalibration(df, seed)

    gamma = config.graph_gamma
    df["C_graph"] = (df["C_norm"] * (1.0 + gamma * df["graph_centrality"]) / (1.0 + gamma)).clip(0.0, 1.0)
    df["RTO_graph"] = (df["RTO_norm"] + config.graph_zeta_r * df["cascade_susceptibility"]).clip(0.0, 1.0)
    alpha, rho, eta = dsmp_cfg.normalized_weights()
    df["static_risk_pred"] = (df["P_predictive"] * df["C_norm"]).clip(0.0, 1.0)
    df["static_risk_recalibrated"] = (df["P_recalibrated"] * df["C_norm"]).clip(0.0, 1.0)
    df["dynamic_score_pred"] = (alpha * df["static_risk_pred"] + rho * df["RTO_norm"] + eta * df["U_norm"]).clip(0.0, 1.0)
    df["dynamic_score_graph_pred"] = (
        alpha * df["P_predictive"] * df["C_graph"] + rho * df["RTO_graph"] + eta * df["U_norm"]
    ).clip(0.0, 1.0)
    df["dynamic_score_graph_recalibrated"] = (
        alpha * df["P_recalibrated"] * df["C_graph"] + rho * df["RTO_graph"] + eta * df["U_norm"]
    ).clip(0.0, 1.0)
    df["dynamic_score_graph_true_probability"] = (
        alpha * df["p_true"] * df["C_graph"] + rho * df["RTO_graph"] + eta * df["U_norm"]
    ).clip(0.0, 1.0)
    df["dynamic_score_conditional_rto_pred"] = (
        alpha * df["static_risk_pred"] + rho * df["P_predictive"] * df["RTO_norm"] + eta * df["U_norm"]
    ).clip(0.0, 1.0)
    df["true_priority"] = df["p_true"] * df["C_true"] + 0.20 * df["p_true"] * df["RTO_norm"]
    df["true_loss_before"] = df["true_priority"]
    df["robust_topsis_score"] = robust_topsis_score(df)
    return df, graph


def _action_from_score(scores: pd.Series) -> pd.Series:
    values = scores.to_numpy(dtype=float)
    actions = np.full(len(values), "defer", dtype=object)
    actions[(values >= 0.18) & (values < 0.35)] = "inspect"
    actions[(values >= 0.35) & (values < 0.55)] = "repair"
    actions[(values >= 0.55) & (values < 0.75)] = "reinforce"
    actions[values >= 0.75] = "replace"
    return pd.Series(actions, index=scores.index)


def _maintenance_action_from_score(scores: pd.Series) -> pd.Series:
    values = scores.to_numpy(dtype=float)
    actions = np.full(len(values), "defer", dtype=object)
    actions[(values >= 0.08) & (values < 0.35)] = "repair"
    actions[(values >= 0.35) & (values < 0.65)] = "reinforce"
    actions[values >= 0.65] = "replace"
    return pd.Series(actions, index=scores.index)


def _voi_actions_with_probability(df: pd.DataFrame, prob_col: str) -> pd.Series:
    actions = []
    gate = DSMPConfig().uncertainty_gate
    for _, row in df.iterrows():
        best_int_action, _ = best_terminal_action(float(row[prob_col]), float(row["C_norm"]))
        nvi_inspect = approximate_evsi(row, "inspect")
        nvi_monitor = approximate_evsi(row, "monitor")
        if float(row["U_norm"]) >= gate and max(nvi_inspect, nvi_monitor) > 0.0:
            action = "inspect" if nvi_inspect >= nvi_monitor else "monitor"
        else:
            action = best_int_action
        actions.append("reinforce" if action == "reinforce_or_replace" else action)
    return pd.Series(actions, index=df.index)


def _effects_frame(actions: pd.Series) -> pd.DataFrame:
    rows = []
    for action in actions:
        row = ACTION_EFFECTS[str(action)].copy()
        row["action"] = str(action)
        rows.append(row)
    return pd.DataFrame(rows)


def candidate_table(df: pd.DataFrame, method: str) -> pd.DataFrame:
    out = df[["asset_id"]].copy()
    if method == "proposed_full":
        out["score"] = df["dynamic_score_graph_pred"]
        out["action"] = _voi_actions_with_probability(df, "P_predictive")
    elif method == "proposed_recalibrated":
        out["score"] = df["dynamic_score_graph_recalibrated"]
        out["action"] = _voi_actions_with_probability(df, "P_recalibrated")
    elif method == "proposed_true_probability":
        out["score"] = df["dynamic_score_graph_true_probability"]
        out["action"] = _voi_actions_with_probability(df, "p_true")
    elif method == "dsmp_direct":
        out["score"] = df["dynamic_score_graph_pred"]
        out["action"] = _maintenance_action_from_score(out["score"])
    elif method == "dsmp_direct_recalibrated":
        out["score"] = df["dynamic_score_graph_recalibrated"]
        out["action"] = _maintenance_action_from_score(out["score"])
    elif method == "dsmp_direct_true_probability":
        out["score"] = df["dynamic_score_graph_true_probability"]
        out["action"] = _maintenance_action_from_score(out["score"])
    elif method == "no_voi":
        out["score"] = df["dynamic_score_graph_pred"]
        out["action"] = _maintenance_action_from_score(out["score"])
    elif method == "conditional_rto":
        out["score"] = df["dynamic_score_conditional_rto_pred"]
        out["action"] = _maintenance_action_from_score(out["score"])
    elif method == "risk_only":
        out["score"] = df["static_risk_pred"]
        out["action"] = _maintenance_action_from_score(out["score"])
    elif method == "severity_only":
        out["score"] = df["severity"] / 6.0
        out["action"] = _maintenance_action_from_score(out["score"])
    elif method == "consequence_only":
        out["score"] = df["C_norm"]
        out["action"] = _maintenance_action_from_score(out["score"])
    elif method == "static_rcm":
        out["score"] = (0.62 * df["severity"] / 6.0 + 0.38 * df["class_criticality"]).clip(0.0, 1.0)
        out["action"] = _maintenance_action_from_score(out["score"])
    elif method == "voi_only":
        out["score"] = (df["U_norm"] * df["C_norm"]).clip(0.0, 1.0)
        out["action"] = np.where(df["U_norm"] >= 0.62, "inspect", "monitor")
    elif method == "no_rto":
        out["score"] = (0.69 * df["static_risk_pred"] + 0.31 * df["U_norm"]).clip(0.0, 1.0)
        out["action"] = _maintenance_action_from_score(out["score"])
    elif method == "no_uncertainty":
        out["score"] = (0.69 * df["static_risk_pred"] + 0.31 * df["RTO_graph"]).clip(0.0, 1.0)
        out["action"] = _maintenance_action_from_score(out["score"])
    elif method == "no_graph":
        out["score"] = df["dynamic_score_pred"]
        out["action"] = _maintenance_action_from_score(out["score"])
    elif method == "minmax_norm":
        tmp = df.copy()
        tmp["P_var_norm"] = (tmp["P_var_delta"] - tmp["P_var_delta"].min()) / (tmp["P_var_delta"].max() - tmp["P_var_delta"].min() + 1e-9)
        tmp["RTO_norm"] = tmp["RTO_norm_pf"]
        alpha, rho, eta = DSMPConfig().normalized_weights()
        out["score"] = (alpha * tmp["P_predictive"] * tmp["C_norm"] + rho * tmp["RTO_norm"] + eta * tmp["U_norm"]).clip(0.0, 1.0)
        out["action"] = _maintenance_action_from_score(out["score"])
    elif method == "robust_topsis":
        out["score"] = df["robust_topsis_score"]
        out["action"] = _maintenance_action_from_score(out["score"])
    elif method == "true_probability_upper_bound":
        out["score"] = df["true_priority"].clip(0.0, 1.0)
        out["action"] = _maintenance_action_from_score(out["score"])
    else:
        raise ValueError(f"Unknown validation method: {method}")
    effects = _effects_frame(out["action"])
    out = pd.concat([out.reset_index(drop=True), effects.drop(columns=["action"])], axis=1)
    out["estimated_benefit"] = out["score"] * (0.55 * out["beta_x"] + 0.25 * out["beta_rto"] + 0.20 * out["beta_u"])
    return out


def all_action_table(
    df: pd.DataFrame,
    score_col: str = "dynamic_score_graph_pred",
    true_benefit: bool = False,
    allowed_actions: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    rows = []
    for idx, row in df.reset_index(drop=True).iterrows():
        for action, effect in ACTION_EFFECTS.items():
            if action == "defer":
                continue
            if allowed_actions is not None and action not in allowed_actions:
                continue
            benefit = true_action_benefit(row, action) if true_benefit else float(row[score_col]) * (
                0.55 * effect["beta_x"] + 0.25 * effect["beta_rto"] + 0.20 * effect["beta_u"]
            )
            rows.append(
                {
                    "asset_id": row["asset_id"],
                    "row_idx": idx,
                    "action": action,
                    "score": float(row[score_col]),
                    "estimated_benefit": float(benefit),
                    **effect,
                }
            )
    return pd.DataFrame(rows)


def select_plan(candidates: pd.DataFrame, budget: float, capacity: float, ratio: bool = False) -> pd.DataFrame:
    table = candidates.copy()
    table = table[table["action"] != "defer"].copy()
    table["rank_value"] = table["estimated_benefit"] / (table["cost"] + 1e-9) if ratio else table["score"]
    selected = []
    used_assets = set()
    used_budget = 0.0
    used_capacity = 0.0
    for _, row in table.sort_values("rank_value", ascending=False).iterrows():
        if row["asset_id"] in used_assets:
            continue
        if used_budget + float(row["cost"]) <= budget and used_capacity + float(row["crew"]) <= capacity:
            selected.append(row.to_dict())
            used_assets.add(row["asset_id"])
            used_budget += float(row["cost"])
            used_capacity += float(row["crew"])
    return pd.DataFrame(selected)


def true_action_benefit(row: pd.Series, action: str) -> float:
    effect = ACTION_EFFECTS[action]
    p_before = float(row["p_true"])
    c = float(row["C_true"])
    rto = float(row["RTO_norm"])
    before = p_before * c + 0.20 * p_before * rto
    p_after = p_before * np.exp(-2.35 * effect["beta_x"])
    rto_after = rto * (1.0 - effect["beta_rto"])
    after = p_after * c + 0.20 * p_after * rto_after
    return float(max(before - after, 0.0))


def oracle_plan(df: pd.DataFrame, budget: float, capacity: float) -> pd.DataFrame:
    return select_plan(all_action_table(df, true_benefit=True, allowed_actions=DIRECT_INTERVENTIONS), budget, capacity, ratio=True)


def plan_for_method(df: pd.DataFrame, method: str, budget: float, capacity: float) -> pd.DataFrame:
    ratio_intervention_scores = {
        "dsmp_direct": "dynamic_score_graph_pred",
        "dsmp_direct_recalibrated": "dynamic_score_graph_recalibrated",
        "dsmp_direct_true_probability": "dynamic_score_graph_true_probability",
        "greedy_cost_risk": "static_risk_pred",
        "robust_topsis": "robust_topsis_score",
        "true_probability_upper_bound": "true_priority",
    }
    if method in ratio_intervention_scores:
        return select_plan(
            all_action_table(df, score_col=ratio_intervention_scores[method], allowed_actions=DIRECT_INTERVENTIONS),
            budget,
            capacity,
            ratio=True,
        )
    return select_plan(candidate_table(df, method), budget, capacity, ratio=True)


def evaluate_plan(df: pd.DataFrame, plan: pd.DataFrame) -> dict[str, float]:
    effects = {row.asset_id: row for row in plan.itertuples(index=False)} if len(plan) else {}
    p_after = df["p_true"].to_numpy(dtype=float).copy()
    rto_after = df["RTO_norm"].to_numpy(dtype=float).copy()
    u_after = df["U_norm"].to_numpy(dtype=float).copy()
    rto_hours_after = df["delta_RTO_hours"].to_numpy(dtype=float).copy()
    for i, asset_id in enumerate(df["asset_id"]):
        row = effects.get(asset_id)
        if row is None:
            continue
        effect = ACTION_EFFECTS[row.action]
        p_after[i] *= np.exp(-2.35 * effect["beta_x"])
        rto_after[i] *= 1.0 - effect["beta_rto"]
        rto_hours_after[i] *= 1.0 - effect["beta_rto"]
        u_after[i] *= 1.0 - effect["beta_u"]
    c = df["C_true"].to_numpy(dtype=float)
    before = df["true_loss_before"].to_numpy(dtype=float)
    after = p_after * c + 0.20 * p_after * rto_after
    cost = float(plan["cost"].sum()) if len(plan) else 0.0
    crew = float(plan["crew"].sum()) if len(plan) else 0.0
    downtime_before = float((df["p_true"] * df["delta_RTO_hours"]).sum())
    downtime_after = float(np.sum(p_after * rto_hours_after))
    return {
        "loss_before": float(before.sum()),
        "loss_after": float(after.sum()),
        "risk_reduction": float(before.sum() - after.sum()),
        "rrpc": float((before.sum() - after.sum()) / (cost + 1e-9)),
        "cost": cost,
        "crew": crew,
        "n_actions": int(len(plan)),
        "downtime_reduction": downtime_before - downtime_after,
        "residual_recovery_exposure": float(rto_after.sum()),
        "residual_uncertainty": float(u_after.sum()),
    }


def run_budget_metrics(df: pd.DataFrame, budget: float, capacity: float) -> list[dict[str, float]]:
    methods = [
        "proposed_full",
        "proposed_recalibrated",
        "proposed_true_probability",
        "dsmp_direct",
        "dsmp_direct_recalibrated",
        "dsmp_direct_true_probability",
        "no_voi",
        "conditional_rto",
        "risk_only",
        "severity_only",
        "consequence_only",
        "static_rcm",
        "robust_topsis",
        "true_probability_upper_bound",
        "voi_only",
        "greedy_cost_risk",
    ]
    oracle_eval = evaluate_plan(df, oracle_plan(df, budget, capacity))
    records = []
    for method in methods:
        plan = plan_for_method(df, method, budget, capacity)
        metrics = evaluate_plan(df, plan)
        metrics.update(
            {
                "method": method,
                "budget": budget,
                "capacity": capacity,
                "regret": metrics["loss_after"] - oracle_eval["loss_after"],
                "oracle_loss_after": oracle_eval["loss_after"],
            }
        )
        records.append(metrics)
    return records


def score_for_method(df: pd.DataFrame, method: str) -> pd.Series:
    if method == "proposed_full":
        return df["dynamic_score_graph_pred"]
    if method == "proposed_recalibrated":
        return df["dynamic_score_graph_recalibrated"]
    if method == "proposed_true_probability":
        return df["dynamic_score_graph_true_probability"]
    if method == "dsmp_direct":
        return df["dynamic_score_graph_pred"]
    if method == "dsmp_direct_recalibrated":
        return df["dynamic_score_graph_recalibrated"]
    if method == "dsmp_direct_true_probability":
        return df["dynamic_score_graph_true_probability"]
    if method == "no_voi":
        return df["dynamic_score_graph_pred"]
    if method == "conditional_rto":
        return df["dynamic_score_conditional_rto_pred"]
    if method == "risk_only":
        return df["static_risk_pred"]
    if method == "severity_only":
        return df["severity"] / 6.0
    if method == "consequence_only":
        return df["C_norm"]
    if method == "static_rcm":
        return (0.62 * df["severity"] / 6.0 + 0.38 * df["class_criticality"]).clip(0.0, 1.0)
    if method == "voi_only":
        return (df["U_norm"] * df["C_norm"]).clip(0.0, 1.0)
    if method == "greedy_cost_risk":
        return df["static_risk_pred"]  # ranking metric of the score that this policy actually uses (revision fix)
    if method == "robust_topsis":
        return df["robust_topsis_score"]
    if method == "true_probability_upper_bound":
        return df["true_priority"]
    raise ValueError(method)


def rank_metric_records(df: pd.DataFrame, seed: int, top_k: int) -> list[dict[str, float]]:
    methods = [
        "proposed_full",
        "proposed_recalibrated",
        "proposed_true_probability",
        "dsmp_direct",
        "dsmp_direct_recalibrated",
        "dsmp_direct_true_probability",
        "conditional_rto",
        "risk_only",
        "severity_only",
        "consequence_only",
        "static_rcm",
        "robust_topsis",
        "true_probability_upper_bound",
        "voi_only",
        "greedy_cost_risk",
    ]
    records = []
    for method in methods:
        metrics = ranking_metrics(score_for_method(df, method), df["true_priority"], k=top_k)
        records.append({"seed": seed, "method": method, **metrics})
    return records


def pareto_random_front(df: pd.DataFrame, budget: float, capacity: float, seed: int, n_samples: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    action_options = all_action_table(df, true_benefit=False)
    by_asset = {a: g.reset_index(drop=True) for a, g in action_options.groupby("asset_id")}
    asset_ids = df["asset_id"].to_numpy()
    records = []
    for sample in range(n_samples):
        chosen = []
        cost = 0.0
        crew = 0.0
        for asset_id in rng.permutation(asset_ids):
            if rng.random() > 0.30:
                continue
            options = by_asset[asset_id]
            opt = options.iloc[int(rng.integers(0, len(options)))]
            if cost + float(opt.cost) <= budget and crew + float(opt.crew) <= capacity:
                chosen.append(opt.to_dict())
                cost += float(opt.cost)
                crew += float(opt.crew)
        plan = pd.DataFrame(chosen)
        ev = evaluate_plan(df, plan)
        records.append(
            {
                "sample": sample,
                "residual_risk": ev["loss_after"],
                "cost": ev["cost"],
                "recovery_exposure": ev["residual_recovery_exposure"],
                "residual_uncertainty": ev["residual_uncertainty"],
                "feasible": ev["cost"] <= budget + 1e-9 and ev["crew"] <= capacity + 1e-9,
            }
        )
    out = pd.DataFrame(records)
    mask = nondominated_mask(out[["residual_risk", "cost", "recovery_exposure", "residual_uncertainty"]].to_numpy())
    return out.loc[mask].reset_index(drop=True)


def _plan_from_action_codes(df: pd.DataFrame, action_codes: np.ndarray) -> pd.DataFrame:
    rows = []
    codes = np.clip(np.rint(action_codes).astype(int), 0, len(NSGA2_ACTIONS) - 1)
    for i, code in enumerate(codes):
        action = NSGA2_ACTIONS[int(code)]
        if action == "defer":
            continue
        effect = ACTION_EFFECTS[action]
        rows.append(
            {
                "asset_id": df["asset_id"].iloc[i],
                "row_idx": i,
                "action": action,
                "score": float(df["dynamic_score_graph_recalibrated"].iloc[i]),
                "estimated_benefit": float(df["dynamic_score_graph_recalibrated"].iloc[i])
                * (0.55 * effect["beta_x"] + 0.25 * effect["beta_rto"] + 0.20 * effect["beta_u"]),
                **effect,
            }
        )
    return pd.DataFrame(rows)


def pareto_nsga2_front(
    df: pd.DataFrame,
    budget: float,
    capacity: float,
    seed: int,
    pop_size: int = 96,
    n_gen: int = 80,
) -> pd.DataFrame:
    """Compute a constrained multiobjective maintenance front using pymoo NSGA-II."""
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem
        from pymoo.core.repair import Repair
        from pymoo.core.sampling import Sampling
        from pymoo.optimize import minimize
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
    except ImportError as exc:
        raise RuntimeError("pymoo>=0.6.1 is required for the NSGA-II Pareto experiment.") from exc

    action_effects = pd.DataFrame([ACTION_EFFECTS[action] for action in NSGA2_ACTIONS])
    cost_lut = action_effects["cost"].to_numpy(dtype=float)
    crew_lut = action_effects["crew"].to_numpy(dtype=float)
    beta_x_lut = action_effects["beta_x"].to_numpy(dtype=float)
    beta_u_lut = action_effects["beta_u"].to_numpy(dtype=float)
    beta_rto_lut = action_effects["beta_rto"].to_numpy(dtype=float)
    p_true = df["p_true"].to_numpy(dtype=float)
    c_true = df["C_true"].to_numpy(dtype=float)
    rto_norm = df["RTO_norm"].to_numpy(dtype=float)
    u_norm = df["U_norm"].to_numpy(dtype=float)
    score = df["dynamic_score_graph_recalibrated"].to_numpy(dtype=float)
    action_benefit_lut = 0.55 * beta_x_lut + 0.25 * beta_rto_lut + 0.20 * beta_u_lut

    def _codes_from_plan(plan: pd.DataFrame) -> np.ndarray:
        codes = np.zeros(len(df), dtype=int)
        if plan.empty:
            return codes
        asset_pos = {asset_id: i for i, asset_id in enumerate(df["asset_id"])}
        action_pos = {action: i for i, action in enumerate(NSGA2_ACTIONS)}
        for row in plan.itertuples(index=False):
            idx = asset_pos.get(row.asset_id)
            if idx is not None:
                codes[idx] = action_pos[str(row.action)]
        return codes

    heuristic_seeds = [
        np.zeros(len(df), dtype=int),
        _codes_from_plan(plan_for_method(df, "proposed_full", budget, capacity)),
        _codes_from_plan(plan_for_method(df, "proposed_recalibrated", budget, capacity)),
        _codes_from_plan(plan_for_method(df, "dsmp_direct_recalibrated", budget, capacity)),
        _codes_from_plan(plan_for_method(df, "greedy_cost_risk", budget, capacity)),
        _codes_from_plan(plan_for_method(df, "robust_topsis", budget, capacity)),
    ]

    def _repair_matrix(x: np.ndarray) -> np.ndarray:
        repaired = np.clip(np.rint(x).astype(int), 0, len(NSGA2_ACTIONS) - 1)
        for row in repaired:
            cost = float(cost_lut[row].sum())
            crew = float(crew_lut[row].sum())
            while (cost > budget + 1e-9 or crew > capacity + 1e-9) and np.any(row > 0):
                active = np.flatnonzero(row > 0)
                benefit = score[active] * action_benefit_lut[row[active]]
                resource = cost_lut[row[active]] + crew_lut[row[active]] / max(capacity, 1e-9)
                remove_pos = active[int(np.argmin(benefit / (resource + 1e-12)))]
                cost -= float(cost_lut[row[remove_pos]])
                crew -= float(crew_lut[row[remove_pos]])
                row[remove_pos] = 0
        return repaired

    class SparseMaintenanceSampling(Sampling):
        def _do(self, problem: Problem, n_samples: int, **kwargs) -> np.ndarray:
            rng = np.random.default_rng(seed + 791)
            x = np.zeros((n_samples, len(df)), dtype=int)
            for i, codes in enumerate(heuristic_seeds[:n_samples]):
                x[i] = codes
            action_codes = np.arange(1, len(NSGA2_ACTIONS))
            action_prob = np.asarray([0.13, 0.13, 0.36, 0.24, 0.14], dtype=float)
            action_prob = action_prob / action_prob.sum()
            for sample in range(len(heuristic_seeds), n_samples):
                used_cost = 0.0
                used_crew = 0.0
                for idx in rng.permutation(len(df)):
                    if rng.random() > 0.34:
                        continue
                    code = int(rng.choice(action_codes, p=action_prob))
                    if used_cost + cost_lut[code] <= budget and used_crew + crew_lut[code] <= capacity:
                        x[sample, idx] = code
                        used_cost += float(cost_lut[code])
                        used_crew += float(crew_lut[code])
            return x

    class BudgetCapacityRepair(Repair):
        def _do(self, problem: Problem, x: np.ndarray, **kwargs) -> np.ndarray:
            return _repair_matrix(x)

    class MaintenanceParetoProblem(Problem):
        def __init__(self) -> None:
            super().__init__(
                n_var=len(df),
                n_obj=4,
                n_ieq_constr=2,
                xl=np.zeros(len(df)),
                xu=np.full(len(df), len(NSGA2_ACTIONS) - 1),
                vtype=int,
            )

        def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
            codes = np.clip(np.rint(x).astype(int), 0, len(NSGA2_ACTIONS) - 1)
            cost = cost_lut[codes].sum(axis=1)
            crew = crew_lut[codes].sum(axis=1)
            beta_x = beta_x_lut[codes]
            beta_u = beta_u_lut[codes]
            beta_rto = beta_rto_lut[codes]
            p_after = p_true[None, :] * np.exp(-2.35 * beta_x)
            rto_after = rto_norm[None, :] * (1.0 - beta_rto)
            u_after = u_norm[None, :] * (1.0 - beta_u)
            loss_after = (p_after * c_true[None, :] + 0.20 * p_after * rto_after).sum(axis=1)
            recovery_exposure = rto_after.sum(axis=1)
            residual_uncertainty = u_after.sum(axis=1)
            out["F"] = np.column_stack([loss_after, cost, recovery_exposure, residual_uncertainty])
            out["G"] = np.column_stack([cost - budget, crew - capacity])

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=SparseMaintenanceSampling(),
        crossover=SBX(prob=0.90, eta=15, vtype=float, repair=BudgetCapacityRepair()),
        mutation=PM(eta=20, vtype=float, repair=BudgetCapacityRepair()),
        repair=BudgetCapacityRepair(),
        eliminate_duplicates=True,
    )
    result = minimize(MaintenanceParetoProblem(), algorithm, ("n_gen", n_gen), seed=seed, verbose=False)
    if result.X is None:
        return pd.DataFrame(
            columns=["solution", "residual_risk", "cost", "crew", "recovery_exposure", "residual_uncertainty", "feasible", "n_actions"]
        )

    records = []
    for solution, x in enumerate(np.atleast_2d(result.X)):
        plan = _plan_from_action_codes(df, x)
        ev = evaluate_plan(df, plan)
        records.append(
            {
                "solution": solution,
                "residual_risk": ev["loss_after"],
                "cost": ev["cost"],
                "crew": ev["crew"],
                "recovery_exposure": ev["residual_recovery_exposure"],
                "residual_uncertainty": ev["residual_uncertainty"],
                "feasible": ev["cost"] <= budget + 1e-9 and ev["crew"] <= capacity + 1e-9,
                "n_actions": ev["n_actions"],
            }
        )
    out = pd.DataFrame(records)
    out = out[out["feasible"]].copy()
    if out.empty:
        return out.reset_index(drop=True)
    out = out.drop_duplicates(subset=["residual_risk", "cost", "recovery_exposure", "residual_uncertainty"])
    mask = nondominated_mask(out[["residual_risk", "cost", "recovery_exposure", "residual_uncertainty"]].to_numpy(dtype=float))
    return out.loc[mask].sort_values(["cost", "residual_risk"]).reset_index(drop=True)


def normalization_stress(df: pd.DataFrame, config: ValidationConfig) -> pd.DataFrame:
    base = df.copy()
    base["P_var_norm"] = reference_normalize(base["P_var_delta"], 0.0, 0.030)
    dsmp_cfg = validation_config_to_dsmp(config, config.seed)
    base_ref = compute_scores(base, dsmp_cfg, use_reference_norm=True)["dynamic_score"]
    base_pf = compute_scores(add_reference_and_portfolio_normalizations(df), dsmp_cfg, use_reference_norm=False)["dynamic_score"]
    extremes = df.head(8).copy()
    extremes["asset_id"] = [f"EXT{i:02d}" for i in range(len(extremes))]
    extremes["delta_RTO_hours"] = np.linspace(180.0, 420.0, len(extremes))
    extremes["P_var_delta"] = np.linspace(df["P_var_delta"].max() * 2.0, df["P_var_delta"].max() * 8.0, len(extremes))
    extremes["C_safety"] = 1.0
    extremes["C_operational"] = 1.0
    extremes["C_environmental"] = 1.0
    extended = pd.concat([df, extremes], ignore_index=True)
    ext_pf_norm = add_reference_and_portfolio_normalizations(extended)
    ext_pf = compute_scores(ext_pf_norm, dsmp_cfg, use_reference_norm=False).iloc[: len(df)]["dynamic_score"]
    ext_ref = ext_pf_norm.copy()
    ext_ref["P_var_norm"] = reference_normalize(ext_ref["P_var_delta"], 0.0, 0.030)
    ext_ref = compute_scores(ext_ref, dsmp_cfg, use_reference_norm=True).iloc[: len(df)]["dynamic_score"]
    return pd.DataFrame(
        [
            {"normalization": "reference_based", "rank_reversal_frequency": rank_reversal_frequency(base_ref, ext_ref)},
            {"normalization": "portfolio_minmax", "rank_reversal_frequency": rank_reversal_frequency(base_pf, ext_pf)},
        ]
    )


def graph_perturbation_stress(df: pd.DataFrame, graph: nx.DiGraph, config: ValidationConfig, seed: int, levels=(0.05, 0.15, 0.30)) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 808)
    base_pi = pd.Series({n: v for n, v in zip(df["asset_id"], df["graph_centrality"])})
    base_sigma = pd.Series({n: v for n, v in zip(df["asset_id"], df["cascade_susceptibility"])})
    records = []
    for level in levels:
        g2 = graph.copy()
        for u, v, data in g2.edges(data=True):
            data["weight"] = float(np.clip(data.get("weight", 1.0) + rng.normal(0.0, level), 0.01, 1.50))
        pi = pd.Series(graph_centrality(g2))
        sigma = pd.Series(cascade_susceptibility(g2, zeta=config.graph_zeta))
        pi = pi.reindex(base_pi.index).fillna(0.0)
        sigma = sigma.reindex(base_sigma.index).fillna(0.0)
        score_shift = 0.10 * (pi - base_pi).abs() + 0.15 * (sigma - base_sigma).abs()
        records.append(
            {
                "edge_weight_sigma": level,
                "mean_abs_centrality_change": float((pi - base_pi).abs().mean()),
                "mean_abs_cascade_change": float((sigma - base_sigma).abs().mean()),
                "graph_sensitive_assets": int((score_shift > score_shift.quantile(0.90)).sum()),
            }
        )
    return pd.DataFrame(records)


def voi_stress_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base_u = df["U_norm"].to_numpy(dtype=float)
    base_c = df["C_norm"].to_numpy(dtype=float)
    for p_missing in [0.1, 0.3, 0.5, 0.7]:
        for false_negative in [0.05, 0.15, 0.30]:
            for kappa in [0.01, 0.05, 0.10, 0.20]:
                u = np.clip(base_u + 0.35 * p_missing + 0.20 * false_negative, 0.0, 1.0)
                evsi_inspect = 0.55 * u * base_c * 0.25
                evsi_monitor = 0.35 * u * base_c * 0.25
                nvi = np.maximum(evsi_inspect - kappa, evsi_monitor - 0.8 * kappa)
                route_info = (u >= 0.62) & (nvi > 0.0)
                rows.append(
                    {
                        "p_missing": p_missing,
                        "false_negative": false_negative,
                        "info_cost": kappa,
                        "info_route_fraction": float(route_info.mean()),
                        "mean_nvi": float(nvi.mean()),
                        "posterior_uncertainty_reduction": float((route_info * 0.40 * u).mean()),
                    }
                )
    return pd.DataFrame(rows)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_workflow(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.axis("off")
    labels = ["Ground truth\nsimulator", "Bayesian\nupdate", "DSMP\nscoring", "Action\nselection", "Independent\nevaluation"]
    xs = np.linspace(0.08, 0.92, len(labels))
    for x, label in zip(xs, labels):
        ax.text(x, 0.55, label, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.35", fc="#e8f1fb", ec="#224466"))
    for x1, x2 in zip(xs[:-1], xs[1:]):
        ax.annotate("", xy=(x2 - 0.075, 0.55), xytext=(x1 + 0.075, 0.55), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.set_title("Experiment 1 validation workflow")
    _save(fig, path)


def plot_calibration(calibration: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", lw=1)
    if "model" not in calibration.columns:
        calibration = calibration.assign(model="Gamma-Poisson")
    xmax = 0.35
    ymax = 0.35
    for model, group in calibration.groupby("model"):
        g = group.groupby("bin", as_index=False).agg(pred_mean=("pred_mean", "mean"), observed_rate=("observed_rate", "mean"), n=("n", "sum"))
        g = g[g["n"] > 0]
        if g.empty:
            continue
        ax.plot(g["pred_mean"], g["observed_rate"], marker="o", label=model)
        xmax = max(xmax, float(g["pred_mean"].max()) * 1.2)
        ymax = max(ymax, float(g["observed_rate"].max()) * 1.2)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed failure/event frequency")
    ax.set_title("Calibration curve")
    ax.set_xlim(0, min(1.0, xmax))
    ax.set_ylim(0, min(1.0, ymax))
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_regret(metrics: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for method, group in metrics.groupby("method"):
        g = group.groupby("budget", as_index=False)["regret"].median()
        ax.plot(g["budget"], g["regret"], marker="o", label=display_method_label(method))
    ax.set_xlabel("Budget")
    ax.set_ylabel("Median regret vs perfect-information greedy reference")
    ax.set_title("Regret-vs-budget curves")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, path)


def plot_rank_metrics(rank_metrics_df: pd.DataFrame, path: Path) -> None:
    g = rank_metrics_df.groupby("method", as_index=False).agg(top_k_capture=("top_k_capture", "mean"), spearman=("spearman", "mean"))
    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - 0.18, g["top_k_capture"], width=0.36, label="Top-k capture")
    ax.bar(x + 0.18, g["spearman"], width=0.36, label="Spearman")
    ax.set_xticks(x, [display_method_label(method) for method in g["method"]], rotation=35, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Top-k capture and rank-correlation comparison")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_voi_map(path: Path) -> None:
    u = np.linspace(0, 1, 120)
    cost = np.linspace(0.0, 0.25, 120)
    U, C = np.meshgrid(u, cost)
    nvi = 0.55 * U * 0.70 * 0.25 - C
    route = (U >= 0.62) & (nvi > 0.0)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    im = ax.imshow(route.astype(float), origin="lower", aspect="auto", extent=[0, 1, 0, 0.25], cmap="viridis")
    ax.set_xlabel("Epistemic uncertainty")
    ax.set_ylabel("Information acquisition cost")
    ax.set_title("VoI gate decision map")
    fig.colorbar(im, ax=ax, label="Information action selected")
    _save(fig, path)


def plot_network(df: pd.DataFrame, graph: nx.DiGraph, path: Path) -> None:
    top_nodes = df.sort_values("graph_centrality", ascending=False).head(90)["asset_id"].to_list()
    sub = graph.subgraph(top_nodes).copy()
    if sub.number_of_edges() > 0:
        largest = max(nx.weakly_connected_components(sub), key=len)
        sub = sub.subgraph(largest).copy()
    core_nodes = [node for node in sub.nodes if sub.degree(node) >= 3]
    if len(core_nodes) < 35:
        core_nodes = [node for node in sub.nodes if sub.degree(node) >= 2]
    if len(core_nodes) >= 35:
        sub = sub.subgraph(core_nodes).copy()
        largest = max(nx.weakly_connected_components(sub), key=len)
        sub = sub.subgraph(largest).copy()
    sub = nx.DiGraph(sub)
    if sub.number_of_nodes() > 45 and sub.number_of_edges() > 0:
        prelim_pos = nx.spring_layout(sub, seed=11, weight="weight", k=0.42, iterations=120)
        xy = np.asarray(list(prelim_pos.values()), dtype=float)
        center = xy.mean(axis=0)
        dists = np.linalg.norm(xy - center, axis=1)
        nodes = np.asarray(list(prelim_pos.keys()), dtype=object)
        keep = nodes[dists <= np.quantile(dists, 0.90)].tolist()
        if len(keep) >= 35:
            sub = sub.subgraph(keep).copy()
            largest = max(nx.weakly_connected_components(sub), key=len)
            sub = nx.DiGraph(sub.subgraph(largest).copy())
    while sub.number_of_nodes() > 40:
        leaves = [node for node in sub.nodes if sub.degree(node) <= 1]
        if not leaves:
            break
        sub.remove_nodes_from(leaves)
        if sub.number_of_edges() == 0:
            break
        largest = max(nx.weakly_connected_components(sub), key=len)
        sub = nx.DiGraph(sub.subgraph(largest).copy())
    centrality = df.set_index("asset_id")["graph_centrality"].reindex(sub.nodes).fillna(0.0)
    cascade = df.set_index("asset_id")["cascade_susceptibility"].reindex(sub.nodes).fillna(0.0)
    cmin, cmax = float(centrality.min()), float(centrality.max())
    centrality_scaled = (centrality - cmin) / (cmax - cmin + 1e-12)
    sizes = 18.0 + 58.0 * centrality_scaled.to_numpy(dtype=float)
    pos = nx.spring_layout(sub, seed=11, weight="weight", k=0.42, iterations=260)
    edge_widths = [0.20 + 0.55 * float(data.get("weight", 0.5)) for _, _, data in sub.edges(data=True)]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    nx.draw_networkx_edges(
        sub,
        pos,
        width=edge_widths,
        alpha=0.18,
        edge_color="#6f7882",
        arrows=False,
        ax=ax,
    )
    nodes = nx.draw_networkx_nodes(
        sub,
        pos,
        node_color=cascade.to_numpy(dtype=float),
        node_size=sizes,
        cmap="viridis",
        linewidths=0.35,
        edgecolors="white",
        ax=ax,
    )
    ax.axis("off")
    ax.margins(0.08)
    fig.colorbar(nodes, ax=ax, label="Cascade susceptibility", fraction=0.035, pad=0.02)
    _save(fig, path)


def plot_pareto(front: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    sc = ax.scatter(front["cost"], front["residual_risk"], c=front["residual_uncertainty"], s=45, cmap="viridis")
    ax.set_xlabel("Cost")
    ax.set_ylabel("Residual risk")
    ax.set_title("NSGA-II Pareto front")
    fig.colorbar(sc, ax=ax, label="Residual uncertainty")
    _save(fig, path)


def plot_rank_reversal(stress: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.bar(stress["normalization"], stress["rank_reversal_frequency"])
    ax.set_ylim(0, max(0.05, float(stress["rank_reversal_frequency"].max()) * 1.25))
    ax.set_ylabel("Pairwise rank-reversal frequency")
    ax.set_title("Normalization stress test")
    _save(fig, path)


def write_summary_tables(
    out_dir: Path,
    config: ValidationConfig,
    metrics: pd.DataFrame,
    rank_metrics_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    runtime_rows: list[dict[str, float]],
) -> None:
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"parameter": "n_assets", "value": config.n_assets},
            {"parameter": "horizon_years", "value": config.horizon_years},
            {"parameter": "mc_paths", "value": config.mc_paths},
            {"parameter": "n_seeds", "value": config.n_seeds},
            {"parameter": "budget_grid", "value": ";".join(map(str, config.budget_grid))},
            {"parameter": "action_effects", "value": str(ACTION_EFFECTS)},
        ]
    ).to_csv(table_dir / "table_v1_generator_parameters.csv", index=False)
    pd.DataFrame(
        [
            {"method": "proposed_full", "information": "predictive probability, consequence, recovery, uncertainty, graph, VoI"},
            {"method": "proposed_recalibrated", "information": "cross-fitted Platt probability recalibration plus consequence, recovery, uncertainty, graph, VoI"},
            {"method": "proposed_true_probability", "information": "sensitivity run with simulated true failure probability inside the proposed score"},
            {"method": "dsmp_direct", "information": "cost-aware direct intervention using the DSMP score without information actions"},
            {"method": "dsmp_direct_recalibrated", "information": "cost-aware direct intervention using the recalibrated DSMP score"},
            {"method": "dsmp_direct_true_probability", "information": "cost-aware direct intervention using simulated true probability in the DSMP score"},
            {"method": "no_voi", "information": "same score without EVSI/NVI routing"},
            {"method": "conditional_rto", "information": "recovery conditioned on predicted failure"},
            {"method": "risk_only", "information": "Bayesian probability times consequence, intervention actions only"},
            {"method": "severity_only", "information": "condition severity only"},
            {"method": "consequence_only", "information": "consequence only"},
            {"method": "static_rcm", "information": "severity and functional class"},
            {"method": "robust_topsis", "information": "robust TOPSIS-style MCDA over probability, consequence, recovery, and uncertainty"},
            {"method": "nsga2_pareto", "information": "pymoo NSGA-II constrained multiobjective search over residual risk, cost, recovery exposure, and residual uncertainty"},
            {"method": "true_probability_upper_bound", "information": "upper-bound risk policy using simulated true probability and consequence"},
            {"method": "voi_only", "information": "uncertainty-driven inspection"},
            {"method": "greedy_cost_risk", "information": "benefit-cost heuristic, intervention actions only"},
        ]
    ).to_csv(table_dir / "table_v2_baselines.csv", index=False)

    default_budget = config.budget_grid[min(1, len(config.budget_grid) - 1)]
    focus = metrics[np.isclose(metrics["budget"], default_budget)]
    summary_rows = []
    for method, group in focus.groupby("method"):
        rank_group = rank_metrics_df[rank_metrics_df["method"] == method]
        for metric in ["regret", "rrpc", "downtime_reduction"]:
            row = summarize_metric(group[metric], seed=config.seed)
            row.update({"method": method, "metric": metric})
            summary_rows.append(row)
        if not rank_group.empty:
            for metric in ["spearman", "kendall", "top_k_capture"]:
                row = summarize_metric(rank_group[metric], seed=config.seed)
                row.update({"method": method, "metric": metric})
                summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(table_dir / "table_v3_metric_summary.csv", index=False)

    calibration_focus = metrics[metrics["method"].str.startswith("calibration_", na=False)].copy()
    calibration_rows = []
    for method, group in calibration_focus.groupby("method"):
        for metric in ["brier", "ece", "mse_to_p_true", "mae_to_p_true"]:
            row = summarize_metric(group[metric], seed=config.seed)
            row.update({"method": method.replace("calibration_", ""), "metric": metric})
            calibration_rows.append(row)
    pd.DataFrame(calibration_rows).to_csv(table_dir / "table_v6_probability_recalibration.csv", index=False)

    paired_rows = []
    paired_focus = focus.dropna(subset=["regret"])
    reference = "dsmp_direct_recalibrated" if "dsmp_direct_recalibrated" in set(paired_focus["method"]) else "proposed_recalibrated"
    reference_values = paired_focus[paired_focus["method"] == reference][["seed", "regret"]].rename(columns={"regret": "reference_regret"})
    for method in [
        "proposed_full",
        "proposed_recalibrated",
        "dsmp_direct",
        "dsmp_direct_true_probability",
        "risk_only",
        "static_rcm",
        "robust_topsis",
        "greedy_cost_risk",
        "consequence_only",
        "true_probability_upper_bound",
    ]:
        if method == reference:
            continue
        comparator = paired_focus[paired_focus["method"] == method][["seed", "regret"]].rename(columns={"regret": "comparator_regret"})
        joined = reference_values.merge(comparator, on="seed", how="inner")
        row = paired_wilcoxon_summary(joined["reference_regret"], joined["comparator_regret"], seed=config.seed)
        row.update({"reference": reference, "comparator": method, "metric": "regret", "budget": default_budget})
        paired_rows.append(row)
    paired_df = pd.DataFrame(paired_rows)
    if not paired_df.empty:
        paired_df["holm_adjusted_p"] = holm_bonferroni_adjust(paired_df["p_value"].to_numpy(dtype=float))
    paired_df.to_csv(table_dir / "table_v7_paired_regret_tests.csv", index=False)

    ablation_methods = [
        "proposed_full",
        "proposed_recalibrated",
        "proposed_true_probability",
        "dsmp_direct",
        "dsmp_direct_recalibrated",
        "dsmp_direct_true_probability",
        "no_rto",
        "no_uncertainty",
        "no_voi",
        "no_graph",
        "conditional_rto",
        "minmax_norm",
    ]
    ablation_rows = []
    budget = default_budget
    capacity = budget * config.capacity_per_budget
    oracle_eval = evaluate_plan(sample_df, oracle_plan(sample_df, budget, capacity))
    for method in ablation_methods:
        plan = plan_for_method(sample_df, method, budget, capacity)
        ev = evaluate_plan(sample_df, plan)
        ablation_rows.append(
            {
                "method": method,
                "loss_after": ev["loss_after"],
                "regret": ev["loss_after"] - oracle_eval["loss_after"],
                "risk_reduction": ev["risk_reduction"],
                "downtime_reduction": ev["downtime_reduction"],
                "cost": ev["cost"],
            }
        )
    pd.DataFrame(ablation_rows).to_csv(table_dir / "table_v4_ablation_study.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(table_dir / "table_v5_runtime_feasibility.csv", index=False)


def run_validation_suite(out_dir: Path, config: ValidationConfig) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "processed"
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    data_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)
    table_dir.mkdir(exist_ok=True)

    metric_records = []
    rank_records = []
    calibration_records = []
    sample_df = None
    sample_graph = None
    for offset in range(config.n_seeds):
        seed = config.seed + offset
        df, graph = generate_ground_truth_portfolio(seed, config)
        if sample_df is None:
            sample_df = df
            sample_graph = graph
            df.to_csv(data_dir / "exp1_ground_truth_sample.csv", index=False)
        probability_models = {
            "gamma_poisson": "P_predictive",
            "platt_recalibrated": "P_platt",
            "isotonic_recalibrated": "P_isotonic",
        }
        for model_name, prob_col in probability_models.items():
            cal = calibration_table(df[prob_col], df["F_true"], n_bins=10)
            cal["seed"] = seed
            cal["model"] = model_name.replace("_", " ")
            calibration_records.append(cal)
            metric_records.append(
                {
                    "seed": seed,
                    "method": f"calibration_{model_name}",
                    "budget": np.nan,
                    "capacity": np.nan,
                    "regret": np.nan,
                    "rrpc": np.nan,
                    "downtime_reduction": np.nan,
                    "brier": brier_score(df[prob_col], df["F_true"]),
                    "ece": expected_calibration_error(df[prob_col], df["F_true"], n_bins=10),
                    "mse_to_p_true": float(np.mean((df[prob_col].to_numpy(dtype=float) - df["p_true"].to_numpy(dtype=float)) ** 2)),
                    "mae_to_p_true": float(np.mean(np.abs(df[prob_col].to_numpy(dtype=float) - df["p_true"].to_numpy(dtype=float)))),
                }
            )
        rank_records.extend(rank_metric_records(df, seed, config.top_k))
        for budget in config.budget_grid:
            capacity = budget * config.capacity_per_budget
            for rec in run_budget_metrics(df, budget, capacity):
                rec["seed"] = seed
                rec["brier"] = np.nan
                rec["ece"] = np.nan
                rec["mse_to_p_true"] = np.nan
                rec["mae_to_p_true"] = np.nan
                metric_records.append(rec)

    assert sample_df is not None and sample_graph is not None
    metrics = pd.DataFrame(metric_records)
    rank_df = pd.DataFrame(rank_records)
    calibration_df = pd.concat(calibration_records, ignore_index=True)
    metrics.to_csv(table_dir / "validation_metrics_long.csv", index=False)
    rank_df.to_csv(table_dir / "rank_metrics.csv", index=False)
    calibration_df.to_csv(table_dir / "calibration_bins.csv", index=False)

    budget = config.budget_grid[min(1, len(config.budget_grid) - 1)]
    capacity = budget * config.capacity_per_budget
    runtime_rows = []
    for name, fn in [
        ("score_order", lambda: plan_for_method(sample_df, "proposed_full", budget, capacity)),
        ("recalibrated_score_order", lambda: plan_for_method(sample_df, "proposed_recalibrated", budget, capacity)),
        ("dsmp_direct_recalibrated", lambda: plan_for_method(sample_df, "dsmp_direct_recalibrated", budget, capacity)),
        ("greedy_cost_risk", lambda: plan_for_method(sample_df, "greedy_cost_risk", budget, capacity)),
        ("robust_topsis", lambda: plan_for_method(sample_df, "robust_topsis", budget, capacity)),
        ("oracle_greedy_true_benefit", lambda: oracle_plan(sample_df, budget, capacity)),
    ]:
        start = time.perf_counter()
        plan = fn()
        elapsed = time.perf_counter() - start
        runtime_rows.append(
            {
                "solver": name,
                "runtime_seconds": elapsed,
                "feasible": bool(plan["cost"].sum() <= budget + 1e-9 and plan["crew"].sum() <= capacity + 1e-9),
                "n_actions": int(len(plan)),
            }
        )
    start = time.perf_counter()
    random_pareto = pareto_random_front(sample_df, budget, capacity, seed=config.seed + 500)
    runtime_rows.append(
        {
            "solver": "random_pareto_search",
            "runtime_seconds": time.perf_counter() - start,
            "feasible": bool(random_pareto["feasible"].all()),
            "n_actions": int(len(random_pareto)),
        }
    )
    random_pareto.to_csv(table_dir / "figure_v7_random_pareto_front.csv", index=False)
    start = time.perf_counter()
    pareto = pareto_nsga2_front(sample_df, budget, capacity, seed=config.seed + 600)
    runtime_rows.append(
        {
            "solver": "nsga2_pareto_pymoo",
            "runtime_seconds": time.perf_counter() - start,
            "feasible": bool(not pareto.empty and pareto["feasible"].all()),
            "n_actions": int(len(pareto)),
        }
    )
    pareto.to_csv(table_dir / "figure_v7_pareto_front.csv", index=False)

    norm_stress = normalization_stress(sample_df, config)
    graph_stress = graph_perturbation_stress(sample_df, sample_graph, config, config.seed)
    voi_stress = voi_stress_table(sample_df)
    norm_stress.to_csv(table_dir / "figure_v8_rank_reversal_stress.csv", index=False)
    graph_stress.to_csv(table_dir / "graph_perturbation_stress.csv", index=False)
    voi_stress.to_csv(table_dir / "voi_stress.csv", index=False)

    write_summary_tables(out_dir, config, metrics, rank_df, sample_df, runtime_rows)

    plot_workflow(fig_dir / "figure_v1_validation_workflow.pdf")
    plot_calibration(calibration_df, fig_dir / "figure_v2_calibration_curve.pdf")
    plot_regret(metrics.dropna(subset=["regret"]), fig_dir / "figure_v3_regret_vs_budget.pdf")
    plot_rank_metrics(rank_df, fig_dir / "figure_v4_topk_rank_comparison.pdf")
    plot_voi_map(fig_dir / "figure_v5_voi_gate_decision_map.pdf")
    plot_network(sample_df, sample_graph, fig_dir / "figure_v6_mining_recovery_network.pdf")
    plot_pareto(pareto, fig_dir / "figure_v7_pareto_front.pdf")
    plot_rank_reversal(norm_stress, fig_dir / "figure_v8_rank_reversal_frequency.pdf")

    return {"processed": data_dir, "figures": fig_dir, "tables": table_dir}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run expanded DSMP validation experiments.")
    parser.add_argument("--out", type=Path, default=Path("validation_results"))
    parser.add_argument("--seed", type=int, default=ValidationConfig().seed)
    parser.add_argument("--n-assets", type=int, default=ValidationConfig().n_assets)
    parser.add_argument("--seeds", type=int, default=ValidationConfig().n_seeds)
    parser.add_argument("--mc-paths", type=int, default=ValidationConfig().mc_paths)
    args = parser.parse_args()
    config = ValidationConfig(seed=args.seed, n_assets=args.n_assets, n_seeds=args.seeds, mc_paths=args.mc_paths)
    outputs = run_validation_suite(args.out, config)
    print("Expanded DSMP validation experiments completed.")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
