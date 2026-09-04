"""Robustness experiments added at the revision stage (JRR-26-0219), Section 6.6.

S5. Hazard-regime sweep: the generator is re-parameterized (degradation_scale, demand_scale)
    to produce event rates from about 0.05 to the reported 0.76, and calibration, skill and
    regret comparisons are repeated in every regime (50 seeds each).
S6. Probability-quality sweep: the probability inside the score is interpolated between the
    recalibrated count-based proxy and the simulated true probability, to show how the two
    forms of the operator (system-exposure, Eq. 16; expected-loss, Eq. 17) and the greedy
    probability--consequence baseline respond to better structural evidence.
S7. Cascade-aware ground truth: the true loss is extended with a one-step dependency cascade
    over the same dependency graph, to test whether the graph-augmented score is credited when
    the ground truth contains system effects.
S9. Covariate-augmented calibration operator: Eq. (3) implemented with observable condition
    descriptors (inspection severity, exposure) in addition to the count-based proxy.
S8. Two-epoch information protocol: inspections in the first epoch reveal the physical failure
    probability of the inspected assets before the second-epoch decision, so that the value of
    information can be credited by the loss.

All defaults of the original validation protocol are unchanged; the reported protocol is the
regime (degradation_scale, demand_scale) = (1, 1).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .revision_experiments import _direct_plan_from_score, _median_summary, _paired
from .validation_experiments import (
    ACTION_EFFECTS,
    DIRECT_INTERVENTIONS,
    ValidationConfig,
    evaluate_plan,
    generate_ground_truth_portfolio,
    oracle_plan,
    plan_for_method,
    validation_config_to_dsmp,
)
from .validation_metrics import brier_score, expected_calibration_error, holm_bonferroni_adjust

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9.5})

REGIMES = {
    "R76 (reported)": dict(degradation_scale=1.0, demand_scale=1.0),
    "R50": dict(degradation_scale=1.0, demand_scale=0.6),
    "R33": dict(degradation_scale=0.7, demand_scale=0.6),
    "R18": dict(degradation_scale=0.5, demand_scale=0.6),
    "R05": dict(degradation_scale=0.35, demand_scale=0.5),
}


# ----------------------------------------------------------------------------
# S5: hazard-regime sweep
# ----------------------------------------------------------------------------
def run_hazard_regimes(config: ValidationConfig, budget: float, out_dir: Path) -> pd.DataFrame:
    capacity = budget * config.capacity_per_budget
    baselines = ["dsmp_direct", "dsmp_direct_true_probability", "greedy_cost_risk", "risk_only", "robust_topsis", "static_rcm", "consequence_only"]
    records = []
    for regime, scales in REGIMES.items():
        cfg = ValidationConfig(seed=config.seed, n_assets=config.n_assets, n_seeds=config.n_seeds, mc_paths=config.mc_paths, **scales)
        portfolios = [(cfg.seed + o, generate_ground_truth_portfolio(cfg.seed + o, cfg)[0]) for o in range(cfg.n_seeds)]
        rates = np.asarray([float(df["F_true"].mean()) for _, df in portfolios])
        for k, (seed, df) in enumerate(portfolios):
            f_true = df["F_true"].to_numpy(dtype=float)
            p_true = df["p_true"].to_numpy(dtype=float)
            base_rate_other = float((rates.sum() - rates[k]) / (len(rates) - 1))  # leave-one-seed-out base rate
            oracle_ev = evaluate_plan(df, oracle_plan(df, budget, capacity))
            oracle_loss = oracle_ev["loss_after"]
            loss_before = oracle_ev["loss_before"]
            row = {
                "regime": regime,
                "seed": seed,
                "event_rate": float(f_true.mean()),
                "brier_gamma_poisson": brier_score(df["P_predictive"], f_true),
                "brier_platt": brier_score(df["P_platt"], f_true),
                "brier_base_rate": brier_score(np.full(len(df), base_rate_other), f_true),
                "brier_true": brier_score(p_true, f_true),
                "ece_platt": expected_calibration_error(df["P_platt"], f_true, n_bins=10),
                "oracle_reduction": loss_before - oracle_loss,
            }
            row["bss_platt"] = 1.0 - row["brier_platt"] / row["brier_base_rate"]
            row["bss_gamma_poisson"] = 1.0 - row["brier_gamma_poisson"] / row["brier_base_rate"]
            row["bss_true"] = 1.0 - row["brier_true"] / row["brier_base_rate"]
            ev = evaluate_plan(df, plan_for_method(df, "dsmp_direct_recalibrated", budget, capacity))
            row["regret_dsmp_direct_recalibrated"] = ev["loss_after"] - oracle_loss
            alpha, rho, eta = validation_config_to_dsmp(cfg, seed).normalized_weights()
            score_const = (alpha * base_rate_other * df["C_graph"] + rho * df["RTO_graph"] + eta * df["U_norm"]).clip(0, 1)
            ev = evaluate_plan(df, _direct_plan_from_score(df, score_const, budget, capacity))
            row["regret_dsmp_constant_probability"] = ev["loss_after"] - oracle_loss
            for method in baselines:
                ev = evaluate_plan(df, plan_for_method(df, method, budget, capacity))
                row[f"regret_{method}"] = ev["loss_after"] - oracle_loss
            records.append(row)
        print(f"  S5 regime {regime}: event rate {rates.mean():.3f}")
    long = pd.DataFrame(records)
    long.to_csv(out_dir / "S5_hazard_regimes_long.csv", index=False)
    rows = []
    for regime in REGIMES:
        g = long[long["regime"] == regime]
        row = {"regime": regime, **REGIMES[regime], "n_seeds": len(g)}
        for col in ["event_rate", "brier_gamma_poisson", "brier_platt", "brier_base_rate", "brier_true", "ece_platt", "bss_platt", "bss_gamma_poisson", "bss_true"]:
            row[col] = float(g[col].median())
        for col in [c for c in g.columns if c.startswith("regret_")]:
            s = _median_summary(g[col].to_numpy(), seed=config.seed)
            row[f"{col}_median"] = s["median"]
            row[f"{col}_ci_low"] = s["ci_low"]
            row[f"{col}_ci_high"] = s["ci_high"]
            row[f"{col}_normalized"] = float(np.median(g[col].to_numpy() / g["oracle_reduction"].to_numpy()))
        ref = g["regret_dsmp_direct_recalibrated"].to_numpy()
        pvals = []
        for method in ["greedy_cost_risk", "risk_only", "robust_topsis", "static_rcm", "dsmp_direct", "dsmp_constant_probability"]:
            pr = _paired(ref, g[f"regret_{method}"].to_numpy(), seed=config.seed)
            row[f"effect_vs_{method}"] = pr["median_effect"]
            row[f"effect_ci_low_vs_{method}"] = pr["effect_ci_low"]
            row[f"effect_ci_high_vs_{method}"] = pr["effect_ci_high"]
            row[f"win_vs_{method}"] = pr["win_fraction"]
            pvals.append(pr["p_value"])
        holm = holm_bonferroni_adjust(np.asarray(pvals))
        for method, hp in zip(["greedy_cost_risk", "risk_only", "robust_topsis", "static_rcm", "dsmp_direct", "dsmp_constant_probability"], holm):
            row[f"holm_p_vs_{method}"] = float(hp)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "S5_hazard_regimes_summary.csv", index=False)
    plot_hazard_regimes(summary, out_dir / "figure_S5_hazard_regimes.pdf")
    return summary


def plot_hazard_regimes(summary: pd.DataFrame, path: Path) -> None:
    s = summary.sort_values("event_rate")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax = axes[0]
    for col, label, style in [
        ("regret_dsmp_direct_recalibrated", "DSMP direct, Platt-recalibrated", dict(color="#d62728", marker="o", lw=2.4)),
        ("regret_dsmp_direct", "DSMP direct, uncalibrated", dict(color="#ff7f0e", marker="o", lw=1.8)),
        ("regret_dsmp_constant_probability", "DSMP direct, constant base-rate probability", dict(color="#8c564b", marker="v", lw=1.4, ls="-.")),
        ("regret_greedy_cost_risk", "Greedy probability–consequence", dict(color="#2ca02c", marker="^", lw=1.8)),
        ("regret_risk_only", "Probability–consequence score", dict(color="#2ca02c", marker="^", lw=1.4, ls="--")),
        ("regret_robust_topsis", "Robust TOPSIS MCDA", dict(color="#9467bd", marker="D", lw=1.8)),
        ("regret_static_rcm", "Static RCM-like", dict(color="#7f7f7f", marker="x", lw=1.6, ls=":")),
        ("regret_dsmp_direct_true_probability", "DSMP direct with true $P_f$", dict(color="#d62728", marker="*", lw=1.6, ls="--")),
    ]:
        ax.plot(s["event_rate"], s[f"{col}_normalized"], label=label, markersize=5, **style)
    ax.set_xlabel("Median simulated event rate of the regime")
    ax.set_ylabel("Normalized regret at budget 12\n(0 = perfect-information greedy reference, 1 = no action; seed-level median)")
    ax.set_xscale("log")
    ax.set_xticks([0.05, 0.1, 0.2, 0.35, 0.5, 0.76])
    ax.set_xticklabels(["0.05", "0.1", "0.2", "0.35", "0.5", "0.76"])
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", frameon=True, fontsize=8.5)
    ax = axes[1]
    ax.plot(s["event_rate"], s["bss_platt"], marker="o", lw=2.2, color="#1f77b4", label="Platt-recalibrated Gamma–Poisson")
    ax.plot(s["event_rate"], s["bss_true"], marker="*", lw=1.6, ls="--", color="black", label="Simulated true probability")
    ax.axhline(0.0, color="gray", lw=1, ls=":")
    ax.set_xlabel("Median simulated event rate of the regime")
    ax.set_ylabel("Brier skill score vs. constant base-rate predictor\n(seed-level median)")
    ax.set_xscale("log")
    ax.set_xticks([0.05, 0.1, 0.2, 0.35, 0.5, 0.76])
    ax.set_xticklabels(["0.05", "0.1", "0.2", "0.35", "0.5", "0.76"])
    ax.grid(alpha=0.25)
    ax.legend(loc="center right", frameon=True)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# S6: probability-quality sweep (reported regime)
# ----------------------------------------------------------------------------
def run_probability_quality(portfolios: list[tuple[int, pd.DataFrame]], config: ValidationConfig, budget: float, out_dir: Path) -> pd.DataFrame:
    capacity = budget * config.capacity_per_budget
    weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    records = []
    rates = np.asarray([float(df["F_true"].mean()) for _, df in portfolios])
    for k, (seed, df) in enumerate(portfolios):
        oracle_loss = evaluate_plan(df, oracle_plan(df, budget, capacity))["loss_after"]
        alpha, rho, eta = validation_config_to_dsmp(config, seed).normalized_weights()
        f_true = df["F_true"].to_numpy(dtype=float)
        base = np.full(len(df), float((rates.sum() - rates[k]) / (len(rates) - 1)))  # leave-one-seed-out base rate
        for w in weights:
            p_w = (w * df["p_true"] + (1.0 - w) * df["P_platt"]).clip(0, 1)
            variants = {
                "dsmp_sys": (alpha * p_w * df["C_graph"] + rho * df["RTO_graph"] + eta * df["U_norm"]).clip(0, 1),
                "dsmp_exp": (alpha * p_w * df["C_graph"] + rho * p_w * df["RTO_graph"] + eta * df["U_norm"]).clip(0, 1),
                "dsmp_sys_no_uncertainty": ((alpha / (alpha + rho)) * p_w * df["C_graph"] + (rho / (alpha + rho)) * df["RTO_graph"]).clip(0, 1),
                "dsmp_sys_no_uncertainty_no_graph": ((alpha / (alpha + rho)) * p_w * df["C_norm"] + (rho / (alpha + rho)) * df["RTO_norm"]).clip(0, 1),
                "dsmp_exp_no_uncertainty": ((alpha / (alpha + rho)) * p_w * df["C_graph"] + (rho / (alpha + rho)) * p_w * df["RTO_graph"]).clip(0, 1),
                "dsmp_exp_no_uncertainty_no_graph": ((alpha / (alpha + rho)) * p_w * df["C_norm"] + (rho / (alpha + rho)) * p_w * df["RTO_norm"]).clip(0, 1),
                "greedy_prob_consequence": (p_w * df["C_norm"]).clip(0, 1),
            }
            row = {"seed": seed, "w_true": w, "brier": brier_score(p_w, f_true), "bss": 1.0 - brier_score(p_w, f_true) / brier_score(base, f_true)}
            for name, score in variants.items():
                ev = evaluate_plan(df, _direct_plan_from_score(df, score, budget, capacity))
                row[f"regret_{name}"] = ev["loss_after"] - oracle_loss
            records.append(row)
    long = pd.DataFrame(records)
    long.to_csv(out_dir / "S6_probability_quality_long.csv", index=False)
    rows = []
    for w, g in long.groupby("w_true"):
        row = {"w_true": w, "n_seeds": len(g), "brier_median": float(g["brier"].median()), "bss_median": float(g["bss"].median())}
        for name in ["dsmp_sys", "dsmp_exp", "dsmp_sys_no_uncertainty", "dsmp_sys_no_uncertainty_no_graph", "dsmp_exp_no_uncertainty", "dsmp_exp_no_uncertainty_no_graph", "greedy_prob_consequence"]:
            s = _median_summary(g[f"regret_{name}"].to_numpy(), seed=config.seed)
            row[f"regret_{name}_median"] = s["median"]
            row[f"regret_{name}_ci_low"] = s["ci_low"]
            row[f"regret_{name}_ci_high"] = s["ci_high"]
        for name in ["dsmp_exp", "dsmp_sys_no_uncertainty", "dsmp_sys_no_uncertainty_no_graph", "dsmp_exp_no_uncertainty", "dsmp_exp_no_uncertainty_no_graph", "greedy_prob_consequence"]:
            pr = _paired(g["regret_dsmp_sys"].to_numpy(), g[f"regret_{name}"].to_numpy(), seed=config.seed)
            row[f"effect_{name}_minus_sys"] = pr["median_effect"]
            row[f"effect_ci_low_{name}_minus_sys"] = pr["effect_ci_low"]
            row[f"effect_ci_high_{name}_minus_sys"] = pr["effect_ci_high"]
            row[f"p_{name}_minus_sys"] = pr["p_value"]
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "S6_probability_quality_summary.csv", index=False)
    return summary


# ----------------------------------------------------------------------------
# S7: cascade-aware ground truth (reported regime)
# ----------------------------------------------------------------------------
def _dependency_matrix(df: pd.DataFrame, graph: nx.DiGraph) -> np.ndarray:
    ids = {a: i for i, a in enumerate(df["asset_id"])}
    n = len(df)
    W = np.zeros((n, n))
    for u, v, data in graph.edges(data=True):
        if u in ids and v in ids:
            W[ids[u], ids[v]] = float(data.get("weight", 1.0))
    row_sums = W.sum(axis=1, keepdims=True)
    return np.divide(W, row_sums, out=np.zeros_like(W), where=row_sums > 0)


def _cascade_loss(p: np.ndarray, c_eff: np.ndarray, rto: np.ndarray) -> float:
    return float(np.sum(p * c_eff + 0.20 * p * rto))


def _cascade_oracle_plan(df: pd.DataFrame, c_eff: np.ndarray, budget: float, capacity: float) -> pd.DataFrame:
    p = df["p_true"].to_numpy(dtype=float)
    rto = df["RTO_norm"].to_numpy(dtype=float)
    rows = []
    for i in range(len(df)):
        for action in DIRECT_INTERVENTIONS:
            e = ACTION_EFFECTS[action]
            p_after = p[i] * np.exp(-2.35 * e["beta_x"])
            rto_after = rto[i] * (1.0 - e["beta_rto"])
            benefit = (p[i] * c_eff[i] + 0.20 * p[i] * rto[i]) - (p_after * c_eff[i] + 0.20 * p_after * rto_after)
            rows.append({"asset_id": df["asset_id"].iloc[i], "row_idx": i, "action": action, "score": 0.0, "estimated_benefit": max(benefit, 0.0), **e})
    table = pd.DataFrame(rows)
    table["rank_value"] = table["estimated_benefit"] / (table["cost"] + 1e-9)
    table = table.sort_values("rank_value", ascending=False, kind="mergesort")
    used, ub, uc, keep = set(), 0.0, 0.0, []
    for idx, asset, cost, crew in zip(table.index, table["asset_id"], table["cost"], table["crew"]):
        if asset in used:
            continue
        if ub + cost <= budget and uc + crew <= capacity:
            keep.append(idx)
            used.add(asset)
            ub += cost
            uc += crew
    return table.loc[keep].reset_index(drop=True)


def _cascade_evaluate(df: pd.DataFrame, plan: pd.DataFrame, c_eff: np.ndarray) -> float:
    p_after = df["p_true"].to_numpy(dtype=float).copy()
    rto_after = df["RTO_norm"].to_numpy(dtype=float).copy()
    effects = {row.asset_id: row for row in plan.itertuples(index=False)} if len(plan) else {}
    for i, asset in enumerate(df["asset_id"]):
        row = effects.get(asset)
        if row is None:
            continue
        e = ACTION_EFFECTS[row.action]
        p_after[i] *= np.exp(-2.35 * e["beta_x"])
        rto_after[i] *= 1.0 - e["beta_rto"]
    return _cascade_loss(p_after, c_eff, rto_after)


def run_cascade_truth(portfolios: list[tuple[int, pd.DataFrame, nx.DiGraph]], config: ValidationConfig, budget: float, out_dir: Path) -> pd.DataFrame:
    capacity = budget * config.capacity_per_budget
    kappas = [0.0, 0.25, 0.5, 1.0, 2.0]
    records = []
    for seed, df, graph in portfolios:
        T = _dependency_matrix(df, graph)
        c_true = df["C_true"].to_numpy(dtype=float)
        upstream = T.T @ c_true  # consequence of the assets that depend on each asset (edge u -> v: u depends on v)
        alpha, rho, eta = validation_config_to_dsmp(config, seed).normalized_weights()
        scores = {
            "graph_uncalibrated": df["dynamic_score_graph_pred"],
            "no_graph_uncalibrated": df["dynamic_score_pred"],
            "graph_recalibrated": df["dynamic_score_graph_recalibrated"],
            "no_graph_recalibrated": (alpha * df["P_recalibrated"] * df["C_norm"] + rho * df["RTO_norm"] + eta * df["U_norm"]).clip(0, 1),
            "greedy_prob_consequence": df["static_risk_pred"],
            "graph_true_probability": df["dynamic_score_graph_true_probability"],
            "no_graph_true_probability": (alpha * df["p_true"] * df["C_norm"] + rho * df["RTO_norm"] + eta * df["U_norm"]).clip(0, 1),
        }
        plans = {name: _direct_plan_from_score(df, score, budget, capacity) for name, score in scores.items()}
        for kappa in kappas:
            c_eff = c_true + kappa * upstream
            oracle_loss = _cascade_evaluate(df, _cascade_oracle_plan(df, c_eff, budget, capacity), c_eff)
            row = {"seed": seed, "kappa": kappa, "mean_upstream_share": float(np.mean(kappa * upstream) / np.mean(c_true))}
            for name, plan in plans.items():
                row[f"regret_{name}"] = _cascade_evaluate(df, plan, c_eff) - oracle_loss
            records.append(row)
    long = pd.DataFrame(records)
    long.to_csv(out_dir / "S7_cascade_truth_long.csv", index=False)
    rows = []
    for kappa, g in long.groupby("kappa"):
        row = {"kappa": kappa, "n_seeds": len(g), "mean_upstream_share": float(g["mean_upstream_share"].median())}
        for name in ["graph_uncalibrated", "no_graph_uncalibrated", "graph_recalibrated", "no_graph_recalibrated", "greedy_prob_consequence", "graph_true_probability", "no_graph_true_probability"]:
            s = _median_summary(g[f"regret_{name}"].to_numpy(), seed=config.seed)
            row[f"regret_{name}_median"] = s["median"]
            row[f"regret_{name}_ci_low"] = s["ci_low"]
            row[f"regret_{name}_ci_high"] = s["ci_high"]
        for suffix in ["uncalibrated", "recalibrated", "true_probability"]:
            pr = _paired(g[f"regret_graph_{suffix}"].to_numpy(), g[f"regret_no_graph_{suffix}"].to_numpy(), seed=config.seed)
            row[f"effect_no_graph_minus_graph_{suffix}"] = pr["median_effect"]
            row[f"effect_ci_low_{suffix}"] = pr["effect_ci_low"]
            row[f"effect_ci_high_{suffix}"] = pr["effect_ci_high"]
            row[f"p_{suffix}"] = pr["p_value"]
            row[f"graph_wins_{suffix}"] = pr["win_fraction"]
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "S7_cascade_truth_summary.csv", index=False)
    return summary


# ----------------------------------------------------------------------------
# S8: two-epoch information protocol (reported regime)
# ----------------------------------------------------------------------------
def _state_after(plan: pd.DataFrame, p: np.ndarray, rto: np.ndarray, u: np.ndarray, asset_ids: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p, rto, u = p.copy(), rto.copy(), u.copy()
    factor = np.ones(len(p))
    effects = {row.asset_id: row for row in plan.itertuples(index=False)} if len(plan) else {}
    for i, asset in enumerate(asset_ids):
        row = effects.get(asset)
        if row is None:
            continue
        e = ACTION_EFFECTS[row.action]
        factor[i] = np.exp(-2.35 * e["beta_x"])
        p[i] *= factor[i]
        rto[i] *= 1.0 - e["beta_rto"]
        u[i] *= 1.0 - e["beta_u"]
    return p, rto, u, factor


def _loss(p: np.ndarray, c: np.ndarray, rto: np.ndarray) -> float:
    return float(np.sum(p * c + 0.20 * p * rto))


def _direct_plan_excluding(df: pd.DataFrame, score: pd.Series, budget: float, capacity: float, exclude: np.ndarray) -> pd.DataFrame:
    sc = score.copy()
    sc[exclude] = -1.0  # excluded assets never selected (negative benefit)
    plan = _direct_plan_from_score(df, sc, budget, capacity)
    return plan[plan["score"] >= 0.0].reset_index(drop=True) if len(plan) else plan


def _greedy_true_plan(df: pd.DataFrame, p: np.ndarray, c: np.ndarray, rto: np.ndarray, budget: float, capacity: float) -> pd.DataFrame:
    rows = []
    for i in range(len(df)):
        for action in DIRECT_INTERVENTIONS:
            e = ACTION_EFFECTS[action]
            p_after = p[i] * np.exp(-2.35 * e["beta_x"])
            rto_after = rto[i] * (1.0 - e["beta_rto"])
            benefit = (p[i] * c[i] + 0.20 * p[i] * rto[i]) - (p_after * c[i] + 0.20 * p_after * rto_after)
            rows.append({"asset_id": df["asset_id"].iloc[i], "row_idx": i, "action": action, "score": 0.0, "estimated_benefit": max(benefit, 0.0), **e})
    table = pd.DataFrame(rows)
    table["rank_value"] = table["estimated_benefit"] / (table["cost"] + 1e-9)
    table = table.sort_values("rank_value", ascending=False, kind="mergesort")
    used, ub, uc, keep = set(), 0.0, 0.0, []
    for idx, asset, cost, crew in zip(table.index, table["asset_id"], table["cost"], table["crew"]):
        if asset in used:
            continue
        if ub + cost <= budget and uc + crew <= capacity:
            keep.append(idx)
            used.add(asset)
            ub += cost
            uc += crew
    return table.loc[keep].reset_index(drop=True)


def _priority_order(inspect: np.ndarray, rule: str, param: float, u: np.ndarray, df: pd.DataFrame, belief: np.ndarray, dsmp_score) -> np.ndarray:
    idx = np.flatnonzero(inspect)
    if rule == "top_consequence":
        key = -df["C_graph"].to_numpy(dtype=float)[idx]
    elif rule == "top_score":
        key = -dsmp_score(belief, u).to_numpy()[idx]
    elif rule in ("top_uncertainty_consequence", "gate"):
        key = -(u * df["C_norm"].to_numpy(dtype=float))[idx]
    else:
        key = np.arange(len(idx))
    return idx[np.argsort(key, kind="mergesort")]


def two_epoch_policy(df: pd.DataFrame, config: ValidationConfig, seed: int, budget: float, capacity: float, rule: str, param: float, sigma_obs: float, rng: np.random.Generator, resource_model: str = "shared") -> dict[str, float]:
    """Run one policy over two epochs and return the total true loss and bookkeeping.

    resource_model: "shared"      -- inspections consume the maintenance budget and crew (Table 4 of the manuscript);
                    "budget_only" -- inspections consume the maintenance budget but not the crew;
                    "dedicated"   -- inspections are paid from a dedicated inspection resource (neither budget nor crew).
    """
    alpha, rho, eta = validation_config_to_dsmp(config, seed).normalized_weights()
    n = len(df)
    p_true = df["p_true"].to_numpy(dtype=float)
    c_true = df["C_true"].to_numpy(dtype=float)
    rto = df["RTO_norm"].to_numpy(dtype=float)
    u = df["U_norm"].to_numpy(dtype=float)
    c_graph = df["C_graph"].to_numpy(dtype=float)
    rto_graph = df["RTO_graph"].to_numpy(dtype=float)
    belief = df["P_platt"].to_numpy(dtype=float).copy()
    if rule == "true_probability":
        belief = p_true.copy()
    elif rule == "uncalibrated":
        belief = df["P_predictive"].to_numpy(dtype=float).copy()

    def dsmp_score(b: np.ndarray, uu: np.ndarray) -> pd.Series:
        return pd.Series(np.clip(alpha * b * c_graph + rho * rto_graph + eta * uu, 0, 1), index=df.index)

    def greedy_score(b: np.ndarray) -> pd.Series:
        return pd.Series(np.clip(b * df["C_norm"].to_numpy(dtype=float), 0, 1), index=df.index)

    # ---- epoch 1: choose the inspection set
    inspect = np.zeros(n, dtype=bool)
    nvi = 0.55 * u * df["C_norm"].to_numpy(dtype=float) * 200_000.0 * 0.25 - 3_000.0
    if rule == "gate":
        inspect = (u >= param) & (nvi > 0)
    elif rule == "top_uncertainty_consequence":
        order = np.argsort(-(u * df["C_norm"].to_numpy(dtype=float)))
        inspect[order[: int(param)]] = True
    elif rule == "top_consequence":
        order = np.argsort(-df["C_graph"].to_numpy(dtype=float))
        inspect[order[: int(param)]] = True
    elif rule == "top_score":
        order = np.argsort(-dsmp_score(belief, u).to_numpy())
        inspect[order[: int(param)]] = True
    elif rule == "random":
        inspect[rng.choice(n, size=int(param), replace=False)] = True
    e_ins = ACTION_EFFECTS["inspect"]
    unit_cost = e_ins["cost"] if resource_model in ("shared", "budget_only") else 0.0
    unit_crew = e_ins["crew"] if resource_model == "shared" else 0.0
    cost_ins = float(inspect.sum()) * unit_cost
    crew_ins = float(inspect.sum()) * unit_crew
    if cost_ins > budget or crew_ins > capacity:  # trim the inspection set (in priority order) to the affordable size
        max_n = int(min(budget // unit_cost if unit_cost > 0 else n, capacity // unit_crew if unit_crew > 0 else n))
        keep_idx = _priority_order(inspect, rule, param, u, df, belief, dsmp_score)[:max_n]
        inspect = np.zeros(n, dtype=bool)
        inspect[keep_idx] = True
        cost_ins = float(inspect.sum()) * unit_cost
        crew_ins = float(inspect.sum()) * unit_crew
    score1 = greedy_score(belief) if rule == "greedy" else dsmp_score(belief, u)
    plan1 = _direct_plan_excluding(df, score1, budget - cost_ins, capacity - crew_ins, inspect)
    p1, rto1, u1, factor1 = _state_after(plan1, p_true, rto, u, df["asset_id"])
    u1 = np.where(inspect, u1 * (1.0 - e_ins["beta_u"]), u1)
    loss1 = _loss(p1, c_true, rto1)
    cost1 = float(plan1["cost"].sum()) if len(plan1) else 0.0
    # ---- information outcome: inspected assets reveal the physical failure probability (noisy)
    belief2 = belief * factor1
    if inspect.any():
        observed = np.clip(p_true[inspect] * factor1[inspect] + rng.normal(0.0, sigma_obs, size=int(inspect.sum())), 0.0, 1.0)
        belief2 = belief2.copy()
        belief2[inspect] = observed
    # ---- epoch 2: direct interventions with the updated beliefs
    score2 = greedy_score(belief2) if rule == "greedy" else dsmp_score(belief2, u1)
    plan2 = _direct_plan_from_score(df, score2, budget, capacity)
    p2, rto2, u2, _ = _state_after(plan2, p1, rto1, u1, df["asset_id"])
    loss2 = _loss(p2, c_true, rto2)
    return {
        "loss_epoch1": loss1,
        "loss_epoch2": loss2,
        "total_loss": loss1 + loss2,
        "n_inspected": int(inspect.sum()),
        "inspection_cost_charged": cost_ins,
        "cost_epoch1_direct": cost1,
        "cost_epoch1_inspection": cost_ins,
        "cost_epoch2": float(plan2["cost"].sum()) if len(plan2) else 0.0,
    }


def run_two_epoch_voi(portfolios: list[tuple[int, pd.DataFrame]], config: ValidationConfig, budget: float, out_dir: Path) -> pd.DataFrame:
    capacity = budget * config.capacity_per_budget
    policies = [
        ("direct_only_platt", "none", 0),
        ("direct_only_uncalibrated", "uncalibrated", 0),
        ("direct_only_greedy", "greedy", 0),
        ("direct_only_true_probability", "true_probability", 0),
        ("gate_u0.62", "gate", 0.62),
        ("gate_u0.45", "gate", 0.45),
        ("gate_u0.30", "gate", 0.30),
        ("inspect_top30_uncertainty_x_consequence", "top_uncertainty_consequence", 30),
        ("inspect_top60_uncertainty_x_consequence", "top_uncertainty_consequence", 60),
        ("inspect_top120_uncertainty_x_consequence", "top_uncertainty_consequence", 120),
        ("inspect_top30_consequence", "top_consequence", 30),
        ("inspect_top60_consequence", "top_consequence", 60),
        ("inspect_top120_consequence", "top_consequence", 120),
        ("inspect_top30_score", "top_score", 30),
        ("inspect_top60_score", "top_score", 60),
        ("inspect_top120_score", "top_score", 120),
        ("inspect_random60", "random", 60),
    ]
    records = []
    for resource_model in ["shared", "budget_only", "dedicated"]:
      for sigma_obs in [0.05, 0.20]:
        for seed, df in portfolios:
            rng = np.random.default_rng(seed + 9001)
            # two-epoch oracle
            p_true = df["p_true"].to_numpy(dtype=float)
            c_true = df["C_true"].to_numpy(dtype=float)
            rto = df["RTO_norm"].to_numpy(dtype=float)
            u = df["U_norm"].to_numpy(dtype=float)
            o1 = _greedy_true_plan(df, p_true, c_true, rto, budget, capacity)
            p1, rto1, u1, _ = _state_after(o1, p_true, rto, u, df["asset_id"])
            o2 = _greedy_true_plan(df, p1, c_true, rto1, budget, capacity)
            p2, rto2, _, _ = _state_after(o2, p1, rto1, u1, df["asset_id"])
            oracle_total = _loss(p1, c_true, rto1) + _loss(p2, c_true, rto2)
            for name, rule, param in policies:
                res = two_epoch_policy(df, config, seed, budget, capacity, rule, param, sigma_obs, np.random.default_rng(seed + 9001), resource_model=resource_model)
                res.update({"policy": name, "rule": rule, "param": param, "sigma_obs": sigma_obs, "resource_model": resource_model, "seed": seed, "regret_total": res["total_loss"] - oracle_total, "oracle_total": oracle_total})
                records.append(res)
    long = pd.DataFrame(records)
    long.to_csv(out_dir / "S8_two_epoch_voi_long.csv", index=False)
    rows = []
    for (resource_model, sigma_obs, name), g in long.groupby(["resource_model", "sigma_obs", "policy"], sort=False):
        ref = long[(long["resource_model"] == resource_model) & (long["sigma_obs"] == sigma_obs) & (long["policy"] == "direct_only_platt")].set_index("seed")["regret_total"]
        gg = g.set_index("seed").loc[ref.index]
        row = {"resource_model": resource_model, "sigma_obs": sigma_obs, "policy": name, "rule": gg["rule"].iloc[0], "param": gg["param"].iloc[0], "n_seeds": len(gg), "n_inspected_median": float(gg["n_inspected"].median())}
        s = _median_summary(gg["regret_total"].to_numpy(), seed=config.seed)
        row.update({"regret_median": s["median"], "regret_ci_low": s["ci_low"], "regret_ci_high": s["ci_high"]})
        row["loss_epoch1_median"] = float(gg["loss_epoch1"].median())
        row["loss_epoch2_median"] = float(gg["loss_epoch2"].median())
        if name != "direct_only_platt":
            pr = _paired(ref.to_numpy(), gg["regret_total"].to_numpy(), seed=config.seed)
            row.update({"effect_vs_direct_platt": pr["median_effect"], "effect_ci_low": pr["effect_ci_low"], "effect_ci_high": pr["effect_ci_high"], "p_value": pr["p_value"], "worse_fraction": pr["win_fraction"]})
        rows.append(row)
    summary = pd.DataFrame(rows)
    for (resource_model, sigma_obs), g in summary.groupby(["resource_model", "sigma_obs"]):
        mask = (summary["sigma_obs"] == sigma_obs) & (summary["resource_model"] == resource_model)
        pv = summary.loc[mask, "p_value"].to_numpy(dtype=float)
        summary.loc[mask, "holm_p"] = holm_bonferroni_adjust(pv)
    summary.to_csv(out_dir / "S8_two_epoch_voi_summary.csv", index=False)
    plot_two_epoch(summary, out_dir / "figure_S8_two_epoch_voi.pdf")
    return summary


def plot_two_epoch(summary: pd.DataFrame, path: Path, sigma: float = 0.05) -> None:
    titles = {"shared": "(a) Inspection uses maintenance budget and crew (Table 4)", "budget_only": "(b) Inspection uses the maintenance budget only", "dedicated": "(c) Dedicated inspection resource"}
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    handles = None
    for ax, resource_model in zip(axes, ["shared", "budget_only", "dedicated"]):
        s = summary[(summary["sigma_obs"] == sigma) & (summary["resource_model"] == resource_model)]
        ref = float(s[s["policy"] == "direct_only_platt"]["regret_median"].iloc[0])
        ax.axhline(ref, color="#d62728", lw=2, label="Direct only, Platt-recalibrated (no inspection)")
        ref_g = float(s[s["policy"] == "direct_only_greedy"]["regret_median"].iloc[0])
        ax.axhline(ref_g, color="#2ca02c", lw=1.4, ls="--", label="Direct only, greedy probability–consequence")
        ref_t = float(s[s["policy"] == "direct_only_true_probability"]["regret_median"].iloc[0])
        ax.axhline(ref_t, color="black", lw=1.2, ls=":", label="Direct only with true $P_f$ (upper bound)")
        for rule, label, style in [
            ("top_consequence", "Inspect top-$n$ by consequence, then act", dict(color="#9467bd", marker="D")),
            ("top_uncertainty_consequence", "Inspect top-$n$ by uncertainty × consequence, then act", dict(color="#ff7f0e", marker="^")),
            ("top_score", "Inspect top-$n$ by DSMP score, then act", dict(color="#1f77b4", marker="o")),
            ("random", "Inspect $n$ random assets, then act", dict(color="#7f7f7f", marker="x")),
        ]:
            g = s[s["rule"] == rule].sort_values("param")
            ax.errorbar(g["n_inspected_median"], g["regret_median"], yerr=[g["regret_median"] - g["regret_ci_low"], g["regret_ci_high"] - g["regret_median"]], label=label, capsize=3, lw=1.6, **style)
        gate = s[s["rule"] == "gate"]
        ax.scatter(gate["n_inspected_median"], gate["regret_median"], marker="s", s=48, color="#8c564b", zorder=5, label="VoI gate ($u^\\star$ = 0.62, 0.45, 0.30)")
        ax.set_xlabel("Number of assets inspected in epoch 1 (median)")
        ax.set_title(titles[resource_model], fontsize=10.5)
        ax.grid(alpha=0.25)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    axes[0].set_ylabel("Two-epoch regret vs. sequential perfect-information reference\n(normalized expected-loss units; median, 95% CI)")
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_hazard_regimes_combined(s5: pd.DataFrame, s9: pd.DataFrame, path: Path) -> None:
    """Figure for Section 6.6: normalized regret and Brier skill across hazard regimes, including the covariate-augmented calibration."""
    a = s5.sort_values("event_rate")
    b = s9.set_index("regime").loc[a["regime"]].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    ax = axes[0]
    for frame, col, label, style in [
        (b, "regret_dsmp_sys0_covariate", "DSMP, $\\eta=0$, no graph terms, covariate-calibrated $P$ (Eq. 3 with severity)", dict(color="#d62728", marker="o", lw=2.6)),
        (b, "regret_dsmp_sys_covariate", "DSMP reported configuration, covariate-calibrated $P$", dict(color="#d62728", marker="o", lw=1.4, ls="--")),
        (b, "regret_dsmp_sys0_platt", "DSMP, $\\eta=0$, no graph terms, Platt-recalibrated count proxy", dict(color="#ff7f0e", marker="s", lw=1.4, ls="--")),
        (a, "regret_dsmp_direct_recalibrated", "DSMP reported configuration, Platt-recalibrated count proxy", dict(color="#ff7f0e", marker="s", lw=2.0)),
        (b, "regret_greedy_covariate", "Greedy probability–consequence, covariate-calibrated $P$", dict(color="#2ca02c", marker="^", lw=1.6)),
        (a, "regret_greedy_cost_risk", "Greedy probability–consequence, uncalibrated proxy", dict(color="#2ca02c", marker="^", lw=1.4, ls="--")),
        (a, "regret_robust_topsis", "Robust TOPSIS MCDA", dict(color="#9467bd", marker="D", lw=1.4, ls="--")),
        (b, "regret_static_rcm", "Static RCM-like (severity and class), same action rule", dict(color="#7f7f7f", marker="x", lw=1.8, ls=":")),
        (b, "regret_dsmp_exp_true_probability", "DSMP with true $P_f$ (upper bound)", dict(color="black", marker="*", lw=1.2, ls=":")),
    ]:
        ax.plot(a["event_rate"], frame[f"{col}_normalized"], label=label, markersize=5, **style)
    ax.set_xlabel("Median simulated event rate of the regime")
    ax.set_ylabel("Normalized regret at budget 12\n(0 = perfect-information greedy reference, 1 = no action; seed-level median)")
    ax.set_xscale("log")
    ax.set_xticks([0.05, 0.1, 0.2, 0.35, 0.5, 0.76])
    ax.set_xticklabels(["0.05", "0.1", "0.2", "0.35", "0.5", "0.76"])
    ax.set_ylim(0, 0.8)
    ax.grid(alpha=0.25)
    ax = axes[1]
    ax.plot(a["event_rate"], b["bss_covariate"], marker="o", lw=2.4, color="#d62728", label="Covariate-calibrated $P$ (Eq. 3 with severity), within-seed cross-fit")
    ax.plot(a["event_rate"], b["bss_covariate_cross_seed"], marker="o", lw=1.4, ls="--", color="#d62728", label="Covariate-calibrated $P$, trained on independent seeds")
    ax.plot(a["event_rate"], a["bss_platt"], marker="s", lw=2.0, color="#ff7f0e", label="Platt-recalibrated count proxy (reported)")
    ax.plot(a["event_rate"], a["bss_true"], marker="*", lw=1.2, ls=":", color="black", label="Simulated true probability")
    ax.axhline(0.0, color="gray", lw=1, ls=":")
    ax.set_xlabel("Median simulated event rate of the regime")
    ax.set_ylabel("Brier skill score vs. constant base-rate predictor\n(seed-level median)")
    ax.set_xscale("log")
    ax.set_xticks([0.05, 0.1, 0.2, 0.35, 0.5, 0.76])
    ax.set_xticklabels(["0.05", "0.1", "0.2", "0.35", "0.5", "0.76"])
    ax.set_ylim(-0.05, 1.0)
    ax.grid(alpha=0.25)
    h0, l0 = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, loc="lower center", ncol=2, frameon=False, fontsize=8.6, bbox_to_anchor=(0.5, -0.30))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# S9: covariate-augmented calibration operator (Eq. 3 with condition descriptors)
# ----------------------------------------------------------------------------
def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def fit_logistic_ridge(X: np.ndarray, y: np.ndarray, penalty: float = 1e-4) -> np.ndarray:
    """Penalized logistic regression by L-BFGS-B with a ridge penalty of 1e-4 on the coefficients (intercept unpenalized)."""
    from scipy.optimize import minimize

    Xb = np.column_stack([np.ones(len(X)), X])
    if len(np.unique(y)) < 2:
        return np.r_[_logit([np.mean(y) if len(y) else 0.5])[0], np.zeros(X.shape[1])]

    def objective(w: np.ndarray) -> float:
        z = np.clip(Xb @ w, -35.0, 35.0)
        q = np.clip(1.0 / (1.0 + np.exp(-z)), 1e-8, 1.0 - 1e-8)
        nll = -(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)).mean()
        return float(nll + penalty * np.sum(w[1:] ** 2))

    res = minimize(objective, x0=np.zeros(Xb.shape[1]), method="L-BFGS-B")
    return res.x


def apply_logistic(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    Xb = np.column_stack([np.ones(len(X)), X])
    z = np.clip(Xb @ w, -35.0, 35.0)
    return np.clip(1.0 / (1.0 + np.exp(-z)), 0.0, 1.0)


def covariate_features(df: pd.DataFrame) -> np.ndarray:
    """Descriptor vector z_i of Eq. (3): logit of the count-based proxy, inspection severity, exposure."""
    return np.column_stack([
        _logit(df["P_predictive"].to_numpy(dtype=float)),
        df["severity"].to_numpy(dtype=float) / 6.0,
        df["environmental_exposure"].to_numpy(dtype=float),
    ])


def cross_fitted_covariate_recalibration(df: pd.DataFrame, seed: int, n_splits: int = 5) -> np.ndarray:
    X = covariate_features(df)
    y = df["F_true"].to_numpy(dtype=float)
    n = len(df)
    rng = np.random.default_rng(seed + 1203)  # same fold assignment as the Platt cross-fit
    indices = rng.permutation(n)
    folds = np.array_split(indices, n_splits)
    out = np.empty(n)
    for fold in folds:
        train = np.setdiff1d(indices, fold, assume_unique=True)
        w = fit_logistic_ridge(X[train], y[train])
        out[fold] = apply_logistic(X[fold], w)
    return out


def run_covariate_calibration(config: ValidationConfig, budget: float, out_dir: Path) -> pd.DataFrame:
    capacity = budget * config.capacity_per_budget
    records = []
    for regime, scales in REGIMES.items():
        cfg = ValidationConfig(seed=config.seed, n_assets=config.n_assets, n_seeds=config.n_seeds, mc_paths=config.mc_paths, **scales)
        portfolios = [(cfg.seed + o, generate_ground_truth_portfolio(cfg.seed + o, cfg)[0]) for o in range(cfg.n_seeds)]
        rates = np.asarray([float(df["F_true"].mean()) for _, df in portfolios])
        # cross-seed covariate calibrator (two halves), for the transfer check
        half = len(portfolios) // 2
        halves = [(portfolios[:half], portfolios[half:]), (portfolios[half:], portfolios[:half])]
        cross = {}
        for train_set, test_set in halves:
            Xtr = np.vstack([covariate_features(d) for _, d in train_set])
            ytr = np.concatenate([d["F_true"].to_numpy(dtype=float) for _, d in train_set])
            w = fit_logistic_ridge(Xtr, ytr)
            for seed, d in test_set:
                cross[seed] = apply_logistic(covariate_features(d), w)
        for k, (seed, df) in enumerate(portfolios):
            f_true = df["F_true"].to_numpy(dtype=float)
            base_rate_other = float((rates.sum() - rates[k]) / (len(rates) - 1))
            brier_base = brier_score(np.full(len(df), base_rate_other), f_true)
            p_cov = cross_fitted_covariate_recalibration(df, seed)
            p_cov_cross = cross[seed]
            oracle_ev = evaluate_plan(df, oracle_plan(df, budget, capacity))
            oracle_loss = oracle_ev["loss_after"]
            alpha, rho, eta = validation_config_to_dsmp(cfg, seed).normalized_weights()
            a2, r2 = alpha / (alpha + rho), rho / (alpha + rho)
            row = {
                "regime": regime, "seed": seed, "event_rate": float(f_true.mean()), "oracle_reduction": oracle_ev["loss_before"] - oracle_loss,
                "brier_platt": brier_score(df["P_platt"], f_true), "brier_covariate": brier_score(p_cov, f_true), "brier_covariate_cross_seed": brier_score(p_cov_cross, f_true),
                "ece_covariate": expected_calibration_error(p_cov, f_true, n_bins=10),
                "bss_platt": 1.0 - brier_score(df["P_platt"], f_true) / brier_base,
                "bss_covariate": 1.0 - brier_score(p_cov, f_true) / brier_base,
                "bss_covariate_cross_seed": 1.0 - brier_score(p_cov_cross, f_true) / brier_base,
                "bss_true": 1.0 - brier_score(df["p_true"], f_true) / brier_base,
            }
            pc = pd.Series(p_cov, index=df.index)
            scores = {
                "dsmp_sys_platt": df["dynamic_score_graph_recalibrated"],
                "dsmp_sys_covariate": (alpha * pc * df["C_graph"] + rho * df["RTO_graph"] + eta * df["U_norm"]).clip(0, 1),
                "dsmp_exp_covariate": (a2 * pc * df["C_norm"] + r2 * pc * df["RTO_norm"]).clip(0, 1),
                "dsmp_exp_platt": (a2 * df["P_platt"] * df["C_norm"] + r2 * df["P_platt"] * df["RTO_norm"]).clip(0, 1),
                "dsmp_sys0_covariate": (a2 * pc * df["C_norm"] + r2 * df["RTO_norm"]).clip(0, 1),
                "dsmp_sys0_platt": (a2 * df["P_platt"] * df["C_norm"] + r2 * df["RTO_norm"]).clip(0, 1),
                "greedy_covariate": (pc * df["C_norm"]).clip(0, 1),
                "greedy_platt": (df["P_platt"] * df["C_norm"]).clip(0, 1),
                "static_rcm": (0.62 * df["severity"] / 6.0 + 0.38 * df["class_criticality"]).clip(0, 1),
                "dsmp_exp_true_probability": (a2 * df["p_true"] * df["C_norm"] + r2 * df["p_true"] * df["RTO_norm"]).clip(0, 1),
            }
            for name, score in scores.items():
                ev = evaluate_plan(df, _direct_plan_from_score(df, score, budget, capacity))
                row[f"regret_{name}"] = ev["loss_after"] - oracle_loss
                if name in ("static_rcm", "dsmp_sys_platt", "dsmp_sys0_covariate"):
                    row[f"rrpc_{name}"] = ev["rrpc"]
                    row[f"downtime_{name}"] = ev["downtime_reduction"]
            records.append(row)
        print(f"  S9 regime {regime}: BSS covariate {np.median([r['bss_covariate'] for r in records if r['regime'] == regime]):.3f}")
    long = pd.DataFrame(records)
    long.to_csv(out_dir / "S9_covariate_calibration_long.csv", index=False)
    rows = []
    names = ["dsmp_sys_platt", "dsmp_sys_covariate", "dsmp_exp_platt", "dsmp_exp_covariate", "dsmp_sys0_platt", "dsmp_sys0_covariate", "greedy_platt", "greedy_covariate", "static_rcm", "dsmp_exp_true_probability"]
    for regime in REGIMES:
        g = long[long["regime"] == regime]
        row = {"regime": regime, "n_seeds": len(g)}
        for col in ["event_rate", "brier_platt", "brier_covariate", "brier_covariate_cross_seed", "ece_covariate", "bss_platt", "bss_covariate", "bss_covariate_cross_seed", "bss_true"]:
            row[col] = float(g[col].median())
        for name in names:
            s = _median_summary(g[f"regret_{name}"].to_numpy(), seed=config.seed)
            row[f"regret_{name}_median"], row[f"regret_{name}_ci_low"], row[f"regret_{name}_ci_high"] = s["median"], s["ci_low"], s["ci_high"]
            row[f"regret_{name}_normalized"] = float(np.median(g[f"regret_{name}"].to_numpy() / g["oracle_reduction"].to_numpy()))
        ref = g["regret_dsmp_sys0_covariate"].to_numpy()
        pv = []
        comps = ["static_rcm", "greedy_covariate", "dsmp_exp_covariate", "dsmp_sys_covariate", "dsmp_sys_platt", "dsmp_sys0_platt"]
        for name in comps:
            pr = _paired(ref, g[f"regret_{name}"].to_numpy(), seed=config.seed)
            row[f"effect_vs_{name}"], row[f"effect_ci_low_vs_{name}"], row[f"effect_ci_high_vs_{name}"], row[f"win_vs_{name}"] = pr["median_effect"], pr["effect_ci_low"], pr["effect_ci_high"], pr["win_fraction"]
            pv.append(pr["p_value"])
        for name, hp in zip(comps, holm_bonferroni_adjust(np.asarray(pv))):
            row[f"holm_p_vs_{name}"] = float(hp)
        for ref_name in ["dsmp_sys_platt", "dsmp_sys0_platt"]:
            pr = _paired(g[f"regret_{ref_name}"].to_numpy(), g["regret_static_rcm"].to_numpy(), seed=config.seed)
            row[f"rcm_effect_vs_{ref_name}"], row[f"rcm_ci_low_vs_{ref_name}"], row[f"rcm_ci_high_vs_{ref_name}"], row[f"rcm_p_vs_{ref_name}"], row[f"{ref_name}_wins_vs_rcm"] = pr["median_effect"], pr["effect_ci_low"], pr["effect_ci_high"], pr["p_value"], pr["win_fraction"]
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "S9_covariate_calibration_summary.csv", index=False)
    return summary


# ----------------------------------------------------------------------------
# S10: budget grid under the common action rule (revision, Section 6.3)
# ----------------------------------------------------------------------------
def run_budget_grid_common_rule(portfolios: list[tuple[int, pd.DataFrame]], config: ValidationConfig, out_dir: Path,
                                budgets: tuple[float, ...] = (8.0, 12.0, 16.0, 20.0)) -> pd.DataFrame:
    """Regret of the main policies at every budget of the grid when all of them use the
    enumerated-intervention, benefit-per-cost rule (the common action rule of Section 6.3).

    Columns: rcm_enum (severity-based RCM-like score), dsmp_cov (system-exposure form, eta = 0,
    no graph terms, condition-informed probability), dsmp_sys0_platt (same score with the Platt
    count proxy), dsmp_recal (reported configuration, Platt count proxy), greedy (probability--
    consequence with the uncalibrated proxy, as in the main tables)."""
    records = []
    for seed, df in portfolios:
        alpha, rho, eta = validation_config_to_dsmp(config, seed).normalized_weights()
        a2, r2 = alpha / (alpha + rho), rho / (alpha + rho)
        pc = pd.Series(cross_fitted_covariate_recalibration(df, seed), index=df.index)
        scores = {
            "rcm_enum": (0.62 * df["severity"] / 6.0 + 0.38 * df["class_criticality"]).clip(0, 1),
            "dsmp_cov": (a2 * pc * df["C_norm"] + r2 * df["RTO_norm"]).clip(0, 1),
            "dsmp_sys0_platt": (a2 * df["P_platt"] * df["C_norm"] + r2 * df["RTO_norm"]).clip(0, 1),
            "dsmp_recal": df["dynamic_score_graph_recalibrated"],
            "greedy": df["static_risk_pred"],
        }
        for budget in budgets:
            capacity = budget * config.capacity_per_budget
            ref_loss = evaluate_plan(df, oracle_plan(df, budget, capacity))["loss_after"]
            row = {"seed": seed, "budget": budget}
            for name, score in scores.items():
                row[name] = evaluate_plan(df, _direct_plan_from_score(df, score, budget, capacity))["loss_after"] - ref_loss
            records.append(row)
    long = pd.DataFrame(records)
    long.to_csv(out_dir / "S10_budget_grid_common_rule.csv", index=False)
    rows = []
    for budget, g in long.groupby("budget"):
        row = {"budget": budget, "n_seeds": len(g)}
        for name in ["rcm_enum", "dsmp_cov", "dsmp_sys0_platt", "dsmp_recal", "greedy"]:
            s = _median_summary(g[name].to_numpy(), seed=config.seed)
            row[f"{name}_median"], row[f"{name}_ci_low"], row[f"{name}_ci_high"] = s["median"], s["ci_low"], s["ci_high"]
        for ref_name in ["dsmp_recal", "dsmp_cov"]:
            pr = _paired(g[ref_name].to_numpy(), g["rcm_enum"].to_numpy(), seed=config.seed)
            row[f"rcm_effect_vs_{ref_name}"], row[f"rcm_ci_low_vs_{ref_name}"], row[f"rcm_ci_high_vs_{ref_name}"] = pr["median_effect"], pr["effect_ci_low"], pr["effect_ci_high"]
            row[f"rcm_p_vs_{ref_name}"], row[f"{ref_name}_wins_vs_rcm"] = pr["p_value"], pr["win_fraction"]
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "S10_budget_grid_common_rule_summary.csv", index=False)
    return summary


def compare_weight_grid_with_rcm(s1_long_path: Path, s10_long_path: Path, out_dir: Path, budget: float = 12.0) -> pd.DataFrame:
    """Paired comparison of every point of the S1 weight grid with the RCM-like rule under the
    common action rule (Table 13, last row). Requires the S1 long file of revision_experiments.py."""
    long = pd.read_csv(s1_long_path)
    s10 = pd.read_csv(s10_long_path)
    rcm = s10[s10["budget"] == budget].set_index("seed")["rcm_enum"]
    rcm_median = float(rcm.median())
    frames = []
    for variant, block in long.groupby("variant"):
        rows = []
        for (alpha, rho, eta), d in block.groupby(["alpha", "rho", "eta"]):
            d = d.set_index("seed")
            pr = _paired(d["regret"].to_numpy(), rcm.loc[d.index].to_numpy())
            rows.append({"variant": variant, "alpha": alpha, "rho": rho, "eta": eta, "regret_median": float(d["regret"].median()),
                         "rcm_effect": pr["median_effect"], "p_value": pr["p_value"], "dsmp_wins": pr["win_fraction"]})
        res = pd.DataFrame(rows)
        res["holm_p"] = holm_bonferroni_adjust(res["p_value"].to_numpy())
        res["dsmp_lower_median"] = res["regret_median"] < rcm_median
        frames.append(res)
    res = pd.concat(frames, ignore_index=True)
    res.to_csv(out_dir / "S10_weight_grid_vs_rcm_common_rule.csv", index=False)
    agg = res.groupby("variant").apply(lambda g: pd.Series({
        "grid_points": len(g),
        "grid_share_lower_median_than_rcm": float(g["dsmp_lower_median"].mean()),
        "grid_share_lower_and_holm_significant": float((g["dsmp_lower_median"] & (g["holm_p"] < 0.05)).mean()),
        "grid_share_rcm_significantly_better": float((~g["dsmp_lower_median"] & (g["holm_p"] < 0.05)).mean()),
        "min_seed_share_dsmp_wins": float(g["dsmp_wins"].min()),
    })).reset_index()
    agg.to_csv(out_dir / "S10_weight_grid_vs_rcm_common_rule_aggregate.csv", index=False)
    return agg

# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def run_all(out_dir: Path, config: ValidationConfig, budget: float = 12.0, skip: tuple[str, ...] = ()) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    portfolios = []
    for offset in range(config.n_seeds):
        seed = config.seed + offset
        df, graph = generate_ground_truth_portfolio(seed, config)
        portfolios.append((seed, df, graph))
    print(f"generated {len(portfolios)} reported-regime portfolios in {time.perf_counter() - t0:.1f} s")
    if "S6" not in skip:
        t0 = time.perf_counter()
        run_probability_quality([(s, d) for s, d, _ in portfolios], config, budget, out_dir)
        print(f"S6 probability-quality sweep done in {time.perf_counter() - t0:.1f} s")
    if "S7" not in skip:
        t0 = time.perf_counter()
        run_cascade_truth(portfolios, config, budget, out_dir)
        print(f"S7 cascade-aware truth done in {time.perf_counter() - t0:.1f} s")
    if "S8" not in skip:
        t0 = time.perf_counter()
        run_two_epoch_voi([(s, d) for s, d, _ in portfolios], config, budget, out_dir)
        print(f"S8 two-epoch VoI protocol done in {time.perf_counter() - t0:.1f} s")
    if "S5" not in skip:
        t0 = time.perf_counter()
        run_hazard_regimes(config, budget, out_dir)
        print(f"S5 hazard regimes done in {time.perf_counter() - t0:.1f} s")
    if "S9" not in skip:
        t0 = time.perf_counter()
        run_covariate_calibration(config, budget, out_dir)
        print(f"S9 covariate-augmented calibration done in {time.perf_counter() - t0:.1f} s")
    if "S10" not in skip:
        t0 = time.perf_counter()
        run_budget_grid_common_rule([(s, d) for s, d, _ in portfolios], config, out_dir)
        s1_long = out_dir.parent / "revision_results" / "S1_weight_sensitivity_long.csv"
        if s1_long.exists():
            compare_weight_grid_with_rcm(s1_long, out_dir / "S10_budget_grid_common_rule.csv", out_dir, budget=budget)
        print(f"S10 budget grid under the common action rule done in {time.perf_counter() - t0:.1f} s")
    s5_path, s9_path = out_dir / "S5_hazard_regimes_summary.csv", out_dir / "S9_covariate_calibration_summary.csv"
    if s5_path.exists() and s9_path.exists():
        plot_hazard_regimes_combined(pd.read_csv(s5_path), pd.read_csv(s9_path), out_dir / "figure_S5_hazard_regimes_combined.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run revision-stage robustness experiments (S5-S10).")
    parser.add_argument("--out", type=Path, default=Path("robustness_results"))
    parser.add_argument("--seed", type=int, default=ValidationConfig().seed)
    parser.add_argument("--n-assets", type=int, default=ValidationConfig().n_assets)
    parser.add_argument("--seeds", type=int, default=ValidationConfig().n_seeds)
    parser.add_argument("--mc-paths", type=int, default=ValidationConfig().mc_paths)
    parser.add_argument("--budget", type=float, default=12.0)
    parser.add_argument("--skip", type=str, default="", help="comma-separated experiment ids to skip, e.g. S5,S8")
    args = parser.parse_args()
    config = ValidationConfig(seed=args.seed, n_assets=args.n_assets, n_seeds=args.seeds, mc_paths=args.mc_paths)
    run_all(args.out, config, budget=args.budget, skip=tuple(x for x in args.skip.split(",") if x))


if __name__ == "__main__":
    main()
