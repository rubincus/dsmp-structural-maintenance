"""Supplementary DSMP validation experiments.

The script reuses the unchanged validation generator and policy machinery from
``validation_experiments.py`` and adds four robustness and calibration analyses:

S1. Multi-seed sensitivity of the regret comparison to the DSMP weights (alpha, rho, eta).
S2. Multi-seed ablation of the score terms under a single, common action-selection rule.
S3. Cross-seed probability recalibration (calibrator trained on independent seeds and
    evaluated on entirely held-out seeds), together with base-rate reference predictors.
S4. Event base rate, irreducible Brier component, and Brier skill scores.

All outputs are written as CSV tables and PDF figures.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .validation_experiments import (
    ACTION_EFFECTS,
    DIRECT_INTERVENTIONS,
    ValidationConfig,
    all_action_table,
    candidate_table,
    display_method_label,
    evaluate_plan,
    generate_ground_truth_portfolio,
    oracle_plan,
    plan_for_method,
    select_plan,
    validation_config_to_dsmp,
)
from .validation_metrics import (
    apply_isotonic_regression,
    apply_platt_scaling,
    bootstrap_ci,
    brier_score,
    expected_calibration_error,
    fit_isotonic_regression,
    fit_platt_scaling,
    holm_bonferroni_adjust,
    ranking_metrics,
)

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 10})


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _direct_plan_from_score(df: pd.DataFrame, score: pd.Series, budget: float, capacity: float) -> pd.DataFrame:
    """Vectorized equivalent of ``select_plan(all_action_table(..., allowed_actions=DIRECT_INTERVENTIONS), ratio=True)``.

    The greedy benefit--cost selection, the at-most-one-action-per-asset rule, and the
    budget/crew constraints are identical to the validation code; only the table
    construction is vectorized to make grid sweeps affordable.
    """
    s = np.asarray(score, dtype=float)
    n = len(s)
    rows = []
    for action in DIRECT_INTERVENTIONS:
        e = ACTION_EFFECTS[action]
        benefit = s * (0.55 * e["beta_x"] + 0.25 * e["beta_rto"] + 0.20 * e["beta_u"])
        rows.append(pd.DataFrame({"asset_id": df["asset_id"].to_numpy(), "row_idx": np.arange(n), "action": action, "score": s,
                                  "estimated_benefit": benefit, "cost": e["cost"], "crew": e["crew"], "beta_x": e["beta_x"], "beta_u": e["beta_u"], "beta_rto": e["beta_rto"]}))
    table = pd.concat(rows, ignore_index=True)
    table["rank_value"] = table["estimated_benefit"] / (table["cost"] + 1e-9)
    table = table.sort_values("rank_value", ascending=False, kind="mergesort")
    used = set()
    used_budget = 0.0
    used_capacity = 0.0
    keep = []
    for idx, asset, cost, crew in zip(table.index, table["asset_id"].to_numpy(), table["cost"].to_numpy(), table["crew"].to_numpy()):
        if asset in used:
            continue
        if used_budget + cost <= budget and used_capacity + crew <= capacity:
            keep.append(idx)
            used.add(asset)
            used_budget += cost
            used_capacity += crew
    return table.loc[keep].reset_index(drop=True)


def _regret_of_score(df: pd.DataFrame, score: pd.Series, budget: float, capacity: float, oracle_loss: float) -> dict[str, float]:
    ev = evaluate_plan(df, _direct_plan_from_score(df, score, budget, capacity))
    ev["regret"] = ev["loss_after"] - oracle_loss
    return ev


def _median_summary(values: np.ndarray, seed: int = 0) -> dict[str, float]:
    lo, hi = bootstrap_ci(values, seed=seed, statistic="median")
    return {"median": float(np.nanmedian(values)), "ci_low": lo, "ci_high": hi, "mean": float(np.nanmean(values))}


def _paired(reference: np.ndarray, comparator: np.ndarray, seed: int = 0) -> dict[str, float]:
    diff = comparator - reference
    lo, hi = bootstrap_ci(diff, seed=seed, statistic="median")
    if np.allclose(diff, 0.0):
        p = 1.0
    else:
        p = float(wilcoxon(diff).pvalue)
    return {"median_effect": float(np.median(diff)), "effect_ci_low": lo, "effect_ci_high": hi, "p_value": p, "win_fraction": float(np.mean(diff > 0))}


# ----------------------------------------------------------------------------
# S1: weight sensitivity
# ----------------------------------------------------------------------------
def weight_grid(step: float = 0.05, alpha_min: float = 0.20) -> list[tuple[float, float, float]]:
    grid = []
    values = np.round(np.arange(0.0, 0.5 + 1e-9, step), 3)
    for rho in values:
        for eta in values:
            alpha = round(1.0 - rho - eta, 3)
            if alpha >= alpha_min - 1e-9:
                grid.append((float(alpha), float(rho), float(eta)))
    return grid


def run_weight_sensitivity(portfolios: list[tuple[int, pd.DataFrame]], config: ValidationConfig, budget: float, out_dir: Path) -> pd.DataFrame:
    capacity = budget * config.capacity_per_budget
    grid = weight_grid()
    baselines = ["dsmp_direct", "greedy_cost_risk", "risk_only", "robust_topsis", "static_rcm"]
    records = []
    for seed, df in portfolios:
        oracle_loss = evaluate_plan(df, oracle_plan(df, budget, capacity))["loss_after"]
        base_regret = {}
        for method in baselines:
            ev = evaluate_plan(df, plan_for_method(df, method, budget, capacity))
            base_regret[method] = ev["loss_after"] - oracle_loss
        for alpha, rho, eta in grid:
            for variant, prob_col in [("recalibrated", "P_recalibrated"), ("uncalibrated", "P_predictive")]:
                score = (alpha * df[prob_col] * df["C_graph"] + rho * df["RTO_graph"] + eta * df["U_norm"]).clip(0.0, 1.0)
                ev = _regret_of_score(df, score, budget, capacity, oracle_loss)
                rec = {"seed": seed, "variant": variant, "alpha": alpha, "rho": rho, "eta": eta, "regret": ev["regret"], "rrpc": ev["rrpc"], "downtime_reduction": ev["downtime_reduction"]}
                for method in baselines:
                    rec[f"regret_{method}"] = base_regret[method]
                records.append(rec)
    long = pd.DataFrame(records)
    long.to_csv(out_dir / "S1_weight_sensitivity_long.csv", index=False)

    summary_rows = []
    for (variant, alpha, rho, eta), g in long.groupby(["variant", "alpha", "rho", "eta"]):
        row = {"variant": variant, "alpha": alpha, "rho": rho, "eta": eta, "n_seeds": len(g)}
        row.update({f"regret_{k}": v for k, v in _median_summary(g["regret"].to_numpy(), seed=config.seed).items()})
        for method in baselines:
            pr = _paired(g["regret"].to_numpy(), g[f"regret_{method}"].to_numpy(), seed=config.seed)
            row[f"win_vs_{method}"] = pr["win_fraction"]
            row[f"median_effect_vs_{method}"] = pr["median_effect"]
            row[f"p_vs_{method}"] = pr["p_value"]
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "S1_weight_sensitivity_summary.csv", index=False)

    # Aggregate: fraction of the grid where the recalibrated DSMP direct policy beats each baseline
    agg_rows = []
    for variant, g in summary.groupby("variant"):
        row = {"variant": variant, "n_grid_points": len(g)}
        for method in baselines:
            row[f"grid_fraction_median_better_than_{method}"] = float((g[f"median_effect_vs_{method}"] > 0).mean())
            row[f"grid_fraction_win_ge_0.5_vs_{method}"] = float((g[f"win_vs_{method}"] >= 0.5).mean())
            holm = holm_bonferroni_adjust(g[f"p_vs_{method}"].to_numpy())
            row[f"grid_fraction_better_and_holm_significant_vs_{method}"] = float(((holm < 0.05) & (g[f"median_effect_vs_{method}"].to_numpy() > 0)).mean())
            row[f"min_win_vs_{method}"] = float(g[f"win_vs_{method}"].min())
        row["min_median_regret"] = float(g["regret_median"].min())
        row["max_median_regret"] = float(g["regret_median"].max())
        best = g.loc[g["regret_median"].idxmin()]
        row["best_alpha"], row["best_rho"], row["best_eta"] = float(best["alpha"]), float(best["rho"]), float(best["eta"])
        agg_rows.append(row)
    pd.DataFrame(agg_rows).to_csv(out_dir / "S1_weight_sensitivity_aggregate.csv", index=False)

    # Figure: heatmaps over (rho, eta) for the recalibrated variant
    plot_weight_sensitivity(summary, out_dir / "figure_S1_weight_sensitivity.pdf")
    return summary


def plot_weight_sensitivity(summary: pd.DataFrame, path: Path) -> None:
    g = summary[summary["variant"] == "recalibrated"]
    rhos = np.sort(g["rho"].unique())
    etas = np.sort(g["eta"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    panels = [
        ("regret_median", "Median regret (50 seeds)", "viridis_r"),
        ("win_vs_greedy_cost_risk", "Seed fraction with lower regret\nthan greedy probability–consequence", "viridis"),
        ("win_vs_robust_topsis", "Seed fraction with lower regret\nthan robust TOPSIS", "viridis"),
    ]
    for ax, (col, title, cmap) in zip(axes, panels):
        mat = np.full((len(etas), len(rhos)), np.nan)
        for _, r in g.iterrows():
            mat[np.searchsorted(etas, r["eta"]), np.searchsorted(rhos, r["rho"])] = r[col]
        im = ax.imshow(mat, origin="lower", aspect="auto", cmap=cmap, extent=[rhos.min() - 0.025, rhos.max() + 0.025, etas.min() - 0.025, etas.max() + 0.025])
        ax.set_xlabel(r"Recovery weight $\rho$")
        ax.set_ylabel(r"Uncertainty weight $\eta$")
        ax.set_title(title, fontsize=10.5)
        ax.plot([0.25], [0.20], marker="*", color="red", markersize=13, markeredgecolor="white")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# S2: multi-seed ablation under a common action rule
# ----------------------------------------------------------------------------
ABLATION_LABELS = {
    "full_uncalibrated": "DSMP direct, uncalibrated (reference)",
    "full_recalibrated": "DSMP direct, Platt-recalibrated",
    "full_true_probability": "DSMP direct with true $P_f$",
    "no_rto": "No recovery term",
    "no_uncertainty": "No uncertainty term",
    "no_graph": "No graph augmentation",
    "conditional_rto": "Conditional recovery",
    "minmax_norm": "Portfolio min--max normalization",
    "voi_uncalibrated": "DSMP + VoI, uncalibrated",
    "threshold_no_voi": "Threshold-mapped actions, no VoI",
}


def ablation_scores(df: pd.DataFrame, config: ValidationConfig) -> dict[str, pd.Series]:
    alpha, rho, eta = validation_config_to_dsmp(config, config.seed).normalized_weights()
    a2 = alpha / (alpha + eta)
    e2 = eta / (alpha + eta)
    a3 = alpha / (alpha + rho)
    r3 = rho / (alpha + rho)
    tmp = df.copy()
    pv = tmp["P_var_delta"]
    pv_minmax = (pv - pv.min()) / (pv.max() - pv.min() + 1e-9)
    wv, wq, wm = np.asarray(config.uncertainty_weights) / np.sum(config.uncertainty_weights)
    u_minmax = (wv * pv_minmax + wq * tmp["data_quality_penalty"] + wm * tmp["model_mismatch"]).clip(0, 1)
    return {
        "full_uncalibrated": df["dynamic_score_graph_pred"],
        "full_recalibrated": df["dynamic_score_graph_recalibrated"],
        "full_true_probability": df["dynamic_score_graph_true_probability"],
        "no_rto": (a2 * df["P_predictive"] * df["C_graph"] + e2 * df["U_norm"]).clip(0, 1),
        "no_uncertainty": (a3 * df["P_predictive"] * df["C_graph"] + r3 * df["RTO_graph"]).clip(0, 1),
        "no_graph": df["dynamic_score_pred"],
        "conditional_rto": (alpha * df["P_predictive"] * df["C_graph"] + rho * df["P_predictive"] * df["RTO_graph"] + eta * df["U_norm"]).clip(0, 1),
        "minmax_norm": (alpha * df["P_predictive"] * df["C_graph"] + rho * (df["RTO_norm_pf"] + config.graph_zeta_r * df["cascade_susceptibility"]).clip(0, 1) + eta * u_minmax).clip(0, 1),
    }


def run_multiseed_ablation(portfolios: list[tuple[int, pd.DataFrame]], config: ValidationConfig, budget: float, out_dir: Path) -> pd.DataFrame:
    capacity = budget * config.capacity_per_budget
    records = []
    for seed, df in portfolios:
        oracle_loss = evaluate_plan(df, oracle_plan(df, budget, capacity))["loss_after"]
        for name, score in ablation_scores(df, config).items():
            ev = _regret_of_score(df, score, budget, capacity, oracle_loss)
            records.append({"seed": seed, "variant": name, **{k: ev[k] for k in ["regret", "risk_reduction", "downtime_reduction", "cost", "n_actions"]}})
        for name, method in [("voi_uncalibrated", "proposed_full"), ("threshold_no_voi", "no_voi")]:
            ev = evaluate_plan(df, plan_for_method(df, method, budget, capacity))
            ev["regret"] = ev["loss_after"] - oracle_loss
            records.append({"seed": seed, "variant": name, **{k: ev[k] for k in ["regret", "risk_reduction", "downtime_reduction", "cost", "n_actions"]}})
    long = pd.DataFrame(records)
    long.to_csv(out_dir / "S2_multiseed_ablation_long.csv", index=False)
    ref = long[long["variant"] == "full_uncalibrated"].set_index("seed")["regret"]
    rows = []
    for name in ABLATION_LABELS:
        g = long[long["variant"] == name].set_index("seed").loc[ref.index]
        row = {"variant": name, "label": ABLATION_LABELS[name], "n_seeds": len(g)}
        for metric in ["regret", "risk_reduction", "downtime_reduction", "cost"]:
            s = _median_summary(g[metric].to_numpy(), seed=config.seed)
            row[f"{metric}_median"] = s["median"]
            row[f"{metric}_ci_low"] = s["ci_low"]
            row[f"{metric}_ci_high"] = s["ci_high"]
        if name != "full_uncalibrated":
            pr = _paired(ref.to_numpy(), g["regret"].to_numpy(), seed=config.seed)
            row.update({f"regret_{k}": v for k, v in pr.items()})
        rows.append(row)
    summary = pd.DataFrame(rows)
    pvals = summary["regret_p_value"].to_numpy(dtype=float)
    summary["regret_holm_p"] = holm_bonferroni_adjust(pvals)
    summary.to_csv(out_dir / "S2_multiseed_ablation_summary.csv", index=False)
    return summary


# ----------------------------------------------------------------------------
# S3/S4: cross-seed recalibration, base rates and skill scores
# ----------------------------------------------------------------------------
def _calibration_row(pred: np.ndarray, f_true: np.ndarray, p_true: np.ndarray) -> dict[str, float]:
    return {
        "brier": brier_score(pred, f_true),
        "ece": expected_calibration_error(pred, f_true, n_bins=10),
        "mse_to_p_true": float(np.mean((np.asarray(pred, dtype=float) - p_true) ** 2)),
    }


def run_cross_seed_recalibration(portfolios: list[tuple[int, pd.DataFrame]], config: ValidationConfig, budget: float, out_dir: Path) -> pd.DataFrame:
    capacity = budget * config.capacity_per_budget
    seeds = [s for s, _ in portfolios]
    half = len(seeds) // 2
    folds = [(seeds[:half], seeds[half:]), (seeds[half:], seeds[:half])]
    by_seed = dict(portfolios)
    records = []
    platt_params = []
    for fold_id, (train_seeds, test_seeds) in enumerate(folds):
        train = pd.concat([by_seed[s] for s in train_seeds], ignore_index=True)
        a, b = fit_platt_scaling(train["P_predictive"], train["F_true"])
        xp, yp = fit_isotonic_regression(train["P_predictive"], train["F_true"])
        base_rate_train = float(train["F_true"].mean())
        platt_params.append({"fold": fold_id, "platt_a": float(a), "platt_b": float(b), "train_base_rate": base_rate_train, "train_seeds": f"{train_seeds[0]}-{train_seeds[-1]}", "test_seeds": f"{test_seeds[0]}-{test_seeds[-1]}"})
        print(f"fold {fold_id}: Platt a={a:.4f}, b={b:.4f}; train base rate={base_rate_train:.4f}")
        for seed in test_seeds:
            df = by_seed[seed]
            f_true = df["F_true"].to_numpy(dtype=float)
            p_true = df["p_true"].to_numpy(dtype=float)
            oracle_loss = evaluate_plan(df, oracle_plan(df, budget, capacity))["loss_after"]
            alpha, rho, eta = validation_config_to_dsmp(config, seed).normalized_weights()
            models = {
                "gamma_poisson": df["P_predictive"].to_numpy(dtype=float),
                "platt_within_seed_crossfit": df["P_platt"].to_numpy(dtype=float),
                "isotonic_within_seed_crossfit": df["P_isotonic"].to_numpy(dtype=float),
                "platt_cross_seed": apply_platt_scaling(df["P_predictive"], a, b),
                "isotonic_cross_seed": apply_isotonic_regression(df["P_predictive"], xp, yp),
                "constant_base_rate_train_seeds": np.full(len(df), base_rate_train),
                "constant_base_rate_same_seed": np.full(len(df), float(f_true.mean())),
                "true_probability": p_true,
            }
            for name, pred in models.items():
                row = {"fold": fold_id, "seed": seed, "model": name, "base_rate_train": base_rate_train, "event_rate_seed": float(f_true.mean()), "irreducible_brier": float(np.mean(p_true * (1 - p_true)))}
                row.update(_calibration_row(pred, f_true, p_true))
                # downstream decision metric for probability inputs usable by the score
                if name in {"gamma_poisson", "platt_within_seed_crossfit", "platt_cross_seed", "isotonic_cross_seed", "constant_base_rate_train_seeds", "true_probability"}:
                    score = (alpha * pd.Series(pred, index=df.index) * df["C_graph"] + rho * df["RTO_graph"] + eta * df["U_norm"]).clip(0, 1)
                    ev = _regret_of_score(df, score, budget, capacity, oracle_loss)
                    row["regret"] = ev["regret"]
                    rk = ranking_metrics(score, df["true_priority"], k=config.top_k)
                    row.update({f"rank_{k}": v for k, v in rk.items()})
                records.append(row)
    long = pd.DataFrame(records).sort_values(["model", "seed"], kind="mergesort").reset_index(drop=True)
    long.to_csv(out_dir / "S3_cross_seed_recalibration_long.csv", index=False)
    pd.DataFrame(platt_params).to_csv(out_dir / "S3_cross_seed_platt_parameters.csv", index=False)
    rows = []
    for model, g in long.groupby("model", sort=False):
        row = {"model": model, "n_test_seeds": len(g)}
        for metric in ["brier", "ece", "mse_to_p_true", "regret", "rank_spearman", "rank_top_k_capture"]:
            if metric in g and g[metric].notna().any():
                s = _median_summary(g[metric].dropna().to_numpy(), seed=config.seed)
                row[f"{metric}_median"] = s["median"]
                row[f"{metric}_ci_low"] = s["ci_low"]
                row[f"{metric}_ci_high"] = s["ci_high"]
        rows.append(row)
    summary = pd.DataFrame(rows)
    # Brier skill score relative to the cross-seed constant base-rate predictor (seed-wise, then median)
    ref = long[long["model"] == "constant_base_rate_train_seeds"].set_index("seed")["brier"]
    bss_rows = []
    for model, g in long.groupby("model", sort=False):
        gg = g.set_index("seed").sort_index()
        bss = 1.0 - gg["brier"] / ref.loc[gg.index]
        s = _median_summary(bss.to_numpy(), seed=config.seed)
        bss_rows.append({"model": model, "bss_median": s["median"], "bss_ci_low": s["ci_low"], "bss_ci_high": s["ci_high"]})
    summary = summary.merge(pd.DataFrame(bss_rows), on="model")
    summary.to_csv(out_dir / "S3_cross_seed_recalibration_summary.csv", index=False)

    base = pd.DataFrame(
        [
            {"quantity": "event_rate_median_over_seeds", "value": float(long.groupby("seed")["event_rate_seed"].first().median())},
            {"quantity": "event_rate_min_over_seeds", "value": float(long.groupby("seed")["event_rate_seed"].first().min())},
            {"quantity": "event_rate_max_over_seeds", "value": float(long.groupby("seed")["event_rate_seed"].first().max())},
            {"quantity": "irreducible_brier_median_over_seeds", "value": float(long.groupby("seed")["irreducible_brier"].first().median())},
            {"quantity": "share_p_true_below_0.05_median", "value": float(np.median([float((df["p_true"] < 0.05).mean()) for _, df in portfolios]))},
            {"quantity": "share_p_true_above_0.95_median", "value": float(np.median([float((df["p_true"] > 0.95).mean()) for _, df in portfolios]))},
            {"quantity": "share_p_true_between_median", "value": float(np.median([float(((df["p_true"] >= 0.05) & (df["p_true"] <= 0.95)).mean()) for _, df in portfolios]))},
            {"quantity": "voi_gate_open_assets_per_seed_mean", "value": float(np.mean([int((df["U_norm"] >= 0.62).sum()) for _, df in portfolios]))},
            {"quantity": "voi_gate_open_assets_per_seed_max", "value": float(np.max([int((df["U_norm"] >= 0.62).sum()) for _, df in portfolios]))},
            {"quantity": "U_norm_q90_median_over_seeds", "value": float(np.median([float(df["U_norm"].quantile(0.90)) for _, df in portfolios]))},
            {"quantity": "U_norm_max_median_over_seeds", "value": float(np.median([float(df["U_norm"].max()) for _, df in portfolios]))},
        ]
    )
    base.to_csv(out_dir / "S4_base_rate_summary.csv", index=False)
    return summary


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def run_all(out_dir: Path, config: ValidationConfig, budget: float = 12.0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    portfolios = []
    for offset in range(config.n_seeds):
        seed = config.seed + offset
        df, _ = generate_ground_truth_portfolio(seed, config)
        portfolios.append((seed, df))
    print(f"generated {len(portfolios)} portfolios in {time.perf_counter() - t0:.1f} s")
    t0 = time.perf_counter()
    run_multiseed_ablation(portfolios, config, budget, out_dir)
    print(f"S2 ablation done in {time.perf_counter() - t0:.1f} s")
    t0 = time.perf_counter()
    run_cross_seed_recalibration(portfolios, config, budget, out_dir)
    print(f"S3/S4 cross-seed recalibration done in {time.perf_counter() - t0:.1f} s")
    t0 = time.perf_counter()
    run_weight_sensitivity(portfolios, config, budget, out_dir)
    print(f"S1 weight sensitivity done in {time.perf_counter() - t0:.1f} s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run revision-stage supplementary experiments.")
    parser.add_argument("--out", type=Path, default=Path("revision_results"))
    parser.add_argument("--seed", type=int, default=ValidationConfig().seed)
    parser.add_argument("--n-assets", type=int, default=ValidationConfig().n_assets)
    parser.add_argument("--seeds", type=int, default=ValidationConfig().n_seeds)
    parser.add_argument("--mc-paths", type=int, default=ValidationConfig().mc_paths)
    parser.add_argument("--budget", type=float, default=12.0)
    args = parser.parse_args()
    config = ValidationConfig(seed=args.seed, n_assets=args.n_assets, n_seeds=args.seeds, mc_paths=args.mc_paths)
    run_all(args.out, config, budget=args.budget)


if __name__ == "__main__":
    main()
