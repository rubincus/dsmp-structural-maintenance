"""Checks added at the second revision stage (JRR-26-0219.R1), Sections 3.4 and 6.6.

S11. Reduced score forms (eta = 0, no graph terms) under imperfect and exact failure
     probabilities. The epistemic term is removed and the remaining weights are renormalized,
     alpha' = alpha / (alpha + rho) and rho' = rho / (alpha + rho), so that only the ratio
     rho'/alpha' matters for the benefit-per-cost ordering (0.25/0.55 = 0.455 for the reported
     weights). The check records, for the system-exposure form (Eq. 16) and the expected-loss
     form (Eq. 17): the regret relative to the perfect-information greedy reference, the overlap
     of the selected asset set with the reference set, the share of selected assets with
     P_true > 0.95, and the sensitivity of the regret to the ratio rho'/alpha'; for the Platt
     proxy, the condition-informed probability, and the simulated true probability in the five
     hazard regimes, and along the probability-quality sweep of the reported regime.
S12. Additivity check of the score, Eq. (16): interaction-aware form, Eq. (18), with
     risk--uncertainty (chi) and risk--recovery (psi) interaction weights, divided by
     (1 + chi + psi) so that it stays in [0, 1]; and the rank dependence among the three
     attributes of the score.

All defaults of the validation protocol are unchanged; the reported regime is
(degradation_scale, demand_scale) = (1, 1).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from .revision_experiments import _direct_plan_from_score, _median_summary, _paired
from .robustness_experiments import REGIMES, cross_fitted_covariate_recalibration
from .validation_experiments import (
    ValidationConfig,
    evaluate_plan,
    generate_ground_truth_portfolio,
    oracle_plan,
    validation_config_to_dsmp,
)
from .validation_metrics import holm_bonferroni_adjust

# Recovery-to-consequence weight ratios rho'/alpha' of the reduced forms. 0.20 is the weight of
# failure-conditioned recovery in the ground-truth loss (Table 4); 0.224 is the ratio implied by
# the true benefit of a repair, (1 - 0.85 e^{-0.8225}) * 0.20 / (1 - e^{-0.8225}); 0.455 is the
# ratio of the reported weights after removing the epistemic term.
_REPAIR_RATIO = 0.20 * (1.0 - 0.85 * np.exp(-2.35 * 0.35)) / (1.0 - np.exp(-2.35 * 0.35))
RATIOS = {
    "0.20": 0.20,
    "0.224": float(_REPAIR_RATIO),
    "0.455": 0.25 / 0.55,
    "1.0": 1.0,
    "2.0": 2.0,
}
INTERACTION_LEVELS = (0.0, 0.1, 0.25, 0.5)


def reduced_score(p: pd.Series, c: pd.Series, rto: pd.Series, ratio: float, form: str) -> pd.Series:
    """Eq. (16) (form='sys') or Eq. (17) (form='exp') with eta = 0 and weights alpha' + rho' = 1."""
    a = 1.0 / (1.0 + ratio)
    b = ratio / (1.0 + ratio)
    if form == "sys":
        return (a * p * c + b * rto).clip(0.0, 1.0)
    return (a * p * c + b * p * rto).clip(0.0, 1.0)


def _plan_diagnostics(df: pd.DataFrame, plan: pd.DataFrame, ref_assets: set) -> dict[str, float]:
    assets = set(plan["asset_id"]) if len(plan) else set()
    p = df.set_index("asset_id")["p_true"]
    sel = p.loc[sorted(assets)] if assets else pd.Series(dtype=float)
    union = assets | ref_assets
    return {
        "n_actions": len(assets),
        "share_selected_p_true_gt_095": float((sel > 0.95).mean()) if len(sel) else np.nan,
        "mean_p_true_selected": float(sel.mean()) if len(sel) else np.nan,
        "jaccard_with_reference": float(len(assets & ref_assets) / len(union)) if union else 1.0,
    }


# ----------------------------------------------------------------------------
# S11: reduced forms, weights, and plan diagnostics
# ----------------------------------------------------------------------------
def run_reduced_forms(config: ValidationConfig, budget: float, out_dir: Path) -> pd.DataFrame:
    capacity = budget * config.capacity_per_budget
    records = []
    for regime, scales in REGIMES.items():
        cfg = ValidationConfig(seed=config.seed, n_assets=config.n_assets, n_seeds=config.n_seeds, mc_paths=config.mc_paths, **scales)
        for offset in range(cfg.n_seeds):
            seed = cfg.seed + offset
            df, _ = generate_ground_truth_portfolio(seed, cfg)
            ref = oracle_plan(df, budget, capacity)
            ref_ev = evaluate_plan(df, ref)
            ref_assets = set(ref["asset_id"])
            p_true = df["p_true"]
            ref_sel = p_true[df["asset_id"].isin(ref_assets)]
            base = {
                "regime": regime, "seed": seed, "event_rate": float(df["F_true"].mean()),
                "share_p_true_gt_095_portfolio": float((p_true > 0.95).mean()),
                "share_p_true_lt_005_portfolio": float((p_true < 0.05).mean()),
                "reference_n_actions": len(ref_assets),
                "reference_share_selected_p_true_gt_095": float((ref_sel > 0.95).mean()),
            }
            probs = {
                "platt": df["P_platt"],
                "covariate": pd.Series(cross_fitted_covariate_recalibration(df, seed), index=df.index),
                "true": p_true,
            }
            if regime.startswith("R76"):
                for w in (0.25, 0.5, 0.75):
                    probs[f"w{w:.2f}"] = (w * p_true + (1.0 - w) * df["P_platt"]).clip(0.0, 1.0)
            for pname, p in probs.items():
                for rlabel, ratio in RATIOS.items():
                    for form in ("sys", "exp"):
                        score = reduced_score(p, df["C_norm"], df["RTO_norm"], ratio, form)
                        plan = _direct_plan_from_score(df, score, budget, capacity)
                        ev = evaluate_plan(df, plan)
                        records.append({**base, "probability": pname, "ratio_label": rlabel, "ratio": ratio, "form": form,
                                        "regret": ev["loss_after"] - ref_ev["loss_after"], **_plan_diagnostics(df, plan, ref_assets)})
        print(f"  S11 regime {regime} done")
    long = pd.DataFrame(records)
    long.to_csv(out_dir / "S11_reduced_forms_long.csv", index=False)

    rows = []
    keys = ["regime", "probability", "ratio_label"]
    for (regime, prob, rlabel), g in long.groupby(keys, sort=False):
        sys_ = g[g["form"] == "sys"].set_index("seed").sort_index()
        exp_ = g[g["form"] == "exp"].set_index("seed").sort_index()
        row = {"regime": regime, "probability": prob, "ratio_label": rlabel, "ratio": float(g["ratio"].iloc[0]), "n_seeds": len(sys_),
               "event_rate": float(sys_["event_rate"].median()),
               "share_p_true_gt_095_portfolio": float(sys_["share_p_true_gt_095_portfolio"].median()),
               "reference_share_selected_p_true_gt_095": float(sys_["reference_share_selected_p_true_gt_095"].median())}
        for form, d in (("sys", sys_), ("exp", exp_)):
            s = _median_summary(d["regret"].to_numpy(), seed=config.seed)
            row[f"regret_{form}_median"], row[f"regret_{form}_ci_low"], row[f"regret_{form}_ci_high"] = s["median"], s["ci_low"], s["ci_high"]
            row[f"jaccard_{form}_median"] = float(d["jaccard_with_reference"].median())
            row[f"share_selected_p_true_gt_095_{form}_median"] = float(d["share_selected_p_true_gt_095"].median())
            row[f"n_actions_{form}_median"] = float(d["n_actions"].median())
        pr = _paired(sys_["regret"].to_numpy(), exp_.loc[sys_.index, "regret"].to_numpy(), seed=config.seed)
        row["effect_exp_minus_sys"], row["effect_ci_low"], row["effect_ci_high"] = pr["median_effect"], pr["effect_ci_low"], pr["effect_ci_high"]
        row["p_value"], row["sys_wins_fraction"] = pr["p_value"], pr["win_fraction"]
        rows.append(row)
    summary = pd.DataFrame(rows)
    for regime in summary["regime"].unique():
        mask = summary["regime"] == regime
        summary.loc[mask, "holm_p"] = holm_bonferroni_adjust(summary.loc[mask, "p_value"].to_numpy(dtype=float))
    summary.to_csv(out_dir / "S11_reduced_forms_summary.csv", index=False)
    return summary


# ----------------------------------------------------------------------------
# S12: additivity check with the interaction-aware form, Eq. (18)
# ----------------------------------------------------------------------------
def run_additivity_check(config: ValidationConfig, budget: float, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    capacity = budget * config.capacity_per_budget
    records, corr = [], []
    for offset in range(config.n_seeds):
        seed = config.seed + offset
        df, _ = generate_ground_truth_portfolio(seed, config)
        ref_loss = evaluate_plan(df, oracle_plan(df, budget, capacity))["loss_after"]
        alpha, rho, eta = validation_config_to_dsmp(config, seed).normalized_weights()
        a2, r2 = alpha / (alpha + rho), rho / (alpha + rho)
        p_cov = pd.Series(cross_fitted_covariate_recalibration(df, seed), index=df.index)
        configurations = {
            "reported (count proxy, graph, eta = 0.20)": dict(p=df["P_platt"], c=df["C_graph"], rto=df["RTO_graph"], a=alpha, r=rho, e=eta),
            "reduced (condition-informed P, eta = 0, no graph)": dict(p=p_cov, c=df["C_norm"], rto=df["RTO_norm"], a=a2, r=r2, e=0.0),
        }
        u = df["U_norm"]
        for cname, k in configurations.items():
            additive = (k["a"] * k["p"] * k["c"] + k["r"] * k["rto"] + k["e"] * u).clip(0.0, 1.0)
            add_plan = _direct_plan_from_score(df, additive, budget, capacity)
            add_assets = set(add_plan["asset_id"])
            for chi in INTERACTION_LEVELS:
                for psi in INTERACTION_LEVELS:
                    score = (additive + chi * k["p"] * k["c"] * u + psi * k["p"] * k["c"] * k["rto"]) / (1.0 + chi + psi)
                    plan = _direct_plan_from_score(df, score, budget, capacity)
                    assets = set(plan["asset_id"])
                    ev = evaluate_plan(df, plan)
                    union = assets | add_assets
                    records.append({
                        "configuration": cname, "seed": seed, "chi": chi, "psi": psi,
                        "regret": ev["loss_after"] - ref_loss,
                        "kendall_tau_with_additive": float(kendalltau(additive, score).statistic),
                        "jaccard_with_additive_plan": float(len(assets & add_assets) / len(union)) if union else 1.0,
                    })
        pc = df["P_platt"] * df["C_graph"]
        corr.append({
            "seed": seed,
            "spearman_PC_RTO": float(spearmanr(pc, df["RTO_graph"]).statistic),
            "spearman_PC_U": float(spearmanr(pc, u).statistic),
            "spearman_RTO_U": float(spearmanr(df["RTO_graph"], u).statistic),
            "spearman_Pcov_C_RTO": float(spearmanr(p_cov * df["C_norm"], df["RTO_norm"]).statistic),
            "spearman_Pcov_C_U": float(spearmanr(p_cov * df["C_norm"], u).statistic),
            "spearman_Coperational_RTO": float(spearmanr(df["C_operational"], df["RTO_norm"]).statistic),
            "spearman_C_RTO": float(spearmanr(df["C_norm"], df["RTO_norm"]).statistic),
        })
    long = pd.DataFrame(records)
    long.to_csv(out_dir / "S12_additivity_long.csv", index=False)
    rows = []
    for cname, g in long.groupby("configuration", sort=False):
        add = g[(g["chi"] == 0.0) & (g["psi"] == 0.0)].set_index("seed").sort_index()
        block = []
        for (chi, psi), d in g.groupby(["chi", "psi"]):
            d = d.set_index("seed").loc[add.index]
            s = _median_summary(d["regret"].to_numpy(), seed=config.seed)
            row = {"configuration": cname, "chi": chi, "psi": psi, "n_seeds": len(d), "regret_median": s["median"], "regret_ci_low": s["ci_low"], "regret_ci_high": s["ci_high"],
                   "kendall_tau_median": float(d["kendall_tau_with_additive"].median()), "kendall_tau_min": float(d["kendall_tau_with_additive"].min()),
                   "jaccard_median": float(d["jaccard_with_additive_plan"].median()), "identical_plan_share": float((d["jaccard_with_additive_plan"] >= 1.0 - 1e-12).mean())}
            if chi == 0.0 and psi == 0.0:
                row.update({"effect_vs_additive": 0.0, "effect_ci_low": 0.0, "effect_ci_high": 0.0, "p_value": np.nan, "interaction_wins_fraction": np.nan})
            else:
                pr = _paired(add["regret"].to_numpy(), d["regret"].to_numpy(), seed=config.seed)
                row.update({"effect_vs_additive": pr["median_effect"], "effect_ci_low": pr["effect_ci_low"], "effect_ci_high": pr["effect_ci_high"],
                            "p_value": pr["p_value"], "interaction_wins_fraction": float(np.mean(d["regret"].to_numpy() < add["regret"].to_numpy()))})
            block.append(row)
        block = pd.DataFrame(block)
        block["holm_p"] = holm_bonferroni_adjust(block["p_value"].to_numpy(dtype=float))
        rows.append(block)
    summary = pd.concat(rows, ignore_index=True)
    summary.to_csv(out_dir / "S12_additivity_summary.csv", index=False)
    corr = pd.DataFrame(corr)
    corr.to_csv(out_dir / "S12_attribute_dependence_long.csv", index=False)
    corr_summary = corr.drop(columns="seed").describe(percentiles=[0.25, 0.5, 0.75]).T
    corr_summary.to_csv(out_dir / "S12_attribute_dependence_summary.csv")
    return summary, corr_summary


def run_all(out_dir: Path, config: ValidationConfig, budget: float = 12.0, skip: tuple[str, ...] = ()) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if "S11" not in skip:
        t0 = time.perf_counter()
        run_reduced_forms(config, budget, out_dir)
        print(f"S11 reduced forms done in {time.perf_counter() - t0:.1f} s")
    if "S12" not in skip:
        t0 = time.perf_counter()
        run_additivity_check(config, budget, out_dir)
        print(f"S12 additivity check done in {time.perf_counter() - t0:.1f} s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the second-revision checks (S11-S12).")
    parser.add_argument("--out", type=Path, default=Path("revision2_results"))
    parser.add_argument("--seed", type=int, default=ValidationConfig().seed)
    parser.add_argument("--n-assets", type=int, default=ValidationConfig().n_assets)
    parser.add_argument("--seeds", type=int, default=ValidationConfig().n_seeds)
    parser.add_argument("--mc-paths", type=int, default=ValidationConfig().mc_paths)
    parser.add_argument("--budget", type=float, default=12.0)
    parser.add_argument("--skip", type=str, default="", help="comma-separated experiment ids to skip, e.g. S12")
    args = parser.parse_args()
    config = ValidationConfig(seed=args.seed, n_assets=args.n_assets, n_seeds=args.seeds, mc_paths=args.mc_paths)
    run_all(args.out, config, budget=args.budget, skip=tuple(x for x in args.skip.split(",") if x))


if __name__ == "__main__":
    main()
