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
S13. Information carried by the recovery time. In the generator the recovery time grows with
     the degradation state, so it is itself a condition indicator. The check repeats the
     comparison of the two reduced forms (and of the greedy rule without recovery term) when
     (a) the degradation state that enters the recovery-time model is permuted among the assets
     of each structural class, while the centrality, cascade, and lognormal delay terms stay
     with the asset (ValidationConfig.rto_degradation_coupling = "permuted"), and (b) the score
     uses a recovery time observed with multiplicative lognormal error, hours x exp(e) with e
     normal of standard deviation 0.25 or 0.50 on the log scale (the loss keeps the true one).
S14. Inspection-targeting rules of the two-epoch protocol (S8) compared at equal numbers of
     inspections, in pairs that change one element of the targeting at a time: the epistemic
     factor with the same consequence definition, or the graph augmentation of the consequence.
     A diagnostic records, per seed, the Spearman correlation across assets between the epistemic
     term and the absolute error |P_platt - p_true| of the calibrated probability.

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
from .robustness_experiments import REGIMES, cross_fitted_covariate_recalibration, two_epoch_policy, two_epoch_reference_loss
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


# ----------------------------------------------------------------------------
# S13: information carried by the recovery time
# ----------------------------------------------------------------------------
RTO_VARIANTS = ("reported generator", "RTO decoupled from degradation", "RTO observed, lognormal error 0.25", "RTO observed, lognormal error 0.50")


def _observed_rto(df: pd.DataFrame, sigma: float, seed: int) -> pd.Series:
    """Recovery time observed with multiplicative lognormal error, normalized with the reference bounds (0, 96) h."""
    rng = np.random.default_rng(seed + 60_000 + int(round(100 * sigma)))
    hours = df["delta_RTO_hours"].to_numpy(dtype=float) * np.exp(rng.normal(0.0, sigma, size=len(df)))
    return pd.Series(np.clip(hours / 96.0, 0.0, 1.0), index=df.index)


def run_recovery_information_check(config: ValidationConfig, budget: float, out_dir: Path) -> pd.DataFrame:
    capacity = budget * config.capacity_per_budget
    ratio = 0.25 / 0.55
    records = []
    for regime, scales in REGIMES.items():
        cfg = ValidationConfig(seed=config.seed, n_assets=config.n_assets, n_seeds=config.n_seeds, mc_paths=config.mc_paths, **scales)
        cfg_d = ValidationConfig(seed=config.seed, n_assets=config.n_assets, n_seeds=config.n_seeds, mc_paths=config.mc_paths,
                                 rto_degradation_coupling="permuted", **scales)
        for offset in range(cfg.n_seeds):
            seed = cfg.seed + offset
            base, _ = generate_ground_truth_portfolio(seed, cfg)
            decoupled, _ = generate_ground_truth_portfolio(seed, cfg_d)
            refs = {}
            for name, frame in (("base", base), ("decoupled", decoupled)):
                refs[name] = evaluate_plan(frame, oracle_plan(frame, budget, capacity))["loss_after"]
            probs = {
                "base": {"platt": base["P_platt"], "covariate": pd.Series(cross_fitted_covariate_recalibration(base, seed), index=base.index)},
                "decoupled": {"platt": decoupled["P_platt"], "covariate": pd.Series(cross_fitted_covariate_recalibration(decoupled, seed), index=decoupled.index)},
            }
            settings = {
                RTO_VARIANTS[0]: ("base", base["RTO_norm"]),
                RTO_VARIANTS[1]: ("decoupled", decoupled["RTO_norm"]),
                RTO_VARIANTS[2]: ("base", _observed_rto(base, 0.25, seed)),
                RTO_VARIANTS[3]: ("base", _observed_rto(base, 0.50, seed)),
            }
            for variant, (which, rto_used) in settings.items():
                frame = base if which == "base" else decoupled
                row0 = {"regime": regime, "seed": seed, "variant": variant,
                        "spearman_rto_used_p_true": float(spearmanr(rto_used, frame["p_true"]).statistic),
                        "spearman_rto_used_x": float(spearmanr(rto_used, frame["X_current"]).statistic)}
                for pname, p in probs[which].items():
                    row = dict(row0, probability=pname)
                    scores = {
                        "sys": reduced_score(p, frame["C_norm"], rto_used, ratio, "sys"),
                        "exp": reduced_score(p, frame["C_norm"], rto_used, ratio, "exp"),
                        "no_recovery": (p * frame["C_norm"]).clip(0.0, 1.0),
                    }
                    for form, score in scores.items():
                        ev = evaluate_plan(frame, _direct_plan_from_score(frame, score, budget, capacity))
                        row[f"regret_{form}"] = ev["loss_after"] - refs[which]
                    records.append(row)
        print(f"  S13 regime {regime} done")
    long = pd.DataFrame(records)
    long.to_csv(out_dir / "S13_recovery_information_long.csv", index=False)
    rows = []
    for (variant, regime, prob), g in long.groupby(["variant", "regime", "probability"], sort=False):
        g = g.sort_values("seed")
        row = {"variant": variant, "regime": regime, "probability": prob, "n_seeds": len(g),
               "spearman_rto_used_p_true": float(g["spearman_rto_used_p_true"].median()),
               "spearman_rto_used_x": float(g["spearman_rto_used_x"].median())}
        for form in ("sys", "exp", "no_recovery"):
            s = _median_summary(g[f"regret_{form}"].to_numpy(), seed=config.seed)
            row[f"regret_{form}_median"], row[f"regret_{form}_ci_low"], row[f"regret_{form}_ci_high"] = s["median"], s["ci_low"], s["ci_high"]
        for other in ("exp", "no_recovery"):
            pr = _paired(g["regret_sys"].to_numpy(), g[f"regret_{other}"].to_numpy(), seed=config.seed)
            row[f"effect_{other}_minus_sys"], row[f"effect_ci_low_{other}"], row[f"effect_ci_high_{other}"] = pr["median_effect"], pr["effect_ci_low"], pr["effect_ci_high"]
            row[f"p_{other}"], row[f"sys_wins_vs_{other}"] = pr["p_value"], pr["win_fraction"]
        rows.append(row)
    summary = pd.DataFrame(rows)
    for variant in summary["variant"].unique():
        mask = summary["variant"] == variant
        for other in ("exp", "no_recovery"):
            summary.loc[mask, f"holm_p_{other}"] = holm_bonferroni_adjust(summary.loc[mask, f"p_{other}"].to_numpy(dtype=float))
    summary.to_csv(out_dir / "S13_recovery_information_summary.csv", index=False)
    return summary


# ----------------------------------------------------------------------------
# S14: inspection-targeting rules compared at equal numbers of inspections
# ----------------------------------------------------------------------------
# Two-epoch protocol of S8 (robustness_experiments.two_epoch_policy, same portfolios, resources,
# update, and evaluation). The rules are paired so that each comparison changes one element of
# the targeting: the epistemic factor (U x C against C with the same consequence definition), the
# graph augmentation of the consequence (C^G against C), or targeting against random selection.
# The rules of S8 are recomputed here with the same random streams, so their rows coincide with
# those of S8_two_epoch_voi_long.csv.
INSPECTION_RULES = [
    ("inspect_top30_consequence", "top_consequence", 30),
    ("inspect_top60_consequence", "top_consequence", 60),
    ("inspect_top30_consequence_no_graph", "top_consequence_no_graph", 30),
    ("inspect_top60_consequence_no_graph", "top_consequence_no_graph", 60),
    ("inspect_top30_uncertainty_x_consequence", "top_uncertainty_consequence", 30),
    ("inspect_top60_uncertainty_x_consequence", "top_uncertainty_consequence", 60),
    ("inspect_top30_uncertainty_x_graph_consequence", "top_uncertainty_graph_consequence", 30),
    ("inspect_top60_uncertainty_x_graph_consequence", "top_uncertainty_graph_consequence", 60),
    ("inspect_random60", "random", 60),
]
# (rule, compared_with, contrast); effect = regret(rule) - regret(compared_with), negative = rule better.
INSPECTION_PAIRS = [
    ("inspect_top30_uncertainty_x_consequence", "inspect_top30_consequence_no_graph", "epistemic factor, consequence without graph terms"),
    ("inspect_top60_uncertainty_x_consequence", "inspect_top60_consequence_no_graph", "epistemic factor, consequence without graph terms"),
    ("inspect_top30_uncertainty_x_graph_consequence", "inspect_top30_consequence", "epistemic factor, graph-augmented consequence"),
    ("inspect_top60_uncertainty_x_graph_consequence", "inspect_top60_consequence", "epistemic factor, graph-augmented consequence"),
    ("inspect_top30_consequence", "inspect_top30_consequence_no_graph", "graph augmentation of the consequence"),
    ("inspect_top60_consequence", "inspect_top60_consequence_no_graph", "graph augmentation of the consequence"),
    ("inspect_top60_consequence", "inspect_random60", "targeting against random selection"),
    ("inspect_top60_consequence_no_graph", "inspect_random60", "targeting against random selection"),
    ("inspect_top60_uncertainty_x_consequence", "inspect_random60", "targeting against random selection"),
    ("inspect_top60_uncertainty_x_graph_consequence", "inspect_random60", "targeting against random selection"),
]


def run_inspection_rules(config: ValidationConfig, budget: float, out_dir: Path) -> pd.DataFrame:
    """Run the targeting rules of INSPECTION_RULES in the two-epoch protocol of S8 (reported regime)."""
    capacity = budget * config.capacity_per_budget
    portfolios = [(config.seed + k, generate_ground_truth_portfolio(config.seed + k, config)[0]) for k in range(config.n_seeds)]
    records = []
    for resource_model in ("shared", "budget_only", "dedicated"):
        for sigma_obs in (0.05, 0.20):
            for seed, df in portfolios:
                reference_total = two_epoch_reference_loss(df, budget, capacity)
                for name, rule, param in INSPECTION_RULES:
                    res = two_epoch_policy(df, config, seed, budget, capacity, rule, param, sigma_obs,
                                           np.random.default_rng(seed + 9001), resource_model=resource_model)
                    res.update({"policy": name, "rule": rule, "param": param, "sigma_obs": sigma_obs, "resource_model": resource_model,
                                "seed": seed, "regret_total": res["total_loss"] - reference_total, "oracle_total": reference_total})
                    records.append(res)
        print(f"  S14 resource model {resource_model} done")
    long = pd.DataFrame(records)
    long.to_csv(out_dir / "S14_inspection_rules_long.csv", index=False)
    return long


def run_uncertainty_error_association(config: ValidationConfig, out_dir: Path) -> pd.DataFrame:
    """Diagnostic of S14: per-seed Spearman correlations across assets (reported regime) between the
    epistemic term U_norm, the degradation state X, and the error of the Platt-calibrated
    probability relative to the simulated true probability, P_platt - p_true (absolute and signed)."""
    rows = []
    for k in range(config.n_seeds):
        seed = config.seed + k
        df, _ = generate_ground_truth_portfolio(seed, config)
        signed = df["P_platt"] - df["p_true"]
        rows.append({"seed": seed,
                     "spearman_U_abs_error": float(spearmanr(df["U_norm"], signed.abs()).statistic),
                     "spearman_U_signed_error": float(spearmanr(df["U_norm"], signed).statistic),
                     "spearman_X_abs_error": float(spearmanr(df["X_current"], signed.abs()).statistic),
                     "spearman_U_X": float(spearmanr(df["U_norm"], df["X_current"]).statistic)})
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "S14_uncertainty_error_association.csv", index=False)
    return out


def compare_inspection_rules(long: pd.DataFrame, out_dir: Path, seed: int = 42) -> pd.DataFrame:
    """Paired comparisons of INSPECTION_PAIRS (same number of inspected assets, same resources), with
    Holm adjustment within each resource model and noise level; also writes the per-rule medians."""
    rows, summary = [], []
    for (resource_model, sigma_obs), g in long.groupby(["resource_model", "sigma_obs"], sort=False):
        piv = g.pivot(index="seed", columns="policy", values="regret_total").sort_index()
        for name, _, _ in INSPECTION_RULES:
            s = _median_summary(piv[name].to_numpy(), seed=seed)
            summary.append({"resource_model": resource_model, "sigma_obs": sigma_obs, "policy": name, "n_seeds": len(piv),
                            "regret_median": s["median"], "regret_ci_low": s["ci_low"], "regret_ci_high": s["ci_high"]})
        block = []
        for a, b, contrast in INSPECTION_PAIRS:
            pr = _paired(piv[b].to_numpy(), piv[a].to_numpy(), seed=seed)
            block.append({"resource_model": resource_model, "sigma_obs": sigma_obs, "contrast": contrast, "rule": a, "compared_with": b,
                          "regret_rule_median": float(piv[a].median()), "regret_compared_median": float(piv[b].median()),
                          "effect_rule_minus_compared": pr["median_effect"], "effect_ci_low": pr["effect_ci_low"], "effect_ci_high": pr["effect_ci_high"],
                          "p_value": pr["p_value"], "rule_better_fraction": float(np.mean(piv[a].to_numpy() < piv[b].to_numpy()))})
        block = pd.DataFrame(block)
        block["holm_p"] = holm_bonferroni_adjust(block["p_value"].to_numpy(dtype=float))
        rows.append(block)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(out_dir / "S14_inspection_rules_equal_n.csv", index=False)
    pd.DataFrame(summary).to_csv(out_dir / "S14_inspection_rules_summary.csv", index=False)
    return out


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
    if "S13" not in skip:
        t0 = time.perf_counter()
        run_recovery_information_check(config, budget, out_dir)
        print(f"S13 recovery-information check done in {time.perf_counter() - t0:.1f} s")
    if "S14" not in skip:
        t0 = time.perf_counter()
        compare_inspection_rules(run_inspection_rules(config, budget, out_dir), out_dir, seed=config.seed)
        run_uncertainty_error_association(config, out_dir)
        print(f"S14 inspection-rule comparison done in {time.perf_counter() - t0:.1f} s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the second-revision checks (S11-S14).")
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
