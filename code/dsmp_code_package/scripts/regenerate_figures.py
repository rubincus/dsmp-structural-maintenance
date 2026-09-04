"""Regenerate validation figures from the stored 50-seed result tables.

Uses one consistent policy naming convention, larger legend/axis fonts, and no raw
code identifiers in legends.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11.5,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
    }
)

# Single naming convention shared by tables, figures, and the response letters.
LABELS = {
    "proposed_full": "DSMP + VoI, uncalibrated",
    "proposed_recalibrated": "DSMP + VoI, Platt-recalibrated",
    "proposed_true_probability": "DSMP + VoI with true $P_f$",
    "dsmp_direct": "DSMP direct, uncalibrated",
    "dsmp_direct_recalibrated": "DSMP direct, Platt-recalibrated",
    "dsmp_direct_true_probability": "DSMP direct with true $P_f$",
    "no_voi": "No VoI gate (threshold-mapped)",
    "conditional_rto": "Conditional recovery",
    "risk_only": "Probability--consequence score",
    "severity_only": "Severity-only",
    "consequence_only": "Consequence-only",
    "static_rcm": "Static RCM-like",
    "voi_only": "VoI-only inspection",
    "greedy_cost_risk": "Greedy probability--consequence",
    "robust_topsis": "Robust TOPSIS MCDA",
    "true_probability_upper_bound": "True-probability upper bound",
}
ORDER = list(LABELS)


def label(method: str) -> str:
    return LABELS.get(method, method.replace("_", " ")).replace("--", "–")


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(tables: Path, path: Path) -> None:
    cal = pd.read_csv(tables / "calibration_bins.csv")
    names = {"gamma poisson": "Gamma–Poisson predictive", "platt recalibrated": "Platt recalibration (out-of-fold)", "isotonic recalibrated": "Isotonic recalibration (out-of-fold)"}
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", lw=1, label="Perfect calibration")
    for model in ["gamma poisson", "platt recalibrated", "isotonic recalibrated"]:
        g = cal[cal["model"] == model].groupby("bin", as_index=False).agg(pred_mean=("pred_mean", "mean"), observed_rate=("observed_rate", "mean"), n=("n", "sum"))
        g = g[g["n"] > 0]
        ax.plot(g["pred_mean"], g["observed_rate"], marker="o", lw=1.8, label=names[model])
    ax.set_xlabel("Mean predicted probability (bin)")
    ax.set_ylabel("Observed failure frequency (bin)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=True)
    _save(fig, path)


def plot_regret(tables: Path, path: Path) -> None:
    m = pd.read_csv(tables / "validation_metrics_long.csv").dropna(subset=["regret"])
    styles = {
        "dsmp_direct_recalibrated": dict(color="#d62728", lw=2.6, ls="-", marker="o"),
        "dsmp_direct": dict(color="#ff7f0e", lw=2.0, ls="-", marker="o"),
        "proposed_full": dict(color="#1f77b4", lw=2.0, ls="-", marker="s"),
        "proposed_recalibrated": dict(color="#1f77b4", lw=1.6, ls="--", marker="s"),
        "greedy_cost_risk": dict(color="#2ca02c", lw=1.8, ls="-", marker="^"),
        "risk_only": dict(color="#2ca02c", lw=1.6, ls="--", marker="^"),
        "robust_topsis": dict(color="#9467bd", lw=1.8, ls="-", marker="D"),
        "conditional_rto": dict(color="#8c564b", lw=1.4, ls="-.", marker="v"),
        "no_voi": dict(color="#e377c2", lw=1.4, ls="-.", marker="v"),
        "static_rcm": dict(color="#7f7f7f", lw=1.6, ls=":", marker="x"),
        "severity_only": dict(color="#bcbd22", lw=1.4, ls=":", marker="x"),
        "consequence_only": dict(color="#17becf", lw=1.4, ls=":", marker="x"),
        "voi_only": dict(color="black", lw=1.2, ls=":", marker="+"),
        "dsmp_direct_true_probability": dict(color="#d62728", lw=1.6, ls="--", marker="*"),
        "proposed_true_probability": dict(color="#1f77b4", lw=1.2, ls=":", marker="*"),
        "true_probability_upper_bound": dict(color="black", lw=1.6, ls="--", marker="*"),
    }
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for method in ORDER:
        if method == "voi_only":  # zero direct risk reduction; regret 11.6--28 dominates the axis (reported in the table)
            continue
        g = m[m["method"] == method].groupby("budget", as_index=False)["regret"].median()
        if g.empty:
            continue
        lab = label(method) + (", threshold-mapped" if method == "static_rcm" else "")  # common-rule values are given in the text (revision)
        ax.plot(g["budget"], g["regret"], label=lab, markersize=5, **styles.get(method, {}))
    ax.set_xlabel("Budget level (replacement-equivalent cost units)")
    ax.set_ylabel("Median regret over 50 seeds\n(normalized expected-loss units)")
    ax.set_xticks([8, 12, 16, 20])
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, ncol=1)
    _save(fig, path)


def plot_rank_metrics(tables: Path, path: Path) -> None:
    r = pd.read_csv(tables / "rank_metrics.csv")
    g = r.groupby("method").agg(top_k_capture=("top_k_capture", "median"), spearman=("spearman", "median"), kendall=("kendall", "median"))
    g = g.loc[[m for m in ORDER if m in g.index]]
    y = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    ax.barh(y - 0.27, g["top_k_capture"], height=0.26, label="Top-$k$ capture ($k=30$)")
    ax.barh(y, g["spearman"], height=0.26, label="Spearman correlation")
    ax.barh(y + 0.27, g["kendall"], height=0.26, label="Kendall correlation")
    ax.set_yticks(y, [label(m) for m in g.index])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Median value over 50 seeds (against true preventable-loss ranking)")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", frameon=True)
    _save(fig, path)


def plot_voi_map(path: Path) -> None:
    u = np.linspace(0, 1, 240)
    cost = np.linspace(0.0, 0.25, 240)
    U, C = np.meshgrid(u, cost)
    nvi = 0.55 * U * 0.70 * 0.25 - C
    route = (U >= 0.62) & (nvi > 0.0)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.contourf(U, C, route.astype(float), levels=[-0.5, 0.5, 1.5], colors=["#dfe6ee", "#3b7dd8"], alpha=0.9)
    ax.axvline(0.62, color="black", ls="--", lw=1.2)
    ax.plot(u, np.clip(0.55 * u * 0.70 * 0.25, 0, None), color="black", lw=1.4)
    ax.text(0.64, 0.225, r"$u^\star = 0.62$", fontsize=10)
    ax.text(0.80, 0.03, "Information action\n(inspect / monitor)", color="white", fontsize=10, ha="center")
    ax.text(0.30, 0.15, "Direct intervention\nor defer", fontsize=10, ha="center")
    ax.set_xlabel(r"Epistemic uncertainty $\widetilde{U}_i$")
    ax.set_ylabel(r"Normalized information cost $\kappa_e$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.25)
    _save(fig, path)


def plot_rank_reversal(tables: Path, path: Path) -> None:
    s = pd.read_csv(tables / "figure_v8_rank_reversal_stress.csv")
    names = {"reference_based": "Reference-based\nnormalization", "portfolio_minmax": "Portfolio min–max\nnormalization"}
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    bars = ax.bar([names[n] for n in s["normalization"]], s["rank_reversal_frequency"], color=["#2a9d8f", "#e76f51"], width=0.55)
    for b, v in zip(bars, s["rank_reversal_frequency"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.4f}", ha="center", fontsize=10)
    ax.set_ylim(0, max(0.05, float(s["rank_reversal_frequency"].max()) * 1.3))
    ax.set_ylabel("Pairwise rank-reversal frequency")
    _save(fig, path)


def plot_pareto(tables: Path, path: Path) -> None:
    front = pd.read_csv(tables / "figure_v7_pareto_front.csv")
    fig, ax = plt.subplots(figsize=(6.6, 4.9))
    sc = ax.scatter(front["cost"], front["residual_risk"], c=front["residual_uncertainty"], s=48, cmap="viridis", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Intervention cost (replacement-equivalent units)")
    ax.set_ylabel("Residual risk (normalized expected-loss units)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Residual epistemic uncertainty (sum over assets)")
    ax.grid(alpha=0.25)
    _save(fig, path)


def plot_bayesian_update(path: Path) -> None:
    from scipy.special import gammaln

    def gamma_pdf(x, a, b):
        return np.exp(a * np.log(b) - gammaln(a) + (a - 1.0) * np.log(x) - b * x)

    x = np.linspace(0.001, 2.0, 600)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(x, gamma_pdf(x, 2, 4), lw=2, label="Prior Gamma(2, 4)")
    ax.plot(x, gamma_pdf(x, 12, 10), lw=2, label="Posterior Gamma(12, 10)")
    ax.axvline(2 / 4, linestyle="--", color="C0", label="Prior mean")
    ax.axvline(12 / 10, linestyle="--", color="C1", label="Posterior mean")
    ax.set_xlabel(r"Failure-occurrence rate $\lambda$ (events/year)")
    ax.set_ylabel("Density")
    ax.legend(frameon=True)
    _save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--methodological", action="store_true", help="also regenerate Figures 3 and 4 from the seed-42 validation portfolio")
    args = parser.parse_args()
    out = args.out
    if args.methodological:
        plot_methodological_figures(out)
    plot_calibration(args.tables, out / "validation" / "figure_v2_calibration_curve.pdf")
    plot_regret(args.tables, out / "validation" / "figure_v3_regret_vs_budget.pdf")
    plot_rank_metrics(args.tables, out / "validation" / "figure_v4_topk_rank_comparison.pdf")
    plot_voi_map(out / "validation" / "figure_v5_voi_gate_decision_map.pdf")
    plot_rank_reversal(args.tables, out / "validation" / "figure_v8_rank_reversal_frequency.pdf")
    plot_pareto(args.tables, out / "validation" / "figure_v7_pareto_front.pdf")
    plot_bayesian_update(out / "fig05_bayesian_update_gamma_poisson.pdf")
    print("figures written to", out)


def plot_methodological_figures(out: Path) -> None:
    """Regenerate the single-asset weight surface and the rank-stability check (Figures 3 and 4)
    from the first validation portfolio (seed 42) with legible labels."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from dsmp.sensitivity import lambda_eta_surface, rank_stability
    from dsmp.validation_experiments import ValidationConfig, generate_ground_truth_portfolio

    df, _ = generate_ground_truth_portfolio(42, ValidationConfig())
    row = df.sort_values("dynamic_score", ascending=False).iloc[0]
    surface = lambda_eta_surface(row)
    pivot = surface.pivot_table(index="eta", columns="rho", values="score")
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    im = ax.imshow(pivot.to_numpy(), origin="lower", aspect="auto", extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()])
    ax.set_xlabel(r"Recovery weight $\rho$")
    ax.set_ylabel(r"Uncertainty weight $\eta$")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r"Risk--resilience score $\widetilde{R}_i$ ($\alpha = 1-\rho-\eta$)")
    _save(fig, out / "fig09_lambda_eta_sensitivity.pdf")

    stab = rank_stability(df, seed=123)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(stab["sigma"], stab["mean_rank_correlation"], marker="o", lw=2, label="Mean Spearman correlation")
    ax.fill_between(stab["sigma"], stab["mean_rank_correlation"] - stab["std_rank_correlation"], stab["mean_rank_correlation"] + stab["std_rank_correlation"], alpha=0.25, label=r"$\pm$ one standard deviation (250 replications)")
    ax.set_xlabel(r"Perturbation standard deviation added to $\widetilde{R}_i$")
    ax.set_ylabel("Rank correlation with\nunperturbed ranking")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", frameon=True)
    _save(fig, out / "fig10_rank_stability_uncertainty.pdf")


if __name__ == "__main__":
    main()
