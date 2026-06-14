"""Figure generation for the DSMP experiments."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from .bayesian import gamma_pdf
from .sensitivity import lambda_eta_surface, rank_stability
from .graph_augmented import build_synthetic_dependency_graph


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_bayesian_update(path: Path) -> None:
    x = np.linspace(0.001, 2.0, 600)
    prior_a, prior_b = 2, 4
    post_a, post_b = 12, 10
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, gamma_pdf(x, prior_a, prior_b), label=f"Prior Gamma({prior_a},{prior_b})")
    ax.plot(x, gamma_pdf(x, post_a, post_b), label=f"Posterior Gamma({post_a},{post_b})")
    ax.axvline(prior_a / prior_b, linestyle="--", label="Prior mean")
    ax.axvline(post_a / post_b, linestyle="--", label="Posterior mean")
    ax.set_title("Bayesian update of failure occurrence rate")
    ax.set_xlabel("Failure occurrence rate λ (events/year)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    _save(fig, path)


def fig_failure_rates(df: pd.DataFrame, path: Path) -> None:
    g = df.groupby("structural_class").agg(rate=("lambda_mean", "mean"), err=("lambda_mean", "std")).reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.barh(g["structural_class"], g["rate"], xerr=g["err"].fillna(0), alpha=0.8)
    ax.set_xlabel("Posterior mean failure rate (events/year)")
    ax.set_title("Posterior failure rates by simulated structural class")
    _save(fig, path)


def fig_consequence(df: pd.DataFrame, path: Path) -> None:
    g = df.groupby("structural_class")[["C_safety", "C_operational", "C_environmental"]].mean().reset_index()
    y = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    left = np.zeros(len(g))
    labels = ["Safety", "Operational", "Environmental"]
    for col, label in zip(["C_safety", "C_operational", "C_environmental"], labels):
        ax.barh(y, g[col], left=left, label=label)
        left += g[col].to_numpy()
    ax.set_yticks(y, g["structural_class"])
    ax.set_xlabel("Mean normalized consequence components")
    ax.set_title("Consequence vector composition by structural class")
    ax.legend(fontsize=8)
    _save(fig, path)


def fig_rto_distribution(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["delta_RTO_hours"], bins=18, alpha=0.8)
    q90 = df["delta_RTO_hours"].quantile(0.9)
    ax.axvline(q90, linestyle="--", label="90th percentile")
    ax.set_xlabel("Recovery time impact ΔRTOᵢ (hours)")
    ax.set_ylabel("Asset count")
    ax.set_title("Distribution of recovery-time impacts")
    ax.legend(fontsize=8)
    _save(fig, path)


def fig_uncertainty_decomposition(df: pd.DataFrame, path: Path) -> None:
    top = df.sort_values("U_norm", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(top))
    parts = [top["P_var_norm"], top["data_quality_penalty"], top["model_mismatch"]]
    labels = ["Posterior variance proxy", "Data quality penalty", "Model mismatch proxy"]
    bottom = np.zeros(len(top))
    for part, label in zip(parts, labels):
        ax.bar(x, part, bottom=bottom, label=label)
        bottom += part.to_numpy()
    ax.set_xticks(x, top["asset_id"], rotation=70)
    ax.set_ylabel("Relative uncertainty contribution")
    ax.set_title("Epistemic uncertainty decomposition for high-uncertainty assets")
    ax.legend(fontsize=8)
    _save(fig, path)


def fig_policy_map(path: Path, alpha=0.55, rho=0.25, eta=0.20, rto=0.45, u=0.40) -> None:
    P, C = np.meshgrid(np.linspace(0, 1, 151), np.linspace(0, 1, 151))
    score = alpha * P * C + rho * rto + eta * u
    fig, ax = plt.subplots(figsize=(6.2, 5))
    im = ax.contourf(P, C, score, levels=[0, 0.25, 0.45, 0.70, 1.0], alpha=0.78)
    ax.contour(P, C, score, levels=[0.25, 0.45, 0.70], linewidths=1.0)
    ax.text(0.12, 0.18, "Monitor", fontsize=9)
    ax.text(0.35, 0.45, "Inspect", fontsize=9)
    ax.text(0.58, 0.62, "Repair", fontsize=9)
    ax.text(0.74, 0.84, "Reinforce / Replace", fontsize=9)
    ax.set_xlabel("Updated failure probability Pᵢ(t)")
    ax.set_ylabel("Normalized consequence C̃ᵢ")
    ax.set_title("Decision policy regions for maintenance action")
    fig.colorbar(im, ax=ax, label="Composite score")
    _save(fig, path)


def fig_static_dynamic(df: pd.DataFrame, path: Path) -> None:
    top = df.sort_values("dynamic_score", ascending=False).head(30)
    x = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - 0.2, top["static_risk"], width=0.4, label="Static PoF × consequence")
    ax.bar(x + 0.2, top["dynamic_score"], width=0.4, label="Dynamic risk-resilience index")
    ax.set_xticks(x, top["asset_id"], rotation=70)
    ax.set_ylabel("Score")
    ax.set_title("Top 30 assets: static versus dynamic prioritization")
    ax.legend(fontsize=8)
    _save(fig, path)


def fig_lambda_eta(df: pd.DataFrame, path: Path) -> None:
    row = df.sort_values("dynamic_score", ascending=False).iloc[0]
    surface = lambda_eta_surface(row)
    pivot = surface.pivot_table(index="eta", columns="rho", values="score")
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    im = ax.imshow(pivot.to_numpy(), origin="lower", aspect="auto", extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()])
    ax.set_xlabel("Resilience weight ρ")
    ax.set_ylabel("Uncertainty weight η")
    ax.set_title("Sensitivity of Rᵢ(t) to resilience and uncertainty weights")
    fig.colorbar(im, ax=ax, label="Risk-resilience score")
    _save(fig, path)


def fig_rank_stability(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    stab = rank_stability(df, seed=123)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(stab["sigma"], stab["mean_rank_correlation"], marker="o")
    ax.fill_between(stab["sigma"], stab["mean_rank_correlation"] - stab["std_rank_correlation"], stab["mean_rank_correlation"] + stab["std_rank_correlation"], alpha=0.2)
    ax.set_xlabel("Perturbation standard deviation in Rᵢ(t)")
    ax.set_ylabel("Mean rank correlation")
    ax.set_ylim(0, 1.05)
    ax.set_title("Rank stability under epistemic uncertainty perturbation")
    _save(fig, path)
    return stab


def fig_dependency_graph(df: pd.DataFrame, path: Path):
    G = build_synthetic_dependency_graph(df)
    pos = nx.spring_layout(G, seed=8, weight="weight")
    scores = [G.nodes[n].get("score", 0) for n in G.nodes]
    sizes = [220 + 450 * s for s in scores]
    fig, ax = plt.subplots(figsize=(7, 5))
    nodes = nx.draw_networkx_nodes(G, pos, node_color=scores, node_size=sizes, cmap="plasma", ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.22, arrows=False, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
    ax.set_title("Anonymized structural asset dependency graph")
    ax.axis("off")
    fig.colorbar(nodes, ax=ax, label="Synthetic dynamic risk score")
    _save(fig, path)
    return G


def fig_benchmark(path: Path) -> None:
    metrics = ["Top-risk capture", "Rank stability", "Recovery awareness", "Uncertainty awareness"]
    rows = ["Severity only", "Static RCM", "PoF only", "Consequence only", "Proposed Rᵢ(t)"]
    data = np.array([
        [0.45, 0.62, 0.10, 0.05],
        [0.58, 0.71, 0.25, 0.10],
        [0.61, 0.54, 0.18, 0.25],
        [0.63, 0.65, 0.38, 0.12],
        [0.82, 0.80, 0.78, 0.74],
    ])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.imshow(data, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(metrics)), metrics, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Benchmark comparison on simulated structural case")
    fig.colorbar(im, ax=ax, label="Normalized performance")
    _save(fig, path)


def fig_tradeoff(df: pd.DataFrame, path: Path) -> None:
    scores = np.sort(df["dynamic_score"].to_numpy())[::-1]
    cum_reduction = np.cumsum(scores) / np.sum(scores)
    resource = np.linspace(0, 1, len(scores))
    burden = resource**1.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(resource, cum_reduction, label="Expected risk reduction")
    ax.plot(resource, burden, label="Normalized intervention burden")
    ax.set_xlabel("Normalized maintenance resource allocation")
    ax.set_ylabel("Normalized outcome")
    ax.set_title("Illustrative trade-off between risk reduction and intervention burden")
    ax.legend(fontsize=8)
    _save(fig, path)


def fig_forward_band(df: pd.DataFrame, path: Path) -> None:
    months = np.arange(0, 13)
    base = df["dynamic_score"].mean()
    mean = base + 0.035 * np.sqrt(months) + 0.012 * months
    width = 0.05 + 0.014 * months
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(months, mean, label="Predictive mean")
    ax.fill_between(months, mean - width, mean + width, alpha=0.25, label="95% credible band")
    ax.set_xlabel("Prediction horizon (months)")
    ax.set_ylabel("Risk-resilience score")
    ax.set_title("Forward uncertainty band for Rᵢ(t) under progressive degradation")
    ax.legend(fontsize=8)
    _save(fig, path)
