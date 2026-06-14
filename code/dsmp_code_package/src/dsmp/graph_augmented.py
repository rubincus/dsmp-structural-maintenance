"""Graph-augmented consequence and recovery modeling."""
from __future__ import annotations
import numpy as np
import pandas as pd
import networkx as nx
from .config import DSMPConfig


def build_synthetic_dependency_graph(df: pd.DataFrame, config: DSMPConfig = DSMPConfig()) -> nx.DiGraph:
    """Build a simulated weighted dependency graph.

    Edges are synthetic and encode plausible coupling by structural class and high-score
    hubs. No edge is intended to represent a real industrial topology.
    """
    rng = np.random.default_rng(config.seed + 101)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_node(row["asset_id"], structural_class=row["structural_class"], score=float(row.get("dynamic_score", 0.0)))
    assets = df["asset_id"].to_list()
    # Connect within structural classes.
    for cls, group in df.groupby("structural_class"):
        ids = group["asset_id"].to_list()
        for i in ids:
            for j in ids:
                if i != j and rng.random() < 0.10:
                    G.add_edge(i, j, weight=float(rng.uniform(0.15, 0.75)))
    # Add cross-class dependencies from high-scoring assets.
    hubs = df.sort_values("dynamic_score", ascending=False).head(max(6, len(df)//12))["asset_id"].to_list()
    for h in hubs:
        targets = rng.choice(assets, size=min(6, len(assets)), replace=False)
        for t in targets:
            if h != t:
                G.add_edge(h, t, weight=float(rng.uniform(0.25, 1.0)))
    return G


def graph_centrality(G: nx.DiGraph) -> dict[str, float]:
    try:
        pr = nx.pagerank(G, weight="weight")
    except Exception:
        pr = nx.degree_centrality(G)
    vals = np.array(list(pr.values()), dtype=float)
    lo, hi = vals.min(), vals.max()
    return {k: float((v - lo) / (hi - lo + 1e-12)) for k, v in pr.items()}


def cascade_susceptibility(G: nx.DiGraph, zeta: float = 0.35, n_iter: int = 60) -> dict[str, float]:
    """Seed-anchored Katz-type cascade susceptibility for each seed node."""
    nodes = list(G.nodes())
    idx = {n: k for k, n in enumerate(nodes)}
    n = len(nodes)
    W = np.zeros((n, n), dtype=float)
    for u, v, data in G.edges(data=True):
        W[idx[u], idx[v]] = float(data.get("weight", 1.0))
    row_sums = W.sum(axis=1, keepdims=True)
    T = np.divide(W, row_sums, out=np.zeros_like(W), where=row_sums > 0)
    if n == 0:
        return {}
    X0 = np.eye(n, dtype=float)
    X = X0.copy()
    for _ in range(n_iter):
        X_new = np.minimum(1.0, X0 + zeta * T @ X)
        if np.linalg.norm(X_new - X, ord=np.inf) < 1e-10:
            break
        X = X_new
    raw = np.clip((X.sum(axis=0) - 1.0) / max(n - 1, 1), 0, 1)
    sigma = {node: float(raw[k]) for node, k in idx.items()}
    vals = np.array(list(sigma.values()), dtype=float)
    lo, hi = vals.min(), vals.max()
    return {k: float((v - lo) / (hi - lo + 1e-12)) for k, v in sigma.items()}


def apply_graph_augmentation(df: pd.DataFrame, G: nx.DiGraph, config: DSMPConfig = DSMPConfig()) -> pd.DataFrame:
    out = df.copy()
    pi = graph_centrality(G)
    sigma = cascade_susceptibility(G, zeta=config.graph_zeta)
    out["graph_centrality"] = out["asset_id"].map(pi).fillna(0.0)
    out["cascade_susceptibility"] = out["asset_id"].map(sigma).fillna(0.0)
    gamma = config.graph_gamma
    out["C_graph"] = (out["C_norm"] * (1.0 + gamma * out["graph_centrality"]) / (1.0 + gamma)).clip(0, 1)
    out["RTO_graph"] = (out["RTO_norm"] + config.graph_zeta_r * out["cascade_susceptibility"]).clip(0, 1)
    alpha, rho, eta = config.normalized_weights()
    out["dynamic_score_graph"] = (alpha * out["P_plugin"] * out["C_graph"] + rho * out["RTO_graph"] + eta * out["U_norm"]).clip(0, 1)
    return out
