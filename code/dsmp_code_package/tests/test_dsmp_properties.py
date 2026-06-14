"""Minimal property tests for the DSMP operator.

Run:
    python tests/test_dsmp_properties.py
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from dsmp.config import DSMPConfig
from dsmp.data_generator import generate_synthetic_assets
from dsmp.bayesian import gamma_poisson_update
from dsmp.normalization import add_reference_and_portfolio_normalizations, reference_normalize
from dsmp.scoring import compute_scores, assign_threshold_actions
from dsmp.voi_gate import apply_voi_gate
from dsmp.optimization import build_action_table, greedy_knapsack_plan
from dsmp.graph_augmented import build_synthetic_dependency_graph, apply_graph_augmentation


def make_df():
    cfg = DSMPConfig(n_assets=40, seed=123)
    df = generate_synthetic_assets(cfg)
    df = gamma_poisson_update(df, cfg)
    df = add_reference_and_portfolio_normalizations(df)
    df = compute_scores(df, cfg)
    return df


def test_boundedness():
    df = make_df()
    assert (df["dynamic_score"] >= 0).all()
    assert (df["dynamic_score"] <= 1).all()


def test_classical_risk_limit():
    cfg = DSMPConfig(n_assets=40, seed=123, alpha=1.0, rho=0.0, eta=0.0)
    df = generate_synthetic_assets(cfg)
    df = gamma_poisson_update(df, cfg)
    df = add_reference_and_portfolio_normalizations(df)
    df = compute_scores(df, cfg)
    assert np.allclose(df["dynamic_score"], df["static_risk"], atol=1e-12)


def test_monotonicity_single_asset():
    cfg = DSMPConfig()
    alpha, rho, eta = cfg.normalized_weights()
    p, c, r, u = 0.3, 0.4, 0.2, 0.1
    base = alpha * p * c + rho * r + eta * u
    assert alpha * (p + 0.1) * c + rho * r + eta * u >= base
    assert alpha * p * (c + 0.1) + rho * r + eta * u >= base
    assert alpha * p * c + rho * (r + 0.1) + eta * u >= base
    assert alpha * p * c + rho * r + eta * (u + 0.1) >= base


def test_predictive_probability_formula_and_plugin_bound():
    cfg = DSMPConfig(n_assets=40, seed=123)
    df = generate_synthetic_assets(cfg)
    df = gamma_poisson_update(df, cfg)
    expected = 1.0 - (df["post_beta"] / (df["post_beta"] + cfg.horizon_years)) ** df["post_alpha"]
    assert np.allclose(df["P_predictive"], expected, atol=1e-12)
    assert (df["P_predictive"] <= df["P_plugin"] + 1e-12).all()
    assert np.allclose(df["P_gap_plugin_predictive"], df["P_plugin"] - df["P_predictive"], atol=1e-12)


def test_reference_normalization_portfolio_invariance():
    base = reference_normalize(np.array([20.0, 60.0]), 0.0, 100.0)
    extended = reference_normalize(np.array([20.0, 60.0, 1000.0]), 0.0, 100.0)
    assert np.allclose(base, extended[:2], atol=1e-12)


def test_threshold_and_voi_actions_are_valid():
    df = make_df()
    df = assign_threshold_actions(df)
    df = apply_voi_gate(df)
    assert set(df["threshold_action"]).issubset({"monitor", "inspect", "repair", "reinforce_or_replace"})
    assert set(df["voi_action"]).issubset({"defer", "inspect", "monitor", "repair", "reinforce", "replace"})

    high_value_uncertain = df.iloc[[0]].copy()
    high_value_uncertain["U_norm"] = 1.0
    high_value_uncertain["C_norm"] = 1.0
    high_value_uncertain["P_plugin"] = 0.01
    routed = apply_voi_gate(high_value_uncertain, DSMPConfig(uncertainty_gate=0.50))
    assert routed["voi_action"].iloc[0] == "inspect"


def test_graph_augmentation_is_bounded():
    cfg = DSMPConfig(n_assets=40, seed=123)
    df = make_df()
    graph = build_synthetic_dependency_graph(df, cfg)
    augmented = apply_graph_augmentation(df, graph, cfg)
    for col in ["graph_centrality", "cascade_susceptibility", "C_graph", "RTO_graph", "dynamic_score_graph"]:
        assert (augmented[col] >= -1e-12).all()
        assert (augmented[col] <= 1.0 + 1e-12).all()


def test_greedy_allocation_respects_budget_capacity_and_one_action_per_asset():
    cfg = DSMPConfig(n_assets=40, seed=123, budget=120_000.0, capacity=10)
    df = generate_synthetic_assets(cfg)
    df = gamma_poisson_update(df, cfg)
    df = add_reference_and_portfolio_normalizations(df)
    df = compute_scores(df, cfg)
    actions = build_action_table(df)
    plan = greedy_knapsack_plan(actions, cfg)
    assert plan["cost"].sum() <= cfg.budget + 1e-9
    assert plan["capacity"].sum() <= cfg.capacity
    assert plan["asset_id"].is_unique


if __name__ == "__main__":
    test_boundedness()
    test_classical_risk_limit()
    test_monotonicity_single_asset()
    test_predictive_probability_formula_and_plugin_bound()
    test_reference_normalization_portfolio_invariance()
    test_threshold_and_voi_actions_are_valid()
    test_graph_augmentation_is_bounded()
    test_greedy_allocation_respects_budget_capacity_and_one_action_per_asset()
    print("All DSMP property tests passed.")
