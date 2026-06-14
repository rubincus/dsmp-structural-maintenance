"""Tests for the expanded validation protocol."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from dsmp.epistemic import data_quality_penalty, model_discrepancy_penalty
from dsmp.validation_experiments import (
    ValidationConfig,
    candidate_table,
    evaluate_plan,
    generate_ground_truth_portfolio,
    oracle_plan,
    run_validation_suite,
    select_plan,
)


def test_epistemic_inputs_are_auditable_and_bounded():
    q_bad, comps_bad = data_quality_penalty(
        n_available=[1],
        n_required=[10],
        dt_last=[12.0],
        t_ref=4.0,
        p_missing=[0.7],
        n_conflict=[4],
        n_records=[5],
    )
    q_good, comps_good = data_quality_penalty(
        n_available=[10],
        n_required=[10],
        dt_last=[0.1],
        t_ref=4.0,
        p_missing=[0.0],
        n_conflict=[0],
        n_records=[10],
    )
    assert comps_bad.ge(0).all().all() and comps_bad.le(1).all().all()
    assert comps_good.ge(0).all().all() and comps_good.le(1).all().all()
    assert float(q_bad.iloc[0]) > float(q_good.iloc[0])
    mismatch = model_discrepancy_penalty([0.2, 0.9], [0.3, 0.1], m_ref=1.0)
    assert mismatch.between(0, 1).all()


def test_ground_truth_portfolio_columns_and_bounds():
    cfg = ValidationConfig(seed=7, n_assets=35, n_seeds=1, mc_paths=32)
    df, graph = generate_ground_truth_portfolio(7, cfg)
    required = {
        "asset_id",
        "P_predictive",
        "P_plugin",
        "p_true",
        "F_true",
        "dynamic_score_graph_pred",
        "data_quality_penalty",
        "model_mismatch",
        "graph_centrality",
        "cascade_susceptibility",
    }
    assert required.issubset(df.columns)
    assert len(graph.nodes) == len(df)
    assert df["p_true"].between(0, 1).all()
    assert set(df["F_true"].unique()).issubset({0, 1})
    assert (df["P_predictive"] <= df["P_plugin"] + 1e-12).all()
    assert df["dynamic_score_graph_pred"].between(0, 1).all()
    assert df["data_quality_penalty"].between(0, 1).all()
    assert df["model_mismatch"].between(0, 1).all()


def test_plan_feasibility_and_oracle_no_worse_than_selected_policy():
    cfg = ValidationConfig(seed=8, n_assets=45, n_seeds=1, mc_paths=32)
    df, _ = generate_ground_truth_portfolio(8, cfg)
    budget = 4.0
    capacity = 6.0
    selected = select_plan(candidate_table(df, "proposed_full"), budget, capacity)
    oracle = oracle_plan(df, budget, capacity)
    assert selected["cost"].sum() <= budget + 1e-12
    assert selected["crew"].sum() <= capacity + 1e-12
    assert selected["asset_id"].is_unique
    assert oracle["cost"].sum() <= budget + 1e-12
    assert oracle["crew"].sum() <= capacity + 1e-12
    assert evaluate_plan(df, oracle)["loss_after"] <= evaluate_plan(df, selected)["loss_after"] + 1e-9


def test_validation_suite_writes_required_artifacts(tmp_path):
    cfg = ValidationConfig(seed=9, n_assets=25, n_seeds=1, mc_paths=16, budget_grid=(2.0,))
    outputs = run_validation_suite(tmp_path, cfg)
    assert (outputs["figures"] / "figure_v2_calibration_curve.pdf").exists()
    assert (outputs["figures"] / "figure_v7_pareto_front.pdf").exists()
    pareto = pd.read_csv(outputs["tables"] / "figure_v7_pareto_front.csv")
    assert not pareto.empty
    assert pareto["feasible"].all()
    assert (outputs["tables"] / "table_v3_metric_summary.csv").exists()
    assert (outputs["tables"] / "table_v4_ablation_study.csv").exists()


if __name__ == "__main__":
    test_epistemic_inputs_are_auditable_and_bounded()
    test_ground_truth_portfolio_columns_and_bounds()
    test_plan_feasibility_and_oracle_no_worse_than_selected_policy()
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_validation_suite_writes_required_artifacts(Path(d))
    print("All validation design tests passed.")
