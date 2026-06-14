"""Run all DSMP modeling experiments and regenerate data/figures.

Usage from package root:
    python -m src.dsmp.run_all_experiments --out output
or after installing in editable mode:
    python -m dsmp.run_all_experiments --out output
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from .config import DSMPConfig
from .data_generator import generate_synthetic_assets
from .bayesian import gamma_poisson_update
from .normalization import add_reference_and_portfolio_normalizations
from .scoring import compute_scores, assign_threshold_actions, verify_index_properties
from .voi_gate import apply_voi_gate
from .optimization import build_action_table, greedy_knapsack_plan, score_order_plan, random_pareto_front
from .graph_augmented import build_synthetic_dependency_graph, apply_graph_augmentation
from .sensitivity import rank_stability, weight_simplex_robustness
from . import figures as figs


def run_pipeline(out_dir: Path, config: DSMPConfig = DSMPConfig()) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    data_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)
    table_dir.mkdir(exist_ok=True)

    # 1. Evidence ingestion and feature construction.
    df_raw = generate_synthetic_assets(config)
    df_raw.to_csv(data_dir / "01_synthetic_raw_assets.csv", index=False)

    # 2. Bayesian updating.
    df = gamma_poisson_update(df_raw, config)

    # 3. Normalization, consequence, uncertainty, and scoring.
    df = add_reference_and_portfolio_normalizations(df)
    df = compute_scores(df, config, use_reference_norm=True)
    df = assign_threshold_actions(df, config)

    # 4. Value-of-information action gate.
    df = apply_voi_gate(df, config)

    # 5. Graph augmentation.
    G = build_synthetic_dependency_graph(df, config)
    df_graph = apply_graph_augmentation(df, G, config)
    df_graph.to_csv(data_dir / "02_scored_assets_with_graph_terms.csv", index=False)

    # 6. Optimization and planning.
    actions = build_action_table(df)
    greedy = greedy_knapsack_plan(actions, config)
    score_plan = score_order_plan(df, config)
    pareto = random_pareto_front(df, actions, config, n_samples=1200)
    actions.to_csv(table_dir / "action_table.csv", index=False)
    greedy.to_csv(table_dir / "greedy_knapsack_plan.csv", index=False)
    score_plan.to_csv(table_dir / "score_order_plan.csv", index=False)
    pareto.to_csv(table_dir / "illustrative_pareto_front.csv", index=False)

    # 7. Sensitivity and robustness.
    stability = rank_stability(df, seed=config.seed)
    robustness = weight_simplex_robustness(df, config)
    stability.to_csv(table_dir / "rank_stability.csv", index=False)
    robustness.to_csv(table_dir / "weight_simplex_robustness.csv", index=False)

    # 8. Figures.
    figs.fig_bayesian_update(fig_dir / "fig05_bayesian_update_gamma_poisson.pdf")
    figs.fig_consequence(df, fig_dir / "fig06_consequence_decomposition.pdf")
    figs.fig_failure_rates(df, fig_dir / "fig07_posterior_failure_rates.pdf")
    figs.fig_lambda_eta(df, fig_dir / "fig09_lambda_eta_sensitivity.pdf")
    figs.fig_rank_stability(df, fig_dir / "fig10_rank_stability_uncertainty.pdf")
    figs.fig_rto_distribution(df, fig_dir / "fig11_rto_distribution.pdf")
    figs.fig_uncertainty_decomposition(df, fig_dir / "fig12_epistemic_uncertainty_decomposition.pdf")
    figs.fig_policy_map(fig_dir / "fig13_decision_policy_map.pdf")
    figs.fig_static_dynamic(df, fig_dir / "fig15_static_vs_dynamic_ranking.pdf")
    figs.fig_tradeoff(df, fig_dir / "fig16_maintenance_tradeoff.pdf")
    figs.fig_dependency_graph(df, fig_dir / "fig04_system_dependency_graph.pdf")
    figs.fig_benchmark(fig_dir / "fig17_benchmark_comparison.pdf")
    figs.fig_forward_band(df, fig_dir / "fig18_forward_uncertainty_band.pdf")

    # 9. Report verification.
    checks = verify_index_properties(df)
    pd.DataFrame([checks]).to_csv(table_dir / "property_checks.csv", index=False)
    return {"scored_assets": data_dir / "02_scored_assets_with_graph_terms.csv", "figures": fig_dir, "tables": table_dir}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DSMP synthetic modeling experiments.")
    parser.add_argument("--out", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument("--seed", type=int, default=DSMPConfig().seed)
    parser.add_argument("--n-assets", type=int, default=DSMPConfig().n_assets)
    parser.add_argument("--alpha", type=float, default=DSMPConfig().alpha)
    parser.add_argument("--rho", type=float, default=DSMPConfig().rho)
    parser.add_argument("--eta", type=float, default=DSMPConfig().eta)
    args = parser.parse_args()
    cfg = DSMPConfig(seed=args.seed, n_assets=args.n_assets, alpha=args.alpha, rho=args.rho, eta=args.eta)
    outputs = run_pipeline(args.out, cfg)
    print("DSMP experiments completed.")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
