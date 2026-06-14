# Dynamic Structural Maintenance Prioritization — Python Modeling Codes

This package contains the `.py` implementation of the modeling experiments and algorithms described in the manuscript **Dynamic Structural Maintenance Prioritization Using Bayesian Failure Probability, Resilience Metrics, and Epistemic Uncertainty**.

The code implements the full computational workflow:

1. Synthetic structural asset generation.
2. Gamma-Poisson Bayesian failure-rate updating.
3. Plug-in and exact posterior predictive failure probabilities.
4. Consequence vector aggregation.
5. Recovery-time normalization and reference-based normalization.
6. Epistemic uncertainty decomposition.
7. Dynamic risk-resilience scoring.
8. Value-of-information action gate.
9. Threshold action assignment.
10. Resource-constrained maintenance allocation.
11. Multiobjective/Pareto protocol using NSGA-II with random feasible-search reference.
12. Graph-augmented consequence and recovery terms.
13. Weight-simplex robustness and rank-stability analysis.
14. PDF figure regeneration.
15. Unit tests for boundedness, monotonicity, and classical-risk limiting case.

## Quick start

```bash
pip install -r requirements.txt
python run_all_experiments.py --out output
python tests/test_dsmp_properties.py
```

The pipeline writes:

- `output/data/01_synthetic_raw_assets.csv`
- `output/data/02_scored_assets_with_graph_terms.csv`
- `output/tables/*.csv`
- `output/figures/*.pdf`

Figure filenames follow the manuscript numbering for the implemented reproducibility
figures, including `fig04`, `fig05`, `fig06`, `fig07`, `fig09`, `fig10`, `fig11`,
`fig12`, `fig13`, `fig15`, `fig16`, `fig17`, and `fig18`.

## Expanded validation protocol

The companion design `Diseno_Puente_Bayessiano.pdf` is implemented as an
expanded validation layer. It adds ground-truth synthetic degradation,
auditable epistemic inputs, independent calibration/regret/ranking metrics,
VoI stress testing, graph perturbation, normalization stress, and an NSGA-II
Pareto validation front.

Representative run:

```bash
python run_validation_experiments.py --out validation_design_results --seed 42 --n-assets 300 --seeds 10 --mc-paths 128
python tests/test_validation_design.py
```

Protocol-scale run using the design default of 50 repeated seeds:

```bash
python scripts/run_all.py --seed 42 --out validation_results --n-assets 300 --seeds 50 --mc-paths 256
```

The validation pipeline writes:

- `processed/exp1_ground_truth_sample.csv`
- `tables/table_v1_generator_parameters.csv`
- `tables/table_v2_baselines.csv`
- `tables/table_v3_metric_summary.csv`
- `tables/table_v4_ablation_study.csv`
- `tables/table_v5_runtime_feasibility.csv`
- `figures/figure_v1_validation_workflow.pdf` through `figures/figure_v8_rank_reversal_frequency.pdf`

## Main scripts

- `src/dsmp/data_generator.py`: simulated structural portfolio generator.
- `src/dsmp/bayesian.py`: Gamma-Poisson update, predictive probability, posterior variance.
- `src/dsmp/normalization.py`: portfolio-relative and reference-based normalization.
- `src/dsmp/scoring.py`: dynamic index and threshold action mapping.
- `src/dsmp/voi_gate.py`: value-of-information gate and action routing.
- `src/dsmp/optimization.py`: score-order plan, greedy knapsack, and planning utilities.
- `src/dsmp/validation_experiments.py`: ground-truth validation, recalibration, paired tests, VoI stress, graph stress, and `pymoo` NSGA-II Pareto search.
- `src/dsmp/graph_augmented.py`: centrality and cascade-susceptibility augmentation.
- `src/dsmp/sensitivity.py`: rank stability and weight robustness.
- `src/dsmp/figures.py`: figure generation.
- `src/dsmp/run_all_experiments.py`: full experiment driver.

## Scientific note

The dataset is simulation-based and synthetically generated. No field asset owner, operational site, or proprietary inspection record is encoded. The purpose is methodological reproducibility and controlled validation of the decision workflow.
