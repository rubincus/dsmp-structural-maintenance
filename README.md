# dsmp-cmes-structural-maintenance

Simulation-based Bayesian decision framework for risk-resilience structural maintenance prioritization under epistemic uncertainty.

This repository accompanies the manuscript **"Simulation-Based Bayesian Computational Decision Model for Risk-Resilience Structural Maintenance under Epistemic Uncertainty"**, prepared in the *Computer Modeling in Engineering & Sciences* (CMES) LaTeX format. The project implements and documents a reproducible Dynamic Structural Maintenance Prioritization (DSMP) workflow for converting uncertain structural evidence, simulated degradation, consequences, recovery exposure, graph descriptors, and resource constraints into auditable maintenance decisions.

The repository is organized so that the computational model, validation experiments, bridge FEM files, reported results, and manuscript source remain synchronized.

## Main Idea

The DSMP model links four decision layers:

1. **Bayesian evidence updating**: inspection/count evidence is converted into predictive failure probabilities using a Gamma-Poisson model.
2. **Risk-resilience scoring**: failure probability, consequences, recovery-time exposure, and epistemic uncertainty are combined into an interpretable maintenance-priority score.
3. **Decision routing and optimization**: value-of-information (VoI) logic, graph-aware descriptors, and budget/capacity constraints guide inspection, monitoring, repair, reinforcement, replacement, or deferment decisions.
4. **Simulation-based validation**: a controlled degradation and limit-state simulator provides ground-truth failure probabilities and oracle preventable losses, enabling calibration, regret, ranking, robustness, and Pareto feasibility checks.

The manuscript emphasizes the decision layer and its validation rather than claiming field validation. All reported datasets are simulated by the executable protocol.

## Repository Structure

```text
.
├── bridge_fem_model/
│   ├── cv_bridge_generator_V15.py
│   └── data/
├── code/
│   └── dsmp_code_package/
│       ├── src/dsmp/
│       ├── scripts/
│       ├── tests/
│       ├── pyproject.toml
│       ├── requirements.txt
│       └── README.md
├── manuscript/
│   └── cmes/
│       ├── cmes_manuscript.tex
│       ├── cmes_manuscript.pdf
│       ├── references.bib
│       ├── Definitions/
│       ├── figures/
│       └── tables/
└── results/
    └── validation_results_50x256/
```

## Article-to-Code Map

| Manuscript component | Code location | Purpose |
| --- | --- | --- |
| DSMP parameters and weights | `code/dsmp_code_package/src/dsmp/config.py` | Central configuration for scoring weights, thresholds, budgets, action costs, and simulation settings. |
| Simulated structural portfolio | `data_generator.py`, `validation_experiments.py` | Generates simulated assets, degradation states, evidence quality, consequences, recovery exposure, and validation ground truth. |
| Bayesian updating | `bayesian.py` | Implements Gamma-Poisson updating, posterior failure-rate estimates, predictive failure probability, and posterior variance. |
| Consequence and uncertainty normalization | `normalization.py`, `epistemic.py` | Builds normalized consequence, recovery, data-quality, model-discrepancy, and epistemic-uncertainty terms. |
| DSMP score and action labels | `scoring.py` | Computes the DSMP score and maps priority scores into threshold-based maintenance actions. |
| Value of information | `voi_gate.py` | Routes uncertain assets toward inspection or monitoring when information acquisition is expected to be valuable. |
| Constrained maintenance planning | `optimization.py`, `validation_experiments.py` | Builds feasible maintenance plans under budget and crew/capacity constraints, including greedy and score-ordered policies. |
| Graph-aware recovery and cascade descriptors | `graph_augmented.py`, `validation_experiments.py` | Adds network centrality and cascade-susceptibility terms to represent system exposure. |
| Probability recalibration and validation metrics | `validation_metrics.py` | Provides Brier score, expected calibration error, Platt scaling, isotonic recalibration, rank metrics, bootstrap intervals, Wilcoxon tests, Holm correction, and rank-reversal metrics. |
| Full validation protocol | `validation_experiments.py`, `scripts/run_all.py` | Runs the 300-asset, multi-seed, Monte Carlo validation protocol and writes manuscript-ready tables and figures. |
| Figure generation | `figures.py`, `validation_experiments.py` | Regenerates methodological and validation figures used in the manuscript. |
| CMES manuscript | `manuscript/cmes/cmes_manuscript.tex` | Full LaTeX article formatted with the Tech Science Press CMES template. |

## Installation

From the repository root:

```powershell
Set-Location code/dsmp_code_package
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

For running tests, install `pytest` as a development dependency:

```powershell
pip install pytest
python -m pytest
```

## Running the DSMP Workflow

To run the base DSMP modeling workflow:

```powershell
Set-Location code/dsmp_code_package
python run_all_experiments.py --out output
```

This generates simulated assets, DSMP scores, action plans, tables, and figures under `output/`.

## Running the Validation Protocol

The manuscript reports the protocol-scale validation with 300 assets, 50 random seeds, and 256 Monte Carlo paths per asset. To reproduce the main validation run:

```powershell
Set-Location code/dsmp_code_package
python scripts/run_all.py --seed 42 --out validation_results_50x256 --n-assets 300 --seeds 50 --mc-paths 256
```

The reported validation outputs are also stored in:

```text
results/validation_results_50x256/
```

Key generated products include:

- `processed/`: simulated validation datasets and policy-level records.
- `tables/`: calibration, baselines, ablations, paired tests, runtime, robustness, and simulator-parameter tables.
- `figures/`: validation workflow, calibration curve, regret-vs-budget, ranking diagnostics, VoI gate, graph-aware network, Pareto front, and rank-reversal stress test.

## Compiling the CMES Manuscript

From the CMES manuscript folder:

```powershell
Set-Location manuscript/cmes
pdflatex -interaction=nonstopmode -halt-on-error cmes_manuscript.tex
bibtex cmes_manuscript
pdflatex -interaction=nonstopmode -halt-on-error cmes_manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error cmes_manuscript.tex
```

The compiled manuscript is:

```text
manuscript/cmes/cmes_manuscript.pdf
```

Before submission, replace the corresponding-author email placeholder in `manuscript/cmes/cmes_manuscript.tex`.

## Bridge FEM Files

The folder `bridge_fem_model/` contains the bridge-generation script and CSV files used as part of the broader structural-maintenance project context. These files support the engineering setting of the study and can be used to connect FEM-derived bridge information with the DSMP decision workflow.

## Scientific Scope

This repository provides a simulation-based computational decision framework. It is designed for reproducibility, auditability, and benchmarking of maintenance decision policies under controlled assumptions. Field deployment would require site-specific structural models, inspection histories, monitoring data, calibrated deterioration processes, action-effect models, and engineering review.

## Citation

If this repository is used, please cite the accompanying manuscript once bibliographic information is available.
