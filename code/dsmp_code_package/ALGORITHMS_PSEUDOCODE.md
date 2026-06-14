# Algorithms Implemented in the Python Package

## Algorithm 1 — Evidence ingestion and feature construction

**Input:** raw inspection table, monitoring streams, maintenance history.  
**Output:** clean feature table `X_t`.

1. Remove identifiers that could reveal site, company, asset name, or location.
2. Map each record to a simulated asset identifier `i ∈ S`.
3. Encode failure mode, component class, severity, recurrence, and inspection date.
4. Compute time since last observation and count evidence `n_i(t)`.
5. Estimate data-quality score from completeness, recency, and consistency.
6. Return normalized feature table.

Implementation: `src/dsmp/data_generator.py`.

---

## Algorithm 2 — Bayesian failure-probability updating

**Input:** count evidence `n_i(t)`, exposure time `T_i`, prior `(α_0, β_0)`, horizon `τ`.  
**Output:** posterior failure probability `P_i(τ)`.

1. For each asset `i`:
2. Set `α_i = α_0 + n_i(t)`.
3. Set `β_i = β_0 + T_i`.
4. Compute posterior mean rate `λ̂_i = α_i / β_i`.
5. Compute plug-in probability `P_i(τ) = 1 − exp(−λ̂_i τ)`.
6. Optionally compute exact predictive probability `P_i^pred(τ) = 1 − (β_i/(β_i + τ))^{α_i}`.
7. Store posterior variance proxy `α_i / β_i²`.
8. Return probabilities and uncertainty descriptors.

Implementation: `src/dsmp/bayesian.py`.

---

## Algorithm 3 — Dynamic risk-resilience prioritization

**Input:** `P_i(t)`, `C̃_i`, `RTÕ_i`, `Ũ_i(t)`, weights `(α, ρ, η)`.  
**Output:** ranked list and action class.

1. For each asset `i`:
2. Compute `R̃_i(t) = α P_i(t) C̃_i + ρ RTÕ_i + η Ũ_i(t)`.
3. Map the score to an action region by thresholds.
4. Sort assets by decreasing `R̃_i(t)`.
5. Return ranked action list.

Implementation: `src/dsmp/scoring.py`.

---

## Algorithm 4 — Value-of-information gate

**Input:** scored assets, uncertainty threshold `u*`, information actions `{inspect, monitor}`, intervention actions `{repair, reinforce, replace}`.  
**Output:** action adjusted by net value of information.

1. For each asset `i`:
2. Compute terminal loss for feasible intervention actions.
3. Estimate EVSI for inspection and monitoring.
4. Compute net value `NVI_i(e) = EVSI_i(e) − κ_e`.
5. If `Ũ_i(t) ≥ u*` and `max_e NVI_i(e) > 0`, assign the best information action.
6. Otherwise assign the best terminal intervention or defer action.
7. Return action-gated plan.

Implementation: `src/dsmp/voi_gate.py`.

---

## Algorithm 5 — Reference-based normalization

**Input:** raw attribute `y_i`, fixed reference bounds `(y_-^ref, y_+^ref)`.  
**Output:** normalized score `ỹ_i^ref`.

1. Compute `(y_i − y_-^ref)/(y_+^ref − y_-^ref)`.
2. Clip the result to `[0, 1]`.
3. Use this fixed map rather than portfolio extrema to prevent rank reversal from adding/removing assets.

Implementation: `src/dsmp/normalization.py`.

---

## Algorithm 6 — Resource-constrained maintenance scheduling

**Input:** ranked assets, resource capacity `A_t`, budget `B_t`, action costs.  
**Output:** feasible maintenance plan.

1. Initialize empty plan, used budget, and used capacity.
2. Rank candidate actions by benefit-cost ratio.
3. Add an action if budget, capacity, and one-action-per-asset constraints are satisfied.
4. Defer infeasible actions to the next planning window.
5. Return feasible plan.

Implementation: `src/dsmp/optimization.py`.

---

## Algorithm 7 — Multiobjective policy sampling / Pareto protocol

**Input:** action table, budget, capacity, number of sampled feasible policies.  
**Output:** nondominated policies over residual risk, cost, recovery exposure, and residual uncertainty.

1. Randomly sample feasible policies under budget and capacity constraints.
2. Compute residual risk, cost, recovery exposure, and residual uncertainty for each policy.
3. Filter nondominated policies.
4. Return an illustrative Pareto front for protocol verification.

Implementation: `src/dsmp/optimization.py`.

---

## Algorithm 8 — Graph-augmented consequence and recovery

**Input:** weighted dependency graph `G = (S, E, W)`, base consequence, base recovery penalty.  
**Output:** graph-augmented consequence and recovery terms.

1. Compute normalized centrality `π_i` for every asset.
2. Compute cascade susceptibility `σ_i` using a seed-anchored Katz-type iteration.
3. Amplify consequence as `C̃_i^G = C̃_i (1 + γπ_i)/(1 + γ)`.
4. Escalate recovery as `RTÕ_i^G = clip(RTÕ_i + ζ_R σ_i, 0, 1)`.
5. Recompute the dynamic score with graph-aware terms.

Implementation: `src/dsmp/graph_augmented.py`.

---

## Algorithm 9 — Sensitivity and rank-stability analysis

**Input:** baseline scores, perturbation levels, simulations.  
**Output:** rank-stability curve.

1. For each perturbation level:
2. Add Gaussian perturbations to the dynamic score.
3. Re-rank the assets.
4. Compute Spearman correlation against the baseline ranking.
5. Return the mean and standard deviation of rank correlation.

Implementation: `src/dsmp/sensitivity.py`.

---

## Algorithm 10 — Weight-simplex robustness

**Input:** baseline scores, elicited weights, number of samples.  
**Output:** rank variability and top-k retention statistics.

1. Sample `(α, ρ, η)` from a Dirichlet distribution centered on elicited weights.
2. Recompute the dynamic score for every sample.
3. Store the induced ranking.
4. Report mean rank, rank standard deviation, and top-10 frequency.

Implementation: `src/dsmp/sensitivity.py`.
