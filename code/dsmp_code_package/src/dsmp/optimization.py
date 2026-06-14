"""Resource-constrained and multiobjective maintenance allocation."""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from .config import DSMPConfig
from .voi_gate import ACTIONS


def build_action_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        for action_name in ["inspect", "monitor", "repair", "reinforce", "replace"]:
            action = ACTIONS[action_name]
            if action.kind == "information":
                risk_reduction = 0.06 * float(row["dynamic_score"]) + 0.08 * float(row["U_norm"])
                capacity = 1
            else:
                risk_reduction = float(row["dynamic_score"]) * action.effectiveness
                capacity = 1 if action_name == "repair" else 2
            rows.append(
                {
                    "asset_id": row["asset_id"],
                    "action": action_name,
                    "cost": action.cost,
                    "capacity": capacity,
                    "risk_reduction": min(risk_reduction, float(row["dynamic_score"])),
                    "score": float(row["dynamic_score"]),
                }
            )
    return pd.DataFrame(rows)


def greedy_knapsack_plan(action_table: pd.DataFrame, config: DSMPConfig = DSMPConfig()) -> pd.DataFrame:
    """Feasible lower bound for the multiple-choice knapsack allocation."""
    table = action_table.copy()
    table["benefit_cost_ratio"] = table["risk_reduction"] / (table["cost"] + 1.0)
    selected = []
    used_assets = set()
    budget = 0.0
    capacity = 0
    for _, row in table.sort_values("benefit_cost_ratio", ascending=False).iterrows():
        if row["asset_id"] in used_assets:
            continue
        if budget + row["cost"] <= config.budget and capacity + row["capacity"] <= config.capacity:
            selected.append(row.to_dict())
            used_assets.add(row["asset_id"])
            budget += float(row["cost"])
            capacity += int(row["capacity"])
    return pd.DataFrame(selected)


def score_order_plan(df: pd.DataFrame, config: DSMPConfig = DSMPConfig()) -> pd.DataFrame:
    """Cost-agnostic ranking plan corresponding to the score-ordering rule."""
    ordered = df.sort_values("dynamic_score", ascending=False).copy()
    ordered["planned_order"] = range(1, len(ordered) + 1)
    return ordered[["planned_order", "asset_id", "dynamic_score", "threshold_action", "voi_action"]]


def random_pareto_front(df: pd.DataFrame, action_table: pd.DataFrame, config: DSMPConfig = DSMPConfig(), n_samples: int = 1500) -> pd.DataFrame:
    """Small illustrative Pareto front from random feasible policies.

    This is not a claim of algorithmic optimality. It fixes the evaluation protocol for
    multiobjective comparison: residual risk, cost, recovery exposure, residual uncertainty.
    """
    rng = np.random.default_rng(config.seed + 12)
    assets = df["asset_id"].to_numpy()
    actions_by_asset = {a: action_table[action_table.asset_id == a] for a in assets}
    records = []
    for k in range(n_samples):
        budget = 0.0
        capacity = 0
        reduction = 0.0
        chosen = []
        shuffled = rng.permutation(assets)
        for asset_id in shuffled:
            options = actions_by_asset[asset_id]
            # bias toward doing nothing/cheaper actions by choosing from a subset
            if rng.random() < 0.58:
                continue
            opt = options.sample(n=1, random_state=int(rng.integers(1, 1_000_000))).iloc[0]
            if budget + opt.cost <= config.budget and capacity + opt.capacity <= config.capacity:
                budget += float(opt.cost)
                capacity += int(opt.capacity)
                reduction += float(opt.risk_reduction)
                chosen.append(asset_id)
        residual_risk = max(float(df["dynamic_score"].sum()) - reduction, 0.0)
        residual_uncertainty = float(df.loc[~df.asset_id.isin(chosen), "U_norm"].sum())
        recovery_exposure = float(df.loc[~df.asset_id.isin(chosen), "RTO_norm"].sum())
        records.append({"sample": k, "residual_risk": residual_risk, "cost": budget, "recovery_exposure": recovery_exposure, "residual_uncertainty": residual_uncertainty})
    front = pd.DataFrame(records)
    return nondominated(front, minimize_cols=["residual_risk", "cost", "recovery_exposure", "residual_uncertainty"])


def nondominated(df: pd.DataFrame, minimize_cols: list[str]) -> pd.DataFrame:
    values = df[minimize_cols].to_numpy(dtype=float)
    keep = np.ones(len(values), dtype=bool)
    for i, v in enumerate(values):
        if not keep[i]:
            continue
        dominated = np.all(values <= v, axis=1) & np.any(values < v, axis=1)
        if dominated.any():
            keep[i] = False
    return df.loc[keep].reset_index(drop=True)
