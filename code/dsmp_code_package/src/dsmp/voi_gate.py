"""Value-of-information gate for uncertainty-driven maintenance actions."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .config import DSMPConfig


@dataclass(frozen=True)
class ActionSpec:
    name: str
    cost: float
    effectiveness: float
    kind: str  # information, intervention, null


ACTIONS = {
    "defer": ActionSpec("defer", cost=0.0, effectiveness=0.0, kind="null"),
    "inspect": ActionSpec("inspect", cost=3_000.0, effectiveness=0.0, kind="information"),
    "monitor": ActionSpec("monitor", cost=5_000.0, effectiveness=0.0, kind="information"),
    "repair": ActionSpec("repair", cost=18_000.0, effectiveness=0.45, kind="intervention"),
    "reinforce": ActionSpec("reinforce", cost=42_000.0, effectiveness=0.70, kind="intervention"),
    "replace": ActionSpec("replace", cost=75_000.0, effectiveness=0.95, kind="intervention"),
}


def terminal_loss(p_failure: float, consequence: float, action: ActionSpec) -> float:
    """Loss = direct cost + residual expected failure loss.

    Consequence is scaled to monetary-equivalent units for the decision gate only.
    """
    consequence_scale = 200_000.0
    return action.cost + (1.0 - action.effectiveness) * p_failure * consequence * consequence_scale


def best_terminal_action(p_failure: float, consequence: float) -> tuple[str, float]:
    candidates = [ACTIONS[k] for k in ["defer", "repair", "reinforce", "replace"]]
    losses = [(a.name, terminal_loss(p_failure, consequence, a)) for a in candidates]
    return min(losses, key=lambda x: x[1])


def approximate_evsi(row: pd.Series, info_action: str) -> float:
    """Approximate EVSI from expected posterior variance contraction.

    This is a transparent sparse-data proxy: inspection and monitoring reduce posterior
    uncertainty by different factors, and the expected decision loss reduction is proportional
    to the current uncertainty contribution and consequence.
    """
    if info_action not in ["inspect", "monitor"]:
        raise ValueError("info_action must be inspect or monitor")
    reduction = 0.55 if info_action == "inspect" else 0.35
    consequence_scale = 200_000.0
    expected_gain = reduction * float(row["U_norm"]) * float(row["C_norm"]) * consequence_scale * 0.25
    return expected_gain - ACTIONS[info_action].cost


def apply_voi_gate(df: pd.DataFrame, config: DSMPConfig = DSMPConfig()) -> pd.DataFrame:
    """Route high-uncertainty assets to information actions if net value is positive."""
    out = df.copy()
    actions = []
    nvi_values = []
    for _, row in out.iterrows():
        best_int_action, _ = best_terminal_action(float(row["P_plugin"]), float(row["C_norm"]))
        nvi_inspect = approximate_evsi(row, "inspect")
        nvi_monitor = approximate_evsi(row, "monitor")
        if float(row["U_norm"]) >= config.uncertainty_gate and max(nvi_inspect, nvi_monitor) > 0:
            action = "inspect" if nvi_inspect >= nvi_monitor else "monitor"
            nvi = max(nvi_inspect, nvi_monitor)
        else:
            action = best_int_action
            nvi = max(nvi_inspect, nvi_monitor)
        actions.append(action)
        nvi_values.append(nvi)
    out["voi_action"] = actions
    out["NVI_best"] = nvi_values
    return out
