"""Configuration objects for Dynamic Structural Maintenance Prioritization (DSMP)."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


@dataclass(frozen=True)
class DSMPConfig:
    """Global configuration for synthetic experiments and decision modeling."""
    seed: int = 20260610
    n_assets: int = 120
    horizon_years: float = 1.0
    prior_alpha: float = 2.0
    prior_beta: float = 4.0
    # risk, recovery, uncertainty weights: alpha + rho + eta = 1
    alpha: float = 0.55
    rho: float = 0.25
    eta: float = 0.20
    # consequence weights: safety, operational/production, environmental
    consequence_weights: Tuple[float, float, float] = (0.45, 0.40, 0.15)
    # uncertainty decomposition weights: posterior variance, data quality, model mismatch
    uncertainty_weights: Tuple[float, float, float] = (0.40, 0.35, 0.25)
    eps: float = 1.0e-9
    # decision thresholds for score-to-action mapping
    thresholds: Tuple[float, float, float] = (0.25, 0.45, 0.70)
    # VoI gate
    uncertainty_gate: float = 0.62
    # graph parameters
    graph_gamma: float = 0.40
    graph_zeta: float = 0.35
    graph_zeta_r: float = 0.35
    # optimization budget/capacity
    budget: float = 450_000.0
    capacity: int = 35
    output_dir: Path = field(default_factory=lambda: Path("output"))

    def normalized_weights(self) -> tuple[float, float, float]:
        s = self.alpha + self.rho + self.eta
        if s <= 0:
            raise ValueError("alpha + rho + eta must be positive")
        return self.alpha / s, self.rho / s, self.eta / s

    def normalized_consequence_weights(self) -> tuple[float, float, float]:
        s = sum(self.consequence_weights)
        if s <= 0:
            raise ValueError("consequence weights must sum to a positive number")
        return tuple(w / s for w in self.consequence_weights)

    def normalized_uncertainty_weights(self) -> tuple[float, float, float]:
        s = sum(self.uncertainty_weights)
        if s <= 0:
            raise ValueError("uncertainty weights must sum to a positive number")
        return tuple(w / s for w in self.uncertainty_weights)


STRUCTURAL_CLASSES = [
    "Primary frame",
    "Secondary frame",
    "Support node",
    "Connection cluster",
    "Foundation interface",
    "Service access structure",
]

FAILURE_MODES = [
    "crack",
    "fatigue crack",
    "corrosion",
    "deformation",
    "looseness",
    "fracture",
]

# Class-level baseline descriptors used only for synthetic generation.
CLASS_PRIORS: Dict[str, Dict[str, float]] = {
    "Primary frame": {"rate": 0.11, "rto": 42, "safety": 0.88, "operational": 0.78, "environmental": 0.35},
    "Secondary frame": {"rate": 0.08, "rto": 30, "safety": 0.62, "operational": 0.60, "environmental": 0.25},
    "Support node": {"rate": 0.14, "rto": 55, "safety": 0.82, "operational": 0.85, "environmental": 0.45},
    "Connection cluster": {"rate": 0.16, "rto": 38, "safety": 0.72, "operational": 0.70, "environmental": 0.30},
    "Foundation interface": {"rate": 0.07, "rto": 70, "safety": 0.95, "operational": 0.90, "environmental": 0.55},
    "Service access structure": {"rate": 0.05, "rto": 18, "safety": 0.40, "operational": 0.32, "environmental": 0.20},
}
