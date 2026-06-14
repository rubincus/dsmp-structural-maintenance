"""Auditable epistemic-input construction for validation experiments."""
from __future__ import annotations
import numpy as np
import pandas as pd


def _as_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def data_quality_components(
    n_available,
    n_required,
    dt_last,
    t_ref: float,
    p_missing,
    n_conflict,
    n_records,
    eps: float = 1e-9,
) -> pd.DataFrame:
    """Compute observable quality components q_comp, q_rec, q_sensor, and q_cons."""
    n_available = _as_array(n_available)
    n_required = np.maximum(_as_array(n_required), eps)
    dt_last = np.maximum(_as_array(dt_last), 0.0)
    p_missing = np.clip(_as_array(p_missing), 0.0, 1.0)
    n_conflict = np.maximum(_as_array(n_conflict), 0.0)
    n_records = np.maximum(_as_array(n_records), 0.0)
    t_ref = max(float(t_ref), eps)
    return pd.DataFrame(
        {
            "q_comp": np.clip(n_available / n_required, 0.0, 1.0),
            "q_rec": np.clip(np.exp(-dt_last / t_ref), 0.0, 1.0),
            "q_sensor": np.clip(1.0 - p_missing, 0.0, 1.0),
            "q_cons": np.clip(1.0 - n_conflict / (n_records + eps), 0.0, 1.0),
        }
    )


def data_quality_penalty(
    n_available,
    n_required,
    dt_last,
    t_ref: float,
    p_missing,
    n_conflict,
    n_records,
    weights: tuple[float, float, float, float] = (0.30, 0.25, 0.25, 0.20),
) -> tuple[pd.Series, pd.DataFrame]:
    """Return Q_i(t) = 1 - weighted observable data-quality score."""
    comps = data_quality_components(n_available, n_required, dt_last, t_ref, p_missing, n_conflict, n_records)
    w = np.asarray(weights, dtype=float)
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("quality weights must be non-negative and sum to a positive number")
    w = w / w.sum()
    quality = comps.to_numpy(dtype=float) @ w
    return pd.Series(np.clip(1.0 - quality, 0.0, 1.0), name="data_quality_penalty"), comps


def model_discrepancy_penalty(m_phys, m_data, m_ref: float = 1.0) -> pd.Series:
    """Return M_i(t), the clipped disagreement between physics and data estimates."""
    m_ref = max(float(m_ref), 1e-9)
    penalty = np.abs(_as_array(m_phys) - _as_array(m_data)) / m_ref
    return pd.Series(np.clip(penalty, 0.0, 1.0), name="model_mismatch")


def epistemic_uncertainty(
    p_var_norm,
    q_penalty,
    m_penalty,
    weights: tuple[float, float, float] = (0.40, 0.35, 0.25),
) -> pd.Series:
    """Return U_i(t) = alpha1 Var[P_i] + alpha2 Q_i + alpha3 M_i."""
    w = np.asarray(weights, dtype=float)
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("uncertainty weights must be non-negative and sum to a positive number")
    w = w / w.sum()
    arr = np.column_stack([_as_array(p_var_norm), _as_array(q_penalty), _as_array(m_penalty)])
    return pd.Series(np.clip(arr @ w, 0.0, 1.0), name="U_norm")

