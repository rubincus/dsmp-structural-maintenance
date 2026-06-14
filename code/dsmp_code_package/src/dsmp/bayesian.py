"""Bayesian failure-rate updating for count-type evidence."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import DSMPConfig


def gamma_poisson_update(df: pd.DataFrame, config: DSMPConfig = DSMPConfig()) -> pd.DataFrame:
    """Update Gamma-Poisson failure occurrence rates and failure probabilities.

    Posterior: lambda_i | D_t ~ Gamma(alpha0 + n_i, beta0 + T_i)
    Mean rate: E[lambda_i | D_t] = alpha_i / beta_i
    Plug-in probability: 1 - exp(-E[lambda] tau)
    Predictive probability: 1 - (beta_i/(beta_i + tau))**alpha_i
    Delta-method variance proxy for the plug-in probability.
    """
    out = df.copy()
    alpha_i = config.prior_alpha + out["events"].astype(float)
    beta_i = config.prior_beta + out["exposure_years"].astype(float)
    out["post_alpha"] = alpha_i
    out["post_beta"] = beta_i
    out["lambda_mean"] = alpha_i / beta_i
    out["lambda_var"] = alpha_i / (beta_i**2)
    tau = float(config.horizon_years)
    out["P_plugin"] = 1.0 - np.exp(-out["lambda_mean"] * tau)
    out["P_predictive"] = 1.0 - (out["post_beta"] / (out["post_beta"] + tau)) ** out["post_alpha"]
    derivative = tau * np.exp(-out["lambda_mean"] * tau)
    out["P_var_delta"] = (derivative**2) * out["lambda_var"]
    out["P_gap_plugin_predictive"] = out["P_plugin"] - out["P_predictive"]
    return out


def gamma_pdf(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Gamma density with rate parameter beta."""
    from scipy.special import gammaln
    x = np.asarray(x, dtype=float)
    log_pdf = alpha * np.log(beta) - gammaln(alpha) + (alpha - 1.0) * np.log(np.maximum(x, 1e-300)) - beta * x
    return np.exp(log_pdf)
