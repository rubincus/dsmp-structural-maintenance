"""Independent validation metrics for DSMP experiments."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import kendalltau, spearmanr
from scipy.stats import wilcoxon


def brier_score(pred, observed) -> float:
    pred = np.clip(np.asarray(pred, dtype=float), 0.0, 1.0)
    observed = np.asarray(observed, dtype=float)
    return float(np.mean((pred - observed) ** 2))


def calibration_table(pred, observed, n_bins: int = 10) -> pd.DataFrame:
    pred = np.clip(np.asarray(pred, dtype=float), 0.0, 1.0)
    observed = np.asarray(observed, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(pred, bins, right=True) - 1
    ids = np.clip(ids, 0, n_bins - 1)
    records = []
    for b in range(n_bins):
        mask = ids == b
        n = int(mask.sum())
        if n:
            pred_mean = float(pred[mask].mean())
            observed_rate = float(observed[mask].mean())
        else:
            pred_mean = float((bins[b] + bins[b + 1]) / 2.0)
            observed_rate = np.nan
        records.append(
            {
                "bin": b,
                "bin_low": float(bins[b]),
                "bin_high": float(bins[b + 1]),
                "n": n,
                "pred_mean": pred_mean,
                "observed_rate": observed_rate,
                "abs_error": float(abs(pred_mean - observed_rate)) if n else np.nan,
            }
        )
    return pd.DataFrame(records)


def expected_calibration_error(pred, observed, n_bins: int = 10) -> float:
    table = calibration_table(pred, observed, n_bins=n_bins)
    total = max(int(table["n"].sum()), 1)
    valid = table.dropna(subset=["abs_error"])
    return float(((valid["n"] / total) * valid["abs_error"]).sum())


def _logit_prob(pred, eps: float = 1e-6) -> np.ndarray:
    pred = np.clip(np.asarray(pred, dtype=float), eps, 1.0 - eps)
    return np.log(pred / (1.0 - pred))


def fit_platt_scaling(pred, observed) -> tuple[float, float]:
    """Fit logistic recalibration q=sigmoid(a*logit(p)+b)."""
    x = _logit_prob(pred)
    y = np.asarray(observed, dtype=float)
    if len(np.unique(y)) < 2:
        return 0.0, float(_logit_prob([np.mean(y) if len(y) else 0.5])[0])

    def objective(params: np.ndarray) -> float:
        a, b = params
        z = np.clip(a * x + b, -35.0, 35.0)
        q = 1.0 / (1.0 + np.exp(-z))
        q = np.clip(q, 1e-8, 1.0 - 1e-8)
        nll = -(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)).mean()
        return float(nll + 1e-4 * (a * a + b * b))

    result = minimize(objective, x0=np.asarray([1.0, 0.0]), method="L-BFGS-B", bounds=[(-20.0, 20.0), (-20.0, 20.0)])
    if not result.success:
        return 1.0, 0.0
    return float(result.x[0]), float(result.x[1])


def apply_platt_scaling(pred, a: float, b: float) -> np.ndarray:
    x = _logit_prob(pred)
    z = np.clip(a * x + b, -35.0, 35.0)
    return np.clip(1.0 / (1.0 + np.exp(-z)), 0.0, 1.0)


def fit_isotonic_regression(pred, observed) -> tuple[np.ndarray, np.ndarray]:
    """Fit a one-dimensional isotonic calibrator with the PAVA algorithm."""
    x = np.asarray(pred, dtype=float)
    y = np.asarray(observed, dtype=float)
    order = np.argsort(x, kind="mergesort")
    x_sorted = x[order]
    y_sorted = y[order]

    # Equal predictor values must share one fitted value. Consolidating them
    # before PAVA also makes the result independent of the sort implementation
    # used by a particular NumPy version.
    x_unique, starts, counts = np.unique(x_sorted, return_index=True, return_counts=True)
    y_sums = np.add.reduceat(y_sorted, starts)
    blocks: list[dict[str, float]] = []
    for xi, count, y_sum in zip(x_unique, counts, y_sums):
        weight = float(count)
        blocks.append({"x_sum": float(xi) * weight, "weight": weight, "y_sum": float(y_sum)})
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            left_mean = left["y_sum"] / left["weight"]
            right_mean = right["y_sum"] / right["weight"]
            if left_mean <= right_mean:
                break
            merged = {
                "x_sum": left["x_sum"] + right["x_sum"],
                "weight": left["weight"] + right["weight"],
                "y_sum": left["y_sum"] + right["y_sum"],
            }
            blocks[-2:] = [merged]
    x_points = np.asarray([b["x_sum"] / b["weight"] for b in blocks], dtype=float)
    y_points = np.asarray([b["y_sum"] / b["weight"] for b in blocks], dtype=float)
    if len(x_points) == 1:
        x_points = np.asarray([0.0, 1.0])
        y_points = np.repeat(y_points[0], 2)
    return x_points, np.clip(y_points, 0.0, 1.0)


def apply_isotonic_regression(pred, x_points: np.ndarray, y_points: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    order = np.argsort(x_points)
    x_points = np.asarray(x_points, dtype=float)[order]
    y_points = np.asarray(y_points, dtype=float)[order]
    return np.clip(np.interp(pred, x_points, y_points, left=y_points[0], right=y_points[-1]), 0.0, 1.0)


def cross_fitted_recalibration(pred, observed, method: str = "platt", n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Return out-of-fold recalibrated probabilities."""
    pred = np.asarray(pred, dtype=float)
    observed = np.asarray(observed, dtype=float)
    n = len(pred)
    if n == 0:
        return pred.copy()
    n_splits = max(2, min(int(n_splits), n))
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    folds = np.array_split(indices, n_splits)
    out = np.empty(n, dtype=float)
    for fold in folds:
        train = np.setdiff1d(indices, fold, assume_unique=True)
        if method == "platt":
            a, b = fit_platt_scaling(pred[train], observed[train])
            out[fold] = apply_platt_scaling(pred[fold], a, b)
        elif method == "isotonic":
            x_points, y_points = fit_isotonic_regression(pred[train], observed[train])
            out[fold] = apply_isotonic_regression(pred[fold], x_points, y_points)
        else:
            raise ValueError(f"Unknown recalibration method: {method}")
    return np.clip(out, 0.0, 1.0)


def top_k_capture(score, true_priority, k: int) -> float:
    score = np.asarray(score, dtype=float)
    true_priority = np.asarray(true_priority, dtype=float)
    k = max(1, min(int(k), len(score)))
    top_score = set(np.argsort(-score)[:k].tolist())
    top_true = set(np.argsort(-true_priority)[:k].tolist())
    return float(len(top_score & top_true) / k)


def ranking_metrics(score, true_priority, k: int = 30) -> dict[str, float]:
    score = np.asarray(score, dtype=float)
    true_priority = np.asarray(true_priority, dtype=float)
    sp = spearmanr(score, true_priority).correlation
    kd = kendalltau(score, true_priority).correlation
    return {
        "spearman": float(sp) if np.isfinite(sp) else 0.0,
        "kendall": float(kd) if np.isfinite(kd) else 0.0,
        "top_k_capture": top_k_capture(score, true_priority, k),
    }


def bootstrap_ci(values, confidence: float = 0.95, n_boot: int = 1000, seed: int = 0, statistic: str = "mean") -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot[i] = np.nanmedian(sample) if statistic == "median" else sample.mean()
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(boot, alpha)), float(np.quantile(boot, 1.0 - alpha))


def summarize_metric(values, seed: int = 0) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    mean_lo, mean_hi = bootstrap_ci(values, seed=seed, statistic="mean")
    median_lo, median_hi = bootstrap_ci(values, seed=seed, statistic="median")
    return {
        "mean": float(np.nanmean(values)),
        "median": float(np.nanmedian(values)),
        "std": float(np.nanstd(values)),
        "mean_ci95_low": mean_lo,
        "mean_ci95_high": mean_hi,
        "median_ci95_low": median_lo,
        "median_ci95_high": median_hi,
    }


def paired_wilcoxon_summary(reference_values, comparator_values, seed: int = 0) -> dict[str, float]:
    """Summarize paired differences comparator-reference for regret-like metrics."""
    reference = np.asarray(reference_values, dtype=float)
    comparator = np.asarray(comparator_values, dtype=float)
    mask = np.isfinite(reference) & np.isfinite(comparator)
    diff = comparator[mask] - reference[mask]
    if len(diff) == 0:
        return {"n_pairs": 0, "median_absolute_effect": np.nan, "mean_absolute_effect": np.nan, "effect_ci95_low": np.nan, "effect_ci95_high": np.nan, "p_value": np.nan}
    lo, hi = bootstrap_ci(diff, seed=seed, statistic="median")
    if np.allclose(diff, 0.0):
        p_value = 1.0
    elif len(diff) < 3:
        p_value = np.nan
    else:
        try:
            p_value = float(wilcoxon(diff).pvalue)
        except ValueError:
            p_value = np.nan
    return {
        "n_pairs": int(len(diff)),
        "median_absolute_effect": float(np.nanmedian(diff)),
        "mean_absolute_effect": float(np.nanmean(diff)),
        "effect_ci95_low": lo,
        "effect_ci95_high": hi,
        "p_value": p_value,
    }


def holm_bonferroni_adjust(p_values) -> np.ndarray:
    """Return Holm-Bonferroni adjusted p-values in the original order."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan, dtype=float)
    valid = np.isfinite(p_values)
    if not valid.any():
        return adjusted
    valid_indices = np.where(valid)[0]
    valid_p = np.clip(p_values[valid], 0.0, 1.0)
    order = np.argsort(valid_p)
    running_max = 0.0
    m = len(valid_p)
    for rank, local_idx in enumerate(order):
        candidate = (m - rank) * valid_p[local_idx]
        running_max = max(running_max, candidate)
        adjusted[valid_indices[local_idx]] = min(running_max, 1.0)
    return adjusted


def rank_reversal_frequency(score_before, score_after, eps: float = 1e-12) -> float:
    before = np.asarray(score_before, dtype=float)
    after = np.asarray(score_after, dtype=float)
    n = len(before)
    if n < 2:
        return 0.0
    reversals = 0
    total = 0
    for i in range(n - 1):
        db = before[i] - before[i + 1 :]
        da = after[i] - after[i + 1 :]
        valid = (np.abs(db) > eps) & (np.abs(da) > eps)
        total += int(valid.sum())
        reversals += int((np.sign(db[valid]) != np.sign(da[valid])).sum())
    return float(reversals / max(total, 1))


def nondominated_mask(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    keep = np.ones(len(values), dtype=bool)
    for i, v in enumerate(values):
        if not keep[i]:
            continue
        dominated = np.all(values <= v, axis=1) & np.any(values < v, axis=1)
        if dominated.any():
            keep[i] = False
    return keep
