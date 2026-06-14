"""Simulated structural asset generator.

The generated data are synthetic and non-identifiable. The generator preserves the
mathematical ingredients required by the article: count evidence, exposure, consequences,
recovery time, data quality, and model mismatch.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .config import DSMPConfig, STRUCTURAL_CLASSES, FAILURE_MODES, CLASS_PRIORS


def generate_synthetic_assets(config: DSMPConfig = DSMPConfig()) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    classes = rng.choice(STRUCTURAL_CLASSES, size=config.n_assets, p=[0.18, 0.20, 0.17, 0.20, 0.12, 0.13])
    modes = rng.choice(FAILURE_MODES, size=config.n_assets, p=[0.28, 0.22, 0.14, 0.16, 0.12, 0.08])

    rows = []
    for k, (cls, mode) in enumerate(zip(classes, modes), start=1):
        prior = CLASS_PRIORS[cls]
        severity = int(np.clip(np.round(rng.normal(3.4, 1.4)), 1, 6))
        age = float(np.clip(rng.gamma(shape=4.0, scale=2.5), 0.5, 25.0))
        inspections = int(np.clip(rng.poisson(lam=3.5) + 1, 1, 14))
        exposure = max(age, 0.25)
        # Event rate increases with severity and mode aggressiveness.
        mode_factor = {
            "crack": 1.1,
            "fatigue crack": 1.45,
            "corrosion": 0.90,
            "deformation": 0.95,
            "looseness": 0.75,
            "fracture": 1.65,
        }[mode]
        rate = prior["rate"] * mode_factor * (0.65 + 0.14 * severity)
        events = int(rng.poisson(lam=max(rate * exposure, 0.01)))
        # Consequence channels, with controlled noise.
        cs = float(np.clip(rng.normal(prior["safety"], 0.12) + 0.025 * (severity - 3), 0, 1))
        cp = float(np.clip(rng.normal(prior["operational"], 0.14) + 0.020 * events, 0, 1))
        ce = float(np.clip(rng.normal(prior["environmental"], 0.10), 0, 1))
        # Long-tailed recovery impact.
        rto = float(np.clip(rng.lognormal(mean=np.log(prior["rto"] + 1), sigma=0.35) - 1, 2, 120))
        data_quality_penalty = float(np.clip(1.0 - inspections / 14.0 + rng.normal(0, 0.08), 0, 1))
        model_mismatch = float(np.clip(rng.beta(2.0, 5.0) + 0.06 * (mode in ["fracture", "fatigue crack"]), 0, 1))
        rows.append(
            {
                "asset_id": f"A{k:03d}",
                "structural_class": cls,
                "failure_mode": mode,
                "severity": severity,
                "age_years": age,
                "events": events,
                "inspections": inspections,
                "exposure_years": exposure,
                "C_safety": cs,
                "C_operational": cp,
                "C_environmental": ce,
                "delta_RTO_hours": rto,
                "data_quality_penalty": data_quality_penalty,
                "model_mismatch": model_mismatch,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the DSMP simulated structural dataset.")
    parser.add_argument("--out", type=Path, default=Path("output/data/simulated_structural_assets.csv"))
    parser.add_argument("--n-assets", type=int, default=DSMPConfig().n_assets)
    parser.add_argument("--seed", type=int, default=DSMPConfig().seed)
    args = parser.parse_args()
    cfg = DSMPConfig(seed=args.seed, n_assets=args.n_assets)
    df = generate_synthetic_assets(cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} synthetic assets to {args.out}")


if __name__ == "__main__":
    main()
