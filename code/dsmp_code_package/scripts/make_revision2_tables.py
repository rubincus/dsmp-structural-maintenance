"""Write the LaTeX rows of the second-revision table (Table 17 of the manuscript) from the
S11 result files, and print the numbers quoted in the text of Sections 3.4, 6.5, and 6.6.

Usage: python scripts/make_revision2_tables.py <results/revision2_results>
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsmp.revision_experiments import _median_summary, _paired  # noqa: E402
from dsmp.validation_metrics import holm_bonferroni_adjust  # noqa: E402

res = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parents[1] / "results" / "revision2_results"
long = pd.read_csv(res / "S11_reduced_forms_long.csv", dtype={"ratio_label": str})
REG = {"R76 (reported)": "0.76 (reported)", "R50": "0.50", "R33": "0.33", "R18": "0.19", "R05": "0.05"}
PROB = {"platt": "Count proxy", "covariate": "Condition-informed", "true": "Exact"}
USED, TRUTH = "0.455", "0.20"


def fmt(x: float) -> str:
    return f"{x:.3f}".replace("-", "$-$") if x < 0 else f"{x:.3f}"


def sgn(x: float) -> str:
    return ("+" if x >= 0 else "$-$") + f"{abs(x):.3f}"


rows = []
for regime in REG:
    for prob in PROB:
        g = long[(long["regime"] == regime) & (long["probability"] == prob)]
        sys_ = g[(g["form"] == "sys") & (g["ratio_label"] == USED)].set_index("seed").sort_index()
        exp_ = g[(g["form"] == "exp") & (g["ratio_label"] == USED)].set_index("seed").loc[sys_.index]
        exp_t = g[(g["form"] == "exp") & (g["ratio_label"] == TRUTH)].set_index("seed").loc[sys_.index]
        s_sys = _median_summary(sys_["regret"].to_numpy(), seed=42)
        s_exp = _median_summary(exp_["regret"].to_numpy(), seed=42)
        s_ext = _median_summary(exp_t["regret"].to_numpy(), seed=42)
        pr = _paired(sys_["regret"].to_numpy(), exp_["regret"].to_numpy(), seed=42)
        pr_t = _paired(sys_["regret"].to_numpy(), exp_t["regret"].to_numpy(), seed=42)
        rows.append({"regime": regime, "prob": prob,
                     "ref_share": float(sys_["reference_share_selected_p_true_gt_095"].median()),
                     "event_rate": float(sys_["event_rate"].median()),
                     "sys": s_sys, "exp": s_exp, "exp_truth": s_ext, "pr": pr, "pr_truth": pr_t})
p = np.asarray([r["pr"]["p_value"] for r in rows])
holm = holm_bonferroni_adjust(p)
p_t = np.asarray([r["pr_truth"]["p_value"] for r in rows])
holm_t = holm_bonferroni_adjust(p_t)

lines = []
for k, r in enumerate(rows):
    first = r["prob"] == "platt"
    reg = REG[r["regime"]] if first else ""
    share = f"{r['ref_share']:.2f}" if first else ""
    hp = holm[k]
    hp_s = "$<0.001$" if hp < 0.001 else f"{hp:.3f}"
    pr = r["pr"]
    lines.append(
        f"{reg} & {share} & {PROB[r['prob']]} & {fmt(r['sys']['median'])} [{fmt(r['sys']['ci_low'])}, {fmt(r['sys']['ci_high'])}] & "
        f"{fmt(r['exp']['median'])} [{fmt(r['exp']['ci_low'])}, {fmt(r['exp']['ci_high'])}] & {fmt(r['exp_truth']['median'])} & "
        f"{sgn(pr['median_effect'])} [{sgn(pr['effect_ci_low'])}, {sgn(pr['effect_ci_high'])}] & {pr['win_fraction']:.2f} & {hp_s} \\\\"
    )
    if r["prob"] == "true" and r["regime"] != "R05":
        lines.append("\\addlinespace")
out = res / "table_S11_reduced_forms_rows.tex"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {out}")
print("\n".join(lines))

print("\nexp form with the ground-truth weight 0.20 against sys form with the reported ratio 0.455:")
for k, r in enumerate(rows):
    pt = r["pr_truth"]
    print(f"  {r['regime']:15s} {r['prob']:10s} sys={r['sys']['median']:.3f} exp0.20={r['exp_truth']['median']:.3f} "
          f"effect={pt['median_effect']:+.3f} [{pt['effect_ci_low']:+.3f},{pt['effect_ci_high']:+.3f}] syswins={pt['win_fraction']:.2f} holm={holm_t[k]:.3g}")

print("\nexact probability, reported regime, by weight ratio:")
g = long[(long["regime"] == "R76 (reported)") & (long["probability"] == "true")]
print(g.groupby(["ratio_label", "form"])["regret"].median().unstack().round(3))
