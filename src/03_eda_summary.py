"""Step 3: coverage and descriptive statistics used in the synopsis.

Run:  python src/03_eda_summary.py
Outputs: reports/coverage.csv, reports/descriptives.csv, reports/figures/der_by_class.png,
         reports/figures/panel_preview.png
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import PROCESSED, ROOT, FIG

FEATS = ["d_unemp", "gdp_growth", "hpi_growth", "inflation", "lt_rate", "log_assets", "cet1_ratio", "stage2_share", "unemp"]
NAVY, GREY = "#1F3864", "#595959"

p = pd.read_csv(PROCESSED / "panel_eu_irb.csv", parse_dates=["quarter_end"])
cc = p.dropna(subset=FEATS + ["d_der"])
r4 = cc.dropna(subset=["rise_next"])
test = r4[r4.quarter_end >= "2024-06-30"]
cov = pd.Series({
    "Banks": p.lei.nunique(), "Counterparty countries (EU/EEA)": p.cpty_country.nunique(),
    "Quarterly reference dates": p.quarter_end.nunique(),
    "Bank-country-class cells": p.groupby(["lei", "cpty_country", "exposure_class"]).ngroups,
    "Cell-quarters in panel": len(p), "Complete cases for RQ1-RQ2 (d_der)": len(cc),
    "Complete cases for RQ3-RQ4 (next-quarter target)": len(r4),
    "Out-of-time test cases (origins Jun 2024-Mar 2025)": len(test),
    "Share of next-quarter rises > 0.10 pp": round(r4.rise_next.mean(), 4),
})
(ROOT / "reports").mkdir(exist_ok=True)
cov.to_csv(ROOT / "reports" / "coverage.csv", header=["value"])
desc = cc[["der", "d_der"] + FEATS].describe(percentiles=[.5]).T[["count", "mean", "std", "min", "50%", "max"]]
desc.round(3).to_csv(ROOT / "reports" / "descriptives.csv")
print(cov.to_string()); print(desc.round(2).to_string())

FIG.mkdir(parents=True, exist_ok=True)
w = p.assign(w=p.orig_exp).groupby(["quarter_end", "exposure_class"]).apply(
    lambda d: 100 * d.def_exp.sum() / d.orig_exp.sum(), include_groups=False).unstack()
fig, ax = plt.subplots(figsize=(8, 3.6), dpi=200)
styles = {"Corporates": "-", "Retail secured by real estate": "--", "Retail qualifying revolving": ":", "Other retail": "-."}
for c in w.columns:
    ax.plot(w.index, w[c], styles.get(c, "-"), color=NAVY if c in ("Corporates", "Other retail") else GREY, lw=1.6, label=c)
ax.set_ylabel("Defaulted exposure ratio (%)", fontsize=9)
ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=8)
ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
ax.grid(axis="y", color="#DDDDDD", lw=0.6)
plt.tight_layout(); plt.savefig(FIG / "der_by_class.png", facecolor="white"); plt.close()

prev = p[(p.der > 0) & p.d_der.notna() & (p.orig_exp > 1000)].sample(8, random_state=42).sort_values(["exposure_class", "quarter_end"])[["lei", "cpty_country", "exposure_class", "quarter_end", "orig_exp", "def_exp", "der", "d_der", "d_unemp"]].copy()
prev["lei"] = prev["lei"].str[:8] + "…"; prev["quarter_end"] = prev["quarter_end"].dt.date
for col in ["orig_exp", "def_exp"]: prev[col] = prev[col].map(lambda v: f"{v:,.1f}")
for col in ["der", "d_der", "d_unemp"]: prev[col] = prev[col].map(lambda v: f"{v:.3f}")
fig, ax = plt.subplots(figsize=(10, 2.6), dpi=200); ax.axis("off")
t = ax.table(cellText=prev.values, colLabels=prev.columns, loc="center", cellLoc="left", colLoc="left")
t.auto_set_font_size(False); t.set_fontsize(7.5); t.scale(1, 1.35)
for (r, c), cell in t.get_celld().items():
    cell.set_edgecolor("#BBBBBB"); cell.set_linewidth(0.4)
    if r == 0: cell.set_facecolor("#E7EEF7"); cell.set_text_props(weight="bold")
t.auto_set_column_width(list(range(prev.shape[1])))
plt.savefig(FIG / "panel_preview.png", bbox_inches="tight", facecolor="white"); plt.close()
