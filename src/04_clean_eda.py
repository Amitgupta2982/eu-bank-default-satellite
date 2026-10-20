"""Step 4: data cleaning log and exploratory data analysis (EDA) for the interim report.

Run:  python src/04_clean_eda.py
Outputs (reports/): cleaning_log.csv, missing_values.csv, outliers.csv, descriptives_by_class.csv,
          correlations.csv, figures/fig_*.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import INTERIM, PROCESSED, ROOT, FIG, EU_EEA, CLASSES, MIN_EXPOSURE_EUR_M

REP = ROOT / "reports"
FIG.mkdir(parents=True, exist_ok=True)
# validated categorical palette (dataviz reference slots 1-4); text stays in ink colours
PAL = {"Corporates": "#2a78d6", "Retail secured by real estate": "#eb6834",
       "Retail qualifying revolving": "#1baf7a", "Other retail": "#eda100"}
STY = {"Corporates": "-", "Retail secured by real estate": "--", "Retail qualifying revolving": ":", "Other retail": "-."}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
FEATS = ["unemp", "d_unemp", "gdp_growth", "hpi_growth", "inflation", "lt_rate",
         "log_assets", "cet1_ratio", "stage2_share"]


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- cleaning log
cells = pd.read_csv(INTERIM / "eba_irb_cells.csv")
codes = pd.read_csv(INTERIM / "eba_country_codes.csv", dtype=str, keep_default_na=False)
codes = codes.rename(columns={"Country": "country_code", "ISO_code": "cpty_country"})
codes[["exercise", "country_code"]] = codes[["exercise", "country_code"]].astype(int)
c = cells.merge(codes[["exercise", "country_code", "cpty_country"]], on=["exercise", "country_code"], how="left")
log = [("Extracted IRB cells (4 classes, country breakdown), 2021-2025 exercises", len(c))]
dups = c.duplicated(["lei", "period", "country_code", "exposure"]).sum()
log.append((f"Exact duplicate keys removed (bank, quarter, country, class): {dups}", len(c) - dups))
c = c[c.cpty_country.isin(EU_EEA)]
log.append(("Kept EU/EEA counterparty countries", len(c)))
c = c[c.orig >= MIN_EXPOSURE_EUR_M]
log.append((f"Removed near-empty books (original exposure < EUR {MIN_EXPOSURE_EUR_M:g} million)", len(c)))
p = pd.read_csv(PROCESSED / "panel_eu_irb.csv", parse_dates=["quarter_end"])
log.append(("Analysis panel (after joins and lags)", len(p)))
cc = p.dropna(subset=FEATS + ["d_der"])
log.append(("Complete records for RQ1-RQ2 (d_der and all lagged inputs present)", len(cc)))
r4 = cc.dropna(subset=["rise_next"])
log.append(("Complete records for RQ3-RQ4 (next-quarter target present)", len(r4)))
pd.DataFrame(log, columns=["step", "records"]).to_csv(REP / "cleaning_log.csv", index=False)

# ---------------------------------------------------------------- missing values
mv = []
first_q = p.groupby(["lei", "cpty_country", "exposure_class"]).quarter_end.transform("min") == p.quarter_end
for v in ["d_der"] + FEATS + ["ecb_rate_house"]:
    n = p[v].isna().sum()
    mv.append((v, n, round(100 * n / len(p), 2)))
mvdf = pd.DataFrame(mv, columns=["variable", "missing", "pct"])
mvdf.to_csv(REP / "missing_values.csv", index=False)
first_q_missing = int((p.d_der.isna() & first_q).sum())
gap_missing = int(p.d_der.isna().sum() - first_q_missing)
miss_ctry = p[p.lt_rate.isna()].cpty_country.value_counts().head(5).to_dict()
miss_s2 = p[p.stage2_share.isna()].lei.nunique()

# ---------------------------------------------------------------- outliers (IQR rule + winsorising)
q1, q3 = cc.d_der.quantile([.25, .75]); iqr = q3 - q1
lo_f, hi_f = q1 - 3 * iqr, q3 + 3 * iqr
p01, p99 = cc.d_der.quantile([.01, .99])
out = pd.DataFrame([
    ("d_der", round(q1, 3), round(q3, 3), int(((cc.d_der < lo_f) | (cc.d_der > hi_f)).sum()),
     round(p01, 3), round(p99, 3), int(((cc.d_der < p01) | (cc.d_der > p99)).sum())),
], columns=["variable", "q1", "q3", "beyond_3IQR", "p1", "p99", "winsorised"])
out.to_csv(REP / "outliers.csv", index=False)
small = cc[(cc.d_der < p01) | (cc.d_der > p99)].orig_exp.median()

# ---------------------------------------------------------------- descriptives
cc = cc.assign(d_der_w=cc.d_der.clip(p01, p99))
desc = cc.groupby("exposure_class").agg(records=("der", "size"), der_mean=("der", "mean"), der_median=("der", "median"),
                                        d_der_mean=("d_der_w", "mean"), d_der_sd=("d_der_w", "std"),
                                        rise_rate=("rise_next", "mean")).round(3)
desc.to_csv(REP / "descriptives_by_class.csv")
corr = cc[["d_der_w"] + FEATS].corr().round(2)
corr.to_csv(REP / "correlations.csv")

# ---------------------------------------------------------------- Figure: DER by class over time (exposure-weighted)
w = p.groupby(["quarter_end", "exposure_class"]).apply(
    lambda d: 100 * d.def_exp.sum() / d.orig_exp.sum(), include_groups=False).unstack()
fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=220)
for cl in CLASSES.values():
    ax.plot(w.index, w[cl], STY[cl], color=PAL[cl], lw=2, label=cl)
    ax.text(w.index[-1] + pd.Timedelta(days=25), w[cl].iloc[-1], f"{w[cl].iloc[-1]:.1f}%", va="center", fontsize=8, color=INK2)
ax.set_ylabel("Defaulted exposure ratio (%)"); style(ax)
ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
plt.tight_layout(); plt.savefig(FIG / "fig_der_by_class.png", facecolor="white"); plt.close()

# ---------------------------------------------------------------- Figure: distribution of d_der
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), dpi=220)
axes[0].hist(cc.d_der.clip(-5, 5), bins=80, color="#2a78d6", edgecolor="white", linewidth=0.4)
axes[0].set_title("Raw (display clipped at ±5 pp)", fontsize=9, color=INK)
axes[1].hist(cc.d_der_w, bins=60, color="#2a78d6", edgecolor="white", linewidth=0.4)
axes[1].set_title(f"Winsorised at 1st/99th percentile ({p01:.2f}, {p99:.2f})", fontsize=9, color=INK)
for a in axes:
    a.set_xlabel("Quarterly change in DER (pp)"); style(a)
axes[0].set_ylabel("Records")
plt.tight_layout(); plt.savefig(FIG / "fig_d_der_hist.png", facecolor="white"); plt.close()

# ---------------------------------------------------------------- Figure: binned d_der vs d_unemp (RQ1/RQ2)
cc["bin"] = pd.qcut(cc.d_unemp, 5, duplicates="drop")
fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=220)
for cl in ["Retail secured by real estate", "Other retail", "Corporates"]:
    g = cc[cc.exposure_class == cl].groupby("bin", observed=True).agg(x=("d_unemp", "mean"), y=("d_der_w", "mean"), n=("d_der_w", "size"),
                                                                        s=("d_der_w", "sem"))
    ax.errorbar(g.x, g.y, yerr=1.96 * g.s, fmt="o", ls=STY[cl], color=PAL[cl],
                lw=2, ms=6, capsize=2, label=cl)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xlabel("Year-on-year change in unemployment, lagged one quarter (pp), quintile means")
ax.set_ylabel("Mean change in DER (pp)"); style(ax)
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
plt.tight_layout(); plt.savefig(FIG / "fig_binned_unemp.png", facecolor="white"); plt.close()

# ---------------------------------------------------------------- Figure: correlation heatmap (diverging)
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f4f3f0", "#2a78d6"])
labels = ["d_der", "unemp", "d_unemp", "gdp", "hpi", "infl", "lt_rate", "log_assets", "cet1", "stage2"]
fig, ax = plt.subplots(figsize=(5.6, 4.6), dpi=220)
im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right"); ax.set_yticks(range(len(labels)), labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color=INK)
cb = plt.colorbar(im, fraction=0.04); cb.outline.set_visible(False)
plt.tight_layout(); plt.savefig(FIG / "fig_corr.png", facecolor="white"); plt.close()

# ---------------------------------------------------------------- Figure: share of rises by quarter (RQ4 balance)
rr = r4.groupby("quarter_end").rise_next.mean() * 100
fig, ax = plt.subplots(figsize=(7.2, 2.8), dpi=220)
cols = ["#2a78d6" if d < pd.Timestamp("2024-06-30") else "#eb6834" for d in rr.index]
ax.bar(rr.index, rr.values, width=60, color=cols)
ax.axhline(100 * r4.rise_next.mean(), color=INK2, lw=0.8, ls="--")
ax.text(rr.index[0], 100 * r4.rise_next.mean() + 0.8, f"Average {100 * r4.rise_next.mean():.1f}%", fontsize=8, color=INK2)
ax.set_ylabel("Records followed by a rise > 0.10 pp (%)"); style(ax)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#2a78d6", label="Training origins"), Patch(color="#eb6834", label="Test origins (Jun 2024 to Mar 2025)")],
          frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
plt.tight_layout(); plt.savefig(FIG / "fig_rise_rate.png", facecolor="white"); plt.close()

summary = dict(dups=int(dups), first_q_missing=first_q_missing, gap_missing=gap_missing, miss_lt_ctry=miss_ctry,
               miss_s2_banks=int(miss_s2), p01=float(p01), p99=float(p99), beyond3iqr=int(out.beyond_3IQR[0]),
               winsorised=int(out.winsorised[0]), small_median_exp=float(small), lo_f=float(lo_f), hi_f=float(hi_f))
pd.Series(summary).to_json(REP / "eda_summary.json")
print(pd.DataFrame(log, columns=["step", "records"]).to_string()); print(mvdf.to_string()); print(out.to_string())
print(desc.to_string()); print(summary); print(corr.loc["d_der_w"].to_string())
