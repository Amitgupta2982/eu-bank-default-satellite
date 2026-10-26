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
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11.5, "axes.edgecolor": INK2,
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
fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=220)
for cl in CLASSES.values():
    ax.plot(w.index, w[cl], STY[cl], color=PAL[cl], lw=2, label=cl)
    ax.text(w.index[-1] + pd.Timedelta(days=25), w[cl].iloc[-1], f"{w[cl].iloc[-1]:.1f}%", va="center", fontsize=10, color=INK2)
for cl, dates in {"Other retail": ["2020-09-30", "2022-12-31", "2024-12-31"], "Corporates": ["2020-09-30", "2023-03-31", "2024-09-30"]}.items():
    for d in dates:
        dt = pd.Timestamp(d); v = w.loc[dt, cl]
        ax.plot([dt], [v], "o", color=PAL[cl], ms=5)
        ax.annotate(f"{v:.1f}%", (dt, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9.5, color=INK)
ax.set_ylim(0.8, w.max().max() + 0.5)
ax.set_ylabel("Defaulted exposure ratio (%)"); ax.set_xlabel("Quarter end"); style(ax)
ax.legend(frameon=False, ncol=2, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.2))
plt.tight_layout(); plt.savefig(FIG / "fig_der_by_class.png", facecolor="white", bbox_inches="tight", pad_inches=0.1); plt.close()

# ---------------------------------------------------------------- Figure: distribution of d_der
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), dpi=220)
axes[0].hist(cc.d_der.clip(-5, 5), bins=np.linspace(-5, 5, 81), color="#2a78d6", edgecolor="white", linewidth=0.4)
axes[0].set_title("Raw values (shown within ±5 pp)", fontsize=11, color=INK)
axes[1].hist(cc.d_der_w, bins=np.linspace(-5, 5, 81), color="#2a78d6", edgecolor="white", linewidth=0.4)
axes[1].set_title("Capped at 1st and 99th percentiles", fontsize=11, color=INK)
for a in axes:
    a.set_xlabel("Quarterly change in DER (pp)"); style(a)
axes[0].set_ylabel("Records"); axes[1].set_ylabel("Records")
plt.tight_layout(); plt.savefig(FIG / "fig_d_der_hist.png", facecolor="white", bbox_inches="tight", pad_inches=0.1); plt.close()

# ---------------------------------------------------------------- Figure: binned d_der vs d_unemp (RQ1/RQ2)
cc["bin"] = pd.qcut(cc.d_unemp, 5, duplicates="drop")
fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=220)
for off, cl in zip([-0.045, -0.015, 0.015, 0.045], ["Retail secured by real estate", "Other retail", "Corporates", "Retail qualifying revolving"]):
    g = cc[cc.exposure_class == cl].groupby("bin", observed=True).agg(x=("d_unemp", "mean"), y=("d_der_w", "mean"), n=("d_der_w", "size"),
                                                                        s=("d_der_w", "sem"))
    ax.errorbar(g.x + off, g.y, yerr=1.96 * g.s, fmt="o", ls=STY[cl], color=PAL[cl],
                lw=2, ms=6, capsize=2, label=cl)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xlabel("Change in unemployment over the past year (pp), grouped into fifths")
ax.set_ylabel("Mean change in DER (pp)"); style(ax)
ax.legend(frameon=False, fontsize=10, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.24))
plt.tight_layout(); plt.savefig(FIG / "fig_binned_unemp.png", facecolor="white", bbox_inches="tight", pad_inches=0.1); plt.close()

# ---------------------------------------------------------------- Figure: correlation heatmap (diverging)
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f4f3f0", "#2a78d6"])
labels = ["Change in DER", "Unemployment", "Change in unemp.", "GDP growth", "House prices", "Inflation",
          "Long-term rate", "Bank size", "CET1 ratio", "Stage 2 share"]
fig, ax = plt.subplots(figsize=(7.6, 6.6), dpi=220)
mask = np.triu(np.ones_like(corr.values, dtype=bool), k=1)
im = ax.imshow(np.ma.array(corr.values, mask=mask), cmap=cmap, vmin=-1, vmax=1)
ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right"); ax.set_yticks(range(len(labels)), labels)
for i in range(len(labels)):
    for j in range(i + 1):
        ax.text(j, i, f"{(round(corr.values[i, j], 2) or 0.0):.2f}", ha="center", va="center", fontsize=11, color=INK)
ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
cb = plt.colorbar(im, fraction=0.04); cb.outline.set_visible(False)
plt.tight_layout(); plt.savefig(FIG / "fig_corr.png", facecolor="white", bbox_inches="tight", pad_inches=0.1); plt.close()

# ---------------------------------------------------------------- Figure: share of rises by quarter (RQ4 balance)
rr = r4.groupby("quarter_end").rise_next.mean() * 100
fig, ax = plt.subplots(figsize=(7.2, 3.5), dpi=220)
cols = ["#2a78d6" if d < pd.Timestamp("2024-06-30") else "#eb6834" for d in rr.index]
ax.bar(rr.index, rr.values, width=60, color=cols)
for d, v in zip(rr.index, rr.values):
    ax.text(d, v + 0.6, f"{v:.0f}", ha="center", va="bottom", fontsize=8.5, color=INK)
ax.axhline(100 * r4.rise_next.mean(), color=INK2, lw=0.8, ls="--")
ax.text(rr.index[-1] + pd.Timedelta(days=50), 100 * r4.rise_next.mean(), f"Average\n{100 * r4.rise_next.mean():.1f}%", fontsize=10, color=INK2, va="center")
import matplotlib.dates as mdates
ax.set_xticks(rr.index, [f"Q{(d.month - 1) // 3 + 1}-{str(d.year)[2:]}" for d in rr.index], rotation=90, fontsize=9)
ax.set_ylim(0, rr.max() + 5)
ax.set_ylabel("Followed by a rise (%)"); ax.set_xlabel("Forecast quarter"); style(ax)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#2a78d6", label="Training origins"), Patch(color="#eb6834", label="Test origins (Jun 2024 to Mar 2025)")],
          frameon=False, fontsize=10, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.42))
plt.tight_layout(); plt.savefig(FIG / "fig_rise_rate.png", facecolor="white", bbox_inches="tight", pad_inches=0.1); plt.close()

summary = dict(dups=int(dups), first_q_missing=first_q_missing, gap_missing=gap_missing, miss_lt_ctry=miss_ctry,
               miss_s2_banks=int(miss_s2), p01=float(p01), p99=float(p99), beyond3iqr=int(out.beyond_3IQR[0]),
               winsorised=int(out.winsorised[0]), small_median_exp=float(small), lo_f=float(lo_f), hi_f=float(hi_f))
pd.Series(summary).to_json(REP / "eda_summary.json")
print(pd.DataFrame(log, columns=["step", "records"]).to_string()); print(mvdf.to_string()); print(out.to_string())
print(desc.to_string()); print(summary); print(corr.loc["d_der_w"].to_string())
