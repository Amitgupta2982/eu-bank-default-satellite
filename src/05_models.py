"""Step 5: preliminary models for RQ1-RQ4 (interim report).

Run:  python src/05_models.py
Outputs (reports/): rq1_rq2_coefficients.csv, rq1_robustness.csv, vif.csv, rq3_results.csv, rq4_results.csv,
          tests.json, shap_importance.csv, figures/fig_rq2_slopes.png, fig_roc.png, fig_shap.png, fig_rq3_errors.png
"""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, f1_score, precision_score, recall_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import PROCESSED, ROOT, FIG, SEED, RISE_THRESHOLD_PP

REP = ROOT / "reports"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREEN, GOLD = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11.5, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
np.random.seed(SEED)

MACRO = ["unemp", "d_unemp", "gdp_growth", "hpi_growth", "inflation", "lt_rate"]
BANK = ["log_assets", "cet1_ratio", "stage2_share"]
FEATS = MACRO + BANK
CLS = ["Corporates", "Retail secured by real estate", "Retail qualifying revolving", "Other retail"]
SHORT = {"Corporates": "Corp", "Retail secured by real estate": "RE", "Retail qualifying revolving": "QRRE",
         "Other retail": "OthRet"}
TRAIN_END = pd.Timestamp("2024-03-31")   # last training forecast origin (target quarter Jun 2024)
VAL_START = pd.Timestamp("2023-06-30")   # tuning hold-out inside the training period: origins Jun 2023 - Mar 2024


def style(ax, axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


p = pd.read_csv(PROCESSED / "panel_eu_irb.csv", parse_dates=["quarter_end"])
p["fe"] = p.lei + "|" + p.exposure_class
p["cell"] = p.lei + "|" + p.cpty_country + "|" + p.exposure_class
for c in CLS:
    p[SHORT[c]] = (p.exposure_class == c).astype(int)
cc = p.dropna(subset=FEATS + ["d_der"]).copy()
lo, hi = cc.d_der.quantile([.01, .99])
cc["y"] = cc.d_der.clip(lo, hi)


# ================================================================ RQ1 / RQ2: fixed-effects regressions
def within(df, cols, fe="fe"):
    """Remove bank x class means (fixed effects by the within transformation)."""
    return df[cols] - df.groupby(fe)[cols].transform("mean")


def fe_ols(df, xcols, y="y"):
    d = within(df, [y] + xcols)
    n_fe = df.fe.nunique()
    m = sm.OLS(d[y], d[xcols]).fit(cov_type="cluster", cov_kwds={"groups": df.cpty_country.astype("category").cat.codes})
    # small-sample correction for the absorbed fixed effects (as in Stata xtreg/areg)
    k = len(xcols)
    adj = np.sqrt((len(df) - k) / (len(df) - k - n_fe))
    se = m.bse * adj
    ss_w = (d[y] ** 2).sum()
    r2w = 1 - (m.resid ** 2).sum() / ss_w
    return m.params, se, r2w, m


X1 = FEATS + ["cross_border", "trend"]
b1, se1, r2w1, m1 = fe_ols(cc, X1)
G = cc.cpty_country.nunique()
tcrit = stats.t.ppf(.975, G - 1)


def coef_table(b, se, label):
    t = b / se
    pv = 2 * stats.t.sf(np.abs(t), G - 1)
    return pd.DataFrame({"model": label, "term": b.index, "coef": b.values, "se": se.values, "t": t.values,
                         "p": pv, "ci_lo": (b - tcrit * se).values, "ci_hi": (b + tcrit * se).values})


rq1 = coef_table(b1, se1, "RQ1")

# wild cluster bootstrap (Rademacher, restricted under H0: beta_d_unemp = 0) for the key coefficient
d1 = within(cc, ["y"] + X1)
groups = cc.cpty_country.values
ug = np.unique(groups)
gidx = np.searchsorted(ug, groups)
Xr = d1[[c for c in X1 if c != "d_unemp"]].values
fit_r = np.linalg.lstsq(Xr, d1.y.values, rcond=None)[0]
yhat_r, u_r = Xr @ fit_r, d1.y.values - Xr @ fit_r
Xf = d1[X1].values
XtX_inv = np.linalg.inv(Xf.T @ Xf)
j = X1.index("d_unemp")


def cl_t(yv):
    bb = XtX_inv @ Xf.T @ yv
    e = yv - Xf @ bb
    S = np.zeros((Xf.shape[1], Xf.shape[1]))
    sc = np.zeros((len(ug), Xf.shape[1]))
    np.add.at(sc, gidx, Xf * e[:, None])
    V = XtX_inv @ (sc.T @ sc) @ XtX_inv
    return bb[j] / np.sqrt(V[j, j])


t_obs = cl_t(d1.y.values)
rng = np.random.default_rng(SEED)
t_b = np.array([cl_t(yhat_r + u_r * rng.choice([-1, 1], len(ug))[gidx]) for _ in range(999)])
p_wild = float((np.abs(t_b) >= abs(t_obs)).mean())

# RQ2: unemployment slope by class (reference = Retail secured by real estate)
for c in ["Corp", "QRRE", "OthRet"]:
    cc[f"dU_x_{c}"] = cc.d_unemp * cc[c]
X2 = X1 + ["dU_x_Corp", "dU_x_QRRE", "dU_x_OthRet"]
b2, se2, r2w2, m2 = fe_ols(cc, X2)
rq2 = coef_table(b2, se2, "RQ2")
# class-specific slopes with clustered SEs (linear combinations)
V2 = m2.cov_params() * ((len(cc) - len(X2)) / (len(cc) - len(X2) - cc.fe.nunique()))
slopes = []
for c, term in [("Retail secured by real estate", None), ("Corporates", "dU_x_Corp"),
                ("Retail qualifying revolving", "dU_x_QRRE"), ("Other retail", "dU_x_OthRet")]:
    w = pd.Series(0.0, index=X2); w["d_unemp"] = 1
    if term:
        w[term] = 1
    est, s = float(w @ b2), float(np.sqrt(w @ V2 @ w))
    slopes.append((c, est, s, est - tcrit * s, est + tcrit * s, int((cc.exposure_class == c).sum())))
slopes = pd.DataFrame(slopes, columns=["class", "slope", "se", "ci_lo", "ci_hi", "n"])

pd.concat([rq1, rq2]).round(4).to_csv(REP / "rq1_rq2_coefficients.csv", index=False)
slopes.round(4).to_csv(REP / "rq2_class_slopes.csv", index=False)

# multicollinearity check
vif = pd.DataFrame({"variable": X1, "vif": [variance_inflation_factor(sm.add_constant(cc[X1]).values, i + 1)
                                              for i in range(len(X1))]}).round(2)
vif.to_csv(REP / "vif.csv", index=False)

# robustness: (a) books >= EUR 10m; (b) ECB lending rate instead of long-term yield (euro area + BG)
rob = []
s10 = cc[cc.orig_exp >= 10]
b, s, r2, _ = fe_ols(s10, X1)
rob.append(("Books >= EUR 10 million", len(s10), b["d_unemp"], s["d_unemp"], r2))
ea = cc.dropna(subset=["ecb_rate_house"])
Xe = [c if c != "lt_rate" else "ecb_rate_house" for c in X1]
b, s, r2, _ = fe_ols(ea, Xe)
rob.append(("ECB house-purchase rate replaces lt_rate (euro area + BG)", len(ea), b["d_unemp"], s["d_unemp"], r2))
dc = cc.assign(y=cc.d_der)  # no winsorising
b, s, r2, _ = fe_ols(dc, X1)
rob.append(("Raw d_der (not winsorised)", len(dc), b["d_unemp"], s["d_unemp"], r2))
rob = pd.DataFrame(rob, columns=["specification", "n", "coef_d_unemp", "se", "r2_within"])
rob["p"] = 2 * stats.t.sf(np.abs(rob.coef_d_unemp / rob.se), G - 1)
rob.round(4).to_csv(REP / "rq1_robustness.csv", index=False)

# ================================================================ RQ3 / RQ4 data: out-of-time split
r = cc.dropna(subset=["d_der_next", "rise_next"]).copy()
tr = r[r.quarter_end <= TRAIN_END].copy()
te = r[r.quarter_end > TRAIN_END].copy()
# winsorising limits from the training period only (no look-ahead)
lo_t, hi_t = tr.d_der_next.quantile([.01, .99])
lo_x, hi_x = tr.d_der.quantile([.01, .99])
for d in (tr, te):
    d["y_next"] = d.d_der_next.clip(lo_t, hi_t)
    d["d_der_c"] = d.d_der.clip(lo_x, hi_x)
F = FEATS + ["der", "d_der_c", "cross_border", "Corp", "QRRE", "OthRet"]
fit_tr, val = tr[tr.quarter_end < VAL_START], tr[tr.quarter_end >= VAL_START]


# ---------------------------------------------------------------- RQ3 regression
def lin_fit(a, cols=F):
    return sm.OLS(a.y_next, sm.add_constant(a[cols])).fit()


lin = lin_fit(tr)
pred_lin = lin.predict(sm.add_constant(te[F]))

grid = [(d, lr, mcw) for d in (2, 3, 4) for lr in (0.03, 0.1) for mcw in (5, 20)]
best = None
for d, lr, mcw in grid:
    m = xgb.XGBRegressor(n_estimators=1500, max_depth=d, learning_rate=lr, min_child_weight=mcw, subsample=0.8,
                         colsample_bytree=0.8, reg_lambda=1.0, objective="reg:squarederror", random_state=SEED,
                         early_stopping_rounds=50, eval_metric="rmse", n_jobs=4)
    m.fit(fit_tr[F], fit_tr.y_next, eval_set=[(val[F], val.y_next)], verbose=False)
    sc = m.best_score
    if best is None or sc < best[0]:
        best = (sc, d, lr, mcw, m.best_iteration + 1)
_, d_, lr_, mcw_, nt_ = best
reg = xgb.XGBRegressor(n_estimators=nt_, max_depth=d_, learning_rate=lr_, min_child_weight=mcw_, subsample=0.8,
                       colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=4)
reg.fit(tr[F], tr.y_next)
pred_xgb = reg.predict(te[F])
val_lin_rmse = float(np.sqrt(((val.y_next - lin_fit(fit_tr).predict(sm.add_constant(val[F]))) ** 2).mean()))

yt = te.y_next.values
preds = {"No-change baseline (0)": np.zeros(len(te)), "Training-mean baseline": np.full(len(te), tr.y_next.mean()),
         "Linear satellite (OLS)": pred_lin.values, "XGBoost regressor": pred_xgb}
rq3 = pd.DataFrame([(k, np.sqrt(((yt - v) ** 2).mean()), np.abs(yt - v).mean()) for k, v in preds.items()],
                   columns=["model", "rmse", "mae"])
rq3["oos_r2"] = 1 - (rq3.rmse ** 2) / ((yt - yt.mean()) ** 2).mean()
rq3.round(4).to_csv(REP / "rq3_results.csv", index=False)


def dm_test(e1, e2, cluster, loss="abs"):
    """Diebold-Mariano: d = L(e_linear) - L(e_ML); variance clustered by cell (repeated cells in the test period)."""
    L = np.abs if loss == "abs" else np.square
    dd = L(e1) - L(e2)
    mm = sm.OLS(dd, np.ones(len(dd))).fit(cov_type="cluster", cov_kwds={"groups": pd.Series(cluster).astype("category").cat.codes})
    return float(dd.mean()), float(mm.tvalues[0]), float(mm.pvalues[0])


e_lin, e_xgb = yt - pred_lin.values, yt - pred_xgb
dm_abs = dm_test(e_lin, e_xgb, te.cell.values, "abs")
dm_sq = dm_test(e_lin, e_xgb, te.cell.values, "sq")

# error by class (figure)
te["ae_lin"], te["ae_xgb"], te["ae_0"] = np.abs(e_lin), np.abs(e_xgb), np.abs(yt)
mae_cls = te.groupby("exposure_class")[["ae_0", "ae_lin", "ae_xgb"]].mean().reindex(CLS)

# ---------------------------------------------------------------- RQ4 classification
ytr, yte = tr.rise_next.values, te.rise_next.values
spw = (ytr == 0).sum() / (ytr == 1).sum()
scaler = StandardScaler().fit(fit_tr[F])
logit_v = LogisticRegression(class_weight="balanced", max_iter=2000).fit(scaler.transform(fit_tr[F]), fit_tr.rise_next)

bestc = None
for d, lr, mcw in grid:
    m = xgb.XGBClassifier(n_estimators=1500, max_depth=d, learning_rate=lr, min_child_weight=mcw, subsample=0.8,
                          colsample_bytree=0.8, scale_pos_weight=spw, random_state=SEED, early_stopping_rounds=50,
                          eval_metric="auc", n_jobs=4)
    m.fit(fit_tr[F], fit_tr.rise_next, eval_set=[(val[F], val.rise_next)], verbose=False)
    if bestc is None or m.best_score > bestc[0]:
        bestc = (m.best_score, d, lr, mcw, m.best_iteration + 1, m)
_, dc_, lrc_, mcwc_, ntc_, mval = bestc


def best_thr(y, s):
    ths = np.quantile(s, np.linspace(0.05, 0.95, 91))
    f = [f1_score(y, s >= t) for t in ths]
    return float(ths[int(np.argmax(f))])


thr_log = best_thr(val.rise_next.values, logit_v.predict_proba(scaler.transform(val[F]))[:, 1])
thr_xgb = best_thr(val.rise_next.values, mval.predict_proba(val[F])[:, 1])

scaler = StandardScaler().fit(tr[F])
logit = LogisticRegression(class_weight="balanced", max_iter=2000).fit(scaler.transform(tr[F]), ytr)
clf = xgb.XGBClassifier(n_estimators=ntc_, max_depth=dc_, learning_rate=lrc_, min_child_weight=mcwc_, subsample=0.8,
                        colsample_bytree=0.8, scale_pos_weight=spw, random_state=SEED, n_jobs=4)
clf.fit(tr[F], ytr)
s_log = logit.predict_proba(scaler.transform(te[F]))[:, 1]
s_xgb = clf.predict_proba(te[F])[:, 1]
# calibrated (unweighted) logit for the Brier comparison: class weights shift probabilities upward
logit_u = LogisticRegression(max_iter=2000).fit(scaler.transform(tr[F]), ytr)
s_logu = logit_u.predict_proba(scaler.transform(te[F]))[:, 1]
clf_u = xgb.XGBClassifier(n_estimators=ntc_, max_depth=dc_, learning_rate=lrc_, min_child_weight=mcwc_, subsample=0.8,
                          colsample_bytree=0.8, random_state=SEED, n_jobs=4).fit(tr[F], ytr)
s_xgbu = clf_u.predict_proba(te[F])[:, 1]


def delong(y, s1, s2):
    """DeLong et al. (1988) test for two correlated AUCs (fast midrank form, Sun & Xu, 2014)."""
    pos, neg = s1[y == 1], s1[y == 0]
    def comps(sc):
        P, N = sc[y == 1], sc[y == 0]
        m_, n_ = len(P), len(N)
        allr = stats.rankdata(np.r_[P, N])
        v10 = (allr[:m_] - stats.rankdata(P)) / n_
        v01 = 1 - (allr[m_:] - stats.rankdata(N)) / m_
        return v10.mean(), v10, v01
    a1, v10a, v01a = comps(s1)
    a2, v10b, v01b = comps(s2)
    S10 = np.cov(np.vstack([v10a, v10b]))
    S01 = np.cov(np.vstack([v01a, v01b]))
    S = S10 / len(v10a) + S01 / len(v01a)
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (a1 - a2) / np.sqrt(var)
    return float(a1), float(a2), float(z), float(2 * stats.norm.sf(abs(z))), float(np.sqrt(var))


auc_x, auc_l, z_dl, p_dl, se_dl = delong(yte, s_xgb, s_log)
base = yte.mean()
rows = [("Naive: flag all", 0.5, base * (1 - base), base, 1.0, 2 * base / (1 + base))]
conf = {}
for name, s, thr in [("Logistic regression", s_log, thr_log), ("XGBoost classifier", s_xgb, thr_xgb)]:
    yp = s >= thr
    conf[name] = dict(tp=int((yp & (yte == 1)).sum()), fp=int((yp & (yte == 0)).sum()),
                      fn=int((~yp & (yte == 1)).sum()), tn=int((~yp & (yte == 0)).sum()))
    rows.append((name, roc_auc_score(yte, s), brier_score_loss(yte, s), precision_score(yte, yp),
                 recall_score(yte, yp), f1_score(yte, yp)))
rows.append(("Logistic regression (unweighted, calibration check)", roc_auc_score(yte, s_logu),
             brier_score_loss(yte, s_logu), np.nan, np.nan, np.nan))
rows.append(("XGBoost classifier (unweighted, calibration check)", roc_auc_score(yte, s_xgbu),
             brier_score_loss(yte, s_xgbu), np.nan, np.nan, np.nan))
rq4 = pd.DataFrame(rows, columns=["model", "auc", "brier", "precision", "recall", "f1"])
rq4.round(4).to_csv(REP / "rq4_results.csv", index=False)

# ---------------------------------------------------------------- SHAP
NICE = {"unemp": "Unemployment rate", "d_unemp": "Change in unemployment", "gdp_growth": "GDP growth",
        "hpi_growth": "House price growth", "inflation": "Inflation", "lt_rate": "Long-term rate",
        "log_assets": "Bank size (log assets)", "cet1_ratio": "CET1 ratio", "stage2_share": "Stage 2 share",
        "der": "Current DER level", "d_der_c": "Current change in DER", "cross_border": "Cross-border flag",
        "Corp": "Class: Corporates", "QRRE": "Class: QRRE", "OthRet": "Class: Other retail"}
sv_c = shap.TreeExplainer(clf).shap_values(te[F])
sv_r = shap.TreeExplainer(reg).shap_values(te[F])
imp = pd.DataFrame({"feature": F, "rq3_regressor": np.abs(sv_r).mean(0), "rq4_classifier": np.abs(sv_c).mean(0)})
imp["rq3_share"] = imp.rq3_regressor / imp.rq3_regressor.sum()
imp["rq4_share"] = imp.rq4_classifier / imp.rq4_classifier.sum()
imp.sort_values("rq4_share", ascending=False).round(4).to_csv(REP / "shap_importance.csv", index=False)

# ---------------------------------------------------------------- figures
# Figure: RQ2 class slopes with 95% CI
fig, ax = plt.subplots(figsize=(7.2, 3.5), dpi=220)
sl = slopes.set_index("class").reindex(CLS[::-1])
yy = np.arange(len(sl))
ax.errorbar(sl.slope, yy, xerr=[sl.slope - sl.ci_lo, sl.ci_hi - sl.slope], fmt="o", color=BLUE, ecolor=BLUE,
            ms=6, lw=2, capsize=3)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_yticks(yy, [f"{c}\n(n = {n:,})" for c, n in zip(sl.index, sl.n)])
for yi, (v, hi_) in enumerate(zip(sl.slope, sl.ci_hi)):
    ax.text(hi_ + 0.004, yi, f"{v:+.3f}", va="center", fontsize=10, color=INK2)
ax.set_xlabel("Change in DER (pp) per 1 pp rise in unemployment")
ax.set_xlim(-0.065, 0.06)
style(ax, "x")
plt.tight_layout(); plt.savefig(FIG / "fig_rq2_slopes.png", facecolor="white", bbox_inches="tight", pad_inches=0.1); plt.close()

# Figure: ROC curves
fig, ax = plt.subplots(figsize=(5.4, 5.0), dpi=220)
for name, s, col, ls in [("XGBoost", s_xgb, BLUE, "-"), ("Logistic regression", s_log, ORANGE, "--")]:
    fpr, tpr, _ = roc_curve(yte, s)
    ax.plot(fpr, tpr, ls, color=col, lw=2, label=f"{name} (AUC = {roc_auc_score(yte, s):.3f})")
ax.plot([0, 1], [0, 1], ":", color=INK2, lw=1, label="No skill (AUC = 0.500)")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.legend(frameon=False, fontsize=10, loc="lower right"); style(ax, "both")
plt.tight_layout(); plt.savefig(FIG / "fig_roc.png", facecolor="white", bbox_inches="tight", pad_inches=0.1); plt.close()

# Figure: SHAP importance (both models), with value labels
o = imp.sort_values("rq4_share")
NICE["log_assets"] = "Bank size"
fig, ax = plt.subplots(figsize=(6.6, 7.2), dpi=220)
yy = np.arange(len(o)); h = 0.4
ax.barh(yy + h / 2, 100 * o.rq4_share, h, color=BLUE, label="RQ4 classifier (rise next quarter)")
ax.barh(yy - h / 2, 100 * o.rq3_share, h, color=ORANGE, label="RQ3 regressor (change next quarter)")
for yv, a4, a3 in zip(yy, 100 * o.rq4_share, 100 * o.rq3_share):
    ax.text(a4 + 0.4, yv + h / 2, f"{a4:.1f}", va="center", fontsize=10, color=INK)
    ax.text(a3 + 0.4, yv - h / 2, f"{a3:.1f}", va="center", fontsize=10, color=INK)
ax.set_yticks(yy, [NICE[f] for f in o.feature])
ax.set_xlabel("Share of total importance (mean |SHAP|, %)"); style(ax, "x"); ax.set_xlim(0, 44)
ax.legend(frameon=False, fontsize=10.5, loc="lower right")
plt.tight_layout(); plt.savefig(FIG / "fig_shap.png", facecolor="white", bbox_inches="tight", pad_inches=0.1); plt.close()

# Figure: RQ3 MAE by class (horizontal bars, value labels; with the no-change baseline)
fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=220)
yy = np.arange(len(CLS))[::-1] * 1.0; h = 0.26
for k, (col, lab, c) in enumerate([("ae_0", "No-change baseline", GREEN), ("ae_lin", "Linear satellite (OLS)", ORANGE),
                                    ("ae_xgb", "XGBoost regressor", BLUE)]):
    pos = yy + (1 - k) * h
    ax.barh(pos, mae_cls[col], h * 0.88, color=c, label=lab)
    for py, v in zip(pos, mae_cls[col]):
        ax.text(v + 0.008, py, f"{v:.3f}", va="center", ha="left", fontsize=11, color=INK)
ax.set_yticks(yy, ["Corporates", "Retail secured\nby real estate", "Retail qualifying\nrevolving", "Other retail"])
ax.set_xlabel("Test mean absolute error (pp) — shorter bar = more accurate"); ax.set_xlim(0, 0.78); style(ax, "x")
ax.legend(frameon=False, fontsize=11, ncol=3, loc="lower center", bbox_to_anchor=(0.42, 1.0), handlelength=1.2, columnspacing=1.0)
plt.savefig(FIG / "fig_rq3_errors.png", facecolor="white", bbox_inches="tight", pad_inches=0.1); plt.close()


# ---------------------------------------------------------------- extra checks (interim v3)
# RQ3 on the raw (uncapped) test target
y_raw = te.d_der_next.values
raw = {k: (float(np.sqrt(((y_raw - v) ** 2).mean())), float(np.abs(y_raw - v).mean())) for k, v in preds.items()}
# RQ4: cell-block bootstrap of the AUC difference (records of the same cell resampled together)
cells_te = te.cell.values
uc = np.unique(cells_te)
idx_by_cell = {c: np.where(cells_te == c)[0] for c in uc}
rng_b = np.random.default_rng(SEED)
diffs = []
for _ in range(999):
    pick = np.concatenate([idx_by_cell[c] for c in rng_b.choice(uc, len(uc))])
    if yte[pick].min() == yte[pick].max():
        continue
    diffs.append(roc_auc_score(yte[pick], s_xgb[pick]) - roc_auc_score(yte[pick], s_log[pick]))
diffs = np.array(diffs)
boot_auc = dict(ci_lo=float(np.percentile(diffs, 2.5)), ci_hi=float(np.percentile(diffs, 97.5)),
                share_le0=float((diffs <= 0).mean()), se=float(diffs.std()))
# RQ1 residual diagnostics
res1 = m1.resid
d1r = cc.assign(res=res1.values).sort_values("quarter_end")
d1r["res_lag"] = d1r.groupby("cell").res.shift(1)
dd = d1r.dropna(subset=["res_lag"])
ar1_res = float(np.corrcoef(dd.res, dd.res_lag)[0, 1])
diag = dict(resid_skew=float(stats.skew(res1)), resid_kurt=float(stats.kurtosis(res1)), resid_ar1=ar1_res,
            country_quarters=int(cc.groupby(["cpty_country", "quarter_end"]).ngroups))
json.dump(dict(raw_target=raw, boot_auc=boot_auc, diag=diag), open(REP / "extra_checks.json", "w"), indent=1)
print(raw, boot_auc, diag)

tests = dict(n_rq1=len(cc), n_fe=int(cc.fe.nunique()), clusters=int(G), r2w_rq1=float(r2w1), r2w_rq2=float(r2w2),
             t_obs=float(t_obs), p_wild=p_wild, win_lo=float(lo), win_hi=float(hi),
             n_train=len(tr), n_test=len(te), n_fit=len(fit_tr), n_val=len(val), test_cells=int(te.cell.nunique()),
             xgb_reg=dict(depth=d_, lr=lr_, mcw=mcw_, n_trees=nt_, val_rmse=float(best[0]), val_lin_rmse=val_lin_rmse),
             xgb_clf=dict(depth=dc_, lr=lrc_, mcw=mcwc_, n_trees=ntc_, val_auc=float(bestc[0])),
             thr_log=thr_log, thr_xgb=thr_xgb, spw=float(spw), test_rise_rate=float(base), train_rise_rate=float(ytr.mean()),
             dm_abs=dict(dbar=dm_abs[0], stat=dm_abs[1], p=dm_abs[2]), dm_sq=dict(dbar=dm_sq[0], stat=dm_sq[1], p=dm_sq[2]),
             delong=dict(auc_xgb=auc_x, auc_log=auc_l, z=z_dl, p=p_dl, se=se_dl, diff=auc_x - auc_l),
             lin_r2_train=float(lin.rsquared), confusion=conf, win_next=[float(lo_t), float(hi_t)])
json.dump(tests, open(REP / "tests.json", "w"), indent=1)

pd.set_option("display.width", 200)
print(rq1.round(4).to_string()); print(rq2.round(4).tail(4).to_string()); print(slopes.round(4).to_string())
print(vif.to_string()); print(rob.round(4).to_string()); print(rq3.round(4).to_string()); print(rq4.round(4).to_string())
print(imp.sort_values("rq4_share", ascending=False).round(3).to_string()); print(mae_cls.round(4))
print(json.dumps(tests, indent=1))
