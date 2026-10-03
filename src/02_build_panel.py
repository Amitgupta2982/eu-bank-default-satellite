"""Step 2: build the analysis panel (bank x counterparty country x IRB class x quarter).

Run:  python src/02_build_panel.py
Inputs : data/interim/eba_irb_cells.csv, eba_bank_items.csv, eba_country_codes.csv,
         data/raw/eurostat_macro_long.csv, data/raw/ecb_mir_long.csv
Output : data/processed/panel_eu_irb.csv
"""
import numpy as np
import pandas as pd
from config import INTERIM, RAW, PROCESSED, CLASSES, EU_EEA, MIN_EXPOSURE_EUR_M, RISE_THRESHOLD_PP

EUROSTAT_GEO = {"GR": "EL"}  # Eurostat uses EL for Greece


def quarter(period):
    """202409 -> pandas Period 2024Q3."""
    p = str(int(period))
    return pd.Period(year=int(p[:4]), quarter=(int(p[4:]) - 1) // 3 + 1, freq="Q")


def macro_table():
    m = pd.read_csv(RAW / "eurostat_macro_long.csv")
    m["value"] = pd.to_numeric(m["value"], errors="coerce")
    monthly = m[m.dataset == "prc_hicp_manr"].copy()
    monthly["q"] = pd.PeriodIndex(pd.to_datetime(monthly["time"]), freq="Q")
    infl = monthly.groupby(["geo", "q"])["value"].mean().rename("inflation")
    qd = m[m.dataset != "prc_hicp_manr"].copy()
    qd["q"] = pd.PeriodIndex(qd["time"].str.replace("-", ""), freq="Q")
    wide = qd.pivot_table(index=["geo", "q"], columns="dataset", values="value")
    wide = wide.rename(columns={"une_rt_q": "unemp", "namq_10_gdp": "gdp_growth",
                                "prc_hpi_q": "hpi_growth", "irt_lt_mcby_q": "lt_rate"})
    wide = wide.join(infl, how="outer").reset_index().sort_values(["geo", "q"])
    # year-on-year change in unemployment (pp)
    wide["d_unemp"] = wide.groupby("geo")["unemp"].diff(4)
    e = pd.read_csv(RAW / "ecb_mir_long.csv")
    e["q"] = pd.PeriodIndex(pd.to_datetime(e["time"]), freq="Q")
    ecb = e.pivot_table(index=["geo", "q"], columns="series", values="value", aggfunc="mean")
    ecb = ecb.rename(columns={"hh_house": "ecb_rate_house", "nfc": "ecb_rate_nfc"}).reset_index()
    ecb["geo"] = ecb["geo"].replace({"GR": "EL"})
    return wide.merge(ecb, on=["geo", "q"], how="left")


def bank_controls():
    b = pd.read_csv(INTERIM / "eba_bank_items.csv")
    b["q"] = b["period"].map(quarter)
    ta = b[b.item == "total_assets"].groupby(["lei", "q"])["amount"].first().rename("total_assets")
    cet1 = b[b.item == "cet1_ratio"].groupby(["lei", "q"])["amount"].first().rename("cet1_ratio")
    gl = b[b.item == "gross_loans_ac"].pivot_table(index=["lei", "q"], columns="stage", values="amount", aggfunc="first")
    s2 = (100 * gl[2] / gl[[1, 2, 3]].sum(axis=1)).rename("stage2_share")
    out = pd.concat([ta, cet1, s2], axis=1).reset_index()
    out["log_assets"] = np.log(out["total_assets"])
    # CET1 is reported as a ratio (e.g. 0.15); convert to %
    if out["cet1_ratio"].median() < 1:
        out["cet1_ratio"] = out["cet1_ratio"] * 100
    return out[["lei", "q", "log_assets", "cet1_ratio", "stage2_share"]]


def build():
    c = pd.read_csv(INTERIM / "eba_irb_cells.csv")
    codes = pd.read_csv(INTERIM / "eba_country_codes.csv", dtype=str, keep_default_na=False)
    codes = codes.rename(columns={"Country": "country_code", "ISO_code": "cpty_country"})
    codes["exercise"] = codes["exercise"].astype(int)
    codes["country_code"] = codes["country_code"].astype(int)
    c = c.merge(codes[["exercise", "country_code", "cpty_country"]], on=["exercise", "country_code"], how="left")
    c = c[c.cpty_country.isin(EU_EEA)].copy()
    c["exposure_class"] = c["exposure"].map(CLASSES)
    c["q"] = c["period"].map(quarter)
    c = c[(c.orig >= MIN_EXPOSURE_EUR_M) & c.defaulted.notna()].copy()
    c["der"] = 100 * c["defaulted"] / c["orig"]
    c = c.sort_values(["lei", "cpty_country", "exposure", "q"])
    g = c.groupby(["lei", "cpty_country", "exposure"])
    prev_q, prev_der = g["q"].shift(1), g["der"].shift(1)
    consecutive = (c["q"] - prev_q).map(lambda x: getattr(x, "n", np.nan)) == 1
    c["d_der"] = np.where(consecutive, c["der"] - prev_der, np.nan)
    nxt_q, nxt_d = g["q"].shift(-1), g["d_der"].shift(-1)
    nxt_ok = (nxt_q - c["q"]).map(lambda x: getattr(x, "n", np.nan)) == 1
    c["rise_next"] = np.where(nxt_ok & nxt_d.notna(), (nxt_d > RISE_THRESHOLD_PP).astype(float), np.nan)
    c["d_der_next"] = np.where(nxt_ok, nxt_d, np.nan)
    # macro and bank controls lagged one quarter
    c["q_lag"] = c["q"] - 1
    c["geo"] = c["cpty_country"].replace({"GR": "EL"})
    mac = macro_table().rename(columns={"q": "q_lag"})
    c = c.merge(mac, on=["geo", "q_lag"], how="left")
    bc = bank_controls().rename(columns={"q": "q_lag"})
    c = c.merge(bc, on=["lei", "q_lag"], how="left")
    c["cross_border"] = (c["cpty_country"] != c["nsa"]).astype(int)
    qs = sorted(c["q"].unique())
    c["trend"] = c["q"].map({q: i + 1 for i, q in enumerate(qs)})
    c["quarter_end"] = c["q"].dt.end_time.dt.date
    cols = ["lei", "nsa", "cpty_country", "exposure_class", "quarter_end", "trend", "orig", "defaulted", "der",
            "d_der", "d_der_next", "rise_next", "unemp", "d_unemp", "gdp_growth", "hpi_growth", "inflation",
            "lt_rate", "ecb_rate_house", "ecb_rate_nfc", "log_assets", "cet1_ratio", "stage2_share", "cross_border"]
    out = c[cols].rename(columns={"orig": "orig_exp", "defaulted": "def_exp"})
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROCESSED / "panel_eu_irb.csv", index=False, float_format="%.6g")
    return out


if __name__ == "__main__":
    p = build()
    print(p.shape)
