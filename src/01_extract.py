"""Step 1: download raw EBA, Eurostat and ECB data and write compact interim files.

Run:  python src/01_extract.py
Raw EBA files are large (about 120 MB each) and are not committed to Git; they are
re-downloaded into data/raw/eba/<year>/. The compact outputs in data/interim/ are committed.
"""
import csv, io, json, urllib.request
import pandas as pd
from config import RAW, INTERIM, EBA_URLS, CRE_ITEMS, OTH_ITEMS, CLASSES, IRB_PORTFOLIO, EUROSTAT, ECB_MIR


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print("downloading", url)
        urllib.request.urlretrieve(url, dest)
    return dest


def extract_cre(year):
    path = download(EBA_URLS[year] + "tr_cre.csv", RAW / "eba" / str(year) / "tr_cre.csv")
    cells = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            suf = r["Item"][-5:]
            if suf not in CRE_ITEMS or r["Portfolio"] != str(IRB_PORTFOLIO) or r["Country"] == "0":
                continue
            if int(r["Exposure"]) not in CLASSES:
                continue
            key = (r["LEI_Code"], r["NSA"], r["Period"], r["Country"], r["Exposure"])
            cells.setdefault(key, {})[CRE_ITEMS[suf]] = r["Amount"]
    rows = [dict(lei=k[0], nsa=k[1], period=k[2], country_code=k[3], exposure=k[4],
                 orig=v.get("orig"), defaulted=v.get("def"), exercise=year) for k, v in cells.items()]
    return pd.DataFrame(rows)


def extract_oth(year):
    path = download(EBA_URLS[year] + "tr_oth.csv", RAW / "eba" / str(year) / "tr_oth.csv")
    df = pd.read_csv(path, dtype=str)
    df["suf"] = df["Item"].str[-5:]
    df = df[df["suf"].isin(OTH_ITEMS)]
    stage = df["ASSETS_Stages"] if "ASSETS_Stages" in df else ""
    return pd.DataFrame({"lei": df["LEI_Code"], "period": df["Period"], "item": df["suf"].map(OTH_ITEMS),
                         "stage": stage, "amount": df["Amount"], "exercise": year})


def country_map(year):
    path = download(EBA_URLS[year] + "TR_Metadata.xlsx", RAW / "eba" / str(year) / "TR_Metadata.xlsx")
    m = pd.read_excel(path, sheet_name="Country", dtype=str)
    m.columns = [c.strip() for c in m.columns]
    return m.assign(exercise=year)[["exercise", "Country", "Label", "ISO_code"]]


def eurostat():
    rows = []
    for ds, q in EUROSTAT.items():
        since = "2019-01" if ds == "prc_hicp_manr" else "2019-Q1"
        url = f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{ds}?{q}&sinceTimePeriod={since}"
        j = json.load(urllib.request.urlopen(url))
        ids, size = j["id"], j["size"]
        gi, ti = ids.index("geo"), ids.index("time")
        stride = [1] * len(ids)
        for k in range(len(ids) - 2, -1, -1):
            stride[k] = stride[k + 1] * size[k + 1]
        geo = j["dimension"]["geo"]["category"]["index"]
        tim = j["dimension"]["time"]["category"]["index"]
        for g, a in geo.items():
            for t, b in tim.items():
                v = j["value"].get(str(a * stride[gi] + b * stride[ti]))
                if v is not None:
                    rows.append((ds, g, t, v))
    return pd.DataFrame(rows, columns=["dataset", "geo", "time", "value"])


def ecb():
    out = []
    for name, key in ECB_MIR.items():
        url = f"https://data-api.ecb.europa.eu/service/data/MIR/{key}?format=csvdata&startPeriod=2019-01"
        d = pd.read_csv(io.StringIO(urllib.request.urlopen(url).read().decode()))
        out.append(pd.DataFrame({"series": name, "geo": d["REF_AREA"], "time": d["TIME_PERIOD"], "value": d["OBS_VALUE"]}))
    return pd.concat(out)


if __name__ == "__main__":
    INTERIM.mkdir(parents=True, exist_ok=True)
    pd.concat([extract_cre(y) for y in EBA_URLS]).to_csv(INTERIM / "eba_irb_cells.csv", index=False)
    pd.concat([extract_oth(y) for y in EBA_URLS]).to_csv(INTERIM / "eba_bank_items.csv", index=False)
    pd.concat([country_map(y) for y in EBA_URLS]).to_csv(INTERIM / "eba_country_codes.csv", index=False)
    eurostat().to_csv(RAW / "eurostat_macro_long.csv", index=False)
    ecb().to_csv(RAW / "ecb_mir_long.csv", index=False)
    print("done")
