"""Shared settings for the EU bank default satellite-model project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"

# EBA EU-wide Transparency Exercise full-database files (2021-2025 exercises)
EBA_URLS = {
    2021: "https://www.eba.europa.eu/assets/TE2021/Full_database/",
    2022: "https://www.eba.europa.eu/assets/TE2022/Full_database/",
    2023: "https://www.eba.europa.eu/assets/TE2023/Full_database/837203/",
    2024: "https://www.eba.europa.eu/assets/TE2024/Full_database/256109/",
    2025: "https://www.eba.europa.eu/assets/TE2025/Full_database/883401/",
}

# Item codes change each year (first two digits = exercise year); the last five digits are stable.
CRE_ITEMS = {"20502": "orig", "20505": "orig", "20512": "def", "20515": "def"}
OTH_ITEMS = {"21010": "total_assets", "20140": "cet1_ratio", "21019": "gross_loans_ac"}

# IRB exposure classes kept (EBA metadata, sheet "Exposure")
CLASSES = {303: "Corporates", 406: "Retail secured by real estate",
           409: "Retail qualifying revolving", 410: "Other retail"}
IRB_PORTFOLIO = 2

EU_EEA = ["AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE", "IT",
          "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE", "NO", "IS", "LI"]

# Eurostat series (JSON-stat API) and filters
EUROSTAT = {
    "une_rt_q": "sex=T&age=Y15-74&unit=PC_ACT&s_adj=SA",
    "namq_10_gdp": "unit=CLV_PCH_SM&s_adj=SCA&na_item=B1GQ",
    "prc_hpi_q": "purchase=TOTAL&unit=RCH_A",
    "prc_hicp_manr": "coicop=CP00&unit=RCH_A",
    "irt_lt_mcby_q": "int_rt=MCBY",
}
# ECB MFI interest rate statistics: composite cost of borrowing (euro area countries)
ECB_MIR = {"hh_house": "M..B.A2C.AM.R.A.2250.EUR.N", "nfc": "M..B.A2I.AM.R.A.2240.EUR.N"}

MIN_EXPOSURE_EUR_M = 1.0    # drop near-empty books (EUR million); EUR 10m used as a robustness check
RISE_THRESHOLD_PP = 0.10    # RQ4 target: next-quarter rise in DER above 0.10 pp
TEST_START = "2024-09-30"   # out-of-time test period: Sep 2024 - Jun 2025
SEED = 42
