"""
prep_anc_for_combine.py — split ANC wide file into per-year long CSVs that
combine_dhis2_data.py can ingest.
"""
import pandas as pd
import re
from pathlib import Path

ANC_RAW  = Path("/Users/rachelmurray-watson/Documents/Heat_data/ANC_data_2011_2024.csv")
RAW_DIR  = Path("/Users/rachelmurray-watson/Documents/Heat_data/DHIS2_Malawi/raw_pulls")
LABEL    = "anc_total_visits"

raw = pd.read_csv(ANC_RAW)
val_cols = [c for c in raw.columns if c.startswith("HMIS Total Antenatal Visits")]

long = raw.melt(
    id_vars=["organisationunitname"],
    value_vars=val_cols,
    var_name="col",
    value_name="value",
)
pat = re.compile(r"HMIS Total Antenatal Visits (\w+) (\d{4})")
parts = long["col"].str.extract(pat)
long["month"] = parts[0]
long["year"]  = parts[1].astype(int)
long["period"] = long["month"] + " " + long["year"].astype(str)  # "January 2024"
long = long.rename(columns={"organisationunitname": "facility"})
long["value"] = pd.to_numeric(long["value"], errors="coerce")
long = long.dropna(subset=["value"])

RAW_DIR.mkdir(parents=True, exist_ok=True)
for yr, grp in long.groupby("year"):
    out = RAW_DIR / f"{LABEL}_{yr}.csv"
    grp[["facility", "period", "value"]].to_csv(out, index=False)
    print(f"→ {out.name}  ({len(grp):,} rows)")
