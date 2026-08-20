"""
build_anc_panel.py — reshape ANC wide CSV and join onto weather panel.
"""
import pandas as pd
import re
from pathlib import Path

RAW_PATH   = "/Users/rachelmurray-watson/Documents/Heat_data/ANC_data_2011_2024.csv"
PANEL_DIR  = "/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/Indices/"
TEMPLATE   = f"{PANEL_DIR}regression_panel_bcg_under1.csv"   # any existing panel
OUT_PATH   = f"{PANEL_DIR}regression_panel_anc_total_visits.csv"

# ---- 1. Wide → long -------------------------------------------------------
raw = pd.read_csv(RAW_PATH)
id_cols = ["organisationunitid", "organisationunitname",
           "organisationunitcode", "organisationunitdescription"]
val_cols = [c for c in raw.columns if c.startswith("HMIS Total Antenatal Visits")]

long = raw.melt(id_vars=id_cols, value_vars=val_cols,
                var_name="col", value_name="anc_total_visits")

# "HMIS Total Antenatal Visits January 2024" → date
pat = re.compile(r"HMIS Total Antenatal Visits (\w+) (\d{4})")
parts = long["col"].str.extract(pat)
long["date"] = pd.to_datetime(parts[0] + " " + parts[1], format="%B %Y")
long = long.rename(columns={"organisationunitname": "facility"})
long = long[["facility", "date", "anc_total_visits"]]

# Coerce to numeric; blanks → NaN
long["anc_total_visits"] = pd.to_numeric(long["anc_total_visits"], errors="coerce")

# OPTIONAL: treat DHIS2 sentinel "8" as suspect. Uncomment if you agree
# after inspecting — I'd look at facility-level histograms first.
# suspect = long.groupby("facility")["anc_total_visits"].apply(
#     lambda s: (s == 8).mean() > 0.5)
# long.loc[long["facility"].isin(suspect[suspect].index), "anc_total_visits"] = pd.NA

# ---- 2. Facility name normalisation --------------------------------------
def norm(s):
    return (str(s).strip().lower()
            .replace("  ", " ")
            .replace(" hc", " health centre")
            .replace(" h/c", " health centre"))
long["_key"] = long["facility"].map(norm)

# ---- 3. Merge onto template panel to inherit Dist + weather --------------
tmpl = pd.read_csv(TEMPLATE, parse_dates=["date"])
tmpl["_key"] = tmpl["facility"].map(norm)

keep_cols = ["facility", "Dist", "date", "wbgt_day", "precip_month", "_key"]
keep_cols = [c for c in keep_cols if c in tmpl.columns]
tmpl_slim = tmpl[keep_cols].drop_duplicates(["_key", "date"])

merged = tmpl_slim.merge(
    long[["_key", "date", "anc_total_visits"]],
    on=["_key", "date"], how="left",
)

# ---- 4. Diagnostics ------------------------------------------------------
matched   = merged["anc_total_visits"].notna().groupby(merged["facility"]).any().sum()
in_raw    = long["_key"].nunique()
in_tmpl   = tmpl["_key"].nunique()
overlap   = set(long["_key"]) & set(tmpl["_key"])
print(f"Facilities in ANC raw:      {in_raw}")
print(f"Facilities in template:     {in_tmpl}")
print(f"Name overlap:               {len(overlap)}")
print(f"Facilities w/ ≥1 ANC obs:   {matched}")
print(f"Rows w/ ANC value:          {merged['anc_total_visits'].notna().sum():,}")

merged.drop(columns="_key").to_csv(OUT_PATH, index=False)
print(f"→ {OUT_PATH}")
