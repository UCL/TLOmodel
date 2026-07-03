"""
combine_dhis2_data.py

Combines the per-indicator, per-year CSV chunks produced by
download_dhis2_data.py into a single long-format panel, then pivots to a
facility-month-wide table ready for merging with the WBGT facility panel.

Mirrors collate_NASA_nc_files.py: reads everything from a flat input
directory, standardises columns, concatenates, writes one combined output.

Usage:
    python combine_dhis2_data.py
    python combine_dhis2_data.py --dry-run   # just lists files found, no combination
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
RAW_DIR = Path("/Users/rachelmurray-watson/Documents/Heat_data/DHIS2_Malawi/raw_pulls")
OUT_DIR = Path("/Users/rachelmurray-watson/Documents/Heat_data/DHIS2_Malawi/combined")

LONG_OUTPUT_PATH = OUT_DIR / "dhis2_panel_long.csv"
WIDE_OUTPUT_PATH = OUT_DIR / "dhis2_panel_wide_facility_month.csv"

# Filename pattern produced by download_dhis2_data.py: {label}_{year}.csv
FILENAME_PATTERN = re.compile(r"^(?P<label>.+)_(?P<year>\d{4})\.csv$")

# ---------------------------------------------------------------------------


def find_chunk_files() -> list[Path]:
    files = sorted(RAW_DIR.glob("*.csv"))
    matched = [f for f in files if FILENAME_PATTERN.match(f.name)]
    unmatched = [f for f in files if not FILENAME_PATTERN.match(f.name)]
    if unmatched:
        print(f"Warning: {len(unmatched)} files in {RAW_DIR} don't match the expected "
              f"'{{label}}_{{year}}.csv' pattern and will be skipped: "
              f"{[f.name for f in unmatched[:5]]}{'...' if len(unmatched) > 5 else ''}")
    return matched


def load_and_standardise(path: Path) -> pd.DataFrame | None:
    """
    Load one chunk and standardise column names. The analytics endpoint
    (outputIdScheme=NAME) typically returns columns like 'Data', 'Period',
    'Organisation unit', 'Value' — but exact naming can vary by DHIS2
    version, so this maps common variants defensively rather than assuming
    one fixed schema.
    """
    match = FILENAME_PATTERN.match(path.name)
    label, year = match.group("label"), match.group("year")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  FAILED to read {path.name}: {e}")
        return None

    if df.empty:
        print(f"  {path.name}: empty, skipping")
        return None

    # Defensive column-name mapping across common DHIS2 analytics variants
    rename_map = {}
    for col in df.columns:
        c = col.strip().lower()
        if c in ("organisation unit", "orgunit", "org unit", "ou"):
            rename_map[col] = "facility"
        elif c == "period":
            rename_map[col] = "period"
        elif c == "value":
            rename_map[col] = "value"
    df = df.rename(columns=rename_map)

    missing = {"facility", "period", "value"} - set(df.columns)
    if missing:
        print(f"  {path.name}: missing expected column(s) {missing} after mapping "
              f"(found columns: {list(df.columns)}) — skipping")
        return None

    df = df[["facility", "period", "value"]].copy()
    df["indicator"] = label
    df["year"] = int(year)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="List chunk files found, skip combination")
    args = parser.parse_args()

    chunk_files = find_chunk_files()
    print(f"Found {len(chunk_files)} chunk files in {RAW_DIR}")

    if args.dry_run:
        for f in chunk_files[:20]:
            print(f"  {f.name}")
        if len(chunk_files) > 20:
            print(f"  ... and {len(chunk_files) - 20} more")
        return

    if not chunk_files:
        print("No chunk files found — check RAW_DIR path.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for f in chunk_files:
        df = load_and_standardise(f)
        if df is not None:
            frames.append(df)

    if not frames:
        print("No chunks loaded successfully — nothing to combine.")
        sys.exit(1)

    long_df = pd.concat(frames, ignore_index=True)

    # DHIS2 monthly periods are typically 'YYYYMM' strings — parse to a proper date
    long_df["period_parsed"] = pd.to_datetime(long_df["period"], format="%Y%m", errors="coerce")
    n_unparsed = long_df["period_parsed"].isna().sum()
    if n_unparsed:
        print(f"Warning: {n_unparsed} rows had a period that didn't parse as YYYYMM — "
              f"check period format (examples: {long_df.loc[long_df['period_parsed'].isna(), 'period'].unique()[:5]})")

    long_df = long_df.sort_values(["indicator", "facility", "period_parsed"])
    long_df.to_csv(LONG_OUTPUT_PATH, index=False)
    print(f"\nLong-format panel: {len(long_df)} rows -> {LONG_OUTPUT_PATH}")
    print(f"  Facilities: {long_df['facility'].nunique()}")
    print(f"  Indicators: {long_df['indicator'].nunique()} ({sorted(long_df['indicator'].unique())})")
    print(f"  Period range: {long_df['period_parsed'].min()} to {long_df['period_parsed'].max()}")

    # Pivot to facility-month wide format: one row per facility-month, one column per indicator
    wide_df = long_df.pivot_table(
        index=["facility", "period_parsed"],
        columns="indicator",
        values="value",
        aggfunc="first",  # should be unique per facility-period-indicator; 'first' guards against accidental dupes
    ).reset_index()
    wide_df.to_csv(WIDE_OUTPUT_PATH, index=False)
    print(f"\nWide facility-month panel: {len(wide_df)} rows -> {WIDE_OUTPUT_PATH}")

    print("\nNext step: merge this wide panel against your facility coordinates file")
    print("(matching on facility name — check for spelling/naming mismatches first),")
    print("then feed into wbgt_facility_panel.py alongside the WBGT extreme indices.")


if __name__ == "__main__":
    main()
