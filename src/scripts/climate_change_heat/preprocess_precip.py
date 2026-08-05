"""
preprocess_precip.py

Convert the two WIDE precipitation matrices (facility columns x month rows,
integer-indexed 0..N-1) into ONE LONG file keyed on (facility, date), ready to
merge into regression_panel_{indicator}.csv by NAME + DATE (never by position).

Outputs:  precip_long.csv  with columns
          facility, date, precip_month, precip_5day

!! The one assumption you MUST verify: PRECIP_START. The matrices carry no
   dates — only row indices 0..335 (= 28 years of months). If row 0 is not the
   month below, every precip value attaches to the wrong calendar month and the
   whole control is silently wrong. The script prints a seasonality check so you
   can confirm: Malawi rainfall peaks Dec-Mar and is ~0 Jun-Sep. If the printed
   monthly means don't show that, PRECIP_START is wrong.
"""

import sys
import pandas as pd
import numpy as np

# ---- VERIFY THIS -----------------------------------------------------------
PRECIP_START = "2011-01-01"     # assumed calendar month of row 0
# ---------------------------------------------------------------------------

MONTHLY_PATH = "/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/Precip_data/historical_monthly_total_by_all_facilities.csv"
FIVEDAY_PATH = "/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/Precip_data/historical_daily_total_by_all_facilities_five_day_cumulative.csv"
OUT_PATH     = "/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/Indices/precip_long.csv"


def load_wide_to_long(path, value_name, start=PRECIP_START):
    wide = pd.read_csv(path, index_col=0)
    # attach explicit dates to the positional row index
    dates = pd.date_range(start, periods=len(wide), freq="MS")
    wide.index = dates
    wide.index.name = "date"
    # normalise facility names (strip whitespace; keep disambiguating suffixes)
    wide.columns = wide.columns.astype(str).str.strip()
    long = (wide.stack()
                .rename(value_name)
                .rename_axis(index=["date", "facility"])
                .reset_index())
    return long


def main():
    m  = load_wide_to_long(MONTHLY_PATH, "precip_month")
    d5 = load_wide_to_long(FIVEDAY_PATH, "precip_5day")

    # sanity: both files must describe the same facilities/rows
    if m.shape[0] != d5.shape[0]:
        print(f"[warn] row counts differ: monthly={m.shape[0]}, 5day={d5.shape[0]}")

    precip = m.merge(d5, on=["facility", "date"], how="outer")
    precip = precip.sort_values(["facility", "date"]).reset_index(drop=True)

    # ---- AUDIT 1: coverage ------------------------------------------------
    n_fac  = precip["facility"].nunique()
    n_mon  = precip["date"].nunique()
    span   = f"{precip['date'].min():%Y-%m} to {precip['date'].max():%Y-%m}"
    na_m   = precip["precip_month"].isna().mean()
    na_5   = precip["precip_5day"].isna().mean()
    print(f"facilities: {n_fac} | months: {n_mon} | span: {span}")
    print(f"missing: precip_month {na_m:.1%}, precip_5day {na_5:.1%}")
    print("  NOTE: high missingness here is expected — these matrices run to 2038,")
    print("  and post-data months are empty. Judge coverage only within your")
    print("  analysis window after joining, NOT on this whole-file number.")

    # ---- AUDIT 2: seasonality (does PRECIP_START make sense?) --------------
    by_month = (precip.assign(m=precip["date"].dt.month)
                      .groupby("m")["precip_month"].mean())
    print("\nMean monthly precip by calendar month "
          "(expect HIGH Dec-Mar, LOW Jun-Sep for Malawi):")
    for mm in range(1, 13):
        bar = "#" * int(by_month.get(mm, 0) / max(by_month.max(), 1) * 40)
        print(f"  {mm:2d}: {by_month.get(mm, float('nan')):7.1f}  {bar}")
    wet = by_month.reindex([12, 1, 2, 3]).mean()
    dry = by_month.reindex([6, 7, 8, 9]).mean()
    if wet > dry:
        print(f"\n  OK: wet-season mean ({wet:.0f}) > dry-season mean ({dry:.0f}) "
              "-> PRECIP_START looks right.")
    else:
        print(f"\n  !! WARNING: wet-season mean ({wet:.0f}) <= dry-season "
              f"({dry:.0f}). PRECIP_START is probably WRONG — fix it before use.")

    precip.to_csv(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH}  ({len(precip):,} rows)")
    print("Join into panels with:  panel.merge(precip, on=['facility','date'], "
          "how='left')  and assert coverage.")


if __name__ == "__main__":
    main()
