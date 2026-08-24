"""
Combine WBGT-model deficits with TLO HSI projections.

For each (indicator, ssp, tier, year), compute:
  - HSIs_expected      : TLO count from hsi_event_counts_by_facility_monthly
  - deficit_prop        : from WBGT model (mu_b - mu_a)/mu_b
  - HSIs_lost          : HSIs_expected * deficit_pct / 100

Two outputs per (indicator, ssp, tier): facility-year, district-year.
"""

import argparse
from pathlib import Path
import pandas as pd
from tlo import Date
from tlo.analysis.utils import extract_results

# ---- Config ----
MIN_YEAR = 2025
MAX_YEAR = 2041
PREFIX_ON_FILENAME = "1"

# WBGT indicator → TLO TREATMENT_ID prefixes
WBGT_TO_TLO = {
    "vmmc_first_visits":       ("Hiv_Prevention_Circumcision",),
    "anc_total_visits":        ("AntenatalCare_",),
    "bcg_under1":              ("Epi_Childhood_Bcg",),
    "measles1_under1":         ("Epi_Childhood_MeaslesRubella",),
    "penta3_under1":           ("Epi_Childhood_DtpHibHep",),
    "fully_immunised_under1":  ("Epi_Childhood_",),   # union — flag below
    "pnc_within_2wks":         ("PostnatalCare_",),
    "pnc_mother_checked_48h":  ("PostnatalCare_",),
    "fp_total_clients":        ("Contraception_",),
    "opd_attendance":          ("FirstAttendance_NonEmergency",),
    "ipd_total_admissions":    ("Inpatient_Care",),
}

SSP_SCENARIOS = ["ssp126", "ssp245", "ssp585"]
WBGT_MODELS = ["lowest", "median", "highest"]
WBGT_VAR = "wbgt_day"

TLO_DRAW = 0
TLO_RESULTS_FOLDER = Path("/Users/rachelmurray-watson/PycharmProjects/TLOmodel/"
                          "outputs/rm916@ic.ac.uk/"
                          "baseline_run_with_pop_new_worst_case-2026-05-21T110005Z")

HEAT_OUT_DIR = Path("/Users/rachelmurray-watson/Documents/Heat_data/Model_outputs")
OUT_DIR = HEAT_OUT_DIR / "combined_wbgt_tlo"
OUT_DIR.mkdir(exist_ok=True)

FACILITY_LIST = Path("/Users/rachelmurray-watson/PycharmProjects/TLOmodel/"
                     "resources/healthsystem/organisation/ResourceFile_Master_Facilities_List.csv")


# ---------------------------------------------------------------------------
# TLO HSI extraction: facility × year, filtered to prefixes
# ---------------------------------------------------------------------------
def make_tlo_series_builder(target_period, prefixes):
    def _series(_df):
        sentinel = pd.Series(0, index=pd.MultiIndex.from_tuples([("__sentinel__", 0)],
                                                                 names=["facility", "year"]),
                             dtype=float)
        if _df is None or _df.empty:
            return sentinel
        _df = _df.copy()
        _df["date"] = pd.to_datetime(_df["date"], errors="coerce")
        _df = _df.loc[_df["date"].between(*target_period)]
        if _df.empty:
            return sentinel
        rows = []
        for date, counts in zip(_df["date"], _df["counts"]):
            yr = date.year
            for key, n in counts.items():
                fac, _, tid = key.partition(":")
                if tid.startswith(prefixes):
                    rows.append((fac, yr, n))
        if not rows:
            return sentinel
        out = pd.DataFrame(rows, columns=["facility", "year", "n"])
        return out.groupby(["facility", "year"])["n"].sum()
    return _series


def load_tlo_hsi_by_facility_year(prefixes, target_period, draw):
    """Return DataFrame with columns [facility, year, HSIs_expected]."""
    raw = extract_results(
        TLO_RESULTS_FOLDER,
        module="tlo.methods.healthsystem.summary",
        key="hsi_event_counts_by_facility_monthly",
        custom_generate_series=make_tlo_series_builder(target_period, prefixes),
        do_scaling=False,
    )
    # raw index = (facility, year); columns = (draw, run)
    s = raw[draw].mean(axis=1)  # mean across runs
    df = s.rename("HSIs_expected").reset_index()
    df = df[df["facility"] != "__sentinel__"]
    df = df[~df["facility"].astype(str).isin(["nan", "NaN", ""])]
    return df


def load_facility_to_district():
    fl = pd.read_csv(FACILITY_LIST)
    name_col = "Facility_Name" if "Facility_Name" in fl.columns else "facility_name"
    dist_col = "District" if "District" in fl.columns else "district"
    return fl.set_index(name_col)[dist_col]


# ---------------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------------
def combine_for_indicator(wbgt_indicator, tlo_prefixes, target_period, fac_to_dist):
    print(f"\n=== {wbgt_indicator} ← TLO prefixes {tlo_prefixes} ===")

    # 1) Pull TLO HSI counts once (doesn't depend on ssp/tier)
    tlo = load_tlo_hsi_by_facility_year(tlo_prefixes, target_period, TLO_DRAW)
    if tlo.empty:
        print(f"  [skip] no TLO HSIs match {tlo_prefixes}")
        return
    tlo["District"] = tlo["facility"].map(fac_to_dist)
    n_missing_dist = tlo["District"].isna().sum()
    if n_missing_dist:
        print(f"  [warn] {n_missing_dist} facility rows unmatched to district — dropped for district agg only")
    print(f"  TLO: {len(tlo):,} facility-year rows, total HSIs={tlo['HSIs_expected'].sum():,.0f}")

    for ssp in SSP_SCENARIOS:
        for tier in WBGT_MODELS:
            # 2) WBGT facility-level projections (facility-month)
            fac_path = HEAT_OUT_DIR / f"projection_facility_{wbgt_indicator}_{ssp}_{tier}_{WBGT_VAR}.csv"
            if not fac_path.exists():
                print(f"  [skip {ssp}/{tier}] no {fac_path.name}")
                continue

            fac = pd.read_csv(fac_path)
            # Aggregate WBGT facility-month → facility-year
            fac_yr = (fac.groupby(["facility", "year"])
                         .agg(mu_a=("mu_a", "sum"), mu_b=("mu_b", "sum"))
                         .reset_index())
            fac_yr["deficit_pct"] = (fac_yr["mu_b"] - fac_yr["mu_a"]) / fac_yr["mu_b"] * 100.0

            # 3) Join to TLO by facility × year
            merged = tlo.merge(fac_yr[["facility", "year", "deficit_pct"]],
                               on=["facility", "year"], how="left")
            n_unmatched = merged["deficit_pct"].isna().sum()
            if n_unmatched:
                print(f"  [{ssp}/{tier}] {n_unmatched}/{len(merged)} facility-year rows have no WBGT deficit")

            merged["HSIs_lost"] = merged["HSIs_expected"] * merged["deficit_pct"] / 100.0
            merged["people_impacted"] = merged["HSIs_lost"]  # 1:1 mapping

            out_fac = OUT_DIR / f"{PREFIX_ON_FILENAME}_combined_facility_{wbgt_indicator}_{ssp}_{tier}.csv"
            merged.to_csv(out_fac, index=False)

            # 4) District-year aggregation
            # Note: we sum HSIs and lost-HSIs at the district level, then recompute deficit_pct
            #       from the aggregates rather than averaging facility-level %s.
            dist = (merged.dropna(subset=["District"])
                          .groupby(["District", "year"])
                          .agg(HSIs_expected=("HSIs_expected", "sum"),
                               HSIs_lost=("HSIs_lost", "sum"))
                          .reset_index())
            dist["deficit_pct"] = dist["HSIs_lost"] / dist["HSIs_expected"] * 100.0
            dist["people_impacted"] = dist["HSIs_lost"]

            out_dist = OUT_DIR / f"{PREFIX_ON_FILENAME}_combined_district_{wbgt_indicator}_{ssp}_{tier}.csv"
            dist.to_csv(out_dist, index=False)

            tot_hsi = merged["HSIs_expected"].sum()
            tot_lost = merged["HSIs_lost"].sum(skipna=True)
            print(f"  [{ssp}/{tier}] wrote {out_fac.name}, {out_dist.name} — "
                  f"HSIs={tot_hsi:,.0f}, lost={tot_lost:,.0f} ({tot_lost/tot_hsi*100:.2f}%)")


def main():
    target_period = (Date(MIN_YEAR, 1, 1), Date(MAX_YEAR, 12, 31))
    fac_to_dist = load_facility_to_district()

    for wbgt_ind, tlo_prefixes in WBGT_TO_TLO.items():
        combine_for_indicator(wbgt_ind, tlo_prefixes, target_period, fac_to_dist)


if __name__ == "__main__":
    main()
