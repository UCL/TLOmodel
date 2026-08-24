"""
Combine WBGT-model deficits with TLO HSI projections.

For each (indicator, ssp, tier, year), compute:
  - HSIs_expected      : TLO count from hsi_event_counts_by_facility_monthly
  - deficit_prop        : from WBGT model (mu_b - mu_a)/mu_b
  - HSIs_lost          : HSIs_expected * deficit_prop / 100

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


FACILITY_INFO = Path("/Users/rachelmurray-watson/PycharmProjects/TLOmodel/"
                     "resources/climate_change_impacts/facilities_with_lat_long_region.csv")

DISTRICT_NORMALISATIONS = {
    "Blanytyre":    "Blantyre",
    "Nkhatabay":    "Nkhata Bay",
    "Mzimba North": "Mzimba",
    "Mzimba South": "Mzimba",
}


def load_facility_to_district():
    """Map facility name → district using the WBGT registry (same source
    the WBGT panels use, so districts line up across the pipeline)."""
    fl = pd.read_csv(FACILITY_INFO, low_memory=False)
    s = fl.drop_duplicates("Fname").set_index("Fname")["Dist"]
    s = s.replace(DISTRICT_NORMALISATIONS)
    return s

# ---------------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------------
def combine_for_indicator(wbgt_indicator, tlo_prefixes, target_period, fac_to_dist):
    print(f"\n=== {wbgt_indicator} ← TLO prefixes {tlo_prefixes} ===")

    # 1) Pull TLO HSI counts once
    tlo = load_tlo_hsi_by_facility_year(tlo_prefixes, target_period, TLO_DRAW)
    if tlo.empty:
        print(f"  [skip] no TLO HSIs match {tlo_prefixes}")
        return
    tlo["District"] = tlo["facility"].map(fac_to_dist)
    tlo_total_all = tlo["HSIs_expected"].sum()

    for ssp in SSP_SCENARIOS:
        for tier in WBGT_MODELS:
            fac_path = HEAT_OUT_DIR / f"projection_facility_{wbgt_indicator}_{ssp}_{tier}_{WBGT_VAR}.csv"
            if not fac_path.exists():
                print(f"  [skip {ssp}/{tier}] no {fac_path.name}")
                continue

            fac = pd.read_csv(fac_path)
            modelled_facs = set(fac["facility"].unique())

            # ------- restrict TLO to modelled facilities -------
            tlo_r = tlo[tlo["facility"].isin(modelled_facs)].copy()
            n_dropped_fac = tlo["facility"].nunique() - tlo_r["facility"].nunique()
            hsi_dropped = tlo_total_all - tlo_r["HSIs_expected"].sum()
            print(f"  [{ssp}/{tier}] restricting to {tlo_r['facility'].nunique()} modelled facilities "
                  f"(dropped {n_dropped_fac} facilities, {hsi_dropped:,.0f} HSIs = "
                  f"{100*hsi_dropped/tlo_total_all:.1f}% of TLO volume)")
            if tlo_r.empty:
                print(f"  [{ssp}/{tier}] no overlap — skipping")
                continue

            # Aggregate WBGT facility-month → facility-year
            fac_yr = (fac.groupby(["facility", "year"])
                         .agg(mu_a=("mu_a", "sum"), mu_b=("mu_b", "sum"))
                         .reset_index())
            fac_yr["deficit_pct"] = (fac_yr["mu_b"] - fac_yr["mu_a"]) / fac_yr["mu_b"] * 100.0
            fac_yr["deficit_pct"]
            merged = tlo_r.merge(
                fac_yr[["facility", "year", "deficit_pct"]],
                on=["facility", "year"], how="left",
            )
            n_still_missing = merged["deficit_pct"].isna().sum()
            if n_still_missing:
                print(f"  [{ssp}/{tier}] {n_still_missing} facility-years still missing deficit "
                      f"(WBGT projection didn't cover that year for that facility)")

            merged["deficit_pct_loss"] = merged["deficit_pct"].clip(lower=0)
            merged["HSIs_lost_net"]    = merged["HSIs_expected"] * merged["deficit_pct"]      / 100.0
            merged["HSIs_lost_only"]   = merged["HSIs_expected"] * merged["deficit_pct_loss"] / 100.0
            merged["people_impacted_net"]  = merged["HSIs_lost_net"]
            merged["people_impacted_only"] = merged["HSIs_lost_only"]

            out_fac = OUT_DIR / f"combined_facility_{wbgt_indicator}_{ssp}_{tier}.csv"
            merged.to_csv(out_fac, index=False)

            dist = (merged.dropna(subset=["District"])
                          .groupby(["District", "year"])
                          .agg(HSIs_expected =("HSIs_expected",  "sum"),
                               HSIs_lost_net =("HSIs_lost_net",  "sum"),
                               HSIs_lost_only=("HSIs_lost_only", "sum"))
                          .reset_index())
            dist["deficit_pct_net"]        = dist["HSIs_lost_net"]  / dist["HSIs_expected"] * 100.0
            dist["deficit_pct_only"]       = dist["HSIs_lost_only"] / dist["HSIs_expected"] * 100.0
            dist["people_impacted_net"]    = dist["HSIs_lost_net"]
            dist["people_impacted_only"]   = dist["HSIs_lost_only"]

            out_dist = OUT_DIR / f"combined_district_{wbgt_indicator}_{ssp}_{tier}.csv"
            dist.to_csv(out_dist, index=False)

            tot_hsi   = merged["HSIs_expected"].sum()
            tot_net   = merged["HSIs_lost_net"].sum(skipna=True)
            tot_only  = merged["HSIs_lost_only"].sum(skipna=True)
            print(f"  [{ssp}/{tier}] wrote {out_fac.name}, {out_dist.name} — "
                  f"HSIs={tot_hsi:,.0f}, lost_net={tot_net:,.0f}, lost_only={tot_only:,.0f} "
                  f"({tot_only/tot_hsi*100:.2f}% loss-only)")

def main():
    target_period = (Date(MIN_YEAR, 1, 1), Date(MAX_YEAR, 12, 31))
    fac_to_dist = load_facility_to_district()

    for wbgt_ind, tlo_prefixes in WBGT_TO_TLO.items():
        combine_for_indicator(wbgt_ind, tlo_prefixes, target_period, fac_to_dist)


if __name__ == "__main__":
    main()
