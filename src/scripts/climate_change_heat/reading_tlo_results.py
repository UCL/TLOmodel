"""
Combine WBGT-model deficits with TLO HSI projections.

For each (indicator, ssp, tier, year), compute HSIs expected from TLO and
apply the WBGT model's per-facility deficit to estimate HSIs lost to heat.

Two sets of outputs:

1. Modelled-facility only (unchanged): TLO HSIs restricted to facilities the
   WBGT model was fit on.
       combined_facility_<ind>_<ssp>_<tier>.csv   facility-year
       combined_district_<ind>_<ssp>_<tier>.csv   district-year

2. District-scaled: per district-year, the loss rate across ALL WBGT-modelled
   facilities in the district (ratio of sums, weighted by each facility's
   no-heat DHIS2 expectation mu_b) is applied to the district's FULL TLO volume.
   Independent of TLO<->WBGT facility name matching. Years missing from a
   projection use the district's rate pooled over the period. Districts with no
   WBGT-modelled facility are omitted (→ grey on maps).
       tlo_district_totals_<ind>.csv                     district-year, unrestricted TLO
       combined_district_scaled_<ind>_<ssp>_<tier>.csv   district-year
"""

import argparse
import difflib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tlo import Date
from tlo.analysis.utils import extract_results

# ---- Config ----
MIN_YEAR = 2025
MAX_YEAR = 2040   # figures report 2025–2040; also avoids median-tier gap in 2041
LAG_SUFFIX = "_with_lags"

# TLO TREATMENT_IDs per DHIS2 indicator.
# Entries ending in "*" are prefixes; anything else must match the TREATMENT_ID exactly.
# CONFIRM exact IDs against your TLO version (grep TREATMENT_ID in the module files).
WBGT_TO_TLO = {
    "anc_total_visits":         ("AntenatalCare_Outpatient",),    # 8 routine ANC contacts
    "pnc_within_2wks":          ("PostnatalCare_Maternal",),      # HSI_Labour_ReceivesPostnatalCheck (mothers only)
    "fp_total_clients":         ("Contraception_*",),             # absolute counts inflated by blank-footprint resupply
    "opd_attendance":           ("FirstAttendance_NonEmergency*",),
    "cervical_screening_total": ("CervicalCancer_Screening*",),
}


def matches(tid, patterns):
    """True if tid equals a pattern, or starts with a pattern ending in '*'."""
    return any(tid.startswith(p[:-1]) if p.endswith("*") else tid == p for p in patterns)

SSP_SCENARIOS = ["ssp126", "ssp245", "ssp585"]
WBGT_MODELS = ["lowest", "median", "highest"]
WBGT_VAR = "wbgt5x_day" #"wbgt5x_day"

TLO_DRAW = 0
SCALING_FACTOR = 145.39   # simulated → national population, for these runs
DROP_BLANK_FOOTPRINT = False  # True: keep only non-blank-footprint HSIs (pending discussion)
TLO_RESULTS_FOLDER = Path("/Users/rachelmurray-watson/PycharmProjects/TLOmodel/"
                          "outputs/rm916@ic.ac.uk/"
                          "baseline_run_with_pop_new_worst_case-2026-05-21T110005Z")

HEAT_OUT_DIR = Path("/Users/rachelmurray-watson/Documents/Heat_data/Model_outputs")
OUT_DIR = HEAT_OUT_DIR / "combined_wbgt_tlo"
OUT_DIR.mkdir(exist_ok=True)

FACILITY_INFO = Path("/Users/rachelmurray-watson/PycharmProjects/TLOmodel/"
                     "resources/climate_change_impacts/facilities_with_lat_long_region.csv")

DISTRICT_NORMALISATIONS = {
    "Blanytyre":    "Blantyre",
    "Nkhatabay":    "Nkhata Bay",
    "Mzimba North": "Mzimba",
    "Mzimba South": "Mzimba",
}

FACILITY_DISTRICT_OVERRIDES = {
    "Chilumba Rural Hospital": "Karonga",
}

ABBREVIATIONS = {
    r"\bhc\b": "health centre",
    r"\bh/c\b": "health centre",
    r"\bdist\b": "district",
    r"\bhosp\b": "hospital",
}

def clean_name(name):
    """Same normaliser as the WBGT panel script."""
    name = str(name).lower().strip()
    name = re.sub(r"\s*\([^)]*\)", "", name)   # drop "(...)" suffixes
    name = re.sub(r"[.,]", "", name)
    name = re.sub(r"[+&]", " and ", name)      # NEW: A+A -> A and A
    name = re.sub(r"\s+", " ", name)
    for pattern, expansion in ABBREVIATIONS.items():
        name = re.sub(pattern, expansion, name)
    return name.strip()

def load_facility_to_district():
    """Facility name → district via the WBGT registry (same source the
    WBGT panels use, so districts line up)."""
    fl = pd.read_csv(FACILITY_INFO, low_memory=False)
    s = fl.drop_duplicates("Fname").set_index("Fname")["Dist"]
    s = s.replace(DISTRICT_NORMALISATIONS)
    # Facilities missing from the registry
    s = pd.concat([s, pd.Series(FACILITY_DISTRICT_OVERRIDES)])
    return s


# ---------------------------------------------------------------------------
# TLO HSI extraction: facility × year, filtered to prefixes
# ---------------------------------------------------------------------------
def make_tlo_series_builder(target_period, prefixes):
    def _series(_df):
        sentinel = pd.Series(
            0,
            index=pd.MultiIndex.from_tuples([("__sentinel__", 0)],
                                            names=["facility", "year"]),
            dtype=float,
        )
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
                if matches(tid, prefixes):
                    rows.append((fac, yr, n))
        if not rows:
            return sentinel
        out = pd.DataFrame(rows, columns=["facility", "year", "n"])
        return out.groupby(["facility", "year"])["n"].sum()
    return _series


def load_tlo_hsi_by_facility_year(prefixes, target_period, draw):
    raw = extract_results(
        TLO_RESULTS_FOLDER,
        module="tlo.methods.healthsystem.summary",
        key="hsi_event_counts_by_facility_monthly",
        custom_generate_series=make_tlo_series_builder(target_period, prefixes),
        do_scaling=False,
    )
    s = raw[draw].mean(axis=1) * SCALING_FACTOR  # mean across runs, scaled to national population
    df = s.rename("HSIs_expected").reset_index()
    df = df[df["facility"] != "__sentinel__"]
    df = df[~df["facility"].astype(str).isin(["nan", "NaN", ""])]
    return df


def _national_counts_by_year(key, prefixes, target_period, draw):
    """National HSI counts per year for treatment IDs matching prefixes, from a summary log key
    whose TREATMENT_ID column holds {treatment_id: count} dicts."""
    def _series(_df):
        _df = _df.copy()
        _df["date"] = pd.to_datetime(_df["date"], errors="coerce")
        _df = _df.loc[_df["date"].between(*target_period)]
        rows = [(d.year, sum(v for k, v in tids.items() if matches(k, prefixes)))
                for d, tids in zip(_df["date"], _df["TREATMENT_ID"])]
        if not rows:
            return pd.Series({0: 0.0})
        return pd.DataFrame(rows, columns=["year", "n"]).groupby("year")["n"].sum()

    raw = extract_results(
        TLO_RESULTS_FOLDER,
        module="tlo.methods.healthsystem.summary",
        key=key,
        custom_generate_series=_series,
        do_scaling=False,
    )
    return raw[draw].mean(axis=1)


def non_blank_fraction(prefixes, target_period, draw):
    """Share of matching HSIs with a non-blank appointment footprint, by year (national).

    Blank-footprint HSIs (e.g. contraceptive resupply) involve no facility visit, so they
    should not count towards DHIS2-comparable service volume. The facility-level log has no
    footprint information, so the national share is applied uniformly across facilities.
    """
    all_hsi = _national_counts_by_year("HSI_Event", prefixes, target_period, draw)
    non_blank = _national_counts_by_year("HSI_Event_non_blank_appt_footprint",
                                         prefixes, target_period, draw)
    frac = (non_blank / all_hsi).rename("nb_frac")
    frac.index.name = "year"
    return frac.reset_index()


# ---------------------------------------------------------------------------
# District scaling
# ---------------------------------------------------------------------------
def wbgt_district_rates(fac_yr, fac_to_dist):
    """District loss rates from ALL WBGT-modelled facilities (independent of TLO matching).

    Ratio of sums weighted by each facility's no-heat DHIS2 expectation (mu_b):
        rate = sum(mu_b - mu_a) / sum(mu_b)
    Loss-only clips at facility-year before summing (same as the facility-level outputs).

    fac_yr: WBGT facility-year (facility, year, mu_a, mu_b)
    """
    d = fac_yr.copy()
    d["District"] = d["facility"].map(fac_to_dist)
    unmapped = sorted(d.loc[d["District"].isna(), "facility"].unique())
    d = d.dropna(subset=["District"])
    d = d[d["mu_b"] > 0]
    d["lost_net"] = d["mu_b"] - d["mu_a"]
    d["lost_only"] = d["lost_net"].clip(lower=0)

    # District-year rate
    dy = (d.groupby(["District", "year"])
           .agg(exp=("mu_b", "sum"),
                net=("lost_net", "sum"),
                only=("lost_only", "sum"),
                n_fac=("facility", "nunique"))
           .reset_index())
    dy["rate_net"] = dy["net"] / dy["exp"]
    dy["rate_only"] = dy["only"] / dy["exp"]

    # District rate pooled over the period (fallback for years missing from the projection)
    dp = (d.groupby("District")
           .agg(exp=("mu_b", "sum"),
                net=("lost_net", "sum"),
                only=("lost_only", "sum"),
                n_fac=("facility", "nunique"))
           .reset_index())
    dp["rate_net_pooled"] = dp["net"] / dp["exp"]
    dp["rate_only_pooled"] = dp["only"] / dp["exp"]
    return dy, dp, unmapped


def scale_to_district(dy, dp, tlo_dist_tot):
    """Apply WBGT district loss rates to full district TLO volume.

    dy, dp:        district-year and pooled rates from wbgt_district_rates()
    tlo_dist_tot:  district-year, unrestricted TLO (District, year, HSIs_all)
    """

    out = (tlo_dist_tot
           .merge(dy[["District", "year", "rate_net", "rate_only"]],
                  on=["District", "year"], how="left")
           .merge(dp[["District", "rate_net_pooled", "rate_only_pooled"]],
                  on="District", how="left"))

    out["rate_source"] = np.where(out["rate_only"].notna(), "district-year",
                         np.where(out["rate_only_pooled"].notna(), "district-pooled", "none"))
    out["rate_net"] = out["rate_net"].fillna(out["rate_net_pooled"])
    out["rate_only"] = out["rate_only"].fillna(out["rate_only_pooled"])

    out["HSIs_expected"] = out["HSIs_all"]
    out["HSIs_lost_net"] = out["HSIs_all"] * out["rate_net"]
    out["HSIs_lost_only"] = out["HSIs_all"] * out["rate_only"]
    out["deficit_pct_net"] = out["rate_net"] * 100.0
    out["deficit_pct_only"] = out["rate_only"] * 100.0
    out["people_impacted_net"] = out["HSIs_lost_net"]
    out["people_impacted_only"] = out["HSIs_lost_only"]

    return out[["District", "year", "HSIs_expected", "HSIs_lost_net", "HSIs_lost_only",
                "deficit_pct_net", "deficit_pct_only",
                "people_impacted_net", "people_impacted_only", "rate_source"]]


# ---------------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------------
def combine_for_indicator(wbgt_indicator, tlo_prefixes, target_period, fac_to_dist):
    print(f"\n=== {wbgt_indicator} ← TLO prefixes {tlo_prefixes} ===")

    tlo = load_tlo_hsi_by_facility_year(tlo_prefixes, target_period, TLO_DRAW)

    if tlo.empty:
        print(f"  [skip] no TLO HSIs match {tlo_prefixes}")
        return
    # Optionally drop blank-footprint HSIs (no facility visit) using the national non-blank share by year
    if DROP_BLANK_FOOTPRINT:
        nb = non_blank_fraction(tlo_prefixes, target_period, TLO_DRAW)
        print("  non-blank share by year: "
              + ", ".join(f"{int(y)}={f:.2f}" for y, f in zip(nb["year"], nb["nb_frac"])))
        raw_total = tlo["HSIs_expected"].sum()
        tlo = tlo.merge(nb, on="year", how="left")
        n_no_frac = tlo["nb_frac"].isna().sum()
        if n_no_frac:
            print(f"  [warn] {n_no_frac} facility-years with no non-blank share (year missing from "
                  f"summary log) — left unadjusted")
        tlo["HSIs_expected"] = tlo["HSIs_expected"] * tlo["nb_frac"].fillna(1.0)
        tlo = tlo.drop(columns="nb_frac")
        print(f"  TLO (scaled): {raw_total:,.0f} HSIs → {tlo['HSIs_expected'].sum():,.0f} non-blank "
              f"({100*tlo['HSIs_expected'].sum()/raw_total:.1f}%)")

    tlo["District"] = tlo["facility"].map(fac_to_dist)
    tlo_total_all = tlo["HSIs_expected"].sum()
    n_missing_dist = tlo["District"].isna().sum()
    if n_missing_dist:
        hsi_missing_dist = tlo.loc[tlo["District"].isna(), "HSIs_expected"].sum()
        print(f"  [warn] {n_missing_dist}/{len(tlo)} TLO facility-years unmatched to district "
              f"({hsi_missing_dist:,.0f} HSIs = {100*hsi_missing_dist/tlo_total_all:.1f}% of TLO volume; "
              f"excluded from district totals)")
    print(f"  TLO: {len(tlo):,} facility-year rows, total HSIs={tlo_total_all:,.0f}")

    # Unrestricted TLO district-year totals (denominator for scaling) — before any WBGT restriction
    tlo_dist_tot = (tlo.dropna(subset=["District"])
                       .groupby(["District", "year"], as_index=False)["HSIs_expected"].sum()
                       .rename(columns={"HSIs_expected": "HSIs_all"}))
    out_tot = OUT_DIR / f"tlo_district_totals_{wbgt_indicator}.csv"
    tlo_dist_tot.to_csv(out_tot, index=False)
    print(f"  wrote {out_tot.name} ({tlo_dist_tot['District'].nunique()} districts)")

    for ssp in SSP_SCENARIOS:
        for tier in WBGT_MODELS:
            fac_path = HEAT_OUT_DIR / f"projection_facility_{wbgt_indicator}_{ssp}_{tier}_{WBGT_VAR}{LAG_SUFFIX}.csv"
            if not fac_path.exists():
                print(f"  [skip {ssp}/{tier}] no {fac_path.name}")
                continue

            fac = pd.read_csv(fac_path)
            fac["_clean"] = fac["facility"].map(clean_name)
            wbgt_clean_to_name = dict(zip(fac["_clean"], fac["facility"]))
            wbgt_clean_names = list(wbgt_clean_to_name.keys())

            # Map TLO facilities -> WBGT facilities via clean-name exact, then fuzzy
            def map_tlo_to_wbgt(tlo_name):
                c = clean_name(tlo_name)
                if c in wbgt_clean_to_name:
                    return wbgt_clean_to_name[c], "exact"
                close = difflib.get_close_matches(c, wbgt_clean_names, n=1, cutoff=0.99)
                if close:
                    return wbgt_clean_to_name[close[0]], "fuzzy"
                return None, "failed"

            tlo_matched = tlo.copy()
            mapped = tlo_matched["facility"].apply(map_tlo_to_wbgt)
            tlo_matched["facility_wbgt"] = mapped.map(lambda x: x[0])
            tlo_matched["_match_type"] = mapped.map(lambda x: x[1])

            n_exact = (tlo_matched["_match_type"] == "exact").sum()
            n_fuzzy = (tlo_matched["_match_type"] == "fuzzy").sum()
            n_failed = (tlo_matched["_match_type"] == "failed").sum()
            print(f"  [{ssp}/{tier}] TLO->WBGT match: exact={n_exact}, fuzzy={n_fuzzy}, failed={n_failed}")

            # Log a sample of fuzzy matches so you can eyeball them
            fuzzy_sample = (
                tlo_matched.loc[tlo_matched["_match_type"] == "fuzzy", ["facility", "facility_wbgt"]]
                .drop_duplicates()
                .head(20)
            )
            if not fuzzy_sample.empty:
                print("  fuzzy matches (sample):")
                for _, r in fuzzy_sample.iterrows():
                    print(f"    '{r['facility']}' -> '{r['facility_wbgt']}'")

            # HONESTY CUT: drop failed matches
            tlo_r = tlo_matched.dropna(subset=["facility_wbgt"]).copy()
            tlo_r["facility"] = tlo_r["facility_wbgt"]  # remap to WBGT names
            tlo_r = tlo_r.drop(columns=["facility_wbgt", "_match_type"])
            n_dropped_fac = tlo["facility"].nunique() - tlo_r["facility"].nunique()
            hsi_dropped = tlo_total_all - tlo_r["HSIs_expected"].sum()
            print(f"  [{ssp}/{tier}] restricting to {tlo_r['facility'].nunique()} modelled facilities "
                  f"(dropped {n_dropped_fac} facilities, {hsi_dropped:,.0f} HSIs = "
                  f"{100*hsi_dropped/tlo_total_all:.1f}% of TLO volume)")
            if tlo_r.empty:
                print(f"  [{ssp}/{tier}] no overlap — skipping")
                continue

            # Aggregate WBGT facility-month → facility-year (volume-weighted deficit)
            fac_yr = (fac.groupby(["facility", "year"])
                         .agg(mu_a=("mu_a", "sum"), mu_b=("mu_b", "sum"))
                         .reset_index())
            fac_yr["deficit_pct"] = (fac_yr["mu_b"] - fac_yr["mu_a"]) / fac_yr["mu_b"] * 100.0

            merged = tlo_r.merge(
                fac_yr[["facility", "year", "deficit_pct"]],
                on=["facility", "year"], how="left",
            )
            n_still_missing = merged["deficit_pct"].isna().sum()
            if n_still_missing:
                print(f"  [{ssp}/{tier}] {n_still_missing} facility-years still missing deficit "
                      f"(WBGT projection didn't cover that year for that facility)")

            merged["deficit_pct_loss"]     = merged["deficit_pct"].clip(lower=0)
            merged["HSIs_lost_net"]        = merged["HSIs_expected"] * merged["deficit_pct"]      / 100.0
            merged["HSIs_lost_only"]       = merged["HSIs_expected"] * merged["deficit_pct_loss"] / 100.0
            merged["people_impacted_net"]  = merged["HSIs_lost_net"]
            merged["people_impacted_only"] = merged["HSIs_lost_only"]

            out_fac = OUT_DIR / f"combined_facility_{wbgt_indicator}_{ssp}_{tier}.csv"
            merged.to_csv(out_fac, index=False)

            # District-year: sum HSIs and lost-HSIs, then recompute %
            dist = (merged.dropna(subset=["District"])
                          .groupby(["District", "year"])
                          .agg(HSIs_expected =("HSIs_expected",  "sum"),
                               HSIs_lost_net =("HSIs_lost_net",  "sum"),
                               HSIs_lost_only=("HSIs_lost_only", "sum"))
                          .reset_index())
            dist["deficit_pct_net"]      = dist["HSIs_lost_net"]  / dist["HSIs_expected"] * 100.0
            dist["deficit_pct_only"]     = dist["HSIs_lost_only"] / dist["HSIs_expected"] * 100.0
            dist["people_impacted_net"]  = dist["HSIs_lost_net"]
            dist["people_impacted_only"] = dist["HSIs_lost_only"]

            out_dist = OUT_DIR / f"combined_district_{wbgt_indicator}_{ssp}_{tier}.csv"
            dist.to_csv(out_dist, index=False)

            tot_hsi  = merged["HSIs_expected"].sum()
            tot_net  = merged["HSIs_lost_net"].sum(skipna=True)
            tot_only = merged["HSIs_lost_only"].sum(skipna=True)
            print(f"  [{ssp}/{tier}] wrote {out_fac.name}, {out_dist.name} — "
                  f"HSIs={tot_hsi:,.0f}, lost_net={tot_net:,.0f}, lost_only={tot_only:,.0f} "
                  f"({tot_only/tot_hsi*100:.2f}% loss-only)")

            # ---- District-scaled: WBGT district rate (all modelled facilities) × full district TLO volume ----
            dy, dp, unmapped = wbgt_district_rates(fac_yr, fac_to_dist)
            if unmapped:
                print(f"  [{ssp}/{tier}] [warn] {len(unmapped)} WBGT facilities not in registry → "
                      f"excluded from district rates: {unmapped[:10]}{' …' if len(unmapped) > 10 else ''}")
            n_fac_used = dp["n_fac"].sum()
            print(f"  [{ssp}/{tier}] district rates from {n_fac_used} WBGT facilities "
                  f"across {len(dp)} districts")
            scaled = scale_to_district(dy, dp, tlo_dist_tot)
            src_vol = scaled.groupby("rate_source")["HSIs_expected"].sum()
            src_tot = src_vol.sum()
            src_str = ", ".join(f"{k}={100*v/src_tot:.1f}%" for k, v in src_vol.items())
            no_rate = scaled.loc[scaled["rate_source"] == "none", "District"].unique()

            scaled = scaled[scaled["rate_source"] != "none"]  # no modelled facility → grey on maps
            out_scaled = OUT_DIR / f"combined_district_scaled_{wbgt_indicator}_{ssp}_{tier}.csv"
            scaled.to_csv(out_scaled, index=False)

            s_hsi  = scaled["HSIs_expected"].sum()
            s_only = scaled["HSIs_lost_only"].sum()
            print(f"  [{ssp}/{tier}] wrote {out_scaled.name} — HSIs={s_hsi:,.0f}, "
                  f"lost_only={s_only:,.0f} ({s_only/s_hsi*100:.2f}% loss-only); "
                  f"rate source by volume: {src_str}")
            if len(no_rate):
                print(f"  [{ssp}/{tier}] {len(no_rate)} districts with no modelled facility "
                      f"(omitted): {', '.join(sorted(no_rate))}")


def main():
    target_period = (Date(MIN_YEAR, 1, 1), Date(MAX_YEAR, 12, 31))
    fac_to_dist = load_facility_to_district()
    for wbgt_ind, tlo_prefixes in WBGT_TO_TLO.items():
        combine_for_indicator(wbgt_ind, tlo_prefixes, target_period, fac_to_dist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()  # no args currently — paths hard-coded
    main()
