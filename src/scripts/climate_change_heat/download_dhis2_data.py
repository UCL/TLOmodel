"""
download_dhis2_data.py

Bulk-downloads DHIS2 data valuesfor a set of
chosen indicators/data elements, across all Malawi facilities, over a date
range — chunked by year and by indicator group to avoid timeouts, and
resumable (skips chunks that have already been downloaded).

Before running this:
  1. Run fetch_dhis2_metadata.py and find the UIDs of the indicators/data
     elements you want in the resulting CSVs.
  2. Fill in INDICATOR_IDS / DATA_ELEMENT_IDS below.
  3. Decide your org unit level (facility, district, zone) via ORG_UNIT_ID
     and ORG_UNIT_LEVEL.

Usage:
    python download_dhis2_data.py
    python download_dhis2_data.py --dry-run to check login credentials
"""

import argparse
import getpass
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_URL = "https://dhis2.health.gov.mw/api"
OUT_DIR = Path("/Users/rachelmurray-watson/Documents/Heat_data/DHIS2_Malawi/raw_pulls")

# Keep them as a dict of {short_label: uid} so filenames stay readable.
# All are raw counts/totals, and hopefully more related to service/demand (but not sickness demand) side rather than caused by humidity
INDICATOR_IDS: dict[str, str] = {
    # --- reproductive / maternal health ---
    "anc1_coverage": "dSCLYaL8ouA",              # RHD N ANC 1 coverage (HMIS 15)
    "anc4_coverage": "yXtwUarjGFD",               # RHD N ANC 4th Visit Coverage
    "institutional_delivery_rate": "FKdaVOCu65k", # % of deliveries conducted in facility
    "fp_total_clients": "f3QE2jkAkAU",            # PI FP - Total number of clients who received FP (all methods, all visit types)

    # --- general service utilisation ---
    "opd_attendance": "g7azgsbbrEr",              # CLIN OPD Attendance (raw count)
    "ipd_total_admissions": "xSBdiGepfRL",         # CLIN Total Admissions from all causes

    # --- HIV service volume (partial coverage only — see note below) ---
    "vmmc_first_visits": "jblfwnIgu1i",            # HIV P VMMC Total With 1st Visits

}
DATA_ELEMENT_IDS: dict[str, str] = {
    "pnc_mother_checked_48h": "JSN2CbjKFt9",       # RHD PNC # Mother Checked in <48 Hours (first PNC contact, WHO-standard window)

    # --- ANC schedule completion: ANC4 / ANC1 -----------------------------
    # DENOMINATOR (ANC1): women who entered ANC = the "already due" cohort.
    "anc_new_attendees":            "mraYTWoViTj",  # HMIS Total # of New Antenatal Attendees

    # --- ANC timing: 1st-trimester start / new attendees ------------------
    "anc_first_trimester_starts":   "WR9HAMKhBZG",  # HMIS # of Pregnant Women Starting Antenatal Care First Trimester
    # --- Immunisation series completion / dropout -------------------------
    # DENOMINATOR (series start): Penta1 is NOT in the metadata (only Penta III,
    # Polio III, BCG, Measles-1st exist), so BCG is used as the series-start
    # proxy. Standard Penta1->Penta3 dropout is not computable from this instance.
    "bcg_under1":                   "NeEMZqYtG4c",  # HMIS # of Under 1 Children Given BCG (series-start proxy)
    # NUMERATORS (series completion, pick per specification):
    "penta3_under1":                "nMbqdoAszVh",  # HMIS # of Under 1 Children Given Pentavalent - III
    "measles1_under1":              "i7EICUDBS9M",  # HMIS # of Under 1 Children Given Measles 1st Doses at 9M
    "fully_immunised_under1":       "ueZ9XGbE7Dn",  # HMIS # of Fully Immunised under 1 children
    # (optional extra completion numerator: Polio III = D3m5WvOSixM)

    # --- Postnatal contact: PNC within 2 weeks / deliveries ----------------
    # NUMERATOR: scheduled postnatal contact.
    "pnc_within_2wks":              "lM48Ysgzz0H",  # HMIS # of Postpartum Care Within 2 Weeks of Delivery
    "pnc_first_visit_2wks":         "slV0W4q2ssz",  # HMIS Postnatal first visit within 2 weeks of delivery
    # DENOMINATOR: the delivered cohort (pick one; live births is the cleaner base).
    "live_births_total":            "ftyQGpirFHE",  # HMIS Total # of Live Births
    "skilled_deliveries":           "Dv7Hcho5dCr",  # HMIS # of Deliveries Attended by Skilled Health Personnel
}

# Root org unit to pull under (e.g. national root UID) — data will be
# returned disaggregated down to ORG_UNIT_LEVEL beneath it.
ORG_UNIT_ID = "Facility"
ORG_UNIT_LEVEL = 4  # adjust once you've confirmed Malawi's hierarchy depth (e.g. facility level)

START_YEAR = 2015
END_YEAR = 2024

REQUEST_TIMEOUT = 120  # seconds
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 5

# ---------------------------------------------------------------------------


def get_credentials():
    username = input("DHIS2 username: ").strip()
    password = getpass.getpass("DHIS2 password: ")
    return username, password


def check_uids_resolved(targets: dict[str, str]) -> None:
    """
    Fail loudly if any UID is still a REPLACE_* placeholder (or the org unit
    root is unset). Consistent with the pipeline's silent-failure policy:
    an unresolved UID should stop the run, not quietly download nothing.
    """
    unresolved = [f"{label} ({uid})" for label, uid in targets.items()
                  if "REPLACE" in uid.upper()]
    if unresolved:
        print("ERROR: the following targets still have placeholder UIDs — "
              "look them up in the fetch_dhis2_metadata.py output and fill them in:")
        for item in unresolved:
            print(f"  - {item}")
        sys.exit(1)
    if "REPLACE" in ORG_UNIT_ID.upper():
        print("ERROR: ORG_UNIT_ID is still a placeholder — set the national root UID "
              "(and confirm ORG_UNIT_LEVEL) before running.")
        sys.exit(1)


def fetch_year_chunk(session: requests.Session, dx_uid: str, year: int) -> pd.DataFrame | None:
    """
    Pull one indicator/data element for one calendar year, aggregated to
    ORG_UNIT_LEVEL beneath ORG_UNIT_ID, using the analytics endpoint.
    Retries on transient failures (timeouts, 5xx) before giving up.

    IMPORTANT: pe must list the 12 monthly period IDs (e.g. "201501") rather
    than the bare year ("2015") — DHIS2 treats a bare 4-digit value as the
    ANNUAL period, which silently returns one yearly total per facility
    instead of 12 monthly rows.
    """
    monthly_periods = ";".join(f"{year}{month:02d}" for month in range(1, 13))

    params = {
        "dimension": [f"dx:{dx_uid}", f"pe:{monthly_periods}", f"ou:LEVEL-{ORG_UNIT_LEVEL};{ORG_UNIT_ID}"],
        "displayProperty": "NAME",
        "outputIdScheme": "NAME",
    }

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = session.get(f"{BASE_URL}/analytics.json", params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                headers = [h["name"] for h in data.get("headers", [])]
                rows = data.get("rows", [])
                if not rows:
                    return pd.DataFrame(columns=headers)
                return pd.DataFrame(rows, columns=headers)
            else:
                print(f"    attempt {attempt}: HTTP {resp.status_code} — {resp.text[:200]}")
        except requests.exceptions.RequestException as e:
            print(f"    attempt {attempt}: request error — {e}")

        if attempt < RETRY_ATTEMPTS:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    print(f"    FAILED after {RETRY_ATTEMPTS} attempts — skipping {dx_uid}, {year}")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="List planned chunks without downloading")
    args = parser.parse_args()

    all_targets = {**INDICATOR_IDS, **DATA_ELEMENT_IDS}
    if not all_targets:
        print("No INDICATOR_IDS or DATA_ELEMENT_IDS configured yet.")
        print("Run fetch_dhis2_metadata.py first and fill in the UIDs at the top of this script.")
        sys.exit(1)

    years = list(range(START_YEAR, END_YEAR + 1))
    planned_chunks = [(label, uid, year) for label, uid in all_targets.items() for year in years]

    print(f"Planned: {len(all_targets)} indicators/data elements x {len(years)} years = {len(planned_chunks)} chunks")

    if args.dry_run:
        # Dry-run lists everything, placeholders included, so you can see what
        # still needs a UID before committing to a live pull.
        for label, uid, year in planned_chunks:
            flag = "  [UID NOT SET]" if "REPLACE" in uid.upper() else ""
            print(f"  would fetch: {label} ({uid}), {year}{flag}")
        return

    # Live run: refuse to start until every UID (and the org root) is resolved.
    check_uids_resolved(all_targets)

    username, password = get_credentials()
    session = requests.Session()
    session.auth = (username, password)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for label, uid, year in planned_chunks:
        out_path = OUT_DIR / f"{label}_{year}.csv"
        if out_path.exists():
            print(f"Skipping {label}, {year} — already downloaded ({out_path.name})")
            continue

        print(f"Fetching {label} ({uid}), {year}...")
        df = fetch_year_chunk(session, uid, year)
        if df is not None:
            df.to_csv(out_path, index=False)
            print(f"  -> {len(df)} rows written to {out_path.name}")

    print("\nDone. Combine per-year CSVs per indicator with pandas.concat if you want")
    print("single long-format files, similar to how collate_NASA_nc_files.py merges years.")
    print("Completion RATIOS (ANC4/ANC1, first-trimester/ANC1, Penta3/BCG, PNC/live-births)")
    print("are built downstream in combine_dhis2_data.py — this script only pulls the raw")
    print("numerator/denominator components.")


if __name__ == "__main__":
    main()
