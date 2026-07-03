"""
download_dhis2_data.py

Bulk-downloads DHIS2 data values (via the analytics endpoint) for a set of
chosen indicators/data elements, across all Malawi facilities, over a date
range — chunked by year and by indicator group to avoid timeouts, and
resumable (skips chunks that have already been downloaded).

Mirrors the structure of collate_NASA_nc_files.py: config at the top,
per-chunk try/except so one failure doesn't kill the whole run, --dry-run
support, and a guard against re-downloading existing output.

Before running this:
  1. Run fetch_dhis2_metadata.py and find the UIDs of the indicators/data
     elements you want in the resulting CSVs.
  2. Fill in INDICATOR_IDS / DATA_ELEMENT_IDS below.
  3. Decide your org unit level (facility, district, zone) via ORG_UNIT_ID
     and ORG_UNIT_LEVEL.

Usage:
    python download_dhis2_data.py
    python download_dhis2_data.py --dry-run
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

# Selected via fetch_dhis2_metadata.py output + manual review (see project notes).
# Keep them as a dict of {short_label: uid} so filenames stay readable.
# All are raw counts/totals rather than per-capita or per-1000 versions, since
# facility fixed effects in the panel regression already absorb catchment
# population differences (see rationale on OPD attendance for the general logic).
INDICATOR_IDS: dict[str, str] = {
    # --- reproductive / maternal health ---
    "anc1_coverage": "dSCLYaL8ouA",              # RHD N ANC 1 coverage (HMIS 15)
    "anc4_coverage": "yXtwUarjGFD",               # RHD N ANC 4th Visit Coverage
    "institutional_delivery_rate": "FKdaVOCu65k", # % of deliveries conducted in facility
    "fp_total_clients": "f3QE2jkAkAU",            # PI FP - Total number of clients who received FP (all methods, all visit types)

    # --- general service utilisation ---
    "opd_attendance": "g7azgsbbrEr",              # CLIN OPD Attendance (raw count)
    "ipd_total_admissions": "xSBdiGepfRL",         # CLIN Total Admissions from all causes

    # --- disease-specific ---
    "malaria_confirmed_cases": "lKItZy1EliJ",      # Confirmed malaria cases (microscopy or RDT)
    "diarrhoea_incidence_rate": "qjYPD4XbpBM",     # CHD Diarrhoea incidence rate (per 1,000)
    "pneumonia_incidence_u5": "TwSerlwIRI3",       # CHD N Pneumonia incidence rate in children under 5

    # --- HIV service volume (partial coverage only — see note below) ---
    "vmmc_first_visits": "jblfwnIgu1i",            # HIV P VMMC Total With 1st Visits
    # NOTE: general ART clinic visit counts are NOT available as an aggregate
    # indicator/data element — they live in DHIS2's Tracker program (individual-
    # level records via /api/trackedEntityInstances or /api/events), which needs
    # a different pull strategy than dataValueSets/analytics used here. Revisit
    # separately if ART visit volume becomes important to the panel.
}
DATA_ELEMENT_IDS: dict[str, str] = {
    "tb_notified_cases": "ToIzb32HQA7",            # TB New — New and relapse TB cases notified
    "pnc_mother_checked_48h": "JSN2CbjKFt9",       # RHD PNC # Mother Checked in <48 Hours (first PNC contact, WHO-standard window)
}

# Root org unit to pull under (e.g. national root UID) — data will be
# returned disaggregated down to ORG_UNIT_LEVEL beneath it.
ORG_UNIT_ID = "REPLACE_WITH_NATIONAL_ROOT_UID"
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


def fetch_year_chunk(session: requests.Session, dx_uid: str, year: int) -> pd.DataFrame | None:
    """
    Pull one indicator/data element for one calendar year, aggregated to
    ORG_UNIT_LEVEL beneath ORG_UNIT_ID, using the analytics endpoint.
    Retries on transient failures (timeouts, 5xx) before giving up.
    """
    params = {
        "dimension": [f"dx:{dx_uid}", f"pe:{year}", f"ou:LEVEL-{ORG_UNIT_LEVEL};{ORG_UNIT_ID}"],
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
        for label, uid, year in planned_chunks:
            print(f"  would fetch: {label} ({uid}), {year}")
        return

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


if __name__ == "__main__":
    main()
