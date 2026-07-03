"""
fetch_dhis2_metadata.py

Pulls the full list of indicators and data elements (with IDs, names,
descriptions, and — for indicators — numerator/denominator formulas) from
the Malawi DHIS2 instance and writes them to CSV for browsing offline.

Run this FIRST, before downloading any actual data. The output CSVs let you
search/filter for the health indices you want (ANC, malaria, diarrhoea,
immunization, etc.) and grab their DHIS2 UIDs to feed into
download_dhis2_data.py.

Usage:
    python fetch_dhis2_metadata.py
    python fetch_dhis2_metadata.py --dry-run   # just test auth, no full pull
"""

import argparse
import getpass
import sys
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_URL = "https://dhis2.health.gov.mw/api"
OUT_DIR = Path("/Users/rachelmurray-watson/Documents/Heat_data/DHIS2_Malawi/metadata")

INDICATOR_FIELDS = "id,name,shortName,description,numeratorDescription,denominatorDescription,indicatorType[name]"
DATA_ELEMENT_FIELDS = "id,name,shortName,description,valueType,domainType,categoryCombo[name]"

PAGE_SIZE = 500  # DHIS2 default max is usually 500-1000; adjust if the server complains

# ---------------------------------------------------------------------------


def get_credentials():
    """Prompt for username/password rather than hardcoding them."""
    username = input("DHIS2 username: ").strip()
    password = getpass.getpass("DHIS2 password: ")
    return username, password


def test_auth(session: requests.Session) -> bool:
    """Quick check that credentials work before doing a full pull."""
    resp = session.get(f"{BASE_URL}/me.json", params={"fields": "username,organisationUnits[name]"})
    if resp.status_code == 200:
        me = resp.json()
        print(f"Authenticated as: {me.get('username')}")
        return True
    else:
        print(f"Auth failed: HTTP {resp.status_code} — {resp.text[:300]}")
        return False


def fetch_paginated(session: requests.Session, endpoint: str, fields: str) -> list[dict]:
    """
    Fetch all records from a metadata endpoint (indicators, dataElements, ...)
    using DHIS2's paging, since a full-country instance can have thousands
    of entries and a single request may be truncated or time out.
    """
    all_records = []
    page = 1
    while True:
        params = {"fields": fields, "paging": "true", "pageSize": PAGE_SIZE, "page": page}
        resp = session.get(f"{BASE_URL}/{endpoint}.json", params=params)
        if resp.status_code != 200:
            print(f"  Request failed on page {page}: HTTP {resp.status_code}")
            break

        data = resp.json()
        key = endpoint  # DHIS2 nests results under the endpoint name, e.g. "indicators"
        records = data.get(key, [])
        all_records.extend(records)

        pager = data.get("pager", {})
        total_pages = pager.get("pageCount", 1)
        print(f"  {endpoint}: page {page}/{total_pages} ({len(records)} records)")

        if page >= total_pages:
            break
        page += 1

    return all_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Only test authentication, skip the full pull")
    args = parser.parse_args()

    username, password = get_credentials()
    session = requests.Session()
    session.auth = (username, password)

    if not test_auth(session):
        sys.exit(1)

    if args.dry_run:
        print("Dry run complete — auth works. Re-run without --dry-run for the full metadata pull.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nFetching indicators (name, formula, description)...")
    indicators = fetch_paginated(session, "indicators", INDICATOR_FIELDS)
    df_indicators = pd.json_normalize(indicators)
    df_indicators.to_csv(OUT_DIR / "malawi_dhis2_indicators.csv", index=False)
    print(f"  -> {len(df_indicators)} indicators written to {OUT_DIR / 'malawi_dhis2_indicators.csv'}")

    print("\nFetching data elements (raw counted quantities)...")
    data_elements = fetch_paginated(session, "dataElements", DATA_ELEMENT_FIELDS)
    df_de = pd.json_normalize(data_elements)
    df_de.to_csv(OUT_DIR / "malawi_dhis2_data_elements.csv", index=False)
    print(f"  -> {len(df_de)} data elements written to {OUT_DIR / 'malawi_dhis2_data_elements.csv'}")

    print("\nDone. Open the CSVs and filter by name (e.g. 'ANC', 'malaria', 'diarrhoea',")
    print("'measles', 'outpatient') to find the indicators/data elements you want, and")
    print("note their 'id' column values — you'll need those UIDs for download_dhis2_data.py.")


if __name__ == "__main__":
    main()
