"""
fetch_dhis2_org_metadata.py
===========================

One-time pull of DHIS2 facility (org-unit) METADATA: UID, code, coordinates,
and the district ancestor. This is what download_dhis2_data.py threw away by
using outputIdScheme=NAME.

Because the facility names in your wide panel are the canonical DHIS2 org-unit
names (same instance), you join this metadata to the panel by name EXACTLY —
no fuzzy matching. You then get:
  * lon/lat   -> extract WBGT at the DHIS2 coordinates (exposure no longer
                 depends on the registry name-match at all)
  * district  -> a reporting-side district for the match guard
  * ou_uid / ou_code -> a stable key to attach registry covariates

Writes: facility_org_metadata.csv  (one row per facility)

Run:  python fetch_dhis2_org_metadata.py
Uses the same BASE_URL / auth prompt as download_dhis2_data.py.
"""

import getpass
import json
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://dhis2.health.gov.mw/api"
OUT_PATH = Path("/Users/rachelmurray-watson/Documents/Heat_data/DHIS2_Malawi/combined/facility_org_metadata.csv")

ORG_UNIT_LEVEL = 4       # facility level (same as download_dhis2_data.py)
DISTRICT_LEVEL = 3       # district level in the Malawi hierarchy — CHECK against your instance
REQUEST_TIMEOUT = 120


def _point_from(ou):
    """Return (lon, lat) from either the modern 'geometry' field or the
    legacy 'coordinates' string. DHIS2 stores points as [lon, lat]."""
    geom = ou.get("geometry")
    if isinstance(geom, dict) and geom.get("type") == "Point":
        c = geom.get("coordinates")
        if isinstance(c, list) and len(c) == 2:
            return float(c[0]), float(c[1])
    coords = ou.get("coordinates")
    if coords:
        try:
            c = json.loads(coords)
            if isinstance(c, list) and len(c) == 2 and isinstance(c[0], (int, float)):
                return float(c[0]), float(c[1])
        except Exception:
            pass
    return None, None


def _district_from(ou):
    """District = the ancestor at DISTRICT_LEVEL. Falls back to the
    second-to-last ancestor if levels aren't returned."""
    anc = ou.get("ancestors", []) or []
    for a in anc:
        if a.get("level") == DISTRICT_LEVEL:
            return a.get("name")
    # fallback: ancestors are ordered root -> parent; district is usually the
    # last one before the facility itself
    return anc[-1]["name"] if anc else None


def main():
    username = input("DHIS2 username: ").strip()
    password = getpass.getpass("DHIS2 password: ")
    session = requests.Session()
    session.auth = (username, password)

    params = {
        "fields": "id,name,code,geometry,coordinates,ancestors[name,level]",
        "filter": f"level:eq:{ORG_UNIT_LEVEL}",
        "paging": "false",
    }
    resp = session.get(f"{BASE_URL}/organisationUnits.json", params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    ous = resp.json().get("organisationUnits", [])
    print(f"Fetched {len(ous)} org units at level {ORG_UNIT_LEVEL}")

    rows = []
    for ou in ous:
        lon, lat = _point_from(ou)
        rows.append({
            "facility": ou.get("name"),      # == the name in your wide panel
            "ou_uid": ou.get("id"),
            "ou_code": ou.get("code"),
            "district": _district_from(ou),
            "lon": lon,
            "lat": lat,
        })

    df = pd.DataFrame(rows)
    n_coord = int(df[["lon", "lat"]].notna().all(axis=1).sum())
    n_dist = int(df["district"].notna().sum())
    dup = df["facility"].duplicated(keep=False).sum()
    print(f"  with coordinates: {n_coord}/{len(df)}")
    print(f"  with district:    {n_dist}/{len(df)}")
    if dup:
        print(f"  ⚠ {dup} rows share a facility NAME (name not unique even in DHIS2) — "
              f"join on ou_uid for those, or disambiguate by district")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\n-> {OUT_PATH}")
    print("Join this to the wide panel on 'facility' (exact), then use lon/lat for WBGT "
          "extraction and 'district' for the match guard.")


if __name__ == "__main__":
    main()
