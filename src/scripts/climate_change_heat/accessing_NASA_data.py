"""
Scan the raw NEX-GDDP-CMIP6 directory tree for missing (model, variable, year)
combinations, then download just those files via THREDDS NCSS.

This replaces manually noticing "sfcWind is short by N years" from error
messages — it checks every variable for every active model up front and
backfills everything in one pass.

Usage:
    python find_and_download_missing_NASA_data.py            # scan + download
    python find_and_download_missing_NASA_data.py --dry-run  # scan only, no downloads
"""

import re
import sys
import time
from pathlib import Path

import requests

# ============================================================================
# Configuration - must match collate_NASA_nc_files.py / accessing_NASA_data.py
# ============================================================================

BASE_DIR = Path(
    "/Users/rachelmurray-watson/Documents/Heat_data/NASA_GDDP-CMIP6"
)

SCENARIOS = ["ssp126", "ssp245", "ssp585"]
VARIABLES = ["pr", "hurs", "huss", "rlds", "rsds", "sfcWind", "tas", "tasmax", "tasmin"]
YEARS = list(range(2025, 2041))

# Only the models that actually have data on THREDDS for this scenario/period
# (matches the 9 models your collate script found files for)
ACTIVE_MODELS = [
    "ACCESS-ESM1-5",
    "CMCC-ESM2",
    "MPI-ESM1-2-HR",
    "GISS-E2-1-G",
    "CMCC-CM2-SR5",
    "ACCESS-CM2",
    "MIROC6",
    "CanESM5",
    "MIROC-ES2L",
]

# Most models use r1i1p1f1; a few use a different variant label
VARIANT_OVERRIDES = {
    "UKESM1-0-LL": "r1i1p1f2",
    "MIROC-ES2L": "r1i1p1f2",
}
DEFAULT_VARIANT = "r1i1p1f1"

# Malawi bounding box
NORTH = -9.36366167
SOUTH = -17.12627881
WEST = 32.67161823
EAST = 35.91841716

NCSS_BASE = "https://ds.nccs.nasa.gov/thredds/ncss/grid/AMES/NEX/GDDP-CMIP6"
TIMEOUT = 300

FILENAME_RE = re.compile(
    r"^(?P<variable>[A-Za-z]+)_day_(?P<model>[A-Za-z0-9\-]+)_(?P<scenario>[a-z0-9]+)_"
    r"(?P<variant>r\d+i\d+p\d+f\d+)_gn_(?P<year>\d{4})_malawi\.nc$"
)


def variant_for(model: str) -> str:
    return VARIANT_OVERRIDES.get(model, DEFAULT_VARIANT)


def find_missing():
    """
    Return a list of (model, scenario, variable, year) tuples that are
    missing from BASE_DIR/{model}/{scenario}/{variable}/.
    """
    missing = []
    report_lines = []

    for model in ACTIVE_MODELS:
        for scenario in SCENARIOS:
            for variable in VARIABLES:
                var_dir = BASE_DIR / model / scenario / variable

                present_years = set()
                if var_dir.exists():
                    for f in var_dir.glob(f"{variable}_day_{model}_{scenario}_*_gn_*_malawi.nc"):
                        m = FILENAME_RE.match(f.name)
                        if m:
                            present_years.add(int(m.group("year")))

                missing_years = sorted(set(YEARS) - present_years)

                if missing_years:
                    report_lines.append(
                        f"  {model:20s} {variable:10s}: missing {len(missing_years)} "
                        f"year(s) -> {missing_years}"
                    )
                    for year in missing_years:
                        missing.append((model, scenario, variable, year))

    print("=" * 70)
    print("Missing-file scan")
    print("=" * 70)
    if report_lines:
        print("\n".join(report_lines))
    else:
        print("  Nothing missing - all models/variables/years are present.")
    print(f"\nTotal missing files: {len(missing)}")
    print("=" * 70)

    return missing


def download_one(model, scenario, variable, year):
    variant = variant_for(model)
    out_dir = BASE_DIR / model / scenario / variable
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{variable}_day_{model}_{scenario}_{variant}_gn_{year}_malawi.nc"

    url = (
        f"{NCSS_BASE}/{model}/{scenario}/{variant}/{variable}/"
        f"{variable}_day_{model}_{scenario}_{variant}_gn_{year}_v2.0.nc"
        f"?var={variable}&north={NORTH}&south={SOUTH}&west={WEST}&east={EAST}"
        f"&horizStride=1&time_start={year}-01-01T12:00:00Z&time_end={year}-12-31T12:00:00Z"
        f"&accept=netcdf4&addLatLon=true"
    )

    print(f"\n→ {model} {scenario} {variable} {year}")
    print(f"  NCSS: {url}")

    try:
        resp = requests.get(url, timeout=TIMEOUT, stream=True)
        if resp.status_code == 404:
            print("  ✗ Not found (404) - not available on THREDDS")
            return False
        resp.raise_for_status()

        with open(out_file, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

        size_mb = out_file.stat().st_size / (1024 * 1024)
        print(f"  ✔ Saved: {out_file.name} ({size_mb:.1f} MB)")
        return True

    except requests.exceptions.RequestException as e:
        print(f"  ⚠ Error: {e}")
        return False


def main():
    dry_run = "--dry-run" in sys.argv

    missing = find_missing()

    if not missing:
        return

    if dry_run:
        print("\n--dry-run set: not downloading. Re-run without --dry-run to fetch these files.")
        return

    print(f"\nDownloading {len(missing)} missing file(s)...\n")
    succeeded = 0
    failed = []

    for model, scenario, variable, year in missing:
        ok = download_one(model, scenario, variable, year)
        if ok:
            succeeded += 1
        else:
            failed.append((model, scenario, variable, year))
        time.sleep(1)  # be polite to the THREDDS server

    print("\n" + "=" * 70)
    print(f"Done. {succeeded}/{len(missing)} succeeded.")
    if failed:
        print(f"\n{len(failed)} file(s) still missing (not available on THREDDS or failed):")
        for model, scenario, variable, year in failed:
            print(f"  {model} {scenario} {variable} {year}")
    print("\nNEXT STEP: re-run collate_NASA_nc_files.py to rebuild Combined/")
    print("with the newly downloaded data, then re-run calculating_wbgt_thermofeel.py.")
    print("=" * 70)


if __name__ == "__main__":
    main()
