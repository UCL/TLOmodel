"""
Scan the raw NEX-GDDP-CMIP6 directory tree for missing OR corrupt
(model, variable, year) combinations, then download just those files
via THREDDS NCSS.

A file is considered bad if it is:
  - absent entirely
  - zero bytes
  - unreadable by xarray (truncated / corrupt download)

Usage:
    python find_and_download_missing_NASA_data.py            # scan + download
    python find_and_download_missing_NASA_data.py --dry-run  # scan only, no downloads
"""

import re
import sys
import time
from pathlib import Path

import requests
import xarray as xr

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(
    "/Users/rachelmurray-watson/Documents/Heat_data/NASA_GDDP-CMIP6"
)

SCENARIOS  = ["ssp126", "ssp245", "ssp585"]
VARIABLES  = ["pr", "hurs", "huss", "rlds", "rsds", "sfcWind", "tas", "tasmax", "tasmin"]
YEARS      = list(range(2025, 2041))

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

VARIANT_OVERRIDES = {
    "UKESM1-0-LL":  "r1i1p1f2",
    "MIROC-ES2L":   "r1i1p1f2",
}
DEFAULT_VARIANT = "r1i1p1f1"

# Malawi bounding box
NORTH =  -9.36366167
SOUTH = -17.12627881
WEST  =  32.67161823
EAST  =  35.91841716

NCSS_BASE = "https://ds.nccs.nasa.gov/thredds/ncss/grid/AMES/NEX/GDDP-CMIP6"
TIMEOUT   = 300

FILENAME_RE = re.compile(
    r"^(?P<variable>[A-Za-z]+)_day_(?P<model>[A-Za-z0-9\-]+)_(?P<scenario>[a-z0-9]+)_"
    r"(?P<variant>r\d+i\d+p\d+f\d+)_gn_(?P<year>\d{4})_malawi\.nc$"
)

# ============================================================================
# Helpers
# ============================================================================

def variant_for(model: str) -> str:
    return VARIANT_OVERRIDES.get(model, DEFAULT_VARIANT)


def is_valid_netcdf(path: Path) -> tuple[bool, str]:
    """
    Return (True, "") if the file is a readable NetCDF,
    otherwise (False, reason_string).
    """
    if not path.exists():
        return False, "does not exist"
    if path.stat().st_size == 0:
        return False, "zero bytes"
    try:
        ds = xr.open_dataset(path, engine="netcdf4")
        ds.close()
        return True, ""
    except Exception as e:
        return False, f"corrupt ({e})"


# ============================================================================
# Scan
# ============================================================================

def find_missing_or_corrupt() -> list[tuple]:
    """
    Return a list of (model, scenario, variable, year, reason) tuples for
    every file that is absent OR fails the NetCDF validation check.
    Corrupt files are deleted so they will be re-downloaded cleanly.
    """
    missing   = []   # (model, scenario, variable, year, reason)
    report    = []

    for model in ACTIVE_MODELS:
        for scenario in SCENARIOS:
            for variable in VARIABLES:
                var_dir = BASE_DIR / model / scenario / variable

                bad_years: dict[int, str] = {}  # year -> reason

                # ── Check every year we expect ────────────────────────────
                for year in YEARS:
                    variant  = variant_for(model)
                    filename = (
                        f"{variable}_day_{model}_{scenario}_{variant}"
                        f"_gn_{year}_malawi.nc"
                    )
                    filepath = var_dir / filename

                    ok, reason = is_valid_netcdf(filepath)

                    if not ok:
                        bad_years[year] = reason

                        # Remove corrupt file so the download lands cleanly
                        if filepath.exists():
                            print(f"  [DELETE] {filepath.name}  ({reason})")
                            filepath.unlink()

                if bad_years:
                    year_list = sorted(bad_years)
                    reasons   = ", ".join(
                        f"{y}:{r}" for y, r in sorted(bad_years.items())
                    )
                    report.append(
                        f"  {model:20s} {scenario:8s} {variable:10s}: "
                        f"{len(bad_years)} bad → {year_list}\n"
                        f"    reasons: {reasons}"
                    )
                    for year, reason in bad_years.items():
                        missing.append((model, scenario, variable, year, reason))

    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 70)
    print("Missing / corrupt file scan")
    print("=" * 70)
    if report:
        print("\n".join(report))
    else:
        print("  Nothing missing — all files present and valid.")
    print(f"\nTotal files to (re)download: {len(missing)}")
    print("=" * 70)

    return missing


# ============================================================================
# Download
# ============================================================================

def download_one(model: str, scenario: str, variable: str, year: int) -> bool:
    variant = variant_for(model)
    out_dir = BASE_DIR / model / scenario / variable
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = (
        out_dir
        / f"{variable}_day_{model}_{scenario}_{variant}_gn_{year}_malawi.nc"
    )

    url = (
        f"{NCSS_BASE}/{model}/{scenario}/{variant}/{variable}/"
        f"{variable}_day_{model}_{scenario}_{variant}_gn_{year}_v2.0.nc"
        f"?var={variable}"
        f"&north={NORTH}&south={SOUTH}&west={WEST}&east={EAST}"
        f"&horizStride=1"
        f"&time_start={year}-01-01T12:00:00Z"
        f"&time_end={year}-12-31T12:00:00Z"
        f"&accept=netcdf4&addLatLon=true"
    )

    print(f"\n→ {model} {scenario} {variable} {year}")
    print(f"  URL: {url}")

    try:
        resp = requests.get(url, timeout=TIMEOUT, stream=True)

        if resp.status_code == 404:
            print("  ✗ 404 — not available on THREDDS")
            return False

        resp.raise_for_status()

        with open(out_file, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                fh.write(chunk)

        # ── Validate the file we just wrote ──────────────────────────────
        ok, reason = is_valid_netcdf(out_file)
        if not ok:
            print(f"  ✗ Downloaded file is still bad ({reason}) — deleting")
            out_file.unlink(missing_ok=True)
            return False

        size_mb = out_file.stat().st_size / (1024 * 1024)
        print(f"  ✔ Saved: {out_file.name}  ({size_mb:.1f} MB)")
        return True

    except requests.exceptions.RequestException as e:
        print(f"  ⚠ Request error: {e}")
        return False


# ============================================================================
# Main
# ============================================================================

def main():
    dry_run = "--dry-run" in sys.argv

    missing = find_missing_or_corrupt()

    if not missing:
        print("\nNothing to do.")
        return

    if dry_run:
        print(
            "\n--dry-run set: skipping downloads. "
            "Re-run without --dry-run to fetch these files."
        )
        return

    print(f"\nDownloading {len(missing)} file(s)...\n")
    succeeded = 0
    failed    = []

    for model, scenario, variable, year, reason in missing:
        ok = download_one(model, scenario, variable, year)
        if ok:
            succeeded += 1
        else:
            failed.append((model, scenario, variable, year, reason))
        time.sleep(1)   # be polite to the THREDDS server

    # ── Final report ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"Done.  {succeeded}/{len(missing)} succeeded.")

    if failed:
        print(f"\n{len(failed)} file(s) still missing / unavailable:")
        for model, scenario, variable, year, reason in failed:
            print(f"  {model:20s} {scenario:8s} {variable:10s} {year}  [{reason}]")

    print("\nNEXT STEPS:")
    print("  1. Re-run collate_NASA_nc_files.py  → rebuild Combined/")
    print("  2. Re-run calculating_wbgt_thermofeel.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
