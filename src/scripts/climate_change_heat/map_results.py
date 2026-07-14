"""
map_results.py

District-level choropleth maps of WBGT-attributable service disruption for the
DHIS2 / WBGT negative-binomial pipeline (companion to model_of_wbgt_dhis2.py).

Reads the per-facility-month prediction files that that script writes, aggregates
the two-model difference up to district level as a PERCENTAGE of baseline
(no-weather) expected appointments, and draws:

    (a) a single historical map          -> results_negbin_predictions_{service}.csv
    (b) a 3 x 3 SSP x model-tier grid     -> projection_{scenario}_{model}_{service}.csv

Population weighting is deliberately OUT of scope here.

--------------------------------------------------------------------------------
FACILITY -> DISTRICT
--------------------------------------------------------------------------------
No matching is done here. wbgt_facility_panels_all_indicators.py already resolved
each reporting facility to a district (its `Dist` covariate) and wrote it into
regression_panel_{indicator}.csv. The `facility` column in the prediction files
comes straight through from that panel, so facility -> Dist is an exact join on
the panel itself -- 100% coverage, no fuzzy matching, no wrong districts.

SIGN CONVENTION: difference = y_pred_base - y_pred_wx, so a deficit (heat
suppressing appointments) is difference > 0. We take that positive part below.
"""

import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SERVICE = "vmmc_first_visits"          # must match `indicator`/`service` upstream

MODEL_OUT_DIR = Path("/Users/rachelmurray-watson/Documents/Heat_data/Model_outputs")
MAP_DIR       = Path("/Users/rachelmurray-watson/Documents/Heat_data/Model_outputs/Maps")
MAP_DIR.mkdir(parents=True, exist_ok=True)

# Regression panels written by wbgt_facility_panels_all_indicators.py (carry Dist)
PANEL_DIR      = Path("/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/Indices")
PANEL_FAC_COL  = "facility"
PANEL_DIST_COL = "Dist"

# Shapefiles (admin2 ships inside the repo; rivers is cosmetic)
ADMIN2_SHP = Path("/Users/rachelmurray-watson/PycharmProjects/TLOmodel/resources/mapping/"
                  "ResourceFile_mwi_admbnda_adm2_nso_20181016.shp")
WATER_SHP  = Path("/Users/rachelmurray-watson/Documents/Heat_data/"
                  "Water_Supply_Control-Rivers-shp/Water_Supply_Control-Rivers.shp")

ON_UNMATCHED = "raise"                 # "raise" or "warn" for facilities absent from the panel

# Projection dimensions (match the model script)
SSP_SCENARIOS = ["ssp126", "ssp245", "ssp585"]
WBGT_MODELS   = ["lowest", "median", "highest"]

# Shared colour scale for the 9-panel grid (set both to None to autoscale)
GRID_VMIN = 0.0
GRID_VMAX = 5.0

# % denominator:  "all" = share of all baseline expected appts (interpretable);
#                 "deficit" = baseline over deficit months only (old precip script).
PCT_DENOMINATOR = "all"

CMAP = "Blues"

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def _harmonise_district_names(s: pd.Series) -> pd.Series:
    return (s.replace({"Mzimba North": "Mzimba", "Mzimba South": "Mzimba"})
             .replace({"Blantyre City": "Blantyre", "Mzuzu City": "Mzuzu",
                       "Lilongwe City": "Lilongwe", "Zomba City": "Zomba"}))


print("Loading shapefiles...")
malawi_admin2 = gpd.read_file(ADMIN2_SHP)
malawi_admin2["ADM2_EN"] = _harmonise_district_names(malawi_admin2["ADM2_EN"])
if malawi_admin2.crs is None:
    warnings.warn("admin2 shapefile had no CRS; assuming EPSG:4326.")
    malawi_admin2 = malawi_admin2.set_crs("EPSG:4326")

if WATER_SHP.exists():
    water_bodies = gpd.read_file(WATER_SHP)
    if water_bodies.crs is None:
        water_bodies = water_bodies.set_crs("EPSG:4326")
else:
    warnings.warn(f"water shapefile not found at {WATER_SHP}; maps drawn without it.")
    water_bodies = None

# ---------------------------------------------------------------------------
# Facility -> district: read Dist straight from the regression panel
# ---------------------------------------------------------------------------
print("Reading facility -> district from regression panel...")
_panel_path = PANEL_DIR / f"regression_panel_{SERVICE}.csv"
if not _panel_path.exists():
    raise FileNotFoundError(
        f"{_panel_path} not found -- point PANEL_DIR/SERVICE at the panel that "
        "wbgt_facility_panels_all_indicators.py wrote.")

_panel = pd.read_csv(_panel_path, usecols=[PANEL_FAC_COL, PANEL_DIST_COL])
crosswalk = (_panel.dropna(subset=[PANEL_DIST_COL])
             .drop_duplicates(PANEL_FAC_COL)
             .rename(columns={PANEL_FAC_COL: "facility", PANEL_DIST_COL: "district"}))
crosswalk["district"] = _harmonise_district_names(crosswalk["district"])

stray = set(crosswalk["district"]) - set(malawi_admin2["ADM2_EN"])
if stray:
    warnings.warn(f"Dist values not in admin2 shapefile (won't colour): {stray}")
print(f"  crosswalk: {len(crosswalk)} facilities, "
      f"{crosswalk['district'].nunique()} districts")


# ---------------------------------------------------------------------------
# prediction file -> per-district % disruption
# ---------------------------------------------------------------------------
def district_percentage_disruption(pred_path: Path) -> pd.Series:
    df = pd.read_csv(pred_path, parse_dates=["date"])
    for c in ("difference", "y_pred_base", "facility"):
        if c not in df.columns:
            raise KeyError(f"'{c}' missing from {pred_path.name} "
                           f"(have {list(df.columns)})")

    df = df.merge(crosswalk, on="facility", how="left")
    unmatched = sorted(df.loc[df["district"].isna(), "facility"].unique())
    if unmatched:
        msg = (f"{len(unmatched)} facilities in {pred_path.name} absent from the "
               f"panel crosswalk: {unmatched[:5]}")
        if ON_UNMATCHED == "raise":
            raise ValueError(msg + "  -- the predictions and panel disagree on "
                             "facilities; check SERVICE. Or set ON_UNMATCHED='warn'.")
        warnings.warn(msg)
        df = df.dropna(subset=["district"])

    deficit = df.assign(deficit=df["difference"].clip(lower=0))
    num = deficit.groupby("district")["deficit"].sum()
    if PCT_DENOMINATOR == "all":
        den = df.groupby("district")["y_pred_base"].sum()
    elif PCT_DENOMINATOR == "deficit":
        den = deficit.loc[deficit["deficit"] > 0].groupby("district")["y_pred_base"].sum()
    else:
        raise ValueError("PCT_DENOMINATOR must be 'all' or 'deficit'")

    return ((num / den) * 100.0).reindex(num.index)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _draw_choropleth(ax, pct_by_district: pd.Series, vmin, vmax, annotate=True):
    gdf = malawi_admin2.copy()
    gdf["pct"] = gdf["ADM2_EN"].map(pct_by_district)
    if water_bodies is not None:
        water_bodies.plot(ax=ax, facecolor="none", edgecolor="#999999",
                          linewidth=0.5, hatch="xxx")
        water_bodies.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)
    gdf.dropna(subset=["pct"]).plot(
        ax=ax, column="pct", cmap=CMAP, edgecolor="black", alpha=1,
        legend=False, vmin=vmin, vmax=vmax)
    if annotate:
        m, sd = gdf["pct"].mean(), gdf["pct"].std()
        ax.text(0.01, 0.10, f"Mean: {m:.2f}%\nSD: {sd:.2f}%",
                transform=ax.transAxes, fontsize=10, verticalalignment="top")
    return gdf


def plot_historical():
    path = MODEL_OUT_DIR / f"results_negbin_predictions_{SERVICE}.csv"
    print(f"\nHistorical map <- {path.name}")
    pct = district_percentage_disruption(path)
    fig, ax = plt.subplots(figsize=(10, 10))
    vmin = 0.0 if GRID_VMIN is None else GRID_VMIN
    vmax = float(np.nanmax(pct.values)) if GRID_VMAX is None else GRID_VMAX
    _draw_choropleth(ax, pct, vmin, vmax)
    ax.set_xlabel("Longitude", fontsize=10); ax.set_ylabel("Latitude", fontsize=10)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, orientation="vertical", shrink=0.8,
                 label="Potential disruption (%)")
    out = MAP_DIR / f"wbgt_disruption_map_historical_{SERVICE}.png"
    plt.tight_layout(); plt.savefig(out, dpi=600); plt.close()
    print(f"  saved {out}")


def plot_projection_grid():
    print("\nProjection grid (SSP x model tier)")
    fig, axes = plt.subplots(len(SSP_SCENARIOS), len(WBGT_MODELS), figsize=(18, 18))
    for i, scenario in enumerate(SSP_SCENARIOS):
        for j, model in enumerate(WBGT_MODELS):
            ax = axes[i, j]
            path = MODEL_OUT_DIR / f"projection_{scenario}_{model}_{SERVICE}.csv"
            if not path.exists():
                ax.set_axis_off()
                ax.text(0.5, 0.5, f"{scenario}/{model}\n(missing)", ha="center",
                        va="center", transform=ax.transAxes, fontsize=11, color="grey")
                print(f"  {scenario}/{model}: file not found -- blank panel")
                continue
            pct = district_percentage_disruption(path)
            _draw_choropleth(ax, pct, GRID_VMIN, GRID_VMAX)
            ax.set_title(f"{scenario}: {model}", fontsize=14)
            if i == len(SSP_SCENARIOS) - 1:
                ax.set_xlabel("Longitude", fontsize=10)
            if j == 0:
                ax.set_ylabel("Latitude", fontsize=10)
            print(f"  {scenario}/{model}: {pct.notna().sum()} districts")

    sm = plt.cm.ScalarMappable(cmap=CMAP,
                               norm=mcolors.Normalize(vmin=GRID_VMIN, vmax=GRID_VMAX))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, orientation="vertical", shrink=0.8,
                 label="Potential disruption (%)")
    out = MAP_DIR / f"wbgt_disruption_maps_projection_grid_{SERVICE}.png"
    plt.savefig(out, dpi=600); plt.close()
    print(f"  saved {out}")


if __name__ == "__main__":
    plot_historical()
    plot_projection_grid()
    print("\nDone.")
