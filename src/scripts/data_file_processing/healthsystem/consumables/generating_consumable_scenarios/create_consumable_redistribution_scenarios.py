import datetime
from pathlib import Path
import pickle
import calendar

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import time

from typing import Literal, Optional, Dict, Tuple, Iterable
from functools import reduce
import requests
from collections import defaultdict

from pulp import LpProblem, LpMaximize, LpVariable, LpBinary, LpStatus, value, lpSum, LpContinuous, PULP_CBC_CMD
from math import ceil

# define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

# define folder pathways
outputfilepath = Path("./outputs/consumables_impact_analysis")
resourcefilepath = Path("./resources")
path_for_new_resourcefiles = resourcefilepath / "healthsystem/consumables"
# Set local shared drive source
path_to_share = Path(  # <-- point to the shared folder
    '/Users/sm2511/CloudStorage/OneDrive-SharedLibraries-ImperialCollegeLondon/TLOModel - WP - Documents/'
)

# 1. Import and clean data files
#**********************************
# Import Cleaned OpenLMIS data from 2018
lmis = (pd.read_csv(outputfilepath / "ResourceFile_Consumables_availability_and_usage.csv")
        [['district', 'fac_type_tlo', 'fac_name', 'month', 'item_code', 'available_prop',
       'closing_bal', 'amc', 'dispensed', 'received']])

# Drop duplicated facility, item, month combinations
print(lmis.shape, "rows before collapsing duplicates")
key_cols = ["district", "item_code", "fac_name", "month"] # keys that define a unique record

# helper to keep one facility level per group (mode → most common; fallback to first non-null)
def _mode_or_first(s: pd.Series):
    s = s.dropna()
    if s.empty:
        return np.nan
    m = s.mode()
    return m.iloc[0] if not m.empty else s.iloc[0]

lmis = (
    lmis
    .groupby(key_cols, as_index=False)
    .agg(
        closing_bal=("closing_bal", "sum"),
        dispensed=("dispensed", "sum"),
        received=("received", "sum"),
        amc=("amc", "sum"),
        available_prop=("available_prop", "mean"),
        fac_type_tlo=("fac_type_tlo", _mode_or_first),   # optional; remove if not needed
    )
)

print(lmis.shape, "rows after collapsing duplicates")

# Import data on facility location
location = (pd.read_excel(path_to_share / "07 - Data/Facility_GPS_Coordinates/gis_data_for_openlmis/LMISFacilityLocations_raw.xlsx")
        [['LMIS Facility List', 'LATITUDE', 'LONGITUDE']])
# Find duplicates in facilty names in the location dataset
duplicates = location[location['LMIS Facility List'].duplicated(keep=False)]
location = location.drop(duplicates[duplicates['LATITUDE'].isna()].index).reset_index(drop=True) # Drop those duplicates where location is missing
# Import ownership data
ownership = (pd.read_csv(path_to_share / "07 - Data/Consumables data/OpenLMIS/lmis_facility_ownership.csv"))[['fac_name', 'fac_owner']]
ownership = ownership.drop_duplicates(subset=['fac_name'])

# Merge OpenLMIS and location and ownership data
lmis = lmis.merge(location, left_on='fac_name', right_on = 'LMIS Facility List', how = 'left', validate='m:1')
lmis = lmis.merge(ownership, on='fac_name', how = 'left', validate='m:1')
lmis.rename(columns = {'LATITUDE':'lat', 'LONGITUDE':'long', 'fac_type_tlo': 'Facility_Level'}, inplace = True)

# Cleaning to match date to the same format as consumable availability RF in TLO model
month_map = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}
lmis["month"] = lmis["month"].map(month_map)
lmis["Facility_Level"] = lmis["Facility_Level"].str.replace("Facility_level_", "", regex=False)

# Clean data types before analysis
# 1) Normalize fac_name
lmis["fac_name"] = (
    lmis["fac_name"]
    .astype("string")            # Pandas string dtype (not just object)
    .str.normalize("NFKC")       # unify unicode forms
    .str.strip()                 # trim leading/trailing spaces
    .str.replace(r"\s+", "_", regex=True)  # collapse internal whitespace
)

# 2) Normalize other key columns used in grouping/joins
lmis["item_code"] = lmis["item_code"].astype("string").str.strip()
lmis["district"] = lmis["district"].astype("string").str.strip().str.replace(r"\s+", "_", regex=True)
lmis["Facility_Level"] = lmis["Facility_Level"].astype("string").str.strip()

# 3) Ensure numeric types (quietly coerce bad strings to NaN)
lmis["amc"] = pd.to_numeric(lmis["amc"], errors="coerce")
lmis["closing_bal"] = pd.to_numeric(lmis["closing_bal"], errors="coerce")

# Keep only those facilities whose location is available
old_facility_count = lmis.fac_name.nunique()
lmis = lmis[lmis.lat.notna()]
new_facility_count = lmis.fac_name.nunique()
print(f"{old_facility_count - new_facility_count} facilities out of {old_facility_count} in the lmis data dropped due to "
      f"missing location information")

# Explore missingness
def compute_opening_balance(df: pd.DataFrame) -> pd.Series:
    """
    Compute opening balance from same-month records.

    Formula:
        OB = closing_bal - received + dispensed
    Any negative OB values are replaced with 0.
    Equivalent to: OB_(m) = CB_(m-1)
    """
    ob = df["closing_bal"] - df["received"] + df["dispensed"]
    return ob.clip(lower=0)

# 1. Compute opening balance
lmis["opening_bal"] = compute_opening_balance(lmis).replace([np.inf, -np.inf], np.nan).fillna(0.0)
# Mechanistic probability (p_mech = OB / AMC)
amc_safe = np.maximum(1e-6, lmis["amc"].astype(float))
lmis["p_mech"] = np.clip(lmis["opening_bal"] / amc_safe, 0.0, 1.0)
# Identify inconsistent rows (where reported p > mechanistic p)
mask_inconsistent = lmis["p_mech"] < lmis["available_prop"]
# Adjust opening balance upward to match reported availability
lmis.loc[mask_inconsistent, "opening_bal"] = (
    lmis.loc[mask_inconsistent, "available_prop"] * lmis.loc[mask_inconsistent, "amc"]
)
print(f"Adjusted {mask_inconsistent.sum():,} rows "
      f"({mask_inconsistent.mean()*100:.2f}%) where recorded availability "
      f"exceeded mechanistic availability.")

lmis.reset_index(inplace=True, drop = True)

# TODO assume something else about location of these facilities with missing location - eg. centroid of district?
# only 16 facilties have missing information

# ----------------------------------------------------------------------------------------------------------------------
# 1) Data exploration
# ----------------------------------------------------------------------------------------------------------------------
def generate_stock_adequacy_heatmap(
    df: pd.DataFrame,
    figures_path: Path = Path("figures"),
    filename: str = "heatmap_adequacy_opening_vs_3xamc.png",
    y_var: str = "district", # the variable on the y-axis of the heatmap
    value_var: str = "item_code", # the count variable on the basis of which the values are calculated
    value_label: str = "", # label describing the values in the heatmap
    include_missing_as_fail: bool = False,   # if True, items with NaN opening/amc count as NOT adequate
    amc_threshold: float = 3.0,
    compare: str = "ge" ,        # "ge" for >= threshold*AMC, "le" for <= threshold*AMC
    decimals: int = 0,
    cmap: str = "RdYlGn",
    figsize= None,
    xtick_rotation: int = 45,
    ytick_rotation: int = 0,
    annotation: bool = True,
):
    """
    Heatmap values: for each (month, district), the % of item_code groups where
    sum(opening_balance over Facility_ID) >= 3 * sum(amc over Facility_ID).
    """
    df = df.copy()

    # --- 1. Ensure month is int and build label ---
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["month"])
    df["month"] = df["month"].astype(int)

    df["_month_label"] = df["month"].map(lambda m: calendar.month_abbr[m])

    # ---- 2) Aggregate to (month, district, item_code) over facilities ----
    agg = (
        df.groupby(["month", "_month_label", y_var, value_var], dropna=False)
          .agg(opening_bal=("opening_bal", "sum"),
               amc=("amc", "sum"))
          .reset_index()
    )

    # Keep:
    # - 1. all rows where amc != 0
    # - 2. rows where the (fac_name, item_code) pair never had any non-zero amc
    # (because this would indicate that their AMC may in fact be zero)
    # - 3. rows where both Opening balance and AMC are not zero
    agg = agg[(agg["amc"] != 0)]
    agg = agg[~((agg["amc"] == 0) & (agg["opening_bal"] == 0))]

    # ---- 3) Adequacy indicator per (month, district, item_code) ----
    if include_missing_as_fail:
        # NaNs treated as fail -> fill with NaN-safe compare: set False when either missing
        ok = agg[["opening_bal", "amc"]].notna().all(axis=1)
        left = agg["opening_bal"]
        right = amc_threshold * agg["amc"]
        if compare == "le":
            cond = (left <= right) & ok
        else:  # default to ">="
            cond = (left >= right) & ok
    else:
        valid = agg.dropna(subset=["opening_bal", "amc"])
        cond = pd.Series(False, index=agg.index)
        left = valid["opening_bal"]
        right = amc_threshold * valid["amc"]
        if compare == "le":
            cond.loc[valid.index] = left <= right
        else:
            cond.loc[valid.index] = left >= right

    agg["condition_met"] = cond.astype(int)

    # --- % meeting condition per (month, district) across item_code ---
    if include_missing_as_fail:
        denom = agg.groupby(["month", "_month_label", y_var])[value_var].nunique()
        numer = agg.groupby(["month", "_month_label", y_var])["condition_met"].sum()
    else:
        valid_mask = agg[["opening_bal", "amc"]].notna().all(axis=1)
        denom = agg[valid_mask].groupby(["month", "_month_label", y_var])[value_var].nunique()
        numer = agg[valid_mask].groupby(["month", "_month_label", y_var])["condition_met"].sum()

    pct = (numer / denom * 100).replace([np.inf, -np.inf], np.nan).reset_index(name="pct_meeting")

    # ---- 5) Pivot: districts (rows) x months (columns) ----
    # Sort months by _month_sort and use _month_label as the displayed column name
    month_order = (
        pct[["month", "_month_label"]]
        .drop_duplicates()
        .sort_values("month")
        ["_month_label"]
        .tolist()
    )
    heatmap_df = pct.pivot(index=y_var, columns="_month_label", values="pct_meeting")
    heatmap_df = heatmap_df.reindex(columns=month_order)

    # --- Add average row and column ---
    # Column average (mean of each month)
    heatmap_df.loc["Average"] = heatmap_df.mean(axis=0)
    # Row average (mean of each y_var)
    heatmap_df["Average"] = heatmap_df.mean(axis=1)

    # Fix overall average (bottom-right)
    overall_avg = heatmap_df.loc["Average", "Average"]
    heatmap_df.loc["Average", "Average"] = overall_avg

    # Optional rounding for nicer colorbar ticks (doesn't affect color)
    if decimals is not None:
        heatmap_df = heatmap_df.round(decimals)

    # --- Dynamic figure size ---
    if figsize is None:
        n_rows = len(heatmap_df)
        n_cols = len(heatmap_df.columns)
        height = max(6, n_rows * 0.2)  # taller if many rows
        width = max(8, n_cols * 0.6)
        figsize = (width, height)

    # ---- 6) Plot heatmap ----
    sns.set(font_scale=1.0)
    fig, ax = plt.subplots(figsize=figsize)

    cbar_kws = value_label
    hm = sns.heatmap(
        heatmap_df,
        cmap=cmap,
        cbar_kws={"label": value_label},
        ax=ax,
        annot=annotation, annot_kws={"size": 10},
        vmin = 0, vmax = 100)

    ax.set_xlabel("Month")
    ax.set_ylabel(f"{y_var}")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_rotation)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=ytick_rotation)

    # Keep colorbar ticks plain (no scientific notation)
    try:
        cbar_ax = ax.figure.axes[-1]
        cbar_ax.ticklabel_format(style="plain")
    except Exception:
        pass

    # ---- 7) Save & return ----
    figures_path.mkdir(parents=True, exist_ok=True)
    outpath = figures_path / filename
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return fig, ax, heatmap_df

# ----------------------------------------------------------------------------------------------------------------------
# 2) Estimate travel time matrix
# ----------------------------------------------------------------------------------------------------------------------
def _chunk_indices(n: int, chunk: int):
    """Yield (start, end) index pairs for chunking 0..n-1."""
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        yield s, e

def build_travel_time_matrix(
    fac_df: pd.DataFrame,
    *,
    id_col: str = "fac_name",
    lat_col: str = "lat",
    lon_col: str = "long",
    mode: Literal["car", "bicycle"] = "car",
    backend: Literal["ors", "osrm"] = "ors",
    # ORS options
    ors_api_key: Optional[str] = None,
    ors_base_url: str = "https://api.openrouteservice.org/v2/matrix",
    # OSRM options (self-hosted or public; note: public often has only 'car')
    osrm_base_url: str = "https://router.project-osrm.org",
    osrm_profile_map: dict = None,
    # matrix request chunking (keeps requests within API limits)
    max_chunk: int = 40,
    timeout: int = 60,
) -> pd.DataFrame:
    """
    Build an NxN *road* travel-time matrix (minutes) for facilities, by CAR or BICYCLE

    backends:
      - 'ors'  -> uses OpenRouteService Matrix API (profiles: driving-car, cycling-regular).
                  Requires ors_api_key. Has rate/size limits; we auto-chunk.
      - 'osrm' -> uses OSRM 'table' service (profiles typically 'car' or 'bike').
                  For bicycle, you'll likely need a self-hosted OSRM with the bicycle profile.

    Parameters
    ----------
    mode : 'car' | 'bicycle'
        Travel mode on roads.
    max_chunk : int
        Max #origins (and #destinations) per sub-matrix request to keep within API limits.

    Returns
    -------
    pd.DataFrame
        Square DataFrame (minutes), index/columns = facility names.
    """
    facs = fac_df[[id_col, lat_col, lon_col]].dropna().drop_duplicates().reset_index(drop=True)
    ids = facs[id_col].tolist()
    lats = facs[lat_col].to_numpy()
    lons = facs[lon_col].to_numpy()
    n = len(ids)

    T = pd.DataFrame(np.full((n, n), np.nan, dtype=float), index=ids, columns=ids)
    np.fill_diagonal(T.values, 0.0)

    if n == 0:
        return T

    if backend == "ors":
        if ors_api_key is None:
            raise ValueError("OpenRouteService requires ors_api_key.")
        profile = "driving-car" if mode == "car" else "cycling-regular"

        # ORS expects [lon, lat]
        coords = [[float(lons[i]), float(lats[i])] for i in range(n)]

        headers = {"Authorization": ors_api_key, "Content-Type": "application/json"}

        # Chunk both sources and destinations to stay under limits
        for si, sj in _chunk_indices(n, max_chunk):
            for di, dj in _chunk_indices(n, max_chunk):
                sources = list(range(si, sj))
                destinations = list(range(di, dj))

                body = {
                    "locations": coords,
                    "sources": sources,
                    "destinations": destinations,
                    "metrics": ["duration"],
                }
                url = f"{ors_base_url}/{profile}"
                r = requests.post(url, json=body, headers=headers, timeout=timeout)
                r.raise_for_status()
                data = r.json()

                # durations in seconds; fill submatrix
                durs = data.get("durations")
                if durs is None:
                    raise RuntimeError(f"ORS returned no durations for block {si}:{sj} x {di}:{dj}")
                sub = (np.array(durs, dtype=float) / 60.0)  # minutes
                T.iloc[si:sj, di:dj] = sub

    elif backend == "osrm":
        # Map desired mode to OSRM profile names
        if osrm_profile_map is None:
            osrm_profile_map = {"car": "car", "bicycle": "bike"}
        profile = osrm_profile_map.get(mode)
        if profile is None:
            raise ValueError(f"No OSRM profile mapped for mode='{mode}'.")

        # NOTE: The public OSRM demo often supports *car only*.
        # For bicycle, run your own OSRM with the bicycle profile.
        # We use the OSRM 'table' service; POST with 'sources' and 'destinations' indices.

        coords = ";".join([f"{lons[i]},{lats[i]}" for i in range(n)])  # lon,lat

        for si, sj in _chunk_indices(n, max_chunk):
            for di, dj in _chunk_indices(n, max_chunk):
                sources = ";".join(map(str, range(si, sj)))
                destinations = ";".join(map(str, range(di, dj)))

                url = (
                    f"{osrm_base_url}/table/v1/{profile}/{coords}"
                    f"?sources={sources}&destinations={destinations}&annotations=duration"
                )
                r = requests.get(url, timeout=timeout)
                r.raise_for_status()
                data = r.json()

                durs = data.get("durations")
                if durs is None:
                    raise RuntimeError(f"OSRM returned no durations for block {si}:{sj} x {di}:{dj}")
                sub = (np.array(durs, dtype=float) / 60.0)  # minutes
                T.iloc[si:sj, di:dj] = sub

    else:
        raise ValueError("backend must be 'ors' or 'osrm'.")

    # Safety: replace any residual NaNs (unroutable pairs) with inf or a large number
    T = T.fillna(np.inf)
    return T

# Because ORS and ORSM can only handle a limited number of elements at a time, it's better to run this by district
# each of which already has under 50 facilities
def build_time_matrices_by_district(
    df: pd.DataFrame,
    *,
    district_col: str = "district",
    id_col: str = "fac_name",
    lat_col: str = "lat",
    lon_col: str = "long",
    mode: str = "car",
    backend: str = "osrm",
    osrm_base_url: str = "https://router.project-osrm.org",
    ors_api_key: str | None = None,
    max_chunk: int = 50,               # safe for both OSRM/ORS
) -> dict[str, pd.DataFrame]:
    """
    Returns a dict: {district -> square minutes matrix DataFrame}, computed within each district only.
    """
    matrices = {}
    # unique facilities per district (drop duplicates in case of repeated rows)
    fac_cols = [district_col, id_col, lat_col, lon_col]
    fac_coords = df[fac_cols].dropna().drop_duplicates()

    for d, facs_d in fac_coords.groupby(district_col):
        # Skip tiny groups
        if len(facs_d) < 2:
            continue

        T = build_travel_time_matrix(
            fac_df=facs_d[[id_col, lat_col, lon_col]],
            id_col=id_col, lat_col=lat_col, lon_col=lon_col,
            mode=mode, backend=backend,
            osrm_base_url=osrm_base_url,
            ors_api_key=ors_api_key,
            max_chunk=max_chunk,
        )
        matrices[d] = T

    return matrices

# -----------------------------------------------
# 3) Data prep for redistribution linear program
# -----------------------------------------------
def presumed_availability(ob, amc, eps=1e-9) -> float:
    """
    Presumed likelihood of cons availability =  p = min(1, OB/AMC) at month start (no additional receipts considered
    at this point in time).
    """
    return float(min(1.0, max(0.0, (ob / max(eps, amc)))))


def build_edges_within_radius(
    time_matrix: pd.DataFrame,
    max_minutes: float
) -> Dict[str, set]:
    """
    For each facility g, return set of receivers f such that T[g,f] <= max_minutes, f != g.
    """
    edges = {}
    for g in time_matrix.index:
        feasible = set(time_matrix.columns[(time_matrix.loc[g] <= max_minutes) & (time_matrix.columns != g)])
        edges[g] = feasible
    return edges

def build_edges_within_radius_flat(T_by_dist: dict, max_minutes: float) -> dict[str, set[str]]:
    """
    Takes the district-wise dictionary of time travel matrices and converts it into a flat dictionary of facilities
    and their edge neighbours depending on the maximum allowable travel distance.
    T_by_dist: {district -> square DataFrame of minutes (index/cols = facility IDs)}
    Returns: {facility_id -> set(of facility_ids)} for all districts combined.
    """
    edges: dict[str, set[str]] = {}
    for _, T in T_by_dist.items():
        for g in T.index:
            row = T.loc[g].to_numpy()
            feasible_mask = (row <= max_minutes) & np.isfinite(row)
            # Exclude self
            feasible = [f for f in T.columns[feasible_mask] if f != g]
            if g not in edges:
                edges[g] = set()
            edges[g].update(feasible)
    return edges

# a = build_edges_within_radius(T_car, max_minutes = 18)

# Defining clusters of health facilities within district
# This function helps find the facilities which would be appropriate cluster centers
def _farthest_first_seeds(T: pd.DataFrame, k: int, big: float = 1e9) -> list:
    """
    Pick k seed medoids via farthest-first traversal on a travel-time matrix.
    Treat inf/NaN distances as 'big' so disconnected components get separate seeds.
    """
    n = T.shape[0]
    facs = T.index.tolist()
    D = T.to_numpy().astype(float)
    D[~np.isfinite(D)] = big

    # Start at the row with largest average distance (covers sparse areas first)
    start = int(np.nanargmax(np.nanmean(D, axis=1))) # the remotest facility
    seeds_idx = [start]

    # Iteratively add the point with max distance to its nearest seed
    for _ in range(1, k):
        # min distance to any existing seed for every point
        min_to_seed = np.min(D[:, seeds_idx], axis=1) # this has a dimension of [facs, number of seeds]
        next_idx = int(np.argmax(min_to_seed)) # for each facility find the distance to its nearest seed
        # and the facility farthest from the nearest seed becomes the next seed
        if next_idx in seeds_idx:
            # Fallback: pick any non-seed with highest min_to_seed
            candidates = [i for i in range(n) if i not in seeds_idx]
            if not candidates:
                break
            next_idx = int(candidates[np.argmax(min_to_seed[candidates])])
        seeds_idx.append(next_idx)

    return [facs[i] for i in seeds_idx] # list of length k representing the clustering points

# Assign each facility to its nearest seed subject to a hard cluster capacity (≤ X members)
def _assign_to_cluster_with_fixed_capacity(T: pd.DataFrame, seeds: list, capacity: int, big: float = 1e9) -> Dict[str, int]:
    """
    Greedy assignment of facilities to nearest seed that still has capacity (based on maximum cluster size).
    Returns: mapping facility -> seed_index (position in seeds list).
    """
    facs = T.index.tolist()
    D = T.loc[facs, seeds].to_numpy().astype(float) # Distance of all facilities from the k seeds
    D[~np.isfinite(D)] = big

    # each facility: nearest distance to any seed (for stable ordering)
    nearest = D.min(axis=1) # find the shortest distance to a cluster for each facility
    order = np.argsort(nearest)  # sort all facilities in ascending order of their distance from the nearest facility

    cap_left = {j: capacity for j in range(len(seeds))} # the capacity left for each seed
    assign = {}

    for idx in order:
        f = facs[idx]
        # try seeds in ascending distance
        seq = np.argsort(D[idx, :]) # the sequence of seeds most suitable for idx
        placed = False
        for j in seq:
            if cap_left[j] > 0:
                assign[f] = j
                cap_left[j] -= 1
                placed = True
                break
        if not placed:
            # total capacity >= n, so this should be rare; if it happens, put in least-loaded seed
            j = min(cap_left, key=lambda jj: cap_left[jj])
            assign[f] = j
            cap_left[j] -= 1

    return assign

def capacity_clusters_for_district(
    T_d: pd.DataFrame, cluster_size: int = 5, big: float = 1e9, refine_swaps: int = 0
) -> Dict[str, str]:
    """
    Build ~equal-size clusters (size<=cluster_size) from a district's travel-time matrix via
    capacity-constrained k-medoids (farthest-first seeds + greedy capacity assignment).

    Returns: {facility_id -> cluster_id} (cluster ids like 'C00','C01',...)
    """
    facs = T_d.index.tolist()
    n = len(facs)
    if n == 0:
        return {}
    if n <= cluster_size:
        return {f: "C00" for f in facs}

    k = ceil(n / cluster_size)
    seeds = _farthest_first_seeds(T_d, k=k, big=big)
    assign_seed_idx = _assign_to_cluster_with_fixed_capacity(T_d, seeds=seeds, capacity=cluster_size, big=big)

    # Optional tiny refinement: (off by default)
    # You could add 1–2 passes of medoid swap within clusters to reduce intra-cluster travel.

    # Build cluster ids in seed order
    seed_to_cid = {j: f"C{j:02d}" for j in range(len(seeds))}
    return {f: seed_to_cid[assign_seed_idx[f]] for f in facs}

def build_capacity_clusters_all(
    T_by_dist: Dict[str, pd.DataFrame], cluster_size: int = 5
) -> pd.Series:
    """
    Apply capacity clustering to all districts.
    Args:
      T_by_dist: {'DistrictName': time_matrix (minutes, square DF with facility ids)}
      cluster_size: desired max cluster size (e.g., 5)

    Returns:
      pd.Series mapping facility_id -> cluster_id names scoped by district, e.g. 'Nkhotakota#C03'
    """
    mappings = []
    for d, T_d in T_by_dist.items():
        if T_d is None or T_d.empty:
            continue
        local_map = capacity_clusters_for_district(T_d, cluster_size=cluster_size)
        if not local_map:
            continue
        s = pd.Series(local_map, name="cluster_id")
        s = s.map(lambda cid: f"{d}#{cid}")  # scope cluster name by district
        mappings.append(s)
    if not mappings:
        return pd.Series(dtype=object)
    return pd.concat(mappings)

# -----------------------------------------------
# 3) LP/MILP Redistribution
# -----------------------------------------------
def redistribute_radius_lp(
    df: pd.DataFrame,
    time_matrix: Dict[str, pd.DataFrame] | pd.DataFrame,
    radius_minutes: float,
    # policy knobs
    tau_keep: float = 1.0,                # donors must keep ≥ tau_keep * AMC
    tau_tar: float = 1.0,                 # receivers target OB = AMC
    K_in: int = 1,                        # per-item: max donors per receiver
    K_out: int = 10,                       # per-item: max receivers per donor
    Qmin_proportion: float = 0.25,        # min lot as a fraction of receiver AMC (e.g., 0.25 ≈ 7–8 days)
    eligible_levels: Iterable[str] = ("1a","1b"),
    # schema
    id_cols=("district","month","item_code"),
    facility_col="fac_name",
    level_col="Facility_Level",
    amc_col="amc",
    # outputs/behaviour
    return_edge_log: bool = True,
    floor_to_baseline: bool = True,       # if True, never let reported availability drop below baseline
    # numerics
    amc_eps: float = 1e-6,
    eps: float = 1e-9,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Pairwise (radius) redistribution with per-item degree caps.

    MILP per (district, month, item):
      variables: t[g,f] ≥ 0 (transfer), y[g,f] ∈ {0,1} (edge activation)
      objective: maximize Σ_p
      key constraints:
        - donors keep ≥ tau_keep * AMC
        - receivers (only eligible levels) limited to deficit: tau_tar*AMC - OB
        - travel-time ≤ radius
        - min lot: t[g,f] ≥ Qmin * y[g,f], with Qmin = Qmin_proportion * AMC_receiver
        - edge capacity: t[g,f] ≤ min(surplus_g, deficit_f) * y[g,f]
        - degree caps per item: inbound ≤ K_in, outbound ≤ K_out
    Availability is recomputed mechanistically and written back **only** where a transfer occurred.
    """
    out = df.copy()

    # Opening balance
    out["OB"] = out["opening_bal"]
    # Preserve only necessary columns
    selected_cols = list(id_cols) + [level_col, facility_col, 'OB', amc_col, 'available_prop']
    out = out[selected_cols]

    # clean AMC & level
    out[amc_col] = pd.to_numeric(out[amc_col], errors="coerce").fillna(0.0)
    out[level_col] = out[level_col].astype(str)

    # Container for updated OB and edge log
    out["OB_prime"] = out["OB"]
    edge_rows = [] if return_edge_log else None

    # Group at (district, month, item_code)
    group_cols = list(id_cols)

    skipped_nodes = []
    for (d, m, i), g in out.groupby(group_cols, sort=False):
        g = g.copy()

        # --- Pick the travel-time matrix slice ---
        if isinstance(time_matrix, dict):
            T_d = time_matrix.get(d)
            if T_d is None or T_d.empty:
                continue
        else:
            T_d = time_matrix  # single matrix for all, if you pass one

        facs_slice = g[facility_col].dropna().unique().tolist()
        facs = [f for f in facs_slice if f in T_d.index and f in T_d.columns]
        if len(facs) < 2:
            continue

        T_sub = T_d.loc[facs, facs].replace(np.nan, np.inf)

        # --- Pull per-fac data for this item ---
        AMC = (g.set_index(facility_col)[amc_col].astype(float)
                 .replace([np.inf, -np.inf], np.nan).fillna(0.0))
        OB0 = (g.set_index(facility_col)["OB"].astype(float)
                 .replace([np.inf, -np.inf], np.nan).fillna(0.0))
        LVL = (g.set_index(facility_col)[level_col].astype(str))


        # Align to facs and guard AMC
        AMC = AMC.reindex(facs).fillna(0.0)
        AMC_guard = AMC.copy()
        AMC_guard[AMC_guard <= 0.0] = amc_eps
        OB0 = OB0.reindex(facs).fillna(0.0)
        LVL = LVL.reindex(facs)

        # --- Surplus / deficit ---
        surplus = np.maximum(0.0, OB0.values - tau_keep * AMC_guard.values)  # donors
        deficit = np.maximum(0.0, tau_tar * AMC_guard.values - OB0.values)   # receivers

        # Leave AMC == 0 untouched
        #recv_pos_mask = AMC.values > amc_eps  # forbid AMC≈0 from receiving
        #deficit = np.where(recv_pos_mask, deficit0, 0.0)

        # Only eligible levels can receive
        elig_recv = LVL.isin(eligible_levels).values
        deficit = np.where(elig_recv, deficit, 0.0)

        donors = [f for f, s in zip(facs, surplus) if s > eps]
        recvs  = [f for f, h in zip(facs, deficit) if h > eps]
        if not donors or not recvs:
            continue

        s_map = dict(zip(facs, surplus))
        h_map = dict(zip(facs, deficit))
        qmin_map = dict(zip(facs, Qmin_proportion * AMC_guard.values))

        # --- Feasible edges (within radius), HARD PRUNE if capacity < qmin ---
        M_edge = {}   # capacity per edge
        Qmin = {}     # min lot per edge
        for g_fac in donors:
            row = T_sub.loc[g_fac].to_numpy()
            feas_idx = np.where((row <= radius_minutes) & np.isfinite(row))[0]
            for idx in feas_idx:
                f_fac = T_sub.columns[idx]
                if f_fac == g_fac or f_fac not in recvs:
                    continue
                M = min(s_map[g_fac], h_map[f_fac])
                if not np.isfinite(M) or M <= eps:
                    continue
                qmin = float(qmin_map[f_fac])
                if not np.isfinite(qmin) or qmin <= eps or M < qmin:
                    # cannot move at least qmin -> drop the edge
                    continue
                M_edge[(g_fac, f_fac)] = float(M)
                Qmin[(g_fac, f_fac)] = float(qmin)

        if not M_edge:
            continue

        # --- MILP (per item) ---
        prob = LpProblem(f"Radius_{d}_{m}_{i}", LpMaximize)

        # decision vars
        t = {e: LpVariable(f"t_{e[0]}->{e[1]}", lowBound=0, upBound=M_edge[e], cat=LpContinuous)
             for e in M_edge.keys()}
        y = {e: LpVariable(f"y_{e[0]}->{e[1]}", lowBound=0, upBound=1, cat=LpBinary)
             for e in M_edge.keys()}
        p = {f: LpVariable(f"p_{f}", lowBound=0, upBound=1) for f in
             facs}  # or only for eligible receivers

        # objective: maximize total shipped
        prob += lpSum(p[f] for f in recvs)  # or for level-eligible facilities

        AMC_guard = AMC.reindex(facs).fillna(0.0)
        AMC_guard[AMC_guard <= 0.0] = amc_eps

        # donor outflow caps (per item)
        for g_fac in donors:
            vars_out = [t[(g_fac, f_fac)] for (gg, f_fac) in t.keys() if gg == g_fac]
            if vars_out:
                s_cap = float(max(0.0, OB0[g_fac] - tau_keep * AMC_guard[g_fac]))
                prob += lpSum(vars_out) <= s_cap

        # receiver inflow caps (per item; eligibility already enforced)
        for f_fac in recvs:
            vars_in = [t[(g_fac, f_fac)] for (g_fac, ff) in t.keys() if ff == f_fac]
            if vars_in:
                h_cap = float(max(0.0, tau_tar * AMC_guard[f_fac] - OB0[f_fac]))
                prob += lpSum(vars_in) <= h_cap

        # link t & y + min-lot
        for e, M in M_edge.items():
            prob += t[e] <= M * y[e]
            qmin = min(Qmin[e], M_edge[e]) # TODO should this be qmin = Qmin[e]?
            if qmin > eps:
                prob += t[e] >= qmin * y[e]

        # per-item degree caps
        for f_fac in recvs:
            inbound_y = [y[(g_fac, f_fac)] for (g_fac, ff) in y.keys() if ff == f_fac]
            if inbound_y:
                prob += lpSum(inbound_y) <= K_in
        for g_fac in donors:
            outbound_y = [y[(g_fac, f_fac)] for (gg, f_fac) in y.keys() if gg == g_fac]
            if outbound_y:
                prob += lpSum(outbound_y) <= K_out

        # 3) Availability linearization per facility
        # Need inflow and outflow expressions from your t[(g,f)]
        for f_fac in facs:
            inflow = lpSum(t[(g_fac, f_fac)] for (g_fac, ff) in t.keys() if ff == f_fac)
            outflow = lpSum(t[(f_fac, h_fac)] for (gg, h_fac) in t.keys() if gg == f_fac)
            prob += float(AMC_guard[f_fac]) * p[f_fac] <= float(OB0.get(f_fac, 0.0)) + inflow - outflow

        # solve
        status = prob.solve(PULP_CBC_CMD(msg=False, cuts=0, presolve=True, threads=1))
        if LpStatus[prob.status] != "Optimal":
            skipped_nodes.append((d, m, i))
            continue

        # --- Apply transfers & log ---
        delta = {f: 0.0 for f in facs}  # net change in OB by facility (this item)
        any_transfer = False
        for (g_fac, f_fac), var in t.items():
            moved = float(value(var) or 0.0)
            if moved > eps:
                any_transfer = True
                delta[g_fac] -= moved
                delta[f_fac] += moved
                if return_edge_log:
                    tm = float(T_sub.loc[g_fac, f_fac]) if np.isfinite(T_sub.loc[g_fac, f_fac]) else np.nan
                    edge_rows.append({
                        "district": d, "month": m, "item_code": i,
                        "donor_fac": g_fac, "receiver_fac": f_fac,
                        "units_moved": moved, "travel_minutes": tm
                    })

        if not any_transfer:
            continue

        sel = (out["district"].eq(d) & out["month"].eq(m) & out["item_code"].eq(i))
        out.loc[sel, "OB_prime"] = out.loc[sel].apply(
            lambda r: r["OB"] + delta.get(r[facility_col], 0.0),
            axis=1
        )
    print("Skipped ", len(skipped_nodes), "district-month-item combinations - no optimal solution")

    # ---------- Availability: update ONLY where positive transfers happened ----------
    changed_mask = (out["OB_prime"] - out["OB"]) > 1e-6
    denom = np.maximum(amc_eps, out[amc_col].astype(float).values)
    p_mech = np.minimum(1.0, np.maximum(0.0, out["OB_prime"].values / denom))

    # start from baseline
    new_p = out["available_prop"].astype(float).values if floor_to_baseline else out["available_prop"].astype(float).values.copy()
    # update only changed rows; optionally floor to baseline
    if floor_to_baseline:
        new_p[changed_mask] = np.maximum(p_mech[changed_mask], out["available_prop"].astype(float).values[changed_mask])
    else:
        new_p[changed_mask] = p_mech[changed_mask]

    # force non-eligible levels to keep baseline (mirrors pooling)
    non_elig = ~out[level_col].isin(eligible_levels)
    new_p[non_elig] = out.loc[non_elig, "available_prop"].astype(float).values

    out["available_prop_redis"] = new_p

    edge_df = pd.DataFrame(edge_rows) if return_edge_log else None
    return out, edge_df

def redistribute_pooling_lp(
    df: pd.DataFrame,
    tau_min: float = 0.25,   # lower floor in "months of demand" (≈ 7–8 days) - minimum transfer required
    tau_max: float = 3.0,    # upper ceiling (storage/policy max)
    tau_donor_keep: float = 3.0, # minimum the donor keeps before donating
    id_cols=("district","month","item_code"),
    facility_col="fac_name",
    level_col="Facility_Level",
    amc_col="amc",
    close_cols=("closing_bal","received","dispensed"),
    amc_eps: float = 1e-6,                  # threshold to treat AMC as "zero"
    return_move_log: bool = True, # return a detailed df showing net movement of consumables after redistribution
    pooling_level: str = "district",  # "district" or "cluster"
    cluster_map: pd.Series | None = None,  # required if pooling_level=="cluster"; this specifes which cluster each facility belongs to
    floor_to_baseline: bool = True # if True, report availability floored to baseline (no decrease in outputs)
) -> pd.DataFrame:
    """
    Scenario 3: district-level pooling/push .
    Maximizes total availability with:
      - NaN/inf guards on AMC/OB
      - duplicate facility IDs collapsed within group
      - floors scaled if total stock < sum floors
      - optional 'excess' sink if total stock > sum ceilings
      - availability computed safely; AMC≈0 rows keep baseline (optional)

    Pooling redistribution that can operate at the district level (default)
    or within fixed-size clusters inside districts.

    If pooling_level == "cluster", you must pass cluster_map: Series mapping facility_id -> cluster_id
    (cluster ids should already be district-scoped, e.g., "Dedza#C01").

        Returns:
      - out: original df plus columns:
          OB, OB_prime, available_prop_redis, received_from_pool
        where received_from_pool = OB_prime - OB (pos=received, neg=donated)
      - (optional) move_log: per (district, month, item, facility) net movement summary
    """
    if pooling_level not in ("district", "cluster"):
        raise ValueError("pooling_level must be 'district' or 'cluster'.")

    closing_bal, received, dispensed = close_cols
    out = df.copy()

    # Opening balance
    # Ensure OB is consistent with observed availability
    amc_safe = np.maximum(1e-6, lmis["amc"].astype(float))
    lmis["p_mech"] = np.clip(lmis["opening_bal"] / amc_safe, 0.0, 1.0)

    mask_inconsistent = lmis["p_mech"] < lmis["available_prop"]
    lmis.loc[mask_inconsistent, "opening_bal"] = (
        lmis.loc[mask_inconsistent, "available_prop"] * lmis.loc[mask_inconsistent, "amc"]
    )
    out["OB"] = out["opening_bal"]

    # Default (will overwrite per group)
    out["OB_prime"] = out["OB"]

    # Attach cluster if needed
    if pooling_level == "cluster":
        if cluster_map is None:
            raise ValueError("cluster_map is required when pooling_level='cluster'.")
        # cluster_map: index = facility_id (facility_col), value = cluster_id (already district-scoped)
        out = out.merge(
            cluster_map.rename("cluster_id"),
            how="left",
            left_on=facility_col,
            right_index=True,
        )
        if out["cluster_id"].isna().any():
            # facilities missing a cluster—assign singleton clusters to keep them
            out["cluster_id"] = out["cluster_id"].fillna(
                out["district"].astype(str) + "#CXX_" + out[facility_col].astype(str))

    group_cols = list(id_cols)
    node_label = "district"
    if pooling_level == "cluster":
        group_cols = ["cluster_id", "month", "item_code"]
        node_label = "cluster_id"
    move_rows = []  # optional movement log
    # TODO could remove the movement log

    skipped_nodes = []  # collect nodes that did NOT solve optimally
    for keys, g in out.groupby(group_cols, sort=False):
        g = g.copy()
        # Resolve node ID for logging and selection masks
        if pooling_level == "district":
            node_val, m, i = g["district"].iloc[0], keys[1], keys[2]
        else:
            node_val, m, i = keys

        # Build per-facility Series (unique index) for AMC, OB, Level, Baseline p
        AMC = (g.set_index(facility_col)[amc_col]
               .astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0))
        OB0 = (g.set_index(facility_col)["OB"]
               .astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0))
        LVL = (g.set_index(facility_col)[level_col].astype(str)
               .replace({np.nan: ""}))

        # collapse duplicates if any
        if AMC.index.duplicated().any():
            AMC = AMC[~AMC.index.duplicated(keep="first")]
        if LVL.index.duplicated().any():
            LVL = LVL[~LVL.index.duplicated(keep="first")]
        if OB0.index.duplicated().any():
            OB0 = OB0.groupby(level=0).sum()

        total_stock = float(OB0.sum())
        if total_stock <= 1e-9:
            continue

        # Participants (positive demand)
        mask_pos = AMC >= amc_eps
        facs_pos = AMC.index[mask_pos].tolist()
        if not facs_pos:
            # nothing to reallocate to; they will be donors only (handled by OB' defaults)
            continue

        AMC_pos = AMC.loc[facs_pos]
        OB0_pos = OB0.loc[facs_pos]
        LVL_pos = LVL.reindex(facs_pos)

        # policy floors/ceilings
        tau_min_floor = (tau_min * AMC_pos).astype(float)
        donor_protect  = np.minimum(OB0_pos, tau_donor_keep * AMC_pos)  # retain min(OB, tau_donor_keep*AMC)
        LB0 = np.maximum(tau_min_floor, donor_protect)

        UB = (tau_max * AMC_pos).astype(float)
        UB.loc[~LVL_pos.isin(["1a", "1b"])] = np.minimum(
            OB0_pos.loc[~LVL_pos.isin(["1a", "1b"])],
            UB.loc[~LVL_pos.isin(["1a", "1b"])]
        )

        # Feasibility: scale only the tau_min component if sum LB > total_stock
        sum_LB0 = float(LB0.sum())
        if total_stock + 1e-9 < sum_LB0:
            # Scale down the tau_min part (not the donor protection)
            base_guard = donor_protect
            extra = np.maximum(0.0, tau_min_floor - np.minimum(tau_min_floor, base_guard))
            need = float(extra.sum())
            budget = total_stock - float(base_guard.sum())
            scale = 0.0 if need <= 1e-12 else max(0.0, min(1.0, budget / max(1e-9, need)))
            tau_min_scaled = np.minimum(base_guard, tau_min_floor) + extra * scale
            LB = np.maximum(base_guard, tau_min_scaled)
        else:
            LB = LB0

        # ---- Excess sink if ceilings bind
        sum_UB = float(UB.sum())
        allow_excess_sink = total_stock > sum_UB + 1e-9

        # 1) Per-facility feasibility guard
        bad = LB > UB + 1e-12
        if bad.any():
            # clip LB to UB; if that still leaves negative room, the facility is degenerate
            LB = np.minimum(LB, UB - 1e-9)

        # ---------- LP ----------
        prob = LpProblem(f"Pooling_{node_val}_{m}_{i}", LpMaximize)
        x = {f: LpVariable(f"x_{f}", lowBound=0) for f in facs_pos}
        p = {f: LpVariable(f"p_{f}", lowBound=0, upBound=1) for f in facs_pos}
        excess = LpVariable("excess", lowBound=0) if allow_excess_sink else None
        # note that even though facilities with AMC == 0 are not considered for optimisation, their postive OB is
        # included in the total stock

        # Objective: maximize total availability
        prob += lpSum(p.values())

        # Conservation
        if excess is None:
            prob += lpSum(x.values()) == total_stock
        else:
            prob += lpSum(x.values()) + excess == total_stock

        # Bounds + linearization
        for f in facs_pos:
            prob += x[f] >= float(LB.loc[f])  # donor protection + tau_min (scaled) + (optional) no-harm
            prob += x[f] <= float(UB.loc[f])  # eligibility-aware ceiling
            prob += float(max(AMC_pos.loc[f], amc_eps)) * p[f] <= x[f] # TODO CHECK max(AMC_pos.loc[f], amc_eps) or just AMC_pos

        # Solve
        prob.solve(PULP_CBC_CMD(msg=False, cuts=0, presolve=True, threads=1))
        if LpStatus[prob.status] != "Optimal":
            skipped_nodes.append((node_val, m, i))
            #print("NO Optimal solution found", node_val, m, i)
            continue
        #else:
            #print("Optimal solution found", node_val, m, i)

        # Apply solution to OB'
        x_sol = {f: float(value(var) or 0.0) for f, var in x.items()}

        # Selection mask for writing back
        if pooling_level == "district":
            sel = (out["district"].eq(node_val) & out["month"].eq(m) & out["item_code"].eq(i))
        else:
            sel = (out["cluster_id"].eq(node_val) & out["month"].eq(m) & out["item_code"].eq(i))

        # Facilities with AMC>=eps get x_f
        mask_rows_pos = sel & out[facility_col].isin(facs_pos)
        out.loc[mask_rows_pos, "OB_prime"] = out.loc[mask_rows_pos, facility_col].map(x_sol).values

        # Facilities with AMC<eps donate entirely: OB' = min(OB, tau_donor_keep*AMC) but since AMC≈0, pLB=0 -> OB' = 0
        # (this matches the "donate to pool" assumption)
        mask_rows_zero = sel & ~out[facility_col].isin(facs_pos)
        out.loc[mask_rows_zero, "OB_prime"] = 0.0

        if return_move_log:
            for f in AMC.index:  # include amc≈0 facilities (x_f=0)
                x_f = x_sol.get(f, 0.0) if f in facs_pos else 0.0
                net = x_f - float(OB0.get(f, 0.0))
                move_rows.append({
                    node_label: node_val,
                    "month": m,
                    "item_code": i,
                    "facility": f,
                    "received_from_pool": net,
                    "x_allocated": x_f,
                    "OB0_agg": float(OB0.get(f, 0.0)),
                    "eligible_receiver": bool(LVL.get(f, "") in {"1a", "1b"}),
                })

    # --- Availability after redistribution: update ONLY where OB' changed ---
    amc_safe_all = out[amc_col].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    denom = np.maximum(amc_eps, amc_safe_all.values)

    # Start from baseline everywhere
    out["available_prop_redis"] = out["available_prop"].astype(float).values

    # Change availability only for those rows where OB has increased
    changed = (out["OB_prime"] - out["OB"]) > 1e-6
    p_new = np.minimum(1.0, np.maximum(0.0, out.loc[changed, "OB_prime"].values / denom[changed]))
    if floor_to_baseline:
        p_new = np.maximum(p_new, out.loc[changed, "available_prop"].astype(float).values)

    out.loc[changed, "available_prop_redis"] = p_new
    out["received_from_pool"] = out["OB_prime"] - out["OB"]

    move_log = pd.DataFrame(move_rows) if return_move_log else None

    # Force non-eligible levels back to baseline (mirror analysis scope)
    non_elig = ~out[level_col].isin(["1a", "1b"])
    out.loc[non_elig, "available_prop_redis"] = out.loc[non_elig, "available_prop"].values  # this should ideally happen automatically
    # however, there are facilities at levels 2-4 whether some stock out was experienced even though OB > AMC
    # We want to retain the original probability in these cases because our overall analysis is restricted to levels 1a and 1b

    # Per-row movement
    out["received_from_pool"] = out["OB_prime"] - out["OB"]

    # Check if the rules are correctly applied
    # This section until  if return_move_log:, is not required for the solution
    #---------------------------------------------------------------------------------
    # --- Build masks for skipping ---
    # 1. Nodes that failed to solve optimally
    # Exclude any node/month/item combinations that didn't solve optimally or which were skipped due to AMC == 0
    if skipped_nodes:
        skipped_df = pd.DataFrame(skipped_nodes, columns=[node_label, "month", "item_code"])

        # Merge to flag rows belonging to skipped groups
        out = out.merge(
            skipped_df.assign(skip_flag=True),
            on=[node_label, "month", "item_code"],
            how="left",
        )
        mask_skip_solution = out["skip_flag"].fillna(False)
    else:
        mask_skip_solution = pd.Series(False, index=out.index)

    #out[mask_skip_solution].to_csv(outputfilepath / 'skipped_nodes_no_optiimal_soln.csv', index = False)

    # 2. Facilities with AMC effectively zero
    mask_skip_amc = out["amc"].astype(float) <= 1e-9

    # Combined skip mask
    mask_skip = mask_skip_solution | mask_skip_amc
    mask_solved = ~mask_skip
    print(f"Skipping {mask_skip.sum()} rows due to non-optimal LPs or AMC≈0")

    # No facility should end below min(OB, tau_donor_keep*AMC) (# Lower bound check)
    tol = 1e-6 #tolerance
    viol_lb = (
        (out.loc[mask_solved, "OB_prime"] <
         (np.minimum(out.loc[mask_solved, "OB"], tau_donor_keep * out.loc[mask_solved, "amc"]) - tol))
    )

    # No facility ends up above upper bounds (# Upper bound check)
    elig = out.loc[mask_solved, "Facility_Level"].isin(["1a", "1b"]).values
    ub = np.where(
        elig,
        tau_max * out.loc[mask_solved, "amc"],
        np.minimum(out.loc[mask_solved, "OB"], tau_max * out.loc[mask_solved, "amc"])
    )
    viol_ub = out.loc[mask_solved, "OB_prime"].values > (ub + tol)

    temp = out[mask_solved]
    if viol_lb.any():
        print("For the following rows (facility, item and month combinations), unclear why OB_prime < tau_donor_keep * AMC "
              "which violates a constraint in the LPP")
        print(temp[viol_lb][['Facility_Level', 'amc', 'OB', 'OB_prime']])
        temp[viol_lb][['Facility_Level', 'fac_name', 'amc', 'OB', 'OB_prime']].to_csv('violates_lb.csv')
    if viol_ub.any():
        print("For the following rows (facility, item and month combinations), unclear why OB_prime > tau_max * AMC "
              "which violates a constraint in the LPP")
        print(temp[viol_ub][['Facility_Level', 'amc', 'OB', 'OB_prime']])
        temp[viol_ub][['Facility_Level', 'fac_name', 'amc', 'OB', 'OB_prime']].to_csv('violates_ub.csv')

    if return_move_log:
        move_log = pd.DataFrame(move_rows)
        return out, move_log

    return out
# pooled_df, pool_moves = redistribute_pooling_lp(lmis, tau_min=0.25, tau_max=3.0, return_move_log=True)

# IMPLEMENT
# 1) Build a time matrix
fac_coords = lmis[['fac_name', 'district', 'lat','long']]
#T_car = build_time_matrices_by_district(
#    fac_coords,
#    mode="car",
#    backend="osrm",
#    osrm_base_url="https://router.project-osrm.org",
#    max_chunk=50)

# Store dictionary in pickle format
#with open(outputfilepath / "T_car2.pkl", "wb") as f:
#    pickle.dump(T_car, f)
# -> Commented out because it takes long to run. The result has been stored in pickle format

# Load pre-generated dictionary
with open(outputfilepath / "T_car2.pkl", "rb") as f:
    T_car = pickle.load(f)
# T_car2 was created after cleaning fac names and getting rid of spaces in the text

#edges_flat = build_edges_within_radius_flat(T_car, max_minutes= 60)

# 2) Explore the availability and distances to make decisions about optimisation rules
# Plot stock adequacy by district and month to assess what bounds to set when pooling
fig, ax, hm_df = generate_stock_adequacy_heatmap(df = lmis, figures_path = outputfilepath,
                                                 y_var = 'district', value_var = 'item_code',
                                                 value_label= f"% of consumables with Opening Balance ≥ 3 × AMC",
                                                 amc_threshold = 3, compare = "ge",
                                                 filename = "mth_district_stock_adequacy_3amc.png", figsize = (12,10))
fig, ax, hm_df = generate_stock_adequacy_heatmap(df = lmis, figures_path = outputfilepath,
                                                 y_var = 'district', value_var = 'item_code',
                                                 value_label= f"% of consumables with Opening Balance ≥ 1 × AMC",
                                                 amc_threshold = 1, compare = "ge",
                                                 filename = "mth_district_stock_adequacy_1amc.png", figsize = (12,10))
fig, ax, hm_df = generate_stock_adequacy_heatmap(df = lmis, figures_path = outputfilepath,
                                                 y_var = 'district', value_var = 'item_code',
                                                 value_label= f"% of consumables with Opening Balance <= 1 × AMC",
                                                 amc_threshold = 1, compare = "le",
                                                 filename = "mth_district_stock_inadequacy_1amc.png", figsize = (12,10))
fig, ax, hm_df = generate_stock_adequacy_heatmap(df = lmis, figures_path = outputfilepath,
                                                 y_var = 'item_code', value_var = 'fac_name',
                                                 value_label= f"% of facilities with Opening Balance ≥ 3 × AMC",
                                                 amc_threshold = 3, compare = "ge",
                                                 filename = "mth_item_stock_adequacy_3amc.png")
fig, ax, hm_df = generate_stock_adequacy_heatmap(df = lmis, figures_path = outputfilepath,
                                                 y_var = 'item_code', value_var = 'fac_name',
                                                 value_label= f"% of facilities with Opening Balance <= 1 × AMC",
                                                 amc_threshold = 1, compare = "le",
                                                 filename = "mth_item_stock_inadequacy_1amc.png")


# Browse the number of eligible neighbours depending on allowable travel time
results = []
for mins in [30, 60, 90, 120]:
    edges_flat = build_edges_within_radius_flat(T_car, max_minutes=mins)
    neighbors_count = pd.Series({fac: len(neigh) for fac, neigh in edges_flat.items()})
    mean = neighbors_count.mean()
    sem = neighbors_count.sem()  # standard error of mean
    ci95 = 1.96 * sem
    results.append({"radius": mins, "mean": mean, "ci95": ci95})

results_df = pd.DataFrame(results)

# Plot
plt.figure(figsize=(6,4))
plt.bar(results_df["radius"], results_df["mean"], yerr=results_df["ci95"], capsize=5, color="skyblue")
plt.xlabel("Travel time radius (minutes)")
plt.ylabel("Average number of facilities within radius")
plt.title("Average connectivity of facilities with 95% CI")
plt.xticks(results_df["radius"])
plt.savefig(outputfilepath / "neighbour_count_by_max_travel_time")

# A manual check shows that for distances greater than 60 minutes ORS underestimates the travel time a little
# TODO consider using google maps API

#Drop NAs
# TODO find a more generalisable solution for this issue (within the optimisation functions)
#lmis = lmis[(lmis.amc != 0) & (lmis.amc.notna())]

# ----------------------------------------------------------------------------------------------------------------------
# 3) Implement pooled redistribution
# ----------------------------------------------------------------------------------------------------------------------
# Build clusters from per-district travel-time matrices
#    T_car_by_dist: {"District A": DF(index=fac_ids, cols=fac_ids), ...}
cluster_size = 3
cluster_series = build_capacity_clusters_all(T_car, cluster_size=cluster_size)
# cluster_series is a pd.Series: index=facility_id, value like "District A#C00", "District A#C01", ...

# a) Run optimisation at district level
print("Now running Pooled Redistribution at District level")
start = time.time()
pooled_district_df, cluster_district_moves = redistribute_pooling_lp(
    df=lmis,  # the LMIS dataframe
    tau_min=0, tau_max=3.0,
    tau_donor_keep = 1.0,
    pooling_level="district",
    cluster_map=None,
    return_move_log=True,
    floor_to_baseline=True
)
print(pooled_district_df.drop(columns = ['LMIS Facility List', 'lat', 'long', 'fac_owner']).groupby('Facility_Level')[['available_prop_redis', 'available_prop']].mean())
pooled_district_df[['district', 'item_code',	'fac_name', 'month', 'amc', 'available_prop', 'Facility_Level',
                    'OB', 'OB_prime', 'available_prop_redis', 'received_from_pool']].to_csv(
                    outputfilepath/ 'clustering_district_df.csv', index=False)
tlo_pooled_district = (
        pooled_district_df
        .groupby(["item_code", "district", "Facility_Level", "month"], as_index=False)
        .agg(available_prop_scenario16=("available_prop_redis", "mean"))
        .sort_values(["item_code","district","Facility_Level","month"])
    )
end = time.time()
print(f"Completed in {end - start:.3f} seconds")

#  b) Run optimisation at cluster (size = 3) level
print("Now running pooled redistribution at Cluster (Size = 3) level")
start = time.time()
pooled_cluster_df, cluster_moves = redistribute_pooling_lp(
    df=lmis,  # the LMIS dataframe
    tau_min=0, tau_max=3.0,
    tau_donor_keep = 1.0,
    pooling_level="cluster",
    cluster_map=cluster_series,
    return_move_log=True,
    floor_to_baseline=True
)
print(pooled_cluster_df.drop(columns = ['LMIS Facility List', 'lat', 'long', 'fac_owner']).groupby('Facility_Level')[['available_prop_redis', 'available_prop']].mean())
pooled_cluster_df.to_csv(outputfilepath/ 'clustering_n3_df.csv', index=False)

tlo_pooled_cluster = (
        pooled_cluster_df
        .groupby(["item_code", "district", "Facility_Level", "month"], as_index=False)
        .agg(available_prop_scenario17=("available_prop_redis", "mean"))
        .sort_values(["item_code","district","Facility_Level","month"])
    )

end = time.time()
print(f"Completed in {end - start:.3f} seconds")

# c) Implement pairwise redistribution
print("Now running pairwise redistribution with maximum radius 60 minutes")
start = time.time()
# c.i) 1-hour radius
large_radius_df, large_radius_moves = redistribute_radius_lp(
    df=lmis,
    time_matrix=T_car,
    radius_minutes=60,      # facilities within 1 hour by car
    tau_keep=1.5,           # donor must keep 1 × AMC
    tau_tar=1.0,            # receivers target 1× AMC
    K_in=2,           # at most 1 inbound transfers per item
    K_out=10,          # at most 10 outbound transfers
    Qmin_proportion=0.25,          # min lot = one week of demand
    eligible_levels=("1a", "1b"),  # only 1a/1b can receive
)
print(large_radius_df.groupby('Facility_Level')[['available_prop_redis', 'available_prop']].mean())
large_radius_df.to_csv(outputfilepath/ 'large_radius_df.csv', index=False)

tlo_large_radius = (
        large_radius_df
        .groupby(["item_code", "district", "Facility_Level", "month"], as_index=False)
        .agg(available_prop_scenario18=("available_prop_redis", "mean"))
        .sort_values(["item_code","district","Facility_Level","month"])
    )

end = time.time()
print(f"Completed in {end - start:.3f} seconds")

# c.ii) 30-minute radius
print("Now running pairwise redistribution with maximum radius 30 minutes")
start = time.time()
small_radius_df, small_radius_moves = redistribute_radius_lp(
    df=lmis,
    time_matrix=T_car,
    radius_minutes=30,      # facilities within 1 hour by car
    tau_keep=1.5,           # donor must keep 1 × AMC
    tau_tar=1.0,            # receivers target 1 × AMC
    K_in=2,           # at most 1 inbound transfers per item
    K_out=10,          # at most 10 outbound transfers
    Qmin_proportion=0.25,          # min lot = one week of demand
    eligible_levels=("1a", "1b"),  # only 1a/1b can receive
)
print(small_radius_df.groupby('Facility_Level')[['available_prop_redis', 'available_prop']].mean())
small_radius_df.to_csv(outputfilepath/ 'small_radius_df.csv', index=False)

tlo_small_radius = (
        small_radius_df
        .groupby(["item_code", "district", "Facility_Level", "month"], as_index=False)
        .agg(available_prop_scenario19=("available_prop_redis", "mean"))
        .sort_values(["item_code","district","Facility_Level","month"])
    )

end = time.time()
print(f"Completed in {end - start:.3f} seconds")

# ----------------------------------------------------------------------------------------------------------------------
# 4) Compile update probabilities and merge with Resourcefile
# ----------------------------------------------------------------------------------------------------------------------
# 1) Merge the new dataframes together
# ----------------------------------------------------------------------------------------------------------------------
tlo_redis = reduce(
    lambda left, right: pd.merge(
        left, right,
        on=["item_code", "district", "Facility_Level", "month"],
        how="outer"
    ),
    [tlo_pooled_district, tlo_pooled_cluster, tlo_large_radius, tlo_small_radius]
)

# Edit new dataframe to match mfl formatting
list_of_new_scenario_variables = ['available_prop_scenario16', 'available_prop_scenario17',
                              'available_prop_scenario18', 'available_prop_scenario19']
tlo_redis = tlo_redis[['item_code', 'month', 'district', 'Facility_Level'] + list_of_new_scenario_variables].dropna()
tlo_redis["item_code"] = tlo_redis["item_code"].astype(float).astype(int)

# Load master facility list
mfl = pd.read_csv(resourcefilepath / "healthsystem" / "organisation" / "ResourceFile_Master_Facilities_List.csv")
mfl["District"] = mfl["District"].astype("string").str.strip().str.replace(r"\s+", "_", regex=True)
districts = set(mfl[mfl.District.notna()]["District"].unique())
kch = (mfl.Region == 'Central') & (mfl.Facility_Level == '3')
qech = (mfl.Region == 'Southern') & (mfl.Facility_Level == '3')
mch = (mfl.Region == 'Northern') & (mfl.Facility_Level == '3')
zmh = mfl.Facility_Level == '4'
mfl.loc[kch, "District"] = "Lilongwe"
mfl.loc[qech, "District"] = "Blantyre"
mfl.loc[mch, "District"] = "Mzimba"
mfl.loc[zmh, "District"] = "Zomba"

# Do some mapping to make the Districts line-up with the definition of Districts in the model
rename_and_collapse_to_model_districts = {
    'Nkhota_Kota': 'Nkhotakota',
    'Mzimba_South': 'Mzimba',
    'Mzimba_North': 'Mzimba',
    'Nkhata_bay': 'Nkhata_Bay',
}

tlo_redis['district_std'] = tlo_redis['district'].replace(rename_and_collapse_to_model_districts)
# Take averages (now that 'Mzimba' is mapped-to by both 'Mzimba South' and 'Mzimba North'.)
tlo_redis = tlo_redis.groupby(by=['district_std', 'Facility_Level', 'month', 'item_code'])[list_of_new_scenario_variables].mean().reset_index()

# Fill in missing data:
# 1) Cities to get same results as their respective regions
copy_source_to_destination = {
    'Mzimba': 'Mzuzu_City',
    'Lilongwe': 'Lilongwe_City',
    'Zomba': 'Zomba_City',
    'Blantyre': 'Blantyre_City'
}

for source, destination in copy_source_to_destination.items():
    new_rows = tlo_redis.loc[(tlo_redis.district_std == source) & (tlo_redis.Facility_Level.isin(['1a', '1b', '2']))].copy()
    new_rows.district_std = destination
    tlo_redis = pd.concat([tlo_redis, new_rows], axis=0, ignore_index=True)

# 2) Fill in Likoma (for which no data) with the means
means = tlo_redis.loc[tlo_redis.Facility_Level.isin(['1a', '1b', '2'])].groupby(by=['Facility_Level', 'month', 'item_code'])[
    list_of_new_scenario_variables].mean().reset_index()
new_rows = means.copy()
new_rows['district_std'] = 'Likoma'
tlo_redis = pd.concat([tlo_redis, new_rows], axis=0, ignore_index=True)
assert sorted(set(districts)) == sorted(set(pd.unique(tlo_redis.district_std)))

# 3) copy the results for 'Mwanza/1b' to be equal to 'Mwanza/1a'.
mwanza_1a = tlo_redis.loc[(tlo_redis.district_std == 'Mwanza') & (tlo_redis.Facility_Level == '1a')]
mwanza_1b = tlo_redis.loc[(tlo_redis.district_std == 'Mwanza') & (tlo_redis.Facility_Level == '1a')].copy().assign(Facility_Level='1b')
tlo_redis= pd.concat([tlo_redis, mwanza_1b], axis=0, ignore_index=True)

# 4) Copy all the results to create a level 0 with an availability equal to half that in the respective 1a
all_1a = tlo_redis.loc[tlo_redis.Facility_Level == '1a']
all_0 = tlo_redis.loc[tlo_redis.Facility_Level == '1a'].copy().assign(Facility_Level='0')
all_0[list_of_new_scenario_variables] *= 0.5
tlo_redis = pd.concat([tlo_redis, all_0], axis=0, ignore_index=True)

# Now, merge-in facility_id
tlo_redis = tlo_redis.merge(mfl[['District', 'Facility_Level', 'Facility_ID']],
                    left_on=['district_std', 'Facility_Level'],
                    right_on=['District', 'Facility_Level'], how='left', indicator=True, validate = 'm:1')
tlo_redis = tlo_redis[tlo_redis.Facility_ID.notna()][['Facility_ID', 'month', 'item_code'] + list_of_new_scenario_variables]
assert sorted(set(mfl.loc[mfl.Facility_Level != '5','Facility_ID'].unique())) == sorted(set(pd.unique(tlo_redis.Facility_ID)))

# Load original availability dataframe
# ----------------------------------------------------------------------------------------------------------------------
tlo_availability_df = pd.read_csv(path_for_new_resourcefiles / "ResourceFile_Consumables_availability_small.csv")
list_of_old_scenario_variables = [f"available_prop_scenario{i}" for i in range(1, 17)]
tlo_availability_df = tlo_availability_df[['Facility_ID', 'month', 'item_code'] + list_of_old_scenario_variables]

# Attach district,  facility level and item_category to this dataset
program_item_mapping = pd.read_csv(path_for_new_resourcefiles  / 'ResourceFile_Consumables_Item_Designations.csv')[['Item_Code', 'item_category']] # Import item_category
program_item_mapping = program_item_mapping.rename(columns ={'Item_Code': 'item_code'})[program_item_mapping.item_category.notna()]
fac_levels = {'0', '1a', '1b', '2', '3', '4'}
tlo_availability_df = tlo_availability_df.merge(mfl[['District', 'Facility_Level', 'Facility_ID']],
                    on = ['Facility_ID'], how='left')
tlo_availability_df = tlo_availability_df.merge(program_item_mapping,
                    on = ['item_code'], how='left')
tlo_availability_df = tlo_availability_df[~tlo_availability_df[['District', 'Facility_Level', 'month', 'item_code']].duplicated()]

# Interpolate missing values in tlo_redis for all levels except 0
# ----------------------------------------------------------------------------------------------------------------------
# Generate the dataframe that has the desired size and shape
fac_ids = set(mfl.loc[mfl.Facility_Level != '5'].Facility_ID)
item_codes = set(tlo_availability_df.item_code.unique())
months = range(1, 13)

# Create a MultiIndex from the product of fac_ids, months, and item_codes
index = pd.MultiIndex.from_product([fac_ids, months, item_codes], names=['Facility_ID', 'month', 'item_code'])

# Initialize a DataFrame with the MultiIndex and columns, filled with NaN
full_set = pd.DataFrame(index=index, columns=list_of_new_scenario_variables)
full_set = full_set.astype(float)  # Ensure all columns are float type and filled with NaN

# Insert the data, where it is available.
full_set = full_set.combine_first(tlo_redis.set_index(['Facility_ID', 'month', 'item_code'])[list_of_new_scenario_variables])

# Fill in the blanks with rules for interpolation.
facilities_by_level = defaultdict(set)
for ix, row in mfl.iterrows():
    facilities_by_level[row['Facility_Level']].add(row['Facility_ID'])

items_by_category = defaultdict(set)
for ix, row in program_item_mapping.iterrows():
    items_by_category[row['item_category']].add(row['item_code'])

def get_other_facilities_of_same_level(_fac_id):
    """Return a set of facility_id for other facilities that are of the same level as that provided."""
    for v in facilities_by_level.values():
        if _fac_id in v:
            return v - {_fac_id}

def get_other_items_of_same_category(_item_code):
    """Return a set of item_codes for other items that are in the same category/program as that provided."""
    for v in items_by_category.values():
        if _item_code in v:
            return v - {_item_code}
def interpolate_missing_with_mean(_ser):
    """Return a series in which any values that are null are replaced with the mean of the non-missing."""
    if pd.isnull(_ser).all():
        raise ValueError
    return _ser.fillna(_ser.mean())

# Create new dataset that include the interpolations (The operation is not done "in place", because the logic is based
# on what results are missing before the interpolations in other facilities).
full_set_interpolated = full_set * np.nan
full_set_interpolated[list_of_new_scenario_variables] = full_set[list_of_new_scenario_variables]

for fac in fac_ids:
    for item in item_codes:
        for col in list_of_new_scenario_variables:
            print(f"Now doing: fac={fac}, item={item}, column={col}")

            # Get records of the availability of this item in this facility.
            _monthly_records = full_set.loc[(fac, slice(None), item), col].copy()

            if pd.notnull(_monthly_records).any():
                # If there is at least one record of this item at this facility, then interpolate the missing months from
                # the months for there are data on this item in this facility. (If none are missing, this has no effect).
                _monthly_records = interpolate_missing_with_mean(_monthly_records)

            else:
                # If there is no record of this item at this facility, check to see if it's available at other facilities
                # of the same level
                # Or if there is no record of item at other facilities at this level, check to see if other items of this category
                # are available at this facility level
                facilities = list(get_other_facilities_of_same_level(fac))

                other_items = get_other_items_of_same_category(item)
                items = list(other_items) if other_items else other_items

                recorded_at_other_facilities_of_same_level = pd.notnull(
                    full_set.loc[(facilities, slice(None), item), col]
                ).any()

                if not items:
                    category_recorded_at_other_facilities_of_same_level = False
                else:
                    # Filter only items that exist in the MultiIndex at this facility
                    valid_items = [
                        itm for itm in items
                        if any((fac, m, itm) in full_set.index for m in months)
                    ]

                    category_recorded_at_other_facilities_of_same_level = pd.notnull(
                        full_set.loc[(fac, slice(None), valid_items), col]
                    ).any()

                if recorded_at_other_facilities_of_same_level:
                    # If it recorded at other facilities of same level, find the average availability of the item at other
                    # facilities of the same level.
                    print("Data for facility ", fac, " extrapolated from other facilities within level - ", facilities)
                    facilities = list(get_other_facilities_of_same_level(fac))
                    _monthly_records = interpolate_missing_with_mean(
                        full_set.loc[(facilities, slice(None), item), col].groupby(level=1).mean()
                    )

                elif category_recorded_at_other_facilities_of_same_level and valid_items:
                    # If it recorded at other facilities of same level, find the average availability of the item at other
                    # facilities of the same level.
                    print("Data for item ", item, " extrapolated from other items within category - ", valid_items)

                    _monthly_records = interpolate_missing_with_mean(
                        full_set.loc[(fac, slice(None), valid_items), col].groupby(level=1).mean()
                    )

                else:
                    # If it is not recorded at other facilities of same level, then assume that there is no change
                    print("No interpolation worked")
                    _monthly_records = _monthly_records.fillna(1.0)

            # Insert values (including corrections) into the resulting dataset.
            full_set_interpolated.loc[(fac, slice(None), item), col] = _monthly_records.values
            # temporary code
            assert full_set_interpolated.loc[(fac, slice(None), item), col].mean() >= 0

# Check that there are not missing values
assert not pd.isnull(full_set_interpolated).any().any()

full_set_interpolated = full_set_interpolated.reset_index()

# Merge with original dataset
tlo_availability_df = tlo_availability_df.merge(full_set_interpolated, on = ['Facility_ID', 'month', 'item_code'],
                                                how = 'left', validate = '1:1')
for var in list_of_new_scenario_variables:
    unchanged_levels = tlo_availability_df.Facility_Level.isin(['0', '2', '3', '4'])
    tlo_availability_df.loc[unchanged_levels, var] = tlo_availability_df.loc[unchanged_levels, 'available_prop']

tlo_availability_df.groupby('Facility_Level')[['available_prop'] + list_of_new_scenario_variables].mean()

# --- Check that the exported file has the properties required of it by the model code. --- #
# check_format_of_consumables_file(df=full_df_with_scenario, fac_ids=fac_ids)
# TODO Add check

# Save updated consumable availability resource file with scenario data
df_for_plots = tlo_availability_df.copy()
tlo_availability_df = tlo_availability_df.drop(columns = ['District', 'Facility_Level',
       'item_category'])
tlo_availability_df.to_csv(
    path_for_new_resourcefiles / "ResourceFile_Consumables_availability_small.csv",
    index=False
)

# Plot final availability
def plot_availability_heatmap(
    df: pd.DataFrame,
    x_var: str = None,
    y_var: str  = None,
    value_var: str  = None,
    scenario_cols: list[str]  = None,
    filter_dict: dict  = None,
    aggfunc: str = "mean",
    cmap: str = "RdYlGn",
    vmin: float = 0,
    vmax: float = 1,
    figsize: tuple = (12, 8),
    annot: bool = True,
    aggregate_label: str  = None,
    rename_dict: dict  = None,
    title: str = None,
    output_path: Path  = None,
):
    """
    Flexible heatmap generator that supports both:
    1. Single value_var column (long format)
    2. Multiple scenario columns (wide format, like available_prop_scen1–16)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    x_var : str, optional
        Column name for x-axis (ignored if scenario_cols is given).
    y_var : str, optional
        Column name for y-axis.
    value_var : str, optional
        Value variable for color intensity (ignored if scenario_cols is given).
    scenario_cols : list of str, optional
        List of scenario columns (e.g., [f"available_prop_scen{i}" for i in range(1,17)]).
    filter_dict : dict, optional
        Filters to apply before plotting, e.g. {"Facility_Level": "1a"}.
    aggfunc : str or callable, default "mean"
        Aggregation for pivot table.
    cmap : str
        Colormap.
    vmin, vmax : float
        Color scale range.
    figsize : tuple
        Figure size.
    annot : bool
        Annotate cells with values.
    aggregate_label : str, optional
        Adds an aggregate (mean) row at bottom if provided.
    rename_dict : dict, optional
        Rename columns (for pretty scenario names, etc.)
    title : str, optional
        Title for the plot.
    output_path : Path, optional
        Save path for PNG; if None, displays interactively.
    """

    df_plot = df.copy()

    # Apply filters
    if filter_dict:
        for k, v in filter_dict.items():
            if isinstance(v, (list, tuple, set)):
                df_plot = df_plot[df_plot[k].isin(v)]
            else:
                df_plot = df_plot[df_plot[k] == v]

    # Rename columns if requested
    if rename_dict:
        df_plot = df_plot.rename(columns=rename_dict)

    # Melt if scenario columns are provided
    if scenario_cols is not None:
        melted = df_plot.melt(
            id_vars=[c for c in df_plot.columns if c not in scenario_cols],
            value_vars=scenario_cols,
            var_name="scenario",
            value_name="value"
        )
        #melted["scenario_num"] = (
        #    melted["scenario"].str.extract(r"(\d+)").astype(float)
        #)
        df_plot = melted
        x_var = "scenario"
        value_var = "value"

    # Pivot table for heatmap
    heatmap_data = df_plot.pivot_table(
        index=y_var,
        columns=x_var,
        values=value_var,
        aggfunc=aggfunc
    )

    # Add aggregate row
    if aggregate_label:
        heatmap_data.loc[aggregate_label] = heatmap_data.mean(numeric_only=True)

    # Plot
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        heatmap_data,
        annot=annot,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        fmt=".2f",
        cbar_kws={"label": value_var.replace("_", " ").capitalize()},
        annot_kws={"size": 9}
    )

    # Titles & labels
    if title:
        ax.set_title(title, fontsize=14)
    ax.set_xlabel(x_var.replace("_", " ").capitalize())
    ax.set_ylabel(y_var.replace("_", " ").capitalize())

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


clean_category_names = {'cancer': 'Cancer', 'cardiometabolicdisorders': 'Cardiometabolic Disorders',
                        'contraception': 'Contraception', 'general': 'General', 'hiv': 'HIV', 'malaria': 'Malaria',
                        'ncds': 'Non-communicable Diseases', 'neonatal_health': 'Neonatal Health',
                        'other_childhood_illnesses': 'Other Childhood Illnesses', 'reproductive_health': 'Reproductive Health',
                        'road_traffic_injuries': 'Road Traffic Injuries', 'tb': 'Tuberculosis',
                        'undernutrition': 'Undernutrition', 'epi': 'Expanded programme on immunization'}
df_for_plots['item_category_clean'] = df_for_plots['item_category'].map(clean_category_names)

scenario_cols = ['available_prop_scenario1', 'available_prop_scenario2', 'available_prop_scenario3',
                 'available_prop_scenario6', 'available_prop_scenario7', 'available_prop_scenario8',
                 'available_prop_scenario16', 'available_prop_scenario17', 'available_prop_scenario18', 'available_prop_scenario19']
rename_dict = {'available_prop': 'Actual',
               'available_prop_scenario1': 'Non-therapeutic consumables',
               'available_prop_scenario2': 'Vital medicines',
               'available_prop_scenario3': 'Pharmacist-managed',
                'available_prop_scenario6': '75th percentile facility',
               'available_prop_scenario7': '90th percentile acility',
               'available_prop_scenario8': 'Best facility',
                 'available_prop_scenario16': 'District Pooling',
               'available_prop_scenario17' : 'Cluster Pooling',
               'available_prop_scenario18': 'Pairwise exchange (60-min radius)',
               'available_prop_scenario19': 'Pairwise exchange (30-min radius)'}
scenario_names = list(rename_dict.values())

plot_availability_heatmap(
    df=df_for_plots,
    scenario_cols=scenario_names,
    y_var="item_category_clean",
    filter_dict={"Facility_Level": ["1a"]},
    aggregate_label="Average",
    title="Availability across Scenarios — Level 1a",
    rename_dict = rename_dict,
    aggfunc = 'mean',
    cmap = "RdYlGn",
    output_path = outputfilepath / 'availability_1a.png'
)
plot_availability_heatmap(
    df=df_for_plots,
    scenario_cols=scenario_names,
    y_var="item_category_clean",
    filter_dict={"Facility_Level": ["1b"]},
    aggregate_label="Average",
    title="Availability across Scenarios — Level 1b",
    rename_dict = rename_dict,
    aggfunc = 'mean',
    cmap = "RdYlGn",
    output_path = outputfilepath / 'availability_1b.png'
)



