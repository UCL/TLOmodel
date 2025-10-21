import datetime
from pathlib import Path
import pickle
import calendar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import Literal, Optional, Dict, Tuple
import requests

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

# Keep only those facilities whose location is available
lmis = lmis[lmis.lat.notna()]
# TODO assume something else about location of these facilities with missing location - eg. centroid of district?
# only 16 facilties have missing information

# For now, let's keep only 8 facilities in 1 district
#lmis = lmis[lmis.district == 'Balaka']
# Keep only facilities with Facility_Level == '1a'
#lmis = lmis[lmis["Facility_Level"] == '1a']

# -----------------------------------
# 1) Data exploration
# -----------------------------------
def compute_opening_balance(df: pd.DataFrame) -> pd.Series:
    """
    Opening balance from same-month records:
    OB = closing_bal - received + dispensed.
    This is equivalent to OB_(m) = CB_(m-1)
    """
    return df["closing_bal"] - df["received"] + df["dispensed"]

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
    df["opening_bal"] =  compute_opening_balance(df).replace([np.inf, -np.inf], np.nan).fillna(0.0)

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
        annot=True, annot_kws={"size": 10},
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

# -----------------------------------
# 2) Estimate travel time matrix
# -----------------------------------

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
    time_matrix: pd.DataFrame,
    radius_minutes: float,
    tau_keep: float = 2.0, # Keeper minimum (2 --> Donors want to keep at least 2 X AMC)
    tau_tar: float = 1.0, # Receiver target (1 --> Receivers want OB = AMC)
    alpha: float = 0.7, # Donation cap (between 0 and 1)
    # alpha can be set to 1 if donors only want to keep a min of tau_keep X AMC
    K_in: int = 2,     # max donors per receiver (per item)
    K_out: int = 2,    # max receivers per donor (per item)
    # TODO the above limits should be across items not per item
    Qmin_days: float = 7.0,  # minimum lot size in "days of demand" at receiver
    # i.e. at least Qmin_days worth of stock should be supplied for the transfer to be justifiable
    id_cols=("district","month","item_code"),
    facility_col="fac_name",
    amc_col="amc",
    close_cols=("closing_bal","received","dispensed"),
    return_edge_log: bool = True,
) -> pd.DataFrame:
    """
    Scenario 1 or 2: local swaps within a time radius.
    Presumed p' = min(1, OB'/AMC).
    Objective: maximize total filled deficits (sum of transfers), which is a proxy for maximizing total availability.

    Returns a copy of df with a new column 'available_prop_redis' for redistributed probability per row.
    """
    closing_bal, received, dispensed = close_cols
    out = df.copy()
    out["OB"] =  compute_opening_balance(out).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["OB_prime"] = out["OB"] # container for updated OB by (district, month, item, facility)

    # iterate per (district, month, item)
    group_cols = list(id_cols)
    EPS = 1e-9
    EPS_AMC = 1e-6

    edge_logs = []   # will log dicts with district, month, item_code, donor, receiver, units_moved, travel_minutes

    for (d, m, i), g in out.groupby(group_cols, sort=False):
        # facilities in this slice
        g = g.copy()

        # ---------- (1) Select the correct time matrix for this district / slice ----------
        if isinstance(time_matrix, dict):
            T_d = time_matrix.get(d)
            if T_d is None or T_d.empty:
                continue
        else:
            T_d = time_matrix

        # facilities present in this slice AND in the matrix index/cols
        slice_facs = g[facility_col].dropna().unique().tolist()
        present = [f for f in slice_facs if (f in  T_d.index and f in T_d.columns)]
        if len(present) < 2:
            continue

        # subset and sanitize travel-times (NaN -> inf)
        # This is to make sure that the redistribution LP is run only for those pairs for which a travel time could be calculated
        T_sub = T_d.loc[present, present].replace(np.nan, np.inf)

        # ---------- (2) Clean inputs (AMC, OB) ----------
        # build per-fac arrays
        AMC = (g.set_index(facility_col)[amc_col]
               .astype(float)
               .replace([np.inf, -np.inf], np.nan)
               .fillna(0.0))
        AMC[AMC <= 0.0] = EPS_AMC  # avoid zero AMC

        OB0 = (g.set_index(facility_col)["OB"]
               .astype(float)
               .replace([np.inf, -np.inf], np.nan)
               .fillna(0.0))

        # keep only present facilities
        AMC = AMC.reindex(present).fillna(EPS_AMC)
        OB0 = OB0.reindex(present).fillna(0.0)

        # ---------- (3) Surplus/deficit with finite guards ----------
        # donor surplus cap and receiver deficit
        surplus_cap = alpha * np.maximum(0.0, OB0.values - tau_keep * AMC.values)
        deficit = np.maximum(0.0, tau_tar * AMC.values - OB0.values)
        surplus_cap = np.where(np.isfinite(surplus_cap), surplus_cap, 0.0)
        deficit = np.where(np.isfinite(deficit), deficit, 0.0)

        donors_mask = surplus_cap > EPS
        recv_mask = deficit > EPS
        if not donors_mask.any() or not recv_mask.any():
            continue

        donors = list(AMC.index[donors_mask])
        receivers = list(AMC.index[recv_mask])

        smax_map = {fac: float(x) for fac, x in zip(AMC.index, surplus_cap)} # creates a dict with key = facility and value = donatable surplus
        # this is the upper bound for how much a facility can donate
        deficit_map = {fac: float(x) for fac, x in zip(AMC.index, deficit)} # creates a dict with key = facility and value = deficit
        # this is the upper bound for how much a facility can receive

        # ---------- (4) Build feasible edges within radius (finite & <= radius) ----------
        feasible = {}
        for g_fac in present:
            row = T_sub.loc[g_fac].to_numpy()
            mask = (row <= radius_minutes) & np.isfinite(row)
            feas = set(T_sub.columns[mask]) - {g_fac}
            feasible[g_fac] = feas

        # ---------- (5) Edge set with tight M_edge and min-lot per receiver ----------
        Qmin_units = {f: float(max(Qmin_days * AMC[f], 0.0)) for f in present}
        M_edge = {}
        for g_fac in donors:
            smax = smax_map.get(g_fac, 0.0)
            if smax <= EPS:
                continue
            for f_fac in feasible.get(g_fac, set()):
                if f_fac not in receivers:
                    continue
                h = deficit_map.get(f_fac, 0.0)
                M = min(h, smax)
                if np.isfinite(M) and M > EPS:
                    M_edge[(g_fac, f_fac)] = M

        if not M_edge:
            continue  # nothing feasible to move

        # Build MILP
        prob = LpProblem(f"Redistribution_{d}_{m}_{i}", LpMaximize)

        # Variables
        t = {}   # transfer amounts per (g,f): continuous >=0
        y = {}   # edge-activation binaries
        for (g_fac, f_fac), M in M_edge.items():
            t[(g_fac, f_fac)] = LpVariable(f"t_{g_fac}->{f_fac}", lowBound=0, upBound=M, cat=LpContinuous)
            y[(g_fac, f_fac)] = LpVariable(f"y_{g_fac}->{f_fac}", lowBound=0, upBound=1, cat=LpBinary)

        # objective: maximize total shipped (proxy for availability gain)
        prob += lpSum(t.values())

        # donor outflow caps
        for g_fac in donors:
            prob += lpSum(t[(g_fac, f_fac)] for f_fac in receivers if (g_fac, f_fac) in t) <= smax_map.get(g_fac, 0.0)

        # receiver inflow caps
        for f_fac in receivers:
            prob += lpSum(t[(g_fac, f_fac)] for g_fac in donors if (g_fac, f_fac) in t) <= deficit_map.get(f_fac, 0.0)

        # link t and y; enforce min-lot clipped by edge capacity
        for (g_fac, f_fac), M in M_edge.items():
            prob += t[(g_fac, f_fac)] <= M * y[(g_fac, f_fac)]
            qmin = min(Qmin_units.get(f_fac, 0.0), M)
            # if qmin is basically zero, don't force a min lot
            if qmin > EPS:
                prob += t[(g_fac, f_fac)] >= qmin * y[(g_fac, f_fac)]

        # cardinality limits
        for f_fac in receivers:
            prob += lpSum(y[(g_fac, f_fac)] for g_fac in donors if (g_fac, f_fac) in y) <= K_in
        for g_fac in donors:
            prob += lpSum(y[(g_fac, f_fac)] for f_fac in receivers if (g_fac, f_fac) in y) <= K_out

        # solve
        prob.solve(PULP_CBC_CMD(msg=False, cuts=0, presolve=True, threads=1))

        if LpStatus[prob.status] not in ("Optimal", "Optimal Infeasible", "Not Solved"):
            continue

        # apply transfers & LOG
        delta = {fac: 0.0 for fac in present}
        for (g_fac, f_fac), var in t.items():
            moved = float(value(var) or 0.0)
            if moved > EPS_AMC:
                delta[g_fac] -= moved
                delta[f_fac] += moved
                if return_edge_log:
                    edge_logs.append({
                        "district": d, "month": m, "item_code": i,
                        "donor_fac": g_fac, "receiver_fac": f_fac,
                        "units_moved": moved,
                        "travel_minutes": float(T_sub.loc[g_fac, f_fac]) if np.isfinite(
                            T_sub.loc[g_fac, f_fac]) else np.nan
                    })

        # write OB'
        sel = (out["district"].eq(d) & out["month"].eq(m) & out["item_code"].eq(i))
        out.loc[sel, "OB_prime"] = out.loc[sel].apply(
            lambda r: (r["OB"] + delta.get(r[facility_col], 0.0)), axis=1
        )

        print(d, m, i,
              "donors:", len(donors),
              "receivers:", len(receivers),
              "edges:", len(M_edge))

    # ---------- (8) Mechanistic availability after redistribution ----------
    out["available_prop_redis"] = np.minimum(
        1.0,
        np.maximum(0.0, out["OB_prime"] / np.maximum(EPS_AMC, out[amc_col].astype(float).values))
    )

    # build log DataFrames
    edge_log_df = pd.DataFrame(edge_logs,
                               columns=["district", "month", "item_code", "donor_fac", "receiver_fac", "units_moved",
                                        "travel_minutes"]) if return_edge_log else None

    return out, edge_log_df

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
    keep_baseline_for_amc0: bool = True,   # leave baseline availability where AMC≈0
    amc_eps: float = 1e-6,                  # threshold to treat AMC as "zero"
    return_move_log: bool = True, # return a detailed df showing net movement of consumables after redistribution
    pooling_level: str = "district",  # "district" or "cluster"
    cluster_map: pd.Series | None = None,  # required if pooling_level=="cluster"; this specifes which cluster each facility belongs to
    enforce_no_harm: bool = True,  # if True, enforce p_f >= baseline at each facility (hard constraint)
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

    # Safe opening balance
    out["OB"] = compute_opening_balance(out).replace([np.inf, -np.inf], np.nan).fillna(0.0)

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
        Pbase = (g.set_index(facility_col)["available_prop"]
                 .astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0))

        # collapse duplicates if any
        if AMC.index.duplicated().any():
            AMC = AMC[~AMC.index.duplicated(keep="first")]
        if LVL.index.duplicated().any():
            LVL = LVL[~LVL.index.duplicated(keep="first")]
        if Pbase.index.duplicated().any():
            Pbase = Pbase.groupby(level=0).mean()  # average baseline if duplicates
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
        Pbase_pos = Pbase.reindex(facs_pos).fillna(0.0)

        # policy floors/ceilings (raw)
        tau_min_floor = (tau_min * AMC_pos).astype(float)
        tau_max_ceiling = (tau_max * AMC_pos).astype(float)

        # ---- Donor protection: x_f >= pLB_f = min(OB, tau_donor_keep*AMC)
        pLB = np.minimum(OB0_pos, tau_donor_keep * AMC_pos)  # the lower bound for a donor facility is OB if that is less than
        #tau_donor_keep X AMC or tau_donor_keep X AMC is that is lower than OB (i.e. only OB - 3 X AMC can be donated, if tau_donor_keep = 3)

        # ---- Receiver eligibility: only 1a/1b may increase above OB
        eligible_mask = LVL_pos.isin(["1a", "1b"])
        # For eligible: UB = tau_max*AMC
        # For ineligible: UB = min(OB, tau_max*AMC)  (can donate, cannot receive above OB)
        UB = tau_max_ceiling.copy()
        UB.loc[~eligible_mask] = np.minimum(OB0_pos.loc[~eligible_mask], tau_max_ceiling.loc[~eligible_mask])

        # ---- Lower bound assembly BEFORE scaling tau_min
        LB0 = np.maximum(pLB, tau_min_floor)  # enforces receiver's lower bound

        # ---- Optional "no-harm": p_f >= baseline -> x_f >= AMC_f * p_base_f
        if enforce_no_harm:
            no_harm_lb = (AMC_pos * Pbase_pos).astype(float)
            LB0 = np.maximum(LB0, no_harm_lb)

        # ---- Feasibility: scale ONLY the tau_min component if needed
        sum_LB0 = float(LB0.sum())
        if total_stock + 1e-9 < sum_LB0:
            # We want LB = max(pLB, tau_min_scaled, [no_harm_lb])
            # Scale only the portion of tau_min_floor that lies above pLB (and above no_harm if on).
            base_guard = pLB.copy()
            if enforce_no_harm:
                base_guard = np.maximum(base_guard, (AMC_pos * Pbase_pos))

            # Decompose tau_min contribution above base_guard
            tau_min_above = np.maximum(0.0, tau_min_floor - np.minimum(tau_min_floor, base_guard))
            need = float(tau_min_above.sum())
            # Available budget for that component:
            budget = total_stock - float(base_guard.sum())
            scale = 0.0 if need <= 1e-12 else max(0.0, min(1.0, budget / max(1e-9, need)))

            tau_min_scaled = np.minimum(base_guard, tau_min_floor) + tau_min_above * scale
            LB = np.maximum(base_guard, tau_min_scaled)
        else:
            LB = LB0

        # ---- Excess sink if ceilings bind
        sum_UB = float(UB.sum())
        allow_excess_sink = total_stock > sum_UB + 1e-9

        # ---------- LP ----------
        prob = LpProblem(f"Pooling_{node_val}_{m}_{i}", LpMaximize)
        x = {f: LpVariable(f"x_{f}", lowBound=0) for f in facs_pos}
        p = {f: LpVariable(f"p_{f}", lowBound=0, upBound=1) for f in facs_pos}
        excess = LpVariable("excess", lowBound=0) if allow_excess_sink else None

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
            prob += float(AMC_pos.loc[f]) * p[f] <= x[f]  # p <= x/AMC

        # Solve
        prob.solve(PULP_CBC_CMD(msg=False, cuts=0, presolve=True, threads=1))
        if LpStatus[prob.status] != "Optimal":
            continue

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

        # Facilities with AMC<eps donate entirely: OB' = min(OB, 3*AMC) but since AMC≈0, pLB=0 -> OB' = 0
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

    # ---------- Availability after redistribution ----------
    amc_safe = out[amc_col].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pos_mask_all = amc_safe >= amc_eps

    p_mech = np.zeros(len(out), dtype=float) # mechanistically calculated p based on balances and AMC
    # (different from the base which was calculated from actual stkout_days recorded_
    denom = np.maximum(amc_eps, amc_safe[pos_mask_all].to_numpy())
    p_mech[pos_mask_all.values] = np.minimum(
        1.0, np.maximum(0.0, out.loc[pos_mask_all, "OB_prime"].to_numpy() / denom)
    )

    if keep_baseline_for_amc0:
        base = (out.loc[~pos_mask_all, "available_prop"]
                .astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)).to_numpy()
        p_mech[~pos_mask_all.values] = base
    else:
        p_mech[~pos_mask_all.values] = (out.loc[~pos_mask_all, "OB_prime"].to_numpy() > 0.0).astype(float)

    # For levels 1a and 1b force availability to be max of old (actual) versus new (mechnistically calculated)
    out["available_prop_redis_raw"] = p_mech
    out["available_prop_redis"] = (
        np.maximum(p_mech, out["available_prop"].astype(float).values)
        if floor_to_baseline else p_mech
    )
    # For levels other than 1a and 1b force availability to be equal to old
    mask_non_elig = ~out["Facility_Level"].isin(["1a", "1b"])
    out.loc[mask_non_elig, "available_prop_redis"] = out.loc[mask_non_elig,"available_prop"] # this should ideally happen automatically
    # however, there are facilities at levels 2-4 whether some stock out was experienced even though OB > AMC
    # We want to retain the original probability in these cases because our overall analysis is restricted to levels 1a and 1b

    # Per-row movement
    out["received_from_pool"] = out["OB_prime"] - out["OB"]

    # Check if the rules are correctly applied
    # No facility should end below min(OB, tau_donor_keep*AMC)
    viol_lb = out["OB_prime"] < (np.minimum(out["OB"], tau_donor_keep * out["amc"]) - 1e-6)
    #assert not viol_lb.any(), "Donor-protection LB violated (OB' < min(OB, tau_donor_keep*AMC))."
    print(out[viol_lb][['available_prop_redis', 'available_prop', 'OB', 'OB_prime', 'amc']])
    # Non-eligible levels never increase prob
    #mask_non_elig = ~out["Facility_Level"].isin(["1a", "1b"])
    #mask_correct = (out["available_prop_redis"] == out["available_prop"])
    #print(out[mask_non_elig & ~mask_correct][['available_prop_redis', 'available_prop', 'OB', 'OB_prime', 'amc']])

    if return_move_log:
        move_log = pd.DataFrame(move_rows)
        return out, move_log

    return out
# pooled_df, pool_moves = redistribute_pooling_lp(lmis, tau_min=0.25, tau_max=3.0, return_move_log=True)

# ------------------------
# 4) HIGH-LEVEL DRIVER
# ------------------------
def run_redistribution_scenarios(
    df: pd.DataFrame,
    scenario: Literal["radius_30","radius_60","pooling"],
    time_matrix: Optional[pd.DataFrame] = None,
    # tunables
    tau_keep: float = 2.0,
    tau_tar: float = 1.0,
    alpha: float = 0.7,
    K_in: int = 2,
    K_out: int = 2,
    Qmin_days: float = 7.0,
    tau_min: float = 0.75,
    tau_max: float = 3.0,
    # schema
    id_cols=("district","month","item_code"),
    facility_col="fac_name",
    amc_col="amc",
    close_cols=("closing_bal","received","dispensed"),
    # aggregation for TLO input
    aggregate_for_tlo: bool = True,
    level_col: str = "Facility_Level"
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Runs exactly one scenario and returns:
      - detailed_df: row-level dataframe with 'available_prop_redis'
      - tlo_df (optional): aggregated to (item_code, district, Facility_Level, month) with mean availability

    For radius scenarios, 'time_matrix' is required. For pooling, it is ignored.
    """
    if scenario in ("radius_30","radius_60"):
        if time_matrix is None:
            raise ValueError("time_matrix is required for radius-based scenarios.")
        radius = 30 if scenario=="radius_30" else 60
        detailed = redistribute_radius_lp(
            df=df,
            time_matrix=time_matrix,
            radius_minutes=radius,
            tau_keep=tau_keep,
            tau_tar=tau_tar,
            alpha=alpha,
            K_in=K_in,
            K_out=K_out,
            Qmin_days=Qmin_days,
            id_cols=id_cols,
            facility_col=facility_col,
            amc_col=amc_col,
            close_cols=close_cols,
        )
    elif scenario == "pooling":
        detailed = redistribute_pooling_lp(
            df=df,
            tau_min=tau_min,
            tau_max=tau_max,
            id_cols=id_cols,
            facility_col=facility_col,
            amc_col=amc_col,
            close_cols=close_cols,
            return_move_log=False,
        )
    else:
        raise ValueError("scenario must be one of 'radius_30', 'radius_60', 'pooling'.")

    if not aggregate_for_tlo:
        return detailed, None

    # Aggregate to TLO input grain: mean across facilities (match your current pipeline)
    tlo = (
        detailed
        .groupby(["item_code", "district", level_col, "month"], as_index=False)
        .agg(new_available_prop=("available_prop_redis", "mean"))
        .sort_values(["item_code","district",level_col,"month"])
    )
    return tlo

# IMPLEMENT
# 1) Build a time matrix
fac_coords = lmis[['fac_name', 'district', 'lat','long']]
#T_car = build_time_matrices_by_district(
#    fac_coords,
#    mode="car",
#    backend="osrm",
#    osrm_base_url="https://router.project-osrm.org",
#    max_chunk=50)

# Plot stock adequacy by district and month
fig, ax, hm_df = generate_stock_adequacy_heatmap(df = lmis, figures_path = outputfilepath,
                                                 y_var = 'district', value_var = 'item_code',
                                                 value_label= f"% of consumables with Opening Balance ≥ 3 × AMC",
                                                 amc_threshold = 3, compare = "ge",
                                                 filename = "mth_district_stock_adequacy_3amc.png", figsize = (12,10))
fig, ax, hm_df = generate_stock_adequacy_heatmap(df = lmis, figures_path = outputfilepath,
                                                 y_var = 'district', value_var = 'item_code',
                                                 value_label= f"% of consumables with Opening Balance ≥ 3 × AMC",
                                                 amc_threshold = 2, compare = "ge",
                                                 filename = "mth_district_stock_adequacy_2amc.png", figsize = (12,10))
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
# Store dictionary in pickle format
#with open(outputfilepath / "T_car.pkl", "wb") as f:
#    pickle.dump(T_car, f)
# -> Commented out because it takes long to run. The result has been stored in pickle format

# Load pre-generated dictionary
with open(outputfilepath / "T_car.pkl", "rb") as f:
    T_car = pickle.load(f)

#edges_flat = build_edges_within_radius_flat(T_car, max_minutes= 60)

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

#lmis = lmis[lmis.district == 'Lilongwe']

# Trying the updated pool-based redistribution
# 1) Build clusters from your per-district travel-time matrices (minutes)
#    T_car_by_dist: {"District A": DF(index=fac_ids, cols=fac_ids), ...}
cluster_size = 3
cluster_series = build_capacity_clusters_all(T_car, cluster_size=cluster_size)
# cluster_series is a pd.Series: index=facility_id, value like "District A#C00", "District A#C01", ...

# 2) Pool at the cluster level
pooled_df, cluster_moves = redistribute_pooling_lp(
    df=lmis,  # your LMIS dataframe
    tau_min=0.25, tau_max=3.0,
    pooling_level="cluster",
    cluster_map=cluster_series,
    return_move_log=True,
)
pooled_df.to_csv(outputfilepath/ 'clustering_n3_df.csv', index=False)

# 3) Pool at the full district level (upper bound)
pooled_district_df, district_moves = redistribute_pooling_lp(
    df=lmis,
    tau_min=0.25, tau_max=3.0,
    pooling_level="district",
    return_move_log=True,
)
pooled_df.to_csv(outputfilepath/ 'clustering_district_df.csv', index=False)

# 2) Run Scenario 1 (1-hour radius)

tlo_30 = run_redistribution_scenarios(
    lmis,
    scenario="radius_30",
    time_matrix=T_car,
    tau_keep=2.0, tau_tar=1.0, alpha=1,
    K_in=2, K_out=2, Qmin_days=7.0,
)

tlo_60 = run_redistribution_scenarios(
    lmis,
    scenario="radius_60",
    time_matrix=T_car,
    tau_keep=2.0, tau_tar=1.0, alpha=1,
    K_in=2, K_out=2, Qmin_days=7.0,
)

tlo_pooling = run_redistribution_scenarios(
    lmis,
    scenario="pooling",
    tau_min=0.75, tau_max=3.0
)

tlo_availability_df = pd.read_csv(path_for_new_resourcefiles / "ResourceFile_Consumables_availability_small.csv")
# Drop any scenario data previously included in the resourcefile
tlo_availability_df = tlo_availability_df[['Facility_ID', 'month','item_code', 'available_prop']]

# Import item_category
program_item_mapping = pd.read_csv(path_for_new_resourcefiles  / 'ResourceFile_Consumables_Item_Designations.csv')[['Item_Code', 'item_category']]
program_item_mapping = program_item_mapping.rename(columns ={'Item_Code': 'item_code'})[program_item_mapping.item_category.notna()]

# 1.1.1 Attach district,  facility level and item_category to this dataset
#----------------------------------------------------------------
# Get TLO Facility_ID for each district and facility level
mfl = pd.read_csv(resourcefilepath / "healthsystem" / "organisation" / "ResourceFile_Master_Facilities_List.csv")
districts = set(pd.read_csv(resourcefilepath / 'demography' / 'ResourceFile_Population_2010.csv')['District'])
fac_levels = {'0', '1a', '1b', '2', '3', '4'}
tlo_availability_df = tlo_availability_df.merge(mfl[['District', 'Facility_Level', 'Facility_ID']],
                    on = ['Facility_ID'], how='left')

tlo_availability_df = tlo_availability_df.merge(program_item_mapping,
                    on = ['item_code'], how='left')
tlo_availability_df = tlo_availability_df[~tlo_availability_df[['District', 'Facility_Level', 'month', 'item_code']].duplicated()]


comparison_df = tlo_30.rename(columns ={'new_available_prop': 'available_prop_30mins'}).merge(tlo_availability_df, left_on = ['district', 'Facility_Level', 'month', 'item_code'],
                             right_on = ['District', 'Facility_Level', 'month', 'item_code'], how = 'left', validate = "1:1")
comparison_df = comparison_df.merge(tlo_60.rename(columns ={'new_available_prop': 'available_prop_60mins'}), on = ['district', 'Facility_Level', 'month', 'item_code'], how = 'left', validate = "1:1")
comparison_df = comparison_df.merge(tlo_pooling.rename(columns ={'new_available_prop': 'available_prop_pooling'}), on = ['district', 'Facility_Level', 'month', 'item_code'], how = 'left', validate = "1:1")
print(comparison_df['available_prop'].mean(),comparison_df['available_prop_30mins'].mean(),
comparison_df['available_prop_60mins'].mean(), comparison_df['available_prop_pooling'].mean())

comparison_df['available_prop_30mins'] = np.maximum(
    comparison_df['available_prop_30mins'],
    comparison_df['available_prop']
)
comparison_df['available_prop_60mins'] = np.maximum(
    comparison_df['available_prop_60mins'],
    comparison_df['available_prop']
)
comparison_df['available_prop_pooling'] = np.maximum(
    comparison_df['available_prop_pooling'],
    comparison_df['available_prop']
)
print(comparison_df['available_prop'].mean(),comparison_df['available_prop_30mins'].mean(),
comparison_df['available_prop_60mins'].mean(), comparison_df['available_prop_pooling'].mean())

# TODO keep only government facilities?
