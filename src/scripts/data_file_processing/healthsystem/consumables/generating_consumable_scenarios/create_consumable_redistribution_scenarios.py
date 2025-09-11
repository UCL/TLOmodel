import datetime
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import Literal, Optional, Dict, Tuple
import requests

from pulp import LpProblem, LpMaximize, LpVariable, LpBinary, LpStatus, value, lpSum, LpContinuous

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
# 1) Estimate travel time matrix
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
# 2) Data prep for redistribution linear program
# -----------------------------------------------
def compute_opening_balance(df: pd.DataFrame) -> pd.Series:
    """
    Opening balance from same-month records:
    OB = closing_bal - received + dispensed.
    This is equivalent to OB_(m) = CB_(m-1)
    """
    return df["closing_bal"] - df["received"] + df["dispensed"]


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
) -> pd.DataFrame:
    """
    Scenario 1 or 2: local swaps within a time radius.
    Presumed p' = min(1, OB'/AMC).
    Objective: maximize total filled deficits (sum of transfers), which is a proxy for maximizing total availability.

    Returns a copy of df with a new column 'available_prop_redis' for redistributed probability per row.
    """
    closing_bal, received, dispensed = close_cols
    df = df.copy()
    df["OB"] = compute_opening_balance(df)

    # container for updated OB by (district, month, item, facility)
    df["OB_prime"] = df["OB"]

    # iterate per (district, month, item)
    group_cols = list(id_cols)
    for (d, m, i), g in df.groupby(group_cols):
        # facilities in this slice
        g = g.copy()
        facs = g[facility_col].tolist()

        # build per-fac arrays
        AMC = g.set_index(facility_col)[amc_col].astype(float)
        OB0 = g.set_index(facility_col)["OB"].astype(float)

        # donor surplus cap and receiver deficit
        surplus_cap = alpha * np.maximum(0.0, OB0.values - tau_keep * AMC.values)
        deficit     = np.maximum(0.0, tau_tar * AMC.values - OB0.values)
        donors_mask = surplus_cap > 1e-9 # boolean vector marking facilities that actually have positive surplus
        recv_mask   = deficit > 1e-9 # boolean vector marking facilities that actually have a real need (deficit > 0).

        if not donors_mask.any() or not recv_mask.any():
            # nothing to do for this slice
            continue

        donors = list(AMC.index[donors_mask])
        receivers = list(AMC.index[recv_mask])

        # Feasible edges within radius
        #feasible = build_edges_within_radius(time_matrix.loc[donors + receivers, donors + receivers], radius_minutes)
        T_d = time_matrix[d]  # time_matrix is dict {district -> DataFrame}

        # 2) limit to only facilities present in this slice
        slice_facs = g[facility_col].unique().tolist()
        T_d = T_d.loc[slice_facs, slice_facs]

        # 3) build feasible edges for this radius
        feasible = {}
        for g_fac in T_d.index:
            feas_mask = (T_d.loc[g_fac].to_numpy() <= radius_minutes) & np.isfinite(T_d.loc[g_fac].to_numpy())
            feasible[g_fac] = set(T_d.columns[feas_mask]) - {g_fac}

        # Minimum lot (units) for each receiver depends on its AMC
        AMC_map = AMC.to_dict()
        Qmin_units = {f: Qmin_days * max(1e-9, AMC_map[f]) for f in receivers}

        # Big-M per edge: cannot exceed receiver deficit
        deficit_map = dict(zip(AMC.index, deficit)) # creates a dict with key = facility and value = deficit
        # this is the upper bound for how much a facility can receive
        smax_map = dict(zip(AMC.index, surplus_cap)) # creates a dict with key = facility and value = donatable surplus
        # this is the upper bound for how much a facility can donate

        # Build MILP
        prob = LpProblem(f"Redistribution_{d}_{m}_{i}", LpMaximize)

        # Variables
        t = {}   # transfer amounts per (g,f): continuous >=0
        y = {}   # edge-activation binaries

        for g_fac in donors:
            for f_fac in feasible.get(g_fac, set()):
                if f_fac in receivers:
                    t[(g_fac, f_fac)] = LpVariable(f"t_{g_fac}->{f_fac}", lowBound=0, cat=LpContinuous)
                    y[(g_fac, f_fac)] = LpVariable(f"y_{g_fac}->{f_fac}", lowBound=0, upBound=1, cat=LpBinary)

        # Objective: maximize total shipped (sum of t)
        prob += lpSum(t.values())

        # Donor outflow caps
        for g_fac in donors:
            prob += lpSum(t[(g_fac, f_fac)] for f_fac in receivers if (g_fac, f_fac) in t) <= smax_map[g_fac]

        # Receiver inflow caps
        for f_fac in receivers:
            prob += lpSum(t[(g_fac, f_fac)] for g_fac in donors if (g_fac, f_fac) in t) <= deficit_map[f_fac]

        # Edge linking + minimum lot size + K_in/K_out
        for (g_fac, f_fac), var in t.items():
            M = deficit_map[f_fac]
            prob += var <= M * y[(g_fac, f_fac)]
            # minimum lot only makes sense if the edge is used at all; also don't force Qmin to exceed deficit
            prob += var >= min(Qmin_units[f_fac], deficit_map[f_fac]) * y[(g_fac, f_fac)]

        # Max donors per receiver
        for f_fac in receivers:
            prob += lpSum(y[(g_fac, f_fac)] for g_fac in donors if (g_fac, f_fac) in y) <= K_in

        # Max receivers per donor
        for g_fac in donors:
            prob += lpSum(y[(g_fac, f_fac)] for f_fac in receivers if (g_fac, f_fac) in y) <= K_out

        # Solve (default solver)
        prob.solve()

        # Apply transfers to OB'
        if LpStatus[prob.status] not in ("Optimal","Optimal Infeasible","Not Solved"):
            # If not solved (rare), skip changes
            continue

        # net change per facility
        delta = {fac: 0.0 for fac in facs}
        for (g_fac, f_fac), var in t.items():
            moved = value(var) or 0.0
            delta[g_fac] -= moved
            delta[f_fac] += moved

        # write back OB'
        sel = (df["district"].eq(d) & df["month"].eq(m) & df["item_code"].eq(i))
        df.loc[sel, "OB_prime"] = df.loc[sel].apply(
            lambda r: r["OB"] + delta.get(r[facility_col], 0.0), axis=1
        )

    # Mechanistic availability after redistribution
    df["available_prop_redis"] = np.minimum(1.0, np.maximum(0.0, df["OB_prime"] / np.maximum(1e-9, df[amc_col])))
    return df

def redistribute_pooling_lp(
    df: pd.DataFrame,
    tau_min: float = 0.25,   # optional lower floor (this is the same as 7 days Q_min)
    tau_max: float = 3.0,    # storage/USAID style max
    id_cols=("district","month","item_code"),
    facility_col="fac_name",
    amc_col="amc",
    close_cols=("closing_bal","received","dispensed"),
) -> pd.DataFrame:
    """
    Scenario 3: district-level pooling/push. Linear program maximizing total presumed availability.

    We linearize p = min(1, x/AMC) by introducing p_f in [0,1] with constraint AMC_f * p_f <= x_f.

    Returns df with 'available_prop_redis' updated for the pooling scenario.
    """
    closing_bal, received, dispensed = close_cols
    df = df.copy()
    df["OB"] = compute_opening_balance(df)

    df["OB_prime"] = df["OB"]

    group_cols = list(id_cols)
    for (d, m, i), g in df.groupby(group_cols):
        g = g.copy()
        facs = g[facility_col].tolist()
        AMC = g.set_index(facility_col)[amc_col].astype(float)
        OB0 = g.set_index(facility_col)["OB"].astype(float)

        total_stock = float(OB0.sum())
        if total_stock <= 1e-9:
            # nothing to allocate
            continue

        # LP
        prob = LpProblem(f"Pooling_{d}_{m}_{i}", LpMaximize)

        # Decision vars
        x = {f: LpVariable(f"x_{f}", lowBound=0) for f in facs}      # allocated opening stock
        p = {f: LpVariable(f"p_{f}", lowBound=0, upBound=1) for f in facs}  # availability

        # Objective: maximize sum p_f (optionally weight by item importance)
        prob += lpSum(p.values())

        # Conservation
        prob += lpSum(x.values()) == total_stock

        # Guardrails (optional)
        for f in facs:
            prob += x[f] >= tau_min * AMC[f]
            prob += x[f] <= tau_max * AMC[f]

        # Linearization: AMC_f * p_f <= x_f
        for f in facs:
            prob += AMC[f] * p[f] <= x[f]

        prob.solve()

        if LpStatus[prob.status] not in ("Optimal","Optimal Infeasible","Not Solved"):
            continue

        # Write back OB' and p'
        x_sol = {f: value(var) or 0.0 for f, var in x.items()}
        # OB' equals allocated x
        sel = (df["district"].eq(d) & df["month"].eq(m) & df["item_code"].eq(i))
        df.loc[sel, "OB_prime"] = df.loc[sel].apply(lambda r: x_sol.get(r[facility_col], r["OB"]), axis=1)

    # Mechanistic availability p' = min(1, OB'/AMC)
    df["available_prop_redis"] = np.minimum(1.0, np.maximum(0.0, df["OB_prime"] / np.maximum(1e-9, df[amc_col])))
    return df

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

# 2) Run Scenario 1 (1-hour radius)

#Drop NAs
# TODO find a more generalisable solution for this issue (within the optimisation functions)
lmis = lmis[(lmis.amc != 0) & (lmis.amc.notna())]

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


comparison_df = tlo_30.merge(tlo_availability_df, left_on = ['district', 'Facility_Level', 'month', 'item_code'],
                             right_on = ['District', 'Facility_Level', 'month', 'item_code'], how = 'left', validate = "1:1")
comparison_df = comparison_df.merge(tlo_60, on = ['district', 'Facility_Level', 'month', 'item_code'], how = 'left', validate = "1:1")
print(comparison_df['new_available_prop_x'].mean(),comparison_df['new_available_prop_y'].mean(),
comparison_df['available_prop'].mean())

comparison_df['new_available_prop_x'] = np.maximum(
    comparison_df['new_available_prop_x'],
    comparison_df['available_prop']
)
comparison_df['new_available_prop_y'] = np.maximum(
    comparison_df['new_available_prop_y'],
    comparison_df['available_prop']
)

print(comparison_df['new_available_prop_x'].mean(),comparison_df['new_available_prop_y'].mean(),
comparison_df['available_prop'].mean())

# TODO keep only government facilities?

