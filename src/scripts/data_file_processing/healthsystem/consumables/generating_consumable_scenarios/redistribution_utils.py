"""
Utility functions for the consumable redistribution scenario analysis.

This module contains the building blocks used by `create_consumable_redistribution_scenarios.py`:

1. Travel-time matrix construction (OSRM / OpenRouteService)
2. Facility clustering for neighbourhood pools (capacity-constrained k-medoids heuristic)
3. The two redistribution optimisation models:
   - `redistribute_pooling_lp`  : proactive, centrally coordinated pooling (pure LP)
   - `redistribute_radius_lp`   : reactive, radius-limited pairwise exchange (MILP)
4. Validation checks (`validate_redistribution_output`) enforcing the models' key invariants:
   - availability of non-eligible facility levels is never changed,
   - post-redistribution availability is never below baseline (no-harm),
   - stock is conserved within each redistribution group,
   - donors are never drawn below their protection floor.
5. Self-contained smoke tests on synthetic data (`run_smoke_tests`)
6. Figures summarising how much of the anticipated stock-out risk is averted by redistribution.

Model notes (updated formulation)
---------------------------------
Pooling LP: The lower bound is ensures  donor protection, LB = min(OB, tau_keep * AMC).
Only facilities at levels 1a/1b may receive stock above their opening balance (eligibility-aware UB).

Pairwise MILP: the minimum-lot size is receiver-specific (Qmin_proportion * AMC_receiver) and edges
that cannot carry at least this quantity are pruned before optimisation.
The binary edge-activation variables y and the coupling t <= M*y support optional per-item
exchange-count caps (K_in / K_out), but these are not imposed by default: the travel-time
radius already bounds the candidate neighbour set and Qmin already screens out shipments too
small to be worth dispatching, so no separate degree cap is needed to keep solutions sensible.
K_in/K_out remain available as parameters for a stricter delivery-capacity sensitivity check.
"""
import textwrap
from math import ceil
from pathlib import Path
from typing import Dict, Iterable, Literal, Optional, Tuple

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from pulp import (
    PULP_CBC_CMD,
    LpBinary,
    LpContinuous,
    LpMaximize,
    LpProblem,
    LpStatus,
    LpVariable,
    lpSum,
    value,
)

ELIGIBLE_LEVELS = ("1a", "1b")


# ======================================================================================
# 1) Travel-time matrices
# ======================================================================================
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
    ors_api_key: Optional[str] = None,
    ors_base_url: str = "https://api.openrouteservice.org/v2/matrix",
    osrm_base_url: str = "https://router.project-osrm.org",
    osrm_profile_map: dict = None,
    max_chunk: int = 40,
    timeout: int = 60,
) -> pd.DataFrame:
    """
    Build an NxN *road* travel-time matrix (minutes) for facilities, by CAR or BICYCLE.

    backends:
      - 'ors'  -> OpenRouteService Matrix API (requires ors_api_key; auto-chunked).
      - 'osrm' -> OSRM 'table' service (public server generally supports 'car' only).

    Returns a square DataFrame (minutes) with index/columns = facility ids; unroutable pairs = inf.
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
        coords = [[float(lons[i]), float(lats[i])] for i in range(n)]  # ORS expects [lon, lat]
        headers = {"Authorization": ors_api_key, "Content-Type": "application/json"}

        for si, sj in _chunk_indices(n, max_chunk):
            for di, dj in _chunk_indices(n, max_chunk):
                body = {
                    "locations": coords,
                    "sources": list(range(si, sj)),
                    "destinations": list(range(di, dj)),
                    "metrics": ["duration"],
                }
                r = requests.post(f"{ors_base_url}/{profile}", json=body, headers=headers, timeout=timeout)
                r.raise_for_status()
                durs = r.json().get("durations")
                if durs is None:
                    raise RuntimeError(f"ORS returned no durations for block {si}:{sj} x {di}:{dj}")
                T.iloc[si:sj, di:dj] = np.array(durs, dtype=float) / 60.0  # minutes

    elif backend == "osrm":
        if osrm_profile_map is None:
            osrm_profile_map = {"car": "car", "bicycle": "bike"}
        profile = osrm_profile_map.get(mode)
        if profile is None:
            raise ValueError(f"No OSRM profile mapped for mode='{mode}'.")

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
                durs = r.json().get("durations")
                if durs is None:
                    raise RuntimeError(f"OSRM returned no durations for block {si}:{sj} x {di}:{dj}")
                T.iloc[si:sj, di:dj] = np.array(durs, dtype=float) / 60.0  # minutes

    else:
        raise ValueError("backend must be 'ors' or 'osrm'.")

    return T.fillna(np.inf)


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
    max_chunk: int = 50,
) -> dict[str, pd.DataFrame]:
    """Return {district -> square minutes matrix DataFrame}, computed within each district only."""
    matrices = {}
    fac_coords = df[[district_col, id_col, lat_col, lon_col]].dropna().drop_duplicates()

    for d, facs_d in fac_coords.groupby(district_col):
        if len(facs_d) < 2:
            continue
        matrices[d] = build_travel_time_matrix(
            fac_df=facs_d[[id_col, lat_col, lon_col]],
            id_col=id_col, lat_col=lat_col, lon_col=lon_col,
            mode=mode, backend=backend,
            osrm_base_url=osrm_base_url, ors_api_key=ors_api_key, max_chunk=max_chunk,
        )
    return matrices


def build_edges_within_radius_flat(T_by_dist: dict, max_minutes: float) -> dict[str, set[str]]:
    """
    Flatten district-wise travel-time matrices into {facility -> set(neighbours within max_minutes)}.
    """
    edges: dict[str, set[str]] = {}
    for _, T in T_by_dist.items():
        for g in T.index:
            row = T.loc[g].to_numpy()
            feasible_mask = (row <= max_minutes) & np.isfinite(row)
            feasible = [f for f in T.columns[feasible_mask] if f != g]
            edges.setdefault(g, set()).update(feasible)
    return edges


# ======================================================================================
# 2) Facility clustering for neighbourhood pools
# ======================================================================================
def _farthest_first_seeds(T: pd.DataFrame, k: int, big: float = 1e9) -> list:
    """
    Pick k seed medoids via farthest-first traversal on a travel-time matrix.
    Treat inf/NaN distances as 'big' so disconnected components get separate seeds.
    """
    n = T.shape[0]
    facs = T.index.tolist()
    D = T.to_numpy().astype(float)
    D[~np.isfinite(D)] = big

    start = int(np.nanargmax(np.nanmean(D, axis=1)))  # the remotest facility
    seeds_idx = [start]

    for _ in range(1, k):
        min_to_seed = np.min(D[:, seeds_idx], axis=1)
        next_idx = int(np.argmax(min_to_seed))
        if next_idx in seeds_idx:
            candidates = [i for i in range(n) if i not in seeds_idx]
            if not candidates:
                break
            next_idx = int(candidates[np.argmax(min_to_seed[candidates])])
        seeds_idx.append(next_idx)

    return [facs[i] for i in seeds_idx]


def _assign_to_cluster_with_fixed_capacity(
    T: pd.DataFrame, seeds: list, capacity: int, big: float = 1e9
) -> Dict[str, int]:
    """Greedy assignment of facilities to nearest seed that still has capacity."""
    facs = T.index.tolist()
    D = T.loc[facs, seeds].to_numpy().astype(float)
    D[~np.isfinite(D)] = big

    nearest = D.min(axis=1)
    order = np.argsort(nearest)

    cap_left = {j: capacity for j in range(len(seeds))}
    assign = {}

    for idx in order:
        f = facs[idx]
        placed = False
        for j in np.argsort(D[idx, :]):
            if cap_left[j] > 0:
                assign[f] = j
                cap_left[j] -= 1
                placed = True
                break
        if not placed:
            j = min(cap_left, key=lambda jj: cap_left[jj])
            assign[f] = j
            cap_left[j] -= 1

    return assign


def capacity_clusters_for_district(T_d: pd.DataFrame, cluster_size: int = 3, big: float = 1e9) -> Dict[str, str]:
    """
    Build ~equal-size clusters (size <= cluster_size) from a district's travel-time matrix via
    capacity-constrained k-medoids (farthest-first seeds + greedy capacity assignment).
    Returns {facility_id -> cluster_id} with ids like 'C00', 'C01', ...
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
    seed_to_cid = {j: f"C{j:02d}" for j in range(len(seeds))}
    return {f: seed_to_cid[assign_seed_idx[f]] for f in facs}


def build_capacity_clusters_all(T_by_dist: Dict[str, pd.DataFrame], cluster_size: int = 3) -> pd.Series:
    """
    Apply capacity clustering to all districts.

    Returns a pd.Series mapping (district, facility_id) -> district-scoped cluster_id
    (e.g. 'Nkhotakota#C03'), with a 2-level MultiIndex ["district", "fac_name"].

    The index is deliberately keyed on (district, facility) rather than facility alone:
    facility names are not guaranteed to be globally unique across districts, and a
    facility-only index previously caused a many-to-many merge (row-count blow-up, and
    with it stock-conservation/ceiling violations) in `redistribute_pooling_lp` whenever a
    facility name happened to recur in more than one district.
    """
    mappings = []
    for d, T_d in T_by_dist.items():
        if T_d is None or T_d.empty:
            continue
        local_map = capacity_clusters_for_district(T_d, cluster_size=cluster_size)
        if not local_map:
            continue
        s = pd.Series(local_map, name="cluster_id").map(lambda cid: f"{d}#{cid}")
        s.index = pd.MultiIndex.from_product([[d], s.index], names=["district", "fac_name"])
        mappings.append(s)
    if not mappings:
        return pd.Series(dtype=object)
    return pd.concat(mappings)


# ======================================================================================
# 3a) Pooling optimisation (pure LP)
# ======================================================================================
def redistribute_pooling_lp(
    df: pd.DataFrame,
    tau_max: float = 3.0,          # upper ceiling (storage/policy max), multiple of AMC
    tau_donor_keep: float = 1.5,   # minimum the donor keeps before donating, multiple of AMC
    id_cols=("district", "month", "item_code"),
    facility_col="fac_name",
    level_col="Facility_Level",
    amc_col="amc",
    eligible_levels: Iterable[str] = ELIGIBLE_LEVELS,
    amc_eps: float = 1e-6,
    return_move_log: bool = True,
    pooling_level: str = "district",       # "district", "cluster", or "national"
    cluster_map: pd.Series | None = None,  # required if pooling_level == "cluster"
    floor_to_baseline: bool = True,
):
    """
    Proactive pooled redistribution (national-, district-, or cluster-level), solved as a
    pure LP per (pool, month, item):

        max  sum_f p_f
        s.t. sum_f x_f + excess = TotalStock
             LB_f <= x_f <= UB_f
             AMC_f * p_f <= x_f
             0 <= p_f <= 1

    with LB_f = min(OB_f, tau_donor_keep * AMC_f)          [donor protection only]
         UB_f = tau_max * AMC_f            if level in eligible_levels
              = min(OB_f, tau_max * AMC_f) otherwise       [eligibility-aware ceiling]

    Because tau_donor_keep < tau_max, LB <= UB holds by construction for every facility, and
    because each LB_f <= OB_f, sum(LB) <= TotalStock always holds; both invariants are asserted.

    `pooling_level` controls the pool boundary only; the LP formulation is identical:
      - "district"  : pool per (district, month, item)                 [default]
      - "cluster"   : pool per (cluster_id, month, item); requires `cluster_map`
      - "national"  : pool per (month, item) across the whole country

    If a facility appears more than once within a single pool (e.g. duplicate raw records
    for the same district/item/month/facility), its solved balance x_sol is split across
    those rows in proportion to each row's original opening balance (equal split if the
    facility's total original balance was zero) -- this keeps stock conservation exact
    regardless of how many raw rows a facility contributes to a pool.

    Returns (out, move_log) if return_move_log else out, where `out` carries columns
    OB, OB_prime, available_prop_redis and received_from_pool.
    """
    if pooling_level not in ("district", "cluster", "national"):
        raise ValueError("pooling_level must be 'district', 'cluster', or 'national'.")
    if tau_donor_keep > tau_max:
        raise ValueError("tau_donor_keep must not exceed tau_max (LB <= UB requires it).")

    out = df.copy()
    out["OB"] = out["opening_bal"]
    out["OB_prime"] = out["OB"]

    if pooling_level == "cluster":
        if cluster_map is None:
            raise ValueError("cluster_map is required when pooling_level='cluster'.")
        cmap = cluster_map.rename("cluster_id")
        if isinstance(cmap.index, pd.MultiIndex):
            # Preferred: (district, fac_name) -> cluster_id, as returned by
            # build_capacity_clusters_all(). Merging on both keys avoids a many-to-many
            # fan-out when a facility name recurs in more than one district.
            out = out.merge(cmap, how="left", left_on=["district", facility_col], right_index=True)
        else:
            # Backward-compatible fallback for a plain fac_name-indexed Series. Only safe if
            # facility names are unique across the whole dataset (not just within a district).
            out = out.merge(cmap, how="left", left_on=facility_col, right_index=True)
        if out["cluster_id"].isna().any():
            # facilities missing a cluster - assign singleton clusters to keep them
            out["cluster_id"] = out["cluster_id"].fillna(
                out["district"].astype(str) + "#CXX_" + out[facility_col].astype(str))

    if pooling_level == "district":
        group_cols, node_label = list(id_cols), "district"
    elif pooling_level == "cluster":
        group_cols, node_label = ["cluster_id", "month", "item_code"], "cluster_id"
    else:  # national
        group_cols, node_label = ["month", "item_code"], "national"

    # Internal node key used to identify a facility WITHIN a pool. For district/cluster pooling,
    # `district` (or the district-prefixed cluster_id) is already part of group_cols, so facility
    # names only need to be unique within a district to correctly identify a node -- true even if
    # the same name recurs in a different district. National pooling drops district from the pool
    # boundary entirely, so facility_col alone is no longer a safe identity key: two different real
    # facilities that happen to share a name would otherwise be silently treated as one LP node
    # (their stock summed, and only one of their AMC/Facility_Level values used). Disambiguate with
    # a district-qualified key in that case; `facility_col` itself is left untouched for output.
    if pooling_level == "national":
        out["_node_key"] = out["district"].astype(str) + "||" + out[facility_col].astype(str)
    else:
        out["_node_key"] = out[facility_col]

    # Diagnostic: how often does a facility contribute more than one raw row to a single pool
    # (same node, same pool)? This is handled correctly below (solved balance split
    # proportionally across the duplicate rows), but a large count usually indicates an
    # upstream data-deduplication issue worth investigating (e.g. in the LMIS
    # collapse-duplicates step) -- for national pooling, genuine duplicates are counted after
    # disambiguating by district, so a facility-name collision across two districts does NOT
    # show up here (it's handled as two distinct nodes, not flagged as a duplicate row).
    dup_counts = out.groupby(group_cols + ["_node_key"], dropna=False).size()
    n_dup = int((dup_counts > 1).sum())
    if n_dup:
        print(f"Pooling ({pooling_level}): {n_dup:,} (pool, facility) combinations have more "
              f"than one row in the input data; each facility's solved allocation will be "
              f"split across its rows in proportion to their original opening balance.")

    move_rows = []
    skipped_nodes = []
    eligible_levels = set(eligible_levels)

    for keys, g in out.groupby(group_cols, sort=False):
        # Resolve node ID for logging and selection masks
        if pooling_level == "district":
            node_val, m, i = g["district"].iloc[0], keys[1], keys[2]
        elif pooling_level == "national":
            node_val, m, i = "National", keys[0], keys[1]
        else:
            node_val, m, i = keys

        AMC = (g.set_index("_node_key")[amc_col]
               .astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0))
        OB0 = (g.set_index("_node_key")["OB"]
               .astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0))
        LVL = g.set_index("_node_key")[level_col].astype(str)
        # node -> (district, facility) lookup, for human-readable move-log output only
        NODE_INFO = g.drop_duplicates("_node_key").set_index("_node_key")[["district", facility_col]]

        # collapse duplicates if any (true same-node duplicate raw rows only -- see _node_key)
        if AMC.index.duplicated().any():
            AMC = AMC[~AMC.index.duplicated(keep="first")]
        if LVL.index.duplicated().any():
            LVL = LVL[~LVL.index.duplicated(keep="first")]
        if OB0.index.duplicated().any():
            OB0 = OB0.groupby(level=0).sum()

        total_stock = float(OB0.sum())
        if total_stock <= 1e-9:
            continue

        # Participants (positive expected demand)
        mask_pos = AMC >= amc_eps
        facs_pos = AMC.index[mask_pos].tolist()
        if not facs_pos:
            continue

        AMC_pos = AMC.loc[facs_pos]
        OB0_pos = OB0.loc[facs_pos]
        LVL_pos = LVL.reindex(facs_pos)

        # ---- Bounds (updated formulation: donor protection only; no equity floor) ----
        LB = np.minimum(OB0_pos, tau_donor_keep * AMC_pos).astype(float)
        UB = (tau_max * AMC_pos).astype(float)
        non_elig_mask = ~LVL_pos.isin(eligible_levels)
        UB.loc[non_elig_mask] = np.minimum(OB0_pos.loc[non_elig_mask], UB.loc[non_elig_mask])

        # Structural invariants (would previously have required clipping/scaling)
        assert (LB.values <= UB.values + 1e-9).all(), \
            f"LB > UB in pool {node_val}/{m}/{i}: formulation invariant violated"
        assert LB.sum() <= total_stock + 1e-6, \
            f"sum(LB) > TotalStock in pool {node_val}/{m}/{i}: formulation invariant violated"

        # Excess sink active only if ceilings bind
        allow_excess_sink = total_stock > float(UB.sum()) + 1e-9

        # ---------- LP ----------
        prob = LpProblem(f"Pooling_{node_val}_{m}_{i}", LpMaximize)
        x = {f: LpVariable(f"x_{f}", lowBound=0) for f in facs_pos}
        p = {f: LpVariable(f"p_{f}", lowBound=0, upBound=1) for f in facs_pos}
        excess = LpVariable("excess", lowBound=0) if allow_excess_sink else None
        # note: facilities with AMC ~ 0 are excluded from optimisation but their positive OB is
        # included in total_stock (they donate to the pool)

        prob += lpSum(p.values())

        if excess is None:
            prob += lpSum(x.values()) == total_stock
        else:
            prob += lpSum(x.values()) + excess == total_stock

        for f in facs_pos:
            prob += x[f] >= float(LB.loc[f])
            prob += x[f] <= float(UB.loc[f])
            prob += float(max(AMC_pos.loc[f], amc_eps)) * p[f] <= x[f]

        prob.solve(PULP_CBC_CMD(msg=False, cuts=0, presolve=True, threads=1))
        if LpStatus[prob.status] != "Optimal":
            skipped_nodes.append((node_val, m, i))
            continue

        x_sol = {f: float(value(var) or 0.0) for f, var in x.items()}

        if pooling_level == "district":
            sel = (out["district"].eq(node_val) & out["month"].eq(m) & out["item_code"].eq(i))
        elif pooling_level == "national":
            sel = (out["month"].eq(m) & out["item_code"].eq(i))
        else:
            sel = (out["cluster_id"].eq(node_val) & out["month"].eq(m) & out["item_code"].eq(i))

        # Nodes with AMC >= eps get x_f, split across any duplicate raw rows for that node in
        # proportion to each row's original opening balance (equal split if the node's total
        # original OB in this pool was zero) -- this keeps the written-out total exactly equal
        # to x_sol[f] regardless of how many rows the node occupies.
        pos_rows = out.loc[sel & out["_node_key"].isin(facs_pos), ["_node_key", "OB"]]
        if not pos_rows.empty:
            node_total_ob = pos_rows.groupby("_node_key")["OB"].transform("sum")
            node_row_count = pos_rows.groupby("_node_key")["OB"].transform("count")
            safe_total_ob = node_total_ob.where(node_total_ob > 1e-9, other=1.0)  # avoid 0/0
            share = np.where(node_total_ob > 1e-9, pos_rows["OB"] / safe_total_ob, 1.0 / node_row_count)
            x_target = pos_rows["_node_key"].map(x_sol).to_numpy(dtype=float)
            out.loc[pos_rows.index, "OB_prime"] = share * x_target

        # Nodes with AMC < eps donate entirely (their stock went into total_stock)
        mask_rows_zero = sel & ~out["_node_key"].isin(facs_pos)
        out.loc[mask_rows_zero, "OB_prime"] = 0.0

        if return_move_log:
            for nk in AMC.index:
                x_f = x_sol.get(nk, 0.0) if nk in facs_pos else 0.0
                info = NODE_INFO.loc[nk] if nk in NODE_INFO.index else None
                move_rows.append({
                    node_label: node_val, "month": m, "item_code": i,
                    "district": info["district"] if info is not None else node_val,
                    "facility": info[facility_col] if info is not None else nk,
                    "received_from_pool": x_f - float(OB0.get(nk, 0.0)),
                    "x_allocated": x_f,
                    "OB0_agg": float(OB0.get(nk, 0.0)),
                    "eligible_receiver": bool(LVL.get(nk, "") in eligible_levels),
                })

    print(f"Pooling: skipped {len(skipped_nodes)} {node_label}-month-item combinations (no optimal solution)")

    # Snap solver numerical wobble back to the original balance (CBC returns values within its
    # own feasibility tolerance, so x can differ from OB by ~1e-6 relative even when no stock moved)
    wobble = (out["OB_prime"] - out["OB"]).abs() <= 1e-6 * np.maximum(1.0, out["OB"].abs())
    out.loc[wobble, "OB_prime"] = out.loc[wobble, "OB"]

    # --- Availability after redistribution: update ONLY where OB' increased ---
    amc_safe_all = out[amc_col].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    denom = np.maximum(amc_eps, amc_safe_all.values)

    out["available_prop_redis"] = out["available_prop"].astype(float).values
    changed = (out["OB_prime"] - out["OB"]) > 1e-6
    p_new = np.minimum(1.0, np.maximum(0.0, out.loc[changed, "OB_prime"].values / denom[changed]))
    if floor_to_baseline:
        p_new = np.maximum(p_new, out.loc[changed, "available_prop"].astype(float).values)
    out.loc[changed, "available_prop_redis"] = p_new

    out["received_from_pool"] = out["OB_prime"] - out["OB"]

    # Structural guarantee: non-eligible levels can never gain stock (UB <= OB), so their
    # availability is never recomputed. Assert rather than silently overwrite.
    non_elig_rows = ~out[level_col].isin(eligible_levels)
    assert np.allclose(
        out.loc[non_elig_rows, "available_prop_redis"].astype(float),
        out.loc[non_elig_rows, "available_prop"].astype(float),
        equal_nan=True,
    ), "Availability changed for non-eligible facility levels: eligibility constraint violated"

    if return_move_log:
        return out, pd.DataFrame(move_rows)
    return out


# ======================================================================================
# 3b) Pairwise (radius) redistribution (MILP)
# ======================================================================================
def redistribute_radius_lp(
    df: pd.DataFrame,
    time_matrix: Dict[str, pd.DataFrame] | pd.DataFrame,
    radius_minutes: float,
    tau_keep: float = 1.5,          # donors must keep >= tau_keep * AMC
    tau_tar: float = 1.0,           # receivers target OB = tau_tar * AMC
    K_in: Optional[int] = None,     # per-item: max distinct donors per receiver (None = no cap)
    K_out: Optional[int] = None,    # per-item: max distinct receivers per donor (None = no cap)
    Qmin_proportion: float = 0.25,  # min lot as a fraction of receiver AMC (~1 week)
    eligible_levels: Iterable[str] = ELIGIBLE_LEVELS,
    id_cols=("district", "month", "item_code"),
    facility_col="fac_name",
    level_col="Facility_Level",
    amc_col="amc",
    return_edge_log: bool = True,
    floor_to_baseline: bool = True,
    amc_eps: float = 1e-6,
    eps: float = 1e-9,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Reactive pairwise redistribution, solved as a MILP per (district, month, item):

      variables: t[g,f] >= 0 (transfer), y[g,f] in {0,1} (edge activation), p[f] in [0,1]
      objective: maximize sum of p over eligible receivers
      key constraints:
        - donors keep >= tau_keep * AMC                     (outflow cap)
        - receivers limited to deficit tau_tar*AMC - OB     (inflow cap; eligibility pre-filtered)
        - travel time <= radius, edge capacity >= Qmin      (edges pruned before optimisation)
        - t <= M * y  and  t >= Qmin_f * y                  (activation coupling + minimum lot)
        - sum_f y <= K_out per donor; sum_g y <= K_in per receiver, only if K_out/K_in given

    The number of counterparties a facility ends up trading with is not capped by default:
    the travel-time radius already bounds the candidate neighbour set, and Qmin_proportion
    already screens out shipments too small to be worth dispatching, so an explicit degree
    cap is not needed to keep the solution logistically sensible. K_in/K_out are kept as
    optional parameters (default None -> unconstrained) so a stricter delivery-capacity
    scenario can still be run as a sensitivity check without changing the model structure.

    Availability is recomputed mechanistically and written back only where stock increased.
    """
    eligible_levels = set(eligible_levels)
    out = df.copy()
    out["OB"] = out["opening_bal"]
    selected_cols = list(id_cols) + [level_col, facility_col, "OB", amc_col, "available_prop"]
    out = out[selected_cols]

    out[amc_col] = pd.to_numeric(out[amc_col], errors="coerce").fillna(0.0)
    out[level_col] = out[level_col].astype(str)
    out["OB_prime"] = out["OB"]

    edge_rows = [] if return_edge_log else None
    skipped_nodes = []

    for (d, m, i), g in out.groupby(list(id_cols), sort=False):
        # --- Pick the travel-time matrix slice ---
        if isinstance(time_matrix, dict):
            T_d = time_matrix.get(d)
            if T_d is None or T_d.empty:
                continue
        else:
            T_d = time_matrix

        facs_slice = g[facility_col].dropna().unique().tolist()
        facs = [f for f in facs_slice if f in T_d.index and f in T_d.columns]
        if len(facs) < 2:
            continue

        T_sub = T_d.loc[facs, facs].replace(np.nan, np.inf)

        AMC = (g.set_index(facility_col)[amc_col].astype(float)
               .replace([np.inf, -np.inf], np.nan).fillna(0.0)).reindex(facs).fillna(0.0)
        OB0 = (g.set_index(facility_col)["OB"].astype(float)
               .replace([np.inf, -np.inf], np.nan).fillna(0.0)).reindex(facs).fillna(0.0)
        LVL = g.set_index(facility_col)[level_col].astype(str).reindex(facs)

        AMC_guard = AMC.copy()
        AMC_guard[AMC_guard <= 0.0] = amc_eps

        # --- Surplus / deficit ---
        surplus = np.maximum(0.0, OB0.values - tau_keep * AMC_guard.values)   # donors
        deficit = np.maximum(0.0, tau_tar * AMC_guard.values - OB0.values)    # receivers
        # Only eligible levels can receive
        deficit = np.where(LVL.isin(eligible_levels).values, deficit, 0.0)

        donors = [f for f, s in zip(facs, surplus) if s > eps]
        recvs = [f for f, h in zip(facs, deficit) if h > eps]
        if not donors or not recvs:
            continue

        s_map = dict(zip(facs, surplus))
        h_map = dict(zip(facs, deficit))
        qmin_map = dict(zip(facs, Qmin_proportion * AMC_guard.values))

        # --- Feasible edges (within radius); prune edges that cannot carry the minimum lot ---
        M_edge, Qmin = {}, {}
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
                    continue  # edge cannot carry a worthwhile shipment -> pruned
                M_edge[(g_fac, f_fac)] = float(M)
                Qmin[(g_fac, f_fac)] = qmin

        if not M_edge:
            continue

        # --- MILP (per item) ---
        prob = LpProblem(f"Radius_{d}_{m}_{i}", LpMaximize)
        t = {e: LpVariable(f"t_{e[0]}->{e[1]}", lowBound=0, upBound=M_edge[e], cat=LpContinuous)
             for e in M_edge}
        y = {e: LpVariable(f"y_{e[0]}->{e[1]}", cat=LpBinary) for e in M_edge}
        p = {f: LpVariable(f"p_{f}", lowBound=0, upBound=1) for f in facs}

        prob += lpSum(p[f] for f in recvs)

        # donor outflow caps
        for g_fac in donors:
            vars_out = [t[(gg, ff)] for (gg, ff) in t if gg == g_fac]
            if vars_out:
                prob += lpSum(vars_out) <= float(s_map[g_fac])

        # receiver inflow caps (eligibility already enforced via deficit)
        for f_fac in recvs:
            vars_in = [t[(gg, ff)] for (gg, ff) in t if ff == f_fac]
            if vars_in:
                prob += lpSum(vars_in) <= float(h_map[f_fac])

        # activation coupling + minimum lot
        # (edges with M < Qmin were pruned, so Qmin <= M and the pair is always consistent)
        for e in M_edge:
            prob += t[e] <= M_edge[e] * y[e]
            prob += t[e] >= Qmin[e] * y[e]

        # per-item degree caps: count distinct transfer events (only if explicitly requested;
        # by default connectivity is left to be bounded by radius + Qmin alone)
        if K_in is not None:
            for f_fac in recvs:
                inbound_y = [y[(gg, ff)] for (gg, ff) in y if ff == f_fac]
                if inbound_y:
                    prob += lpSum(inbound_y) <= K_in
        if K_out is not None:
            for g_fac in donors:
                outbound_y = [y[(gg, ff)] for (gg, ff) in y if gg == g_fac]
                if outbound_y:
                    prob += lpSum(outbound_y) <= K_out

        # availability linearisation per facility
        for f_fac in facs:
            inflow = lpSum(t[(gg, ff)] for (gg, ff) in t if ff == f_fac)
            outflow = lpSum(t[(gg, ff)] for (gg, ff) in t if gg == f_fac)
            prob += float(AMC_guard[f_fac]) * p[f_fac] <= float(OB0.get(f_fac, 0.0)) + inflow - outflow

        status = prob.solve(PULP_CBC_CMD(msg=False, cuts=0, presolve=True, threads=1))
        if LpStatus[prob.status] != "Optimal":
            skipped_nodes.append((d, m, i))
            continue

        # --- Apply transfers & log ---
        delta = {f: 0.0 for f in facs}
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
                        "units_moved": moved, "travel_minutes": tm,
                    })

        if not any_transfer:
            continue

        sel = (out["district"].eq(d) & out["month"].eq(m) & out["item_code"].eq(i))
        out.loc[sel, "OB_prime"] = out.loc[sel].apply(
            lambda r: r["OB"] + delta.get(r[facility_col], 0.0), axis=1
        )

    print(f"Pairwise: skipped {len(skipped_nodes)} district-month-item combinations (no optimal solution)")

    # Snap solver numerical wobble back to the original balance
    wobble = (out["OB_prime"] - out["OB"]).abs() <= 1e-6 * np.maximum(1.0, out["OB"].abs())
    out.loc[wobble, "OB_prime"] = out.loc[wobble, "OB"]

    # ---------- Availability: update ONLY where positive transfers happened ----------
    changed_mask = (out["OB_prime"] - out["OB"]) > 1e-6
    denom = np.maximum(amc_eps, out[amc_col].astype(float).values)
    p_mech = np.minimum(1.0, np.maximum(0.0, out["OB_prime"].values / denom))

    new_p = out["available_prop"].astype(float).values.copy()
    if floor_to_baseline:
        new_p[changed_mask] = np.maximum(p_mech[changed_mask], new_p[changed_mask])
    else:
        new_p[changed_mask] = p_mech[changed_mask]
    out["available_prop_redis"] = new_p

    # Structural guarantee: only eligible levels can gain stock, so only they can change.
    non_elig = ~out[level_col].isin(eligible_levels)
    assert np.allclose(
        out.loc[non_elig, "available_prop_redis"].astype(float),
        out.loc[non_elig, "available_prop"].astype(float),
        equal_nan=True,
    ), "Availability changed for non-eligible facility levels: eligibility constraint violated"

    edge_df = pd.DataFrame(edge_rows) if return_edge_log else None
    return out, edge_df


# ======================================================================================
# 4) Validation of redistribution outputs
# ======================================================================================
def validate_redistribution_output(
    out: pd.DataFrame,
    scenario_name: str = "",
    group_cols=("district", "month", "item_code"),
    level_col: str = "Facility_Level",
    eligible_levels: Iterable[str] = ELIGIBLE_LEVELS,
    tau_donor_keep: float = 1.5,
    tau_max: float | None = 3.0,
    amc_col: str = "amc",
    tol: float = 1e-4,
    conservation: Literal["exact", "leq"] = "leq",
    strict: bool = True,
) -> dict:
    """
    Check the key invariants of a redistribution output dataframe (columns required:
    OB, OB_prime, available_prop, available_prop_redis, Facility_Level, amc + group_cols).

    1. Eligibility: availability of non-eligible levels is unchanged.
    2. No-harm: available_prop_redis >= available_prop everywhere.
    3. Conservation: within each group, sum(OB_prime) == sum(OB) ("exact", pairwise) or
       sum(OB_prime) <= sum(OB) ("leq", pooling, where the excess sink may absorb stock).
    4. Donor protection: OB_prime >= min(OB, tau_donor_keep * AMC) - tol.
    5. Ceiling (if tau_max given): OB_prime <= UB + tol with the eligibility-aware UB.

    Returns a dict of results; raises AssertionError on failure if strict=True.
    """
    eligible_levels = set(eligible_levels)
    results = {}
    failures = []

    elig_mask = out[level_col].isin(eligible_levels)
    amc = pd.to_numeric(out[amc_col], errors="coerce").fillna(0.0)

    # 1. Eligibility: non-eligible availability unchanged
    non_elig = out[~elig_mask]
    diff = (non_elig["available_prop_redis"].astype(float)
            - non_elig["available_prop"].astype(float)).abs()
    n_bad = int((diff > tol).sum())
    results["non_eligible_p_unchanged"] = (n_bad == 0)
    if n_bad:
        failures.append(f"{n_bad} non-eligible rows have changed availability")

    # 2. No-harm: p' >= p
    harm = (out["available_prop_redis"].astype(float)
            < out["available_prop"].astype(float) - tol)
    results["no_harm"] = (int(harm.sum()) == 0)
    if harm.any():
        failures.append(f"{int(harm.sum())} rows have available_prop_redis < available_prop")

    # 3. Stock conservation per group
    grp = out.groupby(list(group_cols))[["OB", "OB_prime"]].sum()
    delta = grp["OB_prime"] - grp["OB"]
    # scale tolerance with group size (unit-level rounding accumulates)
    grp_tol = tol * np.maximum(1.0, grp["OB"].abs())
    if conservation == "exact":
        bad = (delta.abs() > grp_tol)
    else:  # "leq": stock may be lost to the excess sink but never created
        bad = (delta > grp_tol)
    results["stock_conserved"] = (int(bad.sum()) == 0)
    if bad.any():
        failures.append(
            f"{int(bad.sum())} groups violate stock conservation "
            f"(mode='{conservation}'); worst: {delta[bad].abs().max():.3f} units"
        )

    # 4. Donor protection
    floor = np.minimum(out["OB"].astype(float), tau_donor_keep * amc)
    below = out["OB_prime"].astype(float) < (floor - tol * np.maximum(1.0, floor))
    # facilities with AMC ~ 0 in solved pooling groups legitimately donate everything (floor=0)
    results["donor_protection"] = (int(below.sum()) == 0)
    if below.any():
        failures.append(f"{int(below.sum())} rows drawn below donor-protection floor")

    # 5. Ceiling
    if tau_max is not None:
        ub = np.where(elig_mask, tau_max * amc,
                      np.minimum(out["OB"].astype(float), tau_max * amc))
        above = out["OB_prime"].astype(float) > ub + tol * np.maximum(1.0, ub)
        results["ceiling_respected"] = (int(above.sum()) == 0)
        if above.any():
            failures.append(f"{int(above.sum())} rows exceed the eligibility-aware ceiling")

    results["all_passed"] = not failures
    label = f" [{scenario_name}]" if scenario_name else ""
    if failures:
        msg = f"Redistribution validation FAILED{label}:\n  - " + "\n  - ".join(failures)
        if strict:
            raise AssertionError(msg)
        print(msg)
    else:
        print(f"Redistribution validation passed{label}: "
              f"{', '.join(k for k, v in results.items() if v is True and k != 'all_passed')}")
    return results


# ======================================================================================
# 5) Smoke tests on synthetic data
# ======================================================================================
def _make_synthetic_lmis(seed: int = 0) -> pd.DataFrame:
    """
    One district, one item, one month, six facilities with known surpluses/deficits:
      - F_hosp  (level 2):  OB = 10 x AMC   -> large donor, must never gain
      - F_don1  (1a):       OB = 4  x AMC   -> donor
      - F_don2  (1b):       OB = 2  x AMC   -> small donor
      - F_short1(1a):       OB = 0.2x AMC   -> receiver
      - F_short2(1b):       OB = 0.5x AMC   -> receiver
      - F_zero  (1a):       OB = 0          -> receiver
    """
    rows = [
        # fac_name, level, amc, ob_mult
        ("F_hosp", "2", 100.0, 10.0),
        ("F_don1", "1a", 50.0, 4.0),
        ("F_don2", "1b", 80.0, 2.0),
        ("F_short1", "1a", 60.0, 0.2),
        ("F_short2", "1b", 40.0, 0.5),
        ("F_zero", "1a", 30.0, 0.0),
    ]
    df = pd.DataFrame(rows, columns=["fac_name", "Facility_Level", "amc", "ob_mult"])
    df["district"] = "TestDistrict"
    df["month"] = 1
    df["item_code"] = "42"
    df["opening_bal"] = df["amc"] * df["ob_mult"]
    df["available_prop"] = np.clip(df["ob_mult"], 0.0, 1.0) * 0.9  # baseline slightly below mechanistic
    return df.drop(columns="ob_mult")


def _make_synthetic_lmis_two_districts() -> pd.DataFrame:
    """
    Two districts for regression-testing national-level pooling:
      - District1: a single eligible receiver (F1_recv, 1a, OB = 0.1x AMC) with no local donor,
        so district-level pooling cannot change it (the pool's only member is itself).
      - District2: an eligible donor (F2_don, 1a, OB = 4x AMC) with ample surplus.
    National pooling should top up F1_recv using District2's surplus; district pooling should not.
    """
    rows = [
        ("F1_recv", "District1", "1a", 50.0, 0.1),
        ("F2_don", "District2", "1a", 50.0, 4.0),
    ]
    df = pd.DataFrame(rows, columns=["fac_name", "district", "Facility_Level", "amc", "ob_mult"])
    df["month"] = 1
    df["item_code"] = "42"
    df["opening_bal"] = df["amc"] * df["ob_mult"]
    df["available_prop"] = np.clip(df["ob_mult"], 0.0, 1.0) * 0.9
    return df.drop(columns="ob_mult")


def _make_synthetic_lmis_national_name_collision() -> pd.DataFrame:
    """
    Two districts sharing a facility name 'F_shared', but with DIFFERENT facility levels --
    used to regression-test that national pooling (the one pooling level whose pool boundary
    excludes district) still treats them as two distinct facilities rather than silently
    merging them into one LP node keyed on fac_name alone (which would "keep first" whichever
    district's AMC/Facility_Level happened to sort first, letting a non-eligible facility in
    one district piggy-back on an eligible facility's eligibility in another):
      - DistrictA/F_shared (1a, AMC=50): eligible receiver, OB = 0.1x AMC.
      - DistrictB/F_shared (2,  AMC=50): NON-eligible (level 2), OB = 0.1x AMC -> must never gain.
      - DistrictA/F_donA   (1a, AMC=50): donor, OB = 4x AMC.
    """
    rows = [
        ("F_shared", "DistrictA", "1a", 50.0, 0.1),
        ("F_shared", "DistrictB", "2", 50.0, 0.1),
        ("F_donA", "DistrictA", "1a", 50.0, 4.0),
    ]
    df = pd.DataFrame(rows, columns=["fac_name", "district", "Facility_Level", "amc", "ob_mult"])
    df["month"] = 1
    df["item_code"] = "42"
    df["opening_bal"] = df["amc"] * df["ob_mult"]
    df["available_prop"] = np.clip(df["ob_mult"], 0.0, 1.0) * 0.9
    return df.drop(columns="ob_mult")


def _make_synthetic_lmis_cluster_name_collision() -> pd.DataFrame:
    """
    Two districts that both contain a facility literally named 'F_shared' -- a name collision --
    used to regression-test that cluster pooling never merges facilities across districts just
    because they share a name (see build_capacity_clusters_all / redistribute_pooling_lp).
    """
    rows = [
        ("F_shared", "DistrictA", "1a", 50.0, 0.1),   # receiver
        ("F_donA", "DistrictA", "1a", 50.0, 4.0),     # donor
        ("F_shared", "DistrictB", "1a", 50.0, 0.1),   # receiver, same name, different district
        ("F_donB", "DistrictB", "1a", 50.0, 4.0),     # donor
    ]
    df = pd.DataFrame(rows, columns=["fac_name", "district", "Facility_Level", "amc", "ob_mult"])
    df["month"] = 1
    df["item_code"] = "42"
    df["opening_bal"] = df["amc"] * df["ob_mult"]
    df["available_prop"] = np.clip(df["ob_mult"], 0.0, 1.0) * 0.9
    return df.drop(columns="ob_mult")


def _make_synthetic_lmis_duplicate_rows() -> pd.DataFrame:
    """
    One district where a single facility (F_dup) is represented by two raw rows for the same
    (district, item, month) -- simulating an upstream data-deduplication gap -- used to
    regression-test that redistribute_pooling_lp splits the solved allocation across duplicate
    rows (in proportion to their original opening balance) rather than writing the full solved
    amount to each row.
    """
    rows = [
        ("F_dup", "1a", 50.0, 5.0),     # half of F_dup's true opening balance
        ("F_dup", "1a", 50.0, 5.0),     # other half (duplicate row, same facility/item/month)
        ("F_don", "1a", 50.0, 200.0),   # ample donor
    ]
    df = pd.DataFrame(rows, columns=["fac_name", "Facility_Level", "amc", "opening_bal"])
    df["district"] = "TestDistrict"
    df["month"] = 1
    df["item_code"] = "42"
    df["available_prop"] = 0.1
    return df


def _make_synthetic_time_matrix() -> dict:
    """All six synthetic facilities within 20 minutes of each other, except F_zero at 90 min."""
    facs = ["F_hosp", "F_don1", "F_don2", "F_short1", "F_short2", "F_zero"]
    T = pd.DataFrame(20.0, index=facs, columns=facs)
    np.fill_diagonal(T.values, 0.0)
    T.loc["F_zero", :] = 90.0
    T.loc[:, "F_zero"] = 90.0
    T.loc["F_zero", "F_zero"] = 0.0
    return {"TestDistrict": T}


def run_smoke_tests(verbose: bool = True) -> bool:
    """
    End-to-end checks of both redistribution models on synthetic data. Verifies:
      - non-eligible levels' availability never changes,
      - post-redistribution availability >= baseline,
      - stock conservation,
      - donor protection and ceilings,
      - radius feasibility (the out-of-range facility receives nothing in the pairwise model).
    Returns True if all tests pass (raises AssertionError otherwise).
    """
    df = _make_synthetic_lmis()

    # ---- Pooling ----
    pooled, moves = redistribute_pooling_lp(df, tau_max=3.0, tau_donor_keep=1.5,
                                            pooling_level="district", return_move_log=True)
    validate_redistribution_output(pooled, "smoke:pooling", conservation="leq", strict=True)

    p_short = pooled.set_index("fac_name")
    assert p_short.loc["F_short1", "OB_prime"] > p_short.loc["F_short1", "OB"], \
        "Pooling smoke test: understocked eligible facility did not receive stock"
    assert p_short.loc["F_hosp", "OB_prime"] <= p_short.loc["F_hosp", "OB"] + 1e-6, \
        "Pooling smoke test: level-2 facility gained stock"
    assert p_short.loc["F_hosp", "OB_prime"] >= 1.5 * 100.0 - 1e-6, \
        "Pooling smoke test: donor drawn below 1.5 x AMC"

    # ---- Pairwise (default: no degree cap; connectivity bounded by radius + Qmin only) ----
    T_car = _make_synthetic_time_matrix()
    paired, edges = redistribute_radius_lp(df, time_matrix=T_car, radius_minutes=60,
                                           tau_keep=1.5, tau_tar=1.0,
                                           Qmin_proportion=0.25)
    validate_redistribution_output(paired, "smoke:pairwise", tau_max=None,
                                   conservation="exact", strict=True)

    q = paired.set_index("fac_name")
    assert q.loc["F_zero", "OB_prime"] == q.loc["F_zero", "OB"], \
        "Pairwise smoke test: out-of-radius facility received stock"
    assert q.loc["F_short1", "OB_prime"] > q.loc["F_short1", "OB"], \
        "Pairwise smoke test: in-radius understocked facility did not receive stock"
    # receivers should not be filled above tau_tar * AMC
    assert q.loc["F_short1", "OB_prime"] <= 1.0 * 60.0 + 1e-6, \
        "Pairwise smoke test: receiver filled above tau_tar x AMC"
    if edges is not None and len(edges):
        # every executed transfer must be at least the receiver's minimum lot
        merged = edges.merge(df[["fac_name", "amc"]], left_on="receiver_fac", right_on="fac_name")
        assert (merged["units_moved"] >= 0.25 * merged["amc"] - 1e-6).all(), \
            "Pairwise smoke test: transfer below minimum lot size"

    # ---- Pairwise with an explicit degree cap requested (sensitivity-check code path) ----
    # F_hosp and F_don1/F_don2 can together reach both F_short1 and F_short2 within 60 min in
    # the synthetic district, so K_out=1 should force each donor to serve at most one receiver.
    paired_capped, edges_capped = redistribute_radius_lp(
        df, time_matrix=T_car, radius_minutes=60,
        tau_keep=1.5, tau_tar=1.0, K_in=1, K_out=1, Qmin_proportion=0.25,
    )
    validate_redistribution_output(paired_capped, "smoke:pairwise_capped", tau_max=None,
                                   conservation="exact", strict=True)
    if edges_capped is not None and len(edges_capped):
        assert edges_capped.groupby("donor_fac")["receiver_fac"].nunique().max() <= 1, \
            "Pairwise smoke test: explicit K_out=1 not respected"
        assert edges_capped.groupby("receiver_fac")["donor_fac"].nunique().max() <= 1, \
            "Pairwise smoke test: explicit K_in=1 not respected"

    # ---- National pooling: cross-district mixing that district pooling cannot do ----
    nat_df = _make_synthetic_lmis_two_districts()
    dist_only, _ = redistribute_pooling_lp(nat_df, tau_max=3.0, tau_donor_keep=1.5,
                                           pooling_level="district")
    f1_district = dist_only.set_index("fac_name").loc["F1_recv", "OB_prime"]
    assert abs(f1_district - 5.0) < 1e-6, \
        "National smoke test setup invalid: district pooling unexpectedly changed the isolated receiver"

    national, _ = redistribute_pooling_lp(nat_df, tau_max=3.0, tau_donor_keep=1.5,
                                          pooling_level="national")
    validate_redistribution_output(national, "smoke:national_pooling",
                                   group_cols=("month", "item_code"),
                                   conservation="leq", strict=True)
    f1_national = national.set_index("fac_name").loc["F1_recv", "OB_prime"]
    assert f1_national > f1_district + 1e-6, \
        "National pooling smoke test: cross-district surplus did not reach the isolated receiver"

    # ---- National pooling: regression test for facility-name collisions across districts ----
    nat_coll_df = _make_synthetic_lmis_national_name_collision()
    nat_coll, _ = redistribute_pooling_lp(nat_coll_df, tau_max=3.0, tau_donor_keep=1.5,
                                          pooling_level="national")
    validate_redistribution_output(nat_coll, "smoke:national_name_collision",
                                   group_cols=("month", "item_code"), conservation="leq", strict=True)
    b_recv = nat_coll.loc[(nat_coll["fac_name"] == "F_shared") & (nat_coll["district"] == "DistrictB"),
                          "OB_prime"].iloc[0]
    assert abs(b_recv - 5.0) < 1e-6, \
        ("National pooling smoke test: a non-eligible (level 2) facility in DistrictB "
         "incorrectly gained stock via a same-named eligible facility in DistrictA")

    # ---- Cluster pooling: regression test for facility-name collisions across districts ----
    coll_df = _make_synthetic_lmis_cluster_name_collision()
    cluster_map = pd.Series({
        ("DistrictA", "F_shared"): "DistrictA#C00", ("DistrictA", "F_donA"): "DistrictA#C00",
        ("DistrictB", "F_shared"): "DistrictB#C00", ("DistrictB", "F_donB"): "DistrictB#C00",
    })
    cluster_map.index.set_names(["district", "fac_name"], inplace=True)
    clustered, _ = redistribute_pooling_lp(coll_df, tau_max=3.0, tau_donor_keep=1.5,
                                           pooling_level="cluster", cluster_map=cluster_map)
    assert len(clustered) == len(coll_df), \
        "Cluster smoke test: facility-name collision across districts caused row duplication"
    validate_redistribution_output(clustered, "smoke:cluster_name_collision",
                                   conservation="leq", strict=True)
    a_recv = clustered.loc[(clustered["fac_name"] == "F_shared") & (clustered["district"] == "DistrictA"),
                           "OB_prime"].iloc[0]
    b_recv = clustered.loc[(clustered["fac_name"] == "F_shared") & (clustered["district"] == "DistrictB"),
                           "OB_prime"].iloc[0]
    assert a_recv > 5.0 + 1e-6 and b_recv > 5.0 + 1e-6, \
        "Cluster smoke test: same-named receiver in one or both districts was not topped up"

    # ---- Duplicate raw rows: regression test for split-not-copied write-back ----
    dup_df = _make_synthetic_lmis_duplicate_rows()
    dup_pooled, dup_moves = redistribute_pooling_lp(dup_df, tau_max=3.0, tau_donor_keep=1.5,
                                                     pooling_level="district")
    validate_redistribution_output(dup_pooled, "smoke:duplicate_rows", conservation="leq", strict=True)
    fdup_total_after = dup_pooled.loc[dup_pooled["fac_name"] == "F_dup", "OB_prime"].sum()
    fdup_solved = dup_moves.loc[dup_moves["facility"] == "F_dup", "x_allocated"].iloc[0]
    assert abs(fdup_total_after - fdup_solved) < 1e-6, \
        "Duplicate-row smoke test: solved allocation was not split correctly across duplicate rows"

    if verbose:
        print("All smoke tests passed.")
    return True


# ======================================================================================
# 6) Visualisations
# ======================================================================================
SCENARIO_COLORS = {
    "National pooling": "#08306b",
    "District pooling": "#1f78b4",
    "Neighbourhood pooling": "#a6cee3",
    "Pairwise exchange (Large radius)": "#33a02c",
    "Pairwise exchange (Small radius)": "#b2df8a",
}


def summarise_stockout_prevention(
    scenario_dfs: Dict[str, pd.DataFrame],
    eligible_levels: Iterable[str] = ELIGIBLE_LEVELS,
    level_col: str = "Facility_Level",
    amc_col: str = "amc",
    by: list[str] | None = None,
) -> pd.DataFrame:
    """
    For each scenario dataframe (with columns available_prop and available_prop_redis), compute
    at eligible levels: baseline expected stock-out risk (1 - p), residual risk after
    redistribution (1 - p'), and the share of anticipated risk averted. Optionally disaggregated
    by `by` (e.g. ["month"] or ["district"]).
    """
    records = []
    for name, df in scenario_dfs.items():
        d = df[df[level_col].isin(set(eligible_levels))].copy()
        d = d[pd.to_numeric(d[amc_col], errors="coerce").fillna(0.0) > 0]
        d["risk_base"] = 1.0 - d["available_prop"].astype(float)
        d["risk_post"] = 1.0 - d["available_prop_redis"].astype(float)
        keys = by or []
        grouped = d.groupby(keys) if keys else [((), d)]
        for key, gd in grouped:
            base, post = gd["risk_base"].mean(), gd["risk_post"].mean()
            rec = {"scenario": name, "risk_base": base, "risk_post": post,
                   "risk_averted": base - post,
                   "share_averted": (base - post) / base if base > 0 else np.nan}
            if keys:
                key = key if isinstance(key, tuple) else (key,)
                rec.update(dict(zip(keys, key)))
            records.append(rec)
    return pd.DataFrame(records)


def plot_stockout_prevention(
    scenario_dfs: Dict[str, pd.DataFrame],
    figures_path: Path,
    eligible_levels: Iterable[str] = ELIGIBLE_LEVELS,
    figname: str = "stockout_risk_averted_summary.png",
) -> pd.DataFrame:
    """
    Two-panel appendix figure:
      (a) expected stock-out risk (share of facility-item-months out of stock) at baseline vs
          after redistribution, by scenario - a dumbbell chart making the size of the remaining
          problem visually explicit;
      (b) share of the anticipated stock-out risk averted by each scenario, with the
          across-district distribution shown as individual points.
    Returns the national summary dataframe (also useful as an appendix table).
    """
    summary = summarise_stockout_prevention(scenario_dfs, eligible_levels=eligible_levels)
    by_district = summarise_stockout_prevention(scenario_dfs, eligible_levels=eligible_levels,
                                                by=["district"])

    order = list(scenario_dfs.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # ---- Panel (a): dumbbell, baseline vs post ----
    ax = axes[0]
    ypos = np.arange(len(order))[::-1]
    for yy, name in zip(ypos, order):
        row = summary[summary["scenario"] == name].iloc[0]
        ax.plot([row["risk_base"], row["risk_post"]], [yy, yy], color="grey", lw=2, zorder=1)
        ax.scatter(row["risk_base"], yy, s=70, color="#b2182b", zorder=2,
                   label="Baseline" if yy == ypos[0] else None)
        ax.scatter(row["risk_post"], yy, s=70,
                   color=SCENARIO_COLORS.get(name, "#1f78b4"), zorder=2,
                   label="After redistribution" if yy == ypos[0] else None)
        ax.annotate(f"-{row['share_averted']:.0%}",
                    xy=(row["risk_post"], yy), xytext=(-8, 8),
                    textcoords="offset points", ha="right", fontsize=9, color="#333333")
    ax.set_yticks(ypos)
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel("Expected stock-out risk at levels 1a/1b\n(mean probability a consumable is out of stock)")
    ax.set_xlim(left=0)
    ax.legend(fontsize=8, loc="lower right", frameon=True)
    ax.set_title("(a) Stock-out risk before and after redistribution", fontsize=10)

    # ---- Panel (b): share of risk averted, national bar + district points ----
    ax2 = axes[1]
    bar_vals = [summary.loc[summary["scenario"] == s, "share_averted"].iloc[0] for s in order]
    bars = ax2.bar(range(len(order)), bar_vals,
                   color=[SCENARIO_COLORS.get(s, "#1f78b4") for s in order], alpha=0.85, zorder=1)
    rng = np.random.default_rng(0)
    for xi, s in enumerate(order):
        pts = by_district.loc[by_district["scenario"] == s, "share_averted"].dropna()
        jitter = rng.uniform(-0.18, 0.18, len(pts))
        ax2.scatter(xi + jitter, pts, s=10, color="black", alpha=0.35, zorder=2)
    for b, v in zip(bars, bar_vals):
        ax2.annotate(f"{v:.0%}", xy=(b.get_x() + b.get_width() / 2, v),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels(order, rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("Share of anticipated stock-out risk averted")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.set_title("(b) Share of stock-out risk averted\n(bars: national; points: districts)", fontsize=10)

    fig.tight_layout()
    figures_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_path / figname, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {figures_path / figname}")
    return summary


def plot_stockout_prevention_by_month(
    scenario_dfs: Dict[str, pd.DataFrame],
    figures_path: Path,
    eligible_levels: Iterable[str] = ELIGIBLE_LEVELS,
    figname: str = "stockout_risk_averted_by_month.png",
) -> pd.DataFrame:
    """
    Seasonality view: monthly expected stock-out risk at baseline (single reference line)
    and after each redistribution scenario. Highlights whether redistribution helps most
    in the months when stock-outs are worst.
    """
    monthly = summarise_stockout_prevention(scenario_dfs, eligible_levels=eligible_levels, by=["month"])
    monthly["month"] = pd.to_numeric(monthly["month"], errors="coerce")
    monthly = monthly.sort_values("month")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    first = list(scenario_dfs.keys())[0]
    base = monthly[monthly["scenario"] == first]
    ax.plot(base["month"], base["risk_base"], color="#b2182b", lw=2.2,
            marker="o", ms=4, label="Baseline (no redistribution)")
    for name in scenario_dfs.keys():
        md = monthly[monthly["scenario"] == name]
        ax.plot(md["month"], md["risk_post"], lw=1.6, marker="o", ms=3,
                color=SCENARIO_COLORS.get(name, None), label=name)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Month")
    ax.set_ylabel("Expected stock-out risk at levels 1a/1b\n(mean probability out of stock)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, frameon=True)
    ax.set_title("Monthly stock-out risk, before and after redistribution", fontsize=11)

    fig.tight_layout()
    figures_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_path / figname, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {figures_path / figname}")
    return monthly


# --------------------------------------------------------------------------------------
# Violin plots of change in availability (carried over from the original script)
# --------------------------------------------------------------------------------------
def prep_violin_df(df, scenario_name, keep_facs_with_no_change=True,
                   eligible_levels: Iterable[str] = ELIGIBLE_LEVELS):
    out = df.copy()
    out["delta_p"] = out["available_prop_redis"] - out["available_prop"]
    mask = out["Facility_Level"].isin(set(eligible_levels)) & (out["amc"] > 0)
    if not keep_facs_with_no_change:
        mask &= (out["OB_prime"] > out["OB"])
    return out.loc[mask, ["district", "delta_p"]].assign(scenario=scenario_name)


def _add_custom_legend(fig=None, legend_location="upper right"):
    iqr_patch = mpatches.Rectangle((0, 0), 1, 1, facecolor="grey", edgecolor="black",
                                   linewidth=1, label="Interquartile range (IQR)")
    median_patch = mlines.Line2D([], [], color="#b2182b", marker="o", linestyle="None",
                                 markersize=5, label="Median")
    mean_patch = mlines.Line2D([], [], color="#b2182b", marker="D", linestyle="None",
                               markersize=6, label="Mean")
    handles = [iqr_patch, median_patch, mean_patch]
    if fig is None:
        plt.legend(handles=handles, loc=legend_location, fontsize=8, frameon=True)
    else:
        fig.legend(handles=handles, loc=legend_location, ncol=3, fontsize=8, frameon=True)


# Default figure convention: x-axis category labels are wrapped to a fixed character width and
# rotated 90 degrees, rather than angled -- keeps long scenario/category names fully legible
# without overlapping regardless of how many categories are on the axis.
DEFAULT_XTICK_WRAP_WIDTH = 20


def wrap_and_rotate_xticklabels(ax, width: int = DEFAULT_XTICK_WRAP_WIDTH, fontsize: int = 9):
    """Wrap each x-tick label to `width` characters and rotate 90 degrees. Call this on a
    matplotlib Axes AFTER the axis's category order is finalised (e.g. after the last plotting
    call on `ax`)."""
    labels = [textwrap.fill(t.get_text(), width) for t in ax.get_xticklabels()]
    ax.set_xticks(ax.get_xticks())  # fix tick positions before relabelling (avoids a matplotlib warning)
    ax.set_xticklabels(labels, rotation=90, ha='center', va='top', fontsize=fontsize)


def do_violin_plot_change_in_p(
    violin_df: pd.DataFrame,
    figname: str,
    figures_path: Path,
    by_district: bool = False,
    district_col: str = "district",
    ncol: int = 4,
    legend_location="upper right",
):
    """Violin + box + mean/median overlay plots of change in availability (national or by district)."""
    figures_path.mkdir(parents=True, exist_ok=True)

    if not by_district:
        mean_df = violin_df.groupby("scenario", as_index=False)["delta_p"].mean()
        median_df = violin_df.groupby("scenario", as_index=False)["delta_p"].median()

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.violinplot(data=violin_df, x="scenario", y="delta_p", cut=0, density_norm="width",
                       inner=None, linewidth=0.8, color="#4C72B0", alpha=0.6, ax=ax)
        sns.boxplot(data=violin_df, x="scenario", y="delta_p", width=0.03, showcaps=True,
                    showfliers=False,
                    boxprops={"facecolor": "grey", "edgecolor": "black", "linewidth": 1},
                    whiskerprops={"linewidth": 1}, medianprops={"linewidth": 0}, ax=ax)
        sns.scatterplot(data=mean_df, x="scenario", y="delta_p", color="#b2182b", marker="D",
                        s=60, zorder=10, ax=ax)
        sns.scatterplot(data=median_df, x="scenario", y="delta_p", color="#b2182b", marker="o",
                        s=45, zorder=11, ax=ax)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylabel("Change in probability of availability (Δp)")
        ax.set_xlabel("")
        _add_custom_legend(legend_location=legend_location)
        wrap_and_rotate_xticklabels(ax)
        fig.tight_layout()
        fig.savefig(figures_path / figname, dpi=600)
        plt.close(fig)
        return

    g = sns.catplot(data=violin_df, x="scenario", y="delta_p", col=district_col, col_wrap=ncol,
                    kind="violin", cut=0, density_norm="width", inner=None, linewidth=0.6,
                    color="#4C72B0", alpha=0.6, height=3, aspect=1)
    for ax, (district, df_d) in zip(g.axes.flat, violin_df.groupby(district_col)):
        mean_df = df_d.groupby("scenario", as_index=False)["delta_p"].mean()
        median_df = df_d.groupby("scenario", as_index=False)["delta_p"].median()
        sns.boxplot(data=df_d, x="scenario", y="delta_p", width=0.03, showcaps=True,
                    showfliers=False,
                    boxprops={"facecolor": "grey", "edgecolor": "black", "linewidth": 0.8},
                    whiskerprops={"linewidth": 0.8}, medianprops={"linewidth": 0}, ax=ax)
        sns.scatterplot(data=mean_df, x="scenario", y="delta_p", color="#b2182b", marker="D",
                        s=35, zorder=10, ax=ax)
        sns.scatterplot(data=median_df, x="scenario", y="delta_p", color="#b2182b", marker="o",
                        s=30, zorder=11, ax=ax)
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_xlabel("")
        ax.set_ylabel("Δp")
        wrap_and_rotate_xticklabels(ax, width=12, fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_title(district, fontsize=9)
    _add_custom_legend(fig=g.fig, legend_location=legend_location)
    g.fig.tight_layout()
    g.fig.savefig(figures_path / figname, dpi=600)
    plt.close()


def generate_stock_adequacy_heatmap(
    df: pd.DataFrame,
    figures_path: Path,
    filename: str = "heatmap_adequacy_opening_vs_3xamc.png",
    y_var: str = "district",
    value_var: str = "item_code",
    value_label: str = "",
    include_missing_as_fail: bool = False,
    amc_threshold: float = 3.0,
    compare: str = "ge",
    decimals: int = 0,
    cmap: str = "RdYlGn",
    figsize=None,
    xtick_rotation: int = 45,
    ytick_rotation: int = 0,
    annotation: bool = True,
    footnote: str = None,
):
    """
    Heatmap: for each (month, y_var), the % of value_var groups whose summed opening balance
    is >= (or <=) amc_threshold x summed AMC.
    """
    import calendar

    df = df.copy()
    df["month"] = pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["month"])
    df["month"] = df["month"].astype(int)
    df["_month_label"] = df["month"].map(lambda m: calendar.month_abbr[m])

    agg = (
        df.groupby(["month", "_month_label", y_var, value_var], dropna=False)
        .agg(opening_bal=("opening_bal", "sum"), amc=("amc", "sum"))
        .reset_index()
    )
    agg = agg[(agg["amc"] != 0)]
    agg = agg[~((agg["amc"] == 0) & (agg["opening_bal"] == 0))]

    if include_missing_as_fail:
        ok = agg[["opening_bal", "amc"]].notna().all(axis=1)
        left, right = agg["opening_bal"], amc_threshold * agg["amc"]
        cond = ((left <= right) if compare == "le" else (left >= right)) & ok
    else:
        valid = agg.dropna(subset=["opening_bal", "amc"])
        cond = pd.Series(False, index=agg.index)
        left, right = valid["opening_bal"], amc_threshold * valid["amc"]
        cond.loc[valid.index] = (left <= right) if compare == "le" else (left >= right)

    agg["condition_met"] = cond.astype(int)

    if include_missing_as_fail:
        denom = agg.groupby(["month", "_month_label", y_var])[value_var].nunique()
        numer = agg.groupby(["month", "_month_label", y_var])["condition_met"].sum()
    else:
        valid_mask = agg[["opening_bal", "amc"]].notna().all(axis=1)
        denom = agg[valid_mask].groupby(["month", "_month_label", y_var])[value_var].nunique()
        numer = agg[valid_mask].groupby(["month", "_month_label", y_var])["condition_met"].sum()

    pct = (numer / denom * 100).replace([np.inf, -np.inf], np.nan).reset_index(name="pct_meeting")

    month_order = (pct[["month", "_month_label"]].drop_duplicates()
                   .sort_values("month")["_month_label"].tolist())
    heatmap_df = pct.pivot(index=y_var, columns="_month_label", values="pct_meeting")
    heatmap_df = heatmap_df.reindex(columns=month_order)

    heatmap_df.loc["Average"] = heatmap_df.mean(axis=0)
    heatmap_df["Average"] = heatmap_df.mean(axis=1)

    if decimals is not None:
        heatmap_df = heatmap_df.round(decimals)

    if figsize is None:
        n_rows, n_cols = len(heatmap_df), len(heatmap_df.columns)
        figsize = (max(8, n_cols * 0.6), max(6, n_rows * 0.2))

    sns.set(font_scale=1.0)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(heatmap_df, cmap=cmap, cbar_kws={"label": value_label}, ax=ax,
                annot=annotation, annot_kws={"size": 10}, vmin=0, vmax=100)
    ax.set_xlabel("Month")
    ax.set_ylabel(f"{y_var}")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=xtick_rotation)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=ytick_rotation)

    try:
        ax.figure.axes[-1].ticklabel_format(style="plain")
    except Exception:
        pass

    if footnote is not None:
        fig.subplots_adjust(bottom=0.08)
        fig.text(0.5, 0.035, footnote, ha="center", va="top", fontsize=10)

    figures_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_path / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig, ax, heatmap_df


if __name__ == "__main__":
    run_smoke_tests()
