"""Effect modification of WBGT associations with service utilisation.

Uses the exact estimation samples and spline design from asborbed_fe_pois.py.
Primary outputs: stratum-specific IRRs, ratios of IRRs, and cluster-robust
interaction tests for a common, observed-temperature contrast.

IMPORTANT:
  Add '_nb_data': nb_data to fit_indicator()'s return dictionary.
  Set CLIMATOLOGY_FILE to a facility-level CSV computed from a FIXED historical
  weather reference period if EM_CONTINUOUS=True. Do not estimate climatology
  from outcome-specific regression samples.
  Absolute deficit calculations from the old script are deliberately omitted:
  they are NOT heat-attributable losses and their rescaled-draw CIs are invalid
  for that estimand. Use a separately specified counterfactual and bootstrap.
  Joint adjusted model is an optional, more restrictive sensitivity analysis.
  Wald tests use asymptotic chi-square reference; with few clusters, validate
  important results using a cluster bootstrap.
"""

import os
import re
import time
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import patsy
import pyfixest as pf
from scipy import sparse, stats

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MODEL_MODULE = "asborbed_fe_pois"  # stem of the model script (filename is spelled "asborbed")

# Facility metadata. Modifier columns are looked for in the regression panel
# (carried through nb_data) and, if MODIFIER_FILE is set, in that CSV (one row
# per facility; the facility-name column must match the panel's `facility`).
MODIFIER_FILE: str | None = None
# Explicit column names; None = auto-detect (must match exactly one candidate).
MODIFIER_COLS: dict[str, str | None] = {"level": "Ftype", "authority": "A105", "urban": "Resid"}

# Categorical modifiers ("climatology" = facility-mean-WBGT terciles is still
# supported but no longer a default; the continuous version replaces it).
EM_MODIFIERS = ["level", "authority", "urban"]
EM_CONTINUOUS = False  # Enable only with a fixed-period climatology file               # exposure x facility-mean WBGT (continuous)
EM_CONT_PCTILES = (10, 50, 90)     # percentiles of facility climatology to report IRRs at
EM_HI_PCTILE = 95          # exposure contrast: pooled-sample percentile ...
EM_REF_PCTILE = 50         # ... versus pooled-sample percentile
EM_N_DRAWS = 1000          # coefficient draws for absolute-deficit CIs
EM_DRAW_CHUNK = 100
EM_MIN_FACILITIES = 10     # strata smaller than this are excluded (and reported)
EM_MIN_HOT_ROWS = 30       # min hot facility-months in a stratum for a hot deficit
EM_RUN_JOINT = False  # Restrictive sensitivity model; enable if required        # additive model with all modifiers at once
CLIMATOLOGY_FILE = None  # CSV: one row per facility, columns facility, mean_wbgt
SUPPORT_LO_Q = 0.10
SUPPORT_HI_Q = 0.90
MIN_CONTRAST_DEGC = 0.5
EM_SEED = 42
FDR_ALPHA = 0.05
SAT_COL = "em_g"           # stratum column added to the data


# ---------------------------------------------------------------------------
# Metadata: locate columns, classify into strata
# ---------------------------------------------------------------------------
def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


_CANDIDATES = {
    "level": ["ftype", "facilitytype", "facilitylevel", "level"],
    "authority": ["owner", "facowner", "facilityowner", "managingauthority", "authority",
                  "ownership", "managedby", "agency"],
    "urban": ["urbanrural", "urban", "rural", "setting", "urbanicity", "popdensity",
              "populationdensity"],
}


def _pick_column(kind, columns):
    explicit = MODIFIER_COLS.get(kind)
    if explicit:
        if explicit not in columns:
            raise KeyError(f"MODIFIER_COLS[{kind!r}]={explicit!r} not found. Available: {list(columns)}")
        return explicit
    hits = [c for c in columns if _norm(c) in _CANDIDATES[kind]]
    if len(hits) != 1:
        raise KeyError(
            f"modifier {kind!r}: need exactly one candidate column, found {hits}. "
            f"Set MODIFIER_COLS[{kind!r}]. Available columns: {list(columns)}"
        )
    return hits[0]


def _level_group(v):
    t = str(v).strip().lower()
    if t in {"0", "1a"}:
        return "Primary"
    if t == "1b":
        return "Health centre"
    if t in {"2", "3", "4"}:
        return "Hospital"
    if "hospital" in t:
        return "Hospital"
    if "health cent" in t:
        return "Health centre"
    raise ValueError(f"Unrecognised facility level: {v!r}; explicitly map this value")


def _authority_group(v):
    t = str(v).strip().lower()
    if any(k in t for k in ("cham", "mission", "christian", "church", "catholic",
                            "presbyterian", "anglican", "adventist")):
        return "CHAM"
    if any(k in t for k in ("govern", "moh", "ministry", "public")):
        return "Government"
    if any(k in t for k in ("private", "ngo", "company", "armed", "other")):
        return "Other"
    raise ValueError(f"Unrecognised managing authority: {v!r}; explicitly map this value")


def _urban_group(v):
    t = str(v).strip().lower()
    if "peri" in t:
        return "Peri-urban"
    if "urban" in t:
        return "Urban"
    if "rural" in t:
        return "Rural"
    raise ValueError(f"urban/rural label {v!r} not recognised (expected urban/peri-urban/rural)")


def _facility_sources(results, cfg):
    """One row per facility (key = stripped facility name) with every panel
    column carried in nb_data, plus MODIFIER_FILE columns if given."""
    frames = []
    for r in results:
        nb = r["_nb_data"]
        frames.append(nb.drop_duplicates("facility").assign(
            fkey=lambda d: d["facility"].astype(str).str.strip()))
    joined = pd.concat(frames)
    src = joined.drop_duplicates("fkey").set_index("fkey")
    if MODIFIER_FILE:
        reg = pd.read_csv(MODIFIER_FILE)
        key = [c for c in reg.columns if _norm(c) in ("facility", "facilityname", "fname", "name")]
        if len(key) != 1:
            raise KeyError(f"{MODIFIER_FILE}: need exactly one facility-name column, found {key} "
                           f"in {list(reg.columns)}")
        reg["fkey"] = reg[key[0]].astype(str).str.strip()
        dup = reg["fkey"][reg["fkey"].duplicated()].unique().tolist()
        if dup:
            raise ValueError(f"{MODIFIER_FILE}: duplicated facility names, e.g. {dup[:10]}")
        reg = reg.set_index("fkey")
        extra = [c for c in reg.columns if c not in src.columns]
        src = src.join(reg[extra], how="left")
    return src


def _static_modifiers(results, cfg):
    """Facility-level strata for level / authority / urban, built once over the
    union of fitted facilities. Writes the facility table, the raw->group
    mapping and modifier crosstabs for inspection."""
    src = _facility_sources(results, cfg)
    out = pd.DataFrame(index=src.index)
    out["district"] = src[cfg.district_col].values
    mapping_rows = []
    for kind in ("level", "authority", "urban"):
        if kind not in cfg.modifiers:
            continue
        col = _pick_column(kind, src.columns)
        raw = src[col]
        if raw.isna().any():
            bad = raw.index[raw.isna()].tolist()
            raise ValueError(f"modifier {kind!r} ({col!r}): {len(bad)} facilities lack metadata, e.g. {bad[:10]}")
        if kind == "level":
            grp = raw.map(_level_group)
        elif kind == "authority":
            grp = raw.map(_authority_group)
        else:
            if pd.api.types.is_numeric_dtype(raw):
                grp = pd.qcut(raw, 3, labels=["Low density", "Mid density", "High density"]).astype(str)
            else:
                grp = raw.map(_urban_group)
        out[f"{kind}_raw"] = raw.values
        out[kind] = grp.values
        m = pd.DataFrame({"modifier": kind, "source_column": col, "raw": raw.astype(str).values,
                          "group": grp.values}).groupby(["modifier", "source_column", "raw", "group"]) \
            .size().rename("n_facilities").reset_index()
        mapping_rows.append(m)

    out.reset_index().to_csv(f"{cfg.out_dir}em_facility_modifiers_{cfg.out_tag}.csv", index=False)
    if mapping_rows:
        mp = pd.concat(mapping_rows)
        mp.to_csv(f"{cfg.out_dir}em_modifier_mapping_{cfg.out_tag}.csv", index=False)
        print("\nModifier mapping (raw label -> group) — CHECK THIS:")
        print(mp.to_string(index=False))

    # Crosstabs between modifiers + within-district variation
    kinds = [k for k in ("level", "authority", "urban") if k in out.columns]
    ct_rows = []
    for i, a in enumerate(kinds):
        for b in kinds[i + 1:]:
            ct = pd.crosstab(out[a], out[b])
            print(f"\nFacility counts: {a} x {b}\n{ct}")
            ct_rows.append(ct.stack().rename("n_facilities").reset_index()
                           .rename(columns={a: "group_a", b: "group_b"}).assign(modifier_a=a, modifier_b=b))
    for k in kinds:
        per_dist = out.groupby("district")[k].nunique()
        print(f"\n{k}: {int((per_dist >= 2).sum())}/{len(per_dist)} districts contain >=2 groups")
    if ct_rows:
        pd.concat(ct_rows).to_csv(f"{cfg.out_dir}em_modifier_crosstabs_{cfg.out_tag}.csv", index=False)
    return out


def _climatology_groups(nb, wbgt_var):
    fm = nb.assign(fkey=nb["facility"].astype(str).str.strip()).groupby("fkey")[wbgt_var].mean()
    return pd.qcut(fm, 3, labels=["Cool", "Mid", "Hot"]).astype(str)


def _assign_strata(nb, fac_group, min_fac):
    """Attach SAT_COL; drop strata with < min_fac facilities. Reference
    stratum (index 0) = most facilities."""
    fkey = nb["facility"].astype(str).str.strip()
    g = fkey.map(fac_group)
    if g.isna().any():
        miss = fkey[g.isna()].unique().tolist()
        raise ValueError(f"{len(miss)} facilities missing a stratum, e.g. {miss[:10]}")
    d = nb.assign(**{SAT_COL: g.values})
    nfac = d.groupby(SAT_COL)["facility"].nunique().sort_values(ascending=False)
    nobs = d.groupby(SAT_COL).size()
    info = pd.DataFrame({"level": nfac.index, "n_facilities": nfac.values,
                         "n_obs": nobs.loc[nfac.index].values})
    info["included"] = info["n_facilities"] >= min_fac
    small = info.loc[~info["included"], "level"].tolist()
    if small:
        print(f"    excluded strata (<{min_fac} facilities): {small}")
    levels = info.loc[info["included"], "level"].tolist()
    if len(levels) < 2:
        raise ValueError(f"fewer than 2 strata with >={min_fac} facilities: {info.to_dict('records')}")
    d = d[d[SAT_COL].isin(levels)].copy()
    return d.reset_index(drop=True), levels, info


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def _coef_block(model):
    names = list(model._coefnames)
    beta = model.coef().loc[names].values.astype(float)
    V = np.asarray(model._vcov, dtype=float)
    dg = np.diag(V)
    if not np.all(np.isfinite(dg)) or np.any(dg < 0):
        raise ValueError(f"invalid vcov diagonal (negative/non-finite): {dg}")
    return names, beta, V


def _wald_F(beta, V, R, n_clusters):
    """Asymptotic cluster-robust Wald chi-square; reject singular hypotheses."""
    R = np.atleast_2d(R)
    q = R.shape[0]
    if n_clusters <= q + 1:
        raise ValueError(f"Only {n_clusters} clusters for {q} restrictions")
    M = R @ V @ R.T
    if np.linalg.matrix_rank(M) < q:
        raise ValueError(f"Singular Wald covariance ({q} restrictions)")
    W = float((R @ beta) @ np.linalg.solve(M, R @ beta))
    return float(stats.chi2.sf(W, q))


def _common_contrast(data, wb, group_col, levels, cfg):
    """Use the intersection of each group's central exposure interval.

    The contrast is shared by all groups and always lies in the empirical
    central range of each group; report group-specific counts near endpoints.
    """
    intervals = data.groupby(group_col)[wb].quantile(
        [SUPPORT_LO_Q, SUPPORT_HI_Q]).unstack()
    low = float(intervals.loc[levels, SUPPORT_LO_Q].max())
    high = float(intervals.loc[levels, SUPPORT_HI_Q].min())
    if high - low < MIN_CONTRAST_DEGC:
        raise ValueError(f"Insufficient common WBGT support: [{low:.2f}, {high:.2f}]")
    return low, high


def _exposure_weights(design_info, shift, spline_cols, lag_cols, hi, ref, cumulative):
    """Weights on exposure columns for the log-IRR of WBGT=hi vs WBGT=ref.
    Contemporaneous: spline contrast only (as irr_contrast in the model script).
    Cumulative: additionally sustained exposure through each lag term."""
    Bh = np.asarray(patsy.build_design_matrices([design_info], {"x": np.array([hi]) - shift})[0]).ravel()
    Br = np.asarray(patsy.build_design_matrices([design_info], {"x": np.array([ref]) - shift})[0]).ravel()
    w = {c: float(Bh[i] - Br[i]) for i, c in enumerate(spline_cols)}
    if cumulative:
        for lc in lag_cols:
            w[lc] = float(hi - ref)
    return w


def _vec(names_idx, weights, suffix, label):
    v = np.zeros(len(names_idx))
    for c, w in weights.items():
        key = f"{c}__{suffix}"
        if key not in names_idx:
            raise ValueError(f"{label}: exposure column {key!r} dropped by fepois (collinear) — refusing to zero-pad")
        v[names_idx[key]] = w
    return v


def _fit_saturated(data, levels, ctrl_cols, exp_cols, cluster_col, use_zone_year):
    cols = {}
    for i, lvl in enumerate(levels):
        ind = (data[SAT_COL].values == lvl).astype(float)
        for c in ctrl_cols + exp_cols:
            cols[f"{c}__g{i}"] = data[c].values * ind
    X = pd.DataFrame(cols, index=data.index)
    base = pd.DataFrame({
        "y_int": data["y_int"].values,
        "facility": data["facility"].values,
        "fe_month_g": data["month"].astype(str).values + "_" + data[SAT_COL].astype(str).values,
        cluster_col: data[cluster_col].values,
    }, index=data.index)
    fe = "facility + fe_month_g"
    if use_zone_year:
        base["fe_zy_g"] = data["zone_year"].astype(str).values + "_" + data[SAT_COL].astype(str).values
        fe += " + fe_zy_g"
    d = pd.concat([base, X], axis=1)
    ctrl_terms = [f"{c}__g{i}" for i in range(len(levels)) for c in ctrl_cols]
    exp_terms = [f"{c}__g{i}" for i in range(len(levels)) for c in exp_cols]
    vc = {"CRV1": cluster_col}
    m_base = pf.fepois(f"y_int ~ {' + '.join(ctrl_terms)} | {fe}", data=d, vcov=vc)
    m_wx = pf.fepois(f"y_int ~ {' + '.join(ctrl_terms + exp_terms)} | {fe}", data=d, vcov=vc)
    return d, m_base, m_wx, exp_terms


def _bh(pvals, alpha=FDR_ALPHA):
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    if ok.sum() == 0:
        return q
    po = p[ok]
    n = len(po)
    order = np.argsort(po)
    adj = po[order] * n / np.arange(1, n + 1)
    adj = np.clip(np.minimum.accumulate(adj[::-1])[::-1], 0, 1)
    qo = np.empty(n)
    qo[order] = adj
    q[ok] = qo
    return q


def _z_ci(logv, se):
    p = 2.0 * (1.0 - stats.norm.cdf(abs(logv / se))) if se > 0 else np.nan
    return np.exp(logv), np.exp(logv - 1.96 * se), np.exp(logv + 1.96 * se), p


# ---------------------------------------------------------------------------
# One indicator x modifier
# ---------------------------------------------------------------------------
def _run_one(ind, r, nb, mod, fac_group, cfg, fit_fn, rng):
    wb = cfg.wbgt_var
    spline_cols = list(r["_spline_cols"])
    exp_cols = spline_cols + list(cfg.lag_cols)
    ctrl_cols = list(cfg.ctrl_cols)

    data, levels, strata_info = _assign_strata(nb, fac_group, cfg.min_facilities)
    strata_info.insert(0, "indicator", ind)
    strata_info.insert(1, "modifier", mod)
    strata_info["reference"] = strata_info["level"] == levels[0]

    d, m_base, m_wx, exp_terms = _fit_saturated(
        data, levels, ctrl_cols, exp_cols, cfg.cluster_col, cfg.use_zone_year)

    names, beta, V = _coef_block(m_wx)
    nidx = {n: i for i, n in enumerate(names)}
    missing = [t for t in exp_terms if t not in nidx]
    if missing:
        raise ValueError(f"exposure columns dropped by fepois: {missing}")
    n_clusters = data[cfg.cluster_col].nunique()
    G = len(levels)

    ref, hi = _common_contrast(data, wb, SAT_COL, levels, cfg)
    design_info = r["_design_map"][wb]
    shift = r["_shifts"][wb]

    modes = [("contemporaneous", False)] + ([("cumulative", True)] if cfg.lag_cols else [])
    irr_rows, test = [], {"indicator": ind, "modifier": mod, "reference": levels[0],
                          "n_strata": G, "n_clusters": n_clusters, "contrast_hi": hi, "contrast_ref": ref,
                          "status": "ok"}

    # ---- joint heterogeneity test over ALL exposure coefficients -----------
    rows = []
    for i in range(1, G):
        for c in exp_cols:
            row = np.zeros(len(names))
            row[nidx[f"{c}__g{i}"]] = 1.0
            row[nidx[f"{c}__g0"]] = -1.0
            rows.append(row)
    test["p_joint_exposure"] = _wald_F(beta, V, np.array(rows), n_clusters)

    # ---- IRR per stratum, ratio vs reference, contrast test -----------------
    for mode, cum in modes:
        w = _exposure_weights(design_info, shift, spline_cols, cfg.lag_cols, hi, ref, cum)
        vecs = [_vec(nidx, w, f"g{i}", f"{ind}/{mod}") for i in range(G)]
        for i, lvl in enumerate(levels):
            lg, se = float(vecs[i] @ beta), float(np.sqrt(max(vecs[i] @ V @ vecs[i], 0)))
            irr, lo, up, _ = _z_ci(lg, se)
            row = {"indicator": ind, "modifier": mod, "level": lvl, "reference": levels[0], "mode": mode,
                   "n_facilities": int(strata_info.set_index("level").loc[lvl, "n_facilities"]),
                   "n_obs": int(strata_info.set_index("level").loc[lvl, "n_obs"]),
                   "wbgt_hi": hi, "wbgt_ref": ref, "irr": irr, "irr_lo": lo, "irr_hi": up}
            if i > 0:
                dv = vecs[i] - vecs[0]
                lr, se_r = float(dv @ beta), float(np.sqrt(max(dv @ V @ dv, 0)))
                rr, rlo, rhi, rp = _z_ci(lr, se_r)
                row.update({"irr_ratio_vs_ref": rr, "ratio_lo": rlo, "ratio_hi": rhi, "ratio_p": rp})
            irr_rows.append(row)
        Rc = np.array([vecs[i] - vecs[0] for i in range(1, G)])
        test[f"p_contrast_{mode}"] = _wald_F(beta, V, Rc, n_clusters)

    # ---- observed support at the two endpoints, by stratum -----------------
    for row in irr_rows:
        sub = data.loc[data[SAT_COL] == row["level"], wb]
        width = max(0.5, 0.05 * (hi - ref))
        row["n_near_ref"] = int(((sub >= ref - width) & (sub <= ref + width)).sum())
        row["n_near_hi"] = int(((sub >= hi - width) & (sub <= hi + width)).sum())
        row["support_band_halfwidth_degC"] = width
    return {"irr": irr_rows, "strata": strata_info, "test": test, "levels": levels}


# ---------------------------------------------------------------------------
# Continuous modifier: facility-mean WBGT
# ---------------------------------------------------------------------------
def _load_fixed_climatology(path):
    if not path:
        raise ValueError("Set CLIMATOLOGY_FILE to a fixed-reference-period weather CSV")
    d = pd.read_csv(path)
    if not {"facility", "mean_wbgt"}.issubset(d.columns):
        raise ValueError("Climatology CSV must contain facility and mean_wbgt")
    d["facility"] = d["facility"].astype(str).str.strip()
    if d["facility"].duplicated().any() or d["mean_wbgt"].isna().any():
        raise ValueError("Climatology CSV has duplicate facilities or missing means")
    return d.set_index("facility")["mean_wbgt"].astype(float).to_dict()


def _modes(cfg):
    return [("contemporaneous", False)] + ([("cumulative", True)] if cfg.lag_cols else [])


def _contrast_matrix(design_info, shift, spline_cols, lag_cols, hi_arr, ref_arr, cumulative):
    """Per-facility exposure weights for log-IRR of WBGT=hi_i vs ref_i (arrays)."""
    Bh = np.asarray(patsy.build_design_matrices([design_info], {"x": np.asarray(hi_arr) - shift})[0])
    Br = np.asarray(patsy.build_design_matrices([design_info], {"x": np.asarray(ref_arr) - shift})[0])
    W = {c: Bh[:, i] - Br[:, i] for i, c in enumerate(spline_cols)}
    if cumulative:
        for lc in lag_cols:
            W[lc] = np.asarray(hi_arr) - np.asarray(ref_arr)
    return W


def _run_continuous(ind, r, nb, fac_mean, cfg, fit_fn):
    """Exposure x centred facility-mean WBGT in one pooled fepois."""
    wb = cfg.wbgt_var
    spline_cols = list(r["_spline_cols"])
    exp_cols = spline_cols + list(cfg.lag_cols)
    ctrl_int = [c for c in cfg.ctrl_cols if not c.startswith("year_fe")]
    fac_mean = pd.Series(fac_mean)

    d = nb.copy()
    fk = d["facility"].astype(str).str.strip()
    zraw = fk.map(fac_mean)
    if zraw.isna().any():
        raise ValueError(f"{int(zraw.isna().sum())} rows with facility missing a mean WBGT")
    fm = fac_mean[fk.unique()]
    z0 = float(fm.mean())
    zc = zraw.values - z0

    int_cols = {}
    for c in exp_cols + ctrl_int:
        int_cols[c] = f"{c}__zc"
        d[int_cols[c]] = d[c].values * zc
    rhs = list(cfg.ctrl_cols) + exp_cols + list(int_cols.values())
    model = fit_fn(rhs, d, cfg.cluster_col)
    names, beta, V = _coef_block(model)
    nidx = {n: i for i, n in enumerate(names)}
    missing = [c for c in exp_cols + [int_cols[c] for c in exp_cols] if c not in nidx]
    if missing:
        raise ValueError(f"continuous model: exposure/interaction columns dropped: {missing}")
    n_clusters = d[cfg.cluster_col].nunique()

    hi = float(np.percentile(d[wb], cfg.hi_pctile))
    ref = float(np.percentile(d[wb], cfg.ref_pctile))
    design_info, shift = r["_design_map"][wb], r["_shifts"][wb]

    R = np.zeros((len(exp_cols), len(names)))
    for i, c in enumerate(exp_cols):
        R[i, nidx[int_cols[c]]] = 1.0
    test = {"indicator": ind, "modifier": "mean_wbgt", "reference": "continuous",
            "n_strata": np.nan, "n_clusters": n_clusters, "contrast_hi": hi, "contrast_ref": ref,
            "status": "ok", "p_joint_exposure": _wald_F(beta, V, R, n_clusters)}

    grid = {p: float(np.percentile(fm.values, p)) for p in cfg.cont_pctiles}
    lo_p, hi_p = min(cfg.cont_pctiles), max(cfg.cont_pctiles)
    span = grid[hi_p] - grid[lo_p]
    rows = []
    for mode, cum in _modes(cfg):
        w = _exposure_weights(design_info, shift, spline_cols, cfg.lag_cols, hi, ref, cum)
        ve, vs = np.zeros(len(names)), np.zeros(len(names))
        for c, wt in w.items():
            ve[nidx[c]] = wt
            vs[nidx[int_cols[c]]] = wt
        lr, se = float(vs @ beta), float(np.sqrt(max(vs @ V @ vs, 0)))
        rr, lo, up, pv = _z_ci(lr, se)
        base = {"indicator": ind, "modifier": "mean_wbgt", "mode": mode, "wbgt_hi": hi, "wbgt_ref": ref}
        rows.append({**base, "row_type": "per_degC", "irr_ratio": rr, "ratio_lo": lo, "ratio_hi": up, "ratio_p": pv})
        rr2, lo2, up2, _ = _z_ci(lr * span, se * abs(span))
        rows.append({**base, "row_type": f"p{lo_p}_to_p{hi_p}", "clim_span_degC": span,
                     "irr_ratio": rr2, "ratio_lo": lo2, "ratio_hi": up2, "ratio_p": pv})
        test[f"p_contrast_{mode}"] = _wald_F(beta, V, vs[None, :], n_clusters)
        for p, zval in grid.items():
            v = ve + vs * (zval - z0)
            lg, se_g = float(v @ beta), float(np.sqrt(max(v @ V @ v, 0)))
            irr, glo, gup, _ = _z_ci(lg, se_g)
            near = fm[(fm >= np.percentile(fm.values, max(p - 5, 0))) &
                      (fm <= np.percentile(fm.values, min(p + 5, 100)))].index
            sel = fk.isin(set(near)).values
            rows.append({**base, "row_type": "at_percentile", "clim_percentile": p, "clim_mean_wbgt": zval,
                         "irr": irr, "irr_lo": glo, "irr_hi": gup,
                         "support_share_ge_hi": float((d[wb].values[sel] >= hi).mean())})

    # ---- (2) own contrast and (3) common-support window ----------------------
    g = d.assign(_fk=fk.values).groupby("_fk")[wb]
    own = pd.DataFrame({"ref": g.quantile(cfg.ref_pctile / 100.0), "hi": g.quantile(cfg.hi_pctile / 100.0)})
    bands = {}
    for p in grid:
        q_lo, q_hi = np.percentile(fm.values, max(p - 5, 0)), np.percentile(fm.values, min(p + 5, 100))
        bands[p] = fm[(fm >= q_lo) & (fm <= q_hi)].index
    # Overlap of central exposure ranges across all three climatology bands.
    ranges = []
    for p in grid:
        vals = d.loc[fk.isin(set(bands[p])), wb]
        ranges.append((float(vals.quantile(SUPPORT_LO_Q)),
                       float(vals.quantile(SUPPORT_HI_Q))))
    win_ref = max(x[0] for x in ranges)
    win_hi = min(x[1] for x in ranges)
    if win_hi - win_ref < MIN_CONTRAST_DEGC:
        raise ValueError("Insufficient common temperature support across climatology bands")
    for mode, cum in _modes(cfg):
        base = {"indicator": ind, "modifier": "mean_wbgt", "mode": mode}
        vband = {}
        for p, zval in grid.items():
            idx = bands[p]
            h, rf = own.loc[idx, "hi"].values, own.loc[idx, "ref"].values
            za = fm[idx].values - z0
            W = _contrast_matrix(design_info, shift, spline_cols, cfg.lag_cols, h, rf, cum)
            v = np.zeros(len(names))
            for c, wv in W.items():
                v[nidx[c]] += float(wv.mean())
                v[nidx[int_cols[c]]] += float((wv * za).mean())
            vband[p] = v
            lg, se_g = float(v @ beta), float(np.sqrt(max(v @ V @ v, 0)))
            irr, glo, gup, _ = _z_ci(lg, se_g)
            rows.append({**base, "row_type": "at_percentile_own", "clim_percentile": p, "clim_mean_wbgt": zval,
                         "n_facilities_band": len(idx), "wbgt_hi": float(h.mean()), "wbgt_ref": float(rf.mean()),
                         "irr": irr, "irr_lo": glo, "irr_hi": gup})
        dv = vband[hi_p] - vband[lo_p]
        lr, se = float(dv @ beta), float(np.sqrt(max(dv @ V @ dv, 0)))
        rr, lo, up, pv = _z_ci(lr, se)
        rows.append({**base, "row_type": f"own_p{hi_p}_vs_p{lo_p}", "irr_ratio": rr, "ratio_lo": lo,
                     "ratio_hi": up, "ratio_p": pv})
        test[f"p_own_contrast_{mode}"] = _wald_F(beta, V, dv[None, :], n_clusters)

        w = _exposure_weights(design_info, shift, spline_cols, cfg.lag_cols, win_hi, win_ref, cum)
        ve, vs = np.zeros(len(names)), np.zeros(len(names))
        for c, wt in w.items():
            ve[nidx[c]] = wt
            vs[nidx[int_cols[c]]] = wt
        lr, se = float(vs @ beta), float(np.sqrt(max(vs @ V @ vs, 0)))
        rr, lo, up, pv = _z_ci(lr, se)
        wbase = {**base, "wbgt_hi": win_hi, "wbgt_ref": win_ref}
        rows.append({**wbase, "row_type": "common_per_degC", "irr_ratio": rr, "ratio_lo": lo,
                     "ratio_hi": up, "ratio_p": pv})
        test[f"p_common_contrast_{mode}"] = _wald_F(beta, V, vs[None, :], n_clusters)
        for p, zval in grid.items():
            v = ve + vs * (zval - z0)
            lg, se_g = float(v @ beta), float(np.sqrt(max(v @ V @ v, 0)))
            irr, glo, gup, _ = _z_ci(lg, se_g)
            sel = fk.isin(set(bands[p])).values
            wv_ = d[wb].values[sel]
            rows.append({**wbase, "row_type": "at_percentile_common", "clim_percentile": p,
                         "clim_mean_wbgt": zval, "irr": irr, "irr_lo": glo, "irr_hi": gup,
                         "support_share_in_window": float(((wv_ >= win_ref) & (wv_ <= win_hi)).mean())})
    return rows, test


# ---------------------------------------------------------------------------
# Additive model with all modifiers at once
# ---------------------------------------------------------------------------
def _run_joint(ind, r, nb, fac_groups, cfg, fit_fn, cont_mean=None):
    wb = cfg.wbgt_var
    spline_cols = list(r["_spline_cols"])
    exp_cols = spline_cols + list(cfg.lag_cols)
    d = nb.copy()
    fkey = d["facility"].astype(str).str.strip()
    levels_by = {}
    for m, fg in fac_groups.items():
        g = fkey.map(fg)
        if g.isna().any():
            raise ValueError(f"joint model: facilities missing stratum for {m!r}")
        d[f"em_{m}"] = g.values
    # drop strata with too few facilities, per modifier
    for m in fac_groups:
        nfac = d.groupby(f"em_{m}")["facility"].nunique().sort_values(ascending=False)
        keep = nfac[nfac >= cfg.min_facilities].index.tolist()
        if len(keep) < 2:
            raise ValueError(f"joint model: modifier {m!r} has <2 strata with >={cfg.min_facilities} facilities")
        d = d[d[f"em_{m}"].isin(keep)]
        levels_by[m] = keep  # ordered by facility count (reference first)
    # keeping all modifiers' strata may shrink others; recompute order on final sample
    for m in fac_groups:
        nfac = d.groupby(f"em_{m}")["facility"].nunique().sort_values(ascending=False)
        levels_by[m] = nfac.index.tolist()
    d = d.reset_index(drop=True)

    inter_cols, block = {}, {}
    for m, lv in levels_by.items():
        for j, lvl in enumerate(lv[1:], start=1):
            ind_arr = (d[f"em_{m}"].values == lvl).astype(float)
            for c in exp_cols:
                name = f"{c}__{m}{j}"
                d[name] = d[c].values * ind_arr
                inter_cols[name] = (m, lvl, c)
                block.setdefault(m, []).append(name)
    zc_cols = {}
    if cont_mean is not None:
        fk = d["facility"].astype(str).str.strip()
        cm = pd.Series(cont_mean)
        zj = fk.map(cm).values - float(cm[fk.unique()].mean())
        for c in exp_cols:
            zc_cols[c] = f"{c}__zc"
            d[zc_cols[c]] = d[c].values * zj
        block["mean_wbgt"] = list(zc_cols.values())
    rhs = list(cfg.ctrl_cols) + exp_cols + list(inter_cols) + list(zc_cols.values())
    model = fit_fn(rhs, d, cfg.cluster_col)
    names, beta, V = _coef_block(model)
    nidx = {n: i for i, n in enumerate(names)}
    missing = [c for c in exp_cols + list(inter_cols) + list(zc_cols.values()) if c not in nidx]
    if missing:
        raise ValueError(f"joint model: exposure/interaction columns dropped: {missing}")
    n_clusters = d[cfg.cluster_col].nunique()

    hi = float(np.percentile(d[wb], cfg.hi_pctile))
    ref = float(np.percentile(d[wb], cfg.ref_pctile))
    design_info, shift = r["_design_map"][wb], r["_shifts"][wb]

    out_rows, tests = [], {}
    for m, lv in levels_by.items():
        R = np.zeros((len(block[m]), len(names)))
        for i, n in enumerate(block[m]):
            R[i, nidx[n]] = 1.0
        tests[m] = _wald_F(beta, V, R, n_clusters)
        for mode, cum in [("contemporaneous", False)] + ([("cumulative", True)] if cfg.lag_cols else []):
            w = _exposure_weights(design_info, shift, spline_cols, cfg.lag_cols, hi, ref, cum)
            for j, lvl in enumerate(lv[1:], start=1):
                v = np.zeros(len(names))
                for c, wt in w.items():
                    v[nidx[f"{c}__{m}{j}"]] = wt
                lr, se = float(v @ beta), float(np.sqrt(max(v @ V @ v, 0)))
                rr, lo, up, p = _z_ci(lr, se)
                out_rows.append({"indicator": ind, "modifier": m, "level": lvl, "reference": lv[0],
                                 "mode": mode, "adj_irr_ratio": rr, "adj_lo": lo, "adj_hi": up, "adj_p": p,
                                 "n_facilities": int(d.loc[d[f"em_{m}"] == lvl, "facility"].nunique())})
    if cont_mean is not None:
        R = np.zeros((len(block["mean_wbgt"]), len(names)))
        for i, n in enumerate(block["mean_wbgt"]):
            R[i, nidx[n]] = 1.0
        tests["mean_wbgt"] = _wald_F(beta, V, R, n_clusters)
        for mode, cum in _modes(cfg):
            w = _exposure_weights(design_info, shift, spline_cols, cfg.lag_cols, hi, ref, cum)
            v = np.zeros(len(names))
            for c, wt in w.items():
                v[nidx[zc_cols[c]]] = wt
            lr, se = float(v @ beta), float(np.sqrt(max(v @ V @ v, 0)))
            rr, lo, up, p = _z_ci(lr, se)
            out_rows.append({"indicator": ind, "modifier": "mean_wbgt", "level": "per +1 degC", "reference": "",
                             "mode": mode, "adj_irr_ratio": rr, "adj_lo": lo, "adj_hi": up, "adj_p": p,
                             "n_facilities": int(d["facility"].nunique())})
    return out_rows, tests, levels_by


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_effect_modification(results, cfg, fit_fn):
    for r in results:
        if "_nb_data" not in r:
            raise KeyError("fit_indicator must return '_nb_data': nb_data "
                           "(add this one line to its return dict)")
    os.makedirs(cfg.out_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    static = _static_modifiers(results, cfg)
    fixed_climatology = _load_fixed_climatology(cfg.climatology_file) if cfg.continuous else None
    static_mods = [m for m in ("level", "authority", "urban") if m in cfg.modifiers]
    fac_static = {m: static[m].to_dict() for m in static_mods}

    irr_all, strata_all, test_rows, joint_rows, cont_all = [], [], [], [], []
    for r in results:
        ind, nb = r["indicator"], r["_nb_data"]
        print(f"\n→ effect modification: {ind}")
        fkeys = set(nb["facility"].astype(str).str.strip())
        for m in static_mods:
            miss = fkeys - set(fac_static[m])
            if miss:
                raise ValueError(f"{ind}/{m}: {len(miss)} facilities not in metadata table, e.g. {sorted(miss)[:10]}")
        fac_groups = {m: fac_static[m] for m in static_mods}
        if "climatology" in cfg.modifiers:
            fac_groups["climatology"] = _climatology_groups(nb, cfg.wbgt_var).to_dict()

        for mod in cfg.modifiers:
            t0 = time.time()
            try:
                res = _run_one(ind, r, nb, mod, fac_groups[mod], cfg, fit_fn, rng)
                irr_all += res["irr"]
                strata_all.append(res["strata"])
                test_rows.append(res["test"])
                print(f"    {mod}: strata={res['levels']}  p_joint={res['test']['p_joint_exposure']:.3g}  "
                      f"({time.time() - t0:.0f}s)")
            except Exception as e:  # recorded, not silent
                print(f"    [{ind}/{mod}] FAILED: {e}")
                test_rows.append({"indicator": ind, "modifier": mod, "status": f"FAILED: {e}"})

        cont_mean = fixed_climatology if cfg.continuous else None
        if cont_mean is not None:
            t0 = time.time()
            try:
                crow, ctest = _run_continuous(ind, r, nb, cont_mean, cfg, fit_fn)
                cont_all += crow
                test_rows.append(ctest)
                print(f"    mean_wbgt (continuous): p_joint={ctest['p_joint_exposure']:.3g}  "
                      f"p_pooled={ctest['p_contrast_contemporaneous']:.3g}  "
                      f"p_common={ctest['p_common_contrast_contemporaneous']:.3g}  "
                      f"p_own={ctest['p_own_contrast_contemporaneous']:.3g}  ({time.time() - t0:.0f}s)")
            except Exception as e:  # recorded, not silent
                print(f"    [{ind}/mean_wbgt] FAILED: {e}")
                test_rows.append({"indicator": ind, "modifier": "mean_wbgt", "status": f"FAILED: {e}"})

        if cfg.run_joint:
            try:
                jr, jt, _ = _run_joint(ind, r, nb, fac_groups, cfg, fit_fn, cont_mean=cont_mean)
                joint_rows += jr
                for m, p in jt.items():
                    for t in test_rows:
                        if t.get("indicator") == ind and t.get("modifier") == m:
                            t["p_joint_exposure_adjusted"] = p
                print(f"    joint adjusted: " + ", ".join(f"{m} p={p:.3g}" for m, p in jt.items()))
            except Exception as e:
                print(f"    [{ind}/joint] FAILED: {e}")
                joint_rows.append({"indicator": ind, "modifier": "ALL", "status": f"FAILED: {e}"})

    tests = pd.DataFrame(test_rows)
    # Prespecified families: each hypothesis type across all indicators/modifiers.
    # Exploratory q-values; distinguish joint vs contrast tests in reporting.
    for pc in [c for c in tests.columns if c.startswith("p_")]:
        tests[pc.replace("p_", "q_", 1)] = _bh(tests[pc].values)

    tag = cfg.out_tag
    pd.DataFrame(irr_all).to_csv(f"{cfg.out_dir}em_irr_{tag}.csv", index=False)
    if cont_all:
        pd.DataFrame(cont_all).to_csv(f"{cfg.out_dir}em_continuous_{tag}.csv", index=False)
    pd.concat(strata_all).to_csv(f"{cfg.out_dir}em_strata_{tag}.csv", index=False) if strata_all else None
    tests.to_csv(f"{cfg.out_dir}em_tests_{tag}.csv", index=False)
    if joint_rows:
        pd.DataFrame(joint_rows).to_csv(f"{cfg.out_dir}em_joint_adjusted_{tag}.csv", index=False)
    print(f"\nEffect-modification outputs written to {cfg.out_dir} (em_*_{tag}.csv)")
    return tests


if __name__ == "__main__":
    import importlib

    M = importlib.import_module(MODEL_MODULE)
    results = []
    for ind in M.COUNT_INDICATORS:
        res = M.fit_indicator(ind, f"{M.PANEL_DIR}regression_panel_{ind}.csv")
        if res is not None:
            results.append(res)
    if not results:
        raise RuntimeError("No indicators fitted")

    ctrl = list(M.YEAR_FE_COLS) + ["covid"] + (["precip_c"] if M.USE_PRECIP else [])
    cfg = SimpleNamespace(
        wbgt_var=M.WBGT_VAR,
        lag_cols=[f"{M.WBGT_VAR}_lag{k}_c" for k in M.LAG_MONTHS] if M.SA_LAG else [],
        ctrl_cols=ctrl,
        cluster_col=M.CLUSTER_COL,
        district_col=M.DISTRICT_COL,
        use_zone_year=M.USE_ZONE_YEAR_FE,
        modifiers=EM_MODIFIERS,
        continuous=EM_CONTINUOUS, cont_pctiles=EM_CONT_PCTILES,
        hi_pctile=EM_HI_PCTILE, ref_pctile=EM_REF_PCTILE,
        n_draws=EM_N_DRAWS, chunk=EM_DRAW_CHUNK,
        min_facilities=EM_MIN_FACILITIES, min_hot_rows=EM_MIN_HOT_ROWS,
        run_joint=EM_RUN_JOINT, seed=EM_SEED,
        climatology_file=CLIMATOLOGY_FILE,
        out_dir=M.OUT_DIR,
        out_tag=f"{M.WBGT_VAR}{M.SUFFIX}{M.LAG_SUFFIX}",
    )
    run_effect_modification(results, cfg, fit_fn=M.fit_pois_absorbed)
