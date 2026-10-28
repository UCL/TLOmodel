#!/usr/bin/env python
"""
compare_era5_cmip6_wbgt.py

Compares facility-level WBGT5x used to FIT the models (ERA5, 2016-2024, from the
regression panels) with WBGT5x used for PROJECTIONS (NEX-GDDP-CMIP6, 2025-2040,
from wbgt_extreme_indices_facility_{tier}_{ssp}.csv). Reads existing files only.

Because the two periods do not overlap, the raw difference = model bias + warming.
To separate them, each CMIP6 series is detrended back to the ERA5 period midpoint
using its own 2025-2040 linear trend (facility-weighted annual means):

    total    = mean(CMIP6 2025-40) - mean(ERA5 2016-24)
    warming  = CMIP6 trend x (mid-year CMIP6 - mid-year ERA5)
    bias     = total - warming            (CMIP6 backcast to 2020 minus ERA5)

The same backcast is applied to every CMIP6 value to estimate the share of
facility-months above the ERA5 p95 that is due to bias alone.
This is an approximation: a 16-year trend from one realisation is noisy, so
treat the bias/warming split as indicative, and compare across tiers.

Outputs (in --out):
  summary_{ssp}.csv             one row per tier
  by_calendar_month_{ssp}.csv   mean ERA5, CMIP6 and difference by calendar month
  quantiles_{ssp}.csv           pooled facility-month quantiles
  era5_vs_cmip6_{ssp}.png       climatology, quantile differences, hot-month shares

Usage:
  python compare_era5_cmip6_wbgt.py --ssp ssp245
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

D = "/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/Indices/"
V = "wbgt5x_day"
TIERS = ["lowest", "median", "highest"]
ERA_YEARS = (2016, 2024)
CMIP_YEARS = (2025, 2040)
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]
MIN_COMMON_FACILITIES = 50


def load(path: Path, var: str, y0: int, y1: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    if "facility" not in df.columns and "facility_id" in df.columns:
        df = df.rename(columns={"facility_id": "facility"})
    for c in ("facility", "date", var):
        if c not in df.columns:
            raise KeyError(f"{path.name}: column {c!r} missing")
    df["facility"] = df["facility"].astype(str).str.strip()
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df = df[["facility", "date", var]].dropna()
    return df[df["date"].dt.year.between(y0, y1)]


def load_era5(d: Path) -> pd.DataFrame:
    """Union of all regression panels (WBGT is facility-level, identical across indicators)."""
    paths = sorted(d.glob("regression_panel_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No regression_panel_*.csv in {d}")
    parts = []
    for p in paths:
        try:
            parts.append(load(p, V, *ERA_YEARS))
        except KeyError:
            continue  # panel without wbgt5x_day
    if not parts:
        raise KeyError(f"No regression panel in {d} contains {V!r}")
    era = pd.concat(parts, ignore_index=True)
    # Same facility-month must have the same WBGT in every panel
    spread = era.groupby(["facility", "date"])[V].agg(lambda s: s.max() - s.min())
    if (spread > 1e-6).any():
        raise ValueError(f"{int((spread > 1e-6).sum())} facility-months have different {V} across panels")
    return era.drop_duplicates(["facility", "date"]).reset_index(drop=True)


def trend_per_year(df: pd.DataFrame) -> float:
    ann = df.groupby(df["date"].dt.year)[V].mean()
    return float(np.polyfit(ann.index.values, ann.values, 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssp", default="ssp245")
    ap.add_argument("--dir", default=D)
    ap.add_argument("--pctile", type=float, default=95.0)
    ap.add_argument("--out", default="era5_vs_cmip6_out")
    a = ap.parse_args()
    d, out = Path(a.dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    era_all = load_era5(d)
    cmip = {t: load(d / f"wbgt_extreme_indices_facility_{t}_{a.ssp}.csv", V, *CMIP_YEARS) for t in TIERS}

    common = set(era_all.facility)
    for t in TIERS:
        common &= set(cmip[t].facility)
    if len(common) < MIN_COMMON_FACILITIES:
        raise ValueError(f"Only {len(common)} facilities in common between ERA5 and CMIP6 -- check names")
    era = era_all[era_all.facility.isin(common)]
    cmip = {t: c[c.facility.isin(common)] for t, c in cmip.items()}
    print(f"{len(common)} common facilities; ERA5 {len(era):,} facility-months "
          f"({ERA_YEARS[0]}-{ERA_YEARS[1]})")

    thr = float(np.percentile(era[V], a.pctile))
    era_mid = era["date"].dt.year.mean()
    era_month = era.groupby(era.date.dt.month)[V].mean()

    rows, months, qs = [], {"era5": era_month}, {"era5": np.quantile(era[V], QUANTILES)}
    backcast_frames = {}
    for t in TIERS:
        c = cmip[t]
        slope = trend_per_year(c)
        c_mid = c["date"].dt.year.mean()
        total = c[V].mean() - era[V].mean()
        warming = slope * (c_mid - era_mid)
        bias = total - warming
        # CMIP6 shifted back to the ERA5 period using its own trend
        back = c.assign(**{V: c[V] - slope * (c["date"].dt.year - era_mid)})
        backcast_frames[t] = back

        fac = (c.groupby("facility")[V].mean() - era.groupby("facility")[V].mean()).dropna()
        hot_era = 100 * (era[V] > thr).mean()
        hot_back = 100 * (back[V] > thr).mean()
        hot_fut = 100 * (c[V] > thr).mean()
        rows.append({
            "ssp": a.ssp, "tier": t,
            "era5_mean": era[V].mean(), "cmip6_mean": c[V].mean(),
            "total_diff": total,
            "cmip6_trend_per_decade": 10 * slope,
            "warming_component": warming,
            "bias_component": bias,
            "facility_diff_p05": fac.quantile(0.05), "facility_diff_p95": fac.quantile(0.95),
            "era5_threshold": thr,
            "pct_hot_era5": hot_era,
            "pct_hot_cmip6_backcast": hot_back,
            "pct_hot_cmip6_2025_40": hot_fut,
            "pct_points_bias": hot_back - hot_era,
            "pct_points_warming": hot_fut - hot_back,
        })
        months[t] = c.groupby(c.date.dt.month)[V].mean()
        qs[t] = np.quantile(c[V], QUANTILES)

    res = pd.DataFrame(rows)
    res.to_csv(out / f"summary_{a.ssp}.csv", index=False)
    bm = pd.DataFrame(months)
    for t in TIERS:
        bm[f"{t}_minus_era5"] = bm[t] - bm["era5"]
    bm.index.name = "month"
    bm.to_csv(out / f"by_calendar_month_{a.ssp}.csv")
    qdf = pd.DataFrame(qs, index=QUANTILES)
    qdf.index.name = "quantile"
    qdf.to_csv(out / f"quantiles_{a.ssp}.csv")

    pd.set_option("display.width", 200)
    print("\n", res.drop(columns=["ssp"]).round(2).to_string(index=False))
    print("\nBy calendar month (degC):\n", bm.round(2))
    print("\nQuantiles (degC):\n", qdf.round(2))

    # ---- Figure ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(bm.index, bm["era5"], "k-", lw=2, label=f"ERA5 {ERA_YEARS[0]}-{ERA_YEARS[1]}")
    for t in TIERS:
        ax[0].plot(bm.index, bm[t], lw=1.2, label=f"CMIP6 {t} {CMIP_YEARS[0]}-{CMIP_YEARS[1]}")
    ax[0].set(title="WBGT5x by calendar month", xlabel="Month", ylabel="°C")
    ax[0].legend(fontsize=7)
    for t in TIERS:
        ax[1].plot(qdf.index, qdf[t] - qdf["era5"], "o-", ms=3, label=t)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set(title="CMIP6 minus ERA5, by quantile", xlabel="Quantile", ylabel="°C")
    ax[1].legend(fontsize=7)
    x = np.arange(len(TIERS))
    ax[2].bar(x - 0.25, res.pct_hot_era5, 0.25, label="ERA5", color="0.4")
    ax[2].bar(x, res.pct_hot_cmip6_backcast, 0.25, label="CMIP6 backcast (bias only)", color="#B2ABD2")
    ax[2].bar(x + 0.25, res.pct_hot_cmip6_2025_40, 0.25, label="CMIP6 2025-40", color="#B5515B")
    ax[2].set_xticks(x, TIERS)
    ax[2].set(title=f"% facility-months > ERA5 p{a.pctile:g} ({thr:.1f}°C)", ylabel="%")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f"era5_vs_cmip6_{a.ssp}.png", dpi=200)
    print(f"\nWritten to {out.resolve()}")


if __name__ == "__main__":
    main()
