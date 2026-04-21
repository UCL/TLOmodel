"""
proportion_disrupted_by_district.py

Computes the proportion of the population experiencing at least one weather
disruption (cancelled OR delayed) per year, by district, and plots a choropleth.

Numerator:   unique Person_IDs appearing in EITHER weather disruption log in a
             given year, attributed to the district of the facility they attended.
             Both logs are read together per run so Person_IDs are properly
             unioned — no double counting.
Denominator: alive model population by district of residence from the demography
             population log for the same year.

The per-year proportion is computed first (within each run), then averaged
across runs and years.

Also reports:
- Scaled mean annual disrupted people (× SCALING_FACTOR)
- Truly unique people ever disrupted across the full 2025–2040 period

Limitation: facility district is used as a proxy for residential district in
the numerator. Cross-district care-seeking introduces some geographic
misattribution, which should be noted in any write-up.
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from plot_configurations import (FS_TICK, FS_LABEL, FS_TITLE, FS_LEGEND,
                                 FS_PANEL, FS_SUPTITLE, SCENARIO_COLOURS,
                                 apply_style)

SCALING_FACTOR = 145.39


def _read_log(results_folder: Path, draw: int, run: int,
              module: str, log_key: str) -> pd.DataFrame:
    """Read one log key from a single draw/run, returning empty DataFrame on failure."""
    run_folder = results_folder / str(draw) / str(run)
    if not run_folder.exists():
        return pd.DataFrame()

    for suffix in [".pickle", ".pkl"]:
        fpath = run_folder / f"{module}__{log_key}{suffix}"
        if fpath.exists():
            with open(fpath, "rb") as f:
                return pickle.load(f)
        fpath = run_folder / f"{module}{suffix}"
        if fpath.exists():
            with open(fpath, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict) and log_key in data:
                return data[log_key]

    return pd.DataFrame()


def _get_n_runs(results_folder: Path, draw: int) -> int:
    draw_folder = results_folder / str(draw)
    if not draw_folder.exists():
        return 0
    return sum(1 for p in draw_folder.iterdir()
               if p.is_dir() and p.name.isdigit())


def _disrupted_persons_by_district(
    results_folder: Path, draw: int, run: int, year: int,
    fac_to_district: pd.Series,
) -> pd.Series:
    """
    Count unique Person_IDs with at least one disruption (cancelled OR delayed)
    in `year`, attributed to facility district. Both logs are read and
    Person_IDs are unioned before counting — guaranteed no double counting.
    """
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31")

    frames = []
    for key in ["Weather_cancelled_HSI_Event_full_info",
                "Weather_delayed_HSI_Event_full_info"]:
        df = _read_log(results_folder, draw, run,
                       "tlo.methods.healthsystem.summary", key)
        if df.empty:
            continue
        if "Person_ID" not in df.columns or "RealFacility_ID" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df.loc[df["date"].between(start, end)].copy()
        df = df[df["RealFacility_ID"].notna() & (df["RealFacility_ID"] != "unknown")]
        if not df.empty:
            frames.append(df[["Person_ID", "RealFacility_ID"]])

    if not frames:
        return pd.Series(dtype=float)

    combined = pd.concat(frames, ignore_index=True).drop_duplicates()
    combined["district"] = combined["RealFacility_ID"].map(fac_to_district)
    combined = combined.dropna(subset=["district"])
    return combined.groupby("district")["Person_ID"].nunique().astype(float)


def _population_by_district(
    results_folder: Path, draw: int, run: int, year: int,
) -> pd.Series:
    """
    Return mean alive population by district of residence for `year`.
    Averages across however many logging events fired within the year.
    """
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31")

    df = _read_log(results_folder, draw, run, "tlo.methods.demography", "population")
    if df.empty or "district_of_residence" not in df.columns:
        return pd.Series(dtype=float)

    df["date"] = pd.to_datetime(df["date"])
    df = df.loc[df["date"].between(start, end)]
    if df.empty:
        return pd.Series(dtype=float)

    totals = {}
    n = 0
    for _, row in df.iterrows():
        d = row["district_of_residence"]
        if isinstance(d, dict):
            for district, count in d.items():
                totals[district] = totals.get(district, 0) + count
            n += 1

    if n == 0:
        return pd.Series(dtype=float)

    return pd.Series({k: v / n for k, v in totals.items()}, dtype=float)


def _unique_disrupted_persons_all_years(
    results_folder: Path, draw: int, run: int,
    target_years: list, fac_to_district: pd.Series,
) -> int:
    """
    Return the count of unique Person_IDs disrupted at least once across ALL
    years in a single run. The same person disrupted in multiple years is
    counted only once — this is a true union across the full period.
    """
    all_persons = set()
    for year in target_years:
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year}-12-31")
        for key in ["Weather_cancelled_HSI_Event_full_info",
                    "Weather_delayed_HSI_Event_full_info"]:
            df = _read_log(results_folder, draw, run,
                           "tlo.methods.healthsystem.summary", key)
            if df.empty or "Person_ID" not in df.columns:
                continue
            df["date"] = pd.to_datetime(df["date"])
            df = df.loc[df["date"].between(start, end)]
            all_persons.update(df["Person_ID"].dropna().astype(int).tolist())
    return len(all_persons)


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path):
    apply_style()

    # ── CONFIG ────────────────────────────────────────────────────────────────
    min_year = 2025
    max_year = 2041
    main_text = True
    mode_2 = True
    CI_LOWER = 0.025
    CI_UPPER = 0.975

    if main_text:
        scenario_names = ["No disruptions", "Default", "Worst Case"]
        scenarios_of_interest = [0, 1, 2]
        suffix = "main_text_mode_2" if mode_2 else "main_text_mode_1"

    target_years = list(range(min_year, max_year))

    # ── FACILITY → DISTRICT MAPPING ───────────────────────────────────────────
    facilities_df = pd.read_csv(
        resourcefilepath / "climate_change_impacts" / "facilities_with_lat_long_region.csv",
        low_memory=False,
    )[["Fname", "Dist"]].drop_duplicates(subset="Fname")

    dist_fixes = {
        "Blanytyre": "Blantyre",
        "Nkhatabay": "Nkhata Bay",
        "Mzimba North": "Mzimba",
        "Mzimba South": "Mzimba",
    }
    facilities_df["Dist"] = facilities_df["Dist"].replace(dist_fixes)
    fac_to_district = facilities_df.set_index("Fname")["Dist"]

    # ── MAIN EXTRACTION LOOP ──────────────────────────────────────────────────
    all_scenario_results = {}

    for draw in scenarios_of_interest:
        scen = scenario_names[draw]
        print(f"\nDraw {draw}: {scen}")

        if scen == "No disruptions":
            all_scenario_results[scen] = None
            continue

        n_runs = _get_n_runs(results_folder, draw)
        if n_runs == 0:
            print(f"  No runs found for draw {draw}, skipping.")
            all_scenario_results[scen] = None
            continue

        print(f"  {n_runs} runs found")

        run_mean_proportions = []
        run_country_disrupted = []
        run_country_population = []

        for run in range(n_runs):
            yearly_prop = []
            yearly_disrupted_total = []
            yearly_population_total = []

            for year in target_years:
                disrupted = _disrupted_persons_by_district(
                    results_folder, draw, run, year, fac_to_district)
                population = _population_by_district(
                    results_folder, draw, run, year)

                if disrupted.empty or population.empty:
                    continue

                districts = disrupted.index.union(population.index)
                disrupted = disrupted.reindex(districts, fill_value=0)
                population = population.reindex(districts, fill_value=0)

                with np.errstate(divide="ignore", invalid="ignore"):
                    prop = (
                        disrupted
                        .div(population.where(population > 0, np.nan))
                        .fillna(0)
                        .clip(upper=1.0)
                    )
                yearly_prop.append(prop)
                yearly_disrupted_total.append(disrupted.sum())
                yearly_population_total.append(population.sum())

            if not yearly_prop:
                continue

            run_mean = pd.concat(yearly_prop, axis=1).mean(axis=1)
            run_mean_proportions.append(run_mean)

            mean_disrupted = np.mean(yearly_disrupted_total)
            mean_population = np.mean(yearly_population_total)
            run_country_disrupted.append(mean_disrupted)
            run_country_population.append(mean_population)

            print(f"    Run {run}: country mean proportion = {run_mean.mean() * 100:.3f}%  |  "
                  f"mean annual disrupted (unscaled) = {mean_disrupted:,.0f}  |  "
                  f"mean annual disrupted (scaled)   = {mean_disrupted * SCALING_FACTOR:,.0f}  |  "
                  f"mean annual population (scaled)  = {mean_population * SCALING_FACTOR:,.0f}")

        if not run_mean_proportions:
            all_scenario_results[scen] = None
            continue

        runs_df = pd.concat(run_mean_proportions, axis=1)

        country_disrupted_arr = np.array(run_country_disrupted)
        country_population_arr = np.array(run_country_population)
        country_prop_arr = country_disrupted_arr / np.where(
            country_population_arr > 0, country_population_arr, np.nan
        )

        all_scenario_results[scen] = {
            "mean": runs_df.mean(axis=1),
            "lower": runs_df.quantile(CI_LOWER, axis=1),
            "upper": runs_df.quantile(CI_UPPER, axis=1),
            "country_disrupted_mean": float(np.nanmean(country_disrupted_arr)),
            "country_disrupted_lower": float(np.nanpercentile(country_disrupted_arr, CI_LOWER * 100)),
            "country_disrupted_upper": float(np.nanpercentile(country_disrupted_arr, CI_UPPER * 100)),
            "country_population_mean": float(np.nanmean(country_population_arr)),
            "country_prop_mean": float(np.nanmean(country_prop_arr)),
            "country_prop_lower": float(np.nanpercentile(country_prop_arr, CI_LOWER * 100)),
            "country_prop_upper": float(np.nanpercentile(country_prop_arr, CI_UPPER * 100)),
        }

    # ── UNIQUE PEOPLE ACROSS FULL PERIOD ──────────────────────────────────────
    # True union of Person_IDs across all years per run — a person disrupted
    # in multiple years is counted only once.
    print("\n── Unique people ever disrupted across full period ───────────────")
    unique_results = {}
    for draw in scenarios_of_interest:
        scen = scenario_names[draw]
        if scen == "No disruptions":
            continue
        n_runs = _get_n_runs(results_folder, draw)
        counts_scaled = []
        for run in range(n_runs):
            n = _unique_disrupted_persons_all_years(
                results_folder, draw, run, target_years, fac_to_district)
            counts_scaled.append(n * SCALING_FACTOR)
        counts_scaled = np.array(counts_scaled)
        unique_results[scen] = {
            "mean": float(np.mean(counts_scaled)),
            "lower": float(np.quantile(counts_scaled, CI_LOWER)),
            "upper": float(np.quantile(counts_scaled, CI_UPPER)),
        }
        print(f"  {scen}: {np.mean(counts_scaled):,.0f} unique people ever disrupted "
              f"[{np.quantile(counts_scaled, CI_LOWER):,.0f}–"
              f"{np.quantile(counts_scaled, CI_UPPER):,.0f}] "
              f"over {len(target_years)}-year period (scaled)")

    # ── CSV: district-level ───────────────────────────────────────────────────
    rows = []
    for scen, result in all_scenario_results.items():
        if result is None:
            continue
        for district in result["mean"].index:
            rows.append({
                "Scenario": scen,
                "district": district,
                "prop_disrupted_mean": round(result["mean"][district], 6),
                "prop_disrupted_lower": round(result["lower"][district], 6),
                "prop_disrupted_upper": round(result["upper"][district], 6),
                "pct_disrupted_mean": round(result["mean"][district] * 100, 4),
                "pct_disrupted_lower": round(result["lower"][district] * 100, 4),
                "pct_disrupted_upper": round(result["upper"][district] * 100, 4),
            })
    out_csv = output_folder / f"proportion_disrupted_by_district_{suffix}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nDistrict CSV saved: {out_csv}")

    # ── CSV: country-level summary (scaled) ───────────────────────────────────
    country_rows = []
    for scen, result in all_scenario_results.items():
        if result is None:
            continue
        unique = unique_results.get(scen, {})
        country_rows.append({
            "Scenario": scen,
            # Mean annual disrupted people (scaled to real population)
            "mean_annual_disrupted_mean_scaled": round(result["country_disrupted_mean"] * SCALING_FACTOR, 0),
            "mean_annual_disrupted_lower_scaled": round(result["country_disrupted_lower"] * SCALING_FACTOR, 0),
            "mean_annual_disrupted_upper_scaled": round(result["country_disrupted_upper"] * SCALING_FACTOR, 0),
            # Mean annual population (scaled)
            "mean_annual_population_scaled": round(result["country_population_mean"] * SCALING_FACTOR, 0),
            # % of population disrupted per year
            "country_pct_disrupted_mean": round(result["country_prop_mean"] * 100, 4),
            "country_pct_disrupted_lower": round(result["country_prop_lower"] * 100, 4),
            "country_pct_disrupted_upper": round(result["country_prop_upper"] * 100, 4),
            # Unique people ever disrupted across full period (scaled)
            "unique_ever_disrupted_mean": round(unique.get("mean", np.nan), 0),
            "unique_ever_disrupted_lower": round(unique.get("lower", np.nan), 0),
            "unique_ever_disrupted_upper": round(unique.get("upper", np.nan), 0),
            "n_years": len(target_years),
        })
    out_country_csv = output_folder / f"country_disrupted_totals_{suffix}.csv"
    pd.DataFrame(country_rows).to_csv(out_country_csv, index=False)
    print(f"Country CSV saved: {out_country_csv}")

    # Print summary to console
    print("\n── Country-level summary (scaled to real population) ─────────────")
    for row in country_rows:
        print(
            f"  {row['Scenario']}:\n"
            f"    Mean annual disrupted:  {row['mean_annual_disrupted_mean_scaled']:,.0f} people/year "
            f"[{row['mean_annual_disrupted_lower_scaled']:,.0f}–"
            f"{row['mean_annual_disrupted_upper_scaled']:,.0f}]\n"
            f"    % of population:        {row['country_pct_disrupted_mean']:.2f}% "
            f"[{row['country_pct_disrupted_lower']:.2f}–{row['country_pct_disrupted_upper']:.2f}]\n"
            f"    Unique ever disrupted:  {row['unique_ever_disrupted_mean']:,.0f} people "
            f"[{row['unique_ever_disrupted_lower']:,.0f}–"
            f"{row['unique_ever_disrupted_upper']:,.0f}] "
            f"over {row['n_years']} years"
        )

    # ── MAP ───────────────────────────────────────────────────────────────────
    malawi = gpd.read_file(
        resourcefilepath / "mapping" / "ResourceFile_mwi_admbnda_adm2_nso_20181016.shp"
    )
    for old, new in [
        ("Blantyre City", "Blantyre"), ("Mzuzu City", "Mzuzu"),
        ("Lilongwe City", "Lilongwe"), ("Zomba City", "Zomba"),
    ]:
        malawi["ADM2_EN"] = malawi["ADM2_EN"].replace(old, new)

    map_draws = [
        d for d in scenarios_of_interest
        if scenario_names[d] != "No disruptions"
           and all_scenario_results.get(scenario_names[d]) is not None
    ]
    if not map_draws:
        print("No disruption scenarios to map.")
        return

    all_means = pd.concat([
        all_scenario_results[scenario_names[d]]["mean"] for d in map_draws
    ])
    vmax_pct = max(np.ceil(all_means.max() * 1000) / 10, 0.1)

    panel_labels = list("ABCDEF")

    fig, axes = plt.subplots(1, len(map_draws),
                             figsize=(10 * len(map_draws), 8))
    if len(map_draws) == 1:
        axes = [axes]

    for i, (ax, draw) in enumerate(zip(axes, map_draws)):
        scen = scenario_names[draw]
        result = all_scenario_results[scen]

        malawi["pct"] = malawi["ADM2_EN"].map(result["mean"] * 100)

        malawi.plot(
            column="pct", ax=ax, legend=True, cmap="Oranges",
            edgecolor="black",
            vmin=0, vmax=vmax_pct,
            legend_kwds={
                "label": "% of district population\nwith ≥1 disruption per year",
                "shrink": 0.7,
            },
            missing_kwds={"color": "lightgrey", "label": "No data"},
        )

        ax.set_title(f"({panel_labels[i]}) {scen}", fontsize=FS_TITLE, fontweight="bold")
        fig.axes[-1].set_ylabel(
            "% population with ≥1 disruption/year",
            fontsize=FS_LABEL, fontweight="bold",
        )
        fig.axes[-1].tick_params(labelsize=FS_TICK)
        ax.axis("off")

    fig.tight_layout()
    out_map = output_folder / f"map_proportion_disrupted_by_district_{suffix}.png"
    fig.savefig(out_map, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nMap saved: {out_map}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    parser.add_argument("--output_folder", type=Path, default=None)
    parser.add_argument("--resourcefilepath", type=Path,
                        default=Path("./resources"))
    args = parser.parse_args()

    apply(
        results_folder=args.results_folder,
        output_folder=args.output_folder or args.results_folder,
        resourcefilepath=args.resourcefilepath,
    )
