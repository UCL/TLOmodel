"""
District-level analysis for the Nurse Shortages project.

Outputs (written to the scenario outputs folder):
  Excel
    - district_staff_counts_2026_2027.xlsx
    - district_staff_scaling.xlsx
    - district_staff_scaling_health_outcomes.xlsx
    - district_deaths_dalys_barplot_data.xlsx  (% sheets, then absolute "_abs" sheets)
    - district_vs_national_validation.xlsx     (with --save-figures)
  Figures (with --save-figures)
    - Maps of % and absolute DALYs/deaths averted; only districts with a
      significant effect are coloured, the rest are grey.
    - Bar plots of absolute and % DALYs/deaths averted by district
      (two panels side by side).
    - Staff scale-up factors and change in workforce size by district.

All "averted" results compare each scenario with the baseline of the same
health-system setting, run by run, over 2027–2034.
"""

import argparse
import math
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

from scripts.nurses_analyses.nurses_scenario_analyses import StaffingScenario
from tlo.analysis.utils import compute_summary_statistics, extract_results


def summarize(results):
    """Mean and 95% CI of the mean (mean ± 1.96 × SE), same layout as TLO's summarize."""
    return compute_summary_statistics(
        results, central_measure="mean", use_standard_error=True
    ).rename(columns={"central": "mean"}, level=1)


DALY_DEATH_METADATA_COLUMNS = {"date", "year", "sex", "age_range", "li_wealth", "district_of_residence"}

COMPARISON_YEARS = (2027, 2034)

# Likoma results are combined with Nkhata Bay throughout.
DISTRICT_MERGES = {"Likoma": "Nkhata Bay"}

HEALTHSYSTEMS = ["Default", "Improved"]

# (scenario prefix, short label, colour); the order is the plotting order.
SCENARIOS = [
    ("Baseline Nurses", "Baseline", "black"),
    ("Fewer Nurses", "Fewer nurses", "indianred"),
    ("More Nurses", "More nurses", "steelblue"),
    ("More CNP staff", "More CNP", "darkgreen"),
    ("More Nurses by District", "More nurses by district", "mediumpurple"),
    ("More CNP staff by District", "More CNP by district", "orange"),
]
COMPARISON_SCENARIOS = [s for s in SCENARIOS if s[0] != "Baseline Nurses"]

# Map panel order
MAP_SCENARIO_PREFIXES = [
    "More Nurses",
    "More CNP staff",
    "Fewer Nurses",
    "More Nurses by District",
    "More CNP staff by District",
]

NOT_SIGNIFICANT_COLOUR = "lightgrey"

OUTCOMES = {
    # outcome: (log module, log key, singular label used in validation sheet names)
    "DALYs": ("tlo.methods.healthburden", "dalys_stacked", "DALY"),
    "Deaths": ("tlo.methods.demography", "death", "Death"),
}


def scenario_name(prefix, healthsystem):
    return f"{prefix} / {healthsystem} Healthsystem Function"


def short_label(name):
    prefix = name.split(" / ")[0]
    return next(label for p, label, _ in SCENARIOS if p == prefix)


# ---------------------------------------------------------------------------
# Helpers for comparing scenarios
# ---------------------------------------------------------------------------

def find_difference_relative_to_comparison_series(
    _ser: pd.Series,
    comparison: str,
    scaled: bool = False,
    drop_comparison: bool = True,
):
    return (
        _ser.unstack(level=0)
        .apply(
            lambda x: (x - x[comparison]) / (x[comparison] if scaled else 1.0),
            axis=1,
        )
        .drop(columns=([comparison] if drop_comparison else []))
        .stack()
    )


def find_difference_relative_to_comparison_series_dataframe(_df: pd.DataFrame, **kwargs):
    return pd.concat(
        {idx: find_difference_relative_to_comparison_series(row, **kwargs) for idx, row in _df.iterrows()},
        axis=1,
    ).T


def set_param_names_as_column_index_level_0(_df, param_names):
    """Set column index level 0 (draw numbers) to scenario names."""
    ordered_param_names = {i: x for i, x in enumerate(param_names)}
    names_of_cols_level0 = [ordered_param_names.get(col) for col in _df.columns.levels[0]]
    _df.columns = _df.columns.set_levels(names_of_cols_level0, level=0)
    return _df


def calculate_averted_by_district(outcome_by_district, baseline_scenario, percent):
    """
    DALYs/deaths averted by district relative to the baseline, from run-to-run
    differences, then summarised.

    percent=True  -> 100 * (baseline - scenario) / baseline
    percent=False -> baseline - scenario

    Returns {scenario: DataFrame indexed by district with mean/lower/upper}.
    """
    diff = (-100.0 if percent else -1.0) * find_difference_relative_to_comparison_series_dataframe(
        outcome_by_district,
        comparison=baseline_scenario,
        scaled=percent,
    )
    summarized = summarize(diff)
    return {
        scenario: pd.DataFrame({stat: summarized[(scenario, stat)] for stat in ["mean", "lower", "upper"]})
        for scenario in summarized.columns.get_level_values(0).unique()
    }


def is_significant(result_df):
    """
    Significant if the effect is positive and the lower bound is not negative,
    or the effect is negative and the upper bound is not positive.
    """
    return (
        ((result_df["mean"] > 0) & (result_df["lower"] >= 0))
        | ((result_df["mean"] < 0) & (result_df["upper"] <= 0))
    )


def means_long(results_dicts, value_name):
    """Means from one or more result dicts, as long format District/Scenario/value."""
    wide = pd.concat(
        [pd.DataFrame({scenario: df["mean"] for scenario, df in results.items()}) for results in results_dicts],
        axis=1,
    )
    long = wide.stack().reset_index()
    long.columns = ["District", "Scenario", value_name]
    return long


# ---------------------------------------------------------------------------
# Extraction: DALYs and deaths
# ---------------------------------------------------------------------------

def _cause_columns(df):
    return [
        c for c in df.columns
        if c not in DALY_DEATH_METADATA_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]


def extract_national_annual(results_folder, outcome):
    """National total for each year (used for validation)."""
    module, key, _ = OUTCOMES[outcome]

    def generate(df):
        # Use the logged year if present.
        if "year" not in df.columns:
            df = df.assign(year=df["date"].dt.year)
        if outcome == "DALYs":
            return df.groupby("year")[_cause_columns(df)].sum().sum(axis=1)
        return df.groupby("year")["person_id"].count()

    return extract_results(results_folder, module=module, key=key, custom_generate_series=generate, do_scaling=True)


def extract_by_district(results_folder, outcome, by_year):
    """
    Totals by district of residence.
    by_year=True  -> by year and district, all years (used for validation)
    by_year=False -> summed over COMPARISON_YEARS (used for the analysis)
    """
    module, key, _ = OUTCOMES[outcome]

    def generate(df):
        if outcome == "DALYs" or not by_year or "year" not in df.columns:
            df = df.assign(year=df["date"].dt.year)
        if not by_year:
            df = df[df["year"].between(*COMPARISON_YEARS)]
        df = df.assign(district_of_residence=df["district_of_residence"].replace(DISTRICT_MERGES))
        group_cols = ["year", "district_of_residence"] if by_year else "district_of_residence"
        if outcome == "DALYs":
            total_dalys = df[_cause_columns(df)].sum(axis=1)
            return df.assign(total_dalys=total_dalys).groupby(group_cols)["total_dalys"].sum()
        return df.groupby(group_cols)["person_id"].count()

    return extract_results(results_folder, module=module, key=key, custom_generate_series=generate, do_scaling=True)


def validate_district_totals(national, by_district_and_year, outcome):
    """Check district totals equal national totals for every scenario, draw and run."""
    district_totals = by_district_and_year.groupby(level="year").sum()
    comparison = pd.concat(
        {"National": national, "District Total": district_totals, "Difference": national - district_totals},
        axis=1,
    )
    label = OUTCOMES[outcome][2].upper()
    print(f"\n-- {label} VALIDATION --\n")
    print(comparison)
    assert np.allclose(national.values, district_totals.values)
    print(f"\n{label} validation passed.\n")
    return district_totals, comparison


# ---------------------------------------------------------------------------
# Extraction: staff
# ---------------------------------------------------------------------------

def get_yearly_hr_count(_df):
    if "GenericClinic" not in _df.columns:
        return None

    years = _df["date"].dt.year.rename("year")

    # Expand facility dictionary; column names hold facility ID and cadre.
    staff_df = _df["GenericClinic"].apply(pd.Series)
    facility_ids = [int(c.split("FacilityID_")[1].split("_")[0]) for c in staff_df.columns]
    cadres = [c.split("Officer_")[-1] for c in staff_df.columns]

    mfl = pd.read_csv(
        Path("./resources/healthsystem/organisation/ResourceFile_Master_Facilities_List.csv")
    ).set_index("Facility_ID")

    # Facilities at levels 3+ have no district; use their names so they are not dropped.
    for fid in {128, 129, 130, 131, 132}:
        mfl.loc[fid, "District"] = mfl.loc[fid, "Facility_Name"]

    districts = [mfl.loc[fid, "District"] if fid in mfl.index else "Unknown" for fid in facility_ids]
    districts = [DISTRICT_MERGES.get(d, d) for d in districts]

    staff_df.columns = pd.MultiIndex.from_arrays([districts, cadres], names=["District", "Cadre"])

    # Sum by year, then over facilities within district/cadre.
    staff_df = staff_df.groupby(years).sum()
    staff_df = staff_df.T.groupby(level=[0, 1]).sum().T

    return staff_df.stack([0, 1])


def extract_staff_counts(results_folder):
    return extract_results(
        results_folder,
        module="tlo.methods.healthsystem.summary",
        key="number_of_hcw_staff",
        custom_generate_series=get_yearly_hr_count,
        do_scaling=False,
    )


def calculate_staff_counts(staff_mean, years=(2026, 2027)):
    """Mean staff counts for Clinical, Nursing_and_Midwifery and Pharmacy by scenario, district and year."""
    rows = []
    for scenario in staff_mean.columns:
        for cadre in ["Clinical", "Nursing_and_Midwifery", "Pharmacy"]:
            cadre_staff = staff_mean[scenario].xs(cadre, level="Cadre")
            cadre_staff = cadre_staff.loc[cadre_staff.index.get_level_values("year").isin(years)]
            for (year, district), staff_number in cadre_staff.items():
                rows.append({
                    "Scenario": scenario,
                    "Cadre": cadre,
                    "District": district,
                    "Year": int(year),
                    "Staff": staff_number,
                })
    return pd.DataFrame(rows)


def add_scaleup_factors(by_year):
    """Add 2027:2024 and 2027:2019 scale-up factors to a table with year columns."""
    by_year["Scale2027_2024"] = by_year[2027] / by_year[2024]
    by_year["Scale2027_2019"] = by_year[2027] / by_year[2019]
    return by_year


def order_districts(df, district_order):
    """Keep districts in district_order, as an ordered categorical."""
    df = df[df["District"].isin(district_order)].copy()
    df["District"] = pd.Categorical(df["District"], categories=district_order, ordered=True)
    return df


def staff_scaling_summary(staff_mean, scenarios, staff_type, cadres):
    """
    Staff in 2019, 2024 and 2027 and scale-up factors by district,
    summing the given cadres.
    """
    selected = staff_mean.loc[staff_mean.index.get_level_values("Cadre").isin(cadres)]
    totals = selected.groupby(level=["District", "year"]).sum()

    rows = []
    for scenario in scenarios:
        by_year = add_scaleup_factors(totals[scenario].unstack(level="year")[[2019, 2024, 2027]])
        by_year = by_year.rename(columns={2019: "Staff2019", 2024: "Staff2024", 2027: "Staff2027"})
        by_year["Scenario"] = scenario
        rows.append(by_year.reset_index())

    summary = pd.concat(rows, ignore_index=True)
    summary["Staff_Type"] = staff_type
    return summary


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

# Scenario/cadre combinations for the staff plots; order sets the legend order.
STAFF_LINE_COMBINATIONS = [
    ("Baseline Nurses", "Nursing_and_Midwifery", "Baseline nurses", "black"),
    ("Fewer Nurses", "Nursing_and_Midwifery", "Fewer nurses", "indianred"),
    ("More Nurses", "Nursing_and_Midwifery", "More nurses", "steelblue"),
    ("More CNP staff", "Clinical", "More CNP - Clinical", "darkgreen"),
    ("More CNP staff", "Pharmacy", "More CNP - Pharmacy", "darkorange"),
    ("More Nurses by District", "Nursing_and_Midwifery", "More nurses by district", "mediumpurple"),
    ("More CNP staff by District", "Clinical", "More CNP by district - Clinical", "limegreen"),
    ("More CNP staff by District", "Pharmacy", "More CNP by district - Pharmacy", "gold"),
]


def _legend_in_columns(ax, ncol, bbox_y):
    """Legend below the plot, filled column by column."""
    handles, labels = ax.get_legend_handles_labels()
    order = [i for c in range(ncol) for i in range(c, len(labels), ncol)]
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper center",
        bbox_to_anchor=(0.5, bbox_y),
        ncol=ncol,
        numpoints=1,
        handlelength=2.5,
        frameon=False,
    )


def plot_staff_change_lines_by_district(
    staff_df,
    district_order,
    healthsystem="Default",
    year_before=2026,
    year_after=2027,
    symlog_threshold=20,
):
    """Change in workforce size (year_after - year_before) by cadre and district."""
    x = np.arange(len(district_order))
    fig, ax = plt.subplots(figsize=(16, 8))

    for prefix, cadre, label, color in STAFF_LINE_COMBINATIONS:
        scen_df = staff_df[
            (staff_df["Scenario"] == scenario_name(prefix, healthsystem)) & (staff_df["Cadre"] == cadre)
        ]
        if scen_df.empty:
            continue

        by_year = scen_df.pivot_table(
            index="District", columns="Year", values="Staff", aggfunc="sum"
        ).reindex(index=district_order, columns=[year_before, year_after])
        change = by_year[year_after] - by_year[year_before]

        ax.plot(
            x,
            change,
            color=to_rgba(color, 0.5),
            linewidth=0.8,
            linestyle="--",
            marker="o",
            markersize=5,
            markerfacecolor=to_rgba(color, 0.85),
            markeredgecolor=to_rgba(color, 0.85),
            label=label,
        )

    ax.axhline(0, linestyle="--", linewidth=1)

    # Symmetrical log scale
    ax.set_yscale("symlog", linthresh=symlog_threshold, linscale=0.8)
    ticks = [-500, -200, -100, -50, -20, -10, -5, 0, 5, 10, 20, 50, 100, 200, 500]
    low, high = ax.get_ylim()
    ax.set_yticks([t for t in ticks if low <= t <= high])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_locator(plt.NullLocator())

    ax.set_xticks(x)
    ax.set_xticklabels(district_order, rotation=90)
    ax.set_xlabel("District")
    ax.set_ylabel(f"Change in number of staff ({year_after} - {year_before})")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"Change in workforce size by cadre and district, {year_before} to {year_after}")

    _legend_in_columns(ax, ncol=4, bbox_y=-0.32)
    fig.subplots_adjust(bottom=0.38, right=0.98)
    return fig


def plot_staff_scaleup_factors_by_district(scaleup_plot_data):
    """
    Staff scale-up factors by district, Default health system only
    (Improved scenarios have the same scale-up factors).
    """
    districts = sorted(scaleup_plot_data["District"].unique())

    figures = {}
    for scaleup_column, ylabel in [
        ("Scale2027_2024", "Workforce size scale-up factor (2027:2024)"),
        ("Scale2027_2019", "Workforce size scale-up factor (2027:2019)"),
    ]:
        fig, ax = plt.subplots(figsize=(16, 8))

        for prefix, cadre, label, color in STAFF_LINE_COMBINATIONS:
            scenario_data = scaleup_plot_data[
                (scaleup_plot_data["Scenario"] == scenario_name(prefix, "Default"))
                & (scaleup_plot_data["Cadre"] == cadre)
            ]
            if scenario_data.empty:
                continue

            scenario_data = scenario_data.set_index("District").reindex(districts).reset_index()
            ax.plot(
                scenario_data["District"],
                scenario_data[scaleup_column],
                color=color,
                marker="o",
                label=label,
                alpha=0.7,
                linestyle="-",
                linewidth=1,
                markersize=5,
            )

        # No scale-up relative to the comparison year
        ax.axhline(1.0, linestyle="--", linewidth=1.5)
        ax.set_xlabel("District")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=90)

        _legend_in_columns(ax, ncol=4, bbox_y=-0.25)
        figures[scaleup_column] = fig

    return figures


def plot_district_maps(district_map, results, healthsystem, title, colorbar_label):
    """
    One map per scenario. Only districts with a significant effect are
    coloured, and the colour scale is set from those values only;
    the other districts are grey.
    """
    scenario_names = [scenario_name(p, healthsystem) for p in MAP_SCENARIO_PREFIXES]

    gdf = district_map.copy()
    district_key = gdf["ADM2_EN"].replace(DISTRICT_MERGES)
    for scenario in scenario_names:
        significant_means = results[scenario]["mean"].where(is_significant(results[scenario]))
        gdf[scenario] = district_key.map(significant_means)

    values = gdf[scenario_names].values.astype(float)
    vmax = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
    vmax = vmax or 1.0  # avoid a zero-width colour scale

    ncols = 3
    nrows = math.ceil(len(scenario_names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 9), constrained_layout=True)
    axes = np.array(axes).flatten()

    for ax, scenario in zip(axes, scenario_names):
        gdf.plot(
            column=scenario,
            cmap="coolwarm_r",
            edgecolor="black",
            linewidth=0.4,
            legend=False,
            vmin=-vmax,
            vmax=vmax,
            ax=ax,
            missing_kwds={"color": NOT_SIGNIFICANT_COLOUR, "edgecolor": "black", "linewidth": 0.4},
        )
        ax.set_title(short_label(scenario), fontsize=12)
        ax.axis("off")

    for ax in axes[len(scenario_names):]:
        ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap="coolwarm_r", norm=plt.Normalize(-vmax, vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, location="right", shrink=0.85, pad=0.02, label=colorbar_label)
    fig.legend(
        handles=[Patch(facecolor=NOT_SIGNIFICANT_COLOUR, edgecolor="black", label="Not significant")],
        loc="lower right",
        frameon=False,
    )
    fig.suptitle(title, fontsize=15)
    return fig


def plot_averted_by_district(absolute_results, percent_results, healthsystem, outcome, district_order):
    """
    Absolute (left) and % (right) DALYs/deaths averted by district, in the
    same layout as the age-group plots. No value annotations.
    """
    bar_height = 0.15
    offsets = (np.arange(len(COMPARISON_SCENARIOS)) - (len(COMPARISON_SCENARIOS) - 1) / 2) * bar_height
    y = np.arange(len(district_order))

    fig, axes = plt.subplots(ncols=2, figsize=(19, 16), sharey=True)
    ax_absolute, ax_percent = axes

    for panel_index, (ax, panel_data, xlabel) in enumerate([
        (ax_absolute, absolute_results, f"{outcome} averted"),
        (ax_percent, percent_results, f"% {outcome} averted"),
    ]):
        for offset, (prefix, label, color) in zip(offsets, COMPARISON_SCENARIOS):
            # Reverse so the first district is at the top.
            df = panel_data[scenario_name(prefix, healthsystem)].reindex(district_order).iloc[::-1]
            means, lowers, uppers = (df[c].to_numpy() for c in ["mean", "lower", "upper"])
            ypos = y + offset

            ax.barh(ypos, means, height=bar_height * 0.9, color=color, label=label if panel_index == 0 else None)
            ax.errorbar(
                means,
                ypos,
                xerr=np.vstack([means - lowers, uppers - means]),
                fmt="none",
                elinewidth=1,
                capsize=1.5,
                color="black",
                alpha=0.4,
            )

        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_yticks(y)
        ax.set_yticklabels(district_order[::-1])

    ax_absolute.set_title(f"Absolute {outcome} averted")
    ax_percent.set_title(f"Percentage {outcome} averted")
    ax_absolute.set_ylabel("District")

    handles, labels = ax_absolute.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=5, frameon=True)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.09, wspace=0.12)
    return fig


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------

def _format_worksheet(worksheet):
    """Freeze the header, add filters and fit column widths."""
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    for column_cells in worksheet.columns:
        max_length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
        worksheet.column_dimensions[column_cells[0].column_letter].width = min(max_length + 2, 45)


def export_district_outcomes_to_excel(results_folder, plot_data):
    """
    Export the data underlying the district bar charts, one worksheet each.
    plot_data: list of (sheet_name, {scenario: DataFrame}, outcome, unit),
    unit being " (%)" for percentages or "" for absolute numbers.
    """
    output_file = results_folder / "district_deaths_dalys_barplot_data.xlsx"

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for sheet_name, scenario_data, outcome, unit in plot_data:
            frames = []
            for scenario, district_df in scenario_data.items():
                temp = district_df.copy()
                temp.index.name = "District"
                temp = temp.reset_index()
                temp.insert(0, "Scenario", scenario)
                temp["Lower error"] = temp["mean"] - temp["lower"]
                temp["Upper error"] = temp["upper"] - temp["mean"]
                frames.append(temp)

            excel_df = (
                pd.concat(frames, ignore_index=True)
                if frames
                else pd.DataFrame(columns=["Scenario", "District", "mean", "lower", "upper",
                                           "Lower error", "Upper error"])
            ).rename(columns={
                "mean": f"Mean {outcome} averted{unit}",
                "lower": f"Lower bound{unit}",
                "upper": f"Upper bound{unit}",
            })

            excel_df.to_excel(writer, sheet_name=sheet_name, index=False)
            _format_worksheet(writer.sheets[sheet_name])

    print(f"District bar-plot data saved to: {output_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Analyse DALYs/Deaths across nurse staffing scenarios by district")
    parser.add_argument("--scenario-outputs-folder", type=Path, required=True,
                        help="Path to folder containing scenario outputs")
    parser.add_argument("--show-figures", action="store_true", help="Whether to interactively show figures")
    parser.add_argument("--save-figures", action="store_true", help="Whether to save figures to results folder")
    args = parser.parse_args()

    results_folder = args.scenario_outputs_folder
    param_names = tuple(StaffingScenario()._scenarios.keys())

    # Malawi district boundaries; Likoma is combined with Nkhata Bay.
    district_map = gpd.read_file(Path("resources/mapping/ResourceFile_mwi_admbnda_adm2_nso_20181016.shp"))
    district_order = [d for d in district_map["ADM2_EN"].tolist() if d not in DISTRICT_MERGES]

    figures = {}  # file name (without .pdf) -> figure

    # ---- Staff -------------------------------------------------------------
    staff_counts = set_param_names_as_column_index_level_0(extract_staff_counts(results_folder), param_names)
    staff_mean = summarize(staff_counts).xs("mean", axis=1, level=1)

    # Staff counts by district, 2026 vs 2027
    staff_counts_by_district = order_districts(calculate_staff_counts(staff_mean), district_order)
    staff_counts_by_district = staff_counts_by_district.sort_values(["Scenario", "Cadre", "District", "Year"])
    staff_counts_by_district.to_excel(results_folder / "district_staff_counts_2026_2027.xlsx", index=False)

    figures["staff_change_by_district_default_healthsystem"] = plot_staff_change_lines_by_district(
        staff_counts_by_district, district_order, healthsystem="Default",
    )

    # Scale-up factors by district and cadre (Default health system)
    scaleup_cadres = {
        "Baseline Nurses": ["Nursing_and_Midwifery"],
        "Fewer Nurses": ["Nursing_and_Midwifery"],
        "More Nurses": ["Nursing_and_Midwifery"],
        "More Nurses by District": ["Nursing_and_Midwifery"],
        "More CNP staff": ["Clinical", "Pharmacy"],
        "More CNP staff by District": ["Clinical", "Pharmacy"],
    }
    scaleup_rows = []
    for prefix, cadres in scaleup_cadres.items():
        scenario = scenario_name(prefix, "Default")
        by_year = staff_mean[scenario].unstack(level="year")[[2019, 2024, 2027]]
        by_year = add_scaleup_factors(by_year[by_year.index.get_level_values("Cadre").isin(cadres)].copy())
        by_year["Scenario"] = scenario
        scaleup_rows.append(by_year.reset_index())

    scaleup_plot_data = order_districts(pd.concat(scaleup_rows, ignore_index=True), district_order)
    scaleup_plot_data = scaleup_plot_data.sort_values(["District", "Cadre", "Scenario"])
    for column, fig in plot_staff_scaleup_factors_by_district(scaleup_plot_data).items():
        figures[f"staff_scaleup_{column.removeprefix('Scale')}_by_district"] = fig

    # Staff scaling summary: nurses, and the three CNP cadres combined
    staff_summary = pd.concat(
        [
            staff_scaling_summary(
                staff_mean,
                [scenario_name(p, hs) for hs in HEALTHSYSTEMS for p in ["More Nurses", "More Nurses by District"]],
                "Nurses",
                ["Nursing_and_Midwifery"],
            ),
            staff_scaling_summary(
                staff_mean,
                [scenario_name(p, hs) for hs in HEALTHSYSTEMS for p in ["More CNP staff", "More CNP staff by District"]],
                "CNP",
                ["Clinical", "Nursing_and_Midwifery", "Pharmacy"],
            ),
        ],
        ignore_index=True,
    )[["District", "Staff_Type", "Staff2019", "Staff2024", "Staff2027",
       "Scale2027_2024", "Scale2027_2019", "Scenario"]]
    staff_summary.to_excel(results_folder / "district_staff_scaling.xlsx", index=False)

    # ---- DALYs and deaths averted ------------------------------------------
    averted = {}     # averted[outcome][healthsystem] = {"percent": {...}, "absolute": {...}}
    validation = {}  # validation[outcome] = (national, by district and year, district totals, comparison)

    for outcome in OUTCOMES:
        national = extract_national_annual(results_folder, outcome)
        by_district_and_year = extract_by_district(results_folder, outcome, by_year=True)
        validation[outcome] = (national, by_district_and_year,
                               *validate_district_totals(national, by_district_and_year, outcome))

        totals = set_param_names_as_column_index_level_0(
            extract_by_district(results_folder, outcome, by_year=False), param_names
        )

        averted[outcome] = {}
        for hs in HEALTHSYSTEMS:
            hs_totals = totals.loc[:, totals.columns.get_level_values(0).isin(
                [scenario_name(p, hs) for p, _, _ in SCENARIOS]
            )]
            baseline = scenario_name("Baseline Nurses", hs)
            percent = calculate_averted_by_district(hs_totals, baseline, percent=True)
            absolute = calculate_averted_by_district(hs_totals, baseline, percent=False)
            averted[outcome][hs] = {"percent": percent, "absolute": absolute}

            name = outcome.lower()
            figures[f"district_{name}_{hs.lower()}"] = plot_district_maps(
                district_map, percent, hs,
                f"% {outcome} averted (vs Baseline): {hs} Healthsystem",
                f"{outcome} averted in percentage (mean)",
            )
            figures[f"district_{name}_absolute_{hs.lower()}"] = plot_district_maps(
                district_map, absolute, hs,
                f"{outcome} averted (vs Baseline): {hs} Healthsystem",
                f"{outcome} averted (mean)",
            )
            figures[f"district_{name}_barplots_{hs.lower()}"] = plot_averted_by_district(
                absolute, percent, hs, outcome, district_order,
            )

    # Bar-plot data: % sheets first, then absolute sheets
    export_district_outcomes_to_excel(
        results_folder,
        [(f"{o}_{hs}", averted[o][hs]["percent"], o, " (%)") for o in OUTCOMES for hs in HEALTHSYSTEMS]
        + [(f"{o}_{hs}_abs", averted[o][hs]["absolute"], o, "") for o in OUTCOMES for hs in HEALTHSYSTEMS],
    )

    # Staff scaling with % DALYs and % deaths averted (means)
    for outcome in OUTCOMES:
        staff_summary = staff_summary.merge(
            means_long([averted[outcome][hs]["percent"] for hs in HEALTHSYSTEMS], f"Percent_{outcome}_Averted"),
            on=["District", "Scenario"],
            how="left",
        )
    staff_summary["Scenario"] = staff_summary["Scenario"].replace({
        scenario_name(p, hs): f"{p.replace('More CNP staff', 'More CNP')} / {hs}"
        for p, _, _ in COMPARISON_SCENARIOS for hs in HEALTHSYSTEMS
    })
    staff_summary.to_excel(results_folder / "district_staff_scaling_health_outcomes.xlsx", index=False)

    if args.save_figures:
        for name, fig in figures.items():
            fig.savefig(results_folder / f"{name}.pdf", dpi=300, bbox_inches="tight")

        with pd.ExcelWriter(results_folder / "district_vs_national_validation.xlsx", engine="openpyxl") as writer:
            for outcome, (national, by_district_and_year, district_totals, comparison) in validation.items():
                singular = OUTCOMES[outcome][2]
                national.to_excel(writer, sheet_name=f"National_{outcome}")
                by_district_and_year.to_excel(writer, sheet_name=f"District_{outcome}")
                district_totals.to_excel(writer, sheet_name=f"District_{singular}_Totals")
                comparison.to_excel(writer, sheet_name=f"{singular}_Comparison")

    if args.show_figures:
        plt.show()
