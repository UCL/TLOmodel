import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.nurses_analyses.nurses_scenario_analyses import StaffingScenario

from tlo.analysis.utils import (
    extract_results,
    summarize,
)


# -----------------------------------------------------------------------------
# Rename draw numbers to scenario names
# -----------------------------------------------------------------------------
def set_param_names_as_column_index_level_0(_df, param_names):

    ordered_param_names = {
        i: x for i, x in enumerate(param_names)
    }

    names_of_cols_level0 = [
        ordered_param_names.get(col)
        for col in _df.columns.levels[0]
    ]

    _df.columns = _df.columns.set_levels(
        names_of_cols_level0,
        level=0
    )

    return _df


# -----------------------------------------------------------------------------
# Extract annual staffing counts
# -----------------------------------------------------------------------------
def get_yearly_hr_count(df):

    if 'GenericClinic' not in df.columns:
        return None

    df['year'] = df['date'].dt.year

    # Expand dictionary
    staff_df = df['GenericClinic'].apply(pd.Series)

    # Keep cadre names only
    staff_df.columns = [
        c.split('Officer_')[-1]
        for c in staff_df.columns
    ]

    # Sum facilities within cadre
    staff_df = staff_df.groupby(level=0, axis=1).sum()

    # Add year
    staff_df['year'] = df['year']

    # Annual totals
    staff_df = staff_df.groupby('year').sum()

    # Scale population
    # POP_SCALE = 145.39609
    # staff_df = staff_df * POP_SCALE

    return staff_df.stack()


def extract_staff_counts(results_folder):

    return extract_results(
        results_folder,
        module="tlo.methods.healthsystem.summary",
        key="number_of_hcw_staff",
        custom_generate_series=get_yearly_hr_count,
        do_scaling=False,
    )


# -----------------------------------------------------------------------------
# Prepare plotting dataframe
# -----------------------------------------------------------------------------
def prepare_staffing_totals(summary_df):

    scenarios = (
        summary_df.columns
        .get_level_values(0)
        .unique()
    )

    results = {}

    for scenario in scenarios:

        mean_df = summary_df[(scenario, "mean")].unstack()

        # Nurses
        nurses = mean_df["Nursing_and_Midwifery"]

        # Other cadres
        other_cadres = mean_df.drop(
            columns=["Nursing_and_Midwifery"],
            errors="ignore"
        ).sum(axis=1)

        results[scenario] = pd.DataFrame({
            "Nurses": nurses,
            "Other cadres": other_cadres,
        })

    return results


# -----------------------------------------------------------------------------
# Plot staffing counts
# -----------------------------------------------------------------------------
def plot_staffing_counts(
    staffing_results,
    scenarios,
    title,
):

    fig, ax = plt.subplots(figsize=(10, 6))

    label_map = {
        "Baseline Nurses": "Baseline nurses",
        "Fewer Nurses": "Fewer nurses",
        "More Nurses": "More nurses",
    }

    # Plot nurse scenarios
    for scenario in scenarios:

        df = staffing_results[scenario]

        label = None

        for key in label_map:
            if key in scenario:
                label = f"Nurses, {label_map[key]}"

        ax.plot(
            df.index,
            df["Nurses"],
            linewidth=2,
            label=label,
        )

    # Plot other cadres once
    other_df = staffing_results[scenarios[0]]

    ax.plot(
        other_df.index,
        other_df["Other cadres"],
        linewidth=2.5,
        linestyle="--",
        color="black",
        label="Other cadres total",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual staff count")

    ax.set_title(title)

    ax.legend()

    ax.grid(alpha=0.3)

    fig.tight_layout()

    return fig, ax


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenario-outputs-folder",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--show-figures",
        action="store_true",
    )

    parser.add_argument(
        "--save-figures",
        action="store_true",
    )

    args = parser.parse_args()

    results_folder = args.scenario_outputs_folder

    # Scenario names
    param_names = tuple(
        StaffingScenario()._scenarios.keys()
    )

    # Extract
    staff_counts = extract_staff_counts(
        results_folder
    ).pipe(
        set_param_names_as_column_index_level_0,
        param_names=param_names,
    )

    # Summarize
    summarized_staff_counts = summarize(
        staff_counts
    )

    # Prepare totals
    staffing_results = prepare_staffing_totals(
        summarized_staff_counts
    )

    # Scenario groups
    default_hs_scenarios = [
        "Baseline Nurses / Default Healthsystem Function",
        "Fewer Nurses / Default Healthsystem Function",
        "More Nurses / Default Healthsystem Function",
    ]

    improved_hs_scenarios = [
        "Baseline Nurses / Improved Healthsystem Function",
        "Fewer Nurses / Improved Healthsystem Function",
        "More Nurses / Improved Healthsystem Function",
    ]

    # Plot default HS
    fig1, ax1 = plot_staffing_counts(
        staffing_results,
        default_hs_scenarios,
        title="Annual staffing count\nDefault Healthsystem",
    )

    # Plot improved HS
    fig2, ax2 = plot_staffing_counts(
        staffing_results,
        improved_hs_scenarios,
        title="Annual staffing count\nImproved Healthsystem",
    )

    if args.save_figures:
        fig1.savefig(
            results_folder / "annual_staffing_default_hs.pdf",
            bbox_inches="tight"
        )

        fig2.savefig(
            results_folder / "annual_staffing_improved_hs.pdf",
            bbox_inches="tight"
        )

    if args.show_figures:
        plt.show()
