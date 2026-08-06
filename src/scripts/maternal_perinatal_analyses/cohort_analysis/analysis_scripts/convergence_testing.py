from pathlib import Path

from tlo.util import read_csv_files, parse_csv_values_for_columns_with_mixed_datatypes

import numpy as np
import pandas as pd
import joypy
import math

from matplotlib.ticker import PercentFormatter

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as mticker

import matplotlib.colors as colors
import seaborn as sns

import os
from scipy.stats import t

from tableone import TableOne

# from scripts.comparison_of_horizontal_and_vertical_programs.economic_analysis_for_manuscript.roi_analysis_horizontal_vs_vertical import \
#     icers_summarized
from tlo import Date
from tlo.analysis.utils import extract_results, get_scenario_outputs, get_scenario_info, parse_log_file
from src.scripts.costing.cost_estimation import (do_stacked_bar_plot_of_cost_by_category,
    estimate_input_cost_of_scenarios, summarize_cost_data
)
outputspath = './outputs/sejjj49@ucl.ac.uk/'
resourcefilepath = Path("./resources")

def increase_cohort_size(df, target_val):
    additional_rows = target_val - len(df)
    rows_to_add = pd.DataFrame(columns=df.columns)

    # Loop to fill the required additional rows
    while additional_rows > 0:
        if additional_rows >= len(df):
            rows_to_add = pd.concat([rows_to_add, df], ignore_index=True)
            additional_rows -= len(df)
        else:
            rows_to_add = pd.concat([rows_to_add, df.iloc[:additional_rows]], ignore_index=True)
            additional_rows = 0

    # Concatenate the original DataFrame with the additional rows
    preg_pop = pd.concat([df, rows_to_add], ignore_index=True)
    return preg_pop

# 1.) Testing convergence for properties of cohort for different sample sizes
cohort = read_csv_files(Path(f'{resourcefilepath}/ResourceFile_MaternalCohort'),
                        files='ResourceFile_All2025PregnanciesCohortModel')

for col in cohort.columns:
    cohort[col] = cohort[col].apply(
        parse_csv_values_for_columns_with_mixed_datatypes
    )

print(f"{len(cohort)} pregnancies in initial cohort")

cohort_20k = increase_cohort_size(cohort, 20_000)
cohort_40k = increase_cohort_size(cohort, 40_000)
cohort_60k = increase_cohort_size(cohort, 60_000)
cohort_80k = increase_cohort_size(cohort, 80_000)
cohort_100k = increase_cohort_size(cohort, 100_000)


# Continuous variables - Age, BMI, Parity
def plot_ridgeline_comparison(
    dataframes,
    labels=None,
    column="age_years",
    overlap=1.5,
    figsize=(10, 7),
):
    """
    Compare the distribution of one numeric column across multiple DataFrames
    using a ridgeline plot.

    Parameters
    ----------
    dataframes : list of pandas.DataFrame
        DataFrames containing the column to compare.

    labels : list of str, optional
        Labels for the DataFrames. If omitted, sample sizes are used.

    column : str, default="age_years"
        Numeric column to plot.

    overlap : float, default=1.5
        Amount of overlap between density curves.

    figsize : tuple, default=(10, 7)
        Figure size.

    Returns
    -------
    fig, axes
        Matplotlib figure and JoyPy axes.
    """

    if not dataframes:
        raise ValueError("dataframes must contain at least one DataFrame.")

    for i, df in enumerate(dataframes):
        if column not in df.columns:
            raise KeyError(
                f"Column '{column}' is missing from DataFrame {i}."
            )

    if labels is None:
        labels = [
            f"n = {df[column].notna().sum():,}"
            for df in dataframes
        ]

    if len(labels) != len(dataframes):
        raise ValueError(
            "labels must have the same length as dataframes."
        )

    # Combine the six DataFrames into long format
    plot_df = pd.concat(
        [
            pd.DataFrame({
                column: pd.to_numeric(
                    df[column], errors="coerce"
                ),
                "dataset": label,
            })
            for df, label in zip(dataframes, labels)
        ],
        ignore_index=True,
    ).dropna(subset=[column])

    # Preserve the supplied dataset order
    plot_df["dataset"] = pd.Categorical(
        plot_df["dataset"],
        categories=labels,
        ordered=True,
    )

    fig, axes = joypy.joyplot(
        data=plot_df,
        by="dataset",
        column=column,
        overlap=overlap,
        figsize=figsize,
        linewidth=1,
        fade=True,
        grid="x",
    )

    # Add each dataset's mean as a vertical line on its own ridge
    means = (
        plot_df
        .groupby("dataset", observed=False)[column]
        .mean()
        .reindex(labels)
    )

    # JoyPy creates one density axis per group plus a final shared axis
    ridge_axes = axes[:len(labels)]

    for ax, mean_value in zip(ridge_axes, means):
        ax.axvline(
            mean_value,
            linestyle="--",
            linewidth=1.5,
        )

    fig.suptitle(
        f"Distribution of {column.replace('_', ' ').title()} "
        "by Generated Sample Size",
        y=1.02,
    )

    fig.supxlabel(column.replace("_", " ").title())
    fig.supylabel("Generated dataset")

    plt.tight_layout()
    plt.show()

    return fig, axes

dataframes = [
    cohort,
    cohort_20k,
    cohort_40k,
    cohort_60k,
    cohort_80k,
    cohort_100k,
]
labels = [
    "n = 13k",
    "n = 20k",
    "n = 40k",
    "n = 60k",
    "n = 80k",
    "n = 100k",
]
fig, axes = plot_ridgeline_comparison(
    dataframes=dataframes,
    labels=labels,
    column="age_years",
)









#  ======================================= DEFINE SCENARIO INFORMATION  ===============================================
scenario = 'testing_scenario_747943'
