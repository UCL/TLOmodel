from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# HPV CALIBRATION AND VALIDATION PLOTTING
# =============================================================================
#
# Required input files:
#
#   grid_search_results.csv
#   grid_search_replicates.csv
#   best_b_hpv_validation.csv
#
# Main outputs:
#
#   01_primary_calibration_by_beta.png
#   02_calibration_loss.png
#   03_seed_stability.png
#   04_age_validation_hiv_negative.png
#   05_age_validation_hiv_positive.png
#   06_age_validation_combined.png
#   07_xpert_P1_P2_P3_validation.png
#   08_xpert_P1_P2_P3_residuals.png
#   09_overall_hiv_validation.png
#   10_observed_vs_simulated.png
#   11_supervisor_dashboard.png
#   supervisor_validation_summary.csv
#
# Calibration hierarchy:
#
#   Level 1: b_hpv
#       -> CHAI any HR-HPV prevalence
#       -> HIV-negative women aged 25-49
#
#   Level 2: HIV validation
#       -> HIV-positive women aged 25-49
#
#   Level 3: Age validation
#       -> 25-29, 30-34, 35-39, 40-44, 45-49
#
#   Level 4: Genotype validation
#       -> P1 = hr1 = HPV16
#       -> P2 = hr2 + hr3 = HPV18/45
#       -> P3 = hr4 + hr5 = selected HR types
#
#   P4 = hr6
#       -> currently no direct observed validation target
#
# =============================================================================


# =============================================================================
# === MODIFIED: GLOBAL FIGURE STYLE ===========================================
# =============================================================================

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.titlesize": 16,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# =============================================================================
# === MODIFIED: CURRENT CHAI OBSERVED TARGETS =================================
# =============================================================================
#
# IMPORTANT:
#
# These replace the OLD 625-person observed targets in the previous plotting
# script.
#
# Format:
#
#     target_name: (number_positive, denominator)
#
# =============================================================================

OBSERVED_COUNTS = {

    # -------------------------------------------------------------------------
    # LEVEL 1
    # Primary calibration target
    # -------------------------------------------------------------------------

    "CHAI_HRHPV_HIVneg_25_49": (
        564,
        2083,
    ),

    # -------------------------------------------------------------------------
    # LEVEL 2
    # HIV-positive overall validation
    # -------------------------------------------------------------------------

    "CHAI_HRHPV_HIVpos_25_49": (
        872,
        2424,
    ),

    # -------------------------------------------------------------------------
    # LEVEL 3
    # Age-specific HR-HPV prevalence: HIV-negative
    # -------------------------------------------------------------------------

    "CHAI_HRHPV_HIVneg_25_29": (
        194,
        600,
    ),

    "CHAI_HRHPV_HIVneg_30_34": (
        123,
        476,
    ),

    "CHAI_HRHPV_HIVneg_35_39": (
        116,
        452,
    ),

    "CHAI_HRHPV_HIVneg_40_44": (
        79,
        325,
    ),

    "CHAI_HRHPV_HIVneg_45_49": (
        52,
        230,
    ),

    # -------------------------------------------------------------------------
    # LEVEL 3
    # Age-specific HR-HPV prevalence: HIV-positive
    # -------------------------------------------------------------------------

    "CHAI_HRHPV_HIVpos_25_29": (
        177,
        434,
    ),

    "CHAI_HRHPV_HIVpos_30_34": (
        169,
        445,
    ),

    "CHAI_HRHPV_HIVpos_35_39": (
        193,
        576,
    ),

    "CHAI_HRHPV_HIVpos_40_44": (
        196,
        562,
    ),

    "CHAI_HRHPV_HIVpos_45_49": (
        137,
        407,
    ),

    # -------------------------------------------------------------------------
    # LEVEL 4
    # Genotype / Xpert validation
    # -------------------------------------------------------------------------

    # P1 = hr1 = HPV16

    "CHAI_HPV16_HIVneg_25_49": (
        107,
        2083,
    ),

    "CHAI_HPV16_HIVpos_25_49": (
        191,
        2424,
    ),

    # P2 = hr2 + hr3 = HPV18/45

    "CHAI_HPV18_45_HIVneg_25_49": (
        84,
        2083,
    ),

    "CHAI_HPV18_45_HIVpos_25_49": (
        130,
        2424,
    ),
}


# =============================================================================
# === NEW: AGE GROUP DEFINITIONS ==============================================
# =============================================================================

AGE_GROUPS = [
    "25–29",
    "30–34",
    "35–39",
    "40–44",
    "45–49",
]


AGE_COLUMNS_HIVNEG = [
    "CHAI_HRHPV_HIVneg_25_29",
    "CHAI_HRHPV_HIVneg_30_34",
    "CHAI_HRHPV_HIVneg_35_39",
    "CHAI_HRHPV_HIVneg_40_44",
    "CHAI_HRHPV_HIVneg_45_49",
]


AGE_COLUMNS_HIVPOS = [
    "CHAI_HRHPV_HIVpos_25_29",
    "CHAI_HRHPV_HIVpos_30_34",
    "CHAI_HRHPV_HIVpos_35_39",
    "CHAI_HRHPV_HIVpos_40_44",
    "CHAI_HRHPV_HIVpos_45_49",
]


# =============================================================================
# === NEW: XPERT CHANNEL DEFINITIONS ==========================================
# =============================================================================
#
# Current HPV model mapping:
#
#   P1 = hr1
#   P2 = hr2 + hr3
#   P3 = hr4 + hr5
#   P4 = hr6
#
# =============================================================================

P_CHANNEL_LABELS = {
    "P1": "P1\nHPV16",
    "P2": "P2\nHPV18/45",
    "P3": "P3\nSelected HR",
    "P4": "P4\nOther HR",
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def observed_prevalence(
    target: str,
) -> float:
    """Return observed prevalence from positive count and denominator."""

    positive, denominator = OBSERVED_COUNTS[target]

    return positive / denominator


def observed_standard_error(
    target: str,
) -> float:
    """Binomial standard error of observed prevalence."""

    prevalence = observed_prevalence(target)

    _, denominator = OBSERVED_COUNTS[target]

    return math.sqrt(
        prevalence
        * (1.0 - prevalence)
        / denominator
    )


def observed_ci_from_counts(
    target: str,
) -> tuple[float, float]:
    """
    Normal-approximation 95% CI.

    This is consistent with the calibration loss framework currently used.
    """

    prevalence = observed_prevalence(target)

    standard_error = observed_standard_error(target)

    lower = max(
        0.0,
        prevalence - 1.96 * standard_error,
    )

    upper = min(
        1.0,
        prevalence + 1.96 * standard_error,
    )

    return lower, upper


def observed_errorbar(
    target: str,
) -> tuple[float, float]:

    prevalence = observed_prevalence(target)

    lower, upper = observed_ci_from_counts(target)

    return (
        prevalence - lower,
        upper - prevalence,
    )


def percent_axis(
    axis: plt.Axes,
) -> None:

    axis.yaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda value, _:
                f"{value * 100:.0f}%"
        )
    )


def percent_x_axis(
    axis: plt.Axes,
) -> None:

    axis.xaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda value, _:
                f"{value * 100:.0f}%"
        )
    )


def require_columns(
    frame: pd.DataFrame,
    required: list[str],
    frame_name: str,
) -> None:

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:

        raise KeyError(
            f"\nMissing columns in {frame_name}:\n"
            + "\n".join(missing)
        )


# =============================================================================
# LOAD RESULTS
# =============================================================================


def load_inputs(
    result_directory: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    grid_path = (
        result_directory
        / "grid_search_results.csv"
    )

    replicate_path = (
        result_directory
        / "grid_search_replicates.csv"
    )

    validation_path = (
        result_directory
        / "best_b_hpv_validation.csv"
    )

    for path in [
        grid_path,
        replicate_path,
        validation_path,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Cannot find required file:\n{path}"
            )

    results = pd.read_csv(
        grid_path
    )

    replicates = pd.read_csv(
        replicate_path
    )

    validation = pd.read_csv(
        validation_path
    )

    require_columns(
        results,
        [
            "b_hpv",
            "loss",
            "observed",
            "sim_mean",
            "sim_sd",
            "sim_mcse",
            "n_seeds",
        ],
        "grid_search_results.csv",
    )

    require_columns(
        replicates,
        [
            "b_hpv",
            "seed",
            "CHAI_HRHPV_HIVneg_25_49",
        ],
        "grid_search_replicates.csv",
    )

    require_columns(
        validation,
        [
            "target",
            "purpose",
            "observed",
            "sim_mean",
            "sim_sd",
            "residual",
        ],
        "best_b_hpv_validation.csv",
    )

    return (
        results,
        replicates,
        validation,
    )


# =============================================================================
# BEST BETA
# =============================================================================


def get_best_row(
    results: pd.DataFrame,
) -> pd.Series:

    return results.loc[
        results["loss"].idxmin()
    ]


def get_best_beta(
    results: pd.DataFrame,
) -> float:

    best_row = get_best_row(
        results
    )

    return float(
        best_row["b_hpv"]
    )


# =============================================================================
# FIGURE 1
# PRIMARY CALIBRATION
# =============================================================================


def plot_primary_calibration(
    results: pd.DataFrame,
    output_directory: Path,
) -> None:

    results = (
        results
        .sort_values("b_hpv")
        .reset_index(drop=True)
    )

    best_row = get_best_row(
        results
    )

    best_beta = float(
        best_row["b_hpv"]
    )

    target = (
        "CHAI_HRHPV_HIVneg_25_49"
    )

    observed = observed_prevalence(
        target
    )

    observed_lower, observed_upper = (
        observed_ci_from_counts(
            target
        )
    )

    figure, axis = plt.subplots(
        figsize=(9, 5.8)
    )

    axis.errorbar(
        results["b_hpv"],
        results["sim_mean"],
        yerr=results["sim_sd"],
        marker="o",
        linewidth=2,
        markersize=7,
        capsize=4,
        label="Model mean ± SD",
    )

    axis.axhline(
        observed,
        linestyle="--",
        linewidth=2,
        label=(
            f"CHAI observed = "
            f"{observed * 100:.1f}%"
        ),
    )

    axis.axhspan(
        observed_lower,
        observed_upper,
        alpha=0.12,
        label="Observed 95% CI",
    )

    axis.axvline(
        best_beta,
        linestyle=":",
        linewidth=2,
        label=(
            f"Best $b_{{HPV}}$ = "
            f"{best_beta:.2f}"
        ),
    )

    axis.scatter(
        [best_beta],
        [best_row["sim_mean"]],
        s=100,
        zorder=5,
    )

    axis.annotate(
        (
            f"Best fit\n"
            f"$b_{{HPV}}$ = {best_beta:.2f}\n"
            f"Model = "
            f"{best_row['sim_mean'] * 100:.2f}%"
        ),
        xy=(
            best_beta,
            best_row["sim_mean"],
        ),
        xytext=(20, -55),
        textcoords="offset points",
        arrowprops={
            "arrowstyle": "->"
        },
    )

    axis.set_title(
        "Primary calibration: overall HR-HPV prevalence"
    )

    axis.set_xlabel(
        "$b_{HPV}$"
    )

    axis.set_ylabel(
        "HR-HPV prevalence"
    )

    percent_axis(axis)

    axis.grid(
        alpha=0.2
    )

    axis.legend(
        frameon=False
    )

    figure.tight_layout()

    figure.savefig(
        output_directory
        / "01_primary_calibration_by_beta.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# =============================================================================
# FIGURE 2
# CALIBRATION LOSS
# =============================================================================


def plot_calibration_loss(
    results: pd.DataFrame,
    output_directory: Path,
) -> None:

    results = (
        results
        .sort_values("b_hpv")
        .reset_index(drop=True)
    )

    best_beta = get_best_beta(
        results
    )

    figure, axis = plt.subplots(
        figsize=(8.5, 5.5)
    )

    axis.plot(
        results["b_hpv"],
        results["loss"],
        marker="o",
        linewidth=2,
    )

    axis.axvline(
        best_beta,
        linestyle="--",
        label=(
            f"Minimum loss: "
            f"$b_{{HPV}}$={best_beta:.2f}"
        ),
    )

    axis.set_title(
        "Calibration loss across tested $b_{HPV}$ values"
    )

    axis.set_xlabel(
        "$b_{HPV}$"
    )

    axis.set_ylabel(
        "Squared standardized residual"
    )

    axis.grid(
        alpha=0.2
    )

    axis.legend(
        frameon=False
    )

    figure.tight_layout()

    figure.savefig(
        output_directory
        / "02_calibration_loss.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# =============================================================================
# FIGURE 3
# SEED STABILITY
# =============================================================================


def plot_seed_stability(
    replicates: pd.DataFrame,
    results: pd.DataFrame,
    output_directory: Path,
) -> None:

    target = (
        "CHAI_HRHPV_HIVneg_25_49"
    )

    observed = observed_prevalence(
        target
    )

    best_beta = get_best_beta(
        results
    )

    figure, axis = plt.subplots(
        figsize=(9, 5.8)
    )

    for seed, frame in replicates.groupby(
        "seed"
    ):

        frame = frame.sort_values(
            "b_hpv"
        )

        axis.plot(
            frame["b_hpv"],
            frame[target],
            marker="o",
            linewidth=1.2,
            alpha=0.55,
            label=f"Seed {seed}",
        )

    mean_by_beta = (
        replicates
        .groupby("b_hpv")[target]
        .mean()
        .sort_index()
    )

    axis.plot(
        mean_by_beta.index,
        mean_by_beta.values,
        marker="D",
        linewidth=3,
        color="black",
        label="Mean across seeds",
    )

    axis.axhline(
        observed,
        linestyle="--",
        linewidth=2,
        label=(
            f"CHAI observed = "
            f"{observed * 100:.1f}%"
        ),
    )

    axis.axvline(
        best_beta,
        linestyle=":",
        linewidth=2,
        label=(
            f"Best $b_{{HPV}}$="
            f"{best_beta:.2f}"
        ),
    )

    axis.set_title(
        "Stochastic stability of the primary calibration target"
    )

    axis.set_xlabel(
        "$b_{HPV}$"
    )

    axis.set_ylabel(
        "HR-HPV prevalence"
    )

    percent_axis(
        axis
    )

    axis.grid(
        alpha=0.2
    )

    axis.legend(
        frameon=False,
        ncol=2,
    )

    figure.tight_layout()

    figure.savefig(
        output_directory
        / "03_seed_stability.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# =============================================================================
# AGE VALIDATION DATA
# =============================================================================


def extract_age_validation(
    best_row: pd.Series,
    hiv_status: str,
) -> pd.DataFrame:

    if hiv_status == "HIV-negative":

        columns = (
            AGE_COLUMNS_HIVNEG
        )

    elif hiv_status == "HIV-positive":

        columns = (
            AGE_COLUMNS_HIVPOS
        )

    else:

        raise ValueError(
            "hiv_status must be "
            "'HIV-negative' or 'HIV-positive'."
        )

    rows = []

    for (
        age_label,
        base_column,
    ) in zip(
        AGE_GROUPS,
        columns,
    ):

        mean_column = (
            f"{base_column}__sim_mean"
        )

        sd_column = (
            f"{base_column}__sim_sd"
        )

        observed_column = (
            f"{base_column}__observed"
        )

        if mean_column not in best_row.index:

            raise KeyError(
                f"Missing age-specific column: "
                f"{mean_column}"
            )

        if observed_column in best_row.index:

            observed = float(
                best_row[
                    observed_column
                ]
            )

        else:

            observed = (
                observed_prevalence(
                    base_column
                )
            )

        rows.append(
            {
                "age": age_label,
                "target": base_column,
                "observed": observed,
                "sim_mean": float(
                    best_row[
                        mean_column
                    ]
                ),
                "sim_sd": float(
                    best_row[
                        sd_column
                    ]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# FIGURES 4 AND 5
# AGE-SPECIFIC VALIDATION
# =============================================================================


def plot_age_validation_single(
    best_row: pd.Series,
    hiv_status: str,
    output_directory: Path,
    filename: str,
) -> None:

    frame = extract_age_validation(
        best_row,
        hiv_status,
    )

    x = np.arange(
        len(frame)
    )

    observed_lower = []
    observed_upper = []

    for target in frame["target"]:

        low, high = (
            observed_errorbar(
                target
            )
        )

        observed_lower.append(
            low
        )

        observed_upper.append(
            high
        )

    figure, axis = plt.subplots(
        figsize=(9, 5.8)
    )

    axis.errorbar(
        x,
        frame["observed"],
        yerr=[
            observed_lower,
            observed_upper,
        ],
        marker="o",
        linewidth=2,
        markersize=7,
        capsize=4,
        label="CHAI observed (95% CI)",
    )

    axis.errorbar(
        x,
        frame["sim_mean"],
        yerr=frame["sim_sd"],
        marker="s",
        linewidth=2,
        markersize=7,
        capsize=4,
        label="Model mean ± SD",
    )

    axis.set_xticks(
        x
    )

    axis.set_xticklabels(
        frame["age"]
    )

    axis.set_xlabel(
        "Age group (years)"
    )

    axis.set_ylabel(
        "HR-HPV prevalence"
    )

    axis.set_title(
        f"Age-specific HR-HPV prevalence: {hiv_status}"
    )

    percent_axis(
        axis
    )

    axis.grid(
        alpha=0.2
    )

    axis.legend(
        frameon=False
    )

    figure.tight_layout()

    figure.savefig(
        output_directory
        / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# =============================================================================
# FIGURE 6
# COMBINED AGE VALIDATION
# =============================================================================
def plot_age_validation_combined(
    best_row: pd.Series,
    output_directory: Path,
) -> None:

    negative = extract_age_validation(
        best_row,
        "HIV-negative",
    )

    positive = extract_age_validation(
        best_row,
        "HIV-positive",
    )

    x = np.arange(len(AGE_GROUPS))

    # More vertical space for title + legend
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.8),
        sharey=True,
    )

    for axis, frame, status in [
        (
            axes[0],
            negative,
            "HIV-negative",
        ),
        (
            axes[1],
            positive,
            "HIV-positive",
        ),
    ]:

        observed_lower = []
        observed_upper = []

        for target in frame["target"]:

            low, high = observed_errorbar(
                target
            )

            observed_lower.append(low)
            observed_upper.append(high)

        axis.errorbar(
            x,
            frame["observed"],
            yerr=[
                observed_lower,
                observed_upper,
            ],
            marker="o",
            linewidth=2,
            capsize=4,
            label="CHAI observed (95% CI)",
        )

        axis.errorbar(
            x,
            frame["sim_mean"],
            yerr=frame["sim_sd"],
            marker="s",
            linewidth=2,
            capsize=4,
            label="Model mean ± SD",
        )

        axis.set_xticks(x)
        axis.set_xticklabels(AGE_GROUPS)

        axis.set_xlabel(
            "Age group (years)"
        )

        axis.set_title(
            status,
            pad=10,
        )

        percent_axis(axis)

        axis.grid(
            alpha=0.2
        )

    axes[0].set_ylabel(
        "HR-HPV prevalence"
    )

    # ---------------------------------------------------------
    # FIX:
    # Main title and legend now occupy different vertical space
    # ---------------------------------------------------------

    figure.suptitle(
        "Age-specific validation of HR-HPV prevalence",
        fontsize=16,
        y=0.98,
    )

    handles, labels = (
        axes[0]
        .get_legend_handles_labels()
    )

    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=2,
        frameon=False,
    )

    # Reserve top 14% for title + legend
    figure.tight_layout(
        rect=[
            0,
            0,
            1,
            0.86,
        ]
    )

    figure.savefig(
        output_directory
        / "06_age_validation_combined.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# =============================================================================
# P1 / P2 / P3 VALIDATION DATA
# =============================================================================
P_CHANNEL_GRID_COLUMNS = {

    # -------------------------------------------------------------------------
    # P1 = hr1 = HPV16
    # -------------------------------------------------------------------------

    "P1": {
        "title": "P1: HPV16",
        "hivneg": "CHAI_HPV16_HIVneg_25_49",
        "hivpos": "CHAI_HPV16_HIVpos_25_49",
    },

    # -------------------------------------------------------------------------
    # P2 = hr2 + hr3 = HPV18/45
    # -------------------------------------------------------------------------

    "P2": {
        "title": "P2: HPV18/45",
        "hivneg": "CHAI_HPV18_45_HIVneg_25_49",
        "hivpos": "CHAI_HPV18_45_HIVpos_25_49",
    },

    # -------------------------------------------------------------------------
    # P3 = hr4 + hr5
    # -------------------------------------------------------------------------

    "P3": {
        "title": "P3: selected high-risk HPV types",
        "hivneg": "External_P3_HIVneg_25_50",
        "hivpos": "External_P3_HIVpos_25_50",
    },
}

def plot_single_p_channel_by_beta(
    results: pd.DataFrame,
    validation: pd.DataFrame,
    channel: str,
    output_directory: Path,
) -> None:

    config = P_CHANNEL_GRID_COLUMNS[channel]

    hivneg_base = config["hivneg"]
    hivpos_base = config["hivpos"]

    hivneg_mean = (
        f"{hivneg_base}__sim_mean"
    )

    hivneg_sd = (
        f"{hivneg_base}__sim_sd"
    )

    hivpos_mean = (
        f"{hivpos_base}__sim_mean"
    )

    hivpos_sd = (
        f"{hivpos_base}__sim_sd"
    )

    # ---------------------------------------------------------
    # Check that grid-search results actually contain the
    # required P-channel outputs.
    # ---------------------------------------------------------

    required = [
        hivneg_mean,
        hivneg_sd,
        hivpos_mean,
        hivpos_sd,
    ]

    missing = [
        column
        for column in required
        if column not in results.columns
    ]

    if missing:

        print()
        print(
            f"WARNING: Cannot generate {channel} vs b_hpv."
        )

        print(
            "Missing grid-search columns:"
        )

        for column in missing:
            print(
                f"    {column}"
            )

        return

    results = (
        results
        .sort_values("b_hpv")
        .copy()
    )

    best_beta = get_best_beta(
        results
    )

    # ---------------------------------------------------------
    # Observed values are read from validation CSV.
    # ---------------------------------------------------------

    neg_match = validation.loc[
        validation["target"]
        == hivneg_base
    ]

    pos_match = validation.loc[
        validation["target"]
        == hivpos_base
    ]

    if (
        neg_match.empty
        or pos_match.empty
    ):

        print(
            f"WARNING: observed targets missing for {channel}."
        )

        return

    observed_neg = float(
        neg_match.iloc[0][
            "observed"
        ]
    )

    observed_pos = float(
        pos_match.iloc[0][
            "observed"
        ]
    )

    # ---------------------------------------------------------
    # Observed SE
    # ---------------------------------------------------------

    observed_neg_se = float(
        neg_match.iloc[0].get(
            "observed_se",
            np.nan,
        )
    )

    observed_pos_se = float(
        pos_match.iloc[0].get(
            "observed_se",
            np.nan,
        )
    )

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    # =========================================================
    # HIV-negative model
    # =========================================================

    axis.errorbar(
        results["b_hpv"],
        results[hivneg_mean],
        yerr=results[hivneg_sd],
        color="tab:blue",
        marker="o",
        linewidth=2,
        markersize=6,
        capsize=4,
        label="Model: HIV-negative",
    )

    # =========================================================
    # HIV-positive model
    # =========================================================

    axis.errorbar(
        results["b_hpv"],
        results[hivpos_mean],
        yerr=results[hivpos_sd],
        color="tab:orange",
        marker="s",
        linewidth=2,
        markersize=6,
        capsize=4,
        label="Model: HIV-positive",
    )

    # =========================================================
    # Observed HIV-negative
    # =========================================================

    axis.axhline(
        observed_neg,
        color="tab:blue",
        linestyle="--",
        linewidth=1.8,
        label=(
            f"Observed: HIV-negative "
            f"({observed_neg * 100:.1f}%)"
        ),
    )

    if not np.isnan(
        observed_neg_se
    ):

        axis.axhspan(
            max(0, observed_neg - 1.96 * observed_neg_se,),
            observed_neg
            + 1.96
            * observed_neg_se,
            color="tab:blue",
            alpha=0.08,
        )

    # =========================================================
    # Observed HIV-positive
    # =========================================================

    axis.axhline(
        observed_pos,
        linestyle="--",
        linewidth=1.8,
        color="tab:orange",
        label=(
            f"Observed: HIV-positive "
            f"({observed_pos * 100:.1f}%)"
        ),
    )

    if not np.isnan(
        observed_pos_se
    ):

        axis.axhspan(
            max(
                0,
                observed_pos
                - 1.96
                * observed_pos_se,
            ),
            observed_pos
            + 1.96
            * observed_pos_se,
            color="tab:orange",
            alpha=0.08,
        )

    # =========================================================
    # Best b_hpv
    # =========================================================

    axis.axvline(
        best_beta,
        linestyle=":",
        linewidth=2,
        label=(
            f"Calibrated "
            f"$b_{{HPV}}$={best_beta:.2f}"
        ),
    )

    axis.set_title(
        config["title"],
        fontsize=15,
        pad=12,
    )

    axis.set_xlabel(
        "$b_{HPV}$"
    )

    axis.set_ylabel(
        "Prevalence"
    )

    percent_axis(
        axis
    )

    axis.grid(
        alpha=0.2
    )

    axis.legend(
        frameon=False,
        loc="best",
    )

    figure.tight_layout()

    figure.savefig(
        output_directory
        / f"07_{channel}_prevalence_by_beta.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

def plot_all_p_channels_by_beta(
    results: pd.DataFrame,
    validation: pd.DataFrame,
    output_directory: Path,
) -> None:

    for channel in [
        "P1",
        "P2",
        "P3",
    ]:

        plot_single_p_channel_by_beta(
            results,
            validation,
            channel,
            output_directory,
        )

def plot_p_channels_combined_by_beta(
    results: pd.DataFrame,
    validation: pd.DataFrame,
    output_directory: Path,
) -> None:

    results = (
        results
        .sort_values("b_hpv")
        .copy()
    )

    best_beta = get_best_beta(
        results
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5.5),
        sharex=True,
    )

    successful_panels = 0

    for axis, channel in zip(
        axes,
        ["P1", "P2", "P3"],
    ):

        config = (
            P_CHANNEL_GRID_COLUMNS[
                channel
            ]
        )

        neg = config["hivneg"]
        pos = config["hivpos"]

        neg_mean = (
            f"{neg}__sim_mean"
        )

        neg_sd = (
            f"{neg}__sim_sd"
        )

        pos_mean = (
            f"{pos}__sim_mean"
        )

        pos_sd = (
            f"{pos}__sim_sd"
        )

        required = [
            neg_mean,
            neg_sd,
            pos_mean,
            pos_sd,
        ]

        if any(
            column
            not in results.columns
            for column in required
        ):

            axis.text(
                0.5,
                0.5,
                (
                    f"{channel}\n"
                    "Grid-search output\n"
                    "not available"
                ),
                ha="center",
                va="center",
                transform=axis.transAxes,
            )

            axis.set_axis_off()

            continue

        neg_match = validation.loc[
            validation["target"]
            == neg
        ]

        pos_match = validation.loc[
            validation["target"]
            == pos
        ]

        if (
            neg_match.empty
            or pos_match.empty
        ):

            axis.set_axis_off()

            continue

        successful_panels += 1

        observed_neg = float(
            neg_match.iloc[0][
                "observed"
            ]
        )

        observed_pos = float(
            pos_match.iloc[0][
                "observed"
            ]
        )

        # -----------------------------------------------------
        # Model trajectories
        # -----------------------------------------------------

        axis.errorbar(
            results["b_hpv"],
            results[neg_mean],
            yerr=results[neg_sd],
            color="tab:blue",
            marker="o",
            linewidth=2,
            capsize=3,
            label="Model: HIV-negative",
        )

        axis.errorbar(
            results["b_hpv"],
            results[pos_mean],
            yerr=results[pos_sd],
            color="tab:orange",
            marker="s",
            linewidth=2,
            capsize=3,
            label="Model: HIV-positive",
        )

        # -----------------------------------------------------
        # Observed targets
        # -----------------------------------------------------

        axis.axhline(
            observed_neg,
            linestyle="--",
            color="tab:blue",
            linewidth=1.5,
            label="Observed: HIV-negative",
        )

        axis.axhline(
            observed_pos,
            color="tab:orange",
            linestyle="--",
            linewidth=1.5,
            label="Observed: HIV-positive",
        )

        # -----------------------------------------------------
        # Best beta
        # -----------------------------------------------------

        axis.axvline(
            best_beta,
            linestyle=":",
            linewidth=1.5,
        )

        axis.set_title(
            config["title"],
            pad=10,
        )

        axis.set_xlabel(
            "$b_{HPV}$"
        )

        percent_axis(
            axis
        )

        axis.grid(
            alpha=0.2
        )

    axes[0].set_ylabel(
        "Prevalence"
    )

    # ---------------------------------------------------------
    # Legend below title — NOT overlapping
    # ---------------------------------------------------------

    if successful_panels > 0:

        handles, labels = (
            axes[0]
            .get_legend_handles_labels()
        )

        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(
                0.5,
                0.91,
            ),
            ncol=4,
            frameon=False,
        )

    figure.suptitle(
        (
            "HPV genotype-group prevalence "
            "across transmission parameter values"
        ),
        fontsize=16,
        y=0.99,
    )

    figure.tight_layout(
        rect=[
            0,
            0,
            1,
            0.84,
        ]
    )

    figure.savefig(
        output_directory
        / "07_P1_P2_P3_prevalence_by_beta_combined.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

def build_p_channel_validation(
    validation: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    # -------------------------------------------------------------------------
    # P1 = hr1 = HPV16
    # -------------------------------------------------------------------------

    for (
        hiv_status,
        target,
    ) in [

        (
            "HIV-negative",
            "CHAI_HPV16_HIVneg_25_49",
        ),

        (
            "HIV-positive",
            "CHAI_HPV16_HIVpos_25_49",
        ),
    ]:

        match = validation.loc[
            validation["target"]
            == target
        ]

        if not match.empty:

            row = match.iloc[0]

            rows.append(
                {
                    "channel": "P1",
                    "hiv_status": hiv_status,
                    "observed": float(
                        row["observed"]
                    ),
                    "sim_mean": float(
                        row["sim_mean"]
                    ),
                    "sim_sd": float(
                        row["sim_sd"]
                    ),
                    "source": "CHAI",
                }
            )

    # -------------------------------------------------------------------------
    # P2 = hr2 + hr3 = HPV18/45
    # -------------------------------------------------------------------------

    for (
        hiv_status,
        target,
    ) in [

        (
            "HIV-negative",
            "CHAI_HPV18_45_HIVneg_25_49",
        ),

        (
            "HIV-positive",
            "CHAI_HPV18_45_HIVpos_25_49",
        ),
    ]:

        match = validation.loc[
            validation["target"]
            == target
        ]

        if not match.empty:

            row = match.iloc[0]

            rows.append(
                {
                    "channel": "P2",
                    "hiv_status": hiv_status,
                    "observed": float(
                        row["observed"]
                    ),
                    "sim_mean": float(
                        row["sim_mean"]
                    ),
                    "sim_sd": float(
                        row["sim_sd"]
                    ),
                    "source": "CHAI",
                }
            )

    # -------------------------------------------------------------------------
    # P3 = hr4 + hr5
    #
    # External validation currently available.
    # -------------------------------------------------------------------------

    for (
        hiv_status,
        target,
    ) in [

        (
            "HIV-negative",
            "External_P3_HIVneg_25_50",
        ),

        (
            "HIV-positive",
            "External_P3_HIVpos_25_50",
        ),
    ]:

        match = validation.loc[
            validation["target"]
            == target
        ]

        if not match.empty:

            row = match.iloc[0]

            rows.append(
                {
                    "channel": "P3",
                    "hiv_status": hiv_status,
                    "observed": float(
                        row["observed"]
                    ),
                    "sim_mean": float(
                        row["sim_mean"]
                    ),
                    "sim_sd": float(
                        row["sim_sd"]
                    ),
                    "source": "External",
                }
            )

    # -------------------------------------------------------------------------
    # P4 = hr6
    #
    # No direct observed validation target is currently available.
    #
    # IMPORTANT:
    # We deliberately do NOT invent an observed P4 value.
    # -------------------------------------------------------------------------

    return pd.DataFrame(
        rows
    )


# =============================================================================
# FIGURE 7
# P1 / P2 / P3 OBSERVED VS MODEL
# =============================================================================


def plot_p_channel_validation(
    validation: pd.DataFrame,
    output_directory: Path,
) -> None:

    frame = build_p_channel_validation(
        validation
    )

    if frame.empty:

        print(
            "WARNING: No P-channel validation targets found."
        )

        return

    channels = [
        "P1",
        "P2",
        "P3",
    ]

    # =========================================================================
    # MODIFIED:
    # Increase figure height to reserve enough space for
    # title + legend + value labels
    # =========================================================================

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 6.3),              # <<< MODIFIED
        sharey=True,
    )

    for (
        axis,
        hiv_status,
    ) in zip(
        axes,
        [
            "HIV-negative",
            "HIV-positive",
        ],
    ):

        subset = frame[
            frame["hiv_status"]
            == hiv_status
        ].copy()

        subset["channel"] = pd.Categorical(
            subset["channel"],
            categories=channels,
            ordered=True,
        )

        subset = subset.sort_values(
            "channel"
        )

        x = np.arange(
            len(subset)
        )

        width = 0.32                      # <<< MODIFIED


        # =====================================================================
        # OBSERVED
        # =====================================================================

        observed_bars = axis.bar(
            x - width / 2,
            subset["observed"],
            width,
            color="tab:blue",             # <<< MODIFIED
            label="Observed",
            zorder=2,
        )


        # =====================================================================
        # MODEL
        # =====================================================================

        model_bars = axis.bar(
            x + width / 2,
            subset["sim_mean"],
            width,
            yerr=subset["sim_sd"],
            color="tab:orange",           # <<< MODIFIED
            capsize=5,
            error_kw={
                "elinewidth": 1.5,
                "capthick": 1.5,
            },
            label="Model mean ± SD",
            zorder=2,
        )


        # =====================================================================
        # AXIS
        # =====================================================================

        axis.set_xticks(
            x
        )

        axis.set_xticklabels(
            [
                P_CHANNEL_LABELS[
                    str(channel)
                ]
                for channel
                in subset["channel"]
            ]
        )

        axis.set_title(
            hiv_status,
            fontsize=13,
            pad=12,
        )

        axis.set_xlabel(
            "HPV genotype group"
        )

        percent_axis(
            axis
        )

        axis.grid(
            axis="y",
            alpha=0.2,
            zorder=0,
        )

        for bar in observed_bars:

            height = bar.get_height()

            axis.annotate(
                f"{height * 100:.1f}%",
                xy=(
                    bar.get_x()
                    + bar.get_width() / 2,
                    height,
                ),
                xytext=(
                    0,
                    6,
                ),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )


        for (
            bar,
            model_mean,
            model_sd,
        ) in zip(
            model_bars,
            subset["sim_mean"],
            subset["sim_sd"],
        ):

            label_y = (
                float(model_mean)
                + float(model_sd)
            )

            axis.annotate(
                f"{model_mean * 100:.1f}%",
                xy=(
                    bar.get_x()
                    + bar.get_width() / 2,
                    label_y,
                ),
                xytext=(
                    0,
                    5,
                ),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )


        # =====================================================================
        # NEW:
        # Add headroom so value labels are not clipped
        # =====================================================================

        current_bottom, current_top = (
            axis.get_ylim()
        )

        axis.set_ylim(
            current_bottom,
            current_top * 1.10,
        )


    # =========================================================================
    # Y LABEL
    # =========================================================================

    axes[0].set_ylabel(
        "HPV prevalence"
    )


    # =========================================================================
    # MODIFIED:
    # MAIN TITLE AT TOP
    # =========================================================================

    figure.suptitle(
        (
            "Validation of HPV genotype groups "
            "corresponding to Xpert channels"
        ),
        fontsize=16,
        y=0.98,                           # <<< MODIFIED
    )


    # =========================================================================
    # MODIFIED:
    # LEGEND BELOW MAIN TITLE
    #
    # Title:
    # y = 0.98
    #
    # Legend:
    # y = 0.91
    #
    # Panels:
    # below ~0.84
    #
    # Therefore they no longer overlap.
    # =========================================================================

    handles, labels = (
        axes[0]
        .get_legend_handles_labels()
    )

    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            0.91,                         # <<< MODIFIED
        ),
        ncol=2,
        frameon=False,
    )


    # =========================================================================
    # MODIFIED:
    # Reserve top 15% for title + legend
    # =========================================================================

    figure.tight_layout(
        rect=[
            0,
            0,
            1,
            0.84,                         # <<< MODIFIED
        ]
    )


    # =========================================================================
    # SAVE
    # =========================================================================

    figure.savefig(
        output_directory
        / "07_xpert_P1_P2_P3_validation.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# =============================================================================
# FIGURE 8
# P1 / P2 / P3 RESIDUALS
# =============================================================================


def plot_p_channel_residuals(
    validation: pd.DataFrame,
    output_directory: Path,
) -> None:

    frame = build_p_channel_validation(
        validation
    )

    if frame.empty:
        return

    frame["difference_pp"] = (
        frame["sim_mean"]
        - frame["observed"]
    ) * 100

    channels = [
        "P1",
        "P2",
        "P3",
    ]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharey=True,
    )

    for (
        axis,
        hiv_status,
    ) in zip(

        axes,

        [
            "HIV-negative",
            "HIV-positive",
        ],
    ):

        subset = frame[
            frame["hiv_status"]
            == hiv_status
        ].copy()

        subset["channel"] = pd.Categorical(
            subset["channel"],
            categories=channels,
            ordered=True,
        )

        subset = subset.sort_values(
            "channel"
        )

        x = np.arange(
            len(subset)
        )

        bars = axis.bar(
            x,
            subset["difference_pp"],
            width=0.55,
        )

        axis.axhline(
            0,
            linewidth=1.5,
        )

        axis.set_xticks(
            x
        )

        axis.set_xticklabels(
            [
                P_CHANNEL_LABELS[
                    str(channel)
                ]
                for channel
                in subset["channel"]
            ]
        )

        axis.set_title(
            hiv_status
        )

        axis.set_xlabel(
            "HPV genotype group"
        )

        axis.grid(
            axis="y",
            alpha=0.2,
        )

        for (
            bar,
            difference,
        ) in zip(
            bars,
            subset["difference_pp"],
        ):

            if difference >= 0:

                y = difference + 0.3
                va = "bottom"

            else:

                y = difference - 0.3
                va = "top"

            axis.text(
                bar.get_x()
                + bar.get_width() / 2,
                y,
                f"{difference:+.1f} pp",
                ha="center",
                va=va,
                fontsize=9,
            )

    axes[0].set_ylabel(
        "Model − observed (percentage points)"
    )

    figure.suptitle(
        "Genotype-specific model discrepancy",
        y=1.02,
    )

    figure.tight_layout()

    figure.savefig(
        output_directory
        / "08_xpert_P1_P2_P3_residuals.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# =============================================================================
# FIGURE 9
# OVERALL HIV VALIDATION
# =============================================================================


def plot_overall_hiv_validation(
    validation: pd.DataFrame,
    output_directory: Path,
) -> None:

    wanted = [
        "CHAI_HRHPV_HIVneg_25_49",
        "CHAI_HRHPV_HIVpos_25_49",
    ]

    frame = validation[
        validation["target"].isin(
            wanted
        )
    ].copy()

    if frame.empty:

        print(
            "WARNING: Overall HIV validation targets not found."
        )

        return

    label_map = {
        "CHAI_HRHPV_HIVneg_25_49":
            "HIV-negative",

        "CHAI_HRHPV_HIVpos_25_49":
            "HIV-positive",
    }

    order_map = {
        target: index
        for index, target
        in enumerate(wanted)
    }

    frame["label"] = (
        frame["target"]
        .map(label_map)
    )

    frame["order"] = (
        frame["target"]
        .map(order_map)
    )

    frame = frame.sort_values(
        "order"
    )

    x = np.arange(
        len(frame)
    )

    width = 0.32

    figure, axis = plt.subplots(
        figsize=(7.5, 5.5)
    )

    observed_bars = axis.bar(
        x - width / 2,
        frame["observed"],
        width,
        label="CHAI observed",
    )

    model_bars = axis.bar(
        x + width / 2,
        frame["sim_mean"],
        width,
        yerr=frame["sim_sd"],
        capsize=4,
        label="Model mean ± SD",
    )

    axis.set_xticks(
        x
    )

    axis.set_xticklabels(
        frame["label"]
    )

    axis.set_ylabel(
        "HR-HPV prevalence"
    )

    axis.set_title(
        "Overall HR-HPV prevalence by HIV status"
    )

    percent_axis(
        axis
    )

    axis.grid(
        axis="y",
        alpha=0.2,
    )

    axis.legend(
        frameon=False
    )

    for bars in [
        observed_bars,
        model_bars,
    ]:

        for bar in bars:

            height = (
                bar.get_height()
            )

            axis.text(
                bar.get_x()
                + bar.get_width() / 2,
                height + 0.008,
                f"{height * 100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    figure.tight_layout()

    figure.savefig(
        output_directory
        / "09_overall_hiv_validation.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# =============================================================================
# FIGURE 10
# OBSERVED VS SIMULATED
# =============================================================================


def plot_observed_vs_simulated(
    validation: pd.DataFrame,
    best_beta: float,
    output_directory: Path,
) -> None:

    frame = validation.dropna(
        subset=[
            "observed",
            "sim_mean",
        ]
    ).copy()

    if frame.empty:
        return

    figure, axis = plt.subplots(
        figsize=(7.5, 7.5)
    )

    purpose_markers = {

        "calibration":
            "o",

        "HIV_validation":
            "s",

        "age_validation":
            "^",

        "genotype_validation":
            "D",

        "external_validation":
            "P",
    }

    for (
        purpose,
        purpose_frame,
    ) in frame.groupby(
        "purpose"
    ):

        marker = (
            purpose_markers
            .get(
                purpose,
                "o",
            )
        )

        axis.scatter(
            purpose_frame["observed"],
            purpose_frame["sim_mean"],
            s=75,
            marker=marker,
            alpha=0.8,
            label=(
                str(purpose)
                .replace(
                    "_",
                    " ",
                )
                .title()
            ),
        )

    lower = min(
        frame["observed"].min(),
        frame["sim_mean"].min(),
    )

    upper = max(
        frame["observed"].max(),
        frame["sim_mean"].max(),
    )

    margin = 0.03

    axis.plot(
        [
            lower - margin,
            upper + margin,
        ],
        [
            lower - margin,
            upper + margin,
        ],
        linestyle="--",
        color="black",
        linewidth=1.5,
        label="Perfect agreement",
    )

    axis.set_xlim(
        max(
            0,
            lower - margin,
        ),
        upper + margin,
    )

    axis.set_ylim(
        max(
            0,
            lower - margin,
        ),
        upper + margin,
    )

    axis.set_xlabel(
        "Observed prevalence"
    )

    axis.set_ylabel(
        "Model-predicted prevalence"
    )

    axis.set_title(
        (
            "Observed vs model-predicted prevalence\n"
            f"Calibrated $b_{{HPV}}$ = "
            f"{best_beta:.2f}"
        )
    )

    percent_x_axis(
        axis
    )

    percent_axis(
        axis
    )

    axis.grid(
        alpha=0.2
    )

    axis.legend(
        frameon=False
    )

    figure.tight_layout()

    figure.savefig(
        output_directory
        / "10_observed_vs_simulated.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# =============================================================================
# FIGURE 11
# SUPERVISOR DASHBOARD
# =============================================================================


def plot_supervisor_dashboard(
    results: pd.DataFrame,
    replicates: pd.DataFrame,
    validation: pd.DataFrame,
    output_directory: Path,
) -> None:

    results = (
        results
        .sort_values("b_hpv")
        .reset_index(drop=True)
    )

    best_row = get_best_row(
        results
    )

    best_beta = float(
        best_row["b_hpv"]
    )

    primary_target = (
        "CHAI_HRHPV_HIVneg_25_49"
    )

    observed_primary = (
        observed_prevalence(
            primary_target
        )
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
    )

    # =========================================================================
    # PANEL A
    # PRIMARY CALIBRATION
    # =========================================================================

    axis = axes[
        0,
        0,
    ]

    axis.errorbar(
        results["b_hpv"],
        results["sim_mean"],
        yerr=results["sim_sd"],
        marker="o",
        linewidth=2,
        capsize=3,
        label="Model ± SD",
    )

    axis.axhline(
        observed_primary,
        linestyle="--",
        label="CHAI observed",
    )

    axis.axvline(
        best_beta,
        linestyle=":",
        label=(
            f"Best b="
            f"{best_beta:.2f}"
        ),
    )

    axis.set_title(
        "A. Primary calibration"
    )

    axis.set_xlabel(
        "$b_{HPV}$"
    )

    axis.set_ylabel(
        "HR-HPV prevalence"
    )

    percent_axis(
        axis
    )

    axis.grid(
        alpha=0.2
    )

    axis.legend(
        frameon=False
    )

    # =========================================================================
    # PANEL B
    # STOCHASTIC STABILITY
    # =========================================================================

    axis = axes[
        0,
        1,
    ]

    best_replicates = (
        replicates[
            np.isclose(
                replicates["b_hpv"],
                best_beta,
            )
        ]
        .copy()
    )

    axis.scatter(
        best_replicates[
            "seed"
        ].astype(str),
        best_replicates[
            primary_target
        ],
        s=70,
        label="Individual seed",
    )

    axis.axhline(
        best_replicates[
            primary_target
        ].mean(),
        linewidth=2,
        label="Model mean",
    )

    axis.axhline(
        observed_primary,
        linestyle="--",
        label="CHAI observed",
    )

    axis.set_title(
        "B. Stochastic stability"
    )

    axis.set_xlabel(
        "Simulation seed"
    )

    axis.set_ylabel(
        "HR-HPV prevalence"
    )

    percent_axis(
        axis
    )

    axis.grid(
        alpha=0.2
    )

    axis.legend(
        frameon=False
    )

    # =========================================================================
    # PANEL C
    # AGE-SPECIFIC VALIDATION
    # =========================================================================

    axis = axes[
        1,
        0,
    ]

    age_frame = (
        extract_age_validation(
            best_row,
            "HIV-negative",
        )
    )

    x = np.arange(
        len(age_frame)
    )

    axis.plot(
        x,
        age_frame["observed"],
        marker="o",
        linewidth=2,
        label="CHAI observed",
    )

    axis.errorbar(
        x,
        age_frame["sim_mean"],
        yerr=age_frame["sim_sd"],
        marker="s",
        linewidth=2,
        capsize=3,
        label="Model ± SD",
    )

    axis.set_xticks(
        x
    )

    axis.set_xticklabels(
        AGE_GROUPS
    )

    axis.set_title(
        "C. Age-specific validation (HIV-negative)"
    )

    axis.set_xlabel(
        "Age group"
    )

    axis.set_ylabel(
        "HR-HPV prevalence"
    )

    percent_axis(
        axis
    )

    axis.grid(
        alpha=0.2
    )

    axis.legend(
        frameon=False
    )

    # =========================================================================
    # PANEL D
    # P1 / P2 / P3 VALIDATION
    # =========================================================================

    axis = axes[
        1,
        1,
    ]

    p_frame = (
        build_p_channel_validation(
            validation
        )
    )

    p_frame = (
        p_frame[
            p_frame["hiv_status"]
            == "HIV-negative"
        ]
        .copy()
    )

    p_frame["channel"] = (
        pd.Categorical(
            p_frame["channel"],
            categories=[
                "P1",
                "P2",
                "P3",
            ],
            ordered=True,
        )
    )

    p_frame = p_frame.sort_values(
        "channel"
    )

    x = np.arange(
        len(p_frame)
    )

    width = 0.35

    axis.bar(
        x - width / 2,
        p_frame["observed"],
        width,
        label="Observed",
    )

    axis.bar(
        x + width / 2,
        p_frame["sim_mean"],
        width,
        yerr=p_frame["sim_sd"],
        capsize=3,
        label="Model",
    )

    axis.set_xticks(
        x
    )

    axis.set_xticklabels(
        [
            P_CHANNEL_LABELS[
                str(channel)
            ]
            for channel
            in p_frame["channel"]
        ]
    )

    axis.set_title(
        "D. Genotype-group validation"
    )

    axis.set_ylabel(
        "HPV prevalence"
    )

    percent_axis(
        axis
    )

    axis.grid(
        axis="y",
        alpha=0.2,
    )

    axis.legend(
        frameon=False
    )

    # =========================================================================
    # DASHBOARD TITLE
    # =========================================================================

    figure.suptitle(
        (
            "HPV model calibration and validation\n"
            f"$b_{{HPV}}$ = {best_beta:.2f} | "
            f"{int(best_row['n_seeds'])} stochastic seeds"
        ),
        fontsize=17,
        y=0.99,
    )

    figure.tight_layout(
        rect=(
            0,
            0,
            1,
            0.94,
        )
    )

    figure.savefig(
        output_directory
        / "11_supervisor_dashboard.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# =============================================================================
# SUPERVISOR SUMMARY TABLE
# =============================================================================


def create_supervisor_summary(
    validation: pd.DataFrame,
    best_beta: float,
    output_directory: Path,
) -> None:

    summary = (
        validation
        .copy()
    )

    summary[
        "observed_percent"
    ] = (
        summary["observed"]
        * 100
    )

    summary[
        "model_percent"
    ] = (
        summary["sim_mean"]
        * 100
    )

    summary[
        "model_sd_percent"
    ] = (
        summary["sim_sd"]
        * 100
    )

    summary[
        "difference_pp"
    ] = (
        (
            summary["sim_mean"]
            - summary["observed"]
        )
        * 100
    )

    if "observed_se" in summary.columns:

        summary[
            "standardized_difference"
        ] = (
            summary["residual"]
            / summary["observed_se"]
        )

    summary.insert(
        0,
        "best_b_hpv",
        best_beta,
    )

    summary.to_csv(
        output_directory
        / "supervisor_validation_summary.csv",
        index=False,
    )


# =============================================================================
# CONSOLE SUMMARY
# =============================================================================


def print_calibration_summary(
    results: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:

    best_row = get_best_row(
        results
    )

    best_beta = float(
        best_row["b_hpv"]
    )

    observed = float(
        best_row["observed"]
    )

    simulated = float(
        best_row["sim_mean"]
    )

    difference = (
        simulated
        - observed
    )

    print()
    print("=" * 78)
    print("HPV CALIBRATION SUMMARY")
    print("=" * 78)

    print(
        f"Best b_hpv:                "
        f"{best_beta:.4f}"
    )

    print(
        f"Observed HR-HPV:           "
        f"{observed * 100:.3f}%"
    )

    print(
        f"Simulated HR-HPV:          "
        f"{simulated * 100:.3f}%"
    )

    print(
        f"Difference:                "
        f"{difference * 100:+.3f} pp"
    )

    print(
        f"Simulation SD:             "
        f"{best_row['sim_sd'] * 100:.3f} pp"
    )

    print(
        f"Simulation MCSE:           "
        f"{best_row['sim_mcse'] * 100:.3f} pp"
    )

    print(
        f"Standardized residual:     "
        f"{best_row['standardized_residual']:.3f}"
    )

    print(
        f"Calibration loss:          "
        f"{best_row['loss']:.4f}"
    )

    print(
        f"Number of seeds:           "
        f"{int(best_row['n_seeds'])}"
    )

    print()
    print("-" * 78)
    print("VALIDATION TARGETS")
    print("-" * 78)

    for _, row in validation.iterrows():

        target = str(
            row["target"]
        )

        observed_value = float(
            row["observed"]
        )

        simulated_value = float(
            row["sim_mean"]
        )

        difference_pp = (
            simulated_value
            - observed_value
        ) * 100

        print(
            f"{target:<42} "
            f"Obs={observed_value * 100:6.2f}%  "
            f"Model={simulated_value * 100:6.2f}%  "
            f"Diff={difference_pp:+6.2f} pp"
        )

    print("=" * 78)
    print()


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Create supervisor-ready HPV calibration "
            "and validation figures."
        )
    )

    parser.add_argument(
        "result_directory",
        type=Path,
        help=(
            "Directory containing "
            "grid_search_results.csv, "
            "grid_search_replicates.csv and "
            "best_b_hpv_validation.csv."
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help=(
            "Optional output directory. "
            "Default: <result_directory>/"
            "supervisor_calibration_figures"
        ),
    )

    arguments = (
        parser.parse_args()
    )

    result_directory = (
        arguments.result_directory
    )

    if not result_directory.exists():

        raise FileNotFoundError(
            f"Result directory does not exist:\n"
            f"{result_directory}"
        )

    output_directory = (
        arguments.output_directory
        or result_directory
        / "supervisor_calibration_figures"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================================
    # LOAD
    # =========================================================================

    (
        results,
        replicates,
        validation,
    ) = load_inputs(
        result_directory
    )

    best_row = get_best_row(
        results
    )

    best_beta = get_best_beta(
        results
    )

    # =========================================================================
    # PRINT BASIC INFORMATION
    # =========================================================================

    print()
    print("=" * 78)
    print("HPV CALIBRATION FIGURE GENERATOR")
    print("=" * 78)

    print(
        "Tested b_hpv values:"
    )

    print(
        sorted(
            results[
                "b_hpv"
            ].unique()
        )
    )

    print()

    print(
        "Simulation seeds:"
    )

    print(
        sorted(
            replicates[
                "seed"
            ].unique()
        )
    )

    print()

    print(
        f"Minimum-loss b_hpv: "
        f"{best_beta:.4f}"
    )

    # =========================================================================
    # GENERATE FIGURES
    # =========================================================================

    print()
    print(
        "Generating Figure 1..."
    )

    plot_primary_calibration(
        results,
        output_directory,
    )

    print(
        "Generating Figure 2..."
    )

    plot_calibration_loss(
        results,
        output_directory,
    )

    print(
        "Generating Figure 3..."
    )

    plot_seed_stability(
        replicates,
        results,
        output_directory,
    )

    print(
        "Generating Figure 4..."
    )

    plot_age_validation_single(
        best_row,
        "HIV-negative",
        output_directory,
        "04_age_validation_hiv_negative.png",
    )

    print(
        "Generating Figure 5..."
    )

    plot_age_validation_single(
        best_row,
        "HIV-positive",
        output_directory,
        "05_age_validation_hiv_positive.png",
    )

    print(
        "Generating Figure 6..."
    )

    plot_age_validation_combined(
        best_row,
        output_directory,
    )

    print(
        "Generating P1/P2/P3 prevalence trajectories..."
    )

    # Separate figures:
    # P1 vs b_hpv
    # P2 vs b_hpv
    # P3 vs b_hpv
    plot_all_p_channels_by_beta(
        results,
        validation,
        output_directory,
    )

    # Combined three-panel figure
    plot_p_channels_combined_by_beta(
        results,
        validation,
        output_directory,
    )

    print(
        "Generating Figure 7..."
    )

    plot_p_channel_validation(
        validation,
        output_directory,
    )

    print(
        "Generating Figure 8..."
    )

    plot_p_channel_residuals(
        validation,
        output_directory,
    )

    print(
        "Generating Figure 9..."
    )

    plot_overall_hiv_validation(
        validation,
        output_directory,
    )

    print(
        "Generating Figure 10..."
    )

    plot_observed_vs_simulated(
        validation,
        best_beta,
        output_directory,
    )

    print(
        "Generating Figure 11..."
    )

    plot_supervisor_dashboard(
        results,
        replicates,
        validation,
        output_directory,
    )

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================

    create_supervisor_summary(
        validation,
        best_beta,
        output_directory,
    )

    # =========================================================================
    # CONSOLE SUMMARY
    # =========================================================================

    print_calibration_summary(
        results,
        validation,
    )

    # =========================================================================
    # FINISH
    # =========================================================================

    print("=" * 78)
    print("OUTPUT COMPLETE")
    print("=" * 78)

    print(
        f"\nFigures saved to:\n"
        f"{output_directory}\n"
    )

    print(
        "Recommended figures for supervisor meeting:"
    )

    print(
        "  01_primary_calibration_by_beta.png"
    )

    print(
        "  03_seed_stability.png"
    )

    print(
        "  06_age_validation_combined.png"
    )

    print(
        "  07_xpert_P1_P2_P3_validation.png"
    )

    print(
        "  08_xpert_P1_P2_P3_residuals.png"
    )

    print(
        "  11_supervisor_dashboard.png"
    )

    print()

    print(
        "Summary table:"
    )

    print(
        "  supervisor_validation_summary.csv"
    )

    print()


# =============================================================================
# RUN
# =============================================================================


if __name__ == "__main__":
    main()
