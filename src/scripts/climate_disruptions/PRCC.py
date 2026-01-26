"""
PRCC Sensitivity Analysis Script
Produces publication-ready PRCC figure for the climate disruption sensitivity analysis.

Outputs:
- (A1) PRCC for Total DALYs
- (A2) PRCC for Proportion Delayed
- (A3) PRCC for Proportion Cancelled
- (B) Parameter correlation matrix (LHS validation)
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata

# =============================================================================
# Configuration
# =============================================================================

results_folder = Path(
    '/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/climate_scenario_runs_baseline-2025-12-04T163755Z')
output_folder = Path(
    '/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/climate_scenario_runs_baseline-2025-12-04T163755Z')

# Analysis parameters
N_DRAWS = 200
LHS_FILE = Path('/Users/rem76/PycharmProjects/TLOmodel/resources/lhs_grid.json')  # Adjust path as needed

PARAMETER_INFO = {
    "rescaling_prob_disruption": {
        "symbol": "α",
        "name": "Disruption scaling",
    },
    "rescaling_prob_seeking_after_disruption": {
        "symbol": "β",
        "name": "Re-seeking scaling",
    },
    "scale_factor_delay_in_seeking_care_weather": {
        "symbol": "δ",
        "name": "Base delay",
    },
    "scale_factor_priority_and_delay": {
        "symbol": "γ",
        "name": "Priority scaling",
    },
    "scale_factor_severity_disruption_and_delay": {
        "symbol": "ε",
        "name": "Severity scaling",
    },
}


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_prcc(params_df: pd.DataFrame, outcome_series: pd.Series) -> pd.DataFrame:
    """Calculate Partial Rank Correlation Coefficients (PRCC)."""

    common_idx = params_df.index.intersection(outcome_series.index)
    params_aligned = params_df.loc[common_idx]
    outcome_aligned = outcome_series.loc[common_idx]

    valid_mask = ~(params_aligned.isna().any(axis=1) | outcome_aligned.isna())
    params_clean = params_aligned[valid_mask]
    outcome_clean = outcome_aligned[valid_mask]

    ranked_params = params_clean.apply(rankdata)
    ranked_outcome = rankdata(outcome_clean)

    results = []
    param_names = params_clean.columns.tolist()

    for target_param in param_names:
        other_params = [p for p in param_names if p != target_param]

        if not other_params:
            corr, p_val = spearmanr(
                params_clean[target_param], outcome_clean
            )
        else:
            X = ranked_params[other_params].values
            y_target = ranked_params[target_param].values
            X_int = np.column_stack([np.ones(len(y_target)), X])

            beta_target = np.linalg.lstsq(X_int, y_target, rcond=None)[0]
            resid_target = y_target - X_int @ beta_target

            beta_outcome = np.linalg.lstsq(X_int, ranked_outcome, rcond=None)[0]
            resid_outcome = ranked_outcome - X_int @ beta_outcome

            corr, p_val = spearmanr(resid_target, resid_outcome)

        results.append(
            {
                "parameter": target_param,
                "prcc": corr,
                "p_value": p_val,
            }
        )

    return pd.DataFrame(results)


def plot_prcc_horizontal_bars(prcc_results, outcome_name, ax):
    """Horizontal PRCC bar plot."""

    prcc_sorted = prcc_results.copy()
    prcc_sorted["abs_prcc"] = prcc_sorted["prcc"].abs()
    prcc_sorted = prcc_sorted.sort_values("abs_prcc", ascending=True).reset_index(drop=True)

    labels = []
    for param in prcc_sorted["parameter"]:
        info = PARAMETER_INFO.get(param)
        labels.append(
            f"{info['symbol']} ({info['name']})" if info else param
        )

    colors = ["#DC2626" if x > 0 else "#2563EB" for x in prcc_sorted["prcc"]]

    y_pos = np.arange(len(prcc_sorted))
    ax.barh(
        y_pos,
        prcc_sorted["prcc"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        height=0.7,
        alpha=0.85,
    )

    for idx, row in prcc_sorted.iterrows():
        if row["p_value"] < 0.001:
            marker = "***"
        elif row["p_value"] < 0.01:
            marker = "**"
        elif row["p_value"] < 0.05:
            marker = "*"
        else:
            marker = ""

        if marker:
            ax.text(
                row["prcc"] + (0.03 if row["prcc"] > 0 else -0.06),
                idx,
                marker,
                va="center",
                ha="left" if row["prcc"] > 0 else "right",
                fontsize=12,
                color="#FCD34D",
                fontweight="bold",
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlim(-1.1, 1.1)
    ax.set_xlabel("Partial Rank Correlation Coefficient")
    ax.set_title(outcome_name, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.3)


def plot_parameter_correlation_matrix(params_df, ax):
    """Spearman correlation matrix for LHS validation."""

    corr = params_df.corr(method="spearman")
    labels = [
        PARAMETER_INFO.get(p, {}).get("symbol", p[:6])
        for p in corr.columns
    ]

    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Spearman correlation")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12, style="italic")
    ax.set_yticklabels(labels, fontsize=12, style="italic")
    ax.set_title("Parameter correlations (LHS validation)", fontweight="bold")


def create_combined_prcc_figure(params_df, outputs):
    """Assemble full PRCC figure."""

    fig = plt.figure(figsize=(16, 10))

    ax1 = fig.add_axes([0.05, 0.55, 0.28, 0.4])
    ax2 = fig.add_axes([0.38, 0.55, 0.28, 0.4])
    ax3 = fig.add_axes([0.71, 0.55, 0.28, 0.4])
    ax4 = fig.add_axes([0.25, 0.08, 0.5, 0.38])

    for i, (name, series) in enumerate(
        {
            "Total DALYs": outputs["total_dalys"],
            "Prop. delayed": outputs["prop_delayed"],
            "Prop. cancelled": outputs["prop_cancelled"],
        }.items()
    ):
        ax = [ax1, ax2, ax3][i]
        prcc = calculate_prcc(params_df, series)
        plot_prcc_horizontal_bars(prcc, name, ax)
        ax.text(-0.12, 1.05, f"(A{i+1})", transform=ax.transAxes, fontweight="bold")

    plot_parameter_correlation_matrix(params_df, ax4)
    ax4.text(-0.08, 1.08, "(B)", transform=ax4.transAxes, fontweight="bold")

    fig.text(
        0.5,
        0.01,
        "* p<0.05   ** p<0.01   *** p<0.001",
        ha="center",
        fontsize=10,
    )

    return fig


def load_outputs_from_results(results_folder: Path) -> dict:
    """
    Load or compute output metrics from simulation results.

    Adapt file paths and column names to match your actual results structure.
    """

    # Option 1: If you have a summary CSV with all draws
    summary_file = results_folder / "summary_outcomes.csv"
    if summary_file.exists():
        df = pd.read_csv(summary_file)
        return {
            "total_dalys": df.set_index("draw")["total_dalys"],
            "prop_delayed": df.set_index("draw")["prop_delayed"],
            "prop_cancelled": df.set_index("draw")["prop_cancelled"],
        }

    # Option 2: Load from individual draw folders
    total_dalys = {}
    prop_delayed = {}
    prop_cancelled = {}

    for draw in range(N_DRAWS):
        draw_folder = results_folder / f"draw_{draw}"
        if not draw_folder.exists():
            continue

        try:
            results_file = draw_folder / "results.csv"
            if results_file.exists():
                df = pd.read_csv(results_file)
                total_dalys[draw] = df["dalys"].sum() if "dalys" in df.columns else np.nan
                prop_delayed[draw] = df["delayed"].mean() if "delayed" in df.columns else np.nan
                prop_cancelled[draw] = df["cancelled"].mean() if "cancelled" in df.columns else np.nan
        except Exception as e:
            print(f"Warning: Could not load draw {draw}: {e}")
            continue

    return {
        "total_dalys": pd.Series(total_dalys, name="total_dalys"),
        "prop_delayed": pd.Series(prop_delayed, name="prop_delayed"),
        "prop_cancelled": pd.Series(prop_cancelled, name="prop_cancelled"),
    }


def apply(results_folder: Path, output_folder: Path):
    """
    Produce a publication-ready PRCC figure for the climate disruption
    sensitivity analysis.
    """

    with open(LHS_FILE) as f:
        lhs_grid = json.load(f)

    records = []
    for i, draw in enumerate(lhs_grid[:N_DRAWS]):
        record = {"draw": i}
        params = draw.get("HealthSystem", draw)
        for p in PARAMETER_INFO:
            val = params.get(p, np.nan)
            if isinstance(val, list):
                val = val[0] if val else np.nan
            record[p] = val
        records.append(record)

    params_df = pd.DataFrame(records).set_index("draw").dropna(axis=1, how="all")

    outputs = load_outputs_from_results(results_folder)

    fig = create_combined_prcc_figure(params_df, outputs)

    fig.savefig(output_folder / "prcc_combined_figure.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
    )
