"""
PRCC Sensitivity Analysis Script
Produces publication-ready PRCC figure for the climate disruption sensitivity analysis.

Outputs:
- (A) PRCC for Total DALYs
- (B) PRCC for Proportion Delayed
- (C) PRCC for Proportion Cancelled
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

# =============================================================================
# Configuration
# =============================================================================

N_DRAWS = 50

PARAMETER_INFO = {
    "scale_factor_prob_disruption": {
        "symbol": "α",
        "name": "Disruption scaling",
    },
    "scale_factor_reseeking_healthcare_post_disruption": {
        "symbol": "β",
        "name": "Re-seeking scaling",
    },
    "delay_in_seeking_care_weather": {
        "symbol": "δ",
        "name": "Base delay",
    },
    "scale_factor_appointment_urgency": {
        "symbol": "γ",
        "name": "Priority scaling",
    },
    "scale_factor_severity_disruption_and_delay": {
        "symbol": "ε",
        "name": "Severity scaling",
    },
}

# ── Publication rcParams ──────────────────────────────────────────────────────
PUB_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

COLOUR_POS = "#0072B2"
COLOUR_NEG = "#D55E00"
COLOUR_NS = "#999999"


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
            corr, p_val = spearmanr(params_clean[target_param], outcome_clean)
        else:
            X = ranked_params[other_params].values
            y_t = ranked_params[target_param].values
            X_int = np.column_stack([np.ones(len(y_t)), X])

            resid_target = y_t - X_int @ np.linalg.lstsq(X_int, y_t, rcond=None)[0]
            resid_outcome = ranked_outcome - X_int @ np.linalg.lstsq(X_int, ranked_outcome, rcond=None)[0]

            corr, p_val = spearmanr(resid_target, resid_outcome)

        results.append({"parameter": target_param, "prcc": corr, "p_value": p_val})

    return pd.DataFrame(results)


def _sig_label(p_val: float) -> str:
    if p_val < 0.001: return "***"
    if p_val < 0.01:  return "**"
    if p_val < 0.05:  return "*"
    return "ns"


def plot_prcc_horizontal_bars(prcc_results, outcome_name, ax, show_ylabel=True):
    """Publication-quality horizontal PRCC bar plot."""

    df = prcc_results.copy()
    df["abs_prcc"] = df["prcc"].abs()
    df = df.sort_values("abs_prcc", ascending=True).reset_index(drop=True)

    labels = []
    for param in df["parameter"]:
        info = PARAMETER_INFO.get(param)
        labels.append(f"{info['symbol']} — {info['name']}" if info else param)

    colors = []
    for _, row in df.iterrows():
        if row["p_value"] >= 0.05:
            colors.append(COLOUR_NS)
        elif row["prcc"] > 0:
            colors.append(COLOUR_POS)
        else:
            colors.append(COLOUR_NEG)

    y_pos = np.arange(len(df))

    ax.barh(
        y_pos, df["prcc"],
        color=colors, edgecolor="white", linewidth=0.4,
        height=0.62, alpha=0.92, zorder=3,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels if show_ylabel else [""] * len(labels), fontsize=7)

    ax.axvline(0, color="black", linewidth=0.8, zorder=4)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel("PRCC", fontsize=8)
    ax.set_title(outcome_name, fontweight="bold", fontsize=9, pad=4)

    ax.xaxis.grid(True, linestyle=":", linewidth=0.4, color="0.75", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_outputs_from_results(results_folder: Path) -> dict:
    """Load pre-computed disruption and DALY outputs."""

    disruption_file = results_folder / "prcc_disruption_summary.csv"
    if not disruption_file.exists():
        raise FileNotFoundError(
            f"Run comparison_actual_vs_expected_disruption_realfacility.py first:\n  {disruption_file}"
        )
    disruption_df = pd.read_csv(disruption_file).set_index("draw")

    daly_file = results_folder / "dalys_by_cause_all_draws_parameter_SA.csv"
    if not daly_file.exists():
        raise FileNotFoundError(
            f"Run the DALY analysis script first:\n  {daly_file}"
        )
    daly_df = pd.read_csv(daly_file, index_col=0)

    total_dalys = daly_df.sum(axis=0)
    total_dalys.index = [int(c.replace("Draw ", "")) for c in total_dalys.index]
    total_dalys = total_dalys.sort_index()

    common = disruption_df.index.intersection(total_dalys.index)
    if len(common) < len(disruption_df):
        print(f"  Warning: {len(disruption_df) - len(common)} draws missing from DALY file")

    for name, series in [
        ("total_dalys", total_dalys),
        ("prop_delayed", disruption_df["prop_delayed"]),
        ("prop_cancelled", disruption_df["prop_cancelled"]),
    ]:
        print(f"  {name}: {series.notna().sum()} valid draws  "
              f"[{series.min():.4g} – {series.max():.4g}]")

    return {
        "total_dalys": total_dalys,
        "prop_delayed": disruption_df["prop_delayed"],
        "prop_cancelled": disruption_df["prop_cancelled"],
    }


def create_combined_prcc_figure(params_df, outputs, output_folder):
    """Assemble publication-ready PRCC figure (190 mm wide, double-column)."""

    outcome_items = [
        ("total_dalys", "Total DALYs"),
        ("prop_delayed", "Prop. delayed"),
        ("prop_cancelled", "Prop. cancelled"),
    ]

    with plt.rc_context(PUB_RC):
        fig = plt.figure(figsize=(7.48, 3.2))

        gs = fig.add_gridspec(
            1, 3,
            left=0.13, right=0.98,
            top=0.93, bottom=0.22,
            wspace=0.42,
        )
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

        for i, (out_key, out_label) in enumerate(outcome_items):
            ax = axes[i]
            prcc = calculate_prcc(params_df, outputs[out_key])

            # Save PRCC values for this outcome
            prcc.to_csv(output_folder / f"prcc_{out_key}.csv", index=False)

            plot_prcc_horizontal_bars(prcc, out_label, ax, show_ylabel=(i == 0))
            ax.text(
                -0.14 if i == 0 else -0.05, 1.06,
                f"({chr(65 + i)})",
                transform=ax.transAxes,
                fontweight="bold", fontsize=9,
            )

        # Figure-level legend, centred under all three panels
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLOUR_POS, label="Positive (p < 0.05)"),
            Patch(facecolor=COLOUR_NEG, label="Negative (p < 0.05)"),
            Patch(facecolor=COLOUR_NS, label="Non-significant"),
        ]
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=3,
            fontsize=6.5,
            frameon=True,
            edgecolor="0.7",
            fancybox=False,
        )

    return fig


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path):
    """Produce a publication-ready PRCC figure for the climate disruption sensitivity analysis."""

    lhs_file = Path("/Users/rem76/PycharmProjects/TLOmodel/src/scripts/climate_disruptions/lhs_parameter_draws.json")
    if not lhs_file.exists():
        raise FileNotFoundError(f"LHS draws file not found:\n  {lhs_file}")

    with open(lhs_file) as f:
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

    fig = create_combined_prcc_figure(params_df, outputs, output_folder)

    out_path = output_folder / "prcc_combined_figure.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


import pandas as pd

df = pd.read_csv("prcc_total_dalys.csv")
print(df[df["parameter"].isin(["scale_factor_prob_disruption",
                               "scale_factor_reseeking_healthcare_post_disruption"])]
      [["parameter", "prcc", "p_value"]])
# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Produce PRCC sensitivity figure for climate disruption analysis."
    )
    parser.add_argument("results_folder", type=Path, )
    parser.add_argument("--resourcefilepath", type=Path, default=Path("./resources"), )
    args = parser.parse_args()

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=args.resourcefilepath,
    )
