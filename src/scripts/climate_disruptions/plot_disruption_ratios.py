import pickle
from pathlib import Path
import pandas as pd

for mode in ['main_text_mode_1', 'main_text_mode_2']:
    base = Path('/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk') / mode
    print(f'\n=== {mode} ===')

    # just look at draw 1 (Default), run 0
    log_path = base / '1' / '0' / 'tlo.methods.healthsystem.summary.pickle'
    with open(log_path, 'rb') as f:
        log = pickle.load(f)

    for key in ['Weather_delayed_HSI_Event_full_info', 'Weather_cancelled_HSI_Event_full_info']:
        if key not in log:
            print(f'  {key}: not found')
            continue
        df = log[key]
        total_events = len(df)
        unique_person_treatment = df.groupby(['Person_ID', 'TREATMENT_ID']).size()
        unique_combinations = len(unique_person_treatment)
        max_repeats = unique_person_treatment.max()
        mean_repeats = unique_person_treatment.mean()
        print(f'  {key}:')
        print(f'    Total events:            {total_events:,}')
        print(f'    Unique person×treatment: {unique_combinations:,}')
        print(f'    Mean disruptions per unique appointment: {mean_repeats:.2f}')
        print(f'    Max disruptions for one appointment:     {max_repeats}')

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FS_TICK = 13
FS_LABEL = 15
FS_TITLE = 16
FS_LEGEND = 12

COLOURS = ["#EDC7CF", "#ADB993", "#6F8AB7", "#C9B8D8"]


def load_hsi_counts(folder: Path, suffix: str, season: str) -> pd.DataFrame:
    """
    Load hsi_ran_counts_by_season_{suffix}.csv, filter to the requested season,
    and normalise scenario name casing.
    """
    p = folder / f"hsi_ran_counts_by_season_{suffix}.csv"
    df = pd.read_csv(p)
    df["Scenario"] = df["Scenario"].str.strip().str.title()
    df = df[df["season"] == season]
    return df.set_index("Scenario")


def ratio_with_ci(
    num_mean: float, num_lo: float, num_hi: float,
    den_mean: float, den_lo: float, den_hi: float,
) -> tuple[float, float, float]:
    """Point estimate + 95 % CI via delta method."""
    if den_mean == 0:
        return np.nan, np.nan, np.nan
    r = num_mean / den_mean
    se_num = (num_hi - num_lo) / (2 * 1.96)
    se_den = (den_hi - den_lo) / (2 * 1.96)
    frac_var = (se_num / num_mean) ** 2 + (se_den / den_mean) ** 2 if num_mean != 0 else 0
    se_r = r * np.sqrt(frac_var)
    return r, r - 1.96 * se_r, r + 1.96 * se_r


def compute_ratio(
    num_df: pd.DataFrame, num_scen: str,
    den_df: pd.DataFrame, den_scen: str,
) -> tuple[float, float, float]:
    def _get(df, scen, col):
        if scen not in df.index:
            raise KeyError(f"Scenario '{scen}' not found. Available: {list(df.index)}")
        return float(df.loc[scen, col])

    return ratio_with_ci(
        _get(num_df, num_scen, "total_hsi_ran_mean"),
        _get(num_df, num_scen, "total_hsi_ran_lower"),
        _get(num_df, num_scen, "total_hsi_ran_upper"),
        _get(den_df, den_scen, "total_hsi_ran_mean"),
        _get(den_df, den_scen, "total_hsi_ran_lower"),
        _get(den_df, den_scen, "total_hsi_ran_upper"),
    )


# ── Plot helpers ──────────────────────────────────────────────────────────────

def draw_ratio_bar(ax, point, lo, hi, title, panel_letter, colour, season_label):
    ax.bar([0], [point], color=colour, alpha=0.85, width=0.4, zorder=2)
    ax.errorbar(
        [0], [point],
        yerr=[[point - lo], [hi - point]],
        fmt="none", color="black", lw=1.8, capsize=8, capthick=1.8, zorder=3,
    )
    ax.axhline(1.0, color="grey", lw=1.2, ls="--", alpha=0.7)
    ax.set_xticks([])
    ax.set_ylabel(f"Ratio of total HSIs ran)",
                  fontsize=FS_LABEL, fontweight="bold")
    ax.set_title(f"({panel_letter})  {title}", fontsize=FS_TITLE,
                 fontweight="bold", loc="left")
    ax.set_ylim(bottom=0)
    plt.setp(ax.yaxis.get_majorticklabels(), fontsize=FS_TICK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.text(0, hi * 1.02, f"{point:.4f}\n[{lo:.4f}, {hi:.4f}]",
            ha="center", va="bottom", fontsize=FS_LEGEND)
    ax.legend(fontsize=FS_LEGEND, framealpha=0.9)


# ── Main ──────────────────────────────────────────────────────────────────────

def apply(mode1_folder: Path, mode2_folder: Path, season: str, output_folder: Path):
    season_label = "(wet season)" if season == "wet_season" else "full year"

    m1 = load_hsi_counts(mode1_folder, "main_text_mode_1", season)
    m2 = load_hsi_counts(mode2_folder, "main_text_mode_2", season)

    print(f"Season: {season_label}")
    print(f"Mode-1 scenarios: {list(m1.index)}")
    print(f"Mode-2 scenarios: {list(m2.index)}")
    print()
    print("HSI counts loaded:")
    for scen in m1.index:
        print(f"  Mode 1 | {scen}: {m1.loc[scen, 'total_hsi_ran_mean']:,.0f} "
              f"[{m1.loc[scen, 'total_hsi_ran_lower']:,.0f}, "
              f"{m1.loc[scen, 'total_hsi_ran_upper']:,.0f}]")
    for scen in m2.index:
        print(f"  Mode 2 | {scen}: {m2.loc[scen, 'total_hsi_ran_mean']:,.0f} "
              f"[{m2.loc[scen, 'total_hsi_ran_lower']:,.0f}, "
              f"{m2.loc[scen, 'total_hsi_ran_upper']:,.0f}]")

    ratios = [
        ("A",
         *compute_ratio(m1, "Default", m1, "No Disruptions"),
         "Default mode 1 / No Disruptions mode 1"),
        ("B",
         *compute_ratio(m2, "No Disruptions", m1, "No Disruptions"),
         "No Disruptions mode 2 / No Disruptions mode 1"),
        ("C",
         *compute_ratio(m2, "Default", m1, "No Disruptions"),
         "Default mode 2 / No Disruptions mode 1"),
        ("D",
         *compute_ratio(m2, "Default", m2, "No Disruptions"),
         "Default mode 2 / No Disruptions mode 2"),
    ]

    # ── 2×2 combined figure ───────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (letter, r, lo, hi, label), col in zip(
        axes.flatten(), ratios, COLOURS
    ):
        draw_ratio_bar(ax, r, lo, hi, f"{label})",
                       letter, col, season_label)

    fig.suptitle(
        f"Ratio of total HSIs ran — {season_label} (2025–2040)",
        fontsize=FS_TITLE, fontweight="bold",
    )
    fig.tight_layout()
    out = output_folder / f"disruption_ratios_2x2_{season}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")

    # ── Summary bar chart ─────────────────────────────────────────────────────
    tick_labels = [
        "A\nClimate\n(mode 1)",
        "B\nMode 2\n(no climate)",
        "C\nClimate +\nmode 2",
        "D\nClimate\n(mode 2)",
    ]
    points = [r for _, r, *_ in ratios]
    lowers = [r - lo for _, r, lo, *_ in ratios]
    uppers = [hi - r for _, r, lo, hi, *_ in ratios]

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(range(4), points, color=COLOURS, alpha=0.85, width=0.5, zorder=2)
    ax2.errorbar(range(4), points, yerr=[lowers, uppers],
                 fmt="none", color="black", lw=1.8, capsize=8, capthick=1.8, zorder=3)
    ax2.axhline(1.0, color="grey", lw=1.2, ls="--", alpha=0.7)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(tick_labels, fontsize=FS_TICK)
    ax2.set_ylabel(f"Ratio of total HSIs ran\n({season_label})",
                   fontsize=FS_LABEL, fontweight="bold")
    ax2.set_title(
        f"Ratio of total HSIs ran (2025–2040)",
        fontsize=FS_TITLE, fontweight="bold",
    )
    ax2.set_ylim(bottom=0)
    plt.setp(ax2.yaxis.get_majorticklabels(), fontsize=FS_TICK)
    ax2.legend(fontsize=FS_LEGEND, framealpha=0.9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig2.tight_layout()
    out2 = output_folder / f"disruption_ratios_summary_bar_{season}.png"
    fig2.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {out2}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    rows = [
        {
            "panel": letter,
            "season": season,
            "numerator": label.split(" / ")[0],
            "denominator": label.split(" / ")[1],
            "ratio": round(r, 6),
            "ci_lower": round(lo, 6),
            "ci_upper": round(hi, 6),
        }
        for letter, r, lo, hi, label in ratios
    ]
    df_out = pd.DataFrame(rows)
    csv_out = output_folder / f"disruption_ratios_summary_{season}.csv"
    df_out.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}")
    print()
    print(df_out.to_string(index=False))

    # ── Load disruption counts from main text summary ─────────────────────────
    def load_summary(folder: Path, suffix: str) -> pd.DataFrame:
        p = folder / f"main_text_summary_{suffix}.csv"
        df = pd.read_csv(p)
        df["Scenario"] = df["Scenario"].str.strip().str.title()
        return df.set_index("Scenario")

    sum_m1 = load_summary(mode1_folder, "main_text_mode_1")
    sum_m2 = load_summary(mode2_folder, "main_text_mode_2")

    # ── Plot: HSIs ran vs disrupted — Default scenario, two panels ────────────

    modes = ["Mode 1", "Mode 2"]
    colours_ran = ["#EDC7CF", "#6F8AB7"]
    colours_dis = ["#C0392B", "#2C3E7A"]

    ran_means = [m1.loc["Default", "total_hsi_ran_mean"],
                 m2.loc["Default", "total_hsi_ran_mean"]]
    ran_lowers = [m1.loc["Default", "total_hsi_ran_mean"] - m1.loc["Default", "total_hsi_ran_lower"],
                  m2.loc["Default", "total_hsi_ran_mean"] - m2.loc["Default", "total_hsi_ran_lower"]]
    ran_uppers = [m1.loc["Default", "total_hsi_ran_upper"] - m1.loc["Default", "total_hsi_ran_mean"],
                  m2.loc["Default", "total_hsi_ran_upper"] - m2.loc["Default", "total_hsi_ran_mean"]]

    dis_means = [sum_m1.loc["Default", "total_hsi_disrupted_mean"],
                 sum_m2.loc["Default", "total_hsi_disrupted_mean"]]
    dis_lowers = [
        sum_m1.loc["Default", "total_hsi_disrupted_mean"] - sum_m1.loc["Default", "total_hsi_disrupted_lower"],
        sum_m2.loc["Default", "total_hsi_disrupted_mean"] - sum_m2.loc["Default", "total_hsi_disrupted_lower"]]
    dis_uppers = [
        sum_m1.loc["Default", "total_hsi_disrupted_upper"] - sum_m1.loc["Default", "total_hsi_disrupted_mean"],
        sum_m2.loc["Default", "total_hsi_disrupted_upper"] - sum_m2.loc["Default", "total_hsi_disrupted_mean"]]

    fig3, (ax_ran, ax_dis) = plt.subplots(1, 2, figsize=(10, 5))

    for ax, means, lowers, uppers, colours, ylabel, title in [
        (ax_ran, ran_means, ran_lowers, ran_uppers, colours_ran,
         "Total HSIs ran (2025–2040)", "(A) HSIs ran"),
        (ax_dis, dis_means, dis_lowers, dis_uppers, colours_dis,
         "HSIs weather-disrupted (2025–2040)", "(B) HSIs weather disrupted"),
    ]:
        ax.bar(modes, means, color=colours, alpha=0.85, width=0.5, zorder=2)
        ax.errorbar(modes, means, yerr=[lowers, uppers],
                    fmt="none", color="black", lw=1.5, capsize=6, capthick=1.5, zorder=3)
        ax.set_ylabel(ylabel, fontsize=FS_LABEL, fontweight="bold")
        ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", loc="left")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(FS_TICK)
        plt.setp(ax.yaxis.get_majorticklabels(), fontsize=FS_TICK)
        plt.setp(ax.xaxis.get_majorticklabels(), fontsize=FS_TICK)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig3.suptitle("Default scenario: HSIs ran vs disrupted by mode (2025–2040)",
                  fontsize=FS_TITLE, fontweight="bold")
    fig3.tight_layout()
    out3 = output_folder / f"hsi_ran_vs_disrupted_default_{season}.png"
    fig3.savefig(out3, dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved: {out3}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode1_folder", type=Path, required=True)
    parser.add_argument("--mode2_folder", type=Path, required=True)
    parser.add_argument("--season", type=str, default="wet_season",
                        choices=["wet_season", "full_year"],
                        help="wet_season = Nov - Apr only; full_year = all months")
    parser.add_argument("--output_folder", type=Path, default=None)
    args = parser.parse_args()

    out = args.output_folder or args.mode2_folder
    out.mkdir(parents=True, exist_ok=True)
    apply(mode1_folder=args.mode1_folder, mode2_folder=args.mode2_folder,
          season=args.season, output_folder=out)
