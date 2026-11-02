from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Observed values used by the calibration code.
OBSERVED = {
    "HRHPV_non_WLWH": (181 / 625, 625),
    "HRHPV_WLWH": (295 / 625, 625),
    "HPV16_non_WLWH": (35 / 625, 625),
    "HPV16_WLWH": (63 / 625, 625),
    "HPV18_45_non_WLWH": (47 / 625, 625),
    "HPV18_45_WLWH": (51 / 625, 625),
    "P3_non_WLWH": (93 / 625, 625),
    "P3_WLWH": (168 / 625, 625),
}


# Column names written by grid_search_results.csv.
GROUPS = [
    (
        "HR-HPV",
        "hrhpv",
        [
            ("sim_mean", "sim_sd", "HRHPV_non_WLWH", "Non-WLWH"),
            (
                "Chinula_2026_HRHPV_WLWH_25_50__sim_mean",
                "Chinula_2026_HRHPV_WLWH_25_50__sim_sd",
                "HRHPV_WLWH",
                "WLWH",
            ),
        ],
    ),
    (
        "HPV16",
        "hpv16",
        [
            (
                "Chinula_2026_HPV16_non_WLWH_25_50__sim_mean",
                "Chinula_2026_HPV16_non_WLWH_25_50__sim_sd",
                "HPV16_non_WLWH",
                "Non-WLWH",
            ),
            (
                "Chinula_2026_HPV16_WLWH_25_50__sim_mean",
                "Chinula_2026_HPV16_WLWH_25_50__sim_sd",
                "HPV16_WLWH",
                "WLWH",
            ),
        ],
    ),
    (
        "HPV18/45",
        "hpv18_45",
        [
            (
                "Chinula_2026_HPV18_45_non_WLWH_25_50__sim_mean",
                "Chinula_2026_HPV18_45_non_WLWH_25_50__sim_sd",
                "HPV18_45_non_WLWH",
                "Non-WLWH",
            ),
            (
                "Chinula_2026_HPV18_45_WLWH_25_50__sim_mean",
                "Chinula_2026_HPV18_45_WLWH_25_50__sim_sd",
                "HPV18_45_WLWH",
                "WLWH",
            ),
        ],
    ),
    (
        "HPV31/33/35/52/58 (P3)",
        "p3",
        [
            (
                "Chinula_2026_P3_non_WLWH_25_50__sim_mean",
                "Chinula_2026_P3_non_WLWH_25_50__sim_sd",
                "P3_non_WLWH",
                "Non-WLWH",
            ),
            (
                "Chinula_2026_P3_WLWH_25_50__sim_mean",
                "Chinula_2026_P3_WLWH_25_50__sim_sd",
                "P3_WLWH",
                "WLWH",
            ),
        ],
    ),
]


# Map summary-table mean columns to the corresponding replicate-table columns.
SUMMARY_TO_RAW = {
    "sim_mean": "Chinula_2026_HRHPV_non_WLWH_25_50",
    "Chinula_2026_HRHPV_WLWH_25_50__sim_mean": "Chinula_2026_HRHPV_WLWH_25_50",
    "Chinula_2026_HPV16_non_WLWH_25_50__sim_mean": "Chinula_2026_HPV16_non_WLWH_25_50",
    "Chinula_2026_HPV16_WLWH_25_50__sim_mean": "Chinula_2026_HPV16_WLWH_25_50",
    "Chinula_2026_HPV18_45_non_WLWH_25_50__sim_mean": "Chinula_2026_HPV18_45_non_WLWH_25_50",
    "Chinula_2026_HPV18_45_WLWH_25_50__sim_mean": "Chinula_2026_HPV18_45_WLWH_25_50",
    "Chinula_2026_P3_non_WLWH_25_50__sim_mean": "Chinula_2026_P3_non_WLWH_25_50",
    "Chinula_2026_P3_WLWH_25_50__sim_mean": "Chinula_2026_P3_WLWH_25_50",
}

OBSERVED_KEY_BY_RAW = {
    "Chinula_2026_HRHPV_non_WLWH_25_50": "HRHPV_non_WLWH",
    "Chinula_2026_HRHPV_WLWH_25_50": "HRHPV_WLWH",
    "Chinula_2026_HPV16_non_WLWH_25_50": "HPV16_non_WLWH",
    "Chinula_2026_HPV16_WLWH_25_50": "HPV16_WLWH",
    "Chinula_2026_HPV18_45_non_WLWH_25_50": "HPV18_45_non_WLWH",
    "Chinula_2026_HPV18_45_WLWH_25_50": "HPV18_45_WLWH",
    "Chinula_2026_P3_non_WLWH_25_50": "P3_non_WLWH",
    "Chinula_2026_P3_WLWH_25_50": "P3_WLWH",
}


def observed_ci(observed: float, denominator: int) -> tuple[float, float]:
    standard_error = math.sqrt(observed * (1.0 - observed) / denominator)
    return (
        max(0.0, observed - 1.96 * standard_error),
        min(1.0, observed + 1.96 * standard_error),
    )


def resolve_inputs(result_path: Path) -> tuple[Path, Path]:
    if result_path.is_dir():
        result_directory = result_path
        grid_csv = result_directory / "grid_search_results.csv"
    else:
        grid_csv = result_path
        result_directory = grid_csv.parent

    if not grid_csv.exists():
        raise FileNotFoundError(f"Cannot find grid results CSV: {grid_csv}")

    return result_directory, grid_csv


def load_replicates_from_folder(result_path: Path, folder_number: int) -> pd.DataFrame:
    """Load raw beta-by-seed results from one completed HPC result folder."""
    if result_path.is_file():
        csv_files = [result_path]
        source_name = result_path.parent.name
    else:
        source_name = result_path.name
        preferred = result_path / "grid_search_replicates.csv"
        if preferred.exists():
            csv_files = [preferred]
        else:
            csv_files = sorted(result_path.rglob("grid_seed_*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No grid_search_replicates.csv or grid_seed_*.csv found in {result_path}"
        )

    frames = [pd.read_csv(csv_file) for csv_file in csv_files]
    frame = pd.concat(frames, ignore_index=True)

    required = {"b_hpv", *SUMMARY_TO_RAW.values()}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(
            f"Raw replicate columns are missing from {result_path}:\n"
            + "\n".join(missing)
        )

    # Retain the actual seed when present. If it is absent, use the folder
    # number so each folder still appears as a separate trajectory.
    if "seed" not in frame.columns:
        frame["seed"] = folder_number

    frame["source_folder"] = source_name
    frame["seed_label"] = frame["seed"].astype(str)
    return frame


def combine_seed_folders(result_paths: list[Path]) -> pd.DataFrame:
    frames = [
        load_replicates_from_folder(path, folder_number=index)
        for index, path in enumerate(result_paths, start=1)
    ]
    combined = pd.concat(frames, ignore_index=True)
    combined["b_hpv"] = pd.to_numeric(combined["b_hpv"], errors="raise")

    # If the same numerical seed occurs in more than one folder, include the
    # folder name in its label so trajectories cannot be confused.
    seed_sources = combined.groupby("seed_label")["source_folder"].nunique()
    repeated_labels = set(seed_sources[seed_sources > 1].index)
    if repeated_labels:
        print(
            "WARNING: the same numerical seed appears in multiple folders: "
            + ", ".join(sorted(repeated_labels))
            + ". Folder names will be added to the plot labels. Confirm that "
              "the five simulations really used different seed values."
        )
    mask = combined["seed_label"].isin(repeated_labels)
    combined.loc[mask, "seed_label"] = (
        combined.loc[mask, "source_folder"] + ":" + combined.loc[mask, "seed_label"]
    )

    duplicate = combined.duplicated(["b_hpv", "seed_label"], keep=False)
    if duplicate.any():
        rows = combined.loc[duplicate, ["b_hpv", "seed_label", "source_folder"]]
        raise ValueError(
            "Duplicate beta/seed results were found. Do not provide the same "
            f"folder twice.\n{rows.to_string(index=False)}"
        )

    return combined.sort_values(["seed_label", "b_hpv"]).reset_index(drop=True)


def summarise_replicates(replicates: pd.DataFrame) -> pd.DataFrame:
    """Recreate grid_search_results.csv across all supplied seed folders."""
    primary_raw = SUMMARY_TO_RAW["sim_mean"]
    observed, denominator = OBSERVED["HRHPV_non_WLWH"]
    observed_se = math.sqrt(observed * (1.0 - observed) / denominator)
    rows = []

    for beta, beta_frame in replicates.groupby("b_hpv", sort=True):
        primary_values = pd.to_numeric(beta_frame[primary_raw], errors="coerce").dropna()
        if primary_values.empty:
            raise ValueError(f"No primary-target values found for b_hpv={beta}")

        sim_mean = float(primary_values.mean())
        sim_sd = float(primary_values.std(ddof=1)) if len(primary_values) > 1 else np.nan
        sim_mcse = sim_sd / math.sqrt(len(primary_values)) if len(primary_values) > 1 else np.nan
        residual = sim_mean - observed
        standardized_residual = residual / observed_se

        row = {
            "b_hpv": float(beta),
            "loss": float(standardized_residual ** 2),
            "observed": observed,
            "observed_se": observed_se,
            "sim_mean": sim_mean,
            "sim_sd": sim_sd,
            "sim_mcse": sim_mcse,
            "residual": residual,
            "standardized_residual": standardized_residual,
            "n_seeds": int(beta_frame["seed_label"].nunique()),
        }

        for summary_mean, raw_column in SUMMARY_TO_RAW.items():
            if summary_mean == "sim_mean":
                continue
            values = pd.to_numeric(beta_frame[raw_column], errors="coerce").dropna()
            summary_sd = summary_mean.replace("__sim_mean", "__sim_sd")
            summary_observed = summary_mean.replace("__sim_mean", "__observed")
            row[summary_mean] = float(values.mean()) if not values.empty else np.nan
            row[summary_sd] = (
                float(values.std(ddof=1)) if len(values) > 1 else np.nan
            )
            observed_key = OBSERVED_KEY_BY_RAW[raw_column]
            row[summary_observed] = OBSERVED[observed_key][0]

        rows.append(row)

    return pd.DataFrame(rows).sort_values("b_hpv").reset_index(drop=True)


def validate_grid_results(results: pd.DataFrame) -> None:
    required = {"b_hpv", "loss", "sim_mean"}
    for _, _, specifications in GROUPS:
        required.update(item[0] for item in specifications)

    missing = sorted(required.difference(results.columns))
    if missing:
        raise KeyError("Missing required columns:\n" + "\n".join(missing))


def draw_group(
    axis: plt.Axes,
    results: pd.DataFrame,
    title: str,
    specifications: list[tuple[str, str, str, str]],
) -> None:
    for mean_column, sd_column, observed_key, status in specifications:
        observed, denominator = OBSERVED[observed_key]
        colour = "tab:blue" if status == "Non-WLWH" else "tab:orange"
        y_error = results[sd_column] if sd_column in results.columns else None

        axis.errorbar(
            results["b_hpv"],
            results[mean_column],
            yerr=y_error,
            marker="o",
            capsize=3,
            color=colour,
            label=f"Simulated {status}",
        )
        axis.axhline(
            observed,
            linestyle="--",
            color=colour,
            alpha=0.8,
            label=f"Observed {status} ({observed:.3f})",
        )

        lower, upper = observed_ci(observed, denominator)
        axis.axhspan(lower, upper, color=colour, alpha=0.06)

    axis.set_title(title)
    axis.set_xlabel("b_hpv")
    axis.set_ylabel("Prevalence")
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)


def make_grid_figures(results: pd.DataFrame, output_directory: Path) -> float:
    results = results.sort_values("b_hpv").reset_index(drop=True)
    best_row = results.loc[results["loss"].idxmin()]
    best_beta = float(best_row["b_hpv"])

    # Figure 1: primary calibration target.
    observed, denominator = OBSERVED["HRHPV_non_WLWH"]
    lower, upper = observed_ci(observed, denominator)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.errorbar(
        results["b_hpv"],
        results["sim_mean"],
        yerr=results["sim_sd"] if "sim_sd" in results.columns else None,
        marker="o",
        capsize=3,
        label="Simulated mean (error bars: SD)",
    )
    axis.axhline(observed, linestyle="--", color="black", label=f"Observed ({observed:.3f})")
    axis.axhspan(lower, upper, color="grey", alpha=0.15, label="Observed 95% CI")
    axis.axvline(best_beta, linestyle=":", color="tab:red", label=f"Minimum-loss b={best_beta:.2f}")
    axis.set(xlabel="b_hpv", ylabel="HR-HPV prevalence", title="Primary calibration target: Non-WLWH")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_directory / "figure_1_primary_target_by_b.png", dpi=300)
    plt.close(figure)

    # Figure 2: loss function.
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(results["b_hpv"], results["loss"], marker="o")
    axis.axvline(best_beta, linestyle="--", color="tab:red", label=f"Minimum-loss b={best_beta:.2f}")
    axis.set(xlabel="b_hpv", ylabel="Squared standardized residual", title="Calibration loss across b_hpv")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_directory / "figure_2_loss_by_b.png", dpi=300)
    plt.close(figure)

    # Figure 3: four-panel overview.
    figure, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True)
    for axis, (title, _, specifications) in zip(axes.flat, GROUPS):
        draw_group(axis, results, title, specifications)
    figure.suptitle("HPV prevalence across tested b_hpv values", fontsize=15)
    figure.tight_layout()
    figure.savefig(output_directory / "figure_3_all_hpv_groups_by_b.png", dpi=300)
    plt.close(figure)

    # Figures 4-7: one larger figure per HPV group.
    for number, (title, file_label, specifications) in enumerate(GROUPS, start=4):
        figure, axis = plt.subplots(figsize=(8, 5))
        draw_group(axis, results, title, specifications)
        figure.tight_layout()
        figure.savefig(output_directory / f"figure_{number}_{file_label}_by_b.png", dpi=300)
        plt.close(figure)

    return best_beta


def draw_seed_trajectories(
    axis: plt.Axes,
    replicates: pd.DataFrame,
    raw_column: str,
    observed_key: str,
    title: str,
) -> None:
    """Draw one prevalence-versus-b line for every simulation seed."""
    for seed_label, seed_frame in replicates.groupby("seed_label", sort=True):
        seed_frame = seed_frame.sort_values("b_hpv")
        axis.plot(
            seed_frame["b_hpv"],
            seed_frame[raw_column],
            marker="o",
            linewidth=1.2,
            alpha=0.65,
            label=f"Seed {seed_label}",
        )

    mean_by_beta = replicates.groupby("b_hpv")[raw_column].mean().sort_index()
    axis.plot(
        mean_by_beta.index,
        mean_by_beta.values,
        color="black",
        marker="D",
        linewidth=2.8,
        label="Mean across seeds",
    )

    observed, denominator = OBSERVED[observed_key]
    lower, upper = observed_ci(observed, denominator)
    axis.axhline(observed, color="tab:red", linestyle="--", linewidth=2, label="Observed")
    axis.axhspan(lower, upper, color="tab:red", alpha=0.08, label="Observed 95% CI")
    axis.set_title(title)
    axis.set_xlabel("b_hpv")
    axis.set_ylabel("Prevalence")
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.2)


def make_seed_trajectory_figures(
    replicates: pd.DataFrame,
    output_directory: Path,
) -> None:
    target_specs = [
        (raw_column, OBSERVED_KEY_BY_RAW[raw_column], observed_key.replace("_", " "))
        for raw_column, observed_key in OBSERVED_KEY_BY_RAW.items()
    ]

    # Eight-panel overview requested for comparing stochastic seed variation.
    figure, axes = plt.subplots(4, 2, figsize=(14, 19), sharex=True)
    for axis, (raw_column, observed_key, title) in zip(axes.flat, target_specs):
        draw_seed_trajectories(axis, replicates, raw_column, observed_key, title)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, fontsize=9)
    figure.suptitle("HPV prevalence across b_hpv: individual seed trajectories", y=0.995, fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    figure.savefig(output_directory / "figure_seed_trajectories_all_targets.png", dpi=300)
    plt.close(figure)

    # A larger primary-target figure suitable for calibration discussions.
    primary_raw = SUMMARY_TO_RAW["sim_mean"]
    figure, axis = plt.subplots(figsize=(10, 6))
    draw_seed_trajectories(
        axis,
        replicates,
        primary_raw,
        "HRHPV_non_WLWH",
        "Primary calibration target: HR-HPV Non-WLWH",
    )
    axis.legend(ncol=2, fontsize=9)
    figure.tight_layout()
    figure.savefig(output_directory / "figure_seed_trajectories_primary_target.png", dpi=300)
    plt.close(figure)


def make_target_comparison_figure(
    comparison_csv: Path,
    output_directory: Path,
    best_beta: float,
) -> None:
    if not comparison_csv.exists():
        print("target_comparison.csv not found; skipping observed-versus-predicted figure.")
        return

    comparison = pd.read_csv(comparison_csv)
    required = {"observed", "original_mean", "calibrated_mean"}
    if not required.issubset(comparison.columns):
        print("target_comparison.csv is incomplete; skipping comparison figure.")
        return

    lower = float(comparison[list(required)].min().min())
    upper = float(comparison[list(required)].max().max())
    margin = max(0.01, 0.05 * (upper - lower))

    figure, axis = plt.subplots(figsize=(7, 7))
    axis.scatter(comparison["observed"], comparison["original_mean"], marker="x", s=70, label="Original b=0.75")
    axis.scatter(comparison["observed"], comparison["calibrated_mean"], marker="o", s=70, label=f"Calibrated b={best_beta:.2f}")
    axis.plot([lower - margin, upper + margin], [lower - margin, upper + margin], linestyle="--", color="black", label="Perfect agreement")
    axis.set(xlabel="Observed prevalence", ylabel="Model-predicted prevalence", title="Observed versus predicted HPV prevalence")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_directory / "figure_8_observed_vs_predicted.png", dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine separate HPV seed folders and plot calibration outputs."
    )
    parser.add_argument(
        "result_paths",
        nargs="+",
        type=Path,
        help="Two or more result folders containing grid_search_replicates.csv.",
    )
    parser.add_argument("--output-directory", type=Path, default=None)
    arguments = parser.parse_args()

    missing_paths = [path for path in arguments.result_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "These result paths do not exist:\n" + "\n".join(map(str, missing_paths))
        )

    default_parent = arguments.result_paths[0].parent
    output_directory = arguments.output_directory or (
        default_parent / "combined_hpv_seed_figures"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    replicates = combine_seed_folders(arguments.result_paths)
    results = summarise_replicates(replicates)
    replicates.to_csv(output_directory / "combined_grid_search_replicates.csv", index=False)
    results.to_csv(output_directory / "combined_grid_search_results.csv", index=False)

    seed_counts = replicates.groupby("b_hpv")["seed_label"].nunique()
    if seed_counts.nunique() != 1:
        print("WARNING: different b_hpv values have different numbers of seeds:")
        print(seed_counts.to_string())

    validate_grid_results(results)
    best_beta = make_grid_figures(results, output_directory)
    make_seed_trajectory_figures(replicates, output_directory)

    print(f"Combined result folders: {len(arguments.result_paths)}")
    print(f"Unique seed trajectories: {replicates['seed_label'].nunique()}")
    print("Seeds per b_hpv:")
    print(seed_counts.to_string())
    print(f"Minimum-loss tested b_hpv: {best_beta:.6f}")
    print(f"Figures saved to: {output_directory}")


if __name__ == "__main__":
    main()
