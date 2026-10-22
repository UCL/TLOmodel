from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
# MODIFIED: use the actual TLOmodel project path on the Q drive.
sys.path.insert(0, r"Q:\TLOmodel\src")
import math
import datetime
import multiprocessing

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# MODIFIED: scipy.optimize.minimize_scalar is no longer needed because the
# calibration now uses a pre-specified parallel grid of b_hpv values.
from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file

from tlo.methods import (
    demography,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    hpv,
    measles,
    simplified_births,
    symptommanager,
    tb,
)

# MODIFIED: restore the actual project root used locally and on DIDE Windows HPC.
TLO_ROOT = Path(r"Q:\TLOmodel")
RESOURCE_FILEPATH = TLO_ROOT / "resources"
run_time = datetime.datetime.now().strftime("%d_%m_%Y_%H%M%S")
# MODIFIED: the folder name now states that this is a parallel grid calibration.
OUTPUT_ROOT = (TLO_ROOT / "outputs" / f"hpv_b_hpv_parallel_grid_calibration_{run_time}")

START_DATE = Date(2010, 1, 1)
END_DATE = Date(2023, 1, 1)
ORIGINAL_B_HPV = 0.75

TEST_MODE = False
MAX_PARALLEL_RUNS = 5

if TEST_MODE:

    POPULATION_SIZE = 2000
    SEARCH_SEEDS = [1001, 1002,]
    CONFIRMATION_SEEDS = [2001, 2002, 2003,]

else:
    POPULATION_SIZE = 10000
    SEARCH_SEEDS = [1001, 1002, 1003, 1004, 1005,]
    CONFIRMATION_SEEDS = list(range(2001, 2021,))

# MODIFIED: replace the sequential bounded optimiser with a fixed fine grid.
# Every beta uses the same SEARCH_SEEDS, allowing fair paired comparisons.
B_HPV_GRID = [0.65, 0.66, 0.67, 0.68, 0.69, 0.70,]

DEFAULT_SENSITIVITY = 1.0
DEFAULT_SPECIFICITY = 1.0

@dataclass(frozen=True)
class ObservedTarget:

    name: str
    log_column: str
    positive: int
    denominator: int

    window_start: pd.Timestamp
    window_end: pd.Timestamp

    purpose: str
    sensitivity: float = DEFAULT_SENSITIVITY
    specificity: float = DEFAULT_SPECIFICITY

    @property
    def observed(self,) -> float:
        return (self.positive / self.denominator)

    @property
    def observed_se(self,) -> float:
        """ SE = sqrt[p(1-p)/n] """

        p = self.observed
        return math.sqrt(p * (1.0 - p) / self.denominator)

STUDY_START = pd.Timestamp("2020-06-01")

STUDY_END = pd.Timestamp("2022-02-28")

CALIBRATION_TARGET = ObservedTarget(
    name=("Chinula_2026_" "HRHPV_non_WLWH_25_50"),
    log_column=("Calib_Any_F_25_50_" "HIVneg_Prev"),
    positive=181,
    denominator=625,
    window_start=STUDY_START,
    window_end=STUDY_END,
    purpose="calibration",
)

VALIDATION_TARGETS = [
    ObservedTarget(
        name=("Chinula_2026_" "HRHPV_WLWH_25_50"),
        log_column=("Calib_Any_F_25_50_" "HIVpos_Prev"),
        positive=295,
        denominator=625,
        window_start=STUDY_START,
        window_end=STUDY_END,
        purpose="validation",
    ),

    # -------------------------------------------------------------------------
    # HPV16 = Xpert P1
    # -------------------------------------------------------------------------
    ObservedTarget(
        name=("Chinula_2026_" "HPV16_non_WLWH_25_50"),
        log_column=("Calib_hr1_F_25_50_" "HIVneg_Prev"),
        positive=35,
        denominator=625,
        window_start=STUDY_START,
        window_end=STUDY_END,
        purpose="validation",
    ),

    ObservedTarget(
        name=("Chinula_2026_" "HPV16_WLWH_25_50"),
        log_column=("Calib_hr1_F_25_50_" "HIVpos_Prev"),
        positive=63,
        denominator=625,
        window_start=STUDY_START,
        window_end=STUDY_END,
        purpose="validation",
    ),

    # -------------------------------------------------------------------------
    # HPV18/45 = Xpert P2
    # -------------------------------------------------------------------------
    ObservedTarget(
        name=("Chinula_2026_" "HPV18_45_non_WLWH_25_50"),
        log_column=("Calib_Xpert_P2_F_25_50_" "HIVneg_Prev"),
        positive=47,
        denominator=625,
        window_start=STUDY_START,
        window_end=STUDY_END,
        purpose="validation",
    ),

    ObservedTarget(
        name=("Chinula_2026_" "HPV18_45_WLWH_25_50"),
        log_column=("Calib_Xpert_P2_F_25_50_" "HIVpos_Prev"),
        positive=51,
        denominator=625,
        window_start=STUDY_START,
        window_end=STUDY_END,
        purpose="validation",
    ),

    # -------------------------------------------------------------------------
    # HPV31/33/35/52/58 = Xpert P3
    # -------------------------------------------------------------------------
    ObservedTarget(
        name=("Chinula_2026_" "P3_non_WLWH_25_50"),
        log_column=("Calib_Xpert_P3_F_25_50_" "HIVneg_Prev"),
        positive=93,
        denominator=625,
        window_start=STUDY_START,
        window_end=STUDY_END,
        purpose="validation",
    ),

    ObservedTarget(
        name=("Chinula_2026_" "P3_WLWH_25_50"),
        log_column=("Calib_Xpert_P3_F_25_50_" "HIVpos_Prev"),
        positive=168,
        denominator=625,
        window_start=STUDY_START,
        window_end=STUDY_END,
        purpose="validation",
    ),
]

ALL_TARGETS = [CALIBRATION_TARGET, *VALIDATION_TARGETS,]

# =============================================================================
# 3. MEASUREMENT MODEL
# =============================================================================

def expected_test_positive_prevalence(
    true_prevalence: float,
    sensitivity: float,
    specificity: float,
) -> float:

    expected = (sensitivity * true_prevalence + (1.0 - specificity) * (1.0 - true_prevalence))

    return float(np.clip(expected,0.0,1.0,))

# =============================================================================
# 4. HANDLE HPV LOG DATES
# =============================================================================
def add_log_date(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    result = summary.copy()

    if "date" not in result.columns:
        raise KeyError("HPV summary does not contain a 'date' column.")

    result["_calibration_date"] = pd.to_datetime(result["date"], errors="coerce",)

    result = result.loc[result["_calibration_date"].notna()].copy()

    if result.empty:
        raise ValueError("No valid dates were found in the HPV summary log.")
    return result

# =============================================================================
# 5. EXTRACT MODEL VALUE FOR ONE REAL-WORLD STUDY
# =============================================================================
def extract_target_prediction(summary: pd.DataFrame, target: ObservedTarget,) -> float:
    summary = add_log_date(summary)
    if target.log_column not in summary.columns:

        raise KeyError(
            f"Missing HPV logging column:\n"
            f"{target.log_column}\n\n"
            "Before running calibration, add:\n\n"
            "CALIBRATION_AGE_RANGES = {\n"
            "    '25_50': (25, 51),\n"
            "    '25_59': (25, 60),\n"
            "}\n\n"
            "to hpv.py."
        )

    in_window = summary.loc[
        (summary["_calibration_date"] >= target.window_start)
        &
        (summary["_calibration_date"] <= target.window_end)
    ].copy()

    if in_window.empty:
        midpoint = (target.window_start + (target.window_end - target.window_start) / 2)
        nearest_idx = (summary["_calibration_date"] - midpoint).abs().idxmin()
        in_window = summary.loc[[nearest_idx]].copy()

        print(
            f"WARNING: no model log date "
            f"falls inside {target.name}; "
            f"using nearest date "
            f"{in_window['_calibration_date'].iloc[0].date()}."
        )

    latent_prevalence = float(
        pd.to_numeric(in_window[target.log_column], errors="coerce",).mean()
    )

    if pd.isna(latent_prevalence):
        raise ValueError(
            f"Model prediction is missing "
            f"for {target.name}."
        )

    return expected_test_positive_prevalence(
        true_prevalence=(latent_prevalence),
        sensitivity=(target.sensitivity),
        specificity=(target.specificity),
    )

# =============================================================================
# 6. BUILD MODEL
# =============================================================================
def make_base_simulation(
    output_dir: Path,
    seed: int,
) -> Simulation:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_config = {
        "filename": ("hpv_calibration"),
        "directory": (output_dir),
        "custom_levels": {
            "*": logging.WARNING,
            "tlo.methods.hpv": (logging.INFO),
        },
    }

    # -------------------------------------------------------------------------
    # Create simulation
    # -------------------------------------------------------------------------
    sim = Simulation(
        start_date=(START_DATE),
        seed=int(seed),
        log_config=(log_config),
        show_progress_bar=False,
        resourcefilepath=str(RESOURCE_FILEPATH),
    )

    # -------------------------------------------------------------------------
    # Register modules
    # -------------------------------------------------------------------------
    sim.register(
        demography.Demography(),
        simplified_births.SimplifiedBirths(),
        enhanced_lifestyle.Lifestyle(),
        healthsystem.HealthSystem(service_availability=["*"],
                                  mode_appt_constraints=1,
                                  cons_availability="default",
                                  ignore_priority=False,
                                  capabilities_coefficient=1.0,
                                  use_funded_or_actual_staffing="actual",
                                  disable=False,
                                  disable_and_reject_all=False,
                                  ),
        symptommanager.SymptomManager(),
        healthseekingbehaviour.HealthSeekingBehaviour(),
        healthburden.HealthBurden(),
        epi.Epi(),
        hiv.Hiv(),
        measles.Measles(),
        tb.Tb(),
        hpv.HPV(),
    )

    return sim

# =============================================================================
# 7. FIND HPV MODULE
# =============================================================================
def get_hpv_module(sim: Simulation,) -> hpv.HPV:

    for module in sim.modules.values():
        if isinstance(module, hpv.HPV,):
            return module

    raise KeyError(
        "No hpv.HPV module found. " f"Registered modules: "
        f"{list(sim.modules.keys())}"
    )


# =============================================================================
# 8. RUN ONE SIMULATION
# =============================================================================

def run_one_simulation(b_hpv: float, seed: int, run_directory: Path,) -> dict[str, Any]:
    if b_hpv <= 0.0:
        raise ValueError(
            f"b_hpv must be positive; "
            f"received {b_hpv}."
        )

    sim = make_base_simulation(output_dir=(run_directory), seed=seed,)

    hpv_module = get_hpv_module(sim)

    hpv_module.parameters["b_hpv"] = float(b_hpv)

    sim.make_initial_population(n=POPULATION_SIZE)

    sim.simulate(end_date=END_DATE)

    parsed_logs = parse_log_file(sim.log_filepath)

    if ("tlo.methods.hpv" not in parsed_logs):
        raise KeyError(
            "Could not find 'tlo.methods.hpv' in parsed logs.\n"
            f"Available loggers: "
            f"{list(parsed_logs.keys())}"
        )

    if ("summary" not in parsed_logs["tlo.methods.hpv"]):

        raise KeyError("Could not find HPV 'summary' log.")

    hpv_summary = (
        parsed_logs["tlo.methods.hpv"]["summary"]
        .copy()
    )

    result: dict[str, Any,] = {
        "b_hpv": float(b_hpv),
        "seed": int(seed),
    }

    for target in ALL_TARGETS:
        result[target.name] = extract_target_prediction(
            summary=(hpv_summary),
            target=(target),
        )

    return result

def safe_beta_label(value: float,) -> str:

    return (f"{value:.8f}".rstrip("0").rstrip(".").replace(".","p",))

def evaluate_b_hpv(
    b_hpv: float,
    seeds: list[int],
    output_root: Path,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
]:
    replicate_records = []
    beta_label = safe_beta_label(b_hpv)

    jobs = []

    for seed in seeds:
        run_directory = (
            output_root
            / f"beta_{beta_label}"
            / f"seed_{seed}"
        )

        jobs.append(
            (
                float(b_hpv),
                int(seed),
                run_directory,
            )
        )

    max_workers = min(
        MAX_PARALLEL_RUNS,
        len(jobs),
    )

    print(
        f"Running b_hpv={b_hpv:.6f} "
        f"with {len(jobs)} seeds "
        f"using {max_workers} parallel workers."
    )

    with ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:

        future_to_seed = {}

        for beta_value, seed, run_directory in jobs:
            future = executor.submit(
                run_one_simulation,
                beta_value,
                seed,
                run_directory,
            )

            future_to_seed[future] = seed

        for future in as_completed(future_to_seed):

            seed = future_to_seed[future]

            try:
                result = future.result()

            except Exception as error:
                raise RuntimeError(
                    f"Simulation failed for "
                    f"b_hpv={b_hpv:.6f}, "
                    f"seed={seed}."
                ) from error

            replicate_records.append(result)

            print(
                f"Completed "
                f"b_hpv={b_hpv:.6f}, "
                f"seed={seed}"
            )

    replicate_records.sort(
        key=lambda record: record["seed"]
    )

    replicate_df = pd.DataFrame(replicate_records)

    if replicate_df.empty:
        raise ValueError(
            f"No simulation results were produced "
            f"for b_hpv={b_hpv:.6f}."
        )

    # =============================================================================
    # PRIMARY CALIBRATION TARGET
    # =============================================================================

    primary_values = pd.to_numeric(
        replicate_df[CALIBRATION_TARGET.name],
        errors="coerce",
    ).dropna()

    if primary_values.empty:

        raise ValueError(
            f"No valid simulation values "
            f"for "
            f"{CALIBRATION_TARGET.name}."
        )

    simulated_mean = float(primary_values.mean())

    simulated_sd = (float(primary_values.std(ddof=1))
        if len(primary_values) > 1
        else np.nan
    )

    # -------------------------------------------------------------------------
    # Monte-Carlo Standard Error
    # -------------------------------------------------------------------------
    simulation_mcse = (
        simulated_sd / math.sqrt(len(primary_values))

        if len(primary_values) > 1
        else np.nan
    )

    residual = (simulated_mean - CALIBRATION_TARGET.observed)

    standardized_residual = (residual / CALIBRATION_TARGET.observed_se)

    # -------------------------------------------------------------------------
    # Least-squares loss
    # -------------------------------------------------------------------------
    loss = (standardized_residual ** 2)

    # -------------------------------------------------------------------------
    # Save candidate summary
    # -------------------------------------------------------------------------

    summary: dict[str, Any,] = {
        "b_hpv": float(b_hpv),
        "loss": float(loss),
        "observed": (CALIBRATION_TARGET.observed),
        "observed_se": (CALIBRATION_TARGET.observed_se),
        "sim_mean": (simulated_mean),
        "sim_sd": (simulated_sd),
        "sim_mcse": (simulation_mcse),
        "residual": (residual),
        "standardized_residual": (standardized_residual),
        "n_seeds": int(len(primary_values)),
    }

    for target in VALIDATION_TARGETS:

        values = pd.to_numeric(replicate_df[target.name],
            errors="coerce",
        ).dropna()

        if values.empty:

            summary[
                f"{target.name}"
                f"__sim_mean"
            ] = np.nan

            summary[
                f"{target.name}"
                f"__sim_sd"
            ] = np.nan

        else:
            summary[
                f"{target.name}"
                f"__sim_mean"
            ] = float(values.mean())

            summary[
                f"{target.name}"
                f"__sim_sd"
            ] = (float(values.std(ddof=1))
                if len(values) > 1
                else np.nan
            )

        summary[
            f"{target.name}"
            f"__observed"
        ] = (target.observed)

    return (summary, replicate_df,)

# =============================================================================
# 11. PARALLEL GRID SEARCH
# =============================================================================
# MODIFIED: summarise already-completed replicates without launching new runs.
def summarise_replicates(
    b_hpv: float,
    replicate_df: pd.DataFrame,
) -> dict[str, Any]:

    primary_values = pd.to_numeric(
        replicate_df[CALIBRATION_TARGET.name],
        errors="coerce",
    ).dropna()

    if primary_values.empty:
        raise ValueError(
            f"No valid simulation values for {CALIBRATION_TARGET.name} "
            f"at b_hpv={b_hpv:.6f}."
        )

    simulated_mean = float(primary_values.mean())
    simulated_sd = (
        float(primary_values.std(ddof=1))
        if len(primary_values) > 1
        else np.nan
    )
    simulation_mcse = (
        simulated_sd / math.sqrt(len(primary_values))
        if len(primary_values) > 1
        else np.nan
    )
    residual = simulated_mean - CALIBRATION_TARGET.observed
    standardized_residual = residual / CALIBRATION_TARGET.observed_se
    loss = standardized_residual ** 2

    summary: dict[str, Any] = {
        "b_hpv": float(b_hpv),
        "loss": float(loss),
        "observed": CALIBRATION_TARGET.observed,
        "observed_se": CALIBRATION_TARGET.observed_se,
        "sim_mean": simulated_mean,
        "sim_sd": simulated_sd,
        "sim_mcse": simulation_mcse,
        "residual": residual,
        "standardized_residual": standardized_residual,
        "n_seeds": int(len(primary_values)),
    }

    for target in VALIDATION_TARGETS:
        values = pd.to_numeric(
            replicate_df[target.name],
            errors="coerce",
        ).dropna()

        summary[f"{target.name}__sim_mean"] = (
            float(values.mean()) if not values.empty else np.nan
        )
        summary[f"{target.name}__sim_sd"] = (
            float(values.std(ddof=1))
            if len(values) > 1
            else np.nan
        )
        summary[f"{target.name}__observed"] = target.observed

    return summary


# MODIFIED: create one process pool for every beta-by-seed combination.
# This avoids nested process pools and allows different beta values to run
# concurrently, rather than waiting for one beta to finish before starting next.
def run_parallel_grid_search() -> tuple[pd.DataFrame, pd.DataFrame]:
    jobs = []

    for beta in B_HPV_GRID:
        beta_label = safe_beta_label(beta)

        for seed in SEARCH_SEEDS:
            run_directory = (
                OUTPUT_ROOT
                / "grid_search"
                / f"beta_{beta_label}"
                / f"seed_{seed}"
            )
            jobs.append((float(beta), int(seed), run_directory))

    max_workers = min(MAX_PARALLEL_RUNS, len(jobs))

    print(
        f"Running {len(B_HPV_GRID)} beta values x "
        f"{len(SEARCH_SEEDS)} seeds = {len(jobs)} simulations "
        f"using {max_workers} parallel workers."
    )

    replicate_records = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {
            executor.submit(
                run_one_simulation,
                beta,
                seed,
                run_directory,
            ): (beta, seed)
            for beta, seed, run_directory in jobs
        }

        for future in as_completed(future_to_job):
            beta, seed = future_to_job[future]

            try:
                result = future.result()
            except Exception as error:
                raise RuntimeError(
                    f"Grid simulation failed for b_hpv={beta:.6f}, "
                    f"seed={seed}."
                ) from error

            replicate_records.append(result)
            print(f"Completed grid b_hpv={beta:.6f}, seed={seed}")

    all_replicates = (
        pd.DataFrame(replicate_records)
        .sort_values(["b_hpv", "seed"])
        .reset_index(drop=True)
    )

    if all_replicates.empty:
        raise ValueError("The parallel grid search produced no results.")

    all_replicates.to_csv(
        OUTPUT_ROOT / "grid_search_replicates.csv",
        index=False,
    )

    summaries = []
    for beta in B_HPV_GRID:
        beta_replicates = all_replicates.loc[
            np.isclose(all_replicates["b_hpv"], float(beta))
        ].copy()

        summaries.append(
            summarise_replicates(float(beta), beta_replicates)
        )

    grid_results = (
        pd.DataFrame(summaries)
        .sort_values("b_hpv")
        .reset_index(drop=True)
    )

    grid_results.to_csv(
        OUTPUT_ROOT / "grid_search_results.csv",
        index=False,
    )

    return grid_results, all_replicates

# =============================================================================
# 13. BUILD FINAL CALIBRATION / VALIDATION TABLE
# =============================================================================
def make_target_comparison(
    original_replicates: (pd.DataFrame),
    calibrated_replicates: (pd.DataFrame),
) -> pd.DataFrame:

    rows = []

    for target in ALL_TARGETS:
        original_values = pd.to_numeric(
            original_replicates[target.name],
            errors="coerce",
        ).dropna()

        # ---------------------------------------------------------------------
        # Calibrated model outputs
        # ---------------------------------------------------------------------
        calibrated_values = pd.to_numeric(
            calibrated_replicates[target.name],
            errors="coerce",
        ).dropna()

        original_mean = float(original_values.mean())
        calibrated_mean = float(calibrated_values.mean())

        # ---------------------------------------------------------------------
        # Observed confidence interval
        # ---------------------------------------------------------------------
        observed_lower = max(0.0, target.observed - 1.96 * target.observed_se,)
        observed_upper = min(1.0, target.observed + 1.96 * target.observed_se,)

        rows.append(
            {"target": (target.name),
                "purpose": (target.purpose),
                "observed": (target.observed),
                "observed_se": (target.observed_se),
                "observed_ci_lower": (observed_lower),
                "observed_ci_upper": (observed_upper),
                "original_mean": (original_mean),
                "calibrated_mean": (calibrated_mean),
                "original_residual": (original_mean - target.observed),
                "calibrated_residual": (calibrated_mean - target.observed),
            }
        )

    return pd.DataFrame(rows)

# =============================================================================
# 14. PLOTS
# =============================================================================
def make_plots(
    grid_results: pd.DataFrame,  # MODIFIED: results from the fixed parallel grid
    target_comparison: pd.DataFrame,
    best_beta: float,
) -> None:

    # =========================================================================
    # FIGURE 1 b_hpv versus model HR-HPV prevalence
    # =========================================================================

    plt.figure(figsize=(8, 5))
    plt.plot(grid_results["b_hpv"],  # MODIFIED
        grid_results["sim_mean"],
        marker="o",
        label="Model mean",
    )

    # Observed prevalence line
    plt.axhline(
        CALIBRATION_TARGET.observed,
        linestyle="--",
        label=(
            "Observed Non-WLWH "
            "HR-HPV = "
            f"{CALIBRATION_TARGET.observed:.3f}"
        ),
    )

    # -------------------------------------------------------------------------
    # Observed 95% CI
    # -------------------------------------------------------------------------
    observed_lower = ( CALIBRATION_TARGET.observed - 1.96 * CALIBRATION_TARGET.observed_se)
    observed_upper = (CALIBRATION_TARGET.observed + 1.96 * CALIBRATION_TARGET.observed_se)

    plt.axhspan(
        max(0.0, observed_lower,),
        min(1.0, observed_upper,),
        alpha=0.15,
        label="Observed 95% CI",
    )

    plt.xlabel("b_hpv")
    plt.ylabel("HR-HPV prevalence")
    plt.legend()
    plt.tight_layout()
    plt.title("Parallel grid search: b_hpv versus HR-HPV prevalence")  # MODIFIED
    plt.savefig(OUTPUT_ROOT / "figure_1_grid_prevalence.png", dpi=300,)  # MODIFIED
    plt.close()

    # =========================================================================
    # FIGURE 2 b_hpv versus least-squares loss
    # =========================================================================
    grid_results = (grid_results.sort_values("b_hpv").copy())  # MODIFIED
    plt.figure(figsize=(8, 5))
    plt.plot(grid_results["b_hpv"],  # MODIFIED
        grid_results["loss"],
        marker="o",
    )
    plt.axvline(best_beta, linestyle="--", label=(f"Calibrated "f"b_hpv={best_beta:.4f}"),)
    plt.xlabel("b_hpv")
    plt.ylabel("Squared standardized residual")
    plt.title("Parallel grid calibration loss")  # MODIFIED
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_ROOT / "figure_2_grid_loss.png", dpi=300,)  # MODIFIED
    plt.close()

    # =========================================================================
    # FIGURE 3 Observed versus predicted: original model calibrated model
    # =========================================================================
    lower = float(
        min(
            target_comparison["observed"].min(),
            target_comparison["original_mean"].min(),
            target_comparison["calibrated_mean"].min(),
        )
    )

    upper = float(
        max(
            target_comparison["observed"].max(),
            target_comparison["original_mean"].max(),
            target_comparison["calibrated_mean"].max(),
        )
    )

    plt.figure(figsize=(7, 7))

    # Original model
    plt.scatter(
        target_comparison["observed"],
        target_comparison["original_mean"],
        marker="x",
        s=70,
        label=(
            f"Original "
            f"b_hpv={ORIGINAL_B_HPV}"
        ),
    )

    # Calibrated model
    plt.scatter(
        target_comparison["observed"],
        target_comparison["calibrated_mean"],
        marker="o",
        s=70,
        label=(
            f"Calibrated "
            f"b_hpv={best_beta:.4f}"
        ),
    )

    # Perfect model = observed line
    plt.plot([lower, upper,], [lower, upper,],
        linestyle="--",
        label=("Perfect agreement"),
    )

    plt.xlabel("Observed prevalence")
    plt.ylabel("Model-predicted prevalence")
    plt.title("Calibration and validation: " "observed versus predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_ROOT/"figure_3_observed_vs_predicted.png", dpi=300,)
    plt.close()


# MODIFIED: plot the simulated prevalence of every HPV group across b_hpv.
# These plots use grid_search_results.csv and do not require new simulations.
def make_hpv_group_plots(
    grid_results: pd.DataFrame,
    output_directory: Path = OUTPUT_ROOT,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    results = grid_results.sort_values("b_hpv").copy()

    plot_groups = [
        ("HR-HPV", [CALIBRATION_TARGET, VALIDATION_TARGETS[0]], "hrhpv"),
        ("HPV16", [VALIDATION_TARGETS[1], VALIDATION_TARGETS[2]], "hpv16"),
        ("HPV18/45", [VALIDATION_TARGETS[3], VALIDATION_TARGETS[4]], "hpv18_45"),
        ("HPV31/33/35/52/58 (P3)", [VALIDATION_TARGETS[5], VALIDATION_TARGETS[6]], "p3"),
    ]

    for title, targets, file_label in plot_groups:
        plt.figure(figsize=(8, 5))

        for target in targets:
            is_primary = target.name == CALIBRATION_TARGET.name
            mean_column = "sim_mean" if is_primary else f"{target.name}__sim_mean"
            sd_column = "sim_sd" if is_primary else f"{target.name}__sim_sd"
            status = "Non-WLWH" if "non_WLWH" in target.name else "WLWH"

            if mean_column not in results.columns:
                raise KeyError(
                    f"Required plotting column is missing: {mean_column}. "
                    "Use grid_search_results.csv from this calibration script."
                )

            y_error = results[sd_column] if sd_column in results.columns else None
            plt.errorbar(
                results["b_hpv"],
                results[mean_column],
                yerr=y_error,
                marker="o",
                capsize=3,
                label=f"Simulated {status}",
            )
            plt.axhline(
                target.observed,
                linestyle="--",
                alpha=0.8,
                label=f"Observed {status} ({target.observed:.3f})",
            )

        plt.xlabel("b_hpv")
        plt.ylabel("Prevalence")
        plt.ylim(bottom=0.0)
        plt.title(f"{title} prevalence across b_hpv values")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            output_directory / f"figure_group_{file_label}_by_b.png",
            dpi=300,
        )
        plt.close()

# =============================================================================
# 15. MAIN CALIBRATION WORKFLOW
# =============================================================================
def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True,)
    if not RESOURCE_FILEPATH.exists():
        raise FileNotFoundError(
            f"Resource folder not found:\n"
            f"{RESOURCE_FILEPATH}\n\n"
            "Run this script from the TLOmodel project root or change RESOURCE_FILEPATH."
        )

    print("\n======================================")
    print("HPV SINGLE-PARAMETER CALIBRATION")
    print("======================================")
    print(f"TEST_MODE = " f"{TEST_MODE}")
    print(f"Population size = " f"{POPULATION_SIZE}")
    print(
        "Calibration target = "
        f"{CALIBRATION_TARGET.observed:.4f} "
        f"("
        f"{CALIBRATION_TARGET.positive}"
        f"/"
        f"{CALIBRATION_TARGET.denominator}"
        f")"
    )

    print(
        f"Observed SE = "
        f"{CALIBRATION_TARGET.observed_se:.4f}"
    )

    print(
        f"Output folder = "
        f"{OUTPUT_ROOT}"
    )

    # =========================================================================
    # STEP 1 PARALLEL FIXED-GRID SEARCH  # MODIFIED
    # =========================================================================
    print(
        "\n=== STEP 1: "
        "PARALLEL FIXED-GRID SEARCH ==="  # MODIFIED
    )

    print(f"b_hpv grid = {B_HPV_GRID}")  # MODIFIED
    print(f"Maximum simultaneous simulations = {MAX_PARALLEL_RUNS}")  # MODIFIED

    grid_results, grid_replicates = run_parallel_grid_search()  # MODIFIED
    min_model_prev = float(grid_results["sim_mean"].min())
    max_model_prev = float(grid_results["sim_mean"].max())

    if not (
        min_model_prev <= CALIBRATION_TARGET.observed <= max_model_prev):

        print(
            "\nWARNING:\n"
            "Observed prevalence is outside "
            "the range generated by B_HPV_GRID.\n"  # MODIFIED
            "Expand B_HPV_GRID before accepting the result."
        )

    # MODIFIED: select the tested b_hpv with the smallest loss. There is no
    # sequential optimiser, so all grid values can be submitted at once.
    best_row = grid_results.loc[grid_results["loss"].idxmin()]
    best_beta = float(best_row["b_hpv"])
    best_loss = float(best_row["loss"])

    print(
        f"\nSearch-stage best "
        f"b_hpv = "
        f"{best_beta:.6f}"
    )

    print(
        f"Search-stage minimum "
        f"loss = "
        f"{best_loss:.6f}"  # MODIFIED
    )

    if np.isclose(best_beta, min(B_HPV_GRID)) or np.isclose(  # MODIFIED
        best_beta, max(B_HPV_GRID)
    ):

        print(
            "\nWARNING:\n"
            "Best b_hpv is at a grid boundary.\n"  # MODIFIED
            "Expand B_HPV_GRID before "
            "accepting the final result."
        )

    # =========================================================================
    # STEP 3 INDEPENDENT-SEED CONFIRMATION
    # =========================================================================
    print(
        "\n=== STEP 3: "
        "CALIBRATED PARAMETER "
        "CONFIRMATION ==="
    )

    (calibrated_summary, calibrated_replicates,) = evaluate_b_hpv(
        b_hpv=(best_beta),
        seeds=(CONFIRMATION_SEEDS),
        output_root=(OUTPUT_ROOT/"confirmation_calibrated"),
    )

    # =========================================================================
    # STEP 4 ORIGINAL b_hpv = 0.75 Use exactly SAME confirmation seeds
    # =========================================================================
    print(
        "\n=== STEP 4: "
        "ORIGINAL PARAMETER "
        "COMPARISON ==="
    )

    (original_summary, original_replicates,) = evaluate_b_hpv(

        b_hpv=(ORIGINAL_B_HPV),
        seeds=(CONFIRMATION_SEEDS),
        output_root=(OUTPUT_ROOT/"confirmation_original"),
    )

    # =========================================================================
    # STEP 5 CALIBRATION + VALIDATION TABLE
    # =========================================================================
    target_comparison = (
        make_target_comparison(
            original_replicates=(original_replicates),
            calibrated_replicates=(calibrated_replicates),
        )
    )

    target_comparison.to_csv(OUTPUT_ROOT/"target_comparison.csv", index=False,)

    # =========================================================================
    # STEP 6 FINAL SUMMARY
    # =========================================================================
    final_summary = pd.DataFrame(
        [
            {
                "version": ("original"),
                "b_hpv": (ORIGINAL_B_HPV),
                "calibration_loss": (original_summary["loss"]),
                "calibration_target_mean": (original_summary["sim_mean"]),
                "calibration_target_observed": (CALIBRATION_TARGET.observed),
                "calibration_target_sim_sd": (original_summary["sim_sd"]),
                "calibration_target_mcse": (original_summary["sim_mcse"]),
            },

            {"version": ("calibrated"),
                "b_hpv": (best_beta),
                "calibration_loss": (calibrated_summary["loss"]),
                "calibration_target_mean": (calibrated_summary["sim_mean"]),
                "calibration_target_observed": (CALIBRATION_TARGET.observed),
                "calibration_target_sim_sd": (calibrated_summary["sim_sd"]),
                "calibration_target_mcse": (calibrated_summary["sim_mcse"]),
            },
        ]
    )

    final_summary.to_csv(OUTPUT_ROOT/"final_calibration_summary.csv", index=False,)

    # =========================================================================
    # STEP 7 CREATE FIGURES
    # =========================================================================
    make_plots(
        grid_results=(grid_results),  # MODIFIED
        target_comparison=(target_comparison),
        best_beta=(best_beta),
    )

    # MODIFIED: create four additional figures comparing Non-WLWH and WLWH
    # prevalence for HR-HPV, HPV16, HPV18/45 and P3 at every tested b_hpv.
    make_hpv_group_plots(
        grid_results=grid_results,
        output_directory=OUTPUT_ROOT,
    )

    # =========================================================================
    # FINAL CONSOLE OUTPUT
    # =========================================================================
    print("\n======================================")
    print("FINAL CALIBRATION RESULT")
    print("======================================")
    print(final_summary.to_string(index=False))
    print("\nCalibration + ""validation targets:")
    print(target_comparison.to_string(index=False))
    print(f"\nAll results saved to:\n" f"{OUTPUT_ROOT}")

# =============================================================================
# 16. RUN SCRIPT
# =============================================================================
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
