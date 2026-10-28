from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_THREAD_LIMIT"] = "1"

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
    simplified_births,
    symptommanager,
    tb,
)

# =============================================================================
# 1. PROJECT PATHS
# =============================================================================
TLO_ROOT = Path(r"Q:\TLOmodel")
RESOURCE_FILEPATH = TLO_ROOT / "resources"

START_DATE = Date(2010, 1, 1)
END_DATE = Date(2024, 1, 1)

TEST_MODE = False
def get_available_cores() -> int:
    hipercow_cores = os.environ.get("HIPERCOW_CORES")

    if hipercow_cores is not None:
        try:
            cores = int(hipercow_cores)
            if cores >= 1:
                return cores
        except ValueError:
            pass

    # Local-machine fallback.
    return max(1, multiprocessing.cpu_count())

if TEST_MODE:

    POPULATION_SIZE = 10000
    SEARCH_SEEDS = [1001, 1002]

else:
    POPULATION_SIZE = 100000
    SEARCH_SEEDS = [1001, 1002, 1003, 1004, 1005]

# B_HPV_GRID = [0.75]

B_HPV_GRID = np.round(
    np.arange(0.68, 0.74, 0.01),2,).tolist()

AVAILABLE_CORES = get_available_cores()

MAX_PARALLEL_RUNS = min(
    AVAILABLE_CORES,
    len(B_HPV_GRID) * len(SEARCH_SEEDS),
)

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

run_time = datetime.datetime.now().strftime("%d_%m_%Y_%H%M%S")

OUTPUT_ROOT = (TLO_ROOT/ "outputs"/ (f"hpv_CHAI_b_hpv_calibration_{len(SEARCH_SEEDS)}seeds_{run_time}"))

CHAI_STUDY_START = pd.Timestamp("2022-03-17")
CHAI_STUDY_END = pd.Timestamp("2023-01-13")

EXTERNAL_STUDY_START = pd.Timestamp("2020-06-01")
EXTERNAL_STUDY_END = pd.Timestamp("2022-02-28")

# =============================================================================
# 9. PRIMARY CALIBRATION TARGET
# =============================================================================
CALIBRATION_TARGET = ObservedTarget(

    name=("CHAI_HRHPV_" "HIVneg_25_49"),
    log_column=(
        "Calib_Any_F_25_49_"
        "HIVneg_Prev"
    ),

    positive=564,
    denominator=2083,

    window_start=CHAI_STUDY_START,
    window_end=CHAI_STUDY_END,

    purpose="calibration",
)

# =============================================================================
# 10. CHAI OVERALL VALIDATION TARGETS
# =============================================================================
VALIDATION_TARGETS = [

    # =========================================================================
    # A. ANY HIGH-RISK HPV — HIV POSITIVE
    # =========================================================================

    ObservedTarget(

        name=(
            "CHAI_HRHPV_"
            "HIVpos_25_49"
        ),

        log_column=(
            "Calib_Any_F_25_49_"
            "HIVpos_Prev"
        ),

        positive=872,
        denominator=2424,

        window_start=CHAI_STUDY_START,
        window_end=CHAI_STUDY_END,

        purpose="HIV_validation",
    ),


    # =========================================================================
    # B. HPV16 — HIV NEGATIVE
    # =========================================================================

    ObservedTarget(

        name=(
            "CHAI_HPV16_"
            "HIVneg_25_49"
        ),

        log_column=(
            "Calib_hr1_F_25_49_"
            "HIVneg_Prev"
        ),

        positive=107,
        denominator=2083,

        window_start=CHAI_STUDY_START,
        window_end=CHAI_STUDY_END,

        purpose="genotype_validation",
    ),


    # =========================================================================
    # C. HPV16 — HIV POSITIVE
    # =========================================================================

    ObservedTarget(

        name=(
            "CHAI_HPV16_"
            "HIVpos_25_49"
        ),

        log_column=(
            "Calib_hr1_F_25_49_"
            "HIVpos_Prev"
        ),

        positive=191,
        denominator=2424,

        window_start=CHAI_STUDY_START,
        window_end=CHAI_STUDY_END,

        purpose="genotype_validation",
    ),


    # =========================================================================
    # D. HPV18/45 — HIV NEGATIVE
    # =========================================================================

    ObservedTarget(

        name=(
            "CHAI_HPV18_45_"
            "HIVneg_25_49"
        ),

        log_column=(
            "Calib_Xpert_P2_F_25_49_"
            "HIVneg_Prev"
        ),

        positive=84,
        denominator=2083,

        window_start=CHAI_STUDY_START,
        window_end=CHAI_STUDY_END,

        purpose="genotype_validation",
    ),


    # =========================================================================
    # E. HPV18/45 — HIV POSITIVE
    # =========================================================================

    ObservedTarget(

        name=(
            "CHAI_HPV18_45_"
            "HIVpos_25_49"
        ),

        log_column=(
            "Calib_Xpert_P2_F_25_49_"
            "HIVpos_Prev"
        ),

        positive=130,
        denominator=2424,

        window_start=CHAI_STUDY_START,
        window_end=CHAI_STUDY_END,

        purpose="genotype_validation",
    ),


    # =========================================================================
    # 11. CHAI AGE-SPECIFIC HIV-NEGATIVE VALIDATION
    # =========================================================================

    ObservedTarget(

        name="CHAI_HRHPV_HIVneg_25_29",

        log_column=(
            "Calib_Any_F_25_29_"
            "HIVneg_Prev"
        ),

        positive=194,
        denominator=600,

        window_start=CHAI_STUDY_START,
        window_end=CHAI_STUDY_END,

        purpose="age_validation",
    ),


    ObservedTarget(

        name="CHAI_HRHPV_HIVneg_30_34",

        log_column=(
            "Calib_Any_F_30_34_"
            "HIVneg_Prev"
        ),

        positive=123,
        denominator=476,

        window_start=CHAI_STUDY_START,
        window_end=CHAI_STUDY_END,

        purpose="age_validation",
    ),


    ObservedTarget(

        name="CHAI_HRHPV_HIVneg_35_39",

        log_column=(
            "Calib_Any_F_35_39_"
            "HIVneg_Prev"
        ),

        positive=116,
        denominator=452,

        window_start=CHAI_STUDY_START,
        window_end=CHAI_STUDY_END,

        purpose="age_validation",
    ),


    ObservedTarget(

        name="CHAI_HRHPV_HIVneg_40_44",

        log_column=(
            "Calib_Any_F_40_44_"
            "HIVneg_Prev"
        ),

        positive=79,
        denominator=325,

        window_start=CHAI_STUDY_START,
        window_end=CHAI_STUDY_END,

        purpose="age_validation",
    ),


    ObservedTarget(

        name="CHAI_HRHPV_HIVneg_45_49",

        log_column=(
            "Calib_Any_F_45_49_"
            "HIVneg_Prev"
        ),

        positive=52,
        denominator=230,

        window_start=CHAI_STUDY_START,
        window_end=CHAI_STUDY_END,

        purpose="age_validation",
    ),


    # =========================================================================
    # 12. CHAI AGE-SPECIFIC HIV-POSITIVE VALIDATION
    # =========================================================================
    ObservedTarget(
        name="CHAI_HRHPV_HIVpos_25_29",
        log_column=(
            "Calib_Any_F_25_29_"
            "HIVpos_Prev"
        ),

        positive=177,
        denominator=434,

        window_start=CHAI_STUDY_START,
        window_end=CHAI_STUDY_END,

        purpose="HIV_age_validation",
    ),


    ObservedTarget(

        name="CHAI_HRHPV_HIVpos_30_34",

        log_column=(
            "Calib_Any_F_30_34_"
            "HIVpos_Prev"
        ),

        positive=169,
        denominator=445,

        window_start=CHAI_STUDY_START,
        window_end=CHAI_STUDY_END,

        purpose="HIV_age_validation",
    ),


    ObservedTarget(

        name="CHAI_HRHPV_HIVpos_35_39",

        log_column=(
            "Calib_Any_F_35_39_"
            "HIVpos_Prev"
        ),

        positive=193,
        denominator=576,

        window_start=CHAI_STUDY_START,
        window_end=CHAI_STUDY_END,

        purpose="HIV_age_validation",
    ),


    ObservedTarget(

        name="CHAI_HRHPV_HIVpos_40_44",

        log_column=(
            "Calib_Any_F_40_44_"
            "HIVpos_Prev"
        ),

        positive=196,
        denominator=562,

        window_start=CHAI_STUDY_START,
        window_end=CHAI_STUDY_END,

        purpose="HIV_age_validation",
    ),


    ObservedTarget(
        name="CHAI_HRHPV_HIVpos_45_49",

        log_column=(
            "Calib_Any_F_45_49_"
            "HIVpos_Prev"
        ),

        positive=137,
        denominator=407,

        window_start=CHAI_STUDY_START,
        window_end=CHAI_STUDY_END,

        purpose="HIV_age_validation",
    ),


    # =========================================================================
    # 13. EXTERNAL P3 VALIDATION
    # =========================================================================

    ObservedTarget(

        name=(
            "External_P3_"
            "HIVneg_25_50"
        ),

        log_column=(
            "Calib_Xpert_P3_F_25_50_"
            "HIVneg_Prev"
        ),

        positive=93,
        denominator=625,

        window_start=EXTERNAL_STUDY_START,
        window_end=EXTERNAL_STUDY_END,

        purpose="external_genotype_validation",
    ),


    ObservedTarget(

        name=(
            "External_P3_"
            "HIVpos_25_50"
        ),

        log_column=(
            "Calib_Xpert_P3_F_25_50_"
            "HIVpos_Prev"
        ),

        positive=168,
        denominator=625,

        window_start=EXTERNAL_STUDY_START,
        window_end=EXTERNAL_STUDY_END,

        purpose="external_genotype_validation",
    ),
]

# =============================================================================
# 14. ALL TARGETS
# =============================================================================

ALL_TARGETS = [
    CALIBRATION_TARGET,
    *VALIDATION_TARGETS,
]

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
            "Before running calibration,"
            "hpv.py should contain age ranges "
            "equivalent to:\n\n"

            "CALIBRATION_AGE_RANGES = {\n"
            "    '25_29': (25, 30),\n"
            "    '30_34': (30, 35),\n"
            "    '35_39': (35, 40),\n"
            "    '40_44': (40, 45),\n"
            "    '45_49': (45, 50),\n"
            "    '25_49': (25, 50),\n"
            "    '25_50': (25, 51),\n"
            "}\n"
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

# =============================================================================
# 11. PARALLEL GRID SEARCH
# =============================================================================
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

    with ProcessPoolExecutor(max_workers=max_workers, max_tasks_per_child=1,) as executor:
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

def make_validation_table(
    best_beta: float,
    all_replicates: pd.DataFrame,
) -> pd.DataFrame:

    best_replicates = (
        all_replicates.loc[
            np.isclose(all_replicates["b_hpv"],best_beta,)
        ].copy())

    rows = []

    # Include calibration target first, then validation targets.
    for target in ALL_TARGETS:

        values = pd.to_numeric(
            best_replicates[target.name],
            errors="coerce",
        ).dropna()

        simulated_mean = (
            float(values.mean())
            if not values.empty
            else np.nan
        )

        simulated_sd = (float(values.std(ddof=1))
            if len(values) > 1
            else np.nan
        )

        rows.append(
            {
                "target": (target.name),
                "purpose": (target.purpose),
                "observed": (target.observed),
                "observed_se": (target.observed_se),
                "sim_mean": (simulated_mean),
                "sim_sd": (simulated_sd),
                "residual": (simulated_mean - target.observed
                    if not np.isnan(simulated_mean)
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(
        rows
    )

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
    # STEP 1 PARALLEL FIXED-GRID SEARCH
    # =========================================================================
    print(
        "\n=== STEP 1: "
        "PARALLEL FIXED-GRID SEARCH ==="
    )
    grid_results, grid_replicates = run_parallel_grid_search()

    # ============================================================
    # STEP 2: SELECT BEST b_hpv
    # ============================================================
    best_row = grid_results.loc[grid_results["loss"].idxmin()]
    best_beta = float(best_row["b_hpv"])
    best_loss = float(best_row["loss"])

    print(
        f"\nSearch-stage best "
        f"b_hpv = "
        f"{best_beta:.6f}"
    )

    print(
        f"Loss = "
        f"{best_loss:.6f}"
    )

    # =========================================================================
    # STEP 3. GRID BOUNDARY CHECK
    # =========================================================================
    if (np.isclose(best_beta, min(B_HPV_GRID))
        or np.isclose(best_beta, max(B_HPV_GRID)
    )):
        print(
            "\nWARNING:\n"
            "The minimum-loss b_hpv based on the "
            "multi-seed mean is at a grid boundary.\n"
            "Expand B_HPV_GRID before performing "
            "the fine calibration search."
        )

    # =========================================================================
    # STEP 4. BUILD VALIDATION TABLE
    # =========================================================================

    validation_table = (
        make_validation_table(best_beta=(best_beta), all_replicates=(grid_replicates),)
    )

    validation_table.to_csv(OUTPUT_ROOT / "best_b_hpv_validation.csv", index=False,)

    # ============================================================
    # FINAL OUTPUT
    # ============================================================

    print("\n======================================")
    print("GRID SEARCH COMPLETE")
    print("======================================")

    display_columns = [
        "b_hpv",
        "loss",
        "observed",
        "sim_mean",
        "sim_sd",
        "sim_mcse",
        "residual",
        "standardized_residual",
        "n_seeds",
    ]

    print("\nCalibration results:")

    print(grid_results[display_columns]
        .to_string(index=False)
    )

    print("\n======================================")

    print("VALIDATION AT BEST b_hpv")

    print("======================================")

    print(validation_table.to_string(index=False))

    print("\n======================================")

    print("FINAL RESULT")

    print("======================================")

    print(
        f"\nBest b_hpv = "
        f"{best_beta:.6f}"
    )

    print(
        f"Based on "
        f"{len(SEARCH_SEEDS)} "
        f"stochastic seeds."
    )

    print(
        "\nCalibration:"
        "\n  CHAI HIV-negative "
        "overall hrHPV prevalence."
    )

    print(
        "\nValidation:"
        "\n  - CHAI HIV-positive overall hrHPV"
        "\n  - CHAI age-specific HIV-negative hrHPV"
        "\n  - CHAI age-specific HIV-positive hrHPV"
        "\n  - CHAI HPV16"
        "\n  - CHAI HPV18/45"
        "\n  - External P3"
    )

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_ROOT}"
    )

# =============================================================================
# 16. RUN SCRIPT
# =============================================================================
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
