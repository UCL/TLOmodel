from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize_scalar
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

RESOURCE_FILEPATH = Path("./resources").resolve()

OUTPUT_ROOT = Path("./outputs/hpv_b_hpv_single_target_calibration").resolve()

START_DATE = Date(2010, 1, 1)
END_DATE = Date(2023, 1, 1)
ORIGINAL_B_HPV = 0.75

TEST_MODE = True

if TEST_MODE:

    POPULATION_SIZE = 2000
    SEARCH_SEEDS = [1001, 1002,]
    CONFIRMATION_SEEDS = [2001, 2002, 2003,]
    OPTIMIZER_MAXITER = 8

else:
    POPULATION_SIZE = 20000
    SEARCH_SEEDS = [1001, 1002, 1003, 1004, 1005,]
    CONFIRMATION_SEEDS = list(range(2001, 2021,))
    OPTIMIZER_MAXITER = 25

B_HPV_BOUNDS = (0.05, 3.0,)
B_HPV_XATOL = 0.01

RANGE_CHECK_BETAS = [0.25, 0.50, 0.75, 1.00, 1.50,]

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

    for seed in seeds:
        print(
            f"Running "
            f"b_hpv={b_hpv:.6f}, "
            f"seed={seed}"
        )

        run_directory = (output_root/f"beta_{beta_label}"/f"seed_{seed}")

        result = run_one_simulation(
            b_hpv=(b_hpv),
            seed=(seed),
            run_directory=(run_directory),
        )

        replicate_records.append(result)

    replicate_df = pd.DataFrame(replicate_records)

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
# 11. DIAGNOSTIC RANGE CHECK
# =============================================================================
def run_range_check() -> pd.DataFrame:
    records = []

    for beta in RANGE_CHECK_BETAS:

        print(
            f"\nRange check: "
            f"b_hpv={beta}"
        )

        summary, _ = evaluate_b_hpv(
            b_hpv=float(beta),
            seeds=(SEARCH_SEEDS),
            output_root=(OUTPUT_ROOT / "range_check"),
        )

        records.append(summary)

    result = (pd.DataFrame(records).sort_values("b_hpv").reset_index(drop=True))

    result.to_csv(OUTPUT_ROOT / "range_check_results.csv", index = False,)

    return result

SEARCH_CACHE: dict[float, dict[str, Any],] = {}

def objective_function(b_hpv: float,) -> float:

    cache_key = round(float(b_hpv), 8,)

    if (cache_key not in SEARCH_CACHE):

        print(
            f"\nOptimizer evaluating "
            f"b_hpv={b_hpv:.6f}"
        )

        summary, _ = evaluate_b_hpv(b_hpv=float(b_hpv),
            seeds=(SEARCH_SEEDS),
            output_root=(OUTPUT_ROOT / "optimizer"),
        )

        SEARCH_CACHE[cache_key] = summary

        pd.DataFrame(SEARCH_CACHE.values()).sort_values("b_hpv").to_csv(
            OUTPUT_ROOT / "optimizer_evaluations.csv",
            index=False,
        )

    return float(SEARCH_CACHE[cache_key]["loss"])

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
    range_check: pd.DataFrame,
    optimizer_df: pd.DataFrame,
    target_comparison: pd.DataFrame,
    best_beta: float,
) -> None:

    # =========================================================================
    # FIGURE 1 b_hpv versus model HR-HPV prevalence
    # =========================================================================

    plt.figure(figsize=(8, 5))
    plt.plot(range_check["b_hpv"],
        range_check["sim_mean"],
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
    plt.title("Diagnostic range check: ""b_hpv versus HR-HPV prevalence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_ROOT / "figure_1_range_check.png", dpi=300,)
    plt.close()

    # =========================================================================
    # FIGURE 2 b_hpv versus least-squares loss
    # =========================================================================
    optimizer_df = (optimizer_df.sort_values("b_hpv").copy())
    plt.figure(figsize=(8, 5))
    plt.plot(optimizer_df["b_hpv"],
        optimizer_df["loss"],
        marker="o",
    )
    plt.axvline(best_beta, linestyle="--", label=(f"Calibrated "f"b_hpv={best_beta:.4f}"),)
    plt.xlabel("b_hpv")
    plt.ylabel("Squared standardized residual")
    plt.title("One-dimensional least-squares ""calibration objective")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_ROOT / "figure_2_optimizer_loss.png", dpi=300,)
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
    # STEP 1 DIAGNOSTIC RANGE CHECK
    # =========================================================================
    print(
        "\n=== STEP 1: "
        "RANGE CHECK ==="
    )

    range_check = (run_range_check())
    min_model_prev = float(range_check["sim_mean"].min())
    max_model_prev = float(range_check["sim_mean"].max())

    if not (
        min_model_prev <= CALIBRATION_TARGET.observed <= max_model_prev):

        print(
            "\nWARNING:\n"
            "Observed prevalence is outside "
            "the range generated by "
            "RANGE_CHECK_BETAS.\n"

            "This does not automatically "
            "invalidate the optimizer, "

            "but you should inspect "
            "B_HPV_BOUNDS and Figure 1."
        )

    # =========================================================================
    # STEP 2 ONE-DIMENSIONAL LEAST-SQUARES OPTIMISATION
    # =========================================================================

    print(
        "\n=== STEP 2: "
        "ONE-DIMENSIONAL "
        "LEAST-SQUARES OPTIMIZATION ==="
    )

    optimization = minimize_scalar(
        objective_function,
        bounds=(B_HPV_BOUNDS),
        method="bounded",
        options={"xatol": (B_HPV_XATOL), "maxiter": (OPTIMIZER_MAXITER),},
    )

    # -------------------------------------------------------------------------
    # Check optimiser success
    # -------------------------------------------------------------------------
    if not optimization.success:
        print(
            "\nWARNING: optimizer did not "
            "report convergence:"
        )

        print(optimization.message)

    # -------------------------------------------------------------------------
    # Best b_hpv
    # -------------------------------------------------------------------------

    best_beta = float(optimization.x)

    optimizer_df = (pd.DataFrame(SEARCH_CACHE.values()).sort_values("b_hpv").reset_index(drop=True))

    optimizer_df.to_csv(OUTPUT_ROOT/"optimizer_evaluations.csv", index=False,)

    print(
        f"\nSearch-stage best "
        f"b_hpv = "
        f"{best_beta:.6f}"
    )

    print(
        f"Search-stage minimum "
        f"loss = "
        f"{optimization.fun:.6f}"
    )

    lower_bound, upper_bound = (B_HPV_BOUNDS)

    boundary_margin = (0.05 * (upper_bound - lower_bound))

    if (
        best_beta <= lower_bound + boundary_margin
        or
        best_beta >= upper_bound - boundary_margin
    ):

        print(
            "\nWARNING:\n"
            "Best b_hpv is close to a "
            "search boundary.\n"
            "Expand B_HPV_BOUNDS before "
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
        range_check=(range_check),
        optimizer_df=(optimizer_df),
        target_comparison=(target_comparison),
        best_beta=(best_beta),
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
    main()
