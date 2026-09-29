"""Compare HBP DALYs averted with the sum from their interventions.

All differences and sums are calculated for each run before the median and
95% uncertainty interval are calculated.

python src/scripts/lcoa_inputs_from_tlo_analyses/compare_hbp_dalys_averted.py \
  outputs/s.bhatia@imperial.ac.uk/effect_of_each_treatment_id-combined \
  outputs/s.bhatia@imperial.ac.uk/effect_of_each_hbp-combined \
  --target-start 2026-01-01 \
  --target-end 2040-12-31
"""

import argparse
import pickle
import warnings
from datetime import date
from pathlib import Path

import pandas as pd

from tlo import Date
from tlo.analysis.utils import compute_summary_statistics, extract_results

from scripts.lcoa_inputs_from_tlo_analyses.results_processing_utils import (
    find_difference_extra_relative_to_comparison,
    get_parameter_names_from_scenario_file,
    make_get_num_dalys_by_cause_label_and_period,
    set_param_names_as_column_index_level_0,
)
from scripts.lcoa_inputs_from_tlo_analyses.scenario_effect_of_hbps import EffectOfEachHBP


PERIOD_LENGTH_YEARS = 1
DEFAULT_OUTPUT_FILE = Path("hbp_dalys_averted_comparison.csv")
HBP_RESULTS_PICKLE = (
    Path(__file__).resolve().parents[3]
    / "outputs"
    / "generated_outputs"
    / "2040-12-31_hbp_fullresults.pkl"
)

# Edit these lists to change the treatment IDs included in each HBP total.
HBP_TO_TREATMENT_IDS = {
    "LIT-HBP": [
        "Alri_Pneumonia_Treatment_Outpatient_*",
        "AntenatalCare_FollowUp_*",
        "AntenatalCare_Outpatient_*",
        "Contraception_Routine_*",
        "Epi_Childhood_DtpHibHep_*",
        "Epi_Childhood_MeaslesRubella_*",
        "Epi_Childhood_Rota_*",
        "Epi_Pregnancy_Td_*",
        "Hiv_Prevention_Circumcision_*",
        "Hiv_Prevention_Infant_*",
        "Hiv_Test_*",
        "DeliveryCare_Basic_*",
        "DeliveryCare_Comprehensive_*",
        "DeliveryCare_Neonatal_*",
        "PostnatalCare_Maternal_Inpatient_*",
        "PostnatalCare_Maternal_*",
        "Malaria_Prevention_Iptp_*",
        "Malaria_Treatment_*",
        "PostnatalCare_Neonatal_*",
        "PostnatalCare_Neonatal_Inpatient_*",
        "PostnatalCare_TreatmentForObstetricFistula_*",
        "Tb_Prevention_Ipt_*",
        "Tb_Treatment_*",
    ],
    "TLO-HBP": [
        "Alri_Pneumonia_Treatment_Outpatient_*",
        "AntenatalCare_Outpatient_*",
        "CardioMetabolicDisorders_Treatment_*",
        "Contraception_Routine_*",
        "Diarrhoea_Treatment_Inpatient_*",
        "Diarrhoea_Treatment_Outpatient_*",
        "Epi_Childhood_MeaslesRubella_*",
        "Hiv_Prevention_Prep_*",
        "Hiv_Treatment_*",
        "Malaria_Treatment_*",
        "Malaria_Treatment_Complicated_*",
        "Measles_Treatment_*",
        "Schisto_MDA_*",
        "Undernutrition_Feeding_*",
        "Undernutrition_Feeding_Inpatient_*",
        "Undernutrition_Feeding_Outpatient_*",
        "Undernutrition_Feeding_Supplementary_*",
    ],
}


def parse_iso_date(value: str) -> Date:
    parsed = date.fromisoformat(value)
    return Date(parsed.year, parsed.month, parsed.day)


def format_hbp_scenario_name(scenario_name: str) -> str:
    names = {
        "LCOA EHP from RWE": "LIT-HBP",
        "LCOA EHP from TLO": "TLO-HBP",
    }
    return names.get(scenario_name, scenario_name.removeprefix("Only "))


def get_hbp_parameter_names() -> tuple[str, ...]:
    return tuple(EffectOfEachHBP()._scenarios.keys())


def set_hbp_names(
    dataframe: pd.DataFrame,
    parameter_names: tuple[str, ...],
) -> pd.DataFrame:
    draw_names = dict(enumerate(parameter_names))
    names = [draw_names.get(draw) for draw in dataframe.columns.levels[0]]
    if any(name is None for name in names):
        raise ValueError("Could not map every HBP draw number to a scenario name.")
    dataframe.columns = dataframe.columns.set_levels(
        [format_hbp_scenario_name(name) for name in names],
        level=0,
    )
    return dataframe


def extract_dalys(
    results_folder: Path,
    target_period: tuple[Date, Date],
    parameter_names: tuple[str, ...],
    hbp_results: bool = False,
) -> pd.DataFrame:
    get_dalys = make_get_num_dalys_by_cause_label_and_period(
        target_period,
    )
    dalys = extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked_by_age_and_time",
        custom_generate_series=get_dalys,
        do_scaling=True,
        autodiscover=True,
    )
    if hbp_results:
        return dalys.pipe(set_hbp_names, parameter_names=parameter_names)
    return dalys.pipe(
        set_param_names_as_column_index_level_0,
        param_names=parameter_names,
    )


def available_configured_draws(
    dataframe: pd.DataFrame,
    draws: set[str],
    source: str,
) -> set[str]:
    """Return configured draws that exist, warning about those that do not."""
    available = set(dataframe.columns.get_level_values("draw"))
    missing = draws - available
    if missing:
        warnings.warn(
            f"Ignoring configured draws missing from {source}: {sorted(missing)}",
            stacklevel=2,
        )
    return draws & available


def runs_for_draw(dataframe: pd.DataFrame, draw: str) -> set:
    draw_columns = dataframe.xs(draw, level="draw", axis=1, drop_level=False)
    return set(draw_columns.columns.get_level_values("run"))


def warn_about_excluded_runs(draw: str, runs: set, common_runs: set, source: str) -> None:
    excluded = runs - common_runs
    if excluded:
        warnings.warn(
            f"Ignoring runs from {source} draw {draw!r} that are not shared by "
            f"all included draws: {sorted(excluded)}",
            stacklevel=2,
        )


def dalys_averted(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return Nothing minus each other draw, retaining individual runs."""
    return -1.0 * pd.DataFrame(
        find_difference_extra_relative_to_comparison(
            dataframe.sum(axis=0, min_count=1),
            comparison="Nothing",
        )
    ).T


def dalys_averted_by_cause(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return Nothing minus each draw by cause and run, summed over periods."""
    if "label" not in dataframe.index.names:
        raise ValueError("DALY data must have a 'label' index level.")

    dalys_by_cause = dataframe.groupby(level="label").sum(min_count=1)
    cause_specific_results = []
    for cause_label, row in dalys_by_cause.iterrows():
        difference = -1.0 * pd.DataFrame(
            find_difference_extra_relative_to_comparison(
                row,
                comparison="Nothing",
            )
        ).T
        difference.index = pd.Index([cause_label], name="label")
        cause_specific_results.append(difference)
    return pd.concat(cause_specific_results)


def save_dalys_averted_by_cause(
    dalys_averted_by_cause_summary: pd.DataFrame,
    pickle_path: Path = HBP_RESULTS_PICKLE,
) -> None:
    """Add cause-specific DALYs averted to an existing or new results pickle."""
    if pickle_path.exists():
        with open(pickle_path, "rb") as file:
            results = pickle.load(file)
        if not isinstance(results, dict):
            raise TypeError(f"Expected a dictionary in {pickle_path}, got {type(results).__name__}.")
    else:
        results = {}
        pickle_path.parent.mkdir(parents=True, exist_ok=True)

    results["dalys_averted_by_cause"] = dalys_averted_by_cause_summary
    with open(pickle_path, "wb") as file:
        pickle.dump(results, file)


def compare_dalys_averted(
    treatment_id_results: Path,
    hbp_results: Path,
    target_period: tuple[Date, Date],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    treatment_dalys = extract_dalys(
        treatment_id_results,
        target_period,
        get_parameter_names_from_scenario_file(),
    )
    hbp_dalys = extract_dalys(
        hbp_results,
        target_period,
        get_hbp_parameter_names(),
        hbp_results=True,
    )

    configured_hbps = set(HBP_TO_TREATMENT_IDS)
    configured_treatments = {
        treatment
        for treatments in HBP_TO_TREATMENT_IDS.values()
        for treatment in treatments
    }
    treatment_draws = set(treatment_dalys.columns.get_level_values("draw"))
    if "Nothing" not in treatment_draws:
        raise ValueError("The mandatory Nothing draw is missing from treatment-ID results.")
    available_treatments = available_configured_draws(
        treatment_dalys,
        configured_treatments,
        "treatment-ID results",
    )
    available_hbps = available_configured_draws(
        hbp_dalys,
        configured_hbps,
        "HBP results",
    )
    if not available_hbps:
        raise ValueError("None of the configured HBP draws exist in the HBP results.")

    hbp_to_available_treatments = {
        hbp: [treatment for treatment in treatments if treatment in available_treatments]
        for hbp, treatments in HBP_TO_TREATMENT_IDS.items()
        if hbp in available_hbps
    }

    treatment_runs = {
        draw: runs_for_draw(treatment_dalys, draw)
        for draw in available_treatments | {"Nothing"}
    }
    hbp_runs = {
        draw: runs_for_draw(hbp_dalys, draw)
        for draw in hbp_to_available_treatments
    }
    all_run_sets = [*treatment_runs.values(), *hbp_runs.values()]
    common_runs = set.intersection(*all_run_sets)
    if not common_runs:
        raise ValueError("There are no run identifiers shared by all included draws.")
    for draw, runs in treatment_runs.items():
        warn_about_excluded_runs(draw, runs, common_runs, "treatment-ID results")
    for draw, runs in hbp_runs.items():
        warn_about_excluded_runs(draw, runs, common_runs, "HBP results")

    treatment_columns = treatment_dalys.columns.get_level_values("draw").isin(
        available_treatments | {"Nothing"}
    ) & treatment_dalys.columns.get_level_values("run").isin(common_runs)
    treatment_averted = dalys_averted(treatment_dalys.loc[:, treatment_columns])

    nothing_columns = (
        (treatment_dalys.columns.get_level_values("draw") == "Nothing")
        & treatment_dalys.columns.get_level_values("run").isin(common_runs)
    )
    hbp_columns = (
        hbp_dalys.columns.get_level_values("draw").isin(available_hbps)
        & hbp_dalys.columns.get_level_values("run").isin(common_runs)
    )
    nothing_dalys = treatment_dalys.loc[:, nothing_columns]
    selected_hbp_dalys = hbp_dalys.loc[:, hbp_columns]
    if not nothing_dalys.index.equals(selected_hbp_dalys.index):
        raise ValueError("Treatment-ID and HBP DALY row indexes do not match.")
    combined_hbp_dalys = pd.concat([nothing_dalys, selected_hbp_dalys], axis=1)
    hbp_averted_by_cause = dalys_averted_by_cause(combined_hbp_dalys)
    hbp_averted = hbp_averted_by_cause.sum(axis=0, min_count=1).to_frame().T

    treatment_total_dalys = treatment_dalys.loc[:, treatment_columns].sum(
        axis=0,
        min_count=1,
    ).to_frame().T
    hbp_total_dalys = selected_hbp_dalys.sum(axis=0, min_count=1).to_frame().T

    intervention_sums = {}
    intervention_daly_sums = {}
    for hbp, treatments in hbp_to_available_treatments.items():
        if treatments:
            selected = treatment_averted.loc[:, pd.IndexSlice[treatments, :]]
            run_sums = selected.T.groupby(level="run", sort=False)[0].sum(min_count=1)

            selected_dalys = treatment_total_dalys.loc[
                :, pd.IndexSlice[treatments, :]
            ]
            daly_run_sums = selected_dalys.T.groupby(level="run", sort=False)[0].sum(
                min_count=1
            )
        else:
            warnings.warn(
                f"No configured treatment draws are available for {hbp}; "
                "the intervention sums will be zero.",
                stacklevel=2,
            )
            run_sums = pd.Series(0.0, index=sorted(common_runs), name=0)
            run_sums.index.name = "run"
            daly_run_sums = run_sums.copy()
        intervention_sums[hbp] = run_sums
        intervention_daly_sums[hbp] = daly_run_sums
    intervention_sums = pd.concat(intervention_sums, names=["draw", "run"]).to_frame().T
    intervention_daly_sums = pd.concat(
        intervention_daly_sums,
        names=["draw", "run"],
    ).to_frame().T

    # Compare like-for-like runs before calculating any summary statistics.
    additivity_difference = hbp_averted - intervention_sums
    daly_difference = hbp_total_dalys - intervention_daly_sums

    hbp_summary = compute_summary_statistics(hbp_averted, central_measure="median")
    hbp_by_cause_summary = compute_summary_statistics(
        hbp_averted_by_cause,
        central_measure="median",
    )
    intervention_summary = compute_summary_statistics(intervention_sums, central_measure="median")
    additivity_difference_summary = compute_summary_statistics(
        additivity_difference,
        central_measure="median",
    )
    hbp_dalys_summary = compute_summary_statistics(
        hbp_total_dalys,
        central_measure="median",
    )
    intervention_dalys_summary = compute_summary_statistics(
        intervention_daly_sums,
        central_measure="median",
    )
    daly_difference_summary = compute_summary_statistics(
        daly_difference,
        central_measure="median",
    )

    rows = []
    for hbp in hbp_to_available_treatments:
        rows.append(
            {
                "hbp": hbp,
                "hbp_dalys_averted_median": hbp_summary.loc[0, (hbp, "central")],
                "hbp_dalys_averted_lower_95_ui": hbp_summary.loc[0, (hbp, "lower")],
                "hbp_dalys_averted_upper_95_ui": hbp_summary.loc[0, (hbp, "upper")],
                "sum_intervention_dalys_averted_median": intervention_summary.loc[
                    0, (hbp, "central")
                ],
                "sum_intervention_dalys_averted_lower_95_ui": intervention_summary.loc[
                    0, (hbp, "lower")
                ],
                "sum_intervention_dalys_averted_upper_95_ui": intervention_summary.loc[
                    0, (hbp, "upper")
                ],
                "hbp_minus_sum_interventions_median": additivity_difference_summary.loc[
                    0, (hbp, "central")
                ],
                "hbp_minus_sum_interventions_lower_95_ui": additivity_difference_summary.loc[
                    0, (hbp, "lower")
                ],
                "hbp_minus_sum_interventions_upper_95_ui": additivity_difference_summary.loc[
                    0, (hbp, "upper")
                ],
                "hbp_dalys_median": hbp_dalys_summary.loc[0, (hbp, "central")],
                "hbp_dalys_lower_95_ui": hbp_dalys_summary.loc[0, (hbp, "lower")],
                "hbp_dalys_upper_95_ui": hbp_dalys_summary.loc[0, (hbp, "upper")],
                "sum_intervention_dalys_median": intervention_dalys_summary.loc[
                    0, (hbp, "central")
                ],
                "sum_intervention_dalys_lower_95_ui": intervention_dalys_summary.loc[
                    0, (hbp, "lower")
                ],
                "sum_intervention_dalys_upper_95_ui": intervention_dalys_summary.loc[
                    0, (hbp, "upper")
                ],
                "hbp_minus_sum_intervention_dalys_median": daly_difference_summary.loc[
                    0, (hbp, "central")
                ],
                "hbp_minus_sum_intervention_dalys_lower_95_ui": daly_difference_summary.loc[
                    0, (hbp, "lower")
                ],
                "hbp_minus_sum_intervention_dalys_upper_95_ui": daly_difference_summary.loc[
                    0, (hbp, "upper")
                ],
            }
        )
    return pd.DataFrame(rows).set_index("hbp"), hbp_by_cause_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("treatment_id_results", type=Path)
    parser.add_argument("hbp_results", type=Path)
    parser.add_argument("--target-start", type=parse_iso_date, required=True)
    parser.add_argument("--target-end", type=parse_iso_date, required=True)
    args = parser.parse_args()

    if not args.target_start < args.target_end:
        parser.error("--target-start must be earlier than --target-end.")

    comparison, hbp_by_cause_summary = compare_dalys_averted(
        args.treatment_id_results,
        args.hbp_results,
        (args.target_start, args.target_end),
    )
    comparison.to_csv(DEFAULT_OUTPUT_FILE)
    print(f"Comparison saved to {DEFAULT_OUTPUT_FILE.resolve()}")
    save_dalys_averted_by_cause(hbp_by_cause_summary)
    print(f"Cause-specific DALYs averted saved to {HBP_RESULTS_PICKLE}")


if __name__ == "__main__":
    main()
