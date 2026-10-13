"""Extract population, HSI counts, and deaths for treatment-id sanity checks.

Run from the repository root, for example::

    python src/scripts/lcoa_inputs_from_tlo_analyses/sanity_checks.py \
        outputs/s.bhatia@imperial.ac.uk/effect_of_each_treatment_id-2026-10-09T145410Z \
        outputs/generated_outputs \
        --target-start=2010-01-01 --target-end=2025-12-31

The output folder defaults to the results folder. Comparisons use the Nothing
scenario; use --do-comparison=False for results without that scenario.
"""

import argparse
import pickle
from datetime import date
from pathlib import Path

import pandas as pd

from scripts.lcoa_inputs_from_tlo_analyses.results_processing_utils import (
    find_difference_extra_relative_to_comparison,
    get_counts_of_hsi_by_short_treatment_id,
    get_parameter_names_from_scenario_file,
    get_total_population_by_year,
    make_get_num_deaths_by_cause_label_and_period,
    make_get_counts_of_hsis_by_period,
    set_param_names_as_column_index_level_0,
)
from tlo import Date
from tlo.analysis.utils import compute_summary_statistics, extract_results


EXCLUDED_HSIs = [
    "FirstAttendance_Emergency",
    "FirstAttendance_NonEmergency",
    "Inpatient_Care",
]
# FirstAttendance_SpuriousEmergencyCare has a non-blank appointment footprint.


def parse_iso_date(value: str) -> Date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use an ISO date (YYYY-MM-DD).") from exc
    return Date(parsed.year, parsed.month, parsed.day)


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "t", "1", "yes", "y"}:
        return True
    if normalized in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value '{value}'. Use True or False.")


def apply(
    results_folder: Path,
    target_period_tuple: tuple[Date, Date],
    do_comparison: bool = True,
) -> dict:
    """Return summarized sanity-check results for the selected period."""
    param_names = get_parameter_names_from_scenario_file()
    results = {}

    get_num_deaths_by_cause_label_and_period = make_get_num_deaths_by_cause_label_and_period(
        target_period_tuple,
    )

    # Get total population by year
    print("Extracting population data...")
    total_population_by_year = (
        extract_results(
            results_folder,
            module='tlo.methods.demography',
            key='population',
            custom_generate_series=lambda _df: get_total_population_by_year(_df, target_period_tuple),
            do_scaling=True,
            autodiscover=True
        ).pipe(set_param_names_as_column_index_level_0, param_names=param_names)
    )

    cols_to_drop = [
        col for col in total_population_by_year.columns
        if col[0].strip().startswith(tuple(EXCLUDED_HSIs))
    ]

    print(f"Dropping columns from total_population_by_year: {cols_to_drop}")
    total_population_by_year.drop(columns=cols_to_drop, inplace=True)

    total_population_by_year = compute_summary_statistics(total_population_by_year, central_measure='median')
    results['total_population_by_year'] = total_population_by_year

    print("Extracting HSI event counts...")
    counts_of_hsi_by_short_treatment_id = (
        extract_results(
            results_folder,
            module="tlo.methods.healthsystem.summary",
            key="HSI_Event",
            custom_generate_series=lambda _df: get_counts_of_hsi_by_short_treatment_id(_df, target_period_tuple),
            do_scaling=True,
            autodiscover=True,
        )
        .pipe(set_param_names_as_column_index_level_0, param_names=param_names)
        .fillna(0.0)
        .sort_index()
    ).drop(EXCLUDED_HSIs, errors='ignore')

    results['counts_of_hsi_by_short_treatment_id'] = (
        compute_summary_statistics(counts_of_hsi_by_short_treatment_id, 'median')
    )

    print("Extracting yearly HSI event counts...")
    get_num_hsi_by_period = make_get_counts_of_hsis_by_period(target_period_tuple)
    counts_of_hsi_by_period = (
        extract_results(
            results_folder,
            module="tlo.methods.healthsystem.summary",
            key="HSI_Event",
            custom_generate_series=get_num_hsi_by_period,
            do_scaling=True,
            autodiscover=True,
        )
        .pipe(set_param_names_as_column_index_level_0, param_names=param_names)
        .fillna(0.0)
        .sort_index()
    ).drop(columns=cols_to_drop, errors='ignore')

    results['counts_of_hsi_by_period'] = (
        compute_summary_statistics(counts_of_hsi_by_period, 'median')
    )

    print("Extracting deaths by cause and period...")
    num_deaths = (
        extract_results(
            results_folder,
            module="tlo.methods.demography",
            key="death",
            custom_generate_series=get_num_deaths_by_cause_label_and_period,
            do_scaling=True,
            autodiscover=True,
        ).pipe(set_param_names_as_column_index_level_0, param_names=param_names)
    )


    num_deaths = num_deaths.drop(columns=cols_to_drop, errors='ignore')

    if do_comparison:
        num_deaths_averted = compute_summary_statistics(
            -1.0 * pd.DataFrame(
                find_difference_extra_relative_to_comparison(
                    num_deaths.sum(), comparison='Nothing')).T,
            central_measure='median'
        ).iloc[0].unstack()

        pc_deaths_averted = 100.0 * compute_summary_statistics(
            -1.0 * pd.DataFrame(
                find_difference_extra_relative_to_comparison(
                    num_deaths.sum(), comparison='Nothing', scaled=True)).T,
            central_measure='median'
        ).iloc[0].unstack()
    else:
        num_deaths_averted = None
        pc_deaths_averted = None

    num_deaths = compute_summary_statistics(num_deaths, central_measure='median')

    results['num_deaths'] = num_deaths
    results['num_deaths_averted'] = num_deaths_averted
    results['pc_deaths_averted'] = pc_deaths_averted

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_folder", type=Path)
    parser.add_argument("output_folder", type=Path, nargs="?", default=None)
    parser.add_argument("--target-start", type=parse_iso_date, required=True)
    parser.add_argument("--target-end", type=parse_iso_date, required=True)
    parser.add_argument("--do-comparison", type=parse_bool, default=True)
    args = parser.parse_args()

    if not args.target_start < args.target_end:
        parser.error("--target-start must be earlier than --target-end.")

    out = args.output_folder if args.output_folder is not None else args.results_folder
    results = apply(
        results_folder=args.results_folder,
        target_period_tuple=(args.target_start, args.target_end),
        do_comparison=False,
    )
    out.mkdir(parents=True, exist_ok=True)
    outfile = out / f"{args.target_end:%Y-%m-%d}_sanity_checks.pkl"
    with outfile.open("wb") as f:
        pickle.dump(results, f)
    print(f"Sanity checks complete! Results saved to {outfile}")


if __name__ == "__main__":
    main()
