"""Analyse jointly simulated health benefit packages against an exported Nothing draw."""

import argparse
import pickle
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact

from scripts.lcoa_inputs_from_tlo_analyses.analysis_effect_of_treatment_ids import (
    YEAR_OF_SWITCH,
    analysis_metadata,
    parse_iso_date,
    process_scenarios,
)

from tlo import Date
from tlo.analysis.utils import compute_summary_statistics, get_filtered_treatment_ids

EXCLUDED_HSIS = {
    "FirstAttendance_Emergency",
    "FirstAttendance_NonEmergency",
    "FirstAttendance_SpuriousEmergencyCare",
    "Inpatient_Care",
}
HBP_LABELS = {"LIT_HBP": "LIT-HBP", "TLO_HBP": "TLO-HBP"}
BASELINE_WIDE_KEYS = (
    "total_population_by_year",
    "counts_of_hsi_by_short_treatment_id",
    "counts_of_hsi_by_period",
    "num_deaths",
    "discounted_dalys",
    "capacity_fractions",
)
BASELINE_COST_KEYS = ("input_costs", "final_costs")


def treatment_id(name: str) -> str:
    return name.removeprefix("Only ").removesuffix("_*")


def read_packages(results_folder: Path) -> dict[str, set[str]]:

    LIT_HBP = (
        ["Alri_Pneumonia_Treatment_Outpatient",
         "AntenatalCare_FollowUp",
         "AntenatalCare_Outpatient",
         "Contraception_Routine",
         "Epi_Childhood_DtpHibHep",
         "Epi_Childhood_MeaslesRubella",
         "Epi_Childhood_Rota",
         "Epi_Pregnancy_Td",
         "Hiv_Prevention_Circumcision",
         "Hiv_Prevention_Infant",
         "Hiv_Test",
         "DeliveryCare_Basic",
         "DeliveryCare_Comprehensive",
         ## Excluded because DeliveryCare_Neonatal is unused in TLO.
         ## "DeliveryCare_Neonatal",
         "PostnatalCare_Maternal_Inpatient",
         "PostnatalCare_Maternal",
         "Malaria_Prevention_Iptp",
         "Malaria_Treatment",
         "PostnatalCare_Neonatal",
         "PostnatalCare_Neonatal_Inpatient",
         "PostnatalCare_TreatmentForObstetricFistula",
         "Tb_Prevention_Ipt",
         "Tb_Treatment"]
    )

    TLO_HBP = (
         [
          "Contraception_Routine",
          "Epi_Childhood_MeaslesRubella",
          "Hiv_Prevention_Infant",
          "Hiv_Treatment",
          "Malaria_Treatment_Complicated",
          "Measles_Treatment",
          "Schisto_MDA",
          "Undernutrition_Feeding_Inpatient",
          "Undernutrition_Feeding_Outpatient",
          "Undernutrition_Feeding_Supplementary"]
    )
    packages = {'LIT_HBP': set(LIT_HBP), 'TLO_HBP': set(TLO_HBP)}
    return packages


def overlap_and_fisher_test(packages: dict[str, set[str]], universe: set[str]) -> dict:
    universe = {treatment_id(name) for name in universe} - EXCLUDED_HSIS
    members = {name: package & universe for name, package in packages.items()}
    outside_universe = {name: package - universe - EXCLUDED_HSIS for name, package in packages.items()}
    first, second = members.values()
    counts = {
        "first_only": len(first - second),
        "both": len(first & second),
        "second_only": len(second - first),
        "neither": len(universe - (first | second)),
    }
    odds_ratio, p_value = fisher_exact(
        [[counts["both"], counts["first_only"]], [counts["second_only"], counts["neither"]]],
        alternative="two-sided",
    )
    return {
        "members": members, "universe_size": len(universe), "counts": counts,
        "outside_universe": outside_universe,
        "odds_ratio": float(odds_ratio), "p_value": float(p_value),
    }


def match_interventions(packages: dict[str, set[str]], totals: dict[str, pd.Series]) -> dict[str, list[str]]:
    draw_names = {}
    for name in totals["averted"].index.get_level_values("draw").unique():
        key = treatment_id(name)
        if key in draw_names:
            raise ValueError(f"Ambiguous individual-treatment draw for {key}.")
        draw_names[key] = name

    # DeliveryCare_Comprehensive is missing from the individual-treatment simulations.
    # Temporarily use DeliveryCare_Basic's DALY results as its placeholder until
    # the missing treatment ID has been simulated.
    draw_names["DeliveryCare_Comprehensive"] = draw_names["DeliveryCare_Basic"]

    matched = {}
    for name, members in packages.items():
        missing = members - draw_names.keys()
        if missing:
            raise ValueError(f"{name} is missing individual-treatment results for {sorted(missing)}.")
        matched[name] = [draw_names[member] for member in sorted(members)]
    return matched


def common_run_ids(results_folder: Path, packages: dict, baseline: dict, interventions: dict, totals: dict) -> set[int]:
    available = {f"Nothing/{key}": set(baseline[key].columns) for key in BASELINE_WIDE_KEYS}
    for key in BASELINE_COST_KEYS:
        available[f"Nothing/{key}"] = set(baseline[key]["run"])
    for draw, name in enumerate(packages):
        available[name] = {
            int(path.name)
            for path in (results_folder / str(draw)).iterdir()
            if path.is_dir() and path.name.isdecimal()
            and all((path / f"{module}.pickle").is_file() for module in (
                "tlo.methods.demography", "tlo.methods.healthburden", "tlo.methods.healthsystem.summary",
            ))
        }
    for draw in sorted({draw for members in interventions.values() for draw in members}):
        available[draw] = set(totals["accrued"].xs(draw).dropna().index) & set(
            totals["averted"].xs(draw).dropna().index
        )

    common = set.intersection(*available.values())
    if not common:
        raise ValueError("No run IDs are shared by the HBP draws, Nothing baseline and individual interventions.")
    excluded = {name: sorted(runs - common) for name, runs in available.items() if runs - common}
    if excluded:
        print(f"Excluded unmatched run IDs by source: {excluded}")
    print(f"Matched run IDs: {sorted(common)}")
    return common


def additive_dalys(joint: dict[str, pd.Series], individual: dict[str, pd.Series],
                   interventions: dict[str, list[str]], runs: set[int]) -> tuple[dict, pd.DataFrame]:
    """Compare joint and additive discounted outcomes within each matched run."""
    results_by_run = {}
    summaries = {}
    for metric in ("averted", "accrued"):
        additive = {}
        for package, members in interventions.items():
            by_member = individual[metric].unstack("run").loc[members, sorted(runs)]
            additive[package] = by_member.sum(axis=0, min_count=len(members))
        additive = pd.concat(additive, names=["draw", "run"])
        paired_joint = joint[metric].reindex(additive.index)
        for name, values in (
            (f"joint_{metric}", paired_joint),
            (f"additive_{metric}", additive),
            (f"joint_minus_additive_{metric}", paired_joint - additive),
        ):
            results_by_run[name] = values
            summaries[name] = compute_summary_statistics(
                values.to_frame().T, "median"
            ).iloc[0].unstack("stat")
    return results_by_run, pd.concat(summaries, axis=1)



def additive_dalys_by_cause(joint: pd.DataFrame, individual: pd.DataFrame,
                            interventions: dict[str, list[str]], runs: set[int]) -> tuple[dict, dict]:
    """Sum each member's cause effects within runs, including repeated placeholder draws."""
    causes = joint.index.union(individual.index)
    joint = joint.reindex(causes, fill_value=0.0)
    individual = individual.reindex(causes, fill_value=0.0)
    additive = {}
    for package, members in interventions.items():
        columns = pd.MultiIndex.from_product([members, sorted(runs)], names=["draw", "run"])
        selected = individual.loc[:, columns]
        additive[package] = selected.T.groupby(level="run").sum(min_count=len(members)).T
    additive = pd.concat(additive, axis=1, names=["draw", "run"])
    joint = joint.reindex(columns=additive.columns)
    by_run = {
        "joint": joint,
        "additive": additive,
        "joint_minus_additive": joint - additive,
    }
    summaries = {name: compute_summary_statistics(values, "median") for name, values in by_run.items()}
    return by_run, summaries



def additive_incremental_costs(joint: pd.DataFrame, individual: pd.DataFrame,
                               interventions: dict[str, list[str]], runs: set[int]) -> tuple[dict, pd.DataFrame]:
    """Compare discounted incremental costs relative to Nothing within matched runs."""
    additive = {}
    for package, members in interventions.items():
        # Repeated Basic members preserve the temporary Comprehensive placeholder.
        selected = individual.loc[members, sorted(runs)]
        if selected.isna().any().any():
            raise ValueError(f"Missing individual incremental costs for {package} in matched runs.")
        additive[package] = selected.sum(axis=0, min_count=len(members))
    additive = pd.concat(additive, names=["draw", "run"])
    joint = joint.iloc[0].reindex(additive.index)
    if joint.isna().any():
        raise ValueError("Missing joint incremental costs in matched runs.")
    by_run = {"joint": joint, "additive": additive, "joint_minus_additive": joint - additive}
    summary = pd.concat({
        method: compute_summary_statistics(values.to_frame().T, "median").iloc[0].unstack("stat")
        for method, values in by_run.items()
    }, axis=1)
    return by_run, summary


def export_cost_comparison(results: dict, outfile: Path) -> None:
    summary = results["additive_incremental_cost_summary"].copy()
    summary.columns = [
        f"{method}_incremental_cost_usd_{dict(central='median', lower='lower_95_ui', upper='upper_95_ui')[stat]}"
        for method, stat in summary.columns
    ]
    summary.rename_axis("hbp").reset_index().to_csv(outfile, index=False)
    print(f"Saved discounted incremental cost comparison to {outfile}")


def apply(results_folder: Path, treatment_results_file: Path, resourcefilepath: Path,
          target_period: tuple[Date, Date]) -> dict:
    with treatment_results_file.open("rb") as file:
        treatment_results = pickle.load(file)
    if treatment_results.get("analysis_metadata") != analysis_metadata(target_period):
        raise ValueError("Regenerate the treatment-ID export with the same analysis dates and costing conventions.")
    baseline = treatment_results.get("nothing_by_run")
    individual = treatment_results.get("discounted_daly_totals_by_run")
    individual_causes = treatment_results.get("discounted_dalys_averted_by_cause_by_run")
    if individual_causes is None:
        raise ValueError("Regenerate the treatment-ID pickle to export discounted DALYs averted by cause and run.")
    individual_costs = treatment_results.get("overall_cost_difference_by_run")
    if individual_costs is None:
        raise ValueError("Regenerate the treatment-ID pickle to export run-level incremental costs.")
    treatment_hsi_counts = treatment_results.get("counts_of_hsi_by_period")
    del treatment_results
    if baseline is None or individual is None:
        raise ValueError("The treatment-ID pickle must contain Nothing and individual discounted DALYs by run.")
    missing = set((*BASELINE_WIDE_KEYS, *BASELINE_COST_KEYS, "daly_discount_base_year")) - baseline.keys()
    if missing:
        raise ValueError(f"Regenerate the treatment-ID export. The Nothing section is missing {sorted(missing)}.")

    packages = read_packages(results_folder)
    interventions = match_interventions(packages, individual)
    runs = common_run_ids(results_folder, packages, baseline, interventions, individual)
    results = process_scenarios(
        results_folder, resourcefilepath, target_period, tuple(packages),
        nothing_by_run=baseline, runs=runs,
    )
    by_run, summary = additive_dalys(results["discounted_daly_totals_by_run"], individual, interventions, runs)
    cause_by_run, cause_summary = additive_dalys_by_cause(
        results["discounted_dalys_averted_by_cause_by_run"], individual_causes, interventions, runs
    )
    pd.testing.assert_series_equal(
        cause_by_run["additive"].sum().sort_index(),
        by_run["additive_averted"].sort_index(), check_names=False, rtol=1e-10, atol=1e-6,
    )
    cost_by_run, cost_summary = additive_incremental_costs(
        results["incremental_cost_by_run"], individual_costs, interventions, runs
    )
    universe = get_filtered_treatment_ids(depth=None)
    universe = ["_".join(s.split("_")[:-1]) for s in universe]
    universe = [x for x in universe if x not in EXCLUDED_HSIS]
    ## Add treatment-ids that are not discovered by get_filtered_treatment_ids
    ## See Issue 1929
    universe = universe + ['Alri_Pneumonia_Treatment_Inpatient', 'Alri_Pneumonia_Treatment_Inpatient_Followup', 'DeliveryCare_Comprehensive']
    universe = set(universe)
    overlap = overlap_and_fisher_test(packages, universe)
    results.update(
        hbp_draw_labels=tuple(packages),
        hbp_packages=packages,
        hbp_overlap=overlap,
        matched_runs=sorted(runs),
        additive_dalys_by_run=by_run,
        additive_dalys_summary=summary,
        additive_incremental_cost_by_run=cost_by_run,
        additive_incremental_cost_summary=cost_summary,
        additive_dalys_by_cause_by_run=cause_by_run,
        additive_dalys_by_cause_summary=cause_summary,
        dalys_and_costs_from_lcoa=None,
    )
    return results

# python src/scripts/lcoa_inputs_from_tlo_analyses/analysis_effect_of_hbps.py \
# outputs/s.bhatia@imperial.ac.uk/effect_of_each_hbp-2026-11-09T160813Z \
# outputs/generated_outputs/2040-12-31_fullresults.pkl \
# outputs/generated_outputs \
# --target-start=2026-01-01 \
# --target-end=2040-12-31

def export_daly_comparison(results: dict, outfile: Path) -> None:
    """Export discounted DALYs averted in whole DALYs, with run-wise difference intervals."""
    columns = {}
    for method in ("joint", "additive", "joint_minus_additive"):
        summary = results["additive_dalys_by_cause_summary"][method]
        total = results["additive_dalys_summary"][f"{method}_averted"]
        for stat, suffix in (("central", "median"), ("lower", "lower_95_ui"), ("upper", "upper_95_ui")):
            causes = summary.xs(stat, axis=1, level="stat").T
            causes["All causes"] = total[stat]
            causes.index.name = "hbp"
            causes.columns.name = "cause"
            columns[f"{method}_dalys_averted_{suffix}"] = causes.stack()
    pd.DataFrame(columns).reset_index().to_csv(outfile, index=False)
    print(f"Saved discounted DALY comparison to {outfile}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_folder", type=Path, help="HBP simulation batch with its draw manifest")
    parser.add_argument("treatment_results_file", type=Path, help="Treatment-ID analysis pickle for the same dates")
    parser.add_argument("output_folder", type=Path, help="Directory for the HBP analysis pickle")
    parser.add_argument("--target-start", type=parse_iso_date, required=True)
    parser.add_argument("--target-end", type=parse_iso_date, required=True)
    args = parser.parse_args()
    if not args.target_start < args.target_end or not args.target_start.year <= YEAR_OF_SWITCH <= args.target_end.year:
        parser.error(f"Use increasing target dates that include {YEAR_OF_SWITCH}.")
    results = apply(
        args.results_folder, args.treatment_results_file, Path("resources"),
        (args.target_start, args.target_end),
    )
    args.output_folder.mkdir(parents=True, exist_ok=True)
    outfile = args.output_folder / f"{args.target_end:%Y-%m-%d}_hbp_fullresults.pkl"
    with outfile.open("wb") as file:
        pickle.dump(results, file)
    print(f"Saved HBP analysis to {outfile}")
    export_cost_comparison(
        results, args.output_folder / f"{args.target_end:%Y-%m-%d}_hbp_incremental_cost_comparison.csv"
    )
    export_daly_comparison(
        results, args.output_folder / f"{args.target_end:%Y-%m-%d}_hbp_dalys_averted_comparison.csv"
    )


if __name__ == "__main__":
    main()
