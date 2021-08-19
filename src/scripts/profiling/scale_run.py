"""
A run of the full model at scale using all disease modules considered complete and all
modules for birth / labour / newborn outcome.

Simulation parameters can be set using command line arguments - run with --help option
for more details. By default a 20 year simulation is run with an initial population size
of 20k and with logged events at level WARNING or above recorded.

For use in profiling.
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import pandas as pd
import shared

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    cardio_metabolic_disorders,
    care_of_women_during_pregnancy,
    contraception,
    demography,
    depression,
    diarrhoea,
    dx_algorithm_adult,
    dx_algorithm_child,
    enhanced_lifestyle,
    epi,
    epilepsy,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    malaria,
    newborn_outcomes,
    oesophagealcancer,
    other_adult_cancers,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager,
)

# Parse arguments defining run options
parser = argparse.ArgumentParser(description="Run model at scale")
parser.add_argument(
    "--years",
    type=int,
    help="Number of years to simulate for (plus any months specified by --months)",
    default=20
)
parser.add_argument(
    "--months",
    type=int,
    help="Number of months to simulate for (plus any years specified by --years)",
    default=0
)
parser.add_argument(
    "--initial-population",
    type=int,
    help="Initial population size",
    default=20000
)
parser.add_argument(
    "--tlo-dir",
    type=Path,
    help="Root TLOmodel directory",
    default="."
)
parser.add_argument(
    "--output-dir",
    type=Path,
    help="Directory to write output to",
    default="./outputs"
)
parser.add_argument(
    "--log-filename", type=str, help="Filename to use for log", default="for_profiling"
)
parser.add_argument(
    "--log-level",
    type=str,
    help="Level to log at",
    choices=("CRITICAL", "DEBUG", "FATAL", "WARNING", "INFO"),
    default="WARNING"
)
parser.add_argument(
    "--parse-log-file",
    help=(
        "Parse log file to create log dataframe at end of simulation (only useful with "
        "interactive -i runs)"
    ),
    action="store_true",
)
parser.add_argument(
    "--show-progress-bar",
    help="Show progress bar during simulation rather than log output",
    action="store_true",
)
parser.add_argument(
    "--seed",
    help="Seed for base pseudo-random number generator",
    type=int,
    default=0,
)
parser.add_argument(
    "--disable-health-system",
    help=(
        "Disable health system - i.e. no processing happens by the health system but "
        "all HSI Events run"
    ),
    action="store_true"
)
parser.add_argument(
    "--disable-spurious-symptoms",
    help="Disable the generation of spurious symptoms in SymptomManager",
    action="store_true"
)
parser.add_argument(
    "--capabilities-coefficient",
    help="Capabilities coefficient to use in HealthSystem",
    type=float,
    default=0.01,
)
parser.add_argument(
    "--mode-appt-constraints",
    help=(
        "Mode of constraints to use in HealthSystem (0: no constraints - all events "
        "run with no squeeze factor, 1: elastic, all events run with squeeze factor, "
        "2: hard, only events with no squeeze factor run"
    ),
    choices=(0, 1, 2),
    type=int,
    default=2,
)
parser.add_argument(
    "--ignore-warnings",
    help=(
        "Ignore any warnings (prevents warning messages being printed). Useful when "
        "combined with --show-progress-bar to avoid disruption of progress bar display"
    ),
    action="store_true"
)
parser.add_argument(
    "--save-args-json",
    help="Save the parsed arguments to a JSON file",
    action="store_true"
)
parser.add_argument(
    "--save-final-population",
    help="Save the final population dataframe to a pickle file",
    action="store_true"
)
parser.add_argument(
    "--record-hsi-event-details",
    help=(
        "Keep a record of set of non-target specific details of HSI events that are "
        "run and output to a JSON file 'hsi_event_details.json' in output directory."
    ),
    action="store_true"
)
args = parser.parse_args()

if args.ignore_warnings:
    warnings.filterwarnings("ignore")

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

if args.save_args_json:
    # Save arguments to a JSON file
    with open(args.output_dir / "arguments.json", "w") as f:
        args_dict = {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        }
        json.dump(args_dict, f, indent=4)

# Simulation period
start_date = Date(2010, 1, 1)
end_date = start_date + pd.DateOffset(years=args.years, months=args.months)

# The resource files
resourcefilepath = Path(args.tlo_dir / "resources")

log_config = {
    "filename": args.log_filename,
    "directory": args.output_dir,
    "custom_levels": {"*": getattr(logging, args.log_level)}
}

sim = Simulation(
    start_date=start_date,
    seed=args.seed,
    log_config=log_config,
    show_progress_bar=args.show_progress_bar
)

# Register the appropriate modules
sim.register(
    # Standard modules:
    demography.Demography(resourcefilepath=resourcefilepath),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
    symptommanager.SymptomManager(
        resourcefilepath=resourcefilepath,
        spurious_symptoms=not args.disable_spurious_symptoms,
    ),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
    healthburden.HealthBurden(resourcefilepath=resourcefilepath),

    # HealthSystem
    healthsystem.HealthSystem(
        resourcefilepath=resourcefilepath,
        disable=args.disable_health_system,
        mode_appt_constraints=args.mode_appt_constraints,
        capabilities_coefficient=args.capabilities_coefficient,
        record_hsi_event_details=args.record_hsi_event_details
    ),
    dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
    dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath),

    # Modules for birth/labour/newborns
    contraception.Contraception(resourcefilepath=resourcefilepath),
    pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
    care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(
        resourcefilepath=resourcefilepath),
    labour.Labour(resourcefilepath=resourcefilepath),
    newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
    postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),

    # Disease modules considered complete:
    cardio_metabolic_disorders.CardioMetabolicDisorders(
        resourcefilepath=resourcefilepath),
    depression.Depression(resourcefilepath=resourcefilepath),
    diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath),
    epi.Epi(resourcefilepath=resourcefilepath),
    epilepsy.Epilepsy(resourcefilepath=resourcefilepath),
    hiv.Hiv(resourcefilepath=resourcefilepath),
    malaria.Malaria(resourcefilepath=resourcefilepath),
    oesophagealcancer.OesophagealCancer(resourcefilepath=resourcefilepath),
    other_adult_cancers.OtherAdultCancer(resourcefilepath=resourcefilepath)
)

# Run the simulation
sim.make_initial_population(n=args.initial_population)
shared.schedule_profile_log(sim)
sim.simulate(end_date=end_date)
shared.print_checksum(sim)

if args.save_final_population:
    sim.population.props.to_pickle(args.output_dir / "final_population.pkl")

if args.parse_log_file:
    log_df = parse_log_file(sim.log_filepath)

if args.record_hsi_event_details:
    with open(args.output_dir / "hsi_event_details.json", "w") as f:
        json.dump(list(sim.modules['HealthSystem'].hsi_event_details), f)
