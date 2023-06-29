import argparse
import json
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Type

import pandas as pd

if TYPE_CHECKING:
    from pyinstrument import Profiler

from _paths import TLO_OUTPUT_DIR, TLO_ROOT
from shared import print_checksum, schedule_profile_log

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file as parse_log_file_fn  # avoid name conflicts
from tlo.methods.fullmodel import fullmodel

HELP_STR = (
    "A run of the full model at scale using all disease modules considered complete and all"
    "modules for birth / labour / newborn outcome.\n"
    "Simulation parameters can be set using command line arguments - run with --help option"
    "for more details.\n"
    "By default a 20 year simulation is run with an initial population size"
    "of 20k and with logged events at level WARNING or above recorded.\n"
    "For use in profiling."
)


def scale_run(
    years: int = 20,
    months: int = 0,
    initial_population: int = 50000,
    tlo_dir: Path = TLO_ROOT,
    output_dir: Path = TLO_OUTPUT_DIR,
    log_filename: str = "scale_run_profiling",
    log_level: Literal["CRITICAL", "DEBUG", "FATAL", "WARNING", "INFO"] = "WARNING",
    parse_log_file: bool = False,
    show_progress_bar: bool = False,
    seed: int = 0,
    disable_health_system: bool = False,
    disable_spurious_symptoms: bool = False,
    capabilities_coefficient: float = None,
    mode_appt_constraints: Literal[0, 1, 2] = 2,
    save_final_population: bool = False,
    record_hsi_event_details: bool = False,
    profiler: Optional[Type["Profiler"]] = None,
) -> None:
    # Start profiler if one has been passed
    if profiler is not None:
        profiler.start()

    # Simulation period
    start_date = Date(2010, 1, 1)
    end_date = start_date + pd.DateOffset(years=years, months=months)

    # The resource files
    resourcefilepath = Path(tlo_dir / "resources")

    log_config = {
        "filename": log_filename,
        "directory": output_dir,
        "custom_levels": {"*": getattr(logging, log_level)},
    }

    sim = Simulation(
        start_date=start_date,
        seed=seed,
        log_config=log_config,
        show_progress_bar=show_progress_bar,
    )

    # Register the appropriate modules with the arguments passed through
    sim.register(
        *fullmodel(
            resourcefilepath=resourcefilepath,
            use_simplified_births=False,
            module_kwargs={
                "HealthSystem": {
                    "disable": disable_health_system,
                    "mode_appt_constraints": mode_appt_constraints,
                    "capabilities_coefficient": capabilities_coefficient,
                    "hsi_event_count_log_period": "simulation"
                    if record_hsi_event_details
                    else None,
                },
                "SymptomManager": {"spurious_symptoms": not disable_spurious_symptoms},
            },
        )
    )

    # Run the simulation
    sim.make_initial_population(n=initial_population)
    schedule_profile_log(sim)
    sim.simulate(end_date=end_date)
    print_checksum(sim)

    if save_final_population:
        sim.population.props.to_pickle(output_dir / "final_population.pkl")

    if parse_log_file:
        parse_log_file_fn(sim.log_filepath)

    if record_hsi_event_details:
        with open(output_dir / "hsi_event_details.json", "w") as json_file:
            json.dump(
                [
                    event_details._asdict()
                    for event_details in sim.modules[
                        "HealthSystem"
                    ].hsi_event_counts.keys()
                ],
                json_file,
            )

    # Stop profiling session
    if profiler is not None:
        profiler.stop()
    return


if __name__ == "__main__":
    # Parse arguments defining run options
    parser = argparse.ArgumentParser(description=HELP_STR)
    parser.add_argument(
        "--years",
        type=int,
        help="Number of years to simulate for (plus any months specified by --months)",
        default=20,
    )
    parser.add_argument(
        "--months",
        type=int,
        help="Number of months to simulate for (plus any years specified by --years)",
        default=0,
    )
    parser.add_argument(
        "--initial-population", type=int, help="Initial population size", default=50000
    )
    parser.add_argument(
        "--tlo-dir", type=Path, help="Root TLOmodel directory", default=TLO_ROOT
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to write output to",
        default=TLO_OUTPUT_DIR,
    )
    parser.add_argument(
        "--log-filename",
        type=str,
        help="Filename to use for log",
        default="for_profiling",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        help="Level to log at",
        choices=("CRITICAL", "DEBUG", "FATAL", "WARNING", "INFO"),
        default="WARNING",
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
        action="store_true",
    )
    parser.add_argument(
        "--disable-spurious-symptoms",
        help="Disable the generation of spurious symptoms in SymptomManager",
        action="store_true",
    )
    parser.add_argument(
        "--capabilities-coefficient",
        help=(
            "Capabilities coefficient to use in HealthSystem. If not specified the ratio of"
            " the initial population to the estimated 2010 population will be used."
        ),
        type=float,
        default=None,
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
        action="store_true",
    )
    parser.add_argument(
        "--save-args-json",
        help="Save the parsed arguments to a JSON file",
        action="store_true",
    )
    parser.add_argument(
        "--save-final-population",
        help="Save the final population dataframe to a pickle file",
        action="store_true",
    )
    parser.add_argument(
        "--record-hsi-event-details",
        help=(
            "Keep a record of set of non-target specific details of HSI events that are "
            "run and output to a JSON file 'hsi_event_details.json' in output directory."
        ),
        action="store_true",
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

    inputs = vars(args)
    inputs.pop("ignore_warnings")
    inputs.pop("save_args_json")

    scale_run(**inputs)
