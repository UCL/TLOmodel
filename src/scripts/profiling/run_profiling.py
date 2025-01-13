import argparse
import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from psutil import disk_io_counters
from pyinstrument import Profiler
from pyinstrument.renderers import ConsoleRenderer, HTMLRenderer
from pyinstrument.session import Session
from scale_run import save_arguments_to_json, scale_run
from shared import memory_statistics

try:
    from ansi2html import Ansi2HTMLConverter
    ANSI2HTML_AVAILABLE = True
except ImportError:
    ANSI2HTML_AVAILABLE = False

from tlo import Simulation

_PROFILING_RESULTS: Path = (Path(__file__).parents[3] / "profiling_results").resolve()


def current_time(format_str: str = "%Y-%m-%d_%H%M") -> str:
    """Produces a string of the current time in the specified format."""
    return datetime.utcnow().strftime(format_str)


def parse_key_value_string(key_value_string: str) -> Tuple[str, str]:
    """Parse a key-value pair from the command-line into a tuple.

    Input is a string of the format::

        "{key}={value}"

    which will be parsed into a key-value tuple as::

        ("{key}", "{value}")

    Note that the key and value are always interpreted as strings.
    """
    sep = "="
    key, *value = key_value_string.split(sep)
    if len(value) == 0:
        msg = "Key-value pair should be in format {key}={value} with no spaces."
        raise argparse.ArgumentTypeError(msg)
    key = key.strip()
    # Allow for separator string appearing in value by rejoining
    value = sep.join(value)
    return (key, value)


def simulation_statistics(
    simulation: Simulation,
) -> Dict[str, Union[int, float]]:
    """
    Extract variables from the completed modelling simulation, to include in the
    profiling run report.
    Statistics are returned as a dictionary.

    Key / value pairs are:
    pop_df_rows: int
        Number of rows in the final population DataFrame
    pop_df_cols: int
        Number of cols in the final population DataFrame
    pop_df_mem_MB: float
        Size in MBs of the final population DataFrame
    pop_df_times_extended: int
        Number of times the population DataFrame had to be expanded
    """

    # Population DataFrame statistics
    population_dataframe = simulation.population.props
    return {
        "pop_df_rows": population_dataframe.shape[0],
        "pop_df_cols": population_dataframe.shape[1],
        "pop_df_mem_MB": population_dataframe.memory_usage(index=True, deep=True).sum()
        / 1e6,
        "pop_df_times_extended": int(
            np.ceil(
                (population_dataframe.shape[0] - simulation.population.initial_size)
                / simulation.population.new_rows.shape[0]
            )
        ),
    }


def disk_statistics(
    disk_usage: Dict[str, Union[int, float]]
) -> Dict[str, Union[int, float]]:
    """
    Extract disk I/O statistics from the profiled run.
    Statistics are returned as a dictionary.

    Key / value pairs are:
    disk_reads: int
        Number of times the disk was read during the simulation
    disk_writes: int
        Number of times the disk was written to during the simulation
    disk_read_MB: float
        Memory read in MBs from the disk during simulation
    disk_write_MB: float
        Memory written in MBs from the disk during simulation
    disk_read_s: float
        Time in seconds spent reading from disk during simulation
    disk_write_s: float
        Time in seconds spent writing to disk during simulation
    """
    return {
        "disk_reads": disk_usage["read_count"],
        "disk_writes": disk_usage["write_count"],
        "disk_read_MB": disk_usage["read_bytes"] / 1e6,
        "disk_write_MB": disk_usage["write_bytes"] / 1e6,
        "disk_read_s": disk_usage["read_time"] / 1e3,
        "disk_write_s": disk_usage["write_time"] / 1e3,
    }


def profiling_session_statistics(
    session: Session,
) -> Dict[str, float]:
    """
    Extract important profiling statistics from the session that was captured.
    Statistics are returned as a dictionary.

    Key / value pairs are:
    start_time: float
        Time (stored as a float representing the number of seconds from a reference time)
        the profiling session started.
    duration: float
        Number of seconds that the profiling session lasted for.
    cpu_time: float
        Number of seconds of CPU time that were used by the program during the profiling
        session.
    """
    return {
        "start_time": session.start_time,
        "duration": session.duration,
        "cpu_time": session.cpu_time,
    }


def record_run_statistics(
    output_file: str,
    profiling_session: Session,
    completed_sim: Simulation,
    disk_usage: Dict[str, Union[int, float]],
    additional_stats: Dict[str, str],
) -> None:
    """
    Organise all statistics to be collected from the profiling run into a single dict,
    which is then be dumped to a JSON file.

    :param output_file: JSON file / path to write to.
    :param html_output_file: The name of the output HTML file from the profiling run,
     if it was produced.
    :param profiled_session: The Session object representing the profiling session.
    :param completed_sim: The end-state of the simulation.
    :param disk_usage: Usage stats for the disk I/O operations during the profiling run.
    :param additional_stats: Dict of any additional information passed by the user that
     should be recorded.
    """
    stats_dict = {
        # Statistics from the profiled session itself
        **profiling_session_statistics(profiling_session),
        # Disk input/output statistics
        **disk_statistics(disk_usage),
        # Process memory statistics
        **memory_statistics(),
        # Statistics from end end-state of the simulation
        **simulation_statistics(completed_sim),
        # User-defined additional stats (if any)
        **additional_stats,
    }
    with open(output_file, "w") as f:
        json.dump(stats_dict, f, indent=2)


def run_profiling(
    root_output_dir: Path = _PROFILING_RESULTS,
    output_name: str = "profiling",
    write_html: bool = False,
    write_pyisession: bool = False,
    write_flat_html: bool = True,
    interval: float = 2e-1,
    initial_population: int = 50000,
    simulation_years: int = 5,
    simulation_months: int = 0,
    mode_appt_constraints: Literal[0, 1, 2] = 2,
    additional_stats: Optional[List[Tuple[str, str]]] = None,
    show_progress_bar: bool = False,
    disable_log_output_to_stdout: bool = False,
) -> None:
    """
    Uses pyinstrument to profile the scale_run simulation,
    writing the output in the requested formats.
    """
    if write_flat_html and not ANSI2HTML_AVAILABLE:
        # Check if flat HTML output requested but ansi2html module not available at
        # _start_ of function to avoid erroring after a potentially long profiling run
        msg = "ansi2html required for flat HTML output."
        raise ValueError(msg)

    additional_stats = dict(() if additional_stats is None else additional_stats)

    # Create the profiler to record the stack
    # An instance of a Profiler can be start()-ed and stop()-ped multiple times,
    # combining the recorded sessions into one at the end.
    # As such, the same profiler can be used to record the profile of multiple scripts,
    # however this may create large datafiles so using separate profilers is preferable
    profiler = Profiler(interval=interval)

    # Create the directory that this profiling run will live in
    output_dir = root_output_dir / current_time("%Y/%m/%d/%H%M")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scale_run_args = {
        "years": simulation_years,
        "months": simulation_months,
        "initial_population": initial_population,
        "log_filename": "scale_run_profiling",
        "log_level": "WARNING",
        "parse_log_file": True,
        "show_progress_bar": show_progress_bar,
        "seed": 0,
        "disable_health_system": False,
        "disable_spurious_symptoms": False,
        "capabilities_coefficient": None,
        "mode_appt_constraints": mode_appt_constraints,
        "save_final_population": False,
        "record_hsi_event_details": False,
        "ignore_warnings": True,
        "log_final_population_checksum": False,
        "disable_log_output_to_stdout": disable_log_output_to_stdout,
    }

    output_arg_file = output_dir / f"{output_name}.args.json"
    print(f"Writing {output_arg_file}", end="...", flush=True)
    save_arguments_to_json(scale_run_args, output_arg_file)
    print("done")

    print(f"[{current_time('%H:%M:%S')}:INFO] Starting profiling runs")

    # Profile scale_run
    disk_at_start = disk_io_counters()
    completed_simulation, logs_dict = scale_run(
        **scale_run_args, output_dir=output_dir, profiler=profiler
    )
    disk_at_end = disk_io_counters()

    print(f"[{current_time('%H:%M:%S')}:INFO] Profiling runs complete")

    # Fetch the recorded session: if multiple scripts are to be profiled,
    # this needs to be done after each model "run",
    # and p needs to be re-initialised before starting the next model run.
    scale_run_session = profiler.last_session
    # Infer disk usage statistics
    disk_usage = {
        key: getattr(disk_at_end, key) - getattr(disk_at_start, key)
        for key in disk_at_start._fields
    }

    # Write outputs to files
    # HTML (if requested)
    if write_html:
        output_html_file = output_dir / f"{output_name}.html"
        # Renderer initialisation options:
        # show_all: removes library calls where identifiable
        # timeline: if true, samples are left in chronological order rather than total time
        html_renderer = HTMLRenderer(
            show_all=False,
            timeline=False,
            processor_options={"show_regex": ".*/tlo/.*", "hide_regex": ".*/pandas/.*"}
        )
        print(f"Writing {output_html_file}", end="...", flush=True)
        with open(output_html_file, "w") as f:
            f.write(html_renderer.render(scale_run_session))
        print("done")
        # If we wrote a HTML output, include this in the additional statistics to write.
        # Warn the user if they have overwritten the reserved keyword,
        # then overwrite the value provided anyway.
        if "html_output" in additional_stats:
            warnings.warn(
                f"User-provided statistic for 'html_output' was provided: "
                "this is being overwritten with the path to the HTML output.\n"
                f"\tWas        : {additional_stats['html_output']}"
                f"\tReplaced by: {output_html_file}"
            )
        additional_stats["html_output"] = str(output_html_file.name)

    if write_pyisession:
        output_ipysession_file = output_dir / f"{output_name}.pyisession"
        print(f"Writing {output_ipysession_file}", end="...", flush=True)
        scale_run_session.save(output_ipysession_file)
        print("done")
        
    if write_flat_html:
        output_html_file = output_dir / f"{output_name}.flat.html"
        console_renderer = ConsoleRenderer(
            show_all=False,
            timeline=False,
            color=True,
            flat=True,
            processor_options={"show_regex": ".*/tlo/.*", "hide_regex": ".*/pandas/.*", "filter_threshold": 1e-3}
        )
        converter = Ansi2HTMLConverter(title=output_name)
        print(f"Writing {output_html_file}", end="...", flush=True)
        with open(output_html_file, "w") as f:
            f.write(converter.convert(console_renderer.render(scale_run_session)))
        print("done")
        additional_stats["flat_html_output"] = str(output_html_file.name)

    # Write the statistics file, main output
    output_stat_file = output_dir / f"{output_name}.stats.json"
    print(f"Writing {output_stat_file}", end="...", flush=True)
    record_run_statistics(
        output_stat_file,
        profiling_session=scale_run_session,
        completed_sim=completed_simulation,
        disk_usage=disk_usage,
        additional_stats=additional_stats,
    )
    print("done")
    
    # Write out logged profiling statistics
    logged_statistics_file = output_dir / f"{output_name}.logged-stats.csv"
    print(f"Writing {logged_statistics_file}", end="...", flush=True)
    logs_dict["tlo.profiling"]["stats"].to_csv(logged_statistics_file, index=False)
    print("done")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Produces profiling runs for a selection of models and parameters, "
            "writing the results in HTML and/or JSON format. "
            "Output names will default to the profiling timestamp if not provided."
        )
    )
    parser.add_argument(
        "--root-output-dir",
        type=Path,
        help=(
            "Root directory to write profiling outputs to. Results will be written to "
            "a subdirectory within this directory based on the date-time at which the "
            "profiling run was started, specifically using the format codes from "
            "strftime, within a directory corresponding to 'YY/mm/dd/HHMM'."
        ),
        default=_PROFILING_RESULTS,
    )
    parser.add_argument(
        "--output-name",
        type=str,
        help=(
            "Name to give to the output file(s). "
            "File extensions will be automatically appended."
        ),
        default="profiling",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Write HTML output in addition to statistics output.",
        dest="write_html",
    )
    parser.add_argument(
        "--pyisession",
        help="Write raw profiler pyisession output.",
        action="store_true",
        dest="write_pyisession",
    )
    parser.add_argument(
        "--flat-html",
        action="store_true",
        help="Write flat HTML output in addition to statistics output.",
        dest="write_flat_html",
    )
    parser.add_argument(
        "-i",
        "--interval-seconds",
        dest="interval",
        type=float,
        help="Interval in seconds between capture frames for profiling.",
        default=1e-1,
    )
    parser.add_argument(
        "-y",
        "--simulation-years",
        type=int,
        help="Number of years to simulate for (plus --simulation-months months)",
        default=5,
    )
    parser.add_argument(
        "-m",
        "--simulation-months",
        type=int,
        help="Number of months to simulate for (plus --simulation-years years)",
        default=0,
    )
    parser.add_argument(
        "-p",
        "--initial-population",
        type=int,
        help="Initial population size",
        default=50000,
    )
    parser.add_argument(
        "--mode-appt-constraints",
        help=(
            "Mode of constraints to use in HealthSystem (0: no constraints - all events"
            " run with no squeeze factor, 1: elastic, all events run with squeeze "
            "factor, 2: hard, only events with no squeeze factor run"
        ),
        choices=(0, 1, 2),
        type=int,
        default=2,
    )
    parser.add_argument(
        "--additional-stats",
        metavar="KEY=VALUE",
        type=parse_key_value_string,
        nargs="*",
        default=[],
        help=(
            "Set a number of key-value pairs (do not put spaces before or after the = "
            "sign). If a value contains spaces, you should define it with double "
            'quotes: foo="this is a sentence". Note that values are always treated '
            "as strings."
        ),
    )
    parser.add_argument(
        "--show-progress-bar",
        help="Show simulation progress bar during simulation rather than log output",
        action="store_true",
    )
    parser.add_argument(
        "--disable-log-output-to-stdout",
        help="Disable simulation log output being displayed in stdout stream",
        action="store_true",
    )

    args = parser.parse_args()

    # Pass to the profiling "script"
    run_profiling(**vars(args))
