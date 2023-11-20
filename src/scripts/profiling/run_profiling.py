import argparse
import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from psutil import disk_io_counters
from pyinstrument import Profiler
from pyinstrument.renderers import HTMLRenderer
from pyinstrument.session import Session
from .scale_run import save_arguments_to_json, scale_run

from tlo import Simulation


_TLO_ROOT: Path = Path(__file__).parents[3].resolve()
_PROFILING_RESULTS: Path = (_TLO_ROOT / "profiling_results").resolve()


def current_time(format_str: str = "%Y-%m-%d_%H%M") -> str:
    """Produces a string of the current time in the specified format."""
    return datetime.utcnow().strftime(format_str)


def parse_keyword_args(items: List[str] = None, sep: str = "=") -> Dict[str, str]:
    """Parse a series of key-value pairs from the command-line into a dictionary.

    Input is a list of strings of the format
    KEY{sep}VALUE,
    which will be parsed into a key-value pair as
    "KEY" : "VALUE".

    Note that keys and values are always interpreted as strings.
    """
    if items is None:
        return dict()
    else:
        d = dict()
        for item in items:
            separated_string = item.split(sep)
            key = separated_string[0].strip()
            if len(separated_string) > 1:
                # rejoin the remaining values provided
                value = sep.join(separated_string[1:])
            d[key] = value
    return d


def record_simulation_statistics(s: Simulation) -> Dict[str, Union[int, float]]:
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
    if Simulation is None:
        return dict()

    # Population DataFrame statistics
    pops = s.population
    pop_stats = {
        "pop_df_rows": pops.props.shape[0],
        "pop_df_cols": pops.props.shape[1],
        "pop_df_mem_MB": pops.props.memory_usage(index=True, deep=True).sum() / 1e6,
        "pop_df_times_extended": int(
            np.ceil((pops.props.shape[0] - pops.initial_size) / pops.new_rows.shape[0])
        ),
    }

    return pop_stats


def record_disk_statistics(
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
    if disk_usage is None:
        return {}

    return {
        "disk_reads": disk_usage["read_count"],
        "disk_writes": disk_usage["write_count"],
        "disk_read_MB": disk_usage["read_bytes"] / 1e6,
        "disk_write_MB": disk_usage["write_bytes"] / 1e6,
        "disk_read_s": disk_usage["read_time"] / 1e3,
        "disk_write_s": disk_usage["write_time"] / 1e3,
    }


def record_profiling_session_statistics(
    session: Session,
) -> Dict[str, Union[int, float]]:
    """
    Extract important profiling statistics from the session that was captured.
    Statistics are returned as a dictionary.

            "start_time": self.start_time,
            "duration": self.duration,
            "sample_count": self.sample_count,
            "start_call_stack": self.start_call_stack,
            "program": self.program,
            "cpu_time": self.cpu_time,

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
    if session is None:
        return {}

    return {
        "start_time": session.start_time,
        "duration": session.duration,
        "cpu_time": session.cpu_time,
    }


def record_run_statistics(
    output_file: str,
    html_output_file: str = None,
    profiling_session: Session = None,
    completed_sim: Simulation = None,
    disk_usage: Dict[str, Union[int, float]] = None,
    **additional_stats: str,
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
    # Record statistics as [key, value] pairs
    stats_dict = dict()
    # Start with user-defined additional stats, if they exist
    if additional_stats is not None:
        stats_dict.update(additional_stats)

    # If we wrote a HTML output, include this in the additional statistics to write.
    # Warn the user if they have overwritten the reserved keyword,
    # then overwrite the value provided anyway.
    if html_output_file is not None:
        if "html_output" in additional_stats.keys():
            warnings.warn(
                f"User-provided statistic for 'html_output' was provided: "
                "this is being overwritten with the path to the HTML output.\n"
                f"\tWas        : {additional_stats['html_output']}"
                f"\tReplaced by: {html_output_file}"
            )
        stats_dict["html_output"] = html_output_file

    # Fetch statistics from the profiled session itself
    stats_dict.update(record_profiling_session_statistics(profiling_session))

    # Fetch statistics from end end-state of the simulation
    stats_dict.update(record_simulation_statistics(completed_sim))

    # Fetch disk I/O statistics
    stats_dict.update(record_disk_statistics(disk_usage))

    # Having computed all statistics, save the file
    with open(output_file, "w") as f:
        json.dump(stats_dict, f, indent=2)
    return


def run_profiling(
    root_output_dir: Path = _PROFILING_RESULTS,
    output_name: str = "profiling",
    write_html: bool = False,
    interval: float = 1e-1,
    **additional_stats: str,
) -> None:
    """
    Uses pyinstrument to profile the scale_run simulation,
    writing the output in the requested formats.
    """

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

    scale_run_parameters = {
        "years": 0,
        "months": 1,
        "initial_population": 50000,
        "log_filename": "scale_run_profiling",
        "log_level": "DEBUG",
        "parse_log_file": False,
        "show_progress_bar": True,
        "seed": 0,
        "disable_health_system": False,
        "disable_spurious_symptoms": False,
        "capabilities_coefficient": None,
        "mode_appt_constraints": 0,
        "save_final_population": False,
        "record_hsi_event_details": False,
        "ignore_warnings": True,
    }

    save_arguments_to_json(scale_run_parameters, output_dir / "parameters.json")

    print(f"[{current_time('%H:%M:%S')}:INFO] Starting profiling runs")

    # Profile scale_run
    disk_at_start = disk_io_counters()
    completed_simulation = scale_run(
        **scale_run_parameters, output_dir=output_dir, profiler=profiler
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
        html_renderer = HTMLRenderer(show_all=False, timeline=False)
        print(f"Writing {output_html_file}", end="...", flush=True)
        with open(output_html_file, "w") as f:
            f.write(html_renderer.render(scale_run_session))
        print("done")

    # Write the statistics file, main output
    output_stat_file = output_dir / f"{output_name}.stats.json"
    print(f"Writing {output_stat_file}", end="...", flush=True)
    record_run_statistics(
        output_stat_file,
        html_output_file=str(output_html_file.name),
        profiling_session=scale_run_session,
        completed_sim=completed_simulation,
        disk_usage=disk_usage,
        **additional_stats,
    )
    print("done")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Produces profiling runs for a selection of models and parameters, "
            "writing the results in HTML and/or JSON format. "
            "Output names will default to the profiling timestamp if not provided."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Redirect the output(s) to this directory.",
        default=_PROFILING_RESULTS,
    )
    parser.add_argument(
        "--output-name",
        type=str,
        help="Name to give to the output file(s). "
        "File extensions will automatically appended.",
        default="profiling",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Write HTML output in addition to statistics output.",
        dest="write_html",
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
        "--additional-stats",
        metavar="KEY=VALUE",
        nargs="*",
        help="Set a number of key-value pairs "
        "(do not put spaces before or after the = sign). "
        "If a value contains spaces, you should define "
        "it with double quotes: "
        'foo="this is a sentence". Note that '
        "values are always treated as strings.",
    )

    args = parser.parse_args()
    # Parse additional run statistics from the command line
    command_line_stats = parse_keyword_args(args.additional_stats)

    # Pass to the profiling "script"
    run_profiling(
        root_output_dir=args.output_dir,
        output_name=args.output_name,
        write_html=args.write_html,
        interval=args.interval,
        **command_line_stats,
    )
