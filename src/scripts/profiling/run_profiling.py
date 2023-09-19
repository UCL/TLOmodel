import argparse
import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from _parameters import scale_run_parameters
from _paths import PROFILING_RESULTS
from psutil import disk_io_counters
from pyinstrument import Profiler
from pyinstrument.renderers import HTMLRenderer, JSONRenderer
from scale_run import scale_run

from tlo import Simulation

HELP_STR = (
    "Produces profiling runs for a selection of models and parameters,\n"
    "writing the results in HTML and/or JSON format.\n"
    "Output names will default to the profiling timestamp if not provided."
)


def current_time(formatstr: str = "%Y-%m-%d_%H%M") -> str:
    """Produces a string of the current time in the specified format."""
    return datetime.utcnow().strftime(formatstr)


def record_run_statistics(output_file: str, s: Simulation, disk_usage=None) -> None:
    """
    After concluding the profiling session, but before cleanup,
    record statistics from the simulation outputs that we wish
    to present alongside the profiler outputs.

    Key / value pairs are:
    pop_df_rows: int
        Number of rows in the final population DataFrame
    pop_df_cols: int
        Number of cols in the final population DataFrame
    pop_df_mem_mb: float
        Size in MBs of the final population DataFrame
    pop_df_times_extended: int
        Number of times the population DataFrame had to be expanded
    disk_reads: n
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

    :param output_file: File name to write to.
    :param s: The completed simulation to read stats from.
    :param disk_usage: Dictionary with keys corresponding to disk usage during the simulation;
        - read_count:  number of reads
        - write_count: number of writes
        - read_bytes:  number of bytes read
        - write_bytes: number of bytes written
        - read_time:   time spent reading from disk (in ms)
        - write_time:  time spent writing to disk (in ms)
    """
    # Record statistics as [key, value] pairs

    # Population DataFrame statistics
    pops = s.population
    pop_stats = {
        "pop_df_rows": pops.props.shape[0],
        "pop_df_cols": pops.props.shape[1],
        "pop_df_mem_mb": pops.props.memory_usage(index=True, deep=True).sum() / 1e6,
        "pop_df_times_extended": int(
            np.ceil((pops.props.shape[0] - pops.initial_size) / pops.new_rows.shape[0])
        ),
    }

    # Disk I/O statistics
    disk_stats = {
        "disk_reads": disk_usage["read_count"],
        "disk_writes": disk_usage["write_count"],
        "disk_read_MB": disk_usage["read_bytes"] / 1e6,
        "disk_write_MB": disk_usage["write_bytes"] / 1e6,
        "disk_read_s": disk_usage["read_time"] / 1e3,
        "disk_write_s": disk_usage["write_time"] / 1e3,
    }

    # Can combine dictionaries using either
    # {**dict1, **dict2}, or
    # dict1 | dict2 [python >=3.10 only]
    stats_dict = {**pop_stats, **disk_stats}

    # Having computed all statistics, save the file
    with open(output_file, "w") as f:
        json.dump(stats_dict, f, indent=2)
    return


def run_profiling(
    output_dir: Path = PROFILING_RESULTS,
    output_name: Path = None,
    write_pyis: bool = True,
    write_html: bool = False,
    write_json: bool = False,
    write_stats: bool = False,
    interval: float = 1e-1,
) -> None:
    """
    Uses pyinstrumment to profile the scale_run simulation,
    writing the output in the requested formats.
    """
    # Suppress "ignore" warnings
    warnings.filterwarnings("ignore")

    # Create the directory that this profiling run will live in
    output_dir = output_dir / current_time("%Y/%m/%d/%H%M")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Assign output filenames
    output_stem = "output" if output_name is None else output_name.stem
    output_pyis_file = output_dir / f"{output_stem}.pyisession"
    output_html_file = output_dir / f"{output_stem}.html"
    output_json_file = output_dir / f"{output_stem}.json"
    output_stat_file = output_dir / f"{output_stem}.stats.json"

    # Create the profiler to record the stack
    # An instance of a Profiler can be start()-ed and stop()-ped multiple times,
    # combining the recorded sessions into one at the end.
    # As such, the same profiler can be used to record the profile of multiple scripts,
    # however this may create large datafiles so using separate profilers is preferable
    p = Profiler(interval=interval)

    print(f"[{current_time('%H:%M:%S')}:INFO] Starting profiling runs")

    # Profile scale_run
    disk_at_start = disk_io_counters()
    completed_simulation = scale_run(**scale_run_parameters, profiler=p)
    disk_at_end = disk_io_counters()

    print(f"[{current_time('%H:%M:%S')}:INFO] Profiling runs complete")

    # Fetch the recorded session: if multiple scripts are to be profiled,
    # this needs to be done after each model "run",
    # and p needs to be re-initialised before starting the next model run.
    scale_run_session = p.last_session
    # Infer disk usage statistics
    disk_usage = {
        key: disk_at_end[key] - disk_at_start[key] for key in disk_at_start.keys()
    }

    # Write outputs to files
    # Renderer initialisation options:
    # show_all: removes library calls where identifiable
    # timeline: if true, samples are left in chronological order rather than total time
    if write_pyis:
        print(f"Writing {output_html_file}", end="...", flush=True)
        scale_run_session.save(output_pyis_file)
        print("done")
    if write_html:
        html_renderer = HTMLRenderer(show_all=False, timeline=False)
        print(f"Writing {output_html_file}", end="...", flush=True)
        with open(output_html_file, "w") as f:
            f.write(html_renderer.render(scale_run_session))
        print("done")
    if write_json:
        json_renderer = JSONRenderer(show_all=False, timeline=False)
        print(f"Writing {output_json_file}", end="...", flush=True)
        with open(output_json_file, "w") as f:
            f.write(json_renderer.render(scale_run_session))
        print("done")
    if write_stats:
        print(f"Writing {output_stat_file}", end="...", flush=True)
        record_run_statistics(output_stat_file, completed_simulation)
        print("done")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=HELP_STR)
    parser.add_argument(
        "--pyis",
        action="store_true",
        help="Write .ipysession output.",
        dest="write_pyis",
    )
    parser.add_argument(
        "--html", action="store_true", help="Write HTML output.", dest="write_html"
    )
    parser.add_argument(
        "--json", action="store_true", help="Write JSON output.", dest="write_json"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Write additional statistics file as output",
        dest="write_stats",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Redirect the output(s) to this directory.",
        default=PROFILING_RESULTS,
    )
    parser.add_argument(
        "--output_name",
        type=Path,
        help="Name to give to the output file(s). File extensions will automatically appended.",
        default=None,
    )
    parser.add_argument(
        "-i",
        "--interval-seconds",
        dest="interval",
        type=float,
        help="Interval in seconds between capture frames for profiling.",
        default=1e-1,
    )

    args = parser.parse_args()
    run_profiling(**vars(args))
