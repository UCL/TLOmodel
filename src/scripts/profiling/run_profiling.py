import argparse
import os
import warnings
from datetime import datetime
from pathlib import Path

from _parameters import scale_run_parameters
from _paths import PROFILING_RESULTS
from pyinstrument import Profiler
from pyinstrument.renderers import HTMLRenderer, JSONRenderer
from scale_run import scale_run

HELP_STR = (
    "Produces profiling runs for a selection of models and parameters,\n"
    "writing the results in HTML and/or JSON format.\n"
    "Output names will default to the profiling timestamp if not provided."
)


def current_time(formatstr: str = "%Y-%m-%d_%H%M") -> str:
    """Produces a string of the current time in the specified format."""
    return datetime.utcnow().strftime(formatstr)


def run_profiling(
    output_dir: Path = PROFILING_RESULTS,
    output_name: Path = None,
    write_pyis: bool = True,
    write_html: bool = False,
    write_json: bool = False,
    interval: float = 1e-1,
) -> None:
    # Suppress "ignore" warnings
    warnings.filterwarnings("ignore")

    # Create the directory that this profiling run will live in
    output_dir = output_dir / current_time("%Y/%m/%d/%H%M")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Assign output filenames
    if output_name is None:
        output_pyis_file = output_dir / "output.pyisession"
        output_html_file = output_dir / "output.html"
        output_json_file = output_dir / "output.json"
    else:
        output_pyis_file = output_dir / f"{output_name.stem}.pyisession"
        output_html_file = output_dir / f"{output_name.stem}.html"
        output_json_file = output_dir / f"{output_name.stem}.json"

    # Create the profiler to record the stack
    # An instance of a Profiler can be start()-ed and stop()-ped multiple times,
    # combining the recorded sessions into one at the end.
    # As such, the same profiler can be used to record the profile of multiple scripts,
    # however this may create large datafiles so using separate profilers is preferable
    p = Profiler(interval=interval)

    print(f"[{current_time('%H:%M:%S')}:INFO] Starting profiling runs")

    # Profile scale_run
    scale_run(**scale_run_parameters, profiler=p)

    print(f"[{current_time('%H:%M:%S')}:INFO] Profiling runs complete")

    # Fetch the recorded session: if multiple scripts are to be profiled,
    # this needs to be done after each model "run",
    # and p needs to be re-initialised before starting the next model run.
    scale_run_session = p.last_session

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
