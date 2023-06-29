import argparse
from datetime import datetime
import os
from pathlib import Path
import warnings

from pyinstrument import Profiler
from pyinstrument.renderers import HTMLRenderer, JSONRenderer

from _paths import PROFILING_HTML, PROFILING_JSON
from _parameters import scale_run_parameters
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
    html_dir: Path = PROFILING_HTML,
    json_dir: Path = PROFILING_JSON,
    output_name: Path = None,
    write_html: bool = True,
    write_json: bool = True,
) -> None:
    # Suppress "ignore" warnings
    warnings.filterwarnings("ignore")

    # Grab the timestamp that this function started running at,
    # this will be recorded in metadata and used in default output names
    timestamp = current_time()

    # Setup output directory(ies) if not present
    for dir in [html_dir, json_dir]:
        if not os.path.exists(dir):
            os.mkdir(dir)
    # Assign output filenames
    if output_name is None:
        output_html_file = html_dir / f"{timestamp}.html"
        output_json_file = json_dir / f"{timestamp}.json"
    else:
        output_html_file = html_dir / f"{output_name.stem}.html"
        output_json_file = json_dir / f"{output_name.stem}.json"

    # Create the profiler to record the stack
    # An instance of a Profiler can be start()-ed and stop()-ped multiple times, combining the recorded sessions into one at the end.
    # As such, the same profiler can be used to record the profile of multiple scripts, however this may create large datafiles so using separate profilers is preferable
    p = Profiler(interval=1e-3)

    print(f"[{current_time('%H:%M:%S')}:INFO] Starting profiling runs")

    # Profile scale_run
    scale_run(**scale_run_parameters, profiler=p)

    print(f"[{current_time('%H:%M:%S')}:INFO] Profiling runs complete")

    # Fetch the recorded session: if multiple scripts are to be profiled, this needs to be done after each model "run", and p needs to be re-initialised before starting the next model run.
    scale_run_session = p.last_session

    # Write outputs to files
    # Renderer initialisation options:
    # show_all: removes library calls where identifiable
    # timeline: if true, samples are left in chronological order rather than total time
    if write_html:
        html_renderer = HTMLRenderer(show_all=False, timeline=False)
        print(f"Writing output to: {output_html_file}", end="...", flush=True)
        with open(output_html_file, "w") as f:
            f.write(html_renderer.render(scale_run_session))
        print("done")
    if write_json:
        json_renderer = JSONRenderer(show_all=False, timeline=False)
        print(f"Writing output to: {output_json_file}", end="...", flush=True)
        with open(output_json_file, "w") as f:
            f.write(json_renderer.render(scale_run_session))
        print("done")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=HELP_STR)
    parser.add_argument(
        "--html", action="store_true", help="Write HTML output.", dest="write_html"
    )
    parser.add_argument(
        "--json", action="store_true", help="Write JSON output.", dest="write_json"
    )
    parser.add_argument(
        "--html_dir",
        type=Path,
        help="Redirect the HTML output to this directory.",
        default=PROFILING_HTML,
    )
    parser.add_argument(
        "--json_dir",
        type=Path,
        help="Redirect the JSON output to this directory.",
        default=PROFILING_JSON,
    )
    parser.add_argument(
        "--output_name",
        type=Path,
        help="Name to give to the output file(s). File extensions will automatically appended.",
        default=None,
    )

    args = parser.parse_args()
    run_profiling(**vars(args))
