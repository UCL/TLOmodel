import argparse
from datetime import datetime

import os
from pathlib import Path
import warnings

from pyinstrument import Profiler
from pyinstrument.renderers import HTMLRenderer

from _paths import PROFILING_HTML_DIR
from parameters import scale_run_parameters
from scale_run import scale_run


def current_time() -> str:
    """Produces a string of the current time, in YYYY-mm-dd_HHMM format"""
    return datetime.utcnow().strftime("%Y-%m-%d_%H%M")


def profile_all(output_html_dir: str = None) -> None:
    warnings.filterwarnings("ignore")

    # Setup the output file and directory
    if output_html_dir is None:
        output_html_dir = PROFILING_HTML_DIR
    if not os.path.exists(PROFILING_HTML_DIR):
        os.mkdir(PROFILING_HTML_DIR)
    output_html_file = PROFILING_HTML_DIR / (current_time() + ".html")

    # Setup the profiler, to record the stack every interval seconds
    p = Profiler(interval=1e-3)

    # Perform all profiling runs, passing in the profiler so it can be started within each run and halted between for more accurate results
    scale_run(**scale_run_parameters, profiler=p)

    # Recorded sessions are combined, so last_session should fetch the combination of all profiling runs conducted
    profiled_session = p.last_session

    # Parse results into HTML
    # show_all: removes library calls where identifiable
    # timeline: if true, samples are left in chronological order rather than total time
    html_renderer = HTMLRenderer(show_all=False, timeline=False)

    # Write HTML file
    print(f"Writing output to: {output_html_file}", end="...", flush=True)
    with open(output_html_file, "w") as f:
        f.write(html_renderer.render(profiled_session))
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all profiling scripts and save the results."
    )
    parser.add_argument(
        "output_html_dir",
        nargs="?",
        type=str,
        default=None,
        help="Directory into which to write profiling results as HTML files.",
    )

    args = parser.parse_args()
    profile_all(**vars(args))
