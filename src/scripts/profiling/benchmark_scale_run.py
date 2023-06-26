"""
Benchmarking script for scale_run.py.
"""
import argparse
import datetime
import os
from pathlib import Path
import warnings

from pyinstrument import Profiler
from pyinstrument.renderers import HTMLRenderer

from scale_run import main as sc_run


def main(output_html_fname: str = None) -> None:
    warnings.filterwarnings("ignore")

    if output_html_fname is None:
        output_html_fname = output_dir / (
            datetime.datetime.utcnow().strftime("%Y-%m-%d_%H%M")
            + "_scale_run_profile.html"
        )

    LOCATION_OF_THIS_FILE = os.path.dirname(os.path.abspath(__file__))
    TLO_ROOT = (Path(LOCATION_OF_THIS_FILE) / ".." / ".." / "..").resolve()

    # Decide on the parameters to pass to scale_run
    # p has not been started, so these are not part of the profiling output
    years = 0
    months = 1
    initial_population = 50000
    tlo_dir = TLO_ROOT
    output_dir = (TLO_ROOT / "outputs").resolve()
    log_filename = "scale_run_benchmark"
    log_level = "DEBUG"
    parse_log_file = False
    show_progress_bar = True
    seed = 0
    disable_health_system = False
    disable_spurious_symptoms = False
    capabilities_coefficient = None
    mode_appt_constraints = 2
    save_final_population = False
    record_hsi_event_details = False

    # Setup the profiler, to record the stack every interval seconds
    p = Profiler(interval=1e-3)
    # Start the profiler and perform the run
    # p.start()
    sc_run(
        years,
        months,
        initial_population,
        tlo_dir,
        output_dir,
        log_filename,
        log_level,
        parse_log_file,
        show_progress_bar,
        seed,
        disable_health_system,
        disable_spurious_symptoms,
        capabilities_coefficient,
        mode_appt_constraints,
        save_final_population,
        record_hsi_event_details,
        p,
    )
    # profiled_session = p.stop()
    profiled_session = p.last_session

    # Parse results into HTML
    # show_all: removes library calls where identifiable
    # timeline: if true, samples are left in chronological order rather than total time
    html_renderer = HTMLRenderer(show_all=False, timeline=False)

    # Write HTML file
    print(f"Writing output to: {output_html_fname}", end="...", flush=True)
    with open(output_html_fname, "w") as f:
        f.write(html_renderer.render(profiled_session))
    print("done")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark scale_run.py script")
    parser.add_argument(
        "output-html-fname",
        nargs="?",
        type=str,
        default=None,
        help="Filename for the output HTML file containing the profiling results."
        " Generates a default name using the call timestamp if not set.",
    )

    parser.parse_args()
    main(parser.output_html_fname)
