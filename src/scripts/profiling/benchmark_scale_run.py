"""
Benchmarking script for scale_run.py.
"""
import datetime
import os
from pathlib import Path
import warnings

from pyinstrument import Profiler
from pyinstrument.renderers import HTMLRenderer

from scale_run import main as sc_run


def main() -> None:
    warnings.filterwarnings("ignore")

    LOCATION_OF_THIS_FILE = os.path.dirname(os.path.abspath(__file__))
    TLO_ROOT = (Path(LOCATION_OF_THIS_FILE) / ".." / ".." / "..").resolve()

    # Setup the profiler, to record the stack every interval seconds
    p = Profiler(interval=1e-3)

    # Decide on the parameters to pass to scale_run
    # p has not been started, so these are not part of the profiling output
    years = 0
    months = 1
    initial_population = 50000
    tlo_dir = TLO_ROOT
    output_dir = (TLO_ROOT / "outputs").resolve()
    log_filename = "for_profiling"
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

    # Start the profiler and perform the run
    p.start()
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
    )
    profiled_session = p.stop()
    # Remove __main__ call from this script, from the output stack
    profiled_session.root_frame(trim_stem=True)

    # Parse results into HTML
    # show_all: removes library calls where identifiable
    # timeline: if true, samples are left in chronological order rather than total time
    html_renderer = HTMLRenderer(show_all=False, timeline=False)

    # Parse output and write to file
    output_html_file = output_dir / (
        datetime.datetime.utcnow().strftime("%Y-%m-%d_%H%M") + "_scale_run_profile.html"
    )
    print(f"Writing output to: {output_html_file}", end="...", flush=True)
    with open(output_html_file, "w") as f:
        f.write(html_renderer.render(profiled_session))
    print("done")

    return


if __name__ == "__main__":
    main()
