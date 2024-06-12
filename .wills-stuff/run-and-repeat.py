from cProfile import run
from pathlib import Path
import os
import sys
from typing import List, Optional

from parsing_cprofile import fetch_stats_from_files

LOC_OF_THIS_FILE = Path(os.path.abspath(os.path.dirname(__file__)))
SPEEDUP_RESULTS_DIR = LOC_OF_THIS_FILE / "first_appt_experiments"
SCALE_RUN_DIR = (LOC_OF_THIS_FILE / ".." / "src" / "scripts" / "profiling").resolve()

DEFAULT_RUN_ARGS = {
    "log_level": '"CRITICAL"',
    "seed": 0,
    "ignore_warnings": True,
    "show_progress_bar": True,
}


def construct_command(n_years: int, n_months: int, init_pop: int) -> str:
    """
    Create a string that calls the scale_run script (without pyinstrument)
    when executed on the command line.
    """
    command = (
        f"scale_run(years={n_years}, months={n_months}, initial_population={init_pop}"
    )
    for arg, value in DEFAULT_RUN_ARGS.items():
        command += f", {arg}={value}"
    command += ")"
    return command


def run_with_n_reps(
    n_reps: Optional[int] = 10,
    reps_count_from: Optional[int] = 0,
    results_dir: Optional[Path] = SPEEDUP_RESULTS_DIR / "repeats",
) -> None:
    """
    Run the same scale_run simulation a number of times, storing the profiling results for each.
    """
    # Create the output directory
    os.makedirs(results_dir, exist_ok=True)

    # Hacky import :sweat_smile:
    sys.path.append(str(SCALE_RUN_DIR))
    from scale_run import scale_run

    n_months = 12
    init_pop = 10000
    cmd = construct_command(0, n_months, init_pop)

    for rep_number in range(reps_count_from, reps_count_from + n_reps):
        # Run profiling simulations
        output_fname = results_dir / f"m{n_months}-p{init_pop}-repeat{rep_number}.prof"
        print(
            f"\tRepeat {rep_number}\n",
            end="",
            flush=True,
        )
        run(cmd, output_fname)
        print("\t ...done")

    # Remove the imported scale_run as it will need to be re-imported after
    # the following checkout
    del scale_run
    # also purge the hacky path that we inserted
    sys.path.remove(str(SCALE_RUN_DIR))


def report_stats(
    fn_name: List[str] = [
        "do_at_generic_first_appt_non_emergency",
        "_do_on_generic_first_appt",
    ],
    stats_to_fetch: List[str] = ["cumtime"],
    results_dir: Optional[Path] = SPEEDUP_RESULTS_DIR / "repeats",
    baseline_dir: Optional[str] = "baseline",
    experimental_dir: Optional[str] = "experimental",
) -> None:
    """
    Report t

    :param fn_name: List of one or two elements.
    0th element is the name of the function to compare statistics of, on the baseline branch.
    1st element is the name of the function as it appears on the experimental branch.
    Use a list of one element to indicate that the function names are the same on both branches.
    """
    if len(fn_name) == 1:
        fn_name_base = fn_name_exp = fn_name[0]
    elif len(fn_name) == 2:
        fn_name_base = fn_name[0]
        fn_name_exp = fn_name[1]
    else:
        raise ValueError(f"Expected list of 1 or 2 elements for fn_name, got length {len(fn_name)}: {fn_name}")

    baseline_results = fetch_stats_from_files(results_dir / baseline_dir, "*.prof", fn_name_base, *stats_to_fetch)
    experimental_results = fetch_stats_from_files(results_dir / experimental_dir, "*.prof", fn_name_exp, *stats_to_fetch)

    return baseline_results, experimental_results

if __name__ == "__main__":
    # run_with_n_reps(10, results_dir=SPEEDUP_RESULTS_DIR / "repeats" / "experimental")

    print(report_stats())