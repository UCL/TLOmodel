import argparse
from glob import glob
import os
from pathlib import Path
import pstats
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

DIR_OF_THIS_FILE = Path(os.path.dirname(os.path.abspath(__file__)))

def fetch_stats_from_files(
    folder: Path,
    match_pattern: str,
    function: str,
    *stats_to_capture: str,
):
    """
    Load statistics about a particular function from a folder containing profiling outputs.

    Data is returned as an N-by-M numpy array where:
    - N is the number of profiling results files that were found,
    - M = 2 + number of statistics that were requested.
    Each row of the returned array corresponds to one profiling file.
    The columns are (in order) the number of months in the simulation,
    the initial population, then the requested statistics in the order provided in the input.
    """
    profiling_results = [
        Path(f)
        for f in glob(f"{str(folder)}/{match_pattern}", recursive=True)
    ]
    assert profiling_results, f"Did not locate any profiling results in {folder}"

    # N-by-M array,
    # N = number of profiling files,
    # M = 2 + number of stats requested
    # - 0  ; n_months
    # - 1  ; pop_size
    # - 2: ; Requested stats, in order.
    results = np.zeros(
        (
            len(profiling_results),
            2 + len(stats_to_capture),
        ),
        dtype=float,
    )
    for i, file in enumerate(profiling_results):
        results[i, 0], results[i, 1] = stats_from_filename(file)
        stats = pstats.Stats(str(file)).get_stats_profile()

        function_stats = stats.func_profiles[function]
        for j, stat_name in enumerate(stats_to_capture):
            results[i, 2+j] = getattr(function_stats, stat_name)
    return results


def stats_from_filename(fname: Path) -> Tuple[int, int]:
    """
    Parse a filename of the form y*-m*-p*.prof to return the number
    of months and initial population size used in the simulation.
    """
    name_components = fname.stem.split("-")
    assert len(name_components) == 3 #y{}-m{}-p{}.prof

    n_months = int(name_components[0][1:]) * 12 + int(name_components[1][1:])
    pop_size = int(name_components[2][1:])

    return n_months, pop_size


def main(
    function: str = "_do_on_generic_first_appt",
    baseline_results_dir: Path = DIR_OF_THIS_FILE / "baseline",
    experimental_results_dir: Path = DIR_OF_THIS_FILE / "experimental",
    match_pattern: str = "*.prof",
    function_on_baseline: Optional[str] = None,
    stats_to_fetch: List[str] = ["cpu_time"]
) -> None:
    if not function_on_baseline:
        function_on_baseline = function

    baseline_results = fetch_stats_from_files(
        baseline_results_dir,
        match_pattern,
        function_on_baseline,
        *stats_to_fetch,
    )
    experimental_results = fetch_stats_from_files(
        experimental_results_dir,
        match_pattern,
        function,
        *stats_to_fetch,
    )

    # Compare each stat by making pretty pictures
    for i, stat in enumerate(stats_to_fetch):
        # Create plot for statistic
        fig = plt.figure()
        fig.suptitle(f"{stat}")

        baseline_ax = fig.add_subplot(121)
        baseline_ax.set_xlabel("Sim length (months)")
        baseline_ax.set_ylabel("Initial pop size")
        baseline_plot = baseline_ax.tricontourf(
            baseline_results[:, 0], baseline_results[:, 1], baseline_results[:, i]
        )
        baseline_ax.set_title("Baseline")
        fig.colorbar(baseline_plot, ax=baseline_ax)

        experimental_ax = fig.add_subplot(122)
        experimental_ax.set_xlabel("Sim length (months)")
        experimental_ax.set_ylabel("Initial pop size")
        experimental_plot = experimental_ax.tricontourf(
            experimental_results[:, 0], experimental_results[:, 1], experimental_results[:, i]
        )
        experimental_ax.set_title("Experimental")
        fig.colorbar(experimental_plot, ax=experimental_ax)

        fig.tight_layout()
        fig.savefig(DIR_OF_THIS_FILE / f"{stat}.png")
    
    # Compare stats by displaying the difference
    for i, stat in enumerate(stats_to_fetch):
        # Create difference plot for easier interpretation
        fig = plt.figure()
        fig.suptitle(f"Difference of {stat} (exp - base)")

        ax = fig.add_subplot(111)
        ax.set_xlabel("Sim length (months)")
        ax.set_ylabel("Initial pop size")
        plot = ax.tricontourf(
            baseline_results[:, 0], baseline_results[:, 1], experimental_results[:, i] - baseline_results[:, i],
        )
        
        fig.colorbar(plot, ax=ax)
        fig.tight_layout()
        fig.savefig(DIR_OF_THIS_FILE / f"difference-{stat}.png")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare CPU runtime of a particular function between a baseline and experimental branch"
    )
    parser.add_argument(
        "function", type=str, help="Name of function to run comparisons for."
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        nargs="?",
        default=DIR_OF_THIS_FILE / "first_appt_experiments",
        help="Directory containing profiling results, in two further subfolders."
    )
    parser.add_argument(
        "-b",
        "--baseline-results-dir",
        type=str,
        default="baseline",
        help="Subfolder within the results directory that contains the profiling results from baseline runs.",
    )
    parser.add_argument(
        "-e",
        "--experimental-results-dir",
        type=str,
        default="experimental",
        help="Subfolder within the results directory that contains the profiling results from experimental runs.",
    )
    parser.add_argument(
        "-m",
        "--match-pattern",
        type=str,
        default="**/*y*-m*-p*.prof",
        help="Only files that match the pattern provided will be treated as results files.",
    )
    parser.add_argument(
        "-n",
        "--name-on-baseline",
        type=str,
        default=None,
        help="Name of function to profile on the baseline branch, if it is different to the name on the experimental branch."
    )
    parser.add_argument(
        "-s",
        "--stats-to-report",
        nargs="*",
        type=str,
        default=["cumtime"],
        help="Stats to report from the profiling runs",
    )

    args = parser.parse_args()

    # "_do_on_generic_first_appt"
    main(
        function=args.function,
        baseline_results_dir=Path(args.results_dir) / args.baseline_results_dir,
        experimental_results_dir=Path(args.results_dir) / args.experimental_results_dir,
        match_pattern=args.match_pattern,
        function_on_baseline=args.name_on_baseline,
        stats_to_fetch=args.stats_to_report,
    )
