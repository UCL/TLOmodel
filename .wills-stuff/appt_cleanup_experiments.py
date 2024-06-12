from cProfile import run
from pathlib import Path
import os
import sys

import git
import numpy as np

LOC_OF_THIS_FILE = Path(os.path.abspath(os.path.dirname(__file__)))
SPEEDUP_RESULTS_DIR = LOC_OF_THIS_FILE / "first_appt_experiments"
SCALE_RUN_DIR = (LOC_OF_THIS_FILE / ".." / "src" / "scripts" / "profiling").resolve()

DEFAULT_RUN_ARGS = {
    "log_level": '"CRITICAL"',
    "seed": 0,
    "ignore_warnings": True,
    "show_progress_bar": True,
}
POP_SIZES = np.array([5e2, 5e3, 5e4], dtype=int)
N_MONTHS = np.array([6, 12, 24, 36, 48], dtype=int)
START_AT = {
    "experimental": (
        2, # Start from N_MONTHS[this number]
        2, # Start from POP_SIZES[this number]
    ),
    "baseline": (0, 0),
}
END_BEFORE = {
    "experimental": (
        N_MONTHS.size,  # End before N_MONTHS[this number]
        POP_SIZES.size,  # End before POP_SIZES[this number]
    ),
    "baseline": (
        N_MONTHS.size,  # End before N_MONTHS[this number]
        POP_SIZES.size,  # End before POP_SIZES[this number]
    ),
}

REPO = git.Repo(LOC_OF_THIS_FILE / "..")
BRANCHES = {
    # "baseline": "master",
    "experimental": "wgraham/1237-speedup-concept",
}


def construct_command(n_years: int, n_months: int, init_pop: int) -> str:
    """"""
    command = (
        f"scale_run(years={n_years}, months={n_months}, initial_population={init_pop}"
    )
    for arg, value in DEFAULT_RUN_ARGS.items():
        command += f", {arg}={value}"
    command += ")"
    return command

if __name__ == "__main__":
    for branch_type, branch_name in BRANCHES.items():
        print(f"Attempting to run on {branch_type} branch...")

        # Check branch exists in the repo
        assert branch_name in REPO.heads, f"{branch_name} is not a branch in {REPO}"
        # Create the output directory
        output_dir = SPEEDUP_RESULTS_DIR / branch_type
        os.makedirs(output_dir, exist_ok=True)

        # Fetch and check out the branch
        print(f"\tChecking out {branch_name}...")
        working_branch = getattr(REPO.heads, branch_name)
        working_branch.checkout()

        # Hacky import :sweat_smile:
        sys.path.append(str(SCALE_RUN_DIR))
        from scale_run import scale_run

        # Run profiling simulations
        months_start_ind = START_AT[branch_type][0]
        months_end_ind = END_BEFORE[branch_type][0]
        for months in N_MONTHS[months_start_ind : months_end_ind]:
            n_years = months // 12
            n_months = months % 12

            pop_range_start = START_AT[branch_type][1] if months == N_MONTHS[months_start_ind] else 0
            pop_range_end = END_BEFORE[branch_type][1] if months == N_MONTHS[months_end_ind - 1] else POP_SIZES.size
            for pop in POP_SIZES[pop_range_start : pop_range_end]:
                output_fname = output_dir / f"y{n_years}-m{n_months}-p{pop}.prof"
                print(
                    f"\tRunning with {n_years} years, {n_months} months, {pop} initial pops...",
                    end="",
                    flush=True,
                )

                cmd = construct_command(n_years, n_months, pop)
                print(f"\n\t {cmd}")

                run(cmd, output_fname)

                print("\t ...done")

        # Remove the imported scale_run as it will need to be re-imported after
        # the following checkout
        del scale_run
        # also purge the hacky path that we inserted
        sys.path.remove(str(SCALE_RUN_DIR))
