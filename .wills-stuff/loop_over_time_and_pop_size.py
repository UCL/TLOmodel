from cProfile import run
import os
from pathlib import Path
import sys

import numpy as np

SIM_LENGTHS = np.arange(6, 6*6, 6, dtype=int)
POP_SIZES = np.logspace(2, 5, num=7, dtype=float)
DEFAULT_RUN_ARGS = {
    "log_level": '"CRITICAL"',
    "seed": 0,
    "ignore_warnings": True,
    "show_progress_bar": True,
}

DIR_OF_THIS_FILE = Path(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DUMP = DIR_OF_THIS_FILE / "results_dump"
SCALE_RUN_DIR = (DIR_OF_THIS_FILE / ".." / "src" / "scripts" / "profiling").resolve()

# Hacky import :sweat_smile:
sys.path.append(str(SCALE_RUN_DIR))
from scale_run import scale_run

def construct_command(n_years: int, n_months: int, init_pop: int) -> str:
    """"""
    command = f"scale_run(years={n_years}, months={n_months}, initial_population={init_pop}"
    for arg, value in DEFAULT_RUN_ARGS.items():
        command += f", {arg}={value}"
    command += ")"
    return command

if __name__ == "__main__":

    print("Beginning profiling:")
    print("====================")

    for sim_length in SIM_LENGTHS:
        n_years = sim_length // 12
        n_months = sim_length % 12
        for pop_size in POP_SIZES:
            init_pop = int(np.floor(pop_size)) # Cast to avoid decimal people
            output_file_name = RESULTS_DUMP / f"speedup-y{n_years}_m{n_months}_p{init_pop}.prof"

            print(
                f"Running with {n_years} years, {n_months} months, {init_pop} initial pops...",
                end="",
                flush=True,
            )

            cmd = construct_command(n_years, n_months, init_pop)
            print(f"\n\t {cmd}")

            run(cmd, output_file_name)

            print("\n\t ...done")
