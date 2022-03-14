"""Tests to check simulations are deterministic when run with fixed random seed."""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.slow
def test_scale_run_script_deterministic(tmp_path):
    """Test running scale_run profiling scripts gives deterministic output.

    The script is run twice using subprocess with a different PYTHONHASHSEED (random
    seed used in hash randomization) specified for each to allow checking for
    non-determinism caused by relying implicitly on non-repeatable hashing, for example
    iterating over a set.
    """
    script_path = (
        Path(os.path.dirname(__file__))
        / "../src/scripts/profiling/scale_run.py"
    )
    assert os.path.exists(script_path), "Cannot find scale_run script at specified path"
    command_args = [
        sys.executable,
        str(script_path.resolve()),
        "--years",
        "1",
        "--months",
        "6",
        "--initial-population",
        "1000",
        "--seed",
        "645407762",
        "--output-dir",
        str(tmp_path),
        "--save-final-population",
    ]
    final_population_dataframes = []
    env = dict(os.environ)
    for hash_seed in (564059029, 1143360992):
        env["PYTHONHASHSEED"] = str(hash_seed)
        completed_process = subprocess.run(command_args, env=env)
        assert completed_process.returncode == 0, (
            f"Running {' '.join(command_args)} fails to successfully complete "
            f"with stderr output\n\n{completed_process.stderr}"
        )
        final_population_pickle_path = tmp_path / "final_population.pkl"
        assert os.path.exists(final_population_pickle_path), (
            "Pickle of final population dataframe not found"
        )
        final_population_dataframes.append(pd.read_pickle(final_population_pickle_path))
    pd.testing.assert_frame_equal(*final_population_dataframes), (
        f"Running {' '.join(command_args)} twice produces different final populations. "
        "This may be due to for example using an unseeded random number generator such "
        "as routines from `numpy.random` or the built-in `random` module, or due to "
        "the presence of code which iterates over a unordered collections such as set "
        "without fixing the ordering by for example using `sorted`."
    )
