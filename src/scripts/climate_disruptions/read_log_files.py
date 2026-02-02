from tlo.analysis.utils import create_pickles_locally, parse_log_file
"""
General utility functions for TLO analysis
"""
import fileinput
import gzip
import json
import os
import pickle
import warnings
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, TextIO, Tuple, Union

import git
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import squarify

from tlo import Date, Simulation, logging, util
from tlo.logging.reader import LogData
from tlo.util import (
    create_age_range_lookup,
    parse_csv_values_for_columns_with_mixed_datatypes,
    read_csv_files,
)

# File paths

def create_pickles_locally(scenario_output_dir, compressed_file_name_prefix=None):
    """For a run from the Batch system that has not resulted in the creation of the pickles, reconstruct the pickles
     locally."""

    def turn_log_into_pickles(logfile):
        print(f"Opening {logfile}")
        outputs = parse_log_file(logfile)
        for key, output in outputs.items():
            if key.startswith("tlo."):
                print(f" - Writing {key}.pickle")
                with open(logfile.parent / f"{key}.pickle", "wb") as f:
                    pickle.dump(output, f)

    def uncompress_and_save_logfile(compressed_file) -> Path:
        """Uncompress and save a log file and return its path."""
        target = compressed_file.parent / str(compressed_file.name[0:-3])

        # If target already exists, skip decompression
        if target.exists():
            print(f"  Uncompressed log already exists: {target}")
            return target

        # Check if file is actually gzipped
        with open(compressed_file, 'rb') as f:
            magic = f.read(2)

        if magic == b'\x1f\x8b':  # gzip magic number
            with open(target, "wb") as t:
                with gzip.open(compressed_file, 'rb') as s:
                    t.write(s.read())
            return target
        else:
            # File is not gzipped - it's already the log file
            print(f"  File is not gzipped, using directly: {compressed_file}")
            return compressed_file

    draw_folders = [f for f in os.scandir(scenario_output_dir) if f.is_dir()]
    for draw_folder in draw_folders:
        print(draw_folder)
        run_folders = [f for f in os.scandir(draw_folder) if f.is_dir()]
        print(run_folders)
        for run_folder in run_folders:
            # Find the original log-file written by the simulation
            if compressed_file_name_prefix is None:
                logfile = [x for x in os.listdir(run_folder) if x.endswith('.log')][0]
            else:
                compressed_file_name = [
                    x for x in os.listdir(run_folder) if x.startswith(compressed_file_name_prefix)
                ][0]
                logfile = uncompress_and_save_logfile(Path(run_folder) / compressed_file_name)

            turn_log_into_pickles(logfile)
# Parse the log file

scenario_output_dir = '/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/climate_scenario_runs_lhs_param_scan-2026-01-25T135152Z/'

# # get the pickled files if not generated at the batch run
create_pickles_locally(scenario_output_dir = scenario_output_dir, compressed_file_name_prefix='climate_scenario_runs_lhs_param_scan')
