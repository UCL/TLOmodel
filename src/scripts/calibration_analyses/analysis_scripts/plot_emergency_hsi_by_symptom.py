"""
This file is to analyse the proportions of symptoms re. HSI_GenericEmergencyFirstApptAtFacilityLevel1
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import get_cmap

from tlo.analysis.utils import get_scenario_outputs, load_pickled_dataframes

scenario_filename = 'long_run_all_diseases.py'

# Declare usual paths:
outputspath = Path('./outputs/bshe@ic.ac.uk')
rfp = Path('./resources')

# Find results folder (most recent run generated using that scenario_filename)
results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]
print(f"Results folder is: {results_folder}")

# Extract results
log = load_pickled_dataframes(results_folder)['tlo.methods.hsi_generic_first_appts']
