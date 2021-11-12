from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo.analysis.utils import (
    extract_results, extract_str_results,
    get_scenario_outputs, create_pickles_locally, summarize
)

# %% Declare the name of the file that specified the scenarios used in this run.
scenario_filename = 'preliminary_anc_4_test-.py'  # <-- update this to look at other results

# %% Declare usual paths:
outputspath = Path('./outputs/sejjj49@ucl.ac.uk/')
graph_location = 'output_graphs_30k(ANC4TEST)_preliminary_anc_4_test-2021-11-10T153548Z/death'
rfp = Path('./resources')

# Find results folder (most recent run generated using that scenario_filename)
results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]
#create_pickles_locally(results_folder)  # if not created via batch


# ======================================================= ANC 4 ======================================================

# SCENARIO ONE: BASELINE ANC 4 COVERAGE VERSES 90% (UHC) EPMM COVERAGE (CONSUMABLES CONSTRAINED, SQUEEZE CONSTRAINED)

# MATERNAL DEATH
# STILLBIRTH
# NEONATAL DEATH
# HEALTH CARE WORKER TIME
# CONSUMABLES
# (DALYS)

# SCENARIO TWO: BASELINE ANC 4 COVERAGE VERSES 90% (UHC) EPMM COVERAGE (NO CONSUMABLES, QUALITY OR SQUEEZE CONTRAINTS
# IN THE COMPARATOR)

# MATERNAL DEATH
# STILLBIRTH
# NEONATAL DEATH
# HEALTH CARE WORKER TIME
# CONSUMABLES
# (DALYS)

# ======================================================= ANC 8 ======================================================

# SCENARIO ONE: BASELINE ANC 8 COVERAGE VERSES 50% COVERAGE (CONSUMABLES CONSTRAINED, SQUEEZE CONSTRAINED)

# MATERNAL DEATH
# STILLBIRTH
# NEONATAL DEATH
# HEALTH CARE WORKER TIME
# CONSUMABLES
# (DALYS)

# SCENARIO TWO: BASELINE ANC 8 COVERAGE VERSES 50% COVERAGE (NO CONSUMABLES, QUALITY OR SQUEEZE CONTRAINTS
# IN THE COMPARATOR)

# MATERNAL DEATH
# STILLBIRTH
# NEONATAL DEATH
# HEALTH CARE WORKER TIME
# CONSUMABLES
# (DALYS)

# SCENARIO THREE: BASELINE ANC 8 COVERAGE VERSES 90% (UHC) EPMM COVERAGE( CONSUMABLES CONSTRAINED, SQUEEZE CONSTRAINED)

# MATERNAL DEATH
# STILLBIRTH
# NEONATAL DEATH
# HEALTH CARE WORKER TIME
# CONSUMABLES
# (DALYS)

# SCENARIO FOUR: BASELINE ANC 8 COVERAGE VERSES 90% (UHC) EPMM COVERAGE (NO CONSUMABLES, QUALITY OR SQUEEZE CONTRAINTS
# IN THE COMPARATOR)

# MATERNAL DEATH
# STILLBIRTH
# NEONATAL DEATH
# HEALTH CARE WORKER TIME
# CONSUMABLES
# (DALYS)
