"""Produce plots to show the impact of removing each set of Treatments from the healthsystem"""

import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Resource file path
rfp = Path("./resources")

# Where will outputs be found
outputpath = Path("./outputs")  # folder for convenience of storing outputs
results_filename = outputpath / 'health_system_systematic_run.pickle'

with open(results_filename, 'rb') as f:
    results = pickle.load(f)['results']

datestamp = datetime.today().strftime("__%Y_%m_%d")

# %%

# Get total deaths in the duration of each simulation:
deaths = dict()
for key in results.keys():
    deaths[key] = len(results[key]['tlo.methods.demography']['death'])

deaths = pd.Series(deaths)

# compute the excess deaths compared to the all Treatments
excess_deaths = deaths[~(deaths.index == 'Everything')] - deaths['Everything']

excess_deaths.plot.bar()
plt.show()
