"""Produce plots to show the impact of removing each set of Treatments from the healthsystem"""

import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Define paths and filenames
rfp = Path("./resources")
outputpath = Path("./outputs")  # folder for convenience of storing outputs
results_filename = outputpath / '2020_11_23_health_system_systematic_run.pickle'
make_file_name = lambda stub: outputpath / f"{datetime.today().strftime('%Y_%m_%d''')}_{stub}.png"


with open(results_filename, 'rb') as f:
    results = pickle.load(f)['results']


# %% Make summary plots:
# Get total deaths in the duration of each simulation:
deaths = dict()
for key in results.keys():
    deaths[key] = len(results[key]['tlo.methods.demography']['death'])

deaths = pd.Series(deaths)

# compute the excess deaths compared to the No Treatments
excess_deaths = deaths['Nothing'] - deaths[~(deaths.index == 'Nothing')]

excess_deaths.plot.bar()
plt.savefig(make_file_name('Impact_of_each_treatment_id'))
plt.title('The Impact of Each set of Treatment_IDs')
plt.ylabel('Deaths Averted by treatment_id, 2010-2014')
plt.show()
