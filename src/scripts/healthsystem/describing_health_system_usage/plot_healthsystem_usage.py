"""Produce plots to show the usage of the healthcare system.
This uses the file that is created by: /src/scripts/healthburden_analyses/single_long_run/run_model_and_pickle_log.py
"""
# TODO - clean-up this script! :-)

import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tlo.methods import demography

# Define the particular year for the focus of this analysis
year = 2010

# Resource file path
rfp = Path("./resources")

# Where will outputs be found
outputpath = Path("./outputs")  # folder for convenience of storing outputs
results_filename = outputpath / 'long_run.pickle'

with open(results_filename, 'rb') as f:
    output = pickle.load(f)['output']

datestamp = datetime.today().strftime("__%Y_%m_%d")

# %% Scaling Factor
def get_scaling_factor(parsed_output):
    """Find the factor that the model results should be multiplied by to be comparable to data"""
    # Get information about the real population size (Malawi Census in 2018)
    cens_tot = pd.read_csv(rfp / "ResourceFile_PopulationSize_2018Census.csv")['Count'].sum()
    cens_yr = 2018

    # Get information about the model population size in 2018 (and fail if no 2018)
    model_res = parsed_output['tlo.methods.demography']['population']
    model_yr = pd.to_datetime(model_res.date).dt.year

    if cens_yr in model_yr.values:
        model_tot = model_res.loc[model_yr == cens_yr, 'total'].values[0]
    else:
        print("WARNING: Model results do not contain the year of the census, so cannot scale accurately")
        model_tot = model_res.at[abs(model_yr - cens_yr).idxmin(), 'total']

    # Calculate ratio for scaling
    return cens_tot / model_tot

scaling_factor = get_scaling_factor(output)


# %% Process and plot the file for the usage of the healthsytem:

cap = output['tlo.methods.healthsystem']['Capacity'].copy()
cap["date"] = pd.to_datetime(cap["date"])
cap = cap.set_index('date')

frac_time_used = cap['Frac_Time_Used_Overall']
cap = cap.drop(columns = ['Frac_Time_Used_Overall'])

# Plot Fraction of total time of health-care-workers being used
frac_time_used.plot()
plt.title("Fraction of total health-care worker time being used")
plt.xlabel("Date")
plt.savefig(outputpath / 'HSI_Frac_time_used.png')
plt.show()

# Plot the HSI that are taking place, by month
hsi = output['tlo.methods.healthsystem']['HSI_Event'].copy()
hsi["date"] = pd.to_datetime(hsi["date"])
hsi["month"] = hsi["date"].dt.month

# Reduce TREATMENT_ID to the originating module
hsi["Module"] = hsi["TREATMENT_ID"].str.split('_').apply(lambda x: x[0])

evs = hsi.groupby(by=['month', 'Module'])\
    .size().reset_index().rename(columns={0: 'count'})\
    .pivot_table(index='month', columns='Module', values='count', fill_value=0)
evs *= scaling_factor

evs.plot.bar(stacked=True)
plt.title('HSI by Module, per Month')
plt.savefig(outputpath / 'HSI_per_module_per_month.png')
plt.show()
