"""
This will run the DxAlgorithmChild Module
"""
# %% Import Statements and initial declarations
import datetime
import os
from pathlib import Path

import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    contraception,
    demography,
    pneumonia,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    pregnancy_supervisor,
    symptommanager,
)

seed = 123

log_config = {
    "filename": "imci_analysis",   # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.pneumonia": logging.INFO,
        "tlo.methods.dx_algorithm_child": logging.INFO
    }
}

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# Basic arguments required for the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 2)
pop_size = 50000

# This creates the Simulation instance for this run. Because we've passed the `seed` and
# `log_config` arguments, these will override the default behaviour.
sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

# Path to the resource files used by the disease and intervention methods
resources = Path('./resources')

outputpath = Path('./outputs')

# Used to configure health system behaviour
service_availability = ["*"]

# We register all modules in a single call to the register method, calling once with multiple
# objects. This is preferred to registering each module in multiple calls because we will be
# able to handle dependencies if modules are registered together
sim.register(
    demography.Demography(resourcefilepath=resources),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resources),
    healthsystem.HealthSystem(resourcefilepath=resources, service_availability=service_availability),
    symptommanager.SymptomManager(resourcefilepath=resources),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resources),
    contraception.Contraception(resourcefilepath=resources),
    labour.Labour(resourcefilepath=resources),
    pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resources),
    pneumonia.ALRI(resourcefilepath=resources),
    dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resources)
)

# create and run the simulation
sim.make_initial_population(n=pop_size)
sim.simulate(end_date=end_date)

# parse the simulation logfile to get the output dataframes
output = parse_log_file(sim.log_filepath)

# ------------------------------ IMCI PNEUMONIA MANAGEMENT OF SICK CHILDREN ------------------------------
# ----- Model outputs -----
# output of health worker classification
hw_classification_df = output['tlo.methods.dx_algorithm_child']['hw_pneumonia_classification']
hw_classification_df['date'] = pd.to_datetime(hw_classification_df['date']).dt.year
hw_classification_df = hw_classification_df.set_index('date')

# output of IMCI gold standard
imci_gold_classification_df = output['tlo.methods.dx_algorithm_child']['imci_gold_standard_classification']
imci_gold_classification_df['date'] = pd.to_datetime(imci_gold_classification_df['date']).dt.year
imci_gold_classification_df = imci_gold_classification_df.set_index('date')
# -----------------------------------

# ----- Format the data -----
get_mean = hw_classification_df[['common_cold', 'non-severe_pneumonia', 'severe_pneumonia']].mean(axis=0)
hw_classification_mean = pd.DataFrame(get_mean).T
hw_classification_mean['label'] = 'health_worker_classification'
hw_classification_mean.set_index(
        'label',
        drop=True,
        inplace=True
    )

get_mean = imci_gold_classification_df[['common_cold', 'non-severe_pneumonia', 'severe_pneumonia']].mean(axis=0)
imci_gold_classification_mean = pd.DataFrame(get_mean).T
imci_gold_classification_mean['label'] = 'imci_gold_classification'
imci_gold_classification_mean.set_index(
        'label',
        drop=True,
        inplace=True
    )

final_df = pd.concat([hw_classification_mean.T, imci_gold_classification_mean.T], axis=1)  # rotated index is now columns
# ------------------------------------
# ----- Plotting -----
plt.style.use('ggplot')

# Pneumonia IMCI classification by health workers -------
names = list(final_df.columns)
ax1 = final_df.plot.bar(rot=0)
plt.ylabel('average number of cases per year')
# plt.figure(figsize=(9, 3))
plt.title('Mean of health worker classifications vs IMCI gold standard of IMCI pneumonia ')
plt.savefig(outputpath / ("total_health_worker_vs_IMCI_gold_classifications_mean_of_years" + datestamp + ".pdf"), format='pdf')
plt.show()

# save into an cvs file
hw_classification_df.to_csv(r'./outputs/pneum_classification.csv', index=False)

# -------------------------- cross tabulation hw vs imci ----------------------------------
# output of health worker classification vs IMCI classification ---------------------------

# health worker's classification for IMCI-defined no pneumonia
hw_classification_for_imci_no_pneum = output['tlo.methods.dx_algorithm_child']['hw_classification_for_common_cold_by_IMCI']
hw_classification_for_imci_no_pneum['date'] = pd.to_datetime(hw_classification_for_imci_no_pneum['date']).dt.year
hw_classification_for_imci_no_pneum = hw_classification_for_imci_no_pneum.set_index('date')

# ----- Format the data -----
get_mean_hw_class_for_imci_no_pneum = hw_classification_for_imci_no_pneum[['common_cold', 'non-severe_pneumonia',
                                                                           'severe_pneumonia']].mean(axis=0)
hw_mean_class_for_imci_no_pneum = pd.DataFrame(get_mean_hw_class_for_imci_no_pneum).T
hw_mean_class_for_imci_no_pneum['label'] = 'IMCI_no_pneumonia'
hw_mean_class_for_imci_no_pneum.set_index(
        'label',
        drop=True,
        inplace=True
    )

# health worker's classification for IMCI-defined non-severe pneumonia
hw_classification_for_imci_nonsev_pneum = output['tlo.methods.dx_algorithm_child']['hw_classification_for_non-sev_pneumonia_by_IMCI']
hw_classification_for_imci_nonsev_pneum['date'] = pd.to_datetime(hw_classification_for_imci_nonsev_pneum['date']).dt.year
hw_classification_for_imci_nonsev_pneum = hw_classification_for_imci_nonsev_pneum.set_index('date')

# ----- Format the data -----
get_mean_hw_class_for_imci_nonsev_pneum = hw_classification_for_imci_nonsev_pneum[
    ['common_cold', 'non-severe_pneumonia', 'severe_pneumonia']].mean(axis=0)
hw_mean_class_for_imci_nonsev_pneum = pd.DataFrame(get_mean_hw_class_for_imci_nonsev_pneum).T
hw_mean_class_for_imci_nonsev_pneum['label'] = 'IMCI_pneumonia'
hw_mean_class_for_imci_nonsev_pneum.set_index(
        'label',
        drop=True,
        inplace=True
    )

# health worker's classification for IMCI-defined severe pneumonia
hw_classification_for_imci_severe_pneum = output['tlo.methods.dx_algorithm_child']['hw_classification_for_severe_pneumonia_by_IMCI']
hw_classification_for_imci_severe_pneum['date'] = pd.to_datetime(hw_classification_for_imci_severe_pneum['date']).dt.year
hw_classification_for_imci_severe_pneum = hw_classification_for_imci_severe_pneum.set_index('date')

# ----- Format the data -----
get_mean_hw_class_for_imci_severe_pneum = hw_classification_for_imci_severe_pneum[
    ['common_cold', 'non-severe_pneumonia', 'severe_pneumonia']].mean(axis=0)
hw_mean_class_for_imci_severe_pneum = pd.DataFrame(get_mean_hw_class_for_imci_severe_pneum).T
hw_mean_class_for_imci_severe_pneum['label'] = 'IMCI_severe_pneumonia'
hw_mean_class_for_imci_severe_pneum.set_index(
        'label',
        drop=True,
        inplace=True
    )

# join all dataframes
joined_df = pd.concat([hw_mean_class_for_imci_no_pneum.T, hw_mean_class_for_imci_nonsev_pneum.T,
                       hw_mean_class_for_imci_severe_pneum.T], axis=1)  # rotated index is now columns

# ----- Plotting -----
plt.style.use('ggplot')

# Pneumonia IMCI classification by health workers -------
names1 = list(joined_df.columns)
ax2 = joined_df.T.plot.bar(rot=0)
plt.ylabel('average number of cases per year')
# plt.figure(figsize=(9, 3))
plt.title('Mean of health worker classifications vs IMCI gold standard of IMCI pneumonia')
plt.savefig(outputpath / ("health_worker_vs_IMCI_classifications_mean_of_years" + datestamp + ".pdf"), format='pdf')
plt.show()

# save into an cvs file
hw_classification_df.to_csv(r'./outputs/pneum_classification.csv', index=False)
