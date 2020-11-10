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
end_date = Date(2012, 1, 1)
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

# ------------------------------ IMCI PNEUMONIA CLASSIFICATIONS (AS GOLD STANDARD) ------------------------------
# ----- Model outputs -----
# pneum_classification = output['tlo.methods.dx_algorithm_child']['imci_classicications_count']
# pneum_classification['date'] = pd.to_datetime(pneum_classification['date']).dt.year
# pneum_classification = pneum_classification.set_index('date')
#
# # ----- Plotting -----
# # plt.style.use("ggplot")
#
# # Pneumonia IMCI classification
# names = list(pneum_classification.columns)
# print(names)
# ax = pneum_classification.plot.bar(rot=0)
# # plt.figure(figsize=(9, 3))
# plt.show()
#
# # save into an cvs file
# pneum_classification.to_csv(r'./outputs/pneum_classification.csv', index=False)

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
# plt.figure(figsize=(9, 3))
plt.title('Mean of health worker classifications vs IMCI gold standard of IMCI pneumonia ')
plt.savefig(outputpath / ("total_health_worker_vs_IMCI_gold_classifications_mean_of_years" + datestamp + ".pdf"), format='pdf')
plt.show()

# save into an cvs file
hw_classification_df.to_csv(r'./outputs/pneum_classification.csv', index=False)

# # ----- Plotting -----
# # Pneumonia IMCI classification as gold standard -------
# names = list(imci_gold_classification_df.columns)
# print(names)
# ax2 = imci_gold_classification_df.plot.bar(rot=0)
# # plt.figure(figsize=(9, 3))
# plt.show()
#
# # save into an cvs file
# imci_gold_classification_df.to_csv(r'./outputs/pneum_classification.csv', index=False)

# def get_pneumonia_management_information(logfile):
#     output = parse_log_file(logfile)
#     # Calculate the IMCI algorithm from the output counts of ALRI episodes
#     pneum_management_df = output['tlo.methods.pneumonia']['pneumonia_management_child_info']
#     pneum_management_df['year'] = pd.to_datetime(pneum_management_df['date']).dt.year
#     pneum_management_df.drop(columns='date', inplace=True)
#     pneum_management_df.set_index(
#         'year',
#         drop=True,
#         inplace=True
#     )
#
#     # data_items = output['tlo.methods.pneumonia']['pneumonia_management_child_info'].items()
#     # df = pd.DataFrame(data_items)
#     # return print(df)
#
#     # create empty dictionary of {'column_name': column_data}, then fill it with all data
#     df_data = {}
#     for col in pneum_management_df.keys():
#         column_name = f'{col}'
#         column_data = pneum_management_df.values()
#         df_data[column_name] = column_data
#
#     # convert dictionary into pandas dataframe
#     df_results_management = pd.DataFrame(data=df_data, index=sim.population.props.loc['ri_ALRI_status'])
#     return print(df_results_management)
#
#
# get_pneumonia_management_information(log_df)
