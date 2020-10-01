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
end_date = Date(2015, 1, 1)
pop_size = 1000

# This creates the Simulation instance for this run. Because we've passed the `seed` and
# `log_config` arguments, these will override the default behaviour.
sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

# Path to the resource files used by the disease and intervention methods
resources = Path('./resources')

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
log_df = sim.log_filepath  # output

# --------------------------------------- Model outputs ---------------------------------------
pneum_classification_df = log_df['tlo.methods.dx_algorithm_child']['imci_classicications_count']
pneum_classification_df['year'] = pd.to_datetime(pneum_classification_df['date']).dt.year
# pneum_management_df = log_df['tlo.methods.pneumonia']['pneumonia_management_child_info']

# --------------------------------------- Plotting ---------------------------------------
plt.style.use("ggplot")

# Pneumonia incidence
plt.subplot(111)  # numrows, numcols, fignum
plt.plot(pneum_classification_df, pneum_classification_df['year'])
plt.title("Pneumonia classification")
plt.xlabel("Date")
plt.ylabel("number of classifications")
plt.xticks(rotation=90)
plt.legend(["Model"], bbox_to_anchor=(1.04, 1), loc="upper left")

plt.show()

# def get_imci_pneumonia_classification(logfile):
#     output = parse_log_file(logfile)
#     # Calculate the IMCI algorithm from the output counts of ALRI episodes
#     pneum_classification_df = output['tlo.methods.dx_algorithm_child']['imci_classicications_count']
#     pneum_classification_df['year'] = pd.to_datetime(pneum_classification_df['date']).dt.year
#     pneum_classification_df.drop(columns='date', inplace=True)
#     pneum_classification_df.set_index(
#         'year',
#         drop=True,
#         inplace=True
#     )
#     pneum_classification_df.to_csv(r'./outputs/imci_pneumonia_classification.csv', index=False)
#
#     imci_classification_rate = dict()
#     for severity in ['no pneumonia', 'non-severe pneumonia', 'severe pneumonia']:
#         imci_classification_rate[severity] = pneum_classification_df[severity].apply(pd.Series).div([severity],
#                                                                                                     axis=0).dropna()
#     return imci_classification_rate
#
#
# get_imci_pneumonia_classification(log_df)

# def plot_for_column_of_interest(results, column_of_interest):
#     summary_table = dict()
#     for label in results.keys():
#         summary_table.update({label: results[label][column_of_interest]})
#     data = 100 * pd.concat(summary_table, axis=1)
#     data.plot.bar()
#     plt.title(f'IMCI classification severity {column_of_interest}')
#     plt.savefig(("imci_pneumonia_classification" + datestamp + ".pdf"), format='pdf')
#     plt.show()
#
#
# # Plot incidence by pathogen: across the sceanrios
# for column_of_interest in imci_classification_rate[list(imci_classification_rate.keys())[0]].columns:
#     plot_for_column_of_interest(imci_classification_rate, column_of_interest)
#
#
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
