"""
This is the analysis script for the calibration of the ALRI model
"""
# %% Import Statements and initial declarations
import datetime
import os
import random
from pathlib import Path

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    alri,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
)

# %%
outputpath = Path("./outputs")
resourcefilepath = Path("./resources")

# Create name for log-file
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

log_filename = outputpath / 'alri_classification_and_treatment__2022-03-15T170845.log'
# <-- insert name of log file to avoid re-running the simulation // GBD_lri_comparison_50k_pop__2022-03-15T111444.log
# alri_classification_and_treatment__2022-03-15T170845.log

if not os.path.exists(log_filename):
    # If logfile does not exists, re-run the simulation:
    # Do not run this cell if you already have a logfile from a simulation:

    start_date = Date(2010, 1, 1)
    end_date = Date(2025, 12, 31)
    popsize = 5000

    log_config = {
        "filename": "alri_classification_and_treatment",
        "directory": "./outputs",
        "custom_levels": {
            "*": logging.WARNING,
            "tlo.methods.alri": logging.DEBUG,
            "tlo.methods.demography": logging.INFO,
            "tlo.methods.healthburden": logging.INFO,
        }
    }

    seed = random.randint(0, 50000)

    # Establish the simulation object
    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config,
                     show_progress_bar=True, resourcefilepath=resourcefilepath)

    # run the simulation
    sim.register(
        demography.Demography(),
        enhanced_lifestyle.Lifestyle(),
        symptommanager.SymptomManager(),
        healthseekingbehaviour.HealthSeekingBehaviour(),
        healthburden.HealthBurden(),
        simplified_births.SimplifiedBirths(),
        healthsystem.HealthSystem(service_availability=['*'], mode_appt_constraints=0, ignore_priority=True,
                                  capabilities_coefficient=1.0,
                                  disable=True),
        alri.Alri(),
        alri.AlriPropertiesOfOtherModules()
    )

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # display filename
    log_filename = sim.log_filepath
    print(f"log_filename: {log_filename}")

output = parse_log_file(log_filename)

# -----------------------------------------------------------------------------
classification = output['tlo.methods.alri']['classification']
classification.set_index(
    'year',
    drop=True,
    inplace=True
)

# total number of year running in simulation
n_years = classification.index.value_counts().count()
# facility levels
facilities = classification['facility_level'].value_counts().index

# create separate df for each level
grouped = classification.groupby(['facility_level'])
classification_level0 = grouped.get_group('0')
classification_level1a = grouped.get_group('1a')
classification_level1b = grouped.get_group('1b')


# total classifications at each facility level for the standard being oximeter-based classification
total_symptom_classification = classification.groupby(['pulse_ox_classification']).symptom_classification.value_counts()
total_pulse_ox_classification = classification.groupby(
    ['pulse_ox_classification']).pulse_ox_classification.value_counts()
total_hw_classification = classification.groupby(['pulse_ox_classification']).hw_classification.value_counts()
total_final_classification = classification.groupby(['pulse_ox_classification']).final_classification.value_counts()

# yearly mean of total classifications
mean_pulse_ox_classification = total_pulse_ox_classification / n_years
mean_symptom_classification = total_symptom_classification / n_years
mean_hw_classification = total_hw_classification / n_years
mean_final_classification = total_final_classification / n_years
print(mean_symptom_classification)

# labels = ['chest_indrawing_pneumonia', 'fast_breathing_pneumonia', 'danger_signs_pneumonia', 'cough_or_cold',
#           'not_handled_at_facility_0', 'serious_bacterial_infection']

level0_pulse_ox_class = mean_pulse_ox_classification.loc['danger_signs_pneumonia', slice(None)]
level0_symptom_class = mean_symptom_classification.loc['danger_signs_pneumonia', slice(None)]
level0_hw_class = mean_hw_classification.loc['danger_signs_pneumonia', slice(None)]
level0_final_class = mean_final_classification.loc['danger_signs_pneumonia', slice(None)]

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = level0_symptom_class[:].index
sizes = level0_symptom_class[:]

category_names = level0_symptom_class[:].index.get_level_values('symptom_classification')
print(category_names)
results = {
    'symptom-based': level0_symptom_class[:],
    'health_worker': level0_hw_class[:],
    'final classification': level0_final_class[:],
}
print(results)


# ----------------------------
#
# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
#
# def survey(results, category_names):
#     """
#     Parameters
#     ----------
#     results : dict
#         A mapping from question labels to a list of answers per category.
#         It is assumed all lists contain the same number of entries and that
#         it matches the length of *category_names*.
#     category_names : list of str
#         The category labels.
#     """
#     labels = list(results.keys())
#     data = np.array(list(results.values()))
#     data_cum = data.cumsum(axis=1)
#     cmap = plt.get_cmap('RdYlGn')
#     color_values = np.linspace(0.15, 0.85, 5)
#     category_colors = cmap(color_values)
#
#     fig, ax = plt.subplots(figsize=(9.2, 5))
#     ax.invert_yaxis()
#     ax.xaxis.set_visible(False)
#     ax.set_xlim(0, np.sum(data, axis=1).max())
#
#     for i, (colname, color) in enumerate(zip(category_names, category_colors)):
#         widths = data[:, i]
#         starts = data_cum[:, i] - widths
#         rects = ax.barh(labels, widths, left=starts, height=0.5,
#                         label=colname, color=color)
#
#         r, g, b, _ = color
#         text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
#         ax.bar_label(rects, label_type='center', color=text_color)
#     ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
#               loc='lower left', fontsize='small')
#
#     return fig, ax
#
#
# survey(results, category_names)
# plt.show()

# x = labels
# # x = np.arange(len(labels))  # the label locations
#
# width = 0.1  # the width of the bars
#
# fig, ax = plt.subplots()
# ax.bar(x, level0_pulse_ox_class, width, label='oximeter')
# ax.bar(x, level0_symptom_class, width, label='symptom-based')
# ax.bar(x, level0_hw_class, width, label='health worker')
# ax.bar(x, level0_final_class, width, label='final')
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('number of classifications')
# ax.set_title('Classifications against pulse oximeter standard')
# ax.set_xticks(x, labels)
# ax.legend()
#
# fig.tight_layout()
#
# plt.show()
