"""
This will demonstrate the effect of pulse oximetry and oxygen.
"""

# %% Import Statements and initial declarations
import datetime
from pathlib import Path
from tlo.analysis.utils import parse_log_file

import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
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
resourcefilepath = Path("./resources")
outputpath = Path("./outputs")  # folder for convenience of storing outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# Scenarios Definitions:
# Define the 'service_availability' parameter for two sceanrios (without treatment, with treatment)
scenarios = dict()
scenarios['No_oximeter_and_oxygen'] = False
scenarios['With_oximeter_and_oxygen'] = True

# Create dict to capture the outputs
output_files = dict()

# %% Run the Simulation
start_date = Date(2010, 1, 1)
end_date = Date(2012, 12, 31)
popsize = 50000

for label, oximeter_avail in scenarios.items():

    log_config = {
        "filename": f"alri_{label}",
        "directory": "./outputs",
        "custom_levels": {
            "*": logging.WARNING,
            "tlo.methods.alri": logging.INFO,
            "tlo.methods.demography": logging.INFO,
        }
    }

    # add file handler for the purpose of logging
    sim = Simulation(start_date=start_date, log_config=log_config, show_progress_bar=True)

    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath,),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                  service_availability=["*"],  # all treatment allowed
                                  mode_appt_constraints=0,  # mode of constraints to do with officer numbers and time
                                  cons_availability="all",  # mode for consumable constraints (if ignored, all consumables available)
                                  ignore_priority=True,  # do not use the priority information in HSI event to schedule
                                  capabilities_coefficient=1.0,  # multiplier for the capabilities of health officers
                                  disable=True,  # disables the healthsystem (no constraints and no logging) and every HSI runs
                                  disable_and_reject_all=False,  # disable healthsystem and no HSI runs
                                  store_hsi_events_that_have_run=False,  # convenience function for debugging
                                  ),
        alri.Alri(resourcefilepath=resourcefilepath),
        alri.AlriPropertiesOfOtherModules()
    )

    sim.modules['Demography'].parameters['max_age_initial'] = 5
    sim.make_initial_population(n=popsize)

    # Assume perfect sensitivity in hw classification
    p = sim.modules['Alri'].parameters
    p['sensitivity_of_classification_of_fast_breathing_pneumonia_facility_level0'] = 1.0
    p['sensitivity_of_classification_of_danger_signs_pneumonia_facility_level0'] = 1.0
    p['sensitivity_of_classification_of_non_severe_pneumonia_facility_level1'] = 1.0
    p['sensitivity_of_classification_of_severe_pneumonia_facility_level1'] = 1.0
    p['sensitivity_of_classification_of_non_severe_pneumonia_facility_level2'] = 1.0
    p['sensitivity_of_classification_of_severe_pneumonia_facility_level2'] = 1.0

    if oximeter_avail:
        p['override_po_and_oxygen_availability'] = True
        p['override_po_and_oxygen_to_full_availability'] = True
    else:
        p['override_po_and_oxygen_availability'] = True
        p['override_po_and_oxygen_to_full_availability'] = False

    sim.simulate(end_date=end_date)

    # Save the full set of results:
    output_files[label] = sim.log_filepath


# output_files['No_oximeter_and_oxygen'] = outputpath / 'alri_with_treatment__2022-06-30T124146.log'
# output_files['With_oximeter_and_oxygen'] = outputpath / 'alri_with_treatment__2022-06-30T114728.log'


# %% Extract the relevant outputs and make a graph:
def get_death_numbers_from_logfile(logfile):
    # parse the simulation logfile to get the output dataframes
    output = parse_log_file(logfile)

    # calculate death rate
    deaths_df = output['tlo.methods.demography']['death']
    deaths_df['year'] = pd.to_datetime(deaths_df['date']).dt.year
    deaths = deaths_df.loc[deaths_df['cause'].str.startswith('ALRI')].groupby('year').size()
    # breakdown by age group
    # deaths = deaths_df.loc[deaths_df['cause'].str.startswith('ALRI')].groupby('age').size()

    return deaths


deaths = dict()
for label, file in output_files.items():
    deaths[label] = \
        get_death_numbers_from_logfile(file)

# Plot death rates by year: across the scenarios
data = {}
for label in deaths.keys():
    data.update({label: deaths[label].mean()})

plt.bar(data.keys(), data.values(), align='center')
# pd.concat(data, axis=1).plot.bar()
plt.title(f'Mean deaths due to ALRI from {start_date.year} to {end_date.year}')
# plt.savefig(outputpath / ("ALRI_deaths_by_availability_of_PO_and_Ox" + datestamp + ".pdf"), format='pdf')
plt.show()


# -----------------------------------------------------------------------------------------------
# check the case fatality rates (CFR)

# %% Extract the relevant outputs and make a graph:
def get_CFR_from_logfile(logfile):
    # parse the simulation logfile to get the output dataframes
    output = parse_log_file(logfile)

    # calculate CFR
    counts = output['tlo.methods.alri']['event_counts']
    counts['year'] = pd.to_datetime(counts['date']).dt.year
    counts.drop(columns='date', inplace=True)
    counts.set_index(
        'year',
        drop=True,
        inplace=True
    )

    # using the tracker to get the number of cases per year
    number_of_cases = counts.incident_cases

    # using the tracker to get the number of deaths per year
    number_of_deaths = counts.deaths

    # calculate CFR
    CFR_in_percentage = (number_of_deaths / number_of_cases) * 100

    return CFR_in_percentage


cfr = dict()
for label, file in output_files.items():
    cfr[label] = \
        get_CFR_from_logfile(file)

# Plot death rates by year: across the scenarios
data2 = {}
for label in cfr.keys():
    data2.update({label: cfr[label].mean()})

plt.bar(data2.keys(), data2.values(), align='center')
plt.title(f'Mean CFR from {start_date.year} to {end_date.year}')
plt.show()
