"""
This file is to analyse the proportions of symptoms re. HSI_GenericEmergencyFirstApptAtFacilityLevel1
"""

from pathlib import Path

from tlo import Date
from tlo.analysis.utils import get_scenario_outputs, load_pickled_dataframes

scenario_filename = 'long_run_all_diseases.py'

# Declare usual paths:
outputspath = Path('./outputs/bshe@ic.ac.uk')
rfp = Path('./resources')

# Find results folder (most recent run generated using that scenario_filename)
results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]
print(f"Results folder is: {results_folder}")

TARGET_PERIOD = (Date(2015, 1, 1), Date(2019, 12, 31))

# Extract results
log = load_pickled_dataframes(results_folder)['tlo.methods.hsi_generic_first_appts']
symptom = log['symptoms_of_person_at_emergency_hsi'].copy()
symptom = symptom.drop(index=symptom.index[~symptom['date'].between(*TARGET_PERIOD)])

# groupby and get avg annual counts of each group of symptoms
symptom['year'] = symptom.date.dt.year.copy()
symptom.drop(columns='date', inplace=True)
symptom = symptom.groupby(['year', 'message']).size().reset_index(name='count')
symptom = symptom.groupby('message')['count'].mean().reset_index()
symptom['per cent'] = 100 * symptom['count']/symptom['count'].sum()

# get the counts relating to RTI module basing on that RTI has two symptoms: injury and severe_trauma
symptom_rti = symptom[
    symptom.message.isin(['injury', 'severe_trauma', 'injury|severe_trauma', 'severe_trauma|injury'])].copy()

# proportion of RTI counts
percent_rti = 100 * symptom_rti['count'].sum()/symptom['count'].sum()
