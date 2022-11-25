"""
This file is to analyse the proportions of symptoms re. HSI_GenericEmergencyFirstApptAtFacilityLevel1
"""

from pathlib import Path

from matplotlib import pyplot as plt

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
symptom = symptom.sort_values(by=['count'], ascending=False).reset_index(drop=True)
symptom['cumulative per cent'] = symptom['per cent'].cumsum()

# list the possible symptoms and modules called by function do_at_generic_first_appt_emergency
symp = {'alri': ['cough', 'difficult_breathing'],
        'cmd': ['chronic_lower_back_pain_symptoms', 'ever_stroke_damage', 'ever_heart_attack_damage'],
        'depression': ['Injuries_From_Self_Harm'],
        'malaria': ["acidosis", "coma_convulsions", "renal_failure", "shock", "jaundice", "anaemia"],
        'chronic_syndrome': ['craving_sandwiches'],
        'mockitis': ['extreme_pain_in_the_nose'],
        'rti': ['severe_trauma'],
        # 'hiv': ['aids_symptoms'],
        'measles': ['encephalitis'],
        # 'tb': ["fatigue", "night_sweats"],
        'other_cancer': ['early_other_adult_ca_symptom']}

# other alri symptoms ['cyanosis', 'fever', 'tachypnoea', 'chest_indrawing', 'danger_signs']
# other rit symptoms ['injury]
# other measles symptom ['rash', 'fever', 'diarrhoea', 'otitis_media', 'respiratory_symptoms', 'eye_complaint']
# other tb symptom ["fever", "respiratory_symptoms", "fatigue", "night_sweats"]
# 'diarrhoea': ['diarrhoea', 'bloody stools', 'fever', 'vomiting', 'dehydration']
# generic symptoms
# generic = ['fever', 'vomiting', 'stomachache', 'sore_throat', 'respiratory_symptoms', 'headache',
#             'skin_complaint', 'dental_complaint', 'backache', 'injury', 'eye_complaint', 'diarrhoea']

# map to module based on symptoms in a simple way
symptom.message = [x.split('|') for x in symptom.message]
for i in symptom.index:
    for k in symp.keys():
        if set(symp[k]).intersection(set(symptom.loc[i, 'message'])):  # if the message contains symptoms of a module
            symptom.loc[i, 'module'] = k

# fill nan entries
# the null message
symptom.loc[symptom.message.isin([['']]), 'module'] = 'unknown'
# simple message of one generic symptom
symptom.module = symptom.module.fillna('generic and others')

# get counts of modules
mod_by_symp = symptom.groupby('module')['count'].sum().reset_index()
mod_by_symp['proportion'] = 100 * mod_by_symp['count']/mod_by_symp['count'].sum()
mod_by_symp = mod_by_symp.sort_values(by=['count'], ascending=False)

# plot proportion by module
title_of_figure = 'Proportions of modules using AandE/FirstAttendance_Emergency'
fig = plt.figure()
ax = mod_by_symp.plot.bar(x='module', y='proportion')
ax.set_title(title_of_figure)
ax.set_xlabel('module')
ax.set_ylabel('proportion %')
plt.tight_layout()
plt.show()
