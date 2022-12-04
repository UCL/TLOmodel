"""
This file is to analyse the proportions of symptoms re. HSI_GenericEmergencyFirstApptAtFacilityLevel1
"""

from pathlib import Path

from matplotlib import pyplot as plt
import pandas as pd

from tlo import Date
from tlo.analysis.utils import (
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,)

scenario_filename = 'long_run_all_diseases.py'

# Declare usual paths:
outputspath = Path('./outputs/bshe@ic.ac.uk')
rfp = Path('./resources')

# Find results folder (most recent run generated using that scenario_filename)
results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]
# results_folder = Path('./outputs/long_run-2022-12-03T130227Z')
print(f"Results folder is: {results_folder}")

TARGET_PERIOD = (Date(2015, 1, 1), Date(2019, 12, 31))
# TARGET_PERIOD = (Date(2010, 1, 1), Date(2010, 12, 31))


def count_annual_emergency_hsi_by_module(draw=0, run=0):
    """
    This function calculates the annual counts of emergency hsi events by module based on the information of symptoms
    :param draw:
    :param run:
    :return:
    """
    log = load_pickled_dataframes(results_folder, draw, run)['tlo.methods.hsi_generic_first_appts']
    symptom = log['symptoms_of_person_at_emergency_hsi'].copy()
    symptom = symptom.drop(index=symptom.index[~symptom['date'].between(*TARGET_PERIOD)])
    scaling_factor = load_pickled_dataframes(results_folder
                                             )['tlo.methods.population']['scaling_factor']['scaling_factor'].values[0]

    # groupby and get avg annual counts of each group of symptoms
    symptom['year'] = symptom.date.dt.year.copy()
    symptom.drop(columns='date', inplace=True)
    symptom = symptom.groupby(['year', 'message']).size().reset_index(name='count')
    symptom['count'] = symptom['count'] * scaling_factor
    symptom = symptom.groupby('message')['count'].mean().reset_index()
    symptom['per cent'] = 100 * symptom['count'] / symptom['count'].sum()
    symptom = symptom.sort_values(by=['count'], ascending=False).reset_index(drop=True)

    # list the possible symptoms and modules called by function do_at_generic_first_appt_emergency
    symp = {'alri': ['danger_signs'],
            'cmd': ['ever_stroke_damage', 'ever_heart_attack_damage'],
            'depression': ['Injuries_From_Self_Harm'],
            'malaria': ["acidosis", "coma_convulsions", "renal_failure", "shock"],
            'chronic_syndrome': ['craving_sandwiches'],
            'mockitis': ['extreme_pain_in_the_nose'],
            'rti': ['severe_trauma'],
            'measles': ['encephalitis']
            }

    # other alri symptoms ['cough', 'difficult_breathing', 'cyanosis', 'fever', 'tachypnoea', 'chest_indrawing']
    # other cmd symptoms ['chronic_lower_back_pain_symptoms']
    # other rit symptoms ['injury]
    # other malaria symptoms ["jaundice", "anaemia"]
    # other measles symptoms ['rash', 'fever', 'diarrhoea', 'otitis_media', 'respiratory_symptoms', 'eye_complaint']
    # tb symptoms ["fever", "respiratory_symptoms", "fatigue", "night_sweats"]
    # diarrhoea symptoms: ['diarrhoea', 'bloody stools', 'fever', 'vomiting', 'dehydration']
    # hiv symptoms ['aids_symptoms']
    # 'other_cancer': ['early_other_adult_ca_symptom']
    # generic symptoms ['fever', 'vomiting', 'stomachache', 'sore_throat', 'respiratory_symptoms', 'headache',
    #                   'skin_complaint', 'dental_complaint', 'backache', 'injury', 'eye_complaint', 'diarrhoea']

    # map to module based on symptoms in a simple way
    symptom.message = [x.split('|') for x in symptom.message]
    for i in symptom.index:
        symptom.loc[i, 'module'] = ','.join(
            [k for k in symp.keys() if set(symp[k]).intersection(set(symptom.loc[i, 'message']))]
        )  # there may be multiple modules mapped to the same set of symptoms

    # fill nan entries
    # the null message
    null_message_idx = [i for i in symptom.index if symptom.loc[i, 'message'] == ['']]
    symptom.loc[null_message_idx, 'module'] = 'unknown'
    # simple message of one generic symptom or others not included in above modules
    null_module_idx = symptom[symptom.module == ''].index
    symptom.loc[null_module_idx, 'module'] = 'generic and others'

    # get counts of modules
    mod_by_symp = symptom.groupby('module')['count'].sum().reset_index()
    # split multiple-module list into single modules
    mod_by_symp.module = mod_by_symp.module.str.split(',')
    mod_by_symp_split = pd.DataFrame(mod_by_symp.module.tolist(), mod_by_symp.index)
    mod_by_symp_split = mod_by_symp_split.merge(mod_by_symp['count'], left_index=True, right_index=True)
    # count of unique module by symptom
    uni_mod_by_symp = pd.DataFrame(mod_by_symp_split[[0, 'count']], mod_by_symp_split.index).rename(
        columns={0: 'module'})
    for col in mod_by_symp_split.columns[1:-1]:
        df = pd.DataFrame(mod_by_symp_split[[col, 'count']], mod_by_symp_split.index).rename(columns={col: 'module'})
        df = df.dropna()
        uni_mod_by_symp = pd.concat([uni_mod_by_symp, df]).groupby('module')['count'].sum().reset_index()

    return uni_mod_by_symp


# get the mean counts for each draw and run
scenario_info = get_scenario_info(results_folder)
# the first draw and run results
emerg_hsi_count_by_mod = count_annual_emergency_hsi_by_module(0, 0)
# concat all runs results
for d in range(scenario_info['number_of_draws']):
    for r in range(scenario_info['runs_per_draw']):
        if d + r > 0:
            count_df = count_annual_emergency_hsi_by_module(d, r)
            # merge and fillna with 0 value
            emerg_hsi_count_by_mod = emerg_hsi_count_by_mod.merge(count_df, on='module', how='outer').fillna(0)
# calculate mean
emerg_hsi_count_by_mod['average_annual_count'] = emerg_hsi_count_by_mod[emerg_hsi_count_by_mod.columns[1:]].mean(axis=1)
# drop columns
emerg_hsi_count_by_mod = emerg_hsi_count_by_mod[['module', 'average_annual_count']].copy()
# sort by count
emerg_hsi_count_by_mod = emerg_hsi_count_by_mod.sort_values(by=['average_annual_count'], ascending=False)
# percentage
emerg_hsi_count_by_mod['proportion'] = 100 * emerg_hsi_count_by_mod['average_annual_count'] \
                                       / emerg_hsi_count_by_mod['average_annual_count'].sum()

# plot proportion by module
title_of_figure = 'Proportions of modules using AandE/FirstAttendance_Emergency'
fig = plt.figure()
ax = emerg_hsi_count_by_mod.plot.bar(x='module', y='proportion')
ax.set_title(title_of_figure)
ax.set_xlabel('module')
ax.set_ylabel('proportion %')
plt.tight_layout()
plt.show()
