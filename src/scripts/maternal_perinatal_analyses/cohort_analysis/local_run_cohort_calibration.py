import os
from pathlib import Path

import pandas as pd

from tlo import Date, Simulation, logging
from tlo.methods import mnh_cohort_module
from tlo.methods.fullmodel import fullmodel
from tlo.analysis.utils import parse_log_file


resourcefilepath = Path('./resources')
outputpath = Path("./outputs/cohort_testing")  # folder for convenience of storing outputs
population_size = 2000

sim = Simulation(start_date=Date(2024, 1, 1),
                 seed=123,
                 log_config={"filename": "log_cohort_calibration",
                             "custom_levels": {"*": logging.DEBUG},
                             "directory": outputpath})

sim.register(*fullmodel(resourcefilepath=resourcefilepath),
             mnh_cohort_module.MaternalNewbornHealthCohort(resourcefilepath=resourcefilepath))

sim.make_initial_population(n=population_size)
sim.simulate(end_date=Date(2025, 1, 1))

output = parse_log_file(sim.log_filepath)

# output = parse_log_file(
#              '/Users/j_collins/PycharmProjects/TLOmodel/outputs/log_cohort_calibration__2024-10-04T101535.log')

# Make output dataframe
results = pd.DataFrame(columns=['model', 'data', 'source'],
                       index= ['deaths',
                               'MMR',
                               'DALYs',
                               'twins',
                               'ectopic',
                               'abortion',
                               'miscarriage',
                               'syphilis',
                               'anaemia_an',
                               'anaemia_pn'
                               'gdm',
                               'PROM',
                               'pre_eclampsia',
                               'gest-htn',
                               'severe_gest-htn',
                               'severe pre-eclampsia',
                               'eclampsia',
                               'praevia',
                               'abruption',
                               'aph',
                               'OL',
                               'UR',
                               'sepsis',
                               'PPH'])

# total_pregnancies = population_size
total_pregnancies = 2000 + len(output['tlo.methods.contraception']['pregnancy'])
total_births = len(output['tlo.methods.demography']['on_birth'])
prop_live_births = (total_births/total_pregnancies) * 100

# Mortality/DALY
deaths_df = output['tlo.methods.demography']['death']
prop_deaths_df = output['tlo.methods.demography.detail']['properties_of_deceased_persons']

dir_mat_deaths = deaths_df.loc[(deaths_df['label'] == 'Maternal Disorders')]
init_indir_mat_deaths = prop_deaths_df.loc[(prop_deaths_df['is_pregnant'] | prop_deaths_df['la_is_postpartum']) &
                                  (prop_deaths_df['cause_of_death'].str.contains('Malaria|Suicide|ever_stroke|diabetes|'
                                                                     'chronic_ischemic_hd|ever_heart_attack|'
                                                                     'chronic_kidney_disease') |
                                   (prop_deaths_df['cause_of_death'] == 'TB'))]

hiv_mat_deaths =  prop_deaths_df.loc[(prop_deaths_df['is_pregnant'] | prop_deaths_df['la_is_postpartum']) &
                              (prop_deaths_df['cause_of_death'].str.contains('AIDS_non_TB|AIDS_TB'))]

indir_mat_deaths = len(init_indir_mat_deaths) + (len(hiv_mat_deaths) * 0.3)
total_deaths = len(dir_mat_deaths) + indir_mat_deaths

# TOTAL_DEATHS
results.at['deaths', 'model'] = total_deaths
results.at['MMR', 'model'] = (total_deaths / total_births) * 100_000
results.at['DALYs', 'model'] = output['tlo.methods.healthburden']['dalys_stacked']['Maternal Disorders'].sum()

# Maternal conditions
an_comps = output['tlo.methods.pregnancy_supervisor']['maternal_complication']
la_comps = output['tlo.methods.labour']['maternal_complication']
pn_comps = output['tlo.methods.postnatal_supervisor']['maternal_complication']

total_completed_pregnancies = (len(an_comps.loc[an_comps['type'] == 'ectopic_unruptured']) +
                               len(an_comps.loc[an_comps['type'] == 'induced_abortion']) +
                               len(an_comps.loc[an_comps['type'] == 'spontaneous_abortion']) +
                               total_births +
                               len(output['tlo.methods.pregnancy_supervisor']['antenatal_stillbirth']) +
                               len(output['tlo.methods.labour']['intrapartum_stillbirth']))

# Twins (todo)

# Ectopic
results.at['ectopic', 'model'] = (len(an_comps.loc[an_comps['type'] == 'ectopic_unruptured']) / total_pregnancies) * 1000
results.at['ectopic', 'data'] = 10.0
results.at['ectopic', 'source'] = 'Panelli et al.'

# Abortion


# Miscarriage

# Health system
