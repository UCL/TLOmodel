from pathlib import Path

import numpy as np
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    cardio_metabolic_disorders,
    care_of_women_during_pregnancy,
    contraception,
    demography,
    depression,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    malaria,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager,
)

seed = 7844

resourcefilepath = Path("./resources")

log_config = {
        "filename": "dalys",
        "directory": "./outputs",
        "custom_levels": {
            "*": logging.DEBUG}}

sim = Simulation(start_date=Date(2010, 1, 1), seed=seed, log_config=log_config)

sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             contraception.Contraception(resourcefilepath=resourcefilepath),
             enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
             healthburden.HealthBurden(resourcefilepath=resourcefilepath),
             healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                       service_availability=['*'],
                                       mode_appt_constraints=2,
                                       cons_availability='all'),
             symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
             cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=resourcefilepath),
             healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
             malaria.Malaria(resourcefilepath=resourcefilepath),
             hiv.Hiv(resourcefilepath=resourcefilepath),
             depression.Depression(resourcefilepath=resourcefilepath),
             pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
             care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
             labour.Labour(resourcefilepath=resourcefilepath),
             newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
             postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath))

sim.make_initial_population(n=2000)

df = sim.population.props
women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
df.loc[women_repro.index, 'is_pregnant'] = True
df.loc[women_repro.index, 'date_of_last_pregnancy'] = sim.start_date
for person in women_repro.index:
    sim.modules['Labour'].set_date_of_labour(person)

# set risk of maternal death to be high to capture some YLL
sim.modules['Labour'].current_parameters['prob_uterine_rupture'] = 1
sim.modules['Labour'].current_parameters['cfr_uterine_rupture'] = 1

sim.simulate(end_date=Date(2011, 1, 1))

# MATERNAL:
# get log
output = parse_log_file(sim.log_filepath)

# calculate YLD for the year of the simulation (2010)
yld_output = output['tlo.methods.healthburden']['yld_by_causes_of_disability']
yld_2010_df = yld_output.loc[yld_output['year'] == 2010]
yld = yld_2010_df['maternal'].sum()

# next calculate the total years of life lost to maternal causes in 2010
yll_output = output['tlo.methods.healthburden']['yll_by_causes_of_death_stacked']
yll_2010_df = yll_output.loc[yll_output['year'] == 2010]
yll = 0

# cycle over possible causes of maternal death and sum total YLL
for cause in ['ectopic_pregnancy', 'spontaneous_abortion', 'induced_abortion', 'uterine_rupture',
              'postpartum_sepsis', 'postpartum_haemorrhage', 'secondary_postpartum_haemorrhage',
              'severe_pre_eclampsia', 'eclampsia', 'severe_gestational_hypertension', 'antepartum_haemorrhage']:
    if cause in yll_2010_df.columns:
        yll += yll_2010_df[f'{cause}'].sum()

# combine the YLD and YLL in 2010 to estiamte the expected DALYs
expected_dalys = yld + yll
print('expected maternal dalys', expected_dalys)

dalys_ouput = output['tlo.methods.healthburden']['dalys_stacked']
dalys_2010 = dalys_ouput.loc[dalys_ouput['year'] == 2010]
dalys_from_simulation = dalys_2010['Maternal Disorders'].sum()

print('actual maternal dalys', dalys_from_simulation)

# NEONATAL:
# repeat
yld_output = output['tlo.methods.healthburden']['yld_by_causes_of_disability']
yld_2010_df = yld_output.loc[yld_output['year'] == 2010]
yld = 0

for cause in ['Retinopathy of Prematurity', 'Neonatal Encephalopathy', 'Neonatal Sepsis Long term Disability',
              'Preterm Birth Disability']:
    if cause in yld_2010_df.columns:
        yld += yld_2010_df[f'{cause}'].sum()

yll_output = output['tlo.methods.healthburden']['yll_by_causes_of_death_stacked']
yll_2010_df = yll_output.loc[yll_output['year'] == 2010]
yll = 0

for cause in ['early_onset_neonatal_sepsis', 'early_onset_sepsis', 'late_onset_neonatal_sepsis', 'late_onset_sepsis',
              'encephalopathy', 'preterm_other',
              'respiratory_distress_syndrome', 'neonatal_respiratory_depression',
              'congenital_heart_anomaly', 'limb_or_musculoskeletal_anomaly', 'urogenital_anomaly',
              'digestive_anomaly', 'other_anomaly']:
    if cause in yll_2010_df.columns:
        yll += yll_2010_df[f'{cause}'].sum()

expected_dalys = yld + yll
print('expected neonatal dalys', expected_dalys)

dalys_ouput = output['tlo.methods.healthburden']['dalys_stacked']
dalys_2010 = dalys_ouput.loc[dalys_ouput['year'] == 2010]
dalys_from_simulation = dalys_2010['Neonatal Disorders'].sum()

print('actual neonatal dalys', dalys_from_simulation)

# Tim's checks

# number of deaths:
deaths = output['tlo.methods.demography']['death']
num_deaths_mat_disorders = ((pd.to_datetime(deaths.date).dt.year == 2010) & (deaths.label == 'Maternal '
                                                                                             'Disorders')).sum()
# 49 deaths

# number of YLL (stacked onto year 2010)
yll = output['tlo.methods.healthburden']['yll_by_causes_of_death_stacked']
map_to_label_deaths = \
    pd.Series(output['tlo.methods.demography']['mapper_from_tlo_cause_to_common_label'].drop(
        columns={'date'}).loc[0]).to_dict()

sum_yll = yll.loc[(yll.year == 2010)].sum().drop(['sex', 'age_range', 'year'])
sum_yll.index = sum_yll.index.map(map_to_label_deaths)
yll_mat_disorders = sum_yll.groupby(level=0).sum()['Maternal Disorders']  # 2038 life-years
yll_mat_disorders_per_death = yll_mat_disorders / num_deaths_mat_disorders  # 40.1 YLL per death


yld = output['tlo.methods.healthburden']['yld_by_causes_of_disability']
map_to_label_disability = pd.Series(output['tlo.methods.healthburden']['mapper_from_tlo_cause_to_common_label'].drop(
    columns={'date'}).loc[0]).to_dict()
sum_yld = yld.loc[(yld.year == 2010)].sum().drop(['sex', 'age_range', 'year'])
sum_yld.index = sum_yld.index.map(map_to_label_disability)
yld_mat_disorders = sum_yld['Maternal Disorders']

dalys = output['tlo.methods.healthburden']['dalys_stacked']
dalys_mat_disorders = dalys['Maternal Disorders'].sum()

# Check it works!
assert np.isclose(dalys_mat_disorders, yld_mat_disorders + yll_mat_disorders)
