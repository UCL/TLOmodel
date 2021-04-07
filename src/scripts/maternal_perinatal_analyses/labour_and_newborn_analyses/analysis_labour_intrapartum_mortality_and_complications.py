"""This analysis file produces all mortality outputs"""

import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    antenatal_care,
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager,
)

seed = 567

log_config = {
    "filename": "labour_incidence_analysis",   # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.demography": logging.INFO,
        "tlo.methods.labour": logging.DEBUG,
        "tlo.methods.healthsystem": logging.FATAL,
        "tlo.methods.hiv": logging.FATAL,
        "tlo.methods.newborn_outcomes": logging.DEBUG,
        "tlo.methods.antenatal_care": logging.DEBUG,
        "tlo.methods.pregnancy_supervisor": logging.DEBUG,
        "tlo.methods.postnatal_supervisor": logging.DEBUG,
    }
}

# %%
outputpath = Path("./outputs")
resourcefilepath = Path("./resources")

# Create name for log-file
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2011, 1, 1)
popsize = 10000

# add file handler for the purpose of logging
sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)
# service_availability = ['*']

allowed_interventions = ['prophylactic_labour_interventions',
                         'assessment_and_treatment_of_severe_pre_eclampsia',
                         'assessment_and_treatment_of_obstructed_labour',
                         'assessment_and_treatment_of_maternal_sepsis',
                         'assessment_and_treatment_of_hypertension',
                         'assessment_and_treatment_of_eclampsia',
                         'assessment_and_plan_for_referral_antepartum_haemorrhage',
                         'assessment_and_plan_for_referral_uterine_rupture',
                         'active_management_of_the_third_stage_of_labour',
                         'assessment_and_treatment_of_pph_retained_placenta',
                         'assessment_and_treatment_of_pph_uterine_atony']


# run the simulation
sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             contraception.Contraception(resourcefilepath=resourcefilepath),
             enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
             healthburden.HealthBurden(resourcefilepath=resourcefilepath),
             healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                       service_availability=['*']),
             symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
             labour.Labour(resourcefilepath=resourcefilepath),
             newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
             antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
             pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
             postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
             healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))

sim.make_initial_population(n=popsize)
"""params_lab = sim.modules['Labour'].parameters
params_preg = sim.modules['PregnancySupervisor'].parameters
params_post = sim.modules['PostnatalSupervisor'].parameters
params_nb = sim.modules['NewbornOutcomes'].parameters

# 0.07 # todo: need to edit linear models as parameters
params_lab['la_labour_equations']['antepartum_haem_death'] =\
    LinearModel(LinearModelType.MULTIPLICATIVE,
                0.07,
                Predictor().when('la_antepartum_haem_treatment & (__mode_of_delivery__ == "caesarean_section") & '
                                 '(__referral_timing_caesarean__== "prompt_referral")',
                                 params_lab['aph_prompt_treatment_effect_md']),
                Predictor().when('la_antepartum_haem_treatment & (__mode_of_delivery__ == "caesarean_section") & '
                                 '(__referral_timing_caesarean__== "delayed_referral")',
                                 params_lab['aph_delayed_treatment_effect_md']),
                Predictor('received_blood_transfusion', external=True).when(True,
                                                                            params_lab['aph_bt_treatment_effect_md']))

params_lab['cfr_severe_pre_eclamp'] = 0
params_lab['cfr_eclampsia'] = 0
params_lab['cfr_sepsis'] = 0
params_lab['cfr_uterine_rupture'] = 0
params_lab['cfr_pp_pph'] = 0
params_lab['cfr_pp_eclampsia'] = 0
params_lab['cfr_pp_sepsis'] = 0

params_preg['prob_ectopic_pregnancy_death'] = 0
params_preg['prob_induced_abortion_death'] = 0
params_preg['prob_spontaneous_abortion_death'] = 0
params_preg['prob_antepartum_haem_death'] = 0
params_preg['prob_antenatal_spe_death'] = 0
params_preg['prob_antenatal_ec_death'] = 0

params_nb['cfr_mild_enceph'] = 0.2
params_nb['cfr_moderate_enceph'] = 0.2
params_nb['cfr_severe_enceph'] = 0.2
params_nb['cfr_failed_to_transition'] = 0.2
params_nb['cfr_preterm_birth'] = 0.2
params_nb['cfr_neonatal_sepsis'] = 0.2
params_nb['cfr_congenital_anomaly'] = 0.2
params_nb['cfr_rds_preterm'] = 0.2

params_post['cfr_secondary_pph'] = 0
params_post['cfr_postnatal_sepsis'] = 0
params_post['cfr_early_onset_neonatal_sepsis'] = 0.2
params_post['cfr_late_neonatal_sepsis'] = 0.2
params_post['cfr_eclampsia_pn'] = 0
"""

df = sim.population.props

women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
df.loc[women_repro.index, 'is_pregnant'] = True
df.loc[women_repro.index, 'date_of_last_pregnancy'] = start_date
for person in women_repro.index:
    sim.modules['Labour'].set_date_of_labour(person)

sim.simulate(end_date=end_date)

log_df = parse_log_file(sim.log_filepath)

stats_incidence = log_df['tlo.methods.labour']['labour_summary_stats_incidence']
stats_incidence['year'] = pd.to_datetime(stats_incidence['date']).dt.year
stats_incidence.drop(columns='date', inplace=True)
stats_incidence.set_index(
        'year',
        drop=True,
        inplace=True
    )


stats_crude = log_df['tlo.methods.labour']['labour_summary_stats_crude_cases']
stats_crude['date'] = pd.to_datetime(stats_crude['date'])
stats_crude['year'] = stats_crude['date'].dt.year

stats_deliveries = log_df['tlo.methods.labour']['labour_summary_stats_delivery']
stats_deliveries['date'] = pd.to_datetime(stats_deliveries['date'])
stats_deliveries['year'] = stats_deliveries['date'].dt.year

stats_nb = log_df['tlo.methods.newborn_outcomes']['neonatal_summary_stats']
stats_nb['date'] = pd.to_datetime(stats_nb['date'])
stats_nb['year'] = stats_nb['date'].dt.year

stats_md = log_df['tlo.methods.labour']['labour_summary_stats_death']
stats_md['date'] = pd.to_datetime(stats_md['date'])
stats_md['year'] = stats_md['date'].dt.year

stats_preg = log_df['tlo.methods.pregnancy_supervisor']['ps_summary_statistics']
stats_preg['date'] = pd.to_datetime(stats_preg['date'])
stats_preg['year'] = stats_preg['date'].dt.year

stats_postnatal = log_df['tlo.methods.postnatal_supervisor']['postnatal_maternal_summary_stats']
stats_postnatal['date'] = pd.to_datetime(stats_postnatal['date'])
stats_postnatal['year'] = stats_postnatal['date'].dt.year

stats_postnatal_n = log_df['tlo.methods.postnatal_supervisor']['postnatal_neonatal_summary_stats']
stats_postnatal_n['date'] = pd.to_datetime(stats_postnatal_n['date'])
stats_postnatal_n['year'] = stats_postnatal_n['date'].dt.year

# =====================================================================================================================
deaths = log_df['tlo.methods.demography']['death']
deaths['date'] = pd.to_datetime(deaths['date'])
deaths['year'] = deaths['date'].dt.year

births = log_df['tlo.methods.demography']['on_birth']
births['date'] = pd.to_datetime(births['date'])
births['year'] = births['date'].dt.year

total_deaths_2011 = len(deaths.loc[(deaths.year == 2011)])
total_maternal_deaths = len(deaths.loc[(deaths.year == 2011) & (deaths.cause == 'maternal')])
total_neonatal = len(deaths.loc[(deaths.year == 2011) & (deaths.cause == 'neonatal')])

prop_of_total_deaths_maternal = (total_maternal_deaths/total_deaths_2011) * 100
prop_of_total_deaths_neonatal = (total_neonatal/total_deaths_2011) * 100

objects = ('Maternal Deaths', 'GBD Est.', 'Neonatal Deaths', 'GBD Est.')
y_pos = np.arange(len(objects))
plt.bar(y_pos, [prop_of_total_deaths_maternal, 0.98, prop_of_total_deaths_neonatal, 11], align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('% of total death contributed by cause')
plt.title('% of Total Yearly Deaths Attributed to Maternal and Neonatal Causes (2011)')
plt.show()

total_births = len(births.loc[(births.year == 2011)])
mmr_2011 = (total_maternal_deaths/total_births) * 100000
nmr_2011 = (total_neonatal/total_births) * 1000

objects = ('Model MMR', 'DHS Est.', 'GBD Est.')
y_pos = np.arange(len(objects))
plt.bar(y_pos, [mmr_2011, 657, 300], align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Maternal deaths/100,000 births')
plt.title('Maternal mortality rate in 2011')
plt.show()

objects = ('NMR', 'DHS Est.', 'GBD Est.(2015)')
y_pos = np.arange(len(objects))
plt.bar(y_pos, [nmr_2011, 31, 28.6], align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Neonatal deaths/1000 births')
plt.title('Neonatal mortality rate in 2011')
plt.show()
