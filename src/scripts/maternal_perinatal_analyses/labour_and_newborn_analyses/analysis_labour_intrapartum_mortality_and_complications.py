"""This analysis file produces all mortality outputs"""

import datetime
from pathlib import Path

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
    pregnancy_supervisor,
    symptommanager, male_circumcision, hiv, tb, postnatal_supervisor
)

seed = 567

log_config = {
    "filename": "labour_incidence_analysis",   # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
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
end_date = Date(2012, 1, 1)
popsize = 100

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
                 #healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*']),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 male_circumcision.male_circumcision(resourcefilepath=resourcefilepath),
                 hiv.hiv(resourcefilepath=resourcefilepath),
                 tb.tb(resourcefilepath=resourcefilepath),
                 antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))

sim.make_initial_population(n=popsize)
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

x='y'
