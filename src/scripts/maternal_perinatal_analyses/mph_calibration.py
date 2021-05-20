import datetime
import os
from pathlib import Path

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    care_of_women_during_pregnancy,
    contraception,
    demography,
    depression,
    dx_algorithm_adult,
    dx_algorithm_child,
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
    ncds
)


seed = 567

seed = 1111

# The resource files
try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = Path('./resources')

start_date = Date(2010, 1, 1)


log_config = {
        "filename": "calibration_test",  # The name of the output file (a timestamp will be appended).
        "directory": "./outputs",
        # The default output path is `./outputs`. Change it here, if necessary
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

sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             contraception.Contraception(resourcefilepath=resourcefilepath),
             enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
             healthburden.HealthBurden(resourcefilepath=resourcefilepath),
             healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                       service_availability=['*']), #TODO: no restrictions
             ncds.Ncds(resourcefilepath=resourcefilepath),
             newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
             pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
             care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
             symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
             labour.Labour(resourcefilepath=resourcefilepath),
             postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
             healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
             malaria.Malaria(resourcefilepath=resourcefilepath),
             hiv.Hiv(resourcefilepath=resourcefilepath),
             dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath),
             dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
             depression.Depression(resourcefilepath=resourcefilepath),
                 )


sim.make_initial_population(n=1000)
sim.simulate(end_date=Date(2011, 1, 1))

# ================================= Maternal Mortality Ratio =========================================================
log_df = parse_log_file(sim.log_filepath)

# define the log DFs required
deaths = log_df['tlo.methods.demography']['death']
deaths['date'] = pd.to_datetime(deaths['date'])
deaths['year'] = deaths['date'].dt.year

live_births = log_df['tlo.methods.labour']['live_birth']
live_births['date'] = pd.to_datetime(live_births['date'])
live_births['year'] = live_births['date'].dt.year

live_births = log_df['tlo.methods.newborn_outcomes']['live_birth']
live_births['date'] = pd.to_datetime(live_births['date'])
live_births['year'] = live_births['date'].dt.year





# TODO: DOCFR
# TODO: CFR PPH
# TODO: % DEATHS DUE TO PPH
# TODO: CFR SEPSIS
# TODO: % DEATHS DUE TO SEPSIS
# TODO: CFR SPE/EC
# TODO: % DEATHS DUE TO SPE/EC
# TODO: % DEATHS DUE TO INDIRECT CAUSES

#
#
# TODO: SBR (AN/IP)
# TODO: NMR
# TODO : FACILITY DELIVERY RATE
# TODO: HOSPITAL VS HEALTH CENTRE
# TODO: ANC1, ANC4+, MEDIAN FIRST VISIT MONTH, EARLY ANC1
# TODO: PNC1 MATERNAL, PNC1 NEWBORNS
# TODO: CAESAREAN SECTION RATE

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
