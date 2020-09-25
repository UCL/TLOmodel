from pathlib import Path
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt
import os

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    contraception,
    healthburden,
    healthsystem,
    enhanced_lifestyle,
    dx_algorithm_child,
    healthseekingbehaviour,
    symptommanager,
    antenatal_care,
    labour,
    newborn_outcomes,
    pregnancy_supervisor,
    epi,
    measles
)

outputpath = Path("./outputs")  # folder for convenience of storing outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# To reproduce the results, you must set the seed for the Simulation instance. The Simulation
# will seed the random number generators for each module when they are registered.
# If a seed argument is not given, one is generated. It is output in the log and can be
# used to reproduce results of a run
seed = 100

log_config = {
    'filename': 'measles_analysis',   # The name of the output file (a timestamp will be appended).
    'directory': './outputs',  # The default output path is `./outputs`. Change it here, if necessary
    'custom_levels': {  # Customise the output of specific loggers. They are applied in order:
        '*': logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        'tlo.methods.measles': logging.INFO,
        'tlo.methods.demography': logging.INFO,
    }
}

start_date = Date(2010, 1, 1)
end_date = Date(2015, 12, 31)
pop_size = 500

# Path to the resource files used by the disease and intervention methods
resources = Path('./resources')


# ------------------------------------- BASELINE  ------------------------------------- #
sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

# # We register all modules in a single call to the register method, calling once with multiple
# # objects. This is preferred to registering each module in multiple calls because we will be
# # able to handle dependencies if modules are registered together
sim.register(
    demography.Demography(resourcefilepath=resources),
    healthsystem.HealthSystem(
        resourcefilepath=resources,
        service_availability=['*'],
        mode_appt_constraints=0,
        ignore_cons_constraints=True,
        ignore_priority=True,
        capabilities_coefficient=1.0,
        disable=False,
    ),
    symptommanager.SymptomManager(resourcefilepath=resources),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resources),
    dx_algorithm_child.DxAlgorithmChild(),
    healthburden.HealthBurden(resourcefilepath=resources),
    contraception.Contraception(resourcefilepath=resources),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resources),
    labour.Labour(resourcefilepath=resources),
    newborn_outcomes.NewbornOutcomes(resourcefilepath=resources),
    antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resources),
    pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resources),
    epi.Epi(resourcefilepath=resources),
    measles.Measles(resourcefilepath=resources),
)

# create and run the simulation
sim.make_initial_population(n=pop_size)
sim.simulate(end_date=end_date)

# parse the simulation logfile to get the output dataframes
log_df = parse_log_file(sim.log_filepath)

# # ------------------------------------- BASELINE MODEL OUTPUTS  ------------------------------------- #

baseline_measles = log_df['tlo.methods.measles']['incidence']['inc_1000py']
model_date = log_df['tlo.methods.measles']['incidence']['date']

baseline_measles_age = log_df['tlo.methods.measles']['measles_incidence_age_range']

# calculate death rate
deaths_df = log_df['tlo.methods.demography']['death']
deaths_df['year'] = pd.to_datetime(deaths_df['date']).dt.year
baseline_deaths = deaths_df.loc[deaths_df['cause'].str.startswith('measles')].groupby('year').size()


# ------------------------------------- STOP VACCINES FROM 2019  ------------------------------------- #
# vaccines still available from 2010-2018 but not through HSI from 2019 onwards
# no other HSI running throughout simulation
log_config = {
    'filename': 'measles_analysis_stop_vaccines',  # The name of the output file (a timestamp will be appended).
    'directory': './outputs',  # The default output path is `./outputs`. Change it here, if necessary
    'custom_levels': {  # Customise the output of specific loggers. They are applied in order:
        '*': logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        'tlo.methods.measles': logging.INFO,
        'tlo.methods.demography': logging.INFO,
    }
}

sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

sim.register(
    demography.Demography(resourcefilepath=resources),
    healthsystem.HealthSystem(
        resourcefilepath=resources,
        disable_and_reject_all=True,  # disable healthsystem and no HSI runs
    ),
    symptommanager.SymptomManager(resourcefilepath=resources),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resources),
    dx_algorithm_child.DxAlgorithmChild(),
    healthburden.HealthBurden(resourcefilepath=resources),
    contraception.Contraception(resourcefilepath=resources),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resources),
    labour.Labour(resourcefilepath=resources),
    newborn_outcomes.NewbornOutcomes(resourcefilepath=resources),
    antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resources),
    pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resources),
    epi.Epi(resourcefilepath=resources),
    measles.Measles(resourcefilepath=resources),
)

# create and run the simulation
sim.make_initial_population(n=pop_size)
sim.simulate(end_date=end_date)

# parse the simulation logfile to get the output dataframes
log_df2 = parse_log_file(sim.log_filepath)
#
# # ------------------------------------- STOP VACCINE MODEL OUTPUTS  ------------------------------------- #

stop_vaccine_measles = log_df2['tlo.methods.measles']['incidence']['inc_1000py']

stop_vaccine_measles_age = log_df2['tlo.methods.measles']['measles_incidence_age_range']

# calculate death rate
deaths_df = log_df2['tlo.methods.demography']['death']
deaths_df['year'] = pd.to_datetime(deaths_df['date']).dt.year
stop_vaccine_deaths = deaths_df.loc[deaths_df['cause'].str.startswith('measles')].groupby('year').size()

# ------------------------------------- NO VACCINES AVAILABLE  ------------------------------------- #
# vaccines still available from 2010-2018 but not through HSI from 2019 onwards
# no other HSI running throughout simulation
log_config = {
    'filename': 'measles_analysis_no_vaccines',  # The name of the output file (a timestamp will be appended).
    'directory': './outputs',  # The default output path is `./outputs`. Change it here, if necessary
    'custom_levels': {  # Customise the output of specific loggers. They are applied in order:
        '*': logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        'tlo.methods.measles': logging.INFO,
        'tlo.methods.demography': logging.INFO,
    }
}

sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

sim.register(
    demography.Demography(resourcefilepath=resources),
    healthsystem.HealthSystem(
        resourcefilepath=resources,
        disable_and_reject_all=True,  # disable healthsystem and no HSI runs
    ),
    symptommanager.SymptomManager(resourcefilepath=resources),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resources),
    dx_algorithm_child.DxAlgorithmChild(),
    healthburden.HealthBurden(resourcefilepath=resources),
    contraception.Contraception(resourcefilepath=resources),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resources),
    labour.Labour(resourcefilepath=resources),
    newborn_outcomes.NewbornOutcomes(resourcefilepath=resources),
    antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resources),
    pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resources),
    epi.Epi(resourcefilepath=resources),
    measles.Measles(resourcefilepath=resources),
)

# change measles vaccine coverage to 0
sim.modules['Epi'].parameters['baseline_coverage'].MCV1 = 0
sim.modules['Epi'].parameters['baseline_coverage'].MCV2 = 0

sim.modules['Epi'].parameters['district_vaccine_coverage'].MCV1 = 0
sim.modules['Epi'].parameters['district_vaccine_coverage'].MCV2 = 0
sim.modules['Epi'].parameters['district_vaccine_coverage'].MCV1_MR1 = 0
sim.modules['Epi'].parameters['district_vaccine_coverage'].MCV2_MR2 = 0

# create and run the simulation
sim.make_initial_population(n=pop_size)
sim.simulate(end_date=end_date)

# parse the simulation logfile to get the output dataframes
log_df3 = parse_log_file(sim.log_filepath)

# ------------------------------------- NO VACCINES AVAILABLE OUTPUTS  ------------------------------------- #

no_vaccine_measles = log_df3['tlo.methods.measles']['incidence']['inc_1000py']

no_vaccine_measles_age = log_df3['tlo.methods.measles']['measles_incidence_age_range']

# calculate death rate
deaths_df = log_df3['tlo.methods.demography']['death']
deaths_df['year'] = pd.to_datetime(deaths_df['date']).dt.year
no_vaccine_deaths = deaths_df.loc[deaths_df['cause'].str.startswith('measles')].groupby('year').size()

# ------------------------------------- PLOTS  ------------------------------------- #

plt.style.use('ggplot')

# Measles incidence
plt.subplot(221)  # numrows, numcols, fignum
plt.plot(model_date, baseline_measles)
plt.plot(model_date, stop_vaccine_measles)
plt.plot(model_date, no_vaccine_measles)
plt.title('Measles incidence')
plt.xlabel('Date')
plt.ylabel('Incidence per 1000py')
plt.xticks(rotation=90)
plt.legend(['Baseline', 'Stop vaccination 2019', 'No vaccination'], bbox_to_anchor=(1.04, 1), loc='upper left')
# plt.savefig(outputpath / ("Measles_incidence_scenarios" + datestamp + ".pdf"), format='pdf')
plt.show()

# Measles deaths
plt.subplot(222)  # numrows, numcols, fignum
plt.plot(model_date, baseline_deaths)
plt.plot(model_date, stop_vaccine_deaths)
plt.plot(model_date, no_vaccine_deaths)
plt.title('Measles incidence')
plt.xlabel('Date')
plt.ylabel('Number of deaths')
plt.xticks(rotation=90)
plt.legend(['Baseline', 'Stop vaccination 2019', 'No vaccination'], bbox_to_anchor=(1.04, 1), loc='upper left')
# plt.savefig(outputpath / ("Measles_deaths_scenarios" + datestamp + ".pdf"), format='pdf')
plt.show()

# Measles cases age distribution
plt.subplot(223)  # numrows, numcols, fignum
plt.plot(model_date, baseline_measles_age)
plt.title('Age distribution')
plt.xlabel('Date')
plt.ylabel('Proportion of cases')
plt.xticks(rotation=90)
plt.legend(['Baseline'], bbox_to_anchor=(1.04, 1), loc='upper left')
# plt.savefig(outputpath / ("Measles_age_distribution_scenarios" + datestamp + ".pdf"), format='pdf')
plt.show()




