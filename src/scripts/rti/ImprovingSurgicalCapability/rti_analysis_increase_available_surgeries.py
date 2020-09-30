from pathlib import Path

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    symptommanager,
    rti,
)
import numpy as np
from matplotlib import pyplot as plt

log_config = {
    "filename": "rti_health_system_comparison",  # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.rti": logging.INFO,
        "tlo.methods.healthsystem": logging.DEBUG
    }
}
# The Resource files [NB. Working directory must be set to the root of TLO: TLOmodel]
resourcefilepath = Path('./resources')
# Establish the simulation object
yearsrun = 10
start_date = Date(year=2010, month=1, day=1)
end_date = Date(year=(2010 + yearsrun), month=1, day=1)
pop_size = 5000
nsim = 5

list_deaths_without_extra_surg = []
outputs_for_without_extra_surg = dict()
list_tot_dalys_no_extra = []
for i in range(0, nsim):

    sim_with_health_system = Simulation(start_date=start_date)

    sim_with_health_system.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        rti.RTI(resourcefilepath=resourcefilepath)
    )
    logfile_health_sys = sim_with_health_system.configure_logging(filename="LogFile")
    # create and run the simulation
    outputs_for_without_extra_surg[i] = logfile_health_sys
    params = sim_with_health_system.modules['RTI'].parameters
    params['allowed_interventions'] = []
    sim_with_health_system.make_initial_population(n=pop_size)
    sim_with_health_system.simulate(end_date=end_date)
    # parse the simulation logfiles to get the output dataframes
    log_df_with_health_system = parse_log_file(logfile_health_sys)
    deaths_with_med = log_df_with_health_system['tlo.methods.demography']['death']
    tot_death_with_med = len(deaths_with_med.loc[(deaths_with_med['cause'] != 'Other')])
    list_deaths_without_extra_surg.append(tot_death_with_med)
    dalys_df = log_df_with_health_system['tlo.methods.healthburden']['DALYS']
    males_data = dalys_df.loc[dalys_df['sex'] == 'M']
    YLL_males_data = males_data.filter(like='YLL_RTI').columns
    males_dalys = males_data[YLL_males_data].sum(axis=1) + \
                  males_data['YLD_RTI_rt_disability']
    females_data = dalys_df.loc[dalys_df['sex'] == 'F']
    YLL_females_data = females_data.filter(like='YLL_RTI').columns
    females_dalys = females_data[YLL_females_data].sum(axis=1) + \
                    females_data['YLD_RTI_rt_disability']

    tot_dalys = males_dalys.tolist() + females_dalys.tolist()
    list_tot_dalys_no_extra.append(tot_dalys)

outputs_for_with_extra_surg = dict()
list_deaths_with_extra_surg = []
list_tot_dalys_with_extra_surg = []

for i in range(0, nsim):
    sim_without_health_system = Simulation(start_date=start_date)
    sim_without_health_system.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        rti.RTI(resourcefilepath=resourcefilepath)
    )
    logfile_no_health_sys = sim_without_health_system.configure_logging(filename="LogFile")
    outputs_for_with_extra_surg[i] = logfile_no_health_sys
    params = sim_without_health_system.modules['RTI'].parameters
    params['allowed_interventions'] = ['include_spine_surgery', 'include_thoroscopy']
    sim_without_health_system.make_initial_population(n=pop_size)
    sim_without_health_system.simulate(end_date=end_date)
    log_df_without_health_system = parse_log_file(logfile_no_health_sys)
    deaths_without_med = log_df_without_health_system['tlo.methods.demography']['death']
    tot_death_without_med = len(deaths_without_med.loc[(deaths_without_med['cause'] != 'Other')])
    list_deaths_with_extra_surg.append(tot_death_without_med)
    dalys_df = log_df_without_health_system['tlo.methods.healthburden']['DALYS']
    males_data = dalys_df.loc[dalys_df['sex'] == 'M']
    YLL_males_data = males_data.filter(like='YLL_RTI').columns
    males_dalys = males_data[YLL_males_data].sum(axis=1) + \
                  males_data['YLD_RTI_rt_disability']
    females_data = dalys_df.loc[dalys_df['sex'] == 'F']
    YLL_females_data = females_data.filter(like='YLL_RTI').columns
    females_dalys = females_data[YLL_females_data].sum(axis=1) + \
                    females_data['YLD_RTI_rt_disability']

    tot_dalys = males_dalys.tolist() + females_dalys.tolist()
    list_tot_dalys_with_extra_surg.append(tot_dalys)

tot_dalys_in_sims_with_med = []
for list in list_tot_dalys_no_extra:
    tot_dalys_in_sims_with_med.append(sum(list))

tot_dalys_in_sims_without_med = []
for list in list_tot_dalys_with_extra_surg:
    tot_dalys_in_sims_without_med.append(sum(list))
plt.bar(np.arange(2), [np.mean(tot_dalys_in_sims_without_med), np.mean(tot_dalys_in_sims_with_med)])
plt.xticks(np.arange(2), ['Total DALYs'
                          '\n'
                          'without additional surgery',
                          'Total DALYs'
                          '\n'
                          ' with additional surgery'])
plt.title(f"Average total DALYS in simulations with and without additional available surgeries"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/IncreasingSurgicalCapacity/compare_mean_total_DALYS_with_without_extra_surg.png')
plt.clf()
plt.clf()
mean_deaths_no_surg = np.mean(list_deaths_without_extra_surg)
mean_deaths_with_surg = np.mean(list_deaths_with_extra_surg)
plt.bar(np.arange(2), [mean_deaths_no_surg, mean_deaths_with_surg])
plt.xticks([0, 1], ['Total deaths due to RTI'
                    '\n'
                    'without additional '
                    '\n'
                    'available surgeries',
                    'Total deaths due to RTI'
                    '\n'
                    'with additional '
                    '\n'
                    'available surgeries'])
plt.title(f"Average deaths in simulations with and without additional available surgeries"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/IncreasingSurgicalCapacity/compare_mean_total_deaths_with_without_extra_surg.png')
