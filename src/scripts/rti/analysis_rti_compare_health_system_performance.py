from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    dx_algorithm_adult,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    rti,
    symptommanager,
)

# =============================== Analysis description ========================================================
# What I am trying to do here is to see the health benefits achieved by the health system, I am running the model
# with service availability ['*'] and then with service availability []. This will give us an estimate for how the
# health system is performing, by comparing the deaths and dalys with the health system to the deaths and dalys without

# Set up the logging
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
# Create the conditions for the simulations, i.e. how long it runs for, how many people are involved and the
# number of simulations this analysis will be run for, to try and account for some of the variation between simulations
yearsrun = 10
start_date = Date(year=2010, month=1, day=1)
end_date = Date(year=(2010 + yearsrun), month=1, day=1)
pop_size = 50000
nsim = 2
# create empty lists to store the number of deaths with a health system an without a health system and a dictionary
# to store the log files in
list_deaths_with_med = []
outputs_for_with_health_system = dict()
list_tot_dalys_med = []
# iterate over the number of simulations
for i in range(0, nsim):
    # Create sim object and register the relevant modules
    # Do not limit the service availability
    sim_with_health_system = Simulation(start_date=start_date)
    sim_with_health_system.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath),
        dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        rti.RTI(resourcefilepath=resourcefilepath)
    )
    # name the logfile
    logfile_health_sys = sim_with_health_system.configure_logging(filename="LogFile")
    # store the logfile
    outputs_for_with_health_system[i] = logfile_health_sys
    # make initial population
    sim_with_health_system.make_initial_population(n=pop_size)
    # run simulation
    sim_with_health_system.simulate(end_date=end_date)
    # parse the simulation logfiles to get the output dataframes
    log_df_with_health_system = parse_log_file(logfile_health_sys)
    # get a logfile with the deaths of the the simulation with the health system
    deaths_with_med = log_df_with_health_system['tlo.methods.demography']['death']
    # count how many deaths there were in this simulation that were caused by something other than the demography
    rti_deaths = log_df_with_health_system['tlo.methods.demography']['death']
    rti_causes_of_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
    # calculate the total number of rti related deaths
    tot_death_with_med = len(rti_deaths.loc[rti_deaths['cause'].isin(rti_causes_of_deaths)])
    # Store the deaths from RTI in this simulation in the list of deaths with a health system
    list_deaths_with_med.append(tot_death_with_med)
    # Get the DALYs produced in the sim
    dalys_df = log_df_with_health_system['tlo.methods.healthburden']['dalys']
    # DALYs are split into gender and age ranges, therefor to get total DALYs we need to add them together
    # Get male data
    males_data = dalys_df.loc[dalys_df['sex'] == 'M']
    # Get YLL data for all produced by RTI
    YLL_males_data = males_data.filter(like='YLL_RTI').columns
    # Calculate DALYs as YLD + YLL
    males_dalys = males_data[YLL_males_data].sum(axis=1) + males_data['YLD_RTI_rt_disability']
    # Do the above with the female data
    females_data = dalys_df.loc[dalys_df['sex'] == 'F']
    YLL_females_data = females_data.filter(like='YLL_RTI').columns
    females_dalys = females_data[YLL_females_data].sum(axis=1) + females_data['YLD_RTI_rt_disability']
    # Total DALYs in sim is male DALYs plus female DALYs
    tot_dalys = males_dalys.tolist() + females_dalys.tolist()
    # Append this information to the list of total DALYs with a health system
    list_tot_dalys_med.append(tot_dalys)

# Create empty lists to store the outputs of the simulations without a health system in
outputs_for_without_health_system = dict()
list_deaths_no_med = []
list_tot_dalys_no_med = []

for i in range(0, nsim):
    # Create the simulation object and register the relevant modules
    # Make the health system have no service availability
    sim_without_health_system = Simulation(start_date=start_date)
    sim_without_health_system.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=[]),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath),
        dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        rti.RTI(resourcefilepath=resourcefilepath)
    )
    # Get the logfile and store it
    logfile_no_health_sys = sim_without_health_system.configure_logging(filename="LogFile")
    outputs_for_without_health_system[i] = logfile_no_health_sys
    # make initial population
    sim_without_health_system.make_initial_population(n=pop_size)
    # run the sim
    sim_without_health_system.simulate(end_date=end_date)
    # parse the log file
    log_df_without_health_system = parse_log_file(logfile_no_health_sys)
    # Exactly the same process as the above groups of simulations except that the results are stored in a different
    # list
    rti_deaths = log_df_without_health_system['tlo.methods.demography']['death']
    rti_causes_of_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
    # calculate the total number of rti related deaths
    tot_death_without_med = len(rti_deaths.loc[rti_deaths['cause'].isin(rti_causes_of_deaths)])
    list_deaths_no_med.append(tot_death_without_med)
    dalys_df = log_df_without_health_system['tlo.methods.healthburden']['dalys']
    males_data = dalys_df.loc[dalys_df['sex'] == 'M']
    YLL_males_data = males_data.filter(like='YLL_RTI').columns
    males_dalys = males_data[YLL_males_data].sum(axis=1) + males_data['YLD_RTI_rt_disability']
    females_data = dalys_df.loc[dalys_df['sex'] == 'F']
    YLL_females_data = females_data.filter(like='YLL_RTI').columns
    females_dalys = females_data[YLL_females_data].sum(axis=1) + females_data['YLD_RTI_rt_disability']
    tot_dalys = males_dalys.tolist() + females_dalys.tolist()
    list_tot_dalys_no_med.append(tot_dalys)
# ============================================ Plot results =========================================================
# Plot the mean total DALYs with and without a health system
# Add up all the DALYs in the sims with a health system and without
tot_dalys_in_sims_with_med = []
for list in list_tot_dalys_med:
    tot_dalys_in_sims_with_med.append(sum(list))

tot_dalys_in_sims_without_med = []
for list in list_tot_dalys_no_med:
    tot_dalys_in_sims_without_med.append(sum(list))
# Percent reduction in DALYs
percent_change_dalys = \
    np.round(((np.mean(tot_dalys_in_sims_with_med) - np.mean(tot_dalys_in_sims_without_med)) /
              np.mean(tot_dalys_in_sims_without_med)) * 100, 2)
# Plot the output in a bar chart
plt.bar(np.arange(2), [np.mean(tot_dalys_in_sims_without_med), np.mean(tot_dalys_in_sims_with_med)],
        color='lightsteelblue')
plt.xticks(np.arange(2), ['Total DALYs \nwithout health system', 'Total DALYs \nwith health system'])
plt.title(f"Average total DALYS in simulations with and without health system."
          f"\n"
          f"Health system resulted in a {percent_change_dalys}% change in DALYs"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/HealthSystemComparison/compare_mean_total_DALYS_with_without_health_sys.png')
plt.clf()
# Plot the mean total deaths due to road traffic injuries with and without a health system
# Get the average number of deaths with and without a health system
mean_deaths_no_med = np.mean(list_deaths_no_med)
mean_deaths_with_med = np.mean(list_deaths_with_med)
# Percent reduction in deaths
percent_change_deaths = \
    np.round(((np.mean(list_deaths_with_med) - np.mean(list_deaths_no_med)) /
              np.mean(list_deaths_no_med)) * 100, 2)
# Plot this in a bar chart
plt.bar(np.arange(2), [mean_deaths_no_med, mean_deaths_with_med],
        color='lightsalmon')
plt.xticks(np.arange(2), ['Total deaths due to RTI'
                          '\n'
                          'without Health system',
                          'Total deaths due to RTI'
                          '\n'
                          'with Health system'])
plt.title(f"Average deaths in simulations with and without health system"
          f"\n"
          f"Health system resulted in a {percent_change_deaths}% change in deaths"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/HealthSystemComparison/compare_mean_total_deaths_with_without_health_sys.png')
plt.clf()
