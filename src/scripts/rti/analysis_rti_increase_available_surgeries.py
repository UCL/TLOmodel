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
    dx_algorithm_adult,
    dx_algorithm_child
)
import numpy as np
from matplotlib import pyplot as plt
# =============================== Analysis description ========================================================
# What I am trying to do here is to see the health benefits achieved in the population if we include additional
# surgeries, specifically I found a few references to the lack of surgical provided for spinal cord injuries in
# Malawi e.g. see https://journals.sagepub.com/doi/pdf/10.1177/0049475518808969
# I also found reference to the limited availability of thoroscopy in Malawi
# This is a file comparing simulations where we don't include these surgeries and where we do include them
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
pop_size = 50000
nsim = 2
# Create list to store outputs in for the model run without additional surgeries
list_deaths_without_extra_surg = []
list_tot_dalys_no_extra = []
for i in range(0, nsim):
    # Create and run simulations for the normal health systems
    sim_with_health_system = Simulation(start_date=start_date)
    # Register the modules
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
    # Name the logfile
    logfile_health_sys = sim_with_health_system.configure_logging(filename="LogFile")
    # create and run the simulation
    params = sim_with_health_system.modules['RTI'].parameters
    # Allow no extra interventions
    params['allowed_interventions'] = []
    # Create the initial population
    sim_with_health_system.make_initial_population(n=pop_size)
    # Run the simulation
    sim_with_health_system.simulate(end_date=end_date)
    # parse the simulation logfiles to get the output dataframes
    log_df_with_health_system = parse_log_file(logfile_health_sys)
    # get the number of road traffic injury related deaths from the sim
    rti_deaths = log_df_with_health_system['tlo.methods.demography']['death']
    rti_causes_of_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
    # calculate the total number of rti related deaths
    tot_death_with_med = len(rti_deaths.loc[rti_deaths['cause'].isin(rti_causes_of_deaths)])
    # Store the number of DALYs with the normal health system
    list_deaths_without_extra_surg.append(tot_death_with_med)
    # Get the dataframe that logs dalys
    dalys_df = log_df_with_health_system['tlo.methods.healthburden']['dalys']
    # Get the dalys that occured in males
    males_data = dalys_df.loc[dalys_df['sex'] == 'M']
    # Get the male YLL data
    YLL_males_data = males_data.filter(like='YLL_RTI').columns
    # Calculate DALYs in males
    males_dalys = males_data[YLL_males_data].sum(axis=1) + males_data['YLD_RTI_rt_disability']
    # Get the dalys that occured in females
    females_data = dalys_df.loc[dalys_df['sex'] == 'F']
    # Get the female YLL data
    YLL_females_data = females_data.filter(like='YLL_RTI').columns
    # Calculate DALYs in females
    females_dalys = females_data[YLL_females_data].sum(axis=1) + females_data['YLD_RTI_rt_disability']
    # Calculate total DALYs in sim
    tot_dalys = males_dalys.tolist() + females_dalys.tolist()
    # Store the number of DALYs experienced this sim
    list_tot_dalys_no_extra.append(sum(tot_dalys))

# Create lists to store the outputs of the simulations when thoroscopy and spinal cord surgery is included
list_deaths_with_extra_surg = []
list_tot_dalys_with_extra_surg = []

for i in range(0, nsim):
    # Create and run the simulations for the health system when we include spinal cord surgery and thoroscopy
    sim_without_health_system = Simulation(start_date=start_date)
    # Register the modules
    sim_without_health_system.register(
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
    # Name the logfile
    logfile_no_health_sys = sim_without_health_system.configure_logging(filename="LogFile")
    # Include spinal cord surgery and thoroscopy in allowed interventions
    params = sim_without_health_system.modules['RTI'].parameters
    params['allowed_interventions'] = ['include_spine_surgery', 'include_thoroscopy']
    # Create the initial population
    sim_without_health_system.make_initial_population(n=pop_size)
    # Run the simulation
    sim_without_health_system.simulate(end_date=end_date)
    # Parse the logfile
    log_df_without_health_system = parse_log_file(logfile_no_health_sys)
    rti_deaths = log_df_without_health_system['tlo.methods.demography']['death']
    rti_causes_of_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
    # calculate the total number of rti related deaths
    tot_death_without_med = len(rti_deaths.loc[rti_deaths['cause'].isin(rti_causes_of_deaths)])
    # Store the total number of deaths in the simulation with the included interventions
    list_deaths_with_extra_surg.append(tot_death_without_med)
    # get the data which stores the DALYs in the simulation
    dalys_df = log_df_without_health_system['tlo.methods.healthburden']['dalys']
    # Get the dalys experienced by males
    males_data = dalys_df.loc[dalys_df['sex'] == 'M']
    # get the YLL experienced by males in this sim
    YLL_males_data = males_data.filter(like='YLL_RTI').columns
    # Calculate the DALYs experienced by males
    males_dalys = males_data[YLL_males_data].sum(axis=1) + males_data['YLD_RTI_rt_disability']
    # Get the dalys experienced by females
    females_data = dalys_df.loc[dalys_df['sex'] == 'F']
    # get the YLL experienced by feales in this sim
    YLL_females_data = females_data.filter(like='YLL_RTI').columns
    # Calculate the DALYs experienced by feales
    females_dalys = females_data[YLL_females_data].sum(axis=1) + females_data['YLD_RTI_rt_disability']
    # Calculate the total DALYs experienced in the sim
    tot_dalys = males_dalys.tolist() + females_dalys.tolist()
    # Store the total dalys experienced in this sim
    list_tot_dalys_with_extra_surg.append(sum(tot_dalys))

# Create lists to store the outputs of the simulations where we only include spinal cord surgery
list_deaths_with_spinal_surg = []
list_tot_dalys_with_spinal_surg = []

for i in range(0, nsim):
    # Create and run the simulations for the health system when we include spinal cord surgery
    sim_without_health_system = Simulation(start_date=start_date)
    # Register modules
    sim_without_health_system.register(
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
    # Name logfile
    logfile_no_health_sys = sim_without_health_system.configure_logging(filename="LogFile")
    # Include spinal cold surgery in allowed interventions
    params = sim_without_health_system.modules['RTI'].parameters
    params['allowed_interventions'] = ['include_spine_surgery']
    # Create the initial population
    sim_without_health_system.make_initial_population(n=pop_size)
    # Run the simulation
    sim_without_health_system.simulate(end_date=end_date)
    # Parse the logfile
    log_df_without_health_system = parse_log_file(logfile_no_health_sys)
    rti_deaths = log_df_without_health_system['tlo.methods.demography']['death']
    rti_causes_of_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
    # calculate the total number of rti related deaths
    tot_death_without_med = len(rti_deaths.loc[rti_deaths['cause'].isin(rti_causes_of_deaths)])
    # Store the number of deaths in the simulation with the included spinal cord surgery
    list_deaths_with_spinal_surg.append(tot_death_without_med)
    # Get the dalys dataframe
    dalys_df = log_df_without_health_system['tlo.methods.healthburden']['dalys']
    # get male daly data
    males_data = dalys_df.loc[dalys_df['sex'] == 'M']
    # get yll in males
    YLL_males_data = males_data.filter(like='YLL_RTI').columns
    # calculate male DALYs
    males_dalys = males_data[YLL_males_data].sum(axis=1) + males_data['YLD_RTI_rt_disability']
    # get female daly data
    females_data = dalys_df.loc[dalys_df['sex'] == 'F']
    # get yll in females
    YLL_females_data = females_data.filter(like='YLL_RTI').columns
    # calculate dalys in females
    females_dalys = females_data[YLL_females_data].sum(axis=1) + females_data['YLD_RTI_rt_disability']
    # calculate total dalys
    tot_dalys = males_dalys.tolist() + females_dalys.tolist()
    # store total dalys in sim
    list_tot_dalys_with_spinal_surg.append(sum(tot_dalys))

# Create lists to store the outputs for simulations with thoroscopy
list_deaths_with_thoroscopy = []
list_tot_dalys_with_thoroscopy = []

for i in range(0, nsim):
    # Create and run the simulations for the health system when we include thoroscopy
    sim_without_health_system = Simulation(start_date=start_date)
    # Register the modules
    sim_without_health_system.register(
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
    # name logfile
    logfile_no_health_sys = sim_without_health_system.configure_logging(filename="LogFile")
    # allow thoroscopy in simulation
    params = sim_without_health_system.modules['RTI'].parameters
    params['allowed_interventions'] = ['include_thoroscopy']
    # Create initial population
    sim_without_health_system.make_initial_population(n=pop_size)
    # run simulation
    sim_without_health_system.simulate(end_date=end_date)
    # parse logfile
    log_df_without_health_system = parse_log_file(logfile_no_health_sys)
    # get the deaths dataframe
    rti_deaths = log_df_without_health_system['tlo.methods.demography']['death']
    rti_causes_of_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
    # calculate the total number of rti related deaths
    tot_death_without_med = len(rti_deaths.loc[rti_deaths['cause'].isin(rti_causes_of_deaths)])
    # store the number of deaths in the simulation with the included thoroscopy
    list_deaths_with_thoroscopy.append(tot_death_without_med)
    # get the dalys dataframe
    dalys_df = log_df_without_health_system['tlo.methods.healthburden']['dalys']
    # get the male daly data
    males_data = dalys_df.loc[dalys_df['sex'] == 'M']
    # get male yll
    YLL_males_data = males_data.filter(like='YLL_RTI').columns
    # calculate dalys in males
    males_dalys = males_data[YLL_males_data].sum(axis=1) + males_data['YLD_RTI_rt_disability']
    # get female daly data
    females_data = dalys_df.loc[dalys_df['sex'] == 'F']
    # get female yll
    YLL_females_data = females_data.filter(like='YLL_RTI').columns
    # calculate female dalys
    females_dalys = females_data[YLL_females_data].sum(axis=1) + females_data['YLD_RTI_rt_disability']
    # calculate total dalys
    tot_dalys = males_dalys.tolist() + females_dalys.tolist()
    # calculate total dalys
    list_tot_dalys_with_thoroscopy.append(sum(tot_dalys))

# Get outputs from simulations in a usable form start with DALYs///
# Get average dalys per sim for the simulation with the original health system
average_dalys_per_sims_orig = np.mean(list_tot_dalys_no_extra)
# Get average dalys per sim for the simulation with the health system with spinal cord surgery and thoroscopy
average_dalys_per_sim_with_both = np.mean(list_tot_dalys_with_extra_surg)
# Calculate the average reduction in dalys for including spinal cord surgery and thoroscopy
average_daly_change_both = np.round(
    ((average_dalys_per_sim_with_both - average_dalys_per_sims_orig) /
     average_dalys_per_sims_orig) * 100, 2)
# Get the average dalys per sim for the simulation with the health system with spinal cord surgery
average_dalys_with_spinal_surg = np.mean(list_tot_dalys_with_spinal_surg)
# Calculate the average reduction in dalys for including spinal cord surgery
average_daly_change_spinal = np.round(((average_dalys_with_spinal_surg - average_dalys_per_sims_orig) /
                                      average_dalys_per_sims_orig) * 100, 2)
# Get the average dalys per sim for the simulation with the health system with thoroscopy
average_dalys_with_thoroscopy = np.mean(list_tot_dalys_with_thoroscopy)
# Calculate the average reduction in dalys for including spinal cord surgery
average_daly_reduction_thoro = np.round(((average_dalys_with_thoroscopy - average_dalys_per_sims_orig) /
                                         average_dalys_per_sims_orig) * 100, 2)
# plot data for simulations without additional surgery and with additional surgery in a bar chart
plt.bar(np.arange(2), [average_dalys_per_sims_orig, average_dalys_per_sim_with_both],
        color=['lightsteelblue', 'lightsalmon'])
plt.ylabel('Average DALYs')
plt.xticks(np.arange(2), ['Average DALYs'
                          '\n'
                          'without additional surgery',
                          'Average DALYs'
                          '\n'
                          'with additional surgery'])
plt.title(f"Average DALYS in simulations with and without additional available surgeries"
          f"\n"
          f"including spinal cord surgery and thoroscopy altered average DALYs by "
          f"{np.round(average_daly_change_both, 2)}%"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/IncreasingSurgicalCapacity/compare_mean_total_DALYS_with_without_extra_surg.png',
            bbox_inches='tight')
plt.clf()
# plot data for simulations without additional surgery and with seperate runs for spinal cord surgery and
# thoroscopy in a bar chart
plt.bar(np.arange(3), [average_dalys_per_sims_orig,
                       average_dalys_with_spinal_surg,
                       average_dalys_with_thoroscopy],
        color=['lightsteelblue', 'lightsalmon', 'lemonchiffon'])
plt.xticks(np.arange(3), ['Average DALYs,'
                          '\n'
                          'original'
                          '\n'
                          'health system',
                          'Average DALYs,'
                          '\n'
                          'with spinal'
                          '\ncord surgery',
                          'Average DALYs,'
                          '\n'
                          'with thoroscopy'
                          ])
plt.ylabel('Average DALYs')
plt.title(f"Average DALYs in simulations with and additional available surgeries."
          f"\n"
          f"Including spinal cord surgery altered average DALYs by "
          f"{average_daly_change_spinal}%."
          f"\n"
          f"Including thoroscopy altered average DALYs by "
          f"{average_daly_reduction_thoro}%."
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/IncreasingSurgicalCapacity/compare_mean_total_DALYS_with_extra_surg_split.png',
            bbox_inches='tight')
plt.clf()
# Calculate on the effect of including each type of surgery on deaths
# get mean deaths without surgery
mean_deaths_no_surg = np.mean(list_deaths_without_extra_surg)
# get mean deaths with extra surgery
mean_deaths_with_surg = np.mean(list_deaths_with_extra_surg)
# calculate the percentage chang in deaths
average_death_change_both = np.round(
    ((mean_deaths_with_surg - mean_deaths_no_surg) / mean_deaths_no_surg) * 100, 2)
# calculate mean deaths when including spinal cord surgery
mean_deaths_with_spine = np.mean(list_deaths_with_spinal_surg)
# calculate mean deaths when including thoroscopy
mean_deaths_with_thoro = np.mean(list_deaths_with_thoroscopy)
# Calculate the percentage change in deaths due to spinal cord surgery
average_death_change_spine = np.round(
    ((mean_deaths_with_spine - mean_deaths_no_surg) / mean_deaths_no_surg) * 100, 2)
# Calculate the percentage change in deaths due to thoroscopy
average_death_change_thoro = np.round(
    ((mean_deaths_with_thoro - mean_deaths_no_surg) / mean_deaths_no_surg) * 100, 2)
# plot the mean deaths in the simulation without surgery vs with surgery, stating percentage change in the title
plt.bar(np.arange(2), [mean_deaths_no_surg, mean_deaths_with_surg], color=['lightsteelblue', 'lightsalmon'])
plt.xticks(np.arange(2), ['Average deaths '
                          '\n'
                          'due to RTI'
                          '\n'
                          'without additional '
                          '\n'
                          'available surgeries',
                          'Average deaths '
                          '\n'
                          'due to RTI'
                          '\n'
                          'with additional '
                          '\n'
                          'available surgeries'])
plt.ylabel('Average deaths')
plt.title(f"Average deaths in simulations with and without additional available surgeries"
          f"\n"
          f"including spinal cord surgery and thoroscopy changed average deaths by "
          f"{average_death_change_both}%"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/IncreasingSurgicalCapacity/compare_mean_total_deaths_with_without_extra_surg.png',
            bbox_inches='tight')
plt.clf()
# plot the comparison for the mean number of deaths in the normal model run compared to model runs with
# spinal cord surgery and thoroscopy
plt.bar(np.arange(3), [mean_deaths_no_surg, mean_deaths_with_spine, mean_deaths_with_thoro],
        color=['lightsteelblue', 'lightsalmon', 'lemonchiffon'])
plt.xticks(np.arange(3), ['Average deaths '
                          '\n'
                          'due to RTI'
                          '\n'
                          'without additional '
                          '\n'
                          'available surgeries',
                          'Average deaths '
                          '\n'
                          'due to RTI'
                          '\n'
                          'with spinal cord surgery',
                          'Average deaths '
                          '\n'
                          'due to RTI'
                          '\n'
                          'with thoroscopy'
                          ])
plt.ylabel('Average deaths')
plt.title(f"Average deaths in simulations with and without additional available surgeries."
          f"\n"
          f"Including spinal cord surgery changed average deaths by "
          f"{average_death_change_spine}%."
          f"\n"
          f"Including thoroscopy changed average deaths by "
          f"{average_death_change_thoro}%."
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/IncreasingSurgicalCapacity/compare_mean_total_deaths_with_extra_surg_split.png',
            bbox_inches='tight')
plt.clf()
