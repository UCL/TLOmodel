from pathlib import Path

import numpy as np
import pandas as pd
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
    simplified_births,
    symptommanager,
)

# =============================== Analysis description ========================================================
# This analysis file will eventually become what I use to produce the introduction to RTI paper. Here I run the model
# initally only allowing one injury per person, capturing the incidence of RTI and incidence of RTI death, calibrating
# this to the GBD estimates. I then run the model with multiple injuries and compare the outputs, the question being
# asked here is what happens to road traffic injury deaths if we allow multiple injuries to occur

# ============================================== Model run ============================================================
log_config = {
    "filename": "rti_health_system_comparison",  # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.rti": logging.INFO,
        "tlo.methods.healthsystem": logging.DEBUG,
        "tlo.methods.labour": logging.disable(logging.DEBUG)
    }
}
# The Resource files [NB. Working directory must be set to the root of TLO: TLOmodel]
resourcefilepath = Path('./resources')
save_file_path = "C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/"
# Establish the simulation object
yearsrun = 10
start_date = Date(year=2010, month=1, day=1)
end_date = Date(year=(2010 + yearsrun), month=1, day=1)
service_availability = ['*']
pop_size = 30000
nsim = 3
# Create a variable whether to save figures or not (used in debugging)
save_figures = True
imm_death = 0.01
# Iterate over the number of simulations nsim
log_file_location = './outputs/single_injury_model_vs_multiple_injury'
# store relevent model outputs
sing_inj_incidences_of_rti = []
sing_inj_incidences_of_death = []
sing_inj_incidences_of_injuries = []
sing_inj_cause_of_death_in_sim = []
for i in range(0, nsim):
    # Create the simulation object
    sim = Simulation(start_date=start_date)
    # Register the modules
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath),
        dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        rti.RTI(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
    )
    # Get the log file
    logfile = sim.configure_logging(filename="LogFile_single_injury",
                                    directory="./outputs/single_injury_model_vs_multiple_injury/single_injury")
    # create and run the simulation
    sim.make_initial_population(n=pop_size)
    # alter the number of injuries given out
    # Injury vibes number of GBD injury category distribution:
    number_inj_data = [1, 0, 0, 0, 0, 0, 0, 0]

    sim.modules['RTI'].parameters['number_of_injured_body_regions_distribution'] = [
        [1, 2, 3, 4, 5, 6, 7, 8], number_inj_data
    ]
    sim.modules['RTI'].parameters['imm_death_proportion_rti'] = imm_death
    # Run the simulation
    sim.simulate(end_date=end_date)
    # Parse the logfile of this simulation
    log_df = parse_log_file(logfile)
    # Store the incidence of RTI per 100,000 person years in this sim
    sing_inj_incidences_of_rti.append(log_df['tlo.methods.rti']['summary_1m']['incidence of rti per 100,000'].tolist())
    # Store the incidence of death due to RTI per 100,000 person years and the sub categories in this sim
    sing_inj_incidences_of_death.append(
        log_df['tlo.methods.rti']['summary_1m']['incidence of rti death per 100,000'].tolist())
    sing_inj_incidences_of_injuries.append(
        log_df['tlo.methods.rti']['Inj_category_incidence']['tot_inc_injuries'].tolist())
    deaths_in_sim = log_df['tlo.methods.demography']['death']
    rti_deaths = deaths_in_sim.loc[deaths_in_sim['cause'] != 'Other']
    sing_inj_cause_of_death_in_sim.append(rti_deaths['cause'].to_list())

mult_inj_incidences_of_rti = []
mult_inj_incidences_of_death = []
mult_inj_incidences_of_injuries = []
mult_inj_cause_of_death_in_sim = []
# Run model using the Injury Vibes distribution of number of injuries

# Run model using the Injury Vibes distribution of number of injuries
for i in range(0, nsim):
    # Create the simulation object
    sim = Simulation(start_date=start_date)
    # Register the modules
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath),
        dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        rti.RTI(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
    )
    # Get the log file
    logfile = sim.configure_logging(filename="LogFile_multiple_injury",
                                    directory="./outputs/single_injury_model_vs_multiple_injury/multiple_injury")
    # create and run the simulation
    sim.make_initial_population(n=pop_size)
    # alter the number of injuries given out
    # Injury vibes number of GBD injury category distribution:
    number_inj_data = [0.38, 0.25, 0.153, 0.094, 0.055, 0.031, 0.018, 0.019]

    sim.modules['RTI'].parameters['number_of_injured_body_regions_distribution'] = [
        [1, 2, 3, 4, 5, 6, 7, 8], number_inj_data
    ]
    sim.modules['RTI'].parameters['imm_death_proportion_rti'] = imm_death
    # Run the simulation
    sim.simulate(end_date=end_date)
    # Store the incidence of RTI per 100,000 person years in this sim
    mult_inj_incidences_of_rti.append(log_df['tlo.methods.rti']['summary_1m']['incidence of rti per 100,000'].tolist())
    # Store the incidence of death due to RTI per 100,000 person years and the sub categories in this sim
    mult_inj_incidences_of_death.append(
        log_df['tlo.methods.rti']['summary_1m']['incidence of rti death per 100,000'].tolist())
    mult_inj_incidences_of_injuries.append(
        log_df['tlo.methods.rti']['Inj_category_incidence']['tot_inc_injuries'].tolist())
    deaths_in_sim = log_df['tlo.methods.demography']['death']
    rti_deaths = deaths_in_sim.loc[deaths_in_sim['cause'] != 'Other']
    mult_inj_cause_of_death_in_sim.append(rti_deaths['cause'].to_list())

# flatten cause of death lists
sing_data = []
flattened_cause_of_death_sing = [cause for deaths in sing_inj_cause_of_death_in_sim for cause in deaths]
causes_of_death_in_sing = list(set(flattened_cause_of_death_sing))
for cause in causes_of_death_in_sing:
    sing_data.append(flattened_cause_of_death_sing.count(cause))
mult_data = []
flattened_cause_of_death_mult = [cause for deaths in mult_inj_cause_of_death_in_sim for cause in deaths]
causes_of_death_in_mult = list(set(flattened_cause_of_death_mult))
for cause in causes_of_death_in_mult:
    mult_data.append(flattened_cause_of_death_mult.count(cause))

cause_of_death_dict = {
    'Single injury model': [sing_data, causes_of_death_in_sing],
    'Multiple injury model': [mult_data, causes_of_death_in_mult],
}
for result in cause_of_death_dict.keys():
    cause_of_death_as_percent = [i / sum(cause_of_death_dict[result][0]) for i in cause_of_death_dict[result][0]]
    plt.bar(np.arange(len(cause_of_death_dict[result][0])), cause_of_death_as_percent, color='lightsteelblue')
    plt.xticks(np.arange(len(cause_of_death_dict[result][0])), cause_of_death_dict[result][1])
    plt.ylabel('Percent')
    plt.title("The percentage cause of death from\n road traffic injuries in the "
              + result + f" run.\n Number of simulations: {nsim}, population size: {pop_size}, years ran: {yearsrun}")
    plt.savefig(save_file_path + result + f" cause of death by percentage, imm death {imm_death}.png",
                bbox_inches='tight')
    plt.clf()
# Get GBD data to compare model to
data = pd.read_csv('resources/ResourceFile_RTI_GBD_Number_And_Incidence_Data.csv')
data = data.loc[data['metric'] == 'Rate']
data = data.loc[data['year'] > 2009]
death_data = data.loc[data['measure'] == 'Deaths']
in_rti_data = data.loc[data['measure'] == 'Incidence']
gbd_ten_year_average_inc = in_rti_data['val'].mean()
gbd_ten_year_average_inc_upper = in_rti_data['upper'].mean()
gbd_ten_year_average_inc_lower = in_rti_data['lower'].mean()

gbd_ten_year_average_inc_death = death_data['val'].mean()
gbd_ten_year_average_inc_death_upper = death_data['upper'].mean()
gbd_ten_year_average_inc_death_lower = death_data['lower'].mean()

# Single injury run summary stats
single_injury_mean_incidence_rti = np.mean(sing_inj_incidences_of_rti)
single_injury_mean_incidence_rti_death = np.mean(sing_inj_incidences_of_death)
single_injury_mean_incidence_injuries = np.mean(sing_inj_incidences_of_injuries)
# model run with vibes number of injury distribution
multiple_injury_mean_incidence_rti = np.mean(mult_inj_incidences_of_rti)
multiple_injury_mean_incidence_rti_death = np.mean(mult_inj_incidences_of_death)
multiple_injury_mean_incidence_injuries = np.mean(mult_inj_incidences_of_injuries)
#

results_single = [single_injury_mean_incidence_rti, single_injury_mean_incidence_rti_death,
                  single_injury_mean_incidence_injuries]
results_mult = [multiple_injury_mean_incidence_rti, multiple_injury_mean_incidence_rti_death,
                multiple_injury_mean_incidence_injuries]

results_gbd = [gbd_ten_year_average_inc, gbd_ten_year_average_inc_death, gbd_ten_year_average_inc]
n = np.arange(3)
plt.bar(n, results_gbd, width=0.3, color='lightsalmon', label='GBD estimates')
plt.bar(n + 0.3, results_single, width=0.3, color='lightsteelblue', label='Single injury model run')
plt.bar(n + 0.6, results_mult, width=0.3, color='burlywood', label='multiple injury model run')
plt.xticks(n + 0.3, ['Incidence of \nRTI', 'Incidence of \nRTI death', 'Incidence of \ninjuries'])
for i in range(len(results_gbd)):
    plt.annotate(str(np.round(results_gbd[i], 1)), xy=(n[i], results_gbd[i]), ha='center', va='bottom', rotation=60)
for i in range(len(results_single)):
    plt.annotate(str(np.round(results_single[i], 1)), xy=(n[i] + 0.3, results_single[i]), ha='center', va='bottom',
                 rotation=60)
for i in range(len(results_mult)):
    plt.annotate(str(np.round(results_mult[i], 1)), xy=(n[i] + 0.6, results_mult[i]), ha='center', va='bottom',
                 rotation=60)

plt.ylabel('Incidence per 100,000')
plt.legend()
plt.title(f"The effect of allowing multiple injuries\n in the model on population health outcomes.\n"
          f"Number of simulations: {nsim}, population size: {pop_size}, years run: {yearsrun}")
plt.savefig(save_file_path + f"Single_vs_multiple_injuries_full_comp_imm_death_{imm_death}.png",
            bbox_inches='tight')
plt.ylim([0, 1200])
plt.clf()
plt.bar(np.arange(2), results_single, width=0.3, color='lightsalmon', label='GBD estimates')
plt.bar(np.arange(2) + 0.3, results_mult, width=0.3, color='burlywood', label='multiple injury model run')
plt.xticks(n + 0.3, ['Incidence of \nRTI', 'Incidence of \nRTI death', 'Incidence of \ninjuries'])
for i in range(len(results_single)):
    plt.annotate(str(np.round(results_single[i], 1)), xy=(n[i], results_single[i]), ha='center', va='bottom',
                 rotation=60)
for i in range(len(results_mult)):
    plt.annotate(str(np.round(results_mult[i], 1)), xy=(n[i] + 0.3, results_mult[i]), ha='center', va='bottom',
                 rotation=60)
plt.ylabel('Incidence per 100,000')
plt.legend()
plt.title(f"The effect of allowing multiple injuries\n in the model on population health outcomes.\n"
          f"Number of simulations: {nsim}, population size: {pop_size}, years run: {yearsrun}")
plt.savefig(save_file_path + f"Single_vs_multiple_injurie_model_comp_imm_death_{imm_death}.png", bbox_inches='tight')
