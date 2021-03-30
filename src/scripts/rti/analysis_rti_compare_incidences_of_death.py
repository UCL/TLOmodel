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
    symptommanager,
)

# =============================== Analysis description ========================================================
# What I am doing here is fiting the model to the various estimates of death per 100,000 person years available for
# Malawi, starting from the estimate from hospital registry data of 5.1 per 100,000 person years to the highest
# estimate from the WHO of 35 per 100,000 person years, creating a best and worst case scenario of the health burden
# of road traffic injuries in Malawi
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
output_for_different_incidence = dict()
service_availability = ["*"]
# Store the incidence, deaths and DALYs in each simulation
incidence_average = []
list_deaths_average = []
list_tot_dalys_average = []
# Create a dictionary to store the parameters which we will multiply the base rate of injury by to fit to the different
# estimates of death per 100,000

estimates = {'Hospital registry \ndata': 0.31,
             'Samuel et al. \n 2012 estimate': 1.19,
             'WHO \n estimate': 2.12,
             }
# Get the parameters
params = pd.read_excel(Path(resourcefilepath) / 'ResourceFile_RTI.xlsx', sheet_name='parameter_values')
# Get the base rate of road traffic injuries incidence
orig_incidence = float(params.loc[params.parameter_name == 'base_rate_injrti', 'value'].values)
# create empty lists to store the incidence of RTI, RTI death and inpatient days in each sim, corresponding to
# the level of RTI deaths
per_estimate_inc_of_RTI = []
per_estimate_inc_of_RTI_death = []
per_estimate_inpatient_days = []
# Iterate over the estimates of RTI death
for incidence_estimate in estimates.values():
    # create empty lists to store the average incidences of RTI, RTI death and inpatient day usage for this
    # estimate
    average_inc_of_RTI_in_from_this_estimate = []
    average_inc_of_RTI_death_from_this_estimate = []
    average_inpatient_days_from_estimate = []
    for i in range(0, nsim):
        # create variable to track inpatient day usage
        total_inpatient_days_this_sim = 0
        # create simulation object
        sim = Simulation(start_date=start_date)
        # We register all modules in a single call to the register method, calling once with multiple
        # objects. This is preferred to registering each module in multiple calls because we will be
        # able to handle dependencies if modules are registered together
        sim.register(
            demography.Demography(resourcefilepath=resourcefilepath),
            enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
            healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=service_availability),
            symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
            dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath),
            dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
            healthburden.HealthBurden(resourcefilepath=resourcefilepath),
            rti.RTI(resourcefilepath=resourcefilepath)
        )
        # name logfile
        logfile = sim.configure_logging(filename="LogFile")
        # create initial population
        sim.make_initial_population(n=pop_size)
        # Reduce the incidence of road traffic injuries
        params = sim.modules['RTI'].parameters
        params['base_rate_injrti'] = orig_incidence * incidence_estimate
        # Run simulation
        sim.simulate(end_date=end_date)
        # parse logfile
        log_df = parse_log_file(logfile)
        # get the rti summary df
        summary_1m = log_df['tlo.methods.rti']['summary_1m']
        # get the incidence of RTI per 100,000 from the simulation
        incidence_in_sim = summary_1m['incidence of rti per 100,000']
        # get the incidence of RTI death per 100,000 from the simulation
        incidence_of_death_in_sim = summary_1m['incidence of rti death per 100,000']
        # store the average incidence of RTI in this sim
        average_inc_of_RTI_in_from_this_estimate.append(np.mean(incidence_in_sim))
        # store the average incidence of RTI death in this sim
        average_inc_of_RTI_death_from_this_estimate.append(np.mean(incidence_of_death_in_sim))
        # get health system usage df
        inpatient_day_df = log_df['tlo.methods.healthsystem']['HSI_Event'].loc[
            log_df['tlo.methods.healthsystem']['HSI_Event']['TREATMENT_ID'] == 'RTI_MedicalIntervention']
        for person in inpatient_day_df.index:
            # Get the number of inpatient days per person, if there is a key error when trying to access inpatient days
            # it means that this patient didn't require any so append (0)
            try:
                total_inpatient_days_this_sim += \
                    inpatient_day_df.loc[person, 'Number_By_Appt_Type_Code']['InpatientDays']
            except KeyError:
                total_inpatient_days_this_sim += 0
        # store inpatient days used in this simulation
        average_inpatient_days_from_estimate.append(total_inpatient_days_this_sim)
    # store the estimated incidence of rti and rti deaths, and inpatient day usage in this scenario
    per_estimate_inc_of_RTI.append(average_inc_of_RTI_in_from_this_estimate)
    per_estimate_inc_of_RTI_death.append(average_inc_of_RTI_death_from_this_estimate)
    per_estimate_inpatient_days.append(np.mean(average_inpatient_days_from_estimate))

# Create the empty lists to store the average incidence and standard deviation in incidence per sim
average_inc_per_estimate = []
sd_inc_per_estimate = []
# Store the average estimate of number of people involved in road traffic injuries from each simulation and
# the standard deviation
for estimated_inc in per_estimate_inc_of_RTI:
    average_inc_per_estimate.append(np.mean(estimated_inc))
    sd_inc_per_estimate.append(np.std(estimated_inc))
# Create the empty lists to store the average incidence of death and standard deviation in incidence per sim
average_inc_death_per_estimate = []
sd_inc_death_per_estimate = []
# Store the average estimate of number of people who died due to road traffic injuries from each simulation and
# the standard deviation
for estimated_inc_death in per_estimate_inc_of_RTI_death:
    average_inc_death_per_estimate.append(np.mean(estimated_inc_death))
    sd_inc_death_per_estimate.append(np.std(estimated_inc_death))
# Create the empty lists to store the average and standard deviation in  inpatient day usage per sim
average_inpatient_days = []
sd_inpatient_day = []
# Store the average inpatient day usage due to road traffic injuries from each simulation and
# the standard deviation
for estimated_inpatient_day in per_estimate_inpatient_days:
    average_inpatient_days.append(np.mean(estimated_inpatient_day))
    sd_inpatient_day.append(np.std(estimated_inpatient_day))
# Plot the average incidence of RTI per person from each estimated incidence of death/total number of injuries
plt.bar(np.arange(len(estimates)), average_inc_per_estimate, color='lightsteelblue', yerr=sd_inc_per_estimate)
plt.xticks(np.arange(len(estimates)), estimates.keys())
plt.title(f"The estimated incidence of people with road traffic injuries"
          f"\n"
          f"based on different estimates of the incidence of road traffic injury deaths"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/NumberOfPeopleInRTI/Estimates_for_people_with_rti.png', bbox_inches='tight')
plt.clf()
# Plot the average incidence of RTI death to confirm the model fitting is doing what it should
plt.bar(np.arange(len(estimates)), average_inc_death_per_estimate, color='lightsalmon',
        yerr=sd_inc_death_per_estimate)
plt.xticks(np.arange(len(estimates)), estimates.keys())
plt.title(f"The incidence of death due to road traffic injuries for each estimate"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/NumberOfPeopleInRTI/Incidence_of_death_from_rti.png', bbox_inches='tight')
plt.clf()
# plot the average inpatient day usage for each estimate
plt.bar(np.arange(len(estimates)), average_inpatient_days, color='wheat', yerr=sd_inpatient_day)
plt.xticks(np.arange(len(estimates)), estimates.keys())
plt.title(f"The estimated incidence of people with road traffic injuries"
          f"\n"
          f"based on different estimates of the incidence of road traffic injury deaths"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/NumberOfPeopleInRTI/Estimates_inpatient_days_people_with_rti.png', bbox_inches='tight')
plt.clf()
