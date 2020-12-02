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
import pandas as pd
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
pop_size = 25000
nsim = 5
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
per_estimate_inc_of_RTI = []
per_estimate_inc_of_RTI_death = []
for incidence_estimate in estimates.values():
    average_inc_of_RTI_in_from_this_estimate = []
    average_inc_of_RTI_death_from_this_estimate = []
    for i in range(0, nsim):
        sim = Simulation(start_date=start_date)
        # We register all modules in a single call to the register method, calling once with multiple
        # objects. This is preferred to registering each module in multiple calls because we will be
        # able to handle dependencies if modules are registered together
        sim.register(
            demography.Demography(resourcefilepath=resourcefilepath),
            enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
            healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=service_availability),
            symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
            healthburden.HealthBurden(resourcefilepath=resourcefilepath),
            rti.RTI(resourcefilepath=resourcefilepath)
        )
        logfile = sim.configure_logging(filename="LogFile")
        # create and run the simulation
        sim.make_initial_population(n=pop_size)
        params = sim.modules['RTI'].parameters
        params['allowed_interventions'] = []
        orig_inc = params['base_rate_injrti']
        # Reduce the incidence of road traffic injuries
        params['base_rate_injrti'] = orig_incidence * incidence_estimate
        # reduce the proportion of prehospital mortality
        sim.simulate(end_date=end_date)
        log_df = parse_log_file(logfile)
        # get the incidence of RTI per 100,000 from the simulation
        summary_1m = log_df['tlo.methods.rti']['summary_1m']
        incidence_in_sim = summary_1m['incidence of rti per 100,000']
        incidence_of_death_in_sim = summary_1m['incidence of rti death per 100,000']
        average_inc_of_RTI_in_from_this_estimate.append(np.mean(incidence_in_sim))
        average_inc_of_RTI_death_from_this_estimate.append(np.mean(incidence_of_death_in_sim))
    per_estimate_inc_of_RTI.append(average_inc_of_RTI_in_from_this_estimate)
    per_estimate_inc_of_RTI_death.append(average_inc_of_RTI_death_from_this_estimate)
# Get the average estimate of number of people involved in road traffic injuries from each simulation
average_inc_per_estimate = []
sd_inc_per_estimate = []
for estimated_inc in per_estimate_inc_of_RTI:
    average_inc_per_estimate.append(np.mean(estimated_inc))
    sd_inc_per_estimate.append(np.std(estimated_inc))
average_inc_death_per_estimate = []
sd_inc_death_per_estimate = []
for estimated_inc_death in per_estimate_inc_of_RTI_death:
    average_inc_death_per_estimate.append(np.mean(estimated_inc_death))
    sd_inc_death_per_estimate.append(np.std(estimated_inc_death))

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
