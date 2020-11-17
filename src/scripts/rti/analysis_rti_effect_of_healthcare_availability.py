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

# =============================== Analysis description ========================================================
# What I am doing here is to see what happens when we run the model with different capability coefficients, i.e
# what happens when we reduce the capabilities of the health system, artificially clogging it up by reducing the
# parameter:
# capabilities['Total_Minutes_Per_Day'] =  capabilities['Total_Minutes_Per_Day'] * self.capabilities_coefficient
import pandas as pd
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
yearsrun = 5
start_date = Date(year=2010, month=1, day=1)
end_date = Date(year=(2010 + yearsrun), month=1, day=1)
pop_size = 5000
nsim = 2
service_availability = ["*"]
# Create a range of capability coefficients
capabilities_reduction = np.linspace(1, 0, 3)
# Create lists to store the simulation outputs in
all_sim_deaths = []
all_sim_dalys = []
for i in range(0, nsim):
    # create lists to store the deaths and dalys in this set of simulations
    list_deaths = []
    list_tot_dalys = []
    for capability in capabilities_reduction:
        sim = Simulation(start_date=start_date)
        # We register all modules in a single call to the register method, calling once with multiple
        # objects. This is preferred to registering each module in multiple calls because we will be
        # able to handle dependencies if modules are registered together
        sim.register(
            demography.Demography(resourcefilepath=resourcefilepath),
            enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
            healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=service_availability,
                                      capabilities_coefficient=float(capability)),
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
        sim.simulate(end_date=end_date)
        log_df = parse_log_file(logfile)
        # get the death data
        deaths = log_df['tlo.methods.demography']['death']
        tot_death = len(deaths.loc[(deaths['cause'] != 'Other')])
        list_deaths.append(tot_death)
        # get the daly data
        dalys_df = log_df['tlo.methods.healthburden']['DALYS']
        males_data = dalys_df.loc[dalys_df['sex'] == 'M']
        YLL_males_data = males_data.filter(like='YLL_RTI').columns
        males_dalys = males_data[YLL_males_data].sum(axis=1) + \
                      males_data['YLD_RTI_rt_disability']
        females_data = dalys_df.loc[dalys_df['sex'] == 'F']
        YLL_females_data = females_data.filter(like='YLL_RTI').columns
        females_dalys = females_data[YLL_females_data].sum(axis=1) + \
                        females_data['YLD_RTI_rt_disability']

        tot_dalys = males_dalys.tolist() + females_dalys.tolist()
        list_tot_dalys.append(sum(tot_dalys))
    # Append the resulting deaths and dalys from this simulation to the results lists
    all_sim_deaths.append(list_deaths)
    all_sim_dalys.append(list_tot_dalys)

# Average out the deaths and dalys in all the simulations
avg_tot_deaths = [float(sum(col))/len(col) for col in zip(*all_sim_deaths)]
std_tot_deaths = [np.std(col) for col in zip(*all_sim_deaths)]
avg_tot_dalys = [float(sum(col))/len(col) for col in zip(*all_sim_dalys)]
std_tot_dalys = [np.std(col) for col in zip(*all_sim_dalys)]
# Create labels for the x axis
labels = []
for capability in capabilities_reduction:
    labels.append(str(capability))
width = 0.3
# create the bar plots
plt.bar(np.arange(len(avg_tot_deaths)), avg_tot_deaths,
        width=width, color='lightsalmon', label='deaths', yerr=std_tot_deaths)
plt.bar(np.arange(len(avg_tot_dalys)) + width, avg_tot_dalys,
        width=width, color='lightsteelblue', label='DALYs', yerr=std_tot_dalys)
plt.xticks(np.arange(len(avg_tot_deaths)), labels, rotation=45)
plt.xlabel('Capability coefficients')
plt.ylabel('Deaths/DALYs')
plt.legend()
plt.title(f"Average deaths and DALYs in simulations for different capability coefficients"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/CapabilityAnalysis/compare_mean_total_deaths_and_dalys_per_capability_coefficient.png',
            bbox_inches='tight')
