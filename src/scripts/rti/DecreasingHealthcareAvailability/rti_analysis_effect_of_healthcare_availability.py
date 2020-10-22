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
output_for_different_incidence = dict()
service_availability = ["*"]
capabilities_reduction = np.linspace(1, 0, 5)
all_sim_deaths = []
all_sim_dalys = []
for i in range(0, nsim):
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
        deaths_without_med = log_df['tlo.methods.demography']['death']
        tot_death_without_med = len(deaths_without_med.loc[(deaths_without_med['cause'] != 'Other')])
        list_deaths.append(tot_death_without_med)
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
    all_sim_deaths.append(list_deaths)
    all_sim_dalys.append(list_tot_dalys)

all_sim_deaths
avg_tot_deaths = [float(sum(col))/len(col) for col in zip(*all_sim_deaths)]
avg_tot_dalys = [float(sum(col))/len(col) for col in zip(*all_sim_dalys)]
labels = []
for capability in capabilities_reduction:
    labels.append(str(capability))
width = 0.3
plt.bar(np.arange(len(avg_tot_deaths)), avg_tot_deaths, width=width)
plt.bar(np.arange(len(avg_tot_dalys)) + width, avg_tot_dalys, width=width)
plt.xticks(np.arange(len(avg_tot_deaths)), labels, rotation=45)
plt.xlabel('Capability coefficients')
plt.ylabel('Deaths/DALYs')
plt.title(f"Average deaths and DALYs in simulations for different capability coefficients"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/CapabilityAnalysis/compare_mean_total_deaths_and_dalys_per_capability_coefficient.png',
            bbox_inches='tight')
