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
# What I am doing here is modelling the what would happen if we included a number of different intervention strategies.
# I include two here, the first is the enforcements of speed limits, which should result in a 6% reduction in incidence
# and a 15% reduction in mortality according to https://www.bmj.com/content/bmj/344/bmj.e612.full.pdf
# The second is the enforcement of blood alcohol content laws, which should result in a 15% reduction in incidence and
# a 25% reduction in mortality, according to the above source.
# In both these scenarios, there is an asymetric reduction in incidence and mortality, I make the assumption that
# further to the deaths resulting from a percent reduction in incidence, the remaining deaths will be reduced from pre-
# hospital mortality.

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
output_for_different_incidence = dict()
service_availability = ["*"]
# Store the incidence, deaths and DALYs in each simulation
incidence_average = []
list_deaths_average = []
list_tot_dalys_average = []
scenarios = {'None': [1, 1],
             'Speed limit \n enforcement': [(1 - 0.06), (1 - 0.204)],
             'Drink-driving \n law enforcement': [(1 - 0.15), (1 - 0.169)],
             }
# Get the parameters
params = pd.read_excel(Path(resourcefilepath) / 'ResourceFile_RTI.xlsx', sheet_name='parameter_values')
# Get the base rate of road traffic injuries incidence
orig_incidence = float(params.loc[params.parameter_name == 'base_rate_injrti', 'value'].values)
# Get the pre-hospital mortality rate
orig_imm_death = float(params.loc[params.parameter_name == 'imm_death_proportion_rti', 'value'].values)
for i in range(0, nsim):
    incidence = []
    list_deaths = []
    list_tot_dalys = []
    incidences = []
    for scenario_reduction in scenarios.values():
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
        params['base_rate_injrti'] = orig_incidence * scenario_reduction[0]
        orig_imm_death = params['imm_death_proportion_rti']
        # reduce the proportion of prehospital mortality
        params['imm_death_proportion_rti'] = orig_imm_death * scenario_reduction[1]
        incidences.append(params['base_rate_injrti'])
        sim.simulate(end_date=end_date)
        log_df = parse_log_file(logfile)
        # get the incidence of RTI per 100,000 from the simulation
        summary_1m = log_df['tlo.methods.rti']['summary_1m']
        incidence_in_sim = summary_1m['incidence of rti per 100,000']
        mean_incidence = np.mean(summary_1m['incidence of rti per 100,000'])
        incidence.append(mean_incidence)
        # get the number of road traffic injury related deaths from the sim
        deaths_without_med = log_df['tlo.methods.demography']['death']
        tot_death_without_med = len(deaths_without_med.loc[(deaths_without_med['cause'] != 'Other')])
        list_deaths.append(tot_death_without_med)
        # get the dalys produced from the sim
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
    # store the per simulation incidences, deaths and dalys
    incidence_average.append(incidence)
    list_deaths_average.append(list_deaths)
    list_tot_dalys_average.append(list_tot_dalys)
# Get average incidence, death and daly in each intervention scenario
average_incidence = [float(sum(col)) / len(col) for col in zip(*incidence_average)]
average_deaths = [float(sum(col)) / len(col) for col in zip(*list_deaths_average)]
average_tot_dalys = [float(sum(col)) / len(col) for col in zip(*list_tot_dalys_average)]
results_df = pd.DataFrame({
    'incidence per 100,000': average_incidence,
    'average deaths': average_deaths,
    'average total DALYs': average_tot_dalys,
})
# get the percent reduction in incidence, deaths and dalys for each scenario
percent_inc_reduction = [((average_incidence[0] - inc) / average_incidence[0]) * 100 for inc in average_incidence]
percent_deaths_reduction = [((average_deaths[0] - deaths) / average_deaths[0]) * 100 for deaths in average_deaths
                            if average_deaths[0] != 0]
percent_dalys_reduction = [((average_tot_dalys[0] - daly) / average_tot_dalys[0]) * 100 for daly in average_tot_dalys]
# plot the average deaths and dalys in the simulation on the left hand x axis and the average incidence on the right
# hand axis
ax = results_df[['average deaths', 'average total DALYs']].plot(kind='bar', width=.35)
ax2 = results_df['incidence per 100,000'].plot(secondary_y=True, ax=ax)
ax.set_ylabel('Deaths/DALYs')
ax2.set_ylabel('Incidence per 100,000')
ax.set_xticklabels(labels=scenarios.keys())
plt.title(f"The effect of reducing incidence on Average Deaths/DALYS"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/ReducingIncidence/incidence_vs_deaths_DALYS.png', bbox_inches='tight')
plt.clf()
# Plot the percent reduction of deaths and DALYs in each simulation
plt.bar(np.arange(3), percent_inc_reduction, color='lightsteelblue', width=0.25, label='Reduction in incidence')
plt.bar(np.arange(3) + 0.25, percent_deaths_reduction, color='lightsalmon', width=0.25, label='Reduction in deaths')
plt.bar(np.arange(3) + 0.5, percent_dalys_reduction, color='wheat', width=0.25, label='Reduction in DALYs')
plt.xticks(np.arange(3), scenarios.keys())
plt.legend()
plt.xlabel('Percent')
plt.title(f"The percent reduction of incidence, average deaths and DALYS under different scenarios"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/ReducingIncidence/incidence_vs_deaths_DALYS_resulting_reduction.png', bbox_inches='tight')
