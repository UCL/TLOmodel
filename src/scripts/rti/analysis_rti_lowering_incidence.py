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
yearsrun = 10
start_date = Date(year=2010, month=1, day=1)
end_date = Date(year=(2010 + yearsrun), month=1, day=1)
pop_size = 100000
nsim = 5
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
per_scenario_inc = []
per_scenario_death = []
per_scenario_dalys = []
for scenario_reduction in scenarios.values():
    inc = orig_incidence * scenario_reduction[0]
    prop_imm_death = orig_imm_death * scenario_reduction[1]
    in_scenario_incidence = []
    in_scenario_deaths = []
    in_scenario_dalys = []
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
        params['base_rate_injrti'] = inc
        orig_imm_death = params['imm_death_proportion_rti']
        # reduce the proportion of prehospital mortality
        params['imm_death_proportion_rti'] = prop_imm_death
        sim.simulate(end_date=end_date)
        log_df = parse_log_file(logfile)
        # get the incidence of RTI per 100,000 from the simulation
        summary_1m = log_df['tlo.methods.rti']['summary_1m']
        incidence_in_sim = summary_1m['incidence of rti per 100,000']
        mean_incidence = np.mean(summary_1m['incidence of rti per 100,000'])
        in_scenario_incidence.append(mean_incidence)
        # get the number of road traffic injury related deaths from the sim
        rti_deaths = log_df['tlo.methods.demography']['death']
        tot_rti_deaths = len(rti_deaths.loc[(rti_deaths['cause'] != 'Other')])
        in_scenario_deaths.append(tot_rti_deaths)
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
        in_scenario_dalys.append(sum(tot_dalys))
    per_scenario_inc.append(in_scenario_incidence)
    per_scenario_death.append(in_scenario_deaths)
    per_scenario_dalys.append(in_scenario_dalys)

average_incidence = [np.mean(inc_list) for inc_list in per_scenario_inc]
average_deaths = [np.mean(death_list) for death_list in per_scenario_death]
average_tot_dalys = [np.mean(daly_list) for daly_list in per_scenario_dalys]

results_df = pd.DataFrame({
    'incidence per 100,000': average_incidence,
    'average deaths': average_deaths,
    'average total DALYs': average_tot_dalys,
})
# get the percent reduction in incidence, deaths and dalys for each scenario
percent_inc_reduction = [((inc - average_incidence[0]) / average_incidence[0]) * 100 for inc in average_incidence]
percent_deaths_reduction = [((deaths - average_deaths[0]) / average_deaths[0]) * 100 for deaths in average_deaths
                              if average_deaths[0] != 0]
percent_dalys_reduction = [((daly - average_tot_dalys[0]) / average_tot_dalys[0]) * 100 for daly in average_tot_dalys]
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
plt.bar(np.arange(3), percent_inc_reduction, color='lightsteelblue', width=0.25, label='% change in incidence')
plt.bar(np.arange(3) + 0.25, percent_deaths_reduction, color='lightsalmon', width=0.25, label='% change in deaths')
plt.bar(np.arange(3) + 0.5, percent_dalys_reduction, color='wheat', width=0.25, label='% change in DALYs')
plt.xticks(np.arange(3) + 0.25, scenarios.keys())
plt.legend()
plt.ylabel('Percent')
plt.title(f"The percent change of incidence, average deaths and DALYS under different scenarios"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/ReducingIncidence/incidence_vs_deaths_DALYS_resulting_reduction.png', bbox_inches='tight')
