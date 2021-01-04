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
import pandas as pd
import ast
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
per_scenario_inpatient_days = []
per_scenario_consumables = []
for scenario_reduction in scenarios.values():
    inc = orig_incidence * scenario_reduction[0]
    prop_imm_death = orig_imm_death * scenario_reduction[1]
    in_scenario_incidence = []
    in_scenario_deaths = []
    in_scenario_dalys = []
    average_inpatient_days_from_scenario = []
    average_consumables_from_scenario = []
    for i in range(0, nsim):
        total_inpatient_days_this_sim = 0
        sim = Simulation(start_date=start_date)
        # We register all modules in a single call to the register method, calling once with multiple
        # objects. This is preferred to registering each module in multiple calls because we will be
        # able to handle dependencies if modules are registered together
        sim.register(
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
        dalys_df = log_df['tlo.methods.healthburden']['dalys']
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
        inpatient_day_df = log_df['tlo.methods.healthsystem']['HSI_Event'].loc[
            log_df['tlo.methods.healthsystem']['HSI_Event']['TREATMENT_ID'] == 'RTI_MedicalIntervention']
        for person in inpatient_day_df.index:
            # Get the number of inpatient days per person, if there is a key error when trying to access inpatient days it
            # means that this patient didn't require any so append (0)
            try:
                total_inpatient_days_this_sim += \
                    inpatient_day_df.loc[person, 'Number_By_Appt_Type_Code']['InpatientDays']
            except KeyError:
                total_inpatient_days_this_sim += 0
        # Get the number of consumables used in this sim
        average_inpatient_days_from_scenario.append(total_inpatient_days_this_sim)
        consumables_list = log_df['tlo.methods.healthsystem']['Consumables']['Item_Available'].tolist()
        consumables_list_to_dict = []
        for string in consumables_list:
            consumables_list_to_dict.append(ast.literal_eval(string))
        number_of_consumables_in_sim = 0
        for dictionary in consumables_list_to_dict:
            number_of_consumables_in_sim += sum(dictionary.values())
        average_consumables_from_scenario.append(number_of_consumables_in_sim)

    per_scenario_inc.append(in_scenario_incidence)
    per_scenario_death.append(in_scenario_deaths)
    per_scenario_dalys.append(in_scenario_dalys)
    per_scenario_inpatient_days.append(np.mean(average_inpatient_days_from_scenario))
    per_scenario_consumables.append(np.mean(average_consumables_from_scenario))

average_incidence = [np.mean(inc_list) for inc_list in per_scenario_inc]
average_deaths = [np.mean(death_list) for death_list in per_scenario_death]
average_tot_dalys = [np.mean(daly_list) for daly_list in per_scenario_dalys]
average_tot_inpatient_days = [np.mean(day_list) for day_list in per_scenario_inpatient_days]
average_tot_consumables = [np.mean(consumables) for consumables in per_scenario_consumables]
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
percent_inpatient_day_reduction = [((days - average_tot_inpatient_days[0]) /
                                    average_tot_inpatient_days[0]) * 100 for days in average_tot_inpatient_days]
percent_consumable_reduction = [((consumables - average_tot_consumables[0]) / average_tot_consumables[0]) * 100 for
                                consumables in average_tot_consumables]
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
w = 0.15
plt.bar(np.arange(3), percent_inc_reduction, color='lightsteelblue', width=w, label='% change in incidence')
plt.bar(np.arange(3) + 0.15, percent_deaths_reduction, color='lightsalmon', width=w, label='% change in deaths')
plt.bar(np.arange(3) + 0.3, percent_dalys_reduction, color='wheat', width=w, label='% change in DALYs')
plt.bar(np.arange(3) + 0.45, percent_inpatient_day_reduction, color='olive', width=w,
        label='% change in inpatient days')
plt.bar(np.arange(3) + 0.6, percent_consumable_reduction, color='lemonchiffon', width=w, label='% change in '
                                                                                                 'consumables')
plt.xticks(np.arange(3) + 0.3, scenarios.keys())
plt.legend()
plt.ylabel('Percent')
plt.title(f"The percent change of incidence, average deaths and DALYS under different scenarios"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/ReducingIncidence/incidence_vs_deaths_DALYS_resulting_reduction.png', bbox_inches='tight')
# plt.clf()
# plt.bar(np.arange(3), percent_inpatient_day_reduction, color='wheat', width=0.25, label='% change in inpatient days')
# plt.xticks(np.arange(3), scenarios.keys())
# plt.legend()
# plt.ylabel('Percent')
# plt.title(f"The percent change in inpatient days in each scenario"
#           f"\n"
#           f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
# plt.savefig('outputs/ReducingIncidence/incidence_vs_inpatient_days_resulting_reduction.png', bbox_inches='tight')
