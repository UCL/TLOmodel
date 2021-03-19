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
# I include three here, the first is the enforcements of speed limits, which should result in a 6% reduction in
# incidence and a 15% reduction in mortality according to https://www.bmj.com/content/bmj/344/bmj.e612.full.pdf
# The second is the enforcement of blood alcohol content laws, which should result in a 15% reduction in incidence and
# a 25% reduction in mortality, according to the above source.
# The third scenario considered is the use of rumble strips. In Ghana, they reduced chrashes on the highway by 35% and
# fatalities by 55%: https://doi.org/10.1076/icsp.10.1.77.14113. Currently I am just extrapolating these values to the
# whole country...
# TODO: Redo the analysis with a more sophisticated approach
# In all these scenarios, there is an asymetric reduction in incidence and mortality, I make the assumption that
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
pop_size = 50000
nsim = 2
output_for_different_incidence = dict()
# Store the incidence, deaths and DALYs in each simulation
incidence_average = []
list_deaths_average = []
list_tot_dalys_average = []
scenarios = {'None': [1, 1],
             'Speed limit \n enforcement': [(1 - 0.06), (1 - 0.204)],
             'Drink-driving \n law enforcement': [(1 - 0.15), (1 - 0.169)],
             'Rumble strips': [(1 - 0.35), (1 - 0.204)]
             }
# Get the parameters
params = pd.read_excel(Path(resourcefilepath) / 'ResourceFile_RTI.xlsx', sheet_name='parameter_values')
# Get the base rate of road traffic injuries incidence
orig_incidence = float(params.loc[params.parameter_name == 'base_rate_injrti', 'value'].values)
# Get the pre-hospital mortality rate
orig_imm_death = float(params.loc[params.parameter_name == 'imm_death_proportion_rti', 'value'].values)
# Create empty lists to store per scenario outputs in
per_scenario_inc = []
per_scenario_death = []
per_scenario_dalys = []
per_scenario_inpatient_days = []
per_scenario_consumables = []
# iterate over the scenarios
for scenario_reduction in scenarios.values():
    # reduce the incidence in the simulation as per the scenario
    inc = orig_incidence * scenario_reduction[0]
    # reduce the proportions of crashes that result in prehospital mortality as per the scenario
    prop_imm_death = orig_imm_death * scenario_reduction[1]
    # create empty lists to store the within scenario outputs
    in_scenario_incidence = []
    in_scenario_deaths = []
    in_scenario_dalys = []
    average_inpatient_days_from_scenario = []
    average_consumables_from_scenario = []
    for i in range(0, nsim):
        # create a variable tracking the total inpatient days used in this sim
        total_inpatient_days_this_sim = 0
        sim = Simulation(start_date=start_date)
        # register modules
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
        # name logfile
        logfile = sim.configure_logging(filename="LogFile")
        # create and run the simulation
        sim.make_initial_population(n=pop_size)
        # set the incidence used in this simulation
        params = sim.modules['RTI'].parameters
        params['base_rate_injrti'] = inc
        # set the prehospital mortality percentage used in this sim
        params['imm_death_proportion_rti'] = prop_imm_death
        # run the simulation
        sim.simulate(end_date=end_date)
        # parse the logfile
        log_df = parse_log_file(logfile)
        # get the dataframe for the monthly summary of RTI info
        summary_1m = log_df['tlo.methods.rti']['summary_1m']
        # get the incidence of RTI per 100,000 from the simulation
        incidence_in_sim = summary_1m['incidence of rti per 100,000']
        # calculate the mean incidence in this sim
        mean_incidence = np.mean(summary_1m['incidence of rti per 100,000'])
        # store the mean incidence in this simulation
        in_scenario_incidence.append(mean_incidence)
        # get the deaths information
        rti_deaths = log_df['tlo.methods.demography']['death']
        rti_causes_of_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
        # calculate the total number of rti related deaths
        tot_rti_deaths = len(rti_deaths.loc[rti_deaths['cause'].isin(rti_causes_of_deaths)])
        # store the deaths due to RTI in this sim
        in_scenario_deaths.append(tot_rti_deaths)
        # get the dalys dataframe
        dalys_df = log_df['tlo.methods.healthburden']['dalys']
        # get the male data
        males_data = dalys_df.loc[dalys_df['sex'] == 'M']
        # get male yll
        YLL_males_data = males_data.filter(like='YLL_RTI').columns
        # calculate male dalys
        males_dalys = males_data[YLL_males_data].sum(axis=1) + males_data['YLD_RTI_rt_disability']
        # get the female data
        females_data = dalys_df.loc[dalys_df['sex'] == 'F']
        # get female yll
        YLL_females_data = females_data.filter(like='YLL_RTI').columns
        # calculate female dalys
        females_dalys = females_data[YLL_females_data].sum(axis=1) + females_data['YLD_RTI_rt_disability']
        # calculate total dalys
        tot_dalys = males_dalys.tolist() + females_dalys.tolist()
        # store total dalys in sim
        in_scenario_dalys.append(sum(tot_dalys))
        # get the health system usage dataframe, specifically for road traffic injuries
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
        # store the inpatient day data
        average_inpatient_days_from_scenario.append(total_inpatient_days_this_sim)
        # Get the consumable dataframe
        consumables_list = log_df['tlo.methods.healthsystem']['Consumables']['Item_Available'].tolist()
        # create an empty list to store names of consumables used in sim
        consumables_list_to_dict = []
        for string in consumables_list:
            # store name of consumables used in sim
            consumables_list_to_dict.append(ast.literal_eval(string))
        # create variable to store the number of consumables used in this sim
        number_of_consumables_in_sim = 0
        for dictionary in consumables_list_to_dict:
            # iterate over names of consumables used in sim, store the number of consumables used in sim
            number_of_consumables_in_sim += sum(dictionary.values())
        # store number of consumables used
        average_consumables_from_scenario.append(number_of_consumables_in_sim)
    # store the in scenario incidence, death, daly, inpatient day and consumable usage data
    per_scenario_inc.append(in_scenario_incidence)
    per_scenario_death.append(in_scenario_deaths)
    per_scenario_dalys.append(in_scenario_dalys)
    per_scenario_inpatient_days.append(np.mean(average_inpatient_days_from_scenario))
    per_scenario_consumables.append(np.mean(average_consumables_from_scenario))

# Calculate average incidence, death, dalys, inpatient days and consumables used in each scenario
average_incidence = [np.mean(inc_list) for inc_list in per_scenario_inc]
average_deaths = [np.mean(death_list) for death_list in per_scenario_death]
average_tot_dalys = [np.mean(daly_list) for daly_list in per_scenario_dalys]
average_tot_inpatient_days = [np.mean(day_list) for day_list in per_scenario_inpatient_days]
average_tot_consumables = [np.mean(consumables) for consumables in per_scenario_consumables]
# Store part of the data in a dataframe
results_df = pd.DataFrame({
    'incidence per 100,000': average_incidence,
    'average deaths': average_deaths,
    'average total DALYs': average_tot_dalys,
})
# get the percent reduction in incidence, deaths, dalys, inpatient day and consumable usage for each scenario
percent_inc_reduction = [((inc - average_incidence[0]) / average_incidence[0]) * 100 for inc in average_incidence]
percent_deaths_reduction = [((deaths - average_deaths[0]) / average_deaths[0]) * 100 for deaths in average_deaths
                              if average_deaths[0] != 0]
percent_dalys_reduction = [((daly - average_tot_dalys[0]) / average_tot_dalys[0]) * 100 for daly in average_tot_dalys]
percent_inpatient_day_reduction = \
    [((days - average_tot_inpatient_days[0]) / average_tot_inpatient_days[0]) * 100 for days in
     average_tot_inpatient_days]
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
w = 0.8 / len(scenarios)
plt.bar(np.arange(len(scenarios)), percent_inc_reduction, color='lightsteelblue', width=w,
        label='% change in incidence')
plt.bar(np.arange(len(scenarios)) + w, percent_deaths_reduction, color='lightsalmon', width=w,
        label='% change in deaths')
plt.bar(np.arange(len(scenarios)) + 2 * w, percent_dalys_reduction, color='wheat', width=w, label='% change in DALYs')
plt.bar(np.arange(len(scenarios)) + 3 * w, percent_inpatient_day_reduction, color='olive', width=w,
        label='% change in inpatient days')
plt.bar(np.arange(len(scenarios)) + 4 * w, percent_consumable_reduction, color='lemonchiffon', width=w,
        label='% change in consumables')
plt.xticks(np.arange(len(scenarios)) + 2 * w, scenarios.keys())
plt.legend()
plt.ylabel('Percent')
plt.title(f"The percent change of incidence, average deaths and DALYS under different scenarios"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/ReducingIncidence/incidence_vs_deaths_DALYS_resulting_reduction.png', bbox_inches='tight')
