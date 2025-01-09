from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    rti,
    simplified_births,
    symptommanager,
)

seed = 100

log_config_no_hs = {
    "filename": "rti_analysis_no_perfect_healthseeking",   # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.rti": logging.INFO,
    }
}
log_config_with_hs = {
    "filename": "rti_analysis_with_healthsystem",   # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.rti": logging.INFO,
    }
}
start_date = Date(2010, 1, 1)
end_date = Date(2030, 12, 31)
pop_size = 3000

# Creat simulations both with and without the health system
sim_no_health_system = Simulation(start_date=start_date, seed=seed, log_config=log_config_no_hs)

# Path to the resource files used by the disease and intervention methods
# resources = "./resources"
resourcefilepath = Path('./resources')

# Used to configure health system behaviour
service_availability_no_hs = []
# We register all modules in a single call to the register method, calling once with multiple
# objects. This is preferred to registering each module in multiple calls because we will be
# able to handle dependencies if modules are registered together
# Register modules used in each model run, specifying the availability of service from the hs
sim_no_health_system.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        rti.RTI(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath)
        )

# create and run each simulation
sim_no_health_system.make_initial_population(n=pop_size)
sim_no_health_system.simulate(end_date=end_date)

sim_with_health_system = Simulation(start_date=start_date, seed=seed, log_config=log_config_with_hs)

service_availability_with_hs = ['*']

sim_with_health_system.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        rti.RTI(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath)
        )

sim_with_health_system.make_initial_population(n=pop_size)
sim_with_health_system.simulate(end_date=end_date)
# parse the simulation logfile to get the output dataframes
log_df_no_hs = parse_log_file(sim_no_health_system.log_filepath)
log_df_with_hs = parse_log_file(sim_with_health_system.log_filepath)
#
# ###########
# folder = Path("/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/longterm_trends_all_diseases-2024-09-25T110820Z")
#
# pop_model = summarize(extract_results(folder,
#                                       module="tlo.methods.demography",
#                                       key="population",
#                                       column="total",
#                                       index="date",
#                                       do_scaling=True
#                                       ),
#                       collapse_columns=True
#                       )
#
#
# mean_li_urban_immediate_death = summarize(extract_results(
#     folder,
#     module="tlo.methods.demography.detail",
#     key="properties_of_deceased_persons",
#     custom_generate_series = (
#     lambda df: df.loc[(df['li_urban']) &
#                       (df['cause_of_death'].str.contains('RTI_imm_death'))].assign(
#         year=df['date'].dt.year).groupby(['year'])['year'].count()),
#     do_scaling=True))
#
# mean_li_urban_immediate_death = mean_li_urban_immediate_death.reset_index()
# mean_li_urban_immediate_death.columns = ['year', 'lower', 'mean', 'upper']
#
# mean_li_urban_non_immediate_death = summarize(extract_results(
#     folder,
#     module="tlo.methods.demography.detail",
#     key="properties_of_deceased_persons",
#     custom_generate_series = (
#     lambda df: df.loc[(df['li_urban']) &
#                       (df['cause_of_death'].str.contains('RTI_')) &
#                       (~df['cause_of_death'].str.contains('RTI_imm_death'))].assign(
#         year=df['date'].dt.year).groupby(['year'])['year'].count()),
#     do_scaling=True))
#
# mean_li_urban_non_immediate_death = mean_li_urban_non_immediate_death.reset_index()
# mean_li_urban_non_immediate_death.columns = ['year', 'lower', 'mean', 'upper']
#
#
# pop_model = pop_model.reset_index()
# pop_model.columns = ['year', 'lower', 'mean', 'upper']
#
# # plt.plot(range(len(mean_li_urban_non_immediate_death['year'])), mean_li_urban_non_immediate_death['mean']/pop_model['mean'] * 100000, color = 'blue')
# #
# # plt.ylabel("Deaths due to RTI per 100,000")
# # plt.xlabel("Year")
# # #plt.show()
# #
# # plt.plot(range(len(mean_li_urban_immediate_death['year'])), mean_li_urban_immediate_death['mean']/pop_model['mean'] * 100000, color = 'red')
# # plt.ylabel("Deaths due to RTI per 100,000")
# # plt.xlabel("Year")
# # #plt.show()
# #
#
#
# ###### entire population
#
# mean_li_all_immediate_death = summarize(extract_results(
#     folder,
#     module="tlo.methods.demography.detail",
#     key="properties_of_deceased_persons",
#     custom_generate_series = (
#     lambda df: df.loc[
#                       (df['cause_of_death'].str.contains('RTI_imm_death'))].assign(
#         year=df['date'].dt.year).groupby(['year'])['year'].count()),
#     do_scaling=True))
#
# mean_li_all_immediate_death = mean_li_all_immediate_death.reset_index()
# mean_li_all_immediate_death.columns = ['year', 'lower', 'mean', 'upper']
#
# mean_li_all_non_immediate_death = summarize(extract_results(
#     folder,
#     module="tlo.methods.demography.detail",
#     key="properties_of_deceased_persons",
#     custom_generate_series = (
#     lambda df: df.loc[
#                       (df['cause_of_death'].str.contains('RTI_')) &
#                       (~df['cause_of_death'].str.contains('RTI_imm_death'))].assign(
#         year=df['date'].dt.year).groupby(['year'])['year'].count()),
#     do_scaling=True))
# mean_li_all_non_immediate_death = mean_li_all_non_immediate_death.reset_index()
# mean_li_all_non_immediate_death.columns = ['year', 'lower', 'mean', 'upper']
# mean_li_all_non_immediate_death = mean_li_all_non_immediate_death.sort_values(by='year', ascending=True)
#
#
# mean_li_all_death_with_med = summarize(extract_results(
#     folder,
#     module="tlo.methods.demography.detail",
#     key="properties_of_deceased_persons",
#     custom_generate_series = (
#     lambda df: df.loc[(df['cause_of_death'].str.contains('RTI_death_with_med'))].assign(
#         year=df['date'].dt.year).groupby(['year'])['year'].count()),
#     do_scaling=True))
#
# mean_li_all_death_with_med = mean_li_all_death_with_med.reset_index()
# mean_li_all_death_with_med.columns = ['year', 'lower', 'mean', 'upper']
#
# mean_li_all_death_with_med = mean_li_all_death_with_med.sort_values(by='year', ascending=True)
#
# plt.plot(range(len(mean_li_all_non_immediate_death['year'])), mean_li_all_non_immediate_death['mean']/pop_model['mean'] * 100000, color = 'blue')
# plt.title("mean_li_all_immediate_death")
# plt.ylabel("Deaths due to RTI per 100,000")
# plt.xlabel("Year")
# plt.show()
#
# mean_li_all_immediate_death = mean_li_all_immediate_death.sort_values(by='year', ascending=True)
# plt.plot(range(len(mean_li_all_immediate_death['year'])), mean_li_all_immediate_death['mean']/pop_model['mean'] * 100000, color = 'red')
# plt.title("mean_li_all_immediate_death")
#
# plt.ylabel("Deaths due to RTI per 100,000")
# plt.xlabel("Year")
# plt.show()
#
# mean_li_all_death_with_med = mean_li_all_death_with_med.sort_values(by='year', ascending=True)
# print(mean_li_all_death_with_med['year'])
# plt.plot(range(len(mean_li_all_death_with_med['year'])), mean_li_all_death_with_med['mean']/pop_model['mean'][0:len(mean_li_all_death_with_med['mean'])] * 100000, color = 'blue')
# plt.title("mean_li_all_death_with_med")
# plt.ylabel("Deaths due to RTI per 100,000")
# plt.xlabel("Year")
# plt.show()
