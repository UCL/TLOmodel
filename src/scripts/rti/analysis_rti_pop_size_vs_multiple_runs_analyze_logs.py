from matplotlib import pyplot as plt
import numpy as np
import os
from pathlib import Path
import re
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
# Here I am trying to find out if it is better to do single runs with a larger population size vs
# multiple runs with smaller population sizes
# Create function to get the simulation run time

def get_simulation_time(df):
    sim_time_string = df['tlo.simulation']['info']['message'].to_list()[-1]
    seconds = [float(s) for s in re.findall(r'-?\d+\.?\d*', sim_time_string)][0]
    return seconds


# Run a small population size multiple times, storing a few outputs and the simulation time for each run
# create lists to store outputs from the multiple simulations
multiple_sim_run_time = 0
incidence_of_rti_per_month = []
incidence_of_rti_death_per_month = []
prop_circumcised = []
prop_female_sex_worker = []
sim_time_intervals = []
per_run_incidence_of_rti = []
per_run_incidence_of_rti_death = []
per_run_prop_circumcised = []
per_run_prop_female_sex_worker = []

multiple_run_logs = [logfile for logfile in os.listdir("outputs/multiple_runs_vs_population_size") if 'large_pop_size'
                     not in logfile]
smaller_pop_size = []
nsim = len(multiple_run_logs)
for logfile in multiple_run_logs:
    # parse the log file
    log_df = parse_log_file("outputs/multiple_runs_vs_population_size/" + logfile)
    # get the simulation population size
    smaller_pop_size.append(log_df['tlo.methods.demography']['population']['total'].to_list()[0])
    # get the simulation time
    this_sim_time = get_simulation_time(log_df)
    # store in the multiple run sim time total
    multiple_sim_run_time += this_sim_time
    # store the cumulative sum of time
    sim_time_intervals.append(multiple_sim_run_time)
    # get the incidence of rti from this simulation
    rti_outputs = log_df['tlo.methods.rti']
    incidence_of_rti_in_sim = rti_outputs['summary_1m']['incidence of rti per 100,000'].to_list()
    # store each monthly estimate in the list incidence_of_rti_per_month
    for estimate in incidence_of_rti_in_sim:
        incidence_of_rti_per_month.append(estimate)
    # store the simulation monthly estimates in the list per_run_incidence_of_rti
    per_run_incidence_of_rti.append(incidence_of_rti_in_sim)
    # get the incidence of rti death
    incidence_of_death_in_sim = rti_outputs['summary_1m']['incidence of rti death per 100,000'].to_list()
    # store each monthly estimates in the list incidence_of_rti_death_per_month
    for estimate in incidence_of_death_in_sim:
        incidence_of_rti_death_per_month.append(estimate)
    # store the simulation monthly estimates in the list per_run_incidence_of_rti_death
    per_run_incidence_of_rti_death.append(incidence_of_death_in_sim)
    # get the enhanced lifestyle outputs
    enhanced_lifestyle_outputs = log_df['tlo.methods.enhanced_lifestyle']
    # get the proportion of adult men who are circumcised
    proportion_circumcised = enhanced_lifestyle_outputs['prop_adult_men_circumcised']['item_1'].to_list()
    # store each monthly estimates in the list prop_circumcised
    for estimate in proportion_circumcised:
        prop_circumcised.append(estimate)
    # store the simulation monthly estiamtes in the list per_run_prop_circumcised
    per_run_prop_circumcised.append(proportion_circumcised)
    # get the proportion of females aged 15 to 49 who are sex workers
    proportion_females_aged_1549_sexworkers = \
        enhanced_lifestyle_outputs['proportion_1549_women_sexworker']['item_1'].to_list()
    # store each monthly estimates in the list prop_female_sex_worker
    for estimate in proportion_females_aged_1549_sexworkers:
        prop_female_sex_worker.append(estimate)
    # store the simulation monthly estiamtes in the list per_run_prop_female_sex_worker
    per_run_prop_female_sex_worker.append(proportion_females_aged_1549_sexworkers)
# get smaller pop size
smaller_pop_size = smaller_pop_size[0]
# create 'cumulative' lists of the models outputs for each simulation run to see the the effect running
# multiple model runs on the mean and standard deviations of different model outputs, specifically we are creating a
# list of lists for n simulations of the form:
# [[sim_res_1], [sim_res_1, sim_res_2], [sim_res_1, sim_res_2, sim_res_3], ... [sim_res_1, ..., sim_res_n-1, sim_res_n]]
# sample RTI outputs
cumulative_list_of_rti_incidence = [sum(per_run_incidence_of_rti[0:i+1], []) for i in
                                    range(len(per_run_incidence_of_rti))]

cumulative_list_of_rti_death_incidence = [sum(per_run_incidence_of_rti_death[0:i+1], []) for i in
                                          range(len(per_run_incidence_of_rti_death))]

# sample enhanced lifestyle outputs
cumulative_list_of_prop_circumcised = [sum(per_run_prop_circumcised[0:i+1], []) for i in
                                       range(len(per_run_prop_circumcised))]
cumulative_list_prop_female_sex_worker = [sum(per_run_prop_female_sex_worker[0:i+1], []) for i in
                                          range(len(per_run_prop_female_sex_worker))]
# calculate the mean estimate of rti incidence for each additional simulation
mean_rti_inc_per_extra_sim = []
for i in cumulative_list_of_rti_incidence:
    mean_rti_inc_per_extra_sim.append(np.mean(i))
# calculate the standard deviation in the estimate of rti incidence for each additional simulation
std_rti_inc_per_extra_sim = []
for i in cumulative_list_of_rti_incidence:
    std_rti_inc_per_extra_sim.append(np.std(i))
# calculate the mean estimate of rti incidence of death for each additional simulation
mean_rti_death_inc_per_extra_sim = []
for i in cumulative_list_of_rti_death_incidence:
    mean_rti_death_inc_per_extra_sim.append(np.mean(i))
# calculate the standard deviation in the estimate of rti incidence of death for each additional simulation
std_rti_death_inc_per_extra_sim = []
for i in cumulative_list_of_rti_death_incidence:
    std_rti_death_inc_per_extra_sim.append(np.std(i))

# calculate the mean estimate of enhanced_lifestyle proportion of adult males circumcised for each additional simulation
mean_prop_circ_per_extra_sim = []
for i in cumulative_list_of_prop_circumcised:
    mean_prop_circ_per_extra_sim.append(np.mean(i))
# calculate the standard deviation of enhanced_lifestyle proportion of adult males circumcised for each additional
# simulation
std_prop_circ_per_extra_sim = []
for i in cumulative_list_of_prop_circumcised:
    std_prop_circ_per_extra_sim.append(np.std(i))
# calculate the mean estimate of enhanced_lifestyle proportion of females aged 15 to 49 who are sex workers
# for each additional simulation
mean_prop_female_sex_worker_per_extra_sim = []
for i in cumulative_list_prop_female_sex_worker:
    mean_prop_female_sex_worker_per_extra_sim.append(np.mean(i))
# calculate the standard deviation of enhanced_lifestyle proportion of females aged 15 to 49 who are sex workers
# for each additional simulation
std_prop_female_sex_worker_per_extra_sim = []
for i in cumulative_list_prop_female_sex_worker:
    std_prop_female_sex_worker_per_extra_sim.append(np.std(i))

# plot the effect of additional simulations of the estimated incidence of RTI and the variation
plt.bar(np.arange(len(sim_time_intervals)), mean_rti_inc_per_extra_sim, yerr=std_rti_inc_per_extra_sim,
        color='lightsteelblue')
plt.xticks(np.arange(len(sim_time_intervals)), [int(s) for s in sim_time_intervals])
plt.xlabel('Seconds')
plt.ylabel('Incidence of RTI')
plt.title(f"The effect of number of repeated simulations \n on the estimated incidence of RTI and the \n"
          f"variation seen in model outputs, {nsim} simulations, {smaller_pop_size} population")
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/PopSizeVsReps/Reps_inc_rti.png",
            bbox_inches='tight')
plt.clf()
# plot the effect of additional simulations of the estimated incidence of RTI death and the variation
plt.bar(np.arange(len(sim_time_intervals)), mean_rti_death_inc_per_extra_sim, yerr=std_rti_death_inc_per_extra_sim,
        color='lightsteelblue')
plt.xticks(np.arange(len(sim_time_intervals)), [int(s) for s in sim_time_intervals])
plt.xlabel('Seconds')
plt.ylabel('Incidence of RTI death')
plt.title(f"The effect of number of repeated simulations \n on the estimated incidence of RTI death and the \n"
          f"variation seen in model outputs, {nsim} simulations, {smaller_pop_size} population")
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/PopSizeVsReps/Reps_inc_rti_deaths.png",
            bbox_inches='tight')
plt.clf()
# plot the effect of additional simulations of the estimated incidence of RTI standard deviation
plt.bar(np.arange(len(sim_time_intervals)), std_rti_inc_per_extra_sim, color='lightsalmon')
plt.xticks(np.arange(len(sim_time_intervals)), [int(s) for s in sim_time_intervals])
plt.xlabel('Seconds')
plt.ylabel('Standard deviation of incidence of RTI')
plt.title(f"The effect of number of repeated simulations \n on the standard deviation of the incidence of RTI and the "
          f"\n variation seen in model outputs, {nsim} simulations, {smaller_pop_size} population")
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/PopSizeVsReps/Reps_std_inc_rti.png",
            bbox_inches='tight')
plt.clf()
# plot the effect of additional simulations of the estimated incidence of RTI death standard deviation
plt.bar(np.arange(len(sim_time_intervals)), std_rti_death_inc_per_extra_sim, color='lightsalmon')
plt.xticks(np.arange(len(sim_time_intervals)), [int(s) for s in sim_time_intervals])
plt.xlabel('Seconds')
plt.ylabel('Standard deviation of incidence of RTI death')
plt.title(f"The effect of number of repeated simulations \n on the standard deviation of the incidence of RTI death and"
          f"the \n variation seen in model outputs, {nsim} simulations, {smaller_pop_size} population")
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/PopSizeVsReps/Reps_std_inc_rti_death.png",
            bbox_inches='tight')
plt.clf()
# plot the effect of additional simulations of the estimated incidence of enhanced lifestyle proportion of adult males
# who are circumcised
plt.bar(np.arange(len(sim_time_intervals)), mean_prop_circ_per_extra_sim, yerr=std_prop_circ_per_extra_sim,
        color='lightsteelblue')
plt.xticks(np.arange(len(sim_time_intervals)), [int(s) for s in sim_time_intervals])
plt.xlabel('Seconds')
plt.ylabel('Proportion of adult males who are circumcised')
plt.title(f"The effect of number of repeated simulations \n on the proportion of adult males who are circumcised"
          f"\n and the variation seen in model outputs, {nsim} simulations, {smaller_pop_size} population")
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/PopSizeVsReps/Reps_prop_adult_male_circ.png",
            bbox_inches='tight')
plt.clf()
# plot the effect of additional simulations of the estimated incidence of enhanced lifestyle proportion of females aged
# 15-49 who are sex workers
plt.bar(np.arange(len(sim_time_intervals)), mean_prop_female_sex_worker_per_extra_sim,
        yerr=std_prop_female_sex_worker_per_extra_sim, color='lightsteelblue')
plt.xticks(np.arange(len(sim_time_intervals)), [int(s) for s in sim_time_intervals])
plt.xlabel('Seconds')
plt.ylabel('Proportion of females aged \n15-49 who are sex workers')
plt.title(f"The effect of number of repeated simulations \n on the proportion of females aged 15-49 who are sex workers"
          f"\n and the variation seen in model outputs, {nsim} simulations, {smaller_pop_size} population")
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/PopSizeVsReps/Reps_prop_female_sex_worker.png",
            bbox_inches='tight')
plt.clf()
# plot the effect of additional simulations of the estimated incidence of RTI standard deviation
plt.bar(np.arange(len(sim_time_intervals)), std_prop_female_sex_worker_per_extra_sim, color='lightsalmon')
plt.xticks(np.arange(len(sim_time_intervals)), [int(s) for s in sim_time_intervals])
plt.xlabel('Seconds')
plt.ylabel('Standard deviation of proportion of females aged \n15-49 who are sex workers')
plt.title(f"The effect of number of repeated simulations \n on the standard deviation of the proportion of females \n"
          f"aged 15-49 who are sex workers, {nsim} simulations, {smaller_pop_size} population")
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/PopSizeVsReps/Reps_std_prop_female_sex_worker"
            ".png", bbox_inches='tight')
plt.clf()
# plot the effect of additional simulations of the estimated incidence of enhanced_lifestye proportion of circumcised
# standard deviation
plt.bar(np.arange(len(sim_time_intervals)), std_rti_death_inc_per_extra_sim, color='lightsalmon')
plt.xticks(np.arange(len(sim_time_intervals)), [int(s) for s in sim_time_intervals])
plt.xlabel('Seconds')
plt.ylabel('Standard deviation of the proportion of circumcised adult males')
plt.title(f"The effect of number of repeated simulations \n on the standard deviation of the proportion of cirumcised\n"
          f"adult males, {nsim} simulations, {smaller_pop_size} population")
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/PopSizeVsReps/Reps_std_prop_circ.png",
            bbox_inches='tight')
plt.clf()

# Get the log file of the larger population size model run
large_run_log = [logfile for logfile in os.listdir("outputs/multiple_runs_vs_population_size") if logfile
                 not in multiple_run_logs]
log_df = parse_log_file("outputs/multiple_runs_vs_population_size/" + large_run_log[0])
# get the simulation time
large_pop_sim_time = get_simulation_time(log_df)
# get the larger population size
larger_pop_size = log_df['tlo.methods.demography']['population']['total'].to_list()[0]
# get the rti outputs
rti_outputs = log_df['tlo.methods.rti']
# get the incidence of rti from this simulation
incidence_of_rti_in_sim = rti_outputs['summary_1m']['incidence of rti per 100,000'].to_list()
# get the incidence of rti death from this simulation
incidence_of_death_in_sim = rti_outputs['summary_1m']['incidence of rti death per 100,000'].to_list()
# calculate the mean incidence of RTI in the simulation
large_pop_mean_inc_rti = np.mean(incidence_of_rti_in_sim)
# calculate the standard deviation of incidence of RTI in the simulation
large_pop_std_dev_inc_rti = np.std(incidence_of_rti_in_sim)
# calculate the mean incidence of RTI death in the simulation
large_pop_mean_inc_rti_death = np.mean(incidence_of_death_in_sim)
# calculate the standard deviation of incidence of RTI death in the simulation
large_pop_std_dev_inc_rti_death = np.std(incidence_of_death_in_sim)
# get the enhanced lifestyle outputs from the simulation
enhanced_lifestyle_outputs = log_df['tlo.methods.enhanced_lifestyle']
# get the proportion of adult men who are circumcised
large_pop_proportion_circumcised = enhanced_lifestyle_outputs['prop_adult_men_circumcised']['item_1'].to_list()
# get the proportion of females aged 15 to 49 who are sex workers
large_pop_proportion_females_aged_1549_sexworkers = \
    enhanced_lifestyle_outputs['proportion_1549_women_sexworker']['item_1'].to_list()
# calculate the mean proportion of adult males who are circumcised
large_pop_mean_proportion_circumcised = np.mean(large_pop_proportion_circumcised)
# calculate the standard deviation proportion of adult males who are circumcised
large_pop_std_proportion_circumcised = np.std(large_pop_proportion_circumcised)
# calculate the mean proportion of females aged 15-49 who are sex workers
large_pop_mean_proportion_female_sex_workers = np.mean(large_pop_proportion_females_aged_1549_sexworkers)
# calculate the standard deviation proportion of females aged 15-49 who are circumcised
large_pop_std_proportion_female_sex_workers = np.std(large_pop_proportion_females_aged_1549_sexworkers)

# get the average variation per run for the multiple simulation runs
# incidence of rti
mean_incidence_per_run = [np.mean(run) for run in per_run_incidence_of_rti]
std_incidence_per_run = [np.std(run) for run in per_run_incidence_of_rti]
mean_std_incidence_per_run = np.mean(std_incidence_per_run)
# incidence of rti death
mean_incidence_death_per_run = [np.mean(run) for run in per_run_incidence_of_rti_death]
std_incidence_deaths_per_run = [np.std(run) for run in per_run_incidence_of_rti_death]
mean_std_incidence_deaths_per_run = np.mean(std_incidence_deaths_per_run)
# proportion of people circumcised
mean_prop_circumcised_per_run = [np.mean(run) for run in per_run_prop_circumcised]
std_prop_circumcised_per_run = [np.std(run) for run in per_run_prop_circumcised]
mean_std_prop_circumcised_per_run = np.mean(std_prop_circumcised_per_run)
# proportion of female sex workers
mean_prop_female_sex_worker_per_run = [np.mean(run) for run in per_run_prop_female_sex_worker]
std_prop_female_sex_worker_per_run = [np.std(run) for run in per_run_prop_female_sex_worker]
mean_std_prop_female_sex_worker_per_run = np.mean(per_run_prop_female_sex_worker)
# plot the mean estimate of RTI incidence for the multiple simulations with a small population size and the large
# population size, showing the variation in the estimates
plt.bar(np.arange(2), [np.mean(mean_incidence_per_run), large_pop_mean_inc_rti],
        yerr=[std_rti_inc_per_extra_sim[-1], large_pop_std_dev_inc_rti])
plt.xticks(np.arange(2), [f"{nsim} simulations, n = {smaller_pop_size}\n" + str(int(sim_time_intervals[-1])) +
                          " seconds",
                          f"{1} simulation, n = {larger_pop_size}\n" + str(int(large_pop_sim_time)) + " seconds"])
plt.xlabel('Simulation time')
plt.ylabel('Incidence of RTI')
plt.title('Comparing simulation time for simulations with multiple runs vs \n'
          'single run with large population size on the incidence of RTI per 100,000 person years')
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/PopSizeVsReps/multi_vs_single_inc_rti.png",
            bbox_inches='tight')
plt.clf()
# plot the values for the standard deviation of RTI incidence for the multiple simulations with a small population size
# and the large population size,
plt.bar(np.arange(2), [mean_std_incidence_per_run, large_pop_std_dev_inc_rti], color='lightsalmon')
plt.xticks(np.arange(2), [f"{nsim} simulations, n = {smaller_pop_size}\n" + str(int(sim_time_intervals[-1])) +
                          " seconds",
                          f"{1} simulation, n = {larger_pop_size}\n" + str(int(large_pop_sim_time)) + " seconds"])
plt.xlabel('Simulation time')
plt.ylabel('Standard deviation of the incidence of RTI')
plt.title('Comparing simulation time for simulations with multiple runs vs \n'
          'single run with large population size on the s.d. of incidence of RTI per 100,000 person years')
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/PopSizeVsReps/multi_vs_single_std_inc_rti.png",
            bbox_inches='tight')
plt.clf()
# plot the mean estimate of RTI death incidence for the multiple simulations with a small population size and the large
# population size, showing the variation in the estimates
plt.bar(np.arange(2), [np.mean(mean_incidence_death_per_run), large_pop_mean_inc_rti_death],
        yerr=[std_rti_death_inc_per_extra_sim[-1], large_pop_std_dev_inc_rti_death])
plt.xticks(np.arange(2), [f"{nsim} simulations, n = {smaller_pop_size}\n" + str(int(sim_time_intervals[-1])) +
                          " seconds",
                          f"{1} simulation, n = {larger_pop_size}\n" + str(int(large_pop_sim_time)) + " seconds"])
plt.xlabel('Simulation time')
plt.ylabel('Incidence of RTI death')
plt.title('Comparing simulation time for simulations with multiple runs vs \n'
          'single run with large population size on the incidence of RTI death per 100,000 person years')
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/PopSizeVsReps/multi_vs_single_inc_rti_death.png",
            bbox_inches='tight')
plt.clf()
# plot the values for the standard deviation of RTI death incidence for the multiple simulations with a small population
# size and the large population size,
plt.bar(np.arange(2), [mean_std_incidence_deaths_per_run, large_pop_std_dev_inc_rti_death], color='lightsalmon')
plt.xticks(np.arange(2), [f"{nsim} simulations, n = {smaller_pop_size}\n" + str(int(sim_time_intervals[-1])) +
                          " seconds",
                          f"{1} simulation, n = {larger_pop_size}\n" + str(int(large_pop_sim_time)) + " seconds"])
plt.xlabel('Simulation time')
plt.ylabel('Standard deviation of the incidence of RTI death')
plt.title('Comparing simulation time for simulations with multiple runs vs \n'
          'single run with large population size on the s.d. of incidence of RTI death per 100,000 person years')
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/PopSizeVsReps/multi_vs_single_std_inc_rti_death"
            ".png", bbox_inches='tight')
plt.clf()
# plot the mean estimate of the proportion of circumcised adult males for the multiple simulations with a
# small population size and the large population size, showing the variation in the estimates
plt.bar(np.arange(2), [np.mean(mean_prop_circumcised_per_run), large_pop_mean_proportion_circumcised],
        yerr=[std_prop_circ_per_extra_sim[-1], large_pop_std_proportion_circumcised])
plt.xticks(np.arange(2), [f"{nsim} simulations, n = {smaller_pop_size}\n" + str(int(sim_time_intervals[-1])) +
                          " seconds",
                          f"{1} simulation, n = {larger_pop_size}\n" + str(int(large_pop_sim_time)) + " seconds"])
plt.xlabel('Simulation time')
plt.ylabel('Proportion of adult males who are circumcised')
plt.title('Comparing simulation time for simulations with multiple runs vs \n'
          'single run with large population size on the proportion of adult males who are circumcised')
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/PopSizeVsReps/multi_vs_single_prop_adult_male_"
            "circumcised.png", bbox_inches='tight')
plt.clf()
# plot the values for the standard deviation of proprtion of adult males who are circumcised for the multiple
# simulations with a small population size and the large population size,
plt.bar(np.arange(2), [mean_std_prop_circumcised_per_run, large_pop_std_proportion_circumcised], color='lightsalmon')
plt.xticks(np.arange(2), [f"{nsim} simulations, n = {smaller_pop_size}\n" + str(int(sim_time_intervals[-1])) +
                          " seconds",
                          f"{1} simulation, n = {larger_pop_size}\n" + str(int(large_pop_sim_time)) + " seconds"])
plt.xlabel('Simulation time')
plt.ylabel('Standard deviation of the proportion \nof adult males who are circumcised')
plt.title('Comparing simulation time for simulations with multiple runs vs \n'
          'single run with large population size on the s.d. of proportion of \nadult males who are circumcised')
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/PopSizeVsReps/multi_vs_single_circ_adult_males"
            ".png", bbox_inches='tight')
plt.clf()
# plot the mean estimate of the proportion of females ages 15-49 who are sex workers for the multiple simulations with a
# small population size and the large population size, showing the variation in the estimates
plt.bar(np.arange(2),
        [np.mean(mean_prop_female_sex_worker_per_extra_sim), large_pop_mean_proportion_female_sex_workers],
        yerr=[std_prop_female_sex_worker_per_extra_sim[-1], large_pop_std_proportion_female_sex_workers])
plt.xticks(np.arange(2), [f"{nsim} simulations, n = {smaller_pop_size}\n" + str(int(sim_time_intervals[-1])) +
                          " seconds",
                          f"{1} simulation, n = {larger_pop_size}\n" + str(int(large_pop_sim_time)) + " seconds"])
plt.xlabel('Simulation time')
plt.ylabel('Proportion of females aged \n15-49 who are sex workers')
plt.title('Comparing simulation time for simulations with multiple runs vs \n'
          'single run with large population size on the proportion of females ages 15-49 who are sex workers')
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/PopSizeVsReps/multi_vs_single_prop_females_aged_"
            "1549_sex_workers.png", bbox_inches='tight')
plt.clf()
# plot the values for the standard deviation of proprtion of females ages 15-49 who are circumcised for the multiple
# simulations with a small population size and the large population size,
plt.bar(np.arange(2), [mean_std_prop_female_sex_worker_per_run, large_pop_std_proportion_female_sex_workers],
        color='lightsalmon')
plt.xticks(np.arange(2), [f"{nsim} simulations, n = {smaller_pop_size}\n" + str(int(sim_time_intervals[-1])) +
                          " seconds",
                          f"{1} simulation, n = {larger_pop_size}\n" + str(int(large_pop_sim_time)) + " seconds"])
plt.xlabel('Simulation time')
plt.ylabel('Standard deviation of the proportion of females aged\n 15-49 who are sex workers')
plt.title('Comparing simulation time for simulations with multiple runs vs \n'
          'single run with large population size on the s.d. of proportion of females aged \n15-49 who are sex workers')
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/PopSizeVsReps/multi_vs_single_female_sex_workers"
            ".png", bbox_inches='tight')
plt.clf()
