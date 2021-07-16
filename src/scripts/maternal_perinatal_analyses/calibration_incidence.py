from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo.analysis.utils import (
    extract_results, extract_str_results,
    get_scenario_outputs,
)

# %% Declare the name of the file that specified the scenarios used in this run.
scenario_filename = 'multi_run_calibration.py'  # <-- update this to look at other results

# %% Declare usual paths:
outputspath = Path('./outputs/sejjj49@ucl.ac.uk/')
rfp = Path('./resources')

# Find results folder (most recent run generated using that scenario_filename)
results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]
# create_pickles_locally(results_folder)  # if not created via batch


def get_extracted_df(module, key, column):
    extracted_df = extract_str_results(results_folder,
                                       module=f"tlo.methods.{module}",
                                       key=key,
                                       column=column)
    extracted_df.columns = extracted_df.columns.get_level_values(0)
    return extracted_df


def return_df_with_total_non_null_values_per_draw(extracted_df):
    df = pd.DataFrame(index=[0])

    for column in extracted_df:
        null_values = extracted_df[column].isnull()
        df[f"draw {column}"] = len(null_values.loc[~null_values].index)

    return df

# EXTRACTING MEAN BIRTHS AND TOTAL PREGNANCIES


def get_mean_births_across_draws():
    extracted_births = get_extracted_df('demography', 'on_birth', 'child')
    births = return_df_with_total_non_null_values_per_draw(extracted_births)
    mean_births = births.mean(axis=1)
    lower_q_births = births.quantile(0.025, axis=1)
    upper_q_births = births.quantile(0.975, axis=1)

    return[int(mean_births.values), int(lower_q_births.values), int(upper_q_births.values)]


def get_mean_pregnancies_across_draws():
    extracted_preg_poll_preg = get_extracted_df('contraception', 'pregnant_at_age', 'person_id')
    pp_preg = return_df_with_total_non_null_values_per_draw(extracted_preg_poll_preg)
    extracted_con_fail_preg = get_extracted_df('contraception', 'fail_contraception', 'woman_index')
    cf_preg = return_df_with_total_non_null_values_per_draw(extracted_con_fail_preg)
    mean_preg = (pp_preg.mean(axis=1) + cf_preg.mean(axis=1))
    lower_q_preg = (pp_preg.quantile(0.025, axis=1) + cf_preg.quantile(0.025, axis=1))
    upper_q_preg = (pp_preg.quantile(0.975, axis=1) + cf_preg.quantile(0.975, axis=1))

    return [int(mean_preg.values), int(lower_q_preg.values), int(upper_q_preg.values)]


def get_extracted_comp_df(module):
    extracted_df = extract_str_results(results_folder,
                                       module=f"tlo.methods.{module}",
                                       key="maternal_complication",
                                       column="type")
    extracted_df.columns = extracted_df.columns.get_level_values(0)
    return extracted_df


def get_df_comp_frequency_per_run(extracted_df, complication):
    new_df = pd.DataFrame(index=[0])
    for column in extracted_df:
        new_df[f"draw {column}"] = len(extracted_df.loc[extracted_df[column] == complication])
    return new_df


def get_mean_incidence_rate_and_quantiles(extracted_df, complication, denominator):
    df_comp_frequency = get_df_comp_frequency_per_run(extracted_df, complication)

    mean = df_comp_frequency.mean(axis=1)
    rate = (mean / denominator) * 1000
    lq = (df_comp_frequency.quantile(0.025, axis=1) / denominator) * 1000
    uq = (df_comp_frequency.quantile(0.925, axis=1) / denominator) * 1000

    return[int(mean.values), int(rate.values), int(lq.values), int(uq.values)]


def get_total_pregnancies_that_have_ended(extracted_df):
    ended_preg = list()

    for complication in ['ectopic_unruptured', 'spontaneous_abortion', 'induced_abortion']:
        values = get_mean_incidence_rate_and_quantiles(extracted_df, complication, 1000)
        ended_preg.append(values[0])

    stillbirths_extracted = get_extracted_df('pregnancy_supervisor', 'antenatal_stillbirth', 'mother')
    stillbirths = return_df_with_total_non_null_values_per_draw(stillbirths_extracted)
    mean = stillbirths.mean(axis=1)
    ended_preg.append(int(mean.values))

    births = get_mean_births_across_draws()
    ended_preg.append(births[0])

    return sum(ended_preg)


def show_graph_with_quartiles(extracted_df, complication, denominator, target, colours, title, ylabel, labels):
    values = get_mean_incidence_rate_and_quantiles(extracted_df, complication, denominator)
    print(f'{complication} rate', values[1])
    model_rates = [values[1]]
    target_rates = [target]
    ci = [(values[2], values[3])]

    barWidth = 0.35
    y_r = [model_rates[i] - ci[i][1] for i in range(len(ci))]
    r1 = np.arange(len(model_rates))
    r2 = [x + barWidth for x in r1]

    plt.bar(r1, model_rates, width=barWidth, color=colours[0], yerr=y_r, capsize=7, label='model')
    plt.bar(r2, target_rates, width=barWidth, color=colours[1], capsize=7, label='target')

    plt.title(title)
    plt.xticks([r + barWidth for r in range(len(model_rates))], labels)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


antenatal_comps = get_extracted_comp_df('pregnancy_supervisor')
births = get_mean_births_across_draws()
pregnancies = get_mean_pregnancies_across_draws()
total_ended_pregnancies = get_total_pregnancies_that_have_ended(antenatal_comps)

# TODO: check on next run...
show_graph_with_quartiles(antenatal_comps, 'spontaneous_abortion', total_ended_pregnancies, 189, ['blue', 'green'],
                          'Rate of miscarriage in 2010', 'Rate per 1000 pregnancies', ['SA'])
show_graph_with_quartiles(antenatal_comps, 'induced_abortion', total_ended_pregnancies, 86, ['grey', 'pink'],
                          'Rate of abortion in 2010', 'Rate per 1000 pregnancies', ['IA'])
show_graph_with_quartiles(antenatal_comps, 'PROM', births[0], 27, ['grey', 'pink'],
                          'Rate of PROM in 2010', 'Rate per 1000 births', ['PROM'])


# todo: denom needs to only include pregnancies for which entirety of risk has been applied ???
# ectopic pregnancy - denom = all pregnancies (as risk is applied on conception)
# multiples - denom = all pregnancies (as risk is applied on conception)
# praevia - denom = all pregnancies (as risk is applied on conception)

# spontaneous abortion - denom = total pregnancies for which total risk has been applied + abortions...?
