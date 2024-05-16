import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt

from tlo.analysis.utils import extract_results

plt.style.use('seaborn-darkgrid')

"""This file contains functions used through maternal/perinatal analysis and calibration scripts to extract data,
derive outcomes and generate plots"""


def return_95_CI_across_runs(df, sim_years):
    """Returns a list of lists from an outcome DF containing the mean and 95%CI values for that outcome over time.
    The first list containins the mean value of a given outcome per year across runs, the second contains the smaller
     value of the 95% CI and the third list contains the larger value of the 95% Ci"""

    year_means = list()
    lower_CI = list()
    upper_CI = list()

    for year in sim_years:
        if year in df.index:
            row = df.loc[year]
            year_means.append(row.mean())
            ci = st.t.interval(0.95, len(row) - 1, loc=np.mean(row), scale=st.sem(row))
            lower_CI.append(ci[0])
            upper_CI.append(ci[1])
        else:
            year_means.append(0)
            lower_CI.append(0)
            upper_CI.append(0)

    return [year_means, lower_CI, upper_CI]


def get_mean_95_CI_from_list(list_item):
    """Returns the mean and 95% CI of data in a provided list"""
    ci = st.t.interval(0.95, len(list_item) - 1, loc=np.mean(list_item), scale=st.sem(list_item))
    result = [np.mean(list_item), ci[0], ci[1]]
    return result


def get_mean_from_columns(df, function):
    """Returns mean value for each column in a provided data frame"""
    values = list()
    for col in df:
        if function == 'avg':
            values.append(np.mean(df[col]))
        else:
            values.append(sum(df[col]))
    return values


def line_graph_with_ci_and_target_rate(sim_years, data, target_data_dict, ylim, y_label, title,
                                       graph_location, file_name):
    """Outputs and saves line plot of an outcome over time, including uncertainty, in addition to a pre-determined
    calibration target"""
    fig, ax = plt.subplots()
    ax.plot(sim_years, data[0], label="Model", color='deepskyblue')
    ax.fill_between(sim_years, data[1], data[2], label="95% CI", color='b', alpha=.1)

    if target_data_dict['double']:
        plt.errorbar(target_data_dict['first']['year'], target_data_dict['first']['value'],
                     label=target_data_dict['first']['label'], yerr=target_data_dict['first']['ci'],
                     fmt='o', color='darkseagreen', ecolor='green', elinewidth=3, capsize=0)
        plt.errorbar(target_data_dict['second']['year'], target_data_dict['second']['value'],
                     label=target_data_dict['second']['label'], yerr=target_data_dict['second']['ci'],
                     fmt='o', color='darkseagreen', ecolor='green', elinewidth=3, capsize=0)

    elif not target_data_dict['double']:
        plt.errorbar(target_data_dict['first']['year'], target_data_dict['first']['value'],
                     label=target_data_dict['first']['label'], yerr=target_data_dict['first']['ci'],
                     fmt='o', color='darkseagreen', ecolor='green', elinewidth=3, capsize=0)

    plt.xlabel('Year')
    plt.ylabel(y_label)
    plt.title(title)
    ax.set(ylim=(0, ylim))
    plt.xticks(sim_years, labels=sim_years, rotation=45, fontsize=8)
    plt.gca().set_ylim(bottom=0)
    plt.legend()
    plt.savefig(f'{graph_location}/{file_name}.png')
    plt.show()


def simple_line_chart_with_target(sim_years, model_rate, target_rate, y_title, title, file_name, graph_location):
    """Outputs and saves line plot of an outcome over time in addition to a pre-determined
        calibration target"""
    plt.plot(sim_years, model_rate, 'o-g', label="Model", color='deepskyblue')
    plt.plot(sim_years, target_rate, 'o-g', label="Target rate", color='darkseagreen')
    plt.ylabel(y_title)
    plt.xlabel('Year')
    plt.title(title)
    plt.gca().set_ylim(bottom=0)
    plt.legend()
    plt.savefig(f'{graph_location}/{file_name}.png')
    plt.show()


def simple_line_chart_with_ci(sim_years, data, y_title, title, file_name, graph_location):
    """Outputs and saves line plot of an outcome over time, including uncertainty"""
    fig, ax = plt.subplots()
    ax.plot(sim_years, data[0], label="Model (mean)", color='deepskyblue')
    ax.fill_between(sim_years, data[1], data[2], color='b', alpha=.1, label="95% CI")
    plt.ylabel(y_title)
    plt.xlabel('Year')
    plt.title(title)
    plt.legend()
    plt.gca().set_ylim(bottom=0)
    plt.grid(True)
    plt.savefig(f'{graph_location}/{file_name}.png')
    plt.show()


def simple_bar_chart(model_rates, x_title, y_title, title, file_name, sim_years, graph_location):
    """Outputs a simple bar chart over time"""
    bars = sim_years
    x_pos = np.arange(len(bars))
    plt.bar(x_pos, model_rates, label="Model", color='thistle')
    plt.xticks(x_pos, bars, rotation=90)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    plt.legend()
    plt.savefig(f'{graph_location}/{file_name}.png')
    plt.show()


def comparison_graph_multiple_scenarios_multi_level_dict(colours, intervention_years, data_dict, key, y_label, title,
                                                         graph_location, save_name):
    """Outputs and saves line plot of an outcome over time, including uncertainty, in addition to a pre-determined
        calibration target. Data from the model is provided as a dictionary with multiple entries"""
    fig, ax = plt.subplots()

    for k, colour in zip(data_dict, colours):
        ax.plot(intervention_years, data_dict[k][key][0], label=k, color=colour)
        ax.fill_between(intervention_years, data_dict[k][key][1], data_dict[k][key][2], color=colour, alpha=.1)

    plt.ylabel(y_label)
    plt.xlabel('Year')
    plt.title(title)

    if 'nmr' in key:
        plt.gca().set_ylim(bottom=0, top=25)
    elif 'sbr' in key:
        plt.gca().set_ylim(bottom=0, top=20)
    else:
        plt.gca().set_ylim(bottom=0)

    plt.legend()
    plt.xticks(intervention_years, labels=intervention_years, rotation=45, fontsize=8)
    plt.savefig(f'./{graph_location}/{save_name}.png', bbox_inches='tight')
    plt.show()


def comparison_bar_chart_multiple_bars(data, dict_name, intervention_years, colours, y_title, title,
                                       plot_destination_folder, save_name):
    """Outputs a barchart comparing two data sources"""

    N = len(intervention_years)
    ind = np.arange(N)
    if len(data.keys()) > 3:
        width = 0.15
    else:
        width = 0.2

    x_ticks = list()
    for x in range(len(intervention_years)):
        x_ticks.append(x)

    for k, position, colour in zip(data, [ind - width, ind, ind + width, ind + width * 2, ind + width * 3],
                                   colours):
        ci = [(x - y) / 2 for x, y in zip(data[k][dict_name][2], data[k][dict_name][1])]
        plt.bar(position, data[k][dict_name][0], width, label=k, yerr=ci, color=colour)

    plt.ylabel(y_title)
    plt.xlabel('Years')
    plt.title(title)
    plt.legend(loc='best')
    plt.xticks(x_ticks, labels=intervention_years)
    plt.savefig(f'{plot_destination_folder}/{save_name}.png')
    plt.show()


# =========================== FUNCTIONS RETURNING DATA FROM MULTIPLE SCENARIOS =======================================
def get_modules_maternal_complication_dataframes(results_folder):
    """Returns a dataframe from a scenario file which contains the number of maternal complications by type per year
    for a given python script"""
    comp_dfs = dict()

    for module in ['pregnancy_supervisor', 'labour', 'postnatal_supervisor']:
        c_df = extract_results(
            results_folder,
            module=f"tlo.methods.{module}",
            key="maternal_complication",
            custom_generate_series=(
                lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'type'])['person'].count()),
            do_scaling=True
        )
        complications_df = c_df.fillna(0)

        comp_dfs[module] = complications_df

    return comp_dfs


def get_modules_neonatal_complication_dataframes(results_folder):
    """Returns a dataframe from a scenario file which contains the number of neonatal complications by type per year
        for a given python script"""
    comp_dfs = dict()

    for module in ['newborn_outcomes', 'postnatal_supervisor']:
        n_df = extract_results(
            results_folder,
            module=f"tlo.methods.{module}",
            key="newborn_complication",
            custom_generate_series=(
                lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'type'])['newborn'].count()),
            do_scaling=True
        )
        complications_df = n_df.fillna(0)

        comp_dfs[module] = complications_df

    return comp_dfs


def return_birth_data_from_multiple_scenarios(results_folders, sim_years, intervention_years):
    """ Extracts data relating to births from a series of pre-specified scenario files"""

    def extract_births(folder):
        br = extract_results(
            folder,
            module="tlo.methods.demography",
            key="on_birth",
            custom_generate_series=(
                lambda df: df.assign(
                    year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=True
        )
        births_results = br.fillna(0)
        total_births_per_year = return_95_CI_across_runs(births_results, sim_years)

        int_df = births_results.loc[intervention_years[0]: intervention_years[-1]]
        int_births_per_year = return_95_CI_across_runs(int_df, intervention_years)

        agg_births_data = get_mean_from_columns(births_results.loc[intervention_years[0]: intervention_years[-1]],
                                                'agg')
        agg_births = get_mean_95_CI_from_list(agg_births_data)

        return {'total_births': total_births_per_year,
                'int_births': int_births_per_year,
                'agg_births': agg_births,
                'births_data_frame': births_results}

    return {k: extract_births(results_folders[k]) for k in results_folders}


def return_pregnancy_data_from_multiple_scenarios(results_folders, sim_years, intervention_years):
    """ Extracts data relating to pregnancies from a series of pre-specified scenario files"""

    def extract_pregnancies(folder):
        pr = extract_results(
            folder,
            module="tlo.methods.contraception",
            key="pregnancy",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=True
        )
        preg_results = pr.fillna(0)
        total_pregnancies_per_year = return_95_CI_across_runs(preg_results, sim_years)
        total_pregnancies_per_year_int = return_95_CI_across_runs(
             preg_results.loc[intervention_years[0]: intervention_years[-1]], intervention_years)

        agg_preg_data = get_mean_from_columns(preg_results.loc[intervention_years[0]: intervention_years[-1]], 'agg')
        agg_preg = get_mean_95_CI_from_list(agg_preg_data)

        return {'total_preg': total_pregnancies_per_year,
                'int_preg': total_pregnancies_per_year_int,
                'agg_preg': agg_preg,
                'preg_data_frame': preg_results}

    return {k: extract_pregnancies(results_folders[k]) for k in results_folders}


def get_completed_pregnancies_from_multiple_scenarios(comps_df, birth_dict, results_folder):
    """Sums the number of pregnancies that have ended in a given year including ectopic pregnancies,
    abortions, stillbirths and births"""

    ansb_df = extract_results(
        results_folder,
        module="tlo.methods.pregnancy_supervisor",
        key="antenatal_stillbirth",
        custom_generate_series=(
            lambda df: df.assign(
                year=df['date'].dt.year).groupby(['year'])['year'].count()),
        do_scaling=True
    )

    eu = comps_df['pregnancy_supervisor'].loc[(slice(None), 'ectopic_unruptured'), slice(None)]
    eu_f = eu.droplevel(1)

    ia = comps_df['pregnancy_supervisor'].loc[(slice(None), 'induced_abortion'), slice(None)]
    ia_f = ia.droplevel(1)

    sa = comps_df['pregnancy_supervisor'].loc[(slice(None), 'spontaneous_abortion'), slice(None)]
    sa_f = sa.droplevel(1)

    comp_preg_data_frame = eu_f + ia_f + sa_f + ansb_df + birth_dict['births_data_frame']

    return {'comp_preg_data_frame': comp_preg_data_frame}
