import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo.analysis.utils import extract_results

plt.style.use('seaborn-darkgrid')

"""This file contains functions used through maternal/perinatal analysis and calibration scripts to extract results,
derive results and generate plots"""


# =========================================== FUNCTIONS TO EXTRACT RATES  ============================================
def get_mean_and_quants_from_str_df(df, complication, sim_years):
    yearly_mean_number = list()
    yearly_lq = list()
    yearly_uq = list()
    for year in sim_years:
        if complication in df.loc[year].index:
            yearly_mean_number.append(df.loc[year, complication].mean())
            yearly_lq.append(df.loc[year, complication].quantile(0.025))
            yearly_uq.append(df.loc[year, complication].quantile(0.925))
        else:
            yearly_mean_number.append(0)
            yearly_lq.append(0)
            yearly_uq.append(0)

    return [yearly_mean_number, yearly_lq, yearly_uq]


def get_comp_mean_and_rate(complication, denominator_list, df, rate, years):
    yearly_means = get_mean_and_quants_from_str_df(df, complication, years)[0]
    yearly_lq = get_mean_and_quants_from_str_df(df, complication, years)[1]
    yearly_uq = get_mean_and_quants_from_str_df(df, complication, years)[2]

    yearly_mean_rate = [(x / y) * rate for x, y in zip(yearly_means, denominator_list)]
    yearly_lq_rate = [(x / y) * rate for x, y in zip(yearly_lq, denominator_list)]
    yearly_uq_rate = [(x / y) * rate for x, y in zip(yearly_uq, denominator_list)]

    return [yearly_mean_rate, yearly_lq_rate, yearly_uq_rate]


def get_mean_and_quants(df, sim_years):
    year_means = list()
    lower_quantiles = list()
    upper_quantiles = list()

    for year in sim_years:
        if year in df.index:
            year_means.append(df.loc[year].mean())
            lower_quantiles.append(df.loc[year].quantile(0.025))
            upper_quantiles.append(df.loc[year].quantile(0.925))
        else:
            year_means.append(0)
            lower_quantiles.append(0)
            upper_quantiles.append(0)

    return [year_means, lower_quantiles, upper_quantiles]


def get_comp_mean_and_rate_across_multiple_dataframes(complication, denominators, rate, dataframes, sim_years):
    def get_list_of_rates_and_quants(df):
        rates_per_year = list()
        lq_per_year = list()
        uq_per_year = list()
        for year, denominator in zip(sim_years, denominators):
            if year in df.index:
                if complication in df.loc[year].index:
                    rates = (df.loc[year, complication].mean() / denominator) * rate
                    lq = (df.loc[year, complication].quantile(0.025) / denominator) * rate
                    uq = (df.loc[year, complication].quantile(0.925) / denominator) * rate
                    rates_per_year.append(rates)
                    lq_per_year.append(lq)
                    uq_per_year.append(uq)

                else:
                    rates_per_year.append(0)
                    lq_per_year.append(0)
                    uq_per_year.append(0)
            else:
                rates_per_year.append(0)
                lq_per_year.append(0)
                uq_per_year.append(0)

        return [rates_per_year, lq_per_year, uq_per_year]

    if len(dataframes) == 2:
        df_1_data = get_list_of_rates_and_quants(dataframes[0])
        df_2_data = get_list_of_rates_and_quants(dataframes[1])

        total_rates = [x + y for x, y in zip(df_1_data[0], df_2_data[0])]
        total_lq = [x + y for x, y in zip(df_1_data[1], df_2_data[1])]
        total_uq = [x + y for x, y in zip(df_1_data[2], df_2_data[2])]

    else:
        df_1_data = get_list_of_rates_and_quants(dataframes[0])
        df_2_data = get_list_of_rates_and_quants(dataframes[1])
        df_3_data = get_list_of_rates_and_quants(dataframes[2])

        total_rates = [x + y + z for x, y, z in zip(df_1_data[0], df_2_data[0], df_3_data[0])]
        total_lq = [x + y + z for x, y, z in zip(df_1_data[1], df_2_data[1], df_3_data[1])]
        total_uq = [x + y + z for x, y, z in zip(df_1_data[2], df_2_data[2], df_3_data[2])]

    return [total_rates, total_lq, total_uq]


def get_modules_maternal_complication_dataframes(results_folder):
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


def line_graph_with_ci_and_target_rate(sim_years, mean_list, lq_list, uq_list, target_data_dict, y_label, title,
                                       graph_location, file_name):
    fig, ax = plt.subplots()
    ax.plot(sim_years, mean_list, label="Model", color='deepskyblue')
    ax.fill_between(sim_years, lq_list, uq_list, color='b', alpha=.1)

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

    if ('anc_prop' in file_name) or ('sba' in file_name) or ('pnc' in file_name):
        ax.set(ylim=(0,100))
    if 'caesarean' in file_name:
        ax.set(ylim=(0,10))

    plt.gca().set_ylim(bottom=0)
    plt.legend()
    plt.savefig(f'{graph_location}/{file_name}.png')
    plt.show()


def basic_comparison_graph(intervention_years, bdata, idata, x_label, title, graph_location, save_name):
    fig, ax = plt.subplots()
    ax.plot(intervention_years, bdata[0], label="Baseline (mean)", color='deepskyblue')
    ax.fill_between(intervention_years, bdata[1], bdata[2], color='b', alpha=.1, label="UI (2.5-92.5)")
    ax.plot(intervention_years, idata[0], label="Intervention (mean)", color='olivedrab')
    ax.fill_between(intervention_years, idata[1], idata[2], color='g', alpha=.1, label="UI (2.5-92.5)")
    plt.ylabel('Year')
    plt.xlabel(x_label)
    plt.title(title)
    plt.gca().set_ylim(bottom=0)
    plt.legend()
    plt.savefig(f'{graph_location}/{save_name}.png')
    plt.show()


def simple_line_chart(sim_years, model_rate, y_title, title, file_name, graph_location):
    plt.plot(sim_years, model_rate, 'o-g', label="Model", color='deepskyblue')
    plt.ylabel(y_title)
    plt.xlabel('Year')
    plt.title(title)
    plt.gca().set_ylim(bottom=0)
    plt.legend()
    plt.savefig(f'{graph_location}/{file_name}.png')
    plt.show()


def simple_line_chart_with_target(sim_years, model_rate, target_rate, y_title, title, file_name, graph_location):
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
    fig, ax = plt.subplots()
    ax.plot(sim_years, data[0], label="Model (mean)", color='deepskyblue')
    ax.fill_between(sim_years, data[1], data[2], color='b', alpha=.1, label="UI (2.5-92.5)")
    plt.ylabel(y_title)
    plt.xlabel('Year')
    plt.title(title)
    plt.legend()
    plt.gca().set_ylim(bottom=0)
    plt.grid(True)
    plt.savefig(f'{graph_location}/{file_name}.png')
    plt.show()


def simple_bar_chart(model_rates, x_title, y_title, title, file_name, sim_years, graph_location):
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


def return_median_and_mean_squeeze_factor_for_hsi(folder, hsi_string, sim_years, graph_location):
    hsi_med = extract_results(
        folder,
        module="tlo.methods.healthsystem",
        key="HSI_Event",
        custom_generate_series=(
            lambda df: df.loc[df['TREATMENT_ID'].str.contains(hsi_string) & df['did_run']].assign(
                year=df['date'].dt.year).groupby(['year'])['Squeeze_Factor'].median()))

    hsi_mean = extract_results(
        folder,
        module="tlo.methods.healthsystem",
        key="HSI_Event",
        custom_generate_series=(
            lambda df: df.loc[df['TREATMENT_ID'].str.contains(hsi_string) & df['did_run']].assign(
                year=df['date'].dt.year).groupby(['year'])['Squeeze_Factor'].mean()))

    median = [hsi_med.loc[year].median() for year in sim_years]
    lq = [hsi_med.loc[year].quantile(0.025) for year in sim_years]
    uq = [hsi_med.loc[year].quantile(0.925) for year in sim_years]
    data = [median, lq, uq]

    simple_line_chart_with_ci(sim_years, data, 'Median Squeeze Factor', f'Median Yearly Squeeze for HSI {hsi_string}',
                              f'median_sf_{hsi_string}', graph_location)

    mean = [hsi_mean.loc[year].mean() for year in sim_years]
    lq = [hsi_mean.loc[year].quantile(0.025) for year in sim_years]
    uq = [hsi_mean.loc[year].quantile(0.925) for year in sim_years]
    data = [mean, lq, uq]

    simple_line_chart_with_ci(sim_years, data, 'Mean Squeeze Factor', f'Mean Yearly Squeeze for HSI {hsi_string}',
                              f'mean_sf_{hsi_string}', graph_location)


def return_squeeze_plots_for_hsi(folder, hsi_string, sim_years, graph_location):

    hsi = extract_results(
        folder,
        module="tlo.methods.healthsystem",
        key="HSI_Event",
        custom_generate_series=(
            lambda df: df.loc[df['TREATMENT_ID'].str.contains(hsi_string) & df['did_run']].assign(
                year=df['date'].dt.year).groupby(['year'])['Squeeze_Factor'].mean()))

    mean_squeeze_per_year = [hsi.loc[year].to_numpy().mean() for year in sim_years]
    lq_squeeze_per_year = [np.percentile(hsi.loc[year].to_numpy(), 2.5) for year in sim_years]
    uq_squeeze_per_year = [np.percentile(hsi.loc[year].to_numpy(), 92.5) for year in sim_years]
    mean_data = [mean_squeeze_per_year, lq_squeeze_per_year, uq_squeeze_per_year]

    hsi_med = extract_results(
        folder,
        module="tlo.methods.healthsystem",
        key="HSI_Event",
        custom_generate_series=(
            lambda df: df.loc[df['TREATMENT_ID'].str.contains(hsi_string) & df['did_run']].assign(
                year=df['date'].dt.year).groupby(['year'])['Squeeze_Factor'].median()))

    median = [hsi_med.loc[year].median() for year in sim_years]

    hsi_count = extract_results(
        folder,
        module="tlo.methods.healthsystem",
        key="HSI_Event",
        custom_generate_series=(
            lambda df: df.loc[df['TREATMENT_ID'].str.contains(hsi_string) & df['did_run']].assign(
                year=df['date'].dt.year).groupby(['year'])['year'].count()))

    hsi_squeeze = extract_results(
        folder,
        module="tlo.methods.healthsystem",
        key="HSI_Event",
        custom_generate_series=(
            lambda df:
            df.loc[(df['TREATMENT_ID'].str.contains(hsi_string)) & df['did_run'] & (df['Squeeze_Factor'] > 0)
                   ].assign(year=df['date'].dt.year).groupby(['year'])['year'].count()))

    prop_squeeze_year = [(hsi_squeeze.loc[year].to_numpy().mean() / hsi_count.loc[year].to_numpy().mean()) * 100
                         for year in sim_years]
    prop_squeeze_lq = [
        (np.percentile(hsi_squeeze.loc[year].to_numpy(), 2.5) /
         np.percentile(hsi_count.loc[year].to_numpy(), 2.5)) * 100 for year in sim_years]

    prop_squeeze_uq = [
        (np.percentile(hsi_squeeze.loc[year].to_numpy(), 92.5) /
         np.percentile(hsi_count.loc[year].to_numpy(), 92.5)) * 100 for year in sim_years]

    prop_data = [prop_squeeze_year, prop_squeeze_lq, prop_squeeze_uq]

    simple_line_chart_with_ci(sim_years, mean_data, 'Mean Squeeze Factor', f'Mean Yearly Squeeze for HSI {hsi_string}',
                              f'mean_sf_{hsi_string}', graph_location)
    simple_line_chart(sim_years, median, 'Median Squeeze Factor', f'Median Yearly Squeeze for HSI {hsi_string}',
                      f'med_sf_{hsi_string}', graph_location)
    simple_line_chart_with_ci(sim_years, prop_data, '% HSIs', f'Proportion of HSI {hsi_string} where squeeze > 0',
                              f'prop_sf_{hsi_string}', graph_location)


def comparison_graph_multiple_scenarios(colours, intervention_years, data_dict, y_label, title, graph_location, save_name):
    fig, ax = plt.subplots()

    for k, colour in zip(data_dict, colours):
        ax.plot(intervention_years, data_dict[k][0], label=k, color=colour)
        ax.fill_between(intervention_years, data_dict[k][1], data_dict[k][2], color=colour, alpha=.1)

    plt.ylabel(y_label)
    plt.xlabel('Year')
    plt.title(title)
    plt.gca().set_ylim(bottom=0)
    plt.legend()
    plt.savefig(f'./{graph_location}/{save_name}.png')
    plt.show()


def comparison_graph_multiple_scenarios_multi_level_dict(colours, intervention_years, data_dict, key, y_label, title,
                                                         graph_location, save_name):
    fig, ax = plt.subplots()

    for k, colour in zip(data_dict, colours):
        ax.plot(intervention_years, data_dict[k][key][0], label=k, color=colour)
        ax.fill_between(intervention_years, data_dict[k][key][1], data_dict[k][key][2], color=colour, alpha=.1)

    plt.ylabel(y_label)
    plt.xlabel('Year')
    plt.title(title)
    plt.gca().set_ylim(bottom=0)
    plt.legend()
    plt.savefig(f'./{graph_location}/{save_name}.png')
    plt.show()


def comparison_bar_chart_multiple_bars(data, dict_name, intervention_years, colours, y_title, title,
                                       plot_destination_folder, save_name):

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
def return_birth_data_from_multiple_scenarios(results_folders, sim_years, intervention_years):
    """
    Extract mean, lower and upper quantile births per year for a given scenario
    :param folder: results folder for scenario
    :return: list of total births per year of pre defined intervention period (i.e. 2020-2030)
    """
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
        total_births_per_year = get_mean_and_quants(births_results, sim_years)

        int_df = births_results.loc[intervention_years[0]: intervention_years[-1]]
        int_births_per_year = get_mean_and_quants(int_df, intervention_years)

        return {'total_births': total_births_per_year,
                'int_births': int_births_per_year}

    return {k: extract_births(results_folders[k]) for k in results_folders}


def return_pregnancy_data_from_multiple_scenarios(results_folders, sim_years, intervention_years):
    """
    """

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
        total_pregnancies_per_year = get_mean_and_quants(preg_results, sim_years)
        total_pregnancies_per_year_int = get_mean_and_quants(
            preg_results.loc[intervention_years[0]: intervention_years[-1]], intervention_years)


        return {'total_preg': total_pregnancies_per_year,
                'int_preg': total_pregnancies_per_year_int}

    return {k: extract_pregnancies(results_folders[k]) for k in results_folders}


def return_death_data_from_multiple_scenarios(results_folders, births_dict, sim_years, intervention_years):
    """
    Extract mean, lower and upper quantile maternal mortality ratio, neonatal mortality ratio, crude maternal
    deaths and crude neonatal deaths per year for a given scenario
    :param folder: results folder for scenario
    :param births: list. mean number of births per year for a scenario (used as a denominator)
    :return: dict containing mean, LQ, UQ for MMR, NMR, maternal deaths and neonatal deaths
    """

    def extract_deaths(folder, births):

        # Get full death dataframe
        direct_deaths = extract_results(
            folder,
            module="tlo.methods.demography",
            key="death",
            custom_generate_series=(
                lambda df: df.loc[(df['label'] == 'Maternal Disorders')].assign(
                    year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=True)

        indirect_deaths = extract_results(
            folder,
            module="tlo.methods.demography.detail",
            key="properties_of_deceased_persons",
            custom_generate_series=(
                lambda df: df.loc[(df['is_pregnant'] | df['la_is_postpartum']) &
                                  df['cause_of_death'].str.contains(
                                      'AIDS_non_TB|AIDS_TB|TB|Malaria|Suicide|ever_stroke|diabetes|'
                                      'chronic_ischemic_hd|ever_heart_attack|chronic_kidney_disease')].assign(
                    year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=True
        )

        # TOTAL MATERNAL DEATHS
        total_deaths = direct_deaths + indirect_deaths
        td_int = total_deaths.loc[intervention_years[0]: intervention_years[-1]]
        mean_total_deaths_by_year = get_mean_and_quants(total_deaths, sim_years)
        mean_total_deaths_by_year_int = get_mean_and_quants(td_int, intervention_years)

        agg_births = [sum(births['int_births'][0]), sum(births['int_births'][1]), sum(births['int_births'][2])]

        total_mmr_by_year = [[(x / y) * 100000 for x, y in zip(mean_total_deaths_by_year[0],
                                                               births['total_births'][0])],
                             [(x / y) * 100000 for x, y in zip(mean_total_deaths_by_year[1],
                                                               births['total_births'][1])],
                             [(x / y) * 100000 for x, y in zip(mean_total_deaths_by_year[2],
                                                               births['total_births'][2])]]

        total_deaths_by_scenario = [sum(mean_total_deaths_by_year_int[0]), sum(mean_total_deaths_by_year_int[1]),
                                    sum(mean_total_deaths_by_year_int[2])]

        total_mmr_aggregated = [((total_deaths_by_scenario[0] / agg_births[0]) * 100_000),
                                ((total_deaths_by_scenario[1] / agg_births[1]) * 100_000),
                                ((total_deaths_by_scenario[2] / agg_births[2]) * 100_000)]

        # DIRECT MATERNAL DEATHS
        mean_direct_deaths_by_year = get_mean_and_quants(direct_deaths, sim_years)
        mean_direct_deaths_by_year_int = get_mean_and_quants(direct_deaths, intervention_years)

        total_direct_deaths_by_scenario = [sum(mean_direct_deaths_by_year_int[0]),
                                           sum(mean_direct_deaths_by_year_int[1]),
                                           sum(mean_direct_deaths_by_year_int[2])]

        total_direct_mmr_by_year = [[(x / y) * 100000 for x, y in zip(mean_direct_deaths_by_year[0],
                                                                      births['total_births'][0])],
                                    [(x / y) * 100000 for x, y in zip(mean_direct_deaths_by_year[1],
                                                                      births['total_births'][1])],
                                    [(x / y) * 100000 for x, y in zip(mean_direct_deaths_by_year[2],
                                                                      births['total_births'][2])]]

        total_direct_mmr_aggregated = [((total_direct_deaths_by_scenario[0] / agg_births[0]) * 100_000),
                                       ((total_direct_deaths_by_scenario[1] / agg_births[1]) * 100_000),
                                       ((total_direct_deaths_by_scenario[2] / agg_births[2]) * 100_000)]

        # INDIRECT MATERNAL DEATHS
        mean_indirect_deaths_by_year = get_mean_and_quants(indirect_deaths, sim_years)
        mean_indirect_deaths_by_year_int = get_mean_and_quants(indirect_deaths, intervention_years)

        total_indirect_deaths_by_scenario = [sum(mean_indirect_deaths_by_year_int[0]),
                                             sum(mean_indirect_deaths_by_year_int[1]),
                                             sum(mean_indirect_deaths_by_year_int[2])]

        total_indirect_mmr_by_year = [[(x / y) * 100000 for x, y in zip(mean_indirect_deaths_by_year[0],
                                                                        births['total_births'][0])],
                                      [(x / y) * 100000 for x, y in zip(mean_indirect_deaths_by_year[1],
                                                                        births['total_births'][1])],
                                      [(x / y) * 100000 for x, y in zip(mean_indirect_deaths_by_year[2],
                                                                        births['total_births'][2])]]

        total_indirect_mmr_aggregated = [((total_indirect_deaths_by_scenario[0] / agg_births[0]) * 100_000),
                                         ((total_indirect_deaths_by_scenario[1] / agg_births[1]) * 100_000),
                                         ((total_indirect_deaths_by_scenario[2] / agg_births[2]) * 100_000)]

        # NEONATAL DEATHS
        nd = extract_results(
            folder,
            module="tlo.methods.demography.detail",
            key="properties_of_deceased_persons",
            custom_generate_series=(
                lambda df: df.loc[(df['age_days'] < 29)].assign(
                    year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=True)
        neo_deaths = nd.fillna(0)
        neo_deaths_int = neo_deaths.loc[intervention_years[0]: intervention_years[-1]]

        mean_neonatal_deaths_by_year = get_mean_and_quants(neo_deaths, sim_years)
        mean_neonatal_deaths_by_year_int = get_mean_and_quants(neo_deaths_int, intervention_years)

        total_neonatal_deaths_by_scenario = [sum(mean_neonatal_deaths_by_year_int[0]),
                                             sum(mean_neonatal_deaths_by_year_int[1]),
                                             sum(mean_neonatal_deaths_by_year_int[2])]

        total_nmr_by_year = [[(x / y) * 1000 for x, y in zip(mean_neonatal_deaths_by_year[0],
                                                             births['total_births'][0])],
                             [(x / y) * 1000 for x, y in zip(mean_neonatal_deaths_by_year[1],
                                                             births['total_births'][1])],
                             [(x / y) * 1000 for x, y in zip(mean_neonatal_deaths_by_year[2],
                                                             births['total_births'][2])]]

        total_nmr_aggregated = [((total_neonatal_deaths_by_scenario[0] / agg_births[0]) * 1000),
                                ((total_neonatal_deaths_by_scenario[1] / agg_births[1]) * 1000),
                                ((total_neonatal_deaths_by_scenario[2] / agg_births[2]) * 1000)]

        return {'crude_t_deaths': mean_total_deaths_by_year,
                'agg_total': total_deaths_by_scenario,
                'total_mmr': total_mmr_by_year,
                'agg_total_mr': total_mmr_aggregated,

                'crude_dir_m_deaths': mean_direct_deaths_by_year,
                'agg_dir_m_deaths': total_direct_deaths_by_scenario,
                'direct_mmr': total_direct_mmr_by_year,
                'agg_dir_mr': total_direct_mmr_aggregated,

                'crude_ind_m_deaths': mean_indirect_deaths_by_year,
                'agg_ind_m_deaths': total_indirect_deaths_by_scenario,
                'indirect_mmr': total_indirect_mmr_by_year,
                'agg_ind_mr': total_indirect_mmr_aggregated,

                'crude_n_deaths': mean_neonatal_deaths_by_year,
                'agg_n_deaths': total_neonatal_deaths_by_scenario,
                'nmr': total_nmr_by_year,
                'agg_nmr': total_nmr_aggregated}

    # Extract data from scenarios
    return {k: extract_deaths(results_folders[k], births_dict[k]) for k in results_folders}


def get_differences_between_two_outcomes(baseline_data, comparator):
    crude_diff = [x - y for x, y in zip(baseline_data[0], comparator[0])]
    avg_crude_diff = sum(crude_diff) / len(crude_diff)
    percentage_diff = [100 - ((x / y) * 100) for x, y in zip(comparator[0], baseline_data[0])]
    avg_percentage_diff = sum(percentage_diff) / len(percentage_diff)

    return {'crude': crude_diff,
            'crude_avg': avg_crude_diff,
            'percentage': percentage_diff,
            'percentage_avf': avg_percentage_diff}


def return_stillbirth_data_from_multiple_scenarios(results_folders, births_dict, sim_years, intervention_years):
    """
    Extract antenatal and intrapartum stillbirths from a scenario and return crude numbers and stillbirth rate per
    year
    :param folder: results folder for scenario
    :param births: list. mean number of births per year for a scenario (used as a denominator)
    """

    # TODO: should we report total SBR even though only ISBR will really have been effected?

    def extract_stillbirths(folder, births):
        an_stillbirth_results = extract_results(
            folder,
            module="tlo.methods.pregnancy_supervisor",
            key="antenatal_stillbirth",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=True
        )
        an_stillbirth_results = an_stillbirth_results.fillna(0)
        ip_stillbirth_results = extract_results(
            folder,
            module="tlo.methods.labour",
            key="intrapartum_stillbirth",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=True
        )
        ip_stillbirth_results = ip_stillbirth_results.fillna(0)

        all_sb = an_stillbirth_results + ip_stillbirth_results
        all_sb_int = all_sb.loc[intervention_years[0]: intervention_years[-1]]

        # Get stillbirths
        an_still_birth_data = get_mean_and_quants(an_stillbirth_results, sim_years)
        an_still_birth_data_int = get_mean_and_quants(
            an_stillbirth_results.loc[intervention_years[0]:intervention_years[-1]], intervention_years)

        ip_still_birth_data = get_mean_and_quants(ip_stillbirth_results, sim_years)
        ip_still_birth_data_int = get_mean_and_quants(
            ip_stillbirth_results.loc[intervention_years[0]:intervention_years[-1]], intervention_years)

        all_still_birth_data = get_mean_and_quants(all_sb, sim_years)
        all_still_birth_data_int = get_mean_and_quants(all_sb_int, intervention_years)

        # Store mean number of stillbirths, LQ, UQ
        crude_sb = get_mean_and_quants(all_sb, sim_years)
        crud_int_sb= get_mean_and_quants(all_sb_int, intervention_years)

        # Then generate SBR
        def get_sbr(data, births):
            sbr = [[(x / y) * 1000 for x, y in zip(data[0], births[0])],
                   [(x / y) * 1000 for x, y in zip(data[1], births[1])],
                   [(x / y) * 1000 for x, y in zip(data[2], births[2])]]

            return sbr

        an_sbr = get_sbr(an_still_birth_data, births['total_births'])
        an_sbr_int = get_sbr(an_still_birth_data_int, births['int_births'])
        ip_sbr= get_sbr(ip_still_birth_data, births['total_births'])
        ip_sbr_int = get_sbr(ip_still_birth_data_int, births['int_births'])
        total_sbr= get_sbr(all_still_birth_data, births['total_births'])
        total_sbr_into = get_sbr(all_still_birth_data_int, births['int_births'])

        # Return as dict for graphs
        return {'an_sbr': an_sbr,
                'ip_sbr': ip_sbr,
                'sbr': total_sbr,
                'crude_sb': crude_sb,
                'avg_sbr': [(sum(total_sbr_into[0])/len(intervention_years)),
                            (sum(total_sbr_into[1]) / len(intervention_years)),
                            (sum(total_sbr_into[2]) / len(intervention_years)),
                            ],
                'avg_i_sbr': [(sum(ip_sbr_int[0])/len(intervention_years)),
                            (sum(ip_sbr_int[1]) / len(intervention_years)),
                            (sum(ip_sbr_int[2]) / len(intervention_years)),
                            ],
                'avg_a_sbr': [(sum(an_sbr_int[0])/len(intervention_years)),
                            (sum(an_sbr_int[1]) / len(intervention_years)),
                            (sum(an_sbr_int[2]) / len(intervention_years)),
                            ],
                }

    return {k: extract_stillbirths(results_folders[k], births_dict[k]) for k in results_folders}


def return_dalys_from_multiple_scenarios(results_folders, sim_years, intervention_years):

    def get_dalys_from_scenario(results_folder):
        """
        Extracted stacked DALYs from logger for maternal and neonatal disorders
        :param results_folder: results folder for scenario
        :return: Maternal and neonatal dalys [Mean, LQ, UQ]
        """

        # Get DALY df
        dalys_stacked = extract_results(
            results_folder,
            module="tlo.methods.healthburden",
            key="dalys_stacked",
            custom_generate_series=(
                lambda df: df.drop(
                    columns='date').groupby(['year']).sum().stack()),
            do_scaling=True)

        # todo - should this just be at risk or total (gbd suggests total, which this calibrates well with)
        person_years_total = extract_results(
            results_folder,
            module="tlo.methods.demography",
            key="person_years",
            custom_generate_series=(
                lambda df: df.assign(total=(df['M'].apply(lambda x: sum(x.values()))) + df['F'].apply(
                    lambda x: sum(x.values()))).assign(
                    year=df['date'].dt.year).groupby(['year'])['total'].sum()),
            do_scaling=True)

        denom = get_mean_and_quants(person_years_total, sim_years)
        dalys_mat = get_comp_mean_and_rate('Maternal Disorders', denom[0], dalys_stacked, 100000, sim_years)
        dalys_neo = get_comp_mean_and_rate('Neonatal Disorders', denom[0], dalys_stacked, 100000, sim_years)

        denom_int = get_mean_and_quants(person_years_total.loc[intervention_years[0]: intervention_years[-1]],
                                        intervention_years)
        dalys_stacked_int = dalys_stacked.loc[intervention_years[0]: intervention_years[-1]]
        dalys_mat_i = get_comp_mean_and_rate('Maternal Disorders', denom_int[0], dalys_stacked_int, 100000,
                                             intervention_years)
        dalys_neo_i = get_comp_mean_and_rate('Neonatal Disorders', denom_int[0], dalys_stacked_int, 100000,
                                             intervention_years)

        dalys_mat_agg = [(sum(dalys_mat_i[0])/len(intervention_years)),
                         (sum(dalys_mat_i[1])/len(intervention_years)),
                         (sum(dalys_mat_i[2])/len(intervention_years))]

        dalys_neo_agg = [(sum(dalys_neo_i[0])/len(intervention_years)),
                         (sum(dalys_neo_i[1])/len(intervention_years)),
                         (sum(dalys_neo_i[2])/len(intervention_years))]

        mat_causes_death = ['ectopic_pregnancy', 'spontaneous_abortion', 'induced_abortion',
                            'severe_gestational_hypertension', 'severe_pre_eclampsia', 'eclampsia', 'antenatal_sepsis',
                            'uterine_rupture', 'intrapartum_sepsis', 'postpartum_sepsis', 'postpartum_haemorrhage',
                            'secondary_postpartum_haemorrhage', 'antepartum_haemorrhage']

        mat_causes_disab = ['maternal']

        neo_causes_death = ['early_onset_sepsis', 'late_onset_sepsis', 'encephalopathy', 'preterm_other',
                            'respiratory_distress_syndrome', 'neonatal_respiratory_depression']

        neo_causes_disab = ['Retinopathy of Prematurity', 'Neonatal Encephalopathy',
                            'Neonatal Sepsis Long term Disability', 'Preterm Birth Disability']

        yll_stacked = extract_results(
            results_folder,
            module="tlo.methods.healthburden",
            key="yll_by_causes_of_death_stacked",
            custom_generate_series=(
                lambda df: df.drop(
                    columns='date').groupby(['year']).sum().stack()),
            do_scaling=True)

        yld = extract_results(
            results_folder,
            module="tlo.methods.healthburden",
            key="yld_by_causes_of_disability",
            custom_generate_series=(
                lambda df: df.drop(
                    columns='date').groupby(['year']).sum().stack()),
            do_scaling=True)

        def get_output(causes, df):
            mean = list()
            lq = list()
            uq = list()

            for year in sim_years:
                per_year = 0
                per_year_lq = 0
                per_year_uq = 0
                for cause in causes:
                    if cause in df.loc[year].index:
                        per_year += df.loc[year, cause].mean()
                        per_year_lq += df.loc[year, cause].quantile(0.025)
                        per_year_uq += df.loc[year, cause].quantile(0.925)

                mean.append(per_year)
                lq.append(per_year_lq)
                uq.append(per_year_uq)

            return [mean, lq, uq]

        def get_as_rate(values):
            mean = [(x / y) * 100000 for x, y in zip(values[0], denom[0])]
            lq = [(x / y) * 100000 for x, y in zip(values[1], denom[1])]
            uq = [(x / y) * 100000 for x, y in zip(values[2], denom[2])]

            return [mean, lq, uq]

        mat_yll = get_output(mat_causes_death, yll_stacked)
        mat_yll_rate = get_as_rate(mat_yll)
        mat_yld = get_output(mat_causes_disab, yld)
        mat_yld_rate = get_as_rate(mat_yld)
        neo_yll = get_output(neo_causes_death, yll_stacked)
        neo_yll_rate = get_as_rate(neo_yll)
        neo_yld = get_output(neo_causes_disab, yld)
        neo_yld_rate = get_as_rate(neo_yld)

        def extract_dalys_tlo_model(group, years):
            """Extract mean, LQ, UQ DALYs for maternal or neonatal disorders"""

            stacked_dalys = [dalys_stacked.loc[year, f'{group} Disorders'].mean() for year in
                             years if year in years]

            stacked_dalys_lq = [dalys_stacked.loc[year, f'{group} Disorders'].quantile(0.025) for year in
                                years if year in years]

            stacked_dalys_uq = [dalys_stacked.loc[year, f'{group} Disorders'].quantile(0.925) for year in
                                years if year in years]

            return [stacked_dalys, stacked_dalys_lq, stacked_dalys_uq]

        crude_m_dalys = extract_dalys_tlo_model('Maternal', sim_years)
        crude_n_dalys = extract_dalys_tlo_model('Neonatal', sim_years)

        c_m_dalys_int = extract_dalys_tlo_model('Maternal', intervention_years)
        c_n_dalys_int = extract_dalys_tlo_model('Neonatal', intervention_years)


        return {'maternal_dalys_crude': crude_m_dalys,
                'maternal_dalys_rate': dalys_mat,
                'agg_mat_dalys': [sum(c_m_dalys_int[0]), sum(c_m_dalys_int[1]), sum(c_m_dalys_int[2])],
                'avg_mat_dalys_rate': dalys_mat_agg,
                'maternal_yll_crude': mat_yll,
                'maternal_yll_rate': mat_yll_rate,
                'maternal_yld_crude': mat_yld,
                'maternal_yld_rate': mat_yld_rate,
                'neonatal_dalys_crude': crude_n_dalys,
                'neonatal_dalys_rate': dalys_neo,
                'agg_neo_dalys': [sum(c_n_dalys_int[0]), sum(c_n_dalys_int[1]), sum(c_n_dalys_int[2])],
                'avg_neo_dalys_rate': dalys_neo_agg,
                'neonatal_yll_crude': neo_yll,
                'neonatal_yll_rate': neo_yll_rate,
                'neonatal_yld_crude': neo_yld,
                'neonatal_yld_rate': neo_yld_rate}

    # Store DALYs data for baseline and intervention
    return {k: get_dalys_from_scenario(results_folders[k]) for k in results_folders}
