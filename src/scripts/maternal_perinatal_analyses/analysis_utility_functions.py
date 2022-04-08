
import numpy as np
from tlo.analysis.utils import extract_results
from matplotlib import pyplot as plt
plt.style.use('seaborn')

# ==================================================== UTILITY CODE ===================================================
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
            lower_quantiles.append(0)

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


def line_graph_with_ci_and_target_rate(sim_years, mean_list, lq_list, uq_list, target_data_dict, y_label, title,
                                       graph_location, file_name):
    fig, ax = plt.subplots()
    ax.plot(sim_years, mean_list, 'o-g', label="Model", color='deepskyblue')
    ax.fill_between(sim_years, lq_list, uq_list, color='b', alpha=.1, label="UI (2.5-92.5)")

    if target_data_dict['double']:
        plt.errorbar(target_data_dict['first']['year'], target_data_dict['first']['value'],
                     label=target_data_dict['first']['label'], yerr=target_data_dict['first']['ci'],
                     fmt='o', color='darkseagreen', ecolor='green', elinewidth=3, capsize=0)
        plt.errorbar(target_data_dict['second']['year'], target_data_dict['second']['value'],
                     label=target_data_dict['second']['label'], yerr=target_data_dict['second']['ci'],
                     fmt='o', color='red', ecolor='mistyrose', elinewidth=3, capsize=0)

    elif not target_data_dict['double']:
        plt.errorbar(target_data_dict['first']['year'], target_data_dict['first']['value'],
                     label=target_data_dict['first']['label'], yerr=target_data_dict['first']['ci'],
                     fmt='o', color='red', ecolor='pink', elinewidth=3, capsize=0)

    plt.xlabel('Year')
    plt.ylabel(y_label)
    plt.title(title)
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
