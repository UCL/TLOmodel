import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from datetime import date
import numpy as np
from pathlib import Path
load_path = 'C:/Users/ieh19/Desktop/Project 1/model_outputs/'
save_path = 'C:/Users/ieh19/Desktop/Project 1/result_tables/'


def get_number_of_worms(str):
    numbers = [int(s) for s in str.split() if s.isdigit()]
    return numbers[0]

def load_outputs(timestamp, infection):
    infection_outputs = pd.read_csv(load_path + "output_" + infection + '_' + timestamp + ".csv")
    infection_outputs.date = pd.to_datetime(infection_outputs.date)
    infection_outputs.date = infection_outputs.date - np.timedelta64(20, 'Y')
    # if 'High-inf_Prevalence' not in list(infection_outputs.columns):
    #     # infection_outputs['High-inf_Prevalence'] = infection_outputs.apply(lambda row: row.High_infections / (row.Infected + row.Non_infected))
    #     infection_outputs['High-inf_Prevalence'] = infection_outputs['High_infections']/(infection_outputs['Infected'] + infection_outputs['Non_infected'])

    # dalys = pd.read_csv(load_path + "output_daly_" + timestamp + ".csv")
    # dalys.date = pd.to_datetime(dalys.date)
    # dalys.date = dalys.date - np.timedelta64(20, 'Y')
    # dalys = dalys.groupby(['date'], as_index=False).agg({'YLD_Schisto_Haematobium_Schisto_Haematobium_Symptoms': 'sum'})

    prev_years = pd.read_csv(load_path + "output_prevalent_years_" + timestamp + ".csv")
    prev_years.date = pd.to_datetime(prev_years.date)
    prev_years.date = prev_years.date - np.timedelta64(20, 'Y')

    distr = pd.read_csv(load_path + 'output_districts_prev_Haematobium_' + timestamp + ".csv")

    return infection_outputs, dalys, prev_years, distr

def get_averages_prev(all_sims_outputs):
    list_of_dfs = []
    for df in all_sims_outputs:
        list_of_dfs.append(df)
    big_df = pd.concat(list_of_dfs, ignore_index=True)
    avg_df = big_df.groupby(['date', 'Age_group'], as_index=False).agg({'Prevalence': 'mean', 'MeanWormBurden': 'mean', 'High-inf_Prevalence': 'mean'})
    return avg_df

def get_averages_dalys(all_sims_outputs):
    list_of_dfs = []
    for df in all_sims_outputs:
        list_of_dfs.append(df)
    big_df = pd.concat(list_of_dfs, ignore_index=True)
    big_df = big_df[['date', 'YLD_Schisto_Haematobium_Schisto_Haematobium_Symptoms']]
    avg_df = big_df.groupby(['date'], as_index=False).agg({'YLD_Schisto_Haematobium_Schisto_Haematobium_Symptoms': 'mean'})
    return avg_df

def get_averages_prev_and_high_inf_years(all_sims_outputs):
    cols_of_interest = ['Prevalent_years_per_100', 'High_infection_years_per_100']
    list_of_dfs = []
    for df in all_sims_outputs:
        list_of_dfs.append(df)
    big_df = pd.concat(list_of_dfs, ignore_index=True)
    # big_df = big_df[['date', cols_of_interest]]
    avg_df = big_df.groupby(['date', 'Age_group'], as_index=False).agg({cols_of_interest[0]: 'mean', cols_of_interest[1]: 'mean'})

    return avg_df

def get_averages_districts(all_sims_outputs):
    list_of_dfs = []
    for df in all_sims_outputs:
        list_of_dfs.append(df)
    big_df = pd.concat(list_of_dfs, ignore_index=True)
    print(big_df)
    avg_df = big_df.groupby(['District'], as_index=False).agg({'Prevalence': 'mean',
                                                               'MWB': 'mean'})
    return avg_df

def plot_per_age_group(scenarios_list, age, vals):
    assert vals in ['Prevalence', 'High-inf_Prevalence', 'MeanWormBurden']
    fig, ax = plt.subplots(figsize=(9, 7))
    colours = ['b', 'm', 'y', 'g', 'k', 'c', 'r', 'lawngreen', 'purple', 'orange', 'maroon', 'navy']
    colour_counter = 0
    for k in scenarios_list.keys():
        c = colours[colour_counter]
        # c = 'b'
        for ii in range(len(scenarios_list[k]['prev'])):
            df = scenarios_list[k]['prev'][ii]
            df = df[df['Age_group'] == age]
            df = df[df.date >= pd.Timestamp(date(2019, 1, 1))]
            # df.date = df.date - np.timedelta64(2000, 'Y')
            # df = df[(df.date < pd.Timestamp(date(2019, 7, 1))) & (df.date >= pd.Timestamp(date(2014, 1, 1)))]
            # df = df[df.date >= pd.Timestamp(date(2014, 1, 1))]
            if ii < len(scenarios_list[k]['prev']) - 1:
                ax.plot(df.date, df[vals], color=c, label='_nolegend_', linestyle=':', alpha=0.5)
            # last one is an average of the previous scenarios
            else:
                ax.plot(df.date, df[vals], color=c, label=k, linestyle='-')
        # plt.axhline(y=0.05, color='r', linestyle='-')
        plt.axhline(y=0.01, color='r', linestyle='--')
        ax.xaxis_date()
        colour_counter += 1
    plt.xlabel('logging date', fontsize=16)
    plt.ylabel(vals, fontsize=16)
    plt.title(vals + ' per date, ' + age)
    if vals == 'Prevalence':
        # plt.ylim([0, 0.2])
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    if vals == 'High-inf_Prevalence':
        # plt.yscale('log')
        # plt.ylim([0, 0.12])
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    # else:
    #     plt.ylim([0, 2.5])
    ax.grid(linewidth=0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    # locs, labels = plt.xticks()
    # plt.xticks = [int(x) - 2019 for x in labels]
    plt.xticks(rotation='vertical')
    plt.legend(prop={'size': 16})
    plt.show()

def plot_all_age_groups_same_plot(scenarios_list, infection, vals):
    assert vals in ['Prevalence', 'High-inf_Prevalence', 'MeanWormBurden']
    fig, ax = plt.subplots(figsize=(9, 7))
    colours = ['b', 'm', 'y', 'g', 'k', 'c', 'r', 'lawngreen', 'purple', 'orange', 'maroon', 'navy']
    k = list(scenarios.keys())[0]

    for ii in range(len(scenarios_list[k]['prev'])):
        colour_counter = 0
        df = scenarios_list[k]['prev'][ii]
        for age in ['PSAC', 'SAC', 'Adults', 'All']:
            c = colours[colour_counter]
            age_df = df[df['Age_group'] == age]
        # df = df[df.date >= pd.Timestamp(date(2019, 1, 1))]
        # df.date = df.date - np.timedelta64(2000, 'Y')
        # df = df[(df.date < pd.Timestamp(date(2019, 7, 1))) & (df.date >= pd.Timestamp(date(2014, 1, 1)))]
        # df = df[df.date >= pd.Timestamp(date(2014, 1, 1))]
            if ii < len(scenarios_list[k]['prev']) - 1:
                ax.plot(age_df.date, age_df[vals], color=c, label='_nolegend_', linestyle=':', alpha=0.5)
            # last one is an average of the previous scenarios
            else:
                ax.plot(age_df.date, age_df[vals], color=c, label=age, linestyle='-')
            colour_counter += 1
    # plt.axhline(y=0.05, color='r', linestyle='-')
    # plt.axhline(y=0.01, color='k', linestyle='-')
        ax.xaxis_date()

    ax.set(xlabel='logging date',
           ylabel=vals,
           title=vals + ' per date, ' + ', S.' + infection)
    ax.grid(linewidth=0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xticks(rotation='vertical')
    plt.legend()
    plt.show()

def plot_measure(scenarios_list, age, measure_type):
    assert measure_type in ['dalys', 'prev_years', 'high_inf']
    if measure_type == 'dalys':
        value_col = 'YLD_Schisto_Haematobium_Schisto_Haematobium_Symptoms'
        title = 'DALYs per year per 10.000 ppl'
    elif measure_type == 'prev_years':
        value_col = 'Prevalent_years_per_100'
        title = 'Prevalent years per year per 100 ppl, ' + age
    else:
        measure_type = 'prev_years'
        value_col = 'High_infection_years_per_100'
        title = 'High-infection years per year per 100 ppl, ' + age
    fig, ax = plt.subplots(figsize=(9, 7))
    colours = ['b', 'm', 'y', 'g', 'k', 'c', 'r', 'lawngreen', 'purple', 'orange', 'maroon', 'navy']
    colour_counter = 0
    for k in scenarios_list.keys():
        worms = get_number_of_worms(k)
        worms_colour_dict = {5: 'r', 10: 'b', 15: 'g', 20: 'm'}
        c = worms_colour_dict[worms]
        if k[5] == '0':  # PSAC 0%
            ls = '--'
            marker = '+'
        elif k[5] == '2':  # PSAC 25%
            marker = 'o'
            ls = ':'
        else:  # PSAC 50%
            marker = 'v'
            ls = '-'
        marker = None
        # c = colours[colour_counter]
        for ii in range(len(scenarios_list[k]['prev'])):
            df = scenarios_list[k][measure_type][ii]
            df = df[df['Age_group'] == age]
            # df_before_2019 = df[(df.date <= pd.Timestamp(date(2019, 1, 1))) & (df.date >= pd.Timestamp(date(2014, 1, 1)))]
            df = df[df.date >= pd.Timestamp(date(2018, 12, 30))]
            # ax.plot(df_after_2019.date, df_after_2019[value_col], label=k, linestyle=':')
            # ax.plot(df_before_2019.date, df_before_2019[value_col], color='b', label='_nolegend_', linestyle='-')

            # df = df[df.date >= pd.Timestamp(date(2014, 1, 1))]
            if ii < len(scenarios_list[k]['prev']) - 1:
                ()
                # ax.plot(df.date, df[value_col], color=c, label='_nolegend_', linestyle=':', alpha=0.5)
            # last one is an average of the previous scenarios
            else:
                ax.plot(df.date, df[value_col], color=c, marker=marker, linestyle=ls, linewidth=2, label=k)
                # plt.plot(df.date, df[value_col].values, c=c, linestyle='o-')
        # colour_counter += 1
        ax.xaxis_date()
    # plt.yscale('log')
    ax.grid(linewidth=0.5)
    plt.xlabel('logging date', fontsize=16)
    plt.ylabel('years', fontsize=16)
    plt.title(title, fontsize=16)
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    plt.xticks(rotation='vertical', fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(prop={'size': 12})
    plt.show()

def get_expected_prevalence(infection):
    expected_district_prevalence = pd.read_excel(Path("./resources") / 'ResourceFile_Schisto.xlsx',
                                                 sheet_name='District_Params_' + infection.lower())
    expected_district_prevalence.set_index("District", inplace=True)
    expected_district_prevalence = expected_district_prevalence.loc[:, 'Prevalence'].to_dict()
    return expected_district_prevalence

def plot_prevalence_per_district(scenarios_list, infection):
    expected_prev = get_expected_prevalence(infection)
    df = scenarios_list['avg']['distr'].copy()
    df.set_index("District", inplace=True)
    dict_dstr = df.loc[:, 'Prevalence'].to_dict()
    plt.bar(*zip(*dict_dstr.items()), alpha=0.5, label='simulations avg')
    plt.scatter(*zip(*expected_prev.items()), label='data')
    plt.xticks(rotation=90)
    plt.xlabel('District')
    plt.ylabel('Prevalence')
    plt.legend()
    plt.title('Prevalence per district, S.' + infection)
    plt.show()

def find_year_of_elimination(scenarios_list, age_group, level):
    assert level in [0.01, 0.0]
    print('Elimination under', str(level*100), '% will be reached:')
    for k in scenarios_list.keys():
        avg_df = scenarios_list[k]['prev'][-1]
        avg_df = avg_df[avg_df['Age_group'] == age_group]
        date_of_elimination_idx = avg_df[avg_df['High-inf_Prevalence'] <= level].first_valid_index()
        if date_of_elimination_idx is None:
            date_of_elimination = 'Not reached'
            treatment_rounds = 'NR'
        else:
            date_of_elimination = avg_df.loc[date_of_elimination_idx, 'date']
            treatment_rounds = date_of_elimination.year - 2019 + 1
            if date_of_elimination.month < 7:
                treatment_rounds -= 1
            # years_till_elimination = (date_of_elimination - pd.Timestamp(year=2019, month=7, day=1) / np.timedelta64(1, 'Y'))
        print(k, date_of_elimination, treatment_rounds)

def get_average_final(scenarios_list, vals):
    assert vals in ['prev', 'prev_years']
    output_dict = {}
    for k in scenarios_list.keys():
        df = scenarios_list[k][vals][-1]  # take the last dataframe which contains averages
        df = df[df.date <= pd.Timestamp(2024, 6, 1, 3, 36)]  # this is just before the 6th rounds of MDA
        if vals == 'prev':
            df_before = df[df.date == pd.Timestamp(2019, 6, 1, 3, 36)]
        else:
            df_before = df[df.date == pd.Timestamp(2018, 12, 31, 3, 36)]
        df_after = df.iloc[-4:]  # last 4 rows will contain the final averages for each age group
        df = pd.concat([df_before, df_after])
        df['Scenario'] = k
        df.set_index('Age_group', inplace=True)
        output_dict.update({k: df})
    return output_dict

def get_all_sim_distr_averages(scenarios_list):
    list_of_dfs = []
    for k in scenarios_list.keys():
        df = scenarios_list[k]['distr'][-1]  # take the last dataframe which contains averages
        list_of_dfs.append(df)
    big_df = pd.concat(list_of_dfs, ignore_index=True)
    avg_df = big_df.groupby(['District'], as_index=False).agg({'Prevalence': ['mean', 'std'],
                                                               'MWB': ['mean', 'std'],
                                                               'High-infection-prevalence': ['mean', 'std']})
    return avg_df

def save_averages_final_in_csv(scenarios_list, measure):
    assert measure in ['prev', 'prev_years']
    avg_dict = get_average_final(scenarios_list, measure)
    df_list = [df for df in avg_dict.values()]
    avg_df = pd.concat(df_list)
    avg_df.date = avg_df.date.map({pd.Timestamp(2019, 6, 1, 3, 36): 'before', pd.Timestamp(2018, 12, 31, 3, 36): 'before'})
    avg_df.date.fillna('after', inplace=True)
    avg_df_before = avg_df[avg_df['date'] == 'before']
    # first we save the 'before' values - average them because they don't depend on the scenario
    if measure == 'prev':
        avg_df_before = avg_df_before.groupby(level=0).agg(
            {'Prevalence': 'mean', 'MeanWormBurden': 'mean', 'High-inf_Prevalence': 'mean'})
        columns = ['Prevalence', 'MeanWormBurden', 'High-inf_Prevalence']
    else:
        avg_df_before = avg_df_before.groupby(level=0).agg(
            {'Prevalent_years_per_100': 'mean', 'High_infection_years_per_100': 'mean'})
        columns = ['Prevalent_years_per_100', 'High_infection_years_per_100']
    writer = pd.ExcelWriter(save_path + measure + '_before_and_after.xlsx')
    for vals in columns:
        avg_df_val = avg_df[avg_df['date'] == 'after']
        avg_df_val = avg_df_val[[vals, 'Scenario']]
        avg_df_val['Age'] = avg_df_val.index
        pivot = avg_df_val.pivot(index='Age', columns='Scenario', values=vals)
        before_dict = avg_df_before[vals].to_dict()
        pivot['before'] = pivot.index.map(before_dict)
        # pivot.to_csv(save_path + vals + '_average_finals.csv')
        pivot.to_excel(writer, sheet_name=vals)
    writer.save()

def get_average_final_sum_up_years(scenarios_list):
    output_dict = {}
    for k in scenarios_list.keys():
        df = scenarios_list[k]['prev_years'][-1]  # take the last dataframe which contains averages
        df_before = df[df.date == pd.Timestamp(2018, 12, 31, 3, 36)]
        df_before.loc[:, 'date'] = 'before'
        df_before.set_index('Age_group', inplace=True)
        df_after = df[df.date > pd.Timestamp(2018, 12, 31, 3, 36)]
        df_after = df_after.groupby('Age_group').agg({'Prevalent_years_per_100': 'sum', 'High_infection_years_per_100': 'sum'})
        df_after['date'] = 'after'
        df = pd.concat([df_before, df_after])
        df['Scenario'] = k
        output_dict.update({k: df})
    return output_dict

def save_averages_final_in_csv_years(scenarios_list):
    avg_dict = get_average_final_sum_up_years(scenarios_list)
    df_list = [df for df in avg_dict.values()]
    avg_df = pd.concat(df_list)
    # first we save the 'before' values - average them because they don't depend on the scenario
    avg_df_before = avg_df[avg_df['date'] == 'before']
    avg_df_before = avg_df_before.groupby(level=0).agg(
        {'Prevalent_years_per_100': 'mean', 'High_infection_years_per_100': 'mean'})
    columns = ['Prevalent_years_per_100', 'High_infection_years_per_100']
    writer = pd.ExcelWriter(save_path + 'years_cum__before_and_after.xlsx')
    for vals in columns:
        avg_df_val = avg_df[avg_df['date'] == 'after']
        avg_df_val = avg_df_val[[vals, 'Scenario']]
        avg_df_val['Age'] = avg_df_val.index
        pivot = avg_df_val.pivot(index='Age', columns='Scenario', values=vals)
        before_dict = avg_df_before[vals].to_dict()
        pivot['before'] = pivot.index.map(before_dict)
        # pivot.to_csv(save_path + vals + '_average_finals.csv')
        pivot.to_excel(writer, sheet_name=vals)
    writer.save()

save_averages_final_in_csv_years(scenarios)
save_averages_final_in_csv(scenarios, 'prev')


#########################################################################################################

# Load the simulations you want to compare
#
# timestamps1 = ['2020-01-24_17-05-00', '2020-01-24_17-05-28', '2020-01-24_17-05-41']
# timestamps2 = ['2020-01-24_19-36-24', '2020-01-24_19-37-08', '2020-01-24_19-37-18']
# timestamps22 = ['2020-01-28_18-15-26', '2020-01-28_18-16-42', '2020-01-28_18-16-57']
# timestamps3 = ['2020-01-24_15-37-04', '2020-01-24_15-37-24', '2020-01-24_15-37-40']
# timestamps4 = ['2020-01-26_14-21-57', '2020-01-26_14-22-20', '2020-01-26_14-22-43']
# timestamps5 = ['2020-01-24_14-38-32', '2020-01-24_14-38-55', '2020-01-24_14-38-45']
# timestamps6 = ['2020-01-28_18-15-26', '2020-01-28_18-16-42', '2020-01-28_18-16-57']
# timestamps = [timestamps1, timestamps2, timestamps22, timestamps3, timestamps4, timestamps5, timestamps6]
# labels = ['No MDA after 2019',
#           'MDA once per year, PSAC = 0%',
#           'MDA once per year, PSAC = 25%',
#           'MDA once per year, PSAC = 50%',
#           'MDA twice per year, PSAC = 0%',
#           'MDA twice per year, PSAC = 50%',
#           'MDA once per year, PSAC = 25%']

# these are with varying the worm threshold for high-intensity infections among PSAC
scenarios = {
    'PSAC 0%, 5 worms': ['2020-02-03_11-42-43', '2020-02-03_11-45-12', '2020-02-03_11-48-26'],
    'PSAC 0% 10 worms': ['2020-02-02_22-45-31', '2020-02-02_22-45-10', '2020-02-02_22-46-06'],
    'PSAC 0% 15 worms': ['2020-02-03_13-44-24', '2020-02-03_13-43-11', '2020-02-03_13-47-27'],
    'PSAC 0% 20 worms': ['2020-02-03_09-28-23', '2020-02-03_09-29-35', '2020-02-03_09-34-40'],
    'PSAC 25% 5 worms': ['2020-02-03_11-51-32', '2020-02-03_11-51-16', '2020-02-03_11-50-14'],
    'PSAC 25% 10 worms': ['2020-02-02_22-40-56', '2020-02-02_22-43-14', '2020-02-02_22-45-18'],
    'PSAC 25% 15 worms': ['2020-02-03_13-46-28', '2020-02-03_13-47-34', '2020-02-03_13-47-50'],
    'PSAC 25% 20 worms': ['2020-02-03_09-40-34', '2020-02-03_09-38-36', '2020-02-03_09-41-49'],
    'PSAC 50% 5 worms': ['2020-02-03_11-51-57', '2020-02-03_11-52-36', '2020-02-03_11-51-07'],
    'PSAC 50% 10 worms': ['2020-02-02_22-38-14', '2020-02-02_22-39-27', '2020-02-02_22-43-19'],
    'PSAC 50% 15 worms': ['2020-02-03_14-40-06', '2020-02-03_14-40-33', '2020-02-03_14-42-50'],
    'PSAC 50% 20 worms': ['2020-02-03_09-41-29', '2020-02-03_09-42-35', '2020-02-03_09-41-31']
}

# longer simulation with no MDA
scenarios = {
    '25 years, no MDA': ['2020-01-24_12-02-31', '2020-01-24_12-02-55', '2020-01-24_12-03-06']
}


for label, timestamps in scenarios.items():
    # load all results for the given scenario
    prev = []
    dalys = []
    prev_years = []
    distr = []
    for time in timestamps:
        prev_t, dalys_t, prev_years_t, distr_t = load_outputs(time, 'Haematobium')
        prev.append(prev_t)
        # dalys.append(dalys_t)
        prev_years.append(prev_years_t)
        distr.append(distr_t)

    prev.append(get_averages_prev(prev))
    # dalys.append(get_averages_dalys(dalys))
    prev_years.append(get_averages_prev_and_high_inf_years(prev_years))
    distr.append(get_averages_districts(distr))
    scenario_outputs = {'prev': prev, 'dalys': dalys, 'prev_years': prev_years, 'distr': distr}
    scenarios.update({label: scenario_outputs})



# plots
for age_group in ['PSAC', 'SAC']:
    plot_per_age_group(scenarios, age_group, 'haematobium', 'Prevalence')

for age_group in ['PSAC', 'SAC']:
    plot_per_age_group(scenarios, age_group, 'haematobium', 'High-inf_Prevalence')


plot_all_age_groups_same_plot(scenarios, 'haematobium', 'Prevalence')

for age_group in ['PSAC', 'SAC', 'Adults', 'All']:
    plot_per_age_group(scenarios, age_group, 'haematobium', 'MeanWormBurden')

plot_measure(scenarios, 'prev_years')
plot_measure(scenarios, 'PSAC', 'high_inf')

plot_all_age_groups_same_plot(scenarios, 'haematobium', 'Prevalence')

find_year_of_elimination(scenarios, 'PSAC', 0.01)
find_year_of_elimination(scenarios, 'PSAC', 0.00)
save_averages_final_in_csv(scenarios, 'prev')
save_averages_final_in_csv(scenarios, 'prev_years')
districts_averages = get_all_sim_distr_averages(scenarios)

def plot_subset_of_worms(scenarios_list, labels_list, age, vals, title):
    assert vals in ['Prevalence', 'High-inf_Prevalence']
    fig, ax = plt.subplots(figsize=(9, 7))
    colours = ['b', 'm', 'y', 'g', 'k', 'c', 'r', 'lawngreen', 'purple', 'orange', 'maroon', 'navy']
    colour_counter = 0
    for k in labels_list:
        c = colours[colour_counter]
        for ii in range(len(scenarios_list[k]['prev'])):
            df = scenarios_list[k]['prev'][ii]
            df = df[df['Age_group'] == age]
            df = df[df.date >= pd.Timestamp(date(2019, 1, 1))]
            # df.date = df.date - np.timedelta64(2000, 'Y')
            if ii < len(scenarios_list[k]['prev']) - 1:
                ax.plot(df.date, df[vals], color=c, label='_nolegend_', linestyle=':', alpha=0.75, linewidth=2)
            # last one is an average of the previous scenarios
            else:
                label = k[0:(k.find('%')+1)]
                ax.plot(df.date, df[vals], color=c, label=label, linestyle='-',linewidth=4)
        ax.xaxis_date()
        colour_counter += 1
    # ax.set(xlabel='logging date',
    #        ylabel='Prevalence of heavy infections')
    plt.xlabel('logging date', fontsize=16)
    plt.ylabel('Prevalence of heavy infections', fontsize=16)
    if vals == 'Prevalence':
        plt.ylim([0, 0.35])
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    if vals == 'High-inf_Prevalence':
        plt.yscale('log')
        plt.ylim([0, 0.1])
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(linewidth=0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    plt.xticks(rotation='vertical', fontsize=16)
    plt.yticks(fontsize=16)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.025, 0.05, 'High-int threshold: ' + str(get_number_of_worms(title)) + ' worms',
            transform=ax.transAxes, fontsize=16,
            verticalalignment='top', bbox=props)
    plt.legend(prop={'size': 16})
    plt.show()

worms5 = ['PSAC 0%, 5 worms', 'PSAC 25% 5 worms', 'PSAC 50% 5 worms']
worms10 = ['PSAC 0% 10 worms', 'PSAC 25% 10 worms', 'PSAC 50% 10 worms']
worms15 = ['PSAC 0% 15 worms', 'PSAC 25% 15 worms', 'PSAC 50% 15 worms']
worms20 = ['PSAC 0% 20 worms', 'PSAC 25% 20 worms', 'PSAC 50% 20 worms']

plot_subset_of_worms(scenarios, worms5, 'PSAC', 'High-inf_Prevalence', 'PSAC High-intensity threshold = 5 worms')
plot_subset_of_worms(scenarios, worms10, 'PSAC', 'High-inf_Prevalence', 'PSAC High-intensity threshold = 10 worms')
plot_subset_of_worms(scenarios, worms15, 'PSAC', 'High-inf_Prevalence', 'PSAC High-intensity threshold = 15 worms')
plot_subset_of_worms(scenarios, worms20, 'PSAC', 'High-inf_Prevalence', 'PSAC High-intensity threshold = 20 worms')
