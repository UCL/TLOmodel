import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from datetime import date
import numpy as np
from pathlib import Path


def load_outputs(timestamp, infection):
    infection_outputs = pd.read_csv(load_path + "output_" + infection + '_'+ timestamp + ".csv")
    infection_outputs.date = pd.to_datetime(infection_outputs.date)
    infection_outputs.date = infection_outputs.date - np.timedelta64(20, 'Y')
    if 'High-inf_Prevalence' not in list(infection_outputs.columns):
        # infection_outputs['High-inf_Prevalence'] = infection_outputs.apply(lambda row: row.High_infections / (row.Infected + row.Non_infected))
        infection_outputs['High-inf_Prevalence'] = infection_outputs['High_infections']/(infection_outputs['Infected'] + infection_outputs['Non_infected'])

    dalys = pd.read_csv(load_path + "output_daly_" + timestamp + ".csv")
    dalys.date = pd.to_datetime(dalys.date)
    dalys.date = dalys.date - np.timedelta64(20, 'Y')
    dalys = dalys.groupby(['date'], as_index=False).agg({'YLD_Schisto_Haematobium_Schisto_Haematobium_Symptoms': 'sum'})

    prev_years = pd.read_csv(load_path + "output_prevalent_years_" + timestamp + ".csv")
    prev_years.date = pd.to_datetime(prev_years.date)
    prev_years.date = prev_years.date - np.timedelta64(20, 'Y')
    return infection_outputs, dalys, prev_years

def get_averages_prev(all_sims_outputs):
    list_of_dfs = []
    for df in all_sims_outputs:
        list_of_dfs.append(df)
    big_df = pd.concat(list_of_dfs, ignore_index=True)
    avg_df = big_df.groupby(['date', 'Age_group'], as_index=False).agg({'Prevalence': 'mean', 'MeanWormBurden': 'mean', 'High-inf_Prevalence': 'mean'})
    return avg_df

def get_averages_dalys_or_prev_years(all_sims_outputs, vals):
    cols_of_interest = {'dalys': 'YLD_Schisto_Haematobium_Schisto_Haematobium_Symptoms',
                        'prev_years': 'Prevalent_years_this_year_total'}
    list_of_dfs = []
    for df in all_sims_outputs:
        list_of_dfs.append(df)
    big_df = pd.concat(list_of_dfs, ignore_index=True)
    big_df = big_df[['date', cols_of_interest[vals]]]
    avg_df = big_df.groupby(['date'], as_index=False).agg({cols_of_interest[vals]: 'mean'})
    return avg_df

def get_averages_districts(sims):
    list_of_dfs = []
    for k in sims.keys():
        df = sims[k]['distr']
        list_of_dfs.append(df)
    big_df = pd.concat(list_of_dfs, ignore_index=True)
    avg_df = big_df.groupby(['District'], as_index=False).agg({'Prevalence': 'mean', 'MWB': 'mean'})
    return avg_df

def add_average(sim):
    avg_outputs = {'prev': get_averages_prev(sim),
                          # 'dalys': get_averages_dalys(sim, 'dalys'),
                           'prev_years': get_averages_dalys(sim, 'prev_years'),
                           'distr': get_averages_districts(sim)}
    sim.update({'avg': avg_outputs})
    return sim

def plot_per_age_group(scenarios_list, age, infection, vals):
    assert vals in ['Prevalence', 'High-inf_Prevalence', 'MeanWormBurden']
    fig, ax = plt.subplots(figsize=(9,7))
    # fig, ax = plt.subplots()
    colours = ['b', 'm', 'y', 'g', 'k', 'c', 'r', 'darkgreen', 'purple', 'orange']
    colour_counter = 0
    for k in scenarios_list.keys():
        c = colours[colour_counter]
        for ii in range(len(scenarios_list[k]['prev'])):
            df = scenarios_list[k]['prev'][ii]
            df = df[df['Age_group'] == age]
            df = df[df.date >= pd.Timestamp(date(2019, 1, 1))]
            # df.date = df.date - np.timedelta64(2000, 'Y')
            if ii < len(scenarios_list[k]['prev']) - 1:
                # a = 3
                ax.plot(df.date, df[vals], color=c, label='_nolegend_', linestyle=':', alpha=0.5)
            # last one is an average of the previous scenarios
            else:
                ax.plot(df.date, df[vals], color=c, label=k, linestyle='-')
        ax.xaxis_date()
        colour_counter += 1
    ax.set(xlabel='logging date',
           ylabel=vals,
           title=vals + ' per date, ' + age + ', S.' + infection)
    if vals == 'Prevalence':
        plt.ylim([0, 0.35])
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    if vals == 'High-inf_Prevalence':
        plt.yscale('log')
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
    plt.legend()
    plt.show()

plot_per_age_group(scenarios, 'All', 'haematobium', 'Prevalence')
plot_per_age_group(scenarios, 'PSAC', 'haematobium', 'High-inf_Prevalence')

def plot_measure(scenarios_list, measure_type):
    assert measure_type in ['dalys', 'prev_years']
    if measure_type == 'dalys':
        value_col = 'YLD_Schisto_Haematobium_Schisto_Haematobium_Symptoms'
        title = 'DALYs per year per 10.000 ppl'
    else:
        value_col = 'Prevalent_years_this_year_total'
        title = 'Prevalent years per year per 10.000 ppl'
    fig, ax = plt.subplots(figsize=(9, 7))
    colours = ['b', 'm', 'y', 'g', 'k', 'c', 'r']
    colour_counter = 0
    for k in scenarios_list.keys():
        c = colours[colour_counter]
        for ii in range(len(scenarios_list[k]['prev'])):
            df = scenarios_list[k][measure_type][ii]
            # df_before_2019 = df[(df.date <= pd.Timestamp(date(2019, 1, 1))) & (df.date >= pd.Timestamp(date(2014, 1, 1)))]
            # df_after_2019 = df[df.date >= pd.Timestamp(date(2018, 12, 30))]
            # ax.plot(df_after_2019.date, df_after_2019[value_col], label=k, linestyle=':')
            # ax.plot(df_before_2019.date, df_before_2019[value_col], color='b', label='_nolegend_', linestyle='-')

            # df = df[df.date >= pd.Timestamp(date(2014, 1, 1))]
            if ii < len(scenarios_list[k]['prev']) - 1:
                ax.plot(df.date, df[value_col], color=c, label='_nolegend_', linestyle=':', alpha=0.5)
            # last one is an average of the previous scenarios
            else:
                ax.plot(df.date, df[value_col], color=c, label=k, linestyle='-')
        colour_counter += 1
        ax.xaxis_date()
    ax.set(xlabel='logging date',
           ylabel=measure_type,
           title=title)
    ax.xaxis.set_major_formatter(DateFormatter("%m/%y"))
    plt.xticks(rotation='vertical')
    plt.legend()
    plt.show()

plot_measure(scenarios, 'dalys')

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

find_year_of_elimination(scenarios, 'PSAC', 0.01)
find_year_of_elimination(scenarios, 'PSAC', 0.00)

def get_average_final(scenarios_list):
    output_dict = {}
    for k in scenarios_list.keys():
        df = scenarios_list[k]['prev'][-1]  # take the last dataframe which contains averages
        df = df.iloc[-4:]  # last 4 rows will contain the final averages for each age group
        df['Scenario'] = k
        df.set_index('Age_group', inplace=True)
        output_dict.update({k: df})
    return output_dict

def save_averages_final_in_csv(scenarios_list, vals):
    avg_dict = get_average_final(scenarios_list)
    df_list = [df for df in avg_dict.values()]
    avg_df = pd.concat(df_list)

    avg_df.to_csv(load_path + 'average_finals.csv')

# od = get_average_final(scenarios, 'Prevalence')
save_averages_final_in_csv(scenarios, 'Prevalence')

# def plot_dalys_age_group(sim_dict, age):
#     fig, ax = plt.subplots(figsize=(9, 7))
#     for k in sim_dict.keys():
#         df = sim_dict[k]['dalys']
#         df = df[df['Age_group'] == age]
#         df_before_2019 = df[(df.date <= pd.Timestamp(date(2019, 1, 1))) & (df.date >= pd.Timestamp(date(2014, 1, 1)))]
#         df_after_2019 = df[df.date >= pd.Timestamp(date(2019, 1, 1))]
#         ax.plot(df_after_2019.date, df_after_2019.DALY_monthly, label=k, linestyle=':')
#         ax.plot(df_before_2019.date, df_before_2019.DALY_monthly, color='b', label='_nolegend_', linestyle='-')
#         ax.xaxis_date()
#     ax.set(xlabel='logging date',
#            ylabel='DALYs',
#            title='DALYs per year, ' + age)
#     ax.xaxis.set_major_formatter(DateFormatter("%m/%y"))
#     plt.xticks(rotation='vertical')
#     plt.legend()
#     plt.show()

def count_DALYs_after_2019(sim_dict):
    print("Total DALYs after 2019:")
    dalys = []
    sims = []
    for k in sim_dict.keys():
        df = sim_dict[k]['dalys']
        df_after_2019 = df[df.date >= pd.Timestamp(date(2019, 1, 1))]
        value = round(df_after_2019.DALY_cumulative.values[-1])
        print(k, ":", value)
        k = k.replace(', ', ',\n')
        sims.append(k)
        dalys.append(value)
    plt.figure(figsize=(9, 7))
    plt.bar(x=sims, height=dalys)
    # plt.xticks(rotation=45)
    plt.xlabel('simulation')
    plt.ylabel('DALYs after 2019')
    plt.title('DALYs per simulation, years 2019-2025')
    plt.show()

#########################################################################################################

# Load the simulations you want to compare
load_path = 'C:/Users/ieh19/Desktop/Project 1/model_outputs/'
timestamps1 = ['2020-01-24_17-05-00', '2020-01-24_17-05-28', '2020-01-24_17-05-41']
timestamps2 = ['2020-01-24_19-36-24', '2020-01-24_19-37-08', '2020-01-24_19-37-18']
timestamps22 = ['2020-01-28_18-15-26', '2020-01-28_18-16-42', '2020-01-28_18-16-57']
timestamps3 = ['2020-01-24_15-37-04', '2020-01-24_15-37-24', '2020-01-24_15-37-40']
timestamps4 = ['2020-01-26_14-21-57', '2020-01-26_14-22-20', '2020-01-26_14-22-43']
timestamps5 = ['2020-01-24_14-38-32', '2020-01-24_14-38-55', '2020-01-24_14-38-45']
timestamps6 = ['2020-01-28_18-15-26', '2020-01-28_18-16-42', '2020-01-28_18-16-57']
timestamps = [timestamps1, timestamps2, timestamps22, timestamps3, timestamps4, timestamps5, timestamps6]
labels = ['No MDA after 2019',
          'MDA once per year, PSAC = 0%',
          'MDA once per year, PSAC = 25%',
          'MDA once per year, PSAC = 50%',
          'MDA twice per year, PSAC = 0%',
          'MDA twice per year, PSAC = 50%',
          'MDA once per year, PSAC = 25%']

# these are with varying the worm threshold for hihg-intensity infections among PSAC
timestamps1 = ['2020-01-29_00-08-43', '2020-01-29_00-11-27', '2020-01-29_00-11-48']
timestamps2 = ['2020-01-29_19-56-19', '2020-01-29_19-56-55', '2020-01-29_20-04-17']
timestamps22 = ['2020-01-29_10-15-44', '2020-01-29_10-17-43', '2020-01-29_10-18-24']
timestamps3 = ['2020-01-24_19-36-24', '2020-01-24_19-37-08', '2020-01-24_19-37-18']
timestamps4 = ['2020-01-29_00-07-40', '2020-01-29_00-06-18', '2020-01-29_00-10-17']
timestamps5 = ['2020-01-28_21-19-00', '2020-01-28_21-20-01', '2020-01-28_21-19-30']
timestamps55 = ['2020-01-29_10-22-42', '2020-01-29_10-23-09', '2020-01-29_11-07-02']
timestamps6 = ['2020-01-28_18-15-26', '2020-01-28_18-16-42', '2020-01-28_18-16-57']
timestamps66 = ['2020-01-29_21-59-21', '2020-01-29_22-09-25', '2020-01-29_22-27-23']
timestamps7 = ['2020-01-28_17-05-34', '2020-01-28_17-06-44', '2020-01-28_17-06-53']
timestamps77 = ['2020-01-29_22-54-31', '2020-01-29_22-54-49', '2020-01-29_22-55-02']
timestamps8 = ['2020-01-24_15-37-04', '2020-01-24_15-37-24', '2020-01-24_15-37-40']
timestamps = [timestamps1, timestamps2, timestamps22, timestamps3, timestamps4, timestamps5,
              timestamps55, timestamps6, timestamps66, timestamps7, timestamps77, timestamps8]
labels = ['PSAC 0%, 5 worms', 'PSAC 0% 10 worms', 'PSAC 0% 15 worms', 'PSAC 0% 20 worms',
          'PSAC 25% 5 worms', 'PSAC 25% 10 worms', 'PSAC 25% 15 worms', 'PSAC 25% 20 worms',
          'PSAC 50% 5 worms', 'PSAC 50% 10 worms', 'PSAC 50% 15 worms', 'PSAC 50% 20 worms']

# labels = ['25 years, no MDA', '25 years, no MDA, TreatmentSeeking on']
# timestamps = [['2020-01-24_12-02-31', '2020-01-24_12-02-55', '2020-01-24_12-03-06'],
#               ['2020-01-26_21-34-32', '2020-01-26_21-37-27', '2020-01-26_21-36-24']]



# timestamps = timestamps[1:]
# labels = labels[1:]
scenarios = dict((labels[i], timestamps[i]) for i in range(len(labels)))

for label, timestamps in scenarios.items():
    # load all results for the given scenario
    prev = []
    dalys = []
    prev_years = []
    for time in timestamps:
        prev_t, dalys_t, prev_years_t = load_outputs(time, 'Haematobium')
        prev.append(prev_t)
        dalys.append(dalys_t)
        prev_years.append(prev_years_t)

    prev.append(get_averages_prev(prev))
    dalys.append(get_averages_dalys_or_prev_years(dalys, 'dalys'))
    prev_years.append(get_averages_dalys_or_prev_years(prev_years, 'prev_years'))
    scenario_outputs = {'prev': prev, 'dalys': dalys, 'prev_years': prev_years }
    scenarios.update({label: scenario_outputs})


# plots
# for age_group in ['PSAC', 'SAC', 'Adults', 'All']:
#     plot_per_age_group(scenarios, age_group, 'haematobium', 'Prevalence')
for age_group in ['PSAC', 'SAC']:
    plot_per_age_group(scenarios, age_group, 'haematobium', 'Prevalence')

plot_per_age_group(scenarios, 'PSAC', 'haematobium', 'High-inf_Prevalence')
plot_per_age_group(scenarios, 'SAC', 'haematobium', 'High-inf_Prevalence')

for age_group in ['PSAC', 'SAC', 'Adults', 'All']:
    plot_per_age_group(scenarios, age_group, 'haematobium', 'MeanWormBurden')

plot_measure(scenarios, 'dalys')
plot_measure(scenarios, 'prev_years')

# for testing
plot_per_age_group(scenarios, 'PSAC', 'haematobium', 'Prevalence')
plot_per_age_group(scenarios, 'SAC', 'haematobium', 'MeanWormBurden')

plot_per_age_group(simulations_total, 'Adults', 'Prevalence')
plot_per_age_group(simulations_total, 'All', 'Prevalence')
df = simulations_total['MDA twice per year, PSAC coverage 50%']['dalys']

def find_worms_from_string(string):
    numbers_in_string = [int(s) for s in string.split() if s.isdigit()]
    return numbers_in_string[0]

def plot_prevalence_with_worm_thresholds(scenarios_list, age, infection, vals):
    assert vals in ['Prevalence', 'High-inf_Prevalence']
    worms5 = ['PSAC 0%, 5 worms', 'PSAC 25% 5 worms']
    worms10 = ['PSAC 0% 10 worms', 'PSAC 25% 10 worms', 'PSAC 50% 10 worms']
    worms15 = ['PSAC 0% 15 worms', 'PSAC 25% 15 worms']
    worms20 = ['PSAC 0% 20 worms','PSAC 25% 20 worms', 'PSAC 50% 20 worms']
    # fig, ax = plt.subplots(figsize=(9,7))
    fig, ax = plt.subplots()
    colours = ['b', 'm', 'g']
    colour_counter = 0
    for k in worms5:
        df = scenarios_list[k]['prev'][-1]
        df = df[df['Age_group'] == age]
        df = df[df.date >= pd.Timestamp(date(2019, 1, 1))]
        plt.subplot(221)
        plt.plot(df.date, df[vals], color=colours[colour_counter], label=k, linestyle='-')
        colour_counter += 1
    ax.xaxis_date()
    plt.legend()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    colour_counter = 0
    for k in worms10:
        df = scenarios_list[k]['prev'][-1]
        df = df[df['Age_group'] == age]
        df = df[df.date >= pd.Timestamp(date(2019, 1, 1))]
        plt.subplot(222)
        ax.plot(df.date, df[vals], color=colours[colour_counter], label=k, linestyle='-')
        colour_counter += 1
    plt.legend()
    ax.xaxis_date()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    colour_counter = 0
    for k in worms15:
        df = scenarios_list[k]['prev'][-1]
        df = df[df['Age_group'] == age]
        df = df[df.date >= pd.Timestamp(date(2019, 1, 1))]
        plt.subplot(223)
        ax.plot(df.date, df[vals], color=colours[colour_counter], label=k, linestyle='-')
        colour_counter += 1
    ax.xaxis_date()
    plt.legend()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    colour_counter = 0
    for k in worms20:
        df = scenarios_list[k]['prev'][-1]
        df = df[df['Age_group'] == age]
        df = df[df.date >= pd.Timestamp(date(2019, 1, 1))]
        plt.subplot(224)
        ax.plot(df.date, df[vals], color=colours[colour_counter], label=k, linestyle='-')
        colour_counter += 1
    ax.xaxis_date()
    plt.legend()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set(xlabel='logging date',
           ylabel=vals,
           title=vals + ' per date, ' + age + ', S.' + infection)
    ax.grid(linewidth=0.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    plt.xticks(rotation='vertical')
    plt.show()
plot_prevalence_with_worm_thresholds(scenarios, 'PSAC', 'haematobium', 'High-inf_Prevalence')
