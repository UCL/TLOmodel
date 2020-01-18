import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from datetime import date
import numpy as np

def load_outputs(timestamp, infection):
    infection_outputs = pd.read_csv(load_path + "output_" + infection + '_'+ timestamp + ".csv")
    infection_outputs.date = pd.to_datetime(infection_outputs.date)
    infection_outputs.date = infection_outputs.date - np.timedelta64(10, 'Y')
    dalys = pd.read_csv(load_path + "output_daly_" + timestamp + ".csv")
    dalys.date = pd.to_datetime(dalys.date)
    dalys.date = dalys.date - np.timedelta64(10, 'Y')
    prev_years = pd.read_csv(load_path + "output_prevalent_years_" + timestamp + ".csv")
    prev_years.date = pd.to_datetime(prev_years.date)
    prev_years.date = prev_years.date - np.timedelta64(10, 'Y')
    return infection_outputs, dalys, prev_years


def plot_per_age_group(sim_dict, age, infection, vals):
    assert vals in ['Prevalence', 'MeanWormBurden']
    fig, ax = plt.subplots(figsize=(9, 7))
    for k in sim_dict.keys():
        df = sim_dict[k]['prev']
        df = df[df['Age_group'] == age]
        df_before_2019 = df[(df.date <= pd.Timestamp(date(2019, 1, 1))) & (df.date >= pd.Timestamp(date(2014, 1, 1)))]
        df_after_2019 = df[df.date >= pd.Timestamp(date(2019, 1, 1))]
        ax.plot(df_after_2019.date, df_after_2019[vals], label=k, linestyle=':')
        ax.plot(df_before_2019.date, df_before_2019[vals], color='b', label='_nolegend_', linestyle='-')
        ax.xaxis_date()
    ax.set(xlabel='logging date',
           ylabel=vals,
           title=vals + ' per date, ' + age + ', S.' + infection)
    if vals == 'Prevalence':
        plt.ylim([0, 0.5])
    # else:
    #     plt.ylim([0, 2.5])
    # ax.grid()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(DateFormatter("%m/%y"))
    plt.xticks(rotation='vertical')
    plt.legend()
    plt.show()


def plot_measure(sim_dict, measure_type):
    assert measure_type in ['dalys', 'prev_years']
    if measure_type == 'dalys':
        value_col = 'DALY_this_year_total'
        title = 'DALYs per year per 10.000 ppl'
    else:
        value_col = 'Prevalent_years_this_year_total'
        title = 'Prevalent years per year per 10.000 ppl'
    fig, ax = plt.subplots(figsize=(9, 7))
    for k in sim_dict.keys():
        df = sim_dict[k][measure_type]
        df_before_2019 = df[(df.date <= pd.Timestamp(date(2019, 1, 1))) & (df.date >= pd.Timestamp(date(2014, 1, 1)))]
        df_after_2019 = df[df.date >= pd.Timestamp(date(2018, 12, 30))]
        ax.plot(df_after_2019.date, df_after_2019[value_col], label=k, linestyle=':')
        ax.plot(df_before_2019.date, df_before_2019[value_col], color='b', label='_nolegend_', linestyle='-')
        ax.xaxis_date()
    ax.set(xlabel='logging date',
           ylabel=measure_type,
           title=title)
    ax.xaxis.set_major_formatter(DateFormatter("%m/%y"))
    plt.xticks(rotation='vertical')
    plt.legend()
    plt.show()

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

# def plot_cum_inf_time(sim_dict):
#     plt.figure(figsize=(9, 7))
#     for k in sim_dict.keys():
#         df = sim_dict[k]['cum_inf_time']
#         values = df['ss_cumulative_infection_time'].values
#         values = values[values != 0]
#         plt.hist(values, bins=100, alpha=0.5, label=k)
#     plt.xlabel('Cumulative infected times, days')
#     plt.ylabel('Number of observations')
#     plt.title('Distribution of cumulative infection times in ' + str(10) + ' years')
#     plt.legend()
#     plt.show()

#########################################################################################################

# Load the simulations you want to compare
simulations_haematobium = {}
simulations_mansoni = {}
simulations_total = {}

load_path = 'C:/Users/ieh19/Desktop/Project 1/model_outputs/'
timestamps1 = ['2020-01-15_11-20-17', '2020-01-15_11-21-00', '2020-01-15_11-21-12']
timestamps2 = ['2020-01-15_17-06-57', '2020-01-15_17-07-17']
timestamps3 = ['2020-01-16_07-16-09', '2020-01-16_07-16-27']
timestamps = timestamps1 + timestamps2 + timestamps3

labels1 = ['No MDA', 'MDA once per year, SAC 80%', 'MDA twice per year, SAC 80%']
labels2 = ['MDA once per year, SAC 80%, PSAC 50%', 'MDA twice per year, SAC 80%, PSAC 50%']
labels3 = ['MDA once per year, SAC 80%, PSAC 50%, Adults 50%', 'MDA twice per year, SAC 80%, PSAC 50%, Adults 50%']
labels = labels1 + labels2 + labels3

# timestamps = ['2020-01-09_12-02-59', '2020-01-09_12-34-32', '2020-01-09_11-43-25', '2020-01-09_11-24-44',
#               '2020-01-09_13-37-38', '2020-01-09_14-42-29']
#
# labels = ['MDA once per year, PSAC coverage 0%', 'MDA once per year, PSAC coverage 50%',
#           'MDA twice per year, PSAC coverage 0%', 'MDA twice per year, PSAC coverage 50%',
#           'MDA once every two years, PSAC coverage 0%', 'MDA once per year, PSAC & Adults coverage 0%']

sim_dict = dict(zip(timestamps, labels))

for time, label in sim_dict.items():
    prev, dalys, prev_years = load_outputs(time, 'Haematobium')
    outputs_haematobium = {'prev': prev, 'dalys': dalys, 'prev_years': prev_years}
    simulations_haematobium.update({label: outputs_haematobium})
    prev, dalys, prev_years = load_outputs(time, 'Mansoni')
    outputs_mansoni = {'prev': prev, 'dalys': dalys, 'prev_years': prev_years}
    simulations_mansoni.update({label: outputs_mansoni})
    prev, dalys, prev_years = load_outputs(time, 'Total')
    outputs_total = {'prev': prev, 'dalys': dalys, 'prev_years': prev_years}
    simulations_total.update({label: outputs_total})


# plots
for age_group in ['PSAC', 'SAC', 'Adults', 'All']:
    plot_per_age_group(simulations_total, age_group, 'total', 'Prevalence')

for age_group in ['PSAC', 'SAC', 'Adults', 'All']:
    plot_per_age_group(simulations_haematobium, age_group, 'haematobium', 'MeanWormBurden')
for age_group in ['PSAC', 'SAC', 'Adults', 'All']:
    plot_per_age_group(simulations_mansoni, age_group, 'mansoni', 'MeanWormBurden')

plot_measure(simulations_total, 'dalys')
plot_measure(simulations_total, 'prev_years')

# for testing
plot_per_age_group(simulations_total, 'PSAC', 'Prevalence')
plot_per_age_group(simulations_total, 'Adults', 'Prevalence')
plot_per_age_group(simulations_total, 'All', 'Prevalence')
df = simulations_total['MDA twice per year, PSAC coverage 50%']['dalys']

# i don't think this works fine now
count_DALYs_after_2019(simulations)

