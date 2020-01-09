import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from datetime import date


def load_outputs(timestamp):
    prevalence = pd.read_csv(load_path + "output_" + timestamp + ".csv")
    prevalence.date = pd.to_datetime(prevalence.date)
    dalys = pd.read_csv(load_path + "output_daly_" + timestamp + ".csv")
    dalys.date = pd.to_datetime(dalys.date)
    prev_years = pd.read_csv(load_path + "output_prevalent_years_" + timestamp + ".csv")
    prev_years.date = pd.to_datetime(prev_years.date)
    return prevalence, dalys, prev_years


def plot_per_age_group(sim_dict, age, vals):
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
           ylabel='fraction of infected sub-population',
           title=vals + ' per date, ' + age)
    if vals == 'Prevalence':
        plt.ylim([0, 0.5])
    # ax.grid()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(DateFormatter("%m/%y"))
    plt.xticks(rotation='vertical')
    plt.legend()
    plt.show()


def plot_measure(sim_dict, measure_type):
    assert measure_type in ['dalys', 'prev_years']
    if measure_type == 'dalys':
        value_col = 'DALY_this_year'
        title = 'DALYs per year per 10.000 ppl'
    else:
        value_col = 'Prevalent_years_this_year'
        title = 'Prevalent years per year per 10.000 ppl'
    fig, ax = plt.subplots(figsize=(9, 7))
    for k in sim_dict.keys():
        df = sim_dict[k][measure_type]
        df_before_2019 = df[(df.date <= pd.Timestamp(date(2019, 1, 1))) & (df.date >= pd.Timestamp(date(2014, 1, 1)))]
        df_after_2019 = df[df.date >= pd.Timestamp(date(2019, 1, 1))]
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
simulations = {}
load_path = 'C:/Users/ieh19/Desktop/Project 1/model_outputs/'
# symptomless not treated in HSI
timestamps = ['2020-01-09_12-02-59', '2020-01-09_12-34-32', '2020-01-09_11-43-25', '2020-01-09_11-24-44',
              '2020-01-09_13-37-38', '2020-01-09_14-42-29']

labels = ['MDA once per year, PSAC coverage 0%', 'MDA once per year, PSAC coverage 50%',
          'MDA twice per year, PSAC coverage 0%', 'MDA twice per year, PSAC coverage 50%',
          'MDA once every two years, PSAC coverage 0%', 'MDA once per year, PSAC & Adults coverage 0%']

sim_dict = dict(zip(timestamps, labels))

for time, label in sim_dict.items():
    prev, dalys, prev_years = load_outputs(time)
    outputs = {'prev': prev, 'dalys': dalys, 'prev_years': prev_years}
    simulations.update({label: outputs})

# plots
for age_group in ['PSAC', 'SAC', 'Adults', 'All']:
    plot_per_age_group(simulations, age_group, 'Prevalence')

for age_group in ['PSAC', 'SAC', 'Adults', 'All']:
    plot_per_age_group(simulations, age_group, 'MeanWormBurden')

plot_measure(simulations, 'dalys')
plot_measure(simulations, 'prev_years')

# for testing
plot_per_age_group(simulations, 'PSAC', 'Prevalence')
plot_per_age_group(simulations, 'Adults', 'Prevalence')
plot_per_age_group(simulations, 'All', 'Prevalence')
df = simulations['MDA twice per year, PSAC coverage 50%']['dalys']

# i don't think this works fine now
count_DALYs_after_2019(simulations)

