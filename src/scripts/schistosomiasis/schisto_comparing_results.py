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
    cum_inf_time = pd.read_csv(load_path + "output_population_" + timestamp + ".csv")
    cum_inf_time = cum_inf_time[['sex', 'ss_cumulative_infection_time']]
    return prevalence, dalys, cum_inf_time


def plot_prev_age_group(sim_dict, age):
    fig, ax = plt.subplots(figsize=(9, 7))
    for k in sim_dict.keys():
        df = sim_dict[k]['prev']
        df = df[df['Age_group'] == age]
        df_before_2019 = df[df.date <= pd.Timestamp(date(2019, 1, 1))]
        df_after_2019 = df[df.date >= pd.Timestamp(date(2019, 1, 1))]
        ax.plot(df_after_2019.date, df_after_2019.Prevalence, label=k, linestyle=':')
        ax.plot(df_before_2019.date, df_before_2019.Prevalence, color='b', label='_nolegend_', linestyle='-')
        ax.xaxis_date()
    ax.set(xlabel='logging date',
           ylabel='fraction of infected sub-population',
           title='Prevalence per date, ' + age)
    plt.ylim([0, 0.5])
    # ax.grid()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(DateFormatter("%m/%y"))
    plt.xticks(rotation='vertical')
    plt.legend()
    plt.show()

def plot_dalys_age_group(sim_dict, age):
    fig, ax = plt.subplots(figsize=(9, 7))
    for k in sim_dict.keys():
        df = sim_dict[k]['dalys']
        df = df[df['Age_group'] == age]
        df_before_2019 = df[df.date <= pd.Timestamp(date(2019, 1, 1))]
        df_after_2019 = df[df.date >= pd.Timestamp(date(2019, 1, 1))]
        ax.plot(df_after_2019.date, df_after_2019.DALY_monthly, label=k, linestyle=':')
        ax.plot(df_before_2019.date, df_before_2019.DALY_monthly, color='b', label='_nolegend_', linestyle='-')
        ax.xaxis_date()
    ax.set(xlabel='logging date',
           ylabel='DALYs',
           title='DALYs per year, ' + age)
    ax.xaxis.set_major_formatter(DateFormatter("%m/%y"))
    plt.xticks(rotation='vertical')
    plt.legend()
    plt.show()

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

def plot_cum_inf_time(sim_dict):
    plt.figure(figsize=(9, 7))
    for k in sim_dict.keys():
        df = sim_dict[k]['cum_inf_time']
        values = df['ss_cumulative_infection_time'].values
        values = values[values != 0]
        plt.hist(values, bins=100, alpha=0.5, label=k)
    plt.xlabel('Cumulative infected times, days')
    plt.ylabel('Number of observations')
    plt.title('Distribution of cumulative infection times in ' + str(10) + ' years')
    plt.legend()
    plt.show()

#########################################################################################################

# Load the simulations you want to compare
simulations = {}
load_path = 'C:/Users/ieh19/Desktop/Project 1/model_outputs/'
# no repeated appt
# timestamps = ['2019-12-05_16-32-28', '2019-12-05_16-00-46', '2019-12-04_16-40-40', '2019-12-04_16-08-16']
# repeated appt
# timestamps = ['2019-12-06_15-00-53', '2019-12-06_13-14-06', '2019-12-06_14-28-44', '2019-12-06_13-57-36']
timestamps = ['2019-12-06_15-45-31', '2019-12-06_16-36-32', '2019-12-06_17-24-31', '2019-12-06_16-59-42']
# symptomless not treated in HSI
timestamps = ['2019-12-09_12-44-37', '2019-12-09_12-12-52', '2019-12-09_11-18-07', '2019-12-09_11-43-08']
# varying the prob_sent_to_schisto_test_adult
timestamps = ['2019-12-09_12-44-37', '2019-12-09_14-43-20', '2019-12-09_15-15-15', '2019-12-09_11-18-07', '2019-12-09_15-43-34']
# labels = ['MDA once per year, PSAC coverage 0%', 'MDA once per year, PSAC coverage 50%',
#           'MDA twice per year, PSAC coverage 0%', 'MDA twice per year, PSAC coverage 50%']
labels = ['MDA once per year, PSAC = 0.0, p_test_a_2019 = 0.4', 'MDA once per year, PSAC = 0.0, p_test_a_2019 = 0.6',
          'MDA once per year, PSAC = 0.5, p_test_a_2019 = 0.6', 'MDA twice per year, PSAC = 0.0, p_test_a_2019 = 0.4',
          'MDA twice per year, PSAC = 0.0, p_test_a_2019 = 0.6']

sim_dict = dict(zip(timestamps, labels))

for time, label in sim_dict.items():
    prev, dalys, cum_inf_time = load_outputs(time)
    outputs = {'prev': prev, 'dalys': dalys, 'cum_inf_time': cum_inf_time}
    simulations.update({label: outputs})

# plots
for age_group in ['PSAC', 'SAC', 'Adults', 'All']:
    plot_prev_age_group(simulations, age_group)

# DALYs and Cum inf times only make sense when we look at the whole population
plot_prev_age_group(simulations, 'PSAC')
plot_prev_age_group(simulations, 'Adults')

plot_prev_age_group(simulations, 'All')
# plot_dalys_age_group(simulations, 'All')
count_DALYs_after_2019(simulations)
plot_cum_inf_time(simulations)


# for testing
df = simulations['MDA twice per year, PSAC coverage 50%']['dalys']
