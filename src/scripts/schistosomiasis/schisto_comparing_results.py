import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import datetime


def load_outputs(timestamp):
    prevalence = pd.read_csv(load_path + "output_" + timestamp + ".csv")
    prevalence.date = pd.to_datetime(prevalence.date)
    dalys = pd.read_csv(load_path + "output_daly_" + timestamp + ".csv")
    dalys.date = pd.to_datetime(dalys.date)
    return prevalence, dalys


def plot_prev_age_group(sim_dict, age):
    fig, ax = plt.subplots(figsize=(9, 7))
    for k in sim_dict.keys():
        df = sim_dict[k]['prev']
        df = df[df['Age_group'] == age]
        ax.plot(df.date, df.Prevalence, label=k)
        ax.xaxis_date()
    ax.set(xlabel='logging date',
           ylabel='fraction of infected sub-population',
           title='Prevalence per date, ' + age)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(DateFormatter("%y/%m"))
    plt.xticks(rotation='vertical')
    plt.legend()
    plt.show()


def plot_dalys_age_group(sim_dict, age):
    fig, ax = plt.subplots(figsize=(9, 7))
    for k in sim_dict.keys():
        df = sim_dict[k]['dalys']
        df = df[df['Age_group'] == age]
        ax.plot(df.date, df.DALY_yearly, label=k)
        ax.xaxis_date()
    ax.set(xlabel='logging date',
           ylabel='DALYs',
           title='DALYs per date, ' + age)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(DateFormatter("%y/%m"))
    plt.xticks(rotation='vertical')
    plt.legend()
    plt.show()

#########################################################################################################

# Load the simulations you want to compare
simulations = {}
load_path = 'C:/Users/ieh19/Desktop/Project 1/model_outputs/'
timestamps = ['2019-12-02_10-33-48', '2019-12-02_11-21-27', '2019-12-02_12-25-55', '2019-12-02_12-53-29', '2019-12-02_13-00-15']
# timestamps = ['2019-12-02_10-33-48']
for t in timestamps:
    prev, dalys = load_outputs(t)
    outputs = {'prev': prev, 'dalys': dalys}
    simulations.update({t: outputs})

# plots
plot_prev_age_group(simulations, 'SAC')
plot_dalys_age_group(simulations, 'SAC')

# plots
for age_group in ['PSAC', 'SAC', 'Adults', 'All']:
    plot_prev_age_group(simulations, age_group)
    plot_dalys_age_group(simulations, age_group)

