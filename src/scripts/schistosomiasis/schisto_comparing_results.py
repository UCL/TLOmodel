import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

load_path = 'C:/Users/ieh19/Desktop/Project 1/model_outputs/'


def load_outputs(timestamp):
    prevalence = pd.read_csv(load_path + "output_" + timestamp + ".csv")
    dalys = pd.read_csv(load_path + "output_daly_" + timestamp + ".csv")
    return prevalence, dalys


# Load the simulations you want to compare
simulations = {}
timestamps = ['2019-12-02_10-33-48', '2019-12-02_11-21-27']
for t in timestamps:
    prev, dalys = load_outputs(t)
    outputs = {'prev': prev, 'dalys': dalys}
    simulations.update({t: outputs})

    # simulations.update({'timestamp': t, 'outputs': outputs})


# def plot_prev_age_group(sim_dict, age):
#     for k in sim_dict.keys():
#         df = sim_dict[k]['prev']
#         df = df[df['Age_group'] == age]
#         plt.plot(df.date, df.Prevalence, label=k)
#     plt.title('Prevalence per date, ' + age)
#     plt.ylabel('fraction of infected sub-population')
#     plt.xlabel('logging date')
#     plt.xticks(rotation='vertical')
#     plt.ylim([0, 1])
#     plt.legend()
#     plt.show()


def plot_prev_age_group(sim_dict, age):
    fig, ax = plt.subplots(figsize=(9, 7))
    for k in sim_dict.keys():
        df = sim_dict[k]['prev']
        df = df[df['Age_group'] == age]
        ax.plot(df.date, df.Prevalence, label=k)
    ax.set(xlabel='logging date',
           ylabel='fraction of infected sub-population',
           title='Prevalence per date, ' + age)
    # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    # ax.xaxis.set_major_formatter(DateFormatter("%y %m"))
    # ax.grid(True)
    plt.xticks(rotation='vertical')
    plt.legend()
    plt.show()

plot_prev_age_group(simulations, 'SAC')

def plot_dalys_age_group(sim_dict, age):
    for k in sim_dict.keys():
        df = sim_dict[k]['dalys']
        df = df[df['Age_group'] == age]
        plt.scatter(df.date, df.DALY_yearly, label=k)
    plt.title('DALYs per date, ' + age)
    plt.ylabel('DALYs')
    plt.xlabel('logging date')
    plt.xticks(rotation='vertical')
    plt.legend()
    plt.show()


for age_group in ['PSAC', 'SAC', 'Adults', 'All']:
    plot_prev_age_group(simulations, age_group)
    plot_dalys_age_group(simulations, age_group)

