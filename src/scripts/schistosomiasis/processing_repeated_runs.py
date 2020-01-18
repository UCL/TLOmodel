import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from pathlib import Path
from datetime import date
import numpy as np

def load_outputs(timestamp, infection):
    infection_outputs = pd.read_csv(load_path + "output_" + infection + '_'+ timestamp + ".csv")
    infection_outputs.date = pd.to_datetime(infection_outputs.date)
    # infection_outputs.date = infection_outputs.date - np.timedelta64(10, 'Y')
    dalys = pd.read_csv(load_path + "output_daly_" + timestamp + ".csv")
    dalys.date = pd.to_datetime(dalys.date)
    # dalys.date = dalys.date - np.timedelta64(10, 'Y')
    prev_years = pd.read_csv(load_path + "output_prevalent_years_" + timestamp + ".csv")
    prev_years.date = pd.to_datetime(prev_years.date)
    # prev_years.date = prev_years.date - np.timedelta64(10, 'Y')
    if infection != 'Total':
        district_outputs = pd.read_csv(load_path + "output_districts_prev_" + infection + '_'+ timestamp + ".csv")
        return infection_outputs, dalys, prev_years, district_outputs
    else:
        return infection_outputs, dalys, prev_years

def get_averages_prev(sims):
    list_of_dfs = []
    for k in sims.keys():
        df = sims[k]['prev']
        list_of_dfs.append(df)
    big_df = pd.concat(list_of_dfs, ignore_index=True)
    avg_df = big_df.groupby(['date', 'Age_group'], as_index=False).agg({'Prevalence': 'mean', 'MeanWormBurden': 'mean'})
    return avg_df

def get_averages_dalys(sims, value):
    list_of_dfs = []
    cols_of_interest = {'dalys': 'DALY_this_year_total', 'prev_years': 'Prevalent_years_this_year_total'}
    for k in sims.keys():
        df = sims[k][value]
        list_of_dfs.append(df)
    big_df = pd.concat(list_of_dfs, ignore_index=True)
    avg_df = big_df.groupby(['date'], as_index=False).agg({cols_of_interest[value]: 'mean'})
    return avg_df


def get_averages_districts(sims):
    list_of_dfs = []
    for k in sims.keys():
        df = sims[k]['distr']
        list_of_dfs.append(df)
    big_df = pd.concat(list_of_dfs, ignore_index=True)
    avg_df = big_df.groupby(['District'], as_index=False).agg({'Prevalence': 'mean', 'MWB': 'mean'})
    return avg_df


def plot_per_age_group(sim_dict, age, infection, vals):
    assert vals in ['Prevalence', 'MeanWormBurden']
    fig, ax = plt.subplots(figsize=(9, 7))
    for k in sim_dict.keys():
        ls=':'
        if k == 'avg':
            ls='-'
        df = sim_dict[k]['prev']
        df = df[df['Age_group'] == age]
        ax.plot(df.date, df[vals], label=k, linestyle=ls)
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

def get_expected_prevalence(infection):
    expected_district_prevalence = pd.read_excel(Path("./resources") / 'ResourceFile_Schisto.xlsx',
                                                 sheet_name='District_Params_' + infection.lower())
    expected_district_prevalence.set_index("District", inplace=True)
    expected_district_prevalence = expected_district_prevalence.loc[:, 'Prevalence'].to_dict()
    return expected_district_prevalence

def plot_prevalence_per_district(sims, infection):
    expected_prev = get_expected_prevalence(infection)
    df = sims['avg']['distr'].copy()
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

# Load the simulations you want to compare
simulations_haematobium = {}
simulations_mansoni = {}
simulations_total = {}

load_path = 'C:/Users/ieh19/Desktop/Project 1/model_outputs/'
timestamps =['2020-01-17_21-10-12', '2020-01-17_23-24-12', '2020-01-17_23-24-27']
labels = ['sim1', 'sim2', 'sim3']

sim_dict = dict(zip(timestamps, labels))

for time, label in sim_dict.items():
    prev, dalys, prev_years, distr = load_outputs(time, 'Haematobium')
    outputs_haematobium = {'prev': prev, 'dalys': dalys, 'prev_years': prev_years, 'distr': distr}
    simulations_haematobium.update({label: outputs_haematobium})
    prev, dalys, prev_years, distr = load_outputs(time, 'Mansoni')
    outputs_mansoni = {'prev': prev, 'dalys': dalys, 'prev_years': prev_years, 'distr': distr}
    simulations_mansoni.update({label: outputs_mansoni})
    prev, dalys, prev_years = load_outputs(time, 'Total')
    outputs_total = {'prev': prev, 'dalys': dalys, 'prev_years': prev_years}
    simulations_total.update({label: outputs_total})


outputs_haematobium = {'prev': get_averages_prev(simulations_haematobium),
                       'dalys': get_averages_dalys(simulations_haematobium, 'dalys'),
                       'prev_years': get_averages_dalys(simulations_haematobium, 'prev_years'),
                       'distr': get_averages_districts(simulations_haematobium)}
simulations_haematobium.update({'avg': outputs_haematobium})

# plots
for age_group in ['PSAC', 'SAC', 'Adults', 'All']:
    plot_per_age_group(simulations_total, age_group, 'total', 'Prevalence')
for age_group in ['PSAC', 'SAC', 'Adults', 'All']:
    plot_per_age_group(simulations_haematobium, age_group, 'haematobium', 'Prevalence')
for age_group in ['PSAC', 'SAC', 'Adults', 'All']:
    plot_per_age_group(simulations_haematobium, age_group, 'haematobium', 'MeanWormBurden')
plot_prevalence_per_district(simulations_haematobium, 'Haematobium')

for age_group in ['PSAC', 'SAC', 'Adults', 'All']:
    plot_per_age_group(simulations_mansoni, age_group, 'mansoni', 'MeanWormBurden')

