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
    #dalys = pd.read_csv(load_path + "output_daly_" + timestamp + ".csv")
    #dalys.date = pd.to_datetime(dalys.date)
    # dalys.date = dalys.date - np.timedelta64(10, 'Y')
    prev_years = pd.read_csv(load_path + "output_prevalent_years_" + timestamp + ".csv")
    prev_years.date = pd.to_datetime(prev_years.date)
    # prev_years.date = prev_years.date - np.timedelta64(10, 'Y')
    if infection != 'Total':
        district_outputs = pd.read_csv(load_path + "output_districts_prev_" + infection + '_'+ timestamp + ".csv")
        # return infection_outputs, dalys, prev_years, district_outputs
        return infection_outputs, prev_years, district_outputs
    else:
        # return infection_outputs, dalys, prev_years
        return infection_outputs, prev_years

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

def add_average(sim):
    avg_outputs = {'prev': get_averages_prev(sim),
                          # 'dalys': get_averages_dalys(sim, 'dalys'),
                           'prev_years': get_averages_dalys(sim, 'prev_years'),
                           'distr': get_averages_districts(sim)}
    sim.update({'avg': avg_outputs})
    return sim

def plot_per_age_group(sim_dict, age, infection, vals):
    assert vals in ['Prevalence', 'MeanWormBurden']
    fig, ax = plt.subplots(figsize=(9, 7))
    for k in sim_dict.keys():
        ls = ':'
        alpha = 0.5
        if k == 'avg':
            ls = '-'
            alpha = 1
        df = sim_dict[k]['prev']
        df = df[df['Age_group'] == age]
        ax.plot(df.date, df[vals], label=k, linestyle=ls, alpha=alpha)
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
    districts = ['Blantyre', 'Chiradzulu', 'Mulanje', 'Nsanje', 'Nkhotakota', 'Phalombe']
    expected_district_prevalence = expected_district_prevalence[expected_district_prevalence['District'].isin(districts)]
    expected_district_prevalence.set_index("District", inplace=True)
    expected_district_prevalence = expected_district_prevalence.loc[:, 'Prevalence'].to_dict()
    return expected_district_prevalence

def plot_prevalence_per_district(sims, infection):
    expected_prev = get_expected_prevalence(infection)
    df = sims['avg']['distr'].copy()
    districts = ['Blantyre', 'Chiradzulu', 'Mulanje', 'Nsanje', 'Nkhotakota', 'Phalombe']
    df = df[df['District'].isin(districts)]
    df.set_index("District", inplace=True)
    dict_dstr = df.loc[:, 'Prevalence'].to_dict()
    plt.bar(*zip(*dict_dstr.items()), alpha=0.5, label='simulations avg')
    plt.scatter(*zip(*expected_prev.items()), label='data')
    plt.xticks(rotation=90)
    plt.xlabel('District')
    plt.ylabel('Prevalence')
    plt.legend()
    plt.title('Prevalence per district, S.' + infection + ', TreatmentSeeking On')
    plt.show()

# Load the simulations you want to compare
simulations_haematobium = {}
simulations_mansoni = {}
simulations_total = {}

load_path = 'C:/Users/ieh19/Desktop/Project 1/model_outputs/'
# timestamps =['2020-01-17_21-10-12', '2020-01-17_23-24-12', '2020-01-17_23-24-27', '2020-01-19_15-02-43']
# timestamps = ['2020-01-19_22-22-02', '2020-01-19_22-22-18', '2020-01-19_22-22-32', '2020-01-19_22-22-44', '2020-01-19_22-22-55']
# timestamps = ['2020-01-20_09-03-54', '2020-01-20_09-07-28', '2020-01-20_09-10-23', '2020-01-20_09-10-28', '2020-01-20_09-10-23']
# timestamps = ['2020-01-20_17-04-45', '2020-01-20_17-05-07', '2020-01-20_17-05-23', '2020-01-20_19-37-43']
timestamps = ['2020-01-24_12-02-31', '2020-01-24_12-02-55', '2020-01-24_12-03-06']
timestamps = ['2020-01-26_21-34-32', '2020-01-26_21-37-27', '2020-01-26_21-36-24']

labels = ['sim' + str(i) for i in range(1,len(timestamps))]

sim_dict = dict(zip(timestamps, labels))

for time, label in sim_dict.items():
    prev, prev_years, distr = load_outputs(time, 'Haematobium')
    outputs_haematobium = {'prev': prev, 'dalys': '', 'prev_years': prev_years, 'distr': distr}
    simulations_haematobium.update({label: outputs_haematobium})
    # prev, prev_years, distr = load_outputs(time, 'Mansoni')
    # outputs_mansoni = {'prev': prev, 'dalys': '', 'prev_years': prev_years, 'distr': distr}
    # simulations_mansoni.update({label: outputs_mansoni})
    # prev, prev_years = load_outputs(time, 'Total')
    # outputs_total = {'prev': prev, 'dalys': '', 'prev_years': prev_years}
    # simulations_total.update({label: outputs_total})

simulations_haematobium = add_average(simulations_haematobium)
# simulations_mansoni = add_average(simulations_mansoni)

# plots
for age_group in ['PSAC', 'SAC', 'Adults', 'All']:
    plot_per_age_group(simulations_total, age_group, 'total', 'Prevalence')
for age_group in ['PSAC', 'SAC', 'Adults', 'All']:
    plot_per_age_group(simulations_haematobium, age_group, 'haematobium', 'Prevalence')
for age_group in ['PSAC', 'SAC', 'Adults', 'All']:
    plot_per_age_group(simulations_haematobium, age_group, 'haematobium', 'MeanWormBurden')
plot_prevalence_per_district(simulations_haematobium, 'Haematobium')

for age_group in ['PSAC', 'SAC', 'Adults', 'All']:
    plot_per_age_group(simulations_mansoni, age_group, 'mansoni', 'Prevalence')
for age_group in ['PSAC', 'SAC', 'Adults', 'All']:
    plot_per_age_group(simulations_mansoni, age_group, 'mansoni', 'MeanWormBurden')
plot_prevalence_per_district(simulations_mansoni, 'Mansoni')

