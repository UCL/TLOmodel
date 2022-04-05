"""This file is part of the MRes work by Iwona Hawryluk. The model has been updated since this script was created and
it does not run in full."""

import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import contraception, demography, healthburden, healthsystem, schisto


def run_simulation(popsize=10000, haem=True, mansoni=True, mda_execute=True):
    # The resource files
    resourcefilepath = Path("./resources")
    start_date = Date(2010, 1, 1)
    end_date = Date(2011, 2, 1)
    popsize = popsize

    # Establish the simulation object
    sim = Simulation(start_date=start_date)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(schisto.Schisto(resourcefilepath=resourcefilepath, mda_execute=mda_execute))
    if haem:
        sim.register(schisto.Schisto_Haematobium(resourcefilepath=resourcefilepath, symptoms_and_HSI=False))
    if mansoni:
        sim.register(schisto.Schisto_Mansoni(resourcefilepath=resourcefilepath, symptoms_and_HSI=False))

    # Sets all modules to WARNING threshold, then alters schisto to INFO
    custom_levels = {"*": logging.WARNING,
                     "tlo.methods.schisto": logging.INFO,
                     }
    # configure logging after registering modules with custom levels
    logfile = sim.configure_logging(filename="LogFile", custom_levels=custom_levels)

    # Run the simulation
    sim.seed_rngs(int(np.random.uniform(0, 1) * 0 + 1000))
    # initialise the population
    sim.make_initial_population(n=popsize)

    # # start the simulation
    sim.simulate(end_date=end_date)
    # fh.flush()
    output = parse_log_file(logfile)
    return sim, output


sim, output = run_simulation(popsize=10000, haem=True, mansoni=False, mda_execute=False)

# ---------------------------------------------------------------------------------------------------------
#   Saving the results - prevalence, mwb, dalys and parameters used
# ---------------------------------------------------------------------------------------------------------


# prevalence, mean worm burden and states count
def get_timestamp():
    timestamp = str(datetime.datetime.now().replace(microsecond=0))
    timestamp = timestamp.replace(" ", "_")
    timestamp = timestamp.replace(":", "-")
    return timestamp


timestamp = get_timestamp()
print(timestamp)


def save_inf_outputs(infection, save_the_districts=False):
    output_path = 'C:/Users/ieh19/Desktop/Project 1/model_outputs/'
    savepath = output_path + "output_" + infection + '_' + timestamp + ".csv"
    savepath_districts_prev = output_path + "output_districts_prev_" + infection + '_' + timestamp + ".csv"

    output_states = pd.DataFrame([])
    for age_group in ['PSAC', 'SAC', 'Adults', 'All']:
        output['tlo.methods.schisto'][age_group + '_' + infection]['Age_group'] = age_group
        output_states = \
            output_states.append(output['tlo.methods.schisto'][age_group + '_' + infection], ignore_index=True)
    output_states.to_csv(savepath, index=False)

    if save_the_districts:
        districts = ['Blantyre', 'Chiradzulu', 'Mulanje', 'Nsanje', 'Nkhotakota', 'Phalombe']
        distr_end_data = pd.DataFrame(columns=['District', 'Prevalence', 'MWB', 'High-infection-prevalence'])
        for distr in districts:
            prev = output['tlo.methods.schisto'][distr + '_' + infection]['Prevalence'].values[-1]
            mwb = output['tlo.methods.schisto'][distr + '_' + infection]['MeanWormBurden'].values[-1]
            high_inf = output['tlo.methods.schisto'][distr + '_' + infection]['High_infections'].values[-1]
            low_inf = output['tlo.methods.schisto'][distr + '_' + infection]['Low_infections'].values[-1]
            non_inf = output['tlo.methods.schisto'][distr + '_' + infection]['Non_infected'].values[-1]
            high_inf_p = high_inf / (non_inf + low_inf + high_inf)
            new_row = {'District': distr, 'Prevalence': prev, 'MWB': mwb,
                       'High-infection-prevalence': high_inf_p}
            distr_end_data = distr_end_data.append(new_row, ignore_index=True)

        distr_end_data.to_csv(savepath_districts_prev, index=False)


def save_general_outputs_and_params():
    output_path = 'C:/Users/ieh19/Desktop/Project 1/model_outputs/'
    # savepath = output_path + "output_Total" + '_' + timestamp + ".csv"
    savepath_full_pop = output_path + "output_population_" + timestamp + ".csv"
    savepath_prevalent_years = output_path + "output_prevalent_years_" + timestamp + ".csv"
    # savepath_daly = output_path + "output_daly_" + timestamp + ".csv"
    savepath_params = output_path + "input_" + timestamp + ".xlsx"

    df = sim.population.props
    df = df[df['is_alive']]
    df.to_csv(savepath_full_pop, index=False)

    output_states = pd.DataFrame([])
    # for age_group in ['PSAC', 'SAC', 'Adults', 'All']:
    #     output['tlo.methods.schisto'][age_group + '_Total']['Age_group'] = age_group
    #     output_states = output_states.append(output['tlo.methods.schisto'][age_group + '_Total'], ignore_index=True)
    # output_states.to_csv(savepath, index=False)

    # Prevalent years
    for age_group in ['PSAC', 'SAC', 'Adults', 'All']:
        output['tlo.methods.schisto'][age_group + '_PrevalentYears']['Age_group'] = age_group
        output_states = \
            output_states.append(output['tlo.methods.schisto'][age_group + '_PrevalentYears'], ignore_index=True)
    output_states.to_csv(savepath_prevalent_years, index=False)

    # loger_DALY_All = output['tlo.methods.healthburden']['DALYS']
    # loger_DALY_All.to_csv(savepath_daly, index=False)

    # parameters spreadsheet
    parameters_used = pd.read_excel(Path("./resources/ResourceFile_Schisto.xlsx"), sheet_name=None)
    writer = pd.ExcelWriter(savepath_params)
    for sheet_name in parameters_used.keys():
        parameters_used[sheet_name].to_excel(writer, sheet_name=sheet_name)
    writer.save()


save_inf_outputs('Haematobium', True)
# save_inf_outputs('Mansoni', False)
save_general_outputs_and_params()
# ---------------------------------------------------------------------------------------------------------
#   INSPECTING & PLOTTING
# ---------------------------------------------------------------------------------------------------------
df = sim.population.props
df = df[df['is_alive']]
df = df[df['district_of_residence'].isin(['Blantyre', 'Chiradzulu', 'Mulanje', 'Nsanje', 'Nkhotakota', 'Phalombe'])]


def get_logger(infection, cut_off_left=False):
    assert infection in ['Total', 'Haematobium', 'Mansoni'], 'Wrong infection type!'

    loger_PSAC = output['tlo.methods.schisto']['PSAC' + '_' + infection]
    loger_SAC = output['tlo.methods.schisto']['SAC' + '_' + infection]
    loger_Adults = output['tlo.methods.schisto']['Adults' + '_' + infection]
    loger_All = output['tlo.methods.schisto']['All' + '_' + infection]

    loger_PSAC.date = pd.to_datetime(loger_PSAC.date)
    loger_SAC.date = pd.to_datetime(loger_SAC.date)
    loger_Adults.date = pd.to_datetime(loger_Adults.date)
    loger_All.date = pd.to_datetime(loger_All.date)

    if cut_off_left:
        # remove the period of "equilibrating"
        loger_PSAC.date = loger_PSAC.date - np.timedelta64(15, 'Y')
        loger_SAC.date = loger_SAC.date - np.timedelta64(15, 'Y')
        loger_Adults.date = loger_Adults.date - np.timedelta64(15, 'Y')
        loger_All.date = loger_All.date - np.timedelta64(15, 'Y')
        # remove the period of "equilibrating"
        loger_PSAC = loger_PSAC[loger_PSAC.date >= pd.Timestamp(datetime.date(2014, 1, 1))]
        loger_SAC = loger_SAC[loger_SAC.date >= pd.Timestamp(datetime.date(2014, 1, 1))]
        loger_Adults = loger_Adults[loger_Adults.date >= pd.Timestamp(datetime.date(2014, 1, 1))]
        loger_All = loger_All[loger_All.date >= pd.Timestamp(datetime.date(2014, 1, 1))]

    log_dict = {'loger_PSAC': loger_PSAC, 'loger_SAC': loger_SAC,
                'loger_Adults': loger_Adults, 'loger_All': loger_All}
    return log_dict


# Prevalence
def plot_prevalence(loger_inf, infection):
    fig1, ax = plt.subplots(figsize=(9, 7))
    for log in loger_inf.keys():
        ax.plot(loger_inf[log].date, loger_inf[log].Prevalence, label=log[6:])
        ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    plt.xticks(rotation='vertical')
    plt.ylim([0, 1])
    plt.legend()
    plt.title('Prevalence per date, S.' + infection)
    plt.ylabel('fraction of infected sub-population')
    plt.xlabel('logging date')
    plt.show()


def plot_high_inf_prevalence(loger_inf, infection):
    fig1, ax = plt.subplots(figsize=(9, 7))
    for log in loger_inf.keys():
        ax.plot(loger_inf[log].date, loger_inf[log]['High-inf_Prevalence'], label=log[6:])
        ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    plt.xticks(rotation='vertical')
    plt.ylim([0, 1])
    plt.legend()
    plt.title('High-infection prevalence per date, S.' + infection)
    plt.ylabel('fraction of infected sub-population')
    plt.xlabel('logging date')
    plt.show()


plot_prevalence(get_logger('Haematobium', False), 'Haematobium')
# plot_prevalence(get_logger('Mansoni', True), 'Mansoni')
# plot_prevalence(get_logger('Total'), 'Total')
plot_high_inf_prevalence(get_logger('Haematobium'), 'Haematobium')


# Final prevalence for every district
def get_values_per_district(infection):
    districts = ['Blantyre', 'Chiradzulu', 'Mulanje', 'Nsanje', 'Nkhotakota', 'Phalombe']
    # districts = list(df.district_of_residence.unique())
    districts.sort()
    districts_prevalence = {}
    districts_mwb = {}
    districts_high_inf_prev = {}
    for distr in districts:
        prev = output['tlo.methods.schisto'][distr + '_' + infection].Prevalence.values[-1]
        mwb = output['tlo.methods.schisto'][distr + '_' + infection].MeanWormBurden.values[-1]
        high_inf = output['tlo.methods.schisto'][distr + '_' + infection]['High_infections'].values[-1]
        low_inf = output['tlo.methods.schisto'][distr + '_' + infection]['Low_infections'].values[-1]
        non_inf = output['tlo.methods.schisto'][distr + '_' + infection]['Non_infected'].values[-1]
        high_inf_p = high_inf / (non_inf + low_inf + high_inf)
        districts_prevalence.update({distr: prev})
        districts_mwb.update({distr: mwb})
        districts_high_inf_prev.update({distr: high_inf_p})
    return districts_prevalence, districts_mwb, districts_high_inf_prev


def get_expected_prevalence(infection):
    expected_district_prevalence = pd.read_excel(Path("./resources") / 'ResourceFile_Schisto.xlsx',
                                                 sheet_name='District_Params_' + infection.lower())
    expected_district_prevalence = \
        expected_district_prevalence[expected_district_prevalence['District'].isin(['Blantyre',
                                                                                    'Chiradzulu', 'Mulanje', 'Nsanje',
                                                                                    'Nkhotakota', 'Phalombe'])]
    expected_district_prevalence.set_index("District", inplace=True)
    expected_district_prevalence = expected_district_prevalence.loc[:, 'Prevalence'].to_dict()
    return expected_district_prevalence


def plot_prevalence_per_district(infection, model_prev, expected_prev):
    plt.bar(*zip(*model_prev.items()), alpha=0.5, label='model')
    plt.scatter(*zip(*expected_prev.items()), label='data')
    plt.xticks(rotation=90)
    plt.xlabel('District')
    plt.ylabel('Prevalence')
    plt.legend()
    plt.title('Prevalence per district, S.' + infection)
    plt.show()


def plot_mwb_per_district(infection, mwb_distr):
    plt.bar(*zip(*mwb_distr.items()), alpha=0.5, label='model')
    plt.xticks(rotation=90)
    plt.xlabel('District')
    plt.ylabel('Mean Worm Burden')
    plt.legend()
    plt.title('Mean Worm Burden per district, S.' + infection)
    plt.show()


def plot_prev_high_infection_per_district(infection, high_inf_distr):
    plt.bar(*zip(*high_inf_distr.items()), alpha=0.5, label='model')
    plt.xticks(rotation=90)
    plt.xlabel('District')
    plt.ylabel('Prevalence')
    plt.legend()
    plt.title('Prevalence of high infections per district, S.' + infection)
    plt.show()


districts_prevalence, districts_mwb, districts_high_inf_prev = get_values_per_district('Haematobium')
expected_district_prevalence = get_expected_prevalence('Haematobium')
plot_prevalence_per_district('Haematobium', districts_prevalence, expected_district_prevalence)
# plot_mwb_per_district('Haematobium', districts_mwb)
# plot_prev_high_infection_per_district('Haematobium', districts_high_inf_prev)


# Mean Worm Burden per month
def plot_mwb_monthly(loger_inf, infection):
    fig, ax = plt.subplots(figsize=(9, 7))
    for log in loger_inf.keys():
        ax.plot(loger_inf[log].date, loger_inf[log].MeanWormBurden, label=log[6:])
        ax.xaxis_date()
    plt.xticks(rotation='vertical')
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=36))
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    plt.legend()
    # plt.ylim([0, 20])
    plt.title('Mean Worm Burden per date, S.' + infection)
    plt.ylabel('Mean number of worms')
    plt.xlabel('logging date')
    plt.show()


# Worm burden distribution at the end of the simulation & fitted NegBin
def plot_worm_burden_histogram(infection):
    if infection == 'Haematobium':
        prefix = 'sh'
    else:
        prefix = 'sm'
    wb = df[prefix + '_aggregate_worm_burden'].values
    # r = 0.1238771
    # p = 0.00902744486919016
    # print('clumping param k=', r, ', mean = ', p)
    # negbin = np.random.negative_binomial(r, p, size=len(wb))
    plt.hist(wb, bins=100, density=True, label='model')
    # x = np.linspace(0, len(wb))
    # plt.plot(x, negbin, label = 'parameter fit')
    # plt.hist(negbin, bins=100, density=True, label='param fit, k= ' + str(round(r, 2)), alpha=0.3)
    plt.xlabel('Worm burden')
    plt.ylabel('Count')
    # plt.xlim([0, 200])
    plt.legend()
    plt.title('Aggregate worm burden distribution, S.' + infection)
    plt.show()


# Harbouring rates distributions
def plot_harbouring_rates(infection):
    if infection == 'Haematobium':
        prefix = 'sh'
    else:
        prefix = 'sm'
    hr = df[prefix + '_harbouring_rate'].values
    plt.hist(hr, bins=1000, density=True)
    plt.xlabel('Harbouring rates')
    plt.ylabel('Count')
    plt.title('Harbouring rates distribution, S.' + infection)
    plt.show()


# Mean worm burden per age group - bar plots - at the end of the simulation
def plot_mwb_bar_plots(infection):
    if infection == 'Haematobium':
        prefix = 'sh'
    else:
        prefix = 'sm'
    age_map = {'0-4': 'PSAC', '5-9': 'SAC', '10-14': 'SAC'}
    df['age_group'] = df['age_range'].map(age_map)
    df['age_group'].fillna('Adults', inplace=True)  # the reminder will be Adults
    mwb_adults = df[df['age_group'] == 'Adults'][prefix + '_aggregate_worm_burden'].values.mean()
    mwb_sac = df[df['age_group'] == 'SAC'][prefix + '_aggregate_worm_burden'].values.mean()
    mwb_psac = df[df['age_group'] == 'PSAC'][prefix + '_aggregate_worm_burden'].values.mean()
    plt.bar(x=['PSAC', 'SAC', 'Adults'], height=[mwb_psac, mwb_sac, mwb_adults])
    print([mwb_psac, mwb_sac, mwb_adults])
    plt.title('Mean worm burden per age group, S.' + infection)
    plt.ylabel('MWB')
    plt.xlabel('Age group')
    plt.show()


# Mean worm burden age profile - every age - at the end of the simulation
def get_mwb_per_age(infection, age):
    if infection == 'Haematobium':
        prefix = 'sh'
    else:
        prefix = 'sm'
    mean = df[df['age_years'] == age][prefix + '_aggregate_worm_burden'].values.mean()
    return mean


def plot_mwb_age_profile(infection):
    ages = np.arange(0, 80, 1).tolist()
    mean_wb = [get_mwb_per_age(infection, x) for x in ages]
    plt.scatter(ages, mean_wb)
    plt.xlabel('age')
    plt.ylabel('mean worm burden')
    plt.title('Mean Worm Burden age profile, S.' + infection)
    plt.show()


plot_mwb_monthly(get_logger('Haematobium'), 'Haematobium')
plot_worm_burden_histogram('Haematobium')
plot_harbouring_rates('Haematobium')
plot_mwb_bar_plots('Haematobium')
plot_mwb_age_profile('Haematobium')


# Prevalent years
def plot_prevalent_years():
    loger_PrevalentYears_All = output['tlo.methods.schisto']['PrevalentYears_All']
    loger_PrevalentYears_All.date = pd.to_datetime(loger_PrevalentYears_All.date)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(loger_PrevalentYears_All.date, loger_PrevalentYears_All.Prevalent_years_this_year_total, label='All')
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=60))
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    plt.xticks(rotation='vertical')
    plt.legend()
    plt.title('Prevalent years per year')
    plt.ylabel('Years infected')
    plt.xlabel('logging date')
    plt.show()


def plot_high_inf_years():
    loger_High_inf_years = output['tlo.methods.schisto']['PrevalentYears_All']
    loger_High_inf_years.date = pd.to_datetime(loger_High_inf_years.date)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(loger_High_inf_years.date, loger_High_inf_years.High_infection_years_this_year_total, label='All')
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=60))
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    plt.xticks(rotation='vertical')
    plt.legend()
    plt.title('High-infection years per year')
    plt.ylabel('Years infected')
    plt.xlabel('logging date')
    plt.show()


plot_prevalent_years()
plot_high_inf_years()


# DALYS
loger_daly = output['tlo.methods.healthburden']["DALYS"]
loger_daly.drop(columns=['sex', 'YLL_Demography_Other'], inplace=True)
# this adds M and F
loger_daly = loger_daly.groupby(['date', 'age_range'],
                                as_index=False)['YLD_Schisto_Haematobium_Schisto_Haematobium_Symptoms'].sum()
age_map = {'0-4': 'PSAC', '5-9': 'SAC', '10-14': 'SAC'}
loger_daly['age_group'] = loger_daly['age_range'].map(age_map)
loger_daly.fillna('Adults', inplace=True)  # the reminder will be Adults
loger_daly.drop(columns=['age_range'], inplace=True)
loger_daly = loger_daly.groupby(['date', 'age_range'],
                                as_index=False)['YLD_Schisto_Haematobium_Schisto_Haematobium_Symptoms'].sum()
plt.scatter(loger_daly.date[loger_daly.age_group == 'Adults'],
            loger_daly.YLD_Schisto_Haematobium_Schisto_Haematobium_Symptoms[loger_daly.age_group == 'Adults'],
            label='Adults')
plt.scatter(loger_daly.date[loger_daly.age_group == 'PSAC'],
            loger_daly.YLD_Schisto_Haematobium_Schisto_Haematobium_Symptoms[loger_daly.age_group == 'PSAC'],
            label='PSAC')
plt.scatter(loger_daly.date[loger_daly.age_group == 'SAC'],
            loger_daly.YLD_Schisto_Haematobium_Schisto_Haematobium_Symptoms[loger_daly.age_group == 'SAC'],
            label='SAC')
plt.xticks(rotation='vertical')
plt.legend()
plt.title('DALYs due to schistosomiasis')
plt.ylabel('DALYs')
plt.xlabel('logging date')
plt.show()


def count_district_states(df, district):
    df_distr = df[((df.is_alive) & (df['district_of_residence'] == district))].copy()

    count_states = {'Non-infected': 0, 'Low-infection': 0, 'High-infection': 0}
    count_states.update(df_distr['sh_infection_status'].value_counts().to_dict())
    count_states.update({'infected_any': count_states['Low-infection'] + count_states['High-infection']})
    count_states.update({'total_pop_alive': count_states['infected_any'] + count_states['Non-infected']})
    if count_states['total_pop_alive'] != 0.0:
        count_states.update({'Prevalence': count_states['infected_any'] / count_states['total_pop_alive']})
    else:
        count_states.update({'Prevalence': 0.0})
    return count_states['Prevalence']


districts_prevalence = {}
for district in ['Blantyre', 'Chiradzulu', 'Mulanje', 'Nsanje', 'Nkhotakota', 'Phalombe']:
    districts_prevalence.update({district: count_district_states(df, district)})
