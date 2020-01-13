import datetime
import logging
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.dates import DateFormatter

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    healthburden,
    healthsystem,
    contraception,
    schisto
)

popsize = 10000
alpha = None
r0 = None
def run_simulation(alpha=None, r0=None, popsize=10000, haem=True, mansoni=True):
    outputpath = Path("./outputs")  # folder for convenience of storing outputs
    datestamp = datetime.datetime.now().strftime("__%Y_%m_%d_%H_%M")

    # The resource files
    resourcefilepath = Path("./resources")
    start_date = Date(2010, 1, 1)
    end_date = Date(2025, 2, 1)
    popsize = popsize

    # Establish the simulation object
    sim = Simulation(start_date=start_date)

    # Establish the logger
    logfile = outputpath / ('LogFile' + datestamp + '.log')

    if os.path.exists(logfile):
        os.remove(logfile)
    fh = logging.FileHandler(logfile)
    fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
    fh.setFormatter(fr)
    logging.getLogger().addHandler(fh)


    logging.getLogger("tlo.methods.demography").setLevel(logging.WARNING)
    logging.getLogger("tlo.methods.contraception").setLevel(logging.WARNING)
    logging.getLogger("tlo.methods.healthburden").setLevel(logging.WARNING)
    logging.getLogger("tlo.methods.healthsystem").setLevel(logging.WARNING)
    logging.getLogger("tlo.methods.schisto").setLevel(logging.INFO)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(schisto.Schisto(resourcefilepath=resourcefilepath))
    if haem:
        sim.register(schisto.Schisto_Haematobium(resourcefilepath=resourcefilepath, alpha=alpha, r0=r0))
    if mansoni:
        sim.register(schisto.Schisto_Mansoni(resourcefilepath=resourcefilepath, alpha=alpha, r0=r0))

    # Run the simulation and flush the logger
    sim.seed_rngs(0)
    # initialise the population
    sim.make_initial_population(n=popsize)

    # # start the simulation
    sim.simulate(end_date=end_date)
    fh.flush()
    output = parse_log_file(logfile)
    return sim, output

sim, output = run_simulation(popsize=10000, haem=True, mansoni=True)

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
        output_states = output_states.append(output['tlo.methods.schisto'][age_group + '_' + infection], ignore_index=True)
    output_states.to_csv(savepath, index=False)

    if save_the_districts:
        df = sim.population.props
        df = df[df['is_alive']]

        districts = list(df.district_of_residence.unique())
        districts.sort()
        distr_end_data = pd.DataFrame(columns=['District', 'Prevalence', 'MWB', 'High-infection-prevalence'])
        for distr in districts:
            prev = output['tlo.methods.schisto'][distr + '_' + infection].Prevalence.values[-1]
            mwb = output['tlo.methods.schisto'][distr + '_' + infection].MeanWormBurden.values[-1]
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
    savepath = output_path + "output_Total" + '_' + timestamp + ".csv"
    savepath_full_pop = output_path + "output_population_" + timestamp + ".csv"
    savepath_prevalent_years = output_path + "output_prevalent_years_" + timestamp + ".csv"
    savepath_daly = output_path + "output_daly_" + timestamp + ".csv"
    savepath_params = output_path + "input_" + timestamp + ".xlsx"

    sim.population.props.to_csv(savepath_full_pop, index=False)

    output_states = pd.DataFrame([])
    for age_group in ['PSAC', 'SAC', 'Adults', 'All']:
        output['tlo.methods.schisto'][age_group + '_Total']['Age_group'] = age_group
        output_states = output_states.append(output['tlo.methods.schisto'][age_group + '_Total'], ignore_index=True)
    output_states.to_csv(savepath, index=False)

    # Prevalent years
    loger_PrevalentYears_All = output['tlo.methods.schisto']['PrevalentYears_All']
    loger_PrevalentYears_All.to_csv(savepath_prevalent_years, index=False)

    # My own DALYS
    loger_DALY_All = output['tlo.methods.schisto']['DALY_All']
    loger_DALY_All.to_csv(savepath_daly, index=False)

    # parameters spreadsheet
    parameters_used = pd.read_excel(Path("./resources/ResourceFile_Schisto.xlsx"), sheet_name=None)
    writer = pd.ExcelWriter(savepath_params)
    for sheet_name in parameters_used.keys():
        parameters_used[sheet_name].to_excel(writer, sheet_name=sheet_name)
    writer.save()

save_inf_outputs('Haematobium')
save_inf_outputs('Mansoni')
save_general_outputs_and_params()
# ---------------------------------------------------------------------------------------------------------
#   INSPECTING & PLOTTING
# ---------------------------------------------------------------------------------------------------------
# pzq pills used in 2018 for comparing
pzq_2018_haem = sim.modules['Schisto_Haematobium'].PZQ_per_district_used
pzq_2018_mans = sim.modules['Schisto_Mansoni'].PZQ_per_district_used

df = sim.population.props
df = df[df['is_alive']]

def get_logger(infection, cut_off_left=False):
    assert infection in ['Total', 'Haematobium', 'Mansoni'], 'Wrong infection type!'

    loger_PSAC = output['tlo.methods.schisto']['PSAC' + '_' + infection]
    loger_SAC = output['tlo.methods.schisto']['SAC' + '_' + infection]
    loger_Adults = output['tlo.methods.schisto']['Adults' + '_' + infection]
    loger_All = output['tlo.methods.schisto']['All' + '_' + infection]

    if cut_off_left:
        loger_PSAC.date = pd.to_datetime(loger_PSAC.date)
        loger_SAC.date = pd.to_datetime(loger_SAC.date)
        loger_Adults.date = pd.to_datetime(loger_Adults.date)
        loger_All.date = pd.to_datetime(loger_All.date)

        loger_PSAC = loger_PSAC[loger_PSAC.date >= pd.Timestamp(datetime.date(2014, 1, 1))]
        loger_SAC = loger_SAC[loger_SAC.date >= pd.Timestamp(datetime.date(2014, 1, 1))]
        loger_Adults = loger_Adults[loger_Adults.date >= pd.Timestamp(datetime.date(2014, 1, 1))]
        loger_All = loger_All[loger_All.date >= pd.Timestamp(datetime.date(2014, 1, 1))]

    log_dict = {'loger_PSAC': loger_PSAC, 'loger_SAC': loger_SAC,
                'loger_Adults': loger_Adults, 'loger_All': loger_All}
    return log_dict

# Prevalence
def plot_prevalence(loger_inf, infection):
    for log in loger_inf.keys():
        plt.plot(loger_inf[log].date, loger_inf[log].Prevalence, label=log[6:])
    plt.xticks(rotation='vertical')
    # plt.xticks.set_major_formatter(DateFormatter('%m-%Y'))
    plt.ylim([0, 1])
    plt.legend()
    plt.title('Prevalence per date, S.' + infection)
    plt.ylabel('fraction of infected sub-population')
    plt.xlabel('logging date')
    plt.show()

plot_prevalence(get_logger('Mansoni'), 'Mansoni')
plot_prevalence(get_logger('Haematobium'), 'Haematobium')
plot_prevalence(get_logger('Total'), 'Total')

# Final prevalence for every district
def get_values_per_district(infection):
    districts = list(df.district_of_residence.unique())
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

districts_prevalence, districts_mwb, districts_high_inf_prev = get_values_per_district('Mansoni')
expected_district_prevalence = get_expected_prevalence('Mansoni')

plot_prevalence_per_district('Mansoni', districts_prevalence, expected_district_prevalence)
plot_mwb_per_district('Mansoni', districts_mwb)
plot_prev_high_infection_per_district('Mansoni', districts_high_inf_prev)

# Mean Worm Burden per month
def plot_mwb_monthly(loger_inf, infection):
    for log in loger_inf.keys():
        plt.plot(loger_inf[log].date, loger_inf[log].Prevalence, label=log[6:])
    plt.xticks(rotation='vertical')
    # plt.xticks.set_major_formatter(DateFormatter('%m-%Y'))
    plt.legend()
    plt.ylim([0, 1])
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
    mu = wb.mean()
    theta = wb.var()
    r = (1 + mu) / (theta * mu * mu)
    p = mu / theta
    print('clumping param k=', r, ', mean = ', p)
    negbin = np.random.negative_binomial(r, p, size=len(wb))
    plt.hist(wb[wb < 100], bins=100, density=True, label='empirical')
    plt.hist(negbin, bins=100, density=True, label='parametrical, k= ' + str(round(r, 2)), alpha=0.5)
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
    plt.hist(hr, bins=100)
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

plot_mwb_monthly(get_logger('Mansoni'), 'Mansoni')
plot_mwb_monthly(get_logger('Haematobium'), 'Haematobium')

plot_worm_burden_histogram('Haematobium')
plot_worm_burden_histogram('Mansoni')

plot_harbouring_rates('Haematobium')
plot_harbouring_rates('Mansoni')

plot_mwb_bar_plots('Haematobium')
plot_mwb_age_profile('Haematobium')

plot_mwb_bar_plots('Mansoni')
plot_mwb_age_profile('Mansoni')


# Prevalent years
def plot_prevalent_years():
    loger_PrevalentYears_All = output['tlo.methods.schisto']['PrevalentYears_All']
    plt.plot(loger_PrevalentYears_All.date, loger_PrevalentYears_All.Prevalent_years_this_year_total, label='All')
    plt.xticks(rotation='vertical')
    plt.legend()
    plt.title('Prevalent years per year')
    plt.ylabel('Years infected')
    plt.xlabel('logging date')
    plt.show()

# My own DALYS
def plot_DALYs():
    loger_DALY_All = output['tlo.methods.schisto']['DALY_All']
    plt.plot(loger_DALY_All.date, loger_DALY_All.DALY_this_year_total, label='All')
    plt.xticks(rotation='vertical')
    plt.legend()
    plt.title('DALYs per year, schisto module calculation')
    plt.ylabel('DALYs')
    plt.xlabel('logging date')
    plt.show()

plot_prevalent_years()
plot_DALYs()


# # DALYS
# loger_daly = output['tlo.methods.healthburden']["DALYS"]
# loger_daly.drop(columns=['sex', 'YLL_Demography_Other'], inplace=True)
# loger_daly = loger_daly.groupby(['date', 'age_range'], as_index=False)['YLD_Schisto_Schisto_Symptoms'].sum() # this adds M and F
# age_map = {'0-4': 'PSAC', '5-9': 'SAC', '10-14': 'SAC'}
# loger_daly['age_group'] = loger_daly['age_range'].map(age_map)
# loger_daly.fillna('Adults', inplace=True)  # the reminder will be Adults
# loger_daly.drop(columns=['age_range'], inplace=True)
# loger_daly = loger_daly.groupby(['date', 'age_group'], as_index=False)['YLD_Schisto_Schisto_Symptoms'].sum() # this adds M and F
#
# plt.scatter(loger_daly.date[loger_daly.age_group == 'Adults'], loger_daly.YLD_Schisto_Schisto_Symptoms[loger_daly.age_group == 'Adults'], label='Adults')
# plt.scatter(loger_daly.date[loger_daly.age_group == 'PSAC'], loger_daly.YLD_Schisto_Schisto_Symptoms[loger_daly.age_group == 'PSAC'], label='PSAC')
# plt.scatter(loger_daly.date[loger_daly.age_group == 'SAC'], loger_daly.YLD_Schisto_Schisto_Symptoms[loger_daly.age_group == 'SAC'], label='SAC')
# plt.xticks(rotation='vertical')
# plt.legend()
# plt.title('DALYs due to schistosomiasis')
# plt.ylabel('DALYs')
# plt.xlabel('logging date')
# plt.show()

# # ---------------------------------------------------------------------------------------------------------
# #   Check the prevalence of the symptoms
# df = sim.population.props
# params = sim.modules['Schisto'].parameters
# symptoms_params = params['symptoms_haematobium'].set_index('symptoms').to_dict()['prevalence']
#
# tot_pop_alive = len(df.index[df.is_alive])
# huge_list = df['ss_haematobium_specific_symptoms'].dropna().tolist()
# huge_list = [item for sublist in huge_list for item in sublist]
# symptoms_prevalence = [[x, huge_list.count(x)] for x in set(huge_list)]
# symptoms_prevalence = dict(symptoms_prevalence)
# for s in symptoms_prevalence.keys():
#     print(s + ", prevalence = " + str(round(symptoms_prevalence[s] / tot_pop_alive, 3)), ", expected = " + str(symptoms_params[s]))
#
# # get prevalence per age_years
# df_pa = df[['age_years', 'ss_infection_status']][df.is_alive]
# age_groups_count = df_pa['age_years'].value_counts().to_dict()
# infected_age_count = df_pa[df['ss_infection_status'] != 'Non-infected']['age_years'].value_counts().to_dict()
# age_prev = {}
# for k in age_groups_count.keys():
#     if k in infected_age_count.keys():
#         prev = infected_age_count[k] / age_groups_count[k]
#     else:
#         prev = 0
#     age_prev.update({k: prev})
#
# lists = sorted(age_prev.items())  # sorted by key, return a list of tuples
# x, y = zip(*lists)  # unpack a list of pairs into two tuples
# plt.plot(x[0:80], y[0:80])
# plt.xlabel('Age')
# plt.ylabel('Prevalence')
# plt.ylim([0, 1])
# plt.title('Final prevalence per age')
# plt.show()

