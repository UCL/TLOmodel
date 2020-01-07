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

def run_simulation(alpha=None, r0=None, popsize=10000):
    outputpath = ""
    datestamp = datetime.datetime.now().strftime("__%Y_%m_%d_%H_%M")

    # The resource files
    resourcefilepath = Path("./resources")
    start_date = Date(2005, 1, 1)
    end_date = Date(2025, 1, 1)
    popsize = popsize

    # Establish the simulation object
    sim = Simulation(start_date=start_date)

    # Establish the logger
    logfile = outputpath + "LogFile" + datestamp + ".log"

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
    sim.register(schisto.Schisto(resourcefilepath=resourcefilepath, alpha=alpha, r0=r0))

    # Run the simulation and flush the logger
    sim.seed_rngs(0)
    # initialise the population
    sim.make_initial_population(n=popsize)

    # # start the simulation
    sim.simulate(end_date=end_date)
    fh.flush()
    output = parse_log_file(logfile)
    return sim, output

sim, output = run_simulation(popsize=10000)

# ---------------------------------------------------------------------------------------------------------
#   save the district prevalence
df = sim.population.props
def get_prev_per_districts(output, df):
    districts = list(df.district_of_residence.unique())
    districts_prevalence = {}
    for distr in districts:
        prev = output['tlo.methods.schisto'][distr].Prevalence.values[-1]
        districts_prevalence.update({distr: prev})
    return districts_prevalence

# ---------------------------------------------------------------------------------------------------------
#   Check the prevalence of the symptoms
df = sim.population.props
params = sim.modules['Schisto'].parameters
symptoms_params = params['symptoms_haematobium'].set_index('symptoms').to_dict()['prevalence']

tot_pop_alive = len(df.index[df.is_alive])
huge_list = df['ss_haematobium_specific_symptoms'].dropna().tolist()
huge_list = [item for sublist in huge_list for item in sublist]
symptoms_prevalence = [[x, huge_list.count(x)] for x in set(huge_list)]
symptoms_prevalence = dict(symptoms_prevalence)
for s in symptoms_prevalence.keys():
    print(s + ", prevalence = " + str(round(symptoms_prevalence[s] / tot_pop_alive, 3)), ", expected = " + str(symptoms_params[s]))

# get prevalence per age_years
df_pa = df[['age_years', 'ss_infection_status']][df.is_alive]
age_groups_count = df_pa['age_years'].value_counts().to_dict()
infected_age_count = df_pa[df['ss_infection_status'] != 'Non-infected']['age_years'].value_counts().to_dict()
age_prev = {}
for k in age_groups_count.keys():
    if k in infected_age_count.keys():
        prev = infected_age_count[k] / age_groups_count[k]
    else:
        prev = 0
    age_prev.update({k: prev})

lists = sorted(age_prev.items())  # sorted by key, return a list of tuples
x, y = zip(*lists)  # unpack a list of pairs into two tuples
plt.plot(x[0:80], y[0:80])
plt.xlabel('Age')
plt.ylabel('Prevalence')
plt.ylim([0, 1])
plt.title('Final prevalence per age')
plt.show()

# ---------------------------------------------------------------------------------------------------------
#   Saving the results - prevalence, dalys and parameters used
# ---------------------------------------------------------------------------------------------------------
# prevalence and states count
output_path = 'C:/Users/ieh19/Desktop/Project 1/model_outputs/'
timestamp = str(datetime.datetime.now().replace(microsecond=0))
timestamp = timestamp.replace(" ", "_")
timestamp = timestamp.replace(":", "-")
savepath = output_path + "output_" + timestamp + ".csv"
savepath_full_pop = output_path + "output_population_" + timestamp + ".csv"
savepath_daly = output_path + "output_daly_" + timestamp + ".csv"
savepath_params = output_path + "input_" + timestamp + ".xlsx"

sim.population.props.to_csv(savepath_full_pop, index=False)

output_states = pd.DataFrame([])
for age_group in ['PSAC', 'SAC', 'Adults', 'All']:
    output['tlo.methods.schisto'][age_group]['Age_group'] = age_group
    output_states = output_states.append(output['tlo.methods.schisto'][age_group], ignore_index=True)
output_states.to_csv(savepath, index=False)

# dalys calculated by a dedicated schisto-module functionality
def calculate_yearly_dalys(df):
    df['DALY_monthly'] = df['DALY_cumulative'] - df['DALY_cumulative'].shift(1)
    df.loc[0, 'DALY_monthly'] = df.loc[0, 'DALY_cumulative']
    return df

dalys_output = calculate_yearly_dalys(output['tlo.methods.schisto']['DALY_All'])
dalys_output['Age_group'] = 'All'
dalys_output.to_csv(savepath_daly, index=False)

# parameters spreadsheet
parameters_used = pd.read_excel(Path("./resources/ResourceFile_Schisto.xlsx"), sheet_name=None)
writer = pd.ExcelWriter(savepath_params)
for sheet_name in parameters_used.keys():
    parameters_used[sheet_name].to_excel(writer, sheet_name=sheet_name)
writer.save()

# ---------------------------------------------------------------------------------------------------------
#   INSPECTING & PLOTTING
# ---------------------------------------------------------------------------------------------------------
df = sim.population.props
df = df[df['is_alive']]
params = sim.modules['Schisto'].parameters
loger_PSAC = output['tlo.methods.schisto']['PSAC']
loger_SAC = output['tlo.methods.schisto']['SAC']
loger_Adults = output['tlo.methods.schisto']['Adults']
loger_All = output['tlo.methods.schisto']['All']

# loger_PSAC.date = pd.to_datetime(loger_PSAC.date)
# loger_SAC.date = pd.to_datetime(loger_SAC.date)
# loger_Adults.date = pd.to_datetime(loger_Adults.date)
# loger_All.date = pd.to_datetime(loger_All.date)
#
# loger_PSAC = loger_PSAC[loger_PSAC.date >= pd.Timestamp(datetime.date(2014, 1, 1))]
# loger_SAC = loger_SAC[loger_SAC.date >= pd.Timestamp(datetime.date(2014, 1, 1))]
# loger_Adults = loger_Adults[loger_Adults.date >= pd.Timestamp(datetime.date(2014, 1, 1))]
# loger_All = loger_All[loger_All.date >= pd.Timestamp(datetime.date(2014, 1, 1))]

# Prevalence
plt.plot(loger_Adults.date, loger_Adults.Prevalence, label='Adults')
plt.plot(loger_PSAC.date, loger_PSAC.Prevalence, label='PSAC')
plt.plot(loger_SAC.date, loger_SAC.Prevalence, label='SAC')
plt.plot(loger_All.date, loger_All.Prevalence, label='All')
plt.xticks(rotation='vertical')
# plt.xticks.set_major_formatter(DateFormatter('%m-%Y'))
plt.ylim([0, 1])
plt.legend()
plt.title('Prevalence per date')
plt.ylabel('fraction of infected sub-population')
plt.xlabel('logging date')
plt.show()

# Final prevalence for every district
districts = list(df.district_of_residence.unique())
districts.sort()
districts_prevalence = {}
districts_mwb = {}
districts_high_inf_prev = {}
for distr in districts:
    prev = output['tlo.methods.schisto'][distr].Prevalence.values[-1]
    mwb = output['tlo.methods.schisto'][distr].MeanWormBurden.values[-1]
    high_inf = output['tlo.methods.schisto'][distr]['High_infections'].values[-1]
    low_inf = output['tlo.methods.schisto'][distr]['Low_infections'].values[-1]
    non_inf = output['tlo.methods.schisto'][distr]['Non_infected'].values[-1]
    if low_inf != 0:
        high_inf_fraction = high_inf / (low_inf + high_inf)
    else:
        high_inf_fraction = 0
    high_inf_p = high_inf / (non_inf + low_inf + high_inf)
    districts_prevalence.update({distr: prev})
    districts_mwb.update({distr: mwb})
    districts_high_inf_prev.update({distr: high_inf_p})

expected_district_prevalence = params['prevalence_2010'].to_dict()
plt.bar(*zip(*districts_prevalence.items()), alpha=0.5, label='model')
plt.scatter(*zip(*expected_district_prevalence.items()), label='data')
plt.xticks(rotation=90)
plt.xlabel('District')
plt.ylabel('Prevalence')
plt.legend()
plt.title('Prevalence per district')
plt.show()

plt.bar(*zip(*districts_mwb.items()), alpha=0.5, label='model')
plt.xticks(rotation=90)
plt.xlabel('District')
plt.ylabel('Mean Worm Burden')
plt.legend()
plt.title('Mean Worm Burden per district')
plt.show()

plt.bar(*zip(*districts_high_inf_prev.items()), alpha=0.5, label='model')
plt.xticks(rotation=90)
plt.xlabel('District')
plt.ylabel('Prevalence')
plt.legend()
plt.title('Prevalence of high infections per district')
plt.show()

# Mean Worm Burden per month
plt.plot(loger_Adults.date, loger_Adults.MeanWormBurden, label='Adults')
plt.plot(loger_PSAC.date, loger_PSAC.MeanWormBurden, label='PSAC')
plt.plot(loger_SAC.date, loger_SAC.MeanWormBurden, label='SAC')
plt.plot(loger_All.date, loger_All.MeanWormBurden, label='All')
plt.xticks(rotation='vertical')
# plt.xticks.set_major_formatter(DateFormatter('%m-%Y'))
plt.legend()
plt.ylim([0,2])
plt.title('Mean Worm Burden per date')
plt.ylabel('Mean number of worms')
plt.xlabel('logging date')
plt.show()

# Worm burden distribution at the end of the simulation & fitted NegBin
wb = df.ss_aggregate_worm_burden.values
mu = wb.mean()
theta = wb.var()
r = (1 + mu) / (theta * mu * mu)
p = mu / theta
print('clumping param k=', r, ', mean = ', p)
negbin = np.random.negative_binomial(r, p, size=len(wb))
plt.hist(wb[wb < 100], bins=100, density=True, label='empirical')
plt.hist(negbin, bins=100, density=True, label='parametrical, k= ' + str(round(r,2)) , alpha=0.5)
plt.xlabel('Worm burden')
plt.ylabel('Count')
# plt.xlim([0, 200])
plt.legend()
plt.title('Aggregate worm burden distribution')
plt.show()

# Harbouring rates distributions
hr = df.ss_harbouring_rate.values
plt.hist(hr, bins=100)
plt.xlabel('Harbouring rates')
plt.ylabel('Count')
plt.title('Harbouring rates distribution')
plt.show()


# Mean worm burden per age group - bar plots - at the end of the simulation
age_map = {'0-4': 'PSAC', '5-9': 'SAC', '10-14': 'SAC'}
df['age_group'] = df['age_range'].map(age_map)
df['age_group'].fillna('Adults', inplace=True)  # the reminder will be Adults
mwb_adults = df[df['age_group'] == 'Adults']['ss_aggregate_worm_burden'].values.mean()
mwb_sac = df[df['age_group'] == 'SAC']['ss_aggregate_worm_burden'].values.mean()
mwb_psac = df[df['age_group'] == 'PSAC']['ss_aggregate_worm_burden'].values.mean()
plt.bar(x=['PSAC', 'SAC', 'Adults'], height=[mwb_psac, mwb_sac, mwb_adults])
plt.title('Mean worm burden per age group')
plt.ylabel('MWB')
plt.xlabel('Age group')
plt.show()

# Mean worm burden age profile - every age - at the end of the simulation
def get_mwb_per_age(age):
    mean = df[df['age_years'] == age]['ss_aggregate_worm_burden'].values.mean()
    return mean

ages = np.arange(0, 80, 1).tolist()
mean_wb = [get_mwb_per_age(x) for x in ages]
plt.scatter(ages, mean_wb)
plt.xlabel('age')
plt.ylabel('mean worm burden')
plt.show()

# Prevalent years
loger_PrevalentYears_All = output['tlo.methods.schisto']['PrevalentYears_All']
plt.plot(loger_PrevalentYears_All.date, loger_PrevalentYears_All.Prevalent_years_this_year, label='All')
plt.xticks(rotation='vertical')
plt.legend()
plt.title('Prevalent years per year')
plt.ylabel('Years infected')
plt.xlabel('logging date')
plt.show()

# My own DALYS
loger_DALY_All = output['tlo.methods.schisto']['DALY_All']
plt.plot(loger_DALY_All.date, loger_DALY_All.DALY_this_year, label='All')
plt.xticks(rotation='vertical')
plt.legend()
plt.title('DALYs per year, schisto module calculation')
plt.ylabel('DALYs')
plt.xlabel('logging date')
plt.show()



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


