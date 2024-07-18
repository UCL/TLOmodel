"""
* Check key outputs for reporting in the calibration table of the write-up
* Produce representative plots for the default parameters

NB. To see larger effects
* Increase incidence of cancer (see tests)
* Increase symptom onset
* Increase progression rates (see tests)
"""

import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import math
from tlo import Simulation, logging, Date

from tlo.analysis.utils import make_age_grp_types, parse_log_file
from tlo.methods import (
    cervical_cancer,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
    epi,
    tb,
    hiv
)

# Where outputs will go
output_csv_file = Path("./outputs/output1_data.csv")
seed = 100

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

# Set parameters for the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2025, 1, 1)
popsize = 17000

def run_sim(service_availability):
    # Establish the simulation object and set the seed
    sim = Simulation(start_date=start_date, seed=0)
#     sim = Simulation(start_date=start_date, log_config={"filename": "logfile"})

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 cervical_cancer.CervicalCancer(resourcefilepath=resourcefilepath),
#                cc_test.CervicalCancer(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           disable=False,
                                           cons_availability='all'),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 epi.Epi(resourcefilepath=resourcefilepath),
                 tb.Tb(resourcefilepath=resourcefilepath, run_with_checks=False),
                 hiv.Hiv(resourcefilepath=resourcefilepath, run_with_checks=False)
                 )

    logfile = sim._configure_logging(filename="LogFile")

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)


output_csv_file = Path("./outputs/output1_data.csv")
if output_csv_file.exists():
    output_csv_file.unlink()

run_sim(service_availability=['*'])


scale_factor = 17000000 / popsize
print(scale_factor)


# plot number of deaths in past year
out_df = pd.read_csv(output_csv_file)
# out_df = pd.read_csv('C:/Users/User/PycharmProjects/TLOmodel/outputs/output_data.csv', encoding='ISO-8859-1')
out_df = out_df[['n_deaths_past_year', 'rounded_decimal_year']].dropna()
out_df = out_df[out_df['rounded_decimal_year'] >= 2011]
out_df['n_deaths_past_year'] = out_df['n_deaths_past_year'] * scale_factor
print(out_df)
plt.figure(figsize=(10, 6))
plt.plot(out_df['rounded_decimal_year'], out_df['n_deaths_past_year'], marker='o')
plt.title('Total deaths by Year')
plt.xlabel('Year')
plt.ylabel('Total deaths past year')
plt.grid(True)
plt.ylim(0, 10000)
plt.show()


# plot number of cc diagnoses in past year
out_df_4 = pd.read_csv(output_csv_file)
out_df_4 = out_df_4[['n_diagnosed_past_year', 'rounded_decimal_year']].dropna()
out_df_4 = out_df_4[out_df_4['rounded_decimal_year'] >= 2011]
out_df_4['n_diagnosed_past_year'] = out_df_4['n_diagnosed_past_year'] * scale_factor
print(out_df_4)
plt.figure(figsize=(10, 6))
plt.plot(out_df_4['rounded_decimal_year'], out_df_4['n_diagnosed_past_year'], marker='o')
plt.title('Total diagnosed per Year')
plt.xlabel('Year')
plt.ylabel('Total diagnosed per year')
plt.grid(True)
plt.ylim(0,10000)
plt.show()




# plot prevalence of each ce stage
out_df_2 = pd.read_csv(output_csv_file)
columns_to_calculate = ['total_none', 'total_hpv', 'total_cin1', 'total_cin2', 'total_cin3', 'total_stage1',
                        'total_stage2a', 'total_stage2b', 'total_stage3', 'total_stage4']
for column in columns_to_calculate:
    new_column_name = column.replace('total_', '')
    out_df_2[f'proportion_{new_column_name}'] = out_df_2[column] / out_df_2[columns_to_calculate].sum(axis=1)
print(out_df_2)
columns_to_plot = ['proportion_hpv', 'proportion_cin1', 'proportion_cin2', 'proportion_cin3',
                   'proportion_stage1', 'proportion_stage2a', 'proportion_stage2b', 'proportion_stage3',
                   'proportion_stage4']
plt.figure(figsize=(10, 6))
# Initialize the bottom of the stack
bottom = 0
for column in columns_to_plot:
    plt.fill_between(out_df_2['rounded_decimal_year'],
                     bottom,
                     bottom + out_df_2[column],
                     label=column,
                     alpha=0.7)
    bottom += out_df_2[column]
# plt.plot(out_df_2['rounded_decimal_year'], out_df_2['proportion_cin1'], marker='o')
plt.title('Proportion of women aged 15+ with HPV, CIN, cervical cancer')
plt.xlabel('Year')
plt.ylabel('Proportion')
plt.grid(True)
plt.legend(loc='upper right')
plt.ylim(0, 0.10)
plt.show()



# Proportion of people with cervical cancer who are HIV positive
out_df_3 = pd.read_csv(output_csv_file)
out_df_3 = out_df_3[['prop_cc_hiv', 'rounded_decimal_year']].dropna()
plt.figure(figsize=(10, 6))
plt.plot(out_df_3['rounded_decimal_year'], out_df_3['prop_cc_hiv'], marker='o')
plt.title('Proportion of people with cervical cancer who are HIV positive')
plt.xlabel('Year')
plt.ylabel('Proportion')
plt.grid(True)
plt.ylim(0, 1)
plt.show()

# log_config = {
#     "filename": "cervical_cancer_analysis",   # The name of the output file (a timestamp will be appended).
#     "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
#     "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
#         "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
#         "tlo.methods.cervical_cancer": logging.INFO,
#         "tlo.methods.healthsystem": logging.INFO,
#     }
# }



# plot number of women living with unsuppressed HIV
out_df = pd.read_csv(output_csv_file)
out_df = out_df[['n_women_hiv_unsuppressed', 'rounded_decimal_year']].dropna()
out_df = out_df[out_df['rounded_decimal_year'] >= 2011]
out_df['n_women_hiv_unsuppressed'] = out_df['n_women_hiv_unsuppressed'] * scale_factor
print(out_df)
plt.figure(figsize=(10, 6))
plt.plot(out_df['rounded_decimal_year'], out_df['n_women_hiv_unsuppressed'], marker='o')
plt.title('n_women_hiv_unsuppressed')
plt.xlabel('Year')
plt.ylabel('n_women_hiv_unsuppressed')
plt.grid(True)
plt.ylim(0, 300000)
plt.show()




# ---------------------------------------------------------------------------
# output_csv_file = Path("./outputs/output1_data.csv")
# if output_csv_file.exists():
#     output_csv_file.unlink()
#
# run_sim(service_availability=['*'])
#
#
# scale_factor = 17000000 / popsize
# print(scale_factor)
#
#
# # plot number of deaths in past year
# out_df = pd.read_csv(output_csv_file)
# out_df = out_df[['n_deaths_past_year', 'rounded_decimal_year']].dropna()
# out_df = out_df[out_df['rounded_decimal_year'] >= 2011]
# out_df['n_deaths_past_year'] = out_df['n_deaths_past_year'] * scale_factor
# print(out_df)
# plt.figure(figsize=(10, 6))
# plt.plot(out_df['rounded_decimal_year'], out_df['n_deaths_past_year'], marker='o')
# plt.title('Total deaths by Year')
# plt.xlabel('Year')
# plt.ylabel('Total deaths past year')
# plt.grid(True)
# plt.ylim(0, 10000)
# plt.show()
#
#
# # plot number of cc diagnoses in past year
# out_df_4 = pd.read_csv(output_csv_file)
# out_df_4 = out_df_4[['n_diagnosed_past_year', 'rounded_decimal_year']].dropna()
# out_df_4 = out_df_4[out_df_4['rounded_decimal_year'] >= 2011]
# out_df_4['n_diagnosed_past_year'] = out_df_4['n_diagnosed_past_year'] * scale_factor
# print(out_df_4)
# plt.figure(figsize=(10, 6))
# plt.plot(out_df_4['rounded_decimal_year'], out_df_4['n_diagnosed_past_year'], marker='o')
# plt.title('Total diagnosed per Year')
# plt.xlabel('Year')
# plt.ylabel('Total diagnosed per year')
# plt.grid(True)
# plt.ylim(0,10000)
# plt.show()
#
#
#
#
# # plot prevalence of each ce stage
# out_df_2 = pd.read_csv(output_csv_file)
# columns_to_calculate = ['total_none', 'total_hpv', 'total_cin1', 'total_cin2', 'total_cin3', 'total_stage1',
#                         'total_stage2a', 'total_stage2b', 'total_stage3', 'total_stage4']
# for column in columns_to_calculate:
#     new_column_name = column.replace('total_', '')
#     out_df_2[f'proportion_{new_column_name}'] = out_df_2[column] / out_df_2[columns_to_calculate].sum(axis=1)
# print(out_df_2)
# columns_to_plot = ['proportion_hpv', 'proportion_cin1', 'proportion_cin2', 'proportion_cin3',
#                    'proportion_stage1', 'proportion_stage2a', 'proportion_stage2b', 'proportion_stage3',
#                    'proportion_stage4']
# plt.figure(figsize=(10, 6))
# # Initialize the bottom of the stack
# bottom = 0
# for column in columns_to_plot:
#     plt.fill_between(out_df_2['rounded_decimal_year'],
#                      bottom,
#                      bottom + out_df_2[column],
#                      label=column,
#                      alpha=0.7)
#     bottom += out_df_2[column]
# # plt.plot(out_df_2['rounded_decimal_year'], out_df_2['proportion_cin1'], marker='o')
# plt.title('Proportion of women aged 15+ with HPV, CIN, cervical cancer')
# plt.xlabel('Year')
# plt.ylabel('Proportion')
# plt.grid(True)
# plt.legend(loc='upper right')
# plt.ylim(0, 0.10)
# plt.show()
#
#
#
# # Proportion of people with cervical cancer who are HIV positive
# out_df_3 = pd.read_csv(output_csv_file)
# out_df_3 = out_df_3[['prop_cc_hiv', 'rounded_decimal_year']].dropna()
# plt.figure(figsize=(10, 6))
# plt.plot(out_df_3['rounded_decimal_year'], out_df_3['prop_cc_hiv'], marker='o')
# plt.title('Proportion of people with cervical cancer who are HIV positive')
# plt.xlabel('Year')
# plt.ylabel('Proportion')
# plt.grid(True)
# plt.ylim(0, 1)
# plt.show()

# ---------------------------------------------------------------------------------------









"""

plt.figure(figsize=(10, 6))
plt.plot(out_df_2['rounded_decimal_year'], out_df_2['proportion_stage2a'], marker='o')
plt.title('Proportion of women age 15+ with stage2a cervical cancer')
plt.xlabel('Year')
plt.ylabel('Proportion of women age 15+ with stage2a cervical cancer')
plt.grid(True)
plt.ylim(0, 1)
plt.show()







# Use pandas to read the JSON lines file
output_df = pd.read_json(output_txt_file, lines=True)

# Preprocess data
output_df['rounded_decimal_year'] = pd.to_datetime(output_df['rounded_decimal_year']).dt.year
output_df['total_hpv'] = output_df['total_hpv'].fillna(0)  # Fill NaN values with 0

print(output_df['rounded_decimal_year'], output_df['total_hpv'])

"""

"""

# Group by calendar year and sum the 'total_hpv'
grouped_data = output_df.groupby('rounded_decimal_year')['total_hpv'].sum()

# Plot the data
plt.figure(figsize=(10, 6))

"""






"""

def get_summary_stats(logfile):
    output = parse_log_file(logfile)

    # 1) TOTAL COUNTS BY STAGE OVER TIME
    counts_by_stage = output['tlo.methods.cervical_cancer']['summary_stats']
    counts_by_stage['date'] = pd.to_datetime(counts_by_stage['date'])
    counts_by_stage = counts_by_stage.set_index('date', drop=True)

    # 2) NUMBERS UNDIAGNOSED-DIAGNOSED-TREATED-PALLIATIVE CARE OVER TIME (SUMMED ACROSS TYPES OF CANCER)
    def get_cols_excl_none(allcols, stub):
        # helper function to some columns with a certain prefix stub - excluding the 'none' columns (ie. those
        #  that do not have cancer)
        cols = allcols[allcols.str.startswith(stub)]
        cols_not_none = [s for s in cols if ("none" not in s)]
        return cols_not_none

    summary = {
        'total': counts_by_stage[get_cols_excl_none(counts_by_stage.columns, 'total_')].sum(axis=1),
        'udx': counts_by_stage[get_cols_excl_none(counts_by_stage.columns, 'undiagnosed_')].sum(axis=1),
        'dx': counts_by_stage[get_cols_excl_none(counts_by_stage.columns, 'diagnosed_')].sum(axis=1),
        'tr': counts_by_stage[get_cols_excl_none(counts_by_stage.columns, 'treatment_')].sum(axis=1),
        'pc': counts_by_stage[get_cols_excl_none(counts_by_stage.columns, 'palliative_')].sum(axis=1)
    }
    counts_by_cascade = pd.DataFrame(summary)

    # 3) DALYS wrt age (total over whole simulation)
    dalys = output['tlo.methods.healthburden']['dalys']
    dalys = dalys.groupby(by=['age_range']).sum()
    dalys.index = dalys.index.astype(make_age_grp_types())
    dalys = dalys.sort_index()

    # 4) DEATHS wrt age (total over whole simulation)
    deaths = output['tlo.methods.demography']['death']
    deaths['age_group'] = deaths['age'].map(demography.Demography(resourcefilepath=resourcefilepath).AGE_RANGE_LOOKUP)

    x = deaths.loc[deaths.cause == 'CervicalCancer'].copy()
    x['age_group'] = x['age_group'].astype(make_age_grp_types())
    cervical_cancer_deaths = x.groupby(by=['age_group']).size()

    # 5) Rates of diagnosis per year:
    counts_by_stage['year'] = counts_by_stage.index.year
    annual_count_of_dxtr = counts_by_stage.groupby(by='year')[['diagnosed_since_last_log',
                                                               'treated_since_last_log',
                                                               'palliative_since_last_log']].sum()

    return {
        'total_counts_by_stage_over_time': counts_by_stage,
        'counts_by_cascade': counts_by_cascade,
        'dalys': dalys,
        'deaths': deaths,
        'cervical_cancer_deaths': cervical_cancer_deaths,
        'annual_count_of_dxtr': annual_count_of_dxtr
    }


# %% Run the simulation with and without interventions being allowed

# With interventions:
logfile_with_healthsystem = run_sim(service_availability=['*'])
results_with_healthsystem = get_summary_stats(logfile_with_healthsystem)


# Without interventions:
# logfile_no_healthsystem = run_sim(service_availability=[])
# results_no_healthsystem = get_summary_stats(logfile_no_healthsystem)

# %% Produce Summary Graphs:



# Examine Counts by Stage Over Time
counts = results_no_healthsystem['total_counts_by_stage_over_time']
counts.plot(y=['total_stage1', 'total_stage2a', 'total_stage2b', 'total_stage3'])
plt.title('Count in Each Stage of Disease Over Time')
plt.xlabel('Time')
plt.ylabel('Count')
plt.show()



# Examine numbers in each stage of the cascade:
results_with_healthsystem['counts_by_cascade'].plot(y=['udx', 'dx', 'tr', 'pc'])
plt.title('With Health System')
plt.xlabel('Numbers of those With Cancer by Stage in Cascade')
plt.xlabel('Time')
plt.legend(['Undiagnosed', 'Diagnosed', 'Ever treated', 'On Palliative Care'])
plt.show()


results_no_healthsystem['counts_by_cascade'].plot(y=['udx', 'dx', 'tr', 'pc'])
plt.title('With No Health System')
plt.xlabel('Numbers of those With Cancer by Stage in Cascade')
plt.xlabel('Time')
plt.legend(['Undiagnosed', 'Diagnosed', 'On Treatment', 'On Palliative Care'])
plt.show()

# Examine DALYS (summed over whole simulation)
results_no_healthsystem['dalys'].plot.bar(
    y=['YLD_CervicalCancer_0', 'YLL_CervicalCancer_CervicalCancer'],
    stacked=True)
plt.xlabel('Age-group')
plt.ylabel('DALYS')
plt.legend()
plt.title("With No Health System")
plt.show()


# Examine Deaths (summed over whole simulation)
deaths = results_with_healthsystem['cervical_cancer_deaths']

print(deaths)

deaths.index = deaths.index.astype(make_age_grp_types())
# # make a series with the right categories and zero so formats nicely in the grapsh:
agegrps = demography.Demography(resourcefilepath=resourcefilepath).AGE_RANGE_CATEGORIES
totdeaths = pd.Series(index=agegrps, data=np.nan)
totdeaths.index = totdeaths.index.astype(make_age_grp_types())
totdeaths = totdeaths.combine_first(deaths).fillna(0.0)
totdeaths.plot.bar()
plt.title('Deaths due to Cervical Cancer')
plt.xlabel('Age-group')
plt.ylabel('Total Deaths During Simulation')
# plt.gca().get_legend().remove()
plt.show()


# Compare Deaths - with and without the healthsystem functioning - sum over age and time
deaths = {
    'No_HealthSystem': sum(results_no_healthsystem['cervical_cancer_deaths']),
    'With_HealthSystem': sum(results_with_healthsystem['cervical_cancer_deaths'])
}

plt.bar(range(len(deaths)), list(deaths.values()), align='center')
plt.xticks(range(len(deaths)), list(deaths.keys()))
plt.title('Deaths due to Cervical Cancer')
plt.xlabel('Scenario')
plt.ylabel('Total Deaths During Simulation')
plt.show()


# %% Get Statistics for Table in write-up (from results_with_healthsystem);

# ** Current prevalence (end-2019) of people who have diagnosed with cervical
# cancer in 2020 (total; and current stage 1, 2, 3, 4), per 100,000 population aged 20+

counts = results_with_healthsystem['total_counts_by_stage_over_time'][[
    'total_stage1',
    'total_stage2a',
    'total_stage2b',
    'total_stage3',
    'total_stage4'
]].iloc[-1]

totpopsize = results_with_healthsystem['total_counts_by_stage_over_time'][[
    'total_none',
    'total_stage1',
    'total_stage2a',
    'total_stage2b',
    'total_stage3',
    'total_stage4'
]].iloc[-1].sum()

prev_per_100k = 1e5 * counts.sum() / totpopsize

# ** Number of deaths from cervical cancer per year per 100,000 population.
# average deaths per year = deaths over ten years divided by ten, * 100k/population size
(results_with_healthsystem['cervical_cancer_deaths'].sum()/10) * 1e5/popsize

# ** Incidence rate of diagnosis, treatment, palliative care for cervical cancer (all stages combined),
# per 100,000 population
(results_with_healthsystem['annual_count_of_dxtr']).mean() * 1e5/popsize


# ** 5-year survival following treatment
# See separate file

"""
