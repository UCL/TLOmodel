"""
A helper script to see the numbers of women of reproductive age having female sterilisation per 5-years age categories +
total, and the number of all women in the population in 2010 and 2020, to help to calibrate the intervention multipliers
(saved in ResourceFile_Contraception.xlsx in the sheets Interventions_Pop & Interventions_PPFP).
"""
from pathlib import Path

import pandas as pd
from matplotlib import dates as mdates

from tlo.analysis.utils import parse_log_file

# ### TO SET #################################################################################################
datestamp_without_log = '2023-04-26T142627'
logFile_without_interv = 'run_analysis_contraception_no_diseases__' + datestamp_without_log + '.log'
# this is for the runs done by the run_analysis_contraception_no_diseases.py (located in
# src/scripts/contraception/scenarios)
##############################################################################################################


def fullprint(in_to_print):
    with pd.option_context('display.max_rows', None, 'display.max_columns',
                           None):
        print(in_to_print)


def allcolsprint(in_to_print):
    with pd.option_context('display.max_rows', 5, 'display.max_columns',
                           None):
        print(in_to_print)


# Where will outputs go - by default, wherever this script is run
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# Load without simulating again - parse the simulation logfile to get the
# output dataframes
log_df = parse_log_file('outputs/' + logFile_without_interv)

# %% Female sterilization over time by age groups
years = mdates.YearLocator()  # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')

# Load Use Results
co_use_df = log_df['tlo.methods.contraception']['contraception_use_summary_by_age']
co_use_df = co_use_df.set_index('date')
Model_Years = pd.to_datetime(co_use_df.index)
Model_total = co_use_df.loc[['2010-01-01', '2020-12-01']].sum(axis=1)

print("F. STERIL. USE:")
co_use_f_steril_2010_2020_df = co_use_df.loc[['2010-01-01', '2020-12-01'],
                                             ['co_contraception=female_sterilization|age_range=15-19',
                                              'co_contraception=female_sterilization|age_range=20-24',
                                              'co_contraception=female_sterilization|age_range=25-29',
                                              'co_contraception=female_sterilization|age_range=30-34',
                                              'co_contraception=female_sterilization|age_range=35-39',
                                              'co_contraception=female_sterilization|age_range=40-44',
                                              'co_contraception=female_sterilization|age_range=45-49']
                                             ].transpose()

co_use_f_steril_2010_2020_df = co_use_f_steril_2010_2020_df.set_axis(
    ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49'], axis=0
)
co_use_f_steril_2010_2020_SUM_df = pd.DataFrame([co_use_f_steril_2010_2020_df.sum(), Model_total],
                                                columns=list(co_use_f_steril_2010_2020_df.columns),
                                                index=['Total with f. steril.: 15-49', 'Total all women: 15-49'])
co_use_f_steril_2010_2020_df = pd.concat([co_use_f_steril_2010_2020_df, co_use_f_steril_2010_2020_SUM_df])

fullprint(co_use_f_steril_2010_2020_df)
print("\n")
