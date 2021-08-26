"""
define a function which reads in relevant data and returns a calibration score
calculated as the mean deviance of the model outputs and the reported data
data sources are weighted according to their reliability / robustness,
e.g. data considered more reliable than other model outputs which have large uncertainty
the score represents the average amount by which the model outputs differ proportionately from the observed data;
e.g. a score of 0.25 means that on average the model outputs are 25% away from the observed data.
The lower the calibration score the more closely the model outputs can re-produce observed data estimates
source for method: Bansi-Matharu JIAS 2018
https://onlinelibrary.wiley.com/doi/full/10.1002/jia2.25205
"""

import matplotlib.pyplot as plt
import pandas as pd
import datetime
import pickle
from pathlib import Path
import math

# declare the paths
resourcefilepath = Path("./resources")
outputpath = Path("./outputs")  # folder for convenience of storing outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# -------------------------------- DATA -------------------------------- #

# make a dict of all data to be used in calculating calibration score
data_dict = {}

## HIV
# read in resource files for data
xls = pd.ExcelFile(resourcefilepath / 'ResourceFile_HIV.xlsx')

# MPHIA HIV data - age-structured
data_hiv_mphia_inc = pd.read_excel(xls, sheet_name='MPHIA_incidence2015')
data_hiv_mphia_prev = pd.read_excel(xls, sheet_name='MPHIA_prevalence_art2015')

data_dict['mphia_inc_2015'] = data_hiv_mphia_inc.loc[
    (data_hiv_mphia_inc.age == "15-49"), "total_percent_annual_incidence"].values[0]  # inc
data_dict['mphia_prev_2015'] = data_hiv_mphia_prev.loc[
    data_hiv_mphia_prev.age == "Total 15-49", "total percent hiv positive"].values[0]  # prev

# DHS HIV data
data_hiv_dhs_prev = pd.read_excel(xls, sheet_name='DHS_prevalence')
data_dict['dhs_prev_2010'] = data_hiv_dhs_prev.loc[
    (data_hiv_dhs_prev.Year == 2010), "HIV prevalence among general population 15-49"].values[0]
data_dict['dhs_prev_2015'] = data_hiv_dhs_prev.loc[
    (data_hiv_dhs_prev.Year == 2015), "HIV prevalence among general population 15-49"].values[0]

# UNAIDS AIDS deaths data: 2010-
data_hiv_unaids_deaths = pd.read_excel(xls, sheet_name='unaids_mortality_dalys2021')
data_dict['unaids_deaths_per_1000'] = data_hiv_unaids_deaths['AIDS_mortality_per_1000']

## TB
# TB WHO data: 2010-
xls_tb = pd.ExcelFile(resourcefilepath / 'ResourceFile_TB.xlsx')

# TB active incidence per 100k 2010-2017
data_tb_who = pd.read_excel(xls_tb, sheet_name='WHO_activeTB2020')
data_dict['who_tb_inc_per_100k'] = data_tb_who.loc[(data_tb_who.year >= 2010), 'incidence_per_100k']


# TB latent data (Houben & Dodd 2016)
data_tb_latent = pd.read_excel(xls_tb, sheet_name='latent_TB2014_summary')
data_tb_latent_all_ages = data_tb_latent.loc[data_tb_latent.Age_group == '0_80']
data_dict['who_tb_latent_prev'] = data_tb_latent_all_ages.proportion_latent_TB.values[0]

# TB case notification rate NTP: 2012-2019
data_tb_ntp = pd.read_excel(xls_tb, sheet_name='NTP2019')
data_dict['ntp_case_notification_per_100k'] = data_tb_ntp.loc[(data_tb_ntp.year >= 2012), 'case_notification_rate_per_100k']

# TB mortality per 100k excluding HIV: 2010-2017
data_dict['who_tb_deaths_per_100k'] = data_tb_who.loc[(data_tb_who.year >= 2010), 'mortality_tb_excl_hiv_per_100k']

# need weights for each data item
model_weight = 0.5


# -------------------------------- CALIBRATION SCORE -------------------------------- #

# function which returns weighted mean of model outputs compared with data
# requires model outputs as inputs
def weighted_mean(model_dict, data_dict):
    # assert model_output is not empty

    # return calibration score (weighted mean deviance)
    # sqrt( (observed data – model output)^2 | / observed data)
    # sum these for each data item (all male prevalence over time) and divide by # items
    # then weighted sum of all components -> calibration score

    def deviance_function(data, model):
        deviance = math.sqrt(((data - model) ** 2) / data)

        return deviance

    # hiv prevalence in adults 15-49: dhs 2010, 2015
    hiv_prev_dhs = (
                       deviance_function(data_dict['dhs_prev_2010'], model_dict['hiv_prev_adult_2010']) +
                       deviance_function(data_dict['dhs_prev_2015'], model_dict['hiv_prev_adult_2015'])
                   ) / 2

    # hiv prevalence mphia
    hiv_prev_mphia = deviance_function(data_dict['mphia_prev_2015'], model_dict['hiv_prev_adult_2015'])

    # hiv incidence mphia
    hiv_inc_mphia = deviance_function(data_dict['mphia_inc_2015'], model_dict['hiv_inc_adult_2015'])

    # aids deaths unaids 2010-2019
    hiv_deaths_unaids = (
        deviance_function(data_dict['unaids_deaths_per_1000'][0], model_dict['AIDS_mortality_per_1000'][0]) +
        deviance_function(data_dict['unaids_deaths_per_1000'][1], model_dict['AIDS_mortality_per_1000'][1]) +
        deviance_function(data_dict['unaids_deaths_per_1000'][2], model_dict['AIDS_mortality_per_1000'][2]) +
        deviance_function(data_dict['unaids_deaths_per_1000'][3], model_dict['AIDS_mortality_per_1000'][3]) +
        deviance_function(data_dict['unaids_deaths_per_1000'][4], model_dict['AIDS_mortality_per_1000'][4]) +
        deviance_function(data_dict['unaids_deaths_per_1000'][5], model_dict['AIDS_mortality_per_1000'][5]) +
        deviance_function(data_dict['unaids_deaths_per_1000'][6], model_dict['AIDS_mortality_per_1000'][6]) +
        deviance_function(data_dict['unaids_deaths_per_1000'][7], model_dict['AIDS_mortality_per_1000'][7]) +
        deviance_function(data_dict['unaids_deaths_per_1000'][8], model_dict['AIDS_mortality_per_1000'][8]) +
        deviance_function(data_dict['unaids_deaths_per_1000'][9], model_dict['AIDS_mortality_per_1000'][9])
    ) / 10

    # tb active incidence (WHO estimates) 2010 -2017
    tb_incidence_who = (
        deviance_function(data_dict['who_tb_inc_per_100k'][0], model_dict['TB_active_inc_per100k'][0]) +
        deviance_function(data_dict['who_tb_inc_per_100k'][1], model_dict['TB_active_inc_per100k'][1]) +
        deviance_function(data_dict['who_tb_inc_per_100k'][2], model_dict['TB_active_inc_per100k'][2]) +
        deviance_function(data_dict['who_tb_inc_per_100k'][3], model_dict['TB_active_inc_per100k'][3]) +
        deviance_function(data_dict['who_tb_inc_per_100k'][4], model_dict['TB_active_inc_per100k'][4]) +
        deviance_function(data_dict['who_tb_inc_per_100k'][5], model_dict['TB_active_inc_per100k'][5]) +
        deviance_function(data_dict['who_tb_inc_per_100k'][6], model_dict['TB_active_inc_per100k'][6]) +
        deviance_function(data_dict['who_tb_inc_per_100k'][7], model_dict['TB_active_inc_per100k'][7])
    ) / 8

    # tb latent prevalence
    tb_latent_prev = deviance_function(data_dict['who_tb_latent_prev'], model_dict['TB_latent_prev'])

    # tb case notification rate per 100k: 2012-2019
    tb_cnr_ntp = (
        deviance_function(data_dict['ntp_case_notification_per_100k'][0], model_dict['TB_case_notifications_per100k'][2]) +
        deviance_function(data_dict['ntp_case_notification_per_100k'][1], model_dict['TB_case_notifications_per100k'][3]) +
        deviance_function(data_dict['ntp_case_notification_per_100k'][2], model_dict['TB_case_notifications_per100k'][4]) +
        deviance_function(data_dict['ntp_case_notification_per_100k'][3], model_dict['TB_case_notifications_per100k'][5]) +
        deviance_function(data_dict['ntp_case_notification_per_100k'][4], model_dict['TB_case_notifications_per100k'][6]) +
        deviance_function(data_dict['ntp_case_notification_per_100k'][5], model_dict['TB_case_notifications_per100k'][7]) +
        deviance_function(data_dict['ntp_case_notification_per_100k'][6], model_dict['TB_case_notifications_per100k'][8]) +
        deviance_function(data_dict['ntp_case_notification_per_100k'][7], model_dict['TB_case_notifications_per100k'][9])
    ) / 8

    # tb death rate who 2010-2017
    tb_mortality_who = (
        deviance_function(data_dict['who_tb_deaths_per_100k'][0], model_dict['TB_mortality_per_100k'][0]) +
        deviance_function(data_dict['who_tb_deaths_per_100k'][1], model_dict['TB_mortality_per_100k'][1]) +
        deviance_function(data_dict['who_tb_deaths_per_100k'][2], model_dict['TB_mortality_per_100k'][2]) +
        deviance_function(data_dict['who_tb_deaths_per_100k'][3], model_dict['TB_mortality_per_100k'][3]) +
        deviance_function(data_dict['who_tb_deaths_per_100k'][4], model_dict['TB_mortality_per_100k'][4]) +
        deviance_function(data_dict['who_tb_deaths_per_100k'][5], model_dict['TB_mortality_per_100k'][5]) +
        deviance_function(data_dict['who_tb_deaths_per_100k'][6], model_dict['TB_mortality_per_100k'][6]) +
        deviance_function(data_dict['who_tb_deaths_per_100k'][7], model_dict['TB_mortality_per_100k'][7])
    ) / 8

    calibration_score = hiv_prev_dhs + \
                        hiv_prev_mphia + \
                        hiv_inc_mphia + \
                        (hiv_deaths_unaids * model_weight) + \
                        tb_incidence_who + \
                        tb_latent_prev + \
                        tb_cnr_ntp + \
                        tb_mortality_who

    return calibration_score


# -------------------------------- MODEL OUTPUTS -------------------------------- #

# read in all output files
# todo this will be replaced with a read in loop for batch files

# load the results - desktop sample run
with open(outputpath / 'default_run.pickle', 'rb') as f:
    output = pickle.load(f)

# get logged outputs for calibration into dict
model_dict = {}

# HIV - prevalence among in adults aged 15-49
model_hiv_prev = output['tlo.methods.hiv'][
    'summary_inc_and_prev_for_adults_and_children_and_fsw']
model_dict['hiv_prev_adult_2010'] = (model_hiv_prev.loc[
    (model_hiv_prev.date == '2010-01-01'), 'hiv_prev_adult_1549'].values[0]) * 100
model_dict['hiv_prev_adult_2015'] = (model_hiv_prev.loc[
    (model_hiv_prev.date == '2015-01-01'), 'hiv_prev_adult_1549'].values[0]) * 100

# hiv incidence in adults aged 15-49
model_dict['hiv_inc_adult_2015'] = (model_hiv_prev.loc[
    (model_hiv_prev.date == '2015-01-01'), 'hiv_adult_inc_1549'].values[0]) * 100

# aids deaths
# deaths
deaths = output['tlo.methods.demography']['death'].copy()  # outputs individual deaths
deaths = deaths.set_index('date')

# AIDS DEATHS
# person-years all ages (irrespective of HIV status)
py_ = output['tlo.methods.demography']['person_years']
years = pd.to_datetime(py_['date']).dt.year
py = pd.Series(dtype='int64', index=years)
for year in years:
    tot_py = (
        (py_.loc[pd.to_datetime(py_['date']).dt.year == year]['M']).apply(pd.Series) +
        (py_.loc[pd.to_datetime(py_['date']).dt.year == year]['F']).apply(pd.Series)
    ).transpose()
    py[year] = tot_py.sum().values[0]

py.index = pd.to_datetime(years, format='%Y')


# limit to deaths among aged 15+, include HIV/TB deaths
keep = ((deaths.age >= 15) & ((deaths.cause == 'AIDS_TB') | (deaths.cause == 'AIDS_non_TB')))
deaths_AIDS = deaths.loc[keep].copy()
deaths_AIDS['year'] = deaths_AIDS.index.year
tot_aids_deaths = deaths_AIDS.groupby(by=['year']).size()
tot_aids_deaths.index = pd.to_datetime(tot_aids_deaths.index, format='%Y')

# aids mortality rates per 1000 person-years
model_dict['AIDS_mortality_per_1000'] = (tot_aids_deaths / py) * 1000

# tb active incidence per 100k - all ages
TB_inc = output['tlo.methods.tb']['tb_incidence']
TB_inc = TB_inc.set_index('date')
TB_inc.index = pd.to_datetime(TB_inc.index)
model_dict['TB_active_inc_per100k'] = (TB_inc['num_new_active_tb'] / py) * 100000

# tb latent prevalence
latentTB_prev = output['tlo.methods.tb']['tb_prevalence']
model_dict['TB_latent_prev'] = latentTB_prev.loc[['2014-01-01'], ['tbPrevLatent']].values[0][0]

# tb case notifications
tb_notifications = output['tlo.methods.tb']['tb_treatment']
tb_notifications = tb_notifications.set_index('date')
tb_notifications.index = pd.to_datetime(tb_notifications.index)
model_dict['TB_case_notifications_per100k'] = (tb_notifications['tbNewDiagnosis'] / py) * 100000

# tb deaths (non-hiv only)
keep = (deaths.cause == 'TB')
deaths_TB = deaths.loc[keep].copy()
deaths_TB['year'] = deaths_TB.index.year  # count by year
tot_tb_non_hiv_deaths = deaths_TB.groupby(by=['year']).size()
tot_tb_non_hiv_deaths.index = pd.to_datetime(tot_tb_non_hiv_deaths.index, format='%Y')
# tb mortality rates per 100k person-years
model_dict['TB_mortality_per_100k'] = (tot_tb_non_hiv_deaths / py) * 100000

# -------------------------------- CALIBRATION RESULTS -------------------------------- #

# compute calibration score for each
calibration_score = weighted_mean(data_dict, model_dict)

# select best-fitting model
