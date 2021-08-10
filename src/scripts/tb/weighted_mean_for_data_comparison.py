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
    (data_hiv_mphia_inc.age == "15-49"), "total_percent_annual_incidence"].values[0]

data_dict['mphia_prev_2015'] = data_hiv_mphia_prev.loc[
    data_hiv_mphia_prev.age == "Total 15-49", "total percent hiv positive"].values[0]

# DHS HIV data
data_hiv_dhs_prev = pd.read_excel(xls, sheet_name='DHS_prevalence')
data_dict['dhs_prev_2010'] = data_hiv_dhs_prev.loc[
    (data_hiv_dhs_prev.Year == 2010), "HIV prevalence among general population 15-49"]
data_dict['dhs_prev_2015'] = data_hiv_dhs_prev.loc[
    (data_hiv_dhs_prev.Year == 2015), "HIV prevalence among general population 15-49"]

## TB
# TB WHO data
xls_tb = pd.ExcelFile(resourcefilepath / 'ResourceFile_TB.xlsx')

data_tb_who = pd.read_excel(xls_tb, sheet_name='WHO_activeTB2020')
data_dict['who_tb_inc_per_100k_2010'] = data_tb_who.loc[(data_tb_who.year == 2010), 'incidence_per_100k']
data_dict['who_tb_inc_per_100k_2011'] = data_tb_who.loc[(data_tb_who.year == 2011), 'incidence_per_100k']
data_dict['who_tb_inc_per_100k_2012'] = data_tb_who.loc[(data_tb_who.year == 2012), 'incidence_per_100k']
data_dict['who_tb_inc_per_100k_2013'] = data_tb_who.loc[(data_tb_who.year == 2013), 'incidence_per_100k']
data_dict['who_tb_inc_per_100k_2014'] = data_tb_who.loc[(data_tb_who.year == 2014), 'incidence_per_100k']
data_dict['who_tb_inc_per_100k_2015'] = data_tb_who.loc[(data_tb_who.year == 2015), 'incidence_per_100k']
data_dict['who_tb_inc_per_100k_2016'] = data_tb_who.loc[(data_tb_who.year == 2016), 'incidence_per_100k']
data_dict['who_tb_inc_per_100k_2017'] = data_tb_who.loc[(data_tb_who.year == 2017), 'incidence_per_100k']
data_dict['who_tb_inc_per_100k_2018'] = data_tb_who.loc[(data_tb_who.year == 2018), 'incidence_per_100k']

# TB latent data (Houben & Dodd 2016)
data_tb_latent = pd.read_excel(xls_tb, sheet_name='latent_TB2014_summary')
data_tb_latent_all_ages = data_tb_latent.loc[data_tb_latent.Age_group == '0_80']
data_dict['who_tb_latent_prev'] = data_tb_latent_all_ages.proportion_latent_TB.values[0]

# TB case notification rate NTP
data_tb_ntp = pd.read_excel(xls_tb, sheet_name='NTP2019')
data_dict['ntp_case_notification_per_100k_2012'] = data_tb_ntp.loc[
    (data_tb_ntp.year == 2012), 'case_notification_rate_per_100k']
data_dict['ntp_case_notification_per_100k_2013'] = data_tb_ntp.loc[
    (data_tb_ntp.year == 2013), 'case_notification_rate_per_100k']
data_dict['ntp_case_notification_per_100k_2014'] = data_tb_ntp.loc[
    (data_tb_ntp.year == 2014), 'case_notification_rate_per_100k']
data_dict['ntp_case_notification_per_100k_2015'] = data_tb_ntp.loc[
    (data_tb_ntp.year == 2015), 'case_notification_rate_per_100k']
data_dict['ntp_case_notification_per_100k_2016'] = data_tb_ntp.loc[
    (data_tb_ntp.year == 2016), 'case_notification_rate_per_100k']
data_dict['ntp_case_notification_per_100k_2017'] = data_tb_ntp.loc[
    (data_tb_ntp.year == 2017), 'case_notification_rate_per_100k']
data_dict['ntp_case_notification_per_100k_2018'] = data_tb_ntp.loc[
    (data_tb_ntp.year == 2018), 'case_notification_rate_per_100k']
data_dict['ntp_case_notification_per_100k_2019'] = data_tb_ntp.loc[
    (data_tb_ntp.year == 2019), 'case_notification_rate_per_100k']

# need weights for each data item
model_weight = 0.5


# -------------------------------- CALIBRATION SCORE -------------------------------- #

# function which returns weighted mean of model outputs compared with data
# requires model outputs as inputs
def weighted_mean(model_outputs, data_items):
    # assert model_output is not empty

    # return calibration score (weighted mean deviance)
    # sqrt( (observed data – model output)^2 | / observed data)
    # sum these for each data item (all male prevalence over time) and divide by # items
    # then weighted sum of all components -> calibration score

    def deviance_function(data, model):
        deviance = math.sqrt(((data - model) ^ 2) / data)

        return deviance

    # hiv prevalence in adults 15-49: dhs 2010, 2015
    hiv_prev_dhs = sum(deviance_function(data_items['dhs_prev_2010'], model_outputs['hiv_prev_adult_2010']),
                   deviance_function(data_items['dhs_prev_2015'], model_outputs['hiv_prev_adult_2015'])) / 2

    # hiv prevalence mphia

    # hiv incidence mphia

    # aids deaths unaids

    # tb active incidence (WHO estimates)

    # tb death rate ntp

    calibration_score = hiv_prev_dhs
    return (calibration_score)


# -------------------------------- CALIBRATION RESULTS -------------------------------- #

# read in all output files

# get logged outputs for calibration into dict
model_dict = {}

# -------------------------------- MODEL OUTPUTS -------------------------------- #
prev_and_inc_over_time = output['tlo.methods.hiv'][
    'summary_inc_and_prev_for_adults_and_children_and_fsw']
prev_and_inc_over_time = prev_and_inc_over_time.set_index('date')

# HIV - prevalence among in adults aged 15+
prev_and_inc_over_time['hiv_prev_adult_15plus'] * 100

# compute calibration score for each

# select best-fitting model
