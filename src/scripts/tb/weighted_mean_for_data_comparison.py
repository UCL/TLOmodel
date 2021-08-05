"""
define a function which reads in relevant data and
"""

import matplotlib.pyplot as plt
import pandas as pd
import datetime
import pickle
from pathlib import Path

# declare the paths
resourcefilepath = Path("./resources")
outputpath = Path("./outputs")  # folder for convenience of storing outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# read in resource files for data
xls = pd.ExcelFile(resourcefilepath / 'ResourceFile_HIV.xlsx')
# MPHIA HIV data - age-structured
data_hiv_mphia_inc = pd.read_excel(xls, sheet_name='MPHIA_incidence2015')
data_hiv_mphia_prev = pd.read_excel(xls, sheet_name='MPHIA_prevalence_art2015')
# DHS HIV data
data_hiv_dhs_prev = pd.read_excel(xls, sheet_name='DHS_prevalence')

# need weights for each data item
weights = pd.read_excel(xls, sheet_name='calibration_weights')

# function which returns weighted mean of model outputs compared with data
# requires model outputs as inputs
# def weighted_mean(model_output):
    # assert model_output is not empty

    # return calibration score (weighted mean deviance)
    # sqrt( (observed data – model output)^2 | / observed data)
    # sum these for each data item (all male prevalence over time) and divide by # items
    # then weighted sum of all components -> calibration score

    # hiv prevalence dhs

    # hiv prevalence mphia

    # hiv incidence mphia

    # aids deaths unaids

    # tb active incidence (WHO estimates)

    # tb death rate ntp


