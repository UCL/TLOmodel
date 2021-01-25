import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    contraception,
    demography,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    ncds,
    pregnancy_supervisor,
    symptommanager,
)

# %%
resourcefilepath = Path("./resources")
outputpath = Path("./outputs")  # folder for convenience of storing outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")


# ------------------------------------------------- RUN THE SIMULATION -------------------------------------------------

def runsim(params):
    log_config = {'filename': 'LogFile'}
    # add file handler for the purpose of logging

    start_date = Date(2010, 1, 1)
    end_date = Date(2012, 1, 2)
    popsize = 1000

    sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

    # run the simulation
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 ncds.Ncds(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath)
                 )

    p = sim.modules['Ncds'].parameters

    p['nc_diabetes_onset'].loc[
        p['nc_diabetes_onset'].parameter_name == "baseline_annual_probability", "value"] = params[0]
    p['nc_hypertension_onset'].loc[
        p['nc_hypertension_onset'].parameter_name == "baseline_annual_probability", "value"] = params[2]
    p['nc_depression_onset'].loc[
        p['nc_depression_onset'].parameter_name == "baseline_annual_probability", "value"] = params[4]
    p['nc_chronic_ischemic_hd'].loc[
        p['nc_chronic_ischemic_hd'].parameter_name == "baseline_annual_probability", "value"] = params[6]
    p['nc_chronic_kidney_disease'].loc[
        p['nc_chronic_kidney_disease'].parameter_name == "baseline_annual_probability", "value"] = params[8]


    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    return sim

# Extract the relevant outputs and make a graph:
def get_prevalence_from_logfile(sim):
    logfile = sim.log_filepath
    output = parse_log_file(logfile)

    # import conditions and age range from modules
    conditions = sim.modules['Ncds'].conditions
    age_range = sim.modules['Demography'].AGE_RANGE_CATEGORIES

    # Calculate the "incidence rate" from the output counts of incidence
    counts = output['tlo.methods.ncds']['incidence_count_by_condition']

    return


def err(params, data_y1, data_y2):
    sim = runsim(params)

    prevalence_by_age = get_prevalence_from_logfile(sim)  # need an array of numbers
    err1 = prevalence_by_age - data_y1  # need an array of numbers for y_data
    err2 = prevalence_by_age - data_y2

    return np.concatenate((err1, err2))


p0 = [0.003, 0.01, 0.0139, 0.01, 0.0017, 0.01, 0.00324, 0.01, 0.001018, 0.01] # baseline probability of acquiring and dying from: diabetes, hypertension, depression, heart disease, kidney disease
data_x = [1, 2, 3, 4, 5, 6, 7, 8] # number of age cats (0-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80+)
data_y1 = [0.0, 0.0, 0.0029, 0.0126, 0.0382, 0.0820, 0.1323, 0.1936, 0.1761] # prevalence of diabetes
data_y2 = [0.0, 0.0, 0.0091, 0.0370, 0.1180, 0.2655, 0.4551, 0.6476, 0.7375] # prevalence of hypertension
data_y3 = [0.0, 0.0, 0.1505, 0.2049, 0.2492, 0.2510, 0.2277, 0.2031, 0.2019] # prevalence of depression
data_y4 = [0.0, 0.0, 0.0091, 0.0370, 0.1180, 0.2655, 0.4551, 0.6476, 0.7375] # prevalence of chronic ischemic heart disease
data_y5 = [0.0, 0.0, 0.0003, 0.0009, 0.0033, 0.0140, 0.0611, 0.1906, 0.3209] # prevalence of chronic kidney disease

p_best, ier = scipy.optimize.leastsq(err, p0, args=(data_x, data_y1, data_y2, data_y3, data_y4, data_y5))
