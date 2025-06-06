""" Compute the 5-YEAR SURVIVAL following treatment

To this by creating a simulation with:
 * a very high prevalence of tis_t1 at initiation
 * no one on treatment or diagnosed
 * and usual survival
 * no births
 * no new incidence
... and then examine time between date of treatment and date of death for those treated and those not treated.
"""
import datetime
from pathlib import Path

import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    bladder_cancer,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    symptommanager,
)

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

# Set parameters for the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2080, 1, 1)
popsize = 1000


# logger looks at only demography
log_config = {
    "filename": "LogFile",
    "custom_levels": {
        "*": logging.WARNING, "tlo.methods.demography": logging.INFO
    }
}

# Establish the simulation object and set the seed
sim = Simulation(start_date=start_date, seed=0, log_config=log_config, resourcefilepath=resourcefilepath)

# Register the appropriate modules
sim.register(demography.Demography(),
             enhanced_lifestyle.Lifestyle(),
             healthsystem.HealthSystem(disable=True),
             symptommanager.SymptomManager(),
             healthseekingbehaviour.HealthSeekingBehaviour(),
             healthburden.HealthBurden(),
             bladder_cancer.BladderCancer(),
             )

# Make there be a very high initial prevalence in the first stage and no on-going new incidence and no treatment to
# begin with:
sim.modules['BladderCancer'].parameters['r_tis_t1_bladder_cancer_none'] = 0.00
sim.modules['BladderCancer'].parameters['init_prop_bladder_cancer_stage'] = [1.0, 0.0, 0.0]
sim.modules['BladderCancer'].parameters["init_prop_blood_urine_bladder_cancer_by_stage"] = [0.0] * 4
sim.modules['BladderCancer'].parameters["init_prop_with_blood_urine_diagnosed_bladder_cancer_by_stage"] = [0.0] * 4
sim.modules['BladderCancer'].parameters["init_prop_treatment_status_bladder_cancer"] = [0.0] * 4
sim.modules['BladderCancer'].parameters["init_prob_palliative_care"] = 0.4

# Run the simulation
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# Read the output:
output = parse_log_file(sim.log_filepath)

# %% Analyse the output:

# get the person_ids of the original cohort
df = sim.population.props
cohort = df.iloc[1:popsize].index

# get the person_ids of the original cohort who started treatment
treated = pd.DataFrame(df.loc[df.index.isin(cohort) & ~pd.isnull(df.bc_date_treatment), 'bc_date_treatment'].copy())

# for each person that started treatment, get their date of starting treatment
deaths = pd.DataFrame(output['tlo.methods.demography']['death']).copy()

# find the date and cause of death of those persons:
deaths['person_id'] = deaths['person_id'].astype(int)
deaths = deaths.merge(treated, left_on='person_id', right_index=True, how='outer')

cohort_treated = deaths.dropna(subset=['bc_date_treatment']).copy()
cohort_treated['date'] = pd.to_datetime(cohort_treated['date'])
cohort_treated['bc_date_treatment'] = pd.to_datetime(cohort_treated['bc_date_treatment'])
cohort_treated['days_treatment_to_death'] = (cohort_treated['date'] - cohort_treated['bc_date_treatment']).dt.days

# calc % of those that were alive 5 years after starting treatment (not died of any cause):
1 - (
    len(cohort_treated.loc[cohort_treated['days_treatment_to_death'] < (5*365.25)]) /
    len(cohort_treated)
)  # 0.65

# calc % of those that had not died of Bladder cancer 5 years after starting treatment (could have died of another
# cause):
1 - (
    len(cohort_treated.loc[
            (cohort_treated['cause'] == 'BladderCancer') & (cohort_treated['days_treatment_to_death'] < (5*365.25))
            ]) / len(cohort_treated)
)   # 0.72
