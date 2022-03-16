"""
* Check key outputs for reporting in the calibration table of the write-up
* Produce representative plots for the default parameters

NB. To see larger effects
* Increase incidence of cancer (see tests)
* Increase symptom onset (r_dysphagia_stage1)
* Increase progression rates (see tests)
"""

import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo import Date, Simulation
from tlo.analysis.utils import make_age_grp_types, parse_log_file
from tlo.methods import (
    care_of_women_during_pregnancy,
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    newborn_outcomes,
    oesophagealcancer,
    other_adult_cancers,
    postnatal_supervisor,
    pregnancy_supervisor,
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
end_date = Date(2015, 1, 1)
popsize = 10000


def run_sim(service_availability):
    # Establish the simulation object and set the seed
    sim = Simulation(start_date=start_date, seed=0)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=service_availability),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 other_adult_cancers.OtherAdultCancer(resourcefilepath=resourcefilepath),
                 oesophagealcancer.OesophagealCancer(resourcefilepath=resourcefilepath)
                 )

    # Establish the logger
    logfile = sim.configure_logging(filename="LogFile")

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    return logfile


def get_summary_stats(logfile):
    output = parse_log_file(logfile)

    # 1) TOTAL COUNTS BY STAGE OVER TIME
    counts_by_stage = output['tlo.methods.other_adult_cancers']['summary_stats']
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
    dalys = dalys.groupby(by=dalys['age_range']).sum()
    dalys = dalys.set_index(make_age_grp_types().categories)

    # 4) DEATHS wrt age (total over whole simulation)
    # get the deaths dataframe
    deaths = output['tlo.methods.demography']['death']
    # sort deaths by age group
    deaths['age_group'] = deaths['age'].map(demography.Demography(resourcefilepath=resourcefilepath).AGE_RANGE_LOOKUP)
    # isolate other adult cancer deaths
    deaths = deaths.loc[deaths.cause == 'OtherAdultCancer']
    # group deaths by age group
    death_counts_by_age_group = deaths.groupby(by=deaths['age_group']).size()
    # index the death counts by age group
    death_counts_by_age_group = death_counts_by_age_group.reindex(
        pd.Index.intersection(make_age_grp_types().categories, death_counts_by_age_group.index))

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
        'other_adult_cancer_deaths': death_counts_by_age_group,
        'annual_count_of_dxtr': annual_count_of_dxtr
    }


# %% Run the simulation with and without interventions being allowed

# With interventions:
logfile_with_healthsystem = run_sim(service_availability=['*'])
results_with_healthsystem = get_summary_stats(logfile_with_healthsystem)

# Without interventions:
logfile_no_healthsystem = run_sim(service_availability=[])
results_no_healthsystem = get_summary_stats(logfile_no_healthsystem)

# %% Produce Summary Graphs:

# Examine Counts by Stage Over Time
counts = results_no_healthsystem['total_counts_by_stage_over_time']
counts.plot(y=['total_site_confined',
               'total_local_ln',
               'total_metastatic',
               ])
plt.title('Count in Each Stage of Disease Over Time')
plt.xlabel('Time')
plt.ylabel('Count')
plt.show()

# Examine numbers in each stage of the cascade:
results_with_healthsystem['counts_by_cascade'].plot(y=['udx', 'dx', 'tr', 'pc'])
plt.title('With Health System')
plt.xlabel('Numbers of those With Cancer by Stage in Cascade')
plt.xlabel('Time')
plt.legend(['Undiagnosed', 'Diagnosed', 'On Treatment', 'On Palliative Care'])
plt.show()

results_no_healthsystem['counts_by_cascade'].plot(y=['udx', 'dx', 'tr', 'pc'])
plt.title('With No Health System')
plt.xlabel('Numbers of those With Cancer by Stage in Cascade')
plt.xlabel('Time')
plt.legend(['Undiagnosed', 'Diagnosed', 'On Treatment', 'On Palliative Care'])
plt.show()

# Examine DALYS (summed over whole simulation)
results_no_healthsystem['dalys'].plot.bar(
    y=['YLD_OtherAdultCancer_0', 'YLL_OtherAdultCancer_OtherAdultCancer'],
    stacked=True)
plt.xlabel('Age-group')
plt.ylabel('DALYS')
plt.legend()
plt.title("With No Health System")
plt.show()

# Examine Deaths (summed over whole simulation)
deaths = results_no_healthsystem['other_adult_cancer_deaths']
deaths.reindex(pd.Index.intersection(make_age_grp_types().categories, deaths.index))
# # make a series with the right categories and zero so formats nicely in the grapsh:
agegrps = demography.Demography(resourcefilepath=resourcefilepath).AGE_RANGE_CATEGORIES
totdeaths = pd.Series(index=pd.Index.intersection(make_age_grp_types().categories, deaths.index), data=np.nan)
totdeaths = totdeaths.combine_first(deaths).fillna(0.0)
totdeaths.plot.bar()
plt.title('Deaths due to Other Adult Cancer')
plt.xlabel('Age-group')
plt.ylabel('Total Deaths During Simulation')
# plt.gca().get_legend().remove()
plt.show()

# Compare Deaths - with and without the healthsystem functioning - sum over age and time
results_dict = {
    'No_HealthSystem': sum(results_no_healthsystem['other_adult_cancer_deaths']),
    'With_HealthSystem': sum(results_with_healthsystem['other_adult_cancer_deaths'])
}
results_df = pd.DataFrame(results_dict, index=[''])

results_df.plot.bar()
plt.title('Deaths due to OtherAdult Cancer')
plt.xlabel('Scenario')
plt.ylabel('Total Deaths During Simulation')
plt.show()


# %% Get Statistics for Table in write-up (from results_with_healthsystem);

# ** Current prevalence (end-2019) of people who have diagnosed OtherAdult cancer in 2020 (total; and current stage
# 1, 2, 3,
# 4), per 100,000 population aged 20+

counts = results_with_healthsystem['total_counts_by_stage_over_time'][[
    'total_site_confined',
    'total_local_ln',
    'total_metastatic'
]].iloc[-1]

totpopsize = results_with_healthsystem['total_counts_by_stage_over_time'][[
    'total_none',
    'total_site_confined',
    'total_local_ln',
    'total_metastatic'
]].iloc[-1].sum()

prev_per_100k = 1e5 * counts.sum() / totpopsize

# ** Number of deaths from OtherAdult cancer per year per 100,000 population.
# average deaths per year = deaths over ten years divided by ten, * 100k/population size
(results_with_healthsystem['other_adult_cancer_deaths'].sum()/10) * 1e5/popsize

# ** Incidence rate of diagnosis, treatment, palliative care for OtherAdult cancer (all stages combined),
# per 100,000 population
(results_with_healthsystem['annual_count_of_dxtr']).mean() * 1e5/popsize


# ** 5-year survival following treatment
# See separate file
