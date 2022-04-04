"""Run a simulation with no HSI constraints and plot the prevalence and incidence and program coverage trajectories"""
import datetime
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    care_of_women_during_pregnancy,
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    newborn_outcomes,
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

# %% Run the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 1000

# Establish the simulation object
log_config = {
    'filename': 'Logfile',
    'directory': outputpath,
    'custom_levels': {
        '*': logging.WARNING,
        'tlo.methods.hiv': logging.INFO,
        'tlo.methods.demography': logging.INFO
    }
}

# Register the appropriate modules
sim = Simulation(start_date=start_date, seed=100, log_config=log_config)
sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             contraception.Contraception(resourcefilepath=resourcefilepath),
             enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
             healthburden.HealthBurden(resourcefilepath=resourcefilepath),
             healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                       service_availability=['*']),
             labour.Labour(resourcefilepath=resourcefilepath),
             newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
             care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
             symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
             pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
             postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
             healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
             hiv.Hiv(resourcefilepath=resourcefilepath)
             )


# Run the simulation and flush the logger
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# parse the results
output = parse_log_file(sim.log_filepath)

# save the results
with open(outputpath / 'default_run.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)


# %% Get ready to create plots:

# load the results
with open(outputpath / 'default_run.pickle', 'rb') as f:
    output = pickle.load(f)

# load the calibration data
data = pd.read_excel(resourcefilepath / 'ResourceFile_HIV.xlsx', sheet_name='calibration_from_aids_info')
data.index = pd.to_datetime(data['year'], format='%Y')
data = data.drop(columns=['year'])

# %% Function to make standard plot to compare model and data


def make_plot(
    model=None,
    data_mid=None,
    data_low=None,
    data_high=None,
    title_str=None
):
    assert model is not None
    assert title_str is not None

    # Make plot
    fig, ax = plt.subplots()
    ax.plot(model.index, model.values, '-', color='r')

    if data_mid is not None:
        ax.plot(data_mid.index, data_mid.values, '-')
    if (data_low is not None) and (data_high is not None):
        ax.fill_between(data_low.index,
                        data_low,
                        data_high,
                        alpha=0.2)
    plt.title(title_str)
    plt.legend(['Model', 'Data'])
    plt.gca().set_ylim(bottom=0)
    plt.savefig(outputpath / (title_str.replace(" ", "_") + datestamp + ".pdf"), format='pdf')
    plt.show()


# %% : PREVALENCE AND INCIDENCE PLOTS

# adults - prevalence among 15-49 year-olds
prev_and_inc_over_time = output['tlo.methods.hiv'][
    'summary_inc_and_prev_for_adults_and_children_and_fsw']
prev_and_inc_over_time = prev_and_inc_over_time.set_index('date')

# Prevalence 15-49
make_plot(
    title_str="HIV Prevalence in Adults (15-49) (%)",
    model=prev_and_inc_over_time['hiv_prev_adult_1549'] * 100,
    data_mid=data['prev_15_49'],
    data_low=data['prev_15_49_lower'],
    data_high=data['prev_15_49_upper']
)

# Incidence 15-49
make_plot(
    title_str="HIV Incidence in Adults (15-49) (per 100 pyar)",
    model=prev_and_inc_over_time['hiv_adult_inc_1549'] * 100,
    data_mid=data['inc_15_49_per1000'] / 10,
    data_low=data['inc_15_49_per1000lower'] / 10,
    data_high=data['inc_15_49_per1000upper'] / 10
)

# Prevalence Children
make_plot(
    title_str="HIV Prevalence in Children (0-14) (%)",
    model=prev_and_inc_over_time['hiv_prev_child'] * 100,
)

# Incidence Children
make_plot(
    title_str="HIV Prevalence in Children (0-14) (per 100 pyar)",
    model=prev_and_inc_over_time['hiv_child_inc'] * 100,
)

# HIV prevalence among female sex workers:
make_plot(
    title_str="HIV Prevalence among Female Sex Workers (%)",
    model=prev_and_inc_over_time['hiv_prev_fsw'] * 100,
)

# %% : AIDS DEATHS
deaths = output['tlo.methods.demography']['death'].copy()
deaths = deaths.set_index('date')
# limit to deaths among aged 15+
to_drop = ((deaths.age < 15) | (deaths.cause != 'AIDS'))
deaths = deaths.drop(index=to_drop[to_drop].index)

# count by year:
deaths['year'] = deaths.index.year
tot_aids_deaths = deaths.groupby(by=['year']).size()

# person-years among those aged 15+ (irrespective of HIV status)
py_ = output['tlo.methods.demography']['person_years']
years = pd.to_datetime(py_['date']).dt.year
py = pd.Series(index=years)
for year in years:
    tot_py = (
        (py_.loc[pd.to_datetime(py_['date']).dt.year == year]['M']).apply(pd.Series) +
        (py_.loc[pd.to_datetime(py_['date']).dt.year == year]['F']).apply(pd.Series)
    ).transpose()
    py[year] = tot_py[tot_py.index.astype(int) >= 15].sum().values[0]

aids_death_rate = tot_aids_deaths / py
aids_death_rate.index = pd.to_datetime(aids_death_rate.index, format='%Y')
# (NB. this assumes that the data mortality rate is for irrespective of HIV status)
make_plot(
    title_str='Mortality to HIV-AIDS per 100k capita',
    model=aids_death_rate * 100_000,
    data_mid=data['mort_rate100k'],
    data_low=data['mort_rate100k_lower'],
    data_high=data['mort_rate100k_upper']
)


# %% PROGRAM COVERAGE PLOTS
cov_over_time = output['tlo.methods.hiv']['hiv_program_coverage']
cov_over_time = cov_over_time.set_index('date')

# Treatment Cascade ("90-90-90") Plot for Adults
dx = cov_over_time['dx_adult']
art_among_dx = cov_over_time['art_coverage_adult'] / dx
vs_among_art = cov_over_time['art_coverage_adult_VL_suppression']
pd.concat({'diagnosed': dx,
           'art_among_diagnosed': art_among_dx,
           'vs_among_those_on_art': vs_among_art
           }, axis=1).plot()
plt.title('ART Cascade for Adults (15+)')
plt.savefig(outputpath / ("HIV_art_cascade_adults" + datestamp + ".pdf"), format='pdf')
plt.show()

# Per capita testing rates - data from MoH quarterly reports
make_plot(
    title_str="Per capita testing rates for adults (15+)",
    model=cov_over_time["per_capita_testing_rate"],
    data_mid=data["adult_tests_per_capita"]
)

# Percent on ART
make_plot(
    title_str="Percent of Adults (15+) on ART",
    model=cov_over_time["art_coverage_adult"] * 100,
    data_mid=data["percent15plus_on_art"],
    data_low=data["percent15plus_on_art_lower"],
    data_high=data["percent15plus_on_art_upper"]
)

# Circumcision
make_plot(
    title_str="Proportion of Men (15+) That Are Circumcised",
    model=cov_over_time["prop_men_circ"]
)

# PrEP among FSW
make_plot(
    title_str="Proportion of FSW That Are On PrEP",
    model=cov_over_time["prop_fsw_on_prep"]
)

# Behaviour Change
make_plot(
    title_str="Proportion of Adults (15+) Exposed to Behaviour Change Intervention",
    model=cov_over_time["prop_adults_exposed_to_behav_intv"]
)
