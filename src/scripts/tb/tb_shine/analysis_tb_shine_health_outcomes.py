"""
Run the HIV/TB modules with intervention coverage specified at national level
save outputs for plotting (file: output_plots_tb.py)
 """

import datetime
import pickle
import random
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    simplified_births,
    symptommanager,
    tb,
)

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

# %% Run the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2019, 1, 1)
popsize = 1000

# set up the log config
log_config = {
    "filename": "Logfile",
    "directory": outputpath,
    "custom_levels": {
        "*": logging.WARNING,
        "tlo.methods.hiv": logging.INFO,
        "tlo.methods.tb": logging.INFO,
        "tlo.methods.demography": logging.INFO,
        "tlo.methods.healthsystem": logging.INFO,
    },
}

# Register the appropriate modules
# need to call epi before tb to get bcg vax
seed = random.randint(0, 50000)
# seed = 4  # set seed for reproducibility
sim = Simulation(start_date=start_date, seed=seed, log_config=log_config, show_progress_bar=True)
# Used to configure health system behaviour
service_availability = ["*"]
sim.register(
    demography.Demography(resourcefilepath=resourcefilepath),
    simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
    healthsystem.HealthSystem(
        resourcefilepath=resourcefilepath,
        service_availability=["*"],  # all treatment allowed
        mode_appt_constraints=0,  # mode of constraints to do with officer numbers and time
        cons_availability="all",  # mode for consumable constraints (if ignored, all consumables available)
        ignore_priority=True,  # do not use the priority information in HSI event to schedule
        capabilities_coefficient=1.0,  # multiplier for the capabilities of health officers
        disable=False,  # disables the healthsystem (no constraints and no logging) and every HSI runs
        disable_and_reject_all=False,  # disable healthsystem and no HSI runs
        store_hsi_events_that_have_run=False,  # convenience function for debugging
    ),
    symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
    healthburden.HealthBurden(resourcefilepath=resourcefilepath),
    epi.Epi(resourcefilepath=resourcefilepath),
    hiv.Hiv(resourcefilepath=resourcefilepath),
    tb.Tb(resourcefilepath=resourcefilepath),
)

# change IPT high-risk districts to all districts for national-level model
sim.modules["Tb"].parameters["tb_high_risk_distr"] = pd.read_excel(
    resourcefilepath / "ResourceFile_TB.xlsx", sheet_name="all_districts"
)

# choose the scenario, 0=baseline, 5=shorter paediatric treatment
sim.modules["Tb"].parameters["scenario"] = 0

# Run the simulation and flush the logger
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# parse the results
output = parse_log_file(sim.log_filepath)

# save the results, argument 'wb' means write using binary mode. use 'rb' for reading file
with open(outputpath / "default_run.pickle", "wb") as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)

# --------------- LOAD DATA --------------- #
data_who_tb_2020 = pd.read_excel(resourcefilepath / 'ResourceFile_TB.xlsx', sheet_name='WHO_activeTB2020')
data_who_tb_2020.index = pd.to_datetime(data_who_tb_2020['year'], format='%Y')
data_who_tb_2020 = data_who_tb_2020.drop(columns=['year'])

data_who_tb_2014 = pd.read_excel(resourcefilepath / 'ResourceFile_TB.xlsx', sheet_name='latent_TB2014_summary')

# --------------- ANALYSIS PLOTS --------------- #

def make_plot(model=None, data_mid=None, data_low=None, data_high=None, title_str=None):
    assert model is not None
    assert title_str is not None

    # Make plot
    fig, ax = plt.subplots()
    ax.plot(model.index, model.values, "-", color="r")

    if data_mid is not None:
        ax.plot(data_mid.index, data_mid.values, "-")
    if (data_low is not None) and (data_high is not None):
        ax.fill_between(data_low.index, data_low, data_high, alpha=0.2)
    plt.title(title_str)
    plt.xlabel("Years")
    plt.gca().set_ylim(bottom=0)
    # plt.savefig(outputpath / (title_str.replace(" ", "_") + datestamp + ".pdf"), format='pdf')
    plt.show()


# -------------------------------------------------------------------------------------- #
#                                 HEALTH OUTCOME ANALYSIS                                #
# -------------------------------------------------------------------------------------- #
# ----------------------------------- PREVALENCE --------------------------------------- #
tb_prevalence = output['tlo.methods.tb']['tb_prevalence']
tb_prevalence = tb_prevalence.set_index('date')

# PLOT: Latent TB Prevalence
make_plot(
    title_str="Prevalence of Latent TB",
    model=tb_prevalence['tbPrevLatent'],
)

# PLOT: Latent TB Prevalence (0 -15 years)
make_plot(
    title_str="Prevalence of Latent TB (0 - 15 years)",
    model=tb_prevalence['tbPrevLatentChild'],
)

# PLOT: Active TB Prevalence
make_plot(
    title_str="Prevalence of Active TB",
    model=tb_prevalence['tbPrevActive'],
    data_mid=data_who_tb_2020['prevalence_all_ages'],
    data_low=data_who_tb_2020['prevalence_all_ages_low'],
    data_high=data_who_tb_2020['prevalence_all_ages_high'],
)

# PLOT: Active TB Prevalence (0 -15 years)
make_plot(
    title_str="Prevalence of Active TB (0 - 15 years)",
    model=tb_prevalence['tbPrevActiveChild'],
)

# ------------------------------------ INCIDENCE --------------------------------------- #
tb_incidence = output['tlo.methods.tb']['tb_incidence']
tb_incidence = tb_incidence.set_index('date')

# PLOT: Number of New Latent TB Cases
make_plot(
    title_str="Latent TB Incidence",
    model=tb_incidence['num_new_latent_tb'],
)

# PLOT: Number of New Latent TB Cases (0 - 15 years)
make_plot(
    title_str="Latent TB Incidence (0 - 15 years)",
    model=tb_incidence['num_new_latent_tb_child'],
)

# PLOT: Number of New Active TB Cases
make_plot(
    title_str="Active TB Incidence",
    model=tb_incidence['num_new_active_tb'],
)

# PLOT: Number of New Active TB Cases (0 - 15 years)
make_plot(
    title_str="Active TB Incidence (0 - 15 years)",
    model=tb_incidence['num_new_active_tb_child'],
)

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


# Active TB Incidence
TB_inc = output['tlo.methods.tb']['tb_incidence']
TB_inc = TB_inc.set_index('date')
TB_inc.index = pd.to_datetime(TB_inc.index)
activeTB_inc_rate = (TB_inc['num_new_active_tb'] / py) * 100000

# Active TB Incidence (0 - 15 years)
TB_inc_child = output['tlo.methods.tb']['tb_incidence']
TB_inc_child = TB_inc_child.set_index('date')
TB_inc_child.index = pd.to_datetime(TB_inc_child.index)
activeTB_inc_rate_child = (TB_inc_child['num_new_active_tb_child'] / py) * 100000

# PLOT: Active TB Incidence per 100k person years
make_plot(
    title_str="Active TB Incidence (per 100k person-years)",
    model=activeTB_inc_rate,
    data_mid=data_who_tb_2020['incidence_per_100k'],
    data_low=data_who_tb_2020['incidence_per_100k_low'],
    data_high=data_who_tb_2020['incidence_per_100k_high']
)

# PLOT: Active TB Incidence per 100k person years (0 -15 years)
make_plot(
    title_str="Active TB Incidence (0 -15 years) (per 100k person-years)",
    model=activeTB_inc_rate_child,
)

# ------------------------------------- DEATHS ----------------------------------------- #
deaths = output['tlo.methods.demography']['death'].copy()
deaths = deaths.set_index('date')

# Non-HIV TB Deaths (exclude HIV/TB deaths)
to_drop = (deaths.cause != 'TB')
deaths_TB = deaths.drop(index=to_drop[to_drop].index).copy()
deaths_TB['year'] = deaths_TB.index.year
tot_non_hiv_tb_deaths = deaths_TB.groupby(by=['year']).size()
tot_non_hiv_tb_deaths.index = pd.to_datetime(tot_non_hiv_tb_deaths.index, format='%Y')
tot_non_hiv_tb_deaths_rate = (tot_non_hiv_tb_deaths / py) * 100000

# HIV/TB Deaths (exclude non-HIV TB deaths)
to_drop = (deaths.cause != 'AIDS_non_TB')
deaths_TB_HIV = deaths.drop(index=to_drop[to_drop].index).copy()
deaths_TB_HIV['year'] = deaths_TB_HIV.index.year
tot_hiv_tb_deaths = deaths_TB_HIV.groupby(by=['year']).size()
tot_hiv_tb_deaths.index = pd.to_datetime(tot_hiv_tb_deaths.index, format='%Y')
tot_hiv_tb_deaths_rate = (tot_hiv_tb_deaths / py) * 100000

# Total TB Deaths (HIV/TB and non-HIV TB deaths)
tot_tb_deaths = tot_non_hiv_tb_deaths.add(tot_hiv_tb_deaths, fill_value=0)
tot_tb_deaths.index = pd.to_datetime(tot_tb_deaths.index, format='%Y')
tot_tb_deaths_rate = (tot_tb_deaths / py) * 100000

# PLOT: Non-HIV TB Mortality Rate
make_plot(
    title_str="TB Mortality Rate per 100k person years",
    model=tot_non_hiv_tb_deaths_rate,
    data_mid=data_who_tb_2020['mortality_tb_excl_hiv_per_100k'],
    data_low=data_who_tb_2020['mortality_tb_excl_hiv_per_100k_low'],
    data_high=data_who_tb_2020['mortality_tb_excl_hiv_per_100k_high'],
)

# PLOT: HIV/TB Mortality Rate
make_plot(
    title_str="HIV/TB Mortality Rate per 100k person years",
    model=tot_hiv_tb_deaths_rate,
    data_mid=data_who_tb_2020['mortality_tb_hiv_per_100k'],
    data_low=data_who_tb_2020['mortality_tb_hiv_per_100k_low'],
    data_high=data_who_tb_2020['mortality_tb_hiv_per_100k_high'],
)

# PLOT: Total TB Mortality Rate
make_plot(
    title_str="Total TB Mortality Rate per 100k person years",
    model=tot_tb_deaths_rate,
    data_mid=data_who_tb_2020['total_mortality_tb_per_100k'],
    data_low=data_who_tb_2020['total_mortality_tb_per_100k_low'],
    data_high=data_who_tb_2020['total_mortality_tb_per_100k_high'],
)



# Non-HIV TB Deaths (exclude HIV/TB deaths) in Children (0-16 years)
to_drop = ((deaths.cause != 'TB') & (deaths.age < 16))
deaths_TB_child = deaths.drop(index=to_drop[to_drop].index).copy()
deaths_TB_child['year'] = deaths_TB_child.index.year
tot_non_hiv_tb_deaths_child = deaths_TB_child.groupby(by=['year']).size()
tot_non_hiv_tb_deaths_child.index = pd.to_datetime(tot_non_hiv_tb_deaths_child.index, format='%Y')
tot_non_hiv_tb_deaths_rate_child = (tot_non_hiv_tb_deaths_child / py) * 100000

# HIV/TB Deaths (exclude non-HIV TB deaths) in Children (0-16 years)
to_drop = ((deaths.cause != 'AIDS_non_TB') & (deaths.age < 16))
deaths_TB_HIV_child = deaths.drop(index=to_drop[to_drop].index).copy()
deaths_TB_HIV_child['year'] = deaths_TB_HIV_child.index.year
tot_hiv_tb_deaths_child = deaths_TB_HIV_child.groupby(by=['year']).size()
tot_hiv_tb_deaths_child.index = pd.to_datetime(tot_hiv_tb_deaths_child.index, format='%Y')
tot_hiv_tb_deaths_rate_child = (tot_hiv_tb_deaths_child / py) * 100000

# Total TB Deaths (HIV/TB and non-HIV TB deaths) in Children (0-16 years)
tot_tb_deaths_child = tot_non_hiv_tb_deaths_child.add(tot_hiv_tb_deaths_child, fill_value=0)
tot_tb_deaths_child.index = pd.to_datetime(tot_tb_deaths_child.index, format='%Y')
tot_tb_deaths_rate_child = (tot_tb_deaths_child / py) * 100000

# PLOT: Non-HIV TB Mortality Rate (0 -16 years)
make_plot(
    title_str="TB Mortality Rate per 100k person years (0 - 16 years)",
    model=tot_non_hiv_tb_deaths_rate_child,
)

# PLOT: HIV/TB Mortality Rate (0 -16 years)
make_plot(
    title_str="HIV/TB Mortality Rate per 100k person years (0 - 16 years)",
    model=tot_hiv_tb_deaths_rate_child,
)

# PLOT: Total TB Mortality Rate (0 -16 years)
make_plot(
    title_str="Total TB Mortality Rate per 100k person years (0 - 16 years)",
    model=tot_tb_deaths_rate_child,
)
