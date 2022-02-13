"""
Run the HIV/TB modules with intervention coverage specified at national level
save outputs for plotting (file: output_plots_tb.py)
 """

import datetime
import pickle
import random
from pathlib import Path

import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    deviance_measure,
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
end_date = Date(2025, 1, 1)
popsize = 40000
# todo check parameters set below

# set up the log config
log_config = {
    "filename": "deviance_calibrated",
    "directory": outputpath,
    "custom_levels": {
        "*": logging.WARNING,
        # "tlo.methods.deviance_measure": logging.INFO,
        "tlo.methods.hiv": logging.INFO,
        "tlo.methods.tb": logging.INFO,
        "tlo.methods.demography": logging.INFO,
    },
}

# Register the appropriate modules
# need to call epi before tb to get bcg vax
# seed = random.randint(0, 50000)
seed = 3  # set seed for reproducibility
sim = Simulation(start_date=start_date, seed=seed, log_config=log_config, show_progress_bar=True)
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
        disable=True,  # disables the healthsystem (no constraints and no logging) and every HSI runs
        disable_and_reject_all=False,  # disable healthsystem and no HSI runs
        store_hsi_events_that_have_run=False,  # convenience function for debugging
    ),
    symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
    healthburden.HealthBurden(resourcefilepath=resourcefilepath),
    epi.Epi(resourcefilepath=resourcefilepath),
    hiv.Hiv(resourcefilepath=resourcefilepath),
    tb.Tb(resourcefilepath=resourcefilepath),
    # deviance_measure.Deviance(resourcefilepath=resourcefilepath),
)

# todo remove
sim.modules["Hiv"].parameters["beta"] = 0.127623113

# transmission rate active cases -> new latent cases
sim.modules["Tb"].parameters["transmission_rate"] = 19.5

# this is the lifetime risk of active disease, scaled by risk factors
sim.modules["Tb"].parameters["prog_active"] = 0.1  # todo adjust to get correct active case numbers
sim.modules["Tb"].parameters["rr_tb_hiv"] = 25  # 20.6
sim.modules["Tb"].parameters["rr_tb_aids"] = 50  # 26

sim.modules["Tb"].parameters["prob_latent_tb_0_14"] = 0.07  # default
sim.modules["Tb"].parameters["prob_latent_tb_15plus"] = 0.5  # 0.27 default

# Run the simulation and flush the logger
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# parse the results
output = parse_log_file(sim.log_filepath)

# save the results, argument 'wb' means write using binary mode. use 'rb' for reading file
with open(outputpath / "default_run.pickle", "wb") as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)


###################################################
""" load the outputs from a simulation and plot the results with comparison data """

import datetime
import pickle
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo.analysis.utils import compare_number_of_deaths

resourcefilepath = Path("./resources")
outputpath = Path("./outputs")  # folder for convenience of storing outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")


# %% Function to make standard plot to compare model and data
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
    plt.legend(["Model", "Data"])
    plt.gca().set_ylim(bottom=0)
    plt.savefig(
        outputpath / (title_str.replace(" ", "_") + datestamp + ".pdf"), format="pdf"
    )
    # plt.show()


# ---------------------------------------------------------------------- #
# %%: DATA
# ---------------------------------------------------------------------- #
start_date = 2010
end_date = 2020

# load all the data for calibration

# TB WHO data
xls_tb = pd.ExcelFile(resourcefilepath / "ResourceFile_TB.xlsx")

data_tb_who = pd.read_excel(xls_tb, sheet_name="WHO_activeTB2020")
data_tb_who = data_tb_who.loc[
    (data_tb_who.year >= 2010)
]  # include only years post-2010
data_tb_who.index = pd.to_datetime(data_tb_who["year"], format="%Y")
data_tb_who = data_tb_who.drop(columns=["year"])

# TB treatment coverage
data_tb_ntp = pd.read_excel(xls_tb, sheet_name="NTP2019")
data_tb_ntp.index = pd.to_datetime(data_tb_ntp["year"], format="%Y")
data_tb_ntp = data_tb_ntp.drop(columns=["year"])


# ---------------------------------------------------------------------- #
# %%: OUTPUTS
# ---------------------------------------------------------------------- #

# load the results
with open(outputpath / "default_run.pickle", "rb") as f:
    output = pickle.load(f)

# person-years all ages (irrespective of HIV status)
py_ = output["tlo.methods.demography"]["person_years"]
years = pd.to_datetime(py_["date"]).dt.year
py = pd.Series(dtype="int64", index=years)
for year in years:
    tot_py = (
        (py_.loc[pd.to_datetime(py_["date"]).dt.year == year]["M"]).apply(pd.Series)
        + (py_.loc[pd.to_datetime(py_["date"]).dt.year == year]["F"]).apply(pd.Series)
    ).transpose()
    py[year] = tot_py.sum().values[0]

py.index = pd.to_datetime(years, format="%Y")


# ---------------------------------------------------------------------- #
# %%: DISEASE BURDEN
# ---------------------------------------------------------------------- #


# ----------------------------- TB -------------------------------------- #

# Active TB incidence per 100,000 person-years - annual outputs
TB_inc = output["tlo.methods.tb"]["tb_incidence"]
years = pd.to_datetime(TB_inc["date"]).dt.year
TB_inc.index = pd.to_datetime(years, format="%Y")
activeTB_inc_rate = (TB_inc["num_new_active_tb"] / py) * 100000

make_plot(
    title_str="Active TB Incidence (per 100k person-years)",
    model=activeTB_inc_rate,
    data_mid=data_tb_who["incidence_per_100k"],
    data_low=data_tb_who["incidence_per_100k_low"],
    data_high=data_tb_who["incidence_per_100k_high"],
)
plt.show()

# ---------------------------------------------------------------------- #
# latent tb incidence
new_latent = TB_inc["num_new_latent_tb"].copy()
new_latent.iloc[0] = 0  # otherwise super high from baseline

make_plot(
    title_str="Numbers new latent cases",
    model=new_latent,
)
plt.show()

# ---------------------------------------------------------------------- #

# latent TB prevalence
latentTB_prev = output["tlo.methods.tb"]["tb_prevalence"]
latentTB_prev = latentTB_prev.set_index("date")

title_str = "Latent TB prevalence"
make_plot(
    title_str=title_str,
    model=latentTB_prev["tbPrevLatent"],
)
plt.ylim((0, 0.4))
plt.show()

# -----------------------------------------------------------
Tb_tx_coverage = output["tlo.methods.tb"]["tb_treatment"]
Tb_tx_coverage = Tb_tx_coverage.set_index("date")
Tb_tx_coverage.index = pd.to_datetime(Tb_tx_coverage.index)

# TB treatment coverage
make_plot(
    title_str="Percent of TB cases treated",
    model=Tb_tx_coverage["tbTreatmentCoverage"] * 100,
    data_mid=data_tb_ntp["treatment_coverage"],
)
plt.ylim((0, 100))
plt.legend(["TLO", "NTP"])
plt.show()
