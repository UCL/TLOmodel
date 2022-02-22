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
    "filename": "deviance_calibrated",
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

# Scaling Factor
sf = output['tlo.methods.demography']['scaling_factor']['scaling_factor'].values[0]


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
    plt.legend(["TLO Model", "WHO Data"])
    plt.xlabel("Years")
    plt.gca().set_ylim(bottom=0)
    # plt.savefig(outputpath / (title_str.replace(" ", "_") + datestamp + ".pdf"), format='pdf')
    plt.show()


# -------------------------------------------------------------------------------------- #
#                                   CONSUMABLES ANALYSIS                                 #
# -------------------------------------------------------------------------------------- #
# Here I am using the consumables log to plot the use of consumables over time.

# Load the consumables log and set the year as the index
cons = output['tlo.methods.healthsystem']["Consumables"].copy()
cons['year'] = cons['date'].dt.year
cons = cons.set_index('year')

# Format the consumables dataframe, so that the number of each consumable used can be calculated
tb_cons = cons.loc[cons.TREATMENT_ID.str.startswith('Tb')]
tb_cons = tb_cons.assign(Item_Available=tb_cons['Item_Available'].str.strip('{}').str.replace(' ', '').str.split(',')).explode('Item_Available')
tb_cons = tb_cons['Item_Available'].str.split(':', expand=True).rename(columns={0: "Item_Code", 1: "Quantity"})
tb_cons['Quantity'] = tb_cons['Quantity'].astype(int)

# -------------------------------- DIAGNOSTIC TESTS ----------------------------------- #
# Here I am plotting the use of the following diagnostic tests over time:

# (1) Sputum Smear Microscopy (Item Code: 184)
# First, extract all sputum smear microscopy tests from tb_cons (item code 184 = sputum smear microscopy)
# Then, sum the number of tests used per year
microscopy = tb_cons.loc[tb_cons.Item_Code.str.contains('184')].groupby(['year'])['Quantity'].sum().mul(sf)

# Plot the number of tests used per year
make_plot(
    title_str="Sputum Smear Microscopy",
    model=microscopy,
)

# (2) Xpert Test (Item Code: 187)
xpert = tb_cons.loc[tb_cons.Item_Code.str.contains('187')].groupby(['year'])['Quantity'].sum()

make_plot(
    title_str="Xpert Test",
    model=xpert,
)

# (3) Chest X-rays (Item Code: 175)
cxr = tb_cons.loc[tb_cons.Item_Code.str.contains('175')].groupby(['year'])['Quantity'].sum()

make_plot(
    title_str="Chest X-rays",
    model=cxr,
)

diagnostic_tests = tb_cons.loc[tb_cons.Item_Code.str.contains('175|184|187')].groupby(['Item_Code'])['Quantity'].sum()
diagnostic_tests.plot.bar(width=0.4)
plt.title('Total Diagnostic Tests')
plt.xticks([0, 1, 2], ['X-ray', 'Sputum Smear', 'Xpert'], rotation=30, ha='right')
plt.xlabel('Diagnostic Test')
plt.ylabel('Total')
plt.show()

# ----------------------------------- TREATMENTS -------------------------------------- #
# Here I am plotting the use of the following treatments over time:

# (1) First line treatment for children (Item Code: 178)
tx_child = tb_cons.loc[tb_cons.Item_Code.str.contains('178')].groupby(['year'])['Quantity'].sum()

make_plot(
    title_str="First Line Child Treatment",
    model=tx_child,
)

# (2) Shorter first line treatment for children (Item Code: 2675)
tx_child_shorter = tb_cons.loc[tb_cons.Item_Code.str.contains('2675')].groupby(['year'])['Quantity'].sum()

make_plot(
    title_str="First Line Child Treatment (Shorter)",
    model=tx_child_shorter,
)

# (3) Retreatment for children (Item Code: 179)
retx_child = tb_cons.loc[tb_cons.Item_Code.str.contains('179')].groupby(['year'])['Quantity'].sum()

make_plot(
    title_str="Child Retreatment",
    model=retx_child,
)

treatments = tb_cons.loc[tb_cons.Item_Code.str.contains('176|177|178|179|2675')].groupby(['Item_Code'])['Quantity'].sum()
treatments.plot.bar(width=0.4)
plt.title('Total Treatments')
plt.xticks([0, 1, 2, 3, 4], ['Adult tx', 'Adult retx', 'Child tx', 'Child retx', 'Child tx shorter'], rotation=30, ha='right')
plt.xlabel('Treatments')
plt.ylabel('Total')
plt.show()
# ---------------------------------- APPOINTMENTS ------------------------------------- #
# Here I am plotting the use of the following appointments over time:

# (1) Screening

screening = cons.loc[cons.TREATMENT_ID.str.startswith('Tb_ScreeningAndRefer')].groupby(by=['year']).size()

make_plot(
    title_str="Screening Appointments",
    model=screening,
)

# (2) Follow-up

follow_up = cons.loc[cons.TREATMENT_ID.str.startswith('Tb_FollowUp')].groupby(by=['year']).size()

make_plot(
    title_str="Follow-up Appointments",
    model=follow_up,
)

appointments = cons.loc[cons.TREATMENT_ID.str.contains('Tb_ScreeningAndRefer|Tb_FollowUp')].groupby(['TREATMENT_ID']).count()
treatments.plot.bar(width=0.4)
plt.title('Total Appointments')
plt.xticks([0, 1], ['Screening', 'Follow Up'], rotation=30, ha='right')
plt.xlabel('Appointments')
plt.ylabel('Total')
plt.show()
