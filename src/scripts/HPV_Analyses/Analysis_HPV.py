"""
Run the HPV modules
 """

import datetime
import pickle
import random
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file, extract_results
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    measles,
    simplified_births,
    symptommanager,
    hpv,
    hiv,
    tb
)

results_folder = Path("./outputs")

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = './resources'

# %% Run the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2020, 1, 1)
popsize = 2000


# set up the log config
log_config = {
    "filename": "test_runs",
    "directory": outputpath,
    "custom_levels": {
        "*": logging.WARNING,
        "tlo.methods.hpv": logging.INFO,
    },
}
#
# # Register the appropriate modules
# # need to call epi before tb to get bcg vax
seed = random.randint(0, 50000)
# # seed = 41728  # set seed for reproducibility
sim = Simulation(start_date=start_date, seed=seed, log_config=log_config,
                 show_progress_bar=True, resourcefilepath=resourcefilepath)
sim.register(
    demography.Demography(),
    simplified_births.SimplifiedBirths(),
    enhanced_lifestyle.Lifestyle(),
    healthsystem.HealthSystem(service_availability=["*"],  # all treatment allowed
        mode_appt_constraints=1,  # mode of constraints to do with officer numbers and time
        cons_availability="default",  # mode for consumable constraints (if ignored, all consumables available)
        ignore_priority=False,  # do not use the priority information in HSI event to schedule
        capabilities_coefficient=1.0,  # multiplier for the capabilities of health officers
        use_funded_or_actual_staffing="actual",  # actual: use numbers/distribution of staff available currently
        disable=False,  # disables the healthsystem (no constraints and no logging) and every HSI runs
        disable_and_reject_all=False,  # disable healthsystem and no HSI runs
    ),
    symptommanager.SymptomManager(),
    healthseekingbehaviour.HealthSeekingBehaviour(),
    healthburden.HealthBurden(),
    epi.Epi(),
    hpv.HPV(),
    measles.Measles(),
    hiv.Hiv(),
    tb.Tb(),
)
#
# # set the scenario
#sim.modules["HPV"].parameters["r_hpv"] = 0.01
#sim.modules["HPV"].parameters["r_hpv_clear"] = 0.6
#
# # Run the simulation and flush the logger
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)
#
# parse the results
output = parse_log_file(sim.log_filepath)

# save the results, argument 'wb' means write using binary mode. use 'rb' for reading file
with open(outputpath / "default_run.pickle", "wb") as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(dict(output), f, pickle.HIGHEST_PROTOCOL)

# load the results
with open(outputpath / "default_run.pickle", "rb") as f:
    output = pickle.load(f)

#Show the results
hpv_outputs = output["tlo.methods.hpv"]["summary"]
print(hpv_outputs)
# proportion_infected = extract_results(
#     results_folder,
#     module="tlo.methods.hpv",
#     key="summary",
#     column="PropInf",
#     do_scaling=False,
# )
#
# number_infected = extract_results(
#     results_folder,
#     module="tlo.methods.hpv",
#     key="summary",
#     column="TotalInf",
#     do_scaling=True,
# )

hpv_outputs = output["tlo.methods.hpv"]["summary"]
hpv_df = pd.DataFrame(hpv_outputs)

print(hpv_df)
print(hpv_df.columns)

# 1. Total infection
plt.figure(figsize=(8, 5))
plt.plot(hpv_df["Year"], hpv_df["TotalPrev"], marker="o")
plt.xlabel("Year")
plt.ylabel("Total HPV prevalence")
plt.title("Total HPV prevalence over time")
plt.grid(True)
plt.tight_layout()
plt.savefig(outputpath / "hpv_total_prevalence.png", dpi=300)
plt.show()

# 2. HPV  prevalence in different gender
plt.figure(figsize=(8, 5))
plt.plot(hpv_df["Year"], hpv_df["M_Prev"], marker="o", label="Male")
plt.plot(hpv_df["Year"], hpv_df["F_Prev"], marker="o", label="Female")
plt.xlabel("Year")
plt.ylabel("HPV prevalence")
plt.title("HPV prevalence by sex")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(outputpath / "hpv_prevalence_by_sex.png", dpi=300)
plt.show()

# 3. Prevalence of different HPV groups in female
plt.figure(figsize=(8, 5))
plt.plot(hpv_df["Year"], hpv_df["hr1_FemalePrev"], marker="o", label="hr1")
plt.plot(hpv_df["Year"], hpv_df["hr2_FemalePrev"], marker="o", label="hr2")
plt.plot(hpv_df["Year"], hpv_df["hr3_FemalePrev"], marker="o", label="hr3")
plt.xlabel("Year")
plt.ylabel("Prevalence")
plt.title("Female HPV prevalence by group")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(outputpath / "female_hpv_group_prevalence.png", dpi=300)
plt.show()

# 4. Prevalence of different HPV groups in male
plt.figure(figsize=(8, 5))
plt.plot(hpv_df["Year"], hpv_df["hr1_MalePrev"], marker="o", label="hr1")
plt.plot(hpv_df["Year"], hpv_df["hr2_MalePrev"], marker="o", label="hr2")
plt.plot(hpv_df["Year"], hpv_df["hr3_MalePrev"], marker="o", label="hr3")
plt.xlabel("Year")
plt.ylabel("Prevalence")
plt.title("Male HPV prevalence by group")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(outputpath / "male_hpv_group_prevalence.png", dpi=300)
plt.show()

