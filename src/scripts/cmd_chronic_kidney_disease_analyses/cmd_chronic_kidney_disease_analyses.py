"""
Runs the CMD Chronic Kidney Disease module and produces the standard `compare_deaths` analysis to check
the number of deaths modelled against the GBD data.
"""

import matplotlib.pyplot as plt
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import compare_number_of_deaths, get_root_path
from tlo.methods import (
    cardio_metabolic_disorders,
    cmd_chronic_kidney_disease,
    demography,
    depression,
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

# The resource files
root = get_root_path()
resourcefilepath = root / "resources"

log_config = {
    "filename": "cmd_chronic_kidney_disease_analysis",
    "directory": root / "outputs",
    "custom_levels": {
        "*": logging.WARNING,
        "tlo.methods.demography": logging.INFO,
        "tlo.methods.healthburden": logging.INFO,
    }
}

# Set parameters for the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 1_000

def run_sim():
    # Establish the simulation object and set the seed
    sim = Simulation(start_date=start_date, log_config=log_config, resourcefilepath=resourcefilepath)

    # Register the appropriate modules
    sim.register(demography.Demography(),
                 simplified_births.SimplifiedBirths(),
                 enhanced_lifestyle.Lifestyle(),
                 healthsystem.HealthSystem(cons_availability='all'),
                 symptommanager.SymptomManager(),
                 healthseekingbehaviour.HealthSeekingBehaviour(),
                 healthburden.HealthBurden(),
                 cardio_metabolic_disorders.CardioMetabolicDisorders(),
                 cmd_chronic_kidney_disease.CMDChronicKidneyDisease(),
                 depression.Depression(),
                 hiv.Hiv(),
                 epi.Epi(),
                 tb.Tb(),
                 )

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    return sim.log_filepath

# Create df from simulation
logfile = run_sim()

CAUSE_NAME = 'Kidney Disease'

# With health system
comparison = compare_number_of_deaths(
    logfile=logfile,
    resourcefilepath=resourcefilepath
).rename(columns={"model": "model_with_healthsystem"})

# Without health system
no_hs = compare_number_of_deaths(
    logfile=logfile,
    resourcefilepath=resourcefilepath
)["model"]
no_hs.name = "model_no_healthsystem"

comparison = pd.concat([comparison, no_hs], axis=1)

comparison = comparison.loc[
    ("2010-2014", slice(None), slice(None), CAUSE_NAME)
].fillna(0.0)


fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(8, 6))

for ax, sex in zip(axs, ("M", "F")):
    comparison.loc[sex].plot(ax=ax)
    ax.set_ylabel("Deaths per year")
    ax.set_title(f"Sex: {sex}")

axs[-1].set_xlabel("Age group")

plt.tight_layout()
plt.show()
