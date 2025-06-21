"""
Runs the cervical cancer module and produces the standard `compare_deaths` analysis to check the number of deaths
modelled against the GBD data.

"""
from pathlib import Path

import matplotlib.pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import compare_number_of_deaths, get_root_path
from tlo.methods import (
    cervical_cancer,
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

# The resource files
root = get_root_path()
resourcefilepath = root / "resources"

log_config = {
    "filename": "cervical_cancer_analysis",
    "directory": root / "outputs",
    "custom_levels": {
        "*": logging.WARNING,
        "tlo.methods.demography": logging.INFO,
        "tlo.methods.healthburden": logging.INFO,
    }
}

# Set parameters for the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2020, 1, 1)
popsize = 17_000

def run_sim():
    # Establish the simulation object and set the seed
    sim = Simulation(start_date=start_date, log_config=log_config, resourcefilepath=resourcefilepath)

    # Register the appropriate modules
    sim.register(demography.Demography(),
                 cervical_cancer.CervicalCancer(),
                 simplified_births.SimplifiedBirths(),
                 enhanced_lifestyle.Lifestyle(),
                 healthsystem.HealthSystem(cons_availability='all'),
                 symptommanager.SymptomManager(),
                 healthseekingbehaviour.HealthSeekingBehaviour(),
                 healthburden.HealthBurden(),
                 epi.Epi(),
                 tb.Tb(),
                 hiv.Hiv(),
                 )

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    return sim.log_filepath

# Create df from simulation
logfile = run_sim()

CAUSE_NAME = 'Cancer (Cervix)'

comparison = compare_number_of_deaths(
    logfile=logfile, resourcefilepath=resourcefilepath).fillna(0.0)

fig, axs = plt.subplots(nrows=2, ncols=1, sharey=True, sharex=True)
sex = 'F'
for _row, period in enumerate(('2010-2014', '2015-2019')):
    ax = axs[_row]
    comparison.loc[(period, sex, slice(None), CAUSE_NAME)].plot(use_index=True, ax=ax)
    ax.set_ylabel('Deaths per year')
    ax.set_title(f"{period}: {sex}")
    xticks = comparison.index.levels[2]
    ax.set_xticks(range(len(xticks)))
    ax.set_xticklabels(xticks, rotation=90)
fig.tight_layout()
fig.show()
