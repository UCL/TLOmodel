"""Script to run a simplified version of the simulation to look at just Measles Deaths in comparison to GBD estimates"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import compare_number_of_deaths
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    epi,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    measles,
    simplified_births,
    symptommanager,
)

outputpath = Path("./outputs")
resourcefilepath = Path("./resources")

start_date = Date(2010, 1, 1)
end_date = start_date + pd.DateOffset(years=10)
popsize = 50_000


def run_sim():
    """Returns path to logfile for the run"""
    sim = Simulation(
        start_date=start_date,
        seed=0,
        log_config={
            'filename': 'templogfile',
            'directory': outputpath,
            'custom_levels': {
                "*": logging.WARNING,
                'tlo.methods.demography': logging.INFO
                }
        }
    )

    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),

                 epi.Epi(resourcefilepath=resourcefilepath),
                 measles.Measles(resourcefilepath=resourcefilepath),

                 hiv.DummyHivModule(),
                 )

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    return sim.log_filepath


logfile = run_sim()

CAUSE_NAME = 'Measles'

comparison = compare_number_of_deaths(
    logfile=logfile, resourcefilepath=resourcefilepath).fillna(0.0)

fig, axs = plt.subplots(nrows=2, ncols=2, sharey=True, sharex=True)
for _col, sex in enumerate(('M', 'F')):
    for _row, period in enumerate(('2010-2014', '2015-2019')):
        ax = axs[_col][_row]
        comparison.loc[(period, sex, slice(None), CAUSE_NAME)].droplevel([0, 1, 3]).plot(use_index=True, ax=ax)
        ax.set_ylabel('Deaths per year')
        ax.set_title(f"{period}: {sex}")
        xticks = comparison.index.levels[2]
        ax.set_xticks(range(len(xticks)))
        ax.set_xticklabels(xticks, rotation=90)
fig.tight_layout()
fig.show()
