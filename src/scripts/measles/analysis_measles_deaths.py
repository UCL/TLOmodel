from pathlib import Path
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt
import os

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    contraception,
    healthburden,
    healthsystem,
    enhanced_lifestyle,
    dx_algorithm_child,
    healthseekingbehaviour,
    symptommanager,
    antenatal_care,
    labour,
    newborn_outcomes,
    pregnancy_supervisor,
    epi,
    measles
)

# The resource files
resourcefilepath = Path("./resources")

# store output files
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

start_date = Date(2010, 1, 1)
end_date = Date(2011, 12, 31)
popsize = 50000


# don't register epi as want population with no vaccination
def run_sim(service_availability=[]):
    # Establish the simulation object and set the seed
    # seed is not set - each simulation run gets a random seed
    sim = Simulation(start_date=start_date, seed=32,
                     log_config={"filename": "LogFile",
                                 'custom_levels': {"*": logging.WARNING, "tlo.methods.measles": logging.DEBUG,
                                                   "tlo.methods.demography": logging.INFO}
                                 }
                     )

    # Register the appropriate modules
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(
            resourcefilepath=resourcefilepath,
            service_availability=service_availability,
            mode_appt_constraints=0,
            ignore_cons_constraints=True,
            ignore_priority=True,
            capabilities_coefficient=1.0,
            disable=False,
        ),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        dx_algorithm_child.DxAlgorithmChild(),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        contraception.Contraception(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        labour.Labour(resourcefilepath=resourcefilepath),
        newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
        antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
        pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
        # epi.Epi(resourcefilepath=resourcefilepath),
        measles.Measles(resourcefilepath=resourcefilepath),
    )

    # Run the simulation and flush the logger
    sim.make_initial_population(n=popsize)

    sim.simulate(end_date=end_date)

    return sim.log_filepath


def get_summary_stats(logfile):
    output = parse_log_file(logfile)

    # incidence
    measles_output = output["tlo.methods.measles"]["incidence"]
    measles_output = measles_output.set_index('date')
    measles_inc = measles_output["inc_1000py"]

    # deaths
    deaths = output['tlo.methods.demography']['death'].copy()
    deaths = deaths.set_index('date')
    # limit to deaths due to measles
    to_drop = (deaths.cause != 'measles')
    deaths = deaths.drop(index=to_drop[to_drop].index)
    # count by month:
    deaths['month'] = deaths.index.month
    measles_deaths = deaths.groupby(by=['month']).size()

    return {
        'incidence': measles_inc,
        'deaths': measles_deaths,
    }


# ------------------------------------- MODEL OUTPUTS  ------------------------------------- #
# run 1
# run baseline with no vaccination to allow cases to occur
# disable all hsi to prevent any measles treatment
logfile_no_hs = run_sim(service_availability=[])

# run 2
# no vaccination
# allow hsi for measles treatment
logfile_hs = run_sim(service_availability=['*'])

results_no_hs = get_summary_stats(logfile_no_hs)
results_hs = get_summary_stats(logfile_hs)

# ------------------------------------- PLOTS  ------------------------------------- #
results_no_hs['deaths'].sum()
results_hs['deaths'].sum()
