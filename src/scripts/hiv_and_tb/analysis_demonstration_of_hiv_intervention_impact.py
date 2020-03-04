
from pathlib import Path

import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthsystem,
    hiv,
    malecircumcision,
    symptommanager,
    tb,
)

start_date = Date(2010, 1, 1)
end_date = Date(2012, 12, 31)
popsize = 100


# %% Define some helper functions to run and analyse the model

def run_simulation_with_set_intv_parameters(
    fsw_prep=0.0,
    initial_art_coverage=0.0,
    treatment_prob=0.0,
    hv_behav_mod=0.0,
    testing_adj=0.0
):
    """
    This helper function will run a simulation with a given set of paramerers for the HIV interventions,
    over-writing the parameters that are normally imported. It returns the path of the logfile for the simulation
    that is created.

    "fsw_prep": Parameter(Types.REAL, "prob of fsw receiving PrEP"): 0.1
    "initial_art_coverage": Parameter(Types.REAL, "coverage of ART at baseline"): <<table of values>>
    "treatment_prob": Parameter(Types.REAL, "probability of requesting ART following positive HIV test"): 0.3
    "hv_behav_mod": Parameter(Types.REAL, "change in force of infection with behaviour modification"): 0.5
    "testing_adj": Parameter(Types.REAL, "additional HIV testing outside generic appts"): 0.05
    """

    # Define paths
    outputpath = Path("./outputs")
    resourcefilepath = Path("./resources")

    # Create simulation and register the appropriate modules
    sim = Simulation(start_date=start_date)
    sim.seed_rngs(0)
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(
        healthsystem.HealthSystem(
            resourcefilepath=resourcefilepath,
            disable=True,
        )
    )
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
    sim.register(hiv.Hiv(resourcefilepath=resourcefilepath))
    sim.register(tb.Tb(resourcefilepath=resourcefilepath))
    sim.register(malecircumcision.MaleCircumcision(resourcefilepath=resourcefilepath))

    # Overwrite the parameters in the modules:
    sim.modules['Hiv'].parameters['fsw_prep'] = fsw_prep
    sim.modules['Hiv'].parameters['initial_art_coverage']['prop_coverage'] = initial_art_coverage
    sim.modules['Hiv'].parameters['treatment_prob'] = treatment_prob
    sim.modules['Hiv'].parameters['hv_behav_mod'] = hv_behav_mod
    sim.modules['Hiv'].parameters['testing_adj'] = testing_adj

    # Sets all modules to WARNING threshold, then alters hiv and tb to INFO
    custom_levels = {
        "*": logging.WARNING,
        "tlo.methods.hiv": logging.INFO,
        "tlo.methods.tb": logging.INFO,
        "tlo.method.malecircumcision": logging.INFO,
        "tlo.methods.demography": logging.INFO,
    }
    # configure_logging automatically appends datetime
    logfile = sim.configure_logging(filename="LogFile", custom_levels=custom_levels)

    # Run the simulation and flush the logger
    sim.seed_rngs(0)
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    return logfile


def get_key_outputs(logfile):
    """
    This helper function accepts the path of a logfile and computes the following key metrics for each year:
    * Number of new HIV infections
    * Numbers of AIDS deaths
    * Coverage of ART among PLHIV
    * Coverage of ever been diagnosed with HIV among PLHIV
    * Coverage of PrEP among FSW
    """
    output = parse_log_file(logfile)
    r = dict()  # processed results

    def make_year_the_index(df):
        df.set_index(pd.to_datetime(df['date']).dt.year, drop=True, inplace=True)

    # New infections
    make_year_the_index(output['tlo.methods.hiv']['hiv_treatment'])
    r['num_new_infections_15_to_49'] = output['tlo.methods.hiv']['hiv_infected']['num_new_infections_15_to_49']
    r['num_new_infections_0_to_14'] = output['tlo.methods.hiv']['hiv_infected']['num_new_infections_0_to_14']

    # Adult ART coverage
    make_year_the_index(output['tlo.methods.hiv']['hiv_treatment'])
    r['adult_art_cov'] = output['tlo.methods.hiv']['hiv_treatment']['hiv_coverage_adult_art']


    # TODO get other outputs



# %% Run the simulations:

logfile = run_simulation_with_set_intv_parameters()


# %% Make some graphs of the key outputs across the scenarios:

# TODO: Produce basic plots across scenarios

