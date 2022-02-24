"""
Run the HIV/TB modules with intervention coverage specified at national level
run one sim for each scenario
save outputs for plotting (file: output_plots_tb.py)
 """

import datetime
from pathlib import Path
import pickle


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

# %% Set parameters for all sims
start_date = Date(2010, 1, 1)
end_date = Date(2025, 1, 1)
popsize = 10000

scenario_dict = [0, 1, 2, 3, 4]

for scenario in scenario_dict:

    scenario = scenario
    filename = "scenario" + str(scenario)

    # set up the log config
    log_config = {
        "filename": filename,
        "directory": outputpath,
        "custom_levels": {
            "*": logging.WARNING,
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
    )

    # set scenario
    sim.modules["Tb"].parameters["scenario"] = scenario

    # Run the simulation and flush the logger
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # parse the results
    output = parse_log_file(sim.log_filepath)
    # save the results, argument 'wb' means write using binary mode. use 'rb' for reading file
    pickle_name = filename + ".pickle"
    with open(outputpath / pickle_name, "wb") as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)

