"""
Run the HIV/TB modules with intervention coverage specified at national level
save outputs for plotting (file: output_plots_tb.py)
 """

import datetime
import pickle
# import random
from pathlib import Path
from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario
# # Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

# %% Run the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2010, 3, 1)
popsize = 10000
number_of_draws = 3
runs_per_draw = 3
# scenario = 0
# set up the log config
# add deviance measure logger if needed
log_config = {
    "filename": "impact_of_TB_diag_platforms",
    "directory": outputpath,
    "custom_levels": {
        "*": logging.WARNING,
        "tlo.methods.hiv": logging.INFO,
        "tlo.methods.tb": logging.INFO,
        "tlo.methods.demography": logging.INFO,
        "tlo.methods.healthsystem.summary": logging.INFO,
        "tlo.methods.labour.detail": logging.WARNING,  # this logger keeps outputting even when set to warning
    },
}

# Register the appropriate modules
# need to call epi before tb to get bcg vax
# seed = random.randint(0, 50000)
seed = 1821  # set seed for reproducibility

sim = Simulation(start_date=start_date, seed=seed, log_config=log_config, show_progress_bar=True)
sim.register(*fullmodel(
    resourcefilepath=resourcefilepath,
    use_simplified_births=False,
    module_kwargs={
        "SymptomManager": {"spurious_symptoms": True},
        "HealthSystem": {"disable": False,
                         "service_availability": ["*"],
                         "mode_appt_constraints": 0,  # no constraints, no squeeze factor
                         "cons_availability": "default",
                         #"cons_availability": ['default', 'none', 'all'],
                         "beds_availability": "all",
                         "ignore_priority": False,
                         "use_funded_or_actual_staffing": "funded_plus",
                         "capabilities_coefficient": 1.0},
    },
))


# to locally run the file use " tlo scenario-run src/scripts/hiv/projections_jan2023/analysis_full_modelv1.py"
# set the scenario
def draw_parameters(self, draw_number, rng):
    return {
        'HealthSystem': {'cons_availability': ['default', 'all', 'none'][draw_number]},
        'Tb': {
            'xpert': ['default', 'all', 'none'][draw_number],
            'xray': ['default', 'all', 'none'][draw_number],
            'sputum': ['default', 'all', 'none'][draw_number],
            'probability_community_chest_xray': [0.1][draw_number],
        }
    }
# Run the simulation and flush the logger
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# parse the results
output = parse_log_file(sim.log_filepath)

# save the results, argument 'wb' means write using binary mode. use 'rb' for reading file
with open(outputpath / "default_run.pickle", "wb") as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(dict(output), f, pickle.HIGHEST_PROTOCOL)

with open(outputpath / "default_run.pickle", "rb") as f:
    output = pickle.load(f)

if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
