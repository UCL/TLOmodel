"""
A run of the model with logging so as to allow for descriptions of overall Demography and Health Burden.

NB. HealthSystem is disabled and there are no spurious symptoms
"""
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    contraception,
    demography,
    depression,
    diarrhoea,
    dx_algorithm_adult,
    dx_algorithm_child,
    enhanced_lifestyle,
    epi,
    epilepsy,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    malaria,
    oesophagealcancer,
    pregnancy_supervisor,
    symptommanager,
)

# Define output path
outputpath = Path("./outputs")  # folder for convenience of storing outputs
datestamp = datetime.today().strftime("%Y_%m_%d")
results_filename_stub = 'long_run'
results_filename = outputpath / f"{datestamp}_{results_filename_stub}.pickle"

# Key parameters about the simulation:
start_date = Date(2010, 1, 1)
end_date = Date(2030, 12, 31)
pop_size = 20_000

# The resource files
rfp = Path("./resources")

log_config = {
    "filename": "for_profiling",
    "directory": "./outputs",
    "custom_levels": {
        "*": logging.WARNING,
        "tlo.methods.healthburden": logging.INFO,
        "tlo.methods.demography": logging.INFO
    }
}

sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

# Register the appropriate modules
sim.register(
    # Standard modules:
    demography.Demography(resourcefilepath=rfp),
    enhanced_lifestyle.Lifestyle(resourcefilepath=rfp),
    healthsystem.HealthSystem(resourcefilepath=rfp, disable=True),
    symptommanager.SymptomManager(resourcefilepath=rfp, spurious_symptoms=False),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=rfp),
    healthburden.HealthBurden(resourcefilepath=rfp),
    contraception.Contraception(resourcefilepath=rfp),
    labour.Labour(resourcefilepath=rfp),
    pregnancy_supervisor.PregnancySupervisor(resourcefilepath=rfp),
    dx_algorithm_child.DxAlgorithmChild(resourcefilepath=rfp),
    dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=rfp),
    #
    # Disease modules considered complete:
    diarrhoea.Diarrhoea(resourcefilepath=rfp),
    malaria.Malaria(resourcefilepath=rfp),
    hiv.Hiv(resourcefilepath=rfp),
    epi.Epi(resourcefilepath=rfp),
    depression.Depression(resourcefilepath=rfp),
    oesophagealcancer.OesophagealCancer(resourcefilepath=rfp),
    epilepsy.Epilepsy(resourcefilepath=rfp)
)


# Run the simulation
sim.make_initial_population(n=pop_size)
sim.simulate(end_date=end_date)

# %% Parse the log and pickle it:
output = parse_log_file(sim.log_filepath)
with open(results_filename, 'wb') as f:
    pickle.dump({'output': output}, f, pickle.HIGHEST_PROTOCOL)


