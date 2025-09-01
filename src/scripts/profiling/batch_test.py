"""
A run of the model at scale using all disease modules currently included in Master - including logging
* All logging
* Script including parsing logfile

For use in profiling.
"""
import os
import sys
from pathlib import Path

import pandas as pd
import shared

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    cardio_metabolic_disorders,
    demography,
    depression,
    diarrhoea,
    enhanced_lifestyle,
    epi,
    epilepsy,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    malaria,
    oesophagealcancer,
    other_adult_cancers,
    simplified_births,
    symptommanager,
)

seed_arg = sys.argv[1]

# Key parameters about the simulation:
start_date = Date(2010, 1, 1)
end_date = start_date + pd.DateOffset(years=2)

popsize = 2500

# The resource files
resourcefilepath = Path("./resources")

log_config = {
    "filename": f"batch_test_{seed_arg}",
    # Write log to ${AZ_BATCH_TASK_WORKING_DIR} if the variable exists,
    # otherwise to current directory
    "directory": os.getenv("AZ_BATCH_TASK_WORKING_DIR", ".") + "/outputs",
    "custom_levels": {"*": logging.INFO}
}

sim = Simulation(start_date=start_date, seed=int(seed_arg), log_config=log_config)

# Register the appropriate modules
sim.register(
    # Standard modules:
    demography.Demography(resourcefilepath=resourcefilepath),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
    symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
    healthburden.HealthBurden(resourcefilepath=resourcefilepath),

    # HealthSystem
    healthsystem.HealthSystem(resourcefilepath=resourcefilepath),

    # Modules for birth/labour/newborns --> Simplified Births
    simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),

    # Disease modules considered complete:
    depression.Depression(resourcefilepath=resourcefilepath),
    diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath),
    epi.Epi(resourcefilepath=resourcefilepath),
    epilepsy.Epilepsy(resourcefilepath=resourcefilepath),
    hiv.Hiv(resourcefilepath=resourcefilepath),
    malaria.Malaria(resourcefilepath=resourcefilepath),
    cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=resourcefilepath),
    oesophagealcancer.OesophagealCancer(resourcefilepath=resourcefilepath),
    other_adult_cancers.OtherAdultCancer(resourcefilepath=resourcefilepath)
)

# Run the simulation
sim.make_initial_population(n=popsize)
shared.schedule_profile_log(sim)
sim.simulate(end_date=end_date)
shared.print_checksum(sim)

# Parse the log-file
log_df = parse_log_file(sim.log_filepath)

print('TABLES:')
for k, v in log_df.items():
    print(f'{k}: {",".join(v.keys())}')
