import datetime
import os
import time
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
)

start_time = time.time()

# Where will output go
outputpath = Path("./outputs/longterm_trends")

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

start_date = Date(2010, 1, 1)
end_date = Date(2060, 12, 31)
popsize = 2000

log_config = {
    'filename': 'Lonterm_LogFile',
    'custom_levels': {"*": logging.WARNING,
                      "tlo.methods.epi": logging.INFO,
                      "tlo.methods.healthsystem.summary": logging.INFO,
                      "tlo.methods.demography": logging.INFO,}
}

# Establish the simulation object
sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

# ----- Control over the types of intervention that can occur -----
# Make a list that contains the treatment_id that will be allowed. Empty list means nothing allowed.
# '*' means everything. It will allow any treatment_id that begins with a stub (e.g. Mockitis*)
service_availability = ["*"]

# Register the appropriate modules
sim.register(
    demography.Demography(resourcefilepath=resourcefilepath),
    simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
    healthsystem.HealthSystem(
        resourcefilepath=resourcefilepath,
        service_availability=["*"],  # all treatment allowed
        mode_appt_constraints=0,  # o constraints,all HSI events run with no squeeze factor
        cons_availability="all",  # mode for consumable constraints (if ignored, all consumables available)
        ignore_priority=False,  # do not use the priority information in HSI event to schedule
        beds_availability='all', #all beds needs are met
        equip_availability='all', #all equipment needs are met
        capabilities_coefficient=1.0,  # multiplier for the capabilities of health officers
        use_funded_or_actual_staffing="funded",  # actual: use numbers/distribution of staff available currently
        disable=False,  # disables the healthsystem (no constraints and no logging) and every HSI runs
        disable_and_reject_all=False,  # disable healthsystem and no HSI runs
    ),
    symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
    healthburden.HealthBurden(resourcefilepath=resourcefilepath),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
    epi.Epi(resourcefilepath=resourcefilepath))

sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

print("--- %s seconds ---" % (time.time() - start_time))

# %% read the results

# ------------------------------------- MODEL OUTPUTS  ------------------------------------- #

output = parse_log_file(sim.log_filepath)
model_vax_coverage = output["tlo.methods.epi"]["ep_vaccine_coverage"]
model_date = pd.to_datetime(model_vax_coverage.date)
model_date = model_date.apply(lambda x: x.year)
