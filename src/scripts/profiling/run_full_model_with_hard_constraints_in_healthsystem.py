"""
A run of the model that uses a lot of health system functionality -

* Allow appoints to be rolled over if not capacity and runs with reduced capacity
* Has persons attending care due to spurious symptoms

NB. Use the SimplifiedBirths module instead of the set of modules of pregnancy/labour/newborn outcomes.

For use in profiling.
"""

from pathlib import Path

import pandas as pd
import shared

from tlo import Date, Simulation, logging
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
    simplified_births,
    symptommanager,
)
from tlo.methods.cancer_modules import oesophagealcancer, other_adult_cancers

# Key parameters about the simulation:
start_date = Date(2010, 1, 1)
end_date = start_date + pd.DateOffset(years=2)

popsize = 20_000

# The resource files
resourcefilepath = Path("./resources")

log_config = {
    "filename": "for_profiling",
    "directory": "./outputs",
    "custom_levels": {"*": logging.WARNING}
}

sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

# Register the appropriate modules
sim.register(
    # Standard modules:
    demography.Demography(resourcefilepath=resourcefilepath),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
    symptommanager.SymptomManager(resourcefilepath=resourcefilepath, spurious_symptoms=True),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
    healthburden.HealthBurden(resourcefilepath=resourcefilepath),

    # HealthSystem
    healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                              mode_appt_constraints=2,
                              capabilities_coefficient=0.01
                              ),

    # Modules for birth/labour/newborns --> Simplified Births
    simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),

    # Disease modules considered complete:
    cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=resourcefilepath),
    depression.Depression(resourcefilepath=resourcefilepath),
    diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath),
    epi.Epi(resourcefilepath=resourcefilepath),
    epilepsy.Epilepsy(resourcefilepath=resourcefilepath),
    hiv.Hiv(resourcefilepath=resourcefilepath),
    malaria.Malaria(resourcefilepath=resourcefilepath),
    oesophagealcancer.OesophagealCancer(resourcefilepath=resourcefilepath),
    other_adult_cancers.OtherAdultCancer(resourcefilepath=resourcefilepath)
)

# Run the simulation
sim.make_initial_population(n=popsize)
shared.schedule_profile_log(sim)
sim.simulate(end_date=end_date)
shared.print_checksum(sim)
