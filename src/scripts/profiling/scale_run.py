"""
A run of the model at scale using all disease modules currently included in Master - with no logging

For use in profiling.
"""

from pathlib import Path

import pandas as pd

from tlo import Date, Simulation, logging
from tlo.methods import (
    contraception,
    demography,
    depression,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    pregnancy_supervisor,
    symptommanager,
    oesophagealcancer,
    malaria,
    epi,
    epilepsy,
    dx_algorithm_adult, diarrhoea,
)

# Key parameters about the simulation:
start_date = Date(2010, 1, 1)
end_date = start_date + pd.DateOffset(years=1)

popsize = int(100)

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
    healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
    symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
    healthburden.HealthBurden(resourcefilepath=resourcefilepath),
    contraception.Contraception(resourcefilepath=resourcefilepath),
    labour.Labour(resourcefilepath=resourcefilepath),
    pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
    dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
    dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath),
    #
    # Disease modules considered complete:
    diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath),
    malaria.Malaria(resourcefilepath=resourcefilepath),
    epi.Epi(resourcefilepath=resourcefilepath),
    depression.Depression(resourcefilepath=resourcefilepath),
    oesophagealcancer.OesophagealCancer(resourcefilepath=resourcefilepath),
    epilepsy.Epilepsy(resourcefilepath=resourcefilepath)
)

# Run the simulation
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)






