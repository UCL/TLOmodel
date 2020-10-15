"""
A run of the model at scale using all disease modules currently included in Master - with no logging

For use in profiling.
"""

import datetime
from pathlib import Path

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
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
    symptommanager, oesophagealcancer, malaria, epi, epilepsy,
)

# Key parameters about the simulation:
start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 1000

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
    #
    # Disease modules:
    malaria.Malaria(resourcefilepath=resourcefilepath, testing=1),
    epi.Epi(resourcefilepath=resourcefilepath),
    depression.Depression(resourcefilepath=resourcefilepath),
    oesophagealcancer.OesophagealCancer(resourcefilepath=resourcefilepath),
    epilepsy.Epilepsy(resourcefilepath=resourcefilepath)
)

# Run the simulation
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)






