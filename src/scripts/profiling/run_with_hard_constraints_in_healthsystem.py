"""
A run of the model that uses a lot of health system functionality -

* Allow appoints to be rolled over if not capacity and runs with reduced capacity

For use in profiling.
"""
from pathlib import Path

import pandas as pd
import shared

from tlo import Date, Simulation, logging
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
    labour,
    malaria,
    oesophagealcancer,
    pregnancy_supervisor,
    symptommanager,
)

# Key parameters about the simulation:
start_date = Date(2010, 1, 1)
end_date = start_date + pd.DateOffset(years=2)

popsize = 500_000

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
    healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                              mode_appt_constraints=2,
                              capabilities_coefficient=0.01
                              ),
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
shared.schedule_profile_log(sim)
sim.simulate(end_date=end_date)
