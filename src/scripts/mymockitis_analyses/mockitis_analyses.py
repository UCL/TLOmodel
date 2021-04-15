from pathlib import Path

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    contraception,
    demography,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    symptommanager,
)

# The Resource files [NB. Working directory must be set to the root of TLO: TLOmodel]
resourcefilepath = Path('./resources')

# Establish the simulation object
start_date = Date(year=2010, month=1, day=1)
end_date = Date(year=2010, month=12, day=31)
popsize = 2000
log_config = {'filename': 'LogFile'}

sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

# Make all services available:
service_availability = ['*']

# Register the appropriate 'core' modules
sim.register(
    demography.Demography(resourcefilepath=resourcefilepath),
    contraception.Contraception(resourcefilepath=resourcefilepath),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
    healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                              service_availability=service_availability,
                              mode_appt_constraints=2,
                              capabilities_coefficient=1.0,
                              ignore_cons_constraints=False,
                              disable=False),
    # (NB. will run much faster with disable=True in the declaration of the HealthSystem)
    symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
    dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
    healthburden.HealthBurden(resourcefilepath=resourcefilepath),
    # Register disease modules of interest
    # ....
)

# Run the simulation
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# Read the results
output = parse_log_file(sim.log_filepath)
