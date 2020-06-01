from pathlib import Path

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    chronicsyndrome,
    contraception,
    demography,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    mockitis,
    symptommanager,
    rti
)
import os

# The Resource files [NB. Working directory must be set to the root of TLO: TLOmodel]
resourcefilepath = Path('./resources')

# Establish the simulation object
start_date = Date(year=2010, month=1, day=1)
end_date = Date(year=2010, month=12, day=31)
popsize = 2000

sim = Simulation(start_date=start_date)
logfile = sim.configure_logging(filename="LogFile")
# if os.path.exists(logfile):
#     os.remove(logfile)
# Make all services available:
service_availability = ['*']
logging.getLogger('tlo.methods.RTI').setLevel(logging.DEBUG)

# Register the appropriate 'core' modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
# sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                       service_availability=service_availability,
                                       mode_appt_constraints=2,
                                       capabilities_coefficient=1.0,
                                       ignore_cons_constraints=False,
                                       disable=False))
# (NB. will run much faster with disable=True in the declaration of the HealthSystem)
sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
sim.register(healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))
sim.register(dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath))
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))

# Register disease modules of interest:
sim.register(rti.RTI(resourcefilepath=resourcefilepath))

# custom_levels = {
#     # '*': logging.CRITICAL,  # disable logging for all modules
#     'tlo.methods.RTI': logging.INFO,  # enable logging at INFO level
#     'tlo.methods.RTI': logging.DEBUG
#                   }

# Run the simulation
sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# Read the results
output = parse_log_file(logfile)
