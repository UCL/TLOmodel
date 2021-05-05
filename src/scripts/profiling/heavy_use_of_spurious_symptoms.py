"""This profiling script runs a simulation that uses spurious symptoms that incurs a heavy use of the SymptomManager and
 the GenericAppts"""

from pathlib import Path

from pandas import DateOffset

from tlo import Date, Simulation
from tlo.methods import (
    demography,
    healthsystem,
    simplified_births,
    symptommanager,
)


resourcefilepath = Path("./resources")

start_date = Date(2010, 1, 1)
end_date = start_date + DateOffset(months=12)
popsize = 10_000

sim = Simulation(start_date=start_date, seed=0)

# Register the core modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
             symptommanager.SymptomManager(resourcefilepath=resourcefilepath, spurious_symptoms=True),
             healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable_and_reject_all=True)
             )

# Run the simulation
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# Time with updated implementation (n=10k, dur=1 year): 58.46860432624817 s

