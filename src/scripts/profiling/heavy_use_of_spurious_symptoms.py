"""This profiling script runs a simulation that uses spurious symptoms that incurs a heavy use of the SymptomManager and
 the GenericAppts"""

from pathlib import Path

from pandas import DateOffset

from tlo import Date, Simulation, logging
from tlo.methods import demography, healthsystem, simplified_births, symptommanager

resourcefilepath = Path("./resources")

start_date = Date(2010, 1, 1)
end_date = start_date + DateOffset(months=12)
popsize = 10_000

sim = Simulation(start_date=start_date, seed=0,
                 log_config={'custom_levels': {'*': logging.FATAL}}, resourcefilepath=resourcefilepath)

# Register the core modules
sim.register(demography.Demography(),
             simplified_births.SimplifiedBirths(),
             symptommanager.SymptomManager(spurious_symptoms=True),
             healthsystem.HealthSystem(disable_and_reject_all=True)
             )

# Force the rate of symptom occurence to be high (on average once per week / 4 times per month)
sim.modules['SymptomManager'].parameters['generic_symptoms_spurious_occurrence'][[
    'prob_spurious_occurrence_in_children_per_day',
    'prob_spurious_occurrence_in_adults_per_day'
]] = 1.0 / 7.0

# Run the simulation
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# Time with updated implementation (n=10k, dur=1 year): 55.73144602775574 s
# Time with old implementation (n=10k, dur=1 year): too_long_to_wait!
