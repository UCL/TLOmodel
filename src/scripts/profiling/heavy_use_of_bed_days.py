"""This script heavily use the BedDays class and is used to improve performance of the BedDays class"""

import cProfile as cp
from pathlib import Path

import pandas as pd

from tlo import Date, Simulation
from tlo.methods import demography, healthsystem

resourcefilepath = 'resources'
outputpath = Path("./outputs")

start_date = Date(2010, 1, 1)
end_date = Date(2020, 1, 1)
popsize = 10000
days_sim = 1000
default_facility_id = 0
cap_bedtype1 = popsize

sim = Simulation(start_date=start_date, resourcefilepath=resourcefilepath)
sim.register(
    demography.Demography(),
    healthsystem.HealthSystem()
)
hs = sim.modules['HealthSystem']

# Update BedCapacity data defined in HealthSystem Module with a simple table:
hs.parameters['BedCapacity'] = pd.DataFrame(
    index=[0],
    data={
        'Facility_ID': default_facility_id,
        'bedtype1': cap_bedtype1,
    }
)


def impose_bd_footprint(person_id, dur_bed):
    """impose a footprint for a person for a particular duration"""
    hs.bed_days.impose_beddays_footprint(person_id=person_id, footprint={'bedtype1': dur_bed})


# Create the simulation
# end_date = start_date + pd.DateOffset(days=days_sim)
sim.make_initial_population(n=popsize)
cp.run('sim.simulate(end_date=end_date)', filename=outputpath/"bed_days_profiling.prof")

# For each day of the simulation impose a footprint lasting two days for each person
# (Will cause footprints to be continuously extended/re-evaluated)

for date in pd.date_range(start_date, end_date, freq='D'):
    sim.date = date

    for person_id in range(popsize):
        impose_bd_footprint(person_id=person_id, dur_bed=2)

assert 0 == hs.bed_days.bed_tracker['bedtype1'][default_facility_id].sum()
