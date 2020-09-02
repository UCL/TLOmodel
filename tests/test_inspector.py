import os
import time
import inspect  # not sure if required.
from pathlib import Path

import pytest

from tlo import Date, Simulation
from tlo.methods import (
    antenatal_care,
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthsystem,
    labour,
    newborn_outcomes,
    pregnancy_supervisor,
)

def test_something():
    assert (1 == 1)


if __name__ == '__main__':
    t0 = time.time()
    #simulation = simulation()
    #simulation.make_initial_population(n=popsize)
    #simulation.simulate(end_date=end_date)
    t1 = time.time()
    print('Time taken', t1 - t0)
    #test_dypes(simulation)
