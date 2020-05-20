import os
import time
from pathlib import Path

import pytest

from tlo import Date, Simulation
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    healthburden,
    healthsystem,
    oesophageal_cancer,
)

start_date = Date(2010, 1, 1)
end_date = Date(2020, 1, 1)
popsize = 5000


@pytest.fixture(scope='module')
def simulation():
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim = Simulation(start_date=start_date)
    sim.seed_rngs(0)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           mode_appt_constraints=0))
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))

    sim.register(oesophageal_cancer.OesophagealCancer(resourcefilepath=resourcefilepath))

    return sim


def test_run(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)


def test_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


if __name__ == '__main__':
    t0 = time.time()
    simulation = simulation()
    test_run(simulation)
    t1 = time.time()
    print('Time taken', t1 - t0)


"""Other tests:

* that the ca_oesophagus and ca_oesophagus_any properties alwasy correspond - initiatio or after simulation

* that no one has oes_cancer at age less than 20 -- initiation or after simulation

* That the dates of thing in the properites are in right order where appropriate: ca_date_oes_cancer_diagnosis < ca_date_treatment_oesophageal_cancer < ca_date_palliative_care

* To check the working of the HSI, good shortcut would be to increase the baseline risk of cancer:

* at initiation, that date diagnosed is consistent with the age of the person

* at initiation, that not treatment or dianogsis for those with none stages

* check that treatment reduced risk of progression

* check that progression works:
"""
