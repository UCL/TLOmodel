import os
import time
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

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 1000


@pytest.fixture(scope='module')
def simulation():
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim = Simulation(start_date=start_date)
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 demography.Demography(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, mode_appt_constraints=0))

    sim.seed_rngs(1)
    return sim


def __check_properties(df):
    # Here we check none of the properties created in labour are set to True for males or under 15s
    assert not (((df.sex == 'M') | (df.age_years < 15)) & (~df.la_due_date_current_pregnancy.isna() |
                                                           df.la_currently_in_labour |
                                                           (df.la_current_labour_successful_induction != 'not_induced') |
                                                           df.la_intrapartum_still_birth | df.la_previous_cs_delivery |
                                                           df.la_has_previously_delivered_preterm |
                                                           df.la_obstructed_labour | df.la_obstructed_labour_disab |
                                                           df.la_antepartum_haem | df.la_antepartum_haem_disab |
                                                           df.la_uterine_rupture | df.la_uterine_rupture_disab |
                                                           df.la_sepsis | df.la_sepsis_disab | df.la_eclampsia |
                                                           df.la_eclampsia_disab | df.la_postpartum_haem |
                                                           df.la_postpartum_haem_disab | df.la_maternal_death |
                                                           (~df.la_maternal_death_date.isna()))).any()

    # Here we check that neither men nor under 15s can have a parity of >0
    assert not ((df.sex == 'M') & (df.la_parity > 0)).any()
    assert not ((df.age_years < 15) & (df.la_parity > 0)).any()


def test_make_initial_population(simulation):
    simulation.make_initial_population(n=popsize)


def test_initial_population(simulation):
    __check_properties(simulation.population.props)


def test_simulate(simulation):
    simulation.simulate(end_date=end_date)


def test_final_population(simulation):
    __check_properties(simulation.population.props)


def test_dypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    counter = 0
    for orig_type, df_type in zip(orig.dtypes, df.dtypes):
        counter += 1
        assert orig_type == df_type, f"column number {counter}\n - orig: {orig_type},  df: {df_type}"
    assert (df.dtypes == orig.dtypes).all()


if __name__ == '__main__':
    t0 = time.time()
    simulation = simulation()
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)
    t1 = time.time()
    print('Time taken', t1 - t0)
    test_dypes(simulation)
