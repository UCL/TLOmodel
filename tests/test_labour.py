import os
import pandas as pd
import time
from pathlib import Path

import pytest

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
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
popsize = 5000

outputpath = Path("./outputs")  # folder for convenience of storing outputs


@pytest.fixture(scope='module')
def simulation():

    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

    sim = Simulation(start_date=start_date)

    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))

    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(labour.Labour(resourcefilepath=resourcefilepath))
    sim.register(newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath))
    sim.register(antenatal_care.AntenatalCare(resourcefilepath=resourcefilepath))
    sim.register(pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))

    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath, mode_appt_constraints=0))

    # sim.configure_logging('log', directory=tmpdir, custom_levels={'*': logging.WARNING,
    #                                                               'tlo.module.labour': logging.DEBUG,
    #                                                              'tlo.module.newborn_outcomes': logging.DEBUG,
    #                                                              'tlo.module.pregnancy_supervisor': logging.DEBUG,
    #                                                              'tlo.module.antenatal_care': logging.DEBUG})

    sim.seed_rngs(1)
    return sim


def test_run(simulation):

    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)


def __check_properties(df):
    # Here we check none of the properties created in labour are set to True for males or under 15s
    assert not (df.sex == 'M' or df.age_year < 15) and (df.la_due_date_current_pregnancy != pd.NaT or
                                                        df.la_currently_in_labour or
                                                        df.la_current_labour_successful_induction != 'none' or
                                                        df.la_intrapartum_still_birth or df.la_previous_cs_delivery
                                                        or df.la_has_previously_delivered_preterm or
                                                        df.la_obstructed_labour or df.la_obstructed_labour_disab
                                                        or df.la_antepartum_haem or df.la_antepartum_haem_disab or
                                                        df.la_uterine_rupture or df.la_uterine_rupture_disab or
                                                        df.la_sepsis or df.la_sepsis_disab or df.la_eclampsia or
                                                        df.la_eclampsia_disab or df.la_postpartum_haem or
                                                        df.la_postpartum_haem_disab or df.la_maternal_death or
                                                        df.la_maternal_death_date != pd.NaT)

    # Here we check that neither men nor under 15s can have a parity of >0
    assert not df.sex == 'M' and df.la_parity > 0
    assert not df.age_years < 15 and df.la_parity > 0


# def test_make_initial_population(simulation):
#    simulation.make_initial_population(n=popsize)


# def test_initial_population(simulation):
#    __check_properties(simulation.population.props)


# def test_simulate(simulation):
#    simulation.simulate(end_date=end_date)


# def test_final_population(simulation):
#    __check_properties(simulation.population.props)


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
    test_run(simulation)
    t1 = time.time()
    print('Time taken', t1 - t0)
    test_dypes(simulation)
