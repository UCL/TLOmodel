import os
import pandas as pd
import datetime
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

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'


def test_configuration_of_properties():
    # --------------------------------------------------------------------------
    # Create and run a short but big population simulation for use in the tests
    sim = Simulation(start_date=Date(year=2010, month=1, day=1))

    # Register the appropriate modules
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))

    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(labour.Labour(resourcefilepath=resourcefilepath))
    sim.register(newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath))
    sim.register(antenatal_care.AntenatalCare(resourcefilepath=resourcefilepath))
    sim.register(pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))

    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath, mode_appt_constraints=0))

    sim.seed_rngs(0)
    sim.make_initial_population(n=5000)
    sim.simulate(end_date=Date(year=2013, month=1, day=1))
    # --------------------------------------------------------------------------

    # Check types of columns
    df = sim.population.props
    orig = sim.population.new_row
    assert (df.dtypes == orig.dtypes).all()

    # Check that no men or females not of reproductive age have properties altered
    df = sim.population.props
    assert not (df.sex == 'M' or df.age_year < 15) and (df.la_due_date_current_pregnancy != pd.NaT or
                                                        df.la_currently_in_labour or df.la_intrapartum_still_birth
                                                        or df.la_previous_cs_delivery
                                                        or df.la_has_previously_delivered_preterm or
                                                        df.la_obstructed_labour or df.la_obstructed_labour_disab
                                                        or df.la_antepartum_haem or df.la_antepartum_haem_disab or
                                                        df.la_uterine_rupture or df.la_uterine_rupture_disab or
                                                        df.la_sepsis or df.la_sepsis_postpartum or df.la_sepsis_disab or
                                                        df.la_eclampsia or df.la_eclampsia_postpartum or
                                                        df.la_eclampsia_disab or df.la_postpartum_haem or
                                                        df.la_postpartum_haem_disab or df.la_maternal_death or
                                                        df.la_maternal_death_date != pd.NaT)

    # Here we check that neither men nor under 15s can have a parity of >0
    assert not df.sex == 'M' and df.la_parity > 0
    assert not df.age_years < 15 and df.la_parity > 0


def test_configuration_of_mni():
    sim = Simulation(start_date=Date(year=2010, month=1, day=1))

    # Register the appropriate modules
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))

    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(labour.Labour(resourcefilepath=resourcefilepath))
    sim.register(newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath))
    sim.register(antenatal_care.AntenatalCare(resourcefilepath=resourcefilepath))
    sim.register(pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))

    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath, mode_appt_constraints=0))

    sim.seed_rngs(0)
    sim.make_initial_population(n=1000)
    sim.simulate(end_date=Date(year=2015, month=1, day=1))

    mni = sim.modules['Labour'].mother_and_newborn_info


test_configuration_of_properties()
test_configuration_of_mni()
