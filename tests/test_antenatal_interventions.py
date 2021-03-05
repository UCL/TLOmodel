import pytest
import os
import pandas as pd
from pathlib import Path
from tlo import Date, Simulation, logging
from tlo.lm import LinearModel, LinearModelType
from tlo.methods.hsi_generic_first_appts import (
    HSI_GenericEmergencyFirstApptAtFacilityLevel1)

from tlo.methods import (
    antenatal_care,
    contraception,
    demography,
    depression,
    dx_algorithm_adult,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    malaria,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager,
)

seed = 560

log_config = {
    "filename": "pregnancy_supervisor_test",   # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # warning  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.contraception": logging.DEBUG,
        "tlo.methods.labour": logging.DEBUG,
        "tlo.methods.healthsystem": logging.FATAL,
        "tlo.methods.hiv": logging.FATAL,
        "tlo.methods.newborn_outcomes": logging.DEBUG,
        "tlo.methods.antenatal_care": logging.DEBUG,
        "tlo.methods.pregnancy_supervisor": logging.DEBUG,
        "tlo.methods.postnatal_supervisor": logging.DEBUG,
    }
}

# The resource files
try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = Path('./resources')

start_date = Date(2010, 1, 1)


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def set_all_women_as_pregnant(sim):
    """Force all women of reproductive age to be pregnant at the start of the simulation"""
    df = sim.population.props

    women_repro = df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)]
    df.loc[women_repro.index, 'is_pregnant'] = True
    df.loc[women_repro.index, 'date_of_last_pregnancy'] = start_date
    for person in women_repro.index:
        sim.modules['Labour'].set_date_of_labour(person)


def register_core_modules():
    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*']),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath))

    return sim


def register_all_modules():
    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           service_availability=['*']),
                 newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 malaria.Malaria(resourcefilepath=resourcefilepath),
                 hiv.Hiv(resourcefilepath=resourcefilepath),
                 dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
                 depression.Depression(resourcefilepath=resourcefilepath))

    return sim




""""@pytest.mark.group2
def test_abortion_complications_and_post_abortion_care():
    sim = register_core_modules()
    sim.make_initial_population(n=100)
    set_all_women_as_pregnant(sim)

    df = sim.population.props

    params = sim.modules['PregnancySupervisor'].parameters
    params['prob_sepsis_post_abortion'] = 1
    params['prob_haemorrhage_post_abortion'] = 1
    params['ps_linear_equations']['care_seeking_pregnancy_loss'] = \
        LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1)

    person_id = df.loc[df.is_pregnant].index[0]

    df.at[person_id, 'is_pregnant'] = False
    sim.modules['PregnancySupervisor'].abortion_complications.set([person_id], 'haemorrhage')
    sim.modules['PregnancySupervisor'].abortion_complications.set([person_id], 'sepsis')

    t = HSI_GenericEmergencyFirstApptAtFacilityLevel1(module=sim.modules['PregnancySupervisor'], person_id=person_id)
    t.apply(person_id=person_id, squeeze_factor=0.0)

    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
        isinstance(ev[1], antenatal_care.HSI_CareOfWomenDuringPregnancy_PostAbortionCaseManagement)
    ][0]

    # Run the event:
    event.apply(person_id=person_id, squeeze_factor=0.0)

    # Check that the person is now circumcised
    assert df.at[person_id, "ac_post_abortion_care_interventions"] > 0 """




