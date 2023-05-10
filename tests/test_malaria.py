import os
from pathlib import Path

import pandas as pd
import pytest

from tlo import Date, Simulation
from tlo.events import IndividualScopeEventMixin
from tlo.methods import (
    demography,
    diarrhoea,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    malaria,
    simplified_births,
    symptommanager,
)
from tlo.methods.healthsystem import HSI_Event

start_date = Date(2010, 1, 1)
end_date = Date(2015, 12, 31)
popsize = 500
seed = 10

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / "../resources"
except NameError:
    # running interactively
    resourcefilepath = "resources"


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


@pytest.fixture
def sim(seed):

    sim = Simulation(start_date=start_date, seed=seed)

    # Register the appropriate modules
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(
            resourcefilepath=resourcefilepath,
            disable=True,
        ),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        malaria.Malaria(resourcefilepath=resourcefilepath)
    )
    return sim


@pytest.mark.slow
def test_sims(sim):

    # Run the simulation and flush the logger
    sim.make_initial_population(n=popsize)
    check_dtypes(sim)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # IRS rates should be 0 or 0.8
    assert (sim.modules['Malaria'].itn_irs.irs_rate.isin([0, 0.8])).all()

    # check malaria deaths only being scheduled due to severe malaria (not clinical or asym)
    df = sim.population.props
    assert not (~df.ma_date_death.isna() & ~(df.ma_inf_type == "severe")).any()

    # check cases /  treatment are occurring
    assert not (df.ma_clinical_counter == 0).all()
    assert not (df.ma_date_tx == pd.NaT).all()

    # check clinical malaria in pregnancy counter not including males
    assert not any((df.sex == "M") & (df.ma_clinical_preg_counter > 0))

    # check symptoms are being assigned - fever assigned to all clinical cases
    for person in df.index[
        df.is_alive
        & (df.ma_inf_type == "clinical")
        & (sim.date >= df.ma_date_symptoms)
    ]:
        assert "fever" in sim.modules["SymptomManager"].has_what(person)
        assert "Malaria" in sim.modules["SymptomManager"].causes_of(person, "fever")

    # if symptoms due to malaria, check malaria properties correctly set
    for person in df.index[df.is_alive]:
        if "Malaria" in sim.modules["SymptomManager"].causes_of(person, "fever"):
            assert not pd.isnull(df.at[person, "ma_date_infected"])
            assert not df.at[person, "ma_inf_type"] == "none"

    # if infected with malaria, must have date_infected and infection_type
    for person in df.index[df.ma_is_infected]:
        assert not pd.isnull(df.at[person, "ma_date_infected"])
        assert not df.at[person, "ma_inf_type"] == "none"

    # if on treatment, must have treatment start date
    for person in df.index[df.ma_tx]:
        assert not pd.isnull(df.at[person, "ma_date_tx"])


# remove scheduled rdt testing and disable health system, should be no rdts and no treatment
# increase cfr for severe cases (all severe cases will die)
@pytest.mark.slow
def test_remove_malaria_test(seed):

    service_availability = list([" "])  # no treatments available

    end_date = Date(2018, 12, 31)
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the appropriate modules
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(
            resourcefilepath=resourcefilepath,
            service_availability=service_availability,
            mode_appt_constraints=0,
            cons_availability='all',
            ignore_priority=True,
            adopt_priority_policy=False,  # overwrite default
            capabilities_coefficient=0.0,
            disable=False,  # disables the health system constraints so all HSI events run
        ),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        malaria.Malaria(resourcefilepath=resourcefilepath)
    )
    # Run the simulation and flush the logger
    sim.make_initial_population(n=2000)

    # set testing adjustment to 0
    sim.modules['Malaria'].parameters['testing_adj'] = 0

    # increase death rate due to severe malaria
    sim.modules['Malaria'].parameters['cfr'] = 1.0

    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    df = sim.population.props

    # check no-one on malaria treatment
    assert not (df.ma_date_tx == pd.NaT).all()

    # Check malaria cases occurring
    assert 0 < df.ma_clinical_counter.sum()
    assert (df['ma_date_infected'] != pd.NaT).all()
    assert (df['ma_date_symptoms'] != pd.NaT).all()
    assert (df['ma_date_death'] != pd.NaT).all()

    # check all with severe malaria are assigned death date
    for person in df.index[(df.ma_inf_type == "severe")]:
        assert not pd.isnull(df.at[person, "ma_date_death"])

    # Check deaths are occurring
    assert (df.cause_of_death.loc[~df.is_alive & ~df.date_of_birth.isna()].isin({'severe_malaria', 'Malaria'})).any()

    # Check that those with a scheduled malaria death in the past, are now dead with a cause of death severe_malaria
    dead_due_to_malaria = ~df.is_alive & ~df.date_of_birth.isna() & df.cause_of_death.isin(
        {'severe_malaria', 'Malaria'})
    malaria_death_date_in_past = ~pd.isnull(df.ma_date_death) & (df.ma_date_death <= sim.date)
    assert (dead_due_to_malaria == malaria_death_date_in_past).all()


# test everyone regularly and check no treatment without positive rdt
@pytest.mark.slow
def test_schedule_rdt_for_all(sim):

    # Run the simulation and flush the logger
    sim.make_initial_population(n=popsize)
    sim.modules['Malaria'].parameters['testing_adj'] = 10
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    df = sim.population.props

    # check no treatment unless infected
    for person in df.index[df.ma_tx]:
        assert not pd.isnull(df.at[person, "ma_date_infected"])

    # check clinical counter is working
    assert sum(df["ma_clinical_counter"]) > 0


def test_dx_algorithm_for_malaria_outcomes(sim):
    """Create a person and check if the functions in dx_algorithm_child return the correct diagnosis"""

    def make_blank_simulation():
        popsize = 200  # smallest population size that works

        sim.make_initial_population(n=popsize)
        sim.modules['Malaria'].parameters['sensitivity_rdt'] = 1.0
        sim.simulate(end_date=start_date)

        # Create the HSI event that is notionally doing the call on diagnostic algorithm
        class DummyHSIEvent(HSI_Event, IndividualScopeEventMixin):
            def __init__(self, module, person_id):
                super().__init__(module, person_id=person_id)
                self.TREATMENT_ID = 'DummyHSIEvent'

                the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
                the_appt_footprint["Under5OPD"] = 1  # This requires one out patient

                self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
                self.ACCEPTED_FACILITY_LEVEL = '1a'
                self.ALERT_OTHER_DISEASES = []

            def apply(self, person_id, squeeze_factor):
                pass

        hsi_event = DummyHSIEvent(module=sim.modules['Malaria'], person_id=0)

        # check that the queue of events is empty
        assert 0 == len(sim.modules['HealthSystem'].HSI_EVENT_QUEUE)

        return sim, hsi_event

    # ----------------- Person with clinical malaria -----------------
    #
    # Set up the simulation:
    sim, hsi_event = make_blank_simulation()

    # Set up the person - clinical malaria and aged <5 years:
    df = sim.population.props
    df.at[0, 'ma_is_infected'] = True
    df.at[0, 'ma_date_infected'] = sim.date
    df.at[0, 'ma_date_symptoms'] = sim.date
    df.at[0, 'ma_inf_type'] = 'clinical'

    symptom_list = {"fever", "headache", "vomiting", "stomachache"}

    for symptom in symptom_list:
        # no symptom resolution
        sim.modules['SymptomManager'].change_symptom(
            person_id=0,
            symptom_string=symptom,
            disease_module=sim.modules['Malaria'],
            add_or_remove='+'
        )

    person_id = 0
    assert "fever" in sim.modules["SymptomManager"].has_what(person_id)

    assert sim.modules['Malaria'].check_if_fever_is_caused_by_malaria(
        person_id=0,
        hsi_event=hsi_event
    ) == "clinical_malaria"

    # ----------------- Person with severe malaria -----------------
    #
    # Set up the simulation:
    sim, hsi_event = make_blank_simulation()

    # Set up the person - clinical malaria and aged <5 years:
    df = sim.population.props
    person_id = 1

    df.at[person_id, 'ma_is_infected'] = True
    df.at[person_id, 'ma_date_infected'] = sim.date
    df.at[person_id, 'ma_date_symptoms'] = sim.date
    df.at[person_id, 'ma_inf_type'] = 'severe'

    symptom_list = {"fever", "headache", "vomiting", "stomachache"}

    for symptom in symptom_list:
        # no symptom resolution
        sim.modules['SymptomManager'].change_symptom(
            person_id=person_id,
            symptom_string=symptom,
            disease_module=sim.modules['Malaria'],
            add_or_remove='+'
        )

    assert "fever" in sim.modules["SymptomManager"].has_what(person_id)

    assert sim.modules['Malaria'].check_if_fever_is_caused_by_malaria(
        person_id=person_id,
        hsi_event=hsi_event
    ) == "severe_malaria"


# check non-malarial fever returns correct diagnosis string (and no malaria treatment)
def test_dx_algorithm_for_non_malaria_outcomes(seed):
    """Create a person and check if the functions in dx_algorithm_child return the correct diagnosis"""

    def make_blank_simulation():
        popsize = 200  # smallest population size that works

        sim = Simulation(start_date=start_date, seed=seed)

        # Register the appropriate modules
        sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                     simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                     enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                     healthsystem.HealthSystem(
                         resourcefilepath=resourcefilepath,
                         disable=True,  # disables the health system constraints so all HSI events run
                     ),
                     malaria.Malaria(resourcefilepath=resourcefilepath),
                     symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                     healthseekingbehaviour.HealthSeekingBehaviour(
                         resourcefilepath=resourcefilepath,
                         force_any_symptom_to_lead_to_healthcareseeking=True
                         # every symptom leads to health-care seeking
                     ),
                     healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                     diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath),

                     # Supporting modules:
                     diarrhoea.DiarrhoeaPropertiesOfOtherModules()
                     )

        sim.make_initial_population(n=popsize)
        sim.simulate(end_date=start_date)

        # Create the HSI event that is notionally doing the call on diagnostic algorithm
        class DummyHSIEvent(HSI_Event, IndividualScopeEventMixin):
            def __init__(self, module, person_id):
                super().__init__(module, person_id=person_id)
                self.TREATMENT_ID = 'DummyHSIEvent'

                the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
                the_appt_footprint["Under5OPD"] = 1  # This requires one out patient

                self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
                self.ACCEPTED_FACILITY_LEVEL = '1a'
                self.ALERT_OTHER_DISEASES = []

            def apply(self, person_id, squeeze_factor):
                pass

        # assume diarrhoea has created a fever (as an example of a non-malarial fever)
        hsi_event = DummyHSIEvent(module=sim.modules['Diarrhoea'], person_id=0)

        # check that the queue of events is empty
        assert 0 == len(sim.modules['HealthSystem'].HSI_EVENT_QUEUE)

        return sim, hsi_event

    # ----------------- Person with fever (but not malaria) -----------------
    #
    # Set up the simulation:
    sim, hsi_event = make_blank_simulation()

    # Set up the person with fever caused by Diarrhoea and no malaria
    person_id = 0
    sim.population.props.loc[person_id, ["ma_is_infected", " ma_inf_type"]] = (False, "none")

    sim.modules['SymptomManager'].change_symptom(
        person_id=person_id,
        symptom_string="fever",
        disease_module=sim.modules['Diarrhoea'],
        add_or_remove='+'
    )

    assert "fever" in sim.modules["SymptomManager"].has_what(person_id)

    assert sim.modules['Malaria'].check_if_fever_is_caused_by_malaria(
        person_id=0,
        hsi_event=hsi_event
    ) == "negative_malaria_test"


def test_severe_malaria_deaths_perfect_treatment(sim):

    # -------------- Perfect treatment for severe malaria -------------- #
    # set perfect treatment for severe malaria cases - no deaths should occur
    sim.modules['Malaria'].parameters['treatment_adjustment'] = 0

    # Run the simulation and flush the logger
    sim.make_initial_population(n=10)
    # simulate for 0 days, just get everything set up (dxtests etc)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props

    # select person and assign severe malaria
    person_id = 0
    df.loc[person_id, ["ma_is_infected", "ma_inf_type"]] = (True, "severe")

    # put person on treatment
    treatment_appt = malaria.HSI_Malaria_complicated_treatment_adult(person_id=person_id,
                                                                     module=sim.modules['Malaria'])
    treatment_appt.apply(person_id=person_id, squeeze_factor=0.0)
    assert df.at[person_id, 'ma_tx']
    assert df.at[person_id, "ma_date_tx"] == sim.date
    assert df.at[person_id, "ma_tx_counter"] > 0

    # run the death event
    death_event = malaria.MalariaDeathEvent(
        module=sim.modules['Malaria'], individual_id=person_id, cause="Malaria")
    death_event.apply(person_id)

    # should not cause death but result in cure
    assert df.at[person_id, 'is_alive']
    assert df.at[person_id, "ma_inf_type"] == "none"


def test_severe_malaria_deaths_treatment_failure(sim):
    # -------------- treatment failure for severe malaria -------------- #

    # set treatment with zero efficacy for severe malaria cases - death should occur
    sim.modules['Malaria'].parameters['treatment_adjustment'] = 1

    # Run the simulation and flush the logger
    sim.make_initial_population(n=10)
    # simulate for 0 days, just get everything set up (dxtests etc)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props

    # select person and assign severe malaria
    person_id = 0
    df.loc[person_id, ["ma_is_infected", "ma_inf_type"]] = (True, "severe")

    # put person on treatment
    treatment_appt = malaria.HSI_Malaria_complicated_treatment_adult(person_id=person_id,
                                                                     module=sim.modules['Malaria'])
    treatment_appt.apply(person_id=person_id, squeeze_factor=0.0)
    assert df.at[person_id, 'ma_tx']
    assert df.at[person_id, "ma_date_tx"] == sim.date
    assert df.at[person_id, "ma_tx_counter"] > 0

    # run the death event
    death_event = malaria.MalariaDeathEvent(
        module=sim.modules['Malaria'], individual_id=person_id, cause="Malaria")
    death_event.apply(person_id)

    # should cause death - no cure
    assert not df.at[person_id, 'is_alive']
    assert df.at[person_id, 'cause_of_death'] == "Malaria"

    # -------------- no treatment for severe malaria -------------- #
    # set treatment with zero efficacy for severe malaria cases - death should occur
    # select person and assign severe malaria
    person_id = 1
    df.loc[person_id, ["ma_is_infected", "ma_inf_type"]] = (True, "severe")

    assert not df.at[person_id, 'ma_tx']
    assert df.at[person_id, "ma_date_tx"] is pd.NaT
    assert df.at[person_id, "ma_tx_counter"] == 0

    # run the death event
    death_event = malaria.MalariaDeathEvent(
        module=sim.modules['Malaria'], individual_id=person_id, cause="Malaria")
    death_event.apply(person_id)

    # should cause death - no cure
    assert not df.at[person_id, 'is_alive']
    assert df.at[person_id, 'cause_of_death'] == "Malaria"
