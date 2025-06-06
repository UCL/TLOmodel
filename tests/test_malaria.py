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
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    malaria,
    simplified_births,
    symptommanager,
    tb,
)
from tlo.methods.hsi_event import HSI_Event

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

# Create the HSI event that is notionally doing the call on diagnostic algorithm
class DummyHSIEvent(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        self.TREATMENT_ID = "DummyHSIEvent"

        the_appt_footprint = self.sim.modules[
            "HealthSystem"
        ].get_blank_appt_footprint()
        the_appt_footprint["Under5OPD"] = 1  # This requires one out patient

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = "1a"
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        pass


@pytest.fixture
def sim(seed):
    sim = Simulation(start_date=start_date, seed=seed, resourcefilepath=resourcefilepath)

    # Register the appropriate modules
    sim.register(
        demography.Demography(),
        healthsystem.HealthSystem(disable=True),
        simplified_births.SimplifiedBirths(),
        symptommanager.SymptomManager(),
        healthseekingbehaviour.HealthSeekingBehaviour(),
        healthburden.HealthBurden(),
        enhanced_lifestyle.Lifestyle(),
        malaria.Malaria(),
        tb.Tb(),
        hiv.Hiv(),
        epi.Epi(),
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
    for person in df.index[df.ma_tx.isin(["uncomplicated", "complicated"])]:
        assert not pd.isnull(df.at[person, "ma_date_tx"])

# remove scheduled rdt testing and disable health system, should be no rdts and no treatment
# increase cfr for severe cases (all severe cases will die)
@pytest.mark.slow
def test_remove_malaria_test(seed):
    service_availability = list([" "])  # no treatments available

    end_date = Date(2018, 12, 31)
    sim = Simulation(start_date=start_date, seed=seed, resourcefilepath=resourcefilepath)

    # Register the appropriate modules
    sim.register(
        demography.Demography(),
        healthsystem.HealthSystem(service_availability=service_availability,
            mode_appt_constraints=0,
            cons_availability='all',
            ignore_priority=True,
            capabilities_coefficient=0.0,
            disable=False,  # disables the health system constraints so all HSI events run
        ),
        simplified_births.SimplifiedBirths(),
        symptommanager.SymptomManager(),
        healthseekingbehaviour.HealthSeekingBehaviour(),
        healthburden.HealthBurden(),
        enhanced_lifestyle.Lifestyle(),
        malaria.Malaria(),
        tb.Tb(),
        hiv.Hiv(),
        epi.Epi(),
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
    sim.modules['Malaria'].parameters['prob_malaria_case_tests'] = 10
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    df = sim.population.props

    # check no treatment unless infected
    for person in df.index[df.ma_tx.isin(["uncomplicated", "complicated"])]:
        assert not pd.isnull(df.at[person, "ma_date_infected"])

    # check clinical counter is working
    assert sum(df["ma_clinical_counter"]) > 0

@pytest.fixture
def setup_simulation_for_dx_algorithm_test(sim):
    popsize = 200  # smallest population size that works

    sim.make_initial_population(n=popsize)
    sim.modules['Malaria'].parameters['sensitivity_rdt'] = 1.0
    sim.simulate(end_date=start_date)

    # Check that the queue of events is empty
    assert 0 == len(sim.modules['HealthSystem'].HSI_EVENT_QUEUE)
    # Run wrapped test
    yield


@pytest.mark.usefixtures("setup_simulation_for_dx_algorithm_test")
@pytest.mark.parametrize(
    "ma_inf_type, expected_diagnosis",
    [
        pytest.param("clinical", "clinical_malaria", id="Clinical diagnosis"),
        pytest.param("severe", "severe_malaria", id="Severe diagnosis"),
    ],
)
def test_dx_algorithm_for_malaria_outcomes_clinical(
    sim,
    ma_inf_type: str,
    expected_diagnosis: str,
    person_id: int = 0,
):
    """
    Create a person with clinical malaria and check if the functions in
    dx_algorithm_child return the correct diagnosis.
    """
    # Set up the simulation:
    hsi_event = DummyHSIEvent(module=sim.modules["Malaria"], person_id=person_id)

    # Set up the person - clinical malaria and aged <5 years:
    df = sim.population.props
    df.at[person_id, 'ma_is_infected'] = True
    df.at[person_id, 'ma_date_infected'] = sim.date
    df.at[person_id, 'ma_date_symptoms'] = sim.date
    df.at[person_id, 'ma_inf_type'] = ma_inf_type

    symptom_list = {"fever", "headache", "vomiting", "stomachache"}

    for symptom in symptom_list:
        # no symptom resolution
        sim.modules['SymptomManager'].change_symptom(
            person_id=person_id,
            symptom_string=symptom,
            disease_module=sim.modules['Malaria'],
            add_or_remove='+'
        )

    assert "fever" in sim.modules["SymptomManager"].has_what(person_id=person_id)

    def diagnosis_function(tests, use_dict: bool = False, report_tried: bool = False):
        return hsi_event.healthcare_system.dx_manager.run_dx_test(
            tests,
            hsi_event=hsi_event,
            use_dict_for_single=use_dict,
            report_dxtest_tried=report_tried,
        )

    assert sim.modules['Malaria'].check_if_fever_is_caused_by_malaria(
        true_malaria_infection_type = df.at[person_id, "ma_inf_type"],
        diagnosis_function = diagnosis_function,
        person_id=person_id,
    ) == expected_diagnosis


# check non-malarial fever returns correct diagnosis string (and no malaria treatment)
def test_dx_algorithm_for_non_malaria_outcomes(seed):
    """Create a person and check if the functions in dx_algorithm_child return the correct diagnosis"""

    def make_blank_simulation():
        popsize = 200  # smallest population size that works

        sim = Simulation(start_date=start_date, seed=seed, resourcefilepath=resourcefilepath)

        # Register the appropriate modules
        sim.register(demography.Demography(),
                     simplified_births.SimplifiedBirths(),
                     enhanced_lifestyle.Lifestyle(),
                     healthsystem.HealthSystem(
                         disable=True,  # disables the health system constraints so all HSI events run
                     ),
                     malaria.Malaria(),
                     symptommanager.SymptomManager(),
                     healthseekingbehaviour.HealthSeekingBehaviour(
                         force_any_symptom_to_lead_to_healthcareseeking=True
                         # every symptom leads to health-care seeking
                     ),
                     healthburden.HealthBurden(),
                     diarrhoea.Diarrhoea(),

                     # Supporting modules:
                     diarrhoea.DiarrhoeaPropertiesOfOtherModules(),
                     tb.Tb(),
                     hiv.Hiv(),
                     epi.Epi(),
                     ),

        sim.make_initial_population(n=popsize)
        sim.simulate(end_date=start_date)

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

    assert "fever" in sim.modules["SymptomManager"].has_what(person_id=person_id)

    def diagnosis_function(tests, use_dict: bool = False, report_tried: bool = False):
        return hsi_event.healthcare_system.dx_manager.run_dx_test(
            tests,
            hsi_event=hsi_event,
            use_dict_for_single=use_dict,
            report_dxtest_tried=report_tried,
        )

    assert (
        sim.modules["Malaria"].check_if_fever_is_caused_by_malaria(
            true_malaria_infection_type=sim.population.props.at[
                person_id, "ma_inf_type"
            ],
            diagnosis_function=diagnosis_function,
            person_id=person_id,
        )
        == "negative_malaria_test"
    )


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
    treatment_appt = malaria.HSI_Malaria_Treatment_Complicated(person_id=person_id,
                                                               module=sim.modules['Malaria'])
    treatment_appt.apply(person_id=person_id, squeeze_factor=0.0)
    assert df.at[person_id, "ma_tx"] != "none"
    assert df.at[person_id, "ma_date_tx"] == sim.date
    assert df.at[person_id, "ma_tx_counter"] > 0

    # run the death event
    death_event = malaria.MalariaDeathEvent(
        module=sim.modules['Malaria'], person_id=person_id, cause="Malaria")
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
    treatment_appt = malaria.HSI_Malaria_Treatment_Complicated(person_id=person_id,
                                                               module=sim.modules['Malaria'])
    treatment_appt.apply(person_id=person_id, squeeze_factor=0.0)
    assert df.at[person_id, 'ma_tx'] != 'none'
    assert df.at[person_id, "ma_date_tx"] == sim.date
    assert df.at[person_id, "ma_tx_counter"] > 0

    # run the death event
    death_event = malaria.MalariaDeathEvent(
        module=sim.modules['Malaria'], person_id=person_id, cause="Malaria")
    death_event.apply(person_id)

    # should cause death - no cure
    assert not df.at[person_id, 'is_alive']
    assert df.at[person_id, 'cause_of_death'] == "Malaria"

    # -------------- no treatment for severe malaria -------------- #
    # set treatment with zero efficacy for severe malaria cases - death should occur
    # select person and assign severe malaria
    person_id = 1
    df.loc[person_id, ["ma_is_infected", "ma_inf_type"]] = (True, "severe")

    assert df.at[person_id, "ma_tx"] == "none"
    assert df.at[person_id, "ma_date_tx"] is pd.NaT
    assert df.at[person_id, "ma_tx_counter"] == 0

    # run the death event
    death_event = malaria.MalariaDeathEvent(
        module=sim.modules['Malaria'], person_id=person_id, cause="Malaria")
    death_event.apply(person_id)

    # should cause death - no cure
    assert not df.at[person_id, 'is_alive']
    assert df.at[person_id, 'cause_of_death'] == "Malaria"


def get_sim(seed):
    """
    get sim with the checks for configuration of properties running in the malaria module
    """

    start_date = Date(2010, 1, 1)
    sim = Simulation(start_date=start_date, seed=seed, resourcefilepath=resourcefilepath)

    # Register the appropriate modules
    sim.register(
        demography.Demography(),
        healthsystem.HealthSystem(cons_availability="all", disable=False),
        simplified_births.SimplifiedBirths(),
        symptommanager.SymptomManager(),
        healthseekingbehaviour.HealthSeekingBehaviour(),
        healthburden.HealthBurden(),
        enhanced_lifestyle.Lifestyle(),
        malaria.Malaria(),
        tb.Tb(),
        hiv.Hiv(),
        epi.Epi(),
    )

    return sim


def test_individual_testing_and_treatment(sim):
    """ test treatment is initiated for clinical malaria case
    """

    sim = get_sim(seed)

    sim.modules['Malaria'].parameters['prob_malaria_case_tests'] = 1.0  # all cases referred for rdt
    sim.modules["Malaria"].parameters["sensitivity_rdt"] = 1.0

    # Run the simulation and flush the logger
    sim.make_initial_population(n=10)
    # simulate for 0 days, just get everything set up (dxtests etc)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props

    # -------- clinical infection
    person_id = 0

    # assign person_id malaria infection
    df.at[person_id, 'ma_is_infected'] = True
    df.at[person_id, 'ma_date_infected'] = sim.date
    df.at[person_id, 'ma_date_symptoms'] = sim.date
    df.at[person_id, 'ma_inf_type'] = "clinical"
    df.at[person_id, 'age_years'] = 3

    # assign clinical symptoms and schedule rdt
    pollevent = malaria.MalariaUpdateEvent(module=sim.modules['Malaria'])
    pollevent.run()

    assert not pd.isnull(df.at[person_id, "ma_date_symptoms"])
    assert set(sim.modules["SymptomManager"].has_what(person_id=person_id)) == {
        "fever",
        "headache",
        "vomiting",
        "stomachache",
    }

    # check rdt is scheduled
    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
        isinstance(ev[1], malaria.HSI_Malaria_rdt)
    ][0]
    assert date_event > sim.date

    # screen and test person_id
    event.run(squeeze_factor=0.0)

    # check person diagnosed
    assert df.at[person_id, "ma_dx_counter"] == 1

    # check treatment event is scheduled
    date_event, tx_event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
        isinstance(ev[1], malaria.HSI_Malaria_Treatment)
    ][0]
    assert date_event >= sim.date

    # run treatment event and check person is treated and treatment counter incremented
    assert df.at[person_id, "ma_tx_counter"] == 0
    tx_event.run(squeeze_factor=0.0)

    assert df.at[person_id, "ma_tx_counter"] == 1
    assert df.at[person_id, "ma_tx"] != 'none'

    # -------- asymptomatic infection
    person_id = 1

    # assign person_id malaria
    df.at[person_id, 'ma_is_infected'] = True
    df.at[person_id, 'ma_date_infected'] = sim.date
    df.at[person_id, 'ma_date_symptoms'] = sim.date
    df.at[person_id, 'ma_inf_type'] = "asym"
    df.at[person_id, 'age_years'] = 3

    # check no clinical symptoms set and no rdt scheduled
    pollevent = malaria.MalariaUpdateEvent(module=sim.modules['Malaria'])
    pollevent.apply(sim.population)

    assert sim.modules['SymptomManager'].has_what(person_id=person_id) == []

    # check no rdt is scheduled
    assert "malaria.HSI_Malaria_rdt" not in sim.modules['HealthSystem'].find_events_for_person(person_id)

    # screen and test person_id
    rdt_appt = malaria.HSI_Malaria_rdt(person_id=person_id,
                                       module=sim.modules['Malaria'])
    rdt_appt.apply(person_id=person_id, squeeze_factor=0.0)

    # check person diagnosed (with asym infection but no clinical symptoms)
    assert df.at[person_id, "ma_dx_counter"] == 1

    # check treatment event is scheduled
    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
        isinstance(ev[1], malaria.HSI_Malaria_Treatment)
    ][0]
    assert date_event >= sim.date

    # run treatment event and check person is treated and treatment counter incremented
    assert df.at[person_id, "ma_tx_counter"] == 0
    tx_appt = malaria.HSI_Malaria_Treatment(person_id=person_id,
                                            module=sim.modules['Malaria'])
    tx_appt.apply(person_id=person_id, squeeze_factor=0.0)

    assert df.at[person_id, "ma_tx_counter"] == 1
    assert df.at[person_id, "ma_tx"] != 'none'

    # -------- severe infection
    person_id = 2

    # assign person_id malaria
    df.at[person_id, 'ma_is_infected'] = True
    df.at[person_id, 'ma_date_infected'] = sim.date
    df.at[person_id, 'ma_date_symptoms'] = sim.date
    df.at[person_id, 'ma_inf_type'] = "severe"
    df.at[person_id, 'age_years'] = 3

    # assign clinical symptoms and schedule rdt
    pollevent = malaria.MalariaUpdateEvent(module=sim.modules['Malaria'])
    pollevent.apply(sim.population)

    assert not pd.isnull(df.at[person_id, "ma_date_symptoms"])

    # check rdt is scheduled
    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
        isinstance(ev[1], malaria.HSI_Malaria_rdt)
    ][0]
    assert date_event > sim.date

    # screen and test person_id
    rdt_appt = malaria.HSI_Malaria_rdt(person_id=person_id,
                                       module=sim.modules['Malaria'])
    rdt_appt.apply(person_id=person_id, squeeze_factor=0.0)

    # check person diagnosed
    assert df.at[person_id, "ma_dx_counter"] == 1

    # check treatment event is scheduled
    date_event, event = [
        ev for ev in sim.modules['HealthSystem'].find_events_for_person(person_id) if
        isinstance(ev[1], malaria.HSI_Malaria_Treatment_Complicated)
    ][0]
    assert date_event >= sim.date

    # run treatment event and check person is treated and treatment counter incremented
    assert df.at[person_id, "ma_tx_counter"] == 0
    tx_appt = malaria.HSI_Malaria_Treatment_Complicated(person_id=person_id,
                                                        module=sim.modules['Malaria'])
    tx_appt.apply(person_id=person_id, squeeze_factor=0.0)

    assert df.at[person_id, "ma_tx_counter"] == 1
    assert df.at[person_id, "ma_tx"] != "none"


def test_population_testing_and_treatment(sim):
    """ test treatment is initiated for set of clinical cases
    set of clinical cases -> ensure clinical counter recording correct number
    ensure dx_counter correct
    """

    sim = get_sim(seed)

    sim.modules['Malaria'].parameters['prob_malaria_case_tests'] = 1.0  # all cases referred for rdt
    sim.modules["Malaria"].parameters["sensitivity_rdt"] = 1.0

    pop = 100
    # Run the simulation and flush the logger
    sim.make_initial_population(n=pop)
    # simulate for 0 days, just get everything set up (dxtests etc)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props

    # Make no-one has malaria and clear the event queues:
    sim.modules['HealthSystem'].HSI_EVENT_QUEUE = []
    sim.event_queue.queue = []

    df.loc[df.is_alive, "ma_is_infected"] = True
    df.loc[df.is_alive, "ma_date_infected"] = sim.date
    df.loc[df.is_alive, "ma_date_symptoms"] = sim.date
    df.loc[df.is_alive, "ma_inf_type"] = "clinical"

    idx = list(df.loc[df.is_alive].index)

    # assign clinical symptoms and schedule rdt
    pollevent = malaria.MalariaUpdateEvent(module=sim.modules['Malaria'])
    pollevent.apply(sim.population)

    assert df["ma_clinical_counter"].sum() == pop

    # check one rdt is scheduled for each person in idx
    for person in idx:
        assert 1 == len([
            ev[0] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id=person) if
            (isinstance(ev[1], malaria.HSI_Malaria_rdt) & (ev[0] >= sim.date))
        ])

    # run the rdt for everyone
    for person in idx:
        rdt_appt = malaria.HSI_Malaria_rdt(person_id=person,
                                           module=sim.modules['Malaria'])
        rdt_appt.apply(person_id=person, squeeze_factor=0.0)

    assert df.loc[df.is_alive, "ma_clinical_counter"].sum() == len(df.loc[df.is_alive])

    # check 10 treatment events are scheduled
    for person in idx:
        assert 1 == len([
            ev[0] for ev in sim.modules['HealthSystem'].find_events_for_person(person_id=person) if
            (isinstance(ev[1], malaria.HSI_Malaria_Treatment) & (ev[0] >= sim.date))
        ])

    # run the treatment for everyone
    for person in idx:
        tx_appt = malaria.HSI_Malaria_Treatment(person_id=person,
                                                module=sim.modules['Malaria'])
        tx_appt.apply(person_id=person, squeeze_factor=0.0)

    assert df["ma_tx_counter"].sum() == pop


def test_linear_model_for_clinical_malaria(sim):
    sim = get_sim(seed)

    # -------------- Perfect protection through IPTp -------------- #
    # set perfect protection for IPTp against clinical malaria - no cases should occur
    sim.modules['Malaria'].parameters['rr_clinical_malaria_iptp'] = 0

    # Run the simulation and flush the logger
    sim.make_initial_population(n=25)
    # simulate for 0 days, just get everything set up (dxtests etc)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    df = sim.population.props

    # make the whole population infected with parasitaemia
    # and therefore eligible for clinical/severe malaria poll
    df.loc[df.is_alive, "district_of_residence"] = "Balaka"  # put everyone in high-risk district
    df.loc[df.is_alive, "district_num_of_residence"] = 28
    df.loc[df.is_alive, "ma_is_infected"] = True
    df.loc[df.is_alive, "ma_date_infected"] = sim.date
    df.loc[df.is_alive, "ma_inf_type"] = "asym"
    df.loc[df.is_alive, "is_pregnant"] = True
    df.loc[df.is_alive, "ma_iptp"] = True

    # run malaria poll
    pollevent = malaria.MalariaPollingEventDistrict(module=sim.modules['Malaria'])
    pollevent.run()

    # make sure no-one assigned clinical or severe malaria
    assert not (df.loc[df.is_alive, 'ma_inf_type'].isin({'clinical', 'severe'})).any()
