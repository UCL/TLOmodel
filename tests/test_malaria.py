import os
from pathlib import Path

import pandas as pd

from tlo import Date, Simulation
from tlo.methods import (
    antenatal_care,
    contraception,
    demography,
    dx_algorithm_adult,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    malaria,
    newborn_outcomes,
    pregnancy_supervisor,
    symptommanager,
)

start_date = Date(2010, 1, 1)
end_date = Date(2014, 1, 1)
popsize = 1000

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


# @pytest.fixture(scope='module')
def test_sims(tmpdir):
    service_availability = list(["malaria"])
    malaria_testing = 0.35  # adjust this to match rdt/tx levels

    sim = Simulation(start_date=start_date, seed=0)

    # Register the appropriate modules
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(
            resourcefilepath=resourcefilepath,
            service_availability=service_availability,
            mode_appt_constraints=0,
            ignore_cons_constraints=True,
            ignore_priority=True,
            capabilities_coefficient=1.0,
            disable=True,  # disables the health system constraints so all HSI events run
        ),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        dx_algorithm_child.DxAlgorithmChild(),
        dx_algorithm_adult.DxAlgorithmAdult(),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        contraception.Contraception(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        labour.Labour(resourcefilepath=resourcefilepath),
        newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
        antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
        pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
        malaria.Malaria(resourcefilepath=resourcefilepath, testing=malaria_testing, itn=None)
    )

    # Run the simulation and flush the logger
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # check malaria deaths only being scheduled due to severe malaria (not clinical or asym)
    df = sim.population.props
    assert not (
        df.ma_date_death & ((df.ma_inf_type == "clinical") | (df.ma_inf_type == "none"))
    ).any()

    # check cases /  treatment are occurring
    assert not (df.ma_clinical_counter == 0).all()
    assert not (df.ma_date_tx == pd.NaT).all()

    # check clinical malaria in pregnancy counter not including males
    assert not any((df.sex == "M") & (df.ma_clinical_preg_counter > 0))

    # check symptoms are being assigned - fever assigned to all clinical cases
    for person in df.index[df.is_alive & (df.ma_inf_type == "clinical")]:
        assert "fever" in sim.modules["SymptomManager"].has_what(person)
        assert "Malaria" in sim.modules["SymptomManager"].causes_of(person, "fever")
