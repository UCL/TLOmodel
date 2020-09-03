import os
from pathlib import Path

from tlo import Date, Simulation
from tlo.methods import (
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthsystem,
    hiv,
    malecircumcision,
    symptommanager,
    tb,
)

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 50


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_simulation():
    service_availability = list(["hiv*", "tb*", "male_circumcision*"])

    resourcefilepath = Path(os.path.dirname(__file__)) / "../resources"
    sim = Simulation(start_date=start_date)

    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(
        healthsystem.HealthSystem(
            resourcefilepath=resourcefilepath,
            service_availability=service_availability,
            mode_appt_constraints=0,
            ignore_cons_constraints=True,
            ignore_priority=True,
            capabilities_coefficient=1.0,
            disable=True,
        )
    )  # disables the health system constraints
    sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(hiv.Hiv(resourcefilepath=resourcefilepath))
    sim.register(tb.Tb(resourcefilepath=resourcefilepath))
    sim.register(malecircumcision.MaleCircumcision(resourcefilepath=resourcefilepath))

    sim.seed_rngs(0)
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    check_dtypes(sim)

    df = sim.population.props

    # check properties
    # assert ((df.sex == 'M') & (df.hv_sexual_risk == 'low')).all()  # no sex work
    assert not any((df.sex == "M") & (df.hv_sexual_risk == "sex_work"))

    assert not ((df.hv_number_tests >= 1) & ~df.hv_ever_tested).any()

    assert not (df.mc_is_circumcised & (df.sex == "F")).any()

    # check if HIV-TB co-infected, hv_specific_symptoms=aids
    assert not any(df.tb_diagnosed & df.hv_inf & (df.hv_specific_symptoms == "none"))

    # only on cotrim if hiv is diagnosed [hv_date_cotrim = DATE and hv_diagnosed = True]
    assert not any(df.hv_date_cotrim.notnull() & ~df.hv_diagnosed)
