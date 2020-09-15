import os
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    contraception,
    healthburden,
    healthsystem,
    enhanced_lifestyle,
    dx_algorithm_child,
    healthseekingbehaviour,
    symptommanager,
    antenatal_care,
    labour,
    newborn_outcomes,
    pregnancy_supervisor,
    epi,
    measles
)

start_date = Date(2010, 1, 1)
end_date = Date(2025, 1, 1)
popsize = 500

try:
    resources = Path(os.path.dirname(__file__)) / "../resources"
except NameError:
    # running interactively
    resources = "resources"


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


# checking no vaccines administered through health system
# only hpv should stay at zero, other vaccines start as individual events (year=2010-2018)
# coverage should gradually decline for all after 2018
# hard constraints (mode=2) and zero capabilities
def test_no_vaccine(tmpdir):
    log_config = {
        "filename": "measles_test",  # The name of the output file (a timestamp will be appended).
        "directory": tmpdir,  # The default output path is `./outputs`. Change it here, if necessary
        "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
            "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
            "tlo.methods.measles": logging.INFO,
        }
    }

    sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

    sim.register(
        demography.Demography(resourcefilepath=resources),
        healthsystem.HealthSystem(
            resourcefilepath=resources,
            service_availability=[" "],  # no treatment IDs allowed
            mode_appt_constraints=0,
            ignore_cons_constraints=True,
            ignore_priority=True,
            capabilities_coefficient=0.0,  # multiplier for capabilities of health officer
            disable=False,
        ),
        # disables the health system constraints so all HSI events run
        symptommanager.SymptomManager(resourcefilepath=resources),
        healthseekingbehaviour.HealthSeekingBehaviour(),
        dx_algorithm_child.DxAlgorithmChild(),
        # dx_algorithm_adult.DxAlgorithmAdult(),
        healthburden.HealthBurden(resourcefilepath=resources),
        contraception.Contraception(resourcefilepath=resources),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resources),
        labour.Labour(resourcefilepath=resources),
        newborn_outcomes.NewbornOutcomes(resourcefilepath=resources),
        antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resources),
        pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resources),
        epi.Epi(resourcefilepath=resources),
        measles.Measles(resourcefilepath=resources),
    )

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # check we can read the results
    log_df = parse_log_file(sim.log_filepath)
    df = sim.population.props

    # check no vaccine administered from 2019 onwards (through HSIs)
    # vaccines pre-2019 are assigned based on coverage and not dependent on health system capacity
    # check coverage in children born after 2019

    # eligible children born after 2019
    def get_coverage(condition, subset):
        total = sum(subset)
        has_condition = sum(condition & subset)
        coverage = has_condition / total * 100 if total else 0
        assert coverage <= 100
        return coverage

    # age should be <= end_date.year - 2019
    max_age = end_date.year - 2019
    susceptible_cohort = (df.age_years <= max_age)
    unvaccinated = get_coverage(df.va_measles == 0, susceptible_cohort)
    assert unvaccinated == 100

    # check measles incidence (in the unvaccinated cohort) is at pre-EPI levels
    # plot should show rebound after 2019
    output = log_df["tlo.methods.measles"]["incidence"]
    model_inc = output["inc_1000py"]
    model_date = output["date"]

    # ----------------------------------- PLOT -----------------------------------#
    plt.style.use("ggplot")

    # Measles incidence
    plt.subplot(111)  # numrows, numcols, fignum
    plt.plot(model_date, model_inc)
    plt.title("Measles incidence")
    plt.xlabel("Date")
    plt.ylabel("Incidence per 1000py")
    plt.xticks(rotation=90)
    plt.legend(["Model"], bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.show()
