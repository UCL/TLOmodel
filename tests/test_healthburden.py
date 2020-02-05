import os
from pathlib import Path

import pandas as pd

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    chronicsyndrome,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthsystem,
    mockitis,
    symptommanager,
)

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / "../resources"
except NameError:
    # running interactively
    resourcefilepath = "resources"

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 100


# Simply test whether the system runs under multiple configurations of the healthsystem
# NB. Running the dummy Mockitits and ChronicSyndrome modules test all aspects of the healthsystem module.


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_run_with_healthburden_with_dummy_diseases(tmpdir):
    # There should be no events run or scheduled

    # Establish the simulation object
    sim = Simulation(start_date=start_date)

    # Define the service availability as null
    service_availability = []

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(
        healthsystem.HealthSystem(
            resourcefilepath=resourcefilepath,
            service_availability=service_availability,
            capabilities_coefficient=0.0,
            mode_appt_constraints=0,
        )
    )
    sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
    sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
    sim.register(mockitis.Mockitis())
    sim.register(chronicsyndrome.ChronicSyndrome())

    sim.seed_rngs(0)

    # Run the simulation
    f = sim.configure_logging("test_log", directory=tmpdir)
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # read the results
    output = parse_log_file(f)

    # Do the checks
    # correctly configured index (outputs on 31st december in each year of simulation for each age/sex group)
    dalys = output["tlo.methods.healthburden"]["DALYS"]
    age_index = sim.modules["Demography"].AGE_RANGE_CATEGORIES
    sex_index = ["M", "F"]
    year_index = list(range(start_date.year, end_date.year + 1))
    correct_multi_index = pd.MultiIndex.from_product(
        [sex_index, age_index, year_index], names=["sex", "age_range", "year"]
    )
    dalys["year"] = pd.to_datetime(dalys["date"]).dt.year
    assert (pd.to_datetime(dalys["date"]).dt.month == 12).all()
    assert (pd.to_datetime(dalys["date"]).dt.day == 31).all()
    output_multi_index = dalys.set_index(["sex", "age_range", "year"]).index
    assert output_multi_index.equals(correct_multi_index)

    # check that there is a YLD for each module registered
    yld_colnames = list()
    for colname in list(dalys.columns):
        if "YLD" in colname:
            yld_colnames.append(colname)

    module_names_in_output = set()
    for yld_colname in yld_colnames:
        module_names_in_output.add(yld_colname.split("_", 2)[1])
    assert module_names_in_output == {"Mockitis", "ChronicSyndrome"}
