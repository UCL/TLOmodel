import os
from pathlib import Path

import pandas as pd

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    bladder_cancer,
    breast_cancer,
    care_of_women_during_pregnancy,
    chronicsyndrome,
    contraception,
    demography,
    depression,
    diarrhoea,
    enhanced_lifestyle,
    healthburden,
    healthsystem,
    hiv,
    labour,
    malaria,
    mockitis,
    ncds,
    newborn_outcomes,
    oesophagealcancer,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager,
)

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = 'resources'

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
    sim = Simulation(start_date=start_date, seed=0, log_config={'filename': 'test_log', 'directory': tmpdir})

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           disable_and_reject_all=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 mockitis.Mockitis(),
                 chronicsyndrome.ChronicSyndrome())

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)
    check_dtypes(sim)

    # read the results
    output = parse_log_file(sim.log_filepath)

    # Do the checks
    # correctly configured index (outputs on 31st december in each year of simulation for each age/sex group)
    dalys = output['tlo.methods.healthburden']['dalys']
    dalys = dalys.drop(columns=['date'])
    age_index = sim.modules['Demography'].AGE_RANGE_CATEGORIES
    sex_index = ['M', 'F']
    year_index = list(range(start_date.year, end_date.year + 1))
    correct_multi_index = pd.MultiIndex.from_product([sex_index, age_index, year_index],
                                                     names=['sex', 'age_range', 'year'])
    output_multi_index = dalys.set_index(['sex', 'age_range', 'year']).index
    assert output_multi_index.equals(correct_multi_index)

    # check that there is a column for each 'label' that is registered
    assert set(dalys.set_index(['sex', 'age_range', 'year']).columns) == \
           {'Other', 'Mockitis_Disability_And_Death', 'ChronicSyndrome_Disability_And_Death'}


def test_cause_of_disability_being_registered():
    """Test that the modules can declare causes of disability, and that the mappers between tlo causes of disability
    and gbd causes of disability can be created correctly and that these make sense with respect to the corresponding
    mappers for deaths."""

    rfp = Path(os.path.dirname(__file__)) / '../resources'

    sim = Simulation(start_date=Date(2010, 1, 1), seed=0)
    sim.register(
        demography.Demography(resourcefilepath=rfp),
        symptommanager.SymptomManager(resourcefilepath=rfp),
        breast_cancer.BreastCancer(resourcefilepath=rfp),
        enhanced_lifestyle.Lifestyle(resourcefilepath=rfp),
        healthsystem.HealthSystem(resourcefilepath=rfp, disable_and_reject_all=True),
        bladder_cancer.BladderCancer(resourcefilepath=rfp),
        depression.Depression(resourcefilepath=rfp),
        diarrhoea.Diarrhoea(resourcefilepath=rfp),
        hiv.Hiv(resourcefilepath=rfp),
        malaria.Malaria(resourcefilepath=rfp),
        ncds.Ncds(resourcefilepath=rfp),
        oesophagealcancer.OesophagealCancer(resourcefilepath=rfp),
        contraception.Contraception(resourcefilepath=rfp),
        labour.Labour(resourcefilepath=rfp),
        pregnancy_supervisor.PregnancySupervisor(resourcefilepath=rfp),
        care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=rfp),
        postnatal_supervisor.PostnatalSupervisor(resourcefilepath=rfp),
        newborn_outcomes.NewbornOutcomes(resourcefilepath=rfp),
        healthburden.HealthBurden(resourcefilepath=rfp)
    )
    sim.make_initial_population(n=20)
    sim.simulate(end_date=Date(2010, 1, 2))
    check_dtypes(sim)

    mapper_from_tlo_causes, mapper_from_gbd_causes = \
        sim.modules['HealthBurden'].create_mappers_from_causes_of_death_to_label()

    assert set(mapper_from_tlo_causes.keys()) == set(sim.modules['HealthBurden'].causes_of_disability.keys())
    assert set(mapper_from_gbd_causes.keys()) == set(sim.modules['HealthBurden'].parameters['gbd_causes_of_disability'])
    assert set(mapper_from_gbd_causes.values()) == set(mapper_from_tlo_causes.values())

    # todo check correspondence between causes of death and causes of disability
