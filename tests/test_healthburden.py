import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pytest import approx

from tlo import DAYS_IN_YEAR, Date, Module, Simulation, logging
from tlo.analysis.utils import get_mappers_in_fullmodel, parse_log_file
from tlo.events import Event, IndividualScopeEventMixin
from tlo.methods import (
    Metadata,
    chronicsyndrome,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthsystem,
    mockitis,
    symptommanager,
)
from tlo.methods.causes import Cause
from tlo.methods.demography import InstantaneousDeath, age_at_date
from tlo.methods.diarrhoea import increase_risk_of_death, make_treatment_perfect
from tlo.methods.fullmodel import fullmodel
from tlo.methods.healthburden import Get_Current_DALYS

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = 'resources'

start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 100


def extract_mapper(key):
    return pd.Series(key.drop(columns={'date'}).loc[0]).to_dict()


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_run_with_healthburden_with_dummy_diseases(tmpdir, seed):
    """Check that everything runs in the simple cases of Mockitis and Chronic Syndrome and that outputs are as expected.
    """

    # Establish the simulation object
    sim = Simulation(start_date=start_date, seed=seed,
                     log_config={'filename': 'test_log', 'directory': tmpdir}, resourcefilepath=resourcefilepath)

    # Register the appropriate modules
    sim.register(demography.Demography(),
                 enhanced_lifestyle.Lifestyle(),
                 healthsystem.HealthSystem(disable_and_reject_all=True),
                 symptommanager.SymptomManager(),
                 healthburden.HealthBurden(),
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
    pd.testing.assert_index_equal(output_multi_index, correct_multi_index, check_order=False)

    # check that there is a column for each 'label' that is registered
    assert set(dalys.set_index(['sex', 'age_range', 'year']).columns) == \
           {'Other', 'Mockitis_Disability_And_Death', 'ChronicSyndrome_Disability_And_Death'}


@pytest.mark.slow
def test_cause_of_disability_being_registered(seed, tmpdir):
    """Test that the modules can declare causes of disability, and that the mappers between tlo causes of disability
    and gbd causes of disability can be created correctly and that these make sense with respect to the corresponding
    mappers for deaths."""

    sim = Simulation(start_date=Date(2010, 1, 1), seed=seed,
                     log_config={'filename': 'test_log', 'directory': tmpdir}, resourcefilepath=resourcefilepath)
    sim.register(
        *fullmodel(
            module_kwargs={"HealthSystem": {"disable": True}},
        )
    )

    # Increase risk of death of Diarrhoea to ensure that are at least some deaths
    increase_risk_of_death(sim.modules['Diarrhoea'])
    make_treatment_perfect(sim.modules['Diarrhoea'])

    sim.make_initial_population(n=20)
    sim.simulate(end_date=Date(2010, 1, 2))
    check_dtypes(sim)

    hblog = parse_log_file(sim.log_filepath)['tlo.methods.healthburden']
    disability_mapper_from_gbd_causes = extract_mapper(hblog['disability_mapper_from_gbd_cause_to_common_label'])
    disability_mapper_from_tlo_causes = extract_mapper(hblog['disability_mapper_from_tlo_cause_to_common_label'])

    assert set(disability_mapper_from_tlo_causes.keys()) == \
           set(sim.modules['HealthBurden'].causes_of_disability.keys())
    assert set(disability_mapper_from_gbd_causes .keys()) == \
           set(sim.modules['HealthBurden'].parameters['gbd_causes_of_disability'])
    assert set(disability_mapper_from_gbd_causes .values()) == \
           set(disability_mapper_from_tlo_causes.values())


def test_arithmetic_of_disability_aggregation_calcs(seed):
    """Check that disability from different modules are being combined and computed in the correct way"""

    class ModuleWithPersonsAffected(Module):

        def __init__(self, persons_affected, name=None):
            super().__init__(name=name)
            self.persons_affected = persons_affected

    class DiseaseThatCausesA(ModuleWithPersonsAffected):
        METADATA = {Metadata.DISEASE_MODULE, Metadata.USES_HEALTHBURDEN}
        CAUSES_OF_DEATH = {'A': Cause(label='A')}
        CAUSES_OF_DISABILITY = {'A': Cause(label='A')}
        daly_wt = 0.2

        def read_parameters(self, resourcefilepath=None):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            pass

        def report_daly_values(self):
            df = self.sim.population.props
            disability = pd.Series(index=df.loc[df.is_alive].index, data=0.0)
            disability.loc[self.persons_affected] = self.daly_wt
            return disability

    class DiseaseThatCausesB(ModuleWithPersonsAffected):
        METADATA = {Metadata.DISEASE_MODULE, Metadata.USES_HEALTHBURDEN}
        CAUSES_OF_DEATH = {'B': Cause(label='B')}
        CAUSES_OF_DISABILITY = {'B': Cause(label='B')}
        daly_wt = 0.05

        def read_parameters(self, resourcefilepath=None):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            pass

        def report_daly_values(self):
            df = self.sim.population.props
            disability = pd.Series(index=df.loc[df.is_alive].index, data=0.0)
            disability.loc[self.persons_affected] = self.daly_wt
            return disability

    class DiseaseThatCausesAandB(ModuleWithPersonsAffected):
        METADATA = {Metadata.DISEASE_MODULE, Metadata.USES_HEALTHBURDEN}
        CAUSES_OF_DEATH = {'A': Cause(label='A'), 'B': Cause(label='B')}
        CAUSES_OF_DISABILITY = {'A': Cause(label='A'), 'B': Cause(label='B')}
        daly_wt_A = 0.5
        daly_wt_B = 0.09

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            pass

        def report_daly_values(self):
            df = self.sim.population.props
            disability = pd.DataFrame(index=df.loc[df.is_alive].index, columns=['A', 'B'], data=0.0)
            disability.loc[self.persons_affected, 'A'] = self.daly_wt_A
            disability.loc[self.persons_affected, 'B'] = self.daly_wt_B
            return disability

    class DiseaseThatCausesC(ModuleWithPersonsAffected):
        METADATA = {Metadata.DISEASE_MODULE, Metadata.USES_HEALTHBURDEN}
        CAUSES_OF_DEATH = {'C': Cause(label='A')}
        CAUSES_OF_DISABILITY = {'C': Cause(label='C')}
        daly_wt = 0.95

        def read_parameters(self, resourcefilepath=None):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            pass

        def report_daly_values(self):
            df = self.sim.population.props
            disability = pd.Series(index=df.loc[df.is_alive].index, data=0.0)
            disability.loc[self.persons_affected] = self.daly_wt
            return disability

    class DiseaseThatCausesNothing(Module):
        METADATA = {Metadata.DISEASE_MODULE, Metadata.USES_HEALTHBURDEN}
        CAUSES_OF_DEATH = {}
        CAUSES_OF_DISABILITY = {}

        def read_parameters(self, resourcefilepath=None):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            pass

        def report_daly_values(self):
            pass

    start_date = Date(2010, 1, 1)
    sim = Simulation(start_date=start_date, seed=seed, resourcefilepath=resourcefilepath)
    sim.register(
        demography.Demography(),
        enhanced_lifestyle.Lifestyle(),
        healthburden.HealthBurden(),
        DiseaseThatCausesA(persons_affected=0),
        DiseaseThatCausesB(persons_affected=1),
        DiseaseThatCausesAandB(persons_affected=2),
        # intentionally two instances of DiseaseThatCausesC
        DiseaseThatCausesC(persons_affected=3, name='DiseaseThatCausesC1'),
        DiseaseThatCausesC(persons_affected=3, name='DiseaseThatCausesC2'),
        DiseaseThatCausesNothing(),
        # Disable sorting to allow registering multiple instances of DiseaseThatCausesC
        sort_modules=False
    )
    sim.make_initial_population(n=4)
    sim.simulate(end_date=start_date)

    # get the dalys report:
    hb = sim.modules['HealthBurden']
    gcd = Get_Current_DALYS(hb)
    gcd.apply(sim.population)

    # check that persons experiencing DALYS are as expected
    assert set(hb.recognised_modules_names) == set([
        'DiseaseThatCausesA', 'DiseaseThatCausesB', 'DiseaseThatCausesAandB', 'DiseaseThatCausesNothing',
        'DiseaseThatCausesC1', 'DiseaseThatCausesC2'])

    yld = hb.years_lived_with_disability.sum()

    # check that dalys for A and B are being aggregated appropriately despite being declared in multiple modules
    # nb. the record is only for one month.
    assert yld['A'] == approx(
        (sim.modules['DiseaseThatCausesA'].daly_wt + sim.modules['DiseaseThatCausesAandB'].daly_wt_A) / 12
    )
    assert yld['B'] == approx(
        (sim.modules['DiseaseThatCausesB'].daly_wt + sim.modules['DiseaseThatCausesAandB'].daly_wt_B) / 12
    )

    # check that daly weight for people is scaled to less than 1.0 (1/12 to be per month) even if they have two
    # large daly weights imposed at the same time that would sum to more than 1.0.
    assert yld.loc['C'] == approx(1.0 / 12)


def test_arithmetic_of_dalys_calcs(seed):
    """Check that life-years lost are being computed and combined with years lived with disability correctly"""

    class DiseaseThatCausesA(Module):
        """Disease that will:
          * impose disability on person_id=0 at the point 25% through the year;
          * cause the death of the person_id=0 at the point 50% through the year;
        """
        METADATA = {Metadata.DISEASE_MODULE, Metadata.USES_HEALTHBURDEN}
        CAUSES_OF_DEATH = {'cause_of_death_A': Cause(label='Label_A')}
        CAUSES_OF_DISABILITY = {'cause_of_disability_A': Cause(label='Label_A')}
        daly_wt = 0.5

        def read_parameters(self, resourcefilepath=None):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            year = sim.date.year
            days_in_year = sum([pd.Period(f'{year}-{i}-1').daysinmonth for i in range(1, 13)])
            day_25pc = sim.date + pd.DateOffset(days=int(0.25 * days_in_year))
            day_50pc = sim.date + pd.DateOffset(days=int(0.50 * days_in_year))

            self.has_disease = False
            sim.schedule_event(StartOfDiseaseEvent(self, 0), day_25pc)
            sim.schedule_event(InstantaneousDeath(self, individual_id=0, cause='cause_of_death_A'), day_50pc)

        def report_daly_values(self):
            df = sim.population.props
            return pd.Series(index=df.loc[df.is_alive].index, data=self.daly_wt * self.has_disease)

    class StartOfDiseaseEvent(Event, IndividualScopeEventMixin):
        def __init__(self, module, individual_id):
            super().__init__(module, person_id=individual_id)

        def apply(self, individual_id):
            self.module.has_disease = True

    start_date = Date(2010, 1, 1)
    sim = Simulation(start_date=start_date, seed=seed, resourcefilepath=resourcefilepath)
    sim.register(
        demography.Demography(),
        enhanced_lifestyle.Lifestyle(),
        healthburden.HealthBurden(),
        DiseaseThatCausesA(),
    )
    sim.make_initial_population(n=1)

    # To make calcs easy, set the date_of_birth of the person_id=0, to be 1st January 2010
    df = sim.population.props
    df.loc[0, ['is_alive', 'date_of_birth']] = (True, Date(2010, 1, 1))
    sim.simulate(end_date=Date(2010, 12, 31))

    # Examine YLL, YLD and DALYS for 'A' recorded at the end of the simulation
    hb = sim.modules['HealthBurden']
    yld = hb.years_lived_with_disability.reset_index()
    yll = hb.years_life_lost.reset_index()
    dalys = hb.get_dalys(yld=yld, yll=yll)

    daly_wt = sim.modules['DiseaseThatCausesA'].daly_wt

    # Check record of YLD and YLL (accurate to within a day (due to odd number of days in a year))
    assert yld['cause_of_disability_A'].sum() == approx(daly_wt * 0.25, abs=(daly_wt / 365))
    assert yll['cause_of_death_A'].sum() == approx(0.5, abs=1 / 365)
    assert dalys['Label_A'].sum() == approx(0.5 + 0.25 * daly_wt, abs=1 / 365)


def test_airthmetic_of_lifeyearslost(seed, tmpdir):
    """Check that a death causes the right number of life-years-lost to be logged and in the right age-groups (when
    there is no stacking by age or time)."""

    start_date = Date(2010, 1, 1)
    sim = Simulation(start_date=start_date, seed=seed, resourcefilepath=resourcefilepath)
    sim.register(
        demography.Demography(),
        enhanced_lifestyle.Lifestyle(),
        healthburden.HealthBurden(),
    )
    sim.make_initial_population(n=1)

    # Set the date_of_birth of the person_id=0, such that the person is 4.5 years-old on 1st Jan 2010 (so that life-
    #  years lost span 0-4 and 5-9 age-groups)
    dob = start_date - pd.DateOffset(days=int(4.5 * DAYS_IN_YEAR))
    df = sim.population.props
    df.loc[0, ['sex', 'is_alive', 'date_of_birth']] = ('F', True, dob)
    sim.simulate(end_date=Date(2010, 12, 31))

    # Get pointer to internal storage of life-years lost
    hb = sim.modules['HealthBurden']
    yll = hb.years_life_lost

    # check that no life-years-lost
    assert yll.sum().sum() == 0.0

    # reset the date to 1st Jan 2010 and cause the death of the person
    sim.date = Date(2010, 1, 1)
    sim.modules['Demography'].do_death(individual_id=0, cause='Other', originating_module=sim.modules['Demography'])

    # check that the right number of years-life-lost is recorded
    # (= 1.0 as the simulation last 1.0 years and the person was dead throughout)
    assert yll.sum().sum() == approx(1.0)

    # check that age-range is correct (0.5 ly lost among 0-4 year-olds; 0.5 ly lost to 5-9 year-olds)
    assert yll.loc[('F', '0-4', slice(None), 2010)].sum().sum() == approx(0.5, abs=2.0 / DAYS_IN_YEAR)
    assert yll.loc[('F', '5-9', slice(None), 2010)].sum().sum() == approx(0.5, abs=2.0 / DAYS_IN_YEAR)
    assert yll.loc[('F', ['0-4', '5-9'], slice(None), 2010)].sum().sum() == approx(1.0, abs=0.5 / DAYS_IN_YEAR)


@pytest.mark.slow
def test_arithmetic_of_stacked_lifeyearslost(tmpdir, seed):
    """Check that the computation of 'stacked' LifeYearsLost and DALYS is done correctly (i.e., when all the
    future life-years lost are allocated to the year of death (stacked by time); or when all future life-year lost are
    allocated to the year of death and age of death (stacked by age and time)."""

    class DiseaseThatCausesA(Module):
        """Disease that will:
          * impose a disability on person_id=0 on a given date
          * cause the death of the person_id=0 on a given date
        """
        METADATA = {Metadata.DISEASE_MODULE, Metadata.USES_HEALTHBURDEN}
        CAUSES_OF_DEATH = {'cause_of_death_A': Cause(label='Label_A')}
        CAUSES_OF_DISABILITY = {'cause_of_disability_A': Cause(label='Label_A')}
        daly_wt = 0.5
        disability_onset_date = Date(2011, 1, 1)
        death_date = Date(2012, 1, 1)

        def read_parameters(self, resourcefilepath=None):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            self.has_disease = False
            sim.schedule_event(StartOfDiseaseEvent(self, 0), self.disability_onset_date)
            sim.schedule_event(InstantaneousDeath(self, individual_id=0, cause='cause_of_death_A'), self.death_date)

        def report_daly_values(self):
            df = sim.population.props
            return pd.Series(index=df.loc[df.is_alive].index, data=self.daly_wt * self.has_disease)

    class StartOfDiseaseEvent(Event, IndividualScopeEventMixin):
        def __init__(self, module, individual_id):
            super().__init__(module, person_id=individual_id)

        def apply(self, individual_id):
            self.module.has_disease = True

    start_date = Date(2010, 1, 1)
    sim = Simulation(start_date=start_date, seed=seed, log_config={
        'filename': 'tmp',
        'directory': tmpdir,
        'custom_levels': {
            "tlo.methods.healthburden": logging.INFO}}, resourcefilepath=resourcefilepath
                     )
    sim.register(
        demography.Demography(),
        enhanced_lifestyle.Lifestyle(),
        healthburden.HealthBurden(),
        DiseaseThatCausesA()
    )
    sim.make_initial_population(n=1)

    AGE_RANGE_LOOKUP = sim.modules['Demography'].AGE_RANGE_LOOKUP
    AGE_RANGE_CATEGORIES = sim.modules['Demography'].AGE_RANGE_CATEGORIES

    # Determine the properties of person_id = 0
    date_of_birth = Date(2010, 1, 1)  # 1st January 2010 will make the calculations easier
    sex = 'F'

    df = sim.population.props
    df.loc[0, ['is_alive', 'date_of_birth']] = (True, date_of_birth)
    df.loc[0, 'sex'] = sex
    sim.simulate(end_date=Date(2029, 12, 31))

    daly_wt = sim.modules['DiseaseThatCausesA'].daly_wt
    death_date = sim.modules['DiseaseThatCausesA'].death_date
    disability_onset_date = sim.modules['DiseaseThatCausesA'].disability_onset_date
    age_limit_for_yll = sim.modules['HealthBurden'].parameters['Age_Limit_For_YLL']

    age_range_at_disability_onset = AGE_RANGE_LOOKUP[
        int(np.round(age_at_date(disability_onset_date, date_of_birth)))
    ]
    age_at_death = int(np.round(age_at_date(death_date, date_of_birth)))
    age_range_at_death = AGE_RANGE_LOOKUP[age_at_death]

    age_groups_where_yll_are_accrued = set(
        [_grp for _age, _grp in AGE_RANGE_LOOKUP.items()
         if age_at_death <= _age < sim.modules['HealthBurden'].parameters['Age_Limit_For_YLL']
         ]
    )
    age_groups_where_yll_are_not_accrued = set(AGE_RANGE_CATEGORIES) - age_groups_where_yll_are_accrued

    # Examine YLL, YLD and DALYS recorded at the end of the simulation
    log = parse_log_file(sim.log_filepath)['tlo.methods.healthburden']

    # Examine Years Lived with Disability
    yld = log['yld_by_causes_of_disability']
    marker_for_disability = (
        (yld.year == disability_onset_date.year)
        & (yld.age_range == age_range_at_disability_onset)
        & (yld.sex == sex)
    )
    assert (yld.loc[marker_for_disability, 'cause_of_disability_A'] == daly_wt * 1.0).all()
    assert (yld.loc[~marker_for_disability, 'cause_of_disability_A'] == 0.0).all()

    # For the Non-Stacked Results
    # -- YLL
    yll_not_stacked = log['yll_by_causes_of_death']
    yll_by_year_not_stacked = yll_not_stacked.loc[
        (yll_not_stacked.sex == sex),
        ['year', 'age_range', 'cause_of_death_A']
    ].groupby('year')['cause_of_death_A'].sum()
    assert all([yll_by_year_not_stacked.loc[year] == approx(1.0, abs=1 / 364) for year in
                range(death_date.year, sim.end_date.year)])
    assert all([yll_by_year_not_stacked.loc[year] == approx(0.0, abs=1 / 364) for year in
                range(sim.start_date.year, death_date.year)])

    # For the Stacked Results
    # -- YLL (Stacked by time - but not age)
    yll_stacked_by_time = log['yll_by_causes_of_death_stacked']
    yll_stacked_by_time = yll_stacked_by_time.loc[
        (yll_stacked_by_time.sex == sex),
        ['year', 'age_range', 'cause_of_death_A']
    ].groupby(['year', 'age_range'])['cause_of_death_A'].sum().unstack()
    assert 0.0 == yll_stacked_by_time.loc[
        yll_stacked_by_time.index != death_date.year].sum().sum()  # No Yll for years other than year of death
    assert approx(age_limit_for_yll - 2.0, 1 / 364) == yll_stacked_by_time.loc[
        death_date.year].sum()  # In year of death, (age_limit_for_yll - 2)    years of lost life.
    assert (yll_stacked_by_time.loc[death_date.year, yll_stacked_by_time.columns[
        yll_stacked_by_time.columns.isin(age_groups_where_yll_are_accrued)]] > 0).all()
    assert 0.0 == yll_stacked_by_time[
        sorted(age_groups_where_yll_are_not_accrued)
    ].sum().sum()  # There should be no yll for ages above `age_limit_for_yll` because that is the definition

    # -- YLL (Stacked by age and time)
    yll_stacked_by_age_and_time = log['yll_by_causes_of_death_stacked_by_age_and_time']
    yll_stacked_by_age_and_time = yll_stacked_by_age_and_time.loc[
        (yll_stacked_by_age_and_time.sex == sex),
        ['year', 'age_range', 'cause_of_death_A']
    ].groupby(['year', 'age_range'])['cause_of_death_A'].sum().unstack()
    assert 0 == yll_stacked_by_age_and_time[yll_stacked_by_age_and_time.columns[
        yll_stacked_by_age_and_time.columns != age_range_at_death]].sum().sum()  # No YLL for ages other than age at
    #                                                                              death
    assert 0 == yll_stacked_by_age_and_time.loc[
        yll_stacked_by_age_and_time.index != death_date.year].sum().sum()  # No Yll for years other than year of death
    assert yll_stacked_by_age_and_time.sum().sum() == approx(
        yll_stacked_by_time.sum().sum())  # Total YLL matches YLL when stacked by time only

    # Check dalys (not stacked) is as expected:
    dalys_by_year_not_stacked = log['dalys'].loc[
        (log['dalys'].sex == sex), ['year', 'age_range', 'Label_A']
    ].groupby('year')['Label_A'].sum()
    assert dalys_by_year_not_stacked.at[sim.start_date.year] == 0.0
    assert dalys_by_year_not_stacked.at[disability_onset_date.year] == approx(0.5, 1 / 364)
    assert all(
        [dalys_by_year_not_stacked.at[year] == (approx(1.0, 1 / 364))
         for year in range(death_date.year, sim.end_date.year + 1)]
    )

    # Check dalys_stacked_by_time is as expected:
    dalys_by_year_stacked_by_time = log['dalys_stacked'].loc[
        (log['dalys'].sex == sex), ['year', 'age_range', 'Label_A']
    ].groupby(['year', 'age_range'])['Label_A'].sum().unstack()
    assert dalys_by_year_stacked_by_time.loc[sim.start_date.year].sum() == 0.0
    assert dalys_by_year_stacked_by_time.loc[disability_onset_date.year].sum() == approx(0.5, 1 / 364)
    assert dalys_by_year_stacked_by_time.loc[death_date.year].sum() == approx(age_limit_for_yll - 2.0, 1 / 364)
    assert all([dalys_by_year_stacked_by_time.loc[year].sum() == (approx(0.0, 1 / 364)) for year in
                range(death_date.year + 1, sim.end_date.year + 1)])

    # Check dalys_stacked_by_age_and_time is as expected
    dalys_by_year_stacked_by_age_and_time = log['dalys_stacked_by_age_and_time'].loc[
        (log['dalys'].sex == sex), ['year', 'age_range', 'Label_A']
    ].groupby(['year', 'age_range'])['Label_A'].sum().unstack()
    assert 0.0 == dalys_by_year_stacked_by_age_and_time.loc[
        dalys_by_year_stacked_by_age_and_time.index != death_date.year,
        dalys_by_year_stacked_by_age_and_time.columns[
            dalys_by_year_stacked_by_age_and_time.columns != age_range_at_death]
    ].sum().sum()
    assert dalys_by_year_stacked_by_age_and_time[age_range_at_death].values == approx(
        dalys_by_year_stacked_by_time.sum(axis=1).values, 1 / 364)

    # Check that results from daly_stacked_by_time can be extracted into a pd.Series (for use in `extract_results`)
    def fn(df_):
        return df_.drop(columns='date').groupby(['year']).sum().stack()

    ser = fn(log['dalys_stacked'])
    assert ser.loc[(slice(None), 'Label_A')].at[disability_onset_date.year] == approx(0.5, 1 / 364)
    assert ser.loc[(slice(None), 'Label_A')].at[death_date.year] == approx(age_limit_for_yll - 2.0, 1 / 364)


def test_mapper_for_dalys_created(tmpdir, seed):
    """Check that causes of DALYS can be mapped between TLO cause and GBD cause in the case of a cause causing
    deaths but not disability (e.g. 'Congenital birth defects')."""

    class DiseaseThatCausesDeathOnly(Module):
        METADATA = {Metadata.DISEASE_MODULE}
        CAUSES_OF_DEATH = {
            'TLOCauseNameFor_CBD': Cause(gbd_causes='Congenital birth defects', label='Chosen_Label_For_CBD')
        }
        CAUSES_OF_DISABILITY = {}

        def read_parameters(self, resourcefilepath=None):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            pass

    start_date = Date(2010, 1, 1)
    sim = Simulation(start_date=start_date, seed=seed,
                     log_config={'filename': 'test_log', 'directory': tmpdir}, resourcefilepath=resourcefilepath)
    sim.register(
        demography.Demography(),
        enhanced_lifestyle.Lifestyle(),
        healthburden.HealthBurden(),
        DiseaseThatCausesDeathOnly(),
        sort_modules=False
    )
    sim.make_initial_population(n=100)
    sim.simulate(end_date=start_date)

    hblog = parse_log_file(sim.log_filepath)['tlo.methods.healthburden']
    disability_mapper_from_gbd_causes = extract_mapper(hblog['disability_mapper_from_gbd_cause_to_common_label'])
    disability_mapper_from_tlo_causes = extract_mapper(hblog['disability_mapper_from_tlo_cause_to_common_label'])
    daly_mapper_from_gbd_causes = extract_mapper(hblog['daly_mapper_from_gbd_cause_to_common_label'])
    daly_mapper_from_tlo_causes = extract_mapper(hblog['daly_mapper_from_tlo_cause_to_common_label'])

    demog = sim.modules['Demography']
    hb = sim.modules['HealthBurden']

    # 'Congenital birth defects' is a recognised cause of death and disability
    assert 'Congenital birth defects' in set(demog.parameters['gbd_causes_of_death_data'].columns)
    assert 'Congenital birth defects' in set(hb.parameters['gbd_causes_of_disability'])

    # As 'Congenital birth defects' is not defined as a disability per se, it does not need to feature in the
    # mappers for disabilities.
    assert disability_mapper_from_gbd_causes['Congenital birth defects'] == 'Other'
    assert 'TLOCauseNameFor_CBD' not in disability_mapper_from_tlo_causes

    # But... as 'Congenital birth defects' is defined as a cause of death by the TLO module, which can therefore cause a
    # loss of DALYS, the mappers for DALYS should attach it to the chosen label.
    assert daly_mapper_from_tlo_causes['TLOCauseNameFor_CBD'] == 'Chosen_Label_For_CBD'
    assert daly_mapper_from_gbd_causes['Congenital birth defects'] == 'Chosen_Label_For_CBD'


def test_get_mappers_in_fullmodel(tmpdir):
    """Check that `get_mappers_in_fullmodel` works as expected; and, in particular that things that cause death but not
    disability are captured correctly as a cause of DALYS (i.e., 'Congenital birth defects'). """

    mappers = get_mappers_in_fullmodel(resourcefilepath=resourcefilepath, outputpath=tmpdir)
    assert mappers['daly_mapper_from_gbd_cause_to_common_label']['Congenital birth defects'] != 'Other'
