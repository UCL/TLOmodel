"""
A skeleton template for disease methods.
"""
import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, PopulationScopeEventMixin, RegularEvent, IndividualScopeEventMixin


class health_system(Module):
    """ routinely tests proportion of the population and
    determines availability of ART for HIV+ dependent on UNAIDS coverage estimates
    """

    def __init__(self, name=None, workbook_path=None):
        super().__init__(name)
        self.workbook_path = workbook_path
        self.store = {'Time': [], 'Number_tested': [], 'Number_treated': []}

    # TODO: extract testing rates from Leigh's paper by single years of age
    PARAMETERS = {
        'testing_coverage_male': Parameter(Types.REAL, 'proportion of adult male population tested'),
        'testing_coverage_female': Parameter(Types.REAL, 'proportion of adult female population tested'),
        'testing_prob_individual': Parameter(Types.REAL, 'probability of individual being tested after trigger event'),
        'art_coverage': Parameter(Types.DATA_FRAME, 'estimated ART coverage'),
        'rr_testing_high_risk': Parameter(Types.DATA_FRAME,
                                          'relative increase in testing probability if high sexual risk'),
        'previously_negative': Parameter(Types.DATA_FRAME, 'previous negative HIV test result'),
        'previously_positive': Parameter(Types.DATA_FRAME, 'previous positive HIV test result'),

    }

    PROPERTIES = {
        'ever_tested': Property(Types.BOOL, 'ever had a hiv test'),
        'date_tested': Property(Types.DATE, 'date of hiv test'),
        'hiv_diagnosed': Property(Types.BOOL, 'hiv+ and tested'),
        'on_art': Property(Types.BOOL, 'on art'),
        'date_art_start': Property(Types.DATE, 'date art started')

    }

    def read_parameters(self, data_folder):
        params = self.parameters
        params['param_list'] = pd.read_excel(self.workbook_path,
                                             sheet_name='parameters')

        self.param_list.set_index("Parameter", inplace=True)

        params['testing_coverage_male'] = self.param_list.loc['testing_coverage_male_2010', 'Value1']
        params['testing_coverage_female'] = self.param_list.loc['testing_coverage_female_2010', 'Value1']
        params['testing_prob_individual'] = self.param_list.loc['testing_prob_individual', 'Value1']  # dummy value
        params['art_coverage'] = self.param_list.loc['art_coverage', 'Value1']
        params['rr_testing_high_risk'] = self.param_list.loc['rr_testing_high_risk', 'Value1']
        params['previously_negative'] = self.param_list.loc['previously_negative', 'Value1']
        params['previously_positive'] = self.param_list.loc['previously_positive', 'Value1']

        self.parameters['initial_art_coverage'] = pd.read_excel(self.workbook_path,
                                                                sheet_name='coverage')

        self.parameters['testing_rates'] = pd.read_excel(self.workbook_path,
                                                         sheet_name='testing_rates')

    def initialise_population(self, population):
        """ set the default values for the new fields
        """
        df = population.props

        df['ever_tested'] = False  # default: no individuals tested
        df['date_tested'] = pd.NaT
        df['hiv_diagnosed'] = False
        df['on_art'] = False
        df['date_art_start'] = pd.NaT

        self.baseline_tested(population)  # allocate baseline art coverage
        self.baseline_art(population)  # allocate baseline art coverage

    def baseline_tested(self, population):
        """ assign initial art coverage levels
        """
        now = self.sim.date
        df = population.props

        # add age to population.props
        df_age = pd.merge(df, population.age, left_index=True, right_index=True, how='left')

        # get a list of random numbers between 0 and 1 for the whole population
        random_draw = self.sim.rng.random_sample(size=len(df_age))

        # probability of baseline population ever testing for HIV
        art_index_male = df_age.index[
            (random_draw < self.parameters['testing_coverage_male']) & df_age.is_alive & (df_age.sex == 'M') & (
                df_age.years >= 15)]
        # print('art_index: ', art_index)

        art_index_female = df_age.index[
            (random_draw < self.parameters['testing_coverage_female']) & df_age.is_alive & (df_age.sex == 'F') & (
                df_age.years >= 15)]

        # we don't know date tested, assume date = now
        df.loc[art_index_male | art_index_female, 'ever_tested'] = True
        df.loc[art_index_male | art_index_female, 'date_tested'] = now

        # outcome of test
        diagnosed_idx = df_age.index[df.ever_tested & df.is_alive & df.has_hiv]
        df.loc[diagnosed_idx, 'hiv_diagnosed'] = True

    def baseline_art(self, population):
        """ assign initial art coverage levels
        """
        now = self.sim.date
        df = population.props

        worksheet = self.parameters['initial_art_coverage']

        coverage = worksheet.loc[worksheet.year == now.year, ['year', 'single_age', 'sex', 'prop_coverage']]
        # print('coverage: ', coverage.head(20))

        # add age to population.props
        df_age = pd.merge(df, population.age, left_index=True, right_index=True, how='left')
        # print('df_with_age: ', df_with_age.head(10))

        # merge all susceptible individuals with their coverage probability based on sex and age
        df_age = df_age.merge(coverage,

                              left_on=['years', 'sex'],

                              right_on=['single_age', 'sex'],

                              how='left')

        # no data for ages 100+ so fill missing values with 0
        df_age['prop_coverage'] = df_age['prop_coverage'].fillna(0)
        # print('df_with_age_art_prob: ', df_with_age_art_prob.head(20))

        assert df_age.prop_coverage.isna().sum() == 0  # check there is a probability for every individual

        # get a list of random numbers between 0 and 1 for the whole population
        random_draw = self.sim.rng.random_sample(size=len(df_age))

        # probability of baseline population receiving art: requirement = ever_tested
        art_index = df_age.index[
            (random_draw < df_age.prop_coverage) & df_age.has_hiv & df.ever_tested & df.is_alive]
        # print('art_index: ', art_index)

        df.loc[art_index, 'on_art'] = True
        df.loc[art_index, 'date_art_start'] = now

    def initialise_simulation(self, sim):
        sim.schedule_event(TestingEvent(self), sim.date + DateOffset(months=12))
        sim.schedule_event(TreatmentEvent(self), sim.date + DateOffset(months=12))

        # add an event to log to screen
        sim.schedule_event(HealthSystemLoggingEvent(self), sim.date + DateOffset(months=1))

    def on_birth(self, mother, child):
        pass


class TestingEvent(RegularEvent, PopulationScopeEventMixin):
    """A skeleton class for an event
    Regular events automatically reschedule themselves at a fixed frequency,
    and thus implement discrete timestep type behaviour. The frequency is
    specified when calling the base class constructor in our __init__ method.
    """

    def __init__(self, module):
        """One line summary here

        """
        super().__init__(module, frequency=DateOffset(months=12))  # every 12 months
        # make sure any rates are annual if frequency of event is annual

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """

        params = self.module.parameters
        now = self.sim.date
        df = population.props

        # get a list of random numbers between 0 and 1 for the whole population
        random_draw = self.sim.rng.random_sample(size=len(df))

        df_age = pd.merge(df, population.age, left_index=True, right_index=True, how='left')

        # testing rates are for >=15 years only
        df_with_rates = pd.merge(df_age, params['testing_rates'], left_on=['years', 'sex'], right_on=['age', 'sex'],
                                 how='left')
        # print(df_with_rates.head(10))

        high_risk_idx = df_with_rates.index[df_with_rates.sexual_risk_group == 'high']
        sex_work_idx = df_with_rates.index[df_with_rates.sexual_risk_group == 'sex_work']

        # increased testing rate if high risk
        df_with_rates.loc[high_risk_idx | sex_work_idx, 'testing_rates'] *= params['rr_testing_high_risk']
        print(df_with_rates.head(30))

        df_with_rates.loc[df_with_rates.ever_tested & df_with_rates.hiv_diagnosed, 'testing_rates'] *= params[
            'previously_positive']
        df_with_rates.loc[df_with_rates.ever_tested & ~df_with_rates.hiv_diagnosed, 'testing_rates'] *= params[
            'previously_negative']

        # probability of HIV testing, can allow repeat testing
        # if repeat testing, will only store latest test date
        # TODO: if needed, could add a counter for number of hiv tests per person (useful for costing?)
        testing_index = df_with_rates.index[
            (random_draw < df_with_rates.testing_rates) & df_with_rates.is_alive & (df_with_rates.years >= 15)]
        print('testing index', testing_index)

        df.loc[testing_index, 'ever_tested'] = True
        df.loc[testing_index, 'date_tested'] = now

        diagnosed_index = df.index[(df.date_tested == now) & df.is_alive & df.has_hiv]
        # print('testing_index: ', testing_index)
        # print('diagnosed_index: ', testing_diagnosed_index)

        df.loc[diagnosed_index, 'hiv_diagnosed'] = True

        # TODO: include infant testing (soon after birth?)


class IndividualTesting(Event, IndividualScopeEventMixin):
    """ allows hiv tests for individuals triggered by certain events
    e.g. pregnancy or tb diagnosis
    """

    def __init__(self, module, individual):
        super().__init__(module, person=individual)

    def apply(self, individual):
        params = self.module.parameters

        if individual.is_alive & ~individual.hiv_diagnosed:
            # probability of HIV testing
            individual.ever_tested = np.random.choice([True, False], size=1, p=[params['testing_prob_individual'],
                                                                                1 - params['testing_prob_individual']])

            individual.loc[individual.ever_tested, 'date_tested'] = self.sim.date

            individual.loc[individual.ever_tested & individual.is_alive & individual.has_hiv, 'hiv_diagnosed'] = True


# TODO: decide how to define probability of treatment / rates of ART initiation
class TreatmentEvent(RegularEvent, PopulationScopeEventMixin):
    """ assigning ART to diagnosed HIV+ people
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=12))  # every 12 months
        # make sure any rates are annual if frequency of event is annual

    def apply(self, population):
        params = self.module.parameters
        now = self.sim.date
        df = population.props

        # get a list of random numbers between 0 and 1 for the whole population
        random_draw = self.sim.rng.random_sample(size=len(df))

        # probability of HIV treatment
        treatment_index = df.index[(random_draw < params['art_coverage']) & df.has_hiv & df.hiv_diagnosed &
                                   df.is_alive & ~df.on_art]
        # print('treatment_index: ', treatment_index)

        df.loc[treatment_index, 'on_art'] = True
        df.loc[treatment_index, 'date_art_start'] = now


class HealthSystemLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """ produce some outputs to check
        """
        # run this event every 12 months (every year)
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props

        mask = (df['date_tested'] > self.sim.date - DateOffset(months=self.repeat))
        recently_tested = mask.sum()

        currently_on_art = len(df[df.on_art & df.is_alive])

        self.module.store['Time'].append(self.sim.date)
        self.module.store['Number_tested'].append(recently_tested)
        self.module.store['Number_treated'].append(currently_on_art)
