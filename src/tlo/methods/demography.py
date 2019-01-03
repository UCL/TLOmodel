"""
The core demography module and its associated events.

Expects input in format of the 'Demography.xlsx'  of TimH, sent 3/10. Uses the 'Interpolated
population structure' worksheet within to initialise the age & sex distribution of population.
"""
from collections import defaultdict
import logging

import numpy as np
import pandas as pd

from tlo import Date
from tlo import Module, Parameter, Property, Types, DateOffset
from tlo.events import Event, PopulationScopeEventMixin, RegularEvent, IndividualScopeEventMixin

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Limits for setting up age range categories
MIN_AGE_FOR_RANGE = 0
MAX_AGE_FOR_RANGE = 100
AGE_RANGE_SIZE = 5


def make_age_range_lookup():
    """Returns a dictionary mapping age (in years) to age range
    i.e. { 0: '0-4', 1: '0-4', ..., 119: '100+', 120: '100+' }
    """
    def chunks(items, n):
        """Takes a list and divides it into parts of size n"""
        for index in range(0, len(items), n):
            yield items[index:index + n]

    # split all the ages from min to limit (100 years) into 5 year ranges
    parts = chunks(range(MIN_AGE_FOR_RANGE, MAX_AGE_FOR_RANGE), AGE_RANGE_SIZE)

    # any ages >= 100 are in the '100+' category
    default_category = '%d+' % MAX_AGE_FOR_RANGE
    lookup = defaultdict(lambda: default_category)

    # collect the possible ranges
    ranges = []

    # loop over each range and map all ages falling within the range to the range
    for part in parts:
        start = part.start
        end = part.stop - 1
        value = '%s-%s' % (start, end)
        ranges.append(value)
        for i in range(start, part.stop):
            lookup[i] = value

    ranges.append(default_category)
    return ranges, lookup


class Demography(Module):
    """
    The core demography modules handling age and sex of individuals. Also is responsible for their
    'is_alive' status
    """

    def __init__(self, name=None, workbook_path=None):
        super().__init__(name)
        self.workbook_path = workbook_path
    AGE_RANGE_CATEGORIES, AGE_RANGE_LOOKUP = make_age_range_lookup()

    # We should have 21 age range categories
    assert len(AGE_RANGE_CATEGORIES) == 21

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    PARAMETERS = {
        'interpolated_pop': Parameter(Types.DATA_FRAME, 'Interpolated population structure'),
        'fertility_schedule': Parameter(Types.DATA_FRAME, 'Age-spec fertility rates'),
        'mortality_schedule': Parameter(Types.DATA_FRAME, 'Age-spec fertility rates'),
        'fraction_of_births_male': Parameter(Types.REAL, 'Birth Sex Ratio')
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'date_of_birth': Property(Types.DATE, 'Date of birth of this individual'),
        'sex': Property(Types.CATEGORICAL, 'Male or female', categories=['M', 'F']),
        'mother_id': Property(Types.INT, 'Unique identifier of mother of this individual'),
        'is_alive': Property(Types.BOOL, 'Whether this individual is alive'),
        'is_pregnant': Property(Types.BOOL, 'Whether this individual is currently pregnant'),
        'date_of_last_pregnancy': Property(Types.DATE, 'Date of the last pregnancy of this individual'),
        'is_married': Property(Types.BOOL, 'Whether this individual is currently married'),
        'contraception': Property(Types.CATEGORICAL, 'Current contraceptive method',
                                  categories=['not using',
                                              'injections',
                                              'condom',
                                              'periodic abstinence',
                                              'norplant']),
        # Age information is handled by demography module (was part of core module)
        'age_exact_years': Property(Types.REAL, 'The age of the individual in exact years'),
        'age_years': Property(Types.INT, 'The age of the individual in years'),
        'age_range': Property(Types.CATEGORICAL,
                              'The age range category of the individual',
                              categories=AGE_RANGE_CATEGORIES),
    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Loads the 'Interpolated Pop Structure' worksheet from the Demography Excel workbook.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        workbook = pd.read_excel(self.workbook_path, sheet_name=None)
        self.parameters['interpolated_pop'] = workbook['Interpolated Pop Structure']
        self.parameters['fertility_schedule'] = workbook['Age_spec fertility']
        self.parameters['mortality_schedule'] = workbook['Mortality Rate']
        self.parameters['fraction_of_births_male'] = 0.5

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """
        df = population.props

        worksheet = self.parameters['interpolated_pop']

        # get a subset of the rows from the worksheet
        intpop = worksheet.loc[worksheet.year == self.sim.date.year].copy().reset_index()
        intpop['probability'] = intpop.value / intpop.value.sum()
        intpop['month_range'] = 12
        is_age_range = (intpop['age_to'] != intpop['age_from'])
        intpop.loc[is_age_range, 'month_range'] = (intpop['age_to'] - intpop['age_from']) * 12

        pop_sample = intpop.iloc[self.sim.rng.choice(intpop.index.values,
                                                     size=len(df),
                                                     p=intpop.probability.values)]
        pop_sample = pop_sample.reset_index()
        months = pd.Series(pd.to_timedelta(self.sim.rng.randint(low=0,
                                                                high=12,
                                                                size=len(df)),
                                           unit='M',
                                           box=False))

        df.date_of_birth = self.sim.date - (pd.to_timedelta(pop_sample['age_from'], unit='Y') + months)
        df.sex.values[:] = pop_sample['gender'].map({'female': 'F', 'male': 'M'})
        df.mother_id = -1  # we can't use np.nan because that casts the series into a float
        df.is_alive = True

        # assign that half the adult population is married (will be done in lifestyle module)
        df.is_married = False  # TODO: Lifestyle module should look after married property

        adults = (df.age_years >= 18)
        df.loc[adults, 'is_married'] = self.sim.rng.choice([True, False],
                                                           size=adults.sum(),
                                                           p=[0.5, 0.5])

        # assign that none of the adult (woman) population is pregnant
        df.is_pregnant = False
        df.date_of_last_pregnancy = pd.NaT

        df.contraception.values[:] = 'not using'  # this will be ascribed by the lifestype module
        # TODO: Lifestyle module should look after contraception property

        age_in_days = self.sim.date - df.date_of_birth
        df.age_exact_years = age_in_days / np.timedelta64(1, 'Y')
        df.age_years = df.age_exact_years.astype(int)
        df.age_range.values[:] = df.age_years.map(self.AGE_RANGE_LOOKUP).astype('category')

    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """
        # Update age information every day
        sim.schedule_event(AgeUpdateEvent(self, self.AGE_RANGE_LOOKUP),
                           sim.date + DateOffset(days=1))

        # check all population to determine if pregnancy should be triggered (repeats every month)
        sim.schedule_event(PregnancyPoll(self), sim.date+DateOffset(months=1))

        # check all population to determine if person should die (from causes other than those
        # explicitly modelled) (repeats every month)
        sim.schedule_event(OtherDeathPoll(self), sim.date+DateOffset(months=1))

        # Launch the repeating event that will store statistics about the population structure
        sim.schedule_event(DemographyLoggingEvent(self), sim.date + DateOffset(months=12))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        df = self.sim.population.props

        f_male = self.parameters['fraction_of_births_male']
        df.at[child_id, 'sex'] = self.sim.rng.choice(['M', 'F'], p=[f_male, 1 - f_male])

        df.at[child_id, 'date_of_birth'] = self.sim.date
        df.at[child_id, 'mother_id'] = mother_id
        df.at[child_id, 'is_alive'] = True
        df.at[child_id, 'is_pregnant'] = False
        df.at[child_id, 'is_married'] = False
        df.at[child_id, 'date_of_last_pregnancy'] = pd.NaT
        df.at[child_id, 'contraception'] = 'not using'  # TODO: contraception should be governed by lifestyle module

        logger.debug('A new child has been born: %s', child_id)


class AgeUpdateEvent(RegularEvent, PopulationScopeEventMixin):
    """
    This event updates the age_exact_years, age_years and age_range columns for the population based
    on the current simulation date
    """
    def __init__(self, module, age_range_lookup):
        super().__init__(module, frequency=DateOffset(days=1))
        self.age_range_lookup = age_range_lookup

    def apply(self, population):
        df = population.props
        age_in_days = population.sim.date - df.date_of_birth

        df.age_exact_years = age_in_days / np.timedelta64(1, 'Y')
        df.age_years = df.age_exact_years.astype(int)
        df.age_range.values[:] = df.age_years.map(self.age_range_lookup).astype('category')


class PregnancyPoll(RegularEvent, PopulationScopeEventMixin):
    """
    This event looks across each woman in the population to determine who will become pregnant
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):

        logger.debug('Checking to see if anyone should become pregnant....')

        if self.sim.date > Date(2035, 1, 1):
            logger.info('Now after 2035')

        df = population.props  # get the population dataframe

        # get the subset of women from the population dataframe and relevant characteristics
        female = df.loc[(df.sex == 'F') & df.is_alive, ['contraception', 'is_married', 'is_pregnant', 'age_years']]

        # load the fertility schedule (imported datasheet from excel workbook)
        fert_schedule = self.module.parameters['fertility_schedule']

        # --------

        # get the probability of pregnancy for each woman in the model, through merging with the fert_schedule data
        female = female.reset_index().merge(fert_schedule,
                                            left_on=['age_years', 'contraception'],
                                            right_on=['age', 'cmeth'], how='left').set_index('person')

        # clean up items that didn't merge (those with ages not referenced in the fert_schedule sheet)
        female = female[female.basefert_dhs.notna()]

        # zero-out risk of pregnancy if already pregnant
        female.loc[female.is_pregnant, 'basefert_dhs'] = 0

        # flipping the coin to determine if this woman will become pregnant
        newlypregnant = (self.sim.rng.random_sample(size=len(female)) < female.basefert_dhs / 12)

        # the imported number is a yearly proportion. So adjust the rate according
        # to the frequency with which the event is recurring
        # TODO: this should be linked to the self.frequency value

        # updating the pregancy status for that women
        df.loc[female.index[newlypregnant], 'is_pregnant'] = True
        df.loc[female.index[newlypregnant], 'date_of_last_pregnancy'] = self.sim.date

        # loop through each newly pregnant women in order to schedule them a 'delayed birth event'
        personnumbers_of_newlypregnant = female.index[newlypregnant]
        for personnumber in personnumbers_of_newlypregnant:

            logger.debug('Woman number: %s', personnumber)
            logger.debug('Her age is: %s', female.loc[personnumber, 'age_years'])

            # schedule the birth event for this woman (9 months plus/minus 2 wks)
            birth_date_of_child = self.sim.date + DateOffset(months=9) + DateOffset(weeks=-2 + 4 * self.sim.rng.random_sample())

            # Schedule the Birth
            birth = DelayedBirthEvent(self.module, personnumber)
            self.sim.schedule_event(birth, birth_date_of_child)

            logger.debug('The birth is now booked for: %s', birth_date_of_child)


class DelayedBirthEvent(Event, IndividualScopeEventMixin):
    """A one-off event in which a pregnant mother gives birth.
    """

    def __init__(self, module, mother_id):
        """Create a new birth event.

        We need to pass the person this event happens to to the base class constructor
        using super(). We also pass the module that created this event, so that random
        number generators can be scoped per-module.

        :param module: the module that created this event
        :param mother_id: the person giving birth
        """
        super().__init__(module, person=mother_id)

    def apply(self, mother_id):
        """Apply this event to the given person.
        Assuming the person is still alive, we ask the simulation to create a new offspring.
        :param mother_id: the person the event happens to, i.e. the mother giving birth
        """

        logger.debug('@@@@ A Birth is now occuring, to mother %s', mother_id)

        df = self.sim.population.props

        if df.at[mother_id, 'is_alive']:
            self.sim.do_birth(mother_id)

        # Reset the mother's is_pregnant status showing that she is no longer pregnant
        df.at[mother_id, 'is_pregnant'] = False

        # Log the birth:
        logger.info('%s:%s:%s %s', self.sim.strdate, self.__class__.__name__,
                    df.at[mother_id, 'age_years'], 0)


class OtherDeathPoll(RegularEvent, PopulationScopeEventMixin):
    """
    This event looks across each person in the population to see if they should die.
    During development this will be applying WPP all-cause death rates...
    ... But once we have diseases causing their own deaths, it will become the mortality schedule
    ... from all cases other than those explicitly modelled
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        logger.debug('Checking to see if anyone should die...')

        # get the mortality schedule for now...

        # load the mortality schedule (imported datasheet from excel workbook)
        mort_schedule = self.module.parameters['mortality_schedule']

        # get the subset of mortality rates for this year.
        mort_schedule = mort_schedule.loc[mort_schedule.year == self.sim.date.year].copy()

        # create new variable that will align with population.sex
        mort_schedule['sex'] = np.where(mort_schedule['gender'] == 'male', 'M', 'F')

        # get the population
        df = population.props

        # --------
        # add age-groups to each dataframe (this to be done by the population object later)
        # NB... the age-groups here are -1:<0 year-olds; 0:1-4 yearolds; 1:5-9 yearolds; etc..)
        df['agegrp'] = -99
        df['agegrp'] = 1 + np.floor(df.age_years / 5)
        df.loc[df.age_years == 0, 'agegrp'] = -1  # overwriting with -1 for the <0 year-olds

        mort_schedule['agegrp'] = -99
        mort_schedule['agegrp'] = np.floor(mort_schedule.age_from / 5)
        mort_schedule.loc[mort_schedule.age_from == 0, 'agegrp'] = -1
        # -------

        # merge the popualtion dataframe with the parameter dataframe to pick-up the risk of
        # mortality for each person in the model
        df = df.reset_index().merge(mort_schedule,
                                    left_on=['agegrp', 'sex'],
                                    right_on=['agegrp', 'sex'],
                                    how='left').set_index('person')

        # flipping the coin to determine if this woman will die
        outcome = (self.sim.rng.random_sample(size=len(df)) < df.value / 12)

        # the imported number is a yearly proportion. So adjust the rate according
        # to the frequency with which the event is recurring
        # TODO: this should be linked to the self.frequency value

        # loop through to see who is going to die:
        willdie = (df[outcome]).index
        for person in willdie:
            death = InstantaneousDeath(self.module, person, cause='Other')
            # schedule the death for "now"
            self.sim.schedule_event(death, self.sim.date)


class InstantaneousDeath(Event, IndividualScopeEventMixin):
    """
    Performs the Death operation on an individual and logs it.
    """
    def __init__(self, module, individual_id, cause):
        super().__init__(module, person=individual_id)
        self.cause = cause

    def apply(self, individual_id):
        df = self.sim.population.props

        logger.debug("@@@@ A Death is now occuring, to person %s", individual_id)

        if df.at[individual_id, 'is_alive']:
            # here comes the death..
            df.at[individual_id, 'is_alive'] = False
            # the person is now dead

        logger.debug("*******************************************The person %s "
                     "is now officially dead and has died of %s", individual_id, self.cause)

        # Log the death
        logger.info('%s:%s:%s %s', self.sim.strdate, self.__class__.__name__,
                    df.at[individual_id, 'age_years'], self.cause)


class DemographyLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """comments...
        """
        # run this event every 12 months (every year)
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        df = population.props

        sex_count = df[df.is_alive].groupby('sex').size()

        logger.info('%s:%s %s:%s', self.sim.strdate, self.__class__.__name__,
                    'Total', sum(sex_count))

        logger.info('%s:%s %s:%s', self.sim.strdate, self.__class__.__name__,
                    'SexM', sex_count['M'])

        logger.info('%s:%s %s:%s', self.sim.strdate, self.__class__.__name__,
                    'SexF', sex_count['F'])

        m_age_counts = df[df.is_alive & (df.sex == 'M')].groupby('age_range').size()
        f_age_counts = df[df.is_alive & (df.sex == 'F')].groupby('age_range').size()

        logger.info('%s:%s %s:%s', self.sim.strdate, self.__class__.__name__,
                    'AgesM', ','.join(map(str, m_age_counts)))

        logger.info('%s:%s %s:%s', self.sim.strdate, self.__class__.__name__,
                    'AgesF', ','.join(map(str, f_age_counts)))

