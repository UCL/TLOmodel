"""
The core demography module and its associated events.

Expects input in format of the 'Demography.xlsx'  of TimH, sent 3/10. Uses the 'Interpolated
population structure' worksheet within to initialise the age & sex distribution of population.
"""
import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from tlo import Date, DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent

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
        'is_alive': Property(Types.BOOL, 'Whether this individual is alive'),
        'date_of_birth': Property(Types.DATE, 'Date of birth of this individual'),
        'sex': Property(Types.CATEGORICAL, 'Male or female', categories=['M', 'F']),
        'mother_id': Property(Types.INT, 'Unique identifier of mother of this individual'),
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
        months = pd.Series(pd.to_timedelta(self.sim.rng.randint(low=0, high=12, size=len(df)),
                                           unit='M',
                                           box=False))

        # The entire initial population is alive!
        df.is_alive = True

        df.loc[df.is_alive, 'date_of_birth'] = self.sim.date - (pd.to_timedelta(pop_sample['age_from'], unit='Y')
                                                                + months)
        df.loc[df.is_alive, 'sex'] = pop_sample['gender'].map({'female': 'F', 'male': 'M'})
        df.loc[df.is_alive, 'mother_id'] = -1  # we can't use np.nan because that casts the series into a float

        # assign that none of the adult (woman) population is pregnant
        df.loc[df.is_alive, 'is_pregnant'] = False
        df.loc[df.is_alive, 'date_of_last_pregnancy'] = pd.NaT

        # TODO: Lifestyle module should look after contraception property
        df.loc[df.is_alive, 'contraception'] = 'not using'  # this will be ascribed by the lifestype module

        age_in_days = self.sim.date - df.loc[df.is_alive, 'date_of_birth']
        df.loc[df.is_alive, 'age_exact_years'] = age_in_days / np.timedelta64(1, 'Y')
        df.loc[df.is_alive, 'age_years'] = df.loc[df.is_alive, 'age_exact_years'].astype(int)
        df.loc[df.is_alive, 'age_range'] = df.loc[df.is_alive, 'age_years'].map(self.AGE_RANGE_LOOKUP)

        # assign that half the adult population is married (will be done in lifestyle module)
        df.loc[df.is_alive, 'is_married'] = False  # TODO: Lifestyle module should look after married property
        adults = (df.loc[df.is_alive, 'age_years'] >= 18)
        df.loc[adults, 'is_married'] = self.sim.rng.choice([True, False], size=adults.sum(), p=[0.5, 0.5], replace=True)

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

        df.at[child_id, 'is_alive'] = True
        df.at[child_id, 'date_of_birth'] = self.sim.date

        f_male = self.parameters['fraction_of_births_male']
        df.at[child_id, 'sex'] = self.sim.rng.choice(['M', 'F'], p=[f_male, 1 - f_male])

        df.at[child_id, 'mother_id'] = mother_id
        df.at[child_id, 'is_pregnant'] = False
        df.at[child_id, 'date_of_last_pregnancy'] = pd.NaT

        df.at[child_id, 'is_married'] = False
        df.at[child_id, 'contraception'] = 'not using'  # TODO: contraception should be governed by lifestyle module

        df.at[child_id, 'age_exact_years'] = 0.0
        df.at[child_id, 'age_years'] = 0
        df.at[child_id, 'age_range'] = self.AGE_RANGE_LOOKUP[0]

        # Reset the mother's is_pregnant status showing that she is no longer pregnant
        df.at[mother_id, 'is_pregnant'] = False

        # Log the birth:
        logger.info('%s:on_birth:%s',
                    self.sim.strdate,
                    {
                        'mother': mother_id,
                        'child': child_id,
                        'mother_age': df.at[mother_id, 'age_years'],
                        'xxx': 0
                    })


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
        age_in_days = population.sim.date - df.loc[df.is_alive, 'date_of_birth']

        df.loc[df.is_alive, 'age_exact_years'] = age_in_days / np.timedelta64(1, 'Y')
        df.loc[df.is_alive, 'age_years'] = df.loc[df.is_alive, 'age_exact_years'].astype(int)
        df.loc[df.is_alive, 'age_range'] = df.age_years.map(self.age_range_lookup)


class PregnancyPoll(RegularEvent, PopulationScopeEventMixin):
    """
    This event looks across each woman in the population to determine who will become pregnant
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        self.age_low = 15
        self.age_high = 49

    def apply(self, population):

        logger.debug('Checking to see if anyone should become pregnant....')

        if self.sim.date > Date(2035, 1, 1):
            logger.debug('Now after 2035')

        df = population.props  # get the population dataframe

        # get the subset of women from the population dataframe and relevant characteristics
        subset = (df.sex == 'F') & df.is_alive & df.age_years.between(self.age_low, self.age_high) & ~df.is_pregnant
        females = df.loc[subset, ['contraception', 'age_years']]

        # load the fertility schedule (imported datasheet from excel workbook)
        fertility_schedule = self.module.parameters['fertility_schedule']

        # --------

        # get the probability of pregnancy for each woman in the model, through merging with the fert_schedule data
        len_before_merge = len(females)
        females = females.reset_index().merge(fertility_schedule,
                                              left_on=['age_years', 'contraception'],
                                              right_on=['age', 'cmeth'],
                                              how='inner').set_index('person')
        assert len(females) == len_before_merge

        # flipping the coin to determine if this woman will become pregnant
        newly_pregnant = (self.sim.rng.random_sample(size=len(females)) < females.basefert_dhs / 12)

        # the imported number is a yearly proportion. So adjust the rate according
        # to the frequency with which the event is recurring
        # TODO: this should be linked to the self.frequency value

        # updating the pregancy status for that women
        df.loc[females.index[newly_pregnant], 'is_pregnant'] = True
        df.loc[females.index[newly_pregnant], 'date_of_last_pregnancy'] = self.sim.date

        # loop through each newly pregnant women in order to schedule them a 'delayed birth event'
        newly_pregnant_id = females.index[newly_pregnant]
        for female_id in newly_pregnant_id:
            logger.debug('female %d pregnant at age: %d', female_id, females.at[female_id, 'age_years'])

            # schedule the birth event for this woman (9 months plus/minus 2 wks)
            date_of_birth = self.sim.date + \
                DateOffset(months=9) + \
                DateOffset(weeks=-2 + 4 * self.sim.rng.random_sample())

            # Schedule the Birth
            self.sim.schedule_event(DelayedBirthEvent(self.module, female_id),
                                    date_of_birth)

            logger.debug('birth booked for: %s', date_of_birth)


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
        super().__init__(module, person_id=mother_id)

    def apply(self, mother_id):
        """Apply this event to the given person.
        Assuming the person is still alive, we ask the simulation to create a new offspring.
        :param mother_id: the person the event happens to, i.e. the mother giving birth
        """

        logger.debug('@@@@ A Birth is now occuring, to mother %s', mother_id)

        df = self.sim.population.props

        # If the mother is alive and still pregnant
        if df.at[mother_id, 'is_alive'] and df.at[mother_id, 'is_pregnant']:
            self.sim.do_birth(mother_id)


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
        mort_sched = self.module.parameters['mortality_schedule']

        # round current year to closest year in spreadsheet
        # TODO: currently rounding to the closest 5-year entry - fix or remain?
        closest_year = int(round(self.sim.date.year / 5) * 5)

        # get the subset of mortality rates for this year.
        mort_sched = mort_sched.loc[mort_sched.year == closest_year, ['age_from', 'gender', 'value']].copy()

        # create new variable that will align with population.sex
        mort_sched['sex'] = np.where(mort_sched['gender'] == 'male', 'M', 'F')

        # get the population
        df = population.props

        alive = df.loc[df.is_alive, ['sex', 'age_years']].copy()

        # --------
        # TODO: mortality schedule worksheet should be updated to use the property date ranges
        # add age-groups to each dataframe (this to be done by the population object later)
        # NB... the age-groups here are -1:<0 year-olds; 0:1-4 yearolds; 1:5-9 yearolds; etc..)
        alive['agegrp'] = -99
        alive['agegrp'] = 1 + np.floor(alive.age_years / 5)
        alive.loc[alive.age_years == 0, 'agegrp'] = -1  # overwriting with -1 for the <0 year-olds

        mort_sched['agegrp'] = -99
        mort_sched['agegrp'] = np.floor(mort_sched.age_from / 5)
        mort_sched.loc[mort_sched.age_from == 0, 'agegrp'] = -1
        # -------

        # merge the popualtion dataframe with the parameter dataframe to pick-up the risk of
        # mortality for each person in the model
        len_before_merge = len(alive)
        alive = alive.reset_index().merge(mort_sched,
                                          left_on=['agegrp', 'sex'],
                                          right_on=['agegrp', 'sex'],
                                          how='inner').set_index('person')
        assert len(alive) == len_before_merge

        # flipping the coin to determine if this person will die
        will_die = (self.sim.rng.random_sample(size=len(alive)) < alive.value / 12)

        # the imported number is a yearly proportion. So adjust the rate according
        # to the frequency with which the event is recurring
        # TODO: this should be linked to the self.frequency value

        # loop through to see who is going to die:
        for person in alive.index[will_die]:
            # schedule the death for "now"
            self.sim.schedule_event(InstantaneousDeath(self.module, person, cause='Other'),
                                    self.sim.date)


class InstantaneousDeath(Event, IndividualScopeEventMixin):
    """
    Performs the Death operation on an individual and logs it.
    """
    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)
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
        logger.info('%s:death:%s', self.sim.strdate,
                    {
                        'age': df.at[individual_id, 'age_years'],
                        'cause': self.cause
                    })


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

        logger.info('%s:population:%s',
                    self.sim.strdate,
                    {
                        'total': sum(sex_count),
                        'male': sex_count['M'],
                        'female': sex_count['F']
                    })

        # if you groupby both sex and age_range, you weirdly lose categories where size==0, so
        # get the counts separately
        m_age_counts = df[df.is_alive & (df.sex == 'M')].groupby('age_range').size()
        f_age_counts = df[df.is_alive & (df.sex == 'F')].groupby('age_range').size()

        logger.info('%s:age_range_m:%s', self.sim.strdate,
                    list(m_age_counts))

        logger.info('%s:age_range_f:%s', self.sim.strdate,
                    list(f_age_counts))
