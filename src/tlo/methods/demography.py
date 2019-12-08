"""
The core demography module and its associated events.
Expects input in format of the 'Demography.xlsx'  of TimH, sent 3/10. Uses the 'Interpolated
population structure' worksheet within to initialise the age & sex distribution of population.
"""

import logging
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Limits for setting up age range categories
MIN_AGE_FOR_RANGE = 0
MAX_AGE_FOR_RANGE = 100
AGE_RANGE_SIZE = 5

MAX_AGE = 120


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

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    AGE_RANGE_CATEGORIES, AGE_RANGE_LOOKUP = make_age_range_lookup()

    # We should have 21 age range categories
    assert len(AGE_RANGE_CATEGORIES) == 21

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    PARAMETERS = {
        'interpolated_pop': Parameter(Types.DATA_FRAME, 'Interpolated population structure'),
        'mortality_schedule': Parameter(Types.DATA_FRAME, 'Age-spec fertility rates'),
        'fraction_of_births_male': Parameter(Types.REAL, 'Birth Sex Ratio'),
        'district_and_region_data': Parameter(Types.DATA_FRAME,
                                              'Census data on the number of persons in '
                                              'residence in each district')
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'is_alive': Property(Types.BOOL, 'Whether this individual is alive'),
        'date_of_birth': Property(Types.DATE, 'Date of birth of this individual'),
        'date_of_death': Property(Types.DATE, 'Date of death of this individual'),
        'sex': Property(Types.CATEGORICAL, 'Male or female', categories=['M', 'F']),
        'mother_id': Property(Types.INT, 'Unique identifier of mother of this individual'),
        # Age calculation is handled by demography module
        'age_exact_years': Property(Types.REAL, 'The age of the individual in exact years'),
        'age_years': Property(Types.INT, 'The age of the individual in years'),
        'age_range': Property(Types.CATEGORICAL,
                              'The age range category of the individual',
                              categories=AGE_RANGE_CATEGORIES),
        'age_days': Property(Types.INT, 'The age of the individual in whole days'),

        'region_of_residence': Property(Types.STRING, 'The region in which the person in resident'),
        'district_of_residence': Property(Types.STRING, 'The district in which the person is resident'),
    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.
        Loads the 'Interpolated Pop Structure' worksheet from the Demography Excel workbook.
        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        workbook = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_DemographicData.xlsx',
                                 sheet_name=None)

        self.parameters['interpolated_pop'] = workbook['Interpolated Pop Structure']

        # create new variable that will align with population.sex
        ms = workbook['Mortality Rate']
        ms['sex'] = ms.gender.map({'male': 'M', 'female': 'F'})
        # long-list the column to avoid the problem about these being irregular age-groups
        ms_new = []
        for row in ms.itertuples():
            age_high = row.age_to
            if age_high == 99:
                age_high = MAX_AGE
            for age_years in range(row.age_from, age_high + 1):
                ms_new.append(row._replace(age_from=age_years))
        ms_new = pd.DataFrame(ms_new)
        ms_new = ms_new.drop('age_to', axis=1)  # delete the un-needed column
        self.parameters['mortality_schedule'] = ms_new.rename(columns={'age_from': 'age_years'})
        self.parameters['fraction_of_births_male'] = 0.5

        self.parameters['district_and_region_data'] = \
            pd.read_csv(Path(self.resourcefilepath) / 'ResourceFile_District_Population_Data.csv')

    def initialise_population(self, population):
        """Set our property values for the initial population.
        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.
        :param population: the population of individuals
        """
        df = population.props

        df.date_of_death = pd.NaT

        worksheet = self.parameters['interpolated_pop']

        # get a subset of the rows from the interpolated population worksheet
        intpop = worksheet.loc[worksheet.year == self.sim.date.year].copy().reset_index()

        # get the probability of occurrence for each sex+age range in the population
        intpop['probability'] = intpop.value / intpop.value.sum()

        # calculate the month range interval for age ranges which are exactly 1 year
        intpop['month_range'] = 12
        is_age_range = (intpop['age_to'] != intpop['age_from'])
        intpop.loc[is_age_range, 'month_range'] = (intpop['age_to'] - intpop['age_from']) * 12

        # randomly rows indices from the worksheet
        pop_sample = intpop.iloc[self.rng.choice(intpop.index.values,
                                                 size=len(df),
                                                 replace=True,
                                                 p=intpop.probability.values)]
        pop_sample = pop_sample.reset_index()

        # select a random number of months into age_range
        months: pd.Series = pop_sample.month_range * self.rng.random_sample(size=len(df))
        pop_sample['months'] = pd.Series(pd.to_timedelta(months.astype(int), unit='M', box=False))

        # The entire initial population is alive!
        df.is_alive.values[:] = True

        years_ago = pd.to_timedelta(pop_sample['age_from'], unit='Y') + pop_sample['months']
        df.loc[df.is_alive, 'date_of_birth'] = self.sim.date - years_ago
        df.loc[df.is_alive, 'sex'] = pop_sample['gender'].map({'female': 'F', 'male': 'M'})

        # we can't use np.nan because that casts the series into a float
        df.loc[df.is_alive, 'mother_id'] = -1

        # Assign district and region of residence
        district_info = self.parameters['district_and_region_data']
        prob_in_district = district_info['District Total'] / district_info['District Total'].sum()

        assigned_district = district_info.loc[
            self.rng.choice(np.arange(0, len(prob_in_district)), size=df.is_alive.sum(), p=prob_in_district),
            ['District', 'Region']].copy().reset_index(drop=True)

        df.loc[df.is_alive, 'region_of_residence'] = assigned_district['Region']
        df.loc[df.is_alive, 'district_of_residence'] = assigned_district['District']

        # Check for no bad values being assigned to persons in the dataframe:
        assert (not pd.isnull(df['region_of_residence']).any())
        assert (not pd.isnull(df['district_of_residence']).any())

        age_in_days = self.sim.date - df.loc[df.is_alive, 'date_of_birth']
        df.loc[df.is_alive, 'age_exact_years'] = age_in_days / np.timedelta64(1, 'Y')
        df.loc[df.is_alive, 'age_years'] = df.loc[df.is_alive, 'age_exact_years'].astype(int)
        df.loc[df.is_alive, 'age_range'] = df.loc[df.is_alive, 'age_years'].map(self.AGE_RANGE_LOOKUP)
        df.loc[df.is_alive, 'age_days'] = age_in_days.dt.days

    def initialise_simulation(self, sim):
        """Get ready for simulation start.
        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """
        # Update age information every day
        sim.schedule_event(AgeUpdateEvent(self, self.AGE_RANGE_LOOKUP),
                           sim.date + DateOffset(days=1))

        # check all population to determine if person should die (from causes other than those
        # explicitly modelled) (repeats every month)
        sim.schedule_event(OtherDeathPoll(self), sim.date + DateOffset(months=1))

        # Launch the repeating event that will store statistics about the population structure
        sim.schedule_event(DemographyLoggingEvent(self), sim.date + DateOffset(days=0))

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
        df.at[child_id, 'sex'] = self.rng.choice(['M', 'F'], p=[f_male, 1 - f_male])

        df.at[child_id, 'mother_id'] = mother_id

        df.at[child_id, 'age_exact_years'] = 0.0
        df.at[child_id, 'age_years'] = 0
        df.at[child_id, 'age_range'] = self.AGE_RANGE_LOOKUP[0]

        # Child's residence is inherited from the mother
        df.at[child_id, 'region_of_residence'] = df.at[mother_id, 'region_of_residence']
        df.at[child_id, 'district_of_residence'] = df.at[mother_id, 'district_of_residence']

        # Log the birth:
        logger.info('%s|on_birth|%s',
                    self.sim.date,
                    {
                        'mother': mother_id,
                        'child': child_id,
                        'mother_age': df.at[mother_id, 'age_years']
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
        df.loc[df.is_alive, 'age_range'] = df.loc[df.is_alive, 'age_years'].map(self.age_range_lookup)
        df.loc[df.is_alive, 'age_days'] = age_in_days.dt.days


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
        df = population.props
        over_max_age = df.index[df.is_alive & (df.age_years > MAX_AGE)]
        for individual_id in over_max_age:
            self.sim.schedule_event(InstantaneousDeath(self.module, individual_id, 'OverMaxAge'))

        # get the mortality schedule for now...

        # load the mortality schedule (imported datasheet from excel workbook)
        mort_sched = self.module.parameters['mortality_schedule']

        # get the subset of mortality rates for this year.
        # confirms that we go to the five year period that we are in, not the exact year.
        fallbackyear = int(math.floor(self.sim.date.year / 5) * 5)

        mort_sched = mort_sched.loc[mort_sched.year == fallbackyear, ['age_years', 'sex', 'value']].copy()

        # get the population
        alive = df.loc[df.is_alive & (df.age_years <= MAX_AGE), ['sex', 'age_years']].copy()

        # merge the popualtion dataframe with the parameter dataframe to pick-up the risk of
        # mortality for each person in the model
        length_before_merge = len(alive)
        alive = alive.reset_index().merge(mort_sched,
                                          left_on=['age_years', 'sex'],
                                          right_on=['age_years', 'sex'],
                                          how='inner').set_index('person')
        assert length_before_merge == len(alive)

        # flipping the coin to determine if this person will die
        will_die = (self.module.rng.random_sample(size=len(alive)) < alive.value / 12)

        logger.debug('Will die count: %d', will_die.sum())

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
            # here comes the death.......
            df.at[individual_id, 'is_alive'] = False
            # the person is now dead
            df.at[individual_id, 'date_of_death'] = self.sim.date

        logger.debug("*******************************************The person %s "
                     "is now officially dead and has died of %s", individual_id, self.cause)

        # Log the death
        logger.info('%s|death|%s', self.sim.date,
                    {
                        'age': df.at[individual_id, 'age_years'],
                        'cause': self.cause,
                        'person_id': individual_id
                    })

        # Report the deaths to the healthburden module (if present) so that it tracks the live years lost
        if 'HealthBurden' in self.sim.modules.keys():
            date_of_birth = df.at[individual_id, 'date_of_birth']
            sex = df.at[individual_id, 'sex']
            label = self.module.name + '_' + self.cause  # creates a label for these YLL of <ModuleName>_<CauseOfDeath>
            self.sim.modules['HealthBurden'].report_live_years_lost(sex=sex,
                                                                    date_of_birth=date_of_birth,
                                                                    label=label)


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

        logger.info('%s|population|%s',
                    self.sim.date,
                    {
                        'total': sum(sex_count),
                        'male': sex_count['M'],
                        'female': sex_count['F']
                    })

        # if you groupby both sex and age_range, you weirdly lose categories where size==0, so
        # get the counts separately
        m_age_counts = df[df.is_alive & (df.sex == 'M')].groupby('age_range').size()
        f_age_counts = df[df.is_alive & (df.sex == 'F')].groupby('age_range').size()

        logger.info('%s|age_range_m|%s', self.sim.date,
                    m_age_counts.to_dict())

        logger.info('%s|age_range_f|%s', self.sim.date,
                    f_age_counts.to_dict())
