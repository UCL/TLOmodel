"""
The core demography module and its associated events.
Expects input in format of the 'Demography.xlsx'  of TimH, sent 3/10. Uses the 'Interpolated
population structure' worksheet within to initialise the age & sex distribution of population.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.util import create_age_range_lookup

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Limits for setting up age range categories
MIN_AGE_FOR_RANGE = 0
MAX_AGE_FOR_RANGE = 100
AGE_RANGE_SIZE = 5
MAX_AGE = 120


class Demography(Module):
    """
    The core demography modules handling age and sex of individuals. Also is responsible for their
    'is_alive' status
    """

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    AGE_RANGE_CATEGORIES, AGE_RANGE_LOOKUP = create_age_range_lookup(
        min_age=MIN_AGE_FOR_RANGE,
        max_age=MAX_AGE_FOR_RANGE,
        range_size=AGE_RANGE_SIZE)

    # We should have 21 age range categories
    assert len(AGE_RANGE_CATEGORIES) == 21

    # Declare Metadata
    METADATA = {}

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    PARAMETERS = {
        'pop_2010': Parameter(Types.DATA_FRAME, 'Population in 2010 for initialising population'),
        'mortality_schedule': Parameter(Types.DATA_FRAME, 'Age-spec mortality rates from WPP'),
        'fraction_of_births_male': Parameter(Types.REAL, 'Birth Sex Ratio'),
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'is_alive': Property(Types.BOOL, 'Whether this individual is alive'),
        'date_of_birth': Property(Types.DATE, 'Date of birth of this individual'),
        'date_of_death': Property(Types.DATE, 'Date of death of this individual'),
        'cause_of_death': Property(Types.STRING, 'Cause of death of this individual, as entered to the logger'),
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
        'district_num_of_residence': Property(Types.INT, 'The district number in which the person is resident')
    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.
        Loads the 'Interpolated Pop Structure' worksheet from the Demography Excel workbook.
        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """

        # Initial population size:
        self.parameters['pop_2010'] = pd.read_csv(
            Path(self.resourcefilepath) / 'ResourceFile_Population_2010.csv'
        )

        # Fraction of babies that are male
        self.parameters['fraction_of_births_male'] = pd.read_csv(
            Path(self.resourcefilepath) / 'ResourceFile_Pop_Frac_Births_Male.csv'
        ).set_index('Year')['frac_births_male']

        # Mortality schedule:
        self.parameters['mortality_schedule'] = pd.read_csv(
            Path(self.resourcefilepath) / 'ResourceFile_Pop_DeathRates_Expanded_WPP.csv'
        )

    def initialise_population(self, population):
        """Set our property values for the initial population.
        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.
        :param population: the population of individuals
        """
        df = population.props

        init_pop = self.parameters['pop_2010']
        init_pop['prob'] = init_pop['Count'] / init_pop['Count'].sum()

        # randomly pick from the init_pop sheet, to allocate charatceristic to each person in the df
        demog_char_to_assign = init_pop.iloc[self.rng.choice(init_pop.index.values,
                                                             size=len(df),
                                                             replace=True,
                                                             p=init_pop.prob)][
            ['District', 'District_Num', 'Region', 'Sex', 'Age']] \
            .reset_index(drop=True)

        # make a date of birth that is consistent with the allocated age of each person
        demog_char_to_assign['days_since_last_birthday'] = self.rng.randint(0, 365, len(demog_char_to_assign))

        demog_char_to_assign['date_of_birth'] = [
            self.sim.date - DateOffset(years=int(demog_char_to_assign['Age'][i]),
                                       days=int(demog_char_to_assign['days_since_last_birthday'][i]))
            for i in demog_char_to_assign.index]
        demog_char_to_assign['age_in_days'] = self.sim.date - demog_char_to_assign['date_of_birth']

        # Assign the characteristics
        df.is_alive.values[:] = True
        df['date_of_birth'] = demog_char_to_assign['date_of_birth']
        df['date_of_death'] = pd.NaT
        df['cause_of_death'] = ''
        df['sex'].values[:] = demog_char_to_assign['Sex']
        df.loc[df.is_alive, 'mother_id'] = -1
        df.loc[df.is_alive, 'age_exact_years'] = demog_char_to_assign['age_in_days'] / np.timedelta64(1, 'Y')
        df.loc[df.is_alive, 'age_years'] = df.loc[df.is_alive, 'age_exact_years'].astype('int64')
        df.loc[df.is_alive, 'age_range'] = df.loc[df.is_alive, 'age_years'].map(self.AGE_RANGE_LOOKUP)
        df.loc[df.is_alive, 'age_days'] = demog_char_to_assign['age_in_days'].dt.days
        df['region_of_residence'] = demog_char_to_assign['Region']
        df['district_of_residence'] = demog_char_to_assign['District']
        df['district_num_of_residence'] = demog_char_to_assign['District_Num']

        # Check for no bad values being assigned to persons in the dataframe:
        assert (not pd.isnull(df['region_of_residence']).any())
        assert (not pd.isnull(df['district_of_residence']).any())

        # Update other age properties

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

        # Check that the simulation does not run too long
        if self.sim.end_date.year >= 2100:
            raise Exception('Year is after 2100: Demographic data do not extend that far.')

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.
        This is called by the simulation whenever a new person is born.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        df = self.sim.population.props
        rng = self.rng

        fraction_of_births_male = self.parameters['fraction_of_births_male'][self.sim.date.year]

        child = {
            'is_alive': True,
            'date_of_birth': self.sim.date,
            'date_of_death': pd.NaT,
            'cause_of_death': '',
            'sex': 'M' if rng.random_sample() < fraction_of_births_male else 'F',
            'mother_id': mother_id,
            'age_exact_years': 0.0,
            'age_years': 0,
            'age_range': self.AGE_RANGE_LOOKUP[0],
            'region_of_residence': df.at[mother_id, 'region_of_residence'],
            'district_of_residence': df.at[mother_id, 'district_of_residence']
        }

        df.loc[child_id, child.keys()] = child.values()

        # Log the birth:
        logger.info(
            key='on_birth',
            data={'mother': mother_id,
                  'child': child_id,
                  'mother_age': df.at[mother_id, 'age_years']}
        )

    def do_death(self, individual_id, cause, originating_module):
        """Register and log the death of an individual from a specific cause"""

        df = self.sim.population.props
        person = df.loc[individual_id]

        if not person['is_alive']:
            return

        # Register the death:
        df.loc[individual_id, ['is_alive', 'date_of_death', 'cause_of_death']] = (False, self.sim.date, cause)

        # Log the death:
        if 'Contraception' or 'SimplifiedBirths' in self.sim.modules.keys():
            logger.info(
                key='death',
                data={'age': person['age_years'],
                      'sex': person['sex'],
                      'cause': cause,
                      'pregnancy': person['is_pregnant'],
                      'person_id': individual_id
                      })
        else:
            logger.info(
                key='death',
                data={'age': person['age_years'],
                      'sex': person['sex'],
                      'cause': cause,
                      'person_id': individual_id
                      })

        # Release any beds-days that would be used by this person:
        if 'HealthSystem' in self.sim.modules:
            self.sim.modules['HealthSystem'].remove_beddays_footprint(person_id=individual_id)

        # Report the deaths to the healthburden module (if present) so that it tracks the live years lost
        if 'HealthBurden' in self.sim.modules.keys():
            label = originating_module.name + '_' + cause
            # creates a label for these YLL of the form: <ModuleName>_<CauseOfDeath>
            self.sim.modules['HealthBurden'].report_live_years_lost(sex=person['sex'],
                                                                    date_of_birth=person['date_of_birth'],
                                                                    label=label)

    def calc_py_lived_in_last_year(self, delta=pd.DateOffset(years=1), mask=None):
        """
        This is a helper method to compute the person-years that were lived in the previous year by age.
        It outputs a pd.DataFrame with the index being single year of age, 0 to 99.
        """
        df = self.sim.population.props

        # if a mask is passed to the function, restricts the PY calculation to individuals who don't have the condition
        # specified by the mask
        if mask is not None:
            df = df[mask]

        # get everyone who was alive during the previous year
        one_year_ago = self.sim.date - delta
        condition = df.is_alive | (df.date_of_death > one_year_ago)
        df_py = df.loc[condition, ['sex', 'age_exact_years', 'age_years', 'date_of_birth']]

        # renaming columns for clarity
        df_py = df_py.rename({'age_exact_years': 'age_exact_end', 'age_years': 'age_years_end'}, axis=1)

        # exact age at the start
        df_py['age_exact_start'] = (one_year_ago - df_py.date_of_birth) / np.timedelta64(1, 'Y')
        df_py['age_years_start'] = np.floor(df_py.age_exact_start).astype(np.int64)  # int age at start of the period
        df_py['years_in_age_start'] = df_py.age_years_end - df_py.age_exact_start  # time spent in age at start
        df_py['years_in_age_end'] = df_py.age_exact_end - df_py.age_years_end  # time spent in age at end

        # correction for those individuals who started the year and then died at the same age
        condition = df_py.age_years_end == df_py.age_years_start
        df_py.loc[condition, 'years_in_age_start'] = df_py.age_exact_end - df_py.age_exact_start
        df_py.loc[condition, 'years_in_age_end'] = 0

        # zero out entries for those born in the year passed (no time spend in age at start of year)
        condition = df_py.age_exact_start < 0
        df_py.loc[condition, 'years_in_age_start'] = 0
        df_py.loc[condition, 'age_years_start'] = 0
        df_py.loc[condition, 'age_exact_start'] = 0

        # collected all time spent in age at start of period
        df1 = df_py[['sex', 'years_in_age_start', 'age_years_start']].groupby(by=['sex', 'age_years_start']).sum()
        df1 = df1.unstack('sex')
        df1.columns = df1.columns.droplevel(0)
        df1.index.rename('age_years', inplace=True)

        # collect all time spent in age at end of period
        df2 = df_py[['sex', 'years_in_age_end', 'age_years_end']].groupby(by=['sex', 'age_years_end']).sum()
        df2 = df2.unstack('sex')
        df2.columns = df2.columns.droplevel(0)
        df2.index.rename('age_years', inplace=True)

        # add the two time spents together
        py = pd.DataFrame(
            index=pd.Index(data=list(self.AGE_RANGE_LOOKUP.keys()), name='age_years'),
            columns=['M', 'F'],
            data=0.0
        )
        py = py.add(df1, fill_value=0).add(df2, fill_value=0)

        return py


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
        df.loc[df.is_alive, 'age_years'] = df.loc[df.is_alive, 'age_exact_years'].astype('int64')
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
        logger.debug(key='death_poll_start', data='Checking to see if anyone should die...')

        # Get shortcut to main dataframe
        df = population.props

        # Cause the death for anyone that is older than the maximum age
        over_max_age = df.index[df.is_alive & (df.age_years > MAX_AGE)]
        for individual_id in over_max_age:
            self.sim.schedule_event(InstantaneousDeath(self.module, individual_id, 'OverMaxAge'))

        # Get the mortality schedule for now...
        # load the mortality schedule from WPP
        mort_sched = self.module.parameters['mortality_schedule']

        # get the subset of mortality rates for this year.
        # confirms that we go to the five year period that we are in, not the exact year.
        fallbackyear = int(math.floor(self.sim.date.year / 5) * 5)

        mort_sched = mort_sched.loc[
            mort_sched.fallbackyear == fallbackyear, ['age_years', 'sex', 'death_rate']].copy()

        # get the population
        alive = df.loc[df.is_alive & (df.age_years <= MAX_AGE), ['sex', 'age_years']].copy()

        # merge the popualtion dataframe with the parameter dataframe to pick-up the death_rate for each person
        length_before_merge = len(alive)
        alive = alive.reset_index().merge(mort_sched,
                                          left_on=['age_years', 'sex'],
                                          right_on=['age_years', 'sex'],
                                          how='inner').set_index('person')
        assert length_before_merge == len(alive)

        # Work out probability of dying in the time before the next occurrence of this poll
        dur_in_years_between_polls = np.timedelta64(self.frequency.months, 'M') / np.timedelta64(1, 'Y')
        prob_of_dying_during_next_month = 1.0 - np.exp(-alive.death_rate * dur_in_years_between_polls)

        # flipping the coin to determine if this person will die
        will_die = (self.module.rng.random_sample(size=len(alive)) < prob_of_dying_during_next_month)
        logger.debug(key='death_count', data=f'Will die count: {will_die.sum()}')

        # loop through to see who is going to die:
        for person in alive.index[will_die]:
            # schedule the death for some point in the next month
            self.sim.schedule_event(InstantaneousDeath(self.module, person, cause='Other'),
                                    self.sim.date + DateOffset(days=self.module.rng.randint(0, 30)))


class InstantaneousDeath(Event, IndividualScopeEventMixin):
    """
    Call the do_death function to cause the person to die.
    The 'module' passed to this event is the disease module that is causing the death.
    """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)
        self.cause = cause

    def apply(self, individual_id):
        self.sim.modules['Demography'].do_death(individual_id, cause=self.cause, originating_module=self.module)


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

        logger.info(
            key='population',
            data={'total': sum(sex_count),
                  'male': sex_count['M'],
                  'female': sex_count['F']
                  })

        # if you groupby both sex and age_range, you weirdly lose categories where size==0, so
        # get the counts separately
        m_age_counts = df[df.is_alive & (df.sex == 'M')].groupby('age_range').size()
        f_age_counts = df[df.is_alive & (df.sex == 'F')].groupby('age_range').size()

        logger.info(key='age_range_m', data=m_age_counts.to_dict())

        logger.info(key='age_range_f', data=f_age_counts.to_dict())

        # Output by single year of age for under-fives
        # (need to gurantee output always is for each of the years - even if size() is 0)
        num_children = pd.Series(index=range(5), data=0).add(
            df[df.is_alive & (df.age_years < 5)].groupby('age_years').size(),
            fill_value=0
        )

        logger.info(key='num_children', data=num_children.to_dict())

        # Output the person-years lived by single year of age in the past year
        py = self.module.calc_py_lived_in_last_year()
        logger.info(key='person_years', data=py.to_dict())


def scale_to_population(parsed_output, resourcefilepath, rtn_scaling_ratio=False):
    """
    This helper function scales certain outputs so that they can create statistics for the whole population.
    e.g. Population Size, Number of deaths are scaled by the factor of {Model Pop Size at Start of Simulation} to {
    {Real Population at the same time}.

    NB. This file gives precedence to the Malawi Population Census

    :param parsed_outoput: The outputs from parse_output
    :param resourcefilepath: The resourcefilepath
    :return: a new version of parsed_output that includes certain variables scaled
    """

    # Get information about the real population size (Malawi Census in 2018)
    cens_tot = pd.read_csv(Path(resourcefilepath) / "ResourceFile_PopulationSize_2018Census.csv")['Count'].sum()
    cens_yr = 2018

    # Get information about the model population size in 2018 (and fail if no 2018)
    model_res = parsed_output['tlo.methods.demography']['population']
    model_res['year'] = pd.to_datetime(model_res.date).dt.year

    assert cens_yr in model_res.year.values, "Model results do not contain the year of the census, so cannot scale"
    model_tot = model_res.loc[model_res['year'] == cens_yr, 'total'].values[0]

    # Calculate ratio for scaling
    ratio_data_to_model = cens_tot / model_tot

    if rtn_scaling_ratio:
        return ratio_data_to_model

    # Do the scaling on selected columns in the parsed outputs:
    o = parsed_output.copy()

    # Multiply population count summaries by ratio
    o['tlo.methods.demography']['population']['male'] *= ratio_data_to_model
    o['tlo.methods.demography']['population']['female'] *= ratio_data_to_model
    o['tlo.methods.demography']['population']['total'] *= ratio_data_to_model

    o['tlo.methods.demography']['age_range_m'].iloc[:, 1:] *= ratio_data_to_model
    o['tlo.methods.demography']['age_range_f'].iloc[:, 1:] *= ratio_data_to_model

    # For individual-level reporting, construct groupby's and then multipy by ratio
    # 1) Counts of numbers of death by year/age/cause
    deaths = o['tlo.methods.demography']['death']
    deaths.index = pd.to_datetime(deaths['date'])
    deaths['year'] = deaths.index.year.astype(int)

    deaths_groupby_scaled = deaths[['year', 'sex', 'age', 'cause', 'person_id']].groupby(
        by=['year', 'sex', 'age', 'cause']).count().unstack(fill_value=0).stack() * ratio_data_to_model
    deaths_groupby_scaled.rename(columns={'person_id': 'count'}, inplace=True)
    o['tlo.methods.demography'].update({'death_groupby_scaled': deaths_groupby_scaled})

    # 2) Counts of numbers of births by year/age-of-mother
    births = o['tlo.methods.demography']['on_birth']
    births.index = pd.to_datetime(births['date'])
    births['year'] = births.index.year
    births_groupby_scaled = \
        births[['year', 'mother_age', 'mother']].groupby(by=['year', 'mother_age']).count() \
        * ratio_data_to_model
    births_groupby_scaled.rename(columns={'mother': 'count'}, inplace=True)
    o['tlo.methods.demography'].update({'birth_groupby_scaled': births_groupby_scaled})

    # TODO: Do this kind of manipulation for all things in the log that are /
    #  flagged are being subject to scaling - issue raised.
    return o
