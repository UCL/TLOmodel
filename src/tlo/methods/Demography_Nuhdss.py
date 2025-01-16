"""
The core demography module and its associated events.
* Sets initial population size
* Determines the  age-related, and residential location properties.
"""




import math
from collections import defaultdict
from pathlib import Path
from types import MappingProxyType
from typing import Union

import numpy as np
import pandas as pd


from tlo import (
    DAYS_IN_MONTH,
    DAYS_IN_YEAR,
    Date,
    DateOffset,
    Module,
    Parameter,
    Property,
    Types,
    logging,
)
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.logging.helpers import get_dataframe_row_as_dict_for_logging
from tlo.util import DEFAULT_MOTHER_ID, create_age_range_lookup, get_person_id_to_inherit_from

# Standard logger
logger = logging.getLogger("tlo.methods.Demography_Nuhdss")
#logger = logging.getLogger(__name__)
#print(f"Logger name being used: {__name__}.detail")  # Debugging the logger name
logger.setLevel(logging.INFO)

# Detailed logger
logger_detail = logging.getLogger("tlo.methods.Demography_Nuhdss.detail")
logger_detail.setLevel(logging.INFO)

# Population scale factor logger
logger_scale_factor = logging.getLogger('tlo.methods.population')
logger_scale_factor.setLevel(logging.INFO)

# Limits for setting up age range categories
MIN_AGE_FOR_RANGE = 0
MAX_AGE_FOR_RANGE = 100
AGE_RANGE_SIZE = 5
MAX_AGE = 120


# Fnc to swap the contents of row1 and row2 in dataframe df, leaving all other rows unaffected.
def swap_rows(df, row1, row2):
    df.iloc[row1], df.iloc[row2] = df.iloc[row2].copy(), df.iloc[row1].copy()
    return df


def age_at_date(
    date: Union[Date, pd.DatetimeIndex, pd.Series],
    date_of_birth: Union[Date, pd.DatetimeIndex, pd.Series]
) -> float:
    """Compute exact age in years given a date of birth `dob` and date `date`."""
    # Assume a fixed number of days in all years, ignoring variations due to leap years
    return (date - date_of_birth) / pd.Timedelta(days=DAYS_IN_YEAR)

class Demography(Module):
    """
    The core demography module.
    """

    def __init__(self, name=None, resourcefilepath=None, equal_allocation_by_slum: bool = False):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.equal_allocation_by_slum = equal_allocation_by_slum
        self.initial_model_to_data_popsize_ratio = None  # will store scaling factor
        self.popsize_by_year = dict()  # will store total population size each year
        self.slums = None     # will store all the slums in a list
        

    OPTIONAL_INIT_DEPENDENCIES = {'ImprovedHealthSystemAndCareSeekingScenarioSwitcher'}
     #<-- this forces that module to be the first registered module, if it's registered.

    AGE_RANGE_CATEGORIES, AGE_RANGE_LOOKUP = create_age_range_lookup(
        min_age=MIN_AGE_FOR_RANGE,
        max_age=MAX_AGE_FOR_RANGE,
        range_size=AGE_RANGE_SIZE)

    # Convert AGE_RANGE_LOOKUP to read-only mapping to avoid accidental updates
    AGE_RANGE_LOOKUP = MappingProxyType(dict(AGE_RANGE_LOOKUP))

    # We should have 21 age range categories
    assert len(AGE_RANGE_CATEGORIES) == 21

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    PARAMETERS = {
        'max_age_initial': Parameter(Types.INT, 'The oldest age (in whole years) in the initial population'),
        'pop_2015': Parameter(Types.DATA_FRAME, 'Population in 2015 for initialising population'),
        'slum_num_to_slum_name': Parameter(Types.DICT, 'Mapping from slum_num to slum name'),
        'fraction_of_births_male': Parameter(Types.REAL, 'Birth Sex Ratio'),
        
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        #'is_alive': Property(Types.BOOL, 'Whether this individual is alive'),
        'date_of_birth': Property(Types.DATE, 'Date of birth of this individual'),
        'sex': Property(Types.CATEGORICAL, 'Male or female', categories=['M', 'F']),
        'mother_id': Property(Types.INT, 'Unique identifier of mother of this individual'),

        'slum_num_of_residence': Property(
            Types.CATEGORICAL, 
            'The slum number in which the person is resident',
            categories=['SET_AT_RUNTIME']
        ),

        'slum_of_residence': Property(
            Types.CATEGORICAL,
            'The slum (name) of residence (mapped from slum_num_of_residence).',
            categories=['SET_AT_RUNTIME']
        ),

        # Age calculation is handled by demography module
        'age_exact_years': Property(Types.REAL, 'The age of the individual in exact years'),
        'age_years': Property(Types.INT, 'The age of the individual in years'),
        'age_range': Property(Types.CATEGORICAL,
                              'The age range category of the individual',
                              categories=AGE_RANGE_CATEGORIES),
        'age_days': Property(Types.INT, 'The age of the individual in whole days'),
    }

    def read_parameters(self, data_folder):
        """Load the parameters from `ResourceFile_Demography_parameters.csv` and data from other `ResourceFiles`."""

        # General parameters
        self.load_parameters_from_dataframe(pd.read_csv(
            Path(self.resourcefilepath) / 'demography' / 'ResourceFile_Demography_parameters.csv')
        )
         # Initial population size:
        self.parameters['pop_2015'] = pd.read_csv(
            Path(self.resourcefilepath) / 'demography' / 'pop_2015.csv'
        )
        # # Initial population size:
        # self.parameters['pop_2015'] = pd.read_csv(
        #     Path(self.resourcefilepath) / 'demography' / 'ResourceFile_Population_nuhdss_2015.csv'
        # )

        # Lookup dicts to map from slum_num_of_residence (in the df) and slum name 
        self.slums = self.parameters['pop_2015']['slum'].drop_duplicates().to_list()
        self.parameters['slum_num_to_slum_name'] = \
            self.parameters['pop_2015'][['slum_Num', 'slum']].drop_duplicates()\
                                                                     .set_index('slum_Num')['slum']\
                                                                     .to_dict()
                                                                     

    
        # Fraction of babies that are male
        self.parameters['fraction_of_births_male'] = pd.read_csv(
            Path(self.resourcefilepath) / 'demography' / 'ResourceFile_Pop_Frac_Births_Male.csv'
        ).set_index('Year')['frac_births_male']

    def pre_initialise_population(self):
        """
        Define categorical properties for 'region_of_residence' and 'district_of_residence'
        """
        # Define categorical properties for 'slum_of_residence'
        # Nb. This couldn't be done before categories for each of these has been determined following read-in of data
        # and initialising of other modules.
       
        self.PROPERTIES['slum_num_of_residence'] = Property(
            Types.CATEGORICAL,
            'The slum (name) of residence (mapped from slum_num_of_residence).',
            categories=self.parameters['pop_2015']['slum_Num'].unique().tolist()
        )
        self.PROPERTIES['slum_of_residence'] = Property(
            Types.CATEGORICAL,
            'The slum (name) of residence (mapped from slum_num_of_residence).',
            categories=self.parameters['pop_2015']['slum'].unique().tolist()
        )
    
    def initialise_population(self, population):
        """Set properties for this module and compute the initial population scaling factor"""
        df = population.props
        

        # Compute the initial population scaling factor
        self.initial_model_to_data_popsize_ratio = \
            self.compute_initial_model_to_data_popsize_ratio(population.initial_size)

        init_pop = self.parameters['pop_2015']

        init_pop['prob'] = init_pop['Count'] / init_pop['Count'].sum()

        init_pop = self._edit_init_pop_to_prevent_persons_greater_than_max_age(
            init_pop,
            max_age=self.parameters['max_age_initial']
        )
        if self.equal_allocation_by_slum:
            init_pop = self._edit_init_pop_so_that_equal_number_in_each_slum(init_pop)

        # randomly pick from the init_pop sheet, to allocate characteristic to each person in the df
        demog_char_to_assign = init_pop.iloc[self.rng.choice(init_pop.index.values,
                                                             size=len(df),
                                                             replace=True,
                                                             p=init_pop.prob)][
            ['slum', 'slum_Num', 'Sex', 'Age']] \
            .reset_index(drop=True)

        # make a date of birth that is consistent with the allocated age of each person
        demog_char_to_assign['days_since_last_birthday'] = self.rng.randint(0, 365, len(demog_char_to_assign))

        demog_char_to_assign['date_of_birth'] = [
            self.sim.date - DateOffset(years=int(demog_char_to_assign['Age'][i]),
                                       days=int(demog_char_to_assign['days_since_last_birthday'][i]))
            for i in demog_char_to_assign.index]
        
       
        # Assign the characteristics
        #df.is_alive,
        #df.is_alive.values[:] = True
        df['date_of_birth'] = demog_char_to_assign['date_of_birth']
        df['sex'] = demog_char_to_assign['Sex']
        df[ 'mother_id'] = DEFAULT_MOTHER_ID  # Motherless, and their characterists are not inherited
        df['slum_num_of_residence'] = demog_char_to_assign['slum_Num'].values[:]
        df[ 'slum_of_residence'] = demog_char_to_assign['slum'].values[:]
        df[ 'age_exact_years'] = age_at_date(self.sim.date, demog_char_to_assign['date_of_birth'])
        df['age_years'] = df['age_exact_years'].astype('int64')
        df['age_range'] = df['age_years'].map(self.AGE_RANGE_LOOKUP)
        df[ 'age_days'] = (
            self.sim.date - demog_char_to_assign['date_of_birth']
        ).dt.days

        # Ensure first individual in df is a man, to safely exclude person_id=0 from selection of direct birth mothers.
        # If no men are found in df, issue a warning and proceed with female individual at person_id = 0.
        if df.loc[0].sex == 'F':
            diff_id = (df.sex.values != 'F').argmax()
            if diff_id != 0:
                swap_rows(df, 0, diff_id)
            else:
                logger.warning(key="warning",
                               data="No men found. Direct birth mothers search will exclude woman at person_id=0.")

    def initialise_simulation(self, sim):
        """
        * Schedule the AgeUpdateEvent, the OtherDeathPoll and the DemographyLoggingEvent
        * Output to the log the initial population scaling factor.
        """
        # Update age information every day (first time after one day)
        sim.schedule_event(AgeUpdateEvent(self, self.AGE_RANGE_LOOKUP), sim.date + DateOffset(days=1))

        # Launch the repeating event that will store statistics about the population structure
        sim.schedule_event(DemographyLoggingEvent(self), sim.date)

        # Log the initial population scaling-factor (to the logger of this module and that of `tlo.methods.population`)
        for _logger in (logger, logger_scale_factor):
            _logger.warning(
                key='scaling_factor',
                data={'scaling_factor': 1.0 / self.initial_model_to_data_popsize_ratio},
                description='The data-to-model scaling factor (based on the initial population size, used to '
                            'multiply-up results so that they correspond to the real population size.'
            )
       
      
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

        # Determine characteristics that are inherited from mother (and if no mother,
        # from a randomly selected person)
        _id_inherit_from = get_person_id_to_inherit_from(child_id, mother_id, df, rng)
        _slum_num_of_residence = df.at[_id_inherit_from, 'slum_num_of_residence']
        _slum_of_residence = df.at[_id_inherit_from, 'slum_of_residence']
    
        child = {
            #'is_alive': True,
            'date_of_birth': self.sim.date,
            'sex': 'M' if rng.random_sample() < fraction_of_births_male else 'F',
            'mother_id': mother_id,
            'slum_num_of_residence': _slum_num_of_residence,
            'slum_of_residence': _slum_of_residence,
            'age_exact_years': 0.0,
            'age_years': 0,
            'age_range': self.AGE_RANGE_LOOKUP[0]
        }
        df.loc[child_id, child.keys()] = child.values()

        # Log the birth:
        _mother_age_at_birth = df.at[abs(mother_id), 'age_years']  # Log age of mother whether true or direct birth
        _mother_age_at_pregnancy = int(
            age_at_date(
                df.at[mother_id, 'date_of_last_pregnancy'],
                df.at[mother_id, 'date_of_birth']
            )
        ) if mother_id >= 0 else -1  # No pregnancy for direct birth

        logger.info(
            key='on_birth',
            data={'mother': mother_id,  # Keep track of whether true or direct birth by using mother_id
                  'child': child_id,
                  'mother_age': _mother_age_at_birth,
                  'mother_age_at_pregnancy': _mother_age_at_pregnancy}
        )

    def _edit_init_pop_to_prevent_persons_greater_than_max_age(self, df, max_age: int):
        """Return an edited version of the `pd.DataFrame` describing the probability of persons in the population being
        created with certain characteristics to reflect the constraint the persons aged greater than `max_age_initial`
        should not be created."""

        if (max_age == 0) or (max_age > MAX_AGE):
            raise ValueError("The value of parameter `max_age_initial` is not valid.")

        _df = df.drop(df.index[df.Age > max_age])  # Remove characteristics with age greater than max_age
        _df.prob = _df.prob / _df.prob.sum()  # Rescale `prob` so that it sums to 1.0
        return _df.reset_index(drop=True)

    @staticmethod
    def _edit_init_pop_so_that_equal_number_in_each_slum(df) -> pd.DataFrame:
        """Return an edited version of the `pd.DataFrame` describing the probability of persons in the population being
        created with certain characteristics to reflect the constraint of there being an equal number of persons
        in each district."""

        # Get breakdown of Sex/Age within each district
        slum_nums = df['slum_Num'].unique()

        # Target size of each district
        target_size_for_slum = df['Count'].sum() / len(slum_nums)

        # Make new version (a copy) of the dataframe
        df_new = df.copy()

        for slum_num in slum_nums:
            mask_for_slum = df['slum_Num'] == slum_num
            # For each district, compute the age/sex breakdown, and use this with target_size to create updated `Count`
            # values
            df_new.loc[mask_for_slum, 'Count'] = target_size_for_slum * (
                df.loc[mask_for_slum, 'Count'] / df.loc[mask_for_slum, 'Count'].sum()
            )

        # Recompute "prob" column (i.e. the probability of being in that category)
        df_new["prob"] = df_new['Count'] / df_new['Count'].sum()

        # Check that the resulting dataframe is of the same size/shape as the original; that Count and prob make
        # sense; and that we have preserved the age/sex breakdown within each district
        def all_elements_identical(x):
            return np.allclose(x, x[0])

        assert df['Count'].sum() == df_new['Count'].sum()
        assert 1.0 == df['prob'].sum() == df_new['prob'].sum()
        assert all_elements_identical(df_new.groupby('slum_Num')['prob'].sum().values)

        def get_age_sex_breakdown_in_slum(dat, slum_num):
            return (
                dat.loc[df['slum_Num'] == slum_num].groupby(['Age', 'Sex'])['prob'].sum()
                / dat.loc[df['slum_Num'] == slum_num, 'prob'].sum()
            )

        for _d in slum_nums:
            pd.testing.assert_series_equal(
                get_age_sex_breakdown_in_slum(df, _d),
                get_age_sex_breakdown_in_slum(df_new, _d)
            )

        # Return the new dataframe
        return df_new


    def compute_initial_model_to_data_popsize_ratio(self, initial_population_size):
        """Compute ratio of initial model population size to estimated population size in 2010.

        Uses the total of the per-region estimated populations in 2010 used to
        initialise the simulation population as the baseline figure, with this value
        corresponding to the 2010 projected population from [wpp2019]_.

        .. [wpp2019] World Population Prospects 2019. United Nations Department of
        Economic and Social Affairs. URL:
        https://population.un.org/wpp/Download/Standard/Population/

        :param initial_population_size: Initial population size to calculate ratio for.

        :returns: Ratio of ``initial_population`` to 2010 baseline population.
        """
        return initial_population_size / self.parameters['pop_2015']['Count'].sum()


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
        dates_of_birth = df['date_of_birth']
        df['age_exact_years'] = age_at_date(
            self.module.sim.date, dates_of_birth
        )
        df['age_years'] = df['age_exact_years'].astype('int64')
        df['age_range'] = df['age_years'].map(self.age_range_lookup)
        df['age_days'] = (self.module.sim.date - dates_of_birth).dt.days



class DemographyLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """
        Logging event running every year.
        * Update internal storage of population size
        * Output statistics to the log
        """
        # run this event every 12 months (every year)
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        df = population.props

        # 1) Update internal storage of the total population size each year.
        self.module.popsize_by_year[self.sim.date.year] = len(df)
        # 2) Compute Statistics for the log
        sex_count = df.groupby('sex').size()

        logger.info(
            key='population',
            data={'total': sum(sex_count),
                  'male': sex_count['M'],
                  'female': sex_count['F']
                  })

        # (nb. if you groupby both sex and age_range, you weirdly lose categories where size==0, so
        # get the counts separately.)
        #m_age_counts = df[df.is_alive & (df.sex == 'M')].groupby('age_range').size()
        #f_age_counts = df[df.is_alive & (df.sex == 'F')].groupby('age_range').size()

        # Get the counts separately
        m_age_counts = df[df.sex == 'M'].groupby('age_range').size()
        f_age_counts = df[df.sex == 'F'].groupby('age_range').size()

        logger.info(key='age_range_m', data=m_age_counts.to_dict())

        logger.info(key='age_range_f', data=f_age_counts.to_dict())

        # Output by single year of age for under-fives
        # (need to guarantee output always is for each of the years - even if size() is 0)
        num_children = pd.Series(index=range(5), data=0).add(
            df[df.age_years < 5].groupby('age_years').size(),
            fill_value=0
        ).astype(int)

        logger.info(key='num_children', data=num_children.to_dict())

