"""
The core demography module and its associated events.
* Sets initial population size
* Determines the 'is_alive', age-related, and residential location properties.
* Runs the OtherDeathPoll which represents the deaths due to causes other than those represented by disease modules.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods.causes import (
    Cause,
    collect_causes_from_disease_modules,
    create_mappers_from_causes_to_label,
)
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
    The core demography module.
    """

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.popsize_by_year = dict()  # will store total population size each year
        self.causes_of_death = dict()  # will store all the causes of death that are possible in the simulation
        self.gbd_causes_of_death = set()  # will store all the causes of death defined in the GBD data
        self.gbd_causes_of_death_not_represented_in_disease_modules = set()
        #  will store causes of death in GBD not represented in the simulation
        self.other_death_poll = None    # will hold pointer to the OtherDeathPoll object

    AGE_RANGE_CATEGORIES, AGE_RANGE_LOOKUP = create_age_range_lookup(
        min_age=MIN_AGE_FOR_RANGE,
        max_age=MAX_AGE_FOR_RANGE,
        range_size=AGE_RANGE_SIZE)

    # We should have 21 age range categories
    assert len(AGE_RANGE_CATEGORIES) == 21

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    PARAMETERS = {
        'pop_2010': Parameter(Types.DATA_FRAME, 'Population in 2010 for initialising population'),
        'district_num_to_district_name': Parameter(Types.DICT, 'Mapping from district_num to district name'),
        'district_num_to_region_name': Parameter(Types.DICT, 'Mapping from district_num to region name'),
        'all_cause_mortality_schedule': Parameter(Types.DATA_FRAME, 'All-cause age-specific mortality rates from WPP'),
        'fraction_of_births_male': Parameter(Types.REAL, 'Birth Sex Ratio'),
        'gbd_causes_of_death_data': Parameter(Types.DATA_FRAME,
                                              'Proportion of deaths in each age/sex group attributable to each possible'
                                              ' cause of death in the GBD dataset.'),
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
        'district_num_of_residence': Property(Types.INT, 'The district number in which the person is resident'),

        # the categories of these properties are set in `pre_initialise_population`
        'cause_of_death': Property(
            Types.CATEGORICAL,
            'The cause of death of this individual (the tlo_cause defined by the module)',
            categories=['SET_AT_RUNTIME']
        ),

        'district_of_residence': Property(
            Types.CATEGORICAL,
            'The district (name) of residence (mapped from district_num_of_residence).',
            categories=['SET_AT_RUNTIME']
        ),

        'region_of_residence': Property(
            Types.CATEGORICAL,
            'The region of residence (mapped from district_num_of_residence).',
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
        """Read parameter values from file, if required.
        Loads the 'Interpolated Pop Structure' worksheet from the Demography Excel workbook.
        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """

        # Initial population size:
        self.parameters['pop_2010'] = pd.read_csv(
            Path(self.resourcefilepath) / 'demography' / 'ResourceFile_Population_2010.csv'
        )

        # Lookup dicts to map from district_num_of_residence (in the df) and District name and Region name
        self.parameters['district_num_to_district_name'] = \
            self.parameters['pop_2010'][['District_Num', 'District']].drop_duplicates()\
                                                                     .set_index('District_Num')['District']\
                                                                     .to_dict()

        self.parameters['district_num_to_region_name'] = \
            self.parameters['pop_2010'][['District_Num', 'Region']].drop_duplicates()\
                                                                   .set_index('District_Num')['Region']\
                                                                   .to_dict()

        # Fraction of babies that are male
        self.parameters['fraction_of_births_male'] = pd.read_csv(
            Path(self.resourcefilepath) / 'demography' / 'ResourceFile_Pop_Frac_Births_Male.csv'
        ).set_index('Year')['frac_births_male']

        # All-Cause Mortality schedule:
        self.parameters['all_cause_mortality_schedule'] = pd.read_csv(
            Path(self.resourcefilepath) / 'demography' / 'ResourceFile_Pop_DeathRates_Expanded_WPP.csv'
        )

        # GBD Dataset for Causes of Death
        self.parameters['gbd_causes_of_death_data'] = pd.read_csv(
            Path(self.resourcefilepath) / 'gbd' / 'ResourceFile_CausesOfDeath_GBD2019.csv'
        ).set_index(['Sex', 'Age_Grp'])

    def pre_initialise_population(self):
        """
        1) Store all the cause of death represented in the imported GBD data
        2) Process the declarations of causes of death made by the disease modules
        3) Define categorical properties for 'cause_of_death', 'region_of_residence' and 'district_of_residence'
        """

        # 1) Store all the cause of death represented in the imported GBD data
        self.gbd_causes_of_death = set(self.parameters['gbd_causes_of_death_data'].columns)

        # 2) Process the declarations of causes of death made by the disease modules
        self.process_causes_of_death()

        # 3) Define categorical properties for 'cause_of_death', 'region_of_residence' and 'district_of_residence'
        # Nb. This couldn't be done before categories for each of these has been determined following read-in of data
        # and initialising of other modules.
        self.PROPERTIES['cause_of_death'] = Property(
            Types.CATEGORICAL,
            'The cause of death of this individual (the tlo_cause defined by the module)',
            categories=list(self.causes_of_death.keys())
        )
        self.PROPERTIES['district_of_residence'] = Property(
            Types.CATEGORICAL,
            'The district (name) of residence (mapped from district_num_of_residence).',
            categories=self.parameters['pop_2010']['District'].unique().tolist()
        )
        self.PROPERTIES['region_of_residence'] = Property(
            Types.CATEGORICAL,
            'The region of residence (mapped from district_num_of_residence).',
            categories=self.parameters['pop_2010']['Region'].unique().tolist()
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

        # randomly pick from the init_pop sheet, to allocate characteristic to each person in the df
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
        df.loc[df.is_alive, 'date_of_birth'] = demog_char_to_assign['date_of_birth']
        df.loc[df.is_alive, 'date_of_death'] = pd.NaT
        df.loc[df.is_alive, 'cause_of_death'] = np.nan
        df.loc[df.is_alive, 'sex'] = demog_char_to_assign['Sex']
        df.loc[df.is_alive, 'mother_id'] = -1
        df.loc[df.is_alive, 'district_num_of_residence'] = demog_char_to_assign['District_Num'].values[:]
        df.loc[df.is_alive, 'district_of_residence'] = demog_char_to_assign['District'].values[:]
        df.loc[df.is_alive, 'region_of_residence'] = demog_char_to_assign['Region'].values[:]

        df.loc[df.is_alive, 'age_exact_years'] = demog_char_to_assign['age_in_days'] / np.timedelta64(1, 'Y')
        df.loc[df.is_alive, 'age_years'] = df.loc[df.is_alive, 'age_exact_years'].astype('int64')
        df.loc[df.is_alive, 'age_range'] = df.loc[df.is_alive, 'age_years'].map(self.AGE_RANGE_LOOKUP)
        df.loc[df.is_alive, 'age_days'] = demog_char_to_assign['age_in_days'].dt.days

    def initialise_simulation(self, sim):
        """
        * Schedule the age updating
        * Output to the log the dicts that can be used for mapping from causes of death defined here, and those defined
        in the GBD datasets, to a common 'label'.
        """
        # Update age information every day
        sim.schedule_event(AgeUpdateEvent(self, self.AGE_RANGE_LOOKUP), sim.date + DateOffset(days=1))

        # check all population to determine if person should die (from causes other than those
        # explicitly modelled) (repeats every month)
        self.other_death_poll = OtherDeathPoll(self)
        sim.schedule_event(self.other_death_poll, sim.date + DateOffset(months=1))

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
            'cause_of_death': np.nan,
            'sex': 'M' if rng.random_sample() < fraction_of_births_male else 'F',
            'mother_id': mother_id,
            'district_num_of_residence': df.at[mother_id, 'district_num_of_residence'],
            'district_of_residence': df.at[mother_id, 'district_of_residence'],
            'region_of_residence': df.at[mother_id, 'region_of_residence'],
            'age_exact_years': 0.0,
            'age_years': 0,
            'age_range': self.AGE_RANGE_LOOKUP[0]
        }

        df.loc[child_id, child.keys()] = child.values()

        # Log the birth:
        logger.info(
            key='on_birth',
            data={'mother': mother_id,
                  'child': child_id,
                  'mother_age': df.at[mother_id, 'age_years']}
        )

    def on_simulation_end(self):
        """Things to do at end of the simulation:
        * Compute and log the scaling-factor
        """
        sf = self.compute_scaling_factor()
        if not np.isnan(sf):
            logger.info(
                key='scaling_factor',
                data={'scaling_factor': sf},
                description='The scaling factor (if can be computed)'
            )

    def process_causes_of_death(self):
        """
        1) Register all causes of deaths defined by Module
        2) Define the "Other" causes of death (that is managed in this module by the OtherDeathPoll)
        3) Output to the log mappers for causes of death (tlo_cause --> label; gbd_cause --> label).
        """
        # 1) Register all the causes of death from the disease modules: gives dict(<tlo_cause>: <Cause instance>)
        self.causes_of_death = collect_causes_from_disease_modules(
            all_modules=self.sim.modules.values(),
            collect='CAUSES_OF_DEATH',
            acceptable_causes=self.gbd_causes_of_death
        )

        # 2) Define the "Other" tlo_cause of death (that is managed in this module by the OtherDeathPoll)
        self.gbd_causes_of_death_not_represented_in_disease_modules = \
            self.get_gbd_causes_of_death_not_represented_in_disease_modules(self.causes_of_death)
        self.causes_of_death['Other'] = Cause(
            label='Other',
            gbd_causes=self.gbd_causes_of_death_not_represented_in_disease_modules
        )

        # 3) Output to the log mappers for causes of death
        mapper_from_tlo_causes, mapper_from_gbd_causes = self.create_mappers_from_causes_of_death_to_label()
        logger.info(
            key='mapper_from_tlo_cause_to_common_label',
            data=mapper_from_tlo_causes
        )
        logger.info(
            key='mapper_from_gbd_cause_to_common_label',
            data=mapper_from_gbd_causes
        )

    def do_death(self, individual_id: int, cause: str, originating_module: Module):
        """Register and log the death of an individual from a specific cause.
        * 'individual_id' is the index in the population.props dataframe to the (one) person.
        * 'cause' is the "tlo cause" that is defined by the disease module.
        * 'originating_module' is the disease module that is causing the death.
        """

        assert not hasattr(individual_id, '__iter__'), 'do_death must be called for one individual at a time.'

        df = self.sim.population.props
        person = df.loc[individual_id]

        if not person['is_alive']:
            return

        # Check that the cause is declared, and declared for use by the originating module:
        assert cause in self.causes_of_death, f'The cause of death {cause} is not declared.'
        if originating_module is not self:
            assert cause in originating_module.CAUSES_OF_DEATH, \
                f'The cause of death {cause} is not declared for use by the module {originating_module.name}.'

        # Register the death:
        df.loc[individual_id, ['is_alive', 'date_of_death', 'cause_of_death']] = (False, self.sim.date, cause)

        # Log the death
        # - log the line-list of summary information about each death
        data_to_log_for_each_death = {
            'age': person['age_years'],
            'sex': person['sex'],
            'cause': cause,
            'label': self.causes_of_death[cause].label,
            'person_id': individual_id
        }

        if ('Contraception' in self.sim.modules) or ('SimplifiedBirths' in self.sim.modules):
            # If possible, append to the log additional information about pregnancy:
            data_to_log_for_each_death.update({
                'pregnancy': person['is_pregnant'],
            })

        logger.info(key='death', data=data_to_log_for_each_death)

        # - log all the properties for the deceased person
        logger.info(key='properties_of_deceased_persons',
                    data=person.to_dict(),
                    description='values of all properties at the time of death for deceased persons')

        # Report the deaths to the healthburden module (if present) so that it tracks the live years lost
        if 'HealthBurden' in self.sim.modules.keys():
            # report the death so that a computation of lost life-years due to this cause to be recorded
            self.sim.modules['HealthBurden'].report_live_years_lost(sex=person['sex'],
                                                                    date_of_birth=person['date_of_birth'],
                                                                    cause_of_death=cause)

        # Release any beds-days that would be used by this person:
        if 'HealthSystem' in self.sim.modules:
            self.sim.modules['HealthSystem'].remove_beddays_footprint(person_id=individual_id)

    def get_gbd_causes_of_death_not_represented_in_disease_modules(self, causes_of_death):
        """
        Find the causes of death in the GBD datasets that are not represented within the causes of death defined in the
        modules registered in this simulation.
        :return: set of gbd_causes of death that are not represented in disease modules
        """
        all_gbd_causes_in_sim = set()
        for c in causes_of_death.values():
            all_gbd_causes_in_sim.update(c.gbd_causes)

        return self.gbd_causes_of_death - all_gbd_causes_in_sim

    def create_mappers_from_causes_of_death_to_label(self):
        """Use a helper function to create mappers for causes of death to label."""
        return create_mappers_from_causes_to_label(
            causes=self.causes_of_death,
            all_gbd_causes=self.gbd_causes_of_death
        )

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

        # add the two time spent together
        py = pd.DataFrame(
            index=pd.Index(data=list(self.AGE_RANGE_LOOKUP.keys()), name='age_years'),
            columns=['M', 'F'],
            data=0.0
        )
        py = py.add(df1, fill_value=0).add(df2, fill_value=0)

        return py

    def compute_scaling_factor(self):
        """
        Compute the scaling factor, if it is possible to do so.

        The scaling factor is the ratio of {Real Population} to {Model Pop Size}. It is used to multiply model outputs
        in order to produce statistics that will be of the same scale as the real population.

        It is estimated by comparing the population size with the national census in the year that the census was
        conducted.

        If the simulation does not include that year, the scaling factor cannot be computed (in which case, np.nan is
        returned).

        :return: floating point number that is the scaling factor, or np.nan if it cannot be computed.
        """

        # Get Census data
        year_of_census = 2018
        census_popsize = pd.read_csv(
            Path(self.resourcefilepath) / "demography" / "ResourceFile_PopulationSize_2018Census.csv"
        )['Count'].sum()

        # Get model total population size in that same year
        if year_of_census not in self.popsize_by_year:
            return np.nan
        else:
            model_popsize_in_year_of_census = self.popsize_by_year[year_of_census]
            return census_popsize / model_popsize_in_year_of_census

    def compute_initial_population_scaling_factor(self, initial_population):
        """Compute ratio of initial population to estimated population in 2010.

        Uses the total of the per-region estimated populations in 2010 used to
        initialise the simulation population as the baseline figure, with this value
        corresponding to the 2010 projected population from [wpp2019]_.

        .. [wpp2019] World Population Prospects 2019. United Nations Department of
        Economic and Social Affairs. URL:
        https://population.un.org/wpp/Download/Standard/Population/

        :param initial_population: Initial population to calculate ratio for.

        :returns: Ratio of ``initial_population`` to 2010 baseline population.
        """
        baseline_total_population = self.parameters['pop_2010']['Count'].sum()
        return initial_population / baseline_total_population


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
    This event causes deaths to persons from cause that are _not_ being modelled explicitly by a disease module.
    It does this by computing the GBD death rates that are implied by all the causes of death other than those that are
    represented in the disease module registered in this simulation.
    """
    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        self.causes_to_represent = self.module.gbd_causes_of_death_not_represented_in_disease_modules
        self.mort_risk_per_poll = self.get_mort_risk_per_poll()

    def get_mort_risk_per_poll(self):
        """Compute the death-rates to use (i.e., those from causes of death defined in `self.causes_to_represent`).
        This is based on using the WPP schedule for all-cause deaths and scaling by the proportion of deaths caused by
        those caused defined in `self.causes_to_represent` (as per the latest available GBD data).
        These mortality rates are used to compute a risk of death per person per occurrence of the polling event.
        """

        # Get the all-cause risk of death per poll
        all_cause_mort_risk = self.get_all_cause_mort_risk_per_poll()

        # Get the proportion of the total death rates that the OtherDeathPollEvent must represent (and log it)
        prop_of_deaths_to_represent = self.get_proportion_of_deaths_to_represent_as_other_deaths()
        logger.info(
            key='other_deaths',
            data=prop_of_deaths_to_represent.reset_index().to_dict(),
            description='proportion of all deaths that are represented as OtherDeaths'
        )

        # Mulitiply probabilities by the proportion of deaths that are to be represented by OtherDeathPollEvent
        all_cause_mort_risk['age_group'] = all_cause_mort_risk['age_years'].map(self.module.AGE_RANGE_LOOKUP)
        mort_risk = all_cause_mort_risk.merge(prop_of_deaths_to_represent.reset_index(name='prop'),
                                              left_on=['sex', 'age_group'],
                                              right_on=['Sex', 'Age_Grp'],
                                              how='left')
        mort_risk['prop'] = mort_risk['prop'].fillna(method='ffill')
        mort_risk['prob_of_dying_before_next_poll'] *= mort_risk['prop']
        assert not mort_risk['prob_of_dying_before_next_poll'].isna().any()

        return mort_risk[['fallbackyear', 'sex', 'age_years', 'prob_of_dying_before_next_poll']]

    def get_all_cause_mort_risk_per_poll(self):
        """Compute the all-cause risk of death per poll"""
        # Get time elapsed between each poll:
        dur_in_years_between_polls = np.timedelta64(self.frequency.months, 'M') / np.timedelta64(1, 'Y')

        # Compute all-cause mortality risk per poll
        return self.module.parameters['all_cause_mortality_schedule'].assign(
            prob_of_dying_before_next_poll=lambda x: (1.0 - np.exp(-x.death_rate * dur_in_years_between_polls))
        ).drop(columns={'death_rate'})

    def get_proportion_of_deaths_to_represent_as_other_deaths(self):
        """Compute the fraction of deaths that will be reprsented by the OtherDeathPoll"""
        # Get the breakdown of deaths by cause from GBD data:
        gbd_deaths = self.module.parameters['gbd_causes_of_death_data']

        # Find the proportion of deaths to be represented by the OtherDeathPoll
        return gbd_deaths[self.causes_to_represent].sum(axis=1)

    def apply(self, population):
        """Randomly select some persons to die of the 'Other' tlo cause (the causes of death that are not represented
        by the disease modules)."""
        # Get shortcut to main dataframe
        df = population.props

        # Cause the death immediately for anyone that is older than the maximum age
        over_max_age = df.index[df.is_alive & (df.age_years > MAX_AGE)]
        for individual_id in over_max_age:
            self.module.do_death(individual_id=individual_id, cause='Other', originating_module=self.module)

        # Get the mortality schedule for now...
        # - get the subset of mortality rates for this year.
        # confirms that we go to the five year period that we are in, not the exact year.
        fallbackyear = int(math.floor(self.sim.date.year / 5) * 5)

        mort_risk = self.mort_risk_per_poll.loc[
            self.mort_risk_per_poll.fallbackyear == fallbackyear, [
                'age_years', 'sex', 'prob_of_dying_before_next_poll']].copy()

        # get the population
        alive = df.loc[df.is_alive & (df.age_years <= MAX_AGE), ['sex', 'age_years']].copy()

        # merge the population dataframe with the parameter dataframe to pick-up the death_rate for each person
        length_before_merge = len(alive)
        alive = alive.reset_index().merge(mort_risk,
                                          left_on=['age_years', 'sex'],
                                          right_on=['age_years', 'sex'],
                                          how='inner').set_index('person')
        assert length_before_merge == len(alive)

        # flipping the coin to determine if this person will die
        will_die = (self.module.rng.random_sample(size=len(alive)) < alive.prob_of_dying_before_next_poll)

        # loop through to see who is going to die:
        for person in alive.index[will_die]:
            # schedule the death for some point in the next month
            self.sim.schedule_event(InstantaneousDeath(self.module, person, cause='Other'),
                                    self.sim.date + DateOffset(days=self.module.rng.randint(0, 30)))


class InstantaneousDeath(Event, IndividualScopeEventMixin):
    """
    Call the do_death function to cause the person to die.

    Note that no checking is done here. (Checking is done within `do_death` which can also be called directly.)

    The 'individual_id' is the index in the population.props dataframe. It is for _one_ person only.
    The 'cause' is the cause that is defined by the disease module (aka, "tlo cause").
    The 'module' passed to this event is the disease module that is causing the death.
    """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)
        self.cause = cause

    def apply(self, individual_id):
        self.sim.modules['Demography'].do_death(individual_id, cause=self.cause, originating_module=self.module)


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
        self.module.popsize_by_year[self.sim.date.year] = df.is_alive.sum()

        # 2) Compute Statistics for the log
        sex_count = df[df.is_alive].groupby('sex').size()

        logger.info(
            key='population',
            data={'total': sum(sex_count),
                  'male': sex_count['M'],
                  'female': sex_count['F']
                  })

        # (nb. if you groupby both sex and age_range, you weirdly lose categories where size==0, so
        # get the counts separately.)
        m_age_counts = df[df.is_alive & (df.sex == 'M')].groupby('age_range').size()
        f_age_counts = df[df.is_alive & (df.sex == 'F')].groupby('age_range').size()

        logger.info(key='age_range_m', data=m_age_counts.to_dict())

        logger.info(key='age_range_f', data=f_age_counts.to_dict())

        # Output by single year of age for under-fives
        # (need to guarantee output always is for each of the years - even if size() is 0)
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

    DO NOT USE THIS FUNCTION IN NEW CODE --- IT IS PROVIDED TO ENABLE LEGACY CODE TO RUN.

    This helper function scales certain outputs so that they can create statistics for the whole population.
    e.g. Population Size, Number of deaths are scaled by the factor of {Model Pop Size at Start of Simulation} to {
    {Real Population at the same time}.

    NB. This file gives precedence to the Malawi Population Census

    :param parsed_output: The outputs from parse_output
    :param resourcefilepath: The resourcefilepath
    :return: a new version of parsed_output that includes certain variables scaled
    """

    print("DO NOT USE THIS FUNCTION IN NEW CODE --- IT IS PROVIDED TO ENABLE LEGACY CODE TO RUN.")
    print()
    print('The scaling factor can found in the log key=scaling_factor')

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

    # For individual-level reporting, construct groupby's and then multiply by ratio
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

    return o
