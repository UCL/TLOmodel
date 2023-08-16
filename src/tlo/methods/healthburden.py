"""
This Module runs the counting of Life-years Lost, Life-years Lived with Disability,
and Disability-Adjusted Life-years (DALYS).
"""
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from tlo import Date, DateOffset, Module, Parameter, Types, logging
from tlo.events import PopulationScopeEventMixin, Priority, RegularEvent
from tlo.methods import Metadata
from tlo.methods.causes import (
    Cause,
    collect_causes_from_disease_modules,
    create_mappers_from_causes_to_label,
    get_gbd_causes_not_represented_in_disease_modules,
)
from tlo.methods.demography import age_at_date

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HealthBurden(Module):
    """
    This module holds all the stuff to do with recording DALYS
    """

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # instance variables
        self.multi_index_for_age_and_wealth_and_time = None
        self.years_life_lost = None
        self.years_life_lost_stacked_time = None
        self.years_life_lost_stacked_age_and_time = None
        self.years_lived_with_disability = None
        self.recognised_modules_names = None
        self.causes_of_disability = None
        self._causes_of_yll = None
        self._causes_of_dalys = None
        self._years_written_to_log = []

    INIT_DEPENDENCIES = {'Demography'}

    ADDITIONAL_DEPENDENCIES = {'Lifestyle'}

    # Declare Metadata
    METADATA = {}

    PARAMETERS = {
        'DALY_Weight_Database': Parameter(
            Types.DATA_FRAME, 'DALY Weight Database from GBD'),
        'Age_Limit_For_YLL': Parameter(
            Types.REAL, 'The age up to which deaths are recorded as having induced a lost of life years'),
        'gbd_causes_of_disability': Parameter(
            Types.LIST, 'List of the strings of causes of disability defined in the GBD data')
    }

    PROPERTIES = {}

    def read_parameters(self, data_folder):
        p = self.parameters
        p['DALY_Weight_Database'] = pd.read_csv(Path(self.resourcefilepath) / 'ResourceFile_DALY_Weights.csv')
        p['Age_Limit_For_YLL'] = 70.0  # Assumption that only deaths younger than 70y incur years of lost life
        p['gbd_causes_of_disability'] = set(pd.read_csv(
            Path(self.resourcefilepath) / 'gbd' / 'ResourceFile_CausesOfDALYS_GBD2019.csv', header=None)[0].values)

    def initialise_population(self, population):
        pass

    def initialise_simulation(self, sim):
        """Do before simulation starts:
        1) Prepare data storage structures
        2) Collect the module that will use this HealthBurden module
        3) Process the declarations of causes of disability made by the disease modules
        4) Launch the DALY Logger to run every month, starting with the end of the first month of simulation
        5) Schedule `Healthburden_WriteToLog` that will write to log annually (end of each year)
        """

        # 1) Prepare data storage structures
        # Create the sex/age_range/year multi-index for YLL and YLD storage dataframes
        sex_index = ['M', 'F']
        age_index = self.sim.modules['Demography'].AGE_RANGE_CATEGORIES
        wealth_index = sim.modules['Lifestyle'].PROPERTIES['li_wealth'].categories
        year_index = list(range(self.sim.start_date.year, self.sim.end_date.year + 1))

        self.multi_index_for_age_and_wealth_and_time = pd.MultiIndex.from_product(
            [sex_index, age_index, wealth_index, year_index], names=['sex', 'age_range', 'li_wealth', 'year'])

        # Create the YLL and YLD storage data-frame (using sex/age_range/year multi-index)
        self.years_life_lost = pd.DataFrame(index=self.multi_index_for_age_and_wealth_and_time)
        self.years_life_lost_stacked_time = pd.DataFrame(index=self.multi_index_for_age_and_wealth_and_time)
        self.years_life_lost_stacked_age_and_time = pd.DataFrame(index=self.multi_index_for_age_and_wealth_and_time)
        self.years_lived_with_disability = pd.DataFrame(index=self.multi_index_for_age_and_wealth_and_time)

        # 2) Collect the module that will use this HealthBurden module
        self.recognised_modules_names = [
            m.name for m in self.sim.modules.values() if Metadata.USES_HEALTHBURDEN in m.METADATA
        ]

        # Check that all registered disease modules have the report_daly_values() function
        for module_name in self.recognised_modules_names:
            assert getattr(self.sim.modules[module_name], 'report_daly_values', None) and \
                   callable(self.sim.modules[module_name].report_daly_values), 'A module that declares use of ' \
                                                                               'HealthBurden module must have a ' \
                                                                               'callable function "report_daly_values"'

        # 3) Process the declarations of causes of disability and DALYS made by the disease modules
        self.process_causes_of_disability()
        self.process_causes_of_dalys()

        # 4) Launch the DALY Logger to run every month, starting with the end of the first month of simulation
        sim.schedule_event(Get_Current_DALYS(self), sim.date + DateOffset(months=1))

        # 5) Schedule `Healthburden_WriteToLog` that will write to log annually
        last_day_of_the_year = Date(sim.date.year, 12, 31)
        sim.schedule_event(Healthburden_WriteToLog(self), last_day_of_the_year)

    def process_causes_of_disability(self):
        """
        1) Collect causes of disability that are reported by each disease module
        2) Define the "Other" tlo_cause of disability (corresponding to those gbd_causes that are not represented by
        the disease modules in this sim.)
        3) Output to the log mappers for causes of disability to the label
        """
        # 1) Collect causes of disability that are reported by each disease module
        self.causes_of_disability = collect_causes_from_disease_modules(
            all_modules=self.sim.modules.values(),
            collect='CAUSES_OF_DISABILITY',
            acceptable_causes=set(self.parameters['gbd_causes_of_disability'])
        )

        # 2) Define the "Other" tlo_cause of disability
        self.causes_of_disability['Other'] = Cause(
            label='Other',
            gbd_causes=get_gbd_causes_not_represented_in_disease_modules(
                causes=self.causes_of_disability,
                gbd_causes=set(self.parameters['gbd_causes_of_disability'])
            )
        )

        # 3) Output to the log mappers for causes of disability
        mapper_from_tlo_causes, mapper_from_gbd_causes = create_mappers_from_causes_to_label(
            causes=self.causes_of_disability,
            all_gbd_causes=set(self.parameters['gbd_causes_of_disability'])
        )

        logger.info(
            key='disability_mapper_from_tlo_cause_to_common_label',
            data=mapper_from_tlo_causes
        )
        logger.info(
            key='disability_mapper_from_gbd_cause_to_common_label',
            data=mapper_from_gbd_causes
        )

    def process_causes_of_dalys(self):
        """
        1) Collect causes of DALYS (i.e., death _and_ disability) that are reported by each disease module
        2) Define the "Other" tlo_cause of DALYS (corresponding to those gbd_causes that are not represented by
        the disease modules in this sim.)
        3) Output to the log mappers for causes of disability to the label
        """
        ...
        # 1) Collect causes of death and disability that are reported by each disease module,
        #    merging the gbd_causes declared for deaths or disabilities under the same label,

        def merge_dicts_of_causes(d1: Dict, d2: Dict) -> Dict:
            """Combine two dictionaries of the form {tlo_cause_name: Cause}, merging the values of `Cause.gbd_causes`
             where the values of `Cause.label` are common, attaching to the first key in d1 that uses that label."""
            labels_seen = dict()  # Look-up of the form {label: tlo_cause_name} for the tlo_cause_name associated
            #                       (first) with a label.
            merged_causes = dict()  # Dict that will build-up as {tlo_cause_name: Cause} and be returned
            for d in (d1, d2):
                for _tlo_cause_name, _cause in d.items():
                    if _cause.label not in labels_seen:
                        # If label is not already included, add this cause to the merged dict
                        merged_causes[_tlo_cause_name] = _cause
                        labels_seen[_cause.label] = _tlo_cause_name
                    else:
                        # If label is already included, merge the gbd_causes into the cause defined.
                        tlo_cause_name_to_merge_into = labels_seen[_cause.label]
                        merged_causes[tlo_cause_name_to_merge_into].gbd_causes = \
                            merged_causes[tlo_cause_name_to_merge_into].gbd_causes.union(_cause.gbd_causes)
            return merged_causes

        causes_of_death = collect_causes_from_disease_modules(
                all_modules=self.sim.modules.values(),
                collect='CAUSES_OF_DEATH',
                acceptable_causes=self.sim.modules['Demography'].gbd_causes_of_death)
        causes_of_disability = collect_causes_from_disease_modules(
                all_modules=self.sim.modules.values(),
                collect='CAUSES_OF_DISABILITY',
                acceptable_causes=set(self.parameters['gbd_causes_of_disability']))

        causes_of_death_and_disability = merge_dicts_of_causes(
            causes_of_death,
            causes_of_disability,
        )

        # N.B. In the GBD definitions, MANY things which disable but don't kill; but NO things that kill but which
        # don't also disable (because things that kill cause DALYS that way.)
        assert set(self.parameters['gbd_causes_of_disability']).issuperset(
            self.sim.modules['Demography'].gbd_causes_of_death)

        # 2) Define the "Other" cause
        all_gbd_causes_of_death_and_disability = set(self.parameters['gbd_causes_of_disability']).union(
            self.sim.modules['Demography'].gbd_causes_of_death
        )
        causes_of_death_and_disability['Other'] = Cause(
            label='Other',
            gbd_causes=get_gbd_causes_not_represented_in_disease_modules(
                causes=causes_of_death_and_disability,
                gbd_causes=all_gbd_causes_of_death_and_disability
            )
        )

        # 3) Output to the log mappers for causes of DALYs
        mapper_from_tlo_causes, mapper_from_gbd_causes = create_mappers_from_causes_to_label(
            causes=causes_of_death_and_disability,
            all_gbd_causes=all_gbd_causes_of_death_and_disability
        )

        logger.info(
            key='daly_mapper_from_tlo_cause_to_common_label',
            data=mapper_from_tlo_causes
        )
        logger.info(
            key='daly_mapper_from_gbd_cause_to_common_label',
            data=mapper_from_gbd_causes
        )

        # store all possible causes of YLL and DALYS
        self._causes_of_yll = set(causes_of_death.keys()).union({'Other'})
        self._causes_of_dalys = set(mapper_from_tlo_causes.values())

    def on_birth(self, mother_id, child_id):
        pass

    def on_simulation_end(self):
        """Write to the log anything that has not already been logged (i.e., if simulation terminating mid-way through
        a year when the WriteToLog event has not run."""
        self.write_to_log(year=self.sim.date.year)

    def get_dalys(self, yld: pd.DataFrame, yll: pd.DataFrame) -> pd.DataFrame:
        """Returns pd.DataFrame of DALYS that is the sum of the 'Years Lived with Disability' (`yld`) and the 'Years
         of Life Lost' (`yll`), under their common 'labels'. (i.e. multiple causes of yld and of yll may occur on the
         same labels, and these labels unite the causes across yll and yld.)
         """
        # Put YLD under common label
        yld_with_label = yld.rename(
            columns={
                c: self.causes_of_disability[c].label for c in yld.columns
                if c in self.causes_of_disability
            }
        )

        # Put YLL under common label
        yll_with_label = yll.rename(
            columns={
                c: self.sim.modules['Demography'].causes_of_death[c].label for c in yll
                if c in self.sim.modules['Demography'].causes_of_death
            }
        )

        # Join together and add, setting the index to be any columns that are not the causes_of_dalys (e.g. year, sex,
        # age_range, wealth)
        tots = pd.concat(
            [
                yld_with_label.set_index(sorted([i for i in yld_with_label.columns if i not in self._causes_of_dalys])),
                yll_with_label.set_index(sorted([i for i in yll_with_label.columns if i not in self._causes_of_dalys])),
            ],
            axis=1,
        )
        return tots.groupby(tots.columns, axis=1).sum().reset_index()

    def get_daly_weight(self, sequlae_code):
        """
        This can be used to look up the DALY weight for a particular condition identified by the 'sequela code'
        Sequela code for particular conditions can be looked-up in ResourceFile_DALY_Weights.csv
        :param sequela_code:
        :return: the daly weight associated with that sequela code
        """
        w = self.parameters['DALY_Weight_Database']
        daly_wt = w.loc[w['TLO_Sequela_Code'] == sequlae_code, 'disability weight'].values[0]

        # Check that the sequela code was found
        assert (not pd.isnull(daly_wt))

        # Check that the value is within bounds [0,1]
        assert (daly_wt >= 0) & (daly_wt <= 1)

        return daly_wt

    def report_live_years_lost(self, sex=None, wealth=None, date_of_birth=None, age_range=None, cause_of_death=None):
        """
        Calculate and store the period for which there is 'years of lost life' when someone dies (assuming that the
        person has died on today's date in the simulation).
        :param sex: sex of the person that had died
        :param wealth: the value 'li_wealth' for the person at the time of death
        :param date_of_birth: date_of_birth of the person that has died
        :param age_range: The age-range for the person at the time of death
        :param cause_of_death: title for the column in YLL dataframe (of form <ModuleName>_<Cause>)
        """

        def _format_for_multi_index(_yll: pd.Series):
            """Returns pd.Series which is the same as in the argument `_yll` except that the multi-index has been
            expanded to include sex and li_wealth and rearranged so that it matched the expected multi-index format
            (sex/age_range/li_wealth/year)."""
            return pd.DataFrame(_yll)\
                     .assign(sex=sex, li_wealth=wealth)\
                     .set_index(['sex', 'li_wealth'], append=True)\
                     .reorder_levels(['sex', 'age_range', 'li_wealth', 'year'])[_yll.name]

        assert self.years_life_lost.index.equals(self.multi_index_for_age_and_wealth_and_time)
        assert self.years_life_lost_stacked_time.index.equals(self.multi_index_for_age_and_wealth_and_time)
        assert self.years_life_lost_stacked_age_and_time.index.equals(self.multi_index_for_age_and_wealth_and_time)

        # date from which years of life are lost
        date_of_death = self.sim.date

        # Get the years of life lost split out by year and age-group: Not Stacked by time... so counting years of life
        # lost up to the earliest of the age_limit or end of simulation.
        yll = self.decompose_yll_by_age_and_time(start_date=date_of_death,
                                                 end_date=min(
                                                     self.sim.end_date,
                                                     (date_of_birth +
                                                      pd.DateOffset(years=self.parameters['Age_Limit_For_YLL']))
                                                 ),
                                                 date_of_birth=date_of_birth
                                                 )['person_years'].pipe(_format_for_multi_index)

        # Get the years of live lost "stacked by time", whereby all the life-years lost up to the age_limit are ascribed
        # to the year of death.
        yll_stacked_by_time = \
            self.decompose_yll_by_age_and_time(
                start_date=date_of_death,
                end_date=(
                    date_of_birth + pd.DateOffset(years=self.parameters['Age_Limit_For_YLL']) - pd.DateOffset(days=1)),
                date_of_birth=date_of_birth
            ).groupby(level=1).sum()\
             .assign(year=date_of_death.year)\
             .set_index(['year'], append=True)['person_years']\
             .pipe(_format_for_multi_index)

        # Get the years of live lost "stacked by age and time", whereby all the life-years lost up to the age_limit are
        # ascribed to the age of death and to the year of death. This is computed by collapsing the age-dimension of
        # `yll_stacked_by_time` onto the age(-range) of death.
        age_range_to_stack_to = age_range
        yll_stacked_by_age_and_time = pd.DataFrame(yll_stacked_by_time.groupby(level=[0, 2, 3]).sum())\
                                        .assign(age_range=age_range_to_stack_to)\
                                        .set_index(['age_range'], append=True)['person_years']\
                                        .reorder_levels(['sex', 'age_range', 'li_wealth', 'year'])

        # Add the years-of-life-lost from this death to the overall YLL dataframe keeping track
        if cause_of_death not in self.years_life_lost.columns:
            # cause has not been added to the LifeYearsLost dataframe, so make a new columns
            self.years_life_lost[cause_of_death] = 0.0
            self.years_life_lost_stacked_time[cause_of_death] = 0.0
            self.years_life_lost_stacked_age_and_time[cause_of_death] = 0.0

        # Add the life-years-lost from this death to the running total in LifeYearsLost dataframe
        self.years_life_lost[cause_of_death] = self.years_life_lost[cause_of_death].add(
            yll, fill_value=0)
        self.years_life_lost_stacked_time[cause_of_death] = self.years_life_lost_stacked_time[cause_of_death].add(
            yll_stacked_by_time, fill_value=0)
        self.years_life_lost_stacked_age_and_time[cause_of_death] = \
            self.years_life_lost_stacked_age_and_time[cause_of_death].add(yll_stacked_by_age_and_time, fill_value=0)

        # Check that the index of the YLL dataframe is not changed
        assert self.years_life_lost.index.equals(self.multi_index_for_age_and_wealth_and_time)
        assert self.years_life_lost_stacked_time.index.equals(self.multi_index_for_age_and_wealth_and_time)
        assert self.years_life_lost_stacked_age_and_time.index.equals(self.multi_index_for_age_and_wealth_and_time)

    def decompose_yll_by_age_and_time(self, start_date, end_date, date_of_birth):
        """
        This helper function will decompose a period of years of lost life into time-spent in each age group in each
        calendar year
        :return: a dataframe (X) of the person-time (in years) spent by age-group and time-period
        """

        df = pd.DataFrame()

        # Get all the days between start and end (inclusively)
        df['days'] = pd.date_range(start=start_date, end=end_date, freq='D')
        df['year'] = df['days'].dt.year

        # Get the age (in whole years) that this person will be on each day.
        # N.B. This is a slight approximation as it doesn't make allowance for leap-years.
        df['age_in_years'] = age_at_date(df['days'], date_of_birth).astype(int)

        age_range_lookup = self.sim.modules['Demography'].AGE_RANGE_LOOKUP  # get the age_range_lookup from demography
        df['age_range'] = df['age_in_years'].map(age_range_lookup)

        period = pd.DataFrame(df.groupby(by=['year', 'age_range'])['days'].count())
        period['person_years'] = (period['days'] / 365).clip(lower=0.0, upper=1.0)

        period = period.drop(columns=['days'], axis=1)

        return period

    def write_to_log(self, year: int):
        """Write to the log the YLL, YLD and DALYS for a specific year.
        N.B. This is called at the end of the simulation as well as at the end of each year, so we need to check that
        the year is not being written to the log more than once."""

        if year in self._years_written_to_log:
            return  # Skip if the year has already been logged.

        def summarise_results_for_this_year(df, level=[0, 1]) -> pd.DataFrame:
            """Return pd.DataFrame that gives the summary of the `df` for the `year` by certain levels in the df's
            multi-index. The `level` argument gives a list of levels to use in `groupby`: e.g., level=[0,1] gives a
            summary of sex/age-group; and level=[2] gives a summary only by wealth category."""
            return df.loc[(slice(None), slice(None), slice(None), year)] \
                     .groupby(level=level) \
                     .sum() \
                     .reset_index() \
                     .assign(year=year)

        def log_df_line_by_line(key, description, df, force_cols=None) -> None:
            """Log each line of a dataframe to `logger.info`. Each row of the dataframe is one logged entry.
            `force_cols` is the names of the colums that must be included in each logging line (As the parsing of the
            log requires the name of the format of each row to be uniform.)."""
            df[sorted(set(force_cols) - set(df.columns))] = 0.0  # Force the addition of any missing causes
            df = df[sorted(df.columns)]  # sort the columns so that they are always in same order
            for _, row in df.iterrows():
                logger.info(
                    key=key,
                    data=row.to_dict(),
                    description=description,
                )

        # Check that the format of the internal storage is as expected.
        self.check_multi_index()

        # 1) Log the Years Lived With Disability (YLD) (by the 'causes of disability' declared by disease modules).
        log_df_line_by_line(
            key='yld_by_causes_of_disability',
            description='Years lived with disability by the declared cause_of_disability, '
                        'broken down by year, sex, age-group',
            df=(yld := summarise_results_for_this_year(self.years_lived_with_disability)),
            force_cols=sorted(set(self.causes_of_disability.keys())),
        )

        # 2) Log the Years of Live Lost (YLL) (by the 'causes of death' declared by disease modules).
        log_df_line_by_line(
            key='yll_by_causes_of_death',
            description='Years of life lost by the declared cause_of_death, '
                        'broken down by year, sex, age-group. '
                        'No stacking: i.e., each year of life lost is ascribed to the'
                        ' age and year that the person would have lived.',
            df=(yll := summarise_results_for_this_year(self.years_life_lost)),
            force_cols=self._causes_of_yll,
        )
        log_df_line_by_line(
            key='yll_by_causes_of_death_stacked',
            description='Years of life lost by the declared cause_of_death, '
                        'broken down by year, sex, age-group. '
                        'Stacking by time: i.e., every year of life lost is ascribed to'
                        ' the year of the death, but each is ascribed to the age that '
                        'the person would have lived, .',
            df=(yll_stacked_by_time := summarise_results_for_this_year(self.years_life_lost_stacked_time)),
            force_cols=self._causes_of_yll,
        )
        log_df_line_by_line(
            key='yll_by_causes_of_death_stacked_by_age_and_time',
            description='Years of life lost by the declared cause_of_death, '
                        'broken down by year, sex, age-group. '
                        'Stacking by age and time: i.e., all the year of life lost '
                        'are ascribed to the age of the death and the year of the death.',
            df=(yll_stacked_by_age_and_time := summarise_results_for_this_year(
                self.years_life_lost_stacked_age_and_time)),
            force_cols=self._causes_of_yll,
        )

        # 3) Log total DALYS recorded (YLD + LYL) (by the labels declared)
        log_df_line_by_line(
            key='dalys',
            description='DALYS, by the labels are that are declared for each cause_of_death and cause_of_disability'
                        ', broken down by year, sex, age-group. '
                        'No stacking: i.e., each year of life lost is ascribed to the'
                        ' age and year that the person would have lived.',
            df=self.get_dalys(yld=yld, yll=yll),
            force_cols=self._causes_of_dalys,
        )
        log_df_line_by_line(
            key='dalys_stacked',
            description='DALYS, by the labels are that are declared for each cause_of_death and cause_of_disability'
                        ', broken down by year, sex, age-group. '
                        'Stacking by time: i.e., every year of life lost is ascribed to'
                        ' the year of the death, but each is ascribed to the age that '
                        'the person would have lived, .',
            df=self.get_dalys(yld=yld, yll=yll_stacked_by_time),
            force_cols=self._causes_of_dalys,
        )
        log_df_line_by_line(
            key='dalys_stacked_by_age_and_time',
            description='DALYS, by the labels are that are declared for each cause_of_death and cause_of_disability'
                        ', broken down by year, sex, age-group. '
                        'Stacking by age and time: i.e., all the year of life lost '
                        'are ascribed to the age of the death and the year of the death.',
            df=self.get_dalys(yld=yld, yll=yll_stacked_by_age_and_time),
            force_cols=self._causes_of_dalys,
        )

        # 4) Log total DALYS (Stacked by Age and Time), broken down by wealth only (with the YLL stacked by age and
        # time)
        yld_by_wealth = summarise_results_for_this_year(
            self.years_lived_with_disability, level=2
        )
        yll_by_wealth = summarise_results_for_this_year(
            self.years_life_lost_stacked_age_and_time, level=2
        )

        log_df_line_by_line(
            key='dalys_by_wealth_stacked_by_age_and_time',
            description='DALYS, by the labels are that are declared for each cause_of_death and cause_of_disability'
                        ', broken down by year and wealth category.'
                        'Stacking by age and time: i.e., all the year of life lost '
                        'are ascribed to the age of the death and the year of the death.',
            df=self.get_dalys(yld=yld_by_wealth, yll=yll_by_wealth),
            force_cols=self._causes_of_dalys,
        )

        self._years_written_to_log += [year]

    def check_multi_index(self):
        """Check that the multi-index of the dataframes are as expected"""
        assert self.years_life_lost.index.equals(self.multi_index_for_age_and_wealth_and_time)
        assert self.years_life_lost_stacked_time.index.equals(self.multi_index_for_age_and_wealth_and_time)
        assert self.years_life_lost_stacked_age_and_time.index.equals(self.multi_index_for_age_and_wealth_and_time)
        assert self.years_lived_with_disability.index.equals(self.multi_index_for_age_and_wealth_and_time)


class Get_Current_DALYS(RegularEvent, PopulationScopeEventMixin):
    """
    This event runs every months and asks each disease module to report the average disability
    weight for each living person during the previous month. It reconciles this with reports from other disease modules
    to ensure that no person has a total weight greater than one.
    A known (small) limitation of this is that persons who died during the previous month do not contribute any YLD.
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        # Running the DALY Logger

        # Do nothing if no disease modules are registered or no causes of disability are registered
        if (not self.module.recognised_modules_names) or (not self.module.causes_of_disability):
            return

        # Get the population dataframe
        df = self.sim.population.props
        idx_alive = df.loc[df.is_alive].index

        # 1) Ask each disease module to log the DALYS for the previous month
        dalys_from_each_disease_module = list()
        for disease_module_name in self.module.recognised_modules_names:

            disease_module = self.sim.modules[disease_module_name]
            declared_causes_of_disability_module = disease_module.CAUSES_OF_DISABILITY.keys()

            if declared_causes_of_disability_module:
                # if some causes of disability are declared, collect the disability reported by this disease module:
                dalys_from_disease_module = disease_module.report_daly_values()

                # Check type is in acceptable form and make into dataframe if not already
                assert type(dalys_from_disease_module) in (pd.Series, pd.DataFrame)
                if isinstance(dalys_from_disease_module, pd.Series):
                    # if a pd.Series is returned, it implies there is only one cause of disability registered by module:
                    assert 1 == len(declared_causes_of_disability_module), \
                        "pd.Series returned but number of causes of disability declared is not equal to one."

                    # name the returned pd.Series as the only cause of disability that is defined by the module
                    dalys_from_disease_module.name = list(declared_causes_of_disability_module)[0]

                    # convert to pd.DataFrame
                    dalys_from_disease_module = pd.DataFrame(dalys_from_disease_module)

                # Perform checks on what has been returned
                assert set(dalys_from_disease_module.columns) == set(declared_causes_of_disability_module)
                assert set(dalys_from_disease_module.index) == set(idx_alive)
                assert not pd.isnull(dalys_from_disease_module).any().any()
                assert ((dalys_from_disease_module >= 0) & (dalys_from_disease_module <= 1)).all().all()
                assert (dalys_from_disease_module.sum(axis=1) <= 1).all()

                # Append to list of dalys reported by each module
                dalys_from_each_disease_module.append(dalys_from_disease_module)

        # 2) Combine into a single dataframe (each column of this dataframe gives the reports from each module), and
        # add together dalys reported by different modules that have the same cause (i.e., add together columns with
        # the same name).
        disease_specific_daly_values_this_month = pd.concat(
            dalys_from_each_disease_module, axis=1).groupby(axis=1, level=0).sum()

        # 3) Rescale the DALY weights
        # Create a scaling-factor (if total DALYS for one person is more than 1, all DALYS weights are scaled so that
        #   their sum equals one).
        scaling_factor = (disease_specific_daly_values_this_month.sum(axis=1).clip(lower=0, upper=1) /
                          disease_specific_daly_values_this_month.sum(axis=1)).fillna(1.0)

        disease_specific_daly_values_this_month = disease_specific_daly_values_this_month.multiply(scaling_factor,
                                                                                                   axis=0)
        assert ((disease_specific_daly_values_this_month.sum(axis=1) - 1.0) < 1e-6).all()

        # Multiply 1/12 as these weights are for one month only
        disease_specific_daly_values_this_month = disease_specific_daly_values_this_month * (1 / 12)

        # 4) Summarise the results for this month wrt sex/age/wealth
        # - merge in age/wealth/sex information
        disease_specific_daly_values_this_month = disease_specific_daly_values_this_month.merge(
            df.loc[idx_alive, ['sex', 'li_wealth', 'age_range']], left_index=True, right_index=True, how='left')

        # - sum of daly_weight, by sex/age/wealth
        disability_monthly_summary = pd.DataFrame(
            disease_specific_daly_values_this_month.groupby(['sex', 'age_range', 'li_wealth']).sum().fillna(0))

        # - add the year into the multi-index
        disability_monthly_summary['year'] = self.sim.date.year
        disability_monthly_summary.set_index('year', append=True, inplace=True)
        disability_monthly_summary = disability_monthly_summary.reorder_levels(
            ['sex', 'age_range', 'li_wealth', 'year'])

        # 5) Add the monthly summary to the overall dataframe for YearsLivedWithDisability
        dalys_to_add = disability_monthly_summary.sum().sum()     # for checking
        dalys_current = self.module.years_lived_with_disability.sum().sum()  # for checking

        # (Nb. this will add columns that are not otherwise present and add values to columns where they are.)
        combined = self.module.years_lived_with_disability.combine(
            disability_monthly_summary,
            fill_value=0.0,
            func=np.add,
            overwrite=False)

        # Merge into a dataframe with the correct multi-index (the multi-index from combine is subtly different)
        self.module.years_lived_with_disability = \
            pd.DataFrame(index=self.module.multi_index_for_age_and_wealth_and_time)\
              .merge(combined, left_index=True, right_index=True, how='left')

        # Check multi-index is in check and that the addition of DALYS has worked
        assert self.module.years_lived_with_disability.index.equals(self.module.multi_index_for_age_and_wealth_and_time)
        assert abs(self.module.years_lived_with_disability.sum().sum() - (dalys_to_add + dalys_current)) < 1e-5
        self.module.check_multi_index()


class Healthburden_WriteToLog(RegularEvent, PopulationScopeEventMixin):
    """ This event runs every year, as the last event on the last day of the year, and writes to the log the YLD, YLL
    and DALYS accrued in that year."""

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(years=1), priority=Priority.END_OF_DAY)

    def apply(self, population):
        self.module.write_to_log(year=self.sim.date.year)
