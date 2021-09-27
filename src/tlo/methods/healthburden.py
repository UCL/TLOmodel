"""
This Module runs the counting of DALYS
"""
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Types, logging
from tlo.events import PopulationScopeEventMixin, RegularEvent
from tlo.methods import Metadata
from tlo.methods.causes import (
    Cause,
    collect_causes_from_disease_modules,
    create_mappers_from_causes_to_label,
)

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
        self.multi_index = None
        self.YearsLifeLost = None
        self.YearsLivedWithDisability = None
        self.recognised_modules_names = None
        self.causes_of_disability = None

    INIT_DEPENDENCIES = {'Demography'}

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
        """

        # 1) Prepare data storage structures
        # Create the sex/age_range/year multi-index for YLL and YLD storage dataframes
        sex_index = ['M', 'F']
        year_index = list(range(self.sim.start_date.year, self.sim.end_date.year + 1))
        age_index = self.sim.modules['Demography'].AGE_RANGE_CATEGORIES
        multi_index = pd.MultiIndex.from_product([sex_index, age_index, year_index], names=['sex', 'age_range', 'year'])
        self.multi_index = multi_index

        # Create the YLL and YLD storage data-frame (using sex/age_range/year multi-index)
        self.YearsLifeLost = pd.DataFrame(index=multi_index)
        self.YearsLivedWithDisability = pd.DataFrame(index=multi_index)

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

        # 3) Process the declarations of causes of disability made by the disease modules
        self.process_causes_of_disability()

        # 4) Launch the DALY Logger to run every month, starting with the end of the first month of simulation
        sim.schedule_event(Get_Current_DALYS(self), sim.date + DateOffset(months=1))

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
            gbd_causes=self.get_gbd_causes_of_disability_not_represented_in_disease_modules(self.causes_of_disability)
        )

        # 3) Output to the log mappers for causes of disability
        mapper_from_tlo_causes, mapper_from_gbd_causes = self.create_mappers_from_causes_of_death_to_label()
        logger.info(
            key='mapper_from_tlo_cause_to_common_label',
            data=mapper_from_tlo_causes
        )
        logger.info(
            key='mapper_from_gbd_cause_to_common_label',
            data=mapper_from_gbd_causes
        )

    def on_birth(self, mother_id, child_id):
        pass

    def on_simulation_end(self):
        """Log records of:
        1) The Years Lived With Disability (YLD) (by the 'causes of disability' declared by the disease modules)
        2) The Years Life Lost (YLL) (by the 'causes of death' declared by the disease module)
        3) The total DALYS recorded (YLD + YLL) (by the labels that are declared for 'causes of death' and 'causes of
        disability').
        """

        # Check that the multi-index of the dataframes are as expected
        assert self.YearsLifeLost.index.equals(self.multi_index)
        assert self.YearsLivedWithDisability.index.equals(self.multi_index)

        # 1) Log the Years Lived With Disability (YLD) (by the 'causes of disability' declared by disease modules).
        for index, row in self.YearsLivedWithDisability.reset_index().iterrows():
            logger.info(
                key='yld_by_causes_of_disability',
                data=row.to_dict(),
                description='Years lived with disability by the declared cause_of_disability, '
                            'broken down by year, sex, age-group'
            )

        # 2) Log the Years of Live Lost (YLL) (by the 'causes of death' declared by disease modules).
        for index, row in self.YearsLifeLost.reset_index().iterrows():
            logger.info(
                key='yll_by_causes_of_death',
                data=row.to_dict(),
                description='Years of live lost by the declared cause_of_death, '
                            'broken down by year, sex, age-group'
            )

        # 3) Log total DALYS recorded (YLD + LYL) (by the labels declared)
        dalys = self.compute_dalys()
        # - dump to log, line-by-line
        for index, row in dalys.reset_index().iterrows():
            logger.info(
                key='dalys',
                data=row.to_dict(),
                description='DALYS, by the labels are that are declared for each cause_of_death and cause_of_disability'
                            ', broken down by year, sex, age-group'
            )

    def compute_dalys(self):
        """Compute total DALYS (by label), by age, sex and year. Do this by summing the YLD and LYL with respect to the
         label of the corresponding cause of each, and give output by label."""

        yld = self.YearsLivedWithDisability.rename(
            columns={c: self.causes_of_disability[c].label for c in self.YearsLivedWithDisability.columns}
        )

        yll = self.YearsLifeLost.rename(
            columns={c: self.sim.modules['Demography'].causes_of_death[c].label for c in self.YearsLifeLost.columns}
        )

        return yld.add(yll, fill_value=0)

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

    def report_live_years_lost(self, sex, date_of_birth, cause_of_death):
        """
        Calculate the start and end dates of the period for which there is 'years of lost life' when someone died
        (assuming that the person has died on today's date in the simulation).
        :param sex: sex of the person that had died
        :param date_of_birth: date_of_birth of the person that has died
        :param cause_of_death: title for the column in YLL dataframe (of form <ModuleName>_<Cause>)
        """

        assert self.YearsLifeLost.index.equals(self.multi_index)

        # date from which years of life are lost
        start_date = self.sim.date

        # data to count up to for years of life lost (the earliest of the age_limit or end of simulation)
        end_date = min(self.sim.end_date, (date_of_birth + pd.DateOffset(years=self.parameters['Age_Limit_For_YLL'])))

        # get the years of life lost split out by year and age-group
        yll = self.decompose_yll_by_age_and_time(start_date=start_date, end_date=end_date, date_of_birth=date_of_birth)

        # augment the multi-index of yll with sex so that it is sex/age_range/year
        yll['sex'] = sex
        yll.set_index('sex', append=True, inplace=True)
        yll = yll.reorder_levels(['sex', 'age_range', 'year'])

        # Add the years-of-life-lost from this death to the overall YLL dataframe keeping track
        if cause_of_death not in self.YearsLifeLost.columns:
            # cause has not been added to the LifeYearsLost dataframe, so make a new columns
            self.YearsLifeLost[cause_of_death] = 0.0

        # Add the life-years-lost from this death to the running total in LifeYearsLost dataframe
        indx_before = self.YearsLifeLost.index
        self.YearsLifeLost[cause_of_death] = self.YearsLifeLost[cause_of_death].add(yll['person_years'], fill_value=0)
        indx_after = self.YearsLifeLost.index

        # check that the index of the YLL dataframe is not changed
        assert indx_after.equals(indx_before)
        assert indx_after.equals(self.multi_index)

    def decompose_yll_by_age_and_time(self, start_date, end_date, date_of_birth):
        """
        This helper function will decompose a period of years of lost life into time-spent in each age group in each
        calendar year
        :return: a dataframe (X) of the person-time (in years) spent by age-group and time-period
        """

        df = pd.DataFrame()

        # Get all the days between start and end
        df['days'] = pd.date_range(start=start_date, end=end_date, freq='D')
        df['year'] = df['days'].dt.year

        # Get the age that this person will be on each day
        df['age_in_years'] = ((df['days'] - date_of_birth).dt.days.values / 365).astype(int)

        age_range_lookup = self.sim.modules['Demography'].AGE_RANGE_LOOKUP  # get the age_range_lookup from demography
        df['age_range'] = df['age_in_years'].map(age_range_lookup)

        period = pd.DataFrame(df.groupby(by=['year', 'age_range'])['days'].count())
        period['person_years'] = (period['days'] / 365).clip(lower=0.0, upper=1.0)

        period = period.drop(columns=['days'], axis=1)

        return period

    def get_gbd_causes_of_disability_not_represented_in_disease_modules(self, causes_of_disability):
        """
        Find the causes of disability in the GBD datasets that are not represented within the causes of death defined
        in the modules registered in this simulation.
        :return: set of gbd_causes of disability that are not represented in disease modules
        """
        all_gbd_causes_in_sim = set()
        for c in causes_of_disability.values():
            all_gbd_causes_in_sim.update(c.gbd_causes)

        return set(self.parameters['gbd_causes_of_disability']) - all_gbd_causes_in_sim

    def create_mappers_from_causes_of_death_to_label(self):
        """Use a helper function to create mappers for causes of disability to label."""
        return create_mappers_from_causes_to_label(
            causes=self.causes_of_disability,
            all_gbd_causes=set(self.parameters['gbd_causes_of_disability'])
        )


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
        idx_alive = set(df.loc[df.is_alive].index)

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
                if type(dalys_from_disease_module) is pd.Series:
                    # if a pd.Series is returned, it implies there is only one cause of disability registered by module:
                    assert 1 == len(declared_causes_of_disability_module), \
                        "pd.Series returned but number of causes of disability declared is not equal to one."

                    # name the returned pd.Series as the only cause of disability that is defined by the module
                    dalys_from_disease_module.name = list(declared_causes_of_disability_module)[0]

                    # convert to pd.DataFrame
                    dalys_from_disease_module = pd.DataFrame(dalys_from_disease_module)

                # Perform checks on what has been returned
                assert set(dalys_from_disease_module.columns) == set(declared_causes_of_disability_module)
                assert set(dalys_from_disease_module.index) == idx_alive
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

        # 4) Summarise the results for this month wrt age and sex
        # - merge in age/sex information
        disease_specific_daly_values_this_month = disease_specific_daly_values_this_month.merge(
            df.loc[idx_alive, ['sex', 'age_range']], left_index=True, right_index=True, how='left')

        # - sum of daly_weight, by sex and age
        disability_monthly_summary = pd.DataFrame(
            disease_specific_daly_values_this_month.groupby(['sex', 'age_range']).sum().fillna(0))

        # - add the year into the multi-index
        disability_monthly_summary['year'] = self.sim.date.year
        disability_monthly_summary.set_index('year', append=True, inplace=True)
        disability_monthly_summary = disability_monthly_summary.reorder_levels(['sex', 'age_range', 'year'])

        # 5) Add the monthly summary to the overall dataframe for YearsLivedWithDisability
        dalys_to_add = disability_monthly_summary.sum().sum()     # for checking
        dalys_current = self.module.YearsLivedWithDisability.sum().sum()  # for checking

        # (Nb. this will add columns that are not otherwise present and add values to columns where they are.)
        combined = self.module.YearsLivedWithDisability.combine(
            disability_monthly_summary,
            fill_value=0.0,
            func=np.add,
            overwrite=False)

        # Merge into a dataframe with the correct multi-index (the multindex from combine is subtly different)
        self.module.YearsLivedWithDisability = pd.DataFrame(index=self.module.multi_index).merge(
            combined, left_index=True, right_index=True, how='left')

        # Check multi-index is in check and that the addition of DALYS has worked
        assert self.module.YearsLivedWithDisability.index.equals(self.module.multi_index)
        assert abs(self.module.YearsLivedWithDisability.sum().sum() - (dalys_to_add + dalys_current)) < 1e-5
