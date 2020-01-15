"""
This Module runs the counting of DALYS
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class HealthBurden(Module):
    """
    This module holds all the stuff to do with DALYS
    """

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # instance variables
        self.multi_index = None
        self.YearsLifeLost = None
        self.YearsLivedWithDisability = None

    PARAMETERS = {
        'DALY_Weight_Database': Property(Types.DATA_FRAME, 'DALY Weight Database from GBD'),

        'Age_Limit_For_YLL': Property(Types.REAL,
                                      'The age up to which deaths are recorded as having induced a lost of life years')
    }

    PROPERTIES = {}

    def read_parameters(self, data_folder):
        p = self.parameters
        p['DALY_Weight_Database'] = pd.read_csv(Path(self.resourcefilepath) / 'ResourceFile_DALY_Weights.csv')
        p['Age_Limit_For_YLL'] = 70.0  # Assumption that deaths younger than 70y incur years of lost life

    def initialise_population(self, population):
        pass

    def initialise_simulation(self, sim):

        # Create the sex/age_range/year multi-index for YLL and YLD storage dataframes
        first_year = self.sim.start_date.year
        last_year = self.sim.end_date.year

        sex_index = ['M', 'F']
        year_index = list(range(first_year, last_year + 1))
        age_index = self.sim.modules['Demography'].AGE_RANGE_CATEGORIES
        multi_index = pd.MultiIndex.from_product([sex_index, age_index, year_index], names=['sex', 'age_range', 'year'])
        self.multi_index = multi_index

        # Create the YLL and YLD storage data-frame (using sex/age_range/year multi-index)
        self.YearsLifeLost = pd.DataFrame(index=multi_index)
        self.YearsLivedWithDisability = pd.DataFrame(index=multi_index)

        # Check that all registered disease modules have the report_daly_values() function
        assert 'HealthSystem' in self.sim.modules.keys(), "HealthBurden module is dependent on HealthSystem module."

        for module_name in self.sim.modules['HealthSystem'].registered_disease_modules.keys():
            assert 'report_daly_values' in dir(self.sim.modules['HealthSystem'].registered_disease_modules[module_name])

        # Launch the DALY Logger to run every month
        sim.schedule_event(Get_Current_DALYS(self), sim.date)

    def on_birth(self, mother_id, child_id):
        pass

    def on_simulation_end(self):
        logger.debug('This is being called at the end of the simulation. Time to output to the logs....')

        # Label and concantenate YLL and YLD dataframes
        assert self.YearsLifeLost.index.equals(self.multi_index)
        assert self.YearsLivedWithDisability.index.equals(self.multi_index)

        self.YearsLifeLost = self.YearsLifeLost.add_prefix('YLL_')
        self.YearsLivedWithDisability = self.YearsLivedWithDisability.add_prefix('YLD_')

        dalys = self.YearsLifeLost.join(self.YearsLivedWithDisability)

        # Dump the DALYS dateframe to the log

        # 1) Turn multi-index into regular columns
        dalys = dalys.reset_index()

        # 2) Go line-by-line and dump to the log
        for line_num in range(len(dalys)):
            line_as_dict = dalys.loc[line_num].to_dict()
            year = line_as_dict.pop('year')
            year_as_date = pd.Timestamp(year=year, month=12, day=31)  # log output for the year on 31st December
            logger.info('%s|DALYS|%s', year_as_date, line_as_dict)

    def get_daly_weight(self, sequlae_code):
        """
        This can be used to look up the DALY weight for a particular condition identified by the 'sequalue code'
        Sequalae code for particular conditions can be looked-up in ResourceFile_DALY_Weights.csv
        :param sequlae_code:
        :return: the daly weight associated with that sequalae code
        """
        w = self.parameters['DALY_Weight_Database']
        daly_wt = w.loc[w['TLO_Sequela_Code'] == sequlae_code, 'disability weight'].values[0]

        # Check that the sequalae code was found
        assert (not pd.isnull(daly_wt))

        # Check that the value is within bounds [0,1]
        assert (daly_wt >= 0) & (daly_wt <= 1)

        return daly_wt

    def report_live_years_lost(self, sex, date_of_birth, label):
        """
        Calculate the start and end dates of the period for which there is 'years of lost life'
        :param sex: sex of the person that had died
        :param date_of_birth: date_of_birth of the person that has died
        :param label: title for the column in YLL dataframe (of form <ModuleName>_<CauseOfDeath>)
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
        if label not in self.YearsLifeLost.columns:
            # cause has not been added to the LifeYearsLost dataframe, so make a new columns
            self.YearsLifeLost[label] = 0.0

        # Add the life-years-lost from this death to the running total in LifeYearsLost dataframe
        indx_before = self.YearsLifeLost.index
        self.YearsLifeLost[label] = self.YearsLifeLost[label].add(yll['person_years'], fill_value=0)
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
        logger.debug('The DALY Logger is occuring now! %s', self.sim.date)

        # Get the population dataframe
        df = self.sim.population.props

        # Create temporary dataframe for the reporting of daly weights from all disease modules for the previous month
        # (Each column of this dataframe gives the reports from each module.)
        disease_specific_daly_values_this_month = pd.DataFrame(index=df.index[df.is_alive])

        # 1) Ask each disease module to log the DALYS for the previous month

        for disease_module_name in self.sim.modules['HealthSystem'].registered_disease_modules.keys():

            disease_module = self.sim.modules['HealthSystem'].registered_disease_modules[disease_module_name]

            dalys_from_disease_module = disease_module.report_daly_values()

            # Check type is acceptable and make into dataframe if not already
            assert type(dalys_from_disease_module) in (pd.Series, pd.DataFrame)

            if type(dalys_from_disease_module) is pd.Series:
                dalys_from_disease_module = pd.DataFrame(dalys_from_disease_module)

            # Perform checks on what has been returned
            assert df.index.name == dalys_from_disease_module.index.name
            assert len(dalys_from_disease_module) == df.is_alive.sum()
            assert df.is_alive[dalys_from_disease_module.index].all()
            # assert (~pd.isnull(dalys_from_disease_module)).all().all()
            # assert ((dalys_from_disease_module >= 0) & (dalys_from_disease_module <= 1)).all().all()
            # assert (dalys_from_disease_module.sum(axis=1) <= 1).all()

            # Label with the name of the disease module
            dalys_from_disease_module = dalys_from_disease_module.add_prefix(disease_module_name + '_')

            # Add to overall data-frame for this month of report dalys
            disease_specific_daly_values_this_month = pd.concat([disease_specific_daly_values_this_month,
                                                                 dalys_from_disease_module],
                                                                axis=1)

        # 2) Rescale the DALY weights

        # Create a scaling-factor (if total DALYS for one person is more than 1, all DALYS weights are scaled so that
        #   their sum equals one).
        scaling_factor = (disease_specific_daly_values_this_month.sum(axis=1).clip(lower=0, upper=1) /
                          disease_specific_daly_values_this_month.sum(axis=1)).fillna(1.0)

        disease_specific_daly_values_this_month = disease_specific_daly_values_this_month.multiply(scaling_factor,
                                                                                                   axis=0)

        # Multiply 1/12 as these weights are for one month only
        disease_specific_daly_values_this_month = disease_specific_daly_values_this_month * (1 / 12)

        # 3) Summarise the results for this month wrt age and sex

        # merge in age/sex information
        disease_specific_daly_values_this_month = disease_specific_daly_values_this_month.merge(
            df.loc[df.is_alive, ['sex', 'age_range']], left_index=True, right_index=True, how='left')

        # Sum of daly_weight, by sex and age
        disability_monthly_summary = pd.DataFrame(
            disease_specific_daly_values_this_month.groupby(['sex', 'age_range']).sum().fillna(0))

        # Add the year into the multi-index
        disability_monthly_summary['year'] = self.sim.date.year
        disability_monthly_summary.set_index('year', append=True, inplace=True)
        disability_monthly_summary = disability_monthly_summary.reorder_levels(['sex', 'age_range', 'year'])

        # 4) Add the monthly summary to the overall datafrom for YearsLivedWithDisability

        dalys_to_add = disability_monthly_summary.sum().sum()     # for checking
        dalys_current = self.module.YearsLivedWithDisability.sum().sum()

        # This will add columns that are not otherwise present and add values to columns where they are
        combined = self.module.YearsLivedWithDisability.combine(
            disability_monthly_summary,
            fill_value=0.0,
            func=np.add,
            overwrite=False)

        # merge into a dataframe with the correct multi-index (the multindex from combine is subtly different)
        self.module.YearsLivedWithDisability = pd.DataFrame(index=self.module.multi_index).merge(
            combined, left_index=True, right_index=True, how='left')

        # check multi-index is in check and that the addition of DALYS has worked
        assert self.module.YearsLivedWithDisability.index.equals(self.module.multi_index)
        assert abs(self.module.YearsLivedWithDisability.sum().sum() - (dalys_to_add + dalys_current)) < 1e-5
