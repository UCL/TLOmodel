"""
This Module runs the counting of QALYS across all persons and logs it
"""


import logging
import os
import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent, Event

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class HealthBurden(Module):
    """
    This module holds all the stuff to do with DALYS
    """

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    PARAMETERS = {
        'DALY_Weight_Database': Property(Types.DATA_FRAME, 'Weight Database') ,
        # NB. This is a DALY database (From Salomon, Lancet, 2016), so complement of the values must be used.

        'Age_Limit_For_YLL': Property(Types.REAL,
                                      'The age up to which deaths are recorded as having induced a lost of life years')
    }

    PROPERTIES = {}

    def read_parameters(self, data_folder):
        p = self.parameters
        p['DALY_Weight_Database'] = pd.read_csv(os.path.join(self.resourcefilepath, 'ResourceFile_DALY_Weights.csv'))
        p['Age_Limit_For_YLL'] = 70.0   # Assumption that deaths younger than 70y incur years of lost life

    def initialise_population(self, population):
        pass

    def initialise_simulation(self, sim):

        self.Disability = pd.DataFrame()  # This dataframe keeps track of all time lived with disability (age/sex/time)

        # Create dataframe to keep track of the years of live that are lost
        first_year = self.sim.start_date.year
        last_year = self.sim.end_date.year
        self.LifeYearsLost = pd.DataFrame(index=np.arange(first_year,last_year))

        # Check that all registered disease modules have the report_daly_values() function
        for module_name in self.sim.modules['HealthSystem'].registered_disease_modules.keys():
            assert 'report_daly_values' in dir(self.sim.modules['HealthSystem'].registered_disease_modules[module_name])

        # Launch the DALY Logger to run every month
        sim.schedule_event(Log_DALYs(self), sim.date)

    def on_birth(self, mother_id, child_id):
        pass

    def get_daly_weight(self, sequlae_code):
        """
        This can be used to look up the DALY weight for a particular condition identified by the 'sequalue code'
        Sequalae code for particular conditions can be looked-up in ResourceFile_DALY_Weights.csv
        :param sequlae_code:
        :return: the daly weight associatd with that sequalae code
        """
        w = self.parameters['DALY_Weight_Database']
        daly_wt = w.loc[w['TLO_Sequela_Code'] == sequlae_code, 'disability weight'].values[0]

        # Check that the sequalae code was found
        assert (not pd.isnull(daly_wt))

        # Check that the value is within bounds [0,1]
        assert (daly_wt >= 0) & (daly_wt <= 1)

        return daly_wt

    def report_live_years_lost(self,date_of_birth,cause):
        print('reporting that someone had died')

        # calculate the start and end dates of the period for which there is 'years of lost life'
        # NB. the upper limit of the end-date is truncated by the end of the simulation.

        start_date = self.sim.date
        end_date = min(self.sim.end_date , (date_of_birth + pd.DateOffset(years=self.parameters['Age_Limit_For_YLL'] )))

        # get the years of life lost split out by year and age-group
        yll= self.decompose_yll_by_age_and_time(start_date=start_date, end_date=end_date, date_of_birth=date_of_birth)

        # Add the years-of-life-lost from this death to the overall dataframe keeping track
        if cause not in self.LifeYearsLost.columns:
           # cause has not been added to the LifeYearsLost dataframe, so make a new columns
           self.LifeYearsLost[cause]=0.0

        # Add the life-years-lost from this death to the running total in LifeYearsLost dataframe
        self.LifeYearsLost[cause] = self.LifeYearsLost[cause] + yll


    def decompose_yll_by_age_and_time(self,start_date,end_date,date_of_birth):
        """
        This helper function will decompose a period of years of lost life into time-spent in each age group in each
        calendar year
        :param date_of_birth:
        :param date_of_death:
        :param upper_age_limit:
        :return: a dataframe of the person-time (in years) spent by age-group and time-period
        """

        df=pd.DataFrame()

        # Get all the dates between start and end
        df['days']= pd.date_range(start=start_date, end=end_date, freq='D')
        df['year'] = df['days'].dt.year

        # Get the age that this person will be on each day
        df['age_in_years'] =  ( (df['days'] - date_of_birth).dt.days.values/ 365 ).astype(int)

        age_range_lookup = self.sim.modules['Demography'].AGE_RANGE_LOOKUP  # get the age_range_lookup from demography
        df['age_group'] = df['age_in_years'].map(age_range_lookup)

        X=pd.DataFrame(df.groupby(by=['year','age_group'])['days'].count())
        X['person_years']=(X['days']/365).clip(lower=0.0, upper=1.0)

        # replace the multiindex with columns
        X.reset_index(inplace=True)

        X=X.drop(columns=['days'],axis=1)

        return X


class Log_DALYs(RegularEvent, PopulationScopeEventMixin):
    """
    This is the DALY Logger event. It runs every months and asks each disease module to report the average disability
    weight for each living person during the previous month. It reconciles this into a unified value.
    A known (small) limitation of this is that persons who died during the previous month do not contribute any DALYS.
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
            disease_module=self.sim.modules['HealthSystem'].registered_disease_modules[disease_module_name]

            dalys_from_disease_module = disease_module.report_daly_values()

            # Perform checks on what has been returned
            assert type(dalys_from_disease_module) is pd.Series
            assert len(dalys_from_disease_module) == df.is_alive.sum()
            assert (~pd.isnull(dalys_from_disease_module)).all()
            assert ((dalys_from_disease_module >= 0) & (dalys_from_disease_module <= 1)).all()
            assert df.index.name == dalys_from_disease_module.index.name
            assert df.is_alive[dalys_from_disease_module.index].all()

            # Label with the name of the disease module
            dalys_from_disease_module.name=disease_module_name

            # Add to overall data-frame for this month of report dalys
            disease_specific_daly_values_this_month= pd.concat([disease_specific_daly_values_this_month, dalys_from_disease_module], axis=1)


        # 2) Rescale the DALY weights

        # Create a scaling-factor (if total DALYS for one person is more than 1, all DALYS weights are scaled so that
        #   their sum equals one).
        scaling_factor=( disease_specific_daly_values_this_month.sum(axis=1).clip(lower=0,upper=1) / \
            disease_specific_daly_values_this_month.sum(axis=1) ).fillna(1.0)

        disease_specific_daly_values_this_month = disease_specific_daly_values_this_month.multiply(scaling_factor,axis=0)

        # Multiply 1/12 as these weights are for one month only
        disease_specific_daly_values_this_month = disease_specific_daly_values_this_month * (1/12)


        # 3) Summarise the results for this month

        # merge in age/sex information
        disease_specific_daly_values_this_month = disease_specific_daly_values_this_month.merge( \
            df.loc[df.is_alive,['sex','age_range']],left_index=True,right_index=True, how='left')

        # Sum of daly_weight, by sex and age
        Disability_Monthly_Summary = pd.DataFrame( \
            disease_specific_daly_values_this_month.groupby(['sex','age_range']).sum().fillna(0))

        # Replace the multi-index with columns for age/sex
        Disability_Monthly_Summary.reset_index(inplace=True)

        # Add the time-stamp as a column
        Disability_Monthly_Summary['Date']=self.sim.date

        # 4) Add the month's summary to the summary for the whole population
        self.module.Disability = pd.concat([self.module.Disability, Disability_Monthly_Summary], axis=0,ignore_index=True)


class Summarize_And_Log_DALYs(Event, PopulationScopeEventMixin):
    """
    This is event will summarize the DALYS (collapse the dataframes that have been created for Disability and add in the
     Years of Live Lost). It will then output them to the log
    """

    def __init__(self, module):
        super().__init__(module,)

    def apply(self, population):
        pass

        # #       Men
        # logger.info('%s|DALYS_M|%s', self.sim.date,
        #             m_healthstate_sum.to_dict())
        #
        # #       Women
        # logger.info('%s|DALYS_F|%s', self.sim.date,
        #             f_healthstate_sum.to_dict())
