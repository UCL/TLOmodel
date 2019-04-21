"""
This Module runs the counting of QALYS across all persons and logs it
"""

import logging
import os

import pandas as pd

from tlo import DateOffset, Module, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)


class QALY(Module):
    """
    This module holds all the stuff to do with QALYS.
    Disease modules can query this module to find the weights to use.
    A recurring event is run which aggregates and log the total QALYS.
    """

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath


    PARAMETERS = {
        'Weight_Database': Property(Types.DATA_FRAME, 'Weight Database')
        # NB. This is a DALY database (From Salomon, Lancet, 2016), so complement of the values must be used.
    }

    PROPERTIES = {
        'qa_QALY': Property(Types.REAL, 'Summary of health state for previous year')
    }

    def read_parameters(self, data_folder):
        p = self.parameters
        p['Weight_Database'] = pd.read_csv(os.path.join(self.resourcefilepath,'ResourceFile_DALY_Weights.csv'))


    def initialise_population(self, population):
        pass

    def initialise_simulation(self, sim):
        """ Launch the QALY Logger to run every year
        """
        sim.schedule_event(LogQALYs(self), sim.date)

    def on_birth(self, mother_id, child_id):
        pass

    def get_qaly_weight(self, sequlae_code):
        """
        This can be used to look up the QALY code for a particular condition.
        Disease module calling this pass a sequlae code, which has been chosen after consulting the
        ResourceFile_DALY_Weights.csv

        """
        w = self.parameters['Weight_Database']
        daly_wt = w.loc[w['TLO_Sequela_Code'] == sequlae_code, 'disability weight'].values[0]

        # Check that the sequalue code was found
        assert (not pd.isnull(daly_wt))

        qaly_wt = 1 - daly_wt

        # Check that the value is within bounds [0,1]
        assert (qaly_wt>=0) & (qaly_wt<=1)

        return qaly_wt


class LogQALYs(RegularEvent, PopulationScopeEventMixin):
    """
    This is the QALY Logger event. It runs every year and asks each disease module to report the quality of life lived
    for each individual over the previous 12 months. It reconciles this into a unified values and outputs it to the
    log.
    A known limitation of this is that persons who died in the previous year do not contribute any QALYS.
    """

    def __init__(self, module):

        super().__init__(module, frequency=DateOffset(months=12))

    def apply(self, population):

        # Running the QALY Logger
        logger.debug('The QALY Logger is occuring now! %s', self.sim.date)

        disease_specific_qaly_values = pd.DataFrame()

        # 1) Ask each disease module to log the QALYS
        disease_modules = self.sim.modules['HealthSystem'].registered_disease_modules

        for module in disease_modules.values():
            out = module.report_qaly_values()

            # Perform checks on what has been returned
            assert type(out) is pd.Series
#           assert len(out)==self.sim.population.props.is_alive.sum()
#           assert (~pd.isnull(out)).all()
            assert ((out>=0) & (out<=1)).all()
            assert self.sim.population.props.index.name==out.index.name
            assert self.sim.population.props.is_alive[out.index].all()

            # Add to dataframe
            # (each column of this dataframe gives the reports from each module the HealthState)
            disease_specific_qaly_values = pd.concat([disease_specific_qaly_values, out], axis=1)

        # 2) Reconcile the QALY scores into a unifed score
        df = self.sim.population.props
        df['qa_QALY'] = disease_specific_qaly_values.prod(axis=1)

        # 4) Summarise the results and output these to the log

        #   Sum of healthstates, by sex and age
        m_healthstate_sum = df[df.is_alive & (df.sex == 'M')].groupby('age_range')['qa_QALY'].sum()
        f_healthstate_sum = df[df.is_alive & (df.sex == 'F')].groupby('age_range')['qa_QALY'].sum()

        #       Men
        logger.info('%s|QALYS_M|%s', self.sim.date,
                    m_healthstate_sum.to_dict())

        #       Women
        logger.info('%s|QALYS_F|%s', self.sim.date,
                    f_healthstate_sum.to_dict())
