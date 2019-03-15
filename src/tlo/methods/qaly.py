"""
This Module runs the counting of QALYS across all persons and logs it
"""

import logging

import pandas as pd

from tlo import DateOffset, Module, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class QALY(Module):
    """
    This module holds all the stuff to do with QALYS
    """

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    # This is a DALY database, so complement of the values must be used.
    PARAMETERS = {
        'Weight_Database': Property(Types.DATA_FRAME, 'Weight Database')
    }

    PROPERTIES = {
        'qa_QALY': Property(Types.REAL, 'Summary of health state for previous year')
    }

    def read_parameters(self, data_folder):
        p = self.parameters
        p['Weight_Database'] = pd.read_csv(self.resourcefilepath + 'ResourceFile_DALY_Weights.csv')

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """
        pass

    def initialise_simulation(self, sim):
        """ Launch the QALY Logger to run every year
        """
        sim.schedule_event(LogQALYs(self), sim.date)

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        pass

    def get_qaly_weight(self, sequlae_code):
        """
        This can be used to look up the QALY code for a particular condition

        """
        w = self.parameters['Weight_Database']
        daly_wt = w.loc[w['TLO_Sequela_Code'] == sequlae_code, 'disability weight'].values[0]
        qaly_wt = 1 - daly_wt
        return qaly_wt


class LogQALYs(RegularEvent, PopulationScopeEventMixin):
    """A skeleton class for an event

    Regular events automatically reschedule themselves at a fixed frequency,
    and thus implement discrete timestep type behaviour. The frequency is
    specified when calling the base class constructor in our __init__ method.
    """

    def __init__(self, module):
        """
        This is the DALY Logger event. It runs every year, asks each disease module to report the DALY loading
        onto each individual over the pass year, reconciles this into a unified values, and outputs it to the
        log
        """
        super().__init__(module, frequency=DateOffset(months=12))

    def apply(self, population):
        """
        Running the QALY Logger
        """
        logger.debug('The QALY Logger is occuring now!@@@@ %s', self.sim.date)

        disease_specific_qaly_values = pd.DataFrame()

        # 1) Ask each disease module to log the QALYS
        disease_modules = self.sim.modules['HealthSystem'].registered_disease_modules

        for module in disease_modules.values():
            out = module.report_qaly_values()
            # each column of this dataframe gives the reports from each module the HealthState
            disease_specific_qaly_values = pd.concat([disease_specific_qaly_values, out], axis=1)

        # 2) Reconcile the QALY scores into a unifed score
        df = self.sim.population.props

        df['qa_QALY'] = 0  # setting to zero ensures that all dead people are given zero values
        df['qa_QALY'] = disease_specific_qaly_values.prod(axis=1)

        # 4) Summarise the results and output these to the log

        # men: sum of healthstates, by age
        m_healthstate_sum = df[df.is_alive & (df.sex == 'M')].groupby('age_range')['qa_QALY'].sum()
        f_healthstate_sum = df[df.is_alive & (df.sex == 'F')].groupby('age_range')['qa_QALY'].sum()

        logger.info('%s|QALYS_M|%s', self.sim.date,
                    m_healthstate_sum.to_dict())

        logger.info('%s|QALYS_F|%s', self.sim.date,
                    f_healthstate_sum.to_dict())
