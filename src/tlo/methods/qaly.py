"""
This Module runs the counting of QALYS across all persons and logs it
"""

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent
import pandas as pd
import logging

from tlo.methods import healthsystem

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class QALY(Module):
    """
    This module holds all the stuff to do with QALYS
    """

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath=resourcefilepath


    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    PARAMETERS = {'Weight_Database': Property(Types.DATA_FRAME,'Weight Database')}

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {'qa_Overall_Health_State': Property(Types.REAL, 'Summary of health state for previous year')}


    def read_parameters(self, data_folder):

        self.parameters['Weight_Database'] = pd.read_csv(self.resourcefilepath+'ResourceFile_DALY_Weights.csv')

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """
        pass

    def initialise_simulation(self, sim):

        """
        Launch the QALY Logger to run every year
        """

        sim.schedule_event(LogQALYs(self),sim.date)


    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        pass


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
        print('The QALY Logger is occuring now!@@@@',self.sim.date)

        disease_specific_health_states = pd.DataFrame()

        # 1) Ask each disease module to log the QALYS
        disease_modules=self.sim.modules['HealthSystem'].RegisteredDiseaseModules

        for module in disease_modules.values():
            out=module.report_HealthValues()
            disease_specific_health_states = pd.concat([disease_specific_health_states , out],
                                            axis=1)  # each column of this dataframe gives the reports from each module the HealthState


        # 2) Reconcile the QALY scores into a unifed score
        df=self.sim.population.props

        df['qa_Overall_Health_State']=disease_specific_health_states.prod(axis=1)


        # 4) Summarise the results and output these to the log

        # men: sum of healthstates, by age
        m_healthstate_sum = df[df.is_alive & (df.sex == 'M')].groupby('age_range')['qa_Overall_Health_State'].sum()
        f_healthstate_sum = df[df.is_alive & (df.sex == 'F')].groupby('age_range')['qa_Overall_Health_State'].sum()

        logger.info('%s|QALYS_M|%s', self.sim.date,
                    m_healthstate_sum.to_dict())

        logger.info('%s|QALYS_F|%s', self.sim.date,
                    f_healthstate_sum.to_dict())


