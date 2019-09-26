"""
Module responsible for supervision of pregnancy in the population including miscarriage, abortion, onset of labour and
onset of antenatal complications .
"""

import logging

import numpy as np
import pandas as pd
from pathlib import Path

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PregnancySupervisor(Module):
    """
    This module is responsible for supervision of pregnancy in the population including miscarriage, abortion, onset of labour and
    onset of antenatal complications """
    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    PARAMETERS = {

    }

    PROPERTIES = {
        'ps_gestational_age': Property(Types.INT, 'current gestational age of this womans pregnancy in weeks'),
    }

    TREATMENT_ID = ''

    def read_parameters(self, data_folder):
        """
        """
        params = self.parameters

        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_PregnancySupervisor.xlsx',
                            sheet_name='parameter_values')
        dfd.set_index('parameter_name', inplace=True)


#        if 'HealthBurden' in self.sim.modules.keys():
#            params['daly_wt_haemorrhage_moderate'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=339)

    def initialise_population(self, population):

        df = population.props

        df.loc[df.sex == 'F', 'ps_gestational_age'] = 0


    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        """
        event = PregnancySupervisorEvent
        sim.schedule_event(event(self),
                           sim.date + DateOffset(days=0))

        event = PregnancySupervisorLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=0))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        df = self.sim.population.props

        if df.at[child_id, 'sex'] == 'F':
            df.at[child_id, 'ps_gestational_age'] = 0

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug('This is AntenatalCare, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

#    def report_daly_values(self):

    #    logger.debug('This is mockitis reporting my health values')
    #    df = self.sim.population.props  # shortcut to population properties dataframe
    #    p = self.parameters


class PregnancySupervisorEvent(RegularEvent, PopulationScopeEventMixin):
    """
    This event updates the ac_gestational_age for the pregnant population based on the current simulation date
    """

    def __init__(self, module,):
        super().__init__(module, frequency=DateOffset(weeks=1))

    def apply(self, population):
        df = population.props

        gestation_in_days = self.sim.date - df.loc[df.is_pregnant, 'date_of_last_pregnancy']
        gestation_in_weeks = gestation_in_days / np.timedelta64(1, 'W')

        df.loc[df.is_pregnant, 'ps_gestational_age'] = gestation_in_weeks.astype(int)



class PregnancySupervisorLoggingEvent(RegularEvent, PopulationScopeEventMixin):
        """Handles Pregnancy Supervision  logging"""

        def __init__(self, module):
            """schedule logging to repeat every 3 months
            """
            #    self.repeat = 3
            #    super().__init__(module, frequency=DateOffset(days=self.repeat))
            super().__init__(module, frequency=DateOffset(months=3))

        def apply(self, population):
            """Apply this event to the population.
            :param population: the current population
            """
            df = population.props
