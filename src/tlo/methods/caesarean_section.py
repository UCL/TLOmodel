
import logging

import pandas as pd

import numpy as np

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent

from tlo.methods import demography, labour, eclampsia_treatment


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CaesareanSection(Module):
    """ This module manages both emergency and planned deliveries via caesarean section
    """

    PARAMETERS = {
        'prob_pph_cs': Parameter(
            Types.REAL, 'probability of a postpartum haemorrhage following caesarean section'),
        'prob_sepsis_cs': Parameter(
            Types.REAL, 'probability of maternal sepsis following caesarean section'),
        'prob_eclampsia_cs': Parameter(
            Types.REAL, 'probability of eclampsia following caesarean section'),
        'effectiveness_amtsl': Parameter(
            Types.REAL, 'effectiveness of active management of the third stage of labour during caesarean section at '
                        'preventing post partum haemorrhage'),
        'effectiveness_abx': Parameter(
            Types.REAL, 'effectiveness of prophylactic antibiotics in preventing post caesarean infection'),

    }

    PROPERTIES = {

    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we do nothing.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        params = self.parameters

        params['prob_pph_cs'] = 0.083  # Calculated from DHS 2010
        params['prob_sepsis_cs'] = 0.083  # Calculated from DHS 2010
       # params['prob_eclampsia_cs'] = 0.083  # Calculated from DHS 2010
        params['effectiveness_amtsl'] = 0.083  # Calculated from DHS 2010
        params['effectiveness_abx'] = 0.083  # Calculated from DHS 2010

    def initialise_population(self, population):

       pass

    def initialise_simulation(self, sim):

        event = CaesareanLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=0))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        pass


class ElectiveCaesareanSection(Event, IndividualScopeEventMixin):

    """Event handling deliveries for women requiring an emergency caesarean section for any indication
    """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        # First we register that this birth is occurring due to an elective caesarean
        df.at[individual_id, 'la_delivery_mode'] = 'ElCS'  # this property may not be needed
        logger.debug('@@@@ A Delivery is now occuring via elective caesarean section, to mother %s', individual_id)
        df.at[individual_id, 'la_previous_cs'] = +1

        self.sim.schedule_event(PostCaesareanSection(self.module, individual_id, cause='post caesarean'),
                            self.sim.date)


class EmergencyCaesareanSection(Event, IndividualScopeEventMixin):

    """Event handling deliveries for women requiring an emergency caesarean section for any indication
    """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        # First we register that this birth is occurring due to an emergency caesarean
        df.at[individual_id, 'la_delivery_mode'] = 'EmCS'         # this property may not be needed
        logger.debug('@@@@ A Delivery is now occuring via emergency caesarean section, to mother %s', individual_id)
        df.at[individual_id, 'la_previous_cs'] = +1

        # Next we deal with the indications for the EmCS which we switch to false
        if df.at[individual_id, 'la_obstructed_labour']:
            df.at[individual_id, 'la_obstructed_labour'] = False  # Labour cannot be obstructed if the baby is delivered

        if df.at[individual_id, 'la_aph']:
            df.at[individual_id, 'la_aph'] = False  # Placental bleeding cannot continue now placenta is delivered
            # todo: aph is a risk factor for PPH, how to convey this

        if df.at[individual_id, 'la_eclampsia']:
            df.at[individual_id, 'la_eclampsia'] = False  # Assume eclampsia has stopped as placenta is delivered
        # todo: for uterine rupture --> cs --> repair (fails) --> hysterctomy (so they need to pass back to UR event0
        # todo: if treatment is switch to false they still need to go through the death event? or do they?

        # We then schedule the postpartum caesarean event
        self.sim.schedule_event(PostCaesareanSection(self.module, individual_id, cause='post caesarean'),
                                self.sim.date)


class PostCaesareanSection(Event, IndividualScopeEventMixin):

    """Event handling deliveries for women requiring an emergency caesarean section for any indication
    """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        # Todo: decide how to code in effect of upstream interventions that effect sepsis/PPH
        # Todo: consider property 'cs_indication' which we could use to link effects indication to pp outcomes

        # Risk factors?
        if df.at[individual_id, 'la_delivery_mode'] == 'EmCS':
            eff_prob_pph = params['prob_pph_cs']
            random = self.sim.rng.random_sample(size=1)
            if random < eff_prob_pph:
                df.at[individual_id, 'la_pph'] = True

        # Risk factors?
            eff_prob_pph = params['prob_sepsis_cs']
            random = self.sim.rng.random_sample(size=1)
            if random < eff_prob_pph:
                df.at[individual_id, 'la_sepsis'] = True

        # todo: difference in incidence of outcomes for elective vs emergency (i think this will be more important
        # for neonates)


class CaesareanLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """Handles lifestyle logging"""
    def __init__(self, module):
        """schedule logging to repeat every 3 months
        """
        self.repeat = 3
        super().__init__(module, frequency=DateOffset(days=self.repeat))

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props

        logger.debug('%s|person_one|%s',
                          self.sim.date, df.loc[0].to_dict())
