"""
A skeleton template for disease methods.
"""

import logging

import pandas as pd

import numpy as np

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent

from tlo.methods import demography, labour, caesarean_section

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class UterineRuptureTreatment(Module):
    """
    This module manages the surgical treatment of uterine rupture
    """

    PARAMETERS = {
        'prob_cure_blood_transfusion': Parameter(
            Types.REAL, '...'),
        'prob_cure_uterine_repair': Parameter(
            Types.REAL, '...'),
        'prob_cure_hysterectomy': Parameter(
            Types.REAL, '...'),
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

        params['prob_cure_uterine_repair'] = 1
        params['prob_cure_hysterectomy'] = 1

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        df = population.props
        m = self
        rng = m.rng
        params = self.parameters

    def initialise_simulation(self, sim):

        event = UterineRuptureTreatmentLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=0))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        pass


class UterineRuptureTreatmentEvent(Event, IndividualScopeEventMixin):
    """handles the medical and surgical treatment of postpartum haemorrhage
    """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):

        df = self.sim.population.props
        params = self.module.parameters
        m = self

        # First we deliver the baby by caesarean (but without moving to CS event)
        df.at[individual_id, 'la_delivery_mode'] = 'EmCS'
        logger.debug('@@@@ A Delivery is now occuring via emergency caesarean section, to mother %s', individual_id)
        df.at[individual_id, 'la_previous_cs'] = +1

        # Todo: consider how to incorporate the impact of bleeding
        # Next we determine if the uterus can be repaired surgically
        random = self.sim.rng.random_sample()
        if params['prob_cure_uterine_repair'] > random:
            df.at[individual_id, 'la_uterine_rupture'] = False

        # In the instance of failed surgical repair, the woman undergoes a hysterectomy
        else:
            random = self.sim.rng.random_sample()
            if params['prob_cure_hysterectomy'] > random:
                df.at[individual_id, 'la_uterine_rupture'] = False

        # Todo: Should the woman move to post CS event if the UR is not repaired or UR death event?
#        if df.at[individual_id,'la_uterine_rupture']:


class UterineRuptureTreatmentLoggingEvent(RegularEvent, PopulationScopeEventMixin):
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
