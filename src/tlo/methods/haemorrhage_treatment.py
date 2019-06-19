"""
A skeleton template for disease methods.
"""

import logging

import pandas as pd

import numpy as np

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent

from tlo.methods import demography, labour

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class HaemorrhageTreatment(Module):
    """
    This module manages the medical and surgical treatment of maternal haemorrhage including antepartum haemorrhage
    (of all common etiologies) and primary and secondary post-partum haemorrhage
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

        params['prob_cure_blood_transfusion'] = 0.4  # dummy
        params['prob_cure_hysterectomy'] = 0.9  # dummy

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

        event = HaemorrhageTreatmentLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=0))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        pass


class AntepartumHaemorrhageTreatmentEvent(Event, IndividualScopeEventMixin):
    """handles the medical and surgical treatment of postpartum haemorrhage
    """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):

        df = self.sim.population.props
        params = self.module.parameters
        m = self

        # We determine the cause of the bleed based on the incidence
        etiology = ['placenta praevia', 'placental abruption']
        probabilities = [0.67, 0.33]
        random_choice = self.sim.rng.choice(etiology, size=1, p=probabilities)

# =========================== TREATMENT OF PLACENTA PRAEVIA ========================================================
        if random_choice == 'placenta praevia':

            women = df.index[df.is_alive] #dummy
        # Blood transfusion for blood loss
        # Maybe we apply curative effective of blood replacement and if that fails go to death event?
        #
        # Primary treatment is delivery via caesarean section
        else:
            women = df.index[df.is_alive] #dummy
        # First we deal with the management of bleeding
        # Then we schedule safe delivery

# ========================= TREATMENT OF PLACENTAL ABRUPTION ======================================================


class PostpartumHaemorrhageTreatmentEvent(Event, IndividualScopeEventMixin):

    """handles the medical and surgical treatment of postpartum haemorrhage """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self


class HaemorrhageTreatmentLoggingEvent(RegularEvent, PopulationScopeEventMixin):
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
