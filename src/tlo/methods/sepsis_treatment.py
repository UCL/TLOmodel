"""
A skeleton template for disease methods.
"""

import logging

import pandas as pd

import numpy as np

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent

from tlo.methods import demography, eclampsia_treatment, labour


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SepsisTreatment(Module):
    """ This module manages the treatment of antenatal, intrapartum and postpartum sepsis
    """

    PARAMETERS = {
        'prob_cure_antibiotics': Parameter(
            Types.REAL, 'Probability of sepsis resolving following the administration of antibiotics'),
    }

    PROPERTIES = {
        'sep_treat_received': Property(Types.BOOL, 'dummy-has this woman received treatment')  # dummy property

    }

    def read_parameters(self, data_folder):
        params = self.parameters

        params['prob_cure_antibiotics'] = 0.5  # dummy

    def initialise_population(self, population):

        df = population.props  # a shortcut to the data-frame storing data for individuals
        m = self
        rng = m.rng

    def initialise_simulation(self, sim):

        event = SepsisTreatmentLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=0))

    def on_birth(self, mother_id, child_id):

       pass


class SepsisTreatmentEvent(Event, IndividualScopeEventMixin):

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        treatment_effect = params['prob_cure_antibiotics']
        random = self.sim.rng.random_sample()
        if treatment_effect > random:
            df.at[individual_id, 'la_sepsis'] = False
            df.at[individual_id, 'sep_treat_received'] = True

        elif treatment_effect < random:
            self.sim.schedule_event(labour.LabourDeathEvent(self.sim.modules['Labour'],
                                                            individual_id, cause='sepsis'),
                                    self.sim.date)


class PostPartumSepsisTreatmentEvent(Event, IndividualScopeEventMixin):

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        treatment_effect = params['prob_cure_antibiotics']
        random = self.sim.rng.random_sample()
        if treatment_effect > random:
            df.at[individual_id, 'la_sepsis'] = False
            df.at[individual_id, 'sep_treat_received'] = True

        elif treatment_effect < random:
            self.sim.schedule_event(labour.LabourDeathEvent(self.sim.modules['Labour'],
                                                            individual_id, cause='sepsis'),
                                    self.sim.date)


class SepsisTreatmentLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """Handles Eclampsia treatment logging"""
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
