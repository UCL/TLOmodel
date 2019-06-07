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

        params['prob_cure_blood_transfusion'] = 0
        params['prob_cure_uterine_repair'] = 0
        params['prob_cure_hysterectomy'] = 0

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

        # First get and hold all the women who are experiencing an antepartum haemorrhage
        aph_women = df.index[df.is_alive & df.is_pregnant & df.la_aph]

        # Then we determine the cause of the bleed based on the incidence

        etiology = ['placenta previa', 'placental abruption', 'unknown']
        probabilities = [0.63, 0.30, 0.07]

        random_choice = self.sim.rng.choice(etiology, size=len(aph_women), p=probabilities)
        aph_df = pd.DataFrame(random_choice, index=aph_women)
        aph_df.columns = ['aph_status']

        # Get and hold the women experiencing APH by each cause

        previa_idx = aph_df.index[(aph_df.aph_status == 'placenta previa')]
        abruption_idx = aph_df.index[(aph_df.aph_status == 'placental abruption')]
        unk_idx = aph_df.index[(aph_df.aph_status == 'unknown')]

    # =========================== TREATMENT OF PLACENTA PREVIA ========================================================

        for individual_id in previa_idx:
            previa_idx = aph_df.index[(aph_df.aph_status == 'placenta previa')]

    # Primary treatment is delivery via caesarean section
    # Blood transfusion for blood loss

    # ========================= TREATMENT OF PLACENTAL ABRUPTION ======================================================

    # First we deal with the management of bleeding
    # Then we schedule safe delivery


    # todo: Should we have a uterine rupture event?

class PostpartumHaemorrhageTreatmentEvent(Event, IndividualScopeEventMixin):

    """handles the medical and surgical treatment of postpartum haemorrhage """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        # First get and hold all the women who are experiencing a post partum haemorrhage (excluding caesareans)

        aph_pp_women = df.index[df.is_alive & df.la_pph & (df.due_date == self.sim.date - DateOffset(days=2))]

        # Then we determine the cause of the bleed based on the incidence

        etiology = ['uterine atony', 'retained placenta', 'unknown'] # These values will need to be confirmed
        probabilities = [0.80, 0.10, 0.10]

        random_choice = self.sim.rng.choice(etiology, size=len(aph_pp_women), p=probabilities)
        pph_df = pd.DataFrame(random_choice, index=aph_pp_women)
        pph_df.columns = ['pph_status']

        # Get and hold the women experiencing PPH by each cause

        atony_idx = pph_df.index[(pph_df.pph_status == 'uterine atony')]
        retained_idx  = pph_df.index[(pph_df.pph_status == 'retained placenta')]
        unk_idx = pph_df.index[(pph_df.pph_status == 'unknown')]


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
