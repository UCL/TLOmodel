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


class HaemorrhageTreatment(Module):
    """
    This module manages the medical and surgical treatment of maternal haemorrhage including antepartum haemorrhage
    (of all common etiologies) and primary and secondary post-partum haemorrhage
    """

    PARAMETERS = {
        'prob_cure_blood_transfusion': Parameter(
            Types.REAL, '...'),
        'prob_cure_oxytocin': Parameter(
            Types.REAL, 'probability of intravenous oxytocin arresting post-partum haemorrhage'),
        'prob_cure_misoprostol': Parameter(
            Types.REAL, 'probability of rectal misoprostol arresting post-partum haemorrhage'),
        'prob_cure_uterine_massage': Parameter(
            Types.REAL, 'probability of uterine massage arresting post-partum haemorrhage'),
        'prob_cure_uterine_tamponade': Parameter(
            Types.REAL, 'probability of uterine tamponade arresting post-partum haemorrhage'),
        'prob_cure_uterine_ligation': Parameter(
            Types.REAL, 'probability of laparotomy and uterine ligation arresting post-partum haemorrhage'),
        'prob_cure_b_lych': Parameter(
            Types.REAL, 'probability of laparotomy and B-lynch sutures arresting post-partum haemorrhage'),
        'prob_cure_hysterectomy': Parameter(
            Types.REAL, 'probability of total hysterectomy arresting post-partum haemorrhage'),
        'prob_cure_manual_removal': Parameter(
            Types.REAL, 'probability of manual removal of retained products arresting a post partum haemorrhage'),

    }

    PROPERTIES = {

        'hm_pph_treat_received': Property(Types.BOOL, 'dummy-has this woman received treatment'),  # dummy property
        'hm_aph_treat_received': Property(Types.BOOL, 'dummy-has this woman received treatment')  # dummy property

    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we do nothing.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        params = self.parameters

        params['prob_cure_blood_transfusion'] = 0.4  # dummy
        params['prob_cure_oxytocin'] = 0.5  # dummy
        params['prob_cure_misoprostol'] = 0.3  # dummy
        params['prob_cure_uterine_massage'] = 0.15  # dummy
        params['prob_cure_uterine_tamponade'] = 0.6  # dummy
        params['prob_cure_uterine_ligation'] = 0.8  # dummy
        params['prob_cure_b_lych'] = 0.8  # dummy
        params['prob_cure_hysterectomy'] = 0.95  # dummy
        params['prob_cure_manual_removal'] = 0.75  # dummy

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

        df['hm_pph_treat_received'] = False
        df['hm_aph_treat_received'] = False

        # TODO: Not clear if important to save this information to the population props dataframe? (is it used?)

    def initialise_simulation(self, sim):

        event = HaemorrhageTreatmentLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=0))

        #TODO: maybe don't need a regular logging event for the provision of a treatment?

    def on_birth(self, mother_id, child_id):

        df = self.sim.population.props

        df.at[child_id, 'hm_pph_treat_received'] = False


class AntepartumHaemorrhageTreatmentEvent(Event, IndividualScopeEventMixin):
    """handles the medical and surgical treatment of postpartum haemorrhage
    """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):

        df = self.sim.population.props
        params = self.module.parameters
        m = self

        # Currently assuming the cause of APH in labour to be due to either placenta praevia or placental abruption the
        # primary treatment (as per malawian guidelines) is blood replacement and caesarean delivery

        # Todo: How do we apply the impact of blood as a treatment for hemorrhage
        # Todo: need to consider the impact of etiology on bleeding within CS? and potential additional interventions?
        #  may be too granular

        # Women who experience APH are therefore shceduled for an emergency Caesarean section
        # Todo: A/w linking with health system
        df.at[individual_id, 'hm_aph_treat_received'] = True
        self.sim.schedule_event(caesarean_section.EmergencyCaesareanSection(self.sim.modules['CaesareanSection'],
                                                                                individual_id,
                                                                            cause='emergency caesarean'),self.sim.date)


class PostpartumHaemorrhageTreatmentEvent(Event, IndividualScopeEventMixin):

    #TODO: THIS IS NOT BEING USED???


    """handles the medical and surgical treatment of postpartum haemorrhage """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        # First we use a probability weighted random draw to determine the underlying etiology of this womans PPH
        etiology = ['uterine atony', 'retained products']
        probabilities = [0.67, 0.33]# dummy
        random_choice = self.sim.rng.choice(etiology, size=1, p=probabilities)
        # Todo: also consider here if we should add a level of severity of PPH

# ================================= TREATMENT CASCADE FOR ATONIC UTERUS ==============================================

        # Here we use a treatment cascade adapted from Malawian Obs/Gynae guidlines
        # Todo: refine/confirm structure and flow of cascade with Expert

        # Women who are bleeding due to atonic uterus first undergo medical management, oxytocin IV, misoprostol PR
        # and uterine massage in an attempt to stop bleeding
        if random_choice == 'uterine atony':
            random = self.sim.rng.random_sample()
            if params['prob_cure_oxytocin'] > random:
                df.at[individual_id, 'la_pph'] = False
                df.at[individual_id, 'hm_pph_treat_received'] = True
            else:
                random = self.sim.rng.random_sample()
                if params['prob_cure_misoprostol'] > random:
                    df.at[individual_id, 'la_pph'] = False
                    df.at[individual_id, 'hm_pph_treat_received'] = True
                else:
                    random = self.sim.rng.random_sample()
                    if params['prob_cure_uterine_massage'] > random:
                        df.at[individual_id, 'la_pph'] = False
                        df.at[individual_id, 'hm_pph_treat_received'] = True
                        # Todo: consider the impact of oxy + miso + massage as ONE value, Discuss with expert

        # In bleeding refractory to medical management women then undergo surgical management of bleed
                    else:
                        random = self.sim.rng.random_sample()
                        if params['prob_cure_uterine_ligation'] > random:
                            df.at[individual_id, 'la_pph'] = False
                            df.at[individual_id, 'hm_pph_treat_received'] = True
                        else:
                            random = self.sim.rng.random_sample()
                            if params['prob_cure_b_lych'] > random:
                                df.at[individual_id, 'la_pph'] = False
                                df.at[individual_id, 'hm_pph_treat_received'] = True
                            else:
                                random = self.sim.rng.random_sample()
                                # Todo: similarly consider bunching surgical interventions
                                if params['prob_cure_hysterectomy'] > random:
                                    df.at[individual_id, 'la_pph'] = False
                                    df.at[individual_id, 'hm_pph_treat_received'] = True

        # TODO: consider where to house property recording infertility secondary to hysterectomy
        # TODO: can we put a stop/break in the cascade dependent on availbility of consumables/staff time
        # TODO: Again how to apply the effect of blood?

# ================================= TREATMENT CASCADE FOR RETAINED PRODUCTS/PLACENTA ==================================

        # If a woman is bleeding due to retained products of conception treatment is applied here
        if random_choice == 'retained products':
            random = self.sim.rng.random_sample()
            if params['prob_cure_manual_removal'] > random:
                df.at[individual_id, 'la_pph'] = False

        if df.at[individual_id,'la_pph']:
            self.sim.schedule_event(labour.PostPartumDeathEvent(self.sim.modules['Labour'], individual_id,
                                                                cause='post partum haemorrhage'), self.sim.date)


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
