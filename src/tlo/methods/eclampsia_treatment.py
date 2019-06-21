import logging

import pandas as pd

import numpy as np

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent

from tlo.methods import demography, labour, caesarean_section

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class EclampsiaTreatment(Module):
    """
    This module overseas the treatment and prevention of intra and postpartum eclamptic seizures for
    women receiving skilled birth attendance
    """

    PARAMETERS = {

        'prob_cure_mgso4': Parameter(
            Types.REAL, 'relative risk of additional seizures following of administration of magnesium sulphate'),
        'prob_prevent_mgso4': Parameter(
            Types.REAL, 'relative risk of eclampsia following administration of magnesium sulphate in women '
                        'with severe preeclampsia'),
        'prob_cure_diazepam': Parameter(
            Types.REAL, 'relative risk of additional seizures following of administration of diazepam')

    }

    PROPERTIES = {

        'ect_treat_received': Property(Types.BOOL, 'dummy-has this woman received treatment')  # dummy property
    }

    def read_parameters(self, data_folder):

        params = self.parameters

        params['prob_cure_mgso4'] = 0.57  # probability taken from RR of 0.43for additional seizures (vs diazepam alone)
        params['prob_prevent_mgso4'] = 0.41  # Risk reduction of eclampsia in women who have pre-eclampsia
        params['prob_cure_diazepam'] = 0.8

    def initialise_population(self, population):

        df = population.props  # a shortcut to the data-frame storing data for individuals
        m = self
        rng = m.rng

        df['ect_treat_received'] = False

    def initialise_simulation(self, sim):

        event = EclampsiaTreatmentLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=0))

    def on_birth(self, mother_id, child_id):

        df = self.sim.population.props

        df.at[child_id, 'ect_treat_received'] = False


class EclampsiaPreventionEvent(Event, IndividualScopeEventMixin):
    """This event will handle the administration of prophylactic anti-convulsants to women with pre-eclampsia.
    """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        # TODO: Confirm if MgSO4 is regularly given prophylactically for women with severe pre-eclampsia

        # Could be a population level event to looks at all women with pre eclampsia who may interact with the health
        # system before delivery (or just a antepartum intervention- happens before complications)


class EclampsiaTreatmentEvent(Event, IndividualScopeEventMixin):
    """This event will handle the treatment cascade for women who experiences eclampsia during labour
    """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        # We first apply the probability that first line treatment (IV Magnesium Sulfate) will prevent additional
        # ecalamptic seizures
        treatment_effect = params['prob_cure_mgso4']
        random = self.sim.rng.random_sample()
        if treatment_effect > random:
            df.at[individual_id, 'la_eclampsia'] = False
            df.at[individual_id, 'ect_treat_received'] = True
        # Following successful treatment of seizures women will be scheduled to undergo assisted vaginal delivery
            # todo: Here we will schedule an assisted vaginal delivery

        # If first line treatment is unsuccessful, second line (IV Diazepam)is attempted
        elif treatment_effect < random:
            secondary_treatment = params['prob_cure_diazepam']
            random = self.sim.rng.random_sample()
            if secondary_treatment > random:
                df.at[individual_id, 'la_eclampsia'] = False
                df.at[individual_id, 'ect_treat_received'] = True
        # Following successful treatment of seizures women will be scheduled to undergo assisted vaginal delivery

        # Following unsuccessful treatment of seizures the woman will undergo an emergency caesarean section
            elif secondary_treatment < random:
                self.sim.schedule_event(
                    caesarean_section.EmergencyCaesareanSection(self.sim.modules['CaesareanSection'],
                                                                individual_id,
                                                                cause='emergency caesarean'), self.sim.date)


class EclampsiaTreatmentEventPostPartum(Event, IndividualScopeEventMixin):
    """This event will handle the treatment cascade for women who experiences eclampsia following labour
    """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        # Women who experience post partum eclampsia will undergo medical management of their seizures
        treatment_effect = params['prob_cure_mgso4']
        random = self.sim.rng.random_sample()
        if treatment_effect > random:
            df.at[individual_id, 'la_eclampsia'] = False
            df.at[individual_id, 'ect_treat_received'] = True
            #

        elif treatment_effect < random:
            secondary_treatment = params['prob_cure_diazepam']
            random = self.sim.rng.random_sample()
            if secondary_treatment > random:
                df.at[individual_id, 'la_eclampsia'] = False
                df.at[individual_id, 'ect_treat_received'] = True
            elif secondary_treatment < random:
                self.sim.schedule_event(labour.PostPartumDeathEvent(self.sim.modules['Labour'], individual_id,
                                                                 cause='postpartum'), self.sim.date)


class EclampsiaTreatmentLoggingEvent(RegularEvent, PopulationScopeEventMixin):
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
