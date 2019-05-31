import logging

import pandas as pd

import numpy as np

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent

from tlo.methods import demography, Labour


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class EclampsiaTreatmentSBA(Module):
    """
    This module overseas the treatment and prevention of intra and postpartum eclamptic seizures for
    women recieving skilled birth attendance
    """

    PARAMETERS = {

        'prob_cure_mgso4': Parameter(
            Types.REAL, 'probability of administration of magnesium sulphate preventing additonal eclamptic seizures'),
        'prob_prevent_mgso4': Parameter(
            Types.REAL, 'probability of administration of magnesium sulphate preventing eclampsia in severe '
                        'preeclampsia'),
        'prob_cure_diazepam': Parameter(
            Types.REAL, 'probability of administration of diazepam preventing additional eclamptic seizures'),
        'prob_cure_thiopental': Parameter(
            Types.REAL, 'probability of administration of thiopental preventing additional eclamptic seizures'),

    }

    PROPERTIES = {

        'ect_treat_received': Property(Types.BOOL, 'dummy-has this woman received treatment'),

       # temp presently
    }

    def read_parameters(self, data_folder):

        params = self.parameters

        params['prob_cure_mgso4'] = 0.43  # Risk reduction for additional seizures (vs diazepam alone)
        params['prob_prevent_mgso4'] = 0.41  # Risk reduction of eclampsia in women who have pre-eclampsia
        params['prob_cure_diazepam'] = 0.2  # Dummy - to include if no effectiveness data?
        params['prob_cure_thiopental'] = 0.2  # Dummy - to include if no effectiveness data?

    def initialise_population(self, population):

        df = population.props  # a shortcut to the data-frame storing data for individuals
        m = self
        rng = m.rng

        df['ect_treat_received'] = False

    def initialise_simulation(self, sim):

        event = EclampsiaTreatmentLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=0))

        self.baseline_labour_scheduler(sim.population)

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

        # Todo: clarify with Asif if there is a better way of doing this in individual events

        # 1.)Get and hold all women who are have an eclamptic fit in labour

        receiving_treatment_idx = df.index[(df.la_eclampsia == True) & (df.due_date == self.sim.date)]

        # 2.)Apply the probability that first line treatment will stop/prevenet seizures

        # treatment_effect = pd.Series(params['prob_cure_mgso4'], index=receiving_treatment_idx)

        treatment_effect = pd.Series(0.4, index=receiving_treatment_idx)

        random_draw = pd.Series(self.sim.rng.random_sample(size=len(receiving_treatment_idx)),
                                index=df.index[(df.la_eclampsia == True) & (df.due_date == self.sim.date)])

        dfx = pd.concat([treatment_effect, random_draw], axis=1)
        dfx.columns = ['treatment_effect', 'random_draw']
        successful_treatment = dfx.index[dfx.treatment_effect > dfx.random_draw]
        unsuccessful_treatment = dfx.index[dfx.treatment_effect < dfx.random_draw]

        # 2.) For those where eclampsia is stopped reset property to false

        df.loc[successful_treatment, 'la_eclampsia'] = False
        df.loc[successful_treatment, 'ect_treat_received'] = True

        # 3.) Get and hold all women whose seizures have stopped and schedule an assisted vaginal delivery as per
        # guidelines

        for individual_id in successful_treatment:
            women=df.index[df.is_pregnant] # placeholder

            # HERE SCHEDULE ASSISTED VAGINAL DELIVERY

        # 4.) Get and hold all women whose seizures havent stopped and apply probability of second line treatment
        # stopping seizures

        second_treatment_effect = pd.Series(params['prob_cure_diazepam'], index=unsuccessful_treatment)

        random_draw = pd.Series(self.sim.rng.random_sample(size=len(unsuccessful_treatment)),
                                index=dfx.index[dfx.treatment_effect < dfx.random_draw])

        dfx = pd.concat([second_treatment_effect, random_draw], axis=1)
        dfx.columns = ['second_treatment_effect', 'random_draw']
        successful_treatment_secondary = dfx.index[dfx.treatment_effect > dfx.random_draw]
        unsuccessful_treatment_secondary = dfx.index[dfx.treatment_effect < dfx.random_draw]

        df.loc[successful_treatment_secondary, 'la_eclampsia'] = False
        df.loc[successful_treatment_secondary, 'ect_treat_received'] = True

        # 5.) If suitable and seizures controlled schedule assisted vaginal delivery

        for individual_id in successful_treatment_secondary:
            women=df.index[df.is_pregnant] # placeholder
            # SCHEDULE AVD

        # 5.) If seizures not controlled schedule emergency caesarean section

        for individual_id in unsuccessful_treatment_secondary:
            women = df.index[df.is_pregnant]  # placeholder
            # SCHEDULE CS

        # Todo: Consider where death event will be scheduled?


class EclampsiaTreatmentEventPostPartum(Event, IndividualScopeEventMixin):
    """This event will handle the treatment cascade for women who experiences eclampsia following labour
    """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self


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
