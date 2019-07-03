import logging

import pandas as pd

import numpy as np

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent

from tlo.methods import demography, eclampsia_treatment, sepsis_treatment, labour, newborn_outcomes


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class NewbornOutcomes(Module):
    """
   This module is responsible for the outcomes of newborns immediately following delivery
    """

    PARAMETERS = {
        'prob_cba': Parameter(
            Types.REAL, 'baseline probability of a child being born with a congenital anomaly'),
        'prob_neonatal_sepsis': Parameter(
            Types.REAL, 'baseline probability of a child developing sepsis following birth'),
        'prob_neonatal_enceph': Parameter(
            Types.REAL, 'baseline probability of a child developing neonatal encephalopathy following delivery'),
        'prob_ptb_comps': Parameter(
            Types.REAL, 'baseline probability of a child developing complications associated with pre-term birth'),
        'cfr_cba': Parameter(
            Types.REAL, 'case fatality rate for a newborn with a congenital birth anomaly'),
        'cfr_neonatal_sepsis': Parameter(
            Types.REAL, 'case fatality rate for a newborn due to neonatal sepsis'),
        'cfr_neonatal_enceph': Parameter(
            Types.REAL, 'case fatality rate for a newborn due to neonatal encephalopathy'),
        'cfr_ptb_comps': Parameter(
            Types.REAL, 'case fatality rate for a newborn due to preterm birth complications'),
    }

    PROPERTIES = {
        'nb_early_preterm': Property(Types.BOOL, 'this child has been born early preterm (24-33 weeks gestation)'),
        'nb_late_preterm': Property(Types.BOOL, 'this child has been born late preterm (34-36 weeks gestation)'),
        'nb_congenital_anomaly': Property(Types.BOOL, 'this child has been born with a congenital anomaly'),
        'nb_neonatal_sepsis': Property(Types.BOOL, 'this child has developed neonatal sepsis following birth'),
        'nb_neonatal_enchep': Property(Types.BOOL, 'this child has developed neonatal encephalopathy secondary to '
                                                   'intrapartum related complications'),
        'nb_ptb_comps': Property(Types.BOOL, 'this child has developed complications associated with pre-term birth'),
        'nb_death_after_birth': Property(Types.BOOL, 'this child has died following complications after birth')
    }

    def read_parameters(self, data_folder):

        params = self.parameters

        params['prob_cba'] = 0.1  # DUMMY
        params['prob_neonatal_sepsis'] = 0.15  # DUMMY
        params['prob_neonatal_enceph'] = 0.16  # DUMMY
        params['prob_ptb_comps'] = 0.15  # DUMMY
        params['cfr_cba'] = 0.2  # DUMMY
        params['cfr_neonatal_sepsis'] = 0.1  # DUMMY
        params['cfr_neonatal_enceph'] = 0.15  # DUMMY
        params['cfr_ptb_comps'] = 0.3  # DUMMY

    def initialise_population(self, population):

        df = population.props  # a shortcut to the data-frame storing data for individuals
        m = self
        rng = m.rng

        df['nb_early_preterm'] = False
        df['nb_late_preterm'] = False
        df['nb_congenital_anomaly'] = False
        df['nb_neonatal_sepsis'] = False
        df['nb_neonatal_enchep'] = False
        df['nb_ptb_comps'] = False
        df['nb_death_after_birth'] = False

    def initialise_simulation(self, sim):

        event = NewbornOutcomesLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=0))

    def on_birth(self, mother_id, child_id):

        df = self.sim.population.props

        df.at[child_id, 'nb_congenital_anomaly'] = False
        df.at[child_id, 'nb_neonatal_sepsis'] = False
        df.at[child_id, 'nb_neonatal_enchep'] = False
        df.at[child_id, 'nb_ptb_comps'] = False
        df.at[child_id, 'nb_death_after_birth'] = False

        # Newborns delivered at less than 37 weeks are allocated as either late or early preterm based on the
        # gestation at labour
        if df.at[mother_id, 'la_labour'] == 'early_preterm_labour':
            df.at[child_id, 'nb_early_preterm'] = True
        elif df.at[mother_id, 'la_labour'] == 'late_preterm_labour':
            df.at[child_id, 'nb_late_preterm'] = True
        else:
            df.at[child_id, 'nb_preterm'] = False

        if df.at[child_id, 'is_alive'] & ~df.at[mother_id, 'la_still_birth_this_delivery']:
            self.sim.schedule_event(newborn_outcomes.NewbornOutcomeEvent(self.sim.modules['NewbornOutcomes'], child_id,
                                                            cause='newborn outcomes event'), self.sim.date)


class NewbornOutcomeEvent(Event, IndividualScopeEventMixin):

    """ This event determines if , following delivery, a newborn has experienced any complications """
    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

    # =================================== COMPLICATIONS IN PRETERM INFANTS ============================================

        # First we determine if newborns delivered preterm will develop any complications
        # Todo: Determine how likelihood of complications other than "preterm birth complications" are impacted by
        #  gestational age at delivery

        if df.at[individual_id, 'nb_early_preterm'] or df.at[individual_id,'nb_late_preterm']:
            rf1 = 1
            riskfactors = rf1
            eff_prob_cba = riskfactors * params['prob_cba']
            random = self.sim.rng.random_sample(size=1)
            if random < eff_prob_cba:
                df.at[individual_id, 'nb_congenital_anomaly'] = True

            rf1 = 1
            riskfactors = rf1
            eff_prob_sepsis = riskfactors * params['prob_neonatal_sepsis']
            random = self.sim.rng.random_sample(size=1)
            if random < eff_prob_sepsis:
                df.at[individual_id, 'nb_neonatal_sepsis'] = True

            rf1 = 1
            riskfactors = rf1
            eff_prob_enchep = riskfactors * params['prob_neonatal_enceph']
            random = self.sim.rng.random_sample(size=1)
            if random < eff_prob_enchep:
                df.at[individual_id, 'nb_neonatal_enchep'] = True

            rf1 = 1
            riskfactors = rf1
            eff_prob_ptbc = riskfactors * params['prob_ptb_comps']
            random = self.sim.rng.random_sample(size=1)
            if random < eff_prob_ptbc:
                df.at[individual_id, 'nb_ptb_comps'] = True

        if ~df.at[individual_id, 'nb_early_preterm'] & ~df.at[individual_id, 'nb_late_preterm']:
            rf1 = 1
            riskfactors = rf1
            eff_prob_cba = riskfactors * params['prob_cba']
            random = self.sim.rng.random_sample(size=1)
            if random < eff_prob_cba:
                df.at[individual_id, 'nb_congenital_anomaly'] = True

            rf1 = 1
            riskfactors = rf1
            eff_prob_sepsis = riskfactors * params['prob_neonatal_sepsis']
            random = self.sim.rng.random_sample(size=1)
            if random < eff_prob_sepsis:
                df.at[individual_id, 'nb_neonatal_sepsis'] = True

            rf1 = 1
            riskfactors = rf1
            eff_prob_enchep = riskfactors * params['prob_neonatal_enceph']
            random = self.sim.rng.random_sample(size=1)
            if random < eff_prob_enchep:
                df.at[individual_id, 'nb_neonatal_enchep'] = True

        # Currently we schedule all newborns who have experienced any of the above complications to go through the
        # NewbornDeathEvent
        if df.at[individual_id, 'nb_neonatal_enchep'] or df.at[individual_id, 'nb_neonatal_sepsis'] or \
            df.at[individual_id, 'nb_congenital_anomaly'] or df.at[individual_id, 'nb_ptb_comps']:
            self.sim.schedule_event(newborn_outcomes.NewbornDeathEvent(self.module, individual_id,
                                                                       cause='neonatal compilications')
                                                      ,self.sim.date)
            #TODO is hte passing of 'clause' requied here? is any other cause ever used? The event itself already seems "cause specific"?

class NewbornDeathEvent(Event, IndividualScopeEventMixin):

    """ This event determines if neonates who have experience complications will die because of them  """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        # Get and hold all newborns that have experienced complications following birth and apply case fatality rate

        if df.at[individual_id, 'nb_congenital_anomaly']:
            random = self.sim.rng.random_sample()
            if random > params['cfr_cba']:
                df.at[individual_id,'nb_death_after_birth'] =True

        if df.at[individual_id, 'nb_neonatal_sepsis']:
            random = self.sim.rng.random_sample()
            if random > params['cfr_neonatal_sepsis']:
                df.at[individual_id, 'nb_death_after_birth'] = True

        if df.at[individual_id, 'nb_neonatal_enchep']:
            random = self.sim.rng.random_sample()
            if random > params['cfr_neonatal_enceph']:
                df.at[individual_id, 'nb_death_after_birth'] = True

        if df.at[individual_id, 'nb_ptb_comps']:
            random = self.sim.rng.random_sample()
            if random > params['cfr_ptb_comps']:
                df.at[individual_id, 'nb_death_after_birth'] = True

        # Schedule the death of any newborns who have died from their complications, whilst cause of death is recorded
        # as "neonatal complications" we will have contributory factors recorded as the properties of newborns

        if df.at[individual_id, 'nb_death_after_birth']:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                  cause="neonatal complications"), self.sim.date)

        #TODO: just confirm that you want all the deaths to happen on the same of birth

class NewbornOutcomesLoggingEvent(RegularEvent, PopulationScopeEventMixin):
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
