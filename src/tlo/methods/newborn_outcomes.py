import logging

import pandas as pd

import numpy as np

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent

from tlo.methods import demography, labour, newborn_outcomes


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NewbornOutcomes(Module):
    """
   This module is responsible for the outcomes of newborns immediately following delivery
    """

    PARAMETERS = {
        'base_incidence_low_birth_weight': Parameter(
            Types.REAL, 'baseline incidence of low birth weight for newborns'),
        'prob_cba': Parameter(
            Types.REAL, 'baseline probability of a child being born with a congenital anomaly'),
        'prob_neonatal_sepsis': Parameter(
            Types.REAL, 'baseline probability of a child developing sepsis following birth'),
        'prob_neonatal_enceph': Parameter(
            Types.REAL, 'baseline probability of a child developing neonatal encephalopathy following delivery'),
        'prob_ptb_comps': Parameter(
            Types.REAL, 'baseline probability of a child developing complications associated with pre-term birth'),
        'prob_low_birth_weight': Parameter(
            Types.REAL, 'baseline probability of a child being born low birth weight'),
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
        # Should this maybe be categorical with all the potential complications

        'nb_low_birth_weight': Property(Types.BOOL, 'this child has been born weighing <= 2.5kg'),
        'nb_death_after_birth': Property(Types.BOOL, 'this child has died following complications after birth')
    }


    def read_parameters(self, data_folder):

        params = self.parameters

        params['base_incidence_low_birth_weight'] = 0.12 #dummy
        params['prob_cba'] = 0.1  # DUMMY
        params['prob_neonatal_sepsis'] = 0.15  # DUMMY
        params['prob_neonatal_enceph'] = 0.16  # DUMMY
        params['prob_ptb_comps'] = 0.15  # DUMMY
        params['cfr_cba'] = 0.2  # DUMMY
        params['cfr_neonatal_sepsis'] = 0.1  # DUMMY
        params['cfr_neonatal_enceph'] = 0.15  # DUMMY
        params['cfr_ptb_comps'] = 0.3  # DUMMY

        # Todo: Meet with TC to discuss as DALYS >150

        # if 'HealthBurden' in self.sim.modules.keys():
        #    params['daly_wt_mild_sneezing'] = self.sim.modules['HealthBurden'].get_daly_weight(50)
        #    params['daly_wt_coughing'] = self.sim.modules['HealthBurden'].get_daly_weight(50)
        #    params['daly_wt_advanced'] = self.sim.modules['HealthBurden'].get_daly_weight(589)

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
        df['nb_low_birth_weight'] = False
        df['nb_death_after_birth'] = False

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

    def initialise_simulation(self, sim):

        event = NewbornOutcomesLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=0))

    def on_birth(self, mother_id, child_id):

        df = self.sim.population.props
        mni = self.sim.modules['Labour'].mother_and_newborn_info

        df.at[child_id,'nb_early_preterm'] = False
        df.at[child_id, 'nb_late_preterm'] = False
        df.at[child_id, 'nb_congenital_anomaly'] = False
        df.at[child_id, 'nb_neonatal_sepsis'] = False
        df.at[child_id, 'nb_neonatal_enchep'] = False
        df.at[child_id, 'nb_ptb_comps'] = False
        df.at[child_id, 'nb_low_birth_weight'] = False
        df.at[child_id, 'nb_death_after_birth'] = False

        # Newborns delivered at less than 37 weeks are allocated as either late or early preterm based on the
        # gestation at labour
        if mni[mother_id]['labour_state'] == 'EPTL':
            df.at[child_id, 'nb_early_preterm'] = True

        elif mni[mother_id]['labour_state'] == 'LPTL':
            df.at[child_id, 'nb_late_preterm'] = True
        else:
            df.at[child_id, 'nb_early_preterm'] = False
            df.at[child_id, 'nb_late_preterm'] = False

        if df.at[child_id, 'is_alive'] & ~mni[mother_id]['stillbirth_in_labour']:
            self.sim.schedule_event(newborn_outcomes.NewbornOutcomeEvent(self.sim.modules['NewbornOutcomes'], child_id,
                                                            cause='newborn outcomes event'), self.sim.date)

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.info('This is NewbornOutcomes, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

#   def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

#        logger.debug('This is NewbornOutcomes reporting my health values')

#        df = self.sim.population.props  # shortcut to population properties dataframe
#        p = self.parameters

#        health_values = df.loc[df.is_alive, 'mi_specific_symptoms'].map({
#            'none': 0,
#            'mild sneezing': p['daly_wt_mild_sneezing'],
#            'coughing and irritable': p['daly_wt_coughing'],
#            'extreme emergency': p['daly_wt_advanced']
#        })
#        health_values.name = 'Mockitis Symptoms'    # label the cause of this disability

#        return health_values.loc[df.is_alive]   # returns the series


class NewbornOutcomeEvent(Event, IndividualScopeEventMixin):

    """ This event determines if , following delivery, a newborn has experienced any complications """
    def __init__(self, module, individual_id,cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self
        mni = self.sim.modules['Labour'].mother_and_newborn_info

    # ===================================== LOW BIRTH-WEIGHT STATUS ==================================================
        # todo: n.b. low birth weight and SGA may be handeled in conjuction with malnutrition?

        # Here we determine if a newly born neonate will be of low birth weight
        # Birth order, mother weight, mother height, mother education and family wealth - Guassian coefficients-
        # not risks.
        rf1 = 1
      #  rf2 = 1
        risk_factors = rf1#'*rf2
        eff_prob_lbw = risk_factors * params['base_incidence_low_birth_weight']
        random = self.sim.rng.random_sample(size=1)
        if random < eff_prob_lbw:
            df.at[individual_id, 'nb_low_birth_weight'] = True

    # ===================================== CONGENITAL ANOMALY ======================================================

        # ?? Will we determine congential anomaly antenatally (there will be a risk associated with still birth) so we
        # wouldnt need to apply a risk here because its predtermined


    # =================================== COMPLICATIONS IN PRETERM INFANTS ============================================

        # First we determine if newborns delivered preterm will develop any complications
        # Todo: Determine how likelihood of complications other than "preterm birth complications" are impacted by
        #  gestational age at delivery

        if df.at[individual_id, 'nb_early_preterm'] or df.at[individual_id,'nb_late_preterm']:

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

            # If this neonate has been delivered in a facility

        #   if mni[individual_id]['delivery_setting'] == 'FD': todo: how to link to mother ID
            event = HSI_NewbornOutcomes_ReceivesCareFollowingDelivery(self.module, person_id=individual_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                    priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1)
                                                                    )
            logger.info('This is NewbornOutcomesEvent scheduling HSI_NewbornOutcomes_ReceivesCareFollowingDeliveryd '
                        'for person %d', individual_id)

            # todo: will we consider care seeking for newborns from women who deliver in the community??

            self.sim.schedule_event(newborn_outcomes.NewbornDeathEvent(self.module, individual_id,
                                                                       cause='neonatal compilications')
                                                      ,self.sim.date)
            logger.info('This is NewbornOutcomesEvent scheduling NewbornDeathEvent for person %d', individual_id)



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
            logger.info('This is NewbornDeathEvent scheduling a death for person %d on date %s who died due to '
                        ' complications following birth', individual_id, self.sim.date)

            logger.info('%s|neonatal_death|%s', self.sim.date, #TODO: are we just recording all neonatal deaths?
                        {'age': df.at[individual_id, 'age_years'],
                         'person_id': individual_id})


        #TODO: just confirm that you want all the deaths to happen on the same of birth


# ================================ Health System Interaction Events ================================================

class HSI_NewbornOutcomes_ReceivesCareFollowingDelivery(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    This is event manages care received by newborns following a facility delivery, including referall for additional
     treatment in the case of an emergency
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['AccidentsandEmerg'] = 1  # Todo: review

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'NewbornOutcomes_ReceivesCareFollowingDelivery'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = self.sim.modules['HealthSystem'].get_blank_cons_footprint()
        self.ACCEPTED_FACILITY_LEVELS = [1,2,3]     # This enforces that the apppointment must be run at that facility-level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):


        logger.info('This is HSI_NewbornOutcomes_ReceivesCareFollowingDelivery,'
                     '  person %d is receiving care from a skilled birth attendant following their birth',
                     person_id)




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
