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
            Types.REAL, 'baseline incidence of low birth weight for neonates'),
        'base_incidence_sga': Parameter(
            Types.REAL, 'baseline incidence of small for gestational age for neonates'),
        'prob_cba': Parameter(
            Types.REAL, 'baseline probability of a neonate being born with a congenital anomaly'),
        'prob_early_onset_neonatal_sepsis': Parameter(
            Types.REAL, 'baseline probability of a neonate developing sepsis following birth'),
        'prob_birth_asphyxia_iprc': Parameter(
            Types.REAL, 'baseline probability of a neonate devloping intrapartum related complications '
                        '(previously birth asphyxia) following delivery '),
        'prob_ivh_preterm': Parameter(
            Types.REAL, 'baseline probability of a preterm neonate developing an intravascular haemorrhage as a result '
                        'of prematurity '),
        'prob_nec_preterm': Parameter(
            Types.REAL,
            'baseline probability of a preterm neonate developing necrotising enterocolitis as a result of prematurity '),
        'prob_nrds_preterm': Parameter(
            Types.REAL,
            'baseline probability of a preterm neonate developing newborn respiratory distress syndrome as a result of '
            'prematurity '),
        'prob_low_birth_weight': Parameter(
            Types.REAL, 'baseline probability of a neonate being born low birth weight'),
        'prob_early_breastfeeding_hf': Parameter(
            Types.REAL, 'probability that a neonate will be breastfed within the first hour following birth when '
                        'delivered at a health facility'),
        'prob_early_breastfeeding_hb': Parameter(
            Types.REAL, 'probability that a neonate will be breastfed within the first hour following birth when '
                        'delivered at home'),
        'cfr_cba': Parameter(
            Types.REAL, 'case fatality rate for a neonate with a congenital birth anomaly'),
        'cfr_neonatal_sepsis': Parameter(
            Types.REAL, 'case fatality rate for a neonate due to neonatal sepsis'),
        'cfr_neonatal_enceph': Parameter(
            Types.REAL, 'case fatality rate for a neonate due to neonatal encephalopathy'),
        'cfr_ptb_comps': Parameter(
            Types.REAL, 'case fatality rate for a neonate due to preterm birth complications'),
    }

    PROPERTIES = {
        'nb_early_preterm': Property(Types.BOOL, 'whether this neonate has been born early preterm (24-33 weeks '
                                                 'gestation)'),
        'nb_late_preterm': Property(Types.BOOL, 'whether this neonate has been born late preterm (34-36 weeks '
                                                'gestation)'),
        'nb_congenital_anomaly': Property(Types.BOOL, 'whether this neonate has been born with a congenital anomaly'),
        'nb_early_onset_neonatal_sepsis': Property(Types.BOOL, 'whethert his neonate has developed neonatal sepsis'
                                                               ' following birth'),
        'nb_birth_asphyxia': Property(Types.BOOL, 'whether this neonate has been born asphyxiated and apneic due to '
                                                  'intrapartum related complications'),
        'nb_hypoxic_ischemic_enceph': Property(Types.BOOL, 'whether a perinatally asphyixiated neonate has developed '
                                                           'hypoxic ischemic encephalopathy'),
        'nb_intravascular_haem': Property(Types.BOOL, 'whether this neonate has developed an intravascular haemorrhage '
                                                      'following preterm birth'),
        'nb_necrotising_entero': Property(Types.BOOL, 'whether this neonate has developed necrotising enterocolitis '
                                                      'following preterm birth'),
        'nb_resp_distress_synd': Property(Types.BOOL, 'whether this neonate has developed newborn respiritory distress '
                                                      'syndrome following preterm birth '),
        'nb_birth_weight': Property(Types.CATEGORICAL,'extremely low birth weight (<1000g), '
                                                                                  'very low birth weight (<1500g), '
                                                                                  'low birth weight (<2500g),'
                                                      ' normal birth weight (>2500g)',
                                categories=['ext_LBW', 'very_LBW', 'LBW', 'NBW']),
        'nb_size_for_gestational_age': Property(Types.CATEGORICAL, 'small for gestational age, average for gestational'
                                                                   ' age, large for gestational age',
                                                categories=['SGA', 'AGA', 'LGA']),
        'nb_early_breastfeeding': Property(Types.BOOL, 'whether this neonate is exclusively breastfed after birth'),
        'nb_death_after_birth': Property(Types.BOOL, 'whether this child has died following complications after birth'),

         }

    def read_parameters(self, data_folder):

        params = self.parameters

        params['base_incidence_low_birth_weight'] = 0.12 #dummy (DHS prevelance 12%)
        params['base_incidence_sga'] = 0.12 #dummy
        params['prob_cba'] = 0.1  # DUMMY
        params['prob_early_onset_neonatal_sepsis'] = 0.15  # DUMMY
        params['prob_birth_asphyxia_iprc'] = 0.16  # DUMMY
        params['prob_ivh_preterm'] = 0.1
        params['prob_nec_preterm'] = 0.1
        params['prob_nrds_preterm'] = 0.1
        params['prob_early_breastfeeding_hb'] = 0.67 #DHS 2015
        params['prob_early_breastfeeding_hf'] = 0.77 #DHS 2015
        params['prob_successful_resuscitation'] = 0.6 # DUMMY
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
        df['nb_early_onset_neonatal_sepsis'] = False
        df['nb_birth_asphyxia'] = False
        df['nb_hypoxic_ischemic_enceph'] = False
        df['nb_intravascular_haem'] = False
        df['nb_necrotising_entero'] = False
        df['nb_resp_distress_synd'] = False
        df['nb_birth_weight'] = None
        df['nb_size_for_gestational_age'] = None
        df['nb_early_breastfeeding'] = False
        df['nb_death_after_birth'] = False

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

    def initialise_simulation(self, sim):

        event = NewbornOutcomesLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=0))

    def on_birth(self, mother_id, child_id):

        df = self.sim.population.props
        mni = self.sim.modules['Labour'].mother_and_newborn_info

        df.at[child_id, 'nb_early_preterm'] = False
        df.at[child_id, 'nb_late_preterm'] = False
        df.at[child_id, 'nb_congenital_anomaly'] = False
        df.at[child_id, 'nb_early_onset_neonatal_sepsis'] = False
        df.at[child_id, 'nb_birth_asphyxia'] = False
        df.at[child_id, 'nb_intravascular_haem'] = False
        df.at[child_id, 'nb_necrotising_entero'] = False
        df.at[child_id, 'nb_resp_distress_synd'] = False
        df.at[child_id, 'nb_ptb_comps'] = False
        df.at[child_id, 'nb_birth_weight'] = None
        df.at[child_id, 'nb_size_for_gestational_age'] = None
        df.at[child_id, 'nb_early_breastfeeding'] = False
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

        # TODO: Meet with TC to discuss how to accurately select DALYs for complications


class NewbornOutcomeEvent(Event, IndividualScopeEventMixin):
    """ This event determines if , following delivery, a newborn has experienced any complications """

    def __init__(self, module, individual_id,cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self
        mni = self.sim.modules['Labour'].mother_and_newborn_info

    # First we identify the mother of this newborn, allowing us to access information about her labour and delivery
        mother_id = df.at[individual_id, 'mother_id']

# ================================== BIRTH-WEIGHT AND SIZE FOR GESTATIONAL AGE =========================================

    # TODO: both these properties may be created and handeled by the malnutrion module so the below code is placeholding
    # TODO: consider if this will be drawn from a distribution

    # Here we determine if a newly born neonate will be of low birth weight and or small for gestational age

        # DUMMY
        rf1 = 1
        #  rf2 = 1
        risk_factors = rf1  # '*rf2
        eff_prob_lbw = risk_factors * params['base_incidence_low_birth_weight']
        random = self.sim.rng.random_sample(size=1)
        if random < eff_prob_lbw:
            df.at[individual_id, 'nb_birth_weight'] = 'LBW'  # currently limiting to just LBW

        rf1 = 1
        #  rf2 = 1
        risk_factors = rf1  # '*rf2
        eff_prob_sga = risk_factors * params['base_incidence_sga']
        random = self.sim.rng.random_sample(size=1)
        if random < eff_prob_sga:
            df.at[individual_id, 'nb_size_for_gestational_age'] = 'SGA'


# ================================== COMPLICATIONS IN NEONATES AT BIRTH ===============================================

# --------------------------------------- UNDIAGNOSED CONGENITAL ANOMALY ----------------------------------------------
    # Here we apply the incidence of undiagnosed congenital anomaly only being discovered after birth
        # TODO: apply risk of undiagnosed congenital birth anomoly AND the impact of CBAs on at birth comps.

# -----------------------------------------  EARLY ONSET SEPSIS (<72hrs post) -----------------------------------------

        # TODO: ensure birth weight/size GA status is applied as a risk factor if appropriate
        # TODO: link in PTB as key risk factor in these complications

        # Facility Deliveries...
        if mni[mother_id]['delivery_setting'] == 'FD':

            rf1 = 1
            riskfactors = rf1
            eff_prob_sepsis = riskfactors * mni[mother_id]['risk_newborn_sepsis']

            random = self.sim.rng.random_sample(size=1)
            if random < eff_prob_sepsis:
                df.at[individual_id, 'nb_early_onset_neonatal_sepsis'] = True
                logger.info('Neonate %d has developed early onset sepsis in a health facility on date %s',individual_id,
                            self.sim.date)

        # Home births...
        elif mni[mother_id]['delivery_setting'] == 'HB':

            rf1 = 1
            riskfactors = rf1
            eff_prob_sepsis = riskfactors * params['prob_early_onset_neonatal_sepsis']
            random = self.sim.rng.random_sample(size=1)
            if random < eff_prob_sepsis:
                df.at[individual_id, 'nb_early_onset_neonatal_sepsis'] = True
                logger.info('Neonate %d has developed early onset sepsis following a home birth on date %s',
                            individual_id, self.sim.date)

# --------------------------------------------  BIRTH ASPHYXIA  --------------------------------------------------------
        # TODO: apply cord issues?
        # TODO: apply severity score/ HIE grading? if we can find links for risk factors. Majority will resolve with
        #  resus/some supportive care but significant injury will lead to HEI of differing gradeE
        #TODO: or do we model enceph as a whole and do HIE as a proportion but not the entire picture
       
        # Facility Deliveries...
        if mni[mother_id]['delivery_setting'] == 'FD':
            rf1 = 1
            riskfactors = rf1
            eff_prob_ba = riskfactors * mni[mother_id]['risk_newborn_ba']
            random = self.sim.rng.random_sample(size=1)
            if random < eff_prob_ba:
                df.at[individual_id, 'nb_birth_asphyxia'] = True
                logger.info('Neonate %d has been born asphyxiated in a health facility on date %s',
                            individual_id, self.sim.date)

        # Home births...
        elif mni[mother_id]['delivery_setting'] == 'HB':
            rf1 = 1
            riskfactors = rf1
            eff_prob_sepsis = riskfactors * params['prob_birth_asphyxia_iprc']
            random = self.sim.rng.random_sample(size=1)
            if random < eff_prob_sepsis:
                df.at[individual_id, 'nb_birth_asphyxia'] = True
                logger.info('Neonate %d has been born asphyxiated following a home birth on date %s',
                            individual_id, self.sim.date)


# ================================== COMPLICATIONS IN NEONATES DELIVERED PRETERM  ======================================

        # Here we apply the incidence of complications associated with prematurity for which this neonate will need
        # additional care:

        if df.at[individual_id, 'nb_early_preterm'] or df.at[individual_id,'nb_late_preterm']:

            rf1 = 1
            riskfactors = rf1
            eff_prob_ivh = riskfactors * params['prob_ivh_preterm']
            random = self.sim.rng.random_sample(size=1)
            if random < eff_prob_ivh:
                df.at[individual_id, 'nb_intravascular_haem'] = True
                logger.info('Neonate %d has developed intravascular haemorrhage secondary to prematurity',
                            individual_id)

            rf1 = 1
            riskfactors = rf1
            eff_prob_nec = riskfactors * params['prob_nec_preterm']
            random = self.sim.rng.random_sample(size=1)
            if random < eff_prob_nec:
                df.at[individual_id, 'nb_necrotising_entero'] = True
                logger.info('Neonate %d has developed necrotising enterocolitis secondary to prematurity',
                            individual_id)

            rf1 = 1
            riskfactors = rf1
            eff_prob_rds = riskfactors * params['prob_nrds_preterm']
            random = self.sim.rng.random_sample(size=1)
            if random < eff_prob_rds:
                df.at[individual_id, 'nb_resp_distress_synd'] = True
                logger.info('Neonate %d has developed newborn respiritory distress syndrome secondary to prematurity',
                            individual_id)

# ======================================= SCHEDULING NEWBORN CARE  ====================================================

        # If this neonate has been delivered in a facility...
        if mni[mother_id]['delivery_setting'] == 'FD':
            event = HSI_NewbornOutcomes_ReceivesCareFollowingDelivery(self.module, person_id=individual_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                    priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=14)
                                                                    )
            logger.info('This is NewbornOutcomesEvent scheduling HSI_NewbornOutcomes_ReceivesCareFollowingDelivery '
                        'for person %d', individual_id)

            # TODO: will we consider care seeking for newborns from women who deliver in the community??
            # and apply care practices of homebirth (breast feeding etc) 

        # All neonates are scheduled death event
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
        # TODO: death from NEC/ NRDS/IVH will not be handled here?

        if df.at[individual_id, 'nb_early_onset_neonatal_sepsis']:
            random = self.sim.rng.random_sample()
            if random > params['cfr_neonatal_sepsis']:
                df.at[individual_id, 'nb_death_after_birth'] = True

        if df.at[individual_id, 'nb_birth_asphyxia']:
            random = self.sim.rng.random_sample()
            if random > params['cfr_neonatal_enceph']:
                df.at[individual_id, 'nb_death_after_birth'] = True

        if df.at[individual_id, 'nb_congenital_anomaly']:
            random = self.sim.rng.random_sample()
            if random > params['cfr_cba']:
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
        # (am I just dealing with the babies who die on day 1)


# ================================ Health System Interaction Events ================================================

class HSI_NewbornOutcomes_ReceivesCareFollowingDelivery(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    This is event manages care received by newborns following a facility delivery, including referral for additional
     treatment in the case of an emergency
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # Todo: review  (DUMMY)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        dummy_pkg_code = pd.unique(consumables.loc[consumables[
                                                       'Intervention_Pkg'] ==
                                                   'HIV Testing Services',
                                                   'Intervention_Pkg_Code'])[0]  # DUMMY

        the_cons_footprint = {
            'Intervention_Package_Code': [dummy_pkg_code],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'NewbornOutcomes_ReceivesCareFollowingDelivery'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = [1, 2, 3]
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        logger.info('This is HSI_NewbornOutcomes_ReceivesCareFollowingDelivery, neonate %d is receiving care from a '
                    'skilled birth attendant following their birth',
                     person_id)


# ----------------------------------- CHLORHEXIDINE CORD CARE ----------------------------------------------------------

# ------------------------------ EARLY INITIATION OF BREAST FEEDING ----------------------------------------------------

        random = self.sim.rng.random_sample(size=1)
        if random < params['prob_early_breastfeeding_hf']:
            df.at[person_id, 'nb_early_breastfeeding'] = True
            logger.info(
                'Neonate %d has started breastfeeding within 1 hour of birth', person_id)
        else:
            logger.info(
                'Neonate %d did not start breastfeeding within 1 hour of birth', person_id)

# ------------------------------ ANTIBIOTIC PROPHYLAXSIS ---------------------------------------------------------------

        # 3.) Antibiotics for maternal risk factors
        # TODO: need to confirm if this is practice in malawi and find the appropriate risk factors

        # 4.) TBC- Vit K,Syphlis prophylaxis, malaria prophylaxsis
        # todo: these interventions would impact outcomes over the next few days?

        # Recalculate risk of complications
        # Schedule additional HSIs
        # eventually schedule NICU admission for v.sick neonates


class HSI_NewbornOutcomes_ReceivesNewbornResuscitation(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    This is event manages the administration of newborn resuscitation in the event of birth asphyxia
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # Todo: review  (DUMMY)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        dummy_pkg_code = pd.unique(consumables.loc[consumables[
                                                       'Intervention_Pkg'] ==
                                                   'HIV Testing Services',
                                                   'Intervention_Pkg_Code'])[0]  # DUMMY

        the_cons_footprint = {
            'Intervention_Package_Code': [dummy_pkg_code],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'NewbornOutcomes_ReceivesNewbornResuscitation'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = [1,2,3]
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        logger.info('This is HSI_NewbornOutcomes_ReceivesNewbornResuscitation, neonate %d is receiving newborn '
                    'resuscitation following birth ', person_id)

        random = self.sim.rng.random_sample(size=1)
        if random < params['prob_successful_resuscitation']:
            df.at[person_id, 'nb_birth_asphyxia'] = False
            logger.info(
                'Neonate %d has been successfully resuscitated after delivery with birth asphyxia', person_id)

        # TODO: success by severity? Scheduling ICU,

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
