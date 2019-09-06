import logging

import pandas as pd
from pathlib import Path
import numpy as np
import random


from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent

from tlo.methods import demography, labour, newborn_outcomes


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NewbornOutcomes(Module):
    """
   This module is responsible for the outcomes of newborns immediately following delivery
    """
    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.newborn_care_info = dict()

    PARAMETERS = {
        'base_incidence_low_birth_weight': Parameter(
            Types.REAL, 'baseline incidence of low birth weight for neonates'),
        'base_incidence_sga': Parameter(
            Types.REAL, 'baseline incidence of small for gestational age for neonates'),
        'prob_cba': Parameter(
            Types.REAL, 'baseline probability of a neonate being born with a congenital anomaly'),
        'prob_early_onset_neonatal_sepsis': Parameter(
            Types.REAL, 'baseline probability of a neonate developing sepsis following birth'),
        'prob_resp_depression': Parameter(
            Types.REAL, 'baseline probability of a neonate developing intrapartum related complications '
                        '(previously birth asphyxia) following delivery '),
        'prob_encephalopathy': Parameter(
            Types.REAL, 'baseline probability of a neonate developing encephalopathy of any severity following birth'),
        'prob_ivh_preterm': Parameter(
            Types.REAL, 'baseline probability of a preterm neonate developing an intravascular haemorrhage as a result '
                        'of prematurity '),
        'prob_nec_preterm': Parameter(
            Types.REAL,
            'baseline probability of a preterm neonate developing necrotising enterocolitis as a result of '
            'prematurity'),
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
        'prob_facility_offers_kmc': Parameter(
            Types.REAL, 'probability that the facility in which a low birth weight neonate is born will offer kangaroo'
                        ' mother care for low birth weight infants delivered at home'),
        'cfr_neonatal_sepsis': Parameter(
            Types.REAL, 'case fatality rate for a neonate due to neonatal sepsis'),
        'cfr_enceph_mild_mod': Parameter(
            Types.REAL, 'case fatality rate for a neonate due to mild/moderate neonatal encephalopathy'),
        'cfr_enceph_severe': Parameter(
            Types.REAL, 'case fatality rate for a neonate due to sever neonatal encephalopathy'),

    }

    PROPERTIES = {
        'nb_early_preterm': Property(Types.BOOL, 'whether this neonate has been born early preterm (24-33 weeks '
                                                 'gestation)'),
        'nb_late_preterm': Property(Types.BOOL, 'whether this neonate has been born late preterm (34-36 weeks '
                                                'gestation)'),
        'nb_congenital_anomaly': Property(Types.BOOL, 'whether this neonate has been born with a congenital anomaly'),
        'nb_early_onset_neonatal_sepsis': Property(Types.BOOL, 'whether his neonate has developed neonatal sepsis'
                                                               ' following birth'),
        'nb_respiratory_depression ': Property(Types.BOOL, 'whether this neonate has been born asphyxiated and apneic '
                                                           'due to intrapartum related complications'),
        'nb_hypoxic_ischemic_enceph': Property(Types.BOOL, 'whether a perinatally asphyxiated neonate has developed '
                                                           'hypoxic ischemic encephalopathy'),
        'nb_encephalopathy': Property(Types.CATEGORICAL, 'None, mild encephalopathy, moderate encephalopathy, '
                                                         'severe encephalopathy',
                                                categories=['none', 'mild_enceph', 'moderate_enceph', 'severe_enceph']),
        'nb_intravascular_haem': Property(Types.BOOL, 'whether this neonate has developed an intravascular haemorrhage '
                                                      'following preterm birth'),
        'nb_necrotising_entero': Property(Types.BOOL, 'whether this neonate has developed necrotising enterocolitis '
                                                      'following preterm birth'),
        'nb_resp_distress_synd': Property(Types.BOOL, 'whether this neonate has developed newborn respiritory distress '
                                                      'syndrome following preterm birth '),
        'nb_retinopathy_prem': Property(Types.CATEGORICAL, 'Level of visual disturbance due to retinopathy of'
                                                           ' prematurity: None, mild, moderate, severe, blindness',
                                        categories=['none', 'mild', 'moderate','severe', 'blindness']),
        'nb_ongoing_impairment': Property(Types.CATEGORICAL,'none, mild motor, mild motor and cognitive, moderate motor'
                                                            ' moderate motor and cognitive, severe motor, severe motor '
                                                            'and cognitive',
                                                categories=['none', 'mild_mot', 'mild_mot_cog', 'mod_mot', 'mod_mot_cog',
                                                            'severe_mot',' severe_mot_cog']),
        'nb_birth_weight': Property(Types.CATEGORICAL, 'extremely low birth weight (<1000g), very low birth weight '
                                                      '(<1500g), low birth weight (<2500g),'
                                                      ' normal birth weight (>2500g)',
                                                categories=['ext_LBW', 'very_LBW', 'LBW', 'NBW']),
        'nb_size_for_gestational_age': Property(Types.CATEGORICAL, 'small for gestational age, average for gestational'
                                                                   ' age, large for gestational age',
                                                categories=['SGA', 'AGA', 'LGA']),
        'nb_early_breastfeeding': Property(Types.BOOL, 'whether this neonate is exclusively breastfed after birth'),
        'nb_kmc': Property(Types.BOOL, 'whether this neonate received kangaroo mother care following birth'),
        'nb_death_after_birth': Property(Types.BOOL, 'whether this child has died following complications after birth'),
        'nb_death_after_birth_date': Property(Types.DATE, 'date on which the child died after birth'),

    }

    def read_parameters(self, data_folder):

        params = self.parameters

        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_NewbornOutcomes.xlsx',
                            sheet_name='parameter_values')

        dfd.set_index('parameter_name', inplace=True)

        params['base_incidence_low_birth_weight'] = dfd.loc['base_incidence_low_birth_weight', 'value']
        # dummy (DHS prevalence 12%)
        params['base_incidence_sga'] = dfd.loc['base_incidence_sga', 'value']  # dummy
        params['prob_early_onset_neonatal_sepsis'] = dfd.loc['prob_early_onset_neonatal_sepsis', 'value']  # DUMMY
        params['prob_resp_depression'] = dfd.loc['prob_resp_depression', 'value']  # DUMMY
        params['prob_encephalopathy'] = dfd.loc['prob_encephalopathy', 'value']  # DUMMY  # DUMMY
        params['prob_ivh_preterm'] = dfd.loc['prob_ivh_preterm', 'value']
        params['prob_nec_preterm'] = dfd.loc['prob_nec_preterm', 'value']
        params['prob_nrds_preterm'] = dfd.loc['prob_nrds_preterm', 'value']
        params['prob_early_breastfeeding_hb'] = dfd.loc['prob_early_breastfeeding_hb', 'value']  # DHS 2015
        params['prob_early_breastfeeding_hf'] = dfd.loc['prob_early_breastfeeding_hf', 'value']  # DHS 2015
        params['prob_facility_offers_kmc'] = dfd.loc['prob_facility_offers_kmc', 'value']  # DHS 2015
        params['prob_successful_resuscitation'] = dfd.loc['prob_successful_resuscitation', 'value']  # DUMMY
        params['cfr_neonatal_sepsis'] = dfd.loc['cfr_neonatal_sepsis', 'value'] # DUMMY
        params['cfr_enceph_mild_mod'] = dfd.loc['cfr_enceph_mild_mod', 'value']  # DUMMY
        params['cfr_enceph_severe'] = dfd.loc['cfr_enceph_severe', 'value']  # DUMMY


        if 'HealthBurden' in self.sim.modules.keys():
            params['daly_wt_mild_motor_cognitive_<28wks'] = self.sim.modules['HealthBurden'].get_daly_weight(357)
            params['daly_wt_mild_motor_cognitive_32_36wks'] = self.sim.modules['HealthBurden'].get_daly_weight(359)
            params['daly_wt_mild_motor_<28wks'] = self.sim.modules['HealthBurden'].get_daly_weight(371)
            params['daly_wt_moderate_motor_<28wks'] = self.sim.modules['HealthBurden'].get_daly_weight(378)
            params['daly_wt_severe_motor_<28wks'] = self.sim.modules['HealthBurden'].get_daly_weight(383)
            params['daly_wt_mild_motor_28_32wks'] = self.sim.modules['HealthBurden'].get_daly_weight(372)
            params['daly_wt_moderate_motor_28_32wks'] = self.sim.modules['HealthBurden'].get_daly_weight(377)
            params['daly_wt_severe_motor_28_32wks'] = self.sim.modules['HealthBurden'].get_daly_weight(375)
            params['daly_wt_mild_motor_32_36wks'] = self.sim.modules['HealthBurden'].get_daly_weight(373)
            params['daly_wt_moderate_motor_32_36wks'] = self.sim.modules['HealthBurden'].get_daly_weight(379)
            params['daly_wt_severe_motor_32_36wks'] = self.sim.modules['HealthBurden'].get_daly_weight(366)
            params['daly_wt_mild_vision_rptb'] = self.sim.modules['HealthBurden'].get_daly_weight(404)
            params['daly_wt_moderate_vision_rptb'] = self.sim.modules['HealthBurden'].get_daly_weight(405)
            params['daly_wt_severe_vision_rptb'] = self.sim.modules['HealthBurden'].get_daly_weight(402)
            params['daly_wt_blindness_rptb'] = self.sim.modules['HealthBurden'].get_daly_weight(386)
            params['daly_wt_mild_motor_enceph'] = self.sim.modules['HealthBurden'].get_daly_weight(416)
            params['daly_wt_moderate_motor_enceph'] = self.sim.modules['HealthBurden'].get_daly_weight(411)
            params['daly_wt_severe_motor_enceph'] = self.sim.modules['HealthBurden'].get_daly_weight(410)
            params['daly_wt_mild_motor_cognitive_enceph'] = self.sim.modules['HealthBurden'].get_daly_weight(419)
            params['daly_wt_severe_motor_cognitive_enceph'] = self.sim.modules['HealthBurden'].get_daly_weight(420)
            params['daly_wt_mild_motor_sepsis'] = self.sim.modules['HealthBurden'].get_daly_weight(431)
            params['daly_wt_moderate_motor_sepsis'] = self.sim.modules['HealthBurden'].get_daly_weight(438)
            params['daly_wt_severe_motor_sepsis'] = self.sim.modules['HealthBurden'].get_daly_weight(435)
            params['daly_wt_severe_infection_sepsis'] = self.sim.modules['HealthBurden'].get_daly_weight(436)
            params['daly_wt_mild_motor_cognitive_sepsis'] = self.sim.modules['HealthBurden'].get_daly_weight(441)
            params['daly_wt_mild_motor_cognitive_haemolytic'] = self.sim.modules['HealthBurden'].get_daly_weight(457)
            params['daly_wt_severe_motor_cognitive_haemolytic'] = self.sim.modules['HealthBurden'].get_daly_weight(455)


    def initialise_population(self, population):

        df = population.props  # a shortcut to the data-frame storing data for individuals
        m = self

        df['nb_early_preterm'] = False
        df['nb_late_preterm'] = False
        df['nb_congenital_anomaly'] = False
        df['nb_early_onset_neonatal_sepsis'] = False
        df['nb_respiratory_depression '] = False
        df['nb_hypoxic_ischemic_enceph'] = False
        df['nb_encephalopathy'] = 'none'
        df['nb_intravascular_haem'] = False
        df['nb_necrotising_entero'] = False
        df['nb_resp_distress_synd'] = False
        df['nb_retinopathy_prem'] = 'none'
        df['nb_ongoing_impairment'] = 'none'
        df['nb_birth_weight'] = 'NBW'
        df['nb_size_for_gestational_age'] = None
        df['nb_early_breastfeeding'] = False
        df['nb_kmc'] = False
        df['nb_death_after_birth'] = False
        df['nb_death_after_birth_date'] = pd.NaT

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
        df.at[child_id, 'nb_respiratory_depression '] = False
        df.at[child_id, 'nb_hypoxic_ischemic_enceph'] = False
        df.at[child_id, 'nb_encephalopathy'] = 'none'
        df.at[child_id, 'nb_intravascular_haem'] = False
        df.at[child_id, 'nb_necrotising_entero'] = False
        df.at[child_id, 'nb_resp_distress_synd'] = False
        df.at[child_id, 'nb_retinopathy_prem'] = 'none'
        df.at[child_id, 'nb_ongoing_impairment'] = 'none'
        df.at[child_id, 'nb_birth_weight'] = 'NBW'
        df.at[child_id, 'nb_size_for_gestational_age'] = None
        df.at[child_id, 'nb_early_breastfeeding'] = False
        df.at[child_id, 'nb_kmc'] = False
        df.at[child_id, 'nb_death_after_birth'] = False
        df.at[child_id, 'nb_death_after_birth_date'] = pd.NaT

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

#    def report_daly_values(self):

        # TODO: discuss with TC 1.) will these all be managed here. 2.) application of incidence of these comps?


class NewbornOutcomeEvent(Event, IndividualScopeEventMixin):
    """ This event determines if , following delivery, a newborn has experienced any complications """

    def __init__(self, module, individual_id,cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self
        mni = self.sim.modules['Labour'].mother_and_newborn_info
        mother_id = df.at[individual_id, 'mother_id']

    # For all newborns we first generate a dictionary that will store the prophylactic interventions then receive at
    # birth if delivered in a facility

        nci = self.module.newborn_care_info
        nci[individual_id] = {'cord_care': False,
                              'bcg_vacc': False,
                              'polio_vacc': False,
                              'vit_k': False,
                              'tetra_eye_d': False,
                              'proph_abx': False,
                              'ongoing_sepsis_risk': mni[mother_id]['risk_newborn_sepsis']}


# ================================== BIRTH-WEIGHT AND SIZE FOR GESTATIONAL AGE =========================================

    # !!!!!!!!!!!!! PLACEHOLDER PRIOR TO MALNUTRITION MODULE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # TODO: currently everyone at baseline is NBW- either we apply prevlance of LBW at birth or ignore for now
        # DUMMY
        # Applying a dummy incidence of low birth weight in term infants
        if ~df.at[individual_id, 'nb_early_preterm'] & ~df.at[individual_id, 'nb_late_preterm']:
            rf1 = 1
            risk_factors = rf1
            eff_prob_lbw = risk_factors * params['base_incidence_low_birth_weight']
            random = self.sim.rng.random_sample(size=1)
            if random < eff_prob_lbw:
                df.at[individual_id, 'nb_birth_weight'] = 'LBW'  # currently limiting to just LBW

        if df.at[individual_id, 'nb_early_preterm']:
            df.at[individual_id, 'nb_birth_weight'] = 'LBW'

        if df.at[individual_id, 'nb_late_preterm']:
            prob = 0.7  # Dummy probability of LBW in late preterm infants
            random = self.sim.rng.random_sample(size=1)
            if random < prob:
                df.at[individual_id,'nb_birth_weight'] ='LBW'

        rf1 = 1
        #  rf2 = 1
        risk_factors = rf1  # '*rf2
        eff_prob_sga = risk_factors * params['base_incidence_sga']
        random = self.sim.rng.random_sample(size=1)
        if random < eff_prob_sga:
            df.at[individual_id, 'nb_size_for_gestational_age'] = 'SGA'

# ================================== COMPLICATIONS FOLLOWING BIRTH ====================================================

# -----------------------------------  EARLY ONSET SEPSIS (<72hrs post) -----------------------------------------------

        # As the probability of complications varies between is in part determined by setting of delivery, we apply a
        # different risk varied by this neonate's place of birth

        # If this was a facility delivery...
        if mni[mother_id]['delivery_setting'] == 'FD':

            rf1 = 1  # TBC
            riskfactors = rf1
            ind_sep_risk = nci[individual_id]['ongoing_sepsis_risk']
            float(ind_sep_risk)
            eff_prob_sepsis = riskfactors * ind_sep_risk

            #  Store updated sepsis risk- variable will be changed in HSI
            nci[individual_id]['ongoing_sepsis_risk'] = eff_prob_sepsis

        # If the neonate was born at home we calculate the risk and change the variable now..
        elif mni[mother_id]['delivery_setting'] == 'HB':

            rf1 = 1  # TBC
            riskfactors = rf1
            eff_prob_sepsis = riskfactors * params['prob_early_onset_neonatal_sepsis']

            random = self.sim.rng.random_sample(size=1)
            if random < eff_prob_sepsis:
                df.at[individual_id, 'nb_early_onset_neonatal_sepsis'] = True

                logger.info('Neonate %d has developed early onset sepsis following a home birth on date %s',
                            individual_id, self.sim.date)

                logger.info('%s|early_onset_nb_sep_hb|%s', self.sim.date,
                            {'person_id': individual_id})

# --------------------------------------------  RESP DEPRESSION  --------------------------------------------------------

        # cord:
        # mni[mother_id]['cord_prolapse'] = True

        # If this was a facility delivery...
        if mni[mother_id]['delivery_setting'] == 'FD':
            rf1 = 1
            riskfactors = rf1
            eff_prob_ba = riskfactors * mni[mother_id]['risk_newborn_ba']

            random = self.sim.rng.random_sample(size=1)
            if random < eff_prob_ba:
                df.at[individual_id, 'nb_respiratory_depression '] = True

                logger.info('Neonate %d has been born asphyxiated in a health facility on date %s',
                            individual_id, self.sim.date)

                logger.info('%s|birth_asphyxia_fd|%s', self.sim.date,
                            {'person_id': individual_id})

        # if it was a home births...
        elif mni[mother_id]['delivery_setting'] == 'HB':
            rf1 = 1
            riskfactors = rf1
            eff_prob_sepsis = riskfactors * params['prob_resp_depression']

            random = self.sim.rng.random_sample(size=1)
            if random < eff_prob_sepsis:
                df.at[individual_id, 'nb_respiratory_depression '] = True

                logger.info('Neonate %d has been born asphyxiated following a home birth on date %s',
                            individual_id, self.sim.date)

                logger.info('%s|birth_asphyxia_hb|%s', self.sim.date,
                            {'person_id': individual_id})

# ------------------------------------  NEONATAL ENCEPHALOPATHY --------------------------------------------------------

        # Here we apply the incidence and grade of neonatal encephalopathy to children delivered at home

        # TODO: should this become before BA as BA is an outcome of NE/outcome of NE causative agent?
        rf1 = 1
        riskfactors = rf1
        eff_prob_enceph = riskfactors * params['prob_encephalopathy']
        random = self.sim.rng.random_sample(size=1)
        if random < eff_prob_enceph:
            random2 = self.sim.rng.choice(('mild', 'moderate','severe'), p=[0.422, 0.338, 0.24])
            if random2 == 'mild':
                df.at[individual_id, 'nb_encephalopathy'] = 'mild_enceph'
            elif random2 == 'moderate':
                df.at[individual_id, 'nb_encephalopathy'] = 'moderate_enceph'
            else:
                df.at[individual_id, 'nb_encephalopathy'] = 'severe_enceph'

# ================================== COMPLICATIONS IN NEONATES DELIVERED PRETERM  ======================================

        # Here we apply the incidence of complications associated with prematurity for which this neonate will need
        # additional care:

        # TODO: APPLY RISK REDUCTION WITH STEROIDS (if mni[][delivery_setting] == 'FD') etc etc (or should this still be
        # kept with the mother as its not where they were delivered that meant they got steroids but rather if they
        # were availible

        # TODO: consider applying the incidence of motor impairment, cognitive impairment , retinopathy (CP?) to allow
        #  for DALY counting

        if df.at[individual_id, 'nb_early_preterm'] & (df.at[mother_id,'ac_gestational_age'] < 32):
            # LINKED WITH <32 weeks gest & VLBW/LBW (maybe exclude others)
            rf1 = 1
            riskfactors = rf1
            eff_prob_ivh = riskfactors * params['prob_ivh_preterm']
            random = self.sim.rng.random_sample(size=1)
            if random < eff_prob_ivh:
                df.at[individual_id, 'nb_intravascular_haem'] = True
                logger.info('Neonate %d has developed intravascular haemorrhage secondary to prematurity',
                            individual_id)

        if df.at[individual_id, 'nb_early_preterm'] or df.at[individual_id, 'nb_late_preterm']:
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

            #TODO: RETINOPATHY
            #TODO: WHERE IS BEST TO APPLY 'IMPAIRMENT' VARIABLE

# ======================================= SCHEDULING NEWBORN CARE  ====================================================

        # If this neonate has been delivered in a facility they not need to seek care to receive care after delivery...
        if mni[mother_id]['delivery_setting'] == 'FD':
            event = HSI_NewbornOutcomes_ReceivesCareFollowingDelivery(self.module, person_id=individual_id)
#            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
#                                                                    priority=0,
#                                                                    topen=self.sim.date,
#                                                                    tclose=self.sim.date + DateOffset(days=14)
#                                                                    )
#            logger.info('This is NewbornOutcomesEvent scheduling HSI_NewbornOutcomes_ReceivesCareFollowingDelivery '
#                        'for person %d following a facility delivery', individual_id)

        # TODO: Finalise care seeking for home birth neonates who develop a complication (below is a dummy)

        # If this neonate was delivered at home and develops a complications we determine the liklihood of care seeking
        if (mni[mother_id]['delivery_setting'] == 'HB') & (df.at[individual_id, 'nb_respiratory_depression '] or
                                                           df.at[individual_id, 'nb_early_onset_neonatal_sepsis'] or
                                                           (df.at[individual_id, 'nb_encephalopathy'] == 'mild_enceph')
                                                           or (df.at[individual_id, 'nb_encephalopathy']
                                                               == 'moderate_enceph')
                                                           or(df.at[individual_id, 'nb_encephalopathy'] ==
                                                              'severe_enceph')): #What about preterm comps?
            prob = 0.75  # DUMMY VALUE
#            random = self.sim.rng.random_sample(size=1)
#            if random < prob:
#                event = HSI_NewbornOutcomes_ReceivesCareFollowingDelivery(self.module, person_id=individual_id)
#                self.sim.modules['HealthSystem'].schedule_hsi_event(event,
#                                                                    priority=0, # Should this be the same as FD
#                                                                   topen=self.sim.date,
#                                                                    tclose=self.sim.date + DateOffset(days=14)
#                                                                    )
#                logger.info('This is NewbornOutcomesEvent scheduling HSI_NewbornOutcomes_ReceivesCareFollowingDelivery '
#                            'for person %d following a home birth', individual_id)


# ============================================ NEWBORN CARE PRACTICES AT HOME ==========================================

        # and apply care practices of home birth (breast feeding etc)
        if (mni[mother_id]['delivery_setting'] == 'HB') & (~df.at[individual_id, 'nb_respiratory_depression '] &
                                                           ~df.at[individual_id, 'nb_early_onset_neonatal_sepsis'] &
                                                           (df.at[individual_id, 'nb_encephalopathy'] == 'none')):

            random = self.sim.rng.random_sample(size=1)
            if random < params['prob_early_breastfeeding_hb']:
                df.at[individual_id, 'nb_early_breastfeeding'] = True
                logger.info(
                    'Neonate %d has started breastfeeding within 1 hour of birth', individual_id)
            else:
                logger.info(
                    'Neonate %d did not start breastfeeding within 1 hour of birth', individual_id)

# ===================================== SCHEDULING NEWBORN DEATH EVENT  ================================================

        # All neonates are scheduled death event
        self.sim.schedule_event(newborn_outcomes.NewbornDeathEvent(self.module, individual_id,
                                                                       cause='neonatal compilications'), self.sim.date)

        logger.info('This is NewbornOutcomesEvent scheduling NewbornDeathEvent for person %d', individual_id)


class NewbornDeathEvent(Event, IndividualScopeEventMixin):

    """ This event determines if neonates who have experience complications will die because of them  """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        # TODO: Currently death from preterm birth complications isnt handled here
        # TODO: will we treat unsuccessful BA and enceph as the same?

    # Here we look at all neonates who have experienced a complication that has been unsucessfully treated and apply
    # case fatality rates

        if df.at[individual_id, 'nb_early_onset_neonatal_sepsis']:
            random = self.sim.rng.random_sample()
            if random < params['cfr_neonatal_sepsis']:
                df.at[individual_id, 'nb_death_after_birth'] = True
                df.at[individual_id, 'nb_death_after_birth_date'] = self.sim.date

                logger.info(
                    'Neonate %d has died from neonatal sepsis', individual_id)

        if df.at[individual_id, 'nb_respiratory_depression ']:
            random = self.sim.rng.random_sample()
            if random > params['cfr_enceph_mild_mod']: #dummy
                df.at[individual_id, 'nb_death_after_birth'] = True

        if (df.at[individual_id, 'nb_encephalopathy'] == 'mild_enceph') or\
           (df.at[individual_id, 'nb_encephalopathy'] == 'moderate_enceph'):
                random = self.sim.rng.random_sample()
                if random < params['cfr_enceph_mild_mod']:
                    df.at[individual_id, 'nb_death_after_birth'] = True
                    df.at[individual_id, 'nb_death_after_birth_date'] = self.sim.date

                    logger.info( 'Neonate %d has died from mild/moderate encephalopathy', individual_id)

        if df.at[individual_id, 'nb_encephalopathy'] == 'severe_enceph':
            random = self.sim.rng.random_sample()
            if random < params['cfr_enceph_severe']:
                df.at[individual_id, 'nb_death_after_birth'] = True
                df.at[individual_id, 'nb_death_after_birth_date'] = self.sim.date

                logger.info(
                    'Neonate %d has died from severe encephalopathy', individual_id)

        if df.at[individual_id, 'nb_congenital_anomaly']: # will this live here?
            random = self.sim.rng.random_sample()
            if random < params['cfr_cba']:
                df.at[individual_id, 'nb_death_after_birth'] = True
                df.at[individual_id, 'nb_death_after_birth_date'] = self.sim.date

                logger.info(
                    'Neonate %d has died from a congenital birth anomaly', individual_id)

        # Schedule the death of any newborns who have died from their complications, whilst cause of death is recorded
        # as "neonatal complications" we will have contributory factors recorded as the properties of newborns

        if df.at[individual_id, 'nb_death_after_birth']:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                  cause="neonatal complications"), self.sim.date)


            logger.info('This is NewbornDeathEvent scheduling a death for person %d on date %s who died due to '
                        ' complications following birth', individual_id, self.sim.date)

            logger.info('%s|neonatal_death_48hrs|%s', self.sim.date,
                        {'age': df.at[individual_id, 'age_years'],
                         'person_id': individual_id})


        #TODO: from Tim C i would say that I'm dealing with deaths in the first 48 hours, could build in a date offset?


# ================================ HEALTH SYSTEM INTERACTION EVENTS ================================================

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
        nci = self.module.newborn_care_info

        logger.info('This is HSI_NewbornOutcomes_ReceivesCareFollowingDelivery, neonate %d is receiving care from a '
                    'skilled birth attendant following their birth',
                    person_id)

# ----------------------------------- CHLORHEXIDINE CORD CARE ----------------------------------------------------------
        nci[person_id]['cord_care'] = True

# ------------------------------------- VACCINATIONS (BCG/POLIO) -------------------------------------------------------
        nci[person_id]['bcg_vacc'] = True
        nci[person_id]['polio_vacc'] = True

# ------------------------------------------ VITAMIN K  ----------------------------------------------------------------
        nci[person_id]['vit_k'] = True

# --------------------------------------- TETRACYCLINE EYE DROPS -------------------------------------------------------
        nci[person_id]['tetra_eye_d'] = True

# --------------------------------- ANTIBIOTIC PROPHYLAXIS (MATERNAL RISK FACTORS)--------------------------------------
        # TODO: need to confirm if this is practice in malawi and find the appropriate risk factors

        nci[person_id]['proph_abx'] = True

# ----------------------------------------- KANGAROO MOTHER CARE -------------------------------------------------------
        # CURRENTLY ONLY APPLYING TO STABLE NEWBORNS (or should anyone be eligible- cochrane looks at discharge
        # mortality)

        if (df.at[person_id, 'nb_birth_weight'] == 'LBW') & (~df.at[person_id, 'nb_respiratory_depression '] &
                                                             ~df.at[person_id, 'nb_early_onset_neonatal_sepsis']
                                                             & (df.at[person_id, 'nb_encephalopathy'] == 'none')):
            random = self.sim.rng.random_sample(size=1)
            if random < params['prob_facility_offers_kmc']: # CHECK THIS THINKING WITH TIM C
                df.at[person_id, 'nb_kmc'] = True
                # TODO: reduce incidence of sepsis

# ------------------------------ EARLY INITIATION OF BREAST FEEDING ----------------------------------------------------
#
        random = self.sim.rng.random_sample(size=1)
        if random < params['prob_early_breastfeeding_hf']:
            df.at[person_id, 'nb_early_breastfeeding'] = True
            logger.info(
                'Neonate %d has started breastfeeding within 1 hour of birth', person_id)
        else:
            logger.info(
                'Neonate %d did not start breastfeeding within 1 hour of birth', person_id)

# ------------------------------ RECALCULATE SEPSIS RISK---------------------------------------------------------------

        # TODO code in effect of prophylaxsis

        random = self.sim.rng.random_sample(size=1)
        if random < nci[person_id]['ongoing_sepsis_risk']:
            df.at[person_id, 'nb_early_onset_neonatal_sepsis'] = True

            logger.info('Neonate %d has developed early onset sepsis in a health facility on date %s',
                        person_id, self.sim.date)

            logger.info('%s|early_onset_nb_sep_fd|%s', self.sim.date,
                        {'person_id': person_id})

#  ================================ SCHEDULE ADDITIONAL TREATMENT ===================================================

        if df.at[person_id, 'nb_respiratory_depression ']:
            logger.info('This is HSI_NewbornOutcomes_ReceivesCareFollowingDelivery: scheduling resuscitation for '
                        'neonateperson %d who has experienced birth asphyxia following delivery', person_id)

            event = HSI_NewbornOutcomes_ReceivesNewbornResuscitation(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                        priority=0,
                                                        topen=self.sim.date,
                                                        tclose=self.sim.date + DateOffset(days=14)
                                                        )

        if df.at[person_id, 'nb_early_onset_neonatal_sepsis']:
            logger.info('This is HSI_NewbornOutcomes_ReceivesCareFollowingDelivery: scheduling treatment for neonate'
                        'person %d who has developed early onset sepsis following delivery', person_id)

            event = HSI_NewbornOutcomes_ReceivesNewbornResuscitation(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                        priority=0,
                                                        topen=self.sim.date,
                                                        tclose=self.sim.date + DateOffset(days=14)
                                                        )


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
        self.ACCEPTED_FACILITY_LEVELS = [1, 2, 3]
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        logger.info('This is HSI_NewbornOutcomes_ReceivesNewbornResuscitation, neonate %d is receiving newborn '
                    'resuscitation following birth ', person_id)

        random = self.sim.rng.random_sample(size=1)
        if random < params['prob_successful_resuscitation']:
            df.at[person_id, 'nb_respiratory_depression '] = False # Link to severe enchep
            logger.info(
                'Neonate %d has been successfully resuscitated after delivery with birth asphyxia', person_id)
        # else:
            # SCHEDULE NICU?

class HSI_NewbornOutcomes_ReceivesTreatmentForSepsis(Event, IndividualScopeEventMixin):
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
        self.TREATMENT_ID = 'NewbornOutcomes_ReceivesTreatmentForSepsis'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = [1, 2, 3]
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        logger.info('This is HSI_NewbornOutcomes_ReceivesTreatmentForSepsis, neonate %d is receiving treatment '
                    'for early onsent neonatal sepsis following birth ', person_id)

        # todo: SEPTIC NEWBORNS WOULD BE ADMITTED? AND RECEIVE 7-10 DAY ABX?
        treatment_effect = params['prob_cure_antibiotics']
        random = self.sim.rng.random_sample(size=1)
        if treatment_effect > random:
            df.at[person_id,'nb_early_onset_neonatal_sepsis'] = False
            print('Treatment success- antibiotics')
            # schedule inpatient admission


class NewbornOutcomesLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """Handles lifestyle logging"""
    def __init__(self, module):
        """schedule logging to repeat every 3 months
        """
    #    self.repeat = 3
    #    super().__init__(module, frequency=DateOffset(days=self.repeat))
        super().__init__(module, frequency=DateOffset(months=12))

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props

        # First 48 hours NMR:

        one_year_prior = self.sim.date - np.timedelta64(1, 'Y')
        live_births = df.index[(df.date_of_birth > one_year_prior) & (df.date_of_birth < self.sim.date)]
        live_births_sum = len(live_births)
        print(live_births_sum)

        deaths = df.index[(df.nb_death_after_birth == True) & (df.nb_death_after_birth_date > one_year_prior) &
                          (df.nb_death_after_birth_date < self.sim.date)]

        cumm_deaths = len(deaths)
        print(cumm_deaths)

        nmr = cumm_deaths / live_births_sum * 1000
        print('The neonatal mortality ratio for this year is', nmr)
