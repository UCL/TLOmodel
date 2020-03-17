from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import demography
from tlo.methods.healthsystem import HSI_Event

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NewbornOutcomes(Module):
    """This module is responsible for the outcomes of newborns immediately following delivery and interventions provided
     by skilled birth attendants to newborns following delivery"""

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)

        self.resourcefilepath = resourcefilepath
        self.newborn_care_info = dict()

    PARAMETERS = {
        'base_incidence_low_birth_weight': Parameter(
            Types.REAL, 'baseline incidence of low birth weight for neonates'),
        'base_incidence_sga': Parameter(
            Types.REAL, 'baseline incidence of small for gestational age for neonates'),
        'prob_congenital_ba': Parameter(
            Types.REAL, 'baseline probability of a neonate being born with a congenital anomaly'),
        'prob_cba_type': Parameter(
            Types.LIST, 'Probability of types of CBA'),
        'prob_early_onset_neonatal_sepsis': Parameter(
            Types.REAL, 'baseline probability of a neonate developing sepsis following birth'),
        'prob_failure_to_transition': Parameter(
            Types.REAL, 'baseline probability of a neonate developing intrapartum related complications '
                        '(previously birth asphyxia) following delivery '),
        'prob_encephalopathy': Parameter(
            Types.REAL, 'baseline probability of a neonate developing encephalopathy of any severity following birth'),
        'prob_enceph_severity': Parameter(
            Types.LIST, 'probability of the severity of encephalopathy in a newborn who is encephalopathic'),
        'prob_retinopathy_preterm': Parameter(
            Types.REAL,
            'baseline probability of a preterm neonate developing retinopathy of prematurity '),
        'prob_retinopathy_severity': Parameter(
            Types.LIST,
            'probabilities of severity of retinopathy'),
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
        'prob_successful_resuscitation': Parameter(
            Types.REAL, 'probability newborn resuscitation will be successful'),
        'cfr_neonatal_sepsis': Parameter(
            Types.REAL, 'case fatality rate for a neonate due to neonatal sepsis'),
        'cfr_encephalopathy': Parameter(
            Types.REAL, 'case fatality rate for a neonate due to encephalopathy'),
        'cfr_failed_to_transition': Parameter(
            Types.REAL, 'case fatality rate for a neonate following failure to transition'),
        'cfr_preterm_birth': Parameter(
            Types.REAL, 'case fatality rate for a neonate following failure to transition'),
        'prob_care_seeking_for_complication': Parameter(
            Types.REAL, 'baseline probability that a mother will seek care for an unwell neonate following delivery'),
    }

    PROPERTIES = {
        'nb_early_preterm': Property(Types.BOOL, 'whether this neonate has been born early preterm (24-33 weeks '
                                                 'gestation)'),
        'nb_late_preterm': Property(Types.BOOL, 'whether this neonate has been born late preterm (34-36 weeks '
                                                'gestation)'),
        'nb_congenital_anomaly': Property(Types.CATEGORICAL, 'Congenital Anomalies: None, Orthopedic, Gastrointestinal,'
                                                             'Neurological, Cosmetic, Other',
                                          #  todo: May need more specificity
                                          categories=['none', 'ortho', 'gastro', 'neuro', 'cosmetic', 'other']),
        'nb_early_onset_neonatal_sepsis': Property(Types.BOOL, 'whether his neonate has developed neonatal sepsis'
                                                               ' following birth'),
        'nb_failed_to_transition': Property(Types.BOOL, 'whether this neonate has failed to transition to breathing on '
                                                         'their own following birth'),
        'nb_encephalopathy': Property(Types.CATEGORICAL, 'None, mild encephalopathy, moderate encephalopathy, '
                                                         'severe encephalopathy',
                                      categories=['none', 'mild_enceph', 'moderate_enceph', 'severe_enceph']),
        'nb_retinopathy_prem': Property(Types.CATEGORICAL, 'Level of visual disturbance due to retinopathy of'
                                                           ' prematurity: None, mild, moderate, severe, blindness',
                                        categories=['none', 'mild', 'moderate', 'severe', 'blindness']),
        'nb_ongoing_impairment': Property(Types.CATEGORICAL, 'none, mild motor, mild motor and cognitive, '
                                                             'moderate motor, moderate motor and cognitive, '
                                                             'severe motor, severe motor and cognitive',
                                          categories=['none', 'mild_mot', 'mild_mot_cog', 'mod_mot', 'mod_mot_cog',
                                                      'severe_mot', ' severe_mot_cog']),
        'nb_birth_weight': Property(Types.CATEGORICAL, 'extremely low birth weight (<1000g), very low birth weight '
                                                       '(<1500g), low birth weight (<2500g),'
                                                       'normal birth weight (>2500g)',
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

        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_NewbornOutcomes.xlsx',
                            sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)

        # TODO: Discuss with team the best way to organise and apply DALY weights for newborns

        # The below code is commented out whilst we determine the best way too apply DALYs and prelonged disability to
        # newborns following birth...

#        if 'HealthBurden' in self.sim.modules.keys():
#            params['nb_daly_wts'] = {
#                'mild_motor_cognitive_<28wks': self.sim.modules['HealthBurden'].get_daly_weight(357),
#                'mild_motor_cognitive_32_36wks': self.sim.modules['HealthBurden'].get_daly_weight(359),
#                'mild_motor_<28wks': self.sim.modules['HealthBurden'].get_daly_weight(371),
#                'moderate_motor_<28wks': self.sim.modules['HealthBurden'].get_daly_weight(378),
#                'severe_motor_<28wks': self.sim.modules['HealthBurden'].get_daly_weight(383),
#                'mild_motor_28_32wks': self.sim.modules['HealthBurden'].get_daly_weight(372),
#                'moderate_motor_28_32wks': self.sim.modules['HealthBurden'].get_daly_weight(377),
#                'severe_motor_28_32wks': self.sim.modules['HealthBurden'].get_daly_weight(375),
#                'mild_motor_32_36wks': self.sim.modules['HealthBurden'].get_daly_weight(373),
#                'moderate_motor_32_36wks': self.sim.modules['HealthBurden'].get_daly_weight(379),
#                'severe_motor_32_36wks': self.sim.modules['HealthBurden'].get_daly_weight(366),
#                'mild_vision_rptb': self.sim.modules['HealthBurden'].get_daly_weight(404),
#                'moderate_vision_rptb': self.sim.modules['HealthBurden'].get_daly_weight(405),
#                'severe_vision_rptb': self.sim.modules['HealthBurden'].get_daly_weight(402),
#                'blindness_rptb': self.sim.modules['HealthBurden'].get_daly_weight(386),
#                'mild_motor_enceph': self.sim.modules['HealthBurden'].get_daly_weight(416),
#                'moderate_motor_enceph': self.sim.modules['HealthBurden'].get_daly_weight(411),
#                'severe_motor_enceph': self.sim.modules['HealthBurden'].get_daly_weight(410),
#                'mild_motor_cognitive_enceph': self.sim.modules['HealthBurden'].get_daly_weight(419),
#                'severe_motor_cognitive_enceph': self.sim.modules['HealthBurden'].get_daly_weight(420),
#                'mild_motor_sepsis': self.sim.modules['HealthBurden'].get_daly_weight(431),
#                'moderate_motor_sepsis': self.sim.modules['HealthBurden'].get_daly_weight(438),
#                'severe_motor_sepsis': self.sim.modules['HealthBurden'].get_daly_weight(435),
#                'severe_infection_sepsis': self.sim.modules['HealthBurden'].get_daly_weight(436),
#                'mild_motor_cognitive_sepsis': self.sim.modules['HealthBurden'].get_daly_weight(441),
#                'mild_motor_cognitive_haemolytic': self.sim.modules['HealthBurden'].get_daly_weight(457),
#                'severe_motor_cognitive_haemolytic': self.sim.modules['HealthBurden'].get_daly_weight(455)}

# ======================================= LINEAR MODEL EQUATIONS ======================================================
        # All linear equations used in this module are stored within the nb_newborn_equations parameter below

        params = self.parameters

        params['nb_newborn_equations'] = {
            'lbw': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['base_incidence_low_birth_weight'],
             Predictor('age_years').when('.between(0,2)', 1)),

            'sga': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['base_incidence_sga'],
             Predictor('age_years').when('.between(0,2)', 1)),

            'neonatal_sepsis_home_birth': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['prob_early_onset_neonatal_sepsis'],
             Predictor('age_years').when('.between(0,2)', 1)),

            'neonatal_sepsis_facility_delivery': LinearModel(
             LinearModelType.MULTIPLICATIVE,  # todo: this needs to be nci[child_id]['ongoing_sepsis_risk']
             params['prob_early_onset_neonatal_sepsis'],
             Predictor('age_years').when('.between(0,2)', 1)),

            'neonatal_sepsis_death': LinearModel(
             LinearModelType.MULTIPLICATIVE,  # todo: this needs to be nci[child_id]['ongoing_sepsis_risk']
             params['cfr_neonatal_sepsis'],
             Predictor('age_years').when('.between(0,2)', 1)),

            'failure_to_transition': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['prob_failure_to_transition'],
             Predictor('age_years').when('.between(0,2)', 1)),

            'failed_to_transition_death': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['cfr_failed_to_transition'],
             Predictor('age_years').when('.between(0,2)', 1)),

            'encephalopathy': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['prob_encephalopathy'],
             Predictor('age_years').when('.between(0,2)', 1)),

            'encephalopathy_death': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['cfr_encephalopathy'],
             Predictor('nb_encephalopathy').when('mild', 2).when('moderate', 3).when('severe', 4)),

            'retinopathy': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['prob_retinopathy_preterm'],
             Predictor('age_years').when('.between(0,2)', 1)),

            'preterm_birth_death': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['cfr_preterm_birth'],
             Predictor('age_years').when('.between(0,2)', 1)),

            'care_seeking_for_complication': LinearModel(
             LinearModelType.MULTIPLICATIVE,
             params['prob_care_seeking_for_complication'],
             Predictor('age_years').when('.between(0,2)', 1))

        }

    def initialise_population(self, population):

        df = population.props

        df.loc[df.is_alive, 'nb_early_preterm'] = False
        df.loc[df.is_alive, 'nb_late_preterm'] = False
        df.loc[df.is_alive, 'nb_congenital_anomaly'].values[:] = 'none'
        df.loc[df.is_alive, 'nb_early_onset_neonatal_sepsis'] = False
        df.loc[df.is_alive, 'nb_failed_to_transition'] = False
        df.loc[df.is_alive, 'nb_encephalopathy'].values[:] = 'none'
        df.loc[df.is_alive, 'nb_retinopathy_prem'].values[:] = 'none'
        df.loc[df.is_alive, 'nb_ongoing_impairment'].values[:] = 'none'
        df.loc[df.is_alive, 'nb_birth_weight'].values[:] = 'NBW'
        df.loc[df.is_alive, 'nb_size_for_gestational_age'].values[:] = 'AGA'
        df.loc[df.is_alive, 'nb_early_breastfeeding'] = False
        df.loc[df.is_alive, 'nb_kmc'] = False
        df.loc[df.is_alive, 'nb_death_after_birth'] = False
        df.loc[df.is_alive, 'nb_death_after_birth_date'] = pd.NaT

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

    def initialise_simulation(self, sim):

        event = NewbornOutcomesLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=0))

    def eval(self, eq, person_id):
        """Compares the result of a specific linear equation with a random draw providing a boolean for the outcome
        under examination"""
        return self.rng.random_sample(size=1) < eq.predict(self.sim.population.props.loc[[person_id]])[person_id]

    def set_neonatal_death_status(self, individual_id, cause):
        """This function  is called for neonates that have experienced a complication after birth and determines if it
        will cause their death. Properties in the DF are set accordingly."""
        df = self.sim.population.props
        params = self.parameters

        if self.module.eval(params['nb_newborn_equations'][f'{cause}_death'], individual_id):
            df.at[individual_id, 'nb_death_after_birth'] = True
            df.at[individual_id, 'nb_death_after_birth_date'] = self.sim.date

            logger.debug(F'This is NewbornOutcomes scheduling a death for person %d on date %s who died due to {cause}'
                         'complications following birth', individual_id, self.sim.date)

    def on_birth(self, mother_id, child_id):
        """The on_birth function of this module is used to apply the probability that a newborn will experience
        complications following delivery (which may or may not be attributable to the delivery process). This section of
        code is subject to change as it contains some dummy code (low birth weight/SGA application) and needs review by
        a clinician"""

        df = self.sim.population.props
        params = self.parameters
        nci = self.newborn_care_info

        # mni dictionary is deleted on maternal death/disease reset. Condition on mother being alive to read in mni
        # from labour
        if df.at[mother_id, 'is_alive'] or (~df.at[mother_id, 'is_alive'] &
                                            ~df.at[mother_id, 'la_intrapartum_still_birth']):
            mni = self.sim.modules['Labour'].mother_and_newborn_info
            m = mni[mother_id]

        df.at[child_id, 'nb_early_preterm'] = False
        df.at[child_id, 'nb_late_preterm'] = False
        df.at[child_id, 'nb_congenital_anomaly'] = 'none'
        df.at[child_id, 'nb_early_onset_neonatal_sepsis'] = False
        df.at[child_id, 'nb_failed_to_transition'] = False
        df.at[child_id, 'nb_encephalopathy'] = 'none'
        df.at[child_id, 'nb_retinopathy_prem'] = 'none'
        df.at[child_id, 'nb_ongoing_impairment'] = 'none'
        df.at[child_id, 'nb_birth_weight'] = 'NBW'
        df.at[child_id, 'nb_size_for_gestational_age'] = 'AGA'
        df.at[child_id, 'nb_early_breastfeeding'] = False
        df.at[child_id, 'nb_kmc'] = False
        df.at[child_id, 'nb_death_after_birth'] = False
        df.at[child_id, 'nb_death_after_birth_date'] = pd.NaT

        child = df.loc[child_id]

        # Here we set the variables for newborns delivered at less than 37 weeks, allocating them to either late or
        # early  preterm based on the gestation at labour

        if m['labour_state'] == 'early_preterm_labour':
            df.at[child_id, 'nb_early_preterm'] = True
            logger.info('%s|early_preterm|%s', self.sim.date,
                        {'age': df.at[child_id, 'age_years'],
                         'person_id': child_id})

        elif m['labour_state'] == 'late_preterm_labour':
            df.at[child_id, 'nb_late_preterm'] = True
            logger.info('%s|late_preterm|%s', self.sim.date,
                        {'age': df.at[child_id, 'age_years'],
                         'person_id': child_id})
        else:
            df.at[child_id, 'nb_early_preterm'] = False
            df.at[child_id, 'nb_late_preterm'] = False

        # Check no children born at term or postterm women are incorrectly categorised as preterm
        if m['labour_state'] == 'TL':
            assert ~df.at[child_id, 'nb_early_preterm']
            assert ~df.at[child_id, 'nb_late_preterm']
        if m['labour_state'] == 'POTL':
            assert ~df.at[child_id, 'nb_early_preterm']
            assert ~df.at[child_id, 'nb_late_preterm']

        if child.nb_early_preterm or child.nb_late_preterm:
            logger.info('%s|preterm_birth|%s', self.sim.date,
                        {'age': df.at[child_id, 'age_years'],
                         'person_id': child_id})

        # TODO: will this log correctly considering intrapartum stillbirths will undergo instantaneous death
        # TODO: will labour initialise first ensuring InstantaneousDeath fires before this logic?

        # Here we apply the prevalence of congenital birth anomalies in infants who have survived to delivery
        if child.is_alive and ~m['stillbirth_in_labour']:
            if self.rng.random_sample() < params['prob_congenital_ba']:
                etiology = ['none', 'ortho', 'gastro', 'neuro', 'cosmetic', 'other']
                probabilities = params['prob_cba_type']
                random_choice = self.rng.choice(etiology, size=1, p=probabilities)
                df.at[child_id, 'nb_congenital_anomaly'] = random_choice

        # For all newborns we first generate a dictionary that will store the prophylactic interventions then receive at
        # birth if delivered in a facility

            # Check the mothers MNI dictionary has data
            assert mni[mother_id]['risk_newborn_sepsis'] is not None

            # Set up NCI dictionary
            nci[child_id] = {'cord_care': False,
                             'vit_k': False,
                             'tetra_eye_d': False,
                             'proph_abx': False,
                             'ongoing_sepsis_risk': mni[mother_id]['risk_newborn_sepsis']}
            # check this is carrying over correctly

            # TODO:  review in context of properties

    # ============(dummy)================= BIRTH-WEIGHT AND SIZE FOR GESTATIONAL AGE ===============(dummy)============

            # We apply an incidence of low birth weight and small for gestational age for preterm infants. This section
            # of code is place holding for code from the yet unfinished malnutrition module
            if ~child.nb_early_preterm & ~child.nb_late_preterm:
                if self.eval(params['nb_newborn_equations']['lbw'], child_id):
                    df.at[child_id, 'nb_birth_weight'] = 'LBW'  # currently limiting to just LBW

            if child.nb_early_preterm:
                df.at[child_id, 'nb_birth_weight'] = 'LBW'

            # Magic number 0.7 is an arbitrary probability determining LBW in late preterm infants
            if child.nb_late_preterm & (self.rng.random_sample() < 0.7):
                df.at[child_id, 'nb_birth_weight'] = 'LBW'

            if self.eval(params['nb_newborn_equations']['sga'], child_id):
                df.at[child_id, 'nb_size_for_gestational_age'] = 'SGA'

                # TODO:  SGA should be managed here (owned by me- but not LBW)

    # ================================== COMPLICATIONS FOLLOWING BIRTH ================================================
            # Here, using linear equations, we determine individual newborn risk of complications following delivery.
            # This risk is either stored or applied depending on the complication, as detailed below

            # SEPSIS....
            if m['delivery_setting'] == 'home_birth' and self.eval(params['nb_newborn_equations'][
                                                                       'neonatal_sepsis_home_birth'], child_id):
                df.at[child_id, 'nb_early_onset_neonatal_sepsis'] = True

                logger.info('Neonate %d has developed early onset sepsis following a home birth on date %s',
                            child_id, self.sim.date)
                logger.info('%s|early_onset_nb_sep_hb|%s', self.sim.date, {'person_id': child_id})

            elif m['delivery_setting'] == 'facility_delivery':
                nci[child_id]['ongoing_sepsis_risk'] = params['nb_newborn_equations'][
                    'neonatal_sepsis_facility_delivery'].predict(df.loc[[child_id]])[child_id]

            # ENCEPHALOPATHY....
            # todo: term only
            if ~child.nb_early_preterm & ~child.nb_late_preterm:
                if self.eval(params['nb_newborn_equations']['encephalopathy'], child_id):
                    # For a newborn who is encephalopathic we then set the severity
                    severity_enceph = self.rng.choice(('mild', 'moderate', 'severe'), p=params['prob_enceph_severity'])
                    if severity_enceph == 'mild':
                        df.at[child_id, 'nb_encephalopathy'] = 'mild_enceph'
                    elif severity_enceph == 'moderate':
                        df.at[child_id, 'nb_encephalopathy'] = 'moderate_enceph'
                    else:
                        df.at[child_id, 'nb_encephalopathy'] = 'severe_enceph'

                    # Check all encephalopathy cases receive a grade
                    assert df.at[child_id, 'nb_encephalopathy'] != 'none'

            # PRETERM BIRTH COMPS
            # todo: just apply risk of death

            # retinopathy is a specific complication of prematurity that we do apply explicitly to map with DALYs

            if (child.nb_early_preterm or child.nb_late_preterm) and self.eval(params['nb_newborn_equations'][
                                                                                   'retinopathy'], child_id):
                random_draw = self.rng.choice(('mild', 'moderate', 'severe', 'blindness'),
                                                      p=params['prob_retinopathy_severity'])
                df.at[child_id, 'nb_retinopathy_prem'] = random_draw
                logger.debug(f'Neonate %d has developed {random_draw }retinopathy of prematurity',
                             child_id)

            # TODO: Consider application of impairment variable and how based to decide probabilities

            # FAILURE TO TRANSITION...
            if self.eval(params['nb_newborn_equations']['failure_to_transition'], child_id):
                df.at[child_id, 'nb_failed_to_transition'] = True
                logger.debug(f'Neonate %d has failed to transition from foetal to neonatal state and is not breathing '
                             f'following delivery', child_id)

    # ===================================== DISABILITY/LONG TERM IMPAIRMENT  ==========================================

            # This is a placeholder for dealing with properties that will capture the long term outcome sequalae
            # associated with a number of these complications. This will likely be stored in the nb_ongoing_impairment
            # property

    # ======================================= SCHEDULING NEWBORN CARE  ================================================
        # Neonates who were delivered in a facility are automatically scheduled to receive care after birth
            if m['delivery_setting'] == 'facility_delivery':
                # TODO: currently level 2 event is not ever scheduled
                event = HSI_NewbornOutcomes_ReceivesSkilledAttendanceFollowingBirthFacilityLevel1(self,
                                                                                                  person_id=child_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(event, priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1))

                logger.debug('This is NewbornOutcomesEvent scheduling HSI_NewbornOutcomes_ReceivesSkilledAttendance'
                             'FollowingBirthFacilityLevel1 for child %d following a facility delivery', child_id)

        # If this neonate was delivered at home and develops a complications we determine the likelihood of care
        # seeking
            if (m['delivery_setting'] == 'home_birth') & (child.nb_failed_to_transition or
                                                          child.nb_early_onset_neonatal_sepsis or
                                                          child.nb_encephalopathy != 'none'):
                if self.eval(params['nb_newborn_equations']['care_seeking_for_complication'], child_id):
                    event = HSI_NewbornOutcomes_ReceivesSkilledAttendanceFollowingBirthFacilityLevel1(self,
                                                                                                      person_id=
                                                                                                      child_id)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(event, priority=0,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(days=1))

                    logger.debug('This is NewbornOutcomesEvent scheduling HSI_NewbornOutcomes_ReceivesSkilledAttendance'
                                 'FollowingBirthFacilityLevel1 for child %d whose mother has sought care after a '
                                 'complication has developed following a home_birth', child_id)

    # ============================================ BREAST FEEDING AT HOME ============================================
                # Using DHS data we apply a one of probability that women who deliver at home will initiate
                # breastfeeding within one hour of birth
            if (m['delivery_setting'] == 'home_birth') and (~child.nb_failed_to_transition and
                                                            ~child.nb_early_onset_neonatal_sepsis and
                                                            (child.nb_encephalopathy == 'none')):

                if self.rng.random_sample() < params['prob_early_breastfeeding_hb']:
                    df.at[child_id, 'nb_early_breastfeeding'] = True
                    logger.debug(
                        'Neonate %d has started breastfeeding within 1 hour of birth', child_id)
                else:
                    logger.debug('Neonate %d did not start breastfeeding within 1 hour of birth', child_id)

    # ===================================== SCHEDULING NEWBORN DEATH EVENT  ============================================
                # All newborns are then scheduled to pass through a newborn death event to determine likelihood of death
                # in the presence of complications

            self.sim.schedule_event(NewbornDeathEvent(self, child_id), self.sim.date + DateOffset(days=3))
            logger.info('This is NewbornOutcomesEvent scheduling NewbornDeathEvent for person %d', child_id)

            # In the rare instance that a baby has been delivered following a mothers death in labour, we now delete
            # the mni dictionary record for that woman
            if m['death_in_labour']:
                del mni[mother_id]

    def on_hsi_alert(self, person_id, treatment_id):

        logger.info('This is NewbornOutcomes, being alerted about a health system interaction '
                    'person %d for: %s', person_id, treatment_id)

    def report_daly_values(self):
        logger.debug('This is Newborn Outcomes reporting my health values')

        # This section of code is just a dummy whilst we review how best to apply DALYs in the newborn module

        df = self.sim.population.props

        health_values_1 = df.loc[df.is_alive, 'nb_early_onset_neonatal_sepsis'].map(
                    {False: 0, True: 0.324})  # p['daly_wt_mild_motor_sepsis']
        health_values_1.name = 'Sepsis Motor Impairment'
        health_values_df = pd.concat([health_values_1.loc[df.is_alive]], axis=1)

        return health_values_df


class NewbornDeathEvent(Event, IndividualScopeEventMixin):
    """This is the NewbornDeathEvent. It is scheduled for all newborns via the on_birth function. This event determines
    if those newborns who have experienced any complications will die due to them. This event will likely be changed as
     at present we dont deal with death of newborns due to complications of pre-term birth or congenital anomalies
     awaiting discussion regarding ongoing newborn care (NICU)"""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        child = df.loc[individual_id]

        # Check the correct amount of time has passed between birth and the death event
        assert (self.sim.date - df.at[individual_id, 'date_of_birth']) == pd.to_timedelta(3, unit='D')

        # Using the set_neonatal_death_status function, defined above, it is determined if newborns who have experienced
        # complications will die because of them. Successful treatment in a HSI will turn off theses properties meaning
        # newborns will not die due to the complication. This logic needs to be reviewed

        if child.nb_early_onset_neonatal_sepsis:
            self.module.set_neonatal_death_status(individual_id, cause='neonatal_sepsis')

        if child.nb_encephalopathy != 'none':
            self.module.set_neonatal_death_status(individual_id, cause='encephalopathy')

        if child.nb_failed_to_transition:
            self.module.set_neonatal_death_status(individual_id, cause='failed_to_transition')

        if child.nb_early_preterm or child.nb_early_preterm:
            self.module.set_neonatal_death_status(individual_id, cause='preterm_birth')

        child = df.loc[individual_id]
        if child.nb_death_after_birth:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                  cause="neonatal complications"), self.sim.date)

        #  Todo: make sure this is delayed enough following HSI?
        #  TODO: Tim C suggested we need to create an offset (using a distribution?) so we're generating deaths for the
        #   first 48 hours
        # TODO: use append to add cause of death to mni?

# ================================ HEALTH SYSTEM INTERACTION EVENTS ================================================

class HSI_NewbornOutcomes_ReceivesSkilledAttendanceFollowingBirthFacilityLevel1(HSI_Event, IndividualScopeEventMixin):
    """."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, NewbornOutcomes)

        self.TREATMENT_ID = 'NewbornOutcomes_ReceivesSkilledAttendanceFollowingBirthFacilityLevel1'
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['NormalDelivery'] = 1  # ???

        # Define the necessary information for an HSI
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        logger.info('This is HSI_NewbornOutcomes_ReceivesSkilledAttendanceFollowingBirthFacilityLevel: child %d is '
                    'receiving care following delivery in a health facility on date %s', person_id, self.sim.date)


class NewbornOutcomesLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """ This is NewbornOutcomesLoggingEvent. Currently it produces a yearly output of neonatal and infant mortality.
    It is incomplete as both neonatal and infant mortality will be effected by deaths outside of this module """
    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=12))

    def apply(self, population):
        df = self.sim.population.props

        # Here we calculated the infant mortality rate, deaths in the first year of life per 1000 live births
        one_year_prior = self.sim.date - np.timedelta64(1, 'Y')
        live_births_sum = len(df.index[(df.date_of_birth > one_year_prior) & (df.date_of_birth < self.sim.date)])

        cumm_deaths = len(df.index[df.nb_death_after_birth & (df.nb_death_after_birth_date > one_year_prior) &
                          (df.nb_death_after_birth_date < self.sim.date)])

        if cumm_deaths == 0:
            imr = 0
        else:
            imr = cumm_deaths / live_births_sum * 1000

        logger.info(f'The infant mortality ratio on date %s for this year is {imr} per 1000 live births',
                    self.sim.date)

        # Here we calculated the neonatal mortality rate, deaths in the 28 days of life per 1000 live births
        neonatal_deaths = len(df.index[~df.is_alive & ((df.nb_death_after_birth_date - df.date_of_birth)
                                       < pd.Timedelta(28, unit='D'))])

        if neonatal_deaths == 0:
            nmr = 0
        else:
            nmr = cumm_deaths / live_births_sum * 1000

        logger.info(f'The neonatal mortality ratio on date %s for this year is {nmr} per 1000 live births',
                    self.sim.date)
