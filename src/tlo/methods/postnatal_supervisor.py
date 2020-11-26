from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging, util
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import demography
from tlo.methods import Metadata
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.dxmanager import DxTest
from tlo.util import BitsetHandler


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PostnatalSupervisor(Module):
    """"""

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        self.postnatal_tracker = dict()

    METADATA = {Metadata.DISEASE_MODULE,
                Metadata.USES_HEALTHSYSTEM}  # declare that this is a disease module (leave as empty set otherwise)

    PARAMETERS = {
        'prob_htn_resolves': Parameter(
            Types.REAL, 'probability hypertension resolves during postpartum'),
        'prob_secondary_pph': Parameter(
            Types.REAL, 'baseline probability of secondary PPH'),
        'cfr_secondary_pph': Parameter(
            Types.REAL, 'case fatality rate for secondary pph'),
        'cfr_postnatal_sepsis': Parameter(
            Types.REAL, 'case fatality rate for postnatal sepsis'),
        'prob_secondary_pph_severity': Parameter(
            Types.LIST, 'probability of mild, moderate or severe secondary PPH'),
        'prob_obstetric_fistula': Parameter(
            Types.REAL, 'probability of a woman developing an obstetric fistula after birth'),
        'baseline_prob_anaemia_per_week': Parameter(
            Types.REAL, 'Weekly probability of anaemia in pregnancy'),
        'prob_endometritis_pn': Parameter(
            Types.REAL, 'probability of endometritis in week one'),
        'prob_urinary_tract_inf_pn': Parameter(
            Types.REAL, 'probability of urinary tract infection in week one'),
        'prob_skin_soft_tissue_inf_pn': Parameter(
            Types.REAL, 'probability of skin and soft tissue infection in week one'),
        'prob_other_inf_pn': Parameter(
            Types.REAL, 'probability of other maternal infections in week one'),
        'prob_late_sepsis_endometritis': Parameter(
            Types.REAL, 'probability of developing sepsis following postpartum endometritis infection'),
        'prob_late_sepsis_urinary_tract_inf': Parameter(
            Types.REAL, 'probability of developing sepsis following postpartum UTI'),
        'prob_late_sepsis_skin_soft_tissue_inf': Parameter(
            Types.REAL, 'probability of developing sepsis following postpartum skin/soft tissue infection'),
        'prob_late_sepsis_other_maternal_infection_pp': Parameter(
            Types.REAL, 'probability of developing sepsis following postpartum other infection'),
        'prob_late_onset_neonatal_sepsis': Parameter(
            Types.REAL, 'probability of late onset neonatal sepsis (all cause)'),
        'cfr_late_neonatal_sepsis': Parameter(
            Types.REAL, 'Risk of death from late neonatal sepsis'),
        'prob_attend_pnc2': Parameter(
            Types.REAL, 'Probability that a woman receiving PNC1 care will return for PNC2 care'),
        'prob_attend_pnc3': Parameter(
            Types.REAL, 'Probability that a woman receiving PNC2 care will return for PNC3 care'),
        'postpartum_sepsis_treatment_effect': Parameter(
            Types.REAL, 'Treatment effect for postpartum sepsis'),
        'secondary_pph_treatment_effect': Parameter(
            Types.REAL, 'Treatment effect for secondary pph'),
        'neonatal_sepsis_treatment_effect': Parameter(
            Types.REAL, 'Treatment effect for neonatal sepsis'),
        'weekly_prob_postnatal_death': Parameter(
            Types.REAL, 'Weekly risk of postnatal death'),
        'severity_late_infection_pn': Parameter(
            Types.LIST, 'probability of mild infection, sepsis or severe sepsis in the later postnatal period'),
        'prob_care_seeking_postnatal_emergency': Parameter(
            Types.REAL, 'baseline probability '),

    }

    PROPERTIES = {
        'pn_id_most_recent_child': Property(Types.INT, 'person_id of a mothers most recent child'),
        'pn_postnatal_period_in_weeks': Property(Types.INT, 'The number of weeks a woman is in the postnatal period '
                                                            '(1-6)'),
        'pn_pnc_visits_maternal': Property(Types.INT, 'The number of postnatal care visits a woman has undergone '
                                                      'following her most recent delivery'),
        'pn_pnc_visits_neonatal': Property(Types.INT, 'The number of postnatal care visits a neonate has undergone '
                                                      'following delivery'),
        'pn_postpartum_haem_secondary': Property(Types.BOOL, 'Whether this woman is experiencing a secondary '
                                                             'postpartum haemorrhage'),
        'pn_postpartum_haem_secondary_severity': Property(Types.CATEGORICAL, 'severity of a womans secondary PPH ',
                                                          categories=['none', 'mild', 'moderate', 'severe']),
        'pn_postpartum_haem_secondary_treatment': Property(Types.BOOL, 'Whether this woman has received treatment for '
                                                                       'secondary PPH'),
        'pn_sepsis_late_postpartum': Property(Types.BOOL, 'Whether this woman is experiencing postnatal (day7+) '
                                                          'sepsis'),
        'pn_sepsis_late_postpartum_treatment': Property(Types.BOOL, 'Whether this woman has received treatment for '
                                                                    'postpartum sepsis'),
        'pn_maternal_pp_infection': Property(Types.INT, 'bitset column for infection'),
        'pn_obstetric_fistula': Property(Types.BOOL, 'Whether this woman has developed an obstetric fistula '
                                                     'following childbirth'),
        'pn_sepsis_late_neonatal': Property(Types.BOOL, 'Whether this neonate has developed late neonatal sepsis '
                                                        'following discharge'),
        'pn_sepsis_late_neonatal_treatment': Property(Types.BOOL, 'Whether this neonate has received treatment for '
                                                                  'neonatal sepsis'),
        'pn_anaemia_following_pregnancy': Property(Types.CATEGORICAL, 'severity of anaemia following pregnancy',
                                                   categories=['none', 'non_severe', 'severe']),
        'pn_emergency_event_mother': Property(Types.BOOL, 'Whether a mother is experiencing an emergency complication'
                                                          ' postnatally'),
        'dummy_count': Property(Types.BOOL, 'dummy'),

    }

    def read_parameters(self, data_folder):

        params = self.parameters
        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_PostnatalSupervisor.xlsx',
                            sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)

    #    if 'HealthBurden' in self.sim.modules.keys():
    #        params['daly_wt_abortive_outcome'] = self.sim.modules['HealthBurden'].get_daly_weight(352)

    # TODO: Disability

    # ==================================== LINEAR MODEL EQUATIONS =====================================================

        params['pn_linear_equations'] = {
            'secondary_postpartum_haem': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_secondary_pph']),

            'secondary_postpartum_haem_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_secondary_pph'],
                Predictor('pn_postpartum_haem_secondary_treatment').when(True, params[
                    'secondary_pph_treatment_effect'])),

            'postnatal_sepsis_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_postnatal_sepsis'],
                Predictor('pn_sepsis_late_postpartum_treatment').when(True, params[
                    'postpartum_sepsis_treatment_effect'])),

            'obstetric_fistula': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_obstetric_fistula']),

            'endometritis': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_endometritis_pn']),

            'urinary_tract_inf': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_urinary_tract_inf_pn']),

            'skin_soft_tissue_inf': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_skin_soft_tissue_inf_pn']),

            'other_maternal_infection': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_other_inf_pn']),

            'sepsis_late_postpartum': LinearModel(
                LinearModelType.ADDITIVE,
                0,  # todo: ???
                Predictor('pn_maternal_pp_infection').apply(
                    lambda x: params['prob_late_sepsis_endometritis']
                    if x & self.postpartum_infections_late.element_repr('endometritis') else 0),
                Predictor('pn_maternal_pp_infection').apply(
                    lambda x: params['prob_late_sepsis_urinary_tract_inf']
                    if x & self.postpartum_infections_late.element_repr('urinary_tract_inf') else 0),
                Predictor('pn_maternal_pp_infection').apply(
                    lambda x: params['prob_late_sepsis_skin_soft_tissue_inf']
                    if x & self.postpartum_infections_late.element_repr('skin_soft_tissue_inf') else 0),
                Predictor('pn_maternal_pp_infection').apply(
                    lambda x: params['prob_late_sepsis_other_maternal_infection_pp']
                    if x & self.postpartum_infections_late.element_repr('other_maternal_infection') else 0)),

            'anaemia_after_pregnancy': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['baseline_prob_anaemia_per_week']),
            #    Predictor('ps_deficiencies_in_pregnancy').apply(
            #        lambda x: params['rr_anaemia_if_iron_deficient']
            #        if x & self.deficiencies_in_pregnancy.element_repr('iron') else 1),
            #    Predictor('ps_deficiencies_in_pregnancy').apply(
            #        lambda x: params['rr_anaemia_if_folate_deficient']
            #        if x & self.deficiencies_in_pregnancy.element_repr('folate') else 1),
            #    Predictor('ps_deficiencies_in_pregnancy').apply(
            #        lambda x: params['rr_anaemia_if_b12_deficient']
            #        if x & self.deficiencies_in_pregnancy.element_repr('b12') else 1)),

            'late_onset_neonatal_sepsis': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_late_onset_neonatal_sepsis']),

            'late_neonatal_sepsis_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_late_neonatal_sepsis'],
                Predictor('pn_sepsis_late_neonatal_treatment').when(True, params[
                    'neonatal_sepsis_treatment_effect'])),

            'postnatal_death_weekly': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['weekly_prob_postnatal_death']),
                # todo: predictors, HTN, anaemia?

            'care_seeking_postnatal_complication': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_care_seeking_postnatal_emergency']),
        }

    def initialise_population(self, population):

        df = population.props

        df.loc[df.is_alive, 'pn_id_most_recent_child'] = -1
        df.loc[df.is_alive, 'pn_postnatal_period_in_weeks'] = 0
        df.loc[df.is_alive, 'pn_pnc_visits_maternal'] = 0
        df.loc[df.is_alive, 'pn_pnc_visits_neonatal'] = 0
        df.loc[df.is_alive, 'pn_postpartum_haem_secondary'] = False
        df.loc[df.is_alive, 'pn_postpartum_haem_secondary_severity'] = 'none'
        df.loc[df.is_alive, 'pn_sepsis_late_postpartum'] = False
        df.loc[df.is_alive, 'pn_sepsis_late_postpartum_treatment'] = False
        # df.loc[df.is_alive, 'pn_maternal_infection'] = 'none'
        df.loc[df.is_alive, 'pn_sepsis_late_neonatal'] = False
        df.loc[df.is_alive, 'pn_sepsis_late_neonatal_treatment'] = False
        df.loc[df.is_alive, 'pn_anaemia_following_pregnancy'] = 'none'
        df.loc[df.is_alive, 'pn_obstetric_fistula'] = False
        df.loc[df.is_alive, 'pn_emergency_event_mother'] = False
        df.loc[df.is_alive, 'dummy_count'] = False

        # This biset property stores infections that can occur in the postnatal period
        self.postpartum_infections_late = BitsetHandler(self.sim.population, 'pn_maternal_pp_infection',
                                               ['endometritis', 'urinary_tract_inf', 'skin_soft_tissue_inf',
                                                'other_maternal_infection'])

    def initialise_simulation(self, sim):
        sim.schedule_event(PostnatalSupervisorEvent(self),
                           sim.date + DateOffset(days=0))

        sim.schedule_event(PostnatalLoggingEvent(self),
                           sim.date + DateOffset(years=1))

        # Define the conditions we want to track
        self.postnatal_tracker = {'endometritis': 0, 'urinary_tract_inf': 0, 'skin_soft_tissue_inf': 0,
                                  'other_maternal_infection': 0,'secondary_pph': 0, 'postnatal_death': 0,
                                  'secondary_pph_death': 0,
                                  'postnatal_sepsis': 0, 'sepsis_death': 0, 'fistula': 0, 'postnatal_anaemia': 0,
                                  'late_neonatal_sepsis': 0, 'neonatal_death': 0, 'neonatal_sepsis_death': 0}

        # Register dx_tests used as assessment for postnatal conditions during PNC visits
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assessment_for_postnatal_sepsis=DxTest(
                property='pn_sepsis_late_postpartum',
                sensitivity=0.99),

            assessment_for_secondary_pph=DxTest(
                property='pn_postpartum_haem_secondary',
                sensitivity=0.99),

            assessment_for_hypertension=DxTest(
                property='ps_htn_disorders', target_categories=['gest_htn', 'mild_pre_eclamp', 'severe_pre_eclamp',
                                                                'eclampsia'],
                sensitivity=0.99),

            assessment_for_neonatal_sepsis=DxTest(
                property='pn_sepsis_late_neonatal',
                sensitivity=0.99))

    def on_birth(self, mother_id, child_id):
        df = self.sim.population.props
        params = self.parameters

        df.at[child_id, 'pn_id_most_recent_child'] = -1
        df.at[child_id, 'pn_postnatal_period_in_weeks'] = 0
        df.at[child_id, 'pn_pnc_visits_maternal'] = 0
        df.at[child_id, 'pn_pnc_visits_neonatal'] = 0
        df.at[child_id, 'pn_postpartum_haem_secondary'] = False
        df.at[child_id, 'pn_postpartum_haem_secondary_severity'] = 'none'
        df.at[child_id, 'pn_sepsis_late_postpartum'] = False
        df.at[child_id, 'pn_sepsis_late_postpartum_treatment'] = False
        # df.at[child_id, 'pn_maternal_infection'] = 'none'
        df.at[child_id, 'pn_maternal_infection_severity'] = 'none'
        df.at[child_id, 'pn_obstetric_fistula'] = False
        df.at[child_id, 'pn_sepsis_late_neonatal'] = False
        df.at[child_id, 'pn_sepsis_late_neonatal_treatment'] = False
        df.at[child_id, 'pn_anaemia_following_pregnancy'] = 'none'
        df.at[child_id, 'pn_emergency_event_mother'] = False
        df.at[child_id, 'dummy_count'] = 'none'

        # We store the ID number of the child this woman has just delivered
        df.at[mother_id, 'pn_id_most_recent_child'] = child_id

        # Here we determine if, following childbirth, this woman will develop a fistula
        risk_of_fistula = params['pn_linear_equations'][
            'obstetric_fistula'].predict(df.loc[[mother_id]])[mother_id]

        if self.rng.random_sample() < risk_of_fistula:
            df.at[mother_id, 'pn_obstetric_fistula'] = True
            self.postnatal_tracker['fistula'] += 1

    def on_hsi_alert(self, person_id, treatment_id):
        logger.debug(key='message', data=f'This is PostnatalSupervisor, being alerted about a health system interaction '
                                         f'person {person_id} for: {treatment_id}')

    def report_daly_values(self):
        df = self.sim.population.props

        logger.debug(key='message', data='This is PostnatalSupervisor reporting my health values')

    def apply_linear_model(self, lm, df):
        """
        Helper function will apply the linear model (lm) on the dataframe (df) to get a probability of some event
        happening to each individual. It then returns a series with same index with bools indicating the outcome based
        on the toss of the biased coin.
        :param lm: The linear model
        :param df: The dataframe
        :return: Series with same index containing outcomes (bool)
        """
        return self.rng.random_sample(len(df)) < lm.predict(df)

    def set_infections(self, individual_id, infection):
        """This function is called by the PostnatalWeekOne event to calculate a womans risk of developing an infection
        in the first week after birth """
        df = self.sim.population.props
        params = self.parameters

        risk_infection = params['pn_linear_equations'][f'{infection}'].predict(df.loc[[
            individual_id]])[individual_id]

        if risk_infection < self.rng.random_sample():
            self.postpartum_infections_late.set([individual_id], f'{infection}')
            self.postnatal_tracker[f'{infection}'] +=1

    def set_postnatal_complications_mothers(self, week):
        """This function is called by the PostnatalSupervisor event. It applies risk of key complication of women
         during each week of the postnatal period"""
        df = self.sim.population.props
        params = self.parameters

        # TODO: this method should replace all time where temporary data frames have been created (PS module)
        # TODO: caveat on the woman not already having each complication

        def onset(eq):
            onset_condition = self.apply_linear_model(
                params['pn_linear_equations'][f'{eq}'],
                df.loc[df['is_alive'] & df['la_is_postpartum'] & (df['pn_postnatal_period_in_weeks'] == week)])
            return onset_condition

        # -------------------------------------- INFECTIONS ---------------------------------------------------------
        onset_endo = onset('endometritis')
        if not onset_endo.loc[onset_endo].empty:
            logger.debug(key='message', data=f'The following women have developed endometritis during week {week} '
                                             f'of the postnatal period,{onset_endo.loc[onset_endo].index}')
            self.postpartum_infections_late.set(onset_endo.loc[onset_endo].index, 'endometritis')

        onset_uti = onset('urinary_tract_inf')
        if not onset_uti.loc[onset_uti].empty:
            logger.debug(key='message', data=f'The following women have developed a UTI during week {week} '
                                             f'of the postnatal period, {onset_uti.loc[onset_uti].index}')
            self.postpartum_infections_late.set(onset_uti.loc[onset_uti].index, 'urinary_tract_inf')

        onset_ssti = onset('skin_soft_tissue_inf')
        if not onset_ssti.loc[onset_ssti].empty:
            logger.debug(key='message', data=f'The following women have developed a skin/soft tissue infection during '
                                             f'week {week} of the postnatal period {onset_ssti.loc[onset_ssti].index}')
            self.postpartum_infections_late.set(onset_ssti.loc[onset_ssti].index, 'skin_soft_tissue_inf')

        onset_other_inf = onset('other_maternal_infection')
        if not onset_other_inf.loc[onset_other_inf].empty:
            logger.debug(key='message', data=f'The following women have developed another infection during '
                                             f'week {week} of the postnatal period, '
                                             f'{onset_other_inf.loc[onset_other_inf].index}')
            self.postpartum_infections_late.set(onset_other_inf.loc[onset_other_inf].index, 'skin_soft_tissue_inf')

        # -------------------------------------- SEPSIS --------------------------------------------------------------
        onset_sepsis = onset('sepsis_late_postpartum')
        df.loc[onset_sepsis.loc[onset_sepsis].index, 'pn_sepsis_late_postpartum'] = True
        df.loc[onset_sepsis.loc[onset_sepsis].index, 'pn_emergency_event_mother'] = True
        if not onset_sepsis.loc[onset_sepsis].empty:
            logger.debug(key='message', data=f'The following women have developed sepsis during week {week} of '
                                             f'the postnatal period, {onset_sepsis.loc[onset_sepsis].index}')
            self.postnatal_tracker['postnatal_sepsis'] += len(onset_sepsis.loc[onset_sepsis])

        # ------------------------------------ SECONDARY PPH ----------------------------------------------------------
        onset_pph = onset('secondary_postpartum_haem')
        df.loc[onset_pph.loc[onset_pph].index, 'pn_postpartum_haem_secondary'] = True
        df.loc[onset_pph.loc[onset_pph].index, 'pn_emergency_event_mother'] = True
        if not onset_pph.loc[onset_pph].empty:
            logger.debug(key='message', data=f'The following women have developed secondary pph during week {week}'
                                             f' of the postnatal period, {onset_pph.loc[onset_pph].index}')
            self.postnatal_tracker['secondary_pph'] += len(onset_pph.loc[onset_pph])

        # ----------------------------------------- ANAEMIA ----------------------------------------------------------
        onset_anaemia = onset('anaemia_after_pregnancy')
        random_choice_severity = pd.Series(self.rng.choice(['non_severe', 'severe'], p=[0.5, 0.5],
                                                           size=len(onset_anaemia.loc[onset_anaemia])),
                                                           index=onset_anaemia.loc[onset_anaemia].index)

        df.loc[onset_anaemia.loc[onset_anaemia].index, 'pn_anaemia_following_pregnancy'] = random_choice_severity
        if not onset_anaemia.loc[onset_anaemia].empty:
                logger.debug(key='message', data=f'The following women have developed anaemia during week {week}'
                                                 f' of the postnatal period, {onset_anaemia.loc[onset_anaemia].index}')
                self.postnatal_tracker['postnatal_anaemia'] += len(onset_anaemia.loc[onset_anaemia])

        # --------------------------------------- RESOLUTION OF HYPERTENSION ------------------------------------------
        women_with_htn = df.loc[
            df['is_alive'] & df['la_is_postpartum'] & (df['pn_postnatal_period_in_weeks'] == week) &
            (df['ps_htn_disorders'] != 'none')]
        resolvers = pd.Series(self.rng.random_sample(len(women_with_htn)) < params['prob_htn_resolves'],
                                 index=women_with_htn.index)
        df.loc[resolvers.loc[resolvers].index, 'ps_htn_disorders'] = 'none'

        # ----------------------------------------- CARE SEEKING -----------------------------------------------------
        care_seeking = self.apply_linear_model(
            params['pn_linear_equations']['care_seeking_postnatal_complication'],
            df.loc[df['is_alive'] & df['la_is_postpartum'] & (df['pn_postnatal_period_in_weeks'] == week) &
                   df['pn_emergency_event_mother']])
        df.loc[care_seeking.index, 'pn_emergency_event_mother'] = False

        for person in care_seeking.loc[care_seeking].index:
            pass
            # todo: schedule hsi

        for person in care_seeking.loc[~care_seeking].index:
            self.sim.schedule_event(LatePostpartumDeathEvent(self, person),
                                    self.sim.date)

        # ---------------------------------------------- DEATH? -------------------------------------------------------

    def set_postnatal_complications_neonates(self, upper_and_lower_day_limits):
        """This function is called by the PostnatalSupervisor event. It applies risk of key complication to neonates
        during each week of the postnatal period"""
        params = self.parameters
        df = self.sim.population.props

        onset_sepsis = self.apply_linear_model(
            params['pn_linear_equations']['late_onset_neonatal_sepsis'],
            df.loc[df['is_alive'] & (df['age_days'] > upper_and_lower_day_limits[0]) &
                   (df['age_days'] < upper_and_lower_day_limits[1])])

        df.loc[onset_sepsis.loc[onset_sepsis].index, 'pn_sepsis_late_neonatal'] = False

        # todo : careseeking
        # todo: ROD

    def maternal_postnatal_care_contact_intervention_bundle(self, individual_id, hsi_event):
        """This function is called by each of the postnatal care visits. Currently it the interventions include
        assessment for sepsis, secondary pph and hypertension. If these are detected women are admitted for treatment"""

        needs_admission = False

        # --------------------------MATERNAL ASSESSMENT AND TREATMENT ---------------------------------------------
        # SEPSIS
        if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run=
                                                                   'assessment_for_postnatal_sepsis',
                                                                   hsi_event=hsi_event):
            logger.debug(key='message', data=f'Mother {individual_id} has been assessed and diagnosed with postpartum '    
                                             f'sepsis, she will be admitted for treatment')
            needs_admission = True

        # HAEMORRHAGE
        if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run=
                                                                   'assessment_for_secondary_pph',
                                                                   hsi_event=hsi_event):
            logger.debug(key='message', data=f'Mother {individual_id} has been assessed and diagnosed with secondary '
                                             f'postpartum haemorrhage hypertension, she will be admitted for '
                                             f'treatment')
            needs_admission = True

        # HYPERTENSION
        if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run=
                                                                   'assessment_for_hypertension',
                                                                   hsi_event=hsi_event):
            logger.debug(key='message',
                         data=f'Mother {individual_id} has been assessed and diagnosed with postpartum '
                         f'hypertension, she will be admitted for treatment')
            needs_admission = True

        if needs_admission:
            admission_event = HSI_PostnatalSupervisor_PostnatalWardInpatientCare(
                self, person_id=individual_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(admission_event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

    def neonatal_postnatal_care_contact_intervention_bundle(self, individual_id, hsi_event):
        """This function is called by each of the postnatal care visits. Currently it the interventions include
        assessment for neonatal sepsis. If detected, neonates are admitted for treatment"""
        # ------------------------------ESSENTIAL NEWBORN CARE (IF HOME BIRTH & VISIT ONE)--------------------------

        # Tetracycline
        # Cord care
        # Vit k
        # starting breast feeding

        # TODO: how will we apply the effect of late essential interventions if we have already applied sepsis risk

        # --------------------------NEONATAL ASSESSMENT AND TREATMENT ---------------------------------------------
        # SEPSIS
        if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run=
                                                                   'assessment_for_neonatal_sepsis',
                                                                   hsi_event=hsi_event):
            logger.debug(key='message', data=f'Neonate {individual_id} has been assessed and diagnosed with neonatal '
            f'sepsis, they will be admitted for treatment')

            sepsis_treatment = HSI_PostnatalSupervisor_NeonatalWardInpatientCare(
                self, person_id=individual_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(sepsis_treatment,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

    def maternal_postnatal_care_care_seeking(self, individual_id, recommended_day_next_pnc, next_pnc_visit,
                                             maternal_pnc):
        """This function is called by HSI_PostnatalSupervisor_PostanatalCareContact(...). It determines if an indiviudal
         will return to attend her next PNC visit"""
        df = self.sim.population.props
        params = self.parameters

        ppp_in_days = self.sim.date - df.at[individual_id, 'la_date_most_recent_delivery']
        days_calc = pd.to_timedelta(recommended_day_next_pnc, unit='D') - ppp_in_days
        date_next_pnc = self.sim.date + days_calc

        if self.rng.random_sample() < params[f'prob_attend_{next_pnc_visit}']:
            self.sim.modules['HealthSystem'].schedule_hsi_event(maternal_pnc,
                                                                priority=0,
                                                                topen=date_next_pnc,
                                                                tclose=date_next_pnc + DateOffset(days=3))


class PostnatalSupervisorEvent(RegularEvent, PopulationScopeEventMixin):
    """ This is the PostnatalSupervisorEvent. It runs every week and applies risk of disease onset/resolution to women
    in the postnatal period of their pregnancy (48hrs - +42days post birth) """

    def __init__(self, module, ):
        super().__init__(module, frequency=DateOffset(weeks=1))

    def apply(self, population):
        df = population.props
        pregnancy_deficiencies = self.sim.modules['PregnancySupervisor'].deficiencies_in_pregnancy

        # TODO: Move to pregnancy supervisor event (PregnancyAndPostnatalSupervisor) as that event already runs weekly
        # TODO: does it matter were apply risk of HTN resolution to women without HTN (opposite for anaemia)
        # TODO: this needs to only pick up women after labour pp
        # TODO: inpatient property

        # ================================ UPDATING LENGTH OF POSTPARTUM PERIOD  IN WEEKS  ============================
        # Here we update how far into the postpartum period each woman who has recently delivered is
        alive_and_recently_delivered = df.is_alive & df.la_is_postpartum
        ppp_in_days = self.sim.date - df.loc[alive_and_recently_delivered, 'la_date_most_recent_delivery']

        # Round function used to ensure women are categorised in the correct week (was previously automatically
        # rounding down)
        ppp_in_weeks = round(ppp_in_days / np.timedelta64(1, 'W'))

        df.loc[alive_and_recently_delivered, 'pn_postnatal_period_in_weeks'] = ppp_in_weeks
        logger.debug(key='message', data=f'updating postnatal periods on date {self.sim.date}')

        zero_women = df.is_alive & df.la_is_postpartum & (df.pn_postnatal_period_in_weeks == 0.0)
        df.loc[zero_women, 'pn_postnatal_period_in_weeks'] = 1.0
        assert (df.loc[alive_and_recently_delivered, 'pn_postnatal_period_in_weeks'] > 0).all().all()

        # ================================= COMPLICATIONS/CARE SEEKING FOR WOMEN ======================================
        for week in [2, 3, 4, 5, 6]:
            self.module.set_postnatal_complications_mothers(week=week)

        # ================================= COMPLICATIONS/CARE SEEKING FOR NEONATES ===================================
        for upper_and_lower_day_limits in [[7, 15], [14, 22], [21, 29], [28, 36], [35, 43]]:
            self.module.set_postnatal_complications_neonates(upper_and_lower_day_limits=upper_and_lower_day_limits)

        # -------------------------------------- RESETTING VARIABLES --------------------------------------------------
        # Here, one week after we stop applying risk of postpartum complications, we reset key postpartum variables
        week_7_postnatal_women = df.is_alive & df.la_is_postpartum & (df.pn_postnatal_period_in_weeks == 7)
        df.loc[week_7_postnatal_women, 'pn_postnatal_period_in_weeks'] = 0
        df.loc[week_7_postnatal_women, 'pn_pnc_visits_maternal'] = 0
        df.loc[week_7_postnatal_women, 'la_is_postpartum'] = False

        df.loc[week_7_postnatal_women, 'ps_htn_disorders'] = 'none'
        df.loc[week_7_postnatal_women, 'ps_anaemia_in_pregnancy'] = 'none'
        df.loc[week_7_postnatal_women, 'pn_anaemia_following_pregnancy'] = 'none'
        df.loc[week_7_postnatal_women, 'pn_sepsis_late_postpartum'] = False
        df.loc[week_7_postnatal_women, 'pn_postpartum_haem_secondary'] = False

        self.module.postpartum_infections_late.unset(week_7_postnatal_women,'endometritis', 'urinary_tract_inf',
                                                     'skin_soft_tissue_inf', 'other_maternal_infection')
        # todo: unset deficiencies
        # todo: unset neonate properties

class PostnatalWeekOneEvent(Event, IndividualScopeEventMixin):
    """This is PostnatalWeekOneEvent. It is scheduled for both mothers (via tha labour module) and newborns following
     birth. This event is used to apply risk of complications occuring during the first week of the postnatal period
     prior to being picked up by the PostnatalSupervisorEvent."""

    def __init__(self, module, individual_id, mother_or_child):
        super().__init__(module, person_id=individual_id)

        self.mother_or_child = mother_or_child

    def apply(self, individual_id):
        df = self.sim.population.props
        mother = df.loc[individual_id]
        params = self.module.parameters

        # todo:delete
        if self.mother_or_child == 'mother':
            df.at[individual_id, 'dummy_count'] += 1
            assert df.at[individual_id, 'dummy_count'] == 1
            assert mother.la_is_postpartum
            assert ~mother.pn_sepsis_late_postpartum
            assert ~mother.pn_postpartum_haem_secondary
            assert (self.sim.date - df.at[individual_id, 'la_date_most_recent_delivery']) < pd.to_timedelta(7, unit='d')
        # todo assert check to make sure not an inpatient?

        if not mother.is_alive:
            return

        # TODO: 'onset' date of all these condtions should be randomised throughout the remaining days of someones
        #  week1 (or we randomise the day they arrive at this event so that onset happens on the day the event runs
        #  which is scattered around week 1)

        if self.mother_or_child == 'mother':
            #  -------------------------------------------- INFECTIONS ------------------------------------------------
            # TODO conidtion on having not already had the same type of infection immediately after birth
            for infection in ['endometritis', 'urinary_tract_inf', 'skin_soft_tissue_inf', 'other_maternal_infection']:
                self.module.set_infections(individual_id, infection=infection)

            # ---------------------------------------------- SEPSIS ---------------------------------------------------
            risk_sepsis = params['pn_linear_equations']['sepsis_late_postpartum'].predict(df.loc[[
                individual_id]])[individual_id]
        
            if risk_sepsis < self.module.rng.random_sample():
                logger.debug(key='message',
                             data=f'mother {individual_id} has developed sepsis later in the postnatal'
                                  f'period')
                df.at[individual_id, 'pn_sepsis_late_postpartum'] = True
                self.module.postnatal_tracker['postnatal_sepsis'] += 1

            #  ---------------------------------------- SECONDARY PPH ------------------------------------------------
            risk_secondary_pph = params['pn_linear_equations']['secondary_postpartum_haem'].predict(df.loc[[
                individual_id]])[individual_id]
        
            if risk_secondary_pph < self.module.rng.random_sample():
                logger.debug(key='message',
                             data=f'mother {individual_id} has developed a secondary postpartum haemorrhage  in the '
                                 f'postnatal period')
                df.at[individual_id, 'pn_postpartum_haem_secondary'] = True
                self.module.postnatal_tracker['secondary_pph'] += 1

            # --------------------------------------- NEW ONSET ANAEMIA -----------------------------------------------
            # TODO: onset deficiencies here as well?
            if mother.ps_anaemia_in_pregnancy == 'none':
                risk_anaemia_after_pregnancy = params['pn_linear_equations']['anaemia_after_pregnancy'].predict(df.loc[[
                    individual_id]])[individual_id]
        
                if risk_anaemia_after_pregnancy < self.module.rng.random_sample():
                    random_choice_severity = self.module.rng.choice(['non_severe', 'severe'], p=[0.5, 0.5], size=1)
                    df.at[individual_id, 'pn_anaemia_following_pregnancy'] = random_choice_severity

            # ----------------------------------------- CARE SEEKING  -------------------------------------------------
            # TODO move away from death events to death decided on day of onset?
            if df.at[individual_id, 'pn_sepsis_late_postpartum'] or df.at[individual_id,
                                                                          'pn_postpartum_haem_secondary']:
                prob_care_seeking = params['pn_linear_equations']['care_seeking_postnatal_complication'].predict(
                    df.loc[[individual_id]])[individual_id]

                if prob_care_seeking < self.module.rng.random_sample():
                    pass
                    # todo: schedule correct HSI


                # TODO: shedule to occur just before week 2 starts?
                # todo: CAVEAT ON CARE SEEKING?
                self.sim.schedule_event(LatePostpartumDeathEvent(self.module, individual_id),
                                        self.sim.date + DateOffset(days=6))

        if self.mother_or_child == 'child':
            logger.debug(key='message', data=f'neonate {individual_id} has developed neonatal sepsis')

            df.at[individual_id, 'pn_sepsis_late_neonatal'] = True
            self.module.postnatal_tracker['late_neonatal_sepsis'] += 1

            self.sim.schedule_event(LatePostpartumDeathEvent(self.module, individual_id),
                                    (self.sim.date + pd.Timedelta(days=6)))

            # todo: additional Care seeking?

class LatePostpartumDeathEvent(Event, IndividualScopeEventMixin):
    """This is LatePostpartumDeathEvent. It is scheduled from SecondaryPostpartumHaemorrhageOnsetEvent,
    LatePostpartumSepsisOnsetEvent or LateNeonatalSepsisOnsetEvent for women and newborns who have developed
    complications in the postpartum period. It uses the linear model to calculte risk of death and schedules death in
    that instance"""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        mother = df.loc[individual_id]

        if df.at[individual_id, 'is_alive']:
            logger.debug(key='message', data=f'mother {individual_id} has reached late postpartum death event')

            postnatal_death = False

            if mother.pn_postpartum_haem_secondary:
                risk_of_death = params['pn_linear_equations']['secondary_postpartum_haem_death'].predict(df.loc[[
                    individual_id]])[individual_id]

                if self.module.rng.random_sample() < risk_of_death:
                    postnatal_death = True
                else:
                    df.at[individual_id, 'pn_postpartum_haem_secondary'] = False

            if mother.pn_sepsis_late_postpartum:
                assert (self.module.postpartum_infections_late.has_any(
                    [individual_id], 'endometritis', 'urinary_tract_inf', 'skin_soft_tissue_inf',
                    'other_maternal_infection', first=True))

                risk_of_death = params['pn_linear_equations']['postnatal_sepsis_death'].predict(df.loc[[
                                individual_id]])[individual_id]

                if self.module.rng.random_sample() < risk_of_death:
                    postnatal_death = True
                else:
                    df.at[individual_id, 'pn_sepsis_late_postpartum'] = False
                    self.module.postpartum_infections_late.unset([individual_id], 'endometritis',
                                                                 'urinary_tract_inf','skin_soft_tissue_inf',
                                                                 'other_maternal_infection')

            if postnatal_death:
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                      cause='secondary_pph'), self.sim.date)
                self.module.postnatal_tracker['postnatal_death'] += 1



        """if self.cause == 'neonatal_sepsis':
                assert df.at[individual_id, 'pn_sepsis_late_neonatal']

                risk_of_death = params['pn_linear_equations']['late_neonatal_sepsis_death'].predict(df.loc[[
                    individual_id]])[individual_id]

                if self.module.rng.random_sample() < risk_of_death:
                    logger.debug(key='message', data=f'person {individual_id} has died due to late neonatal sepsis on '
                                                     f'date {self.sim.date}')

                    self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                          cause='neonatal_sepsis'), self.sim.date)

                    self.module.postnatal_tracker['neonatal_death'] += 1
                    self.module.postnatal_tracker['neonatal_sepsis_death'] += 1

                else:
                    df.at[individual_id, 'pn_sepsis_late_neonatal'] = False"""


class HSI_PostnatalSupervisor_PostnatalCareContactOne(HSI_Event, IndividualScopeEventMixin):
    """This is HSI_PostnatalSupervisor_PostnatalCareContactOneMaternal. It is scheduled by
    HSI_Labour_ReceivesCareForPostpartumPeriod or PostpartumLabourAtHomeEvent. This event is the first PNC visit women
    are reccomended to undertake. If women deliver at home this should be within 12 hours, if in a facility then before
     48 hours. This event is currently unfinished"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, PostnatalSupervisor)

        self.TREATMENT_ID = 'PostnatalSupervisor_PostnatalCareContactOneMaternal'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        child_id = int(df.at[person_id, 'pn_id_most_recent_child'])

        # TODO: currently storing treatment and assessment of sepsis/pph/htn within this HSI but should mya

        assert df.at[person_id, 'la_is_postpartum']
        print(person_id)
        assert df.at[person_id, 'pn_pnc_visits_maternal'] == 0
        assert df.at[child_id, 'pn_pnc_visits_neonatal'] == 0

        if df.at[person_id, 'is_alive']:
            logger.debug(key='message', data=f'Mother {person_id} and child {child_id} have arrived for PNC1 on date'
                                             f' {self.sim.date}')

            maternal_pnc = HSI_PostnatalSupervisor_PostnatalCareContactTwo(
                                                             self.module, person_id=person_id)

            df.at[person_id, 'pn_pnc_visits_maternal'] += 1
            df.at[child_id, 'pn_pnc_visits_neonatal'] += 1

            self.module.maternal_postnatal_care_contact_intervention_bundle(person_id, self)
            self.module.neonatal_postnatal_care_contact_intervention_bundle(child_id, self)
            self.module.maternal_postnatal_care_care_seeking(person_id, 7, 'pnc2', maternal_pnc)


class HSI_PostnatalSupervisor_PostnatalCareContactTwo(HSI_Event, IndividualScopeEventMixin):
    """ This is HSI_PostnatalSupervisor_PostnatalCareContactTwoMaternal. It is scheduled by
    HSI_PostnatalSupervisor_PostnatalCareContactOneMaternal This event is the second PNC visit women
    are recommended to undertake around week 1. This event is currently unfinished"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, PostnatalSupervisor)

        self.TREATMENT_ID = 'PostnatalSupervisor_PostnatalCareContactTwoMaternal'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        child_id = int(df.at[person_id, 'pn_id_most_recent_child'])

        assert df.at[person_id, 'pn_pnc_visits_maternal'] == 1
        assert df.at[child_id, 'pn_pnc_visits_neonatal'] == 1
        assert df.at[person_id, 'la_is_postpartum']

        if df.at[person_id, 'is_alive']:
            logger.debug(key='message', data=f'Mother {person_id} and child {child_id} have arrived for PNC2 on date'
                                             f' {self.sim.date}')

            df.at[person_id, 'pn_pnc_visits_maternal'] += 1
            df.at[child_id, 'pn_pnc_visits_neonatal'] += 1

            maternal_pnc = HSI_PostnatalSupervisor_PostnatalCareContactThree(
                self.module, person_id=person_id)

            self.module.maternal_postnatal_care_contact_intervention_bundle(person_id, self)
            self.module.neonatal_postnatal_care_contact_intervention_bundle(child_id, self)
            self.module.maternal_postnatal_care_care_seeking(person_id, 42, 'pnc3', maternal_pnc)


class HSI_PostnatalSupervisor_PostnatalCareContactThree(HSI_Event, IndividualScopeEventMixin):
    """ This is HSI_PostnatalSupervisor_PostnatalCareContactThreeMaternal. It is scheduled by
    HSI_PostnatalSupervisor_PostnatalCareContactOneMaternal This event is the third PNC visit women
    are recommended to undertake around week 6. This event is currently unfinished"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, PostnatalSupervisor)

        self.TREATMENT_ID = 'PostnatalSupervisor_PostnatalCareContactThreeMaternal'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        child_id = int(df.at[person_id, 'pn_id_most_recent_child'])

        assert df.at[person_id, 'pn_pnc_visits_maternal'] == 2
        assert df.at[child_id, 'pn_pnc_visits_neonatal'] == 2
        assert df.at[person_id, 'la_is_postpartum']

        if df.at[person_id, 'is_alive']:
            logger.debug(key='message', data=f'Mother {person_id} and child {child_id} have arrived for PNC3 on date'
                                             f' {self.sim.date}')

            df.at[person_id, 'pn_pnc_visits_maternal'] += 1
            df.at[child_id, 'pn_pnc_visits_neonatal'] += 1

            self.module.maternal_postnatal_care_contact_intervention_bundle(person_id, self)
            self.module.neonatal_postnatal_care_contact_intervention_bundle(child_id, self)


class HSI_PostnatalSupervisor_PostnatalWardInpatientCare(HSI_Event, IndividualScopeEventMixin):
    """This is HSI_PostnatalSupervisor_InpatientCareForMaternalSepsis. It is scheduled by any of the PNC HSIs for women
    who are assessed as being septic and require treatment as an inpatient"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, PostnatalSupervisor)

        self.TREATMENT_ID = 'PostnatalSupervisor_PostnatalWardInpatientCare'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'IPAdmission': 1})  # TODO: how many days?
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        if not mother.is_alive:
            return

        # --------------------------------------- SEPSIS TREATMENT ---------------------------------------------------
        if mother.pn_sepsis_late_postpartum:
            # First check the availability of consumables for treatment
            pkg_code_sepsis = pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'Maternal sepsis case management',
                                'Intervention_Pkg_Code'])[0]

            consumables_needed_sepsis = {'Intervention_Package_Code': {pkg_code_sepsis: 1}, 'Item_Code': {}}

            outcome_of_request_for_consumables_sep = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self,
                cons_req_as_footprint=consumables_needed_sepsis,
                to_log=False)

            # If available then treatment is delivered and they are logged
            if outcome_of_request_for_consumables_sep:
                logger.debug(key='message',
                             data=f'mother {person_id} has received treatment for sepsis as an inpatient')

                self.sim.modules['HealthSystem'].request_consumables(
                    hsi_event=self,
                    cons_req_as_footprint=consumables_needed_sepsis,
                    to_log=True)

                df.at[person_id, 'pn_sepsis_late_postpartum_treatment'] = True

            else:
                logger.debug(key='message',
                             data=f'mother {person_id} was unable to receive treatment for sepsis due to '
                             f'limited resources')

        # ------------------------------------- SECONDARY PPH TREATMENT -----------------------------------------------
        if mother.pn_postpartum_haem_secondary:
            # todo: check MSTG has specific section on secondary PPH
            # First check the availability of consumables for treatment
            pkg_code_pph = pd.unique(consumables.loc[consumables['Intervention_Pkg'] == 'Treatment of postpartum '
                                                                                        'hemorrhage',
                                                     'Intervention_Pkg_Code'])[0]

            consumables_needed_pph = {'Intervention_Package_Code': {pkg_code_pph: 1}, 'Item_Code': {}}

            outcome_of_request_for_consumables_pph = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self,
                cons_req_as_footprint=consumables_needed_pph,
                to_log=False)

            # If available then treatment is delivered and they are logged
            if outcome_of_request_for_consumables_pph:
                logger.debug(key='message', data=f'mother {person_id} has received treatment for secondary postpartum '
                f'haemorrhage as an inpatient')
                self.sim.modules['HealthSystem'].request_consumables(
                    hsi_event=self,
                    cons_req_as_footprint=consumables_needed_pph,
                    to_log=True)

                df.at[person_id, 'pn_postpartum_haem_secondary_treatment'] = True

            else:
                logger.debug(key='message',
                             data=f'mother {person_id} was unable to receive treatment for secondary pph due '
                             f'to limited resources')


class HSI_PostnatalSupervisor_NeonatalWardInpatientCare(HSI_Event, IndividualScopeEventMixin):
    """"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, PostnatalSupervisor)

        self.TREATMENT_ID = 'PostnatalSupervisor_NeonatalWardInpatientCare'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'IPAdmission': 1})  # TODO: how many days?
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        child = df.loc[person_id]

        if not child.is_alive:
            return

        if child.pn_sepsis_late_neonatal:
            pkg_code_sep = pd.unique(consumables.loc[
                                         consumables['Intervention_Pkg'] == 'Newborn sepsis - full supportive care',
                                         'Intervention_Pkg_Code'])[0]

            consumables_needed = {'Intervention_Package_Code': {pkg_code_sep: 1}, 'Item_Code': {}}

            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self,
                cons_req_as_footprint=consumables_needed,
                to_log=False)

            if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_sep]:
                self.sim.modules['HealthSystem'].request_consumables(
                    hsi_event=self,
                    cons_req_as_footprint=consumables_needed,
                    to_log=True)

                df.at[person_id, 'pn_sepsis_late_neonatal_treatment'] = True

            else:
                logger.debug(key='message', data=f'neonate {person_id} was unable to receive treatment for sepsis due to '
                                                f'limited resources')


class PostnatalLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """"""

    def __init__(self, module):
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(years=self.repeat))

    def apply(self, population):
        df = self.sim.population.props

        # Previous Year...
        one_year_prior = self.sim.date - np.timedelta64(1, 'Y')

        # Denominators...
        total_births_last_year = len(df.index[(df.date_of_birth > one_year_prior) & (df.date_of_birth < self.sim.date)])
        if total_births_last_year == 0:
            total_births_last_year = 1

        ra_lower_limit = 14
        ra_upper_limit = 50
        women_reproductive_age = df.index[(df.is_alive & (df.sex == 'F') & (df.age_years > ra_lower_limit) &
                                           (df.age_years < ra_upper_limit))]
        total_women_reproductive_age = len(women_reproductive_age)

        total_pph = self.module.postnatal_tracker['secondary_pph']
        total_pn_death = self.module.postnatal_tracker['postnatal_death']
        total_pph_death = self.module.postnatal_tracker['secondary_pph_death']
        total_sepsis = self.module.postnatal_tracker['postnatal_sepsis']
        total_sepsis_death = self.module.postnatal_tracker['sepsis_death']
        total_fistula = self.module.postnatal_tracker['fistula']
        total_endo = self.module.postnatal_tracker['endometritis']
        total_uti = self.module.postnatal_tracker['urinary_tract_inf']
        total_ssti = self.module.postnatal_tracker['skin_soft_tissue_inf']
        total_other_inf = self.module.postnatal_tracker['other_maternal_infection']
        total_anaemia = self.module.postnatal_tracker['postnatal_anaemia']

        dict_for_output = {'total_fistula': total_fistula,
                           'total_endo': total_endo,
                           'total_uti': total_uti,
                           'total_ssti': total_ssti,
                           'total_other_inf': total_other_inf,
                           'total_anaemia': total_anaemia,
                           'total_pph': total_pph,
                           'total_pph_death': total_pph_death,
                           'total_sepsis': total_sepsis,
                           'total_sepsis_death': total_sepsis_death,
                           'total_deaths': total_pn_death,
                           'pn_mmr': (total_pn_death/total_births_last_year) * 100000}

        logger.info(key='postnatal_summary_stats', data=dict_for_output, description= 'Yearly summary statistics '
                                                                                      'output from the postnatal '
                                                                                      'supervisor module')

        self.module.postnatal_tracker = {'endometritis': 0, 'urinary_tract_inf': 0, 'skin_soft_tissue_inf': 0,
                                         'other_maternal_infection': 0,'secondary_pph': 0, 'postnatal_death': 0,
                                         'secondary_pph_death': 0,'postnatal_sepsis': 0, 'sepsis_death': 0,
                                         'fistula': 0, 'postnatal_anaemia': 0, 'late_neonatal_sepsis': 0,
                                         'neonatal_death': 0, 'neonatal_sepsis_death': 0}
