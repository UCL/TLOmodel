from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging, util
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel
from tlo.methods import Metadata, postnatal_supervisor_lm
from tlo.methods.causes import Cause
from tlo.methods.dxmanager import DxTest
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.hiv import HSI_Hiv_TestAndRefer
from tlo.util import BitsetHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PostnatalSupervisor(Module):
    """ This module is responsible for the key conditions/complications experienced by a mother and by a neonate
    following labour and the immediate postpartum period (this period, from birth to day +1, is covered by the labour
    module).

    For mothers: This module applies risk of complications across the postnatal period, which is  defined as birth
    until day 42 post-delivery. PostnatalWeekOne Event represents the first week post birth where risk of complications
    remains high. The primary mortality causing complications here are infection/sepsis, secondary postpartum
    haemorrhage and hypertension. Women with or without complications may/may not seek Postnatal Care at day 7 post
    birth (we assume women who seek care will also bring their newborns with them to the HSI where they will also be
    assessed). This HSI assesses mothers and newborns and schedules admissions if complications are found. The
    PostnatalSupervisor Event applies risk of complications weekly from week 2-6 (ending on day 42). This event also
    determines additional care seeking for mothers who are unwell during this time period. All maternal variables are
    reset on day 42.

    For neonates: This module applies risk of complications during the neonatal period, from birth until day 28. The
    PostnatalWeekOne Event applies risk of early onset neonatal sepsis (sepsis onsetting prior to day 7 of life). Care
    may be sought (as described above) and neonates can be admitted for treatment. The PostnatalSupervisor Event applies
     risk of late onset neonatal sepsis from week 2-4 (ending on day 28). This event also determines additional
    care seeking for neonates who are unwell during this time period. All neonatal variables are reset on day 28.
    """

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # This dictionary is used to count each occurrence of an 'event' of interest. These stored counts are used
        # in the LoggingEvent to calculate key outcomes (i.e. incidence rates, neonatal mortality rate etc)
        self.postnatal_tracker = dict()

    INIT_DEPENDENCIES = {'Demography', 'HealthSystem'}

    ADDITIONAL_DEPENDENCIES = {'Labour', 'Lifestyle', 'NewbornOutcomes', 'PregnancySupervisor'}

    METADATA = {Metadata.DISEASE_MODULE,
                Metadata.USES_HEALTHSYSTEM,
                Metadata.USES_HEALTHBURDEN}  # declare that this is a disease module (leave as empty set otherwise)

    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        'maternal': Cause(gbd_causes='Maternal disorders', label='Maternal disorders'),
        'neonatal': Cause(gbd_causes='Neonatal disorders', label='Neonatal Disorders')
    }

    # Declare Causes of Disability
    # todo - distinguish the dalys bewteen maternal and neonatal (here, lumping all under maternal)
    CAUSES_OF_DISABILITY = {
        'maternal': Cause(gbd_causes='Maternal disorders', label='Maternal disorders')
    }

    PARAMETERS = {

        # NATURAL HISTORY PARAMETERS...
        'prob_htn_resolves': Parameter(
            Types.REAL, 'weekly probability hypertension resolves during postpartum'),
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
        'prevalence_type_of_fistula': Parameter(
            Types.LIST, 'prevalence of 1.) vesicovaginal 2.)rectovaginal fistula '),
        'prob_iron_def_per_week_pn': Parameter(
            Types.REAL, 'weekly probability of a women developing iron deficiency following pregnancy'),
        'prob_folate_def_per_week_pn': Parameter(
            Types.REAL, 'weekly probability of a women developing folate deficiency following pregnancy '),
        'prob_b12_def_per_week_pn': Parameter(
            Types.REAL, 'weekly probability of a women developing b12 deficiency following pregnancy '),
        'baseline_prob_anaemia_per_week': Parameter(
            Types.REAL, 'Weekly probability of anaemia in pregnancy'),
        'prob_type_of_anaemia_pn': Parameter(
            Types.LIST, 'probability of a woman with anaemia having mild, moderate or severe anaemia'),
        'rr_anaemia_if_iron_deficient_pn': Parameter(
            Types.REAL, 'risk of developing anaemia when iron deficient'),
        'rr_anaemia_if_folate_deficient_pn': Parameter(
            Types.REAL, 'risk of developing anaemia when folate deficient'),
        'rr_anaemia_if_b12_deficient_pn': Parameter(
            Types.REAL, 'risk of developing anaemia when b12 deficient'),
        'prob_early_onset_neonatal_sepsis_week_1': Parameter(
            Types.REAL, 'Baseline probability of a newborn developing sepsis in week one of life'),
        'cfr_early_onset_neonatal_sepsis': Parameter(
            Types.REAL, 'case fatality for early onset neonatal sepsis'),
        'prob_sepsis_disabilities': Parameter(
            Types.LIST, 'Probabilities of varying disability levels after neonatal sepsis'),
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
        'prob_htn_persists': Parameter(
            Types.REAL, 'Probability that women who are hypertensive during pregnancy remain hypertensive in the '
                        'postnatal period'),
        'weekly_prob_gest_htn_pn': Parameter(
            Types.REAL, 'weekly probability of a woman developing gestational hypertension during the postnatal '
                        'period'),
        'weekly_prob_pre_eclampsia_pn': Parameter(
            Types.REAL, 'weekly probability of a woman developing mild pre-eclampsia during the postnatal period'),
        'probs_for_mgh_matrix_pn': Parameter(
            Types.LIST, 'probability of mild gestational hypertension moving between states: gestational '
                        'hypertension, severe gestational hypertension, mild pre-eclampsia, severe pre-eclampsia, '
                        'eclampsia'),
        'probs_for_sgh_matrix_pn': Parameter(
            Types.LIST, 'probability of severe gestational hypertension moving between states: gestational '
                        'hypertension, severe gestational hypertension, mild pre-eclampsia, severe pre-eclampsia, '
                        'eclampsia'),
        'probs_for_mpe_matrix_pn': Parameter(
            Types.LIST, 'probability of mild pre-eclampsia moving between states: gestational '
                        'hypertension, severe gestational hypertension, mild pre-eclampsia, severe pre-eclampsia, '
                        'eclampsia'),
        'probs_for_spe_matrix_pn': Parameter(
            Types.LIST, 'probability of severe pre-eclampsia moving between states: gestational '
                        'hypertension, severe gestational hypertension, mild pre-eclampsia, severe pre-eclampsia, '
                        'eclampsia'),
        'probs_for_ec_matrix_pn': Parameter(
            Types.LIST, 'probability of eclampsia moving between states: gestational hypertension, severe gestational '
                        'hypertension, mild pre-eclampsia, severe pre-eclampsia, '
                        'eclampsia'),
        'cfr_eclampsia_pn': Parameter(
            Types.REAL, 'case fatality rate of eclampsia in the postnatal period'),
        'cfr_severe_htn_pn': Parameter(
            Types.REAL, 'case fatality rate of severe hypertension in the postnatal period'),
        'weekly_prob_postnatal_death': Parameter(
            Types.REAL, 'Weekly risk of postnatal death'),
        'severity_late_infection_pn': Parameter(
            Types.LIST, 'probability of mild infection, sepsis or severe sepsis in the later postnatal period'),

        # TREATMENT/HEALTH SYSTEM PARAMETERS
        'rr_iron_def_ifa_pn': Parameter(
            Types.REAL, 'effect of iron and folic acid treatment on risk of iron deficiency'),
        'rr_folate_def_ifa_pn': Parameter(
            Types.REAL, 'effect of iron and folic acid treatment on risk of folate deficiency '),
        'treatment_effect_early_init_bf': Parameter(
            Types.REAL, 'effect of early initiation of breastfeeding on neonatal sepsis rates '),
        'treatment_effect_abx_prom': Parameter(
            Types.REAL, 'effect of early antibiotics given to a mother with PROM on neonatal sepsis rates '),
        'treatment_effect_inj_abx_sep': Parameter(
            Types.REAL, 'effect of injectable antibiotics on neonatal sepsis mortality'),
        'treatment_effect_supp_care_sep': Parameter(
            Types.REAL, 'effect of full supportive care on neonatal sepsis mortality'),
        'treatment_effect_parenteral_antibiotics': Parameter(
            Types.REAL, 'Treatment effect of parenteral antibiotics on maternal sepsis mortality '),
        'treatment_effect_bemonc_care_pph': Parameter(
            Types.REAL, 'Treatment effect of BEmONC care on postpartum haemorrhage mortality'),
        'treatment_effect_anti_htns': Parameter(
            Types.REAL, 'Treatment effect of hypertensive therapy on death from eclampsia'),
        'treatment_effect_mag_sulph': Parameter(
            Types.REAL, 'Treatment effect of magnesium sulphate therapy on death from eclampsia'),
        'neonatal_sepsis_treatment_effect': Parameter(
            Types.REAL, 'Treatment effect for neonatal sepsis'),
        'treatment_effect_clean_birth': Parameter(
            Types.REAL, 'Treatment effect of clean birth practices on early onset neonatal sepsis risk'),
        'treatment_effect_cord_care': Parameter(
            Types.REAL, 'Treatment effect of chlorhexidine cord care on early onset neonatal sepsis risk'),
        'treatment_effect_anti_htns_progression_pn': Parameter(
            Types.REAL, 'Treatment effect of oral anti hypertensives on progression from mild/mod to severe gestational'
                        'hypertension'),
        'prob_attend_pnc2': Parameter(
            Types.REAL, 'Probability that a woman receiving PNC1 care will return for PNC2 care'),
        'prob_attend_pnc3': Parameter(
            Types.REAL, 'Probability that a woman receiving PNC2 care will return for PNC3 care'),
        'prob_care_seeking_postnatal_emergency': Parameter(
            Types.REAL, 'baseline probability of emergency care seeking for women in the postnatal period'),
        'prob_care_seeking_postnatal_emergency_neonate': Parameter(
            Types.REAL, 'baseline probability care will be sought for a neonate with a complication'),
        'odds_care_seeking_fistula_repair': Parameter(
            Types.REAL, 'odds of a woman seeking care for treatment of obstetric fistula'),
        'aor_cs_fistula_age_15_19': Parameter(
            Types.REAL, 'odds ratio for care seeking for treatment of obstetric fistula in 15-19 year olds'),
        'aor_cs_fistula_age_lowest_education': Parameter(
            Types.REAL, 'odds ratio for care seeking for treatment of obstetric fistula in women in the lowest '
                        'education quintile'),
        'prob_pnc1_at_day_7': Parameter(
            Types.REAL, 'baseline probability a woman will seek PNC for her and her newborn at day + 7 '),
        'multiplier_for_care_seeking_with_comps': Parameter(
            Types.REAL, 'number by which prob_pnc1_at_day_7 is multiplied by to increase care seeking for PNC1 in women'
                        ' with complications '),
        'sensitivity_bp_monitoring_pn': Parameter(
            Types.REAL, 'sensitivity of BP monitoring during PNC'),
        'specificity_bp_monitoring_pn': Parameter(
            Types.REAL, 'specificity of BP monitoring during PNC'),
        'sensitivity_urine_protein_1_plus_pn': Parameter(
            Types.REAL, 'sensitivity of urine dipstick during PNC'),
        'specificity_urine_protein_1_plus_pn': Parameter(
            Types.REAL, 'specificity of urine dipstick monitoring during PNC'),
        'sensitivity_poc_hb_test_pn': Parameter(
            Types.REAL, 'sensitivity of point of care testing during PNC'),
        'specificity_poc_hb_test_pn': Parameter(
            Types.REAL, 'specificity of urine dipstick monitoring during PNC'),
        'sensitivity_maternal_sepsis_assessment': Parameter(
            Types.REAL, 'sensitivity of assessment for maternal sepsis'),
        'sensitivity_pph_assessment': Parameter(
            Types.REAL, 'sensitivity of assessment for secondary pph'),
        'sensitivity_lons_assessment': Parameter(
            Types.REAL, 'sensitivity of assessment for late onset neonatal sepsis'),
        'sensitivity_eons_assessment': Parameter(
            Types.REAL, 'sensitivity of assessment for earl onset neonatal sepsis'),
        'prob_intervention_delivered_sep_assessment_pnc': Parameter(
            Types.REAL, 'probability a woman will be assessed for sepsis during PNC'),
        'prob_intervention_delivered_pph_assessment_pnc': Parameter(
            Types.REAL, 'probability a woman will be assessed for PPH during PNC'),
        'prob_intervention_delivered_urine_ds_pnc': Parameter(
            Types.REAL, 'probability a woman will receive a urine disptick during PNC'),
        'prob_intervention_delivered_bp_pnc': Parameter(
            Types.REAL, 'probability a woman will receive blood pressure testing during PNC'),
        'prob_intervention_poct_pnc': Parameter(
            Types.REAL, 'probability a woman will receive point of care Hb testing during PNC'),
        'prob_intervention_neonatal_sepsis_pnc': Parameter(
            Types.REAL, 'probability a newborn will be assessed for sepsis during PNC'),
        'prob_intervention_delivered_hiv_test_pnc': Parameter(
            Types.REAL, 'probability a newborn will receive a HIV test at 6 weeks postnatal '),
        'prob_intervention_delivered_depression_screen_pnc': Parameter(
            Types.REAL, 'probability a mother will receive a depression screen during PNC'),
    }

    PROPERTIES = {
        'pn_id_most_recent_child': Property(Types.INT, 'person_id of a mothers most recent child (if twins id of the '
                                                       'first born)'),
        'pn_postnatal_period_in_weeks': Property(Types.REAL, 'The number of weeks a woman is in the postnatal period '
                                                             '(1-6)'),
        'pn_pnc_visits_maternal': Property(Types.INT, 'The number of postnatal care visits a woman has undergone '
                                                      'following her most recent delivery'),
        'pn_pnc_visits_neonatal': Property(Types.INT, 'The number of postnatal care visits a neonate has undergone '
                                                      'following delivery'),
        'pn_htn_disorders': Property(Types.CATEGORICAL, 'Hypertensive disorders of the postnatal period',
                                     categories=['none', 'resolved', 'gest_htn', 'severe_gest_htn', 'mild_pre_eclamp',
                                                 'severe_pre_eclamp', 'eclampsia']),
        'pn_mag_sulph_treatment': Property(Types.BOOL, 'Whether this woman has received magnesium sulphate as '
                                                       'treatment for eclampsia/ severe pre-eclampsia'),
        'pn_gest_htn_on_treatment': Property(Types.BOOL, 'Whether this woman is receiving regular oral medication for '
                                                         'hypertension in the postnatal period'),
        'pn_iv_anti_htn_treatment': Property(Types.BOOL, 'Whether this woman has received IV anti hypertensive therapy '
                                                         'during a hypertensive emergency'),
        'pn_postpartum_haem_secondary': Property(Types.BOOL, 'Whether this woman is experiencing a secondary '
                                                             'postpartum haemorrhage'),
        'pn_postpartum_haem_secondary_treatment': Property(Types.BOOL, 'Whether this woman has received treatment for '
                                                                       'secondary PPH'),
        'pn_sepsis_late_postpartum': Property(Types.BOOL, 'Whether this woman is experiencing postnatal (day7+) '
                                                          'sepsis'),
        'pn_sepsis_late_postpartum_treatment': Property(Types.BOOL, 'Whether this woman has received treatment for '
                                                                    'postpartum sepsis'),
        'pn_maternal_pp_infection': Property(Types.INT, 'bitset column for infection'),
        'pn_obstetric_fistula': Property(Types.CATEGORICAL, 'Type of fistula developed after birth',
                                         categories=['none', 'vesicovaginal', 'rectovaginal']),
        'pn_sepsis_early_neonatal': Property(Types.BOOL, 'Whether this neonate has developed early onset neonatal'
                                                         ' sepsis during week one of life'),
        'pn_sepsis_late_neonatal': Property(Types.BOOL, 'Whether this neonate has developed late neonatal sepsis '
                                                        'following discharge'),
        'pn_neonatal_sepsis_disab': Property(Types.CATEGORICAL, 'Level of disability experience from a neonate post '
                                                                'sepsis', categories=['none', 'mild_motor_and_cog',
                                                                                      'mild_motor', 'moderate_motor',
                                                                                      'severe_motor']),
        'pn_sepsis_neonatal_inj_abx': Property(Types.BOOL, 'Whether this neonate has received injectable antibiotics'
                                                           ' as treatment for late onset sepsis'),
        'pn_sepsis_neonatal_full_supp_care': Property(Types.BOOL, 'Whether this neonate has received full '
                                                                  'supportive care as treatment for late onset sepsis'),
        'pn_deficiencies_following_pregnancy': Property(Types.INT, 'bitset column, stores types of anaemia causing '
                                                                   'deficiencies following pregnancy'),
        'pn_anaemia_following_pregnancy': Property(Types.CATEGORICAL, 'severity of anaemia following pregnancy',
                                                   categories=['none', 'mild', 'moderate', 'severe']),
        'pn_emergency_event_mother': Property(Types.BOOL, 'Whether a mother is experiencing an emergency complication'
                                                          ' postnatally'),
    }

    def read_parameters(self, data_folder):
        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_PostnatalSupervisor.xlsx',
                            sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)

    def initialise_population(self, population):

        df = population.props

        df.loc[df.is_alive, 'pn_id_most_recent_child'] = -1
        df.loc[df.is_alive, 'pn_postnatal_period_in_weeks'] = 0
        df.loc[df.is_alive, 'pn_pnc_visits_maternal'] = 0
        df.loc[df.is_alive, 'pn_pnc_visits_neonatal'] = 0
        df.loc[df.is_alive, 'pn_htn_disorders'] = 'none'
        df.loc[df.is_alive, 'pn_mag_sulph_treatment'] = False
        df.loc[df.is_alive, 'pn_gest_htn_on_treatment'] = False
        df.loc[df.is_alive, 'pn_iv_anti_htn_treatment'] = False
        df.loc[df.is_alive, 'pn_postpartum_haem_secondary'] = False
        df.loc[df.is_alive, 'pn_postpartum_haem_secondary_treatment'] = False
        df.loc[df.is_alive, 'pn_sepsis_late_postpartum'] = False
        df.loc[df.is_alive, 'pn_neonatal_sepsis_disab'] = 'none'
        df.loc[df.is_alive, 'pn_sepsis_early_neonatal'] = False
        df.loc[df.is_alive, 'pn_sepsis_late_neonatal'] = False
        df.loc[df.is_alive, 'pn_sepsis_late_neonatal'] = False
        df.loc[df.is_alive, 'pn_sepsis_neonatal_inj_abx'] = False
        df.loc[df.is_alive, 'pn_sepsis_neonatal_full_supp_care'] = False
        df.loc[df.is_alive, 'pn_anaemia_following_pregnancy'] = 'none'
        df.loc[df.is_alive, 'pn_obstetric_fistula'] = 'none'
        df.loc[df.is_alive, 'pn_emergency_event_mother'] = False

        # This biset property stores infections that can occur in the postnatal period
        self.postpartum_infections_late = BitsetHandler(self.sim.population, 'pn_maternal_pp_infection',
                                                        ['endometritis', 'urinary_tract_inf', 'skin_soft_tissue_inf',
                                                         'other_maternal_infection'])

        # This biset property deficiencies that can lead to anaemia
        self.deficiencies_following_pregnancy = BitsetHandler(self.sim.population,
                                                              'pn_deficiencies_following_pregnancy',
                                                              ['iron', 'folate', 'b12'])

    def initialise_simulation(self, sim):

        # Schedule the first instance of the PostnatalSupervisorEvent
        sim.schedule_event(PostnatalSupervisorEvent(self),
                           sim.date + DateOffset(days=0))

        # Register logging event
        sim.schedule_event(PostnatalLoggingEvent(self),
                           sim.date + DateOffset(years=1))

        # Define the events we want to track in the postnatal_tracker...
        self.postnatal_tracker = {'endometritis': 0, 'urinary_tract_inf': 0, 'skin_soft_tissue_inf': 0,
                                  'other_maternal_infection': 0, 'secondary_pph': 0, 'postnatal_death': 0,
                                  'postnatal_sepsis': 0, 'fistula': 0, 'postnatal_anaemia': 0,
                                  'early_neonatal_sepsis': 0, 'late_neonatal_sepsis': 0, 'neonatal_death': 0,
                                  'neonatal_sepsis_death': 0}

        # Register dx_tests used as assessment for postnatal conditions during PNC visits

        params = self.parameters

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(

            # This dx_test represents blood pressure measurement delivered during PNC
            blood_pressure_measurement_pn=DxTest(
                property='pn_htn_disorders', target_categories=['gest_htn', 'mild_pre_eclamp', 'severe_gest_htn',
                                                                'severe_pre_eclamp', 'eclampsia'],
                sensitivity=params['sensitivity_bp_monitoring_pn'],
                specificity=params['specificity_bp_monitoring_pn']),

            # This test represents a urine dipstick which is used to measuring the presence and amount of protein in a
            # womans urine, proteinuria being indicative of pre-eclampsia/eclampsia
            urine_dipstick_protein_pn=DxTest(
                property='pn_htn_disorders', target_categories=['mild_pre_eclamp', 'severe_pre_eclamp', 'eclampsia'],
                sensitivity=params['sensitivity_urine_protein_1_plus_pn'],
                specificity=params['specificity_urine_protein_1_plus_pn']),

            # This test represents point of care haemoglobin testing used in PNC to detect anaemia (all-severity)
            point_of_care_hb_test_pn=DxTest(
                property='pn_anaemia_following_pregnancy', target_categories=['mild', 'moderate', 'severe'],
                sensitivity=params['sensitivity_poc_hb_test_pn'],
                specificity=params['specificity_poc_hb_test_pn']),

            # This test represents clinical assessment of a mother for signs of sepsis
            assessment_for_postnatal_sepsis=DxTest(
                property='pn_sepsis_late_postpartum',
                sensitivity=params['sensitivity_maternal_sepsis_assessment']),

            # This test represents clinical assessment of a mother for signs of secondary postpartum bleeding
            assessment_for_secondary_pph=DxTest(
                property='pn_postpartum_haem_secondary',
                sensitivity=params['sensitivity_pph_assessment']),

            # This test represents clinical assessment of a newborn for signs of later onset neonatal sepsis
            assessment_for_late_onset_neonatal_sepsis=DxTest(
                property='pn_sepsis_late_neonatal',
                sensitivity=params['sensitivity_lons_assessment']),

            # This test represents clinical assessment of a newborn for signs of early onset neonatal sepsis
            assessment_for_early_onset_neonatal_sepsis=DxTest(
                property='pn_sepsis_early_neonatal',
                sensitivity=params['sensitivity_eons_assessment']),
        )

        # ======================================= LINEAR MODEL EQUATIONS =============================================
        # All linear equations used in this module are stored within the pn_linear_equations
        # parameter below

        # TODO: process of 'selection' of important predictors in linear equations is ongoing, a linear model that
        #  is empty of predictors at the end of this process will be converted to a set probability

        params['pn_linear_equations'] = {

            # This equation is used to determine a mothers risk of developing obstetric fistula after birth
            'obstetric_fistula': LinearModel.custom(postnatal_supervisor_lm.predict_obstetric_fistula,
                                                    parameters=params),

            # This equation is used to determine a mothers risk of secondary postpartum haemorrhage
            'secondary_postpartum_haem': LinearModel.custom(postnatal_supervisor_lm.predict_secondary_postpartum_haem,
                                                            parameters=params),

            # This equation is used to determine a mothers risk of dying following a secondary postpartum haemorrhage.
            # Risk of death is modified by the effect of treatment, if delivered
            'secondary_postpartum_haem_death': LinearModel.custom(
                postnatal_supervisor_lm.predict_secondary_postpartum_haem_death, parameters=params),

            # This equation is used to determine a mothers risk of developing endometritis infection
            'endometritis': LinearModel.custom(postnatal_supervisor_lm.predict_endometritis, parameters=params),

            # This equation is used to determine a mothers risk of developing a urinary tract infection
            'urinary_tract_inf': LinearModel.custom(postnatal_supervisor_lm.predict_urinary_tract_inf,
                                                    parameters=params),

            # This equation is used to determine a mothers risk of developing a skin or soft tissue infection
            'skin_soft_tissue_inf': LinearModel.custom(postnatal_supervisor_lm.predict_skin_soft_tissue_inf,
                                                       parameters=params),

            # This equation is used to determine a mothers risk of developing another infection, not defined above
            'other_maternal_infection': LinearModel.custom(postnatal_supervisor_lm.predict_other_maternal_infection,
                                                           parameters=params),

            # This equation is used to determine a mothers risk of developing sepsis following one of more of the above
            # infections

            'sepsis_late_postpartum': LinearModel.custom(postnatal_supervisor_lm.predict_sepsis_late_postpartum,
                                                         module=self),

            # This equation is used to determine a mothers risk of dying due to sepsis in the postnatal period. Risk of
            # death is modified by the effect of treatment, if delivered
            'postnatal_sepsis_death': LinearModel.custom(postnatal_supervisor_lm.predict_postnatal_sepsis_death,
                                                         parameters=params),

            # This equation is used to determine a mothers risk of developing gestational hypertension in the postnatal
            # period
            'gest_htn_pn': LinearModel.custom(postnatal_supervisor_lm.predict_gest_htn_pn, parameters=params),

            # This equation is used to determine a mothers risk of developing pre-eclampsia in in the postnatal
            # period
            'pre_eclampsia_pn': LinearModel.custom(postnatal_supervisor_lm.predict_pre_eclampsia_pn, parameters=params),

            # This equation is used to determine a mothers risk of dying from eclampsia that has developed in the
            # postnatal period. Risk of death is mitigated by treatment effects, if treatment is delivered
            'eclampsia_death_pn': LinearModel.custom(postnatal_supervisor_lm.predict_eclampsia_death_pn,
                                                     parameters=params),

            # This equation is used to determine a mothers risk of dying due to severe hypertension
            'death_from_hypertensive_disorder_pn': LinearModel.custom(
                postnatal_supervisor_lm.predict_death_from_hypertensive_disorder_pn, parameters=params),

            # This equation is used to determine a mothers risk of developing anaemia postnatal
            'anaemia_after_pregnancy': LinearModel.custom(postnatal_supervisor_lm.predict_anaemia_after_pregnancy,
                                                          module=self),

            # This equation is used to determine a neonates risk of developing early onset neonatal sepsis
            # (sepsis onsetting prior to day 7) in the first week of life
            'early_onset_neonatal_sepsis_week_1': LinearModel.custom(
                postnatal_supervisor_lm.predict_early_onset_neonatal_sepsis_week_1, parameters=params),

            # This equation is used to determine a neonates risk of dying following early onset sepsis in week one.
            # Risk of death is mitigated by treatment effects, if delivered
            'early_onset_neonatal_sepsis_week_1_death': LinearModel.custom(
                postnatal_supervisor_lm.predict_early_onset_neonatal_sepsis_week_1_death, parameters=params),

            # This equation is used to determine a neonates risk of developing late onset neonatal sepsis
            # (sepsis onsetting between 7 and day 28) after  the first week of life
            'late_onset_neonatal_sepsis': LinearModel.custom(
                postnatal_supervisor_lm.predict_late_onset_neonatal_sepsis, parameters=params),

            # This equation is used to determine a neonates risk of dying following late onset neonatal sepsis
            # (sepsis onsetting between 7 and day 28) after the first week of life
            'late_neonatal_sepsis_death': LinearModel.custom(
                postnatal_supervisor_lm.predict_late_neonatal_sepsis_death, parameters=params),

            # This equation is used to determine the probability that a woman will seek care for PNC after delivery
            'care_seeking_for_first_pnc_visit': LinearModel.custom(
                postnatal_supervisor_lm.predict_care_seeking_for_first_pnc_visit, parameters=params),

            # This equation is used to determine if a mother will seek care for treatment in the instance of an
            # emergency complication postnatally (sepsis or haemorrhage)
            'care_seeking_postnatal_complication_mother': LinearModel.custom(
                postnatal_supervisor_lm.predict_care_seeking_postnatal_complication_mother, parameters=params),

            # This equation is used to determine if a mother will seek care for treatment for her newborn in the
            # instance of them developing an emergency complication postnatally (sepsis)
            'care_seeking_postnatal_complication_neonate': LinearModel.custom(
                postnatal_supervisor_lm.predict_care_seeking_postnatal_complication_neonate, parameters=params),

            # This equation is used to determine if a mother will seek care for treatment for her obstetric fistula
            'care_seeking_for_fistula_repair': LinearModel.custom(
                postnatal_supervisor_lm.predict_care_seeking_for_fistula_repair, parameters=params),

        }

        if 'Hiv' not in self.sim.modules:
            logger.warning(key='message', data='HIV module is not registered in this simulation run and therefore HIV '
                                               'testing will not happen in postnatal care')

    def on_birth(self, mother_id, child_id):
        df = self.sim.population.props

        df.at[child_id, 'pn_id_most_recent_child'] = -1
        df.at[child_id, 'pn_postnatal_period_in_weeks'] = 0
        df.at[child_id, 'pn_pnc_visits_maternal'] = 0
        df.at[child_id, 'pn_pnc_visits_neonatal'] = 0
        df.at[child_id, 'pn_htn_disorders'] = 'none'
        df.at[child_id, 'pn_mag_sulph_treatment'] = False
        df.at[child_id, 'pn_gest_htn_on_treatment'] = False
        df.at[child_id, 'pn_iv_anti_htn_treatment'] = False
        df.at[child_id, 'pn_postpartum_haem_secondary'] = False
        df.at[child_id, 'pn_postpartum_haem_secondary_treatment'] = False
        df.at[child_id, 'pn_sepsis_late_postpartum'] = False
        df.at[child_id, 'pn_sepsis_late_postpartum_treatment'] = False
        df.at[child_id, 'pn_sepsis_early_neonatal'] = False
        df.at[child_id, 'pn_sepsis_late_neonatal'] = False
        df.at[child_id, 'pn_neonatal_sepsis_disab'] = 'none'
        df.at[child_id, 'pn_sepsis_neonatal_inj_abx'] = False
        df.at[child_id, 'pn_sepsis_neonatal_full_supp_care'] = False
        df.at[child_id, 'pn_obstetric_fistula'] = 'none'
        df.at[child_id, 'pn_anaemia_following_pregnancy'] = 'none'
        df.at[child_id, 'pn_emergency_event_mother'] = False

    def further_on_birth_postnatal_supervisor(self, mother_id, child_id):
        df = self.sim.population.props
        params = self.parameters
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        store_dalys_in_mni = self.sim.modules['PregnancySupervisor'].store_dalys_in_mni

        # We store the ID number of the child this woman has most recently delivered as a property of the woman. This is
        # because PNC is scheduled for the woman during the Labour Module but must act on both mother and child. If this
        # is a twin birth, only the id of the first born child is stored
        df.at[mother_id, 'pn_id_most_recent_child'] = child_id

        if df.at[mother_id, 'is_alive']:
            # Here we determine if, following childbirth, this woman will develop a fistula
            risk_of_fistula = params['pn_linear_equations'][
                'obstetric_fistula'].predict(df.loc[[mother_id]])[mother_id]

            # todo: link with PNC (assessment), allow treatment post day 42 (dalys?)

            if self.rng.random_sample() < risk_of_fistula:
                # We determine the specific type of fistula this woman is experiencing, to match with DALY weights
                fistula_type = self.rng.choice(['vesicovaginal', 'rectovaginal'],
                                               p=params['prevalence_type_of_fistula'])
                df.at[mother_id, 'pn_obstetric_fistula'] = fistula_type

                logger.debug(key='msg', data=f'Mother {mother_id} has developed a {fistula_type} post delivery')

                # Store the onset weight for daly calculations
                store_dalys_in_mni(mother_id, f'{fistula_type}_fistula_onset')
                self.postnatal_tracker['fistula'] += 1

                # Determine if she will seek care for repair
                care_seeking_for_repair = params['pn_linear_equations'][
                    'care_seeking_for_fistula_repair'].predict(df.loc[[mother_id]])[mother_id]

                # Schedule repair to occur for 1 week postnatal
                if care_seeking_for_repair:
                    repair_hsi = HSI_PostnatalSupervisor_TreatmentForObstetricFistula(
                        self, person_id=mother_id)
                    repair_date = self.sim.date + DateOffset(days=(self.rng.randint(7, 42)))

                    logger.debug(key='msg', data=f'Mother {mother_id} will seek care for fistula repair on '
                                                 f'{repair_date}')
                    self.sim.modules['HealthSystem'].schedule_hsi_event(repair_hsi,
                                                                        priority=0,
                                                                        topen=repair_date,
                                                                        tclose=repair_date + DateOffset(days=7))

            # ======================= CONTINUATION OF COMPLICATIONS INTO THE POSTNATAL PERIOD =========================
            # Certain conditions experienced in pregnancy are liable to continue into the postnatal period

            # HYPERTENSIVE DISORDERS...
            # The majority of hypertension related to pregnancy resolve with delivery of the foetus. However the
            # condition
            # may persist (and even onset within the postnatal period...)
            if df.at[mother_id, 'ps_htn_disorders'] != 'none':
                if self.rng.random_sample() < params['prob_htn_persists']:
                    logger.debug(key='message', data=f'mother {mother_id} will remain hypertensive despite '
                                                     f'successfully delivering')
                    df.at[mother_id, 'pn_htn_disorders'] = df.at[mother_id, 'ps_htn_disorders']

                else:
                    if not pd.isnull(mni[mother_id]['hypertension_onset']):
                        store_dalys_in_mni(mother_id, 'hypertension_resolution')
                    df.at[mother_id, 'pn_htn_disorders'] = 'resolved'

            # Currently we assume women who received antihypertensive in the antenatal period will continue to use them
            if df.at[mother_id, 'ac_gest_htn_on_treatment']:
                df.at[mother_id, 'pn_gest_htn_on_treatment'] = True

            #  DEFICIENCIES/ANAEMIA...
            # We carry across any deficiencies that may increase this womans risk of postnatal anaemia
            if self.sim.modules['PregnancySupervisor'].deficiencies_in_pregnancy.has_any([mother_id], 'iron',
                                                                                         first=True):
                self.deficiencies_following_pregnancy.set([mother_id], 'iron')
            if self.sim.modules['PregnancySupervisor'].deficiencies_in_pregnancy.has_any([mother_id], 'folate',
                                                                                         first=True):
                self.deficiencies_following_pregnancy.set([mother_id], 'folate')
            if self.sim.modules['PregnancySupervisor'].deficiencies_in_pregnancy.has_any([mother_id], 'b12',
                                                                                         first=True):
                self.deficiencies_following_pregnancy.set([mother_id], 'b12')

            # And similarly, if she is already anaemic then she remains so in the postnatal period
            if df.at[mother_id, 'ps_anaemia_in_pregnancy'] != 'none':
                df.at[mother_id, 'pn_anaemia_following_pregnancy'] = df.at[mother_id, 'ps_anaemia_in_pregnancy']

            # Finally we call a function in the PregnancySupervisor module to reset the variables from pregnancy
            self.sim.modules['PregnancySupervisor'].pregnancy_supervisor_property_reset(
                ind_or_df='individual', id_or_index=mother_id)

    def on_hsi_alert(self, person_id, treatment_id):
        logger.debug(key='message', data=f'This is PostnatalSupervisor, being alerted about a health system '
                                         f'interaction person {person_id} for: {treatment_id}')

    def report_daly_values(self):
        logger.debug(key='message', data='This is PostnatalSupervisor reporting my health values')
        df = self.sim.population.props

        daly_series = pd.Series(data=0, index=df.index[df.is_alive])

        return daly_series

    def set_infections(self, individual_id, infection):
        """
        This function is called by the PostnatalWeekOne event to calculate a womans risk of developing an infection
        in the first week after birth and store that infection.
        :param individual_id: individual_id
        :param infection: the infection for which risk of infection is being determined
        """
        df = self.sim.population.props
        params = self.parameters

        # Individual risk is calculated via the linear model
        risk_infection = params['pn_linear_equations'][f'{infection}'].predict(df.loc[[
            individual_id]])[individual_id]

        # If the infection will happen, it is stored in the bit set property and tracked
        if risk_infection > self.rng.random_sample():
            logger.debug(key='msg', data=f'Mother {individual_id} has developed {infection} infection in week one of '
                                         f'the postnatal period')
            self.postpartum_infections_late.set([individual_id], f'{infection}')
            self.postnatal_tracker[f'{infection}'] += 1

    def set_postnatal_complications_mothers(self, week):
        """
        This function is called by the PostnatalSupervisor event. It applies risk of key complications to a subset of
        women during each week of the postnatal period starting from week 2. Currently this includes infection, sepsis,
        anaemia and hypertension
        :param week: week in the postnatal period used to select women in the data frame.
         """
        df = self.sim.population.props
        params = self.parameters
        apply_linear_model = self.sim.modules['PregnancySupervisor'].apply_linear_model
        store_dalys_in_mni = self.sim.modules['PregnancySupervisor'].store_dalys_in_mni
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        def onset(eq):
            """
            Runs a specific equation within the linear model for the appropriate subset of women in the postnatal period
             and returns a BOOL series
            :param eq: linear model equation
            :return: BOOL series
            """
            onset_condition = apply_linear_model(
                params['pn_linear_equations'][f'{eq}'],
                df.loc[df['is_alive'] & df['la_is_postpartum'] & (df['pn_postnatal_period_in_weeks'] == week) &
                       ~df['hs_is_inpatient']])
            return onset_condition

        # -------------------------------------- INFECTIONS ---------------------------------------------------------
        # First we use the onset function to determine any women will develop any infections that will precede
        # sepsis at this point in their postnatal period
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
            self.postpartum_infections_late.set(onset_other_inf.loc[onset_other_inf].index, 'other_maternal_infection')

        # -------------------------------------- SEPSIS --------------------------------------------------------------
        # Next we run the linear model to see if any of the women who developed infections will lead go on to develop
        # maternal postnatal sepsis
        onset_sepsis = onset('sepsis_late_postpartum')
        df.loc[onset_sepsis.loc[onset_sepsis].index, 'pn_sepsis_late_postpartum'] = True

        # If sepsis develops we use this property to denote that these women are experiencing an emergency and may need
        # to seek care
        df.loc[onset_sepsis.loc[onset_sepsis].index, 'pn_emergency_event_mother'] = True
        for person in onset_sepsis.loc[onset_sepsis].index:
            store_dalys_in_mni(person, 'sepsis_onset')

        if not onset_sepsis.loc[onset_sepsis].empty:
            logger.debug(key='message', data=f'The following women have developed sepsis during week {week} of '
                                             f'the postnatal period, {onset_sepsis.loc[onset_sepsis].index}')
            self.postnatal_tracker['postnatal_sepsis'] += len(onset_sepsis.loc[onset_sepsis])

        # ------------------------------------ SECONDARY PPH ----------------------------------------------------------
        # Next we determine if any women will experience postnatal bleeding
        onset_pph = onset('secondary_postpartum_haem')
        df.loc[onset_pph.loc[onset_pph].index, 'pn_postpartum_haem_secondary'] = True

        # And set the emergency property
        df.loc[onset_pph.loc[onset_pph].index, 'pn_emergency_event_mother'] = True
        for person in onset_pph.loc[onset_pph].index:
            store_dalys_in_mni(person, 'secondary_pph_onset')

        if not onset_pph.loc[onset_pph].empty:
            logger.debug(key='message', data=f'The following women have developed secondary pph during week {week}'
                                             f' of the postnatal period, {onset_pph.loc[onset_pph].index}')
            self.postnatal_tracker['secondary_pph'] += len(onset_pph.loc[onset_pph])

        # ---------------------------------- DEFICIENCIES AND ANAEMIA --------------------------------------------------
        def apply_risk(deficiency):
            if deficiency == 'iron' or deficiency == 'folate':
                # First we select a subset of the pregnant population who are not suffering from the deficiency in
                # question. (When applying risk of iron/folate deficiency we fist apply risk to women not on iron/folic
                # acid treatment)
                selected_women = ~self.deficiencies_following_pregnancy.has_all(
                        df.is_alive & df.la_is_postpartum & ~df.la_iron_folic_acid_postnatal &
                        (df.pn_postnatal_period_in_weeks == week), deficiency)

            else:
                # As IFA treatment does not effect B12 we select the appropriate women regardless of IFA treatment
                # status
                selected_women = ~self.deficiencies_following_pregnancy.has_all(
                        df.is_alive & df.la_is_postpartum & (df.pn_postnatal_period_in_weeks == week), deficiency)

            # We determine their risk of deficiency
            new_def = pd.Series(self.rng.random_sample(len(selected_women)) < params[f'prob_{deficiency}_def_per'
                                                                                     f'_week_pn'],
                                index=selected_women.index)

            # And change their property accordingly
            self.deficiencies_following_pregnancy.set(new_def.loc[new_def].index, deficiency)

            if deficiency == 'b12':
                return
            else:
                # Next we select women who aren't deficient of iron/folate but are receiving IFA treatment
                def_treatment = ~self.deficiencies_following_pregnancy.has_all(
                        df.is_alive & df.la_is_postpartum & df.la_iron_folic_acid_postnatal &
                        (df.pn_postnatal_period_in_weeks == week), deficiency)

                # We reduce their individual risk of deficiencies due to treatment and make changes to the data frame
                risk_of_def = params[f'prob_{deficiency}_def_per_week_pn'] * params[f'rr_{deficiency}_def_ifa_pn']
                new_def = pd.Series(self.rng.random_sample(len(def_treatment)) < risk_of_def, index=def_treatment.index)

                self.deficiencies_following_pregnancy.set(new_def.loc[new_def].index, deficiency)

        # Now we run the function for each
        for deficiency in ['iron', 'folate', 'b12']:
            apply_risk(deficiency)

        # ----------------------------------------- ANAEMIA ----------------------------------------------------------
        # Then we apply a risk of anaemia developing in this week, and determine its severity
        onset_anaemia = onset('anaemia_after_pregnancy')
        random_choice_severity = pd.Series(self.rng.choice(['mild', 'moderate', 'severe'],
                                                           p=params['prob_type_of_anaemia_pn'],
                                                           size=len(onset_anaemia.loc[onset_anaemia])),
                                           index=onset_anaemia.loc[onset_anaemia].index)

        df.loc[onset_anaemia.loc[onset_anaemia].index, 'pn_anaemia_following_pregnancy'] = random_choice_severity

        for person in onset_anaemia.loc[onset_anaemia].index:
            store_dalys_in_mni(person, f'{df.at[person, "pn_anaemia_following_pregnancy"]}_anaemia_pp_onset')

        if not onset_anaemia.loc[onset_anaemia].empty:
            logger.debug(key='message', data=f'The following women have developed anaemia during week {week}'
                                             f' of the postnatal period, {onset_anaemia.loc[onset_anaemia].index}')
            self.postnatal_tracker['postnatal_anaemia'] += len(onset_anaemia.loc[onset_anaemia])

        # --------------------------------------- HYPERTENSION ------------------------------------------
        # For women who are still experiencing a hypertensive disorder of pregnancy we determine if that will now
        # resolve

        women_with_htn = df.loc[
            df['is_alive'] & df['la_is_postpartum'] & (df['pn_postnatal_period_in_weeks'] == week) &
            ~df['hs_is_inpatient'] &
            (df['pn_htn_disorders'].str.contains('gest_htn|severe_gest_htn|mild_pre_eclamp|severe_pre_eclamp|'
                                                 'eclampsia'))]

        resolvers = pd.Series(self.rng.random_sample(len(women_with_htn)) < params['prob_htn_resolves'],
                              index=women_with_htn.index)

        for person in resolvers.loc[resolvers].index:
            if not pd.isnull(mni[person]['hypertension_onset']):
                store_dalys_in_mni(person, 'hypertension_resolution')

        df.loc[resolvers.loc[resolvers].index, 'pn_htn_disorders'] = 'resolved'

        # And for the women who's hypertension doesnt resolve we now see if it will progress to a worsened state
        # This uses the transition_states function to move women between states based on the probability matrix

        def apply_risk(selected, risk_of_gest_htn_progression):

            disease_states = ['gest_htn', 'severe_gest_htn', 'mild_pre_eclamp', 'severe_pre_eclamp', 'eclampsia']
            prob_matrix = pd.DataFrame(columns=disease_states, index=disease_states)

            risk_ghtn_remains_mild = (0.9 - risk_of_gest_htn_progression)

            params['probs_for_mgh_matrix'] = [risk_ghtn_remains_mild, risk_of_gest_htn_progression, 0.1, 0.0, 0.0]

            prob_matrix['gest_htn'] = params['probs_for_mgh_matrix_pn']
            prob_matrix['severe_gest_htn'] = params['probs_for_sgh_matrix_pn']
            prob_matrix['mild_pre_eclamp'] = params['probs_for_mpe_matrix_pn']
            prob_matrix['severe_pre_eclamp'] = params['probs_for_spe_matrix_pn']
            prob_matrix['eclampsia'] = params['probs_for_ec_matrix_pn']

            current_status = df.loc[selected, "pn_htn_disorders"]
            new_status = util.transition_states(current_status, prob_matrix, self.rng)
            df.loc[selected, "pn_htn_disorders"] = new_status

            # Then we determine which women have transitioned to the most severe states, and may choose to seek care
            assess_status_change_for_severe_pre_eclampsia = (current_status != "severe_pre_eclamp") & \
                                                            (new_status == "severe_pre_eclamp")

            new_onset_severe_pre_eclampsia = assess_status_change_for_severe_pre_eclampsia[
                assess_status_change_for_severe_pre_eclampsia]

            # We log the women who have transitioned to severe pre-eclampsia and set pn_emergency_event_mother to True
            # so they may seek care
            if not new_onset_severe_pre_eclampsia.empty:
                logger.debug(key='message',
                             data='The following women have developed severe pre-eclampsia following their '
                             f'pregnancy {new_onset_severe_pre_eclampsia.index}')

                for person in new_onset_severe_pre_eclampsia.index:
                    df.at[person, 'pn_emergency_event_mother'] = True

            # This process is repeated for women who have now developed eclampsia
            assess_status_change_for_eclampsia = (current_status != "eclampsia") & (new_status == "eclampsia")
            new_onset_eclampsia = assess_status_change_for_eclampsia[assess_status_change_for_eclampsia]

            if not new_onset_eclampsia.empty:
                logger.debug(key='message', data=f'The following women have developed eclampsia during week {week} of '
                                                 f'the postnatal period: {new_onset_eclampsia.index}')

                for person in new_onset_eclampsia.index:
                    df.at[person, 'pn_emergency_event_mother'] = True

        women_with_htn_not_on_anti_htns =\
            df.is_alive & \
            df.la_is_postpartum & \
            (df.pn_postnatal_period_in_weeks == week) & \
            (df['pn_htn_disorders'].str.contains('gest_htn|severe_gest_htn|mild_pre_eclamp|severe_pre_eclamp|'
                                                 'eclampsia')) & \
            ~df.pn_gest_htn_on_treatment & ~df.hs_is_inpatient

        women_with_htn_on_anti_htns = \
            df.is_alive & \
            df.la_is_postpartum & \
            (df.pn_postnatal_period_in_weeks == week) & \
            (df['pn_htn_disorders'].str.contains('gest_htn|severe_gest_htn|mild_pre_eclamp|severe_pre_eclamp|'
                                                 'eclampsia')) & \
            df.pn_gest_htn_on_treatment & ~df.hs_is_inpatient

        risk_progression_mild_to_severe_htn = 0.1

        apply_risk(women_with_htn_not_on_anti_htns, risk_progression_mild_to_severe_htn)
        apply_risk(women_with_htn_on_anti_htns, (risk_progression_mild_to_severe_htn *
                                                 params['treatment_effect_anti_htns_progression_pn']))

        #  -------------------------------- RISK OF PRE-ECLAMPSIA HYPERTENSION --------------------------------------
        # Here we apply a risk to women developing de-novo hypertensive in the later postnatal period
        pre_eclampsia = apply_linear_model(
            params['pn_linear_equations']['pre_eclampsia_pn'],
            df.loc[df['is_alive'] & df['la_is_postpartum'] & (df['pn_postnatal_period_in_weeks'] == week) &
                   (df['pn_htn_disorders'] == 'none')])

        df.loc[pre_eclampsia.loc[pre_eclampsia].index, 'ps_prev_pre_eclamp'] = True
        df.loc[pre_eclampsia.loc[pre_eclampsia].index, 'pn_htn_disorders'] = 'mild_pre_eclamp'

        if not pre_eclampsia.loc[pre_eclampsia].empty:
            logger.debug(key='message', data=f'The following women have developed pre_eclampsia in week {week} of the'
                                             f' postnatal period: {pre_eclampsia.loc[pre_eclampsia].index}')

        #  -------------------------------- RISK OF GESTATIONAL HYPERTENSION --------------------------------------
        gest_hypertension = apply_linear_model(
            params['pn_linear_equations']['gest_htn_pn'],
            df.loc[df['is_alive'] & df['la_is_postpartum'] & (df['pn_postnatal_period_in_weeks'] == week) &
                   (df['pn_htn_disorders'] == 'none')])

        df.loc[gest_hypertension.loc[gest_hypertension].index, 'pn_htn_disorders'] = 'gest_htn'

        if not gest_hypertension.loc[gest_hypertension].empty:
            logger.debug(key='message', data=f'The following women have developed gestational hypertension in week '
                                             f'{week} of the postnatal period '
                                             f'{gest_hypertension.loc[gest_hypertension].index}')

        # -------------------------------- RISK OF DEATH SEVERE HYPERTENSION ------------------------------------------
        # Risk of death is applied to women with severe hypertensive disease
        at_risk_of_death_htn = apply_linear_model(params['pn_linear_equations']['death_from_hypertensive_disorder_pn'],
                                                  df.loc[df['is_alive'] & df['la_is_postpartum'] &
                                                         (df['pn_postnatal_period_in_weeks'] == week) &
                                                         (df['pn_htn_disorders'].str.contains('severe_gest_htn|'
                                                                                              'severe_pre_eclamp'))])

        if not at_risk_of_death_htn.loc[at_risk_of_death_htn].empty:
            self.postnatal_tracker['postnatal_death'] += \
                    len(at_risk_of_death_htn.loc[at_risk_of_death_htn].index)

            logger.debug(key='message',
                         data=f'The following women have died due to severe hypertensive disorder,'
                              f'{at_risk_of_death_htn.loc[at_risk_of_death_htn].index} in the postnatal period')

            # Those women who die have InstantaneousDeath scheduled
            for person in at_risk_of_death_htn.loc[at_risk_of_death_htn].index:
                self.sim.modules['Demography'].do_death(individual_id=person, cause='maternal',
                                                        originating_module=self.sim.modules['PostnatalSupervisor'])
                del self.sim.modules['PregnancySupervisor'].mother_and_newborn_info[person]

        # ----------------------------------------- CARE SEEKING ------------------------------------------------------
        # We now use the the pn_emergency_event_mother property that has just been set for women who are experiencing
        # severe complications to select a subset of women who may choose to seek care
        care_seeking = apply_linear_model(
            params['pn_linear_equations']['care_seeking_postnatal_complication_mother'],
            df.loc[df['is_alive'] & df['la_is_postpartum'] & (df['pn_postnatal_period_in_weeks'] == week) &
                   df['pn_emergency_event_mother'] & ~df['hs_is_inpatient']])

        # Reset this property to stop repeat care seeking
        df.loc[care_seeking.index, 'pn_emergency_event_mother'] = False

        # Schedule the HSI event
        for person in care_seeking.loc[care_seeking].index:
            admission_event = HSI_PostnatalSupervisor_PostnatalWardInpatientCare(
                self, person_id=person)
            self.sim.modules['HealthSystem'].schedule_hsi_event(admission_event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        # For women who do not seek care we immediately apply risk of death due to complications
        for person in care_seeking.loc[~care_seeking].index:
            self.apply_risk_of_maternal_or_neonatal_death_postnatal(mother_or_child='mother', individual_id=person)

    def apply_risk_of_neonatal_complications_in_week_one(self, child_id, mother_id):
        """
        This function is called by PostnatalWeekOneEvent for newborns to determine risk of complications. Currently
        this is limited to early onset neonatal sepsis
        :param child_id: childs individual id
        :return:
        """
        df = self.sim.population.props
        params = self.parameters
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        abx_status = mni[mother_id]['abx_for_prom_given']

        logger.debug(key='message', data=f'Newborn {child_id} has arrived at PostnatalWeekOneEvent')

        # We then apply a risk that this womans newborn will develop sepsis during week one
        risk_eons = params['pn_linear_equations']['early_onset_neonatal_sepsis_week_1'].predict(
            df.loc[[child_id]], received_abx_for_prom=abx_status)[child_id]

        if self.rng.random_sample() < risk_eons:
            df.at[child_id, 'pn_sepsis_early_neonatal'] = True
            logger.debug(key='message', data=f'Newborn {child_id} has developed early onset sepsis in week one of '
                                             f'the neonatal period')
            self.postnatal_tracker['early_neonatal_sepsis'] += 1

    def set_postnatal_complications_neonates(self, upper_and_lower_day_limits):
        """
        This function is called by the PostnatalSupervisor event. It applies risk of key complication to neonates
        during each week of the neonatal period after week one (weeks 2, 3 & 4). This is currently limited to sepsis but
         may be expanded at a later date
        :param upper_and_lower_day_limits: 2 value list of the first and last day of each week of the neonatal period
        """
        params = self.parameters
        df = self.sim.population.props
        apply_linear_model = self.sim.modules['PregnancySupervisor'].apply_linear_model

        # Here we apply risk of late onset neonatal sepsis (sepsis onsetting after day 7) to newborns
        onset_sepsis = apply_linear_model(
            params['pn_linear_equations']['late_onset_neonatal_sepsis'],
            df.loc[df['is_alive'] & (df['age_days'] > upper_and_lower_day_limits[0]) &
                   (df['age_days'] < upper_and_lower_day_limits[1]) & ~df['hs_is_inpatient']])

        df.loc[onset_sepsis.loc[onset_sepsis].index, 'pn_sepsis_late_neonatal'] = True
        self.postnatal_tracker['late_neonatal_sepsis'] += 1

        # Then we determine if care will be sought for newly septic newborns
        care_seeking = apply_linear_model(
            params['pn_linear_equations']['care_seeking_postnatal_complication_neonate'],
            df.loc[onset_sepsis.loc[onset_sepsis].index])

        # We schedule the HSI according
        for person in care_seeking.loc[care_seeking].index:
            admission_event = HSI_PostnatalSupervisor_NeonatalWardInpatientCare(
                self, person_id=person)
            self.sim.modules['HealthSystem'].schedule_hsi_event(admission_event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        # And apply risk of death for newborns for which care is not sought
        for person in care_seeking.loc[~care_seeking].index:
            self.apply_risk_of_maternal_or_neonatal_death_postnatal(mother_or_child='child', individual_id=person)

    def assessment_for_maternal_complication_during_pnc(self, individual_id, hsi_event):
        """
        This function is called by each of the postnatal care visit HSIs and represents assessment of mothers for
        sepsis, postnatal bleeding, hypertension and anaemia. If these conditions are detected during PNC then women
        are admitted for treatment
        :param individual_id: individual_id (mother)
        :param hsi_event: HSI event in which this function is called
        """
        params = self.parameters
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        df = self.sim.population.props

        # We create a variable that will be set to true if a health work detects a complication and chooses to admit
        # (in case of multiple complications requiring admission)
        needs_admission = False

        # similarly for detection of these conditions
        hypertension_diagnosed = False
        proteinuria_diagnosed = False

        # Each woman is screened for postnatal depression
        if 'Depression' in self.sim.modules.keys():
            if self.rng.random_sample() < params['prob_intervention_delivered_depression_screen_pnc']:
                logger.debug(key='msg', data=f'Mother {individual_id} will now be receive screening for depression'
                                             f' during PNC and commence treatment as appropriate')
                if not df.at[individual_id, 'de_ever_diagnosed_depression']:
                    self.sim.modules['Depression'].do_when_suspected_depression(individual_id, hsi_event)

        # Define the consumables
        item_code_urine_dipstick = pd.unique(
            consumables.loc[consumables['Items'] == 'Test strips, urine analysis', 'Item_Code'])[0]

        consumables_dipstick = {
            'Intervention_Package_Code': {},
            'Item_Code': {item_code_urine_dipstick: 1}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=hsi_event,
            cons_req_as_footprint=consumables_dipstick)

        if outcome_of_request_for_consumables['Item_Code'][item_code_urine_dipstick] and (self.rng.random_sample()
                                                                                          < params['prob_intervention'
                                                                                                   '_delivered_urine_'
                                                                                                   'ds_pnc']):

            # If the consumables are available the test is ran. Urine testing in ANC is predominantly used to detected
            # protein in the urine (proteinuria) which is indicative of pre-eclampsia
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run='urine_dipstick_protein_pn',
                                                                       hsi_event=hsi_event):
                # We use a temporary variable to store if proteinuria is detected (only sever diseases or new diagnosis
                # lead to admission)
                if (df.at[individual_id, 'pn_htn_disorders'] == 'severe_pre_eclampsia') or\
                    (df.at[individual_id, 'pn_htn_disorders'] == 'eclampsia') or not df.at[individual_id,
                                                                                           'pn_gest_htn_on_treatment']:
                    proteinuria_diagnosed = True

        else:
            logger.debug(key='msg', data='Urine dipstick testing was not completed in this PNC visit due to '
                                         'unavailable consumables')

        # The process is repeated for blood pressure monitoring- although not conditioned on consumables
        if self.rng.random_sample() < params['prob_intervention_delivered_bp_pnc']:
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run='blood_pressure_measurement_pn',
                                                                       hsi_event=hsi_event):

                if (df.at[individual_id, 'pn_htn_disorders'] == 'severe_pre_eclampsia') or \
                    (df.at[individual_id, 'pn_htn_disorders'] == 'eclampsia') or not df.at[individual_id,
                                                                                           'pn_gest_htn_on_treatment']:
                    hypertension_diagnosed = True

                if not df.at[individual_id, 'pn_gest_htn_on_treatment']:
                    self.sim.modules['PregnancySupervisor'].store_dalys_in_mni(individual_id, 'hypertension_onset')

        if hypertension_diagnosed or proteinuria_diagnosed:
            needs_admission = True

        # SEPSIS
        # Women are assessed for key complications after child birth
        if self.rng.random_sample() < params['prob_intervention_delivered_sep_assessment_pnc']:
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run='assessment_for_postnatal_'
                                                                                       'sepsis',
                                                                       hsi_event=hsi_event):
                logger.debug(key='message', data=f'Mother {individual_id} has been assessed and diagnosed with '
                                                 f'postpartum sepsis, she will be admitted for treatment')

                needs_admission = True

        # HAEMORRHAGE
        if self.rng.random_sample() < params['prob_intervention_delivered_pph_assessment_pnc']:
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run='assessment_for_secondary_pph',
                                                                       hsi_event=hsi_event):
                logger.debug(key='message', data=f'Mother {individual_id} has been assessed and diagnosed with'
                                                 f' secondary postpartum haemorrhage hypertension, she will be '
                                                 f'admitted for treatment')
                needs_admission = True

        if self.rng.random_sample() < params['prob_intervention_poct_pnc']:
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run='point_of_care_hb_test_pn',
                                                                       hsi_event=hsi_event):
                logger.debug(key='message',
                             data=f'Mother {individual_id} has been assessed and diagnosed with postpartum '
                             f'anaemia, she will be admitted for treatment')
                needs_admission = True

        # If any of the above complications have been detected then the woman is admitted from PNC visit to the
        # Postnatal ward for further care
        if needs_admission:
            admission_event = HSI_PostnatalSupervisor_PostnatalWardInpatientCare(
                self, person_id=individual_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(admission_event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

    def assessment_for_neonatal_complications_during_pnc(self, individual_id, hsi_event, pnc_visit):
        """
        This function is called by each of the postnatal care visit HSI and represents assessment of neonates for
        sepsis. If sepsis is detected during PNC then the neonate is  admitted for treatment.
        :param individual_id: individual_id (child)
        :param hsi_event: HSI event in which this function is called
        """
        params = self.parameters
        df = self.sim.population.props
        mother_id = df.at[individual_id, 'mother_id']

        # As with assessment_for_maternal_complication_during_pnc neonates are assessed for complications and admitted
        # as required (we specify which PNC visit this is occurring in due to a different property being used for the
        # type of sepsis)

        hsi_event.target = int(individual_id)

        # SEPSIS
        if (pnc_visit == 'pnc1') and (self.rng.random_sample() < params['prob_intervention_neonatal_sepsis_pnc']):
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
              dx_tests_to_run='assessment_for_early_onset_neonatal_sepsis', hsi_event=hsi_event):
                logger.debug(key='message', data=f'Neonate {individual_id} has been assessed and diagnosed with early '
                                                 f'onset neonatal sepsis, they will be admitted for treatment')

                sepsis_treatment = HSI_PostnatalSupervisor_NeonatalWardInpatientCare(
                    self, person_id=individual_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(sepsis_treatment,
                                                                    priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1))

        elif (pnc_visit == 'pnc2') and (self.rng.random_sample() < params['prob_intervention_neonatal_sepsis_pnc']):
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run='assessment_for_late_onset_'
                                                                                       'neonatal_sepsis',
                                                                       hsi_event=hsi_event):
                logger.debug(key='message', data=f'Neonate {individual_id} has been assessed and diagnosed with late '
                                                 f'onset neonatal sepsis, they will be admitted for treatment')

                sepsis_treatment = HSI_PostnatalSupervisor_NeonatalWardInpatientCare(
                    self, person_id=individual_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(sepsis_treatment,
                                                                    priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1))

        if pnc_visit == 'pnc2' and (self.rng.random_sample() < params['prob_intervention_delivered_hiv_test_pnc']):
            if 'Hiv' in self.sim.modules.keys():
                if ~df.at[individual_id, 'hv_diagnosed'] and df.at[mother_id, 'hv_diagnosed']:

                    logger.debug(key='message', data='Neonate of a HIV +ve mother has been referred for '
                                                     'HIV testing during PNC at 6 weeks')

                    individual_id = int(individual_id)

                    self.sim.modules['HealthSystem'].schedule_hsi_event(
                        HSI_Hiv_TestAndRefer(person_id=individual_id, module=self.sim.modules['Hiv']),
                        topen=self.sim.date,
                        tclose=None,
                        priority=0)

        hsi_event.target = mother_id

    def apply_risk_of_maternal_or_neonatal_death_postnatal(self, mother_or_child, individual_id):
        """
        This function is called to calculate an individuals risk of death following the onset of a complication. For
        individuals who dont seek care this is immediately after the onset of complications. For those who seek care it
        is called at the end of the HSI to allow for treatment effects. Either a mother or a child can be passed to the
        function.
        :param mother_or_child: Person of interest for the effect of this function - pass 'mother' or 'child' to
         apply risk of death correctly
        :param individual_id: individual_id
        """
        df = self.sim.population.props
        params = self.parameters

        # Select the individuals row in the data frame to prevent repeated at based indexing
        if mother_or_child == 'mother':
            mother = df.loc[individual_id]
        if mother_or_child == 'child':
            child = df.loc[individual_id]

        # ================================== MATERNAL DEATH EQUATIONS ==============================================
        # We cycle through the possible causes of death to allow for increased risk of death in women with multiple
        # complications
        if mother_or_child == 'mother':
            postnatal_death = False

            # If the mother has had a hemorrhage and hasn't sought care, we calculate her risk of death
            if mother.pn_postpartum_haem_secondary:
                risk_of_death = params['pn_linear_equations']['secondary_postpartum_haem_death'].predict(df.loc[[
                    individual_id]])[individual_id]

                if self.rng.random_sample() < risk_of_death:
                    postnatal_death = True

                # If she will survive we reset the relevant variable in the data frame
                else:
                    df.at[individual_id, 'pn_postpartum_haem_secondary'] = False

            # If the mother is septic and hasn't sought care, we calculate her risk of death
            if mother.pn_sepsis_late_postpartum:

                risk_of_death = params['pn_linear_equations']['postnatal_sepsis_death'].predict(df.loc[[
                    individual_id]])[individual_id]

                if self.rng.random_sample() < risk_of_death:
                    postnatal_death = True

                # If she will survive we reset the relevant variable in the data frame
                else:
                    df.at[individual_id, 'pn_sepsis_late_postpartum'] = False
                    self.postpartum_infections_late.unset(
                        [individual_id], 'endometritis', 'urinary_tract_inf', 'skin_soft_tissue_inf',
                        'other_maternal_infection')

            # Finally if the mother has eclampsia we calculate risk of death
            if mother.pn_htn_disorders == 'eclampsia':
                risk_of_death = params['pn_linear_equations']['eclampsia_death_pn'].predict(
                            df.loc[[individual_id]])[individual_id]

                if self.rng.random_sample() < risk_of_death:
                    postnatal_death = True
                else:
                    df.at[individual_id, 'pn_htn_disorders'] = 'severe_pre_eclamp'

            # If she has died due to either (or both) of these causes, we schedule the DeathEvent
            if postnatal_death:
                logger.debug(key='message', data=f'mother {individual_id} has died due to complications of the '
                                                 f'postnatal period')

                self.sim.modules['Demography'].do_death(individual_id=individual_id, cause='maternal',
                                                        originating_module=self.sim.modules['PostnatalSupervisor'])

                self.postnatal_tracker['postnatal_death'] += 1
                del self.sim.modules['PregnancySupervisor'].mother_and_newborn_info[individual_id]

        # ================================== NEONATAL DEATH EQUATIONS ==============================================
        if mother_or_child == 'child':

            # Neonates can have either early or late onset sepsis, not both at once- so we use either equation
            # depending on this neonates current condition
            if child.pn_sepsis_early_neonatal:
                risk_of_death = params['pn_linear_equations']['early_onset_neonatal_sepsis_week_1_death'].predict(
                    df.loc[[individual_id]])[individual_id]
            elif child.pn_sepsis_late_neonatal:
                risk_of_death = params['pn_linear_equations']['late_neonatal_sepsis_death'].predict(df.loc[[
                    individual_id]])[individual_id]

            if child.pn_sepsis_late_neonatal or child.pn_sepsis_early_neonatal:

                # If this neonate will die then we make the appropriate changes
                if self.rng.random_sample() < risk_of_death:
                    logger.debug(key='message', data=f'person {individual_id} has died due to late neonatal sepsis on '
                                                     f'date {self.sim.date}')

                    self.sim.modules['Demography'].do_death(individual_id=individual_id, cause='neonatal',
                                                            originating_module=self.sim.modules['PostnatalSupervisor'])

                    self.postnatal_tracker['neonatal_death'] += 1
                    self.postnatal_tracker['neonatal_sepsis_death'] += 1

                # Otherwise we reset the variables in the data frame
                else:
                    df.at[individual_id, 'pn_sepsis_late_neonatal'] = False
                    df.at[individual_id, 'pn_sepsis_early_neonatal'] = False

                    # For surviving neonates we determine if they will experience any long term disability from their
                    # infection
                    disability_categories = ['none',
                                             'mild_motor_and_cog',  # Mild motor plus cognitive impairments due to...
                                             'mild_motor',  # Mild motor impairment due to...
                                             'moderate_motor',  # Moderate motor impairment due to...
                                             'severe_motor']  # Severe motor impairment due to...
                    choice = self.rng.choice
                    df.at[individual_id, 'pn_neonatal_sepsis_disab'] = choice(disability_categories,
                                                                              p=params['prob_sepsis_disabilities'])

    def maternal_postnatal_care_care_seeking(self, individual_id, recommended_day_next_pnc, next_pnc_visit,
                                             maternal_pnc):
        """
        This function is called by HSI_PostnatalSupervisor_PostnatalCareContact. It determines if a mother
        will return to attend her next PNC visit in the schedule
        :param individual_id: individual_id
        :param recommended_day_next_pnc: int signifying number of days post birth the next visit should occur
        :param next_pnc_visit: string signifying next visit in schedule i.e. 'pnc2'
        :param maternal_pnc: HSI to be scheduled
        """
        df = self.sim.population.props
        params = self.parameters

        # Calculate how many days since this woman has given birth
        ppp_in_days = self.sim.date - df.at[individual_id, 'la_date_most_recent_delivery']

        # Calculate home many days until the next visit should be scheduled
        days_calc = pd.to_timedelta(recommended_day_next_pnc, unit='D') - ppp_in_days
        date_next_pnc = self.sim.date + days_calc

        # Apply a probability that she will chose to return for the next visit
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
        store_dalys_in_mni = self.sim.modules['PregnancySupervisor'].store_dalys_in_mni
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        # ================================ UPDATING LENGTH OF POSTPARTUM PERIOD  IN WEEKS  ============================
        # Here we update how far into the postpartum period each woman who has recently delivered is
        alive_and_recently_delivered = df.is_alive & df.la_is_postpartum
        ppp_in_days = self.sim.date - df.loc[alive_and_recently_delivered, 'la_date_most_recent_delivery']
        ppp_in_weeks = ppp_in_days / np.timedelta64(1, 'W')
        rounded_weeks = np.ceil(ppp_in_weeks)

        df.loc[alive_and_recently_delivered, 'pn_postnatal_period_in_weeks'] = rounded_weeks
        logger.debug(key='message', data=f'updating postnatal periods on date {self.sim.date}')

        # Check that all women are week 1 or above
        assert (df.loc[alive_and_recently_delivered, 'pn_postnatal_period_in_weeks'] > 0).all().all()

        # ================================= COMPLICATIONS/CARE SEEKING FOR WOMEN ======================================
        # This function is called to apply risk of complications to women in weeks 2, 3, 4, 5 and 6 of the postnatal
        # period
        for week in [2, 3, 4, 5, 6]:
            self.module.set_postnatal_complications_mothers(week=week)

        # ================================= COMPLICATIONS/CARE SEEKING FOR NEONATES ===================================
        # Next this function is called to apply risk of complications to neonates in week 2, 3 and 4 of the neonatal
        # period. Upper and lower limit days in the week are used to define one week.
        for upper_and_lower_day_limits in [[7, 15], [14, 22], [21, 29]]:
            self.module.set_postnatal_complications_neonates(upper_and_lower_day_limits=upper_and_lower_day_limits)

        # -------------------------------------- RESETTING VARIABLES --------------------------------------------------
        # Finally we reset any variables that have been modified during this module
        # We make these changes 2 weeks after the end of the postnatal and neonatal period in case of either mother or
        # newborn are receiving treatment following the last PNC visit (around day 42)

        # Maternal variables
        week_8_postnatal_women = df.is_alive & df.la_is_postpartum & (df.pn_postnatal_period_in_weeks == 8)
        df.loc[week_8_postnatal_women, 'pn_postnatal_period_in_weeks'] = 0
        df.loc[week_8_postnatal_women, 'pn_pnc_visits_maternal'] = 0
        df.loc[week_8_postnatal_women, 'la_is_postpartum'] = False
        df.loc[week_8_postnatal_women, 'pn_sepsis_late_postpartum'] = False
        df.loc[week_8_postnatal_women, 'pn_postpartum_haem_secondary'] = False
        self.module.postpartum_infections_late.unset(week_8_postnatal_women, 'endometritis', 'urinary_tract_inf',
                                                     'skin_soft_tissue_inf', 'other_maternal_infection')
        self.module.deficiencies_following_pregnancy.unset(week_8_postnatal_women, 'iron', 'folate', 'b12')

        week_8_postnatal_women_htn = \
            df.is_alive & df.la_is_postpartum & (df.pn_postnatal_period_in_weeks == 8) & \
            (df.pn_htn_disorders.str.contains('gest_htn|severe_gest_htn|mild_pre_eclamp|severe_pre_eclamp|eclampsia'))

        # Schedule date of resolution for any women with hypertension
        for person in week_8_postnatal_women_htn.loc[week_8_postnatal_women_htn].index:
            if not pd.isnull(mni[person]['hypertension_onset']):
                store_dalys_in_mni(person, 'hypertension_resolution')

        df.loc[week_8_postnatal_women, 'pn_htn_disorders'] = 'none'

        week_8_postnatal_women_anaemia = \
            df.is_alive & df.la_is_postpartum & (df.pn_postnatal_period_in_weeks == 8) & \
            (df.pn_anaemia_following_pregnancy != 'none')

        # Schedule date of resolution for any women with anaemia
        for person in week_8_postnatal_women_anaemia.loc[week_8_postnatal_women_anaemia].index:
            for severity in df.at[person, 'pn_anaemia_following_pregnancy']:
                store_dalys_in_mni(person, f'{severity}_anaemia_pp_resolution')

        df.loc[week_8_postnatal_women, 'pn_anaemia_following_pregnancy'] = 'none'

        # Reset any treatment variables
        df.loc[week_8_postnatal_women, 'pn_postpartum_haem_secondary_treatment'] = False
        df.loc[week_8_postnatal_women, 'pn_mag_sulph_treatment'] = False
        df.loc[week_8_postnatal_women, 'pn_gest_htn_on_treatment'] = False
        df.loc[week_8_postnatal_women, 'pn_iv_anti_htn_treatment'] = False
        df.loc[week_8_postnatal_women, 'pn_emergency_event_mother'] = False

        # Schedule event that deletes the mni dictionary, we reset in greater than one months time to allow dalys to be
        # counted

        for person in week_8_postnatal_women.loc[week_8_postnatal_women].index:
            mni[person]['delete_mni'] = True

        # Neonatal variables
        week_6_postnatal_neonates = df.is_alive & (df['age_days'] > 42) & (df['age_days'] < 49)
        df.loc[week_6_postnatal_neonates, 'pn_pnc_visits_neonatal'] = 0
        df.loc[week_6_postnatal_neonates, 'pn_sepsis_early_neonatal'] = False
        df.loc[week_6_postnatal_neonates, 'pn_sepsis_late_neonatal'] = False
        df.loc[week_6_postnatal_neonates, 'pn_sepsis_neonatal_inj_abx'] = False
        df.loc[week_6_postnatal_neonates, 'pn_sepsis_neonatal_full_supp_care'] = False


class PostnatalWeekOneEvent(Event, IndividualScopeEventMixin):
    """
    This is PostnatalWeekOneEvent. It is scheduled for all mothers who survive labour and the first 48 hours after
    birth. This event applies risk of key complications that can occur in the first week after birth to both these
    mothers and their recently delivered newborn. This event also determines if a woman will seek care, with her
    newborn, for their first postnatal care visit on day 7 after birth. For women who dont seek care for themselves,
    or their newborns, risk of death is applied.
    """

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        store_dalys_in_mni = self.sim.modules['PregnancySupervisor'].store_dalys_in_mni
        mother = df.loc[individual_id]
        child_id = int(df.at[individual_id, 'pn_id_most_recent_child'])
        child = df.loc[child_id]

        # If the child of interest is the first born in a twin pair, we store the id number of their sibling to allow
        # application of risk to both babies
        if df.at[child_id, 'nb_is_twin']:
            child_two_id = df.at[child_id, 'nb_twin_sibling_id']
            child_two = df.loc[child_two_id]

        # Run a number of checks to ensure only the correct women/children arrive here
        assert mother.la_is_postpartum
        assert (self.sim.date - mother.la_date_most_recent_delivery) < pd.to_timedelta(7, unit='d')
        assert child.age_days < 7
        if df.at[child_id, 'nb_is_twin']:
            assert child_two.age_days < 7

        # If both the mother and newborn have died then this even wont run (it is possible for one or the other to have
        # died prior to this event- hence repeat checks on is_alive throughout)
        if ~df.at[child_id, 'nb_is_twin'] and ~mother.is_alive and ~child.is_alive:
            return

        # If the mother and both neonates in a twin pair have died then this event wont run
        if df.at[child_id, 'nb_is_twin']:
            if ~mother.is_alive and ~child.is_alive and ~child_two.is_alive:
                return

        # ===============================  MATERNAL COMPLICATIONS IN WEEK ONE  =======================================
        #  ------------------------------------- INFECTIONS AND SEPSIS ----------------------------------------------
        # We determine if the mother will develop any postnatal infections and if they will progress into sepsis
        if mother.is_alive:
            logger.debug(key='message', data=f'Mother {individual_id} has arrived at PostnatalWeekOneEvent')

            for infection in ['endometritis', 'urinary_tract_inf', 'skin_soft_tissue_inf', 'other_maternal_infection']:
                self.module.set_infections(individual_id, infection=infection)

            risk_sepsis = params['pn_linear_equations']['sepsis_late_postpartum'].predict(df.loc[[
                individual_id]])[individual_id]

            if risk_sepsis > self.module.rng.random_sample():
                logger.debug(key='message',
                             data=f'mother {individual_id} has developed postnatal sepsis during week one of the'
                                  f' postnatal period')

                df.at[individual_id, 'pn_sepsis_late_postpartum'] = True
                self.module.postnatal_tracker['postnatal_sepsis'] += 1
                store_dalys_in_mni(individual_id, 'sepsis_onset')

        #  ---------------------------------------- SECONDARY PPH ------------------------------------------------
            # Next we apply risk of secondary postpartum bleeding
            risk_secondary_pph = params['pn_linear_equations']['secondary_postpartum_haem'].predict(df.loc[[
                    individual_id]])[individual_id]

            if risk_secondary_pph > self.module.rng.random_sample():
                logger.debug(key='message',
                             data=f'mother {individual_id} has developed a secondary postpartum haemorrhage during'
                                  f' week one of the postnatal period')

                df.at[individual_id, 'pn_postpartum_haem_secondary'] = True
                self.module.postnatal_tracker['secondary_pph'] += 1
                store_dalys_in_mni(individual_id, 'secondary_pph_onset')

        # ----------------------------------- NEW ONSET DEFICIENCIES AND ANAEMIA -------------------------------------
            # And then risk of developing deficiencies or anaemia
            if ~self.module.deficiencies_following_pregnancy.has_any([individual_id], 'iron', first=True):

                if ~mother.la_iron_folic_acid_postnatal:
                    if self.module.rng.random_sample() < params['prob_iron_def_per_week_pn']:
                        self.module.deficiencies_following_pregnancy.set([individual_id], 'iron')

                elif mother.la_iron_folic_acid_postnatal:
                    risk_of_iron_def = params['prob_iron_def_per_week_pn'] * params['rr_iron_def_ifa_pn']
                    if self.module.rng.random_sample() < risk_of_iron_def:
                        self.module.deficiencies_following_pregnancy.set([individual_id], 'iron')

            if ~self.module.deficiencies_following_pregnancy.has_any([individual_id], 'folate', first=True):

                if ~mother.la_iron_folic_acid_postnatal:
                    if self.module.rng.random_sample() < params['prob_folate_def_per_week_pn']:
                        self.module.deficiencies_following_pregnancy.set([individual_id], 'folate')

                elif mother.la_iron_folic_acid_postnatal:
                    risk_of_folate_def = params['prob_folate_def_per_week_pn'] * params['rr_folate_def_ifa_pn']
                    if self.module.rng.random_sample() < risk_of_folate_def:
                        self.module.deficiencies_following_pregnancy.set([individual_id], 'folate')

            if ~self.module.deficiencies_following_pregnancy.has_any([individual_id], 'b12', first=True):
                if self.module.rng.random_sample() < params['prob_b12_def_per_week_pn']:
                    self.module.deficiencies_following_pregnancy.set([individual_id], 'b12')

            if mother.pn_anaemia_following_pregnancy == 'none':
                risk_anaemia_after_pregnancy = params['pn_linear_equations']['anaemia_after_pregnancy'].predict(df.loc[[
                        individual_id]])[individual_id]

                if risk_anaemia_after_pregnancy > self.module.rng.random_sample():
                    random_choice_severity = self.module.rng.choice(['mild', 'moderate', 'severe'],
                                                                    p=params['prob_type_of_anaemia_pn'], size=1)
                    df.at[individual_id, 'pn_anaemia_following_pregnancy'] = random_choice_severity

                    store_dalys_in_mni(individual_id, f'{ df.at[individual_id, "pn_anaemia_following_pregnancy"] }_'
                                                      f'anaemia_pp_onset')

            # -------------------------------------------- HYPERTENSION -----------------------------------------------
            # For women who remain hypertensive after delivery we apply a probability that this will resolve in the
            # first week after birth

            if 'none' not in mother.pn_htn_disorders:
                if 'resolved' in mother.pn_htn_disorders:
                    pass
                elif self.module.rng.random_sample() < params['prob_htn_resolves']:
                    if not pd.isnull(mni[individual_id]['hypertension_onset']):
                        # Store date of resolution for women who were aware of their hypertension (in keeping with daly
                        # weight definition)
                        store_dalys_in_mni(individual_id, 'hypertension_resolution')
                    df.at[individual_id, 'pn_htn_disorders'] = 'resolved'

                else:

                    # If not, we apply a risk that the hypertension might worsen and progress into a more severe form
                    disease_states = ['gest_htn', 'severe_gest_htn', 'mild_pre_eclamp', 'severe_pre_eclamp',
                                      'eclampsia']
                    prob_matrix = pd.DataFrame(columns=disease_states, index=disease_states)

                    prob_matrix['gest_htn'] = params['probs_for_mgh_matrix_pn']
                    prob_matrix['severe_gest_htn'] = params['probs_for_sgh_matrix_pn']
                    prob_matrix['mild_pre_eclamp'] = params['probs_for_mpe_matrix_pn']
                    prob_matrix['severe_pre_eclamp'] = params['probs_for_spe_matrix_pn']
                    prob_matrix['eclampsia'] = params['probs_for_ec_matrix_pn']

                    # We modify the probability of progressing from mild to severe gestational hypertension for women
                    # who are on anti hypertensives
                    if ~df.at[individual_id, 'pn_gest_htn_on_treatment']:
                        prob_matrix['gest_htn'][2] = 0.1
                    else:
                        treatment_reduced_risk = 0.1 * params['treatment_effect_anti_htns_progression_pn']
                        prob_matrix['gest_htn'][2] = treatment_reduced_risk
                        prob_matrix['gest_htn'][0] = 1 - (treatment_reduced_risk + 0.1)

                    current_status = df.loc[[individual_id], 'pn_htn_disorders']
                    new_status = util.transition_states(current_status, prob_matrix, self.module.rng)
                    df.loc[[individual_id], "pn_htn_disorders"] = new_status

                    # We capture the women who progress to the most severe forms
                    assess_status_change_for_severe_pre_eclampsia = \
                        (current_status != "severe_pre_eclamp") & (new_status == "severe_pre_eclamp")
                    assess_status_change_for_eclampsia = (current_status != "eclampsia") & (new_status == "eclampsia")

                    new_onset_severe_pre_eclampsia = assess_status_change_for_severe_pre_eclampsia[
                            assess_status_change_for_severe_pre_eclampsia]
                    if not new_onset_severe_pre_eclampsia.empty:
                        logger.debug(key='message',
                                     data=f'mother {individual_id} has developed severe pre-eclampsia in week one of '
                                          f'the postnatal period ')

                    new_onset_eclampsia = assess_status_change_for_eclampsia[assess_status_change_for_eclampsia]
                    if not new_onset_eclampsia.empty:
                        logger.debug(key='message',
                                     data=f'mother {individual_id} has developed eclampsia in week one of the '
                                          f'postnatal period ')

        #  ---------------------------- RISK OF POSTPARTUM PRE-ECLAMPSIA/HYPERTENSION --------------------------------
            # Women who are normatensive after delivery may develop new hypertension for the first time after birth

            if df.at[individual_id, 'pn_htn_disorders'] == 'none':
                risk_pe_after_pregnancy = params['pn_linear_equations']['pre_eclampsia_pn'].predict(df.loc[[
                    individual_id]])[individual_id]

                if risk_pe_after_pregnancy > self.module.rng.random_sample():
                    df.at[individual_id, 'pn_htn_disorders'] = 'mild_pre_eclamp'

                    logger.debug(key='message',
                                 data=f'mother {individual_id} has developed mild pre-eclampsia in week one of the '
                                 f'postnatal period ')
                    df.at[individual_id, 'ps_prev_pre_eclamp'] = True
                else:
                    risk_gh_after_pregnancy = params['pn_linear_equations']['gest_htn_pn'].predict(df.loc[[
                        individual_id]])[individual_id]
                    if risk_gh_after_pregnancy > self.module.rng.random_sample():
                        logger.debug(key='message',
                                     data=f'mother {individual_id} has developed gestational hypertension in week one '
                                          f'of the postnatal period ')
                        df.at[individual_id, 'pn_htn_disorders'] = 'gest_htn'

        # ===============================  NEONATAL COMPLICATIONS IN WEEK ONE  =======================================
        # Newborns may develop early onset sepsis in the first week of life, we apply that risk here

        if child.is_alive:
            self.module.apply_risk_of_neonatal_complications_in_week_one(child_id=child_id, mother_id=individual_id)

        # If this event is looking at a twin pair we apply risk to the second twin here
        if child.nb_is_twin:
            if child_two.is_alive:
                self.module.apply_risk_of_neonatal_complications_in_week_one(child_id=child_two_id,
                                                                             mother_id=individual_id)
        # ===================================== CARE SEEKING FOR PNC 1 ==============================================
        # Women who deliver in facilities are asked to return for a first postnatal check up at day 7 post birth
        # We assume that women with complications, or the mothers of babies with complications, are more likely to seek
        # postnatal care- and more quickly

        mother_has_complications = False
        child_has_complications = False
        child_two_has_complications = False

        # (we check is_alive as this event can run if either the mother or child is dead (not both)

        if mother.is_alive:
            # We use a temporary variable to determine if the mother has any complications that may trigger care seeking
            if ~df.at[individual_id, 'pn_sepsis_late_postpartum'] and \
                ~df.at[individual_id, 'pn_postpartum_haem_secondary'] and \
                (df.at[individual_id, 'pn_htn_disorders'] != 'severe_pre_eclamp') and\
               (df.at[individual_id, 'pn_htn_disorders'] != 'eclampsia'):
                mother_has_complications = False

            elif df.at[individual_id, 'pn_sepsis_late_postpartum'] or \
                df.at[individual_id, 'pn_postpartum_haem_secondary'] or \
                (df.at[individual_id, 'pn_htn_disorders'] == 'severe_pre_eclamp') or (df.at[individual_id,
                                                                                            'pn_htn_disorders'] ==
                                                                                      'eclampsia'):

                mother_has_complications = True

        # Repeat that process for the child
        if child.is_alive:
            if ~df.at[child_id, 'pn_sepsis_early_neonatal']:
                child_has_complications = False
            elif df.at[child_id, 'pn_sepsis_early_neonatal']:
                child_has_complications = True

        if df.at[child_id, 'nb_is_twin']:
            if df.at[child_two_id, 'is_alive']:
                if ~df.at[child_two_id, 'pn_sepsis_early_neonatal']:
                    child_two_has_complications = False
                elif df.at[child_two_id, 'pn_sepsis_early_neonatal']:
                    child_two_has_complications = True

        # If neither the mother or the child are experiencing any complications in the first week after birth,
        # we determine if they will present for the scheduled PNC check up at day 7

        prob_will_seek_care = params['pn_linear_equations']['care_seeking_for_first_pnc_visit'].predict(
            df.loc[[individual_id]])[individual_id]

        if not child_has_complications and not mother_has_complications and not child_two_has_complications:
            if self.module.rng.random_sample() < prob_will_seek_care:
                days_until_day_7 = self.sim.date - mother.la_date_most_recent_delivery
                days_until_day_7_int = int(days_until_day_7 / np.timedelta64(1, 'D'))

                pnc_one = HSI_PostnatalSupervisor_PostnatalCareContactOne(
                        self.module, person_id=individual_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(pnc_one,
                                                                    priority=0,
                                                                    topen=self.sim.date +
                                                                    DateOffset(days=days_until_day_7_int),
                                                                    tclose=None)

        # For women where either they or their baby has developed a complication during this time we assume likelihood
        # of care seeking for postnatal check up is higher

        elif mother_has_complications or child_has_complications or child_two_has_complications:
            prob_care_seeking = prob_will_seek_care * params['multiplier_for_care_seeking_with_comps']

            if prob_care_seeking > self.module.rng.random_sample():
                # And we assume they will present earlier than day 7
                admission_event = HSI_PostnatalSupervisor_PostnatalCareContactOne(
                        self.module, person_id=individual_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(admission_event,
                                                                    priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=None)

            # For 'non-care seekers' risk of death is applied immediately
            else:
                if mother_has_complications:
                    self.module.apply_risk_of_maternal_or_neonatal_death_postnatal(mother_or_child='mother',
                                                                                   individual_id=individual_id)
                if child_has_complications:
                    self.module.apply_risk_of_maternal_or_neonatal_death_postnatal(mother_or_child='child',
                                                                                   individual_id=child_id)
                if child_two_has_complications:
                    self.module.apply_risk_of_maternal_or_neonatal_death_postnatal(mother_or_child='child',
                                                                                   individual_id=child_two_id)


class HSI_PostnatalSupervisor_PostnatalCareContactOne(HSI_Event, IndividualScopeEventMixin):
    """This is HSI_PostnatalSupervisor_PostnatalCareContactOneMaternal. It is scheduled by
    PostnatalWeekOneEvent for women who decide to seek care for a postnatal checkup. This event is the first PNC visit
     women are recommended to attend after discharge, occurring on day 7 post birth. """
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, PostnatalSupervisor)

        self.TREATMENT_ID = 'PostnatalSupervisor_PostnatalCareContactOne'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        child_id = int(df.at[person_id, 'pn_id_most_recent_child'])

        # We run checks to make sure only the correct people are passed to the event
        assert df.at[person_id, 'la_is_postpartum']
        assert df.at[person_id, 'pn_pnc_visits_maternal'] == 0

        # Update the properties of the child
        if ~df.at[child_id, 'nb_is_twin']:
            assert df.at[child_id, 'pn_pnc_visits_neonatal'] == 0

            # If both mother and child have died the event doesnt run
            if ~df.at[person_id, 'is_alive'] and ~df.at[child_id, 'is_alive']:
                return

        # If the child is part of a twin pair
        elif df.at[child_id, 'nb_is_twin']:

            # Check this is both of their first visits
            child_two_id = df.at[child_id, 'nb_twin_sibling_id']
            assert df.at[child_id, 'pn_pnc_visits_neonatal'] == 0
            assert df.at[child_two_id, 'pn_pnc_visits_neonatal'] == 0

            # If neither of the twins or the mother is alive, the event doesnt run
            if ~df.at[person_id, 'is_alive'] and ~df.at[child_id, 'is_alive'] and ~df.at[child_two_id, 'is_alive']:
                return

        logger.debug(key='message', data=f'Mother {person_id} or child {child_id} have arrived for PNC1 on date'
                                         f' {self.sim.date}')

        # If the mother is alive she is assessed for complications
        if df.at[person_id, 'is_alive']:
            df.at[person_id, 'pn_pnc_visits_maternal'] += 1
            self.module.assessment_for_maternal_complication_during_pnc(person_id, self)

        # If the child/children is/are alive they are assessed for complications
        if df.at[child_id, 'is_alive']:
            df.at[child_id, 'pn_pnc_visits_neonatal'] += 1
            self.module.assessment_for_neonatal_complications_during_pnc(child_id, self, pnc_visit='pnc1')

        # If twins, the second child is also assessed
        if df.at[child_id, 'nb_is_twin']:
            if df.at[child_two_id, 'is_alive']:
                df.at[child_two_id, 'pn_pnc_visits_neonatal'] += 1
                self.module.assessment_for_neonatal_complications_during_pnc(child_two_id, self, pnc_visit='pnc1')

        # If either remain alive we determine if they will return for visit two
        # TODO: just realised im scheduling this event always using the mother id- even though its possible they could
        #  be dead (does that mater as long as it runs for the child?)
        pnc2 = HSI_PostnatalSupervisor_PostnatalCareContactTwo(self.module, person_id=person_id)
        self.module.maternal_postnatal_care_care_seeking(person_id, 42, 'pnc2', pnc2)

    def did_not_run(self):
        logger.debug(key='message', data='HSI_PostnatalSupervisor_PostnatalCareContactOne: did not run')

    def not_available(self):
        logger.debug(key='message', data='HSI_PostnatalSupervisor_PostnatalCareContactOne: cannot not run with '
                                         'this configuration')


class HSI_PostnatalSupervisor_PostnatalCareContactTwo(HSI_Event, IndividualScopeEventMixin):
    """ This is HSI_PostnatalSupervisor_PostnatalCareContactTwoMaternal. It is scheduled by
    HSI_PostnatalSupervisor_PostnatalCareContactOneMaternal This event is the second PNC visit women
    are recommended to undertake at 6 weeks postnatal. """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, PostnatalSupervisor)

        self.TREATMENT_ID = 'PostnatalSupervisor_PostnatalCareContactTwoMaternal'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        child_id = int(df.at[person_id, 'pn_id_most_recent_child'])

        # We run checks to make sure only the correct people are passed to the event
        if df.at[person_id, 'is_alive']:
            assert df.at[person_id, 'la_is_postpartum']
            assert df.at[person_id, 'pn_pnc_visits_maternal'] == 1

        if ~df.at[child_id, 'nb_is_twin']:
            if df.at[child_id, 'is_alive']:
                assert df.at[child_id, 'pn_pnc_visits_neonatal'] == 1
            if ~df.at[person_id, 'is_alive'] and ~df.at[child_id, 'is_alive']:
                return

        elif df.at[child_id, 'nb_is_twin']:
            child_two_id = df.at[child_id, 'nb_twin_sibling_id']
            if df.at[child_id, 'is_alive']:
                assert df.at[child_id, 'pn_pnc_visits_neonatal'] == 1
            if df.at[child_two_id, 'is_alive']:
                assert df.at[child_two_id, 'pn_pnc_visits_neonatal'] == 1

            if ~df.at[person_id, 'is_alive'] and ~df.at[child_id, 'is_alive'] and ~df.at[child_two_id, 'is_alive']:
                return

        logger.debug(key='message', data=f'Mother {person_id} or child {child_id} have arrived for PNC2 on date'
                                         f' {self.sim.date}')

        # If the mother is alive she is assessed for complications
        if df.at[person_id, 'is_alive']:
            df.at[person_id, 'pn_pnc_visits_maternal'] += 1
            self.module.assessment_for_maternal_complication_during_pnc(person_id, self)

        # If the child/children is/are alive they are assessed for complications
        if df.at[child_id, 'is_alive']:
            df.at[child_id, 'pn_pnc_visits_neonatal'] += 1
            self.module.assessment_for_neonatal_complications_during_pnc(child_id, self, pnc_visit='pnc2')

        if df.at[child_id, 'nb_is_twin']:
            if df.at[child_two_id, 'is_alive']:
                df.at[child_two_id, 'pn_pnc_visits_neonatal'] += 1
                self.module.assessment_for_neonatal_complications_during_pnc(child_two_id, self, pnc_visit='pnc2')

    def did_not_run(self):
        logger.debug(key='message', data='HSI_PostnatalSupervisor_PostnatalCareContactTwo: did not run')

    def not_available(self):
        logger.debug(key='message', data='HSI_PostnatalSupervisor_PostnatalCareContactTwo: cannot not run with '
                                         'this configuration')


class HSI_PostnatalSupervisor_PostnatalWardInpatientCare(HSI_Event, IndividualScopeEventMixin):
    """This is HSI_PostnatalSupervisor_PostnatalWardInpatientCare. It is scheduled by the PostnatalSupervisorEvent for
    women who develop complications during the postnatal period and decide to seek care or by either of the PNC HSIs for
     women assessed as requiring further care due to complications detected. Treatment delivered in this event includes
     management of hypertensive disorders, maternal sepsis, anaemia and secondary postpartum haemorrhage"""
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, PostnatalSupervisor)

        self.TREATMENT_ID = 'PostnatalSupervisor_PostnatalWardInpatientCare'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'IPAdmission': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 5})

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        assert df.at[person_id, 'la_is_postpartum']

        if not mother.is_alive:
            return

        # These event cycles through each of the key maternal complications that a mother could be admitted due to and
        # delivers treatment accordingly

        # --------------------------------------- SEPSIS TREATMENT ---------------------------------------------------
        if mother.pn_sepsis_late_postpartum:
            # First check the availability of consumables for treatment
            pkg_code_sepsis = pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'Maternal sepsis case management',
                                'Intervention_Pkg_Code'])[0]

            all_available = self.get_all_consumables(
                pkg_codes=[pkg_code_sepsis])

            # If available then treatment is delivered
            if all_available:
                logger.debug(key='message',
                             data=f'mother {person_id} has received treatment for sepsis as an inpatient')

                df.at[person_id, 'pn_sepsis_late_postpartum_treatment'] = True
            else:
                logger.debug(key='message',
                             data=f'mother {person_id} was unable to receive treatment for sepsis due to '
                             f'limited resources')

        # ------------------------------------- SECONDARY PPH TREATMENT -----------------------------------------------
        if mother.pn_postpartum_haem_secondary:
            # First check the availability of consumables for treatment
            pkg_code_pph = pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'Treatment of postpartum hemorrhage',
                                'Intervention_Pkg_Code'])[0]

            all_available = self.get_all_consumables(
                pkg_codes=[pkg_code_pph])

            # If available then treatment is delivered
            if all_available:
                logger.debug(key='message', data=f'mother {person_id} has received treatment for secondary postpartum '
                                                 f'haemorrhage as an inpatient')
                df.at[person_id, 'pn_postpartum_haem_secondary_treatment'] = True

            else:
                logger.debug(key='message',
                             data=f'mother {person_id} was unable to receive treatment for secondary pph due '
                             f'to limited resources')

        # ------------------------------------- HYPERTENSION TREATMENT -----------------------------------------------
        # Treatment for complications of hypertension include two interventions, anti hypertensive therapy and
        # magnesium
        if mother.pn_htn_disorders == 'gest_htn' or mother.pn_htn_disorders == 'mild_pre_eclamp':
            approx_days_of_pn_period = (6 - df.at[person_id, 'pn_postnatal_period_in_weeks']) * 7

            # Define the consumables and check their availability
            item_code_methyldopa = pd.unique(
                consumables.loc[consumables['Items'] == 'Methyldopa 250mg_1000_CMST', 'Item_Code'])[0]

            consumables_gest_htn_treatment = {
                'Intervention_Package_Code': {},
                'Item_Code': {item_code_methyldopa: 4 * approx_days_of_pn_period}}

            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self,
                cons_req_as_footprint=consumables_gest_htn_treatment)

            # If they are available then the woman is started on treatment
            if outcome_of_request_for_consumables['Item_Code'][item_code_methyldopa]:
                df.at[person_id, 'pn_gest_htn_on_treatment'] = True
                logger.debug(key='msg', data=f'Mother {person_id} has been started on regular antihypertensives due to '
                                             f'her HDP')

        if mother.pn_htn_disorders == 'severe_pre_eclamp' or \
            mother.pn_htn_disorders == 'eclampsia' or\
           mother.pn_htn_disorders == 'severe_gest_htn':

            # Define required consumables
            item_code_hydralazine = pd.unique(
                consumables.loc[consumables['Items'] == 'Hydralazine, powder for injection, 20 mg ampoule',
                                'Item_Code'])[0]
            item_code_wfi = pd.unique(
                consumables.loc[consumables['Items'] == 'Water for injection, 10ml_Each_CMST', 'Item_Code'])[0]
            item_code_needle = pd.unique(
                consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]
            item_code_gloves = pd.unique(
                consumables.loc[consumables['Items'] == 'Gloves, exam, latex, disposable, pair', 'Item_Code'])[0]
            item_code_methyldopa = pd.unique(
                consumables.loc[consumables['Items'] == 'Methyldopa 250mg_1000_CMST', 'Item_Code'])[0]

            # As we need multiple of one consumable we use the old method to check availablity
            consumables_gest_htn_treatment = {
                'Intervention_Package_Code': {},
                'Item_Code': {item_code_hydralazine: 1, item_code_wfi: 1, item_code_needle: 1,
                              item_code_gloves: 1, item_code_methyldopa: 4}}

            # Then query if these consumables are available during this HSI
            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self,
                cons_req_as_footprint=consumables_gest_htn_treatment)

            # If they are available then the woman is started on treatment
            if (outcome_of_request_for_consumables['Item_Code'][item_code_hydralazine]) and \
                (outcome_of_request_for_consumables['Item_Code'][item_code_wfi]) and \
                (outcome_of_request_for_consumables['Item_Code'][item_code_needle]) and \
                (outcome_of_request_for_consumables['Item_Code'][item_code_gloves]) and \
               (outcome_of_request_for_consumables['Item_Code'][item_code_methyldopa]):
                df.at[person_id, 'pn_iv_anti_htn_treatment'] = True

                if df.at[person_id, 'pn_htn_disorders'] == 'severe_gest_htn':
                    df.at[person_id, 'pn_htn_disorders'] = 'gest_htn'

            if mother.pn_htn_disorders == 'severe_pre_eclamp' or mother.pn_htn_disorders == 'eclampsia':
                pkg_code_severe_pre_eclampsia = pd.unique(
                    consumables.loc[consumables['Intervention_Pkg'] == 'Management of eclampsia',
                                    'Intervention_Pkg_Code'])[0]

                all_available = self.get_all_consumables(
                    pkg_codes=[pkg_code_severe_pre_eclampsia])

                if all_available:
                    df.at[person_id, 'pn_mag_sulph_treatment'] = True
                    # TODO: reduce risk of progression from spe to ec when progression function when neater

        # ------------------------------------- ANAEMIA TREATMENT -----------------------------------------------
        if mother.pn_anaemia_following_pregnancy != 'none':
            pass
        # Todo: add v1.1

        # Following treatment we use this function to determine if this woman will survive
        self.module.apply_risk_of_maternal_or_neonatal_death_postnatal(mother_or_child='mother',
                                                                       individual_id=person_id)

    def did_not_run(self):
        person_id = self.target

        logger.debug(key='message', data='HSI_PostnatalSupervisor_PostnatalWardInpatientCare: did not run')

        # If they event can't run we use the death function to determine if the mother has died without care
        self.module.apply_risk_of_maternal_or_neonatal_death_postnatal(mother_or_child='mother',
                                                                       individual_id=person_id)
        return False

    def not_available(self):
        person_id = self.target

        logger.debug(key='message', data='HSI_PostnatalSupervisor_PostnatalWardInpatientCare: cannot not run with '
                                         'this configuration')

        # If they event can't run we use the death function to determine if the mother has died without care
        self.module.apply_risk_of_maternal_or_neonatal_death_postnatal(mother_or_child='mother',
                                                                       individual_id=person_id)


class HSI_PostnatalSupervisor_TreatmentForObstetricFistula(HSI_Event, IndividualScopeEventMixin):
    """This is HSI_PostnatalSupervisor_TreatmentForObstetricFistula. It is scheduled by the on_birth function of this
     module for women have developed obstetric fistula and choose to seek care. Treatment delivered in this event
    includes surgical correction of obstetric fistula
    """
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, PostnatalSupervisor)

        self.TREATMENT_ID = 'PostnatalSupervisor_TreatmentForObstetricFistula'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'InpatientDays': 5})
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 5})

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]

        if not mother.is_alive:
            return

        # todo: no relevant consumables package identified

        self.sim.modules['PregnancySupervisor'].store_dalys_in_mni(
            person_id, f'{df.at[person_id, "pn_obstetric_fistula"]}_fistula_resolution')

        df.at[person_id, 'pn_obstetric_fistula'] = 'none'
        logger.debug(key='msg', data=f'Mother {person_id} has undergone surgical correction of her obstetric fistula')


class HSI_PostnatalSupervisor_NeonatalWardInpatientCare(HSI_Event, IndividualScopeEventMixin):
    """This is HSI_PostnatalSupervisor_NeonatalWardInpatientCare. It is scheduled by any of the PNC HSIs for neonates
    who are require inpatient care due to a complication of the postnatal period. Treatment is delivered in this
    event"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, PostnatalSupervisor)

        self.TREATMENT_ID = 'PostnatalSupervisor_NeonatalWardInpatientCare'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'IPAdmission': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 5})

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        child = df.loc[person_id]

        if not child.is_alive:
            return

        # Here we deliver treatment to any neonates who have been admitted to either early or late onset sepsis
        if child.pn_sepsis_early_neonatal or child.pn_sepsis_late_neonatal:
            pkg_code_sep = pd.unique(consumables.loc[
                                         consumables['Intervention_Pkg'] == 'Newborn sepsis - full supportive care',
                                         'Intervention_Pkg_Code'])[0]

            all_available = self.get_all_consumables(
                pkg_codes=[pkg_code_sep])

            if all_available:
                df.at[person_id, 'pn_sepsis_neonatal_full_supp_care'] = True
            else:
                logger.debug(key='message', data=f'neonate {person_id} was unable to receive treatment for sepsis '
                                                 f'due to limited resources')

            self.module.apply_risk_of_maternal_or_neonatal_death_postnatal(mother_or_child='child',
                                                                           individual_id=person_id)

    def did_not_run(self):
        person_id = self.target

        logger.debug(key='message', data='HSI_PostnatalSupervisor_NeonatalWardInpatientCare: did not run')

        # If they event can't run we use the death function to determine if the neonate has died without care
        self.module.apply_risk_of_maternal_or_neonatal_death_postnatal(mother_or_child='child',
                                                                       individual_id=person_id)

        # In order to make sure all women have risk of death applied we dont allow this event to run again
        return False

    def not_available(self):
        person_id = self.target

        logger.debug(key='message', data='HSI_PostnatalSupervisor_NeonatalWardInpatientCare: cannot not run with '
                                         'this configuration')
        self.module.apply_risk_of_maternal_or_neonatal_death_postnatal(mother_or_child='child',
                                                                       individual_id=person_id)


class PostnatalLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    This is the PostnatalLoggingEvent. It runs every year and uses the dataframe and postnatal_tracker to generate
    summary statistics used in analysis files. This is not a finalised event and is liable to change
    """
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

        # ra_lower_limit = 14
        # ra_upper_limit = 50
        # women_reproductive_age = df.index[(df.is_alive & (df.sex == 'F') & (df.age_years > ra_lower_limit) &
        #                                   (df.age_years < ra_upper_limit))]
        # total_women_reproductive_age = len(women_reproductive_age)

        total_pph = self.module.postnatal_tracker['secondary_pph']
        total_pn_death = self.module.postnatal_tracker['postnatal_death']
        total_sepsis = self.module.postnatal_tracker['postnatal_sepsis']
        total_fistula = self.module.postnatal_tracker['fistula']
        total_endo = self.module.postnatal_tracker['endometritis']
        total_uti = self.module.postnatal_tracker['urinary_tract_inf']
        total_ssti = self.module.postnatal_tracker['skin_soft_tissue_inf']
        total_other_inf = self.module.postnatal_tracker['other_maternal_infection']
        total_anaemia = self.module.postnatal_tracker['postnatal_anaemia']

        maternal_dict_for_output = {'total_fistula': total_fistula,
                                    'total_endo': total_endo,
                                    'total_uti': total_uti,
                                    'total_ssti': total_ssti,
                                    'total_other_inf': total_other_inf,
                                    'total_anaemia': total_anaemia,
                                    'total_pph': total_pph,
                                    'total_sepsis': total_sepsis,
                                    'total_deaths': total_pn_death,
                                    'pn_mmr': (total_pn_death/total_births_last_year) * 100000}

        logger.info(key='postnatal_maternal_summary_stats', data=maternal_dict_for_output,
                    description='Yearly maternal summary statistics output from the postnatal supervisor module')

        total_early_sepsis = self.module.postnatal_tracker['early_neonatal_sepsis']
        total_late_sepsis = self.module.postnatal_tracker['late_neonatal_sepsis']
        total_pn_neonatal_deaths = self.module.postnatal_tracker['neonatal_death']

        neonatal_dict_for_output = {'eons': total_early_sepsis,
                                    'lons': total_late_sepsis,
                                    'total_deaths': total_pn_neonatal_deaths,
                                    'pn_nmr': (total_pn_neonatal_deaths/total_births_last_year) * 1000}

        logger.info(key='postnatal_neonatal_summary_stats', data=neonatal_dict_for_output,
                    description='Yearly neonatal summary statistics output from the postnatal supervisor module')

        self.module.postnatal_tracker = {'endometritis': 0,
                                         'urinary_tract_inf': 0,
                                         'skin_soft_tissue_inf': 0,
                                         'other_maternal_infection': 0,
                                         'secondary_pph': 0,
                                         'postnatal_death': 0,
                                         'postnatal_sepsis': 0,
                                         'fistula': 0,
                                         'postnatal_anaemia': 0,
                                         'early_neonatal_sepsis': 0,
                                         'late_neonatal_sepsis': 0,
                                         'neonatal_death': 0,
                                         'neonatal_sepsis_death': 0}
