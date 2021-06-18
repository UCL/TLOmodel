from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging, util
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel
from tlo.methods import Metadata, postnatal_supervisor_lm
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

        # First we define dictionaries which will store the current parameters of interest (to allow parameters to
        # change between 2010 and 2020) and the linear models
        self.current_parameters = dict()
        self.pn_linear_models = dict()

    METADATA = {Metadata.DISEASE_MODULE,
                Metadata.USES_HEALTHSYSTEM,
                Metadata.USES_HEALTHBURDEN}  # declare that this is a disease module (leave as empty set otherwise)

    PARAMETERS = {
        # OBSTETRIC FISTULA
        'prob_obstetric_fistula': Parameter(
            Types.LIST, 'probability of a woman developing an obstetric fistula after birth'),
        'rr_obstetric_fistula_obstructed_labour': Parameter(
            Types.LIST, 'relative risk of obstetric fistula in women who experienced obstructed labour'),
        'prevalence_type_of_fistula': Parameter(
            Types.LIST, 'prevalence of 1.) vesicovaginal 2.)rectovaginal fistula '),

        # SECONDARY POSTPARTUM HAEMORRHAGE
        'prob_secondary_pph': Parameter(
            Types.LIST, 'baseline probability of secondary PPH'),
        'rr_secondary_pph_endometritis': Parameter(
            Types.LIST, 'relative risk of secondary postpartum haemorrhage in women with sepsis secondary to '
                        'endometritis'),
        'prob_secondary_pph_severity': Parameter(
            Types.LIST, 'probability of mild, moderate or severe secondary PPH'),
        'cfr_secondary_postpartum_haemorrhage_pn': Parameter(
            Types.LIST, 'case fatality rate for secondary pph'),

        # HYPERTENSIVE DISORDERS
        'prob_htn_resolves': Parameter(
            Types.LIST, 'Weekly probability of resolution of hypertension'),
        'weekly_prob_gest_htn_pn': Parameter(
            Types.LIST, 'weekly probability of a woman developing gestational hypertension during the postnatal '
                        'period'),
        'rr_gest_htn_obesity': Parameter(
            Types.LIST, 'Relative risk of gestational hypertension for women who are obese'),
        'prob_pre_eclampsia_per_month': Parameter(
            Types.LIST, 'underlying risk of pre-eclampsia per month without the impact of risk factors'),
        'weekly_prob_pre_eclampsia_pn': Parameter(
            Types.LIST, 'weekly probability of a woman developing mild pre-eclampsia during the postnatal period'),
        'rr_pre_eclampsia_obesity': Parameter(
            Types.LIST, 'Relative risk of pre-eclampsia for women who are obese'),
        'rr_pre_eclampsia_chronic_htn': Parameter(
            Types.LIST, 'Relative risk of pre-eclampsia in women who are chronically hypertensive'),
        'rr_pre_eclampsia_diabetes_mellitus': Parameter(
            Types.LIST, 'Relative risk of pre-eclampsia in women who have diabetes mellitus'),
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
        'cfr_eclampsia_pn': Parameter(
            Types.LIST, 'case fatality rate of eclampsia in the postnatal period'),
        'cfr_severe_htn_pn': Parameter(
            Types.LIST, 'case fatality rate of severe hypertension in the postnatal period'),
        # todo: death from severe pe?

        # ANAEMIA
        'baseline_prob_anaemia_per_week': Parameter(
            Types.LIST, 'Weekly probability of anaemia in pregnancy'),
        'rr_anaemia_if_iron_deficient_pn': Parameter(
            Types.LIST, 'risk of developing anaemia when iron deficient'),
        'rr_anaemia_if_folate_deficient_pn': Parameter(
            Types.LIST, 'risk of developing anaemia when folate deficient'),
        'rr_anaemia_if_b12_deficient_pn': Parameter(
            Types.LIST, 'risk of developing anaemia when b12 deficient'),
        'rr_anaemia_maternal_malaria': Parameter(
            Types.LIST, 'relative risk of anaemia secondary to malaria infection'),
        'rr_anaemia_recent_haemorrhage': Parameter(
            Types.LIST, 'relative risk of anaemia secondary to recent haemorrhage'),
        'rr_anaemia_hiv_no_art': Parameter(
            Types.LIST, 'relative risk of anaemia for a woman with HIV not on ART'),
        'prob_iron_def_per_week_pn': Parameter(
            Types.LIST, 'weekly probability of a women developing iron deficiency following pregnancy'),
        'prob_folate_def_per_week_pn': Parameter(
            Types.LIST, 'weekly probability of a women developing folate deficiency following pregnancy '),
        'prob_b12_def_per_week_pn': Parameter(
            Types.LIST, 'weekly probability of a women developing b12 deficiency following pregnancy '),
        'prob_type_of_anaemia_pn': Parameter(
            Types.LIST, 'probability of a woman with anaemia having mild, moderate or severe anaemia'),

        # MATERNAL SEPSIS
        'prob_late_sepsis_endometritis': Parameter(
            Types.LIST, 'probability of developing sepsis following postpartum endometritis infection'),
        'rr_sepsis_endometritis_post_cs': Parameter(
            Types.LIST, 'relative risk of endometritis following caesarean delivery'),
        'prob_late_sepsis_urinary_tract': Parameter(
            Types.LIST, 'probability of developing sepsis following postpartum UTI'),
        'prob_late_sepsis_skin_soft_tissue': Parameter(
            Types.LIST, 'probability of developing sepsis following postpartum skin/soft tissue infection'),
        'rr_sepsis_sst_post_cs': Parameter(
            Types.LIST, 'relative risk of skin/soft tissue sepsis following caesarean delivery'),
        'cfr_postpartum_sepsis_pn': Parameter(
            Types.LIST, 'case fatality rate for postnatal sepsis'),

        # NEWBORN SEPSIS
        'prob_early_onset_neonatal_sepsis_week_1': Parameter(
            Types.LIST, 'Baseline probability of a newborn developing sepsis in week one of life'),
        'rr_eons_maternal_chorio': Parameter(
            Types.LIST, 'relative risk of EONS in newborns whose mothers have chorioamnionitis'),
        'rr_eons_maternal_prom': Parameter(
            Types.LIST, 'relative risk of EONS in newborns whose mothers have PROM'),
        'rr_eons_preterm_neonate': Parameter(
            Types.LIST, 'relative risk of EONS in preterm newborns'),
        'cfr_early_onset_neonatal_sepsis': Parameter(
            Types.LIST, 'case fatality for early onset neonatal sepsis'),
        'prob_late_onset_neonatal_sepsis': Parameter(
            Types.LIST, 'probability of late onset neonatal sepsis (all cause)'),
        'cfr_late_neonatal_sepsis': Parameter(
            Types.LIST, 'Risk of death from late neonatal sepsis'),
        'prob_sepsis_disabilities': Parameter(
            Types.LIST, 'Probabilities of varying disability levels after neonatal sepsis'),

        # CARE SEEKING
        'odds_will_attend_pnc': Parameter(
            Types.LIST, 'baseline odss a woman will seek PNC for her and her newborn following delivery'),
        'or_pnc_age_30_35': Parameter(
            Types.LIST, 'odds ratio for women aged 30-35 attending PNC'),
        'or_pnc_age_>35': Parameter(
            Types.LIST, 'odds ratio for women aged > 35 attending PNC'),
        'or_pnc_rural': Parameter(
            Types.LIST, 'odds ratio for women who live rurally attending PNC'),
        'or_pnc_wealth_level_1': Parameter(
            Types.LIST, 'odds ratio for women from the highest wealth level attending PNC'),
        'or_pnc_anc4+': Parameter(
            Types.LIST, 'odds ratio for women who attended ANC4+ attending PNC'),
        'or_pnc_caesarean_delivery': Parameter(
            Types.LIST, 'odds ratio for women who delivered by caesarean attending PNC'),
        'or_pnc_facility_delivery': Parameter(
            Types.LIST, 'odds ratio for women who delivered in a health facility attending PNC'),
        'or_pnc_parity_>4': Parameter(
            Types.LIST, 'odds ratio for women with a parity of >4 attending PNC'),
        'prob_attend_pnc2': Parameter(
            Types.LIST, 'Probability that a woman receiving PNC1 care will return for PNC2 care'),
        'prob_attend_pnc3': Parameter(
            Types.LIST, 'Probability that a woman receiving PNC2 care will return for PNC3 care'),
        'prob_care_seeking_postnatal_emergency': Parameter(
            Types.LIST, 'baseline probability of emergency care seeking for women in the postnatal period'),
        'prob_care_seeking_postnatal_emergency_neonate': Parameter(
            Types.LIST, 'baseline probability care will be sought for a neonate with a complication'),
        'odds_care_seeking_fistula_repair': Parameter(
            Types.LIST, 'odds of a woman seeking care for treatment of obstetric fistula'),
        'aor_cs_fistula_age_15_19': Parameter(
            Types.LIST, 'odds ratio for care seeking for treatment of obstetric fistula in 15-19 year olds'),
        'aor_cs_fistula_age_lowest_education': Parameter(
            Types.LIST, 'odds ratio for care seeking for treatment of obstetric fistula in women in the lowest '
                        'education quintile'),

        # TREATMENT EFFECTS
        'treatment_effect_iron_folic_acid_anaemia': Parameter(
            Types.LIST, 'relative effect of daily iron and folic acid treatment on risk of maternal anaemia '),
        'rr_iron_def_ifa_pn': Parameter(
            Types.LIST, 'effect of iron and folic acid treatment on risk of iron deficiency'),
        'rr_folate_def_ifa_pn': Parameter(
            Types.LIST, 'effect of iron and folic acid treatment on risk of folate deficiency '),
        'effect_of_ifa_for_resolving_anaemia': Parameter(
            Types.LIST, 'treatment effectiveness of starting iron and folic acid on resolving anaemia'),
        'treatment_effect_blood_transfusion_anaemia': Parameter(
            Types.LIST, 'treatment effectiveness of blood transfusion for anaemia following pregnancy'),
        'treatment_effect_early_init_bf': Parameter(
            Types.LIST, 'effect of early initiation of breastfeeding on neonatal sepsis rates '),
        'treatment_effect_abx_prom': Parameter(
            Types.LIST, 'effect of early antibiotics given to a mother with PROM on neonatal sepsis rates '),
        'treatment_effect_inj_abx_sep': Parameter(
            Types.LIST, 'effect of injectable antibiotics on neonatal sepsis mortality'),
        'treatment_effect_supp_care_sep': Parameter(
            Types.LIST, 'effect of full supportive care on neonatal sepsis mortality'),
        'treatment_effect_parenteral_antibiotics': Parameter(
            Types.LIST, 'Treatment effect of parenteral antibiotics on maternal sepsis mortality '),
        'treatment_effect_bemonc_care_pph': Parameter(
            Types.LIST, 'Treatment effect of BEmONC care on postpartum haemorrhage mortality'),
        'treatment_effect_anti_htns': Parameter(
            Types.LIST, 'Treatment effect of hypertensive therapy on death from eclampsia'),
        'treatment_effect_mag_sulph': Parameter(
            Types.LIST, 'Treatment effect of magnesium sulphate therapy on death from eclampsia'),
        'neonatal_sepsis_treatment_effect': Parameter(
            Types.LIST, 'Treatment effect for neonatal sepsis'),
        'treatment_effect_clean_birth': Parameter(
            Types.LIST, 'Treatment effect of clean birth practices on early onset neonatal sepsis risk'),
        'treatment_effect_cord_care': Parameter(
            Types.LIST, 'Treatment effect of chlorhexidine cord care on early onset neonatal sepsis risk'),
        'treatment_effect_anti_htns_progression_pn': Parameter(
            Types.LIST, 'Treatment effect of oral anti hypertensives on progression from mild/mod to severe gestational'
                        'hypertension'),

        # ASSESSMENT SENSITIVITIES
        'sensitivity_bp_monitoring_pn': Parameter(
            Types.LIST, 'sensitivity of BP monitoring during PNC'),
        'specificity_bp_monitoring_pn': Parameter(
            Types.LIST, 'specificity of BP monitoring during PNC'),
        'sensitivity_urine_protein_1_plus_pn': Parameter(
            Types.LIST, 'sensitivity of urine dipstick during PNC'),
        'specificity_urine_protein_1_plus_pn': Parameter(
            Types.LIST, 'specificity of urine dipstick monitoring during PNC'),
        'sensitivity_poc_hb_test_pn': Parameter(
            Types.LIST, 'sensitivity of point of care testing during PNC'),
        'specificity_poc_hb_test_pn': Parameter(
            Types.LIST, 'specificity of urine dipstick monitoring during PNC'),
        'sensitivity_fbc_hb_test_pn': Parameter(
            Types.LIST, 'sensitivity of full blood count'),
        'specificity_fbc_hb_test_pn': Parameter(
            Types.LIST, 'specificity of full blood count'),
        'sensitivity_maternal_sepsis_assessment': Parameter(
            Types.LIST, 'sensitivity of assessment for maternal sepsis'),
        'sensitivity_pph_assessment': Parameter(
            Types.LIST, 'sensitivity of assessment for secondary pph'),
        'sensitivity_lons_assessment': Parameter(
            Types.LIST, 'sensitivity of assessment for late onset neonatal sepsis'),
        'sensitivity_eons_assessment': Parameter(
            Types.LIST, 'sensitivity of assessment for earl onset neonatal sepsis'),
        'prob_intervention_delivered_sep_assessment_pnc': Parameter(
            Types.LIST, 'probability a woman will be assessed for sepsis during PNC'),
        'prob_intervention_delivered_pph_assessment_pnc': Parameter(
            Types.LIST, 'probability a woman will be assessed for PPH during PNC'),
        'prob_intervention_delivered_urine_ds_pnc': Parameter(
            Types.LIST, 'probability a woman will receive a urine disptick during PNC'),
        'prob_intervention_delivered_bp_pnc': Parameter(
            Types.LIST, 'probability a woman will receive blood pressure testing during PNC'),
        'prob_intervention_poct_pnc': Parameter(
            Types.LIST, 'probability a woman will receive point of care Hb testing during PNC'),
        'prob_intervention_neonatal_sepsis_pnc': Parameter(
            Types.LIST, 'probability a newborn will be assessed for sepsis during PNC'),
        'prob_intervention_delivered_hiv_test_pnc': Parameter(
            Types.LIST, 'probability a newborn will receive a HIV test at 6 weeks postnatal '),
        'prob_intervention_delivered_depression_screen_pnc': Parameter(
            Types.LIST, 'probability a mother will receive a depression screen during PNC'),
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
        parameter_dataframe = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_PostnatalSupervisor.xlsx',
                                            sheet_name='parameter_values')
        self.load_parameters_from_dataframe(parameter_dataframe)

        # For the first period (2010-2015) we use the first value in each list as a parameter
        for key, value in self.parameters.items():
            self.current_parameters[key] = self.parameters[key][0]

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

        # Register dx_tests used as assessment for postnatal conditions during PNC visits
        params = self.current_parameters

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

            full_blood_count_hb_pn=DxTest(
                property='pn_anaemia_following_pregnancy', target_categories=['mild', 'moderate', 'severe'],
                sensitivity=params['sensitivity_fbc_hb_test_pn'],
                specificity=params['specificity_fbc_hb_test_pn']),

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

        self.pn_linear_models = {

            # This equation is used to determine a mothers risk of developing obstetric fistula after birth
            'obstetric_fistula': LinearModel.custom(postnatal_supervisor_lm.predict_obstetric_fistula,
                                                    parameters=params),

            # This equation is used to determine a mothers risk of secondary postpartum haemorrhage
            'secondary_postpartum_haem': LinearModel.custom(postnatal_supervisor_lm.predict_secondary_postpartum_haem,
                                                            parameters=params),

            # This equation is used to determine a mothers risk of dying following a secondary postpartum haemorrhage.
            # Risk of death is modified by the effect of treatment, if delivered
            'secondary_postpartum_haemorrhage_death': LinearModel.custom(
                postnatal_supervisor_lm.predict_secondary_postpartum_haem_death, parameters=params),

            # This equation is used to determine a mothers risk of developing a sepsis secondary to endometritis
            'sepsis_endometritis_late_postpartum': LinearModel.custom(
                postnatal_supervisor_lm.predict_sepsis_endometritis_late_postpartum, parameters=params),

            # This equation is used to determine a mothers risk of developing sepsis secondary to skin or soft tissue
            # infection
            'sepsis_sst_late_postpartum': LinearModel.custom(
                postnatal_supervisor_lm.predict_sepsis_sst_late_postpartum, parameters=params),

            # This equation is used to determine a mothers risk of dying due to sepsis in the postnatal period. Risk of
            # death is modified by the effect of treatment, if delivered
            'postpartum_sepsis_death': LinearModel.custom(postnatal_supervisor_lm.predict_postnatal_sepsis_death,
                                                         parameters=params),

            # This equation is used to determine a mothers risk of developing gestational hypertension in the postnatal
            # period
            'gest_htn_pn': LinearModel.custom(postnatal_supervisor_lm.predict_gest_htn_pn, parameters=params),

            # This equation is used to determine a mothers risk of developing pre-eclampsia in in the postnatal
            # period
            'pre_eclampsia_pn': LinearModel.custom(postnatal_supervisor_lm.predict_pre_eclampsia_pn, parameters=params),

            # This equation is used to determine a mothers risk of dying from eclampsia that has developed in the
            # postnatal period. Risk of death is mitigated by treatment effects, if treatment is delivered
            'eclampsia_death': LinearModel.custom(postnatal_supervisor_lm.predict_eclampsia_death_pn,
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
        params = self.current_parameters
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        store_dalys_in_mni = self.sim.modules['PregnancySupervisor'].store_dalys_in_mni

        # We store the ID number of the child this woman has most recently delivered as a property of the woman. This is
        # because PNC is scheduled for the woman during the Labour Module but must act on both mother and child. If this
        # is a twin birth, only the id of the first born child is stored
        df.at[mother_id, 'pn_id_most_recent_child'] = child_id

        if df.at[mother_id, 'is_alive']:
            # Here we determine if, following childbirth, this woman will develop a fistula
            risk_of_fistula = self.pn_linear_models[
                'obstetric_fistula'].predict(df.loc[[mother_id]])[mother_id]

            # todo: link with PNC (assessment), allow treatment post day 42 (dalys?)

            if self.rng.random_sample() < risk_of_fistula:
                # We determine the specific type of fistula this woman is experiencing, to match with DALY weights
                fistula_type = self.rng.choice(['vesicovaginal', 'rectovaginal'],
                                               p=params['prevalence_type_of_fistula'])
                df.at[mother_id, 'pn_obstetric_fistula'] = fistula_type

                # Store the onset weight for daly calculations
                store_dalys_in_mni(mother_id, f'{fistula_type}_fistula_onset')

                logger.info(key='maternal_complication', data={'mother': mother_id,
                                                               'type': f'{fistula_type}_fistula',
                                                               'timing': 'postnatal'})

                # Determine if she will seek care for repair
                care_seeking_for_repair = self.pn_linear_models[
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
            if df.at[mother_id, 'ps_htn_disorders'] != 'none':
                df.at[mother_id, 'pn_htn_disorders'] = df.at[mother_id, 'ps_htn_disorders']

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

    def on_hsi_alert(self, person_id, treatment_id):
        logger.debug(key='message', data=f'This is PostnatalSupervisor, being alerted about a health system '
                                         f'interaction person {person_id} for: {treatment_id}')

    def report_daly_values(self):
        logger.debug(key='message', data='This is PostnatalSupervisor reporting my health values')
        df = self.sim.population.props

        daly_series = pd.Series(data=0, index=df.index[df.is_alive])

        return daly_series

    def apply_linear_model(self, lm, df):
        """
        Helper function will apply the linear model (lm) on the dataframe (df) to get a probability of some event
        happening to each individual. It then returns a series with same index with bools indicating the outcome based
        on the toss of the biased coin.
        :param lm: The linear model
        :param df: The dataframe
        :return: Series with same index containing outcomes (bool)
        """
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        mni_dict = pd.DataFrame.from_dict(mni, orient='index')

        mode_of_delivery = pd.Series(False, index=df.index)
        received_abx_for_prom = pd.Series(False, index=df.index)
        endometritis = pd.Series(False, index=df.index)
        chorio = pd.Series(False, index=df.index)

        if 'mode_of_delivery' in mni_dict.keys():
            mode_of_delivery = pd.Series(mni_dict['mode_of_delivery'], index=df.index)

        if 'abx_for_prom_given' in mni_dict.keys():
            received_abx_for_prom = pd.Series(mni_dict['abx_for_prom_given'], index=df.index)

        if 'chorio_lab' in mni_dict.keys():
            chorio = (df['ps_chorioamnionitis']) | (df['la_sepsis'] & mni_dict['chorio_lab'])

        if 'endo_pp' in mni_dict.keys():
            endometritis = pd.Series(mni_dict['endo_pp'], index=df.index)

        maternal_prom = pd.Series(df['ps_premature_rupture_of_membranes'], index=df.index)

        return self.rng.random_sample(len(df)) < lm.predict(df,
                                                            mode_of_delivery=mode_of_delivery,
                                                            received_abx_for_prom=received_abx_for_prom,
                                                            maternal_prom=maternal_prom,
                                                            endometritis=endometritis,
                                                            maternal_chorioamnionitis=chorio,
                                                            )

    def set_postnatal_complications_mothers(self, week):
        """
        This function is called by the PostnatalSupervisor event. It applies risk of key complications to a subset of
        women during each week of the postnatal period starting from week 2. Currently this includes infection, sepsis,
        anaemia and hypertension
        :param week: week in the postnatal period used to select women in the data frame.
         """
        df = self.sim.population.props
        params = self.current_parameters
        store_dalys_in_mni = self.sim.modules['PregnancySupervisor'].store_dalys_in_mni
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info


        def onset(eq):
            """
            Runs a specific equation within the linear model for the appropriate subset of women in the postnatal period
             and returns a BOOL series
            :param eq: linear model equation
            :return: BOOL series
            """
            onset_condition = self.apply_linear_model(
                self.pn_linear_models[f'{eq}'],
                df.loc[df['is_alive'] & df['la_is_postpartum'] & (df['pn_postnatal_period_in_weeks'] == week) &
                       ~df['hs_is_inpatient']])
            return onset_condition

        # -------------------------------------- SEPSIS --------------------------------------------------------------
        onset_sepsis_endo = onset('sepsis_endometritis_late_postpartum')
        onset_sepsis_sst = onset('sepsis_sst_late_postpartum')
        at_sepsis_urinary = df.is_alive & df.la_is_postpartum & (df.pn_postnatal_period_in_weeks == week) &\
                             ~df.hs_is_inpatient

        onset_sepsis_urinary = pd.Series(
            self.rng.random_sample(len(at_sepsis_urinary.loc[at_sepsis_urinary])) <
            params['prob_late_sepsis_urinary_tract'], index=at_sepsis_urinary.loc[at_sepsis_urinary].index)

        df.loc[onset_sepsis_endo.loc[onset_sepsis_endo].index, 'pn_sepsis_late_postpartum'] = True
        df.loc[onset_sepsis_endo.loc[onset_sepsis_endo].index, 'pn_emergency_event_mother'] = True
        onset_sepsis_endo.loc[onset_sepsis_endo].index.to_series().apply(store_dalys_in_mni,
                                                                         mni_variable='sepsis_onset')
        for person in onset_sepsis_endo.loc[onset_sepsis_endo].index:
            mni[person]['endo_pp'] = True

        df.loc[onset_sepsis_sst.loc[onset_sepsis_sst].index, 'pn_sepsis_late_postpartum'] = True
        df.loc[onset_sepsis_sst.loc[onset_sepsis_sst].index, 'pn_emergency_event_mother'] = True
        onset_sepsis_sst.loc[onset_sepsis_sst].index.to_series().apply(store_dalys_in_mni, mni_variable='sepsis_onset')

        df.loc[onset_sepsis_urinary.loc[onset_sepsis_urinary].index, 'pn_sepsis_late_postpartum'] = True
        df.loc[onset_sepsis_urinary.loc[onset_sepsis_urinary].index, 'pn_emergency_event_mother'] = True
        onset_sepsis_urinary.loc[onset_sepsis_urinary].index.to_series().apply(store_dalys_in_mni,
                                                                               mni_variable='sepsis_onset')

        # todo: this doesent tell us the primary source of infection?
        new_sepsis = df.is_alive & df.la_is_postpartum & (df.pn_postnatal_period_in_weeks == week) &\
                             ~df.hs_is_inpatient & df.pn_sepsis_late_postpartum

        for person in new_sepsis.loc[new_sepsis].index:
            logger.info(key='maternal_complication', data={'mother': person,
                                                           'type': 'sepsis',
                                                           'timing': 'postnatal'})

        # ------------------------------------ SECONDARY PPH ----------------------------------------------------------
        # Next we determine if any women will experience postnatal bleeding
        onset_pph = onset('secondary_postpartum_haem')
        df.loc[onset_pph.loc[onset_pph].index, 'pn_postpartum_haem_secondary'] = True

        # And set the emergency property
        df.loc[onset_pph.loc[onset_pph].index, 'pn_emergency_event_mother'] = True
        for person in onset_pph.loc[onset_pph].index:
            store_dalys_in_mni(person, 'secondary_pph_onset')
            logger.info(key='maternal_complication', data={'mother': person,
                                                           'type': 'secondary_postpartum_haemorrhage',
                                                           'timing': 'postnatal'})

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
            for person in new_def.loc[new_def].index:
                logger.info(key='maternal_complication', data={'mother': person,
                                                               'type': f'{deficiency}_deficiency',
                                                               'timing': 'postnatal'})
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

                for person in new_def.loc[new_def].index:
                    logger.info(key='maternal_complication', data={'mother': person,
                                                                   'type': f'{deficiency}_deficiency',
                                                                   'timing': 'postnatal'})

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
            logger.info(key='maternal_complication', data={'mother': person,
                                                           'type': f'{df.at[person, "pn_anaemia_following_pregnancy"]}'
                                                                   f'_anaemia',
                                                           'timing': 'postnatal'})

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

            risk_ghtn_remains_mild = 1 - (risk_of_gest_htn_progression + params['probs_for_mgh_matrix_pn'][2])

            # update the probability matrix according to treatment
            params['probs_for_mgh_matrix_pn'] = [risk_ghtn_remains_mild, risk_of_gest_htn_progression,
                                              params['probs_for_mgh_matrix_pn'][2], 0.0, 0.0]

            prob_matrix['gest_htn'] = params['probs_for_mgh_matrix_pn']
            prob_matrix['severe_gest_htn'] = params['probs_for_sgh_matrix_pn']
            prob_matrix['mild_pre_eclamp'] = params['probs_for_mpe_matrix_pn']
            prob_matrix['severe_pre_eclamp'] = params['probs_for_spe_matrix_pn']

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
                for person in new_onset_severe_pre_eclampsia.index:
                    df.at[person, 'pn_emergency_event_mother'] = True
                    logger.info(key='maternal_complication', data={'mother': person,
                                                                   'type': 'severe_pre_eclamp',
                                                                   'timing': 'postnatal'})

            # This process is repeated for women who have now developed eclampsia
            assess_status_change_for_eclampsia = (current_status != "eclampsia") & (new_status == "eclampsia")
            new_onset_eclampsia = assess_status_change_for_eclampsia[assess_status_change_for_eclampsia]

            if not new_onset_eclampsia.empty:
                for person in new_onset_eclampsia.index:
                    df.at[person, 'pn_emergency_event_mother'] = True
                    logger.info(key='maternal_complication', data={'mother': person,
                                                                   'type': 'eclampsia',
                                                                   'timing': 'postnatal'})

            # And severe gestational hypertension - which doesnt trigger care seeking
            assess_status_change_for_sgh = (current_status != "severe_gest_htn") & (new_status == "severe_gest_htn")
            new_onset_sgh = assess_status_change_for_sgh[assess_status_change_for_sgh]

            for person in new_onset_sgh.index:
                logger.info(key='maternal_complication', data={'person': person,
                                                               'type': 'severe_gest_htn',
                                                               'timing': 'postnatal'})

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

        risk_progression_mild_to_severe_htn = params['probs_for_mgh_matrix_pn'][1]

        apply_risk(women_with_htn_not_on_anti_htns, risk_progression_mild_to_severe_htn)
        apply_risk(women_with_htn_on_anti_htns, (risk_progression_mild_to_severe_htn *
                                                 params['treatment_effect_anti_htns_progression_pn']))

        #  -------------------------------- RISK OF PRE-ECLAMPSIA HYPERTENSION --------------------------------------
        # Here we apply a risk to women developing de-novo hypertensive in the later postnatal period
        pre_eclampsia = self.apply_linear_model(
            self.pn_linear_models['pre_eclampsia_pn'],
            df.loc[df['is_alive'] & df['la_is_postpartum'] & (df['pn_postnatal_period_in_weeks'] == week) &
                   (df['pn_htn_disorders'] == 'none')])

        df.loc[pre_eclampsia.loc[pre_eclampsia].index, 'ps_prev_pre_eclamp'] = True
        df.loc[pre_eclampsia.loc[pre_eclampsia].index, 'pn_htn_disorders'] = 'mild_pre_eclamp'

        if not pre_eclampsia.loc[pre_eclampsia].empty:
            logger.debug(key='message', data=f'The following women have developed pre_eclampsia in week {week} of the'
                                             f' postnatal period: {pre_eclampsia.loc[pre_eclampsia].index}')

        #  -------------------------------- RISK OF GESTATIONAL HYPERTENSION --------------------------------------
        gest_hypertension = self.apply_linear_model(
            self.pn_linear_models['gest_htn_pn'],
            df.loc[df['is_alive'] & df['la_is_postpartum'] & (df['pn_postnatal_period_in_weeks'] == week) &
                   (df['pn_htn_disorders'] == 'none')])

        df.loc[gest_hypertension.loc[gest_hypertension].index, 'pn_htn_disorders'] = 'gest_htn'

        if not gest_hypertension.loc[gest_hypertension].empty:
            logger.debug(key='message', data=f'The following women have developed gestational hypertension in week '
                                             f'{week} of the postnatal period '
                                             f'{gest_hypertension.loc[gest_hypertension].index}')

        # -------------------------------- RISK OF DEATH SEVERE HYPERTENSION ------------------------------------------
        # Risk of death is applied to women with severe hypertensive disease
        at_risk_of_death_htn = self.apply_linear_model(self.pn_linear_models['death_from_'
                                                                                     'hypertensive_disorder_pn'],
                                                  df.loc[df['is_alive'] & df['la_is_postpartum'] &
                                                         (df['pn_postnatal_period_in_weeks'] == week) &
                                                         (df['pn_htn_disorders'].str.contains('severe_gest_htn|'
                                                                                              'severe_pre_eclamp'))])

        # Those women who die have InstantaneousDeath scheduled
        for person in at_risk_of_death_htn.loc[at_risk_of_death_htn].index:
            logger.info(key='direct_maternal_death', data={'person': person, 'preg_state': 'postnatal',
                                                            'year': self.sim.date.year})

            self.sim.modules['Demography'].do_death(individual_id=person, cause='maternal',
                                                    originating_module=self.sim.modules['PostnatalSupervisor'])

            del self.sim.modules['PregnancySupervisor'].mother_and_newborn_info[person]

        # ----------------------------------------- CARE SEEKING ------------------------------------------------------
        # We now use the the pn_emergency_event_mother property that has just been set for women who are experiencing
        # severe complications to select a subset of women who may choose to seek care
        care_seeking = self.apply_linear_model(
            self.pn_linear_models['care_seeking_postnatal_complication_mother'],
            df.loc[df['is_alive'] & df['la_is_postpartum'] & (df['pn_postnatal_period_in_weeks'] == week) &
                   df['pn_emergency_event_mother'] & ~df['hs_is_inpatient']])

        # Reset this property to stop repeat care seeking
        df.loc[care_seeking.index, 'pn_emergency_event_mother'] = False

        # TODO: should these women go through a&e?

        # Schedule the HSI event
        for person in care_seeking.loc[care_seeking].index:
            from tlo.methods.labour import HSI_Labour_ReceivesPostnatalCheck
            admission_event = HSI_Labour_ReceivesPostnatalCheck (self, person_id=person)
            self.sim.modules['HealthSystem'].schedule_hsi_event(admission_event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        # For women who do not seek care we immediately apply risk of death due to complications
        for person in care_seeking.loc[~care_seeking].index:
            self.apply_risk_of_maternal_or_neonatal_death_postnatal(mother_or_child='mother', individual_id=person)

        if week == 6:
            # We call a function in the PregnancySupervisor module to reset the variables from pregnancy which are not
            # longer needed
            week_6_women = df.is_alive & df.la_is_postpartum & (df.pn_postnatal_period_in_weeks == 6)

            self.sim.modules['PregnancySupervisor'].pregnancy_supervisor_property_reset(
                ind_or_df='individual', id_or_index=week_6_women.loc[week_6_women].index)

    def apply_risk_of_neonatal_complications_in_week_one(self, child_id, mother_id):
        """
        This function is called by PostnatalWeekOneEvent for newborns to determine risk of complications. Currently
        this is limited to early onset neonatal sepsis
        :param child_id: childs individual id
        :return:
        """
        df = self.sim.population.props
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        mni_df = pd.DataFrame.from_dict(mni, orient='index')

        logger.debug(key='message', data=f'Newborn {child_id} has arrived at PostnatalWeekOneEvent')

        # Set external variables used in the linear model equation
        maternal_prom = pd.Series(df.at[mother_id, 'ps_premature_rupture_of_membranes'], index=df.loc[[child_id]].index)

        if 'abx_for_prom_given' in mni_df.keys():
            received_abx_for_prom = pd.Series(mni_df.at[mother_id, 'abx_for_prom_given'],
                                              index=df.loc[[child_id]].index)

        # todo: fix
        #if 'chorio_lab' in mni_df.keys():
        #    chorio = (df['ps_chorioamnionitis']) | (df['la_sepsis'] & mni_dict['chorio_lab'])
        chorio = pd.Series(False, index=df.loc[[child_id]].index)

        # We then apply a risk that this womans newborn will develop sepsis during week one
        risk_eons = self.pn_linear_models['early_onset_neonatal_sepsis_week_1'].predict(
            df.loc[[child_id]], received_abx_for_prom=received_abx_for_prom,
            maternal_chorioamnionitis=chorio,
            maternal_prom=maternal_prom)[child_id]

        if self.rng.random_sample() < risk_eons:
            df.at[child_id, 'pn_sepsis_early_neonatal'] = True
            self.sim.modules['NewbornOutcomes'].newborn_care_info[child_id]['sepsis_postnatal'] = True

            logger.info(key='newborn_complication', data={'newborn': child_id,
                                                          'type': 'early_onset_sepsis'})

    def set_postnatal_complications_neonates(self, upper_and_lower_day_limits):
        """
        This function is called by the PostnatalSupervisor event. It applies risk of key complication to neonates
        during each week of the neonatal period after week one (weeks 2, 3 & 4). This is currently limited to sepsis but
         may be expanded at a later date
        :param upper_and_lower_day_limits: 2 value list of the first and last day of each week of the neonatal period
        """
        params = self.current_parameters
        df = self.sim.population.props

        # Here we apply risk of late onset neonatal sepsis (sepsis onsetting after day 7) to newborns
        # TODO: only neonates born in this sim?
        onset_sepsis = self.apply_linear_model(
            self.pn_linear_models['late_onset_neonatal_sepsis'],
            df.loc[df['is_alive'] & (df['age_days'] > upper_and_lower_day_limits[0]) &
                   (df['age_days'] < upper_and_lower_day_limits[1]) & (df['date_of_birth'] > self.sim.start_date) &
                   ~df['hs_is_inpatient']])

        df.loc[onset_sepsis.loc[onset_sepsis].index, 'pn_sepsis_late_neonatal'] = True
        for person in onset_sepsis.loc[onset_sepsis].index:
            self.sim.modules['NewbornOutcomes'].newborn_care_info[person]['sepsis_postnatal'] = True

            logger.info(key='newborn_complication', data={'newborn': person,
                                                          'type': 'late_onset_sepsis'})

        # Then we determine if care will be sought for newly septic newborns
        care_seeking = self.apply_linear_model(
            self.pn_linear_models['care_seeking_postnatal_complication_neonate'],
            df.loc[onset_sepsis.loc[onset_sepsis].index])

        # We schedule the HSI according
        for person in care_seeking.loc[care_seeking].index:
            admission_event = HSI_NewbornOutcomes_ReceivesPostnatalCheck(
                self, person_id=person)
            self.sim.modules['HealthSystem'].schedule_hsi_event(admission_event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        # And apply risk of death for newborns for which care is not sought
        for person in care_seeking.loc[~care_seeking].index:
            self.apply_risk_of_maternal_or_neonatal_death_postnatal(mother_or_child='child', individual_id=person)

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

        # Select the individuals row in the data frame to prevent repeated at based indexing
        if mother_or_child == 'mother':
            mother = df.loc[individual_id]
        if mother_or_child == 'child':
            child = df.loc[individual_id]

        # ================================== MATERNAL DEATH EQUATIONS ==============================================

        if mother_or_child == 'mother':
            causes = list()
            if mother.pn_postpartum_haem_secondary:
                causes.append('secondary_postpartum_haemorrhage')
            if mother.pn_sepsis_late_postpartum:
                causes.append('postpartum_sepsis')
            if mother.pn_htn_disorders == 'eclampsia':
                causes.append('eclampsia')

            risks = dict()
            for cause in causes:
                risk = {f'{cause}': self.pn_linear_models[f'{cause}_death'].predict(
                    df.loc[[individual_id]])[individual_id]}
                risks.update(risk)

            result = 1
            for cause in risks:
                result *= (1 - risks[cause])

            # If random draw is less that the total risk of death, she will die and the primary cause is then
            # determined
            if self.rng.random_sample() < (1 - result):
                probs = list()

                # Cycle over each cause in the dictionary and divide by the sum of the probabilities
                for cause in risks:
                    risks[cause] = risks[cause] / sum(risks.values())
                    probs.append(risks[cause])

                cause_of_death = self.rng.choice(causes, p=probs)

                logger.info(key='direct_maternal_death', data={'person': individual_id, 'preg_state': 'postnatal',
                                                               'year': self.sim.date.year})

                self.sim.modules['Demography'].do_death(individual_id=individual_id, cause=f'{cause_of_death}',
                                                        originating_module=self.sim.modules['PostnatalSupervisor'])

                del self.sim.modules['PregnancySupervisor'].mother_and_newborn_info[individual_id]

            else:
                if mother.pn_postpartum_haem_secondary:
                    df.at[individual_id, 'pn_postpartum_haem_secondary'] = False
                if mother.pn_sepsis_late_postpartum:
                    df.at[individual_id, 'pn_sepsis_late_postpartum'] = False
                if mother.pn_htn_disorders == 'eclampsia':
                    df.at[individual_id, 'pn_htn_disorders'] = 'severe_pre_eclamp'

        # ================================== NEONATAL DEATH EQUATIONS ==============================================
        if mother_or_child == 'child':

            # Neonates can have either early or late onset sepsis, not both at once- so we use either equation
            # depending on this neonates current condition
            if child.pn_sepsis_early_neonatal:
                risk_of_death = self.pn_linear_models['early_onset_neonatal_sepsis_week_1_death'].predict(
                    df.loc[[individual_id]])[individual_id]
            elif child.pn_sepsis_late_neonatal:
                risk_of_death = self.pn_linear_models['late_neonatal_sepsis_death'].predict(df.loc[[
                    individual_id]])[individual_id]

            if child.pn_sepsis_late_neonatal or child.pn_sepsis_early_neonatal:

                # If this neonate will die then we make the appropriate changes
                if self.rng.random_sample() < risk_of_death:
                    logger.debug(key='message', data=f'person {individual_id} has died due to late neonatal sepsis on '
                                                     f'date {self.sim.date}')

                    # todo: add cause of death here!
                    self.sim.modules['Demography'].do_death(individual_id=individual_id, cause='neonatal',
                                                            originating_module=self.sim.modules['PostnatalSupervisor'])

                # Otherwise we reset the variables in the data frame
                else:
                    df.at[individual_id, 'pn_sepsis_late_neonatal'] = False
                    df.at[individual_id, 'pn_sepsis_early_neonatal'] = False


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

        # Set mni[person]['delete_mni'] to True meaning after the next DALY event each womans MNI dict is delted
        for person in week_8_postnatal_women.loc[week_8_postnatal_women].index:
            mni[person]['delete_mni'] = True
            logger.info(key='pnc_mother', data={'total_visits': df.at[person, 'pn_pnc_visits_maternal']})

        df.loc[week_8_postnatal_women, 'pn_postnatal_period_in_weeks'] = 0
        df.loc[week_8_postnatal_women, 'pn_pnc_visits_maternal'] = 0
        df.loc[week_8_postnatal_women, 'la_is_postpartum'] = False
        df.loc[week_8_postnatal_women, 'pn_sepsis_late_postpartum'] = False
        df.loc[week_8_postnatal_women, 'pn_postpartum_haem_secondary'] = False
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

        # For the neonates we now determine if they will develop any long term neurodevelopmental impairment following
        # survival of the neonatal period
        week_6_postnatal_neonates = df.is_alive & (df['age_days'] > 42) & (df['age_days'] < 49) & \
                                    (df.date_of_birth > self.sim.start_date)
        for person in week_6_postnatal_neonates.loc[week_6_postnatal_neonates].index:
            self.sim.modules['NewbornOutcomes'].set_disability_status(person)
            logger.info(key='pnc_child', data={'total_visits': df.at[person, 'pn_pnc_visits_neonatal']})

        # And then reset any key variables
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
        params = self.module.current_parameters
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        nci = self.sim.modules['NewbornOutcomes'].newborn_care_info
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
        #  ------------------------------------- SEPSIS ----------------------------------------------
        # We determine if the mother will develop postnatal sepsis
        if mother.is_alive:
            logger.debug(key='message', data=f'Mother {individual_id} has arrived at PostnatalWeekOneEvent')
            mni[individual_id]['passed_through_week_one'] = True

            # define external variable for linear model
            mode_of_delivery = pd.Series(mni[individual_id]['mode_of_delivery'], index=df.loc[[individual_id]].index)

            risk_sepsis_endometritis = self.module.pn_linear_models['sepsis_endometritis_late_postpartum'].predict(
                df.loc[[individual_id]], mode_of_delivery=mode_of_delivery)[individual_id]
            risk_sepsis_skin_soft_tissue = self.module.pn_linear_models['sepsis_sst_late_postpartum'].predict(
                df.loc[[individual_id]], mode_of_delivery=mode_of_delivery)[individual_id]
            risk_sepsis_urinary_tract = params['prob_late_sepsis_urinary_tract']

            endo_result = risk_sepsis_endometritis > self.module.rng.random_sample()
            if endo_result:
                mni[individual_id]['endo_pp'] = True

            if endo_result or (risk_sepsis_urinary_tract > self.module.rng.random_sample()) or \
               (risk_sepsis_skin_soft_tissue > self.module.rng.random_sample()):

                df.at[individual_id, 'pn_sepsis_late_postpartum'] = True
                store_dalys_in_mni(individual_id, 'sepsis_onset')

                logger.info(key='maternal_complication', data={'mother': individual_id,
                                                               'type': 'sepsis',
                                                               'timing': 'postnatal'})

        #  ---------------------------------------- SECONDARY PPH ------------------------------------------------
            # Next we apply risk of secondary postpartum bleeding
            endometritis = pd.Series(mni[individual_id]['endo_pp'], index=df.loc[[individual_id]].index)

            risk_secondary_pph = self.module.pn_linear_models['secondary_postpartum_haem'].predict(df.loc[[
                    individual_id]], endometritis=endometritis)[individual_id]

            if risk_secondary_pph > self.module.rng.random_sample():
                logger.debug(key='message',
                             data=f'mother {individual_id} has developed a secondary postpartum haemorrhage during'
                                  f' week one of the postnatal period')

                df.at[individual_id, 'pn_postpartum_haem_secondary'] = True
                store_dalys_in_mni(individual_id, 'secondary_pph_onset')

                logger.info(key='maternal_complication', data={'mother': individual_id,
                                                               'type': 'secondary_postpartum_haemorrhage',
                                                               'timing': 'postnatal'})

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
                recent_bleeding = False # TODO: FIX!
                maternal_malaria = False
                hiv_no_art = False

                risk_anaemia_after_pregnancy = self.module.pn_linear_models['anaemia_after_pregnancy'].predict(
                    df.loc[[individual_id]], hiv_no_art=hiv_no_art, recent_bleeding=recent_bleeding,
                    maternal_malaria=maternal_malaria)[individual_id]

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

                    # We modify the probability of progressing from mild to severe gestational hypertension for women
                    # who are on anti hypertensives
                    if df.at[individual_id, 'pn_gest_htn_on_treatment']:
                        treatment_reduced_risk = prob_matrix['gest_htn'][1] * \
                                                 params['treatment_effect_anti_htns_progression_pn']
                        prob_matrix['gest_htn'][1] = treatment_reduced_risk
                        prob_matrix['gest_htn'][0] = 1 - (treatment_reduced_risk + prob_matrix['gest_htn'][2])

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
                risk_pe_after_pregnancy = self.module.pn_linear_models['pre_eclampsia_pn'].predict(df.loc[[
                    individual_id]])[individual_id]

                if risk_pe_after_pregnancy > self.module.rng.random_sample():
                    df.at[individual_id, 'pn_htn_disorders'] = 'mild_pre_eclamp'

                    logger.debug(key='message',
                                 data=f'mother {individual_id} has developed mild pre-eclampsia in week one of the '
                                 f'postnatal period ')
                    df.at[individual_id, 'ps_prev_pre_eclamp'] = True
                else:
                    risk_gh_after_pregnancy = self.module.pn_linear_models['gest_htn_pn'].predict(df.loc[[
                        individual_id]])[individual_id]
                    if risk_gh_after_pregnancy > self.module.rng.random_sample():
                        logger.debug(key='message',
                                     data=f'mother {individual_id} has developed gestational hypertension in week one '
                                          f'of the postnatal period ')
                        df.at[individual_id, 'pn_htn_disorders'] = 'gest_htn'

        # ===============================  NEONATAL COMPLICATIONS IN WEEK ONE  =======================================
        # Newborns may develop early onset sepsis in the first week of life, we apply that risk here

        if child.is_alive:
            nci[individual_id]['passed_through_week_one'] = True
            self.module.apply_risk_of_neonatal_complications_in_week_one(child_id=child_id, mother_id=individual_id)

        # If this event is looking at a twin pair we apply risk to the second twin here
        if child.nb_is_twin:
            if child_two.is_alive:
                nci[individual_id]['passed_through_week_one'] = True
                self.module.apply_risk_of_neonatal_complications_in_week_one(child_id=child_two_id,
                                                                             mother_id=individual_id)

        # ===============================  POSTNATAL CHECK  =========================================================
        # Likelihood of undergoing a postnatal check up is determined following birth in the labour module for all
        # women. At this stage some women will have received a check, some will be scheduled for a check and others are
        # not scheduled for check up
        from tlo.methods.labour import HSI_Labour_ReceivesPostnatalCheck
        from tlo.methods.newborn_outcomes import HSI_NewbornOutcomes_ReceivesPostnatalCheck

        # todo: set facility levels
        # todo: what if they've already had their check up? assume they would go back...

        pnc_one_maternal = HSI_Labour_ReceivesPostnatalCheck(
            self.module, person_id=individual_id, facility_level_of_this_hsi=1)

        # If a mother has developed complications in the first week after birth and has been predicted to attend PNC
        # anyway she will attend now. If she was not predicted to attend but now develops complications she may choose
        # to seek care
        if mother.is_alive:
            if df.at[individual_id, 'pn_sepsis_late_postpartum'] or \
                df.at[individual_id, 'pn_postpartum_haem_secondary'] or \
                (df.at[individual_id, 'pn_htn_disorders'] == 'severe_pre_eclamp') or \
               (df.at[individual_id, 'pn_htn_disorders'] == 'eclampsia'):

                if (mni[individual_id]['future_pnc_check']) or (self.module.rng.random_sample() <
                                                                params['prob_care_seeking_postnatal_emergency']):

                    self.sim.modules['HealthSystem'].schedule_hsi_event\
                        (pnc_one_maternal, priority=0,
                         topen=self.sim.date, tclose=None)

            else:
                # Women without complications in week one are scheduled to attend PNC in the future
                if mni[individual_id]['future_pnc_check']:
                    self.sim.modules['HealthSystem'].schedule_hsi_event \
                        (pnc_one_maternal, priority=0,
                         topen=self.sim.date + pd.DateOffset(self.module.rng.rand_int(0, 41)), tclose=None)

        # This process is then repeated for any children the mother may have
        if child.is_alive:
            if df.at[child_id, 'pn_sepsis_early_neonatal']:
                if (nci[child_id]['future_pnc_check']) or (self.module.rng.random_sample() <
                                                           params['prob_care_seeking_postnatal_emergency_neonate']):

                    pnc_one_neonatal = HSI_NewbornOutcomes_ReceivesPostnatalCheck(
                        self.module, person_id=child_id, facility_level_of_this_hsi=1)

                    self.sim.modules['HealthSystem'].schedule_hsi_event\
                        (pnc_one_neonatal, priority=0,
                         topen=self.sim.date, tclose=None)

        if df.at[child_id, 'nb_is_twin']:
            if df.at[child_two_id, 'is_alive'] and df.at[child_two_id, 'pn_sepsis_early_neonatal']:
                if (nci[child_id]['future_pnc_check']) or (self.module.rng.random_sample() <
                                                            params['prob_care_seeking_postnatal_emergency_neonate']):
                    pnc_one_neonatal = HSI_NewbornOutcomes_ReceivesPostnatalCheck(
                            self.module, person_id=child_id, facility_level_of_this_hsi=1)

                    self.sim.modules['HealthSystem'].schedule_hsi_event \
                            (pnc_one_neonatal, priority=0,
                             topen=self.sim.date, tclose=None)


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
        self.ACCEPTED_FACILITY_LEVEL = 1
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
