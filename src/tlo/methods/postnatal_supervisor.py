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
    """ This module is responsible for the key conditions/complications experienced by a mother and by a neonate
    following labour and the immediate postpartum period (this period, from birth to day +1, is covered by the labour
    module).

    For mothers: This module applies risk of complications across the postnatal period, which is  defined as birth
    until day 42 post-delivery. PostnatalWeekOne Event represents the first week post birth where risk of complications
    remains high. The primary mortality causing complications here are infection/sepsis and secondary postpartum
    haemorrhage. Women with or without complications may/may not seek Postnatal Care at day 7 post birth (we assume
    women who seek care will also bring their newborns with them to the HSI where they will also be assessed). This HSI
    assesses mothers and newborns and schedules admissions if complications are found. The PostnatalSupervisor Event
    applies risk of complications weekly from week 2-6 (ending on day 42). This event also determines additional
    care seeking for mothers who are unwell during this time period. All maternal variables are reset on day 42.

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

    METADATA = {Metadata.DISEASE_MODULE,
                Metadata.USES_HEALTHSYSTEM,
                Metadata.USES_HEALTHBURDEN}  # declare that this is a disease module (leave as empty set otherwise)

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
        'prevalence_type_of_fistula': Parameter(
            Types.LIST, 'prevalence of 1.) vesicovaginal 2.)rectovaginal fistula '),
        'prob_iron_def_per_week_pn': Parameter(
            Types.REAL, 'weekly probability of a women developing iron deficiency following pregnancy'),
        'rr_iron_def_ifa_pn': Parameter(
            Types.REAL, 'effect of iron and folic acid treatment on risk of iron deficiency'),
        'prob_folate_def_per_week_pn': Parameter(
            Types.REAL, 'weekly probability of a women developing folate deficiency following pregnancy '),
        'rr_folate_def_ifa_pn': Parameter(
            Types.REAL, 'effect of iron and folic acid treatment on risk of folate deficiency '),
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
        'treatment_effect_early_init_bf': Parameter(
            Types.REAL, 'effect of early initiation of breastfeeding on neonatal sepsis rates '),
        'treatment_effect_inj_abx_sep': Parameter(
            Types.REAL, 'effect of injectable antibiotics on neonatal sepsis mortality'),
        'treatment_effect_supp_care_sep': Parameter(
            Types.REAL, 'effect of full supportive care on neonatal sepsis mortality'),
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
            Types.REAL, 'weekly probability of a woman developing gestational hypertension during the postnatal period'),
        'weekly_prob_pre_eclampsia_pn': Parameter(
            Types.REAL, 'weekly probability of a woman developing mild pre-eclampsia during the postnatal period'),
        'prob_attend_pnc2': Parameter(
            Types.REAL, 'Probability that a woman receiving PNC1 care will return for PNC2 care'),
        'prob_attend_pnc3': Parameter(
            Types.REAL, 'Probability that a woman receiving PNC2 care will return for PNC3 care'),
        'treatment_effect_parenteral_antibiotics': Parameter(
            Types.REAL, 'Treatment effect of parenteral antibiotics on maternal sepsis mortality '),
        'treatment_effect_bemonc_care_pph': Parameter(
            Types.REAL, 'Treatment effect of BEmONC care on postpartum haemorrhage mortality'),
        'neonatal_sepsis_treatment_effect': Parameter(
            Types.REAL, 'Treatment effect for neonatal sepsis'),
        'weekly_prob_postnatal_death': Parameter(
            Types.REAL, 'Weekly risk of postnatal death'),
        'severity_late_infection_pn': Parameter(
            Types.LIST, 'probability of mild infection, sepsis or severe sepsis in the later postnatal period'),
        'prob_care_seeking_postnatal_emergency': Parameter(
            Types.REAL, 'baseline probability '),
        'prob_care_seeking_postnatal_emergency_neonate': Parameter(
            Types.REAL, 'baseline probability care will be sought for a neonate with a complication'),
        'prob_pnc1_at_day_7': Parameter(
            Types.REAL, 'baseline probability a woman will seek PNC for her and her newborn at day + 7 '),
        'multiplier_for_care_seeking_with_comps': Parameter(
            Types.REAL, 'number by which prob_pnc1_at_day_7 is multiplied by to increase care seeking for PNC1 in women'
                        ' with complications '),
    }

    PROPERTIES = {
        'pn_id_most_recent_child': Property(Types.INT, 'person_id of a mothers most recent child'),
        'pn_postnatal_period_in_weeks': Property(Types.INT, 'The number of weeks a woman is in the postnatal period '
                                                            '(1-6)'),
        'pn_pnc_visits_maternal': Property(Types.INT, 'The number of postnatal care visits a woman has undergone '
                                                      'following her most recent delivery'),
        'pn_pnc_visits_neonatal': Property(Types.INT, 'The number of postnatal care visits a neonate has undergone '
                                                      'following delivery'),
        'pn_htn_disorders': Property(Types.CATEGORICAL, 'Hypertensive disorders of the postnatal period',
                                         categories=['none', 'gest_htn', 'severe_gest_htn', 'mild_pre_eclamp',
                                                 'severe_pre_eclamp', 'eclampsia']),
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

        params = self.parameters
        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_PostnatalSupervisor.xlsx',
                            sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)

        if 'HealthBurden' in self.sim.modules.keys():
            params['pn_daly_weights'] = {
                'haemorrhage_moderate': self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=339),
                'haemorrhage_severe': self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=338),
                'maternal_sepsis': self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=340),
                'vv_fistula': self.sim.modules['HealthBurden'].get_daly_weight(349),
                'rv_fistula': self.sim.modules['HealthBurden'].get_daly_weight(350),
                'mild_anaemia': self.sim.modules['HealthBurden'].get_daly_weight(335),
                'moderate_anaemia': self.sim.modules['HealthBurden'].get_daly_weight(336),
                'severe_anaemia': self.sim.modules['HealthBurden'].get_daly_weight(337),
                'mild_motor_sepsis_neonate': self.sim.modules['HealthBurden'].get_daly_weight(431),
                'moderate_motor_sepsis_neonate': self.sim.modules['HealthBurden'].get_daly_weight(438),
                'severe_motor_sepsis_neonate': self.sim.modules['HealthBurden'].get_daly_weight(435),
                'mild_motor_cognitive_sepsis_neonate': self.sim.modules['HealthBurden'].get_daly_weight(441)}

        # ======================================= LINEAR MODEL EQUATIONS =============================================
        # All linear equations used in this module are stored within the pn_linear_equations
        # parameter below

        # TODO: process of 'selection' of important predictors in linear equations is ongoing, a linear model that
        #  is empty of predictors at the end of this process will be converted to a set probability

        params['pn_linear_equations'] = {
            # This equation is used to determine a mothers risk of developing obstetric fistula after birth
            'obstetric_fistula': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_obstetric_fistula']),

            # This equation is used to determine a mothers risk of secondary postpartum haemorrhage
            'secondary_postpartum_haem': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_secondary_pph']),

            # This equation is used to determine a mothers risk of dying following a secondary postpartum haemorrhage
            'secondary_postpartum_haem_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_secondary_pph'],
                Predictor('pn_postpartum_haem_secondary_treatment').when(True, params['treatment_effect_bemonc_care_'
                                                                                      'pph'])),

            # This equation is used to determine a mothers risk of developing endometritis infection
            'endometritis': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_endometritis_pn']),

            # This equation is used to determine a mothers risk of developing a urinary tract infection
            'urinary_tract_inf': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_urinary_tract_inf_pn']),

            # This equation is used to determine a mothers risk of developing a skin or soft tissue infection
            'skin_soft_tissue_inf': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_skin_soft_tissue_inf_pn']),

            # This equation is used to determine a mothers risk of developing another infection, not defined above
            'other_maternal_infection': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_other_inf_pn']),

            # This equation is used to determine a mothers risk of developing sepsis following one of more of the above
            # infections
            'sepsis_late_postpartum': LinearModel(
                LinearModelType.ADDITIVE,
                0,
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

            # This equation is used to determine a mothers risk of dying following a secondary postpartum haemorrhage
            'postnatal_sepsis_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_postnatal_sepsis'],
                Predictor('pn_sepsis_late_postpartum_treatment').when(True, params['treatment_effect_parenteral_'
                                                                                   'antibiotics'])),

            'gest_htn_pn': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['weekly_prob_gest_htn_pn']),
            'pre_eclampsia_pn': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['weekly_prob_pre_eclampsia_pn']),


            # This equation is used to determine a mothers risk of developing anaemia postnatal
            'anaemia_after_pregnancy': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['baseline_prob_anaemia_per_week'],
                Predictor('pn_deficiencies_following_pregnancy').apply(
                    lambda x: params['rr_anaemia_if_iron_deficient_pn']
                    if x & self.deficiencies_following_pregnancy.element_repr('iron') else 1),
                Predictor('pn_deficiencies_following_pregnancy').apply(
                    lambda x: params['rr_anaemia_if_folate_deficient_pn']
                    if x & self.deficiencies_following_pregnancy.element_repr('folate') else 1),
                Predictor('pn_deficiencies_following_pregnancy').apply(
                    lambda x: params['rr_anaemia_if_b12_deficient_pn']
                    if x & self.deficiencies_following_pregnancy.element_repr('b12') else 1)),

            # This equation is used to determine a neonates risk of developing early onset neonatal sepsis
            # (sepsis onsetting prior to day 7) in the first week of life
            'early_onset_neonatal_sepsis_week_1': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_early_onset_neonatal_sepsis_week_1'],
                #  TODO: will these need to be properties of the newborn to apply here too
                #    Predictor('clean_birth', external=True).when('True', params['treatment_effect_clean_birth']),
                #    Predictor('cord_care_given', external=True).when('True', params['treatment_effect_cord_care']),

                Predictor('nb_early_init_breastfeeding').when(True, params['treatment_effect_early_init_bf'])),


            # This equation is used to determine a neonates risk of dying following early onset sepsis in week one
            'early_onset_neonatal_sepsis_week_1_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_early_onset_neonatal_sepsis'],
                Predictor('pn_sepsis_neonatal_inj_abx').when(True, params['treatment_effect_inj_abx_sep']),
                Predictor('pn_sepsis_neonatal_full_supp_care').when(True, params['treatment_effect_supp_care_'
                                                                                      'sep'])),

            # This equation is used to determine a neonates risk of developing late onset neonatal sepsis
            # (sepsis onsetting between 7 and day 28) after  the first week of life
            'late_onset_neonatal_sepsis': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_late_onset_neonatal_sepsis'],
                Predictor('nb_early_init_breastfeeding').when(True, params['treatment_effect_early_init_bf'])),

            # This equation is used to determine a neonates risk of dying following late onset neonatal sepsis
            # (sepsis onsetting between 7 and day 28) after the first week of life
            'late_neonatal_sepsis_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_late_neonatal_sepsis'],
                Predictor('pn_sepsis_neonatal_inj_abx').when(True, params['treatment_effect_inj_abx_sep']),
                Predictor('pn_sepsis_neonatal_full_supp_care').when(True, params['treatment_effect_supp_care_sep'])),

            # This equation is used to determine if a mother will seek care for treatment in the instance of an
            # emergency complication postnatally (sepsis or haemorrhage)
            'care_seeking_postnatal_complication_mother': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_care_seeking_postnatal_emergency']),

            # This equation is used to determine if a mother will seek care for treatment for her newborn in the
            # instance of them developing an emergency complication postnatally (sepsis)
            'care_seeking_postnatal_complication_neonate': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_care_seeking_postnatal_emergency_neonate']),
        }

    def initialise_population(self, population):

        df = population.props

        df.loc[df.is_alive, 'pn_id_most_recent_child'] = -1
        df.loc[df.is_alive, 'pn_postnatal_period_in_weeks'] = 0
        df.loc[df.is_alive, 'pn_pnc_visits_maternal'] = 0
        df.loc[df.is_alive, 'pn_pnc_visits_neonatal'] = 0
        df.loc[df.is_alive, 'pn_htn_disorders'] = 'none'
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
        self.deficiencies_following_pregnancy = BitsetHandler(self.sim.population, 'pn_deficiencies_following_'
                                                                                   'pregnancy',
                                                              ['iron', 'folate', 'b12'])

    def initialise_simulation(self, sim):

        # Schedule the first instance of the PostnatalSupervisorEvent
        sim.schedule_event(PostnatalSupervisorEvent(self),
                           sim.date + DateOffset(days=0))

        # Register logging event
        sim.schedule_event(PostnatalLoggingEvent(self),
                           sim.date + DateOffset(years=1))

        # Define the events we want to track in the postnatal_tracker...
        self.postnatal_tracker = {'endometritis': 0,
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

        # Register dx_tests used as assessment for postnatal conditions during PNC visits

        # As with the labour module these dx_tests represent a probability that one of the following clinical outcomes
        # will be detected by the health care worker and treatment will be initiated

        # TODO: vary by facility 'type'
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            assessment_for_postnatal_sepsis=DxTest(
                property='pn_sepsis_late_postpartum',
                sensitivity=0.99),

            assessment_for_secondary_pph=DxTest(
                property='pn_postpartum_haem_secondary',
                sensitivity=0.99),

            assessment_for_hypertension=DxTest(
                property='pn_htn_disorders', target_categories=['gest_htn', 'mild_pre_eclamp', 'severe_pre_eclamp',
                                                                'eclampsia'],
                sensitivity=0.99),

            assessment_for_late_onset_neonatal_sepsis=DxTest(
                property='pn_sepsis_late_neonatal',
                sensitivity=0.99),

            assessment_for_early_onset_neonatal_sepsis=DxTest(
                property='pn_sepsis_early_neonatal',
                sensitivity=0.99),

            assessment_for_anaemia=DxTest(
                property='pn_anaemia_following_pregnancy', target_categories=['mild', 'moderate', 'severe'],
                sensitivity=0.99))

    def on_birth(self, mother_id, child_id):
        df = self.sim.population.props
        params = self.parameters

        df.at[child_id, 'pn_id_most_recent_child'] = -1
        df.at[child_id, 'pn_postnatal_period_in_weeks'] = 0
        df.at[child_id, 'pn_pnc_visits_maternal'] = 0
        df.at[child_id, 'pn_pnc_visits_neonatal'] = 0
        df.at[child_id, 'pn_htn_disorders'] = 'none'
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

        # We store the ID number of the child this woman has most recently delivered as a property of the woman. This is
        # because PNC is scheduled for the woman during the Labour Module but must act on both mother and child
        df.at[mother_id, 'pn_id_most_recent_child'] = child_id

        # Here we determine if, following childbirth, this woman will develop a fistula
        risk_of_fistula = params['pn_linear_equations'][
            'obstetric_fistula'].predict(df.loc[[mother_id]])[mother_id]

        if self.rng.random_sample() < risk_of_fistula:
            # We determine the specific type of fistula this woman is experiencing, to match with DALY weights
            fistula_type = self.rng.choice(['vesicovaginal', 'rectovaginal'], p=params['prevalence_type_of_fistula'])
            df.at[mother_id, 'pn_obstetric_fistula'] = fistula_type
            self.postnatal_tracker['fistula'] += 1

        # ======================= CONTINUATION OF COMPLICATIONS INTO THE POSTNATAL PERIOD =========================
        # Certain conditions experienced in pregnancy are liable to continue into the postnatal period

        # HYPERTENSIVE DISORDERS...
        # The majority of hypertension related to pregnancy resolved with delivery of the foetus. However the condition
        # may persist (and even onset within the postnatal period...)
        if df.at[mother_id, 'ps_htn_disorders'] == 'gest_htn' or 'severe_gest_htn' or 'mild_pre_eclamp' or \
                                                   'severe_pre_eclamp':
            if self.rng.random_sample() < params['prob_htn_persists']:
                logger.debug(key='message', data=f'mother {mother_id} will remain hypertensive despite successfully '
                                                 f'delivering')
                df.at[mother_id, 'pn_htn_disorders'] = df.at[mother_id, 'ps_htn_disorders']

        #  DEFICIENCIES/ANAEMIA...
        # We carry across any deficiencies that may increase this womans risk of postnatal anaemia
        if self.sim.modules['PregnancySupervisor'].deficiencies_in_pregnancy.has_any([mother_id], 'iron', first=True):
            self.deficiencies_following_pregnancy.set([mother_id], 'iron')
        if self.sim.modules['PregnancySupervisor'].deficiencies_in_pregnancy.has_any([mother_id], 'folate', first=True):
            self.deficiencies_following_pregnancy.set([mother_id], 'folate')
        if self.sim.modules['PregnancySupervisor'].deficiencies_in_pregnancy.has_any([mother_id], 'b12', first=True):
            self.deficiencies_following_pregnancy.set([mother_id], 'b12')

        # And similarly, if she is already anemic then she remains so in the postnatal period
        if df.at[mother_id, 'ps_anaemia_in_pregnancy'] != 'none':
            df.at[mother_id, 'pn_anaemia_following_pregnancy'] = df.at[mother_id, 'ps_anaemia_in_pregnancy']

        # Finally we call a function in the PregnancySupervisor module to reset the variables from pregnancy
        self.sim.modules['PregnancySupervisor'].property_reset(mother_id)

    def on_hsi_alert(self, person_id, treatment_id):
        logger.debug(key='message', data=f'This is PostnatalSupervisor, being alerted about a health system '
                                         f'interaction person {person_id} for: {treatment_id}')

    def report_daly_values(self):

        # TODO: discuss with TH regarding 'transient' disability
        # TODO: add other complications if they remain in the module

        logger.debug(key='message', data='This is PostnatalSupervisor reporting my health values')
        df = self.sim.population.props
        p = self.parameters['pn_daly_weights']

        health_values_1 = df.loc[df.is_alive, 'pn_obstetric_fistula'].map(
            {'none': 0,
             'vesicovaginal': p['vv_fistula'],
             'rectovaginal': p['rv_fistula']})
        health_values_1.name = 'Fistula'

        health_values_2 = df.loc[df.is_alive, 'pn_postpartum_haem_secondary'].map(
            {False: 0, True: p['haemorrhage_moderate']})
        health_values_2.name = 'Secondary PPH'

        health_values_3 = df.loc[df.is_alive, 'pn_sepsis_late_postpartum'].map(
            {False: 0, True: p['maternal_sepsis']})
        health_values_3.name = 'Postnatal Sepsis'

        health_values_4 = df.loc[df.is_alive, 'pn_anaemia_following_pregnancy'].map(
            {'none': 0,
             'mild': p['mild_anaemia'],
             'moderate': p['moderate_anaemia'],
             'severe': p['severe_anaemia']})
        health_values_4.name = 'Postnatal Anaemia'

        health_values_5 = df.loc[df.is_alive, 'pn_neonatal_sepsis_disab'].map(
            {'none': 0,
             'mild_motor': p['mild_motor_sepsis_neonate'],
             'mild_motor_and_cog': p['mild_motor_cognitive_sepsis_neonate'],
             'moderate_motor': p['moderate_motor_sepsis_neonate'],
             'severe_motor': p['severe_motor_sepsis_neonate']})
        health_values_5.name = 'Neonatal Sepsis Long term Disability'

        health_values_df = pd.concat([health_values_1.loc[df.is_alive], health_values_2.loc[df.is_alive],
                                      health_values_3.loc[df.is_alive], health_values_4.loc[df.is_alive],
                                      health_values_5.loc[df.is_alive]], axis=1)

        return health_values_df

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
        if risk_infection < self.rng.random_sample():
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

        def onset(eq):
            """
            Runs a specific equation within the linear model for the appropriate subset of women in the postnatal period
             and returns a BOOL series
            :param eq: linear model equation
            :return: BOOL series
            """
            onset_condition = self.apply_linear_model(
                params['pn_linear_equations'][f'{eq}'],
                df.loc[df['is_alive'] & df['la_is_postpartum'] & (df['pn_postnatal_period_in_weeks'] == week)])
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
            self.postpartum_infections_late.set(onset_other_inf.loc[onset_other_inf].index, 'skin_soft_tissue_inf')

        # -------------------------------------- SEPSIS --------------------------------------------------------------
        # Next we run the linear model to see if any of the women who developed infections will lead go on to develop
        # maternal postnatal sepsis
        onset_sepsis = onset('sepsis_late_postpartum')
        df.loc[onset_sepsis.loc[onset_sepsis].index, 'pn_sepsis_late_postpartum'] = True

        # If sepsis develops we use this property to denote that these women are experiencing an emergency and may need
        # to seek care
        df.loc[onset_sepsis.loc[onset_sepsis].index, 'pn_emergency_event_mother'] = True
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
        if not onset_pph.loc[onset_pph].empty:
            logger.debug(key='message', data=f'The following women have developed secondary pph during week {week}'
                                             f' of the postnatal period, {onset_pph.loc[onset_pph].index}')
            self.postnatal_tracker['secondary_pph'] += len(onset_pph.loc[onset_pph])

        # ------------------------------------- IRON DEFICIENCY ------------------------------------------------------
        # Now we will determine if any women will develop deficiencies that can lead to anaemia

        # First we select a subset of the postnatal population who are not iron deficient and are not receiving iron
        # supplements
        iron_def_no_ifa = ~self.deficiencies_following_pregnancy.has_all(
                df.is_alive & df.la_is_postpartum & ~df.la_iron_folic_acid_postnatal &
                (df.pn_postnatal_period_in_weeks == week), 'iron')

        # We determine their risk of iron deficiency
        new_iron_def = pd.Series(self.rng.random_sample(len(iron_def_no_ifa)) < params['prob_iron_def_per_week_pn'],
                                 index=iron_def_no_ifa.index)

        # And change their property accordingly
        self.deficiencies_following_pregnancy.set(new_iron_def.loc[new_iron_def].index, 'iron')

        # Next we select women who aren't iron deficient but are receiving iron supplementation
        iron_def_ifa = ~self.deficiencies_following_pregnancy.has_all(
            df.is_alive & df.la_is_postpartum & df.la_iron_folic_acid_postnatal &
            (df.pn_postnatal_period_in_weeks == week), 'iron')

        # We reduce their individual risk of deficiencies due to treatment and make changes to the data frame
        risk_of_iron_def = params['prob_iron_def_per_week_pn'] * params['rr_iron_def_ifa_pn']

        new_iron_def = pd.Series(self.rng.random_sample(len(iron_def_ifa)) < risk_of_iron_def,
                                 index=iron_def_ifa.index)

        self.deficiencies_following_pregnancy.set(new_iron_def.loc[new_iron_def].index, 'iron')

        # ------------------------------------- FOLATE DEFICIENCY ------------------------------------------------------
        # This process is then repeated for folate and B12...
        folate_def_no_ifa = ~self.deficiencies_following_pregnancy.has_all(
            df.is_alive & df.la_is_postpartum & ~df.la_iron_folic_acid_postnatal &
            (df.pn_postnatal_period_in_weeks == week), 'folate')

        new_folate_def = pd.Series(
                self.rng.random_sample(len(folate_def_no_ifa)) < params['prob_folate_def_per_week_pn'],
                index=folate_def_no_ifa.index)

        self.deficiencies_following_pregnancy.set(new_folate_def.loc[new_folate_def].index, 'folate')

        folate_def_ifa = ~self.deficiencies_following_pregnancy.has_all(
            df.is_alive & df.la_is_postpartum & df.la_iron_folic_acid_postnatal &
            (df.pn_postnatal_period_in_weeks == week), 'folate')

        risk_of_folate_def = params['prob_folate_def_per_week_pn'] * params['rr_folate_def_ifa_pn']

        new_folate_def = pd.Series(self.rng.random_sample(len(folate_def_ifa)) < risk_of_folate_def,
                                       index=folate_def_ifa.index)

        self.deficiencies_following_pregnancy.set(new_folate_def.loc[new_folate_def].index, 'folate')

        # ------------------------------------- B12 DEFICIENCY ------------------------------------------------------
        b12_def = ~self.deficiencies_following_pregnancy.has_all(
            df.is_alive & df.la_is_postpartum & (df.pn_postnatal_period_in_weeks == week), 'b12')

        new_b12_def = pd.Series(self.rng.random_sample(len(b12_def)) < params['prob_b12_def_per_week_pn'],
                                    index=b12_def.index)

        self.deficiencies_following_pregnancy.set(new_b12_def.loc[new_b12_def].index, 'b12')

        # ----------------------------------------- ANAEMIA ----------------------------------------------------------
        # Then we apply a risk of anaemia developing in this week, and determine its severity
        onset_anaemia = onset('anaemia_after_pregnancy')
        random_choice_severity = pd.Series(self.rng.choice(['mild', 'moderate', 'severe'],
                                                           p=params['prob_type_of_anaemia_pn'],
                                                           size=len(onset_anaemia.loc[onset_anaemia])),
                                           index=onset_anaemia.loc[onset_anaemia].index)

        df.loc[onset_anaemia.loc[onset_anaemia].index, 'pn_anaemia_following_pregnancy'] = random_choice_severity
        if not onset_anaemia.loc[onset_anaemia].empty:
                logger.debug(key='message', data=f'The following women have developed anaemia during week {week}'
                                                 f' of the postnatal period, {onset_anaemia.loc[onset_anaemia].index}')
                self.postnatal_tracker['postnatal_anaemia'] += len(onset_anaemia.loc[onset_anaemia])

        # --------------------------------------- HYPERTENSION ------------------------------------------
        # For women who are still experiencing a hypertensive disorder of pregnancy we determine if that will now
        # resolve

        # 1.) RESOLUTION
        women_with_htn = df.loc[
            df['is_alive'] & df['la_is_postpartum'] & (df['pn_postnatal_period_in_weeks'] == week) &
            (df['pn_htn_disorders'] != 'none')]
        resolvers = pd.Series(self.rng.random_sample(len(women_with_htn)) < params['prob_htn_resolves'],
                              index=women_with_htn.index)
        df.loc[resolvers.loc[resolvers].index, 'pn_htn_disorders'] = 'none'

        # 2.) PROGRESSION
        women_with_htn = df.is_alive & df.la_is_postpartum & (df.pn_postnatal_period_in_weeks == week) & \
                         (df.pn_htn_disorders != 'none')

        disease_states = ['gest_htn', 'severe_gest_htn', 'mild_pre_eclamp', 'severe_pre_eclamp', 'eclampsia']
        prob_matrix = pd.DataFrame(columns=disease_states, index=disease_states)

        prob_matrix['gest_htn'] = [0.8, 0.1, 0.1, 0.0, 0.0]
        prob_matrix['severe_gest_htn'] = [0.0, 0.8, 0.0, 0.2, 0.0]
        prob_matrix['mild_pre_eclamp'] = [0.0, 0.0, 0.8, 0.2, 0.0]
        prob_matrix['severe_pre_eclamp'] = [0.0, 0.0, 0.0, 0.6, 0.4]
        prob_matrix['eclampsia'] = [0.0, 0.0, 0.0, 0.0, 1]

        current_status = df.loc[women_with_htn, "pn_htn_disorders"]
        new_status = util.transition_states(current_status, prob_matrix, self.rng)
        df.loc[women_with_htn, "pn_htn_disorders"] = new_status
        assess_status_change_for_severe_pre_eclampsia = (current_status != "severe_pre_eclamp") & \
                                                        (new_status == "severe_pre_eclamp")

        new_onset_severe_pre_eclampsia = assess_status_change_for_severe_pre_eclampsia[
            assess_status_change_for_severe_pre_eclampsia]

        # For these women we schedule them to an onset event where they may seek care
        if not new_onset_severe_pre_eclampsia.empty:
            logger.debug(key='message',
                         data='The following women have developed severe pre-eclampsia following their '
                         f'pregnancy {new_onset_severe_pre_eclampsia.index}')

            for person in new_onset_severe_pre_eclampsia.index:
                df.at[person, 'pn_emergency_event_mother'] = True

        assess_status_change_for_eclampsia = (current_status != "eclampsia") & (new_status == "eclampsia")
        new_onset_eclampsia = assess_status_change_for_eclampsia[assess_status_change_for_eclampsia]

        if not new_onset_eclampsia.empty:
            logger.debug(key='message', data='The following women have developed eclampsia following their '
                                             f'pregnancy {new_onset_eclampsia.index}')
            for person in new_onset_eclampsia.index:
                df.at[person, 'pn_emergency_event_mother'] = True

        # 3.) ONSET
        #  -------------------------------- RISK OF PRE-ECLAMPSIA HYPERTENSION --------------------------------------
        pre_eclampsia = self.apply_linear_model(
            params['pn_linear_equations']['pre_eclampsia_pn'],
            df.loc[df['is_alive'] & df['la_is_postpartum'] & (df['pn_postnatal_period_in_weeks'] == week) &
                   (df['pn_htn_disorders'] == 'none')])

        df.loc[pre_eclampsia.loc[pre_eclampsia].index, 'ps_prev_pre_eclamp'] = True  # todo: conventions
        df.loc[pre_eclampsia.loc[pre_eclampsia].index, 'pn_htn_disorders'] = 'mild_pre_eclamp'

        if not pre_eclampsia.loc[pre_eclampsia].empty:
            logger.debug(key='message', data=f'The following women have developed pre_eclampsia '
                                             f'{pre_eclampsia.loc[pre_eclampsia].index}')

        #  -------------------------------- RISK OF GESTATIONAL HYPERTENSION --------------------------------------
        gest_hypertension = self.apply_linear_model(
            params['pn_linear_equations']['gest_htn_pn'],
            df.loc[df['is_alive'] & df['la_is_postpartum'] & (df['pn_postnatal_period_in_weeks'] == week) &
                   (df['pn_htn_disorders'] == 'none')])

        df.loc[gest_hypertension.loc[gest_hypertension].index, 'pn_htn_disorders'] = 'gest_htn'

        if not gest_hypertension.loc[gest_hypertension].empty:
            logger.debug(key='message', data=f'The following women have developed gestational hypertension'
                                             f'{gest_hypertension.loc[gest_hypertension].index}')

        # ----------------------------------------- CARE SEEKING -----------------------------------------------------
        # We now use the the pn_emergency_event_mother property that has just been set for women who are experiencing
        # sepsis or hemorrhage to
        care_seeking = self.apply_linear_model(
            params['pn_linear_equations']['care_seeking_postnatal_complication_mother'],
            df.loc[df['is_alive'] & df['la_is_postpartum'] & (df['pn_postnatal_period_in_weeks'] == week) &
                   df['pn_emergency_event_mother']])

        # Reset this property to stop repeat care seeking
        df.loc[care_seeking.index, 'pn_emergency_event_mother'] = False

        # Schedule the HSI event
        # TODO: is there a way we can check if and when this woman will go to the next PNC visit
        # TODO: if she will wont be attending- and does choose to seek care she can go to PN ward
        # todo: if she will be attending she doesnt need to seek care?

        for person in care_seeking.loc[care_seeking].index:
            admission_event = HSI_PostnatalSupervisor_PostnatalWardInpatientCare(
                self, person_id=person)
            self.sim.modules['HealthSystem'].schedule_hsi_event(admission_event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        for person in care_seeking.loc[~care_seeking].index:
            self.apply_risk_of_maternal_or_neonatal_death_postnatal(mother_or_child='mother', individual_id=person)

    def set_postnatal_complications_neonates(self, upper_and_lower_day_limits):
        """
        This function is called by the PostnatalSupervisor event. It applies risk of key complication to neonates
        during each week of the neonatal period after week one (weeks 2, 3 & 4). This is currently limited to sepsis but
         may be expanded at a later date
        :param upper_and_lower_day_limits: 2 value list of the first and last day of each week of the neonatal period
        """
        params = self.parameters
        df = self.sim.population.props

        # Here we apply risk of late onset neonatal sepsis (sepsis onsetting after day 7) to newborns
        onset_sepsis = self.apply_linear_model(
            params['pn_linear_equations']['late_onset_neonatal_sepsis'],
            df.loc[df['is_alive'] & (df['age_days'] > upper_and_lower_day_limits[0]) &
                   (df['age_days'] < upper_and_lower_day_limits[1])])

        df.loc[onset_sepsis.loc[onset_sepsis].index, 'pn_sepsis_late_neonatal'] = False
        self.postnatal_tracker['late_neonatal_sepsis'] += 1

        # Then we determine if care will be sought for newly septic newborns
        care_seeking = self.apply_linear_model(
            params['pn_linear_equations']['care_seeking_postnatal_complication_neonate'],
            df.loc[onset_sepsis.loc[onset_sepsis].index])

        for person in care_seeking.loc[care_seeking].index:
            admission_event = HSI_PostnatalSupervisor_NeonatalWardInpatientCare(
                self, person_id=person)
            self.sim.modules['HealthSystem'].schedule_hsi_event(admission_event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        for person in care_seeking.loc[~care_seeking].index:
            self.apply_risk_of_maternal_or_neonatal_death_postnatal(mother_or_child='child', individual_id=person)

    def assessment_for_maternal_complication_during_pnc(self, individual_id, hsi_event):
        """
        This function is called by each of the postnatal care visit HSI and represents assessment of mothers for sepsis,
         postnatal bleeding and hypertension. If these conditions are detected during PNC then women are admitted for
         treatment
        :param individual_id: individual_id (mother)
        :param hsi_event: HSI event in which this function is called
        """

        # We create a variable that will be set to true if a health work detects a complication and chooses to admit
        # (in case of multiple complications requiring admission)
        needs_admission = False

        # SEPSIS
        # Women are assessed for key complications after child birth
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

        # ANAEMIA
        if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run=
                                                                   'assessment_for_anaemia',
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

    def assessment_for_neonatal_complications_during_pnc(self, individual_id, hsi_event):
        """
        This function is called by each of the postnatal care visit HSI and represents assessment of neonates for
        sepsis. If sepsis is detected during PNC then the neonate is  admitted for treatment.
        :param individual_id: individual_id (child)
        :param hsi_event: HSI event in which this function is called
        """

        # SEPSIS
        if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run=
                                                                   'assessment_for_early_onset_neonatal_sepsis',
                                                                   hsi_event=hsi_event):
            logger.debug(key='message', data=f'Neonate {individual_id} has been assessed and diagnosed with early onset '
                                             f'neonatal sepsis, they will be admitted for treatment')

            sepsis_treatment = HSI_PostnatalSupervisor_NeonatalWardInpatientCare(
                self, person_id=individual_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(sepsis_treatment,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run=
                                                                   'assessment_for_late_onset_neonatal_sepsis',
                                                                   hsi_event=hsi_event):
            logger.debug(key='message', data=f'Neonate {individual_id} has been assessed and diagnosed with late onset '
                                             f'neonatal sepsis, they will be admitted for treatment')

            sepsis_treatment = HSI_PostnatalSupervisor_NeonatalWardInpatientCare(
                self, person_id=individual_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(sepsis_treatment,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        # ANAEMIA TREATMENT?

    def apply_risk_of_maternal_or_neonatal_death_postnatal(self, mother_or_child, individual_id):
        """

        :param mother_or_child: Person of interest for the effect of this function - pass 'mother' or 'child' to
         apply risk of death correctly
        :param individual_id: individual_id
        """
        df = self.sim.population.props

        if mother_or_child == 'mother':
            mother = df.loc[individual_id]
        if mother_or_child == 'child':
            child = df.loc[individual_id]
        params = self.parameters

        # ================================== MATERNAL DEATH EQUATIONS ==============================================
        if mother_or_child == 'mother':
            # Create a variable signifying death, in case of multiple complications contributing to death
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
                assert (self.postpartum_infections_late.has_any(
                        [individual_id], 'endometritis', 'urinary_tract_inf', 'skin_soft_tissue_inf',
                        'other_maternal_infection', first=True))

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

            if mother.pn_htn_disorders == 'eclampsia':
                risk_of_death = params['ps_linear_equations']['eclampsia_death_pn'].predict(
                            df.loc[[individual_id]])[individual_id]

                if self.rng.random_sample() < risk_of_death:
                    postnatal_death = True
                else:
                    df.at[individual_id, 'pn_htn_disorders'] = 'severe_pre_eclamp'

            # If she has died due to either (or both) of these causes, we schedule the DeathEvent
            if postnatal_death:
                logger.debug(key='message', data=f'mother {individual_id} has died due to complications of the '
                                                 f'postnatal period')

                self.sim.schedule_event(demography.InstantaneousDeath(self, individual_id,
                                                                      cause='postnatal_week_one'), self.sim.date)
                self.postnatal_tracker['postnatal_death'] += 1

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

                    self.sim.schedule_event(demography.InstantaneousDeath(
                        self, individual_id, cause='neonatal_sepsis'), self.sim.date)

                    self.postnatal_tracker['neonatal_death'] += 1
                    self.postnatal_tracker['neonatal_sepsis_death'] += 1

                # Otherwise we reset the variables in the data frame
                else:
                    df.at[individual_id, 'pn_sepsis_late_neonatal'] = False
                    df.at[individual_id, 'pn_sepsis_early_neonatal'] = False

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

        # TODO: Move to pregnancy supervisor event (PregnancyAndPostnatalSupervisor) as that event already runs weekly
        # TODO: does it matter were apply risk of HTN resolution to women without HTN (opposite for anaemia)
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

        # Women who were rounded down to 0 are classified as week 1 in the postnatal period
        zero_women = df.is_alive & df.la_is_postpartum & (df.pn_postnatal_period_in_weeks == 0.0)
        df.loc[zero_women, 'pn_postnatal_period_in_weeks'] = 1.0

        # Check that all women are week 1 or above
        assert (df.loc[alive_and_recently_delivered, 'pn_postnatal_period_in_weeks'] > 0).all().all()

        # ================================= COMPLICATIONS/CARE SEEKING FOR WOMEN ======================================
        # This function is called to apply risk of complications to women in weeks 2, 3, 4, 5 and 6 of the postnatal
        # period
        for week in [2, 3, 4, 5, 6]:
            self.module.set_postnatal_complications_mothers(week=week)

        # ================================= COMPLICATIONS/CARE SEEKING FOR NEONATES ===================================
        # Next this function is called to apply risk of complictions to neonates in week 2, 3 and 4 of the neonatal
        # period. Upper and lower limit days in the week are used to define one week.
        for upper_and_lower_day_limits in [[7, 15], [14, 22], [21, 29]]:
            self.module.set_postnatal_complications_neonates(upper_and_lower_day_limits=upper_and_lower_day_limits)

        # -------------------------------------- RESETTING VARIABLES --------------------------------------------------
        # Finally when women reach the end of the postnatal period (after 42 days from birth) we reset all the key
        # variables from this module- allowing them to be set again in future pregnancies

        # TODO: this is too close to the last PNC visit on day 42
        # Maternal variables
        week_7_postnatal_women = df.is_alive & df.la_is_postpartum & (df.pn_postnatal_period_in_weeks == 7)
        df.loc[week_7_postnatal_women, 'pn_postnatal_period_in_weeks'] = 0
        df.loc[week_7_postnatal_women, 'pn_pnc_visits_maternal'] = 0
        df.loc[week_7_postnatal_women, 'la_is_postpartum'] = False

        df.loc[week_7_postnatal_women, 'pn_htn_disorders'] = 'none'
        df.loc[week_7_postnatal_women, 'pn_anaemia_following_pregnancy'] = 'none'
        self.module.postpartum_infections_late.unset(week_7_postnatal_women, 'endometritis', 'urinary_tract_inf',
                                                     'skin_soft_tissue_inf', 'other_maternal_infection')
        self.module.deficiencies_following_pregnancy.unset(week_7_postnatal_women, 'iron', 'folate',
                                                     'b12')
        df.loc[week_7_postnatal_women, 'pn_sepsis_late_postpartum'] = False
        df.loc[week_7_postnatal_women, 'pn_postpartum_haem_secondary'] = False
        
        df.loc[week_7_postnatal_women, 'pn_postpartum_haem_secondary_treatment'] = False
        
        # Neonatal variables 
        week_7_postnatal_neonates = df.is_alive & (df['age_days'] > 42) & (df['age_days'] < 49)
        df.loc[week_7_postnatal_neonates, 'pn_pnc_visits_neonatal'] = 0
        df.loc[week_7_postnatal_neonates, 'pn_sepsis_early_neonatal'] = False
        df.loc[week_7_postnatal_neonates, 'pn_sepsis_late_neonatal'] = False
        df.loc[week_7_postnatal_neonates, 'pn_sepsis_neonatal_inj_abx'] = False
        df.loc[week_7_postnatal_neonates, 'pn_sepsis_neonatal_full_supp_care'] = False

        # todo: unset deficiencies


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
        child_id = int(df.at[individual_id, 'pn_id_most_recent_child'])

        mother = df.loc[individual_id]
        child = df.loc[child_id]

        params = self.module.parameters

        # TODO: 'onset' date of all these condtions should be randomised throughout the remaining days of someones
        #  week1 (or we randomise the day they arrive at this event so that onset happens on the day the event runs
        #  which is scattered around week 1)

        # Run a number of checks to ensure only the correct women arrive here
        assert mother.la_is_postpartum
        assert ~mother.pn_sepsis_late_postpartum
        assert ~mother.pn_postpartum_haem_secondary
        assert (self.sim.date - mother.la_date_most_recent_delivery) < pd.to_timedelta(7, unit='d')

        assert child.age_days < 7
        assert ~child.pn_sepsis_early_neonatal
        # todo assert check to make sure not an inpatient?

        # If both the mother and newborn have died then this even wont run (it is possible for one or the other to have
        # died prior to this event- hence repeat checks on is_alive throughout)
        if ~mother.is_alive and ~child.is_alive:
            return

        x='y'
        # ===============================  MATERNAL COMPLICATIONS IN WEEK ONE  =======================================
        #  ------------------------------------- INFECTIONS AND SEPSIS ----------------------------------------------
        # TODO condition on having not already had the same type of infection immediately after birth

        # We determine if the mother will develop any postnatal infections and if they will progress into sepsis
        if mother.is_alive:
            logger.debug(key='message', data=f'Mother {individual_id} has arrived at PostnatalWeekOneEvent')

            for infection in ['endometritis', 'urinary_tract_inf', 'skin_soft_tissue_inf', 'other_maternal_infection']:
                    self.module.set_infections(individual_id, infection=infection)

            risk_sepsis = params['pn_linear_equations']['sepsis_late_postpartum'].predict(df.loc[[
                individual_id]])[individual_id]

            if risk_sepsis < self.module.rng.random_sample():
                logger.debug(key='message',
                             data=f'mother {individual_id} has developed postnatal sepsis during week one of the'
                                  f' postnatal period')

                df.at[individual_id, 'pn_sepsis_late_postpartum'] = True
                self.module.postnatal_tracker['postnatal_sepsis'] += 1

        #  ---------------------------------------- SECONDARY PPH ------------------------------------------------
            risk_secondary_pph = params['pn_linear_equations']['secondary_postpartum_haem'].predict(df.loc[[
                    individual_id]])[individual_id]

            if risk_secondary_pph < self.module.rng.random_sample():
                logger.debug(key='message',
                             data=f'mother {individual_id} has developed a secondary postpartum haemorrhage during'
                                  f' week one of the postnatal period')

                df.at[individual_id, 'pn_postpartum_haem_secondary'] = True
                self.module.postnatal_tracker['secondary_pph'] += 1

        # ----------------------------------- NEW ONSET DEFICIENCIES AND ANAEMIA -------------------------------------
            if ~self.module.deficiencies_following_pregnancy.has_any([individual_id], 'iron', first=True):

                if ~mother.la_iron_folic_acid_postnatal:
                    if self.module.rng.random_sample() < params['prob_iron_def_per_week_pn']:
                        self.module.deficiencies_following_pregnancy.set([individual_id],'iron')

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

                if risk_anaemia_after_pregnancy < self.module.rng.random_sample():
                    random_choice_severity = self.module.rng.choice(['mild', 'moderate', 'severe'], p=[0.33, 0.33,
                                                                                                       0.34], size=1)
                    df.at[individual_id, 'pn_anaemia_following_pregnancy'] = random_choice_severity

        # -------------------------------------------- HYPERTENSION --------------------------------------------------
            if mother.pn_htn_disorders != 'none':
                if self.module.rng.random_sample() < params['prob_htn_resolves']:
                    df.at[individual_id, 'pn_htn_disorders'] = 'none'

                else:
                    disease_states = ['gest_htn', 'severe_gest_htn', 'mild_pre_eclamp', 'severe_pre_eclamp',
                                      'eclampsia']
                    prob_matrix = pd.DataFrame(columns=disease_states, index=disease_states)

                    prob_matrix['gest_htn'] = [0.8, 0.1, 0.1, 0.0, 0.0]
                    prob_matrix['severe_gest_htn'] = [0.0, 0.8, 0.0, 0.2, 0.0]
                    prob_matrix['mild_pre_eclamp'] = [0.0, 0.0, 0.8, 0.2, 0.0]
                    prob_matrix['severe_pre_eclamp'] = [0.0, 0.0, 0.0, 0.6, 0.4]
                    prob_matrix['eclampsia'] = [0.0, 0.0, 0.0, 0.0, 1]

                #    current_status = df.loc[individual_id, 'pn_htn_disorders']
                #    new_status = util.transition_states(current_status, prob_matrix, self.module.rng)
                #    df.at[individual_id, "pn_htn_disorders"] = new_status

        #  ---------------------------- RISK OF POSTPARTUM PRE-ECLAMPSIA/HYPERTENSION --------------------------------
            if mother.pn_htn_disorders == 'none':
                risk_pe_after_pregnancy = params['pn_linear_equations']['pre_eclampsia_pn'].predict(df.loc[[
                    individual_id]])[individual_id]
                if risk_pe_after_pregnancy < self.module.rng.random_sample():
                    df.at[individual_id, 'pn_htn_disorders'] = 'mild_pre_eclamp'
                    df.at[individual_id, 'ps_prev_pre_eclamp'] = True  # todo: conventions
                else:
                    risk_gh_after_pregnancy = params['pn_linear_equations']['gest_htn_pn'].predict(df.loc[[
                        individual_id]])[individual_id]
                    if risk_gh_after_pregnancy < self.module.rng.random_sample():
                        df.at[individual_id, 'pn_htn_disorders'] = 'gest_htn'

        # ===============================  NEONATAL COMPLICATIONS IN WEEK ONE  =======================================
        if child.is_alive:
            logger.debug(key='message', data=f'Newborn {child_id} has arrived at PostnatalWeekOneEvent')

            # We then apply a risk that this womans newborn will develop sepsis during week one
            risk_eons = params['pn_linear_equations']['early_onset_neonatal_sepsis_week_1'].predict(
                        df.loc[[child_id]])[child_id]

            if self.module.rng.random_sample() < risk_eons:
                df.at[child_id, 'pn_sepsis_early_neonatal'] = True
                self.module.postnatal_tracker['early_neonatal_sepsis'] += 1

        # ===================================== CARE SEEKING FOR PNC 1 ==============================================
        # Women who deliver in facilities are asked to return for a first postnatal check up at day 7 post birth
        # For women where neither they, nor their newborn, develop any complications during the first week after birth,
        # we determine their probability that they will present for this visit on day seven

        # TODO: this is too complicated (simplify)

        # Currently we allow care seeking if
        mother_has_complications = None
        child_has_complications = None

        if mother.is_alive:
            if ~mother.pn_sepsis_late_postpartum and ~mother.pn_postpartum_haem_secondary and \
            mother.pn_htn_disorders != 'severe_pre_eclamp' and mother.pn_htn_disorders != 'eclampsia':
                mother_has_complications = False
            elif mother.pn_sepsis_late_postpartum or mother.pn_postpartum_haem_secondary or\
            mother.pn_htn_disorders == 'severe_pre_eclamp' or mother.pn_htn_disorders == 'eclampsia':
                mother_has_complications = True

        if child.is_alive:
            if ~child.pn_sepsis_early_neonatal:
                child_has_complications = False
            elif child.pn_sepsis_early_neonatal:
                child_has_complications = True

        if child_has_complications == False and mother_has_complications == False:
            if self.module.rng.random_sample() < params['prob_pnc1_at_day_7']:
                days_until_day_7 = self.sim.date - mother.la_date_most_recent_delivery
                days_until_day_7_int = int(days_until_day_7 / np.timedelta64(1, 'D'))

                pnc_one = HSI_PostnatalSupervisor_PostnatalCareContactOne(
                        self.module, person_id=individual_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(pnc_one,
                                                                    priority=0,
                                                                    topen=self.sim.date +
                                                                    DateOffset(days=days_until_day_7_int),
                                                                    tclose=None)

        # For women where either they or their baby has developed a complication during this time we assume liklihood
        # of care seeking for postnatal check up is higher

        elif mother_has_complications or child_has_complications:
            prob_care_seeking = params['prob_pnc1_at_day_7'] * params['multiplier_for_care_seeking_with_comps']

            if prob_care_seeking < self.module.rng.random_sample():
                    
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
        print(child_id)
        assert df.at[person_id, 'pn_pnc_visits_maternal'] == 0
        assert df.at[child_id, 'pn_pnc_visits_neonatal'] == 0

        if df.at[person_id, 'is_alive']:
            logger.debug(key='message', data=f'Mother {person_id} and child {child_id} have arrived for PNC1 on date'
                                             f' {self.sim.date}')

            maternal_pnc = HSI_PostnatalSupervisor_PostnatalCareContactTwo(
                                                             self.module, person_id=person_id)

            df.at[person_id, 'pn_pnc_visits_maternal'] += 1
            df.at[child_id, 'pn_pnc_visits_neonatal'] += 1

            self.module.assessment_for_maternal_complication_during_pnc(person_id, self)
            self.module.assessment_for_neonatal_complications_during_pnc(child_id, self)
            self.module.maternal_postnatal_care_care_seeking(person_id, 42, 'pnc2', maternal_pnc)


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

        print(person_id)
        print(child_id)
        print(df.at[person_id, 'pn_pnc_visits_maternal'])
        print(df.at[child_id, 'pn_pnc_visits_neonatal'])

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

            self.module.assessment_for_maternal_complication_during_pnc(person_id, self)
            self.module.assessment_for_neonatal_complications_during_pnc(child_id, self)
            # self.module.maternal_postnatal_care_care_seeking(person_id, 42, 'pnc3', maternal_pnc)


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

            self.module.assessment_for_maternal_complication_during_pnc(person_id, self)
            self.module.assessment_for_neonatal_complications_during_pnc(child_id, self)


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

        # ------------------------------------- HYPERTENSION TREATMENT -----------------------------------------------




        # ------------------------------------- ANAEMIA TREATMENT -----------------------------------------------


        self.module.apply_risk_of_maternal_or_neonatal_death_postnatal(mother_or_child='mother',
                                                                       individual_id=person_id)


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

        # TODO: manage patients admitted post birth

        if not child.is_alive:
            return

        if child.pn_sepsis_early_neonatal or child.pn_sepsis_late_neonatal:
            pkg_code_sep = pd.unique(consumables.loc[
                                         consumables['Intervention_Pkg'] == 'Newborn sepsis - full supportive care',
                                         'Intervention_Pkg_Code'])[0]

            consumables_needed = {'Intervention_Package_Code': {pkg_code_sep: 1}, 'Item_Code': {}}

            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self,
                cons_req_as_footprint=consumables_needed)

            if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_sep]:
                df.at[person_id, 'pn_sepsis_neonatal_full_supp_care'] = True

            else:
                logger.debug(key='message', data=f'neonate {person_id} was unable to receive treatment for sepsis '
                                                 f'due to limited resources')

            self.module.apply_risk_of_maternal_or_neonatal_death_postnatal(mother_or_child='child',
                                                                           individual_id=person_id)


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
