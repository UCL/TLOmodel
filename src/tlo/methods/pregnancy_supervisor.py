from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging, util
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import demography
from tlo.methods.labour import LabourOnsetEvent
from tlo.methods.antenatal_care import HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareContact,\
    HSI_CareOfWomenDuringPregnancy_MaternalEmergencyAssessment
from tlo.methods import Metadata
from tlo.util import BitsetHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PregnancySupervisor(Module):
    """This module is responsible for simulating the antenatal period of pregnancy (the period from conception until
     the termination of pregnancy). A number of outcomes are managed by this module including early pregnancy loss
     (induced/spontaneous abortion, ectopic pregnancy and antenatal stillbirth) and pregnancy complications of the
     antenatal period (nutritional deficiencies , anaemia, placental praevia/abruption,
     premature rupture of membranes (PROM), chorioamnionitis, hypertensive disorders (gestational hypertension,
     pre-eclampsia, eclampsia), gestational diabetes, maternal death). This module calculates likelihood of care seeking
     for routine antenatal care and emergency obstetric care in the event of severe complications."""

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # Here we define the pregnancy_disease_tracker dictionary used by the logger to calculate summary stats
        self.pregnancy_disease_tracker = dict()

    METADATA = {Metadata.DISEASE_MODULE,
                Metadata.USES_HEALTHBURDEN}  # declare that this is a disease module (leave as empty set otherwise)

    PARAMETERS = {
        'prob_ectopic_pregnancy': Parameter(
            Types.REAL, 'probability of a womans pregnancy being ectopic and not uterine implantation'),
        'rr_ectopic_smoker': Parameter(
            Types.REAL, 'relative risk of ectopic pregnancy in a smoker'),
        'prob_multiples': Parameter(
            Types.REAL, 'probability that a woman is currently carrying more than one pregnancy'),
        'prob_placenta_praevia': Parameter(
            Types.REAL, 'probability that this womans pregnancy will be complicated by placenta praevia'),
        'prob_spontaneous_abortion_per_month': Parameter(
            Types.REAL, 'underlying risk of spontaneous_abortion per month without the impact of risk factors'),
        'rr_spont_abortion_age_35': Parameter(
            Types.REAL, 'relative risk of spontaneous abortion in women aged 35 years or older'),
        'rr_spont_abortion_age_31_34': Parameter(
            Types.REAL, 'relative risk of spontaneous abortion in women aged 31-34 years old'),
        'prob_induced_abortion_per_month': Parameter(
            Types.REAL, 'underlying risk of induced abortion per month without the impact of risk factors'),
        'prob_haemorrhage_post_abortion': Parameter(
            Types.REAL, 'probability of haemorrhage following an abortion'),
        'prob_sepsis_post_abortion': Parameter(
            Types.REAL, 'probability of sepsis following an abortion'),
        'prob_injury_post_abortion': Parameter(
            Types.REAL, 'probability of injury following an abortion'),
        'baseline_prob_early_labour_onset': Parameter(
            Types.REAL, 'monthly baseline risk of early labour onset'),
        'rr_preterm_labour_post_prom': Parameter(
            Types.REAL, 'relative risk of early labour onset following PROM'),
        'prob_iron_def_per_month': Parameter(
            Types.REAL, 'monthly risk of a pregnant woman becoming iron deficient'),
        'prob_folate_def_per_month': Parameter(
            Types.REAL, 'monthly risk of a pregnant woman becoming folate deficient'),
        'rr_iron_def_ifa': Parameter(
            Types.REAL, 'treatment effect of iron supplementation on iron deficiency '),
        'rr_folate_def_ifa': Parameter(
            Types.REAL, 'treatment effect of folate supplementation on folate deficiency'),
        'prob_b12_def_per_month': Parameter(
            Types.REAL, 'monthly risk of a pregnant woman becoming b12 deficient'),
        'baseline_prob_anaemia_per_month': Parameter(
            Types.REAL, 'baseline risk of a woman developing anaemia secondary only to pregnant'),
        'rr_anaemia_if_iron_deficient': Parameter(
            Types.REAL, 'relative risk of a woman developing anaemia in pregnancy if she is iron deficient'),
        'rr_anaemia_if_folate_deficient': Parameter(
            Types.REAL, 'relative risk of a woman developing anaemia in pregnancy if she is folate deficient'),
        'rr_anaemia_if_b12_deficient': Parameter(
            Types.REAL, 'relative risk of a woman developing anaemia in pregnancy if she is b12 deficient'),
        'rr_anaemia_iron_folic_acid': Parameter(
            Types.REAL, 'risk reduction of maternal anaemia for women taking daily iron/folic acid'),
        'rr_anaemia_maternal_malaria': Parameter(
            Types.REAL, 'relative risk of anaemia secondary to malaria infection'),
        'prob_mild_mod_sev_anaemia': Parameter(
            Types.LIST, 'probabilities that a womans anaemia will be mild, moderate or severe'),
        'prob_pre_eclampsia_per_month': Parameter(
            Types.REAL, 'underlying risk of pre-eclampsia per month without the impact of risk factors'),
        'rr_pre_eclamp_calcium': Parameter(
            Types.REAL, 'risk reduction of pre-eclampsia for women taking daily calcium supplementation'),
        'prob_gest_htn_per_month': Parameter(
            Types.REAL, 'underlying risk of gestational hypertension per month without the impact of risk factors'),
        'rr_gest_htn_calcium': Parameter(
            Types.REAL, 'risk reduction of gestational hypertension for women taking daily calcium supplementation'),
        'prob_gest_diab_per_month': Parameter(
            Types.REAL, 'underlying risk of gestational diabetes per month without the impact of risk factors'),
        'prob_placental_abruption_per_month': Parameter(
            Types.REAL, 'monthly probability that a woman will develop placental abruption'),
        'prob_antepartum_haem_per_month': Parameter(
            Types.REAL, 'monthly probability that a woman will develop antepartum bleeding during pregnancy'),
        'prob_aph_placenta_praevia': Parameter(
            Types.REAL, 'risk of antepartum haemorrhage due to ongoing placenta praevia'),
        'prob_aph_placental_abruption': Parameter(
            Types.REAL, 'risk of antepartum haemorrhage due to placental abruption'),
        'prob_mod_sev_aph': Parameter(
            Types.LIST, 'probabilities that APH is mild/moderate or severe'),
        'prob_prom_per_month': Parameter(
            Types.REAL, 'monthly probability that a woman will experience premature rupture of membranes'),
        'prob_chorioamnionitis_post_prom': Parameter(
            Types.REAL, 'probability of a women developing chorioamnionitis following PROM '),
        'prob_still_birth_per_month': Parameter(
            Types.REAL, 'underlying risk of stillbirth per month without the impact of risk factors'),
        'rr_still_birth_gest_diab': Parameter(
            Types.REAL, 'relative risk of still birth in women with gestational diabetes'),
        'rr_still_birth_mild_pre_eclamp': Parameter(
            Types.REAL, 'relative risk of still birth in women with mild pre-eclampsia'),
        'rr_still_birth_gest_htn': Parameter(
            Types.REAL, 'relative risk of still birth in women with mild gestational hypertension'),
        'rr_still_birth_severe_gest_htn': Parameter(
            Types.REAL, 'relative risk of still birth in women with severe gestational hypertension'),
        'rr_still_birth_severe_pre_eclamp': Parameter(
            Types.REAL, 'relative risk of still birth in women with severe pre-eclampsia'),
        'rr_still_birth_food_supps': Parameter(
            Types.REAL, 'risk reduction of still birth for women receiving nutritional supplements'),
        'monthly_cfr_gest_htn': Parameter(
            Types.REAL, 'monthly risk of death associated with gestational hypertension'),
        'monthly_cfr_severe_gest_htn': Parameter(
            Types.REAL, 'monthly risk of death associated with severe gestational hypertension'),
        'monthly_cfr_mild_pre_eclamp': Parameter(
            Types.REAL, 'monthly risk of death associated with mild pre-eclampsia'),
        'monthly_cfr_severe_pre_eclamp': Parameter(
            Types.REAL, 'monthly risk of death associated with severe pre-eclampsia'),
        'monthly_cfr_gest_diab': Parameter(
            Types.REAL, 'monthly risk of death associated with gestational diabetes'),
        'monthly_cfr_severe_anaemia': Parameter(
            Types.REAL, 'monthly risk of death associated with severe anaemia'),
        'prob_ectopic_pregnancy_death': Parameter(
            Types.REAL, 'probability of a woman dying from a ruptured ectopic pregnancy'),
        'treatment_effect_ectopic_pregnancy': Parameter(
            Types.REAL, 'Treatment effect of ectopic pregnancy case management'),
        'prob_induced_abortion_death': Parameter(
            Types.REAL, 'underlying risk of death following an induced abortion'),
        'prob_spontaneous_abortion_death': Parameter(
            Types.REAL, 'underlying risk of death following an spontaneous abortion'),
        'treatment_effect_post_abortion_care': Parameter(
            Types.REAL, 'Treatment effect of post abortion care'),
        'prob_antepartum_haem_stillbirth': Parameter(
            Types.REAL, 'probability of stillbirth for a woman suffering acute antepartum haemorrhage'),
        'prob_antepartum_haem_death': Parameter(
            Types.REAL, 'probability of death for a woman suffering acute antepartum haemorrhage'),
        'prob_antenatal_spe_death': Parameter(
            Types.REAL, 'probability of death for a woman experiencing acute severe pre-eclampsia'),
        'prob_antenatal_ec_death': Parameter(
            Types.REAL, 'probability of death for a woman experiencing eclampsia'),
        'prob_monthly_death_severe_htn': Parameter(
            Types.REAL, 'monthly risk of death for a woman with severe hypertension'),
        'treatment_effect_severe_pre_eclampsia': Parameter(
            Types.REAL, 'treatment effect of  treatment of severe pre-eclampsia'),
        'prob_antenatal_ec_still_birth': Parameter(
            Types.REAL, 'probability of a stillbirth following an episode of eclampsia'),
        'treatment_effect_eclampsia': Parameter(
            Types.REAL, 'treatment effect of treatment for eclampsia'),
        'prob_first_anc_visit_gestational_age': Parameter(
            Types.LIST, 'probability of initiation of ANC by month'),
        'prob_four_or_more_anc_visits': Parameter(
            Types.REAL, 'probability of a woman undergoing 4 or more basic ANC visits'),
        'prob_eight_or_more_anc_visits': Parameter(
            Types.REAL, 'probability of a woman undergoing 8 or more basic ANC visits'),
        'probability_htn_persists': Parameter(
            Types.REAL, 'probability of a womans hypertension persisting post birth'),
        'prob_3_early_visits': Parameter(
            Types.REAL, 'DUMMY'),  # TODO: Remove
        'prob_seek_care_pregnancy_complication': Parameter(
            Types.REAL, 'Probability that a woman who is pregnant will seek care in the event of a complication'),
        'prob_seek_care_pregnancy_loss': Parameter(
            Types.REAL, 'Probability that a woman who has developed complications post pregnancy loss will seek care'),
    }

    PROPERTIES = {
        'ps_gestational_age_in_weeks': Property(Types.INT, 'current gestational age, in weeks, of this womans '
                                                           'pregnancy'),
        'ps_ectopic_pregnancy': Property(Types.BOOL, 'Whether this womans pregnancy is ectopic'),
        'ps_multiple_pregnancy': Property(Types.BOOL, 'Whether this womans is pregnant with multiple fetuses'),
        'ps_placenta_praevia': Property(Types.BOOL, 'Whether this womans pregnancy will be complicated by placenta'
                                                    'praevia'),
        'ps_deficiencies_in_pregnancy': Property(Types.INT, 'bitset column, stores types of anaemia causing '
                                                            'deficiencies in pregnancy'),
        'ps_anaemia_in_pregnancy': Property(Types.CATEGORICAL, 'whether a woman has anaemia in pregnancy and its '
                                                               'severity',
                                            categories=['none', 'mild', 'moderate', 'severe']),
        'ps_will_attend_four_or_more_anc': Property(Types.BOOL, 'Whether this womans is predicted to attend 4 or more '
                                                                'antenatal care visits during her pregnancy'),
        'ps_abortion_complications': Property(Types.INT, 'Bitset column holding types of abortion complication'),
        'ps_antepartum_still_birth': Property(Types.BOOL, 'whether this woman has experienced an antepartum still birth'
                                                          'of her current pregnancy'),
        'ps_previous_stillbirth': Property(Types.BOOL, 'whether this woman has had any previous pregnancies end in '
                                                       'still birth'),
        'ps_htn_disorders': Property(Types.CATEGORICAL, 'if this woman suffers from a hypertensive disorder of '
                                                        'pregnancy',
                                     categories=['none', 'gest_htn', 'severe_gest_htn', 'mild_pre_eclamp',
                                                 'severe_pre_eclamp', 'eclampsia']),
        'ps_prev_pre_eclamp': Property(Types.BOOL, 'whether this woman has experienced pre-eclampsia in a previous '
                                                   'pregnancy'),
        'ps_gest_diab': Property(Types.BOOL, 'whether this woman has gestational diabetes'),
        'ps_prev_gest_diab': Property(Types.BOOL, 'whether this woman has ever suffered from gestational diabetes '
                                                  'during a previous pregnancy'),
        'ps_placental_abruption': Property(Types.BOOL, 'Whether this woman is experiencing placental abruption'),
        'ps_antepartum_haemorrhage': Property(Types.CATEGORICAL, 'severity of this womans antepartum '
                                                                          'haemorrhage',
                                                       categories=['none', 'mild_moderate', 'severe']),
        'ps_premature_rupture_of_membranes': Property(Types.BOOL, 'whether this woman has experience rupture of '
                                                                  'membranes before the onset of labour. If this is '
                                                                  '<37 weeks from gestation the woman has preterm '
                                                                  'premature rupture of membranes'),

        'ps_chorioamnionitis': Property(Types.BOOL, 'whether this woman has chorioamnionitis following PROM'),
        'ps_emergency_event': Property(Types.BOOL, 'signifies a woman in undergoing an acute emergency event in her '
                                                   'pregnancy- used to consolidated care seeking in the instance of '
                                                   'multiple complications')
    }

    def read_parameters(self, data_folder):

        params = self.parameters
        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_PregnancySupervisor.xlsx',
                            sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)

        # Here we map 'disability' parameters to associated DALY weights to be passed to the health burden module
        if 'HealthBurden' in self.sim.modules.keys():
            params['ps_daly_weights'] = \
                {'abortive_outcome': self.sim.modules['HealthBurden'].get_daly_weight(352),
                 'ectopic_pregnancy': self.sim.modules['HealthBurden'].get_daly_weight(351),
                 'mild_iron_def_anaemia': self.sim.modules['HealthBurden'].get_daly_weight(476),
                 'moderate_iron_def_anaemia': self.sim.modules['HealthBurden'].get_daly_weight(480),
                 'severe_iron_def_anaemia': self.sim.modules['HealthBurden'].get_daly_weight(478),
                 'haemorrhage_moderate': self.sim.modules['HealthBurden'].get_daly_weight(339),
                 'haemorrhage_severe': self.sim.modules['HealthBurden'].get_daly_weight(338),
                 'maternal_sepsis': self.sim.modules['HealthBurden'].get_daly_weight(340),
                 'mild_htn_disorder': self.sim.modules['HealthBurden'].get_daly_weight(343),
                 'uncomplicated_dm': self.sim.modules['HealthBurden'].get_daly_weight(971)}

        # ==================================== LINEAR MODEL EQUATIONS =================================================
        # All linear equations used in this module are stored within the ps_linear_equations parameter below

        # TODO: process of 'selection' of important predictors in linear equations is ongoing, a linear model that
        #  is empty of predictors at the end of this process will be converted to a set probability

        params['ps_linear_equations'] = {

            # This equation calculates a womans risk of her current pregnancy being ectopic (implantation of the
            # embryo outside of the uterus). This risk is applied once per pregnancy
            'ectopic': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_ectopic_pregnancy'],
                Predictor('li_tob').when(True, params['rr_ectopic_smoker'])),

            # This equation is used to determine her current pregnancy will be twins. This risk is applied once per
            # pregnancy
            'multiples': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_multiples']),

            # This equation calculates a womans risk of placenta praevia (placenta partially/completely covers the
            # cervix). This risk is applied once per pregnancy
            'placenta_praevia': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_placenta_praevia']),

            # This equation calculates a womans monthly risk of spontaneous abortion (miscarriage)
            'spontaneous_abortion': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_spontaneous_abortion_per_month'],
                Predictor('age_years').when('>34', params['rr_spont_abortion_age_35'])
                                      .when('.between(30,35)', params['rr_spont_abortion_age_31_34'])),
            # TODO: previous SA as predictor

            # This equation calculates a womans monthly risk of induced abortion
            'induced_abortion': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_induced_abortion_per_month']),

            # This equation calculates a womans monthly risk of labour onsetting prior to her 'assigned' due date.
            # This drives preterm birth rates
            'early_onset_labour': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['baseline_prob_early_labour_onset'],
                Predictor('ps_premature_rupture_of_membranes').when(True, params['rr_preterm_labour_post_prom'])),

            # This equation calculates a womans monthly risk of developing anaemia during her pregnancy. This is
            # currently influenced by nutritional deficiencies and malaria status
            'maternal_anaemia': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['baseline_prob_anaemia_per_month'],
                Predictor('ps_deficiencies_in_pregnancy').apply(
                    lambda x: params['rr_anaemia_if_iron_deficient']
                    if x & self.deficiencies_in_pregnancy.element_repr('iron') else 1),
                Predictor('ps_deficiencies_in_pregnancy').apply(
                    lambda x: params['rr_anaemia_if_folate_deficient']
                    if x & self.deficiencies_in_pregnancy.element_repr('folate') else 1),
                Predictor('ps_deficiencies_in_pregnancy').apply(
                    lambda x: params['rr_anaemia_if_b12_deficient']
                    if x & self.deficiencies_in_pregnancy.element_repr('b12') else 1)),
            #   Predictor('ma_is_infected').when(True, params['rr_anaemia_maternal_malaria']),

            # This equation calculates a womans monthly risk of developing pre-eclampsia during her pregnancy. This is
            # currently influenced receipt of calcium supplementation
            'pre_eclampsia': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_pre_eclampsia_per_month'],
                Predictor('ac_receiving_calcium_supplements').when(True, params['rr_pre_eclamp_calcium'])),

            # This equation calculates a womans monthly risk of developing gestational hypertension
            # during her pregnancy. This is currently influenced receipt of calcium supplementation
            'gest_htn': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_gest_htn_per_month'],
                Predictor('ac_receiving_calcium_supplements').when(True, params['rr_gest_htn_calcium'])),

            # This equation calculates a womans monthly risk of developing gestational diabetes
            # during her pregnancy.
            'gest_diab': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_gest_diab_per_month']),

            # This equation calculates a womans monthly risk of developing placental abruption
            # during her pregnancy.
            'placental_abruption': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_placental_abruption_per_month']),

            # This equation calculates a womans monthly risk of developing antepartum haemorrhage during her pregnancy.
            # APH can only occur in the presence of one of two preceding causes (placenta praevia and placental
            # abruption) hence the use of an additive model
            'antepartum_haem': LinearModel(
                LinearModelType.ADDITIVE,
                0,
                Predictor('ps_placenta_praevia').when(True, params['prob_aph_placenta_praevia']),
                Predictor('ps_placental_abruption').when(True, params['prob_aph_placental_abruption']),
                ),
            # This equation calculates a womans monthly risk of developing premature rupture of membranes during her
            # pregnancy.
            'premature_rupture_of_membranes': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_prom_per_month']),

            # This equation calculates a womans risk of developing chorioamnionitis following PROM .
            'chorioamnionitis': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_chorioamnionitis_post_prom']),

            # This equation calculates a womans monthly risk of antenatal still birth
            'antenatal_stillbirth': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_still_birth_per_month'],
                Predictor('ps_gest_diab').when(True, params['rr_still_birth_gest_diab']),
                Predictor('ps_htn_disorders').when('mild_pre_eclamp', params['rr_still_birth_mild_pre_eclamp'])
                                             .when('gest_htn', params['rr_still_birth_gest_htn'])
                                             .when('severe_gest_htn', params['rr_still_birth_severe_gest_htn'])
                                             .when('severe_pre_eclamp', params['rr_still_birth_severe_pre_eclamp'])),
                #   Predictor('ma_is_infected').when(True, params['rr_still_birth_maternal_anaemia]),
                # TODO: chorio and other infections

            # This equation calculates a risk of dying after ectopic pregnancy and is mitigated by treatment
            'ectopic_pregnancy_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_ectopic_pregnancy_death'],
                Predictor('ac_ectopic_pregnancy_treated').when(True, params['treatment_effect_ectopic_pregnancy'])),

            # This equation calculates a risk of dying after complications following an induced abortion
            'induced_abortion_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_induced_abortion_death']),

            # This equation calculates a risk of dying after complications following a spontaneous abortion
            'spontaneous_abortion_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_spontaneous_abortion_death']),

            # This equation calculates a risk of still birth following an antepartum haemorrhage
            'antepartum_haemorrhage_stillbirth': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_antepartum_haem_stillbirth']),

            # This equation calculates a risk of dying following an antepartum haemorrhage
            'antepartum_haemorrhage_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_antepartum_haem_death']),

            # This equation calculates a risk of dying from eclampsia and is mitigated by treatment
            'eclampsia_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_antenatal_ec_death'],
                Predictor('ac_eclampsia_treatment').when(True, params['treatment_effect_eclampsia'])),

            # This equation calculates a risk of still birth following eclampsia
            'eclampsia_still_birth': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_antenatal_ec_still_birth']),

            'death_from_hypertensive_disorder': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_monthly_death_severe_htn']),

            'chorioamnionitis_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_chorioamnionitis']),

            'chorioamnionitis_still_birth': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_still_birth_chorioamnionitis']),

            # This equation calculates a the probability a woman will seek care following complications associated with
            # pregnancy loss (ectopic, induced/spontaneous abortion)
            'care_seeking_pregnancy_loss': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_seek_care_pregnancy_loss']),

            # This equation calculates a the probability a woman will seek care due to complications during pregnancy
            'care_seeking_pregnancy_complication': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_seek_care_pregnancy_complication']),

            # This equation calculates a the probability a woman will attend at least 4 ANC contacts during her
            # pregnancy - derived from Wingstons analysis of DHS data
            'four_or_more_anc_visits': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_four_or_more_anc_visits']),

            # This equation calculates a the probability a woman will attend at least 8 ANC contacts during her
            # pregnancy - derived from Wingstons analysis of DHS data
            'eight_or_more_anc_visits': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_eight_or_more_anc_visits']),
        }

    def initialise_population(self, population):

        df = population.props

        df.loc[df.is_alive, 'ps_gestational_age_in_weeks'] = 0
        df.loc[df.is_alive, 'ps_ectopic_pregnancy'] = False
        df.loc[df.is_alive, 'ps_placenta_praevia'] = False
        df.loc[df.is_alive, 'ps_multiple_pregnancy'] = False
        df.loc[df.is_alive, 'ps_anaemia_in_pregnancy'] = 'none'
        df.loc[df.is_alive, 'ps_will_attend_four_or_more_anc'] = False
        df.loc[df.is_alive, 'ps_antepartum_still_birth'] = False
        df.loc[df.is_alive, 'ps_previous_stillbirth'] = False
        df.loc[df.is_alive, 'ps_htn_disorders'] = 'none'
        df.loc[df.is_alive, 'ps_prev_pre_eclamp'] = False
        df.loc[df.is_alive, 'ps_gest_diab'] = False
        df.loc[df.is_alive, 'ps_prev_gest_diab'] = False
        df.loc[df.is_alive, 'ps_placental_abruption'] = False
        df.loc[df.is_alive, 'ps_antepartum_haemorrhage'] = 'none'
        df.loc[df.is_alive, 'ps_premature_rupture_of_membranes'] = False
        df.loc[df.is_alive, 'ps_chorioamnionitis'] = False
        df.loc[df.is_alive, 'ps_emergency_event'] = False

        # This biset property stores nutritional deficiencies that can occur in the antenatal period
        self.deficiencies_in_pregnancy = BitsetHandler(self.sim.population, 'ps_deficiencies_in_pregnancy',
                                                       ['iron', 'b12', 'folate'])

        # This biset property stores 'types' of complication that can occur after an abortion
        self.abortion_complications = BitsetHandler(self.sim.population, 'ps_abortion_complications',
                                                    ['sepsis', 'haemorrhage', 'injury'])

    def initialise_simulation(self, sim):

        # Register and schedule the PregnancySupervisorEvent
        sim.schedule_event(PregnancySupervisorEvent(self),
                           sim.date + DateOffset(days=0))

        # Register and schedule logging event
        sim.schedule_event(PregnancyLoggingEvent(self),
                           sim.date + DateOffset(years=1))

        # Define the conditions/outcomes we want to track
        self.pregnancy_disease_tracker = {'ectopic_pregnancy': 0, 'multiples': 0, 'placenta_praevia': 0,
                                          'placental_abruption': 0,'induced_abortion': 0, 'spontaneous_abortion': 0,
                                          'ectopic_pregnancy_death': 0, 'induced_abortion_death': 0,
                                          'spontaneous_abortion_death': 0, 'iron_def': 0, 'folate_def': 0, 'b12_def': 0,
                                          'maternal_anaemia': 0, 'antenatal_death': 0, 'antenatal_stillbirth': 0,
                                          'new_onset_pre_eclampsia': 0, 'new_onset_gest_diab': 0,
                                          'new_onset_gest_htn': 0, 'new_onset_severe_pe': 0, 'new_onset_eclampsia': 0,
                                          'antepartum_haem': 0, 'antepartum_haem_death': 0, 'prom': 0, 'pre_term': 0,
                                          'women_at_6_months': 0}

    def on_birth(self, mother_id, child_id):
        df = self.sim.population.props

        df.at[child_id, 'ps_gestational_age_in_weeks'] = 0
        df.at[child_id, 'ps_ectopic_pregnancy'] = False
        df.at[child_id, 'ps_placenta_praevia'] = False
        df.at[child_id, 'ps_multiple_pregnancy'] = False
        df.at[child_id, 'ps_anaemia_in_pregnancy'] = 'none'
        df.at[child_id, 'ps_will_attend_four_or_more_anc'] = False
        df.at[child_id, 'ps_antepartum_still_birth'] = False
        df.at[child_id, 'ps_previous_stillbirth'] = False
        df.at[child_id, 'ps_htn_disorders'] = 'none'
        df.at[child_id, 'ps_prev_pre_eclamp'] = False
        df.at[child_id, 'ps_gest_diab'] = False
        df.at[child_id, 'ps_prev_gest_diab'] = False
        df.at[child_id, 'ps_placental_abruption'] = False
        df.at[child_id, 'ps_antepartum_haemorrhage'] = 'none'
        df.at[child_id, 'ps_premature_rupture_of_membranes'] = False
        df.at[child_id, 'ps_chorioamnionitis'] = False
        df.at[child_id, 'ps_emergency_event'] = False

        # We reset all womans gestational age when they deliver as they are no longer pregnant
        df.at[mother_id, 'ps_gestational_age_in_weeks'] = 0

        # We currently assume that hyperglycemia due to gestational diabetes resolves following birth
        df.at[mother_id, 'ps_gest_diab'] = False

    def on_hsi_alert(self, person_id, treatment_id):
        logger.debug(key='message', data='This is PregnancySupervisor, being alerted about a health system interaction '
                                         f'person {person_id} for: {treatment_id}')

    def report_daly_values(self):
        df = self.sim.population.props
        p = self.parameters['ps_daly_weights']

        # TODO: discuss with TH best way to code this considering we dont store date of onset for each relevant variable

        logger.debug(key='message', data='This is PregnancySupervisor reporting my health values')

        health_values_1 = df.loc[df.is_alive, 'ps_ectopic_pregnancy'].map(
            {False: 0, True: p['ectopic_pregnancy']})
        health_values_1.name = 'Ectopic Pregnancy'

        health_values_2 = df.loc[df.is_alive, 'ps_anaemia_in_pregnancy'].map(
            {'none': 0,
             'mild': p['mild_iron_def_anaemia'],
             'moderate': p['moderate_iron_def_anaemia'],
             'severe': p['severe_iron_def_anaemia']})
        health_values_2.name = 'Anaemia during Pregnancy'

        health_values_3 = df.loc[df.is_alive, 'ps_antepartum_haemorrhage'].map(
            {'none': 0,
             'mild_moderate': p['haemorrhage_moderate'],
             'severe': p['haemorrhage_severe']})
        health_values_3.name = 'Antepartum haemorrhage'

        health_values_4 = df.loc[df.is_alive, 'ps_htn_disorders'].map(
            {'none': 0,
             'gest_htn': p['mild_htn_disorder'],
             'severe_gest_htn': p['mild_htn_disorder'],
             'mild_pre_eclamp': p['mild_htn_disorder'],
             'severe_pre_eclamp': p['mild_htn_disorder'],
             'eclampsia': p['mild_htn_disorder']})
        health_values_4.name = 'Hypertensive disorders'

        health_values_5 = df.loc[df.is_alive, 'ps_gest_diab'].map(
            {False: 0, True: p['uncomplicated_dm']})
        health_values_5.name = 'Gestational Diabetes'

        health_values_df = pd.concat([health_values_1.loc[df.is_alive], health_values_2.loc[df.is_alive],
                                      health_values_3.loc[df.is_alive], health_values_4.loc[df.is_alive],
                                      health_values_5.loc[df.is_alive]], axis=1)

        return health_values_df

        # TODO: Map DALY weights to bitset handler for abortion complications
        # TODO: Map DALY weights to sepsis from chorioamnionitis & abortion
        # TODO: Map DALY weights for more severe htn outcomes

        # TODO: continue discussion with TH regarding best way to map daly weights during pregnancy (cant use approach
        #  from depression as I dont record date of initiation/resolution of complications in the data frame)

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

    def apply_risk_of_spontaneous_abortion(self, gestation_of_interest):
        """
        This function applies risk of spontaneous abortion to a slice of data frame and is called by
        PregnancySupervisorEvent. It calls the apply_risk_of_abortion_complications for women who loose their pregnancy.
        :param gestation_of_interest: gestation in weeks
        """
        df = self.sim.population.props
        params = self.parameters

        # We use the apply_linear_model to determine if any women will experience spontaneous miscarriage
        spont_abortion = self.apply_linear_model(
            params['ps_linear_equations']['spontaneous_abortion'],
            df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == gestation_of_interest) &
                   ~df['ps_ectopic_pregnancy'] & ~df['ac_inpatient']])

        # The abortion function is called for women who lose their pregnancy. It resets properties, set complications
        # and care seeking
        for person in spont_abortion.loc[spont_abortion].index:
            self.apply_risk_of_abortion_complications(person, 'spontaneous_abortion')

    def apply_risk_of_induced_abortion(self, gestation_of_interest):
        """
        This function applies risk of induced abortion to a slice of data frame and is called by
        PregnancySupervisorEvent. It calls the apply_risk_of_abortion_complications for women who loose their pregnancy.
        :param gestation_of_interest: gestation in weeks
        """
        df = self.sim.population.props
        params = self.parameters

        # This function follows the same pattern as apply_risk_of_spontaneous_abortion
        abortion = self.apply_linear_model(
            params['ps_linear_equations']['induced_abortion'],
            df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == gestation_of_interest) &
                   ~df['ps_ectopic_pregnancy'] & ~df['ac_inpatient']])

        for person in abortion.loc[abortion].index:
            # Similarly the abortion function is called for each of these women
            self.apply_risk_of_abortion_complications(person, 'induced_abortion')

    def apply_risk_of_abortion_complications(self, individual_id, cause):
        """
        This function makes changes to the data frame for women who have experienced induced or spontaneous abortion.
        Additionally it determines if a woman will develop complications associated with pregnancy loss, manages care
        seeking and schedules potential death
        :param individual_id: individual_id
        :param cause: 'type' of abortion (spontaneous abortion OR induced abortion) (str)
        """
        df = self.sim.population.props
        params = self.parameters

        # Women who have an abortion have key pregnancy variables reset
        df.at[individual_id, 'is_pregnant'] = False
        df.at[individual_id, 'la_due_date_current_pregnancy'] = pd.NaT
        df.at[individual_id, 'ac_total_anc_visits_current_pregnancy'] = 0

        # We store the type of abortion for analysis
        self.pregnancy_disease_tracker[f'{cause}'] += 1

        # We apply a risk of developing specific complications associated with abortion type
        if cause == 'spontaneous_abortion' or cause == 'induced_abortion':
            if self.rng.random_sample() < params['prob_haemorrhage_post_abortion']:
                self.abortion_complications.set([individual_id], 'haemorrhage')
            if self.rng.random_sample() < params['prob_sepsis_post_abortion']:
                self.abortion_complications.set([individual_id], 'sepsis')
        if cause == 'spontaneous_abortion':
            if self.rng.random_sample() < params['prob_injury_post_abortion']:
                self.abortion_complications.set([individual_id], 'injury')

        # Determine if this woman will seek care, and schedule presentation to the health system
        if self.abortion_complications.has_any([individual_id], 'sepsis', 'haemorrhage', 'injury', first=True):
            self.care_seeking_pregnancy_loss_complications(individual_id)

            # Schedule possible death
            self.sim.schedule_event(EarlyPregnancyLossDeathEvent(self, individual_id, cause=f'{cause}'),
                                    self.sim.date + DateOffset(days=7))

            # We reset gestational age
            df.at[individual_id, 'ps_gestational_age_in_weeks'] = 0

    def apply_risk_of_deficiencies_and_anaemia(self, gestation_of_interest):
        """
        This function applies risk of deficiencies and anaemia to a slice of the data frame. It is called by
        PregnancySupervisorEvent
        :param gestation_of_interest: gestation in weeks
        """
        df = self.sim.population.props
        params = self.parameters

        # This function iterates through the three key anaemia causing deficiencies (iron, folate
        # and b12) and determines a the risk of onset for a subset of pregnant women. Following this these woman have a
        # probability of anaemia calculated and relevant changes to the data frame occur

        # ------------------------------------- IRON DEFICIENCY ------------------------------------------------------
        # First we select a subset of the pregnant population who are not iron deficient and are not receiving iron
        # supplements
        iron_def_no_ifa = ~self.deficiencies_in_pregnancy.has_all(
            df.is_alive & df.is_pregnant & (df.ps_gestational_age_in_weeks == gestation_of_interest) & ~df.ac_inpatient
            & ~df.la_currently_in_labour & ~df.ac_receiving_iron_folic_acid, 'iron')

        # We determine their risk of iron deficiency
        new_iron_def = pd.Series(self.rng.random_sample(len(iron_def_no_ifa)) < params['prob_iron_def_per_month'],
                                 index=iron_def_no_ifa.index)

        # And change their property accordingly
        self.deficiencies_in_pregnancy.set(new_iron_def.loc[new_iron_def].index, 'iron')
        self.pregnancy_disease_tracker['iron_def'] += len(new_iron_def.loc[new_iron_def])

        # Next we select women who aren't iron deficient but are receiving iron supplementation
        iron_def_ifa = ~self.deficiencies_in_pregnancy.has_all(
            df.is_alive & df.is_pregnant & (df.ps_gestational_age_in_weeks == gestation_of_interest) & ~df.ac_inpatient
            & ~df.la_currently_in_labour & df.ac_receiving_iron_folic_acid, 'iron')

        # We reduce their individual risk of deficiencies due to treatment and make changes to the data frame
        risk_of_iron_def = params['prob_iron_def_per_month'] * params['rr_iron_def_ifa']

        new_iron_def = pd.Series(self.rng.random_sample(len(iron_def_ifa)) < risk_of_iron_def,
                                 index=iron_def_ifa.index)

        self.deficiencies_in_pregnancy.set(new_iron_def.loc[new_iron_def].index, 'iron')
        self.pregnancy_disease_tracker['iron_def'] += len(new_iron_def.loc[new_iron_def])

        # ------------------------------------- FOLATE DEFICIENCY ------------------------------------------------------
        # This process is then repeated for folate and B12...
        folate_def_no_ifa = ~self.deficiencies_in_pregnancy.has_all(
            df.is_alive & df.is_pregnant & (df.ps_gestational_age_in_weeks == gestation_of_interest) & ~df.ac_inpatient
            & ~df.la_currently_in_labour & ~df.ac_receiving_iron_folic_acid, 'folate')

        new_folate_def = pd.Series(self.rng.random_sample(len(folate_def_no_ifa)) < params['prob_folate_def_per_month'],
                                   index=folate_def_no_ifa.index)

        self.deficiencies_in_pregnancy.set(new_folate_def.loc[new_folate_def].index, 'folate')
        self.pregnancy_disease_tracker['folate_def'] += len(new_folate_def.loc[new_folate_def])

        folate_def_ifa = ~self.deficiencies_in_pregnancy.has_all(
            df.is_alive & df.is_pregnant & (df.ps_gestational_age_in_weeks == gestation_of_interest) &
            ~df.ac_inpatient & ~df.la_currently_in_labour & df.ac_receiving_iron_folic_acid, 'folate')

        risk_of_folate_def = params['prob_folate_def_per_month'] * params['rr_folate_def_ifa']

        new_folate_def = pd.Series(self.rng.random_sample(len(folate_def_ifa)) < risk_of_folate_def,
                                   index=folate_def_ifa.index)

        self.deficiencies_in_pregnancy.set(new_folate_def.loc[new_folate_def].index, 'folate')
        self.pregnancy_disease_tracker['folate_def'] += len(new_folate_def.loc[new_folate_def])

        # ------------------------------------- B12 DEFICIENCY ------------------------------------------------------
        b12_def = ~self.deficiencies_in_pregnancy.has_all(
            df.is_alive & df.is_pregnant & (df.ps_gestational_age_in_weeks == gestation_of_interest) &
            ~df.ac_inpatient & ~df.la_currently_in_labour, 'b12')

        new_b12_def = pd.Series(self.rng.random_sample(len(b12_def)) < params['prob_b12_def_per_month'],
                                index=b12_def.index)

        self.deficiencies_in_pregnancy.set(new_b12_def.loc[new_b12_def].index, 'b12')
        self.pregnancy_disease_tracker['b12_def'] += len(new_b12_def.loc[new_b12_def])

        # TODO: Modify risk of future deficiencies in women treated for anaemia (with iron/folate/b12)

        # ------------------------------------------ ANAEMIA ---------------------------------------------------------
        # Now we determine if a subset of pregnant women will become anaemic using a linear model, in which the
        # preceding deficiencies act as predictors
        anaemia = self.apply_linear_model(
            params['ps_linear_equations']['maternal_anaemia'],
            df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == gestation_of_interest) &
                   ~df['ps_ectopic_pregnancy'] & ~df['ac_inpatient'] & ~df['la_currently_in_labour']])

        # We use a weight random draw to determine the severity of the anaemia
        random_choice_severity = pd.Series(self.rng.choice(['mild', 'moderate', 'severe'],
                                                           p=params['prob_mild_mod_sev_anaemia'],
                                                           size=len(anaemia.loc[anaemia])),
                                           index=anaemia.loc[anaemia].index)

        df.loc[anaemia.loc[anaemia].index, 'ps_anaemia_in_pregnancy'] = random_choice_severity

        if not anaemia.loc[anaemia].empty:
            logger.debug(key='message', data=f'The following women have developed anaemia during week '
                                             f'{gestation_of_interest}'
                                             f' of the postnatal period, {anaemia.loc[anaemia]}')
            self.pregnancy_disease_tracker['maternal_anaemia'] += len(anaemia.loc[anaemia])

    def apply_risk_of_gestational_diabetes(self, gestation_of_interest):
        """
        This function applies risk of gestational diabetes to a slice of the data frame. It is called by
        PregnancySupervisorEvent.
        :param gestation_of_interest: gestation in weeks
        """
        df = self.sim.population.props
        params = self.parameters

        gest_diab = self.apply_linear_model(
            params['ps_linear_equations']['gest_diab'], df.loc[df['is_alive'] & df['is_pregnant'] & (
                df['ps_gestational_age_in_weeks'] == gestation_of_interest) & ~df['ps_gest_diab'] &
                                                               ~df['ps_ectopic_pregnancy'] & ~df['ac_inpatient'] &
                                                               ~df['la_currently_in_labour']])

        df.loc[gest_diab.loc[gest_diab].index, 'ps_gest_diab'] = True
        df.loc[gest_diab.loc[gest_diab].index, 'ps_prev_gest_diab'] = True
        self.pregnancy_disease_tracker['new_onset_gest_diab'] += len(gest_diab.loc[gest_diab])

        if not gest_diab.loc[gest_diab].empty:
            logger.debug(key='message', data=f'The following women have developed gestational diabetes,'
                                             f'{gest_diab.loc[gest_diab].index}')

    def apply_risk_of_hypertensive_disorders(self, gestation_of_interest):
        """
        This function applies risk of mild pre-eclampsia and mild gestational diabetes to a slice of the data frame. It
        is called by PregnancySupervisorEvent.
        :param gestation_of_interest: gestation in weeks
        """
        df = self.sim.population.props
        params = self.parameters

        #  ----------------------------------- RISK OF PRE-ECLAMPSIA ----------------------------------------------
        # We assume all women must developed a mild pre-eclampsia/gestational hypertension before progressing to a more
        # severe disease - we do not apply incidence of severe pre-eclampsia/eclampsia explicitly like this - see
        # PregnancySupervisorEvent)
        pre_eclampsia = self.apply_linear_model(
            params['ps_linear_equations']['pre_eclampsia'],
            df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == gestation_of_interest) &
                   (df['ps_htn_disorders'] == 'none') & ~df['ps_ectopic_pregnancy'] & ~df['ac_inpatient']
                   & ~df['la_currently_in_labour']])

        df.loc[pre_eclampsia.loc[pre_eclampsia].index, 'ps_prev_pre_eclamp'] = True
        df.loc[pre_eclampsia.loc[pre_eclampsia].index, 'ps_htn_disorders'] = 'mild_pre_eclamp'
        self.pregnancy_disease_tracker['new_onset_pre_eclampsia'] += len(pre_eclampsia.loc[pre_eclampsia])

        if not pre_eclampsia.loc[pre_eclampsia].empty:
            logger.debug(key='message', data=f'The following women have developed pre_eclampsia '
                                             f'{pre_eclampsia.loc[pre_eclampsia].index}')

        #  -------------------------------- RISK OF GESTATIONAL HYPERTENSION --------------------------------------
        # For women who dont develop pre-eclampsia during this month, we apply a risk of gestational hypertension
        gest_hypertension = self.apply_linear_model(
            params['ps_linear_equations']['gest_htn'],
            df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == gestation_of_interest)
                   & (df['ps_htn_disorders'] == 'none') & ~df['ps_ectopic_pregnancy'] & ~df['ac_inpatient']
                   & ~df['la_currently_in_labour']])

        df.loc[gest_hypertension.loc[gest_hypertension].index, 'ps_htn_disorders'] = 'gest_htn'
        self.pregnancy_disease_tracker['new_onset_gest_htn'] += len(gest_hypertension.loc[gest_hypertension])
        if not gest_hypertension.loc[gest_hypertension].empty:
            logger.debug(key='message', data=f'The following women have developed gestational hypertension'
                                             f'{gest_hypertension.loc[gest_hypertension].index}')

    def apply_risk_of_progression_of_hypertension(self, gestation_of_interest):
        """
        This function applies a risk of progression of hypertensive disorders to women who are experiencing one of the
        hypertensive disorders. It is called by PregnancySupervisorEvent
        :param gestation_of_interest: gestation in weeks
        """
        df = self.sim.population.props

        # Select the relevant women
        selected = df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == gestation_of_interest) & \
                   (df.ps_htn_disorders != 'none') & ~df.la_currently_in_labour & ~df.ac_inpatient

        # Define the possible states that can be moved between
        disease_states = ['gest_htn', 'severe_gest_htn', 'mild_pre_eclamp', 'severe_pre_eclamp', 'eclampsia']
        prob_matrix = pd.DataFrame(columns=disease_states, index=disease_states)

        # Probability of moving between states is stored in a matrix
        prob_matrix['gest_htn'] = [0.8, 0.1, 0.1, 0.0, 0.0]
        prob_matrix['severe_gest_htn'] = [0.0, 0.8, 0.0, 0.2, 0.0]
        prob_matrix['mild_pre_eclamp'] = [0.0, 0.0, 0.8, 0.2, 0.0]
        prob_matrix['severe_pre_eclamp'] = [0.0, 0.0, 0.0, 0.6, 0.4]
        prob_matrix['eclampsia'] = [0.0, 0.0, 0.0, 0.0, 1]

        # We update the data frame with transitioned states (which may not have changed)
        current_status = df.loc[selected, "ps_htn_disorders"]
        new_status = util.transition_states(current_status, prob_matrix, self.rng)
        df.loc[selected, "ps_htn_disorders"] = new_status

        # We evaluate the series of women in this function and select the women who have transitioned to severe
        # pre-eclampsia
        assess_status_change_for_severe_pre_eclampsia = (current_status != "severe_pre_eclamp") & \
                                                        (new_status == "severe_pre_eclamp")
        new_onset_severe_pre_eclampsia = assess_status_change_for_severe_pre_eclampsia[
            assess_status_change_for_severe_pre_eclampsia]

        # For these women we set ps_emergency_event to True to signify they may seek care
        if not new_onset_severe_pre_eclampsia.empty:
            logger.debug(key='message',
                         data='The following women have developed severe pre-eclampsia during their '
                              f'pregnancy {new_onset_severe_pre_eclampsia.index} on {self.sim.date}')
            self.pregnancy_disease_tracker['new_onset_severe_pe'] += len(new_onset_severe_pre_eclampsia)

            for person in new_onset_severe_pre_eclampsia.index:
                df.at[person, 'ps_emergency_event'] = True

        # This process is then repeated for women who have developed eclampsia
        assess_status_change_for_eclampsia = (current_status != "eclampsia") & (new_status == "eclampsia")
        new_onset_eclampsia = assess_status_change_for_eclampsia[assess_status_change_for_eclampsia]

        if not new_onset_eclampsia.empty:
            logger.debug(key='message', data='The following women have developed eclampsia during their '
                                             f'pregnancy {new_onset_eclampsia.index} on {self.sim.date}')
            self.pregnancy_disease_tracker['new_onset_eclampsia'] += len(new_onset_eclampsia)

            for person in new_onset_eclampsia.index:
                df.at[person, 'ps_emergency_event'] = True

    def apply_risk_of_death_from_hypertension(self, gestation_of_interest):
        """
        This function applies risk of death to women with severe hypertensive disease (severe gestational hypertension/
        severe pre-eclampsia). For women who die this function schedules InstantaneousDeathEvent.
        :param gestation_of_interest: gestation in weeks
        """
        df = self.sim.population.props
        params = self.parameters

        at_risk_of_death_htn = self.apply_linear_model(
            params['ps_linear_equations']['death_from_hypertensive_disorder'],
            df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == gestation_of_interest) &
                   ~df['ps_ectopic_pregnancy'] & ~df['ac_inpatient'] & ~df['la_currently_in_labour'] &
                   (df['ps_htn_disorders'] == ('severe_gest_htn', 'severe_pre_eclamp'))])

        if not at_risk_of_death_htn.loc[at_risk_of_death_htn].empty:
            self.pregnancy_disease_tracker['antenatal_death'] += \
                len(at_risk_of_death_htn.loc[at_risk_of_death_htn].index)
            logger.debug(key='message',
                         data=f'The following women have died due to severe hypertensive disorder,'
                              f'{at_risk_of_death_htn.loc[at_risk_of_death_htn].index}')

            for person in at_risk_of_death_htn.loc[at_risk_of_death_htn].index:
                self.sim.schedule_event(demography.InstantaneousDeath(self, person,
                                                                      cause='maternal'), self.sim.date)

    def apply_risk_of_placental_abruption(self, gestation_of_interest):
        """
        This function applies risk of placental abruption to a slice of the dataframe. It is called by
        PregnancySupervisorEvent.
        :param gestation_of_interest: gestation in weeks
        """
        df = self.sim.population.props
        params = self.parameters

        placenta_abruption = self.apply_linear_model(
            params['ps_linear_equations']['placental_abruption'],
            df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == gestation_of_interest) &
                   ~df['ps_placental_abruption'] & ~df['ps_ectopic_pregnancy'] & ~df['ac_inpatient'] &
                   ~df['la_currently_in_labour']])

        df.loc[placenta_abruption.loc[placenta_abruption].index, 'ps_placental_abruption'] = True
        self.pregnancy_disease_tracker['placental_abruption'] += len(placenta_abruption.loc[placenta_abruption])

        if not placenta_abruption.loc[placenta_abruption].empty:
            logger.debug(key='message', data=f'The following women have developed placental abruption,'
                                             f'{placenta_abruption.loc[placenta_abruption].index}')
        # todo: this should cause some kind of emergency care seeking/response?
        # todo: link to stillbirth

    def apply_risk_of_antepartum_haemorrhage(self, gestation_of_interest):
        """
        This function applies risk of antepartum haemorrhage to a slice of the dataframe. It is called by
        PregnancySupervisorEvent.
        :param gestation_of_interest: gestation in weeks
        """
        df = self.sim.population.props
        params = self.parameters

        antepartum_haemorrhage = self.apply_linear_model(
            params['ps_linear_equations']['antepartum_haem'],
            df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == gestation_of_interest) &
                   ~df['ps_ectopic_pregnancy'] & ~df['ac_inpatient'] & ~df['la_currently_in_labour'] &
                   (df['ps_antepartum_haemorrhage'] == 'none')])

        # Weighted random draw is used to determine severity (for DALY weight mapping)
        random_choice_severity = pd.Series(self.rng.choice(['mild_moderate', 'severe'],
                                                           p=params['prob_mod_sev_aph'],
                                                           size=len(
                                                               antepartum_haemorrhage.loc[antepartum_haemorrhage])),
                                           index=antepartum_haemorrhage.loc[antepartum_haemorrhage].index)

        # We store the severity of the bleed and signify this woman is experiencing an emergency event
        df.loc[antepartum_haemorrhage.loc[antepartum_haemorrhage].index, 'ps_antepartum_haemorrhage'] = \
            random_choice_severity
        df.loc[antepartum_haemorrhage.loc[antepartum_haemorrhage].index, 'ps_emergency_event'] = True

        if not antepartum_haemorrhage.loc[antepartum_haemorrhage].empty:
            logger.debug(key='message', data=f'The following women are experiencing an antepartum haemorrhage,'
                                             f'{antepartum_haemorrhage.loc[antepartum_haemorrhage].index}')

        self.pregnancy_disease_tracker['antepartum_haem'] += len(antepartum_haemorrhage.loc[antepartum_haemorrhage])

    def apply_risk_of_premature_rupture_of_membranes(self, gestation_of_interest):
        """
        This function applies risk of premature rupture of membranes to a slice of the dataframe. It is called by
        PregnancySupervisorEvent.
        :param gestation_of_interest: gestation in weeks
        """
        df = self.sim.population.props
        params = self.parameters

        prom = self.apply_linear_model(
            params['ps_linear_equations']['premature_rupture_of_membranes'],
            df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == gestation_of_interest) &
                   ~df['ps_ectopic_pregnancy'] & ~df['ac_inpatient'] & ~df['la_currently_in_labour']])

        df.loc[prom.loc[prom].index, 'ps_premature_rupture_of_membranes'] = True

        # We allow women to seek care for PROM
        df.loc[prom.loc[prom].index, 'ps_emergency_event'] = True
        self.pregnancy_disease_tracker['prom'] += len(prom.loc[prom])

        if not prom.loc[prom].empty:
            logger.debug(key='message', data=f'The following women have experience premature rupture of membranes'
                                             f'{prom.loc[prom].index}')

            # All women with PROM are schedule this event in a weeks time. Women who havent sought care have a risk of
            # infection applied (women who have sought care will have increase risk of infection during delivery-
            # delivery is indicated as treatment for PROM in later stages)
            for person in prom.loc[prom].index:
                self.sim.schedule_event(ChorioamnionitisEvent(self, person),
                                        (self.sim.date + pd.Timedelta(days=7)))

    def apply_risk_of_preterm_labour(self, gestation_of_interest):
        """
        This function applies risk of preterm labour to a slice of the dataframe. It is called by
        PregnancySupervisorEvent.
        :param gestation_of_interest: gestation in weeks
        """
        df = self.sim.population.props
        params = self.parameters

        preterm_labour = self.apply_linear_model(
            params['ps_linear_equations']['early_onset_labour'],
            df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == gestation_of_interest)
                   & ~df['ps_ectopic_pregnancy'] & ~df['ac_inpatient'] & ~df['la_currently_in_labour']])

        if not preterm_labour.loc[preterm_labour].empty:
            logger.debug(key='message',
                         data=f'The following women will go into preterm labour at some point before the '
                              f'next month of their pregnancy: {preterm_labour.loc[preterm_labour].index}')
        self.pregnancy_disease_tracker['pre_term'] += len(preterm_labour.loc[preterm_labour])

        # To prevent clustering of labour onset we scatter women to go into labour on a random day before their
        # next month gestation
        for person in preterm_labour.loc[preterm_labour].index:
            if df.at[person, 'ps_gestational_age_in_weeks'] == 22:
                poss_day_onset = (27 - 22) * 7
                # We only allow labour to onset from 24 weeks (to match with our definition of preterm labour)
                onset_day = self.rng.randint(14, poss_day_onset)
                print('cooey onset day', onset_day)
            elif df.at[person, 'ps_gestational_age_in_weeks'] == 27:
                poss_day_onset = (31 - 27) * 7
                onset_day = self.rng.randint(0, poss_day_onset)
            elif df.at[person, 'ps_gestational_age_in_weeks'] == 31:
                poss_day_onset = (35 - 31) * 7
                onset_day = self.rng.randint(0, poss_day_onset)
            elif df.at[person, 'ps_gestational_age_in_weeks'] == 35:
                poss_day_onset = (37 - 35) * 7
                onset_day = self.rng.randint(0, poss_day_onset)

            # Due date is set and labour onset is scheduled
            # todo: could this be set in labour
            df.at[person, 'la_due_date_current_pregnancy'] = self.sim.date + DateOffset(days=onset_day)
            due_date = df.at[person, 'la_due_date_current_pregnancy']

            logger.debug(key='message', data=f'Mother {person} will go into preterm labour on '
                                             f'{self.sim.date + DateOffset(days=onset_day)}')

            self.sim.schedule_event(LabourOnsetEvent(self.sim.modules['Labour'], person),
                                    due_date)

    def apply_risk_of_still_birth(self, gestation_of_interest):
        """
        This function applies risk of still birth to a slice of the data frame. It is called by PregnancySupervisorEvent
        :param gestation_of_interest: INT used to select women from the data frame at certain gestations
        """
        df = self.sim.population.props
        params = self.parameters

        still_birth = self.apply_linear_model(
            params['ps_linear_equations']['antenatal_stillbirth'],
            df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == gestation_of_interest) &
                   ~df['ps_ectopic_pregnancy'] & ~df['ac_inpatient'] & ~df['la_currently_in_labour']])

        self.pregnancy_disease_tracker['antenatal_stillbirth'] += len(still_birth.loc[still_birth])

        # We reset the relevant pregnancy variables
        df.loc[still_birth.loc[still_birth].index, 'ps_antepartum_still_birth'] = True
        df.loc[still_birth.loc[still_birth].index, 'ps_previous_stillbirth'] = True
        df.loc[still_birth.loc[still_birth].index, 'is_pregnant'] = False
        df.loc[still_birth.loc[still_birth].index, 'la_due_date_current_pregnancy'] = pd.NaT
        df.loc[still_birth.loc[still_birth].index, 'ps_gestational_age_in_weeks'] = 0

        # And any pregnancy disease variabl...
        # TODO: reset disease variables for stillbirth

        if not still_birth.loc[still_birth].empty:
            logger.debug(key='message', data=f'The following women have have experienced an antepartum'
                                             f' stillbirth,{still_birth.loc[still_birth]}')

    def update_variables_post_still_birth_for_individual(self, individual_id):
        """
        This function resets variables for indivduals who have undergone still birth
        :param individual_id: individual_id
        """
        df = self.sim.population.props
        params = self.parameters
        logger.debug(key='message',
                     data=f'person {individual_id} has experience an antepartum stillbirth on '
                          f'date {self.sim.date}')

        self.pregnancy_disease_tracker['antenatal_stillbirth'] += 1

        df.at[individual_id, 'ps_antepartum_still_birth'] = True
        df.at[individual_id, 'ps_previous_stillbirth'] = True
        df.at[individual_id, 'is_pregnant'] = False
        df.at[individual_id, 'la_due_date_current_pregnancy'] = pd.NaT
        df.at[individual_id, 'ps_gestational_age_in_weeks'] = 0

    def care_seeking_pregnancy_loss_complications(self, individual_id):
        """
        This function manages care seeking for women experiencing ectopic pregnancy or complications following
        spontaneous/induced abortion.
        :param individual_id: individual_id
        :return: Returns True/False value to signify care seeking
        """
        df = self.sim.population.props
        params = self.parameters

        # Determine probability of care seeking via the linear model
        if self.rng.random_sample() < \
            params['ps_linear_equations']['care_seeking_pregnancy_loss'].predict(
                df.loc[[individual_id]])[individual_id]:
            logger.debug(key='message', data=f'Mother {individual_id} will seek care following pregnancy loss')

            # We assume women will seek care via HSI_GenericEmergencyFirstApptAtFacilityLevel1 and will be admitted for
            # care
            from tlo.methods.hsi_generic_first_appts import (
                HSI_GenericEmergencyFirstApptAtFacilityLevel1)

            event = HSI_GenericEmergencyFirstApptAtFacilityLevel1(
                module=self,
                person_id=individual_id)

            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))
            return True

        else:
            logger.debug(key='message', data=f'Mother {individual_id} will not seek care following pregnancy loss')
            return False

    def property_reset(self, individual_id):
        """
        This function is called by PostnatalSupervisor on birth to reset pregnancy variables
        :param individual_id: individual_id
        """
        df = self.sim.population.props

        df.at[individual_id, 'ps_anaemia_in_pregnancy'] = 'none'
        df.at[individual_id, 'ps_htn_disorders'] = 'none'
        self.deficiencies_in_pregnancy.unset(individual_id, 'iron', 'folate', 'b12')


class PregnancySupervisorEvent(RegularEvent, PopulationScopeEventMixin):
    """ This is the PregnancySupervisorEvent, it is a weekly event which has two primary functions.
    1.) It updates the gestational age (in weeks) of all women who are pregnant.
    2.) It applies risk of complications/outcomes and care seeking during the antenatal period at pre-defined time
    points of pregnancy (defined below)"""

    def __init__(self, module, ):
        super().__init__(module, frequency=DateOffset(weeks=1))

    def apply(self, population):
        df = population.props
        params = self.module.parameters

        # =================================== UPDATING LENGTH OF PREGNANCY ============================================
        # Length of pregnancy is commonly measured as gestational age which commences on the first day of a womans last
        # menstrual period (therefore including around 2 weeks in which a woman isnt pregnant)

        # We calculate a womans gestational age by first calculating the foetal age (measured from conception) and then
        # adding 2 weeks. The literature describing the epidemiology of maternal conditions almost exclusivly uses
        # gestational age

        alive_and_preg = df.is_alive & df.is_pregnant
        foetal_age_in_days = self.sim.date - df.loc[alive_and_preg, 'date_of_last_pregnancy']
        foetal_age_in_weeks = round(foetal_age_in_days / np.timedelta64(1, 'W'))

        df.loc[alive_and_preg, 'ps_gestational_age_in_weeks'] = foetal_age_in_weeks.astype('int64') + 2

        # foetal_age_in_weeks rounds first week down to 0- here we force the lowest GA possible to be 3
        women_ga_2 = df.is_alive & df.is_pregnant & (df.ps_gestational_age_in_weeks == 2.0)
        df.loc[women_ga_2, 'ps_gestational_age_in_weeks'] = 3.0

        logger.debug(key='message', data=f'updating gestational ages on date {self.sim.date}')

        assert (df.loc[alive_and_preg, 'ps_gestational_age_in_weeks'] > 2).all().all()

        # =========================== APPLYING RISK OF ADVERSE PREGNANCY OUTCOMES =====================================
        # The aim of this event is to apply risk of certain outcomes of pregnancy at relevant points in a womans
        # gestation. Risk of complications that occur only once during pregnancy (below) are applied within the event,
        # otherwise code applying risk is stored in functions (below)

        # At the beginning of pregnancy (3 weeks GA (and therefore 1st week of pregnancy) we determine if a woman will
        # develop ectopic pregnancy, multiple pregnancy,placenta praevia and if/when she will seek care for her first
        # antenatal visit

        #  ------------------------------APPLYING RISK OF ECTOPIC PREGNANCY -------------------------------------------
        # We use the apply_linear_model function to determine which women will develop ectopic pregnancy
        ectopic_risk = self.module.apply_linear_model(
            params['ps_linear_equations']['ectopic'],
            df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == 3)])

        # Make the appropriate changes to the data frame and log the number of ectopic pregnancies
        df.loc[ectopic_risk.loc[ectopic_risk].index, 'ps_ectopic_pregnancy'] = True
        self.module.pregnancy_disease_tracker['ectopic_pregnancy'] += len(ectopic_risk.loc[ectopic_risk])

        if not ectopic_risk.loc[ectopic_risk].empty:
            logger.debug(key='message', data=f'The following women have experience an ectopic pregnancy,'
                                             f'{ectopic_risk.loc[ectopic_risk].index}')

            # For women whose pregnancy is ectopic we scheduled them to the EctopicPregnancyEvent in between 3-5 weeks
            # of pregnancy (this simulates time period prior to which symptoms onset- and may trigger care seeking)
            for person in ectopic_risk.loc[ectopic_risk].index:
                self.sim.schedule_event(EctopicPregnancyEvent(self.module, person),
                                        (self.sim.date + pd.Timedelta(days=7 * 3 + self.module.rng.randint(0, 7 * 2))))

        #  ---------------------------- APPLYING RISK OF MULTIPLE PREGNANCY -------------------------------------------
        # For the women who aren't having an ectopic, we determine if they may be carrying multiple pregnancies and make
        # changes accordingly
        multiples = self.module.apply_linear_model(
            params['ps_linear_equations']['multiples'],
            df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == 3) &
                   ~df['ps_ectopic_pregnancy']])

        df.loc[multiples.loc[multiples].index, 'ps_multiple_pregnancy'] = True
        self.module.pregnancy_disease_tracker['multiples'] += len(multiples.loc[multiples])

        if not multiples.loc[multiples].empty:
            logger.debug(key='message', data=f'The following women are pregnant with multiples, '
                                             f'{multiples.loc[multiples].index}')

        # TODO: This may not remain in the model due to complexities

        #  -----------------------------APPLYING RISK OF PLACENTA PRAEVIA  -------------------------------------------
        # Next,we apply a one of risk of placenta praevia (placenta will grow to cover the cervix either partially or
        # completely) which will increase likelihood of bleeding later in pregnancy
        placenta_praevia = self.module.apply_linear_model(
            params['ps_linear_equations']['placenta_praevia'],
            df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == 3) &
                   ~df['ps_ectopic_pregnancy']])

        df.loc[placenta_praevia.loc[placenta_praevia].index, 'ps_placenta_praevia'] = True
        self.module.pregnancy_disease_tracker['placenta_praevia'] += len(placenta_praevia.loc[placenta_praevia])

        if not placenta_praevia.loc[multiples].empty:
            logger.debug(key='message',
                         data=f'The following womens pregnancy is complicated by placenta praevia '
                              f'{placenta_praevia.loc[placenta_praevia].index,}')

        # ----------------------------------- SCHEDULING FIRST ANC VISIT -----------------------------------------
        # Finally for these women we determine care seeking for the first antenatal care contact of their
        # pregnancy. We use a linear model to determine if women will attend four or more visits
        anc_attendance = self.module.apply_linear_model(
            params['ps_linear_equations']['four_or_more_anc_visits'],
            df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == 3) &
                   ~df['ps_ectopic_pregnancy']])

        # This property is used  during antenatal care to ensure the right % of women attend four or more visits
        df.loc[anc_attendance.loc[anc_attendance].index, 'ps_will_attend_four_or_more_anc'] = True

        # Gestation of first ANC visit (month) is selected via a random weighted draw to represent the distribution
        # of first ANC attendance by gestational age
        for person in anc_attendance.index:
            random_draw_gest_at_anc = self.module.rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                                             p=params['prob_first_anc_visit_gestational_age'])

            # We use month ten to capture women who will never attend ANC during their pregnancy
            if random_draw_gest_at_anc == 10:
                pass
            else:
                first_anc_date = self.sim.date + DateOffset(months=random_draw_gest_at_anc)
                first_anc_appt = HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareContact(
                    self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person)

                self.sim.modules['HealthSystem'].schedule_hsi_event(first_anc_appt, priority=0,
                                                                    topen=first_anc_date,
                                                                    tclose=first_anc_date + DateOffset(days=7))

        # ------------------------ APPLY RISK OF ADDITIONAL PREGNANCY COMPLICATIONS -----------------------------------
        # The following functions apply risk of key complications/outcomes of pregnancy as specific time points of a
        # mothers gestation in weeks. These 'gestation_of_interest' parameters roughly represent the last week in each
        # month of pregnancy. These time  points at which risk is applied, vary between complications according to their
        # epidemiology

        # The application of these risk is intentionally ordered as described below

        # Women in the first five months of pregnancy are at risk of spontaneous abortion (miscarriage)
        for gestation_of_interest in [4, 8, 13, 17, 22]:
            self.module.apply_risk_of_spontaneous_abortion(gestation_of_interest=gestation_of_interest)

        # From the second month of pregnancy until month 5 women who do not experience spontaneous abortion may undergo
        # induced abortion
        for gestation_of_interest in [8, 13, 17, 22]:
            self.module.apply_risk_of_induced_abortion(gestation_of_interest=gestation_of_interest)

        # Every month a risk of micronutrient deficiency and maternal anaemia is applied
        for gestation_of_interest in [4, 8, 13, 17, 22, 27, 31, 35, 40]:
            self.module.apply_risk_of_deficiencies_and_anaemia(gestation_of_interest=gestation_of_interest)

        # In the later months of pregnancy we apply a background risk of still birth. Risk of still birth is applied to
        # all women every month, with additional risk applied to women who experience acute pregnancy emergencies
        # (antepartum haemorrhage, eclampsia, chorioamnionitis)
        for gestation_of_interest in [27, 31, 35, 40, 41, 42, 43, 44, 45]:
            self.module.apply_risk_of_still_birth(gestation_of_interest=gestation_of_interest)

        # For women whose pregnancy will continue will apply a risk of developing a number of acute and chronic
        # (length of pregnancy) complications
        for gestation_of_interest in [22, 27, 31, 35, 40]:
            self.module.apply_risk_of_hypertensive_disorders(gestation_of_interest=gestation_of_interest)
            self.module.apply_risk_of_gestational_diabetes(gestation_of_interest=gestation_of_interest)
            self.module.apply_risk_of_placental_abruption(gestation_of_interest=gestation_of_interest)
            self.module.apply_risk_of_antepartum_haemorrhage(gestation_of_interest=gestation_of_interest)
            self.module.apply_risk_of_premature_rupture_of_membranes(gestation_of_interest=gestation_of_interest)

        for gestation_of_interest in [27, 31, 35, 40]:
            # Women with hypertension are at risk of there condition progression, this risk is applied months 6-9
            self.module.apply_risk_of_progression_of_hypertension(gestation_of_interest=gestation_of_interest)
            # And of death...
            self.module.apply_risk_of_death_from_hypertension(gestation_of_interest=gestation_of_interest)

        # From month 5-8 we apply risk of a woman going into early labour
        for gestation_of_interest in [22, 27, 31, 35]:
            self.module.apply_risk_of_preterm_labour(gestation_of_interest=gestation_of_interest)

        # ------------------------------- CARE SEEKING FOR PREGNANCY EMERGENCIES --------------------------------------
        # Every week when the event runs we determine if any women who have experience an emergency event in pregnancy
        # will seek care
        care_seeking = self.module.apply_linear_model(
            params['ps_linear_equations']['care_seeking_pregnancy_complication'],
            df.loc[df['is_alive'] & df['is_pregnant'] & ~df['ps_ectopic_pregnancy'] & df['ps_emergency_event'] &
                   ~df['ac_inpatient'] & ~df['la_currently_in_labour'] & (df['la_due_date_current_pregnancy']
                                                                          != self.sim.date)])

        # We reset this variable to prevent additional unnecessary care seeking next month
        df.loc[care_seeking.index, 'ps_emergency_event'] = False

        # We assume women who seek care will present to a form of Maternal Assessment Unit- not through normal A&E
        for person in care_seeking.loc[care_seeking].index:
            logger.debug(key='message', data=f'Mother {person} will seek care following acute pregnancy'
                                             f'complications')

            acute_pregnancy_hsi = HSI_CareOfWomenDuringPregnancy_MaternalEmergencyAssessment(
                self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person)

            self.sim.modules['HealthSystem'].schedule_hsi_event(acute_pregnancy_hsi, priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

            # TODO: follow through all conditions to make sure r.o.d is applied (via HSIs?)

        if not care_seeking.loc[~care_seeking].empty:
            logger.debug(key='message', data=f'The following women will not seek care after experiencing a '
                                             f'pregnancy emergency: {care_seeking.loc[~care_seeking].index}')

            # -------- APPLYING RISK OF DEATH/STILL BIRTH FOR NON-CARE SEEKERS FOLLOWING PREGNANCY EMERGENCIES --------
            for person in care_seeking.loc[~care_seeking].index:
                mother = df.loc[person]

                antenatal_death = False
                still_birth = False

                # As women could have more than one severe complication at once we apply risk of death from each
                # complication in turn
                if mother.ps_antepartum_haemorrhage != 'none':
                    risk_of_death = params['ps_linear_equations']['antepartum_haemorrhage_death'].predict(
                        df.loc[[person]])[person]

                    if self.module.rng.random_sample() < risk_of_death:
                        antenatal_death = True

                    # For women who survive we determine if they will experience a stillbirth
                    else:
                        risk_of_still_birth = params['ps_linear_equations'][
                            'antepartum_haemorrhage_stillbirth'].predict(df.loc[[person]])[person]
                        if self.module.rng.random_sample() < risk_of_still_birth:
                            still_birth = True

                        # And reset the relevant variables
                        df.at[person, 'ps_antepartum_haemorrhage'] = 'none'

                # This process if repeated for eclampsia
                if mother.ps_htn_disorders == 'eclampsia':
                    risk_of_death = params['ps_linear_equations']['eclampsia_death'].predict(
                        df.loc[[person]])[person]

                    if self.module.rng.random_sample() < risk_of_death:
                        antenatal_death = True
                    else:
                        risk_of_still_birth = params['ps_linear_equations'][
                            'eclampsia_still_birth'].predict(df.loc[[person]])[person]
                        if self.module.rng.random_sample() < risk_of_still_birth:
                            still_birth = True

                        df.at[person, 'ps_htn_disorders'] = 'severe_pre_eclamp'

                # For women who will die we schedule the InstantaneousDeath Event
                if antenatal_death:
                    logger.debug(key='message',
                                 data=f'mother {person} has died following a pregnancy emergency on '
                                      f'date {self.sim.date}')

                    self.sim.schedule_event(demography.InstantaneousDeath(self, person,
                                                                          cause='maternal'), self.sim.date)
                    self.module.pregnancy_disease_tracker['antenatal_death'] += 1

                # And for women who lose their pregnancy we reset the relevant variables
                elif still_birth:
                    self.module.update_variables_post_still_birth_for_individual(person)

            # todo: whats the best way to apply to the data frame and avoid a for loop? (to allow for multiple
            #  possible causes of death?)


class EctopicPregnancyEvent(Event, IndividualScopeEventMixin):
    """This is EctopicPregnancyEvent. It is scheduled by the set_pregnancy_complications function within
     PregnancySupervisorEvent. This event makes changes to the data frame for women with ectopic pregnancies, applies a
      probability of care seeking and schedules the EctopicRuptureEvent."""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props

        # Check only the right women have arrived here
        assert df.at[individual_id, 'ps_ectopic_pregnancy']
        assert df.at[individual_id, 'ps_gestational_age_in_weeks'] < 9
        assert ~df.at[individual_id, 'ac_inpatient']

        if not df.at[individual_id, 'is_alive']:
            return

        # reset pregnancy variables
        df.at[individual_id, 'is_pregnant'] = False
        df.at[individual_id, 'ps_gestational_age_in_weeks'] = 0
        df.at[individual_id, 'la_due_date_current_pregnancy'] = pd.NaT

        # Determine if women will seek care at this stage
        if not self.module.care_seeking_pregnancy_loss_complications(individual_id):

            # For women who dont seek care (and get treatment) we schedule EctopicPregnancyRuptureEvent (simulating
            # fallopian tube rupture) in an additional 2-4 weeks from this event (if care seeking is unsuccessful
            # then this event is scheduled by the HSI (did_not_run)
            self.sim.schedule_event(EctopicPregnancyRuptureEvent(self.module, individual_id),
                                    (self.sim.date + pd.Timedelta(days=7 * 2 + self.module.rng.randint(0, 7 * 2))))


class EctopicPregnancyRuptureEvent(Event, IndividualScopeEventMixin):
    """This is EctopicPregnancyRuptureEvent. It is scheduled by the EctopicPregnancyEvent for women who have
    experienced an ectopic pregnancy which has ruptured due to lack of treatment. This event manages care seeking post
    rupture and schedules EarlyPregnancyLossDeathEvent"""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props

        # Check the right woman has arrived at this event
        assert df.at[individual_id, 'ps_ectopic_pregnancy']

        if df.at[individual_id, 'is_alive']:
            logger.debug(key='message', data=f'persons {individual_id} untreated ectopic pregnancy has now ruptured on '
                                             f'date {self.sim.date}')

            # We see if this woman will now seek care following rupture
            self.module.care_seeking_pregnancy_loss_complications(individual_id)

        # We delayed the death event by three days to allow any treatment effects to mitigate risk of death
        self.sim.schedule_event(EarlyPregnancyLossDeathEvent(self.module, individual_id, cause='ectopic_pregnancy'),
                                self.sim.date + DateOffset(days=3))


class EarlyPregnancyLossDeathEvent(Event, IndividualScopeEventMixin):
    """This is EarlyPregnancyLossDeathEvent. It is scheduled by the EctopicPregnancyRuptureEvent & abortion for
    women who are at risk of death following a loss of their pregnancy"""

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

        self.cause = cause

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters

        if df.at[individual_id, 'is_alive']:

            # Individual risk of death is calculated through the linear model
            risk_of_death = params['ps_linear_equations'][f'{self.cause}_death'].predict(df.loc[[individual_id]])[
                individual_id]

            # If the death occurs we record it here
            if self.module.rng.random_sample() < risk_of_death:
                logger.debug(key='message', data=f'person {individual_id} has died due to {self.cause} on date '
                                                 f'{self.sim.date}')

                self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                      cause='maternal'), self.sim.date)
                self.module.pregnancy_disease_tracker[f'{self.cause}_death'] += 1
                self.module.pregnancy_disease_tracker['antenatal_death'] += 1


class ChorioamnionitisEvent(Event, IndividualScopeEventMixin):
    """This is ChorioamnionitisEvent. It is scheduled by the apply_risk_of_premature_rupture_of_membranes function in
    the PregnancySupervisorEvent. This event applies risk of chorioamnionitis to women with PROM who did not seek care
    for treatment and are at risk of infection"""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        mother = df.loc[individual_id]

        if not mother.is_alive:
            return

        # Check the correct women are sent here
        assert mother.ps_premature_rupture_of_membranes

        # Risk of chorioamnionitis following PROM is only applied to women who are not inpatients
        if ~mother.ac_inpatient and ~mother.la_currently_in_labour:

            # If this woman will develop infection we see if she will seek care for treatment
            if self.module.rng.random_sample() < params['prob_chorioamnionitis_post_prom']:
                df.at[individual_id, 'ps_chorioamnionitis'] = True

                care_seeking = params['ps_linear_equations'][
                    'care_seeking_pregnancy_complication'].predict(df.loc[[individual_id]])[individual_id]

                # If she does seek care we schedule the appropriate HSI
                if self.module.rng.random_sample() < care_seeking:
                    acute_pregnancy_hsi = HSI_CareOfWomenDuringPregnancy_MaternalEmergencyAssessment(
                        self.sim.modules['CareOfWomenDuringPregnancy'], person_id=individual_id)

                    self.sim.modules['HealthSystem'].schedule_hsi_event(acute_pregnancy_hsi, priority=0,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(days=1))

                # Otherwise we apply risk of both death and still birth due to this infection
                else:
                    risk_of_death = params['ps_linear_equations'][
                        'chorioamnionitis_death'].predict(df.loc[[individual_id]])[individual_id]
                    risk_of_still_birth = params['ps_linear_equations'][
                        'chorioamnionitis_still_birth'].predict(df.loc[[individual_id]])[individual_id]

                    if self.module.rng.random_sample() < risk_of_death:
                        self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                              cause='maternal'), self.sim.date)
                        self.module.pregnancy_disease_tracker['antenatal_death'] += 1

                    elif self.module.rng.random_sample() < risk_of_still_birth:
                        self.module.update_variables_post_still_birth_for_individual(individual_id)


class PregnancyLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """This is PregnancyLoggingEvent. It runs yearly to produce summary statistics around pregnancy."""

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

        # Numerators
        antenatal_maternal_deaths = self.module.pregnancy_disease_tracker['antenatal_death']
        antepartum_stillbirths = self.module.pregnancy_disease_tracker['antenatal_stillbirth']

        if antenatal_maternal_deaths == 0:
            antenatal_maternal_deaths = 1

        if antepartum_stillbirths == 0:
            antepartum_stillbirths = 1

        total_ectopics = self.module.pregnancy_disease_tracker['ectopic_pregnancy']
        total_multiples = self.module.pregnancy_disease_tracker['multiples']
        total_abortions_t = self.module.pregnancy_disease_tracker['induced_abortion']
        total_spontaneous_abortions_t = self.module.pregnancy_disease_tracker['spontaneous_abortion']
        total_iron_def = self.module.pregnancy_disease_tracker['iron_def']
        total_folate_def = self.module.pregnancy_disease_tracker['folate_def']
        total_b12_def = self.module.pregnancy_disease_tracker['b12_def']
        total_anaemia_cases = self.module.pregnancy_disease_tracker['maternal_anaemia']
        total_ia_deaths = self.module.pregnancy_disease_tracker['induced_abortion_death']
        total_sa_deaths = self.module.pregnancy_disease_tracker['spontaneous_abortion_death']
        crude_new_onset_pe = self.module.pregnancy_disease_tracker['new_onset_pre_eclampsia']
        crude_new_gh = self.module.pregnancy_disease_tracker['new_onset_gest_htn']
        crude_new_gd = self.module.pregnancy_disease_tracker['new_onset_gest_diab']
        crude_new_spe = self.module.pregnancy_disease_tracker['new_onset_severe_pe']
        crude_new_ec = self.module.pregnancy_disease_tracker['new_onset_eclampsia']
        placenta_praevia = self.module.pregnancy_disease_tracker['placenta_praevia']
        placental_abruption = self.module.pregnancy_disease_tracker['placental_abruption']
        crude_aph = self.module.pregnancy_disease_tracker['antepartum_haem']
        prom = self.module.pregnancy_disease_tracker['prom']
        preterm = self.module.pregnancy_disease_tracker['pre_term']

        crude_aph_death = self.module.pregnancy_disease_tracker['antepartum_haem_death']
        total_ectopic_deaths = self.module.pregnancy_disease_tracker['ectopic_pregnancy_death']
        women_month_6 = self.module.pregnancy_disease_tracker['women_at_6_months']

        dict_for_output = {'repro_women': total_women_reproductive_age,
                           'crude_deaths': antenatal_maternal_deaths,
                           'antenatal_mmr': (antenatal_maternal_deaths/total_births_last_year) * 100000,
                           'crude_antenatal_sb': antepartum_stillbirths,
                           'antenatal_sbr': (antepartum_stillbirths/total_births_last_year) * 100,
                           'total_spontaneous_abortions': total_spontaneous_abortions_t,
                           'spontaneous_abortion_rate': (total_spontaneous_abortions_t /
                                                         total_women_reproductive_age) * 1000,
                           'spontaneous_abortion_death_rate': (total_sa_deaths / total_women_reproductive_age) * 1000,
                           'total_induced_abortions': total_abortions_t,
                           'induced_abortion_rate': (total_abortions_t / total_women_reproductive_age) * 1000,
                           'induced_abortion_death_rate': (total_ia_deaths / total_women_reproductive_age) * 1000,
                           'crude_multiples': total_multiples,
                           'multiples_rate': (total_multiples / total_women_reproductive_age) * 1000,
                           'crude_ectopics': total_ectopics,
                           'ectopic_rate': (total_ectopics / total_women_reproductive_age) * 1000,
                           'crude_iron_def': total_iron_def,
                           'crude_folate_def': total_folate_def,
                           'crude_b12_def': total_b12_def,
                           'crude_anaemia': total_anaemia_cases,
                           'anaemia_rate': (total_anaemia_cases/total_women_reproductive_age) * 1000,
                           'crude_pe': crude_new_onset_pe,
                           'crude_gest_htn': crude_new_gh,
                           'crude_gd': crude_new_gd,
                           'crude_spe': crude_new_spe,
                           'crude_ec': crude_new_ec,
                           'crude_p_p': placenta_praevia,
                           'crude_p_a': placental_abruption,
                           'crude_aph': crude_aph,
                           'crude_prom': prom,
                           'crude_pre_term': preterm}

        logger.info(key='ps_summary_statistics', data=dict_for_output,
                    description='Yearly summary statistics output from the pregnancy supervisor module')

        self.module.pregnancy_disease_tracker = {'ectopic_pregnancy': 0, 'multiples': 0, 'placenta_praevia': 0,
                                                 'placental_abruption': 0,'induced_abortion': 0,
                                                 'spontaneous_abortion': 0,
                                                 'ectopic_pregnancy_death': 0, 'induced_abortion_death': 0,
                                                 'spontaneous_abortion_death': 0, 'iron_def': 0, 'folate_def': 0,
                                                 'b12_def': 0,
                                                 'maternal_anaemia': 0, 'antenatal_death': 0,
                                                 'antenatal_stillbirth': 0,
                                                 'new_onset_pre_eclampsia': 0, 'new_onset_gest_diab': 0,
                                                 'new_onset_gest_htn': 0, 'new_onset_severe_pe': 0,
                                                 'new_onset_eclampsia': 0,
                                                 'antepartum_haem': 0, 'antepartum_haem_death': 0, 'prom': 0,
                                                 'pre_term': 0,
                                                 'women_at_6_months':0}
