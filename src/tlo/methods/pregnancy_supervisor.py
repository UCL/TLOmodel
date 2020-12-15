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
     for routine antenatal care and emergency obstetric care in the event of severe complications.."""

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
        'prob_multiples': Parameter(
            Types.REAL, 'probability that a woman is currently carrying more than one pregnancy'),
        'prob_placenta_praevia': Parameter(
            Types.REAL, 'probability that this womans pregnancy will be complicated by placenta praevia'),
        'prob_spontaneous_abortion_per_month': Parameter(
            Types.REAL, 'underlying risk of spontaneous_abortion per month without the impact of risk factors'),
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
        'prob_prom_per_month': Parameter(
            Types.REAL, 'monthly probability that a woman will experience premature rupture of membranes'),
        'prob_chorioamnionitis_post_prom': Parameter(
            Types.REAL, 'probability of a women developing chorioamnionitis following PROM '),
        'prob_still_birth_per_month': Parameter(
            Types.REAL, 'underlying risk of stillbirth per month without the impact of risk factors'),
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
        'ps_antepartum_haemorrhage': Property(Types.BOOL, 'whether this woman has developed an antepartum haemorrhage'),
        'ps_antepartum_haemorrhage_severity': Property(Types.CATEGORICAL, 'severity of this womans antepartum '
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
                params['prob_ectopic_pregnancy']),

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
                params['prob_spontaneous_abortion_per_month']),

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

            # This equation calculates a womans monthly risk of antenatal still birth (excluding acute causes)
            'antenatal_stillbirth': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_still_birth_per_month'],
                Predictor('ac_receiving_diet_supplements').when(True, params['rr_still_birth_food_supps'])),

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

            # This equation calculates a risk of dying from severe pre-eclampsia and is mitigated by treatment
            'severe_pre_eclamp_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_antenatal_spe_death'],
                Predictor('ac_severe_pre_eclampsia_treatment').when(True, params['treatment_effect_severe_pre_'
                                                                                 'eclampsia'])),

            # This equation calculates a risk of dying from eclampsia and is mitigated by treatment
            'eclampsia_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_antenatal_ec_death'],
                Predictor('ac_eclampsia_treatment').when(True, params['treatment_effect_eclampsia'])),

            # This equation calculates a risk of still birth following eclampsia
            'eclampsia_still_birth': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_antenatal_ec_still_birth']),

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
        df.loc[df.is_alive, 'ps_antepartum_haemorrhage'] = False
        df.loc[df.is_alive, 'ps_antepartum_haemorrhage_severity'] = 'none'
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
        df.at[child_id, 'ps_antepartum_haemorrhage'] = False
        df.at[child_id, 'ps_antepartum_haemorrhage_severity'] = 'none'
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

        health_values_3 = df.loc[df.is_alive, 'ps_antepartum_haemorrhage_severity'].map(
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

    def set_deficiencies_and_anaemia_status(self, ga_in_weeks):
        """
        This function applies risk of deficiencies and anaemia to a slice of the data frame. It is abstracted to a
        function to prevent repeats in the set_pregnancy_complication function
        :param ga_in_weeks: INT used to select women from the data frame at certain gestations
        """
        df = self.sim.population.props
        params = self.parameters

        # Applied monthly, this function iterates through the three key anaemia causing deficiencies (iron, folate
        # and b12) and determines a the risk of onset for a subset of pregnant women. Following this these woman have a
        # probability of anaemia calculated and relevant changes to the data frame occur

        # ------------------------------------- IRON DEFICIENCY ------------------------------------------------------
        # First we select a subset of the pregnant population who are not iron deficient and are not receiving iron
        # supplements
        iron_def_no_ifa = ~self.deficiencies_in_pregnancy.has_all(
            df.is_alive & df.is_pregnant & (df.ps_gestational_age_in_weeks == ga_in_weeks) & ~df.ac_inpatient &
            ~df.la_currently_in_labour & ~df.ac_receiving_iron_folic_acid, 'iron')

        # We determine their risk of iron deficiency
        new_iron_def = pd.Series(self.rng.random_sample(len(iron_def_no_ifa)) < params['prob_iron_def_per_month'],
                                 index=iron_def_no_ifa.index)

        # And change their property accordingly
        self.deficiencies_in_pregnancy.set(new_iron_def.loc[new_iron_def].index, 'iron')
        self.pregnancy_disease_tracker['iron_def'] += len(new_iron_def.loc[new_iron_def])

        # Next we select women who aren't iron deficient but are receiving iron supplementation
        iron_def_ifa = ~self.deficiencies_in_pregnancy.has_all(
            df.is_alive & df.is_pregnant & (df.ps_gestational_age_in_weeks == ga_in_weeks) & ~df.ac_inpatient &
            ~df.la_currently_in_labour & df.ac_receiving_iron_folic_acid, 'iron')

        # We reduce their individual risk of deficiencies due to treatment and make changes to the data frame
        risk_of_iron_def = params['prob_iron_def_per_month'] * params['rr_iron_def_ifa']

        new_iron_def = pd.Series(self.rng.random_sample(len(iron_def_ifa)) < risk_of_iron_def,
                                 index=iron_def_ifa.index)

        self.deficiencies_in_pregnancy.set(new_iron_def.loc[new_iron_def].index, 'iron')
        self.pregnancy_disease_tracker['iron_def'] += len(new_iron_def.loc[new_iron_def])

        # ------------------------------------- FOLATE DEFICIENCY ------------------------------------------------------
        # This process is then repeated for folate and B12...
        folate_def_no_ifa = ~self.deficiencies_in_pregnancy.has_all(
            df.is_alive & df.is_pregnant & (df.ps_gestational_age_in_weeks == ga_in_weeks) & ~df.ac_inpatient &
            ~df.la_currently_in_labour & ~df.ac_receiving_iron_folic_acid, 'folate')

        new_folate_def = pd.Series(self.rng.random_sample(len(folate_def_no_ifa)) < params['prob_folate_def_per_month'],
                                   index=folate_def_no_ifa.index)

        self.deficiencies_in_pregnancy.set(new_folate_def.loc[new_folate_def].index, 'folate')
        self.pregnancy_disease_tracker['folate_def'] += len(new_folate_def.loc[new_folate_def])

        folate_def_ifa = ~self.deficiencies_in_pregnancy.has_all(
            df.is_alive & df.is_pregnant & (df.ps_gestational_age_in_weeks == ga_in_weeks) & ~df.ac_inpatient &
            ~df.la_currently_in_labour & df.ac_receiving_iron_folic_acid, 'folate')

        risk_of_folate_def = params['prob_folate_def_per_month'] * params['rr_folate_def_ifa']

        new_folate_def = pd.Series(self.rng.random_sample(len(folate_def_ifa)) < risk_of_folate_def,
                                   index=folate_def_ifa.index)

        self.deficiencies_in_pregnancy.set(new_folate_def.loc[new_folate_def].index, 'folate')
        self.pregnancy_disease_tracker['folate_def'] += len(new_folate_def.loc[new_folate_def])

        # ------------------------------------- B12 DEFICIENCY ------------------------------------------------------
        b12_def = ~self.deficiencies_in_pregnancy.has_all(
            df.is_alive & df.is_pregnant & (df.ps_gestational_age_in_weeks == ga_in_weeks) & ~df.ac_inpatient &
            ~df.la_currently_in_labour, 'b12')

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
            df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == ga_in_weeks) &
                   ~df['ps_ectopic_pregnancy'] & ~df['ac_inpatient'] & ~df['la_currently_in_labour']])

        # We use a weight random draw to determine the severity of the anaemia
        random_choice_severity = pd.Series(self.rng.choice(['mild', 'moderate', 'severe'],
                                                           p=params['prob_mild_mod_sev_anaemia'],
                                                           size=len(anaemia.loc[anaemia])),
                                           index=anaemia.loc[anaemia].index)

        df.loc[anaemia.loc[anaemia].index, 'ps_anaemia_in_pregnancy'] = random_choice_severity

        if not anaemia.loc[anaemia].empty:
            logger.debug(key='message', data=f'The following women have developed anaemia during week {ga_in_weeks}'
                                             f' of the postnatal period, {anaemia.loc[anaemia]}')
            self.pregnancy_disease_tracker['maternal_anaemia'] += len(anaemia.loc[anaemia])

    def set_pregnancy_complications(self, ga_in_weeks):
        """
        This function is called by the PregnancySupervisorEvent and applies risk of the following complications to women
         (dependent on their current gestation): Ectopic pregnancy, multiple pregnancy, placenta praevia,
         spontaneous abortion, induced abortion, still birth, placental abruption, deficiencies & anaemia
         (via set_deficiencies_and_anaemia_status function), antepartum haemorrhage, gestational hypertension,
         pre-eclampsia, eclampsia, gestational diabetes. For women with hypertension this function calculates their
         risk of progression to a more severe state of disease. Additionally this function determines care seeking for
         women's first antenatal care contact (ANC1) and care seeking for an emergency related to pregnancy. Finally it
         applies risk of death to women who develop complications but dont seek care.

        :param ga_in_weeks: INT used to select women from the data frame at certain gestation
        """
        df = self.sim.population.props
        params = self.parameters

        # This function is called in the PregnancySupervisorEvent. Broadly this represents application of risk on a
        # monthly basis, with some exceptions explained below. Susceptibility to/likelihood of experiencing certain
        # outcomes depends on gestational age therefore the ga_in_weeks parameter is used to select a subset of pregnant
        # women to determine which risks should be applied. The relevant weeks of interest are shown in the Pregnancy
        # SupervisorEvent

        # ======================================= FIRST WEEK OF PREGNANCY ============================================
        if ga_in_weeks == 1:
            #  --------------------------------- RISK OF ECTOPIC PREGNANCY -------------------------------------------
            # For all women who are in the first week of their pregnancy we apply a risk that this pregnancy will be
            # ectopic (fetus implanted outside the uterus)
            ectopic_risk = self.apply_linear_model(
                params['ps_linear_equations']['ectopic'],
                df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == ga_in_weeks)])

            df.loc[ectopic_risk.loc[ectopic_risk].index, 'ps_ectopic_pregnancy'] = True
            self.pregnancy_disease_tracker['ectopic_pregnancy'] += len(ectopic_risk.loc[ectopic_risk])

            if not ectopic_risk.loc[ectopic_risk].empty:
                logger.debug(key='message', data=f'The following women have experience an ectopic pregnancy,'
                                                 f'{ectopic_risk.loc[ectopic_risk].index}')

            # For women whose pregnancy is ectopic we scheduled them to the EctopicPregnancyEvent in between 4-6 weeks
            # of pregnancy (this simulates time period prior to which symptoms onset- and may trigger care seeking)
            for person in ectopic_risk.loc[ectopic_risk].index:
                self.sim.schedule_event(EctopicPregnancyEvent(self, person),
                                        (self.sim.date + pd.Timedelta(days=7 * 4 + self.rng.randint(0, 7 * 2))))

            #  --------------------------------- RISK OF MULTIPLE PREGNANCY -------------------------------------------
            # For the 'week 1 women' who aren't having an ectopic, we determine if they may be carrying twins
            multiples = self.apply_linear_model(
                params['ps_linear_equations']['multiples'],
                df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == ga_in_weeks) &
                       ~df['ps_ectopic_pregnancy']])

            df.loc[multiples.loc[multiples].index, 'ps_multiple_pregnancy'] = True
            self.pregnancy_disease_tracker['multiples'] += len(multiples.loc[multiples])

            if not multiples.loc[multiples].empty:
                logger.debug(key='message', data=f'The following women are pregnant with multiples, '
                                                 f'{multiples.loc[multiples].index}')

            # TODO: This may not remain in the model due to complexities

            #  --------------------------------- RISK OF PLACENTA PRAEVIA  -------------------------------------------
            # Next, still looking at 'week 1 women' we apply a one of risk of placenta praevia (placenta will grow to
            # cover the cervix either partially or completely) which will increase likelihood of bleeding later in
            # pregnancy
            placenta_praevia = self.apply_linear_model(
                    params['ps_linear_equations']['placenta_praevia'],
                    df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == ga_in_weeks) &
                           ~df['ps_ectopic_pregnancy']])

            df.loc[placenta_praevia.loc[placenta_praevia].index, 'ps_placenta_praevia'] = True
            self.pregnancy_disease_tracker['placenta_praevia'] += len(placenta_praevia.loc[placenta_praevia])

            if not placenta_praevia.loc[multiples].empty:
                logger.debug(key='message',
                             data=f'The following womens pregnancy is complicated by placenta praevia '
                                  f'{placenta_praevia.loc[placenta_praevia].index,}')

            # ----------------------------------- SCHEDULING FIRST ANC VISIT -----------------------------------------
            # Finally for 'week 1 women' we determine care seeking for the first antenatal care contact of their
            # pregnancy

            # We use a linear model to determine if women will attend four or more visits
            anc_attendance = self.apply_linear_model(
                params['ps_linear_equations']['four_or_more_anc_visits'],
                df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == ga_in_weeks) &
                       ~df['ps_ectopic_pregnancy']])

            # This property is used  during antenatal care to ensure the right % of women attend four or more visits
            df.loc[anc_attendance.loc[anc_attendance].index, 'ps_will_attend_four_or_more_anc'] = True

            # Gestation of first ANC visit (month) is selected via a random weighted draw to represent the distribution
            # of first ANC attendance by gestational age
            for person in anc_attendance.index:
                random_draw_gest_at_anc = self.rng.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
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

        # ======================================= MONTHS 1-4 OF PREGNANCY ============================================
        elif ga_in_weeks == 4 or ga_in_weeks == 8 or ga_in_weeks == 13 or ga_in_weeks == 17:
            # Next we apply risks of complications/outcomes that can occur on months 1, 2, 3 or 4 of pregnancy

            #  ----------------------------- RISK OF SPONTANEOUS ABORTION ---------------------------------------------
            # A monthly risk of spontaneous abortion is first applied
            miscarriage = self.apply_linear_model(
                params['ps_linear_equations']['spontaneous_abortion'],
                df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == ga_in_weeks) &
                       ~df['ps_ectopic_pregnancy'] & ~df['ac_inpatient']])

            for person in miscarriage.loc[miscarriage].index:
                # the abortion function is called for women who lose their pregnancy (reset properties,
                # set complications and care seeking)
                self.abortion(person, 'spontaneous_abortion')

            #  -------------------------------- RISK OF INDUCED ABORTION ----------------------------------------------
            # For women who are 2-4 months pregnant we apply a risk of induced abortion
            if not ga_in_weeks == 4:
                abortion = self.apply_linear_model(
                    params['ps_linear_equations']['induced_abortion'],
                    df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == ga_in_weeks) &
                           ~df['ps_ectopic_pregnancy'] & ~df['ac_inpatient']])

                for person in abortion.loc[abortion].index:

                    # Similarly the abortion function is called for each of these women
                    self.abortion(person, 'induced_abortion')

            #  ------------------------ RISK OF NUTRITIONAL DEFICIENCIES AND ANAEMIA ----------------------------------
            # Finally we determine if any women will develop any deficiencies or anaemia during this month of their
            # pregnancy
            self.set_deficiencies_and_anaemia_status(ga_in_weeks)

        # ======================================= MONTHS 5-9 OF PREGNANCY ============================================
        # Next we apply risks of complications/outcomes that can occur on months 5, 6, 7, 8 or 9 of pregnancy. We
        # selecting a slice from the data frame we ensure women are not currently inpatients or in labour as onsetting
        # new conditions can lead to care seeking which disrupts the event flow in labour/antenatally

        elif ga_in_weeks == 22 or ga_in_weeks == 27 or ga_in_weeks == 31 or ga_in_weeks == 35 or ga_in_weeks == 40:
            #  ----------------------------- RISK OF SPONTANEOUS ABORTION ---------------------------------------------
            # We allow induced/spontaneous abortion to occur in month five- after which risk of stillbirth is applied
            if ga_in_weeks == 22:
                miscarriage = self.apply_linear_model(
                    params['ps_linear_equations']['spontaneous_abortion'],
                    df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == ga_in_weeks) &
                           ~df['ps_ectopic_pregnancy'] & ~df['ac_inpatient'] & ~df['la_currently_in_labour']])

                for person in miscarriage.loc[miscarriage].index:
                    self.abortion(person, 'spontaneous_abortion')

            #  -------------------------------- RISK OF INDUCED ABORTION ----------------------------------------------
                abortion = self.apply_linear_model(
                    params['ps_linear_equations']['induced_abortion'],
                    df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == ga_in_weeks) &
                           ~df['ps_ectopic_pregnancy'] & ~df['ac_inpatient'] & ~df['la_currently_in_labour']])

                for person in abortion.loc[abortion].index:
                    self.abortion(person, 'induced_abortion')

            else:
                #  ----------------------------- RISK OF STILL BIRTH --------------------------------------------------
                # From month 6 onwards we apply an baseline risk of still birth (still birth is also driven by acute
                # pregnancy events)
                still_birth = self.apply_linear_model(
                    params['ps_linear_equations']['antenatal_stillbirth'],
                    df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == ga_in_weeks) &
                           ~df['ps_ectopic_pregnancy'] & ~df['ac_inpatient'] & ~df['la_currently_in_labour']])

                self.pregnancy_disease_tracker['antenatal_stillbirth'] += len(still_birth.loc[still_birth])

                # We reset the relevant pregnancy variables
                df.loc[still_birth.loc[still_birth].index, 'ps_antepartum_still_birth'] = True
                df.loc[still_birth.loc[still_birth].index, 'ps_previous_stillbirth'] = True
                df.loc[still_birth.loc[still_birth].index, 'is_pregnant'] = False
                df.loc[still_birth.loc[still_birth].index, 'la_due_date_current_pregnancy'] = pd.NaT
                df.loc[still_birth.loc[still_birth].index, 'ps_gestational_age_in_weeks'] = 0

                # TODO: reset disease variables for stillbirth

                if not still_birth.loc[still_birth].empty:
                    logger.debug(key='message', data=f'The following women have have experienced an antepartum'
                                                     f' stillbirth,{still_birth.loc[still_birth]}')

            #  ------------------------ RISK OF NUTRITIONAL DEFICIENCIES AND ANAEMIA ----------------------------------
            self.set_deficiencies_and_anaemia_status(ga_in_weeks)

            #  ----------------------------------- RISK OF PRE-ECLAMPSIA ----------------------------------------------
            # From month five we allow women to develop a number of other key conditions  # TODO CARRY ON HERE
            pre_eclampsia = self.apply_linear_model(
                params['ps_linear_equations']['pre_eclampsia'],
                df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == ga_in_weeks) &
                       (df['ps_htn_disorders'] == 'none') & ~df['ps_ectopic_pregnancy'] & ~df['ac_inpatient']
                       & ~df['la_currently_in_labour']])

            df.loc[pre_eclampsia.loc[pre_eclampsia].index, 'ps_prev_pre_eclamp'] = True
            df.loc[pre_eclampsia.loc[pre_eclampsia].index, 'ps_htn_disorders'] = 'mild_pre_eclamp'
            self.pregnancy_disease_tracker['new_onset_pre_eclampsia'] += len(pre_eclampsia.loc[pre_eclampsia])

            if not pre_eclampsia.loc[pre_eclampsia].empty:
                logger.debug(key='message', data=f'The following women have developed pre_eclampsia '
                                                 f'{pre_eclampsia.loc[pre_eclampsia].index}')

            #  -------------------------------- RISK OF GESTATIONAL HYPERTENSION --------------------------------------
            gest_hypertension = self.apply_linear_model(
                params['ps_linear_equations']['gest_htn'],
                df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == ga_in_weeks)
                       & (df['ps_htn_disorders'] == 'none') & ~df['ps_ectopic_pregnancy'] & ~df['ac_inpatient']
                       & ~df['la_currently_in_labour']])

            df.loc[gest_hypertension.loc[gest_hypertension].index, 'ps_htn_disorders'] = 'gest_htn'
            self.pregnancy_disease_tracker['new_onset_gest_htn'] += len(gest_hypertension.loc[gest_hypertension])
            if not gest_hypertension.loc[gest_hypertension].empty:
                logger.debug(key='message', data=f'The following women have developed gestational hypertension'
                                                 f'{gest_hypertension.loc[gest_hypertension].index}')

            #  ---------------------------------- RISK OF GESTATIONAL DIABETES ----------------------------------------
            gest_diab = self.apply_linear_model(
                params['ps_linear_equations']['gest_diab'],
                df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == ga_in_weeks) &
                       ~df['ps_gest_diab'] & ~df['ps_ectopic_pregnancy'] & ~df['ac_inpatient'] &
                       ~df['la_currently_in_labour'] ])

            df.loc[gest_diab.loc[gest_diab].index, 'ps_gest_diab'] = True
            df.loc[gest_diab.loc[gest_diab].index, 'ps_prev_gest_diab'] = True
            self.pregnancy_disease_tracker['new_onset_gest_diab'] += len(gest_diab.loc[gest_diab])

            if not gest_diab.loc[gest_diab].empty:
                logger.debug(key='message', data=f'The following women have developed gestational diabetes,'
                                                 f'{gest_diab.loc[gest_diab].index}')

            # ---------------------------------- RISK OF DISEASE PROGRESSION -----------------------------------------
            if ga_in_weeks != 22:
                # We first define the possible states that can be moved between
                selected = df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == ga_in_weeks) & \
                            (df.ps_htn_disorders != 'none') & ~df.la_currently_in_labour & ~df.ac_inpatient

                disease_states = ['gest_htn', 'severe_gest_htn', 'mild_pre_eclamp', 'severe_pre_eclamp', 'eclampsia']
                prob_matrix = pd.DataFrame(columns=disease_states, index=disease_states)

                # TODO: these should be parameters
                # Probability of moving between states is stored in a matrix
                prob_matrix['gest_htn'] = [0.8, 0.1, 0.1, 0.0, 0.0]
                prob_matrix['severe_gest_htn'] = [0.0, 0.8, 0.0, 0.2, 0.0]
                prob_matrix['mild_pre_eclamp'] = [0.0, 0.0, 0.8, 0.2, 0.0]
                prob_matrix['severe_pre_eclamp'] = [0.0, 0.0, 0.0, 0.6, 0.4]
                prob_matrix['eclampsia'] = [0.0, 0.0, 0.0, 0.0, 1]

                # todo: I think eventually these values would need to be manipulated by treatment effects?
                #  (or will things like calcium only effect onset of pre-eclampsia not progression)
                # todo: record new cases of spe/ec

                # We update the dataframe with transitioned states (which may not have changed)
                current_status = df.loc[selected, "ps_htn_disorders"]
                new_status = util.transition_states(current_status, prob_matrix, self.rng)
                df.loc[selected, "ps_htn_disorders"] = new_status

                # We evaluate the series of women in this function and select the women who have transitioned to severe
                # pre-eclampsia
                assess_status_change_for_severe_pre_eclampsia = (current_status != "severe_pre_eclamp") & \
                                                                (new_status == "severe_pre_eclamp")
                new_onset_severe_pre_eclampsia = assess_status_change_for_severe_pre_eclampsia[
                    assess_status_change_for_severe_pre_eclampsia]

                # For these women we schedule them to an onset event where they may seek care
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

            # ---------------------------------- RISK OF PLACENTAL ABRUPTION -----------------------------------------
            placenta_abruption = self.apply_linear_model(
                params['ps_linear_equations']['placental_abruption'],
                df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == ga_in_weeks) &
                       ~df['ps_placental_abruption'] & ~df['ps_ectopic_pregnancy'] & ~df['ac_inpatient'] &
                       ~df['la_currently_in_labour']])

            df.loc[placenta_abruption.loc[placenta_abruption].index, 'ps_placental_abruption'] = True
            self.pregnancy_disease_tracker['placental_abruption'] += len(placenta_abruption.loc[placenta_abruption])

            if not placenta_abruption.loc[placenta_abruption].empty:
                logger.debug(key='message', data=f'The following women have developed placental abruption,'
                                                 f'{placenta_abruption.loc[placenta_abruption].index}')
            # todo: this should cause some kind of emergency care seeking/response?

            # ---------------------------------- RISK OF ANTEPARTUM HAEMORRHAGE --------------------------------------
            antepartum_haemorrhage = self.apply_linear_model(
                params['ps_linear_equations']['antepartum_haem'],
                df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == ga_in_weeks) &
                       ~df['ps_ectopic_pregnancy'] & ~df['ac_inpatient'] & ~df['la_currently_in_labour'] &
                       ~df['ps_antepartum_haemorrhage']])

            random_choice_severity = pd.Series(self.rng.choice(['mild_moderate', 'severe'], p=[0.5, 0.5],
                                               size=len(antepartum_haemorrhage.loc[antepartum_haemorrhage])),
                                               index=antepartum_haemorrhage.loc[antepartum_haemorrhage].index)

            df.loc[antepartum_haemorrhage.loc[antepartum_haemorrhage].index, 'ps_antepartum_haemorrhage'] = True
            df.loc[antepartum_haemorrhage.loc[antepartum_haemorrhage].index, 'ps_antepartum_haemorrhage_severity'] = \
                random_choice_severity
            df.loc[placenta_abruption.loc[placenta_abruption].index, 'ps_emergency_event'] = True

            if not antepartum_haemorrhage.loc[antepartum_haemorrhage].empty:
                logger.debug(key='message', data=f'The following women are experiencing an antepartum haemorrhage,'
                                                 f'{antepartum_haemorrhage.loc[antepartum_haemorrhage].index}')

            self.pregnancy_disease_tracker['antepartum_haem'] += len(antepartum_haemorrhage.loc[antepartum_haemorrhage])

            # ------------------------------- RISK OF PREMATURE RUPTURE OF MEMBRANES ----------------------------------
            prom = self.apply_linear_model(
                params['ps_linear_equations']['premature_rupture_of_membranes'],
                df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == ga_in_weeks) &
                       ~df['ps_ectopic_pregnancy'] & ~df['ac_inpatient'] & ~df['la_currently_in_labour']])

            df.loc[prom.loc[prom].index, 'ps_premature_rupture_of_membranes'] = True
            df.loc[prom.loc[prom].index, 'ps_emergency_event'] = True
            self.pregnancy_disease_tracker['prom'] += len(prom.loc[prom])

            if not prom.loc[prom].empty:
                logger.debug(key='message', data=f'The following women have experience premature rupture of membranes'
                                                 f'{prom.loc[prom].index}')

            # TODO: apply risk of chorioamnionitis
            # -------------------------------------- RISK OF PRE TERM LABOUR ------------------------------------------
            if ga_in_weeks != 40:
                preterm_labour = self.apply_linear_model(
                    params['ps_linear_equations']['early_onset_labour'],
                    df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == ga_in_weeks)
                           & ~df['ps_ectopic_pregnancy'] & ~df['ac_inpatient'] & ~df['la_currently_in_labour']])

                if not preterm_labour.loc[preterm_labour].empty:
                    logger.debug(key='message',
                                 data=f'The following women will go into preterm labour at some point before the '
                                 f'next month of their pregnancy: {preterm_labour.loc[preterm_labour].index}')
                self.pregnancy_disease_tracker['pre_term'] += len(preterm_labour.loc[preterm_labour])

                for person in preterm_labour.loc[preterm_labour].index:
                    if df.at[person, 'ps_gestational_age_in_weeks'] == 22:
                        poss_day_onset = (27 - 22) * 7
                        # We only allow labour to onset from 24 weeks
                        onset_day = self.rng.randint(14, poss_day_onset)
                    elif df.at[person, 'ps_gestational_age_in_weeks'] == 27:
                        poss_day_onset = (31 - 27) * 7
                        onset_day = self.rng.randint(0, poss_day_onset)
                    elif df.at[person, 'ps_gestational_age_in_weeks'] == 31:
                        poss_day_onset = (35 - 31) * 7
                        onset_day = self.rng.randint(0, poss_day_onset)
                    elif df.at[person, 'ps_gestational_age_in_weeks'] == 35:
                        poss_day_onset = (37 - 35) * 7
                        onset_day = self.rng.randint(0, poss_day_onset)

                    df.at[person, 'la_due_date_current_pregnancy'] = self.sim.date + DateOffset(days=onset_day)
                    due_date = df.at[person, 'la_due_date_current_pregnancy']

                    logger.debug(key='message', data=f'Mother {person} will go into preterm labour on '
                                                     f'{self.sim.date + DateOffset(days=onset_day)}')

                    self.sim.schedule_event(LabourOnsetEvent(self.sim.modules['Labour'], person),
                                            due_date)

            # ---------------------------------- CARE SEEKING FOR COMPLICATIONS  --------------------------------------
            care_seeking = self.apply_linear_model(
                params['ps_linear_equations']['care_seeking_pregnancy_complication'],
                df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == ga_in_weeks) &
                       ~df['ps_ectopic_pregnancy'] & df['ps_emergency_event'] & ~df['ac_inpatient'] &
                       ~df['la_currently_in_labour'] & (df['la_due_date_current_pregnancy'] != self.sim.date)])

            df.loc[care_seeking.index, 'ps_emergency_event'] = False

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

                # todo: whats the best way to apply to the data frame and avoid a for loop? (to allow for multiple
                #  possible causes of death?)

            for person in care_seeking.loc[~care_seeking].index:
                df.at[person, 'ps_emergency_event'] = False
                mother = df.loc[person]

                antenatal_death = False
                still_birth = False

                if mother.ps_antepartum_haemorrhage:
                    risk_of_death = params['ps_linear_equations']['antepartum_haemorrhage_death'].predict(
                        df.loc[[person]])[person]

                    if self.rng.random_sample() < risk_of_death:
                        antenatal_death = True
                    else:
                        risk_of_still_birth = params['ps_linear_equations'][
                            'antepartum_haemorrhage_stillbirth'].predict(df.loc[[person]])[person]
                        if self.rng.random_sample() < risk_of_still_birth:
                            still_birth = True

                        df.at[person, 'ps_antepartum_haemorrhage'] = False
                        df.at[person, 'ps_antepartum_haemorrhage_severity'] = 'none'

                if mother.ps_htn_disorders == 'eclampsia':
                    risk_of_death = params['ps_linear_equations']['eclampsia_death'].predict(
                        df.loc[[person]])[person]

                    if self.rng.random_sample() < risk_of_death:
                        antenatal_death = True
                    else:
                        risk_of_still_birth = params['ps_linear_equations'][
                            'eclampsia_still_birth'].predict(df.loc[[person]])[person]
                        if self.rng.random_sample() < risk_of_still_birth:
                            still_birth = True

                        df.at[person, 'ps_htn_disorders'] = 'severe_pre_eclamp'

                if antenatal_death:
                    logger.debug(key='message',
                                 data=f'mother {person} has died following a pregnancy emergency on '
                                 f'date {self.sim.date}')

                    self.sim.schedule_event(demography.InstantaneousDeath(self, person,
                                                                          cause='maternal'), self.sim.date)
                    self.pregnancy_disease_tracker['antenatal_death'] += 1

                elif still_birth:
                    logger.debug(key='message',
                                 data=f'person {person} has experience an antepartum stillbirth on '
                                 f'date {self.sim.date}')

                    self.pregnancy_disease_tracker['antenatal_stillbirth'] += 1

                    df.at[person, 'ps_antepartum_still_birth'] = True
                    df.at[person, 'ps_previous_stillbirth'] = True
                    df.at[person, 'is_pregnant'] = False
                    df.at[person, 'la_due_date_current_pregnancy'] = pd.NaT
                    df.at[person, 'ps_gestational_age_in_weeks'] = 0

            # TODO: apply risk of infection to prom? (then if infection another round of care seeking/death)

        # ======================================= WEEKS 41-45 OF PREGNANCY ============================================

        # ------------------------------------------ RISK OF STILLBIRTH  ---------------------------------------------
        elif ga_in_weeks >= 41:
            still_birth = self.apply_linear_model(
                params['ps_linear_equations']['antenatal_stillbirth'],
                df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == ga_in_weeks) &
                       ~df['ps_ectopic_pregnancy'] & ~df['ac_inpatient'] & ~df['la_currently_in_labour']])

            self.pregnancy_disease_tracker['antenatal_stillbirth'] += len(still_birth.loc[still_birth])

            df.loc[still_birth.loc[still_birth].index, 'ps_antepartum_still_birth'] = True
            df.loc[still_birth.loc[still_birth].index, 'ps_previous_stillbirth'] = True
            df.loc[still_birth.loc[still_birth].index, 'is_pregnant'] = False
            df.loc[still_birth.loc[still_birth].index, 'la_due_date_current_pregnancy'] = pd.NaT
            df.loc[still_birth.loc[still_birth].index, 'ps_gestational_age_in_weeks'] = 0

            if not still_birth.loc[still_birth].empty:
                logger.debug(key='message', data=f'The following women have have experienced an antepartum'
                                                 f' stillbirth,{still_birth.loc[still_birth]}')

        # TODO: monthly risk of death??

    def abortion(self, individual_id, cause):
        """This function makes changes to the dataframe for women who have experienced induced or spontaneous abortion.
        Additionally it determines if a woman will develop complications associated with pregnancy loss and schedules
        the relevant death events"""
        df = self.sim.population.props
        params = self.parameters

        if df.at[individual_id, 'is_alive']:
            # Women who have an abortion have key pregnancy variables reset
            df.at[individual_id, 'is_pregnant'] = False
            df.at[individual_id, 'la_due_date_current_pregnancy'] = pd.NaT
            df.at[individual_id, 'ac_total_anc_visits_current_pregnancy'] = 0

            # We store the type of abortion for analysis
            self.pregnancy_disease_tracker[f'{cause}'] += 1

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

    def care_seeking_pregnancy_loss_complications(self, individual_id):
        df = self.sim.population.props
        params = self.parameters

        if self.rng.random_sample() < \
            params['ps_linear_equations']['care_seeking_pregnancy_loss'].predict(
                df.loc[[individual_id]])[individual_id]:
            logger.debug(key='message', data=f'Mother {individual_id} will seek care following pregnancy loss')

            from tlo.methods.hsi_generic_first_appts import (
                HSI_GenericEmergencyFirstApptAtFacilityLevel1)

            event = HSI_GenericEmergencyFirstApptAtFacilityLevel1(
                module=self,
                person_id=individual_id)

            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        else:
            logger.debug(key='message', data=f'Mother {individual_id} will not seek care following pregnancy loss')

    def property_reset(self, individual_id):
        df = self.sim.population.props

        df.at[individual_id, 'ps_anaemia_in_pregnancy'] = 'none'
        df.at[individual_id, 'ps_htn_disorders'] = 'none'
        self.deficiencies_in_pregnancy.unset(individual_id, 'iron', 'folate', 'b12')


class PregnancySupervisorEvent(RegularEvent, PopulationScopeEventMixin):
    """ This is the PregnancySupervisorEvent. It runs weekly. It updates gestational age of pregnancy in weeks.
    Presently this event has been hollowed out, additionally it will and uses set_pregnancy_complications function to
    determine if women will experience complication.. """

    def __init__(self, module, ):
        super().__init__(module, frequency=DateOffset(weeks=1))

    def apply(self, population):
        df = population.props
        params = self.module.parameters

        # ============================== UPDATING GESTATIONAL AGE IN WEEKS  ==========================================
        # Here we update the gestational age in weeks of all currently pregnant women in the simulation
        alive_and_preg = df.is_alive & df.is_pregnant
        gestation_in_days = self.sim.date - df.loc[alive_and_preg, 'date_of_last_pregnancy']
        gestation_in_weeks = gestation_in_days / np.timedelta64(1, 'W')

        df.loc[alive_and_preg, 'ps_gestational_age_in_weeks'] = gestation_in_weeks.astype('int64')
        logger.debug(key='message', data=f'updating gestational ages on date {self.sim.date}')

        # ---------------------- APPLYING RISK AND OUTCOMES DURING PREGNANCY -----------------------------------------
        for ga_in_weeks in [1, 4, 8, 13, 17, 22, 27, 31, 35, 40, 41, 42, 43, 44, 45]:
            self.module.set_pregnancy_complications(ga_in_weeks=ga_in_weeks)


class EctopicPregnancyEvent(Event, IndividualScopeEventMixin):
    """This is EctopicPregnancyEvent. It is scheduled by the PregnancySupervisorEvent. This event makes changes to the
    data frame for women with ectopic pregnancies, applies a probability of careseeking and schedules the
    EctopicRuptureEvent. This event is unfinished"""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters

        # Check only the right women have arrived here
        assert df.at[individual_id, 'ps_ectopic_pregnancy']

        # reset pregnancy variables
        if df.at[individual_id, 'is_alive']:
            df.at[individual_id, 'is_pregnant'] = False
            df.at[individual_id, 'ps_gestational_age_in_weeks'] = 0
            df.at[individual_id, 'la_due_date_current_pregnancy'] = pd.NaT

            self.module.care_seeking_pregnancy_loss_complications(individual_id)

        # TODO: The rupture event should only be scheduled with unsuccessful care seeking/ failed/incorrect treatment

        # As there is no treatment in the model currently, all women will eventually experience a rupture and may die
        self.sim.schedule_event(EctopicPregnancyRuptureEvent(self.module, individual_id),
                                (self.sim.date + pd.Timedelta(days=7 * 4 + self.module.rng.randint(0, 7 * 2))))

        # TODO: Currently only ruptured ectopics pass through the death event, is that ok?


class EctopicPregnancyRuptureEvent(Event, IndividualScopeEventMixin):
    """This is EctopicPregnancyRuptureEvent. It is scheduled by the EctopicPregnancyEvent for women who have
    experienced an ectopic pregnancy which has ruptured due to lack of treatment. This event manages care seeking and
    schedules EarlyPregnancyLossDeathEvent"""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props

        # Check the right woman has arrived at this event
        assert df.at[individual_id, 'ps_ectopic_pregnancy']
        # assert self.sim.date - df.at[individual_id, 'la_due_date_current_pregnancy'] < pd.Timedelta(43, unit='d')

        if df.at[individual_id, 'is_alive']:
            logger.debug(key='message', data=f'persons {individual_id} untreated ectopic pregnancy has now ruptured on '
                                             f'date {self.sim.date}')

            self.module.care_seeking_pregnancy_loss_complications(individual_id)

        # We delayed the death event by three days to allow any treatment effects to mitigate risk of death
        self.sim.schedule_event(EarlyPregnancyLossDeathEvent(self.module, individual_id, cause='ectopic_pregnancy'),
                                self.sim.date + DateOffset(days=3))


class EarlyPregnancyLossDeathEvent(Event, IndividualScopeEventMixin):
    """This is EarlyPregnancyLossDeathEvent. It is scheduled by the EctopicPregnancyRuptureEvent & AbortionEvent for
    women who are at risk of death following a loss of their pregnancy. Currently this event applies a risk of death,
    it is unfinished."""

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

        self.cause = cause

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        pac_interventions = self.sim.modules['CareOfWomenDuringPregnancy'].pac_interventions

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

            # Here we remove treatments from a woman who has survived her post abortion complications
            # todo:remove
            elif (self.cause == 'induced_abortion' or self.cause == 'spontaneous_abortion') and \
                                                   df.at[individual_id, 'ac_post_abortion_care_interventions'] > 0:
                pac_interventions.unset(individual_id, 'mva', 'd_and_c', 'misoprostol', 'analgesia', 'antibiotics',
                                        'blood_products')
                u = pac_interventions.uncompress()
                assert (u.loc[individual_id] == [False, False, False, False, False, False]).all()



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
