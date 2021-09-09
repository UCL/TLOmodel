from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging, util, Date
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata, labour
from tlo.methods.causes import Cause
from tlo.util import BitsetHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

        # First we define dictionaries which will store the current parameters of interest (to allow parameters to
        # change between 2010 and 2020) and the linear models
        self.current_parameters = dict()
        self.ps_linear_models = dict()

        # Here we define the mother and newborn information dictionary which stored surplus information about women
        # across the length of pregnancy and the postnatal period
        self.mother_and_newborn_info = dict()

        # This variable will store a Bitset handler for the property ps_deficiencies_in_pregnancy
        self.deficiencies_in_pregnancy = None

        # This variable will store a Bitset handler for the property ps_abortion_complications
        self.abortion_complications = None

    INIT_DEPENDENCIES = {'Demography'}
    ADDITIONAL_DEPENDENCIES = {
        'CareOfWomenDuringPregnancy', 'Contraception', 'Labour', 'HealthSystem', 'Lifestyle', 'Hiv', 'Malaria',
        'CardioMetabolicDisorders'
    }

    METADATA = {Metadata.DISEASE_MODULE,
                Metadata.USES_HEALTHBURDEN}

    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        'ectopic_pregnancy': Cause(gbd_causes='Maternal disorders', label='Maternal Disorders'),
        'spontaneous_abortion': Cause(gbd_causes='Maternal disorders', label='Maternal Disorders'),
        'induced_abortion': Cause(gbd_causes='Maternal disorders', label='Maternal Disorders'),
        'antepartum_haemorrhage': Cause(gbd_causes='Maternal disorders', label='Maternal Disorders'),
        'severe_gestational_hypertension': Cause(gbd_causes='Maternal disorders', label='Maternal Disorders'),
        'severe_pre_eclampsia': Cause(gbd_causes='Maternal disorders', label='Maternal Disorders'),
        'eclampsia': Cause(gbd_causes='Maternal disorders', label='Maternal Disorders'),
        'antenatal_sepsis': Cause(gbd_causes='Maternal disorders', label='Maternal Disorders')}

    # Declare Causes of Disability
    CAUSES_OF_DISABILITY = {
        'maternal': Cause(gbd_causes='Maternal disorders', label='Maternal Disorders')
    }

    PARAMETERS = {
        # ECTOPIC PREGNANCY...
        'prob_ectopic_pregnancy': Parameter(
            Types.LIST, 'probability of ectopic pregnancy'),
        'prob_care_seeking_ectopic_pre_rupture': Parameter(
            Types.LIST, 'probability a woman will seek care for ectopic pregnancy prior to rupture'),
        'prob_ectopic_pregnancy_death': Parameter(
            Types.LIST, 'probability of a woman dying from a ruptured ectopic pregnancy'),

        # TWINS...
        'prob_multiples': Parameter(
            Types.LIST, 'probability that a woman is currently carrying more than one pregnancy'),

        # PLACENTA PRAEVIA
        'prob_placenta_praevia': Parameter(
            Types.LIST, 'probability that this womans pregnancy will be complicated by placenta praevia'),
        'rr_placenta_praevia_previous_cs': Parameter(
            Types.LIST, 'relative risk of placenta praevia in a woman who has previously delivered via caesarean '
                        'section'),

        # SYPHILIS
        'prob_syphilis_during_pregnancy': Parameter(
            Types.LIST, 'probability that this womans will develop syphilis during her pregnancy'),

        # SPONTANEOUS AND INDUCED ABORTION
        'prob_previous_miscarriage_at_baseline': Parameter(
            Types.LIST, 'probability that a woman at baseline will have previously experienced a miscarriage'),
        'prob_spontaneous_abortion_per_month': Parameter(
            Types.LIST, 'underlying risk of spontaneous abortion per month'),
        'rr_spont_abortion_age_35': Parameter(
            Types.LIST, 'relative risk of spontaneous abortion in women aged 35 years or older'),
        'rr_spont_abortion_age_31_34': Parameter(
            Types.LIST, 'relative risk of spontaneous abortion in women aged 31-34 years old'),
        'rr_spont_abortion_prev_sa': Parameter(
            Types.LIST, 'relative risk of spontaneous abortion in women who have previously experiences spontaneous '
                        'abortion'),
        'prob_complicated_sa': Parameter(
            Types.LIST, 'probability that a woman who experiences spontaneous abortion with experience any '
                        'complications'),
        'prob_induced_abortion_per_month': Parameter(
            Types.LIST, 'underlying risk of induced abortion per month'),
        'prob_complicated_ia': Parameter(
            Types.LIST, 'probability that a woman who experiences induced abortion with experience any '
                        'complications'),
        'prob_haemorrhage_post_abortion': Parameter(
            Types.LIST, 'probability of haemorrhage following an abortion'),
        'prob_sepsis_post_abortion': Parameter(
            Types.LIST, 'probability of sepsis following an abortion'),
        'prob_injury_post_abortion': Parameter(
            Types.LIST, 'probability of injury following an abortion'),
        'prob_induced_abortion_death': Parameter(
            Types.LIST, 'underlying risk of death following an induced abortion'),
        'prob_spontaneous_abortion_death': Parameter(
            Types.LIST, 'underlying risk of death following an spontaneous abortion'),

        # NUTRIENT DEFICIENCIES...
        'prob_iron_def_per_month': Parameter(
            Types.LIST, 'monthly risk of a pregnant woman becoming iron deficient'),
        'prob_folate_def_per_month': Parameter(
            Types.LIST, 'monthly risk of a pregnant woman becoming folate deficient'),
        'prob_b12_def_per_month': Parameter(
            Types.LIST, 'monthly risk of a pregnant woman becoming b12 deficient'),

        # ANAEMIA...
        'baseline_prob_anaemia_per_month': Parameter(
            Types.LIST, 'baseline risk of a woman developing anaemia secondary only to pregnant'),
        'rr_anaemia_if_iron_deficient': Parameter(
            Types.LIST, 'relative risk of a woman developing anaemia in pregnancy if she is iron deficient'),
        'rr_anaemia_if_folate_deficient': Parameter(
            Types.LIST, 'relative risk of a woman developing anaemia in pregnancy if she is folate deficient'),
        'rr_anaemia_if_b12_deficient': Parameter(
            Types.LIST, 'relative risk of a woman developing anaemia in pregnancy if she is b12 deficient'),
        'rr_anaemia_maternal_malaria': Parameter(
            Types.LIST, 'relative risk of anaemia secondary to malaria infection'),
        'rr_anaemia_hiv_no_art': Parameter(
            Types.LIST, 'relative risk of anaemia for a woman with HIV not on ART'),
        'prob_mild_mod_sev_anaemia': Parameter(
            Types.LIST, 'probabilities that a womans anaemia will be mild, moderate or severe'),

        # GESTATIONAL DIABETES...
        'prob_gest_diab_per_month': Parameter(
            Types.LIST, 'underlying risk of gestational diabetes per month without the impact of risk factors'),
        'rr_gest_diab_obesity': Parameter(
            Types.LIST, 'Relative risk of gestational diabetes for women who are obese'),

        # HYPERTENSIVE DISORDERS...
        'prob_gest_htn_per_month': Parameter(
            Types.LIST, 'underlying risk of gestational hypertension per month without the impact of risk factors'),
        'rr_gest_htn_obesity': Parameter(
            Types.LIST, 'Relative risk of gestational hypertension for women who are obese'),
        'prob_pre_eclampsia_per_month': Parameter(
            Types.LIST, 'underlying risk of pre-eclampsia per month without the impact of risk factors'),
        'rr_pre_eclampsia_obesity': Parameter(
            Types.LIST, 'Relative risk of pre-eclampsia for women who are obese'),
        'rr_pre_eclampsia_multiple_pregnancy': Parameter(
            Types.LIST, 'Relative risk of pre-eclampsia for women who are pregnant with twins'),
        'rr_pre_eclampsia_chronic_htn': Parameter(
            Types.LIST, 'Relative risk of pre-eclampsia in women who are chronically hypertensive'),
        'rr_pre_eclampsia_diabetes_mellitus': Parameter(
            Types.LIST, 'Relative risk of pre-eclampsia in women who have diabetes mellitus'),
        'probs_for_mgh_matrix': Parameter(
            Types.LIST, 'probability of mild gestational hypertension moving between states: gestational '
                        'hypertension, severe gestational hypertension, mild pre-eclampsia, severe pre-eclampsia, '
                        'eclampsia'),
        'probs_for_sgh_matrix': Parameter(
            Types.LIST, 'probability of severe gestational hypertension moving between states: gestational '
                        'hypertension, severe gestational hypertension, mild pre-eclampsia, severe pre-eclampsia, '
                        'eclampsia'),
        'probs_for_mpe_matrix': Parameter(
            Types.LIST, 'probability of mild pre-eclampsia moving between states: gestational hypertension,'
                        ' severe gestational hypertension, mild pre-eclampsia, severe pre-eclampsia, eclampsia'),
        'probs_for_spe_matrix': Parameter(
            Types.LIST, 'probability of severe pre-eclampsia moving between states: gestational hypertension,'
                        ' severe gestational hypertension, mild pre-eclampsia, severe pre-eclampsia, eclampsia'),
        'probs_for_ec_matrix': Parameter(
            Types.LIST, 'probability of eclampsia moving between states: gestational hypertension,'
                        ' severe gestational hypertension, mild pre-eclampsia, severe pre-eclampsia, eclampsia'),
        'prob_severe_pre_eclampsia_death': Parameter(
            Types.LIST, 'probability of death for a woman experiencing acute severe pre-eclampsia'),
        'prob_eclampsia_death': Parameter(
            Types.LIST, 'probability of death for a woman experiencing eclampsia'),
        'prob_monthly_death_severe_htn': Parameter(
            Types.LIST, 'monthly risk of death for a woman with severe hypertension'),

        # PLACENTAL ABRUPTION...
        'prob_placental_abruption_per_month': Parameter(
            Types.LIST, 'monthly probability that a woman will develop placental abruption'),
        'rr_placental_abruption_hypertension': Parameter(
            Types.LIST, 'Relative risk of placental abruption in women with hypertension'),
        'rr_placental_abruption_previous_cs': Parameter(
            Types.LIST, 'Relative risk of placental abruption in women who delivered previously via caesarean section'),

        # ANTEPARTUM HAEMORRHAGE...
        'prob_aph_placenta_praevia': Parameter(
            Types.LIST, 'risk of antepartum haemorrhage due to ongoing placenta praevia'),
        'prob_aph_placental_abruption': Parameter(
            Types.LIST, 'risk of antepartum haemorrhage due to placental abruption'),
        'prob_mod_sev_aph': Parameter(
            Types.LIST, 'probabilities that APH is mild/moderate or severe'),
        'prob_antepartum_haemorrhage_death': Parameter(
            Types.LIST, 'probability of death for a woman suffering acute antepartum haemorrhage'),

        # PROM...
        'prob_prom_per_month': Parameter(
            Types.LIST, 'monthly probability that a woman will experience premature rupture of membranes'),

        # CHORIOAMNIONITIS...
        'prob_chorioamnionitis': Parameter(
            Types.LIST, 'monthly probability of a women developing chorioamnionitis'),
        'prob_antenatal_sepsis_death': Parameter(
            Types.LIST, 'case fatality rate for chorioamnionitis'),

        # PRETERM LABOUR...
        'baseline_prob_early_labour_onset': Parameter(
            Types.LIST, 'monthly baseline risk of labour onsetting before term'),
        'rr_preterm_labour_post_prom': Parameter(
            Types.LIST, 'relative risk of early labour onset following PROM'),
        'rr_preterm_labour_anaemia': Parameter(
            Types.LIST, 'relative risk of early labour onset in women with anaemia'),
        'rr_preterm_labour_malaria': Parameter(
            Types.LIST, 'relative risk of early labour onset in women with malaria'),
        'rr_preterm_labour_multiple_pregnancy': Parameter(
            Types.LIST, 'relative risk of early labour onset in women pregnant with twins'),

        # ANTENATAL STILLBIRTH
        'prob_still_birth_per_month': Parameter(
            Types.LIST, 'underlying risk of stillbirth per month without the impact of risk factors'),
        'rr_still_birth_ga_41': Parameter(
            Types.LIST, 'relative risk of still birth in women with gestational age 41 weeks'),
        'rr_still_birth_ga_42': Parameter(
            Types.LIST, 'relative risk of still birth in women with gestational age 42 weeks'),
        'rr_still_birth_ga_>42': Parameter(
            Types.LIST, 'relative risk of still birth in women with gestational age > 42 weeks'),

        'rr_still_birth_gest_diab': Parameter(
            Types.LIST, 'relative risk of still birth in women with gestational diabetes'),
        'rr_still_birth_diab_mellitus': Parameter(
            Types.LIST, 'relative risk of still birth in women with diabetes mellitus'),
        'rr_still_birth_maternal_malaria': Parameter(
            Types.LIST, 'relative risk of still birth in women with malaria'),
        'rr_still_birth_maternal_syphilis': Parameter(
            Types.LIST, 'relative risk of still birth in women with syphilis'),
        'rr_still_birth_pre_eclampsia': Parameter(
            Types.LIST, 'relative risk of still birth in women with pre-eclampsia'),
        'rr_still_birth_eclampsia': Parameter(
            Types.LIST, 'relative risk of still birth in women with eclampsia'),
        'rr_still_birth_gest_htn': Parameter(
            Types.LIST, 'relative risk of still birth in women with mild gestational hypertension'),
        'rr_still_birth_chronic_htn': Parameter(
            Types.LIST, 'relative risk of still birth in women with chronic hypertension'),
        'rr_still_birth_aph': Parameter(
            Types.LIST, 'relative risk of still birth in women with antepartum haemorrhage'),
        'rr_still_birth_chorio': Parameter(
            Types.LIST, 'relative risk of still birth in women with chorioamnionitis'),


        # CARE SEEKING (NOT ANC)...
        'prob_seek_care_pregnancy_complication': Parameter(
            Types.LIST, 'Probability that a woman who is pregnant will seek care in the event of a complication'),
        'prob_seek_care_pregnancy_loss': Parameter(
            Types.LIST, 'Probability that a woman who has developed complications post pregnancy loss will seek care'),
        'prob_seek_care_induction': Parameter(
            Types.LIST, 'Probability that a woman who is post term will seek care for induction of labour'),

        # CARE SEEKING (ANC)...
        'prob_anc1_months_1_to_4': Parameter(
            Types.LIST, 'list of probabilities that a woman will attend her first ANC visit at either month 1, 2, 3 or'
                        ' 4 of pregnancy'),
        'prob_anc1_months_5_to_9': Parameter(
            Types.LIST, 'list of probabilities that a woman will attend her first ANC visit on months 5-10'),
        'odds_early_init_anc4': Parameter(
            Types.LIST, 'probability of a woman undergoing 4 or more basic ANC visits with the first visit occurring '
                        'prior or during month 4 of pregnancy (EANC4+)'),
        'aor_early_anc4_20_24': Parameter(
            Types.LIST, 'adjusted odds ratio of EANC4+ in women aged 20-24'),
        'aor_early_anc4_25_29': Parameter(
            Types.LIST, 'adjusted odds ratio of EANC4+ in women aged 25-29'),
        'aor_early_anc4_30_34': Parameter(
            Types.LIST, 'adjusted odds ratio of EANC4+ in women aged 30-34'),
        'aor_early_anc4_35_39': Parameter(
            Types.LIST, 'adjusted odds ratio of EANC4+ in women aged 35-39'),
        'aor_early_anc4_40_44': Parameter(
            Types.LIST, 'adjusted odds ratio of EANC4+ in women aged 40-44'),
        'aor_early_anc4_45_49': Parameter(
            Types.LIST, 'adjusted odds ratio of EANC4+ in women aged 45-49'),
        'aor_early_anc4_2010': Parameter(
            Types.LIST, 'adjusted odds ratio of EANC4+ in 2010'),
        'aor_early_anc4_2015': Parameter(
            Types.LIST, 'adjusted odds ratio of EANC4+ in 2015'),
        'aor_early_anc4_parity_2_3': Parameter(
            Types.LIST, 'adjusted odds ratio of EANC4+ in women with a parity of 2-3'),
        'aor_early_anc4_parity_4_5': Parameter(
            Types.LIST, 'adjusted odds ratio of EANC4+ in women with a parity of 4-5'),
        'aor_early_anc4_parity_6+': Parameter(
            Types.LIST, 'adjusted odds ratio of EANC4+ in women with a parity of 6+'),
        'aor_early_anc4_primary_edu': Parameter(
            Types.LIST, 'adjusted odds ratio of EANC4+ in women with primary education'),
        'aor_early_anc4_secondary_edu': Parameter(
            Types.LIST, 'adjusted odds ratio of EANC4+ in women with secondary education'),
        'aor_early_anc4_tertiary_edu': Parameter(
            Types.LIST, 'adjusted odds ratio of EANC4+ in women with tertiary education'),
        'aor_early_anc4_middle_wealth': Parameter(
            Types.LIST, 'adjusted odds ratio of EANC4+ in women in the middle wealth quintile'),
        'aor_early_anc4_richer_wealth': Parameter(
            Types.LIST, 'adjusted odds ratio of EANC4+ in women in the richer wealth quintile'),
        'aor_early_anc4_richest_wealth': Parameter(
            Types.LIST, 'adjusted odds ratio of EANC4+ in women in the richest wealth quintile'),
        'aor_early_anc4_married': Parameter(
            Types.LIST, 'adjusted odds ratio of EANC4+ in women who are married'),
        'aor_early_anc4_previously_married': Parameter(
            Types.LIST, 'adjusted odds ratio of EANC4+ in women who were previously married (divorced/widowed)'),
        'prob_late_initiation_anc4': Parameter(
            Types.LIST, 'probability a woman will undertake 4 or more ANC visits with the first being after 4 months'),
        'prob_early_initiation_anc_below4': Parameter(
            Types.LIST, 'probabilities a woman will attend fewer than 4 ANC visits but the first visit will occur '
                        'before month 4'),
        'prob_early_anc_at_facility_level_1_2': Parameter(
            Types.LIST, 'probabilities a woman will attend ANC 1 at facility levels 1 or 2'),

        # TREATMENT EFFECTS...
        'treatment_effect_ectopic_pregnancy_treatment': Parameter(
            Types.LIST, 'Treatment effect of ectopic pregnancy case management'),
        'treatment_effect_post_abortion_care': Parameter(
            Types.LIST, 'Treatment effect of post abortion care'),
        'treatment_effect_iron_def_ifa': Parameter(
            Types.LIST, 'treatment effect of iron supplementation on iron deficiency '),
        'treatment_effect_folate_def_ifa': Parameter(
            Types.LIST, 'treatment effect of folate supplementation on folate deficiency'),
        'treatment_effect_iron_folic_acid_anaemia': Parameter(
            Types.LIST, 'relative effect of daily iron and folic acid treatment on risk of maternal anaemia '),
        'treatment_effect_calcium_pre_eclamp': Parameter(
            Types.LIST, 'risk reduction of pre-eclampsia for women taking daily calcium supplementation'),
        'treatment_effect_gest_htn_calcium': Parameter(
            Types.LIST, 'Effect of calcium supplementation on risk of developing gestational hypertension'),
        'treatment_effect_anti_htns_progression': Parameter(
            Types.LIST, 'Effect of anti hypertensive medication in reducing the risk of progression from mild to severe'
                        ' hypertension'),
        'prob_glycaemic_control_diet_exercise': Parameter(
            Types.LIST, 'probability a womans GDM is controlled by diet and exercise during the first month of '
                        'treatment'),
        'prob_glycaemic_control_orals': Parameter(
            Types.LIST, 'probability a womans GDM is controlled by oral anti-diabetics during the first month of '
                        'treatment'),
        'prob_glycaemic_control_insulin': Parameter(
            Types.LIST, 'probability a womans GDM is controlled by insulin during the first month of '
                        'treatment'),
        'treatment_effect_gdm_case_management': Parameter(
            Types.LIST, 'Treatment effect of GDM case management on mothers risk of stillbirth '),
        'treatment_effect_still_birth_food_sups': Parameter(
            Types.LIST, 'risk reduction of still birth for women receiving nutritional supplements'),

    }

    PROPERTIES = {
        'ps_gestational_age_in_weeks': Property(Types.REAL, 'current gestational age, in weeks, of a womans '
                                                            'pregnancy'),
        'ps_date_of_anc1': Property(Types.DATE, 'Date first ANC visit is scheduled for'),
        'ps_ectopic_pregnancy': Property(Types.CATEGORICAL, 'Whether a womans is experiencing ectopic pregnancy and'
                                                            ' its current state',
                                         categories=['none', 'not_ruptured', 'ruptured']),
        'ps_multiple_pregnancy': Property(Types.BOOL, 'Whether a womans is pregnant with multiple fetuses'),
        'ps_placenta_praevia': Property(Types.BOOL, 'Whether a womans pregnancy will be complicated by placenta'
                                                    'praevia'),
        'ps_syphilis': Property(Types.BOOL, 'Whether a womans has syphilis during pregnancy'),
        'ps_deficiencies_in_pregnancy': Property(Types.INT, 'bitset column, stores types of anaemia causing '
                                                            'deficiencies in pregnancy'),
        'ps_anaemia_in_pregnancy': Property(Types.CATEGORICAL, 'Whether a woman has anaemia in pregnancy and its '
                                                               'severity',
                                            categories=['none', 'mild', 'moderate', 'severe']),

        'ps_anc4': Property(Types.BOOL, 'Whether this womans is predicted to attend 4 or more antenatal care visits '
                                        'during her pregnancy'),
        'ps_abortion_complications': Property(Types.INT, 'Bitset column holding types of abortion complication'),
        'ps_prev_spont_abortion': Property(Types.BOOL, 'Whether this woman has had any previous pregnancies end in '
                                                       'spontaneous abortion'),
        'ps_prev_stillbirth': Property(Types.BOOL, 'Whether this woman has had any previous pregnancies end in '
                                                   'still birth'),
        'ps_htn_disorders': Property(Types.CATEGORICAL, 'if this woman suffers from a hypertensive disorder of '
                                                        'pregnancy',
                                     categories=['none', 'gest_htn', 'severe_gest_htn', 'mild_pre_eclamp',
                                                 'severe_pre_eclamp', 'eclampsia']),
        'ps_prev_pre_eclamp': Property(Types.BOOL, 'whether this woman has experienced pre-eclampsia in a previous '
                                                   'pregnancy'),
        'ps_gest_diab': Property(Types.CATEGORICAL, 'whether this woman is experiencing gestational diabetes',
                                 categories=['none', 'uncontrolled', 'controlled']),
        'ps_prev_gest_diab': Property(Types.BOOL, 'whether this woman has ever suffered from gestational diabetes '
                                                  'during a previous pregnancy'),
        'ps_placental_abruption': Property(Types.BOOL, 'Whether this woman is experiencing placental abruption'),
        'ps_antepartum_haemorrhage': Property(Types.CATEGORICAL, 'severity of this womans antepartum haemorrhage',
                                              categories=['none', 'mild_moderate', 'severe']),
        'ps_premature_rupture_of_membranes': Property(Types.BOOL, 'whether this woman has experience rupture of '
                                                                  'membranes before the onset of labour. If this is '
                                                                  '<37 weeks from gestation the woman has preterm '
                                                                  'premature rupture of membranes'),
        'ps_chorioamnionitis': Property(Types.BOOL, 'Whether a womans is experiencing chorioamnionitis'),
        'ps_emergency_event': Property(Types.BOOL, 'signifies a woman in undergoing an acute emergency event in her '
                                                   'pregnancy- used to consolidated care seeking in the instance of '
                                                   'multiple complications')
    }

    def read_parameters(self, data_folder):

        parameter_dataframe = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_PregnancySupervisor.xlsx',
                                            sheet_name='parameter_values')
        self.load_parameters_from_dataframe(parameter_dataframe)

        # For the first period (2010-2015) we use the first value in each list as a parameter
        for key, value in self.parameters.items():
            self.current_parameters[key] = self.parameters[key][0]

        # Here we map 'disability' parameters to associated DALY weights to be passed to the health burden module.
        # Currently this module calculates and reports all DALY weights from all maternal modules
        if 'HealthBurden' in self.sim.modules.keys():
            self.parameters['ps_daly_weights'] = \
                {'abortion': self.sim.modules['HealthBurden'].get_daly_weight(352),
                 'abortion_haem': self.sim.modules['HealthBurden'].get_daly_weight(339),
                 'abortion_sep': self.sim.modules['HealthBurden'].get_daly_weight(340),
                 'ectopic': self.sim.modules['HealthBurden'].get_daly_weight(351),
                 'ectopic_rupture': self.sim.modules['HealthBurden'].get_daly_weight(338),
                 'mild_mod_aph': self.sim.modules['HealthBurden'].get_daly_weight(339),
                 'severe_aph': self.sim.modules['HealthBurden'].get_daly_weight(338),
                 'chorio': self.sim.modules['HealthBurden'].get_daly_weight(340),
                 'mild_anaemia': self.sim.modules['HealthBurden'].get_daly_weight(476),
                 'mild_anaemia_pp': self.sim.modules['HealthBurden'].get_daly_weight(476),
                 'moderate_anaemia': self.sim.modules['HealthBurden'].get_daly_weight(480),
                 'moderate_anaemia_pp': self.sim.modules['HealthBurden'].get_daly_weight(478),
                 'severe_anaemia': self.sim.modules['HealthBurden'].get_daly_weight(478),
                 'severe_anaemia_pp': self.sim.modules['HealthBurden'].get_daly_weight(478),
                 'eclampsia': self.sim.modules['HealthBurden'].get_daly_weight(861),
                 'hypertension': self.sim.modules['HealthBurden'].get_daly_weight(343),
                 'gest_diab': self.sim.modules['HealthBurden'].get_daly_weight(971),
                 'obstructed_labour': self.sim.modules['HealthBurden'].get_daly_weight(348),
                 'uterine_rupture': self.sim.modules['HealthBurden'].get_daly_weight(338),
                 'sepsis': self.sim.modules['HealthBurden'].get_daly_weight(340),
                 'mild_mod_pph': self.sim.modules['HealthBurden'].get_daly_weight(339),
                 'severe_pph': self.sim.modules['HealthBurden'].get_daly_weight(338),
                 'secondary_pph': self.sim.modules['HealthBurden'].get_daly_weight(339),
                 'vesicovaginal_fistula': self.sim.modules['HealthBurden'].get_daly_weight(349),
                 'rectovaginal_fistula': self.sim.modules['HealthBurden'].get_daly_weight(350),
                 }

        # Finally we generate a local variable containing only the parameters for 2010-2015 (this is overridden at 2015)

    def initialise_population(self, population):

        df = population.props

        df.loc[df.is_alive, 'ps_gestational_age_in_weeks'] = 0
        df.loc[df.is_alive, 'ps_date_of_anc1'] = pd.NaT
        df.loc[df.is_alive, 'ps_ectopic_pregnancy'] = 'none'
        df.loc[df.is_alive, 'ps_placenta_praevia'] = False
        df.loc[df.is_alive, 'ps_multiple_pregnancy'] = False
        df.loc[df.is_alive, 'ps_syphilis'] = False
        df.loc[df.is_alive, 'ps_deficiencies_in_pregnancy'] = 0
        df.loc[df.is_alive, 'ps_anaemia_in_pregnancy'] = 'none'
        df.loc[df.is_alive, 'ps_anc4'] = False
        df.loc[df.is_alive, 'ps_abortion_complications'] = 0
        df.loc[df.is_alive, 'ps_prev_spont_abortion'] = False
        df.loc[df.is_alive, 'ps_prev_stillbirth'] = False
        df.loc[df.is_alive, 'ps_htn_disorders'] = 'none'
        df.loc[df.is_alive, 'ps_prev_pre_eclamp'] = False
        df.loc[df.is_alive, 'ps_gest_diab'] = 'none'
        df.loc[df.is_alive, 'ps_prev_gest_diab'] = False
        df.loc[df.is_alive, 'ps_placental_abruption'] = False
        df.loc[df.is_alive, 'ps_antepartum_haemorrhage'] = 'none'
        df.loc[df.is_alive, 'ps_premature_rupture_of_membranes'] = False
        df.loc[df.is_alive, 'ps_chorioamnionitis'] = False
        df.loc[df.is_alive, 'ps_emergency_event'] = False

        # This bitset property stores nutritional deficiencies that can occur in the antenatal period
        self.deficiencies_in_pregnancy = BitsetHandler(self.sim.population, 'ps_deficiencies_in_pregnancy',
                                                       ['iron', 'b12', 'folate'])

        # This bitset property stores 'types' of complication that can occur after an abortion
        self.abortion_complications = BitsetHandler(self.sim.population, 'ps_abortion_complications',
                                                    ['sepsis', 'haemorrhage', 'injury', 'other'])

        # Here we set properties in the population that effect the rates of complications in the baseline year
        reproductive_age_women = df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)

        previous_miscarriage = pd.Series(
            self.rng.random_sample(len(reproductive_age_women.loc[reproductive_age_women])) <
            self.current_parameters['prob_previous_miscarriage_at_baseline'],
            index=reproductive_age_women.loc[reproductive_age_women].index)

        df.loc[previous_miscarriage.loc[previous_miscarriage].index, 'ps_prev_spont_abortion'] = True

    def initialise_simulation(self, sim):

        # Register and schedule the PregnancySupervisorEvent
        sim.schedule_event(PregnancySupervisorEvent(self),
                           sim.date + DateOffset(days=0))

        # Register and schedule logging event
        sim.schedule_event(PregnancyLoggingEvent(self),
                           sim.date + DateOffset(years=1))

        # Register and schedule the parameter update event
        sim.schedule_event(ParameterUpdateEvent(self),
                           Date(2015, 1, 1))

        # ==================================== LINEAR MODEL EQUATIONS =================================================
        # Here we scale linear models according to distribution of predictors in the dataframe at baseline
        params = self.current_parameters

        # First we define the target intercept values for each model and store as dictionaries
        target_intercepts_other = {'placenta_praevia': params['prob_placenta_praevia'],
                                   'maternal_anaemia': params['baseline_prob_anaemia_per_month'],
                                   'gest_diab': params['prob_gest_diab_per_month'],
                                   'gest_htn': params['prob_gest_htn_per_month'],
                                   'pre_eclampsia': params['prob_pre_eclampsia_per_month'],
                                   'placental_abruption': params['prob_placental_abruption_per_month'],
                                   'antenatal_stillbirth': params['prob_still_birth_per_month'],
                                   'early_initiation_anc4': params['odds_early_init_anc4']}

        target_intercepts_sa = {'ga_4': params['prob_spontaneous_abortion_per_month'][0],
                                'ga_8': params['prob_spontaneous_abortion_per_month'][1],
                                'ga_13': params['prob_spontaneous_abortion_per_month'][2],
                                'ga_17': params['prob_spontaneous_abortion_per_month'][3],
                                'ga_22': params['prob_spontaneous_abortion_per_month'][4]}

        target_intercepts_ptl = {'ga_22': params['baseline_prob_early_labour_onset'][0],
                                 'ga_27': params['baseline_prob_early_labour_onset'][1],
                                 'ga_31': params['baseline_prob_early_labour_onset'][2],
                                 'ga_35': params['baseline_prob_early_labour_onset'][3]}

        # Then create dictionaries storing some non scaled intercept - i.e. 1, next to each model
        unscaled_intercepts = dict()
        sa_unscaled_intercepts = dict()
        ptl_unscaled_intercepts = dict()

        for key in target_intercepts_other.keys():
            unscaled_intercepts.update({key: 1.0})
        for key in target_intercepts_sa.keys():
            sa_unscaled_intercepts.update({key: 1.0})
        for key in target_intercepts_ptl.keys():
            ptl_unscaled_intercepts.update({key: 1.0})

        # Define functions that generate linear models with a given dictionary of intercept valyues
        def make_linear_models_standard_intercepts(intercept_dict):
            """
            Creates linear models with provided intercepts
            :param intercept_list: intercepts for models (list)
            :return: linear_models dict.
            """
            linear_models_standard_intercepts = {
                # This equation calculates a womans risk of placenta praevia (placenta partially/completely covers the
                # cervix). This risk is applied once per pregnancy
                'placenta_praevia': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    intercept_dict['placenta_praevia'],
                    Predictor('la_previous_cs_delivery').when(True, params['rr_placenta_praevia_previous_cs'])),

                # This equation calculates a womans monthly risk of developing anaemia during her pregnancy. This is
                # currently influenced by nutritional deficiencies and malaria status
                'maternal_anaemia': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    intercept_dict['maternal_anaemia'],
                    Predictor('ps_deficiencies_in_pregnancy').apply(
                        lambda x: params['rr_anaemia_if_iron_deficient']
                        if x & self.deficiencies_in_pregnancy.element_repr('iron') else 1),
                    Predictor('ps_deficiencies_in_pregnancy').apply(
                        lambda x: params['rr_anaemia_if_folate_deficient']
                        if x & self.deficiencies_in_pregnancy.element_repr('folate') else 1),
                    Predictor('ps_deficiencies_in_pregnancy').apply(
                        lambda x: params['rr_anaemia_if_b12_deficient']
                        if x & self.deficiencies_in_pregnancy.element_repr('b12') else 1),
                    Predictor('ma_is_infected').when(True, params['rr_anaemia_maternal_malaria']),
                    Predictor().when('hv_inf & (hv_art != "not")', params['rr_anaemia_hiv_no_art']),
                    Predictor('ac_receiving_iron_folic_acid').when(True, params['treatment_effect_iron_folic_acid_'
                                                                                'anaemia'])),

                # This equation calculates a womans monthly risk of developing gestational diabetes
                # during her pregnancy.This is currently influenced by obesity
                'gest_diab': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    intercept_dict['gest_diab'],
                    Predictor('li_bmi', conditions_are_mutually_exclusive=True)
                    .when('4', params['rr_gest_diab_obesity'])
                    .when('5', params['rr_gest_diab_obesity'])),

                # This equation calculates a womans monthly risk of developing gestational hypertension
                # during her pregnancy. This is currently influenced receipt of calcium supplementation
                'gest_htn': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    intercept_dict['gest_htn'],
                    Predictor('li_bmi', conditions_are_mutually_exclusive=True)
                    .when('4', params['rr_gest_htn_obesity'])
                    .when('5', params['rr_gest_htn_obesity']),
                    Predictor('ac_receiving_calcium_supplements').when(True, params['treatment_effect_gest_htn_'
                                                                                    'calcium'])),

                # This equation calculates a womans monthly risk of developing pre-eclampsia during her pregnancy.
                # This is
                # currently influenced by receipt of calcium supplementation
                'pre_eclampsia': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    intercept_dict['pre_eclampsia'],
                    Predictor('li_bmi', conditions_are_mutually_exclusive=True)
                    .when('4', params['rr_pre_eclampsia_obesity'])
                    .when('5', params['rr_pre_eclampsia_obesity']),
                    Predictor('ps_multiple_pregnancy').when(True, params['rr_pre_eclampsia_multiple_pregnancy']),
                    Predictor('nc_hypertension').when(True, params['rr_pre_eclampsia_chronic_htn']),
                    Predictor('nc_diabetes').when(True, params['rr_pre_eclampsia_diabetes_mellitus']),
                    Predictor('ac_receiving_calcium_supplements').when(True, params['treatment_effect_calcium_pre_'
                                                                                    'eclamp'])),

                # This equation calculates a womans monthly risk of developing placental abruption
                # during her pregnancy.
                'placental_abruption': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    intercept_dict['placental_abruption'],
                    Predictor('la_previous_cs_delivery').when(True, params['rr_placental_abruption_previous_cs']),
                    Predictor('ps_htn_disorders', conditions_are_mutually_exclusive=True)
                    .when('mild_pre_eclamp', params['rr_placental_abruption_hypertension'])
                    .when('gest_htn', params['rr_placental_abruption_hypertension'])
                    .when('severe_gest_htn', params['rr_placental_abruption_hypertension'])
                    .when('severe_pre_eclamp', params['rr_placental_abruption_hypertension'])),


                # This equation calculates a womans monthly risk of antenatal still birth
                'antenatal_stillbirth': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    intercept_dict['antenatal_stillbirth'],
                    Predictor('ps_gestational_age_in_weeks').when('41', params['rr_still_birth_ga_41']),
                    Predictor('ps_gestational_age_in_weeks').when('42', params['rr_still_birth_ga_42']),
                    Predictor('ps_gestational_age_in_weeks').when('>42', params['rr_still_birth_ga_>42']),
                    Predictor('ps_htn_disorders', conditions_are_mutually_exclusive=True)
                        .when('mild_pre_eclamp', params['rr_still_birth_pre_eclampsia'])
                        .when('gest_htn', params['rr_still_birth_gest_htn'])
                        .when('severe_gest_htn', params['rr_still_birth_gest_htn'])
                        .when('severe_pre_eclamp', params['rr_still_birth_pre_eclampsia']),
                    Predictor('ps_antepartum_haemorrhage').when('!= "none"', params['rr_still_birth_aph']),
                    Predictor('ps_chorioamnionitis').when(True, params['rr_still_birth_chorio']),
                    Predictor('nc_hypertension').when(True, params['rr_still_birth_chronic_htn']),
                    Predictor('ps_gest_diab').when('uncontrolled', params['rr_still_birth_gest_diab']),
                    Predictor().when('(ps_gest_diab == "controlled ") & (ac_gest_diab_on_treatment != "none")',
                                     params['rr_still_birth_gest_diab'] * params['treatment_effect_gdm_case_'
                                                                                 'management']),
                    Predictor('ma_is_infected').when(True, params['rr_still_birth_maternal_malaria']),
                    Predictor('nc_diabetes').when(True, params['rr_still_birth_diab_mellitus']),
                    Predictor('ps_syphilis').when(True, params['rr_still_birth_maternal_syphilis'])),

                # This equation calculates a the probability a woman will attend at least 4 ANC contacts during her
                # pregnancy - derived from Wingstons analysis of DHS data
                'early_initiation_anc4': LinearModel(
                    LinearModelType.LOGISTIC,
                    intercept_dict['early_initiation_anc4'],
                    Predictor('age_years').when('.between(19,25)', params['aor_early_anc4_20_24'])
                                          .when('.between(24,30)', params['aor_early_anc4_25_29'])
                                          .when('.between(29,35)', params['aor_early_anc4_30_34'])
                                          .when('.between(34,40)', params['aor_early_anc4_35_39'])
                                          .when('.between(39,45)', params['aor_early_anc4_40_44'])
                                          .when('.between(44,50)', params['aor_early_anc4_45_49']),
                    # TODO: this is effecting the output somehow?
                    Predictor('year', external=True).when('<2015', params['aor_early_anc4_2010'])
                                                    .when('>2014', params['aor_early_anc4_2015']),
                    Predictor('la_parity').when('.between(1,4)', params['aor_early_anc4_parity_2_3'])
                                          .when('.between(3,6)', params['aor_early_anc4_parity_4_5'])
                                          .when('>5', params['aor_early_anc4_parity_6+']),
                    Predictor('li_ed_lev').when('2', params['aor_early_anc4_primary_edu'])
                                          .when('3', params['aor_early_anc4_secondary_edu']),
                    Predictor('li_wealth').when('1', params['aor_early_anc4_richest_wealth'])
                                          .when('2', params['aor_early_anc4_richer_wealth'])
                                          .when('3', params['aor_early_anc4_middle_wealth']),
                    Predictor('li_mar_stat').when('2', params['aor_early_anc4_married'])
                                            .when('3', params['aor_early_anc4_previously_married']))}

            return linear_models_standard_intercepts

        # As some models use predictors as a proxy intercept they are defined separately
        def make_spontaneous_abortion_linear_model(intercept_dict):
            spontaneous_abortion_linear_model = {
             # This equation calculates a womans monthly risk of spontaneous abortion (miscarriage) and is applied
             # monthly until 28 weeks gestation
             'spontaneous_abortion': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    1,
                    Predictor('ps_gestational_age_in_weeks').when(4, intercept_dict['ga_4'])
                                                            .when(8, intercept_dict['ga_8'])
                                                            .when(13, intercept_dict['ga_13'])
                                                            .when(17, intercept_dict['ga_17'])
                                                            .when(22, intercept_dict['ga_22']),
                    Predictor('ps_prev_spont_abortion').when(True, params['rr_spont_abortion_prev_sa']),
                    Predictor('age_years').when('>34', params['rr_spont_abortion_age_35'])
                                          .when('.between(30,35)', params['rr_spont_abortion_age_31_34']))}

            return spontaneous_abortion_linear_model

        def make_ptl_linear_model(intercept_dict):
            ptl_model = {
                'early_onset_labour': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    1,
                    Predictor('ps_gestational_age_in_weeks').when(22, intercept_dict['ga_22'])
                                                            .when(27, intercept_dict['ga_27'])
                                                            .when(31, intercept_dict['ga_31'])
                                                            .when(35, intercept_dict['ga_35']),
                    Predictor('ps_premature_rupture_of_membranes').when(True, params['rr_preterm_labour_post_prom']),
                    Predictor().when('ps_anaemia_in_pregnancy != "none"', params['rr_preterm_labour_anaemia']),
                    Predictor('ma_is_infected').when(True, params['rr_preterm_labour_malaria']),
                    Predictor('ps_multiple_pregnancy').when(True, params['rr_preterm_labour_multiple_pregnancy']))
            }
            return ptl_model

        # Call the above functions to generate the unscaled models
        linear_models_standard_intercepts = make_linear_models_standard_intercepts(unscaled_intercepts)
        sa_lm = make_spontaneous_abortion_linear_model(sa_unscaled_intercepts)
        ptl_lm = make_ptl_linear_model(ptl_unscaled_intercepts)

        # Creat a function that returns a scaled intercept from the target intercept according to the distribution of
        # predictor variables within the dataframe
        def get_scaled_intercept(model, target_intercept):
            df = self.sim.population.props
            unscaled_lm = model
            target_mean = target_intercept
            actual_mean = unscaled_lm.predict(df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) &
                                                     (df.age_years < 50)], year=self.sim.date.year).mean()
            scaled_intercept = 1.0 * (target_mean / actual_mean) \
                if (target_mean != 0 and actual_mean != 0 and ~np.isnan(actual_mean)) else 1.0

            return scaled_intercept

        # Then we create the models with the newly scaled intercepts
        scaled_intercepts_sa_dict = dict()
        for k in target_intercepts_sa:
            scaled_intercepts_sa_dict.update({k: get_scaled_intercept(sa_lm['spontaneous_abortion'],
                                                                      target_intercepts_sa[k])})
        scaled_sa_model = make_spontaneous_abortion_linear_model(scaled_intercepts_sa_dict)

        scaled_intercepts_ptl_dict = dict()
        for k in target_intercepts_ptl:
            scaled_intercepts_ptl_dict.update({k: get_scaled_intercept(ptl_lm['early_onset_labour'],
                                                                       target_intercepts_ptl[k])})
        scaled_ptl_model = make_ptl_linear_model(scaled_intercepts_ptl_dict)

        scaled_intercepts_other_dict = dict()
        for model_name, model in zip(linear_models_standard_intercepts.keys(),
                                     linear_models_standard_intercepts.values()):
            scaled_intercepts_other_dict.update({model_name: get_scaled_intercept(model,
                                                                                  target_intercepts_other[model_name])})

        scaled_linear_models = make_linear_models_standard_intercepts(scaled_intercepts_other_dict)

        # Define any models that dont need to be scaled at baseline
        models_not_needing_scaling = {

            # This equation calculates a womans monthly risk of developing antepartum haemorrhage during her pregnancy.
            # APH can only occur in the presence of one of two preceding causes (placenta praevia and placental
            # abruption) hence the use of an additive model
            'antepartum_haem': LinearModel(
                LinearModelType.ADDITIVE,
                0,
                Predictor('ps_placenta_praevia').when(True, params['prob_aph_placenta_praevia']),
                Predictor('ps_placental_abruption').when(True, params['prob_aph_placental_abruption'])),


            # This equation calculates a risk of dying after ectopic pregnancy and is mitigated by treatment
            'ectopic_pregnancy_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_ectopic_pregnancy_death'],
                Predictor('ac_ectopic_pregnancy_treated').when(True, params['treatment_effect_ectopic_pregnancy_'
                                                                            'treatment'])),

            # This equation calculates a risk of dying after complications following an induced abortion. It is reduced
            # by treatment
            'induced_abortion_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_induced_abortion_death'],
                Predictor('ac_post_abortion_care_interventions').when('>0',
                                                                      params['treatment_effect_post_abortion_care'])),

            # This equation calculates a risk of dying after complications following a spontaneous abortion. It is
            # reduced by treatment
            'spontaneous_abortion_death': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_spontaneous_abortion_death'],
                Predictor('ac_post_abortion_care_interventions').when('>0',
                                                                      params['treatment_effect_post_abortion_care'])),
        }

        # And store all models within the same dictionary file
        for model_dict in [scaled_sa_model, scaled_ptl_model, scaled_linear_models, models_not_needing_scaling]:
            self.ps_linear_models.update(model_dict)

    def on_birth(self, mother_id, child_id):
        df = self.sim.population.props

        df.at[child_id, 'ps_gestational_age_in_weeks'] = 0
        df.at[child_id, 'ps_date_of_anc1'] = pd.NaT
        df.at[child_id, 'ps_ectopic_pregnancy'] = 'none'
        df.at[child_id, 'ps_placenta_praevia'] = False
        df.at[child_id, 'ps_multiple_pregnancy'] = False
        df.at[child_id, 'ps_syphilis'] = False
        df.at[child_id, 'ps_deficiencies_in_pregnancy'] = 0
        df.at[child_id, 'ps_anaemia_in_pregnancy'] = 'none'
        df.at[child_id, 'ps_anc4'] = False
        df.at[child_id, 'ps_abortion_complications'] = 0
        df.at[child_id, 'ps_prev_spont_abortion'] = False
        df.at[child_id, 'ps_prev_stillbirth'] = False
        df.at[child_id, 'ps_htn_disorders'] = 'none'
        df.at[child_id, 'ps_prev_pre_eclamp'] = False
        df.at[child_id, 'ps_gest_diab'] = 'none'
        df.at[child_id, 'ps_prev_gest_diab'] = False
        df.at[child_id, 'ps_placental_abruption'] = False
        df.at[child_id, 'ps_antepartum_haemorrhage'] = 'none'
        df.at[child_id, 'ps_premature_rupture_of_membranes'] = False
        df.at[child_id, 'ps_chorioamnionitis'] = False
        df.at[child_id, 'ps_emergency_event'] = False

    def further_on_birth_pregnancy_supervisor(self, mother_id):
        """
        This function is called by the on_birth function of NewbornOutcomes module. This function contains additional
        code related to the pregnancy supervisor module that should be ran on_birth for all births - it has been
        parcelled into functions to ensure each modules (pregnancy,antenatal care, labour, newborn, postnatal) on_birth
        code is ran in the correct sequence (as this can vary depending on how modules are registered)
        :param mother_id: mothers individual id
        """
        df = self.sim.population.props
        mni = self.mother_and_newborn_info

        if df.at[mother_id, 'is_alive']:

            # Check only the correct women arrive at this function
            assert not df.at[mother_id, 'la_intrapartum_still_birth']
            assert not df.at[mother_id, 'ps_multiple_pregnancy'] and (mni[mother_id]['twin_count'] == 1) and \
                   not mni[mother_id]['single_twin_still_birth']

            # We reset all womans gestational age when they deliver as they are no longer pregnant
            df.at[mother_id, 'ps_gestational_age_in_weeks'] = 0
            df.at[mother_id, 'ps_date_of_anc1'] = pd.NaT

            # And store her anaemia status to calculate the prevalence of anaemia on birth
            logger.info(key='anaemia_on_birth', data={'mother': mother_id,
                                                      'anaemia_status': df.at[mother_id, 'ps_anaemia_in_pregnancy']})

            # We currently assume that hyperglycemia due to gestational diabetes resolves following birth
            if df.at[mother_id, 'ps_gest_diab'] != 'none':
                df.at[mother_id, 'ps_gest_diab'] = 'none'

                # We store the date of resolution for women who were aware of their diabetes (as the DALY weight only
                # occurs after diagnosis)
                if not pd.isnull(mni[mother_id]['gest_diab_onset']):
                    self.store_dalys_in_mni(mother_id, 'gest_diab_resolution')

    def on_hsi_alert(self, person_id, treatment_id):
        logger.debug(key='message', data='This is PregnancySupervisor, being alerted about a health system interaction '
                                         f'person {person_id} for: {treatment_id}')

    def report_daly_values(self):
        df = self.sim.population.props
        p = self.parameters['ps_daly_weights']
        mni = self.mother_and_newborn_info

        logger.debug(key='message', data='This is PregnancySupervisor reporting my health values')
        monthly_daly = dict()

        # First we define a function that calculates disability associated with 'acute' complications of pregnancy
        def acute_daly_calculation(person, complication):
            # We cycle through each complication for all women in the mni, if the condition has never ocurred then we
            # pass
            if pd.isnull(mni[person][f'{complication}_onset']):
                return

            # If the complication has onset within the last month...
            elif (self.sim.date - DateOffset(months=1)) <= mni[person][f'{complication}_onset'] <= self.sim.date:

                # We assume that any woman who experiences an acute event receives the whole weight for that daly
                monthly_daly[person] += p[f'{complication}']

                # Ensure some weight is assigned
                if mni[person][f'{complication}_onset'] != self.sim.date:
                    assert monthly_daly[person] > 0

                mni[person][f'{complication}_onset'] = pd.NaT

        # Next we define a function that calculates disability associated with 'chronic' complications of pregnancy
        def chronic_daly_calculations(person, complication):
            if pd.isnull(mni[person][f'{complication}_onset']):
                return
            else:
                if pd.isnull(mni[person][f'{complication}_resolution']):

                    # If the complication has not yet resolved, and started more than a month ago, the woman gets a
                    # months disability
                    if mni[person][f'{complication}_onset'] < (self.sim.date - DateOffset(months=1)):
                        weight = (p[f'{complication}'] / 365.25) * (365.25 / 12)
                        monthly_daly[person] += weight

                    # Otherwise, if the complication started this month she gets a daly weight relative to the number of
                    # days she has experience the complication
                    elif (self.sim.date - DateOffset(months=1)) <= mni[person][
                         f'{complication}_onset'] <= self.sim.date:
                        days_since_onset = pd.Timedelta((self.sim.date - mni[person][f'{complication}_onset']),
                                                        unit='d')
                        daly_weight = days_since_onset.days * (p[f'{complication}'] / 365.25)

                        monthly_daly[person] += daly_weight
                        assert monthly_daly[person] >= 0

                else:
                    # Its possible for a condition to resolve (via treatment) and onset within the same month
                    # (i.e. anaemia). If so, here we calculate how many days this month an individual has suffered
                    if mni[person][f'{complication}_resolution'] < mni[person][f'{complication}_onset']:

                        if (mni[person][f'{complication}_resolution'] == (self.sim.date - DateOffset(months=1))) and \
                          (mni[person][f'{complication}_onset'] == self.sim.date):
                            return

                        else:
                            # Calculate daily weight and how many days this woman hasnt had the complication
                            daily_weight = p[f'{complication}'] / 365.25
                            days_without_complication = pd.Timedelta((
                                mni[person][f'{complication}_onset'] - mni[person][f'{complication}_resolution']),
                                unit='d')

                            # Use the average days in a month to calculate how many days shes had the complication this
                            # month
                            avg_days_in_month = 365.25 / 12
                            days_with_comp = avg_days_in_month - days_without_complication.days

                            monthly_daly[person] += daily_weight * days_with_comp

                            assert monthly_daly[person] >= 0
                            mni[person][f'{complication}_resolution'] = pd.NaT

                    else:
                        # If the complication has truly resolved, check the dates make sense
                        assert mni[person][f'{complication}_resolution'] >= mni[person][f'{complication}_onset']

                        # We calculate how many days she has been free of the complication this month to determine how
                        # many days she has suffered from the complication this month
                        days_free_of_comp_this_month = pd.Timedelta((self.sim.date - mni[person][f'{complication}_'
                                                                                                 f'resolution']),
                                                                    unit='d')
                        mid_way_calc = (self.sim.date - DateOffset(months=1)) + days_free_of_comp_this_month
                        days_with_comp_this_month = pd.Timedelta((self.sim.date - mid_way_calc), unit='d')
                        daly_weight = days_with_comp_this_month.days * (p[f'{complication}'] / 365.25)
                        monthly_daly[person] += daly_weight

                        assert monthly_daly[person] >= 0
                        # Reset the dates to stop additional disability being applied
                        mni[person][f'{complication}_onset'] = pd.NaT
                        mni[person][f'{complication}_resolution'] = pd.NaT

        # Then for each alive person in the MNI we cycle through all the complications that can lead to disability and
        # calculate their individual daly weight for the month
        for person in list(mni):
            if df.at[person, 'is_alive']:
                monthly_daly[person] = 0

                for complication in ['abortion', 'abortion_haem', 'abortion_sep', 'ectopic', 'ectopic_rupture',
                                     'mild_mod_aph', 'severe_aph', 'chorio', 'eclampsia', 'obstructed_labour',
                                     'sepsis', 'uterine_rupture',  'mild_mod_pph', 'severe_pph', 'secondary_pph']:
                    acute_daly_calculation(complication=complication, person=person)

                for complication in ['hypertension', 'gest_diab', 'mild_anaemia', 'moderate_anaemia',
                                     'severe_anaemia', 'mild_anaemia_pp', 'moderate_anaemia_pp', 'severe_anaemia_pp',
                                     'vesicovaginal_fistula', 'rectovaginal_fistula']:
                    chronic_daly_calculations(complication=complication, person=person)

                if monthly_daly[person] > 1:
                    monthly_daly[person] = 1

                if mni[person]['delete_mni'] and (df.at[person, 'is_pregnant'] or
                                                  (df.at[person, 'ps_ectopic_pregnancy'] != 'none')):
                    mni[person]['delete_mni'] = False

                elif mni[person]['delete_mni'] and not df.at[person, 'is_pregnant'] and (df.at[person,
                                                                                               'ps_ectopic_pregnancy']
                                                                                         == 'none'):
                    del mni[person]

        daly_series = pd.Series(data=0, index=df.index[df.is_alive])
        daly_series[monthly_daly.keys()] = list(monthly_daly.values())

        return daly_series

    def store_dalys_in_mni(self, individual_id, mni_variable):
        """
        This function is called across the maternal health modules and stores onset/resolution dates for complications
        in an indiviudals MNI dictionary
        :param individual_id: individual_id
        :param mni_variable: key of mni dict being assigned
        :return:
        """
        mni = self.mother_and_newborn_info

        assert individual_id in mni

        logger.debug(key='msg', data=f'{mni_variable} is being stored for mother {individual_id} on {self.sim.date}')
        mni[individual_id][f'{mni_variable}'] = self.sim.date

    def pregnancy_supervisor_property_reset(self, ind_or_df, id_or_index):
        """
        This function is called when properties housed in the PregnancySupervisorModule should be reset. For example
        following pregnancy loss
        :param ind_or_df: "individual/data_frame"(STR) whether this function has been called to reset properties for an
         individual row of the DF or a slice of the data frame
        :param id_or_index: pass the function either an individual ID (INT) or index of subset of data frame
        :return:
        """

        df = self.sim.population.props

        if ind_or_df == 'individual':
            set = df.at
            # todo: seems like this crashes when it shouldnt have even been called
            # assert not set[id_or_index, 'is_pregnant']

        else:
            set = df.loc
            assert not set[id_or_index, 'is_pregnant'].any()

        set[id_or_index, 'ps_gestational_age_in_weeks'] = 0
        set[id_or_index, 'ps_date_of_anc1'] = pd.NaT
        set[id_or_index, 'ps_multiple_pregnancy'] = False
        set[id_or_index, 'ps_placenta_praevia'] = False
        set[id_or_index, 'ps_syphilis'] = False
        set[id_or_index, 'ps_anaemia_in_pregnancy'] = 'none'
        set[id_or_index, 'ps_anc4'] = False
        set[id_or_index, 'ps_htn_disorders'] = 'none'
        set[id_or_index, 'ps_gest_diab'] = 'none'
        set[id_or_index, 'ps_placental_abruption'] = False
        set[id_or_index, 'ps_antepartum_haemorrhage'] = 'none'
        set[id_or_index, 'ps_premature_rupture_of_membranes'] = False
        set[id_or_index, 'ps_chorioamnionitis'] = False
        set[id_or_index, 'ps_emergency_event'] = False
        self.deficiencies_in_pregnancy.unset(id_or_index, 'iron')
        self.deficiencies_in_pregnancy.unset(id_or_index, 'folate')
        self.deficiencies_in_pregnancy.unset(id_or_index, 'b12')

    def apply_linear_model(self, lm, df_slice):
        """
        Helper function will apply the linear model (lm) on the dataframe (df) to get a probability of some event
        happening to each individual. It then returns a series with same index with bools indicating the outcome based
        on the toss of the biased coin.
        :param lm: The linear model
        :param df_slice: The dataframe
        :return: Series with same index containing outcomes (bool)
        """

        return self.rng.random_sample(len(df_slice)) < lm.predict(df_slice,
                                                                  year=self.sim.date.year)

    def schedule_anc_one(self, individual_id, anc_month):
        """
        This functions calculates the correct date each woman will attend ANC and schedules the first ANC visit for
        newly pregnant women depending on their predicted month of attendance
        :param anc_month: month of pregnancy that woman will attend ANC 1
        :param individual_id: individual_id
        """
        df = self.sim.population.props
        params = self.current_parameters

        # Define the weeks of each month of pregnancy
        months_min_max = {1: [0, 4], 2: [5, 8], 3: [9, 13], 4: [14, 17], 5: [18, 22],
                          6: [23, 27], 7: [28, 31], 8: [32, 35], 9: [36, 40]}

        # As care seeking is applied at week 3 gestational age, women who seek care within month one must attend within
        # the next week
        if anc_month == 1:
            days_until_anc = self.rng.randint(0, 7)

        else:
            # Otherwise we draw a week between the min max weeks for predicted month of visit, and then a random day
            weeks_of_visit = (self.rng.randint(months_min_max[anc_month][0], months_min_max[anc_month][1]) - 3)
            days_until_anc = (weeks_of_visit * 7) + self.rng.randint(0, 6)

        first_anc_date = self.sim.date + DateOffset(days=days_until_anc)

        # We store that date as a property which is used by the HSI to ensure the event only runs when it
        # should
        df.at[individual_id, 'ps_date_of_anc1'] = first_anc_date
        logger.debug(key='msg', data=f'{individual_id} will attend ANC 1 in {anc_month} months on '
                                     f'{first_anc_date}')

        # We used a weighted draw to decide what facility level this woman will seek care at, as ANC is offered
        # at multiple levels

        facility_level = int(self.rng.choice([1, 2], p=params['prob_early_anc_at_facility_level_1_2']))

        from tlo.methods.care_of_women_during_pregnancy import (
            HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareContact,
        )

        first_anc_appt = HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareContact(
            self.sim.modules['CareOfWomenDuringPregnancy'], person_id=individual_id,
            facility_level_of_this_hsi=facility_level)

        self.sim.modules['HealthSystem'].schedule_hsi_event(first_anc_appt, priority=0,
                                                            topen=first_anc_date,
                                                            tclose=first_anc_date + DateOffset(days=3))

    def apply_risk_of_spontaneous_abortion(self, gestation_of_interest):
        """
        This function applies risk of spontaneous abortion to a slice of data frame and is called by
        PregnancySupervisorEvent. It calls the apply_risk_of_abortion_complications function for women who loose their
        pregnancy.
        :param gestation_of_interest: gestation in weeks
        """
        df = self.sim.population.props

        # We use the apply_linear_model to determine if any women will experience spontaneous miscarriage
        spont_abortion = self.apply_linear_model(
            self.ps_linear_models['spontaneous_abortion'],
            df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == gestation_of_interest) &
                   (df['ps_ectopic_pregnancy'] == 'none') & ~df['hs_is_inpatient']])

        # The apply_risk_of_abortion_complications function is called for women who lose their pregnancy. It resets
        # properties, set complications and care seeking
        for person in spont_abortion.loc[spont_abortion].index:
            logger.info(key='maternal_complication', data={'person': person,
                                                           'type': 'spontaneous_abortion',
                                                           'timing': 'antenatal'})

            self.apply_risk_of_abortion_complications(person, 'spontaneous_abortion')

    def apply_risk_of_induced_abortion(self, gestation_of_interest):
        """
        This function applies risk of induced abortion to a slice of data frame and is called by
        PregnancySupervisorEvent. It calls the apply_risk_of_abortion_complications for women who loose their pregnancy.
        :param gestation_of_interest: gestation in weeks
        """
        df = self.sim.population.props
        params = self.current_parameters

        # TODO: risk of IA should be limited to women with an unintended pregnancy but currently the proportion of
        #  unintended pregnancies is too low to generate correct abortion rate. Discussed with TC.

        # This function follows the same pattern as apply_risk_of_spontaneous_abortion (only women with unintended
        # pregnancy may seek induced abortion)
        at_risk = df.is_alive & df.is_pregnant & (df.ps_gestational_age_in_weeks == gestation_of_interest) & \
                  (df.ps_ectopic_pregnancy == 'none') & ~df.hs_is_inpatient

        abortion = pd.Series(self.rng.random_sample(len(at_risk.loc[at_risk])) <
                             params['prob_induced_abortion_per_month'], index=at_risk.loc[at_risk].index)

        for person in abortion.loc[abortion].index:
            # Similarly the abortion function is called for each of these women
            logger.info(key='maternal_complication', data={'person': person,
                                                           'type': 'induced_abortion',
                                                           'timing': 'antenatal'})

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
        params = self.current_parameters

        # Women who have an abortion have key pregnancy variables reset
        df.at[individual_id, 'is_pregnant'] = False
        self.mother_and_newborn_info[individual_id]['delete_mni'] = True

        self.sim.modules['Labour'].reset_due_date(
            ind_or_df='individual', id_or_index=individual_id, new_due_date=pd.NaT)

        self.pregnancy_supervisor_property_reset(
            ind_or_df='individual', id_or_index=individual_id)

        self.sim.modules['CareOfWomenDuringPregnancy'].care_of_women_in_pregnancy_property_reset(
            ind_or_df='individual', id_or_index=individual_id)

        # Women who have spontaneous abortion are at higher risk of future spontaneous abortions, we store this
        # accordingly
        if cause == 'spontaneous_abortion':
            df.at[individual_id, 'ps_prev_spont_abortion'] = True

        complicated_sa = self.rng.random_sample() < params['prob_complicated_sa']
        complicated_ia = self.rng.random_sample() < params['prob_complicated_ia']

        # We apply a risk of developing specific complications associated with abortion type and store using a bitset
        # property
        if (cause == 'induced_abortion') and complicated_ia:
            if self.rng.random_sample() < params['prob_injury_post_abortion']:
                self.abortion_complications.set([individual_id], 'injury')
                logger.info(key='maternal_complication', data={'person': individual_id,
                                                               'type': f'{cause}_injury',
                                                               'timing': 'antenatal'})

        if ((cause == 'spontaneous_abortion') and complicated_sa) or ((cause == 'induced_abortion') and complicated_ia):
            if self.rng.random_sample() < params['prob_haemorrhage_post_abortion']:
                self.abortion_complications.set([individual_id], 'haemorrhage')
                self.store_dalys_in_mni(individual_id, 'abortion_haem_onset')
                logger.info(key='maternal_complication', data={'person': individual_id,
                                                               'type': f'{cause}_haemorrhage',
                                                               'timing': 'antenatal'})

            if self.rng.random_sample() < params['prob_sepsis_post_abortion']:
                self.abortion_complications.set([individual_id], 'sepsis')
                self.store_dalys_in_mni(individual_id, 'abortion_sep_onset')
                logger.info(key='maternal_complication', data={'person': individual_id,
                                                               'type': f'{cause}_sepsis',
                                                               'timing': 'antenatal'})

            if not self.abortion_complications.has_any([individual_id], 'sepsis', 'haemorrhage', 'injury', first=True):
                self.abortion_complications.set([individual_id], 'other')
                logger.info(key='maternal_complication', data={'person': individual_id,
                                                               'type': f'{cause}_other_comp',
                                                               'timing': 'antenatal'})

        # Then we determine if this woman will seek care, and schedule presentation to the health system
        if self.abortion_complications.has_any([individual_id], 'sepsis', 'haemorrhage', 'injury', 'other', first=True):

            logger.info(key='maternal_complication', data={'person': individual_id,
                                                           'type': f'complicated_{cause}',
                                                           'timing': 'antenatal'})

            # We assume only women with complicated abortions will experience disability
            self.store_dalys_in_mni(individual_id, 'abortion_onset')

            # Determine if those women will seek care
            self.care_seeking_pregnancy_loss_complications(individual_id, cause='abortion')

            # Schedule possible death
            self.sim.schedule_event(EarlyPregnancyLossDeathEvent(self, individual_id, cause=f'{cause}'),
                                    self.sim.date + DateOffset(days=7))

    def apply_risk_of_deficiencies_and_anaemia(self, gestation_of_interest):
        """
        This function applies risk of deficiencies and anaemia to a slice of the data frame. It is called by
        PregnancySupervisorEvent
        :param gestation_of_interest: gestation in weeks
        """
        df = self.sim.population.props
        params = self.current_parameters

        # This function iterates through the three key anaemia causing deficiencies (iron, folate
        # and b12) and determines a the risk of onset for a subset of pregnant women. Following this, woman have a
        # probability of anaemia calculated and relevant changes to the data frame occur

        def apply_risk(deficiency):

            if deficiency == 'iron' or deficiency == 'folate':
                # First we select a subset of the pregnant population who are not suffering from the deficiency in
                # question. (When applying risk of iron/folate deficiency we fist apply risk to women not on iron/folic
                # acid treatment)
                selected_women = ~self.deficiencies_in_pregnancy.has_all(
                    df.is_alive & df.is_pregnant & (df.ps_gestational_age_in_weeks == gestation_of_interest) &
                    ~df.hs_is_inpatient & ~df.la_currently_in_labour & ~df.ac_receiving_iron_folic_acid, deficiency)

            else:
                # As IFA treatment does not effect B12 we select the appropriate women regardless of IFA treatment
                # status
                selected_women = ~self.deficiencies_in_pregnancy.has_all(
                    df.is_alive & df.is_pregnant & (df.ps_gestational_age_in_weeks == gestation_of_interest)
                    & ~df.hs_is_inpatient & ~df.la_currently_in_labour, deficiency)

            # We determine their risk of deficiency
            new_def = pd.Series(self.rng.random_sample(len(selected_women)) < params[f'prob_{deficiency}_def_per'
                                                                                     f'_month'],
                                index=selected_women.index)

            # And change their property accordingly
            self.deficiencies_in_pregnancy.set(new_def.loc[new_def].index, deficiency)
            for person in new_def.loc[new_def].index:
                logger.info(key='maternal_complication', data={'person': person,
                                                               'type': f'{deficiency}_deficiency',
                                                               'timing': 'antenatal'})

            if deficiency == 'b12':
                return
            else:
                # Next we select women who aren't deficient of iron/folate but are receiving IFA treatment
                def_treatment = ~self.deficiencies_in_pregnancy.has_all(
                    df.is_alive & df.is_pregnant & (df.ps_gestational_age_in_weeks == gestation_of_interest)
                    & ~df.hs_is_inpatient & ~df.la_currently_in_labour & df.ac_receiving_iron_folic_acid, deficiency)

                # We reduce their individual risk of deficiencies due to treatment and make changes to the data frame
                risk_of_def = params[f'prob_{deficiency}_def_per_month'] * params[
                    f'treatment_effect_{deficiency}_def_ifa']

                new_def = pd.Series(self.rng.random_sample(len(def_treatment)) < risk_of_def, index=def_treatment.index)

                self.deficiencies_in_pregnancy.set(new_def.loc[new_def].index, deficiency)

        # Now we run the function for each
        for deficiency in ['iron', 'folate', 'b12']:
            apply_risk(deficiency)

        # ------------------------------------------ ANAEMIA ---------------------------------------------------------
        # Now we determine if a subset of pregnant women will become anaemic using a linear model, in which the
        # preceding deficiencies act as predictors
        anaemia = self.apply_linear_model(
            self.ps_linear_models['maternal_anaemia'],
            df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == gestation_of_interest) &
                   (df['ps_ectopic_pregnancy'] == 'none') & (df['ps_anaemia_in_pregnancy'] == 'none')
                   & ~df['hs_is_inpatient'] & ~df['la_currently_in_labour']])

        # We use a weight random draw to determine the severity of the anaemia
        random_choice_severity = pd.Series(self.rng.choice(['mild', 'moderate', 'severe'],
                                                           p=params['prob_mild_mod_sev_anaemia'],
                                                           size=len(anaemia.loc[anaemia])),
                                           index=anaemia.loc[anaemia].index)

        df.loc[anaemia.loc[anaemia].index, 'ps_anaemia_in_pregnancy'] = random_choice_severity

        for person in anaemia.loc[anaemia].index:
            # We store onset date of anaemia according to severity, as weights vary
            self.store_dalys_in_mni(person, f'{df.at[person, "ps_anaemia_in_pregnancy"]}_anaemia_onset')

            # todo: remove this logging if we only are bothered with prevalence at birth
            logger.info(key='maternal_complication', data={'person': person,
                                                           'type': f'{df.at[person, "ps_anaemia_in_pregnancy"]}_'
                                                                   f'anaemia',
                                                           'timing': 'antenatal'})

    def apply_risk_of_gestational_diabetes(self, gestation_of_interest):
        """
        This function applies risk of gestational diabetes to a slice of the data frame. It is called by
        PregnancySupervisorEvent.
        :param gestation_of_interest: gestation in weeks
        """
        df = self.sim.population.props

        gest_diab = self.apply_linear_model(
            self.ps_linear_models['gest_diab'], df.loc[df['is_alive'] & df['is_pregnant'] &
                                                       (df['ps_gestational_age_in_weeks'] == gestation_of_interest) &
                                                       (df['ps_gest_diab'] == 'none') &
                                                       (df['ps_ectopic_pregnancy'] == 'none') & ~df['hs_is_inpatient']
                                                       & ~df['la_currently_in_labour']])

        # Gestational diabetes, at onset, is defined as uncontrolled prior to treatment
        df.loc[gest_diab.loc[gest_diab].index, 'ps_gest_diab'] = 'uncontrolled'
        df.loc[gest_diab.loc[gest_diab].index, 'ps_prev_gest_diab'] = True

        for person in gest_diab.loc[gest_diab].index:
            logger.info(key='maternal_complication', data={'person': person,
                                                           'type': 'gest_diab',
                                                           'timing': 'antenatal'})

    def apply_risk_of_hypertensive_disorders(self, gestation_of_interest):
        """
        This function applies risk of mild pre-eclampsia and mild gestational diabetes to a slice of the data frame. It
        is called by PregnancySupervisorEvent.
        :param gestation_of_interest: gestation in weeks
        """
        df = self.sim.population.props

        #  ----------------------------------- RISK OF PRE-ECLAMPSIA ----------------------------------------------
        # We assume all women must developed a mild pre-eclampsia/gestational hypertension before progressing to a more
        # severe disease - we do not apply incidence of severe pre-eclampsia/eclampsia explicitly like this - see
        # PregnancySupervisorEvent)
        pre_eclampsia = self.apply_linear_model(
            self.ps_linear_models['pre_eclampsia'],
            df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == gestation_of_interest) &
                   (df['ps_htn_disorders'] == 'none') & (df['ps_ectopic_pregnancy'] == 'none') & ~df['hs_is_inpatient']
                   & ~df['la_currently_in_labour']])

        df.loc[pre_eclampsia.loc[pre_eclampsia].index, 'ps_prev_pre_eclamp'] = True
        df.loc[pre_eclampsia.loc[pre_eclampsia].index, 'ps_htn_disorders'] = 'mild_pre_eclamp'

        for person in pre_eclampsia.loc[pre_eclampsia].index:
            logger.info(key='maternal_complication', data={'person': person,
                                                           'type': 'mild_pre_eclamp',
                                                           'timing': 'antenatal'})

        #  -------------------------------- RISK OF GESTATIONAL HYPERTENSION --------------------------------------
        # For women who dont develop pre-eclampsia during this month, we apply a risk of gestational hypertension
        gest_hypertension = self.apply_linear_model(
            self.ps_linear_models['gest_htn'],
            df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == gestation_of_interest)
                   & (df['ps_htn_disorders'] == 'none') & (df['ps_ectopic_pregnancy'] == 'none')
                   & ~df['hs_is_inpatient'] & ~df['la_currently_in_labour']])

        df.loc[gest_hypertension.loc[gest_hypertension].index, 'ps_htn_disorders'] = 'gest_htn'

        for person in gest_hypertension.loc[gest_hypertension].index:
            logger.info(key='maternal_complication', data={'person': person,
                                                           'type': 'mild_gest_htn',
                                                           'timing': 'antenatal'})

    def apply_risk_of_progression_of_hypertension(self, gestation_of_interest):
        """
        This function applies a risk of progression of hypertensive disorders to women who are experiencing one of the
        hypertensive disorders. It is called by PregnancySupervisorEvent
        :param gestation_of_interest: gestation in weeks
        """
        df = self.sim.population.props
        params = self.current_parameters

        def apply_risk(selected, risk_of_gest_htn_progression):

            # Define the possible states that can be moved between
            disease_states = ['gest_htn', 'severe_gest_htn', 'mild_pre_eclamp', 'severe_pre_eclamp', 'eclampsia']
            prob_matrix = pd.DataFrame(columns=disease_states, index=disease_states)

            # Probability of moving between states is stored in a matrix. Risk of progression from mild gestational
            # hypertension to severe gestational hypertension is modified by treatment effect

            risk_ghtn_remains_mild = 1 - (risk_of_gest_htn_progression + params['probs_for_mgh_matrix'][2])

            # We reset the parameter here to allow for testing with the original parameter
            params['probs_for_mgh_matrix'] = [risk_ghtn_remains_mild, risk_of_gest_htn_progression,
                                              params['probs_for_mgh_matrix'][2], 0.0, 0.0]

            prob_matrix['gest_htn'] = params['probs_for_mgh_matrix']
            prob_matrix['severe_gest_htn'] = params['probs_for_sgh_matrix']
            prob_matrix['mild_pre_eclamp'] = params['probs_for_mpe_matrix']
            prob_matrix['severe_pre_eclamp'] = params['probs_for_spe_matrix']
            prob_matrix['eclampsia'] = params['probs_for_ec_matrix']

            # We update the data frame with transitioned states (which may not have changed)
            current_status = df.loc[selected, "ps_htn_disorders"]
            new_status = util.transition_states(current_status, prob_matrix, self.rng)
            df.loc[selected, "ps_htn_disorders"] = new_status

            def log_new_progressed_cases(disease):
                assess_status_change = (current_status != disease) & (new_status == disease)
                new_onset_disease = assess_status_change[assess_status_change]

                if not new_onset_disease.empty:
                    for person in new_onset_disease.index:
                        logger.info(key='maternal_complication', data={'person': person,
                                                                       'type': disease,
                                                                       'timing': 'antenatal'})

                    if disease == 'severe_pre_eclamp':
                        df.loc[new_onset_disease.index, 'ps_emergency_event'] = True
                    elif disease == 'eclampsia':
                        df.loc[new_onset_disease.index, 'ps_emergency_event'] = True
                        new_onset_disease.index.to_series().apply(self.store_dalys_in_mni,
                                                                  mni_variable='eclampsia_onset')

            for disease in ['mild_pre_eclamp', 'severe_pre_eclamp', 'eclampsia', 'severe_gest_htn']:
                log_new_progressed_cases(disease)

        # Here we select the women in the data frame who are at risk of progression.
        women_not_on_anti_htns = \
            df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == gestation_of_interest) & \
            (df.ps_htn_disorders.str.contains('gest_htn|mild_pre_eclamp|severe_gest_htn|severe_pre_eclamp')) \
            & ~df.la_currently_in_labour & ~df.hs_is_inpatient & ~df.ac_gest_htn_on_treatment

        women_on_anti_htns = \
            df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == gestation_of_interest) & \
            (df.ps_htn_disorders.str.contains('gest_htn|mild_pre_eclamp|severe_gest_htn|severe_pre_eclamp'))\
            & ~df.la_currently_in_labour & ~df.hs_is_inpatient & df.ac_gest_htn_on_treatment

        for v in women_not_on_anti_htns.loc[women_not_on_anti_htns].index:
            assert v not in women_on_anti_htns.loc[women_on_anti_htns].index

        risk_progression_mild_to_severe_htn = params['probs_for_mgh_matrix'][1]

        apply_risk(women_not_on_anti_htns, risk_progression_mild_to_severe_htn)
        apply_risk(women_on_anti_htns, (risk_progression_mild_to_severe_htn *
                                        params['treatment_effect_anti_htns_progression']))

    def apply_risk_of_death_from_hypertension(self, gestation_of_interest):
        """
        This function applies risk of death to women with severe hypertensive disease (severe gestational hypertension/
        severe pre-eclampsia). For women who die this function schedules InstantaneousDeathEvent.
        :param gestation_of_interest: gestation in weeks
        """
        df = self.sim.population.props
        params = self.current_parameters
        mni = self.mother_and_newborn_info

        # Risk of death is applied to women with severe hypertensive disease
        at_risk = df.is_alive & df.is_pregnant & (df.ps_gestational_age_in_weeks == gestation_of_interest) & \
                  (df.ps_ectopic_pregnancy == 'none') & ~df.hs_is_inpatient & ~df.la_currently_in_labour & \
                  (df.ps_htn_disorders == 'severe_gest_htn')

        at_risk_of_death_htn = pd.Series(self.rng.random_sample(len(at_risk.loc[at_risk])) <
                                         params['prob_monthly_death_severe_htn'], index=at_risk.loc[at_risk].index)

        if not at_risk_of_death_htn.loc[at_risk_of_death_htn].empty:
            logger.debug(key='message',
                         data=f'The following women have died due to severe gestational hypertension'
                              f'{at_risk_of_death_htn.loc[at_risk_of_death_htn].index}')

            # Those women who die have InstantaneousDeath scheduled
            for person in at_risk_of_death_htn.loc[at_risk_of_death_htn].index:
                self.sim.modules['Demography'].do_death(individual_id=person, cause='severe_gestational_hypertension',
                                                        originating_module=self.sim.modules['PregnancySupervisor'])

                logger.info(key='direct_maternal_death', data={'person': person, 'preg_state': 'antenatal',
                                                               'year': self.sim.date.year})

                del mni[person]

    def apply_risk_of_placental_abruption(self, gestation_of_interest):
        """
        This function applies risk of placental abruption to a slice of the dataframe. It is called by
        PregnancySupervisorEvent.
        :param gestation_of_interest: gestation in weeks
        """
        df = self.sim.population.props

        placenta_abruption = self.apply_linear_model(
            self.ps_linear_models['placental_abruption'],
            df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == gestation_of_interest) &
                   ~df['ps_placental_abruption'] & (df['ps_ectopic_pregnancy'] == 'none') & ~df['hs_is_inpatient'] &
                   ~df['la_currently_in_labour']])

        df.loc[placenta_abruption.loc[placenta_abruption].index, 'ps_placental_abruption'] = True
        for person in placenta_abruption.loc[placenta_abruption].index:
            logger.info(key='maternal_complication', data={'person': person,
                                                           'type': 'placental_abruption',
                                                           'timing': 'antenatal'})

    def apply_risk_of_antepartum_haemorrhage(self, gestation_of_interest):
        """
        This function applies risk of antepartum haemorrhage to a slice of the dataframe. It is called by
        PregnancySupervisorEvent.
        :param gestation_of_interest: gestation in weeks
        """
        df = self.sim.population.props
        params = self.current_parameters

        antepartum_haemorrhage = self.apply_linear_model(
            self.ps_linear_models['antepartum_haem'],
            df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == gestation_of_interest) &
                   (df['ps_ectopic_pregnancy'] == 'none') & ~df['hs_is_inpatient'] & ~df['la_currently_in_labour'] &
                   (df['ps_antepartum_haemorrhage'] == 'none')])

        # Weighted random draw is used to determine severity (for DALY weight mapping)
        random_choice_severity = pd.Series(self.rng.choice(
            ['mild_moderate', 'severe'], p=params['prob_mod_sev_aph'], size=len(
                antepartum_haemorrhage.loc[antepartum_haemorrhage])),
            index=antepartum_haemorrhage.loc[antepartum_haemorrhage].index)

        # We store the severity of the bleed and signify this woman is experiencing an emergency event
        df.loc[antepartum_haemorrhage.loc[antepartum_haemorrhage].index, 'ps_antepartum_haemorrhage'] = \
            random_choice_severity

        df.loc[antepartum_haemorrhage.loc[antepartum_haemorrhage].index, 'ps_emergency_event'] = True

        # Store onset to calculate daly weights
        severe_women = (df.loc[antepartum_haemorrhage.loc[antepartum_haemorrhage].index, 'ps_antepartum_haemorrhage']
                        == 'severe')

        severe_women.loc[severe_women].index.to_series().apply(self.store_dalys_in_mni, mni_variable='severe_aph_onset')
        for person in severe_women.loc[severe_women].index:
            logger.info(key='maternal_complication', data={'person': person,
                                                           'type': 'severe_antepartum_haemorrhage',
                                                           'timing': 'antenatal'})

        non_severe_women = (df.loc[antepartum_haemorrhage.loc[antepartum_haemorrhage].index,
                                   'ps_antepartum_haemorrhage'] != 'severe')

        non_severe_women.loc[non_severe_women].index.to_series().apply(self.store_dalys_in_mni,
                                                                       mni_variable='mild_mod_aph_onset')
        for person in non_severe_women.loc[non_severe_women].index:
            logger.info(key='maternal_complication', data={'person': person,
                                                           'type': 'mild_mod_antepartum_haemorrhage',
                                                           'timing': 'antenatal'})

    def apply_risk_of_sepsis_post_prom(self, gestation_of_interest):
        """
        This function applies risk of chorioamnionitis to women who have experienced premature rupture of membranes
        during labour
        :param gestation_of_interest: gestation in weeks
        """

        df = self.sim.population.props
        params = self.current_parameters

        risk_of_chorio = df.is_alive & df.is_pregnant & (df.ps_gestational_age_in_weeks == gestation_of_interest) & \
                          (df.ps_ectopic_pregnancy == 'none') & ~df.hs_is_inpatient & ~df.la_currently_in_labour & \
                         df.ps_premature_rupture_of_membranes

        infection = pd.Series(self.rng.random_sample(len(risk_of_chorio.loc[risk_of_chorio])) <
                              params['prob_chorioamnionitis'], index=risk_of_chorio.loc[risk_of_chorio].index)

        df.loc[infection.loc[infection].index, 'ps_chorioamnionitis'] = True
        df.loc[infection.loc[infection].index, 'ps_emergency_event'] = True

        infection.loc[infection].index.to_series().apply(self.store_dalys_in_mni, mni_variable='chorio_onset')

        for person in infection.loc[infection].index:
            self.mother_and_newborn_info[person]['chorio_in_preg'] = True
            logger.info(key='maternal_complication', data={'person': person,
                                                           'type': 'clinical_chorioamnionitis',
                                                           'timing': 'antenatal'})

    def apply_risk_of_premature_rupture_of_membranes(self, gestation_of_interest):
        """
        This function applies risk of premature rupture of membranes to a slice of the dataframe. It is called by
        PregnancySupervisorEvent.
        :param gestation_of_interest: gestation in weeks
        """
        df = self.sim.population.props
        params = self.current_parameters

        at_risk = df.is_alive & df.is_pregnant & (df.ps_gestational_age_in_weeks == gestation_of_interest) & \
                  (df.ps_ectopic_pregnancy == 'none') & ~df.hs_is_inpatient & ~df.la_currently_in_labour

        prom = pd.Series(self.rng.random_sample(len(at_risk.loc[at_risk])) < params['prob_prom_per_month'],
                         index=at_risk.loc[at_risk].index)

        df.loc[prom.loc[prom].index, 'ps_premature_rupture_of_membranes'] = True
        # We allow women to seek care for PROM
        df.loc[prom.loc[prom].index, 'ps_emergency_event'] = True

        for person in prom.loc[prom].index:
            logger.info(key='maternal_complication', data={'person': person,
                                                           'type': 'PROM',
                                                           'timing': 'antenatal'})

    def apply_risk_of_preterm_labour(self, gestation_of_interest):
        """
        This function applies risk of preterm labour to a slice of the dataframe. It is called by
        PregnancySupervisorEvent.
        :param gestation_of_interest: gestation in weeks
        """
        df = self.sim.population.props

        preterm_labour = self.apply_linear_model(
            self.ps_linear_models['early_onset_labour'],
            df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == gestation_of_interest)
                   & (df['ps_ectopic_pregnancy'] == 'none') & ~df['hs_is_inpatient'] & ~df['la_currently_in_labour']])

        # To prevent clustering of labour onset we scatter women to go into labour on a random day before their
        # next month gestation
        for person in preterm_labour.loc[preterm_labour].index:
            if df.at[person, 'ps_gestational_age_in_weeks'] == 22:
                poss_day_onset = (27 - 22) * 7
                # We only allow labour to onset from 24 weeks (to match with our definition of preterm labour)
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
            else:
                # If any other gestational ages are pass, the function should end
                logger.debug(key='msg', data=f'Mother {person} was passed to the preterm birth function at'
                                             f' {df.at[person, "ps_gestational_age_in_weeks"]}. No changes will be '
                                             f'made')
                return

            # Due date is updated
            new_due_date = self.sim.date + DateOffset(days=onset_day)

            self.sim.modules['Labour'].reset_due_date(ind_or_df='individual', id_or_index=person,
                                                      new_due_date=new_due_date)
            logger.info(key='message', data=f'Mother {person} will go into preterm labour on '
                                            f'{new_due_date}')

            # And the labour onset event is scheduled for the new due date
            self.sim.schedule_event(labour.LabourOnsetEvent(self.sim.modules['Labour'], person),
                                    new_due_date)

    def update_variables_post_still_birth_for_data_frame(self, women):
        """
        This function updates variables for a slice of the dataframe who have experience antepartum stillbirth
        :param women: women who are experiencing stillbirth
        """
        df = self.sim.population.props
        mni = self.mother_and_newborn_info

        # We reset the relevant pregnancy variables
        df.loc[women.index, 'ps_prev_stillbirth'] = True

        # And reset relevant variables
        df.loc[women.index, 'is_pregnant'] = False

        # We turn the 'delete_mni' key to true- so after the next daly poll this womans entry is deleted
        for person in women.index:
            mni[person]['delete_mni'] = True
            logger.info(key='antenatal_stillbirth', data={'mother': person})

        # Call functions across the modules to ensure properties are rest
        self.sim.modules['Labour'].reset_due_date(
            ind_or_df='data_frame', id_or_index=women.index, new_due_date=pd.NaT)

        self.pregnancy_supervisor_property_reset(
            ind_or_df='data_frame', id_or_index=women.index)

        self.sim.modules['CareOfWomenDuringPregnancy'].care_of_women_in_pregnancy_property_reset(
            ind_or_df='data_frame', id_or_index=women.index)

    def update_variables_post_still_birth_for_individual(self, individual_id):
        """
        This function is called to reset all the relevant pregnancy and treatment variables for a woman who undergoes
        stillbirth outside of the PregnancySupervisor polling event.
        :param individual_id: individual_id
        """
        df = self.sim.population.props
        mni = self.mother_and_newborn_info

        logger.debug(key='message', data=f'person {individual_id} has experience an antepartum stillbirth on '
                                         f'date {self.sim.date}')

        df.at[individual_id, 'ps_prev_stillbirth'] = True
        df.at[individual_id, 'is_pregnant'] = False
        mni[individual_id]['delete_mni'] = True
        logger.info(key='antenatal_stillbirth', data={'mother': individual_id})

        self.sim.modules['Labour'].reset_due_date(
            ind_or_df='individual', id_or_index=individual_id, new_due_date=pd.NaT)

        self.pregnancy_supervisor_property_reset(
            ind_or_df='individual', id_or_index=individual_id)

        self.sim.modules['CareOfWomenDuringPregnancy'].care_of_women_in_pregnancy_property_reset(
            ind_or_df='individual', id_or_index=individual_id)

    def apply_risk_of_still_birth(self, gestation_of_interest):
        """
        This function applies risk of still birth to a slice of the data frame. It is called by PregnancySupervisorEvent
        :param gestation_of_interest: INT used to select women from the data frame at certain gestation
        """
        df = self.sim.population.props

        still_birth = self.apply_linear_model(
            self.ps_linear_models['antenatal_stillbirth'],
            df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == gestation_of_interest) &
                   (df['ps_ectopic_pregnancy'] == 'none') & ~df['hs_is_inpatient'] & ~df['la_currently_in_labour'] &
                   ~df['ps_emergency_event']])

        self.update_variables_post_still_birth_for_data_frame(still_birth.loc[still_birth])

    def induction_care_seeking_and_still_birth_risk(self, gestation_of_interest):
        """
        This function is called for post term women and applies a probability that they will seek care for induction
        and if not will experience antenatal stillbirth
        :param gestation_of_interest: INT used to select women from the data frame at certain gestation
        """
        df = self.sim.population.props
        params = self.current_parameters

        # We select the appropriate women
        post_term_women = df.is_alive & df.is_pregnant & (df.ps_gestational_age_in_weeks == gestation_of_interest) & \
                           (df.ps_ectopic_pregnancy == 'none') & ~df.hs_is_inpatient & ~df.la_currently_in_labour & \
                          ~df.ps_emergency_event

        # Apply a probability they will seek care for induction
        care_seekers = pd.Series(self.rng.random_sample(len(post_term_women.loc[post_term_women]))
                                 < params['prob_seek_care_induction'],
                                 index=post_term_women.loc[post_term_women].index)

        # If they do, we scheduled them to preset to a health facility immediately (this HSI schedules the correct
        # labour modules)
        for person in care_seekers.loc[care_seekers].index:
            from tlo.methods.care_of_women_during_pregnancy import (
                HSI_CareOfWomenDuringPregnancy_PresentsForInductionOfLabour,
            )

            induction = HSI_CareOfWomenDuringPregnancy_PresentsForInductionOfLabour(
                self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person)

            self.sim.modules['HealthSystem'].schedule_hsi_event(induction, priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        # We apply risk of still birth to those who dont seek care
        non_care_seekers = df.loc[care_seekers.loc[~care_seekers].index]
        still_birth_risk = self.ps_linear_models['antenatal_stillbirth'].predict(non_care_seekers)
        weekly_risk = still_birth_risk / 4.5
        still_birth = self.rng.random_sample(len(weekly_risk)) < weekly_risk

        self.update_variables_post_still_birth_for_data_frame(still_birth.loc[still_birth])

    def care_seeking_pregnancy_loss_complications(self, individual_id, cause):
        """
        This function manages care seeking for women experiencing ectopic pregnancy or complications following
        spontaneous/induced abortion.
        :param individual_id: individual_id
        :param cause: 'abortion', 'ectopic_pre_rupture', 'ectopic_ruptured'
        :return: Returns True/False value to signify care seeking
        """
        params = self.current_parameters

        # Care seeking probability varies according to complication
        if cause == 'ectopic_pre_rupture':
            care_seeking = self.rng.random_sample() < params['prob_care_seeking_ectopic_pre_rupture']
        else:
            care_seeking = self.rng.random_sample() < params['prob_seek_care_pregnancy_loss']

        if care_seeking:
            logger.debug(key='message', data=f'Mother {individual_id} will seek care following pregnancy loss')

            # We assume women will seek care via HSI_GenericEmergencyFirstApptAtFacilityLevel1 and will be admitted for
            # care in CareOfWomenDuringPregnancy module

            from tlo.methods.hsi_generic_first_appts import (
                HSI_GenericEmergencyFirstApptAtFacilityLevel1,
            )

            event = HSI_GenericEmergencyFirstApptAtFacilityLevel1(
                self.sim.modules['PregnancySupervisor'],
                person_id=individual_id)

            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))
            return True

        else:
            logger.debug(key='message', data=f'Mother {individual_id} will not seek care following pregnancy loss')
            return False

    def apply_risk_of_death_from_monthly_complications(self, individual_id):
        """
        This calculated risk of death for women who have developed complications but have not received treatment.
        It is called by the PregnancySupervisor Event and HSI_CareOfWomenDuringPregnancy_MaternalEmergencyAssessment
         if care cant be delivered .
        :param individual_id: individual_id
        """

        df = self.sim.population.props
        params = self.current_parameters

        mother = df.loc[individual_id]
        causes = list()

        # Create list of the potential causes of death if they are present in the mother
        if mother.ps_antepartum_haemorrhage == 'severe':
            causes.append('antepartum_haemorrhage')
        if mother.ps_htn_disorders == 'severe_pre_eclamp':
            causes.append('severe_pre_eclampsia')
        if mother.ps_htn_disorders == 'eclampsia':
            causes.append('eclampsia')
        if mother.ps_chorioamnionitis:
            causes.append('antenatal_sepsis')

        risks = dict()
        for cause in causes:
            risk = {f'{cause}': params[f'prob_{cause}_death']}
            risks.update(risk)

        result = 1
        for cause in risks:
            result *= (1 - risks[cause])

        # If random draw is less that the total risk of death, she will die and the primary cause is then
        # determined
        if self.rng.random_sample() < (1 - result):
            denominator = sum(risks.values())
            probs = list()

            # Cycle over each cause in the dictionary and divide by the sum of the probabilities
            for cause in risks:
                risks[cause] = risks[cause] / denominator
                probs.append(risks[cause])

            cause_of_death = self.rng.choice(causes, p=probs)

            self.sim.modules['Demography'].do_death(individual_id=individual_id, cause=f'{cause_of_death}',
                                                    originating_module=self.sim.modules['PregnancySupervisor'])

            logger.info(key='direct_maternal_death', data={'person': individual_id, 'preg_state': 'antenatal',
                                                           'year': self.sim.date.year})

            del self.mother_and_newborn_info[individual_id]

    def generate_mother_and_newborn_dictionary_for_individual(self, individual_id):
        """ This function generates variables within the mni dictionary for women. It is abstracted to a function for
        testing purposes"""
        mni = self.mother_and_newborn_info
        df = self.sim.population.props

        assert df.at[individual_id, 'is_pregnant']

        mni[individual_id] = {'delete_mni': False,
                              'abortion_onset': pd.NaT,
                              'abortion_haem_onset': pd.NaT,
                              'abortion_sep_onset': pd.NaT,
                              'eclampsia_onset': pd.NaT,
                              'mild_mod_aph_onset': pd.NaT,
                              'severe_aph_onset': pd.NaT,
                              'chorio_onset': pd.NaT,
                              'chorio_in_preg': False,
                              'ectopic_onset': pd.NaT,
                              'ectopic_rupture_onset': pd.NaT,
                              'gest_diab_onset': pd.NaT,
                              'gest_diab_diagnosed_onset': pd.NaT,
                              'gest_diab_resolution': pd.NaT,
                              'mild_anaemia_onset': pd.NaT,
                              'mild_anaemia_resolution': pd.NaT,
                              'moderate_anaemia_onset': pd.NaT,
                              'moderate_anaemia_resolution': pd.NaT,
                              'severe_anaemia_onset': pd.NaT,
                              'severe_anaemia_resolution': pd.NaT,
                              'mild_anaemia_pp_onset': pd.NaT,
                              'mild_anaemia_pp_resolution': pd.NaT,
                              'moderate_anaemia_pp_onset': pd.NaT,
                              'moderate_anaemia_pp_resolution': pd.NaT,
                              'severe_anaemia_pp_onset': pd.NaT,
                              'severe_anaemia_pp_resolution': pd.NaT,
                              'hypertension_onset': pd.NaT,
                              'hypertension_resolution': pd.NaT,
                              'obstructed_labour_onset': pd.NaT,
                              'sepsis_onset': pd.NaT,
                              'uterine_rupture_onset': pd.NaT,
                              'mild_mod_pph_onset': pd.NaT,
                              'severe_pph_onset': pd.NaT,
                              'secondary_pph_onset': pd.NaT,
                              'vesicovaginal_fistula_onset': pd.NaT,
                              'vesicovaginal_fistula_resolution': pd.NaT,
                              'rectovaginal_fistula_onset': pd.NaT,
                              'rectovaginal_fistula_resolution': pd.NaT,
                              'test_run': False,  # used by labour module when running some model tests
                              'pred_syph_infect': pd.NaT,

                              # todo: delete and delete usage (just for checking)
                              'cs_indication': 'none'

                              }


class PregnancySupervisorEvent(RegularEvent, PopulationScopeEventMixin):
    """ This is the PregnancySupervisorEvent, it is a weekly event which has two primary functions.
    1.) It updates the gestational age (in weeks) of all women who are pregnant.
    2.) It applies risk of complications/outcomes and care seeking during the antenatal period at pre-defined time
    points of pregnancy (defined below)"""

    def __init__(self, module, ):
        super().__init__(module, frequency=DateOffset(weeks=1))

    def apply(self, population):
        df = population.props
        params = self.module.current_parameters
        mni = self.module.mother_and_newborn_info

        # =================================== UPDATING LENGTH OF PREGNANCY ============================================
        # Length of pregnancy is commonly measured as gestational age which commences on the first day of a womans last
        # menstrual period (therefore including around 2 weeks in which a woman isnt pregnant)

        # We calculate a womans gestational age by first calculating the foetal age (measured from conception) and then
        # adding 2 weeks. The literature describing the epidemiology of maternal conditions almost exclusively uses
        # gestational age

        alive_and_preg = df.is_alive & df.is_pregnant
        foetal_age_in_days = self.sim.date - df.loc[alive_and_preg, 'date_of_last_pregnancy']
        foetal_age_in_weeks = foetal_age_in_days / np.timedelta64(1, 'W')
        rounded_weeks = np.ceil(foetal_age_in_weeks)
        df.loc[alive_and_preg, "ps_gestational_age_in_weeks"] = rounded_weeks + 2

        logger.debug(key='message', data=f'updating gestational ages on date {self.sim.date}')
        assert (df.loc[alive_and_preg, 'ps_gestational_age_in_weeks'] > 1).all().all()

        # Here we begin to populate the mni dictionary for each newly pregnant woman. Within this module this dictionary
        # contains information about the onset of complications in order to calculate monthly DALYs
        newly_pregnant = df.loc[alive_and_preg & (df['ps_gestational_age_in_weeks'] == 3)]
        for person in newly_pregnant.index:
            self.module.generate_mother_and_newborn_dictionary_for_individual(person)

        # =========================== APPLYING RISK OF ADVERSE PREGNANCY OUTCOMES =====================================
        # The aim of this event is to apply risk of certain outcomes of pregnancy at relevant points in a womans
        # gestation. Risk of complications that occur only once during pregnancy (below) are applied within the event,
        # otherwise code applying risk is stored in functions (above)

        # At the beginning of pregnancy (3 weeks GA (and therefore the first week a woman is pregnant) we determine if
        # a woman will develop ectopic pregnancy, multiple pregnancy, placenta praevia and if/when she will seek care
        # for her first antenatal visit

        #  ------------------------------APPLYING RISK OF ECTOPIC PREGNANCY -------------------------------------------
        # We use the apply_linear_model function to determine which women will develop ectopic pregnancy - this format
        # is similar to the functions which apply risk of complication
        new_pregnancy = df.is_alive & df.is_pregnant & (df.ps_gestational_age_in_weeks == 3)
        ectopic_risk = pd.Series(self.module.rng.random_sample(len(new_pregnancy.loc[new_pregnancy])) <
                                 params['prob_ectopic_pregnancy'], index=new_pregnancy.loc[new_pregnancy].index)

        # Make the appropriate changes to the data frame and log the number of ectopic pregnancies
        df.loc[ectopic_risk.loc[ectopic_risk].index, 'ps_ectopic_pregnancy'] = 'not_ruptured'

        # For women whose pregnancy is ectopic we scheduled them to the EctopicPregnancyEvent in between 3-5 weeks
        # of pregnancy (this simulates time period prior to which symptoms onset- and may trigger care seeking)
        for person in ectopic_risk.loc[ectopic_risk].index:
            logger.info(key='maternal_complication', data={'person': person,
                                                           'type': 'ectopic_unruptured',
                                                           'timing': 'antenatal'})

            self.sim.schedule_event(EctopicPregnancyEvent(self.module, person),
                                    (self.sim.date + pd.Timedelta(days=7 * 3 + self.module.rng.randint(0, 7 * 2))))

        #  ---------------------------- APPLYING RISK OF MULTIPLE PREGNANCY -------------------------------------------
        # For the women who aren't having an ectopic, we determine if they may be carrying multiple pregnancies and make
        # changes accordingly
        multiple_risk = df.is_alive & df.is_pregnant & (df.ps_gestational_age_in_weeks == 3) & \
                         (df.ps_ectopic_pregnancy == 'none')

        multiples = pd.Series(self.module.rng.random_sample(len(multiple_risk.loc[multiple_risk]))
                              < params['prob_multiples'],  index=multiple_risk.loc[multiple_risk].index)

        df.loc[multiples.loc[multiples].index, 'ps_multiple_pregnancy'] = True

        for person in multiples.loc[multiples].index:
            logger.info(key='maternal_complication', data={'person': person,
                                                           'type': 'multiple_pregnancy',
                                                           'timing': 'antenatal'})

        #  -----------------------------APPLYING RISK OF PLACENTA PRAEVIA  -------------------------------------------
        # Next,we apply a one off risk of placenta praevia (placenta will grow to cover the cervix either partially or
        # completely) which will increase likelihood of bleeding later in pregnancy
        placenta_praevia = self.module.apply_linear_model(
            self.module.ps_linear_models['placenta_praevia'],
            df.loc[new_pregnancy & (df['ps_ectopic_pregnancy'] == 'none')])

        df.loc[placenta_praevia.loc[placenta_praevia].index, 'ps_placenta_praevia'] = True
        for person in placenta_praevia.loc[placenta_praevia].index:
            logger.info(key='maternal_complication', data={'person': person,
                                                           'type': 'placenta_praevia',
                                                           'timing': 'antenatal'})

        #  ------------------------- APPLYING RISK OF SYPHILIS INFECTION DURING PREGNANCY  ---------------------------
        # Finally apply risk that syphilis will develop during pregnancy
        at_risk_women = df.is_alive & df.is_pregnant & (df.ps_gestational_age_in_weeks == 3) & (df.ps_ectopic_pregnancy
                                                                                                == 'none')

        syphilis_risk = pd.Series(self.module.rng.random_sample(len(at_risk_women.loc[at_risk_women]))
                                  < params['prob_syphilis_during_pregnancy'],
                                  index=at_risk_women.loc[at_risk_women].index)

        # Schedule point of onset randomly during possible length of pregnancy
        for person in syphilis_risk.loc[syphilis_risk].index:
            onset_day = self.module.rng.randint(0, 280)
            mni[person]['pred_syph_infect'] = self.sim.date + pd.Timedelta(days=onset_day)
            self.sim.schedule_event(SyphilisInPregnancyEvent(self.module, person),
                                    (self.sim.date + pd.Timedelta(days=onset_day)))

        # ----------------------------------- SCHEDULING FIRST ANC VISIT -----------------------------------------
        # Finally for these women we determine care seeking for the first antenatal care contact of their
        # pregnancy. We use a linear model to predict if these women will attend early ANC and at least 4 visits

        # First we identify all the women predicted to attend ANC, with the first visit occurring before 4 months
        early_initiation_anc4 = self.module.apply_linear_model(
            self.module.ps_linear_models['early_initiation_anc4'],
            df.loc[new_pregnancy & (df['ps_ectopic_pregnancy'] == 'none')])

        # Of the women who will not attend ANC4 early, we determine who will attend ANC4
        late_initation_anc4 = pd.Series(self.module.rng.random_sample(
            len(early_initiation_anc4.loc[~early_initiation_anc4])) < params['prob_late_initiation_anc4'],
                                        index=early_initiation_anc4.loc[~early_initiation_anc4].index)

        for v in late_initation_anc4.loc[late_initation_anc4].index:
            assert v not in early_initiation_anc4.loc[early_initiation_anc4].index

        df.loc[early_initiation_anc4.loc[early_initiation_anc4].index, 'ps_anc4'] = True
        df.loc[late_initation_anc4.loc[late_initation_anc4].index, 'ps_anc4'] = True

        anc_below_4 = df.is_alive & df.is_pregnant & (df.ps_gestational_age_in_weeks == 3) & \
                      (df.ps_ectopic_pregnancy == 'none') & ~df.ps_anc4

        early_initiation_anc_below_4 = pd.Series(self.module.rng.random_sample(len(anc_below_4.loc[anc_below_4]))
                                                 < params['prob_early_initiation_anc_below4'],
                                                 index=anc_below_4.loc[anc_below_4].index)

        def schedulde_early_visit(df_slice):
            for person in df_slice.index:
                random_draw_gest_at_anc = self.module.rng.choice([1, 2, 3, 4],
                                                                 p=params['prob_anc1_months_1_to_4'])
                self.module.schedule_anc_one(individual_id=person, anc_month=random_draw_gest_at_anc)

        for slice in [early_initiation_anc4.loc[early_initiation_anc4],
                      early_initiation_anc_below_4.loc[early_initiation_anc_below_4]]:
            schedulde_early_visit(slice)

        def schedulde_late_visit(df_slice):
            for person in df_slice.index:
                random_draw_gest_at_anc = self.module.rng.choice([5, 6, 7, 8, 9, 10],
                                                                 p=params['prob_anc1_months_5_to_9'])

                # We use month ten to capture women who will never attend ANC during their pregnancy
                if random_draw_gest_at_anc != 10:
                    self.module.schedule_anc_one(individual_id=person, anc_month=random_draw_gest_at_anc)

        for slice in [late_initation_anc4.loc[late_initation_anc4],
                      early_initiation_anc_below_4.loc[~early_initiation_anc_below_4]]:
            schedulde_late_visit(slice)

        # ------------------------ APPLY RISK OF ADDITIONAL PREGNANCY COMPLICATIONS -----------------------------------
        # The following functions apply risk of key complications/outcomes of pregnancy as specific time points of a
        # mothers gestation in weeks. These 'gestation_of_interest' parameters roughly represent the last week in each
        # month of pregnancy. These time  points at which risk is applied, vary between complications according to their
        # epidemiology

        # The application of these risk is intentionally ordered as described below...

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

        # For women whose pregnancy will continue will apply a risk of developing a number of acute and chronic
        # (length of pregnancy) complications
        for gestation_of_interest in [22, 27, 31, 35, 40]:
            self.module.apply_risk_of_hypertensive_disorders(gestation_of_interest=gestation_of_interest)
            self.module.apply_risk_of_gestational_diabetes(gestation_of_interest=gestation_of_interest)
            self.module.apply_risk_of_placental_abruption(gestation_of_interest=gestation_of_interest)
            self.module.apply_risk_of_antepartum_haemorrhage(gestation_of_interest=gestation_of_interest)
            self.module.apply_risk_of_sepsis_post_prom(gestation_of_interest=gestation_of_interest)
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

        # Any women for whom ps_emergency_event == True may chose to seek care for one or more severe complications
        # (antepartum haemorrhage, severe pre-eclampsia, eclampsia or premature rupture of membranes) - this is distinct
        # from care seeking following abortion/ectopic

        potential_care_seekers = df.is_alive & df.is_pregnant & (df.ps_ectopic_pregnancy == 'none') \
                                 & df.ps_emergency_event & ~df.hs_is_inpatient & ~df.la_currently_in_labour & \
                                 (df.la_due_date_current_pregnancy != self.sim.date)

        care_seeking = pd.Series(self.module.rng.random_sample(len(potential_care_seekers.loc[potential_care_seekers]))
                                 < params['prob_seek_care_pregnancy_complication'],
                                 index=potential_care_seekers.loc[potential_care_seekers].index)

        # We assume women who seek care will present to a form of Maternal Assessment Unit- not through normal A&E
        for person in care_seeking.loc[care_seeking].index:
            logger.debug(key='message', data=f'Mother {person} will seek care following acute pregnancy'
                                             f'complications')

            from tlo.methods.care_of_women_during_pregnancy import (
                HSI_CareOfWomenDuringPregnancy_MaternalEmergencyAssessment,
            )

            acute_pregnancy_hsi = HSI_CareOfWomenDuringPregnancy_MaternalEmergencyAssessment(
                self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person)

            self.sim.modules['HealthSystem'].schedule_hsi_event(acute_pregnancy_hsi, priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        # -------- APPLYING RISK OF DEATH/STILL BIRTH FOR NON-CARE SEEKERS FOLLOWING PREGNANCY EMERGENCIES --------
        # We select the women who have chosen not to seek care following pregnancy emergency- and we now apply risk of
        # death

        if not care_seeking.loc[~care_seeking].empty:
            # We reset this variable to prevent additional unnecessary care seeking next month
            df.loc[care_seeking.loc[~care_seeking].index, 'ps_emergency_event'] = False

            logger.debug(key='message', data=f'The following women will not seek care after experiencing a '
                                             f'pregnancy emergency: {care_seeking.loc[~care_seeking].index}')

            # As women may have experience more than one complication during the moth we determine here which of the
            # complication will be the primary cause of death
            for person in care_seeking.loc[~care_seeking].index:
                self.module.apply_risk_of_death_from_monthly_complications(person)

        # ============================ RISK OF STILLBIRTH ========================================================
        for gestation_of_interest in [27, 31, 35, 40]:
            self.module.apply_risk_of_still_birth(gestation_of_interest=gestation_of_interest)

        # ============================ POST TERM RISK OF STILLBIRTH ==================================================
        # Finally we determine if women who are post term will seek care for induction/experience stillbirth
        for gestation_of_interest in [41, 42, 43, 44, 45]:
            self.module.induction_care_seeking_and_still_birth_risk(gestation_of_interest=gestation_of_interest)

        # Finally reset the emergency event property for care seeking women (used to ensure risk of stillbirth is
        # applied to women who arent seeking care that month)
        df.loc[care_seeking.index, 'ps_emergency_event'] = False


class EctopicPregnancyEvent(Event, IndividualScopeEventMixin):
    """This is EctopicPregnancyEvent. It is scheduled by the set_pregnancy_complications function within
     PregnancySupervisorEvent for women who have experienced ectopic pregnancy. This event makes changes to the data
     frame for women with ectopic pregnancies, applies a probability of care seeking and schedules the
     EctopicRuptureEvent."""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props

        # Check only the right women have arrived here
        assert df.at[individual_id, 'ps_ectopic_pregnancy'] == 'not_ruptured'
        assert df.at[individual_id, 'ps_gestational_age_in_weeks'] < 9

        if not df.at[individual_id, 'is_alive']:
            return

        # reset pregnancy variables and store onset for daly calcualtion
        df.at[individual_id, 'is_pregnant'] = False
        self.module.store_dalys_in_mni(individual_id, 'ectopic_onset')

        self.sim.modules['Labour'].reset_due_date(
            ind_or_df='individual', id_or_index=individual_id, new_due_date=pd.NaT)

        self.module.pregnancy_supervisor_property_reset(
            ind_or_df='individual', id_or_index=individual_id)

        # Determine if women will seek care at this stage
        care_seeking_result = self.module.care_seeking_pregnancy_loss_complications(individual_id,
                                                                                    cause='ectopic_pre_rupture')
        if not care_seeking_result:

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
        assert df.at[individual_id, 'ps_ectopic_pregnancy'] == 'not_ruptured'

        if not df.at[individual_id, 'is_alive']:
            return

        logger.debug(key='message', data=f'persons {individual_id} untreated ectopic pregnancy has now ruptured on '
                                         f'date {self.sim.date}')
        logger.info(key='maternal_complication', data={'person': individual_id,
                                                       'type': 'ectopic_ruptured',
                                                       'timing': 'antenatal'})

        # Set the variable
        df.at[individual_id, 'ps_ectopic_pregnancy'] = 'ruptured'
        self.module.store_dalys_in_mni(individual_id, 'ectopic_rupture_onset')

        # We see if this woman will now seek care following rupture
        self.module.care_seeking_pregnancy_loss_complications(individual_id, cause='ectopic_ruptured')

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
        mni = self.module.mother_and_newborn_info

        if not df.at[individual_id, 'is_alive']:
            return

        # Individual risk of death is calculated through the linear model
        risk_of_death = self.module.ps_linear_models[f'{self.cause}_death'].predict(df.loc[[individual_id]])[
                individual_id]

        # If the death occurs we record it here
        if self.module.rng.random_sample() < risk_of_death:
            logger.debug(key='message', data=f'person {individual_id} has died due to {self.cause} on date '
                                             f'{self.sim.date}')

            logger.info(key='direct_maternal_death', data={'person': individual_id, 'preg_state': 'antenatal',
                                                           'year': self.sim.date.year})

            self.sim.modules['Demography'].do_death(individual_id=individual_id, cause=f'{self.cause}',
                                                    originating_module=self.sim.modules['PregnancySupervisor'])

            if individual_id in mni:
                mni[individual_id]['delete_mni'] = True

        else:
            # Otherwise we reset any variables
            if self.cause == 'ectopic_pregnancy':
                df.at[individual_id, 'ps_ectopic_pregnancy'] = 'none'
                if individual_id in mni:
                    mni[individual_id]['delete_mni'] = True

            else:
                self.module.abortion_complications.unset(individual_id, 'sepsis', 'haemorrhage', 'injury')
                self.sim.modules['CareOfWomenDuringPregnancy'].pac_interventions.unset(individual_id,
                                                                                       'mva', 'd_and_c', 'misoprostol',
                                                                                       'antibiotics', 'blood_products',
                                                                                       'injury_repair')
                if individual_id in mni:
                    mni[individual_id]['delete_mni'] = True


class GestationalDiabetesGlycaemicControlEvent(Event, IndividualScopeEventMixin):
    """
    This is GestationalDiabetesGlycaemicControlEvent. It is scheduled by CareOfWomenDuringPregnancy module after a
    woman is started on treatment for gestational diabetes. This event determines if the treatment a woman has been
    started on for GDM will effectively control her blood sugars
    """

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.current_parameters
        mother = df.loc[individual_id]

        if not mother.is_alive or not mother.is_pregnant or (mother.ps_gestational_age_in_weeks < 20):
            return

        if mother.ps_gest_diab != 'none':
            # Check only the right women are sent here
            assert mother.ac_gest_diab_on_treatment != 'none'

            # We apply a probability that the treatment this woman is receiving for her GDM (diet and exercise/
            # oral anti-diabetics/ insulin) will not control this womans hyperglycaemia
            if self.module.rng.random_sample() > params[
                    f'prob_glycaemic_control_{mother.ac_gest_diab_on_treatment }']:

                # If so we reset her diabetes status as uncontrolled, her treatment is ineffective at reducing risk of
                # still birth, and when she returns for follow up she should be started on the next treatment available
                df.at[individual_id, 'ps_gest_diab'] = 'uncontrolled'


class SyphilisInPregnancyEvent(Event, IndividualScopeEventMixin):
    """
    This is SyphilisInPregnancyEvent. It is scheduled by PregnancySupervisorEvent module after a
    woman becomes pregnant and is will experience syphilis during their pregnancy. """

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        mni = self.module.mother_and_newborn_info

        if not df.at[individual_id, 'is_alive'] or not df.at[individual_id, 'is_pregnant'] or individual_id not in mni:
            return

        elif not (mni[individual_id]['pred_syph_infect'] == self.sim.date):
            return

        else:
            df.at[individual_id, 'ps_syphilis'] = True
            logger.info(key='maternal_complication', data={'person': individual_id,
                                                           'type': 'syphilis',
                                                           'timing': 'antenatal'})


class ParameterUpdateEvent(Event, PopulationScopeEventMixin):
    """This is ParameterUpdateEvent. It is scheduled to occur once on 2015 to update parameters being used by the
    maternal and newborn health model"""
    def __init__(self, module):
        super().__init__(module)

    def apply(self, population):

        logger.info(key='msg', data='Now updating parameters in the maternal and perinatal health modules...')

        def switch_parameters(master_params, current_params):
            for key, value in current_params.items():
                current_params[key] = master_params[key][1]

        switch_parameters(self.module.parameters, self.module.current_parameters)

        switch_parameters(self.sim.modules['CareOfWomenDuringPregnancy'].parameters,
                          self.sim.modules['CareOfWomenDuringPregnancy'].current_parameters)

        switch_parameters(self.sim.modules['Labour'].parameters, self.sim.modules['Labour'].current_parameters)

        switch_parameters(self.sim.modules['NewbornOutcomes'].parameters,
                          self.sim.modules['NewbornOutcomes'].current_parameters)

        switch_parameters(self.sim.modules['PostnatalSupervisor'].parameters,
                          self.sim.modules['PostnatalSupervisor'].current_parameters)


class PregnancyLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """This is PregnancyLoggingEvent. It runs yearly to produce summary statistics around pregnancy."""

    def __init__(self, module):
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(years=self.repeat))

    def apply(self, population):
        df = self.sim.population.props

        women_reproductive_age = len(df.index[(df.is_alive & (df.sex == 'F') & (df.age_years > 14) &
                                               (df.age_years < 50))])
        pregnant_at_year_end = len(df.index[df.is_alive & df.is_pregnant])
        women_with_previous_sa = len(df.index[(df.is_alive & (df.sex == 'F') & (df.age_years > 14) &
                                               (df.age_years < 50) & df.ps_prev_spont_abortion)])
        women_with_previous_pe = len(df.index[(df.is_alive & (df.sex == 'F') & (df.age_years > 14) &
                                               (df.age_years < 50) & df.ps_prev_pre_eclamp)])
        women_with_hysterectomy = len(df.index[(df.is_alive & (df.sex == 'F') & (df.age_years > 14) &
                                               (df.age_years < 50) & df.la_has_had_hysterectomy)])
        yearly_prev_sa = (women_with_previous_sa / women_reproductive_age) * 100
        yearly_prev_pe = (women_with_previous_pe / women_reproductive_age) * 100
        yearly_prev_hysterectomy = (women_with_hysterectomy/women_reproductive_age) * 100

        logger.info(key='preg_info',
                    data={'women_repro_age': women_reproductive_age,
                          'women_pregnant': pregnant_at_year_end,
                          'prev_sa': yearly_prev_sa,
                          'prev_pe': yearly_prev_pe,
                          'hysterectomy': yearly_prev_hysterectomy})
