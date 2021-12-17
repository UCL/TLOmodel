from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging, util, Date
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel
from tlo.methods import Metadata, labour, pregnancy_supervisor_lm
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

        # This variable will store a Bitset handler for the property ps_abortion_complications
        self.abortion_complications = None

    INIT_DEPENDENCIES = {'Demography'}

    OPTIONAL_INIT_DEPENDENCIES = {'HealthBurden'}

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


        # ANAEMIA...
        'baseline_prob_anaemia_per_month': Parameter(
            Types.LIST, 'baseline risk of a woman developing anaemia secondary only to pregnant'),
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
        'prob_anc1_months_2_to_4': Parameter(
            Types.LIST, 'list of probabilities that a woman will attend her first ANC visit at either month 2, 3 or'
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

        # ANALYSIS PARAMETERS...
        'anc_service_structure': Parameter(
            Types.INT, 'stores type of ANC service being delivered in the model (anc4 or anc8) and is used in analysis'
                       ' scripts to change ANC structure'),
        'switch_anc_coverage': Parameter(
            Types.BOOL, 'used to signal if a change in parameters governing ANC coverage should be made at some '
                        'predetermined time point'),
        'target_anc_coverage_for_analysis': Parameter(
            Types.REAL, 'contains a target level of coverage for analysis so that a linear model of choice can be '
                        'scaled to force set level of intervention coverage'),

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
        # load parameters from the resource file
        parameter_dataframe = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_PregnancySupervisor.xlsx',
                                            sheet_name='parameter_values')
        self.load_parameters_from_dataframe(parameter_dataframe)

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

    def initialise_population(self, population):
        df = population.props

        df.loc[df.is_alive, 'ps_gestational_age_in_weeks'] = 0
        df.loc[df.is_alive, 'ps_date_of_anc1'] = pd.NaT
        df.loc[df.is_alive, 'ps_ectopic_pregnancy'] = 'none'
        df.loc[df.is_alive, 'ps_placenta_praevia'] = False
        df.loc[df.is_alive, 'ps_multiple_pregnancy'] = False
        df.loc[df.is_alive, 'ps_syphilis'] = False
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


        # This bitset property stores 'types' of complication that can occur after an abortion
        self.abortion_complications = BitsetHandler(self.sim.population, 'ps_abortion_complications',
                                                    ['sepsis', 'haemorrhage', 'injury', 'other'])

        # Finally, for women of reproductive age at baseline, we determine if they have ever previous experience a
        # miscarriage. This impacts future likelihood of miscarriage.
        reproductive_age_women = df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)

        previous_miscarriage = pd.Series(
            self.rng.random_sample(len(reproductive_age_women.loc[reproductive_age_women])) <
            self.parameters['prob_previous_miscarriage_at_baseline'][0],
            index=reproductive_age_women.loc[reproductive_age_women].index)

        df.loc[previous_miscarriage.loc[previous_miscarriage].index, 'ps_prev_spont_abortion'] = True

    def initialise_simulation(self, sim):
        # When the simulation initialises we use the dictionary self.current_parameters to store the parameters of
        # interest for the time period of 2010-2015. These are later updated in ParameterUpdateEvent.
        for key, value in self.parameters.items():
            if type(value) is list:
                self.current_parameters[key] = self.parameters[key][0]
            else:
                self.current_parameters[key] = self.parameters[key]

        # Next we register and schedule the PregnancySupervisorEvent
        sim.schedule_event(PregnancySupervisorEvent(self),
                           sim.date + DateOffset(days=0))

        # ..and register and schedule logging event
        sim.schedule_event(PregnancyLoggingEvent(self),
                           sim.date + DateOffset(years=1))

        # ...and register and schedule the parameter update event
        sim.schedule_event(ParameterUpdateEvent(self),
                           Date(2015, 1, 1))

        # ... and finally register and schedule the parameter override event. This is used in analysis scripts to change
        # key parameters after the simulation 'burn in' period
        sim.schedule_event(OverrideKeyParameterForAnalysis(self),
                           Date(2021, 1, 1))

        # ==================================== LINEAR MODEL EQUATIONS =================================================
        # Next we scale linear models according to distribution of predictors in the dataframe at baseline
        params = self.current_parameters

        # First we create all of the custom linear models used within this module and store them in
        # pregnancy_supervisor_lm.py
        self.ps_linear_models = {

            # This equation predicts women's probability of attending four ANC visits with the first visit occurring
            # during or prior to the fourth month of pregnancy
            'early_initiation_anc4': LinearModel.custom(pregnancy_supervisor_lm.early_initiation_anc4,
                                                        parameters=params),

            # This equation determines the probability of death following en ectopic pregnancy
            'ectopic_pregnancy_death': LinearModel.custom(pregnancy_supervisor_lm.ectopic_pregnancy_death,
                                                          parameters=params),

            # This equation determines the monthly probability of a women experiencing a miscarriage prior to 28 weeks
            # gestation
            'spontaneous_abortion': LinearModel.custom(pregnancy_supervisor_lm.spontaneous_abortion, parameters=params),

            # This equation determines the probability of death following a complicated miscarriage
            'spontaneous_abortion_death': LinearModel.custom(pregnancy_supervisor_lm.spontaneous_abortion_death,
                                                             parameters=params),

            # This equation determines the probability of death following an induced abortion
            'induced_abortion_death': LinearModel.custom(pregnancy_supervisor_lm.induced_abortion_death,
                                                         parameters=params),

            # This equation determines the monthly probability of a woman determining anaemia during her pregnancy
            'maternal_anaemia': LinearModel.custom(pregnancy_supervisor_lm.maternal_anaemia, parameters=params),

            # This equation determines the monthly probability of a women going into labour before reaching term
            # gestation (i.e. 37 weeks or more)
            'early_onset_labour': LinearModel.custom(pregnancy_supervisor_lm.preterm_labour, parameters=params),

            # This equation determines the per-pregnancy probability of a woman developing placenta praevia, where her
            # placenta is either fully or partially covering the cervix. Praevia is a predictor or antenatal bleeding
            'placenta_praevia': LinearModel.custom(pregnancy_supervisor_lm.placenta_praevia, parameters=params),

            # This equations determines the monthly probability of a woman developing placental abruption during
            # pregnancy which is a strong predictor of antenatal bleeding
            'placental_abruption': LinearModel.custom(pregnancy_supervisor_lm.placental_abruption, parameters=params),

            # This equation determines the monthly probability of a women developing antepartum haemorrhage. Haemorrhage
            # may only occur in the presence of either praevia or abruption
            'antepartum_haem': LinearModel.custom(pregnancy_supervisor_lm.antepartum_haem, parameters=params),

            # This equation determines the monthly probability of a women developing gestational diabetes
            'gest_diab': LinearModel.custom(pregnancy_supervisor_lm.gest_diab, parameters=params),

            # This equation determines the monthly probability of a women developing gestational hypertension
            'gest_htn': LinearModel.custom(pregnancy_supervisor_lm.gest_htn, parameters=params),

            # This equation determines the monthly probability of a women developing mild pre-eclampsia
            'pre_eclampsia': LinearModel.custom(pregnancy_supervisor_lm.pre_eclampsia, parameters=params),

            # This equation determines the monthly probability of a women experiencing an antenatal stillbirth,
            # pregnancy loss following 28 weeks gestation
            'antenatal_stillbirth': LinearModel.custom(pregnancy_supervisor_lm.antenatal_stillbirth, parameters=params),
        }

        # Next we create a dict with all the models to be scaled and the 'target' rate parameter
        mod = self.ps_linear_models
        models_to_be_scaled = {'pp': [mod['placenta_praevia'], 'prob_placenta_praevia'],
                               'an': [mod['maternal_anaemia'], 'baseline_prob_anaemia_per_month'],
                               'gd': [mod['gest_diab'], 'prob_gest_diab_per_month'],
                               'gh': [mod['gest_htn'], 'prob_gest_htn_per_month'],
                               'pe': [mod['pre_eclampsia'], 'prob_pre_eclampsia_per_month'],
                               'pa': [mod['placental_abruption'], 'prob_placental_abruption_per_month'],
                               'ansb': [mod['antenatal_stillbirth'], 'prob_still_birth_per_month'],
                               'anc': [mod['early_initiation_anc4'], 'odds_early_init_anc4'],
                               'sa': [mod['spontaneous_abortion'], 'prob_spontaneous_abortion_per_month'],
                               'ptb': [mod['early_onset_labour'], 'baseline_prob_early_labour_onset']}

        # Scale all models updating the parameter used as the intercept of the linear models
        for k in models_to_be_scaled:
            self.scale_linear_model_at_initialisation(models_to_be_scaled[k][0], models_to_be_scaled[k][1])

    def scale_linear_model_at_initialisation(self, model, parameter_key):
        """
        This function scales the intercept value of linear models according to the distribution of predictor values
        within the data frame. The parameter value (intercept of the model) is then updated accordingly
        :param model: model object to be scaled
        :param parameter_key: key (str) relating to the parameter which holds the target rate for the model
        :return:
        """

        df = self.sim.population.props
        params = self.current_parameters

        # Create a function that runs a linear model with an intercept of 1 and generates a scaled intercept
        def return_scaled_intercept(target, logistic_model):
            mean = model.predict(
                df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)],
                year=self.sim.date.year).mean()

            if logistic_model:
                mean = mean / (1 - mean)

            scaled_intercept = 1.0 * (target / mean) if (target != 0 and mean != 0 and ~np.isnan(mean)) else 1.0
            return scaled_intercept

        # The target value is stored within the parameter
        target = params[parameter_key]

        if (parameter_key != 'prob_spontaneous_abortion_per_month') and (parameter_key != 'baseline_prob_early_'
                                                                                          'labour_onset'):

            # Override the intercept parameter with a value of one
            params[parameter_key] = 1

            # Function varies depending on the input/output of the model (i.e. if logistic)
            if parameter_key == 'odds_early_init_anc4':
                params[parameter_key] = return_scaled_intercept(target, logistic_model=True)
            else:
                params[parameter_key] = return_scaled_intercept(target, logistic_model=False)

        else:
            # These models use predictor values dependent on gestational age to replace the intercept value
            # (see pregnancy_supervisor_lm.py) so we set those predictors to be one
            if parameter_key == 'prob_spontaneous_abortion_per_month':
                params[parameter_key] = [1, 1, 1, 1, 1]
            elif parameter_key == 'baseline_prob_early_labour_onset':
                params[parameter_key] = [1, 1, 1, 1]

            # And scale the values accordingly
            scaled_intercepts = list()
            for item in target:
                intercept = return_scaled_intercept(item, logistic_model=False)
                scaled_intercepts.append(intercept)

            params[parameter_key] = scaled_intercepts

    def on_birth(self, mother_id, child_id):
        df = self.sim.population.props

        df.at[child_id, 'ps_gestational_age_in_weeks'] = 0
        df.at[child_id, 'ps_date_of_anc1'] = pd.NaT
        df.at[child_id, 'ps_ectopic_pregnancy'] = 'none'
        df.at[child_id, 'ps_placenta_praevia'] = False
        df.at[child_id, 'ps_multiple_pregnancy'] = False
        df.at[child_id, 'ps_syphilis'] = False
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
        This function is called by the on_birth function of NewbornOutcomes module or following an intrapartum
        stillbirth in the Labour Module. This function contains additional code related to the pregnancy supervisor
        module that should be ran on_birth. These additional on_birth functions ensure each modules
        (pregnancy,antenatal care, labour, newborn, postnatal) on_birth code is ran in the correct sequence
        (as this can vary depending on how modules are registered)
        :param mother_id: mothers individual id
        """
        df = self.sim.population.props
        mni = self.mother_and_newborn_info

        if df.at[mother_id, 'is_alive']:

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
        """
        This function calculates and reports the monthly daly weight values accumulated from Maternal Disorders. For
        simplicity all daly weights from Maternal Disorders are reported in this module (though may be attributable to
        conditions occurring antenatally, intrapartum or postnatally). Individual level monthly-daly weights are
        calculated using the mni dictionary where date of complication onset/resolution is stored.
        :return: daly_series
        """

        df = self.sim.population.props
        p = self.parameters['ps_daly_weights']
        mni = self.mother_and_newborn_info

        logger.debug(key='message', data='This is PregnancySupervisor reporting my health values')
        monthly_daly = dict()

        # First we define a function that calculates disability associated with 'acute' complications of pregnancy
        def acute_daly_calculation(person, complication):
            # If the woman has not experience the complication of interest in the past month she does not accrue dalys
            if pd.isnull(mni[person][f'{complication}_onset']):
                return

            # If the complication has onset within the last month...
            elif (self.sim.date - DateOffset(months=1)) <= mni[person][f'{complication}_onset'] <= self.sim.date:

                # We assume that any woman who experiences an acute event receives the whole weight for that daly
                monthly_daly[person] += p[f'{complication}']

                # Ensure some weight is assigned
                if mni[person][f'{complication}_onset'] != self.sim.date:
                    assert monthly_daly[person] > 0

                # Reset the variable within the mni dictionary to prevent double counting
                mni[person][f'{complication}_onset'] = pd.NaT

        # Next we define a function that calculates disability associated with 'chronic' complications of pregnancy
        def chronic_daly_calculations(person, complication):
            # If the complication hasn't occurred, the function ends
            if pd.isnull(mni[person][f'{complication}_onset']):
                return
            else:
                # If the complication has not yet resolved, and started more than a month ago, the woman gets a
                # months disability
                if pd.isnull(mni[person][f'{complication}_resolution']):
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

                # ensure value doesnt exceed one
                if monthly_daly[person] > 1:
                    monthly_daly[person] = 1

                # delete_mni is used to signify that pregnancy has ended. We delete the mni variable for women whose
                # pregnancy has ended prematurely via this monthly function to allow for daly weights to be calculated
                # for women who are no long pregnant -this check ensures women who are still pregnant do not have the
                # entry in the mni deleted
                if mni[person]['delete_mni'] and (df.at[person, 'is_pregnant'] or
                                                  df.at[person, 'la_is_postpartum'] or
                                                  (df.at[person, 'ps_ectopic_pregnancy'] != 'none')):
                    mni[person]['delete_mni'] = False

                # otherwise the entry can be deleted
                elif mni[person]['delete_mni'] and \
                    ~df.at[person, 'is_pregnant'] and\
                    ~df.at[person, 'la_is_postpartum'] and \
                    (df.at[person, 'ps_ectopic_pregnancy'] == 'none'):
                    del mni[person]

        daly_series = pd.Series(data=0, index=df.index[df.is_alive])
        daly_series[monthly_daly.keys()] = list(monthly_daly.values())

        return daly_series

    def store_dalys_in_mni(self, individual_id, mni_variable):
        """
        This function is called across the maternal health modules and stores onset/resolution dates for complications
        in an indiviudal's MNI dictionary to be used in report_daly_values
        :param individual_id: individual_id
        :param mni_variable: key of mni dict being assigned
        """
        mni = self.mother_and_newborn_info

        # Women no longer in the mni dict cannot accrue disability
        if individual_id not in mni:
            return

        logger.debug(key='msg', data=f'{mni_variable} is being stored for mother {individual_id} on {self.sim.date}')
        mni[individual_id][f'{mni_variable}'] = self.sim.date

    def pregnancy_supervisor_property_reset(self, ind_or_df, id_or_index):
        """
        This function is called when all properties housed in the PregnancySupervisorModule should be reset. For example
        following pregnancy loss
        :param ind_or_df: "individual/data_frame"(STR) whether this function has been called to reset properties for an
         individual row of the DF or a slice of the data frame
        :param id_or_index: pass the function either an individual ID (INT) or index of subset of data frame
        """

        df = self.sim.population.props

        if ind_or_df == 'individual':
            update = df.at
        else:
            update = df.loc

        update[id_or_index, 'ps_gestational_age_in_weeks'] = 0
        update[id_or_index, 'ps_date_of_anc1'] = pd.NaT
        update[id_or_index, 'ps_multiple_pregnancy'] = False
        update[id_or_index, 'ps_placenta_praevia'] = False
        update[id_or_index, 'ps_syphilis'] = False
        update[id_or_index, 'ps_anaemia_in_pregnancy'] = 'none'
        update[id_or_index, 'ps_anc4'] = False
        update[id_or_index, 'ps_htn_disorders'] = 'none'
        update[id_or_index, 'ps_gest_diab'] = 'none'
        update[id_or_index, 'ps_placental_abruption'] = False
        update[id_or_index, 'ps_antepartum_haemorrhage'] = 'none'
        update[id_or_index, 'ps_premature_rupture_of_membranes'] = False
        update[id_or_index, 'ps_chorioamnionitis'] = False
        update[id_or_index, 'ps_emergency_event'] = False

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
        This functions calculates the correct date each woman will attend her first ANC contact and schedules the visit
         for newly pregnant women depending on their predicted month of attendance
        :param anc_month: month of pregnancy that woman will attend ANC 1
        :param individual_id: individual_id
        """
        df = self.sim.population.props
        params = self.current_parameters

        # Define the weeks of each month of pregnancy
        months_min_max = {2: [5, 8], 3: [9, 13], 4: [14, 17], 5: [18, 22], 6: [23, 27], 7: [28, 31], 8: [32, 35],
                          9: [36, 40]}

        # As care seeking is applied at week 8 gestational age, women who seek care within month two must attend within
        # the next week
        if anc_month == 2:
            days_until_anc = self.rng.randint(0, 6)
        else:
            # Otherwise we draw a week between the min max weeks for predicted month of visit, and then a random day
            weeks_of_visit = (self.rng.randint(months_min_max[anc_month][0], months_min_max[anc_month][1]) - 8)
            days_until_anc = (weeks_of_visit * 7) + self.rng.randint(0, 6)

        first_anc_date = self.sim.date + DateOffset(days=days_until_anc)

        # We store that date as a property which is used by the HSI to ensure the event only runs when it should
        df.at[individual_id, 'ps_date_of_anc1'] = first_anc_date

        # We allow for two possible structure of ANC service delivery, focused ANC (4 visits recommended) or 8 contact
        # scheduled (8 visits recommended). This is to perform comparative analysis.

        # Import the HSIs
        from tlo.methods.care_of_women_during_pregnancy import (
            HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareContact,
            HSI_CareOfWomenDuringPregnancy_FocusedANCVisit
        )

        # This parameter determines which service structure will be used. Scheduling of further ANC visits occurs within
        # the first visit
        if params['anc_service_structure'] == 8:
            first_anc_appt = HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareContact(
                self.sim.modules['CareOfWomenDuringPregnancy'], person_id=individual_id)

        elif params['anc_service_structure'] == 4:
            first_anc_appt = HSI_CareOfWomenDuringPregnancy_FocusedANCVisit(
                self.sim.modules['CareOfWomenDuringPregnancy'], person_id=individual_id)

        self.sim.modules['HealthSystem'].schedule_hsi_event(first_anc_appt, priority=0,
                                                            topen=first_anc_date,
                                                            tclose=first_anc_date + DateOffset(days=1))

    def apply_risk_of_spontaneous_abortion(self, gestation_of_interest):
        """
        This function applies risk of spontaneous abortion to a slice of data frame and is called by
        PregnancySupervisorEvent. It calls the do_after_abortion function for women who loose their
        pregnancy.
        :param gestation_of_interest: gestation in weeks
        """
        df = self.sim.population.props

        # We use the apply_linear_model to determine if any women will experience spontaneous miscarriage
        spont_abortion = self.apply_linear_model(
            self.ps_linear_models['spontaneous_abortion'],
            df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == gestation_of_interest) &
                   (df['ps_ectopic_pregnancy'] == 'none') & ~df['hs_is_inpatient']])

        # The do_after_abortion function is called for women who lose their pregnancy. It resets properties, set
        # potential complications and care seeking
        for person in spont_abortion.loc[spont_abortion].index:
            self.do_after_abortion(person, 'spontaneous_abortion')

    def apply_risk_of_induced_abortion(self, gestation_of_interest):
        """
        This function applies risk of induced abortion to a slice of data frame and is called by
        PregnancySupervisorEvent. It calls the do_after_abortion for women who loose their pregnancy.
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
            self.do_after_abortion(person, 'induced_abortion')

    def do_after_abortion(self, individual_id, type_abortion):
        """
        This function is called for all women who experience a spontaneous or induced abortion. The function logs the
        pregnancy loss, resets key variables and determines risk of complication.
        :param individual_id: individual id
        :param type_abortion: STR "induced" or "spontaneous"
        """
        df = self.sim.population.props
        params = self.current_parameters

        # Log the pregnancy loss
        logger.info(key='maternal_complication', data={'person': individual_id,
                                                       'type': f'{type_abortion}',
                                                       'timing': 'antenatal'})

        # This function officially ends a pregnancy through the contraception module (updates 'is_pregnant' and
        # determines post pregnancy contraception)
        self.sim.modules['Contraception'].end_pregnancy(individual_id)

        # Set the delete_mni variable true so after daly weights are calculated the woman is removed from the mni
        self.mother_and_newborn_info[individual_id]['delete_mni'] = True

        # Reset key pregnancy variables across modules
        self.sim.modules['Labour'].reset_due_date(
            ind_or_df='individual', id_or_index=individual_id, new_due_date=pd.NaT)

        self.pregnancy_supervisor_property_reset(
            ind_or_df='individual', id_or_index=individual_id)

        self.sim.modules['CareOfWomenDuringPregnancy'].care_of_women_in_pregnancy_property_reset(
            ind_or_df='individual', id_or_index=individual_id)

        # Now determine if this pregnancy loss will lead to any complications, log the complicated pregnancy loss and
        # call the function which applies risk of each complication
        if type_abortion == 'spontaneous_abortion':
            df.at[individual_id, 'ps_prev_spont_abortion'] = True
            risk_of_complications = params['prob_complicated_sa']

        else:
            risk_of_complications = params['prob_complicated_ia']

        if self.rng.random_sample() < risk_of_complications:
            logger.info(key='maternal_complication', data={'person': individual_id,
                                                           'type': f'complicated_{type_abortion}',
                                                           'timing': 'antenatal'})

            self.apply_risk_of_abortion_complications(individual_id, f'{type_abortion}')

    def apply_risk_of_abortion_complications(self, individual_id, cause):
        """
        This function makes stores the type of complication experience by a woman following abortion.
        :param individual_id: individual_id
        :param cause: 'type' of abortion (spontaneous abortion OR induced abortion) (str)
        """
        params = self.current_parameters

        # We apply a risk of developing specific complications associated with abortion type and store using a bitset
        # property
        if cause == 'induced_abortion':
            if self.rng.random_sample() < params['prob_injury_post_abortion']:
                self.abortion_complications.set([individual_id], 'injury')
                logger.info(key='maternal_complication', data={'person': individual_id,
                                                               'type': f'{cause}_injury',
                                                               'timing': 'antenatal'})

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

        # We assume only women with complicated abortions will experience disability
        self.store_dalys_in_mni(individual_id, 'abortion_onset')

        # Determine if those women will seek care
        self.care_seeking_pregnancy_loss_complications(individual_id, cause='abortion')

        # Schedule possible death
        self.sim.schedule_event(EarlyPregnancyLossDeathEvent(self, individual_id, cause=f'{cause}'),
                                self.sim.date + DateOffset(days=7))

    def apply_risk_of_anaemia(self, gestation_of_interest):
        """
        This function applies risk of anaemia to a slice of the data frame. It is called by PregnancySupervisorEvent
        :param gestation_of_interest: gestation in weeks
        """
        df = self.sim.population.props
        params = self.current_parameters

        # We determine if a subset of pregnant women will become anaemic using a linear model, in which the
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
        # severe disease
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

            # ... and then log new progressed cases
            def log_new_progressed_cases(disease):
                # Find those women who have experience progression
                assess_status_change = (current_status != disease) & (new_status == disease)
                new_onset_disease = assess_status_change[assess_status_change]

                # Set the emergency variable for those who need to seek care, and update the mni dict is appropriate
                if not new_onset_disease.empty:
                    if disease == 'severe_pre_eclamp':
                        df.loc[new_onset_disease.index, 'ps_emergency_event'] = True
                    elif disease == 'eclampsia':
                        df.loc[new_onset_disease.index, 'ps_emergency_event'] = True
                        new_onset_disease.index.to_series().apply(self.store_dalys_in_mni,
                                                                  mni_variable='eclampsia_onset')

                    # And log all of the new onset cases of any hypertensive disease
                    for person in new_onset_disease.index:
                        logger.info(key='maternal_complication', data={'person': person,
                                                                       'type': disease,
                                                                       'timing': 'antenatal'})
                        if disease == 'severe_pre_eclamp':
                            self.mother_and_newborn_info[person]['new_onset_spe'] = True

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

        # Check theres no accidental cross over between these subsets
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

        # Store complication onset and log each new case of APH
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

        # Select women who are at risk of infection post premature rupture of membranes
        risk_of_chorio = df.is_alive & df.is_pregnant & (df.ps_gestational_age_in_weeks == gestation_of_interest) & \
                         (df.ps_ectopic_pregnancy == 'none') & ~df.hs_is_inpatient & ~df.la_currently_in_labour & \
                         df.ps_premature_rupture_of_membranes

        # For those who will develop infection we set the key variables, store onset and log a new case
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

        # We turn the 'delete_mni' key to true- so after the next daly poll this womans entry is deleted, and reset
        # pregnancy status and update contraceptive status
        for person in women.index:
            self.sim.modules['Contraception'].end_pregnancy(person)
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
        mni[individual_id]['delete_mni'] = True
        logger.info(key='antenatal_stillbirth', data={'mother': individual_id})

        # Reset pregnancy and schedule possible update of contraception
        self.sim.modules['Contraception'].end_pregnancy(individual_id)

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
        and if not will experience risk of antenatal stillbirth
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

        # For those who dont seek care we a apply a weekly risk of stillbirth (this function is called weekly for women
        # who are post term)
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
        This function calculates the risk of death for women who have developed complications but have not received
        treatment.  It is called by the PregnancySupervisor Event and HSI_CareOfWomenDuringPregnancy_Maternal
        EmergencyAssessment if care cant be delivered .
        :param individual_id: individual_id
        """

        df = self.sim.population.props
        params = self.current_parameters
        mni = self.mother_and_newborn_info

        mother = df.loc[individual_id]
        causes = list()

        # Check the dataframe for the inidivual in question to find if they have experience any life threatening
        # complications. Store these as a list (allowing for multiple causes to occur each month)
        if mother.ps_antepartum_haemorrhage != 'none':
            causes.append('antepartum_haemorrhage')

        if (mother.ps_htn_disorders == 'severe_pre_eclamp') and mni[individual_id]['new_onset_spe']:
            causes.append('severe_pre_eclampsia')

        if mother.ps_htn_disorders == 'eclampsia':
            causes.append('eclampsia')

        if mother.ps_chorioamnionitis:
            causes.append('antenatal_sepsis')

        # Create a dictionary containing the causes of death a woman is experiencing as the associated CFR
        risks = dict()
        for cause in causes:
            risk = {f'{cause}': params[f'prob_{cause}_death']}
            risks.update(risk)

        # Calculate the total risk of death for all causes that are present
        result = 1
        for cause in risks:
            result *= (1 - risks[cause])

        # If random draw is less that the total risk of death, she will die and the primary cause is then
        # determined
        if self.rng.random_sample() < (1 - result):
            denominator = sum(risks.values())
            probs = list()

            # Cycle over each cause in the dictionary and divide CFR by the sum of the probabilities
            for cause in risks:
                risks[cause] = risks[cause] / denominator
                probs.append(risks[cause])

            # Now use the list of probabilities to conduct a weighted random draw to determine primary cause of death
            cause_of_death = self.rng.choice(causes, p=probs)

            self.sim.modules['Demography'].do_death(individual_id=individual_id, cause=f'{cause_of_death}',
                                                    originating_module=self.sim.modules['PregnancySupervisor'])

            del mni[individual_id]

        else:
            # If a death does not occur we reset the death causing properties (if appropriate)

            if mother.ps_antepartum_haemorrhage != 'none':
                df.at[individual_id, 'ps_antepartum_haemorrhage'] = 'none'

            if (mother.ps_htn_disorders == 'severe_pre_eclamp') and mni[individual_id]['new_onset_spe']:
                mni[individual_id]['new_onset_spe'] = False

            if mother.ps_htn_disorders == 'eclampsia':
                df.at[individual_id, 'ps_htn_disorders'] = 'severe_pre_eclamp'

            if mother.ps_chorioamnionitis:
                df.at[individual_id, 'ps_chorioamnionitis'] = False

    def schedule_first_anc_contact_for_new_pregnancy(self, gestation_of_interest):
        """
        This function is called by the PregnancySupervisorEvent for all pregnant women at 8 weeks gestational age to
         determine if/when they will attend their first ANC visit.
        :param gestation_of_interest: INT used to select women from the data frame at certain gestation
        """
        df = self.sim.population.props
        params = self.current_parameters

        # First we identify all the women predicted to attend ANC, with the first visit occurring before 4 months
        early_initiation_anc4 = self.apply_linear_model(
            self.ps_linear_models['early_initiation_anc4'],
            df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == gestation_of_interest) &
                   (df['ps_ectopic_pregnancy'] == 'none')])

        # Of the women who will not attend ANC4 early, we determine who will attend ANC4 later in pregnancy
        late_initation_anc4 = pd.Series(self.rng.random_sample(
            len(early_initiation_anc4.loc[~early_initiation_anc4])) < params['prob_late_initiation_anc4'],
                                        index=early_initiation_anc4.loc[~early_initiation_anc4].index)

        # Check there are no duplicates
        for v in late_initation_anc4.loc[late_initation_anc4].index:
            assert v not in early_initiation_anc4.loc[early_initiation_anc4].index

        # Update this variable used in the ANC HSIs for scheduling the next visits
        df.loc[early_initiation_anc4.loc[early_initiation_anc4].index, 'ps_anc4'] = True
        df.loc[late_initation_anc4.loc[late_initation_anc4].index, 'ps_anc4'] = True

        # Select any women who are not predicted to attend ANC4
        anc_below_4 = df.is_alive & df.is_pregnant & (df.ps_gestational_age_in_weeks == gestation_of_interest) & \
                     (df.ps_ectopic_pregnancy == 'none') & ~df.ps_anc4

        # See if any of the women who wont attend ANC4 will still attend their first visit early in pregnancy
        early_initiation_anc_below_4 = pd.Series(self.rng.random_sample(len(anc_below_4.loc[anc_below_4]))
                                                 < params['prob_early_initiation_anc_below4'],
                                                 index=anc_below_4.loc[anc_below_4].index)

        # Call the functions that schedule the HSIs according to the predicted month of gestation at which each woman
        # will attend her first visit
        def schedulde_early_visit(df_slice):
            for person in df_slice.index:
                random_draw_gest_at_anc = self.rng.choice([2, 3, 4], p=params['prob_anc1_months_2_to_4'])
                self.schedule_anc_one(individual_id=person, anc_month=random_draw_gest_at_anc)

        for slice in [early_initiation_anc4.loc[early_initiation_anc4],
                      early_initiation_anc_below_4.loc[early_initiation_anc_below_4]]:
            schedulde_early_visit(slice)

        def schedulde_late_visit(df_slice):
            for person in df_slice.index:
                random_draw_gest_at_anc = self.rng.choice([5, 6, 7, 8, 9, 10], p=params['prob_anc1_months_5_to_9'])

                # We use month ten to capture women who will never attend ANC during their pregnancy
                if random_draw_gest_at_anc != 10:
                    self.schedule_anc_one(individual_id=person, anc_month=random_draw_gest_at_anc)

        for slice in [late_initation_anc4.loc[late_initation_anc4],
                      early_initiation_anc_below_4.loc[~early_initiation_anc_below_4]]:
            schedulde_late_visit(slice)

    def generate_mother_and_newborn_dictionary_for_individual(self, individual_id):
        """
        This function is called at the initialisation of all women's pregnancy (and within testing files) to create the
         set of core variables to be stored within the mni dictionary (that are require for pregnancy to run).
        :param individual_id: individual_id
        """
        mni = self.mother_and_newborn_info
        df = self.sim.population.props

        assert df.at[individual_id, 'is_pregnant']

        mni[individual_id] = {'delete_mni': False,  # if True, mni deleted in report_daly_values function
                              'abortion_onset': pd.NaT,
                              'abortion_haem_onset': pd.NaT,
                              'abortion_sep_onset': pd.NaT,
                              'eclampsia_onset': pd.NaT,
                              'mild_mod_aph_onset': pd.NaT,
                              'severe_aph_onset': pd.NaT,
                              'chorio_onset': pd.NaT,
                              'chorio_in_preg': False, # use in predictor in newborn linear models
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
                              'pred_syph_infect': pd.NaT,  # date syphilis is predicted to onset
                              'new_onset_spe': False,
                              'cs_indication': 'none'
                              }


class PregnancySupervisorEvent(RegularEvent, PopulationScopeEventMixin):
    """ This is the PregnancySupervisorEvent, it is a weekly event which has four primary functions.
    1.) It updates the gestational age (in weeks) of all women who are pregnant
    2.) It applies monthly risk of key complications associated with pregnancy
    3.) It determines if women who experience life seeking complications associated with pregnancy will seek care
    4.) It applies risk of death and stillbirth to women who do not seek care following complications"""
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
        # (this simulates time period prior to which symptoms onset- and may trigger care seeking)
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

        # ------------------------ APPLY RISK OF ADDITIONAL PREGNANCY COMPLICATIONS -----------------------------------
        # The following functions apply risk of key complications/outcomes of pregnancy at specific time points of a
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

        # Next, at 8 weeks gestation, we determine if/when women will seek antenatal care
        for gestation_of_interest in [8]:
            self.module.schedule_first_anc_contact_for_new_pregnancy(gestation_of_interest=gestation_of_interest)

        # Every month a risk of maternal anaemia is applied
        for gestation_of_interest in [4, 8, 13, 17, 22, 27, 31, 35, 40]:
            self.module.apply_risk_of_anaemia(gestation_of_interest=gestation_of_interest)

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
        # Next we apply a background risk of antenatal stillbirth...
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

        if (
            ~df.at[individual_id, 'is_alive'] or
            (df.at[individual_id, 'ps_ectopic_pregnancy'] != 'not_ruptured') or
            (df.at[individual_id, 'ps_gestational_age_in_weeks'] < 9)
        ):
            return

        # reset pregnancy variables and store onset for daly calculation
        self.sim.modules['Contraception'].end_pregnancy(individual_id)
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

        if not df.at[individual_id, 'is_alive'] or (df.at[individual_id, 'ps_ectopic_pregnancy'] != 'not_ruptured'):
            return

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
            if mother.ac_gest_diab_on_treatment != 'none':
                return
            else:
                # We apply a probability that the treatment this woman is receiving for her GDM (diet and exercise/
                # oral anti-diabetics/ insulin) will not control this womans hyperglycaemia
                if self.module.rng.random_sample() > params[f'prob_glycaemic_' \
                                                            f'control_{mother.ac_gest_diab_on_treatment }']:

                    # If so we reset her diabetes status as uncontrolled, her treatment is ineffective at reducing
                    # risk of still birth, and when she returns for follow up she should be started on the next
                    # treatment available
                    df.at[individual_id, 'ps_gest_diab'] = 'uncontrolled'


class SyphilisInPregnancyEvent(Event, IndividualScopeEventMixin):
    """
    This is SyphilisInPregnancyEvent. It is scheduled by PregnancySupervisorEvent module after a
    woman becomes pregnant and is predicted to experience syphilis during their pregnancy. This event onsets Syphilis
    in those women """

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

        # Create a function to update the current_params dictionary owned by each of the maternal and newborn health
        # modules. The function sets the parameter to the second value in a list of two from the modules parameters
        def switch_parameters(master_params, current_params):
            for key, value in master_params.items():
                if type(value) is list:
                    current_params[key] = master_params[key][1]
                else:
                    current_params[key] = master_params[key]

        # Call the function for each of the maternal and newborn modules
        switch_parameters(self.module.parameters, self.module.current_parameters)

        switch_parameters(self.sim.modules['CareOfWomenDuringPregnancy'].parameters,
                          self.sim.modules['CareOfWomenDuringPregnancy'].current_parameters)

        switch_parameters(self.sim.modules['Labour'].parameters, self.sim.modules['Labour'].current_parameters)

        switch_parameters(self.sim.modules['NewbornOutcomes'].parameters,
                          self.sim.modules['NewbornOutcomes'].current_parameters)

        switch_parameters(self.sim.modules['PostnatalSupervisor'].parameters,
                          self.sim.modules['PostnatalSupervisor'].current_parameters)

        # scale the linear models again according to the distribution of the population
        mod = self.module.ps_linear_models

        models_to_be_scaled = {'pp': [mod['placenta_praevia'], 'prob_placenta_praevia'],
                               'an': [mod['maternal_anaemia'], 'baseline_prob_anaemia_per_month'],
                               'gd': [mod['gest_diab'], 'prob_gest_diab_per_month'],
                               'gh': [mod['gest_htn'], 'prob_gest_htn_per_month'],
                               'pe': [mod['pre_eclampsia'], 'prob_pre_eclampsia_per_month'],
                               'pa': [mod['placental_abruption'], 'prob_placental_abruption_per_month'],
                               'ansb': [mod['antenatal_stillbirth'], 'prob_still_birth_per_month'],
                               'anc': [mod['early_initiation_anc4'], 'odds_early_init_anc4'],
                               'sa': [mod['spontaneous_abortion'], 'prob_spontaneous_abortion_per_month'],
                               'ptb': [mod['early_onset_labour'], 'baseline_prob_early_labour_onset']}

        for k in models_to_be_scaled:
            self.module.scale_linear_model_at_initialisation(models_to_be_scaled[k][0], models_to_be_scaled[k][1])


class OverrideKeyParameterForAnalysis(Event, PopulationScopeEventMixin):
    """
    This is OverrideKeyParameterForAnalysis. This event is scheduled in initialise_simulation and allows for a parameter
     value/values to be overridden at a set time point within a simulation run.
    """
    def __init__(self, module):
        super().__init__(module)

    def apply(self, population):
        params = self.module.current_parameters
        df = self.sim.population.props

        # When this parameter is set as True, the following paramters are overridden when the event is called. Otherwise
        # no parameters are updated.
        if params['switch_anc_coverage']:
            target = params['target_anc_coverage_for_analysis']
            params['odds_early_init_anc4'] = 1
            mean = self.module.ps_linear_modek['early_initiation_anc4'].predict(
                    df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)],
                    year=self.sim.date.year).mean()

            mean = mean / (1 - mean)
            scaled_intercept = 1.0 * (target / mean) if (target != 0 and mean != 0 and ~np.isnan(mean)) else 1.0
            params['odds_early_init_anc4'] = scaled_intercept


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
