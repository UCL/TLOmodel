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
    """This module is responsible for supervision of pregnancy in the population including incidence of ectopic
    pregnancy, multiple pregnancy, spontaneous abortion, induced abortion, and onset of maternal diseases associated
    with pregnancy (anaemia, sepsis hypertenisve disorders and gestational diabetes). This module also
    applies the risk of antenatal death and stillbirth. The module is currently unfinished."""

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
        'baseline_prob_early_labour_onset': Parameter(
            Types.REAL, 'monthly baseline risk of early labour onset'),
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
        'prob_antenatal_death_per_month': Parameter(
            Types.REAL, 'underlying risk of antenatal maternal death per month without the impact of risk factors'),
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
        'prob_any_complication_induced_abortion': Parameter(
            Types.REAL, 'probability of a woman that undergoes an induced abortion experiencing any complications'),
        'prob_any_complication_spontaneous_abortion': Parameter(
            Types.REAL, 'probability of a woman that experiences a late miscarriage experiencing any complications'),
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
                                            categories=['none','mild', 'moderate', 'severe']),
        'ps_will_attend_four_or_more_anc': Property(Types.BOOL, 'Whether this womans is predicted to attend 4 or more '
                                                                'antenatal care visits during her pregnancy'),
        'ps_will_attend_eight_or_more_anc': Property(Types.BOOL, 'DUMMY'),  # todo: remove? 
        'ps_induced_abortion_complication': Property(Types.CATEGORICAL, 'severity of complications faced following '
                                                                        'induced abortion',
                                                     categories=['none', 'mild', 'moderate', 'severe']),
        'ps_spontaneous_abortion_complication': Property(Types.CATEGORICAL, 'severity of complications faced following '
                                                                            'induced abortion',
                                                         categories=['none', 'mild', 'moderate', 'severe']),
        'ps_antepartum_still_birth': Property(Types.BOOL, 'whether this woman has experienced an antepartum still birth'
                                                          'of her current pregnancy'),
        'ps_previous_stillbirth': Property(Types.BOOL, 'whether this woman has had any previous pregnancies end in '
                                                       'still birth'),  # consider if this should be an interger
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
        'ps_antepartum_haemorrhage_severity': Property(Types.CATEGORICAL,'severity of this womans antepartum '
                                                                         'haemorrhage',
                                                       categories=['none', 'mild_moderate', 'severe']),
        'ps_premature_rupture_of_membranes': Property(Types.BOOL, 'whether this woman has experience rupture of '
                                                                  'membranes before the onset of labour. If this is '
                                                                  '<37 weeks from gestation the woman has preterm '
                                                                  'premature rupture of membranes'),

        'ps_chorioamnionitis': Property(Types.BOOL, 'whether this woman has chorioamnionitis following PROM'),
        'dummy_anc_counter': Property(Types.INT, 'DUMMY'),  # TODO: remove 
        'ps_will_attend_3_early_visits': Property(Types.BOOL, 'DUMMY'),  # TODO: remove
        'ps_emergency_event': Property(Types.BOOL, 'signifies a woman in undergoing an acute emergency event in her '
                                                   'pregnancy- used to consolidated care seeking in the instance of '
                                                   'multiple complications')
    }

    def read_parameters(self, data_folder):

        params = self.parameters
        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_PregnancySupervisor.xlsx',
                            sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)

        # TODO: DALYs are not finalised/captured fully yet from this model awaiting updates
        if 'HealthBurden' in self.sim.modules.keys():
            params['daly_wt_abortive_outcome'] = self.sim.modules['HealthBurden'].get_daly_weight(352)

    # ==================================== LINEAR MODEL EQUATIONS =====================================================
        # All linear equations used in this module are stored within the ps_linear_equations parameter below
        # (predictors not yet included)
        # TODO: predictors not yet included/finalised

        params['ps_linear_equations'] = {
                'ectopic': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_ectopic_pregnancy']),

                'multiples': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_multiples']),

                'placenta_praevia': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_placenta_praevia']),

                'spontaneous_abortion': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_spontaneous_abortion_per_month']),

                'induced_abortion': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_induced_abortion_per_month']),

                'early_onset_labour': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['baseline_prob_early_labour_onset']),
                # TODO: PROM occurring in the month needs to modify risk here

                #  'abortion_complication_severity': LinearModel(
                #    LinearModelType.MULTIPLICATIVE,
                #    params['prob_moderate_or_severe_abortion_comps']),
                # todo: TAKE FULL EQUATION FROM Kalilani-PhirI ET AL. The severity of abortion complications in
                #  malawi

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

                #    Predictor('ac_receiving_iron_folic_acid').when(True, params['rr_anaemia_iron_folic_acid'])),
                #   Predictor('ac_date_ifa_runs_out').when(True, params['rr_anaemia_iron_folic_acid'])),

                # TODO: i struggled to work out where i needed to write the function that would work out the date for
                #  this predictor. The treatment effect 'rr_anaemia_iron_folic_acid' should only work if self.sim.date
                #  is prior to 'ac_date_ifa_runs_out' when this equation is called

                'pre_eclampsia': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_pre_eclampsia_per_month'],
                    Predictor('ac_receiving_calcium_supplements').when(True, params['rr_pre_eclamp_calcium'])),

                'gest_htn': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_gest_htn_per_month'],
                    Predictor('ac_receiving_calcium_supplements').when(True, params['rr_gest_htn_calcium'])),

                'gest_diab': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_gest_diab_per_month']),

                'placental_abruption': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_placental_abruption_per_month']),

                'antepartum_haem': LinearModel(
                    LinearModelType.ADDITIVE,
                    0,
                    Predictor('ps_placenta_praevia').when(True, params['prob_aph_placenta_praevia']),
                    Predictor('ps_placental_abruption').when(True, params['prob_aph_placental_abruption']),
                ),
                #  todo: not sure its mathematically right to use an additive model?

                'premature_rupture_of_membranes': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_prom_per_month']),

                'chorioamnionitis': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_chorioamnionitis_post_prom']),

                'antenatal_stillbirth': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_still_birth_per_month'],
                    Predictor('ac_receiving_diet_supplements').when(True, params['rr_still_birth_food_supps'])),

                'antenatal_death': LinearModel(
                    LinearModelType.ADDITIVE,
                    0,
                    Predictor('ps_htn_disorders').when('gest_htn', params['monthly_cfr_gest_htn'])
                                                 .when('severe_gest_htn', params['monthly_cfr_severe_gest_htn'])
                                                 .when('mild_pre_eclamp', params['monthly_cfr_mild_pre_eclamp'])
                                                 .when('severe_pre_eclamp', params['monthly_cfr_severe_pre_eclamp']),
                    Predictor('ps_gest_diab').when(True, params['monthly_cfr_gest_diab']),
                    Predictor('ps_anaemia_in_pregnancy').when('severe', params['monthly_cfr_severe_anaemia'])),

                # TODO: Will add treatment effects as appropriate
                # TODO: HIV/TB/Malaria to be added as predictors to generate HIV/TB associated maternal deaths
                #  (will need to occure postnatally to)
                # TODO: other antenatal factors - anaemia (proxy for haemorrhage?)

                'ectopic_pregnancy_death': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_ectopic_pregnancy_death'],
                    Predictor('ac_ectopic_pregnancy_treated').when(True, params['treatment_effect_ectopic_pregnancy'])),

                'induced_abortion_death': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_induced_abortion_death'],
                    Predictor('ac_post_abortion_care_interventions').when('>0',
                                                                          params[
                                                                              'treatment_effect_post_abortion_care'])),

                'spontaneous_abortion_death': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_spontaneous_abortion_death'],
                    Predictor('ac_post_abortion_care_interventions').when('>0',
                                                                          params[
                                                                              'treatment_effect_post_abortion_care'])),

                # TODO: both of the above death models need to vary by severity
                # TODO: both of the above death models need to vary treatment effect by specific treatment,
                #  this is just any treatment

                'antepartum_haemorrhage_stillbirth': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_antepartum_haem_stillbirth']),

                'antepartum_haemorrhage_death': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_antepartum_haem_death']),

                'severe_pre_eclamp_death': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_antenatal_spe_death'],
                    Predictor('ac_severe_pre_eclampsia_treatment').when(True, params['treatment_effect_'
                                                                                     'severe_pre_eclampsia'])),

                'eclampsia_death': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_antenatal_ec_death'],
                    Predictor('ac_eclampsia_treatment').when(True, params['treatment_effect_eclampsia'])),

                'eclampsia_still_birth': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_antenatal_ec_still_birth']),

                'care_seeking_pregnancy_loss': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_seek_care_pregnancy_loss']),

                'care_seeking_pregnancy_complication': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_seek_care_pregnancy_complication']),

                # TODO: severity of bleed should be a predictor for both of the 2 equations

                'four_or_more_anc_visits': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_four_or_more_anc_visits']),

                'eight_or_more_anc_visits': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_eight_or_more_anc_visits']),

        }

        # Create bitset handler for these two columns so they can be used as lists later on

    def initialise_population(self, population):

        df = population.props

        df.loc[df.is_alive, 'ps_gestational_age_in_weeks'] = 0
        df.loc[df.is_alive, 'ps_ectopic_pregnancy'] = False
        df.loc[df.is_alive, 'ps_placenta_praevia'] = False
        df.loc[df.is_alive, 'ps_multiple_pregnancy'] = False
        df.loc[df.is_alive, 'ps_anaemia_in_pregnancy'] = 'none'
        df.loc[df.is_alive, 'ps_will_attend_four_or_more_anc'] = False
        df.loc[df.is_alive, 'ps_will_attend_eight_or_more_anc'] = False
        df.loc[df.is_alive, 'ps_induced_abortion_complication'] = 'none'
        df.loc[df.is_alive, 'ps_spontaneous_abortion_complication'] = 'none'
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
        df.loc[df.is_alive, 'dummy_anc_counter'] = 0
        df.loc[df.is_alive, 'ps_will_attend_3_early_visits'] = False
        df.loc[df.is_alive, 'ps_emergency_event'] = False

        self.deficiencies_in_pregnancy = BitsetHandler(self.sim.population, 'ps_deficiencies_in_pregnancy',
                                                        ['iron', 'b12', 'folate'])

    def initialise_simulation(self, sim):

        sim.schedule_event(PregnancySupervisorEvent(self),
                           sim.date + DateOffset(days=0))

        sim.schedule_event(PregnancyLoggingEvent(self),
                           sim.date + DateOffset(years=1))

        # Define the conditions we want to track
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
        params = self.parameters

        df.at[child_id, 'ps_gestational_age_in_weeks'] = 0
        df.at[child_id, 'ps_ectopic_pregnancy'] = False
        df.at[child_id, 'ps_placenta_praevia'] = False
        df.at[child_id, 'ps_multiple_pregnancy'] = False
        df.at[child_id, 'ps_anaemia_in_pregnancy'] = 'none'
        df.at[child_id, 'ps_will_attend_four_or_more_anc'] = False
        df.at[child_id, 'ps_induced_abortion_complication'] = 'none'
        df.at[child_id, 'ps_spontaneous_abortion_complication'] = 'none'
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
        df.at[child_id, 'dummy_anc_counter'] = 0
        df.at[child_id, 'ps_will_attend_3_early_visits'] = False
        df.at[child_id, 'ps_emergency_event'] = False

        # We reset all womans gestational age when they deliver
        df.at[mother_id, 'ps_gestational_age_in_weeks'] = 0

        # =========================================== RESET GDM STATUS ================================================
        # We assume that hyperglycaemia from gestational diabetes resolves following birth
        df.at[mother_id, 'ps_gest_diab'] = False
        # TODO: link with future T2DM


    def on_hsi_alert(self, person_id, treatment_id):
        logger.debug(key='message', data='This is PregnancySupervisor, being alerted about a health system interaction '
                                         f'person {person_id} for: {treatment_id}')

    def report_daly_values(self):
        df = self.sim.population.props

        # TODO: Dummy code, waiting for new DALY set up
        logger.debug(key='message', data='This is PregnancySupervisor reporting my health values')

        health_values_1 = df.loc[df.is_alive, 'ps_ectopic_pregnancy'].map(
            {False: 0, True: 0.2})
        health_values_1.name = 'Ectopic Pregnancy'

        health_values_df = health_values_1
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
        # TODO: have this function live in one module and call from other modules (not copied)

        return self.rng.random_sample(len(df)) < lm.predict(df)

    def set_deficiencies_and_anaemia_status(self, ga_in_weeks):
        """This function applies risk of deficiencies and anaemia to a slice of the dataframe. It is abstracted to a
        function to prevent repeats in the set_pregnancy_complication function'"""
        df = self.sim.population.props
        params = self.parameters

        # todo: reset at end of pregnancy
        # todo: should women who have had deficiencies correct as an inpatient, and therefore are on treatment,
        #  have reduced risk of reoccuring deficiency (course is whole of pregnancy)
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

        # ------------------------------------------ ANAEMIA ---------------------------------------------------------
        # Now we determine if a subset of pregnant women will become anaemic using a linear model, in which the
        # preceding deficiencies act as predictors
        anaemia = self.apply_linear_model(
            params['ps_linear_equations']['maternal_anaemia'],
            df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == ga_in_weeks) &
                   ~df['ps_ectopic_pregnancy'] & ~df['ac_inpatient'] & ~df['la_currently_in_labour']])

        # We apply a random risk of severity
        # TODO: parameterise and repalce
        # TODO: should a second linear equation determine severity (or use some kind of multinomal regression)

        random_choice_severity = pd.Series(self.rng.choice(['mild', 'moderate', 'severe'], p=[0.33, 0.33, 0.34],
                                                           size=len(anaemia.loc[anaemia])),
                                           index=anaemia.loc[anaemia].index)

        df.loc[anaemia.loc[anaemia].index, 'ps_anaemia_in_pregnancy'] = random_choice_severity

        if not anaemia.loc[anaemia].empty:
            logger.debug(key='message', data=f'The following women have developed anaemia during week {ga_in_weeks}'
                                             f' of the postnatal period, {anaemia.loc[anaemia]}')
            self.pregnancy_disease_tracker['maternal_anaemia'] += len(anaemia.loc[anaemia])

    def set_pregnancy_complications(self, ga_in_weeks):
        """"""
        df = self.sim.population.props
        params = self.parameters

        # ======================================= FIRST WEEK OF PREGNANCY ============================================
        if ga_in_weeks == 1:
            #  --------------------------------- RISK OF ECTOPIC PREGNANCY -------------------------------------------
            ectopic_risk = self.apply_linear_model(
                params['ps_linear_equations']['ectopic'],
                df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == ga_in_weeks)])

            df.loc[ectopic_risk.loc[ectopic_risk].index, 'ps_ectopic_pregnancy'] = True
            self.pregnancy_disease_tracker['ectopic_pregnancy'] += len(ectopic_risk.loc[ectopic_risk])

            if not ectopic_risk.loc[ectopic_risk].empty:
                logger.debug(key='message', data=f'The following women have experience an ectopic pregnancy,'
                                                 f'{ectopic_risk.loc[ectopic_risk].index}')

            for person in ectopic_risk.loc[ectopic_risk].index:
                self.sim.schedule_event(EctopicPregnancyEvent(self, person),
                                        (self.sim.date + pd.Timedelta(days=7 * 4 + self.rng.randint(0, 7 * 2))))

            #  --------------------------------- RISK OF MULTIPLE PREGNANCY -------------------------------------------
            multiples = self.apply_linear_model(
                params['ps_linear_equations']['multiples'],
                df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == ga_in_weeks) &
                       ~df['ps_ectopic_pregnancy']])

            df.loc[multiples.loc[multiples].index, 'ps_multiple_pregnancy'] = True
            self.pregnancy_disease_tracker['multiples'] += len(multiples.loc[multiples])

            if not multiples.loc[multiples].empty:
                logger.debug(key='message', data=f'The following women are pregnant with multiples, '
                                                 f'{multiples.loc[multiples].index}')

            #  --------------------------------- RISK OF PLACENTA PRAEVIA  -------------------------------------------
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
            # TODO: In the final equation we will assume women dont start attending until it is realistic that they are
            #  aware they are pregnant. The current probabilities are dummys (have requested data from author of study
            #  for whom this equation is based on)
            # TODO: need to calibrate to ensure that 95% attend 1 ANC

            #    will_attend_anc4 = params['ps_linear_equations']['four_or_more_anc_visits'].predict(df.loc[[person]])[
            #        person]
            #   if self.module.rng.random_sample() < will_attend_anc4:
            #        df.at[person, 'ps_will_attend_four_or_more_anc'] = True
            #     will_attend_anc8 = params['ps_linear_equations']['eight_or_more_anc_visits'].predict(
            #     df.loc[[person]])[person]

            women_pregnancy_continues = df.loc[
                df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == ga_in_weeks) &
                ~df['ps_ectopic_pregnancy']]

            early_anc_3 = pd.Series(self.rng.random_sample(len(women_pregnancy_continues)) <
                                    params['prob_3_early_visits'], index=women_pregnancy_continues.index)

            df.loc[early_anc_3.loc[early_anc_3].index, 'ps_will_attend_3_early_visits'] = 'none'

            for person in early_anc_3.loc[early_anc_3].index:
                first_anc_appt = HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareContact(
                    self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person)
                first_anc_date = self.sim.date + DateOffset(months=2)
                self.sim.modules['HealthSystem'].schedule_hsi_event(first_anc_appt, priority=0,
                                                                    topen=first_anc_date,
                                                                    tclose=first_anc_date + DateOffset(days=7))
            for person in early_anc_3.loc[~early_anc_3].index:
                random_draw_gest_at_anc = self.rng.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                                 p=params['prob_first_anc_visit_gestational_age'])

                first_anc_date = self.sim.date + DateOffset(months=random_draw_gest_at_anc)
                first_anc_appt = HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareContact(
                    self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person)

                self.sim.modules['HealthSystem'].schedule_hsi_event(first_anc_appt, priority=0,
                                                                    topen=first_anc_date,
                                                                    tclose=first_anc_date + DateOffset(days=7))

        # ======================================= MONTHS 1-4 OF PREGNANCY ============================================
        elif ga_in_weeks == 4 or ga_in_weeks == 8 or ga_in_weeks == 13 or ga_in_weeks == 17:

            #  ----------------------------- RISK OF SPONTANEOUS ABORTION ---------------------------------------------
            miscarriage = self.apply_linear_model(
                params['ps_linear_equations']['spontaneous_abortion'],
                df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == ga_in_weeks) &
                       ~df['ps_ectopic_pregnancy'] & ~df['ac_inpatient']])

            for person in miscarriage.loc[miscarriage].index:
                self.abortion(person, 'spontaneous_abortion')

            #  -------------------------------- RISK OF INDUCED ABORTION ----------------------------------------------
            if not ga_in_weeks == 4:
                abortion = self.apply_linear_model(
                    params['ps_linear_equations']['induced_abortion'],
                    df.loc[df['is_alive'] & df['is_pregnant'] & (df['ps_gestational_age_in_weeks'] == ga_in_weeks) &
                           ~df['ps_ectopic_pregnancy'] & ~df['ac_inpatient']])

                for person in abortion.loc[abortion].index:
                    self.abortion(person, 'induced_abortion')

            #  ------------------------ RISK OF NUTRITIONAL DEFICIENCIES AND ANAEMIA ----------------------------------
            self.set_deficiencies_and_anaemia_status(ga_in_weeks)

        # ======================================= MONTHS 5-9 OF PREGNANCY ============================================
        elif ga_in_weeks == 22 or ga_in_weeks == 27 or ga_in_weeks == 31 or ga_in_weeks == 35 or ga_in_weeks == 40:

            #  ----------------------------- RISK OF SPONTANEOUS ABORTION ---------------------------------------------
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

            #  ------------------------ RISK OF NUTRITIONAL DEFICIENCIES AND ANAEMIA ----------------------------------
            self.set_deficiencies_and_anaemia_status(ga_in_weeks)

            #  ----------------------------------- RISK OF PRE-ECLAMPSIA ----------------------------------------------
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
                                                                          cause='antenatal_emergency'), self.sim.date)
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
        elif ga_in_weeks == 41 or ga_in_weeks == 42 or ga_in_weeks == 43 or ga_in_weeks == 44 or ga_in_weeks == 45:
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

            # For women who miscarry after 13 weeks or induce an abortion at any gestation we apply a probability that
            # this woman will experience complications from this pregnancy loss
            if (cause == 'spontaneous_abortion' and df.at[individual_id, 'ps_gestational_age_in_weeks'] >= 13) or \
                cause == 'induced_abortion':
                if params[f'prob_any_complication_{cause}'] < self.rng.random_sample():
                    # We categorise complications as mild, moderate or severe mapping to data from a Malawian study
                    # (categories are used to determine treatment in Post Abortion Care)

                    # TODO: replace with LM equation taken from Kalilani-Phiri et al. paper.
                    severity = ['mild', 'moderate', 'severe']
                    probabilities = [0.724, 0.068, 0.208]
                    random_draw = self.rng.choice(severity, p=probabilities)
                    df.at[individual_id, f'ps_{cause}_complication'] = random_draw

                    # Determine if this woman will seek care, and schedule presentation to the health system
                    self.care_seeking_pregnancy_loss_complications(individual_id)

                    # Schedule possible death
                    self.sim.schedule_event(EarlyPregnancyLossDeathEvent(self, individual_id,
                                                                         cause=f'{cause}'),
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
                                                                      cause=f'{self.cause}'), self.sim.date)
                self.module.pregnancy_disease_tracker[f'{self.cause}_death'] += 1
                self.module.pregnancy_disease_tracker['antenatal_death'] += 1

            # Here we remove treatments from a woman who has survived her post abortion complications
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
