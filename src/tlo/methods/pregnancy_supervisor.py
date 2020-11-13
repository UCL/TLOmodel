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

    METADATA = {Metadata.DISEASE_MODULE}  # declare that this is a disease module (leave as empty set otherwise)

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
                                            categories=['none', 'non_severe', 'severe']),
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
        self.pregnancy_disease_tracker = {'ectopic_pregnancy': 0, 'induced_abortion': 0, 'spontaneous_abortion': 0,
                                          'ectopic_pregnancy_death': 0, 'induced_abortion_death': 0,
                                          'spontaneous_abortion_death': 0, 'maternal_anaemia': 0, 'antenatal_death': 0,
                                          'antenatal_stillbirth': 0, 'new_onset_pre_eclampsia': 0,
                                          'new_onset_gest_htn': 0, 'antepartum_haem': 0, 'antepartum_haem_death': 0,
                                          'women_at_6_months' :0}

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

        # ========================== RISK OF ONGOING HTN /RESETTING STATUS AFTER BIRTH ================================
        # Here we apply a one of probability that women who have experienced a hypertensive disorder during pregnancy
        # will remain hypertensive after birth into the postnatal period
        if df.at[mother_id, 'ps_htn_disorders'] == 'gest_htn' or 'severe_gest_htn' or 'mild_pre_eclamp' or \
                                                   'severe_pre_eclamp':
            if self.rng.random_sample() < params['probability_htn_persists']:
                logger.debug(key='message', data=f'mother {mother_id} will remain hypertensive despite successfully '
                                                 f'delivering')
            else:
                df.at[mother_id, 'ps_htn_disorders'] = 'none'

        # ================================= RISK OF DE NOVO HTN =======================================================
        # Finally we apply a risk of de novo hypertension in women who have been normatensive during pregnancy

        risk_of_gest_htn = params['ps_linear_equations']['gest_htn'].predict(df.loc[[mother_id]])[mother_id]
        risk_of_mild_pre_eclampsia = params['ps_linear_equations']['pre_eclampsia'].predict(df.loc[[mother_id]])[
            mother_id]

        if self.rng.random_sample() < risk_of_gest_htn:
            df.at[mother_id, 'ps_htn_disorders'] = 'gest_htn'
        else:
            if self.rng.random_sample() < risk_of_mild_pre_eclampsia:
                df.at[mother_id, 'ps_htn_disorders'] = 'mild_pre_eclamp'

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

    def set_pregnancy_complications(self, df_slice, complication):
        """This function is called from within the PregnancySupervisorEvent. It calculates risk of a number of pregnancy
        outcomes/ complications for pregnant women in the data frame using the linear model equations defined above.
        Properties are modified depending on the complication passed to the function and the result of a random draw"""
        df = self.sim.population.props
        params = self.parameters

        # Run checks on women passed to this function
        if not df_slice.empty:
            for person in df_slice.index:
                assert df.at[person, 'is_alive'] and df.at[person, 'is_pregnant'] and df.at[person, 'sex'] == 'F' and \
                       51 > df.at[person, 'age_years'] > 14

        # --------------------------------------- CALCULATION OF RISK -------------------------------------------------
        # We apply the results of the linear model to the index of women in question
        if not complication == 'deficiencies' or complication == 'antepartum_haem':
            result = params['ps_linear_equations'][f'{complication}'].predict(df_slice)

            # And use the result of a random draw to determine which women will experience the complication
            random_draw = pd.Series(self.rng.random_sample(size=len(df_slice)), index=df_slice.index)
            temp_df = pd.concat([result, random_draw], axis=1)
            temp_df.columns = ['result', 'random_draw']

            # Then we use this index to make changes to the data frame and schedule any events required
            positive_index = temp_df.index[temp_df.random_draw < temp_df.result]

        # ------------------------------------------- ECTOPIC --------------------------------------------------------
        # This is done by cycling through each possible complication that can be passed to this function...
        if complication == 'ectopic':
            # Women suffering from an ectopic pregnancy are scheduled to an event at between 4-6 weeks gestation where
            # they may seek care

            df.loc[positive_index, 'ps_ectopic_pregnancy'] = True
            self.pregnancy_disease_tracker['ectopic_pregnancy'] += len(positive_index)
            for person in positive_index:
                self.sim.schedule_event(EctopicPregnancyEvent(self, person),
                                        (self.sim.date + pd.Timedelta(days=7 * 4 + self.rng.randint(0, 7 * 2))))
            if not positive_index.empty:
                logger.debug(key='message', data=f'The following women have experience an ectopic pregnancy,'
                                                 f'{positive_index}')

        # ------------------------------------------- MULTIPLES -------------------------------------------------------
        if complication == 'multiples':
            df.loc[positive_index, 'ps_multiple_pregnancy'] = True
            if not positive_index.empty:
                logger.debug(key='message', data=f'The following women are pregnant with multiples, {positive_index}')

        # --------------------------------------- PLACENTA PRAEVIA ----------------------------------------------------
        if complication == 'placenta_praevia':
            df.loc[positive_index, 'ps_placenta_praevia'] = True
            if not positive_index.empty:
                logger.debug(key='message', data=f'The following womens pregnancy is complicated by placenta praevia '
                                                 f'{positive_index}')

        # --------------------------------------- ABORTION/MISCARRIAGE-------------------------------------------------
        # Women who experience pregnancy loss pass through the abortion function
        if complication == 'spontaneous_abortion' or complication == 'induced_abortion':
            for person in positive_index:
                # TODO: this actually probably doesnt need to be its own function as its only called once
                self.abortion(person, complication)
            if not positive_index.empty:
                logger.debug(key='message', data=f'The following women have experience a {complication}, '
                                                 f'{positive_index}')

        # ----------------------------------- MICRONUTRIENT DEFICIENCIES ----------------------------------------------
        if complication == 'deficiencies':
            # todo: remove for loop
            # todo: reset at end of pregnancy
            # todo: move somewhere else?
            # todo: should women who have had deficiencies correct as an inpatient, and therefore are on treatment,
            #  have reduced risk of reoccuring deficiency (course is whole of pregnancy)

            for person in df_slice.index:
                if not self.deficiencies_in_pregnancy.has_any([person], 'iron', first=True):
                    if df.at[person, 'ac_receiving_iron_folic_acid']:
                        risk_iron_def = params['prob_iron_def_per_month'] * params['rr_iron_def_ifa']
                    else:
                        risk_iron_def = params['prob_iron_def_per_month']
                    if self.rng.random_sample() < risk_iron_def:
                        self.deficiencies_in_pregnancy.set(person, 'iron')
                        logger.debug(key='message', data=f'{person} has become iron deficient')

                if not self.deficiencies_in_pregnancy.has_any([person], 'folate', first=True):
                    if df.at[person, 'ac_receiving_iron_folic_acid']:
                        risk_folate_def = params['prob_folate_def_per_month'] * params['rr_folate_def_ifa']
                    else:
                        risk_folate_def = params['prob_folate_def_per_month']
                    if self.rng.random_sample() < risk_folate_def:
                        self.deficiencies_in_pregnancy.set(person, 'folate')
                        logger.debug(key='message', data=f'{person} has become folate deficient')

                if not self.deficiencies_in_pregnancy.has_any([person], 'b12', first=True):
                    if self.rng.random_sample() < params['prob_b12_def_per_month']:
                        self.deficiencies_in_pregnancy.set(person, 'b12')
                        logger.debug(key='message', data=f'{person} has become B12 deficient')

        # -------------------------------------------- ANAEMIA  -----------------------------------------------------
        # Women with new onset anaemia have that property set
        if complication == 'maternal_anaemia':
            # TODO: should a second linear equation determine severity (or use some kind of multinomal regression)
            if not positive_index.empty:
                random_choice_severity = pd.Series(self.rng.choice(['non_severe', 'severe'], p=[0.5, 0.5],
                                                                   size=len(positive_index)), index=positive_index)
                df.loc[positive_index, 'ps_anaemia_in_pregnancy'] = random_choice_severity
                self.pregnancy_disease_tracker['maternal_anaemia'] += len(positive_index)
                logger.debug(key='message', data=f'The following women have developed anaemia during their pregnancy '
                                                 f'{positive_index}')

        # -------------------------------------------- PROM  ---------------------------------------------------------
        if complication == 'premature_rupture_of_membranes':
            df.loc[positive_index, 'ps_premature_rupture_of_membranes'] = True
            df.loc[positive_index, 'ps_emergency_event'] = True
            if not positive_index.empty:
                logger.debug(key='message', data=f'The following women have experience premature rupture of membranes'
                                                 f'{positive_index}')

        # ----------------------------------------- PRE-ECLAMPSIA  ---------------------------------------------------
        if complication == 'pre_eclampsia':
            # Check only women without current hypertensive disorder can develop hypertension
            for person in positive_index:
                assert df.at[person, 'ps_htn_disorders'] == 'none'

            df.loc[positive_index, 'ps_prev_pre_eclamp'] = True
            df.loc[positive_index, 'ps_htn_disorders'] = 'mild_pre_eclamp'
            self.pregnancy_disease_tracker['new_onset_pre_eclampsia'] += len(positive_index)

            if not positive_index.empty:
                logger.debug(key='message', data=f'The following women have developed pre_eclampsia {positive_index}')

        # ----------------------------------- GESTATIONAL HYPERTENSION  -----------------------------------------------
        if complication == 'gest_htn':
            for person in positive_index:
                assert df.at[person, 'ps_htn_disorders'] == 'none'

            df.loc[positive_index, 'ps_htn_disorders'] = 'gest_htn'
            self.pregnancy_disease_tracker['new_onset_gest_htn'] += len(positive_index)
            if not positive_index.empty:
                logger.debug(key='message', data=f'The following women have developed gestational hypertension'
                                                 f'{positive_index}')

        # ------------------------------------- GESTATIONAL DIABETES  -------------------------------------------------
        # The same process is followed for gestational diabetes
        if complication == 'gest_diab':
            for person in positive_index:
                assert ~df.at[person, 'ps_gest_diab']

            df.loc[positive_index, 'ps_gest_diab'] = True
            df.loc[positive_index, 'ps_prev_gest_diab'] = True
            if not positive_index.empty:
                logger.debug(key='message', data=f'The following women have developed gestational diabetes,'
                                                 f'{positive_index}')

        # ------------------------------------- PLACENTAL ABRUPTION  --------------------------------------------------
        if complication == 'placental_abruption':
            df.loc[positive_index, 'ps_placental_abruption'] = True
            if not positive_index.empty:
                logger.debug(key='message', data=f'The following women have developed placental abruption,'
                                                 f'{positive_index}')

            # todo: this should cause some kind of emergency care seeking/response?

        # ------------------------------------- ANTEPARTUM HAEMORRHAGE  -----------------------------------------------
        if complication == 'antepartum_haem':
            if not positive_index.empty:
                logger.debug(key='message', data=f'The following women are experiencing an antepartum haemorrhage,'
                                                 f'{positive_index}')
                for person in positive_index:
                    random_choice_severity = self.rng.choice(['mild_moderate', 'severe'], p=[0.5, 0.5])

                    # todo: ensure if positive index >1 that choice is truly random
                    df.at[person, 'ps_antepartum_haemorrhage'] = True
                    df.at[person, 'ps_antepartum_haemorrhage_severity'] = random_choice_severity
                    df.at[person, 'ps_emergency_event'] = True

                    assert df.at[person, 'ps_placenta_praevia'] or df.at[person, 'ps_placental_abruption']

                    self.pregnancy_disease_tracker['antepartum_haem'] += len(positive_index)

        # ------------------------------------- ANTEPARTUM DEATH  ----------------------------------------------------
        # TODO: possibly remove (ALTHOUGH MAY BE USEFUL FOR INDIRECT CAUSES OF DEATH ONLY)
        # This function is also used to calculate and apply risk of death and stillbirth to women each month
        if complication == 'antenatal_death':
            self.pregnancy_disease_tracker['antenatal_death'] += len(positive_index)
            for person in positive_index:
                death = demography.InstantaneousDeath(self.sim.modules['Demography'], person,
                                                      cause='antenatal death')
                self.sim.schedule_event(death, self.sim.date)
            if not positive_index.empty:
                logger.debug(key='message', data=f'The following women have died during pregnancy,{positive_index}')

        # ---------------------------------------- STILL BIRTH -------------------------------------------------------
        if complication == 'antenatal_stillbirth':
            self.pregnancy_disease_tracker['antenatal_stillbirth'] += len(positive_index)

            df.loc[positive_index, 'ps_antepartum_still_birth'] = True
            df.loc[positive_index, 'ps_previous_stillbirth'] = True
            df.loc[positive_index, 'is_pregnant'] = False
            df.loc[positive_index, 'la_due_date_current_pregnancy'] = pd.NaT
            df.loc[positive_index, 'ps_gestational_age_in_weeks'] = 0

            if not positive_index.empty:
                logger.debug(key='message', data=f'The following women have have experienced an antepartum'
                                                 f' stillbirth,{positive_index}')

    def preterm_labour_onset(self, df_slice, month):
        """This function applies the a risk of early labour onset during months 5, 6, 7 and 8. Risk is determined via
        the linear model considering a womans risk factors and pre-ceding events that may lead to preterm labour like
        PROM. Women who go into labour early are passed to the labour module, no additional antenatal complications are
        applied."""
        params = self.parameters
        df = self.sim.population.props

        # We determine which women are going to go into labour early this month using prediction from the linear model
        result = params['ps_linear_equations']['early_onset_labour'].predict(df_slice)
        random_draw = pd.Series(self.rng.random_sample(size=len(df_slice)), index=df_slice.index)
        temp_df = pd.concat([result, random_draw], axis=1)
        temp_df.columns = ['result', 'random_draw']
        preterm_labour = temp_df.index[temp_df.random_draw < temp_df.result]

        if not preterm_labour.empty:
            logger.debug(key='message', data=f'The following women will go into preterm labour at some point before the '
                                             f'next month of their pregnancy: {preterm_labour}')

        # To avoid clustering of labour onset each month we randomly select a number of days in the future that this
        # woman will go into early labour, before the next month occurs
        for person in preterm_labour:
            if month == 5:
                poss_day_onset = (27-22) * 7
                # We only allow labour to onset from 24 weeks
                onset_day = self.rng.randint(14, poss_day_onset)
            elif month == 6:
                poss_day_onset = (31-27) * 7
                onset_day = self.rng.randint(0, poss_day_onset)
            elif month == 7:
                poss_day_onset = (35-31) * 7
                onset_day = self.rng.randint(0, poss_day_onset)
            elif month == 8:
                poss_day_onset = (37-35) * 7
                onset_day = self.rng.randint(0, poss_day_onset)

            df.at[person, 'la_due_date_current_pregnancy'] = self.sim.date + DateOffset(days=onset_day)
            df.at[person, 'la_currently_in_labour'] = True  # todo: this ok?

            logger.debug(key='message', data=f'Mother {person} will go into preterm labour on '
                                             f'{self.sim.date + DateOffset(days=onset_day)}')

            self.sim.schedule_event(LabourOnsetEvent(self.sim.modules['Labour'], person),
                                    df.at[person, 'la_due_date_current_pregnancy'])

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
                                            self.sim.date + DateOffset(days=3))
            # We reset gestational age
            df.at[individual_id, 'ps_gestational_age_in_weeks'] = 0

    def disease_progression(self, selected):
        """This function uses util.transition_states to apply a probability of transitioning from one state of
        hypertensive disorder to another during each month of pregnancy"""
        df = self.sim.population.props

        # We first define the possible states that can be moved between
        disease_states = ['gest_htn', 'severe_gest_htn', 'mild_pre_eclamp', 'severe_pre_eclamp', 'eclampsia']
        prob_matrix = pd.DataFrame(columns=disease_states, index=disease_states)

        # TODO: these should be parameters
        # Probability of moving between states is stored in a matrix
        prob_matrix['gest_htn'] = [0.8, 0.1, 0.1, 0.0, 0.0]
        prob_matrix['severe_gest_htn'] = [0.0, 0.8, 0.0, 0.2, 0.0]
        prob_matrix['mild_pre_eclamp'] = [0.0, 0.0, 0.8, 0.2, 0.0]
        prob_matrix['severe_pre_eclamp'] = [0.0, 0.0, 0.0, 0.6, 0.4]
        prob_matrix['eclampsia'] = [0.0, 0.0, 0.0, 0.0, 1]

        # todo: I think eventually these values would need to be manipulated by treatment effects? (or will things
        #  like calcium only effect onset of pre-eclampsia not progression)

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
            logger.debug(key='message', data='The following women have developed severe pre-eclampsia during their '
                                             f'pregnancy {new_onset_severe_pre_eclampsia.index} on {self.sim.date}')

            for person in new_onset_severe_pre_eclampsia.index:
                df.at[person, 'ps_emergency_event'] = True

        # This process is then repeated for women who have developed eclampsia
        assess_status_change_for_eclampsia = (current_status != "eclampsia") & (new_status == "eclampsia")
        new_onset_eclampsia = assess_status_change_for_eclampsia[assess_status_change_for_eclampsia]

        if not new_onset_eclampsia.empty:
            logger.debug(key='message', data='The following women have developed eclampsia during their '
                                             f'pregnancy {new_onset_eclampsia.index} on {self.sim.date}')

            for person in new_onset_eclampsia.index:
                df.at[person, 'ps_emergency_event'] = True

    def care_seeking_pregnancy_emergency(self, df_slice):
        """This function determines if women who are experiencing a pregnancy emergency
        (PROM, APH, Severe pre-eclampsia/eclampsia) will seek care for treatment. Women who dont seek care have a risk
        of additional complications/death applied in this function also"""

        params = self.parameters
        df = self.sim.population.props

        result = params['ps_linear_equations']['care_seeking_pregnancy_complication'].predict(df_slice)
        random_draw = pd.Series(self.rng.random_sample(size=len(df_slice)), index=df_slice.index)
        temp_df = pd.concat([result, random_draw], axis=1)
        temp_df.columns = ['result', 'random_draw']

        care_seekers = temp_df.index[temp_df.random_draw < temp_df.result]
        non_care_seekers = temp_df.index[temp_df.random_draw > temp_df.result]

        for person in care_seekers:
            logger.debug(key='message', data=f'Mother {person} will seek care following acute pregnancy'
                                             f'complications')

            acute_pregnancy_hsi = HSI_CareOfWomenDuringPregnancy_MaternalEmergencyAssessment(
                self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person)

            self.sim.modules['HealthSystem'].schedule_hsi_event(acute_pregnancy_hsi, priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        if not non_care_seekers.empty:
            logger.debug(key='message', data=f'The following women will not seek care after experiencing a pregnancy '
                                             f'emergency: {non_care_seekers}')

            # TODO: apply risk of death to APH, ECLAMPSIA, SEVERE PRE-ECLAMPSIA
            # TODO: apply risk of stillbirth to survivors
            # TODO: reset eclampsia to severe PE, reset APH?
            # TODO: apply risk of infection to prom? (then if infection another round of care seeking/death)

            # aph_women = df.loc[non_care_seekers, 'ps_antepartum_haemorrhage']
            # ec_women = (df.loc[non_care_seekers, 'ps_htn_disorder'] == 'eclampsia')

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


class PregnancySupervisorEvent(RegularEvent, PopulationScopeEventMixin):
    """ This is the PregnancySupervisorEvent. It runs weekly. It updates gestational age of pregnancy in weeks.
    Presently this event has been hollowed out, additionally it will and uses set_pregnancy_complications function to
    determine if women will experience complication.. """

    def __init__(self, module, ):
        super().__init__(module, frequency=DateOffset(weeks=1))

    def apply(self, population):
        df = population.props
        params = self.module.parameters

        # ===================================== UPDATING GESTATIONAL AGE IN WEEKS  ====================================
        # Here we update the gestational age in weeks of all currently pregnant women in the simulation
        alive_and_preg = df.is_alive & df.is_pregnant
        gestation_in_days = self.sim.date - df.loc[alive_and_preg, 'date_of_last_pregnancy']
        gestation_in_weeks = gestation_in_days / np.timedelta64(1, 'W')

        df.loc[alive_and_preg, 'ps_gestational_age_in_weeks'] = gestation_in_weeks.astype('int64')
        logger.debug(key='message', data=f'updating gestational ages on date {self.sim.date}')

        # ===============PREGNANCY COMPLICATIONS - ECTOPIC PREGNANCY, MULTIPLES, PLACENTATION==========================
        # Here we use the set_pregnancy_complications function to calculate each womans risk of ectopic pregnancy,
        # conduct a draw and edit relevant properties defined above
        newly_pregnant_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 1)]
        self.module.set_pregnancy_complications(newly_pregnant_df, 'ectopic')

        # For women who don't experience and ectopic pregnancy we use the same function to assess risk of multiple
        # pregnancy
        no_ectopic_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 1) &
                               ~df.ps_ectopic_pregnancy & (df.dummy_anc_counter == 0)]
        self.module.set_pregnancy_complications(no_ectopic_df, 'multiples')
        # TODO: Review the necessity of including multiple pregnancies

        # Finally we ascertain if this womans pregnancy will have normal or abnormal placentation
        self.module.set_pregnancy_complications(no_ectopic_df, 'placenta_praevia')

        # ====================================== Scheduling first ANC visit ===========================================
        # For women whose pregnancy continues, we determine at what stage in their pregnancy they will seek antenatal
        # care

        for person in no_ectopic_df.index:
            # We use a probability weighted random draw to determine when this woman will attend ANC1,
            # and scheduled the visit accordingly
            # assert df.loc[person, 'dummy_anc_counter'] == 0
            df.loc[person, 'dummy_anc_counter'] += 1

            # TODO: In the final equation we will assume women dont start attending until it is realistic that they are
            #  aware they are pregnant. The current probabilities are dummys (have requested data from author of study
            #  for whom this equation is based on)

            # TODO: need to calibrate to ensure that 95% attend 1 ANC

        #    will_attend_anc4 = params['ps_linear_equations']['four_or_more_anc_visits'].predict(df.loc[[person]])[
        #        person]
        #    if self.module.rng.random_sample() < will_attend_anc4:
        #        df.at[person, 'ps_will_attend_four_or_more_anc'] = True

        #    will_attend_anc8 = params['ps_linear_equations']['eight_or_more_anc_visits'].predict(df.loc[[person]])[
        #                person]
        
            will_attend_early_anc3 = params['prob_3_early_visits']
        
            if self.module.rng.random_sample() < will_attend_early_anc3:
                df.at[person, 'ps_will_attend_3_early_visits'] = True
                random_draw_gest_at_anc = 2
            else:
                random_draw_gest_at_anc = self.module.rng.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                                 p=params['prob_first_anc_visit_gestational_age'])

            first_anc_date = self.sim.date + DateOffset(months=random_draw_gest_at_anc)
            first_anc_appt = HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareContact(
                self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person)

            self.sim.modules['HealthSystem'].schedule_hsi_event(first_anc_appt, priority=0,
                                                                topen=first_anc_date,
                                                                tclose=first_anc_date + DateOffset(days=7))

        # =========================================== MONTH 1 =========================================================
        # Here we look at all the women who have reached one month gestation and apply the risk of early pregnancy loss
        month_1_df = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks
                                                                                       == 4)]
        self.module.set_pregnancy_complications(month_1_df, 'spontaneous_abortion')

        # Women whose pregnancy continues may develop deficiencies of key vitamins/iron during pregnancy
        month_1_no_spontaneous_abortion_df = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                                                    (df.ps_gestational_age_in_weeks == 4)]
        self.module.set_pregnancy_complications(month_1_no_spontaneous_abortion_df, 'deficiencies')

        # Finally we determine if any women will become anaemic
        self.module.set_pregnancy_complications(month_1_no_spontaneous_abortion_df, 'maternal_anaemia')

        # =========================================== MONTH 2 =========================================================
        # Now we use the set_pregnancy_complications function to calculate risk and set properties for women whose
        # pregnancy is not ectopic

        # spontaneous_abortion:
        month_2_df = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                            (df.ps_gestational_age_in_weeks == 8)]
        self.module.set_pregnancy_complications(month_2_df, 'spontaneous_abortion')

        # Here we use the an index of women who will not miscarry to determine who will seek an induced abortion
        # induced_abortion:
        month_2_no_spontaneous_abortion_df = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                                                 (df.ps_gestational_age_in_weeks == 8)]
        self.module.set_pregnancy_complications(month_2_no_spontaneous_abortion_df, 'induced_abortion')

        month_2_no_induced_abortion_df = df.loc[
            ~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 8)]

        # deficiencies
        self.module.set_pregnancy_complications(month_2_no_induced_abortion_df, 'deficiencies')

        # anaemia
        month_2_no_anaemia_df = df.loc[
            ~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 8) &
            (df.ps_anaemia_in_pregnancy == 'none')]

        self.module.set_pregnancy_complications(month_2_no_anaemia_df, 'maternal_anaemia')

        # =========================================== MONTH 3 =========================================================
        # spontaneous_abortion
        month_3_df = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                             (df.ps_gestational_age_in_weeks == 13)]
        self.module.set_pregnancy_complications(month_3_df, 'spontaneous_abortion')

        # induced_abortion:
        month_3_no_spontaneous_abortion_df = df.loc[~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive &
                                                 (df.ps_gestational_age_in_weeks == 13)]
        self.module.set_pregnancy_complications(month_3_no_spontaneous_abortion_df, 'induced_abortion')

        month_3_no_induced_abortion_df = df.loc[
            ~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 13) &
            (df.ps_anaemia_in_pregnancy == 'none')]

        # deficiencies
        self.module.set_pregnancy_complications(month_3_no_induced_abortion_df, 'deficiencies')

        # anaemia
        month_3_no_anaemia_df = df.loc[
            ~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks ==13) &
            (df.ps_anaemia_in_pregnancy == 'none')]

        self.module.set_pregnancy_complications(month_3_no_anaemia_df, 'maternal_anaemia')

        # ============================================ MONTH 4 ========================================================
        # spontaneous_abortion
        month_4_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 17)]
        self.module.set_pregnancy_complications(month_4_df, 'spontaneous_abortion')

        # induced_abortion:
        month_4_no_spontaneous_abortion_df = df.loc[df.is_pregnant & df.is_alive &
                                                    (df.ps_gestational_age_in_weeks == 17)]
        self.module.set_pregnancy_complications(month_4_no_spontaneous_abortion_df, 'induced_abortion')

        month_4_no_induced_abortion_df = df.loc[
            ~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 17) &
            (df.ps_anaemia_in_pregnancy == 'none')]

        # deficiencies
        self.module.set_pregnancy_complications(month_4_no_induced_abortion_df, 'deficiencies')

        # anaemia
        month_4_no_anaemia_df = df.loc[
            ~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 17) &
            (df.ps_anaemia_in_pregnancy == 'none')]

        self.module.set_pregnancy_complications(month_4_no_anaemia_df, 'maternal_anaemia')

        # ============================================= MONTH 5 =======================================================
        # spontaneous_abortion
        month_5_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 22)]
        self.module.set_pregnancy_complications(month_5_df, 'spontaneous_abortion')

        # induced_abortion:
        month_5_no_spontaneous_abortion_df = df.loc[df.is_pregnant & df.is_alive &
                                                    (df.ps_gestational_age_in_weeks == 22)]
        self.module.set_pregnancy_complications(month_5_no_spontaneous_abortion_df, 'induced_abortion')

        month_5_no_induced_abortion_df = df.loc[
            df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 22) &
            (df.ps_anaemia_in_pregnancy == 'none')]

        # deficiencies
        self.module.set_pregnancy_complications(month_5_no_induced_abortion_df, 'deficiencies')

        # anaemia
        month_5_no_anaemia_df = df.loc[
            ~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 22) &
            (df.ps_anaemia_in_pregnancy == 'none')]

        self.module.set_pregnancy_complications(month_5_no_anaemia_df, 'maternal_anaemia')

        # Here we begin to apply the risk of developing complications which present later in pregnancy including
        # pre-eclampsia, gestational hypertension and gestational diabetes

        # TODO: discuss with Tim H and Britta how hypertension should be handled in pregnancy, im using a
        #  very binary variable indicating hypertension.

        # pre-eclampsia
        # Only women without pre-existing hypertensive disorders of pregnancy are can develop the disease now
        month_5_no_htn_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 22) &
                                (df.ps_htn_disorders == 'none')]
        self.module.set_pregnancy_complications(month_5_no_htn_df, 'pre_eclampsia')

        # gestational hypertension
        # This is the same for new onset hypertension
        month_5_no_pe_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 22)
                                  & (df.ps_htn_disorders == 'none')]
        self.module.set_pregnancy_complications(month_5_no_pe_df, 'gest_htn')

        # gestational diabetes
        month_5_no_diab_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 22) &
                                 ~df.ps_gest_diab]
        self.module.set_pregnancy_complications(month_5_no_diab_df, 'gest_diab')

        # PROM
        month_5_all_women_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 22)]
        self.module.set_pregnancy_complications(month_5_all_women_df, 'premature_rupture_of_membranes')

        # PRETERM LABOUR ONSET
        month_5_all_women_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 22)]
        self.module.preterm_labour_onset(month_5_all_women_df, 5)

        #  !!!!!!!!!!!!!!!!!!!!CARE SEEKING FOR EMERGENCIES (FOR WOMEN NOT GOING INTO EARLY LABOUR) !!!!!!!!!!!!!
        month_5_not_preterm_labour_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 22) &
                                               ~df.la_currently_in_labour & df.ps_emergency_event]
        self.module.care_seeking_pregnancy_emergency(month_5_not_preterm_labour_df)

        # From month 5 we apply a monthly risk of antenatal death that considers the impact of maternal diseases
        # death
        #  self.module.set_pregnancy_complications(month_5_all_women_df, 'antenatal_death')

        # =========================== MONTH 6 RISK APPLICATION =======================================================
        # TODO: should this be 28 weeks to align with still birth definition
        # From month 6 it is possible women could be in labour at the time of this event so we exclude them

        # still birth
        month_6_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 27) &
                             ~df.la_currently_in_labour]
        self.module.set_pregnancy_complications(month_6_df, 'antenatal_stillbirth')

        month_6_preg_continues_no_stillbirth = df.loc[df.is_pregnant & df.is_alive &
                                                      (df.ps_gestational_age_in_weeks == 27)
                                                      & ~df.la_currently_in_labour]

        # deficiencies
        self.module.set_pregnancy_complications(month_6_preg_continues_no_stillbirth, 'deficiencies')

        # anaemia
        month_6_no_anaemia_df = df.loc[
            ~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 27) &
            (df.ps_anaemia_in_pregnancy == 'none')]

        self.module.set_pregnancy_complications(month_6_no_anaemia_df, 'maternal_anaemia')

        # pre-eclampsia
        month_6_preg_continues_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 27) &
                                        ~df.la_currently_in_labour & (df.ps_htn_disorders == 'none')]
        self.module.set_pregnancy_complications(month_6_preg_continues_df, 'pre_eclampsia')

        # gestational hypertension
        month_6_no_pe_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 27) &
                               ~df.la_currently_in_labour & (df.ps_htn_disorders == 'none')]
        self.module.set_pregnancy_complications(month_6_no_pe_df, 'gest_htn')

        # gestational diabetes
        month_6_no_diab_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 27) &
                                 ~df.ps_gest_diab & ~df.la_currently_in_labour]
        self.module.set_pregnancy_complications(month_6_no_diab_df, 'gest_diab')

        # From month six we also determine if women suffering from any hypertensive disorders of pregnancy will progress
        # from one disease to another
        month_6_htn_disorder_df = df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 27) & (
            df.ps_htn_disorders != 'none') & ~df.la_currently_in_labour
        self.module.disease_progression(month_6_htn_disorder_df)

        # Placental abruption
        month_6_preg_continues_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 27) &
                                           ~df.la_currently_in_labour]
        self.module.set_pregnancy_complications(month_6_preg_continues_df, 'placental_abruption')

        # Antepartum haemorrhage
        # From month six we apply a risk of antenatal bleeding to all women
        self.module.set_pregnancy_complications(month_6_preg_continues_df, 'antepartum_haem')

        # PRETERM LABOUR ONSET
        month_6_all_women_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 27)]
        self.module.preterm_labour_onset(month_6_all_women_df, 6)

        #  !!!!!!!!!!!!!!!!!!!!CARE SEEKING FOR EMERGENCIES (FOR WOMEN NOT GOING INTO EARLY LABOUR) !!!!!!!!!!!!!
        month_6_not_preterm_labour_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 27) &
                                               ~df.la_currently_in_labour & df.ps_emergency_event]
        self.module.care_seeking_pregnancy_emergency(month_6_not_preterm_labour_df)

        # death
        # month_6_all_women_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 22) &
        #                           ~df.la_currently_in_labour]
        # self.module.set_pregnancy_complications(month_6_all_women_df, 'antenatal_death')

        # =========================== MONTH 7 RISK APPLICATION =======================================================
        # still birth
        month_7_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 31) &
                             ~df.la_currently_in_labour]

        self.module.set_pregnancy_complications(month_7_df, 'antenatal_stillbirth')

        month_7_preg_continues_no_stillbirth = df.loc[df.is_pregnant & df.is_alive &
                                                      (df.ps_gestational_age_in_weeks == 31)
                                                      & ~df.la_currently_in_labour]

        # deficiencies
        self.module.set_pregnancy_complications(month_7_preg_continues_no_stillbirth, 'deficiencies')

        # anaemia
        month_7_no_anaemia_df = df.loc[
            ~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 31) &
            (df.ps_anaemia_in_pregnancy == 'none')]

        self.module.set_pregnancy_complications(month_7_no_anaemia_df, 'maternal_anaemia')

        # pre-eclampsia
        month_7_preg_continues_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 31) &
                                        ~df.la_currently_in_labour & (df.ps_htn_disorders == 'none')]
        self.module.set_pregnancy_complications(month_7_preg_continues_df, 'pre_eclampsia')

        # gestational hypertension
        month_7_no_pe_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 31) &
                               ~df.la_currently_in_labour & (df.ps_htn_disorders == 'none')]
        self.module.set_pregnancy_complications(month_7_no_pe_df, 'gest_htn')

        # gestational diabetes
        month_7_no_diab_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 31) &
                                 ~df.ps_gest_diab & ~df.la_currently_in_labour]
        self.module.set_pregnancy_complications(month_7_no_diab_df, 'gest_diab')

        # disease progression
        month_7_htn_disorder_df = df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 31) & (
            df.ps_htn_disorders != 'none') & ~df.la_currently_in_labour
        self.module.disease_progression(month_7_htn_disorder_df)

        # Placental abruption
        month_7_preg_continues_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 31) &
                                           ~df.la_currently_in_labour]
        self.module.set_pregnancy_complications(month_7_preg_continues_df, 'placental_abruption')

        # Antepartum haemorrhage
        self.module.set_pregnancy_complications(month_7_preg_continues_df, 'antepartum_haem')

        # PRETERM LABOUR ONSET
        month_7_all_women_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 31)]
        self.module.preterm_labour_onset(month_7_all_women_df, 7)

        #  !!!!!!!!!!!!!!!!!!!!CARE SEEKING FOR EMERGENCIES (FOR WOMEN NOT GOING INTO EARLY LABOUR) !!!!!!!!!!!!!
        month_7_not_preterm_labour_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 31) &
                                               ~df.la_currently_in_labour & df.ps_emergency_event]
        self.module.care_seeking_pregnancy_emergency(month_7_not_preterm_labour_df)

        # death
        # month_7_all_women_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 31) &
        #                          ~df.la_currently_in_labour]
        # self.module.set_pregnancy_complications(month_7_all_women_df, 'antenatal_death')

        # =========================== MONTH 8 RISK APPLICATION ========================================================
        # still birth
        month_8_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 35) &
                             ~df.la_currently_in_labour]

        self.module.set_pregnancy_complications(month_8_df, 'antenatal_stillbirth')

        month_8_preg_continues_no_stillbirth = df.loc[df.is_pregnant & df.is_alive &
                                                      (df.ps_gestational_age_in_weeks == 35)
                                                      & ~df.la_currently_in_labour]

        # deficiencies
        self.module.set_pregnancy_complications(month_8_preg_continues_no_stillbirth, 'deficiencies')

        # anaemia
        month_8_no_anaemia_df = df.loc[
            ~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 35) &
            (df.ps_anaemia_in_pregnancy == 'none')]

        self.module.set_pregnancy_complications(month_8_no_anaemia_df, 'maternal_anaemia')

        # pre-eclampsia
        month_8_preg_continues_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 35) &
                                        ~df.la_currently_in_labour & (df.ps_htn_disorders == 'none')]
        self.module.set_pregnancy_complications(month_8_preg_continues_df, 'pre_eclampsia')

        # gestational hypertension
        month_8_no_pe_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 35)
                               & (df.ps_htn_disorders == 'none') & ~df.la_currently_in_labour]
        self.module.set_pregnancy_complications(month_8_no_pe_df, 'gest_htn')

        # gestational diabetes
        month_8_no_diab_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 35) &
                                 ~df.ps_gest_diab & ~df.la_currently_in_labour]
        self.module.set_pregnancy_complications(month_8_no_diab_df, 'gest_diab')

        # disease progression
        month_8_htn_disorder_df = df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 35) & (
            df.ps_htn_disorders != 'none')
        self.module.disease_progression(month_8_htn_disorder_df)

        # Placental abruption
        month_8_preg_continues_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 35) &
                                           ~df.la_currently_in_labour]
        self.module.set_pregnancy_complications(month_8_preg_continues_df, 'placental_abruption')

        # Antepartum haemorrhage
        self.module.set_pregnancy_complications(month_8_preg_continues_df, 'antepartum_haem')

        # PRETERM LABOUR ONSET
        month_8_all_women_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 35)]
        self.module.preterm_labour_onset(month_8_all_women_df, 8)

        #  !!!!!!!!!!!!!!!!!!!!CARE SEEKING FOR EMERGENCIES (FOR WOMEN NOT GOING INTO EARLY LABOUR) !!!!!!!!!!!!!
        month_8_not_preterm_labour_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 35) &
                                               ~df.la_currently_in_labour & df.ps_emergency_event]
        self.module.care_seeking_pregnancy_emergency(month_8_not_preterm_labour_df)

        # death
        # month_8_all_women_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 35) &
        #                           ~df.la_currently_in_labour]
        # self.module.set_pregnancy_complications(month_8_all_women_df, 'antenatal_death')

        # =========================== MONTH 9 RISK APPLICATION ========================================================
        # still birth
        month_9_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 40) &
                             ~df.la_currently_in_labour]

        self.module.set_pregnancy_complications(month_9_df, 'antenatal_stillbirth')

        month_9_preg_continues_no_stillbirth = df.loc[df.is_pregnant & df.is_alive &
                                                      (df.ps_gestational_age_in_weeks == 40)
                                                      & ~df.la_currently_in_labour]

        # deficiencies
        self.module.set_pregnancy_complications(month_9_preg_continues_no_stillbirth, 'deficiencies')

        # anaemia
        month_9_no_anaemia_df = df.loc[
            ~df.ps_ectopic_pregnancy & df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 40) &
            (df.ps_anaemia_in_pregnancy == 'none')]

        self.module.set_pregnancy_complications(month_9_no_anaemia_df, 'maternal_anaemia')

        # pre-eclampsia
        month_9_preg_continues_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 40) &
                                          ~df.la_currently_in_labour & (df.ps_htn_disorders == 'none')]
        self.module.set_pregnancy_complications(month_9_preg_continues_df, 'pre_eclampsia')

        # gestational hypertension
        month_9_no_pe_df = df.loc[df.is_pregnant & df.is_alive & ~df.la_currently_in_labour &
                                 (df.ps_gestational_age_in_weeks == 40) & (df.ps_htn_disorders == 'none')]
        self.module.set_pregnancy_complications(month_9_no_pe_df, 'gest_htn')

        # gestational diabetes
        month_9_no_diab_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 40) &
                                    ~df.ps_gest_diab & ~df.la_currently_in_labour]
        self.module.set_pregnancy_complications(month_9_no_diab_df, 'gest_diab')

        # disease progression
        month_9_htn_disorder_df = df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 40) & (
            df.ps_htn_disorders != 'none') & ~df.la_currently_in_labour
        self.module.disease_progression(month_9_htn_disorder_df)

        # Placental abruption
        month_9_preg_continues_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 40) &
                                           ~df.la_currently_in_labour]
        self.module.set_pregnancy_complications(month_9_preg_continues_df, 'placental_abruption')

        # Antepartum haemorrhage
        # From month six we apply a risk of antenatal bleeding to all women
        self.module.set_pregnancy_complications(month_9_preg_continues_df, 'antepartum_haem')

        # Care seeking for emergencies
        month_9_not_in_labour_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 40) &
                                               ~df.la_currently_in_labour & df.ps_emergency_event]
        self.module.care_seeking_pregnancy_emergency(month_9_not_in_labour_df)

        # death
        # month_9_all_women_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 40) &
        #                            ~df.la_currently_in_labour]
        # self.module.set_pregnancy_complications(month_9_all_women_df, 'antenatal_death')

        # =========================== WEEK 41 RISK APPLICATION ========================================================
        # Risk of still birth increases significantly in women who carry pregnancies beyond term
        week_41_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 41) &
                             ~df.la_currently_in_labour]

        self.module.set_pregnancy_complications(week_41_df, 'antenatal_stillbirth')

        # =========================== WEEK 42 RISK APPLICATION ========================================================
        week_42_df= df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 42) &
                             ~df.la_currently_in_labour]

        self.module.set_pregnancy_complications(week_42_df, 'antenatal_stillbirth')

        # =========================== WEEK 43 RISK APPLICATION ========================================================
        week_43_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 43) &
                             ~df.la_currently_in_labour]

        self.module.set_pregnancy_complications(week_43_df, 'antenatal_stillbirth')
        # =========================== WEEK 44 RISK APPLICATION ========================================================
        week_44_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 44) &
                             ~df.la_currently_in_labour]

        self.module.set_pregnancy_complications(week_44_df, 'antenatal_stillbirth')
        # =========================== WEEK 45 RISK APPLICATION ========================================================
        week_45_df = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks == 45) &
                             ~df.la_currently_in_labour]

        self.module.set_pregnancy_complications(week_45_df, 'antenatal_stillbirth')


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


class PregnancyDeathEvent(Event, IndividualScopeEventMixin):
    """"""

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        mother = df.loc[individual_id]

        if mother.is_alive and mother.is_pregnant and ~mother.la_currently_in_labour:
            logger.debug(key='message', data=f'Mother {individual_id} has reached PregnancyDeathEvent on '
                                             f'{self.sim.date}')

            antenatal_death = False
            still_birth = False

            if mother.ps_antepartum_haemorrhage:
                risk_of_death = params['ps_linear_equations']['antepartum_haemorrhage_death'].predict(
                    df.loc[[individual_id]])[individual_id]

                if self.module.rng.random_sample() < risk_of_death:
                    antenatal_death = True
                else:
                    risk_of_still_birth = params['ps_linear_equations'][
                        'antepartum_haemorrhage_stillbirth'].predict(df.loc[[individual_id]])[individual_id]
                    if self.module.rng.random_sample() < risk_of_still_birth:
                        still_birth = True

                    df.at[individual_id, 'ps_antepartum_haemorrhage'] = False
                    df.at[individual_id, 'ps_antepartum_haemorrhage_severity'] = 'none'

            if mother.ps_htn_disorders == 'eclampsia':
                risk_of_death = params['ps_linear_equations']['eclampsia_death'].predict(
                        df.loc[[individual_id]])[individual_id]

                if self.module.rng.random_sample() < risk_of_death:
                    antenatal_death = True
                else:
                    risk_of_still_birth = params['ps_linear_equations'][
                        'eclampsia_still_birth'].predict(df.loc[[individual_id]])[individual_id]
                    if self.module.rng.random_sample() < risk_of_still_birth:
                        still_birth = True

                    df.at[individual_id, 'ps_htn_disorders'] = 'severe_pre_eclamp'

            if antenatal_death:
                logger.debug(key='message', data=f'mother {individual_id} has died following a pregnancy emergency on '
                                                 f'date {self.sim.date}')

                self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                      cause='antenatal_emergency'), self.sim.date)
                self.module.pregnancy_disease_tracker['antenatal_death'] += 1

            elif still_birth:
                logger.debug(key='message', data=f'person {individual_id} has experience an antepartum stillbirth on '
                                                 f'date {self.sim.date}')

                self.module.pregnancy_disease_tracker['antenatal_stillbirth'] += 1

                df.at[individual_id, 'ps_antepartum_still_birth'] = True
                df.at[individual_id, 'ps_previous_stillbirth'] = True
                df.at[individual_id, 'is_pregnant'] = False
                df.at[individual_id, 'la_due_date_current_pregnancy'] = pd.NaT
                df.at[individual_id, 'ps_gestational_age_in_weeks'] = 0


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
        total_abortions_t = self.module.pregnancy_disease_tracker['induced_abortion']
        total_spontaneous_abortions_t = self.module.pregnancy_disease_tracker['spontaneous_abortion']
        total_anaemia_cases = self.module.pregnancy_disease_tracker['maternal_anaemia']
        total_ectopic_deaths = self.module.pregnancy_disease_tracker['ectopic_pregnancy_death']
        total_ia_deaths = self.module.pregnancy_disease_tracker['induced_abortion_death']
        total_sa_deaths = self.module.pregnancy_disease_tracker['spontaneous_abortion_death']
        crude_new_onset_pe = self.module.pregnancy_disease_tracker['new_onset_pre_eclampsia']
        crude_new_gh = self.module.pregnancy_disease_tracker['new_onset_gest_htn']
        crude_aph = self.module.pregnancy_disease_tracker['antepartum_haem']
        crude_aph_death = self.module.pregnancy_disease_tracker['antepartum_haem_death']

        women_month_6 = self.module.pregnancy_disease_tracker['women_at_6_months']


        dict_for_output = {'repro_women': total_women_reproductive_age,
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
                           'crude_ectopics': total_ectopics,
                           'ectopic_rate': (total_ectopics / total_women_reproductive_age) * 1000,
                           'ectopic_death_rate': (total_ectopic_deaths / total_women_reproductive_age) * 1000,
                           'crude_anaemia': total_anaemia_cases,
                           'anaemia_rate': (total_anaemia_cases/total_women_reproductive_age) * 1000,
                           'crude_pe': crude_new_onset_pe,
                           'crude_gest_htn': crude_new_gh,
                           'crude_aph':crude_aph,
                           'crude_aph_death':crude_aph_death,
                           'women_month_6': women_month_6}

        logger.info(key='ps_summary_statistics', data=dict_for_output,
                    description='Yearly summary statistics output from the pregnancy supervisor module')

        self.module.pregnancy_disease_tracker = {'ectopic_pregnancy': 0, 'induced_abortion': 0, 'spontaneous_abortion': 0,
                                               'ectopic_pregnancy_death': 0, 'induced_abortion_death': 0,
                                               'spontaneous_abortion_death': 0, 'maternal_anaemia': 0,
                                               'antenatal_death': 0, 'antenatal_stillbirth': 0,
                                               'new_onset_pre_eclampsia': 0, 'new_onset_gest_htn': 0,
                                               'antepartum_haem': 0, 'antepartum_haem_death': 0, 'women_at_6_months':0}
