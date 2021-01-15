from pathlib import Path

import pandas as pd
import numpy as np

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata
from tlo.methods import demography

from tlo.methods.healthsystem import HSI_Event
from tlo.methods.dxmanager import DxTest
from tlo.methods.labour import LabourOnsetEvent
from tlo.methods.tb import HSI_TbScreening

from tlo.util import BitsetHandler
# TODO: when HIV is updated in master include: from tlo.methods.hiv import HSI_Hiv_TestAndRefer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CareOfWomenDuringPregnancy(Module):
    """This is the CareOfWomenDuringPregnancy module which contains health system interaction events relevant to
     pregnancy and pregnancy loss including:

     1.) HSI_CareOfWomenDuringPregnancy_AntenatalCareContact (1-8) representing all 8 routine antenatal care contacts
        (ANC) recommended during pregnancy (with sequential scheduling of each event occurring within the HSI)

     2.) HSI_CareOfWomenDuringPregnancy_PostAbortionCaseManagement representing treatment for complications following
         abortion (post abortion care of PAC) for women seeking care from the community

     3.) HSI_CareOfWomenDuringPregnancy_TreatmentForEctopicPregnancy representing treatment for ectopic pregnancy for
         women seeking care from the community

     4.) HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare which represents antenatal inpatient care for women
         who require admission following complications of pregnancy, detected via ANC or following care seeking from the
         community including treatment and/or referral for (hypertension, diabetes, antepartum haemorrhage, anaemia,
         premature of membranes, chorioamnionitis)

    Additionally the module stores a number of HSIs which represent follow up for women who are scheduled for additional
    testing following an admission and initiation of treatment (i.e. anaemia or gestational diabetes). Individual
    interventions are stored as functions within the module to prevent repetition.
    """

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # This dictionary is used by to track the frequency of certain events in the module which are processed by the
        # logging event
        self.anc_tracker = dict()

    METADATA = {
        Metadata.USES_HEALTHSYSTEM,
    }

    PARAMETERS = {
        'prob_anc_continues': Parameter(
            Types.REAL, 'probability a woman will return for a subsequent ANC appointment'),
        'prob_an_ip_at_facility_level_1_2_3': Parameter(
            Types.REAL, 'probabilities that antenatal inpatient care will be scheduled at facility level 1, 2 or 3'),
        'prob_intervention_delivered_urine_ds': Parameter(
            Types.REAL, 'probability a woman will receive the intervention "urine dipstick" given that the HSI has ran '
                        'and the consumables are available (proxy for clinical quality)'),
        'prob_intervention_delivered_bp': Parameter(
            Types.REAL, 'probability a woman will receive the intervention "blood pressure measurement" given that the '
                        'HSI has ran and the consumables are available (proxy for clinical quality)'),
        'prob_intervention_delivered_ifa': Parameter(
            Types.REAL, 'probability a woman will receive the intervention "iron and folic acid" given that the HSI'
                        ' has ran and the consumables are available (proxy for clinical quality)'),
        'prob_intervention_delivered_bep': Parameter(
            Types.REAL, 'probability a woman will receive the intervention "Balance energy and protein supplements" '
                        'given that the HSI has ran and the consumables are available (proxy for clinical quality)'),
        'prob_intervention_delivered_llitn': Parameter(
            Types.REAL, 'probability a woman will receive the intervention "Long lasting insecticide treated net" '
                        'given that the HSI has ran and the consumables are available (proxy for clinical quality)'),
        'prob_intervention_delivered_tb_screen': Parameter(
            Types.REAL, 'probability a woman will receive the intervention "TB screen" given that the HSI has ran '
                        'and the consumables are available (proxy for clinical quality)'),
        'prob_intervention_delivered_tt': Parameter(
            Types.REAL, 'probability a woman will receive the intervention "tetanus toxoid" given that the HSI has '
                        'ran and the consumables are available (proxy for clinical quality)'),
        'prob_intervention_delivered_calcium': Parameter(
            Types.REAL, 'probability a woman will receive the intervention "urine dipstick" given that the HSI has ran '
                        'and the consumables are available (proxy for clinical quality)'),
        'prob_intervention_delivered_poct': Parameter(
            Types.REAL, 'probability a woman will receive the intervention "point of care Hb testing" given that the '
                        'HSI has ran and the consumables are available (proxy for clinical quality)'),
        'prob_intervention_delivered_albendazole': Parameter(
            Types.REAL, 'probability a woman will receive the intervention "albendazole" given that the HSI has ran '
                        'and the consumables are available (proxy for clinical quality)'),
        'prob_intervention_delivered_hepb_test': Parameter(
            Types.REAL, 'probability a woman will receive the intervention "Hepatitis B test" given that the HSI has '
                        'ran and the consumables are available (proxy for clinical quality)'),
        'prob_intervention_delivered_syph_test': Parameter(
            Types.REAL, 'probability a woman will receive the intervention "Syphilis test" given that the HSI has ran '
                        'and the consumables are available (proxy for clinical quality)'),
        'prob_intervention_delivered_hiv_test': Parameter(
            Types.REAL, 'probability a woman will receive the intervention "HIV test" given that the HSI has ran '
                        'and the consumables are available (proxy for clinical quality)'),
        'prob_intervention_delivered_iptp': Parameter(
            Types.REAL, 'probability a woman will receive the intervention "IPTp" given that the HSI has ran '
                        'and the consumables are available (proxy for clinical quality)'),
        'prob_intervention_delivered_gdm_test': Parameter(
            Types.REAL, 'probability a woman will receive the intervention "GDM screening" given that the HSI has ran '
                        'and the consumables are available (proxy for clinical quality)'),
        'sensitivity_bp_monitoring': Parameter(
            Types.REAL, 'sensitivity of blood pressure monitoring to detect hypertension'),
        'specificity_bp_monitoring': Parameter(
            Types.REAL, 'specificity of blood pressure monitoring to detect hypertension'),
        'sensitivity_urine_protein_1_plus': Parameter(
            Types.REAL, 'sensitivity of a urine dipstick test to detect proteinuria at 1+'),
        'specificity_urine_protein_1_plus': Parameter(
            Types.REAL, 'specificity of a urine dipstick test to detect proteinuria at 1+'),
        'sensitivity_poc_hb_test': Parameter(
            Types.REAL, 'sensitivity of a point of care Hb test to detect anaemia'),
        'specificity_poc_hb_test': Parameter(
            Types.REAL, 'specificity of a point of care Hb test to detect anaemia'),
        'sensitivity_fbc_hb_test': Parameter(
            Types.REAL, 'sensitivity of a Full Blood Count test to detect anaemia'),
        'specificity_fbc_hb_test': Parameter(
            Types.REAL, 'specificity of a Full Blood Count test to detect anaemia'),
        'sensitivity_blood_test_glucose': Parameter(
            Types.REAL, 'sensitivity of a blood test to detect raised blood glucose'),
        'specificity_blood_test_glucose': Parameter(
            Types.REAL, 'specificity of a blood test to detect raised blood glucose'),
        'effect_of_ifa_for_resolving_anaemia': Parameter(
            Types.REAL, 'treatment effectiveness of starting iron and folic acid on resolving anaemia'),
        'treatment_effect_blood_transfusion_anaemia': Parameter(
            Types.REAL, 'treatment effectiveness of blood transfusion for anaemia in pregnancy'),
        'effect_of_iron_replacement_for_resolving_anaemia': Parameter(
            Types.REAL, 'treatment effectiveness of iron replacement in resolving anaemia'),
        'effect_of_iron_replacement_for_resolving_iron_def': Parameter(
            Types.REAL, 'treatment effectiveness of iron replacement in resolving iron deficiency'),
        'effect_of_folate_replacement_for_resolving_anaemia': Parameter(
            Types.REAL, 'treatment effectiveness of folate replacement in resolving anaemia'),
        'effect_of_folate_replacement_for_resolving_folate_def': Parameter(
            Types.REAL, 'treatment effectiveness of folate replacement in resolving folate deficiency'),
        'effect_of_b12_replacement_for_resolving_anaemia': Parameter(
            Types.REAL, 'treatment effectiveness of b12 replacement in resolving anaemia'),
        'effect_of_b12_replacement_for_resolving_b12_def': Parameter(
            Types.REAL, 'treatment effectiveness of b12 replacement in resolving b12 deficiency'),
        'cfr_severe_pre_eclampsia': Parameter(
            Types.REAL, 'case fatality rate severe pre-eclampsia'),
        'multiplier_spe_cfr_eclampsia': Parameter(
            Types.REAL, 'cfr multiplier to generate cfr for eclampsia '),
        'treatment_effect_mag_sulph': Parameter(
            Types.REAL, 'treatment effectiveness magnesium sulphate on mortality due to severe pre-eclampsia/'
                        'eclampsia'),
        'treatment_effect_iv_anti_htn': Parameter(
            Types.REAL, 'treatment effectiveness IV antihypertensives on mortality due to severe pre-eclampsia/'
                        'eclampsia'),
        'prob_still_birth_aph': Parameter(
            Types.REAL, 'probability of a still birth following APH prior to treatment'),
        'prob_still_birth_spe_ec': Parameter(
            Types.REAL, 'probability of a still birth following severe pre-eclampsia/eclampsia prior to treatment'),
        'cfr_aph': Parameter(
            Types.REAL, 'case fatality rate antepartum haemorrhage'),
        'treatment_effect_bt_aph': Parameter(
            Types.REAL, 'treatment effect of blood transfusion on antepartum haemorrhage mortality '),
    }

    PROPERTIES = {
        'ac_total_anc_visits_current_pregnancy': Property(Types.INT, 'rolling total of antenatal visits this woman has '
                                                                     'attended during her pregnancy'),
        'ac_facility_type': Property(Types.CATEGORICAL, 'Type of facility that a woman will receive ANC in for her '
                                                        'pregnancy', categories=['none', 'health_centre', 'hospital']),
        'ac_date_next_contact': Property(Types.DATE, 'Date on which this woman is scheduled to return for her next '
                                                     'ANC contact'),
        'ac_to_be_admitted': Property(Types.BOOL, 'Whether this woman requires admission following an ANC visit'),
        'ac_receiving_iron_folic_acid': Property(Types.BOOL, 'whether this woman is receiving daily iron & folic acid '
                                                             'supplementation'),
        'ac_receiving_bep_supplements': Property(Types.BOOL, 'whether this woman is receiving daily balanced energy '
                                                             'and protein supplementation'),
        'ac_receiving_calcium_supplements': Property(Types.BOOL, 'whether this woman is receiving daily calcium '
                                                                 'supplementation'),
        'ac_doses_of_iptp_received': Property(Types.INT, 'Number of doses of intermittent preventative treatment in'
                                                         ' pregnancy received during this pregnancy'),
        'ac_itn_provided': Property(Types.BOOL, 'Whether this woman is provided with an insecticide treated bed net '
                                                'during the appropriate ANC visit'),
        'ac_ttd_received': Property(Types.INT, 'Number of doses of tetanus toxoid administered during this pregnancy'),
        'ac_gest_htn_on_treatment': Property(Types.BOOL, 'Whether this woman has been initiated on treatment for '
                                                         'gestational hypertension'),
        'ac_gest_diab_on_treatment': Property(Types.CATEGORICAL, 'Treatment this woman is receiving for gestational '
                                                                 'diabetes', categories=['none', 'diet_exercise',
                                                                                         'orals', 'insulin']),
        'ac_ectopic_pregnancy_treated': Property(Types.BOOL, 'Whether this woman has received treatment for an ectopic '
                                                             'pregnancy'),
        'ac_post_abortion_care_interventions': Property(Types.INT, 'bitset list of interventions delivered to a woman '
                                                                   'undergoing post abortion care'),
        'ac_received_abx_for_prom': Property(Types.BOOL, 'Whether this woman has received antibiotics as treatment for '
                                                         'premature rupture of membranes'),
        'ac_received_abx_for_chorioamnionitis': Property(Types.BOOL, 'Whether this woman has received antibiotics as '
                                                                     'treatment for chorioamnionitis rupture of '
                                                                     'membranes'),
        'ac_mag_sulph_treatment': Property(Types.BOOL, 'Whether this woman has received magnesium sulphate for '
                                                       'treatment of severe pre-eclampsia/eclampsia'),
        'ac_iv_anti_htn_treatment': Property(Types.BOOL, 'Whether this woman has received intravenous antihypertensive '
                                                         'drugs for treatment of severe hypertension'),
        'ac_received_blood_transfusion': Property(Types.BOOL, 'Whether this woman has received a blood transfusion '
                                                              'antenatally'),
        'ac_admitted_for_immediate_delivery': Property(Types.CATEGORICAL, 'Admission type for women needing urgent '
                                                                          'delivery in the antenatal period',
                                                       categories=['none', 'induction_now', 'induction_future',
                                                                   'caesarean_now', 'caesarean_future']),
        'ac_inpatient': Property(Types.BOOL, 'Whether this woman is currently an inpatient on the antenatal ward'),
    }

    def read_parameters(self, data_folder):
        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_AntenatalCare.xlsx',
                            sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)

        params = self.parameters

        # ==================================== LINEAR MODEL EQUATIONS =================================================
        # All linear equations used in this module are stored within the ac_linear_equations parameter below

        # TODO: process of 'selection' of important predictors in linear equations is ongoing, a linear model that
        #  is empty of predictors at the end of this process will be converted to a set probability

        params['ac_linear_equations'] = {

            # This equation is used to determine if a woman will choose to attend the next ANC contact in the routine
            # schedule
            'anc_continues': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['prob_anc_continues']),

            # This equation is used to determine if a woman will die following treatment for severe pre-eclampsia/
            # eclampsia. Risk of death is reduced by treatment
            'ec_spe_death_post_treatment': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_severe_pre_eclampsia'],
                Predictor('ps_htn_disorders').when('eclampsia', params['multiplier_spe_cfr_eclampsia']),
                Predictor('ac_mag_sulph_treatment').when(True, params['treatment_effect_mag_sulph']),
                Predictor('ac_iv_anti_htn_treatment').when(True, params['treatment_effect_iv_anti_htn'])),

            # This equation is used to determine if a woman will die following treatment for antepartum haemorrhage.
            # Risk of death is reduced by treatment
            'aph_death_post_treatment': LinearModel(
                LinearModelType.MULTIPLICATIVE,
                params['cfr_aph'],
                Predictor('ac_received_blood_transfusion').when(True, params['treatment_effect_bt_aph'])),
        }

    def initialise_population(self, population):

        df = population.props

        df.loc[df.is_alive, 'ac_total_anc_visits_current_pregnancy'] = 0
        df.loc[df.is_alive, 'ac_facility_type'] = 'none'
        df.loc[df.is_alive, 'ac_date_next_contact'] = pd.NaT
        df.loc[df.is_alive, 'ac_to_be_admitted'] = False
        df.loc[df.is_alive, 'ac_receiving_iron_folic_acid'] = False
        df.loc[df.is_alive, 'ac_receiving_bep_supplements'] = False
        df.loc[df.is_alive, 'ac_receiving_calcium_supplements'] = False
        df.loc[df.is_alive, 'ac_doses_of_iptp_received'] = 0
        df.loc[df.is_alive, 'ac_itn_provided'] = False
        df.loc[df.is_alive, 'ac_ttd_received'] = 0
        df.loc[df.is_alive, 'ac_gest_htn_on_treatment'] = False
        df.loc[df.is_alive, 'ac_gest_diab_on_treatment'] = 'none'
        df.loc[df.is_alive, 'ac_ectopic_pregnancy_treated'] = False
        df.loc[df.is_alive, 'ac_received_abx_for_prom'] = False
        df.loc[df.is_alive, 'ac_received_abx_for_chorioamnionitis'] = False
        df.loc[df.is_alive, 'ac_mag_sulph_treatment'] = False
        df.loc[df.is_alive, 'ac_iv_anti_htn_treatment'] = False
        df.loc[df.is_alive, 'ac_received_blood_transfusion'] = False
        df.loc[df.is_alive, 'ac_admitted_for_immediate_delivery'] = 'none'

        # This property stores the possible interventions a woman can receive during post abortion care
        self.pac_interventions = BitsetHandler(self.sim.population, 'ac_post_abortion_care_interventions',
                                               ['mva', 'd_and_c', 'misoprostol', 'antibiotics', 'blood_products',
                                                'injury_repair'])

    def initialise_simulation(self, sim):

        # We register the logging event and schedule to run each year
        sim.schedule_event(AntenatalCareLoggingEvent(self),
                           sim.date + DateOffset(years=1))

        # Then populate the tracker dictionary
        self.anc_tracker = {'total_first_anc_visits': 0, 'cumm_ga_at_anc1': 0, 'total_anc1_first_trimester': 0,
                            'anc8+': 0, 'timely_ANC3': 0, 'diet_supp_6_months': 0}

        # ==================================== REGISTERING DX_TESTS =================================================
        params = self.parameters
        # Next we register the relevant dx_tests used within this module...
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(

            # This test represents measurement of blood pressure used in ANC screening to detect hypertension in
            # pregnancy
            blood_pressure_measurement=DxTest(
                property='ps_htn_disorders', target_categories=['gest_htn', 'mild_pre_eclamp', 'severe_gest_htn',
                                                                'severe_pre_eclamp', 'eclampsia'],
                sensitivity=params['sensitivity_bp_monitoring'],
                specificity=params['specificity_bp_monitoring']),

            # This test represents a urine dipstick which is used to measuring the presence and amount of protein in a
            # womans urine, proteinuria being indicative of pre-eclampsia/eclampsia
            urine_dipstick_protein=DxTest(
                property='ps_htn_disorders', target_categories=['mild_pre_eclamp', 'severe_pre_eclamp', 'eclampsia'],
                sensitivity=params['sensitivity_urine_protein_1_plus'],
                specificity=params['specificity_urine_protein_1_plus']),

            # This test represents point of care haemoglobin testing used in ANC to detect anaemia (all-severity)
            point_of_care_hb_test=DxTest(
                property='ps_anaemia_in_pregnancy', target_categories=['mild', 'moderate', 'severe'],
                sensitivity=params['sensitivity_poc_hb_test'],
                specificity=params['specificity_poc_hb_test']),

            # This test represents laboratory based full blood count testing used in hospitals to determine severity of
            # anaemia via Hb levels
            full_blood_count_hb=DxTest(
                property='ps_anaemia_in_pregnancy', target_categories=['mild', 'moderate', 'severe'],
                sensitivity=params['sensitivity_fbc_hb_test'],
                specificity=params['specificity_fbc_hb_test']),

            # This test represents point of care glucose testing used in ANC to detect hyperglycemia, associated with
            # gestational diabetes
            blood_test_glucose=DxTest(
                property='ps_gest_diab',
                sensitivity=params['sensitivity_blood_test_glucose'],
                specificity=params['specificity_blood_test_glucose']))

    # This function is used within this and other modules to reset properties from this module when a woman is no longer
    # pregnant to ensure in future pregnancies properties arent incorrectly set to certain values
    def care_of_women_in_pregnancy_property_reset(self, ind_or_df, id_or_index):
        df = self.sim.population.props

        if ind_or_df == 'individual':
            set = df.at
        else:
            set = df.loc

        set[id_or_index, 'ac_total_anc_visits_current_pregnancy'] = 0
        set[id_or_index, 'ac_to_be_admitted'] = False
        set[id_or_index, 'ac_date_next_contact'] = pd.NaT
        set[id_or_index, 'ac_facility_type'] = 'none'
        set[id_or_index, 'ac_receiving_iron_folic_acid'] = False
        set[id_or_index, 'ac_receiving_bep_supplements'] = False
        set[id_or_index, 'ac_receiving_calcium_supplements'] = False
        set[id_or_index, 'ac_doses_of_iptp_received'] = 0
        set[id_or_index, 'ac_itn_provided'] = False
        set[id_or_index, 'ac_ttd_received'] = 0
        set[id_or_index, 'ac_gest_htn_on_treatment'] = False
        set[id_or_index, 'ac_gest_diab_on_treatment'] = 'none'
        set[id_or_index, 'ac_ectopic_pregnancy_treated'] = False
        set[id_or_index, 'ac_received_abx_for_prom'] = False
        set[id_or_index, 'ac_received_abx_for_chorioamnionitis'] = False
        set[id_or_index, 'ac_mag_sulph_treatment'] = False
        set[id_or_index, 'ac_iv_anti_htn_treatment'] = False
        set[id_or_index, 'ac_received_blood_transfusion'] = False
        set[id_or_index, 'ac_admitted_for_immediate_delivery'] = 'none'

    def on_birth(self, mother_id, child_id):
        df = self.sim.population.props

        df.at[child_id, 'ac_total_anc_visits_current_pregnancy'] = 0
        df.at[child_id, 'ac_to_be_admitted'] = False
        df.at[child_id, 'ac_date_next_contact'] = pd.NaT
        df.at[child_id, 'ac_facility_type'] = 'none'
        df.at[child_id, 'ac_receiving_iron_folic_acid'] = False
        df.at[child_id, 'ac_receiving_bep_supplements'] = False
        df.at[child_id, 'ac_receiving_calcium_supplements'] = False
        df.at[child_id, 'ac_doses_of_iptp_received'] = 0
        df.at[child_id, 'ac_itn_provided'] = False
        df.at[child_id, 'ac_ttd_received'] = 0
        df.at[child_id, 'ac_gest_htn_on_treatment'] = False
        df.at[child_id, 'ac_gest_diab_on_treatment'] = 'none'
        df.at[child_id, 'ac_ectopic_pregnancy_treated'] = False
        df.at[child_id, 'ac_received_abx_for_prom'] = False
        df.at[child_id, 'ac_received_abx_for_chorioamnionitis'] = False
        df.at[child_id, 'ac_mag_sulph_treatment'] = False
        df.at[child_id, 'ac_iv_anti_htn_treatment'] = False
        df.at[child_id, 'ac_received_blood_transfusion'] = False
        df.at[child_id, 'ac_admitted_for_immediate_delivery'] = 'none'

        # On_birth we run a check at birth to make sure no women exceed 8 visits- this test ensures women are not being
        # scheduled more ANC than required
        # logic
        assert df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] < 9

        # We log the total number of ANC contacts a woman has undergone at the time of birth via this data frame
        total_anc_visit_count = {'person_id': mother_id,
                                 'age': df.at[mother_id, 'age_years'],
                                 'date_of_delivery': self.sim.date,
                                 'total_anc': df.at[mother_id, 'ac_total_anc_visits_current_pregnancy']}

        logger.info(key='anc_count_on_birth', data=total_anc_visit_count, description='A dictionary containing the '
                                                                                      'number of ANC visits each woman'
                                                                                      'has on birth')

        # We then reset all relevant variables pertaining to care received during the antenatal period to avoid
        # treatments remaining in place for future pregnancies
        self.care_of_women_in_pregnancy_property_reset(
            ind_or_df='individual', id_or_index=mother_id)

    def on_hsi_alert(self, person_id, treatment_id):
        logger.debug(key='message', data=f'This is CareOfWomenDuringPregnancy, being alerted about a health system '
                                         f'interaction person {person_id} for: {treatment_id}')

    #  ================================ ADDITIONAL ANTENATAL HELPER FUNCTIONS =========================================

    def determine_gestational_age_for_next_contact(self, person_id):
        """
        This function is called by each of the ANC HSIs to determine the number of weeks before a woman is required to
        return for her next ANC contact in the schedule
        :param person_id: individual_id
        """
        df = self.sim.population.props
        mother = df.loc[person_id]

        # The recommended ANC schedule (gestational age in weeks at which it is recommended women attend) is
        # ANC1 - 12wks, ANC2 - 20wks, ANC3 - 26wks, ANC4 - 30wks, ANC5 - 34wks, ANC6 - 36wks, ANC7 - 38wks,
        # ANC8 - 40 wks

        # Using a womans gestational age at the time of her current visit, we calculate how many weeks in the future
        # until she should return for her next visit in the schedule
        if mother.ps_gestational_age_in_weeks < 20:
            recommended_gestation_next_anc = 20
            return recommended_gestation_next_anc

        elif 20 >= mother.ps_gestational_age_in_weeks < 26:
            recommended_gestation_next_anc = 26

        elif 26 >= mother.ps_gestational_age_in_weeks < 30:
            recommended_gestation_next_anc = 30

        elif 30 >= mother.ps_gestational_age_in_weeks < 34:
            recommended_gestation_next_anc = 34

        elif 34 >= mother.ps_gestational_age_in_weeks < 36:
            recommended_gestation_next_anc = 36

        elif 36 >= mother.ps_gestational_age_in_weeks < 38:
            recommended_gestation_next_anc = 38

        elif 38 >= mother.ps_gestational_age_in_weeks < 40:
            recommended_gestation_next_anc = 40

        # We schedule women who present very late for ANC to return in two weeks
        elif mother.ps_gestational_age_in_weeks >= 40:
            recommended_gestation_next_anc = 42

        return recommended_gestation_next_anc

    def antenatal_care_scheduler(self, individual_id, visit_to_be_scheduled, recommended_gestation_next_anc,
                                 facility_level):
        """
        This function is responsible for scheduling a womans next ANC contact in the schedule if she chooses to seek
        care again.  It is called by each of the ANC HSIs.
        :param individual_id: individual_id
        :param visit_to_be_scheduled: Number if next visit in the schedule (2-8)
        :param recommended_gestation_next_anc: Gestational age in weeks a woman should be for the next visit in her
        schedule
        :param facility_level: facility level that the next ANC contact in the schedule will occur at
        """
        df = self.sim.population.props
        params = self.parameters

        # We check that women will only be scheduled for the next ANC contact in the schedule
        assert df.at[individual_id, 'ps_gestational_age_in_weeks'] < recommended_gestation_next_anc

        # This function houses the code that schedules the next visit, it is abstracted to prevent repetition
        def set_anc_date(visit_number):

            # We store the ANC contacts as variables prior to scheduling. Facility level of the next contact is carried
            # forward from a womans first ANC contact (we assume she will always seek care within the same facility
            # level)
            if visit_number == 2:
                visit = HSI_CareOfWomenDuringPregnancy_SecondAntenatalCareContact(
                    self, person_id=individual_id, facility_level_of_this_hsi=facility_level)

            elif visit_number == 3:
                visit = HSI_CareOfWomenDuringPregnancy_ThirdAntenatalCareContact(
                    self, person_id=individual_id, facility_level_of_this_hsi=facility_level)

            elif visit_number == 4:
                visit = HSI_CareOfWomenDuringPregnancy_FourthAntenatalCareContact(
                    self, person_id=individual_id, facility_level_of_this_hsi=facility_level)

            elif visit_number == 5:
                visit = HSI_CareOfWomenDuringPregnancy_FifthAntenatalCareContact(
                    self, person_id=individual_id, facility_level_of_this_hsi=facility_level)

            elif visit_number == 6:
                visit = HSI_CareOfWomenDuringPregnancy_SixthAntenatalCareContact(
                    self, person_id=individual_id, facility_level_of_this_hsi=facility_level)

            elif visit_number == 7:
                visit = HSI_CareOfWomenDuringPregnancy_SeventhAntenatalCareContact(
                    self, person_id=individual_id, facility_level_of_this_hsi=facility_level)

            elif visit_number == 8:
                visit = HSI_CareOfWomenDuringPregnancy_EighthAntenatalCareContact(
                    self, person_id=individual_id, facility_level_of_this_hsi=facility_level)

            # If this woman has attended less than 4 visits, and is predicted to attend > 4 (as determined via the
            # PregnancySupervisor module when ANC1 is scheduled) her subsequent ANC appointment is automatically
            # scheduled
            if visit_number < 4:
                if df.at[individual_id, 'ps_will_attend_four_or_more_anc']:

                    # We subtract this womans current gestational age from the recommended gestational age for the next
                    # contact
                    weeks_due_next_visit = int(recommended_gestation_next_anc - df.at[individual_id,
                                                                                      'ps_gestational_age_in_weeks'])

                    # And use this value as the number of weeks until she is required to return for her next ANC
                    visit_date = self.sim.date + DateOffset(weeks=weeks_due_next_visit)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(visit, priority=0,
                                                                        topen=visit_date,
                                                                        tclose=visit_date + DateOffset(days=7))
                    logger.debug(key='message', data=f'mother {individual_id} will seek ANC {visit_number} contact on'
                                                     f' {visit_date}')

                    # We store the date of her next visit and use this date as part of a check when the ANC HSIs run
                    df.at[individual_id, 'ac_date_next_contact'] = visit_date

                else:
                    # If she is not predicted to attend 4 or more visits, we use an equation from the linear model to
                    # determine if she will seek care for her next contact
                    will_anc_continue = params['ac_linear_equations']['anc_continues'].predict(df.loc[[
                        individual_id]])[individual_id]

                    # If so, the HSI is scheduled in the same way
                    if self.rng.random_sample() < will_anc_continue:
                        weeks_due_next_visit = int(recommended_gestation_next_anc - df.at[individual_id,
                                                                                          'ps_gestational_age_in_'
                                                                                          'weeks'])
                        visit_date = self.sim.date + DateOffset(weeks=weeks_due_next_visit)
                        self.sim.modules['HealthSystem'].schedule_hsi_event(visit, priority=0,
                                                                            topen=visit_date,
                                                                            tclose=visit_date + DateOffset(days=7))

                        logger.debug(key='message',data=f'mother {individual_id} will seek ANC {visit_number} contact '
                                                        f'on {visit_date}')

                        df.at[individual_id, 'ac_date_next_contact'] = visit_date
                    else:
                        # If additional ANC care is not sought nothing happens
                        logger.debug(key='message', data=f'mother {individual_id} will not seek any additional '
                                                         f'antenatal care for this pregnancy')

            elif visit_number >= 4:
                # After 4 or more visits we use the linear model equation to determine if the woman will seek care for
                # her next contact

                if self.rng.random_sample() < params[
                    'ac_linear_equations']['anc_continues'].predict(df.loc[[individual_id]])[individual_id]:
                    weeks_due_next_visit = int(recommended_gestation_next_anc - df.at[individual_id,
                                                                                      'ps_gestational_age_in_'
                                                                                      'weeks'])
                    visit_date = self.sim.date + DateOffset(weeks=weeks_due_next_visit)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(visit, priority=0,
                                                                        topen=visit_date,
                                                                        tclose=visit_date + DateOffset(days=7))
                    logger.debug(key='message', data=f'mother {individual_id} will seek ANC {visit_number} contact on'
                                                     f'{visit_date}')
                    df.at[individual_id, 'ac_date_next_contact'] = visit_date

                else:
                    logger.debug(key='message', data=f'mother {individual_id} will not seek any additional antenatal '
                                                     f'care for this pregnancy')

        # We run the function to schedule the HSI
        if 2 <= visit_to_be_scheduled <= 8:
            set_anc_date(visit_to_be_scheduled)

    def schedule_admission(self, individual_id):
        """
        This function is called within each of the ANC HSIs for women who require admission due to a complication
        detected during ANC
        :param individual_id: individual_id
        """

        df = self.sim.population.props
        params = self.parameters

        # Use a weighted random draw to determine which level of facility the woman will be admitted too
        facility_level = int(self.rng.choice([1, 2, 3], p=params['prob_an_ip_at_facility_level_1_2_3']))

        # Schedule the HSI
        inpatient = HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare(
            self.sim.modules['CareOfWomenDuringPregnancy'], person_id=individual_id,
            facility_level_this_hsi=facility_level)

        self.sim.modules['HealthSystem'].schedule_hsi_event(inpatient, priority=0,
                                                            topen=self.sim.date,
                                                            tclose=self.sim.date + DateOffset(days=1))

        logger.debug(key='msg', data=f'Mother {individual_id} will be admitted to antenatal ward after a complication '
                                     f'was detected in routine ANC')

        # Reset the variable to prevent future scheduling errors
        df.at[individual_id, 'ac_to_be_admitted'] = False

    # ================================= INTERVENTIONS DELIVERED DURING ANC ============================================
    # The following functions contain the interventions that are delivered as part of routine ANC contacts. Functions
    # are called from within the ANC HSIs. Which interventions are called depends on the mothers gestation and the
    # number of visits she has attended at the time each HSI runs (see ANC HSIs)

    def screening_interventions_delivered_at_every_contact(self, hsi_event):
        """
        This function contains the screening interventions which are delivered at every ANC contact regardless of the
        womans gestational age and include blood pressure measurement and urine dipstick testing
        :param hsi_event: HSI event in which the function has been called:
        """
        person_id = hsi_event.target
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        df = self.sim.population.props
        params = self.parameters

        # Define the consumables
        item_code_urine_dipstick = pd.unique(
            consumables.loc[consumables['Items'] == 'Test strips, urine analysis', 'Item_Code'])[0]

        consumables_dipstick = {
            'Intervention_Package_Code': {},
            'Item_Code': {item_code_urine_dipstick: 1}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=hsi_event,
            cons_req_as_footprint=consumables_dipstick)

        # Delivery of the intervention is conditioned on the availability of the consumable and a random draw against a
        # probability that the intervention would be delivered (used to calibrate to SPA data- acts as proxy for
        # clinical quality)
        if outcome_of_request_for_consumables['Item_Code'][item_code_urine_dipstick] and (self.rng.random_sample() <
                                                                                          params['prob_intervention_'
                                                                                                 'delivered_urine_ds']):

            # If the consumables are available the test is ran. Urine testing in ANC is predominantly used to detected
            # protein in the urine (proteinuria) which is indicative of pre-eclampsia
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run='urine_dipstick_protein',
                                                                       hsi_event=hsi_event):

                # We use a temporary variable to store if proteinuria is detected
                proteinuria_diagnosed = True
            else:
                proteinuria_diagnosed = False
        else:
            logger.debug(key='msg', data='Urine dipstick testing was not completed in this ANC visit due to '
                                         'unavailable consumables')
            proteinuria_diagnosed = False

        # The process is repeated for blood pressure monitoring- although not conditioned on consumables
        if self.rng.random_sample() < params['prob_intervention_delivered_bp']:
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run='blood_pressure_measurement',
                                                                       hsi_event=hsi_event):
                hypertension_diagnosed = True
            else:
                hypertension_diagnosed = False
        else:
            hypertension_diagnosed = False

        # If either high blood pressure or proteinuria are detected (or both) we assume this woman needs to be admitted
        # for further treatment following this ANC contact
        if hypertension_diagnosed or proteinuria_diagnosed:
            df.at[person_id, 'ac_to_be_admitted'] = True

    def interventions_initiated_at_first_contact(self, hsi_event):
        """
        This function contains all the interventions that should be delivered during the initial ANC contact (contact 1)
        including initiation of daily iron and folic acid supplementation, daily balance energy and protein
        supplementation, provision of an insecticide treated bed net and screening for TB.
        :param hsi_event: HSI event in which the function has been called:
        """
        person_id = hsi_event.target
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        df = self.sim.population.props
        params = self.parameters

        # We calculate an approximate number of days left of a womans pregnancy to capture the consumables required for
        # daily iron/folic acid tablets
        if df.at[person_id, 'ps_gestational_age_in_weeks'] < 40:
            approx_days_of_pregnancy = (40 - df.at[person_id, 'ps_gestational_age_in_weeks']) * 7
        else:
            approx_days_of_pregnancy = 14

        # Iron/folic acid & BEP supplements...
        # Define the consumables
        item_code_iron_folic_acid = pd.unique(
            consumables.loc[consumables['Items'] == 'Ferrous Salt + Folic Acid, tablet, 200 + 0.25 mg', 'Item_Code'])[0]
        item_code_diet_supps = pd.unique(
            consumables.loc[consumables['Items'] == 'Dietary supplements (country-specific)', 'Item_Code'])[0]

        consumables_anc1 = {
            'Intervention_Package_Code': {},
            'Item_Code': {item_code_iron_folic_acid: approx_days_of_pregnancy,
                          item_code_diet_supps: approx_days_of_pregnancy}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=hsi_event,
            cons_req_as_footprint=consumables_anc1)

        # As with previous interventions - condition on consumables and probability intervention is delivered
        if outcome_of_request_for_consumables['Item_Code'][item_code_iron_folic_acid] and \
         (self.rng.random_sample() < params['prob_intervention_delivered_ifa']):
            df.at[person_id, 'ac_receiving_iron_folic_acid'] = True

        if outcome_of_request_for_consumables['Item_Code'][item_code_diet_supps] and \
         (self.rng.random_sample() < params['prob_intervention_delivered_bep']):
            df.at[person_id, 'ac_receiving_bep_supplements'] = True

        # LLITN provision...
        # We define the required consumables
        pkg_code_obstructed_llitn = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'ITN distribution to pregnant women',
                            'Intervention_Pkg_Code'])[0]

        consumables_llitn = {
            'Intervention_Package_Code': {pkg_code_obstructed_llitn: 1},
            'Item_Code': {}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=hsi_event,
            cons_req_as_footprint=consumables_llitn)

        # If available, women are provided with a bed net at ANC1. The effect of these nets is determined
        # through the malaria module - not yet coded. n.b any interventions involving non-obstetric diseases have been
        # discussed with Tara
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_obstructed_llitn] \
            and (self.rng.random_sample() < params['prob_intervention_delivered_llitn']):

            df.at[person_id, 'ac_itn_provided'] = True

        # TB screening...
        # Currently we schedule women to the TB screening HSI in the TB module, however this may over-use resources so
        # possible the TB screening should also just live in this code
        if 'tb' in self.sim.modules.keys():
            if self.rng.random_sample() < params['prob_intervention_delivered_tb_screen']:
                tb_screen = HSI_TbScreening(
                    module=self.sim.modules['tb'], person_id=person_id)

                self.sim.modules['HealthSystem'].schedule_hsi_event(tb_screen, priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1))

    def tetanus_vaccination(self, hsi_event):
        """

        :param hsi_event:
        :return:
        """

        # TODO: DOCUMENTATION UP TO HERE!
        person_id = hsi_event.target
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        df = self.sim.population.props
        params = self.parameters

        # Tetanus vaccination/booster...
        # Define required consumables
        pkg_code_tet = pd.unique(
            consumables.loc[consumables["Intervention_Pkg"] == "Tetanus toxoid (pregnant women)",
                            "Intervention_Pkg_Code"])[0]

        all_available = hsi_event.get_all_consumables(
            pkg_codes=[pkg_code_tet])

        if all_available and (self.rng.random_sample() < params['prob_intervention_delivered_tt']) :
            df.at[person_id, 'ac_ttd_received'] += 1

    def calcium_supplementation(self, hsi_event):
        """This function manages the intervention calcium supplementation"""
        df = self.sim.population.props
        params = self.parameters
        person_id = hsi_event.target
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        if ~df.at[person_id, 'ac_receiving_calcium_supplements']:

            # TODO: Confirm risk factors that define 'high risk of pre-eclampsia' and condition appropriately

            # Define consumables
            item_code_calcium_supp = pd.unique(
                consumables.loc[consumables['Items'] == 'Calcium, tablet, 600 mg', 'Item_Code'])[0]

            approx_days_of_pregnancy = (40 - df.at[person_id, 'ps_gestational_age_in_weeks']) * 7
            dose = approx_days_of_pregnancy * 3  # gives daily dose of 1.8g

            consumables_calcium = {
                'Intervention_Package_Code': {},
                'Item_Code': {item_code_calcium_supp: dose}}

            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=hsi_event,
                cons_req_as_footprint=consumables_calcium)

            if outcome_of_request_for_consumables['Item_Code'][item_code_calcium_supp] and\
                (self.rng.random_sample() < params['prob_intervention_delivered_calcium']):
                df.at[person_id, 'ac_receiving_calcium_supplements'] = True

    def point_of_care_hb_testing(self, hsi_event):
        """

        :param hsi_event:
        :return:
        """

        # TODO: consumables for POC test and FBC currently assumed to be the same (incorrect?)

        person_id = hsi_event.target
        df = self.sim.population.props
        params = self.parameters
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        # Define the required consumables
        item_code_hb_test = pd.unique(
            consumables.loc[consumables['Items'] == 'Haemoglobin test (HB)', 'Item_Code'])[0]
        item_code_blood_tube = pd.unique(
            consumables.loc[consumables['Items'] == 'Blood collecting tube, 5 ml', 'Item_Code'])[0]
        item_code_needle = pd.unique(
            consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]
        item_code_gloves = pd.unique(
            consumables.loc[consumables['Items'] == 'Gloves, exam, latex, disposable, pair', 'Item_Code'])[0]

        consumables_hb_test = {
            'Intervention_Package_Code': {},
            'Item_Code': {item_code_hb_test: 1, item_code_blood_tube: 1, item_code_needle: 1, item_code_gloves: 1}}

        # Confirm their availability
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=hsi_event,
            cons_req_as_footprint=consumables_hb_test)

        if (outcome_of_request_for_consumables['Item_Code'][item_code_hb_test]) and\
            (outcome_of_request_for_consumables['Item_Code'][item_code_blood_tube]) and \
                (outcome_of_request_for_consumables['Item_Code'][item_code_needle]) and \
                (outcome_of_request_for_consumables['Item_Code'][item_code_gloves]) and \
            (self.rng.random_sample() < params['prob_intervention_delivered_poct']):

                if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run='point_of_care_hb_test',
                                                                           hsi_event=hsi_event):
                    df.at[person_id, 'ac_to_be_admitted'] = True

    def albendazole_administration(self, hsi_event):
        """

        :param hsi_event:
        :return:
        """
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        params = self.parameters

        # We run this function to store the associated consumables with albendazole administration. This intervention
        # has no effect in the model due to limited evidence
        item_code_albendazole = pd.unique(
            consumables.loc[consumables['Items'] == 'Albendazole 200mg_1000_CMST', 'Item_Code'])[0]

        consumables_albendazole = {
            'Intervention_Package_Code': {},
            'Item_Code': {item_code_albendazole: 2}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=hsi_event,
            cons_req_as_footprint=consumables_albendazole)

        if outcome_of_request_for_consumables['Item_Code'][item_code_albendazole] and \
            (self.rng.random_sample() < params['prob_intervention_delivered_bp']):
            logger.debug(key='message', data='albendazole given')

    def hep_b_testing(self, hsi_event):
        """

        """
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        params = self.parameters

        # This intervention is a place holder prior to the Hepatitis B module being coded
        item_code_hep_test = pd.unique(
            consumables.loc[consumables['Items'] == 'Hepatitis B test kit-Dertemine_100 tests_CMST', 'Item_Code'])[0]
        item_code_blood_tube = pd.unique(
            consumables.loc[consumables['Items'] == 'Blood collecting tube, 5 ml', 'Item_Code'])[0]
        item_code_needle = pd.unique(
            consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]
        item_code_gloves = pd.unique(
            consumables.loc[consumables['Items'] == 'Gloves, exam, latex, disposable, pair', 'Item_Code'])[0]

        consumables_hep_b_test = {
                'Intervention_Package_Code': {},
                'Item_Code': {item_code_hep_test: 1, item_code_blood_tube: 1, item_code_needle:1, item_code_gloves: 1}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=hsi_event,
            cons_req_as_footprint=consumables_hep_b_test)

        if (outcome_of_request_for_consumables['Item_Code'][item_code_hep_test]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_blood_tube]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_needle]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_gloves]) and \
            (self.rng.random_sample() < params['prob_intervention_delivered_hepb_test']):
            logger.debug(key='message', data='hepatitis B test given')

    def syphilis_testing(self, hsi_event):
        """This function manages Syphilis testing. Syphilis is not explicitly modelled and therefore this function
        merely records consumable use"""
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        params = self.parameters

        # TODO: There are additional consumables associate with treatment that wont be counted if we dont model the
        #  disease
        # todo this intervention is in the EHP

        item_code_syphilis_test = pd.unique(
            consumables.loc[consumables['Items'] == 'Test, Rapid plasma reagin (RPR)', 'Item_Code'])[0]
        item_code_blood_tube = pd.unique(
            consumables.loc[consumables['Items'] == 'Blood collecting tube, 5 ml', 'Item_Code'])[0]
        item_code_needle = pd.unique(
            consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]
        item_code_gloves = pd.unique(
            consumables.loc[consumables['Items'] == 'Gloves, exam, latex, disposable, pair', 'Item_Code'])[0]

        consumables_syphilis = {
            'Intervention_Package_Code': {},
            'Item_Code': {item_code_syphilis_test: 1, item_code_blood_tube: 1, item_code_needle: 1,
                          item_code_gloves: 1}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=hsi_event,
            cons_req_as_footprint=consumables_syphilis)

        if (outcome_of_request_for_consumables['Item_Code'][item_code_syphilis_test]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_blood_tube]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_needle]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_gloves]) and \
            (self.rng.random_sample() < params['prob_intervention_delivered_syph_test']):
            logger.debug(key='message', data='syphilis test given')

    def hiv_testing(self, hsi_event):
        """

        :param hsi_event:
        :return:
        """
        person_id = hsi_event.target
        pass

        # TODO: await for new HIV code in master for this to work properly

    #    if 'hiv' in self.sim.modules.keys():
    #        if self.rng.random_sample() < params['prob_intervention_delivered_hiv_test']:
    #           self.sim.modules['HealthSystem'].schedule_hsi_event(
    #               HSI_Hiv_TestAndRefer(person_id=person_id, module=self.sim.modules['Hiv']),
    #               topen=self.sim.date,
    #               tclose=None,
    #               priority=0
    #               )
    #       else:
    #            logger.warning(key='message', data='Mother has not been referred for HIV testing')
#       else:
    #        logger.warning(key='message', data='HIV module is not registered in this simulation run and therefore HIV '
    #                                           'testing will not happen in antenatal care')

    def iptp_administration(self, hsi_event):
        """

        """
        df = self.sim.population.props
        params = self.parameters
        person_id = hsi_event.target
        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

        # todo: ensure conditioning on ma_tx when malaria code is available (as below)

        # if (not df.at[person_id, "ma_tx"]
        #    and not df.at[person_id, "ma_tx"]
        #    and df.at[person_id, "is_alive"]):

        # Test to ensure only 5 doses are able to be administered
        assert df.at[person_id, 'ac_doses_of_iptp_received'] < 6

        # Define and check the availability of consumables
        pkg_code_iptp = pd.unique(
            consumables.loc[consumables["Intervention_Pkg"] == "IPT (pregnant women)", "Intervention_Pkg_Code"])[0]

        all_available = hsi_event.get_all_consumables(
            pkg_codes=[pkg_code_iptp])

        if all_available and (self.rng.random_sample() < params['prob_intervention_delivered_iptp']):
            logger.debug(key='message', data=f'giving IPTp for person {person_id}')

            # IPTP is a single dose drug given at a number of time points during pregnancy. Therefore the number of
            # doses received during this pregnancy are stored as an integer
            df.at[person_id, 'ac_doses_of_iptp_received'] += 1

    def gdm_screening(self, hsi_event):
        """This intervention screens women with risk factors for gestational diabetes and schedules the appropriate
        testing"""
        df = self.sim.population.props
        params = self.parameters
        person_id = hsi_event.target
        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

        # We check if this women has any of the key risk factors, if so they are sent for additional blood tests
        if df.at[person_id, 'li_bmi'] >= 4 or df.at[person_id, 'ps_prev_gest_diab'] or df.at[person_id,
                                                                                             'ps_prev_stillbirth']:

            # We define the required consumables for testing
            item_code_glucose_test = pd.unique(
                consumables.loc[consumables['Items'] == 'Blood glucose level test', 'Item_Code'])[0]
            item_code_blood_tube = pd.unique(
                consumables.loc[consumables['Items'] == 'Blood collecting tube, 5 ml', 'Item_Code'])[0]
            item_code_needle = pd.unique(
                consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]
            item_code_gloves = pd.unique(
                consumables.loc[consumables['Items'] == 'Gloves, exam, latex, disposable, pair', 'Item_Code'])[0]

            consumables_gdm_testing = {
                'Intervention_Package_Code': {},
                'Item_Code': {item_code_glucose_test: 1, item_code_blood_tube: 1, item_code_needle: 1,
                              item_code_gloves: 1}}

            # Then query if these consumables are available during this HSI
            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=hsi_event,
                cons_req_as_footprint=consumables_gdm_testing)

            # If they are available, the test is conducted
            if (outcome_of_request_for_consumables['Item_Code'][item_code_glucose_test]) and \
                (outcome_of_request_for_consumables['Item_Code'][item_code_blood_tube]) and \
                (outcome_of_request_for_consumables['Item_Code'][item_code_needle]) and \
                (outcome_of_request_for_consumables['Item_Code'][item_code_gloves]) and \
                (self.rng.random_sample() < params['prob_intervention_delivered_gdm_test']):

                # If the test accurately detects a woman has gestational diabetes the consumables are recorded and she
                # is referred for treatment
                if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run='blood_test_glucose',
                                                                           hsi_event=hsi_event):

                    # We assume women with a positive GDM screen will be admitted
                    df.at[person_id, 'ac_to_be_admitted'] = True

    def anc_catch_up_interventions(self, hsi_event):
        """This function actions all the interventions a woman presenting to ANC1 at >20 will need administering."""
        self.hiv_testing(hsi_event=hsi_event)
        self.hep_b_testing(hsi_event=hsi_event)
        self.syphilis_testing(hsi_event=hsi_event)
        self.point_of_care_hb_testing(hsi_event=hsi_event)
        self.tetanus_vaccination(hsi_event=hsi_event)

        self.albendazole_administration(hsi_event=hsi_event)
        self.iptp_administration(hsi_event=hsi_event)
        self.calcium_supplementation(hsi_event=hsi_event)

    # =============================== INTERVENTIONS DELIVERED DURING INPATIENT CARE ===================================
    # Following functions contain code for the interventions which are called by antenatal HSIs such as Post Abortion
    # Care and the ANC contacts

    def post_abortion_care_interventions(self, hsi_event, individual_id, severity):
        """This function is called by HSI_CareOfWomenDuringPregnancy_PostAbortionCaseManagement. It houses the
        interventions delivered to women needing PAC. Likelihood of intervention is determined by severity of
        complications (not specific complications). Treatments are stored using bitset handler and mitigate risk of
        death in EarlyPregnancyLossDeathEvent in the PregnancySupversior Module """

        params = self.parameters
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        # First, based on severity of complications, we determine what form of uterine evacuation this woman will
        # undergo and store it accordingly
        evac_procedures = ['d_and_c', 'mva', 'misoprostol']
        probability_of_evac_procedure = params[f'prob_evac_procedure_{severity}_ac']
        random_draw = self.rng.choice(evac_procedures, p=probability_of_evac_procedure)

        self.pac_interventions.set(individual_id, random_draw)
        # TODO: consumables/equipment for evacuation methods
        # TODO: should surgical PAC be referred up to a higher facility

        # Next we determine if this woman will require analgesia
        if self.rng.random_sample() < params[f'prob_analgesia_{severity}_ac']:

            # As analgesia is not modelled to effect outcomes, we check the availability of consumables and log
            # accordingly
            item_code_paracetamol = pd.unique(
                consumables.loc[consumables['Items'] == 'Paracetamol, tablet, 500 mg', 'Item_Code'])[0]
            item_code_pethidine = pd.unique(
                consumables.loc[
                    consumables['Items'] == 'Pethidine, 50 mg/ml, 2 ml ampoule', 'Item_Code'])[0]
            item_code_needle = pd.unique(
                consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]
            item_code_gloves = pd.unique(
                consumables.loc[consumables['Items'] == 'Gloves, exam, latex, disposable, pair', 'Item_Code'])[0]

            consumables_analgesia_pac = {
                'Intervention_Package_Code': {},
                'Item_Code': {item_code_paracetamol: 8,  # 24 hour dose
                              item_code_pethidine: 1,
                              item_code_needle: 1,
                              item_code_gloves: 1}}

            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=hsi_event,
                cons_req_as_footprint=consumables_analgesia_pac,
                to_log=False)

            if outcome_of_request_for_consumables:
                self.sim.modules['HealthSystem'].request_consumables(
                    hsi_event=hsi_event,
                    cons_req_as_footprint=consumables_analgesia_pac,
                    to_log=True)
                self.pac_interventions.set(individual_id, 'analgesia')

        # Next it is determine if this woman will need to receive antibiotics
        if self.rng.random_sample() < params[f'prob_antibiotics_{severity}_ac']:

            pkg_code_infection = pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'Maternal sepsis case management',
                                'Intervention_Pkg_Code'])[0]

            consumables_needed_sepsis = {'Intervention_Package_Code': {pkg_code_infection: 1}, 'Item_Code': {}}

            outcome_of_request_for_consumables_sep = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=hsi_event,
                cons_req_as_footprint=consumables_needed_sepsis,
                to_log=False)

            if outcome_of_request_for_consumables_sep:
                self.pac_interventions.set(individual_id, 'antibiotics')
                self.sim.modules['HealthSystem'].request_consumables(
                    hsi_event=hsi_event,
                    cons_req_as_footprint=consumables_needed_sepsis,
                    to_log=True)

        # And finally if this woman needs a blood transfusion
        if self.rng.random_sample() < params[f'prob_blood_transfusion_{severity}_ac']:

            item_code_bt1 = pd.unique(consumables.loc[consumables['Items'] == 'Blood, one unit', 'Item_Code'])[0]
            item_code_bt2 = \
            pd.unique(consumables.loc[consumables['Items'] == 'Lancet, blood, disposable', 'Item_Code'])[0]
            item_code_bt3 = pd.unique(consumables.loc[consumables['Items'] == 'Test, hemoglobin', 'Item_Code'])[0]
            item_code_bt4 = pd.unique(consumables.loc[consumables['Items'] == 'IV giving/infusion set, with needle',
                                                      'Item_Code'])[0]

            consumables_needed_bt = {'Intervention_Package_Code': {}, 'Item_Code': {item_code_bt1: 2, item_code_bt2: 1,
                                                                                    item_code_bt3: 1, item_code_bt4: 2}}

            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=hsi_event, cons_req_as_footprint=consumables_needed_bt, to_log=False)

            if outcome_of_request_for_consumables:
                self.pac_interventions.set(individual_id, 'blood_products')
                self.sim.modules['HealthSystem'].request_consumables(
                    hsi_event=hsi_event, cons_req_as_footprint=consumables_needed_bt, to_log=True)

    def full_blood_count_testing(self, hsi_event):
        """This function manages the intervention full blood count hb testing"""
        df = self.sim.population.props
        person_id = hsi_event.target
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        # Define the required consumables
        item_code_hb_test = pd.unique(
            consumables.loc[consumables['Items'] == 'Haemoglobin test (HB)', 'Item_Code'])[0]
        item_code_blood_tube = pd.unique(
            consumables.loc[consumables['Items'] == 'Blood collecting tube, 5 ml', 'Item_Code'])[0]
        item_code_needle = pd.unique(
            consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]
        item_code_gloves = pd.unique(
            consumables.loc[consumables['Items'] == 'Gloves, exam, latex, disposable, pair', 'Item_Code'])[0]

        consumables_hb_test = {
            'Intervention_Package_Code': {},
            'Item_Code': {item_code_hb_test: 1, item_code_blood_tube: 1, item_code_needle: 1, item_code_gloves: 1}}

        # Confirm their availability
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=hsi_event,
            cons_req_as_footprint=consumables_hb_test,
            to_log=False)

        # Check consumables
        if outcome_of_request_for_consumables:

            # Log if available
            self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=hsi_event,
                cons_req_as_footprint=consumables_hb_test,
                to_log=True)

            # Run dx_test for anaemia

            # If a woman is not truly anaemic but the FBC returns a result of anaemia, due to tests specificity, we
            # assume the reported anaemia is mild
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                dx_tests_to_run='full_blood_count_hb', hsi_event=hsi_event) and (df.at[person_id,
                                                                                       'ps_anaemia_'
                                                                                       'in_pregnancy'] == 'none'):
                    return 'non_severe'

            # If the test correctly identifies a woman's anaemia we assume it correctly identifies its severity
            elif self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                dx_tests_to_run='full_blood_count_hb', hsi_event=hsi_event) and (df.at[person_id,
                                                                                       'ps_anaemia'
                                                                                       '_in_pregnancy'] != 'none'):
                return df.at[person_id, 'ps_anaemia_in_pregnancy']

            # We return a none value if no anaemia was detected
            else:
                return 'none'

        else:
            logger.debug(key='message', data=f'There were inadequate consumables to conduct an FBC so mother '
                                             f'{person_id} did not receive one')
            return 'no_test'

    def treatment_of_anaemia_causing_deficiencies(self, individual_id, hsi_event):
        """This function is called for women admitted during pregnancy with anaemia. It contains treatment for
        underlying deficiencies that can contribute to maternal anaemia"""
        # TODO: this might get removed as not part of the EHP

        df = self.sim.population.props
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        params = self.parameters
        pregnancy_deficiencies = self.sim.modules['PregnancySupervisor'].deficiencies_in_pregnancy

        # Check and request consumables
        item_code_elemental_iron = pd.unique(
            consumables.loc[
                consumables['Items'] ==
                'Iron syrup, 20 mg/ml', 'Item_Code'])[0]

        item_code_folate = pd.unique(  # todo: no consumable for folate
            consumables.loc[
                consumables['Items'] ==
                'Ferrous Salt + Folic Acid, tablet, 200 + 0.25 mg', 'Item_Code'])[0]

        item_code_b12 = pd.unique(
            consumables.loc[
                consumables['Items'] ==
                'vitamin B12 (cyanocobalamine) 1 mg/ml, 1 ml, inj._100_IDA', 'Item_Code'])[0]

        consumables_ifa = {
            'Intervention_Package_Code': {},
            'Item_Code': {item_code_elemental_iron: 1,
                          item_code_folate: 1,
                          item_code_b12: 1}}  # TODO: correct quantities (pregnancy long courses)

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=hsi_event,
            cons_req_as_footprint=consumables_ifa)

        # Treatment is provided dependent on deficiencies present
        # We apply a probability that treatment will resolve a womans anaemia and the deficiency (understanding that
        # anaemia can be multi-factoral)

        # TODO: note to TC r.e. complexity

        if pregnancy_deficiencies.has_any([individual_id], 'iron', first=True):
            if outcome_of_request_for_consumables['Item_Code'][item_code_elemental_iron]:
                if self.rng.random_sample() < params['effect_of_iron_replacement_for_resolving_anaemia']:
                    df.at[individual_id, 'ps_anaemia_in_pregnancy'] = 'none'

                if self.rng.random_sample() < params['effect_of_iron_replacement_for_resolving_iron_def']:
                    pregnancy_deficiencies.unset([individual_id], 'iron')

        if pregnancy_deficiencies.has_any([individual_id], 'folate', first=True):
            if outcome_of_request_for_consumables['Item_Code'][item_code_folate]:
                if self.rng.random_sample() < params['effect_of_folate_replacement_for_resolving_anaemia']:
                    df.at[individual_id, 'ps_anaemia_in_pregnancy'] = 'none'
                if self.rng.random_sample() < params['effect_of_folate_replacement_for_resolving_folate_def']:
                    pregnancy_deficiencies.unset([individual_id], 'folate')

        if pregnancy_deficiencies.has_any([individual_id], 'b12', first=True):
            if outcome_of_request_for_consumables['Item_Code'][item_code_b12]:
                if self.rng.random_sample() < params['effect_of_b12_replacement_for_resolving_anaemia']:
                    df.at[individual_id, 'ps_anaemia_in_pregnancy'] = 'none'
                if self.rng.random_sample() < params['effect_of_b12_replacement_for_resolving_b12_def']:
                    pregnancy_deficiencies.unset([individual_id], 'b12')

    def start_iron_and_folic_acid(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        approx_days_of_pregnancy = (40 - df.at[individual_id, 'ps_gestational_age_in_weeks']) * 7

        # Check availibility of consumables
        item_code_iron_folic_acid = pd.unique(
            consumables.loc[
                consumables['Items'] ==
                'Ferrous Salt + Folic Acid, tablet, 200 + 0.25 mg', 'Item_Code'])[0]

        consumables_ifa = {
            'Intervention_Package_Code': {},
            'Item_Code': {item_code_iron_folic_acid: approx_days_of_pregnancy}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self,
            cons_req_as_footprint=consumables_ifa)

        # Start iron and folic acid treatment
        if outcome_of_request_for_consumables['Item_Code'][item_code_iron_folic_acid]:
            df.at[individual_id, 'ac_receiving_iron_folic_acid'] = True

            # Women started on IFA at this stage are already anaemic, we here apply a probability that
            # starting on a course of IFA will correct anaemia prior to follow up
            if self.module.rng.random_sample() < params['effect_of_ifa_for_resolving_anaemia']:
                df.at[individual_id, 'ps_anaemia_in_pregnancy'] = 'none'

    def antenatal_blood_transfusion(self, individual_id, hsi_event, cause):
        """This function houses the antenatal blood transfusion intervention"""

        df = self.sim.population.props
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        params = self.parameters

        # assert df.at[individual_id, 'ps_anaemia_in_pregnancy'] == 'severe'

        # Check for consumables
        item_code_blood = pd.unique(consumables.loc[consumables['Items'] == 'Blood, one unit', 'Item_Code'])[0]
        item_code_needle = pd.unique(consumables.loc[consumables['Items'] == 'Lancet, blood, disposable',
                                                     'Item_Code'])[0]
        item_code_test = pd.unique(consumables.loc[consumables['Items'] == 'Test, hemoglobin', 'Item_Code'])[0]
        item_code_giving_set = pd.unique(consumables.loc[consumables['Items'] == 'IV giving/infusion set, with needle',
                                                         'Item_Code'])[0]

        consumables_needed_bt = {'Intervention_Package_Code': {},
                                 'Item_Code': {item_code_blood: 2, item_code_needle: 1, item_code_test: 1,
                                               item_code_giving_set: 2}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=hsi_event, cons_req_as_footprint=consumables_needed_bt)

        if (outcome_of_request_for_consumables['Item_Code'][item_code_blood]) and \
                (outcome_of_request_for_consumables['Item_Code'][item_code_test]) and \
                (outcome_of_request_for_consumables['Item_Code'][item_code_needle]) and \
                (outcome_of_request_for_consumables['Item_Code'][item_code_giving_set]):

            logger.debug(key='msg', data=f'Mother {individual_id} is receiving an antenatal blood transfusion due '
                                         f'to {cause}')

            if cause == 'severe_anaemia':
                # If available, we apply a probability that a transfusion of 2 units RBCs will correct this woman's
                # severe anaemia
                if params['treatment_effect_blood_transfusion_anaemia'] < self.rng.random_sample():
                    df.at[individual_id, 'ps_anaemia_in_pregnancy'] = 'none'
            elif cause == 'abortion':
                self.pac_interventions.set(individual_id, 'blood_products')
            else:
                df.at[individual_id, 'ac_received_blood_transfusion'] = True

    def initiate_maintenance_anti_hypertensive_treatment(self, individual_id, hsi_event):
        """

        :param individual_id:
        :param hsi_event:
        :return:
        """
        df = self.sim.population.props
        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]
        approx_days_of_pregnancy = (40 - df.at[individual_id, 'ps_gestational_age_in_weeks']) * 7

        item_code_methyldopa = pd.unique(
            consumables.loc[consumables['Items'] == 'Methyldopa 250mg_1000_CMST', 'Item_Code'])[0]

        consumables_gest_htn_treatment = {
            'Intervention_Package_Code': {},
            'Item_Code': {item_code_methyldopa: 4 * approx_days_of_pregnancy }}

        # Then query if these consumables are available during this HSI
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=hsi_event,
            cons_req_as_footprint=consumables_gest_htn_treatment)

        # If they are available then the woman is started on treatment
        if outcome_of_request_for_consumables['Item_Code'][item_code_methyldopa]:
            df.at[individual_id, 'ac_gest_htn_on_treatment'] = True

    def initiate_treatment_for_severe_hypertension(self, individual_id, hsi_event ):
        """

        :param individual_id:
        :param hsi_event:
        :return:
        """

        df = self.sim.population.props
        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

        item_code_hydralazine = pd.unique(
            consumables.loc[consumables['Items'] == 'Hydralazine, powder for injection, 20 mg ampoule',
                            'Item_Code'])[0]
        item_code_wfi = pd.unique(
            consumables.loc[consumables['Items'] == 'Water for injection, 10ml_Each_CMST', 'Item_Code'])[0]
        item_code_needle = pd.unique(
            consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]
        item_code_gloves = pd.unique(
            consumables.loc[consumables['Items'] == 'Gloves, exam, latex, disposable, pair', 'Item_Code'])[0]

        consumables_gest_htn_treatment = {
            'Intervention_Package_Code': {},
            'Item_Code': {item_code_hydralazine: 1,
                          item_code_wfi: 1,
                          item_code_gloves: 1,
                          item_code_needle: 1}}

        # Then query if these consumables are available during this HSI
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=hsi_event,
            cons_req_as_footprint=consumables_gest_htn_treatment)

        # If they are available then the woman is started on treatment
        if (outcome_of_request_for_consumables['Item_Code'][item_code_hydralazine]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_wfi]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_needle]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_gloves]):

            if df.at[individual_id, 'ps_htn_disorders'] == 'severe_gest_htn':
                df.at[individual_id, 'ps_htn_disorders'] = 'gest_htn'

            if df.at[individual_id, 'ps_htn_disorders'] == 'severe_pre_eclamp' or \
                df.at[individual_id, 'ps_htn_disorders'] == 'eclampsia':
                logger.debug(key='msg', data=f'Mother {individual_id} has been given intravenous anti-hypertensive as '
                                             f'part of treatment regime for severe pre-eclampsia/eclampsia')
                df.at[individual_id, 'ac_iv_anti_htn_treatment'] = True

    def treatment_for_severe_pre_eclampsia_or_eclampsia(self, individual_id, cause, hsi_event):
        """

        :param individual_id:
        :param cause:
        :param hsi_event:
        :return:
        """
        df = self.sim.population.props
        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

        pkg_code_eclampsia_and_spe = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Management of eclampsia',
                            'Intervention_Pkg_Code'])[0]

        all_available = hsi_event.get_all_consumables(
            pkg_codes=[pkg_code_eclampsia_and_spe])

        if all_available:
            df.at[individual_id, 'ac_mag_sulph_treatment'] = True

            # df.at[individual_id, f'ac_{cause}treatment'] = True

    def antibiotics_for_prom(self, individual_id, hsi_event):
        df = self.sim.population.props

        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

        item_code_benpen = pd.unique(
            consumables.loc[consumables['Items'] == 'Benzathine benzylpenicillin, powder for injection, 2.4 million IU',
                            'Item_Code'])[0]
        item_code_wfi = pd.unique(
            consumables.loc[consumables['Items'] == 'Water for injection, 10ml_Each_CMST', 'Item_Code'])[0]
        item_code_needle = pd.unique(
            consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]
        item_code_gloves = pd.unique(
            consumables.loc[consumables['Items'] == 'Gloves, exam, latex, disposable, pair', 'Item_Code'])[0]

        consumables_abx_for_prom = {
            'Intervention_Package_Code': {},
            'Item_Code': {item_code_benpen: 4, item_code_wfi: 1, item_code_needle: 1,
                          item_code_gloves: 1}}

        # todo: consumables for whole inpatient stay?

        # Then query if these consumables are available during this HSI
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=hsi_event,
            cons_req_as_footprint=consumables_abx_for_prom)

        if (outcome_of_request_for_consumables['Item_Code'][item_code_benpen]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_wfi]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_needle]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_gloves]):

            df.at[individual_id, 'ac_received_abx_for_prom'] = True

    def antibiotics_for_chorioamnionitis(self, individual_id, hsi_event):
        df = self.sim.population.props
        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

        item_code_benpen = pd.unique(
            consumables.loc[consumables['Items'] == 'Benzathine benzylpenicillin, powder for injection, 2.4 million IU',
                            'Item_Code'])[0]
        item_code_genta = pd.unique(
            consumables.loc[consumables['Items'] == 'Gentamycin, injection, 40 mg/ml in 2 ml vial',
                            'Item_Code'])[0]  # TODO: dose?
        item_code_wfi = pd.unique(
            consumables.loc[consumables['Items'] == 'Water for injection, 10ml_Each_CMST', 'Item_Code'])[0]
        item_code_needle = pd.unique(
            consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]
        item_code_gloves = pd.unique(
            consumables.loc[consumables['Items'] == 'Gloves, exam, latex, disposable, pair', 'Item_Code'])[0]

        consumables_abx_for_chorio = {
            'Intervention_Package_Code': {},
            'Item_Code': {item_code_benpen: 4, item_code_genta: 1, item_code_wfi: 1, item_code_needle: 1,
                          item_code_gloves: 1}}

        # todo: consumables for whole inpatient stay?

        # Then query if these consumables are available during this HSI
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=hsi_event,
            cons_req_as_footprint=consumables_abx_for_chorio)

        if (outcome_of_request_for_consumables['Item_Code'][item_code_benpen]) and\
            (outcome_of_request_for_consumables['Item_Code'][item_code_genta]) and\
            (outcome_of_request_for_consumables['Item_Code'][item_code_wfi]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_needle]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_gloves]):

            df.at[individual_id, 'ac_received_abx_for_chorioamnionitis'] = True

    def apply_risk_of_death_following_antenatal_treatment(self, individual_id):
        df = self.sim.population.props
        params = self.parameters
        mother = df.loc[individual_id]

        death = False
        still_birth = False

        if mother.ps_htn_disorders == 'severe_pre_eclampsia' or mother.ps_htn_disorders == 'eclampsia':

            risk_of_death = params['ac_linear_equations']['ec_spe_death_post_treatment'].predict(
                            df.loc[[individual_id]])[individual_id]

            if self.rng.random_sample() < risk_of_death:
                logger.debug(key='msg', data=f'Mother {individual_id} has died following treatment for severe '
                                             f'pre-eclampsia/eclampsia')

                death = True
            else:
                if df.at[individual_id, 'ps_htn_disorders'] == 'eclampsia':
                    df.at[individual_id, 'ps_htn_disorders'] = 'severe_pre_eclamp'

                # todo: does treatment effect risk of still birth
                if self.rng.random_sample() < params['prob_still_birth_spe_ec']:
                    logger.debug(key='msg', data=f'Mother {individual_id} has experience a still birth following severe'
                                             f'pre-eclampsia/eclampsia')
                    still_birth = True

        if mother.ps_antepartum_haemorrhage and (mother.ac_admitted_for_immediate_delivery == 'induction_future' or
                                                 mother.ac_admitted_for_immediate_delivery == 'caesarean_future'):

            risk_of_death = params['ac_linear_equations']['aph_death_post_treatment'].predict(
                df.loc[[individual_id]])[individual_id]

            if self.rng.random_sample() < risk_of_death:
                logger.debug(key='msg', data=f'Mother {individual_id} has died following treatment for antepartum '
                                             f'haemorrhage')

                death = True

            elif self.rng.random_sample() < params['prob_still_birth_aph']:
                logger.debug(key='msg', data=f'Mother {individual_id} has experience a still birth following antepartum '
                                             f'bleeding')
                still_birth = True

        if death:
            return 'maternal death'
        elif still_birth:
            return 'still_birth'
        else:
            return 'survival'


class HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareContact(HSI_Event, IndividualScopeEventMixin):
    """ This is the  HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareContact. It will be scheduled by the
    PregnancySupervisor Module.It will be responsible for the management of monitoring and treatment interventions
    delivered in a woman's first antenatal care visit. It will also go on the schedule the womans next ANC
    appointment."""

    def __init__(self, module, person_id, facility_level_of_this_hsi):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_PresentsForFirstAntenatalCareVisit'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'AntenatalFirst': 1})
        self.ACCEPTED_FACILITY_LEVEL = facility_level_of_this_hsi
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]

        # If the mother is no longer alive
        if not mother.is_alive:
            return

        # The contents of this HSI will only run for women who are still pregnant and are not currently in labour
        if mother.is_pregnant and ~mother.la_currently_in_labour:

            # Women may have lost a pregnancy and become pregnant again meaning this event will still run, this prevents
            # incorrect HSIs running
            if mother.ps_date_of_anc1 != self.sim.date or (mother.ac_total_anc_visits_current_pregnancy > 0) or \
              (mother.ps_gestational_age_in_weeks < 7):
                logger.debug(key='msg', data=f'mother {person_id} has arrived at ANC1 that was scheduled in a previous '
                                             f'pregnancy and therefore the event will not run')
                return

            assert mother.ac_total_anc_visits_current_pregnancy == 0

            # We capture the facility type that this HSI is occurring at (dependent on facility level) - we assume women
            # will present to the same facility level/type for any future ANC visits
            if self.ACCEPTED_FACILITY_LEVEL == 1:
                facility_type = self.module.rng.choice(['health_centre', 'hospital'], p=[0.5, 0.5])
                df.at[person_id, 'ac_facility_type'] = facility_type
                logger.info(key='anc_facility_type', data=f'{facility_type}')
            elif self.ACCEPTED_FACILITY_LEVEL > 1:
                logger.info(key='anc_facility_type', data='hospital')
                df.at[person_id, 'ac_facility_type'] = 'hospital'

            logger.debug(key='message',
                         data=f'mother {person_id} presented for ANC1 at a {df.at[person_id, "ac_facility_type"]} at '
                              f'gestation {df.at[person_id, "ps_gestational_age_in_weeks"]} ')

            # We add a visit to a rolling total of ANC visits in this pregnancy
            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            # And ensure only women whose first contact with ANC services are attending this event
            assert mother.ps_gestational_age_in_weeks is not None
            assert mother.ps_gestational_age_in_weeks is not pd.NaT
            assert mother.ps_gestational_age_in_weeks >= 7

            # We store some information for summary statistics
            self.module.anc_tracker['cumm_ga_at_anc1'] += df.at[person_id, 'ps_gestational_age_in_weeks']
            self.module.anc_tracker['total_first_anc_visits'] += 1
            if mother.ps_gestational_age_in_weeks < 14:
                self.module.anc_tracker['total_anc1_first_trimester'] += 1

            # The following function returns the timing of the next visit this woman must attend in her ANC shedule
            # which is used to scheduled her next visit
            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)

            #  =================================== INTERVENTIONS ====================================================
            # First all women, regardless of ANC contact or gestation, undergo urine and blood pressure measurement
            self.module.screening_interventions_delivered_at_every_contact(hsi_event=self)

            # Then, the appropriate interventions are delivered according gestational age. It is assumed that women who
            # present late to ANC1 are 'caught up' with the interventions they missed from previous visits
            # (as per malawi guidelines)

            # If this woman is presenting prior to the suggested gestation for ANC2, she receives only the interventions
            # for ANC1
            if mother.ps_gestational_age_in_weeks < 20:
                # These are the interventions delivered at ANC1
                self.module.interventions_initiated_at_first_contact(hsi_event=self)
                self.module.hiv_testing(hsi_event=self)
                self.module.hep_b_testing(hsi_event=self)
                self.module.syphilis_testing(hsi_event=self)
                self.module.point_of_care_hb_testing(hsi_event=self)
                self.module.tetanus_vaccination(hsi_event=self)

                # She is then assessed to see if she will attend the next ANC contact in the schedule
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=2,
                                                     recommended_gestation_next_anc=gest_age_next_contact,
                                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)

            elif mother.ps_gestational_age_in_weeks < 26:
                self.module.interventions_initiated_at_first_contact(hsi_event=self)
                self.module.anc_catch_up_interventions(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=2,
                                                     recommended_gestation_next_anc=gest_age_next_contact,
                                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)

            elif mother.ps_gestational_age_in_weeks < 40:
                self.module.interventions_initiated_at_first_contact(hsi_event=self)
                self.module.anc_catch_up_interventions(hsi_event=self)

                # Women presenting at >26 are indicated to require screening for gestational diabetes
                # (usually delivered in ANC 2)
                self.module.gdm_screening(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=2,
                                                     recommended_gestation_next_anc=gest_age_next_contact,
                                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)

            elif mother.ps_gestational_age_in_weeks >= 40:
                self.module.interventions_initiated_at_first_contact(hsi_event=self)
                self.module.anc_catch_up_interventions(hsi_event=self)
                self.module.gdm_screening(hsi_event=self)
                # todo: schedule for induction?

            # If the woman has had any complications detected during ANC she is admitted for treatment to be initiated
            if df.at[person_id, 'ac_to_be_admitted']:
                self.module.schedule_admission(person_id)

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareVisit: did not run')

    def not_available(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareVisit: cannot not run with '
                                         'this configuration')


class HSI_CareOfWomenDuringPregnancy_SecondAntenatalCareContact(HSI_Event, IndividualScopeEventMixin):
    """This is the  HSI_CareOfWomenDuringPregnancy_SecondAntenatalCareContact. It will be scheduled by the
    HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareContact.It will be responsible for the management of monitoring
    and treatment interventions delivered in a woman's second antenatal care visit. It will also go on the schedule the
    womans next ANC appointment"""
    def __init__(self, module, person_id, facility_level_of_this_hsi):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_SecondAntenatalCareVisit'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = facility_level_of_this_hsi
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]

        if not mother.is_alive:
            return

        # todo: add note r.e. squeeze factor

        if mother.is_pregnant and ~mother.la_currently_in_labour and ~mother.ac_inpatient:

            if self.sim.date != mother.ac_date_next_contact:
                return

            logger.info(key='anc_facility_type', data=f'{df.at[person_id, "ac_facility_type"]}')
            logger.debug(key='message',
                         data=f'mother {person_id}presented for ANC 2 at a {df.at[person_id, "ac_facility_type"]} at '
                              f'gestation {df.at[person_id, "ps_gestational_age_in_weeks"]} ')

            assert mother.ac_total_anc_visits_current_pregnancy == 1
            assert mother.ps_gestational_age_in_weeks is not None
            assert mother.ps_gestational_age_in_weeks is not pd.NaT
            assert mother.ps_gestational_age_in_weeks >= 19

            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)
            self.module.screening_interventions_delivered_at_every_contact(hsi_event=self)
            self.module.tetanus_vaccination(hsi_event=self)

            if mother.ps_gestational_age_in_weeks < 26:
                self.module.albendazole_administration(hsi_event=self)
                self.module.iptp_administration(hsi_event=self)
                self.module.calcium_supplementation(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=3,
                                                     recommended_gestation_next_anc=gest_age_next_contact,
                                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)

            elif mother.ps_gestational_age_in_weeks < 30:
                self.module.iptp_administration(hsi_event=self)
                self.module.calcium_supplementation(hsi_event=self)
                self.module.gdm_screening(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=3,
                                                     recommended_gestation_next_anc=gest_age_next_contact,
                                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)

            elif mother.ps_gestational_age_in_weeks < 34:
                self.module.iptp_administration(hsi_event=self)
                self.module.calcium_supplementation(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=3,
                                                     recommended_gestation_next_anc=gest_age_next_contact,
                                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)

            elif mother.ps_gestational_age_in_weeks < 36:
                self.module.iptp_administration(hsi_event=self)
                self.module.calcium_supplementation(hsi_event=self)

                self.module.hep_b_testing(hsi_event=self)
                self.module.syphilis_testing(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=3,
                                                     recommended_gestation_next_anc=gest_age_next_contact,
                                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)

            elif mother.ps_gestational_age_in_weeks < 38:
                self.module.calcium_supplementation(hsi_event=self)
                self.module.point_of_care_hb_testing(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=3,
                                                     recommended_gestation_next_anc=gest_age_next_contact,
                                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)

            elif mother.ps_gestational_age_in_weeks < 40:
                self.module.calcium_supplementation(hsi_event=self)
                self.module.iptp_administration(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=3,
                                                     recommended_gestation_next_anc=gest_age_next_contact,
                                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)

            elif mother.ps_gestational_age_in_weeks >= 40:
                pass

            if df.at[person_id, 'ac_to_be_admitted']:
                self.module.schedule_admission(person_id)

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        return actual_appt_footprint

    def did_not_run(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_SecondAntenatalCareVisit: did not run')

    def not_available(self):
        logger.debug(key='message',data='HSI_CareOfWomenDuringPregnancy_SecondAntenatalCareVisit: cannot not run with '
                                        'this configuration')


class HSI_CareOfWomenDuringPregnancy_ThirdAntenatalCareContact(HSI_Event, IndividualScopeEventMixin):
    """This is the  HSI_CareOfWomenDuringPregnancy_ThirdAntenatalCareContact. It will be scheduled by the
    HSI_CareOfWomenDuringPregnancy_SecondAntenatalCareContact.It will be responsible for the management of monitoring
    and treatment interventions delivered in a woman's third antenatal care visit. It will also go on the schedule the
    womans next ANC appointment"""
    def __init__(self, module, person_id, facility_level_of_this_hsi):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_PresentsForSubsequentAntenatalCareVisit'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
        self.ACCEPTED_FACILITY_LEVEL = facility_level_of_this_hsi
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]

        if not mother.is_alive:
            return

        # todo: add note r.e. squeeze factor

        if mother.is_pregnant and ~mother.la_currently_in_labour and ~mother.ac_inpatient:

            if self.sim.date != mother.ac_date_next_contact:
                return

            logger.info(key='anc_facility_type', data=f'{df.at[person_id, "ac_facility_type"]}')
            logger.debug(key='message',
                         data=f'mother {person_id}presented for ANC 3 at a {df.at[person_id, "ac_facility_type"]} at '
                              f'gestation {df.at[person_id, "ps_gestational_age_in_weeks"]} ')

            assert mother.ac_total_anc_visits_current_pregnancy == 2
            assert mother.ps_gestational_age_in_weeks is not None
            assert mother.ps_gestational_age_in_weeks is not pd.NaT
            assert mother.ps_gestational_age_in_weeks >= 25

            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            self.module.screening_interventions_delivered_at_every_contact(hsi_event=self)
            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)

            if mother.ps_gestational_age_in_weeks < 30:
                self.module.iptp_administration(hsi_event=self)
                self.module.calcium_supplementation(hsi_event=self)
                self.module.gdm_screening(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=4,
                                                     recommended_gestation_next_anc=gest_age_next_contact,
                                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)

            elif mother.ps_gestational_age_in_weeks < 34:
                self.module.iptp_administration(hsi_event=self)
                self.module.calcium_supplementation(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=4,
                                                     recommended_gestation_next_anc=gest_age_next_contact,
                                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)

            elif mother.ps_gestational_age_in_weeks < 36:
                self.module.iptp_administration(hsi_event=self)
                self.module.calcium_supplementation(hsi_event=self)

                self.module.hep_b_testing(hsi_event=self)
                self.module.syphilis_testing(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=4,
                                                     recommended_gestation_next_anc=gest_age_next_contact,
                                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)

            elif mother.ps_gestational_age_in_weeks < 38:
                self.module.calcium_supplementation(hsi_event=self)
                self.module.point_of_care_hb_testing(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=4,
                                                     recommended_gestation_next_anc=gest_age_next_contact,
                                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)

            elif mother.ps_gestational_age_in_weeks < 40:
                self.module.calcium_supplementation(hsi_event=self)
                self.module.iptp_administration(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=4,
                                                     recommended_gestation_next_anc=gest_age_next_contact,
                                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)

            elif mother.ps_gestational_age_in_weeks >= 40:
                pass

            if df.at[person_id, 'ac_to_be_admitted']:
                self.module.schedule_admission(person_id)

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        return actual_appt_footprint

    def did_not_run(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_ThirdAntenatalCareContact: did not run')

    def not_available(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_ThirdAntenatalCareContact: cannot not run '
                                         'with this configuration')


class HSI_CareOfWomenDuringPregnancy_FourthAntenatalCareContact(HSI_Event, IndividualScopeEventMixin):
    """This is the  HSI_CareOfWomenDuringPregnancy_FourthAntenatalCareContact. It will be scheduled by the
    HSI_CareOfWomenDuringPregnancy_ThirdAntenatalCareContact.It will be responsible for the management of monitoring
    and treatment interventions delivered in a woman's fourth antenatal care visit. It will also go on the schedule the
    womans next ANC appointment"""

    def __init__(self, module, person_id, facility_level_of_this_hsi):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_FourthAntenatalCareVisit'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = facility_level_of_this_hsi
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]

        if not mother.is_alive:
            return

        # todo: add note r.e. squeeze factor

        if mother.is_pregnant and ~mother.la_currently_in_labour and ~mother.ac_inpatient:

            if self.sim.date != mother.ac_date_next_contact:
                return

            logger.info(key='anc_facility_type', data=f'{df.at[person_id, "ac_facility_type"]}')
            logger.debug(key='message',
                         data=f'mother {person_id}presented for ANC 4 at a {df.at[person_id, "ac_facility_type"]} at '
                              f'gestation {df.at[person_id, "ps_gestational_age_in_weeks"]} ')

            assert mother.ac_total_anc_visits_current_pregnancy == 3
            assert mother.ps_gestational_age_in_weeks is not None
            assert mother.ps_gestational_age_in_weeks is not pd.NaT
            assert mother.ps_gestational_age_in_weeks >= 29

            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            self.module.screening_interventions_delivered_at_every_contact(hsi_event=self)
            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)

            if df.at[person_id, 'ps_gestational_age_in_weeks'] < 34:
                self.module.iptp_administration(hsi_event=self)
                self.module.calcium_supplementation(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=5,
                                                     recommended_gestation_next_anc=gest_age_next_contact,
                                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] < 36:
                self.module.iptp_administration(hsi_event=self)
                self.module.calcium_supplementation(hsi_event=self)

                self.module.hep_b_testing(hsi_event=self)
                self.module.syphilis_testing(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=5,
                                                     recommended_gestation_next_anc=gest_age_next_contact,
                                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] < 38:
                self.module.calcium_supplementation(hsi_event=self)
                self.module.point_of_care_hb_testing(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=5,
                                                     recommended_gestation_next_anc=gest_age_next_contact,
                                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] < 40:
                self.module.calcium_supplementation(hsi_event=self)
                self.module.iptp_administration(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=5,
                                                     recommended_gestation_next_anc=gest_age_next_contact,
                                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] >= 40:
                pass

            if df.at[person_id, 'ac_to_be_admitted']:
                self.module.schedule_admission(person_id)

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        return actual_appt_footprint

    def did_not_run(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_FourthAntenatalCareContact: did not run')

    def not_available(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_FourthAntenatalCareContact: cannot not run '
                                         'with this configuration')


class HSI_CareOfWomenDuringPregnancy_FifthAntenatalCareContact(HSI_Event, IndividualScopeEventMixin):
    """This is the  HSI_CareOfWomenDuringPregnancy_FifthAntenatalCareContact. It will be scheduled by the
    HSI_CareOfWomenDuringPregnancy_FourthAntenatalCareContact.It will be responsible for the management of monitoring
    and treatment interventions delivered in a woman's fifth antenatal care visit. It will also go on the schedule the
    womans next ANC appointment"""
    def __init__(self, module, person_id, facility_level_of_this_hsi):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_EmergencyTreatment'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
        self.ACCEPTED_FACILITY_LEVEL = facility_level_of_this_hsi
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]

        if not mother.is_alive:
            return

        # todo: add note r.e. squeeze factor

        if mother.is_pregnant and ~mother.la_currently_in_labour and ~mother.ac_inpatient:

            if self.sim.date != mother.ac_date_next_contact:
                return

            logger.info(key='anc_facility_type', data=f'{df.at[person_id, "ac_facility_type"]}')
            logger.debug(key='message',
                         data=f'mother {person_id}presented for ANC 5 at a {df.at[person_id, "ac_facility_type"]} at '
                              f'gestation {df.at[person_id, "ps_gestational_age_in_weeks"]} ')

            assert mother.ps_gestational_age_in_weeks is not None
            assert mother.ps_gestational_age_in_weeks is not pd.NaT
            assert mother.ps_gestational_age_in_weeks >= 33
            assert mother.ac_total_anc_visits_current_pregnancy == 4

            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            self.module.screening_interventions_delivered_at_every_contact(hsi_event=self)
            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)

            if mother.ps_gestational_age_in_weeks < 36:
                self.module.iptp_administration(hsi_event=self)
                self.module.calcium_supplementation(hsi_event=self)

                self.module.hep_b_testing(hsi_event=self)
                self.module.syphilis_testing(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=6,
                                                     recommended_gestation_next_anc=gest_age_next_contact,
                                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)

            elif mother.ps_gestational_age_in_weeks < 38:
                self.module.calcium_supplementation(hsi_event=self)
                self.module.point_of_care_hb_testing(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=6,
                                                     recommended_gestation_next_anc=gest_age_next_contact,
                                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)

            elif mother.ps_gestational_age_in_weeks < 40:
                self.module.calcium_supplementation(hsi_event=self)
                self.module.iptp_administration(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=6,
                                                     recommended_gestation_next_anc=gest_age_next_contact,
                                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)

            elif mother.ps_gestational_age_in_weeks >= 40:
                pass

            if df.at[person_id, 'ac_to_be_admitted']:
                self.module.schedule_admission(person_id)

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        return actual_appt_footprint

    def did_not_run(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_FifthAntenatalCareContact: did not run')

    def not_available(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_FifthAntenatalCareContact: cannot not run '
                                         'with this configuration')


class HSI_CareOfWomenDuringPregnancy_SixthAntenatalCareContact(HSI_Event, IndividualScopeEventMixin):
    """This is the  HSI_CareOfWomenDuringPregnancy_SixthAntenatalCareContact. It will be scheduled by the
    HSI_CareOfWomenDuringPregnancy_FifthAntenatalCareContact.It will be responsible for the management of monitoring
    and treatment interventions delivered in a woman's sixth antenatal care visit. It will also go on the schedule the
    womans next ANC appointment"""

    def __init__(self, module, person_id, facility_level_of_this_hsi):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_SixthAntenatalCareVisit'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = facility_level_of_this_hsi
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]

        if not mother.is_alive:
            return

        # todo: add note r.e. squeeze factor

        if mother.is_pregnant and ~mother.la_currently_in_labour and ~mother.ac_inpatient:

            if self.sim.date != mother.ac_date_next_contact:
                return

            logger.info(key='anc_facility_type', data=f'{df.at[person_id, "ac_facility_type"]}')
            logger.debug(key='message',
                         data=f'mother {person_id}presented for ANC 6 at a {df.at[person_id, "ac_facility_type"]} at '
                              f'gestation {df.at[person_id, "ps_gestational_age_in_weeks"]} ')

            assert mother.ac_total_anc_visits_current_pregnancy == 5
            assert mother.ps_gestational_age_in_weeks is not None
            assert mother.ps_gestational_age_in_weeks is not pd.NaT
            assert mother.ps_gestational_age_in_weeks >= 35

            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            self.module.screening_interventions_delivered_at_every_contact(hsi_event=self)
            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)

            if mother.ps_gestational_age_in_weeks < 38:
                self.module.calcium_supplementation(hsi_event=self)
                self.module.point_of_care_hb_testing(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=7,
                                                     recommended_gestation_next_anc=gest_age_next_contact,
                                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)

            elif mother.ps_gestational_age_in_weeks < 40:
                self.module.calcium_supplementation(hsi_event=self)
                self.module.iptp_administration(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=7,
                                                     recommended_gestation_next_anc=gest_age_next_contact,
                                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)

            elif mother.ps_gestational_age_in_weeks >= 40:
                pass

            if df.at[person_id, 'ac_to_be_admitted']:
                self.module.schedule_admission(person_id)

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        return actual_appt_footprint

    def did_not_run(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_SixthAntenatalCareContact: did not run')

    def not_available(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_SixthAntenatalCareContact: cannot not run with'
                                         'this configuration')


class HSI_CareOfWomenDuringPregnancy_SeventhAntenatalCareContact(HSI_Event, IndividualScopeEventMixin):
    """This is the  HSI_CareOfWomenDuringPregnancy_SeventhAntenatalCareContact. It will be scheduled by the
    HSI_CareOfWomenDuringPregnancy_SixthAntenatalCareContact.It will be responsible for the management of monitoring
    and treatment interventions delivered in a woman's seventh antenatal care visit. It will also go on the schedule the
    womans next ANC appointment"""

    def __init__(self, module, person_id, facility_level_of_this_hsi):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_SeventhAntenatalCareContact'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
        self.ACCEPTED_FACILITY_LEVEL = facility_level_of_this_hsi
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]

        if not mother.is_alive:
            return

        # todo: add note r.e. squeeze factor

        if mother.is_pregnant and ~mother.la_currently_in_labour and ~mother.ac_inpatient:

            if self.sim.date != mother.ac_date_next_contact:
                return

            logger.info(key='anc_facility_type', data=f'{df.at[person_id, "ac_facility_type"]}')
            logger.debug(key='message',
                         data=f'mother {person_id}presented for ANC 7 at a {df.at[person_id, "ac_facility_type"]} at '
                              f'gestation {df.at[person_id, "ps_gestational_age_in_weeks"]} ')

            assert mother.ac_total_anc_visits_current_pregnancy == 6
            assert mother.ps_gestational_age_in_weeks is not None
            assert mother.ps_gestational_age_in_weeks is not pd.NaT
            assert mother.ps_gestational_age_in_weeks >= 37

            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            self.module.screening_interventions_delivered_at_every_contact(hsi_event=self)
            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)

            if mother.ps_gestational_age_in_weeks < 40:
                self.module.calcium_supplementation(hsi_event=self)
                self.module.iptp_administration(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=8,
                                                     recommended_gestation_next_anc=gest_age_next_contact,
                                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] >= 40:
                pass

            if df.at[person_id, 'ac_to_be_admitted']:
                self.module.schedule_admission(person_id)

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        return actual_appt_footprint

    def did_not_run(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_SeventhAntenatalCareContact: did not run')

    def not_available(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_SeventhAntenatalCareContact: cannot not run'
                                         ' with this configuration')


class HSI_CareOfWomenDuringPregnancy_EighthAntenatalCareContact(HSI_Event, IndividualScopeEventMixin):
    """This is the  HSI_CareOfWomenDuringPregnancy_EighthAntenatalCareContact. It will be scheduled by the
        HSI_CareOfWomenDuringPregnancy_SeventhAntenatalCareContact.It will be responsible for the management of monitoring
        and treatment interventions delivered in a woman's eighth antenatal care visit """
    def __init__(self, module, person_id, facility_level_of_this_hsi):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_EighthAntenatalCareContact'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = facility_level_of_this_hsi
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]

        if not mother.is_alive:
            return

        # todo: add note r.e. squeeze factor

        if mother.is_pregnant and ~mother.la_currently_in_labour and ~mother.ac_inpatient:

            if self.sim.date != mother.ac_date_next_contact:
                return

            logger.info(key='anc_facility_type', data=f'{df.at[person_id, "ac_facility_type"]}')
            logger.debug(key='message',
                         data=f'mother {person_id}presented for ANC 7 at a {df.at[person_id, "ac_facility_type"]} at '
                              f'gestation {df.at[person_id, "ps_gestational_age_in_weeks"]} ')

            assert mother.ac_total_anc_visits_current_pregnancy == 7
            assert mother.ps_gestational_age_in_weeks is not None
            assert mother.ps_gestational_age_in_weeks is not pd.NaT
            assert mother.ps_gestational_age_in_weeks >= 39

            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1
            self.module.anc_tracker['anc8+'] += 1

            self.module.screening_interventions_delivered_at_every_contact(hsi_event=self)
            self.module.calcium_supplementation(hsi_event=self)

            if df.at[person_id, 'ac_to_be_admitted']:
                self.module.schedule_admission(person_id)

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        return actual_appt_footprint

    def did_not_run(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_EighthAntenatalCareContact: did not run')

    def not_available(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_EighthAntenatalCareContact: cannot not run'
                                         ' with this configuration')


class HSI_CareOfWomenDuringPregnancy_MaternalEmergencyAssessment(HSI_Event, IndividualScopeEventMixin):
    """"""
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_MaternalEmergencyAdmission'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        # TODO: We assume women present to this event during pregnancy with either APH, severe PE, eclampsia and
        #  infection (not modelled) - WOMEN WILL NEED ASSESSMENT TO ENSURE THEY GET THE RIGHT TREATMENT even if cause
        #  may be apparent

        # TODO: consider funneling this through the diagnostic algorithm module

        # TODO - Eclampsia - no assessment, immediate treatment, admit (for delivery plan)
        # TODO - Severe pre-eclampsia - blood pressure, urine dip, immediate treatment admit (for delivery plan)
        # TODO - APH - ultrasound?, bloods (hb), immediate treatment admit
        # TODO - Infection/sepsis - BP, temp etc, admit
        # TODO- PROM, examination/patient report

        # TODO: store diagnosis/cause and send to inpatient event- cause variable used to give treatment etc

        if df.at[person_id, 'is_alive'] and ~df.at[person_id, 'la_currently_in_labour']:
            df.at[person_id, 'ac_inpatient'] = True

            facility_level = int(self.module.rng.choice([1, 2, 3], p=[0.33, 0.33, 0.34]))
            admission = HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare(
                self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id,
                facility_level_this_hsi=facility_level)

            self.sim.modules['HealthSystem'].schedule_hsi_event(admission, priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

    def did_not_run(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_MaternalEmergencyAssessment: did not run')

    def not_available(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_MaternalEmergencyAssessment: cannot not run'
                                         ' with this configuration')


class HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare(HSI_Event, IndividualScopeEventMixin):
    """"""
    def __init__(self, module, person_id, facility_level_this_hsi):
        super().__init__(module, person_id=person_id)

        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_AntenatalWardInpatientCare'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1   # TODO: replace with THs inpatient function
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = facility_level_this_hsi
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]
        params = self.module.parameters
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        # TODO: effect of squeeze
        # todo: conidtion on not already being on traetment
        # TODO: ensure ac_inpatient = False on dx from inpatient services
        # todo: induction vs caesarean?


        if not mother.is_alive:
            return

        if mother.is_pregnant and ~mother.la_currently_in_labour:
            df.at[person_id, 'ac_inpatient'] = True

            logger.debug(key='message', data=f'Mother {person_id} has been admitted for treatment of a complication of '
                                             f'her pregnancy ')

            # ================================= INITIATE TREATMENT FOR ANAEMIA ========================================
            if mother.ps_anaemia_in_pregnancy != 'none':
                # Women who are referred from ANC or an outpatient appointment following point of care Hb testing are
                # admitted for a full blood count
                fbc_result = self.module.full_blood_count_testing(self)
                assert fbc_result == 'no_test' or 'none' or 'mild' or 'moderate' or 'severe'

                # If the test is not carried out, no treatment is provided and the woman is discharged
                if fbc_result == 'no_test':
                    logger.debug(key='message', data=f'No FBC given due to resource constraints')

                # If the result returns none, anaemia has not been detected via an FBC and the woman is discharged
                # without treatment
                elif fbc_result == 'none':
                    logger.debug(key='message', data=f'Mother {person_id} has not had anaemia detected via an FBC and '
                                                     f'will be discharged')

                # If the FBC detected non severe anaemia (Hb >7) she is treated
                elif fbc_result == 'non_severe':

                    # Women not receiving daily iron supplementation are started on this
                    if ~mother.ac_receiving_iron_folic_acid:
                        self.module.start_iron_and_folic_acid(person_id, self)

                    # Some anaemia causing deficiencies (folate, B12)  are detected through MCV reading on the
                    # FBC, and therefore can be treated
                    self.module.treatment_of_anaemia_causing_deficiencies(person_id, self)

                    # She is scheduled to return for a repeat point of care test in four weeks time to assess if she
                    # remains anaemic
                    follow_up_date = self.sim.date + DateOffset(days=28)

                    outpatient_checkup = HSI_CareOfWomenDuringPregnancy_AntenatalOutpatientFollowUp(
                        self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(outpatient_checkup, priority=0,
                                                                        topen=follow_up_date,
                                                                        tclose=follow_up_date + DateOffset(days=7))

                elif fbc_result == 'severe':
                    # In the case of severe anaemia (Hb <7) a woman receives a blood transfusion in addition to other
                    # treatments
                    self.module.antenatal_blood_transfusion(person_id, self, cause='severe_anaemia')
                    self.module.treatment_of_anaemia_causing_deficiencies(person_id, self)
                    if mother.ac_receiving_iron_folic_acid == 'False':
                        self.module.start_iron_and_folic_acid(person_id, self)

                    follow_up_date = self.sim.date + DateOffset(days=28)

                    outpatient_checkup = HSI_CareOfWomenDuringPregnancy_AntenatalOutpatientFollowUp(
                        self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(outpatient_checkup, priority=0,
                                                                        topen=follow_up_date,
                                                                        tclose=follow_up_date + DateOffset(days=7))

                # TODO: check malaria and HIV status and refer for treatment?
                # TODO: deworming/schisto treatment

            # ======================== INITIATE TREATMENT FOR GESTATIONAL DIABETES (case management) ==================
            if mother.ps_gest_diab == 'uncontrolled' and mother.ac_gest_diab_on_treatment == 'none':
                df.at[person_id, 'ac_gest_diab_on_treatment'] = 'diet_exercise'

                from tlo.methods.pregnancy_supervisor import GestationalDiabetesGlycaemicControlEvent
                self.sim.schedule_event(GestationalDiabetesGlycaemicControlEvent(
                    self.sim.modules['PregnancySupervisor'], person_id), self.sim.date + DateOffset(days=7))

                check_up_date = self.sim.date + DateOffset(days=28)

                outpatient_checkup = HSI_CareOfWomenDuringPregnancy_AntenatalOutpatientManagementOfGestationalDiabetes(
                        self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(outpatient_checkup, priority=0,
                                                                    topen=check_up_date,
                                                                    tclose=check_up_date + DateOffset(days=3))

            # =============================== INITIATE TREATMENT FOR HYPERTENSION =====================================
            if (mother.ps_htn_disorders == 'gest_htn' or mother.ps_htn_disorders == 'mild_pre_eclamp') and \
                ~mother.ac_gest_htn_on_treatment:
                self.module.initiate_maintenance_anti_hypertensive_treatment(person_id, self)

            elif mother.ps_htn_disorders == 'severe_gest_htn' and ~mother.ac_gest_htn_on_treatment:
                if ~mother.ac_gest_htn_on_treatment:
                    self.module.initiate_maintenance_anti_hypertensive_treatment(person_id, self)

                self.module.initiate_treatment_for_severe_hypertension(person_id, self)

            # todo: CS for eclampsia
            elif mother.ps_htn_disorders == 'severe_pre_eclamp' or mother.ps_htn_disorders == 'eclampsia':
                df.at[person_id, 'ac_admitted_for_immediate_delivery'] = 'induction_now'
                logger.debug(key='msg', data=f'{person_id} will be admitted for induction due to spe/ec')

                if ~mother.ac_gest_htn_on_treatment:
                    self.module.initiate_maintenance_anti_hypertensive_treatment(person_id, self)

                self.module.treatment_for_severe_pre_eclampsia_or_eclampsia(person_id, cause=mother.ps_htn_disorders,
                                                                            hsi_event=self)
                self.module.initiate_treatment_for_severe_hypertension(person_id, self)

            # ========================= INITIATE TREATMENT FOR ANTEPARTUM HAEMORRHAGE =================================
            if mother.ps_antepartum_haemorrhage != 'none':
                # ---------------------- APH SECONDARY TO PLACENTAL ABRUPTION -----------------------------------------
                if mother.ps_placental_abruption and mother.ps_gestational_age_in_weeks >= 28:
                    df.at[person_id, 'ac_admitted_for_immediate_delivery'] = 'caesarean_now'
                    logger.debug(key='msg', data=f'{person_id} will be admitted for caesarean due to APH')

                elif mother.ps_placental_abruption and mother.ps_gestational_age_in_weeks < 28:
                    df.at[person_id, 'ac_admitted_for_immediate_delivery'] = 'caesarean_future'
                    logger.debug(key='msg', data=f'{person_id} will be admitted for caesarean when her gestation has '
                                                 f'increased due to APH')
                    self.module.antenatal_blood_transfusion(person_id, self, cause='antepartum_haem')
                    #  TODO: inpatient days

                # ---------------------- APH SECONDARY TO PLACENTA PRAEVIA -----------------------------------------
                if mother.ps_placenta_praevia and mother.ps_antepartum_haemorrhage == 'severe' and\
                    mother.ps_gestational_age_in_weeks >= 28:
                    df.at[person_id, 'ac_admitted_for_immediate_delivery'] = 'caesarean_now'
                    logger.debug(key='msg', data=f'{person_id} will be admitted for caesarean due to APH')

                if mother.ps_placenta_praevia and mother.ps_antepartum_haemorrhage == 'mild_moderate':
                    df.at[person_id, 'ac_admitted_for_immediate_delivery'] = 'caesarean_future'
                    logger.debug(key='msg', data=f'{person_id} will be admitted for caesarean when her gestation has '
                                                 f'increased due to APH')

                    self.module.antenatal_blood_transfusion(person_id, self, cause='antepartum_haem')
                    #  TODO: inpatient days

            # ============================ INITIATE TREATMENT FOR PROM +/- CHORIO ====================================
            if mother.ps_premature_rupture_of_membranes and ~mother.ps_chorioamnionitis:
                self.module.antibiotics_for_prom(person_id, self)

                if mother.ps_gestational_age_in_weeks > 34:
                    df.at[person_id, 'ac_admitted_for_immediate_delivery'] = 'induction_now'
                    logger.debug(key='msg', data=f'{person_id} will be admitted for induction due to prom/chorio')

                elif mother.ps_gestational_age_in_weeks < 35:
                    df.at[person_id, 'ac_admitted_for_immediate_delivery'] = 'induction_future'
                    logger.debug(key='msg', data=f'{person_id} will be admitted for induction when her gestation has '
                                                 f'increase due prom/chorio')

                    # TODO: inpatient days

            elif mother.ps_premature_rupture_of_membranes and mother.ps_chorioamnionitis:
                self.module.antibiotics_for_chorioamnionitis(person_id, self)
                # todo:APPLY EFFECT OF ANTIBIOTICS ON IP SEPSIS DEATH IN LABOUR
                df.at[person_id, 'ac_admitted_for_immediate_delivery'] = 'induction_now'
                logger.debug(key='msg', data=f'{person_id} will be admitted for induction due to prom/chorio')

            # ======================== APPLY RISK OF DEATH (WHERE APPROPRIATE) =======================================
            # TODO: post SPE/EC, post APH (if not admitted for delivery (or including those?))
            # TODO: is this needed?

            treatment_outcome = self.module.apply_risk_of_death_following_antenatal_treatment(person_id)

            if treatment_outcome == 'maternal_death':
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, person_id,
                                                                      cause='maternal'), self.sim.date)

            elif treatment_outcome == 'still_birth':
                self.sim.modules['PregnancySupervisor'].update_variables_post_still_birth_for_individual(person_id)

            else:
                # ======================== ADMISSION FOR DELIVERY (INDUCTION) ========================================
                # TODO: WOMEN MAY GO INTO LABOUR WHILST WAITING FOR ADMISSION
                # TODO: ENSURE THEY ARE ON THE RIGHT PATHWAY
                if df.at[person_id, 'ac_admitted_for_immediate_delivery'] == 'induction_now':
                    self.sim.schedule_event(LabourOnsetEvent(self.sim.modules['Labour'], person_id),
                                            self.sim.date)

                elif df.at[person_id, 'ac_admitted_for_immediate_delivery'] == 'caesarean_now':
                    self.sim.schedule_event(LabourOnsetEvent(self.sim.modules['Labour'], person_id),
                                            self.sim.date)

                elif df.at[person_id, 'ac_admitted_for_immediate_delivery'] == 'caesarean_future' or \
                    df.at[person_id, 'ac_admitted_for_immediate_delivery'] == 'induction_future':

                    if mother.ps_gestational_age_in_weeks < 37:
                        days_until_safe_for_cs = int((37 * 7) - (mother.ps_gestational_age_in_weeks * 7))
                    else:
                        days_until_safe_for_cs = 1

                    admission_date = self.sim.date + DateOffset(days=days_until_safe_for_cs)
                    logger.debug(key='msg', data=f'Mother {person_id} will move to labour ward for '
                                                f'{df.at[person_id, "ac_admitted_for_immediate_delivery"]} on '
                                                f'{admission_date}')
                    self.sim.schedule_event(LabourOnsetEvent(self.sim.modules['Labour'], person_id),
                                            admission_date)

                else:
                    df.at[person_id, 'ac_inpatient'] = False

    def did_not_run(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare: did not run')

    def not_available(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare: cannot not run'
                                         ' with this configuration')


class HSI_CareOfWomenDuringPregnancy_AntenatalOutpatientFollowUp(HSI_Event, IndividualScopeEventMixin):
    """"""
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_AntenatalOutpatientFollowUp'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1   # TODO: this might not take the right persons time
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        mother = df.loc[person_id]

        #  ------------------------------------- Follow up Hb testing -------------------------------------------------
        # TODO: prevent double referral if attend ANC inbetween treatment and follow up and become anaemic again

        if mother.is_alive and ~mother.la_currently_in_labour:
            if mother.ps_anaemia_in_pregnancy != 'none':
                fbc_result = self.module.full_blood_count_testing(self)
                if fbc_result == 'no_test':
                    logger.debug(key='message', data=f'No FBC given due to resource constraints')
                elif fbc_result == 'none':
                    logger.debug(key='message', data=f'Mother {person_id} has not had anaemia detected via an FBC')
                elif fbc_result == 'mild' or fbc_result == 'moderate' or fbc_result == 'severe':

                    facility_level = int(self.module.rng.choice([1, 2, 3], p=[0.33, 0.33, 0.34]))
                    admission = HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare(
                        self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id,
                        facility_level_this_hsi=facility_level)

                    self.sim.modules['HealthSystem'].schedule_hsi_event(admission, priority=0,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(days=1))



class HSI_CareOfWomenDuringPregnancy_AntenatalOutpatientManagementOfGestationalDiabetes(HSI_Event,
                                                                                        IndividualScopeEventMixin):
    """"""
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_AntenatalOutpatientFollowUp'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1   # TODO: this might not take the right persons time
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        mother = df.loc[person_id]

        # todo: discuss with asif
        from tlo.methods.pregnancy_supervisor import GestationalDiabetesGlycaemicControlEvent

        if not mother.is_alive:
            return

        print(person_id)
        print(mother.ps_gestational_age_in_weeks)

        if mother.is_pregnant and ~mother.la_currently_in_labour and mother.ps_gest_diab != 'none' \
            and mother.ac_gest_diab_on_treatment != 'none' and mother.ps_gestational_age_in_weeks > 21:
            logger.debug(key='msg', data=f'Mother {person_id} has presented for review of her GDM')

            # Nothing happens to women who arrive at follow up with well controlled GDM (treatment is effective). We now
            # assume that the treatment they are on (started in AntenatalWardInpatientCare) remains effective for the
            # length of their pregnancy
            if mother.ps_gest_diab == 'controlled' and mother.ac_gest_diab_on_treatment != 'none':
                logger.debug(key='msg', data=f'Mother {person_id} has well controlled GDM on current treatment and '
                                             f'doesnt need a further check up at present')
                # todo: give consumables for rest of pregnancy?

            # If the treatment a woman was started on has not controlled her hyperglycemia she will be started on the
            # next treatment
            elif mother.ps_gest_diab == 'uncontrolled':

                # Women for whom diet and exercise was not effective in controlling hyperglycemia are started on oral
                # meds
                if mother.ac_gest_diab_on_treatment == 'diet_exercise':

                    # Currently we assume women are given enough tablets to last the length of their pregnancy
                    # todo: separate HSI to give out meds?
                    approx_days_of_pregnancy = (40 - df.at[person_id, 'ps_gestational_age_in_weeks']) * 7

                    item_code_oral_antidiabetics = pd.unique(
                        consumables.loc[consumables['Items'] == 'Glibenclamide 5mg_1000_CMST', 'Item_Code'])[0]

                    consumables_anti_diabs = {
                        'Intervention_Package_Code': {},
                        'Item_Code': {item_code_oral_antidiabetics: approx_days_of_pregnancy * 2}} # dose 5mg BD

                    outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                        hsi_event=self,
                        cons_req_as_footprint=consumables_anti_diabs)

                    # If the meds are available women are started on that treatment
                    if outcome_of_request_for_consumables['Item_Code'][item_code_oral_antidiabetics]:
                        df.at[person_id, 'ac_gest_diab_on_treatment'] = 'orals'

                        # Assume new treatment is effective in controlling blood glucose on intiation
                        df.at[person_id, 'ps_gest_diab'] = 'controlled'

                        # Schedule GestationalDiabetesGlycaemicControlEvent which determines if this new treatment will
                        # effectively control blood glucose prior to next follow up
                        self.sim.schedule_event(GestationalDiabetesGlycaemicControlEvent(
                            self.sim.modules['PregnancySupervisor'], person_id), self.sim.date + DateOffset(days=7))

                        # Schedule follow-up
                        check_up_date = self.sim.date + DateOffset(days=28)

                        outpatient_checkup =\
                            HSI_CareOfWomenDuringPregnancy_AntenatalOutpatientManagementOfGestationalDiabetes(
                                self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id)
                        self.sim.modules['HealthSystem'].schedule_hsi_event(outpatient_checkup, priority=0,
                                                                            topen=check_up_date,
                                                                            tclose=check_up_date + DateOffset(days=3))

                # This process is repeated for mothers for whom oral medication is not effectively controlling their
                # blood sugar- they are started on insulin
                if mother.ac_gest_diab_on_treatment == 'orals':

                    item_code_oral_insulin = pd.unique(
                        consumables.loc[consumables['Items'] == 'Insulin soluble 100 IU/ml, 10ml_each_CMST',
                                        'Item_Code'])[0]

                    consumables_insulin = {
                        'Intervention_Package_Code': {},
                        'Item_Code': {item_code_oral_insulin: 5}}  # ???dose

                    outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                        hsi_event=self,
                        cons_req_as_footprint=consumables_insulin)

                    if outcome_of_request_for_consumables['Item_Code'][item_code_oral_insulin]:
                        df.at[person_id, 'ac_gest_diab_on_treatment'] = 'insulin'
                        df.at[person_id, 'ps_gest_diab'] = 'controlled'

                        self.sim.schedule_event(GestationalDiabetesGlycaemicControlEvent(
                            self.sim.modules['PregnancySupervisor'], person_id), self.sim.date + DateOffset(days=7))

                        check_up_date = self.sim.date + DateOffset(days=28)

                        outpatient_checkup = \
                            HSI_CareOfWomenDuringPregnancy_AntenatalOutpatientManagementOfGestationalDiabetes(
                                self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id)
                        self.sim.modules['HealthSystem'].schedule_hsi_event(outpatient_checkup, priority=0,
                                                                            topen=check_up_date,
                                                                            tclose=check_up_date + DateOffset(days=3))

                if mother.ac_gest_diab_on_treatment == 'insulin':
                    # As insulin is the 'highest' level management we assume women with poorly controlled blood sugars
                    # on insulin have a dose amendment and therefore no additional consumables are given
                    df.at[person_id, 'ps_gest_diab'] = 'controlled'

                    self.sim.schedule_event(GestationalDiabetesGlycaemicControlEvent(
                        self.sim.modules['PregnancySupervisor'], person_id), self.sim.date + DateOffset(days=7))

                    check_up_date = self.sim.date + DateOffset(days=28)

                    outpatient_checkup = \
                        HSI_CareOfWomenDuringPregnancy_AntenatalOutpatientManagementOfGestationalDiabetes(
                            self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(outpatient_checkup, priority=0,
                                                                        topen=check_up_date,
                                                                        tclose=check_up_date + DateOffset(days=3))


class HSI_CareOfWomenDuringPregnancy_PostAbortionCaseManagement(HSI_Event, IndividualScopeEventMixin):
    """"""
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_PostAbortionCaseManagement'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 1 # any hospital?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]
        params = self.module.parameters
        abortion_complications = self.sim.modules['PregnancySupervisor'].abortion_complications
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        # TODO: APPLY TREATMENT EFFECTS

        if not mother.is_alive:
            return

        # We check only women with complications post abortion are sent to this event
        assert abortion_complications.has_any([person_id], 'sepsis', 'haemorrhage', 'injury', first=True)

        pkg_code_infection = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Post-abortion case management',
                            'Intervention_Pkg_Code'])[0]

        all_available = self.get_all_consumables(
            pkg_codes=[pkg_code_infection])

        # if all_available:
            # evac_procedures = ['d_and_c', 'mva', 'misoprostol']
            # probability_of_evac_procedure = params[f'prob_evac_procedure_{severity}_ac']
            # random_draw = self.module.rng.choice(evac_procedures, p=probability_of_evac_procedure)

            # self.module.pac_interventions.set(person_id, random_draw)

        if all_available and abortion_complications.has_any([person_id], 'sepsis', first=True):
            self.module.pac_interventions.set(person_id, 'antibiotics')

        if all_available and abortion_complications.has_any([person_id], 'injury', first=True):
            self.module.pac_interventions.set(person_id, 'injury_repair')
            # todo: this is too simple- need for surgery more intensive care

        if abortion_complications.has_any([person_id], 'haemorrhage', first=True):
            self.module.antenatal_blood_transfusion(person_id, self, cause='abortion')


    def did_not_run(self):
        pass

    def not_available(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_PostAbortionCaseManagement: cannot not run '
                                         'with this configuration')


class HSI_CareOfWomenDuringPregnancy_TreatmentForEctopicPregnancy(HSI_Event, IndividualScopeEventMixin):
    """"""
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_TreatmentForEctopicPregnancy'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['MajorSurg'] = 1  # TODO: higher level as surgery? # TODO: add some inpatient time
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        mother = df.loc[person_id]

        # TODO: DIFFERENCE BETWEEN PRE AND POST RUPTURE - PROBS DONT HAVE TIME RIGHT NOW

        assert mother.ps_ectopic_pregnancy

        if mother.is_alive:
            logger.debug(key='message', data='This is HSI_CareOfWomenDuringPregnancy_TreatmentForEctopicPregnancy, '
                                             f'person {person_id} has been diagnosed with ectopic pregnancy after '
                                             f'presenting and will now undergo surgery')

            # We define the required consumables
            ectopic_pkg = pd.unique(consumables.loc[consumables['Intervention_Pkg'] == 'Ectopic case management',
                                                    'Intervention_Pkg_Code'])[0]

            all_available = self.get_all_consumables(
                pkg_codes=[ectopic_pkg])

            if all_available:
                # If available, the treatment can go ahead
                logger.debug(key='message',
                             data='Consumables required for ectopic surgery are available and therefore have been used')

                # Treatment variable set to true, reducing risk of death at death event in PregnancySupervisor
                df.at[person_id, 'ac_ectopic_pregnancy_treated'] = True
                df.at[person_id, 'ps_ectopic_pregnancy'] = False
            else:
                logger.debug(key='message',
                             data='Consumables required for surgery are unavailable and therefore have not '
                                  'been used')

    def did_not_run(self):
        pass

        # todo: women who cant have treatment should be at risk of rupture/death - schedule event here

    def not_available(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_TreatmentForEctopicPregnancy: cannot not run '
                                         'with this configuration')
        # todo: women who cant have treatment should be at risk of rupture - schedule event here


class AntenatalCareLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(years=self.repeat))

    def apply(self, population):
        df = self.sim.population.props

        one_year_prior = self.sim.date - np.timedelta64(1, 'Y')

        total_births_last_year = len(df.index[(df.date_of_birth > one_year_prior) & (df.date_of_birth < self.sim.date)])
        if total_births_last_year == 0:
            total_births_last_year = 1

        total_anc1_visits = self.module.anc_tracker['total_first_anc_visits']
        if total_anc1_visits == 0:
            total_anc1_visits = 1

        anc1_in_first_trimester = self.module.anc_tracker['total_anc1_first_trimester']
        cumm_gestation = self.module.anc_tracker['cumm_ga_at_anc1']
        early_anc3 = self.module.anc_tracker['timely_ANC3']
        diet_sup_6_months = self.module.anc_tracker['diet_supp_6_months']

        ra_lower_limit = 14
        ra_upper_limit = 50
        women_reproductive_age = df.index[(df.is_alive & (df.sex == 'F') & (df.age_years > ra_lower_limit) &
                                           (df.age_years < ra_upper_limit))]
        total_women_reproductive_age = len(women_reproductive_age)

        dict_for_output = {'mean_ga_first_anc': cumm_gestation/total_anc1_visits,
                           'proportion_anc1_first_trimester': (anc1_in_first_trimester/total_anc1_visits) * 100,
                           'early_anc3_proportion_of_births': (early_anc3/total_births_last_year) * 100,
                           'early_anc3': early_anc3,
                           'diet_supps_6_months': diet_sup_6_months}

        # TODO: check logic for ANC4+ calculation

        logger.info(key='anc_summary_statistics',
                    data=dict_for_output,
                    description='Yearly summary statistics output from the antenatal care module')

        self.module.anc_tracker = {'total_first_anc_visits': 0, 'cumm_ga_at_anc1': 0, 'total_anc1_first_trimester': 0,
                                  'anc8+': 0, 'timely_ANC3': 0, 'diet_supp_6_months':0}

