from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType
from tlo.methods import Metadata
from tlo.methods.dxmanager import DxTest
from tlo.methods.healthsystem import HSI_Event
# from tlo.methods.tb import HSI_TbScreening
from tlo.methods.hiv import HSI_Hiv_TestAndRefer
from tlo.methods.labour import LabourOnsetEvent
from tlo.util import BitsetHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

        # This dictionary is used to track the frequency of certain events in the module which are processed by the
        # logging event
        self.anc_tracker = dict()

    INIT_DEPENDENCIES = {'Demography', 'HealthSystem', 'PregnancySupervisor'}
    ADDITIONAL_DEPENDENCIES = {'Contraception', 'Labour', 'Lifestyle'}

    METADATA = {
        Metadata.USES_HEALTHSYSTEM,
    }

    PARAMETERS = {
        'prob_anc_continues': Parameter(
            Types.REAL, 'probability a woman will return for a subsequent ANC appointment'),
        'prob_an_ip_at_facility_level_1_2_3': Parameter(
            Types.LIST, 'probabilities that antenatal inpatient care will be scheduled at facility level 1, 2 or 3'),
        'squeeze_factor_threshold_anc': Parameter(
            Types.REAL, 'squeeze factor threshold over which an ANC appointment cannot run'),
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
        'prob_intervention_delivered_depression_screen': Parameter(
            Types.REAL, 'probability a woman will receive the intervention "depression screen" '
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
        'prob_evac_procedure_pac': Parameter(
            Types.LIST, 'Probabilities that a woman will receive D&C, MVA or misoprostal as treatment for abortion '),
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
                property='ps_htn_disorders',
                target_categories=['gest_htn', 'mild_pre_eclamp', 'severe_gest_htn', 'severe_pre_eclamp', 'eclampsia'],
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
                property='ps_gest_diab', target_categories=['uncontrolled'],
                sensitivity=params['sensitivity_blood_test_glucose'],
                specificity=params['specificity_blood_test_glucose']))

        if 'Hiv' not in self.sim.modules:
            logger.warning(key='message', data='HIV module is not registered in this simulation run and therefore HIV '
                                               'testing will not happen in antenatal care')

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

    def further_on_birth_care_of_women_in_pregnancy(self, mother_id):
        """
        This function is called by the on_birth function of NewbornOutcomes module. This function contains additional
        code related to the antenatal care module that should be ran on_birth for all births - it has been parcelled
        into functions to ensure each modules (pregnancy,antenatal care, labour, newborn, postnatal) on_birth code is
        ran in the correct sequence (as this can vary depending on how modules are registered)
        :param mother_id: mothers individual id
        """
        df = self.sim.population.props

        #  run a check at birth to make sure no women exceed 8 visits
        assert df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] < 9

        # We log the total number of ANC contacts a woman has undergone at the time of birth via this data frame

        total_anc_visit_count = {'person_id': mother_id,
                                 'age': df.at[mother_id, 'age_years'],
                                 'date_of_delivery': self.sim.date,
                                 'total_anc': df.at[mother_id, 'ac_total_anc_visits_current_pregnancy']}

        logger.info(key='anc_count_on_birth', data=total_anc_visit_count,
                    description='A dictionary containing the number of ANC visits each woman has on birth')

        # We then reset all relevant variables pertaining to care received during the antenatal period to avoid
        # treatments remaining in place for future pregnancies
        self.care_of_women_in_pregnancy_property_reset(ind_or_df='individual', id_or_index=mother_id)

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

        elif 20 <= mother.ps_gestational_age_in_weeks < 26:
            recommended_gestation_next_anc = 26

        elif 26 <= mother.ps_gestational_age_in_weeks < 30:
            recommended_gestation_next_anc = 30

        elif 30 <= mother.ps_gestational_age_in_weeks < 34:
            recommended_gestation_next_anc = 34

        elif 34 <= mother.ps_gestational_age_in_weeks < 36:
            recommended_gestation_next_anc = 36

        elif 36 <= mother.ps_gestational_age_in_weeks < 38:
            recommended_gestation_next_anc = 38

        elif 38 <= mother.ps_gestational_age_in_weeks < 40:
            recommended_gestation_next_anc = 40

        # We schedule women who present very late for ANC to return in two weeks
        elif 42 > mother.ps_gestational_age_in_weeks >= 40:
            recommended_gestation_next_anc = 42

        # Return a gestation beyond the normal length of pregnancy. This wont be used for scheduling because women
        # arent scheduled ANC past 42 weeks (see next function)
        # TODO: replace with cleaner fix for next version
        else:
            recommended_gestation_next_anc = 50

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

        # Prevent women returning to ANC at very late gestations- this needs to be reviewed and linked with induction
        if df.at[individual_id, 'ps_gestational_age_in_weeks'] >= 42:
            return

        # We check that women will only be scheduled for the next ANC contact in the schedule
        assert df.at[individual_id, 'ps_gestational_age_in_weeks'] < recommended_gestation_next_anc

        # This function houses the code that schedules the next visit, it is abstracted to prevent repetition
        def set_anc_date(visit_to_be_scheduled):

            # We store the ANC contacts as variables prior to scheduling. Facility level of the next contact is carried
            # forward from a womans first ANC contact (we assume she will always seek care within the same facility
            # level)
            if visit_to_be_scheduled == 2:
                visit = HSI_CareOfWomenDuringPregnancy_SecondAntenatalCareContact(
                    self, person_id=individual_id, facility_level_of_this_hsi=facility_level)

            elif visit_to_be_scheduled == 3:
                visit = HSI_CareOfWomenDuringPregnancy_ThirdAntenatalCareContact(
                    self, person_id=individual_id, facility_level_of_this_hsi=facility_level)

            elif visit_to_be_scheduled == 4:
                visit = HSI_CareOfWomenDuringPregnancy_FourthAntenatalCareContact(
                    self, person_id=individual_id, facility_level_of_this_hsi=facility_level)

            elif visit_to_be_scheduled == 5:
                visit = HSI_CareOfWomenDuringPregnancy_FifthAntenatalCareContact(
                    self, person_id=individual_id, facility_level_of_this_hsi=facility_level)

            elif visit_to_be_scheduled == 6:
                visit = HSI_CareOfWomenDuringPregnancy_SixthAntenatalCareContact(
                    self, person_id=individual_id, facility_level_of_this_hsi=facility_level)

            elif visit_to_be_scheduled == 7:
                visit = HSI_CareOfWomenDuringPregnancy_SeventhAntenatalCareContact(
                    self, person_id=individual_id, facility_level_of_this_hsi=facility_level)

            elif visit_to_be_scheduled == 8:
                visit = HSI_CareOfWomenDuringPregnancy_EighthAntenatalCareContact(
                    self, person_id=individual_id, facility_level_of_this_hsi=facility_level)

            # If this woman has attended less than 4 visits, and is predicted to attend > 4 (as determined via the
            # PregnancySupervisor module when ANC1 is scheduled) her subsequent ANC appointment is automatically
            # scheduled
            if visit_to_be_scheduled <= 4:
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
                    logger.debug(key='message', data=f'mother {individual_id} will seek ANC {visit_to_be_scheduled} '
                                                     f'contact on {visit_date}')

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

                        logger.debug(key='message', data=f'mother {individual_id} will seek ANC '
                                                         f'{visit_to_be_scheduled} contact on {visit_date}')

                        df.at[individual_id, 'ac_date_next_contact'] = visit_date
                    else:
                        # If additional ANC care is not sought nothing happens
                        logger.debug(key='message', data=f'mother {individual_id} will not seek any additional '
                                                         f'antenatal care for this pregnancy')

            elif visit_to_be_scheduled > 4:
                # After 4 or more visits we use the linear model equation to determine if the woman will seek care for
                # her next contact

                if self.rng.random_sample() < \
                  params['ac_linear_equations']['anc_continues'].predict(df.loc[[individual_id]])[individual_id]:
                    weeks_due_next_visit = int(recommended_gestation_next_anc - df.at[individual_id,
                                                                                      'ps_gestational_age_in_'
                                                                                      'weeks'])
                    visit_date = self.sim.date + DateOffset(weeks=weeks_due_next_visit)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(visit, priority=0,
                                                                        topen=visit_date,
                                                                        tclose=visit_date + DateOffset(days=7))
                    logger.debug(key='message', data=f'mother {individual_id} will seek ANC {visit_to_be_scheduled} '
                                                     f'contact on {visit_date}')
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

        # Use a weighted random draw to determine which level of facility the woman will be admitted too
        # facility_level = int(self.rng.choice([1, 2, 3], p=params['prob_an_ip_at_facility_level_1_2_3']))
        facility_level = self.rng.choice(['1a', '1b'], p=[0.5, 0.5])  # todo - note choice of facility_levels

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
        # consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        df = self.sim.population.props
        params = self.parameters
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        hypertension_diagnosed = False
        proteinuria_diagnosed = False

        # Define the consumables
        get_item_code = self.sim.modules['HealthSystem'].get_item_code_from_item_name
        item_code_urine_dipstick = get_item_code('Test strips, urine analysis')

        # consumables_dipstick = {
        #     'Intervention_Package_Code': {},
        #     'Item_Code': {item_code_urine_dipstick: 1}}

        # outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
        #     hsi_event=hsi_event,
        #     cons_req_as_footprint=consumables_dipstick)
        outcome_of_request_for_consumables = {
            'Intervention_Package_Code': defaultdict(lambda: True),
            'Item_Code': defaultdict(lambda: True)
        }
        # Delivery of the intervention is conditioned on the availability of the consumable and a random draw against a
        # probability that the intervention would be delivered (used to calibrate to SPA data- acts as proxy for
        # clinical quality)
        if self.rng.random_sample() < params['prob_intervention_delivered_urine_ds']:
            if outcome_of_request_for_consumables['Item_Code'][item_code_urine_dipstick]:

                # If the consumables are available the test is ran. Urine testing in ANC is predominantly used to
                # detected protein in the urine (proteinuria) which is indicative of pre-eclampsia
                if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run='urine_dipstick_protein',
                                                                           hsi_event=hsi_event):

                    # We use a temporary variable to store if proteinuria is detected
                    proteinuria_diagnosed = True
                    logger.debug(key='msg', data=f'Urine dip stick testing detected proteinuria for mother '
                                                 f'{person_id}')
            else:
                logger.debug(key='msg', data='Urine dipstick testing was not completed in this ANC visit due to '
                                             'unavailable consumables')

        # The process is repeated for blood pressure monitoring- although not conditioned on consumables
        if self.rng.random_sample() < params['prob_intervention_delivered_bp']:
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run='blood_pressure_measurement',
                                                                       hsi_event=hsi_event):
                hypertension_diagnosed = True
                logger.debug(key='msg', data=f'Blood pressure testing detected hypertension for mother {person_id}')

                if ~df.at[person_id, 'ac_gest_htn_on_treatment'] and\
                    (df.at[person_id, 'ps_htn_disorders'] != 'none') and pd.isnull(mni[person_id]['hypertension'
                                                                                                  '_onset']):

                    # We store date of onset to calculate dalys- only women who are aware of diagnosis experience DALYs
                    # (see daly weight for hypertension)
                    self.sim.modules['PregnancySupervisor'].store_dalys_in_mni(person_id, 'hypertension_onset')

        # If either high blood pressure or proteinuria are detected (or both) we assume this woman needs to be admitted
        # for further treatment following this ANC contact

        # Only women who are not on treatment OR are determined to have severe disease whilst on treatment are admitted
        if hypertension_diagnosed or proteinuria_diagnosed:
            if (df.at[person_id, 'ps_htn_disorders'] == 'severe_pre_eclamp') or \
                (df.at[person_id, 'ps_htn_disorders'] == 'eclampsia') or not df.at[person_id,
                                                                                   'ac_gest_htn_on_treatment']:

                df.at[person_id, 'ac_to_be_admitted'] = True
                logger.debug(key='msg', data=f'Mother {person_id} has had hypertension or proteinuria detected in ANC '
                                             f'and will now need admission')

        # Here we conduct screening and initiate treatment for depression as needed
        if 'Depression' in self.sim.modules:
            if self.rng.random_sample() < params['prob_intervention_delivered_depression_screen']:
                logger.debug(key='msg', data=f'Mother {person_id} will now be receive screening for depression during'
                                             f' ANC  and commence treatment as appropriate')
                if ~df.at[person_id, 'de_ever_diagnosed_depression']:
                    self.sim.modules['Depression'].do_when_suspected_depression(person_id, hsi_event)

    def interventions_initiated_at_first_contact(self, hsi_event):
        """
        This function contains all the interventions that should be delivered during the initial ANC contact (contact 1)
        including initiation of daily iron and folic acid supplementation, daily balance energy and protein
        supplementation, provision of an insecticide treated bed net and screening for TB.
        :param hsi_event: HSI event in which the function has been called
        """
        person_id = hsi_event.target
        # consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        df = self.sim.population.props
        params = self.parameters

        # We calculate an approximate number of days left of a womans pregnancy to capture the consumables required for
        # daily iron/folic acid tablets
        # if df.at[person_id, 'ps_gestational_age_in_weeks'] < 40:
        #     approx_days_of_pregnancy = (40 - df.at[person_id, 'ps_gestational_age_in_weeks']) * 7
        # else:
        #     approx_days_of_pregnancy = 14

        # Iron/folic acid & BEP supplements...
        # Define the consumables
        get_item_code = self.sim.modules['HealthSystem'].get_item_code_from_item_name
        item_code_iron_folic_acid = get_item_code('Ferrous Salt + Folic Acid, tablet, 200 + 0.25 mg')
        item_code_diet_supps = get_item_code('Dietary supplements (country-specific)')

        # consumables_anc1 = {
        #     'Intervention_Package_Code': {},
        #     'Item_Code': {item_code_iron_folic_acid: approx_days_of_pregnancy,
        #                   item_code_diet_supps: approx_days_of_pregnancy}}

        # outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
        #     hsi_event=hsi_event,
        #     cons_req_as_footprint=consumables_anc1)
        outcome_of_request_for_consumables = {
            'Intervention_Package_Code': defaultdict(lambda: True),
            'Item_Code': defaultdict(lambda: True)
        }
        # As with previous interventions - condition on consumables and probability intervention is delivered
        if outcome_of_request_for_consumables['Item_Code'][item_code_iron_folic_acid] and (self.rng.random_sample() <
                                                                                           params['prob_intervention_'
                                                                                                  'delivered_ifa']):
            df.at[person_id, 'ac_receiving_iron_folic_acid'] = True
            logger.debug(key='msg', data=f'Mother {person_id} has been commenced on IFA supplements during ANC')

        if outcome_of_request_for_consumables['Item_Code'][item_code_diet_supps] and (self.rng.random_sample() <
                                                                                      params['prob_intervention'
                                                                                             '_delivered_bep']):
            df.at[person_id, 'ac_receiving_bep_supplements'] = True
            logger.debug(key='msg', data=f'Mother {person_id} has been commenced on BEP supplements during ANC')

        # LLITN provision...
        # We define the required consumables
        # pkg_code_obstructed_llitn = pd.unique(
        #     consumables.loc[consumables['Intervention_Pkg'] == 'ITN distribution to pregnant women',
        #                     'Intervention_Pkg_Code'])[0]
        pkg_code_obstructed_llitn = 0
        # consumables_llitn = {
        #     'Intervention_Package_Code': {pkg_code_obstructed_llitn: 1},
        #     'Item_Code': {}}
        # remove use of package_codes
        # consumables_llitn = {
        #     'Intervention_Package_Code': {},
        #     'Item_Code': {}}

        # outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
        #     hsi_event=hsi_event,
        #     cons_req_as_footprint=consumables_llitn)
        outcome_of_request_for_consumables = {
            'Intervention_Package_Code': defaultdict(lambda: True),
            'Item_Code': defaultdict(lambda: True)
        }

        # If available, women are provided with a bed net at ANC1. The effect of these nets is determined
        # through the malaria module - not yet coded. n.b any interventions involving non-obstetric diseases have been
        # discussed with Tara
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_obstructed_llitn] and \
           (self.rng.random_sample() < params['prob_intervention_delivered_llitn']):
            df.at[person_id, 'ac_itn_provided'] = True
            logger.debug(key='msg', data=f'Mother {person_id} has been provided with a LLITN during ANC')

        # TODO: TB module in master is currently commented out, this is legacy code and a placeholder to ensure women
        #  are screened for TB

        # TB screening...
        # Currently we schedule women to the TB screening HSI in the TB module, however this may over-use resources so
        # possible the TB screening should also just live in this code
        # if 'tb' in self.sim.modules.keys():
        #    if self.rng.random_sample() < params['prob_intervention_delivered_tb_screen']:
        #        logger.debug(key='msg', data=f'Mother {person_id} has been referred to the TB module for screening '
        #                                     f'during ANC')

        #        tb_screen = HSI_TbScreening(
        #            module=self.sim.modules['tb'], person_id=person_id)

        #        self.sim.modules['HealthSystem'].schedule_hsi_event(tb_screen, priority=0,
        #                                                            topen=self.sim.date,
        #                                                            tclose=self.sim.date + DateOffset(days=1))

    def tetanus_vaccination(self, hsi_event):
        """
        This function contains the intervention tetanus vaccination. A booster dose of the vaccine is given to all women
         during ANC. Effect of vaccination is managed by the EPI module and therefore here we just capture consumables
         and number of doses
        :param hsi_event: HSI event in which the function has been called
        """
        person_id = hsi_event.target
        # consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        df = self.sim.population.props
        params = self.parameters

        # Define required consumables
        # pkg_code_tet = pd.unique(
        #     consumables.loc[consumables["Intervention_Pkg"] == "Tetanus toxoid (pregnant women)",
        #                     "Intervention_Pkg_Code"])[0]
        #
        # all_available = hsi_event.get_all_consumables(
        #     pkg_codes=[pkg_code_tet])
        all_available = True

        # If the consumables are available and the HCW will deliver the vaccine, the intervention is given
        if self.rng.random_sample() < params['prob_intervention_delivered_tt']:
            if all_available:
                df.at[person_id, 'ac_ttd_received'] += 1
                logger.debug(key='msg', data=f'Mother {person_id} has received a tetanus booster during ANC')

    def calcium_supplementation(self, hsi_event):
        """This function contains the intervention calcium supplementation delivered during ANC.
        :param hsi_event: HSI event in which the function has been called
        """
        df = self.sim.population.props
        params = self.parameters
        person_id = hsi_event.target
        # consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        if ~df.at[person_id, 'ac_receiving_calcium_supplements']:

            # Define consumables
            # item_code_calcium_supp = pd.unique(
            #     consumables.loc[consumables['Items'] == 'Calcium, tablet, 600 mg', 'Item_Code'])[0]
            item_code_calcium_supp = 0

            # Calculate the approximate dose for the remainder of pregnancy
            # approx_days_of_pregnancy = (40 - df.at[person_id, 'ps_gestational_age_in_weeks']) * 7
            # dose = approx_days_of_pregnancy * 3

            # consumables_calcium = {
            #     'Intervention_Package_Code': {},
            #     'Item_Code': {item_code_calcium_supp: dose}}

            # outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            #     hsi_event=hsi_event,
            #     cons_req_as_footprint=consumables_calcium)
            outcome_of_request_for_consumables = {
                'Intervention_Package_Code': defaultdict(lambda: True),
                'Item_Code': defaultdict(lambda: True)
            }

            # If the consumables are available and the HCW will provide the tablets, the intervention is given
            if outcome_of_request_for_consumables['Item_Code'][item_code_calcium_supp] and \
               (self.rng.random_sample() < params['prob_intervention_delivered_calcium']):
                df.at[person_id, 'ac_receiving_calcium_supplements'] = True
                logger.debug(key='msg', data=f'Mother {person_id} has been started on calcium supplements during ANC')

    def point_of_care_hb_testing(self, hsi_event):
        """
        This function contains the intervention point of care haemoglobin testing provided to women during ANC to detect
        anaemia during pregnancy
        :param hsi_event: HSI event in which the function has been called
        """
        person_id = hsi_event.target
        df = self.sim.population.props
        params = self.parameters
        # consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        # Define the required consumables
        # item_code_hb_test = pd.unique(
        #     consumables.loc[consumables['Items'] == 'Haemoglobin test (HB)', 'Item_Code'])[0]
        # item_code_blood_tube = pd.unique(
        #     consumables.loc[consumables['Items'] == 'Blood collecting tube, 5 ml', 'Item_Code'])[0]
        # item_code_needle = pd.unique(
        #     consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]
        # item_code_gloves = pd.unique(
        #     consumables.loc[consumables['Items'] == 'Gloves, exam, latex, disposable, pair', 'Item_Code'])[0]
        item_code_hb_test = 0
        item_code_blood_tube = 0
        item_code_needle = 0
        item_code_gloves = 0
        # consumables_hb_test = {
        #     'Intervention_Package_Code': {},
        #     'Item_Code': {item_code_hb_test: 1, item_code_blood_tube: 1, item_code_needle: 1, item_code_gloves: 1}}

        # Confirm their availability
        # outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
        #     hsi_event=hsi_event,
        #     cons_req_as_footprint=consumables_hb_test)
        outcome_of_request_for_consumables = {
            'Intervention_Package_Code': defaultdict(lambda: True),
            'Item_Code': defaultdict(lambda: True)
        }

        # If the consumables are available and the HCW will provide the test, the test is given
        if (outcome_of_request_for_consumables['Item_Code'][item_code_hb_test]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_blood_tube]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_needle]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_gloves]) and (self.rng.random_sample() <
                                                                                     params['prob_intervention_'
                                                                                            'delivered_poct']):

            # We run the test through the dx_manager and if a woman has anaemia and its detected she will be admitted
            # for further care
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run='point_of_care_hb_test',
                                                                       hsi_event=hsi_event):
                df.at[person_id, 'ac_to_be_admitted'] = True
                logger.debug(key='msg', data=f'Mother {person_id} has had anaemia detected after a POC test and will '
                                             f'require admission')

    def albendazole_administration(self, hsi_event):
        """
        This function contains the intervention albendazole administration (de-worming) and is provided to women during
         ANC
        :param hsi_event: HSI event in which the function has been called
        """
        # consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        params = self.parameters
        person_id = hsi_event.target

        # We run this function to store the associated consumables with albendazole administration. This intervention
        # has no effect in the model due to limited evidence
        # item_code_albendazole = pd.unique(
        #     consumables.loc[consumables['Items'] == 'Albendazole 200mg_1000_CMST', 'Item_Code'])[0]
        item_code_albendazole = 0

        # consumables_albendazole = {
        #     'Intervention_Package_Code': {},
        #     'Item_Code': {item_code_albendazole: 2}}

        # outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
        #     hsi_event=hsi_event,
        #     cons_req_as_footprint=consumables_albendazole)
        outcome_of_request_for_consumables = {
            'Intervention_Package_Code': defaultdict(lambda: True),
            'Item_Code': defaultdict(lambda: True)
        }

        # If the consumables are available and the HCW will provide the tablets, the intervention is given
        if outcome_of_request_for_consumables['Item_Code'][item_code_albendazole] and (self.rng.random_sample() <
                                                                                       params['prob_intervention_'
                                                                                              'delivered_bp']):
            logger.debug(key='msg', data=f' Mother {person_id} has received albendazole during ANC')

    def hep_b_testing(self, hsi_event):
        """
        This function contains the intervention Hepatitis B testing and is provided to women during ANC. As Hepatitis
        B is not modelled currently this intervention just maps consumables used during ANC
        :param hsi_event: HSI event in which the function has been called
        """
        # consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        params = self.parameters
        person_id = hsi_event.target

        # This intervention is a place holder prior to the Hepatitis B module being coded
        # Define the consumables
        # item_code_hep_test = pd.unique(
        #     consumables.loc[consumables['Items'] == 'Hepatitis B test kit-Dertemine_100 tests_CMST', 'Item_Code'])[0]
        # item_code_blood_tube = pd.unique(
        #     consumables.loc[consumables['Items'] == 'Blood collecting tube, 5 ml', 'Item_Code'])[0]
        # item_code_needle = pd.unique(
        #     consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]
        # item_code_gloves = pd.unique(
        #     consumables.loc[consumables['Items'] == 'Gloves, exam, latex, disposable, pair', 'Item_Code'])[0]
        item_code_hep_test = 0
        item_code_blood_tube = 0
        item_code_needle = 0
        item_code_gloves = 0

        # consumables_hep_b_test = {
        #     'Intervention_Package_Code': {},
        #     'Item_Code': {item_code_hep_test: 1, item_code_blood_tube: 1, item_code_needle: 1, item_code_gloves: 1}}

        # outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
        #     hsi_event=hsi_event,
        #     cons_req_as_footprint=consumables_hep_b_test)
        outcome_of_request_for_consumables = {
            'Intervention_Package_Code': defaultdict(lambda: True),
            'Item_Code': defaultdict(lambda: True)
        }

        # If the consumables are available and the HCW will provide the test, the test is delivered
        if (outcome_of_request_for_consumables['Item_Code'][item_code_hep_test]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_blood_tube]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_needle]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_gloves]) and (self.rng.random_sample() <
                                                                                     params['prob_intervention_'
                                                                                            'delivered_hepb_test']):
            logger.debug(key='msg', data=f'Mother {person_id} has received Hep B testing during ANC')

    def syphilis_testing(self, hsi_event):
        """
        This function contains the intervention Syphilis testing and is provided to women during ANC. As Syphilis is
        not modelled currently this intervention just maps consumables used during ANC
        :param hsi_event: HSI event in which the function has been called
        """
        # consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        params = self.parameters
        person_id = hsi_event.target

        # Define the consumables
        # item_code_syphilis_test = pd.unique(
        #     consumables.loc[consumables['Items'] == 'Test, Rapid plasma reagin (RPR)', 'Item_Code'])[0]
        # item_code_blood_tube = pd.unique(
        #     consumables.loc[consumables['Items'] == 'Blood collecting tube, 5 ml', 'Item_Code'])[0]
        # item_code_needle = pd.unique(
        #     consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]
        # item_code_gloves = pd.unique(
        #     consumables.loc[consumables['Items'] == 'Gloves, exam, latex, disposable, pair', 'Item_Code'])[0]
        item_code_syphilis_test = 0
        item_code_blood_tube = 0
        item_code_needle = 0
        item_code_gloves = 0

        # consumables_syphilis = {
        #     'Intervention_Package_Code': {},
        #     'Item_Code': {item_code_syphilis_test: 1, item_code_blood_tube: 1, item_code_needle: 1,
        #                   item_code_gloves: 1}}

        # outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
        #     hsi_event=hsi_event,
        #     cons_req_as_footprint=consumables_syphilis)
        outcome_of_request_for_consumables = {
            'Intervention_Package_Code': defaultdict(lambda: True),
            'Item_Code': defaultdict(lambda: True)
        }

        # If the consumables are available and the HCW will provide the test, the test is delivered
        if (outcome_of_request_for_consumables['Item_Code'][item_code_syphilis_test]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_blood_tube]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_needle]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_gloves]) and (self.rng.random_sample() <
                                                                                     params['prob_intervention_'
                                                                                            'delivered_syph_test']):
            logger.debug(key='msg', data=f'Mother {person_id} has received syphilis testing during ANC')

    def hiv_testing(self, hsi_event):
        """
        This function contains the scheduling for HIV testing and is provided to women during ANC.
        :param hsi_event: HSI event in which the function has been called
        """
        df = self.sim.population.props
        params = self.parameters
        person_id = hsi_event.target

        if 'Hiv' in self.sim.modules:
            if (self.rng.random_sample() < params['prob_intervention_delivered_hiv_test']) and ~df.at[person_id,
                                                                                                      'hv_diagnosed']:
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                   HSI_Hiv_TestAndRefer(person_id=person_id, module=self.sim.modules['Hiv']),
                   topen=self.sim.date,
                   tclose=None,
                   priority=0)

    def iptp_administration(self, hsi_event):
        """
        This function contains the intervention intermittent preventative treatment in pregnancy (for malaria) for women
        during ANC
        :param hsi_event: HSI event in which the function has been called
        """
        df = self.sim.population.props
        params = self.parameters
        person_id = hsi_event.target
        # consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

        if 'Malaria' in self.sim.modules:
            if not df.at[person_id, "ma_tx"] and df.at[person_id, "is_alive"]:

                # Test to ensure only 5 doses are able to be administered
                assert df.at[person_id, 'ac_doses_of_iptp_received'] < 6

                # Define and check the availability of consumables
                # pkg_code_iptp = pd.unique(
                #     consumables.loc[consumables["Intervention_Pkg"] == "IPT (pregnant women)",
                #                     "Intervention_Pkg_Code"])[0]

                # all_available = hsi_event.get_all_consumables(
                #     pkg_codes=[pkg_code_iptp])
                all_available = True

                if self.rng.random_sample() < params['prob_intervention_delivered_iptp']:
                    if all_available:
                        logger.debug(key='message', data=f'giving IPTp for person {person_id}')

                        # IPTP is a single dose drug given at a number of time points during pregnancy. Therefore the
                        # number of doses received during this pregnancy are stored as an integer
                        df.at[person_id, 'ac_doses_of_iptp_received'] += 1
                        logger.debug(key='message', data=f'Mother {person_id} has a dose of IPTP during pregnancy')

    def gdm_screening(self, hsi_event):
        """This function contains intervention of gestational diabetes screening during ANC. Screening is only conducted
         on women with pre-specified risk factors for the disease.
        :param hsi_event: HSI event in which the function has been called
        """
        df = self.sim.population.props
        params = self.parameters
        person_id = hsi_event.target
        # consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

        # We check if this women has any of the key risk factors, if so they are sent for additional blood tests
        if df.at[person_id, 'li_bmi'] >= 4 or df.at[person_id, 'ps_prev_gest_diab'] or df.at[person_id,
                                                                                             'ps_prev_stillbirth']:

            # We define the required consumables for testing
            # item_code_glucose_test = pd.unique(
            #     consumables.loc[consumables['Items'] == 'Blood glucose level test', 'Item_Code'])[0]
            # item_code_blood_tube = pd.unique(
            #     consumables.loc[consumables['Items'] == 'Blood collecting tube, 5 ml', 'Item_Code'])[0]
            # item_code_needle = pd.unique(
            #     consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]
            # item_code_gloves = pd.unique(
            #     consumables.loc[consumables['Items'] == 'Gloves, exam, latex, disposable, pair', 'Item_Code'])[0]
            item_code_glucose_test = 0
            item_code_blood_tube = 0
            item_code_needle = 0
            item_code_gloves = 0

            # consumables_gdm_testing = {
            #     'Intervention_Package_Code': {},
            #     'Item_Code': {item_code_glucose_test: 1, item_code_blood_tube: 1, item_code_needle: 1,
            #                   item_code_gloves: 1}}

            # Then query if these consumables are available during this HSI
            # outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            #     hsi_event=hsi_event,
            #     cons_req_as_footprint=consumables_gdm_testing)
            outcome_of_request_for_consumables = {
                'Intervention_Package_Code': defaultdict(lambda: True),
                'Item_Code': defaultdict(lambda: True)
            }

            # If they are available, the test is conducted
            if (outcome_of_request_for_consumables['Item_Code'][item_code_glucose_test]) and \
                (outcome_of_request_for_consumables['Item_Code'][item_code_blood_tube]) and \
                (outcome_of_request_for_consumables['Item_Code'][item_code_needle]) and \
                (outcome_of_request_for_consumables['Item_Code'][item_code_gloves]) and (self.rng.random_sample() <
                                                                                         params['prob_intervention_'
                                                                                                'delivered_gdm_test']):

                # If the test accurately detects a woman has gestational diabetes the consumables are recorded and she
                # is referred for treatment
                if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run='blood_test_glucose',
                                                                           hsi_event=hsi_event):

                    # We assume women with a positive GDM screen will be admitted (if they are not already receiving
                    # outpatient care)
                    if df.at[person_id, 'ac_gest_diab_on_treatment'] == 'none':

                        # Store onset after diagnosis as daly weight is tied to diagnosis
                        self.sim.modules['PregnancySupervisor'].store_dalys_in_mni(person_id, 'gest_diab_onset')

                        df.at[person_id, 'ac_to_be_admitted'] = True

                        logger.debug(key='msg', data=f'Mother {person_id} has had GDM detected after a blood glucose '
                                                     f'testing and will require admission')
                    else:
                        logger.debug(key='msg', data=f'Mother {person_id} has had GDM detected but is already receiving'
                                                     f'treatment as an outpatient so no further action has been taken')

    def anc_catch_up_interventions(self, hsi_event):
        """This function contains a collection of interventions that are delivered to women who present to ANC at later
         gestations and have missed key interventions that should have been delivered towards the beggining of the ANC
         schedule
        :param hsi_event: HSI event in which the function has been called
        """

        self.hiv_testing(hsi_event=hsi_event)
        self.hep_b_testing(hsi_event=hsi_event)
        self.syphilis_testing(hsi_event=hsi_event)
        self.point_of_care_hb_testing(hsi_event=hsi_event)
        self.tetanus_vaccination(hsi_event=hsi_event)

        self.albendazole_administration(hsi_event=hsi_event)
        self.iptp_administration(hsi_event=hsi_event)
        self.calcium_supplementation(hsi_event=hsi_event)

    def check_anc1_can_run(self, hsi_event, individual_id, squeeze_factor, gest_age_next_contact):
        """
        This function is called by the first ANC contact to determine if it can run
        :param hsi_event: HSI event in which the function has been called
        :param individual_id: individual id
        :param squeeze_factor: squeeze_factor of the HSI calling this function
        :param gest_age_next_contact: gestational age, in weeks, this woman is due to return for her next ANC
        :returns True/False as to whether the event can run
        """
        df = self.sim.population.props
        params = self.parameters

        visit = HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareContact(
            self.sim.modules['CareOfWomenDuringPregnancy'],
            person_id=individual_id, facility_level_of_this_hsi=hsi_event.ACCEPTED_FACILITY_LEVEL)

        date_difference = self.sim.date - df.at[individual_id, 'ps_date_of_anc1']

        # Only women who are alive, still pregnant and not in labour can attend ANC1
        if not df.at[individual_id, 'is_alive'] or not df.at[individual_id, 'is_pregnant'] or df.at[individual_id,
                                                                                                    'la_currently_in_'
                                                                                                    'labour']:
            return False

        # Here we block the event from running for previously scheduled ANC1 HSIs for women who have lost a pregnancy
        # and become pregnant again

        elif (date_difference > pd.to_timedelta(7, unit='D')) or \
            (df.at[individual_id, 'ac_total_anc_visits_current_pregnancy'] > 0) or (df.at[individual_id,
                                                                                    'ps_gestational_age_in_weeks'] < 7):

            logger.debug(key='msg', data=f'mother {individual_id} has arrived at ANC1 that was scheduled in a previous '
                                         f'pregnancy and therefore the event will not run')

            return False

        # If the woman is an inpatient when ANC1 is scheduled, she will try and return at the next appropriate
        # gestational age
        elif df.at[individual_id, 'hs_is_inpatient']:
            # We assume that she will return for her first appointment at the next gestation in the schedule
            logger.debug(key='msg', data=f'mother {individual_id} is scheduled to attend ANC today but is currently an '
                                         f'inpatient- she will be scheduled to arrive at her next visit instead and'
                                         f' no interventions will be delivered here')

            weeks_due_next_visit = int(gest_age_next_contact - df.at[individual_id, 'ps_gestational_age_in_weeks'])
            visit_date = self.sim.date + DateOffset(weeks=weeks_due_next_visit)

            self.sim.modules['HealthSystem'].schedule_hsi_event(visit, priority=0,
                                                                topen=visit_date,
                                                                tclose=visit_date + DateOffset(days=7))
            df.at[individual_id, 'ps_date_of_anc1'] = visit_date
            return False

        # Finally, if the squeeze factor is too high the event wont run and she will return tomorrow
        elif squeeze_factor > params['squeeze_factor_threshold_anc']:
            logger.debug(key='msg', data=f'Mother {individual_id} cannot receive ANC today as the squeeze factor is '
                                         f'too high')

            self.sim.modules['HealthSystem'].schedule_hsi_event(visit, priority=0,
                                                                topen=self.sim.date + DateOffset(days=1),
                                                                tclose=self.sim.date + DateOffset(days=2))
            return False

        else:
            return True

    def check_subsequent_anc_can_run(self, hsi_event, individual_id, this_contact, this_visit_number, squeeze_factor,
                                     gest_age_next_contact):
        """
        This function is called by all subsequent ANC contacts to determine if they can run
        :param hsi_event: HSI event in which the function has been called
        :param individual_id: individual id
        :param this_contact: HSI object of the current ANC contact that needs to be rebooked
        :param this_visit_number: Number of the next ANC contact in the schedule
        :param squeeze_factor: squeeze_factor of the HSI calling this function
        :param gest_age_next_contact: gestational age, in weeks, this woman is due to return for her next ANC
        :returns True/False as to whether the event can run
        """

        df = self.sim.population.props
        params = self.parameters

        date_difference = self.sim.date - df.at[individual_id, 'ac_date_next_contact']

        ga_for_anc_dict = {2: 20, 3: 26, 4: 30, 5: 34, 6: 36, 7: 38, 8: 40}

        # If women have died, are no longer pregnant, are in labour or more than a week has past since the HSI was
        # scheduled then it will not run
        if ~df.at[individual_id, 'is_alive'] \
            or ~df.at[individual_id, 'is_pregnant'] \
            or df.at[individual_id, 'la_currently_in_labour']\
            or (df.at[individual_id, 'ps_gestational_age_in_weeks'] < ga_for_anc_dict[this_visit_number]) \
           or (date_difference > pd.to_timedelta(7, unit='D')):
            return False

        # If the woman is currently an inpatient then she will return at the next point in the contact schedule but
        # receive the care she has missed in this visit
        elif df.at[individual_id, 'hs_is_inpatient']:
            logger.debug(key='msg', data=f'Mother {individual_id} was due to receive ANC today but she is an inpatient'
                                         f'- we will now determine if she will return for this visit in the future')
            self.antenatal_care_scheduler(individual_id, visit_to_be_scheduled=this_visit_number,
                                          recommended_gestation_next_anc=gest_age_next_contact,
                                          facility_level=hsi_event.ACCEPTED_FACILITY_LEVEL)
            return False

        # If the squeeze factor is too high she will return tomorrow
        elif squeeze_factor > params['squeeze_factor_threshold_anc']:
            logger.debug(key='msg', data=f'Mother {individual_id} cannot receive ANC today as the squeeze factor is '
                                         f'too high')

            self.sim.modules['HealthSystem'].schedule_hsi_event(this_contact, priority=0,
                                                                topen=self.sim.date + DateOffset(days=1),
                                                                tclose=self.sim.date + DateOffset(days=2))
            return False

        else:
            return True

    # =============================== INTERVENTIONS DELIVERED DURING INPATIENT CARE ===================================
    # The following functions contain code for the interventions which are called by antenatal HSIs (not including
    # routine ANC) this includes post abortion/ectopic care and antenatal inpatient care

    def full_blood_count_testing(self, hsi_event):
        """This function contains the intervention 'full blood count testing' and represents blood testing requiring a
        laboratory. It is called by HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare for women admitted due to
        anaemia
        :param hsi_event: HSI event in which the function has been called
        :returns: result of the FBC ['no_test', 'none', 'mild_mod', 'severe'] (STR)
        """
        df = self.sim.population.props
        person_id = hsi_event.target
        # consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        # Define the required consumables
        # item_code_hb_test = pd.unique(
        #     consumables.loc[consumables['Items'] == 'Haemoglobin test (HB)', 'Item_Code'])[0]
        # item_code_blood_tube = pd.unique(
        #     consumables.loc[consumables['Items'] == 'Blood collecting tube, 5 ml', 'Item_Code'])[0]
        # item_code_needle = pd.unique(
        #     consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]
        # item_code_gloves = pd.unique(
        #     consumables.loc[consumables['Items'] == 'Gloves, exam, latex, disposable, pair', 'Item_Code'])[0]

        item_code_hb_test = 0
        item_code_blood_tube = 0
        item_code_needle = 0
        item_code_gloves = 0

        # consumables_hb_test = {
        #     'Intervention_Package_Code': {},
        #     'Item_Code': {item_code_hb_test: 1, item_code_blood_tube: 1, item_code_needle: 1, item_code_gloves: 1}}

        # Confirm their availability
        # outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
        #     hsi_event=hsi_event,
        #     cons_req_as_footprint=consumables_hb_test)
        outcome_of_request_for_consumables = {
            'Intervention_Package_Code': defaultdict(lambda: True),
            'Item_Code': defaultdict(lambda: True)
        }

        # Check consumables
        if (outcome_of_request_for_consumables['Item_Code'][item_code_hb_test]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_blood_tube]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_needle]) and \
           (outcome_of_request_for_consumables['Item_Code'][item_code_gloves]):

            # Run dx_test for anaemia...
            # If a woman is not truly anaemic but the FBC returns a result of anaemia, due to tests specificity, we
            # assume the reported anaemia is mild

            test_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                dx_tests_to_run='full_blood_count_hb', hsi_event=hsi_event)

            if test_result and (df.at[person_id, 'ps_anaemia_in_pregnancy'] == 'none'):
                return 'non_severe'

            # If the test correctly identifies a woman's anaemia we assume it correctly identifies its severity
            elif test_result and (df.at[person_id, 'ps_anaemia_in_pregnancy'] != 'none'):
                return df.at[person_id, 'ps_anaemia_in_pregnancy']

            # We return a none value if no anaemia was detected
            else:
                return 'none'

        else:
            logger.debug(key='message', data=f'There were inadequate consumables to conduct an FBC so mother '
                                             f'{person_id} did not receive one')
            return 'no_test'

    def treatment_of_anaemia_causing_deficiencies(self, individual_id, hsi_event):
        """
        This function contains treatment for deficiencies that may be contributing to anaemia in pregnancy. It is called
         by HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare for women admitted due to anaemia
        :param individual_id: individual_id
        :param hsi_event: HSI event in which the function has been called
        """
        df = self.sim.population.props
        # consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        params = self.parameters
        pregnancy_deficiencies = self.sim.modules['PregnancySupervisor'].deficiencies_in_pregnancy
        store_dalys_in_mni = self.sim.modules['PregnancySupervisor'].store_dalys_in_mni

        # Calculate the approximate dose for the remainder of pregnancy
        # approx_days_of_pregnancy = (40 - df.at[individual_id, 'ps_gestational_age_in_weeks']) * 7

        # # Check and request consumables
        # item_code_elemental_iron = pd.unique(
        #     consumables.loc[
        #         consumables['Items'] ==
        #         'ferrous sulphate 200 mg, coated (65 mg iron)_1000_IDA', 'Item_Code'])[0]
        #
        # # n.b. folate doesnt have a specific consumable so this is a place holder
        # item_code_folate = pd.unique(
        #     consumables.loc[
        #         consumables['Items'] ==
        #         'Ferrous Salt + Folic Acid, tablet, 200 + 0.25 mg', 'Item_Code'])[0]
        #
        # item_code_b12 = pd.unique(
        #     consumables.loc[
        #         consumables['Items'] ==
        #         'vitamin B12 (cyanocobalamine) 1 mg/ml, 1 ml, inj._100_IDA', 'Item_Code'])[0]

        # Check and request consumables
        item_code_elemental_iron = 0

        # n.b. folate doesnt have a specific consumable so this is a place holder
        item_code_folate = 0

        item_code_b12 = 0

        # If iron or folate deficient, a woman will need to take additional daily supplements. If B12 deficient this
        # should occur monthly
        # consumables_ifa = {
        #     'Intervention_Package_Code': {},
        #     'Item_Code': {item_code_elemental_iron: approx_days_of_pregnancy,
        #                   item_code_folate: approx_days_of_pregnancy,
        #                   item_code_b12: np.ceil(approx_days_of_pregnancy / 31)}}

        # outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
        #     hsi_event=hsi_event,
        #     cons_req_as_footprint=consumables_ifa)
        outcome_of_request_for_consumables = {
            'Intervention_Package_Code': defaultdict(lambda: True),
            'Item_Code': defaultdict(lambda: True)
        }

        # Treatment is provided dependent on deficiencies present
        if pregnancy_deficiencies.has_any([individual_id], 'iron', first=True):

            # If the treatment is available we assume it has two effect, first in resolving the deficiency and second in
            # resolving a womans current anaemia
            if outcome_of_request_for_consumables['Item_Code'][item_code_elemental_iron]:
                if self.rng.random_sample() < params['effect_of_iron_replacement_for_resolving_anaemia']:

                    # If the woman is no longer anaemic after treatment we store a date of resolution for daly
                    # calculations
                    if df.at[individual_id, "ps_anaemia_in_pregnancy"] != 'none':
                        store_dalys_in_mni(individual_id, f'{df.at[individual_id, "ps_anaemia_in_pregnancy"]}_'
                                                          f'anaemia_resolution')

                        df.at[individual_id, 'ps_anaemia_in_pregnancy'] = 'none'

                if self.rng.random_sample() < params['effect_of_iron_replacement_for_resolving_iron_def']:
                    pregnancy_deficiencies.unset([individual_id], 'iron')

        if pregnancy_deficiencies.has_any([individual_id], 'folate', first=True):
            if outcome_of_request_for_consumables['Item_Code'][item_code_folate]:
                if self.rng.random_sample() < params['effect_of_folate_replacement_for_resolving_anaemia']:

                    if df.at[individual_id, "ps_anaemia_in_pregnancy"] != 'none':
                        store_dalys_in_mni(individual_id, f'{df.at[individual_id, "ps_anaemia_in_pregnancy"]}_'
                                                          f'anaemia_resolution')

                        df.at[individual_id, 'ps_anaemia_in_pregnancy'] = 'none'

                if self.rng.random_sample() < params['effect_of_folate_replacement_for_resolving_folate_def']:
                    pregnancy_deficiencies.unset([individual_id], 'folate')

        if pregnancy_deficiencies.has_any([individual_id], 'b12', first=True):
            if outcome_of_request_for_consumables['Item_Code'][item_code_b12]:
                if self.rng.random_sample() < params['effect_of_b12_replacement_for_resolving_anaemia']:

                    if df.at[individual_id, "ps_anaemia_in_pregnancy"] != 'none':
                        store_dalys_in_mni(individual_id, f'{df.at[individual_id, "ps_anaemia_in_pregnancy"]}_'
                                                          f'anaemia_resolution')

                        df.at[individual_id, 'ps_anaemia_in_pregnancy'] = 'none'

                if self.rng.random_sample() < params['effect_of_b12_replacement_for_resolving_b12_def']:
                    pregnancy_deficiencies.unset([individual_id], 'b12')

    def start_iron_and_folic_acid(self, individual_id, hsi_event):
        """
        This function contains initiation of iron and folic acid supplementation. It is called
        by HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare for women admitted due to anaemia who have not been
         given iron and folic acid supplements during ANC
        :param individual_id: individual_id
        :param hsi_event: HSI event in which the function has been called
        """
        df = self.sim.population.props
        params = self.parameters
        # consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        store_dalys_in_mni = self.sim.modules['PregnancySupervisor'].store_dalys_in_mni

        # Calculate the approximate dose for the remainder of pregnancy
        # approx_days_of_pregnancy = (40 - df.at[individual_id, 'ps_gestational_age_in_weeks']) * 7

        # Check availability of consumables
        # item_code_iron_folic_acid = pd.unique(
        #     consumables.loc[
        #         consumables['Items'] ==
        #         'Ferrous Salt + Folic Acid, tablet, 200 + 0.25 mg', 'Item_Code'])[0]
        item_code_iron_folic_acid = 0

        # consumables_ifa = {
        #     'Intervention_Package_Code': {},
        #     'Item_Code': {item_code_iron_folic_acid: approx_days_of_pregnancy}}

        # outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
        #     hsi_event=hsi_event,
        #     cons_req_as_footprint=consumables_ifa)
        outcome_of_request_for_consumables = {
            'Intervention_Package_Code': defaultdict(lambda: True),
            'Item_Code': defaultdict(lambda: True)
        }

        # Start iron and folic acid treatment
        if outcome_of_request_for_consumables['Item_Code'][item_code_iron_folic_acid]:
            df.at[individual_id, 'ac_receiving_iron_folic_acid'] = True
            logger.debug(key='msg', data=f'Mother {individual_id} has been started on IFA supplementation after being '
                                         f'admitted for anaemia')

            # Women started on IFA at this stage are already anaemic, we here apply a probability that
            # starting on a course of IFA will correct anaemia prior to follow up
            if self.rng.random_sample() < params['effect_of_ifa_for_resolving_anaemia']:

                # Store date of resolution for daly calculations
                store_dalys_in_mni(individual_id, f'{df.at[individual_id, "ps_anaemia_in_pregnancy"]}_'
                                                  f'anaemia_resolution')

                df.at[individual_id, 'ps_anaemia_in_pregnancy'] = 'none'

    def antenatal_blood_transfusion(self, individual_id, hsi_event, cause):
        """
        This function contains the intervention 'blood transfusion'. It is called by either
        HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare or HSI_CareOfWomenDuringPregnancy_PostAbortionCase
        Management for women requiring blood for either haemorrhage or severe anaemia.
        given iron and folic acid supplements during ANC
        :param individual_id: individual_id
        :param hsi_event: HSI event in which the function has been called
        :param cause: (STR) cause of woman needed transfusion ['antepartum_haem', 'abortion', 'severe_anaemia']
        """
        df = self.sim.population.props
        # consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        params = self.parameters
        store_dalys_in_mni = self.sim.modules['PregnancySupervisor'].store_dalys_in_mni

        # Check for consumables
        # item_code_blood = pd.unique(consumables.loc[consumables['Items'] == 'Blood, one unit', 'Item_Code'])[0]
        # item_code_needle = pd.unique(consumables.loc[consumables['Items'] == 'Lancet, blood, disposable',
        #                                              'Item_Code'])[0]
        # item_code_test = pd.unique(consumables.loc[consumables['Items'] == 'Test, hemoglobin', 'Item_Code'])[0]
        # item_code_giving_set = pd.unique(consumables.loc[consumables['Items'] ==
        # 'IV giving/infusion set, with needle','Item_Code'])[0]
        item_code_blood = 0
        item_code_needle = 0
        item_code_test = 0
        item_code_giving_set = 0

        # consumables_needed_bt = {'Intervention_Package_Code': {},
        #                          'Item_Code': {item_code_blood: 2, item_code_needle: 1, item_code_test: 1,
        #                                        item_code_giving_set: 2}}

        # outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
        #     hsi_event=hsi_event, cons_req_as_footprint=consumables_needed_bt)
        outcome_of_request_for_consumables = {
            'Intervention_Package_Code': defaultdict(lambda: True),
            'Item_Code': defaultdict(lambda: True)
        }

        # If the consumables are available the intervention is delivered
        if (outcome_of_request_for_consumables['Item_Code'][item_code_blood]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_test]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_needle]) and \
           (outcome_of_request_for_consumables['Item_Code'][item_code_giving_set]):

            logger.debug(key='msg', data=f'Mother {individual_id} is receiving an antenatal blood transfusion due '
                                         f'to {cause}')

            if cause == 'severe_anaemia':
                # If the woman is receiving blood due to anaemia we apply a probability that a transfusion of 2 units
                # RBCs will correct this woman's severe anaemia
                if params['treatment_effect_blood_transfusion_anaemia'] > self.rng.random_sample():
                    store_dalys_in_mni(individual_id, 'severe_anaemia_resolution')
                    df.at[individual_id, 'ps_anaemia_in_pregnancy'] = 'none'

            # If the woman has experience haemorrhage post abortion we store that the intervention has been received in
            # this property which reduces her risk of death
            elif cause == 'abortion':
                self.pac_interventions.set(individual_id, 'blood_products')

            # If the cause is antepartum haemorrhage we use this property to reduce a womans risk of death following
            # treatment
            else:
                df.at[individual_id, 'ac_received_blood_transfusion'] = True

    def initiate_maintenance_anti_hypertensive_treatment(self, individual_id, hsi_event):
        """
        This function contains initiation of oral antihypertensive medication for women with high blood pressure. It is
        called by HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare for women who have been identified as having
         high blood pressure in pregnancy but are not yet receiving treatment
        :param individual_id: individual_id
        :param hsi_event: HSI event in which the function has been called
        """
        df = self.sim.population.props
        # consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

        # Calculate the approximate dose for the remainder of pregnancy
        # approx_days_of_pregnancy = (40 - df.at[individual_id, 'ps_gestational_age_in_weeks']) * 7

        # Define the consumables and check their availability
        # item_code_methyldopa = pd.unique(
        #     consumables.loc[consumables['Items'] == 'Methyldopa 250mg_1000_CMST', 'Item_Code'])[0]
        item_code_methyldopa = 0

        # consumables_gest_htn_treatment = {
        #     'Intervention_Package_Code': {},
        #     'Item_Code': {item_code_methyldopa: 4 * approx_days_of_pregnancy}}

        # outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
        #     hsi_event=hsi_event,
        #     cons_req_as_footprint=consumables_gest_htn_treatment)
        outcome_of_request_for_consumables = {
            'Intervention_Package_Code': defaultdict(lambda: True),
            'Item_Code': defaultdict(lambda: True)
        }

        # If they are available then the woman is started on treatment
        if outcome_of_request_for_consumables['Item_Code'][item_code_methyldopa]:
            df.at[individual_id, 'ac_gest_htn_on_treatment'] = True
            logger.debug(key='msg', data=f'Mother {individual_id} has been started on regular antihypertensives due to '
                                         f'her HDP')

    def initiate_treatment_for_severe_hypertension(self, individual_id, hsi_event):
        """
        This function contains initiation of intravenous antihypertensive medication for women with severely high blood
        pressure. It is called by HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare for women who have been
        admitted due to severely high blood pressure (severe gestational hypertension, severe pre-eclampsia or
        eclampsia)
        :param individual_id: individual_id
        :param hsi_event: HSI event in which the function has been called
        """
        df = self.sim.population.props
        # consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

        # Define the consumables and check their availability
        # item_code_hydralazine = pd.unique(
        #     consumables.loc[consumables['Items'] == 'Hydralazine, powder for injection, 20 mg ampoule',
        #                     'Item_Code'])[0]
        # item_code_wfi = pd.unique(
        #     consumables.loc[consumables['Items'] == 'Water for injection, 10ml_Each_CMST', 'Item_Code'])[0]
        # item_code_needle = pd.unique(
        #     consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]
        # item_code_gloves = pd.unique(
        #     consumables.loc[consumables['Items'] == 'Gloves, exam, latex, disposable, pair', 'Item_Code'])[0]
        item_code_hydralazine = 0
        item_code_wfi = 0
        item_code_needle = 0
        item_code_gloves = 0

        # consumables_gest_htn_treatment = {
        #     'Intervention_Package_Code': {},
        #     'Item_Code': {item_code_hydralazine: 1,
        #                   item_code_wfi: 1,
        #                   item_code_gloves: 1,
        #                   item_code_needle: 1}}

        # Then query if these consumables are available during this HSI
        # outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
        #     hsi_event=hsi_event,
        #     cons_req_as_footprint=consumables_gest_htn_treatment)
        outcome_of_request_for_consumables = {
            'Intervention_Package_Code': defaultdict(lambda: True),
            'Item_Code': defaultdict(lambda: True)
        }

        # If they are available then the woman is started on treatment
        if (outcome_of_request_for_consumables['Item_Code'][item_code_hydralazine]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_wfi]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_needle]) and \
           (outcome_of_request_for_consumables['Item_Code'][item_code_gloves]):

            # We assume women treated with antihypertensives would no longer be severely hypertensive- meaning they
            # are not at risk of death from severe gestational hypertension in the PregnancySupervisor event
            if df.at[individual_id, 'ps_htn_disorders'] == 'severe_gest_htn':
                df.at[individual_id, 'ps_htn_disorders'] = 'gest_htn'
                logger.debug(key='msg', data=f'Mother {individual_id} has been given intravenous anti-hypertensive as '
                                             f'part of treatment regime for severe gestational hypertension')

            # We dont assume antihypertensives convert severe pre-eclampsia/eclampsia to a more mild version of the
            # disease (as the disease is multi-system and hypertension is only one contributing factor to mortality) but
            # instead use this property to reduce risk of acute death from this episode of disease
            if (df.at[individual_id, 'ps_htn_disorders'] == 'severe_pre_eclamp') or (df.at[individual_id,
                                                                                           'ps_htn_disorders'] ==
                                                                                     'eclampsia'):

                logger.debug(key='msg', data=f'Mother {individual_id} has been given intravenous anti-hypertensive as '
                                             f'part of treatment regime for severe pre-eclampsia/eclampsia')
                df.at[individual_id, 'ac_iv_anti_htn_treatment'] = True

    def treatment_for_severe_pre_eclampsia_or_eclampsia(self, individual_id, hsi_event):
        """
        This function contains initiation of intravenous magnesium sulphate medication for women with severely
        pre-eclampsia/eclampsia It is called by HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare for women who
        have been admitted with those conditions
        :param individual_id: individual_id
        :param hsi_event: HSI event in which the function has been called
        """
        df = self.sim.population.props

        # consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]
        # Define the consumable package code
        # # pkg_code_eclampsia_and_spe = pd.unique(
        # #     consumables.loc[consumables['Intervention_Pkg'] == 'Management of eclampsia',
        # #                     'Intervention_Pkg_Code'])[0]
        #
        # all_available = hsi_event.get_all_consumables(
        #     pkg_codes=[pkg_code_eclampsia_and_spe])
        # remove use of package_codes
        all_available = True

        # If available deliver the treatment
        if all_available:
            df.at[individual_id, 'ac_mag_sulph_treatment'] = True
            logger.debug(key='msg', data=f'Mother {individual_id} has received magnesium sulphate during her admission '
                                         f'for severe pre-eclampsia/eclampsia')

    def antibiotics_for_prom(self, individual_id, hsi_event):
        """
        This function contains initiation of antibiotics for women with who have been admitted following premature
        rupture of membranes .It is called by HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare
        :param individual_id: individual_id
        :param hsi_event: HSI event in which the function has been called
        """
        df = self.sim.population.props
        # consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

        # Define consumables and check their availability
        # item_code_benpen = pd.unique(
        #     consumables.loc[consumables['Items'] ==
        #     'Benzathine benzylpenicillin, powder for injection, 2.4 million IU', 'Item_Code'])[0]
        # item_code_wfi = pd.unique(
        #     consumables.loc[consumables['Items'] == 'Water for injection, 10ml_Each_CMST', 'Item_Code'])[0]
        # item_code_needle = pd.unique(
        #     consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]
        # item_code_gloves = pd.unique(
        #     consumables.loc[consumables['Items'] == 'Gloves, exam, latex, disposable, pair', 'Item_Code'])[0]
        item_code_benpen = 0
        item_code_wfi = 0
        item_code_needle = 0
        item_code_gloves = 0
        # consumables_abx_for_prom = {
        #     'Intervention_Package_Code': {},
        #     'Item_Code': {item_code_benpen: 4, item_code_wfi: 1, item_code_needle: 1,
        #                   item_code_gloves: 1}}

        # Then query if these consumables are available during this HSI
        # outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
        #     hsi_event=hsi_event,
        #     cons_req_as_footprint=consumables_abx_for_prom)
        outcome_of_request_for_consumables = {
            'Intervention_Package_Code': defaultdict(lambda: True),
            'Item_Code': defaultdict(lambda: True)
        }

        if (outcome_of_request_for_consumables['Item_Code'][item_code_benpen]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_wfi]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_needle]) and \
           (outcome_of_request_for_consumables['Item_Code'][item_code_gloves]):
            df.at[individual_id, 'ac_received_abx_for_prom'] = True
            logger.debug(key='msg', data=f'Mother {individual_id} has received antibiotics following admission due to '
                                         f'PROM')

    def antibiotics_for_chorioamnionitis(self, individual_id, hsi_event):
        """
        This function contains initiation of antibiotics for women with who have been admitted due to chorioamnionitis
        infection .It is called by HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare
        :param individual_id: individual_id
        :param hsi_event: HSI event in which the function has been called
        """
        df = self.sim.population.props
        # consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

        # Define the consumables and check their availability
        # item_code_benpen = pd.unique(
        #     consumables.loc[consumables['Items'] ==
        #     'Benzathine benzylpenicillin, powder for injection, 2.4 million IU', 'Item_Code'])[0]
        # item_code_genta = pd.unique(
        #     consumables.loc[consumables['Items'] == 'Gentamycin, injection, 40 mg/ml in 2 ml vial',
        #                     'Item_Code'])[0]
        # item_code_wfi = pd.unique(
        #     consumables.loc[consumables['Items'] == 'Water for injection, 10ml_Each_CMST', 'Item_Code'])[0]
        # item_code_needle = pd.unique(
        #     consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]
        # item_code_gloves = pd.unique(
        #     consumables.loc[consumables['Items'] == 'Gloves, exam, latex, disposable, pair', 'Item_Code'])[0]
        item_code_benpen = 0
        item_code_genta = 0
        item_code_wfi = 0
        item_code_needle = 0
        item_code_gloves = 0
        # consumables_abx_for_chorio = {
        #     'Intervention_Package_Code': {},
        #     'Item_Code': {item_code_benpen: 4, item_code_genta: 1, item_code_wfi: 1, item_code_needle: 1,
        #                   item_code_gloves: 1}}

        # Then query if these consumables are available during this HSI
        # outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
        #     hsi_event=hsi_event,
        #     cons_req_as_footprint=consumables_abx_for_chorio)
        outcome_of_request_for_consumables = {
            'Intervention_Package_Code': defaultdict(lambda: True),
            'Item_Code': defaultdict(lambda: True)
        }

        if (outcome_of_request_for_consumables['Item_Code'][item_code_benpen]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_genta]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_wfi]) and \
            (outcome_of_request_for_consumables['Item_Code'][item_code_needle]) and \
           (outcome_of_request_for_consumables['Item_Code'][item_code_gloves]):
            df.at[individual_id, 'ac_received_abx_for_chorioamnionitis'] = True
            logger.debug(key='msg', data=f'Mother {individual_id} has received antibiotics after being admitted for '
                                         f'chorioamnionitis')

    def ectopic_pregnancy_treatment_doesnt_run(self, individual_id):
        """
        This function is called within HSI_CareOfWomenDuringPregnancy_TreatmentForEctopicPregnancy if the event cannot
        run/the intervention cannot be delivered. This ensures that women with ectopic pregnancies that haven't ruptured
        will experience rupture and risk of death without treatment
        :param individual_id: individual_id
        :return:
        """
        df = self.sim.population.props

        from tlo.methods.pregnancy_supervisor import EctopicPregnancyRuptureEvent

        # If this event cannot run we ensure all women will eventually experience rupture due to untreated ectopic
        if df.at[individual_id, 'ps_ectopic_pregnancy'] == 'not_ruptured':
            self.sim.schedule_event(EctopicPregnancyRuptureEvent(
                self.sim.modules['PregnancySupervisor'], individual_id), self.sim.date + DateOffset(days=7))

    def calculate_beddays(self, individual_id):
        """
        This function is called by HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare to calculate the number of
        beddays required by a women following admission. This is determined according to the reason for her admission
        and her gestation
        :param individual_id: individual_id
        :return:
        """
        df = self.sim.population.props
        mother = df.loc[individual_id]

        # Women with severe pre-eclampsia/eclampsia, severe haemorrhage, moderate haemorrhage at later gestation,
        # premature rupture of membranes complicated by chorioamnionitis, or at later gestation can be delivered
        # immediately and will only require a day in the antenatal ward for treatment before being admitted for
        # delivery
        if (mother.ps_htn_disorders == 'severe_pre_eclamp') or \
            (mother.ps_htn_disorders == 'eclampsia') or \
            mother.ps_placental_abruption or \
            (mother.ps_placenta_praevia and (mother.ps_antepartum_haemorrhage == 'severe')) or \
            (mother.ps_placenta_praevia and (mother.ps_antepartum_haemorrhage == 'mild_moderate') and
             (mother.ps_gestational_age_in_weeks >= 37)) or\
            (mother.ps_premature_rupture_of_membranes and (mother.ps_chorioamnionitis != 'none')) or \
            (mother.ps_premature_rupture_of_membranes and (mother.ps_chorioamnionitis == 'none') and
             (mother.ps_gestational_age_in_weeks >= 34)):
            beddays = 1

        # Otherwise women will remain as an inpatient until their gestation is greater, to improve newborn outcomes
        elif (mother.ps_placenta_praevia and (mother.ps_antepartum_haemorrhage == 'mild_moderate') and
              (mother.ps_gestational_age_in_weeks < 37)) or (mother.ps_premature_rupture_of_membranes and
                                                             (mother.ps_chorioamnionitis == 'none') and
                                                             (mother.ps_gestational_age_in_weeks < 34)):

            beddays = int((37 * 7) - (mother.ps_gestational_age_in_weeks * 7))

        else:
            beddays = 1

        return beddays


class HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareContact(HSI_Event, IndividualScopeEventMixin):
    """ This is the  HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareContact which represents the first routine
    antenatal care contact (ANC1). It is scheduled by the PregnancySupervisor Event for women who choose to seek
    routine antenatal care during their pregnancy. It is recommended that this visit occur before 12 weeks gestation.
    This event delivers the interventions to women which are part of ANC1. Additionally interventions that should be
    offered in the early ANC contacts are provided to women who present to ANC1 later in their pregnancy. Scheduling
    the next ANC contact in the occurs during this HSI along with admission to antenatal inpatient ward in the
    case of complications """

    def __init__(self, module, person_id, facility_level_of_this_hsi):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_FirstAntenatalCareContact'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'AntenatalFirst': 1})
        self.ACCEPTED_FACILITY_LEVEL = facility_level_of_this_hsi
        assert self.ACCEPTED_FACILITY_LEVEL not in {'0'}  # TODO: also not None, but that causes error in doc build
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]

        # Calculate when this woman should return for her next visit
        gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)

        # Check this visit can run
        can_anc1_run = self.module.check_anc1_can_run(self, person_id, squeeze_factor, gest_age_next_contact)

        if can_anc1_run:
            # Check only women who have not attended any ANC contacts during this pregnancy arrive at this
            # event
            assert mother.ac_total_anc_visits_current_pregnancy == 0

            # We generate the facility type that this HSI is occurring at (dependent on facility level) - we currently
            # assume women will present to the same facility level and type for any future ANC visits

            if self.ACCEPTED_FACILITY_LEVEL in ('1a', '1b'):
                # Assume a 50/50 chance of health centre or hospital in level 1, however this will need editing
                facility_type = self.module.rng.choice(['health_centre', 'hospital'], p=[0.5, 0.5])
                df.at[person_id, 'ac_facility_type'] = facility_type
                logger.info(key='anc_facility_type', data=f'{facility_type}')

            elif self.ACCEPTED_FACILITY_LEVEL in ('2', '3', '4'):
                logger.info(key='anc_facility_type', data='hospital')
                df.at[person_id, 'ac_facility_type'] = 'hospital'

            logger.debug(key='message', data=f'mother {person_id} presented for ANC1 at a '
                                             f'{df.at[person_id, "ac_facility_type"]} at gestation '
                                             f'{df.at[person_id, "ps_gestational_age_in_weeks"]} ')

            # We add a visit to a rolling total of ANC visits in this pregnancy
            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            # And ensure only women whose first contact with ANC services are attending this event
            assert mother.ps_gestational_age_in_weeks is not None
            assert mother.ps_gestational_age_in_weeks is not pd.NaT
            assert mother.ps_gestational_age_in_weeks >= 7

            #  =================================== INTERVENTIONS ====================================================
            # First all women, regardless of ANC contact or gestation, undergo urine and blood pressure measurement
            # and depression screening
            self.module.screening_interventions_delivered_at_every_contact(hsi_event=self)

            # Then, the appropriate interventions are delivered according gestational age. It is assumed that women who
            # present late to ANC1 are 'caught up' with the interventions they missed from previous visits
            # (as per malawi guidelines)

            # If this woman is presenting prior to the suggested gestation for ANC2 (visit at 20 weeks), she receives
            # only the interventions for ANC1
            if mother.ps_gestational_age_in_weeks < 20:
                # These are the interventions delivered at ANC1
                self.module.interventions_initiated_at_first_contact(hsi_event=self)
                self.module.hiv_testing(hsi_event=self)
                self.module.hep_b_testing(hsi_event=self)
                self.module.syphilis_testing(hsi_event=self)
                self.module.point_of_care_hb_testing(hsi_event=self)
                self.module.tetanus_vaccination(hsi_event=self)

            # If she presents after 20 weeks she is provided interventions delivered at the first catch up and then
            # the appropriate catch up interventions
            elif mother.ps_gestational_age_in_weeks > 20:
                self.module.interventions_initiated_at_first_contact(hsi_event=self)
                self.module.anc_catch_up_interventions(hsi_event=self)

            # Any women presenting for ANC1 after 26 week are also required to have a GDM screen
            if mother.ps_gestational_age_in_weeks >= 26:
                self.module.gdm_screening(hsi_event=self)

            # Then we determine if this woman will return for her next ANC visit
            if mother.ps_gestational_age_in_weeks < 40:
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=2,
                                                     recommended_gestation_next_anc=gest_age_next_contact,
                                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)

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
    """This is the  HSI_CareOfWomenDuringPregnancy_SecondAntenatalCareContact which represents the second routine
    antenatal care contact (ANC2). It is scheduled by the HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareContact for
    women who choose to seek additional ANC after their previous visit. It is recommended that this visit occur at 20
    weeks gestation. This event delivers the interventions to women which are part of ANC2. Additionally interventions
    that should be delivered according to a womans gestational age and position in her ANC schedule are delivered.
    Finally scheduling the next ANC contact in the occurs during this HSI along with admission to antenatal inpatient
    ward in the case of complications"""

    def __init__(self, module, person_id, facility_level_of_this_hsi):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_SecondAntenatalCareContact'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
        self.ACCEPTED_FACILITY_LEVEL = facility_level_of_this_hsi
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]

        # Here we define variables used within the function that checks in this ANC visit can run
        gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)
        this_contact = HSI_CareOfWomenDuringPregnancy_SecondAntenatalCareContact(
            self.module, person_id=person_id, facility_level_of_this_hsi=self.ACCEPTED_FACILITY_LEVEL)

        # Run the check
        can_anc_run = self.module.check_subsequent_anc_can_run(self, individual_id=person_id, this_contact=this_contact,
                                                               this_visit_number=2, squeeze_factor=squeeze_factor,
                                                               gest_age_next_contact=gest_age_next_contact)

        if can_anc_run:

            logger.info(key='anc_facility_type', data=f'{df.at[person_id, "ac_facility_type"]}')
            logger.debug(key='msg', data=f'mother {person_id}presented for ANC 2 at a '
                                         f'{df.at[person_id, "ac_facility_type"]} at gestation '
                                         f'{df.at[person_id, "ps_gestational_age_in_weeks"]} ')

            assert mother.ac_total_anc_visits_current_pregnancy == 1
            assert mother.ps_gestational_age_in_weeks is not None
            assert mother.ps_gestational_age_in_weeks is not pd.NaT
            assert mother.ps_gestational_age_in_weeks >= 19

            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            #  =================================== INTERVENTIONS ====================================================
            # First we administer the administer the interventions all women will receive at this contact regardless of
            # gestational age
            self.module.screening_interventions_delivered_at_every_contact(hsi_event=self)
            self.module.tetanus_vaccination(hsi_event=self)
            self.module.calcium_supplementation(hsi_event=self)

            # And we schedule the next ANC appointment
            if mother.ps_gestational_age_in_weeks < 40:
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=3,
                                                     recommended_gestation_next_anc=gest_age_next_contact,
                                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)

            # Then we administer interventions that are due to be delivered at this womans gestational age, which may be
            # in addition to intervention delivered in ANC2
            if mother.ps_gestational_age_in_weeks < 26:
                self.module.albendazole_administration(hsi_event=self)
                self.module.iptp_administration(hsi_event=self)

            elif mother.ps_gestational_age_in_weeks < 30:
                self.module.iptp_administration(hsi_event=self)
                self.module.gdm_screening(hsi_event=self)

            elif mother.ps_gestational_age_in_weeks < 34:
                self.module.iptp_administration(hsi_event=self)

            elif mother.ps_gestational_age_in_weeks < 36:
                self.module.iptp_administration(hsi_event=self)
                self.module.hep_b_testing(hsi_event=self)
                self.module.syphilis_testing(hsi_event=self)

            elif mother.ps_gestational_age_in_weeks < 38:
                self.module.point_of_care_hb_testing(hsi_event=self)

            elif mother.ps_gestational_age_in_weeks < 40:
                self.module.iptp_administration(hsi_event=self)

            elif mother.ps_gestational_age_in_weeks >= 40:
                pass

            if df.at[person_id, 'ac_to_be_admitted']:
                self.module.schedule_admission(person_id)

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        return actual_appt_footprint

    def did_not_run(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_SecondAntenatalCareVisit: did not run')

    def not_available(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_SecondAntenatalCareVisit: cannot not run with '
                                         'this configuration')


class HSI_CareOfWomenDuringPregnancy_ThirdAntenatalCareContact(HSI_Event, IndividualScopeEventMixin):
    """This is the  HSI_CareOfWomenDuringPregnancy_ThirdAntenatalCareContact which represents the third routine
    antenatal care contact (ANC3). It is scheduled by the HSI_CareOfWomenDuringPregnancy_SecondAntenatalCareContact for
    women who choose to seek additional ANC after their previous visit. It is recommended that this visit occur at 26
    weeks gestation. This event delivers the interventions to women which are part of ANC3. Additionally interventions
    that should be delivered according to a womans gestational age and position in her ANC schedule are delivered.
    Finally scheduling the next ANC contact in the occurs during this HSI along with admission to antenatal inpatient
    ward in the case of complications"""

    def __init__(self, module, person_id, facility_level_of_this_hsi):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_ThirdAntenatalCareContact'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
        self.ACCEPTED_FACILITY_LEVEL = facility_level_of_this_hsi
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]

        # Here we define variables used within the function that checks in this ANC visit can run
        gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)
        this_contact = HSI_CareOfWomenDuringPregnancy_ThirdAntenatalCareContact(
            self.module, person_id=person_id, facility_level_of_this_hsi=self.ACCEPTED_FACILITY_LEVEL)

        # Run the check
        can_anc_run = self.module.check_subsequent_anc_can_run(self, person_id, this_contact, 3, squeeze_factor,
                                                               gest_age_next_contact)

        if can_anc_run:

            logger.info(key='anc_facility_type', data=f'{df.at[person_id, "ac_facility_type"]}')
            logger.debug(key='message', data=f'mother {person_id}presented for ANC 3 at a '
                                             f'{df.at[person_id, "ac_facility_type"]} at gestation '
                                             f'{df.at[person_id, "ps_gestational_age_in_weeks"]} ')

            assert mother.ac_total_anc_visits_current_pregnancy == 2
            assert mother.ps_gestational_age_in_weeks is not None
            assert mother.ps_gestational_age_in_weeks is not pd.NaT
            assert mother.ps_gestational_age_in_weeks >= 25

            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            #  =================================== INTERVENTIONS ====================================================
            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)
            self.module.screening_interventions_delivered_at_every_contact(hsi_event=self)
            self.module.calcium_supplementation(hsi_event=self)

            if mother.ps_gestational_age_in_weeks < 40:
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=4,
                                                     recommended_gestation_next_anc=gest_age_next_contact,
                                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)

            if mother.ps_gestational_age_in_weeks < 30:
                self.module.iptp_administration(hsi_event=self)
                self.module.gdm_screening(hsi_event=self)

            elif mother.ps_gestational_age_in_weeks < 34:
                self.module.iptp_administration(hsi_event=self)

            elif mother.ps_gestational_age_in_weeks < 36:
                self.module.iptp_administration(hsi_event=self)
                self.module.hep_b_testing(hsi_event=self)
                self.module.syphilis_testing(hsi_event=self)

            elif mother.ps_gestational_age_in_weeks < 38:
                self.module.point_of_care_hb_testing(hsi_event=self)

            elif mother.ps_gestational_age_in_weeks < 40:
                self.module.iptp_administration(hsi_event=self)

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
    """This is the  HSI_CareOfWomenDuringPregnancy_FourthAntenatalCareContact which represents the fourth routine
    antenatal care contact (ANC4). It is scheduled by the HSI_CareOfWomenDuringPregnancy_ThirdAntenatalCareContact for
    women who choose to seek additional ANC after their previous visit. It is recommended that this visit occur at 30
    weeks gestation. This event delivers the interventions to women which are part of ANC4. Additionally interventions
    that should be delivered according to a womans gestational age and position in her ANC schedule are delivered.
    Finally scheduling the next ANC contact in the occurs during this HSI along with admission to antenatal inpatient
    ward in the case of complications"""

    def __init__(self, module, person_id, facility_level_of_this_hsi):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_FourthAntenatalCareContact'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = facility_level_of_this_hsi
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]

        # Here we define variables used within the function that checks in this ANC visit can run
        gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)
        this_contact = HSI_CareOfWomenDuringPregnancy_FourthAntenatalCareContact(
            self.module, person_id=person_id, facility_level_of_this_hsi=self.ACCEPTED_FACILITY_LEVEL)

        # Run the check
        can_anc_run = self.module.check_subsequent_anc_can_run(self, person_id, this_contact, 4, squeeze_factor,
                                                               gest_age_next_contact)

        if can_anc_run:

            logger.info(key='anc_facility_type', data=f'{df.at[person_id, "ac_facility_type"]}')
            logger.debug(key='message', data=f'mother {person_id}presented for ANC 4 at a '
                                             f'{df.at[person_id, "ac_facility_type"]} at gestation '
                                             f'{df.at[person_id, "ps_gestational_age_in_weeks"]} ')

            assert mother.ac_total_anc_visits_current_pregnancy == 3
            assert mother.ps_gestational_age_in_weeks is not None
            assert mother.ps_gestational_age_in_weeks is not pd.NaT
            assert mother.ps_gestational_age_in_weeks >= 29

            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            #  =================================== INTERVENTIONS ====================================================
            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)
            self.module.screening_interventions_delivered_at_every_contact(hsi_event=self)
            self.module.calcium_supplementation(hsi_event=self)

            if mother.ps_gestational_age_in_weeks < 40:
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=5,
                                                     recommended_gestation_next_anc=gest_age_next_contact,
                                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)

            if mother.ps_gestational_age_in_weeks < 34:
                self.module.iptp_administration(hsi_event=self)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] < 36:
                self.module.iptp_administration(hsi_event=self)
                self.module.hep_b_testing(hsi_event=self)
                self.module.syphilis_testing(hsi_event=self)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] < 38:
                self.module.point_of_care_hb_testing(hsi_event=self)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] < 40:
                self.module.iptp_administration(hsi_event=self)

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
    """This is the  HSI_CareOfWomenDuringPregnancy_FifthAntenatalCareContact which represents the fifth routine
    antenatal care contact (ANC5). It is scheduled by the HSI_CareOfWomenDuringPregnancy_FourthAntenatalCareContact for
    women who choose to seek additional ANC after their previous visit. It is recommended that this visit occur at 34
    weeks gestation. This event delivers the interventions to women which are part of ANC5. Additionally interventions
    that should be delivered according to a womans gestational age and position in her ANC schedule are delivered.
    Finally scheduling the next ANC contact in the occurs during this HSI along with admission to antenatal inpatient
    ward in the case of complications"""

    def __init__(self, module, person_id, facility_level_of_this_hsi):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_FifthAntenatalCareContact'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
        self.ACCEPTED_FACILITY_LEVEL = facility_level_of_this_hsi
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]

        # Here we define variables used within the function that checks in this ANC visit can run
        gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)
        this_contact = HSI_CareOfWomenDuringPregnancy_FifthAntenatalCareContact(
            self.module, person_id=person_id, facility_level_of_this_hsi=self.ACCEPTED_FACILITY_LEVEL)

        # Run the check
        can_anc_run = self.module.check_subsequent_anc_can_run(self, person_id, this_contact, 5, squeeze_factor,
                                                               gest_age_next_contact)

        if can_anc_run:
            logger.info(key='anc_facility_type', data=f'{df.at[person_id, "ac_facility_type"]}')
            logger.debug(key='msg', data=f'mother {person_id}presented for ANC 5 at a '
                                         f'{df.at[person_id, "ac_facility_type"]} at gestation '
                                         f'{df.at[person_id, "ps_gestational_age_in_weeks"]} ')

            assert mother.ps_gestational_age_in_weeks is not None
            assert mother.ps_gestational_age_in_weeks is not pd.NaT
            assert mother.ps_gestational_age_in_weeks >= 33
            assert mother.ac_total_anc_visits_current_pregnancy == 4

            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            #  =================================== INTERVENTIONS ====================================================
            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)
            self.module.screening_interventions_delivered_at_every_contact(hsi_event=self)
            self.module.calcium_supplementation(hsi_event=self)

            if mother.ps_gestational_age_in_weeks < 40:
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=6,
                                                     recommended_gestation_next_anc=gest_age_next_contact,
                                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)

            if mother.ps_gestational_age_in_weeks < 36:
                self.module.iptp_administration(hsi_event=self)
                self.module.hep_b_testing(hsi_event=self)
                self.module.syphilis_testing(hsi_event=self)

            elif mother.ps_gestational_age_in_weeks < 38:
                self.module.point_of_care_hb_testing(hsi_event=self)

            elif mother.ps_gestational_age_in_weeks < 40:
                self.module.iptp_administration(hsi_event=self)

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
    """This is the  HSI_CareOfWomenDuringPregnancy_SixthAntenatalCareContact which represents the sixth routine
    antenatal care contact (ANC6). It is scheduled by the HSI_CareOfWomenDuringPregnancy_FifthAntenatalCareContact for
    women who choose to seek additional ANC after their previous visit. It is recommended that this visit occur at 36
    weeks gestation. This event delivers the interventions to women which are part of ANC6. Additionally interventions
    that should be delivered according to a womans gestational age and position in her ANC schedule are delivered.
    Finally scheduling the next ANC contact in the occurs during this HSI along with admission to antenatal inpatient
    ward in the case of complications"""

    def __init__(self, module, person_id, facility_level_of_this_hsi):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_SixthAntenatalCareContact'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = facility_level_of_this_hsi
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]

        # Here we define variables used within the function that checks in this ANC visit can run
        gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)
        this_contact = HSI_CareOfWomenDuringPregnancy_SixthAntenatalCareContact(
            self.module, person_id=person_id, facility_level_of_this_hsi=self.ACCEPTED_FACILITY_LEVEL)

        # Run the check
        can_anc_run = self.module.check_subsequent_anc_can_run(self, person_id, this_contact, 6, squeeze_factor,
                                                               gest_age_next_contact)

        if can_anc_run:

            logger.info(key='anc_facility_type', data=f'{df.at[person_id, "ac_facility_type"]}')
            logger.debug(key='msg', data=f'mother {person_id}presented for ANC 6 at a '
                                         f'{df.at[person_id, "ac_facility_type"]} at gestation '
                                         f'{df.at[person_id, "ps_gestational_age_in_weeks"]} ')

            assert mother.ac_total_anc_visits_current_pregnancy == 5
            assert mother.ps_gestational_age_in_weeks is not None
            assert mother.ps_gestational_age_in_weeks is not pd.NaT
            assert mother.ps_gestational_age_in_weeks >= 35

            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)

            #  =================================== INTERVENTIONS ====================================================
            self.module.screening_interventions_delivered_at_every_contact(hsi_event=self)
            self.module.calcium_supplementation(hsi_event=self)

            if mother.ps_gestational_age_in_weeks < 40:
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=7,
                                                     recommended_gestation_next_anc=gest_age_next_contact,
                                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)

            if mother.ps_gestational_age_in_weeks < 38:
                self.module.point_of_care_hb_testing(hsi_event=self)

            elif mother.ps_gestational_age_in_weeks < 40:
                self.module.iptp_administration(hsi_event=self)

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
    """"This is the  HSI_CareOfWomenDuringPregnancy_SeventhAntenatalCareContact which represents the seventh routine
    antenatal care contact (ANC7). It is scheduled by the HSI_CareOfWomenDuringPregnancy_SixthAntenatalCareContact for
    women who choose to seek additional ANC after their previous visit. It is recommended that this visit occur at 36
    weeks gestation. This event delivers the interventions to women which are part of ANC7. Additionally interventions
    that should be delivered according to a womans gestational age and position in her ANC schedule are delivered.
    Finally scheduling the next ANC contact in the occurs during this HSI along with admission to antenatal inpatient
    ward in the case of complications"""

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

        # Here we define variables used within the function that checks in this ANC visit can run
        gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)
        this_contact = HSI_CareOfWomenDuringPregnancy_SeventhAntenatalCareContact(
            self.module, person_id=person_id, facility_level_of_this_hsi=self.ACCEPTED_FACILITY_LEVEL)

        # Run the check
        can_anc_run = self.module.check_subsequent_anc_can_run(self, person_id, this_contact, 7, squeeze_factor,
                                                               gest_age_next_contact)

        if can_anc_run:
            logger.info(key='anc_facility_type', data=f'{df.at[person_id, "ac_facility_type"]}')
            logger.debug(key='msg', data=f'mother {person_id}presented for ANC 7 at a '
                                         f'{df.at[person_id, "ac_facility_type"]} at gestation '
                                         f'{df.at[person_id, "ps_gestational_age_in_weeks"]} ')

            assert mother.ac_total_anc_visits_current_pregnancy == 6
            assert mother.ps_gestational_age_in_weeks is not None
            assert mother.ps_gestational_age_in_weeks is not pd.NaT
            assert mother.ps_gestational_age_in_weeks >= 37

            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            #  =================================== INTERVENTIONS ====================================================
            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)
            self.module.screening_interventions_delivered_at_every_contact(hsi_event=self)
            self.module.calcium_supplementation(hsi_event=self)

            if mother.ps_gestational_age_in_weeks < 40:
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
    """"This is the  HSI_CareOfWomenDuringPregnancy_EighthAntenatalCareContact which represents the eighth routine
    antenatal care contact (ANC8). It is scheduled by the HSI_CareOfWomenDuringPregnancy_SeventhAntenatalCareContact for
    women who choose to seek additional ANC after their previous visit. It is recommended that this visit occur at 36
    weeks gestation. This event delivers the interventions to women which are part of ANC8. Additionally interventions
    that should be delivered according to a womans gestational age and position in her ANC schedule are delivered.
    Finally scheduling the next ANC contact in the occurs during this HSI along with admission to antenatal inpatient
    ward in the case of complications"""

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

        # Here we define variables used within the function that checks in this ANC visit can run
        gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)
        this_contact = HSI_CareOfWomenDuringPregnancy_EighthAntenatalCareContact(
            self.module, person_id=person_id, facility_level_of_this_hsi=self.ACCEPTED_FACILITY_LEVEL)

        # Run the check
        can_anc_run = self.module.check_subsequent_anc_can_run(self, person_id, this_contact, 8, squeeze_factor,
                                                               gest_age_next_contact)

        if can_anc_run:

            logger.info(key='anc_facility_type', data=f'{df.at[person_id, "ac_facility_type"]}')
            logger.debug(key='msg', data=f'mother {person_id}presented for ANC 7 at a'
                                         f' {df.at[person_id, "ac_facility_type"]} at gestation '
                                         f'{df.at[person_id, "ps_gestational_age_in_weeks"]} ')

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


class HSI_CareOfWomenDuringPregnancy_PresentsForInductionOfLabour(HSI_Event, IndividualScopeEventMixin):
    """
    This is HSI_CareOfWomenDuringPregnancy_PresentsForInductionOfLabour. It is schedule by the PregnancySupervisor Event
    for women who present to the health system for induction as their labour has progressed longer than expected.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_PresentsForInductionOfLabour'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        # If the woman is no longer alive, pregnant is in labour or is an inpatient already then the event doesnt run
        if not df.at[person_id, 'is_alive'] or not df.at[person_id, 'is_pregnant'] or \
           not df.at[person_id, 'la_currently_in_labour'] or not df.at[person_id, 'hs_is_inpatient']:
            return

        # We set this admission property to show shes being admitted for induction of labour and hand her over to the
        # labour events
        df.at[person_id, 'ac_admitted_for_immediate_delivery'] = 'induction_now'
        logger.debug(key='msg', data=f'Mother {person_id} will move to labour ward for '
                                     f'{df.at[person_id, "ac_admitted_for_immediate_delivery"]} today')

        self.sim.schedule_event(LabourOnsetEvent(self.sim.modules['Labour'], person_id), self.sim.date)


class HSI_CareOfWomenDuringPregnancy_MaternalEmergencyAssessment(HSI_Event, IndividualScopeEventMixin):
    """
    This is HSI_CareOfWomenDuringPregnancy_MaternalEmergencyAssessment. It is schedule by the PregnancySupervisor Event
    for women who choose to seek care for emergency treatment in pregnancy (due to severe pre-eclampsia/eclampsia,
    antepartum haemorrhage, premature rupture of membranes or chorioamnionitis). It is assumed women present to this
    event as their first point of contact for an emergency in pregnancy, and therefore circumnavigate regular A&E.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_MaternalEmergencyAssessment'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        # TODO: remove and schedule through ED

        if not df.at[person_id, 'is_alive'] or not df.at[person_id, 'is_pregnant']:
            return

        if ~df.at[person_id, 'hs_is_inpatient'] and ~df.at[person_id, 'la_currently_in_labour']:
            logger.debug(key='msg', data=f'Mother {person_id} has presented at HSI_CareOfWomenDuringPregnancy_Maternal'
                                         f'EmergencyAssessment to seek care for a complication ')

            facility_level = self.module.rng.choice(['1a', '1b'], p=[0.5, 0.5])  # todo note choice

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
    """
    This is HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare. This HSI can be scheduled by any of the ANC HSIs
    for women who have been identified as having complications or by HSI_CareOfWomenDuringPregnancy_
    MaternalEmergencyAssessment or HSI_CareOfWomenDuringPregnancy_AntenatalOutpatientFollowUp.This HSI represents the
    antenatal ward which would deliver care to women experiencing complications associated with their pregnancy
    including anaemia, hypertension, gestational diabetes, antepartum haemorrhage, premature rupture of membranes or
    chorioamnionitis. For women whom delivery is indicated as part of treatment for a complications they are scheduled
    to the LabourOnset Event
    """

    def __init__(self, module, person_id, facility_level_this_hsi):
        super().__init__(module, person_id=person_id)

        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_AntenatalWardInpatientCare'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'InpatientDays': 1})
        self.ACCEPTED_FACILITY_LEVEL = facility_level_this_hsi
        self.ALERT_OTHER_DISEASES = []

        beddays = self.module.calculate_beddays(person_id)
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': beddays})

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]

        logger.debug(key='msg', data=f'Bed-days allocated to this event:'
                                     f' {self.bed_days_allocated_to_this_event}')

        if not mother.is_alive:
            return

        if mother.is_pregnant and ~mother.la_currently_in_labour and ~mother.hs_is_inpatient:

            logger.debug(key='message', data=f'Mother {person_id} has been admitted for treatment of a complication of '
                                             f'her pregnancy ')

            # The event represents inpatient care delivered within the antenatal ward at a health facility. Therefore
            # it is assumed that women with a number of different complications could be sent to this HSI for treatment.

            # ================================= INITIATE TREATMENT FOR ANAEMIA ========================================
            # Women who are referred from ANC or an outpatient appointment following point of care Hb which detected
            # anaemia first have a full blood count test to determine the severity of their anaemia
            if mother.ps_anaemia_in_pregnancy != 'none':

                # This test returns one of a number of possible outcomes as seen below...
                fbc_result = self.module.full_blood_count_testing(self)
                assert fbc_result == 'no_test' or 'none' or 'mild' or 'moderate' or 'severe'

                # If the test is not carried out, no treatment is provided and the woman is discharged
                if fbc_result == 'no_test':
                    logger.debug(key='message', data='No FBC given due to resource constraints')

                # If the result returns none, anaemia has not been detected via an FBC and the woman is discharged
                # without treatment
                elif fbc_result == 'none':
                    logger.debug(key='message', data=f'Mother {person_id} has not had anaemia detected via an FBC and '
                                                     f'will be discharged')

                # If the FBC detected non severe anaemia (Hb >7) she is treated
                elif fbc_result == 'mild' or fbc_result == 'moderate':

                    # Women are started on daily iron and folic acid supplementation (if they are not already receiving
                    # supplements) as treatment for mild/moderate anaemia
                    if ~mother.ac_receiving_iron_folic_acid:
                        self.module.start_iron_and_folic_acid(person_id, self)

                    # Some anaemia causing deficiencies (folate, B12)  are detected via a FBC, and therefore can be
                    # these deficiencies can be treated via this function
                    self.module.treatment_of_anaemia_causing_deficiencies(person_id, self)

                elif fbc_result == 'severe':
                    # In the case of severe anaemia (Hb <7) then, in addition to the above treatments, this woman
                    # should receive a blood transfusion to correct her anaemia
                    self.module.antenatal_blood_transfusion(person_id, self, cause='severe_anaemia')
                    self.module.treatment_of_anaemia_causing_deficiencies(person_id, self)
                    if ~mother.ac_receiving_iron_folic_acid:
                        self.module.start_iron_and_folic_acid(person_id, self)

                if fbc_result == 'mild' or fbc_result == 'moderate' or fbc_result == 'severe':
                    # Women treated for anaemia will need follow up to ensure the treatment has been effective. Clinical
                    # guidelines suggest follow up one month after treatment

                    # To avoid issues with scheduling of events we assume women who are not scheduled to return to
                    # routine ANC OR their next ANC appointment is more than a month away will be asked to routine for
                    # follow up
                    follow_up_date = self.sim.date + DateOffset(days=28)
                    if pd.isnull(mother.ac_date_next_contact) or ((mother.ac_date_next_contact - self.sim.date) >
                                                                  pd.to_timedelta(28, unit='D')):

                        outpatient_checkup = HSI_CareOfWomenDuringPregnancy_AntenatalOutpatientManagementOfAnaemia(
                            self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id)

                        self.sim.modules['HealthSystem'].schedule_hsi_event(outpatient_checkup, priority=0,
                                                                            topen=follow_up_date,
                                                                            tclose=follow_up_date + DateOffset(days=7))

            # ======================== INITIATE TREATMENT FOR GESTATIONAL DIABETES (case management) ==================
            # Women admitted with gestational diabetes are given dietary and exercise advice as first line treatment
            if (mother.ps_gest_diab == 'uncontrolled') and (mother.ac_gest_diab_on_treatment == 'none'):
                df.at[person_id, 'ac_gest_diab_on_treatment'] = 'diet_exercise'
                df.at[person_id, 'ps_gest_diab'] = 'controlled'

                # We then schedule GestationalDiabetesGlycaemicControlEvent which determines if this treatment will be
                # effective in controlling this womans blood sugar prior to her next check up
                from tlo.methods.pregnancy_supervisor import (
                    GestationalDiabetesGlycaemicControlEvent,
                )
                self.sim.schedule_event(GestationalDiabetesGlycaemicControlEvent(
                    self.sim.modules['PregnancySupervisor'], person_id), self.sim.date + DateOffset(days=7))

                # We then schedule this woman to return for blood sugar testing to evaluate the effectiveness of her
                # treatment and potentially move to second line treatment
                check_up_date = self.sim.date + DateOffset(days=28)

                outpatient_checkup = HSI_CareOfWomenDuringPregnancy_AntenatalOutpatientManagementOfGestationalDiabetes(
                    self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(outpatient_checkup, priority=0,
                                                                    topen=check_up_date,
                                                                    tclose=check_up_date + DateOffset(days=3))

            # =============================== INITIATE TREATMENT FOR HYPERTENSION =====================================
            # Treatment delivered to mothers with hypertension is dependent on severity. Women admitted due to more mild
            # hypertension are started on regular oral antihypertensives therapy (reducing risk of progression to more
            # severe hypertension)

            if ((mother.ps_htn_disorders == 'gest_htn') or
                (mother.ps_htn_disorders == 'mild_pre_eclamp')) and \
               ~mother.ac_gest_htn_on_treatment:
                self.module.initiate_maintenance_anti_hypertensive_treatment(person_id, self)

            # Women with severe gestational hypertension are also started on routine oral antihypertensives (if not
            # already receiving- this will prevent progression once this episode of severe hypertension has been
            # rectified)
            elif mother.ps_htn_disorders == 'severe_gest_htn':
                if ~mother.ac_gest_htn_on_treatment:
                    self.module.initiate_maintenance_anti_hypertensive_treatment(person_id, self)

                # In addition, women with more severe disease are given intravenous anti hypertensives to reduce risk
                # of death
                self.module.initiate_treatment_for_severe_hypertension(person_id, self)

            # Treatment guidelines dictate that women with severe forms of pre-eclampsia should be admitted for delivery
            # to reduce risk of death and pregnancy loss
            elif (mother.ps_htn_disorders == 'severe_pre_eclamp') or (mother.ps_htn_disorders == 'eclampsia'):

                # This property stores what type of delivery this woman is being admitted for
                df.at[person_id, 'ac_admitted_for_immediate_delivery'] = 'induction_now'
                logger.debug(key='msg', data=f'{person_id} will be admitted for induction due to '
                                             f'{mother.ps_htn_disorders}')

                # Women are started on oral antihypertensives also
                if ~mother.ac_gest_htn_on_treatment:
                    self.module.initiate_maintenance_anti_hypertensive_treatment(person_id, self)

                # And are given intravenous magnesium sulfate which reduces risk of death from eclampsia and reduces a
                # womans risk of progressing from severe pre-eclampsia to eclampsia during the intrapartum period
                self.module.treatment_for_severe_pre_eclampsia_or_eclampsia(person_id,
                                                                            hsi_event=self)
                # Finally intravenous antihypertensives are also given
                self.module.initiate_treatment_for_severe_hypertension(person_id, self)

            # ========================= INITIATE TREATMENT FOR ANTEPARTUM HAEMORRHAGE =================================
            # Treatment delivered to mothers due to haemorrhage in the antepartum period is dependent on the underlying
            # etiology of the bleeding (in this model, whether a woman is experiencing a placental abruption or
            # placenta praevia)

            if mother.ps_antepartum_haemorrhage != 'none':
                # ---------------------- APH SECONDARY TO PLACENTAL ABRUPTION -----------------------------------------
                if mother.ps_placental_abruption:
                    # Women experiencing placenta abruption at are admitted for immediate
                    # caesarean delivery due to high risk of mortality/morbidity
                    df.at[person_id, 'ac_admitted_for_immediate_delivery'] = 'caesarean_now'
                    logger.debug(key='msg', data=f'{person_id} will be admitted for caesarean due to APH')

                # ---------------------- APH SECONDARY TO PLACENTA PRAEVIA -----------------------------------------
                if mother.ps_placenta_praevia:
                    # The treatment plan for a woman with placenta praevia is dependent on both the severity of the
                    # bleed and her current gestation at the time of bleeding

                    if mother.ps_antepartum_haemorrhage == 'severe':
                        # Women experiencing severe bleeding are admitted immediately for caesarean section
                        df.at[person_id, 'ac_admitted_for_immediate_delivery'] = 'caesarean_now'
                        logger.debug(key='msg', data=f'{person_id} will be admitted for caesarean due to APH')

                    elif (mother.ps_antepartum_haemorrhage != 'severe') and (mother.ps_gestational_age_in_weeks >= 37):
                        # Women experiencing mild or moderate bleeding but who are around term gestation are admitted
                        # for caesarean
                        df.at[person_id, 'ac_admitted_for_immediate_delivery'] = 'caesarean_now'
                        logger.debug(key='msg', data=f'{person_id} will be admitted for caesarean due to APH')

                    elif (mother.ps_antepartum_haemorrhage != 'severe') and (mother.ps_gestational_age_in_weeks < 37):
                        # Women with more mild bleeding remain as inpatients until their gestation has increased and
                        # then will be delivered by caesarean - (no risk of death associated with mild/moderate bleeds)
                        df.at[person_id, 'ac_admitted_for_immediate_delivery'] = 'caesarean_future'
                        logger.debug(key='msg', data=f'{person_id} will be admitted for caesarean when her gestation '
                                                     f'has increased due to APH')

                        # self.module.antenatal_blood_transfusion(person_id, self, cause='antepartum_haem')

                assert df.at[person_id, 'ac_admitted_for_immediate_delivery'] != 'none'

            # ===================================== INITIATE TREATMENT FOR PROM =======================================
            # Treatment for women with premature rupture of membranes is dependent upon a womans gestational age and if
            # she also has an infection of membrane surrounding the foetus (the chorion)

            if mother.ps_premature_rupture_of_membranes and ((mother.ps_chorioamnionitis == 'none') or
                                                             (mother.ps_chorioamnionitis == 'histological')):
                # If the woman has PROM but no infection, she is given prophylactic antibiotics which will reduce
                # the risk of neonatal infection
                self.module.antibiotics_for_prom(person_id, self)

                # Guidelines suggest women over 34 weeks of gestation should be admitted for induction to to
                # increased risk of morbidity and mortality
                if mother.ps_gestational_age_in_weeks >= 34:
                    df.at[person_id, 'ac_admitted_for_immediate_delivery'] = 'induction_now'
                    logger.debug(key='msg', data=f'{person_id} will be admitted for induction due to prom/chorio')

                # Otherwise they may stay as an inpatient until their gestation as increased prior to delivery
                elif mother.ps_gestational_age_in_weeks < 34:
                    df.at[person_id, 'ac_admitted_for_immediate_delivery'] = 'induction_future'
                    logger.debug(key='msg', data=f'{person_id} will be admitted for induction when her gestation'
                                                 f' has increase due prom/chorio')

            # ============================== INITIATE TREATMENT FOR CHORIOAMNIONITIS ==================================
            if mother.ps_chorioamnionitis == 'clinical':
                self.module.antibiotics_for_chorioamnionitis(person_id, self)
                df.at[person_id, 'ac_admitted_for_immediate_delivery'] = 'induction_now'
                logger.debug(key='msg', data=f'{person_id} will be admitted for induction due to prom/chorio')

            # ======================== ADMISSION FOR DELIVERY (INDUCTION) ========================================
            # Women for whom immediate delivery is indicated are schedule to move straight to the labour model where
            # they will have the appropriate properties set and facility delivery at a hospital scheduled (mode of
            # delivery will match the recommended mode here)
            if (df.at[person_id, 'ac_admitted_for_immediate_delivery'] == 'induction_now') or (df.at[person_id,
                                                                                                     'ac_admitted_'
                                                                                                     'for_immediate_'
                                                                                                     'delivery'] ==
                                                                                               'caesarean_now'):

                logger.debug(key='msg', data=f'Mother {person_id} will move to labour ward for '
                                             f'{df.at[person_id, "ac_admitted_for_immediate_delivery"]} today')

                self.sim.schedule_event(LabourOnsetEvent(self.sim.modules['Labour'], person_id), self.sim.date)

            # Women who require delivery BUT are not in immediate risk of morbidity/mortality will remain as
            # inpatients until they can move to the labour model. Currently it is possible for women to go into
            # labour whilst as an inpatient - it is assumed they are delivered via the mode recommended here
            # (i.e induction/caesarean)
            elif (df.at[person_id, 'ac_admitted_for_immediate_delivery'] == 'caesarean_future') or (df.at[person_id,
                                                                                                          'ac_admitted_'
                                                                                                          'for_'
                                                                                                          'immediate_'
                                                                                                          'delivery'] ==
                                                                                                    'induction_future'):

                # Here we calculate how many days this woman needs to remain on the antenatal ward before she can go
                # for delivery (assuming delivery is indicated to occur at 37 weeks)
                if mother.ps_gestational_age_in_weeks < 37:
                    days_until_safe_for_cs = int((37 * 7) - (mother.ps_gestational_age_in_weeks * 7))
                else:
                    days_until_safe_for_cs = 1

                # We schedule the LabourOnset event for this woman will be able to progress for delivery
                admission_date = self.sim.date + DateOffset(days=days_until_safe_for_cs)
                logger.debug(key='msg', data=f'Mother {person_id} will move to labour ward for '
                                             f'{df.at[person_id, "ac_admitted_for_immediate_delivery"]} on '
                                             f'{admission_date}')

                self.sim.schedule_event(LabourOnsetEvent(self.sim.modules['Labour'], person_id),
                                        admission_date)

    def did_not_run(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare: did not run')

    def not_available(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare: cannot not run'
                                         ' with this configuration')


class HSI_CareOfWomenDuringPregnancy_AntenatalOutpatientManagementOfAnaemia(HSI_Event, IndividualScopeEventMixin):
    """
    HSI_CareOfWomenDuringPregnancy_AntenatalManagementOfAnaemia. It is scheduled by HSI_CareOfWomenDuringPregnancy_
    AntenatalWardInpatientCare for women who have developed anaemia during pregnancy. This event manages repeat blood
    testing for women who were found to be anaemic and treated. If the woman remains anaemic she is readmitted to the
    inpatient ward for further care.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_AntenatalOutpatientManagementOfAnaemia'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]

        if not mother.is_alive or not mother.is_pregnant:
            return

        # We only run the event if the woman is not already in labour or already admitted due to something else
        if ~mother.la_currently_in_labour and ~mother.hs_is_inpatient:

            # Health care worker performs a full blood count
            fbc_result = self.module.full_blood_count_testing(self)

            # If the consumables are not available the test isn't given
            if fbc_result == 'no_test':
                logger.debug(key='message', data='No FBC given due to resource constraints')

            # If the test determines the woman is no longer anaemia then no further action is taken at this time
            elif fbc_result == 'none':
                logger.debug(key='message', data=f'Mother {person_id} has not had anaemia detected via an FBC')

            # If she is determined to still be anaemic she is admitted for additional treatment via the inpatient event
            elif fbc_result == 'mild' or fbc_result == 'moderate' or fbc_result == 'severe':

                facility_level = self.module.rng.choice(['1a', '1b'], p=[0.5, 0.5])  # todo note choice

                admission = HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare(
                    self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id,
                    facility_level_this_hsi=facility_level)

                self.sim.modules['HealthSystem'].schedule_hsi_event(admission, priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1))

    def did_not_run(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_AntenatalOutpatientManagementOfAnaemia: did '
                                         'not run')

    def not_available(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_AntenatalOutpatientManagementOfAnaemia: '
                                         'cannot not run with this configuration')


class HSI_CareOfWomenDuringPregnancy_AntenatalOutpatientManagementOfGestationalDiabetes(HSI_Event,
                                                                                        IndividualScopeEventMixin):
    """
     This is HSI_CareOfWomenDuringPregnancy_AntenatalOutpatientManagementOfGestationalDiabetes. It is scheduled by
     HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare for women who have developed gestational diabetes during
     pregnancy. This event manages repeat blood testing for women who were found to have GDM and treated. If the woman
     remains hyperglycaemic she is moved to the next line treatment and scheduled to return for follow up.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_AntenatalOutpatientFollowUp'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        # consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        mother = df.loc[person_id]

        from tlo.methods.pregnancy_supervisor import GestationalDiabetesGlycaemicControlEvent

        if not mother.is_alive or not mother.is_pregnant:
            return

        if ~mother.la_currently_in_labour and ~mother.hs_is_inpatient and mother.ps_gest_diab != 'none' \
                and (mother.ac_gest_diab_on_treatment != 'none') and (mother.ps_gestational_age_in_weeks > 21):
            logger.debug(key='msg', data=f'Mother {person_id} has presented for review of her GDM')

            # Nothing happens to women who arrive at follow up with well controlled GDM (treatment is effective). We now
            # assume that the treatment they are on (started in AntenatalWardInpatientCare) remains effective for the
            # length of their pregnancy
            if (mother.ps_gest_diab == 'controlled') and (mother.ac_gest_diab_on_treatment != 'none'):
                logger.debug(key='msg', data=f'Mother {person_id} has well controlled GDM on current treatment and '
                                             f'doesnt need a further check up at present')

            # If the treatment a woman was started on has not controlled her hyperglycemia she will be started on the
            # next treatment
            elif mother.ps_gest_diab == 'uncontrolled':

                # Women for whom diet and exercise was not effective in controlling hyperglycemia are started on oral
                # meds
                if mother.ac_gest_diab_on_treatment == 'diet_exercise':

                    # Currently we assume women are given enough tablets to last the length of their pregnancy
                    # approx_days_of_pregnancy = (40 - df.at[person_id, 'ps_gestational_age_in_weeks']) * 7

                    # item_code_oral_antidiabetics = pd.unique(
                    #     consumables.loc[consumables['Items'] == 'Glibenclamide 5mg_1000_CMST', 'Item_Code'])[0]
                    item_code_oral_antidiabetics = 0

                    # consumables_anti_diabs = {
                    #     'Intervention_Package_Code': {},
                    #     'Item_Code': {item_code_oral_antidiabetics: approx_days_of_pregnancy * 2}}  # dose 5mg BD

                    # outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                    #     hsi_event=self,
                    #     cons_req_as_footprint=consumables_anti_diabs)
                    outcome_of_request_for_consumables = {
                        'Intervention_Package_Code': defaultdict(lambda: True),
                        'Item_Code': defaultdict(lambda: True)
                    }

                    # If the meds are available women are started on that treatment
                    if outcome_of_request_for_consumables['Item_Code'][item_code_oral_antidiabetics]:
                        df.at[person_id, 'ac_gest_diab_on_treatment'] = 'orals'

                        # Assume new treatment is effective in controlling blood glucose on initiation
                        df.at[person_id, 'ps_gest_diab'] = 'controlled'

                        # Schedule GestationalDiabetesGlycaemicControlEvent which determines if this new treatment will
                        # effectively control blood glucose prior to next follow up
                        self.sim.schedule_event(GestationalDiabetesGlycaemicControlEvent(
                            self.sim.modules['PregnancySupervisor'], person_id), self.sim.date + DateOffset(days=7))

                        # Schedule follow-up
                        check_up_date = self.sim.date + DateOffset(days=28)

                        outpatient_checkup = \
                            HSI_CareOfWomenDuringPregnancy_AntenatalOutpatientManagementOfGestationalDiabetes(
                                self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id)
                        self.sim.modules['HealthSystem'].schedule_hsi_event(outpatient_checkup, priority=0,
                                                                            topen=check_up_date,
                                                                            tclose=check_up_date + DateOffset(days=3))

                # This process is repeated for mothers for whom oral medication is not effectively controlling their
                # blood sugar- they are started on insulin
                if mother.ac_gest_diab_on_treatment == 'orals':

                    # item_code_oral_insulin = pd.unique(
                    #     consumables.loc[consumables['Items'] == 'Insulin soluble 100 IU/ml, 10ml_each_CMST',
                    #                     'Item_Code'])[0]
                    item_code_oral_insulin = 0

                    # consumables_insulin = {
                    #     'Intervention_Package_Code': {},
                    #     'Item_Code': {item_code_oral_insulin: 5}}

                    # outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                    #     hsi_event=self,
                    #     cons_req_as_footprint=consumables_insulin)
                    outcome_of_request_for_consumables = {
                        'Intervention_Package_Code': defaultdict(lambda: True),
                        'Item_Code': defaultdict(lambda: True)
                    }

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

    def did_not_run(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_AntenatalOutpatientManagementOfGestational'
                                         'Diabetes: did '
                                         'not run')

    def not_available(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_AntenatalOutpatientManagementOfGestational'
                                         'Diabetes: cannot not run with this configuration')


class HSI_CareOfWomenDuringPregnancy_PostAbortionCaseManagement(HSI_Event, IndividualScopeEventMixin):
    """
    This is HSI_CareOfWomenDuringPregnancy_PostAbortionCaseManagement. It is scheduled by
    HSI_GenericEmergencyFirstApptAtFacilityLevel1 for women who have presented to hospital due to the complications of
    either induced or spontaneous abortion. This event manages interventions delivered for women who are experiencing
    either sepsis, haemorrhage or injury post-abortion.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_PostAbortionCaseManagement'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'InpatientDays': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 3})

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]
        params = self.module.parameters
        abortion_complications = self.sim.modules['PregnancySupervisor'].abortion_complications
        # consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        if not mother.is_alive:
            return

        # We check only women with complications post abortion are sent to this event
        assert abortion_complications.has_any([person_id], 'sepsis', 'haemorrhage', 'injury', first=True)

        # Check the availability of consumables
        # pkg_code_infection = pd.unique(
        #     consumables.loc[consumables['Intervention_Pkg'] == 'Post-abortion case management',
        #                     'Intervention_Pkg_Code'])[0]

        # all_available = self.get_all_consumables(
        #     pkg_codes=[pkg_code_infection])
        all_available = True

        # If consumables are available then individual interventions can be delivered
        if all_available:
            evac_procedures = ['d_and_c', 'mva', 'misoprostol']
            probability_of_evac_procedure = params['prob_evac_procedure_pac']
            random_draw = self.module.rng.choice(evac_procedures, p=probability_of_evac_procedure)
            self.module.pac_interventions.set(person_id, random_draw)

        # Women who are septic following their abortion are given antibiotics
        if abortion_complications.has_any([person_id], 'sepsis', first=True):
            if all_available:
                self.module.pac_interventions.set(person_id, 'antibiotics')

        # Minor injuries following induced abortion are treated
        if abortion_complications.has_any([person_id], 'injury', first=True):
            if all_available:
                self.module.pac_interventions.set(person_id, 'injury_repair')

        # And women who experience haemorrhage are provided with blood
        if abortion_complications.has_any([person_id], 'haemorrhage', first=True):
            self.module.antenatal_blood_transfusion(person_id, self, cause='abortion')

    def did_not_run(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_PostAbortionCaseManagement: did not run')

    def not_available(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_PostAbortionCaseManagement: cannot not run '
                                         'with this configuration')


class HSI_CareOfWomenDuringPregnancy_TreatmentForEctopicPregnancy(HSI_Event, IndividualScopeEventMixin):
    """
    This is HSI_CareOfWomenDuringPregnancy_TreatmentForEctopicPregnancy. It is scheduled by
    HSI_GenericEmergencyFirstApptAtFacilityLevel1 for women who have presented to hospital due to ectopic pregnancy.
    This event manages interventions delivered as part of the case management of ectopic pregnancy
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_TreatmentForEctopicPregnancy'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'MajorSurg': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'
        self.ALERT_OTHER_DISEASES = []
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 5})

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        # consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        mother = df.loc[person_id]

        if not mother.is_alive:
            return

        # Check only the correct women are sent to this event
        assert mother.ps_ectopic_pregnancy != 'none'

        logger.debug(key='message', data='This is HSI_CareOfWomenDuringPregnancy_TreatmentForEctopicPregnancy, '
                                         f'person {person_id} has been diagnosed with ectopic pregnancy after '
                                         f'presenting and will now undergo treatment')

        # We define the required consumables and check their availability
        # ectopic_pkg = pd.unique(consumables.loc[consumables['Intervention_Pkg'] == 'Ectopic case management',
        #                                         'Intervention_Pkg_Code'])[0]

        # all_available = self.get_all_consumables(
        #     items_codes=[self.module.item)
        all_available = True  # remove use of package_codes

        # If they are available then treatment can go ahead
        if all_available:
            self.sim.modules['PregnancySupervisor'].mother_and_newborn_info[person_id]['delete_mni'] = True

            logger.debug(key='message', data=f'Mother {person_id} will now undergo surgery due to ectopic pregnancy as '
                                             f'consumables are available')

            # For women who have sought care after they have experienced rupture we use this treatment variable to
            # reduce risk of death (women who present prior to rupture do not pass through the death event as we assume
            # rupture is on the causal pathway to death - hence no treatment property)
            if mother.ps_ectopic_pregnancy == 'ruptured':
                df.at[person_id, 'ac_ectopic_pregnancy_treated'] = True

        else:
            # However if treatment cant be delivered for women who have not yet experienced rupture (due to lack of
            # consumables) we schedule these women to arrive at the rupture event as they have not received treatment
            self.module.ectopic_pregnancy_treatment_doesnt_run(person_id)
            logger.debug(key='msg', data=f'Mother {person_id} could not receive treatment due to insufficient '
                                         f'consumables')

    def did_not_run(self):
        person_id = self.target

        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_TreatmentForEctopicPregnancy: did not run')
        self.module.ectopic_pregnancy_treatment_doesnt_run(person_id)

        return False

    def not_available(self):
        person_id = self.target

        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_TreatmentForEctopicPregnancy: cannot not run '
                                         'with this configuration')
        self.module.ectopic_pregnancy_treatment_doesnt_run(person_id)

        return False


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

        # ra_lower_limit = 14
        # ra_upper_limit = 50
        # women_reproductive_age = df.index[(df.is_alive & (df.sex == 'F') & (df.age_years > ra_lower_limit) &
        #                                   (df.age_years < ra_upper_limit))]
        # total_women_reproductive_age = len(women_reproductive_age)

        dict_for_output = {'mean_ga_first_anc': cumm_gestation / total_anc1_visits,
                           'proportion_anc1_first_trimester': (anc1_in_first_trimester / total_anc1_visits) * 100,
                           'early_anc3_proportion_of_births': (early_anc3 / total_births_last_year) * 100,
                           'early_anc3': early_anc3,
                           'diet_supps_6_months': diet_sup_6_months}

        logger.info(key='anc_summary_statistics',
                    data=dict_for_output,
                    description='Yearly summary statistics output from the antenatal care module')

        for k in self.module.anc_tracker:
            self.module.anc_tracker[k] = 0
