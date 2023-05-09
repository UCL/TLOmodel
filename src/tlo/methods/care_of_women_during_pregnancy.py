from pathlib import Path

import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import Metadata, pregnancy_helper_functions
from tlo.methods.dxmanager import DxTest
from tlo.methods.epi import HSI_TdVaccine
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.labour import LabourOnsetEvent
from tlo.methods.malaria import HSI_MalariaIPTp
from tlo.methods.tb import HSI_Tb_ScreeningAndRefer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CareOfWomenDuringPregnancy(Module):
    """This is the CareOfWomenDuringPregnancy module which contains health system interaction events relevant to
     pregnancy and pregnancy loss including:

     1.) HSI_CareOfWomenDuringPregnancy_AntenatalCareContact (1-8) representing all 8 routine antenatal care contacts
        (ANC) recommended during pregnancy (with sequential scheduling of each event occurring within the HSI)

     2.) HSI_CareOfWomenDuringPregnancy_FocusedANCVisit which replicates the pre 2016 structure of ANC (focused ANC)
         used in some analysis scripts

     3.) HSI_CareOfWomenDuringPregnancy_PostAbortionCaseManagement representing treatment for complications following
         abortion (post abortion care of PAC) for women seeking care from the community

     4.) HSI_CareOfWomenDuringPregnancy_TreatmentForEctopicPregnancy representing treatment for ectopic pregnancy for
         women seeking care from the community

     5.) HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare which represents antenatal inpatient care for women
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

        # First we define dictionaries which will store the current parameters of interest (to allow parameters to
        # change between 2010 and 2020)
        self.current_parameters = dict()

        # and then define a dictionary which will hold the required consumables for each intervention
        self.item_codes_preg_consumables = dict()

        # Finally set up a counter for ANC visits.
        self.anc_counter = dict()

    INIT_DEPENDENCIES = {'Demography', 'HealthSystem', 'PregnancySupervisor'}

    ADDITIONAL_DEPENDENCIES = {'Contraception', 'Labour', 'Lifestyle'}

    METADATA = {
        Metadata.USES_HEALTHSYSTEM,
    }

    PARAMETERS = {

        # n.b. Parameters are stored as LIST variables due to containing values to match both 2010 and 2015 data.

        # CARE SEEKING...
        'prob_seek_anc2': Parameter(
            Types.LIST, 'Probability a women who is not predicted to attended four or more ANC visits will attend '
                        'ANC2'),
        'prob_seek_anc3': Parameter(
            Types.LIST, 'Probability a women who is not predicted to attended four or more ANC visits will attend '
                        'ANC3'),
        'prob_seek_anc5': Parameter(
            Types.LIST, 'Probability a women who is predicted to attend four or more ANC visits will attend ANC5'),
        'prob_seek_anc6': Parameter(
            Types.LIST, 'Probability a women who is predicted to attend four or more ANC visits will attend ANC6'),
        'prob_seek_anc7': Parameter(
            Types.LIST, 'Probability a women who is predicted to attend four or more ANC visits will attend ANC7'),
        'prob_seek_anc8': Parameter(
            Types.LIST, 'Probability a women who is predicted to attend four or more ANC visits will attend ANC8'),

        # TREATMENT EFFECTS...
        'effect_of_ifa_for_resolving_anaemia': Parameter(
            Types.LIST, 'treatment effectiveness of starting iron and folic acid on resolving anaemia'),
        'treatment_effect_blood_transfusion_anaemia': Parameter(
            Types.LIST, 'treatment effectiveness of blood transfusion for anaemia in pregnancy'),

        # INTERVENTION PROBABILITIES...
        'squeeze_factor_threshold_anc': Parameter(
            Types.INT, 'squeeze factor threshold over which an ANC appointment cannot run'),
        'prob_intervention_delivered_urine_ds': Parameter(
            Types.LIST, 'probability a woman will receive the intervention "urine dipstick" given that the HSI has ran '
                        'and the consumables are available (proxy for clinical quality)'),
        'prob_intervention_delivered_bp': Parameter(
            Types.LIST, 'probability a woman will receive the intervention "blood pressure measurement" given that the '
                        'HSI has ran and the consumables are available (proxy for clinical quality)'),
        'prob_intervention_delivered_ifa': Parameter(
            Types.LIST, 'probability a woman will receive the intervention "iron and folic acid" given that the HSI'
                        ' has ran and the consumables are available (proxy for clinical quality)'),
        'prob_adherent_ifa': Parameter(
            Types.LIST, 'probability a woman who is given iron and folic acid will adhere to the treatment for their'
                        ' pregnancy'),
        'prob_intervention_delivered_poct': Parameter(
            Types.LIST, 'probability a woman will receive the intervention "point of care Hb testing" given that the '
                        'HSI has ran and the consumables are available (proxy for clinical quality)'),
        'prob_intervention_delivered_syph_test': Parameter(
            Types.LIST, 'probability a woman will receive the intervention "Syphilis test" given that the HSI has ran '
                        'and the consumables are available (proxy for clinical quality)'),
        'prob_intervention_delivered_gdm_test': Parameter(
            Types.LIST, 'probability a woman will receive the intervention "GDM screening" given that the HSI has ran '
                        'and the consumables are available (proxy for clinical quality)'),
        'prob_delivery_modes_ec': Parameter(
            Types.LIST, 'probabilities that a woman admitted with eclampsia will deliver normally, via caesarean or '
                        'via assisted vaginal delivery'),
        'prob_delivery_modes_spe': Parameter(
            Types.LIST, 'probabilities that a woman admitted with severe pre-eclampsia will deliver normally, via '
                        'caesarean or via assisted vaginal delivery'),

        # ASSESSMENT SENSITIVITIES/SPECIFICITIES...
        'sensitivity_bp_monitoring': Parameter(
            Types.LIST, 'sensitivity of blood pressure monitoring to detect hypertension'),
        'specificity_bp_monitoring': Parameter(
            Types.LIST, 'specificity of blood pressure monitoring to detect hypertension'),
        'sensitivity_urine_protein_1_plus': Parameter(
            Types.LIST, 'sensitivity of a urine dipstick test to detect proteinuria at 1+'),
        'specificity_urine_protein_1_plus': Parameter(
            Types.LIST, 'specificity of a urine dipstick test to detect proteinuria at 1+'),
        'sensitivity_poc_hb_test': Parameter(
            Types.LIST, 'sensitivity of a point of care Hb test to detect anaemia'),
        'specificity_poc_hb_test': Parameter(
            Types.LIST, 'specificity of a point of care Hb test to detect anaemia'),
        'sensitivity_fbc_hb_test': Parameter(
            Types.LIST, 'sensitivity of a Full Blood Count test to detect anaemia'),
        'specificity_fbc_hb_test': Parameter(
            Types.LIST, 'specificity of a Full Blood Count test to detect anaemia'),
        'sensitivity_blood_test_glucose': Parameter(
            Types.LIST, 'sensitivity of a blood test to detect raised blood glucose'),
        'specificity_blood_test_glucose': Parameter(
            Types.LIST, 'specificity of a blood test to detect raised blood glucose'),
        'sensitivity_blood_test_syphilis': Parameter(
            Types.LIST, 'sensitivity of a blood test to detect syphilis'),
        'specificity_blood_test_syphilis': Parameter(
            Types.LIST, 'specificity of a blood test to detect syphilis'),

        'squeeze_threshold_for_delay_three_an': Parameter(
            Types.LIST, 'squeeze factor value over which an individual within a antenatal HSI is said to experience '
                        'type 3 delay i.e. delay in receiving appropriate care'),
    }

    PROPERTIES = {
        'ac_total_anc_visits_current_pregnancy': Property(Types.INT, 'rolling total of antenatal visits this woman has '
                                                                     'attended during her pregnancy'),
        'ac_date_next_contact': Property(Types.DATE, 'Date on which this woman is scheduled to return for her next '
                                                     'ANC contact'),
        'ac_to_be_admitted': Property(Types.BOOL, 'Whether this woman requires admission following an ANC visit'),
        'ac_receiving_iron_folic_acid': Property(Types.BOOL, 'whether this woman is receiving daily iron & folic acid '
                                                             'supplementation'),
        'ac_receiving_bep_supplements': Property(Types.BOOL, 'whether this woman is receiving daily balanced energy '
                                                             'and protein supplementation'),
        'ac_receiving_calcium_supplements': Property(Types.BOOL, 'whether this woman is receiving daily calcium '
                                                                 'supplementation'),
        'ac_gest_htn_on_treatment': Property(Types.BOOL, 'Whether this woman has been initiated on treatment for '
                                                         'gestational hypertension'),
        'ac_gest_diab_on_treatment': Property(Types.CATEGORICAL, 'Treatment this woman is receiving for gestational '
                                                                 'diabetes', categories=['none', 'diet_exercise',
                                                                                         'orals', 'insulin']),
        'ac_ectopic_pregnancy_treated': Property(Types.BOOL, 'Whether this woman has received treatment for an ectopic '
                                                             'pregnancy'),
        'ac_received_post_abortion_care': Property(Types.BOOL, 'bitset list of interventions delivered to a woman '
                                                               'undergoing post abortion care'),
        'ac_received_abx_for_prom': Property(Types.BOOL, 'Whether this woman has received antibiotics as treatment for '
                                                         'premature rupture of membranes'),
        'ac_mag_sulph_treatment': Property(Types.BOOL, 'Whether this woman has received magnesium sulphate for '
                                                       'treatment of severe pre-eclampsia/eclampsia'),
        'ac_iv_anti_htn_treatment': Property(Types.BOOL, 'Whether this woman has received intravenous antihypertensive '
                                                         'drugs for treatment of severe hypertension'),
        'ac_admitted_for_immediate_delivery': Property(Types.CATEGORICAL, 'Admission type for women needing urgent '
                                                                          'delivery in the antenatal period',
                                                       categories=['none', 'induction_now', 'induction_future',
                                                                   'caesarean_now', 'caesarean_future', 'avd_now']),
    }

    def read_parameters(self, data_folder):
        parameter_dataframe = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_AntenatalCare.xlsx',
                                            sheet_name='parameter_values')
        self.load_parameters_from_dataframe(parameter_dataframe)

    def initialise_population(self, population):
        df = population.props

        df.loc[df.is_alive, 'ac_total_anc_visits_current_pregnancy'] = 0
        df.loc[df.is_alive, 'ac_date_next_contact'] = pd.NaT
        df.loc[df.is_alive, 'ac_to_be_admitted'] = False
        df.loc[df.is_alive, 'ac_receiving_iron_folic_acid'] = False
        df.loc[df.is_alive, 'ac_receiving_bep_supplements'] = False
        df.loc[df.is_alive, 'ac_receiving_calcium_supplements'] = False
        df.loc[df.is_alive, 'ac_gest_htn_on_treatment'] = False
        df.loc[df.is_alive, 'ac_gest_diab_on_treatment'] = 'none'
        df.loc[df.is_alive, 'ac_ectopic_pregnancy_treated'] = False
        df.loc[df.is_alive, 'ac_received_post_abortion_care'] = False
        df.loc[df.is_alive, 'ac_received_abx_for_prom'] = False
        df.loc[df.is_alive, 'ac_mag_sulph_treatment'] = False
        df.loc[df.is_alive, 'ac_iv_anti_htn_treatment'] = False
        df.loc[df.is_alive, 'ac_admitted_for_immediate_delivery'] = 'none'

    def get_and_store_pregnancy_item_codes(self):
        """
        This function defines the required consumables for each intervention delivered during this module and stores
        them in a module level dictionary called within HSIs
        """
        get_list_of_items = pregnancy_helper_functions.get_list_of_items

        # ---------------------------------- BLOOD TEST EQUIPMENT ---------------------------------------------------
        self.item_codes_preg_consumables['blood_test_equipment'] = \
            get_list_of_items(self, ['Disposables gloves, powder free, 100 pieces per box'])

        # ---------------------------------- IV DRUG ADMIN EQUIPMENT  -------------------------------------------------
        self.item_codes_preg_consumables['iv_drug_equipment'] = \
            get_list_of_items(self, ['Cannula iv  (winged with injection pot) 18_each_CMST',
                                     'Giving set iv administration + needle 15 drops/ml_each_CMST',
                                     'Disposables gloves, powder free, 100 pieces per box'])

        # -------------------------------------------- ECTOPIC PREGNANCY ---------------------------------------------
        self.item_codes_preg_consumables['ectopic_pregnancy_core'] = \
            get_list_of_items(self, ['Halothane (fluothane)_250ml_CMST',
                                     'Scalpel blade size 22 (individually wrapped)_100_CMST'])

        self.item_codes_preg_consumables['ectopic_pregnancy_optional'] = \
            get_list_of_items(self, ['Sodium chloride, injectable solution, 0,9 %, 500 ml',
                                     'Paracetamol, tablet, 500 mg',
                                     'Pethidine, 50 mg/ml, 2 ml ampoule',
                                     'Suture pack',
                                     'Gauze, absorbent 90cm x 40m_each_CMST',
                                     'Cannula iv  (winged with injection pot) 18_each_CMST'])

        # ------------------------------------------- POST ABORTION CARE - GENERAL  -----------------------------------
        self.item_codes_preg_consumables['post_abortion_care_core'] = \
            get_list_of_items(self, ['Misoprostol, tablet, 200 mcg'])

        self.item_codes_preg_consumables['post_abortion_care_optional'] = \
            get_list_of_items(self, ['Complete blood count',
                                     'Blood collecting tube, 5 ml',
                                     'Disposables gloves, powder free, 100 pieces per box',
                                     'Paracetamol, tablet, 500 mg',
                                     'Pethidine, 50 mg/ml, 2 ml ampoule'])

        # ------------------------------------------- POST ABORTION CARE - SEPSIS -------------------------------------
        self.item_codes_preg_consumables['post_abortion_care_sepsis_core'] = \
            get_list_of_items(self, ['Benzylpenicillin 3g (5MU), PFR_each_CMST',
                                     'Gentamycin, injection, 40 mg/ml in 2 ml vial'])
        #  'Metronidazole, injection, 500 mg in 100 ml vial'])

        self.item_codes_preg_consumables['post_abortion_care_sepsis_optional'] = \
            get_list_of_items(self, ['Sodium chloride, injectable solution, 0,9 %, 500 ml',
                                     'Cannula iv  (winged with injection pot) 18_each_CMST',
                                     'Disposables gloves, powder free, 100 pieces per box',
                                     'Giving set iv administration + needle 15 drops/ml_each_CMST',
                                     'Oxygen, 1000 liters, primarily with oxygen cylinders'])

        # ------------------------------------------- POST ABORTION CARE - SHOCK -------------------------------------
        self.item_codes_preg_consumables['post_abortion_care_shock'] = \
            get_list_of_items(self, ['Sodium chloride, injectable solution, 0,9 %, 500 ml',
                                     'Oxygen, 1000 liters, primarily with oxygen cylinders'])

        self.item_codes_preg_consumables['post_abortion_care_shock_optional'] = \
            get_list_of_items(self, ['Cannula iv  (winged with injection pot) 18_each_CMST',
                                     'Disposables gloves, powder free, 100 pieces per box',
                                     'Giving set iv administration + needle 15 drops/ml_each_CMST'])

        # ---------------------------------- URINE DIPSTICK ----------------------------------------------------------
        self.item_codes_preg_consumables['urine_dipstick'] = get_list_of_items(self, ['Urine analysis'])

        # ---------------------------------- IRON AND FOLIC ACID ------------------------------------------------------
        self.item_codes_preg_consumables['iron_folic_acid'] = get_list_of_items(
            self, ['Ferrous Salt + Folic Acid, tablet, 200 + 0.25 mg'])

        # --------------------------------- BALANCED ENERGY AND PROTEIN ----------------------------------------------
        self.item_codes_preg_consumables['balanced_energy_protein'] = get_list_of_items(
            self, ['Dietary supplements (country-specific)'])

        # --------------------------------- INSECTICIDE TREATED NETS ------------------------------------------------
        self.item_codes_preg_consumables['itn'] = get_list_of_items(self, ['Insecticide-treated net'])

        # --------------------------------- CALCIUM SUPPLEMENTS -----------------------------------------------------
        self.item_codes_preg_consumables['calcium'] = get_list_of_items(self, ['Calcium, tablet, 600 mg'])

        # -------------------------------- HAEMOGLOBIN TESTING -------------------------------------------------------
        self.item_codes_preg_consumables['hb_test'] = get_list_of_items(self, ['Haemoglobin test (HB)'])

        # ------------------------------------------- ALBENDAZOLE -----------------------------------------------------
        self.item_codes_preg_consumables['albendazole'] = get_list_of_items(self, ['Albendazole 200mg_1000_CMST'])

        # ------------------------------------------- HEP B TESTING ---------------------------------------------------
        self.item_codes_preg_consumables['hep_b_test'] = get_list_of_items(
            self, ['Hepatitis B test kit-Dertemine_100 tests_CMST'])

        # ------------------------------------------- SYPHILIS TESTING ------------------------------------------------
        self.item_codes_preg_consumables['syphilis_test'] = get_list_of_items(
            self, ['Test, Rapid plasma reagin (RPR)'])

        # ------------------------------------------- SYPHILIS TREATMENT ----------------------------------------------
        self.item_codes_preg_consumables['syphilis_treatment'] = get_list_of_items(
            self, ['Benzathine benzylpenicillin, powder for injection, 2.4 million IU'])

        # ----------------------------------------------- IPTP --------------------------------------------------------
        self.item_codes_preg_consumables['iptp'] = get_list_of_items(
            self, ['Sulfamethoxazole + trimethropin, tablet 400 mg + 80 mg'])

        # ----------------------------------------------- GDM TEST ----------------------------------------------------
        self.item_codes_preg_consumables['gdm_test'] = get_list_of_items(self, ['Blood glucose level test'])

        # ------------------------------------------ FULL BLOOD COUNT -------------------------------------------------
        self.item_codes_preg_consumables['full_blood_count'] = get_list_of_items(self, ['Complete blood count'])

        # ---------------------------------------- BLOOD TRANSFUSION -------------------------------------------------
        self.item_codes_preg_consumables['blood_transfusion'] = get_list_of_items(self, ['Blood, one unit'])

        # --------------------------------------- ORAL ANTIHYPERTENSIVES ---------------------------------------------
        self.item_codes_preg_consumables['oral_antihypertensives'] = get_list_of_items(
            self, ['Methyldopa 250mg_1000_CMST'])

        # -------------------------------------  INTRAVENOUS ANTIHYPERTENSIVES ---------------------------------------
        self.item_codes_preg_consumables['iv_antihypertensives'] = get_list_of_items(
            self, ['Hydralazine, powder for injection, 20 mg ampoule'])

        # ---------------------------------------- MAGNESIUM SULPHATE ------------------------------------------------
        self.item_codes_preg_consumables['magnesium_sulfate'] = get_list_of_items(
            self, ['Magnesium sulfate, injection, 500 mg/ml in 10-ml ampoule'])

        # ---------------------------------------- MANAGEMENT OF ECLAMPSIA --------------------------------------------
        self.item_codes_preg_consumables['eclampsia_management_optional'] = get_list_of_items(
            self, ['Misoprostol, tablet, 200 mcg',
                   'Oxytocin, injection, 10 IU in 1 ml ampoule',
                   'Sodium chloride, injectable solution, 0,9 %, 500 ml',
                   'Cannula iv  (winged with injection pot) 18_each_CMST',
                   'Giving set iv administration + needle 15 drops/ml_each_CMST',
                   'Disposables gloves, powder free, 100 pieces per box',
                   'Oxygen, 1000 liters, primarily with oxygen cylinders',
                   'Complete blood count',
                   'Blood collecting tube, 5 ml',
                   'Foley catheter',
                   'Bag, urine, collecting, 2000 ml'])

        # -------------------------------------- ANTIBIOTICS FOR PROM ------------------------------------------------
        self.item_codes_preg_consumables['abx_for_prom'] = get_list_of_items(
            self, ['Benzathine benzylpenicillin, powder for injection, 2.4 million IU'])

        # ----------------------------------- ORAL DIABETIC MANAGEMENT -----------------------------------------------
        self.item_codes_preg_consumables['oral_diabetic_treatment'] = get_list_of_items(
            self, ['Glibenclamide 5mg_1000_CMST'])

        # ---------------------------------------- INSULIN ----------------------------------------------------------
        self.item_codes_preg_consumables['insulin_treatment'] = get_list_of_items(
            self, ['Insulin soluble 100 IU/ml, 10ml_each_CMST'])

    def initialise_simulation(self, sim):

        # We call the following function to store the required consumables for the simulation run within the appropriate
        # dictionary
        self.get_and_store_pregnancy_item_codes()

        # set up anc counter
        self.anc_counter = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}

        # Schedule logging event
        sim.schedule_event(CareOfWomenDuringPregnancyLoggingEvent(self), sim.date + DateOffset(days=364))

        # For the first period (2010-2015) we use the first value in each list as a parameter
        pregnancy_helper_functions.update_current_parameter_dictionary(self, list_position=0)

        # ==================================== REGISTERING DX_TESTS =================================================
        params = self.current_parameters

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
                property='ps_htn_disorders',
                target_categories=['mild_pre_eclamp', 'severe_pre_eclamp', 'eclampsia'],
                sensitivity=params['sensitivity_urine_protein_1_plus'],
                specificity=params['specificity_urine_protein_1_plus']),


            # This test represents point of care haemoglobin testing used in ANC to detect anaemia (all-severity)
            point_of_care_hb_test=DxTest(
                property='ps_anaemia_in_pregnancy',
                target_categories=['mild', 'moderate', 'severe'],
                sensitivity=params['sensitivity_poc_hb_test'],
                specificity=params['specificity_poc_hb_test']),

            # This test represents laboratory based full blood count testing used in hospitals to determine severity of
            # anaemia via Hb levels
            full_blood_count_hb=DxTest(
                property='ps_anaemia_in_pregnancy',
                target_categories=['mild', 'moderate', 'severe'],
                sensitivity=params['sensitivity_fbc_hb_test'],
                specificity=params['specificity_fbc_hb_test']),

            # This test represents point of care glucose testing used in ANC to detect hyperglycemia, associated with
            # gestational diabetes
            blood_test_glucose=DxTest(
                property='ps_gest_diab',
                target_categories=['uncontrolled'],
                sensitivity=params['sensitivity_blood_test_glucose'],
                specificity=params['specificity_blood_test_glucose']),

            # This test represents point of care testing for syphilis
            blood_test_syphilis=DxTest(
                property='ps_syphilis',
                sensitivity=params['sensitivity_blood_test_syphilis'],
                specificity=params['specificity_blood_test_syphilis']))

        if 'Hiv' not in self.sim.modules:
            logger.debug(key='message', data='HIV module is not registered in this simulation run and therefore HIV '
                                             'testing will not happen in antenatal care')

    def care_of_women_in_pregnancy_property_reset(self, id_or_index):
        """
        This function is called following birth/pregnancy loss to reset the variables stored in this module. This
        prevents women experiencing the effects of these properties in future pregnancies
        :param id_or_index: individual id OR set of indexes to change the properties
        :return:
        """
        df = self.sim.population.props

        df.loc[id_or_index, 'ac_total_anc_visits_current_pregnancy'] = 0
        df.loc[id_or_index, 'ac_to_be_admitted'] = False
        df.loc[id_or_index, 'ac_date_next_contact'] = pd.NaT
        df.loc[id_or_index, 'ac_receiving_iron_folic_acid'] = False
        df.loc[id_or_index, 'ac_receiving_bep_supplements'] = False
        df.loc[id_or_index, 'ac_receiving_calcium_supplements'] = False
        df.loc[id_or_index, 'ac_gest_htn_on_treatment'] = False
        df.loc[id_or_index, 'ac_gest_diab_on_treatment'] = 'none'
        df.loc[id_or_index, 'ac_ectopic_pregnancy_treated'] = False
        df.loc[id_or_index, 'ac_received_post_abortion_care'] = False
        df.loc[id_or_index, 'ac_received_abx_for_prom'] = False
        df.loc[id_or_index, 'ac_mag_sulph_treatment'] = False
        df.loc[id_or_index, 'ac_iv_anti_htn_treatment'] = False
        df.loc[id_or_index, 'ac_admitted_for_immediate_delivery'] = 'none'

    def on_birth(self, mother_id, child_id):
        df = self.sim.population.props

        df.at[child_id, 'ac_total_anc_visits_current_pregnancy'] = 0
        df.at[child_id, 'ac_to_be_admitted'] = False
        df.at[child_id, 'ac_date_next_contact'] = pd.NaT
        df.at[child_id, 'ac_receiving_iron_folic_acid'] = False
        df.at[child_id, 'ac_receiving_bep_supplements'] = False
        df.at[child_id, 'ac_receiving_calcium_supplements'] = False
        df.at[child_id, 'ac_gest_htn_on_treatment'] = False
        df.at[child_id, 'ac_gest_diab_on_treatment'] = 'none'
        df.at[child_id, 'ac_ectopic_pregnancy_treated'] = False
        df.at[child_id, 'ac_received_post_abortion_care'] = False
        df.at[child_id, 'ac_received_abx_for_prom'] = False
        df.at[child_id, 'ac_mag_sulph_treatment'] = False
        df.at[child_id, 'ac_iv_anti_htn_treatment'] = False
        df.at[child_id, 'ac_admitted_for_immediate_delivery'] = 'none'

    def further_on_birth_care_of_women_in_pregnancy(self, mother_id):
        """
        This function is called by the on_birth function of NewbornOutcomes module following a live birth or the Labour
         module following an intrapartum stillbirth . This function contains additional code related to the antenatal
         care module that should be ran following all births/late stillbirths - this is to ensure each modules
         (pregnancy,antenatal care, labour, newborn, postnatal) on_birth code is ran in the correct sequence
        :param mother_id: mothers individual id
        """
        df = self.sim.population.props
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        if df.at[mother_id, 'is_alive']:

            #  run a check at birth to make sure no women exceed 8 visits
            if df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] > 9:
                logger.info(key='error', data=f'Mother {mother_id} attended >8 ANC visits during her pregnancy')

            # We log the total number of ANC contacts a woman has undergone at the time of birth via this dictionary
            if 'ga_anc_one' in mni[mother_id]:
                ga_anc_one = mni[mother_id]['ga_anc_one']
            else:
                ga_anc_one = 0

            total_anc_visit_count = {'person_id': mother_id,
                                     'total_anc': df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'],
                                     'ga_anc_one': ga_anc_one}

            logger.info(key='anc_count_on_birth', data=total_anc_visit_count,
                        description='A dictionary containing the number of ANC visits each woman has on birth')

    def on_hsi_alert(self, person_id, treatment_id):
        logger.debug(key='message', data=f'This is CareOfWomenDuringPregnancy, being alerted about a health system '
                                         f'interaction person {person_id} for: {treatment_id}')

    #  ================================ ADDITIONAL ANTENATAL HELPER FUNCTIONS =========================================
    def get_approx_days_of_pregnancy(self, person_id):
        """
        This function calculates the approximate number of days remaining in a womans pregnancy- assuming all
         pregnancies go to full term (40 weeks gestational age)
        :param person_id: Mothers individual id
        :return: Approximate number of days left in a term pregnancy
        """
        df = self.sim.population.props

        approx_days = (40 - df.at[person_id, 'ps_gestational_age_in_weeks']) * 7

        # Ensure only a positive number of days is returned
        if approx_days <= 1:
            approx_days = 7

        return round(approx_days)

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
        else:
            recommended_gestation_next_anc = 50

        return recommended_gestation_next_anc

    def antenatal_care_scheduler(self, individual_id, visit_to_be_scheduled, recommended_gestation_next_anc):
        """
        This function is responsible for scheduling a womans next ANC contact in the schedule if she chooses to seek
        care again.  It is called by each of the ANC HSIs.
        :param individual_id: individual_id
        :param visit_to_be_scheduled: Number if next visit in the schedule (2-8)
        :param recommended_gestation_next_anc: Gestational age in weeks a woman should be for the next visit in her
        schedule
        """
        df = self.sim.population.props
        params = self.current_parameters

        # Prevent women returning to ANC at very late gestational age
        if df.at[individual_id, 'ps_gestational_age_in_weeks'] >= 42:
            return

        # We check that women will only be scheduled for the next ANC contact in the schedule
        if df.at[individual_id, 'ps_gestational_age_in_weeks'] > recommended_gestation_next_anc:
            logger.info(key='error', data=f'Attempted to schedule an ANC visit for mother {individual_id} at a'
                                          f' gestation lower than her current gestation')
            return

        visit_dict = {2: HSI_CareOfWomenDuringPregnancy_SecondAntenatalCareContact(self, person_id=individual_id),
                      3: HSI_CareOfWomenDuringPregnancy_ThirdAntenatalCareContact(self, person_id=individual_id),
                      4: HSI_CareOfWomenDuringPregnancy_FourthAntenatalCareContact(self, person_id=individual_id),
                      5: HSI_CareOfWomenDuringPregnancy_FifthAntenatalCareContact(self, person_id=individual_id),
                      6: HSI_CareOfWomenDuringPregnancy_SixthAntenatalCareContact(self, person_id=individual_id),
                      7: HSI_CareOfWomenDuringPregnancy_SeventhAntenatalCareContact(self, person_id=individual_id),
                      8: HSI_CareOfWomenDuringPregnancy_EighthAntenatalCareContact(self, person_id=individual_id)}

        if self.sim.modules['PregnancySupervisor'].current_parameters['anc_service_structure'] == 8:
            visit = visit_dict[visit_to_be_scheduled]
        else:
            visit = HSI_CareOfWomenDuringPregnancy_FocusedANCVisit(self, person_id=individual_id,
                                                                   visit_number=visit_to_be_scheduled)

        def calculate_visit_date_and_schedule_visit(visit):
            # We subtract this womans current gestational age from the recommended gestational age for the next
            # contact
            weeks_due_next_visit = int(recommended_gestation_next_anc - df.at[individual_id,
                                                                              'ps_gestational_age_in_weeks'])

            # And use this value as the number of weeks until she is required to return for her next ANC
            visit_date = self.sim.date + DateOffset(weeks=weeks_due_next_visit)
            self.sim.modules['HealthSystem'].schedule_hsi_event(visit,
                                                                priority=0,
                                                                topen=visit_date,
                                                                tclose=visit_date + DateOffset(days=7))

            # We store the date of her next visit and use this date as part of a check when the ANC HSIs run
            df.at[individual_id, 'ac_date_next_contact'] = visit_date

        # If this woman has attended less than 4 visits, and is predicted to attend > 4 (as determined via the
        # PregnancySupervisor module when ANC1 is scheduled) her subsequent ANC appointment is automatically
        # scheduled
        if (visit_to_be_scheduled <= 4) and df.at[individual_id, 'ps_anc4']:
            calculate_visit_date_and_schedule_visit(visit)

        elif ((visit_to_be_scheduled < 4) and not df.at[individual_id, 'ps_anc4']) or (visit_to_be_scheduled > 4):
            if self.rng.random_sample() < params[f'prob_seek_anc{visit_to_be_scheduled}']:
                calculate_visit_date_and_schedule_visit(visit)

    def schedule_admission(self, individual_id):
        """
        This function is called within each of the ANC HSIs for women who require admission due to a complication
        detected during ANC
        :param individual_id: individual_id
        """
        df = self.sim.population.props

        # check correct women have been sent
        if not df.at[individual_id, 'ac_to_be_admitted']:
            logger.info(key='error', data=f'Mother {individual_id} was scheduled for admission despite not requiring'
                                          f' it')
            return

        logger.info(key='anc_interventions', data={'mother': individual_id, 'intervention': 'admission'})

        inpatient = HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare(
            self.sim.modules['CareOfWomenDuringPregnancy'], person_id=individual_id)

        self.sim.modules['HealthSystem'].schedule_hsi_event(inpatient, priority=0,
                                                            topen=self.sim.date,
                                                            tclose=self.sim.date + DateOffset(days=1))

        # Reset the variable to prevent future scheduling errors
        df.at[individual_id, 'ac_to_be_admitted'] = False

    def call_if_maternal_emergency_assessment_cant_run(self, hsi_event):
        """
        This function is called if HSI_CareOfWomenDuringPregnancy_MaternalEmergencyAssessment is unable to run to ensure
         women still experience risk of death associated with the complication they had sought treatment for (as risk of
        death is applied following treatment within the HSI)
        :param hsi_event: HSI event in which the function has been called:
        """
        df = self.sim.population.props
        individual_id = hsi_event.target
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        if df.at[individual_id, 'is_pregnant'] and not df.at[individual_id, 'la_currently_in_labour']:
            logger.debug(key='message', data=f'HSI_CareOfWomenDuringPregnancy_MaternalEmergencyAssessment: did not'
                                             f' run for person {individual_id}')

            self.sim.modules['PregnancySupervisor'].apply_risk_of_death_from_monthly_complications(individual_id)
            if df.at[individual_id, 'is_alive']:
                mni[individual_id]['delay_one_two'] = False
                mni[individual_id]['delay_three'] = False

    # ================================= INTERVENTIONS DELIVERED DURING ANC ============================================
    # The following functions contain the interventions that are delivered as part of routine ANC contacts. Functions
    # are called from within the ANC HSIs. Which interventions are called depends on the mothers gestation and the
    # number of visits she has attended at the time each HSI runs (see ANC HSIs)

    def check_intervention_should_run_and_update_mni(self, person_id, int_1, int2):
        """
        This function is called to check if specific interventions within the ANC matrix should run for an individual.
        If the individual has received the intervention the appropriate amount of times per pregnancy then the
        intervention wont run again
        :param person_id: individual id
        :param int_1: first intervention (i.e. first tetanus vaccine)
        :param int2: second intervention (i.e. second tetanus vaccine)
        :return BOOL (should the intervention be delivered)
        """
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        # If both of the interventions have been delivered, return false to prevent the intervention being delivered
        # again
        if int_1 and int2 in mni[person_id]['anc_ints']:
            return False

        # If the first intervention hasn't already been given, store within the mni, return True so the intervention is
        # delivered
        elif (int_1 not in mni[person_id]['anc_ints']) and (int2 not in mni[person_id]['anc_ints']):
            mni[person_id]['anc_ints'].append(int_1)
            return True

        # If the second intervention hasn't already been given, store within the mni, return True so the intervention is
        # delivered
        elif (int_1 in mni[person_id]['anc_ints']) and int2 not in mni[person_id]['anc_ints']:
            mni[person_id]['anc_ints'].append(int2)
            return True

        else:
            # If no conditions are met return true to prevent interventions not running
            return True

    def screening_interventions_delivered_at_every_contact(self, hsi_event):
        """
        This function contains the screening interventions which are delivered at every ANC contact regardless of the
        womans gestational age and include blood pressure measurement and urine dipstick testing
        :param hsi_event: HSI event in which the function has been called:
        """
        person_id = hsi_event.target
        df = self.sim.population.props
        params = self.current_parameters
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        hypertension_diagnosed = False
        proteinuria_diagnosed = False

        # Delivery of the intervention is conditioned on a random draw against a probability that the intervention
        # would be delivered (used to calibrate to SPA data- acts as proxy for clinical quality)
        if self.rng.random_sample() < params['prob_intervention_delivered_urine_ds']:

            # check consumables
            avail = pregnancy_helper_functions.return_cons_avail(
                self, hsi_event, self.item_codes_preg_consumables, core='urine_dipstick')

            # If the intervention will be delivered the dx_manager runs, returning True if the consumables are
            # available and the test detects protein in the urine
            if avail and self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                         dx_tests_to_run='urine_dipstick_protein', hsi_event=hsi_event):

                # We use a temporary variable to store if proteinuria is detected
                proteinuria_diagnosed = True
                logger.info(key='anc_interventions', data={'mother': person_id, 'intervention': 'dipstick'})

        # The process is repeated for blood pressure monitoring
        if self.rng.random_sample() < params['prob_intervention_delivered_bp']:

            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run='blood_pressure_measurement',
                                                                       hsi_event=hsi_event):
                hypertension_diagnosed = True
                logger.info(key='anc_interventions', data={'mother': person_id, 'intervention': 'bp_measurement'})

                if not df.at[person_id, 'ac_gest_htn_on_treatment'] and\
                    (df.at[person_id, 'ps_htn_disorders'] != 'none') and pd.isnull(mni[person_id]['hypertension'
                                                                                                  '_onset']):

                    # We store date of onset to calculate dalys- only women who are aware of diagnosis experience DALYs
                    # (see daly weight for hypertension)
                    pregnancy_helper_functions.store_dalys_in_mni(person_id, mni, 'hypertension_onset', self.sim.date)

        # If either high blood pressure or proteinuria are detected (or both) we assume this woman needs to be admitted
        # for further treatment following this ANC contact

        # Only women who are not on treatment OR are determined to have severe disease whilst on treatment are admitted
        if hypertension_diagnosed or proteinuria_diagnosed:
            if (((df.at[person_id, 'ps_htn_disorders'] == 'severe_pre_eclamp') and mni[person_id]['new_onset_spe']) or
                (df.at[person_id, 'ps_htn_disorders'] == 'eclampsia') or
               not df.at[person_id, 'ac_gest_htn_on_treatment']):

                df.at[person_id, 'ac_to_be_admitted'] = True

        # Here we conduct screening and initiate treatment for depression as needed
        if 'Depression' in self.sim.modules:
            logger.info(key='anc_interventions', data={'mother': person_id, 'intervention': 'depression_screen'})

            self.sim.modules['Depression'].do_on_presentation_to_care(person_id=person_id,
                                                                      hsi_event=hsi_event)

    def iron_and_folic_acid_supplementation(self, hsi_event):
        """This function contains the intervention iron and folic acid supplementation delivered during ANC.
        :param hsi_event: HSI event in which the function has been called
        """
        df = self.sim.population.props
        person_id = hsi_event.target
        params = self.current_parameters
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        if not df.at[person_id, 'ac_receiving_iron_folic_acid']:

            # check consumable availability
            days = self.get_approx_days_of_pregnancy(person_id)
            avail = pregnancy_helper_functions.return_cons_avail(
                self, hsi_event, self.item_codes_preg_consumables, core='iron_folic_acid', number=days)

            # As with previous interventions - condition on consumables and probability intervention is delivered
            if avail and (self.rng.random_sample() < params['prob_intervention_delivered_ifa']):

                logger.info(key='anc_interventions', data={'mother': person_id, 'intervention': 'iron_folic_acid'})

                # Importantly, only women who will be adherent to iron will experience the benefits of the
                # treatment effect
                if self.rng.random_sample() < params['prob_adherent_ifa']:
                    df.at[person_id, 'ac_receiving_iron_folic_acid'] = True

                    # Women started on IFA at this stage may already be anaemic, we here apply a probability that
                    # starting on a course of IFA will correct anaemia prior to follow up
                    if self.rng.random_sample() < params['effect_of_ifa_for_resolving_anaemia']:

                        # Store date of resolution for daly calculations
                        pregnancy_helper_functions.store_dalys_in_mni(
                            person_id, mni, f'{df.at[person_id, "ps_anaemia_in_pregnancy"]}_anaemia_resolution',
                            self.sim.date)

                        df.at[person_id, 'ps_anaemia_in_pregnancy'] = 'none'

    def balance_energy_and_protein_supplementation(self, hsi_event):
        """This function contains the intervention balance energy and protein supplementation delivered during ANC.
        :param hsi_event: HSI event in which the function has been called
        """
        df = self.sim.population.props
        person_id = hsi_event.target

        # Check the woman is not already receiving the supplements
        if not df.at[person_id, 'ac_receiving_bep_supplements']:

            # If the consumables are available...
            days = self.get_approx_days_of_pregnancy(person_id)
            avail = pregnancy_helper_functions.return_cons_avail(
                self, hsi_event, self.item_codes_preg_consumables, core='balanced_energy_protein', number=days)

            # And she is deemed to be at risk (i.e. BMI < 18) she is started on supplements
            if avail and (df.at[person_id, 'li_bmi'] == 1):
                df.at[person_id, 'ac_receiving_bep_supplements'] = True
                logger.info(key='anc_interventions', data={'mother': person_id, 'intervention': 'b_e_p'})

    def insecticide_treated_bed_net(self, hsi_event):
        """This function simply logs a consumable request for insecticide treated bed nets. Coverage of ITN and its
        effect is managed through the malaria module's calculation of malaria incidence.
        :param hsi_event: HSI event in which the function has been called
        """

        hsi_event.get_consumables(item_codes=self.item_codes_preg_consumables['itn'])

    def tb_screening(self, hsi_event):
        """
        This function schedules HSI_TbScreening which represents screening for TB. Screening is only scheduled if
        if the TB module is registered.
        :param hsi_event: HSI event in which the function has been called
        """

        # Currently we schedule women to the TB screening HSI in the TB module
        if 'Tb' in self.sim.modules:
            tb_screen = HSI_Tb_ScreeningAndRefer(
                module=self.sim.modules['Tb'], person_id=hsi_event.target)

            self.sim.modules['HealthSystem'].schedule_hsi_event(tb_screen, priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

    def tetanus_vaccination(self, hsi_event):
        """
        This function contains the intervention tetanus vaccination. A booster dose of the vaccine is given to all women
         during ANC. Effect of vaccination is managed by the EPI module and therefore here we just capture consumables
         and number of doses
        :param hsi_event: HSI event in which the function has been called
        """
        person_id = hsi_event.target
        df = self.sim.population.props

        if 'Epi' in self.sim.modules:

            # Define the HSI in which the vaccine is delivered
            vaccine_hsi = HSI_TdVaccine(self.sim.modules['Epi'], person_id=person_id)

            # Identify individuals district of residence in order to determine district level coverage of TT
            ind_district = df.at[person_id, 'district_num_of_residence']
            vaccine_coverage_df = self.sim.modules['Epi'].parameters['district_vaccine_coverage']

            # If the year is 2010-2018 we condition the HSI being scheduled on the district level coverage
            if self.sim.date.year <= 2018:
                coverage_year = self.sim.date.year

                tt2_coverage = vaccine_coverage_df.loc[(vaccine_coverage_df['District'] == ind_district) &
                                                       (vaccine_coverage_df['Year'] == coverage_year)]['TT2+']

                if self.rng.random_sample() < tt2_coverage.values:

                    self.sim.modules['HealthSystem'].schedule_hsi_event(vaccine_hsi, priority=0,
                                                                        topen=self.sim.date)
            else:
                # After 2018 all women are scheduled the HSI and consumable availability will determine intervention
                # delivery
                self.sim.modules['HealthSystem'].schedule_hsi_event(vaccine_hsi, priority=0,
                                                                    topen=self.sim.date)

    def calcium_supplementation(self, hsi_event):
        """This function contains the intervention calcium supplementation delivered during ANC.
        :param hsi_event: HSI event in which the function has been called
        """
        df = self.sim.population.props
        person_id = hsi_event.target

        # If the woman is not already receiving supplements AND has been designated as high risk for pre-eclampsia
        # (as defined by ANC guidelines) then she will receive the interventions, conditional on consumables
        if not df.at[person_id, 'ac_receiving_calcium_supplements'] and ((df.at[person_id, 'la_parity'] == 0)
                                                                         or (df.at[person_id, 'la_parity'] > 4)):

            days = self.get_approx_days_of_pregnancy(person_id) * 3
            avail = pregnancy_helper_functions.return_cons_avail(
                self, hsi_event, self.item_codes_preg_consumables, core='calcium', number=days)

            if avail:
                df.at[person_id, 'ac_receiving_calcium_supplements'] = True
                logger.info(key='anc_interventions', data={'mother': person_id, 'intervention': 'calcium'})

    def point_of_care_hb_testing(self, hsi_event):
        """
        This function contains the intervention point of care haemoglobin testing provided to women during ANC1/ANC6
        to detect anaemia during pregnancy
        :param hsi_event: HSI event in which the function has been called
        """
        person_id = hsi_event.target
        df = self.sim.population.props
        params = self.current_parameters

        # If this woman has already had her Hb checked twice during pregnancy she will not receive another Hb test
        if not self.check_intervention_should_run_and_update_mni(person_id, 'hb_1', 'hb_2'):
            return

        # Run check against probability of testing being delivered
        if self.rng.random_sample() < params['prob_intervention_delivered_poct']:
            logger.info(key='anc_interventions', data={'mother': person_id, 'intervention': 'hb_screen'})

            hsi_event.get_consumables(item_codes=self.item_codes_preg_consumables['blood_test_equipment'])

            # We run the test through the dx_manager and if a woman has anaemia and its detected she will be admitted
            # for further care
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run='point_of_care_hb_test',
                                                                       hsi_event=hsi_event):
                df.at[person_id, 'ac_to_be_admitted'] = True

    def albendazole_administration(self, hsi_event):
        """
        This function contains the intervention albendazole administration (de-worming) and is provided to women during
         ANC
        :param hsi_event: HSI event in which the function has been called
        """
        person_id = hsi_event.target
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        # If this woman has already had deworming the intervention is not delivered again
        if 'albend' in mni[person_id]['anc_ints']:
            return

        mni[person_id]['anc_ints'].append('albend')

        # We run this function to store the associated consumables with albendazole administration. This
        # intervention has no effect in the model due to limited evidence

        # If the consumables are available and the HCW will provide the tablets, the intervention is given
        avail = hsi_event.get_consumables(item_codes=self.item_codes_preg_consumables['albendazole'])

        if avail:
            logger.info(key='anc_interventions', data={'mother': person_id, 'intervention': 'albendazole'})

    def hep_b_testing(self, hsi_event):
        """
        This function contains the intervention Hepatitis B testing and is provided to women during ANC. As Hepatitis
        B is not modelled currently this intervention just maps consumables used during ANC
        :param hsi_event: HSI event in which the function has been called
        """
        person_id = hsi_event.target
        cons = self.item_codes_preg_consumables

        # If this woman has already been tested for hep b twice in her pregnancy the intervention will not run
        if not self.check_intervention_should_run_and_update_mni(person_id, 'hep_b_1', 'hep_b_2'):
            return

        # This intervention is a place holder prior to the Hepatitis B module being coded
        # Define the consumables
        avail = hsi_event.get_consumables(item_codes=cons['hep_b_test'],
                                          optional_item_codes=cons['blood_test_equipment'])

        # We log all the consumables required above but we only condition the event test happening on the
        # availability of the test itself
        if avail:
            logger.info(key='anc_interventions', data={'mother': person_id, 'intervention': 'hep_b'})

    def syphilis_screening_and_treatment(self, hsi_event):
        """
        This function contains the intervention Syphilis testing and is provided to women during ANC. As Syphilis is
        not modelled currently this intervention just maps consumables used during ANC
        :param hsi_event: HSI event in which the function has been called
        """
        params = self.current_parameters
        person_id = hsi_event.target
        df = self.sim.population.props

        # If this woman has already been screened twice for syphilis then the intervention will not run
        if not self.check_intervention_should_run_and_update_mni(person_id, 'syph_1', 'syph_2'):
            return

        # See if she will receive testing
        if self.rng.random_sample() < params['prob_intervention_delivered_syph_test']:
            logger.info(key='anc_interventions', data={'mother': person_id, 'intervention': 'syphilis_test'})

            avail = pregnancy_helper_functions.return_cons_avail(
                self, hsi_event, self.item_codes_preg_consumables, core='syphilis_test',
                optional='blood_test_equipment')

            test = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                         dx_tests_to_run='blood_test_syphilis', hsi_event=hsi_event)

            # If the testing occurs and detects syphilis she will get treatment (if consumables are available)
            if avail and test:

                avail = pregnancy_helper_functions.return_cons_avail(
                    self, hsi_event, self.item_codes_preg_consumables, core='syphilis_treatment',
                    optional='blood_test_equipment')

                if avail:
                    # We assume that treatment is 100% effective at curing infection
                    df.at[person_id, 'ps_syphilis'] = False
                    logger.info(key='anc_interventions', data={'mother': person_id, 'intervention': 'syphilis_treat'})

    def hiv_testing(self, hsi_event):
        """
        This function contains the scheduling for HIV testing and is provided to women during ANC.
        :param hsi_event: HSI event in which the function has been called
        """
        person_id = hsi_event.target
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        # If she has already been tested for HIV she will not be tested again during ANC
        if 'hiv' in mni[person_id]['anc_ints']:
            return

        if 'Hiv' in self.sim.modules:
            # The decision of whether the woman gets a test is determined by the Hiv module
            test_scheduled = self.sim.modules['Hiv'].decide_whether_hiv_test_for_mother(
                person_id, referred_from="care_of_women_during_pregnancy")

            if test_scheduled:
                mni[person_id]['anc_ints'].append('hiv')

            logger.info(key='anc_interventions', data={'mother': person_id, 'intervention': 'hiv_screen'})

    def iptp_administration(self, hsi_event):
        """
        This function schedules HSI_MalariaIPTp for women who should receive IPTp during pregnancy (if the malaria
        module is registered)
        :param hsi_event: HSI event in which the function has been called
        """
        person_id = hsi_event.target

        # If the Malaria module is registered women are scheduled to receive IPTp via this HSI event
        if 'Malaria' in self.sim.modules:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_MalariaIPTp(person_id=person_id,
                                module=self.sim.modules['Malaria']), topen=self.sim.date, tclose=None, priority=0)

    def gdm_screening(self, hsi_event):
        """This function contains intervention of gestational diabetes screening during ANC. Screening is only conducted
         on women with pre-specified risk factors for the disease.
        :param hsi_event: HSI event in which the function has been called
        """
        df = self.sim.population.props
        params = self.current_parameters
        person_id = hsi_event.target
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        # Women already screened will not be screened again
        if 'gdm_screen' in mni[person_id]['anc_ints']:
            return

        # We check if this women has any of the key risk factors, if so they are sent for additional blood tests
        if df.at[person_id, 'li_bmi'] >= 4 or df.at[person_id, 'ps_prev_gest_diab'] or df.at[person_id,
                                                                                             'ps_prev_stillbirth']:

            # If they are available, the test is conducted
            if self.rng.random_sample() < params['prob_intervention_delivered_gdm_test']:

                avail = pregnancy_helper_functions.return_cons_avail(
                    self, hsi_event, self.item_codes_preg_consumables, core='gdm_test', optional='iv_drug_equipment')

                # If the test accurately detects a woman has gestational diabetes the consumables are recorded and
                # she is referred for treatment
                if avail and self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                             dx_tests_to_run='blood_test_glucose', hsi_event=hsi_event):

                    logger.info(key='anc_interventions', data={'mother': person_id, 'intervention': 'gdm_screen'})
                    mni[person_id]['anc_ints'].append('gdm_screen')

                    # We assume women with a positive GDM screen will be admitted (if they are not already receiving
                    # outpatient care)
                    if df.at[person_id, 'ac_gest_diab_on_treatment'] == 'none':

                        # Store onset after diagnosis as daly weight is tied to diagnosis
                        pregnancy_helper_functions.store_dalys_in_mni(person_id, mni, 'gest_diab_onset',
                                                                      self.sim.date)
                        df.at[person_id, 'ac_to_be_admitted'] = True

    def interventions_delivered_each_visit_from_anc2(self, hsi_event):
        """This function contains a collection of interventions that are delivered to women every time they attend ANC
        from ANC visit 2
        :param hsi_event: HSI event in which the function has been called
        """
        self.screening_interventions_delivered_at_every_contact(hsi_event=hsi_event)
        self.iron_and_folic_acid_supplementation(hsi_event=hsi_event)
        self.balance_energy_and_protein_supplementation(hsi_event=hsi_event)
        self.calcium_supplementation(hsi_event=hsi_event)

    def check_anc1_can_run(self, individual_id, squeeze_factor, gest_age_next_contact):
        """
        This function is called by the first ANC contact and runs a series of checks to determine if the HSI should run
        on the date it has been scheduled for
        :param individual_id: individual id
        :param squeeze_factor: squeeze_factor of the HSI calling this function
        :param gest_age_next_contact: gestational age, in weeks, this woman is due to return for her next ANC
        :returns True/False as to whether the event can run
        """
        df = self.sim.population.props
        params = self.current_parameters
        mother = df.loc[individual_id]

        visit = HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareContact(
            self.sim.modules['CareOfWomenDuringPregnancy'],
            person_id=individual_id)

        # Calculate the difference between the current date and when anc1 was scheduled for this woman at the start of
        # pregnancy
        date_difference = self.sim.date - df.at[individual_id, 'ps_date_of_anc1']

        # Only women who are alive, still pregnant and not in labour can attend ANC1
        if not mother.is_alive or not mother.is_pregnant or mother.la_currently_in_labour:
            return False

        # Here we block the event from running for previously scheduled ANC1 HSIs for women who have lost a pregnancy
        # and become pregnant again
        if (
            (date_difference > pd.to_timedelta(7, unit='D')) or
            (df.at[individual_id, 'ac_total_anc_visits_current_pregnancy'] > 0) or
            (df.at[individual_id, 'ps_gestational_age_in_weeks'] < 7)
             ):
            return False

        # If the woman is an inpatient when ANC1 is scheduled, she will try and return at the next appropriate
        # gestational age
        if df.at[individual_id, 'hs_is_inpatient']:

            # We assume that she will return for her first appointment at the next gestation in the schedule
            logger.debug(key='message', data=f'mother {individual_id} is scheduled to attend ANC today but is '
                                             f'currently an inpatient- she will be scheduled to arrive at her next '
                                             f'visit instead no interventions will be delivered here')

            weeks_due_next_visit = int(gest_age_next_contact - df.at[individual_id, 'ps_gestational_age_in_weeks'])
            visit_date = self.sim.date + DateOffset(weeks=weeks_due_next_visit)

            self.sim.modules['HealthSystem'].schedule_hsi_event(visit, priority=0,
                                                                topen=visit_date,
                                                                tclose=visit_date + DateOffset(days=7))
            df.at[individual_id, 'ps_date_of_anc1'] = visit_date
            return False

        # Finally, if the squeeze factor is too high the event wont run and she will return tomorrow
        if squeeze_factor > params['squeeze_factor_threshold_anc']:

            self.sim.modules['HealthSystem'].schedule_hsi_event(visit, priority=0,
                                                                topen=self.sim.date + DateOffset(days=1),
                                                                tclose=self.sim.date + DateOffset(days=2))
            return False

        return True

    def check_subsequent_anc_can_run(self, individual_id, this_contact, this_visit_number, squeeze_factor,
                                     gest_age_next_contact):
        """
        This function is called by the subsequent ANC contacts and runs a series of checks to determine if the HSI
        should run on the date it has been scheduled for
        :param individual_id: individual id
        :param this_contact: HSI object of the current ANC contact that needs to be rebooked
        :param this_visit_number: Number of the next ANC contact in the schedule
        :param squeeze_factor: squeeze_factor of the HSI calling this function
        :param gest_age_next_contact: gestational age, in weeks, this woman is due to return for her next ANC
        :returns True/False as to whether the event can run
        """

        df = self.sim.population.props
        params = self.current_parameters

        date_difference = self.sim.date - df.at[individual_id, 'ac_date_next_contact']

        ga_for_anc_dict = {2: 20, 3: 26, 4: 30, 5: 34, 6: 36, 7: 38, 8: 40}

        # If women have died, are no longer pregnant, are in labour, are postnatal, are pregnant but with a gestational
        # age lower than required for this anc visit or are 'late' to attend this visit (usually for visits scheduled in
        # one pregnancy but running in a subsequent one) it will not run
        if (not df.at[individual_id, 'is_alive'] or
            not df.at[individual_id, 'is_pregnant'] or
            df.at[individual_id, 'la_currently_in_labour'] or
            df.at[individual_id, 'la_is_postpartum'] or
            (df.at[individual_id, 'ps_gestational_age_in_weeks'] < ga_for_anc_dict[this_visit_number]) or
            (date_difference > pd.to_timedelta(7, unit='D') or
             not df.at[individual_id, 'ac_total_anc_visits_current_pregnancy'] == (this_visit_number - 1))):
            return False

        # If the woman is currently an inpatient then she will return at the next point in the contact schedule but
        # receive the care she has missed in this visit
        if df.at[individual_id, 'hs_is_inpatient']:
            self.antenatal_care_scheduler(individual_id, visit_to_be_scheduled=this_visit_number,
                                          recommended_gestation_next_anc=gest_age_next_contact)
            return False

        # If the squeeze factor is too high she will return tomorrow
        if squeeze_factor > params['squeeze_factor_threshold_anc']:
            self.sim.modules['HealthSystem'].schedule_hsi_event(this_contact, priority=0,
                                                                topen=self.sim.date + DateOffset(days=1),
                                                                tclose=self.sim.date + DateOffset(days=2))
            return False

        return True

    # =============================== INTERVENTIONS DELIVERED DURING INPATIENT CARE ===================================
    # The following functions contain code for the interventions which are called by antenatal HSIs (not including
    # routine ANC) this includes post abortion/ectopic care and antenatal inpatient care

    def full_blood_count_testing(self, hsi_event):
        """This function contains the intervention 'full blood count testing' and represents blood testing requiring a
        laboratory. It is called by HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare for women admitted due to
        anaemia
        :param hsi_event: HSI event in which the function has been called
        :returns: result of the FBC ['none', 'mild_mod', 'severe'] (STR)
        """
        df = self.sim.population.props
        person_id = hsi_event.target

        # Run dx_test for anaemia...
        # If a woman is not truly anaemic but the FBC returns a result of anaemia, due to tests specificity, we
        # assume the reported anaemia is mild
        hsi_event.get_consumables(item_codes=self.item_codes_preg_consumables['blood_test_equipment'])

        test_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                dx_tests_to_run='full_blood_count_hb', hsi_event=hsi_event)

        if test_result and (df.at[person_id, 'ps_anaemia_in_pregnancy'] == 'none'):
            return 'non_severe'

        # If the test correctly identifies a woman's anaemia we assume it correctly identifies its severity
        if test_result and (df.at[person_id, 'ps_anaemia_in_pregnancy'] != 'none'):
            return df.at[person_id, 'ps_anaemia_in_pregnancy']

        # We return a none value if no anaemia was detected
        return 'none'

    def antenatal_blood_transfusion(self, individual_id, hsi_event):
        """
        This function contains the intervention 'blood transfusion'. It is called by either
        HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare or HSI_CareOfWomenDuringPregnancy_PostAbortionCase
        Management for women requiring blood for either haemorrhage or severe anaemia.
        given iron and folic acid supplements during ANC
        :param individual_id: individual_id
        :param hsi_event: HSI event in which the function has been called
        """
        df = self.sim.population.props
        params = self.current_parameters
        store_dalys_in_mni = pregnancy_helper_functions.store_dalys_in_mni
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        # Check for consumables
        avail = pregnancy_helper_functions.return_cons_avail(
            self, hsi_event, self.item_codes_preg_consumables, core='blood_transfusion', optional='iv_drug_equipment')

        sf_check = pregnancy_helper_functions.check_emonc_signal_function_will_run(self.sim.modules['Labour'],
                                                                                   sf='blood_tran',
                                                                                   hsi_event=hsi_event)

        # If the blood is available we assume the intervention can be delivered
        if avail and sf_check:
            pregnancy_helper_functions.log_met_need(self, 'blood_tran', hsi_event)

            # If the woman is receiving blood due to anaemia we apply a probability that a transfusion of 2 units
            # RBCs will correct this woman's severe anaemia
            if params['treatment_effect_blood_transfusion_anaemia'] > self.rng.random_sample():
                store_dalys_in_mni(individual_id, mni, 'severe_anaemia_resolution', self.sim.date)
                df.at[individual_id, 'ps_anaemia_in_pregnancy'] = 'none'

    def initiate_maintenance_anti_hypertensive_treatment(self, individual_id, hsi_event):
        """
        This function contains initiation of oral antihypertensive medication for women with high blood pressure. It is
        called by HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare for women who have been identified as having
         high blood pressure in pregnancy but are not yet receiving treatment
        :param individual_id: individual_id
        :param hsi_event: HSI event in which the function has been called
        """
        df = self.sim.population.props

        # Calculate the approximate dose for the remainder of pregnancy and check availability
        dose = self.get_approx_days_of_pregnancy(individual_id) * 4
        cons = {_i: dose for _i in self.item_codes_preg_consumables['oral_antihypertensives']}
        avail = hsi_event.get_consumables(item_codes=cons)

        # If the consumables are available then the woman is started on treatment
        if avail:
            df.at[individual_id, 'ac_gest_htn_on_treatment'] = True

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
        cons = self.item_codes_preg_consumables

        # Define the consumables and check their availability
        avail = hsi_event.get_consumables(item_codes=cons['iv_antihypertensives'],
                                          optional_item_codes=cons['iv_drug_equipment'])

        # If they are available then the woman is started on treatment
        if avail:
            pregnancy_helper_functions.log_met_need(self, 'iv_htns', hsi_event)

            # We assume women treated with antihypertensives would no longer be severely hypertensive- meaning they
            # are not at risk of death from severe gestational hypertension in the PregnancySupervisor event
            if df.at[individual_id, 'ps_htn_disorders'] == 'severe_gest_htn':
                df.at[individual_id, 'ps_htn_disorders'] = 'gest_htn'

            # We dont assume antihypertensives convert severe pre-eclampsia/eclampsia to a more mild version of the
            # disease (as the disease is multi-system and hypertension is only one contributing factor to mortality) but
            # instead use this property to reduce risk of acute death from this episode of disease
            if (df.at[individual_id, 'ps_htn_disorders'] == 'severe_pre_eclamp') or (df.at[individual_id,
                                                                                           'ps_htn_disorders'] ==
                                                                                     'eclampsia'):
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

        avail = pregnancy_helper_functions.return_cons_avail(
            self, hsi_event, self.item_codes_preg_consumables, core='magnesium_sulfate',
            optional='eclampsia_management_optional')

        # check HCW will deliver intervention
        sf_check = pregnancy_helper_functions.check_emonc_signal_function_will_run(self.sim.modules['Labour'],
                                                                                   sf='anticonvulsant',
                                                                                   hsi_event=hsi_event)

        # If available deliver the treatment
        if avail and sf_check:
            df.at[individual_id, 'ac_mag_sulph_treatment'] = True
            pregnancy_helper_functions.log_met_need(self, 'mag_sulph', hsi_event)

    def antibiotics_for_prom(self, individual_id, hsi_event):
        """
        This function contains initiation of antibiotics for women with who have been admitted following premature
        rupture of membranes .It is called by HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare
        :param individual_id: individual_id
        :param hsi_event: HSI event in which the function has been called
        """
        df = self.sim.population.props
        cons = self.item_codes_preg_consumables

        # check consumables and whether HCW are available to deliver the intervention
        avail = hsi_event.get_consumables(item_codes=cons['abx_for_prom'],
                                          optional_item_codes=cons['iv_drug_equipment'])

        sf_check = pregnancy_helper_functions.check_emonc_signal_function_will_run(self.sim.modules['Labour'],
                                                                                   sf='iv_abx',
                                                                                   hsi_event=hsi_event)

        if avail and sf_check:
            df.at[individual_id, 'ac_received_abx_for_prom'] = True

    def ectopic_pregnancy_treatment_doesnt_run(self, hsi_event):
        """
        This function is called within HSI_CareOfWomenDuringPregnancy_TreatmentForEctopicPregnancy if the event cannot
        run/the intervention cannot be delivered. This ensures that women with ectopic pregnancies that haven't ruptured
        will experience rupture and risk of death without treatment
        :param hsi_event: HSI event in which the function has been called
        """
        individual_id = hsi_event.target
        df = self.sim.population.props

        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_TreatmentForEctopicPregnancy: did not run')

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
            (mother.ps_premature_rupture_of_membranes and mother.ps_chorioamnionitis) or \
            (mother.ps_premature_rupture_of_membranes and not mother.ps_chorioamnionitis and
             (mother.ps_gestational_age_in_weeks >= 34)):
            beddays = 1

        # Otherwise women will remain as an inpatient until their gestation is greater, to improve newborn outcomes
        elif (mother.ps_placenta_praevia and (mother.ps_antepartum_haemorrhage == 'mild_moderate') and
              (mother.ps_gestational_age_in_weeks < 37)) or (mother.ps_premature_rupture_of_membranes and
                                                             not mother.ps_chorioamnionitis and
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

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'AntenatalCare_Outpatient'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'AntenatalFirst': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        # Calculate when this woman should return for her next visit
        gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)

        # Check this visit can run
        can_anc1_run = self.module.check_anc1_can_run(person_id, squeeze_factor, gest_age_next_contact)

        if can_anc1_run:
            self.module.anc_counter[1] += 1

            # store GA at first visit
            mni[person_id]['ga_anc_one'] = df.at[person_id, 'ps_gestational_age_in_weeks']

            # We add a visit to a rolling total of ANC visits in this pregnancy
            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            #  =================================== INTERVENTIONS ====================================================
            # First all women, regardless of ANC contact or gestation, undergo urine and blood pressure measurement
            # and depression screening
            self.module.screening_interventions_delivered_at_every_contact(hsi_event=self)

            # Next, all women attending their first ANC receive the following interventions, regardless of gestational
            # age at presentation
            self.module.iron_and_folic_acid_supplementation(hsi_event=self)
            self.module.balance_energy_and_protein_supplementation(hsi_event=self)
            self.module.insecticide_treated_bed_net(hsi_event=self)
            self.module.tb_screening(hsi_event=self)
            self.module.hiv_testing(hsi_event=self)
            self.module.hep_b_testing(hsi_event=self)
            self.module.syphilis_screening_and_treatment(hsi_event=self)
            self.module.point_of_care_hb_testing(hsi_event=self)
            self.module.tetanus_vaccination(hsi_event=self)

            # If the woman presents after 20 weeks she is provided interventions she has missed by presenting late
            if mother.ps_gestational_age_in_weeks > 19:
                self.module.point_of_care_hb_testing(hsi_event=self)
                self.module.albendazole_administration(hsi_event=self)
                self.module.iptp_administration(hsi_event=self)
                self.module.calcium_supplementation(hsi_event=self)

            # Any women presenting for ANC1 after 26 week are also required to have a GDM screen
            if mother.ps_gestational_age_in_weeks >= 26:
                self.module.gdm_screening(hsi_event=self)

            # Then we determine if this woman will return for her next ANC visit
            if mother.ps_gestational_age_in_weeks < 40:
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=2,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

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

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'AntenatalCare_Outpatient'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]

        # Here we define variables used within the function that checks in this ANC visit can run
        gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)
        this_contact = HSI_CareOfWomenDuringPregnancy_SecondAntenatalCareContact(
            self.module, person_id=person_id)

        # Run the check
        can_anc_run = self.module.check_subsequent_anc_can_run(individual_id=person_id, this_contact=this_contact,
                                                               this_visit_number=2, squeeze_factor=squeeze_factor,
                                                               gest_age_next_contact=gest_age_next_contact)

        if can_anc_run:
            self.module.anc_counter[2] += 1
            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            #  =================================== INTERVENTIONS ====================================================
            # First we administer the administer the interventions all women will receive at this contact regardless of
            # gestational age
            self.module.interventions_delivered_each_visit_from_anc2(hsi_event=self)
            self.module.tetanus_vaccination(hsi_event=self)

            # And we schedule the next ANC appointment
            if mother.ps_gestational_age_in_weeks < 40:
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=3,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

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
                self.module.syphilis_screening_and_treatment(hsi_event=self)

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

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'AntenatalCare_Outpatient'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]

        # Here we define variables used within the function that checks in this ANC visit can run
        gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)
        this_contact = HSI_CareOfWomenDuringPregnancy_ThirdAntenatalCareContact(self.module, person_id=person_id)

        # Run the check
        can_anc_run = self.module.check_subsequent_anc_can_run(person_id, this_contact, 3, squeeze_factor,
                                                               gest_age_next_contact)

        if can_anc_run:
            self.module.anc_counter[3] += 1
            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            #  =================================== INTERVENTIONS ====================================================
            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)
            self.module.interventions_delivered_each_visit_from_anc2(hsi_event=self)

            if mother.ps_gestational_age_in_weeks < 40:
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=4,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            if mother.ps_gestational_age_in_weeks < 30:
                self.module.iptp_administration(hsi_event=self)
                self.module.gdm_screening(hsi_event=self)

            elif mother.ps_gestational_age_in_weeks < 34:
                self.module.iptp_administration(hsi_event=self)

            elif mother.ps_gestational_age_in_weeks < 36:
                self.module.iptp_administration(hsi_event=self)
                self.module.hep_b_testing(hsi_event=self)
                self.module.syphilis_screening_and_treatment(hsi_event=self)

            elif mother.ps_gestational_age_in_weeks < 38:
                self.module.point_of_care_hb_testing(hsi_event=self)

            elif mother.ps_gestational_age_in_weeks < 40:
                self.module.iptp_administration(hsi_event=self)

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

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'AntenatalCare_Outpatient'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]

        # Here we define variables used within the function that checks in this ANC visit can run
        gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)
        this_contact = HSI_CareOfWomenDuringPregnancy_FourthAntenatalCareContact(self.module, person_id=person_id)

        # Run the check
        can_anc_run = self.module.check_subsequent_anc_can_run(person_id, this_contact, 4, squeeze_factor,
                                                               gest_age_next_contact)

        if can_anc_run:
            self.module.anc_counter[4] += 1
            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            #  =================================== INTERVENTIONS ====================================================
            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)
            self.module.interventions_delivered_each_visit_from_anc2(hsi_event=self)

            if mother.ps_gestational_age_in_weeks < 40:
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=5,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            if mother.ps_gestational_age_in_weeks < 34:
                self.module.iptp_administration(hsi_event=self)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] < 36:
                self.module.iptp_administration(hsi_event=self)
                self.module.hep_b_testing(hsi_event=self)
                self.module.syphilis_screening_and_treatment(hsi_event=self)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] < 38:
                self.module.point_of_care_hb_testing(hsi_event=self)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] < 40:
                self.module.iptp_administration(hsi_event=self)

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

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'AntenatalCare_Outpatient'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]

        # Here we define variables used within the function that checks in this ANC visit can run
        gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)
        this_contact = HSI_CareOfWomenDuringPregnancy_FifthAntenatalCareContact(self.module, person_id=person_id)

        # Run the check
        can_anc_run = self.module.check_subsequent_anc_can_run(person_id, this_contact, 5, squeeze_factor,
                                                               gest_age_next_contact)

        if can_anc_run:
            self.module.anc_counter[5] += 1
            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            #  =================================== INTERVENTIONS ====================================================
            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)
            self.module.interventions_delivered_each_visit_from_anc2(hsi_event=self)

            if mother.ps_gestational_age_in_weeks < 40:
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=6,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            if mother.ps_gestational_age_in_weeks < 36:
                self.module.iptp_administration(hsi_event=self)
                self.module.hep_b_testing(hsi_event=self)
                self.module.syphilis_screening_and_treatment(hsi_event=self)

            elif mother.ps_gestational_age_in_weeks < 38:
                self.module.point_of_care_hb_testing(hsi_event=self)

            elif mother.ps_gestational_age_in_weeks < 40:
                self.module.iptp_administration(hsi_event=self)

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

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'AntenatalCare_Outpatient'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]

        # Here we define variables used within the function that checks in this ANC visit can run
        gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)
        this_contact = HSI_CareOfWomenDuringPregnancy_SixthAntenatalCareContact(self.module, person_id=person_id)

        # Run the check
        can_anc_run = self.module.check_subsequent_anc_can_run(person_id, this_contact, 6, squeeze_factor,
                                                               gest_age_next_contact)

        if can_anc_run:
            self.module.anc_counter[6] += 1
            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)

            #  =================================== INTERVENTIONS ====================================================
            self.module.interventions_delivered_each_visit_from_anc2(hsi_event=self)

            if mother.ps_gestational_age_in_weeks < 40:
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=7,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            if mother.ps_gestational_age_in_weeks < 38:
                self.module.point_of_care_hb_testing(hsi_event=self)

            elif mother.ps_gestational_age_in_weeks < 40:
                self.module.iptp_administration(hsi_event=self)

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

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'AntenatalCare_Outpatient'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]

        # Here we define variables used within the function that checks in this ANC visit can run
        gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)
        this_contact = HSI_CareOfWomenDuringPregnancy_SeventhAntenatalCareContact(self.module, person_id=person_id)

        # Run the check
        can_anc_run = self.module.check_subsequent_anc_can_run(person_id, this_contact, 7, squeeze_factor,
                                                               gest_age_next_contact)

        if can_anc_run:
            self.module.anc_counter[7] += 1
            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            #  =================================== INTERVENTIONS ====================================================
            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)
            self.module.interventions_delivered_each_visit_from_anc2(hsi_event=self)

            if mother.ps_gestational_age_in_weeks < 40:
                self.module.iptp_administration(hsi_event=self)
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=8,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

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

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'AntenatalCare_Outpatient'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        # Here we define variables used within the function that checks in this ANC visit can run
        gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)
        this_contact = HSI_CareOfWomenDuringPregnancy_EighthAntenatalCareContact(self.module, person_id=person_id)

        # Run the check
        can_anc_run = self.module.check_subsequent_anc_can_run(person_id, this_contact, 8, squeeze_factor,
                                                               gest_age_next_contact)

        if can_anc_run:
            self.module.anc_counter[8] += 1
            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            self.module.interventions_delivered_each_visit_from_anc2(hsi_event=self)

            if df.at[person_id, 'ac_to_be_admitted']:
                self.module.schedule_admission(person_id)

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        return actual_appt_footprint

    def did_not_run(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_EighthAntenatalCareContact: did not run')

    def not_available(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_EighthAntenatalCareContact: cannot not run'
                                         ' with this configuration')


class HSI_CareOfWomenDuringPregnancy_FocusedANCVisit(HSI_Event, IndividualScopeEventMixin):
    """
     This is HSI_CareOfWomenDuringPregnancy_FocusedANCVisit which is scheduled by the PregnancySupervisor if the
     parameter 'anc_service_structure' == 4. This HSI replicates the ANC service structured used within Malawi prior
     to 2016. We use this HSI to replicate the Focused ANC service structure (4 visits at approx 16, 22, 30, 36 weeks)
     within some analyses as the scheduled of interventions per visit is different from the ANC8 structure. This event
     represents all four ANC visits.
     """
    def __init__(self, module, person_id, visit_number):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.visit_number = visit_number

        self.TREATMENT_ID = 'AntenatalCare_Outpatient'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({('AntenatalFirst' if (self.visit_number == 1)
                                                                  else 'ANCSubsequent'): 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]
        params = self.module.current_parameters
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        # First we determine at what point in this womans pregnancy should she return for another visit
        if mother.ps_gestational_age_in_weeks < 22:
            recommended_gestation_next_anc = 22
        elif 22 <= mother.ps_gestational_age_in_weeks < 30:
            recommended_gestation_next_anc = 30
        elif 30 <= mother.ps_gestational_age_in_weeks < 36:
            recommended_gestation_next_anc = 36
        else:
            recommended_gestation_next_anc = 50

        # We calculate the difference between today's date and the date this event should run
        if self.visit_number == 1:
            date_difference = self.sim.date - df.at[person_id, 'ps_date_of_anc1']
        else:
            date_difference = self.sim.date - df.at[person_id, 'ac_date_next_contact']

        # Only women who are alive, still pregnant, not in labour, less than a week 'over due' for the event, have
        # attended less than four visits and are greater than 7 weeks pregnant will undergo the HSI
        if (
            not df.at[person_id, 'is_alive'] or
            not df.at[person_id, 'is_pregnant'] or
            df.at[person_id, 'la_currently_in_labour'] or
            (date_difference > pd.to_timedelta(7, unit='D')) or
            (df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] >= 4) or
            (df.at[person_id, 'ps_gestational_age_in_weeks'] < 7) or
            self.visit_number > 4 or
            self.visit_number != (df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] + 1)
        ):
            return

        self.module.anc_counter[self.visit_number] += 1

        # Women who are inpatients at the time the HSI should run will return at the next recommended point in
        # pregnancy
        if df.at[person_id, 'hs_is_inpatient'] and (df.at[person_id, 'ps_gestational_age_in_weeks'] < 37):
            weeks_due_next_visit = int(recommended_gestation_next_anc - df.at[person_id, 'ps_gestational_age_in_weeks'])
            visit_date = self.sim.date + DateOffset(weeks=weeks_due_next_visit)
            self.sim.modules['HealthSystem'].schedule_hsi_event(self, priority=0,
                                                                topen=visit_date,
                                                                tclose=visit_date + DateOffset(days=7))

            if self.visit_number == 1:
                df.at[person_id, 'ps_date_of_anc1'] = visit_date
            else:
                df.at[person_id, 'ac_date_next_contact'] = visit_date

        # Finally, if the squeeze factor is too high the event wont run and she will return tomorrow
        elif squeeze_factor > params['squeeze_factor_threshold_anc']:
            self.sim.modules['HealthSystem'].schedule_hsi_event(self, priority=0,
                                                                topen=self.sim.date + DateOffset(days=1),
                                                                tclose=self.sim.date + DateOffset(days=2))
            return

        # updated mni with GA at first visit
        if self.visit_number == 1:
            mni[person_id]['ga_anc_one'] = df.at[person_id, 'ps_gestational_age_in_weeks']

        # We add a visit to a rolling total of ANC visits in this pregnancy used for logging
        df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

        # Next interventions are delivered according to gestational age and visit number
        self.module.screening_interventions_delivered_at_every_contact(hsi_event=self)
        self.module.iron_and_folic_acid_supplementation(hsi_event=self)
        self.module.iptp_administration(hsi_event=self)

        if self.visit_number == 1:
            self.module.tb_screening(hsi_event=self)
            self.module.hiv_testing(hsi_event=self)
            self.module.syphilis_screening_and_treatment(hsi_event=self)
            self.module.point_of_care_hb_testing(hsi_event=self)
            self.module.tetanus_vaccination(hsi_event=self)

        elif (self.visit_number == 2) or ((mother.ps_gestational_age_in_weeks > 20) and (self.visit_number == 1)):
            self.module.albendazole_administration(hsi_event=self)
            self.module.tetanus_vaccination(hsi_event=self)

        elif self.visit_number == 3 or ((mother.ps_gestational_age_in_weeks > 30) and (self.visit_number == 1)):
            self.module.point_of_care_hb_testing(hsi_event=self)

        # Following this the woman's next visit is scheduled (if she hasn't already attended 4 visits)
        if self.visit_number < 4:

            # update the visit number for the event scheduling
            self.visit_number = self.visit_number + 1

            # schedule the next event
            self.module.antenatal_care_scheduler(individual_id=person_id,
                                                 visit_to_be_scheduled=self.visit_number,
                                                 recommended_gestation_next_anc=recommended_gestation_next_anc)

        # If the woman has had any complications detected during ANC she is admitted for treatment to be initiated
        if df.at[person_id, 'ac_to_be_admitted']:
            self.module.schedule_admission(person_id)


class HSI_CareOfWomenDuringPregnancy_PresentsForInductionOfLabour(HSI_Event, IndividualScopeEventMixin):
    """
    This is HSI_CareOfWomenDuringPregnancy_PresentsForInductionOfLabour. It is schedule by the PregnancySupervisor Event
    for women who present to the health system for induction as their labour has progressed longer than expected.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'AntenatalCare_Inpatient'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({})
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'maternity_bed': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        # If the woman is no longer alive, pregnant is in labour or is an inpatient already then the event doesnt run
        if not df.at[person_id, 'is_alive'] or not df.at[person_id, 'is_pregnant'] or \
           df.at[person_id, 'la_currently_in_labour'] or df.at[person_id, 'hs_is_inpatient']:
            return

        # We set this admission property to show shes being admitted for induction of labour and hand her over to the
        # labour events
        df.at[person_id, 'ac_admitted_for_immediate_delivery'] = 'induction_now'
        logger.debug(key='message', data=f'Mother {person_id} will move to labour ward for '
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

        self.TREATMENT_ID = 'AntenatalCare_Inpatient'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({})
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'maternity_bed': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        if not df.at[person_id, 'is_alive'] or not df.at[person_id, 'is_pregnant']:
            return

        if not df.at[person_id, 'hs_is_inpatient'] and not df.at[person_id, 'la_currently_in_labour']:
            admission = HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare(
                self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id)

            self.sim.modules['HealthSystem'].schedule_hsi_event(admission, priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

    def never_ran(self):
        self.module.call_if_maternal_emergency_assessment_cant_run(self)

    def did_not_run(self):
        self.module.call_if_maternal_emergency_assessment_cant_run(self)
        return False

    def not_available(self):
        self.module.call_if_maternal_emergency_assessment_cant_run(self)


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

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'AntenatalCare_Inpatient'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({})
        self.ACCEPTED_FACILITY_LEVEL = '1b'

        beddays = self.module.calculate_beddays(person_id)
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'maternity_bed': beddays})

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        params = self.module.current_parameters
        mother = df.loc[person_id]
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        if not mother.is_alive:
            return

        if not (mother.is_pregnant and not mother.la_currently_in_labour and not mother.hs_is_inpatient):
            return

        # check if she will experience delayed care
        pregnancy_helper_functions.check_if_delayed_care_delivery(self.module, squeeze_factor, person_id, hsi_type='an')

        # The event represents inpatient care delivered within the antenatal ward at a health facility. Therefore
        # it is assumed that women with a number of different complications could be sent to this HSI for treatment.

        # ================================= INITIATE TREATMENT FOR ANAEMIA ========================================
        # Women who are referred from ANC or an outpatient appointment following point of care Hb which detected
        # anaemia first have a full blood count test to determine the severity of their anaemia
        if mother.ps_anaemia_in_pregnancy != 'none':

            # This test returns one of a number of possible outcomes as seen below...
            fbc_result = self.module.full_blood_count_testing(self)
            if fbc_result not in ('none', 'mild', 'moderate', 'severe'):
                logger.info(key='error', data='FBC result error')

            # If the FBC detected non severe anaemia (Hb >7) she is treated
            if fbc_result in ('mild', 'moderate'):

                # Women are started on daily iron and folic acid supplementation (if they are not already receiving
                # supplements) as treatment for mild/moderate anaemia
                if not mother.ac_receiving_iron_folic_acid:
                    self.module.iron_and_folic_acid_supplementation(self)

            elif fbc_result == 'severe':
                # In the case of severe anaemia (Hb <7) then, in addition to the above treatments, this woman
                # should receive a blood transfusion to correct her anaemia
                self.module.antenatal_blood_transfusion(person_id, self)
                if not mother.ac_receiving_iron_folic_acid:
                    self.module.iron_and_folic_acid_supplementation(self)

            if fbc_result in ('mild', 'moderate', 'severe'):
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
            from tlo.methods.pregnancy_supervisor import GestationalDiabetesGlycaemicControlEvent
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

        if mother.ps_htn_disorders in ('gest_htn', 'mild_pre_eclamp') and not mother.ac_gest_htn_on_treatment:
            self.module.initiate_maintenance_anti_hypertensive_treatment(person_id, self)

        # Women with severe gestational hypertension are also started on routine oral antihypertensives (if not
        # already receiving- this will prevent progression once this episode of severe hypertension has been
        # rectified)
        elif mother.ps_htn_disorders == 'severe_gest_htn':
            if not mother.ac_gest_htn_on_treatment:
                self.module.initiate_maintenance_anti_hypertensive_treatment(person_id, self)

            # In addition, women with more severe disease are given intravenous anti hypertensives to reduce risk
            # of death
            self.module.initiate_treatment_for_severe_hypertension(person_id, self)

        # Treatment guidelines dictate that women with severe forms of pre-eclampsia should be admitted for delivery
        # to reduce risk of death and pregnancy loss
        elif mother.ps_htn_disorders in ('severe_pre_eclamp', 'eclampsia'):

            # Women are started on oral antihypertensives
            if not mother.ac_gest_htn_on_treatment:
                self.module.initiate_maintenance_anti_hypertensive_treatment(person_id, self)

            # And are given intravenous magnesium sulfate which reduces risk of death from eclampsia and reduces a
            # womans risk of progressing from severe pre-eclampsia to eclampsia during the intrapartum period
            self.module.treatment_for_severe_pre_eclampsia_or_eclampsia(person_id,
                                                                        hsi_event=self)
            # intravenous antihypertensives are also given
            self.module.initiate_treatment_for_severe_hypertension(person_id, self)

            # Finally This property stores what type of delivery this woman is being admitted for
            delivery_mode = ['induction_now', 'avd_now', 'caesarean_now']

            # Mode of delivery is dependent on individual case severity. We use a probability weighted random draw
            # to determine mode of delivery here
            if mother.ps_htn_disorders == 'eclampsia':
                df.at[person_id, 'ac_admitted_for_immediate_delivery'] = self.module.rng.choice(
                    delivery_mode,  p=params['prob_delivery_modes_ec'])

            elif mother.ps_htn_disorders == 'severe_pre_eclamp':
                df.at[person_id, 'ac_admitted_for_immediate_delivery'] = self.module.rng.choice(
                    delivery_mode, p=params['prob_delivery_modes_spe'])

            # Log the indication for any caesarean deliveries
            if df.at[person_id, 'ac_admitted_for_immediate_delivery'] in ('caesarean_now', 'caesarean_future'):
                mni[person_id]['cs_indication'] = 'spe_ec'

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
                mni[person_id]['cs_indication'] = 'an_aph_pa'

            # ---------------------- APH SECONDARY TO PLACENTA PRAEVIA -----------------------------------------
            if mother.ps_placenta_praevia:
                # The treatment plan for a woman with placenta praevia is dependent on both the severity of the
                # bleed and her current gestation at the time of bleeding

                if mother.ps_antepartum_haemorrhage == 'severe':

                    # Women experiencing severe bleeding are admitted immediately for caesarean section
                    df.at[person_id, 'ac_admitted_for_immediate_delivery'] = 'caesarean_now'
                    mni[person_id]['cs_indication'] = 'an_aph_pp'

                elif mother.ps_gestational_age_in_weeks >= 37:
                    # Women experiencing mild or moderate bleeding but who are around term gestation are admitted
                    # for caesarean
                    df.at[person_id, 'ac_admitted_for_immediate_delivery'] = 'caesarean_now'
                    mni[person_id]['cs_indication'] = 'an_aph_pp'

                elif mother.ps_gestational_age_in_weeks < 37:
                    # Women with more mild bleeding remain as inpatients until their gestation has increased and
                    # then will be delivered by caesarean - (no risk of death associated with mild/moderate bleeds)
                    df.at[person_id, 'ac_admitted_for_immediate_delivery'] = 'caesarean_future'
                    mni[person_id]['cs_indication'] = 'an_aph_pp'

                    # self.module.antenatal_blood_transfusion(person_id, self, cause='antepartum_haem')

            if df.at[person_id, 'ac_admitted_for_immediate_delivery'] == 'none':
                logger.info(key='error', data=f'Mother {person_id} was not admitted for delviery following APH')

        # ===================================== INITIATE TREATMENT FOR PROM =======================================
        # Treatment for women with premature rupture of membranes is dependent upon a womans gestational age and if
        # she also has an infection of membrane surrounding the foetus (the chorion)

        if mother.ps_premature_rupture_of_membranes and not mother.ps_chorioamnionitis:
            # If the woman has PROM but no infection, she is given prophylactic antibiotics which will reduce
            # the risk of maternal and neonatal infection
            self.module.antibiotics_for_prom(person_id, self)

            # Guidelines suggest women over 34 weeks of gestation should be admitted for induction to to
            # increased risk of morbidity and mortality
            if mother.ps_gestational_age_in_weeks >= 34:
                df.at[person_id, 'ac_admitted_for_immediate_delivery'] = 'induction_now'

            # Otherwise they may stay as an inpatient until their gestation as increased prior to delivery
            elif mother.ps_gestational_age_in_weeks < 34:
                df.at[person_id, 'ac_admitted_for_immediate_delivery'] = 'induction_future'

        # ============================== INITIATE TREATMENT FOR CHORIOAMNIONITIS ==================================
        # Women with chorioamnionitis are admitted for delivery (and will receive antibiotics in the labour module)
        if mother.ps_chorioamnionitis:
            df.at[person_id, 'ac_admitted_for_immediate_delivery'] = 'induction_now'

        # ======================== ADMISSION FOR DELIVERY (INDUCTION) ========================================
        # Women for whom immediate delivery is indicated are schedule to move straight to the labour model where
        # they will have the appropriate properties set and facility delivery at a hospital scheduled (mode of
        # delivery will match the recommended mode here)
        if df.at[person_id, 'ac_admitted_for_immediate_delivery'] in ('induction_now', 'caesarean_now'):
            self.sim.schedule_event(LabourOnsetEvent(self.sim.modules['Labour'], person_id), self.sim.date)

        # Women who require delivery BUT are not in immediate risk of morbidity/mortality will remain as
        # inpatients until they can move to the labour model. Currently it is possible for women to go into
        # labour whilst as an inpatient - it is assumed they are delivered via the mode recommended here
        # (i.e induction/caesarean)
        elif df.at[person_id, 'ac_admitted_for_immediate_delivery'] in ('caesarean_future', 'induction_future'):

            # Here we calculate how many days this woman needs to remain on the antenatal ward before she can go
            # for delivery (assuming delivery is indicated to occur at 37 weeks)
            if mother.ps_gestational_age_in_weeks < 37:
                days_until_safe_for_cs = int((37 * 7) - (mother.ps_gestational_age_in_weeks * 7))
            else:
                days_until_safe_for_cs = 1

            # We schedule the LabourOnset event for this woman will be able to progress for delivery
            admission_date = self.sim.date + DateOffset(days=days_until_safe_for_cs)

            logger.debug(key='message', data=f'Mother {person_id} will move to labour ward for '
                                             f'{df.at[person_id, "ac_admitted_for_immediate_delivery"]} on '
                                             f'{admission_date}')

            self.sim.schedule_event(LabourOnsetEvent(self.sim.modules['Labour'], person_id),
                                    admission_date)
        else:
            mni[person_id]['delay_one_two'] = False
            mni[person_id]['delay_three'] = False

    def did_not_run(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare: did not run')
        return False

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

        self.TREATMENT_ID = 'AntenatalCare_FollowUp'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]

        if not mother.is_alive or not mother.is_pregnant:
            return

        # We only run the event if the woman is not already in labour or already admitted due to something else
        if not mother.la_currently_in_labour and not mother.hs_is_inpatient:

            # Health care worker performs a full blood count
            fbc_result = self.module.full_blood_count_testing(self)

            # If she is determined to still be anaemic she is admitted for additional treatment via the inpatient event
            if fbc_result in ('mild', 'moderate', 'severe'):

                admission = HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare(
                    self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id)

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

        self.TREATMENT_ID = 'AntenatalCare_FollowUp'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]

        from tlo.methods.pregnancy_supervisor import GestationalDiabetesGlycaemicControlEvent

        if not mother.is_alive or not mother.is_pregnant:
            return

        if not mother.la_currently_in_labour and not mother.hs_is_inpatient and mother.ps_gest_diab != 'none' \
                and (mother.ac_gest_diab_on_treatment != 'none') and (mother.ps_gestational_age_in_weeks > 21):

            def schedule_gdm_event_and_checkup():
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

            # If the treatment a woman was started on has not controlled her hyperglycemia she will be started on the
            # next treatment
            if mother.ps_gest_diab == 'uncontrolled':

                # Women for whom diet and exercise was not effective in controlling hyperglycemia are started on oral
                # meds
                if mother.ac_gest_diab_on_treatment == 'diet_exercise':
                    dose = self.module.get_approx_days_of_pregnancy(person_id) * 2
                    cons = {_i: dose for _i in self.module.item_codes_preg_consumables['oral_diabetic_treatment']}
                    avail = self.get_consumables(item_codes=cons)

                    # If the meds are available women are started on that treatment
                    if avail:
                        df.at[person_id, 'ac_gest_diab_on_treatment'] = 'orals'

                        # Assume new treatment is effective in controlling blood glucose on initiation
                        df.at[person_id, 'ps_gest_diab'] = 'controlled'

                        # schedule followup
                        schedule_gdm_event_and_checkup()

                # This process is repeated for mothers for whom oral medication is not effectively controlling their
                # blood sugar- they are started on insulin
                if mother.ac_gest_diab_on_treatment == 'orals':
                    cons = {_i: 5 for _i in self.module.item_codes_preg_consumables['insulin_treatment']}
                    avail = self.get_consumables(item_codes=cons)

                    if avail:
                        df.at[person_id, 'ac_gest_diab_on_treatment'] = 'insulin'
                        df.at[person_id, 'ps_gest_diab'] = 'controlled'

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

        self.TREATMENT_ID = 'AntenatalCare_PostAbortion'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({})
        self.ACCEPTED_FACILITY_LEVEL = '1b'  # any hospital?
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'maternity_bed': 3})  # todo: check with TC

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]
        abortion_complications = self.sim.modules['PregnancySupervisor'].abortion_complications

        if not mother.is_alive or not abortion_complications.has_any([person_id], 'sepsis', 'haemorrhage', 'injury',
                                                                     'other', first=True):
            return

        # Determine if there will be a delay due to high squeeze
        pregnancy_helper_functions.check_if_delayed_care_delivery(self.module, squeeze_factor, person_id,
                                                                  hsi_type='an')

        # Request baseline PAC consumables
        baseline_cons = pregnancy_helper_functions.return_cons_avail(
            self.module, self, self.module.item_codes_preg_consumables, core='post_abortion_care_core',
            optional='post_abortion_care_optional')

        # Check HCW availability to deliver surgical removal of retained products
        sf_check = pregnancy_helper_functions.check_emonc_signal_function_will_run(self.sim.modules['Labour'],
                                                                                   sf='retained_prod',
                                                                                   hsi_event=self)

        # Then we determine if a woman gets treatment for her complication depending on availability of the baseline
        # consumables (misoprostol) or a HCW who can conduct MVA/DC (we dont model equipment) and additional
        # consumables for management of her specific complication
        if abortion_complications.has_any([person_id], 'sepsis', first=True):

            cons_for_sepsis_pac = pregnancy_helper_functions.return_cons_avail(
                self.module, self, self.module.item_codes_preg_consumables, core='post_abortion_care_sepsis_core',
                optional='post_abortion_care_sepsis_optional')

            if cons_for_sepsis_pac and (baseline_cons or sf_check):
                df.at[person_id, 'ac_received_post_abortion_care'] = True

        elif abortion_complications.has_any([person_id], 'haemorrhage', first=True):

            cons_for_haemorrhage = pregnancy_helper_functions.return_cons_avail(
                self.module, self, self.module.item_codes_preg_consumables, core='blood_transfusion',
                optional='iv_drug_equipment')

            cons_for_shock = pregnancy_helper_functions.return_cons_avail(
                self.module, self, self.module.item_codes_preg_consumables, core='post_abortion_care_shock',
                optional='post_abortion_care_shock_optional')

            if cons_for_haemorrhage and cons_for_shock and (baseline_cons or sf_check):
                df.at[person_id, 'ac_received_post_abortion_care'] = True

        elif abortion_complications.has_any([person_id], 'injury', first=True):
            cons_for_shock = pregnancy_helper_functions.return_cons_avail(
                self.module, self, self.module.item_codes_preg_consumables, core='post_abortion_care_shock',
                optional='post_abortion_care_shock_optional')

            if cons_for_shock and (baseline_cons or sf_check):
                df.at[person_id, 'ac_received_post_abortion_care'] = True

        elif abortion_complications.has_any([person_id], 'other', first=True) and (baseline_cons or sf_check):
            df.at[person_id, 'ac_received_post_abortion_care'] = True

        if df.at[person_id, 'ac_received_post_abortion_care']:
            pregnancy_helper_functions.log_met_need(self.module, 'pac', self)

    def did_not_run(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_PostAbortionCaseManagement: did not run')
        return False

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

        self.TREATMENT_ID = 'AntenatalCare_PostEctopicPregnancy'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'MajorSurg': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'maternity_bed': 5})  # todo: check with TC

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]

        if not mother.is_alive or (mother.ps_ectopic_pregnancy == 'none'):
            return

        # We define the required consumables and check their availability
        avail = pregnancy_helper_functions.return_cons_avail(
            self.module, self, self.module.item_codes_preg_consumables, core='ectopic_pregnancy_core',
            optional='ectopic_pregnancy_optional')

        # If they are available then treatment can go ahead
        if avail:
            self.sim.modules['PregnancySupervisor'].mother_and_newborn_info[person_id]['delete_mni'] = True
            pregnancy_helper_functions.log_met_need(self.module, 'ep_case_mang', self)

            # For women who have sought care after they have experienced rupture we use this treatment variable to
            # reduce risk of death (women who present prior to rupture do not pass through the death event as we assume
            # rupture is on the causal pathway to death - hence no treatment property)
            if mother.ps_ectopic_pregnancy == 'ruptured':
                df.at[person_id, 'ac_ectopic_pregnancy_treated'] = True

        else:
            # However if treatment cant be delivered for women who have not yet experienced rupture (due to lack of
            # consumables) we schedule these women to arrive at the rupture event as they have not received treatment
            if df.at[person_id, 'ps_ectopic_pregnancy'] == 'not_ruptured':
                self.module.ectopic_pregnancy_treatment_doesnt_run(self)

    def never_ran(self):
        self.module.ectopic_pregnancy_treatment_doesnt_run(self)

    def did_not_run(self):
        self.module.ectopic_pregnancy_treatment_doesnt_run(self)
        return False

    def not_available(self):
        self.module.ectopic_pregnancy_treatment_doesnt_run(self)


class CareOfWomenDuringPregnancyLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """This is CareOfWomenDuringPregnancyLoggingEvent. It runs yearly to capture the number of ANC visits which
    have ran (and interventions are delivered) for women during pregnancy."""

    def __init__(self, module):
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):

        yearly_counts = self.module.anc_counter
        logger.info(key='anc_visits_which_ran', data=yearly_counts)
        self.module.anc_counter = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
