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

        # First we define dictionaries which will store the current parameters of interest (to allow parameters to
        # change between 2010 and 2020)
        self.current_parameters = dict()

        # and then define a dictionary which will hold the required consumables for each intervention
        self.item_codes_for_consumables_required_pregnancy = dict()

    INIT_DEPENDENCIES = {'Demography', 'HealthSystem', 'PregnancySupervisor'}
    ADDITIONAL_DEPENDENCIES = {'Contraception', 'Labour', 'Lifestyle'}

    METADATA = {
        Metadata.USES_HEALTHSYSTEM,
    }

    PARAMETERS = {

        # CARE SEEKING...
        'prob_anc_continues': Parameter(
            Types.LIST, 'probability a woman will return for a subsequent ANC appointment'),
        'prob_an_ip_at_facility_level_1_2_3': Parameter(
            Types.LIST, 'probabilities that antenatal inpatient care will be scheduled at facility level 1, 2 or 3'),

        # TREATMENT EFFECTS...
        'effect_of_ifa_for_resolving_anaemia': Parameter(
            Types.LIST, 'treatment effectiveness of starting iron and folic acid on resolving anaemia'),
        'effect_of_iron_replacement_for_resolving_anaemia': Parameter(
            Types.LIST, 'treatment effectiveness of iron replacement in resolving anaemia'),
        'effect_of_folate_replacement_for_resolving_anaemia': Parameter(
            Types.LIST, 'treatment effectiveness of folate replacement in resolving anaemia'),
        'effect_of_b12_replacement_for_resolving_anaemia': Parameter(
            Types.LIST, 'treatment effectiveness of b12 replacement in resolving anaemia'),
        'treatment_effect_blood_transfusion_anaemia': Parameter(
            Types.LIST, 'treatment effectiveness of blood transfusion for anaemia in pregnancy'),
        'effect_of_iron_replacement_for_resolving_iron_def': Parameter(
            Types.LIST, 'treatment effectiveness of iron replacement in resolving iron deficiency'),
        'effect_of_folate_replacement_for_resolving_folate_def': Parameter(
            Types.LIST, 'treatment effectiveness of folate replacement in resolving folate deficiency'),
        'effect_of_b12_replacement_for_resolving_b12_def': Parameter(
            Types.LIST, 'treatment effectiveness of b12 replacement in resolving b12 deficiency'),
        'prob_evac_procedure_pac': Parameter(
            Types.LIST, 'Probabilities that a woman will receive D&C, MVA or misoprostal as treatment for abortion '),

        # INTERVENTION PROBABILITIES...
        'squeeze_factor_threshold_anc': Parameter(
            Types.LIST, 'squeeze factor threshold over which an ANC appointment cannot run'),
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
        'prob_intervention_delivered_llitn': Parameter(
            Types.LIST, 'probability a woman will receive the intervention "Long lasting insecticide treated net" '
                        'given that the HSI has ran and the consumables are available (proxy for clinical quality)'),
        'prob_intervention_delivered_tt': Parameter(
            Types.LIST, 'probability a woman will receive the intervention "tetanus toxoid" given that the HSI has '
                        'ran and the consumables are available (proxy for clinical quality)'),
        'prob_intervention_delivered_poct': Parameter(
            Types.LIST, 'probability a woman will receive the intervention "point of care Hb testing" given that the '
                        'HSI has ran and the consumables are available (proxy for clinical quality)'),
        'prob_intervention_delivered_syph_test': Parameter(
            Types.LIST, 'probability a woman will receive the intervention "Syphilis test" given that the HSI has ran '
                        'and the consumables are available (proxy for clinical quality)'),
        'prob_intervention_delivered_iptp': Parameter(
            Types.LIST, 'probability a woman will receive the intervention "IPTp" given that the HSI has ran '
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
                                                                   'caesarean_now', 'caesarean_future', 'avd_now']),
    }

    def read_parameters(self, data_folder):
        parameter_dataframe = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_AntenatalCare.xlsx',
                                            sheet_name='parameter_values')
        self.load_parameters_from_dataframe(parameter_dataframe)

        # For the first period (2010-2015) we use the first value in each list as a parameter
        for key, value in self.parameters.items():
            self.current_parameters[key] = self.parameters[key][0]

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
        df.loc[df.is_alive, 'ac_post_abortion_care_interventions'] = 0
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

    def define_required_consumables(self):
        """
        This function defines the required consumables for each intervention delivered during this module and stores
        them in a module level dictionary called within HSIs
        """
        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

        # -------------------------------------------- ECTOPIC PREGNANCY ---------------------------------------------
        item1 = pd.unique(consumables.loc[consumables["Items"] == "Epinephrine, ampoule, 1 mg/ml", "Item_Code"])[0]

        item2 = pd.unique(consumables.loc[consumables["Items"] == "Lidocaine HCl (in dextrose 7.5%), ampoule 2 ml",
                                                                  "Item_Code"])[0]

        item3 = pd.unique(
            consumables.loc[consumables["Items"] == "Sodium chloride, injectable solution, 0,9 %, 500 ml",
                            "Item_Code"])[0]

        item4 = pd.unique(consumables.loc[consumables["Items"] == "Syringe, needle + swab", "Item_Code"])[0]

        item5 = pd.unique(consumables.loc[consumables["Items"] == "Suture pack", "Item_Code"])[0]

        item6 = pd.unique(consumables.loc[consumables["Items"] == "Gauze pad, 10 x 10 cm, sterile", "Item_Code"])[0]

        item7 = pd.unique(consumables.loc[consumables["Items"] == "Ampicillin, powder for injection, 500 mg, vial",
                                          "Item_Code"])[0]

        item8 = pd.unique(consumables.loc[consumables["Items"] == "Gentamycin, injection, 40 mg/ml in 2 ml vial",
                                          "Item_Code"])[0]

        item9 = pd.unique(consumables.loc[consumables["Items"] == "Metronidazole, injection, 500 mg in 100 ml vial",
                                          "Item_Code"])[0]

        item10 = pd.unique(consumables.loc[consumables["Items"] == "Paracetamol, tablet, 500 mg", "Item_Code"])[0]

        item11 = pd.unique(consumables.loc[consumables["Items"] == "Pethidine, 50 mg/ml, 2 ml ampoule", "Item_Code"])[0]

        # key = name of the item, value = item code
        ep_cons = {'epinephrine': item1, 'lidocaine': item2, 'sodium_chloride': item3, 'syringe': item4,
                   'suture_pack': item5,  'gauze': item6, 'ampicillin': item7, 'gentamycin': item8,
                   'metronidazole': item9, 'paracetamol': item10, 'pethidine': item11}

        self.item_codes_for_consumables_required_pregnancy['ectopic_pregnancy'] = ep_cons

        # ------------------------------------------- POST ABORTION CARE ----------------------------------------------
        item1 = pd.unique(consumables.loc[consumables["Items"] == "Misoprostol, tablet, 200 mcg", "Item_Code"])[0]

        item2 = pd.unique(consumables.loc[consumables["Items"] == "Paracetamol, tablet, 500 mg",
                                          "Item_Code"])[0]

        item3 = pd.unique(consumables.loc[consumables["Items"] == "Ampicillin, powder for injection, 500 mg, vial",
                                          "Item_Code"])[0]

        item4 = pd.unique(consumables.loc[consumables["Items"] == "Gentamycin, injection, 40 mg/ml in 2 ml vial",
                                          "Item_Code"])[0]

        item5 = pd.unique(consumables.loc[consumables["Items"] == "Metronidazole, injection, 500 mg in 100 ml vial",
                                          "Item_Code"])[0]

        item6 = pd.unique(consumables.loc[consumables["Items"] == "Tetracycline, tablet, 500 mg", "Item_Code"])[0]

        item7 = pd.unique(consumables.loc[consumables["Items"] == "Lidocaine HCl (in dextrose 7.5%), ampoule 2 ml",
                                          "Item_Code"])[0]

        item8 = pd.unique(consumables.loc[consumables["Items"] == "Syringe, needle + swab", "Item_Code"])[0]

        item9 = pd.unique(consumables.loc[consumables["Items"] == "Methylergometrine, Injection  0.2 mg/ml, 1 ml amp",
                                                                  "Item_Code"])[0]

        item10 = pd.unique(consumables.loc[consumables["Items"] == "Pethidine, 50 mg/ml, 2 ml ampoule", "Item_Code"])[0]

        item11 = pd.unique(consumables.loc[consumables["Items"] == "Sodium chloride, injectable solution, 0,9 %, 500 ml",
                                                                   "Item_Code"])[0]

        pac_cons = {'misoprostol': item1, 'paracetamol': item2, 'ampicillin': item3, 'gentamycin': item4,
                    'metronidazole': item5, 'tetracycline': item6, 'lidocaine': item7, 'syringe': item8,
                    'methylergometrine': item9, 'pethidine': item10, 'sodium_chloride': item11}

        self.item_codes_for_consumables_required_pregnancy['post_abortion_care'] = pac_cons

        # ---------------------------------- URINE DIPSTICK ----------------------------------------------------------
        item = pd.unique(consumables.loc[consumables['Items'] == 'Test strips, urine analysis', 'Item_Code'])[0]
        dp_cons = {'test_strips': item}

        self.item_codes_for_consumables_required_pregnancy['urine_dipstick'] = dp_cons

        # ---------------------------------- IRON AND FOLIC ACID ------------------------------------------------------
        item = pd.unique(
            consumables.loc[consumables['Items'] == 'Ferrous Salt + Folic Acid, tablet, 200 + 0.25 mg',
                            'Item_Code'])[0]

        ifa_cons = {'ifa': item}

        self.item_codes_for_consumables_required_pregnancy['iron_folic_acid'] = ifa_cons

        # --------------------------------- BALANCED ENERGY AND PROTEIN ----------------------------------------------
        item = pd.unique(
            consumables.loc[consumables['Items'] == 'Dietary supplements (country-specific)', 'Item_Code'])[0]

        bep_cons = {'diet_supps': item}

        self.item_codes_for_consumables_required_pregnancy['balanced_energy_protein'] = bep_cons

        # --------------------------------- INSECTICIDE TREATED NETS ------------------------------------------------
        item = pd.unique(
            consumables.loc[consumables['Items'] == 'Insecticide-treated net', 'Item_Code'])[0]

        itn_cons = {'bed_net': item}

        self.item_codes_for_consumables_required_pregnancy['itn'] = itn_cons

        # --------------------------------- TETANUS TOXOID VACCINATION  ------------------------------------------------
        item1 = pd.unique(
            consumables.loc[consumables['Items'] == 'Tetanus toxoid, injection', 'Item_Code'])[0]
        item2 = pd.unique(
            consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]

        tt_cons = {'tetanus_vaccine': item1, 'syringe': item2}

        self.item_codes_for_consumables_required_pregnancy['tetanus_toxoid'] = tt_cons

        # --------------------------------- CALCIUM SUPPLEMENTS -----------------------------------------------------
        item = pd.unique(consumables.loc[consumables['Items'] == 'Calcium, tablet, 600 mg', 'Item_Code'])[0]

        ca_cons = {'calcium_supps': item}

        self.item_codes_for_consumables_required_pregnancy['calcium'] = ca_cons

        # -------------------------------- HAEMOGLOBIN TESTING -------------------------------------------------------
        item1 = pd.unique(
            consumables.loc[consumables['Items'] == 'Haemoglobin test (HB)', 'Item_Code'])[0]
        item2 = pd.unique(
            consumables.loc[consumables['Items'] == 'Blood collecting tube, 5 ml', 'Item_Code'])[0]
        item3 = pd.unique(
            consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]
        item4 = pd.unique(
            consumables.loc[consumables['Items'] == 'Gloves, exam, latex, disposable, pair', 'Item_Code'])[0]

        hb_test_cons = {'hb_test': item1, 'blood_tube': item2, 'syringe':item3, 'gloves': item4}

        self.item_codes_for_consumables_required_pregnancy['hb_test'] = hb_test_cons

        # ------------------------------------------- ALBENDAZOLE -----------------------------------------------------
        item = pd.unique(consumables.loc[consumables['Items'] == 'Albendazole 200mg_1000_CMST', 'Item_Code'])[0]

        alben_cons = {'albendazole': item}

        self.item_codes_for_consumables_required_pregnancy['albendazole'] = alben_cons

        # ------------------------------------------- HEP B TESTING ---------------------------------------------------
        item1 = pd.unique(
            consumables.loc[consumables['Items'] == 'Hepatitis B test kit-Dertemine_100 tests_CMST', 'Item_Code'])[0]
        item2 = pd.unique(
            consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]
        item3 = pd.unique(
            consumables.loc[consumables['Items'] == 'Gloves, exam, latex, disposable, pair', 'Item_Code'])[0]

        hep_b_cons = {'hep_b_test':item1, 'syringe': item2, 'gloves': item3}

        self.item_codes_for_consumables_required_pregnancy['hep_b_test'] = hep_b_cons

        # ------------------------------------------- SYPHILIS TESTING ------------------------------------------------
        item1 = pd.unique(
            consumables.loc[consumables['Items'] == 'Test, Rapid plasma reagin (RPR)', 'Item_Code'])[0]
        item2 = pd.unique(
            consumables.loc[consumables['Items'] == 'Blood collecting tube, 5 ml', 'Item_Code'])[0]
        item3 = pd.unique(
            consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]
        item4 = pd.unique(
            consumables.loc[consumables['Items'] == 'Gloves, exam, latex, disposable, pair', 'Item_Code'])[0]

        syph_test_cons = {'syphilis_test': item1, 'blood_tube': item2, 'syringe': item3, 'gloves': item4}

        self.item_codes_for_consumables_required_pregnancy['syphilis_test'] = syph_test_cons

        # ------------------------------------------- SYPHILIS TREATMENT ----------------------------------------------
        item1 = pd.unique( consumables.loc[consumables['Items'] == 'Benzathine benzylpenicillin, '
                                                                   'powder for injection, 2.4 million IU',
                                                                   'Item_Code'])[0]

        item2 = pd.unique(consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]

        syph_treatment_cons = {'benzylpenicillin': item1, 'syringe': item2}

        self.item_codes_for_consumables_required_pregnancy['syphilis_treatment'] = syph_treatment_cons

        # ----------------------------------------------- IPTP --------------------------------------------------------
        item = pd.unique(consumables.loc[consumables['Items'] == 'Sulfamethoxazole + trimethropin, '
                                                                 'tablet 400 mg + 80 mg', 'Item_Code'])[0]

        iptp_cons = {'iptp': item}

        self.item_codes_for_consumables_required_pregnancy['iptp'] = iptp_cons

        # ----------------------------------------------- GDM TEST ----------------------------------------------------
        item1 = pd.unique(consumables.loc[consumables['Items'] == 'Blood glucose level test', 'Item_Code'])[0]

        item2 = pd.unique(consumables.loc[consumables['Items'] == 'Blood collecting tube, 5 ml', 'Item_Code'])[0]

        item3 = pd.unique(consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]

        item4 = pd.unique(consumables.loc[consumables['Items'] == 'Gloves, exam, latex, disposable, pair',
                                          'Item_Code'])[0]

        gdm_test_cons = {'gdm_test': item1, 'blood_tube': item2, 'syringe': item3, 'gloves': item4}

        self.item_codes_for_consumables_required_pregnancy['gdm_test'] = gdm_test_cons

        # ------------------------------------------ FULL BLOOD COUNT -------------------------------------------------
        item1 = pd.unique(
            consumables.loc[consumables['Items'] == 'Complete blood count', 'Item_Code'])[0]
        item2 = pd.unique(
            consumables.loc[consumables['Items'] == 'Blood collecting tube, 5 ml', 'Item_Code'])[0]
        item3 = pd.unique(
            consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]
        item4 = pd.unique(
            consumables.loc[consumables['Items'] == 'Gloves, exam, latex, disposable, pair', 'Item_Code'])[0]

        fbc_test_cons = {'fbc_test': item1, 'blood_tube': item2, 'syringe': item3, 'gloves': item4}

        self.item_codes_for_consumables_required_pregnancy['full_blood_count'] = fbc_test_cons

        #  --------------------------------- ANAEMIA DEFICIENCY TREATMENT --------------------------------------------
        item = pd.unique(
            consumables.loc[
                consumables['Items'] ==
                'ferrous sulphate 200 mg, coated (65 mg iron)_1000_IDA', 'Item_Code'])[0]

        iron_cons = {'ferrous_sulphate': item}

        self.item_codes_for_consumables_required_pregnancy['iron_treatment'] = iron_cons

        item = pd.unique(
            consumables.loc[
                consumables['Items'] ==
                'vitamin B12 (cyanocobalamine) 1 mg/ml, 1 ml, inj._100_IDA', 'Item_Code'])[0]

        b12_cons = {'vit_b12': item}

        self.item_codes_for_consumables_required_pregnancy['b12_treatment'] = b12_cons

        # ---------------------------------------- BLOOD TRANSFUSION -------------------------------------------------
        item1 = pd.unique(consumables.loc[consumables['Items'] == 'Blood, one unit', 'Item_Code'])[0]

        item2 = pd.unique(consumables.loc[consumables['Items'] == 'IV giving/infusion set, with needle',
                                                      'Item_Code'])[0]

        transfusion_cons = {'blood': item1, 'giving_set': item2}

        self.item_codes_for_consumables_required_pregnancy['blood_transfusion'] = transfusion_cons

        # --------------------------------------- ORAL ANTIHYPERTENSIVES ---------------------------------------------
        item = pd.unique(
            consumables.loc[consumables['Items'] == 'Methyldopa 250mg_1000_CMST', 'Item_Code'])[0]

        oral_anti_htns_cons = {'methyldopa': item}

        self.item_codes_for_consumables_required_pregnancy['oral_antihypertensives'] = oral_anti_htns_cons

        # -------------------------------------  INTRAVENOUS ANTIHYPERTENSIVES ---------------------------------------
        item1 = pd.unique(
            consumables.loc[consumables['Items'] == 'Hydralazine, powder for injection, 20 mg ampoule',
                            'Item_Code'])[0]
        item2 = pd.unique(
            consumables.loc[consumables['Items'] == 'Water for injection, 10ml_Each_CMST', 'Item_Code'])[0]
        item3 = pd.unique(
            consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]
        item4 = pd.unique(
            consumables.loc[consumables['Items'] == 'Gloves, exam, latex, disposable, pair', 'Item_Code'])[0]

        iv_anti_htns_cons = {'hydralazine': item1, 'water_for_injection': item2, 'syringe': item3, 'gloves': item4}

        self.item_codes_for_consumables_required_pregnancy['iv_antihypertensives'] = iv_anti_htns_cons

        # ---------------------------------------- MAGNESIUM SULFATE ------------------------------------------------
        item1 = pd.unique(
            consumables.loc[consumables['Items'] == 'Magnesium sulfate, injection, 500 mg/ml in 10-ml ampoule',
                            'Item_Code'])[0]
        item2 = pd.unique(
            consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]

        ec_treatment_cons = {'mgso4': item1, 'syringe': item2}

        self.item_codes_for_consumables_required_pregnancy['magnesium_sulfate'] = ec_treatment_cons

        # -------------------------------------- ANTIBIOTICS FOR PROM ------------------------------------------------
        item1 = pd.unique(
            consumables.loc[consumables['Items'] == 'Benzathine benzylpenicillin, powder for injection, 2.4 million IU',
                            'Item_Code'])[0]
        item2 = pd.unique(
            consumables.loc[consumables['Items'] == 'Water for injection, 10ml_Each_CMST', 'Item_Code'])[0]
        item3 = pd.unique(
            consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]
        item4 = pd.unique(
            consumables.loc[consumables['Items'] == 'Gloves, exam, latex, disposable, pair', 'Item_Code'])[0]

        abx_prom_cons = {'benzylpenicillin': item1, 'water_for_injection': item2, 'syringe': item3, 'gloves': item4}

        self.item_codes_for_consumables_required_pregnancy['abx_for_prom'] = abx_prom_cons

        # ----------------------------------- ORAL DIABETIC MANAGEMENT -----------------------------------------------
        item = pd.unique(
            consumables.loc[consumables['Items'] == 'Glibenclamide 5mg_1000_CMST', 'Item_Code'])[0]

        oral_anti_diabs_cons = {'glibenclamide': item}

        self.item_codes_for_consumables_required_pregnancy['oral_diabetic_treatment'] = oral_anti_diabs_cons

        # ---------------------------------------- INSULIN ----------------------------------------------------------
        item = pd.unique(
            consumables.loc[consumables['Items'] == 'Insulin soluble 100 IU/ml, 10ml_each_CMST',
                            'Item_Code'])[0]

        insulin_cons = {'insulin': item}

        self.item_codes_for_consumables_required_pregnancy['insulin_treatment'] = insulin_cons

    def initialise_simulation(self, sim):

        # We register the logging event and schedule to run each year
        sim.schedule_event(AntenatalCareLoggingEvent(self),
                           sim.date + DateOffset(years=1))

        # call the following function to store the required consumables for the simulation run within the appropriate
        # dictionary
        self.define_required_consumables()

        # ==================================== REGISTERING DX_TESTS =================================================
        params = self.current_parameters
        # Next we register the relevant dx_tests used within this module...
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(

            # TODO: ADD CONSUMABLES TO DX TESTS WHEN NEW FUNCTIONALITY IS IN

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
                specificity=params['specificity_blood_test_glucose']),

            # This test represents point of care glucose testing used in ANC to detect hyperglycemia, associated with
            # gestational diabetes
            blood_test_syphilis=DxTest(
                property='ps_syphilis',
                sensitivity=params['sensitivity_blood_test_syphilis'],
                specificity=params['specificity_blood_test_syphilis']))

        if 'Hiv' not in self.sim.modules:
            logger.warning(key='message', data='HIV module is not registered in this simulation run and therefore HIV '
                                               'testing will not happen in antenatal care')

    # This function is used within this and other modules to reset properties from this module when a woman is no longer
    # pregnant to ensure in future pregnancies properties arent incorrectly set to certain values
    def care_of_women_in_pregnancy_property_reset(self, ind_or_df, id_or_index):
        df = self.sim.population.props

        if ind_or_df == 'individual':
            set = df.at
            # todo: seems like this crashes when it shouldnt have even been called
            # assert not set[id_or_index, 'is_pregnant']
        else:
            set = df.loc
            assert not set[id_or_index, 'is_pregnant'].any()

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
        df.at[child_id, 'ac_post_abortion_care_interventions'] = 0
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
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        if df.at[mother_id, 'is_alive']:
            #  run a check at birth to make sure no women exceed 8 visits
            assert df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] < 9

            # We log the total number of ANC contacts a woman has undergone at the time of birth via this dictionary
            if 'ga_anc_one' in mni[mother_id].keys():
                ga_anc_one = mni[mother_id]['ga_anc_one']
            else:
                ga_anc_one = 0

            total_anc_visit_count = {'person_id': mother_id,
                                     'total_anc': df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'],
                                     'ga_anc_one': ga_anc_one}

            logger.info(key='anc_count_on_birth', data=total_anc_visit_count,
                        description='A dictionary containing the number of ANC visits each woman has on birth')

            # We then reset all relevant variables pertaining to care received during the antenatal period to avoid
            # treatments remaining in place for future pregnancies
            self.care_of_women_in_pregnancy_property_reset(ind_or_df='individual', id_or_index=mother_id)

    def on_hsi_alert(self, person_id, treatment_id):
        logger.debug(key='message', data=f'This is CareOfWomenDuringPregnancy, being alerted about a health system '
                                         f'interaction person {person_id} for: {treatment_id}')

    #  ================================ ADDITIONAL ANTENATAL HELPER FUNCTIONS =========================================
    def get_approx_days_of_pregnancy(self, person_id):
        df = self.sim.population.props

        approx_days = (40 - df.at[person_id, 'ps_gestational_age_in_weeks']) * 7
        if approx_days == 0:
            approx_days = 7

        return approx_days

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
        params = self.current_parameters

        # Prevent women returning to ANC at very late gestations- this needs to be reviewed and linked with induction
        if df.at[individual_id, 'ps_gestational_age_in_weeks'] >= 42:
            return

        # We check that women will only be scheduled for the next ANC contact in the schedule
        assert df.at[individual_id, 'ps_gestational_age_in_weeks'] < recommended_gestation_next_anc

        # This function determines the correct next visit and if that visit will go ahead
        def select_visit_and_determine_if_woman_will_attend(visit_to_be_scheduled):

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

            # This function uses a womans gestation age to determine when the next visit should occur and schedules it
            # accordingly
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

                logger.debug(key='message', data=f'mother {individual_id} will seek ANC {visit_to_be_scheduled} '
                                                 f'contact on {visit_date}')

                # We store the date of her next visit and use this date as part of a check when the ANC HSIs run
                df.at[individual_id, 'ac_date_next_contact'] = visit_date

            # If this woman has attended less than 4 visits, and is predicted to attend > 4 (as determined via the
            # PregnancySupervisor module when ANC1 is scheduled) her subsequent ANC appointment is automatically
            # scheduled
            if visit_to_be_scheduled <= 4:
                if df.at[individual_id, 'ps_anc4']:
                    calculate_visit_date_and_schedule_visit(visit)
                else:
                    # If she is not predicted to attend 4 or more visits, we use a probability to determine if she will
                    # seek care for her next contact
                    # If so, the HSI is scheduled in the same way
                    if self.rng.random_sample() < params['prob_anc_continues']:
                        calculate_visit_date_and_schedule_visit(visit)
                    else:
                        # If additional ANC care is not sought nothing happens
                        logger.debug(key='message', data=f'mother {individual_id} will not seek any additional '
                                                         f'antenatal care for this pregnancy')
            elif visit_to_be_scheduled > 4:
                # After 4 or more visits we use this probability to determine if the woman will seek care for
                # her next contact
                if self.rng.random_sample() < params['prob_anc_continues']:
                    calculate_visit_date_and_schedule_visit(visit)
                else:
                    logger.debug(key='message', data=f'mother {individual_id} will not seek any additional antenatal '
                                                     f'care for this pregnancy')

        # We run the function to schedule the HSI
        if 2 <= visit_to_be_scheduled <= 8:
            select_visit_and_determine_if_woman_will_attend(visit_to_be_scheduled)

    def schedule_admission(self, individual_id):
        """
        This function is called within each of the ANC HSIs for women who require admission due to a complication
        detected during ANC
        :param individual_id: individual_id
        """
        df = self.sim.population.props

        # check correct women have been sent
        assert df.at[individual_id, 'ac_to_be_admitted']
        logger.info(key='anc_interventions', data={'mother': individual_id, 'intervention': 'admission'})

        # Use a weighted random draw to determine which level of facility the woman will be admitted too
        # facility_level = int(self.rng.choice([1, 2, 3], p=params['prob_an_ip_at_facility_level_1_2_3']))
        facility_level = int(self.rng.choice([1, 2], p=[0.5, 0.5]))

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

    def call_if_maternal_emergency_assessment_cant_run(self, hsi_event):
        df = self.sim.population.props
        individual_id = hsi_event.target

        if df.at[individual_id, 'is_pregnant'] and not df.at[individual_id, 'la_currently_in_labour']:
            logger.debug(key='message', data=f'HSI_CareOfWomenDuringPregnancy_MaternalEmergencyAssessment: did not'
                                             f' run for person {individual_id}')

            self.sim.modules['PregnancySupervisor'].apply_risk_of_death_from_monthly_complications(individual_id)

    # ================================= INTERVENTIONS DELIVERED DURING ANC ============================================
    # The following functions contain the interventions that are delivered as part of routine ANC contacts. Functions
    # are called from within the ANC HSIs. Which interventions are called depends on the mothers gestation and the
    # number of visits she has attended at the time each HSI runs (see ANC HSIs)

    def check_intervention_should_run_and_update_mni(self, person_id, int_1, int2):
        """
        This function is called to check if specific interventions within the ANC matrix should run for an individual.
        If the individual has received the intervention the appropriate amount of times per pregnancy then the
        intervention wont run again
        """
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        if int_1 not in mni[person_id]['anc_ints']:
            mni[person_id]['anc_ints'].append(int_1)
            return True

        elif int2 not in mni[person_id]['anc_ints']:
            mni[person_id]['anc_ints'].append(int2)
            return True

        elif int_1 and int2 in mni[person_id]['anc_ints']:
            return False

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

        # Delivery of the intervention is conditioned on the availability of the consumable and a random draw against a
        # probability that the intervention would be delivered (used to calibrate to SPA data- acts as proxy for
        # clinical quality)
        if self.rng.random_sample() < params['prob_intervention_delivered_urine_ds']:
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run='urine_dipstick_protein',
                                                                       hsi_event=hsi_event):

                # We use a temporary variable to store if proteinuria is detected
                proteinuria_diagnosed = True
                logger.info(key='anc_interventions', data={'mother': person_id, 'intervention': 'dipstick'})

        # The process is repeated for blood pressure monitoring- although not conditioned on consumables
        if self.rng.random_sample() < params['prob_intervention_delivered_bp']:
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run='blood_pressure_measurement',
                                                                       hsi_event=hsi_event):
                hypertension_diagnosed = True
                logger.info(key='anc_interventions', data={'mother': person_id, 'intervention': 'bp_measurement'})

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

        # Here we conduct screening and initiate treatment for depression as needed
        if 'Depression' in self.sim.modules:
            logger.info(key='anc_interventions', data={'mother': person_id, 'intervention': 'depression_screen'})

            if ~df.at[person_id, 'de_ever_diagnosed_depression']:
                self.sim.modules['Depression'].do_when_suspected_depression(person_id, hsi_event)

    def iron_and_folic_acid_supplementation(self, hsi_event):
        """This function contains the intervention iron and folic acid supplementation delivered during ANC.
        :param hsi_event: HSI event in which the function has been called
        """
        df = self.sim.population.props
        person_id = hsi_event.target
        params = self.current_parameters
        module_cons = self.item_codes_for_consumables_required_pregnancy

        if not df.at[person_id, 'ac_receiving_iron_folic_acid']:
            consumables = {
                'Intervention_Package_Code': {},
                'Item_Code': {module_cons['iron_folic_acid']['ifa']: self.get_approx_days_of_pregnancy(person_id)}}

            outcome_of_request_for_consumables = hsi_event.get_all_consumables(footprint=consumables)

            # As with previous interventions - condition on consumables and probability intervention is delivered
            if outcome_of_request_for_consumables and (self.rng.random_sample() < params['prob_intervention_'
                                                                                         'delivered_ifa']):
                logger.info(key='anc_interventions', data={'mother': person_id, 'intervention': 'iron_folic_acid'})

                if self.rng.random_sample() < params['prob_adherent_ifa']:
                    df.at[person_id, 'ac_receiving_iron_folic_acid'] = True

    def balance_energy_and_protein_supplementation(self, hsi_event):
        """This function contains the intervention balance energy and protein supplementation delivered during ANC.
        :param hsi_event: HSI event in which the function has been called
        """
        df = self.sim.population.props
        person_id = hsi_event.target
        module_cons = self.item_codes_for_consumables_required_pregnancy

        if not df.at[person_id, 'ac_receiving_bep_supplements']:
            consumables = {
                'Intervention_Package_Code': {},
                'Item_Code': {module_cons['balanced_energy_protein']['diet_supps']:
                                  self.get_approx_days_of_pregnancy(person_id)}}

            outcome_of_request_for_consumables = hsi_event.get_all_consumables(footprint=consumables)

            if outcome_of_request_for_consumables and (df.at[person_id, 'li_bmi'] == 1):
                df.at[person_id, 'ac_receiving_bep_supplements'] = True
                logger.info(key='anc_interventions', data={'mother': person_id, 'intervention': 'b_e_p'})

    def insecticide_treated_bednet(self, hsi_event):
        df = self.sim.population.props
        person_id = hsi_event.target
        params = self.current_parameters
        module_cons = self.item_codes_for_consumables_required_pregnancy

        outcome_of_request_for_consumables = hsi_event.get_all_consumables(item_codes=module_cons['itn']['bed_net'])

        # If available, women are provided with a bed net at ANC1. The effect of these nets is determined
        # through the malaria module - not yet coded. n.b any interventions involving non-obstetric diseases have been
        # discussed with Tara
        if outcome_of_request_for_consumables and (self.rng.random_sample() < params['prob_intervention_delivered'
                                                                                     '_llitn']):
            df.at[person_id, 'ac_itn_provided'] = True
            logger.info(key='anc_interventions', data={'mother': person_id, 'intervention': 'LLITN'})

    def tb_screening(self, hsi_event):
        pass

        # TODO: TB module in master is currently commented out, this is legacy code and a placeholder to ensure women
        #  are screened for TB
        # TB screening...
        # Currently we schedule women to the TB screening HSI in the TB module, however this may over-use resources so
        # possible the TB screening should also just live in this code
        # if 'tb' in self.sim.modules.keys():
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
        df = self.sim.population.props
        params = self.current_parameters
        module_cons = self.item_codes_for_consumables_required_pregnancy

        if not self.check_intervention_should_run_and_update_mni(person_id, 'tt_1', 'tt_2'):
            return

        # Define required consumables
        outcome_of_request_for_consumables = hsi_event.get_all_consumables(
            item_codes=list(module_cons['tetanus_toxoid'].values()))

        # If the consumables are available and the HCW will deliver the vaccine, the intervention is given
        if outcome_of_request_for_consumables and (self.rng.random_sample() < params['prob_intervention_delivered_tt']):
            df.at[person_id, 'ac_ttd_received'] += 1
            logger.info(key='anc_interventions', data={'mother': person_id, 'intervention': 'tt'})

    def calcium_supplementation(self, hsi_event):
        """This function contains the intervention calcium supplementation delivered during ANC.
        :param hsi_event: HSI event in which the function has been called
        """
        df = self.sim.population.props
        person_id = hsi_event.target
        module_cons = self.item_codes_for_consumables_required_pregnancy

        if not df.at[person_id, 'ac_receiving_calcium_supplements'] and \
            ((df.at[person_id, 'la_parity'] == 0) or (df.at[person_id, 'la_parity'] > 4)):

            consumables = {
                'Intervention_Package_Code': {},
                'Item_Code': {module_cons['calcium']['calcium_supps']:
                                  self.get_approx_days_of_pregnancy(person_id) * 3}}

            outcome_of_request_for_consumables = hsi_event.get_all_consumables(footprint=consumables)

            # If the consumables are available and the HCW will provide the tablets, the intervention is given
            if outcome_of_request_for_consumables:
                df.at[person_id, 'ac_receiving_calcium_supplements'] = True
                logger.info(key='anc_interventions', data={'mother': person_id, 'intervention': 'calcium'})

    def point_of_care_hb_testing(self, hsi_event):
        """
        This function contains the intervention point of care haemoglobin testing provided to women during ANC to detect
        anaemia during pregnancy
        :param hsi_event: HSI event in which the function has been called
        """
        person_id = hsi_event.target
        df = self.sim.population.props
        params = self.current_parameters

        if not self.check_intervention_should_run_and_update_mni(person_id, 'hb_1', 'hb_2'):
            return

        # We log all the consumables required above but we only condition the event test happening on the availability
        # of the test itself
        if self.rng.random_sample() < params['prob_intervention_delivered_poct']:
            logger.info(key='anc_interventions', data={'mother': person_id, 'intervention': 'hb_screen'})

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
        module_cons = self.item_codes_for_consumables_required_pregnancy

        if 'albend' in mni[person_id]['anc_ints']:
            return
        else:
            mni[person_id]['anc_ints'].append('albend')

            # We run this function to store the associated consumables with albendazole administration. This
            # intervention has no effect in the model due to limited evidence
            outcome_of_request_for_consumables = hsi_event.get_all_consumables(
                item_codes=list(module_cons['albendazole'].values()))

            # If the consumables are available and the HCW will provide the tablets, the intervention is given
            if outcome_of_request_for_consumables:
                logger.info(key='anc_interventions', data={'mother': person_id, 'intervention': 'albendazole'})

    def hep_b_testing(self, hsi_event):
        """
        This function contains the intervention Hepatitis B testing and is provided to women during ANC. As Hepatitis
        B is not modelled currently this intervention just maps consumables used during ANC
        :param hsi_event: HSI event in which the function has been called
        """
        person_id = hsi_event.target
        module_cons = self.item_codes_for_consumables_required_pregnancy

        if not self.check_intervention_should_run_and_update_mni(person_id, 'hep_b_1', 'hep_b_2'):
            return
        else:
            # This intervention is a place holder prior to the Hepatitis B module being codes
            outcome_of_request_for_consumables = hsi_event.get_all_consumables(
                item_codes=list(module_cons['hep_b_test'].values()))

            # We log all the consumables required above but we only condition the event test happening on the
            # availability of the test itself
            if outcome_of_request_for_consumables:
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
        module_cons = self.item_codes_for_consumables_required_pregnancy

        if not self.check_intervention_should_run_and_update_mni(person_id, 'syph_1', 'syph_2'):
            return
        else:

            # We log all the consumables required above but we only condition the event test happening on the
            # availability of the test itself
            if self.rng.random_sample() < params['prob_intervention_delivered_syph_test']:
                logger.info(key='anc_interventions', data={'mother': person_id, 'intervention': 'syphilis_test'})

                if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run='blood_test_syphilis',
                                                                           hsi_event=hsi_event):

                    # If the test detects that the mother has syphilis treatment is admistered and we assume her
                    # infection is treated
                    outcome_of_request_for_consumables_treatment = hsi_event.get_all_consumables(
                        item_codes=list(module_cons['syphilis_treatment'].values()))

                    if outcome_of_request_for_consumables_treatment:
                        df.at[person_id, 'ps_syphilis'] = False
                        logger.info(key='anc_interventions', data={'mother': person_id,
                                                                   'intervention': 'syphilis_treat'})

    def hiv_testing(self, hsi_event):
        """
        This function contains the scheduling for HIV testing and is provided to women during ANC.
        :param hsi_event: HSI event in which the function has been called
        """
        df = self.sim.population.props
        person_id = hsi_event.target
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        if 'hiv' in mni[person_id]['anc_ints']:
            return
        else:
            if 'Hiv' in self.sim.modules:
                if ~df.at[person_id, 'hv_diagnosed']:
                    mni[person_id]['anc_ints'].append('hiv')
                    self.sim.modules['HealthSystem'].schedule_hsi_event(
                       HSI_Hiv_TestAndRefer(person_id=person_id, module=self.sim.modules['Hiv']),
                       topen=self.sim.date,
                       tclose=None,
                       priority=0)

                logger.info(key='anc_interventions', data={'mother': person_id, 'intervention': 'hiv_screen'})

    def iptp_administration(self, hsi_event):
        """
        This function contains the intervention intermittent preventative treatment in pregnancy (for malaria) for women
        during ANC
        :param hsi_event: HSI event in which the function has been called
        """
        df = self.sim.population.props
        params = self.current_parameters
        person_id = hsi_event.target
        module_cons = self.item_codes_for_consumables_required_pregnancy

        if 'Malaria' in self.sim.modules:
            if not df.at[person_id, "ma_tx"] and df.at[person_id, "is_alive"]:

                # Test to ensure only 5 doses are able to be administered
                assert df.at[person_id, 'ac_doses_of_iptp_received'] < 6

                outcome_of_request_for_consumables = hsi_event.get_all_consumables(
                    item_codes=list(module_cons['iptp'].values()))

                if (self.rng.random_sample() < params['prob_intervention_delivered_iptp']) and \
                    outcome_of_request_for_consumables:

                    # IPTP is a single dose drug given at a number of time points during pregnancy. Therefore the
                    # number of doses received during this pregnancy are stored as an integer
                    df.at[person_id, 'ac_doses_of_iptp_received'] += 1
                    logger.info(key='anc_interventions', data={'mother': person_id, 'intervention': 'iptp'})

    def gdm_screening(self, hsi_event):
        """This function contains intervention of gestational diabetes screening during ANC. Screening is only conducted
         on women with pre-specified risk factors for the disease.
        :param hsi_event: HSI event in which the function has been called
        """
        df = self.sim.population.props
        params = self.current_parameters
        person_id = hsi_event.target
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        if 'gdm_screen' in mni[person_id]['anc_ints']:
            return
        else:

            # We check if this women has any of the key risk factors, if so they are sent for additional blood tests
            if df.at[person_id, 'li_bmi'] >= 4 or df.at[person_id, 'ps_prev_gest_diab'] or df.at[person_id,
                                                                                                 'ps_prev_stillbirth']:

                # If the test accurately detects a woman has gestational diabetes the consumables are recorded and
                # she is referred for treatment
                if (self.rng.random_sample() < params['prob_intervention_delivered_gdm_test']) and \
                    self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run='blood_test_glucose',
                                                                            hsi_event=hsi_event):

                    logger.info(key='anc_interventions', data={'mother': person_id, 'intervention': 'gdm_screen'})
                    mni[person_id]['anc_ints'].append('gdm_screen')

                    # We assume women with a positive GDM screen will be admitted (if they are not already receiving
                    # outpatient care)
                    if df.at[person_id, 'ac_gest_diab_on_treatment'] == 'none':

                        # Store onset after diagnosis as daly weight is tied to diagnosis
                        self.sim.modules['PregnancySupervisor'].store_dalys_in_mni(person_id, 'gest_diab_onset')

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
        params = self.current_parameters

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
        params = self.current_parameters

        # todo: seems like this is allowing things to run which is shouldnt

        date_difference = self.sim.date - df.at[individual_id, 'ac_date_next_contact']

        ga_for_anc_dict = {2: 20, 3: 26, 4: 30, 5: 34, 6: 36, 7: 38, 8: 40}

        # If women have died, are no longer pregnant, are in labour or more than a week has past since the HSI was
        # scheduled then it will not run
        if ~df.at[individual_id, 'is_alive'] \
            or ~df.at[individual_id, 'is_pregnant'] \
            or df.at[individual_id, 'la_currently_in_labour']\
            or df.at[individual_id, 'la_is_postpartum']\
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
        :returns: result of the FBC ['none', 'mild_mod', 'severe'] (STR)
        """
        df = self.sim.population.props
        person_id = hsi_event.target

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

    def treatment_of_anaemia_causing_deficiencies(self, individual_id, hsi_event):
        """
        This function contains treatment for deficiencies that may be contributing to anaemia in pregnancy. It is called
         by HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare for women admitted due to anaemia
        :param individual_id: individual_id
        :param hsi_event: HSI event in which the function has been called
        """
        df = self.sim.population.props
        module_cons = self.item_codes_for_consumables_required_pregnancy
        params = self.current_parameters
        pregnancy_deficiencies = self.sim.modules['PregnancySupervisor'].deficiencies_in_pregnancy
        store_dalys_in_mni = self.sim.modules['PregnancySupervisor'].store_dalys_in_mni

        # If iron or folate deficient, a woman will need to take additional daily supplements. If B12 deficient this
        # should occur monthly
        # todo: this isnt very tidy, also no consumables for some, also remove?
        consumables = {
            'Intervention_Package_Code': {},
            'Item_Code': {module_cons['iron_treatment']['ferrous_sulphate']:
                              self.get_approx_days_of_pregnancy(individual_id),
                          module_cons['b12_treatment']['vit_b12']:
                              self.get_approx_days_of_pregnancy(individual_id),
                          module_cons['b12_treatment']['vit_b12']:
                              np.ceil(self.get_approx_days_of_pregnancy(individual_id) / 31)}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=hsi_event,
            cons_req_as_footprint=consumables)

        # Treatment is provided dependent on deficiencies present
        if pregnancy_deficiencies.has_any([individual_id], 'iron', first=True):

            # If the treatment is available we assume it has two effect, first in resolving the deficiency and second in
            # resolving a womans current anaemia
            if outcome_of_request_for_consumables['Item_Code'][module_cons['iron_treatment']['ferrous_sulphate']]:
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
            if outcome_of_request_for_consumables['Item_Code'][module_cons['b12_treatment']['vit_b12']]:
                if self.rng.random_sample() < params['effect_of_folate_replacement_for_resolving_anaemia']:

                    if df.at[individual_id, "ps_anaemia_in_pregnancy"] != 'none':
                        store_dalys_in_mni(individual_id, f'{df.at[individual_id, "ps_anaemia_in_pregnancy"]}_'
                                                          f'anaemia_resolution')

                        df.at[individual_id, 'ps_anaemia_in_pregnancy'] = 'none'

                if self.rng.random_sample() < params['effect_of_folate_replacement_for_resolving_folate_def']:
                    pregnancy_deficiencies.unset([individual_id], 'folate')

        if pregnancy_deficiencies.has_any([individual_id], 'b12', first=True):
            if outcome_of_request_for_consumables['Item_Code'][module_cons['b12_treatment']['vit_b12']]:
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
        params = self.current_parameters
        module_cons = self.item_codes_for_consumables_required_pregnancy
        store_dalys_in_mni = self.sim.modules['PregnancySupervisor'].store_dalys_in_mni

        consumables = {
            'Intervention_Package_Code': {},
            'Item_Code': {module_cons['iron_folic_acid']['ifa']: self.get_approx_days_of_pregnancy(individual_id)}}

        outcome_of_request_for_consumables = hsi_event.get_all_consumables(footprint=consumables)

        # Start iron and folic acid treatment
        if outcome_of_request_for_consumables:
            if self.rng.random_sample() < params['prob_adherent_ifa']:
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
        module_cons = self.item_codes_for_consumables_required_pregnancy
        params = self.current_parameters
        store_dalys_in_mni = self.sim.modules['PregnancySupervisor'].store_dalys_in_mni

        outcome_of_request_for_consumables = hsi_event.get_all_consumables(
            item_codes=list(module_cons['blood_transfusion'].values()))

        # If the blood is available we assume the intervention can be delivered
        if outcome_of_request_for_consumables:

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
        module_cons = self.item_codes_for_consumables_required_pregnancy

        consumables = {
            'Intervention_Package_Code': {},
            'Item_Code': {module_cons['oral_antihypertensives']['methyldopa']:
                            (4 * self.get_approx_days_of_pregnancy(individual_id))}}

        outcome_of_request_for_consumables = hsi_event.get_all_consumables(footprint=consumables)

        # If they are available then the woman is started on treatment
        if outcome_of_request_for_consumables:
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
        module_cons = self.item_codes_for_consumables_required_pregnancy

        outcome_of_request_for_consumables = hsi_event.get_all_consumables(
            item_codes=list(module_cons['iv_antihypertensives'].values()))

        # If they hydralazine is available we assume the intervention can be delivered
        if outcome_of_request_for_consumables:

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
        module_cons = self.item_codes_for_consumables_required_pregnancy

        outcome_of_request_for_consumables = hsi_event.get_all_consumables(
            item_codes=list(module_cons['magnesium_sulfate'].values()))

        # If they hydralazine is available we assume the intervention can be delivered
        if outcome_of_request_for_consumables:
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
        module_cons = self.item_codes_for_consumables_required_pregnancy

        outcome_of_request_for_consumables = hsi_event.get_all_consumables(
            item_codes=list(module_cons['abx_for_prom'].values()))

        # If the antibiotics are available we assume the intervention can be delivered
        if outcome_of_request_for_consumables:
            df.at[individual_id, 'ac_received_abx_for_prom'] = True
            logger.debug(key='msg', data=f'Mother {individual_id} has received antibiotics following admission due to '
                                         f'PROM')

    def ectopic_pregnancy_treatment_doesnt_run(self, hsi_event):
        """
        This function is called within HSI_CareOfWomenDuringPregnancy_TreatmentForEctopicPregnancy if the event cannot
        run/the intervention cannot be delivered. This ensures that women with ectopic pregnancies that haven't ruptured
        will experience rupture and risk of death without treatment
        :param individual_id: individual_id
        :return:
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

    def __init__(self, module, person_id, facility_level_of_this_hsi):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_FirstAntenatalCareContact'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'AntenatalFirst': 1})
        self.ACCEPTED_FACILITY_LEVEL = facility_level_of_this_hsi
        assert self.ACCEPTED_FACILITY_LEVEL != 0
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

            anc_rows = {'ga_anc_one': df.at[person_id, 'ps_gestational_age_in_weeks'],
                        'anc_ints': []}
            self.sim.modules['PregnancySupervisor'].mother_and_newborn_info[person_id].update(anc_rows)

            logger.info(key='anc1', data={'mother': person_id,
                                          'gestation': df.at[person_id, 'ps_gestational_age_in_weeks']})

            # We generate the facility type that this HSI is occurring at (dependent on facility level) - we currently
            # assume women will present to the same facility level and type for any future ANC visits

            if self.ACCEPTED_FACILITY_LEVEL == 1:
                # Assume a 50/50 chance of health centre or hospital in level 1, however this will need editing
                facility_type = self.module.rng.choice(['health_centre', 'hospital'], p=[0.5, 0.5])
                df.at[person_id, 'ac_facility_type'] = facility_type
                logger.info(key='anc_facility_type', data=f'{facility_type}')

            elif self.ACCEPTED_FACILITY_LEVEL > 1:
                logger.info(key='anc_facility_type', data='hospital')
                df.at[person_id, 'ac_facility_type'] = 'hospital'

            logger.debug(key='message', data=f'mother {person_id} presented for ANC1 at a '
                                             f'{df.at[person_id, "ac_facility_type"]} at gestation '
                                             f'{df.at[person_id, "ps_gestational_age_in_weeks"]} ')

            # We add a visit to a rolling total of ANC visits in this pregnancy
            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            # And ensure only women whose first contact with ANC services are attending this event
            assert not pd.isnull(mother.ps_gestational_age_in_weeks)
            assert mother.ps_gestational_age_in_weeks >= 7

            #  =================================== INTERVENTIONS ====================================================
            # First all women, regardless of ANC contact or gestation, undergo urine and blood pressure measurement
            # and depression screening
            self.module.screening_interventions_delivered_at_every_contact(hsi_event=self)

            # Next, all women attending their first ANC receive the following interventions, regardless of gestational
            # age at presentation
            self.module.iron_and_folic_acid_supplementation(hsi_event=self)
            self.module.balance_energy_and_protein_supplementation(hsi_event=self)
            self.module.insecticide_treated_bednet(hsi_event=self)
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
            assert not pd.isnull(mother.ps_gestational_age_in_weeks)
            assert mother.ps_gestational_age_in_weeks >= 19

            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            #  =================================== INTERVENTIONS ====================================================
            # First we administer the administer the interventions all women will receive at this contact regardless of
            # gestational age
            self.module.interventions_delivered_each_visit_from_anc2(hsi_event=self)
            self.module.tetanus_vaccination(hsi_event=self)

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
            assert not pd.isnull(mother.ps_gestational_age_in_weeks)
            assert mother.ps_gestational_age_in_weeks >= 25

            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            #  =================================== INTERVENTIONS ====================================================
            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)
            self.module.interventions_delivered_each_visit_from_anc2(hsi_event=self)

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

    def __init__(self, module, person_id, facility_level_of_this_hsi):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_FourthAntenatalCareContact'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
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
            assert not pd.isnull(mother.ps_gestational_age_in_weeks)
            assert mother.ps_gestational_age_in_weeks >= 29

            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            #  =================================== INTERVENTIONS ====================================================
            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)
            self.module.interventions_delivered_each_visit_from_anc2(hsi_event=self)

            if mother.ps_gestational_age_in_weeks < 40:
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=5,
                                                     recommended_gestation_next_anc=gest_age_next_contact,
                                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)

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

            assert not pd.isnull(mother.ps_gestational_age_in_weeks)
            assert mother.ps_gestational_age_in_weeks >= 33
            assert mother.ac_total_anc_visits_current_pregnancy == 4

            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            #  =================================== INTERVENTIONS ====================================================
            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)
            self.module.interventions_delivered_each_visit_from_anc2(hsi_event=self)

            if mother.ps_gestational_age_in_weeks < 40:
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=6,
                                                     recommended_gestation_next_anc=gest_age_next_contact,
                                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)

            if mother.ps_gestational_age_in_weeks < 36:
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
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
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
            assert not pd.isnull(mother.ps_gestational_age_in_weeks)
            assert mother.ps_gestational_age_in_weeks >= 35

            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)

            #  =================================== INTERVENTIONS ====================================================
            self.module.interventions_delivered_each_visit_from_anc2(hsi_event=self)

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
            assert not pd.isnull(mother.ps_gestational_age_in_weeks)
            assert mother.ps_gestational_age_in_weeks >= 37

            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            #  =================================== INTERVENTIONS ====================================================
            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)
            self.module.interventions_delivered_each_visit_from_anc2(hsi_event=self)

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
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
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
            assert not pd.isnull(mother.ps_gestational_age_in_weeks)
            assert mother.ps_gestational_age_in_weeks >= 39

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


class HSI_CareOfWomenDuringPregnancy_PresentsForInductionOfLabour(HSI_Event, IndividualScopeEventMixin):
    """
    This is HSI_CareOfWomenDuringPregnancy_PresentsForInductionOfLabour. It is schedule by the PregnancySupervisor Event
    for women who present to the health system for induction as their labour has progressed longer than expected.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'HSI_CareOfWomenDuringPregnancy_PresentsForInductionOfLabour'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        # If the woman is no longer alive, pregnant is in labour or is an inpatient already then the event doesnt run
        if not df.at[person_id, 'is_alive'] or not df.at[person_id, 'is_pregnant'] or \
            df.at[person_id, 'la_currently_in_labour'] or df.at[person_id, 'hs_is_inpatient']:
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
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        # TODO: remove and schedule through ED

        if not df.at[person_id, 'is_alive'] or not df.at[person_id, 'is_pregnant']:
            return

        if ~df.at[person_id, 'hs_is_inpatient'] and ~df.at[person_id, 'la_currently_in_labour']:
            logger.debug(key='msg', data=f'Mother {person_id} has presented at HSI_CareOfWomenDuringPregnancy_Maternal'
                                         f'EmergencyAssessment to seek care for a complication ')

            facility_level = int(self.module.rng.choice([1, 2], p=[0.5, 0.5]))

            admission = HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare(
                self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id,
                facility_level_this_hsi=facility_level)

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
        params = self.module.current_parameters
        mother = df.loc[person_id]
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

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
                assert fbc_result == 'none' or 'mild' or 'moderate' or 'severe'

                # If the result returns none, anaemia has not been detected via an FBC and the woman is discharged
                # without treatment
                if fbc_result == 'none':
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

                # Women are started on oral antihypertensives
                if ~mother.ac_gest_htn_on_treatment:
                    self.module.initiate_maintenance_anti_hypertensive_treatment(person_id, self)

                # And are given intravenous magnesium sulfate which reduces risk of death from eclampsia and reduces a
                # womans risk of progressing from severe pre-eclampsia to eclampsia during the intrapartum period
                self.module.treatment_for_severe_pre_eclampsia_or_eclampsia(person_id,
                                                                            hsi_event=self)
                # intravenous antihypertensives are also given
                self.module.initiate_treatment_for_severe_hypertension(person_id, self)

                # Finally This property stores what type of delivery this woman is being admitted for

                delivery_mode = ['induction_now', 'avd_now', 'caesarean_now']

                if mother.ps_htn_disorders == 'eclampsia':

                    df.at[person_id, 'ac_admitted_for_immediate_delivery'] = self.module.rng.choice(
                        delivery_mode,  p=params['prob_delivery_modes_ec'])

                elif mother.ps_htn_disorders == 'severe_pre_eclamp':
                    df.at[person_id, 'ac_admitted_for_immediate_delivery'] = self.module.rng.choice(
                        delivery_mode, p=params['prob_delivery_modes_spe'])

                logger.debug(key='msg', data=f'{person_id} will be admitted for delivery  due to '
                                             f'{mother.ps_htn_disorders}')

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

                    # todo:delete
                    mni[person_id]['cs_indication'] = 'an_aph_pa'

                # ---------------------- APH SECONDARY TO PLACENTA PRAEVIA -----------------------------------------
                if mother.ps_placenta_praevia:
                    # The treatment plan for a woman with placenta praevia is dependent on both the severity of the
                    # bleed and her current gestation at the time of bleeding

                    if mother.ps_antepartum_haemorrhage == 'severe':
                        # Women experiencing severe bleeding are admitted immediately for caesarean section
                        df.at[person_id, 'ac_admitted_for_immediate_delivery'] = 'caesarean_now'
                        logger.debug(key='msg', data=f'{person_id} will be admitted for caesarean due to APH')

                        # todo:delete
                        mni[person_id]['cs_indication'] = 'an_aph_pp'

                    elif (mother.ps_antepartum_haemorrhage != 'severe') and (mother.ps_gestational_age_in_weeks >= 37):
                        # Women experiencing mild or moderate bleeding but who are around term gestation are admitted
                        # for caesarean
                        df.at[person_id, 'ac_admitted_for_immediate_delivery'] = 'caesarean_now'
                        logger.debug(key='msg', data=f'{person_id} will be admitted for caesarean due to APH')

                        # todo:delete
                        mni[person_id]['cs_indication'] = 'an_aph_pp'

                    elif (mother.ps_antepartum_haemorrhage != 'severe') and (mother.ps_gestational_age_in_weeks < 37):
                        # Women with more mild bleeding remain as inpatients until their gestation has increased and
                        # then will be delivered by caesarean - (no risk of death associated with mild/moderate bleeds)
                        df.at[person_id, 'ac_admitted_for_immediate_delivery'] = 'caesarean_future'
                        logger.debug(key='msg', data=f'{person_id} will be admitted for caesarean when her gestation '
                                                     f'has increased due to APH')
                        # todo:delete
                        mni[person_id]['cs_indication'] = 'an_aph_pp'

                        # self.module.antenatal_blood_transfusion(person_id, self, cause='antepartum_haem')

                assert df.at[person_id, 'ac_admitted_for_immediate_delivery'] != 'none'

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
                    logger.debug(key='msg', data=f'{person_id} will be admitted for induction due to prom/chorio')

                # Otherwise they may stay as an inpatient until their gestation as increased prior to delivery
                elif mother.ps_gestational_age_in_weeks < 34:
                    df.at[person_id, 'ac_admitted_for_immediate_delivery'] = 'induction_future'
                    logger.debug(key='msg', data=f'{person_id} will be admitted for induction when her gestation'
                                                 f' has increase due prom/chorio')

            # ============================== INITIATE TREATMENT FOR CHORIOAMNIONITIS ==================================
            if mother.ps_chorioamnionitis:
                # TODO: REMOVE THIS FUNCTION AS TREATMENT IS DELIVERED ENTIRELY IN LABOUR MODULE
                #self.module.antibiotics_for_chorioamnionitis(person_id, self)
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
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
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

            # If the test determines the woman is no longer anaemia then no further action is taken at this time
            if fbc_result == 'none':
                logger.debug(key='message', data=f'Mother {person_id} has not had anaemia detected via an FBC')

            # If she is determined to still be anaemic she is admitted for additional treatment via the inpatient event
            elif fbc_result == 'mild' or fbc_result == 'moderate' or fbc_result == 'severe':

                facility_level = int(self.module.rng.choice([1, 2], p=[0.5, 0.5]))

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
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        mother = df.loc[person_id]
        module_cons = self.module.item_codes_for_consumables_required_pregnancy

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
                    consumables = {
                        'Intervention_Package_Code': {},
                        'Item_Code': {
                            module_cons['oral_diabetic_treatment']['glibenclamide']:
                                4 * self.module.get_approx_days_of_pregnancy(person_id)}}

                    outcome_of_request_for_consumables = self.get_all_consumables(footprint=consumables)

                    # If the meds are available women are started on that treatment
                    if outcome_of_request_for_consumables:
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
                    consumables = {
                        'Intervention_Package_Code': {},
                        'Item_Code': {
                            module_cons['insulin_treatment']['insulin']:
                                4 * self.module.get_approx_days_of_pregnancy(person_id)}}

                    outcome_of_request_for_consumables = self.get_all_consumables(footprint=consumables)

                    if outcome_of_request_for_consumables:
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
        self.ACCEPTED_FACILITY_LEVEL = 1  # any hospital?
        self.ALERT_OTHER_DISEASES = []
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 3})

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]
        params = self.module.current_parameters
        abortion_complications = self.sim.modules['PregnancySupervisor'].abortion_complications
        module_cons = self.module.item_codes_for_consumables_required_pregnancy

        # todo: simplify treatment? remove bitset?
        # todo: this should be determiend on consumable availibility. i.e. if septic, and abx avail then treatment etc.

        if not mother.is_alive or not abortion_complications.has_any([person_id], 'sepsis', 'haemorrhage', 'injury',
                                                                     'other', first=True):
            return

        self.get_all_consumables(item_codes=list(module_cons['post_abortion_care'].values()))

        random_draw = self.module.rng.choice(['d_and_c', 'mva', 'misoprostol'], p=params['prob_evac_procedure_pac'])
        self.module.pac_interventions.set(person_id, random_draw)

        # Women who are septic following their abortion are given antibiotics
        if abortion_complications.has_any([person_id], 'sepsis', first=True):
            self.module.pac_interventions.set(person_id, 'antibiotics')

        # Minor injuries following induced abortion are treated
        if abortion_complications.has_any([person_id], 'injury', first=True):
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
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 5})

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        module_cons = self.module.item_codes_for_consumables_required_pregnancy
        mother = df.loc[person_id]

        if not mother.is_alive or (mother.ps_ectopic_pregnancy == 'none'):
            return

        logger.debug(key='message', data='This is HSI_CareOfWomenDuringPregnancy_TreatmentForEctopicPregnancy, '
                                         f'person {person_id} has been diagnosed with ectopic pregnancy after '
                                         f'presenting and will now undergo treatment')

        # We define the required consumables and check their availability
        outcome_of_consumables_check = self.get_all_consumables(
            item_codes=list(module_cons['ectopic_pregnancy'].values()))

        # If they are available then treatment can go ahead
        if outcome_of_consumables_check:
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
            if df.at[person_id, 'ps_ectopic_pregnancy'] == 'not_ruptured':
                self.module.ectopic_pregnancy_treatment_doesnt_run(person_id)
                logger.debug(key='msg', data=f'Mother {person_id} could not receive treatment due to insufficient '
                                             f'consumables')

    def never_ran(self):
        self.module.ectopic_pregnancy_treatment_doesnt_run(self)

    def did_not_run(self):
        self.module.ectopic_pregnancy_treatment_doesnt_run(self)
        return False

    def not_available(self):
        self.module.ectopic_pregnancy_treatment_doesnt_run(self)


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

