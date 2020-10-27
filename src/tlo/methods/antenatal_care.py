from pathlib import Path

import pandas as pd
import numpy as np

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType
from tlo.methods import Metadata
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.dxmanager import DxTest
from tlo.methods.hiv import HSI_Hiv_PresentsForCareWithSymptoms
from tlo.methods.labour import HSI_Labour_ElectiveCaesareanSection
from tlo.methods.tb import HSI_TbScreening
from tlo.util import BitsetHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CareOfWomenDuringPregnancy(Module):
    """This is the CareOfWomenDuringPregnancy module. This module houses all HSIs related to care during the antenatal
    period of pregnancy. Currently the module houses all 8 antenatal care contacts and manages the scheduling of all
    additional antenatal care contacts after visit 1 (which is scheduled in the pregnancy supervisor module.
    Additionally this module houses additional care HSIs for women whos pregnancy is complicated by hypertension,
    anaemia and gestational diabetes. The module is incomplete as additional HSIs for post abortion care are not included.
    """

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # This dictionary is used by to track the frequency of certain events in the module which are processed by the
        # logging event
        self.anc_tracker = dict()

    METADATA = {
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN,
        Metadata.USES_SYMPTOMMANAGER
    }

    PARAMETERS = {
        'prob_evac_procedure_mild_ac': Parameter(
            Types.LIST, 'list of probabilities for method of evacuation for a woman with mild abortion complications'),
        'prob_evac_procedure_moderate_ac': Parameter(
            Types.LIST, 'list of probabilities for method of evacuation for a woman with moderate abortion '
                        'complications'),
        'prob_evac_procedure_severe_ac': Parameter(
            Types.LIST, 'list of probabilities for method of evacuation for a woman with severe abortion '
                        'complications'),
        'prob_analgesia_mild_ac': Parameter(
            Types.REAL, 'probability a woman with mild abortion complications will be given analgesia'),
        'prob_analgesia_moderate_ac': Parameter(
            Types.REAL, 'probability a woman with moderate abortion complications will be given analgesia'),
        'prob_analgesia_severe_ac': Parameter(
            Types.REAL, 'probability a woman with severe abortion complications will be given analgesia'),
        'prob_antibiotics_mild_ac': Parameter(
            Types.REAL, 'probability a woman with mild abortion complications will be given antibiotics'),
        'prob_antibiotics_moderate_ac': Parameter(
            Types.REAL, 'probability a woman with moderate abortion complications will be given antibiotics'),
        'prob_antibiotics_severe_ac': Parameter(
            Types.REAL, 'probability a woman with severe abortion complications will be given antibiotics'),
        'prob_blood_transfusion_mild_ac': Parameter(
            Types.REAL, 'probability a woman with mild abortion complications will be given a blood transfusion'),
        'prob_blood_transfusion_moderate_ac': Parameter(
            Types.REAL, 'probability a woman with moderate abortion complications will be given a blood transfusion'),
        'prob_blood_transfusion_severe_ac': Parameter(
            Types.REAL, 'probability a woman with severe abortion complications will be given a blood transfusion'),
        'prob_anc_continues': Parameter(
            Types.REAL, 'probability a woman will return for a subsequent ANC appointment'),
        'prob_bp_check': Parameter(
            Types.REAL, 'probability a woman will have her blood pressure checked during antenatal care'),
        'prob_urine_dipstick': Parameter(
            Types.REAL, 'probability a woman will have her urine dipsticked during antenatal care'),
        'prob_start_iron_folic_acid': Parameter(
            Types.REAL, 'probability a woman will receive a course of iron and folic acid during antenatal care'),
        'prob_start_diet_supps_acid': Parameter(
            Types.REAL, 'probability a woman will receive a course of nutritional supplements during antenatal care'),
        'prob_poc_hb_test': Parameter(
            Types.REAL, 'probability a woman will receive a blood test during antenatal care'),
        'prob_start_calcium_supp': Parameter(
            Types.REAL, 'probability a woman will receive a course of calcium supplements during antenatal care'),
        'prob_albendazole': Parameter(
            Types.REAL, 'probability a woman will receive a dose of albendazole for deworming in pregnancy'),
        'sensitivity_bp_monitoring': Parameter(
            Types.REAL, 'sensitivity of blood pressure monitoring to detect hypertension'),
        'specificity_bp_monitoring': Parameter(
            Types.REAL, 'specificity of blood pressure monitoring to detect hypertension'),
        'sensitivity_urine_protein_1_plus': Parameter(
            Types.REAL, 'sensitivity of a urine dipstick test to detect proteinuria at 1+'),
        'specificity_urine_protein_1_plus': Parameter(
            Types.REAL, 'specificity of a urine dipstick test to detect proteinuia at 1+'),
        'sensitivity_urine_protein_3_plus': Parameter(
            Types.REAL, 'sensitivity of a urine dipstick test to detect proteinuria at 3+'),
        'specificity_urine_protein_3_plus': Parameter(
            Types.REAL, 'specificity of a urine dipstick test to detect proteinuia at 3+'),
        'sensitivity_urine_glucose': Parameter(
            Types.REAL, 'sensitivity of a urine dipstick test to detect glycosuria'),
        'specificity_urine_glucose': Parameter(
            Types.REAL, 'specificity of a urine dipstick test to detect glycosuria'),
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

    }

    PROPERTIES = {
        'ac_total_anc_visits_current_pregnancy': Property(
            Types.INT,
            'rolling total of antenatal visits this woman has attended during her pregnancy'),
        'ac_to_be_admitted':Property(
            Types.BOOL, 'Whether this woman requires admission following an ANC visit'),
        'ac_receiving_iron_folic_acid': Property(
            Types.BOOL,
            'whether this woman is receiving daily iron & folic acid supplementation'),
        'ac_date_ifa_runs_out': Property(
            Types.DATE,
            'Date on which this woman is no longer taking her iron & folic acid tablets '),
        'ac_receiving_diet_supplements': Property(
            Types.BOOL,
            'whether this woman is receiving daily food supplementation'),
        'ac_date_ds_runs_out': Property(
            Types.DATE,
            'Date on which this woman is no longer taking her iron & folic acid tablets'),
        'ac_receiving_calcium_supplements': Property(
            Types.BOOL,
            'whether this woman is receiving daily calcium supplementation'),
        'ac_date_cal_runs_out': Property(
            Types.DATE,
            'Date on which this woman is no longer taking her calcium tablets'),
        'ac_doses_of_iptp_received': Property(
            Types.INT,
            'Number of doses of intermittent preventative treatment in pregnancy received during this pregnancy'),
        'ac_itn_provided': Property(
            Types.BOOL,
            'Whether this woman is provided with an insecticide treated bed net during the appropriate ANC visit'),
        'ac_ttd_received': Property(
            Types.INT,
            'Number of doses of tetanus toxoid administered during this pregnancy'),
        'ac_gest_htn_on_treatment': Property(
            Types.BOOL,
            'Whether this woman has been initiated on treatment for gestational hypertension'),
        'ac_ectopic_pregnancy_treated': Property(
            Types.BOOL,
            'Whether this woman has received treatment for an ectopic pregnancy'),
        'ac_post_abortion_care_interventions': Property(
            Types.INT,
            'bitset list of interventions delivered to a woman undergoing post abortion care '),
        'ac_severe_pre_eclampsia_treatment': Property(
            Types.BOOL, 'Whether this woman has received treatment for severe pre-eclampsia'),
        'ac_eclampsia_treatment': Property(
            Types.BOOL, 'Whether this woman has received treatment for eclampsia'),

    }

    def read_parameters(self, data_folder):
        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_AntenatalCare.xlsx',
                            sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)

        params = self.parameters

        params['ac_linear_equations'] = {
                'anc_continues': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_anc_continues']),
        }

    def initialise_population(self, population):

        df = population.props

        # todo: these variables need to be reset on pregnancy loss?
        df.loc[df.is_alive, 'ac_total_anc_visits_current_pregnancy'] = 0
        df.loc[df.is_alive, 'ac_to_be_admitted'] = False
        df.loc[df.is_alive, 'ac_receiving_iron_folic_acid'] = False
        df.loc[df.is_alive, 'ac_date_ifa_runs_out'] = pd.NaT
        df.loc[df.is_alive, 'ac_receiving_diet_supplements'] = False
        df.loc[df.is_alive, 'ac_date_ds_runs_out'] = pd.NaT
        df.loc[df.is_alive, 'ac_receiving_calcium_supplements'] = False
        df.loc[df.is_alive, 'ac_date_cal_runs_out'] = pd.NaT
        df.loc[df.is_alive, 'ac_doses_of_iptp_received'] = 0
        df.loc[df.is_alive, 'ac_itn_provided'] = False
        df.loc[df.is_alive, 'ac_ttd_received'] = 0
        df.loc[df.is_alive, 'ac_gest_htn_on_treatment'] = False
        df.loc[df.is_alive, 'ac_ectopic_pregnancy_treated'] = False
        df.loc[df.is_alive, 'ac_severe_pre_eclampsia_treatment'] = False
        df.loc[df.is_alive, 'ac_eclampsia_treatment'] = False

        self.pac_interventions = BitsetHandler(self.sim.population, 'ac_post_abortion_care_interventions',
                                 ['mva', 'd_and_c', 'misoprostol', 'analgesia', 'antibiotics', 'blood_products'])

    def initialise_simulation(self, sim):
        sim.schedule_event(AntenatalCareLoggingEvent(self),
                           sim.date + DateOffset(years=1))

        # Populate the tracker
        self.anc_tracker = {'total_first_anc_visits': 0, 'cumm_ga_at_anc1': 0, 'total_anc1_first_trimester': 0,
                            'anc8+': 0, 'timely_ANC3': 0, 'diet_supp_6_months':0}

        # DX_TESTS
        params = self.parameters

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            blood_pressure_measurement=DxTest(
                property='ps_htn_disorders', target_categories=['gest_htn', 'mild_pre_eclamp', 'severe_pre_eclamp',
                                                                'eclampsia'],
                sensitivity=params['sensitivity_bp_monitoring'],
                specificity=params['specificity_bp_monitoring']),

            urine_dipstick_protein_1_plus=DxTest(
                property='ps_htn_disorders', target_categories=['mild_pre_eclamp'],
                sensitivity=params['sensitivity_urine_protein_1_plus'],
                specificity=params['specificity_urine_protein_1_plus']),

            urine_dipstick_protein_3_plus=DxTest(
                property='ps_htn_disorders', target_categories=['severe_pre_eclamp'],
                sensitivity=params['sensitivity_urine_protein_3_plus'],
                specificity=params['specificity_urine_protein_3_plus']),

            # TODO: in reality a urine dipstick is one test which would detect pre-eclampsia as either mild or severe
            #  from the amount of protein detected in urine (mild +) severe (+++).

            # TODO: Categorical function of dx_test doesnt allow one test to return positive for multiple catagories


            #    urine_dipstick_sugars=DxTest(
            #        property='ps_gest_diab',
            #        sensitivity=params['sensitivity_urine_glucose'],
            #        specificity=params['specificity_urine_glucose']),

            #    urine_dipstick_infection=DxTest(
            #        property='???',
            #        sensitivity=0.9,
            #        specificity=0.9),

            point_of_care_hb_test=DxTest(
                property='ps_anaemia_in_pregnancy', target_categories=['non_severe', 'severe'],
                sensitivity=params['sensitivity_poc_hb_test'],
                specificity=params['specificity_poc_hb_test']),

            full_blood_count_hb=DxTest(
                property='ps_anaemia_in_pregnancy', target_categories=['non_severe', 'severe'],
                sensitivity=params['sensitivity_fbc_hb_test'],
                specificity=params['specificity_fbc_hb_test']),


            blood_test_haemoglobin=DxTest(
                property='ps_anaemia_in_pregnancy',
                sensitivity=params['sensitivity_fbc_hb_test'],
                specificity=params['specificity_poc_hb_test']),

            blood_test_glucose=DxTest(
                property='ps_gest_diab',
                sensitivity=params['sensitivity_blood_test_glucose'],
                specificity=params['specificity_blood_test_glucose'])
            )

    def on_birth(self, mother_id, child_id):
        df = self.sim.population.props

        df.at[child_id, 'ac_total_anc_visits_current_pregnancy'] = 0
        df.at[child_id, 'ac_to_be_admitted'] = False
        df.at[child_id, 'ac_receiving_iron_folic_acid'] = False
        df.at[child_id, 'ac_date_ifa_runs_out'] = pd.NaT
        df.at[child_id, 'ac_receiving_diet_supplements'] = False
        df.at[child_id, 'ac_date_ds_runs_out'] = pd.NaT
        df.at[child_id, 'ac_receiving_calcium_supplements'] = False
        df.at[child_id, 'ac_date_cal_runs_out'] = pd.NaT
        df.at[child_id, 'ac_doses_of_iptp_received'] = 0
        df.at[child_id, 'ac_itn_provided'] = False
        df.at[child_id, 'ac_ttd_received'] = 0
        df.at[child_id, 'ac_gest_htn_on_treatment'] = False
        df.at[child_id, 'ac_ectopic_pregnancy_treated'] = False
        df.at[child_id, 'ac_severe_pre_eclampsia_treatment'] = False
        df.at[child_id, 'ac_eclampsia_treatment'] = False

        # Run a check at birth to make sure no women exceed 8 visits, which shouldn't occur through this logic
        assert df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] < 9

        # We store the total number of ANC vists a woman achieves prior to her birth in this logging dataframe

        total_anc_visit_count={'person_id': mother_id,
                               'age': df.at[mother_id, 'age_years'],
                               'date_of_delivery': self.sim.date,
                               'total_anc': df.at[mother_id, 'ac_total_anc_visits_current_pregnancy']}

        logger.info(key='anc_count_on_birth', data=total_anc_visit_count, description='A dictionary containg the '
                                                                                      'number of ANC visits each woman'
                                                                                      'has on birth')

        # And then reset the variable
        df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] = 0
        df.at[mother_id, 'ac_receiving_iron_folic_acid'] = False
        df.at[mother_id, 'ac_date_ifa_runs_out'] = pd.NaT
        df.at[mother_id, 'ac_receiving_diet_supplements'] = False
        df.at[mother_id, 'ac_date_ds_runs_out'] = pd.NaT
        df.at[mother_id, 'ac_receiving_calcium_supplements'] = False
        df.at[mother_id, 'ac_date_cal_runs_out'] = pd.NaT
        df.at[mother_id, 'ac_doses_of_iptp_received'] = 0
        # TODO: check tara doesnt need this property to last beyond delivery
        df.at[mother_id, 'ac_itn_provided'] = False
        df.at[mother_id, 'ac_ttd_received'] = 0
        # TODO: not sure this variable should be reset to 0 (check with tara)
        df.at[mother_id, 'ac_gest_htn_on_treatment'] = False
        # todo: this will be used postnatally so needs to stay

    def on_hsi_alert(self, person_id, treatment_id):
        logger.debug(key='message', data=f'This is CareOfWomenDuringPregnancy, being alerted about a health system '
        f'interaction person {person_id} for: {treatment_id}')

    def determine_gestational_age_for_next_contact(self, person_id):
        """This function determines when a woman should be scheduled to return to ANC based on her current gestation at
        her most recent visit"""
        df = self.sim.population.props
        mother = df.loc[person_id]

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

        # TODO: this is a quick fix for women who present very late to ANC so that they get some treatment coverage
        #  before they likely give birth
        elif mother.ps_gestational_age_in_weeks >= 40:
            recommended_gestation_next_anc = 45

        return recommended_gestation_next_anc

    # ================================= ANC INTERVENTION FUNCTIONS ====================================================

    def interventions_delivered_at_every_contact(self, hsi_event):
        """This function houses all the interventions that should be delivered at every ANC contact regardless of
        gestational age including blood pressure measurement, urine dipstick, administration of iron and folic acid
        and dietary supplementation"""
        person_id = hsi_event.target
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        df = self.sim.population.props
        params = self.parameters

        # First we define the consumables that are required for these interventions to be delivered

        # This section of code calculates the estimated number of days between this visit and the next, to determine the
        # required number of doses of daily drugs administered to women during this contact

        #  TODO: The following 4 lines fix a bug i cant work out why its happening
        if self.determine_gestational_age_for_next_contact(person_id) is None \
          and df.at[person_id, 'ps_gestational_age_in_weeks'] == 39:
            next_visit = 40
            days_until_next_contact = int(next_visit - df.at[person_id, 'ps_gestational_age_in_weeks']) * 7

        else:
            days_until_next_contact = int(self.determine_gestational_age_for_next_contact(person_id) -
                                          df.at[person_id, 'ps_gestational_age_in_weeks']) * 7

        # Define the required items
        item_code_urine_dipstick = pd.unique(
            consumables.loc[consumables['Items'] == 'Test strips, urine analysis', 'Item_Code'])[0]
        item_code_iron_folic_acid = pd.unique(
            consumables.loc[consumables['Items'] == 'Ferrous Salt + Folic Acid, tablet, 200 + 0.25 mg', 'Item_Code'])[0]
        item_code_diet_supps = pd.unique(
            consumables.loc[consumables['Items'] == 'Dietary supplements (country-specific)', 'Item_Code'])[0]

        consumables_anc1 = {
            'Intervention_Package_Code': {},
            'Item_Code': {item_code_iron_folic_acid: days_until_next_contact,
                          item_code_diet_supps: days_until_next_contact,
                          item_code_urine_dipstick: 1}}

        # Check there availability
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=hsi_event,
            cons_req_as_footprint=consumables_anc1,
            to_log=False)

        # Blood pressure measurement...
        # We apply a probability that the HCW will perform a blood pressure check during this visit
        if self.rng.random_sample() < params['prob_bp_check']:

            # If so, we use the dx_test to determine if this check will correctly identify a woman's hypertension
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
              dx_tests_to_run='blood_pressure_measurement', hsi_event=hsi_event):

                # As severe hypertension is part of the diagnostic criteria for severe pre-eclampsia, we assume all
                # women having a BP test who have severe pre-eclampsia will be severely hypertensive
                if df.at[person_id, 'ps_htn_disorders'] == 'severe_pre_eclamp':
                    hypertension_diagnosed = 'severe'
                else:
                    # Otherwise we assume here hypertension is mild. N.b. severity of gestational hypertension is not
                    # yet modelled
                    # We store this as a temporary variable and determine action after urine dipstick is carried out
                    hypertension_diagnosed = 'mild'
            else:
                # 'none' here means either the woman is normatensive or she has hypertension which was missed by the BP
                # monitoring
                hypertension_diagnosed = 'none'
        else:
            # Here this woman has not had a BP test and therefore any hypertension she has is undiagnosed
            hypertension_diagnosed = 'none'

        # Urine dipstick for protein...
        # Next we apply a probability that the HCW will perform a urine dipstick
        if self.rng.random_sample() < params['prob_urine_dipstick'] and \
            outcome_of_request_for_consumables['Item_Code'][item_code_urine_dipstick]:

            # Severity of proteinuria as determined by the result of the dipstick (i.e. protein +, protein ++,
            # protein +++) is part of the diagnositc criteria for pre-eclampsia

            # We use 2 dx_test functions to replicate this test detecting underlying severity of pre-eclampsia
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
              dx_tests_to_run='urine_dipstick_protein_1_plus', hsi_event=hsi_event):

                # This dx_test evaluates the property 'mild_pre_eclampsia', the severity of proteinuria is stored
                # accordingly
                proteinuria_diagnosed = 'mild'
            elif self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
              dx_tests_to_run='urine_dipstick_protein_3_plus', hsi_event=hsi_event):

                # This dx_test evaluates the property 'severe_pre_eclampsia', the severity of proteinuria is stored
                # accordingly
                proteinuria_diagnosed = 'severe'
            else:
                proteinuria_diagnosed = 'none'

            # Store the used consumables to the log
            consumables_used_dipstick = {
                'Intervention_Package_Code': {},
                'Item_Code': {item_code_urine_dipstick: 1}}
            self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=hsi_event,
                cons_req_as_footprint=consumables_used_dipstick,
                to_log=True)
        else:
            proteinuria_diagnosed = 'none'

        # TODO: urine test for sugars and infection markers not currently included, awaiting clinical input

        # We assume that any women who are diagnosed as hypertensive in pregnancy are admitted, regardless of severity,
        # to commence treatmet
        if hypertension_diagnosed != 'none' or proteinuria_diagnosed != 'none':
            df.at[person_id, 'ac_to_be_admitted'] = True

        # Iron & folic acid / food supplementation...

        # Both availability of consumables and likelihood of practitioner deciding to initiate treatment determines if
        # the intervention is delivered...
        if outcome_of_request_for_consumables['Item_Code'][item_code_iron_folic_acid] and \
          self.rng.random_sample() < params['prob_start_iron_folic_acid']:
            df.at[person_id, 'ac_receiving_iron_folic_acid'] = True

            # We store the date at which this prescription will run out and the woman is no longer experiencing the
            # benefits of this treatment
            df.at[person_id, 'ac_date_ifa_runs_out'] = self.sim.date + DateOffset(days=days_until_next_contact)

            consumables_used_ifa = {
                'Intervention_Package_Code': {},
                'Item_Code': {item_code_iron_folic_acid: days_until_next_contact}}
            self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=hsi_event,
                cons_req_as_footprint=consumables_used_ifa,
                to_log=True)

        if outcome_of_request_for_consumables['Item_Code'][item_code_diet_supps] and \
          self.rng.random_sample() < params['prob_start_diet_supps_acid']:
            df.at[person_id, 'ac_receiving_diet_supplements'] = True
            df.at[person_id, 'ac_date_ds_runs_out'] = self.sim.date + DateOffset(days=days_until_next_contact)

            consumables_used_ds = {
                'Intervention_Package_Code': {},
                'Item_Code': {item_code_diet_supps: days_until_next_contact}}
            self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=hsi_event,
                cons_req_as_footprint=consumables_used_ds,
                to_log=True)

    def interventions_delivered_only_at_first_contact(self, hsi_event):
        """ This function houses the additional interventions that should be delivered only at woman's first ANC contact
         which are not included in the above function. This includes the distribution of insecticide treated bed nets
         and TB screening"""

        person_id = hsi_event.target
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        df = self.sim.population.props

        # TODO: unsure if i should include a probability that the ITN will be given seperate from its availblity as
        #  with the other interventions- this needs to be standardised
        #  (quality)

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
            cons_req_as_footprint=consumables_llitn,
            to_log=False)

        # If available, women are provided with a bed net at ANC1. The effect of these nets is determined
        # through the malaria module - not yet coded. n.b any interventions involving non-obstetric diseases have been
        # discussed with Tara
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_obstructed_llitn]:
            df.at[person_id, 'ac_itn_provided'] = True

            self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=hsi_event,
                cons_req_as_footprint=consumables_llitn,
                to_log=True)

        # TB screening...
        # Currently we schedule women to the TB screening HSI in the TB module, however this may over-use resources so
        # possible the TB screening should also just live in this code
        if 'tb' in self.sim.modules.keys():
            tb_screen = HSI_TbScreening(
                module=self.sim.modules['tb'], person_id=person_id)

            self.sim.modules['HealthSystem'].schedule_hsi_event(tb_screen, priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))
        # TODO: not clear from guidelines the frequency of screening so currently just initiated in first vist (one off)

    def tetanus_vaccination(self, hsi_event):
        """This function manages the administration of tetanus vaccination"""
        person_id = hsi_event.target
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        df = self.sim.population.props

        # Tetanus vaccination/booster...
        # TODO: quality probability?
        # TODO: this should be conditioned on a woman's current vaccination status?

        # Define required consumables
        pkg_code1 = pd.unique(
            consumables.loc[consumables["Intervention_Pkg"] == "Tetanus toxoid (pregnant women)",
                            "Intervention_Pkg_Code"])[0]

        consumables_needed = {
            "Intervention_Package_Code": {pkg_code1: 1},
            "Item_Code": {}}

        # Check their availability
        outcome_of_request_for_consumables = self.sim.modules["HealthSystem"].request_consumables(
            hsi_event=hsi_event, cons_req_as_footprint=consumables_needed, to_log=False)

        # And if available, deliver the vaccination. This is stored as an integer as 2 doses are required during
        # pregnancy
        if outcome_of_request_for_consumables:
            df.at[person_id, 'ac_ttd_received'] += 1

            # Save consumables to the log
            self.sim.modules["HealthSystem"].request_consumables(
                hsi_event=hsi_event, cons_req_as_footprint=consumables_needed, to_log=True)

    def calcium_supplementation(self, hsi_event):
        """This function manages the intervention calcium supplementation"""
        df = self.sim.population.props
        params = self.parameters
        person_id = hsi_event.target
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        # TODO: Confirm risk factors that define 'high risk of pre-eclampsia' and condition appropriately

        # Define consumables
        item_code_calcium_supp = pd.unique(
            consumables.loc[consumables['Items'] == 'Calcium, tablet, 600 mg', 'Item_Code'])[0]

        # And required dose
        days_until_next_contact = (self.determine_gestational_age_for_next_contact(person_id) -
                                   df.at[person_id, 'ps_gestational_age_in_weeks']) * 7

        dose = days_until_next_contact * 3  # gives daily dose of 1.8g

        # Have to convert from int.64 to int for consumables to run
        converted_dose = dose.item()
        converted_days = days_until_next_contact.item()

        consumables_anc_2 = {
            'Intervention_Package_Code': {},
            'Item_Code': {item_code_calcium_supp: converted_dose}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=hsi_event,
            cons_req_as_footprint=consumables_anc_2,
            to_log=False)

        if outcome_of_request_for_consumables['Item_Code'][item_code_calcium_supp] and \
            self.rng.random_sample() < params['prob_start_calcium_supp']:
            df.at[person_id, 'ac_receiving_calcium_supplements'] = True
            df.at[person_id, 'ac_date_cal_runs_out'] = self.sim.date + DateOffset(days=converted_days)

            self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=hsi_event,
                cons_req_as_footprint=consumables_anc_2,
                to_log=True)

    def point_of_care_hb_testing(self, hsi_event):
        """This function manages point of care Hb testing carried out in ANC visits"""

        # TODO: consumables for POC test and FBC currently assumed to be the same (incorrect?)
        # TODO: availability of required equipment?

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
            cons_req_as_footprint=consumables_hb_test,
            to_log=False)

        # Determine if the HCW will deliver this intervention
        if self.rng.random_sample() < params['prob_poc_hb_test']:

            # Check consumables
            if outcome_of_request_for_consumables:

                # Log if available
                self.sim.modules['HealthSystem'].request_consumables(
                    hsi_event=hsi_event,
                    cons_req_as_footprint=consumables_hb_test,
                    to_log=True)

                # Run dx_test for anaemia
                if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run='point_of_care_hb_test',
                                                                           hsi_event=hsi_event):
                    df.at[person_id, 'ac_to_be_admitted'] = True

    def albendazole_administration(self, hsi_event):
        """This function manages the administration of albendazole. Albendazole does not have an affect on outcomes in
        the model"""
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        # We run this function to store the associated consumables with albendazole administration. This intervention
        # has no effect in the model due to limited evidence
        item_code_albendazole = pd.unique(
            consumables.loc[consumables['Items'] == 'Albendazole 200mg_1000_CMST', 'Item_Code'])[0]

        consumables_albendazole = {
            'Intervention_Package_Code': {},
            'Item_Code': {item_code_albendazole: 2}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=hsi_event,
            cons_req_as_footprint=consumables_albendazole,
            to_log=False)

        if outcome_of_request_for_consumables['Item_Code'][item_code_albendazole]:
                logger.debug(key='message', data='albendazole given')
                self.sim.modules['HealthSystem'].request_consumables(
                    hsi_event=hsi_event,
                    cons_req_as_footprint=consumables_albendazole,
                    to_log=True)

    def hep_b_testing(self, hsi_event):
        """ This function manages testing for Hepatitis B. As hepatitis is not yet modelled, this intervention merely
        captures consumables"""
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        # This intervention is a place holder prior to the Hepatitis B module being coded
        item_code_hep_test = pd.unique(
            consumables.loc[consumables['Items'] == 'Hepatitis B test kit-Dertemine_100 tests_CMST', 'Item_Code'])[0]
        item_code_blood_tube = pd.unique(
            consumables.loc[consumables['Items'] == 'Blood collecting tube, 5 ml', 'Item_Code'])[0]
        item_code_needle = pd.unique(
            consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]
        item_code_gloves = pd.unique(
            consumables.loc[consumables['Items'] == 'Gloves, exam, latex, disposable, pair', 'Item_Code'])[0]
        # todo: is the cost of this item 100 tests? unclear
        # todo: replace with dx_test and add scheduling when Hep B coded

        consumables_hep_b_test = {
                'Intervention_Package_Code': {},
                'Item_Code': {item_code_hep_test: 1, item_code_blood_tube: 1, item_code_needle:1, item_code_gloves: 1}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=hsi_event,
            cons_req_as_footprint=consumables_hep_b_test,
            to_log=False)

        if outcome_of_request_for_consumables:
            logger.debug(key='message', data='hepatitis B test given')
            self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=hsi_event,
                cons_req_as_footprint=consumables_hep_b_test,
                to_log=True)

    def syphilis_testing(self, hsi_event):
        """This function manages Syphilis testing. Syphilis is not explicitly modelled and therefore this function
        merely records consumable use"""
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        # TODO: There are additional consumables associate with treatment that wont be counted if we dont model the
        #  disease

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
            cons_req_as_footprint=consumables_syphilis,
            to_log=False)

        if outcome_of_request_for_consumables:
            logger.debug(key='message',data='syphilis test given')
            self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=hsi_event,
                cons_req_as_footprint=consumables_syphilis,
                to_log=True)

    def hiv_testing(self, hsi_event):
        """This function schedules HIV testing for women during ANC"""
        person_id = hsi_event.target

        # todo: should the code for the screening process just live in this function or ok to schedule as additional
        #  HSI?
        # TODO: Also check with tara if treatment in pregnancy is the same (or does the HIV code already allow for
        #  this?)
        if 'hiv' in self.sim.modules.keys():
            hiv_testing = HSI_Hiv_PresentsForCareWithSymptoms(
                module=self.sim.modules['hiv'], person_id=person_id)

            self.sim.modules['HealthSystem'].schedule_hsi_event(hiv_testing, priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))
        else:
            logger.warning(key='message', data='HIV module is not registered in this simulation run and therefore HIV '
                                               'testing will not happen in antenatal care')

    def iptp_administration(self, hsi_event):
        """ This functions manages the administration of Intermittent preventative treatment for the prevention of
        malaria"""
        df = self.sim.population.props
        person_id = hsi_event.target
        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

        # TODO: quality indicator with probability of treatment admistration?
        # todo: ensure conditioning on ma_tx when malaria code is available (as below)

        # if (not df.at[person_id, "ma_tx"]
        #    and not df.at[person_id, "ma_tx"]
        #    and df.at[person_id, "is_alive"]):

        # Test to ensure only 5 doses are able to be administered
        assert df.at[person_id, 'ac_doses_of_iptp_received'] < 6

        # Define and check the availability of consumables
        pkg_code1 = pd.unique(
            consumables.loc[consumables["Intervention_Pkg"] == "IPT (pregnant women)", "Intervention_Pkg_Code"])[0]

        the_cons_footprint = {
            "Intervention_Package_Code": {pkg_code1: 1},
            "Item_Code": {}}

        outcome_of_request_for_consumables = self.sim.modules["HealthSystem"].request_consumables(
                hsi_event=hsi_event, cons_req_as_footprint=the_cons_footprint, to_log=False)

        if outcome_of_request_for_consumables:
            logger.debug(key='message', data=f'giving IPTp for person {person_id}')

            # IPTP is a single dose drug given at a number of time points during pregnancy. Therefore the number of
            # doses received during this pregnancy are stored as an integer
            df.at[person_id, 'ac_doses_of_iptp_received'] += 1

            self.sim.modules["HealthSystem"].request_consumables(
                hsi_event=hsi_event,
                cons_req_as_footprint=the_cons_footprint,
                to_log=True,
                )

    def gdm_screening(self, hsi_event):
        """This intervention screens women with risk factors for gestational diabetes and schedules the appropriate
        testing"""
        df = self.sim.population.props
        person_id = hsi_event.target
        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

        # We check if this women has any of the key risk factors, if so they are sent for additional blood tests
        if df.at[person_id, 'li_bmi'] >= 4 or df.at[person_id, 'ps_prev_gest_diab'] or df.at[person_id,
                                                                                             'ps_previous_stillbirth']:

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
                cons_req_as_footprint=consumables_gdm_testing,
                to_log=False)

            # If they are available, the test is conducted
            if outcome_of_request_for_consumables:

                # If the test accurately detects a woman has gestational diabetes the consumables are recorded and she
                # is referred for treatment
                if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run='blood_test_glucose',
                                                                           hsi_event=hsi_event):
                    self.sim.modules['HealthSystem'].request_consumables(
                        hsi_event=hsi_event,
                        cons_req_as_footprint=consumables_gdm_testing,
                        to_log=True)

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

    def antenatal_care_scheduler(self, individual_id, visit_to_be_scheduled, recommended_gestation_next_anc):
        """This function is responsible for scheduling a womans next antenatal care visit. The function is provided with
        which contact is to be scheduled along with the recommended gestational age a woman should be to attend the next
        visit in the schedule"""
        df = self.sim.population.props
        params = self.parameters

        # Make sure women will be scheduled the correct ANC visit by timing
        assert df.at[individual_id, 'ps_gestational_age_in_weeks'] < recommended_gestation_next_anc

        # The code which determines if and when a woman will undergo another ANC visit. Logic is abstracted into this
        # function to prevent copies of block code
        def set_anc_date(individual_id, visit_number):

            # We store the possible ANC contact that we may schedule as variables
            if visit_number == 2:
                visit = HSI_CareOfWomenDuringPregnancy_SecondAntenatalCareContact(
                    self, person_id=individual_id)

            elif visit_number == 3:
                visit = HSI_CareOfWomenDuringPregnancy_ThirdAntenatalCareContact(
                    self, person_id=individual_id)

            elif visit_number == 4:
                visit = HSI_CareOfWomenDuringPregnancy_FourthAntenatalCareContact(
                 self, person_id=individual_id)

            elif visit_number == 5:
                visit = HSI_CareOfWomenDuringPregnancy_FifthAntenatalCareContact(
                 self, person_id=individual_id)

            elif visit_number == 6:
                visit = HSI_CareOfWomenDuringPregnancy_SixthAntenatalCareContact(
                 self, person_id=individual_id)

            elif visit_number == 7:
                visit = HSI_CareOfWomenDuringPregnancy_SeventhAntenatalCareContact(
                 self, person_id=individual_id)

            elif visit_number == 8:
                visit = HSI_CareOfWomenDuringPregnancy_EighthAntenatalCareContact(
                 self, person_id=individual_id)

            # There are a number of variables that determine if a woman will attend another ANC visit:
            # 1.) If she is predicted to attend > 4 visits
            # 2.) Her gestational age at presentation to this ANC visit
            # 3.) The recommended gestational age for each ANC contact and how that matches to the womans current
            # gestational age

            # If this woman has attended less than 4 visits, and is predicted to attend > 4. Her subsequent ANC
            # appointment is automatically scheduled

            # TODO: copy original code back from draft PR branch

            if visit_number < 3:
                if df.at[individual_id, 'ps_will_attend_3_early_visits']:
                    # We schedule a womans next ANC appointment by subtracting her current gestational age from the
                    # target gestational age from the next visit on the ANC schedule (assuming health care workers would
                    # ask women to return for the next appointment on the schedule, regardless of their gestational age
                    # at presentation)
                    weeks_due_next_visit = int(recommended_gestation_next_anc - df.at[individual_id,
                                                                                      'ps_gestational_age_in_weeks'])
                    visit_date = self.sim.date + DateOffset(weeks=weeks_due_next_visit)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(visit, priority=0,
                                                                        topen=visit_date,
                                                                        tclose=visit_date + DateOffset(days=7))

                else:
                    will_anc_continue = params['ac_linear_equations']['anc_continues'].predict(df.loc[[
                        individual_id]])[individual_id]
                    if self.rng.random_sample() < will_anc_continue:
                        weeks_due_next_visit = int(recommended_gestation_next_anc - df.at[individual_id,
                                                                                          'ps_gestational_age_in_weeks'])
                        visit_date = self.sim.date + DateOffset(weeks=weeks_due_next_visit)
                        self.sim.modules['HealthSystem'].schedule_hsi_event(visit, priority=0,
                                                                            topen=visit_date,
                                                                            tclose=visit_date + DateOffset(days=7))
                    else:
                        logger.debug(key='message', data=f'mother {individual_id} will not seek any additional antenatal'
                                                         f' care for this pregnancy')
            elif visit_number >= 3:
                if self.rng.random_sample() < params['ac_linear_equations']['anc_continues'].predict(df.loc[[
                                                                                    individual_id]])[individual_id]:
                    weeks_due_next_visit = int(recommended_gestation_next_anc - df.at[individual_id,
                                                                                      'ps_gestational_age_in_'
                                                                                      'weeks'])
                    visit_date = self.sim.date + DateOffset(weeks=weeks_due_next_visit)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(visit, priority=0,
                                                                        topen=visit_date,
                                                                        tclose=visit_date + DateOffset(days=7))
                else:
                    logger.debug(key='message', data=f'mother {individual_id} will not seek any additional antenatal '
                                                     f'care for this pregnancy')

        if 2 <= visit_to_be_scheduled <= 8:
            set_anc_date(individual_id, visit_to_be_scheduled)

    def schedule_admission(self, individual_id):
        df = self.sim.population.props

        inpatient = HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare(
            self.sim.modules['CareOfWomenDuringPregnancy'], person_id=individual_id)

        self.sim.modules['HealthSystem'].schedule_hsi_event(inpatient, priority=0,
                                                            topen=self.sim.date,
                                                            tclose=self.sim.date + DateOffset(days=1))

        df.at[individual_id, 'ac_to_be_admitted'] = False

    # ================================= INPATIENT INTERVENTION FUNCTIONS ==============================================
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

    def initiate_treatment_for_gdm(self, individual_id):
        pass

    def start_iron_and_folic_acid(self, individual_id, hsi_event):
        """This function is called to start women on IFA who are not receiving antenatally following diagnosis of
        anaemia"""

        df = self.sim.population.props
        params= self.parameters
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        assert ~df.at[individual_id, 'ac_receiving_iron_folic_acid']

        # Check availibility of consumables
        item_code_iron_folic_acid = pd.unique(
            consumables.loc[
                consumables['Items'] ==
                'Ferrous Salt + Folic Acid, tablet, 200 + 0.25 mg', 'Item_Code'])[0]

        consumables_ifa = {
            'Intervention_Package_Code': {},
            'Item_Code': {item_code_iron_folic_acid: 10}}  # TODO: correct quantity

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=hsi_event,
            cons_req_as_footprint=consumables_ifa,
            to_log=False)

        # Start iron and folic acid treatment
        if outcome_of_request_for_consumables:
            df.at[individual_id, 'ac_receiving_iron_folic_acid'] = True
            # Women started on IFA at this stage are already anaemic, we here apply a probability that starting on a
            #  course of IFA will correct anaemia prior to follow up
            if self.rng.random_sample() < params['effect_of_ifa_for_resolving_anaemia']:
                df.at[individual_id, 'ps_anaemia_in_pregnancy'] = 'none'

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
            cons_req_as_footprint=consumables_ifa,
            to_log=False)

        # Treatment is provided dependent on deficiencies present
        # We apply a probability that treatment will resolve a womans anaemia and the deficiency (understanding that
        # anaemia can be multi-factoral)
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

        # todo: log consumables

    def antenatal_blood_transfusion(self, individual_id, hsi_event):
        """This function houses the antenatal blood transfusion intervention"""

        df = self.sim.population.props
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        params = self.parameters

        assert df.at[individual_id, 'ps_anaemia_in_pregnancy'] == 'severe'

        # Check for consumables
        item_code_bt1 = pd.unique(consumables.loc[consumables['Items'] == 'Blood, one unit', 'Item_Code'])[0]
        item_code_bt2 = pd.unique(consumables.loc[consumables['Items'] == 'Lancet, blood, disposable',
                                                  'Item_Code'])[0]
        item_code_bt3 = pd.unique(consumables.loc[consumables['Items'] == 'Test, hemoglobin', 'Item_Code'])[0]
        item_code_bt4 = pd.unique(consumables.loc[consumables['Items'] == 'IV giving/infusion set, with needle',
                                                  'Item_Code'])[0]
        consumables_needed_bt = {'Intervention_Package_Code': {},
                                 'Item_Code': {item_code_bt1: 2, item_code_bt2: 1, item_code_bt3: 1,
                                               item_code_bt4: 2}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=hsi_event, cons_req_as_footprint=consumables_needed_bt, to_log=False)

        if outcome_of_request_for_consumables:
            self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=hsi_event, cons_req_as_footprint=consumables_needed_bt, to_log=True)

            # If available, we apply a probability that a transfusion of 2 units RBCs will correct this woman's
            # severe anaemia
            if params['treatment_effect_blood_transfusion_anaemia'] < self.rng.random_sample():
                df.at[individual_id, 'ps_anaemia_in_pregnancy'] = 'none'

        # TODO: repeat FBC and transfuse again if still anaemic? unlikely

    def initiate_anti_hypertensive_treatment(self, individual_id, hsi_event):
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
        item_code_methyldopa = pd.unique(
            consumables.loc[consumables['Items'] == 'Methyldopa 250mg_1000_CMST', 'Item_Code'])[0]

        consumables_gest_htn_treatment = {
            'Intervention_Package_Code': {},
            'Item_Code': {item_code_hydralazine: 1, item_code_wfi: 1, item_code_needle: 1,
                          item_code_gloves: 1, item_code_methyldopa: 4}}

        # todo: this is one dose of methyldopa but needs to be enough for whole pregnancy?

        # Then query if these consumables are available during this HSI
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=hsi_event,
            cons_req_as_footprint=consumables_gest_htn_treatment,
            to_log=False)

        # If they are available then the woman is started on treatment
        if outcome_of_request_for_consumables:
            df.at[individual_id, 'ac_gest_htn_on_treatment'] = True

            self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=hsi_event,
                cons_req_as_footprint=consumables_gest_htn_treatment,
                to_log=True)

    def treatment_for_severe_pre_eclampsia_or_eclampsia(self, individual_id, cause, hsi_event):
        df = self.sim.population.props
        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

        pkg_code_severe_preeclampsia = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Management of eclampsia',
                            'Intervention_Pkg_Code'])[0]

        consumables_needed_spe = {'Intervention_Package_Code': {pkg_code_severe_preeclampsia: 1},
                                  'Item_Code': {}}

        outcome_of_request_for_consumables_spe = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=hsi_event, cons_req_as_footprint=consumables_needed_spe, to_log=False)

        if outcome_of_request_for_consumables_spe:
            self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=hsi_event, cons_req_as_footprint=consumables_needed_spe, to_log=True)

            df.at[individual_id, f'ac_{cause}treatment'] = True

    def treatment_of_antepartum_haemorrhage(self, individual_id):
        df = self.sim.population.props
        mother = df.loc[individual_id]

        # If abruption > 28 weeks --> CS
        if mother.ps_antepartum_haemorrhage == 'placental_abruption' and mother.ps_gestational_age_in_weeks >= 28:
            elective_section = HSI_Labour_ElectiveCaesareanSection(
                self.sim.modules['Labour'], person_id=individual_id)

            self.sim.modules['HealthSystem'].schedule_hsi_event(elective_section, priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))
        elif mother.ps_gestational_age_in_weeks < 28:
            pass
            # todo: what would happen here? would they just do a CS anyway? baby survival non-exsistent?

        if mother.ps_antepartum_haemorrhage == 'placenta_praevia' and mother.ps_antepartum_haemorrhage_severity == \
            'severe' and mother.ps_gestational_age_in_weeks >= 28:
            elective_section = HSI_Labour_ElectiveCaesareanSection(
                self.sim.modules['Labour'], person_id=individual_id)

            self.sim.modules['HealthSystem'].schedule_hsi_event(elective_section, priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        if mother.ps_antepartum_haemorrhage == 'placenta_praevia' and \
            mother.ps_antepartum_haemorrhage_severity == \
            'mild_moderate' and mother.ps_gestational_age_in_weeks >= 28:
            days_untill_safe_for_cs = (37 * 7) - (mother.ps_gestational_age_in_weeks * 7)

            #inpatient = HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare(
            #    self.sim.modules['CareOfWomenDuringPregnancy'], person_id=individual_id)

            #self.sim.modules['HealthSystem'].schedule_hsi_event(inpatient, priority=0,
            #                                                    topen=self.sim.date,
            #                                                    tclose=self.sim.date +
            #                                                    DateOffset(days=days_untill_safe_for_cs))

            elective_section = HSI_Labour_ElectiveCaesareanSection(
                self.sim.modules['Labour'], person_id=individual_id)

            self.sim.modules['HealthSystem'].schedule_hsi_event(elective_section, priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date +
                                                                       DateOffset(days=days_untill_safe_for_cs))

            # todo: steriods of GA < 34 weeks

class HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareContact(HSI_Event, IndividualScopeEventMixin):
    """ This is the  HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareContact. It will be scheduled by the
    PregnancySupervisor Module.It will be responsible for the management of monitoring and treatment interventions
    delivered in a woman's first antenatal care visit. It will also go on the schedule the womans next ANC
    appointment."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_PresentsForFirstAntenatalCareVisit'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'AntenatalFirst': 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        logger.debug(key='message', data=f'This is HSI_CareOfWomenDuringPregnancy_PresentsForFirstAntenatalCareVisit, '
                                         f'person {person_id} has presented for the first antenatal care visit of their'
                                         f'pregnancy on date {self.sim.date} at gestation'
                                         f' {df.at[person_id, "ps_gestational_age_in_weeks"]}')

        # We condition this event on the woman being alive, pregnant, not currently in labour and not previously
        # attending an ANC visit
        if df.at[person_id, 'is_alive'] and df.at[person_id, 'is_pregnant'] and \
            ~df.at[person_id, 'la_currently_in_labour'] and df.at[person_id,
                                                                  'ac_total_anc_visits_current_pregnancy'] == 0:

            # We add a visit to a rolling total of ANC visits in this pregnancy
            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            # And ensure only women whose first contact with ANC services are attending this event
            assert df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] == 1
            assert df.at[person_id, 'ps_gestational_age_in_weeks'] is not None
            assert df.at[person_id, 'ps_gestational_age_in_weeks'] is not pd.NaT

            # We store some information for summary statistics
            self.module.anc_tracker['cumm_ga_at_anc1'] += df.at[person_id, 'ps_gestational_age_in_weeks']
            self.module.anc_tracker['total_first_anc_visits'] += 1
            if df.at[person_id, 'ps_gestational_age_in_weeks'] < 14:
                self.module.anc_tracker['total_anc1_first_trimester'] += 1

            # First, we use the follow function to determine at what gestational age this woman should be scheduled to
            # return for her next ANC visit (based on gestational age at presentation to ANC1)
            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)
            self.module.interventions_delivered_at_every_contact(hsi_event=self)

            # Then, the appropriate interventions are delivered according gestational age. It is assumed that women who
            # present late to ANC1 are 'caught up' with the interventions they missed from previous visits
            # (as per malawi guidelines)

            # If this woman is presenting prior to the suggested gestation for ANC2, she receives only the interventions
            # for ANC1
            if df.at[person_id, 'ps_gestational_age_in_weeks'] < 20:
                # These are the interventions delivered at ANC1
                self.module.interventions_delivered_only_at_first_contact(hsi_event=self)
                self.module.hiv_testing(hsi_event=self)
                self.module.hep_b_testing(hsi_event=self)
                self.module.syphilis_testing(hsi_event=self)
                self.module.point_of_care_hb_testing(hsi_event=self)
                self.module.tetanus_vaccination(hsi_event=self)

                # She is then assessed to see if she will attend the next ANC contact in the schedule
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=2,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] < 26:
                self.module.interventions_delivered_only_at_first_contact(hsi_event=self)
                self.module.anc_catch_up_interventions(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=2,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] < 40:
                self.module.interventions_delivered_only_at_first_contact(hsi_event=self)
                self.module.anc_catch_up_interventions(hsi_event=self)

                # Women presenting at >26 are indicated to require screening for gestational diabetes
                # (usually delivered in ANC 2)
                self.module.gdm_screening(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=2,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] >= 40:
                self.module.interventions_delivered_only_at_first_contact(hsi_event=self)
                self.module.anc_catch_up_interventions(hsi_event=self)
                self.module.gdm_screening(hsi_event=self)

                # todo: for now, these women just wont have another visit

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
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_SecondAntenatalCareVisit'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 1
        # TODO: this crashes on facility level 0?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        logger.debug(key='message', data=f'This is HSI_CareOfWomenDuringPregnancy_SecondAntenatalCareVisit, '
                                         f'person {person_id} has presented for the second antenatal care visit of '
                                         f'their pregnancy on date {self.sim.date} at gestation '
                                         f'{df.at[person_id, "ps_gestational_age_in_weeks"]}')

        if df.at[person_id, 'is_alive'] and df.at[person_id, 'is_pregnant'] and \
            ~df.at[person_id, 'la_currently_in_labour'] and \
            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] == 1:

            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1
            assert df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] == 2

            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)
            self.module.interventions_delivered_at_every_contact(hsi_event=self)
            self.module.tetanus_vaccination(hsi_event=self)

            # ========================================== Schedule next visit =======================================
            if df.at[person_id, 'ps_gestational_age_in_weeks'] < 26:
                self.module.albendazole_administration(hsi_event=self)
                self.module.iptp_administration(hsi_event=self)
                self.module.calcium_supplementation(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=3,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] < 30:
                self.module.iptp_administration(hsi_event=self)
                self.module.calcium_supplementation(hsi_event=self)
                self.module.gdm_screening(hsi_event=self)

                self.module.schedule_next_anc(person_id, visit_to_be_scheduled=3,
                                              recommended_gestation_next_anc=gest_age_next_contact)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] < 34:
                self.module.iptp_administration(hsi_event=self)
                self.module.calcium_supplementation(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=3,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] < 36:
                self.module.iptp_administration(hsi_event=self)
                self.module.calcium_supplementation(hsi_event=self)

                self.module.hep_b_testing(hsi_event=self)
                self.module.syphilis_testing(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=3,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] < 38:
                self.module.calcium_supplementation(hsi_event=self)
                self.module.point_of_care_hb_testing(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=3,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] < 40:
                self.module.calcium_supplementation(hsi_event=self)
                self.module.iptp_administration(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=3,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] >= 40:
                pass

            if df.at[person_id, 'ac_to_be_admitted']:
                self.module.schedule_admission(person_id)

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        return actual_appt_footprint

    def did_not_run(self):
        pass

    def not_available(self):
        logger.debug(key='message',data='HSI_CareOfWomenDuringPregnancy_SecondAntenatalCareVisit: cannot not run with '
                                        'this configuration')


class HSI_CareOfWomenDuringPregnancy_ThirdAntenatalCareContact(HSI_Event, IndividualScopeEventMixin):
    """This is the  HSI_CareOfWomenDuringPregnancy_ThirdAntenatalCareContact. It will be scheduled by the
    HSI_CareOfWomenDuringPregnancy_SecondAntenatalCareContact.It will be responsible for the management of monitoring
    and treatment interventions delivered in a woman's third antenatal care visit. It will also go on the schedule the
    womans next ANC appointment"""
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_PresentsForSubsequentAntenatalCareVisit'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        logger.debug(key='message', data=f'This is HSI_CareOfWomenDuringPregnancy_ThirdAntenatalCareContact, '
                                         f'person {person_id} has presented for the third antenatal care visit of '
                                         f'their pregnancy on date {self.sim.date} at gestation '
                                         f'{df.at[person_id, "ps_gestational_age_in_weeks"]}')

        if df.at[person_id, 'is_alive'] and df.at[person_id, 'is_pregnant'] and ~df.at[person_id,
                                                                                       'la_currently_in_labour'] and \
            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] == 2:

            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1
            assert df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] == 3

            if df.at[person_id, 'ps_gestational_age_in_weeks'] < 30:
                self.module.anc_tracker['timely_ANC3'] += 1

            self.module.interventions_delivered_at_every_contact(hsi_event=self)
            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)

            # ========================================== Schedule next visit =======================================

            if df.at[person_id, 'ps_gestational_age_in_weeks'] < 30:
                self.module.iptp_administration(hsi_event=self)
                self.module.calcium_supplementation(hsi_event=self)
                self.module.gdm_screening(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=4,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] < 34:
                self.module.iptp_administration(hsi_event=self)
                self.module.calcium_supplementation(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=4,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] < 36:
                self.module.iptp_administration(hsi_event=self)
                self.module.calcium_supplementation(hsi_event=self)

                self.module.hep_b_testing(hsi_event=self)
                self.module.syphilis_testing(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=4,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] < 38:
                self.module.calcium_supplementation(hsi_event=self)
                self.module.point_of_care_hb_testing(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=4,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] < 40:
                self.module.calcium_supplementation(hsi_event=self)
                self.module.iptp_administration(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=4,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] >= 40:
                pass

            if df.at[person_id, 'ac_to_be_admitted']:
                self.module.schedule_admission(person_id)

            # todo: remove
            if df.at[person_id, 'ps_gestational_age_in_weeks'] < 30 and df.at[person_id,
                                                                              'ac_receiving_diet_supplements']:
                self.module.anc_tracker['diet_supp_6_months'] += 1

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        return actual_appt_footprint

    def did_not_run(self):
        pass

    def not_available(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_ThirdAntenatalCareContact: cannot not run with '
                                         'this configuration')


class HSI_CareOfWomenDuringPregnancy_FourthAntenatalCareContact(HSI_Event, IndividualScopeEventMixin):
    """This is the  HSI_CareOfWomenDuringPregnancy_FourthAntenatalCareContact. It will be scheduled by the
    HSI_CareOfWomenDuringPregnancy_ThirdAntenatalCareContact.It will be responsible for the management of monitoring
    and treatment interventions delivered in a woman's fourth antenatal care visit. It will also go on the schedule the
    womans next ANC appointment"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_FourthAntenatalCareVisit'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        logger.debug(key='message', data=f'This is HSI_CareOfWomenDuringPregnancy_FourthAntenatalCareContact, '
                                         f'person {person_id} has presented for the fourth antenatal care visit of '
                                         f'their pregnancy on date {self.sim.date} at gestation '
                                         f'{df.at[person_id, "ps_gestational_age_in_weeks"]}')

        if df.at[person_id, 'is_alive'] and df.at[person_id, 'is_pregnant'] and \
            ~df.at[person_id, 'la_currently_in_labour'] and \
            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] == 3:

            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1
            assert df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] == 4

            self.module.interventions_delivered_at_every_contact(hsi_event=self)
            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)

            if df.at[person_id, 'ps_gestational_age_in_weeks'] < 34:
                self.module.iptp_administration(hsi_event=self)
                self.module.calcium_supplementation(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=5,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] < 36:
                self.module.iptp_administration(hsi_event=self)
                self.module.calcium_supplementation(hsi_event=self)

                self.module.hep_b_testing(hsi_event=self)
                self.module.syphilis_testing(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=5,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] < 38:
                self.module.calcium_supplementation(hsi_event=self)
                self.module.point_of_care_hb_testing(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=5,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] < 40:
                self.module.calcium_supplementation(hsi_event=self)
                self.module.iptp_administration(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=5,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] >= 40:
                pass

            if df.at[person_id, 'ac_to_be_admitted']:
                self.module.schedule_admission(person_id)

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        return actual_appt_footprint

    def did_not_run(self):
        pass

    def not_available(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_FourthAntenatalCareContact: cannot not run with'
                                         'this configuration')


class HSI_CareOfWomenDuringPregnancy_FifthAntenatalCareContact(HSI_Event, IndividualScopeEventMixin):
    """This is the  HSI_CareOfWomenDuringPregnancy_FifthAntenatalCareContact. It will be scheduled by the
    HSI_CareOfWomenDuringPregnancy_FourthAntenatalCareContact.It will be responsible for the management of monitoring
    and treatment interventions delivered in a woman's fifth antenatal care visit. It will also go on the schedule the
    womans next ANC appointment"""
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_EmergencyTreatment'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        logger.debug(key='message', data=f'This is HSI_CareOfWomenDuringPregnancy_FifthAntenatalCareContact, '
                                         f'person {person_id} has presented for the fifth antenatal care visit of '
                                         f'their pregnancy on date {self.sim.date} at gestation '
                                         f'{df.at[person_id, "ps_gestational_age_in_weeks"]}')

        if df.at[person_id, 'is_alive'] and df.at[person_id, 'is_pregnant'] and \
            ~df.at[person_id, 'la_currently_in_labour'] and \
            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] == 4:

            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1
            assert df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] == 5

            self.module.interventions_delivered_at_every_contact(hsi_event=self)
            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)

            if df.at[person_id, 'ps_gestational_age_in_weeks'] < 36:
                self.module.iptp_administration(hsi_event=self)
                self.module.calcium_supplementation(hsi_event=self)

                self.module.hep_b_testing(hsi_event=self)
                self.module.syphilis_testing(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=6,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] < 38:
                self.module.calcium_supplementation(hsi_event=self)
                self.module.point_of_care_hb_testing(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=6,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] < 40:
                self.module.calcium_supplementation(hsi_event=self)
                self.module.iptp_administration(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=6,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] >= 40:
                pass

            if df.at[person_id, 'ac_to_be_admitted']:
                self.module.schedule_admission(person_id)

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        return actual_appt_footprint

    def did_not_run(self):
        pass

    def not_available(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_FifthAntenatalCareContact: cannot not run '
                                         'with this configuration')


class HSI_CareOfWomenDuringPregnancy_SixthAntenatalCareContact(HSI_Event, IndividualScopeEventMixin):
    """This is the  HSI_CareOfWomenDuringPregnancy_SixthAntenatalCareContact. It will be scheduled by the
    HSI_CareOfWomenDuringPregnancy_FifthAntenatalCareContact.It will be responsible for the management of monitoring
    and treatment interventions delivered in a woman's sixth antenatal care visit. It will also go on the schedule the
    womans next ANC appointment"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_SixthAntenatalCareVisit'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        logger.debug(key='message', data=f'This is HSI_CareOfWomenDuringPregnancy_SixthAntenatalCareContact, '
                                         f'person {person_id} has presented for the sixth antenatal care visit of '
                                         f'their pregnancy on date {self.sim.date} at gestation '
                                         f'{df.at[person_id, "ps_gestational_age_in_weeks"]}')

        if df.at[person_id, 'is_alive'] and df.at[person_id, 'is_pregnant'] and \
            ~df.at[person_id, 'la_currently_in_labour'] and \
            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] == 5:

            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1
            assert df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] == 6

            self.module.interventions_delivered_at_every_contact(hsi_event=self)
            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)

            if df.at[person_id, 'ps_gestational_age_in_weeks'] < 38:
                self.module.calcium_supplementation(hsi_event=self)
                self.module.point_of_care_hb_testing(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=7,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] < 40:
                self.module.calcium_supplementation(hsi_event=self)
                self.module.iptp_administration(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=7,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] >= 40:
                pass

            if df.at[person_id, 'ac_to_be_admitted']:
                self.module.schedule_admission(person_id)

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        return actual_appt_footprint

    def did_not_run(self):
        pass

    def not_available(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_SixthAntenatalCareContact: cannot not run with'
                                         'this configuration')


class HSI_CareOfWomenDuringPregnancy_SeventhAntenatalCareContact(HSI_Event, IndividualScopeEventMixin):
    """This is the  HSI_CareOfWomenDuringPregnancy_SeventhAntenatalCareContact. It will be scheduled by the
    HSI_CareOfWomenDuringPregnancy_SixthAntenatalCareContact.It will be responsible for the management of monitoring
    and treatment interventions delivered in a woman's seventh antenatal care visit. It will also go on the schedule the
    womans next ANC appointment"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_SeventhAntenatalCareContact'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
        self.ACCEPTED_FACILITY_LEVEL = 1  # 2/3?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        logger.debug(key='message', data=f'This is HSI_CareOfWomenDuringPregnancy_SeventhAntenatalCareContact, '
                                         f'person {person_id} has presented for the seventh antenatal care visit of '
                                         f'their pregnancy on date {self.sim.date} at gestation '
                                         f'{df.at[person_id, "ps_gestational_age_in_weeks"]}')

        if df.at[person_id, 'is_alive'] and df.at[person_id, 'is_pregnant'] and \
            ~df.at[person_id, 'la_currently_in_labour'] and \
            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] == 6:

            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1
            assert df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] == 7

            self.module.interventions_delivered_at_every_contact(hsi_event=self)
            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)

            if df.at[person_id, 'ps_gestational_age_in_weeks'] < 40:
                self.module.calcium_supplementation(hsi_event=self)
                self.module.iptp_administration(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=8,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] >= 40:
                pass

            if df.at[person_id, 'ac_to_be_admitted']:
                self.module.schedule_admission(person_id)

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        return actual_appt_footprint

    def did_not_run(self):
        pass

    def not_available(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_SeventhAntenatalCareContact: cannot not run'
                                         ' with this configuration')


class HSI_CareOfWomenDuringPregnancy_EighthAntenatalCareContact(HSI_Event, IndividualScopeEventMixin):
    """This is the  HSI_CareOfWomenDuringPregnancy_EighthAntenatalCareContact. It will be scheduled by the
        HSI_CareOfWomenDuringPregnancy_SeventhAntenatalCareContact.It will be responsible for the management of monitoring
        and treatment interventions delivered in a woman's eighth antenatal care visit """
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_EighthAntenatalCareContact'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        logger.debug(key='message', data=f'This is HSI_CareOfWomenDuringPregnancy_EighthAntenatalCareContact, '
                                         f'person {person_id} has presented for the eighth antenatal care visit of '
                                         f'their pregnancy on date {self.sim.date} at gestation '
                                         f'{df.at[person_id, "ps_gestational_age_in_weeks"]}')

        if df.at[person_id, 'is_alive'] and df.at[person_id, 'is_pregnant'] and \
            ~df.at[person_id, 'la_currently_in_labour'] and \
            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] == 7:

            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1
            assert df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] == 8
            self.module.anc_tracker['anc8+'] += 1

            self.module.interventions_delivered_at_every_contact(hsi_event=self)
            self.module.calcium_supplementation(hsi_event=self)

            if df.at[person_id, 'ac_to_be_admitted']:
                self.module.schedule_admission(person_id)

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        return actual_appt_footprint

    def did_not_run(self):
        pass

    def not_available(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_EighthAntenatalCareContact: cannot not run'
                                         ' with this configuration')


class HSI_CareOfWomenDuringPregnancy_PostAbortionCaseManagement(HSI_Event, IndividualScopeEventMixin):
    """"""
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_PostAbortionCaseManagement'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        person = df.loc[person_id]

        # We check only women with complications post abortion are sent to this event
        assert person.ps_spontaneous_abortion_complication != 'none' \
               or person.ps_induced_abortion_complication != 'none'

        if person.is_alive:
            # Women are able to undergo a number of interventions to manage abortion complications including evacuation
            # of the uterus, antibiotics, pain management and blood transfusion

            logger.debug(key='message', data=f'Person {person_id} is now undergoing Post Abortion Care at a facility')

            # A woman's probability of requiring these interventions is varied by severity of complications and taken
            # from a study in Malawi
            if person.ps_spontaneous_abortion_complication == 'mild' \
                                                              or person.ps_induced_abortion_complication != 'mild':
                logger.debug(key='message', data=f'Person {person_id} is experiencing mild complications of abortion '
                                                 f'and will be treated accordingly')

                # The intervention package is stored in this function
                self.module.post_abortion_care_interventions(self, person_id, 'mild')

            if person.ps_spontaneous_abortion_complication == 'moderate' \
                                                              or person.ps_induced_abortion_complication != 'moderate':

                logger.debug(key='message', data=f'Person {person_id} is experiencing moderate complications of '
                                                 f'abortion and will be treated accordingly')

                self.module.post_abortion_care_interventions(self, person_id, 'moderate')

            if person.ps_spontaneous_abortion_complication == 'severe' \
                                                              or person.ps_induced_abortion_complication != 'severe':

                logger.debug(key='message', data=f'Person {person_id} is experiencing severe complications of abortion'
                                                 f' and will be treated accordingly')

                self.module.post_abortion_care_interventions(self, person_id, 'severe')

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

        assert mother.ps_ectopic_pregnancy

        if mother.is_alive:
            logger.debug(key='message', data='This is HSI_CareOfWomenDuringPregnancy_TreatmentForEctopicPregnancy, '
                                             f'person {person_id} has been diagnosed with ectopic pregnancy after '
                                             f'presenting and will now undergo surgery')

            # We define the required consumables
            # TODO: finalise consumables
            ectopic_pkg = pd.unique(consumables.loc[consumables['Intervention_Pkg'] == 'Ectopic case management',
                                                    'Intervention_Pkg_Code'])[0]

            consumables_needed_surgery = {'Intervention_Package_Code': {ectopic_pkg: 1}, 'Item_Code': {}}

            # Check their availability
            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self, cons_req_as_footprint=consumables_needed_surgery, to_log=False)

            if outcome_of_request_for_consumables:
                # If available, the treatment can go ahead
                logger.debug(key='message',
                             data='Consumables required for ectopic surgery are available and therefore have been used')

                self.sim.modules['HealthSystem'].request_consumables(
                    hsi_event=self, cons_req_as_footprint=consumables_needed_surgery, to_log=True)

                # Treatment variable set to true, reducing risk of death at death event in PregnancySupervisor
                df.at[person_id, 'ac_ectopic_pregnancy_treated'] = True

            else:
                logger.debug(key='message',
                             data='Consumables required for surgery are unavailable and therefore have not '
                                  'been used')

    def did_not_run(self):
        pass

    def not_available(self):
        logger.debug(key='message', data='HSI_CareOfWomenDuringPregnancy_TreatmentForEctopicPregnancy: cannot not run '
                                         'with this configuration')


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

        # TODO: store diagnosis/cause and send to inpatient event- cause variable used to give treatment etc

        admission = HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare(
            self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id)

        self.sim.modules['HealthSystem'].schedule_hsi_event(admission, priority=0,
                                                            topen=self.sim.date,
                                                            tclose=self.sim.date + DateOffset(days=1))


class HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare(HSI_Event, IndividualScopeEventMixin):
    """"""
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_AntenatalWardInpatientCare'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1   # TODO: modfiy according to reason for admission
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mother = df.loc[person_id]

        if mother.is_alive:
            logger.debug(key='message', data=f'Mother {person_id} has been admited for treatment of a complication of '
                                             f'her pregnancy ')

        #  --------------------------------- Treatment of Maternal Anaemia -------------------------------------------
            if mother.ps_anaemia_in_pregnancy != 'none':  # todo: replace
                # Women who are referred from ANC or an outpatient appointment following point of care Hb testing are
                # admitted for a full blood count
                fbc_result = self.module.full_blood_count_testing(self)
                assert fbc_result == 'no_test' or 'none' or 'non_severe' or 'severe'

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
                    # TODO: run test for ferratin
                    self.module.treatment_of_anaemia_causing_deficiencies(person_id, self)

                    # She is scheduled to return for a repeat point of care test in four weeks time to assess if she
                    # remains anaemic
                    outpatient_checkup = HSI_CareOfWomenDuringPregnancy_AntenatalOutpatientFollowUp(
                        self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(outpatient_checkup, priority=0,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(days=28))

                elif fbc_result == 'severe':
                    # In the case of severe anaemia (Hb <7) a woman receives a blood transfusion in addition to other
                    # treatments
                    self.module.antenatal_blood_transfusion(person_id, self)
                    self.module.treatment_of_anaemia_causing_deficiencies(person_id, self)
                    if mother.ac_receiving_iron_folic_acid == 'False':
                        self.module.start_iron_and_folic_acid(person_id, self)

                    outpatient_checkup = HSI_CareOfWomenDuringPregnancy_AntenatalOutpatientFollowUp(
                        self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(outpatient_checkup, priority=0,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(days=28))

                # TODO: check malaria and HIV status and refer for treatment?
                # TODO: deworming/schisto treatment
                # TODO: capture number of inpatient days by severity
                # TODO: comment out non EHP interventions?

        #  --------------------------------- Treatment of Antepartum Haemorrhage -------------------------------------
            # If abruption > 28 weeks --> CS
            if mother.ps_antepartum_haemorrhage != 'none':
                self.module.treatment_of_antepartum_haemorrhage(person_id)

        # ---------------------------------- Treatment of Severe Pre-eclampsia ---------------------------------------
            if mother.ps_htn_disorders == 'severe_pre_eclamp' or mother.ps_htn_disorders == 'eclampsia':
                cause = mother.ps_htn_disorders
                self.module.treatment_for_severe_pre_eclampsia_or_eclampsia(person_id, cause, self)
                self.module.initiate_anti_hypertensive_treatment(person_id, self)
                # TODO: deliver


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

        if mother.is_alive:
            if mother.ps_anaemia_in_pregnancy != 'none':
                fbc_result = self.module.full_blood_count_testing(self)
                if fbc_result == 'no_test':
                    logger.debug(key='message', data=f'No FBC given due to resource constraints')
                elif fbc_result == 'none':
                    logger.debug(key='message', data=f'Mother {person_id} has not had anaemia detected via an FBC')
                elif fbc_result == 'non_severe' or fbc_result == 'severe':

                    admission = HSI_CareOfWomenDuringPregnancy_AntenatalWardInpatientCare(
                        self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(admission, priority=0,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(days=1))



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

