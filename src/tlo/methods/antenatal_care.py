from pathlib import Path

import pandas as pd
import numpy as np

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.dxmanager import DxTest
from tlo.methods.hiv import HSI_Hiv_PresentsForCareWithSymptoms
from tlo.methods.tb import HSI_TbScreening

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
        self.ANCTracker = dict()

    PARAMETERS = {
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
        'prob_blood_test': Parameter(
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
        'sensitivity_blood_test_hb': Parameter(
            Types.REAL, 'sensitivity of a blood test to detect low haemoglobin'),
        'specificity_blood_test_hb': Parameter(
            Types.REAL, 'specificity of a blood test to detect low haemoglobin'),
        'sensitivity_blood_test_glucose': Parameter(
            Types.REAL, 'sensitivity of a blood test to detect raised blood glucose'),
        'specificity_blood_test_glucose': Parameter(
            Types.REAL, 'specificity of a blood test to detect raised blood glucose'),
        'treatment_effect_iron_folate_anaemia': Parameter(
            Types.REAL, 'treatment effectiveness of iron/folate for anaemia in pregnancy'),
        'treatment_effect_blood_transfusion_anaemia': Parameter(
            Types.REAL, 'treatment effectiveness of blood transfusion for anaemia in pregnancy'),
    }

    PROPERTIES = {
        'ac_total_anc_visits_current_pregnancy': Property(
            Types.INT,
            'rolling total of antenatal visits this woman has attended during her pregnancy'),
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
    }

    def read_parameters(self, data_folder):
        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_AntenatalCare.xlsx',
                            sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)

        params = self.parameters

        params['ac_linear_equations'] = {
                'anc_continues': LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    params['prob_anc_continues'])}

    def initialise_population(self, population):

        df = population.props
        df.loc[df.is_alive, 'ac_total_anc_visits_current_pregnancy'] = 0
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

    def initialise_simulation(self, sim):
        sim.schedule_event(AntenatalCareLoggingEvent(self),
                           sim.date + DateOffset(years=1))

        # Populate the tracker
        self.ANCTracker = {'total_first_anc_visits': 0, 'cumm_ga_at_anc1': 0, 'total_anc1_first_trimester': 0,
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

            blood_test_haemoglobin=DxTest(
                property='ps_anaemia_in_pregnancy',
                sensitivity=params['sensitivity_blood_test_hb'],
                specificity=params['specificity_blood_test_hb']),

            blood_test_glucose=DxTest(
                property='ps_gest_diab',
                sensitivity=params['sensitivity_blood_test_glucose'],
                specificity=params['specificity_blood_test_glucose'])
            )

    def on_birth(self, mother_id, child_id):
        df = self.sim.population.props

        df.at[child_id, 'ac_total_anc_visits_current_pregnancy'] = 0
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

        # Run a check at birth to make sure no women exceed 8 visits, which shouldn't occur through this logic
        assert df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] < 9

        # We store the total number of ANC vists a woman achieves prior to her birth in this logging dataframe
        logger.info('%s|total_anc_per_woman|%s', self.sim.date,
                    {'person_id': mother_id,
                     'age': df.at[mother_id, 'age_years'],
                     'total_anc': df.at[mother_id, 'ac_total_anc_visits_current_pregnancy']})

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
        logger.debug('This is CareOfWomenDuringPregnancy, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

    def determine_gestational_age_for_next_contact(self, person_id):
        """This function determines when a woman should be scheduled to return to ANC based on her current gestation at
        her most recent visit"""
        df = self.sim.population.props

        if df.at[person_id, 'ps_gestational_age_in_weeks'] < 20:
            recommended_gestation_next_anc = 20
            return recommended_gestation_next_anc

        elif 20 >= df.at[person_id, 'ps_gestational_age_in_weeks'] < 26:
            recommended_gestation_next_anc = 26
            return recommended_gestation_next_anc

        elif 26 >= df.at[person_id, 'ps_gestational_age_in_weeks'] < 30:
            recommended_gestation_next_anc = 30
            return recommended_gestation_next_anc

        elif 30 >= df.at[person_id, 'ps_gestational_age_in_weeks'] < 34:
            recommended_gestation_next_anc = 34
            return recommended_gestation_next_anc

        elif 34 >= df.at[person_id, 'ps_gestational_age_in_weeks'] < 36:
            recommended_gestation_next_anc = 36
            return recommended_gestation_next_anc

        elif 36 >= df.at[person_id, 'ps_gestational_age_in_weeks'] < 38:
            recommended_gestation_next_anc = 38
            return recommended_gestation_next_anc

        elif 38 >= df.at[person_id, 'ps_gestational_age_in_weeks'] < 40:
            recommended_gestation_next_anc = 40
            return recommended_gestation_next_anc

        # TODO: this is a quick fix for women who present very late to ANC so that they get some treatment coverage
        #  before they likely give birth
        elif df.at[person_id, 'ps_gestational_age_in_weeks'] >= 40:
            recommended_gestation_next_anc = 45
            return recommended_gestation_next_anc

    # ================================= INTERVENTION FUNCTIONS =======================================================
    # Following functions contain code for the interventions which are called by each ANC contact

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

        # If hypertension is diagnosed a woman is referred for  additional treatment. The type of treatment is
        # determined by the presence and severity of proteinuria
        if hypertension_diagnosed == 'mild' and proteinuria_diagnosed == 'none':
            logger.debug('Mother %d has been diagnosed with gestational hypertension and has been referred on to '
                         'additional care from ANC on date %s', person_id, self.sim.date)

            additional_care = HSI_CareOfWomenDuringPregnancy_InitialManagementOfGestationalHypertension(
                self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id)

            self.sim.modules['HealthSystem'].schedule_hsi_event(additional_care, priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=7))

        elif hypertension_diagnosed == 'mild' and proteinuria_diagnosed == 'mild':
            logger.debug('Mother %d has been diagnosed with mild pre-eclampsia and has been referred on to '
                         'additional care from ANC on date %s', person_id, self.sim.date)

            additional_care = HSI_CareOfWomenDuringPregnancy_InitialManagementOfMildPreEclampsia(
                self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id)

            self.sim.modules['HealthSystem'].schedule_hsi_event(additional_care, priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=2))

        elif hypertension_diagnosed == 'severe' and proteinuria_diagnosed == 'severe':
            logger.debug('Mother %d has been diagnosed with severe pre-eclampsia and has been referred on to '
                         'additional care from ANC on date %s', person_id, self.sim.date)

            additional_care = HSI_CareOfWomenDuringPregnancy_InitialManagementOfSeverePreEclampsia(
                self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id)

            self.sim.modules['HealthSystem'].schedule_hsi_event(additional_care, priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=2))

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

    def hb_testing(self, hsi_event):
        """This function manages the intervention haemoglobin testing"""
        person_id = hsi_event.target
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
        if self.rng.random_sample() < params['prob_blood_test']:

            # Check consumables
            if outcome_of_request_for_consumables:

                # Log if available
                self.sim.modules['HealthSystem'].request_consumables(
                    hsi_event=hsi_event,
                    cons_req_as_footprint=consumables_hb_test,
                    to_log=True)

                # Run dx_test for anaemia
                if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run='blood_test_haemoglobin',
                                                                           hsi_event=hsi_event):

                    # TODO: schedule care in the instance of severe anaemia (not yet modelled in preg sup)

                    # Schedule additional care in the instance of diagnosed anaemia
                    additional_care = HSI_CareOfWomenDuringPregnancy_InitialManagementOfMildAnaemiaInPregnancy(
                        self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id)

                    self.sim.modules['HealthSystem'].schedule_hsi_event(additional_care, priority=0,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(days=7))

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
                logger.debug('albendazole given')
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
            logger.debug('hepatitis B test given')
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
            logger.debug('syphilis')
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

        hiv_testing = HSI_Hiv_PresentsForCareWithSymptoms(
            module=self.sim.modules['hiv'], person_id=person_id)

        self.sim.modules['HealthSystem'].schedule_hsi_event(hiv_testing, priority=0,
                                                            topen=self.sim.date,
                                                            tclose=self.sim.date + DateOffset(days=1))

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
            logger.debug("giving IPTp for person %d", person_id)

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

        # We check if this women has any of the key risk factors, if so they are sent for additional blood tests
        if df.at[person_id, 'li_bmi'] >= 4 or df.at[person_id, 'ps_prev_gest_diab'] or df.at[person_id,
                                                                                             'ps_previous_stillbirth']:
            gdm_testing = HSI_CareOfWomenDuringPregnancy_TestingForGestationalDiabetes(
                self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id)

            self.sim.modules['HealthSystem'].schedule_hsi_event(gdm_testing, priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=3))

    def anc_catch_up_interventions(self, hsi_event):
        """This function actions all the interventions a woman presenting to ANC1 at >20 will need administering."""
        self.hiv_testing(hsi_event=hsi_event)
        self.hep_b_testing(hsi_event=hsi_event)
        self.syphilis_testing(hsi_event=hsi_event)
        self.hb_testing(hsi_event=hsi_event)
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
                        logger.debug('mother %d will not seek any additional antenatal care for this pregnancy',
                                     individual_id)
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
                    logger.debug('mother %d will not seek any additional antenatal care for this pregnancy',
                                 individual_id)

        if visit_to_be_scheduled == 2:
            set_anc_date(individual_id, 2)

        if visit_to_be_scheduled == 3:
            set_anc_date(individual_id, 3)

        if visit_to_be_scheduled == 4:
            set_anc_date(individual_id, 4)

        if visit_to_be_scheduled == 5:
            set_anc_date(individual_id, 5)

        if visit_to_be_scheduled == 6:
            set_anc_date(individual_id, 6)

        if visit_to_be_scheduled == 7:
            set_anc_date(individual_id, 7)

        if visit_to_be_scheduled == 8:
            set_anc_date(individual_id, 8)


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

        logger.debug('This is HSI_CareOfWomenDuringPregnancy_PresentsForFirstAntenatalCareVisit, person %d has '
                     'presented for the first antenatal care visit of their pregnancy on date %s at '
                     'gestation %d', person_id, self.sim.date, df.at[person_id, 'ps_gestational_age_in_weeks'])

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
            self.module.ANCTracker['cumm_ga_at_anc1'] += df.at[person_id, 'ps_gestational_age_in_weeks']
            self.module.ANCTracker['total_first_anc_visits'] += 1
            if df.at[person_id, 'ps_gestational_age_in_weeks'] < 14:
                self.module.ANCTracker['total_anc1_first_trimester'] += 1

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
                self.module.hb_testing(hsi_event=self)
                self.module.tetanus_vaccination(hsi_event=self)

                # She is then assessed to see if she will attend the next ANC contact in the schedule
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=2,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif 20 >= df.at[person_id, 'ps_gestational_age_in_weeks'] < 26:
                self.module.interventions_delivered_only_at_first_contact(hsi_event=self)
                self.module.anc_catch_up_interventions(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=2,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif 26 >= df.at[person_id, 'ps_gestational_age_in_weeks'] < 40:
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

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareVisit: did not run')

    def not_available(self):
        logger.debug('HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareVisit: cannot not run with this configuration')


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

        logger.debug('This is HSI_CareOfWomenDuringPregnancy_SecondAntenatalCareVisit, person %d has '
                     'presented for the second antenatal care visit of their pregnancy on date %s at '
                     'gestation %d', person_id, self.sim.date, df.at[person_id, 'ps_gestational_age_in_weeks'])

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

            elif 26 >= df.at[person_id, 'ps_gestational_age_in_weeks'] < 30:
                self.module.iptp_administration(hsi_event=self)
                self.module.calcium_supplementation(hsi_event=self)
                self.module.gdm_screening(hsi_event=self)

                self.module.schedule_next_anc(person_id, visit_to_be_scheduled=3,
                                              recommended_gestation_next_anc=gest_age_next_contact)

            elif 30 >= df.at[person_id, 'ps_gestational_age_in_weeks'] < 34:
                self.module.iptp_administration(hsi_event=self)
                self.module.calcium_supplementation(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=3,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif 34 >= df.at[person_id, 'ps_gestational_age_in_weeks'] < 36:
                self.module.iptp_administration(hsi_event=self)
                self.module.calcium_supplementation(hsi_event=self)

                self.module.hep_b_testing(hsi_event=self)
                self.module.syphilis_testing(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=3,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif 36 >= df.at[person_id, 'ps_gestational_age_in_weeks'] < 38:
                self.module.calcium_supplementation(hsi_event=self)
                self.module.hb_testing(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=3,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif 38 >= df.at[person_id, 'ps_gestational_age_in_weeks'] < 40:
                self.module.calcium_supplementation(hsi_event=self)
                self.module.iptp_administration(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=3,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] >= 40:
                pass

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        return actual_appt_footprint

    def did_not_run(self):
        pass

    def not_available(self):
        logger.debug('HSI_CareOfWomenDuringPregnancy_SecondAntenatalCareVisit: cannot not run with this configuration')


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

        logger.debug('This is HSI_CareOfWomenDuringPregnancy_ThirdAntenatalCareContact, person %d has '
                     'presented for the third antenatal care visit of their pregnancy on date %s at '
                     'gestation %d', person_id, self.sim.date, df.at[person_id, 'ps_gestational_age_in_weeks'])

        if df.at[person_id, 'is_alive'] and df.at[person_id, 'is_pregnant'] and ~df.at[person_id,
                                                                                       'la_currently_in_labour'] and \
            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] == 2:

            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1
            assert df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] == 3

            if df.at[person_id, 'ps_gestational_age_in_weeks'] < 30:
                self.module.ANCTracker['timely_ANC3'] += 1

            self.module.interventions_delivered_at_every_contact(hsi_event=self)
            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)

            # ========================================== Schedule next visit =======================================

            if df.at[person_id, 'ps_gestational_age_in_weeks'] < 30:
                self.module.iptp_administration(hsi_event=self)
                self.module.calcium_supplementation(hsi_event=self)
                self.module.gdm_screening(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=4,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif 30 >= df.at[person_id, 'ps_gestational_age_in_weeks'] < 34:
                self.module.iptp_administration(hsi_event=self)
                self.module.calcium_supplementation(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=4,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif 34 >= df.at[person_id, 'ps_gestational_age_in_weeks'] < 36:
                self.module.iptp_administration(hsi_event=self)
                self.module.calcium_supplementation(hsi_event=self)

                self.module.hep_b_testing(hsi_event=self)
                self.module.syphilis_testing(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=4,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif 36 >= df.at[person_id, 'ps_gestational_age_in_weeks'] < 38:
                self.module.calcium_supplementation(hsi_event=self)
                self.module.hb_testing(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=4,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif 38 >= df.at[person_id, 'ps_gestational_age_in_weeks'] < 40:
                self.module.calcium_supplementation(hsi_event=self)
                self.module.iptp_administration(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=4,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] >= 40:
                pass

            # todo: remove
            if df.at[person_id, 'ps_gestational_age_in_weeks'] < 30 and df.at[person_id,
                                                                              'ac_receiving_diet_supplements']:
                self.module.ANCTracker['diet_supp_6_months'] += 1

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        return actual_appt_footprint

    def did_not_run(self):
        pass

    def not_available(self):
        logger.debug('HSI_CareOfWomenDuringPregnancy_ThirdAntenatalCareContact: cannot not run with this configuration')


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
        logger.debug('This is HSI_CareOfWomenDuringPregnancy_FourthAntenatalCareContact, person %d has '
                     'presented for the fourth antenatal care visit of their pregnancy on date %s at '
                     'gestation %d', person_id, self.sim.date, df.at[person_id, 'ps_gestational_age_in_weeks'])

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

            elif 34 >= df.at[person_id, 'ps_gestational_age_in_weeks'] < 36:
                self.module.iptp_administration(hsi_event=self)
                self.module.calcium_supplementation(hsi_event=self)

                self.module.hep_b_testing(hsi_event=self)
                self.module.syphilis_testing(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=5,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif 36 >= df.at[person_id, 'ps_gestational_age_in_weeks'] < 38:
                self.module.calcium_supplementation(hsi_event=self)
                self.module.hb_testing(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=5,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif 38 >= df.at[person_id, 'ps_gestational_age_in_weeks'] < 40:
                self.module.calcium_supplementation(hsi_event=self)
                self.module.iptp_administration(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=5,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] >= 40:
                pass

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        return actual_appt_footprint

    def did_not_run(self):
        pass

    def not_available(self):
        logger.debug('HSI_CareOfWomenDuringPregnancy_FourthAntenatalCareContact: cannot not run with this configuration')


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

        logger.debug('This is HSI_CareOfWomenDuringPregnancy_FifthAntenatalCareContact, person %d has '
                     'presented for the fifth antenatal care visit of their pregnancy on date %s at '
                     'gestation %d', person_id, self.sim.date, df.at[person_id, 'ps_gestational_age_in_weeks'])

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

            elif 36 >= df.at[person_id, 'ps_gestational_age_in_weeks'] < 38:
                self.module.calcium_supplementation(hsi_event=self)
                self.module.hb_testing(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=6,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif 38 >= df.at[person_id, 'ps_gestational_age_in_weeks'] < 40:
                self.module.calcium_supplementation(hsi_event=self)
                self.module.iptp_administration(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=6,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] >= 40:
                pass

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        return actual_appt_footprint

    def did_not_run(self):
        pass

    def not_available(self):
        logger.debug('HSI_CareOfWomenDuringPregnancy_FifthAntenatalCareContact: cannot not run with this configuration')


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

        logger.debug('This is HSI_CareOfWomenDuringPregnancy_SixthAntenatalCareContact, person %d has '
                     'presented for the sixth antenatal care visit of their pregnancy on date %s at '
                     'gestation %d', person_id, self.sim.date, df.at[person_id, 'ps_gestational_age_in_weeks'])

        if df.at[person_id, 'is_alive'] and df.at[person_id, 'is_pregnant'] and \
            ~df.at[person_id, 'la_currently_in_labour'] and \
            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] == 5:

            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1
            assert df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] == 6

            self.module.interventions_delivered_at_every_contact(hsi_event=self)
            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)

            if df.at[person_id, 'ps_gestational_age_in_weeks'] < 38:
                self.module.calcium_supplementation(hsi_event=self)
                self.module.hb_testing(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=7,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif 38 >= df.at[person_id, 'ps_gestational_age_in_weeks'] < 40:
                self.module.calcium_supplementation(hsi_event=self)
                self.module.iptp_administration(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=7,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] >= 40:
                pass

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        return actual_appt_footprint

    def did_not_run(self):
        pass

    def not_available(self):
        logger.debug('HSI_CareOfWomenDuringPregnancy_SixthAntenatalCareContact: cannot not run with this configuration')


class HSI_CareOfWomenDuringPregnancy_SeventhAntenatalCareContact(HSI_Event, IndividualScopeEventMixin):
    """This is the  HSI_CareOfWomenDuringPregnancy_SeventhAntenatalCareContact. It will be scheduled by the
    HSI_CareOfWomenDuringPregnancy_SixthAntenatalCareContact.It will be responsible for the management of monitoring
    and treatment interventions delivered in a woman's seventh antenatal care visit. It will also go on the schedule the
    womans next ANC appointment"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_PresentsForPostAbortionCare'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
        self.ACCEPTED_FACILITY_LEVEL = 1  # 2/3?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        logger.debug('This is HSI_CareOfWomenDuringPregnancy_SeventhAntenatalCareContact, person %d has '
                     'presented for the seventh antenatal care visit of their pregnancy on date %s at '
                     'gestation %d', person_id, self.sim.date, df.at[person_id, 'ps_gestational_age_in_weeks'])

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

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        return actual_appt_footprint

    def did_not_run(self):
        pass

    def not_available(self):
        logger.debug('HSI_CareOfWomenDuringPregnancy_SeventhAntenatalCareContact: cannot not run with this '
                     'configuration')


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

        logger.debug('This is HSI_CareOfWomenDuringPregnancy_EighthAntenatalCareContact, person %d has '
                     'presented for the eighth antenatal care visit of their pregnancy on date %s at '
                     'gestation %d', person_id, self.sim.date, df.at[person_id, 'ps_gestational_age_in_weeks'])

        if df.at[person_id, 'is_alive'] and df.at[person_id, 'is_pregnant'] and \
            ~df.at[person_id, 'la_currently_in_labour'] and \
            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] == 7:

            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1
            assert df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] == 8
            self.module.ANCTracker['anc8+'] += 1

            self.module.interventions_delivered_at_every_contact(hsi_event=self)
            self.module.calcium_supplementation(hsi_event=self)

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        return actual_appt_footprint

    def did_not_run(self):
        pass

    def not_available(self):
        logger.debug('HSI_CareOfWomenDuringPregnancy_EighthAntenatalCareContact: cannot not run with this'
                     ' configuration')


class HSI_CareOfWomenDuringPregnancy_TestingForGestationalDiabetes(HSI_Event, IndividualScopeEventMixin):
    """This is HSI_CareOfWomenDuringPregnancy_TestingForGestationalDiabetes. This HSI is scheduled for women who screen
    positive for being at risk of gestational diabetes in pregnancy. At risk women are tested for hyperglycaemia and
    referred for treatment if tests are positive"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_TreatmentFollowingAntepartumStillbirth'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ANCSubsequent': 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        logger.debug('This is HSI_CareOfWomenDuringPregnancy_TestingForGestationalDiabetes, person %d has '
                     'presented for additional testing for gestational diabetes on date %s at gestation %d',
                     person_id, self.sim.date, df.at[person_id, 'ps_gestational_age_in_weeks'])

        if df.at[person_id, 'is_alive']:

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
                hsi_event=self,
                cons_req_as_footprint=consumables_gdm_testing,
                to_log=False)

            # If they are available, the test is conducted
            if outcome_of_request_for_consumables:

                # If the test accurately detects a woman has gestational diabetes the consumables are recorded and she
                # is referred for treatment
                if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run='blood_test_glucose',
                                                                           hsi_event=self):

                    self.sim.modules['HealthSystem'].request_consumables(
                        hsi_event=self,
                        cons_req_as_footprint=consumables_gdm_testing,
                        to_log=True)

                    gdm_treatment = HSI_CareOfWomenDuringPregnancy_InitiationOfTreatmentGestationalDiabetes(
                        self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id)

                    self.sim.modules['HealthSystem'].schedule_hsi_event(gdm_treatment, priority=0,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(days=7))

    def not_available(self):
        logger.debug('HSI_CareOfWomenDuringPregnancy_TestingForGestationalDiabetes: cannot not run with this'
                     ' configuration')

class HSI_CareOfWomenDuringPregnancy_InitiationOfTreatmentGestationalDiabetes(HSI_Event, IndividualScopeEventMixin):
    """This HSI is unfinished"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_InitiationOfTreatmentGestationalDiabetes'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        logger.debug('This is HSI_CareOfWomenDuringPregnancy_InitiationOfTreatmentGestationalDiabetes, person %d has '
                     'presented for treatment of gestational diabetes on date %s at gestation %d',
                     person_id, self.sim.date, df.at[person_id, 'ps_gestational_age_in_weeks'])

        # TODO: finish

    def not_available(self):
        logger.debug('HSI_CareOfWomenDuringPregnancy_InitiationOfTreatmentGestationalDiabetes: cannot not run with this'
                     ' configuration')


class HSI_CareOfWomenDuringPregnancy_InitialManagementOfGestationalHypertension(HSI_Event, IndividualScopeEventMixin):
    """This is HSI_ManagementOfHypertensiveDisorder. It is scheduled during ANC for women who are hypertensive during
    pregnancy."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_InitialManagementOfGestationalHypertension'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        logger.debug('This is HSI_CareOfWomenDuringPregnancy_InitialManagementOfGestationalHypertension, person %d has '
                     'presented for treatment of gestational hypertension on date %s at gestation %d',
                     person_id, self.sim.date, df.at[person_id, 'ps_gestational_age_in_weeks'])

        if df.at[person_id, 'is_alive']:

            # We check the availability of consumables, hydralazine for immediate stabilisation of BP followed by a
            # course of daily methyldopa for the remainder of pregnancy
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
                hsi_event=self,
                cons_req_as_footprint=consumables_gest_htn_treatment,
                to_log=False)

            # If they are available then the woman is started on treatment
            if outcome_of_request_for_consumables:
                df.at[person_id, 'ac_gest_htn_on_treatment'] = True

                self.sim.modules['HealthSystem'].request_consumables(
                    hsi_event=self,
                    cons_req_as_footprint=consumables_gest_htn_treatment,
                    to_log=True)

    def did_not_run(self):
        pass


class HSI_CareOfWomenDuringPregnancy_InitialManagementOfMildPreEclampsia(HSI_Event, IndividualScopeEventMixin):
    """This HSI is unfinished"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_InitialManagementOfPreEclampsia'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        logger.debug('This is HSI_CareOfWomenDuringPregnancy_InitialManagementOfMildPreEclampsia, person %d has '
                     'presented for treatment of mild pre-eclampsia on date %s at gestation %d',
                     person_id, self.sim.date, df.at[person_id, 'ps_gestational_age_in_weeks'])


class HSI_CareOfWomenDuringPregnancy_InitialManagementOfSeverePreEclampsia(HSI_Event, IndividualScopeEventMixin):
    """This HSI is unfinished"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_InitialManagementOfSeverePreEclampsia'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        logger.debug('This is HSI_CareOfWomenDuringPregnancy_InitialManagementOfSeverePreEclampsia, person %d has '
                     'presented for treatment of severe pre-eclampsia on date %s at gestation %d',
                     person_id, self.sim.date, df.at[person_id, 'ps_gestational_age_in_weeks'])


class HSI_CareOfWomenDuringPregnancy_InitialManagementOfMildAnaemiaInPregnancy(HSI_Event, IndividualScopeEventMixin):
    """This is HSI_CareOfWomenDuringPregnancy_InitialManagementOfMildAnaemiaInPregnancy. This HSI can be scheduled by
    any ANC appointment HSIs in which a woman is tested positive for mild anaemia. This HSI initiates treatment and
    future preventative treatment for mild anaemia and schedules a future checkup HSI to determine treatment success"""
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_ManagementOfMildAnaemiaInPregnancy'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        params = self.module.parameters

        logger.debug('This is HSI_CareOfWomenDuringPregnancy_InitialManagementOfMildAnaemiaInPregnancy, person %d has '
                     'presented for treatment of mild anaemia on date %s at gestation %d',
                     person_id, self.sim.date, df.at[person_id, 'ps_gestational_age_in_weeks'])

        # TODO: how to manage anaemia in women already receiving iron supplementation antenatally?
        # TODO: are we happy to assume that most women will be treated with iron OR should we allow for testing for
        #  folate, b12 deficiancies (and specific treatments)

        if df.at[person_id, 'is_alive']:

            # Currently the effect of this intervention is limited to women not already receiving daily iron
            if ~df.at[person_id, 'ac_receiving_iron_folic_acid']:

                # We check for consumables
                item_code_iron_folic_acid = pd.unique(
                    consumables.loc[
                        consumables['Items'] == 'Ferrous Salt + Folic Acid, tablet, 200 + 0.25 mg', 'Item_Code'])[0]

                consumables_mild_anaemia_treatment = {
                    'Intervention_Package_Code': {},
                    'Item_Code': {item_code_iron_folic_acid: 1}}

                outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                    hsi_event=self,
                    cons_req_as_footprint=consumables_mild_anaemia_treatment,
                    to_log=False)

                if outcome_of_request_for_consumables:

                    # We assume that treatment has two effects- it reduces the risk of anaemia redeveloping as pregnancy
                    # progresses AND corrects the current anaemia
                    df.at[person_id, 'ac_receiving_iron_folic_acid'] = True
                    logger.debug('mother %d has been started on daily iron and folic acid after a diagnosis of anaemia '
                                 'during ANC')

                    self.sim.modules['HealthSystem'].request_consumables(
                        hsi_event=self,
                        cons_req_as_footprint=consumables_mild_anaemia_treatment,
                        to_log=True)

                    if params['treatment_effect_iron_folate_anaemia'] < self.module.rng.random_sample():
                        df.at[person_id, 'ps_anaemia_in_pregnancy'] = False
                        logger.debug('mother %d has received initial iron and folic acid treatment for anaemia in '
                                     'pregnancy')

                # All women who are seen are scheduled to return between 2-4 weeks for additional Hb testing
                additional_care = HSI_CareOfWomenDuringPregnancy_MonitoringOfAnaemiaInPregnancy(
                    self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id)

                future_appt_date = 7 * 2 + self.module.rng.randint(0, 7 * 2)
                self.sim.modules['HealthSystem'].schedule_hsi_event(additional_care, priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date +
                                                                           DateOffset(days=future_appt_date))

                logger.debug('This is HSI_CareOfWomenDuringPregnancy_InitialManagementOfMildAnaemiaInPregnancy '
                             'scheduling follow-up Hb testing for a mother recently diagnosed with anaemia during '
                             'pregnancy', person_id)


class HSI_CareOfWomenDuringPregnancy_InitialManagementOfSevereAnaemiaInPregnancy(HSI_Event, IndividualScopeEventMixin):
    """This is HSI_CareOfWomenDuringPregnancy_InitialManagementOfSevereAnaemiaInPregnancy. This HSI can be scheduled by
        any ANC appointment HSIs in which a woman is tested positive for severe anaemia. This HSI initiates treatment
        and future preventative treatment for severe anaemia and schedules a future checkup HSI to determine
        treatment success"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_InitialManagementOfSevereAnaemiaInPregnancy'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        params = self.module.parameters

        logger.debug('This is HSI_CareOfWomenDuringPregnancy_InitialManagementOfSevereAnaemiaInPregnancy, person %d'
                     ' has presented for treatment of severe anaemia on date %s at gestation %d',
                     person_id, self.sim.date, df.at[person_id, 'ps_gestational_age_in_weeks'])

        if df.at[person_id, 'is_alive']:

            # Check for consumables
            item_code_bt1 = pd.unique(consumables.loc[consumables['Items'] == 'Blood, one unit', 'Item_Code'])[0]
            item_code_bt2 = pd.unique(consumables.loc[consumables['Items'] == 'Lancet, blood, disposable',
                                                      'Item_Code'])[0]
            item_code_bt3 = pd.unique(consumables.loc[consumables['Items'] == 'Test, hemoglobin', 'Item_Code'])[0]
            item_code_bt4 = pd.unique(consumables.loc[consumables['Items'] == 'IV giving/infusion set, with needle',
                                                      'Item_Code'])[0]
            consumables_needed_bt = {'Intervention_Package_Code': {}, 'Item_Code': {item_code_bt1: 2, item_code_bt2: 1,
                                                                                    item_code_bt3: 1, item_code_bt4: 2}}

            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self, cons_req_as_footprint=consumables_needed_bt, to_log=False)

            if outcome_of_request_for_consumables:

                # If availble, we apply a probability that a transfusion of 2 units RBCs will correct this womans severe
                # anaemia
                if params['treatment_effect_blood_transfusion_anaemia'] < self.module.rng.random_sample():
                        df.at[person_id, 'ps_anaemia_in_pregnancy'] = False

            # TODO: in reality its unlikely that anyone who had a BT would be discharged without an immediate FBC to
            #  check Hb, and if still anaemic would be transfused again?

            # And schedule follow up
            additional_care = HSI_CareOfWomenDuringPregnancy_MonitoringOfAnaemiaInPregnancy(
                    self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id)

            future_appt_date = 7 * 2 + self.module.rng.randint(0, 7 * 2)
            self.sim.modules['HealthSystem'].schedule_hsi_event(additional_care, priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date +
                                                                        DateOffset(days=future_appt_date))


class HSI_CareOfWomenDuringPregnancy_MonitoringOfAnaemiaInPregnancy(HSI_Event, IndividualScopeEventMixin):
    """This is HSI_CareOfWomenDuringPregnancy_MonitoringOfAnaemiaInPregnancy. This HSI can be scheduled by either
    HSI_CareOfWomenDuringPregnancy_InitialManagementOfMildAnaemiaInPregnancy or
    HSI_CareOfWomenDuringPregnancy_InitialManagementOfSevereAnaemiaInPregnancy and rechecks a womans Hb to determine if
    she remains anaemic"""
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'HSI_CareOfWomenDuringPregnancy_MonitoringOfAnaemiaInPregnancy'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        logger.debug(
            'This is HSI_CareOfWomenDuringPregnancy_MonitoringOfAnaemiaInPregnancy, person %d has '
            'presented for monitoring of anaemia status on date %s at gestation %d', person_id, self.sim.date,
            df.at[person_id, 'ps_gestational_age_in_weeks'])

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

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self,
            cons_req_as_footprint=consumables_hb_test,
            to_log=False)

        # Check consumables
        if outcome_of_request_for_consumables:

            self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self,
                cons_req_as_footprint=consumables_hb_test,
                to_log=True)

            # Run dx_test for anaemia
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run='blood_test_haemoglobin',
                                                                       hsi_event=self):
                # TODO: this will need to vary again by severity of anaemia diagnosed

                additional_care = HSI_CareOfWomenDuringPregnancy_InitialManagementOfMildAnaemiaInPregnancy(
                    self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id)

                self.sim.modules['HealthSystem'].schedule_hsi_event(additional_care, priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=7))


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

        total_anc1_visits = self.module.ANCTracker['total_first_anc_visits']
        if total_anc1_visits == 0:
            total_anc1_visits = 1

        anc1_in_first_trimester = self.module.ANCTracker['total_anc1_first_trimester']
        cumm_gestation = self.module.ANCTracker['cumm_ga_at_anc1']
        early_anc3 = self.module.ANCTracker['timely_ANC3']
        diet_sup_6_months = self.module.ANCTracker['diet_supp_6_months']

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

        logger.info('%s|anc_summary_stats|%s', self.sim.date, dict_for_output)

        self.module.ANCTracker = {'total_first_anc_visits': 0, 'cumm_ga_at_anc1': 0, 'total_anc1_first_trimester': 0,
                                  'anc8+': 0, 'timely_ANC3': 0, 'diet_supp_6_months':0}

