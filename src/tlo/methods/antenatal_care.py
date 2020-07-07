from pathlib import Path

import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.dxmanager import DxTest

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CareOfWomenDuringPregnancy(Module):
    """This is the CareOfWomenDuringPregnancy module. This module houses all HSIs related to care during the antenatal
    period. Currently the module houses all 8 antenatal care contacts and manages the scheduling of all additional
    antenatal care contacts after visit 1 (which is scheduled in the pregnancy supervisor module. The majority of this
    module remains hollow prior to completion for June 2020 deadline"""

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
        'sensitivity_bp_monitoring': Parameter(
            Types.REAL, 'sensitivity of blood pressure monitoring to detect hypertension'),
        'specificity_bp_monitoring': Parameter(
            Types.REAL, 'specificity of blood pressure monitoring to detect hypertension'),
        'sensitivity_urine_protein': Parameter(
            Types.REAL, 'sensitivity of a urine dipstick test to detect proteinuria'),
        'specificity_urine_protein': Parameter(
            Types.REAL, 'specificity of a urine dipstick test to detect proteinuia'),
        'sensitivity_urine_glucose': Parameter(
            Types.REAL, 'sensitivity of a urine dipstick test to detect glycosuria'),
        'specificity_urine_glucose': Parameter(
            Types.REAL, 'sensitivity of a urine dipstick test to detect glycosuria'),
        'sensitivity_blood_test_hb': Parameter(
            Types.REAL, 'sensitivity of a blood test to detect low haemoglobin'),
        'specificity_blood_test_hb': Parameter(
            Types.REAL, 'specificity of a blood test to detect low haemoglobin'),

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

    def initialise_simulation(self, sim):
        sim.schedule_event(AntenatalCareLoggingEvent(self),
                           sim.date + DateOffset(years=1))

        # Populate the tracker
        self.ANCTracker = {'total_first_anc_visits': 0, 'cumm_ga_at_anc1': 0, 'total_anc1_first_trimester': 0,
                           'anc8+': 0}

        # DX_TESTS
        params = self.parameters

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            blood_pressure_measurement=DxTest(
                property='ps_currently_hypertensive',
                sensitivity=params['sensitivity_bp_monitoring'],
                specificity=params['specificity_bp_monitoring']),

            urine_dipstick_protein=DxTest(
                property='ps_mild_pre_eclamp',
                sensitivity=params['sensitivity_urine_protein'],
                specificity=params['specificity_urine_protein']),
            # TODO: Categorical function of dx_test doesnt allow for multiple categories to be selected from a list
            #  (here it would be both mild and severe pre-eclampsia)

            urine_dipstick_sugars=DxTest(
                property='ps_gest_diab',
                sensitivity=params['sensitivity_urine_glucose'],
                specificity=params['specificity_urine_glucose']),

            #    urine_dipstick_infection=DxTest(
            #        property='???',
            #        sensitivity=0.9,
            #        specificity=0.9),

            blood_test_haemoglobin=DxTest(
                property='ps_anaemia_in_pregnancy',
                sensitivity=params['sensitivity_blood_test_hb'],
                specificity=params['specificity_blood_test_hb']),
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

        # TODO : ensure we're not using the variable postnatally, if so will need to be reset later

    def on_hsi_alert(self, person_id, treatment_id):
        logger.debug('This is CareOfWomenDuringPregnancy, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

    def determine_gestational_age_for_next_contact(self, person_id):
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
    # Following functions contain code for the delivery of interventions within ANC contacts

    def interventions_delivered_at_every_contact(self, hsi_event):
        """This function houses all the interventions that should be delivered at every ANC contact regardless of
        gestational age"""
        person_id = hsi_event.target
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        df = self.sim.population.props
        params = self.parameters

        # Blood pressure measurement...
        # We apply a probability that the HCW will perform a blood pressure check during this visit
        if self.rng.random_sample() < params['prob_bp_check']:

            # If so, we use the dx_test to determine if this check will correctly identify a woman's hypertension
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
              dx_tests_to_run='blood_pressure_measurement', hsi_event=hsi_event):

                # We store this as a variable and determine action after urine dipstick is carried out
                hypertension_diagnosed = True
            else:
                # False here means either the woman is normatensive or she has hypertension which was missed
                hypertension_diagnosed = False
        else:
            # as above
            hypertension_diagnosed = False

        # Urine dipstick- protein...
        # Next we apply a probability that the HCW will perform a urine dipstick
        if self.rng.random_sample() < params['prob_urine_dipstick']:

            # If so, the dx_test determines if this test will correctly identify proteinurea
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
              dx_tests_to_run='urine_dipstick_protein', hsi_event=hsi_event):
                protein_urea_diagnosed = True
            else:
                protein_urea_diagnosed = False

            # Urine - sugar...
            # Similarly the dx_test determines if the test identified glucosuria
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
              dx_tests_to_run='urine_dipstick_sugars', hsi_event=hsi_event):

                # If the HCW detects glucose in the woman's urine, they are scheduled to undergo additional
                # investigation/treatment in an additional HSI
                additional_care = HSI_CareOfWomenDuringPregnancy_ManagementOfGestationalDiabetes(
                    self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id)

                self.sim.modules['HealthSystem'].schedule_hsi_event(additional_care, priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=7))
        else:
            protein_urea_diagnosed = False

        # If hypertension is diagnosed without protein urea being detected the woman is scheduled for additional care
        # using a cause parameter to dictated actions at the next HSI
        if hypertension_diagnosed and ~protein_urea_diagnosed:
            additional_care = HSI_CareOfWomenDuringPregnancy_ManagementOfHypertensiveDisorder(
                self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id, cause='gest_htn')

            self.sim.modules['HealthSystem'].schedule_hsi_event(additional_care, priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=7))

        # Similarly, it is assumed that hypertension in the presence of proteinuria is due to pre-eclampsia and
        # scheduling occurs
        elif hypertension_diagnosed and protein_urea_diagnosed:
            additional_care = HSI_CareOfWomenDuringPregnancy_ManagementOfHypertensiveDisorder(
                self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id, cause='pre_eclamp')

            self.sim.modules['HealthSystem'].schedule_hsi_event(additional_care, priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=7))

        # Iron & folic acid / food supplementation...
        # Here we document the required consumables for drugs administered in this visit
        item_code_iron_folic_acid = pd.unique(
            consumables.loc[consumables['Items'] == 'Ferrous Salt + Folic Acid, tablet, 200 + 0.25 mg', 'Item_Code'])[0]
        item_code_diet_supps = pd.unique(
            consumables.loc[consumables['Items'] == 'Dietary supplements (country-specific)', 'Item_Code'])[0]

        # todo: this is a hacky quick fix for bug that i cant work out- sort properly
        if self.determine_gestational_age_for_next_contact(person_id) is None and df.at[person_id,
                                                                                        'ps_gestational_age_in_weeks'] \
                                                                                   == 39:
            next_visit = 40

            days_until_next_contact = int(next_visit - df.at[person_id, 'ps_gestational_age_in_weeks']) * 7

        else:
            days_until_next_contact = int(self.determine_gestational_age_for_next_contact(person_id) -
                                          df.at[person_id, 'ps_gestational_age_in_weeks']) * 7

        # And provide women with enough medication until the next visit
        consumables_anc_1 = {
            'Intervention_Package_Code': {},
            'Item_Code': {item_code_iron_folic_acid: days_until_next_contact,
                          item_code_diet_supps: days_until_next_contact}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=hsi_event,
            cons_req_as_footprint=consumables_anc_1,
            to_log=True)

        # Both availability of consumables and likelihood of practitioner deciding to initiate treatment determines if
        # the intervention is delivered...
        if outcome_of_request_for_consumables['Item_Code'][item_code_iron_folic_acid] and \
          self.rng.random_sample() < params['prob_start_iron_folic_acid']:
            df.at[person_id, 'ac_receiving_iron_folic_acid'] = True

            # We store the date at which this prescription will run out and the woman is no longer experiencing the
            # benefits of this treatment
            df.at[person_id, 'ac_date_ifa_runs_out'] = self.sim.date + DateOffset(days=days_until_next_contact)

        if outcome_of_request_for_consumables['Item_Code'][item_code_diet_supps] and \
          self.rng.random_sample() < params['prob_start_diet_supps_acid']:
            df.at[person_id, 'ac_receiving_diet_supplements'] = True
            df.at[person_id, 'ac_date_ds_runs_out'] = self.sim.date + DateOffset(days=days_until_next_contact)

    def interventions_delivered_only_at_first_contact(self, hsi_event):
        """ This function houses the additional interventions that should be delivered at a womans first ANC contact not
        included in the above function"""
        person_id = hsi_event.target
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        df = self.sim.population.props
        params = self.parameters

        # TODO: Discuss the following interventions with Tara...
        # LLITN
        # Tetanus

    def calcium_supplementation(self, hsi_event):
        """This function manages the intervention calcium supplementation"""
        df = self.sim.population.props
        params = self.parameters
        person_id = hsi_event.target
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        # TODO: Add in assessment of high risk women?

        item_code_calcium_supp = pd.unique(
            consumables.loc[consumables['Items'] == 'Calcium, tablet, 600 mg', 'Item_Code'])[0]

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
            to_log=True)

        if outcome_of_request_for_consumables['Item_Code'][item_code_calcium_supp] and \
            self.rng.random_sample() < params['prob_start_calcium_supp']:
            df.at[person_id, 'ac_receiving_calcium_supplements'] = True
            df.at[person_id, 'ac_date_cal_runs_out'] = self.sim.date + DateOffset(days=converted_days)

    def hb_testing(self, hsi_event):
        person_id = hsi_event.target
        params = self.parameters

        # Blood- Hb
        if self.rng.random_sample() < params['prob_blood_test']:
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run='blood_test_haemoglobin',
                                                                       hsi_event=hsi_event):
                additional_care = HSI_CareOfWomenDuringPregnancy_ManagementOfAnaemiaInPregnancy(
                    self.sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id)

                self.sim.modules['HealthSystem'].schedule_hsi_event(additional_care, priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=7))

    def hep_b_testing(self, hsi_event):
        pass

    def syphilis_testing(self, hsi_event):
        pass

    def hiv_testing(self, hsi_event):
        pass

    def albendazole_administration(self, hsi_event):
        pass

    def iptp_administration(self, hsi_event):
        pass

    def anc_interventions_contacts_2_to_8(self, hsi_event):
        """This function actions all the interventions a woman presenting to ANC1 at >20 will need administering."""
        self.hiv_testing(hsi_event=hsi_event)
        self.hep_b_testing(hsi_event=hsi_event)
        self.syphilis_testing(hsi_event=hsi_event)
        self.hb_testing(hsi_event=hsi_event)

        self.albendazole_administration(hsi_event=hsi_event)
        self.iptp_administration(hsi_event=hsi_event)
        self.calcium_supplementation(hsi_event=hsi_event)

    def antenatal_care_scheduler(self, individual_id, visit_to_be_scheduled, recommended_gestation_next_anc):
        """This function is responsible for scheduling a womans next antenatal care visit. The function is provided with
        the number of the next visit a woman is required to attend along with the recommended gestational age a woman
        should be to attend the next visit in the schedule"""
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
            if visit_number < 4:
                if df.at[individual_id, 'ps_will_attend_four_or_more_anc']:

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
                    # Women who were not predicted to attend ANC4+ will have a probability applied that they will not
                    # continue with ANC contacts

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
            elif visit_number >= 4:
                # Here we block women who are not predicted to attend ANC4+ from doing so
                if ~df.at[individual_id, 'ps_will_attend_four_or_more_anc']:
                    return

                else:
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
    """ This is the HSI HSI_FirstAntenatalCareContact. It will be scheduled by the PregnancySupervisor Module.
    It will be responsible for the management of monitoring and treatment interventions delivered in a woman's first
    antenatal care visit. It will also go on the schedule the womans next ANC appointment. It is currently unfinished"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_FirstAntenatalCareVisit'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['AntenatalFirst'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        logger.debug('This is HSI_CareOfWomenDuringPregnancy_PresentsForFirstAntenatalCareVisit, person %d has '
                     'presented for the first antenatal care visit of their pregnancy on date %s at '
                     'gestation %d', person_id, self.sim.date, df.at[person_id, 'ps_gestational_age_in_weeks'])

        # We condition this event on the woman being alive, pregnant and not currently in labour
        if df.at[person_id, 'is_alive'] and df.at[person_id, 'is_pregnant'] and ~df.at[person_id,
                                                                                       'la_currently_in_labour']:

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

            # First, interventions that should be delivered at every ANC visit are administered
            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)
            print(gest_age_next_contact)

            # TODO: may be better to save this as a property is we keep using it

            self.module.interventions_delivered_at_every_contact(hsi_event=self)

            # If this woman is presenting prior to the suggested gestation for ANC2, she receives only the interventions
            # for ANC1
            if df.at[person_id, 'ps_gestational_age_in_weeks'] < 20:
                # These are the interventions delivered at ANC1
                self.module.interventions_delivered_only_at_first_contact(hsi_event=self)
                self.module.hiv_testing(hsi_event=self)
                self.module.hep_b_testing(hsi_event=self)
                self.module.syphilis_testing(hsi_event=self)
                self.module.hb_testing(hsi_event=self)

                # She is then assessed to see if she will attend the next ANC contact in the schedule
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=2,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            # If she presents after the suggested gestation for ANC2, she receives the interventions for ANC1 and ANC2
            elif 20 >= df.at[person_id, 'ps_gestational_age_in_weeks'] < 26:
                self.module.interventions_delivered_only_at_first_contact(hsi_event=self)
                self.module.anc_interventions_contacts_2_to_8(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=2,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            # This pattern continues so that any woman presenting late for ANC will receive all the interventions they
            # have missed from previous visits
            elif 26 >= df.at[person_id, 'ps_gestational_age_in_weeks'] < 30:
                self.module.interventions_delivered_only_at_first_contact(hsi_event=self)
                self.module.anc_interventions_contacts_2_to_8(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=2,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif 30 >= df.at[person_id, 'ps_gestational_age_in_weeks'] < 34:
                self.module.interventions_delivered_only_at_first_contact(hsi_event=self)
                self.module.anc_interventions_contacts_2_to_8(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=2,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif 34 >= df.at[person_id, 'ps_gestational_age_in_weeks'] < 36:
                self.module.interventions_delivered_only_at_first_contact(hsi_event=self)
                self.module.anc_interventions_contacts_2_to_8(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=2,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif 36 >= df.at[person_id, 'ps_gestational_age_in_weeks'] < 38:
                self.module.interventions_delivered_only_at_first_contact(hsi_event=self)
                self.module.anc_interventions_contacts_2_to_8(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=2,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif 38 >= df.at[person_id, 'ps_gestational_age_in_weeks'] < 40:
                self.module.interventions_delivered_only_at_first_contact(hsi_event=self)
                self.module.anc_interventions_contacts_2_to_8(hsi_event=self)

                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=2,
                                                     recommended_gestation_next_anc=gest_age_next_contact)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] >= 40:
                self.module.interventions_delivered_only_at_first_contact(hsi_event=self)
                self.module.anc_interventions_contacts_2_to_8(hsi_event=self)

                # todo: for now, these women just wont have another visit

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareVisit: did not run')

    def not_available(self):
        logger.debug('HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareVisit: cannot not run with this configuration')


class HSI_CareOfWomenDuringPregnancy_SecondAntenatalCareContact(HSI_Event, IndividualScopeEventMixin):
    """"""
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

        if df.at[person_id, 'is_alive'] and df.at[person_id, 'is_pregnant'] and ~df.at[person_id,
                                                                                       'la_currently_in_labour']:
            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1
            assert df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] == 2

            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)
            self.module.interventions_delivered_at_every_contact(hsi_event=self)

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
    """"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_ThirdAntenatalCareVisit'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        if df.at[person_id, 'is_alive'] and df.at[person_id, 'is_pregnant'] and ~df.at[person_id,
                                                                                       'la_currently_in_labour']:
            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1
            assert df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] == 3

            self.module.interventions_delivered_at_every_contact(hsi_event=self)
            gest_age_next_contact = self.module.determine_gestational_age_for_next_contact(person_id)

            # ========================================== Schedule next visit =======================================

            if df.at[person_id, 'ps_gestational_age_in_weeks'] < 30:
                self.module.iptp_administration(hsi_event=self)
                self.module.calcium_supplementation(hsi_event=self)

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

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        return actual_appt_footprint

    def did_not_run(self):
        pass

    def not_available(self):
        logger.debug('HSI_CareOfWomenDuringPregnancy_ThirdAntenatalCareContact: cannot not run with this configuration')


class HSI_CareOfWomenDuringPregnancy_FourthAntenatalCareContact(HSI_Event, IndividualScopeEventMixin):
    """"""

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

        if df.at[person_id, 'is_alive'] and df.at[person_id, 'is_pregnant'] and ~df.at[person_id,
                                                                                       'la_currently_in_labour']:
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
    """"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_FifthAntenatalCareVisit'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        if df.at[person_id, 'is_alive'] and df.at[person_id, 'is_pregnant'] and ~df.at[person_id,
                                                                                       'la_currently_in_labour']:
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
    """"""

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

        if df.at[person_id, 'is_alive'] and df.at[person_id, 'is_pregnant'] and ~df.at[person_id,
                                                                                       'la_currently_in_labour']:
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
    """"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_SeventhAntenatalCareVisit'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        if df.at[person_id, 'is_alive'] and df.at[person_id, 'is_pregnant'] and ~df.at[person_id,
                                                                                       'la_currently_in_labour']:
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
    """"""

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

        if df.at[person_id, 'is_alive'] and df.at[person_id, 'is_pregnant'] and ~df.at[person_id,
                                                                                       'la_currently_in_labour']:
            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1
            assert df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] == 8

            self.module.interventions_delivered_at_every_contact(hsi_event=self)
            self.module.calcium_supplementation(hsi_event=self)

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        return actual_appt_footprint

    def did_not_run(self):
        pass

    def not_available(self):
        logger.debug('HSI_CareOfWomenDuringPregnancy_EighthAntenatalCareContact: cannot not run with this configuration')


class HSI_CareOfWomenDuringPregnancy_ManagementOfHypertensiveDisorder(HSI_Event, IndividualScopeEventMixin):
    """"""

    def __init__(self, module, person_id, cause):
        super().__init__(module, person_id=person_id)
        self.cause = cause

        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_ManagementOfHypertensiveDisorder'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        if self.cause == 'gest_htn':
            pass
        elif self.cause == 'pre_eclamp':
            pass

class HSI_CareOfWomenDuringPregnancy_ManagementOfGestationalDiabetes(HSI_Event, IndividualScopeEventMixin):
    """"""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_ManagementOfGestationalDiabetes'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

class HSI_CareOfWomenDuringPregnancy_ManagementOfAnaemiaInPregnancy(HSI_Event, IndividualScopeEventMixin):
    """"""
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_ManagementOfAnaemiaInPregnancy'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props


class AntenatalCareLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(years=self.repeat))

    def apply(self, population):
        df = self.sim.population.props

        total_anc1_visits = self.module.ANCTracker['total_first_anc_visits']
        if total_anc1_visits == 0:
            total_anc1_visits = 1

        anc1_in_first_trimester = self.module.ANCTracker['total_anc1_first_trimester']
        cumm_gestation = self.module.ANCTracker['cumm_ga_at_anc1']

        ra_lower_limit = 14
        ra_upper_limit = 50
        women_reproductive_age = df.index[(df.is_alive & (df.sex == 'F') & (df.age_years > ra_lower_limit) &
                                           (df.age_years < ra_upper_limit))]
        total_women_reproductive_age = len(women_reproductive_age)

        dict_for_output = {'mean_ga_first_anc': cumm_gestation/total_anc1_visits,
                           'proportion_anc1_first_trimester': (anc1_in_first_trimester/ total_anc1_visits) * 100}

        # TODO: check logic for ANC4+ calculation

        logger.info('%s|anc_summary_stats|%s', self.sim.date, dict_for_output)

        self.module.ANCTracker = {'total_first_anc_visits': 0, 'cumm_ga_at_anc1': 0, 'total_anc1_first_trimester': 0,
                                  'anc8+': 0}

