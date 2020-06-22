"""
The file contains the event HSI_GenericFirstApptAtFacilityLevel1, which describes the first interaction with
the health system following the onset of acute generic symptoms.
"""
from tlo import logging
from tlo.events import IndividualScopeEventMixin
from tlo.methods.chronicsyndrome import HSI_ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.labour import (
    HSI_Labour_PresentsForSkilledBirthAttendanceInLabour,
    HSI_Labour_ReceivesCareForPostpartumPeriod,
)
from tlo.methods.mockitis import HSI_Mockitis_PresentsForCareWithSevereSymptoms

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import pandas as pd


# ---------------------------------------------------------------------------------------------------------
#
#    ** NON-EMERGENCY APPOINTMENTS **
#
# ---------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------
#    HSI_GenericFirstApptAtFacilityLevel1
# ---------------------------------------------------------------------------------------------------------

class HSI_GenericFirstApptAtFacilityLevel1(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.

    It is the generic appointment that describes the first interaction with the health system following the onset of
    acute generic symptoms.

    It occurs at Facility_Level = 1

    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Confirm that this appointment has been created by a registered disease module or HealthSeekingBehaviour
        acceptable_originating_modules = list(self.sim.modules['HealthSystem'].registered_disease_modules.values())
        acceptable_originating_modules.append(self.sim.modules['HealthSeekingBehaviour'])
        assert module in acceptable_originating_modules

        # Work out if this is for a child or an adult
        is_child = self.sim.population.props.at[person_id, 'age_years'] < 5.0

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        if is_child:
            the_appt_footprint['Under5OPD'] = 1.0  # Child out-patient appointment
        else:
            the_appt_footprint['Over5OPD'] = 1.0  # Adult out-patient appointment

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'GenericFirstApptAtFacilityLevel1'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug('This is HSI_GenericFirstApptAtFacilityLevel1 for person %d', person_id)

        # Work out what to do with this person....
        if self.sim.population.props.at[person_id, 'age_years'] < 5.0:
            # It's a child:
            logger.debug('Run the ICMI algorithm for this child')

            # Get the diagnosis from the algorithm
            diagnosis = self.sim.modules['DxAlgorithmChild'].diagnose(person_id=person_id, hsi_event=self)

            # Do something based on this diagnosis...
            if diagnosis == 'measles':
                logger.debug('Start treatment for measles')
            else:
                logger.debug('No treatment. HSI ends.')

        else:
            # It's an adult
            logger.debug('To fill in ... what to with an adult')

            # ---- ASSESS FOR DEPRESSION ----
            if 'Depression' in self.sim.modules:
                depr = self.sim.modules['Depression']
                if (squeeze_factor == 0.0) and (self.module.rng.random() <
                                                depr.parameters['pr_assessed_for_depression_in_generic_appt_level1']):
                    depr.do_when_suspected_depression(person_id=person_id, hsi_event=self)
            # -------------------------------

    def did_not_run(self):
        logger.debug('HSI_GenericFirstApptAtFacilityLevel1: did not run')


# ---------------------------------------------------------------------------------------------------------
#    HSI_GenericFirstApptAtFacilityLevel0
# ---------------------------------------------------------------------------------------------------------

class HSI_GenericFirstApptAtFacilityLevel0(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.

    It is the generic appointment that describes the first interaction with the health system following the onset of
    acute generic symptoms.

    It occurs at Facility_Level = 0

    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Confirm that this appointment has been created by a registered disease module or HealthSeekingBehaviour
        acceptable_originating_modules = list(self.sim.modules['HealthSystem'].registered_disease_modules.values())
        acceptable_originating_modules.append(self.sim.modules['HealthSeekingBehaviour'])
        assert module in acceptable_originating_modules

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ConWithDCSA'] = 1.0  # Consultantion with DCSA

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'GenericFirstApptAtFacilityLevel0'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 0
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug('This is HSI_GenericFirstApptAtFacilityLevel0 for person %d', person_id)

    def did_not_run(self):
        logger.debug('HSI_GenericFirstApptAtFacilityLevel0: did not run')


# ---------------------------------------------------------------------------------------------------------
#
#    ** EMERGENCY APPOINTMENTS **
#
# ---------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------
#    HSI_GenericEmergencyFirstApptAtFacilityLevel1
# ---------------------------------------------------------------------------------------------------------

class HSI_GenericEmergencyFirstApptAtFacilityLevel1(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.

    It is the generic appointment that describes the first interaction with the health system following the onset of
    acute generic symptoms.

    It occurs at Facility_Level = 1

    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Confirm that this appointment has been created by a registered disease module or HealthSeekingBehaviour
        acceptable_originating_modules = list(self.sim.modules['HealthSystem'].registered_disease_modules.values())
        acceptable_originating_modules.append(self.sim.modules['HealthSeekingBehaviour'])
        assert module in acceptable_originating_modules

        # Work out if this is for a child or an adult
        is_child = self.sim.population.props.at[person_id, 'age_years'] < 5.0

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        if is_child:
            the_appt_footprint['Under5OPD'] = 1.0  # Child out-patient appointment
        else:
            the_appt_footprint['Over5OPD'] = 1.0  # Adult out-patient appointment

        df = self.sim.population.props
        # =============================== Adjust generic first appt for RTI requirements ===============================
        columns = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                   'rt_injury_7', 'rt_injury_8']
        persons_injuries = df.loc[[person_id], columns]

        def find_and_count_injuries(dataframe, tloinjcodes):
            index = pd.Index([])
            counts = 0
            for code in tloinjcodes:
                inj = dataframe.apply(lambda row: row.astype(str).str.contains(code).any(0), axis=1)
                if len(inj) > 0:
                    injidx = inj.index[inj]
                    counts += len(injidx)
                    index = index.union(injidx)
            return index, counts
        # ================================ Fractures require x-rays ====================================================
        fracture_codes = ['112', '113', '211', '212', '412', '414', '612', '712', '811', '812', '813']
        idx, counts = find_and_count_injuries(persons_injuries, fracture_codes)
        if len(idx) > 0:
            the_appt_footprint['DiagRadio'] = 1
        # ========================= Traumatic brain injuries require ct scan ===========================================
        codes = ['133', '134', '135']
        idx, counts = find_and_count_injuries(persons_injuries, codes)
        if len(idx) > 0:
            the_appt_footprint['Tomography'] = 1  # This appointment requires a ct scan
        # ============================= Abdominal trauma requires ct scan ==============================================
        codes = ['552', '553', '554']
        idx, counts = find_and_count_injuries(persons_injuries, codes)
        if len(idx) > 0:
            the_appt_footprint['Tomography'] = 1

        # ============================== Spinal cord injury require x ray ==============================================
        codes = ['673', '674', '675', '676']
        idx, counts = find_and_count_injuries(persons_injuries, codes)
        if len(idx) > 0:
            the_appt_footprint['DiagRadio'] = 1  # This appointment requires an x-ray

        # ============================== Dislocations require x ray ==============================================
        codes = ['322', '323', '722', '822']
        idx, counts = find_and_count_injuries(persons_injuries, codes)
        if len(idx) > 0:
            the_appt_footprint['DiagRadio'] = 1  # This appointment requires an x-ray

        # --------------------------------- Soft tissue injury in neck ---------------------------------------------
        codes = ['342', '343']
        idx, counts = find_and_count_injuries(persons_injuries, codes)
        if len(idx) > 0:
            the_appt_footprint['Tomography'] = 1  # This appointment requires a ct scan
            the_appt_footprint['DiagRadio'] = 1  # This appointment requires an x ray

        # --------------------------------- Soft tissue injury in thorax/ lung injury ------------------------------
        codes = ['441', '443', '453']
        idx, counts = find_and_count_injuries(persons_injuries, codes)
        if len(idx) > 0:
            the_appt_footprint['Tomography'] = 1  # This appointment requires a ct scan
            the_appt_footprint['DiagRadio'] = 1  # This appointment requires an x ray


        # -------------------------------- Internal bleeding -------------------------------------------------------
        codes = ['361', '363', '461', '463']
        idx, counts = find_and_count_injuries(persons_injuries, codes)
        idx2, counts = find_and_count_injuries(persons_injuries, ['461', '463'])
        if len(idx) > 0:
            the_appt_footprint['Tomography'] = 1  # This appointment requires a ct scan
            if len(idx2) > 0:
                the_appt_footprint['MinorSurg'] = 1

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'GenericEmergencyFirstApptAtFacilityLevel1'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.debug('This is HSI_GenericEmergencyFirstApptAtFacilityLevel1 for person %d', person_id)
        df = self.sim.population.props
        mni = self.sim.modules['Labour'].mother_and_newborn_info
        labour_list = self.sim.modules['Labour'].women_in_labour

        # simple diagnosis to work out which HSI event to trigger
        symptoms = self.sim.modules['SymptomManager'].has_what(person_id)

        # -----  COMPLICATION DURING BIRTH  -----
        if person_id in labour_list:
            if df.at[person_id, 'la_currently_in_labour'] & (mni[person_id]['sought_care_for_complication']) \
                    & (mni[person_id]['sought_care_labour_phase'] == 'intrapartum'):
                event = HSI_Labour_PresentsForSkilledBirthAttendanceInLabour(
                    module=self.sim.modules['Labour'], person_id=person_id,
                    facility_level_of_this_hsi=int(self.module.rng.choice([1, 2])))
                self.sim.modules['HealthSystem'].schedule_hsi_event(event, priority=1, topen=self.sim.date)

        # -----  COMPLICATION AFTER BIRTH  -----
            if df.at[person_id, 'la_currently_in_labour'] & (mni[person_id]['sought_care_for_complication']) \
                    & (mni[person_id]['sought_care_labour_phase'] == 'postpartum'):
                event = HSI_Labour_ReceivesCareForPostpartumPeriod(
                    module=self.sim.modules['Labour'], person_id=person_id,
                    facility_level_of_this_hsi=int(self.module.rng.choice([1, 2])))
                self.sim.modules['HealthSystem'].schedule_hsi_event(event, priority=1, topen=self.sim.date)

        # -----  SUSPECTED DEPRESSION  -----
        if 'em_Injuries_From_Self_Harm' in symptoms:
            self.sim.modules['Depression'].do_when_suspected_depression(person_id=person_id, hsi_event=self)
            # TODO: Trigger surgical care for injuries.

        # -----  EXAMPLES FOR MOCKITIS AND CHRONIC SYNDROME  -----
        if 'em_craving_sandwiches' in symptoms:
            event = HSI_ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment(
                module=self.sim.modules['ChronicSyndrome'],
                person_id=person_id
            )
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=1,
                                                                topen=self.sim.date
                                                                )

        if 'em_extreme_pain_in_the_nose' in symptoms:
            event = HSI_Mockitis_PresentsForCareWithSevereSymptoms(
                module=self.sim.modules['Mockitis'],
                person_id=person_id
            )
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=1,
                                                                topen=self.sim.date
                                                                )

        if 'RTI' in self.sim.modules:
            if 'em_severe_trauma' in symptoms:
                df = self.sim.population.props
                consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

                columns = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                           'rt_injury_7', 'rt_injury_8']
                persons_injuries = df.loc[[person_id], columns]

                def find_and_count_injuries(dataframe, tloinjcodes):
                    index = pd.Index([])
                    counts = 0
                    for code in tloinjcodes:
                        inj = dataframe.apply(lambda row: row.astype(str).str.contains(code).any(0), axis=1)
                        if len(inj) > 0:
                            injidx = inj.index[inj]
                            counts += len(injidx)
                            index = index.union(injidx)
                    return index, counts
                # Request multiple x-rays here, note that the diagradio requirement for the appointment footprint
                # is dealt with in the RTI module itself.

                fracture_codes = ['112', '113', '211', '212', '412', '414', '612', '712', '811', '812', '813']
                idx, counts = find_and_count_injuries(persons_injuries, fracture_codes)
                if counts > 1:
                    pkg_code_x_ray = pd.unique(
                        consumables.loc[consumables['Intervention_Pkg'] ==
                                        'Treatment of injuries (Fracture and dislocation)',
                                        'Intervention_Pkg_Code'])[0]
                    item_code_x_ray_film = pd.unique(
                        consumables.loc[consumables['Items'] ==
                                        "Monochromatic blue senstive X-ray Film, screen SizeSize: 30cm x 40cm",
                                        'Item_Code'])[0]
                    consumables_x_ray = {
                        'Intervention_Package_Code': {pkg_code_x_ray: 1},
                        'Item_Code': {item_code_x_ray_film: counts}}
                    is_cons_available_1 = self.sim.modules['HealthSystem'].request_consumables(
                        hsi_event=self,
                        cons_req_as_footprint=consumables_x_ray,
                        to_log=False)
                    cond = is_cons_available_1['Intervention_Package_Code'][pkg_code_x_ray]
                    if cond:
                        logger.debug(
                            'This facility has x-ray capability which has been used to diagnose person %d.',
                            person_id)
                        logger.debug(f'Person %d had x-rays for their {counts} fractures')
                    else:
                        logger.debug('Total amount of x-rays required for person %d unavailable', person_id)
                df.loc[person_id, 'rt_diagnosed'] = True
                road_traffic_injuries = self.sim.modules['RTI']
                road_traffic_injuries.rti_do_when_injured(person_id=person_id, hsi_event=self)

    def did_not_run(self):
        logger.debug('HSI_GenericEmergencyFirstApptAtFacilityLevel1: did not run')
        return False  # Labour debugging
        # pass

    def not_available(self):
        pass
