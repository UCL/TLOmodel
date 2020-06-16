from pathlib import Path

import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods.healthsystem import HSI_Event

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CareOfWomenDuringPregnancy(Module):
    """This is the Antenatal Care module. It is responsible for calculating probability of antenatal care seeking and
    houses all Health System Interaction events pertaining to monitoring and treatment of women during the antenatal
    period of their pregnancy. The majority of this module remains hollow prior to completion for June 2020 deadline"""

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        self.ANCTracker = dict()


    PARAMETERS = {
        'prob_seek_care_first_anc': Parameter(
            Types.REAL, 'Probability a woman will access antenatal care for the first time'),  # DUMMY PARAMETER
        'odds_first_anc': Parameter(
            Types.REAL, 'odds of a pregnant women ever seeking to initiate first ANC visit'),
        'or_anc_unmarried': Parameter(
            Types.REAL, 'odds ratio of first ANC visit for unmarried women'),
        'or_anc_wealth_4': Parameter(
            Types.REAL, 'odds ration of first ANC for a woman of wealth level 4'),
        'or_anc_wealth_5': Parameter(
            Types.REAL, 'odds ration of first ANC for a woman of wealth level 5'),
        'or_anc_urban': Parameter(
            Types.REAL, 'odds ration of first ANC for women living in an urban setting'),
    }

    PROPERTIES = {
        'ac_total_anc_visits_current_pregnancy': Property(
            Types.INT,
            'rolling total of antenatal visits this woman has attended during her pregnancy'),
    }

    def read_parameters(self, data_folder):
        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_AntenatalCare.xlsx',
                            sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)

    def initialise_population(self, population):

        df = population.props
        df.loc[df.is_alive, 'ac_total_anc_visits_current_pregnancy'] = 0

    def initialise_simulation(self, sim):
        sim.schedule_event(AntenatalCareLoggingEvent(self),
                           sim.date + DateOffset(years=1))

        self.ANCTracker = {'total_first_anc_visits': 0, 'cumm_ga_at_anc1': 0, 'total_anc1_first_trimester': 0,
                           'anc4+': 0, 'anc8+': 0}

    def on_birth(self, mother_id, child_id):
        df = self.sim.population.props
        df.at[child_id, 'ac_total_anc_visits_current_pregnancy'] = 0

        logger.info('%s|total_anc_per_woman|%s', self.sim.date,
                    {'person_id': mother_id,
                     'age': df.at[mother_id, 'age_years'],
                     'total_anc': df.at[mother_id, 'ac_total_anc_visits_current_pregnancy']})

        # TODO : ensure we're not using the variable postnatally, if so will need to be reset later
        df.at[mother_id, 'ac_total_anc_visits_current_pregnancy'] = 0

    def on_hsi_alert(self, person_id, treatment_id):
        logger.debug('This is CareOfWomenDuringPregnancy, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

    # TODO: here we will store all the interventions that should be carried out at each visit, we will then use
    # gestational age to determine whihc pacakages of interventions should be delivered at each visit (See doc)

    def interventions_for_first_anc_visit(self, individual_id):
        pass

    def interventions_for_second_anc_visit(self, individual_id):
        pass

    def interventions_for_third_anc_visit(self, individual_id):
        pass

    def interventions_for_fourth_anc_visit(self, individual_id):
        pass

    def antenatal_care_scheduler(self, individual_id, visit_to_be_scheduled, recommended_gestation_next_anc):
        """"""
        df = self.sim.population.props

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

            elif visit_number >= 5:
                visit = HSI_CareOfWomenDuringPregnancy_AdditionalAntenatalCareContact(
                 self, person_id=individual_id)

            # There are a number of variables that determine if a woman will attend another ANC visit:
            # 1.) If she is predicted to attend > 4 visits
            # 2.) Her gestational age at presentation to this ANC visit
            # 3.) The recommended gestational age for each ANC contact and how that matches to the womans current
            # gestational age

            # If this woman has attended less than 4 visits, and is predicted to attend > 4. Her subsequent ANC
            # appointment is automatically scheduled
            if visit_number <= 4:
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

                    # TODO: this probability should be modifiable in the linear model, with disease status as a
                    #  predictor?
                    random = self.rng.choice(['stop', 'continue'], p=[0.2, 0.8])
                    if random == 'stop':
                        logger.debug('mother %d will not seek any additional antenatal care for this pregnancy',
                                      individual_id)
                    else:
                        weeks_due_next_visit = int(recommended_gestation_next_anc - df.at[individual_id,
                                                                                          'ps_gestational_age_in_'
                                                                                          'weeks'])
                        visit_date = self.sim.date + DateOffset(weeks=weeks_due_next_visit)
                        self.sim.modules['HealthSystem'].schedule_hsi_event(visit, priority=0,
                                                                            topen=visit_date,
                                                                            tclose=visit_date + DateOffset(days=7))

            elif visit_number > 4:
                # Here we block women who are not predicted to attend ANC4+ from doing so
                if ~df.at[individual_id, 'ps_will_attend_four_or_more_anc']:
                    return

                else:
                    random = self.rng.choice(['stop', 'continue'], p=[0.2, 0.8])
                    if random == 'stop':
                        logger.debug('mother %d will not seek any additional antenatal care for this pregnancy',
                                     individual_id)
                    else:
                        weeks_due_next_visit = int(recommended_gestation_next_anc - df.at[individual_id,
                                                                                          'ps_gestational_age_in_'
                                                                                          'weeks'])
                        visit_date = self.sim.date + DateOffset(weeks=weeks_due_next_visit)
                        self.sim.modules['HealthSystem'].schedule_hsi_event(visit, priority=0,
                                                                            topen=visit_date,
                                                                            tclose=visit_date + DateOffset(days=7))

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

        df.at[person_id,'ac_attended_first_anc'] = True

        if df.at[person_id, 'is_alive'] and df.at[person_id, 'is_pregnant'] and ~df.at[person_id,
                                                                                       'la_currently_in_labour']:
            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            # DEBUGGING FOR NOW
            self.module.ANCTracker['cumm_ga_at_anc1'] += df.at[person_id, 'ps_gestational_age_in_weeks']
            self.module.ANCTracker['total_first_anc_visits'] += 1
            if df.at[person_id, 'ps_gestational_age_in_weeks'] < 14:
                self.module.ANCTracker['total_anc1_first_trimester'] += 1

            # ========================================== Schedule next visit =======================================
            if df.at[person_id, 'ps_gestational_age_in_weeks'] <= 20:
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=2,
                                                     recommended_gestation_next_anc=20)

            elif 20 > df.at[person_id, 'ps_gestational_age_in_weeks'] <= 26:
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=2,
                                                     recommended_gestation_next_anc=26)

            elif 26 > df.at[person_id, 'ps_gestational_age_in_weeks'] <= 30:
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=2,
                                                     recommended_gestation_next_anc=32)

            elif 30 > df.at[person_id, 'ps_gestational_age_in_weeks'] <= 34:
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=2,
                                                     recommended_gestation_next_anc=34)

            elif 34 > df.at[person_id, 'ps_gestational_age_in_weeks'] <= 36:
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=2,
                                                     recommended_gestation_next_anc=36)

            elif 36 > df.at[person_id, 'ps_gestational_age_in_weeks'] <= 38:
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=2,
                                                     recommended_gestation_next_anc=38)

            elif 38 > df.at[person_id, 'ps_gestational_age_in_weeks'] <= 40:
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=2,
                                                     recommended_gestation_next_anc=40)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] > 40:
                pass

            # todo: for now, these women just wont have another visit

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_CareOfWomenDuringPregnancy_FirstAntenatalCareVisit: did not run')

    def not_available(self):
        pass


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

        if df.at[person_id, 'is_alive'] and df.at[person_id, 'is_pregnant'] and \
            ~df.at[person_id, 'la_currently_in_labour']:
            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            # ========================================== Schedule next visit =======================================
            if df.at[person_id, 'ps_gestational_age_in_weeks'] <= 26:
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=3,
                                                     recommended_gestation_next_anc=26)

            elif 26 > df.at[person_id, 'ps_gestational_age_in_weeks'] <= 30:
                self.module.schedule_next_anc(person_id, visit_to_be_scheduled=3,
                                              recommended_gestation_next_anc=30)

            elif 30 > df.at[person_id, 'ps_gestational_age_in_weeks'] <= 34:
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=3,
                                                     recommended_gestation_next_anc=34)

            elif 34 > df.at[person_id, 'ps_gestational_age_in_weeks'] <= 36:
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=3,
                                                     recommended_gestation_next_anc=36)

            elif 36 > df.at[person_id, 'ps_gestational_age_in_weeks'] <= 38:
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=3,
                                                     recommended_gestation_next_anc=38)

            elif 38 > df.at[person_id, 'ps_gestational_age_in_weeks'] <= 40:
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=3,
                                                     recommended_gestation_next_anc=40)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] > 40:
                pass

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        return actual_appt_footprint

    def did_not_run(self):
        pass

    def not_available(self):
        pass


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

        if df.at[person_id, 'is_alive'] and df.at[person_id, 'is_pregnant'] and \
            ~df.at[person_id, 'la_currently_in_labour']:
            df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1

            # ========================================== Schedule next visit =======================================

            if df.at[person_id, 'ps_gestational_age_in_weeks'] <= 30:
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=4,
                                                     recommended_gestation_next_anc=30)

            elif 30 > df.at[person_id, 'ps_gestational_age_in_weeks'] <= 34:
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=4,
                                                     recommended_gestation_next_anc=34)

            elif 34 > df.at[person_id, 'ps_gestational_age_in_weeks'] <= 36:
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=4,
                                                     recommended_gestation_next_anc=36)

            elif 36 > df.at[person_id, 'ps_gestational_age_in_weeks'] <= 38:
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=4,
                                                     recommended_gestation_next_anc=38)

            elif 38 > df.at[person_id, 'ps_gestational_age_in_weeks'] <= 40:
                self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=4,
                                                     recommended_gestation_next_anc=40)

            elif df.at[person_id, 'ps_gestational_age_in_weeks'] > 40:
                pass

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        return actual_appt_footprint

    def did_not_run(self):
        pass

    def not_available(self):
        pass


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

        self.module.ANCTracker['anc4+'] =+ 1

    #    if df.at[person_id, 'is_alive']:
    #        df.at[person_id, 'ac_total_anc_visits_current_pregnancy'] += 1
    #        if df.at[person_id, 'ps_gestational_age_in_weeks'] <= 34:
    #            self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=5,
    #                                                 recommended_gestation_next_anc=34)

    #        elif 34 > df.at[person_id, 'ps_gestational_age_in_weeks'] <= 36:
    #            self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=5,
    #                                                 recommended_gestation_next_anc=36)

    #        elif 36 > df.at[person_id, 'ps_gestational_age_in_weeks'] <= 38:
    #            self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=5,
    #                                                 recommended_gestation_next_anc=38)

    #        elif 38 > df.at[person_id, 'ps_gestational_age_in_weeks'] <= 40:
    #            self.module.antenatal_care_scheduler(person_id, visit_to_be_scheduled=5,
    #                                                 recommended_gestation_next_anc=40)

    #        elif df.at[person_id, 'ps_gestational_age_in_weeks'] > 40:
    #            pass
        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        return actual_appt_footprint

    def did_not_run(self):
        pass

    def not_available(self):
        pass

class HSI_CareOfWomenDuringPregnancy_AdditionalAntenatalCareContact(HSI_Event, IndividualScopeEventMixin):
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

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        return actual_appt_footprint

    def did_not_run(self):
        pass

    def not_available(self):
        pass

# TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! REVIEW BELOW EVENTS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

class HSI_CareOfWomenDuringPregnancy_EmergencyTreatment(HSI_Event, IndividualScopeEventMixin):
    """ This is the HSI EmergencyTreatment. Currently it is not scheduled to run, but will be scheduled via the
    PregnancySupervisor Module for women experiencing an emergency related to pregnancy or a disease of pregnancy.
    This will likely scheduled additional HSI's for treatment but structure hasn't been confirmed. Currently it is
    hollow, and will be completed by June 2020 TLO meeting."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_EmergencyTreatment'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        logger.info('This is HSI_CareOfWomenDuringPregnancy_EmergencyTreatment, person %d has been sent for treatment '
                    'of an antenatal emergency on date %s ', person_id, self.sim.date)

        # Next we define the consumables required for the HSI to run
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Basic ANC', 'Intervention_Pkg_Code']
        )[0]
        consumables_needed = {
            'Intervention_Package_Code': {pkg_code: 1},
            'Item_Code': {},
        }

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed
        )

        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code]:
            logger.debug('PkgCode is available, so use it.')
        else:
            logger.debug('PkgCode is not available, so can' 't use it.')

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_CareOfWomenDuringPregnancy_EmergencyTreatment: did not run')

    def not_available(self):
        pass


class HSI_CareOfWomenDuringPregnancy_PresentsForPostAbortionCare(HSI_Event, IndividualScopeEventMixin):
    """ This is HSI PostAbortionCare. Currently it is not scheduled, but will be scheduled via the PregnancySupervisor
    module for women who seek care following a termination of pregnancy. It will manage treatment for common
    complications of abortion, including manual removal of retained products. Currently it is
    hollow, and will be completed by June 2020 TLO meeting."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_PresentsForPostAbortionCare'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1  # TODO: determine most accurate appt time for this HSI
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 1  # 2/3?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.info('This is HSI_CareOfWomenDuringPregnancy_PresentsForPostAbortionCare, person %d has been sent  for '
                    'treatment following an abortion on date %s ', person_id, self.sim.date)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Post-abortion case management', 'Intervention_Pkg_Code']
        )[0]
        consumables_needed = {
            'Intervention_Package_Code': {pkg_code: 1},
            'Item_Code': {},
        }

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed
        )

        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code]:
            logger.debug('PkgCode is available, so use it.')
        else:
            logger.debug('PkgCode is not available, so can' 't use it.')

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_CareOfWomenDuringPregnancy_PresentsForPostAbortionCare: did not run')

    def not_available(self):
        pass


class HSI_CareOfWomenDuringPregnancy_TreatmentFollowingAntepartumStillbirth(HSI_Event, IndividualScopeEventMixin):
    """ This is HSI TreatmentFollowingAntepartumStillbirth. Currently it is not scheduled but will be scheduled by the
    PregnancySupervisor Event for women who seek care following an antepartum stillbirth. It will manage interventions
    associated with treatment of stillbirth such as induction of labour or caesarean section. Currently it is
    hollow, and will be completed by June 2020 TLO meeting."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, CareOfWomenDuringPregnancy)

        self.TREATMENT_ID = 'CareOfWomenDuringPregnancy_TreatmentFollowingAntepartumStillbirth'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1  # TODO: determine most accurate appt time for this HSI
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.info('This is HSI_CareOfWomenDuringPregnancy_TreatmentFollowingAntepartumStillbirth, person %d has been'
                    ' referred for care following an antenatal stillbirth on date %s ', person_id, self.sim.date)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Post-abortion case management', 'Intervention_Pkg_Code']
        )[0]
        consumables_needed = {
            'Intervention_Package_Code': {pkg_code: 1},
            'Item_Code': {},
        }

        # TODO: Dummy consumables above- review

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed
        )

        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code]:
            logger.debug('PkgCode is available, so use it.')
        else:
            logger.debug('PkgCode is not available, so can' 't use it.')

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_CareOfWomenDuringPregnancy_TreatmentFollowingAntepartumStillbirth: did not run')

    def not_available(self):
        pass


class AntenatalCareLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(years=self.repeat))

    def apply(self, population):
        df = self.sim.population.props

        total_anc1_visits = self.module.ANCTracker['total_first_anc_visits']
        anc1_in_first_trimester = self.module.ANCTracker['total_anc1_first_trimester']
        cumm_gestation = self.module.ANCTracker['cumm_ga_at_anc1']
        total_anc_4 = self.module.ANCTracker['anc4+']

        ra_lower_limit = 14
        ra_upper_limit = 50
        women_reproductive_age = df.index[(df.is_alive & (df.sex == 'F') & (df.age_years > ra_lower_limit) &
                                           (df.age_years < ra_upper_limit))]
        total_women_reproductive_age = len(women_reproductive_age)

        dict_for_output = {'mean_ga_first_anc': cumm_gestation/total_anc1_visits,
                           'proportion_anc1_firs_trimester': (anc1_in_first_trimester/ total_anc1_visits) * 100,
                           'anc4+_repro_age': (total_anc_4/total_women_reproductive_age) * 100,
                           'crude_anc4+': total_anc_4,
                           'crude_women_ra': total_women_reproductive_age}

        # TODO: check logic for ANC4+ calculation

        logger.info('%s|anc_summary_stats|%s', self.sim.date, dict_for_output)

        self.module.ANCTracker = {'total_first_anc_visits': 0, 'cumm_ga_at_anc1': 0, 'total_anc1_first_trimester': 0,
                                  'anc4+': 0, 'anc8+': 0}

