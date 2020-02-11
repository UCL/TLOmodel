import pandas as pd
from pathlib import Path

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods.healthsystem import HSI_Event
from tlo.lm import LinearModel, LinearModelType, Predictor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AntenatalCare(Module):
    """This is the Antenatal Care module. It is responsible for calculating probability of antenatal care seeking and
    houses all Health System Interaction events pertaining to monitoring and treatment of women during the antenatal
    period of their pregnancy. The majority of this module remains hollow prior to completion for June 2020 deadline"""

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

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
        'ac_total_anc_visits': Property(Types.INT, 'rolling total of antenatal visits this woman has attended during '
                                                   'her pregnancy'),
    }

    def read_parameters(self, data_folder):

        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_AntenatalCare.xlsx',
                            sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)

    # =========================================== LINEAR MODEL EQUATIONS ==============================================

        params = self.parameters

        # n.b. this equation is copied from labour.py as a place holder prior to full lit review
        params['anc_equations'] = {
            'care_seeking': LinearModel(
                LinearModelType.LOGISTIC,
                params['odds_first_anc'],
                Predictor('li_mar_stat').when('1', params['or_anc_unmarried']).when('3',
                                                                                    params['or_anc_unmarried']),
                Predictor('li_wealth').when('4', params['or_anc_wealth_4']).when('5',
                                                                                 params['or_anc_wealth_5']),
                Predictor('li_urban').when(True, params['or_anc_urban']))
        }

    def initialise_population(self, population):

        df = population.props
        df.loc[df.sex == 'F', 'ac_total_anc_visits'] = 0

        # Todo: We may (will) need to apply a number of previous ANC visits to women pregnant at baseline?
        # Todo: Similarly need to the schedule additional ANC visits/ care seeking

    def initialise_simulation(self, sim):

        event = AntenatalCareSeeking(self)
        sim.schedule_event(event, sim.date + DateOffset(weeks=8))

    def on_birth(self, mother_id, child_id):

        df = self.sim.population.props

        if df.at[child_id, 'sex'] == 'F':
            df.at[child_id, 'ac_total_anc_visits'] = 0

    def on_hsi_alert(self, person_id, treatment_id):

        logger.debug('This is AntenatalCare, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)


class AntenatalCareSeeking(RegularEvent, PopulationScopeEventMixin):
    """ This is the AntenatalCareSeeking Event. Currently it houses a dummy care seeking equation to determine
    if a pregnant woman will seek care and during which week of gestation she will do so in. This will eventually be
    housed in the PregnancySupervisor module"""

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(weeks=8))

    def apply(self, population):

        df = population.props
        params = self.module.parameters

        # We run this event every 8 weeks, looking at all women who have become pregnant in the previous 8 weeks
        due_anc = df.loc[df.is_pregnant & df.is_alive & (df.ps_gestational_age_in_weeks <= 8)]

        # Using the linear model we determine if these women will ever seek antenatal care
        result = params['anc_equations']['care_seeking'].predict(due_anc)
        random_draw = pd.Series(self.module.rng.random_sample(size=len(due_anc)), index=due_anc.index)
        temp_df = pd.concat([result, random_draw], axis=1)
        temp_df.columns = ['result', 'random_draw']
        positive_index = temp_df.index[temp_df.random_draw < temp_df.result]

        # For those we do, we use a dummy draw to determine at what gestation they will seek care
        for person in positive_index:
            anc_date = df.at[person, 'date_of_last_pregnancy'] + pd.to_timedelta(self.module.rng.choice(range(11, 30),
                                                                                 size=()),
                                                                                 unit='W')
            event = HSI_AntenatalCare_PresentsForFirstAntenatalCareVisit(self.module, person_id=person)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=1,  # ????
                                                                topen=anc_date,
                                                                tclose=anc_date + DateOffset(days=7))
            logger.debug('This is AntenatalCareSeekingEvent, person %d has chosen to seek their first ANC appointment '
                         'on date%s', person, anc_date)

        # Todo, is it here that we should be scheduling future ANC visits or should that be at the first HSI?


class HSI_AntenatalCare_PresentsForFirstAntenatalCareVisit(HSI_Event, IndividualScopeEventMixin):
    """ This is the HSI PresentsForFirstAntenatalCareVisit. Currently it is scheduled by the AntenatalCareSeekingEvent.
    It will be responsible for the management of monitoring and treatment interventions delivered in a woman's first
    antenatal care visit. It will also go on the schedule the womans next ANC appointment. Currently it is hollow, and
    will be completed by June 2020 TLO meeting."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, AntenatalCare)

        self.TREATMENT_ID = 'AntenatalCare_PresentsForFirstAntenatalCareVisit'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['AntenatalFirst'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props

        logger.info('This is HSI_AntenatalCare_PresentsForFirstAntenatalCareVisit, person %d has presented for the '
                    'first antenatal care visit of their pregnancy on date %s at gestation %d', person_id,
                    self.sim.date, df.at[person_id, 'ps_gestational_age_in_weeks'])

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        pkg_code = pd.unique(consumables.loc[consumables[
                                                 'Intervention_Pkg'] == 'Basic ANC',
                                             'Intervention_Pkg_Code'])[0]
        pkg_code_syphilis = pd.unique(consumables.loc[consumables[
                                                 'Intervention_Pkg'] == 'Syphilis detection and treatment '
                                                                        '(pregnant women)',
                                             'Intervention_Pkg_Code'])[0]
        pkg_code_tetanus = pd.unique(consumables.loc[consumables[
                                                 'Intervention_Pkg'] == 'Tetanus toxoid (pregnant women)',
                                             'Intervention_Pkg_Code'])[0]
        pkg_code_ipt = pd.unique(consumables.loc[consumables[
                                                 'Intervention_Pkg'] == 'IPT (pregnant women)',
                                                 'Intervention_Pkg_Code'])[0]

        consumables_needed = {
            'Intervention_Package_Code': [{pkg_code: 1}, {pkg_code_syphilis: 1}, {pkg_code_tetanus: 1},
                                          {pkg_code_ipt: 1}],
            'Item_Code': [],
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
        logger.debug('HSI_AntenatalCare_PresentsForFirstAntenatalCareVisit: did not run')
        pass


class HSI_AntenatalCare_PresentsForSubsequentAntenatalCareVisit(HSI_Event, IndividualScopeEventMixin):
    """This is the HSI PThis is the HSI PresentsForSubsequentAntenatalCareVisit. Currently it is not scheduled to run, but
     will be scheduled by HSI PresentsForFirstANCVists. It will be responsible for the management of monitoring and
     treatment interventions delivered in a woman's subsequent antenatal care visit. It will also go on the schedule the
      womans next ANC appointment. Currently it is hollow, and will be completed by June 2020 TLO meeting. """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, AntenatalCare)

        self.TREATMENT_ID = 'AntenatalCare_PresentsForSubsequentAntenatalCareVisit'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 0
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        logger.info('This is HSI_AntenatalCare_PresentsForSubsequentAntenatalCareVisit, person %d has presented for '
                    'a subsequent antenatal care visit of their pregnancy on date %s', person_id, self.sim.date)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code = pd.unique(consumables.loc[consumables[
                                                 'Intervention_Pkg'] == 'Basic ANC',
                                             'Intervention_Pkg_Code'])[0]
        consumables_needed = {
            'Intervention_Package_Code': [{pkg_code: 1}],
            'Item_Code': [],
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
        logger.debug('HSI_AntenatalCare_PresentsForSubsequentAntenatalCareVisit: did not run')
        pass


class HSI_AntenatalCare_EmergencyTreatment(HSI_Event, IndividualScopeEventMixin):
    """ This is the HSI EmergencyTreatment. Currently it is not scheduled to run, but will be scheduled via the
    PregnancySupervisor Module for women experiencing an emergency related to pregnancy or a disease of pregnancy.
    This will likely scheduled additional HSI's for treatment but structure hasn't been confirmed. Currently it is
    hollow, and will be completed by June 2020 TLO meeting."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, AntenatalCare)

        self.TREATMENT_ID = 'AntenatalCare_EmergencyTreatment'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        logger.info('This is HSI_AntenatalCare_EmergencyTreatment, person %d has been sent  for treatment '
                    'of an antenatal emergency on date %s ', person_id, self.sim.date)

        # Next we define the consumables required for the HSI to run
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code = pd.unique(consumables.loc[consumables[
                                                 'Intervention_Pkg'] == 'Basic ANC',
                                             'Intervention_Pkg_Code'])[0]
        consumables_needed = {
            'Intervention_Package_Code': [{pkg_code: 1}],
            'Item_Code': [],
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
        logger.debug('HSI_AntenatalCare_EmergencyTreatment: did not run')
        pass


class HSI_AntenatalCare_PostAbortionCare(HSI_Event, IndividualScopeEventMixin):  # ??Name
    """ This is HSI PostAbortionCare. Currently it is not scheduled, but will be scheduled via the PregnancySupervisor
    module for women who seek care following a termination of pregnancy. It will manage treatment for common
    complications of abortion, including manual removal of retained products. Currently it is
    hollow, and will be completed by June 2020 TLO meeting."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, AntenatalCare)

        self.TREATMENT_ID = 'AntenatalCare_PostAbortionCare'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1  # TODO: determine most accurate appt time for this HSI
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 1  # 2/3?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.info('This is HSI_AntenatalCare_PostAbortionCare, person %d has been sent  for treatment '
                    'following an abortion on date %s ', person_id, self.sim.date)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code = pd.unique(consumables.loc[consumables[
                                                 'Intervention_Pkg'] == 'Post-abortion case management',
                                             'Intervention_Pkg_Code'])[0]
        consumables_needed = {
            'Intervention_Package_Code': [{pkg_code: 1}],
            'Item_Code': [],
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
        logger.debug('HSI_AntenatalCare_PostAbortionCare: did not run')
        pass


class HSI_AntenatalCare_TreatmentFollowingAntepartumStillbirth(HSI_Event, IndividualScopeEventMixin):  # ??Name
    """ This is HSI TreatmentFollowingAntepartumStillbirth. Currently it is not scheduled but will be scheduled by the
    PregnancySupervisor Event for women who seek care following an antepartum stillbirth. It will manage interventions
    associated with treatment of stillbirth such as induction of labour or caesarean section. Currently it is
    hollow, and will be completed by June 2020 TLO meeting."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, AntenatalCare)

        self.TREATMENT_ID = 'AntenatalCare_TreatmentFollowingAntepartumStillbirth'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1  # TODO: determine most accurate appt time for this HSI
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        logger.info('This is HSI_AntenatalCare_TreatmentFollowingAntepartumStillbirth, person %d has been referred for '
                    'care following an antenatal stillbirth on date %s ', person_id, self.sim.date)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code = pd.unique(consumables.loc[consumables[
                                                 'Intervention_Pkg'] == 'Post-abortion case management',
                                             'Intervention_Pkg_Code'])[0]
        consumables_needed = {
            'Intervention_Package_Code': [{pkg_code: 1}],
            'Item_Code': [],
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
        logger.debug('HSI_AntenatalCare_TreatmentFollowingAntepartumStillbirth: did not run')
        pass
