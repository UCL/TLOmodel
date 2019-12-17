import logging

import pandas as pd
from pathlib import Path

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods.healthsystem import HSI_Event

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AntenatalCare(Module):
    """
    This is the Antenatal Care module. It is responsible for calculating probability of antenatal care seeking and
    houses all Health System Interaction events pertaining to monitoring and treatment of women during the antenatal
    period of their pregnancy
     """

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    PARAMETERS = {
        'prob_seek_care_first_anc': Parameter(
            Types.REAL, 'Probability a woman will access antenatal care for the first time'),  # DUMMY PARAMETER
    }

    PROPERTIES = {
        'ac_total_anc_visits': Property(Types.INT, 'rolling total of antenatal visits this woman has attended during '
                                                   'her pregnancy'),
    }

    def read_parameters(self, data_folder):
        """
        """
        params = self.parameters

        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_AntenatalCare.xlsx',
                            sheet_name='parameter_values')
        dfd.set_index('parameter_name', inplace=True)
        params['prob_seek_care_first_anc'] = dfd.loc['prob_seek_care_first_anc', 'value']

        #  We don't
    def initialise_population(self, population):

        df = population.props

        df.loc[df.sex == 'F', 'ac_total_anc_visits'] = 0

        # Todo: We may (will) need to apply a number of previous ANC visits to women pregnant at baseline?
        # Todo: Similarly need to the schedule additional ANC visits/ care seeking

    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        """

        event = AntenatalCareSeeking(self)
        sim.schedule_event(event, sim.date + DateOffset(weeks=8))

        # Todo: discuss this logic with TC regarding current care seeking approach

        event = AntenatalCareLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=0))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        df = self.sim.population.props

        if df.at[child_id, 'sex'] == 'F':
            df.at[child_id, 'ac_total_anc_visits'] = 0

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug('This is AntenatalCare, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)


class AntenatalCareSeeking(RegularEvent, PopulationScopeEventMixin):
    """ The event manages care seeking for newly pregnant women
    """

    def __init__(self, module):
        """One line summary here

        We need to pass the frequency at which we want to occur to the base class
        constructor using super(). We also pass the module that created this event,
        so that random number generators can be scoped per-module.

        :param module: the module that created this event
        """
        super().__init__(module, frequency=DateOffset(weeks=8))  # could it be 8 weeks?

    def apply(self, population):
        """Apply this event to the population.

        :param population: the current population
        """
        df = population.props
        m = self
        params = self.module.parameters

        # Todo: Reformat so that we use a weibull distribution to for care seeking
        # Todo: to discuss with Tim C the best way to mirror whats happening in malawi (only 50% women have ANC1 in
        #  first trimester)

        pregnant_past_month = df.index[df.is_pregnant & df.is_alive & (df.ps_gestational_age <= 8) &
                                       (df.date_of_last_pregnancy > self.sim.start_date)]

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1 DUMMY CARE SEEKING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        random_draw = pd.Series(self.module.rng.random_sample(size=len(pregnant_past_month)),
                                index=df.index[df.is_pregnant & df.is_alive & (df.ps_gestational_age <= 8) &
                                               (df.date_of_last_pregnancy > self.sim.start_date)])

        prob_care_seeking = float(params['prob_seek_care_first_anc'])
        eff_prob_anc = pd.Series(prob_care_seeking,
                                 index=df.index[df.is_pregnant & df.is_alive & (df.ps_gestational_age <= 8) &
                                                (df.date_of_last_pregnancy > self.sim.start_date)])
        dfx = pd.concat([eff_prob_anc, random_draw], axis=1)
        dfx.columns = ['eff_prob_anc', 'random_draw']
        idx_anc = dfx.index[dfx.eff_prob_anc > dfx.random_draw] # right?

        gestation_at_anc = pd.Series(self.module.rng.choice(range(11, 39), size=len(idx_anc)), index=df.index[idx_anc])
        # THIS IS ALL WRONG DATE WISE
        conception = pd.Series(df.date_of_last_pregnancy, index=df.index[idx_anc])
        dfx = pd.concat([conception, gestation_at_anc], axis=1)
        dfx.columns = ['conception', 'gestation_at_anc']
        dfx['first_anc'] = dfx['conception'] + pd.to_timedelta(dfx['gestation_at_anc'], unit='w')

        for person in idx_anc:
            care_seeking_date = dfx.at[person, 'first_anc']
            print(person, self.sim.date, care_seeking_date)
            event = HSI_AntenatalCare_PresentsForFirstAntenatalCareVisit(self.module, person_id=person)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=1,  # ????
                                                                topen=care_seeking_date,
                                                                tclose=None)
        # Todo, is it here that we should be scheduling future ANC visits or should that be at the first HSI?


class HSI_AntenatalCare_PresentsForFirstAntenatalCareVisit(Event, HSI_Event):
    """
    This is a Health System Interaction Event.This is event manages a woman's firs antenatal care visit
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, AntenatalCare)

        # First we define the treatment ID for this HSI
        self.TREATMENT_ID = 'AntenatalCare_PresentsForFirstAntenatalCareVisit'

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['AntenatalFirst'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        # Define the necessary information for an HSI
        self.ACCEPTED_FACILITY_LEVEL = [0]  # Community?!
        # TODO: ANC should be offered at level 0-2. Can all interventions be given at level 0?
        self.ALERT_OTHER_DISEASES = ['*']

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        params = self.module.parameters

        gestation_at_visit = df.at[person_id, 'ps_gestational_age']
        logger.info('This is HSI_AntenatalCare_PresentsForFirstAntenatalCareVisit, person %d has presented for the '
                    'first antenatal care visit of their pregnancy on date %s at gestation %d', person_id,
                    self.sim.date, gestation_at_visit)

        # Next we define the consumables required for the HSI to run
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code = pd.unique(consumables.loc[consumables[
                                                 'Intervention_Pkg'] == 'Basic ANC',
                                             'Intervention_Pkg_Code'])[0]
        pkg_code_syphilis = pd.unique(consumables.loc[consumables[
                                                 'Intervention_Pkg'] == 'Syphilis detection and treatment '
                                                                        '(pregnant women',
                                             'Intervention_Pkg_Code'])[0]
        # TODO: should this just be an item code to determine syphilis then refer on for treatment?

        pkg_code_tetanus = pd.unique(consumables.loc[consumables[
                                                 'Intervention_Pkg'] == 'Tetanus toxoid (pregnant women)',
                                             'Intervention_Pkg_Code'])[0]
        pkg_code_ipt = pd.unique(consumables.loc[consumables[
                                                 'Intervention_Pkg'] == 'IPT (pregnant women)',
                                             'Intervention_Pkg_Code'])[0]

        # TODO: Additional deworming package may be appropriate here.
        # TODO: a/w ANC guidelines to confirm interventions for 1st ANC only - RPD at MoH contacted

        consumables_needed = {
            'Intervention_Package_Code': [{pkg_code: 1}, {pkg_code_syphilis: 1}, {pkg_code_tetanus: 1},
                                          {pkg_code_ipt: 1}],
            'Item_Code': [],
        }

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed
        )

        # answer comes back in the same format, but with quantities replaced with bools indicating availability
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code]:
            logger.debug('PkgCode is available, so use it.')
        else:
            logger.debug('PkgCode is not available, so can' 't use it.')

        # Return the actual appt footprints
        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        # actual_appt_footprint['AntenatalFirst'] = actual_appt_footprint['AntenatalFirst'] * 2

        return actual_appt_footprint

        # TODO: impact of squeeze factor & cons request output
        # TODO: Intervention code & scheduling of additional ANC visits?

    def did_not_run(self):
        logger.debug('HSI_AntenatalCare_PresentsForFirstAntenatalCareVisit: did not run')
        pass


class HSI_AntenatalCare_PresentsForSubsequentAntenatalCareVisit(Event, HSI_Event):
    """
    This is a Health System Interaction Event.
    This is event manages all subsequent antenatal care visits additional to a woman's first visit
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, AntenatalCare)

        # First we define the treatment ID for this HSI
        self.TREATMENT_ID = 'AntenatalCare_PresentsForSubsequentAntenatalCareVisit'

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        # Define the necessary information for an HSI
        self.ACCEPTED_FACILITY_LEVEL = [0]  # Community?!
        self.ALERT_OTHER_DISEASES = ['*']

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        # TODO: if this format is kept we need to calculate what number ANC visit this is here
        logger.info('This is HSI_AntenatalCare_PresentsForSubsequentAntenatalCareVisit, person %d has presented for '
                    'a subsequent antenatal care visit of their pregnancy on date %s', person_id, self.sim.date)

        # Next we define the consumables required for the HSI to run
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code = pd.unique(consumables.loc[consumables[
                                                 'Intervention_Pkg'] == 'Basic ANC',
                                             'Intervention_Pkg_Code'])[0]
        consumables_needed = {
            'Intervention_Package_Code': [{pkg_code: 1}],
            'Item_Code': [],
        }

        # TODO:Additional consumables to consider depending on GA at visit and a/w guidelines

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed
        )

        # answer comes back in the same format, but with quantities replaced with bools indicating availability
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code]:
            logger.debug('PkgCode is available, so use it.')
        else:
            logger.debug('PkgCode is not available, so can' 't use it.')

            # Return the actual appt footprints
        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        # actual_appt_footprint['AntenatalFirst'] = actual_appt_footprint['AntenatalFirst'] * 2

        return actual_appt_footprint

        # TODO: impact of squeeze factor
        # TODO: Intervention code

    def did_not_run(self):
        logger.debug('HSI_AntenatalCare_PresentsForSubsequentAntenatalCareVisit: did not run')
        pass


class HSI_AntenatalCare_PresentsWithNewOnsetSymptoms(Event, HSI_Event):
    """
    This is a Health System Interaction Event.
    This is event manages diagnosis and referral for women presenting to a health facility with new onset symptoms
    during pregnancy
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, AntenatalCare)

        # First we define the treatment ID for this HSI
        self.TREATMENT_ID = 'AntenatalCare_PresentsWithNewSymptoms'

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1  # TODO: determine most accurate appt time for this HSI
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        # Define the necessary information for an HSI
        self.ACCEPTED_FACILITY_LEVEL = [1]  # 2/3?
        self.ALERT_OTHER_DISEASES = ['*']

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        logger.info('This is HSI_AntenatalCare_PresentsWithNewOnsetSymptoms, person %d has presented for treatment '
                    'of new onset symptoms during their pregnancy on date %s ', person_id, self.sim.date)

        # Next we define the consumables required for the HSI to run
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code = pd.unique(consumables.loc[consumables[
                                                 'Intervention_Pkg'] == 'Basic ANC',
                                             'Intervention_Pkg_Code'])[0]
        consumables_needed = {
            'Intervention_Package_Code': [{pkg_code: 1}],
            'Item_Code': [],
        }

        # TODO: Once diagnostic algorithm is finalised I can accurately request consumables above

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed
        )

        # answer comes back in the same format, but with quantities replaced with bools indicating availability
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code]:
            logger.debug('PkgCode is available, so use it.')
        else:
            logger.debug('PkgCode is not available, so can' 't use it.')

            # Return the actual appt footprints
        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        # actual_appt_footprint['AntenatalFirst'] = actual_appt_footprint['AntenatalFirst'] * 2

        return actual_appt_footprint

        # TODO: impact of squeeze factor
        # TODO: Intervention code

    def did_not_run(self):
        logger.debug('HSI_AntenatalCare_PresentsWithNewOnsetSymptoms: did not run')
        pass


class HSI_AntenatalCare_EmergencyTreatment(Event, HSI_Event):
    # TODO: This will likely evolve into smaller HSI's per emergency
    """
    This is a Health System Interaction Event.
    This is event manages care for a woman who presents to a facility during pregnancy related emergency
    emergency.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, AntenatalCare)

        # First we define the treatment ID for this HSI
        self.TREATMENT_ID = 'AntenatalCare_EmergencyTreatment'

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1  # TODO: determine most accurate appt time for this HSI
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        # Define the necessary information for an HSI
        self.ACCEPTED_FACILITY_LEVEL = [1]  # 2/3?
        self.ALERT_OTHER_DISEASES = ['*']

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        logger.info('This is HSI_AntenatalCare_EmergencyTreatment, person %d has been sent  for treatment '
                    'of an antenatal emergeny on date %s ', person_id, self.sim.date)

        # Next we define the consumables required for the HSI to run
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code = pd.unique(consumables.loc[consumables[
                                                 'Intervention_Pkg'] == 'Basic ANC',
                                             'Intervention_Pkg_Code'])[0]
        consumables_needed = {
            'Intervention_Package_Code': [{pkg_code: 1}],
            'Item_Code': [],
        }

        # TODO: This will be determined by which emergency a woman is referred for?

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed
        )

        # answer comes back in the same format, but with quantities replaced with bools indicating availability
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code]:
            logger.debug('PkgCode is available, so use it.')
        else:
            logger.debug('PkgCode is not available, so can' 't use it.')

            # Return the actual appt footprints
        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        # actual_appt_footprint['AntenatalFirst'] = actual_appt_footprint['AntenatalFirst'] * 2

        return actual_appt_footprint

        # TODO: impact of squeeze factor
        # TODO: Intervention code

    def did_not_run(self):
        logger.debug('HSI_AntenatalCare_EmergencyTreatment: did not run')
        pass


class HSI_AntenatalCare_PostAbortionCare(Event, HSI_Event):  # ??Name
    """
    This is a Health System Interaction Event.
    This is event manages treatment for any woman referred for post-abortion care
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, AntenatalCare)

        # First we define the treatment ID for this HSI
        self.TREATMENT_ID = 'AntenatalCare_PostAbortionCare'

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1  # TODO: determine most accurate appt time for this HSI
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        # Define the necessary information for an HSI
        self.ACCEPTED_FACILITY_LEVEL = [1]  # 2/3?
        self.ALERT_OTHER_DISEASES = ['*']

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        logger.info('This is HSI_AntenatalCare_PostAbortionCare, person %d has been sent  for treatment '
                    'following an abortion on date %s ', person_id, self.sim.date)

        # Next we define the consumables required for the HSI to run
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code = pd.unique(consumables.loc[consumables[
                                                 'Intervention_Pkg'] == 'Post-abortion case management',
                                             'Intervention_Pkg_Code'])[0]
        consumables_needed = {
            'Intervention_Package_Code': [{pkg_code: 1}],
            'Item_Code': [],
        }

        # TODO: There is also a 'post ectopic' intervention package code to consider

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed
        )

        # answer comes back in the same format, but with quantities replaced with bools indicating availability
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code]:
            logger.debug('PkgCode is available, so use it.')
        else:
            logger.debug('PkgCode is not available, so can' 't use it.')

            # Return the actual appt footprints
        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        # actual_appt_footprint['AntenatalFirst'] = actual_appt_footprint['AntenatalFirst'] * 2

        return actual_appt_footprint

        # TODO: impact of squeeze factor
        # TODO: Intervention code

    def did_not_run(self):
        logger.debug('HSI_AntenatalCare_PostAbortionCare: did not run')
        pass


class HSI_AntenatalCare_TreatmentFollowingAntepartumStillbirth(Event, HSI_Event):  # ??Name
    """
    This is a Health System Interaction Event.
    This is event manages treatment for women who have experienced an Antepartum Stillbirth
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, AntenatalCare)

        # First we define the treatment ID for this HSI
        self.TREATMENT_ID = 'AntenatalCare_TreatmentFollowingAntepartumStillbirth'

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['ANCSubsequent'] = 1  # TODO: determine most accurate appt time for this HSI
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint

        # Define the necessary information for an HSI
        self.ACCEPTED_FACILITY_LEVEL = [1]  # 2/3?
        self.ALERT_OTHER_DISEASES = ['*']

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        logger.info('This is HSI_AntenatalCare_TreatmentFollowingAntepartumStillbirth, person %d has been referred for '
                    'care following an antenatal stillbirth on date %s ', person_id, self.sim.date)

        # Next we define the consumables required for the HSI to run
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

        # answer comes back in the same format, but with quantities replaced with bools indicating availability
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code]:
            logger.debug('PkgCode is available, so use it.')
        else:
            logger.debug('PkgCode is not available, so can' 't use it.')

            # Return the actual appt footprints
        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        # actual_appt_footprint['AntenatalFirst'] = actual_appt_footprint['AntenatalFirst'] * 2

        return actual_appt_footprint

        # TODO: impact of squeeze factor
        # TODO: Intervention code

    def did_not_run(self):
        logger.debug('HSI_AntenatalCare_TreatmentFollowingAntepartumStillbirth: did not run')
        pass


class AntenatalCareLoggingEvent(RegularEvent, PopulationScopeEventMixin):
        """Handles Antenatal Care logging"""

        def __init__(self, module):
            """schedule logging to repeat every 3 months
            """
            #    self.repeat = 3
            #    super().__init__(module, frequency=DateOffset(days=self.repeat))
            super().__init__(module, frequency=DateOffset(months=3))

        def apply(self, population):
            """Apply this event to the population.
            :param population: the current population
            """
            df = population.props
