"""
Module responsible for antenatal care provision and care seeking for pregnant women .
"""

import logging

import numpy as np
import pandas as pd
from pathlib import Path
import random

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent

from tlo.methods import demography, healthsystem, healthburden, labour, newborn_outcomes


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AntenatalCare(Module):
    """
    This module is responsible for antenatal care seeking and antenatal care health system interaction events
     for pregnant women """
    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    PARAMETERS = {
        'prob_seek_care_first_anc': Parameter(
            Types.REAL, 'Probability a woman will seeking formal antenatal care'),  # DUMMY PARAMETER
    }

    PROPERTIES = {
        'ac_gestational_age': Property(Types.INT, 'current gestational age of this womans pregnancy in weeks'),
        'ac_total_anc_visits': Property(Types.INT, 'rolling total of antenatal visits this woman has attended during '
                                                   'her pregnancy'),

    }

    TREATMENT_ID = ''

    def read_parameters(self, data_folder):
        """
        """
        params = self.parameters

        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_AntenatalCare.xlsx',
                            sheet_name='parameter_values')

        dfd.set_index('parameter_name', inplace=True)

        params['prob_seek_care_first_anc'] = dfd.loc['prob_seek_care_first_anc', 'value']

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """
        df = population.props

        df.loc[df.sex == 'F', 'ac_gestational_age'] = 0
        df.loc[df.sex == 'F', 'ac_total_anc_visits'] = 0

        # Baseline previous ANC visit history for women who are pregnant at baseline
        # Schedule ALL (?) remaining ANC visits for baseline women (with a function that will determine attendance?)

        # Will we use the MNI, it will be massive

    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        """
        event = GestationUpdateEvent
        sim.schedule_event(event(self),
                           sim.date + DateOffset(days=0))

        event= AntenatalCareSeekingAndScheduling(self)
        sim.schedule_event(event, sim.date + DateOffset(weeks=8))

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
            df.at[child_id, 'ac_gestational_age'] = 0
            df.at[child_id, 'ac_total_anc_visits'] = 0

    def query_symptoms_now(self):
        """
        If this is a registered disease module, this is called by the HealthCareSeekingPoll in order to determine the
        healthlevel of each person. It can be called at any time and must return a Series with length equal to the
        number of persons alive and index matching sim.population.props. The entries encode the symptoms on the
        following "unified symptom scale":
        0=None; 1=Mild; 2=Moderate; 3=Severe; 4=Extreme_Emergency
        """

        raise NotImplementedError

    def report_qaly_values(self):
        """
        If this is a registered disease module, this is called periodically by the QALY module in order to compute the
        total 'Quality of Life' for all alive persons. Each disease module must return a Series with length equal to the
        number of persons alive and index matching sim.population.props. The entries encode a QALY weight, between zero
        and 1, which summarise the quality of life for that persons for the total of the past 12 months. Note that this
        can be called at any time.

        Disease modules should look-up the weights to use by calling QALY.get_qaly_weight(sequaluecode). The sequalue
        code to use can be found in the ResourceFile_DALYWeights. ie. Find the appropriate sequalue in that file, and
        then hard-code the sequale code in this call.
        e.g. p['qalywt_mild_sneezing'] = self.sim.modules['QALY'].get_qaly_weight(50)

        """

        raise NotImplementedError

    def on_healthsystem_interaction(self, person_id, cue_type=None, disease_specific=None):
        """
        If this is a registered disease module, this is called whenever there is any interaction between an individual
        and the healthsystem. All disease modules are notified of all interactions with the healthsystem but can choose
        if they will respond by looking at the arguments that are passed.

        * cue_type: determines what has caused the interaction and can be "HealthCareSeekingPoll", "OutreachEvent",
            "InitialDiseaseCall" or "FollowUp".
        * disease_specific: determines if this interaction has been triggered by, or is otherwise intended to be,
            specfifc to a particular disease. If will either take the value None or the name of the registered disease
            module.


        """
        pass


class GestationUpdateEvent(RegularEvent, PopulationScopeEventMixin):
    """
    This event updates the ac_gestational_age for the pregnant population based on the current simulation date
    """

    def __init__(self, module,):
        super().__init__(module, frequency=DateOffset(weeks=1))

    def apply(self, population):
        df = population.props

        gestation_in_days = self.sim.date - df.loc[df.is_pregnant, 'date_of_last_pregnancy']
        gestation_in_weeks = gestation_in_days/ np.timedelta64(1, 'W')

        df.loc[df.is_pregnant, 'ac_gestational_age'] = gestation_in_weeks.astype(int)

        # TODO: Confirm is resetting through BirthEvent


class AntenatalCareSeekingAndScheduling(RegularEvent, PopulationScopeEventMixin):
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

        # TODO: need to use weibull distribution to determne time to care seeking (?)
        # TODO: to discuss with TIM best way to mirror whats happening in malawi (only 50% women have ANC1 in first
        #  trimester)

        pregnant_past_month = df.index[df.is_pregnant & df.is_alive & (df.ac_gestational_age <= 8) &
                                       (df.date_of_last_pregnancy > self.sim.start_date)]

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1 DUMMY CARE SEEKING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        random_draw = pd.Series(self.module.rng.random_sample(size=len(pregnant_past_month)),
                                index=df.index[df.is_pregnant & df.is_alive & (df.ac_gestational_age <= 8) &
                                               (df.date_of_last_pregnancy > self.sim.start_date)])

        prob_care_seeking = float(params['prob_seek_care_first_anc'])
        eff_prob_anc = pd.Series(prob_care_seeking,
                                 index=df.index[df.is_pregnant & df.is_alive & (df.ac_gestational_age <= 8) &
                                                (df.date_of_last_pregnancy > self.sim.start_date)])
        dfx = pd.concat([eff_prob_anc, random_draw], axis=1)
        dfx.columns = ['eff_prob_anc', 'random_draw']
        idx_anc = dfx.index[dfx.eff_prob_anc > dfx.random_draw] # right?

        gestation_at_anc = pd.Series(self.module.rng.choice(range(9, 39), size=len(idx_anc)), index=df.index[idx_anc])
        # THIS IS ALL WRONG DATE WISE
        conception = pd.Series(df.date_of_last_pregnancy, index=df.index[idx_anc])
        dfx = pd.concat([conception, gestation_at_anc], axis=1)
        dfx.columns = ['conception', 'gestation_at_anc']
        dfx['first_anc'] = dfx['conception'] + pd.to_timedelta(dfx['gestation_at_anc'], unit='w')


        x='y'
        for person in idx_anc:
            care_seeking_date = dfx.at[person, 'first_anc']
            event = HSI_AntenatalCare_PresentsForFirstAntenatalCareVisit(self.module, person_id=person)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                            priority=0, #????
                                                            topen=care_seeking_date,
                                                            tclose=care_seeking_date + DateOffset(days=14)
                                                            )

        # Should we schedule all events here or at ANC 1?


class HSI_AntenatalCare_PresentsForFirstAntenatalCareVisit(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    This is event manages a woman's firs antenatal care visit
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['AntenatalFirst'] = 1

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        pkg_code = pd.unique(consumables.loc[consumables[
                                                       'Intervention_Pkg'] ==
                                                   'Basic ANC',
                                                   'Intervention_Pkg_Code'])[0]

        # Additional consumables: Deworming treatment, syphyllis detection, tetanus toxid

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'AntenatalCare_PresentsForFirstAntenatalCareVisit'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = [1, 2, 3]  # Community?!
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        gestation_at_vist = df.at[person_id, 'ac_gestational_age']
        logger.info('This is HSI_AntenatalCare_PresentsForFirstAntenatalCareVisit, person %d has presented for the '
                    'first antenatal care visit of their pregnancy on date %s at gestation %d', person_id,
                        self.sim.date, gestation_at_vist)

        # consider facility level at which interventions can be delivered- most basic ANC may be able to be delivered
        # at community level but some additional interventions need to be delivered at higher level facilites


class HSI_AntenatalCare_PresentsForSubsequentAntenatalCareVisit(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    This is event manages all subsequent antenatal care visits additional to her first visit
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
    #   the_appt_footprint['Over5OPD'] = 1
        the_appt_footprint['ANCSubsequent'] = 1

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        dummy_pkg_code = pd.unique(consumables.loc[consumables[
                                                       'Intervention_Pkg'] ==
                                                   'HIV Testing Services',
                                                   'Intervention_Pkg_Code'])[0]

        pkg_code = pd.unique(consumables.loc[consumables[
                                                       'Intervention_Pkg'] ==
                                                   'Basic ANC',
                                                   'Intervention_Pkg_Code'])[0]

        # Additional consumables: Deworming treatment, syphyllis detection, tetanus toxoid,

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'AntenatalCare_PresentsForAdditionalAntenatalCareVisit'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = [1, 2, 3]  # Community?!
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        if ~df.at[person_id,'is_pregnant']:
            pass

        logger.info('This is HSI_AntenatalCare_PresentsForFirstAntenatalCareVisit, person %d has presented for the '
                    'first antenatal care visit of their pregnancy on date %s', person_id, self.sim.date)

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
