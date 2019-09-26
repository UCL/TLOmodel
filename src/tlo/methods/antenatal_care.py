"""
Module responsible for antenatal care provision and care seeking for pregnant women .
"""

import logging

import numpy as np
import pandas as pd
from pathlib import Path

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent


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
            Types.REAL, 'Probability a woman will access antenatal care for the first time'),  # DUMMY PARAMETER
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

#        if 'HealthBurden' in self.sim.modules.keys():
#            params['daly_wt_haemorrhage_moderate'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=339)

    def initialise_population(self, population):

        df = population.props

        df.loc[df.sex == 'F', 'ac_gestational_age'] = 0
        df.loc[df.sex == 'F', 'ac_total_anc_visits'] = 0

        # Todo: We may (will) need to apply a number of previous ANC visits to women pregnant at baseline?
        # Todo: Similarly need to the schedule additional ANC visits/ care seeking


    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        """
        event = GestationUpdateEvent
        sim.schedule_event(event(self),
                           sim.date + DateOffset(days=0))

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
            df.at[child_id, 'ac_gestational_age'] = 0
            df.at[child_id, 'ac_total_anc_visits'] = 0

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug('This is AntenatalCare, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

#    def report_daly_values(self):

    #    logger.debug('This is mockitis reporting my health values')
    #    df = self.sim.population.props  # shortcut to population properties dataframe
    #    p = self.parameters


class GestationUpdateEvent(RegularEvent, PopulationScopeEventMixin):
    """
    This event updates the ac_gestational_age for the pregnant population based on the current simulation date
    """

    def __init__(self, module,):
        super().__init__(module, frequency=DateOffset(weeks=1))

    def apply(self, population):
        df = population.props

        gestation_in_days = self.sim.date - df.loc[df.is_pregnant, 'date_of_last_pregnancy']
        gestation_in_weeks = gestation_in_days / np.timedelta64(1, 'W')

        df.loc[df.is_pregnant, 'ac_gestational_age'] = gestation_in_weeks.astype(int)


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

        gestation_at_anc = pd.Series(self.module.rng.choice(range(10, 39), size=len(idx_anc)), index=df.index[idx_anc])
        # THIS IS ALL WRONG DATE WISE
        conception = pd.Series(df.date_of_last_pregnancy, index=df.index[idx_anc])
        dfx = pd.concat([conception, gestation_at_anc], axis=1)
        dfx.columns = ['conception', 'gestation_at_anc']
        dfx['first_anc'] = dfx['conception'] + pd.to_timedelta(dfx['gestation_at_anc'], unit='w')

        for person in idx_anc:
            care_seeking_date = dfx.at[person, 'first_anc']
            event = HSI_AntenatalCare_PresentsForFirstAntenatalCareVisit(self.module, person_id=person)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=1,  # ????
                                                                topen=care_seeking_date,
                                                                tclose=None)

        # Todo, is it here that we should be scheduling future ANC visits or should that be at the first HSI?


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
                                                       'Intervention_Pkg'] == 'Basic ANC',
                                             'Intervention_Pkg_Code'])[0]

        # Todo:Additional consumables to consider: Deworming treatment, syphyllis detection, tetanus toxid

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

        gestation_at_visit = df.at[person_id, 'ac_gestational_age']
        logger.info('This is HSI_AntenatalCare_PresentsForFirstAntenatalCareVisit, person %d has presented for the '
                    'first antenatal care visit of their pregnancy on date %s at gestation %d', person_id,
                        self.sim.date, gestation_at_visit)

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
