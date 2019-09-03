import logging
import os

import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import demography

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Malaria(Module):

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    PARAMETERS = {
        'mal_inc': Parameter(Types.REAL, 'monthly incidence of malaria in all ages'),
        'interv': Parameter(Types.REAL, 'data frame of intervention coverage by year'),

        'p_infection': Parameter(
            Types.REAL, 'Probability that an uninfected individual becomes infected'),
        'stage': Parameter(
            Types.CATEGORICAL, 'malaria stage'),
        'sensitivity_rdt': Parameter(
            Types.REAL, 'Sensitivity of rdt'),
        'cfr': Parameter(
            Types.REAL, 'case-fatality rate for severe malaria'),
        'dur_asym': Parameter(
            Types.REAL, 'duration (days) of asymptomatic malaria'),
        'dur_clin': Parameter(
            Types.REAL, 'duration (days) of clinical symptoms of malaria'),
        'dur_clin_para': Parameter(
            Types.REAL, 'duration (days) of parasitaemia for clinical malaria cases'),
        'rr_hiv': Parameter(
            Types.REAL, 'relative risk of clinical malaria if hiv-positive'),
    }

    PROPERTIES = {
        'ma_is_infected': Property(
            Types.BOOL, 'Current status of mockitis'),
        'ma_status': Property(
            Types.CATEGORICAL,
            'current malaria stage: Uninf=uninfected; Asym=asymptomatic; Clin=clinical; Past=past',
            categories=['Uninf', 'Asym', 'Clin', 'Past']),
        'ma_date_infected': Property(
            Types.DATE, 'Date of latest infection'),
        'ma_date_death': Property(
            Types.DATE, 'Date of scheduled death due to malaria'),
        'ma_itn_use': Property(
            Types.BOOL, 'Person sleeps under a bednet'),
        'ma_irs': Property(
            Types.BOOL, 'Person sleeps in house which has had indoor residual spraying'),
        'ma_tx': Property(Types.BOOL, 'Currently on anti-malarial treatment'),
        'ma_date_tx': Property(
            Types.DATE, 'Date treatment started for most recent malaria episode'),
        'ma_specific_symptoms': Property(
            Types.CATEGORICAL,
            'specific symptoms with malaria infection', categories=['none', 'clinical', 'severe']),
        'ma_unified_symptom_code': Property(Types.CATEGORICAL, 'level of symptoms on the standardised scale, 0-4',
                                            categories=[0, 1, 2, 3, 4]),
    }

    def read_parameters(self, data_folder):
        p = self.parameters

        workbook = pd.read_excel(os.path.join(self.resourcefilepath,
                                              'ResourceFile_malaria.xlsx'), sheet_name=None)

        # baseline characteristics
        p['mal_inc'] = workbook['incidence']
        p['interv'] = workbook['interventions']

        p['p_infection'] = 0.5
        p['stage'] = pd.DataFrame(
            data={
                'level_of_symptoms': ['none',
                                      'clinical',
                                      'severe'],
                'probability': [0, 0.99, 0.01],
            })
        p['sensitivity_rdt'] = 0.95
        p['cfr'] = 0.15
        p['dur_asym'] = 110  # how long will this be detectable in the blood??
        p['dur_clin'] = 5
        p['dur_clin_para'] = 195  # how long will this be detectable in the blood??
        p['rr_hiv'] = 1.42

        # get the DALY weight that this module will use from the weight database (these codes are just random!)
        if 'HealthBurden' in self.sim.modules.keys():
            p['daly_wt_none'] = self.sim.modules['HealthBurden'].get_daly_weight(50)
            p['daly_wt_clinical'] = self.sim.modules['HealthBurden'].get_daly_weight(50)
            p['daly_wt_severe'] = self.sim.modules['HealthBurden'].get_daly_weight(589)

    def initialise_population(self, population):
        df = population.props
        p = self.parameters
        now = self.sim.date

        # Set default for properties
        df['ma_is_infected'] = False
        df['ma_status'].values[:] = 'Uninf'  # default: never infected
        df['ma_date_infected'] = pd.NaT

        # ----------------------------------- BASELINE CASES -----------------------------------

        inc = p['mal_inc']

        # find monthly incidence rate for Jan 2010
        inc_2010 = inc.loc[inc.year == now.year, 'monthly_inc_rate'].values

        risk_ml = pd.Series(0, index=df.index)
        risk_ml.loc[df.is_alive] = 1  # applied to all adults
        # risk_ml.loc[df.hv_inf ] *= p['rr_hiv']

        # weight the likelihood of being sampled by the relative risk
        eligible = df.index[df.is_alive]
        norm_p = pd.Series(risk_ml[eligible])
        norm_p /= norm_p.sum()  # normalise
        ml_idx = self.rng.choice(eligible, size=int(inc_2010 * (len(eligible))), replace=False,
                                 p=norm_p)

        df.loc[ml_idx, 'ma_status'] = 'Clin'

        # Assign time of infections across the month
        # random_date = self.rng.randint(low=0, high=31, size=len(ml_idx))
        # random_days = pd.to_timedelta(random_date, unit='d')
        df.loc[ml_idx, 'ma_date_infected'] = now  # + random_days

        # ----------------------------------- INTERVENTIONS -----------------------------------
        interv = p['interv']

        # find annual intervention coverage levels rate for 2010
        itn_2010 = interv.loc[interv.Year == now.year, 'ITN_coverage'].values
        irs_2010 = interv.loc[interv.Year == now.year, 'IRS_coverage'].values

        # get a list of random numbers between 0 and 1 for each person
        random_draw = self.rng.random_sample(size=len(df))

        # use the same random draws for itn and irs as having one increases likelihood of having other
        itn_idx = df.index[df.is_alive & (random_draw < itn_2010)]
        irs_idx = df.index[df.is_alive & (random_draw < irs_2010)]

        df.loc[itn_idx, 'itn_use'] = True
        df.loc[irs_idx, 'irs'] = True

        # ----------------------------------- SCHEDULED DEATHS -----------------------------------
        # schedule deaths within the next week
        # Assign time of infections across the month
        random_draw = self.rng.random_sample(size=len(df))
        death = df.index[
            (df.ma_specific_symptoms == 'severe') & (df.ma_date_infected == now) & (random_draw < p['cfr'])]

        for person in death:
            random_date = self.rng.randint(low=0, high=7)
            random_days = pd.to_timedelta(random_date, unit='d')

            death_event = MalariaDeathEvent(self.module, individual_id=person,
                                            cause='malaria')  # make that death event
            self.sim.schedule_event(death_event, self.sim.date + random_days)  # schedule the death

        # ----------------------------------- HEALTHCARE-SEEKING -----------------------------------

        # find annual intervention coverage levels rate for 2010
        act = interv.loc[interv.Year == now.year, 'ACT_coverage'].values

        seeks_care = pd.Series(data=False, index=df.loc[ml_idx].index)

        for i in df.loc[ml_idx].index:
            # prob = self.sim.modules['HealthSystem'].get_prob_seek_care(i, symptom_code=4)
            seeks_care[i] = self.rng.rand() < act  # placeholder for coverage / testing rates

        if seeks_care.sum() > 0:

            for person_index in seeks_care.index[seeks_care]:
                # print(person_index)

                logger.debug(
                    'This is Malaria, scheduling HSI_Malaria_rdt for person %d',
                    person_index)

                event = HSI_Malaria_rdt(self, person_id=person_index)
                self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                    priority=2,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(weeks=2)
                                                                    )
        else:
            logger.debug(
                'This is Malaria, There is no new healthcare seeking')

        # ----------------------------------- PARASITE CLEARANCE - NO TREATMENT -----------------------------------
        # schedule self-cure if no treatment, no self-cure from severe malaria

        # asymptomatic
        asym = df.index[(df.ma_status == 'Asym') & (df.ma_date_infected == now)]

        random_date = self.rng.randint(low=0, high=p['dur_asym'], size=len(asym))
        random_days = pd.to_timedelta(random_date, unit='d')

        for person in df.loc[asym].index:
            cure = MalariaParasiteClearanceEvent(self, person)
            self.sim.schedule_event(cure, (self.sim.date + random_days))

        # clinical
        clin = df.index[(df.ma_status == 'Clin') & (df.ma_date_infected == now)]

        for person in df.loc[clin].index:

            date_para = self.rng.randint(low=0, high=p['dur_clin_para'])
            date_para_days = pd.to_timedelta(date_para, unit='d')

            date_clin = self.rng.randint(low=0, high=p['dur_clin'])
            date_clin_days = pd.to_timedelta(date_clin, unit='d')

            cure = MalariaParasiteClearanceEvent(self, person)
            self.sim.schedule_event(cure, (self.sim.date + date_para_days))

            # schedule symptom end (5 days)
            symp_end = MalariaSympEndEvent(self, person)
            self.sim.schedule_event(symp_end, self.sim.date + date_clin_days)

        # ----------------------------------- REGISTER WITH HEALTH SYSTEM -----------------------------------
        self.sim.modules['HealthSystem'].register_disease_module(self)

    def initialise_simulation(self, sim):
        # add the basic event
        event = MalariaEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=1))

        # add an event to log to screen - output on last day of each year
        sim.schedule_event(MalariaLoggingEvent(self), sim.date + DateOffset(days=364))

    def on_birth(self, mother_id, child_id):
        pass

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug('This is Malaria, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        logger.debug('This is malaria reporting my health values')

        df = self.sim.population.props  # shortcut to population properties dataframe

        p = self.parameters

        health_values = df.loc[df.is_alive, 'ma_specific_symptoms'].map({
            'none': 0,
            'clinical': p['daly_wt_clinical'],
            'severe': p['daly_wt_severe']
        })
        health_values.name = 'Malaria Symptoms'  # label the cause of this disability

        return health_values.loc[df.is_alive]  # returns the series


class MalariaEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):

        logger.debug('This is MalariaEvent, tracking the disease progression of the population.')

        df = population.props
        p = self.module.parameters
        rng = self.module.rng
        now = self.sim.date

        # ----------------------------------- NEW INFECTIONS -----------------------------------

        inc_df = p['mal_inc']

        # find monthly incidence rate for Jan 2010
        curr_inc = inc_df.loc[inc_df.year == now.year, 'monthly_inc_rate'].values

        risk_ml = pd.Series(0, index=df.index)
        risk_ml.loc[df.is_alive] = 1  # applied to all adults
        # risk_ml.loc[df.hv_inf ] *= p['rr_hiv']

        # weight the likelihood of being sampled by the relative risk
        eligible = df.index[df.is_alive & (df.ma_status == 'Uninf')]
        norm_p = pd.Series(risk_ml[eligible])
        norm_p /= norm_p.sum()  # normalise
        ml_idx = rng.choice(eligible, size=int(curr_inc * (len(eligible))), replace=False,
                            p=norm_p)

        # if any are infected
        if len(ml_idx):
            logger.debug('This is MalariaEvent, assigning new malaria infections')

            symptoms = rng.choice(
                self.module.parameters['stage']['level_of_symptoms'],
                size=len(ml_idx),
                p=self.module.parameters['stage']['probability'])

            df.loc[ml_idx, 'ma_status'] = 'Clin'
            df.loc[ml_idx, 'ma_date_infected'] = self.sim.date  # scatter across the month
            df.loc[ml_idx, 'ma_specific_symptoms'] = symptoms
            df.loc[ml_idx, 'ma_unified_symptom_code'] = 0


            # ----------------------------------- PARASITE CLEARANCE - NO TREATMENT -----------------------------------
            # schedule self-cure if no treatment, no self-cure from severe malaria

            # asymptomatic
            asym = df.index[(df.ma_status == 'Asym') & (df.ma_date_infected == now)]

            random_date = rng.randint(low=0, high=p['dur_asym'], size=len(asym))
            random_days = pd.to_timedelta(random_date, unit='d')

            for person in df.loc[asym].index:
                cure = MalariaParasiteClearanceEvent(self.module, person)
                self.sim.schedule_event(cure, (self.sim.date + random_days))

            # clinical
            clin = df.index[(df.ma_status == 'Clin') & (df.ma_date_infected == now)]

            for person in df.loc[clin].index:

                date_para = rng.randint(low=0, high=p['dur_clin_para'])
                date_para_days = pd.to_timedelta(date_para, unit='d')
                # print('date_para_days', date_para_days)

                date_clin = rng.randint(low=0, high=p['dur_clin'])
                date_clin_days = pd.to_timedelta(date_clin, unit='d')
                # print('date_clin_days', date_clin_days)

                cure = MalariaParasiteClearanceEvent(self.module, person)
                self.sim.schedule_event(cure, (self.sim.date + date_para_days))

                # schedule symptom end (5 days)
                symp_end = MalariaSympEndEvent(self.module, person)
                self.sim.schedule_event(symp_end, self.sim.date + date_clin_days)

            # ----------------------------------- SCHEDULED DEATHS -----------------------------------
            # schedule deaths within the next week
            # Assign time of infections across the month
            random_draw = rng.random_sample(size=len(df))
            # death = df.index[
            #     (df.ma_specific_symptoms == 'severe') & (df.ma_date_infected == now) & (random_draw < p['cfr'])]

            # the cfr applies to all clinical malaria - specific to severe may be even higher
            # currently no asymptomatic cases so all infected are clinical
            death = df.index[
                (df.ma_specific_symptoms == 'severe') & (df.ma_date_infected == now) & (random_draw < p['cfr'])]

            for person in death:

                logger.debug(
                    'This is MalariaEvent, scheduling malaria death for person %d',
                    person)

                random_date = rng.randint(low=0, high=7)
                random_days = pd.to_timedelta(random_date, unit='d')

                death_event = MalariaDeathEvent(self.module, individual_id=person,
                                                cause='malaria')  # make that death event
                self.sim.schedule_event(death_event, self.sim.date + random_days)  # schedule the death

            # ----------------------------------- HEALTHCARE-SEEKING -----------------------------------

            interv = p['interv']

            # find annual intervention coverage levels rate for 2010
            act = interv.loc[interv.Year == now.year, 'ACT_coverage'].values
            # act = 1

            seeks_care = pd.Series(data=False, index=df.loc[ml_idx].index)

            for i in df.loc[ml_idx].index:
                # prob = self.sim.modules['HealthSystem'].get_prob_seek_care(i, symptom_code=4)
                seeks_care[i] = rng.rand() < act  # placeholder for coverage / testing rates

            if seeks_care.sum() > 0:

                for person_index in seeks_care.index[seeks_care]:
                    # print(person_index)

                    logger.debug(
                        'This is MalariaEvent, scheduling HSI_Malaria_rdt for person %d',
                        person_index)

                    event = HSI_Malaria_rdt(self.module, person_id=person_index)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                        priority=2,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(weeks=2)
                                                                        )
            else:
                logger.debug(
                    'This is MalariaEvent, There is no new healthcare seeking')
        else:
            logger.debug('This is MalariaEvent, no one is newly infected.')



class MalariaDeathEvent(Event, IndividualScopeEventMixin):
    """
    Performs the Death operation on an individual and logs it.
    """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)
        self.cause = cause

    def apply(self, individual_id):
        df = self.sim.population.props

        if df.at[individual_id, 'is_alive']:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id, cause='malaria'),
                                    self.sim.date)

            df.at[individual_id, 'ma_date_death'] = self.sim.date


# ---------------------------------------------------------------------------------
# Health System Interaction Events
# ---------------------------------------------------------------------------------


class HSI_Malaria_rdt(Event, IndividualScopeEventMixin):
    """
    this is a point-of-care malaria rapid diagnostic test, with results within 2 minutes
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['LabParasit'] = 1

        # the OneHealth consumables have Intervention_Pkg_Code= -99 which causes errors
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        # this package contains treatment too
        pkg_code1 = pd.unique(
            consumables.loc[
                consumables['Items'] == 'Malaria test kit (RDT)',
                'Intervention_Pkg_Code'])[0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Malaria_RDT'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = ['*']
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):

        df = self.sim.population.props
        params = self.module.parameters
        rng = self.module.rng

        logger.debug('This is HSI_Malaria_rdt, rdt test for person %d', person_id)

        # check if diagnosed
        if (df.at[person_id, 'ma_status'] != 'Uninfected'):

            # ----------------------------------- SEVERE MALARIA -----------------------------------

            # if severe malaria, treat for complicated malaria
            if (df.at[person_id, 'ma_specific_symptoms'] == 'severe'):

                if (df.at[person_id, 'age_years'] < 15):

                    logger.debug(
                        "This is HSI_Malaria_rdt scheduling HSI_Malaria_tx_compl_child for person %d on date %s",
                        person_id, (self.sim.date + DateOffset(days=1)))

                    treat = HSI_Malaria_tx_compl_child(self.module, person_id=person_id)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(treat,
                                                                        priority=1,
                                                                        topen=self.sim.date + DateOffset(days=1),
                                                                        tclose=None)

                else:
                    logger.debug(
                        "This is HSI_Malaria_rdt scheduling HSI_Malaria_tx_compl_adult for person %d on date %s",
                        person_id, (self.sim.date + DateOffset(days=1)))

                    treat = HSI_Malaria_tx_compl_adult(self.module, person_id=person_id)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(treat,
                                                                        priority=1,
                                                                        topen=self.sim.date + DateOffset(days=1),
                                                                        tclose=None)

            # ----------------------------------- TREATMENT CLINICAL DISEASE -----------------------------------

            else:
                # diagnosis of clinical disease dependent on RDT sensitivity
                diagnosed = self.sim.rng.choice([True, False], size=1, p=[params['sensitivity_rdt'],
                                                                          (1 - params['sensitivity_rdt'])])

                # diagnosis / treatment for children <5
                if diagnosed & (df.at[person_id, 'age_years'] < 5):
                    logger.debug("This is HSI_Malaria_rdt scheduling HSI_Malaria_tx_0_5 for person %d on date %s",
                                 person_id, (self.sim.date + DateOffset(days=1)))

                    treat = HSI_Malaria_tx_0_5(self.module, person_id=person_id)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(treat,
                                                                        priority=1,
                                                                        topen=self.sim.date + DateOffset(days=1),
                                                                        tclose=None)

                # diagnosis / treatment for children 5-15
                if diagnosed & (df.at[person_id, 'age_years'] >= 5) & (df.at[person_id, 'age_years'] < 15):
                    logger.debug("This is HSI_Malaria_rdt scheduling HSI_Malaria_tx_5_15 for person %d on date %s",
                                 person_id, (self.sim.date + DateOffset(days=1)))

                    treat = HSI_Malaria_tx_5_15(self.module, person_id=person_id)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(treat,
                                                                        priority=1,
                                                                        topen=self.sim.date + DateOffset(days=1),
                                                                        tclose=None)

                # diagnosis / treatment for adults
                if diagnosed & (df.at[person_id, 'age_years'] >= 15):
                    logger.debug("This is HSI_Malaria_rdt scheduling HSI_Malaria_tx_adult for person %d on date %s",
                                 person_id, (self.sim.date + DateOffset(days=1)))

                    treat = HSI_Malaria_tx_adult(self.module, person_id=person_id)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(treat,
                                                                        priority=1,
                                                                        topen=self.sim.date + DateOffset(days=1),
                                                                        tclose=None)


class HSI_Malaria_tx_0_5(Event, IndividualScopeEventMixin):
    """
    this is anti-malarial treatment for children <15 kg. Includes treatment plus one rdt
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Under5OPD'] = 1  # This requires one out patient

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(
            consumables.loc[
                consumables['Intervention_Pkg'] == 'Uncomplicated (children, <15 kg)',
                'Intervention_Pkg_Code'])[0]  # this pkg_code includes another rdt

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Malaria_treatment_child0_5'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = ['*']
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        logger.debug('This is HSI_Malaria_tx_0_5, malaria treatment for child %d',
                     person_id)

        df = self.sim.population.props

        df.at[person_id, 'ma_tx'] = True
        df.at[person_id, 'ma_date_tx'] = self.sim.date

        self.sim.schedule_event(MalariaCureEvent(self.module, person_id), self.sim.date + DateOffset(weeks=1))


class HSI_Malaria_tx_5_15(Event, IndividualScopeEventMixin):
    """
    this is anti-malarial treatment for children >15 kg. Includes treatment plus one rdt
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Under5OPD'] = 1  # This requires one out patient

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(
            consumables.loc[
                consumables['Intervention_Pkg'] == 'Uncomplicated (children, >15 kg)',
                'Intervention_Pkg_Code'])[0]  # this pkg_code includes another rdt

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Malaria_treatment_child5_15'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = ['*']
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        logger.debug('This is HSI_Malaria_tx_5_15, malaria treatment for child %d',
                     person_id)

        df = self.sim.population.props

        df.at[person_id, 'ma_tx'] = True
        df.at[person_id, 'ma_date_tx'] = self.sim.date

        self.sim.schedule_event(MalariaCureEvent(self.module, person_id), self.sim.date + DateOffset(weeks=1))


class HSI_Malaria_tx_adult(Event, IndividualScopeEventMixin):
    """
    this is anti-malarial treatment for adults. Includes treatment plus one rdt
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(
            consumables.loc[
                consumables['Intervention_Pkg'] == 'Uncomplicated (adult, >36 kg)',
                'Intervention_Pkg_Code'])[0]  # this pkg_code includes another rdt

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Malaria_treatment_adult'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = ['*']
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        logger.debug('This is HSI_Malaria_tx_adult, malaria treatment for person %d',
                     person_id)

        df = self.sim.population.props

        df.at[person_id, 'ma_tx'] = True
        df.at[person_id, 'ma_date_tx'] = self.sim.date

        self.sim.schedule_event(MalariaCureEvent(self.module, person_id), self.sim.date + DateOffset(weeks=1))


class HSI_Malaria_tx_compl_child(Event, IndividualScopeEventMixin):
    """
    this is anti-malarial treatment for complicated malaria in children
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Under5OPD'] = 1  # This requires one out patient

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(
            consumables.loc[
                consumables['Intervention_Pkg'] == 'Complicated (children, injectable artesunate)',
                'Intervention_Pkg_Code'])[0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Malaria_treatment_complicated_child'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = ['*']
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        logger.debug('This is HSI_Malaria_tx_compl_child, complicated malaria treatment for child %d',
                     person_id)

        df = self.sim.population.props

        df.at[person_id, 'ma_tx'] = True
        df.at[person_id, 'ma_date_tx'] = self.sim.date

        self.sim.schedule_event(MalariaCureEvent(self.module, person_id), self.sim.date + DateOffset(weeks=1))


class HSI_Malaria_tx_compl_adult(Event, IndividualScopeEventMixin):
    """
    this is anti-malarial treatment for complicated malaria in adults
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(
            consumables.loc[
                consumables['Intervention_Pkg'] == 'Complicated (adults, injectable artesunate)',
                'Intervention_Pkg_Code'])[0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Malaria_treatment_complicated_adult'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = ['*']
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        logger.debug('This is HSI_Malaria_tx_compl_adult, complicated malaria treatment for person %d',
                     person_id)

        df = self.sim.population.props

        df.at[person_id, 'ma_tx'] = True
        df.at[person_id, 'ma_date_tx'] = self.sim.date

        self.sim.schedule_event(MalariaCureEvent(self.module, person_id), self.sim.date + DateOffset(weeks=1))


# ---------------------------------------------------------------------------------
# Recovery Events
# ---------------------------------------------------------------------------------
class MalariaCureEvent(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        logger.debug("Stopping malaria treatment and curing person %d", person_id)

        df = self.sim.population.props

        # stop treatment
        if df.at[person_id, 'is_alive']:
            df.at[person_id, 'ma_tx'] = False

            df.at[person_id, 'ma_status'] = 'Uninf'


class MalariaParasiteClearanceEvent(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        logger.debug("This is MalariaParasiteClearanceEvent for person %d", person_id)

        df = self.sim.population.props

        if df.at[person_id, 'is_alive']:

            df.at[person_id, 'ma_status'] = 'Uninf'


class MalariaSympEndEvent(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        logger.debug("This is MalariaSympEndEvent ending symptoms of clinical malaria for person %d", person_id)

        df = self.sim.population.props

        if df.at[person_id, 'is_alive']:
            df.at[person_id, 'mi_specific_symptoms'] = 'none'


# ---------------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------------

class MalariaLoggingEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props
        now = self.sim.date

        # ------------------------------------ INCIDENCE ------------------------------------

        # infected in the last year, clinical and severe cases only
        # incidence rate per 1000 person-years
        tmp = len(
            df.loc[df.is_alive & (df.ma_date_infected > (now - DateOffset(months=self.repeat)))])
        pop = len(df[(df.ma_status == 'Uninf') & df.is_alive])

        inc_1000py = (tmp / pop) * 1000


        tmp2 = len(
            df.loc[df.is_alive & (df.age_years.between(2, 10)) & (df.ma_date_infected > (now - DateOffset(months=self.repeat)))])
        pop2_10 = len(df[df.is_alive & (df.age_years.between(2, 10))])
        inc_1000py_2_10 = (tmp2 / pop2_10) * 1000


        # inc_1000py_hiv = (tmp / len(df[(df.ma_status == 'Uninf') & df.is_alive & df.hv_inf])) * 1000
        inc_1000py_hiv = 0  # if running without hiv/tb

        logger.info('%s|incidence|%s', now,
                    {
                        'number_new_cases': tmp,
                        'population': pop,
                        'inc_1000py': inc_1000py,
                        'inc_1000py_hiv': inc_1000py_hiv,
                        'new_cases_2_10': tmp2,
                        'population2_10': pop2_10,
                        'inc_1000py_2_10': inc_1000py_2_10
                    })

        # ------------------------------------ RUNNING COUNTS ------------------------------------

        counts = {'Uninf': 0, 'Asym': 0, 'Clin': 0}
        counts.update(df.loc[df.is_alive, 'ma_status'].value_counts().to_dict())

        logger.info('%s|status_counts|%s', now, counts)

        # ------------------------------------ PREVALENCE BY AGE ------------------------------------

        # clinical cases including severe
        # assume asymptomatic parasitaemia is too low to detect in surveys
        child2_10_clin = len(df[df.is_alive & (df.ma_status == 'Clin') & (df.age_years.between(2, 10))])

        # population size - children
        child2_10_pop = len(df[df.is_alive & (df.age_years.between(2, 10))])

        # prevalence in children aged 2-10
        child_prev = child2_10_clin / child2_10_pop if child2_10_pop else 0

        # prevalence of clinical including severe in all ages
        total_clin = len(df[df.is_alive & (df.ma_status == 'Clin')])
        pop2 = len(df[ df.is_alive])
        prev_clin = total_clin / pop2

        logger.info('%s|prevalence|%s', now,
                    {
                        'child2_10_prev': child_prev,
                        'clinical_prev': prev_clin,
                    })

        # ------------------------------------ TREATMENT COVERAGE ------------------------------------
        # prop on treatment, all ages

        # people who had treatment start date within last year
        tx = len(df[df.is_alive & (df.ma_date_tx > (now - DateOffset(months=self.repeat)))])

        # people who had malaria infection date within last year
        # includes asymp
        denom = len(df[df.is_alive & (df.ma_date_infected > (now - DateOffset(months=self.repeat)))])
        tx_coverage = tx / denom if denom else 0

        logger.info('%s|tx_coverage|%s', now,
                    {
                        'treatment_coverage': tx_coverage,
                    })

        # ------------------------------------ MORTALITY ------------------------------------
        # deaths reported in the last 12 months / pop size
        deaths = len(df[(df.ma_date_death > (now - DateOffset(months=self.repeat)))])

        mort_rate = deaths / len(df[df.is_alive])

        logger.info('%s|ma_mortality|%s', now,
                    {
                        'mort_rate': mort_rate,
                    })
