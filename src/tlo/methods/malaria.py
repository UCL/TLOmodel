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
            Types.REAL, 'duration (days) of clinical malaria '),
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
        'ma_itn_use': Property(
            Types.BOOL, 'Person sleeps under a bednet'),
        'ma_irs': Property(
            Types.BOOL, 'Person sleeps in house which has had indoor residual spraying'),
        'ml_tx': Property(Types.BOOL, 'Currently on anti-malarial treatment'),
        'ml_specific_symptoms': Property(
            Types.CATEGORICAL,
            'specific symptoms with malaria infection: none; clinical; severe'),
        'ml_unified_symptom_code': Property(Types.CATEGORICAL, 'level of symptoms on the standardised scale, 0-4',
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
                'probability': [0.2, 0.5, 0.3],
            })
        p['sensitivity_rdt'] = 0.95
        p['cfr'] = 0.15
        p['dur_asym'] = 110
        p['dur_clin'] = 5

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

        # get a list of random numbers between 0 and 1 for each susceptible individual
        random_draw = self.rng.random_sample(size=len(df))

        ml_idx = df.index[df.is_alive & (random_draw < inc_2010)]
        df.loc[ml_idx, 'ma_status'] = 'Clin'

        # Assign time of infections across the month
        random_date = self.rng.randint(low=0, high=31, size=len(ml_idx))
        random_days = pd.to_timedelta(random_date, unit='d')
        df.loc[ml_idx, 'ma_date_infected'] = self.sim.date + random_days

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

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

    def initialise_simulation(self, sim):
        # add the basic event
        event = MalariaEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=1))

        # add an event to log to screen
        sim.schedule_event(MalariaLoggingEvent(self), sim.date + DateOffset(months=0))

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

        health_values = df.loc[df.is_alive, 'ml_specific_symptoms'].map({
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

        # get a list of random numbers between 0 and 1 for each susceptible individual
        random_draw = rng.random_sample(size=len(df))

        ml_idx = df.index[df.is_alive & (df.ma_status == 'Uninf') & (random_draw < curr_inc)]

        # if any are infected
        if len(ml_idx):
            logger.debug('This is MalariaEvent, assigning new malaria infections')

            symptoms = rng.choice(
                self.module.parameters['stage']['level_of_symptoms'],
                size=len(ml_idx),
                p=self.module.parameters['stage']['probability'])

            df.loc[ml_idx, 'ma_status'] = 'Clin'
            df.loc[ml_idx, 'ma_date_infected'] = self.sim.date
            df.loc[ml_idx, 'ma_specific_symptoms'] = symptoms
            df.loc[ml_idx, 'ma_unified_symptom_code'] = 0

            # Determine if anyone with symptoms will seek care
            symptoms = df.index[(df['is_alive']) & (
                (df['ma_specific_symptoms'] == 'severe') | ((df['ma_specific_symptoms'] == 'clinical')))]
            # print('symptoms', symptoms)

            # ----------------------------------- SCHEDULED DEATHS -----------------------------------
            # schedule deaths within the next week
            # Assign time of infections across the month
            severe = df.index[(df.ma_specific_symptoms == 'severe') & (df.ma_date_infected == now)]

            random_date = rng.randint(low=0, high=7, size=len(severe))
            random_days = pd.to_timedelta(random_date, unit='d')
            df.loc[severe, 'ma_date_death'] = self.sim.date + random_days

            random_draw = rng.random_sample(size=len(df))
            death = df.index[(df.ma_specific_symptoms == 'severe') & (df.ma_date_infected == now) & (random_draw < p['cfr'])]

            for person in death:
                death_event = MalariaDeathEvent(self, individual_id=person, cause='malaria')  # make that death event
                self.sim.schedule_event(death_event, df.at[person, 'ma_date_death'])  # schedule the death

            # ----------------------------------- HEALTHCARE-SEEKING -----------------------------------

            seeks_care = pd.Series(data=False, index=df.loc[symptoms].index)

            for i in df.loc[symptoms].index:
                prob = self.sim.modules['HealthSystem'].get_prob_seek_care(i, symptom_code=4)
                seeks_care[i] = rng.rand() < prob

            if seeks_care.sum() > 0:

                for person_index in df.index[seeks_care.index]:
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
                    'This is MalariaEvent, There is  no one with new severe symptoms so no new healthcare seeking')
        else:
            logger.debug('This is MalariaEvent, no one is newly infected.')

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

            random_date = rng.randint(low=0, high=p['dur_clin'], size=len(clin))
            random_days = pd.to_timedelta(random_date, unit='d')

            for person in df.loc[clin].index:
                cure = MalariaParasiteClearanceEvent(self.module, person)
                self.sim.schedule_event(cure, (self.sim.date + random_days))

                # schedule symptom end (5 days)
                symp_end = MalariaSympEndEvent(self.module, person)
                self.sim.schedule_event(symp_end, self.sim.date + DateOffset(days=5))


class MalariaDeathEvent(Event, IndividualScopeEventMixin):
    """
    Performs the Death operation on an individual and logs it.
    """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)
        self.cause = cause

    def apply(self, individual_id):
        df = self.sim.population.props

        if df.at[individual_id, 'is_alive'] and not df.at[individual_id, 'ml_tx']:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id, cause='malaria'),
                                    self.sim.date)

        elif df.at[individual_id, 'is_alive'] and df.at[individual_id, 'ml_tx']:
            # schedule death with some probability relating to late / ineffective tx

            cfr_tx = 0.2
            death = self.module.rng.rand() < cfr_tx

            if death:
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id, cause='malaria'),
                                        self.sim.date)

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
        # pkg_code1 = pd.unique(
        #     consumables.loc[
        #         consumables['Items'] == 'malaria P. falciparum + P. pan  RDT',
        #         'Intervention_Pkg_Code'])[0]

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

        logger.debug('This is HSI_Malaria_rdt, rdt test for person %d', person_id)

        # check if diagnosed
        if (df.at[person_id, 'ma_status'] != 'Uninfected'):

            # ----------------------------------- SEVERE MALARIA -----------------------------------

            # if severe malaria, treat for complicated malaria
            if(df.at[person_id, 'ma_specific_symptoms'] == 'severe'):

                if (df.at[person_id, 'age_years'] < 15):

                    logger.debug("This is HSI_Malaria_rdt scheduling HSI_Malaria_tx_compl_child for person %d on date %s",
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

        df.at[person_id, 'ml_tx'] = True

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

        df.at[person_id, 'ml_tx'] = True

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

        df.at[person_id, 'ml_tx'] = True

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

        df.at[person_id, 'ml_tx'] = True

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

        df.at[person_id, 'ml_tx'] = True

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
            df.at[person_id, 'ml_tx'] = False

            df.at[person_id, 'ma_status'] = 'Uninf'


class MalariaParasiteClearanceEvent(Event, IndividualScopeEventMixin):

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        logger.debug("This is MalariaParasiteClearanceEvent for person %d", person_id)

        df = self.sim.population.props

        if df.at[person_id, 'is_alive']:
            df.at[person_id, 'ml_tx'] = False

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
        tmp = len(
            df.loc[df.is_alive & (df.ma_date_infected > (now - DateOffset(months=self.repeat)))])
        # incidence rate per 1000 person-years
        inc_1000py = (tmp / len(df[(df.ma_status == 'Uninf') & df.is_alive])) * 1000

        logger.info('%s|incidence|%s', now, inc_1000py)

        # ------------------------------------ RUNNING COUNTS ------------------------------------

        counts = {'Uninf': 0, 'Asym': 0, 'Clin': 0}
        counts.update(df.loc[df.is_alive, 'ma_status'].value_counts().to_dict())

        logger.info('%s|status_counts|%s', now, counts)

        # ------------------------------------ PREVALENCE BY AGE ------------------------------------

        # if groupby both sex and age_range, you lose categories where size==0, get the counts separately

        # clinical cases including severe
        child2_10_clin = len(df[df.is_alive & (df.ma_status == 'Clin') & (df.age_years.between(2, 10))])

        # severe cases
        child2_10_sev = len(df[df.is_alive & (df.ma_specific_symptoms == 'severe') & (df.age_years.between(2, 10))])

        # population size - children
        child2_10_pop = len(df[df.is_alive & (df.age_years.between(2, 10))])

        # prevalence in children aged 2-10
        child_prev = (child2_10_clin + child2_10_sev) / child2_10_pop if child2_10_pop else 0

        # prevalence of clinical including severe in all ages
        prev_clin = len(df[df.is_alive & (df.ma_status == 'Clin')])

        # prevalence severe in all ages
        prev_sev = len(df[df.is_alive & (df.ma_specific_symptoms == 'severe')])

        # prevalence in all ages
        total_prev = (prev_clin + prev_sev) / len(df[df.is_alive])

        logger.info('%s|prevalence|%s', now,
                    {
                        'child_prev': child_prev,
                        'total_prev': total_prev,
                    })
