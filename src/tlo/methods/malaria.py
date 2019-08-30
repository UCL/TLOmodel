import logging

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods.demography import InstantaneousDeath

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Malaria(Module):

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    PARAMETERS = {
        'p_infection': Parameter(
            Types.REAL, 'Probability that an uninfected individual becomes infected'),
        'stage': Parameter(
            Types.CATEGORICAL, 'malaria stage'),
    }

    PROPERTIES = {
        'ma_is_infected': Property(
            Types.BOOL, 'Current status of mockitis'),
        'ma_status': Property(
            Types.CATEGORICAL, 'current malaria stage: Uninf=uninfected; Asym=asymptomatic; Clin=clinical; Sev=severe; Past=past',
            categories=['Uninf', 'Asym', 'Clin', 'Sev', 'Past']),
        'ma_date_infected': Property(
            Types.DATE, 'Date of latest infection'),

    }

    def read_parameters(self, data_folder):
        p = self.parameters

        p['p_infection'] = 0.5
        p['stage'] = pd.DataFrame(
            data={
                'level_of_symptoms': ['none',
                                      'clinical',
                                      'severe'],
                'probability': [0.2, 0.5, 0.3]
            })

        # get the DALY weight that this module will use from the weight database (these codes are just random!)
        if 'HealthBurden' in self.sim.modules.keys():
            p['daly_wt_none'] = self.sim.modules['HealthBurden'].get_daly_weight(50)
            p['daly_wt_clinical'] = self.sim.modules['HealthBurden'].get_daly_weight(50)
            p['daly_wt_severe'] = self.sim.modules['HealthBurden'].get_daly_weight(589)

    def initialise_population(self, population):

        df = population.props

        # Set default for properties
        df['ma_is_infected'] = False
        df['ma_status'].values[:] = 'Uninf'  # default: never infected
        df['ma_date_infected'] = pd.NaT

        # randomly selected some individuals as infected
        at_risk = df[(df.ma_status == 'Uninf') & df.is_alive].index

        prob_new = pd.Series(self.parameters['p_infection'], index=at_risk)
        print('prob_new: ', prob_new)

        is_newly_infected = prob_new > self.rng.rand(len(prob_new))
        new_case = is_newly_infected[is_newly_infected].index
        print('new_case', new_case)
        df.loc[new_case, 'ma_status'] = 'Clin'

        # Assign time of infections
        # date of infection of infected individuals
        # schedule random day throughout 2010 for infection to begin
        # random draw of days 0-365
        random_date = self.rng.randint(low=0, high=365, size=len(new_case))
        random_days = pd.to_timedelta(random_date, unit='d')
        df.loc[new_case, 'ma_date_infected'] = self.sim.date + random_days

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

    def initialise_simulation(self, sim):

        # add the basic event
        event = MalariaEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=12))

        # add an event to log to screen
        sim.schedule_event(MalariaLoggingEvent(self), sim.date + DateOffset(months=6))


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

        health_values = df.loc[df.is_alive, 'ma_status'].map({
            'Uninf': 0,
            'Asym': 0,
            'Clin': p['daly_wt_clinical'],
            'Sev': p['daly_wt_severe']
        })
        health_values.name = 'Malaria Symptoms'    # label the cause of this disability

        return health_values.loc[df.is_alive]   # returns the series


class MalariaEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=12))

    def apply(self, population):

        logger.debug('This is MalariaEvent, tracking the disease progression of the population.')

        df = population.props

        # 1. get (and hold) index of currently infected and uninfected individuals
        currently_infected = df.index[(df.ma_status == 'Clin') & df.is_alive]
        currently_uninfected = df.index[(df.ma_status != 'Clin') & df.is_alive]

        if df.is_alive.sum():
            prevalence = len(currently_infected) / (
                len(currently_infected) + len(currently_uninfected))
        else:
            prevalence = 0
        print('prevalence', prevalence)

        # 2. handle new infections
        now_infected = np.random.choice([True, False], size=len(currently_uninfected),
                                        p=[prevalence, 1 - prevalence])

        # if any are infected
        if now_infected.sum():
            infected_idx = currently_uninfected[now_infected]

            symptoms = self.module.rng.choice(
                self.module.parameters['stage']['level_of_symptoms'],
                size=now_infected.sum(),
                p=self.module.parameters['stage']['probability'])

            df.loc[infected_idx, 'ma_status'] = 'Clin'
            df.loc[infected_idx, 'ma_date_infected'] = self.sim.date
            df.loc[infected_idx, 'mi_specific_symptoms'] = symptoms
            df.loc[infected_idx, 'mi_unified_symptom_code'] = 0

            # Determine if anyone with severe symptoms will seek care
            serious_symptoms = (df['is_alive']) & ((df['mi_specific_symptoms'] == 'severe'))

            seeks_care = pd.Series(data=False, index=df.loc[serious_symptoms].index)
            for i in df.index[serious_symptoms]:
                prob = self.sim.modules['HealthSystem'].get_prob_seek_care(i, symptom_code=4)
                seeks_care[i] = self.module.rng.rand() < prob

            if seeks_care.sum() > 0:
                for person_index in seeks_care.index[seeks_care is True]:
                    logger.debug(
                        'This is MalariaEvent, scheduling Malaria_PresentsForCareWithSevereSymptoms for person %d',
                        person_index)
                    event = HSI_Malaria_PresentsForCareWithSevereSymptoms(self.module, person_id=person_index)
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


class MalariaDeathEvent(Event, IndividualScopeEventMixin):
    """
    This is the death event for malaria
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe

        # Apply checks to ensure that this death should occur
        if df.at[person_id, 'mi_specific_symptoms'] == 'severe':

            will_die = 0.2 < self.module.rng.rand()

            if will_die:

                # Fire the centralised death event:
                death = InstantaneousDeath(self.module, person_id, cause='Malaria')
                self.sim.schedule_event(death, self.sim.date)


# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# Health System Interaction Events




class HSI_Malaria_rdt(Event, IndividualScopeEventMixin):
    """
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['LabParasit'] = 1  # This requires one out patient

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(
            consumables.loc[
                consumables['Items'] == 'malaria P. falciparum + P. pan  RDT',
                'Intervention_Pkg_Code'])[0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Malaria_RDT'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = []
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):

        logger.debug('This is HSI_Malaria_rdt, rdt test for person %d',
                     person_id)








class HSI_Malaria_PresentsForCareWithSevereSymptoms(Event, IndividualScopeEventMixin):
    """
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Malaria_PresentsForCareWithSevereSymptoms'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = self.sim.modules['HealthSystem'].get_blank_cons_footprint()
        self.ACCEPTED_FACILITY_LEVELS = [0]     # This enforces that the apppointment must be run at that facility-level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):

        logger.debug('This is HSI_Malaria_PresentsForCareWithSevereSymptoms, a first appointment for person %d',
                     person_id)

        df = self.sim.population.props  # shortcut to the dataframe

        if df.at[person_id, 'age_years'] >= 15:
            logger.debug(
                '...This is HSI_Malaria_PresentsForCareWithSevereSymptoms: \
                there should now be treatment for person %d',
                person_id)
            # schedule event


        else:
            logger.debug(
                '...This is HSI_Malaria_PresentsForCareWithSevereSymptoms: there will not be treatment for person %d',
                person_id)



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
        inc_1000py = (tmp / len(df[(df.ma_status == 'Uninf') & df.is_alive ])) * 1000

        logger.info('%s|incidence|%s', now, inc_1000py)

        # ------------------------------------ RUNNING COUNTS ------------------------------------

        counts = {'Uninf': 0, 'Asym': 0, 'Clin': 0, 'Sev': 0}
        counts.update(df.loc[df.is_alive, 'ma_status'].value_counts().to_dict())

        logger.info('%s|status_counts|%s', now, counts)

        # ------------------------------------ PREVALENCE BY AGE ------------------------------------

        # if groupby both sex and age_range, you lose categories where size==0, get the counts separately
        child2_10_clin = len(df[df.is_alive & (df.ma_status == 'Clin') & (df.age_years.between(2,10))])
        child2_10_sev = len(df[df.is_alive & (df.ma_status == 'Sev') & (df.age_years.between(2,10))])
        child2_10_pop = len(df[df.is_alive & (df.age_years.between(2, 10))])
        child_prev = (child2_10_clin +  child2_10_sev) / child2_10_pop if child2_10_pop else 0

        prev_clin = len(df[df.is_alive & (df.ma_status == 'Clin')])
        prev_sev = len(df[df.is_alive & (df.ma_status == 'Sev')])
        total_prev = (prev_clin + prev_sev) / len(df[df.is_alive])

        logger.info('%s|prevalence|%s', now,
                    {
                        'child_prev': child_prev,
                        'total_prev': total_prev,
                    })
