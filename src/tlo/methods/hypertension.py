"""
This is the method for hypertension
Developed by Mikaela Smit, October 2018

"""

import logging

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods.demography import InstantaneousDeath

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Read in data
file_path = '/Users/mc1405/Dropbox/Projects - ongoing/Malawi Project/Thanzi la Onse/04 - Methods Repository/Method_HT.xlsx'
method_ht_data = pd.read_excel(file_path, sheet_name=None, header=0)
HT_prevalence, HT_incidence, HT_treatment, HT_risk = method_ht_data['prevalence2018'], method_ht_data['incidence2018_plus'], \
                                            method_ht_data['treatment_parameters'], method_ht_data['parameters']

class HT(Module):
    """
    This is hypertension.
    It demonstrates the following behaviours in respect of the healthsystem module:

    - Declaration of TREATMENT_ID
    - Registration of the disease module
    - Reading QALY weights and reporting qaly values related to this disease
    - Health care seeking
    - Running an "outreach" event
    """

    PARAMETERS = {
        # ToDo: Update HR and variables to mirror finalised method
        # ToDO: Remove symptoms (and QALYs)

        # 1. Risk parameters
        'prob_HT_basic': Parameter(Types.REAL,               'Probability of hypertension given no pre-existing condition'),
        'prob_HTgivenHC': Parameter(Types.REAL,              'Probability of getting hypertension given pre-existing high cholesterol'),
        'prob_HTgivenDiab': Parameter(Types.REAL,            'Probability of getting hypertension given pre-existing diabetes'),
        'prob_HTgivenHIV': Parameter(Types.REAL,             'Probability of getting hypertension given pre-existing HIV'),
        'prob_death': Parameter(Types.REAL,                  'Probability of dying'),

        # 2. Health care parameters
        'prob_diag': Parameter(Types.REAL,                   'Probability of being diagnosed'),
        'prob_treat': Parameter(Types.REAL,                  'Probability of being treated'),
        'prob_contr': Parameter(Types.REAL,                  'Probability achieving normal blood pressure levels on medication'),
        'level_of_symptoms': Parameter(Types.CATEGORICAL,    'Level of symptoms that the individual will have'),
        'qalywt_mild_sneezing': Parameter(Types.REAL,        'QALY weighting for mild sneezing'),
        'qalywt_coughing': Parameter(Types.REAL,             'QALY weighting for coughing'),
        'qalywt_advanced': Parameter(Types.REAL,             'QALY weighting for extreme emergency'),

    }

    PROPERTIES = {
        # ToDo: Remove death and symptoms

        # 1. Disease properties
        'ht_risk': Property(Types.REAL,                       'Risk of hypertension given pre-existing condition'),
        'ht_date': Property(Types.DATE,                       'Date of latest hypertension'),
        'ht_current_status': Property(Types.BOOL,             'Current hypertension status'),
        'ht_historic_status': Property(Types.CATEGORICAL,     'Historical status: N=never; C=Current, P=Previous',
                                                               categories=['N', 'C', 'P']),
        'ht_death_date': Property(Types.DATE,       'Date of scheduled death of infected individual'),

        # 2. Health care properties
        'ht_diag_date': Property(Types.DATE,                  'Date of latest hypertension diagnosis'),
        'ht_diag_status': Property(Types.CATEGORICAL,         'Status: N=No; Y=Yes',
                                                               categories=['N', 'C', 'P']),
        'ht_treat_date': Property(Types.DATE,                 'Date of latest hypertension treatment'),
        'ht_treat_status': Property(Types.CATEGORICAL,        'Status: N=never; C=Current, P=Previous',
                                                               categories=['N', 'C', 'P']),
        'ht_contr_date': Property(Types.DATE,                 'Date of latest hypertension control'),
        'ht_contr_status': Property(Types.CATEGORICAL,        'Status: N=No; Y=Yes',
                                                               categories=['N', 'C', 'P']),
        'ht_specific_symptoms': Property(Types.CATEGORICAL,   'Level of symptoms for HT specifically',
                                                               categories=['none', 'mild sneezing', 'coughing and irritable', 'extreme emergency']),
        'ht_unified_symptom_code': Property(Types.CATEGORICAL,'Level of symptoms on the standardised scale (governing health-care seeking): '
                                                              '0=None; 1=Mild; 2=Moderate; 3=Severe; 4=Extreme_Emergency',
                                                               categories=[0, 1, 2, 3, 4]),

    }

    def read_parameters(self, data_folder):
        # TODO: Read in risks from file, test it 'holds' them in python
        # TODO: update to data HR and variables here and above

        # 1. Shortcut to parameters
        p = self.parameters

        # 2. Risk parameters
        p['prob_HT_basic']      = 1.0
        p['prob_HTgivenHC']     = 2.0
        p['prob_HTgivenDiab']   = 1.4
        p['prob_HTgivenHIV']    = 1.49
        p['prob_death']         = 0.5

        # 3. Health care parameter
        p['prob_diag']          = 0.5
        p['prob_treat']         = 0.5
        p['prob_contr']         = 0.5
        p['level_of_symptoms']  = pd.DataFrame(data={'level_of_symptoms':
                                      ['none',
                                      'mild sneezing',
                                      'coughing and irritable',
                                      'extreme emergency'],
                                      'probability': [0.25, 0.25, 0.25, 0.25]
                                    })
        p['qalywt_mild_sneezing']     = self.sim.modules['QALY'].get_qaly_weight(50)
        p['qalywt_coughing']          = self.sim.modules['QALY'].get_qaly_weight(50)
        p['qalywt_extreme_emergency'] = self.sim.modules['QALY'].get_qaly_weight(50)


    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        # 1. Define key variables
        df = population.props                               # Shortcut to the data frame storing individual data

        # 2. Disease and health care properties
        df.loc[df.is_alive,'ht_risk'] = 1.0                 # Default setting: no risk given pre-existing conditions
        df.loc[df.is_alive,'ht_current_status'] = False     # Default setting: no one has hypertension
        df.loc[df.is_alive,'ht_historic_status'] = 'N'      # Default setting: no one has hypertension
        df.loc[df.is_alive,'ht_date'] = pd.NaT              # Default setting: no one has hypertension
        df.loc[df.is_alive,'ht_death_date'] = pd.NaT        # Default setting: no one has hypertension
        df.loc[df.is_alive,'ht_diag_date'] = pd.NaT         # Default setting: no one is diagnosed
        df.loc[df.is_alive,'ht_diag_status'] = 'N'          # Default setting: no one is diagnosed
        df.loc[df.is_alive,'ht_treat_date'] = pd.NaT        # Default setting: no one is treated
        df.loc[df.is_alive,'ht_treat_status'] = 'N'         # Default setting: no one is treated
        df.loc[df.is_alive,'ht_contr_date'] = pd.NaT        # Default setting: no one is controlled
        df.loc[df.is_alive,'ht_contr_status'] = 'N'         # Default setting: no one is controlled
        df.loc[df.is_alive,'ht_specific_symptoms'] = 'none' # Default setting: no one has symptoms
        df.loc[df.is_alive,'ht_unified_symptom_code'] = 0   # Default setting: no one has symptoms

        # 3. Assign prevalence as per data
        alive_count = df.is_alive.sum()

        # ToDO: Re-add this later after testing HT alone
        # 3.1 First get relative risk for hypertension
        ht_prob = df.loc[df.is_alive, ['ht_risk', 'age_years']].merge(HT_prevalence, left_on=['age_years'], right_on=['age'],
                                                                      how='left')['probability']
        # df.loc[df.is_alive & ~df.hc_current_status, 'ht_risk'] = self.prob_HT_basic  # Basic risk, no pre-existing conditions
        # df.loc[df.is_alive & df.hc_current_status, 'ht_risk'] = self.prob_HTgivenHC  # Risk if pre-existing high cholesterol
        assert alive_count == len(ht_prob)
        ht_prob = ht_prob * df.loc[df.is_alive, 'ht_risk']
        random_numbers = self.rng.random_sample(size=alive_count)
        df.loc[df.is_alive, 'ht_current_status'] = (random_numbers < ht_prob)
        hypertension_count = (df.is_alive & df.ht_current_status).sum()

        # 3.2. Calculate prevalence
        count =df.ht_current_status.sum()
        prevalence = (sum(count) / alive_count) * 100

        # 4. Assign level of symptoms
        level_of_symptoms = self.parameters['level_of_symptoms']
        symptoms = self.rng.choice(level_of_symptoms.level_of_symptoms,
                                   size=hypertension_count,
                                   p=level_of_symptoms.probability)
        df.loc[df.ht_current_status, 'ht_specific_symptoms'] = symptoms

        # 5. Set date of death # TODO: correct this to parameter not distribution and bbelow in event
        death_years_ahead = np.random.exponential(scale=20, size=count)
        death_td_ahead = pd.to_timedelta(death_years_ahead, unit='y')

        # 6. Set relevant properties of those with prevalent hypertension
        ht_years_ago = 1
        infected_td_ago = pd.to_timedelta(ht_years_ago * 365.25, unit='d')
        df.loc[df.is_alive & df.ht_current_status, 'ht_date'] = self.sim.date - infected_td_ago # TODO: check with Tim if we should make it more 'realistic'. Con: this still allows us  to check prevalent cases against data, no severity diff with t
        df.loc[df.is_alive & df.ht_current_status, 'ht_historic_status'] = 'C'
        df.loc[df.is_alive & df.ht_current_status, 'ht_death_date'] = self.sim.date + death_td_ahead

        print("\n", "Population has been initialised, prevalent cases have been assigned.  "
              "\n", "Prevalence of hypertension is: ", prevalence, "%")


    def initialise_simulation(self, sim):

        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        # 1. Add  basic event
        event = HTEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=1)) # ToDo: need to update this to adjust to time used for this method

        # 2. Add an event to log to screen
        sim.schedule_event(HypLoggingEvent(self), sim.date + DateOffset(months=6))

        # 3. Add shortcut to the data frame
        df = sim.population.props

        # add the death event of infected individuals # TODO: could be used for testing later
        # schedule the mockitis death event
        #people_who_will_die = df.index[df.mi_is_infected]
        #for person_id in people_who_will_die:
        #    self.sim.schedule_event(HypDeathEvent(self, person_id),
        #                            df.at[person_id, 'mi_scheduled_date_death'])

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

        # Schedule the outreach event... # ToDo: need to test this with HT!
        # event = HypOutreachEvent(self, 'this_module_only')
        # self.sim.schedule_event(event, self.sim.date + DateOffset(months=24))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the ID for the mother for this child
        :param child_id: the ID for the new child
        """

        df = self.sim.population.props
        df.at[child_id, 'ht_risk'] = 1.0                        # Default setting: no risk given pre-existing conditions
        df.at[child_id, 'ht_current_status'] = False            # Default setting: no one has hypertension
        df.at[child_id, 'ht_historic_status'] = 'N'             # Default setting: no one has hypertension
        df.at[child_id, 'ht_date'] = pd.NaT                     # Default setting: no one has hypertension
        df.at[child_id, 'ht_death_date'] = pd.NaT               # Default setting: no one has hypertension
        df.at[child_id, 'ht_diag_date'] = pd.NaT                # Default setting: no one is diagnosed
        df.at[child_id, 'ht_diag_status'] = 'N'                 # Default setting: no one is diagnosed
        df.at[child_id, 'ht_diag_date'] = pd.NaT                # Default setting: no one is treated
        df.at[child_id, 'ht_diag_status'] = 'N'                 # Default setting: no one is treated
        df.at[child_id, 'ht_contr_date'] = pd.NaT               # Default setting: no one is controlled
        df.at[child_id, 'ht_contr_status'] = 'N'                # Default setting: no one is controlled
        df.loc[df.is_alive, 'ht_specific_symptoms'] = 'none'    # Default setting: no one has symptoms
        df.loc[df.is_alive, 'ht_unified_symptom_code'] = 0      # Default setting: no one has symptoms

    def on_healthsystem_interaction(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug('This is Hypertension, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)



    def report_qaly_values(self):
        # This must send back a dataframe that reports on the HealthStates for all individuals over
        # the past year

        # logger.debug('This is mockitis reporting my health values')

        df = self.sim.population.props  # shortcut to population properties dataframe

        p = self.parameters

        health_values = df.loc[df.is_alive, 'ht_specific_symptoms'].map({
            'none': 0,
            'mild sneezing': p['qalywt_mild_sneezing'],
            'coughing and irritable': p['qalywt_coughing'],
            'extreme emergency': p['qalywt_advanced']
        })
        return health_values.loc[df.is_alive]


class HTEvent(RegularEvent, PopulationScopeEventMixin):

    """
    This event is occurring regularly at one monthly intervals and controls the infection process
    and onset of symptoms of Mockitis.
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1)) # TODO: change time scale if needed
        self.prob_HT_basic = module.parameters['prob_HT_basic']
        self.prob_HTgivenHC = module.parameters['prob_HTgivenHC']
        self.prob_HTgivenDiab = module.parameters['prob_HTgivenDiab']
        self.prob_HTgivenHIV = module.parameters['prob_HTgivenHIV']
        self.prob_success_treat = module.parameters['prob_success_treat']

        # ToDO: need to add code from original if it bugs.

    def apply(self, population):

        logger.debug('This is HypertenionEvent, tracking the disease progression of the population.')

        # 1. Basic variables
        df = population.props
        rng = self.module.rng

        ht_total = (df.is_alive & df.ht_current_status).sum()

        # 2. Get (and hold) index of people with and w/o hypertension
        currently_ht_yes = df[df.ht_current_status & df.is_alive].index
        currently_ht_no = df[~df.ht_current_status & df.is_alive].index
        alive_count = df.is_alive.sum()

        if df.is_alive.sum():
            prevalence = len(currently_ht_yes) / (
                len(currently_ht_yes) + len(currently_ht_no))
        else:
            prevalence = 0

        print("\n", "Time is: ", self.sim.date, "Prevalence is ]: ", prevalence)

        # 3. Handle new cases of hypertension
        # 3.1 First get relative risk
        ht_prob = df.loc[currently_ht_no, ['age_years', 'ht_risk']].reset_index().merge(HT_incidence,
                                            left_on=['age_years'], right_on=['age'], how='left').set_index(
                                            'person')['probability']
        # df.loc[df.is_alive & ~df.hc_current_status, 'ht_risk'] = self.prob_HT_basic  # Basic risk, no pre-existing conditions
        # df.loc[df.is_alive & df.hc_current_status, 'ht_risk'] = self.prob_HTgivenHC  # Risk if pre-existing high cholesterol
        assert len(currently_ht_no) == len(ht_prob)
        ht_prob = ht_prob * df.loc[currently_ht_no, 'ht_risk']
        random_numbers = rng.random_sample(size=len(ht_prob))
        now_hypertensive = (ht_prob > random_numbers)
        ht_idx = currently_ht_no[now_hypertensive]

        # 3.2. Calculate prevalence
        count = df.ht_current_status.sum()
        prevalence = (sum(count) / alive_count) * 100

        # 4. Assign level of symptoms
        level_of_symptoms = self.parameters['level_of_symptoms']
        symptoms = self.rng.choice(level_of_symptoms.level_of_symptoms,
                                   size=ht_idx,
                                   p=level_of_symptoms.probability)
        df.loc[ht_idx, 'ht_specific_symptoms'] = symptoms

        # 5. Set date of death
        death_years_ahead = np.random.exponential(scale=20, size=ht_idx)
        death_td_ahead = pd.to_timedelta(death_years_ahead, unit='y')

         # 3.3 If newly hypertensive
        df.loc[ht_idx, 'ht_current_status'] = True
        df.loc[ht_idx, 'ht_historic_status'] = 'C'
        df.loc[ht_idx, 'ht_date'] = self.sim.date
        df.loc[ht_idx, 'ht_death_date'] = self.sim.date + death_td_ahead



        print("\n", "Time is: ", self.sim.date, "New cases have been assigned.  "
              "\n", "Prevalence is: ", prevalence, "%")

        # 6. Determine if anyone with severe symptoms will seek care
        serious_symptoms = (df['is_alive']) & ((df['ht_specific_symptoms'] == 'extreme emergency') | (
            df['ht_specific_symptoms'] == 'coughing and irritiable'))

        seeks_care = pd.Series(data=False, index=df.loc[serious_symptoms].index)
        for i in df.index[serious_symptoms]:
            prob = self.sim.modules['HealthSystem'].get_prob_seek_care(i, symptom_code=4)
            seeks_care[i] = self.module.rng.rand() < prob

        if seeks_care.sum() > 0:
            for person_index in seeks_care.index[seeks_care == True]:
                logger.debug(
                    'This is HTN, scheduling Hyp_PresentsForCareWithSevereSymptoms for person %d',
                    person_index)
                event = HSI_Hyp_PresentsForCareWithSevereSymptoms(self.module, person_id=person_index)
                self.sim.modules['HealthSystem'].schedule_event(event,
                                                                priority=2,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(weeks=2)
                                                                )
            else:
                logger.debug(
                    'This is MockitisEvent, There is  no one with new severe symptoms so no new healthcare seeking')
        else:
            logger.debug('This is MockitisEvent, no one is newly infected.')


class HTDeathEvent(Event, IndividualScopeEventMixin):   # TODO: remove afterwards - no HT deaths
    """
    This is the death event for mockitis
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe

        # Apply checks to ensure that this death should occur
        if df.at[person_id, 'ht_current_status'] == 'C':
            # Fire the centralised death event:
            death = InstantaneousDeath(self.module, person_id, cause='HT')
            self.sim.schedule_event(death, self.sim.date)


# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# Health System Interaction Events

class HSI_Hyp_PresentsForCareWithSevereSymptoms(Event, IndividualScopeEventMixin):

    """
    This is a Health System Interaction Event.
    It is first appointment that someone has when they present to the healthcare system with the severe
    symptoms of Mockitis.
    If they are aged over 15, then a decision is taken to start treatment at the next appointment.
    If they are younger than 15, then another initial appointment is scheduled for then are 15 years old.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Hyp_PresentsForCareWithSevereSymptoms'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = self.sim.modules['HealthSystem'].get_blank_cons_footprint()
        self.ALERT_OTHER_DISEASES = []


    def apply(self, person_id):

        logger.debug('This is HSI_Hyp_PresentsForCareWithSevereSymptoms, a first appointment for person %d', person_id)

        df = self.sim.population.props  # shortcut to the dataframe

        if df.at[person_id, 'age_years'] >= 15:
            logger.debug(
                '...This is HSI_Hyp_PresentsForCareWithSevereSymptoms: there should now be treatment for person %d',
                person_id)
            event = HSI_Hyp_StartTreatment(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_event(event,
                                                            priority=2,
                                                            topen=self.sim.date,
                                                            tclose=None)

        else:
            logger.debug(
                '...This is HSI_Hyp_PresentsForCareWithSevereSymptoms: there will not be treatment for person %d',
                person_id)

            date_turns_15 = self.sim.date + DateOffset(years=np.ceil(15 - df.at[person_id, 'age_exact_years']))
            event = HSI_Hyp_PresentsForCareWithSevereSymptoms(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_event(event,
                                                            priority=2,
                                                            topen=date_turns_15,
                                                            tclose=date_turns_15 + DateOffset(months=12))


class HSI_Hyp_StartTreatment(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.

    It is appointment at which treatment for mockitiis is inititaed.

    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient appt
        the_appt_footprint['NewAdult'] = 1  # Plus, an amount of resources similar to an HIV initiation


        # Get the consumables required
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(consumables.loc[consumables[
                                                  'Intervention_Pkg'] == 'First line treatment for new TB cases for adults', 'Intervention_Pkg_Code'])[
            0]
        pkg_code2 = pd.unique(consumables.loc[consumables[
                                                  'Intervention_Pkg'] == 'MDR notification among previously treated patients', 'Intervention_Pkg_Code'])[
            0]

        item_code1 = \
        pd.unique(consumables.loc[consumables['Items'] == 'Ketamine hydrochloride 50mg/ml, 10ml', 'Item_Code'])[0]
        item_code2 = pd.unique(consumables.loc[consumables['Items'] == 'Underpants', 'Item_Code'])[0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1, pkg_code2],
            'Item_Code': [item_code1, item_code2]
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Mockitis_Treatment_Initiation'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        logger.debug('This is HSI_Mockitis_StartTreatment: initiating treatent for person %d', person_id)
        df = self.sim.population.props
        treatmentworks = self.module.rng.rand() < self.module.parameters['p_cure']

        if treatmentworks:
            df.at[person_id, 'mi_is_infected'] = False
            df.at[person_id, 'mi_status'] = 'P'

            # (in this we nullify the death event that has been scheduled.)
            df.at[person_id, 'mi_scheduled_date_death'] = pd.NaT

            df.at[person_id, 'mi_date_cure'] = self.sim.date
            df.at[person_id, 'mi_specific_symptoms'] = 'none'
            df.at[person_id, 'mi_unified_symptom_code'] = 0
            pass

        # Create a follow-up appointment
        target_date_for_followup_appt = self.sim.date + DateOffset(months=6)

        logger.debug('....This is HSI_Mockitis_StartTreatment: scheduling a follow-up appointment for person %d on date %s',
                     person_id, target_date_for_followup_appt)

        followup_appt = HSI_Mockitis_TreatmentMonitoring(self.module, person_id=person_id)

        # Request the heathsystem to have this follow-up appointment
        self.sim.modules['HealthSystem'].schedule_event(followup_appt,
                                                        priority=2,
                                                        topen=target_date_for_followup_appt,
                                                        tclose=target_date_for_followup_appt + DateOffset(weeks=2)
        )




class HSI_Mockitis_TreatmentMonitoring(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.

    It is appointment at which treatment for mockitiis is monitored.
    (In practise, nothing happens!)

    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient appt
        the_appt_footprint['NewAdult'] = 1  # Plus, an amount of resources similar to an HIV initiation


        # Get the consumables required
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code1 = pd.unique(consumables.loc[consumables[
                                                  'Intervention_Pkg'] == 'First line treatment for new TB cases for adults', 'Intervention_Pkg_Code'])[
            0]
        pkg_code2 = pd.unique(consumables.loc[consumables[
                                                  'Intervention_Pkg'] == 'MDR notification among previously treated patients', 'Intervention_Pkg_Code'])[
            0]

        item_code1 = \
        pd.unique(consumables.loc[consumables['Items'] == 'Ketamine hydrochloride 50mg/ml, 10ml', 'Item_Code'])[0]
        item_code2 = pd.unique(consumables.loc[consumables['Items'] == 'Underpants', 'Item_Code'])[0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code1, pkg_code2],
            'Item_Code': [item_code1, item_code2]
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Mockitis_TreatmentMonitoring'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ALERT_OTHER_DISEASES = ['*']

    def apply(self, person_id):

        # There is a follow-up appoint happening now but it has no real effect!

        # Create the next follow-up appointment....
        target_date_for_followup_appt = self.sim.date + DateOffset(months=6)

        logger.debug(
            '....This is HSI_Mockitis_StartTreatment: scheduling a follow-up appointment for person %d on date %s',
            person_id, target_date_for_followup_appt)

        followup_appt = HSI_Mockitis_TreatmentMonitoring(self.module, person_id=person_id)

        # Request the heathsystem to have this follow-up appointment
        self.sim.modules['HealthSystem'].schedule_event(followup_appt,
                                                        priority=2,
                                                        topen=target_date_for_followup_appt,
                                                        tclose=target_date_for_followup_appt + DateOffset(weeks=2))



# ---------------------------------------------------------------------------------



class HypLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """Produce a summmary of the numbers of people with respect to their 'mockitis status'
        """
        # run this event every month
        self.repeat = 6
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props

        infected_total = df.loc[df.is_alive, 'mi_is_infected'].sum()
        proportion_infected = infected_total / len(df)

        mask: pd.Series = (df.loc[df.is_alive, 'mi_date_infected'] >
                           self.sim.date - DateOffset(months=self.repeat))
        infected_in_last_month = mask.sum()
        mask = (df.loc[df.is_alive, 'mi_date_cure'] > self.sim.date - DateOffset(months=self.repeat))
        cured_in_last_month = mask.sum()

        counts = {'N': 0, 'T1': 0, 'T2': 0, 'P': 0}
        counts.update(df.loc[df.is_alive, 'mi_status'].value_counts().to_dict())

        logger.info('%s|summary|%s', self.sim.date,
                    {
                        'TotalInf': infected_total,
                        'PropInf': proportion_infected,
                        'PrevMonth': infected_in_last_month,
                        'Cured': cured_in_last_month,
                    })

        logger.info('%s|status_counts|%s', self.sim.date, counts)
