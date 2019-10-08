"""
This is the method for hypertension
Developed by Mikaela Smit, October 2018
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# TODO: Asif/Stef to check if file path is mac/window flexible
# TODO: circular declaration with diabetes to be addressed
# TODO: Update with BMI categories

def make_hypertension_age_range_lookup(min_age, max_age, range_size):
    """Generates and returns a dictionary mapping age (in years) to age range
    as per data for validation (i.e. 25-35, 35-45, etc until 65)
    """

    def chunks(items, n):
        """Takes a list and divides it into parts of size n"""
        for index in range(0, len(items), n):
            yield items[index:index + n]

    # split all the ages from min to limit (100 years) into 5 year ranges
    parts = chunks(range(min_age, max_age), range_size)

    # any ages >= 100 are in the '100+' category
    # TODO: would be good to have those younger than 25 in 25- instead or just other
    default_category = '%d+' % max_age
    lookup = defaultdict(lambda: default_category)

    # collect the possible ranges
    ranges = []

    # loop over each range and map all ages falling within the range to the range
    for part in parts:
        start = part.start
        end = part.stop - 1
        value = '%s-%s' % (start, end)
        ranges.append(value)
        for i in range(start, part.stop):
            lookup[i] = value

    ranges.append(default_category)
    return ranges, lookup


class Hypertension(Module):
    """
    This is the Hypertension Module.
    Version: September 2019
    The execution of all health systems related interactions for hypertension are controlled through this module.
    """

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        logger.info('----------------------------------------------------------------------')
        logger.info("Running hypertension.  ")
        logger.info('----------------------------------------------------------------------')

    htn_age_range_categories, htn_age_range_lookup = make_hypertension_age_range_lookup(25, 65, 10)

    # We should have 4 age range categories
    assert len(htn_age_range_categories) == 5

    PARAMETERS = {

        # Here we declare parameters for this module. Each parameter has a name, data type,
        # and longer description.

        # Define epidemiological parameters
        'HT_prevalence': Parameter(Types.REAL, 'HT prevalence'),
        'HT_incidence': Parameter(Types.REAL, 'HT incidence'),
        'prob_ht_basic': Parameter(Types.REAL, 'Basic HTN probability'),
        'prob_htgivenbmi': Parameter(Types.REAL, 'HTN probability given BMI'),
        'prob_htgivendiabetes': Parameter(Types.REAL, 'HTN probability given pre-existing diabetes'),
        'initial_prevalence_ht': Parameter(Types.REAL, 'Prevalence of hypertension as per data'),

        # Define health care parameters #ToDO: to add all the HSI parameters here later
        'dalywt_ht': Parameter(Types.REAL, 'DALY weighting for hypertension')
    }

    PROPERTIES = {

        # Next we declare the properties of individuals that this module provides.
        # Again each has a name, type and description.

        # Note that all properties must have a two letter prefix that identifies them to this module.

        # Define disease properties
        'ht_age_range': Property(Types.CATEGORICAL, 'The age range categories for hypertension validation use',
                                 categories=htn_age_range_categories),
        'ht_risk': Property(Types.REAL, 'HTN risk given pre-existing condition or risk factors'),
        'ht_date': Property(Types.DATE, 'Date of latest hypertensive episode'),
        'ht_current_status': Property(Types.BOOL, 'Current hypertension status'),
        'ht_historic_status': Property(Types.CATEGORICAL, 'Historical status: N=never; C=Current, P=Previous',
                                       categories=['N', 'C', 'P']),

        # Define health care properties  #ToDO: to add more if needed once HSi coded
        'ht_diagnosis_date': Property(Types.DATE, 'Date of latest hypertension diagnosis'),
        'ht_diagnosis_status': Property(Types.CATEGORICAL, 'Status: N=No; Y=Yes',
                                        categories=['N', 'C', 'P']),
        'ht_treatment_date': Property(Types.DATE, 'Date of latest hypertension treatment'),
        'ht_treatment_status': Property(Types.CATEGORICAL, 'Status: N=never; C=Current, P=Previous',
                                        categories=['N', 'C', 'P']),
        'ht_control_date': Property(Types.DATE, 'Date of latest hypertension control'),
        'ht_control_status': Property(Types.CATEGORICAL, 'Status: N=No; Y=Yes',
                                      categories=['N', 'C', 'P']),
    }

    def read_parameters(self, data_folder):
        """
        Reads in parameter values from file.

        Loads the 'Method_HT' file containing all parameter values for hypertension method

        :param data_folder: path of a folder supplied to the Simulation containing data files.
        """

        logger.debug('Hypertension method: reading in parameters.  ')

        # Reads in the parameters for hypertension from  file
        workbook = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_Method_HT.xlsx',
                                 sheet_name=None)

        p = self.parameters
        p['HT_prevalence'] = workbook['prevalence2018']  # reads in prevalence parameters
        p['HT_incidence'] = workbook['incidence2018_plus']  # reads in incidence parameters

        self.load_parameters_from_dataframe(workbook['parameters'])  # reads in parameters

        p['dalywt_ht'] = 0.0  # assigns DALYs as none assigned in the daly excel for hypertension

        HT_data = workbook['data']  # reads in data
        df = HT_data.set_index('index')  # sets index

        # TODO: Asif/Stef to see if we can do the below shorter.

        # Read in prevalence data from file and put into model df
        p['initial_prevalence_ht'] = pd.DataFrame({'prevalence': [df.at['b_all', 'value'], df.at['b_25_35', 'value'],
                                                               df.at['b_35_45', 'value'], df.at['b_45_55', 'value'],
                                                               df.at['b_55_65', 'value']],
                                                'min95ci': [df.at['b_all', 'min'], df.at['b_25_35', 'min'],
                                                            df.at['b_35_45', 'min'], df.at['b_45_55', 'min'],
                                                            df.at['b_55_65', 'min']],
                                                'max95ci': [df.at['b_all', 'max'], df.at['b_25_35', 'max'],
                                                            df.at['b_35_45', 'max'], df.at['b_45_55', 'max'],
                                                            df.at['b_55_65', 'max']]
                                                },
                                               index=['total', '25_to_35', '35_to_45', '45_to_55', '55_to_65'])

        p['initial_prevalence_ht'].loc[:, 'prevalence'] *= 100  # Convert data to percentage
        p['initial_prevalence_ht'].loc[:, 'min95ci'] *= 100  # Convert data to percentage
        p['initial_prevalence_ht'].loc[:, 'max95ci'] *= 100  # Convert data to percentage

        logger.debug("Hypertension method: finished reading in parameters.  ")

    def initialise_population(self, population):
        """
        Assigns default values and prevalent cases of hypertension for the initial population.

        This method is called by the simulation and is responsible for assigning default values for every individuals of
        those properties 'owned' by the module, i.e. those declared in the PROPERTIES dictionary above.
        It also assigns prevalent values of hypertension at the start of the model.

        :param population: the population of individuals
        """

        logger.debug('Hypertension method: initialising population.  ')

        # Define key variables
        df = population.props  # Shortcut to the data frame storing individual data
        m = self

        # Define default properties
        df.loc[df.is_alive, 'ht_risk'] = 1.0  # Default: no risk given pre-existing conditions
        df.loc[df.is_alive, 'ht_current_status'] = False  # Default: no one has hypertension
        df.loc[df.is_alive, 'ht_historic_status'] = 'N'  # Default: no one has hypertension
        df.loc[df.is_alive, 'ht_date'] = pd.NaT  # Default: no one has hypertension
        df.loc[df.is_alive, 'ht_diagnosis_date'] = pd.NaT  # Default: no one is diagnosed
        df.loc[df.is_alive, 'ht_diagnosis_status'] = 'N'  # Default: no one is diagnosed
        df.loc[df.is_alive, 'ht_treatment_date'] = pd.NaT  # Default: no one is treated
        df.loc[df.is_alive, 'ht_treatment_status'] = 'N'  # Default: no one is treated
        df.loc[df.is_alive, 'ht_control_date'] = pd.NaT  # Default: no one is controlled
        df.loc[df.is_alive, 'ht_control_status'] = 'N'  # Default: no one is controlled

        # Assign prevalence as per data
        # Get corresponding risk
        ht_probability = df.loc[df.is_alive, ['ht_risk', 'age_years']].merge(self.parameters['HT_prevalence'],
                                                                             left_on=['age_years'], right_on=['age'],
                                                                             how='left')['probability']
        # TODO: update with BMI once merged to master
        df.loc[df.is_alive & df.li_overwt, 'ht_risk'] *= m.prob_htgivenbmi
        # TODO: add diabetes risk after circular declaration has been fixed with Asif/Stef
        # df.loc[df.is_alive & df.d2_current_status, 'ht_risk'] *= self.prob_htgivendiabetes

        # Define key variables
        alive_count = df.is_alive.sum()
        assert alive_count == len(ht_probability)
        df.loc[df.is_alive, 'ht_age_range'] = df.loc[df.is_alive, 'age_years'].map(self.htn_age_range_lookup)

        # Assign prevalent cases of hypertension according to risk
        ht_probability = ht_probability * df.loc[df.is_alive, 'ht_risk']

        # TODO: remove duplication later. Asif and Stef: when the model runs with 1st random_number, numbers are not
        # TODO: [cont] random (younger and older age group is always very high. Even if seed is changed. When executing
        # TODO: [cont] a second time this is fine. Once fixed remove excess code up to below note.
        random_numbers = self.rng.random_sample(size=alive_count)
        random_numbers = self.rng.random_sample(size=alive_count)
        df.loc[df.is_alive, 'ht_current_status'] = (random_numbers < ht_probability)

        # Check random numbers      #TODO: Remove from here ...
        logger.debug('Lets generate the random number check for hypertension')

        pop_over_18 = df.index[(df.age_years >= 18) & (df.age_years < 25)]
        mean_randnum_18_25 = random_numbers[pop_over_18].mean()
        pop_over_25_35 = df.index[(df.age_years >= 25) & (df.age_years < 35)]
        mean_randnum_25_35 = random_numbers[pop_over_25_35].mean()
        pop_over_35_45 = df.index[(df.age_years >= 35) & (df.age_years < 45)]
        mean_randnum_35_45 = random_numbers[pop_over_35_45].mean()
        pop_over_45_55 = df.index[(df.age_years >= 45) & (df.age_years < 55)]
        mean_randnum_45_55 = random_numbers[pop_over_45_55].mean()
        pop_over_55 = df.index[df['age_years'] >= 55]
        mean_randnum_over55 = random_numbers[pop_over_55].mean()

        logger.debug('Lets generate the random number check for hypertension. ')
        logger.debug({
            'Mean rand num in 18-25 yo': mean_randnum_18_25,
            'Mean rand num_in 25-35 yo': mean_randnum_25_35,
            'Mean rand num_in 35-45 yo': mean_randnum_35_45,
            'Mean rand num_in 45-55 yo': mean_randnum_45_55,
            'Mean_rand num_in over 55 yo': mean_randnum_over55
        })

        assert 0.4 < mean_randnum_18_25 < 0.6
        assert 0.4 < mean_randnum_25_35 < 0.6
        assert 0.4 < mean_randnum_35_45 < 0.6
        assert 0.4 < mean_randnum_45_55 < 0.6
        assert 0.4 < mean_randnum_over55 < 0.6

        # TODO: ... to here

        # Assign relevant properties amongst those hypertensive
        count_ht_all = len(df[df.is_alive & df.ht_current_status])  # Count all people with hypertension
        ht_years_ago = np.array([1] * count_ht_all)
        infected_td_ago = pd.to_timedelta(ht_years_ago, unit='y')
        df.loc[df.is_alive & df.ht_current_status, 'ht_date'] = self.sim.date - infected_td_ago
        df.loc[df.is_alive & df.ht_current_status, 'ht_historic_status'] = 'C'

        # Register this disease module with the health system  #TODO: TO DO LATER
        # self.sim.modules['HealthSystem'].register_disease_module(self)

        logger.debug("Hypertension method: finished initialising population.  ")

    def initialise_simulation(self, sim):
        """
        Get ready for simulation start.

        This method is called just before the main simulation begins, and after all
        parameters have been read in and prevalent hypertension has been assigned.
        It is a good place to add initial events to the event queue.

        :param sim: simulation
        """

        logger.debug("Hypertension method: initialising simulation.  ")

        # Add  basic event
        event = HTEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(years=1))

        # Launch the repeating event that will store statistics about hypertension
        sim.schedule_event(HTLoggingEvent(self), sim.date + DateOffset(days=0))
        sim.schedule_event(HTLoggingValidationEvent(self), sim.date + DateOffset(days=0))

        # Schedule the outreach event  # ToDo: need to test this with HT!
        # outreach_event = HT_LaunchOutreachEvent(self)
        # self.sim.schedule_event(outreach_event, self.sim.date+36)

        logger.debug("Hypertension method: finished initialising simulation.  ")

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the ID for the mother for this child
        :param child_id: the ID for the new child
        """

        logger.debug("Hypertension method: on birth is being defined.  ")

        df = self.sim.population.props
        df.at[child_id, 'ht_risk'] = 1.0  # Default: no risk given pre-existing conditions
        df.at[child_id, 'ht_current_status'] = False  # Default: no one has hypertension
        df.at[child_id, 'ht_historic_status'] = 'N'  # Default: no one has hypertension
        df.at[child_id, 'ht_date'] = pd.NaT  # Default: no one has hypertension
        df.at[child_id, 'ht_diagnosis_date'] = pd.NaT  # Default: no one is diagnosed
        df.at[child_id, 'ht_diagnosis_status'] = 'N'  # Default: no one is diagnosed
        df.at[child_id, 'ht_treatment_date'] = pd.NaT  # Default: no one is treated
        df.at[child_id, 'ht_treatment_status'] = 'N'  # Default: no one is treated
        df.at[child_id, 'ht_control_date'] = pd.NaT  # Default: no one is controlled
        df.at[child_id, 'ht_control_status'] = 'N'  # Default: no one is controlled

        logger.debug("Hypertension method: finished defining on birth.  ")

    def on_healthsystem_interaction(self, person_id, treatment_id):  # TODO: update
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug('This is Hypertension, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

    def on_hsi_alert(self, person_id, treatment_id):  # TODO: update
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """
        pass

    def report_daly_values(self):  # TODO: update
        # This must send back a data frame that reports on the HealthStates for all individuals over
        # the past year

        logger.debug('Hypertension method: reporting health values')

        df = self.sim.population.props  # shortcut to population properties data frame

        health_values = df.loc[df.is_alive, 'ht_specific_symptoms'].map({
            'N': 0,
        })
        assert health_values >= 0

        health_values.name = 'HT Symptoms'  # label the cause of this disability

        return health_values.loc[df.is_alive]


class HTEvent(RegularEvent, PopulationScopeEventMixin):
    """
    This event is occurring regularly and controls the disease process of Hypertension.
    It assigns new cases of hypertension and defines all related variables (e.g. date of hypertension)
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(years=1))

        logger.debug('Hypertension method: This is a hypertension Event')

    def apply(self, population):
        logger.debug('Hypertension method: tracking the disease progression of the population.')

        # Basic variables
        df = population.props
        m = self.module
        rng = m.rng

        # Get (and hold) index of people with and w/o hypertension
        currently_ht_yes = df[df.ht_current_status & df.is_alive].index  # holds all people with hypertension
        currently_ht_no = df[~df.ht_current_status & df.is_alive].index  # hold all people w/o hypertension
        age_index = df.loc[currently_ht_no, ['age_years']]  # looks up age index of those w/o hypertension
        alive_count = df.is_alive.sum()  # counts all those with hypertension

        assert alive_count == len(currently_ht_yes) + len(currently_ht_no)  # checks that we didn't loose anyone

        # Handle new cases of hypertension
        # First get relative risk
        ht_probability = df.loc[currently_ht_no,
                                ['age_years', 'ht_risk']].reset_index().merge(self.module.parameters['HT_incidence'],
                                                                              left_on=['age_years'],
                                                                              right_on=['age'],
                                                                              how='left').set_index('person')[
            'probability']
        df['ht_risk'] = 1.0  # Reset risk for all people
        # TODO: update with BMI once merged to master
        df.loc[df.is_alive & df.li_overwt, 'ht_risk'] *= m.prob_htgivenbmi  # Adjust risk if overwt
        # TODO: add diabetes risk after circular declaration has been fixed with Asif/Stef
        # df.loc[df.is_alive & df.d2_current_status, 'ht_risk'] *= m.prob_htgivendiabetes

        ht_probability = ht_probability * df.loc[currently_ht_no, 'ht_risk']  # look up updated risk of hypertension
        random_numbers = rng.random_sample(size=len(ht_probability))  # get random numbers
        now_hypertensive = (ht_probability > random_numbers)  # assign new hypertensive cases as per incidence and risk
        ht_idx = currently_ht_no[now_hypertensive]  # hold id of those with new hypertension

        # Check random numbers      #TODO: Remove from here ... (not petty code - just for testing)
        # random_numbers_df = random_numbers, index = currently_ht_no
        randnum_holder = pd.DataFrame(data=random_numbers, index=currently_ht_no, columns=['nr'])

        pop_over_18 = age_index.index[(age_index.age_years >= 18) & (age_index.age_years < 25)]
        rand_18_25 = (randnum_holder.loc[pop_over_18]).mean()
        rand_18_25_mean = rand_18_25.mean()

        pop_over_25_35 = age_index.index[(age_index.age_years >= 25) & (age_index.age_years < 35)]
        rand_25_35 = (randnum_holder.loc[pop_over_25_35]).mean()
        rand_25_35_mean = rand_25_35.mean()

        pop_over_35_45 = age_index.index[(age_index.age_years >= 35) & (age_index.age_years < 45)]
        rand_35_45 = (randnum_holder.loc[pop_over_35_45]).mean()
        rand_35_45_mean = rand_35_45.mean()

        pop_over_45_55 = age_index.index[(age_index.age_years >= 45) & (age_index.age_years < 55)]
        rand_45_55 = (randnum_holder.loc[pop_over_45_55]).mean()
        rand_45_55_mean = rand_45_55.mean()

        pop_over_55 = age_index.index[(age_index.age_years >= 55)]
        rand_over55 = (randnum_holder.loc[pop_over_55]).mean()
        rand_over55_mean = rand_over55.mean()

        logger.debug('Lets generate the random number check for incident hypertension. ')
        logger.debug({
            'Mean rand num in 18-25 yo': rand_18_25,
            'Mean rand num_in 25-35 yo': rand_25_35,
            'Mean rand num_in 35-45 yo': rand_35_45,
            'Mean rand num_in 45-55 yo': rand_45_55,
            'Mean_rand num_in over 55 yo': rand_over55
        })

        assert 0.4 < rand_18_25_mean < 0.6
        assert 0.4 < rand_25_35_mean < 0.6
        assert 0.4 < rand_35_45_mean < 0.6
        assert 0.4 < rand_45_55_mean < 0.6
        assert 0.4 < rand_over55_mean < 0.6

        # TODO: ... to here

        # 3.2 Assign variables amongst those newly hypertensive
        df.loc[ht_idx, 'ht_current_status'] = True
        df.loc[ht_idx, 'ht_historic_status'] = 'C'
        df.loc[ht_idx, 'ht_date'] = self.sim.date


class HT_LaunchOutreachEvent(Event, PopulationScopeEventMixin):
    """
    This is the event that is run by Hypertension and it is the Outreach Event.
    It will now submit the individual HSI events that occur when each individual is met.
    (i.e. Any large campaign is composed of many individual outreach events).
    """

    def __init__(self, module):
        super().__init__(module)

    def apply(self, population):
        df = self.sim.population.props

        # Find the person_ids who are going to get the outreach
        gets_outreach = df.index[(df['is_alive']) & (df['sex'] == 'F')]
        # self.sim.rng.choice(gets_outreach,2)

        for person_id in gets_outreach:
            # make the outreach event (let this disease module be alerted about it, and also Mockitis)
            outreach_event_for_individual = HSI_HT_Outreach_Individual(self.module, person_id=person_id)

            self.sim.modules['HealthSystem'].schedule_hsi_event(outreach_event_for_individual,
                                                                priority=1,
                                                                topen=self.sim.date,
                                                                tclose=None)


# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# Health System Interaction Events

class HSI_HT_Outreach_Individual(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.

    This event can be used to simulate the occurrence of an 'outreach' intervention.

    NB. This needs to be created and run for each individual that benefits from the outreach campaign.

    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        logger.debug('Outreach event being created.')

        # Define the necessary information for an HSI
        # (These are blank when created; but these should be filled-in by the module that calls it)
        self.TREATMENT_ID = 'HT_Outreach_Individual'
        self.APPT_FOOTPRINT = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        self.APPT_FOOTPRINT['ConWithDCSA'] = 0.5  # outreach event takes small amount of time for DCSA
        self.CONS_FOOTPRINT = self.sim.modules['HealthSystem'].get_blank_cons_footprint()
        self.ACCEPTED_FACILITY_LEVELS = ['*']  # Can occur at any facility level
        self.ALERT_OTHER_DISEASES = ['*']

    def apply(self, person_id):
        logger.debug('Outreach event running now for person: %s', person_id)

        df = self.sim.population.props

        # Find the person_ids who are going to get the outreach
        if df.at[person_id, 'ht_current_status']:
            if df.at[person_id, 'is_alive']:
                referral_event_for_individual = HSI_HT_Refer_Individual(self.module, person_id=person_id)

            self.sim.modules['HealthSystem'].schedule_hsi_event(referral_event_for_individual,
                                                                priority=1,
                                                                topen=self.sim.date,
                                                                tclose=None)

        pass


class HSI_HT_Refer_Individual(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    It is first appointment that someone has when they present to the healthcare system with the severe
    symptoms of Hypertension.
    If they are aged over 15, then a decision is taken to start treatment at the next appointment.
    If they are younger than 15, then another initial appointment is scheduled for then are 15 years old.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'HSI_HT_Refer_Individual'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = self.sim.modules['HealthSystem'].get_blank_cons_footprint()
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):

        logger.debug('This is HSI_HT_PresentsForCareWithSevereSymptoms, a first appointment for person %d', person_id)

        df = self.sim.population.props  # shortcut to the dataframe

        if df.at[person_id, 'age_years'] >= 15:
            logger.debug(
                '...This is HSI_HT_PresentsForCareWithSevereSymptoms: there should now be treatment for person %d',
                person_id)
            event = HSI_HT_Refer_Individual(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_event(event,
                                                            priority=2,
                                                            topen=self.sim.date,
                                                            tclose=None)

        else:
            logger.debug(
                '...This is HSI_HT_PresentsForCareWithSevereSymptoms: there will not be treatment for person %d',
                person_id)

            date_turns_15 = self.sim.date + DateOffset(years=np.ceil(15 - df.at[person_id, 'age_exact_years']))
            event = HSI_HT_Refer_Individual(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_event(event,
                                                            priority=2,
                                                            topen=date_turns_15,
                                                            tclose=date_turns_15 + DateOffset(months=12))


# #class HSI_HT_StartTreatment(Event, IndividualScopeEventMixin):
#     """
#     This is a Health System Interaction Event.
#
#     It is appointment at which treatment for hypertension is inititaed.
#
#     """
#
#     def __init__(self, module, person_id):
#         super().__init__(module, person_id=person_id)
#
#         # Get a blank footprint and then edit to define call on resources of this treatment event
#         the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
#         the_appt_footprint['Over5OPD'] = 1  # This requires one out patient appt
#         the_appt_footprint['NewAdult'] = 1  # Plus, an amount of resources similar to an HIV initiation
#
#
#         # Get the consumables required
#         consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
#         pkg_code1 = pd.unique(consumables.loc[consumables['Intervention_Pkg'] ==
# 'First line treatment for new TB cases for adults', 'Intervention_Pkg_Code'])[
#             0]
#         pkg_code2 = pd.unique(consumables.loc[consumables[
#                                                   'Intervention_Pkg'] ==
# 'MDR notification among previously treated patients', 'Intervention_Pkg_Code'])[
#             0]
#
#         item_code1 = \
#         pd.unique(consumables.loc[consumables['Items'] == 'Ketamine hydrochloride 50mg/ml, 10ml', 'Item_Code'])[0]
#         item_code2 = pd.unique(consumables.loc[consumables['Items'] == 'Underpants', 'Item_Code'])[0]
#
#         the_cons_footprint = {
#             'Intervention_Package_Code': [pkg_code1, pkg_code2],
#             'Item_Code': [item_code1, item_code2]
#         }
#
#         # Define the necessary information for an HSI
#         self.TREATMENT_ID = 'HT_Treatment_Initiation'
#         self.APPT_FOOTPRINT = the_appt_footprint
#         self.CONS_FOOTPRINT = the_cons_footprint
#         self.ALERT_OTHER_DISEASES = []
#
#     def apply(self, person_id):
#         logger.debug('This is HSI_HT_StartTreatment: initiating treatent for person %d', person_id)
#         df = self.sim.population.props
#         treatmentworks = self.module.rng.rand() < self.module.parameters['p_cure']
#
#         if treatmentworks:
#             df.at[person_id, 'mi_is_infected'] = False
#             df.at[person_id, 'mi_status'] = 'P'
#
#             # (in this we nullify the death event that has been scheduled.)
#             df.at[person_id, 'mi_scheduled_date_death'] = pd.NaT
#
#             df.at[person_id, 'mi_date_cure'] = self.sim.date
#             df.at[person_id, 'mi_specific_symptoms'] = 'none'
#             df.at[person_id, 'mi_unified_symptom_code'] = 0
#             pass
#
#         # Create a follow-up appointment
#         target_date_for_followup_appt = self.sim.date + DateOffset(months=6)
#
#         logger.debug('....This is HSI_HT_StartTreatment: scheduling a follow-up appointment for person %d on date %s',
#                      person_id, target_date_for_followup_appt)
#
#         followup_appt = HSI_HT_TreatmentMonitoring(self.module, person_id=person_id)
#
#         # Request the heathsystem to have this follow-up appointment
#         self.sim.modules['HealthSystem'].schedule_event(followup_appt,
#                                                         priority=2,
#                                                         topen=target_date_for_followup_appt,
#                                                         tclose=target_date_for_followup_appt + DateOffset(weeks=2)
#         )
#
#
#
#
# class HSI_HT_TreatmentMonitoring(Event, IndividualScopeEventMixin):
#     """
#     This is a Health System Interaction Event.
#
#     It is appointment at which treatment for hypertension is monitored.
#     (In practise, nothing happens!)
#
#     """
#
#     def __init__(self, module, person_id):
#         super().__init__(module, person_id=person_id)
#
#         # Get a blank footprint and then edit to define call on resources of this treatment event
#         the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
#         the_appt_footprint['Over5OPD'] = 1  # This requires one out patient appt
#         the_appt_footprint['NewAdult'] = 1  # Plus, an amount of resources similar to an HIV initiation
#
#
#         # Get the consumables required
#         consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
#         pkg_code1 = pd.unique(consumables.loc[consumables['Intervention_Pkg'] ==
# 'First line treatment for new TB cases for adults', 'Intervention_Pkg_Code'])[
#             0]
#         pkg_code2 = pd.unique(consumables.loc[consumables[
#                                                   'Intervention_Pkg'] ==
# 'MDR notification among previously treated patients', 'Intervention_Pkg_Code'])[
#             0]
#
#         item_code1 = \
#         pd.unique(consumables.loc[consumables['Items'] == 'Ketamine hydrochloride 50mg/ml, 10ml', 'Item_Code'])[0]
#         item_code2 = pd.unique(consumables.loc[consumables['Items'] == 'Underpants', 'Item_Code'])[0]
#
#         the_cons_footprint = {
#             'Intervention_Package_Code': [pkg_code1, pkg_code2],
#             'Item_Code': [item_code1, item_code2]
#         }
#
#         # Define the necessary information for an HSI
#         self.TREATMENT_ID = 'HT_TreatmentMonitoring'
#         self.APPT_FOOTPRINT = the_appt_footprint
#         self.CONS_FOOTPRINT = the_cons_footprint
#         self.ALERT_OTHER_DISEASES = ['*']
#
#     def apply(self, person_id):
#
#         # There is a follow-up appoint happening now but it has no real effect!
#
#         # Create the next follow-up appointment....
#         target_date_for_followup_appt = self.sim.date + DateOffset(months=6)
#
#         logger.debug(
#             '....This is HSI_HT_StartTreatment: scheduling a follow-up appointment for person %d on date %s',
#             person_id, target_date_for_followup_appt)
#
#         followup_appt = HSI_HT_TreatmentMonitoring(self.module, person_id=person_id)
#
#         # Request the heathsystem to have this follow-up appointment
#         self.sim.modules['HealthSystem'].schedule_event(followup_appt,
#                                                         priority=2,
#                                                         topen=target_date_for_followup_appt,
#                                                         tclose=target_date_for_followup_appt + DateOffset(weeks=2))
#
#

# ---------------------------------------------------------------------------------


class HTLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """This Logging event logs the model prevalence for analysis purposed.
        This Logging event is different to the validation  logging event as it logs prevalence by standard age groups
        used throughout the model. The validation logging event logs prevalence generated by the model to the age
        groups used in the data to ensure the model is running well.
        """

        self.repeat = 12  # run this event every year
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # This code will calculate the prevalence and log it

        df = population.props

        # 1 Calculate prevalence
        count_by_age = df[df.is_alive].groupby('age_range').size()
        count_ht_by_age = df[df.is_alive & (df.ht_current_status)].groupby('age_range').size()
        prevalence_ht_by_age = (count_ht_by_age / count_by_age) * 100
        prevalence_ht_by_age.fillna(0, inplace=True)

        # 2 Log prevalence
        logger.info('%s|ht_prevalence|%s', self.sim.date, prevalence_ht_by_age.to_dict())


class HTLoggingValidationEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """This Logging event logs the data and the model prevalence for validation purposed.
        This Logging event is different than the regular model logging event as it logs prevalence by age groups to
        match those reported in the data. The regular logging event logs prevalence generated by the model to the age
        groups used throughout the model.
        """

        self.repeat = 12  # run this event every year # TODO: only run this for the first 5 years for validation
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # This code logs the prevalence for the first 5 years from the data and the model.
        # Assumption is that Hypertension prevalence will remain stable over 5 years and thus equal to 2010 data

        df = population.props
        p = self.module.parameters

        # Log the data every year (for plotting)
        # TODO: Can this codebelow be shorten (i.e. read in the whole frame?
        # TODO: Can this be logged only first year
        logger.info('%s|test|%s',
                    self.sim.date,
                    {
                        'total': p['initial_prevalence_ht'].loc['total', 'prevalence']})

        logger.info('%s|ht_prevalence_data_validation|%s',
                    self.sim.date,
                    {
                        'total': p['initial_prevalence_ht'].loc['total', 'prevalence'],
                        'total_min': p['initial_prevalence_ht'].loc['total', 'min95ci'],
                        'total_max': p['initial_prevalence_ht'].loc['total', 'max95ci'],
                        'age25to35': p['initial_prevalence_ht'].loc['25_to_35', 'prevalence'],
                        'age25to35_min': p['initial_prevalence_ht'].loc['25_to_35', 'min95ci'],
                        'age25to35_max': p['initial_prevalence_ht'].loc['25_to_35', 'max95ci'],
                        'age35to45': p['initial_prevalence_ht'].loc['35_to_45', 'prevalence'],
                        'age35to45_min': p['initial_prevalence_ht'].loc['35_to_45', 'min95ci'],
                        'age35to45_max': p['initial_prevalence_ht'].loc['35_to_45', 'max95ci'],
                        'age45to55': p['initial_prevalence_ht'].loc['45_to_55', 'prevalence'],
                        'age45to55_min': p['initial_prevalence_ht'].loc['45_to_55', 'min95ci'],
                        'age45to55_max': p['initial_prevalence_ht'].loc['45_to_55', 'max95ci'],
                        'age55to65': p['initial_prevalence_ht'].loc['55_to_65', 'prevalence'],
                        'age55to65_min': p['initial_prevalence_ht'].loc['55_to_65', 'min95ci'],
                        'age55to65_max': p['initial_prevalence_ht'].loc['55_to_65', 'max95ci']
                    })

        # 3.3. Calculate prevalence from the model     #TODO: use groupby (make new cats) -  remove after check
        adults_count_all = len(df[df.is_alive & (df.age_years > 24) & (df.age_years < 65)])
        adults_count_25to35 = len(df[df.is_alive & (df.age_years > 24) & (df.age_years < 35)])
        adults_count_35to45 = len(df[df.is_alive & (df.age_years > 34) & (df.age_years < 45)])
        adults_count_45to55 = len(df[df.is_alive & (df.age_years > 44) & (df.age_years < 55)])
        adults_count_55to65 = len(df[df.is_alive & (df.age_years > 54) & (df.age_years < 65)])

        assert adults_count_all == adults_count_25to35 + adults_count_35to45 + adults_count_45to55 + adults_count_55to65

        # Count adults with hypertension by age group
        count_ht = len(df[df.is_alive & df.ht_current_status & (df.age_years > 24) & (df.age_years < 65)])
        count_ht_25to35 = len(df[df.is_alive & df.ht_current_status & (df.age_years > 24) & (df.age_years < 35)])
        count_ht_35to45 = len(df[df.is_alive & df.ht_current_status & (df.age_years > 34) & (df.age_years < 45)])
        count_ht_45to55 = len(df[df.is_alive & df.ht_current_status & (df.age_years > 44) & (df.age_years < 55)])
        count_ht_55to65 = len(df[df.is_alive & df.ht_current_status & (df.age_years > 54) & (df.age_years < 65)])

        assert count_ht == count_ht_25to35 + count_ht_35to45 + count_ht_45to55 + count_ht_55to65

        # Calculate overall and age-specific prevalence
        prevalence_overall = (count_ht / adults_count_all) * 100
        prevalence_25to35 = (count_ht_25to35 / adults_count_25to35) * 100
        prevalence_35to45 = (count_ht_35to45 / adults_count_35to45) * 100
        prevalence_45to55 = (count_ht_45to55 / adults_count_45to55) * 100
        prevalence_55to65 = (count_ht_55to65 / adults_count_55to65) * 100

        assert prevalence_overall > 0
        assert prevalence_25to35 > 0
        assert prevalence_35to45 > 0
        assert prevalence_45to55 > 0
        assert prevalence_55to65 > 0

        # 3.4 Log prevalence from the model
        logger.info('%s|ht_prevalence_model_validation|%s',
                    self.sim.date,
                    {
                        'total': prevalence_overall,
                        '25to35': prevalence_25to35,
                        '35to45': prevalence_35to45,
                        '45to55': prevalence_45to55,
                        '55to65': prevalence_55to65
                    })

        # TODO: remove up to here

        # 3.3. Calculate prevalence from the model using groupby instead
        # TODO: use groupby (make new cats) below instead - check it is working

        # First by age
        count_by_age_val = df[df.is_alive].groupby('ht_age_range').size()
        count_ht_by_age_val = df[df.is_alive & df.ht_current_status].groupby('ht_age_range').size()
        prevalence_ht_by_age_val = (count_ht_by_age_val / count_by_age_val) * 100
        prevalence_ht_by_age_val.fillna(0, inplace=True)

        # Then overall
        prevalence_ht_all_val = count_ht_by_age_val[0:4].sum() / count_by_age_val[0:4].sum() * 100

        # 3.4 Log prevalence
        logger.info('%s|ht_prevalence_model_validation_2|%s', self.sim.date,
                    {'total': prevalence_ht_all_val,
                     '25to35': prevalence_ht_by_age_val.iloc[0],
                     '35to45': prevalence_ht_by_age_val.iloc[1],
                     '45to55': prevalence_ht_by_age_val.iloc[2],
                     '55to65': prevalence_ht_by_age_val.iloc[3]
                     })
