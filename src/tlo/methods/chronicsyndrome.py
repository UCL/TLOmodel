"""
A skeleton template for disease methods.
"""
import pandas as pd
import numpy as np

import tlo
from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent, IndividualScopeEventMixin, Event


class ChronicSyndrome(Module):
    """
    This is a dummy chronic disease

    All disease modules need to be implemented as a class inheriting from Module.
    They need to provide several methods which will be called by the simulation
    framework:
    * `read_parameters(data_folder)`
    * `initialise_population(population)`
    * `initialise_simulation(sim)`
    * `on_birth(mother, child)`
    """

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    PARAMETERS = {
        'p_acquisition': Parameter(
            Types.REAL, 'Probability that an uninfected individual becomes infected'),
        'level_of_symptoms': Parameter(
            Types.CATEGORICAL, 'Level of symptoms that the individual will have'),
        'p_cure': Parameter(
            Types.REAL, 'Probability that a treatment is succesful in curing the individual'),
        'initial_prevalence': Parameter(
            Types.REAL,'Prevalence of the disease in the initial population'),
        'prob_dev_severe_symptoms_per_year': Parameter(
            Types.REAL, 'Probability per year of severe symptoms developing'),
        'prob_severe_symptoms_seek_emergency_care': Parameter(
            Types.REAL, 'Probability that an individual will seak emergency care on developing extreme illneess')
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'cs_has_cs': Property(Types.BOOL, 'Current status of mockitis'),
        'cs_status': Property(Types.CATEGORICAL,
                              'Historical status: N=never; C=currently 2; P=previously',
                              categories=['N', 'C', 'P']),
        'cs_date_acquired': Property(Types.DATE, 'Date of latest infection'),
        'cs_scheduled_date_death': Property(Types.DATE, 'Date of scheduled death of infected individual'),
        'cs_date_cure': Property(Types.DATE, 'Date an infected individual was cured'),
        'cs_specific_symptoms': Property(Types.CATEGORICAL, 'Level of symptoms for chronic syndrome specifically',
                                categories=['none', 'extreme illness']),
        'cs_unified_symptom_code': Property(Types.CATEGORICAL, 'Level of symptoms on the standardised scale (governing health-care seeking): 0=None; 1=Mild; 2=Moderate; 3=Severe; 4=Extreme_Emergency',
                                categories=[0,1,2,3,4])
    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        For now, we are going to hard code them explicity
        """

        self.parameters['p_acquisition_per_year']=0.10
        self.parameters['p_cure']=0.10
        self.parameters['initial_prevalence']=0.30
        self.parameters['level_of_symptoms'] = pd.DataFrame(data={'level_of_symptoms':['none', 'extreme illness'], 'probability':[0.95,0.05]})
        self.parameters['prob_dev_severe_symptoms_per_year']=0.50
        self.parameters['prob_severe_symptoms_seek_emergency_care']=0.95

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        df = population.props  # a shortcut to the dataframe storing data for individiuals

        # Set default for properties
        df['cs_has_cs'] = False  # default: no individuals infected
        df['cs_status'].values[:] = 'N'  # default: never infected
        df['cs_date_acquired'] = pd.NaT  # default: not a time
        df['cs_scheduled_date_death'] = pd.NaT  # default: not a time
        df['cs_date_cure'] = pd.NaT  # default: not a time
        df['cs_specific_symptoms']='none'
        df['cs_unified_symptom_code']=0


        # randomly selected some individuals as infected
        df['cs_has_cs'] = np.random.choice([True, False], size=len(df), p=[self.parameters['initial_prevalence'], 1-self.parameters['initial_prevalence']])

        df.loc[df['cs_has_cs'] == True, 'cs_status']='C'

        # Assign time of infections and dates of scheduled death for all those infected
        # get all the infected individuals
        acquired_count = df.cs_has_cs.sum()

        # Assign level of symptoms
        symptoms = self.rng.choice(self.parameters['level_of_symptoms']['level_of_symptoms'], size=acquired_count, p=self.parameters['level_of_symptoms']['probability'])
        df.loc[df['cs_has_cs']==True,'cs_specific_symptoms']=symptoms

        # date acquired cs
        acquired_years_ago = np.random.exponential(scale=10, size=acquired_count)  # sample years in the past
        # pandas requires 'timedelta' type for date calculations
        acquired_td_ago = pd.to_timedelta(acquired_years_ago, unit='y')

        # date of death of the infected individuals (in the future)
        death_years_ahead = np.random.exponential(scale=20, size=acquired_count)
        death_td_ahead = pd.to_timedelta(death_years_ahead, unit='y')

        # set the properties of infected individuals
        df.loc[df.cs_has_cs, 'cs_date_infected'] = self.sim.date - acquired_td_ago
        df.loc[df.cs_has_cs, 'cs_scheduled_date_death'] = self.sim.date + death_td_ahead



    def initialise_simulation(self, sim):

        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        # add the basic event (we will implement below)
        event = ChronicSyndromeEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=1))

        # add an event to log to screen
        sim.schedule_event(ChronicSyndromeLoggingEvent(self), sim.date + DateOffset(months=6))

        # # add the death event of individuals with ChronicSyndrome
        df = sim.population.props  # a shortcut to the dataframe storing data for individiuals
        indicies_of_persons_who_will_die=df[df['mi_is_infected']==True].index
        for person_index in indicies_of_persons_who_will_die:
            death_event = ChronicSyndromeDeathEvent(self, person_index)
            self.sim.schedule_event(death_event, df.at[person_index,'mi_scheduled_date_death'])

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].Register_Disease_Module(self)


        # Register with the HealthSystem the treatment interventions that this module runs
        # and define the footprint that each intervention has on the common resources
        self.registered_string_for_treatment='ChronicSyndrome_Treatment'

        # Define the footprint for the intervention on the common resources
        footprint_for_treatment=pd.DataFrame(index=np.arange(1),data={
                                                            'Name':self.registered_string_for_treatment,
                                                            'Nurse_Time':30,
                                                            'Doctor_Time':200,
                                                            'Electricity':True,
                                                            'Water':True})

        self.sim.modules['HealthSystem'].Register_Interventions(footprint_for_treatment)


    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the ID for the mother for this child
        :param child_id: the ID for the new child
        """

        df=self.sim.population.props # shortcut to the population props dataframe

        # Initialise all the properties that this module looks after:
        df.at[child_id,'cs_has_cs'] = False
        df.at[child_id,'cs_status']= 'N'
        df.at[child_id,'cs_date_acquired'] = pd.NaT
        df.at[child_id,'cs_scheduled_date_death'] = pd.NaT
        df.at[child_id,'cs_date_cure'] = pd.NaT
        df.at[child_id,'cs_specific_symptoms'] = 'none'
        df.at[child_id,'cs_unified_symptom_code'] = 0

    def query_symptoms_now(self):
        # This is called by the health-care seeking module
        # All modules refresh the symptomology of persons at this time
        # And report it on the unified symptomology scale

        print("This is chronicsyndome, being asked to report unified symptomology")
        # print('Now being asked to update symptoms')
        # df['cs_unified_symptom_code']
        # df['cs_specific_symptoms']

        # Map the specific symptoms for this disease onto the unified coding scheme
        df=self.sim.population.props # shortcut to population properties dataframe

        df['cs_unified_symptom_code'] = df['cs_specific_symptoms'].map({
            'none':0,
            'extreme illness':4
        })

        return df['cs_unified_symptom_code']

    def on_first_healthsystem_interaction(self,person_id,cue_type):
        print('This is chronicsyndrome, being asked what to do at a health system appointment for person', person_id)

        # Queries whether treatment is allowable under global policy
        Allowable = True

        # Queries whether treatment is available locally
        Available= True

        if (Allowable and Available):
            # # Commission treatment for this individual
            event=ChronicSyndromeTreatmentEvent(self,person_id)
            self.sim.schedule_event(event, self.sim.date)
            pass

    def on_followup_healthsystem_interaction(self, person_id):
        print('This is a follow-up appointment. Nothing to do')

    def report_HealthValues(self):
        #This must send back a dataframe that reports on the HealthStates for all individuals over the past year

        print('This is chronicsyndrome reporting my health values')

        df=self.sim.population.props # shortcut to population properties dataframe

        HealthValues = df['cs_specific_symptoms'].map({
            'none':1,
            'extreme illness':0.2
        })

        return  HealthValues


class ChronicSyndromeEvent(RegularEvent, PopulationScopeEventMixin):

    # This event is occuring regularly at one monthly intervals

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        df = population.props

        # 1. get (and hold) index of currently infected and uninfected individuals
        currently_cs = df.index[df.cs_has_cs & df.is_alive]
        currently_notcs = df.index[~df.cs_has_cs & df.is_alive]

        # 2. handle new cases
        p_aq=self.module.parameters['p_acquisition_per_year']/12
        now_acquired = np.random.choice([True, False], size=len(currently_notcs),
                                        p=[p_aq, 1 - p_aq])

        # if any are new cases
        if now_acquired.sum():
            newcases_idx = currently_notcs[now_acquired]

            death_years_ahead = np.random.exponential(scale=20, size=now_acquired.sum())
            death_td_ahead = pd.to_timedelta(death_years_ahead, unit='y')
            symptoms = self.sim.rng.choice(self.module.parameters['level_of_symptoms']['level_of_symptoms'],
                                            size=now_acquired.sum(), p=self.module.parameters['level_of_symptoms']['probability'])

            df.loc[newcases_idx, 'cs_has_cs'] = True
            df.loc[newcases_idx, 'cs_status'] = 'C'
            df.loc[newcases_idx, 'cs_date_acquired'] = self.sim.date
            df.loc[newcases_idx, 'cs_scheduled_date_death'] = self.sim.date + death_td_ahead
            df.loc[newcases_idx, 'cs_date_cure'] = pd.NaT
            df.loc[newcases_idx, 'cs_specific_symptoms'] = symptoms
            df.loc[newcases_idx, 'cs_unified_symptom_code'] = 0

            # schedule death events for new cases
            for person_index in newcases_idx:
                death_event = ChronicSyndromeDeathEvent(self, person_index)
                self.sim.schedule_event(death_event, df.at[person_index, 'cs_scheduled_date_death'])


        # 3) Handle progression to severe symptoms
        currently_cs_and_not_severe_symptoms_idx=df.index[ (df['cs_has_cs']==True) & (df['is_alive']==True) & (df['cs_specific_symptoms']!='extreme illness') ]
        will_start_severe_symptoms= self.sim.rng.random_sample(size=len(currently_cs_and_not_severe_symptoms_idx)) < (self.module.parameters['prob_dev_severe_symptoms_per_year']/12)
        will_start_severe_symptoms_idx=currently_cs_and_not_severe_symptoms_idx[will_start_severe_symptoms]
        df.loc[will_start_severe_symptoms_idx,'cs_specific_symptoms']='extreme illness'


        # 4) With some probability, the new severe cases seek "Emergency care"...

        if len(will_start_severe_symptoms_idx)>0:
            will_seek_emergency_care= self.sim.rng.random_sample(size=len(will_start_severe_symptoms_idx)) < (self.module.parameters['prob_severe_symptoms_seek_emergency_care'])
            will_seek_emergency_care_idx=will_start_severe_symptoms_idx[will_seek_emergency_care]

            for person_index in will_seek_emergency_care_idx:
                event= tlo.methods.healthsystem.InteractionWithHealthSystem_Emergency(self.module,person_index)
                self.sim.schedule_event(event,self.sim.date)



class ChronicSyndromeDeathEvent(Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):


        df = self.sim.population.props # shortcut to the dataframe

        # Apply checks to ensure that this death should occur
        if df.at[person_id,'mi_status']=='C':
            # Fire the centralised death event:
            death = tlo.methods.demography.InstantaneousDeath(self.module, person_id, cause='ChronicSyndrome')
            self.sim.schedule_event(death, self.sim.date)



class ChronicSyndromeTreatmentEvent(Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        print("We are now ready to treat this person", person_id)

        df = self.sim.population.props
        treatmentworks = self.sim.rng.rand() < self.module.parameters['p_cure']

        if treatmentworks:
            df.at[person_id, 'cs_has_cs'] = False
            df.at[person_id, 'cs_status'] = 'P'
            df.at[person_id, 'cs_scheduled_date_death'] = pd.NaT # (in this we nullify the death event that has been scheduled.)
            df.at[person_id, 'cs_date_cure'] = self.sim.date
            df.at[person_id, 'cs_specific_symptoms'] = 'none'
            df.at[person_id, 'cs_unified_symptom_code'] = 0



class ChronicSyndromeLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """comments...
        """
        # run this event every month
        self.repeat = 6
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props

        hascs_total = (df['cs_has_cs'] & df['is_alive']).sum()
        proportion_infected = hascs_total / df['is_alive'].sum()

