"""
A skeleton template for disease methods.
"""
import pandas as pd
import numpy as np

import tlo
from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent, IndividualScopeEventMixin, Event
from tlo.methods import healthsystem


class Mockitis(Module):
    """
    This is a dummy infectious disease

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
        'p_infection': Parameter(
            Types.REAL, 'Probability that an uninfected individual becomes infected'),
        'level_of_symptoms': Parameter(
            Types.CATEGORICAL, 'Level of symptoms that the individual will have'),
        'p_cure': Parameter(
            Types.REAL, 'Probability that a treatment is successful in curing the individual'),
        'initial_prevalence': Parameter(
            Types.REAL,'Prevalence of the disease in the initial population'
        )
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'mi_is_infected': Property(Types.BOOL, 'Current status of mockitis'),
        'mi_status': Property(Types.CATEGORICAL,
                              'Historical status: N=never; C=currently 2; P=previously',
                              categories=['N', 'C', 'P']),
        'mi_date_infected': Property(Types.DATE, 'Date of latest infection'),
        'mi_scheduled_date_death': Property(Types.DATE, 'Date of scheduled death of infected individual'),
        'mi_date_cure': Property(Types.DATE, 'Date an infected individual was cured'),
        'mi_specific_symptoms': Property(Types.CATEGORICAL, 'Level of symptoms for mockitiis specifically',
                                categories=['none', 'mild sneezing', 'coughing and irritable', 'extreme emergency']),
        'mi_unified_symptom_code': Property(Types.CATEGORICAL,
                                            'Level of symptoms on the standardised scale (governing health-care seeking): 0=None; 1=Mild; 2=Moderate; 3=Severe; 4=Extreme_Emergency',
                                            categories=[0, 1, 2, 3, 4])
    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        For now, we are going to hard code them explicity
        """

        self.parameters['p_infection']=0.001
        self.parameters['p_cure']=0.50
        self.parameters['initial_prevalence']=0.5
        self.parameters['level_of_symptoms'] = pd.DataFrame(data={'level_of_symptoms':['none','mild sneezing','coughing and irritable','extreme emergency'], 'probability':[0.25,0.25,0.25,0.25]})


    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        df = population.props  # a shortcut to the dataframe storing data for individiuals

        # Set default for properties
        df['mi_is_infected'] = False  # default: no individuals infected
        df['mi_status'].values[:] = 'N'  # default: never infected
        df['mi_date_infected'] = pd.NaT  # default: not a time
        df['mi_scheduled_date_death'] = pd.NaT  # default: not a time
        df['mi_date_cure'] = pd.NaT  # default: not a time
        df['mi_specific_symptoms']='none'
        df['mi_unified_symptom_code']=0

        # randomly selected some individuals as infected
        initial_infected = self.parameters['initial_prevalence']
        initial_uninfected = 1 - initial_infected
        df['mi_is_infected'] = np.random.choice([True, False], size=len(df), p=[initial_infected, initial_uninfected])

        df.loc[df['mi_is_infected'] == True, 'mi_status']='C'

        # Assign time of infections and dates of scheduled death for all those infected
        # get all the infected individuals
        infected_count = df.mi_is_infected.sum()

        # Assign level of symptoms
        symptoms= self.rng.choice(self.parameters['level_of_symptoms']['level_of_symptoms'], size=infected_count, p=self.parameters['level_of_symptoms']['probability'])
        df.loc[df['mi_is_infected']==True,'mi_specific_symptoms']=symptoms

        # date of infection of infected individuals
        infected_years_ago = np.random.exponential(scale=5, size=infected_count)  # sample years in the past
        # pandas requires 'timedelta' type for date calculations
        infected_td_ago = pd.to_timedelta(infected_years_ago, unit='y')

        # date of death of the infected individuals (in the future)
        death_years_ahead = np.random.exponential(scale=2, size=infected_count)
        death_td_ahead = pd.to_timedelta(death_years_ahead, unit='y')

        # set the properties of infected individuals
        df.loc[df.mi_is_infected, 'mi_date_infected'] = self.sim.date - infected_td_ago
        df.loc[df.mi_is_infected, 'mi_scheduled_date_death'] = self.sim.date + death_td_ahead



    def initialise_simulation(self, sim):

        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        # add the basic event (we will implement below)
        event = MockitisEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=1))

        # add an event to log to screen
        sim.schedule_event(MockitisLoggingEvent(self), sim.date + DateOffset(months=6))

        # # add the death event of infected individuals
        # schedule the mockitis death event

        df = sim.population.props  # a shortcut to the dataframe storing data for individiuals
        indicies_of_persons_who_will_die=df[df['mi_is_infected']==True].index
        for person_index in indicies_of_persons_who_will_die:
            death_event = MockitisDeathEvent(self, person_index)
            self.sim.schedule_event(death_event, df.at[person_index,'mi_scheduled_date_death'])

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].Register_Disease_Module(self)


    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the ID for the mother for this child
        :param child_id: the ID for the new child
        """

        df=self.sim.population.props # shortcut to the population props dataframe

        # Initialise all the properties that this module looks after:

        child_is_infected= df.at[mother_id, 'mi_is_infected'] # is infected if mother is infected

        if child_is_infected:

            # Scheduled death date
            death_years_ahead = np.random.exponential(scale=20)
            death_td_ahead = pd.to_timedelta(death_years_ahead, unit='y')

            # Level of symptoms
            symptoms = self.rng.choice(self.parameters['level_of_symptoms']['level_of_symptoms'],p=self.parameters['level_of_symptoms']['probability'])

            # Assign properties
            df.at[child_id, 'mi_is_infected'] = True
            df.at[child_id,'mi_status'] = 'C'
            df.at[child_id,'mi_date_infected'] = self.sim.date
            df.at[child_id,'mi_scheduled_date_death'] = self.sim.date + death_td_ahead
            df.at[child_id,'mi_date_cure'] = pd.NaT
            df.at[child_id,'mi_specific_symptoms'] = symptoms
            df.at[child_id,'mi_unified_symptom_code']=0

            # Schedule death event:
            death_event = MockitisDeathEvent(self, child_id)
            self.sim.schedule_event(death_event, df.at[child_id, 'mi_scheduled_date_death'])


        else:

            # Assign the default for a child who is not infected
            df.at[child_id,'mi_is_infected'] = False
            df.at[child_id,'mi_status'] = 'N'
            df.at[child_id,'mi_date_infected'] = pd.NaT
            df.at[child_id,'mi_scheduled_date_death'] = pd.NaT
            df.at[child_id,'mi_date_cure'] = pd.NaT
            df.at[child_id,'mi_specific_symptoms'] = 'none'
            df.at[child_id,'mi_unified_symptom_code']=0


    def query_symptoms_now(self):
        # This is called by the health-care seeking module
        # All modules refresh the symptomology of persons at this time
        # And report it on the unified symptomology scale
        print("This is mockitis, being asked to report unified symptomology")

        df=self.sim.population.props # shortcut to population properties dataframe

        df['mi_unified_symptom_code']=0
        #df['mi_specific_symptoms']

        return df['mi_unified_symptom_code']



class MockitisEvent(RegularEvent, PopulationScopeEventMixin):

    # This event is occuring regularly at one monthly intervals

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        df = population.props

        # 1. get (and hold) index of currently infected and uninfected individuals
        currently_infected = df.index[df.mi_is_infected & df.is_alive]
        currently_uninfected = df.index[~df.mi_is_infected & df.is_alive]
        prevalence = len(currently_infected)/ (len(currently_infected)+len(currently_uninfected))

        # 2. handle new infections
        now_infected = np.random.choice([True, False], size=len(currently_uninfected),
                                        p=[prevalence, 1 - prevalence])


        # if any are infected
        if now_infected.sum():
            infected_idx = currently_uninfected[now_infected]

            death_years_ahead = np.random.exponential(scale=2, size=now_infected.sum())
            death_td_ahead = pd.to_timedelta(death_years_ahead, unit='y')
            symptoms = self.sim.rng.choice(self.module.parameters['level_of_symptoms']['level_of_symptoms'],
                                            size=now_infected.sum(), p=self.module.parameters['level_of_symptoms']['probability'])


            df.loc[infected_idx, 'mi_is_infected'] = True
            df.loc[infected_idx, 'mi_status'] = 'C'
            df.loc[infected_idx, 'mi_date_infected'] = self.sim.date
            df.loc[infected_idx, 'mi_scheduled_date_death'] = self.sim.date + death_td_ahead
            df.loc[infected_idx, 'mi_date_cure'] = pd.NaT
            df.loc[infected_idx, 'mi_specific_symptoms'] = symptoms
            df.loc[infected_idx, 'mi_unified_symptom_code'] = 0

            # schedule death events for newly infected individuals
            for person_index in infected_idx:
                death_event = MockitisDeathEvent(self, person_index)
                self.sim.schedule_event(death_event, df.at[person_index, 'mi_scheduled_date_death'])


        #
        # # 3. handle cures
        # cured = np.random.choice([True, False], size=len(currently_infected), p=[self.p_cure, 1 - self.p_cure])
        #
        # if cured.sum():
        #     cured = currently_infected[cured]
        #     df.loc[cured, 'mi_is_infected'] = False
        #     df.loc[cured, 'mi_status'] = 'P'
        #     df.loc[cured, 'mi_date_death'] = pd.NaT
        #     df.loc[cured, 'mi_date_cure'] = self.sim.date


class MockitisDeathEvent(Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):


        df = self.sim.population.props # shortcut to the dataframe

        # Apply checks to ensure that this death should occur
        if df.at[person_id,'mi_status']=='C':
            # Fire the centralised death event:
            death = tlo.methods.demography.InstantaneousDeath(self.module, person_id, cause='Mockitis')
            self.sim.schedule_event(death, self.sim.date)








class MockitisLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """comments...
        """
        # run this event every month
        self.repeat = 6
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props

        infected_total = df.mi_is_infected.sum()
        proportion_infected = infected_total / len(df)

        mask = (df['mi_date_infected'] > self.sim.date - DateOffset(months=self.repeat))
        infected_in_last_month = mask.sum()
        mask = (df['mi_date_cure'] > self.sim.date - DateOffset(months=self.repeat))
        cured_in_last_month = mask.sum()

        counts = {'N': 0, 'T1': 0, 'T2': 0, 'P': 0}
        counts.update(df['mi_status'].value_counts().to_dict())
        status = 'Status: { N: %(N)d; T1: %(T1)d; T2: %(T2)d; P: %(P)d }' % counts

        print('%s - Mockitis: {TotInf: %d; PropInf: %.3f; PrevMonth: {Inf: %d; Cured: %d}; %s }' %
              (self.sim.date,
               infected_total,
               proportion_infected,
               infected_in_last_month,
               cured_in_last_month,
               status), flush=True)
