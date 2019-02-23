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
            Types.REAL,'Prevalence of the disease in the initial population'
        )
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
        'cs_symptoms': Property(Types.CATEGORICAL, 'Level of symptoms for mockitiis specifically',
                                categories=['none', 'extreme illness'])
    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        For now, we are going to hard code them explicity
        """

        self.parameters['p_acquisition_per_year']=0.02
        self.parameters['p_cure']=0.10
        self.parameters['initial_prevalence']=0.08
        self.parameters['level_of_symptoms'] = pd.DataFrame(data={'level_of_symptoms':['none', 'extreme illness'], 'probability':[0.95,0.05]})


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
        df['cs_symptoms']='none'


        # randomly selected some individuals as infected
        df['cs_has_cs'] = np.random.choice([True, False], size=len(df), p=[self.parameters['initial_prevalence'], 1-self.parameters['initial_prevalence']])

        df.loc[df['cs_has_cs'] == True, 'cs_status']='C'

        # Assign time of infections and dates of scheduled death for all those infected
        # get all the infected individuals
        acquired_count = df.cs_has_cs.sum()

        # Assign level of symptoms
        symptoms = self.rng.choice(self.parameters['level_of_symptoms']['level_of_symptoms'], size=acquired_count, p=self.parameters['level_of_symptoms']['probability'])
        df.loc[df['cs_has_cs']==True,'cs_symptoms']=symptoms

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
        df.at[child_id,'cs_symptoms'] = 'none'



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


        # if any are infected
        if now_acquired.sum():
            newcases_idx = currently_notcs[now_acquired]

            death_years_ahead = np.random.exponential(scale=20, size=now_acquired.sum())
            death_td_ahead = pd.to_timedelta(death_years_ahead, unit='y')
            symptoms = self.sim.rng.choice(self.module.parameters['level_of_symptoms']['level_of_symptoms'],
                                            size=now_infected.sum(), p=self.module.parameters['level_of_symptoms']['probability'])

            df.loc[newcases_idx, 'cs_has_cs'] = True
            df.loc[newcases_idx, 'cs_status'] = 'C'
            df.loc[newcases_idx, 'cs_date_acquired'] = self.sim.date
            df.loc[newcases_idx, 'cs_scheduled_date_death'] = self.sim.date + death_td_ahead
            df.loc[newcases_idx, 'cs_date_cure'] = pd.NaT
            df.loc[newcases_idx, 'cs_symptoms'] = symptoms


            # schedule death events for new cases
            for person_index in newcases_idx:
                death_event = ChronicSyndromeDeathEvent(self, person_index)
                self.sim.schedule_event(death_event, df.at[person_index, 'cs_scheduled_date_death'])





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

