"""
Behavioural intervention for hiv prevention / reduction of transmission
#  TODO: proportion counselled is greater than 1 by 2012
"""
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent


class BehaviourChange(Module):
    """
    intervention - changing behaviour to moderate risk of acquiring hiv
    prevention counselling - e.g. HIV disclosure, sexual risk reduction, alcohol reduction
    family planning education / counselling
    mass media programmes
    - WHO https://www.who.int/bulletin/volumes/88/8/09-068213/en/

    """
    def __init__(self, name=None):
        super().__init__(name)
        self.store = {'Time': [], 'Proportion_hiv_counselled': [], 'Number_recently_counselled': []}

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    PARAMETERS = {
        'p_behaviour': Parameter(Types.REAL, 'Probability that an individual is exposed to behavioural interventions ')
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'behaviour_change': Property(Types.BOOL, 'Exposed to hiv prevention counselling'),
        'date_behaviour_change': Property(Types.DATE, 'date of behavioural counselling')
    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we do nothing.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        params = self.parameters
        params['p_behaviour'] = 0.01

    def initialise_population(self, population):
        pass

    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """
        # add the basic event (we will implement below)
        event = BehaviourChangeEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=1))

        # add an event to log to screen
        sim.schedule_event(BehaviourChangeLoggingEvent(self), sim.date + DateOffset(months=1))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.
        """
        df = self.sim.population.props

        df.at[child_id, 'behaviour_change'] = False
        df.at[child_id, 'date_behaviour_change'] = pd.NaT


class BehaviourChangeEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        self.p_behaviour = module.parameters['p_behaviour']

    def apply(self, population):
        params = self.module.parameters
        now = self.sim.date
        df = population.props

        # get a list of random numbers between 0 and 1 for the whole population
        random_draw = self.sim.rng.random_sample(size=len(df))

        # probability of counselling
        counselling_index = df.index[
            (random_draw < params['p_behaviour']) & ~df.behaviour_change & df.is_alive
            & (df.age_years >= 15)]

        df.loc[counselling_index, 'behaviour_change'] = True
        df.loc[counselling_index, 'date_behaviour_change'] = now


class BehaviourChangeLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """comments...
        """
        # run this event every month
        self.repeat = 6
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props

        total_counselled = len(df[df.is_alive & (df.age_years >= 15) & df.behaviour_change])
        proportion_exposed = total_counselled / len(df[df.is_alive & (df.age_years >= 15)])

        mask = (df['date_behaviour_change'] > self.sim.date - DateOffset(months=self.repeat))
        counselling_in_last_month = mask.sum()

        self.module.store['Time'].append(self.sim.date)
        self.module.store['Proportion_hiv_counselled'].append(proportion_exposed)
        self.module.store['Number_recently_counselled'].append(counselling_in_last_month)
