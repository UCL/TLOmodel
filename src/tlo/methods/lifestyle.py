"""
A skeleton template for disease methods.
"""

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent
import numpy as np
import pandas as pd

class Lifestyle(Module):
    """
    One line summary goes here...
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
        'r_urban': Parameter(Types.REAL, 'probability per 3 mths of change from rural to urban'),
        'r_rural': Parameter(Types.REAL, 'probability per 3 mths of change from urban to rural'),
        'r_overwt': Parameter(Types.REAL, 'probability per 3 mths of change from not_overwt to overwt if male'),
        'r_not_overwt': Parameter(Types.REAL, 'probability per 3 mths of change from overwt to not overwt'),
        'rr_overwt_f': Parameter(Types.REAL, 'risk ratio for becoming overwt if female rather than male'),
        'init_p_urban': Parameter(Types.REAL, 'proportion urban at baseline'),
        'init_p_wealth_urban': Parameter(Types.LIST, 'List of probabilities of category given urban'),
        'init_p_wealth_rural': Parameter(Types.LIST, 'List of probabilities of category given rural'),
        'init_p_overwt_f_rural_agege15': Parameter(Types.REAL, 'proportion overwt at baseline if female rural agege15'),
        'init_p_overwt_f_urban_agege15': Parameter(Types.REAL, 'proportion overwt at baseline if female urban agege15'),
        'init_p_overwt_m_rural_agege15': Parameter(Types.REAL, 'proportion overwt at baseline if male rural agege15'),
        'init_p_overwt_m_urban_agege15': Parameter(Types.REAL, 'proportion overwt at baseline if male urban agege15'),
    }
    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'li_urban': Property(Types.BOOL, 'Currently urban'),
        'li_date_trans_to_urban': Property(Types.DATE, 'date of transition to urban'),
        'li_wealth': Property(Types.CATEGORICAL, 'wealth level', categories=[1, 2, 3, 4, 5]),
        'li_overwt': Property(Types.BOOL, 'currently overweight'),
    }

    def __init__(self):
        super().__init__()
        self.store = {'alive': []}

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.
        Here we do nothing.
        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        self.parameters['r_urban'] = 0.000625
        self.parameters['r_rural'] = 0.00001
        self.parameters['r_overwt'] = 0.001
        self.parameters['r_not_overwt'] = 0.001
        self.parameters['rr_overwt_f'] = 1
        self.parameters['init_p_urban'] = 0.17
        self.parameters['init_p_wealth_urban'] = [0.75, 0.16, 0.05, 0.02, 0.02]
        self.parameters['init_p_wealth_rural'] = [0.11, 0.21, 0.22, 0.23, 0.23]
        self.parameters['init_p_overwt_agelt15'] = [0.0]
        self.parameters['init_p_overwt_f_rural_agege15'] = 0.17
        self.parameters['init_p_overwt_f_urban_agege15'] = 0.32
        self.parameters['init_p_overwt_m_rural_agege15'] = 0.27
        self.parameters['init_p_overwt_m_urban_agege15'] = 0.46





    def initialise_population(self, population):
        """Set our property values for the initial population.
        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.
        :param population: the population of individuals
        """
        df = population.props  # a shortcut to the data-frame storing data for individuals
        df['li_urban'] = False  # default: all individuals rural
        df['li_date_trans_to_urban'] = pd.NaT
        df['li_wealth'].values[:] = 3  # default: all individuals wealth 3
        df['li_overwt'] = False  # default: all not overwt

        #  this below calls the age dataframe / call age.years to get age in years
        age = population.age

        # randomly selected some individuals as urban
        initial_urban = self.parameters['init_p_urban']
        initial_rural = 1 - initial_urban
        df['li_urban'] = np.random.choice([True, False], size=len(df), p=[initial_urban, initial_rural])

        # get the indices of all individuals who are urban
        urban_index = df.index[df.li_urban]
        # randomly sample wealth category according to urban wealth probs and assign to urban ind.
        df.loc[urban_index, 'li_wealth'] = np.random.choice([1, 2, 3, 4, 5],
                                                            size=len(urban_index),
                                                            p=self.parameters['init_p_wealth_urban'])

        # get the indicies of all individual who are rural (i.e. not urban)
        rural_index = df.index[~df.li_urban]
        df.loc[rural_index, 'li_wealth'] = np.random.choice([1, 2, 3, 4, 5],
                                                            size=len(rural_index),
                                                            p=self.parameters['init_p_wealth_rural'])

        i_p_overwt_m_rural_agege15 = self.parameters['init_p_overwt_m_rural_agege15']
        i_p_not_overwt_m_rural_agege15 = 1 - i_p_overwt_m_rural_agege15
        i_p_overwt_m_urban_agege15 = self.parameters['init_p_overwt_m_urban_agege15']
        i_p_not_overwt_m_urban_agege15 = 1 - i_p_overwt_m_urban_agege15
        i_p_overwt_f_rural_agege15 = self.parameters['init_p_overwt_f_rural_agege15']
        i_p_not_overwt_f_rural_agege15 = 1 - i_p_overwt_f_rural_agege15
        i_p_overwt_f_urban_agege15 = self.parameters['init_p_overwt_f_urban_agege15']
        i_p_not_overwt_f_urban_agege15 = 1 - i_p_overwt_f_urban_agege15

        agelt15_index = df.index[age.years < 15]

        agege15_m_rural_index = df.index[(age.years >= 15) & (~df.li_urban) & (df.sex == 'M')]
        agege15_f_rural_index = df.index[(age.years >= 15) & (~df.li_urban) & (df.sex == 'F')]
        agege15_m_urban_index = df.index[(age.years >= 15) & df.li_urban & (df.sex == 'M')]
        agege15_f_urban_index = df.index[(age.years >= 15) & df.li_urban & (df.sex == 'F')]

        df.loc[agelt15_index, 'li_overwt'] = False

        df.loc[agege15_m_rural_index, 'li_overwt'] = np.random.choice([True, False], size=len(agege15_m_rural_index),
                                                                      p=[i_p_overwt_m_rural_agege15,
                                                                         i_p_not_overwt_m_rural_agege15])
        df.loc[agege15_m_urban_index, 'li_overwt'] = np.random.choice([True, False], size=len(agege15_m_urban_index),
                                                                      p=[i_p_overwt_m_urban_agege15,
                                                                         i_p_not_overwt_m_urban_agege15])
        df.loc[agege15_f_rural_index, 'li_overwt'] = np.random.choice([True, False], size=len(agege15_f_rural_index),
                                                                      p=[i_p_overwt_f_rural_agege15,
                                                                         i_p_not_overwt_f_rural_agege15])
        df.loc[agege15_f_urban_index, 'li_overwt'] = np.random.choice([True, False], size=len(agege15_f_urban_index),
                                                                      p=[i_p_overwt_f_urban_agege15,
                                                                         i_p_not_overwt_f_urban_agege15])

    def initialise_simulation(self, sim):
        """Get ready for simulation start.
        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """
        event = LifestyleEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=3))

        event = LifestylesLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=3))

    def on_birth(self, mother, child):
        """Initialise our properties for a newborn individual.
        This is called by the simulation whenever a new person is born.
        :param mother: the mother for this child
        :param child: the new child
        """
        pass


class LifestyleEvent(RegularEvent, PopulationScopeEventMixin):
    """A skeleton class for an event
    Regular events automatically reschedule themselves at a fixed frequency,
    and thus implement discrete timestep type behaviour. The frequency is
    specified when calling the base class constructor in our __init__ method.
    """
    def __init__(self, module):
        """One line summary here
        We need to pass the frequency at which we want to occur to the base class
        constructor using super(). We also pass the module that created this event,
        so that random number generators can be scoped per-module.
        :param module: the module that created this event
        """
        super().__init__(module, frequency=DateOffset(months=3))
        self.r_urban = module.parameters['r_urban']
        self.r_rural = module.parameters['r_rural']
        self.r_overwt = module.parameters['r_overwt']
        self.r_not_overwt = module.parameters['r_not_overwt']
        self.rr_overwt_f = module.parameters['rr_overwt_f']

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props

        # TODO: remove in live code!
        currently_alive = df[df.is_alive]
        people_to_die = currently_alive.sample(n=int(len(currently_alive) * 0.005)).index
        if len(people_to_die):
            df.loc[people_to_die, 'is_alive'] = False

        # 1. get (and hold) index of current urban rural status
        currently_rural = df.index[~df.li_urban & df.is_alive]
        currently_urban = df.index[df.li_urban & df.is_alive]

        # 2. handle new transitions
        now_urban = np.random.choice([True, False],
                                     size=len(currently_rural),
                                     p=[self.r_urban, 1 - self.r_urban])
        # if any have transitioned to urban
        if now_urban.sum():
            urban_idx = currently_rural[now_urban]
            df.loc[urban_idx, 'li_urban'] = True
            df.loc[urban_idx, 'li_date_trans_to_urban'] = self.sim.date

        # 3. handle new transitions to rural
        now_rural = np.random.choice([True, False], size=len(currently_urban), p=[self.r_rural, 1 - self.r_rural])
        # if any have transitioned to rural
        if now_rural.sum():
            rural_idx = currently_urban[now_rural]
            df.loc[rural_idx, 'li_urban'] = False

        # as above - transition between overwt and not overwt
        # transition to ovrwt depends on sex

        currently_not_overwt_f = df.index[~df.li_overwt & df.is_alive & (df.sex == 'F')]
        currently_not_overwt_m = df.index[~df.li_overwt & df.is_alive & (df.sex == 'M')]
        currently_overwt = df.index[df.li_overwt & df.is_alive]

        ri_overwt_f = self.r_overwt*self.rr_overwt_f
        ri_overwt_m = self.r_overwt

        now_overwt_f = np.random.choice([True, False],
                                        size=len(currently_not_overwt_f),
                                        p=[ri_overwt_f, 1 - ri_overwt_f])
        if now_overwt_f.sum():
            overwt_f_idx = currently_not_overwt_f[now_overwt_f]
            df.loc[overwt_f_idx, 'li_overwt'] = True

        now_overwt_m = np.random.choice([True, False],
                                        size=len(currently_not_overwt_m),
                                        p=[ri_overwt_m, 1 - ri_overwt_m])
        if now_overwt_m.sum():
            overwt_m_idx = currently_not_overwt_m[now_overwt_m]
            df.loc[overwt_m_idx, 'li_overwt'] = True

        now_not_overwt = np.random.choice([True, False], size=len(currently_overwt),
                                          p=[self.r_not_overwt, 1 - self.r_not_overwt])
        if now_not_overwt.sum():
            not_overwt_idx = currently_overwt[now_not_overwt]
            df.loc[not_overwt_idx, 'li_overwt'] = False


class LifestylesLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """comments...
        """
        # run this event every 3 month
        self.repeat = 3
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props

        urban_alive = (df.is_alive & df.li_urban).sum()
     #   self.module.store['urban_total'].append(urban_alive)
        alive = df.is_alive.sum()

        self.module.store['alive'].append(alive)

        proportion_urban = urban_alive / (df.is_alive.sum())
        rural_alive = (df.is_alive & (~df.li_urban)).sum()

        mask = (df['li_date_trans_to_urban'] > self.sim.date - DateOffset(months=self.repeat))
        newly_urban_in_last_3mths = mask.sum()

        wealth_count_alive = df.loc[df.is_alive, 'li_wealth'].value_counts()

        print('%s lifestyle urban total:%d , proportion_urban: %f , newly urban: %d, wealth: %s' %
              (self.sim.date, urban_alive, proportion_urban, newly_urban_in_last_3mths, list(wealth_count_alive)), flush=True)


