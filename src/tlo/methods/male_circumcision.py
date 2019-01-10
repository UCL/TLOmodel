"""
A skeleton template for disease methods.
"""
import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent


class male_circumcision(Module):
    """
    male circumcision, without health system links
    """

    def __init__(self, name=None, workbook_path=None):
        super().__init__(name)
        self.workbook_path = workbook_path
        self.store = {'Time': [], 'proportion_circumcised': [], 'recently_circumcised': []}

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    PARAMETERS = {
        'initial_circumcision': Parameter(Types.REAL, 'Prevalence of circumcision in the population at baseline'),
        'prob_circumcision': Parameter(Types.REAL, 'probability of circumcision in the eligible population'),
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'is_circumcised': Property(Types.BOOL, 'individual is circumcised'),
        'date_circumcised': Property(Types.DATE, 'Date of circumcision'),
    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we do nothing.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        params = self.parameters
        now = self.sim.date.year

        params['param_list'] = pd.read_excel(self.workbook_path,
                                             sheet_name='circumcision')
        # print('param_list', self.param_list.head())
        #
        # # params['initial_circumcision'] = self.param_list.loc[self.param_list.year == now, 'coverage'].values
        # params['initial_circumcision'] = 0.23
        # print('initial_circumcision', params['initial_circumcision'])

    def initialise_population(self, population):
        df = population.props
        now = self.sim.date.year

        df['is_circumcised'] = False  # default: no individuals circumcised
        df['date_circumcised'] = pd.NaT  # default: not a time

        self.parameters['initial_circumcision'] = self.param_list.loc[self.param_list.year == now, 'coverage'].values[0]
        # print('initial_circumcision', self.parameters['initial_circumcision'])

        # select all eligible uncircumcised men
        uncircum = df.index[df.is_alive & (df.age_years >= 15) & ~df.is_circumcised & (df.sex == 'M')]
        # print('uncircum', len(uncircum))

        # 2. baseline prevalence of circumcisions
        circum = np.random.choice([True, False], size=len(uncircum),
                                  p=[self.parameters['initial_circumcision'],
                                     1 - self.parameters['initial_circumcision']])
        # TODO: change to self.rng.choice

        # print('circum', circum.sum())

        # if any are infected
        if circum.sum():
            circum_idx = uncircum[circum]
            # print('circum_idx', len(circum_idx))

            df.loc[circum_idx, 'is_circumcised'] = True
            df.loc[circum_idx, 'date_circumcised'] = self.sim.date


    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """
        # add the basic event (we will implement below)
        event = CircumcisionEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=1))

        # add an event to log to screen
        sim.schedule_event(CircumcisionLoggingEvent(self), sim.date + DateOffset(months=1))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.
        """
        df = self.sim.population.props

        df.at[child_id, 'is_circumcised'] = False
        df.at[child_id, 'date_latent_tb'] = pd.NaT


class CircumcisionEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=12))

        # self.param_list = module.parameters['param_list']
        # print('param_list, ', self.param_list)

    def apply(self, population):
        now = self.sim.date
        df = population.props
        params = self.module.parameters

        prob_df = params['param_list']
        # print('test', prob_df)

        # probability of circumcision using reported coverage
        # print('year', now.year)
        prob_circumcision = prob_df.loc[prob_df.year == now.year, 'prob_circum'].values[0]

        # print('prob_circumcision', prob_circumcision)

        # get a list of random numbers between 0 and 1 for the whole population
        random_draw = self.sim.rng.random_sample(size=len(df))

        # probability of circumcision
        circumcision_index = df.index[
            (random_draw < prob_circumcision) & ~df.is_circumcised & df.is_alive & (df.age_years >= 10) & (
                df.age_years < 35) & (df.sex == 'M')]
        df.loc[circumcision_index, 'is_circumcised'] = True
        df.loc[circumcision_index, 'date_circumcised'] = now


class CircumcisionLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """comments...
        """
        # run this event every month
        self.repeat = 6
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props

        circumcised_total = len(df.index[df.is_alive & (df.age_years >= 15) & df.is_circumcised])
        proportion_circumcised = circumcised_total / len(
            df.index[df.is_alive & (df.age_years >= 15) & (df.sex == 'M')])

        mask = (df['date_circumcised'] > self.sim.date - DateOffset(months=self.repeat))
        circumcised_in_last_timestep = mask.sum()

        self.module.store['Time'].append(self.sim.date)
        self.module.store['proportion_circumcised'].append(proportion_circumcised)
        self.module.store['recently_circumcised'].append(circumcised_in_last_timestep)
