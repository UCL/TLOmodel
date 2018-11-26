"""
This is the method for diabetes
Developed by Mikaela Smit, October 2018

"""

import pandas as pd
import numpy as np

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent, Event, IndividualScopeEventMixin

# Read in data
file_path = '/Users/mc1405/Dropbox/Projects - ongoing/Malawi Project/Thanzi la Onse/04 - Methods Repository/Method_Diabetes.xlsx'
method_diab_data = pd.read_excel(file_path, sheet_name=None, header=0)
Diab_prevalence, Diab_incidence, Diab_treatment = method_diab_data['prevalence2018'], method_diab_data['incidence2018_plus'], \
                                            method_diab_data['treatment_parameters']


class Diabetes(Module):
    """
    This is the diabetes module

    All disease modules need to be implemented as a class inheriting from Module.
    They need to provide several methods which will be called by the simulation
    framework:
    * `read_parameters(data_folder)`
    * `initialise_population(population)`
    * `initialise_simulation(sim)`
    * `on_birth(mother, child)`

    """

    # Here we declare parameters for this module. Each parameter has a name, data type,
    PARAMETERS = {
        # Insert if relevant
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'diab_current_status': Property(Types.BOOL, 'Current diabetes status'),
        'diab_historic_status': Property(Types.CATEGORICAL,
                                       'Historical status: N=never; C=Current, P=Previous',
                                       categories=['N', 'C', 'P']),
        'diab_date_case': Property(Types.DATE, 'Date of latest diabetes'),
        'diab_treatment_status': Property(Types.CATEGORICAL,
                                        'Historical status: N=never; C=Current, P=Previous',
                                        categories=['N', 'C', 'P']),
        'diab_date_treatment': Property(Types.DATE, 'Date of latest diabetes treatment'),
    }

    def read_parameters(self, data_folder):
        """Will need to update this to read parameter values from file.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        # 1. Define key variables
        df = population.props  # a shortcut to the dataframe(df) storing data on individuals
        now = self.sim.date

        # 2. Set default values for all variables to be initialised
        df['diab_current_status'] = False  # Default setting: no one has diabetes
        df['diab_historic_status'] = 'N'  # Default setting: no one has diabetes
        df['diab_date_case'] = pd.NaT  # Default setting: no one has a date for diabetes
        df['diab_treatment_status'] = 'N'  # Default setting: no one is treated
        df['diab_date_treatment'] = pd.NaT  # Details setting: no one has a date of treatment

        # 3. Assign prevalence as per data, by using probability by age
        joined = pd.merge(population.age, Diab_prevalence, left_on=['years'], right_on=['age'], how='left')
        random_numbers = np.random.rand(len(df))
        df['diab_current_status'] = (joined.probability > random_numbers)

        # 3.1 Ways to check what's happening
        # temp = pd.merge(population.age, df, left_index=True, right_index=True, how='inner')
        # temp = pd.DataFrame([population.age.years, joined.Proportion, random_numbers, df['diab_current_status']])

        # 4. Count all individuals by status at the start
        diabetes_count = df.diab_current_status.sum()
        pop_count = len(df.diab_current_status)

        # 5. Set date of diabetes amongst those with prevalent cases
        diab_years_ago = np.random.exponential(scale=5, size=diabetes_count)
        infected_td_ago = pd.to_timedelta(diab_years_ago, unit='y')

        # 5.1 Set the properties of those with prevalent diabetes
        df.loc[df.diab_current_status, 'diab_date_case'] = self.sim.date - infected_td_ago
        df.loc[df.diab_current_status, 'diab_historic_status'] = 'C'

        print("\n", "Population has been initialised, prevalent cases have been assigned.  ")

    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """
        # Add the basic event (implement below)
        event = DiabetesEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=1))

        # Add an event to log to screen
        sim.schedule_event(DiabetesLoggingEvent(self), sim.date + DateOffset(months=6))

    def on_birth(self, mother, child):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother: the mother for this child
        :param child: the new child
        """
        pass


class DiabetesEvent(RegularEvent, PopulationScopeEventMixin):
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
        super().__init__(module, frequency=DateOffset(months=1))
        # Insert if neccessary


    def apply(self, population):
        """Apply this event to the population.

        :param population: the current population
        """

        # 1. Basic items and output
        df = population.props
        diab_total = df.diab_current_status.sum()
        proportion_diab = diab_total / len(df)

        # 2. Get (and hold) index of people with and w/o diabetes
        currently_diab_yes = df[df.diab_current_status & df.is_alive].index
        currently_diab_no = df[~df.diab_current_status & df.is_alive].index

        # 3. Handle new cases of diabetes
        ages_of_no_diab = population.age.loc[currently_diab_no]

        joined = pd.merge(ages_of_no_diab.reset_index(), Diab_incidence, left_on=['years'], right_on=['age'], how='left').set_index('person')
        random_numbers = np.random.rand(len(joined))
        now_diabetes = (joined.probability > random_numbers)

        # 3.1 Ways to check what's happening
        temp = pd.merge(population.age, df, left_index=True, right_index=True, how='inner')
        temp_2 = pd.DataFrame([population.age.years, joined.probability, random_numbers, df['diab_current_status']])

        # 3.2 If new diabetes case
        if now_diabetes.sum():

            diab_idx = currently_diab_no[now_diabetes]

            df.loc[diab_idx, 'diab_current_status'] = True
            df.loc[diab_idx, 'diab_historic_status'] = 'C'
            df.loc[diab_idx, 'diab_date_case'] = self.sim.date

        print("\n", "Time is: ", self.sim.date, "New cases have been assigned.  ")

        # 4. Handle cure
        ages_of_yes_diab = population.age.loc[currently_diab_yes]

        joined = pd.merge(ages_of_yes_diab.reset_index(), Diab_treatment, left_on=['years'], right_on=['age'],
                          how='left').set_index('person')
        random_numbers = np.random.rand(len(joined))
        now_treated = (joined.probability > random_numbers)

        # 4.1 If newly treated
        if now_treated.sum():
            diab_idx = currently_diab_yes[now_treated]

            df.loc[diab_idx, 'diab_treatment_status'] = 'C'
            df.loc[diab_idx, 'diab_date_treatment'] = self.sim.date

        print("\n", "Treatment has been assigned.  ")


class DiabetesLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """comments...
        """
        # run this event repeatedly
        self.repeat = 6
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # Get some summary statistics
        df = population.props

        diab_total = df.diab_current_status.sum()
        proportion_diab = diab_total / len(df)

        mask = (df['diab_date_case'] > self.sim.date - DateOffset(months=self.repeat))
        positive_in_last_month = mask.sum()
        mask = (df['diab_date_treatment'] > self.sim.date - DateOffset(months=self.repeat))
        cured_in_last_month = mask.sum()

        counts = {'N': 0, 'C': 0, 'P': 0}
        counts.update(df['diab_historic_status'].value_counts().to_dict())
        status = 'Status: { N: %(N)d; C: %(C)d; P: %(P)d }' % counts

        print("\n", "Output for the 6 months")
        print('%s - Diabetes: {TotDiab: %d; PropDiab: %.3f; PrevMonth: {New: %d; Cured: %d}; %s }' %
              (self.sim.date,
               diab_total,
               proportion_diab,
               positive_in_last_month,
               cured_in_last_month,
               status), flush=True)
