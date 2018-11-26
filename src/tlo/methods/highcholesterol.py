"""
This is the method for high cholesterol
Developed by Mikaela Smit, October 2018

"""

import pandas as pd
import numpy as np

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent, Event, IndividualScopeEventMixin

# Read in data
file_path = '/Users/mc1405/Dropbox/Projects - ongoing/Malawi Project/Thanzi la Onse/04 - Methods Repository/Method_HC.xlsx'
method_hc_data = pd.read_excel(file_path, sheet_name=None, header=0)
HC_prevalence, HC_incidence, HC_treatment = method_hc_data['prevalence2018'], method_hc_data['incidence2018_plus'], \
                                            method_hc_data['treatment_parameters']


class HC(Module):
    """
    This is the high cholesterol module

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
        'prob_HC_basic': Parameter(Types.REAL,
                                   'Probability of getting high cholesterol given no pre-existing condition'),
        'prob_HCgivenDiab': Parameter(Types.REAL,
                                    'Probability of getting high cholesterol given pre-existing diiabetes'),

    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'hc_risk': Property(Types.REAL, 'Risk of high cholesterol given pre-existing condition'),
        'hc_current_status': Property(Types.BOOL, 'Current high cholesterol status'),
        'hc_historic_status': Property(Types.CATEGORICAL,
                                       'Historical status: N=never; C=Current, P=Previous',
                                       categories=['N', 'C', 'P']),
        'hc_date_case': Property(Types.DATE, 'Date of latest high cholesterol'),
        'hc_treatment_status': Property(Types.CATEGORICAL,
                                        'Historical status: N=never; C=Current, P=Previous',
                                        categories=['N', 'C', 'P']),
        'hc_date_treatment': Property(Types.DATE, 'Date of latest high cholesterol treatment'),
    }

    def read_parameters(self, data_folder):
        """Will need to update this to read parameter values from file.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """

        params = self.parameters
        params['prob_HC_basic'] = 1
        params['prob_HCgivenDiab'] = 2  # 1.12
        params['prob_success_treat'] = 0.5

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
        df['hc_risk'] = 'N'              # Default setting: no risk given pre-existing conditions
        df['hc_current_status'] = False  # Default setting: no one has high cholesterol
        df['hc_historic_status'] = 'N'   # Default setting: no one has high cholesterol
        df['hc_date_case'] = pd.NaT      # Default setting: no one has a date for high cholesterol
        df['hc_treatment_status'] = 'N'  # Default setting: no one is treated
        df['hc_date_treatment'] = pd.NaT # Details setting: no one has a date of treatment

        # 3. Assign prevalence as per data, by using probability by age
        joined = pd.merge(population.age, HC_prevalence, left_on=['years'], right_on=['age'], how='left')
        random_numbers = np.random.rand(len(df))

        # 3.1 Depending on pre-existing conditions, get associated risk and update prevalence and assign high cholesterol
        df.loc[~df.diab_current_status, 'hc_risk'] = self.prob_HC_basic         # Basic risk, no pre-existing conditions
        df.loc[df.diab_current_status, 'hc_risk'] = self.prob_HCgivenDiab       # Risk if pre-existing diabetes
        joined.probability_updated = joined.probability * df.hc_risk            # Update 'real' prevalence
        df['hC_current_status'] = (joined.probability_updated > random_numbers) # Assign prevalence at t0

        # 3.1 Ways to check what's happening
        # temp = pd.merge(population.age, df, left_index=True, right_index=True, how='inner')
        # temp = pd.DataFrame([population.age.years, joined.Proportion, random_numbers, df['hc_current_status']])

        # 4. Count all individuals by status at the start
        highcholesterol_count = df.hc_current_status.sum()
        pop_count = len(df.hc_current_status)

        # 5. Set date of high cholesterol amongst those with prevalent cases
        hc_years_ago = np.random.exponential(scale=5, size=highcholesterol_count)
        infected_td_ago = pd.to_timedelta(hc_years_ago, unit='y')

        # 5.1 Set the properties of those with prevalent high cholesterol
        df.loc[df.hc_current_status, 'hc_date_case'] = self.sim.date - infected_td_ago
        df.loc[df.hc_current_status, 'hc_historic_status'] = 'C'

        print("\n", "Population has been initialised, prevalent cases have been assigned.  ")

    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """
        # Add the basic event (implement below)
        event = HCEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=1))

        # Add an event to log to screen
        sim.schedule_event(HCLoggingEvent(self), sim.date + DateOffset(months=6))

    def on_birth(self, mother, child):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother: the mother for this child
        :param child: the new child
        """
        pass


class HCEvent(RegularEvent, PopulationScopeEventMixin):
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
        self.prob_HC_basic = module.parameters['prob_HC_basic']
        self.prob_HCgivenDiab = module.parameters['prob_HCgivenDiab']
        self.prob_success_treat = module.parameters['prob_success_treat']


    def apply(self, population):
        """Apply this event to the population.

        :param population: the current population
        """

        # 1. Basic items and output
        df = population.props
        hc_total = df.hc_current_status.sum()
        proportion_hc = hc_total / len(df)

        # 2. Get (and hold) index of people with and w/o high cholesterol
        currently_hc_yes = df[df.hc_current_status & df.is_alive].index
        currently_hc_no = df[~df.hc_current_status & df.is_alive].index

        # 3. Handle new cases of high cholesterol
        ages_of_no_hc = population.age.loc[currently_hc_no]
        joined = pd.merge(ages_of_no_hc.reset_index(), HC_incidence, left_on=['years'], right_on=['age'], how='left').set_index('person')
        random_numbers = np.random.rand(len(joined))

        # 3.1 Depending on pre-existing conditions, get associated risk and update prevalence and assign high cholesterol
        df.loc[~df.diab_current_status, 'hc_risk'] = self.prob_HC_basic  # Basic risk, no pre-existing conditions
        df.loc[df.diab_current_status, 'hc_risk'] = self.prob_HCgivenDiab  # Risk if pre-existing diabetes
        joined.probability_updated = joined.probability * df.hc_risk  # Update 'real' incidence
        now_highcholesterol = (joined.probability_updated > random_numbers)  # Assign incidence


        # 3.1 Ways to check what's happening
        temp = pd.merge(population.age, df, left_index=True, right_index=True, how='inner')
        temp_2 = pd.DataFrame([population.age.years, joined.probability, random_numbers, df['hc_current_status']])

        # 3.2 If new high cholesterol case
        if now_highcholesterol.sum():

            hc_idx = currently_hc_no[now_highcholesterol]

            df.loc[hc_idx, 'hc_current_status'] = True
            df.loc[hc_idx, 'hc_historic_status'] = 'C'
            df.loc[hc_idx, 'hc_date_case'] = self.sim.date

        print("\n", "Time is: ", self.sim.date, "New cases have been assigned.  ")

        # 4. Handle cure
        ages_of_yes_hc = population.age.loc[currently_hc_yes]

        joined = pd.merge(ages_of_yes_hc.reset_index(), HC_treatment, left_on=['years'], right_on=['age'],
                          how='left').set_index('person')
        random_numbers = np.random.rand(len(joined))
        now_treated = (joined.probability > random_numbers)

        # 4.1 If newly treated
        if now_treated.sum():
            hc_idx = currently_hc_yes[now_treated]

            df.loc[hc_idx, 'hc_treatment_status'] = 'C'
            df.loc[hc_idx, 'hc_date_treatment'] = self.sim.date

        print("\n", "Treatment has been assigned.  ")


class HCLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """comments...
        """
        # run this event repeatedly
        self.repeat = 6
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # Get some summary statistics
        df = population.props

        hc_total = df.hc_current_status.sum()
        proportion_hc = hc_total / len(df)

        mask = (df['hc_date_case'] > self.sim.date - DateOffset(months=self.repeat))
        positive_in_last_month = mask.sum()
        mask = (df['hc_date_treatment'] > self.sim.date - DateOffset(months=self.repeat))
        cured_in_last_month = mask.sum()

        counts = {'N': 0, 'C': 0, 'P': 0}
        counts.update(df['hc_historic_status'].value_counts().to_dict())
        status = 'Status: { N: %(N)d; C: %(C)d; P: %(P)d }' % counts

        print("\n", "Output for the 6 months")
        print('%s - High cholesterol: {TotHC: %d; PropHC: %.3f; PrevMonth: {New: %d; Cured: %d}; %s }' %
              (self.sim.date,
               hc_total,
               proportion_hc,
               positive_in_last_month,
               cured_in_last_month,
               status), flush=True)
