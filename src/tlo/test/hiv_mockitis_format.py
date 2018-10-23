"""
A skeleton template for disease methods.
"""
import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent, Event, IndividualScopeEventMixin


# read in data files #
# use function read.parameters in class HIV to do this?
file_path = 'Q:/Thanzi la Onse/HIV/Method_HIV.xlsx'
method_hiv_data = pd.read_excel(file_path, sheet_name=None, header=0)
HIV_prev, HIV_death, HIV_inc, CD4_base, time_CD4, initial_state_probs, \
age_distr = method_hiv_data['prevalence2018'], method_hiv_data['deaths2009_2021'], \
            method_hiv_data['incidence2009_2021'], \
            method_hiv_data['CD4_distribution2018'], method_hiv_data['Time_spent_by_CD4'], \
            method_hiv_data['Initial_state_probs'], method_hiv_data['age_distribution2018']


class hiv_mock(Module):
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
        'prob_infant_fast_progressor':
            Parameter(Types.LIST, 'Probabilities that infants are fast or slow progressors'),
        'infant_fast_progression':
            Parameter(Types.BOOL, 'Classification of infants as fast progressor'),
        'exp_rate_mort_infant_fast_progressor':
            Parameter(Types.REAL, 'Exponential rate parameter for mortality in infants fast progressors'),
        'weibull_scale_mort_infant_slow_progressor':
            Parameter(Types.REAL, 'Weibull scale parameter for mortality in infants slow progressors'),
        'weibull_shape_mort_infant_slow_progressor':
            Parameter(Types.REAL, 'Weibull shape parameter for mortality in infants slow progressors'),
        'weibull_shape_mort_adult':
            Parameter(Types.REAL, 'Weibull shape parameter for mortality in adults'),
        'proportion_high_sexual_risk_male':
            Parameter(Types.REAL, 'proportion of men who have high sexual risk behaviour'),
        'proportion_high_sexual_risk_female':
            Parameter(Types.REAL, 'proportion of women who have high sexual risk behaviour'),
        'rr_HIV_high_sexual_risk':
            Parameter(Types.REAL, 'relative risk of acquiring HIV with high risk sexual behaviour'),
        'proportion_on_ART_infectious':
            Parameter(Types.REAL, 'proportion of people on ART contributing to transmission as not virally suppressed'),
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'has_HIV': Property(Types.BOOL, 'HIV status'),
        'date_HIV_infection': Property(Types.DATE, 'Date acquired HIV infection'),
        'date_AIDS_death': Property(Types.DATE, 'Projected time of AIDS death if untreated'),
        'on_ART': Property(Types.BOOL, 'Currently on ART'),
        'date_ART_start': Property(Types.DATE, 'Date ART started'),
        'ART_mortality': Property(Types.REAL, 'Mortality rates whilst on ART'),
        'sexual_risk_group': Property(Types.REAL, 'Relative risk of HIV based on sexual risk high/low'),
        'date_death': Property(Types.DATE, 'Date of death'),
        'CD4_state': Property(Types.CATEGORICAL, 'CD4 state: >500; 350-500; 200-350; 0-200',
                              categories=['state1', 'state2', 'state3', 'state4']),
    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we do nothing.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        params = self.parameters
        params['prob_infant_fast_progressor'] = [0.36, 1 - 0.36]
        params['infant_progression_category'] = ['FAST', 'SLOW']
        params['exp_rate_mort_infant_fast_progressor'] = 1.08
        params['weibull_scale_mort_infant_slow_progressor'] = 16
        params['weibull_size_mort_infant_slow_progressor'] = 1
        params['weibull_shape_mort_infant_slow_progressor'] = 2.7
        params['weibull_shape_mort_adult'] = 2
        params['proportion_high_sexual_risk_male'] = 0.0913
        params['proportion_high_sexual_risk_female'] = 0.0095
        params['rr_HIV_high_sexual_risk'] = 2
        params['proportion_on_ART_infectious'] = 0.2


    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        df = population.props
        age = population.age

        df['has_HIV'] = False
        df['date_HIV_infection'] = pd.NaT
        df['date_AIDS_death'] = pd.NaT
        df['sexual_risk_group'] = 1

        print('hello')

        # randomly selected some individuals as infected
        initial_infected = 0.3
        initial_uninfected = 1 - initial_infected
        df['has_HIV'] = np.random.choice([True, False], size=len(df), p=[initial_infected, initial_uninfected])

        # assign high/low sexual risk
        # male_sample = df[(df.sex == 'M') & (age > 15)].sample(
        #     frac=self.parameters['proportion_high_sexual_risk_male']).index
        # female_sample = df[(df.sex == 'F') & (age > 15)].sample(
        #     frac=self.parameters['proportion_high_sexual_risk_female']).index

        # # these individuals have higher risk of hiv
        # df[male_sample | female_sample, 'sexual_risk_group'] = self.parameters['rr_HIV_high_sexual_risk']
        #
        # # assign baseline prevalence
        # now = self.sim.date
        # prevalence = HIV_prev.loc[HIV_prev.year == now.year, ['age', 'sex', 'prevalence']]
        #
        # for i in range(0, 81):
        #     # male
        #     # scale high/low-risk probabilities to sum to 1 for each sub-group
        #     idx = (age == i) & (df.sex == 'M')
        #
        #     if idx.any():
        #         prob_i = df[idx, 'sexual_risk_group']
        #
        #         # sample from uninfected df using prevalence from UNAIDS
        #         fraction_infected = prevalence.loc[(prevalence.sex == 'M') & (prevalence.age == i), 'proportion']
        #         male_infected = df[idx].sample(frac=fraction_infected, weights=prob_i).index
        #         df[male_infected, 'has_HIV'] = True
        #
        #     # female
        #     # scale high/low-risk probabilities to sum to 1 for each sub-group
        #     idx = (age == i) & (df.sex == 'F')
        #
        #     if idx.any():
        #         prob_i = df[idx, 'sexual_risk_group']
        #
        #         # sample from uninfected df using prevalence from UNAIDS
        #         fraction_infected = prevalence.loc[(prevalence.sex == 'F') & (prevalence.age == i), 'proportion']
        #         female_infected = df[idx].sample(frac=fraction_infected, weights=prob_i).index
        #         df[female_infected, 'has_HIV'] = True


    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """
        event = hiv_event(self)
        sim.schedule_event(event, sim.date + DateOffset(months=12))


    def on_birth(self, mother, child):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother: the mother for this child
        :param child: the new child
        """
        pass


class hiv_event(RegularEvent, PopulationScopeEventMixin):
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
        super().__init__(module, frequency=DateOffset(months=12))  # every 12 months

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        pass



