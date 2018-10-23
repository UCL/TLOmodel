"""
TB model
"""

import pandas as pd
import numpy as np

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent


TBincidence = pd.read_excel('Q:/Thanzi la Onse/TB/Method_TB.xlsx', sheet_name='Active_TB_Incidence', header=0)

class TB_baseline(Module):
    """ Set up the baseline population with TB prevalence

    """

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    PARAMETERS = {
        'prop_fast_progressor': Parameter(
            Types.REAL,
            'Proportion of infections that progress directly to active stage, Vynnycky'),
        'transmission_rate': Parameter(
            Types.REAL,
            'TB transmission rate, estimated by Juan'),
        'progression_to_active_rate': Parameter(
            Types.REAL,
            'Combined rate of progression/reinfection/relapse from Juan'),
        'rr_TB_HIV_stages': Parameter(
            Types.REAL,
            'relative risk of TB hin HIV+ compared with HIV- by CD4 stage'),
        'rr_TB_ART': Parameter(
            Types.REAL,
            'relative risk of TB in HIV+ on ART'),
        'rr_TB_malnourished': Parameter(Types.REAL, 'relative risk of TB with malnourishment'),
        'rr_TB_diabetes1': Parameter(Types.REAL, 'relative risk of TB with diabetes type 1'),
        'rr_TB_alcohol': Parameter(Types.REAL, 'relative risk of TB with heavy alcohol use'),
        'rr_TB_smoking': Parameter(Types.REAL, 'relative risk of TB with smoking'),
        'rr_TB_pollution': Parameter(Types.REAL, 'relative risk of TB with indoor air pollution'),
        'rr_infectiousness_HIV': Parameter(
            Types.REAL,
            'relative infectiousness of TB in HIV+ compared with HIV-'),
        'recovery': Parameter(
            Types.REAL,
            'combined rate of diagnosis, treatment and self-cure, from Juan'),
        'TB_mortality_rate': Parameter(
            Types.REAL,
            'mortality rate with active TB'),
        'rr_TB_mortality_HIV': Parameter(
            Types.REAL,
            'relative risk of mortality from TB in HIV+ compared with HIV-'),
        'prob_latent_TB': Parameter(
            Types.REAL,
            'probability of latent TB in baseline pop averaged over whole pop'),
        'force_of_infection': Parameter(
            Types.REAL,
            'force of infection for new latent infections applied across whole pop'),  # dummy value
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'has_TB': Property(Types.CATEGORICAL, 'TB status: Uninfected, Latent, Active'),
        'date_TB_infection': Property(Types.DATE, 'Date acquired TB infection'),
        'date_TB_death': Property(Types.DATE, 'Projected time of TB death if untreated'),
        'on_treatment': Property(Types.BOOL, 'Currently on treatment for TB'),
    }


    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we do nothing.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        params = self.parameters
        params['prop_fast_progressor'] = 0.14
        params['transmission_rate'] = 7.2
        params['progression_to_active_rate'] = 0.5

        params['rr_TB_with_HIV_stages'] = [3.44, 6.76, 13.28, 26.06]
        params['rr_ART'] = 0.39
        params['rr_TB_malnourished'] = 2.1
        params['rr_TB_diabetes1'] = 3
        params['rr_TB_alcohol'] = 2.9
        params['rr_TB_smoking'] = 2.6
        params['rr_TB_pollution'] = 1.5

        params['rr_infectiousness_HIV'] = 0.52
        params['recovery'] = 2
        params['TB_mortality_rate'] = 0.15
        params['rr_TB_mortality_HIV'] = 17.1
        params['prob_latent_TB'] = 0.0015
        params['force_of_infection'] = 0.0015  # dummy value


    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        p = population
        now = self.sim.date

        # set-up baseline population
        p.has_TB = 'Uninfected'
        p.date_TB_infection = False

        # assign active infections
        # include RR here for other attributes e.g. HIV, diabetes etc.
        eff_prob_active_TB = pd.Series(TBincidence.probability_infection)
        has_active_TB = eff_prob_active_TB > self.rng.rand(len(eff_prob_active_TB))
        assign_active_TB = has_active_TB[has_active_TB].index
        p[assign_active_TB, 'has_TB'] = 'Active'
        p[assign_active_TB, 'date_TB_infection'] = now



        # assign latent infections, no age data available yet
        eff_prob_latent_TB = pd.Series(self.parameters['prob_latent_TB'], index=p[p.has_TB == 'Uninfected'].index)
        has_latent_TB = eff_prob_latent_TB > self.rng.rand(len(eff_prob_latent_TB))
        assign_latent_TB = has_latent_TB[has_latent_TB].index
        p[assign_latent_TB, 'has_TB'] = 'Latent'


    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        TB_poll = TB_Event(self)
        # first TB_event happens in 12 months, i.e. 2019
        sim.schedule_event(TB_poll, sim.date + DateOffset(months=12))


    def on_birth(self, mother, child):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother: the mother for this child
        :param child: the new child
        """
        raise NotImplementedError


class TB_Event(RegularEvent, PopulationScopeEventMixin):
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
        super().__init__(module, frequency=DateOffset(months=1))  # every month
        # make sure any rates are monthly if frequency of event occurs monthly

    def apply(self, population):
        """Apply this event to the population.

        :param population: the current population
        """

        params = self.module.parameters
        now = self.sim.date
        rng = self.module.rng

        p = population

        # apply a force of infection to produce new latent cases
        # remember event is occurring each month so scale rates accordingly
        prob_TB_new = pd.Series(params['force_of_infection'], index=p[p.has_TB == 'Uninfected'].index)
        is_newly_infected = prob_TB_new > rng.rand(len(prob_TB_new))
        new_case = is_newly_infected[is_newly_infected].index

        # 14% of new cases become active directly
        p[new_case, 'has_TB'] = np.random.choice(['Latent', 'Active'], size=len(new_case),
                                                 replace=True,
                                                 p=[1 - params['prop_fast_progressor'], params['prop_fast_progressor']])



        # slow progressors with latent TB become active at estimated rate
        # only those with latent TB can develop active TB
        prob_TB_active = pd.Series(params['progression_to_active_rate'], index=p[p.has_TB == 'Latent'].index)

        prog_to_active = prob_TB_active > rng.rand(len(prob_TB_active))
        new_active_case = prog_to_active[prog_to_active].index
        p[new_active_case, 'has_TB'] = 'Active'
        p[new_active_case, 'date_TB_infection'] = now

        # include treatment / recovery
        # move back from active to latent








