import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent

# need to import HIV, HIV_Event, ART, ART_Event, BCG vaccine

# IPT and rifampicin as separate methods


# initial pop data #
inds = pd.read_csv('Q:/Thanzi la Onse/HIV/initial_pop_dataframe2018.csv')

TBincidence = pd.read_excel('Q:/Thanzi la Onse/TB/Method Template TB.xlsx', sheet_name='TB_incidence', header=0)

latent_TB_prevalence_total = 3170000  # Houben model paper
latent_TB_prevalence_children = 525000
latent_TB_prevalence_adults = latent_TB_prevalence_total - latent_TB_prevalence_children


# this class contains all the methods required to set up the baseline population
class TB(Module):
    """ Sets up baseline TB prevalence.

    Methods required:
    * `read_parameters(data_folder)`
    * `initialise_population(population)`
    * `initialise_simulation(sim)`
    * `on_birth(mother, child)`
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
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'has_TB': Property(Types.CATEGORICAL, 'TB status: Uninfected, Latent, Active'),
        'date_TB_infection': Property(Types.DATE, 'Date acquired TB infection'),
        'date_TB_death': Property(Types.DATE, 'Projected time of TB death if untreated'),
        'on_treatment': Property(Types.BOOL, 'Currently on treatment for TB'),
        'date_TB_treatment_start': Property(Types.DATE, 'Date treatment started'),
        'date_death': Property(Types.DATE, 'Date of death'),
    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.
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

    # baseline population
    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """
        # create a population where no-one has TB
        has_TB = 'Uninfected'
        date_TB_infection = None
        date_TB_death = None
        on_treatment = False
        date_TB_treatment_start = None
        date_TB_death = False



    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """
        raise NotImplementedError

    def on_birth(self, mother, child):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother: the mother for this child
        :param child: the new child
        """
        raise NotImplementedError







# functions to be implemented

def force_of_infection_tb(inds):
    infected = len(inds[(inds.tb_status == 'I') & (inds.tb_treat == 0)])  # number infected untreated

    # number co-infected with HIV * relative infectiousness (lower)
    hiv_infected = rel_infectiousness_HIV * len(inds[(inds.tb_status == 'I') & (inds.status == 'I')])

    total_pop = len(inds[(inds.status != 'DH') & (inds.status != 'D')])  # whole population currently alive

    foi = beta * ((infected + hiv_infected) / total_pop)  # force of infection for adults

    return foi


def inf_tb(inds):
    # apply foi to uninfected pop -> latent infection

    return inds


def tb_treatment(inds):
    # apply diagnosis / treatment / self-cure combined rates

    return inds


def progression_tb(inds):
    # apply combined progression / relapse / reinfection rates to infected pop

    return inds


def recover_tb(inds):
    # apply combined diagnosis / treatment / self-cure rates to TB cases

    return inds

# TODO: isoniazid preventive therapy
# TODO: rifampicin / alternative TB treatment
