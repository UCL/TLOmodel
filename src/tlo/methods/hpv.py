from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import pandas as pd

from tlo import DAYS_IN_YEAR, DateOffset, Module, Parameter, Property, Types, logging
from tlo.analysis.utils import get_counts_by_sex_and_age_group
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import Metadata
from tlo.methods.causes import Cause
from tlo.methods.demography import InstantaneousDeath
from tlo.methods.hsi_event import HSI_Event
from tlo.methods.hsi_generic_first_appts import GenericFirstAppointmentsMixin
from tlo.methods.symptommanager import Symptom
from tlo.util import random_date, read_csv_files

if TYPE_CHECKING:
    from tlo.methods.hsi_generic_first_appts import HSIEventScheduler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HPV(Module, GenericFirstAppointmentsMixin):
    """This is a HPV infectious Process.

    It demonstrates the following behaviours in respect of the healthsystem module:

        - Registration of the disease module with healthsystem
        - Reading DALY weights and reporting daly values related to this disease
        - Health care seeking
        - Usual HSI behaviour
        - Restrictive requirements on the facility_level for the HSI_event
        - Use of the SymptomManager
    """

    INIT_DEPENDENCIES = {'Demography', 'SymptomManager'}

    OPTIONAL_INIT_DEPENDENCIES = {'HealthBurden'}

    # Declare Metadata
    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN,
        Metadata.REPORTS_DISEASE_NUMBERS
    }

    # Declare Causes of Death
    CAUSES_OF_DEATH = {}

    # Declare Causes of Disability
    CAUSES_OF_DISABILITY = {}

    PARAMETERS = {
        "r_hpv": Parameter(
            Types.REAL,
            "probability per month of oncogenic hpv infection",
        ),
        "Init_prevalence": Parameter(
            Types.REAL,
            "Initial prevalence of oncogenic hpv infection",
        ),
    }

    PROPERTIES = {
        'hp_is_infected': Property(
            Types.BOOL, 'Is infected with oncogenic hpv infection'),
    }

    def __init__(self, name=None):
        # NB. Parameters passed to the module can be inserted in the __init__ definition.

        super().__init__(name)

    def read_parameters(self, resourcefilepath: Optional[Path] = None):
        """Read in parameters and do the registration of this module and its symptoms"""
        self.load_parameters_from_dataframe(
            read_csv_files(Path(resourcefilepath) / "ResourceFile_HPV",
                           files="parameter_values")
        )

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        df = population.props  # a shortcut to the dataframe storing data for individiuals

        # Set default for properties
        df.loc[df.is_alive, 'hp_is_infected'] = False  # default: no individuals infected

        Susceptible = df.loc[df.is_alive & (df.age_years >= 15)].index


        # randomly selected some individuals as infected
        initial_infected = self.parameters['initial_prevalence']
        index_of_infected = self.rng.choice(Susceptible, size=initial_infected*Susceptible, replace=False)
        df.loc[index_of_infected, 'hp_is_infected'] = True

    def initialise_simulation(self, sim):

        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        # add the basic event
        event = HpvInfectionEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=6))

        # add an event to log to screen
        sim.schedule_event(HpvLoggingEvent(self), sim.date + DateOffset(months=12))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the ID for the mother for this child
        :param child_id: the ID for the new child
        """

        df = self.sim.population.props  # shortcut to the population props dataframe
        df.at[child_id, 'hp_is_infected'] = False

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        logger.debug(key="debug", data="This is hpv reporting my health values")

        df = self.sim.population.props  # shortcut to population properties dataframe

        health_values = pd.Series(index=df.index[df.is_alive], data=0)
        return health_values  # returns the series

    def report_summary_stats(self):
        df = self.sim.population.props
        prevalence_by_age_group_sex = get_counts_by_sex_and_age_group(df, 'hp_is_infected')
        return {'infected': prevalence_by_age_group_sex}

class HpvInfectionEvent(RegularEvent, PopulationScopeEventMixin):
    """
    This event is occurring regularly at one 6 months intervals and controls the infection process of HPV.
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=6))
        assert isinstance(module, HPV)

    def apply(self, population):

        logger.debug(key='debug', data='This is HpvInfectionEvent, tracking the disease progression of the population.')

        df = population.props

        # 1. get (and hold) index of currently infected and uninfected individuals
        currently_infected = df.index[df.hp_is_infected & df.is_alive]
        currently_susc = df.index[df.is_alive & ~df.hp_is_infected]

        if df.is_alive.sum():
            prevalence = len(currently_infected) / (
                len(currently_infected) + len(currently_susc))
        else:
            prevalence = 0

        # 2. handle new infections
        now_infected = self.module.rng.choice([True, False],
                                              size=len(currently_susc),
                                              p=[prevalence, 1 - prevalence])

        # if any are newly infected...
        if now_infected.sum():
            infected_idx = currently_susc[now_infected]

            df.loc[infected_idx, 'hp_is_infected'] = True

            # schedule death events for newly infected individuals

        else:
            logger.debug(key='debug', data='This is HpvInfectionEvent, no one is newly infected.')

class HpvLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """Produce a summmary of the numbers of people with respect to their 'hpv status'
        """
        # run this event every month
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        assert isinstance(module, HPV)

    def apply(self, population):
        # get some summary statistics
        df = population.props

        infected_total = df.loc[df.is_alive, 'hp_is_infected'].sum()
        proportion_infected = infected_total / len(df)

        logger.info(key='summary',
                    data={'TotalInf': infected_total,
                          'PropInf': proportion_infected,
                          })

