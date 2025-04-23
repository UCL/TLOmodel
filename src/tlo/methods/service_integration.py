from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np
import pandas as pd

from tlo import DAYS_IN_YEAR, DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import Metadata
from tlo.methods.causes import Cause
from tlo.methods.demography import InstantaneousDeath
from tlo.methods.hsi_event import HSI_Event
from tlo.methods.hsi_generic_first_appts import GenericFirstAppointmentsMixin
from tlo.methods.symptommanager import Symptom

if TYPE_CHECKING:
    from tlo.methods.hsi_generic_first_appts import HSIEventScheduler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ServiceIntegration(Module, GenericFirstAppointmentsMixin):
    """
    """

    # Declare modules that need to be registered in simulation and initialised before
    # this module
    INIT_DEPENDENCIES = {'Demography', 'HealthSystem', 'SymptomManager'}


    # Declare Metadata
    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }

    # Declare Causes of Death
    CAUSES_OF_DEATH = {}

    # Declare Causes of Disability
    CAUSES_OF_DISABILITY = {
    }

    PARAMETERS = {
        'serv_int_screening': Parameter(Types.REAL, 'Whether coverage of service integration screening '
                                                    'interventions is being increased or not'),
        'serv_int_chronic': Parameter(Types.REAL, 'Whether coverage of service integration chronic care '
                                                    'interventions is being increased or not'),
        'serv_int_mch': Parameter(Types.REAL, 'Whether coverage of service integration maternal child health '
                                                    'interventions is being increased or not'),

    }

    PROPERTIES = {
    }

    def __init__(self, name=None, resourcefilepath=None):
        # NB. Parameters passed to the module can be inserted in the __init__ definition.

        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.
        For now, we are going to hard code them explicity.
        Register the module with the health system and register the symptoms
        """
        self.parameters['serv_int_screening'] = False
        self.parameters['serv_int_chronic'] = False
        self.parameters['serv_int_mch'] = False

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        pass

    def initialise_simulation(self, sim):

        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        # add the basic event (we will implement below)
        event = ServiceIntegrationParameterUpdateEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(years=15))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the ID for the mother for this child
        :param child_id: the ID for the new child
        """

        pass

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        pass

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        pass


class ServiceIntegrationParameterUpdateEvent(Event, PopulationScopeEventMixin):

    # This event is occuring regularly at one monthly intervals

    def __init__(self, module):
        super().__init__(module)
        assert isinstance(module, ServiceIntegration)

    def apply(self, population):
        df = self.sim.population.props
        params = self.module.parameters

        if params['serv_int_screening']:
            pass
        elif params['serv_int_chronic']:
            pass
        elif params['serv_int_mch']:
            pass


        df = population.props
        p = self.module.parameters
        rng: np.random.RandomState = self.module.rng

        # 1. get (and hold) index of currently infected and uninfected individuals
        # currently_cs = df.index[df.cs_has_cs & df.is_alive]
        currently_not_cs = df.index[~df.cs_has_cs & df.is_alive]

        # 2. handle new cases
        p_aq = p['p_acquisition_per_year'] / 12.0
        now_acquired = rng.random_sample(size=len(currently_not_cs)) < p_aq

        # if any are new cases
        if now_acquired.sum():
            newcases_idx = currently_not_cs[now_acquired]

            death_years_ahead = rng.exponential(scale=20, size=now_acquired.sum())
            death_td_ahead = pd.to_timedelta(death_years_ahead * DAYS_IN_YEAR, unit='D')

            df.loc[newcases_idx, 'cs_has_cs'] = True
            df.loc[newcases_idx, 'cs_status'].values[:] = 'C'
            df.loc[newcases_idx, 'cs_date_acquired'] = self.sim.date
            df.loc[newcases_idx, 'cs_scheduled_date_death'] = self.sim.date + death_td_ahead
            df.loc[newcases_idx, 'cs_date_cure'] = pd.NaT

            # schedule death events for new cases
            for person_index in newcases_idx:
                death_event = ChronicSyndromeDeathEvent(self.module, person_index)
                self.sim.schedule_event(death_event, df.at[person_index, 'cs_scheduled_date_death'])

            # Assign symptoms:
            for symp in self.module.parameters['prob_of_symptoms']:
                # persons who will have symptoms (each can occur independently)
                persons_id_with_symp = np.array(newcases_idx)[
                    self.module.rng.rand(len(newcases_idx)) < self.module.parameters['prob_of_symptoms'][symp]
                    ]

                self.sim.modules['SymptomManager'].change_symptom(
                    person_id=list(persons_id_with_symp),
                    symptom_string=symp,
                    add_or_remove='+',
                    disease_module=self.module
                )

        # 3) Handle progression to severe symptoms
        curr_cs_but_not_craving_sandwiches = list(set(df.index[df.cs_has_cs & df.is_alive])
                                                  - set(
            self.sim.modules['SymptomManager'].who_has('craving_sandwiches')))

        become_severe = (
            self.module.rng.random_sample(size=len(curr_cs_but_not_craving_sandwiches))
            < p['prob_dev_severe_symptoms_per_year'] / 12
        )
        become_severe_idx = np.array(curr_cs_but_not_craving_sandwiches)[become_severe]

        self.sim.modules['SymptomManager'].change_symptom(
            person_id=list(become_severe_idx),
            symptom_string='craving_sandwiches',
            add_or_remove='+',
            disease_module=self.module
        )


