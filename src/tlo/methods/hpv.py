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
    Groups:
        g1 = HPV16/18
        g2 = other vaccine-covered high-risk HPV (31/33/45/52/58)
        g3 = other high-risk HPV (35/39/51/56/59/68)

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

    HPV_GROUPS = ['hr1', 'hr2', 'hr3']

    PARAMETERS = {
        "init_prev_hpv_hr1": Parameter(
            Types.REAL,
            "Initial prevalence of hpv 16/18 infection",
        ),
        "init_prev_hpv_hr2": Parameter(
            Types.REAL,
            "Initial prevalence of HPV 31/33/45/52/58 infection",
        ),
        "init_prev_hpv_hr3": Parameter(
            Types.REAL,
            "Initial prevalence of other HR types"
        ),
        # Transmission coefficients β
        "b_hpv": Parameter(
            Types.REAL,
            "Baseline transmission coefficient for HPV Infection",
        ),

        # Clearance probabilities
        "r_clear_12": Parameter(
            Types.REAL,
            "probability of clearing HPV after 12 month",
        ),
        "r_clear_24": Parameter(
            Types.REAL,
            "probability of clearing HPV after 24 month",
        ),
    }

    PROPERTIES = {
        'hp_is_infected': Property(
            Types.BOOL, 'Is infected with oncogenic hpv group'),
        'hp_group': Property(
            Types.STRING, 'Current HPV types carried, separated by "|"'),
        'hp_date_infected': Property(
            Types.DATE, 'Date of most recent infection'),
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
        df.loc[df.is_alive, 'hp_group'] = ''  # default: no hpy type
        df.loc[df.is_alive, 'hp_date_infected'] = pd.NaT  # default: not a time

        eligible = df.index[df.is_alive & (df.age_years >= 15)]

        for group in self.HPV_GROUPS:
            p_init = self.parameters[f'init_prev_hpv_{group}']
            u = self.rng.random(size=len(eligible))
            infected_this_group = eligible[u < p_init]

            for person_id in infected_this_group:
                current_groups = self._get_hpv_group_set(person_id)
                current_groups.add(group)
                self._set_hpv_group_set(person_id, current_groups)

        initially_infected = df.index[df.is_alive & df.hp_is_infected]
        if len(initially_infected) > 0:
            df.loc[initially_infected, 'hp_date_infected'] = self.sim.date

    def _get_hpv_group_set(self,person_id):
        df = self.sim.population.props
        hpv_str = df.at[person_id, 'hp_group']
        if hpv_str is None or hpv_str == '':
            return set()
        return set(hpv_str.split('|'))

    def _set_hpv_group_set(self,person_id, hpv_set):
        df = self.sim.population.props

        if len(hpv_set) == 0:
            df.at[person_id, 'hp_group'] = ''
            df.at[person_id, 'hp_is_infected'] = False
        else:
            df.at[person_id, 'hp_group'] = '|'.join(sorted(hpv_set))
            df.at[person_id, 'hp_is_infected'] = True

    def _add_new_infection_groups(self, person_id, new_groups):

        if len(new_groups) == 0:
            return
        df = self.sim.population.props
        current_groups = self._get_hpv_group_set(person_id)
        updated_groups = current_groups.union(new_groups)

        self._set_hpv_group_set(person_id, updated_groups)
        df.at[person_id, 'hp_date_infected'] = self.sim.date

    def _clear_all_infection(self, person_id):
        """Clear all HPV groups for a person."""
        df = self.sim.population.props
        self._set_hpv_group_set(person_id, set())
        df.at[person_id, 'hp_date_infected'] = pd.NaT

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
        df.at[child_id, 'hp_group'] = ''
        df.at[child_id, 'hp_date_infected'] = pd.NaT

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        logger.debug(key="debug", data="This is hpv reporting my health values")
        df = self.sim.population.props  # shortcut to population properties dataframe
        health_values = pd.Series(index=df.index[df.is_alive], data=0.0)
        return health_values  # returns the series

    def report_summary_stats(self):
        df = self.sim.population.props
        prevalence_by_age_group_sex = get_counts_by_sex_and_age_group(df, 'hp_is_infected')
        return {'infected': prevalence_by_age_group_sex}

class HpvInfectionEvent(RegularEvent, PopulationScopeEventMixin):
    """This event is occurring regularly at one 6 months intervals and controls the infection process of HPV."""

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=6))
        assert isinstance(module, HPV)

    def apply(self, population):
        logger.debug(key='debug', data='This is HpvInfectionEvent, tracking the disease progression of the population.')
        df = population.props
        module = self.module
        now = self.sim.date

        # 1. define eligible population
        eligible = df.index[df.is_alive & (df.age_years >= 15)]
        if len(eligible) == 0:
            return

        # 2. self-clearance
        infected_idx = df.index[df.is_alive & df.hp_is_infected & (df.age_years >= 15)]
        for person_id in infected_idx:
            date_inf = df.at[person_id, 'hp_date_infected']
            #if data is missing. skip for safety
            if pd.isna(date_inf):
                continue

            duration = (now.year - date_inf.year) * 12 + (now.month - date_inf.month)

            clear_now = False

            if duration >= 24:
                p_clear = module.parameters['r_clear_24']
                if module.rng.random() < p_clear:
                    clear_now = True

            elif duration >= 12:
                p_clear = module.parameters['r_clear_12']
                if module.rng.random() < p_clear:
                    clear_now = True

            if clear_now:
                module._clear_all_infection(person_id)

        # 3. recalculate prevalence by HPV group after clearance
        male_idx = df.index[df.is_alive & (df.age_years >= 15) & (df.sex == 'M')]
        female_idx = df.index[df.is_alive & (df.age_years >= 15) & (df.sex == 'F')]

        prev_male = {}
        prev_female = {}

        for group in module.HPV_GROUPS:
            male_carrier = male_idx[df.loc[male_idx, 'hp_group'].fillna('').str.contains(fr'(^|\|){group}(\||$)', regex=True)]
            female_carrier = female_idx[df.loc[female_idx, 'hp_group'].fillna('').str.contains(fr'(^|\|){group}(\||$)', regex=True)]

            prev_male[group] = len(male_carrier)/len(male_idx) if len(male_idx) > 0 else 0
            prev_female[group] = len(female_carrier) / len(female_idx) if len(female_idx) > 0 else 0

        #new infection
        beta=module.parameters['b_hpv']

        for person_id in eligible:
            sex = df.at[person_id,'sex']
            current_groups = module._get_hpv_group_set(person_id)
            new_group = set()

            if sex == 'F':
                source_prev = prev_male
            elif sex == 'M':
                source_prev = prev_female
            else:
                continue

            for group in module.HPV_GROUPS:
                if group in current_groups:
                    continue

                risk = beta * source_prev[group]
                risk = min(max(risk, 0.0), 1.0)

                if module.rng.random() < risk:
                    new_group.add(group)

            if len(new_group) > 0:
                module._add_new_infection_groups(person_id, new_group)

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

