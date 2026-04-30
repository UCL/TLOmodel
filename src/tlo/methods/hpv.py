from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import pandas as pd
import math

# from scripts.diarrhoea_analyses.analysis_diarrhoea_with_and_without_treatment import data
from tlo import DAYS_IN_YEAR, DateOffset, Module, Parameter, Property, Types, logging
from tlo.analysis.utils import get_counts_by_sex_and_age_group
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
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
    """This is an HPV infection Process.
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

    INIT_DEPENDENCIES = {'Demography', 'SymptomManager', 'Hiv'}

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
    AGE_BINS = [15, 20, 25, 35, 45, 55, 200]
    AGE_LABELS = ['15_19', '20_24', '25_34','35_44', '45_54', '55plus']

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

        # ------------------  HPV Transmission  ------------------ #
        # transmission coefficient for HPV Infection
        "b_hpv": Parameter(
            Types.REAL,
            "Baseline transmission coefficient for HPV Infection",
        ),

        # Modifiers
        # "rr_hpv_hiv": Parameter(
        #     Types.REAL,
        #     "Relative risk for HPV infection among HIV positive people",
        # ),
        "rr_hpv_hiv_no_art": Parameter(
            Types.REAL,
            "Relative risk for HPV acquisition among HIV positive people not on ART",
        ),
        "rr_hpv_hiv_art_unsuppressed": Parameter(
            Types.REAL,
            "Relative risk for HPV acquisition among HIV positive people on ART but not virally suppressed",
        ),
        "rr_hr1_vaccinated": Parameter(
            Types.REAL,
            "Relative risk for hr1 infection if vaccinated",
        ),
        "rr_hr2_vaccinated": Parameter(
            Types.REAL,
            "Relative risk for hr2 infection if vaccinated",
        ),
        "rr_hr3_vaccinated": Parameter(
            Types.REAL,
            "Relative risk for hr3 infection if vaccinated",
        ),

        "rr_hpv_age50plus": Parameter(
            Types.REAL,
            "Relative risk multiplier for age >=50",
        ),

        # ------------------  HPV Self-clear  ------------------ #
        # Weibull baseline
        "median_clear_hr1": Parameter(
            Types.REAL,
            "Median months to self-clear for hr1 infection",
        ),
        "median_clear_hr2": Parameter(
            Types.REAL,
            "Median months to self-clear for hr2 infection",
        ),
        "median_clear_hr3": Parameter(
            Types.REAL,
            "Median months to self-clear for hr3 infection",
        ),
        "clear_shape": Parameter(
            Types.REAL,
            "Weibull shape parameter for HPV clearance duration",
        ),

        # Modifiers
        "rr_clear_hiv_no_art": Parameter(
            Types.REAL,
            "Rate ratio for HPV clearance among PLWH not on ART",
        ),
        "rr_clear_hiv_art_unsuppressed": Parameter(
            Types.REAL,
            "Rate ratio for HPV clearance among PLWH on ART but not virally suppressed",
        ),

        ## As MC suggested, remove the immunity part
        # "rr_immunity_hr1": Parameter(
        #     Types.REAL,
        #     "Relative risk for reinfection with hr1 if previously infected",
        # ),
        # "rr_immunity_hr2": Parameter(
        #     Types.REAL,
        #     "Relative risk for reinfection with hr2 if previously infected",
        # ),
        # "rr_immunity_hr3": Parameter(
        #     Types.REAL,
        #     "Relative risk for reinfection with hr3 if previously infected",
        # ),
    }

    PROPERTIES = {
        'hp_is_infected': Property(
            Types.BOOL, 'Is infected with oncogenic hpv group'),
        'hp_infected_hr1': Property(
            Types.BOOL, 'Current infected with hr1'),
        'hp_infected_hr2': Property(
            Types.BOOL, 'Current infected with hr2'),
        'hp_infected_hr3': Property(
            Types.BOOL, 'Current infected with hr3'),
        'hp_date_infected_hr1': Property(
            Types.DATE, 'Date of infection of hr1'),
        'hp_date_infected_hr2': Property(
            Types.DATE, 'Date of infection of hr2'),
        'hp_date_infected_hr3': Property(
            Types.DATE, 'Date of infection of hr3'),
        'hp_date_first_infected': Property(
            Types.DATE, 'Start date of current HPV infection'),
        'hp_duration_hr1': Property(
            Types.INT,'Duration for current hr1 infection'),
        'hp_duration_hr2': Property(
            Types.INT,'Duration for current hr2 infection'),
        'hp_duration_hr3': Property(
            Types.INT,'Duration for current hr3 infection'),
        'hp_duration_all_clear': Property(
            Types.INT, 'Duration for current all HPV infection'),
        # 'hp_date_clear_hr1': Property(
        #     Types.DATE, 'Scheduled clearance date of current hr1 infection'),
        # 'hp_date_clear_hr2': Property(
        #     Types.DATE, 'Scheduled clearance date of current hr2 infection'),
        # 'hp_date_clear_hr3': Property(
        #     Types.DATE, 'Scheduled clearance date of current hr3 infection'),
        'hp_ever_infected_hr1': Property(
            Types.BOOL, 'Ever infected with hr1'),
        'hp_ever_infected_hr2': Property(
            Types.BOOL, 'Ever infected with hr2'),
        'hp_ever_infected_hr3': Property(
            Types.BOOL, 'Ever infected with hr3'),

        'hp_persistent_hr1': Property(
            Types.BOOL, 'Persistent hr1 infection, duration >= 12 months'),
        'hp_persistent_hr2': Property(
            Types.BOOL, 'Persistent hr2 infection, duration >= 12 months'),
        'hp_persistent_hr3': Property(
            Types.BOOL, 'Persistent hr3 infection, duration >= 12 months'),

        # "va_hpv": Property(Types.INT, "number of doses of hpv vaccine received"),
        # "va_hpv_all_doses": Property(Types.BOOL, "whether all doses have been received of the HPV vaccine"),
        # "hv_inf": Property(Types.BOOL,"Is person currently infected with HIV
        #     (NB. AIDS status is determined by presence of the AIDS Symptom.",),
        # "hv_art": Property(Types.CATEGORICAL,
        #     "ART status of person, whether on ART or not; and whether viral load is suppressed or not if on ART.",
        #     categories=["not", "on_VL_suppressed", "on_not_VL_suppressed"],),
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

        df = population.props  # a shortcut to the dataframe storing data for individuals

        # Set default for properties
        df.loc[df.is_alive, 'hp_is_infected'] = False  # default: no individuals infected
        df.loc[df.is_alive, 'hp_infected_hr1'] = False
        df.loc[df.is_alive, 'hp_infected_hr2'] = False
        df.loc[df.is_alive, 'hp_infected_hr3'] = False
        df.loc[df.is_alive, 'hp_date_infected_hr1'] = pd.NaT
        df.loc[df.is_alive, 'hp_date_infected_hr2'] = pd.NaT
        df.loc[df.is_alive, 'hp_date_infected_hr3'] = pd.NaT
        df.loc[df.is_alive, 'hp_date_first_infected'] = pd.NaT
        df.loc[df.is_alive, 'hp_duration_hr1'] = -1
        df.loc[df.is_alive, 'hp_duration_hr2'] = -1
        df.loc[df.is_alive, 'hp_duration_hr3'] = -1
        df.loc[df.is_alive, 'hp_duration_all_clear'] = -1
        # df.loc[df.is_alive, 'hp_date_clear_hr1'] = pd.NaT
        # df.loc[df.is_alive, 'hp_date_clear_hr2'] = pd.NaT
        # df.loc[df.is_alive, 'hp_date_clear_hr3'] = pd.NaT
        df.loc[df.is_alive, 'hp_ever_infected_hr1'] = False
        df.loc[df.is_alive, 'hp_ever_infected_hr2'] = False
        df.loc[df.is_alive, 'hp_ever_infected_hr3'] = False
        eligible = df.index[df.is_alive & (df.age_years >= 15)]
        df.loc[df.is_alive, 'hp_persistent_hr1'] = False
        df.loc[df.is_alive, 'hp_persistent_hr2'] = False
        df.loc[df.is_alive, 'hp_persistent_hr3'] = False

        for group in self.HPV_GROUPS:
            p_init = self.parameters[f'init_prev_hpv_{group}']
            u = self.rng.random(size=len(eligible))
            infected_this_group = eligible[u < p_init]

            for person_id in infected_this_group:
                df.at[person_id, f'hp_infected_{group}'] = True
                df.at[person_id, f'hp_ever_infected_{group}'] = True
                df.at[person_id, 'hp_is_infected'] = True

                # randomly select an infection date for initial population
                previous_infection = int(self.rng.randint(0, 24))  # 0-23month
                infection_date = self.sim.date - DateOffset(months=previous_infection)

                df.at[person_id, f'hp_date_infected_{group}'] = infection_date
                # df.at[person_id, f'hp_date_clear_{group}'] = pd.NaT
                df.at[person_id, f'hp_duration_{group}'] = previous_infection
                df.at[person_id, f'hp_persistent_{group}'] = previous_infection >= 12

        initially_infected = df.index[df.is_alive & df.hp_is_infected]
        for person_id in initially_infected:
            group_dates = []

            for group in self.HPV_GROUPS:
                date = df.at[person_id, f'hp_date_infected_{group}']

                if not pd.isna(date):
                    group_dates.append(date)

            if len(group_dates) > 0:
                df.at[person_id, 'hp_date_first_infected'] = min(group_dates)

    def _get_age_group_series(self, ages):
        return pd.cut(
            ages,
            bins=self.AGE_BINS,
            labels=self.AGE_LABELS,
            right=False  # right side not included
        )
    def _get_age_group(self,age_years):
        for i in range(len(self.AGE_BINS)-1):
            if self.AGE_BINS[i] <= age_years < self.AGE_BINS[i + 1]:
                return self.AGE_LABELS[i]
        return None

    def _build_age_mixing_matrix(self, within=0.80, adjacent=0.15, distant=0.05):
        labels = self.AGE_LABELS
        M = pd.DataFrame(0.0, index=labels, columns=labels, dtype=float)

        for i, label in enumerate(labels):
            row = pd.Series(0.0, index=labels, dtype=float)

            # within-group
            row[label] = within

            # adjacent groups
            neighbors = []
            if i -1 >= 0:
                neighbors.append(labels[i - 1])
            if i + 1 < len(labels):
                neighbors.append(labels[i + 1])

            if len(neighbors) > 0:
                share_adj = adjacent / len(neighbors)
                for nb in neighbors:
                    row[nb] = share_adj

            # distant group
            distant_groups = [x for x in labels if x != label and x not in neighbors]
            if len(distant_groups) > 0:
                share_dist = distant / len(distant_groups)
                for dg in distant_groups:
                    row[dg] = share_dist

            # normalize to exactly
            row = row / row.sum()
            M.loc[label] = row
        return M


    def _months_since(self,start_date,end_date=None):
        if pd.isna(start_date):
            return None
        if pd.isna(end_date):
            end_date = self.sim.date

        months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
        return max(0, int(months))

    def _get_clearance_rr(self, person_id):
        df = self.sim.population.props
        p = self.parameters

        if 'hv_inf' not in df.columns:
            return 1.0

        hv_inf = df.at[person_id, 'hv_inf']

        if pd.isna(hv_inf) or (hv_inf is False):
            return 1.0

        if 'hv_art' not in df.columns:
            return 1.0

        hv_art = df.at[person_id, 'hv_art']

        if hv_art == 'not':
            return p['rr_clear_hiv_no_art']
        elif hv_art == 'on_not_VL_suppressed':
            return p ['rr_clear_hiv_art_unsuppressed']
        else:
            return 1.0

    def _get_clearance_probability(self, group, person_id, duration_months, interval_months = 6):
        p = self.parameters

        median = p[f'median_clear_{group}']
        shape = p['clear_shape']

        # median = scale * (ln 2)^(1/shape)
        scale = median / (math.log(2) ** (1.0 / shape))

        t1 = max(0.0, float(duration_months))
        t0 = max(0.0, t1 - float(interval_months))

        # Weibull baseline cumulative hazard increment over [t0, t1]
        H0_t0 = (t0 / scale) ** shape
        H0_t1 = (t1 / scale) ** shape
        delta_H0 = max(0.0, H0_t1 - H0_t0)

        rr = self._get_clearance_rr(person_id)

        # p = 1 - exp(- rr * delta_H0)
        p_clear = 1.0 - math.exp(-rr * delta_H0)

        return min(max(p_clear, 0.0), 1.0)

    def _get_hpv_group_set(self, person_id):
        df = self.sim.population.props
        current_groups = set()

        for group in self.HPV_GROUPS:
            if df.at[person_id, f'hp_infected_{group}']:
                current_groups.add(group)

        return current_groups

    def _set_hpv_group_set(self,person_id, hpv_set):
        df = self.sim.population.props

        for group in self.HPV_GROUPS:
            df.at[person_id, f'hp_infected_{group}'] = (group in hpv_set)

        df.at[person_id, 'hp_is_infected'] = (len(hpv_set) > 0)

    def _add_new_infection_groups(self, person_id, new_groups):
        if len(new_groups) == 0:
            return

        df = self.sim.population.props
        was_infected_before = df.at[person_id, 'hp_is_infected']
        current_groups = self._get_hpv_group_set(person_id)
        updated_groups = current_groups.union(new_groups)
        self._set_hpv_group_set(person_id, updated_groups)

        # set infection date for new groups
        for group in new_groups:
            df.at[person_id, f'hp_date_infected_{group}'] = self.sim.date
            df.at[person_id, f'hp_ever_infected_{group}'] = True
            df.at[person_id,f'hp_duration_{group}'] = 0

        # start a new HPV infection process only if the person was uninfected/ self-clear
        if not was_infected_before:
            df.at[person_id, 'hp_date_first_infected'] = self.sim.date
            df.at[person_id, 'hp_duration_all_clear'] = -1

        # set first infection date
        if pd.isna(df.at[person_id, 'hp_date_first_infected']):
            df.at[person_id, 'hp_date_first_infected'] = self.sim.date


    def _clear_single_group(self, person_id, group):
        """clear a single HPV group for a person"""
        df = self.sim.population.props

        df.at[person_id, f'hp_infected_{group}'] = False
        df.at[person_id, f'hp_date_infected_{group}'] = pd.NaT
        # df.at[person_id, f'hp_date_clear_{group}'] = pd.NaT
        df.at[person_id, f'hp_duration_{group}'] = -1
        df.at[person_id, f'hp_persistent_{group}'] = False

        still_infected = any(
            df.at[person_id, f'hp_infected_{group}'] for group in self.HPV_GROUPS
        )
        df.at[person_id, 'hp_is_infected'] = still_infected

        if not still_infected:
            start_date = df.at[person_id, 'hp_date_first_infected']
            if not pd.isna(start_date):
                overall_duration = (self.sim.date.year - start_date.year) * 12 + (self.sim.date.month - start_date.month)
                df.at[person_id, 'hp_duration_all_clear'] = overall_duration

            df.at[person_id, 'hp_date_first_infected'] = pd.NaT

    # def _sample_clear_duration(self,group):
    #     """Sample a infection duration for one HPV group using Weibull distribution"""
    #     p = self.parameters
    #
    #     median = p[f'median_clear_{group}']
    #     shape = p['clear_shape']
    #
    #     # WeibullMedian= Scale * （ln 2)^(1/shape)
    #     scale = median / (math.log(2) ** (1.0 / shape))
    #
    #     u = self.rng.random()
    #     duration = scale * ((-math.log(1.0 - u)) ** (1.0 / shape))
    #
    #     duration = max(1,int(round(duration)))
    #     duration = min(duration, 48)
    #     return duration

    def _update_persistence_status(self, threshold_months=12):
        df = self.sim.population.props
        eligible = df.is_alive & (df.age_years >= 15)

        for group in self.HPV_GROUPS:
            inf_col = f'hp_infected_{group}'
            date_col = f'hp_date_infected_{group}'
            dur_col = f'hp_duration_{group}'
            pers_col = f'hp_persistent_{group}'

            non_infected = eligible & ~df[inf_col].fillna(False)
            df.loc[non_infected, dur_col] = -1
            df.loc[non_infected, pers_col] = False

            infected = eligible & df[inf_col].fillna(False)

            for person_id in df.index[infected]:
                date_inf = df.at[person_id, date_col]

                if pd.isna(date_inf):
                    df.at[person_id, dur_col] = -1
                    df.at[person_id, pers_col] = False
                    continue

                duration = self._months_since(date_inf, self.sim.date)

                df.at[person_id, dur_col] = duration
                df.at[person_id, pers_col] = duration >= threshold_months

    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """
        p = self.parameters
        self.lm = {}
        self.age_mixing_matrix = self._build_age_mixing_matrix(
            within=0.80,
            adjacent=0.15,
            distant=0.05
        )
        self._pre_logged_prev = {}

        for group in self.HPV_GROUPS:
            self.lm[group] = LinearModel(
                LinearModelType.MULTIPLICATIVE,
                1.0,

                Predictor('va_hpv')
                .when(1, p[f'rr_{group}_vaccinated'])
                .when(2, p[f'rr_{group}_vaccinated']),

                # Predictor(f'hp_ever_infected_{group}')
                # .when(True, p[f'rr_immunity_{group}'])

                Predictor('age_years', conditions_are_mutually_exclusive=True)
                .when('<15', 0.0)
                .when('>=50', p['rr_hpv_age50plus']),

                Predictor()
                .when('hv_inf & (hv_art =="not")',
                p['rr_hpv_hiv_no_art']
                )
                .when(
                    'hv_inf & (hv_art == "on_not_VL_suppressed")',
                    p['rr_hpv_hiv_art_unsuppressed']
                ),
            )

        # add the basic event
        event = HpvInfectionEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=6))

        # add an event to log to screen
        sim.schedule_event(HpvLoggingEvent(self), sim.date + DateOffset(months=6, days=1))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the ID for the mother for this child
        :param child_id: the ID for the new child
        """

        df = self.sim.population.props  # shortcut to the population props dataframe
        df.at[child_id, 'hp_is_infected'] = False
        df.at[child_id, 'hp_infected_hr1'] = False
        df.at[child_id, 'hp_infected_hr2'] = False
        df.at[child_id, 'hp_infected_hr3'] = False
        df.at[child_id, 'hp_date_infected_hr1'] = pd.NaT
        df.at[child_id, 'hp_date_infected_hr2'] = pd.NaT
        df.at[child_id, 'hp_date_infected_hr3'] = pd.NaT
        df.at[child_id, 'hp_date_first_infected'] = pd.NaT
        df.at[child_id, 'hp_duration_hr1'] = -1
        df.at[child_id, 'hp_duration_hr2'] = -1
        df.at[child_id, 'hp_duration_hr3'] = -1
        df.at[child_id, 'hp_duration_all_clear'] = -1
        # df.at[child_id, 'hp_date_clear_hr1'] = pd.NaT
        # df.at[child_id, 'hp_date_clear_hr2'] = pd.NaT
        # df.at[child_id, 'hp_date_clear_hr3'] = pd.NaT
        df.at[child_id, 'hp_ever_infected_hr1'] = False
        df.at[child_id, 'hp_ever_infected_hr2'] = False
        df.at[child_id, 'hp_ever_infected_hr3'] = False
        df.at[child_id, 'hp_persistent_hr1'] = False
        df.at[child_id, 'hp_persistent_hr2'] = False
        df.at[child_id, 'hp_persistent_hr3'] = False

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
        def report_summary_stats(self):
            df = self.sim.population.props
            summary = {
                'infected_any': get_counts_by_sex_and_age_group(df, 'hp_is_infected')}

            for group in self.HPV_GROUPS:
                summary[f'infected_{group}'] = get_counts_by_sex_and_age_group(df, f'hp_infected_{group}')
                summary[f'persistent_{group}'] = get_counts_by_sex_and_age_group(df, f'hp_persistent_{group}')

        return summary

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
            current_groups = module._get_hpv_group_set(person_id)

            for group in list(current_groups):
                date_inf = df.at[person_id, f'hp_date_infected_{group}']
                if pd.isna(date_inf):
                    continue

                duration_months = module._months_since (date_inf, now)
                if duration_months is None:
                    continue

                df.at[person_id, f'hp_duration_{group}'] = duration_months

                # Clear rate in 6 months
                interval_months = 6
                p_clear = module._get_clearance_probability(
                    group=group,
                    person_id=person_id,
                    duration_months=duration_months,
                    interval_months=interval_months
                )

                if module.rng.random() < p_clear:
                    module._clear_single_group(person_id, group)


        # 3. recalculate prevalence by HPV group after clearance
        df_alive = df.loc[df.is_alive & (df.age_years >= 15)].copy()
        df_alive['age_group'] = module._get_age_group_series(df_alive['age_years'])

        male_df = df_alive.loc[df_alive.sex == 'M']
        female_df = df_alive.loc[df_alive.sex == 'F']

        prev_by_age_male = {}
        prev_by_age_female = {}

        for group in module.HPV_GROUPS:
            prev_by_age_male[group] = (
                male_df.groupby('age_group', observed=True)[f'hp_infected_{group}']
                .mean()
                .reindex(module.AGE_LABELS, fill_value=0.0)
            )
            prev_by_age_female[group] = (
                female_df.groupby('age_group', observed=True)[f'hp_infected_{group}']
                .mean()
                .reindex(module.AGE_LABELS, fill_value=0.0)
            )

            # male_group_inf = df.loc[male_idx, f'hp_infected_{group}'].sum()
            # female_group_inf = df.loc[female_idx, f'hp_infected_{group}'].sum()
            #
            # prev_male[group] = male_group_inf/len(male_idx) if len(male_idx) > 0 else 0
            # prev_female[group] = female_group_inf / len(female_idx) if len(female_idx) > 0 else 0

        # 4. new infection
        interval_years = 0.5

        for person_id in eligible:
            sex = df.at[person_id,'sex']
            current_groups = module._get_hpv_group_set(person_id)
            new_group = set()

            if sex == 'F':
                source_prev_by_age = prev_by_age_male
            elif sex == 'M':
                source_prev_by_age = prev_by_age_female
            else:
                continue

            my_age_group = module._get_age_group(df.at[person_id,'age_years'])
            if my_age_group is None:
                continue

            mix_row = module.age_mixing_matrix.loc[my_age_group]

            for group in module.HPV_GROUPS:
                if group in current_groups:
                    continue

                weighted_prev = float((mix_row * source_prev_by_age[group]).sum())

                beta_name = f'b_hpv_{group}'
                beta = module.parameters[beta_name] if beta_name in module.parameters else module.parameters['b_hpv']

                modifier = module.lm[group].predict(df.loc[[person_id]]).iloc[0]

                lambda_inf = beta * weighted_prev * modifier
                lambda_inf = max(lambda_inf, 0.0)

                p_inf = 1.0 - math.exp(-lambda_inf * interval_years)
                p_inf = min(max(p_inf, 0.0), 1.0)

                if module.rng.random() < p_inf:
                    new_group.add(group)

            if len(new_group) > 0:
                module._add_new_infection_groups(person_id, new_group)

        module._update_persistence_status(threshold_months=12)

class HpvLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """Produce a summmary of the numbers of people with respect to their 'hpv status'
        """
        # run this event every 6/12 month
        self.repeat = 6
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        assert isinstance(module, HPV)

    def apply(self, population):
        # get some summary statistics
        df = population.props
        module = self.module

        eligible = df.index[df.is_alive & (df.age_years >= 15)]
        log_data = {
            'Year':self.sim.date.year,
            'Month':self.sim.date.month,
            'EligibleN':int(len(eligible)),
        }

        if len(eligible) == 0:
            logger.info(key='summary', data=log_data)
            return

        sub = df.loc[eligible].copy()

        sub['age_group'] = module._get_age_group_series(sub['age_years'])
        sub['hiv_group'] = 'HIVneg'

        if 'hv_inf' in sub.columns:
            sub.loc[sub['hv_inf'].fillna(False), 'hiv_group'] = 'HIVpos_unknown'

            if 'hv_art' in sub.columns:
                sub.loc[sub['hv_inf'].fillna(False) & (sub['hv_art'] == 'not'), 'hiv_group'] = 'HIVpos_noART'
                sub.loc[sub['hv_inf'].fillna(False) & (
                        sub['hv_art'] == 'on_not_VL_suppressed'), 'hiv_group'] = 'HIVpos_unsupp'
                sub.loc[
                    sub['hv_inf'].fillna(False) & (sub['hv_art'] == 'on_VL_suppressed'), 'hiv_group'] = 'HIVpos_supp'

        # 1. Overall summary
        total_inf = int(sub['hp_is_infected'].sum())
        log_data['TotalInf'] = total_inf
        log_data['TotalPrev'] = sub['hp_is_infected'].mean()

        for sex_name, sex_df in [('M', sub.loc[sub.sex == 'M']),
                                 ('F', sub.loc[sub.sex == 'F'])]:
            n = len(sex_df)
            log_data[f'{sex_name}_N'] = int(n)
            log_data[f'{sex_name}_Inf'] = int(sex_df['hp_is_infected'].sum()) if n > 0 else 0
            log_data[f'{sex_name}_Prev'] = sex_df['hp_is_infected'].mean() if n > 0 else math.nan

        # 2. Prevalence by sex and age group
        prev_snapshot = {}

        for sex_name, sex_df in [('All', sub),
                                 ('M', sub.loc[sub.sex == 'M']),
                                 ('F', sub.loc[sub.sex == 'F'])]:

            for age_group in module.AGE_LABELS:
                age_df = sex_df.loc[sex_df['age_group'] == age_group]
                n = len(age_df)

                log_data[f'Any_{sex_name}_{age_group}_N'] = int(n)

                if n == 0:
                    log_data[f'Any_{sex_name}_{age_group}_Inf'] = 0
                    log_data[f'Any_{sex_name}_{age_group}_Prev'] = math.nan
                    for hpv_group in module.HPV_GROUPS:
                        log_data[f'{hpv_group}_{sex_name}_{age_group}_Inf'] = 0
                        log_data[f'{hpv_group}_{sex_name}_{age_group}_Prev'] = math.nan
                    continue

                any_inf = int(age_df['hp_is_infected'].sum())
                any_prev = age_df['hp_is_infected'].mean()

                log_data[f'Any_{sex_name}_{age_group}_Inf'] = any_inf
                log_data[f'Any_{sex_name}_{age_group}_Prev'] = any_prev
                prev_snapshot[f'Any_{sex_name}_{age_group}_Prev'] = any_prev

                for hpv_group in module.HPV_GROUPS:
                    inf_n = int(age_df[f'hp_infected_{hpv_group}'].sum())
                    prev = age_df[f'hp_infected_{hpv_group}'].mean()

                    log_data[f'{hpv_group}_{sex_name}_{age_group}_Inf'] = inf_n
                    log_data[f'{hpv_group}_{sex_name}_{age_group}_Prev'] = prev
                    prev_snapshot[f'{hpv_group}_{sex_name}_{age_group}_Prev'] = prev

        # 3. HIV
        for hiv_group, hiv_df in sub.groupby('hiv_group', observed=True):
            n = len(hiv_df)
            log_data[f'Any_{hiv_group}_N'] = int(n)
            log_data[f'Any_{hiv_group}_Inf'] = int(hiv_df['hp_is_infected'].sum()) if n > 0 else 0
            log_data[f'Any_{hiv_group}_Prev'] = hiv_df['hp_is_infected'].mean() if n > 0 else math.nan

            for hpv_group in module.HPV_GROUPS:
                log_data[f'{hpv_group}_{hiv_group}_Prev'] = (
                    hiv_df[f'hp_infected_{hpv_group}'].mean() if n > 0 else math.nan
                )

        # 4. Delta
        prev_logged = getattr(module, '_pre_logged_prev', {})
        for key, current_val in prev_snapshot.items():
            previous_val = prev_logged.get(key, math.nan)
            if pd.isna(previous_val) or pd.isna(current_val):
                log_data[f'{key}_Delta'] = math.nan
            else:
                log_data[f'{key}_Delta'] = current_val - previous_val

        module._pre_logged_prev = prev_snapshot

        # 5. multiplicity of infection
        infection_people = df.index[df.is_alive & (df.age_years >= 15) & df.hp_is_infected]
        n_group_1 = 0
        n_group_2 = 0
        n_group_3 = 0

        male_n_group_1 = 0
        male_n_group_2 = 0
        male_n_group_3 = 0

        female_n_group_1 = 0
        female_n_group_2 = 0
        female_n_group_3 = 0

        for person_id in infection_people:
            n_group = len(module._get_hpv_group_set(person_id))
            sex = df.at[person_id, 'sex']

            if n_group == 1:
                n_group_1 += 1
                if sex == 'M':
                    male_n_group_1 += 1
                elif sex =='F':
                    female_n_group_1 += 1

            elif n_group == 2:
                n_group_2 += 1
                if sex == 'M':
                    male_n_group_2 += 1
                elif sex =='F':
                    female_n_group_2 += 1

            elif n_group == 3:
                n_group_3 += 1
                if sex == 'M':
                    male_n_group_3 += 1
                elif sex =='F':
                    female_n_group_3 += 1

        log_data['InfGroup1'] = n_group_1
        log_data['InfGroup2'] = n_group_2
        log_data['InfGroup3'] = n_group_3

        log_data['MaleGroup1'] = male_n_group_1
        log_data['MaleGroup2'] = male_n_group_2
        log_data['MaleGroup3'] = male_n_group_3

        log_data['FemaleGroup1'] = female_n_group_1
        log_data['FemaleGroup2'] = female_n_group_2
        log_data['FemaleGroup3'] = female_n_group_3

        # 6. Persistent infection 统计
        for hpv_group in module.HPV_GROUPS:
            pers_col = f'hp_persistent_{hpv_group}'

            if pers_col not in sub.columns:
                continue

            persistent = sub[pers_col].fillna(False)

            log_data[f'{hpv_group}_Persistent12_N'] = int(persistent.sum())
            log_data[f'{hpv_group}_Persistent12_Prev'] = float(persistent.mean())

            for sex_name, sex_df in [('M', sub.loc[sub.sex == 'M']),
                                     ('F', sub.loc[sub.sex == 'F'])]:
                n = len(sex_df)
                if n > 0:
                    log_data[f'{hpv_group}_Persistent12_{sex_name}_Prev'] = float(
                        sex_df[pers_col].fillna(False).mean()
                    )
                else:
                    log_data[f'{hpv_group}_Persistent12_{sex_name}_Prev'] = math.nan

            for age_group in module.AGE_LABELS:
                age_df = sub.loc[sub['age_group'] == age_group]
                n = len(age_df)
                if n > 0:
                    log_data[f'{hpv_group}_Persistent12_{age_group}_Prev'] = float(
                        age_df[pers_col].fillna(False).mean()
                    )
                else:
                    log_data[f'{hpv_group}_Persistent12_{age_group}_Prev'] = math.nan

        logger.info(key='summary', data=log_data)
