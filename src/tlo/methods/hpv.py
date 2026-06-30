from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import pandas as pd
import math
import numpy as np

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
        hr1 = HPV16
        hr2 = HPV18/45
        hr3 = HPV31/33/35/52/58
        hr4 = other high-risk HPV

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

    HPV_GROUPS = ['hr1', 'hr2', 'hr3','hr4']
    AGE_BINS = [15, 20, 25, 35, 45, 55, 200]
    AGE_LABELS = ['15_19', '20_24', '25_34', '35_44', '45_54', '55plus']

    PARAMETERS = {
        # ------------------ Initial prevalence ------------------ #
        "init_prev_hpv_hr1": Parameter(
            Types.REAL,
            "Initial prevalence of hpv 16 infection",
        ),
        "init_prev_hpv_hr2": Parameter(
            Types.REAL,
            "Initial prevalence of HPV 18/45 infection",
        ),
        "init_prev_hpv_hr3": Parameter(
            Types.REAL,
            "Initial prevalence of HPV 31/33/52/58/35 infection"
        ),
        "init_prev_hpv_hr4": Parameter(
            Types.REAL,
            "Initial prevalence of other hr-HPV infection",
        ),

        # ------------------  HPV Transmission  ------------------ #
        # transmission coefficient for HPV Infection
        "b_hpv": Parameter(
            Types.REAL,
            "Baseline transmission coefficient for HPV Infection",
        ),

        # Modifiers
        "rr_hpv_hiv_no_art": Parameter(
            Types.REAL,
            "Relative risk for HPV acquisition among HIV positive people who are not virally suppressed",
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
        "rr_hr4_vaccinated": Parameter(
            Types.REAL,
            "Relative risk for hr4 infection if vaccinated",
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
        "median_clear_hr4": Parameter(
            Types.REAL,
            "Median months to self-clear for hr4 infection",
        ),
        "clear_shape": Parameter(
            Types.REAL,
            "Weibull shape parameter for HPV clearance duration",
        ),

        # Modifiers
        "rr_clear_hiv_no_art": Parameter(
            Types.REAL,
            "Rate ratio for HPV clearance among PLWH not on ART or not virally suppressed",
        ),

        # age-mixing
        "age_mixing_within": Parameter(
            Types.REAL,
            "Proportion of sexual mixing occurring within the same age group",
        ),
        "age_mixing_adjacent": Parameter(
            Types.REAL,
            "Proportion of sexual mixing occurring with adjacent age groups",
        ),
        "age_mixing_distant": Parameter(
            Types.REAL,
            "Proportion of sexual mixing occurring with non-adjacent age groups",
        ),
        "hpv_event_frequency_months": Parameter(
            Types.INT,
            "Frequency in months for updating HPV infection and clearance events",
        ),
        "hpv_logging_frequency_months": Parameter(
            Types.INT,
            "Frequency in months for logging events",
        ),
        "persistent_threshold_months": Parameter(
            Types.REAL,
            "Duration threshold in months for defining persistent HPV infection",
        ),
    }

    PROPERTIES = {
        'hp_date_infected_hr1': Property(
            Types.DATE, 'Date of infection of hr1'),
        'hp_date_infected_hr2': Property(
            Types.DATE, 'Date of infection of hr2'),
        'hp_date_infected_hr3': Property(
            Types.DATE, 'Date of infection of hr3'),
        'hp_date_infected_hr4': Property(
            Types.DATE, 'Date of infection of hr4'),
        'hp_duration_hr1': Property(
            Types.REAL, 'Duration for current hr1 infection'),
        'hp_duration_hr2': Property(
            Types.REAL, 'Duration for current hr2 infection'),
        'hp_duration_hr3': Property(
            Types.REAL, 'Duration for current hr3 infection'),
        'hp_duration_hr4': Property(
            Types.REAL, 'Duration for current hr4 infection'),
        'hp_duration_all_clear': Property(
            Types.REAL, 'Duration for current all HPV infection'),
        'hp_persistent_hr1': Property(
            Types.BOOL, 'Persistent hr1 infection, duration >= 12 months'),
        'hp_persistent_hr2': Property(
            Types.BOOL, 'Persistent hr2 infection, duration >= 12 months'),
        'hp_persistent_hr3': Property(
            Types.BOOL, 'Persistent hr3 infection, duration >= 12 months'),
        'hp_persistent_hr4': Property(
            Types.BOOL, 'Persistent hr4 infection, duration >= 12 months'),
    }

    def __init__(self, name=None):
        super().__init__(name)

    def read_parameters(self, resourcefilepath: Optional[Path] = None):
        """Read in parameters and do the registration of this module and its symptoms"""
        self.load_parameters_from_dataframe(
            read_csv_files(Path(resourcefilepath) / "ResourceFile_HPV",
                           files="parameter_values")
        )

    def _get_group_infected_series(self, group, index=None):
        df = self.sim.population.props
        date_col = f'hp_date_infected_{group}'
        infected = df[date_col].notna()

        if index is not None:
            infected = infected.loc[index]

        return infected.fillna(False).astype(bool)

    def _get_hpv_any_infected_series(self, index=None):
        df = self.sim.population.props
        date_cols = [f'hp_date_infected_{group}' for group in self.HPV_GROUPS]
        infected_any = df[date_cols].notna().any(axis=1)

        if index is not None:
            infected_any = infected_any.loc[index]

        return infected_any.fillna(False).astype(bool)

    def _hp_is_infected(self, person_id):
        df = self.sim.population.props
        for group in self.HPV_GROUPS:
            if not pd.isna(df.at[person_id, f'hp_date_infected_{group}']):
                return True
        return False

    def _is_group_infected(self, person_id, group):
        df = self.sim.population.props
        return not pd.isna(df.at[person_id, f'hp_date_infected_{group}'])

    def _get_first_infection_date(self, person_id):
        df = self.sim.population.props
        infection_dates = []

        for group in self.HPV_GROUPS:
            date_inf = df.at[person_id, f'hp_date_infected_{group}']
            if not pd.isna(date_inf):
                infection_dates.append(date_inf)

        return min(infection_dates) if infection_dates else pd.NaT

    def _get_hpv_group_set(self, person_id):
        df = self.sim.population.props
        current_groups = set()

        for group in self.HPV_GROUPS:
            if self._is_group_infected(person_id, group):
                current_groups.add(group)

        return current_groups

    def initialise_population(self, population):
        df = population.props  # a shortcut to the dataframe storing data for individuals

        # Set default for properties
        df.loc[df.is_alive, f'hp_duration_all_clear'] = -1
        for group in self.HPV_GROUPS:
            df.loc[df.is_alive, f'hp_date_infected_{group}'] = pd.NaT
            df.loc[df.is_alive, f'hp_duration_{group}'] = -1
            df.loc[df.is_alive, f'hp_persistent_{group}'] = False

        eligible = df.index[df.is_alive & (df.age_years >= 15)]

        for group in self.HPV_GROUPS:
            p_init = self.parameters[f'init_prev_hpv_{group}']
            u = self.rng.random(size=len(eligible))
            infected_this_group = eligible[u < p_init]

            if len(infected_this_group) == 0:
                continue

            previous_infection = self.rng.randint(
                low=0,
                high=24,
                size=len(infected_this_group),)

            infection_dates = [self.sim.date - DateOffset(months=int(months))
                for months in previous_infection
            ]

            df.loc[infected_this_group, f'hp_date_infected_{group}'] = infection_dates
            df.loc[infected_this_group, f'hp_duration_{group}'] = previous_infection.astype(float)
            df.loc[infected_this_group, f'hp_persistent_{group}'] = (previous_infection >= self.parameters['persistent_threshold_months'])

    # Age mixing
    def _get_age_group_series(self, ages):
        return pd.cut(
            ages,
            bins=self.AGE_BINS,
            labels=self.AGE_LABELS,
            right=False  # right side not included
        )
    def _get_age_group(self, age_years):
        for i in range(len(self.AGE_BINS)-1):
            if self.AGE_BINS[i] <= age_years < self.AGE_BINS[i + 1]:
                return self.AGE_LABELS[i]
        return None

    def _build_age_mixing_matrix(self, within, adjacent, distant):
        labels = self.AGE_LABELS
        total = within + adjacent + distant
        if not np.isclose(total, 1.0):
            raise ValueError(
                f"Age mixing parameters must sum to 1.0, but received {total}."
            )

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

    # Time and clearance functions
    def _months_since(self,start_date,end_date=None):
        if pd.isna(start_date):
            return None

        if end_date is None or pd.isna(end_date):
            end_date = self.sim.date

        return max(0.0, (end_date - start_date).days / 30.5)

    def _get_infection_rr(self, person_id, group):
        df = self.sim.population.props
        p = self.parameters

        modifier = 1.0

        age = df.at[person_id, 'age_years']
        if pd.isna(age) or age < 15:
            return 0.0

        if age >= 50:
            modifier *= p['rr_hpv_age50plus']

        if 'va_hpv' in df.columns:
            va_hpv = df.at[person_id, 'va_hpv']
            if va_hpv in [1, 2]:
                modifier *= p[f'rr_{group}_vaccinated']

        if 'hv_inf' in df.columns:
            hv_inf = df.at[person_id, 'hv_inf']
            if (not pd.isna(hv_inf)) and bool(hv_inf):
                if 'hv_art' in df.columns:
                    hv_art = df.at[person_id, 'hv_art']
                    if hv_art in ['not', 'on_not_VL_suppressed']:
                        modifier *= p['rr_hpv_hiv_no_art']
                else:
                    modifier *= p['rr_hpv_hiv_no_art']

        return max(0.0, float(modifier))

    def _get_clearance_rr(self, person_id):
        df = self.sim.population.props
        p = self.parameters

        # If HIV module is not registered, assume no HIV-related effect on HPV clearance
        if 'Hiv' not in self.sim.modules:
            return 1.0

        if 'hv_inf' not in df.columns:
            return 1.0

        hv_inf = df.at[person_id, 'hv_inf']

        if pd.isna(hv_inf) or (not bool(hv_inf)):
            return 1.0

        if 'hv_art' not in df.columns:
            return p['rr_clear_hiv_no_art']

        hv_art = df.at[person_id, 'hv_art']

        if hv_art in ['not', 'on_not_VL_suppressed']:
            return p['rr_clear_hiv_no_art']

        return 1.0

    def _get_clearance_probability(self, group, person_id, duration_months, interval_months):
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

    def _add_new_infection_groups(self, person_id, new_groups):
        if len(new_groups) == 0:
            return

        df = self.sim.population.props
        was_infected_before = self._hp_is_infected(person_id)

        # set infection date for new groups
        for group in new_groups:
            if self._is_group_infected(person_id, group):
                continue
            df.at[person_id, f'hp_date_infected_{group}'] = self.sim.date
            df.at[person_id,f'hp_duration_{group}'] = 0
            df.at[person_id, f'hp_persistent_{group}'] = False

        # start a new HPV infection process only if the person was uninfected/ self-clear
        if not was_infected_before:
            df.at[person_id, 'hp_duration_all_clear'] = -1

    def _clear_single_group(self, person_id, group):
        """clear a single HPV group for a person"""
        df = self.sim.population.props
        first_infection_date = self._get_first_infection_date(person_id)

        df.at[person_id, f'hp_date_infected_{group}'] = pd.NaT
        df.at[person_id, f'hp_duration_{group}'] = -1
        df.at[person_id, f'hp_persistent_{group}'] = False

        still_infected = self._hp_is_infected(person_id)

        if not still_infected:
            overall_duration = self._months_since(first_infection_date, self.sim.date)
            if overall_duration is not None and not pd.isna(overall_duration):
                df.at[person_id, f'hp_duration_all_clear'] = float(overall_duration)
            else:
                df.at[person_id, f'hp_duration_all_clear'] = -1.0

    def _update_persistence_status(self):
        df = self.sim.population.props
        threshold_months = self.parameters['persistent_threshold_months']
        eligible = df.is_alive & (df.age_years >= 15)

        for group in self.HPV_GROUPS:
            date_col = f'hp_date_infected_{group}'
            dur_col = f'hp_duration_{group}'
            pers_col = f'hp_persistent_{group}'

            ineligible = ~eligible
            df.loc[ineligible, dur_col] = -1.0
            df.loc[ineligible, pers_col] = False

            non_infected = eligible & df[date_col].isna()
            df.loc[non_infected, dur_col] = -1
            df.loc[non_infected, pers_col] = False

            infected = eligible & df[date_col].notna()
            infected_idx = df.index[infected]

            for person_id in infected_idx:
                date_inf = df.at[person_id, date_col]
                duration = self._months_since(date_inf, self.sim.date)

                if duration is None or pd.isna(duration):
                    df.at[person_id, dur_col] = -1.0
                    df.at[person_id, pers_col] = False
                    continue

                duration = float(duration)
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
            within=p['age_mixing_within'],
            adjacent=p['age_mixing_adjacent'],
            distant=p['age_mixing_distant']
        )
        self._pre_logged_prev = {}

        self._last_event_counts = {}

        event = HpvInfectionEvent(
            self,
            frequency_months=int(p['hpv_event_frequency_months'])
        )
        sim.schedule_event(
            event,
            sim.date + DateOffset(months=int(p['hpv_event_frequency_months']))
        )

        sim.schedule_event(
            HpvLoggingEvent(self,frequency_months=int(p['hpv_logging_frequency_months'])),
            sim.date + DateOffset(months=int(p['hpv_logging_frequency_months']))
        )

    def on_birth(self, mother_id, child_id):

        df = self.sim.population.props  # shortcut to the population props dataframe
        for group in self.HPV_GROUPS:
            df.at[child_id, f'hp_date_infected_{group}'] = pd.NaT
            df.at[child_id, f'hp_duration_{group}'] = -1
            df.at[child_id, 'hp_duration_all_clear'] = -1
            df.at[child_id, f'hp_persistent_{group}'] = False

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
        self._update_persistence_status()

        df_report = df.copy()
        df_report['hp_is_infected'] = self._get_hpv_any_infected_series(index=df_report.index)

        summary = {
            'infected_any': get_counts_by_sex_and_age_group(df_report, 'hp_is_infected')}

        for group in self.HPV_GROUPS:
            temp_col = f'hp_infected_{group}'
            df_report[temp_col] = self._get_group_infected_series(group, index=df_report.index)
            summary[f'infected_{group}'] = get_counts_by_sex_and_age_group(df_report, temp_col)
            summary[f'persistent_{group}'] = get_counts_by_sex_and_age_group(df_report, f'hp_persistent_{group}')

        return summary

class HpvInfectionEvent(RegularEvent, PopulationScopeEventMixin):
    """This event is occurring regularly at one 6 months intervals and controls the infection process of HPV."""

    def __init__(self, module, frequency_months):
        self.frequency_months = int(frequency_months)
        super().__init__(module, frequency=DateOffset(months=self.frequency_months))
        assert isinstance(module, HPV)

    def apply(self, population):
        logger.debug(key='debug', data='This is HpvInfectionEvent, tracking the disease progression of the population.')
        df = population.props
        module = self.module
        now = self.sim.date

        event_counts = {
            'NewInf_Total': 0,
            'Clear_Total': 0,
        }

        # HPV group-specific counters
        for group in module.HPV_GROUPS:
            event_counts[f'NewInf_{group}'] = 0
            event_counts[f'Clear_{group}'] = 0

        # Sex-specific counters
        for sex_name in ['M', 'F']:
            event_counts[f'NewInf_{sex_name}'] = 0
            event_counts[f'Clear_{sex_name}'] = 0

        # Age-group-specific counters
        for age_group in module.AGE_LABELS:
            event_counts[f'NewInf_{age_group}'] = 0
            event_counts[f'Clear_{age_group}'] = 0

        # 1. define eligible population
        eligible = df.index[df.is_alive & (df.age_years >= 15)]
        if len(eligible) == 0:
            module._last_event_counts = event_counts
            return

        interval_months = float(self.frequency_months)
        interval_years = self.frequency_months / 12.0

        # 2. self-clearance
        infected_any = module._get_hpv_any_infected_series()
        infected_idx = df.index[df.is_alive & (df.age_years >= 15)& infected_any]

        for person_id in infected_idx:
            current_groups = module._get_hpv_group_set(person_id)

            for group in list(current_groups):
                date_inf = df.at[person_id, f'hp_date_infected_{group}']
                if pd.isna(date_inf):
                    continue

                duration_months = module._months_since (date_inf, now)
                if duration_months is None:
                    continue

                df.at[person_id, f'hp_duration_{group}'] = float(duration_months)

                p_clear = module._get_clearance_probability(
                    group=group,
                    person_id=person_id,
                    duration_months=duration_months,
                    interval_months=float(self.frequency_months)
                )

                if module.rng.random() < p_clear:
                    sex = df.at[person_id, 'sex']
                    age_group = module._get_age_group(df.at[person_id, 'age_years'])

                    event_counts['Clear_Total'] += 1
                    event_counts[f'Clear_{group}'] += 1

                    if sex in ['M', 'F']:
                        event_counts[f'Clear_{sex}'] += 1

                    if age_group in module.AGE_LABELS:
                        event_counts[f'Clear_{age_group}'] += 1
                    module._clear_single_group(person_id, group)

        module._update_persistence_status()

        # 3. recalculate prevalence by HPV group after clearance
        df_alive = df.loc[df.is_alive & (df.age_years >= 15)].copy()
        df_alive['age_group'] = module._get_age_group_series(df_alive['age_years'])

        for group in module.HPV_GROUPS:
            df_alive[f'hp_infected_{group}'] = module._get_group_infected_series(
                group,
                index=df_alive.index
            )

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

        # 4. new infection
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

                modifier = module._get_infection_rr(person_id, group)

                lambda_inf = beta * weighted_prev * modifier
                lambda_inf = max(lambda_inf, 0.0)

                p_inf = 1.0 - math.exp(-lambda_inf * interval_years)
                p_inf = min(max(p_inf, 0.0), 1.0)

                if module.rng.random() < p_inf:
                    new_group.add(group)

            if len(new_group) > 0:
                sex = df.at[person_id, 'sex']
                age_group = module._get_age_group(df.at[person_id, 'age_years'])

                for group in new_group:
                    event_counts['NewInf_Total'] += 1
                    event_counts[f'NewInf_{group}'] += 1

                    if sex in ['M', 'F']:
                        event_counts[f'NewInf_{sex}'] += 1

                    if age_group in module.AGE_LABELS:
                        event_counts[f'NewInf_{age_group}'] += 1
                module._add_new_infection_groups(person_id, new_group)

        module._update_persistence_status()
        module._last_event_counts = event_counts

class HpvLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module, frequency_months):
        """Produce a summmary of the numbers of people with respect to their 'hpv status'"""
        self.repeat = int(frequency_months)
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        assert isinstance(module, HPV)

    def apply(self, population):
        # get some summary statistics
        df = population.props
        module = self.module

        module._update_persistence_status()

        eligible = df.index[df.is_alive & (df.age_years >= 15)]
        log_data = {'EligibleN':int(len(eligible)),}
        alive = df.loc[df.is_alive].copy()

        if 'va_hpv' in alive.columns:
            alive['hpv_vaccinated'] = alive['va_hpv'].isin([1, 2])

            girls_9_14 = alive.loc[
                (alive.sex == 'F') &
                (alive.age_years >= 9) &
                (alive.age_years < 15)
                ]

            n_girls_9_14 = len(girls_9_14)

            log_data['HPVVaccinated_F_9_14_N'] = (
                int(girls_9_14['hpv_vaccinated'].sum())
                if n_girls_9_14 > 0 else 0
            )

            log_data['HPVVaccinated_F_9_14_Coverage'] = (
                float(girls_9_14['hpv_vaccinated'].mean())
                if n_girls_9_14 > 0 else math.nan
            )

            log_data['HPVVaccinated_F_9_14_Denominator'] = int(n_girls_9_14)

        else:
            log_data['HPVVaccinated_F_9_14_N'] = 0
            log_data['HPVVaccinated_F_9_14_Coverage'] = math.nan
            log_data['HPVVaccinated_F_9_14_Denominator'] = 0

        if len(eligible) == 0:
            logger.info(key='summary', data=log_data)
            return


        sub = df.loc[eligible].copy()
        sub['hp_is_infected'] = module._get_hpv_any_infected_series(index=sub.index)

        if 'va_hpv' in sub.columns:
            sub['hpv_vaccinated'] = sub['va_hpv'].isin([1, 2])

            log_data['HPVVaccinated_N'] = int(sub['hpv_vaccinated'].sum())
            log_data['HPVVaccinated_Coverage'] = float(sub['hpv_vaccinated'].mean())

            for sex_name, sex_df in [('M', sub.loc[sub.sex == 'M']),
                                     ('F', sub.loc[sub.sex == 'F'])]:
                n = len(sex_df)
                log_data[f'HPVVaccinated_{sex_name}_N'] = int(sex_df['hpv_vaccinated'].sum()) if n > 0 else 0
                log_data[f'HPVVaccinated_{sex_name}_Coverage'] = float(
                    sex_df['hpv_vaccinated'].mean()) if n > 0 else math.nan

            # Age-specific vaccine coverage
            sub['age_group'] = module._get_age_group_series(sub['age_years'])

            for age_group in module.AGE_LABELS:
                age_df = sub.loc[sub['age_group'] == age_group]
                n = len(age_df)

                log_data[f'HPVVaccinated_{age_group}_N'] = int(age_df['hpv_vaccinated'].sum()) if n > 0 else 0
                log_data[f'HPVVaccinated_{age_group}_Coverage'] = float(
                    age_df['hpv_vaccinated'].mean()) if n > 0 else math.nan

            # Female age-specific vaccine coverage
            female_df = sub.loc[sub.sex == 'F']

            for age_group in module.AGE_LABELS:
                age_df = female_df.loc[female_df['age_group'] == age_group]
                n = len(age_df)

                log_data[f'HPVVaccinated_F_{age_group}_N'] = int(age_df['hpv_vaccinated'].sum()) if n > 0 else 0
                log_data[f'HPVVaccinated_F_{age_group}_Coverage'] = float(
                    age_df['hpv_vaccinated'].mean()) if n > 0 else math.nan

        else:
            log_data['HPVVaccinated_N'] = 0
            log_data['HPVVaccinated_Coverage'] = math.nan

        for hpv_group in module.HPV_GROUPS:
            sub[f'hp_infected_{hpv_group}'] = module._get_group_infected_series(
                hpv_group,
                index=sub.index
            )

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

                    prev_snapshot[f'Any_{sex_name}_{age_group}_Prev'] = math.nan

                    for hpv_group in module.HPV_GROUPS:
                        log_data[f'{hpv_group}_{sex_name}_{age_group}_Inf'] = 0
                        log_data[f'{hpv_group}_{sex_name}_{age_group}_Prev'] = math.nan

                        prev_snapshot[f'{hpv_group}_{sex_name}_{age_group}_Prev'] = math.nan

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
        hiv_log_groups = [
            'HIVneg',
            'HIVpos_unknown',
            'HIVpos_noART',
            'HIVpos_unsupp',
            'HIVpos_supp',
        ]

        for hiv_group in hiv_log_groups:
            log_data[f'Any_{hiv_group}_N'] = 0
            log_data[f'Any_{hiv_group}_Inf'] = 0
            log_data[f'Any_{hiv_group}_Prev'] = math.nan

            for hpv_group in module.HPV_GROUPS:
                log_data[f'{hpv_group}_{hiv_group}_Prev'] = math.nan

        for hiv_group in hiv_log_groups:
            hiv_df = sub.loc[sub['hiv_group'] == hiv_group]
            n = len(hiv_df)

            log_data[f'Any_{hiv_group}_N'] = int(n)

            if n > 0:
                log_data[f'Any_{hiv_group}_Inf'] = int(hiv_df['hp_is_infected'].sum())
                log_data[f'Any_{hiv_group}_Prev'] = float(hiv_df['hp_is_infected'].mean())

                for hpv_group in module.HPV_GROUPS:
                    log_data[f'{hpv_group}_{hiv_group}_Prev'] = float(
                        hiv_df[f'hp_infected_{hpv_group}'].mean()
                    )
            else:
                log_data[f'Any_{hiv_group}_Inf'] = 0
                log_data[f'Any_{hiv_group}_Prev'] = math.nan

                for hpv_group in module.HPV_GROUPS:
                    log_data[f'{hpv_group}_{hiv_group}_Prev'] = math.nan

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
        infection_people = sub.index[sub['hp_is_infected']]
        n_group_1 = 0
        n_group_2 = 0
        n_group_3 = 0
        n_group_4 = 0

        male_n_group_1 = 0
        male_n_group_2 = 0
        male_n_group_3 = 0
        male_n_group_4 = 0

        female_n_group_1 = 0
        female_n_group_2 = 0
        female_n_group_3 = 0
        female_n_group_4 = 0

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

            elif n_group == 4:
                n_group_4 += 1
                if sex == 'M':
                    male_n_group_4 += 1
                elif sex =='F':
                    female_n_group_4 += 1

        log_data['InfGroup1'] = n_group_1
        log_data['InfGroup2'] = n_group_2
        log_data['InfGroup3'] = n_group_3
        log_data['InfGroup4'] = n_group_4

        log_data['MaleGroup1'] = male_n_group_1
        log_data['MaleGroup2'] = male_n_group_2
        log_data['MaleGroup3'] = male_n_group_3
        log_data['MaleGroup4'] = male_n_group_4

        log_data['FemaleGroup1'] = female_n_group_1
        log_data['FemaleGroup2'] = female_n_group_2
        log_data['FemaleGroup3'] = female_n_group_3
        log_data['FemaleGroup4'] = female_n_group_4

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

        # 7. Incidence and clearance counts from the latest HPV infection event
        last_event_counts = getattr(module, '_last_event_counts', {})

        for key, value in last_event_counts.items():
            log_data[key] = value

        logger.info(key='summary', data=log_data)
