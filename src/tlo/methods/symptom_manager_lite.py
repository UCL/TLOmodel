"""Lightweight Symptom Manager with optimized performance"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Union, Dict, Set

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, PopulationScopeEventMixin, RegularEvent
from tlo.methods import Metadata

if TYPE_CHECKING:
    from tlo.population import IndividualProperties

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Symptom:
    """Simplified Symptom class with same interface"""
    def __init__(self,
                 name: str = None,
                 no_healthcareseeking_in_adults: bool = False,
                 no_healthcareseeking_in_children: bool = False,
                 odds_ratio_health_seeking_in_adults: float = None,
                 odds_ratio_health_seeking_in_children: float = None,
                 prob_seeks_emergency_appt_in_adults: float = None,
                 prob_seeks_emergency_appt_in_children: float = None):

        assert isinstance(name, str) and name, 'name of symptom cannot be blank'
        self.name = name
        self.no_healthcareseeking_in_children = no_healthcareseeking_in_children
        self.no_healthcareseeking_in_adults = no_healthcareseeking_in_adults

        # Set defaults
        self.prob_seeks_emergency_appt_in_adults = prob_seeks_emergency_appt_in_adults or 0.0
        self.prob_seeks_emergency_appt_in_children = prob_seeks_emergency_appt_in_children or 0.0
        self.odds_ratio_health_seeking_in_adults = odds_ratio_health_seeking_in_adults or 1.0
        self.odds_ratio_health_seeking_in_children = odds_ratio_health_seeking_in_children or 1.0

        # Validate values
        assert 0.0 <= self.odds_ratio_health_seeking_in_adults
        assert 0.0 <= self.odds_ratio_health_seeking_in_children
        assert 0.0 <= self.prob_seeks_emergency_appt_in_adults <= 1.0
        assert 0.0 <= self.prob_seeks_emergency_appt_in_children <= 1.0

    @staticmethod
    def emergency(name: str, which: str = "both"):
        """Return an emergency symptom instance"""
        from tlo.methods.healthseekingbehaviour import HIGH_ODDS_RATIO

        emergency_in_adults = which in ("adults", "both")
        emergency_in_children = which in ("children", "both")

        return Symptom(
            name=name,
            no_healthcareseeking_in_adults=False,
            no_healthcareseeking_in_children=False,
            prob_seeks_emergency_appt_in_adults=1.0 if emergency_in_adults else 0.0,
            prob_seeks_emergency_appt_in_children=1.0 if emergency_in_children else 0.0,
            odds_ratio_health_seeking_in_adults=HIGH_ODDS_RATIO if emergency_in_adults else 0.0,
            odds_ratio_health_seeking_in_children=HIGH_ODDS_RATIO if emergency_in_children else 0.0,
        )

    def __eq__(self, other):
        return isinstance(other, Symptom) and all(
            getattr(self, p) == getattr(other, p) for p in [
                'name',
                'no_healthcareseeking_in_children',
                'no_healthcareseeking_in_adults',
                'prob_seeks_emergency_appt_in_adults',
                'prob_seeks_emergency_appt_in_children',
                'odds_ratio_health_seeking_in_adults',
                'odds_ratio_health_seeking_in_children'
            ]
        )

    def __hash__(self):
        return hash(self.name)


class DuplicateSymptomWithNonIdenticalPropertiesError(Exception):
    def __init__(self):
        super().__init__("A symptom with this name has been registered already but with different properties")


class SymptomManager(Module):
    """Optimized symptom manager using boolean columns for faster access"""

    INIT_DEPENDENCIES = {'Demography'}
    METADATA = {}
    PROPERTIES = dict()  # Will be populated in pre_initialise_population

    PARAMETERS = {
        'generic_symptoms_spurious_occurrence': Parameter(
            Types.DATA_FRAME, 'probability and duration of spurious occurrences of generic symptoms'),
        'spurious_symptoms': Parameter(
            Types.BOOL, 'whether or not there will be the spontaneous occurrence of generic symptoms'),
    }

    def __init__(self, name=None, resourcefilepath=None, spurious_symptoms=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.spurious_symptoms = spurious_symptoms
        self._persons_with_newly_onset_symptoms = set()

        # Core symptom tracking structures
        self._symptoms: Dict[str, Symptom] = {}  # name -> Symptom object
        self._symptom_columns: Dict[str, str] = {}  # symptom name -> column name
        self._module_symptom_causes: Dict[str, Set[str]] = defaultdict(set)  # module -> symptoms it can cause

        # Add module-symptom tracking
        self._module_symptoms = defaultdict(set)  # module_name -> set of symptoms it can cause
        self._symptom_modules = defaultdict(set)  # symptom_name -> set of modules that can cause it

        # Generic symptoms
        self.generic_symptoms = {
            'fever', 'vomiting', 'stomachache', 'sore_throat', 'respiratory_symptoms',
            'headache', 'skin_complaint', 'dental_complaint', 'backache', 'injury',
            'eye_complaint', 'diarrhoea', 'spurious_emergency_symptom'
        }

    def read_parameters(self, data_folder):
        """Read parameters from files"""
        self.parameters['generic_symptoms_spurious_occurrence'] = \
            pd.read_csv(Path(self.resourcefilepath) / 'ResourceFile_GenericSymptoms_and_HealthSeeking.csv')
        self.load_parameters_from_dataframe(
            pd.read_csv(Path(self.resourcefilepath) / 'ResourceFile_SymptomManager.csv'))

    @property
    def all_registered_symptoms(self):
        """Backward compatibility property that returns registered symptoms as a set"""
        return set(self._symptoms.values())

    def register_symptom(self, *symptoms_to_register: Symptom):
        """Register one or more symptoms"""
        for symptom in symptoms_to_register:
            if symptom.name in self._symptoms:
                if symptom != self._symptoms[symptom.name]:
                    raise DuplicateSymptomWithNonIdenticalPropertiesError()
                # else: it's a duplicate with identical properties - ignore
            else:
                self._symptoms[symptom.name] = symptom
                self._symptom_columns[symptom.name] = f'sy_{symptom.name}'

    def register_generic_symptoms(self):
        """Register generic symptoms from parameters"""
        df = self.parameters['generic_symptoms_spurious_occurrence']
        assert self.generic_symptoms == set(df['name'].to_list())

        symptoms_df = df[[
            'name', 'odds_ratio_health_seeking_in_children', 'odds_ratio_health_seeking_in_adults',
            'prob_seeks_emergency_appt_in_adults', 'prob_seeks_emergency_appt_in_children'
        ]].set_index('name').loc[sorted(self.generic_symptoms)].reset_index()

        for _, row in symptoms_df.iterrows():
            self.register_symptom(Symptom(**row.to_dict()))

    def pre_initialise_population(self):
        """Set up properties before population initialization"""
        self.register_generic_symptoms()

        # Create boolean properties for each symptom
        SymptomManager.PROPERTIES = {
            f'sy_{name}': Property(Types.BOOL, f'Presence of symptom {name}')
            for name in self._symptoms
        }

    def initialise_population(self, population):
        """Initialize population properties"""
        self.recognised_module_names = [
            m.name for m in self.sim.modules.values()
            if Metadata.USES_SYMPTOMMANAGER in m.METADATA
        ]

        # Initialize all symptom columns to False
        df = population.props
        for col in self._symptom_columns.values():
            df[col] = False

        # Set spurious symptoms parameter
        self.spurious_symptoms = (
            self.parameters['spurious_symptoms']
            if self.spurious_symptoms is None
            else self.spurious_symptoms
        )

    def initialise_simulation(self, sim):
        """Schedule events if spurious symptoms are enabled"""
        if self.spurious_symptoms:
            sim.schedule_event(
                SymptomManager_SpuriousSymptomOnset(module=self),
                self.sim.date
            )
            self.spurious_symptom_resolve_event = SymptomManager_SpuriousSymptomResolve(module=self)
            sim.schedule_event(
                self.spurious_symptom_resolve_event,
                self.sim.date
            )

    def on_birth(self, mother_id, child_id):
        """Initialize symptoms for new born"""
        df = self.sim.population.props
        for col in self._symptom_columns.values():
            df.at[child_id, col] = False

    def change_symptom(self, person_id, symptom_string, add_or_remove, disease_module,
                       duration_in_days=None, date_of_onset=None):
        """Add or remove symptoms for persons"""
        df = self.sim.population.props

        # Convert inputs to consistent formats
        person_id = [person_id] if isinstance(person_id, (int, np.integer)) else list(person_id)
        symptom_string = [symptom_string] if isinstance(symptom_string, str) else list(symptom_string)

        # Filter to alive persons
        person_id = df.index[df.is_alive & df.index.isin(person_id)]
        if not len(person_id):
            return

        # Validate inputs
        for sym in symptom_string:
            assert sym in self._symptoms, f'Symptom {sym} not registered'
        assert add_or_remove in ['+', '-']
        assert disease_module.name in ([self.name] + self.recognised_module_names)
        assert (date_of_onset is None) or (
            isinstance(date_of_onset, pd.Timestamp) and (date_of_onset >= self.sim.date)
        )

        # Schedule future onset if needed
        if (date_of_onset is not None) and (date_of_onset > self.sim.date):
            self.sim.schedule_event(
                SymptomManager_AutoOnsetEvent(
                    self, person_id, symptom_string, disease_module, duration_in_days
                ),
                date_of_onset
            )
            return

        # Get column names
        sy_columns = [self._symptom_columns[sym] for sym in symptom_string]

        if add_or_remove == '+':
            self._module_symptoms[disease_module.name].update(symptom_string)
            for sym in symptom_string:
                self._symptom_modules[sym].add(disease_module.name)

            # Add symptoms
            df.loc[person_id, sy_columns] = True
            self._persons_with_newly_onset_symptoms.update(person_id)

            # Schedule auto-resolve if duration specified
            if duration_in_days is not None:
                self.sim.schedule_event(
                    SymptomManager_AutoResolveEvent(
                        self, person_id, symptom_string, disease_module
                    ),
                    self.sim.date + DateOffset(days=int(duration_in_days))
                )
        else:
            # Remove symptoms
            df.loc[person_id, sy_columns] = False

    def who_has(self, list_of_symptoms):
        """Find who has all specified symptoms"""
        list_of_symptoms = [list_of_symptoms] if isinstance(list_of_symptoms, str) else list(list_of_symptoms)
        assert all(symp in self._symptoms for symp in list_of_symptoms), 'Symptom not registered'

        df = self.sim.population.props
        sy_columns = [self._symptom_columns[s] for s in list_of_symptoms]
        has_all = df.loc[df.is_alive, sy_columns].all(axis=1)
        return has_all[has_all].index.tolist()

    def who_not_have(self, symptom_string: str) -> pd.Index:
        """Find who doesn't have a specific symptom"""
        assert symptom_string in self._symptoms, 'Symptom not registered'
        df = self.sim.population.props
        return df.index[df.is_alive & ~df[self._symptom_columns[symptom_string]]]

    def has_what(self, person_id=None, individual_details=None, disease_module=None) -> List[str]:
        """Optimized implementation - fastest way to check symptoms"""
        symptoms = []
        if individual_details is not None:
            # Using IndividualProperties context
            if disease_module is not None:
                symptoms = [
                    sym for sym in self._module_symptoms.get(disease_module.name, set())
                    if individual_details[self._symptom_columns[sym]]]
            else:
                symptoms = [sym for sym in self._symptoms if individual_details[self._symptom_columns[sym]]]
        else:
            # Using person_id
            df = self.sim.population.props
            if disease_module is not None:
                symptoms = [
                    sym for sym in self._module_symptoms.get(disease_module.name, set())
                    if df.at[person_id, self._symptom_columns[sym]]]
            else:
                symptoms = [sym for sym in self._symptoms if df.at[person_id, self._symptom_columns[sym]]]

        return symptoms

    def have_what(self, person_ids: Sequence[int]):
        """Find symptoms for multiple persons"""
        df = self.sim.population.props
        return df.loc[person_ids].apply(
            lambda p: [s for s in self._symptoms if p[self._symptom_columns[s]]],
            axis=1
        ).rename('symptoms')

    def causes_of(self, person_id: int, symptom_string):
        """Find causes of a symptom for a person"""
        assert symptom_string in self._symptoms
        if symptom_string not in self._symptoms:
            return []

        df = self.sim.population.props
        if df.at[person_id, self._symptom_columns[symptom_string]]:
            return list(self._symptom_modules.get(symptom_string, set()))
        return []

    def clear_symptoms(self, person_id: Union[int, Sequence[int]], disease_module: Module):
        """Clear all symptoms caused by a module"""
        df = self.sim.population.props
        person_id = [person_id] if isinstance(person_id, (int, np.integer)) else list(person_id)
        assert df.loc[person_id, 'is_alive'].all(), "One or more persons not alive"
        assert disease_module.name in ([self.name] + self.recognised_module_names)

        # Clear all symptoms this module can cause
        for sym in self._module_symptom_causes.get(disease_module.name, set()):
            df.loc[person_id, self._symptom_columns[sym]] = False

    def caused_by(self, disease_module: Module):
        """Find persons with symptoms caused by a module"""
        df = self.sim.population.props
        symptoms = self._module_symptom_causes.get(disease_module.name, set())
        if not symptoms:
            return {}

        sy_columns = [self._symptom_columns[s] for s in symptoms]
        has_symptoms = df.loc[df.is_alive, sy_columns].any(axis=1)
        return {
            pid: [s for s in symptoms if df.at[pid, self._symptom_columns[s]]]
            for pid in has_symptoms[has_symptoms].index
        }

    def get_persons_with_newly_onset_symptoms(self):
        return self._persons_with_newly_onset_symptoms

    def reset_persons_with_newly_onset_symptoms(self):
        self._persons_with_newly_onset_symptoms.clear()

# ---------------------------------------------------------------------------------------------------------
#   EVENTS (unchanged from original)
# ---------------------------------------------------------------------------------------------------------


class SymptomManager_AutoOnsetEvent(Event, PopulationScopeEventMixin):
    """Event to add symptoms on a future date"""
    def __init__(self, module, person_id, symptom_string, disease_module, duration_in_days):
        super().__init__(module)
        self.person_id = list(person_id)
        self.symptom_string = symptom_string
        self.disease_module = disease_module
        self.duration_in_days = duration_in_days

    def apply(self, population):
        self.module.change_symptom(
            person_id=self.person_id,
            symptom_string=self.symptom_string,
            add_or_remove='+',
            disease_module=self.disease_module,
            duration_in_days=self.duration_in_days
        )


class SymptomManager_AutoResolveEvent(Event, PopulationScopeEventMixin):
    """Event to remove symptoms after duration"""
    def __init__(self, module, person_id, symptom_string, disease_module):
        super().__init__(module)
        self.person_id = list(person_id)
        self.symptom_string = symptom_string
        self.disease_module = disease_module

    def apply(self, population):
        self.module.change_symptom(
            person_id=self.person_id,
            symptom_string=self.symptom_string,
            add_or_remove='-',
            disease_module=self.disease_module
        )

class SymptomManager_SpuriousSymptomOnset(RegularEvent, PopulationScopeEventMixin):
    """Event for spurious symptom onset"""
    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(days=1))
        self.generic_symptoms = self._get_generic_symptoms_dict(
            module.parameters['generic_symptoms_spurious_occurrence'])
        self.rand = module.rng.rand

    def _get_generic_symptoms_dict(self, df):
        df = df.set_index('name')
        return {
            'prob_per_day': {
                'children': df['prob_spurious_occurrence_in_children_per_day'].to_dict(),
                'adults': df['prob_spurious_occurrence_in_adults_per_day'].to_dict()
            },
            'duration_in_days': {
                'children': df['duration_in_days_of_spurious_occurrence_in_children'].astype(int).to_dict(),
                'adults': df['duration_in_days_of_spurious_occurrence_in_adults'].astype(int).to_dict()
            }
        }

    def apply(self, population):
        df = self.sim.population.props
        group_indices = {
            'children': df.index[df.is_alive & (df.age_years < 15)],
            'adults': df.index[df.is_alive & (df.age_years >= 15)]
        }

        for symp in sorted(self.module.generic_symptoms):
            do_not_have = self.module.who_not_have(symp)

            for group in ['children', 'adults']:
                p = self.generic_symptoms['prob_per_day'][group][symp]
                dur = self.generic_symptoms['duration_in_days'][group][symp]
                eligible = group_indices[group][group_indices[group].isin(do_not_have)]
                to_onset = eligible[self.rand(len(eligible)) < p]

                if len(to_onset):
                    self.module.change_symptom(
                        symptom_string=symp,
                        add_or_remove='+',
                        person_id=to_onset,
                        duration_in_days=None,
                        disease_module=self.module,
                    )
                    self.module.spurious_symptom_resolve_event.schedule_symptom_resolve(
                        person_id=to_onset,
                        symptom_string=symp,
                        date_of_resolution=(self.sim.date + pd.DateOffset(days=dur)).date()
                    )

class SymptomManager_SpuriousSymptomResolve(RegularEvent, PopulationScopeEventMixin):
    """Event for spurious symptom resolution"""
    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(days=1))
        self.to_resolve = defaultdict(lambda: defaultdict(set))

    def schedule_symptom_resolve(self, person_id, date_of_resolution, symptom_string):
        self.to_resolve[symptom_string][date_of_resolution].update(person_id)

    def apply(self, population):
        df = population.props
        today = self.sim.date.date()

        for symp, dates in self.to_resolve.items():
            if today in dates:
                pids = dates.pop(today)
                alive = df.index[df.index.isin(pids) & df.is_alive]
                if len(alive):
                    self.module.change_symptom(
                        person_id=alive,
                        add_or_remove='-',
                        symptom_string=symp,
                        disease_module=self.module
                    )
