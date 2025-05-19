"""
Optimized Symptom Manager Lite with module tracking using boolean columns and bitsets for cause tracking.
Maintains the same interface as the original SymptomManager with improved performance.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Union, Dict, Set

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, PopulationScopeEventMixin, RegularEvent
from tlo.methods import Metadata
from tlo.util import BitsetHandler, BitsetDType

if TYPE_CHECKING:
    from tlo.population import IndividualProperties

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Symptom:
    """Data structure to hold information about a symptom."""

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


class SymptomManagerLite(Module):
    """
    Optimized symptom manager with module tracking using:
    - Boolean columns for symptom presence (fast lookup)
    - Bitsets for tracking which modules cause which symptoms
    """

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

        # Core symptom tracking
        self._symptoms = set()  # Set of Symptom objects
        self._symptom_columns = dict()  # symptom name -> column name

        # Module tracking
        self.recognised_module_names = None
        self.spurious_symptom_resolve_event = None
        self._module_bitsets = None  # Will be initialized in initialise_population

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

    def register_symptom(self, *symptoms_to_register: Symptom):
        """Register one or more symptoms"""
        for symptom in symptoms_to_register:
            if symptom.name in {s.name for s in self._symptoms}:
                # Check if this symptom matches any previously registered with same name
                existing = next(s for s in self._symptoms if s.name == symptom.name)
                if symptom != existing:
                    raise DuplicateSymptomWithNonIdenticalPropertiesError()
            else:
                self._symptoms.add(symptom)

    @property
    def all_registered_symptoms(self):
        """Get all registered symptoms as a set"""
        return self._symptoms

    @property
    def symptom_names(self):
        """Get all registered symptom names as a set"""
        return {s.name for s in self._symptoms}

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

        # Create properties for each symptom (boolean column + bitset for causes)
        SymptomManagerLite.PROPERTIES = {}
        for symptom in self._symptoms:
            col_name = f'sy_{symptom.name}'
            self._symptom_columns[symptom.name] = col_name
            SymptomManagerLite.PROPERTIES[col_name] = Property(Types.BOOL, f'Presence of symptom {symptom.name}')
            SymptomManagerLite.PROPERTIES[f'{col_name}_causes'] = Property(
                Types.BITSET, f'Bitmask of modules causing symptom {symptom.name}')

    def initialise_population(self, population):
        """Initialize population properties"""
        self.recognised_module_names = [
            m.name for m in self.sim.modules.values()
            if Metadata.USES_SYMPTOMMANAGER in m.METADATA
        ]
        all_modules = [self.name] + self.recognised_module_names

        # Initialize symptom columns
        df = population.props
        for symptom in self._symptoms:
            col_name = self._symptom_columns[symptom.name]
            df[col_name] = False  # Boolean column for symptom presence
            df[f'{col_name}_causes'] = 0  # Initialize to 0
            df[f'{col_name}_causes'] = df[f'{col_name}_causes'].astype(BitsetDType)  # Convert to BitsetDType

        # Initialize bitset handler for module tracking
        self._module_bitsets = BitsetHandler(
            population=population,
            column=None,
            elements=all_modules
        )

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
                SymptomManagerLite_SpuriousSymptomOnset(module=self),
                self.sim.date
            )
            self.spurious_symptom_resolve_event = SymptomManagerLite_SpuriousSymptomResolve(module=self)
            sim.schedule_event(
                self.spurious_symptom_resolve_event,
                self.sim.date
            )

    def on_birth(self, mother_id, child_id):
        """Initialize symptoms for new born"""
        df = self.sim.population.props
        for symptom in self._symptoms:
            col_name = self._symptom_columns[symptom.name]
            df.at[child_id, col_name] = False
            df.at[child_id, f'{col_name}_causes'] = 0

    def change_symptom(self, person_id, symptom_string, add_or_remove, disease_module,
                       duration_in_days=None, date_of_onset=None):
        """Add or remove symptoms for persons with module tracking"""
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
            assert sym in self._symptom_columns, f'Symptom {sym} not registered'
        assert add_or_remove in ['+', '-']
        assert disease_module.name in ([self.name] + self.recognised_module_names)
        assert (date_of_onset is None) or (
            isinstance(date_of_onset, pd.Timestamp) and (date_of_onset >= self.sim.date)
        )

        # Schedule future onset if needed
        if (date_of_onset is not None) and (date_of_onset > self.sim.date):
            self.sim.schedule_event(
                SymptomManagerLite_AutoOnsetEvent(
                    self, person_id, symptom_string, disease_module, duration_in_days
                ),
                date_of_onset
            )
            return

        # Get column names
        sy_columns = [self._symptom_columns[sym] for sym in symptom_string]
        cause_columns = [f'{col}_causes' for col in sy_columns]

        if add_or_remove == '+':
            # Set symptom to True and add module to causes bitset
            df.loc[person_id, sy_columns] = True
            self._module_bitsets.set(person_id, disease_module.name, columns=cause_columns)
            self._persons_with_newly_onset_symptoms.update(person_id)

            # Schedule auto-resolve if duration specified
            if duration_in_days is not None:
                self.sim.schedule_event(
                    SymptomManagerLite_AutoResolveEvent(
                        self, person_id, symptom_string, disease_module
                    ),
                    self.sim.date + DateOffset(days=int(duration_in_days))
                )
        else:
            # Remove module from causes bitset
            self._module_bitsets.unset(person_id, disease_module.name, columns=cause_columns)

            # Check if any causes remain for each symptom
            for col in cause_columns:
                # Update symptom presence based on remaining causes
                df.loc[person_id, col.replace('_causes', '')] = df.loc[person_id, col] > 0

    def who_has(self, list_of_symptoms):
        """Find who has all specified symptoms"""
        list_of_symptoms = [list_of_symptoms] if isinstance(list_of_symptoms, str) else list(list_of_symptoms)
        assert all(symp in self._symptom_columns for symp in list_of_symptoms), 'Symptom not registered'

        df = self.sim.population.props
        sy_columns = [self._symptom_columns[s] for s in list_of_symptoms]
        has_all = df.loc[df.is_alive, sy_columns].all(axis=1)
        return has_all[has_all].index.tolist()

    def who_not_have(self, symptom_string: str) -> pd.Index:
        """Find who doesn't have a specific symptom"""
        assert symptom_string in self._symptom_columns, 'Symptom not registered'
        df = self.sim.population.props
        return df.index[df.is_alive & ~df[self._symptom_columns[symptom_string]]]

    def has_what(self, person_id=None, individual_details=None, disease_module=None) -> List[str]:
        """Get symptoms for a single person with optional module filtering"""
        if individual_details is not None:
            if disease_module is not None:
                symptoms = []
                for s in self._symptoms:
                    symptom_col = self._symptom_columns[s.name]
                    cause_col = f'{symptom_col}_causes'
                    if individual_details[symptom_col]:
                        if self._module_bitsets.has([individual_details.index], disease_module.name,
                                                    columns=cause_col).item():
                            symptoms.append(s.name)
                return symptoms
            else:
                return [
                    s.name for s in self._symptoms
                    if individual_details[self._symptom_columns[s.name]]
                ]
        else:
            df = self.sim.population.props
            if disease_module is not None:
                symptoms = []
                for s in self._symptoms:
                    symptom_col = self._symptom_columns[s.name]
                    cause_col = f'{symptom_col}_causes'
                    if df.at[person_id, symptom_col]:
                        if self._module_bitsets.has([person_id], disease_module.name, columns=cause_col).item():
                            symptoms.append(s.name)
                return symptoms
            else:
                return [
                    s.name for s in self._symptoms
                    if df.at[person_id, self._symptom_columns[s.name]]
                ]

    def have_what(self, person_ids: Sequence[int]):
        """Find symptoms for multiple persons"""
        df = self.sim.population.props
        return df.loc[person_ids].apply(
            lambda p: [s.name for s in self._symptoms if p[self._symptom_columns[s.name]]],
            axis=1
        ).rename('symptoms')

    def causes_of(self, person_id: int, symptom_string):
        """Find causes of a symptom for a person"""
        assert symptom_string in self._symptom_columns
        df = self.sim.population.props
        if not df.at[person_id, self._symptom_columns[symptom_string]]:
            return []

        cause_col = f'{self._symptom_columns[symptom_string]}_causes'
        return list(self._module_bitsets.get([person_id], first=True, columns=[cause_col]))

    def clear_symptoms(self, person_id: Union[int, Sequence[int]], disease_module: Module):
        """Clear all symptoms caused by a module"""
        df = self.sim.population.props
        person_id = [person_id] if isinstance(person_id, (int, np.integer)) else list(person_id)
        assert df.loc[person_id, 'is_alive'].all(), "One or more persons not alive"
        assert disease_module.name in ([self.name] + self.recognised_module_names)

        # Get all symptom cause columns
        cause_columns = [f'{col}_causes' for col in self._symptom_columns.values()]

        # Remove module from all cause bitsets
        self._module_bitsets.unset(person_id, disease_module.name, columns=cause_columns)

        # Update symptom presence based on remaining causes
        for col in self._symptom_columns.values():
            df.loc[person_id, col] = df.loc[person_id, f'{col}_causes'] > 0

    def caused_by(self, disease_module: Module):
        """Find persons with symptoms caused by a module"""
        df = self.sim.population.props
        alive_idx = df.index[df.is_alive]

        # Find all symptoms caused by this module
        result = {}
        for symptom in self._symptoms:
            col = self._symptom_columns[symptom.name]
            cause_col = f'{col}_causes'

            # Get persons where this module is causing the symptom
            has_symptom_from_module = self._module_bitsets.has(
                alive_idx, disease_module.name, columns=[cause_col])

            # Add to result if any persons found
            pids = has_symptom_from_module[has_symptom_from_module].index
            if len(pids) > 0:
                for pid in pids:
                    if pid not in result:
                        result[pid] = []
                    result[pid].append(symptom.name)

        return result

    def get_persons_with_newly_onset_symptoms(self):
        return self._persons_with_newly_onset_symptoms

    def reset_persons_with_newly_onset_symptoms(self):
        self._persons_with_newly_onset_symptoms.clear()


# ---------------------------------------------------------------------------------------------------------
#   EVENTS (unchanged from original)
# ---------------------------------------------------------------------------------------------------------

class SymptomManagerLite_AutoOnsetEvent(Event, PopulationScopeEventMixin):
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


class SymptomManagerLite_AutoResolveEvent(Event, PopulationScopeEventMixin):
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


class SymptomManagerLite_SpuriousSymptomOnset(RegularEvent, PopulationScopeEventMixin):
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


class SymptomManagerLite_SpuriousSymptomResolve(RegularEvent, PopulationScopeEventMixin):
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
