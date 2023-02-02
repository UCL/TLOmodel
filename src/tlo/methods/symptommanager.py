"""
The Symptom Manager:
* Manages presence of symptoms for all disease modules
* Manages a set of generic symptoms
* Creates occurrences of generic symptom (representing that being caused by diseases not included in the TLO model)

The write-up for the origin of the estimates for the effect of each symptom is:
 Health-seeking behaviour estimates for adults and children.docx

Outstanding issues
* The probability of spurious symptoms is not informed by data.

"""
from collections import defaultdict
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, PopulationScopeEventMixin, RegularEvent
from tlo.methods import Metadata
from tlo.util import BitsetHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------


class Symptom:
    """Data structure to hold the information about a symptom.

    The assumption is that a symptom tends to cause health-care seeking. This can be modified by specifying:
     * if the symptom does not cause healthcare-seeking at all (`no_healthcareseeking_in_`);
     * if the symptom is more or less likely to cause healthcare-seeking compared the "average symptom";
      (`odds_ratio_health_seeking_in_`)
     * the probability that emergency care is sought, if care is sought at all (`prob_seeks_emergency_appt_in_`).

    The default behaviour is for a symptom that causes healthcare-seeking for non-emergency care with the same
    probability as the "average symptom".

    The in-built method `emergency_symptom_with_automatic_healthcareseeking` produces another common type of symptom,
    which gives a very high probability that emergency care is sought.

    The characteristics of Symptoms is separate for adults (peron aged 15+) and children (those aged aged <15).
    """

    def __init__(self,
                 name: str = None,
                 no_healthcareseeking_in_adults: bool = False,
                 no_healthcareseeking_in_children: bool = False,
                 odds_ratio_health_seeking_in_adults: float = None,
                 odds_ratio_health_seeking_in_children: float = None,
                 prob_seeks_emergency_appt_in_adults: float = None,
                 prob_seeks_emergency_appt_in_children: float = None,
                 ):

        # Check that the types are correct and not nonsensical
        assert isinstance(name, str)
        assert name, 'name of symptom cannot be blank'

        assert isinstance(no_healthcareseeking_in_adults, bool)
        assert isinstance(no_healthcareseeking_in_children, bool)

        # Check logic of the arguments: if the symptom does not cause healthcare-seeking behaviour then the other
        # arguments should not be provided.
        if no_healthcareseeking_in_children:
            assert prob_seeks_emergency_appt_in_children is None
            assert odds_ratio_health_seeking_in_children is None

        if no_healthcareseeking_in_adults:
            assert prob_seeks_emergency_appt_in_adults is None
            assert odds_ratio_health_seeking_in_adults is None

        # Define the default behaviour:
        if prob_seeks_emergency_appt_in_adults is None:
            prob_seeks_emergency_appt_in_adults = 0.0  # i.e. Symptom will not cause h.c.s. for emergency care.

        if prob_seeks_emergency_appt_in_children is None:
            prob_seeks_emergency_appt_in_children = 0.0  # i.e. Symptom will not cause h.c.s. for emergency care.

        if odds_ratio_health_seeking_in_adults is None:
            odds_ratio_health_seeking_in_adults = 1.0  # i.e. Symptom has same odds of h.c.s. as the 'default'

        if odds_ratio_health_seeking_in_children is None:
            odds_ratio_health_seeking_in_children = 1.0  # i.e. Symptom has same odds of h.c.s. as the 'default'

        # Check that the odds-ratio of healthcare seeking is greater than or equal to 0.0
        assert isinstance(odds_ratio_health_seeking_in_adults, float)
        assert isinstance(odds_ratio_health_seeking_in_children, float)
        assert 0.0 <= odds_ratio_health_seeking_in_adults
        assert 0.0 <= odds_ratio_health_seeking_in_children

        # Check that probability of seeking an emergency appointment must be between 0.0 and 1.0
        assert isinstance(prob_seeks_emergency_appt_in_adults, float)
        assert isinstance(prob_seeks_emergency_appt_in_children, float)
        assert 0.0 <= prob_seeks_emergency_appt_in_adults <= 1.0
        assert 0.0 <= prob_seeks_emergency_appt_in_children <= 1.0

        # Store properties:
        self.name = name
        self.no_healthcareseeking_in_children = no_healthcareseeking_in_children
        self.no_healthcareseeking_in_adults = no_healthcareseeking_in_adults
        self.prob_seeks_emergency_appt_in_adults = prob_seeks_emergency_appt_in_adults
        self.prob_seeks_emergency_appt_in_children = prob_seeks_emergency_appt_in_children
        self.odds_ratio_health_seeking_in_adults = odds_ratio_health_seeking_in_adults
        self.odds_ratio_health_seeking_in_children = odds_ratio_health_seeking_in_children

    @staticmethod
    def emergency(name: str, which: str = "both"):
        """Return an instance of `Symptom` that will guarantee healthcare-seeking for an Emergency Appointment."""
        from tlo.methods.healthseekingbehaviour import HIGH_ODDS_RATIO

        if name is None:
            raise ValueError('No name given.')

        if which not in ("adults", "children", "both"):
            raise ValueError('Argument not recognised.')

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
            #                                      10_000 is an arbitrarily large odds ratio that will practically
            #                                       ensure that there is healthcare-seeking. `np.inf` might have been
            #                                       used but this is not does not work within the LinearModel.
        )

    def __eq__(self, other):
        """Define the basis upon which tests of equivalence are made for Symptom objects.
        NB. This seems neccessary to enable to checking of equivalency between symptoms registered in different
        places. Without this two instance of the object with the same properties are not recognised as being the 'same'.
        This is done in conjunction with over-riding the hash property."""
        return isinstance(other, Symptom) and all(
            [getattr(self, p) == getattr(other, p) for p in [
                'name',
                'no_healthcareseeking_in_children',
                'no_healthcareseeking_in_adults',
                'prob_seeks_emergency_appt_in_adults',
                'prob_seeks_emergency_appt_in_children',
                'odds_ratio_health_seeking_in_adults',
                'odds_ratio_health_seeking_in_children']
             ])

    def __hash__(self):
        """Override the hash function to force set to rely on __eq__."""
        return 0


class DuplicateSymptomWithNonIdenticalPropertiesError(Exception):
    def __init__(self):
        super().__init__("A symptom with this name has been registered already but with different properties")


class SymptomManager(Module):
    """
    This module is used to track the symptoms of persons. The addition and removal of symptoms by disease modules is
     handled here. This module can also causes symptoms that are not related to any disease module (representing those
     caused by conditions not represented explicitly in the model).
    """

    INIT_DEPENDENCIES = {'Demography'}

    # Declare Metadata
    METADATA = {}

    PROPERTIES = dict()  # updated at ```pre-initialise population``` once symptoms have been registered.

    PARAMETERS = {
        'generic_symptoms_spurious_occurrence': Parameter(
            Types.DATA_FRAME, 'probability and duration of spurious occureneces of generic symptoms')
    }

    def __init__(self, name=None, resourcefilepath=None, spurious_symptoms=False):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.spurious_symptoms = spurious_symptoms
        self._persons_with_newly_onset_symptoms = set()

        self.generic_symptoms = {
            'fever',
            'vomiting',
            'stomachache',
            'sore_throat',
            'respiratory_symptoms',
            'headache',
            'skin_complaint',
            'dental_complaint',
            'backache',
            'injury',
            'eye_complaint',
            'diarrhoea',
            'spurious_emergency_symptom'
        }

        self.all_registered_symptoms = set()
        self.symptom_names = set()
        self.modules_that_can_impose_symptoms = set()

        self.recognised_module_names = None
        self.spurious_symptom_resolve_event = None

    def get_column_name_for_symptom(self, symptom_name):
        """get the column name that corresponds to the symptom_name"""
        return f'sy_{symptom_name}'

    def read_parameters(self, data_folder):
        """Read in the generic symptoms and register them"""
        self.parameters['generic_symptoms_spurious_occurrence'] = \
            pd.read_csv(Path(self.resourcefilepath) / 'ResourceFile_GenericSymptoms_and_HealthSeeking.csv')

    def register_symptom(self, *symptoms_to_register: Symptom):
        """
        Stores the symptom classes that are passed. Registration must be done before 'pre-initialise population' is
        called.
        The disease module associated with each symptom is also stored.
        :param symptoms_to_register: instance(s) of class Symptom
        :return:
        """

        for symptom in symptoms_to_register:
            if symptom.name not in self.symptom_names:
                self.all_registered_symptoms.add(symptom)
                self.symptom_names.add(symptom.name)
            elif symptom not in self.all_registered_symptoms:
                raise DuplicateSymptomWithNonIdenticalPropertiesError

    def register_generic_symptoms(self):
        """Register the genric symptoms, using information read in from the ResourceFile."""
        df = self.parameters['generic_symptoms_spurious_occurrence']

        # Check that information is contained in the ResourceFile for every generic symptom that must be defined
        assert self.generic_symptoms == set(df['name'].to_list())

        symptoms_to_register = df[
            [
                'name',
                'odds_ratio_health_seeking_in_children',
                'odds_ratio_health_seeking_in_adults',
                'prob_seeks_emergency_appt_in_adults',
                'prob_seeks_emergency_appt_in_children',
            ]
        ].set_index('name').loc[sorted(self.generic_symptoms)].reset_index()  # order as `sorted(self.generic_symptoms)`

        for _, _r in symptoms_to_register.iterrows():
            self.register_symptom(Symptom(**_r.to_dict()))

    def pre_initialise_population(self):
        """Register the generic symptoms and define the properties for each symptom"""

        # Register Generic Symptoms
        self.register_generic_symptoms()

        # Set-up properties for the SymptomManager module
        SymptomManager.PROPERTIES = dict()
        for symptom_name in sorted(self.symptom_names):
            symptom_column_name = self.get_column_name_for_symptom(symptom_name)
            SymptomManager.PROPERTIES[symptom_column_name] = Property(Types.INT, f'Presence of symptom {symptom_name}')

    def initialise_population(self, population):
        """
        Establish the Properties and the BitSetHandler for each of the symptoms:
        """
        self.recognised_module_names = [
            m.name for m in self.sim.modules.values() if Metadata.USES_SYMPTOMMANAGER in m.METADATA
        ]
        self.modules_that_can_impose_symptoms = [self.name] + self.recognised_module_names

        # Establish the BitSetHandler for the symptoms
        self.bsh = BitsetHandler(
            population=self.sim.population,
            column=None,
            elements=self.modules_that_can_impose_symptoms
        )
        # NB. Bitset handler will establish such that everyone has no symptoms. i.e. can check below:
        # symptom_col_names = [self.get_column_name_for_symptom(s) for s in self.symptom_names]
        # uncompressed = self.bsh.uncompress(columns=symptom_col_names)
        # for key, u in uncompressed.items():
        #     assert key in symptom_col_names
        #     assert set(u.columns) == set(modules_that_can_impose_symptoms)
        #     assert not u.any().any()

    def initialise_simulation(self, sim):
        """Schedule SpuriousSymptomsOnset/Resolve if the parameter 'spurious_symptoms' is True"""
        if self.spurious_symptoms:
            # Create and schedule the Onset Event
            sim.schedule_event(
                SymptomManager_SpuriousSymptomOnset(module=self),
                self.sim.date
            )

            # Create and schedule the Resolve event (and retain pointer to the event)
            self.spurious_symptom_resolve_event = SymptomManager_SpuriousSymptomResolve(module=self)
            sim.schedule_event(
                self.spurious_symptom_resolve_event,
                self.sim.date
            )

    def on_birth(self, mother_id, child_id):
        """Give a value of 0 for each symptom.
        NB. This will over-write any symptom that has already been set on the child, so is only safe is SymptomManager
        is registered before any Disease Module."""
        df = self.sim.population.props
        for property in self.PROPERTIES:
            df.at[child_id, property] = 0

    def change_symptom(self, person_id, symptom_string, add_or_remove, disease_module,
                       duration_in_days=None, date_of_onset=None):
        """
        This is how disease module report that a person has developed a new symptom or an existing symptom has resolved.
        The sy_ property contains a set of of the disease_module names that currently cause the symptom.
        Check if the set is empty or not to determine if the symptom is currently present.

        :param date_of_onset: Date for the symptoms to start
        :param duration_in_days: If self-resolving, duration of symptoms
        :param person_id: The person_id (int or list of int) for whom the symptom changes
        :param symptom_string: The string for the symptom or list of multiple symptom strings
        :param add_or_remove: '+' to add the symptom or '-' to remove the symptom
        :param disease_module: pointer to the disease module that is reporting this change in symptom
        """
        df = self.sim.population.props

        # Make the person_id into a list if not already a sequence
        if isinstance(person_id, (int, np.integer)):
            person_id = [person_id]

        if isinstance(symptom_string, str):
            symptom_string = [symptom_string]

        # Strip out the person_ids for anyone who is not alive:
        person_id = df.index[df.is_alive & (df.index.isin(person_id))]

        # Check that all symptoms in symptom_string are legitimate
        for sym in symptom_string:
            assert sym in self.symptom_names, f'Symptom {sym} is not recognised'

        # Check that the add/remove signal is legitimate
        assert add_or_remove in ['+', '-']

        # Check that the duration in days makes sense
        if duration_in_days is not None:
            assert int(duration_in_days) > 0

        # Check that the provided disease_module is a disease_module or is the SymptomManager itself
        assert disease_module.name in ([self.name] + self.recognised_module_names)

        # Check that a sensible or no date_of_onset is provided
        assert (date_of_onset is None) or (
            (type(date_of_onset) == pd.Timestamp)
            and (date_of_onset >= self.sim.date)
        )

        # If the date of onset if not equal to today's date, then schedule the auto_onset event
        if (date_of_onset is not None) and (date_of_onset > self.sim.date):
            auto_onset_event = SymptomManager_AutoOnsetEvent(self,
                                                             person_id=person_id,
                                                             symptom_string=symptom_string,
                                                             disease_module=disease_module,
                                                             duration_in_days=duration_in_days)
            self.sim.schedule_event(event=auto_onset_event, date=date_of_onset)
            return

        sy_columns = [self.get_column_name_for_symptom(sym) for sym in symptom_string]

        # Make the operation:
        if add_or_remove == '+':
            # Add this disease module as a cause of this symptom

            self.bsh.set(person_id, disease_module.name, columns=sy_columns)
            self._persons_with_newly_onset_symptoms = self._persons_with_newly_onset_symptoms.union(person_id)

            # If a duration is given, schedule the auto-resolve event to turn off these symptoms after specified time.
            if duration_in_days is not None:
                auto_resolve_event = SymptomManager_AutoResolveEvent(self,
                                                                     person_id=person_id,
                                                                     symptom_string=symptom_string,
                                                                     disease_module=disease_module)
                self.sim.schedule_event(event=auto_resolve_event,
                                        date=self.sim.date + DateOffset(days=int(duration_in_days)))

        else:
            # Remove this disease module as a cause of this symptom
            # But, first, check that this symptom is being caused by this disease module.
            the_disease_module_is_causing_the_symptom = \
                self.bsh.has(person_id, disease_module.name, columns=sy_columns).all().all()
            if not the_disease_module_is_causing_the_symptom:
                logger.debug(key="message",
                             data=f"Request from disease module '{disease_module.name}' to remove the symptom(s) "
                                  f"'{symptom_string}', which it is not currently causing.")

            # Do the remove:
            self.bsh.unset(person_id, disease_module.name, columns=sy_columns)

    def who_has(self, list_of_symptoms):
        """
        This is a helper function to look up who has a particular symptom or set of symptoms.
        It returns a list of indicies for person that have all of the symptoms specified

        :param: list_of_symptoms : string or list of strings for the symptoms of interest
        :return: list of person_ids for those with all of the symptoms in list_of_symptoms who are alive
        """

        # Check formatting of list_of_symptoms is right (must be a list of strings)
        if isinstance(list_of_symptoms, str):
            list_of_symptoms = [list_of_symptoms]
        else:
            list_of_symptoms = list_of_symptoms
        assert len(list_of_symptoms) > 0

        # Check that these are legitimate symptoms
        assert all([symp in self.symptom_names for symp in list_of_symptoms]), 'Symptom not registered'

        # Find who has all the symptoms
        df = self.sim.population.props
        sy_columns = [self.get_column_name_for_symptom(s) for s in list_of_symptoms]
        has_all_symptoms = self.bsh.not_empty(df.is_alive, columns=sy_columns).all(axis=1)
        return has_all_symptoms[has_all_symptoms].index.tolist()

    def who_not_have(self, symptom_string: str) -> pd.Index:
        """
        Get person IDs of individuals who are alive and do not have a symptom.

        :param symptom_string: The string of the symptom.
        :return: Index corresponding to individuals which are alive and do not have symptom.
        """

        df = self.sim.population.props

        # Check that symptom string is OK
        assert type(symptom_string) == str
        assert symptom_string in self.symptom_names, 'Symptom not registered'

        # Does not have symptom:
        return df.index[
            df.is_alive
            & self.bsh.is_empty(
                slice(None), columns=self.get_column_name_for_symptom(symptom_string)
            )
        ]

    def has_what(self, person_id, disease_module: Module = None):
        """
        This is a helper function that will give a list of strings for the symptoms that a _single_ person
        is currently experiencing.
        Optionally can specify disease_module_name to limit to the symptoms caused by that disease module

        :param person_id: the person_of of interest
        :param disease_module: (optional) disease module of interest
        :return: list of strings for the symptoms that are currently being experienced
        """

        assert isinstance(person_id, (int, np.integer)), 'person_id must be a single integer for one particular person'

        df = self.sim.population.props
        assert df.at[person_id, 'is_alive'], "The person is not alive"

        if disease_module is not None:
            assert disease_module.name in ([self.name] + self.recognised_module_names), \
                "Disease Module Name is not recognised"
            sy_columns = [self.get_column_name_for_symptom(s) for s in self.symptom_names]
            person_has = self.bsh.has(
                [person_id], disease_module.name, first=True, columns=sy_columns
            )
            return [s for s in self.symptom_names if person_has[f'sy_{s}']]
        else:
            return [s for s in self.symptom_names if df.loc[person_id, f'sy_{s}'] > 0]

    def have_what(self, person_ids: Sequence[int]):
        """Find the set of symptoms for a list of person_ids.
        NB. This is a fast implementation without the same amount checking as 'has_what'"""
        df = self.sim.population.props
        return df.loc[person_ids].apply(
            lambda p: [s for s in self.symptom_names if p[f'sy_{s}'] > 0], axis=1, result_type='reduce'
        ).rename('symptoms')

    def causes_of(self, person_id: int, symptom_string):
        """
        This is a helper function that will give a list of the disease modules causing a particular symptom for
        a particular person.
        :param person_id:
        :param disease_module:
        :return: list of strings for the disease module name
        """
        assert isinstance(person_id, (int, np.integer)), 'person_id must be a single integer for one particular person'
        assert isinstance(symptom_string, str), 'symptom_string must be a string'

        df = self.sim.population.props
        assert df.at[person_id, 'is_alive'], "The person is not alive"
        assert symptom_string in self.symptom_names

        return list(
            self.bsh.get(
                [person_id],
                first=True,
                columns=self.get_column_name_for_symptom(symptom_string)
            )
        )

    def clear_symptoms(self, person_id: Union[int, Sequence[int]], disease_module: Module):
        """
        Remove all the symptoms for one or more persons caused by a specified disease module

        :param person_id: IDs for one or more persons to clear symptoms for.
        :param disease_module_name: Name of disease module to clear symptoms for.
        """
        df = self.sim.population.props

        if isinstance(person_id, (int, np.integer)):
            person_id = [person_id]
        assert df.loc[person_id, 'is_alive'].all(), "One or more persons not alive"
        assert disease_module.name in ([self.name] + self.recognised_module_names), (
            "Disease module name is not recognised"
        )
        sy_columns = [self.get_column_name_for_symptom(sym) for sym in self.symptom_names]
        self.bsh.unset(person_id, disease_module.name, columns=sy_columns)

    def caused_by(self, disease_module: Module):
        """Find the persons experiencing symptoms due to a particular module.
        Returns a dict of the form {<<person_id>>, <<list_of_symptoms>>}."""

        df = self.sim.population.props
        symptom_columns = [self.get_column_name_for_symptom(s) for s in self.symptom_names]
        symptom_is_caused_by_disease = self.bsh.has(df.is_alive, disease_module.name, columns=symptom_columns)
        symptoms_caused_by_disease = symptom_is_caused_by_disease.apply(lambda row: list(row[row].index), axis=1)
        symptoms_for_each_person = {k: [s[3:] for s in v] for k, v in symptoms_caused_by_disease.items() if len(v) > 0}

        return symptoms_for_each_person

    def get_persons_with_newly_onset_symptoms(self):
        return self._persons_with_newly_onset_symptoms

    def reset_persons_with_newly_onset_symptoms(self):
        self._persons_with_newly_onset_symptoms.clear()

# ---------------------------------------------------------------------------------------------------------
#   EVENTS
# ---------------------------------------------------------------------------------------------------------


class SymptomManager_AutoOnsetEvent(Event, PopulationScopeEventMixin):
    """
    This utility function will add symptoms. It is scheduled by the SymptomManager to let symptoms 'auto-onset' on a
    particular date.
    """

    def __init__(self, module, person_id, symptom_string, disease_module, duration_in_days):
        super().__init__(module)
        assert isinstance(module, SymptomManager)

        if not isinstance(person_id, list):
            person_id = list(person_id)

        self.person_id = person_id
        self.symptom_string = symptom_string
        self.disease_module = disease_module
        self.duration_in_days = duration_in_days

    def apply(self, population):
        self.module.change_symptom(person_id=self.person_id,
                                   symptom_string=self.symptom_string,
                                   add_or_remove='+',
                                   disease_module=self.disease_module,
                                   duration_in_days=self.duration_in_days)


class SymptomManager_AutoResolveEvent(Event, PopulationScopeEventMixin):
    """
    This utility function will remove symptoms. It is scheduled by the SymptomManager to let symptoms 'auto-resolve'
    """

    def __init__(self, module, person_id, symptom_string, disease_module):
        super().__init__(module)
        assert isinstance(module, SymptomManager)

        if not isinstance(person_id, list):
            person_id = list(person_id)

        self.person_id = person_id
        self.symptom_string = symptom_string
        self.disease_module = disease_module

    def apply(self, population):
        self.module.change_symptom(person_id=self.person_id,
                                   symptom_string=self.symptom_string,
                                   add_or_remove='-',
                                   disease_module=self.disease_module)


class SymptomManager_SpuriousSymptomOnset(RegularEvent, PopulationScopeEventMixin):
    """ This event gives the occurrence of generic symptoms that are not caused by a disease module in the TLO model.
    """

    def __init__(self, module):
        """This event occurs every day"""
        super().__init__(module, frequency=DateOffset(days=1))

        assert isinstance(module, SymptomManager)
        self.generic_symptoms = self.get_generic_symptoms_dict(
            self.module.parameters['generic_symptoms_spurious_occurrence'])
        self.rand = self.module.rng.rand

    def get_generic_symptoms_dict(self, generic_sympoms_df):
        """Helper function to store contents of the generic_symptoms dataframe as dicts"""
        df = generic_sympoms_df.set_index('name')

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
        """Determine who will be onset which which symptoms today"""

        df = self.sim.population.props
        group_indices = {
            'children': df.index[df.is_alive & (df.age_years < 15)],
            'adults': df.index[df.is_alive & (df.age_years >= 15)]
        }

        # For each generic symptom, impose it on a random sample of persons who do not have that symptom currently:
        for symp in sorted(self.module.generic_symptoms):
            do_not_have_symptom = self.module.who_not_have(symptom_string=symp)

            for group in ['children', 'adults']:

                p = self.generic_symptoms['prob_per_day'][group][symp]
                dur = self.generic_symptoms['duration_in_days'][group][symp]
                persons_eligible_to_get_symptom = group_indices[group][
                    group_indices[group].isin(do_not_have_symptom)
                ]
                persons_to_onset_with_this_symptom = persons_eligible_to_get_symptom[
                    self.rand(len(persons_eligible_to_get_symptom)) < p
                ]

                # Do onset
                self.sim.modules['SymptomManager'].change_symptom(
                    symptom_string=symp,
                    add_or_remove='+',
                    person_id=persons_to_onset_with_this_symptom,
                    duration_in_days=None,   # <- resolution for these is handled by the SpuriousSymptomsResolve Event
                    disease_module=self.module,
                )

                # Schedule resolution:
                self.module.spurious_symptom_resolve_event.schedule_symptom_resolve(
                    person_id=persons_to_onset_with_this_symptom,
                    symptom_string=symp,
                    date_of_resolution=(self.sim.date + pd.DateOffset(days=dur)).date()
                )


class SymptomManager_SpuriousSymptomResolve(RegularEvent, PopulationScopeEventMixin):
    """ This event resolves the generic symptoms that have been onset by this module.
    """

    def __init__(self, module):
        """This event occurs every day"""
        super().__init__(module, frequency=DateOffset(days=1))
        assert isinstance(module, SymptomManager)

        # Create the dict structures to store information about for whom and when each symptoms must be resolved
        self.to_resolve = dict()
        for symp in sorted(self.module.generic_symptoms):
            self.to_resolve[symp] = defaultdict(set)

    def schedule_symptom_resolve(self, person_id, date_of_resolution, symptom_string):
        """Store information to allow symptoms to be resolved for groups of persons each day"""
        self.to_resolve[symptom_string][date_of_resolution].update(person_id)

    def apply(self, population):
        """Resolve the symptoms when due; a whole group of persons with the same symptoms at once"""
        df = population.props
        date_today = self.sim.date.date()

        for symp in self.to_resolve.keys():
            if date_today in self.to_resolve[symp]:
                person_ids = self.to_resolve[symp].pop(date_today)
                persons = df.loc[person_ids]
                person_ids_alive = persons[persons.is_alive].index
                self.module.change_symptom(
                    person_id=person_ids_alive,
                    add_or_remove='-',
                    symptom_string=symp,
                    disease_module=self.module
                )
