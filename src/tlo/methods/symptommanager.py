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
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, PopulationScopeEventMixin, RegularEvent
from tlo.methods import Metadata

from collections import defaultdict
from tlo.util import BitsetHandler

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------



class Symptom:
    """Data structure to hold the information about a symptom.
    Adult is peron aged 15+
    Child is someone aged <15

    The assumption is that symptom tend to cause health-care seeking. This can be modified by specifying that the
    healthcare seeking is an emergency, or is more/less likely than the 'average symptom', or that the symptom does not
    cause healthcare seeking at all.
    The default behaviour is that a symptom causes health care seeking in the same manner as does the 'average symptom'.

    """

    def __init__(self,
                 name: str = None,
                 no_healthcareseeking_in_adults: bool = False,
                 no_healthcareseeking_in_children: bool = False,
                 emergency_in_adults: bool = False,
                 emergency_in_children: bool = False,
                 odds_ratio_health_seeking_in_adults: float = None,
                 odds_ratio_health_seeking_in_children: float = None
                 ):

        # Check that the types are correct and not nonsensical
        assert isinstance(name, str)
        assert name, 'name of symptom cannot be blank'

        assert isinstance(no_healthcareseeking_in_adults, bool)
        assert isinstance(no_healthcareseeking_in_children, bool)

        assert isinstance(emergency_in_adults, bool)
        assert isinstance(emergency_in_children, bool)

        # Check logic of the arguments that are provided:
        # 1) if the symptom does not cause healthseeking behaviour, it should not be emergency or associated with an
        # odds ratio
        if no_healthcareseeking_in_children:
            assert emergency_in_children is False
            assert odds_ratio_health_seeking_in_children is None

        if no_healthcareseeking_in_adults:
            assert emergency_in_adults is False
            assert odds_ratio_health_seeking_in_adults is None

        # 2) if the symptom is declared as an emergency, it cannot also have an odds ratio for health seeking
        if emergency_in_children:
            assert no_healthcareseeking_in_children is False
            assert odds_ratio_health_seeking_in_children is None

        if emergency_in_adults:
            assert no_healthcareseeking_in_adults is False
            assert odds_ratio_health_seeking_in_adults is None

        # 3) if an odds-ratio is specified, it cannot have the emergency or the no-seeking flags
        if odds_ratio_health_seeking_in_children is not None:
            assert emergency_in_children is False
            assert no_healthcareseeking_in_children is False
            assert isinstance(odds_ratio_health_seeking_in_children, float)
            assert 0 < odds_ratio_health_seeking_in_children

        if odds_ratio_health_seeking_in_adults is not None:
            assert emergency_in_adults is False
            assert no_healthcareseeking_in_adults is False
            assert isinstance(odds_ratio_health_seeking_in_adults, float)
            assert 0 < odds_ratio_health_seeking_in_adults

        # If odds-ratios are not provided (and no other flags provided), default to values of 1.0
        if (
            (odds_ratio_health_seeking_in_children is None) &
            (emergency_in_children is False) &
            (no_healthcareseeking_in_children is False)
        ):
            odds_ratio_health_seeking_in_children = 1.0

        if (
            (odds_ratio_health_seeking_in_adults is None) &
            (emergency_in_adults is False) &
            (no_healthcareseeking_in_adults is False)
        ):
            odds_ratio_health_seeking_in_adults = 1.0

        # Store properties:
        self.name = name
        self.no_healthcareseeking_in_children = no_healthcareseeking_in_children
        self.no_healthcareseeking_in_adults = no_healthcareseeking_in_adults
        self.emergency_in_adults = emergency_in_adults
        self.emergency_in_children = emergency_in_children
        self.odds_ratio_health_seeking_in_adults = odds_ratio_health_seeking_in_adults
        self.odds_ratio_health_seeking_in_children = odds_ratio_health_seeking_in_children


class DuplicateSymptomWithNonIdenticalPropertiesError(Exception):
    def __init__(self):
        super().__init__("A symptom with this name has been registered already but with different properties")


class SymptomManager(Module):
    """
    This module is used to track the symptoms of persons. The addition and removal of symptoms by disease modules is
     handled here. This module can also causes symptoms that are not related to any disease module (representing those
     caused by conditions not represented explicitly in the model).
    """

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
        self.persons_with_newly_onset_symptoms = set()
        self.all_registered_symptoms = set()
        self.symptom_names = set()
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
        :disease_module: the disease module that is registering a particular symptom
        :param symptoms_to_register: instance(s) of class Symptom
        :return:
        """

        for symptom in symptoms_to_register:
            if symptom.name not in self.symptom_names:
                self.all_registered_symptoms.add(symptom)
                self.symptom_names.add(symptom.name)
            elif symptom not in self.all_registered_symptoms:
                raise DuplicateSymptomWithNonIdenticalPropertiesError

    def process_and_register_generic_symptoms_params(self):
        """Process the file that has been read into the parameters for genric symptoms and their occurences"""

        generic_symptoms = self.parameters['generic_symptoms_spurious_occurrence'].copy()

        # ensure types are correct:
        generic_symptoms['duration_in_days_of_spurious_occurrence_in_children'] = \
            generic_symptoms['duration_in_days_of_spurious_occurrence_in_children'].astype(int)
        generic_symptoms['duration_in_days_of_spurious_occurrence_in_adults'] = \
            generic_symptoms['duration_in_days_of_spurious_occurrence_in_adults'].astype(int)
        generic_symptoms.set_index('generic_symptom_name', drop=True, inplace=True)

        # Store the data on generic symptoms
        self.generic_symptoms = generic_symptoms

        # Register the Generic Symptoms
        for generic_symptom_name in self.generic_symptoms.index:
            self.register_symptom(
                Symptom(
                    name=generic_symptom_name,
                    odds_ratio_health_seeking_in_adults=generic_symptoms.at[
                        generic_symptom_name, 'odds_ratio_for_health_seeking_in_adults'],
                    odds_ratio_health_seeking_in_children=generic_symptoms.at[
                        generic_symptom_name, 'odds_ratio_for_health_seeking_in_children'],
                    emergency_in_adults=False,
                    emergency_in_children=False
                )
            )

    def pre_initialise_population(self):
        """Define the properties for each symptom"""

        self.process_and_register_generic_symptoms_params()

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
        modules_that_can_impose_symptoms = [self.name] + self.recognised_module_names

        # Establish the BitSetHandler for each symptoms
        self.bsh = dict()
        for symptom_name in self.symptom_names:
            symptom_column_name = self.get_column_name_for_symptom(symptom_name)
            self.bsh[symptom_name] = BitsetHandler(self.sim.population,
                                                   symptom_column_name,
                                                   modules_that_can_impose_symptoms)

            # NB. Bit Set Handler will establish such that everyone has no symptoms. i.e. check below:
            # u = self.bsh[symptom_name].uncompress()
            # assert set(u.columns) == set(modules_that_can_impose_symptoms)
            # assert not u.any().any()

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
        :param symptom_string: The string for the symptom (must be one of the generic_symptoms)
        :param add_or_remove: '+' to add the symptom or '-' to remove the symptom
        :param disease_module: pointer to the disease module that is reporting this change in symptom
        """
        df = self.sim.population.props

        # Make the person_id into a list
        if not isinstance(person_id, list):
            person_id = [person_id]

        # Strip out the person_ids for anyone who is not alive.
        person_id = list(df.index[df.is_alive & (df.index.isin(person_id))])

        # Check that the symptom_string is legitimate
        assert symptom_string in self.symptom_names, f'Symptom {symptom_string} is not recognised'
        symptom_var_name = 'sy_' + symptom_string
        assert symptom_var_name in df.columns, 'Symptom has not been declared'

        # Check that the add/remove signal is legitimate
        assert add_or_remove in ['+', '-']

        # Check that the duration in days makes sense
        if duration_in_days is not None:
            assert int(duration_in_days) > 0

        # Check that the provided disease_module is a disease_module or is the SymptomManager itself
        assert disease_module.name in ([self.name] + self.recognised_module_names)

        # Check that a sensible or no date_of_onset is provided
        assert (date_of_onset is None) or (isinstance(date_of_onset, pd.Timestamp) and date_of_onset >= self.sim.date)

        # If the date of onset if not equal to today's date, then schedule the auto_onset event
        if (date_of_onset is not None) and (date_of_onset > self.sim.date):
            auto_onset_event = SymptomManager_AutoOnsetEvent(self,
                                                             person_id=person_id,
                                                             symptom_string=symptom_string,
                                                             disease_module=disease_module,
                                                             duration_in_days=duration_in_days)
            self.sim.schedule_event(event=auto_onset_event, date=date_of_onset)
            return

        # Make the operation:
        if add_or_remove == '+':
            # Add this disease module as a cause of this symptom

            self.bsh[symptom_string].set(person_id, disease_module.name)
            self.persons_with_newly_onset_symptoms = self.persons_with_newly_onset_symptoms.union(person_id)

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
            # Check that this symptom is being caused by this diease module.
            assert self.bsh[symptom_string].uncompress(person_id)[disease_module.name].all(), \
                'Error - request from disease module to remove a symptom that it has not caused.'
            # Do the remove:
            self.bsh[symptom_string].unset(person_id, disease_module.name)

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
        has_all_symptoms = pd.Series(index=df.index[df.is_alive], data=True)
        for s in list_of_symptoms:
            has_all_symptoms = has_all_symptoms & self.bsh[s].not_empty(df.is_alive)

        return has_all_symptoms[has_all_symptoms].index.tolist()

    def has_what(self, person_id, disease_module=None):
        """
        This is a helper function that will give a list of strings for the symptoms that a person
        is currently experiencing.
        Optionally can specify disease_module_name to limit to the symptoms caused by that disease module

        :param person_id: the person_of of interest
        :param disease_module: (optional) disease module of interest
        :return: list of strings for the symptoms that are currently being experienced
        """

        assert isinstance(person_id, (int, np.integer)), 'person_id must be a single integer for one particular person'

        df = self.sim.population.props
        assert df.at[person_id, 'is_alive'], "The person is not alive"

        columns = [self.get_column_name_for_symptom(s) for s in self.symptom_names]
        df = self.sim.population.props
        person = df.loc[person_id, columns]

        if disease_module:
            assert disease_module.name in ([self.name] + self.recognised_module_names), \
                "Disease Module Name is not recognised"
            return [s for s in self.symptom_names if disease_module.name in self.bsh[s].to_strings(person[f'sy_{s}'])]

        return [s for s in self.symptom_names if person[f'sy_{s}'] > 0]

    def causes_of(self, person_id, symptom_string):
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

        return list(self.bsh[symptom_string].get([person_id], first=True))

    def clear_symptoms(self, person_id, disease_module):
        """
        This is a helper function that remove all the symptoms in a specified person that is caused by a specified
        disease module

        :param person_id:
        :param disease_module_name:
        """
        df = self.sim.population.props

        assert isinstance(person_id, (int, np.integer)), 'person_id must be a single integer for one particular person'
        assert df.at[person_id, 'is_alive'], "The person is not alive"
        assert disease_module.name in ([self.name] + self.recognised_module_names), \
            "Disease Module Name is not recognised"

        symptoms_caused_by_this_disease_module = self.has_what(person_id, disease_module)

        for symp in symptoms_caused_by_this_disease_module:
            self.change_symptom(
                person_id=person_id,
                symptom_string=symp,
                add_or_remove='-',
                disease_module=disease_module
            )


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
        # strip out those who are not alive
        df = population.props
        people_to_get_symptom = list(df.index[df.is_alive & (df.index.isin(self.person_id))])

        self.module.change_symptom(person_id=people_to_get_symptom,
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
        # strip out those who are not alive
        df = population.props
        people_to_resolve = list(df.index[df.is_alive & (df.index.isin(self.person_id))])

        # find the person_id's for those have this symptom (and this symptom caused by a disease_module if specified)
        bsh = self.module.bsh[self.symptom_string]
        have_symptom_from_disease = bsh.has_any(df.index.isin(self.person_id) & df.is_alive, self.disease_module.name)
        people_index = have_symptom_from_disease.index[have_symptom_from_disease]

        # run the chg_symptom function
        if len(people_index) > 0:
            self.module.change_symptom(person_id=people_to_resolve,
                                       symptom_string=self.symptom_string,
                                       add_or_remove='-',
                                       disease_module=self.disease_module)





class SymptomManager_SpuriousSymptomOnset(RegularEvent, PopulationScopeEventMixin):
    """ This event gives the occurrence of generic symptoms that are not caused by a disease module in the TLO model.
    """

    def __init__(self, module):
        """This event occurs every day"""
        super().__init__(module, frequency=DateOffset(day=1))
        assert isinstance(module, SymptomManager)
        self.generic_symptoms = self.module.generic_symptoms

    def apply(self, population):
        """Determine who will be onset which which symptoms today"""

        # get indices of adults and children
        df = population.props
        children_idx = df.loc[df.is_alive & (df.age_years < 15)].index
        adults_idx = df.loc[df.is_alive & (df.age_years >= 15)].index

        # for each generic symptom, impose it on a random sample of persons
        for symp in self.generic_symptoms.index:
            # children:
            p_symp_children = self.generic_symptoms.at[symp, 'prob_spurious_occurrence_in_children_per_day']
            dur_symp_children = self.generic_symptoms.at[symp, 'duration_in_days_of_spurious_occurrence_in_children']
            children_to_onset_with_this_symptom = \
                list(children_idx[self.module.rng.rand(len(children_idx)) < p_symp_children])

            self.sim.modules['SymptomManager'].change_symptom(
                symptom_string=symp,
                add_or_remove='+',
                person_id=children_to_onset_with_this_symptom,
                duration_in_days=None,   # <- resolution for these is handled by the SpuriousSymptomsResolve Event
                disease_module=self.module,
            )
            # Schedule resolution:
            self.module.spurious_symptom_resolve_event.schedule_symptom_resolve(
                person_id=children_to_onset_with_this_symptom,
                symptom_string=symp,
                date_of_resolution=(self.sim.date + pd.DateOffset(days=int(dur_symp_children))).date()
            )

            # adults:
            p_symp_adults = self.generic_symptoms.at[symp, 'prob_spurious_occurrence_in_adults_per_day']
            dur_symp_adults = self.generic_symptoms.at[symp, 'duration_in_days_of_spurious_occurrence_in_adults']
            adults_to_onset_with_this_symptom = list(adults_idx[self.module.rng.rand(len(adults_idx)) < p_symp_adults])

            self.sim.modules['SymptomManager'].change_symptom(
                symptom_string=symp,
                add_or_remove='+',
                person_id=adults_to_onset_with_this_symptom,
                duration_in_days=None,   # <- resolution for these is handled by the SpuriousSymptomsResolve Event
                disease_module=self.module
            )
            # Schedule resolution:
            self.module.spurious_symptom_resolve_event.schedule_symptom_resolve(
                person_id=adults_to_onset_with_this_symptom,
                symptom_string=symp,
                date_of_resolution=(self.sim.date + pd.DateOffset(days=int(dur_symp_adults))).date()
            )


class SymptomManager_SpuriousSymptomResolve(RegularEvent, PopulationScopeEventMixin):
    """ This event resolves the generic symptoms that have been onset by this module.
    """

    def __init__(self, module):
        """This event occurs every day"""
        super().__init__(module, frequency=DateOffset(days=1))
        assert isinstance(module, SymptomManager)

        self.generic_symptoms = self.module.generic_symptoms

        # Create the dict structures to store information about for whom and when each symptoms must be resolved
        self.to_resolve = dict()
        for symp in self.generic_symptoms.index:
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
                if len(person_ids) > 0:
                    person_ids_alive = list(df.index[df.is_alive & (df.index.isin(person_ids))])
                    self.module.change_symptom(
                        person_id= person_ids_alive,
                        add_or_remove='-',
                        symptom_string=symp,
                        disease_module=self.module
                    )
