"""
The Symptom Manager:
* Manages presence of symptoms for all disease modules
* Manages a set of generic symptoms
* Creates occurrences of generic symptom (representing that being caused by diseases not included in the TLO model)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, PopulationScopeEventMixin, RegularEvent
# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------
from tlo.util import BitsetHandler


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
    print("A symptom with this name has been registered already but with different proprtie")
    pass


class SymptomManager(Module):
    """
    This module is used to track the symptoms of persons. The addition and removal of symptoms is handled here.
    """

    PROPERTIES = dict()  # updated as pre-initialise population once symptoms have been registered.

    PARAMETERS = {
        'generic_symptoms': Parameter(Types.LIST, 'List of generic symptoms'),
        'generic_symptoms_spurious_occurrence': Parameter(Types.DATA_FRAME, 'probability and duration of spurious '
                                                                            'occureneces of generic symptoms')
    }

    def __init__(self, name=None, resourcefilepath=None, spurious_symptoms=False):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.spurious_symptoms = spurious_symptoms
        self.persons_with_newly_onset_symptoms = set()

        self.all_registered_symptoms = set()
        self.symptom_names = set()

    def get_column_name_for_symptom(self, symptom_name):
        """get the column name that corresponds to the symptom_name"""
        return f'sy_{symptom_name}'

    def read_parameters(self, data_folder):
        """Read in the generic symptoms and register them"""

        # Define the Generic Symptoms
        generic_symptoms = pd.read_csv(Path(self.resourcefilepath) /
                                       'ResourceFile_GenericSymptoms_and_HealthSeeking.csv')

        # ensure types are correct:
        generic_symptoms['duration_in_days_of_spurious_occurrence_in_children'] = \
            generic_symptoms['duration_in_days_of_spurious_occurrence_in_children'].astype(int)
        generic_symptoms['duration_in_days_of_spurious_occurrence_in_adults'] = \
            generic_symptoms['duration_in_days_of_spurious_occurrence_in_adults'].astype(int)

        generic_symptoms.set_index('generic_symptom_name', drop=True, inplace=True)
        self.parameters['generic_symptoms'] = list(generic_symptoms.index)
        self.parameters['generic_symptoms_spurious_occurrence'] = \
            generic_symptoms[['prob_spurious_occurrence_in_children_per_month',
                              'prob_spurious_occurrence_in_adults_per_month',
                              'duration_in_days_of_spurious_occurrence_in_children',
                              'duration_in_days_of_spurious_occurrence_in_adults'
                              ]]

        # Register the Generic Symptoms
        for generic_symptom_name in generic_symptoms.index:
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
                self.all_registered_symptoms = self.all_registered_symptoms.union({symptom})
                self.symptom_names = self.symptom_names.union({symptom.name})
            elif symptom not in self.all_registered_symptoms:
                raise DuplicateSymptomWithNonIdenticalPropertiesError

    def pre_initialise_population(self):
        """Define the properties for each symptom"""
        SymptomManager.PROPERTIES = dict()
        for symptom_name in self.symptom_names:
            symptom_column_name = self.get_column_name_for_symptom(symptom_name)
            SymptomManager.PROPERTIES[symptom_column_name] = Property(Types.INT, f'Presence of symptom {symptom_name}')

    def initialise_population(self, population):
        """
        Establish the BitSetHandler for each of the symptoms:
        """
        modules_that_can_impose_symptoms = [self.name] + self.sim.disease_modules_name

        # Establish the BitSetHandler for each symptoms
        self.bsh = dict()
        for symptom_name in self.symptom_names:
            symptom_column_name = self.get_column_name_for_symptom(symptom_name)
            self.bsh[symptom_name] = BitsetHandler(self.sim.population,
                                                   symptom_column_name,
                                                   modules_that_can_impose_symptoms)

            # Check that all individuals do not have this symptom currently:
            u = self.bsh[symptom_name].uncompress()
            assert set(u.columns) == set(modules_that_can_impose_symptoms)
            assert not u.any().any()

    def initialise_simulation(self, sim):
        """Schedule SpuriousSymptomsGenerator if parameter 'spurious_symptoms' is True"""
        if self.spurious_symptoms:
            sim.schedule_event(
                SymptomManager_SpuriousSymptomGenerator(self),
                self.sim.date
            )

    def on_birth(self, mother_id, child_id):
        pass

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
        assert disease_module.name in ([self.name] + self.sim.disease_modules_name)

        # Check that a sensible or no date_of_onset is provided
        assert (date_of_onset is None) or (isinstance(date_of_onset, pd.Timestamp) and date_of_onset >= self.sim.date)

        # If the date of onset if not equal to today's date, then schedule the auto_onset event
        if date_of_onset is not None:
            auto_onset_event = SymptomManager_AutoOnsetEvent(self,
                                                             person_id=person_id,
                                                             symptom_string=symptom_string,
                                                             disease_module=disease_module,
                                                             duration_in_days=duration_in_days)
            self.sim.schedule_event(event=auto_onset_event, date=date_of_onset)

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
        :return: list of person_ids for those with all of the symptoms in list_of_symptoms
        """

        # Check formatting of list_of_symptoms is right (must be a list of strings)
        if isinstance(list_of_symptoms, str):
            list_of_symptoms = [list_of_symptoms]
        else:
            list_of_symptoms = list_of_symptoms
        assert len(list_of_symptoms) > 0

        # Check that these are legitimate symptoms
        assert all([symp in self.symptom_names for symp in list_of_symptoms]), 'Symptom not registered'

        # get the person_id for those who have each symptom
        has_this_symptom = dict()
        for symptom in list_of_symptoms:
            # u = self.bsh[symptom].has_any(df.is_alive)  todo - has_any() doesn't seem to work as expected
            u = self.bsh[symptom].uncompress().any(axis=1)
            has_this_symptom[symptom] = set(u[u].index)

        # find the people who have each of the symptoms listed
        has_all_symptoms = list(set.intersection(*[has_this_symptom[symptom] for symptom in has_this_symptom]))

        return has_all_symptoms

    def has_what(self, person_id, disease_module=None):
        """
        This is a helper function that will give a list of strings for the symptoms that a person
        is currently experiencing.
        Optionally can specify disease_module_name to limit to the symptoms caused by that disease module

        :param person_id: the person_of of interest
        :param disease_module: (optional) disease module of interest.
        :return: list of strings for the symptoms that are currently being experienced
        """

        assert isinstance(person_id, (int, np.int64)), 'person_id must be a single integer for one particular person'

        df = self.sim.population.props
        assert df.at[person_id, 'is_alive'], "The person is not alive"

        if disease_module:
            assert disease_module.name in self.sim.disease_modules_name, \
                "Disease Module Name is not recognised"
            disease_modules_of_interest = [disease_module.name]
        else:
            disease_modules_of_interest = self.sim.disease_modules_name

        symptoms_for_this_person = list()
        for symptom in self.symptom_names:
            if self.bsh[symptom].uncompress([person_id]).loc[person_id, disease_modules_of_interest].any():
                # todo: use has_any built-in function?
                symptoms_for_this_person.append(symptom)

        return symptoms_for_this_person

    def causes_of(self, person_id, symptom_string):
        """
        This is a helper function that will give a list of the disease modules causing a particular symptom for
        a particular person.
        :param person_id:
        :param disease_module:
        :return: list of strings for the disease module name
        """
        assert isinstance(person_id, (int, np.int64)), 'person_id must be a single integer for one particular person'
        assert isinstance(symptom_string, str), 'symptom_string must be a string'

        df = self.sim.population.props
        assert df.at[person_id, 'is_alive'], "The person is not alive"
        assert symptom_string in self.symptom_names

        return list(self.bsh[symptom_string].get([person_id])[person_id])

    def clear_symptoms(self, person_id, disease_module):
        """
        This is a helper function that remove all the symptoms in a specified person that is caused by a specified
        disease module

        :param person_id:
        :param disease_module_name:
        """
        df = self.sim.population.props

        assert isinstance(person_id, (int, np.int64)), 'person_id must be a single integer for one particular person'
        assert df.at[person_id, 'is_alive'], "The person is not alive"
        assert disease_module.name in self.sim.disease_modules_name, "Disease Module Name is not recognised"

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

        # strip out those who do not have this symptom being caused by this disease_module
        for person_id in people_to_resolve:
            if self.symptom_string not in self.module.has_what(person_id, disease_module=self.disease_module):
                people_to_resolve = people_to_resolve.remove(person_id)

        # run the chg_symptom function
        if people_to_resolve:
            self.module.change_symptom(person_id=people_to_resolve,
                                       symptom_string=self.symptom_string,
                                       add_or_remove='-',
                                       disease_module=self.disease_module)


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


class SymptomManager_SpuriousSymptomGenerator(RegularEvent, PopulationScopeEventMixin):
    """ This event gives the occurrence of generic symptoms that are not caused by a disease module in the TLO model.
    The symptoms occur at a randomly-selected time during the month.
    """

    def __init__(self, module):
        """This event occurs every month"""
        super().__init__(module, frequency=DateOffset(months=1))
        assert isinstance(module, SymptomManager)

    def apply(self, population):
        params = self.module.parameters['generic_symptoms_spurious_occurrence']

        # get indices of adults and children
        df = population.props
        children_idx = df.loc[df.is_alive & (df.age_years < 15)].index
        adults_idx = df.loc[df.is_alive & (df.age_years >= 15)].index

        def random_date(start, end):
            """Generate a random datetime between `start` and `end`"""
            return start + DateOffset(
                # Get a random amount of seconds between `start` and `end`
                seconds=self.module.rng.randint(0, int((end - start).total_seconds())),
            )

        # for each generic symptom, impose it on a random sample of persons
        for symp in params.index:
            # children:
            p_symp_children = params.at[symp, 'prob_spurious_occurrence_in_children_per_month']
            dur_symp_children = params.at[symp, 'duration_in_days_of_spurious_occurrence_in_children']
            children_to_onset_with_this_symptom = \
                list(children_idx[self.module.rng.rand(len(children_idx)) < p_symp_children])
            for child in children_to_onset_with_this_symptom:
                self.sim.modules['SymptomManager'].change_symptom(
                    symptom_string=symp,
                    add_or_remove='+',
                    person_id=child,
                    date_of_onset=random_date(self.sim.date, self.sim.date + DateOffset(months=1)),
                    duration_in_days=dur_symp_children,
                    disease_module=self.module
                )

            # adults:
            p_symp_adults = params.at[symp, 'prob_spurious_occurrence_in_adults_per_month']
            dur_symp_adults = params.at[symp, 'duration_in_days_of_spurious_occurrence_in_adults']
            adults_to_onset_with_this_symptom = list(adults_idx[self.module.rng.rand(len(adults_idx)) < p_symp_adults])
            for adult in adults_to_onset_with_this_symptom:
                self.sim.modules['SymptomManager'].change_symptom(
                    symptom_string=symp,
                    add_or_remove='+',
                    person_id=adult,
                    date_of_onset=random_date(self.sim.date, self.sim.date + DateOffset(months=1)),
                    duration_in_days=dur_symp_adults,
                    disease_module=self.module
                )
