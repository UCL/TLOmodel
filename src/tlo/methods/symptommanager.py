"""
The Symptom Manager:
* Manages presence of symptoms for all disease modules
* Manages a set of generic symptoms
* Creates occurrences of generic symptom (representing that being caused by diseases not included in the TLO model)
"""

from pathlib import Path

import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, PopulationScopeEventMixin, RegularEvent


# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------
class Symptom():
    """Data structure to hold the information about a symptom"""
    def __init__(self,
                 name: str = None,
                 emergency_in_adults: bool = False,
                 emergency_in_children: bool = False,
                 odds_ratio_health_seeking_in_adults: float = 1.0,
                 odds_ratio_health_seeking_in_children: float = 1.0
                 ):

        # Check that the types are correct and not nonsensical
        assert isinstance(name, str)
        assert name is not ''

        assert isinstance(emergency_in_adults, bool)
        assert isinstance(emergency_in_children, bool)

        assert isinstance(odds_ratio_health_seeking_in_adults, float)
        assert 0 < odds_ratio_health_seeking_in_adults

        assert isinstance(odds_ratio_health_seeking_in_children, float)
        assert 0 < odds_ratio_health_seeking_in_children

        # if the symptom is declared as an emrgency it cannot also have an odds ratio for health seeking, for that age of person
        if emergency_in_adults:
            assert 1.0 == odds_ratio_health_seeking_in_adults

        if emergency_in_children:
            assert 1.0 == odds_ratio_health_seeking_in_children

        self.name = name
        self.emergency_in_adults = emergency_in_adults
        self.emergency_in_children = emergency_in_children
        self.odds_ratio_health_seeking_in_adults = odds_ratio_health_seeking_in_adults
        self.odds_ratio_health_seeking_in_children = odds_ratio_health_seeking_in_children


class SymptomManager(Module):
    """
    This module is used to track the symptoms of persons. The addition and removal of symptoms is handled here.
    """

    PROPERTIES = dict()  # give blank definition of parameters here. It's updated in 'pre_initialise_population'

    PARAMETERS = {
        'generic_symptoms': Parameter(Types.LIST, 'List of generic symptoms')
    }

    def __init__(self, name=None, resourcefilepath=None, spurious_symptoms=False):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.spurious_symptoms = spurious_symptoms
        self.persons_with_newly_onset_symptoms = set()

        self.all_registered_symptoms = set()
        self.symptom_names = []
        self.symptom_column_names = []

        self.emergency_symptoms_for_adults = set()
        self.emergency_symptoms_for_children = set()
        self.odds_ratio_for_health_care_seeking_for_adults = dict()
        self.odds_ratio_for_health_care_seeking_for_children = dict()

    def read_parameters(self, data_folder):
        """Read in the generic symptoms and register them"""

        # Define the Generic Symptoms
        generic_symptoms = pd.read_csv(Path(self.resourcefilepath) / 'ResourceFile_GenericSymptoms_and_HealthSeeking.csv')
        generic_symptoms.set_index('generic_symptom_name', drop=True, inplace=True)
        self.parameters['generic_symptoms'] = list(generic_symptoms.index)

        # Register the Generic Symptoms
        for generic_symptom_name in generic_symptoms.index:
            self.register_symptom(
                Symptom(
                    name=generic_symptom_name,
                    odds_ratio_health_seeking_in_adults=generic_symptoms.at[generic_symptom_name, 'odds_ratio_for_health_seeking_in_adults'],
                    odds_ratio_health_seeking_in_children=generic_symptoms.at[generic_symptom_name, 'odds_ratio_for_health_seeking_in_children'],
                    emergency_in_adults=False,
                    emergency_in_children=False
                )
            )

    def pre_initialise_population(self):
        """
        Make the properties for each symptom.
        :return:
        """
        # Clear any existing properties (PROPERTIES is a class member, so may have been set elsewhere)
        SymptomManager.PROPERTIES = dict()

        # Make property for each symptom
        self.symptom_names = sorted(self.symptom_names)
        for symptom_name in self.symptom_names:
            symptom_column_name = f'sy_{symptom_name}'
            self.symptom_column_names.append(symptom_column_name)
            SymptomManager.PROPERTIES[symptom_column_name] = Property(Types.LIST, f'Presence of symptom {symptom_name}')

    def initialise_population(self, population):
        """
        Give all individuals the no symptoms (ie. an empty set)
        """
        for person_id in list(population.props.index):
            for symptom_var in self.symptom_column_names:
                population.props.at[person_id, symptom_var] = set()

    def initialise_simulation(self, sim):
        """Schedule SpuriousSymptomsGenerator if parameter 'spurious_symptoms' is True"""
        if self.spurious_symptoms:
            sim.schedule_event(
                SymptomManager_SpuriousSymptomGenerator(),
                self.sim.date
            )

    def on_birth(self, mother_id, child_id):
        """
        Set that the child will have no symptoms by default (empty set)
        """
        df = self.sim.population.props
        for symptom_var in self.symptom_column_names:
            df.at[child_id, symptom_var] = set()

    def register_symptom(self, *symptoms_to_register: Symptom):
        """
        Stores the symptom classes that are passed
        :param symptoms_to_register: instance(s) of class Symptom
        :return:
        """
        for symptom in symptoms_to_register:
            self.all_registered_symptoms = self.all_registered_symptoms.union({symptom})
            self.symptom_names.append(symptom.name)

    def change_symptom(self, person_id, symptom_string, add_or_remove, disease_module,
                       duration_in_days=None, date_of_onset=None):
        """
        This is how disease module report that a person has developed a new symptom or an existing symptom has resolved.
        The sy_ property contains a set of of the disease_module names that currenly cause the symptom.
        Check if the set is empty or not to determine if the sympton is currently present.

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
        assert symptom_string in self.symptom_names, 'Symptom is not recognised'
        symptom_var_name = 'sy_' + symptom_string
        assert symptom_var_name in df.columns, 'Symptom has not been declared'

        # Check that the add/remove signal is legitimate
        assert add_or_remove in ['+', '-']

        # Check that the duration in days makes sense
        if duration_in_days is not None:
            assert int(duration_in_days) > 0

        # Check that the provided disease_module is a registered disease_module
        assert disease_module in self.sim.modules['HealthSystem'].registered_disease_modules.values()

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

            _ = df.loc[person_id, symptom_var_name].apply(lambda x: x.add(disease_module.name))

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
            assert df.loc[person_id, symptom_var_name].apply(lambda x: (disease_module.name in x)).all(), (
                'Error - request from disease module to remove a symptom that it has not caused. ')

            _ = df.loc[person_id, symptom_var_name].apply(lambda x: x.remove(disease_module.name))

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
        df = self.sim.population.props
        symptom_columns = [f'sy_{col}' for col in list_of_symptoms]
        people_with_all_symptoms_mask = df.loc[df.is_alive, symptom_columns].applymap(lambda x: x != set()).all(axis=1)

        return list(people_with_all_symptoms_mask[people_with_all_symptoms_mask].index)

    def has_what(self, person_id, disease_module=None):
        """
        This is a helper function that will give a list of strings for the symptoms that a person
        is currently experiencing.
        Optionally can specify disease_module_name to limit to the symptoms caused by that disease module

        :param person_id: the person_of of interest
        :param disease_module: (optional) disease module of interest.
        :return: list of strings for the symptoms that are currently being experienced
        """
        df = self.sim.population.props

        profile = df.loc[person_id, self.symptom_column_names]

        assert df.at[person_id, 'is_alive'], "The person is not alive"

        if disease_module:
            assert disease_module in self.sim.modules['HealthSystem'].registered_disease_modules.values(), \
                "Disease Module Name is not recognised"

            def filter_symptoms(x):
                return disease_module.name in x
        else:

            def filter_symptoms(x):
                return x != set()

        symptoms_with_prefix = profile[profile.apply(filter_symptoms)].index

        # remove the 'sy_' prefix
        symptoms = [s[3:] for s in symptoms_with_prefix]

        return symptoms

    def causes_of(self, person_id, symptom_string):
        """
        This is a helper function that will give a list of the disease modules causing a particular symptom for
        a particular person.
        :param person_id:
        :param disease_module:
        :return: list of strings for the disease module name
        """
        df = self.sim.population.props

        assert not isinstance(person_id, list), "person_id should be for one person only"
        assert df.at[person_id, 'is_alive'], "The person is not alive"
        assert symptom_string in self.symptom_names

        return list(df.at[person_id, 'sy_' + symptom_string])

    def clear_symptoms(self, person_id, disease_module):
        """
        This is a helper function that remove all the symptoms in a specified person that is caused by a specified
        disease module

        :param person_id:
        :param disease_module_name:
        """
        df = self.sim.population.props

        assert not isinstance(person_id, list), "person_id should be for one person only"
        assert df.at[person_id, 'is_alive'], "The person is not alive"
        assert disease_module in self.sim.modules['HealthSystem'].registered_disease_modules.values(), \
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

class SymptomManager_AutoResolveEvent(Event, PopulationScopeEventMixin):
    """
    This utility function will remove symptoms. It is scheduled by the SymptomManager to let symptoms 'auto-resolve'
    """

    def __init__(self, module, person_id, symptom_string, disease_module):
        super().__init__(module)
        assert isinstance(module, SymptomManager)

        self.person_id = person_id
        self.symptom_string = symptom_string
        self.disease_module = disease_module

    def apply(self, population):
        # extract any persons who have died or who have resolved the symptoms
        df = population.props
        people_to_resolve = df.loc[df.is_alive & df.index.isin(self.person_id), 'sy_' + self.symptom_string]
        people_to_resolve = people_to_resolve[people_to_resolve.apply(lambda x: self.disease_module.name in x)].index
        # run the chg_symptom function
        self.module.change_symptom(person_id=list(people_to_resolve),
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

class SymptomManager_SpuriousSymptomGenerator(RegularEvent, PopulationScopeEventMixin):
    """ This event gives the occurrence of generic symptoms that are not caused by a disease module in the TLO model"""

    def __init__(self, module):
        """This event occures every month"""
        super().__init__(module, frequency=DateOffset(months=1))
        assert isinstance(module, SymptomManager)

    def apply(self, population):

        pass
