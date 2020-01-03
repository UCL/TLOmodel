"""
A skeleton template for disease methods.

"""
import pandas as pd

from tlo import DateOffset, Module, Property, Types, Parameter
from tlo.events import Event, PopulationScopeEventMixin

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------


class SymptomManager(Module):
    """
    This module is used to track the symptoms of persons. The addition and removal of symptoms is handled here.
    """

    PROPERTIES = dict()     # give blank definition of parameters here. It's updated in 'before_make_initial_population'

    PARAMETERS = {
        'list_of_generic_symptoms': Parameter(Types.LIST, 'List of generic symptoms')
    }

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.persons_with_newly_onset_acute_generic_symptoms = list()

    def read_parameters(self, data_folder):
        # Generic Symptoms: pre-defined and used in health seeking behaviour
        self.parameters['list_of_generic_symptoms'] = [
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
        ]

    def before_make_initial_population(self):
        """
        Collect up the SYMPTOMS that are declared by each module and use this to establish the properties
        for this module.
        """
        list_of_registered_symptoms = list()
        for module in self.sim.modules.values():
            try:
                symptoms = module.SYMPTOMS
                assert type(symptoms) is set
                assert all([(symp not in list_of_registered_symptoms) for symp in symptoms]), \
                    'Symptoms are being declared that are already declared by another module.'
                # Add the symptoms to the list
                self.list_of_registered_symptoms.extend(list(symptoms))

            except AttributeError:
                pass

        self.total_list_of_symptoms = self.parameters['list_of_generic_symptoms'] + list_of_registered_symptoms

        for symp in self.total_list_of_symptoms:
            self.PROPERTIES['sy_' + symp] = Property(Types.LIST, 'Presence of symptom ' + symp)
            # TODO: Property to include a type set (using Types.LISThere for now)

    def initialise_population(self, population):
        """
        Give all individuals the no symptoms (ie. an empty set)
        """
        # Get the variable names that are defined
        self.symptom_var_names = [col for col in self.sim.population.props if col.startswith('sy_')]

        for person_id in list(population.props.index):
            for symptom_var in self.symptom_var_names:
               population.props.at[person_id,symptom_var]=set()

    def initialise_simulation(self, sim):
        pass

    def on_birth(self, mother_id, child_id):
        """
        Set that the child will have no symptoms by default (empty set)
        """
        population = self.sim.population
        for symptom_var in self.symptom_var_names:
           population.props.at[child_id,symptom_var]=set()

    # def register_disease_symptoms(self, module, list_of_symptoms):
    #     """
    #     This function is called by disease module that will track specific symptoms in the symptom manager.
    #     It will make sure there is set of unqiue symptoms.
    #     :param module: the disease module that will log symptoms
    #     :param list_of_symptoms: the list of strings that represent the symptoms that will be tracked
    #     :return: void
    #     """
    #     # Check that this module has not already registered its disease symptoms
    #     assert (module.name not in self.disease_module_names_that_have_registered_symptoms), \
    #         'Module has already registered symptoms.'
    #
    #     # Check that the list of symptoms is really a list
    #     assert (type(list_of_symptoms) is list), 'list_of_generic_symptoms is not a list'
    #
    #     # Check that the symptoms in the list of symptoms are not already in the list
    #     assert all([(symp not in self.list_of_registered_symptoms) for symp in list_of_symptoms])
    #
    #     # Add the symptoms to the list
    #     self.list_of_registered_symptoms.extend(list_of_symptoms)
    #
    #     # Record that this module has registered its symptoms
    #     self.disease_module_names_that_have_registered_symptoms.append(module.name)

    def chg_symptom(self, person_id, symptom_string, add_or_remove, disease_module, duration_in_days=None):
        """
        This is how disease module report that a person has developed a new symptom or an existing symptom has resolved.
        The sy_ property contains a set of of the disease_module names that currenly cause the symptom.
        Check if the set is empty or not to determine if the sympton is currently present.

        :param person_id: The person_id (int or list of int) for whom the symptom changes
        :param symptom_string: The string for the symptom (must be one of the list_of_generic_symptoms)
        :param add_or_remove: '+' to add the symptom or '-' to remove the symptom
        :param disease_module: pointer to the disease module that is reporting this change in symptom
        """

        # Make the person_id into a list
        if type(person_id) is not list:
            person_id = [person_id]

        # Strip out the person_ids for anyone who is not alive.
        alive_person_ids = list(self.sim.population.props.index[self.sim.population.props.is_alive])
        person_id = list(set(person_id).intersection(alive_person_ids))

        # Confirm that all person_ids (after stripping) are alive
        assert all([(p in alive_person_ids) for p in person_id])

        # Check that the symptom_string is legitimate
        assert symptom_string in self.list_of_generic_symptoms
        symptom_var_name = 'sy_' + symptom_string
        assert symptom_var_name in self.sim.population.props.columns

        # Check that the add/remove signal is legitimate
        assert add_or_remove in ['+', '-']

        # Check that the duation in days makes sense
        if duration_in_days is not None:
            assert int(duration_in_days) > 0

        # Check that the provided disease_module is a registered disease_module
        assert disease_module in self.sim.modules['HealthSystem'].registered_disease_modules.values()

        # Make the operation:
        if add_or_remove == '+':
            # Add this disease module as a cause of this symptom
            self.sim.population.props.loc[person_id, symptom_var_name].apply(\
                lambda x: x.add(disease_module.name))
            self.persons_with_newly_onset_acute_generic_symptoms = \
                self.persons_with_newly_onset_acute_generic_symptoms + person_id

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
            assert self.sim.population.props.loc[person_id, symptom_var_name].apply(\
                lambda x: (disease_module.name in x))\
                .all(), \
                'Request from disease module to remove a symptom that it has not caused.'

            self.sim.population.props.loc[person_id, symptom_var_name].apply( \
                lambda x: x.remove(disease_module.name))

    def who_has(self, in_list_of_symptoms):
        """
        This is a helper function to look up who has a particular symptom or set of symptoms.
        It retursn a list of indicies for person that have all of the symptoms specifid
        :param: list_of_symptoms : list or strings for the symptoms of interest
        :return: list of person_ids for those with all of the symptoms in list_of_symptoms
        """
        # Check formatting of list_of_symptoms is right (must be a list of strings)
        if type(in_list_of_symptoms) is str:
            list_of_symptoms = [in_list_of_symptoms]
        else:
            list_of_symptoms = in_list_of_symptoms
        assert len(list_of_symptoms)>0

        # Check that these are legitimate symptoms
        assert all([(symp in self.total_list_of_symptoms) for symp in list_of_symptoms])

        # get the person_id for those who have each symptom
        df = self.sim.population.props
        mask_has_symp = pd.Series(data=True, index = df.loc[df['is_alive']].index)
        for symp in list_of_symptoms:
            symp_var_name = 'sy_' + symp
            mask_has_symp = mask_has_symp & df[symp_var_name].apply(lambda x: x != set())

        person_id_with_all_symp = list(mask_has_symp.loc[mask_has_symp].index)
        return person_id_with_all_symp

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
        self.module.chg_symptom(person_id=self.person_id,
                                symptom_string=self.symptom_string,
                                add_or_remove='-',
                                disease_module=self.disease_module)

