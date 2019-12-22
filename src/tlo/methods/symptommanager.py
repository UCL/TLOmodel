"""
A skeleton template for disease methods.

"""

from tlo import DateOffset, Module, Property, Types
from tlo.events import Event, PopulationScopeEventMixin

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------


class SymptomManager(Module):
    """
    This module is used to track the symptoms of persons. The addition and removal of symptoms is handled here.
    """

    list_of_symptoms = ['fever',
                        'vomiting',
                        'stomachache',
                        'sore_throat',
                        'respiratory_symptoms',
                        'headache',
                        'skin_complaint',
                        'dental_complaint',
                        'backache',
                        'injury',
                        'eye_complaint']

    # These are properties of individual for the prescence/abscence of symptoms.
    # A value > 0 implies the symptom is present.

    PROPERTIES = dict()
    for symp in list_of_symptoms:
        PROPERTIES['sy_' + symp] = Property(Types.INT, 'Presence of symptom ' + symp)

    def __init__(self, name=None, resourcefilepath=None):
        # NB. Parameters passed to the module can be inserted in the __init__ definition.

        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.persons_with_newly_onset_acute_generic_symptoms = list()

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.
        To access files use: Path(self.resourcefilepath) / file_name
        """
        pass

    def initialise_population(self, population):
        """
        Initialise all persons to have no symptoms at simulation initiation
        """

        # Get the variable names that are defined
        symptom_var_names = [col for col in self.sim.population.props if col.startswith('sy_')]
        self.symptom_var_names = symptom_var_names

        # Set all to zero
        self.sim.population.props[symptom_var_names] = 0

        # Get and save the list_of_symptoms
        self.list_of_symptoms = [a.split('sy_')[1] for a in symptom_var_names]

    def initialise_simulation(self, sim):
        """
        Before simulation starts, initialise the date_of_last_reported_onset_symptom
        """
        self.date_of_last_reported_onset_symptom = self.sim.date

    def on_birth(self, mother_id, child_id):
        """
        Set that the child will have no symptoms by default
        """
        self.sim.population.props[child_id, self.symptom_var_names] = 0

    def chg_symptom(self, person_id, symptom_string, add_or_remove, disease_module, duration_in_days=None):
        """
        This is how disease module report that a person has developed a new symptom or an existing symptom has resolved.

        :param person_id: The person_id (int or list of int) for whom the symptom changes
        :param symptom_string: The string for the symptom (must be one of the list_of_symptoms)
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
        assert symptom_string in self.list_of_symptoms
        symptom_var_name = 'sy_' + symptom_string
        assert symptom_var_name in self.sim.population.props.columns

        # Check that the add/remove signal is legitimate
        assert add_or_remove in ['+', '-']

        # Check that the duation in days makes sense
        if duration_in_days is not None:
            assert int(duration_in_days) > 0

        # Check that the provided disease_module is a registered disease_module
        assert disease_module in self.sim.modules['HealthSystem'].registered_disease_modules.values()

        # # Empty the list of persons with newly onset acute generic symptoms if this is a new day
        # # [This list is used by HealthCareSeekingBehaviour to pick up the person_ids of those who have onset
        # #  acute generic symptoms during the one day before. So empty the list each new day.]

        # Make the operation:
        if add_or_remove == '+':
            # Add the symptom
            self.sim.population.props.loc[person_id, symptom_var_name] += 1
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
            # Remove the symptom
            # NB. There are no checks that the disease module is only relieving symptoms that it imposed.
            # So a disease module could erronesouly remove a symptom more than once and thus alleviate it even though
            # another disease module would not want it removed.

            assert (self.sim.population.props.loc[person_id, symptom_var_name] > 0).all(), \
                'Warning: Request to remove symptoms from individuals that do not have the symptom'

            self.sim.population.props.loc[person_id, symptom_var_name] = \
                (self.sim.population.props.loc[person_id, symptom_var_name] - 1).clip(lower=0)

        # Check that all the symptom variables are in good condition (no negative values)
        assert (self.sim.population.props[self.symptom_var_names] >= 0).all().all()


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



