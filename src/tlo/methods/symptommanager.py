"""
A skeleton template for disease methods.

"""

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods.healthsystem import HSI_Event
from tlo.population import logger

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
        symptom_var_names = [col for col in self.sim.population.props if col.startswith('sy_') ]
        self.symptom_var_names = symptom_var_names

        # Set all to zero
        self.sim.population.props[symptom_var_names]= 0

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
        self.sim.population.props[child_id,self.symptom_var_names]=0

    def chg_symptom(self, person_id, symptom_string, add_or_remove ,disease_module):
        """
        This is how disease module report that a person has developed a new symptom or an existing symptom has resolved.

        :param person_id: The person_id (int or list of int) for whom the symptom changes
        :param symptom_string: The string for the symptom (must be one of the list_of_symptoms)
        :param add_or_remove: '+' to add the symptom or '-' to remove the symptom
        :param disease_module: pointer to the disease module that is reporting this change in symptom
        """

        # Check that the person_id is for an existing and alive person
        alive_person_ids= list(self.sim.population.props.loc[self.sim.population.props.is_alive==True].index)
        if type(person_id) is not list:
            person_id= [person_id]
        for p in person_id:
            assert p in alive_person_ids

        # Check that the symptom_string is legitimate
        assert symptom_string in self.list_of_symptoms
        symptom_var_name = 'sy_' + symptom_string
        assert symptom_var_name in self.sim.population.props.columns

        # Check that the add/remove signal is legitimate
        assert add_or_remove in ['+','-']

        # Check that the provided disease_module is a registered disease_module
        assert disease_module in self.sim.modules['HealthSystem'].registered_disease_modules.values()

        # # Empty the list of persons with newly onset acute generic symptoms if this is a new day
        # # [This list is used by HealthCareSeekingBehaviour to pick up the person_ids of those who have onset
        # #  acute generic symptoms during the one day before. So empty the list each new day.]
        # # [Not neccessary as this is being cleared out each day by HealthSeekingBehaviourPoll]
        # if self.sim.date >  self.date_of_last_reported_onset_symptom:
        #     self.persons_with_newly_onset_acute_generic_symptoms = list()

        self.date_of_last_reported_onset_symptom = self.sim.date

        # Make the operation:
        if add_or_remove == '+':
            # Add the symptom
            self.sim.population.props.loc[person_id,symptom_var_name]+= 1
            self.persons_with_newly_onset_acute_generic_symptoms = \
                self.persons_with_newly_onset_acute_generic_symptoms + person_id

        else:
            # Remove the symptom
            assert (self.sim.population.props.loc[person_id, symptom_var_name]>0).all(),\
                'Warning: Request to remove symptoms from individuals that do not have the symptom'

            self.sim.population.props.loc[person_id, symptom_var_name] = \
                (self.sim.population.props.loc[person_id, symptom_var_name] - 1).clip(lower=0)


        # TODO: Check that this works for person_id lists of any length

        # Check that all the symptom variables are in good condition (no negative values)
        assert (self.sim.population.props[self.symptom_var_names]>=0).all().all()


