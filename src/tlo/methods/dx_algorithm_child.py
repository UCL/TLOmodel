"""
An example of a diagnostic algorithm that is called during an HSI Event.
"""

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods.healthsystem import HSI_Event

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------

class DxAlgorithmChild(Module):
    """
    This is an example/placeholder to show how a diagnostic algorithm can be used.
    The module contains parameters and a function 'diagnose(...)' which is called by a HSI (usually a Generic HSI)
    and returns a 'diagnosis'.
    """

    # Define parameters
    PARAMETERS = {}

    # No Properties to define
    PROPERTIES = {}

    def __init__(self, name=None, resourcefilepath=None):
        # NB. Parameters passed to the module can be inserted in the __init__ definition.

        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.
        To access files use: Path(self.resourcefilepath) / file_name
        """
        pass

    def initialise_population(self, population):
        pass

    def initialise_simulation(self, sim):
        pass

    def on_birth(self, mother_id, child_id):
        pass

    def diagnose(self,person_id,hsi_event):
        """
        This will diagnose the condition of the person. It is being called from inside an HSI Event.

        :param person_id: The person is to be diagnosed
        :param hsi_event: The calling hsi_event.
        :return: a string representing the diagnosis
        """

        # get the symptoms of the person:
        symptoms = self.sim.populations.props.loc[person_id,self.sim.populations.props.columns.str.startswith('sy_')]

        # make a request for consumables:
        # TODO: insert this demonstration

        # *** Diagnostic algorithm example ***
        if symptoms.sum() > 2:
            diagnosis_str = 'measles'
        else:
            diagnosis_str = 'just_a_common_cold'


        # return the diagnosis as a string
        return diagnosis_str



