"""
An example of a diagnostic algorithm that is called during an HSI Event.
"""
import pandas as pd

from tlo import Module

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
        pass

    def initialise_population(self, population):
        pass

    def initialise_simulation(self, sim):
        pass

    def on_birth(self, mother_id, child_id):
        pass

    def diagnose(self, person_id, hsi_event):
        """
        This will diagnose the condition of the person. It is being called from inside an HSI Event.

        :param person_id: The person is to be diagnosed
        :param hsi_event: The calling hsi_event.
        :return: a string representing the diagnosis
        """

        # TODO: fill in




        # return the diagnosis as a string
        diagnosis_str = '?????'
        return diagnosis_str
