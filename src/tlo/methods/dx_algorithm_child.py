"""
This is the place for all the stuff to do with diagnosing a child that presents for care.
It is expected that the pieces of logic and data that go here will be shared across multiple modules so they
are put here rather than the individual disease modules.
"""
import pandas as pd

from tlo import Module

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------


class DxAlgorithmChild(Module):
    """
    The module contains parameters and functions to 'diagnose(...)' children.
    These functions are called by an HSI (usually a Generic HSI)
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
        """
        Define the Diagnostics Tests that will be used
        dx_manager = self.sim.modules['HealthSystem'].dx_manager

        """
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
