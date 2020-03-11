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

    def diagnose(self, person_id, hsi_event):
        """
        This will diagnose the condition of the person. It is being called from inside an HSI Event.

        :param person_id: The person is to be diagnosed
        :param hsi_event: The calling hsi_event.
        :return: a string representing the diagnosis
        """

        # get the symptoms of the person:
        symptoms = self.sim.population.props.loc[person_id, self.sim.population.props.columns.str.startswith('sy_')]
        num_of_symptoms = sum(symptoms.apply(lambda symp: symp != set()))

        # Make a request for consumables (making reference to the hsi_event from which this is called)
        # TODO: Finish this demonstration **

        # Make request for some consumables
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        item_code_test = pd.unique(
            consumables.loc[consumables['Items'] == 'Proteinuria test (dipstick)', 'Item_Code']
        )[0]
        consumables_needed = {
            'Intervention_Package_Code': {},
            'Item_Code': {item_code_test: 1},
        }

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=hsi_event, cons_req_as_footprint=consumables_needed
        )

        if outcome_of_request_for_consumables['Item_Code'][item_code_test]:
            # The neccessary diagnosis was available...

            # Example of a diangostic algorithm
            if num_of_symptoms > 2:
                diagnosis_str = 'measles'
            else:
                diagnosis_str = 'just_a_common_cold'

        else:
            # Without the diagnostic test, there cannot be a determinant diagnsosi
            diagnosis_str = 'indeterminate'

        # return the diagnosis as a string
        return diagnosis_str
