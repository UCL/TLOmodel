"""
An example of a diagnostic algorithm that is called during an HSI Event.
"""
from tlo import Module, logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------


class DxAlgorithmAdult(Module):
    """
    The module contains parameters and a function 'diagnose(...)' which is called by a HSI (usually a Generic HSI)
    and returns a 'diagnosis'.
    """

    ADDITIONAL_DEPENDENCIES = {'HealthSystem', 'SymptomManager'}

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
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        diagnosis_str = "unknown"

        # get the symptoms of the person:
        # num_of_symptoms = sum(symptoms.apply(lambda symp: symp != set()))
        # symptoms = df.loc[person_id, df.columns.str.startswith('sy_')]

        if "fever" in self.sim.modules["SymptomManager"].has_what(person_id):

            if "Malaria" in self.sim.modules:

                # call the DxTest RDT to diagnose malaria
                dx_result = hs.dx_manager.run_dx_test(
                    dx_tests_to_run='malaria_rdt',
                    hsi_event=hsi_event
                )

                if dx_result:

                    # severe malaria
                    if df.at[person_id, "ma_inf_type"] == "severe":
                        diagnosis_str = "severe_malaria"

                        logger.debug(key='message',
                                     data=f'DxAlgorithmAdult diagnosing severe malaria for person {person_id}'
                                          f'on date {self.sim.date}')

                    # clinical malaria
                    elif df.at[person_id, "ma_inf_type"] == "clinical":

                        diagnosis_str = "clinical_malaria"

                        logger.debug(key='message',
                                     data=f'DxAlgorithmAdult diagnosing clinical malaria for person {person_id}'
                                          f'on date {self.sim.date}')

                    # asymptomatic malaria
                    elif df.at[person_id, "ma_inf_type"] == "asym":

                        diagnosis_str = "clinical_malaria"

                        logger.debug(key='message',
                                     data=f'DxAlgorithmAdult diagnosing clinical malaria for person {person_id}'
                                          f'on date {self.sim.date}')

                else:
                    diagnosis_str = "negative_malaria_test"

            else:
                diagnosis_str = "non-malarial fever"

        logger.debug(key='message',
                     data=f'{person_id} diagnosis is {diagnosis_str}')

        # return the diagnosis as a string
        return diagnosis_str
