"""
This is the place for all the stuff to do with diagnosing a child that presents for care.
It is expected that the pieces of logic and data that go here will be shared across multiple modules so they
are put here rather than the individual disease modules.

There should be a method here to respond to every symptom that a child could present with. Currently this module is
served by the following disease modules:
* Diarrhoea

# todo - this is being deprecated:

"""

from tlo import Module, logging, Parameter, Types
from tlo.methods import Metadata
from tlo.methods.diarrhoea import (
    HSI_Diarrhoea_Treatment_PlanA,
    HSI_Diarrhoea_Treatment_PlanB,
    HSI_Diarrhoea_Treatment_PlanC,
)
from tlo.methods.dxmanager import DxTest

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DxAlgorithmChild(Module):
    """
    The module contains parameters and functions to 'diagnose(...)' children.
    These functions are called by an HSI (usually a Generic HSI)
    """

    ADDITIONAL_DEPENDENCIES = {'HealthSystem', 'SymptomManager'}

    # Declare Metadata
    METADATA = {Metadata.USES_HEALTHSYSTEM}

    PARAMETERS = {
        'prob_uncomplicated_diarrhoea_diagnosed_by_health_worker':
            Parameter(Types.REAL,
                      'probability of uncomplicated diarrhoea being diagnosed by health care worker'
                      ),
        'prob_recommended_treatment_given_by_hw':
            Parameter(Types.REAL,
                      'probability of recommended treatment given by health care worker'
                      ),
        'prob_at_least_ors_given_by_hw':
            Parameter(Types.REAL,
                      'probability of ORS given by health care worker, with or without zinc'
                      ),
        'prob_antibiotic_given_for_dysentery_by_hw':
            Parameter(Types.REAL,
                      'probability of antibiotics given by health care worker, for dysentery'
                      ),
        'prob_multivitamins_given_for_persistent_diarrhoea_by_hw':
            Parameter(Types.REAL,
                      'probability of multivitamins given by health care worker, for persistent diarrhoea'
                      ),
        'prob_hospitalization_referral_for_severe_diarrhoea':
            Parameter(Types.REAL,
                      'probability of hospitalisation of severe diarrhoea'
                      ),
        'sensitivity_danger_signs_visual_inspection':
            Parameter(Types.REAL,
                      'sensitivity of health care workers visual inspection of danger signs'
                      ),
        'specificity_danger_signs_visual_inspection':
            Parameter(Types.REAL,
                      'specificity of health care workers visual inspection of danger signs'
                      ),
    }
    PROPERTIES = {}

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    def read_parameters(self, data_folder):
        pass

    def initialise_population(self, population):
        pass

    def initialise_simulation(self, sim):
        """
        Define the Diagnostics Tests that will be used
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
                                     data=f'dx_algorithm_child diagnosing severe malaria for person'
                                          f'{person_id} on date {self.sim.date}')

                    # clinical malaria
                    elif df.at[person_id, "ma_inf_type"] == "clinical":

                        diagnosis_str = "clinical_malaria"

                        logger.debug(key='message',
                                     data=f'dx_algorithm_child diagnosing clinical malaria for person'
                                          f'{person_id} on date {self.sim.date}')

                    # asymptomatic malaria
                    elif df.at[person_id, "ma_inf_type"] == "asym":

                        diagnosis_str = "clinical_malaria"

                        logger.debug(key='message',
                                     data=f'dx_algorithm_child diagnosing clinical malaria for person {person_id}'
                                          f'on date {self.sim.date}')

                else:
                    diagnosis_str = "negative_malaria_test"

            else:
                diagnosis_str = "non-malarial fever"

        logger.debug(key='message',
                     data=f'{person_id} diagnosis is {diagnosis_str}')

        # return the diagnosis as a string
        return diagnosis_str
