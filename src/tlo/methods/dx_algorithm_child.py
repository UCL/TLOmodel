"""
An example of a diagnostic algorithm that is called during an HSI Event.
"""
import pandas as pd
import logging

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.methods import malaria

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
        df = self.sim.population.props
        params = self.sim.modules['Malaria'].parameters

        diagnosis_str = 'unknown'

        # NOTES:
        # here I tried to call malaria.HSI_Malaria_rdt but couldn't get appropriate return value
        # wanted to return diagnosis = 'clinical_malaria' etc but kept getting 'None' returned
        # we end up with lots of repeated code in this case

        # get the symptoms of the person:
        symptoms = self.sim.population.props.loc[person_id, self.sim.population.props.columns.str.startswith('sy_')]
        num_of_symptoms = sum(symptoms.apply(lambda symp: symp != set()))
        # symptoms = df.loc[person_id, df.columns.str.startswith('sy_')]

        # todo: this!!
        # self.sim.modules['SymptomManager'].has_what(person_id)
        # Out[2]: ['fever', 'vomiting', 'stomachache', 'headache']
        # 'fever' in self.sim.modules['SymptomManager'].has_what(person_id)
        # also use who_has

        if 'fever' in self.sim.modules['SymptomManager'].has_what(person_id):

            # TODO: the rdt request needs a LabPOC appt type, not just generic OPD
            consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
            # this package contains treatment too
            pkg_code1 = pd.unique(
                consumables.loc[
                    consumables['Items'] == 'Malaria test kit (RDT)',
                    'Intervention_Pkg_Code'])[0]

            consumables_needed = {
                'Intervention_Package_Code': [{pkg_code1: 1}],
                'Item_Code': [],
            }

            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=hsi_event, cons_req_as_footprint=consumables_needed, to_log=False
            )

            if outcome_of_request_for_consumables:

                # log the consumable use
                outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                    hsi_event=hsi_event, cons_req_as_footprint=consumables_needed, to_log=True
                )

                # severe malaria
                if df.at[person_id, 'ma_is_infected'] & (df.at[person_id, 'ma_inf_type'] == 'severe'):
                    diagnosis_str = 'severe_malaria'

                logger.debug(
                        "DxAlgorithmChild diagnosing severe malaria for child %d on date %s",
                        person_id, self.sim.date)

                # clinical malaria
                if df.at[person_id, 'ma_is_infected'] & (df.at[person_id, 'ma_inf_type'] == 'clinical'):

                    # diagnosis of clinical disease dependent on RDT sensitivity
                    diagnosed = self.sim.rng.choice([True, False], size=1, p=[params['sensitivity_rdt'],
                                                                              (1 - params['sensitivity_rdt'])])

                    # diagnosis
                    if diagnosed:
                        diagnosis_str = 'clinical_malaria'

                        logger.debug("DxAlgorithmChild diagnosing clinical malaria for child %d on date %s",
                                     person_id, self.sim.date)
                else:
                    diagnosis_str = 'negative_malaria_test'

            else:
                diagnosis_str = 'no_rdt_available'

            logger.debug(f'{person_id} diagnosis is {diagnosis_str}')

        # return the diagnosis as a string
        return diagnosis_str
