"""
This is the place for all the stuff to do with diagnosing a child that presents for care.
It is expected that the pieces of logic and data that go here will be shared across multiple modules so they
are put here rather than the individual disease modules.

There should be a method here to respond to every symptom that a child could present with. Currently this module is
served by the following disease modules:
* Diarrhoea


"""

from tlo import Module, logging
from tlo.methods import Metadata
from tlo.methods.diarrhoea import (
    HSI_Diarrhoea_Dysentery,
    HSI_Diarrhoea_Non_Severe_Persistent_Diarrhoea,
    HSI_Diarrhoea_Severe_Persistent_Diarrhoea,
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

    PARAMETERS = {}
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

        # Test for the visual inspection of 'Danger signs' for a child who is dehydrated
        # todo - this to be parameterised from the resource file and maybe should be declared by the diarrhoea module...
        #  tbd when the ALRI module and this file are finalised.
        if 'Diarrhoea' in self.sim.modules:
            self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
                danger_signs_visual_inspection=DxTest(
                    property='gi_last_diarrhoea_dehydration',
                    target_categories=['severe'],
                    sensitivity=0.90,
                    specificity=0.80
                )
            )

    def on_birth(self, mother_id, child_id):
        pass

    def do_when_diarrhoea(self, person_id, hsi_event):
        """
        This routine is called when Diarrhoea is reported.

        It diagnoses the condition of the child and schedules HSI Events appropriate to the condition.

        See this report https://apps.who.int/iris/bitstream/handle/10665/104772/9789241506823_Chartbook_eng.pdf
        (page 3).
        NB:
            * Provisions for cholera are not included
            * The danger signs are classified collectively and are based on the result of a DxTest representing the
              ability of the clinician to correctly determine the true value of the
              property 'gi_current_severe_dehydration'
        """
        # Create some short-cuts:
        schedule_hsi = self.sim.modules['HealthSystem'].schedule_hsi_event
        df = self.sim.population.props

        def run_dx_test(test):
            return self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run=test, hsi_event=hsi_event)

        symptoms = self.sim.modules['SymptomManager'].has_what(person_id)

        # Gather information that can be reported:
        # 1) Get duration of diarrhoea to date

        if 'Diarrhoea' in self.sim.modules:
            duration_in_days = (self.sim.date - df.at[person_id, 'gi_last_diarrhoea_date_of_onset']).days

            # 2) Get type of diarrhoea
            blood_in_stool = df.at[person_id, 'gi_last_diarrhoea_type'] == 'bloody'

            # 3) Get status of dehydration
            dehydration = 'dehydration' in symptoms

            # Gather information that cannot be reported:
            # 1) Assessment of danger signs
            danger_signs = run_dx_test('danger_signs_visual_inspection')

            # Apply the algorithms:
            # --------   Classify Extent of Dehydration   ---------
            if not dehydration:
                # Those who do NOT have DEHYDRATION

                # Treatment Plan A for uncomplicated diarrhoea (no dehydration and no danger signs)
                schedule_hsi(
                    HSI_Diarrhoea_Treatment_PlanA(
                        person_id=person_id,
                        module=self.sim.modules['Diarrhoea']),
                    priority=0,
                    topen=self.sim.date,
                    tclose=None)

                if duration_in_days >= 14:
                    # 'Non_Severe_Persistent_Diarrhoea'
                    schedule_hsi(
                        HSI_Diarrhoea_Non_Severe_Persistent_Diarrhoea(
                            person_id=person_id,
                            module=self.sim.modules['Diarrhoea']),
                        priority=0,
                        topen=self.sim.date,
                        tclose=None)

            else:
                # Those who do have DEHYDRATION

                # Given that there is dehydration - schedule an HSI if the duration of diarrhoea has been long
                if duration_in_days >= 14:
                    # 'Severe_Persistent_Diarrhoea'
                    schedule_hsi(
                        HSI_Diarrhoea_Severe_Persistent_Diarrhoea(
                            person_id=person_id,
                            module=self.sim.modules['Diarrhoea']),
                        priority=0,
                        topen=self.sim.date,
                        tclose=None)

                if not danger_signs:
                    # Treatment Plan B for some dehydration diarrhoea but not danger signs
                    # TODO:add "...and not other severe classification from other disease modules
                    #  (measles, pneumonia, etc)"
                    schedule_hsi(
                        HSI_Diarrhoea_Treatment_PlanB(
                            person_id=person_id,
                            module=self.sim.modules['Diarrhoea']),
                        priority=0,
                        topen=self.sim.date,
                        tclose=None)
                else:
                    # Danger sign for 'Severe_Dehydration'
                    schedule_hsi(
                        HSI_Diarrhoea_Treatment_PlanC(
                            person_id=person_id,
                            module=self.sim.modules['Diarrhoea']),
                        priority=0,
                        topen=self.sim.date,
                        tclose=None)

            # --------  Classify Whether Dysentery or Not  --------
            # TODO: note that the 'work' inside this is never used because the cure is completed by one of the other HSI
            #  and so this HSI returns prematurely.
            if blood_in_stool:
                # 'Dysentery'
                schedule_hsi(
                    HSI_Diarrhoea_Dysentery(
                        person_id=person_id,
                        module=self.sim.modules['Diarrhoea']),
                    priority=0,
                    topen=self.sim.date,
                    tclose=None)
            # -----------------------------------------------------

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
