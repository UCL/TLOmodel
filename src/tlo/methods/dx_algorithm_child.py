"""
This is the place for all the stuff to do with diagnosing a child that presents for care.
It is expected that the pieces of logic and data that go here will be shared across multiple modules so they
are put here rather than the individual disease modules.
"""

import pandas as pd
from tlo import Module
from tlo.methods.diarrhoea import HSI_Diarrhoea_Severe_Dehydration, HSI_Diarrhoea_Non_Severe_Dehydration, \
    HSI_Diarrhoea_Severe_Persistent_Diarrhoea, HSI_Diarrhoea_Non_Severe_Persistent_Diarrhoea, HSI_Diarrhoea_Dysentery
from tlo.methods.dxmanager import DxTest


class DxAlgorithmChild(Module):
    """
    The module contains parameters and functions to 'diagnose(...)' children.
    These functions are called by an HSI (usually a Generic HSI)
    """

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

        # The details about these tests need to be added to reveal the extent of severe dehydration

        # Test to do..
        # no_blood_in_stool = DxTest(
        #     property='AllTrue',
        #     sensitivity=1.00,
        #     specificity=1.00
        # )
        #
        # dx_manager = self.sim.modules['HealthSystem'].dx_manager
        #
        # dx_manager.register_dx_test(
        #     something=something
        # )

    def on_birth(self, mother_id, child_id):
        pass

    def do_when_diarrhoea(self, person_id, hsi_event):
        """
        This routine is callled when Diarrhoea is reported.

        It diagnsoes

        See this report https://apps.who.int/iris/bitstream/handle/10665/104772/9789241506823_Chartbook_eng.pdf
        (page 3).
        NB:
            * Provisions for cholera are not included
            * The danger signs are classified collectively and are based on the result of a DxTest representing the
                ability of the clinician to correctly determine the real value of the property 'gi_severe_dehydration'
        """
        schedule_hsi = self.sim.modules['HealthSystem'].schedule_hsi_event

        # Create the list to hold the strings for each diagnosis
        diagnosis = []

        # Gather information that can be reported:
        # 1) Get duration of diarrhoea
        duration = 10
        # TODO: fill in these value properly

        # 2) Get type of diarrhoea
        blood_in_stool = False

        # 3) Get status of dehydration
        dehydration = True

        # Gather information that cannot be reported:
        # 1) Assessment of danger signs
        danger_signs = True
        # TODO: add hidden variable of seriousness of dehydration and a DxTest for it.

        # Apply the algorithm:
        # --------   Classify Extent of Dehydration   ---------
        if dehydration and danger_signs:
            diagnosis.append(
                'Severe_Dehydration'
            )
            schedule_hsi(hsi_event=HSI_Diarrhoea_Severe_Dehydration(person_id=person_id, module=self),
                         priority=0,
                         topen=self.sim.date,
                         tclose=None
                         )

        elif dehydration and not danger_signs:
            diagnosis.append(
                'Non_Severe_Dehydration'
            )
            schedule_hsi(hsi_event=HSI_Diarrhoea_Non_Severe_Dehydration(person_id=person_id, module=self),
                         priority=0,
                         topen=self.sim.date,
                         tclose=None
                         )
        # ----------------------------------------------------

        # --------   Classify Type of Diarrhoea   -----------
        if (duration >= 14) and dehydration:
            diagnosis.append(
                'Severe_Persistent_Diarrhoea'
            )
            schedule_hsi(hsi_event=HSI_Diarrhoea_Severe_Persistent_Diarrhoea(person_id=person_id, module=self),
                         priority=0,
                         topen=self.sim.date,
                         tclose=None
                         )
        elif (duration >= 14) and (not dehydration):
            diagnosis.append(
                'Non_Severe_Persistent_Diarrhoea'
            )
            schedule_hsi(hsi_event=HSI_Diarrhoea_Non_Severe_Persistent_Diarrhoea(person_id=person_id, module=self),
                         priority=0,
                         topen=self.sim.date,
                         tclose=None
                         )
        # -----------------------------------------------------

        # --------   Classify Whether Dysentery or Not --------
        if blood_in_stool:
            diagnosis.append(
                'Dysentery'
            )
            schedule_hsi(hsi_event=HSI_Diarrhoea_Dysentery(person_id=person_id, module=self),
                         priority=0,
                         topen=self.sim.date,
                         tclose=None
                         )
        # -----------------------------------------------------

        # Return the diagnosis list
        return diagnosis
