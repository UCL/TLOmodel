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

        # Test for the visual inspection of 'Danger signs' for a child who is dehydrated
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            danger_signs_visual_inspection=DxTest(
                property='gi_current_severe_dehydration',
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
                ability of the clinician to correctly determine the true value of the property 'gi_severe_dehydration'
        """
        schedule_hsi = self.sim.modules['HealthSystem'].schedule_hsi_event
        run_dx_test = lambda test: self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
            dx_tests_to_run=test,
            hsi_event=self
        )
        df = self.sim.population.props

        # Gather information that can be reported:
        # 1) Get duration of diarrhoea to date
        duration_in_days = (self.sim.date - df.at[person_id, 'gi_last_diarrhoea_date_of_onset']).days

        # 2) Get type of diarrhoea
        blood_in_stool = df.at[person_id, 'gi_last_diarrhoea_type'] == 'bloody'

        # 3) Get status of dehydration
        dehydration = 'dehydration' in self.sim.modules['SymptomManager'].has_what(person_id)

        # Gather information that cannot be reported:
        # 1) Assessment of danger signs
        danger_signs = run_dx_test('danger_signs_visual_inspection')

        # TODO: add hidden variable of seriousness of dehydration and a DxTest for it.

        # Apply the algorithms:
        # --------   Classify Extent of Dehydration   ---------
        if dehydration and danger_signs:
            # 'Severe_Dehydration'
            schedule_hsi(hsi_event=HSI_Diarrhoea_Severe_Dehydration(person_id=person_id, module=self),
                         priority=0,
                         topen=self.sim.date,
                         tclose=None
                         )
        elif dehydration and not danger_signs:
            # 'Non_Severe_Dehydration'
            schedule_hsi(hsi_event=HSI_Diarrhoea_Non_Severe_Dehydration(person_id=person_id, module=self),
                         priority=0,
                         topen=self.sim.date,
                         tclose=None
                         )
        # ----------------------------------------------------

        # --------   Classify Type of Diarrhoea   -----------
        if (duration_in_days >= 14) and dehydration:
            # 'Severe_Persistent_Diarrhoea'
            schedule_hsi(hsi_event=HSI_Diarrhoea_Severe_Persistent_Diarrhoea(person_id=person_id, module=self),
                         priority=0,
                         topen=self.sim.date,
                         tclose=None
                         )
        elif (duration_in_days >= 14) and not dehydration:
            # 'Non_Severe_Persistent_Diarrhoea'
            schedule_hsi(hsi_event=HSI_Diarrhoea_Non_Severe_Persistent_Diarrhoea(person_id=person_id, module=self),
                         priority=0,
                         topen=self.sim.date,
                         tclose=None
                         )
        # -----------------------------------------------------

        # --------  Classify Whether Dysentery or Not  --------
        if blood_in_stool:
            # 'Dysentery'
            schedule_hsi(hsi_event=HSI_Diarrhoea_Dysentery(person_id=person_id, module=self),
                         priority=0,
                         topen=self.sim.date,
                         tclose=None
                         )
        # -----------------------------------------------------

        # Return the diagnosis list
        return diagnosis
