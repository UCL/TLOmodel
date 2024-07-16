from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import pandas as pd

from tlo import Module
from tlo.methods import Metadata
from tlo.methods.hsi_generic_first_appts import GenericFirstAppointmentsMixin
from tlo.methods.symptommanager import Symptom

if TYPE_CHECKING:
    from tlo.methods.symptommanager import SymptomManager


class _BaseCancer(Module, GenericFirstAppointmentsMixin):
    """
    NB: INIT_DEPENDENCIES, OPTIONAL_INIT_DEPENDENCIES, METADATA are already pre-filled with values common to all cancer modules. Providing these attributes in the derived classes can be done without listing these common items, they will be added automatically on instantiation.
    """

    __all_cancer_dependencies = {"Demography", "HealthSystem", "SymptomManager"}
    __all_cancer_optionals = {"HealthBurden"}
    __all_cancer_metadata = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN,
    }

    # The name of the resource file in the resource file directory that this
    # module should read parameters from.
    _resource_filename: str = ""
    # List of symptoms that this class will register with the SymptomManager
    # during read_parameters().
    _symptoms_to_register: List[Symptom] = []

    # Directory containing resource files.
    resourcefilepath: Path

    @property
    def symptom_manager(self) -> SymptomManager:
        """
        Points to the SymptomManager instance registered with the simulation.
        """
        return self.sim.modules["SymptomManager"]

    def __init__(
        self, name: Optional[str] = None, resource_filepath: Optional[Path] = None
    ) -> None:
        super().__init__(name=name)
        self.resourcefilepath = resource_filepath

        # Impose common cancer dependencies, optionals, etc
        self.INIT_DEPENDENCIES = self.INIT_DEPENDENCIES.union(
            _BaseCancer.__all_cancer_dependencies
        )
        self.OPTIONAL_INIT_DEPENDENCIES = self.OPTIONAL_INIT_DEPENDENCIES.union(
            _BaseCancer.__all_cancer_optionals
        )
        self.METADATA = self.METADATA.union(_BaseCancer.__all_cancer_metadata)

    def read_parameters(self, *args, **kwargs):
        """
        Setup parameters used by the module, and register any symptoms. 
        """

        # Update parameters from the resourcefile
        self.load_parameters_from_dataframe(
            pd.read_excel(Path(self.resourcefilepath) / self._resource_filename,
                          sheet_name="parameter_values")
        )

        # Register Symptom that this module will use
        for str in self._symptoms_to_register:
            self.symptom_manager.register_symptom(str)

    def initialise_population(self, population):
        return super().initialise_population(population)

    def initialise_simulation(self, sim):
        return super().initialise_simulation(sim)

    def on_birth(self, mother_id, child_id):
        return super().on_birth(mother_id, child_id)

    def on_hsi_alert(self, person_id: int, treatment_id):
        pass

    def report_daily_values():
        raise NotImplementedError("Must be directly implemented by subclass")

    def do_at_generic_first_appt(self, *args, **kwargs) -> None:
        raise NotImplementedError("Must be directly implemented by subclass")
