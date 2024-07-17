from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import pandas as pd

from tlo import Module, Types
from tlo.methods import Metadata
from tlo.methods.hsi_generic_first_appts import GenericFirstAppointmentsMixin
from tlo.methods.symptommanager import Symptom

if TYPE_CHECKING:
    from tlo import Simulation
    from tlo.lm import LinearModel
    from tlo.methods.symptommanager import SymptomManager
    from tlo.population import Population


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
    # Add items that are needed for all cancer modules
    __all_cancer_common_items = {
        "screening_biopsy_endoscopy_cystoscopy_optional": {
            "Specimen container": 1,
            "Lidocaine HCl (in dextrose 7.5%), ampoule 2 ml": 1,
            "Gauze, absorbent 90cm x 40m_each_CMST": 30,
            "Disposables gloves, powder free, 100 pieces per box": 1,
            "Syringe, needle + swab": 1,
        },
        "screening_biopsy_core": {"Biopsy needle": 1},
        "treatment_surgery_core": {
            "Halothane (fluothane)_250ml_CMST": 100,
            "Scalpel blade size 22 (individually wrapped)_100_CMST": 1,
        },
        "treatment_surgery_optional": {
            "Sodium chloride, injectable solution, 0,9 %, 500 ml": 2000,
            "Paracetamol, tablet, 500 mg": 8000,
            "Pethidine, 50 mg/ml, 2 ml ampoule": 6,
            "Suture pack": 1,
            "Gauze, absorbent 90cm x 40m_each_CMST": 30,
            "Cannula iv  (winged with injection pot) 18_each_CMST": 1,
        },
        "palliation": {
            "morphine sulphate 10 mg/ml, 1 ml, injection (nt)_10_IDA": 1,
            "Diazepam, injection, 5 mg/ml, in 2 ml ampoule": 3,
            "Syringe, needle + swab": 4,
        },
        # N.B. This is not an exhaustive list of drugs required for palliation
        "treatment_chemotherapy": {"Cyclophosphamide, 1 g": 16800},
        "iv_drug_cons": {
            "Cannula iv  (winged with injection pot) 18_each_CMST": 1,
            "Giving set iv administration + needle 15 drops/ml_each_CMST": 1,
            "Disposables gloves, powder free, 100 pieces per box": 1,
            "Gauze, swabs 8-ply 10cm x 10cm_100_CMST": 84,
        },
    }

    # Dictionary of additional consumable items and quantities that need to be
    # fetched for this particular cancer.
    _cancer_specific_items: Dict[str, Dict[str, int]] = {}
    # The name of the resource file in the resource file directory that this
    # module should read parameters from.
    _resource_filename: str = ""
    # List of symptoms that this class will register with the SymptomManager
    # during read_parameters().
    _symptoms_to_register: List[Symptom] = []

    # DALY weights from the HealthBurden module
    daly_wts: Dict[str, float]
    # Items codes for consumables
    item_codes: Dict[str, Dict[int, int]]
    # Linear models for cancer stages
    linear_models: Dict[str, LinearModel]
    # Directory containing resource files.
    resourcefilepath: Path

    @property
    def all_consumable_items(self) -> Dict[str, Dict[str, int]]:
        """
        Dictionary of all consumable item names (keys) and their quantity (values)
        that this cancer module requires. Combines the __all_cancer_items with
        the module-(cancer-) specific items.
        """
        return {**_BaseCancer.__all_cancer_common_items, **self._cancer_specific_items}

    @property
    def symptom_manager(self) -> SymptomManager:
        """
        Points to the SymptomManager instance registered with the simulation.
        """
        return self.sim.modules["SymptomManager"]

    def __init__(
        self, name: Optional[str] = None, resourcefilepath: Optional[Path] = None
    ) -> None:
        super().__init__(name=name)
        self.resourcefilepath = resourcefilepath
        self.linear_models = {}
        self.daly_wts = {}
        self.item_codes = {}

        # Impose common cancer dependencies, optionals, etc
        self.INIT_DEPENDENCIES = self.INIT_DEPENDENCIES.union(
            _BaseCancer.__all_cancer_dependencies
        )
        self.OPTIONAL_INIT_DEPENDENCIES = self.OPTIONAL_INIT_DEPENDENCIES.union(
            _BaseCancer.__all_cancer_optionals
        )
        self.METADATA = self.METADATA.union(_BaseCancer.__all_cancer_metadata)

    def _set_item_codes(self) -> None:
        """
        Set the item codes for this type of cancer, which can only be done when the
        HeathSystem is ready.

        Items to fetch are defined by the _cancer_specific_items (for items that only
        this particular type of cancer needs) and __all_cancer_common_items (items that
        all types of cancer need).
        """
        get_item_code = self.sim.modules["HealthSystem"].get_item_code_from_item_name
        self.item_codes = {
            group_name: {
                get_item_code(name): quantity
                for name, quantity in items_in_group.items()
            }
            for group_name, items_in_group in self.all_consumable_items.items()
        }

    def read_parameters(self, *args, **kwargs) -> None:
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

    def initialise_population(self, population: Population) -> None:
        """
        Common initialise_simulation steps for all cancer modules.
        After common steps complete, the initialise_simulation_hook will be invoked, which
        should be explicitly implemented by subclasses.

        Common steps are:
        - Set default values for properties columns in the population DF.
        """
        df = population.props

        for property_name, property in self.PROPERTIES.items():
            # Note that Types.CATEGORICAL Properties are default set to "none" for Cancer modules.
            df.loc[df.is_alive, property_name] = (
                property._default_value
                if property.type_ is not Types.CATEGORICAL
                else "none"
            )

        # Run any cancer-specific setup steps
        self.initialise_population_hook(population=population)

    def initialise_population_hook(self, population: Population) -> None:
        """
        Cancer-specific steps to run during population initialisation.

        These steps will be run after the steps common to all cancer modules.
        Must be implemented by subclass, even if it is just to pass.
        """
        raise NotImplementedError(
            "initialise_population_hook must be explicitly defined by Cancer subclass."
        )

    def initialise_simulation(self, sim: Simulation) -> None:
        """
        Common initialise_simulation steps for all cancer modules.
        After common steps complete, the initialise_simulation_hook will be invoked, which
        should be explicitly implemented by subclasses.

        Common steps are:
        - Set the consumable item codes.
        """
        # Set the consumable item codes now that the HealthSystem has been initialised,
        # storing them in the item_codes attribute.
        self._set_item_codes()

        # Run any cancer-specific steps at this point.
        self.initialise_simulation_hook(sim)

    def initialise_simulation_hook(self, sim: Simulation) -> None:
        """
        Cancer-specific steps to run during simulation initialisation.
        
        These steps will be run after the steps common to all cancer modules.
        Must be implemented by subclass, even if it is just to pass.
        """
        raise NotImplementedError("initialise_simulation_hook must be explicitly defined by Cancer subclass.")

    def on_birth(self, mother_id: int, child_id: int) -> None:
        """
        Initialise DF columns relevant to this module for the newborn child.

        All cancer modules take this opportunity to set the property columns for the newborn
        child to their default values, like is done in the initialise_population method for the
        whole population.
        """
        df = self.sim.population.props

        for property_name, property in self.PROPERTIES.items():
            # Note that Types.CATEGORICAL Properties are default set to "none" for Cancer modules.
            df.loc[child_id, property_name] = (
                property._default_value
                if property.type_ is not Types.CATEGORICAL
                else "none"
            )

    def on_hsi_alert(self, person_id: int, treatment_id):
        """
        All cancer modules by default will pass on this method.
        """
        pass

    def report_daily_values():
        raise NotImplementedError("Must be directly implemented by subclass")

    def do_at_generic_first_appt(self, *args, **kwargs) -> None:
        raise NotImplementedError("Must be directly implemented by subclass")
