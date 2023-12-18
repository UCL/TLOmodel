import warnings

from tlo import Module, Parameter, Types
from tlo.analysis.utils import get_parameters_for_improved_healthsystem_and_healthcare_seeking


class ScenarioSwitcher(Module):
    """The ScenarioSwitcher module.
    This is a utility module that can be used to make changes to parameters in registered simulation models, including
    parameters of the form `pd.Series` and `pd.DataFrame` that cannot be changed via the `Scenario` class (see
    https://github.com/UCL/TLOmodel/issues/988). It loads a ResourceFile that contains parameter value to be updated,
    and makes these changes at the point `pre_initialise_population`. As this module is declared as an (Optional)
    dependency of the module that would be loaded first in the simulation (i.e. `Demography`), this module is
    registered first and so this module's `pre_initialise_population` method is called before any other. This provides
    a close approximation to what would happen if the parameters were being changed by the `Scenario` class."""

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    INIT_DEPENDENCIES = set()

    OPTIONAL_INIT_DEPENDENCIES = set()

    METADATA = set()

    PARAMETERS = {
        "max_healthsystem_function": Parameter(
            Types.BOOL, "If True, over-writes parameters that define maximal health system function."
                        "Parameter passed through to `get_parameters_for_improved_healthsystem_and_healthcare_seeking`."
        ),
        "max_healthcare_seeking": Parameter(
            Types.BOOL, "If True, over-writes parameters that define maximal healthcare-seeking behaviour. "
                        "Parameter passed through to `get_parameters_for_improved_healthsystem_and_healthcare_seeking`."
        ),
    }

    PROPERTIES = {}

    def read_parameters(self, data_folder):
        """Default values for parameters. These are hard-coded."""
        self.parameters["max_healthsystem_function"] = False
        self.parameters["max_healthcare_seeking"] = False

    def initialise_population(self, population):
        pass

    def pre_initialise_population(self):
        """Retrieve parameters to be updated and update them in the other registered disease modules."""

        params_to_update = get_parameters_for_improved_healthsystem_and_healthcare_seeking(
            resourcefilepath=self.resourcefilepath,
            **self.parameters
        )

        for module, params in params_to_update.items():
            for name, updated_value in params.items():
                try:
                    self.sim.modules[module].parameters[name] = updated_value
                except KeyError:
                    warnings.warn(
                        f"A parameter could not be updated by the `ScenarioSwitcher` module: "
                        f"module={module}, name={name}.",
                    )

    def initialise_simulation(self, sim):
        pass

    def on_birth(self, mother_id, child_id):
        pass
