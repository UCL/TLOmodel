import warnings
from typing import Dict

from tlo import Module, Parameter, Types, Date
from tlo.events import RegularEvent, PopulationScopeEventMixin
from tlo.analysis.utils import get_parameters_for_improved_healthsystem_and_healthcare_seeking


class ImprovedHealthSystemAndCareSeekingScenarioSwitcher(Module):
    """This is the `ImprovedHealthSystemAndCareSeekingScenarioSwitcher` module.
    It provides switches that can used by the `Scenario` class to control the overall performance of the HealthSystem
    and healthcare seeking, which are mediated by many parameters across many modules, and which are of the types
    `pd.Series` and `pd.DataFrame`, which cannot be changed via the `Scenario` class (see
    https://github.com/UCL/TLOmodel/issues/988).
    It does this by loading a ResourceFile that contains parameter value to be updated,
    and makes these changes at the point `pre_initialise_population`. As this module is declared as an (Optional)
    dependency of the module that would be loaded first in the simulation (i.e. `Demography`), this module is
    registered first and so this module's `pre_initialise_population` method is called before any other. This provides
    a close approximation to what would happen if the parameters were being changed by the `Scenario` class.
    """

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    INIT_DEPENDENCIES = set()

    OPTIONAL_INIT_DEPENDENCIES = set()

    METADATA = set()

    PARAMETERS = {
        # Each of these parameter is a list of two booleans -- [bool, bool] -- which represent (i) the state during the
        # period before the date of a switch in state, (ii) the state at the date of switch and the period thereafter.

        # -- Health System Strengthening Switches
        "switch_max_healthsystem_function": Parameter(
            Types.LIST, "If True, over-writes parameters that define maximal health system function."
                        "Parameter passed through to `get_parameters_for_improved_healthsystem_and_healthcare_seeking`."
        ),
        "switch_max_healthcare_seeking": Parameter(
            Types.LIST, "If True, over-writes parameters that define maximal healthcare-seeking behaviour. "
                        "Parameter passed through to `get_parameters_for_improved_healthsystem_and_healthcare_seeking`."
        ),

        # This parameter specifies the year in which the state changes. The change occurs on 1st January of that year.
        # If there should not be any switch, then this year can be set to a year that is beyond the end of the
        # simulation.
        "year_of_switch": Parameter(
            Types.INT, "The year in which the state changes. The state changes on 1st January of that year."
        ),
    }

    PROPERTIES = {}

    def read_parameters(self, data_folder):
        """Read-in parameters and process them into the internal storage structures required."""

        # Parameters are hard-coded for this module to not make any changes. (The expectation is that some of these
        # are over-written by the Scenario class.)
        # The first value in the list is used before the year of change, and the second value is used after.
        self.parameters["max_healthsystem_function"] = [False] * 2  # (No use of the "max" scenarios)
        self.parameters["max_healthcare_seeking"] = [False] * 2
        self.parameters["year_of_switch"] = 2100  # (Any change occurs very far in the future)

    def pre_initialise_population(self):
        """Set the parameters for the first period of the simulation.
         Note that this is happening here and not in initialise_simulation because we want to make sure that the
         parameters are changed before other modules call `pre_initialise_population`. We ensure that this module's
         method is the first to be called as this module is declared as an (Optional) dependency of the module that is
         loaded first in the simulation (i.e. `Demography`). This provides a close approximation to what would happen if
         the parameters were being changed by the `Scenario` class."""
        self.update_parameters()

    def update_parameters(self):
        """Update the parameters in the simulation's modules."""

        # Check whether we are currently in the first or second phase of the simulation (i.e., before or after the
        # time of the change, which is at the beginning of the year `year_of_switch`.)
        phase_of_simulation = 0 if self.sim.date.year < self.parameters["year_of_switch"] else 1

        params_to_update = get_parameters_for_improved_healthsystem_and_healthcare_seeking(
            resourcefilepath=self.resourcefilepath,
            max_healthsystem_function=self.parameters['max_healthsystem_function'][phase_of_simulation],
            max_healthcare_seeking=self.parameters['max_healthcare_seeking'][phase_of_simulation],
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

    def initialise_population(self, population):
        pass

    def initialise_simulation(self, sim):
        """Schedule an event at which the parameters are changed."""

        date_of_switch_event = Date(self.parameters["year_of_switch"], 1, 1)  # 1st January of the year specified.
        sim.schedule_event(ScenarioSwitchEvent(module=self), date_of_switch_event)

    def on_birth(self, mother_id, child_id):
        pass


class ScenarioSwitchEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module, parameters_to_update: Dict):
        super().__init__(module)
        self.parameters_to_update = parameters_to_update

    def apply(self, population):
        """Run the function that updates the simulation parameters."""
        self.module.update_parameters()


