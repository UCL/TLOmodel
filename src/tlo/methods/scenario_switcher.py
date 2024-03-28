import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, Iterable

import pandas as pd

from tlo import Module, Parameter, Types, Date, Simulation
from tlo.events import RegularEvent, PopulationScopeEventMixin


def merge_dicts(dicts: Iterable[Dict]):
    """Returns a merge of the dicts given in the iterable."""
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result

# todo - let it happen at a particular time and change from thing to the other,
# todo - make sure it's backward compatible
# todo - explain in docstring that it's only needed because we have to change a whole TONNE of parameters, some of which are pd.DataFrames and pd.Series, and that these span multiple modules.
#   ... and it is isn't trying to "eat the Scenario class's lunch".


class ImprovedHealthSystemAndCareSeekingScenarioSwitcher(Module):
    """The ScenarioSwitcher module.
    A "Switch" is basically a way of changing a bunch of parameters, including complex parameters (like DataFrames) at
     once through a simple type (e.g. bool) that can be done via the Scenario class.
    This module has parameters that specify the position of the switches in two phases of the simulation - before and
     after a change in the state of the simulation (e.g., before and after some intervention). For each phase, it
     works out the values of the module parameters that are consistent with the configuration of switches in that phase.
    It relies on helper function that provide the values of the parameters that are consistent with the value of a
    particular switch (or switches).
    e.g. This module might have a parameter, 'switch_mega_change' = [False, True] to represent that the change happens
     part-way through the simulation. There would be a helper function defined with signature
     `get_params_for_mega_change(switch_mega_change: bool) -> Dict` which would returns the parameters in the modules
     needed for that value of `switch_mega_change`, of the form `{Disease_Module: {Parameter_Name: Parameter_Value}}`.
    """

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.switches = []  # (Will be list of the switches.).
        self.switches_first_phase = {}
        self.switches_second_phase = {}

    INIT_DEPENDENCIES = set()

    OPTIONAL_INIT_DEPENDENCIES = set()

    METADATA = set()

    PARAMETERS = {
        # Each of these parameter is a list of two booleans -- [bool, bool] -- which represent (i) the state during the
        # period before the date of a switch in state, (ii) the state at the date of switch and the period thereafter.
        # These parameters are distinguished from other parameters by the prefix "switch_": parameter names that do not
        # have that prefix will not work as expected.

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

        # Parameters are hard-coded for this module to not make any changes.
        # The expectation is that some of these are over-written by the Scenario class.
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
        # todo check whether this is actually needed, or if could happen at initialise simulation.

        switch_states_in_first_phase = {
            sw: self.parameters[sw][0]  # <-- pick up first entry in list
            for sw in self.switches
        }

        params_to_update = self._get_parameters_to_change(switch_states_in_first_phase)
        self._do_change_in_parameters(self.sim, params_to_update)

    def initialise_population(self, population):
        pass

    def initialise_simulation(self, sim):
        """Schedule an event at which the parameters are changed."""

        date_of_switch_event = Date(self.parameters["year_of_switch"], 1, 1)  # 1st January of the year specified.

        switch_states_in_second_phase = {
            sw: self.parameters[sw][1]  # <-- pick up second entry in list for the second phase of the simulation
            for sw in self.switches
        }
        parameters_to_update = self._get_parameters_to_change(switch_states_in_second_phase)

        sim.schedule_event(
            ScenarioSwitchEvent(
                module=self,
                parameters_to_update=parameters_to_update),
            date_of_switch_event,
        )

    def on_birth(self, mother_id, child_id):
        pass

    def _get_parameters_to_change(self, switches: Dict) -> Dict:
        """Return the parameters in the modules to be updated for this configuration of switches.
        The returned Dict is of the form, e.g.
        {
            'Depression': {
                'pr_assessed_for_depression_for_perinatal_female': 1.0,
                'pr_assessed_for_depression_in_generic_appt_level1': 1.0
                },
            'Hiv': {
                'prob_start_art_or_vs': <<the dataframe named in the corresponding cell in the ResourceFile>>
                }
         }
         """
        return merge_dicts([
            # -- Health System Strengthening Switches
            get_parameters_for_improved_healthsystem_and_healthcare_seeking(
                resourcefilepath=self.resourcefilepath,
                max_healthsystem_function=switches["switch_max_healthsystem_function"],
                max_healthcare_seeking=switches["switch_max_healthcare_seeking"],
            ),
            # todo put in the switches for other HSS as other functions, or as part of the func. above?
        ])

    @staticmethod
    def _do_change_in_parameters(sim: Simulation, params_to_update: Dict):
        """Make the changes to the parameters values held currently in the modules of the simulation."""
        for module, params in params_to_update.items():
            for name, updated_value in params.items():
                try:
                    sim.modules[module].parameters[name] = updated_value
                except KeyError:
                    warnings.warn(
                        f"A parameter could not be updated by the `ScenarioSwitcher` module: "
                        f"module={module}, name={name}.",
                    )


class ScenarioSwitchEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module, parameters_to_update: Dict):
        super().__init__(module)
        self.parameters_to_update = parameters_to_update

    def apply(self, population):
        """Change the parameters in the registered simulation models to those specified."""
        self.module._do_change_in_params(self.sim, self.parameters_to_update)



# ========================================================
# HELPER FUNCTION THAT PROVIDE THE PARAMETERS TO BE UPDATED
# ========================================================

# --- HEALTH SYSTEM STRENGTHENING
#todo - this should be inside the class defined above
def get_parameters_for_improved_healthsystem_and_healthcare_seeking(
    resourcefilepath: Path,
    max_healthsystem_function: Optional[bool] = False,
    max_healthcare_seeking: Optional[bool] = False,
) -> Dict:
    """
    This returns the parameters to be updated to represent certain Health System Strengthening Interventions.

    It reads an Excel workbook to find the parameters that should be updated. Linked sheets in the Excel workbook
    specify updated pd.Series and pd.DataFrames.

    Returns a dictionary of parameters and their updated values to indicate
    an ideal healthcare system in terms of maximum health system function, and/or
    maximum healthcare seeking.

    The return dict is in the form:
    e.g. {
            'Depression': {
                'pr_assessed_for_depression_for_perinatal_female': 1.0,
                'pr_assessed_for_depression_in_generic_appt_level1': 1.0
                },
            'Hiv': {
                'prob_start_art_or_vs': <<the dataframe named in the corresponding cell in the ResourceFile>>
                }
         }
    """

    def read_value(_value):
        """Returns the value, or a dataframe if the value point to a different sheet in the workbook, or a series if the
        value points to sheet in the workbook with only two columns (which become the index and the values)."""
        drop_extra_columns = lambda df: df.dropna(how='all', axis=1)  # noqa E731
        squeeze_single_col_df_to_series = lambda df: \
            df.set_index(df[df.columns[0]])[df.columns[1]] if len(df.columns) == 2 else df  # noqa E731

        def construct_multiindex_if_implied(df):
            """Detect if a multi-index is implied (by the first column header having a "/" in it) and construct this."""
            if isinstance(df, pd.DataFrame) and (len(df.columns) > 1) and ('/' in df.columns[0]):
                idx = df[df.columns[0]].str.split('/', expand=True)
                idx.columns = tuple(df.columns[0].split('/'))

                # Make the dtype as `int` if possible
                for col in idx.columns:
                    try:
                        idx[col] = idx[col].astype(int)
                    except ValueError:
                        pass

                df.index = pd.MultiIndex.from_frame(idx)
                return df.drop(columns=df.columns[0])
            else:
                return df

        if isinstance(_value, str) and _value.startswith("#"):
            sheet_name = _value.lstrip("#").split('!')[0]
            return \
                squeeze_single_col_df_to_series(
                    drop_extra_columns(
                        construct_multiindex_if_implied(
                            pd.read_excel(workbook, sheet_name=sheet_name))))

        elif isinstance(_value, str) and _value.startswith("["):
            # this looks like its intended to be a list
            return eval(_value)
        else:
            return _value

    workbook = pd.ExcelFile(
        resourcefilepath / 'ResourceFile_Improved_Healthsystem_And_Healthcare_Seeking.xlsx')
    # todo - the read-in off this file _might_ be better placed in the calling module, TBD

    # Load the ResourceFile for the list of parameters that may change
    mainsheet = pd.read_excel(workbook, 'main').set_index(['Module', 'Parameter'])

    # Select which columns for parameter changes to extract
    cols = []
    if max_healthsystem_function:
        cols.append('max_healthsystem_function')

    if max_healthcare_seeking:
        cols.append('max_healthcare_seeking')

    # Collect parameters that will be changed (collecting the first encountered non-NAN value)
    params_to_change = mainsheet[cols].dropna(axis=0, how='all')\
                                      .apply(lambda row: [v for v in row if not pd.isnull(v)][0], axis=1)

    # Convert to dictionary
    params = defaultdict(lambda: defaultdict(dict))
    for idx, value in params_to_change.items():
        params[idx[0]][idx[1]] = read_value(value)

    return params



# # POSSIBLE WAY OF ELABORATING THIS FOR OTHER THINGS....
# # --- VERTICAL PROGRAMS
# # N.B. Note that these can be separate functions, or one big functions.
# # N.B. These functions could be on the ScenarioSwitcher class itself.
# def get_parameters_for_vertical_program_scale_up_hiv_and_tb(
#     resourcefilepath: Path,
#     switch_scaleup_hiv: bool,
#     switch_scaleup_tb: bool,
# ) -> Dict:
#     """
#     Returns a dictionary of parameters and their updated values to indicate
#     the possible scale-up for HIV and TB programs.
#
#     The return dict is in the form:
#     e.g. {
#             'Depression': {
#                 'pr_assessed_for_depression_for_perinatal_female': 1.0,
#                 'pr_assessed_for_depression_in_generic_appt_level1': 1.0
#                 },
#             'Hiv': {
#                 'prob_start_art_or_vs': <<the dataframe named in the corresponding cell in the ResourceFile>>
#                 }
#          }
#     """
#     return {}
#
