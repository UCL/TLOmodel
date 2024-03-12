import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, Iterable

import pandas as pd

from tlo import Module, Parameter, Types, Date
from tlo.events import RegularEvent, PopulationScopeEventMixin


def merge_dicts(dicts: Iterable[Dict]):
    """Returns a merge of the dicts given in the iterable."""
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result


class ScenarioSwitcher(Module):
    """The ScenarioSwitcher module. This is a utility module that can be used to make changes to parameters in
    registered simulation models. """

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.parameters_first_phase = {}
        self.parameters_second_phase = {}

    INIT_DEPENDENCIES = set()

    OPTIONAL_INIT_DEPENDENCIES = set()

    METADATA = set()

    PARAMETERS = {
        # Each of these parameter is a list of two booleans -- [bool, bool] -- which represent (i) the state in the time
        # before the date of a switch in state, (ii) the state at the date of switch and thereafter.
        # These parameters are distinguished from other parameters by the prefix "switch_". Parameters that do not have
        # that prefix will not work as expected.

        # -- Health System Strengthening Switches
        "switch_max_healthsystem_function": Parameter(
            Types.LIST, "If True, over-writes parameters that define maximal health system function."
                        "Parameter passed through to `get_parameters_for_improved_healthsystem_and_healthcare_seeking`."
        ),
        "switch_max_healthcare_seeking": Parameter(
            Types.LIST, "If True, over-writes parameters that define maximal healthcare-seeking behaviour. "
                        "Parameter passed through to `get_parameters_for_improved_healthsystem_and_healthcare_seeking`."
        ),
        "switch_all_equipment_available": Parameter(Types.LIST, "-"),               # <-- Dummy
        "switch_inf_beds_available": Parameter(Types.LIST, "_"),                    # <-- Dummy
        "switch_alL_consumables_available": Parameter(Types.LIST, "-"),             # <-- Dummy
        "switch_inf_chw_available": Parameter(Types.LIST, "-"),                     # <-- Dummy
        "switch_inf_hrh_available": Parameter(Types.LIST, "-"),                     # <-- Dummy

        # -- Vertical Programme Scale-up Switches
        "switch_scaleup_hiv": Parameter(Types.LIST, "-"),                           # <-- Dummy
        "switch_scaleup_tb": Parameter(Types.LIST, "-"),                            # <-- Dummy
        "switch_scaleup_malaria": Parameter(Types.LIST, "-"),                       # <-- Dummy
        "switch_scaleup_epi": Parameter(Types.LIST, "-"),                           # <-- Dummy

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

        # Parameters are hard-coded. The expectation is that some of these are over-written by the Scenario class.
        self.parameters["switch_max_healthsystem_function"] = [False, False]
        self.parameters["switch_max_healthcare_seeking"] = [False, False]
        self.parameters["switch_all_equipment_available"] = [False, False]
        self.parameters["switch_inf_beds_available"] = [False, False]
        self.parameters["switch_alL_consumables_available"] = [False, False]
        self.parameters["switch_inf_chw_available"] = [False, False]
        self.parameters["switch_inf_hrh_available"] = [False, False]
        self.parameters["switch_scaleup_hiv"] = [False, False]
        self.parameters["switch_scaleup_tb"] = [False, False]
        self.parameters["switch_scaleup_malaria"] = [False, False]
        self.parameters["self.parameters[switch_scaleup_epi"] = [False, False]
        self.parameters["year_of_switch"] = 2020

        self._process_parameters()

    def _process_parameters(self):
        """Construct the internal storage of the parameters (`self.parameters_first_phase` and
         `self.parameters_second_phase`). Note that this will throw errors for any incorrectly specified parameters.
         """

        # Get list of parameters which encode the change of state (i.e. have prefix of "swtich_").
        switches = [k for k in self.parameters.keys() if k.startswith('switch_')]

        # Check that each identified parameter is a list of two bools.
        for sw in switches:
            p = self.parameters[sw]
            assert isinstance(p, list)
            assert 2 == len(p)
            assert isinstance(p[0], bool) and isinstance(p[1], bool)

        # Assemble the internal storage of these parameters
        self.parameters_first_phase = {
            sw: self.parameters[sw][0]  # <-- pick up first entry in list
            for sw in switches
        }
        self.parameters_second_phase = {
            sw: self.parameters[sw][1]  # <-- pick up second entry in list
            for sw in switches
        }

    def pre_initialise_population(self):
        """Set the parameters for the first period of the simulation.
         Note that this is happening here and not in initialise_simulation because we want to make sure that the
         parameters are changed before other modules call `pre_initialise_population`. We ensure that this module's
         method is the first to be called as this module is declared as an (Optional) dependency of the module that is
         loaded first in the simulation (i.e. `Demography`). This provides a close approximation to what would happen if
         the parameters were being changed by the `Scenario` class."""
        # todo check whether this is actually needed, or if could happen at initialise simulation.

        parameters_to_update = self._get_parameters_to_change(self.parameters_first_phase)
        self._do_change_in_parameters(self.sim, parameters_to_update)

    def initialise_population(self, population):
        pass

    def initialise_simulation(self, sim):
        """Schedule an event at which the parameters are changed."""
        date_of_switch_event = Date(self.parameters["year_of_switch"], 1, 1)  # 1st January of the year specified.
        parameters_to_update = self._get_parameters_to_change(self.parameters_second_phase)
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
        return merge_dicts(
            # -- Health System Strengthening Switches
            get_parameters_for_improved_healthsystem_and_healthcare_seeking(
                resourcefilepath=self.resourcefilepath,
                max_healthsystem_function=switches["max_healthsystem_function"],
                max_healthcare_seeking=switches["max_healthcare_seeking"],
            ),
            # todo put in the switches for other HSS as other functions, or as part of the func. above?

            # -- Vertical Program Switches
            get_parameters_for_vertical_program_scale_up_hiv_and_tb(
                resourcefilepath=self.resourcefilepath,
                switch_scaleup_hiv=switches["switch_scaleup_hiv"],
                switch_scaleup_tb=switches["switch_scaleup_tb"],
            ),
            get_parameters_for_vertical_program_scale_up_malaria(
                resourcefilepath=self.resourcefilepath,
                switch_scaleup_malaria=switches["switch_scaleup_malaria"],
            ),
            get_parameters_for_vertical_program_scale_up_epi(
                resourcefilepath=self.resourcefilepath,
                switch_scaleup_malaria=switches["switch_scaleup_epi"],
            ),
        )

    @staticmethod
    def _do_change_in_parameters(sim, params_to_update):
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

def get_parameters_for_improved_healthsystem_and_healthcare_seeking(
    resourcefilepath: Path,
    max_healthsystem_function: Optional[bool] = False,
    max_healthcare_seeking: Optional[bool] = False,
) -> Dict:
    """
    This returs the parameters to be updated to represent certain Health System Strengthening Interventions.

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


# --- VERTICAL PROGRAMS
# N.B. Note that these can be separate functions, or one big functions.
# N.B. These functions could be on the ScenarioSwitcher class itself.
def get_parameters_for_vertical_program_scale_up_hiv_and_tb(
    resourcefilepath: Path,
    switch_scaleup_hiv: bool,
    switch_scaleup_tb: bool,
) -> Dict:
    """
    Returns a dictionary of parameters and their updated values to indicate
    the possible scale-up for HIV and TB programs.

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
    return {}


def get_parameters_for_vertical_program_scale_up_malaria(
    resourcefilepath: Path,
    switch_scaleup_malaria: bool,
) -> Dict:
    """
    Returns a dictionary of parameters and their updated values to indicate
    the possible scale-up for Malaria programs.

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
    return {}

def get_parameters_for_vertical_program_scale_up_epi(
    resourcefilepath: Path,
    switch_scaleup_epi: bool,
) -> Dict:
    """
    Returns a dictionary of parameters and their updated values to indicate
    the possible scale-up for EPI programs.

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
    return {}

