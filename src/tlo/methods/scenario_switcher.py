import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict

import pandas as pd

from tlo import Module, Parameter, Types


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


def get_parameters_for_improved_healthsystem_and_healthcare_seeking(
    resourcefilepath: Path,
    max_healthsystem_function: Optional[bool] = False,
    max_healthcare_seeking: Optional[bool] = False,
) -> Dict:
    """
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
