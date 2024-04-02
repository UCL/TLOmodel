import pandas as pd

from tests.test_analysis import resourcefilepath
from tlo import Simulation, Date
from tlo.analysis.utils import get_parameters_for_status_quo
from tlo.methods.fullmodel import fullmodel
from tlo.methods.scenario_switcher import get_parameters_for_improved_healthsystem_and_healthcare_seeking, \
    ImprovedHealthSystemAndCareSeekingScenarioSwitcher



def test_scenario_switcher(seed):
    """Check the `ImprovedHealthSystemAndCareSeekingScenarioSwitcher` module can update parameter values in a manner
    similar to them being changed directly after registration in the simulation (as would be done by the Scenario
    class)."""

    sim = Simulation(start_date=Date(2010, 1, 1), seed=seed)
    sim.register(
        *(
            fullmodel(resourcefilepath=resourcefilepath)
            + [ImprovedHealthSystemAndCareSeekingScenarioSwitcher(resourcefilepath=resourcefilepath)]
        )
    )

    # Check that the 'ScenarioSwitcher` is the first registered module.
    assert 'ImprovedHealthSystemAndCareSeekingScenarioSwitcher' == list(sim.modules.keys())[0]

    # Change the parameters for max_healthsystem_function and max_healthcare_seeking via the ScenarioSwitcher
    # (making them True for the whole simulation; by default they would be False).
    sim.modules['ImprovedHealthSystemAndCareSeekingScenarioSwitcher'].parameters['switch_max_healthsystem_function'] \
        = [True, True]
    sim.modules['ImprovedHealthSystemAndCareSeekingScenarioSwitcher'].parameters['switch_max_healthcare_seeking'] \
        = [True, True]

    # Initialise the population
    sim.make_initial_population(n=100)

    # Check that all the parameter values in the simulation are updated to be the value expected.
    updated_values = get_parameters_for_improved_healthsystem_and_healthcare_seeking(
        resourcefilepath=resourcefilepath,
        max_healthsystem_function=True,
        max_healthcare_seeking=True
    )

    for module, param in updated_values.items():
        for name, target_value in param.items():

            actual = sim.modules[module].parameters[name]

            if isinstance(target_value, pd.Series):
                pd.testing.assert_series_equal(target_value, actual)
            elif isinstance(target_value, pd.DataFrame):
                pd.testing.assert_frame_equal(target_value, actual)
            elif isinstance(target_value, list):
                assert all([t == v for t, v in zip(target_value, actual)])
            else:
                assert target_value == actual

    # Spot check for health care seeking being forced to occur for all symptoms
    hcs = sim.modules['HealthSeekingBehaviour'].force_any_symptom_to_lead_to_healthcareseeking
    assert isinstance(hcs, bool) and hcs


def test_get_parameter_functions(seed):
    """Check that the functions that provide updated parameter values provide recognised parameter names and values
    of the appropriate type."""

    # Function that are designed to provide set of parameters to be updated in a `fullmodel` simulation.
    funcs = [
        get_parameters_for_status_quo,
        lambda: get_parameters_for_improved_healthsystem_and_healthcare_seeking(
                resourcefilepath=resourcefilepath,
                max_healthsystem_function=True,
                max_healthcare_seeking=False
            ),
        lambda: get_parameters_for_improved_healthsystem_and_healthcare_seeking(
            resourcefilepath=resourcefilepath,
            max_healthsystem_function=False,
            max_healthcare_seeking=True
            ),
        lambda: get_parameters_for_improved_healthsystem_and_healthcare_seeking(
            resourcefilepath=resourcefilepath,
            max_healthsystem_function=True,
            max_healthcare_seeking=True
        )
    ]

    # Create simulation
    sim = Simulation(start_date=Date(2010, 1, 1), seed=seed)
    sim.register(*fullmodel(resourcefilepath=resourcefilepath))

    for fn in funcs:

        # Get structure containing parameters to be updated:
        params = fn()

        assert isinstance(params, dict)
        # Check each parameter
        for module in params.keys():
            for name, updated_value in params[module].items():

                # Check that the parameter identified exists in the simulation
                assert name in sim.modules[module].parameters, f"Parameter not recognised: {module}:{name}."

                # Check that the original value and the updated value are of the same type.
                original = sim.modules[module].parameters[name]

                assert type(original) is type(updated_value), \
                    f"Updated value type does not match original type: " \
                    f"{module}:{name} >> {updated_value=}, " \
                    f"{type(original)=}, {type(updated_value)=}"

                def is_df_same_size_and_dtype(df1, df2):
                    return (
                        df1.index.equals(df2.index) and
                        all(df1.dtypes == df2.dtypes) and
                        all(df1.columns == df2.columns) if isinstance(df1, pd.DataFrame) else True
                    )

                def is_list_same_size_and_dtype(l1, l2):
                    return (
                        (len(l1) == len(l2)) and
                        all([type(_i) is type(_j) for _i, _j in zip(l1, l2)])
                    )

                # Check that, if the updated value is a pd.DataFrame, it has the same indicies as the original
                if isinstance(original, (pd.DataFrame, pd.Series)):
                    assert is_df_same_size_and_dtype(original, updated_value), \
                        print(f"Dataframe or series if not of the expected size and shape:"
                              f"{module}:{name} >> {updated_value=}, {type(original)=}, {type(updated_value)=}")

                # Check that, if the updated value is a list/tuple, it has the same dimensions as the original
                elif isinstance(original, (list, tuple)):
                    assert is_list_same_size_and_dtype(original, updated_value), \
                        print(f"List/tuple is not of the expected size and containing elements of expected type: "
                              f"{module}:{name} >> {updated_value=}, {type(original)=}, {type(updated_value)=}")
