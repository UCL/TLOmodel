import os
import textwrap
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

from tlo import Date, DateOffset, Module, Property, Simulation, Types, logging
from tlo.analysis.utils import (
    colors_in_matplotlib,
    flatten_multi_index_series_into_dict_for_logging,
    get_coarse_appt_type,
    get_color_cause_of_death_or_daly_label,
    get_color_coarse_appt,
    get_color_short_treatment_id,
    get_filtered_treatment_ids,
    get_parameters_for_improved_healthsystem_and_healthcare_seeking,
    get_parameters_for_status_quo,
    get_root_path,
    merge_log_files,
    mix_scenarios,
    order_of_coarse_appt,
    order_of_short_treatment_ids,
    parse_log_file,
    summarise,
    summarize,
    unflatten_flattened_multi_index_in_logging,
)
from tlo.events import PopulationScopeEventMixin, RegularEvent
from tlo.methods import demography
from tlo.methods.fullmodel import fullmodel
from tlo.methods.scenario_switcher import ImprovedHealthSystemAndCareSeekingScenarioSwitcher

resourcefilepath = Path(os.path.dirname(__file__)) / "../resources"


def test_parse_log():
    log_file = Path(__file__).parent / "resources" / "structured_log.txt"

    output = parse_log_file(log_file)

    assert "tlo.methods.epilepsy" in output
    assert set(output["tlo.methods.epilepsy"].keys()) == {
        "incidence_epilepsy",
        "epilepsy_logging",
        "_metadata",
    }


def test_parse_log_levels(tmpdir):
    # setup a toy simulation to test logging
    logger = logging.getLogger("tlo.methods.dummy")
    start_date = Date(2010, 1, 1)
    end_date = Date(2011, 1, 1)
    pop_size = 100

    class DummyEvent(RegularEvent, PopulationScopeEventMixin):
        def apply(self, population):
            logger.info(key="info_level1", data={"number": np.random.randint(0, 100)})
            logger.info(key="info_level2", data={"number": np.random.randint(0, 100)})
            logger.debug(key="debug_level", data={"number": np.random.randint(0, 100)})

    class Dummy(Module):
        PROPERTIES = {"dummy": Property(Types.INT, description="dummy")}

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def on_birth(self, mother, child):
            pass

        def initialise_simulation(self, sim: Simulation):
            sim.schedule_event(
                DummyEvent(self, frequency=DateOffset(months=1)), start_date
            )

    # test parsing when log level is INFO
    sim = Simulation(
        start_date=start_date, log_config={"filename": "temp", "directory": tmpdir}
    )
    sim.register(Dummy())
    logger.setLevel(logging.INFO)
    sim.make_initial_population(n=pop_size)
    sim.simulate(end_date=end_date)
    output = parse_log_file(sim.log_filepath)

    # At INFO level
    assert (
            len(output["tlo.methods.dummy"]["_metadata"]["tlo.methods.dummy"]) == 2
    )  # should have two tables

    # tables should be at level INFO
    for k, v in output["tlo.methods.dummy"]["_metadata"]["tlo.methods.dummy"].items():
        assert v["level"] == "INFO"

    # test parsing when log level is DEBUG
    sim = Simulation(
        start_date=start_date, log_config={"filename": "temp2", "directory": tmpdir}
    )
    sim.register(Dummy())
    logger.setLevel(logging.DEBUG)
    sim.make_initial_population(n=pop_size)
    sim.simulate(end_date=end_date)
    output = parse_log_file(
        sim.log_filepath, level=logging.INFO
    )  # we're parsing everything above INFO level

    # logged DEBUG but parsed at INFO levels
    assert len(output["tlo.methods.dummy"]["_metadata"]["tlo.methods.dummy"]) == 2
    assert "debug_level" not in output["tlo.methods.dummy"]

    # logged DEBUG and parsed at DEBUG level
    output = parse_log_file(sim.log_filepath, level=logging.DEBUG)
    assert len(output["tlo.methods.dummy"]["_metadata"]["tlo.methods.dummy"]) == 3
    assert "debug_level" in output["tlo.methods.dummy"]


def test_flattening_and_unflattening_multiindex(tmpdir):
    """Check that a pd.Series with multi-index can be "flattened" (in order to log it) and "unflattened" (after reading
    the log) using `flatten_multi_index_series_into_dict_for_logging` and `unflatten_flattened_multi_index_in_logging`,
    respectively."""

    def run_simulation_and_parse_log(series_to_log: pd.Series) -> pd.DataFrame:
        """Run the simulation, convert a pd.Series to a dict using `flatten_multi_index_series_into_dict_for_logging`,
        log it to a particular key on the first date of the simulation, and return the pd.DataFrame created by
        `parse_log` for that key."""

        logger = logging.getLogger("tlo.methods.demography")
        logger.setLevel(logging.INFO)

        class DummyModule(Module):
            def read_parameters(self, data_folder):
                pass

            def initialise_population(self, population):
                pass

            def initialise_simulation(self, sim):
                logger.info(
                    key="key",
                    data=flatten_multi_index_series_into_dict_for_logging(
                        series_to_log
                    ),
                )

        sim = Simulation(
            start_date=sim_start_date,
            seed=0,
            log_config={"filename": "temp", "directory": tmpdir, },
        )
        sim.register(
            demography.Demography(resourcefilepath=resourcefilepath), DummyModule()
        )
        sim.make_initial_population(n=100)
        sim.simulate(end_date=sim_start_date)

        return parse_log_file(sim.log_filepath)["tlo.methods.demography"]["key"]

    sim_start_date = Date(2010, 1, 1)

    for num_of_levels in range(1, 4):
        # Make original pd.Series with column-wise multi-index (with the specified number of levels).
        idx = pd.MultiIndex.from_product(
            [["1", "2", "3"] for _ in range(num_of_levels)],
            names=[f"col_level_{_x}" for _x in range(num_of_levels)],
        )
        original = pd.Series(
            index=idx, data=100 * np.random.random([len(idx)]).round(4)
        )

        # Let this original series be logged in the simulation and get the parsed log;
        df_rtn = run_simulation_and_parse_log(series_to_log=original)

        # Confirm that the original can retrieved from the log using `unflatten_flattened_multi_index_in_logging`
        series_unflattened = unflatten_flattened_multi_index_in_logging(
            df_rtn.loc[pd.to_datetime(df_rtn.date) == sim_start_date].drop(
                columns=["date"]
            )
        ).iloc[0]

        # Check equal
        pd.testing.assert_series_equal(original, series_unflattened.rename(None))


def test_get_root_path():
    """Check that `get_root_path` works as expected."""

    ROOT_PATH = Path(os.path.abspath(Path(os.path.dirname(__file__)) / "../"))

    def is_correct_absolute_path(_path):
        return (ROOT_PATH == _path) and _path.is_absolute() and isinstance(_path, Path)

    assert is_correct_absolute_path(get_root_path())

    for test_dir in [
        os.path.abspath(Path(os.path.dirname(__file__)) / "../src/"),
        os.path.abspath(Path(os.path.dirname(__file__)) / "../resources/"),
        os.path.abspath(Path(os.path.dirname(__file__)) / "../tests/"),
        os.path.abspath(Path(os.path.dirname(__file__)) / "../"),
        os.path.abspath(Path(os.path.dirname(__file__))),
    ]:
        assert is_correct_absolute_path(
            get_root_path(test_dir)
        ), f"Failed on {test_dir=}"


def test_coarse_appt_type():
    """Check the function that maps each appt_types to a coarser definition."""
    appt_types = pd.read_csv(
        resourcefilepath
        / "healthsystem"
        / "human_resources"
        / "definitions"
        / "ResourceFile_Appt_Types_Table.csv"
    )["Appt_Type_Code"].values

    appts = pd.DataFrame(
        {
            "original": pd.Series(appt_types),
            "coarse": pd.Series(appt_types).map(get_coarse_appt_type),
        }
    )

    coarse_appts = appts["coarse"].drop_duplicates()

    assert not pd.isnull(appts).any().any()
    assert 13 == len(coarse_appts)  # 12 coarse categories

    # Check can run sorting on these
    assert 13 == len(sorted(coarse_appts, key=order_of_coarse_appt))


def test_colormap_coarse_appts():
    """Check the function that allocates a unique colour to each coarse appointment type."""
    coarse_appt_types = (
        pd.read_csv(
            resourcefilepath
            / "healthsystem"
            / "human_resources"
            / "definitions"
            / "ResourceFile_Appt_Types_Table.csv"
        )["Appt_Type_Code"]
        .map(get_coarse_appt_type)
        .drop_duplicates()
        .values
    )

    coarse_appt_types = sorted(coarse_appt_types, key=order_of_coarse_appt)

    colors = [get_color_coarse_appt(x) for x in coarse_appt_types]

    assert len(set(colors)) == len(colors)  # No duplicates
    assert all([isinstance(_x, str) for _x in colors])  # All strings
    assert np.nan is get_color_coarse_appt(
        "????"
    )  # Return `np.nan` if appt_type not recognised.
    assert all(
        map(lambda x: x in colors_in_matplotlib(), colors)
    )  # All colors recognised


def test_get_treatment_ids(tmpdir):
    """Check the function that generates the list of TREATMENT_IDs defined in the model."""

    x = get_filtered_treatment_ids()  # All TREATMENT_IDs
    y = get_filtered_treatment_ids(
        depth=1
    )  # TREATMENT_IDs to the first level of depth (i.e. module level)

    assert isinstance(x, list)
    assert all([isinstance(_x, str) for _x in x])

    assert isinstance(y, list)
    assert all([isinstance(_y, str) for _y in y])

    assert len(y) < len(x)


def test_colormap_short_treatment_id():
    """Check the function that allocates a unique colour to each shortened TREATMENT_ID (i.e. each module)"""

    short_treatment_ids = sorted(
        get_filtered_treatment_ids(depth=1), key=order_of_short_treatment_ids
    )
    colors = [get_color_short_treatment_id(x) for x in short_treatment_ids]

    assert len(set(colors)) == len(colors)  # No duplicates
    assert all([isinstance(_x, str) for _x in colors])  # All strings
    assert np.nan is get_color_coarse_appt(
        "????"
    )  # Return `np.nan` if appt_type not recognised.
    assert all(
        map(lambda x: x in colors_in_matplotlib(), colors)
    )  # All colors recognised


def test_colormap_cause_of_death_label(seed):
    """Check that all the Cause-of-Deaths labels defined in the full model are assigned to a unique colour when
     plotting."""

    def get_all_cause_of_death_labels(seed=0) -> List[str]:
        """Return list of all the causes of death defined in the full model."""
        start_date = Date(2010, 1, 1)
        sim = Simulation(start_date=start_date, seed=seed)
        sim.register(
            *fullmodel(resourcefilepath=resourcefilepath, use_simplified_births=False)
        )
        sim.make_initial_population(n=1_000)
        sim.simulate(end_date=start_date)
        mapper, _ = (
            sim.modules["Demography"]
        ).create_mappers_from_causes_of_death_to_label()
        return sorted(set(mapper.values()))

    all_labels = get_all_cause_of_death_labels(seed)

    colors = [get_color_cause_of_death_or_daly_label(_label) for _label in all_labels]

    assert len(set(colors)) == len(colors)  # No duplicates
    assert all([isinstance(_x, str) for _x in colors])  # All strings
    assert np.nan is get_color_coarse_appt(
        "????"
    )  # Return `np.nan` if label is not recognised.
    assert all(
        map(lambda x: x in colors_in_matplotlib(), colors)
    )  # All colors recognised


def test_get_parameter_functions(seed):
    """Check that the functions that provide updated parameter values provide recognised parameter names and values
    of the appropriate type."""

    # Function that are designed to provide set of parameters to be updated in a `fullmodel` simulation.
    funcs = [
        get_parameters_for_status_quo,
        lambda: get_parameters_for_improved_healthsystem_and_healthcare_seeking(
            resourcefilepath=resourcefilepath,
            max_healthsystem_function=True,
            max_healthcare_seeking=False,
        ),
        lambda: get_parameters_for_improved_healthsystem_and_healthcare_seeking(
            resourcefilepath=resourcefilepath,
            max_healthsystem_function=False,
            max_healthcare_seeking=True,
        ),
        lambda: get_parameters_for_improved_healthsystem_and_healthcare_seeking(
            resourcefilepath=resourcefilepath,
            max_healthsystem_function=True,
            max_healthcare_seeking=True,
        ),
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
                assert (
                        name in sim.modules[module].parameters
                ), f"Parameter not recognised: {module}:{name}."

                # Check that the original value and the updated value are of the same type.
                original = sim.modules[module].parameters[name]

                assert type(original) is type(updated_value), (
                    f"Updated value type does not match original type: "
                    f"{module}:{name} >> {updated_value=}, "
                    f"{type(original)=}, {type(updated_value)=}"
                )

                def is_df_same_size_and_dtype(df1, df2):
                    return (
                        df1.index.equals(df2.index)
                        and all(df1.dtypes == df2.dtypes)
                        and all(df1.columns == df2.columns)
                        if isinstance(df1, pd.DataFrame)
                        else True
                    )

                def is_list_same_size_and_dtype(l1, l2):
                    return (len(l1) == len(l2)) and all(
                        [type(_i) is type(_j) for _i, _j in zip(l1, l2)]
                    )

                # Check that, if the updated value is a pd.DataFrame, it has the same indicies as the original
                if isinstance(original, (pd.DataFrame, pd.Series)):
                    assert is_df_same_size_and_dtype(original, updated_value), print(
                        f"Dataframe or series if not of the expected size and shape:"
                        f"{module}:{name} >> {updated_value=}, {type(original)=}, {type(updated_value)=}"
                    )

                # Check that, if the updated value is a list/tuple, it has the same dimensions as the original
                elif isinstance(original, (list, tuple)):
                    assert is_list_same_size_and_dtype(original, updated_value), print(
                        f"List/tuple is not of the expected size and containing elements of expected type: "
                        f"{module}:{name} >> {updated_value=}, {type(original)=}, {type(updated_value)=}"
                    )


def test_mix_scenarios():
    """Check that `mix_scenarios` works as expected."""

    d1 = {"Mod1": {"param_a": "value_in_d1", "param_b": "value_in_d1", }}

    d2 = {"Mod2": {"param_a": "value_in_d2", "param_b": "value_in_d2", }}

    d3 = {"Mod1": {"param_b": "value_in_d3", "param_c": "value_in_d3"}}

    with pytest.warns(UserWarning) as record:
        assert mix_scenarios(d1, d2, d3) == {
            "Mod1": {
                "param_a": "value_in_d1",  # <- only appears in d1, and is included despite d3 also having 'Mod1' key
                "param_b": "value_in_d3",  # <- appears in d1 and d3, but d3 is right-most, so 'wins' (raises Warning)
                "param_c": "value_in_d3",  # <- only appears in d3
            },
            "Mod2": {
                "param_a": "value_in_d2",  # <- only appears in d2 (& attaches to Mod2 despite name being duplicated)
                "param_b": "value_in_d2",  # <- only appears in d2 (& attaches to Mod2 despite name being duplicated)
            },
        }

    assert 1 == len(record)
    assert (
            record.list[0].message.args[0]
            == "Parameter is being updated more than once: module=Mod1, parameter=param_b"
    )

    # Test the behaviour of the `mix_scenarios` taking the value in the right-most dict.
    assert mix_scenarios(
        {
            "Mod1": {
                "param_a": "value_in_dict1",
                "param_b": "value_in_dict1",
                "param_c": "value_in_dict1",
            }
        },
        {
            "Mod1": {
                "param_a": "value_in_dict2",
                "param_b": "value_in_dict2",
                "param_c": "value_in_dict2",
            }
        },
        {
            "Mod1": {
                "param_a": "value_in_dict3",
                "param_b": "value_in_dict_right_most",
                "param_c": "value_in_dict3",
            }
        },
        {"Mod1": {"param_a": "value_in_dict_right_most", "param_c": "value_in_dict4", }},
        {"Mod1": {"param_c": "value_in_dict_right_most", }},
    ) == {
               "Mod1": {
                   "param_a": "value_in_dict_right_most",
                   "param_b": "value_in_dict_right_most",
                   "param_c": "value_in_dict_right_most",
               }
           }


def test_improved_healthsystem_and_care_seeking_scenario_switcher(seed):
    """Check the `ImprovedHealthSystemAndCareSeekingScenarioSwitcher` module can update complex parameter values in a
     manner similar to them being changed directly or mid-way through the simulation."""

    # Define the changes we want the ScenarioSwitcher to implement
    max_healthsystem_function = [False, True]
    max_healthcare_seeking = [False, True]
    year_of_change = 2011

    # Set up a simulation in which the parameters are checked regularly to see if they are correct, given the
    # phase of the simulation.

    class CheckParametersEvent(RegularEvent, PopulationScopeEventMixin):
        def __init__(self, module):
            super().__init__(module, frequency=DateOffset(months=1))  # repeats every month

        def apply(self, population):
            self.module.check_parameters()  # Checks parameters are as expected

    class DummyModule(Module):
        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            # Schedule `CheckParametersEvent` to run immediately (and it will then repeat monthly).
            sim.schedule_event(CheckParametersEvent(self), sim.date)

        def on_birth(self, *args, **kwargs):
            pass

        def check_parameters(self) -> None:
            """Check that the parameter values in the simulation currently match expectations for this phase of the
            simulation."""

            sim = self.sim

            # Work out if we expect to be using the first or the second values for switchers (first value is
            # for times before 1st Jan on the year of the change.
            phase_of_simulation = 0 if sim.date.year < year_of_change else 1

            # Load the parameters that should be being used currently.
            correct_param_values = get_parameters_for_improved_healthsystem_and_healthcare_seeking(
                resourcefilepath=resourcefilepath,
                max_healthsystem_function=max_healthsystem_function[phase_of_simulation],
                max_healthcare_seeking=max_healthcare_seeking[phase_of_simulation],
            )

            for mod, param in correct_param_values.items():
                for name, target_value in param.items():
                    actual = sim.modules[mod].parameters[name]
                    if isinstance(target_value, pd.Series):
                        pd.testing.assert_series_equal(target_value, actual)
                    elif isinstance(target_value, pd.DataFrame):
                        pd.testing.assert_frame_equal(target_value, actual)
                    elif isinstance(target_value, list):
                        assert all([t == v for t, v in zip(target_value, actual)])
                    else:
                        assert target_value == actual
                    print('Parameters all look good.')

            # Check for health care seeking being forced to occur for all symptoms
            hcs = sim.modules["HealthSeekingBehaviour"].force_any_symptom_to_lead_to_healthcareseeking
            assert isinstance(hcs, bool) and (hcs is max_healthcare_seeking[phase_of_simulation])

    sim = Simulation(start_date=Date(2010, 1, 1), seed=seed)
    sim.register(
        *(
                fullmodel(resourcefilepath=resourcefilepath)
                + [
                    ImprovedHealthSystemAndCareSeekingScenarioSwitcher(
                        resourcefilepath=resourcefilepath
                    ),
                    DummyModule(),
                ]
        )
    )

    # Check that the `ImprovedHealthSystemAndCareSeekingScenarioSwitcher` is the first registered module.
    assert (
            "ImprovedHealthSystemAndCareSeekingScenarioSwitcher"
            == list(sim.modules.keys())[0]
    )
    module = sim.modules["ImprovedHealthSystemAndCareSeekingScenarioSwitcher"]

    # Set the changes for the ScenarioSwitcher by manipulating its parameters (mimicking what `Scenario` class does).
    module.parameters["year_of_switch"] = year_of_change
    module.parameters["max_healthsystem_function"] = max_healthsystem_function
    module.parameters["max_healthcare_seeking"] = max_healthcare_seeking

    # Initialise the population
    sim.make_initial_population(n=100)

    # Run the simulation until well after the date of parameter change. (The checking will be occurring every month,
    # and if any errors an `AssertionError` would be raised.)
    sim.simulate(end_date=Date(year_of_change + 2, 1, 1))


def test_summarise():
    """Check that the summarize utility function works as expected."""

    results_multiple_draws = pd.DataFrame(
        columns=pd.MultiIndex.from_tuples(
            [
                ("DrawA", "DrawA_Run1"),
                ("DrawA", "DrawA_Run2"),
                ("DrawB", "DrawB_Run1"),
                ("DrawB", "DrawB_Run2"),
            ],
            names=("draw", "run"),
        ),
        index=["TimePoint0", "TimePoint1"],
        data=np.array([[0, 20, 1000, 2000], [0, 20, 1000, 2000], ]),
    )

    results_one_draw = pd.DataFrame(
        columns=pd.MultiIndex.from_tuples(
            [("DrawA", "DrawA_Run1"), ("DrawA", "DrawA_Run2")], names=("draw", "run")
        ),
        index=["TimePoint0", "TimePoint1"],
        data=np.array([[0, 20], [0, 20]]),
    )

    # Without collapsing and all stats provided
    pd.testing.assert_frame_equal(
        pd.DataFrame(
            columns=pd.MultiIndex.from_tuples(
                [
                    ("DrawA", "lower"),
                    ("DrawA", "central"),
                    ("DrawA", "upper"),
                    ("DrawB", "lower"),
                    ("DrawB", "central"),
                    ("DrawB", "upper"),
                ],
                names=("draw", "stat"),
            ),
            index=["TimePoint0", "TimePoint1"],
            data=np.array(
                [
                    [0.5, 10.0, 19.5, 1025.0, 1500.0, 1975.0],
                    [0.5, 10.0, 19.5, 1025.0, 1500.0, 1975.0],
                ]
            ),
        ),
        summarise(results_multiple_draws, central_measure='mean'),
    )

    # Without collapsing and only mean
    pd.testing.assert_frame_equal(
        pd.DataFrame(
            columns=pd.Index(["DrawA", "DrawB"], name="draw"),
            index=["TimePoint0", "TimePoint1"],
            data=np.array([[10.0, 1500.0], [10.0, 1500.0]]),
        ),
        summarise(results_multiple_draws, central_measure='mean', only_central=True),
    )

    # With collapsing (as only one draw)
    pd.testing.assert_frame_equal(
        pd.DataFrame(
            columns=pd.Index(["lower", "central", "upper"], name="stat"),
            index=["TimePoint0", "TimePoint1"],
            data=np.array([[0.5, 10.0, 19.5], [0.5, 10.0, 19.5], ]),
        ),
        summarise(results_one_draw, central_measure='mean', collapse_columns=True),
    )

    # Check that summarize() produces legacy behaviour:
    pd.testing.assert_frame_equal(
        summarise(results_multiple_draws, central_measure='mean').rename(columns={'central': 'mean'}, level=1),
        summarize(results_multiple_draws)
    )
    pd.testing.assert_frame_equal(
        summarise(results_multiple_draws, central_measure='mean', only_central=True),
        summarize(results_multiple_draws, only_mean=True)
    )
    pd.testing.assert_frame_equal(
        summarise(results_one_draw, central_measure='mean', collapse_columns=True),
        summarize(results_one_draw, collapse_columns=True)
    )

def test_control_loggers_from_same_module_independently(seed, tmpdir):
    """Check that detailed/summary loggers in the same module can configured independently."""

    # Check that the simulation can be set-up to get only the usual demography logger and *not* the detailed
    #  logger, when providing the config_log information when the simulation is initialised."""

    log_config = {
        'filename': 'temp',
        'directory': tmpdir,
        'custom_levels': {
            "*": logging.WARNING,
            'tlo.methods.demography.detail': logging.WARNING,  # <-- Don't explicitly turn off the detailed logger
            'tlo.methods.demography': logging.INFO,  # <-- Turning on the normal logger
        }
    }

    def run_simulation_and_cause_one_death(sim):
        """Register demography in the simulations, runs it and causes one death; return the resulting log."""
        sim.register(demography.Demography(resourcefilepath=resourcefilepath))
        sim.make_initial_population(n=100)
        sim.simulate(end_date=sim.start_date)
        # Cause one death to occur
        sim.modules['Demography'].do_death(
            individual_id=0,
            originating_module=sim.modules['Demography'],
            cause='Other'
        )
        return parse_log_file(sim.log_filepath)

    def check_log(log):
        """Check the usual `tlo.methods.demography' log is created and that check persons have died (which would be
        when the detailed logger would be used)."""
        assert 'tlo.methods.demography' in log.keys()
        assert 1 == len(log['tlo.methods.demography']['death'])

        # Check that the detailed logger is not created.
        assert 'tlo.methods.demography.detail' not in log.keys()

    # 1) Provide custom_logs argument when creating Simulation object
    sim = Simulation(start_date=Date(2010, 1, 1), seed=seed, log_config=log_config)
    check_log(run_simulation_and_cause_one_death(sim))


def test_merge_log_files(tmp_path):
    log_file_path_1 = tmp_path / "log_file_1"
    log_file_path_1.write_text(
        textwrap.dedent(
            """\
            {"uuid": "b07", "type": "header", "module": "m0", "key": "info", "level": "INFO", "columns": {"msg": "str"}, "description": null}
            {"uuid": "b07", "date": "2010-01-01T00:00:00", "values": ["0"]}
            {"uuid": "0b3", "type": "header", "module": "m1", "key": "a", "level": "INFO", "columns": {"msg": "str"}, "description": "A"}
            {"uuid": "0b3", "date": "2010-01-01T00:00:00", "values": ["1"]}
            {"uuid": "ed4", "type": "header", "module": "m2", "key": "b", "level": "INFO", "columns": {"msg": "str"}, "description": "B"}
            {"uuid": "ed4", "date": "2010-01-02T00:00:00", "values": ["2"]}
            {"uuid": "477", "type": "header", "module": "m2", "key": "c", "level": "INFO", "columns": {"msg": "str"}, "description": "C"}
            {"uuid": "477", "date": "2010-01-02T00:00:00", "values": ["3"]}
            {"uuid": "b5c", "type": "header", "module": "m2", "key": "d", "level": "INFO", "columns": {"msg": "str"}, "description": "D"}
            {"uuid": "b5c", "date": "2010-01-03T00:00:00", "values": ["4"]}
            {"uuid": "477", "date": "2010-01-03T00:00:00", "values": ["5"]}
            """
        )
    )
    log_file_path_2 = tmp_path / "log_file_2"
    log_file_path_2.write_text(
        textwrap.dedent(
            """\
            {"uuid": "b07", "type": "header", "module": "m0", "key": "info", "level": "INFO", "columns": {"msg": "str"}, "description": null}
            {"uuid": "b07", "date": "2010-01-04T00:00:00", "values": ["6"]}
            {"uuid": "ed4", "type": "header", "module": "m2", "key": "b", "level": "INFO", "columns": {"msg": "str"}, "description": "B"}
            {"uuid": "ed4", "date": "2010-01-04T00:00:00", "values": ["7"]}
            {"uuid": "ed4", "date": "2010-01-05T00:00:00", "values": ["8"]}
            {"uuid": "0b3", "type": "header", "module": "m1", "key": "a", "level": "INFO", "columns": {"msg": "str"}, "description": "A"}
            {"uuid": "0b3", "date": "2010-01-06T00:00:00", "values": ["9"]}
            {"uuid": "a19", "type": "header", "module": "m3", "key": "e", "level": "INFO", "columns": {"msg": "str"}, "description": "E"}
            {"uuid": "a19", "date": "2010-01-03T00:00:00", "values": ["10"]}
            """
        )
    )
    expected_merged_log_file_content = textwrap.dedent(
        """\
        {"uuid": "b07", "type": "header", "module": "m0", "key": "info", "level": "INFO", "columns": {"msg": "str"}, "description": null}
        {"uuid": "b07", "date": "2010-01-01T00:00:00", "values": ["0"]}
        {"uuid": "0b3", "type": "header", "module": "m1", "key": "a", "level": "INFO", "columns": {"msg": "str"}, "description": "A"}
        {"uuid": "0b3", "date": "2010-01-01T00:00:00", "values": ["1"]}
        {"uuid": "ed4", "type": "header", "module": "m2", "key": "b", "level": "INFO", "columns": {"msg": "str"}, "description": "B"}
        {"uuid": "ed4", "date": "2010-01-02T00:00:00", "values": ["2"]}
        {"uuid": "477", "type": "header", "module": "m2", "key": "c", "level": "INFO", "columns": {"msg": "str"}, "description": "C"}
        {"uuid": "477", "date": "2010-01-02T00:00:00", "values": ["3"]}
        {"uuid": "b5c", "type": "header", "module": "m2", "key": "d", "level": "INFO", "columns": {"msg": "str"}, "description": "D"}
        {"uuid": "b5c", "date": "2010-01-03T00:00:00", "values": ["4"]}
        {"uuid": "477", "date": "2010-01-03T00:00:00", "values": ["5"]}
        {"uuid": "b07", "date": "2010-01-04T00:00:00", "values": ["6"]}
        {"uuid": "ed4", "date": "2010-01-04T00:00:00", "values": ["7"]}
        {"uuid": "ed4", "date": "2010-01-05T00:00:00", "values": ["8"]}
        {"uuid": "0b3", "date": "2010-01-06T00:00:00", "values": ["9"]}
        {"uuid": "a19", "type": "header", "module": "m3", "key": "e", "level": "INFO", "columns": {"msg": "str"}, "description": "E"}
        {"uuid": "a19", "date": "2010-01-03T00:00:00", "values": ["10"]}
        """
    )
    merged_log_file_path = tmp_path / "merged_log_file"
    merge_log_files(log_file_path_1, log_file_path_2, merged_log_file_path)
    merged_log_file_content = merged_log_file_path.read_text()
    assert merged_log_file_content == expected_merged_log_file_content


def test_merge_log_files_with_inconsistent_headers_raises(tmp_path):
    log_file_path_1 = tmp_path / "log_file_1"
    log_file_path_1.write_text(
        textwrap.dedent(
            """\
            {"uuid": "b07", "type": "header", "module": "m0", "key": "info", "level": "INFO", "columns": {"msg": "str"}, "description": null}
            {"uuid": "b07", "date": "2010-01-01T00:00:00", "values": ["0"]}
            """
        )
    )
    log_file_path_2 = tmp_path / "log_file_2"
    log_file_path_2.write_text(
        textwrap.dedent(
            """\
            {"uuid": "b07", "type": "header", "module": "m0", "key": "info", "level": "INFO", "columns": {"msg": "int"}, "description": null}
            {"uuid": "b07", "date": "2010-01-04T00:00:00", "values": [1]}
            """
        )
    )
    merged_log_file_path = tmp_path / "merged_log_file"
    with pytest.raises(RuntimeError, match="Inconsistent header lines"):
        merge_log_files(log_file_path_1, log_file_path_2, merged_log_file_path)


def test_merge_log_files_inplace_raises(tmp_path):
    log_file_path_1 = tmp_path / "log_file_1"
    log_file_path_1.write_text("foo")
    log_file_path_2 = tmp_path / "log_file_2"
    log_file_path_2.write_text("bar")
    with pytest.raises(ValueError, match="output_path"):
        merge_log_files(log_file_path_1, log_file_path_2, log_file_path_1)
    with pytest.raises(ValueError, match="output_path"):
        merge_log_files(log_file_path_1, log_file_path_2, log_file_path_2)
