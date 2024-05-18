"""Creating and running simulation scenarios for TLOmodel.

Scenarios are used to specify, configure and run a single or set of TLOmodel
simulations. A scenario is created by subclassing ``BaseScenario`` and specifying the
scenario options therein. You can override parameters of the simulation modules in
various ways in the scenario. See the ``BaseScenario`` class for more information.

The subclass of ``BaseScenario`` is then used to create *draws*, which can be considered
a fully-specified configuration of the scenario, or a parameter draw.

Each draw is *run* one or more times - run is a single execution of the simulation. Each
run of a draw has a different seed but is otherwise identical. Each run has its own
simulation seed, introducing randomness into each simulation. A collection of runs for a
given draw describes the random variation in the simulation.

A simple example of a subclass of ``BaseScenario``::

    class MyTestScenario(BaseScenario):
        def __init__(self):
            super().__init__(
                seed = 12,
                start_date = Date(2010, 1, 1),
                end_date = Date(2011, 1, 1),
                initial_population_size = 200,
                number_of_draws = 2,
                runs_per_draw = 2,
            )

        def log_configuration(self):
            return {
                'filename': 'my_test_scenario',
                'directory': './outputs',
                'custom_levels': {'*': logging.INFO}
            }

        def modules(self):
            return [
                demography.Demography(resourcefilepath=self.resources),
                enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            ]

        def draw_parameters(self, draw_number, rng):
            return {
                'Lifestyle': {
                    'init_p_urban': rng.randint(10, 20) / 100.0,
                }
            }

In summary:

* A *scenario* specifies the configuration of the simulation. The simulation start and
  end dates, initial population size, logging setup and registered modules. Optionally,
  you can also override parameters of modules.
* A *draw* is a realisation of a scenario configuration. A scenario can have one or more
  draws. Draws are uninteresting unless you are overriding parameters. If you do not
  override any model parameters, you would only have one draw.
* A *run* is the result of running the simulation using a specific configuration. Each
  draw would run one or more times. Each run for the same draw will have an identical
  configuration except the simulation seed.
"""

import abc
import argparse
import datetime
import json
import pickle
from collections.abc import Iterable
from itertools import product
from pathlib import Path, PurePosixPath
from typing import List, Optional

import numpy as np

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.util import str_to_pandas_date

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_INT = 2**31 - 1


class BaseScenario(abc.ABC):
    """An abstract base class for creating scenarios.

    A scenario is a configuration of a simulation. Users should subclass this class and
    must implement the following methods:

    * ``__init__`` - to set scenario attributes,
    * ``log_configuration`` - to configure filename, directory and logging levels for
      simulation output,
    * ``modules`` - to list disease, intervention and health system modules for the
      simulation.

    Users may also optionally implement:

    * ``draw_parameters`` - override parameters for draws from the scenario.
    """
    def __init__(
        self,
        seed: Optional[int] = None,
        start_date: Optional[Date] = None,
        end_date: Optional[Date] = None,
        initial_population_size: Optional[int] = None,
        number_of_draws: int = 1,
        runs_per_draw: int = 1,
        resources_path: Path = Path("./resources"),
    ):
        """
        :param seed: The top-level seed to use for generating per run simulation seeds
            for random number generators.
        :param start_date: Date to start simulation at.
        :param end_date: Date to end simulation at.
        :param initial_population_size: Number of individuals to initialise population
            with.
        :param number_of_draws: Number of draws (distinct parameter sets) over which to
            run simulation.
        :param runs_per_draw: Number of independent model runs to perform per draw.
        :param resources_path: Path to the directory containing resource files.
        """
        self.seed = seed
        self.start_date = start_date
        self.end_date = end_date
        self.pop_size = initial_population_size
        self.number_of_draws = number_of_draws
        self.runs_per_draw = runs_per_draw
        self.resources = resources_path
        self.rng = None
        self.scenario_path = None
        self.arguments = []

    def parse_arguments(self, extra_arguments: List[str]) -> None:
        """Base class command line arguments handling for scenarios. This should not be overridden by subclasses.
        Subclasses can add argument handling to their classes by implementing the `add_arguments` method."""

        if extra_arguments is None:
            return

        assert isinstance(extra_arguments, Iterable), "Arguments must be a list of strings"

        self.arguments = extra_arguments

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--resume-simulation",
            type=str,
            help="Directory containing suspended state files to resume simulation from",
        )
        parser.add_argument(
            "--suspend-date",
            type=str_to_pandas_date,
            help="Date to suspend the simulation at",
        )

        # add arguments from the subclass
        self.add_arguments(parser)

        arguments = parser.parse_args(self.arguments)

        # set the arguments as attributes of the scenario
        for key, value in vars(arguments).items():
            if value is not None:
                if hasattr(self, key):
                    logger.info(key="message", data=f"Overriding attribute: {key}: {getattr(self, key)} -> {value}")
                setattr(self, key, value)

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add scenario-specific arguments that can be passed to scenario from the command line.

        This method is called to add scenario-specific arguments to the command line parser. The method should add
        arguments to the parser using the `add_argument` method. Arguments that have a value of None are not set or
        overridden.

        :param parser: An instance of `argparse.ArgumentParser` to which arguments should be added.

        Example::

            parser.add_argument('--pop-size', type=int, default=20_000, help='Population size')
        """
        pass

    @abc.abstractmethod
    def log_configuration(self, **kwargs):
        """Implementation must return a dictionary configuring logging.

        Example::

            return {
                'filename': 'test_scenario',
                'directory': './outputs',
                'custom_levels': {
                    '*': logging.WARNING,
                    'tlo.methods.demography': logging.INFO
                }
            }
        """

    @abc.abstractmethod
    def modules(self):
        """Implementation must return a list of instances of TLOmodel modules to
        register in the simulation.

        Example::

            return [
                demography.Demography(resourcefilepath=self.resources),
                enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
                healthsystem.HealthSystem(
                    resourcefilepath=self.resources,
                    disable=True,d
                    service_availability=['*']
                ),
                ...
            ]
        """

    def draw_parameters(self, draw_number, rng):
        """Implementation must return a dictionary of parameters to override for each draw.

        The overridden parameters must be scalar (i.e. float, integer or string) as the following examples demonstrate.
        The argument ``draw_number`` and a random number generator are available, if required.

        * Change a parameter to a fixed value: ``{'Labour': {'average_age_at_pregnancy': 25}}``
        * Sample a value from a distribution: ``{'Lifestyle': {'init_p_urban': rng.randint(10, 20) / 100.0}}``
        * Set a value based on the draw number: ``{'Labour': {'average_age_at_pregnancy': [25, 30, 35][draw_number]}}``

        Implementation of this method in a subclass is optional. If no parameters are to be overridden,
        returns ``None``. If no parameters are overridden, only one draw of the scenario is required.

        A full example for a scenario with 10 draws::

            return {
                'Lifestyle': {
                    'init_p_urban': rng.randint(10, 20) / 100.0,
                },
                'Labour': {
                    'average_age_at_pregnancy': -10 * rng.exponential(0.1),
                    'some_other_parameter': np.arange(0.1, 1.1, 0.1)[draw_number]
                },
            }

        :param int draw_number: the specific draw number currently being executed by the simulation engine
        :param numpy.random.RandomState rng: the scenario's random number generator for sampling from distributions
        """
        return None

    def get_log_config(self, override_output_directory=None):
        """Returns the log configuration for the scenario, with some post_processing."""
        log_config = self.log_configuration()

        # If scenario doesn't have log filename specified, we used the scenario script name
        if "filename" not in log_config or log_config["filename"] is None:
            log_config["filename"] = Path(self.scenario_path).stem

        if override_output_directory is not None:
            log_config["directory"] = override_output_directory

        # If directory is specified, we always write log files - so don't print to stdout
        if "directory" in log_config and log_config["directory"] is not None:
            log_config["suppress_stdout"] = True

        return log_config

    def save_draws(self, return_config=False, **kwargs):
        generator = DrawGenerator(self, self.number_of_draws, self.runs_per_draw)
        output_path = self.scenario_path.parent / f"{self.scenario_path.stem}_draws.json"
        # assert not os.path.exists(output_path), f'Cannot save run config to {output_path} - file already exists'
        config = generator.get_run_config(self.scenario_path)
        if len(kwargs) > 0:
            for k, v in kwargs.items():
                config[k] = v
        if "commit" in config:
            github_url = f"https://github.com/UCL/TLOmodel/tree/{config['commit']}"
            config["github"] = github_url
        if return_config:
            return config
        else:
            generator.save_config(config, output_path)
            return output_path


class ScenarioLoader:
    """A utility class to load a scenario class from a file path"""
    def __init__(self, scenario_path):
        scenario_module = ScenarioLoader._load_scenario_script(scenario_path)
        scenario_class = ScenarioLoader._get_scenario_class(scenario_module)
        self.scenario = scenario_class()
        self.scenario.scenario_path = scenario_path

    @staticmethod
    def _load_scenario_script(path):
        import importlib.util
        spec = importlib.util.spec_from_file_location(Path(path).stem, path)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        return foo

    @staticmethod
    def _get_scenario_class(scenario_module):
        import inspect
        classes = inspect.getmembers(scenario_module, inspect.isclass)
        classes = [c for (n, c) in classes if BaseScenario == c.__base__]
        assert len(classes) == 1, "Exactly one subclass of BaseScenario should be defined in the scenario script"
        return classes[0]

    def get_scenario(self):
        return self.scenario


class DrawGenerator:
    """Creates and saves a JSON representation of draws from a scenario."""
    def __init__(self, scenario_class, number_of_draws, runs_per_draw):
        self.scenario = scenario_class

        assert self.scenario.seed is not None, "Must set a seed for the scenario. Add `self.seed = <integer>`"
        self.scenario.rng = np.random.RandomState(seed=self.scenario.seed)
        self.number_of_draws = number_of_draws
        self.runs_per_draw = runs_per_draw
        self.draws = self.setup_draws()

    def setup_draws(self):
        assert self.scenario.number_of_draws > 0, "Number of draws must be greater than 0"
        assert self.scenario.runs_per_draw > 0, "Number of samples/draw must be greater than 0"
        if self.scenario.draw_parameters(1, self.scenario.rng) is None:
            assert self.scenario.number_of_draws == 1, "Number of draws should equal one if no variable parameters"
        return [self.get_draw(d) for d in range(0, self.scenario.number_of_draws)]

    def get_draw(self, draw_number):
        return {
            "draw_number": draw_number,
            "parameters": self.scenario.draw_parameters(draw_number, self.scenario.rng),
        }

    def get_run_config(self, scenario_path):
        return {
            "scenario_script_path": str(PurePosixPath(scenario_path)),
            "scenario_seed": self.scenario.seed,
            "arguments": self.scenario.arguments,
            "runs_per_draw": self.runs_per_draw,
            "draws": self.draws,
        }

    @staticmethod
    def save_config(config, output_path):
        with open(output_path, "w") as f:
            f.write(json.dumps(config, indent=2))


class SampleRunner:
    """Reads scenario draws from a JSON configuration and handles running of samples"""
    def __init__(self, run_configuration_path):
        with open(run_configuration_path, "r") as f:
            self.run_config = json.load(f)
        self.scenario = ScenarioLoader(self.run_config["scenario_script_path"]).get_scenario()
        if self.run_config["arguments"] is not None:
            self.scenario.parse_arguments(self.run_config["arguments"])
        logger.info(key="message", data=f"Loaded scenario using {run_configuration_path}")
        logger.info(key="message", data=f"Found {self.number_of_draws} draws; {self.runs_per_draw} runs/draw")

    @property
    def number_of_draws(self):
        return len(self.run_config["draws"])

    @property
    def runs_per_draw(self):
        return self.run_config["runs_per_draw"]

    def get_draw(self, draw_number):
        total = self.number_of_draws
        assert draw_number < total, f"Cannot get draw {draw_number}; only {total} defined."
        return self.run_config["draws"][draw_number]

    def get_samples_for_draw(self, draw):
        for sample_number in range(0, self.run_config["runs_per_draw"]):
            yield self.get_sample(draw, sample_number)

    def get_sample(self, draw, sample_number):
        assert sample_number < self.scenario.runs_per_draw, \
            f"Cannot get sample {sample_number}; samples/draw={self.scenario.runs_per_draw}"
        sample = draw.copy()
        sample["sample_number"] = sample_number

        # Instead of using the random number generator to create a seed for the simulation, we use an integer hash
        # function to get an integer based on the sum of the scenario seed and sample_number. This means the
        # seed can be created independently and out-of-order (i.e. instead of sampling a seed for each sample in order)
        sample["simulation_seed"] = SampleRunner.low_bias_32(self.run_config["scenario_seed"] + sample_number)
        return sample

    def run_sample_by_number(self, output_directory, draw_number, sample_number):
        """Runs a single sample from a draw, saving the output to the given directory"""
        draw = self.get_draw(draw_number)
        sample = self.get_sample(draw, sample_number)
        log_config = self.scenario.get_log_config(output_directory)

        logger.info(key="message", data=f"Running draw {sample['draw_number']}, sample {sample['sample_number']}")

        # if user has specified a restore simulation, we load it from a pickle file
        if self.scenario.resume_simulation is not None:
            sim = Simulation.load_from_pickle(
                pickle_path=Path(self.scenario.get_log_config()["directory"]) / self.scenario.resume_simulation / str(draw_number) / str(sample_number) / "suspended_simulation.pickle",
            )
            sim.configure_logging(**log_config)
        else:
            sim = Simulation(
                start_date=self.scenario.start_date,
                seed=sample["simulation_seed"],
                log_config=log_config
            )
            sim.register(*self.scenario.modules())

            if sample["parameters"] is not None:
                self.override_parameters(sim, sample["parameters"])

            sim.make_initial_population(n=self.scenario.pop_size)

        # if user has specified a suspend date, we run the simulation to that date and save it to a pickle file
        if self.scenario.suspend_date is not None:
            sim.initialise(end_date=self.scenario.end_date)
            sim.run_simulation_to(to_date=self.scenario.suspend_date)
            sim.save_to_pickle(pickle_path=Path(log_config["directory"]) / "suspended_simulation.pickle")
            sim.finalise()
        elif self.scenario.resume_simulation is not None:
            sim.run_simulation_to(to_date=self.scenario.end_date)
            sim.finalise()
        else:
            # we run the simulation as normal
            sim.simulate(end_date=self.scenario.end_date)

        if sim.log_filepath is not None:
            outputs = parse_log_file(sim.log_filepath)
            for key, output in outputs.items():
                if key.startswith("tlo."):
                    with open(Path(log_config["directory"]) / f"{key}.pickle", "wb") as f:
                        pickle.dump(output, f)

    def run(self):
        """Run all samples for the scenario. Used by `tlo scenario-run` to run the scenario locally"""
        log_config = self.scenario.get_log_config()

        root_dir = draw_dir = None
        if log_config["directory"]:  # i.e. write output files
            timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")
            root_dir = Path(log_config["directory"]) / (Path(log_config["filename"]).stem + "-" + timestamp)

        # loop over draws and samples
        for draw in range(0, self.scenario.number_of_draws):
            for sample in range(0, self.runs_per_draw):
                if root_dir is not None:
                    draw_dir = root_dir / f"{draw}/{sample}"
                    draw_dir.mkdir(parents=True, exist_ok=True)
                self.run_sample_by_number(str(draw_dir), draw, sample)

    @staticmethod
    def override_parameters(sim, overridden_params):
        for module_name, overrides in overridden_params.items():
            if module_name in sim.modules:
                module = sim.modules[module_name]
                for param_name, param_val in overrides.items():
                    assert param_name in module.PARAMETERS, f"{module} does not have parameter '{param_name}'"
                    # assert np.isscalar(param_val),
                    #  f"Parameter value '{param_val}' is not scalar type (float, int, str)"

                    old_value = module.parameters[param_name]
                    assert type(old_value) is type(param_val), f"Cannot override parameter '{param_name}' - wrong type"

                    module.parameters[param_name] = param_val
                    logger.info(
                        key="override_parameter",
                        data={
                            "module": module_name,
                            "name": param_name,
                            "old_value": old_value,
                            "new_value": module.parameters[param_name]
                        }
                    )

    @staticmethod
    def low_bias_32(x):
        """A simple integer hash function with uniform distribution. Following description taken from
        https://github.com/skeeto/hash-prospector

            The integer hash function transforms an integer hash key into an integer hash result. For a hash function,
            the distribution should be uniform. This implies when the hash result is used to calculate hash bucket
            address, all buckets are equally likely to be picked. In addition, similar hash keys should be hashed to
            very different hash results. Ideally, a single bit change in the hash key should influence all bits of the
            hash result.

        :param: x an integer
        :returns: an integer
        """
        x *= 0x7feb352d
        x ^= x >> 15
        x *= 0x846ca68b
        x ^= x >> 16
        return x % (2 ** 32)


def _nested_dictionary_from_flat(flat_dict):
    """
    Helper function for transforming a flat dictionary mapping from 2-tuple keys to
    values to a corresponding nested dictionary with outer dictionary keyed by the first
    key in the tuple and inner dictionaries keyed by the second key in the tuple.
    """
    outer_dict = {}
    for (key_1, key_2), value in flat_dict.items():
        inner_dict = outer_dict.setdefault(key_1, {})
        inner_dict[key_2] = value
    return outer_dict


def make_cartesian_parameter_grid(module_parameter_values_dict):
    """Make a list of dictionaries corresponding to a grid across parameter space.

    The parameters values in each parameter dictionary corresponds to an element in the
    Cartesian product of iterables describing the values taken by a collection of
    module specific parameters.

    Intended for use in ``BaseScenario.draw_parameters`` to determine the set of
    parameters for a scenario draw.

    Example usage (in ``BaseScenario.draw_parameters``)::

        return make_cartesian_parameter_grid(
            {
                "Mockitis": {
                    "numeric_parameter": np.linspace(0, 1.0, 5),
                    "categorical_parameter": ["category_a", "category_b"]
                },
                "HealthSystem": {
                    "cons_availability": ["default", "all"]
                }
            }
        )[draw_number]

    In practice it will be more performant to call ``make_cartesian_parameter_grid``
    once in the scenario ``__init__`` method, store this as an attribute of the scenario
    and reuse this in each call to ``draw_parameters``.

    :param dict module_parameter_values_dict: A dictionary mapping from module names
        (as strings) to dictionaries mapping from (string) parameter names associated
        with the model to iterables of the values over which parameter should take in
        the grid.

    :returns: A list of dictionaries mapping from module names (as strings) to
        dictionaries mapping from (string) parameter names associated with the model to
        parameter values, with each dictionary in the list corresponding to a single
        point in the Cartesian grid across the parameter space.
    """
    flattened_parameter_values_dict = {
        (module, parameter): values
        for module, parameter_values_dict in module_parameter_values_dict.items()
        for parameter, values in parameter_values_dict.items()
    }
    return [
        _nested_dictionary_from_flat(
            {
                key: value
                for key, value in zip(flattened_parameter_values_dict, parameter_values)
            }
        )
        for parameter_values in product(*flattened_parameter_values_dict.values())
    ]
