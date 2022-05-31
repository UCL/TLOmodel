"""Creating and running simulation scenarios for TLOmodel

Scenarios are used to specify, configure and run a single or set of TLOmodel simulations. A scenario is created by
by subclassing BaseScenario and specifying the scenario options therein. You can override parameters of the module
in various ways in the scenario. See the BaseScenario class for more information.

The subclass of BaseScenario is then used to create *draws*, which can be considered a fully-specified configuration
of the scenario, or a parameter draw.

Each draw is *run* one or more times - run is a single execution of the simulation. Each run of a draw has
a different seed but is otherwise identical. Each run has its own simulation seed, introducing randomness into each
simulation. A collection of runs for a given draw describes the random variation in the simulation.

A simple example of a subclass of BaseScenario::

    class MyTestScenario(BaseScenario):
        def __init__(self):
            super().__init__()
            self.seed = 12
            self.start_date = Date(2010, 1, 1)
            self.end_date = Date(2011, 1, 1)
            self.pop_size = 200
            self.number_of_draws = 2
            self.runs_per_draw = 2

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

* A *scenario* specifies the configuration of the simulation. The simulation start and end dates, initial population
  size, logging setup and registered modules. Optionally, you can also override parameters of modules.
* A *draw* is a realisation of a scenario configuration. A scenario can have one or more draws. Draws are uninteresting
  unless you are overriding parameters. If you do not override any model parameters, you would only have one draw.
* A *run* is the result of running the simulation using a specific configuration. Each draw would run one or more
  times. Each run for the same draw would have identical configuration except the simulation seed.
"""
import datetime
import json
import pickle
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from tlo import Simulation, logging
from tlo.analysis.utils import parse_log_file

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_INT = 2**31 - 1


class BaseScenario:
    """An abstract base class for creating Scenarios

    A scenario is a configuration of a simulation. Users should subclass this class and implement the following methods:

    * ``__init__`` - to set scenario attributes
    * ``log_configuration`` - to configure filename, directory and logging levels for simulation output
    * ``modules`` - to list disease, intervention and health system modules for the simulation
    * ``draw_parameters`` - override parameters for draws from the scenario
    """
    def __init__(self):
        """Constructor for BaseScenario
        """
        self.seed = None
        self.rng = None
        self.resources = Path("./resources")
        self.number_of_draws = 1
        self.runs_per_draw = 1
        self.scenario_path = None

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
        raise NotImplementedError

    def modules(self):
        """Implementation must return a list of instances of TLOmodel modules to register in the simulation.

        Example::

            return [
                demography.Demography(resourcefilepath=self.resources),
                enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
                healthsystem.HealthSystem(resourcefilepath=self.resources, disable=True, service_availability=['*']),
                ...
            ]
        """
        raise NotImplementedError

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

    def save_draws(self, **kwargs):
        generator = DrawGenerator(self, self.number_of_draws, self.runs_per_draw)
        output_path = self.scenario_path.parent / f"{self.scenario_path.stem}_draws.json"
        # assert not os.path.exists(output_path), f'Cannot save run config to {output_path} - file already exists'
        config = generator.get_run_config(self.scenario_path)
        if len(kwargs) > 0:
            for k, v in kwargs.items():
                config[k] = v
        if "commit" in config:
            github_url = f"https://github.com/UCL/TLOmodel/blob/{config['commit']}/{config['scenario_script_path']}"
            config["github"] = github_url
        generator.save_config(config, output_path)
        return output_path

    def make_grid(self, ranges: dict) -> pd.DataFrame:
        """Utility method to flatten an n-dimension grid of parameters for use in scenarios

        Typically used in draw_parameters determining a set of parameters for a draw. This function will check that the
        number of draws of the scenario is equal to the number of coordinates in the grid.

        Parameter ``ranges`` is a dictionary of { string key: iterable }, where iterable can be, for example, an
        np.array or list. The function will return a DataFrame where each key is a column and each row represents a
        single coordinate in the grid.

        Usage (in ``draw_parameters``)::

            grid = self.make_grid({'p_one': np.linspace(0, 1.0, 5), 'p_two': np.linspace(3.0, 4.0, 2)})
            return {
                'Mockitis': {
                    grid['p_one'][draw_number],
                    grid['p_two'][draw_number]
                }
            }

        :param dict ranges: each item of dict represents points across a single dimension
        """
        grid = np.meshgrid(*ranges.values())
        flattened = [g.ravel() for g in grid]
        positions = np.stack(flattened, axis=1)
        grid_lookup = pd.DataFrame(positions, columns=ranges.keys())
        assert self.number_of_draws == len(grid_lookup), f"{len(grid_lookup)} coordinates in grid, " \
                                                         f"but number_of_draws is {self.number_of_draws}."
        return grid_lookup


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
        assert self.scenario.number_of_draws > 0, "Number of draws must be greater than one"
        assert self.scenario.runs_per_draw > 0, "Number of samples/draw must be greater than 0"
        if self.scenario.draw_parameters(1, self.scenario.rng) is None:
            assert self.scenario.number_of_draws == 1, "Number of draws should equal one if no variable parameters"
        return [self.get_draw(d) for d in range(0, self.scenario.number_of_draws)]

    def get_draw(self, draw_number):
        return {
            "draw_number": draw_number,
            "draw_seed": self.scenario.rng.randint(MAX_INT),
            "parameters": self.scenario.draw_parameters(draw_number, self.scenario.rng),
        }

    def get_run_config(self, scenario_path):
        return {
            "scenario_script_path": str(PurePosixPath(scenario_path)),
            "scenario_seed": self.scenario.seed,
            "runs_per_draw": self.runs_per_draw,
            "draws": self.draws,
        }

    def save_config(self, config, output_path):
        with open(output_path, "w") as f:
            f.write(json.dumps(config, indent=2))


class SampleRunner:
    """Reads scenario draws from a JSON configuration and handles running of samples"""
    def __init__(self, run_configuration_path):
        with open(run_configuration_path, "r") as f:
            self.run_config = json.load(f)
        self.scenario = ScenarioLoader(self.run_config["scenario_script_path"]).get_scenario()
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
        # function to create an integer based on the sum of the draw_number and sample_number. This means the
        # seed can be created independently and out-of-order (i.e. instead of sampling a seed for each sample in order)
        sample["simulation_seed"] = SampleRunner.low_bias_32(sample["draw_seed"] + sample_number)
        return sample

    def run_sample_by_number(self, output_directory, draw_number, sample_number):
        draw = self.get_draw(draw_number)
        sample = self.get_sample(draw, sample_number)
        self.run_sample(sample, output_directory)

    def run_sample(self, sample, output_directory=None):
        log_config = self.scenario.log_configuration()

        if output_directory is not None:
            log_config["directory"] = output_directory
            # suppress stdout when saving output to directory (either user specified, or set by batch-run process)
            log_config["suppress_stdout"] = True

        sim = Simulation(
            start_date=self.scenario.start_date,
            seed=sample["simulation_seed"],
            log_config=log_config
        )
        sim.register(*self.scenario.modules())

        if sample["parameters"] is not None:
            self.override_parameters(sim, sample["parameters"])

        sim.make_initial_population(n=self.scenario.pop_size)
        sim.simulate(end_date=self.scenario.end_date)
        outputs = parse_log_file(sim.log_filepath)
        for key, output in outputs.items():
            if key.startswith("tlo."):
                with open(Path(log_config["directory"]) / f"{key}.pickle", "wb") as f:
                    pickle.dump(output, f)

    def run(self):
        # this method will execute all runs of each draw, so we save output in directory
        log_config = self.scenario.log_configuration()
        root_dir = draw_dir = None
        if log_config["filename"]:  # i.e. save output?
            timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")
            root_dir = Path(log_config["directory"]) / (Path(log_config["filename"]).stem + "-" + timestamp)

        for draw in range(0, self.scenario.number_of_draws):
            for sample in range(0, self.runs_per_draw):
                if root_dir is not None:
                    draw_dir = root_dir / f"{draw}/{sample}"
                    draw_dir.mkdir(parents=True, exist_ok=True)
                self.run_sample_by_number(draw_dir, draw, sample)

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
                    assert type(old_value) == type(param_val), f"Cannot override parameter '{param_name}' - wrong type"

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
