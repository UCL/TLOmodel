"""Creating and running simulation scenarios for TLOmodel

Scenarios are used to specify, configure and run a single or set of TLOmodel simulations. A scenario is created by
by subclassing BaseScenario and specifying the scenario options therein. You can override parameters of the module
in various ways in the scenario. See the BaseScenario class for more information.

The subclass of BaseScenario is then used to create "draws", which can be considered a fully-specified configuration
of the scenario.

Each "draw" is run one or more times, and each run is called a "sample". A sample runs the TLOmodule simulation once.
Each sample of draw has a different seed but is otherwise identical. Because each sample has its own simulation seed,
introducing randomness into each sample's run. A collection of samples run for a given draw describes the random
variation in the simulation.

In summary:

* A _scenario_ specifies the configuration of the TLOmodule simulation. The simulation start and end dates, initial
population size, logging setup and registered modules. Optionally, you can also override parameters of modules.
* A _draw_ is a realisation of a scenario configuration. A scenario can have one or more draws. Draws are uninteresting
unless you are overriding parameters. If you do not override any model parameters, you would only have one draw.
* A _sample_ is the result of running the simulation using a specific configuration. Each draw would run one or more
samples. Each sample for the same draw would have identical configuration except the simulation seed.
"""
import json
from pathlib import Path

import numpy as np

from tlo import logging, Simulation


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_INT = 2**32 - 1


class BaseScenario:
    """An abstract base class for creating Scenarios

    A scenario is a configuration of a TLOmodule simulation. Users should create a subclass of this class and implement
    the following methods:

    * __init__ - to set scenario attributes
    * log_configuration - to configure filename, directory and logging levels for simulation output
    * modules - to list disease, intervention and health system modules for the simulation
    * draw_parameters - override parameters for draws from the scenario
    """
    def __init__(self):
        """Constructor for BaseScenario
        """
        self.seed = None
        self.rng = None
        self.resources = Path('./resources')
        self.number_of_draws = 1
        self.samples_per_draw = 1
        self.scenario_path = None

    def log_configuration(self):
        """Implementations return a dictionary configuring logging. Example:

        return {
            'filename': 'test_scenario',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO
            }
        }
        """
        raise NotImplemented

    def modules(self):
        """Implementations return a list of instances of module to register in the simulation. Example:

        return [
            demography.Demography(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthsystem.HealthSystem(resourcefilepath=self.resources, disable=True, service_availability=['*']),
            ...
        ]
        """
        raise NotImplemented

    def draw_parameters(self, draw_number, rng):
        """Implementations return a dictionary of parameters to override for each draw.

        The overridden parameters must be scalar (i.e. float, integer or string) as the following examples demonstrate.
        The argument `draw_number` and a random number generator are available, if required.

        * Change a parameter to a fixed value: { 'Labour': { 'average_age_at_pregnancy': 25 } }
        * Sample a value from a distribution: {'Lifestyle': { 'init_p_urban': rng.randint(10, 20) / 100.0 } }
        * Set a value based on the draw number: { 'Labour': { 'average_age_at_pregnancy': [25, 30, 35][draw_number] } }

        Implementing this method in a subclass is optional. If no parameters are to be overridden, returns None. If no
        parameters are overridden, only one draw of the scenario is required.

        A full example for a scenario with 10 draws:

        return {
            'Lifestyle': {
                'init_p_urban': rng.randint(10, 20) / 100.0,
            },
            'Labour': {
                'average_age_at_pregnancy': -10 * rng.exponential(0.1),
                'some_other_parameter': np.arange(0.1, 1.1, 0.1)[draw_number]
            },
        }
        """
        return None

    def save_draws(self):
        generator = DrawGenerator(self, self.number_of_draws, self.samples_per_draw)
        output_path = self.scenario_path.parent / 'run.json'
        # assert not os.path.exists(output_path), f'Cannot save run config to {output_path} - file already exists'
        generator.save_config(str(self.scenario_path), output_path)


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
        spec = importlib.util.spec_from_file_location("scenario_definition", path)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        return foo

    @staticmethod
    def _get_scenario_class(scenario_module):
        import inspect
        classes = inspect.getmembers(scenario_module, inspect.isclass)
        classes = [c for (n, c) in classes if BaseScenario == c.__base__]
        assert len(classes) == 1, 'Exactly one subclass of BaseScenario should be defined in the scenario script'
        return classes[0]

    def get_scenario(self):
        return self.scenario


class DrawGenerator:
    """Creates and saves a JSON representation of draws from a scenario."""
    def __init__(self, scenario_class, number_of_draws, samples_per_draw):
        self.scenario = scenario_class

        assert self.scenario.seed is not None, 'Must set a seed for the scenario. Add `self.seed = <integer>`'
        self.scenario.rng = np.random.RandomState(seed=self.scenario.seed)
        self.number_of_draws = number_of_draws
        self.samples_per_draw = samples_per_draw
        self.draws = self.setup_draws()

    def setup_draws(self):
        assert self.scenario.number_of_draws > 0, "Number of draws must be greater than one"
        assert self.scenario.samples_per_draw > 0, "Number of samples/draw must be greater than 0"
        if self.scenario.draw_parameters(1) is None:
            assert self.scenario.number_of_draws == 1, "Number of draws should equal one if no variable parameters"
        return [self.get_draw(d) for d in range(0, self.scenario.number_of_draws)]

    def get_draw(self, draw_number):
        return {
            'draw_number': draw_number,
            'draw_seed': self.scenario.rng.randint(MAX_INT),
            'parameters': self.scenario.draw_parameters(draw_number, self.scenario.rng),
        }

    def get_run_config(self, scenario_path):
        return {
            'scenario_script_path': scenario_path,
            'scenario_seed': self.scenario.seed,
            'samples_per_draw': self.samples_per_draw,
            'draws': self.draws,
        }

    def save_config(self, scenario_path, output_path):
        with open(output_path, 'w') as f:
            f.write(json.dumps(self.get_run_config(scenario_path), indent=2))


class SampleRunner:
    """Reads scenario draws from a JSON configuration and handles running of samples"""
    def __init__(self, run_configuration_path):
        with open(run_configuration_path, 'r') as f:
            self.run_config = json.load(f)
        self.scenario = ScenarioLoader(self.run_config['scenario_script_path']).get_scenario()

    @property
    def number_of_draws(self):
        return len(self.run_config['draws'])

    @property
    def samples_per_draw(self):
        return self.run_config['samples_per_draw']

    def get_draw(self, draw_number):
        total = self.number_of_draws
        assert draw_number < total, f"Cannot get draw {draw_number}; only {total} defined."
        return self.run_config['draws'][draw_number]

    def get_samples_for_draw(self, draw):
        for sample_number in range(0, self.run_config['samples_per_draw']):
            yield self.get_sample(draw, sample_number)

    def get_sample(self, draw, sample_number):
        assert sample_number < self.scenario.samples_per_draw, \
            f"Cannot get sample {sample_number}; samples/draw={self.scenario.samples_per_draw}"
        sample = draw.copy()
        sample['sample_number'] = sample_number

        # Instead of using the random number generator to create a seed for the simulation, we use an integer hash
        # function to create an integer based on the sum of the draw_number and sample_number. This means the
        # seed can be created independently and out-of-order (i.e. instead of sampling a seed for each sample in order)
        sample['simulation_seed'] = SampleRunner.low_bias_32(sample['draw_seed'] + sample_number)
        return sample

    def run_sample_by_number(self, draw_number, sample_number):
        draw = self.get_draw(draw_number)
        sample = self.get_sample(draw, sample_number)
        self.run_sample(sample)

    def run_sample(self, sample):
        log_config = self.scenario.log_configuration()
        log_config['filename'] = f"{log_config['filename']}_draw{sample['draw_number']}_sample{sample['sample_number']}"

        sim = Simulation(
            start_date=self.scenario.start_date,
            seed=sample['simulation_seed'],
            log_config=log_config
        )
        sim.register(*self.scenario.modules())

        if sample['parameters'] is not None:
            self.override_parameters(sim, sample['parameters'])

        sim.make_initial_population(n=self.scenario.pop_size)
        sim.simulate(end_date=self.scenario.end_date)

    @staticmethod
    def override_parameters(sim, overridden_params):
        for module_name, overrides in overridden_params.items():
            if module_name in sim.modules:
                module = sim.modules[module_name]
                for param_name, param_val in overrides.items():
                    assert param_name in module.PARAMETERS, f"{module} does not have parameter '{param_name}'"
                    assert np.isscalar(param_val), f"Parameter value '{param_val}' is not scalar type (float, int, str)"

                    old_value = module.parameters[param_name]
                    assert type(old_value) == type(param_val), f"Cannot override parameter '{param_name}' - wrong type"

                    module.parameters[param_name] = param_val
                    logger.info(
                        key="override_parameter",
                        data={
                            'module': module_name,
                            'name': param_name,
                            'old_value': old_value,
                            'new_value': module.parameters[param_name]
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
        x ^= (x >> 16) * 0x7feb352d
        x ^= (x >> 15) * 0x846ca68b
        x ^= (x >> 16)
        return x % (2 ** 32)
