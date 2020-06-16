# %% Import Statements
from pathlib import Path

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    contraception,
    demography,
    enhanced_lifestyle,
    healthseekingbehaviour,
    healthsystem,
    labour,
    pregnancy_supervisor,
    symptommanager,
)


def run():
    # To reproduce the results, you need to set the seed for the Simulation instance. The Simulation
    # will seed the random number generators for each module when they are registered.
    # If a seed argument is not given, one is generated. It is output in the log and can be
    # used to reproduce results of a run
    seed = 1

    # By default, all output is recorded at the "INFO" level (and up) to standard out. You can
    # configure the behaviour by passing options to the `log_config` argument of
    # Simulation.
    log_config = {
        "filename": "enhanced_lifestyle",  # The prefix for the output file. A timestamp will be added to this.
    }
    # For default configuration, uncomment the next line
    # log_config = dict()

    # Basic arguments required for the simulation
    start_date = Date(2010, 1, 1)
    end_date = Date(2050, 1, 1)
    pop_size = 1000

    # This creates the Simulation instance for this run. Because we"ve passed the `seed` and
    # `log_config` arguments, these will override the default behaviour.
    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

    # Path to the resource files used by the disease and intervention methods
    resources = "./resources"

    # Used to configure health system behaviour
    service_availability = ["*"]

    # We register all modules in a single call to the register method, calling once with multiple
    # objects. This is preferred to registering each module in multiple calls because we will be
    # able to handle dependencies if modules are registered together
    sim.register(
        demography.Demography(resourcefilepath=resources),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resources),
        healthsystem.HealthSystem(resourcefilepath=resources, disable=True),
        symptommanager.SymptomManager(resourcefilepath=resources),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resources),
        contraception.Contraception(resourcefilepath=resources),
        labour.Labour(resourcefilepath=resources),
        pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resources),
    )

    sim.make_initial_population(n=pop_size)
    sim.simulate(end_date=end_date)
    return sim


# %% Run the Simulation
sim = run()

# %% read the results
output = parse_log_file(sim.log_filepath)

# TODO: remove me before mering
import pickle
output_path = Path("./outputs")
with open(output_path / f"lifestyle_analysis_structured_logs.pickle", "wb") as handle:
    pickle.dump(output, handle)

structured_log = output

with open(output_path / f"lifestyle_analysis_parsed_logs.pickle", "rb") as handle:
    stdlib_log = pickle.load(handle)

for key in stdlib_log['tlo.methods.enhanced_lifestyle']:
    # round floats to the same precision (stdlib has 6 decimals, structured has 7)
    # convert all types to strings, ad this is what happens with stdlib logging
    stdlib_table = stdlib_log['tlo.methods.enhanced_lifestyle'][key].round(6).astype('object')
    structured_table = structured_log['tlo.methods.enhanced_lifestyle'][key].round(6).astype('object')

    # Set both dates to be the correct type (when both as strings, they don't match)
    stdlib_table.date = stdlib_table.date.astype('datetime64[ns]')
    structured_table.date = structured_table.date.astype('datetime64[ns]')

    tables_match = stdlib_table.equals(structured_table)
    print(f"Table {key} matches before and after adding structured logging: {tables_match}")
    assert tables_match
###
