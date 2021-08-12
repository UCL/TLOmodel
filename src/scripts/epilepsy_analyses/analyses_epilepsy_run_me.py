from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    contraception,
    demography,
    epilepsy,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    pregnancy_supervisor,
    symptommanager,
    care_of_women_during_pregnancy,
    newborn_outcomes,
    postnatal_supervisor
)


def run():
    # To reproduce the results, you need to set the seed for the Simulation instance. The Simulation
    # will seed the random number generators for each module when they are registered.
    # If a seed argument is not given, one is generated. It is output in the log and can be
    # used to reproduce results of a run
    seed = 0

    # By default, all output is recorded at the "INFO" level (and up) to standard out. You can
    # configure the behaviour by passing options to the `log_config` argument of
    # Simulation.
    log_config = {
        "filename": "epilepsy_analyses",  # The prefix for the output file. A timestamp will be added to this.
    }
    # For default configuration, uncomment the next line
    # log_config = dict()

    # Basic arguments required for the simulation
    start_date = Date(2010, 1, 1)
    end_date = Date(2011, 1, 2)
    pop_size = 300000

    # This creates the Simulation instance for this run. Because we"ve passed the `seed` and
    # `log_config` arguments, these will override the default behaviour.
    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

    # Path to the resource files used by the disease and intervention methods
    resources = "./resources"

    # We register all modules in a single call to the register method, calling once with multiple
    # objects. This is preferred to registering each module in multiple calls because we will be
    # able to handle dependencies if modules are registered together
    # Register the appropriate modules

    sim.register(
        demography.Demography(resourcefilepath=resources),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resources),
        healthsystem.HealthSystem(resourcefilepath=resources, disable=True),
        symptommanager.SymptomManager(resourcefilepath=resources),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resources),
        healthburden.HealthBurden(resourcefilepath=resources),
        contraception.Contraception(resourcefilepath=resources),
        labour.Labour(resourcefilepath=resources),
        pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resources),
        epilepsy.Epilepsy(resourcefilepath=resources),
        care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resources),
        newborn_outcomes.NewbornOutcomes(resourcefilepath=resources),
        postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resources)
    )

    sim.make_initial_population(n=pop_size)
    sim.simulate(end_date=end_date)
    return sim


# Run the simulation
sim = run()

output = parse_log_file(sim.log_filepath)

comparison = compare_number_of_deaths(logfile=sim.log_filepath, resourcefilepath=resourcefilepath)
# Make a simple bar chart
comparison.loc[('2015-2019', slice(None), '0-4', 'MyCauseName')].sum().plot.bar()
plt.title('Deaths per year due to MyCauseName, 2015-2019')
plt.tight_layout()
plt.show()
