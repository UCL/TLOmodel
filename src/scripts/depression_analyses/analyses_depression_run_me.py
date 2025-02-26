from pathlib import Path

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    care_of_women_during_pregnancy,
    contraception,
    demography,
    depression,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager,
    tb,
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
        "filename": "depression_analyses",  # The prefix for the output file. A timestamp will be added to this.
    }
    # For default configuration, uncomment the next line
    # log_config = dict()

    # Basic arguments required for the simulation
    start_date = Date(2010, 1, 1)
    end_date = Date(2014, 7, 1)
    pop_size = 10000


    # Path to the resource files used by the disease and intervention methods
    resourcefilepath = Path("./resources")

    # This creates the Simulation instance for this run. Because we"ve passed the `seed` and
    # `log_config` arguments, these will override the default behaviour.
    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config, resourcefilepath=resourcefilepath)

    # We register all modules in a single call to the register method, calling once with multiple
    # objects. This is preferred to registering each module in multiple calls because we will be
    # able to handle dependencies if modules are registered together
    # Register the appropriate modules

    sim.register(
        care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(),
        demography.Demography(),
        enhanced_lifestyle.Lifestyle(),
        healthsystem.HealthSystem(disable=True),
        symptommanager.SymptomManager(),
        healthseekingbehaviour.HealthSeekingBehaviour(),
        healthburden.HealthBurden(),
        contraception.Contraception(),
        labour.Labour(),
        newborn_outcomes.NewbornOutcomes(),
        pregnancy_supervisor.PregnancySupervisor(),
        postnatal_supervisor.PostnatalSupervisor(),
        depression.Depression(),
        hiv.Hiv(),
        tb.Tb(),
        epi.Epi()
    )

    sim.make_initial_population(n=pop_size)
    sim.simulate(end_date=end_date)
    return sim


# Run the simulation
sim = run()

output = parse_log_file(sim.log_filepath)
