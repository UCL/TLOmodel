"""
An analysis file for the wasting module
"""
from pathlib import Path

# %% Import statements

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import wasting, demography, healthsystem, symptommanager, healthseekingbehaviour, healthburden, \
    enhanced_lifestyle, labour, newborn_outcomes, care_of_women_during_pregnancy, contraception, pregnancy_supervisor, \
    postnatal_supervisor, hiv

seed = 1

# Path to the resource files used by the disease and intervention methods
resources = Path("./resources")

# configure logging
log_config = {
    "filename": "wasting",  # output filename. A timestamp will be added to this.
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "tlo.methods.demography": logging.INFO,
        "tlo.methods.wasting": logging.INFO,
        '*': logging.WARNING
    }
}

# Basic arguments required for the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2020, 1, 1)
pop_size = 10000

# Create simulation instance for this run.
sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

# Register modules for simulation
sim.register(
    demography.Demography(resourcefilepath=resources),
    healthsystem.HealthSystem(resourcefilepath=resources,
                              service_availability=['*'],
                              cons_availability='default'),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resources),
    healthburden.HealthBurden(resourcefilepath=resources),
    symptommanager.SymptomManager(resourcefilepath=resources),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resources),
    labour.Labour(resourcefilepath=resources),
    care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resources),
    contraception.Contraception(resourcefilepath=resources),
    pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resources),
    postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resources),
    newborn_outcomes.NewbornOutcomes(resourcefilepath=resources),
    hiv.Hiv(resourcefilepath=resources),
    wasting.Wasting(resourcefilepath=resources),
)

sim.make_initial_population(n=pop_size)
sim.simulate(end_date=end_date)

# %% read the results
output = parse_log_file(sim.log_filepath)
print(output)
