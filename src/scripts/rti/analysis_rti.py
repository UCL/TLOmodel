from pathlib import Path

from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    rti,
    simplified_births,
    symptommanager,
)

# To reproduce the results, you must set the seed for the Simulation instance. The Simulation
# will seed the random number generators for each module when they are registered.
# If a seed argument is not given, one is generated. It is output in the log and can be
# used to reproduce results of a run
seed = 100

log_config = {
    "filename": "rti_analysis",   # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.rti": logging.INFO,
        "tlo.methods.healthsystem": logging.INFO,
    }
}

start_date = Date(2010, 1, 1)
end_date = Date(2012, 12, 31)
pop_size = 5000

# This creates the Simulation instance for this run. Because we've passed the `seed` and
# `log_config` arguments, these will override the default behaviour.
sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

# Path to the resource files used by the disease and intervention methods
# resources = "./resources"
resourcefilepath = Path('./resources')

# We register all modules in a single call to the register method, calling once with multiple
# objects. This is preferred to registering each module in multiple calls because we will be
# able to handle dependencies if modules are registered together
sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        rti.RTI(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath)
        )

# create and run the simulation
sim.make_initial_population(n=pop_size)
sim.modules['RTI'].parameters['number_of_injured_body_regions_distribution'] = \
    [[1, 2, 3, 4, 5, 6, 7, 8],
     [0.7093834795440467, 0.2061841889568684, 0.05992798112965075, 0.017418226588785873,
      0.00506265373502088, 0.0014714737295484586, 0.00042768774047753785, 0.00012430857560121887]]
sim.modules['RTI'].parameters['base_rate_injrti'] = 0.004334746692121514
sim.modules['RTI'].parameters['imm_death_proportion_rti'] = 0.007
sim.modules['RTI'].parameters['prob_death_iss_less_than_9'] = 0.97760263 * (102 / 11650)
sim.modules['RTI'].parameters['prob_death_iss_10_15'] = 0.97760263 * (7 / 528)
sim.modules['RTI'].parameters['prob_death_iss_16_24'] = 0.97760263 * (37 / 988)
sim.modules['RTI'].parameters['prob_death_iss_25_35'] = 0.97760263 * (52 / 325)
sim.modules['RTI'].parameters['prob_death_iss_35_plus'] = 0.97760263 * (37 / 136)
sim.modules['RTI'].parameters['rt_emergency_care_ISS_score_cut_off'] = 2
sim.simulate(end_date=end_date)

# parse the simulation logfile to get the output dataframes
log_df = parse_log_file(sim.log_filepath)

# ------------------------------------- MODEL OUTPUTS  ------------------------------------- #

model_rti = log_df["tlo.methods.rti"]["summary_1m"]["incidence of rti per 100,000"]
model_date = log_df["tlo.methods.rti"]["summary_1m"]["date"]
# ------------------------------------- PLOTS  ------------------------------------- #

plt.style.use("ggplot")

# Measles incidence
plt.subplot(111)  # numrows, numcols, fignum
plt.plot(model_date, model_rti)
plt.title("RTI incidence")
plt.xlabel("Date")
plt.ylabel("Incidence per 100,000 person years")
plt.xticks(rotation=90)
plt.legend(["Model"], bbox_to_anchor=(1.04, 1), loc="upper left")

plt.show()
