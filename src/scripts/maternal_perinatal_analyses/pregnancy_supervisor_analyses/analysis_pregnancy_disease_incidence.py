from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd
from tlo import Date, Simulation, logging
from tlo.analysis.utils import (
    parse_log_file,
)
from tlo.methods import demography, contraception, labour, enhanced_lifestyle, newborn_outcomes, healthsystem, \
    pregnancy_supervisor, antenatal_care, symptommanager, healthseekingbehaviour


seed = 567

log_config = {
    "filename": "pregnancy_incidence_analysis",   # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.labour": logging.DEBUG,
        "tlo.methods.healthsystem": logging.FATAL,
        "tlo.methods.hiv": logging.FATAL,
        "tlo.methods.newborn_outcomes": logging.DEBUG,
        "tlo.methods.antenatal_care": logging.DEBUG,
        "tlo.methods.pregnancy_supervisor": logging.DEBUG,
        "tlo.methods.postnatal_supervisor": logging.DEBUG,
    }
}

# %%
resourcefilepath = Path("./resources")


# %% Run the Simulation
start_date = Date(2010, 1, 1)
end_date = Date(2013, 1, 1)
popsize = 10000

sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

# run the simulation
sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             contraception.Contraception(resourcefilepath=resourcefilepath),
             enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
             healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
             symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
             healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
             # healthburden.HealthBurden(resourcefilepath=resourcefilepath),
             labour.Labour(resourcefilepath=resourcefilepath),
             newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
             pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
             antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath))

sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# Get the output from the logfile
log_df = parse_log_file(sim.log_filepath)

stats = log_df['tlo.methods.pregnancy_supervisor']['summary_stats']
stats['date'] = pd.to_datetime(stats['date'])
stats['year'] = stats['date'].dt.year
stats['year'] = stats['year'] - 1
stats.set_index("year", inplace=True)

stats.plot.bar(y='antenatal_mmr', stacked=True)
plt.title("Yearly Antenatal Maternal Mortality Rate")
plt.show()

stats.plot.bar(y='antenatal_sbr', stacked=True)
plt.title("Yearly Antenatal Still Birth Rate")
plt.show()


stats.plot.bar(y='spontaneous_abortion_rate', stacked=True)
plt.title("Yearly spontaneous_abortion_rate Rate")
plt.show()

stats.plot.bar(y='induced_abortion_rate', stacked=True)
plt.title("Yearly induced_abortion_rat Rate")
plt.show()

stats.plot.bar(y='ectopic_rate', stacked=True)
plt.title("Yearly ectopic_rate Rate")
plt.show()

stats.plot.bar(y='anaemia_rate', stacked=True)
plt.title("Yearly anaemia_rate Rate")
plt.show()

stats = log_df['tlo.methods.pregnancy_supervisor']['summary_stats']
