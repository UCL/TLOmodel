# %% Import Statements
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pathlib import Path
from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import demography, enhanced_lifestyle, simplified_births

# Where will outputs go - by default, wherever this script is run
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")


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
        "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
            "tlo.methods.demography": logging.WARNING,
            "tlo.methods.enhanced_lifestyle": logging.INFO
        }
    }
    # For default configuration, uncomment the next line
    # log_config = dict()

    # Basic arguments required for the simulation
    start_date = Date(2010, 1, 1)
    end_date = Date(2040, 1, 1)
    pop_size = 20000

    # This creates the Simulation instance for this run. Because we"ve passed the `seed` and
    # `log_config` arguments, these will override the default behaviour.
    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

    # Path to the resource files used by the disease and intervention methods
    resources = "./resources"

    # We register all modules in a single call to the register method, calling once with multiple
    # objects. This is preferred to registering each module in multiple calls because we will be
    # able to handle dependencies if modules are registered together
    sim.register(
        demography.Demography(resourcefilepath=resources),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resources),
        simplified_births.SimplifiedBirths(resourcefilepath=resources),
    )

    sim.make_initial_population(n=pop_size)
    sim.simulate(end_date=end_date)
    return sim


# %% Run the Simulation
sim = run()

# %% read the results
output = parse_log_file(sim.log_filepath)


def extract_formatted_series(df):
    return pd.Series(index=pd.to_datetime(df['date']), data=df.iloc[:, 1].values)


# Examine Urban Rural Population
urban_rural_pop = output['tlo.methods.enhanced_lifestyle']['urban_rural_pop'].set_index('date')
model_years = pd.to_datetime(urban_rural_pop.index)
total_pop = urban_rural_pop.sum(axis=1)
pop_urban = urban_rural_pop.true
pop_rural = urban_rural_pop.false

fig, ax = plt.subplots()
ax.plot(np.asarray(model_years), total_pop)
ax.plot(np.asarray(model_years), pop_urban)
ax.plot(np.asarray(model_years), pop_rural)

plt.title("Population Distribution(Urban and Rural")
plt.xlabel("Year")
plt.ylabel("Number of individuals")
plt.legend(['Total Population', 'Urban Population', 'Rural Population'])
plt.savefig(outputpath / ('Population Distribution' + datestamp + '.png'), format='png')
plt.show()

# Examine tobacco use between males and females
tob_use = output['tlo.methods.enhanced_lifestyle']['tobacco_use'].set_index('date')
tob_model_years = pd.to_datetime(urban_rural_pop.index)
tob_use_total = tob_use.sum(axis=1)
tob_use_male = tob_use.M
tob_use_female = tob_use.F

tob_fig, ax = plt.subplots()
ax.plot(np.asarray(tob_model_years), tob_use_total)
ax.plot(np.asarray(tob_model_years), tob_use_male)
ax.plot(np.asarray(tob_model_years), tob_use_female)

plt.title('Tobacco use by Gender')
plt.xlabel("Year")
plt.ylabel("Number of individuals")
plt.legend(['Tobacco use total', 'Tobacco use males', 'Tobacco use females'])
plt.savefig(outputpath / ('tobacco use' + datestamp + '.png'), format='png')
plt.show()

# Examine tobacco use by age range
tob_use_by_age_range = output['tlo.methods.enhanced_lifestyle']['tobacco_use_age_range'].set_index('date')
tob_age_model_years = pd.to_datetime(urban_rural_pop.index)
tob_use_1519 = tob_use_by_age_range.tob1519
tob_use_2039 = tob_use_by_age_range.tob2039
tob_use_40 = tob_use_by_age_range.tob40

tob_age_fig, ax = plt.subplots()
ax.plot(np.asarray(tob_model_years), tob_use_1519)
ax.plot(np.asarray(tob_model_years), tob_use_2039)
ax.plot(np.asarray(tob_model_years), tob_use_40)

plt.title('Tobacco use by Age')
plt.xlabel("Year")
plt.ylabel("Number of individuals")
plt.legend(['Tobacco use age15-19', 'Tobacco use age20-39', 'Tobacco use age40+'])
plt.savefig(outputpath / ('tobacco use by age' + datestamp + '.png'), format='png')
plt.show()

# Examine Proportion Men Circumcised:
circ = extract_formatted_series(output['tlo.methods.enhanced_lifestyle']['prop_adult_men_circumcised'])
circ.plot()
plt.title('Proportion of Adult Men Circumcised')
plt.ylim(0, 0.30)
plt.show()

# Examine Proportion Women sex Worker:
fsw = extract_formatted_series(output['tlo.methods.enhanced_lifestyle']['proportion_1549_women_sexworker'])
fsw.plot()
plt.title('Proportion of 15-49 Women Sex Workers')
plt.ylim(0, 0.01)
plt.show()
