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


def examine_urban_population(log_output):
    """Examine the distribution of individuals in both urban and rural areas. we are considering individuals of all
    ages """

    # get urban and rural population distribution from log file. make date an index
    urban_rural_pop = log_output['tlo.methods.enhanced_lifestyle']['urban_rural_pop'].set_index('date')
    model_years = pd.to_datetime(urban_rural_pop.index)
    # get the total population
    total_pop = urban_rural_pop.sum(axis=1)
    # get individuals who are urban
    pop_urban = urban_rural_pop.true
    # get individuals who are rural
    pop_rural = urban_rural_pop.false

    # add data to plot
    fig, ax = plt.subplots()
    ax.plot(np.asarray(model_years), total_pop)
    ax.plot(np.asarray(model_years), pop_urban)
    ax.plot(np.asarray(model_years), pop_rural)

    # display population distribution plot
    plt.title("Population Distribution(Urban and Rural")
    plt.xlabel("Year")
    plt.ylabel("Number of individuals")
    plt.legend(['Total Population', 'Urban Population', 'Rural Population'])
    plt.savefig(outputpath / ('Population Distribution' + datestamp + '.png'), format='png')
    plt.show()


def tobacco_use_male_female(log_output):
    """Examine tobacco use by gender, in this case males and females. Here we are considering individuals of 15 and
    above years old """
    # get tobacco use data males and females from log file
    tob_use = log_output['tlo.methods.enhanced_lifestyle']['tobacco_use'].set_index('date')
    tob_model_years = pd.to_datetime(tob_use.index)
    # total tobacco use
    tob_use_total = tob_use.sum(axis=1)
    # tobacco use in males
    tob_use_male = tob_use.M
    # tobacco use in females
    tob_use_female = tob_use.F

    # add data to plot
    tob_fig, ax = plt.subplots()
    ax.plot(np.asarray(tob_model_years), tob_use_total)
    ax.plot(np.asarray(tob_model_years), tob_use_male)
    ax.plot(np.asarray(tob_model_years), tob_use_female)

    # display tobacco use by gender chat
    plt.title('Tobacco use by gender')
    plt.xlabel("Year")
    plt.ylabel("Number of individuals")
    plt.legend(['Tobacco use total', 'Tobacco use males', 'Tobacco use females'])
    plt.savefig(outputpath / ('tobacco use' + datestamp + '.png'), format='png')
    plt.show()


def tobacco_use_by_age(log_output):
    """Examine tobacco use by different age range. we currently have three ranges namely; 15-19 years, 20-39 years
    and 40+ years """
    # get tobacco use by age from log file
    tob_use_by_age_range = log_output['tlo.methods.enhanced_lifestyle']['tobacco_use_age_range'].set_index('date')
    tob_age_model_years = pd.to_datetime(tob_use_by_age_range.index)
    # tobacco use in age range 15-19
    tob_use_1519 = tob_use_by_age_range.tob1519
    # tobacco use in age range 20-39
    tob_use_2039 = tob_use_by_age_range.tob2039
    # tobacco use in age range 40+
    tob_use_40 = tob_use_by_age_range.tob40

    # add data to plot
    tob_age_fig, ax = plt.subplots()
    ax.plot(np.asarray(tob_age_model_years), tob_use_1519)
    ax.plot(np.asarray(tob_age_model_years), tob_use_2039)
    ax.plot(np.asarray(tob_age_model_years), tob_use_40)

    # display tobacco use by age plot
    plt.title('Tobacco use by age')
    plt.xlabel("Year")
    plt.ylabel("Number of individuals")
    plt.legend(['Tobacco use age15-19', 'Tobacco use age20-39', 'Tobacco use age40+'])
    plt.savefig(outputpath / ('tobacco use by age' + datestamp + '.png'), format='png')
    plt.show()


def proportion_of_men_circumcised():
    """Examine the proportion of men circumcised. """
    # get proportion of men circumcised from log files
    circ = extract_formatted_series(output['tlo.methods.enhanced_lifestyle']['prop_adult_men_circumcised'])
    circ.plot()
    plt.title('Proportion of Adult Men Circumcised')
    plt.ylim(0, 0.30)
    # display plot
    plt.show()


def proportion_of_women_sex_worker():
    """Examine the proportion of Women sex Worker"""
    # get proportion of women sex worker from log file
    fsw = extract_formatted_series(output['tlo.methods.enhanced_lifestyle']['proportion_1549_women_sexworker'])
    fsw.plot()
    plt.title('Proportion of 15-49 Women Sex Workers')
    plt.ylim(0, 0.01)
    # display plot
    plt.show()


# ----------------------------DISPLAY PLOTS-------------------------------------------------------------------
# display population distribution plot
examine_urban_population(output)

# display tobacco use by gender plot
tobacco_use_male_female(output)

# display tobacco use by age plot
tobacco_use_by_age(output)

# display proportion of men circumcised plot
proportion_of_men_circumcised()

# display proportion of women sex workers plot
proportion_of_women_sex_worker()
