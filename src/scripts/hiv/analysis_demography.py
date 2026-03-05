"""
Run the HIV/TB modules with intervention coverage specified at national level
save outputs for plotting (file: output_plots_tb.py)
 """

import datetime
import pickle
import random
from pathlib import Path

from matplotlib import pyplot as plt
import seaborn as sns

from shield import Date, Simulation, logging
from shield.analysis.utils import parse_log_file
from shield.methods import (
    demography,
    simplified_births,
)

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = './resources'

# %% Run the simulation
seed = random.randint(0, 50000)
start_date = Date(2010, 1, 1)
end_date = Date(2020, 1, 1)
# country = "Malta"
popsize = 5000

# set up the log config
log_config = {
    "filename": "test_runs",
    "directory": outputpath,
    "custom_levels": {
        "*": logging.WARNING,
        "shield.methods.demography": logging.INFO,
        "shield.methods.demography.detail": logging.WARNING,
    },
}


def get_sim(seed, country=country):

    start_date = Date(2010, 1, 1)
    sim = Simulation(start_date=start_date, seed=seed,
                     log_config=log_config,
                     resourcefilepath=resourcefilepath,
                     show_progress_bar=True)

    # Register the appropriate modules
    sim.register(demography.Demography(),
                 simplified_births.SimplifiedBirths(),
                 )
    return sim


# simple run
sim = get_sim(seed, country=country)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)





# parse the results
output = parse_log_file(sim.log_filepath)

# save the results, argument 'wb' means write using binary mode. use 'rb' for reading file
with open(outputpath / "default_run.pickle", "wb") as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(dict(output), f, pickle.HIGHEST_PROTOCOL)

# load the results
with open(outputpath / "default_run.pickle", "rb") as f:
    output = pickle.load(f)


#%% plots


# plot populations over time


def plot_stacked_population(output):
    """
    Stacked area plot using seaborn styling:
    - female shaded base
    - male stacked on top
    - total shown as upper boundary line
    """

    sns.set_theme(style="white")   # clean background

    df = output["shield.methods.demography"]["population"].copy()
    scaling_factor = output["shield.methods.population"]["scaling_factor"].values[0][1]
    x = df['date']
    female = df["female"] * scaling_factor
    male = df["male"] * scaling_factor
    total = female + male   # ensures stacking consistency

    fig, ax = plt.subplots(figsize=(10, 6))

    # Female (base layer)
    ax.fill_between(
        x,
        0,
        female,
        label="Female",
        alpha=0.8
    )

    # Male (stacked above female)
    ax.fill_between(
        x,
        female,
        total,
        label="Male",
        alpha=0.8
    )

    # Top boundary line (total population)
    sns.lineplot(
        x=x,
        y=total,
        ax=ax,
        linewidth=2,
        label="Total"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Population")

    # Horizontal grid only
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)

    sns.despine()
    ax.legend()

    plt.tight_layout()
    plt.show()


plot_stacked_population(output)


