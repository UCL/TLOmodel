"""This file is an edited and updated version of the file `schisto_analysis.py` and has been created to allow a check
that the model is working as originally intended."""

from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import (
    format_gbd,
    make_age_grp_types,
    parse_log_file,
    unflatten_flattened_multi_index_in_logging,
)
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    schisto,
    really_simplified_births,
    simplified_births,
    symptommanager,
)

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

resourcefilepath = Path("./resources")
outputpath = Path("./outputs")


# Declare path for output graphs from this script
def make_graph_file_name(name):
    return outputpath / f"Schisto_{name}.png"


# Districts that are high prevalence and for which this model has been calibrated:
fitted_districts = ['Blantyre', 'Chiradzulu', 'Mulanje', 'Nsanje', 'Nkhotakota', 'Phalombe']

# Name of species that being considered:
species = ('mansoni', 'haematobium')


# %% Run the simulation
def run_simulation(popsize,
                   use_really_simplified_births,
                   equal_allocation_by_district,
                   hs_disable_and_reject_all,
                   mda_execute,
                   single_district):
    start_date = Date(2010, 1, 1)
    end_date = Date(2015, 12, 31)
    # For logging
    custom_levels = {
        "*": logging.WARNING,
        "tlo.methods.schisto": logging.INFO,
        "tlo.methods.healthsystem.summary": logging.INFO,
        # "tlo.methods.healthburden": logging.INFO,
        "tlo.methods.hiv": logging.INFO,
        "tlo.methods.alri": logging.INFO,
        "tlo.methods.diarrhoea": logging.INFO,
        "tlo.methods.bladder_cancer": logging.INFO,
    }

    # Establish the simulation object
    sim = Simulation(start_date=start_date, seed=0, log_config={"filename": "schisto_test_runs",
                                                                "custom_levels": custom_levels, })
    sim.register(demography.Demography(resourcefilepath=resourcefilepath,
                                       equal_allocation_by_district=equal_allocation_by_district),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 # healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           disable_and_reject_all=hs_disable_and_reject_all,
                                           cons_availability='all'),
                 *(
                     [really_simplified_births.ReallySimplifiedBirths(
                         resourcefilepath=resourcefilepath)] if use_really_simplified_births else
                     [
                         simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath)
                     ]
                 ),

                 schisto.Schisto(resourcefilepath=resourcefilepath,
                                 mda_execute=mda_execute,
                                 single_district=single_district),
                 )

    # sim.modules["Schisto"].parameters["calibration_scenario"] = 0
    sim.modules["Schisto"].parameters["scaleup_WASH"] = False
    sim.modules["Schisto"].parameters["scaleup_WASH_start_year"] = 2025
    sim.modules["Schisto"].parameters['mda_coverage'] = 0
    sim.modules["Schisto"].parameters['mda_target_group'] = 'SAC'
    sim.modules["Schisto"].parameters['mda_frequency_months'] = 12

    # initialise the population
    sim.make_initial_population(n=popsize)

    # start the simulation
    sim.simulate(end_date=end_date)

    output = parse_log_file(sim.log_filepath)

    return sim, output

# todo update these parameters
sim, output = run_simulation(popsize=1_000,
                             use_really_simplified_births=False,
                             equal_allocation_by_district=True,
                             hs_disable_and_reject_all=False,  # if True, no HSIs run
                             mda_execute=False,
                             single_district=True)

# %% Extract and process the `pd.DataFrame`s needed

with open(outputpath / "test_run.pickle", "wb") as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(dict(output), f, pickle.HIGHEST_PROTOCOL)

# load the results
with open(outputpath / "test_run.pickle", "rb") as f:
    output = pickle.load(f)


def construct_dfs(schisto_log) -> dict:
    """Create dict of pd.DataFrames containing counts of infection_status by date, district and age-group."""
    return {
        k: unflatten_flattened_multi_index_in_logging(v.set_index('date'))
        for k, v in schisto_log.items() if k in [f'infection_status_{s}' for s in species]
    }


dfs = construct_dfs(output['tlo.methods.schisto'])


# %% Plot the district-level prevalence at the end of the simulation and compare with data

def get_model_prevalence_by_district(spec: str, year: int):
    """Get the prevalence of a particular species at end of 2010 (???) """
    _df = dfs[f'infection_status_{spec}']
    t = _df.loc[_df.index.year == year].iloc[-1]  # gets the last entry for 2010 (Dec)
    counts = t.unstack(level=1).groupby(level=0).sum().T
    return ((counts['High-infection'] + counts['Moderate-infection'] + counts['Low-infection']) / counts.sum(
        axis=1)).to_dict()


def get_expected_prevalence_by_district(species: str):
    """Get the prevalence of a particular species from the data (which is for year 2010/2011)."""
    expected_district_prevalence = pd.read_excel(resourcefilepath / 'ResourceFile_Schisto.xlsx',
                                                 sheet_name='OLDDistrict_Params_' + species.lower())
    expected_district_prevalence.set_index("District", inplace=True)
    expected_district_prevalence = expected_district_prevalence.loc[:, 'Prevalence'].to_dict()
    return expected_district_prevalence


def get_model_prevalence_by_district_over_time(spec: str):
    """Get the prevalence every year of the simulation """
    _df = dfs[f'infection_status_{spec}']
    # select the last entry for each year
    df = _df.resample('Y').last()

    # Aggregate the sums of infection statuses by district_of_residence and year
    district_sums = df.groupby(level='district_of_residence', axis=1).sum()

    filtered_columns = df.columns.get_level_values('infection_status').isin(
        ['High-infection', 'Moderate-infection', 'Low-infection'])
    infected = df.loc[:, filtered_columns].groupby(level='district_of_residence', axis=1).sum()

    prop_infected = infected.div(district_sums)

    return prop_infected


def construct_susceptibility_dfs(schisto_log, species: list) -> dict:
    """
    Create a dict of pd.DataFrames containing counts of infection_status by date and district for each species.
    """
    dfs_susc = {}
    for s in species:
        key = f'susceptibility_{s}'
        if key in schisto_log:
            # Simply set the date as the index without any additional processing
            df_processed = schisto_log[key].set_index('date')
            dfs_susc[s] = df_processed
    return dfs_susc


dfs_susc = construct_susceptibility_dfs(output['tlo.methods.schisto'], species)

# %%  -------------------------- PLOTS --------------------------


# PREVALENCE

# Districts with prevalence fitted
fig, axes = plt.subplots(1, 2, sharey=True)
for i, _spec in enumerate(species):
    ax = axes[i]
    pd.DataFrame(data={
        'Data': get_expected_prevalence_by_district(_spec),
        'Model': get_model_prevalence_by_district(_spec, year=2010)}
    ).loc[fitted_districts].plot.bar(ax=ax)
    ax.set_title(f"{_spec}")
    ax.set_xlabel('District (Fitted)')
    ax.set_ylabel('Prevalence, 2010-2011')
    ax.set_ylim(0, 1.0)
    ax.legend(loc=1)
fig.tight_layout()
# fig.savefig(make_graph_file_name('prev_in_districts_all'))
fig.show()

# All Districts - model prev and data prev, one year
fig, axes = plt.subplots(1, 2, sharey=True)
for i, _spec in enumerate(species):
    ax = axes[i]
    pd.DataFrame(data={
        'Data': get_expected_prevalence_by_district(_spec),
        'Model': get_model_prevalence_by_district(_spec, year=2010)}
    ).plot(ax=ax)
    ax.set_title(f"{_spec}")
    ax.set_xlabel('District (All)')
    ax.set_ylabel('Prevalence, 2010-2011')
    # ax.set_ylim(0, 0.50)
    ax.legend(loc=1)
fig.tight_layout()
# fig.savefig(make_graph_file_name('prev_in_districts_fitted'))
fig.show()

# All Districts = prevalence by year
fig, axes = plt.subplots(1, 2, sharey=True)
for i, _spec in enumerate(species):
    ax = axes[i]
    data = get_model_prevalence_by_district_over_time(_spec)
    data.plot(ax=ax)
    ax.set_title(f"{_spec}")
    ax.set_xlabel('')
    ax.set_ylabel('End of year prevalence')
    ax.set_ylim(0, 1.0)
    ax.get_legend().remove()
    # data.to_csv(outputpath / (f"{_spec}" + '.csv'))

    # Plot legend only for the last subplot
    # if i == len(species) -1:
    #     handles, labels = ax.get_legend_handles_labels()  # Get handles and labels for legend
    #     ax.legend(handles, labels, bbox_to_anchor =(1.44,-0.10), loc='lower right')

fig.tight_layout()
# fig.savefig(make_graph_file_name('annual_prev_in_districts'))
fig.show()


# PROPORTION SUSCEPTIBLE

def plot_susceptibility(dfs_susc: dict):
    """
    Plot susceptibility over time for each species.

    Args:
    dfs_susc (dict): A dictionary where each key is a species name and the value is a DataFrame
                     containing susceptibility data with 'date' as the index.
    """
    for species, df in dfs_susc.items():
        plt.figure(figsize=(10, 6))
        for column in df.columns:
            plt.plot(df.index, df[column], label=column)

        plt.title(f'Proportion susceptible {species.capitalize()}')
        plt.xlabel('Date')
        plt.ylabel('Proportion susceptible')
        plt.legend(title="Districts", bbox_to_anchor=(0.5, -0.2), loc="upper center", ncol=4)
        plt.grid(True)
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


# Example usage
plot_susceptibility(dfs_susc)
