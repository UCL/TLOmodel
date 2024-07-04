""" use the outputs from a local run and plot model outputs
against data
This uses the updated prevalence data averaged over 2010-2015 as well
as single year estimates
"""

from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import pandas as pd
# import lacroix
import matplotlib.colors as colors
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from collections import defaultdict

from tlo import Date, Simulation, logging
from tlo.analysis.utils import (
    format_gbd,
    make_age_grp_types,
    parse_log_file,
    compare_number_of_deaths,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
    make_age_grp_lookup,
    make_age_grp_types,
    unflatten_flattened_multi_index_in_logging,
)


resourcefilepath = Path("./resources")
# outputpath = Path("./outputs")

outputpath = Path("./outputs/t.mangal@imperial.ac.uk")

results_folder = get_scenario_outputs("schisto_calibration.py", outputpath)[-1]

# Declare path for output graphs from this script
def make_graph_file_name(name):
    return outputpath / f"Schisto_{name}.png"


# Name of species that being considered:
species = ('mansoni', 'haematobium')

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
scenario_info = get_scenario_info(results_folder)

# Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# %% Extract and process the `pd.DataFrame`s needed


def construct_dfs(schisto_log) -> dict:
    """Create dict of pd.DataFrames containing counts of infection_status by date, district and age-group."""
    return {
        k: unflatten_flattened_multi_index_in_logging(v.set_index('date'))
        for k, v in schisto_log.items() if k in [f'infection_status_{s}' for s in species]
    }


dfs = construct_dfs(log['tlo.methods.schisto'])


# %% get data on prevalence


def get_expected_prevalence_by_district_2010(species: str):
    """Get the prevalence of a particular species from the data (which is for year 2010/2011)."""
    expected_district_prevalence = pd.read_excel(resourcefilepath / 'ResourceFile_Schisto.xlsx',
                                                 sheet_name='District_Params_' + species.lower())
    expected_district_prevalence.set_index("District", inplace=True)
    expected_district_prevalence = expected_district_prevalence.loc[:, 'Prevalence'].to_dict()
    return expected_district_prevalence


def get_expected_prevalence_by_district_2010_2015(species: str):
    """Get the prevalence of a particular species from the data (which is for year 2010/2011)."""
    expected_district_prevalence = pd.read_excel(resourcefilepath / 'ResourceFile_Schisto.xlsx',
                                                 sheet_name='Data_' + species.lower())
    expected_district_prevalence.set_index("District", inplace=True)
    expected_district_prevalence.loc[:, 'mean_prevalence'] = expected_district_prevalence.loc[:, 'mean_prevalence'] / 100
    expected_district_prevalence.loc[:, 'min_prevalence'] = expected_district_prevalence.loc[:, 'min_prevalence'] / 100
    expected_district_prevalence.loc[:, 'max_prevalence'] = expected_district_prevalence.loc[:, 'max_prevalence'] / 100

    district_data = expected_district_prevalence[['mean_prevalence', 'min_prevalence', 'max_prevalence']].to_dict(orient='index')

    return district_data


# %% get model outputs on prevalence

def get_model_prevalence_one_year(spec: str, year: int, include: str, summarise=True):
    """Get the prevalence of a particular species in specified year
     SAC infections only
     include statement selects either moderate|high infections ('HM') or all infections ('HML")
     summarise=True returns median/percentiles, otherwise returns prevalence for each district for each run
     """

    all_prevalence = defaultdict(list)

    # for every model run:
    for i in range(scenario_info["runs_per_draw"]):
        log = load_pickled_dataframes(results_folder, draw=0, run=i)

        dfs = construct_dfs(log['tlo.methods.schisto'])

        _df = dfs[f'infection_status_{spec}']

        # Select columns where 'age_years' level is 'SAC'
        _df = _df.loc[:, _df.columns.get_level_values('age_years') == 'SAC']
        t = _df.loc[_df.index.year == year].iloc[-1]  # gets the last entry for 2010 (Dec)

        # calculate the prop_infected for each year for each district
        counts = t.unstack(level=1).groupby(level=0).sum().T

        if include == "HM":
            tmp = ((counts['High-infection'] + counts['Moderate-infection']) / counts.sum(axis=1)).to_dict()
        else:
            tmp = ((counts['High-infection'] + counts['Moderate-infection'] + counts['Low-infection']) / counts.sum(
                axis=1)).to_dict()

        # Append the results to the all_prevalence dictionary
        for district, prevalence in tmp.items():
            all_prevalence[district].append(prevalence)

    # take the median values of prop_infected across the years and runs for every district
    # output will be dict with key=district, value=median prop_infected, percentiles etc
    # take median/percentiles of each value in dict
    prevalence_stats = {
        district: {
            "median": np.median(prevalences),
            "percentile_2.5": np.percentile(prevalences, 2.5),
            "percentile_97.5": np.percentile(prevalences, 97.5)
        }
        for district, prevalences in all_prevalence.items()
    }

    if summarise:
        return prevalence_stats
    else:
        return all_prevalence


tmp = get_model_prevalence_one_year(spec='haematobium', year=2010, include="HM", summarise=False)


# to get average annual prevalence over time-period 2010-2015
def get_average_model_across_years(spec: str, include:str):
    """Get the prevalence of a particular species in specified year
     SAC infections only
     include statement selects either moderate|high infections ('HM') or all infections ('HML")
     summarise=True returns median/percentiles, otherwise returns prevalence for each district for each run
     """

    prev2010 = get_model_prevalence_one_year(spec=spec, year=2010, include=include, summarise=False)
    prev2011 = get_model_prevalence_one_year(spec=spec, year=2011, include=include, summarise=False)
    prev2012 = get_model_prevalence_one_year(spec=spec, year=2012, include=include, summarise=False)
    prev2013 = get_model_prevalence_one_year(spec=spec, year=2013, include=include, summarise=False)
    prev2014 = get_model_prevalence_one_year(spec=spec, year=2014, include=include, summarise=False)
    prev2015 = get_model_prevalence_one_year(spec=spec, year=2015, include=include, summarise=False)


    # Define a function to accumulate prevalence data from each year's dictionary
    def accumulate_prevalence_data(*yearly_data):
        combined_data = {}
        for year_data in yearly_data:
            for district, values in year_data.items():
                if district not in combined_data:
                    combined_data[district] = []
                combined_data[district].extend(values)
        return combined_data

    # List of all prevalence data
    all_years_data = [prev2010, prev2011, prev2012, prev2013, prev2014, prev2015]
    combined_data = accumulate_prevalence_data(*all_years_data)

    # Calculate median, 2.5th percentile, and 97.5th percentile for each district
    summary_prevalence = {
        district: {
            'median': np.median(values),
            'percentile_2.5': np.percentile(values, 2.5),
            'percentile_97.5': np.percentile(values, 97.5)
        }
        for district, values in combined_data.items()
    }

    return summary_prevalence


tmp2 = get_average_model_across_years(spec='haematobium', include="HM")


# %% plots model outputs with data
def plot_2010model_prevalence_with_2010_2015data(species, include='HM'):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    for i, _spec in enumerate(species):
        ax = axes[i]

        # Fetch expected and model prevalence data
        data_2010 = get_expected_prevalence_by_district_2010_2015(_spec)
        model_prevalence_stats = get_model_prevalence_one_year(_spec, year=2010, include=include)

        # Extract median and confidence intervals from model prevalence stats
        districts = list(model_prevalence_stats.keys())
        median_values = [model_prevalence_stats[d]['median'] for d in districts]
        lower_bounds = [model_prevalence_stats[d]['percentile_2.5'] for d in districts]
        upper_bounds = [model_prevalence_stats[d]['percentile_97.5'] for d in districts]

        # Calculate error bars for model data
        yerr_lower_model = np.array(median_values) - np.array(lower_bounds)
        yerr_upper_model = np.array(upper_bounds) - np.array(median_values)
        yerr_model = [yerr_lower_model, yerr_upper_model]

        # Plot Data 2010 points with error bars as confidence intervals
        x_districts = np.arange(len(districts))
        data_prevalence = np.array([data_2010[d]['mean_prevalence'] for d in districts])
        min_prevalence = np.array([data_2010[d]['min_prevalence'] for d in districts])
        max_prevalence = np.array([data_2010[d]['max_prevalence'] for d in districts])
        yerr_data = [data_prevalence - min_prevalence, max_prevalence - data_prevalence]

        # Stagger the x-coordinates
        offset = 0.2
        x_data = x_districts - offset
        x_model = x_districts + offset

        ax.errorbar(x_data, data_prevalence, yerr=yerr_data, fmt='_', color='blue', label='Data 2010-2015', capsize=4)
        ax.errorbar(x_model, median_values, yerr=yerr_model, fmt='_', color='red',
                    label='Model 2010', capsize=4)

        # Add vertical lines between districts
        for x in x_districts:
            ax.axvline(x - 0.5, color='lightgrey', linestyle='--', linewidth=0.5)

        ax.set_title(f"{_spec}_{include}")
        ax.set_xlabel('District')
        ax.set_ylabel('Prevalence in SAC')
        ax.set_ylim(0, 1.0)  # Adjust as needed
        ax.legend(loc=1)
        ax.set_xticks(x_districts)
        ax.set_xticklabels(districts, rotation=90)

    fig.tight_layout()
    plt.show()


def plot_2010_2015model_prevalence_with_2010_2015data(species, include='HM'):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    for i, _spec in enumerate(species):
        ax = axes[i]

        # Fetch expected and model prevalence data
        data_2010 = get_expected_prevalence_by_district_2010_2015(_spec)
        model_prevalence_stats = get_average_model_across_years(_spec, include=include)

        # Extract median and confidence intervals from model prevalence stats
        districts = list(model_prevalence_stats.keys())
        median_values = [model_prevalence_stats[d]['median'] for d in districts]
        lower_bounds = [model_prevalence_stats[d]['percentile_2.5'] for d in districts]
        upper_bounds = [model_prevalence_stats[d]['percentile_97.5'] for d in districts]

        # Calculate error bars for model data
        yerr_lower_model = np.array(median_values) - np.array(lower_bounds)
        yerr_upper_model = np.array(upper_bounds) - np.array(median_values)
        yerr_model = [yerr_lower_model, yerr_upper_model]

        # Plot Data 2010 points with error bars as confidence intervals
        x_districts = np.arange(len(districts))
        data_prevalence = np.array([data_2010[d]['mean_prevalence'] for d in districts])
        min_prevalence = np.array([data_2010[d]['min_prevalence'] for d in districts])
        max_prevalence = np.array([data_2010[d]['max_prevalence'] for d in districts])
        yerr_data = [data_prevalence - min_prevalence, max_prevalence - data_prevalence]

        # Stagger the x-coordinates
        offset = 0.2
        x_data = x_districts - offset
        x_model = x_districts + offset

        ax.errorbar(x_data, data_prevalence, yerr=yerr_data, fmt='_', color='blue', label='Data 2010-2015', capsize=4)
        ax.errorbar(x_model, median_values, yerr=yerr_model, fmt='_', color='red',
                    label='Model 2010-2015', capsize=4)

        # Add vertical lines between districts
        for x in x_districts:
            ax.axvline(x - 0.5, color='lightgrey', linestyle='--', linewidth=0.5)

        ax.set_title(f"{_spec}_{include}")
        ax.set_xlabel('District')
        ax.set_ylabel('Prevalence in SAC')
        ax.set_ylim(0, 1.0)  # Adjust as needed
        ax.legend(loc=1)
        ax.set_xticks(x_districts)
        ax.set_xticklabels(districts, rotation=90)

    fig.tight_layout()
    plt.show()


# PLOTS
plot_2010model_prevalence_with_2010_2015data(species, include='HML')

plot_2010_2015model_prevalence_with_2010_2015data(species, include='HM')












# ----------- PLOTS -----------------

# Districts with prevalence fitted
# # todo include HM only or HML for all infections in model

def plot_2010model_prevalence_with_2010data(species, include='HM'):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    for i, _spec in enumerate(species):
        ax = axes[i]

        # Fetch expected and model prevalence data
        data_2010 = get_expected_prevalence_by_district_2010(_spec)
        model_prevalence_stats = get_model_prevalence_one_year(_spec, year=2010, include=include)

        # Extract median and confidence intervals
        districts = list(model_prevalence_stats.keys())
        median_values = [model_prevalence_stats[d]['median'] for d in districts]
        lower_bounds = [model_prevalence_stats[d]['percentile_2.5'] for d in districts]
        upper_bounds = [model_prevalence_stats[d]['percentile_97.5'] for d in districts]

        # Calculate error bars
        yerr_lower = [median_values[j] - lower_bounds[j] for j in range(len(median_values))]
        yerr_upper = [upper_bounds[j] - median_values[j] for j in range(len(median_values))]
        yerr = [yerr_lower, yerr_upper]

        # Plot Data 2010 points
        data_prevalence = [data_2010[d] for d in districts]
        x_districts = range(len(districts))

        # Stagger the points
        offset = 0.2
        x_data = [x - offset for x in x_districts]
        x_model = [x + offset for x in x_districts]

        # Plot Data 2010 points
        ax.plot(x_data, data_prevalence, 'o', label='Data 2010', color='blue')

        # Plot Model 2010 median with error bars
        ax.errorbar(x_model, median_values, yerr=yerr, fmt='none', ecolor='red', capsize=4)
        ax.plot(x_model, median_values, '_', markersize=10, color='red', label='Model 2010')

        # Add vertical lines between districts
        for x in x_districts:
            ax.axvline(x - 0.5, color='lightgrey', linestyle='--', linewidth=0.5)

        ax.set_title(f"{_spec}")
        ax.set_xlabel('District')
        ax.set_ylabel('Prevalence 2010')
        ax.set_ylim(0, 0.6)
        ax.legend(loc=1)
        ax.set_xticks(x_districts)
        ax.set_xticklabels(districts, rotation=90)

    fig.tight_layout()
    # fig.savefig(make_graph_file_name('prev_in_districts_all'))
    plt.show()


plot_2010model_prevalence_with_2010data(species, include='HL')


# -----------------------------------------------------------------------
# Average prevalence with summarised survey data 2010-2015


def plot_2010model_prevalence_with_2010_2015data(species, include='HM'):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    for i, _spec in enumerate(species):
        ax = axes[i]

        # Fetch expected and model prevalence data
        data_2010 = get_expected_prevalence_by_district_2010_2015(_spec)
        model_prevalence_stats = get_model_prevalence_one_year(_spec, year=2010, include=include)

        # Extract median and confidence intervals from model prevalence stats
        districts = list(model_prevalence_stats.keys())
        median_values = [model_prevalence_stats[d]['median'] for d in districts]
        lower_bounds = [model_prevalence_stats[d]['percentile_2.5'] for d in districts]
        upper_bounds = [model_prevalence_stats[d]['percentile_97.5'] for d in districts]

        # Calculate error bars for model data
        yerr_lower_model = np.array(median_values) - np.array(lower_bounds)
        yerr_upper_model = np.array(upper_bounds) - np.array(median_values)
        yerr_model = [yerr_lower_model, yerr_upper_model]

        # Plot Data 2010 points with error bars as confidence intervals
        x_districts = np.arange(len(districts))
        data_prevalence = np.array([data_2010[d]['mean_prevalence'] for d in districts])
        min_prevalence = np.array([data_2010[d]['min_prevalence'] for d in districts])
        max_prevalence = np.array([data_2010[d]['max_prevalence'] for d in districts])
        yerr_data = [data_prevalence - min_prevalence, max_prevalence - data_prevalence]

        # Stagger the x-coordinates
        offset = 0.2
        x_data = x_districts - offset
        x_model = x_districts + offset

        ax.errorbar(x_data, data_prevalence, yerr=yerr_data, fmt='_', color='blue', label='Data 2010-2015', capsize=4)
        ax.errorbar(x_model, median_values, yerr=yerr_model, fmt='_', color='red',
                    label='Model', capsize=4)

        # Add vertical lines between districts
        for x in x_districts:
            ax.axvline(x - 0.5, color='lightgrey', linestyle='--', linewidth=0.5)

        ax.set_title(f"{_spec}")
        ax.set_xlabel('District')
        ax.set_ylabel('Prevalence in SAC')
        ax.set_ylim(0, 1.0)  # Adjust as needed
        ax.legend(loc=1)
        ax.set_xticks(x_districts)
        ax.set_xticklabels(districts, rotation=90)

    fig.tight_layout()
    plt.show()

plot_2010model_prevalence_with_2010_2015data(species, include='HML')

