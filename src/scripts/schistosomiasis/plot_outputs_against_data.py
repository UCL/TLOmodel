""" use the outputs from a local run and plot model outputs
against data
This uses the updated prevalence data averaged over 2010-2015 as well
as single year estimates
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import (
    format_gbd,
    make_age_grp_types,
    parse_log_file,
    unflatten_flattened_multi_index_in_logging,
)

resourcefilepath = Path("./resources")
outputpath = Path("./outputs")


# Declare path for output graphs from this script
def make_graph_file_name(name):
    return outputpath / f"Schisto_{name}.png"


# Name of species that being considered:
species = ('mansoni', 'haematobium')


# %% Extract and process the `pd.DataFrame`s needed


def construct_dfs(schisto_log) -> dict:
    """Create dict of pd.DataFrames containing counts of infection_status by date, district and age-group."""
    return {
        k: unflatten_flattened_multi_index_in_logging(v.set_index('date'))
        for k, v in schisto_log.items() if k in [f'infection_status_{s}' for s in species]
    }


dfs = construct_dfs(output['tlo.methods.schisto'])


# %% Plot the district-level prevalence summarised over the simulation period and compare with data

def get_model_prevalence_one_year(spec: str, year: int, include: str):
    """Get the prevalence of a particular species 2010
     SAC infections only, moderate|high infections
     """
    _df = dfs[f'infection_status_{spec}']

    # Select columns where 'age_years' level is 'SAC'
    _df = _df.loc[:, _df.columns.get_level_values('age_years') == 'SAC']
    t = _df.loc[_df.index.year == year].iloc[-1]  # gets the last entry for 2010 (Dec)

    counts = t.unstack(level=1).groupby(level=0).sum().T

    if include == "HM":
        return ((counts['High-infection'] + counts['Moderate-infection']) / counts.sum(axis=1)).to_dict()
    else:
        return ((counts['High-infection'] + counts['Moderate-infection'] + counts['Low-infection']) / counts.sum(
            axis=1)).to_dict()


def get_model_average_prevalence(spec: str, include: str):
    """Get the prevalence of a particular species 2010-2015
     SAC infections only, moderate|high infections
     """
    _df = dfs[f'infection_status_{spec}']
    _df = _df.loc[:, _df.columns.get_level_values('age_years') == 'SAC']
    _df = _df.loc[_df.index <= '2015-12-31']

    # Calculate the ratio for each row and district
    districts = _df.columns.get_level_values('district_of_residence').unique()

    for district in districts:

        if include == 'HM':
            infected_sum = (_df.loc[:, ('High-infection', district, 'SAC')] +
                            _df.loc[:, ('Moderate-infection', district, 'SAC')])
        else:
            infected_sum = (_df.loc[:, ('High-infection', district, 'SAC')] +
                            _df.loc[:, ('Moderate-infection', district, 'SAC')] +
                            _df.loc[:, ('Low-infection', district, 'SAC')])

        total_district_sum = _df.loc[:, (slice(None), district, 'SAC')].sum(
            axis=1)
        prop_infected = infected_sum / total_district_sum
        _df[('prop_infected', district, 'SAC')] = prop_infected

    # Calculate total prop_infected for each district
    prop_infected_data = _df.loc[:, ('prop_infected', slice(None), 'SAC')]

    # Calculate mean across columns (axis=0)
    mean_prop_infected = prop_infected_data.mean(axis=0)

    # Dropping unnecessary levels from index
    mean_prop_infected.index = mean_prop_infected.index.droplevel(['infection_status', 'age_years'])

    return mean_prop_infected


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
    expected_district_prevalence = expected_district_prevalence['mean_prevalence'].to_dict()

    return expected_district_prevalence


# ----------- PLOTS -----------------

# Districts with prevalence fitted
# todo include HM only or HML for all infections in model
fig, axes = plt.subplots(2, 1, sharex=True)
for i, _spec in enumerate(species):
    ax = axes[i]
    pd.DataFrame(data={
        'Data 2010': get_expected_prevalence_by_district_2010(_spec),
        'Model 2010': get_model_prevalence_one_year(_spec, year=2010, include='HML')}
    ).plot.bar(ax=ax)
    ax.set_title(f"{_spec}")
    ax.set_xlabel('District (Fitted)')
    ax.set_ylabel('Prevalence, 2010-2011')
    ax.set_ylim(0, 1.0)
    ax.legend(loc=1)
fig.tight_layout()
# fig.savefig(make_graph_file_name('prev_in_districts_all'))
fig.show()


# -----------------------------------------------------------------------
# Average prevalence with summarised survey data 2010-2015
# todo include HM only or HML for all infections in model
fig, axes = plt.subplots(2, 1, sharex=True)
for i, _spec in enumerate(species):
    ax = axes[i]
    pd.DataFrame(data={
        'Data 2010-2015': get_expected_prevalence_by_district_2010_2015(_spec),
        'Model 2010-2015': get_model_average_prevalence(_spec, include='HM')}
    ).plot.bar(ax=ax)
    ax.set_title(f"{_spec}")
    ax.set_xlabel('District (Fitted)')
    ax.set_ylabel('Prevalence, 2010-2011')
    ax.set_ylim(0, 1.0)
    ax.legend(loc=1)
fig.tight_layout()
# fig.savefig(make_graph_file_name('prev_in_districts_all'))
fig.show()
