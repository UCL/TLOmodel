"""This file is an edited and updated version of the file `schisto_analysis.py` and has been created to allow a check
that the model is working as originally intended."""

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
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    schisto,
    simplified_births,
    symptommanager,
)

resourcefilepath = Path("./resources")
outputpath = Path("./outputs")

# Declare path for output graphs from this script
make_graph_file_name = lambda name: outputpath / f"Schisto_{name}.png"  # noqa: E73

# Districts that are high prevalence and for which this model has been calibrated:
fitted_districts = ['Blantyre', 'Chiradzulu', 'Mulanje', 'Nsanje', 'Nkhotakota', 'Phalombe']

# Name of species that being considered:
species = ('mansoni', 'haematobium')


# %% Run the simulation
popsize = 10_000


def run_simulation(popsize=popsize, mda_execute=True):
    start_date = Date(2010, 1, 1)
    end_date = Date(2011, 2, 1)

    # For logging, set all modules to WARNING threshold, then alters `Shisto` to level "INFO"
    custom_levels = {
        "*": logging.WARNING,
        "tlo.methods.schisto": logging.INFO,
        "tlo.methods.healthburden": logging.INFO
    }

    # Establish the simulation object
    sim = Simulation(start_date=start_date, seed=0, log_config={"filename": __file__[-19:-3],
                                                                "custom_levels": custom_levels}
                     )
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 schisto.Schisto(resourcefilepath=resourcefilepath, mda_execute=mda_execute),
                 )

    # initialise the population
    sim.make_initial_population(n=popsize)

    # start the simulation
    sim.simulate(end_date=end_date)

    output = parse_log_file(sim.log_filepath)
    return sim, output


sim, output = run_simulation(popsize=10000, mda_execute=False)

# %% Extract and process the `pd.DataFrame`s needed


def construct_dfs(schisto_log) -> dict:
    """Create dict of pd.DataFrames containing counts of infection_status by date, district and age-group."""
    return {
        k: unflatten_flattened_multi_index_in_logging(v.set_index('date'))
        for k, v in schisto_log.items() if k in [f'infection_status_{s}' for s in species]
    }


dfs = construct_dfs(output['tlo.methods.schisto'])


# %% Plot the district-level prevalence at the end of the simulation and compare with data

def get_model_prevalence_by_district(spec: str):
    """Get the prevalence of a particular species at end of 2010 (???) for a particular species. """
    _df = dfs[f'infection_status_{spec}']
    t = _df.loc[_df.index.year == 2010].iloc[-1]
    counts = t.unstack(level=1).sum(level=0).T
    return ((counts['High-infection'] + counts['Low-infection']) / counts.sum(axis=1)).to_dict()


def get_expected_prevalence_by_district(species: str):
    """Get the prevalence of a particular species from the data (which is for year 2010/2011)."""
    expected_district_prevalence = pd.read_excel(resourcefilepath / 'ResourceFile_Schisto.xlsx',
                                                 sheet_name='District_Params_' + species.lower())
    expected_district_prevalence.set_index("District", inplace=True)
    expected_district_prevalence = expected_district_prevalence.loc[:, 'Prevalence'].to_dict()
    return expected_district_prevalence


# Districts with prevalence fitted
fig, axes = plt.subplots(1, 2, sharey=True)
for i, _spec in enumerate(species):
    ax = axes[i]
    pd.DataFrame(data={
        'Data': get_expected_prevalence_by_district(_spec),
        'Model': get_model_prevalence_by_district(_spec)}
    ).loc[fitted_districts].plot.bar(ax=ax)
    ax.set_title(f"{_spec}")
    ax.set_xlabel('District (Fitted)')
    ax.set_ylabel('Prevalence, 2010-2011')
    ax.set_ylim(0, 0.50)
    ax.legend(loc=1)
fig.tight_layout()
fig.savefig(make_graph_file_name('prev_in_districts_all'))
fig.show()

# All Districts
fig, axes = plt.subplots(1, 2, sharey=True)
for i, _spec in enumerate(species):
    ax = axes[i]
    pd.DataFrame(data={
        'Data': get_expected_prevalence_by_district(_spec),
        'Model': get_model_prevalence_by_district(_spec)}
    ).plot(ax=ax)
    ax.set_title(f"{_spec}")
    ax.set_xlabel('District (All)')
    ax.set_ylabel('Prevalence, 2010-2011')
    ax.set_ylim(0, 0.50)
    ax.legend(loc=1)
fig.tight_layout()
fig.savefig(make_graph_file_name('prev_in_districts_fitted'))
fig.show()


# %% DALYS

def get_model_dalys_schisto_2010():
    """Get the DALYS attributed to Schistosomiasis in 2010."""
    dalys = output['tlo.methods.healthburden']["dalys"]
    scaling_factor = 14.5e6 / popsize  # todo - this properly once dealt with https://github.com/UCL/TLOmodel/issues/536
    dalys_schisto = dalys.set_index('year').loc[2010].groupby(by='age_range')['Schistosomiasis'].sum()
    dalys_schisto.index = dalys_schisto.index.astype(make_age_grp_types())
    dalys_schisto.name = 'Model'
    return dalys_schisto.sort_index() * scaling_factor


def get_gbd_dalys_schisto_2010():
    """Get the DALYS attributed to Schistosomiasis in 2010"""
    gbd_all = format_gbd(pd.read_csv(resourcefilepath / 'gbd' / 'ResourceFile_Deaths_And_DALYS_GBD2019.csv'))
    return gbd_all.loc[
        (gbd_all.cause_name == 'Schistosomiasis') & (gbd_all.Year == 2010)
        ].groupby(by='Age_Grp')[['GBD_Est', 'GBD_Lower', 'GBD_Upper']].sum()


dat = pd.concat([get_gbd_dalys_schisto_2010(), get_model_dalys_schisto_2010()], axis=1)

fig, ax = plt.subplots()
ax.plot(dat.index, dat.GBD_Est.values, color='b', label='GBD')
ax.fill_between(dat.index, dat.GBD_Lower.values, dat.GBD_Upper.values, alpha=0.5, color='b')
ax.plot(dat.index, dat.Model.values, color='r', label='Model')
ax.set_title("DALYS")
ax.set_xlabel('Age-Group')
ax.set_ylabel('DALYS (2010)')
ax.set_xticklabels(dat.index, rotation=90)
ax.legend(loc=1)
fig.tight_layout()
fig.savefig(make_graph_file_name('dalys_2010'))
fig.show()
