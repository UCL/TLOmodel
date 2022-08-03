"""
Produce comparisons between model and GBD of deaths by cause in a particular period.

This uses the results of the Scenario defined in: src/scripts/long_run/long_run.py but it can edited to look at other
results (change 'scenario_filename').
"""

# todo - use GBD all ages numbers for some outputs (correct uncertainty bounds)

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo.analysis.utils import (
    extract_results,
    format_gbd,
    load_pickled_dataframes,
    make_age_grp_lookup,
    make_age_grp_types,
    make_calendar_period_lookup,
    make_calendar_period_type,
    summarize,
)


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    # Declare path for output graphs from this script
    make_graph_file_name = lambda stub: output_folder / f"{stub}.png"  # noqa: E731

    # Define colours to use:
    colors = {
        'Model': 'royalblue',
        'Census': 'darkred',
        'WPP': 'forestgreen',
        'GBD': 'plum'
    }

    # %% Load and process the GBD data
    gbd_all = format_gbd(pd.read_csv(resourcefilepath / 'gbd' / 'ResourceFile_Deaths_And_DALYS_GBD2019.csv'))

    # update columns name
    gbd_all = gbd_all.rename(columns={
        'Sex': 'sex',
        'Age_Grp': 'age_grp',
        'Period': 'period',
        'GBD_Est': 'mean',
        'GBD_Lower': 'lower',
        'GBD_Upper': 'upper'})

    # update name of DALYS in the gbd dataset:
    gbd_all['measure_name'] = gbd_all['measure_name'].replace({'DALYs (Disability-Adjusted Life Years)': 'DALYs'})

    def make_std_graphs(what='Deaths', period='2010-2014'):
        """Make the standard Graphs for a specific period for either 'Deaths' or 'DALYS'"""

        assert type(what) is str
        assert what in ('Deaths', 'DALYs')
        assert type(period) is str
        assert period in make_calendar_period_lookup()[0]

        # limit to the subject of interest (either 'Deaths' or 'DALYS')
        gbd = gbd_all.loc[gbd_all['measure_name'] == what].copy()

        # %% Load modelling results:

        # Extract results, summing by sex, year, age & label
        if what == 'Deaths':
            results = extract_results(
                results_folder,
                module="tlo.methods.demography",
                key="death",
                custom_generate_series=(
                    lambda df_: df_.assign(
                        year=df_['date'].dt.year
                    ).groupby(['sex', 'year', 'age', 'label'])['person_id'].count()
                ),
                do_scaling=True
            )
        else:
            results = extract_results(
                results_folder,
                module="tlo.methods.healthburden",
                key="dalys",
                custom_generate_series=(
                    lambda df_: df_.drop(
                        columns='date'
                    ).rename(
                        columns={'age_range': 'age_grp'}
                    ).groupby(['sex', 'year', 'age_grp']).sum().stack()
                ),
                do_scaling=True
            )
            results.index = results.index.set_names('label', level=3)

        # Update index to give results by five-year age-group and five-year calendar period
        agegrps, agegrplookup = make_age_grp_lookup()
        calperiods, calperiodlookup = make_calendar_period_lookup()
        results = results.reset_index()
        if 'age_grp' not in results.columns:
            results['age_grp'] = results['age'].map(agegrplookup)
            results = results.drop(columns=['age'])
        results['age_grp'] = results['age_grp'].astype(make_age_grp_types())
        results['period'] = results['year'].map(calperiodlookup).astype(make_calendar_period_type())
        results = results.drop(columns=['year'])

        # groupby, sum and divide by five to give the average number of deaths per year within the five year period:
        results = results.groupby(['period', 'sex', 'age_grp', 'label']).sum().div(5.0)

        # todo - the grouping should be inside the function for the extraction like done in
        #  `analysis_effect_of_each_treatment`...?
        # todo - factorize this.
        # todo - use colormap

        # %% Load the cause-of-deaths mappers and use them to populate the 'label' for gbd outputs
        if what == 'Deaths':
            demoglog = load_pickled_dataframes(results_folder)['tlo.methods.demography']
            mapper_from_gbd_causes = pd.Series(
                demoglog['mapper_from_gbd_cause_to_common_label'].drop(columns={'date'}).loc[0]
            ).to_dict()
        else:
            hblog = load_pickled_dataframes(results_folder)['tlo.methods.healthburden']
            mapper_from_gbd_causes = pd.Series(
                hblog['mapper_from_gbd_cause_to_common_label'].drop(columns={'date'}).loc[0]
                ).to_dict()
        gbd['label'] = gbd['cause_name'].map(mapper_from_gbd_causes)
        assert not gbd['label'].isna().any()

        # %% Make comparable pivot-tables of the GBD and Model Outputs:
        # Summarize results for average number of outcomes (per unified cause) per year within five-year periods and
        # five-year age-groups. (index=sex/age, columns=unified_cause). (Fr the particular period specified.)

        outcome_by_age_pt = dict()

        # - GBD:
        outcome_by_age_pt['GBD'] = gbd.loc[gbd.period == period].groupby(
            ['sex', 'age_grp', 'label'])[['mean', 'lower', 'upper']].sum().unstack().div(5.0)
        # NB. division by 5.0 to make it the average number of outcomes per year within the five-year period.

        # - TLO Model:
        outcome_by_age_pt['Model'] = \
            summarize(results, collapse_columns=True).reset_index().loc[lambda x: (x.period == period)].groupby(
                by=['sex', 'age_grp', 'label']
            )[['mean', 'lower', 'upper']].sum().unstack(fill_value=0.0)

        # %% Make figures of overall summaries of outcomes by cause
        # todo - improve formatting of this one

        dats = ['GBD', 'Model']
        sexes = ['F', 'M']
        sexname = lambda x: 'Females' if x == 'F' else 'Males'  # noqa: E731

        fig, axes = plt.subplots(ncols=2, nrows=2, sharey=True, sharex=True, figsize=(40, 40))

        for col, sex in enumerate(sexes):
            for row, dat in enumerate(dats):
                ax = axes[row][col]
                df = outcome_by_age_pt[dat].loc[sex].loc[:, pd.IndexSlice['mean']] / 1e3

                xs = np.arange(len(df.index))
                df.plot.bar(stacked=True, ax=ax, fontsize=30)
                ax.set_xlabel('Age Group', fontsize=40)
                ax.set_title(f"{sexname(sex)}: {dat}", fontsize=60)
                ax.get_legend().remove()

        # add a big axis, hide frame
        bigax = fig.add_subplot(111, frameon=False)

        # hide tick and tick label of the "big axis"
        bigax.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        bigax.set_ylabel(f"{what} per year (thousands)", fontsize=40)

        fig.legend(loc="center right", fontsize=15)
        fig.tight_layout()
        plt.savefig(make_graph_file_name(f"{what}_{period}_StackedBars_ModelvsGBD"))
        plt.show()

        # %% Plots of age-breakdown of outcomes patten for each cause:

        sexes = ['F', 'M']
        dats = ['GBD', 'Model']

        all_causes = list(results.index.levels[3])
        reformat_cause = lambda x: x.replace(' / ', '_')  # noqa: E731

        for cause in all_causes:
            try:
                outcomes_this_cause = pd.concat(
                    {dat: outcome_by_age_pt[dat].loc[:, (slice(None), cause)] for dat in outcome_by_age_pt.keys()},
                    axis=1
                ).fillna(0.0) / 1e3

                x = list(outcomes_this_cause.index.levels[1])
                xs = np.arange(len(x))

                fig, ax = plt.subplots(ncols=1, nrows=2, sharey=True, sharex=True)
                for row, sex in enumerate(sexes):
                    for dat in dats:
                        ax[row].plot(
                            xs,
                            outcomes_this_cause.loc[(sex,), (dat, 'mean', cause)].values,
                            label=dat,
                            color=colors[dat]
                        )
                        ax[row].fill_between(
                            xs,
                            outcomes_this_cause.loc[(sex,), (dat, 'upper', cause)].values,
                            outcomes_this_cause.loc[(sex,), (dat, 'lower', cause)].values,
                            facecolor=colors[dat], alpha=0.2
                        )
                    ax[row].legend()
                    ax[row].set_xticks(xs)
                    ax[row].set_xticklabels(x, rotation=90)
                    ax[row].set_xlabel('Age Group')
                    ax[row].set_ylabel(f'{what} per year (thousands)')
                    ax[row].set_title(f"{cause}: {sexname(sex)}, {period}")
                    ax[row].legend()

                fig.tight_layout()
                plt.savefig(make_graph_file_name(f"{what}_{period}_Scatter_Plot_{reformat_cause(cause)}"))
                plt.show()

            except KeyError:
                print(f"Could not produce plot for {what}: {reformat_cause(cause)}")

        # %% Plots comparing between model and actual across all ages and sexes:

        # - TLO Model:
        tot_outcomes_by_cause = pd.concat({
            'Model': summarize(results.groupby(by=['label']).sum(), collapse_columns=True).unstack(),
            'GBD': gbd.loc[gbd.period == period].groupby(['label']).sum()[['mean', 'lower', 'upper']].unstack()
        }, axis=1)
        # todo - for GBD, instead use all ages and all sex numbers to get correct uncertainity bounds (the addition of
        #  the bounds for the sub-categories is not correct)

        select_labels = ['AIDS', 'Childhood Diarrhoea', 'Other']

        fig, ax = plt.subplots()
        xylim = tot_outcomes_by_cause.loc[('mean', slice(None))].max().max() / 1e3
        for cause in tot_outcomes_by_cause.index.levels[1]:

            vals = tot_outcomes_by_cause.loc[(slice(None), cause), ] / 1e3

            x = vals.at[('mean', cause), 'GBD']
            xerr = np.array([
                x - vals.at[('lower', cause), 'GBD'],
                vals.at[('upper', cause), 'GBD'] - x
            ]).reshape(2, 1)
            y = vals.at[('mean', cause), 'Model']
            yerr = np.array([
                y - vals.at[('lower', cause), 'Model'],
                vals.at[('upper', cause), 'Model'] - y
            ]).reshape(2, 1)

            ax.errorbar(x=x, y=y, xerr=xerr, yerr=yerr, label=cause)

            # add labels to selected points
            if cause in select_labels:
                ax.annotate(cause,
                            (x, y),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center'
                            )

        line_x = np.linspace(0, xylim)
        ax.plot(line_x, line_x, 'r')
        ax.set(xlim=(0, xylim), ylim=(0, xylim))
        ax.set_xlabel('GBD (thousands)')
        ax.set_ylabel('Model (thousands)')
        ax.set_title(f'{what} per year by Cause {period}')
        plt.savefig(make_graph_file_name(f"{what}_{period}_Scatter_Plot"))
        plt.show()

    # %% Make graphs for each of Deaths and DALYS for a specific period
    make_std_graphs(what='Deaths', period='2010-2014')
    make_std_graphs(what='DALYs', period='2010-2014')

    make_std_graphs(what='Deaths', period='2015-2019')
    make_std_graphs(what='DALYs', period='2015-2019')
