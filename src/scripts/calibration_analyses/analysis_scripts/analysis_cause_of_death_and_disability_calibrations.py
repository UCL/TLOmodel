"""
Produce comparisons between model and GBD of deaths by cause in a particular period.

This uses the results of the Scenario defined in: src/scripts/long_run/long_run.py but it can edited to look at other
results (change 'scenario_filename').
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo.analysis.utils import (
    extract_results,
    format_gbd,
    get_color_cause_of_death_label,
    get_scenario_outputs,
    load_pickled_dataframes,
    make_age_grp_lookup,
    make_age_grp_types,
    make_calendar_period_lookup,
    make_calendar_period_type,
    order_of_cause_of_death_label,
    summarize,
)

PREFIX_ON_FILENAME = '2'


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    # Declare path for output graphs from this script
    make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_{stub}.png"  # noqa: E731

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

        # %% Define useful things for the plotting:
        dats = ['GBD', 'Model']
        sexes = ['F', 'M']

        all_causes = sorted(list(results.index.levels[3]), key=order_of_cause_of_death_label)

        sexname = lambda x: 'Females' if x == 'F' else 'Males'  # noqa: E731
        reformat_cause = lambda x: x.replace(' / ', '_')  # noqa: E731

        # %% Make figures of overall summaries of outcomes by cause

        for _i, sex in enumerate(sexes):

            fig, axes = plt.subplots(ncols=2, nrows=1, sharey=True, sharex=True)
            for row, dat in enumerate(dats):
                ax = axes[row]
                df = outcome_by_age_pt[dat].loc[sex].loc[:, pd.IndexSlice['mean']] / 1e3
                df = df[reversed(order_of_cause_of_death_label(df.columns))]  # Reverse the standard order so that
                #                                                               'Other' is at base

                df.plot.bar(stacked=True, ax=ax,
                            color=[get_color_cause_of_death_label(_label) for _label in df.columns])
                ax.set_xlabel('Age Group')
                ax.set_ylabel(f"{what} per year\n(thousands)")
                ax.set_title(f"{dat}")
                ax.get_legend().remove()
                ax.grid(axis='y')

            # Create figure legend and remove duplicated entries
            handles, labels = ax.get_legend_handles_labels()
            lgd = dict(zip(labels, handles))
            fig.legend(reversed(lgd.values()), reversed(lgd.keys()), loc="center right", ncol=1, fontsize=6)

            fig.suptitle(f'{sexname(sex)}', fontsize=18)
            fig.tight_layout()

            fig.savefig(make_graph_file_name(f"{what}_{period}_{sex}_StackedBars_ModelvsGBD"))
            fig.show()
            plt.close(fig)

        # %% Plots of age-breakdown of outcomes patten for each cause:

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
                plt.savefig(make_graph_file_name(
                    f"{what}_{period}_AgeAndSexSpecificLineGraph_{reformat_cause(cause)}")
                )
                plt.show()
                plt.close(fig)

            except KeyError:
                print(f"Could not produce plot for {what}: {reformat_cause(cause)}")

        # %% Plots comparing between model and actual across all ages and sexes:

        tot_outcomes_by_cause = pd.concat(
            {
                dat: outcome_by_age_pt[dat].sum(axis=0) for dat in outcome_by_age_pt.keys()
            }, axis=1
        )
        # todo N.B. For GBD, should really use all ages and all sex numbers from GBD to get correct uncertainty bounds
        #  (the addition of the bounds for the sub-categories - as done here - is not strictly correct.)
        #  ... OR use formula to make my own explicit assumption about correlation of uncertainty in different age-grps.

        select_labels = ['AIDS', 'Cancer (Other)', 'Measles', 'Other']

        fig, ax = plt.subplots()
        xylim = tot_outcomes_by_cause.loc[('upper', slice(None))].max().max() / 1e3
        line_x = np.linspace(0, xylim)
        ax.plot(line_x, line_x, 'k--')
        ax.fill_between(line_x, line_x*0.9, line_x*1.1, color='grey', alpha=0.5)
        ax.set(xlim=(0, xylim), ylim=(0, xylim))

        for cause in all_causes:

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

            ax.errorbar(x=x, y=y, xerr=xerr, yerr=yerr, label=cause, color=get_color_cause_of_death_label(cause))

            # add labels to selected points
            if cause in select_labels:
                ax.annotate(cause,
                            (x, y),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center'
                            )

        ax.set_xlabel('GBD (thousands)')
        ax.set_ylabel('Model (thousands)')
        ax.set_title(f'{what} per year by Cause {period}')
        ax.legend(ncol=1, prop={'size': 8}, loc='lower right')
        plt.savefig(make_graph_file_name(f"{what}_{period}_Scatter_Plot"))
        plt.show()
        plt.close(fig)

        # %% Assess the "coverage" of the model: i.e. the fraction of deaths/dalys that are causes that are represnted
        # the model.
        outcomes = outcome_by_age_pt['GBD'][("mean")]
        fraction_causes_modelled = 1.0 - outcomes['Other'] / outcomes.sum(axis=1)
        fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True)
        for sex, ax in zip(sexes, axs):
            fraction_causes_modelled.loc[(sex, slice(None))].droplevel(0).plot(
                ax=ax, color=get_color_cause_of_death_label('Other'), lw=5)
            ax.set_ylim(0, 1.0)
            xticks = fraction_causes_modelled.index.levels[1]
            ax.set_xticks(range(len(xticks)))
            ax.set_xticklabels(xticks, rotation=90)
            ax.grid(axis='y')
            ax.set_xlabel('Age-Group')
            ax.set_ylabel('Fraction')
            ax.set_title(f"{sexname(sex)}")
        fig.suptitle(f"Fraction of {what} Represented in the Model", fontsize=16)
        fig.tight_layout()
        plt.savefig(make_graph_file_name(f"{what}_{period}_coverage"))
        plt.show()
        plt.close(fig)

    # %% Make graphs for each of Deaths and DALYS for a specific period
    make_std_graphs(what='Deaths', period='2010-2014')
    # make_std_graphs(what='DALYs', period='2010-2014')  # <-- todo colormapping and order for DALYS

    # make_std_graphs(what='Deaths', period='2015-2019')
    # make_std_graphs(what='DALYs', period='2015-2019')  # <-- todo colormapping and order for DALYS


if __name__ == "__main__":
    outputspath = Path('./outputs/tbh03@ic.ac.uk')
    rfp = Path('./resources')

    # Find results folder (most recent run generated using that scenario_filename)
    scenario_filename = 'long_run_all_diseases.py'
    results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]

    apply(results_folder=results_folder, output_folder=results_folder, resourcefilepath=rfp)
