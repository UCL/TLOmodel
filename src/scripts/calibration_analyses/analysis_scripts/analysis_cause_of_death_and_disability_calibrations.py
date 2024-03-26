"""
Produce comparisons between model and GBD of deaths by cause in a particular period.

This uses the results of the Scenario defined in: src/scripts/long_run/long_run.py but it can edited to look at other
results (change 'scenario_filename').
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date
from tlo.analysis.utils import (
    CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP,
    extract_results,
    format_gbd,
    get_color_cause_of_death_or_daly_label,
    load_pickled_dataframes,
    make_age_grp_lookup,
    make_age_grp_types,
    make_calendar_period_lookup,
    make_calendar_period_type,
    order_of_cause_of_death_or_daly_label,
    plot_clustered_stacked,
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

    def make_std_graphs(what, period):
        """Make the standard Graphs for a specific period for either 'Deaths' or 'DALYS'"""

        assert isinstance(what, str)
        assert what in ('Deaths', 'DALYs')
        assert isinstance(period, str)
        assert period in make_calendar_period_lookup()[0]

        # limit to the subject of interest (either 'Deaths' or 'DALYS')
        gbd = gbd_all.loc[gbd_all['measure_name'] == what].copy()

        # %% Load modelling results:

        def get_counts_of_death_by_period_sex_agegrp_label(df):
            """Aggregate the model outputs into five-year periods for age and time"""
            _, agegrplookup = make_age_grp_lookup()
            _, calperiodlookup = make_calendar_period_lookup()
            df["year"] = df["date"].dt.year
            df["age_grp"] = df["age"].map(agegrplookup).astype(make_age_grp_types())
            df["period"] = df["year"].map(calperiodlookup).astype(make_calendar_period_type())
            return df.groupby(by=["period", "sex", "age_grp", "label"])["person_id"].count()

        def get_dalys_by_period_sex_agegrp_label(df):
            """Sum the dalys by period, sex, age-group and label"""
            _, calperiodlookup = make_calendar_period_lookup()

            df['age_grp'] = df['age_range'].astype(make_age_grp_types())
            df["period"] = df["year"].map(calperiodlookup).astype(make_calendar_period_type())
            df = df.drop(columns=['date', 'age_range', 'year'])
            df = df.groupby(by=["period", "sex", "age_grp"]).sum().stack()
            df.index = df.index.set_names('label', level=3)
            return df

        # Extract results, summing by sex, year, age & label
        if what == 'Deaths':
            results = extract_results(
                results_folder,
                module="tlo.methods.demography",
                key="death",
                custom_generate_series=get_counts_of_death_by_period_sex_agegrp_label,
                do_scaling=True
            )
        else:
            results = extract_results(
                results_folder,
                module="tlo.methods.healthburden",
                key="dalys_stacked_by_age_and_time",  # <-- for DALYS stacked by age and time
                custom_generate_series=get_dalys_by_period_sex_agegrp_label,
                do_scaling=True
            )

        # divide by five to give the average number of deaths per year within the five year period:
        results = results.div(5.0)

        # %% Load the cause-of-deaths mappers and use them to populate the 'label' for gbd outputs
        if what == 'Deaths':
            demoglog = load_pickled_dataframes(results_folder)['tlo.methods.demography']
            mapper_from_gbd_causes = pd.Series(
                demoglog['mapper_from_gbd_cause_to_common_label'].drop(columns={'date'}).loc[0]
            ).to_dict()
        else:
            hblog = load_pickled_dataframes(results_folder)['tlo.methods.healthburden']
            mapper_from_gbd_causes = pd.Series(
                hblog['daly_mapper_from_gbd_cause_to_common_label'].drop(columns={'date'}).loc[0]
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

        all_causes = sorted(list(results.index.levels[3]), key=order_of_cause_of_death_or_daly_label)

        sexname = lambda x: 'Females' if x == 'F' else 'Males'  # noqa: E731
        reformat_cause = lambda x: x.replace(' / ', '_')  # noqa: E731

        # %% Make figures of overall summaries of outcomes by cause
        def _sort_columns(df):
            """ Reverse the standard order of the columns so that 'Other' is at base"""
            return df.sort_index(axis=1, key=order_of_cause_of_death_or_daly_label, ascending=False)

        for sex in sexes:
            for scaled in (False, True):

                _dat = {_dat: outcome_by_age_pt[_dat].loc[sex].loc[:, pd.IndexSlice['mean']].pipe(_sort_columns)
                        for _dat in outcome_by_age_pt.keys()}

                # For scaled plots, zero-out models for age-groups where GBD is zero:
                if scaled:
                    _dat['Model'].loc[(_dat['GBD'].sum(axis=1) == 0.0)] = 0.0

                fig, ax = plt.subplots()
                plot_clustered_stacked(ax=ax,
                                       dfall=({k: v/1e3 for k, v in _dat.items()} if what == 'DALYs' else _dat),
                                       color_for_column_map=get_color_cause_of_death_or_daly_label,
                                       scaled=scaled,
                                       legends=False,
                                       H='',
                                       edgecolor='black',
                                       linewidth=0.4,
                                       )
                ax.set_title(f'{what}: {sexname(sex)}, {period}', fontsize=18)
                ax.set_xlabel('Age Group')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

                if scaled:
                    ax.set_ylim([0, 1.05])
                else:
                    ax.grid(axis='y')
                    if what == 'Deaths':
                        ax.set_ylabel(f"{what} per year\n")
                        ax.set_ylim([0, 25_000])
                        ax.set_yticks(np.arange(0, 30_000, 5_000))
                    else:
                        ax.set_ylabel(f"{what} per year (/1000)\n")
                        ax.set_ylim([0, 2000.0])

                # Create figure legend and remove duplicated entries, but keep the first entries
                handles, labels = ax.get_legend_handles_labels()
                lgd = dict()
                for k, v in zip(labels, handles):
                    lgd.setdefault(k, v)
                # ax.legend(reversed(lgd.values()), reversed(lgd.keys()), loc="upper right", ncol=2, fontsize=8)
                # ax.text(
                # 5.2, 11_000, 'GBD || Model', horizontalalignment='left',  verticalalignment='bottom', fontsize=8)
                ax.legend().set_visible(False)  # Hide legend

                fig.tight_layout()
                fig.savefig(make_graph_file_name(
                    f"{what}_{period}_{sex}_StackedBars_ModelvsGBD_{'scaled' if scaled else ''}"))
                plt.close(fig)

        # Simple pie-charts of just TLO estimates
        normalize_series = lambda ser: ser / ser.sum()  # noqa: E731

        def shift_row_to_top(df, index_to_shift):
            idx = [i for i in df.index if i != index_to_shift]
            return df.loc[[index_to_shift] + idx]

        fig, ax = plt.subplots(figsize=(10, 10))
        slices = normalize_series(
            outcome_by_age_pt['Model'].sum().loc['mean'].sort_values(ascending=True)
        )
        slices = shift_row_to_top(slices, 'Other')
        wedges, texts, autotexts = ax.pie(
            slices.values,
            labels=slices.index,
            colors=map(get_color_cause_of_death_or_daly_label, slices.index),
            startangle=90,
            autopct='%1.1f%%',
        )

        threshold = 3.0
        for label, pct_label in zip(texts, autotexts):
            pct_value = pct_label.get_text().rstrip('%')
            if float(pct_value) < threshold:
                label.set_text('')
                pct_label.set_text('')

        ax.set_title(f'TLO Model: {what}: {period}', fontsize=18)
        ax.legend(
                  title="Causes",
                  bbox_to_anchor=(0.9, -0.05),
                  ncol=2,
        )
        fig.tight_layout()
        fig.savefig(make_graph_file_name(f"{what}_{period}_PieChart_Model"))
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

                fig.patch.set_edgecolor(get_color_cause_of_death_or_daly_label(cause))
                fig.patch.set_linewidth(8)
                fig.tight_layout()
                fig.savefig(make_graph_file_name(
                    f"B_{what}_{period}_AgeAndSexSpecificLineGraph_{reformat_cause(cause)}")
                )
                plt.close(fig)

            except KeyError:
                print(f"Could not produce plot for {what}: {reformat_cause(cause)}")

        # %% "Scatter" Plots comparing between model and actual across all ages and sexes:

        tot_outcomes_by_cause = pd.concat(
            {
                dat: outcome_by_age_pt[dat].sum(axis=0) for dat in outcome_by_age_pt.keys()
            }, axis=1
        )
        # todo N.B. For GBD, would ideally use all ages and all sex numbers from GBD to get correct uncertainty bounds
        #  (the addition of the bounds for the sub-categories - as done here - might over-state the uncertainty.) This
        #  plot should be taken as indicative only.

        select_labels = []

        fig, ax = plt.subplots()
        xylim = tot_outcomes_by_cause.loc[('upper', slice(None))].max().max() / 1e3
        line_x = np.linspace(0, xylim)
        ax.plot(line_x, line_x, 'k--')
        # ax.fill_between(line_x, line_x*0.9, line_x*1.1, color='grey', alpha=0.5)  # grey ribbon around 1:1 line
        ax.set(xlim=(0, xylim), ylim=(0, xylim))

        for cause in all_causes:

            if (cause == 'Other') and (what == 'DALYs'):
                # Skip 'Other' when plotting DALYS as it's misleading. We don't have "Other" (non-modelled) causes
                # of disability.
                continue

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

            ax.errorbar(
                x=x,
                y=y,
                xerr=xerr,
                yerr=yerr,
                label=cause,
                color=get_color_cause_of_death_or_daly_label(cause)
            )

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
        ax.legend().set_visible(False)
        plt.savefig(make_graph_file_name(f"A_{what}_{period}_Scatter_Plot"))
        plt.close(fig)

        # %% Assess the "coverage" of the model: i.e. the fraction of deaths/dalys that are causes that are represented
        # the model.

        # Causes of death/dalys not in the model and the fraction of total deaths they cause
        unmodelled_causes = gbd \
            .loc[(period == period) & (gbd['measure_name'] == what)] \
            .assign(frac_cause=lambda df: df['mean'] / df['mean'].sum()) \
            .groupby(by=['cause_name', 'label'])['frac_cause'].sum() \
            .sort_values(ascending=False) \
            .pipe(lambda df: df.loc[(slice(None), "Other")])

        top_five_causes_not_modelled = ''.join([
            f"* {_cause} ({round(100 * _percent_deaths, 1)}%)\n"
            for _cause, _percent_deaths in unmodelled_causes[0:10].items() if _percent_deaths >= 0.005
        ])

        outcomes = outcome_by_age_pt['GBD'][("mean")]
        fraction_causes_modelled_overall = (1.0 - outcomes['Other'].sum() / outcomes.sum().sum())
        fraction_causes_modelled_by_sex_and_age = (1.0 - outcomes['Other'] / outcomes.sum(axis=1))
        fig, ax = plt.subplots()
        for sex in sexes:
            fraction_causes_modelled_by_sex_and_age.loc[(sex, slice(None))].plot(
                ax=ax,
                color=get_color_cause_of_death_or_daly_label('Other'),
                linestyle=':' if sex == 'F' else '-',
                label=sexname(sex),
                lw=5,
            )
        ax.axhline(fraction_causes_modelled_overall, color='b',
                   label=f'Overall: {round(100 * fraction_causes_modelled_overall)}%')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1.0)
        xticks = fraction_causes_modelled_by_sex_and_age.index.levels[1]
        ax.set_xticks(range(len(xticks)))
        ax.set_xticklabels(xticks, rotation=90)
        ax.grid(axis='y')
        ax.set_xlabel('Age-Group')
        ax.set_ylabel('Fraction')
        ax.set_title(f"Fraction of {what} Represented in the Model")
        ax.text(x=0.5, y=0.05, s=('Main causes not included explicitly:\n\n' + top_five_causes_not_modelled),
                bbox={'edgecolor': 'r', 'facecolor': 'w'})
        fig.tight_layout()
        plt.savefig(make_graph_file_name(f"C_{what}_{period}_coverage"))
        plt.close(fig)

        # Describe the burden with respect to wealth quintile:
        TARGET_PERIOD = (Date(2015, 1, 1), Date(2019, 12, 31))

        def get_total_num_dalys_by_wealth_and_label(_df):
            """Return the total number of DALYS in the TARGET_PERIOD by wealth and cause label."""
            wealth_cats = {5: '0-19%', 4: '20-39%', 3: '40-59%', 2: '60-79%', 1: '80-100%'}

            return _df \
                .loc[_df['year'].between(*[d.year for d in TARGET_PERIOD])] \
                .drop(columns=['date', 'year']) \
                .assign(
                    li_wealth=lambda x: x['li_wealth'].map(wealth_cats).astype(
                        pd.CategoricalDtype(wealth_cats.values(), ordered=True)
                    )
                ).melt(id_vars=['li_wealth'], var_name='label') \
                 .groupby(by=['li_wealth', 'label'])['value'] \
                 .sum()

        total_num_dalys_by_wealth_and_label = summarize(
            extract_results(
                results_folder,
                module="tlo.methods.healthburden",
                key="dalys_by_wealth_stacked_by_age_and_time",
                custom_generate_series=get_total_num_dalys_by_wealth_and_label,
                do_scaling=True,
            ),
            collapse_columns=True,
            only_mean=True,
        ).unstack()

        format_to_plot = total_num_dalys_by_wealth_and_label \
            .sort_index(axis=0) \
            .reindex(columns=CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP.keys(), fill_value=0.0) \
            .sort_index(axis=1, key=order_of_cause_of_death_or_daly_label)

        fig, ax = plt.subplots()
        name_of_plot = 'DALYS by Wealth and Cause, 2015-2019'
        (
            format_to_plot / 1e6
        ).plot.bar(stacked=True, ax=ax,
                   color=[get_color_cause_of_death_or_daly_label(_label) for _label in format_to_plot.columns],
                   )
        ax.axhline(0.0, color='black')
        ax.set_title(name_of_plot)
        ax.set_ylabel('Number of DALYs Averted (/1e6)')
        ax.set_ylim(0, 10)
        ax.set_xlabel('Wealth Percentile')
        ax.grid()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(ncol=3, fontsize=8, loc='upper right')
        ax.legend().set_visible(False)
        fig.tight_layout()
        fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
        plt.close(fig)

    # %% Make graphs for each of Deaths and DALYS for a specific period
    # make_std_graphs(what='Deaths', period='2010-2014')
    # make_std_graphs(what='DALYs', period='2010-2014')

    make_std_graphs(what='DALYs', period='2015-2019')
    make_std_graphs(what='Deaths', period='2015-2019')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources')
    )

    # apply(
    #     results_folder=Path("outputs/long_run_all_diseases-2024-03-05T114732Z"),
    #     output_folder=Path("outputs/long_run_all_diseases-2024-03-05T114732Z"),
    #     resourcefilepath=Path('./resources')
    # )
