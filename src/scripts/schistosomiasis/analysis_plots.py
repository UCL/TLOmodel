

from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import Patch
import ast

import pandas as pd
# import lacroix
import matplotlib.colors as colors
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from collections import defaultdict
import textwrap
from typing import Tuple
from matplotlib.lines import Line2D

from tlo import Date, Simulation, logging
from tlo.analysis.utils import (
    format_gbd,
    make_age_grp_types,
    parse_log_file,
    compare_number_of_deaths,
    extract_params,
    compute_summary_statistics,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    compute_summary_statistics,
    make_age_grp_lookup,
    make_age_grp_types,
    unflatten_flattened_multi_index_in_logging,
)

resourcefilepath = Path("./resources")

output_folder = Path("./outputs/t.mangal@imperial.ac.uk")

results_folder = get_scenario_outputs("schisto_scenarios-2025.py", output_folder)[-1]


# Declare path for output graphs from this script
def make_graph_file_name(name):
    return results_folder / f"Schisto_{name}.png"



#################################################################################
# %% multi-panel epi outputs
#################################################################################

# prevalence in 2050 of each species by strategy
prev_haem_national_plot = pd.read_excel(results_folder / ('prev_haem_national_summary 2024-2050.xlsx'))
prev_mansoni_national_plot = pd.read_excel(results_folder / ('prev_mansoni_national_summary 2024-2050.xlsx'))

# get heavy intensity infections also
prev_haem_national_heavy = pd.read_excel(results_folder / ('prev_haem_national_heavy_summary 2024-2050.xlsx'))
prev_mansoni_national_heavy = pd.read_excel(results_folder / ('prev_mansoni_national_heavy_summary 2024-2050.xlsx'))


# for the ranges plotted as error bars
prev_haem_HML_All_district_summary = pd.read_excel(results_folder / 'prev_haem_HML_All_district_summary 2024-2050.xlsx')
prev_mansoni_HML_All_district_summary = pd.read_excel(results_folder / 'prev_mansoni_HML_All_district_summary 2024-2050.xlsx')

# Filter to year 2050
df_2050 = prev_haem_HML_All_district_summary[prev_haem_HML_All_district_summary['year'] == 2050]
haem_extrema_by_draw = df_2050.groupby('draw')['mean'].agg(['min', 'max']).reset_index()
haem_extrema_by_draw.columns = ['draw', 'min_mean', 'max_mean']

df_2050 = prev_mansoni_HML_All_district_summary[prev_mansoni_HML_All_district_summary['year'] == 2050]
mansoni_extrema_by_draw = df_2050.groupby('draw')['mean'].agg(['min', 'max']).reset_index()
mansoni_extrema_by_draw.columns = ['draw', 'min_mean', 'max_mean']





def plot_species_prevalence_with_heavy(
    df: pd.DataFrame,
    extrema_df: pd.DataFrame,
    heavy_df: pd.DataFrame,
    species_name: str,
    ax,
    year: int = 2050,
    show_legend: bool = False,
    show_ylabel: bool = False
):
    """
    Plot prevalence bars with hatched overlay for heavy intensity prevalence.

    Parameters:
    - df: national prevalence summary with ['date', 'draw', 'mean']
    - extrema_df: min/max district-level summary with ['draw', 'min_mean', 'max_mean']
    - heavy_df: same structure as df but for heavy intensity infections
    - species_name: str for title
    - ax: matplotlib axis
    - year: target year for bar heights
    - show_legend: whether to include legends
    - show_ylabel: whether to include y-axis label
    """
    label_map = {
        'no MDA': 'no MDA',
        'MDA SAC': 'MDA SAC',
        'MDA PSAC+SAC': 'MDA PSAC+SAC',
        'MDA All': 'MDA All',
        'WASH only': 'WASH only'
    }
    colour_map = {
        'no MDA': '#1b9e77',
        'MDA SAC': '#d95f02',
        'MDA PSAC+SAC': '#7570b3',
        'MDA All': '#e7298a',
        'WASH only': '#e6ab02'
    }
    order = ['no MDA', 'MDA SAC', 'MDA PSAC+SAC', 'MDA All', 'WASH only']

    # Filter main data
    df_filtered = df[(df['date'] == year) & (df['draw'].str.contains('Continue WASH'))].copy()
    df_filtered['label'] = df_filtered['draw'].apply(
        lambda x: next((k for k in label_map if k in x), None)
    )
    df_filtered = df_filtered.dropna(subset=['label'])

    # Add WASH only
    df_extra = df[(df['date'] == year) & (df['draw'] == 'Scale-up WASH, no MDA')].copy()
    if not df_extra.empty:
        df_extra['label'] = 'WASH only'
        df_filtered = pd.concat([df_filtered, df_extra], ignore_index=True)

    # Order
    df_filtered['label'] = pd.Categorical(df_filtered['label'], categories=order, ordered=True)
    df_filtered = df_filtered.sort_values('label')

    # Draw lookup
    category_to_draw = df_filtered.set_index('label')['draw'].to_dict()
    draw_extrema = extrema_df.set_index('draw').to_dict(orient='index')

    # Bar heights and positions
    means = df_filtered['mean'].values
    labels = df_filtered['label'].values
    x = np.arange(len(labels))

    # Error bars
    error = np.array([
        [means[i] - draw_extrema[category_to_draw[l]]['min_mean'],
         draw_extrema[category_to_draw[l]]['max_mean'] - means[i]]
        for i, l in enumerate(labels)
    ]).T

    # Main bars
    bar_colors = [colour_map[l] for l in labels]
    ax.bar(x, means, yerr=error, capsize=5, color=bar_colors, edgecolor='black')

    # Overlay hatched bars for heavy prevalence
    for i, label in enumerate(labels):
        draw = category_to_draw[label]
        match = heavy_df[(heavy_df['date'] == year) & (heavy_df['draw'] == draw)]
        if not match.empty:
            heavy_val = match['mean'].values[0]
            ax.bar(
                x[i],
                heavy_val,
                width=0.8,
                color=bar_colors[i],
                edgecolor='black',
                hatch='///',
                linewidth=0.5
            )

    # Axis formatting
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title(f'{species_name}')
    ax.set_ylim(0, 0.5)
    if show_ylabel:
        ax.set_ylabel('Mean Prevalence')
    else:
        ax.set_ylabel('')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Legend
    if show_legend:
        legend_elements = [
            Line2D([0], [0], color='black', linewidth=1.5, label='min–max',
                   marker='|', markersize=15, linestyle='None'),
            Patch(facecolor='white', edgecolor='black', hatch='///', label='Heavy intensity')
        ]
        ax.legend(handles=legend_elements, loc='upper right', frameon=False)




fig, axes = plt.subplots(1, 2, figsize=(9, 6), sharey=True)

plot_species_prevalence_with_heavy(
    df=prev_haem_national_plot,
    extrema_df=haem_extrema_by_draw,
    heavy_df=prev_haem_national_heavy,
    species_name='Haematobium',
    ax=axes[0],
    show_ylabel=True
)

plot_species_prevalence_with_heavy(
    df=prev_mansoni_national_plot,
    extrema_df=mansoni_extrema_by_draw,
    heavy_df=prev_mansoni_national_heavy,
    species_name='Mansoni',
    ax=axes[1],
    show_legend=True
)

plt.tight_layout()
plt.savefig(results_folder / "prevalence_heavy_infection.png", dpi=300)

plt.show()









#################################################################################
# %% Person-years infected
#################################################################################


pc_py_averted = pd.read_excel(results_folder / "pc_py_averted_age_2024-2050.xlsx",
                              index_col=0,
                              header=[0, 1])



def plot_pc_py_averted_ci(df, draw_order, xlabel_labels=None, ylabel=None, ylim=None):
    """
    Plot percentage person-years averted with confidence intervals.

    Parameters:
    - df: pandas DataFrame with multi-index columns (draw, stat),
          index as age groups.
    - draw_order: list of draw names in order to plot.
    - xlabel_labels: list of x-axis labels matching draw_order (optional).
    - ylabel: y-axis label (optional).
    - ylim: y-axis limit (optional).
    """

    # Define plotting and legend order
    ordered_age_groups = ['Infants+PSAC', 'SAC', 'Adults']
    legend_labels = ['PSAC', 'SAC', 'Adults']

    palette = sns.color_palette("Set2", len(ordered_age_groups))
    stagger = np.linspace(-0.2, 0.2, len(ordered_age_groups))
    x_positions = np.arange(len(draw_order))

    fig, ax = plt.subplots(figsize=(1.8 * len(draw_order), 4))

    for i, draw in enumerate(draw_order):
        for j, age_group in enumerate(ordered_age_groups):
            if age_group not in df.index:
                continue

            central = df.loc[age_group, (draw, 'central')]
            lower = df.loc[age_group, (draw, 'lower')]
            upper = df.loc[age_group, (draw, 'upper')]

            yerr_lower = central - lower
            yerr_upper = upper - central

            ax.errorbar(
                x=i + stagger[j],
                y=central,
                yerr=[[yerr_lower], [yerr_upper]],
                fmt='o',
                color=palette[j],
                capsize=5,
                label=legend_labels[j] if i == 0 else ""
            )

    # Vertical dashed lines between draws
    for boundary in range(1, len(draw_order)):
        ax.axvline(boundary - 0.5, color='grey', linestyle='--', linewidth=0.8)

    ax.axhline(0, color='grey', linestyle='-', linewidth=0.8)
    ax.set_xticks(x_positions)

    # Use custom x-axis labels if provided, else use draw_order
    if xlabel_labels is not None:
        ax.set_xticklabels(xlabel_labels, rotation=30, ha='right')
    else:
        ax.set_xticklabels(draw_order, rotation=45, ha='right')

    ax.set_xlim(-0.5, len(draw_order) - 0.5)
    ax.set_ylim(-2, ylim if ylim is not None else ax.get_ylim()[1])
    ax.set_ylabel(ylabel if ylabel else "% person-years averted")

    handles = [plt.Line2D([0], [0], marker='o', color=palette[i], linestyle='', label=legend_labels[i])
               for i in range(len(ordered_age_groups))]
    ax.legend(handles=handles, title='Age Group', loc='upper right')

    plt.tight_layout()
    return fig

# Usage:
draw_order = [
    'Continue WASH, MDA SAC',
    'Continue WASH, MDA PSAC+SAC',
    'Continue WASH, MDA All',
    'Scale-up WASH, no MDA'
]

x_labels = ['MDA SAC', 'MDA PSAC+SAC', 'MDA All', 'WASH only']

fig = plot_pc_py_averted_ci(
    df=pc_py_averted,
    draw_order=draw_order,
    xlabel_labels=x_labels,
    ylabel='% person-years averted',
    ylim=100
)
fig.savefig(results_folder / "pc_py_averted.png", dpi=300)
plt.show()



#################################################################################
# %% DALYS AVERTED
#################################################################################




dalys_averted = pd.read_excel(
    results_folder / 'dalys_averted_district_compared_noMDA2024-2050.xlsx',
    index_col=[0, 1],    # First two columns as MultiIndex: year, district
    header=[0, 1]        # Two-level column MultiIndex: run/draw
)
dalys_averted.index.set_names(['year', 'district'], inplace=True)

# add in the dalys averted for Scale-up WASH, no MDA compared with Continue WASH, no MDA
dalys_averted_compared_ContinueWASHnoMDA = pd.read_excel(
    results_folder / 'dalys_averted_district_compared_continueWASHnoMDA2024-2050.xlsx',
    index_col=[0, 1],    # First two columns as MultiIndex: year, district
    header=[0, 1]        # Two-level column MultiIndex: run/draw
)




def summarise_dalys_averted(df, start_year=2024, end_year=2050):
    """
    Summarise DALYs averted between `start_year` and `end_year` by draw:
    - Sum DALYs across all districts and years for each (run, draw)
    - Compute mean and standard error across runs for each draw
    """

    # Ensure index names are set
    df.index.set_names(['year', 'district'], inplace=True)

    # Filter years
    years = df.index.get_level_values('year')
    df_filtered = df[(years >= start_year) & (years <= end_year)]

    # Sum across years and districts, grouped by (run, draw)
    summed = df_filtered.sum(axis=0)

    # Convert Series to DataFrame with MultiIndex (run, draw)
    summed_df = summed.to_frame(name='dalys').reset_index().set_index(['run', 'draw'])

    # Unstack so that each draw is a column, rows = runs
    matrix = summed_df['dalys'].unstack(level='draw')

    # Calculate summary statistics
    mean = matrix.mean(axis=0)
    se = matrix.sem(axis=0)

    # Combine into single DataFrame
    summary = pd.concat([mean, se], axis=1)
    summary.columns = ['mean', 'se']

    return summary


dalys_summary = summarise_dalys_averted(dalys_averted)

dalys_summary.to_excel(results_folder / "dalys_averted_summary_2024_2050.xlsx")

# add in the row for Scale-up WASH, no MDA
dalys_averted_compared_ContinueWASHnoMDA_summary = summarise_dalys_averted(dalys_averted_compared_ContinueWASHnoMDA)
row_to_add = dalys_averted_compared_ContinueWASHnoMDA_summary.loc['Scale-up WASH, no MDA']
dalys_summary = pd.concat([dalys_summary, row_to_add.to_frame().T])




def plot_dalys_averted_bar(summary_df):
    # Define the draw order and corresponding labels
    draw_order = [
        'Continue WASH, MDA SAC',
        'Continue WASH, MDA PSAC+SAC',
        'Continue WASH, MDA All',
        'Scale-up WASH, no MDA'
    ]
    label_map = {
        'Continue WASH, MDA SAC': 'MDA SAC',
        'Continue WASH, MDA PSAC+SAC': 'MDA PSAC+SAC',
        'Continue WASH, MDA All': 'MDA All',
        'Scale-up WASH, no MDA': 'WASH only'
    }
    colour_map = {
        'no MDA': '#1b9e77',
        'MDA SAC': '#d95f02',
        'MDA PSAC+SAC': '#7570b3',
        'MDA All': '#e7298a',
        'WASH only': '#e6ab02'
    }

    # Filter and prepare the data
    summary_plot = summary_df.loc[draw_order].copy()
    summary_plot['label'] = summary_plot.index.map(label_map)
    summary_plot['lower'] = summary_plot['mean'] - 1.96 * summary_plot['se']
    summary_plot['upper'] = summary_plot['mean'] + 1.96 * summary_plot['se']
    summary_plot['color'] = summary_plot['label'].map(colour_map)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        summary_plot['label'],
        summary_plot['mean'],
        yerr=1.96 * summary_plot['se'],
        capsize=5,
        color=summary_plot['color'],
        edgecolor='black'
    )

    ax.set_ylabel('DALYs averted')
    ax.set_title('')
    ax.axhline(0, color='grey', linewidth=0.8)
    plt.tight_layout()
    return fig


fig = plot_dalys_averted_bar(dalys_summary)
plt.savefig(results_folder / f"dalys_averted.png", dpi=300)
plt.show()


#################################################################################
# %% Prevalence heatmap
#################################################################################


def plot_prevalence_heatmap(df, year=2050, threshold=1.5, filename=None):
    # Extract data for the given year
    df_year = df.loc[year]

    # Mean over runs if columns have a 'run' level
    if isinstance(df_year.columns, pd.MultiIndex) and 'run' in df_year.columns.names:
        mean_df = df_year.groupby(level='draw', axis=1).mean()
    else:
        mean_df = df_year.copy()

    draw_labels = mean_df.columns.tolist()

    # Parse draw labels into Phase and MDA parts
    phase_labels, mda_labels = [], []
    for label in draw_labels:
        try:
            phase_part, mda_part = label.split(', ')
        except Exception:
            phase_part, mda_part = label, ''
        phase_clean = phase_part.replace(' WASH', '')
        phase_labels.append(phase_clean)
        mda_labels.append(mda_part)

    # Keep only draws where phase == 'Continue'
    keep_mask = [p == "Continue" for p in phase_labels]
    mean_df = mean_df.loc[:, keep_mask]
    phase_labels = [p for p, k in zip(phase_labels, keep_mask) if k]
    mda_labels = [m for m, k in zip(mda_labels, keep_mask) if k]
    draw_labels = [d for d, k in zip(draw_labels, keep_mask) if k]

    # Define desired orders (phases reduced to just 'Continue')
    phase_order = ['Continue']
    mda_order = ['no MDA', 'MDA SAC', 'MDA PSAC+SAC', 'MDA All']

    # Sorting helper
    col_df = pd.DataFrame({'phase': phase_labels, 'mda': mda_labels, 'orig': draw_labels})
    col_df['phase_order'] = col_df['phase'].apply(lambda x: phase_order.index(x) if x in phase_order else 99)
    col_df['mda_order'] = col_df['mda'].apply(lambda x: mda_order.index(x) if x in mda_order else 99)
    col_df = col_df.sort_values(by=['phase_order', 'mda_order']).reset_index(drop=True)

    multi_cols = pd.MultiIndex.from_arrays(
        [col_df['phase'], col_df['mda']],
        names=['Phase', 'MDA']
    )

    mean_df = mean_df[col_df['orig']]
    mean_df.columns = multi_cols

    plt.figure(figsize=(14, 10))
    ax = sns.heatmap(
        mean_df,
        cmap='coolwarm',
        cbar_kws={'label': 'Mean prevalence'},
        linewidths=0.5,
        linecolor='gray',
    )

    # Add hatch outline if below threshold
    for y in range(mean_df.shape[0]):
        for x in range(mean_df.shape[1]):
            val = mean_df.iloc[y, x]
            if val < threshold:
                ax.add_patch(
                    plt.Rectangle(
                        (x, y),
                        1, 1,
                        fill=False,
                        edgecolor='grey',
                        lw=1.5,
                        hatch='//'
                    )
                )

    ax.set_ylabel('District')
    plt.title(f'Mean Prevalence by District, Year {year}')

    # ----------- X-axis labels ------------------
    tick_positions = [i + 0.5 for i in range(len(mean_df.columns))]
    tick_labels = col_df['mda'].tolist()
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax.set_xlabel('')

    # # Draw phase label (only Continue now)
    # n_mda = len(mda_order)
    # start, end = 0, len(mean_df.columns) - 1
    # mid = (start + end) / 2 + 0.5
    # ax.text(
    #     x=mid,
    #     y=-0.15,
    #     s="Continue",
    #     ha='center',
    #     va='top',
    #     fontsize=12,
    #     fontweight='bold',
    #     color='black',
    #     transform=ax.get_xaxis_transform()
    # )

    plt.subplots_adjust(bottom=0.2, top=0.9)
    if filename:
        plt.savefig(results_folder / filename, dpi=300)
    plt.show()



path = Path(results_folder / f'prev_haem_H_year_district 2024-2050.xlsx')
prev_haem_H_All_district = pd.read_excel(path, index_col=[0, 1])  # assuming first two columns are index

path2 = Path(results_folder / f'prev_mansoni_H_year_district 2024-2050.xlsx')
prev_mansoni_H_All_district = pd.read_excel(path2, index_col=[0, 1])  # assuming first two columns are index


plot_prevalence_heatmap(prev_haem_H_All_district, year=2050, threshold=0.015, filename='prev_haem_H_district2050.png')
plot_prevalence_heatmap(prev_mansoni_H_All_district, year=2050, threshold=0.015, filename='prev_mansoni_H_district2050.png')


# HML infections
path = Path(results_folder / 'prev_haem_HML_All_district 2024-2050.xlsx')
prev_haem_HML_All_district = pd.read_excel(path, index_col=[0, 1])  # assuming first two columns are index

path2 = Path(results_folder / 'prev_mansoni_HML_All_district 2024-2050.xlsx')
prev_mansoni_HML_All_district = pd.read_excel(path2, index_col=[0, 1])  # assuming first two columns are index


plot_prevalence_heatmap(prev_haem_HML_All_district, year=2050, threshold=0.01, filename='prev_haem_HML_district2050.png')
plot_prevalence_heatmap(prev_mansoni_HML_All_district, year=2050, threshold=0.01, filename='prev_mansoni_HML_district2050.png')



#################################################################################
# %% PREVALENCE OVER TIME
#################################################################################


file_path = results_folder / 'prev_mansoni_national_summary 2024-2050.xlsx'
mansoni_prev = pd.read_excel(file_path)

file_path = results_folder / 'prev_haem_national_summary 2024-2050.xlsx'
haem_prev = pd.read_excel(file_path)




def plot_combined_prevalence(mansoni_df, haem_df, start_year=2022, xtick_step=2):
    """
    Plot combined prevalence for mansoni and haematobium infections (all ages),
    using flat dataframes with columns:
    ['Unnamed: 0', 'date', 'draw', 'mean', 'lower_ci', 'upper_ci'].
    Only plots scenarios where 'draw' contains 'Continue WASH'.

    Parameters
    ----------
    mansoni_df : pd.DataFrame
        Data for S. mansoni.
    haem_df : pd.DataFrame
        Data for S. haematobium.
    start_year : int
        First year to include in the plots (default=2022).
    """

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12), sharex=True)

    # Colours by MDA strategy
    mda_colours = {
        'no MDA': '#1b9e77',
        'MDA SAC': '#d95f02',
        'MDA PSAC+SAC': '#7570b3',
        'MDA All': '#e7298a'
    }

    def get_colour(draw):
        for key in mda_colours:
            if key in draw:
                return mda_colours[key]
        return '#000000'

    # Filter to "Continue WASH" scenarios
    def filter_continue(df):
        return df[df["draw"].str.contains("Continue WASH")]

    def plot_on_ax(df, ax, title):
        df = filter_continue(df)
        df = df[df["date"] >= start_year]  # restrict by start_year
        for draw_name in df["draw"].unique():
            sub = df[df["draw"] == draw_name]
            ax.plot(sub["date"], sub["mean"],
                    label=draw_name.replace("Continue WASH, ", ""),
                    color=get_colour(draw_name), linestyle="-", lw=1.5)
            ax.fill_between(sub["date"], sub["lower_ci"], sub["upper_ci"],
                            color=get_colour(draw_name), alpha=0.3)
        ax.set_title(title)
        ax.set_ylabel("Prevalence")
        ax.set_ylim(0, 0.4)

        # Limit ticks
        years = sorted(df["date"].unique())
        ax.set_xticks(years[::xtick_step])
        ax.set_xticklabels(years[::xtick_step])

    # Mansoni
    plot_on_ax(
        mansoni_df,
        axes[0],
        "Mansoni prevalence"
    )

    # Haem
    plot_on_ax(
        haem_df,
        axes[1],
        "Haematobium prevalence"
    )
    axes[1].set_xlabel("Year")

    # Legends: only MDA strategies matter now
    colour_legend = [
        Line2D([0], [0], color=col, lw=1.5, linestyle="-")
        for col in mda_colours.values()
    ]
    colour_labels = list(mda_colours.keys())
    axes[0].legend(colour_legend, colour_labels, title="MDA Strategy")

    fig.tight_layout()
    return fig



fig = plot_combined_prevalence(mansoni_prev, haem_prev, start_year=2022)
fig.savefig(make_graph_file_name("prevalence_over_time_all_ages"))
plt.show()




#################################################################################
# %% ICERS
#################################################################################


def plot_icer_three_panels(df, context="Continue_WASH"):
    """
    Plot ICER by district for three categories ('MDA SAC', 'MDA PSAC', 'MDA All')
    Only rows where 'wash_strategy' contains the specified context are included.
    """
    # Filter for the specified WASH context
    df_filtered = df[df['wash_strategy'].str.contains(context, na=False)]

    categories = ['MDA SAC', 'MDA PSAC+SAC', 'MDA All']
    titles = {
        'MDA SAC': f'{context} MDA SAC',
        'MDA PSAC+SAC': 'MDA PSAC+SAC',
        'MDA All': 'MDA All'
    }

    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharey=True)

    for ax, category in zip(axes, categories):
        # Subset data for each intervention group
        subset = df_filtered[df_filtered['comparison'].str.contains(category, na=False)]

        if subset.empty:
            ax.text(0.5, 0.5, f'No data for {category}', ha='center', va='center')
            ax.set_title(titles[category])
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Sort by district to ensure consistent plotting
        subset = subset.sort_values('District')

        # Plot points with seaborn
        sns.pointplot(
            data=subset,
            x='District',
            y='mean',
            join=False,
            color='blue',
            ax=ax,
            order=subset['District'].unique()
        )

        # Add error bars
        x_vals = range(len(subset))
        y_vals = subset['mean'].values
        y_err_lower = y_vals - subset['lower'].values
        y_err_upper = subset['upper'].values - y_vals

        ax.errorbar(
            x=x_vals,
            y=y_vals,
            yerr=[y_err_lower, y_err_upper],
            fmt='none',
            ecolor='blue',
            elinewidth=1,
            capsize=3,
            alpha=0.7
        )

        ax.axhline(500, color='grey', linestyle='--', linewidth=1)
        ax.set_ylim(0, None)
        ax.set_title(titles[category])

        if category != 'MDA All':
            ax.set_xlabel('')
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('District')
            ax.set_xticklabels(subset['District'], rotation=45, ha='right')

        ax.set_ylabel('ICER')

    plt.tight_layout()
    plt.savefig(results_folder / f"ICERs_{context}.png", dpi=300)
    plt.show()


file_path = results_folder / f'icer_district_cons_only_2024-2050.xlsx'
icer_district_df = pd.read_excel(file_path)
icer_district_df['draw'] = icer_district_df['comparison'].str.extract(r'^(.*?)\s+vs')

plot_icer_three_panels(icer_district_df, context='Continue WASH')

plot_icer_three_panels(icer_district_df, context='Scale-up WASH')


#################################################################################
# %% PLOTS DALYS VS COSTS
#################################################################################


file_path = results_folder / f'sum_incremental_dalys_averted_district2024-2050.xlsx'
dalys_district_df = pd.read_excel(file_path, header=[0, 1, 2], index_col=0)

file_path = results_folder / f'sum_incremental_full_costs_incurred_district2024-2050.xlsx'
full_costs_district_df = pd.read_excel(file_path, header=[0, 1, 2], index_col=0)
full_costs_district_df.columns = full_costs_district_df.columns.droplevel(2)

file_path = results_folder / f'sum_incremental_cons_costs_incurred_district2024-2050.xlsx'
cons_costs_district_df = pd.read_excel(file_path, header=[0, 1, 2], index_col=0)
cons_costs_district_df.columns = cons_costs_district_df.columns.droplevel(2)




def plot_dalys_vs_costs_by_district_with_thresholds(
    dalys_district_df: pd.DataFrame,
    costs_district_df: pd.DataFrame,
    wash_strategy: str,
    comparison: str,
    plot_summary: bool = True,
    thresholds: list[float] = [500.0],
    scale_x=0.5,
    scale_y=0.5,
):
    """
    Plot DALYs averted (x-axis) vs incremental costs (y-axis) by district
    for a specified WASH strategy and comparison. Points are coloured by
    cost-effectiveness based on the first threshold provided.

    Multiple threshold lines are plotted if specified.
    """
    try:
        dalys_sub = dalys_district_df.xs(wash_strategy, axis=1, level='wash_strategy')[comparison]
        costs_sub = costs_district_df.xs(wash_strategy, axis=1, level='wash_strategy')[comparison]
    except KeyError as e:
        raise KeyError(f"Specified key not found in DataFrame columns: {e}")

    districts = dalys_sub.index
    plt.figure(figsize=(10, 8))

    if plot_summary:
        mean_dalys = dalys_sub.mean(axis=1)
        se_dalys = dalys_sub.std(axis=1, ddof=1) / np.sqrt(dalys_sub.shape[1])
        ci_dalys = 1.96 * se_dalys

        mean_costs = costs_sub.mean(axis=1)
        se_costs = costs_sub.std(axis=1, ddof=1) / np.sqrt(costs_sub.shape[1])
        ci_costs = 1.96 * se_costs

        # Plot district points with error bars
        for district in districts:
            plt.errorbar(
                mean_dalys[district],
                mean_costs[district],
                xerr=ci_dalys[district],
                yerr=ci_costs[district],
                fmt='o',
                capsize=5,
                markersize=6,
                ecolor='#cccccc',  # light grey error bars
                elinewidth=1.5,
                markerfacecolor='white',
                markeredgecolor='black'
            )
            plt.text(
                mean_dalys[district],
                mean_costs[district],
                district,
                fontsize=9,
                color='black',  # ensure label text is black
                alpha=0.8,
                ha='right',
                va='bottom'
            )

        # Axis limits
        x_min, x_max = mean_dalys.min(), mean_dalys.max()
        y_min, y_max = mean_costs.min(), mean_costs.max()
        x_pad = scale_x * (x_max - x_min) if x_max > x_min else 1
        y_pad = scale_y * (y_max - y_min) if y_max > y_min else 1
        x_lims = (x_min - x_pad, x_max + x_pad)
        y_lims = (y_min - y_pad, y_max + y_pad)

        plt.xlim(x_lims)
        plt.ylim(y_lims)

        # Plot ICER threshold lines
        x_line = np.linspace(*x_lims, 100)
        for threshold in thresholds:
            y_line = threshold * x_line
            plt.plot(
                x_line,
                y_line,
                linestyle='--',
                linewidth=1.2,
                label=f"ICER = ${threshold:.0f}/DALY"
            )

        plt.legend(loc='best')

    else:
        num_runs = dalys_sub.shape[1]
        colours = plt.cm.viridis(np.linspace(0, 1, num_runs))

        for run_idx in range(num_runs):
            plt.scatter(
                dalys_sub.iloc[:, run_idx],
                costs_sub.iloc[:, run_idx],
                label=f'Run {run_idx + 1}',
                color=colours[run_idx],
                alpha=0.7,
                s=40,
                edgecolors='none'
            )

        mean_dalys = dalys_sub.mean(axis=1)
        mean_costs = costs_sub.mean(axis=1)
        for district, x, y in zip(districts, mean_dalys, mean_costs):
            plt.text(
                x,
                y,
                district,
                fontsize=9,
                color='black',
                alpha=0.8,
                ha='right',
                va='bottom'
            )

        # Axis limits
        x_min, x_max = mean_dalys.min(), mean_dalys.max()
        y_min, y_max = mean_costs.min(), mean_costs.max()
        x_pad = scale_x * (x_max - x_min) if x_max > x_min else 1
        y_pad = scale_y * (y_max - y_min) if y_max > y_min else 1
        x_lims = (x_min - x_pad, x_max + x_pad)
        y_lims = (y_min - y_pad, y_max + y_pad)
        plt.xlim(x_lims)
        plt.ylim(y_lims)

        x_line = np.linspace(*x_lims, 100)
        for threshold in thresholds:
            y_line = threshold * x_line
            plt.plot(
                x_line,
                y_line,
                linestyle='--',
                linewidth=1.2,
                label=f"ICER = ${threshold:.0f}/DALY"
            )

        plt.legend(loc='best')

    plt.xlabel('DALYs Averted')
    plt.ylabel('Incremental Costs (USD)')
    plt.title(f'{comparison}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(results_folder / f"cost_v_dalys_{wash_strategy}_{comparison}.png", dpi=300)
    plt.show()






plot_dalys_vs_costs_by_district_with_thresholds(
    dalys_district_df=dalys_district_df,
    costs_district_df=full_costs_district_df,
    wash_strategy='Continue WASH',
    comparison='MDA SAC vs no MDA',
    plot_summary=True,
    thresholds=[5,10,20,50,61],
    scale_x=0.5,
    scale_y=0.5,
)



plot_dalys_vs_costs_by_district_with_thresholds(
    dalys_district_df=dalys_district_df,
    costs_district_df=full_costs_district_df,
    wash_strategy='Continue WASH',
    comparison='MDA PSAC+SAC vs MDA SAC',
    plot_summary=True,
    thresholds=[5,10,20,50,61],
    scale_x=0.5,
    scale_y=0.5,
)




plot_dalys_vs_costs_by_district_with_thresholds(
    dalys_district_df=dalys_district_df,
    costs_district_df=full_costs_district_df,
    wash_strategy='Continue WASH',
    comparison='MDA All vs MDA PSAC+SAC',
    plot_summary=True,
    thresholds=[5,10,20,50,61],
    scale_x=0.5,
    scale_y=0.5,
)




# cons only
plot_dalys_vs_costs_by_district_with_thresholds(
    dalys_district_df=dalys_district_df,
    costs_district_df=cons_costs_district_df,
    wash_strategy='Continue WASH',
    comparison='MDA SAC vs no MDA',
    plot_summary=True,
    thresholds=[5,10,20,50,61],
    scale_x=0.5,
    scale_y=0.5)



plot_dalys_vs_costs_by_district_with_thresholds(
    dalys_district_df=dalys_district_df,
    costs_district_df=cons_costs_district_df,
    wash_strategy='Continue WASH',
    comparison='MDA PSAC+SAC vs MDA SAC',
    plot_summary=True,
    thresholds=[5,10,20,50,61],
    scale_x=0.5,
    scale_y=0.5,)



plot_dalys_vs_costs_by_district_with_thresholds(
    dalys_district_df=dalys_district_df,
    costs_district_df=cons_costs_district_df,
    wash_strategy='Continue WASH',
    comparison='MDA All vs MDA PSAC+SAC',
    plot_summary=True,
    thresholds=[5,10,20,50,61],
    scale_x=0.5,
    scale_y=0.5,)




#################################################################################
# %% KAPLAN-MEIER time to elimination
#################################################################################

# heavy infections
path = Path(results_folder) / "prev_haem_H_year_district 2024-2050.xlsx"

prev_haem_H_All_district = pd.read_excel(
    path,
    index_col=[0, 1],
    header=[0, 1]  # explicitly tell pandas the first two rows are headers
)
prev_haem_H_All_district.columns.names = ["draw", "run"]


path2 = Path(results_folder) / "prev_mansoni_H_year_district 2024-2050.xlsx"

prev_mansoni_H_All_district = pd.read_excel(
    path2,
    index_col=[0, 1],
    header=[0, 1]  # explicitly tell pandas the first two rows are headers
)
prev_mansoni_H_All_district.columns.names = ["draw", "run"]


# all infections HML
path3 = Path(results_folder) / "prev_haem_HML_All_district 2024-2050.xlsx"

prev_haem_HML_All_district = pd.read_excel(
    path3,
    index_col=[0, 1],
    header=[0, 1]  # explicitly tell pandas the first two rows are headers
)
prev_haem_HML_All_district.columns.names = ["draw", "run"]


path4 = Path(results_folder) / "prev_mansoni_HML_All_district 2024-2050.xlsx"

prev_mansoni_HML_All_district = pd.read_excel(
    path4,
    index_col=[0, 1],
    header=[0, 1]  # explicitly tell pandas the first two rows are headers
)
prev_mansoni_HML_All_district.columns.names = ["draw", "run"]








def plot_ephp_km_continue(
    df: pd.DataFrame,
    threshold: float = 0.01,
    year_range: tuple = (2020, 2050),
    alpha: float = 1.0,
    figsize: tuple = (10, 6),
    species=None,
    ci: tuple = (0.025, 0.975),   # confidence interval quantiles
):
    """
    Plot Kaplan–Meier style curves showing the proportion of districts
    reaching prevalence < threshold by year, with mean lines and CI bands.

    Rows: MultiIndex (year, district)
    Columns: MultiIndex (draw, run)
    """

    mda_colours = {
        'no MDA': '#1b9e77',      # Teal
        'MDA SAC': '#d95f02',     # Orange
        'MDA PSAC+SAC': '#7570b3',# Purple
        'MDA All': '#e7298a',     # Pink
        'WASH only': '#e6ab02'    # Mustard Yellow
    }

    draw_labels = {
        'Continue WASH, no MDA': 'no MDA',
        'Continue WASH, MDA SAC': 'MDA SAC',
        'Continue WASH, MDA PSAC+SAC': 'MDA PSAC+SAC',
        'Continue WASH, MDA All': 'MDA All',
        'Scale-up WASH, no MDA': 'WASH only',
    }

    # Restrict to plotting window
    years_idx = df.index.get_level_values("year")
    df = df[(years_idx >= year_range[0]) & (years_idx <= year_range[1])]
    total_districts = df.index.get_level_values("district").nunique()
    years = list(range(year_range[0], year_range[1] + 1))

    plt.figure(figsize=figsize)

    for draw, label in draw_labels.items():
        if draw not in df.columns.get_level_values("draw"):
            continue

        runs = df[draw].columns
        trajectories = []

        for run in runs:
            s = df[(draw, run)]
            b = s < threshold

            if not b.any():
                # never below threshold → flat 0
                y = [0.0] * len(years)
            else:
                fy = (
                    b[b].reset_index()
                      .groupby("district")["year"]
                      .min()
                )
                counts_by_year = fy.value_counts().reindex(years, fill_value=0).sort_index()
                prop = counts_by_year.cumsum() / total_districts
                y = prop.reindex(years, fill_value=0.0).tolist()

            trajectories.append(y)

        trajectories = np.array(trajectories)  # shape (n_runs, n_years)

        # Mean and CI
        mean_vals = trajectories.mean(axis=0)
        lower = np.quantile(trajectories, ci[0], axis=0)
        upper = np.quantile(trajectories, ci[1], axis=0)

        # Plot mean line
        plt.step(
            years, mean_vals,
            where="post",
            label=label,
            color=mda_colours.get(label, "grey"),
            linewidth=1.8,
            alpha=alpha,
        )

        # Plot CI band
        plt.fill_between(
            years, lower, upper,
            step="post",
            color=mda_colours.get(label, "grey"),
            alpha=0.2,
        )

    plt.ylabel(f"Proportion < {threshold * 100:.1f}%")
    plt.xlabel("Year")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, axis="y", color="grey", linestyle="-", linewidth=0.5, alpha=0.15)
    plt.legend(title="Strategy", loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.savefig(results_folder / f"ephp_km_plot_{species}_{threshold}_continue.png", dpi=300)
    plt.show()




#
# plot_ephp_km_continue(prev_haem_H_All_district, species='haem', threshold=0.01)
#
# plot_ephp_km_continue(prev_mansoni_H_All_district, species='mansoni', threshold=0.001)
#



plot_ephp_km_continue(prev_haem_HML_All_district, species='haem', threshold=0.02)


plot_ephp_km_continue(prev_mansoni_HML_All_district, species='mansoni', threshold=0.02)


# todo this should be updated as function above
# # get the figures for the SI separately
# def plot_ephp_km_pause(
#     df: pd.DataFrame,
#     threshold: float = 0.015,
#     year_range: tuple = (2024, 2040),
#     alpha: float = 1.0,
#     figsize: tuple = (10, 6),
#         species=None
# ):
#     """
#     Plot Kaplan-Meier style curves in a single panel showing the proportion of districts
#     reaching prevalence < threshold by year, for:
#         - Each MDA strategy under 'Continue WASH'
#         - 'no MDA' under 'Scale-up WASH' as 'WASH only'
#     """
#
#     mda_colours = {
#         'no MDA': '#1b9e77',  # Teal
#         'MDA SAC': '#d95f02',  # Orange
#         'MDA PSAC': '#7570b3',  # Purple
#         'MDA All': '#e7298a',  # Pink
#         'WASH only': '#e6ab02'  # Mustard Yellow – distinct from teal
#     }
#
#     def extract_mda_label(draw_name: str) -> str:
#         """Extract MDA category for legend from draw name."""
#         mda_labels = ["no MDA", "MDA SAC", "MDA PSAC", "MDA All"]
#         for label in mda_labels:
#             if label in draw_name:
#                 return label
#         return "Other"
#
#     df = df.loc[df.index.get_level_values("year") >= year_range[0]]
#
#     # Mean across runs per draw
#     df_mean_runs = df.groupby(axis=1, level="draw").mean()
#
#     # Identify threshold crossing
#     below = (df_mean_runs < threshold).reset_index()
#     long_format = below.melt(id_vars=["year", "district"], var_name="draw", value_name="below_threshold")
#     below_threshold = long_format[long_format["below_threshold"]]
#
#     # First year reaching threshold by draw
#     first_years = below_threshold.groupby(["district", "draw"])["year"].min().reset_index(name="year_ephp")
#
#     total_districts = df.index.get_level_values("district").nunique()
#
#     # Cumulative count of districts reaching EPHP
#     ephp_counts = (
#         first_years.groupby(["draw", "year_ephp"])
#         .size()
#         .groupby(level=0)
#         .cumsum()
#         .reset_index(name="num_districts")
#     )
#     ephp_counts["prop_districts"] = ephp_counts["num_districts"] / total_districts
#     ephp_counts = ephp_counts[ephp_counts["year_ephp"].between(*year_range)]
#     ephp_counts.to_excel(
#         results_folder / f'ephp_counts{species}.xlsx')
#
#     # Draws to plot
#     draw_labels = {
#         'Pause WASH, no MDA': 'no MDA',
#         'Pause WASH, MDA SAC': 'MDA SAC',
#         'Pause WASH, MDA PSAC': 'MDA PSAC',
#         'Pause WASH, MDA All': 'MDA All',
#         'Scale-up WASH, no MDA': 'WASH only',
#     }
#
#     plt.figure(figsize=figsize)
#     for draw, label in draw_labels.items():
#         if draw in ephp_counts["draw"].unique():
#             data = ephp_counts[ephp_counts["draw"] == draw]
#             plt.step(
#                 data["year_ephp"],
#                 data["prop_districts"],
#                 where="post",
#                 label=label,
#                 color=mda_colours.get(label, "grey"),
#                 linewidth=1.8,
#                 alpha=alpha,
#             )
#         else:
#             # Offset zero-line slightly to avoid overlap
#             y_offset = {
#                 'no MDA': -0.001,
#                 'WASH only': 0.001
#             }.get(label, 0)
#
#             plt.step(
#                 [year_range[0], year_range[1]],
#                 [y_offset, y_offset],
#                 where="post",
#                 label=label,
#                 color=mda_colours.get(label, "grey"),
#                 linestyle="--" if label == "WASH only" else "--",
#                 linewidth=1.5,
#                 alpha=alpha,
#             )
#
#     plt.title("")
#     plt.ylabel(f"Proportion < {threshold * 100:.1f}%")
#     plt.xlabel("Year")
#     plt.ylim(-0.05, 1)
#     plt.grid(True, color="grey", linestyle="-", linewidth=0.5, alpha=0.15)
#     plt.legend(title="Strategy", loc="upper left", fontsize="small")
#     plt.tight_layout()
#     plt.savefig(results_folder / f"ephp_km_plot_{species}_pause.png", dpi=300)
#     plt.show()
#
#
# plot_ephp_km_pause(prev_haem_H_All_district, species='haem')
#
# plot_ephp_km_pause(prev_mansoni_H_All_district, species='mansoni')
#
#
# # get the figures for the SI separately
# def plot_ephp_km_scaleup(
#     df: pd.DataFrame,
#     threshold: float = 0.015,
#     year_range: tuple = (2024, 2040),
#     alpha: float = 1.0,
#     figsize: tuple = (10, 6),
#         species=None
# ):
#     """
#     Plot Kaplan-Meier style curves in a single panel showing the proportion of districts
#     reaching prevalence < threshold by year, for:
#         - Each MDA strategy under 'Continue WASH'
#         - 'no MDA' under 'Scale-up WASH' as 'WASH only'
#     """
#
#     mda_colours = {
#         'no MDA': '#1b9e77',  # Teal
#         'MDA SAC': '#d95f02',  # Orange
#         'MDA PSAC': '#7570b3',  # Purple
#         'MDA All': '#e7298a',  # Pink
#         'WASH only': '#e6ab02'  # Mustard Yellow – distinct from teal
#     }
#
#     def extract_mda_label(draw_name: str) -> str:
#         """Extract MDA category for legend from draw name."""
#         mda_labels = ["no MDA", "MDA SAC", "MDA PSAC", "MDA All"]
#         for label in mda_labels:
#             if label in draw_name:
#                 return label
#         return "Other"
#
#     df = df.loc[df.index.get_level_values("year") >= year_range[0]]
#
#     # Mean across runs per draw
#     df_mean_runs = df.groupby(axis=1, level="draw").mean()
#
#     # Identify threshold crossing
#     below = (df_mean_runs < threshold).reset_index()
#     long_format = below.melt(id_vars=["year", "district"], var_name="draw", value_name="below_threshold")
#     below_threshold = long_format[long_format["below_threshold"]]
#
#     # First year reaching threshold by draw
#     first_years = below_threshold.groupby(["district", "draw"])["year"].min().reset_index(name="year_ephp")
#
#     total_districts = df.index.get_level_values("district").nunique()
#
#     # Cumulative count of districts reaching EPHP
#     ephp_counts = (
#         first_years.groupby(["draw", "year_ephp"])
#         .size()
#         .groupby(level=0)
#         .cumsum()
#         .reset_index(name="num_districts")
#     )
#     ephp_counts["prop_districts"] = ephp_counts["num_districts"] / total_districts
#     ephp_counts = ephp_counts[ephp_counts["year_ephp"].between(*year_range)]
#     ephp_counts.to_excel(
#         results_folder / f'ephp_counts{species}_scaleup.xlsx')
#
#     # Draws to plot
#     draw_labels = {
#         'Scale-up WASH, no MDA': 'no MDA',
#         'Scale-up WASH, MDA SAC': 'MDA SAC',
#         'Scale-up WASH, MDA PSAC': 'MDA PSAC',
#         'Scale-up WASH, MDA All': 'MDA All',
#     }
#
#     plt.figure(figsize=figsize)
#     for draw, label in draw_labels.items():
#         if draw in ephp_counts["draw"].unique():
#             data = ephp_counts[ephp_counts["draw"] == draw]
#             plt.step(
#                 data["year_ephp"],
#                 data["prop_districts"],
#                 where="post",
#                 label=label,
#                 color=mda_colours.get(label, "grey"),
#                 linewidth=1.8,
#                 alpha=alpha,
#             )
#         else:
#             # Offset zero-line slightly to avoid overlap
#             y_offset = {
#                 'no MDA': -0.001,
#                 'WASH only': 0.001
#             }.get(label, 0)
#
#             plt.step(
#                 [year_range[0], year_range[1]],
#                 [y_offset, y_offset],
#                 where="post",
#                 label=label,
#                 color=mda_colours.get(label, "grey"),
#                 linestyle="--" if label == "WASH only" else "--",
#                 linewidth=1.5,
#                 alpha=alpha,
#             )
#
#     plt.title("")
#     plt.ylabel(f"Proportion < {threshold * 100:.1f}%")
#     plt.xlabel("Year")
#     plt.ylim(-0.05, 1)
#     plt.grid(True, color="grey", linestyle="-", linewidth=0.5, alpha=0.15)
#     plt.legend(title="Strategy", loc="upper left", fontsize="small")
#     plt.tight_layout()
#     plt.savefig(results_folder / f"ephp_km_plot_{species}_scaleup.png", dpi=300)
#     plt.show()
#
#
# plot_ephp_km_scaleup(prev_haem_H_All_district)
# plot_ephp_km_scaleup(prev_mansoni_H_All_district)
#

