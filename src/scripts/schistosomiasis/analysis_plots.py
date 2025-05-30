

from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

import pandas as pd
# import lacroix
import matplotlib.colors as colors
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from collections import defaultdict
import textwrap
from typing import Tuple

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

from scripts.costing.cost_estimation import (estimate_input_cost_of_scenarios,
                                             summarize_cost_data,
                                             do_stacked_bar_plot_of_cost_by_category,
                                             do_line_plot_of_cost,
                                             create_summary_treemap_by_cost_subgroup,
                                             estimate_projected_health_spending)

resourcefilepath = Path("./resources")

output_folder = Path("./outputs/t.mangal@imperial.ac.uk")

results_folder = get_scenario_outputs("schisto_scenarios.py", output_folder)[-1]


# Declare path for output graphs from this script
def make_graph_file_name(name):
    return results_folder / f"Schisto_{name}.png"






def plot_prevalence_heatmap(df, year=2040, threshold=1.5, filename=None):
    # Extract data for the given year
    df_year = df.loc[year]

    # Mean over runs if columns have a 'run' level
    if isinstance(df_year.columns, pd.MultiIndex) and 'run' in df_year.columns.names:
        mean_df = df_year.groupby(level='draw', axis=1).mean()
    else:
        mean_df = df_year.copy()

    draw_labels = mean_df.columns.tolist()

    # Parse draw labels into Phase and MDA parts
    phase_labels = []
    mda_labels = []

    for label in draw_labels:
        try:
            phase_part, mda_part = label.split(', ')
        except Exception:
            phase_part, mda_part = label, ''
        # Strip " WASH" suffix from phase for cleaner label
        phase_clean = phase_part.replace(' WASH', '')
        phase_labels.append(phase_clean)
        mda_labels.append(mda_part)

    # Define desired orders
    phase_order = ['Pause', 'Continue', 'Scale-up']
    mda_order = ['no MDA', 'MDA SAC', 'MDA PSAC', 'MDA All']

    # Create DataFrame with these two levels to help sorting
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
        linecolor='gray'
    )

    # add red outline if value < threshold
    # for y in range(mean_df.shape[0]):
    #     for x in range(mean_df.shape[1]):
    #         val = mean_df.iloc[y, x]
    #         if val < threshold:
    #             ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor='red', lw=2))

    for y in range(mean_df.shape[0]):
        for x in range(mean_df.shape[1]):
            val = mean_df.iloc[y, x]
            if val < threshold:
                ax.add_patch(
                    plt.Rectangle(
                        (x, y),
                        1, 1,
                        fill=False,
                        edgecolor='black',
                        lw=1.5,
                        hatch='//'
                    )
                )

    ax.set_ylabel('District')
    plt.title(f'Mean Prevalence by District, Year {year}')

    # ----------- Fix x-axis labels ------------------

    n_mda = len(mda_order)

    # Show MDA labels for every draw
    tick_positions = [i + 0.5 for i in range(len(mean_df.columns))]
    tick_labels = col_df['mda'].tolist()

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax.set_xlabel('')  # remove the x-axis label

    # Add vertical lines after every 4th draw
    for idx in range(n_mda, len(mean_df.columns), n_mda):
        ax.axvline(idx, color='white', linestyle='-', linewidth=4)

    # Add phase labels below MDA labels, centred below each group of 4 draws (each phase)
    for i, phase in enumerate(phase_order):
        start = i * n_mda
        end = start + n_mda - 1
        mid = (start + end) / 2 + 0.5

        ax.text(
            x=mid,
            y=-0.15,  # axis fraction coordinates, slightly below the x-axis labels
            s=phase,
            ha='center',
            va='top',
            fontsize=12,
            fontweight='bold',
            color='black',
            transform=ax.get_xaxis_transform()  # x: data, y: axis fraction
        )
    plt.subplots_adjust(bottom=0.2, top=0.9)  # more bottom space for two level labels
    plt.savefig(make_graph_file_name(filename))

    plt.show()


path = Path(results_folder / f'prev_haem_H_year_district 2024-2040')
prev_haem_H_All_district = pd.read_csv(path, index_col=[0, 1])  # assuming first two columns are index

path2 = Path(results_folder / f'prev_mansoni_H_year_district 2024-2040')
prev_mansoni_H_All_district = pd.read_csv(path, index_col=[0, 1])  # assuming first two columns are index


plot_prevalence_heatmap(prev_haem_H_All_district, year=2040, threshold=0.015, filename='prev_haem_H_district2040.png')
plot_prevalence_heatmap(prev_mansoni_H_All_district, year=2040, threshold=0.015, filename='prev_mansoni_H_district2040.png')




#################################################################################
# %% ICERS
#################################################################################

def plot_icer_three_panels(df, context="Continue_WASH"):
    """
    Plot ICER by district for three categories ('MDA SAC', 'MDA PSAC', 'MDA All')
    Only draws containing 'Continue WASH' are included.
    """
    # Filter draws containing 'Continue WASH'
    df_filtered = df[df['draw'].str.contains(context, na=False)]

    categories = ['MDA SAC', 'MDA PSAC', 'MDA All']
    titles = {
        'MDA SAC': f'{context} MDA SAC',
        'MDA PSAC': 'MDA PSAC',
        'MDA All': 'MDA All'
    }

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharey=True)

    for ax, category in zip(axes, categories):
        subset = df_filtered[df_filtered['draw'].str.contains(category, na=False)]
        if subset.empty:
            ax.text(0.5, 0.5, f'No data for {category}', ha='center', va='center')
            ax.set_title(titles[category])
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Sort districts alphabetically to keep consistent order
        subset = subset.sort_values('level_0')

        # Plot points
        sns.pointplot(
            data=subset,
            x='level_0',
            y='mean',
            join=False,
            color='blue',
            ax=ax
        )

        # Add error bars manually
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

        # Show x-axis labels only on the bottom plot (last subplot)
        if category != 'MDA All':
            ax.set_xlabel('')
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('District')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        ax.set_ylabel('ICER')

    plt.tight_layout()
    plt.show()


plot_icer_three_panels(icer_district, context='Continue WASH')

plot_icer_three_panels(icer_district, context='Scale-up WASH')
