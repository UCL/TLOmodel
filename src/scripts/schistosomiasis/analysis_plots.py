

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
    Only rows where 'wash_strategy' contains the specified context are included.
    """
    # Filter for the specified WASH context
    df_filtered = df[df['wash_strategy'].str.contains(context, na=False)]

    categories = ['MDA SAC', 'MDA PSAC', 'MDA All']
    titles = {
        'MDA SAC': f'{context} MDA SAC',
        'MDA PSAC': 'MDA PSAC',
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
    plt.show()


file_path = results_folder / f'icer_district_2024-2040.xlsx'
icer_district_df = pd.read_excel(file_path)
icer_district_df['draw'] = icer_district_df['comparison'].str.extract(r'^(.*?)\s+vs')

plot_icer_three_panels(icer_district_df, context='Continue WASH')

plot_icer_three_panels(icer_district_df, context='Scale-up WASH')


#################################################################################
# %% PLOTS DALYS VS COSTS
#################################################################################



def plot_dalys_vs_costs_by_district(
    dalys_district_df: pd.DataFrame,
    costs_district_df: pd.DataFrame,
    wash_strategy: str,
    comparison: str,
    plot_summary: bool = True,
    threshold: float = 500.0
):
    """
    Plot DALYs averted vs incremental costs by district for a specified WASH strategy and comparison.
    Points are coloured by cost-effectiveness based on a threshold ICER.
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

        # Compute ICERs
        icers = mean_costs / mean_dalys
        cost_effective = icers < threshold

        # Plot with colouring based on cost-effectiveness
        for i, district in enumerate(districts):
            colour = 'blue' if cost_effective[district] else 'red'
            plt.errorbar(
                mean_costs[district], mean_dalys[district],
                xerr=ci_costs[district], yerr=ci_dalys[district],
                fmt='o', capsize=5, markersize=6,
                ecolor='grey', elinewidth=1.5,
                markerfacecolor=colour, markeredgecolor='black'
            )
            plt.text(mean_costs[district], mean_dalys[district], district,
                     fontsize=9, alpha=0.8, ha='right', va='bottom')

        # Plot ICER threshold line
        x_vals = plt.xlim()
        x_line = np.linspace(*x_vals, 100)
        y_line = x_line / threshold
        plt.plot(x_line, y_line, linestyle='--', color='grey', label=f"ICER = ${threshold:.0f}/DALY")

        plt.legend(loc='best')

    else:
        num_runs = dalys_sub.shape[1]
        colors = plt.cm.viridis(np.linspace(0, 1, num_runs))

        for run_idx in range(num_runs):
            plt.scatter(
                costs_sub.iloc[:, run_idx],
                dalys_sub.iloc[:, run_idx],
                label=f'Run {run_idx + 1}',
                color=colors[run_idx],
                alpha=0.7,
                s=40,
                edgecolors='none'
            )

        mean_costs = costs_sub.mean(axis=1)
        mean_dalys = dalys_sub.mean(axis=1)
        for district, x, y in zip(districts, mean_costs, mean_dalys):
            plt.text(x, y, district, fontsize=9, alpha=0.8,
                     horizontalalignment='right', verticalalignment='bottom')

        # plt.legend(title='Simulation Runs', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xlabel('Incremental Costs (USD)')
    plt.ylabel('DALYs Averted')
    plt.title(f'DALYs vs Incremental Costs by District\nStrategy: {wash_strategy} | Comparison: {comparison}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


file_path = results_folder / f'sum_incremental_dalys_averted_district2024-2040.xlsx'
dalys_district_df = pd.read_excel(file_path, header=[0, 1, 2], index_col=0)

file_path = results_folder / f'sum_incremental_full_costs_incurred_district2024-2040.xlsx'
full_costs_district_df = pd.read_excel(file_path, header=[0, 1, 2], index_col=0)
full_costs_district_df.columns = full_costs_district_df.columns.droplevel(2)

file_path = results_folder / f'sum_incremental_cons_costs_incurred_district2024-2040.xlsx'
cons_costs_district_df = pd.read_excel(file_path, header=[0, 1, 2], index_col=0)
cons_costs_district_df.columns = cons_costs_district_df.columns.droplevel(2)

plot_dalys_vs_costs_by_district(
    dalys_district_df=dalys_district_df,
    costs_district_df=full_costs_district_df,
    wash_strategy='Continue WASH',
    comparison='MDA SAC vs no MDA',
    plot_summary=True,
    threshold=120
)

plot_dalys_vs_costs_by_district(
    dalys_district_df=dalys_district_df,
    costs_district_df=full_costs_district_df,
    wash_strategy='Continue WASH',
    comparison='MDA PSAC vs MDA SAC',
    plot_summary=True,
    threshold=120
)

plot_dalys_vs_costs_by_district(
    dalys_district_df=dalys_district_df,
    costs_district_df=full_costs_district_df,
    wash_strategy='Continue WASH',
    comparison='MDA All vs MDA PSAC',
    plot_summary=True,
    threshold=120
)


# cons only
plot_dalys_vs_costs_by_district(
    dalys_district_df=dalys_district_df,
    costs_district_df=cons_costs_district_df,
    wash_strategy='Continue WASH',
    comparison='MDA SAC vs no MDA',
    plot_summary=True,
    threshold=120
)

plot_dalys_vs_costs_by_district(
    dalys_district_df=dalys_district_df,
    costs_district_df=cons_costs_district_df,
    wash_strategy='Continue WASH',
    comparison='MDA PSAC vs MDA SAC',
    plot_summary=True,
    threshold=120
)

plot_dalys_vs_costs_by_district(
    dalys_district_df=dalys_district_df,
    costs_district_df=cons_costs_district_df,
    wash_strategy='Continue WASH',
    comparison='MDA All vs MDA PSAC',
    plot_summary=True,
    threshold=120
)


#################################################################################
# %% KAPLAN-MEIER time to elimination
#################################################################################


def plot_ephp_km_panels(
    df: pd.DataFrame,
    threshold: float = 0.015,
    year_range: tuple = (2024, 2040),
    alpha: float = 1.0,
    figsize: tuple = (8, 12)
):
    """
    Plot Kaplan-Meier-style curves in three vertically stacked panels showing the proportion of districts
    reaching prevalence < threshold by year. Panels are grouped by draw naming patterns: 'Pause', 'Continue', 'Scale-up'.

    Parameters:
        df : pd.DataFrame
            DataFrame with MultiIndex (year, district) and columns with MultiIndex (draw, run)
        threshold : float
            Prevalence threshold for defining EPHP
        year_range : tuple
            Range of years to display on x-axis
        alpha : float
            Transparency for individual draw lines
        figsize : tuple
            Size of the overall figure
    """

    def extract_mda_label(draw_name: str) -> str:
        """Extract MDA category for legend from draw name."""
        mda_labels = ["no MDA", "MDA SAC", "MDA PSAC", "MDA All"]
        draw_lower = draw_name.replace(" ", "").lower()
        for label in mda_labels:
            if label.replace(" ", "").lower() in draw_lower:
                return label
        return "Other"

    # Remove pre-2024 data
    df = df.loc[df.index.get_level_values("year") >= 2024]

    # Step 1: mean across runs for each draw
    df_mean_runs = df.groupby(axis=1, level="draw").mean()

    # Step 2: identify years where prevalence < threshold
    below = (df_mean_runs < threshold).reset_index()
    long_format = below.melt(id_vars=["year", "district"], var_name="draw", value_name="below_threshold")
    below_threshold = long_format[long_format["below_threshold"]]

    # First year each district reaches threshold, by draw
    first_years = below_threshold.groupby(["district", "draw"])["year"].min().reset_index(name="year_ephp")

    # Setup for panel plots
    draw_filters = {
        "Pause": "Pause",
        "Continue": "Continue",
        "Scale-up": "Scale-up"
    }

    total_districts = df.index.get_level_values("district").nunique()

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=figsize, sharex=True, sharey=True)

    # Use darker palette, enough colors for up to 20 draws per panel
    palette = sns.color_palette("dark", n_colors=20)

    for ax, (title, substr) in zip(axes, draw_filters.items()):
        # Filter draws by name substring
        filtered = first_years[first_years["draw"].str.contains(substr)]

        # Count cumulative districts reaching EPHP by year
        ephp_counts = (
            filtered.groupby(["draw", "year_ephp"])
            .size()
            .groupby(level=0)
            .cumsum()
            .reset_index(name="num_districts")
        )
        ephp_counts["prop_districts"] = ephp_counts["num_districts"] / total_districts
        ephp_counts = ephp_counts[ephp_counts["year_ephp"].between(*year_range)]

        # Prepare unique draws and assign colours
        draw_list = ephp_counts["draw"].unique()
        color_dict = dict(zip(draw_list, palette[:len(draw_list)]))

        # To avoid duplicate legend labels for same MDA category
        plotted_labels = set()

        for draw, data in ephp_counts.groupby("draw"):
            label = extract_mda_label(draw)
            if label not in plotted_labels:
                plot_label = label
                plotted_labels.add(label)
            else:
                plot_label = None  # Don't repeat label in legend

            ax.step(
                data["year_ephp"],
                data["prop_districts"],
                where="post",
                label=plot_label,
                color=color_dict[draw],
                alpha=alpha,
                linewidth=1.5,
            )

        ax.set_title(title)
        ax.set_ylabel("Proportion < {:.1f}%".format(threshold * 100))
        ax.grid(True, color="grey", linestyle="-", linewidth=0.5, alpha=0.15)
        ax.legend(loc="upper left", fontsize="small", title="")

    axes[-1].set_xlabel("Year")
    plt.suptitle("Progress Toward EPHP by Year and Strategy", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


plot_ephp_km_panels(prev_haem_H_All_district)
plot_ephp_km_panels(prev_mansoni_H_All_district)
