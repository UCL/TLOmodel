import argparse
from pathlib import Path
from scipy.signal import savgol_filter
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date
from tlo.analysis.utils import (
    CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP,
    extract_results,
    make_age_grp_lookup,
    order_of_cause_of_death_or_daly_label,
    summarize,
)

min_year = 2020
max_year = 2070
spacing_of_years = 1
scenario_names = ["Status Quo", "HTM Scale-up", "Worsening Lifestyle Factors", "Improving Lifestyle Factors",
                  "Maximal Healthcare \nProvision", ]
# scenario_names = ["Status Quo","Worsening Lifestyle Factors", "Improving Lifestyle Factors"]
scenario_markers = ['o', 'X', 'H', '^', 'p']
scenario_colours = ['#0081a7', '#00afb9', '#FEB95F', '#fed9b7', '#f07167', '#9A348E']

li_factors = ['li_urban', 'li_bmi', 'li_low_ex', 'li_high_salt', 'li_high_sugar', 'li_tob', 'li_unimproved_sanitation',
              'li_no_access_handwashing', 'li_no_clean_drinking_water', 'li_wood_burn_stove']

# Improved color scheme with better contrast
factor_colours = {
    'li_urban': '#0081a7',
    'li_bmi': '#e63946',
    'li_low_ex': '#0F0326',
    'li_high_salt': '#fcbf49',
    'li_high_sugar': '#0CF574',
    'li_tob': '#9A348E',
    'li_unimproved_sanitation': '#0081a7',
    'li_no_access_handwashing': '#e63946',
    'li_no_clean_drinking_water': '#0CF574',
    'li_wood_burn_stove': '#9A348E',
}

# Better readable factor names for legend
factor_display_names = {
    'li_urban': 'Urban',
    'li_bmi': 'High BMI',
    'li_low_ex': 'Low Exercise',
    'li_high_salt': 'High Salt',
    'li_high_sugar': 'High Sugar',
    'li_tob': 'Tobacco',
    'li_unimproved_sanitation': 'Improved Sanitation', # we will be looking at the converse of these factors (i.e. 1 - (factor/total))
    'li_no_access_handwashing': 'Improvd Handwashing',
    'li_no_clean_drinking_water': 'Clean Water',
    'li_wood_burn_stove': 'No Wood Burning',
}


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """
    Produce plots describing each li_factor across all scenario draws.
    Lifestyle factors are split into two subplots: one with the first group, one with the remaining.
    All draws are plotted on the same graph.
    """
    target_year_sequence = range(min_year, max_year + 1, spacing_of_years)
    make_graph_file_name = lambda stub: output_folder / f"{stub}.png"

    li_factors_left = [
        'li_urban', 'li_bmi', 'li_low_ex',
        'li_high_salt', 'li_high_sugar', 'li_tob'
    ]
    li_factors_right = [
        'li_unimproved_sanitation',
        'li_no_access_handwashing',
        'li_no_clean_drinking_water',
        'li_wood_burn_stove'
    ]

    # Create figure with improved layout
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=False)  # Don't share y-axis

    all_data = {}
    for li_factor in li_factors:
        all_data[li_factor] = {}

        for draw in range(len(scenario_names)):
            if (draw == 1) or (draw == 2):
                continue
            _, age_grp_lookup = make_age_grp_lookup()

            def get_li_factor_totals(_df):
                _df = _df.copy()
                _df['date'] = pd.to_datetime(_df['date'])
                filtered_df = _df.loc[_df['date'].between(*TARGET_PERIOD)]
                true_columns = [col for col in filtered_df.columns if f'{li_factor}=True' in col]
                total = filtered_df.loc[:, true_columns].sum().sum()
                return pd.Series({'total': total})

            def population_by_agegroup_for_year(_df):
                _df['date'] = pd.to_datetime(_df['date'])
                filtered_df = _df.loc[_df['date'].between(*TARGET_PERIOD)]
                population_by_agegroup = (
                    filtered_df.drop(columns=['date'], errors='ignore')
                    .melt(var_name='age_grp')
                    .set_index('age_grp')['value']
                )
                return population_by_agegroup

            all_years_data_population = {}
            all_years_data_li_factor_standard_years = {}

            for target_year in target_year_sequence:
                TARGET_PERIOD = (Date(target_year, 1, 1), Date(target_year, 12, 31))

                result_data_li_factor = summarize(
                    extract_results(
                        results_folder,
                        module='tlo.methods.enhanced_lifestyle',
                        key=li_factor,
                        custom_generate_series=get_li_factor_totals,
                        do_scaling=True
                    ),
                    only_mean=True,
                    collapse_columns=True
                )[draw]

                num_by_age_F = summarize(
                    extract_results(
                        results_folder,
                        module="tlo.methods.demography",
                        key='age_range_f',
                        custom_generate_series=population_by_agegroup_for_year,
                        do_scaling=True
                    ),
                    collapse_columns=True,
                    only_mean=True
                )[draw]

                num_by_age_M = summarize(
                    extract_results(
                        results_folder,
                        module="tlo.methods.demography",
                        key='age_range_m',
                        custom_generate_series=population_by_agegroup_for_year,
                        do_scaling=True
                    ),
                    collapse_columns=True,
                    only_mean=True
                )[draw]

                num_by_age = num_by_age_F + num_by_age_M
                num_by_age[num_by_age == 0] = np.nan

                all_years_data_population[target_year] = num_by_age['mean'].sum(axis=0)
                all_years_data_li_factor_standard_years[target_year] = result_data_li_factor['mean']

            df_li_factor_standard_years = pd.DataFrame(all_years_data_li_factor_standard_years)

            df_li_factor_normalized = df_li_factor_standard_years.div(
                df_li_factor_standard_years.iloc[:, 0], axis=0
            )

            all_data[li_factor][draw] = df_li_factor_normalized

    print(all_data)

    # Plot data on each subplot
    for ax_idx, (ax, li_factors_group, title) in enumerate(zip(
        axes,
        [li_factors_left, li_factors_right],
        ['Worsening Lifestyle Factors', 'Improving Lifestyle Factors']
    )):

        all_y_values = []

        for li_factor in li_factors_group:
            scenario_color_indices = [0, 3, 4]  # Skip draws 1 and 2
            active_draws = [i for i in range(len(scenario_names)) if i not in [1, 2]]

            for i, draw in enumerate(active_draws):
                df_normalized = all_data[li_factor][draw]
                mean_values = df_normalized.mean(axis=0)
                all_y_values.extend(mean_values.values)

                # Use thicker lines and more distinct styling
                ax.plot(
                    mean_values.index,
                    savgol_filter(mean_values.to_numpy(), window_length=5, polyorder=2),
                    marker=scenario_markers[draw],
                    label=f"{factor_display_names[li_factor]} - {scenario_names[draw]}",
                    color=factor_colours[li_factor],
                    linestyle=['-', '--', ':'][i],  # Different line styles for scenarios
                    linewidth=2.5,
                    markersize=6,
                    alpha=0.4
                )

        if all_y_values:
            y_min, y_max = min(all_y_values), max(all_y_values)
            y_range = y_max - y_min

            # Add padding and set appropriate limits
            if y_range < 0.1:  # Very small range
                ax.set_ylim(y_min - 0.05, y_max + 0.05)
                ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
            else:
                padding = y_range * 0.1
                ax.set_ylim(y_min - padding, y_max + padding)

        ax.set_xlabel('Year', fontsize=14)
        ax.set_title(title, fontsize=14, pad=20)
        ax.tick_params(axis='both', which='major', labelsize=12)

        handles, labels = ax.get_legend_handles_labels()
        factor_groups = {}
        for handle, label in zip(handles, labels):
            factor_name = label.split(' - ')[0]
            scenario_name = label.split(' - ')[1]
            if factor_name not in factor_groups:
                factor_groups[factor_name] = []
            factor_groups[factor_name].append((handle, scenario_name))

        if ax_idx == 0:  # Left plot legend
            ax.legend(
                handles, labels,
                #bbox_to_anchor=(0.02, 0.98),
                loc='upper left',
                fontsize=9,
                ncol=1,
                columnspacing=0.5
            )
        else:  # Right plot legend
            ax.legend(
                handles, labels,
                #bbox_to_anchor=(1.02, 1),
                loc='upper left',
                ncol=1,
                columnspacing=0.5
            )
    axes[0].set_ylabel('Fold change in lifestyle factor', fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    fig.savefig(make_graph_file_name('li_factors_split_normalized'), dpi=300, bbox_inches='tight')
    plt.close(fig)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources')
    )
