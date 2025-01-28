"""
this script uses outputs from scenario_hss_elements_gf or
scenario_vertical_programs_with_and_without_hss_gf
and produces a series of life expectancy estimates for three time periods
results are output into an excel file in results_folder

call function in terminal using:
python
src/scripts/comparison_of_horizontal_and_vertical_programs/global_fund_analyses/get_life_expectancy_from _results.py
results HSS

"""


from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import argparse
import textwrap
import numpy as np

from typing import Tuple
from tlo.analysis.life_expectancy import get_life_expectancy_estimates


def apply(results_folder: Path, output_folder: Path):
    """
    use the outputs from batch runs to generate life expectancy estimates across each draw

    :param results_folder: Path to stored results
    :param output_folder: set to store outputs in same folder as results_folder
    :param HSS_or_HTM: select either 'HTM' or 'HSS' to determine which param_names to import
    :return: an excel workbook with multiple sheet containing life expectancy estimates across each draw
    for men and women separately
    """
    from scripts.comparison_of_horizontal_and_vertical_programs.manuscript_analyses.scenario_hss_htm_paper import \
        HTMWithAndWithoutHSS

    def get_parameter_names_from_scenario_file(e: HTMWithAndWithoutHSS) -> Tuple[str]:
        """Get the tuple of names of the scenarios from `Scenario` class used to create the results."""
        return tuple(e._scenarios.keys())

    # Create a single instance of HTMWithAndWithoutHSS
    e = HTMWithAndWithoutHSS()

    # Get parameter names
    param_names = get_parameter_names_from_scenario_file(e)

    # Define the target periods
    periods = {
        str(year): (datetime.date(year, 1, 1), datetime.date(year, 12, 31))
        for year in [2010] + list(range(2024, e.end_date.year))
    }

    summaries = {}

    # Generate LE estimates for each period and store it in the dictionary
    for year, period in periods.items():
        df = get_life_expectancy_estimates(
            results_folder=results_folder,
            target_period=period,
            summary=True,
        )

        if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels > 1:
            # Check if 'draw' is the name of the first level
            if df.columns.names[0] == 'draw':
                # Replace the 'draw' level with parameter names
                current_levels = df.columns.levels[1]
                new_index = pd.MultiIndex.from_product([param_names, current_levels], names=['draw', df.columns.names[1]])
                df.columns = new_index[:df.columns.size]

        summaries[year] = df

    output_file = output_folder / f'life_expectancy_summary.xlsx'
    with pd.ExcelWriter(output_file) as writer:
        for year, df in summaries.items():
            df.to_excel(writer, sheet_name=f"Summary_{year}")

    def plot_le(df: pd.DataFrame, baseline2024: pd.DataFrame):
        """
        Plots the 2030 summary data with points for means and error bars using lower and upper bounds.

        Parameters:
        df (pd.DataFrame): DataFrame with MultiIndex columns ('draw', 'stat') and rows (e.g., 'M', 'F').
        """

        # Define the columns to select
        column_labels = ['Baseline',
                         'HSS PACKAGE: Realistic',
                         'HTM Programs Scale-up WITHOUT HSS PACKAGE',
                         'HTM Programs Scale-up WITH REALISTIC HSS PACKAGE']
        filtered_df = df.loc[:, column_labels]

        # Define the draws (the unique values from level 'draw' of the column MultiIndex)
        draws = filtered_df.columns.get_level_values('draw').unique()

        # Number of categories
        num_categories = len(column_labels)

        # Define the x-axis positions for each category
        x_positions = np.arange(num_categories)

        fig, ax = plt.subplots(figsize=(12, 6))
        wrapped_xtick_labels = [textwrap.fill(label, 25) for label in column_labels]  # Wrap labels
        spectral = plt.get_cmap('Spectral')
        color = [spectral(0.2), spectral(0.8)]   # Deep Blue

        # Loop through the rows ('M' and 'F') and plot each with error bars
        for i, sex in enumerate(filtered_df.index):
            # Extract the mean, lower, and upper values for this gender
            means = filtered_df.loc[sex, (slice(None), 'median')]
            lower = filtered_df.loc[sex, (slice(None), 'lower')]
            upper = filtered_df.loc[sex, (slice(None), 'upper')]

            # Calculate the error bars (upper and lower differences from the mean)
            yerr = [means.values - lower.values, upper.values - means.values]

            # Plotting
            ax.errorbar(x_positions, means, yerr=yerr, label=f'{sex}', fmt='o', capsize=5, color=color[i])

        # Customize plot
        ax.set_xlabel('')
        ax.set_ylabel('Life expectancy estimate, years')
        ax.set_title('2035 Life Expectancy Estimates')
        ax.set_ylim(50, 80)

        # Set custom x-tick labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(wrapped_xtick_labels, rotation=0, ha='center')

        # Create and add the first legend for sex
        sex_legend = ax.legend(title='Sex', loc='upper left', bbox_to_anchor=(0, 1))
        ax.add_artist(sex_legend)  # Add the first legend manually

        # Add horizontal lines for clarity
        for y in range(50, 81, 5):
            ax.axhline(y=y, color='lightgrey', linestyle='--', linewidth=0.8)

        # Add vertical grey lines between each category
        for i in range(1, len(x_positions)):
            ax.axvline(x=i - 0.5, color='lightgrey', linestyle='--', linewidth=0.8)

        # Show plot
        plt.tight_layout()
        plt.show()

    # Plot life expectancy for 2035
    baseline2024 = summaries['2024']['Baseline']
    plot_le(summaries['2035'], baseline2024)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    parser.add_argument("HSS_or_HTM", type=str, choices=['HSS', 'HTM'], help="Specify either 'HSS' or 'HTM'")
    args = parser.parse_args()

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        HSS_or_HTM=args.HSS_or_HTM
    )


