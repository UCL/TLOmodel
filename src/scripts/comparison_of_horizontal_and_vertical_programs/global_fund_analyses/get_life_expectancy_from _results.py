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
from typing import Tuple
from tlo.analysis.life_expectancy import get_life_expectancy_estimates


def apply(results_folder: Path, output_folder: Path, HSS_or_HTM: str):
    """
    use the outputs from batch runs to generate life expectancy estimates across each draw

    :param results_folder: Path to stored results
    :param output_folder: set to store outputs in same folder as results_folder
    :param HSS_or_HTM: select either 'HTM' or 'HSS' to determine which param_names to import
    :return: an excel workbook with multiple sheet containing life expectancy estimates across each draw
    for men and women separately
    """

    def get_parameter_names_from_scenario_file() -> Tuple[str]:
        """Get the tuple of names of the scenarios from `Scenario` class used to create the results."""
        if HSS_or_HTM == 'HSS':
            from scripts.comparison_of_horizontal_and_vertical_programs.global_fund_analyses.scenario_hss_elements_gf import \
                HSSElements
            e = HSSElements()
        elif HSS_or_HTM == 'HTM':
            from scripts.comparison_of_horizontal_and_vertical_programs.global_fund_analyses.scenario_vertical_programs_with_and_without_hss_gf import \
                HTMWithAndWithoutHSS
            e = HTMWithAndWithoutHSS()
        else:
            raise ValueError("HSS_or_HTM must be either 'HSS' or 'HTM'")

        return tuple(e._scenarios.keys())

    param_names = get_parameter_names_from_scenario_file()

    # Define the target periods
    periods = {
        "2010": (datetime.date(2010, 1, 1), datetime.date(2010, 12, 31)),
        "2024": (datetime.date(2024, 1, 1), datetime.date(2024, 12, 31)),
        "2030": (datetime.date(2030, 1, 1), datetime.date(2030, 12, 31)),
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

    output_file = output_folder / f'life_expectancy_summary_{HSS_or_HTM}.xlsx'
    with pd.ExcelWriter(output_file) as writer:
        for year, df in summaries.items():
            df.to_excel(writer, sheet_name=f"Summary_{year}")

    # todo figure showing LE for each draw, can be different sets depending on HSS or HTM scenarios
    # use 2030 values, all others (2010/2024) should be the same (or similar)



    def plot_le(df: pd.DataFrame):
        """
        Plots the 2030 summary data with points for means and error bars using lower and upper bounds.

        Parameters:
        df (pd.DataFrame): DataFrame with MultiIndex columns ('draw', 'stat') and rows (e.g., 'M', 'F').
        """
        # Define the draws (the unique values from level 'draw' of the column MultiIndex)
        draws = df.columns.get_level_values('draw').unique()
        # draw_indices = list(range(len(draws)))

        fig, ax = plt.subplots(figsize=(12, 6))
        substitute_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        # Create lists to hold custom legend entries for x-axis labels
        xtick_labels = [f'{letter}' for letter in substitute_labels[:len(draws)]]
        xtick_legend_entries = [f'{letter}: {draw}' for letter, draw in zip(substitute_labels[:len(draws)], draws)]

        # Loop through the rows ('M' and 'F') and plot each with error bars
        for sex in df.index:
            # Extract the mean, lower, and upper values for this gender
            means = df.loc[sex, (slice(None), 'mean')]
            lower = df.loc[sex, (slice(None), 'lower')]
            upper = df.loc[sex, (slice(None), 'upper')]

            # Calculate the error bars (upper and lower differences from the mean)
            yerr = [means.values - lower.values, upper.values - means.values]

            # Plotting
            ax.errorbar(draws, means, yerr=yerr, label=f'{sex}', fmt='o', capsize=5)

        # Customize plot
        ax.set_xlabel('')
        ax.set_ylabel('Life expectancy estimate, years')
        ax.set_title('2030 Life Expectancy Estimates')
        ax.set_ylim(50, 80)

        # Set custom x-tick labels
        ax.set_xticklabels(xtick_labels)

        # Adding legend for sex
        # ax.legend(title='Sex', loc='upper left', bbox_to_anchor=(1, 1))
        #
        # # Creating custom legend for x-axis labels
        # ax.legend(handles=[plt.Line2D([0], [0], marker='', color='w', markerfacecolor='black', markersize=10,
        #                               linestyle='')] * len(xtick_legend_entries),
        #           labels=xtick_legend_entries,
        #           title='',
        #           loc='upper left', bbox_to_anchor=(1, 0.5))

        # Create and add the first legend for sex
        sex_legend = ax.legend(title='Sex', loc='upper left', bbox_to_anchor=(0, 1))
        ax.add_artist(sex_legend)  # Add the first legend manually

        # Create the second legend for x-axis labels
        xaxis_legend = ax.legend(handles=[plt.Line2D([0], [0], marker='', color='w', markerfacecolor='black',
                                                     markersize=10, linestyle='')] * len(xtick_legend_entries),
                                 labels=xtick_legend_entries,
                                 title='Draws',
                                 loc='upper left', bbox_to_anchor=(1, 0.5))
        # ax.grid(True)
        # Add vertical grey lines between each category
        for i in range(1, len(draws)):
            ax.axvline(x=i - 0.5, color='lightgrey', linestyle='--', linewidth=0.8)

        # Show plot
        plt.tight_layout()
        plt.show()

    # Plot life expectancy for 2030
    plot_le(summaries['2030'])


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


