
from pathlib import Path
import pandas as pd
import datetime
import argparse
from typing import Tuple

from tlo import Date
from tlo.analysis.utils import get_scenario_outputs
from tlo.analysis.life_expectancy import get_life_expectancy_estimates


outputspath = Path("./outputs")

# Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("scenario_hss_elements_gf.py", outputspath)

results_folder = Path('/Users/tmangal/PycharmProjects/TLOmodel/outputs/hss_elements-2024-08-21T125348Z')

results_folder = Path('/Users/tmangal/PycharmProjects/TLOmodel/outputs/t.mangal@imperial.ac.uk/hss_elements-2024-08-19T105018Z')


def apply(results_folder: Path, output_folder: Path, HSS_or_HTM: str):
    """
    use the outputs from batch runs to generate life expectancy estimates across each draw

    :param results_folder: Path to stored results
    :param output_folder: set to store outputs in same folder as results_folder
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
        "2034": (datetime.date(2034, 1, 1), datetime.date(2034, 12, 31)),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources')
    )
