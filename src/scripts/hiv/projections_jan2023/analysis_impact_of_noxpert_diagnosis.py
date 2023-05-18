"""
Extracts DALYs and mortality from the TB module
 """
# commands for running the analysis script in the terminal
# python src/scripts/hiv/projections_jan2023/analysis_impact_of_noxpert_diagnosis.py --scenario-outputs-folder outputs/nic503@york.ac.uk --show-figures
# python src/scripts/hiv/projections_jan2023/analysis_impact_of_noxpert_diagnosis.py --scenario-outputs-folder outputs/nic503@york.ac.uk --save-figures

import argparse
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tlo import Date
from tlo.analysis.utils import (
    extract_results,
    get_scenario_outputs,
    load_pickled_dataframes,
    get_scenario_info,
    extract_params,
    summarize,
    make_age_grp_lookup,
    make_age_grp_types,
    extract_params,
)

outputspath =   Path('./outputs/nic503@york.ac.uk')
target_period = (Date(2010, 1, 1), Date(2015, 12, 31))
results_folder = get_scenario_outputs('scenario_impact_noXpert_diagnosis.py', outputspath)[-1]
def extract_total_deaths(results_folder):
    def extract_deaths_total(df: pd.DataFrame) -> pd.Series:
        return pd.Series({"Total": len(df)})

    return extract_results(
        outputspath,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=extract_deaths_total,
        do_scaling=True
    )
    #extract_deaths_total.to_excel(outputspath / "sample_summarised_deaths.xlsx", index=True)

def plot_summarized_total_deaths(summarized_total_deaths, param_strings):
    fig, ax = plt.subplots()
    number_of_draws = len(param_strings)
    statistic_values = {
        s: np.array(
            [summarized_total_deaths[(d, s)].values[0] for d in range(number_of_draws)]
        )
        for s in ["mean", "lower", "upper"]
    }
    ax.bar(
        param_strings,
        statistic_values["mean"],
        yerr=[
            statistic_values["mean"] - statistic_values["lower"],
            statistic_values["upper"] - statistic_values["mean"]
        ]
    )
    ax.set_ylabel("Total number of deaths")
    fig.tight_layout()
    return fig, ax

def compute_difference_in_deaths_across_runs(total_deaths, scenario_info):
    deaths_difference_by_run = [
        total_deaths[0][run_number]["Total"] - total_deaths[1][run_number]["Total"]
        for run_number in range(scenario_info["runs_per_draw"])
    ]
    return np.mean(deaths_difference_by_run)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        "Analyse scenario results for testing impact of consumables availability"
    )
    parser.add_argument(
        "--scenario-outputs-folder",
        type=Path,
        required=True,
        help="Path to folder containing scenario outputs",
    )
    parser.add_argument(
        "--show-figures",
        action="store_true",
        help="Whether to interactively show generated Matplotlib figures",
    )
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Whether to save generated Matplotlib figures to results folder",
    )
    args = parser.parse_args()
    # Find results_folder associated with a given batch_file and get most recent
    results_folder = get_scenario_outputs(
        "scenario_impact_noXpert_diagnosis.py", args.scenario_outputs_folder)[-1]

# Load log (useful for checking what can be extracted)
log = load_pickled_dataframes(results_folder)

# Get basic information about the results
scenario_info = get_scenario_info(results_folder)

# Get the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# Create a list of strings summarizing the parameter values in the different draws
param_strings = [f"{row.module_param}={row.value}" for _, row in params.iterrows()]

# We first look at total deaths in the scenario runs
total_deaths = extract_total_deaths(results_folder)

# # Compute and print the difference between the deaths across the scenario draws
# mean_deaths_difference_by_run = compute_difference_in_deaths_across_runs(
#     total_deaths, scenario_info
# )
# print(f"Mean difference in total deaths = {mean_deaths_difference_by_run:.3g}")

# Plot the total deaths across the two scenario draws as a bar plot with error bars
fig_1, ax_1 = plot_summarized_total_deaths(summarize(total_deaths), param_strings)

# Now we look at things in more detail with an age breakdown
if args.save_figures:
    fig_1.savefig(results_folder / "total_deaths_across_scenario_draws.pdf")



#
#
# # Find results_folder associated with a given batch_file and get the most recent
# results_folder = get_scenario_outputs('scenario_impact_noXpert_diagnosis.py', outputspath)[-1]
#
# # # Look at one log (so can decide what to extract)
# log = load_pickled_dataframes(results_folder)
#
# # # Get basic information about the results
# info = get_scenario_info(results_folder)
#
# # 1) Extract the parameters that have varied over the set of simulations
# params = extract_params(results_folder)
# def get_num_deaths(_df):
#     return pd.Series(data=len(_df.loc[pd.to_datetime(_df.date).between(*target_period)]))
#
# deaths_extracted = extract_results(
#     results_folder,
#     module="tlo.methods.demography",
#     key="death",
#     custom_generate_series=get_num_deaths,
#     do_scaling=True
# )
# deaths_extracted.to_excel(outputspath / "summarised_deaths.xlsx", index=True)
#
# def get_num_dalys(_df):
# #         """Return the total number of DALYS (Stacked) by label (total within the TARGET_PERIOD)"""
#         return pd.Series(
#             data=_df
#             .loc[_df.year.between(*[i.year for i in target_period])]
#             .drop(columns=['date', 'sex', 'age_range', 'year'])
#             .sum().sum()
#         )
#
# # def get_num_dalys(_df):
# #     return pd.Series(data=len(_df.loc[pd.to_datetime(_df.date).between(*target_period)]))
#
# dalys_extracted = extract_results(
#     results_folder,
#     module="tlo.methods.healthburden",
#     key="dalys_stacked",
#     custom_generate_series=get_num_dalys,
#     do_scaling=True
# )
# # # Write the DataFrame to Excel
# deaths_extracted.to_excel(outputspath / "summarised_DALYs.xlsx", index=True)
#
#
#
#
#
#
#
#
















# def apply(results_folder: Path, outputspath: Path, resourcefilepath: Path = None):
#     target_period = (Date(2010, 1, 1), Date(2015, 12, 31))
#
#     # Definitions of general helper functions
#     make_graph_file_name = lambda stub: outputspath / f"{stub.replace('*', '_star_')}.png"  # noqa: E731
#
#     def target_period() -> str:
#         """Returns the target period as a string of the form YYYY-YYYY"""
#         return "-".join(str(t.year) for t in target_period)
#
#     def get_parameter_names_from_scenario_file() -> Tuple[str]:
#         """Get the tuple of names of the scenarios from `Scenario` class used to create the results."""
#         from scripts.hiv.projections_jan2023.scenario_impact_noXpert_diagnosis import (ImpactOfNOXpertDiagnosis)
#         e = ImpactOfNOXpertDiagnosis()
#         return tuple(e._scenarios.keys())
#
#     # Collecting basic information associated with the scenario
#
#     def get_num_deaths(_df):
#         """Return total number of Deaths (total within the TARGET_PERIOD)
#         """
#         return pd.Series(data=len(_df.loc[pd.to_datetime(_df.date).between(*target_period)]))
#
#     def get_num_dalys(_df):
#         """Return the total number of DALYS (Stacked) by label (total within the TARGET_PERIOD)"""
#         return pd.Series(
#             data=_df
#             .loc[_df.year.between(*[i.year for i in target_period])]
#             .drop(columns=[])  # drop(columns=['date', 'sex', 'age_range', 'year'])
#             .sum().sum()
#         )
#
#     def set_param_names_as_column_index_level_0(_df):
#         """Set the columns index (level 0) as the param_names."""
#         ordered_param_names_no_prefix = {i: x for i, x in enumerate(param_names)}
#         names_of_cols_level0 = [ordered_param_names_no_prefix.get(col) for col in _df.columns.levels[0]]
#         assert len(names_of_cols_level0) == len(_df.columns.levels[0])
#         _df.columns = _df.columns.set_levels(names_of_cols_level0, level=0)
#         return _df
#
#     param_names = get_parameter_names_from_scenario_file()
#
#     num_deaths = extract_results(
#         results_folder,
#         module='tlo.methods.demography',
#         key='death',
#         custom_generate_series=get_num_deaths,
#         do_scaling=True
#     ).pipe(set_param_names_as_column_index_level_0)
#
#     num_dalys = extract_results(
#         results_folder,
#         module='tlo.methods.healthburden',
#         key='dalys_stacked',
#         custom_generate_series=get_num_dalys,
#         do_scaling=True
#     ).pipe(set_param_names_as_column_index_level_0)
#
#     num_deaths_summarized = summarize(num_deaths).loc[0].unstack()
#     num_dalys_summarized = summarize(num_dalys).loc[0].unstack()
#
#     def make_plot(_df, annotations=None):
#         """Make a vertical bar plot for each row of _df, using the columns to identify the height of the bar and the
#          extent of the error bar."""
#         yerr = np.array([
#             (_df['mean'] - _df['lower']).values,
#             (_df['upper'] - _df['mean']).values,
#         ])
#         xticks = {(i + 0.5): k for i, k in enumerate(_df.index)}
#
#         fig, ax = plt.subplots()
#         ax.bar(
#             xticks.keys(),
#             _df['mean'].values,
#             yerr=yerr,
#             alpha=0.5,
#             ecolor='black',
#             capsize=10,
#         )
#         if annotations:
#             for xpos, ypos, text in zip(xticks.keys(), _df['mean'].values, annotations):
#                 ax.text(xpos, ypos, text, horizontalalignment='center')
#         ax.set_xticks(list(xticks.keys()))
#         ax.set_xticklabels(list(xticks.values()), rotation=90)
#         ax.grid(axis="y")
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         fig.tight_layout()
#         return fig, ax
#
#         # Plot for total number of DALYs from the scenario
#         name_of_plot = f'Total DALYS, {target_period()}'
#         fig, ax = make_plot(num_dalys_summarized / 1e6)
#         ax.set_title(name_of_plot)
#         ax.set_ylabel('DALYS (Millions)')
#         fig.tight_layout()
#         fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
#         fig.show()
#         plt.close(fig)
#
#         # plot of total number of deaths from the scenario
#         name_of_plot= f'Total Deaths, {target_period()}'
#         fig, ax = make_plot(num_deaths_summarized / 1e6)
#         ax.set_title(name_of_plot)
#         ax.set_ylabel('Deaths (Millions)')
#         fig.tight_layout()
#         fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
#         fig.show()
#         plt.close(fig)
#
# if __name__ == "__main__":
#     rfp = Path("./resources")
#
#     parser = argparse.ArgumentParser(
#         description="generate scenario plot",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )
#     parser.add_argument(
#         "--outputs-path",
#         help=(
#             "Directory to write outputs to. If not specified (set to None) outputs "
#             "will be written to value of --results-path argument."
#         ),
#         type=Path,
#         default=None,
#         required=False,
#     )
#     parser.add_argument(
#         "--resources-path",
#         help="Directory containing resource files",
#         type=Path,
#         default=Path('resources'),
#         required=False,
#     )
#     parser.add_argument(
#         "--results-path",
#         "--scenario-outputs-folder",
#         type=Path,
#         help=(
#             "Directory containing results from running "
#             "src/scripts/hiv/projections_jan2023/scenario_impact_of_noxpert_diagnosis.py "
#         ),
#         default=None,
#         required=False
#     )
#     args = parser.parse_args()
#     assert args.results_path is not None
#     results_path = args.results_path
#     output_path = results_path if args.outputs_path is None else args.outputs_path
#
#     apply(
#         results_folder=results_path,
#         outputspath=output_path,
#         resourcefilepath=args.resources_path
#     )
