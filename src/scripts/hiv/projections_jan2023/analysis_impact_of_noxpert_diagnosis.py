"""
Extracts DALYs and mortality from the TB module
 """
# commands for running the analysis script in the terminal
# python src/scripts/hiv/projections_jan2023/analysis_impact_of_noxpert_diagnosis.py --scenario-outputs-folder outputs/nic503@york.ac.uk --show-figures
# python src/scripts/hiv/projections_jan2023/analysis_impact_of_noxpert_diagnosis.py --scenario-outputs-folder outputs/nic503@york.ac.uk --save-figures

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    make_age_grp_lookup,
    make_age_grp_types,
    summarize,
)
from tlo import Date

#outputspath = Path("./outputs")
outputspath = Path("./outputs/nic503@york.ac.uk")

target_period = (Date(2010, 1, 1), Date(2015, 12, 31))
def apply(results_folder: Path, outputspath: Path, resourcefilepath: Path = None):

    target_period = (Date(2010, 1, 1), Date(2015, 12, 31))
# Definitions of general helper functions
make_graph_file_name = lambda stub: outputspath / f"{stub.replace('*', '_star_')}.png"   # noqa: E731

# def target_period() -> str:
#     """Returns the target period as a string of the form YYYY-YYYY"""
#     return "-".join(str(t.year) for t in target_period)

# collecting basic information associated with scenario
# Find results_folder associated with a given batch_file and get most recent
results_folder = get_scenario_outputs('scenario_impact_noXpert_diagnosis.py', outputspath)[-1]
#results_folder = get_scenario_outputs('scenario_impact_noXpert_diagnosis.py', args.scenario_outputs_folder)[-1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

def get_num_deaths(_df):
    """Return total number of Deaths (total within the TARGET_PERIOD)"""
    return len(_df.loc[pd.to_datetime(_df['date'])])


def get_num_deaths(_df):
    """Return total number of deaths (Stacked) by label (total within the TARGET_PERIOD)"""
    return pd.Series(
        data=_df
        .loc[_df.year.between(*[i.year for i in target_period])]
        .drop(columns=[])     #drop(columns=['date', 'sex', 'age_range', 'year'])
        .sum().sum()
    )

def get_num_dalys(_df):
    """Return total number of DALYS (Stacked) by label (total within the TARGET_PERIOD)"""
    return pd.Series(
        data=_df
        .loc[_df.year.between(*[i.year for i in target_period])]
        .drop(columns=[])     #drop(columns=['date', 'sex', 'age_range', 'year'])
        .sum().sum()
    )

num_deaths = extract_results(
    results_folder,
    module='tlo.methods.demography',
    key='death',
    custom_generate_series=get_num_deaths,
    do_scaling=True
)
num_dalys = extract_results(
    results_folder,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=get_num_dalys,
    do_scaling=True
)

num_deaths_summarized = summarize(num_deaths).loc[0].unstack()
num_dalys_summarized = summarize(num_dalys).loc[0].unstack()

def make_plot(_df, annotations=None):
    """Make a vertical bar plot for each row of _df, using the columns to identify the height of the bar and the
     extent of the error bar."""
    yerr = np.array([
        (_df['mean'] - _df['lower']).values,
        (_df['upper'] - _df['mean']).values,
    ])

    xticks = {(i + 0.5): k for i, k in enumerate(_df.index)}

    fig, ax = plt.subplots()
    ax.bar(
        xticks.keys(),
        _df['mean'].values,
        yerr=yerr,
        alpha=0.5,
        ecolor='black',
        capsize=10,
    )
    if annotations:
        for xpos, ypos, text in zip(xticks.keys(), _df['mean'].values, annotations):
            ax.text(xpos, ypos, text, horizontalalignment='center')
    ax.set_xticks(list(xticks.keys()))
    ax.set_xticklabels(list(xticks.values()), rotation=90)
    ax.grid(axis="y")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    return fig, ax

# Plot for total number of DALYs from the scenario
name_of_plot = f'Total DALYS, {target_period()}'
fig, ax = make_plot(num_dalys_summarized / 1e6)
ax.set_title(name_of_plot)
ax.set_ylabel('DALYS (Millions)')
fig.tight_layout()
fig.savefig("DALY_graph.png")
plt.show()

# plot of total number of deaths from the scenario

name_of_plot= f'Total Deaths, {target_period()}'
fig, ax = make_plot(num_deaths_summarized / 1e6)
ax.set_title(name_of_plot)
ax.set_ylabel('Deaths (Millions)')
fig.tight_layout()
fig.savefig("Mortality_graph.png")
plt.show()


if __name__ == "__main__":

    rfp = Path("./resources")
   #rfp=Path("./resources")

    parser = argparse.ArgumentParser(
        description="generate scenario plot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--outputs-path",
        help=(
            "Directory to write outputs to. If not specified (set to None) outputs "
            "will be written to value of --results-path argument."
        ),
        type=Path,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--resources-path",
        help="Directory containing resource files",
        type=Path,
        default=Path('resources'),
        required=False,
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        help=(
            "Directory containing results from running "
            "src/scripts/hiv/projections_jan2023/scenario_impact_of_noxpert_diagnosis.py "
        ),
        default=None,
        required=False
    )
    args = parser.parse_args()
    assert args.results_path is not None
    results_path = args.results_path

    output_path = results_path if args.output_path is None else args.output_path

    apply(
        results_folder=results_path,
        output_folder=output_path,
        resourcefilepath=args.resources_path
    )

