"""Produce plots to show the health impact (deaths, dalys) each the healthcare system (overall health impact) when
running under different MODES and POLICIES (scenario_impact_of_policy.py)"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo.analysis.utils import (
    extract_results,
    make_age_grp_lookup,
    make_calendar_period_lookup,
    summarize,
)


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None, ):
    # Declare path for output graphs from this script
    make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_{stub}.png"  # noqa: E731

    # Define colo(u)rs to use:
    colors = {
        'Model': 'royalblue',
        'Census': 'darkred',
        'WPP': 'forestgreen',
        'GBD': 'plum'
    }

    # Define how to call the sexes:
    sexname = lambda x: 'Females' if x == 'F' else 'Males'  # noqa: E731

    # Get helpers for age and calendar period aggregation
    agegrps, agegrplookup = make_age_grp_lookup()
    calperiods, calperiodlookup = make_calendar_period_lookup()
    adult_age_groups = ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49']
    
    def get_parameter_names_from_scenario_file() -> Tuple[str]:
        """Get the tuple of names of the scenarios from `Scenario` class used to create the results."""
        from scripts.healthsystem.impact_of_policy.scenario_impact_of_policy import (
            ImpactOfHealthSystemMode,
        )
        e = ImpactOfHealthSystemMode()
        return tuple(e._scenarios.keys())
    
    def set_param_names_as_column_index_level_0(_df):
        """Set the columns index (level 0) as the param_names."""
        ordered_param_names_no_prefix = {i: x for i, x in enumerate(param_names)}
        names_of_cols_level0 = [ordered_param_names_no_prefix.get(col) for col in _df.columns.levels[0]]
        assert len(names_of_cols_level0) == len(_df.columns.levels[0])
        _df.columns = _df.columns.set_levels(names_of_cols_level0, level=0)
        return _df

    param_names = get_parameter_names_from_scenario_file()

    pop_model = extract_results(results_folder,
                                module="tlo.methods.demography",
                                key="population",
                                column="total",
                                index="date",
                                do_scaling=True
                                ).pipe(set_param_names_as_column_index_level_0)
    
    pop_model.index = pop_model.index.year
    pop_model_summarized = summarize(pop_model)
    df = pop_model_summarized
    
    df = df/1e6  # Convert into million DALYs
    del df["No Healthcare System"]
    df = df.rename(columns={'Naive status quo': 'NP status quo'})
    df = df.rename(columns={'EHP III status quo': 'HSSP-III HBP status quo'})
    df = df.rename(columns={'LCOA EHP status quo': 'LCOA status quo'})
    df = df.rename(columns={'Vertical Programmes status quo': 'VP status quo'})
    df = df.rename(columns={'Clinically Vulnerable status quo': 'CV status quo'})
    df = df.rename(columns={'CVD status quo': 'CMD status quo'})

    # Print the column levels
    column_levels = df.columns.levels
    print("Column Levels:")
    for level, labels in enumerate(column_levels):
        print(f"Level {level}: {labels}")
    
    unique_draws = df.columns.get_level_values('draw').unique()

    policy_colours = {
                  'No Policy':'#1f77b4',
                  'NP':'#1f77b4',
                  'RMNCH':'#e377c2',
                  'Clinically Vulnerable':'#9467bd',
                  'CV':'#9467bd',
                  'Vertical Programmes':'#2ca02c',
                  'VP':'#2ca02c',
                  'CVD':'#8c564b',
                  'CMD':'#8c564b',
                  'HSSP-III HBP':'#d62728',
                  'LCOA':'#ff7f0e'}

    plt.figure(figsize=(8, 6))
    for hs in ["status quo"]:
        for draw in unique_draws:
            if hs in draw:
                draw_data = df[draw]
                x_values = draw_data.index
                y_values = draw_data['mean']
                lower_bounds = draw_data['lower']
                upper_bounds = draw_data['upper']
                draw_label = draw.replace(" " + hs, "")
                plt.plot(x_values, y_values, label=f'{draw_label}', color = policy_colours[draw_label])
                plt.fill_between(x_values, lower_bounds, upper_bounds, alpha=0.2, color = policy_colours[draw_label])
        plt.title('Population size')
        plt.xlabel("Year")
        plt.ylabel("Population size (millions)")
        plt.grid(True)
        plt.ylim(20,40)
        plt.legend()
        start_year = 2022
        end_year = 2042
        tick_interval = 2
        xtick_positions = np.arange(start_year, end_year+1, tick_interval)
        xtick_labels = [str(year) for year in xtick_positions]
        plt.xticks(xtick_positions, xtick_labels, rotation=0)
        plt.xlim(2023,2042)
        plt.savefig('plots/Population_size.png')
        plt.close()
    
    def get_mean_pop_by_age_for_sex_and_year(sex, year):
        if sex == 'F':
            key = "age_range_f"
        else:
            key = "age_range_m"

        num_by_age = summarize(
            extract_results(results_folder,
                            module="tlo.methods.demography",
                            key=key, # This determines whether male or female
                            custom_generate_series=(
                                lambda df_: df_.loc[pd.to_datetime(df_.date).dt.year == year].drop(
                                    columns=['date']
                                ).melt(
                                    var_name='age_grp'
                                ).set_index('age_grp')['value']
                            ),
                            do_scaling=True
                            ).pipe(set_param_names_as_column_index_level_0),
            collapse_columns=True,
           # only_mean=True
        )
        return num_by_age
    
    lookup_sex = {'F' : 'Female', 'M': 'Male'}
    for year in [2042]:
        #for sex in ['F', 'M']:
        model_F = get_mean_pop_by_age_for_sex_and_year('F', year)
        model_M = get_mean_pop_by_age_for_sex_and_year('M', year)
        df = model_F + model_M
        
        del df["No Healthcare System"]
        df = df.rename(columns={'Naive status quo': 'NP status quo'})
        df = df.rename(columns={'EHP III status quo': 'HSSP-III HBP status quo'})
        df = df.rename(columns={'LCOA EHP status quo': 'LCOA status quo'})
        df = df.rename(columns={'Vertical Programmes status quo': 'VP status quo'})
        df = df.rename(columns={'Clinically Vulnerable status quo': 'CV status quo'})
        df = df.rename(columns={'CVD status quo': 'CMD status quo'})
        
        plt.figure(figsize=(8, 7))
        for hs in ["status quo"]:
            for draw in unique_draws:
                if hs in draw:
                    draw_data = df[draw]/1e6
                    x_values = draw_data.index
                    y_values = draw_data['mean']
                    lower_bounds = draw_data['lower']
                    upper_bounds = draw_data['upper']
                    draw_label = draw.replace(" " + hs, "")
                    plt.plot(x_values, y_values, label=f'{draw_label}', color = policy_colours[draw_label])
                    plt.fill_between(x_values, lower_bounds, upper_bounds, alpha=0.2, color = policy_colours[draw_label])
            plt.xlabel("Year group")
            plt.ylabel("Population size (millions)")
            plt.grid(True)
            plt.gca().set_aspect(2.2, adjustable='box')
            plt.xticks(rotation=90)
            plt.title(str(year))
            plt.savefig('plots/Pop_pyramid_overall_' + str(year) + '.png', dpi=500)


if __name__ == "__main__":
    rfp = Path('resources')

    parser = argparse.ArgumentParser(
        description="Produce plots to show the impact each set of treatments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-path",
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
            "src/scripts/healthsystem/impact_of_policy/scenario_impact_of_policy.py "
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
