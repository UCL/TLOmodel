"""Produce plots to show the health impact (deaths, dalys) each the healthcare system (overall health impact) when
running under different MODES and POLICIES (scenario_impact_of_actual_vs_funded.py)"""

# short tclose -> ideal case
# long tclose -> status quo
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from scipy.optimize import curve_fit

from tlo import Date
from tlo.analysis.utils import extract_results, summarize
from tlo.analysis.life_expectancy import get_life_expectancy_estimates


# Range of years considered
min_year = 2010
max_year = 2040


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None, ):
    """Produce standard set of plots describing the effect of each TREATMENT_ID.
    - We estimate the epidemiological impact as the EXTRA deaths that would occur if that treatment did not occur.
    - We estimate the draw on healthcare system resources as the FEWER appointments when that treatment does not occur.
    """

    TARGET_PERIOD = (Date(min_year, 1, 1), Date(max_year, 1, 1))

    # Definitions of general helper functions
    make_graph_file_name = lambda stub: output_folder / f"{stub.replace('*', '_star_')}.png"  # noqa: E731

    def target_period() -> str:
        """Returns the target period as a string of the form YYYY-YYYY"""
        return "-".join(str(t.year) for t in TARGET_PERIOD)

    def get_parameter_names_from_scenario_file() -> Tuple[str]:
        """Get the tuple of names of the scenarios from `Scenario` class used to create the results."""
        from scripts.healthsystem.impact_of_actual_vs_funded.scenario_impact_of_actual_vs_funded import (
            ImpactOfHealthSystemMode,
        )
        e = ImpactOfHealthSystemMode()
        return tuple(e._scenarios.keys())

    def get_num_deaths(_df):
        """Return total number of Deaths (total within the TARGET_PERIOD)
        """
        return pd.Series(data=len(_df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)]))

    def get_num_dalys(_df):
        """Return total number of DALYs (Stacked) by label (total within the TARGET_PERIOD)"""
        return pd.Series(
            data=_df
            .loc[_df.year.between(*[i.year for i in TARGET_PERIOD])]
            .drop(columns=['date', 'sex', 'age_range', 'year'])
            .sum().sum()
        )

    def get_num_dalys_by_cause(_df):
        """Return number of DALYs by cause by label (total within the TARGET_PERIOD)"""
        return pd.Series(
            data=_df
            .loc[_df.year.between(*[i.year for i in TARGET_PERIOD])]
            .drop(columns=['date', 'sex', 'age_range', 'year'])
            .sum()
        )

    def set_param_names_as_column_index_level_0(_df):
        """Set the columns index (level 0) as the param_names."""
        ordered_param_names_no_prefix = {i: x for i, x in enumerate(param_names)}
        names_of_cols_level0 = [ordered_param_names_no_prefix.get(col) for col in _df.columns.levels[0]]
        assert len(names_of_cols_level0) == len(_df.columns.levels[0])
        _df.columns = _df.columns.set_levels(names_of_cols_level0, level=0)
        return _df

    def find_difference_relative_to_comparison(_ser: pd.Series,
                                               comparison: str,
                                               scaled: bool = False,
                                               drop_comparison: bool = True,
                                               ):
        """Find the difference in the values in a pd.Series with a multi-index, between the draws (level 0)
        within the runs (level 1), relative to where draw = `comparison`.
        The comparison is `X - COMPARISON`."""
        return _ser \
            .unstack(level=0) \
            .apply(lambda x: (x - x[comparison]) / (x[comparison] if scaled else 1.0), axis=1) \
            .drop(columns=([comparison] if drop_comparison else [])) \
            .stack()

    
    def get_counts_of_hsi_by_treatment_id(_df):
        """Get the counts of the short TREATMENT_IDs occurring"""
        _counts_by_treatment_id = _df \
            .loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD), 'TREATMENT_ID'] \
            .apply(pd.Series) \
            .sum() \
            .astype(int)
        return _counts_by_treatment_id.groupby(level=0).sum()
        
    year_target = 2023
    def get_counts_of_hsi_by_treatment_id_by_year(_df):
        """Get the counts of the short TREATMENT_IDs occurring"""
        _counts_by_treatment_id = _df \
            .loc[pd.to_datetime(_df['date']).dt.year ==year_target, 'TREATMENT_ID'] \
            .apply(pd.Series) \
            .sum() \
            .astype(int)
        return _counts_by_treatment_id.groupby(level=0).sum()
    
    def get_counts_of_hsi_by_short_treatment_id(_df):
        """Get the counts of the short TREATMENT_IDs occurring (shortened, up to first underscore)"""
        _counts_by_treatment_id = get_counts_of_hsi_by_treatment_id(_df)
        _short_treatment_id = _counts_by_treatment_id.index.map(lambda x: x.split('_')[0] + "*")
        return _counts_by_treatment_id.groupby(by=_short_treatment_id).sum()
        
    def get_counts_of_hsi_by_short_treatment_id_by_year(_df):
        """Get the counts of the short TREATMENT_IDs occurring (shortened, up to first underscore)"""
        _counts_by_treatment_id = get_counts_of_hsi_by_treatment_id_by_year(_df)
        _short_treatment_id = _counts_by_treatment_id.index.map(lambda x: x.split('_')[0] + "*")
        return _counts_by_treatment_id.groupby(by=_short_treatment_id).sum()
 
        
    # Obtain parameter names for this scenario file
    param_names = get_parameter_names_from_scenario_file()
    print(param_names)

    # ================================================================================================
    # TIME EVOLUTION OF TOTAL DALYs
    # Plot DALYs averted compared to the ``No Policy'' policy
    
    year_target = 2023 # This global variable will be passed to custom function
    def get_num_dalys_by_year(_df):
        """Return total number of DALYs (Stacked) by label (total within the TARGET_PERIOD)"""
        return pd.Series(
            data=_df
            .loc[_df.year == year_target]
            .drop(columns=['date', 'sex', 'age_range', 'year'])
            .sum().sum()
        )
        
    ALL = {}
    # Plot time trend show year prior transition as well to emphasise that until that point DALYs incurred
    # are consistent across different policies
    this_min_year = 2010
    for year in range(this_min_year, max_year+1):
        year_target = year
        num_dalys_by_year = extract_results(
            results_folder,
            module='tlo.methods.healthburden',
            key='dalys_stacked',
            custom_generate_series=get_num_dalys_by_year,
            do_scaling=True
        ).pipe(set_param_names_as_column_index_level_0)
        ALL[year_target] = num_dalys_by_year
    # Concatenate the DataFrames into a single DataFrame
    concatenated_df = pd.concat(ALL.values(), keys=ALL.keys())
    concatenated_df.index = concatenated_df.index.set_names(['date', 'index_original'])
    concatenated_df = concatenated_df.reset_index(level='index_original',drop=True)
    dalys_by_year = concatenated_df
    print(dalys_by_year)
    dalys_by_year.to_csv('ConvertedOutputs/Total_DALYs_with_time.csv', index=True)
    
    # ================================================================================================
    # Print population under each scenario
    pop_model = extract_results(results_folder,
                                module="tlo.methods.demography",
                                key="population",
                                column="total",
                                index="date",
                                do_scaling=True
                                ).pipe(set_param_names_as_column_index_level_0)
    
    pop_model.index = pop_model.index.year
    pop_model = pop_model[(pop_model.index >= this_min_year) & (pop_model.index <= max_year)]
    print(pop_model)
    assert dalys_by_year.index.equals(pop_model.index)
    assert all(dalys_by_year.columns == pop_model.columns)
    pop_model.to_csv('ConvertedOutputs/Population_with_time.csv', index=True)

    # ================================================================================================
    # DALYs BROKEN DOWN BY CAUSES AND YEAR
    # DALYs by cause per year
    # %% Quantify the health losses associated with all interventions combined.
    
    year_target = 2023 # This global variable will be passed to custom function
    def get_num_dalys_by_year_and_cause(_df):
        """Return total number of DALYs (Stacked) by label (total within the TARGET_PERIOD)"""
        return pd.Series(
            data=_df
            .loc[_df.year == year_target]
            .drop(columns=['date', 'sex', 'age_range', 'year'])
            .sum()
        )
        
    ALL = {}
    # Plot time trend show year prior transition as well to emphasise that until that point DALYs incurred
    # are consistent across different policies
    this_min_year = 2010
    for year in range(this_min_year, max_year+1):
        year_target = year
        num_dalys_by_year = extract_results(
            results_folder,
            module='tlo.methods.healthburden',
            key='dalys_stacked',
            custom_generate_series=get_num_dalys_by_year_and_cause,
            do_scaling=True
        ).pipe(set_param_names_as_column_index_level_0)
        ALL[year_target] = num_dalys_by_year #summarize(num_dalys_by_year)

    # Concatenate the DataFrames into a single DataFrame
    concatenated_df = pd.concat(ALL.values(), keys=ALL.keys())

    concatenated_df.index = concatenated_df.index.set_names(['date', 'cause'])
    
    df_total = concatenated_df
    df_total.to_csv('ConvertedOutputs/DALYS_by_cause_with_time.csv', index=True)

    ALL = {}
    # Plot time trend show year prior transition as well to emphasise that until that point DALYs incurred
    # are consistent across different policies
    for year in range(min_year, max_year+1):
        year_target = year
        
        hsi_delivered_by_year = extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='HSI_Event',
                custom_generate_series=get_counts_of_hsi_by_short_treatment_id_by_year,
                do_scaling=True
            ).pipe(set_param_names_as_column_index_level_0)
        ALL[year_target] = hsi_delivered_by_year

    # Concatenate the DataFrames into a single DataFrame
    concatenated_df = pd.concat(ALL.values(), keys=ALL.keys())
    concatenated_df.index = concatenated_df.index.set_names(['date', 'cause'])
    HSI_ran_by_year = concatenated_df

    del ALL
    
    ALL = {}
    # Plot time trend show year prior transition as well to emphasise that until that point DALYs incurred
    # are consistent across different policies
    for year in range(min_year, max_year+1):
        year_target = year
        
        hsi_not_delivered_by_year = extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='Never_ran_HSI_Event',
                custom_generate_series=get_counts_of_hsi_by_short_treatment_id_by_year,
                do_scaling=True
            ).pipe(set_param_names_as_column_index_level_0)
        ALL[year_target] = hsi_not_delivered_by_year

    # Concatenate the DataFrames into a single DataFrame
    concatenated_df = pd.concat(ALL.values(), keys=ALL.keys())
    concatenated_df.index = concatenated_df.index.set_names(['date', 'cause'])
    HSI_never_ran_by_year = concatenated_df
    
    HSI_never_ran_by_year = HSI_never_ran_by_year.fillna(0) #clean_df(
    HSI_ran_by_year = HSI_ran_by_year.fillna(0)
    HSI_total_by_year = HSI_ran_by_year.add(HSI_never_ran_by_year, fill_value=0)
    HSI_ran_by_year.to_csv('ConvertedOutputs/HSIs_ran_by_area_with_time.csv', index=True)
    HSI_never_ran_by_year.to_csv('ConvertedOutputs/HSIs_never_ran_by_area_with_time.csv', index=True)
    print(HSI_ran_by_year)
    print(HSI_never_ran_by_year)
    print(HSI_total_by_year)
    
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
            "src/scripts/healthsystem/impact_of_actual_vs_funded/scenario_impact_of_actual_vs_funded.py "
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
