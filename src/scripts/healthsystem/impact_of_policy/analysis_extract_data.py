"""Script to post-process Thanzi La Onse simulation outputs and extract the data presented in publication titled 'A new approach to Health Benefits Package design: an application of the Thanzi La Onse model in Malawi' by Margherita Molaro, Sakshi Mohan, Bingling She, Martin Chalkley, Tim Colbourn, Joseph H. Collins, Emilia Connolly, Matthew M. Graham, Eva Janoušková, Ines Li Lin, Gerald Manthalu, Emmanuel Mnjowe, Dominic Nkhoma, Pakwanja D. Twea, Andrew N. Phillips, Paul Revill, Asif U. Tamuri, Joseph Mfutso-Bengo, Tara Mangal, and Timothy B. Hallett.
"""

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


# Range of years considered
min_year = 2023
max_year = 2042


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None, ):

    TARGET_PERIOD = (Date(min_year, 1, 1), Date(max_year, 1, 1))

    make_graph_file_name = lambda stub: output_folder / f"{stub.replace('*', '_star_')}.png"

    def target_period() -> str:
        """Returns the target period as a string of the form YYYY-YYYY"""
        return "-".join(str(t.year) for t in TARGET_PERIOD)

    def get_parameter_names_from_scenario_file() -> Tuple[str]:
        """Get the tuple of names of the scenarios from `Scenario` class used to create the results."""
        from scripts.healthsystem.impact_of_policy.scenario_impact_of_policy import (
            ImpactOfHealthSystemMode,
        )
        e = ImpactOfHealthSystemMode()
        return tuple(e._scenarios.keys())

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
        
    # Obtain parameter names for this scenario file
    param_names = get_parameter_names_from_scenario_file()
    print(param_names)
    
    # Adjust names of policies considered to match those used in the publication.
    # We note that, in this publication, we opted to only present the cases identified by the 'status quo' label in the original scenario file. The acronyms used to refer to each policy are presented in Table 1 of the publication.
    param_names_list = list(param_names)
    param_names_list[param_names_list.index('Naive status quo')] = 'NP'
    param_names_list[param_names_list.index('RMNCH status quo')] = 'RMNCH'
    param_names_list[param_names_list.index('Clinically Vulnerable status quo')] = 'CV'
    param_names_list[param_names_list.index('Vertical Programmes status quo')] = 'VP'
    param_names_list[param_names_list.index('CVD status quo')] = 'CMD'
    param_names_list[param_names_list.index('EHP III status quo')] = 'HSSP-III HBP'
    param_names_list[param_names_list.index('LCOA EHP status quo')] = 'LCOA'
    param_names = tuple(param_names_list)
    print(param_names)

    # TIME EVOLUTION OF TOTAL DALYs
    year_target = 2023
    def get_num_dalys_by_year(_df):
        """Return total number of DALYs (Stacked) by label (total within the TARGET_PERIOD)"""
        return pd.Series(
            data=_df
            .loc[_df.year == year_target]
            .drop(columns=['date', 'sex', 'age_range', 'year'])
            .sum().sum()
        )
        
    ALL = {}

    for year in range(min_year, max_year+1):
        year_target = year
        num_dalys_by_year = extract_results(
            results_folder,
            module='tlo.methods.healthburden',
            key='dalys_stacked',
            custom_generate_series=get_num_dalys_by_year,
            do_scaling=True
        ).pipe(set_param_names_as_column_index_level_0)
        ALL[year_target] = num_dalys_by_year
    concatenated_df = pd.concat(ALL.values(), keys=ALL.keys())
    concatenated_df.index = concatenated_df.index.set_names(['date', 'index_original'])
    concatenated_df = concatenated_df.reset_index(level='index_original',drop=True)
    dalys_by_year = concatenated_df
    print(dalys_by_year)
    dalys_by_year = dalys_by_year.drop("No Healthcare System", axis=1, level=0)
    #dalys_by_year.to_csv('ConvertedOutputs/Total_DALYs_with_time.csv', index=True)
    
    pop_model = extract_results(results_folder,
                                module="tlo.methods.demography",
                                key="population",
                                column="total",
                                index="date",
                                do_scaling=True
                                ).pipe(set_param_names_as_column_index_level_0)
    
    pop_model.index = pop_model.index.year
    pop_model = pop_model[(pop_model.index >= min_year) & (pop_model.index <= max_year)]
    print(pop_model)
    pop_model = pop_model.drop("No Healthcare System", axis=1, level=0)
    assert dalys_by_year.index.equals(pop_model.index)
    assert all(dalys_by_year.columns == pop_model.columns)
    pop_model.to_csv('ConvertedOutputs/Population_with_time.csv', index=True)

    year_target = 2023
    def get_num_dalys_by_year_and_cause(_df):
        """Return total number of DALYs (Stacked) by label (total within the TARGET_PERIOD)"""
        return pd.Series(
            data=_df
            .loc[_df.year == year_target]
            .drop(columns=['date', 'sex', 'age_range', 'year'])
            .sum()
        )
        
    ALL = {}

    for year in range(min_year, max_year+1):
        year_target = year
        num_dalys_by_year = extract_results(
            results_folder,
            module='tlo.methods.healthburden',
            key='dalys_stacked',
            custom_generate_series=get_num_dalys_by_year_and_cause,
            do_scaling=True
        ).pipe(set_param_names_as_column_index_level_0)
        ALL[year_target] = num_dalys_by_year

    concatenated_df = pd.concat(ALL.values(), keys=ALL.keys())

    concatenated_df.index = concatenated_df.index.set_names(['date', 'cause'])
    
    df_total = concatenated_df
    df_total = df_total.drop("No Healthcare System", axis=1, level=0)
    df_total.to_csv('ConvertedOutputs/DALYS_by_cause_with_time.csv', index=True)

    ALL = {}
    for year in range(min_year, max_year+1):
        year_target = year
        
        hsi_delivered_by_year = extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='HSI_Event',
                custom_generate_series=get_counts_of_hsi_by_treatment_id_by_year,
                do_scaling=True
            ).pipe(set_param_names_as_column_index_level_0)
        ALL[year_target] = hsi_delivered_by_year

    # Concatenate the DataFrames into a single DataFrame
    concatenated_df = pd.concat(ALL.values(), keys=ALL.keys())
    concatenated_df.index = concatenated_df.index.set_names(['date', 'cause'])
    HSI_ran_by_year = concatenated_df
    HSI_ran_by_year = HSI_ran_by_year.drop("No Healthcare System", axis=1, level=0)
    del ALL
    
    ALL = {}
    
    for year in range(min_year, max_year+1):
        year_target = year
        
        hsi_not_delivered_by_year = extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='Never_ran_HSI_Event',
                custom_generate_series=get_counts_of_hsi_by_treatment_id_by_year,
                do_scaling=True
            ).pipe(set_param_names_as_column_index_level_0)
        ALL[year_target] = hsi_not_delivered_by_year

    concatenated_df = pd.concat(ALL.values(), keys=ALL.keys())
    concatenated_df.index = concatenated_df.index.set_names(['date', 'cause'])
    HSI_never_ran_by_year = concatenated_df
    HSI_never_ran_by_year = HSI_never_ran_by_year.drop("No Healthcare System", axis=1, level=0)
    HSI_never_ran_by_year = HSI_never_ran_by_year.fillna(0)
    HSI_ran_by_year = HSI_ran_by_year.fillna(0)
    HSI_total_by_year = HSI_ran_by_year.add(HSI_never_ran_by_year, fill_value=0)
    HSI_ran_by_year.to_csv('ConvertedOutputs/HSIs_delivered_by_type_and_level_with_time.csv', index=True)
    HSI_total_by_year.to_csv('ConvertedOutputs/HSIs_requested_by_type_and_facility_level_with_time.csv', index=True)
    
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
