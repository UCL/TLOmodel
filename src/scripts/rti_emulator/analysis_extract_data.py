"""Produce plots to show the health impact (deaths, dalys) each the healthcare system (overall health impact) when
running under different MODES and POLICIES (scenario_test_rti_emulator)"""

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
from tlo.analysis.utils import (
    CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP,
    extract_results,
    format_gbd,
    get_color_cause_of_death_or_daly_label,
    load_pickled_dataframes,
    make_age_grp_lookup,
    make_age_grp_types,
    make_calendar_period_lookup,
    make_calendar_period_type,
    order_of_cause_of_death_or_daly_label,
    plot_clustered_stacked,
    summarize,
)
tag = 'emulated_with_conditionality_Nothing'
#tag = 'normal_Nothing'
#tag = 'normal'

# Range of years considered
min_year = 2010
max_year = 2019

appt_dict = {'Under5OPD': 'OPD',
             'Over5OPD': 'OPD',
             'AntenatalFirst': 'AntenatalTotal',
             'ANCSubsequent': 'AntenatalTotal',
             'NormalDelivery': 'Delivery',
             'CompDelivery': 'Delivery',
             'EstMedCom': 'EstAdult',
             'EstNonCom': 'EstAdult',
             'VCTPositive': 'VCTTests',
             'VCTNegative': 'VCTTests',
             'DentAccidEmerg': 'DentalAll',
             'DentSurg': 'DentalAll',
             'DentU5': 'DentalAll',
             'DentO5': 'DentalAll',
             'MentOPD': 'MentalAll',
             'MentClinic': 'MentalAll'
             }

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
        from scripts.rti_emulator.scenario_test_rti_emulator import (
            GenerateDataChains,
        )
        e = GenerateDataChains()
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
 
    def get_annual_num_appts_by_level(results_folder: Path) -> pd.DataFrame:
        """Return pd.DataFrame gives the (mean) simulated annual number of appointments of each type at each level."""

        def get_counts_of_appts(_df):
            """Get the mean number of appointments of each type being used each year at each level.
            Need to rename appts to match standardized categories from the DHIS2 data."""

            def unpack_nested_dict_in_series(_raw: pd.Series):
                return pd.concat(
                    {
                      idx: pd.DataFrame.from_dict(mydict) for idx, mydict in _raw.items()
                     }
                 ).unstack().fillna(0.0).astype(int)

            return _df \
                .loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD), 'Number_By_Appt_Type_Code_And_Level'] \
                .pipe(unpack_nested_dict_in_series) \
                .rename(columns=appt_dict, level=1) \
                .rename(columns={'1b': '2'}, level=0) \
                .groupby(level=[0, 1], axis=1).sum() \
                .mean(axis=0)  # mean over each year (row)

        return summarize(
            extract_results(
                    results_folder,
                    module='tlo.methods.healthsystem.summary',
                    key='HSI_Event',
                    custom_generate_series=get_counts_of_appts,
                    do_scaling=True
                ),
            only_mean=False,
            collapse_columns=True,
            ).unstack().astype(int)
 
    def get_annual_num_appts_by_level_with_confidence_interval(results_folder: Path) -> pd.DataFrame:
        """Return pd.DataFrame gives the (mean) simulated annual number of appointments of each type at each level,
        with 95% confidence interval."""
        #param_names = get_parameter_names_from_scenario_file()
        def get_counts_of_appts(_df):
            """Get the mean number of appointments of each type being used each year at each level.
            Need to rename appts to match standardized categories from the DHIS2 data."""

            def unpack_nested_dict_in_series(_raw: pd.Series):
                return pd.concat(
                    {
                      idx: pd.DataFrame.from_dict(mydict) for idx, mydict in _raw.iteritems()
                     }
                 ).unstack().fillna(0.0).astype(int)

            return _df \
                .loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD), 'Number_By_Appt_Type_Code_And_Level'] \
                .pipe(unpack_nested_dict_in_series) \
                .rename(columns=appt_dict, level=1) \
                .rename(columns={'1b': '2'}, level=0) \
                .groupby(level=[0, 1], axis=1).sum() \
                .mean(axis=0)  # mean over each year (row)

        return summarize(
            extract_results(
                    results_folder,
                    module='tlo.methods.healthsystem.summary',
                    key='HSI_Event',
                    custom_generate_series=get_counts_of_appts,
                    do_scaling=True
                ),
            only_mean=False,
            collapse_columns=True,
            ).unstack().astype(int)
 
 
    model = get_annual_num_appts_by_level(results_folder=results_folder)
    model.to_csv('ConvertedOutputs/Emulator_Files/Total_Appt_Footprint_' + tag + '.csv', index=True)
    #exit(-1)
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
    dalys_by_year_summarise = summarize(dalys_by_year).unstack().astype(int)

    dalys_by_year.to_csv('ConvertedOutputs/Emulator_Files/Total_DALYs_with_time_' + tag + '.csv', index=True)
    dalys_by_year_summarise.to_csv('ConvertedOutputs/Emulator_Files/Total_DALYs_with_time_summarised_' + tag + '.csv', index=True)

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
    pop_model.to_csv('ConvertedOutputs/Emulator_Files/Population_with_time_' + tag + '.csv', index=True)

    # ================================================================================================
    # DEATHSs BROKEN DOWN BY CAUSES
    # %% Quantify the health losses associated with all interventions combined.

    def get_num_deaths_by_cause_label(_df):
        """Return total number of Deaths by label (total by age-group within the TARGET_PERIOD)
        """
        return _df \
            .loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)] \
            .groupby(_df['label']) \
            .size()

    num_deaths_by_cause_label = summarize(
        extract_results(
            results_folder,
            module='tlo.methods.demography',
            key='death',
            custom_generate_series=get_num_deaths_by_cause_label,
            do_scaling=True
        ).pipe(set_param_names_as_column_index_level_0)
    )
    print("Total number of deaths")
    print(num_deaths_by_cause_label)
    
    year_target = 2023 # This global variable will be passed to custom function
    def get_num_deaths_by_year_and_cause(_df):
        """Return total number of DALYs (Stacked) by label (total within the TARGET_PERIOD)"""
        newTARGET_PERIOD = (Date(year_target, 1, 1), Date(year_target, 12, 31))
        return _df \
            .loc[pd.to_datetime(_df.date).between(*newTARGET_PERIOD)] \
            .groupby(_df['label']) \
            .size()
                        #.loc[pd.to_datetime(_df.date).dt.year == year_target] \
    
    ALL = {}
    # Plot time trend show year prior transition as well to emphasise that until that point DALYs incurred
    # are consistent across different policies
    this_min_year = 2010
    for year in range(this_min_year, max_year+1):
        year_target = year
        num_deaths_by_year_and_cause = extract_results(
            results_folder,
            module='tlo.methods.demography',
            key='death',
            custom_generate_series=get_num_deaths_by_year_and_cause,
            do_scaling=True
        ).pipe(set_param_names_as_column_index_level_0)
        ALL[year_target] = num_deaths_by_year_and_cause #summarize(num_dalys_by_year)

    # Concatenate the DataFrames into a single DataFrame
    concatenated_df = pd.concat(ALL.values(), keys=ALL.keys())

    concatenated_df.index = concatenated_df.index.set_names(['date', 'cause'])
    
    df_total = concatenated_df
    print(df_total)
    print(df_total.groupby('cause').cumsum())
    print(summarize(df_total.groupby('cause').sum()))
    df_total.to_csv('ConvertedOutputs/Emulator_Files/Deaths_by_cause_with_time_' + tag + '.csv', index=True)
    df_total_summarise = summarize(df_total).unstack().astype(int)
    df_total_summarise.to_csv('ConvertedOutputs/Emulator_Files/Deaths_by_cause_with_time_summarised_' + tag + '.csv', index=True)


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
    print(df_total)
    df_total.to_csv('ConvertedOutputs/Emulator_Files/DALYS_by_cause_with_time_' + tag + '.csv', index=True)
    df_total_summarise = summarize(df_total).unstack().astype(int)
    print(df_total_summarise)
    df_total_summarise.to_csv('ConvertedOutputs/Emulator_Files/DALYS_by_cause_with_time_summarised_' + tag + '.csv', index=True)
    
    
    
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
    HSI_ran_by_year.to_csv('ConvertedOutputs/Emulator_Files/HSIs_ran_by_area_with_time_' + tag + '.csv', index=True)
    HSI_never_ran_by_year.to_csv('ConvertedOutputs/Emulator_Files/HSIs_never_ran_by_area_with_time_' + tag + '.csv', index=True)
    print(HSI_ran_by_year)
    print(HSI_never_ran_by_year)
    print(HSI_total_by_year)
    
    def get_dalys_by_period_sex_agegrp_label(df):
        """Sum the dalys by period, sex, age-group and label"""
        _, calperiodlookup = make_calendar_period_lookup()

        df['age_grp'] = df['age_range'].astype(make_age_grp_types())
        df["period"] = df["year"].map(calperiodlookup).astype(make_calendar_period_type())
        df = df.drop(columns=['date', 'age_range', 'year'])
        df = df.groupby(by=["period", "sex", "age_grp"]).sum().stack()
        df.index = df.index.set_names('label', level=3)
        return df
    
    results = extract_results(
    results_folder,
    module="tlo.methods.healthburden",
    key="dalys_stacked_by_age_and_time",  # <-- for DALYS stacked by age and time
    custom_generate_series=get_dalys_by_period_sex_agegrp_label,
    do_scaling=True
    )

    # divide by five to give the average number of deaths per year within the five year period:
    results = results.div(5.0)
    results_to_store = summarize((results.loc['2010-2014'] + results.loc['2015-2019'])/2)
    results_to_store.to_csv('ConvertedOutputs/Emulator_Files/DALYs_by_sex_age_' + tag + '.csv', index=True)

    
    def get_total_num_dalys_by_wealth_and_label(_df):
        """Return the total number of DALYS in the TARGET_PERIOD by wealth and cause label."""
        wealth_cats = {5: '0-19%', 4: '20-39%', 3: '40-59%', 2: '60-79%', 1: '80-100%'}

        return _df \
            .loc[_df['year'].between(*[d.year for d in TARGET_PERIOD])] \
            .drop(columns=['date', 'year']) \
            .assign(
                li_wealth=lambda x: x['li_wealth'].map(wealth_cats).astype(
                    pd.CategoricalDtype(wealth_cats.values(), ordered=True)
                )
            ).melt(id_vars=['li_wealth'], var_name='label') \
             .groupby(by=['li_wealth', 'label'])['value'] \
             .sum()

    total_num_dalys_by_wealth_and_label = summarize(extract_results(
            results_folder,
            module="tlo.methods.healthburden",
            key="dalys_by_wealth_stacked_by_age_and_time",
            custom_generate_series=get_total_num_dalys_by_wealth_and_label,
            do_scaling=True,
        ),

        collapse_columns=True,
        only_mean=False,
    ).unstack()
    print(total_num_dalys_by_wealth_and_label)
 
    total_num_dalys_by_wealth_and_label.to_csv('ConvertedOutputs/Emulator_Files/DALYs_by_wealth_' + tag + '.csv', index=True)
    print(total_num_dalys_by_wealth_and_label)
 
    
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
            "src/scripts/rti_emulator/scenario_test_rti_emulator "
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
