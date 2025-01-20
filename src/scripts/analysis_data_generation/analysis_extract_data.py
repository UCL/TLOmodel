"""Produce plots to show the health impact (deaths, dalys) each the healthcare system (overall health impact) when
running under different MODES and POLICIES (scenario_impact_of_actual_vs_funded.py)"""

# short tclose -> ideal case
# long tclose -> status quo
import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt

from tlo import Date
from tlo.analysis.utils import extract_results
from datetime import datetime
from collections import Counter
import ast

# Time simulated to collect data
start_date = Date(2010, 1, 1)
end_date = start_date + pd.DateOffset(months=13)

# Range of years considered
min_year = 2010
max_year = 2040


def all_columns(_df):
    return pd.Series(_df.all())

def check_if_beyond_time_range_considered(progression_properties):
    matching_keys = [key for key in progression_properties.keys() if "rt_date_to_remove_daly" in key]
    if matching_keys:
        for key in matching_keys:
            if progression_properties[key] > end_date:
                print("Beyond time range considered, need at least ",progression_properties[key])

def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None, ):
    """Produce standard set of plots describing the effect of each TREATMENT_ID.
    - We estimate the epidemiological impact as the EXTRA deaths that would occur if that treatment did not occur.
    - We estimate the draw on healthcare system resources as the FEWER appointments when that treatment does not occur.
    """
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)
    
    eval_env = {
        'datetime': datetime,  # Add the datetime class to the eval environment
        'pd': pd,              # Add pandas to handle Timestamp
        'Timestamp': pd.Timestamp,  # Specifically add Timestamp for eval
        'NaT': pd.NaT,
        'nan': float('nan'),       # Include NaN for eval (can also use pd.NA if preferred)
    }
    
    initial_properties_of_interest = ['rt_MAIS_military_score','rt_ISS_score','rt_disability','rt_polytrauma','rt_injury_1','rt_injury_2','rt_injury_3','rt_injury_4','rt_injury_5','rt_injury_6', 'rt_imm_death','sy_injury','sy_severe_trauma','sex','li_urban', 'li_wealth', 'li_mar_stat', 'li_in_ed', 'li_ed_lev']

    # Will be added through computation: age at time of RTI
    # Will be added through computation: total duration of event
    
    initial_rt_event_properties = set()

    num_individuals = 1000
    num_runs = 50
    record = []
    # Include results folder in output file name
    name_tag = str(results_folder).replace("outputs/", "")


    
    for p in range(0,num_individuals):
    
        print("At person = ", p, " out of ", num_individuals)

        individual_event_chains = extract_results(
                results_folder,
                module='tlo.simulation',
                key='event_chains',
                column=str(p),
                do_scaling=False
            )
            
        for r in range(0,num_runs):
            initial_properties = {}
            key_first_event = {}
            key_last_event = {}
            first_event = {}
            last_event = {}
            properties = {}
            average_disability = 0
            total_dt_included = 0
            dt_in_prev_disability = 0
            prev_disability_incurred = 0
            ind_Counter = {'0': Counter(), '1a': Counter(), '1b' : Counter(), '2' : Counter()}
            # Count total appts

            list_for_individual = []
            for item,row in individual_event_chains.iterrows():
                value = individual_event_chains.loc[item,(0, r)]
                if value !='' and isinstance(value, str):
                    evaluated = eval(value, eval_env)
                    list_for_individual.append(evaluated)
                    
            # These are the properties of the individual before the start of the chain of events
            initial_properties = list_for_individual[0]
            
            # Initialise first event by gathering parameters of interest from initial_properties
            first_event = {key: initial_properties[key] for key in initial_properties_of_interest if key in initial_properties}
            
            # The changing or adding of properties from the first_event will be stored in progression_properties
            progression_properties = {}
            
            for i in list_for_individual:
                # Skip the initial_properties, or in other words only consider these if they are 'proper' events
                if 'event' in i:
                    #print(i)
                    if 'RTIPolling' in i['event']:
                        
                        # Keep track of which properties are changed during polling events
                        for key,value in i.items():
                            if 'rt_' in key:
                                initial_rt_event_properties.add(key)
                        
                        # Retain a copy of Polling event
                        polling_event = i.copy()
                        
                        # Update parameters of interest following RTI
                        key_first_event = {key: i[key] if key in i else value for key, value in first_event.items()}
                        
                        # Calculate age of individual at time of event
                        key_first_event['age_in_days_at_event'] = (i['rt_date_inj'] - initial_properties['date_of_birth']).days
                        
                        # Keep track of evolution in individual's properties
                        progression_properties = initial_properties.copy()
                        progression_properties.update(i)
                        
                        # Initialise chain of Dalys incurred
                        if 'rt_disability' in i:
                            prev_disability_incurred = i['rt_disability']
                            prev_date = i['event_date']

                    else:
                        # Progress properties of individual, even if this event is a death
                        progression_properties.update(i)
                        
                        # If disability has changed as a result of this, recalculate and add previous to rolling average
                        if 'rt_disability' in i:
    
                            dt_in_prev_disability = (i['event_date'] - prev_date).days
                            #print("Detected change in disability", i['rt_disability'], "after dt=", dt_in_prev_disability)
                            #print("Adding the following to the average", prev_disability_incurred, " x ", dt_in_prev_disability )
                            average_disability += prev_disability_incurred*dt_in_prev_disability
                            total_dt_included += dt_in_prev_disability
                            # Update variables
                            prev_disability_incurred = i['rt_disability']
                            prev_date = i['event_date']

                    # Update running footprint
                    if 'appt_footprint' in i and i['appt_footprint'] != 'Counter()':
                        footprint = i['appt_footprint']
                        if 'Counter' in footprint:
                            footprint = footprint[len("Counter("):-1]
                        apply = eval(footprint, eval_env)
                        ind_Counter[i['level']].update(Counter(apply))
                    
                    # If the individual has died, ensure chain of event is interrupted here and update rolling average of DALYs
                    if 'is_alive' in i and i['is_alive'] is False:
                        if ((i['event_date'] - polling_event['rt_date_inj']).days) > total_dt_included:
                            dt_in_prev_disability = (i['event_date'] - prev_date).days
                            average_disability += prev_disability_incurred*dt_in_prev_disability
                            total_dt_included += dt_in_prev_disability
                        break
               
            # check_if_beyond_time_range_considered(progression_properties)
            
            # Compute final properties of individual
            key_last_event['is_alive_after_RTI'] = progression_properties['is_alive']
            key_last_event['duration_days'] = (progression_properties['event_date'] - polling_event['rt_date_inj']).days

            # If individual didn't die and the key_last_event didn't result in a final change in DALYs, ensure that the last change is recorded here
            if not key_first_event['rt_imm_death'] and (total_dt_included < key_last_event['duration_days']):
                #print("Number of events", len(list_for_individual))
                #for i in list_for_individual:
                #    if 'event' in i:
                #        print(i)
                dt_in_prev_disability = (progression_properties['event_date'] - prev_date).days
                average_disability += prev_disability_incurred*dt_in_prev_disability
                total_dt_included += dt_in_prev_disability

            # Now calculate the average disability incurred, and store any permanent disability and total footprint
            if not key_first_event['rt_imm_death'] and key_last_event['duration_days']> 0:
                key_last_event['rt_disability_average'] = average_disability/key_last_event['duration_days']
            else:
                key_last_event['rt_disability_average'] = 0.0
            
            key_last_event['rt_disability_permanent'] = progression_properties['rt_disability']
            key_last_event.update({'total_footprint': ind_Counter})

            if key_last_event['duration_days']!=total_dt_included:
                print("The duration of event and total_dt_included don't match", key_last_event['duration_days'], total_dt_included)
                exit(-1)
            
            properties = key_first_event | key_last_event
                
            record.append(properties)
            

    df = pd.DataFrame(record)
    df.to_csv("new_raw_data_" + name_tag + ".csv", index=False)
    
    print(df)
    print(initial_rt_event_properties)
    exit(-1)
            #print(i)

    #dict = {}
    #for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    #    dict[i] = []

    #for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    #    event_chains = extract_results(
    #        results_folder,
    #        module='tlo.simulation'#,
    #        key='event_chains',
    #        column = str(i),
    #        #custom_generate_series=get_num_dalys_by_year,
    #        do_scaling=False
    #    )
    #    print(event_chains)
    #    print(event_chains.index)
    #    print(event_chains.columns.levels)

    #    for index, row in event_chains.iterrows():
    #        if event_chains.iloc[index,0] is not None:
    #            if(event_chains.iloc[index,0]['person_ID']==i): #and 'event' in event_chains.iloc[index,0].keys()):
    #                dict[i].append(event_chains.iloc[index,0])
            #elif (event_chains.iloc[index,0]['person_ID']==i and 'event' not in event_chains.iloc[index,0].keys()):
                #print(event_chains.iloc[index,0]['de_depr'])
               # exit(-1)
    #for item in dict[0]:
    #    print(item)
    
    #exit(-1)
    
    TARGET_PERIOD = (Date(min_year, 1, 1), Date(max_year, 1, 1))

    # Definitions of general helper functions
    lambda stub: output_folder / f"{stub.replace('*', '_star_')}.png"  # noqa: E731

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
            "src/scripts/analysis_data_generation/scenario_generate_chains.py "
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
