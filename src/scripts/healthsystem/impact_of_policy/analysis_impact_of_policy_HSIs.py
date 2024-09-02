"""Produce plots to show the health impact (deaths, dalys) each the healthcare system (overall health impact) when
running under different MODES and POLICIES (scenario_impact_of_policy.py)"""

import argparse
from pathlib import Path
from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from tlo import Date
from tlo.analysis.utils import extract_results, summarize

min_year = 2023
max_year = 2042


policy_colours = {
                  'No Policy':'#1f77b4',
                  'RMNCH':'#e377c2',
                  'Clinically Vulnerable':'#9467bd',
                  'Vertical Programmes':'#2ca02c',
                  'CVD':'#8c564b',
                  'HSSP-III EHP':'#d62728',
                  'LCOA EHP':'#ff7f0e'}

new_order = ['No Policy status quo', 'HSSP-III EHP status quo', 'LCOA EHP status quo', 'Vertical Programmes status quo', 'RMNCH status quo', 'Clinically Vulnerable status quo', 'CVD status quo']

color_palette = {
    'Contraception*': '#1F449C',
    'Malaria*': '#F05039',
    'Hiv*': 'blue',
    'Epi*': 'orange',
    'Tb*': 'purple',
    'FirstAttendance*': 'black',
    'CardioMetabolicDisorders*': 'cyan',
    'Diarrhoea*': 'magenta',
    'Epilepsy*': 'yellow',
    'Alri*': 'brown',
    'Depression*': 'pink',
    'Measles*': 'olive',
    'Inpatient*': 'teal',
    'BreastCancer*': 'lime',
    'OtherAdultCancer*': 'gray',
    'BladderCancer*': 'indigo',
    'Undernutrition*': 'violet',
    'Rti*': 'salmon',
    'OesophagealCancer*': 'gold',
    'ProstateCancer*': 'darkcyan',
    'Schisto*': 'darkred',
    'AntenatalCare*': 'skyblue',
    'Copd*': 'lightcoral',
    'PostnatalCare*': 'lightgreen',
    'DeliveryCare*': 'darkorange',
}


def SortTuple(tup): 
   
    return(sorted(tup, key = lambda x: x[0]))  

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
        from scripts.healthsystem.impact_of_policy.scenario_impact_of_policy import (
            ImpactOfHealthSystemMode,
        )
        e = ImpactOfHealthSystemMode()
        return tuple(e._scenarios.keys())
        
    def clean_df(df):
        df = df.rename(columns={'Naive ideal case': 'No Policy ideal case'})
        df = df.rename(columns={'Naive status quo': 'No Policy status quo'})
        df = df.rename(columns={'EHP III ideal case': 'HSSP-III EHP ideal case'})
        df = df.rename(columns={'EHP III status quo': 'HSSP-III EHP status quo'})
        df = df[new_order]
        return df

    def get_num_deaths(_df):
        """Return total number of Deaths (total within the TARGET_PERIOD)
        """
        return pd.Series(data=len(_df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)]))

    def get_num_dalys(_df):
        """Return total number of DALYS (Stacked) by label (total within the TARGET_PERIOD)"""
        return pd.Series(
            data=_df
            .loc[_df.year.between(*[i.year for i in TARGET_PERIOD])]
            .drop(columns=['date', 'sex', 'age_range', 'year'])
            .sum().sum()
        )

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

    def plot_high_priority_treatments(level_considered, _df, what, priority_level):

        df = _df/1e6

        # Only keep HSIs for level considered
        df = df[df.index.str.contains(level_considered)]

        # Remove _Level_, and some over
        clean_id = df.index.map(lambda x: x.split('_Level')[0])
        df_clean = df.groupby(by=clean_id).sum()

        # Only keep mean, and remove 'status quo' from name of policy
        df_clean = ((df_clean.loc[:,df_clean.columns.get_level_values('stat') == 'mean']).droplevel(level='stat', axis=1))
        df_clean.columns = [col.replace(' status quo', '') for col in df_clean.columns]

        priority_rank = pd.read_excel('resources/healthsystem/priority_policies/ResourceFile_PriorityRanking_ALLPOLICIES.xlsx',sheet_name=None)
                            
        merged_df_2 = pd.DataFrame(index=df_clean.index)

        # Convert 'Name' column into index
        for policy in df_clean.columns:
            if policy == 'Clinically Vulnerable':
                continue

            # Get policy name to use in the priority_rank
            policy_name = policy.replace(' ', '_')
            if policy == 'No Policy':
                policy_name = 'Naive'
            elif policy == 'Vertical Programmes':
                policy_name = 'VerticalProgrammes'
            elif policy == 'HSSP-III EHP':
                policy_name = 'EHP_III'

            
            policy_priority = pd.DataFrame(priority_rank[policy_name])
            policy_priority.set_index('Treatment', inplace=True)
            
            HSI_count = df_clean[policy]

            merged_df = pd.merge(HSI_count, policy_priority,left_index=True, right_index=True, how='inner')
            merged_df_2[policy] = merged_df[policy].where(merged_df['Priority']==priority_level, 0)

        _short_treatment_id = merged_df_2.index.map(lambda x: x.split('_')[0] + "*")
        final_high_priority = merged_df_2.groupby(by=_short_treatment_id).sum()
        if 'FirstAttendance*' in final_high_priority.index:
            final_high_priority = final_high_priority.drop('FirstAttendance*')
        ax = final_high_priority.T.plot.bar(stacked=True, figsize=(10, 6),color=color_palette)

        # Customize the plot
        plt.title(level_considered)
        plt.ylabel('Number of high-priority treatments ' + what + ' (millions)')
        plt.ylim(0,1100)
        plt.legend(title='Columns', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('plots/High_priority_' + what + '_at_' + level_considered + '.png', dpi = 300)
        
    colours = ("#0000CC","#00CCCC","#994C00","#FF8000","#660066","#CC99FF","#009900","#FF0000", "#FF6666")
    hatches = ("*","*","*","*","*","*","*","*","*","*","*","*","*","*","*",)

    pd.set_option('display.max_rows', None)

    # %% Define parameter names
    param_names = get_parameter_names_from_scenario_file()
    
    # Time evolution of HSIs requested and delivered under different areas of health
    plot_1 = True
    
    #================================================================================
    # HSIs delivered and requested by year under different areas of health
    if plot_1:

        ALL = {}
        # Plot time trend show year prior transition as well to emphasise that until that point DALYs incurred
        # are consistent across different policies
        for year in range(min_year, max_year+1):
            year_target = year
            
            hsi_delivered_by_year = summarize(
                extract_results(
                    results_folder,
                    module='tlo.methods.healthsystem.summary',
                    key='HSI_Event',
                    custom_generate_series=get_counts_of_hsi_by_short_treatment_id_by_year,
                    do_scaling=True
                ),
                only_mean=True,
                collapse_columns=True,
            ).pipe(set_param_names_as_column_index_level_0)
            ALL[year_target] = hsi_delivered_by_year

        concatenated_df = pd.concat(ALL.values(), keys=ALL.keys())
        concatenated_df.index = concatenated_df.index.set_names(['date', 'cause'])
        HSI_ran_by_year = concatenated_df

        del ALL
        
        ALL = {}

        for year in range(min_year, max_year+1):
            year_target = year
            
            hsi_not_delivered_by_year = summarize(
                extract_results(
                    results_folder,
                    module='tlo.methods.healthsystem.summary',
                    key='Never_ran_HSI_Event',
                    custom_generate_series=get_counts_of_hsi_by_short_treatment_id_by_year,
                    do_scaling=True
                ),
                only_mean=True,
                collapse_columns=True,
            ).pipe(set_param_names_as_column_index_level_0)
            ALL[year_target] = hsi_not_delivered_by_year

        # Concatenate the DataFrames into a single DataFrame
        concatenated_df = pd.concat(ALL.values(), keys=ALL.keys())
        concatenated_df.index = concatenated_df.index.set_names(['date', 'cause'])
        HSI_never_ran_by_year = concatenated_df
        
        HSI_never_ran_by_year = clean_df(HSI_never_ran_by_year.fillna(0))/1e6
        HSI_ran_by_year = clean_df(HSI_ran_by_year.fillna(0))/1e6
        HSI_total_by_year = HSI_ran_by_year.add(HSI_never_ran_by_year, fill_value=0)
            
        yranges = {'Hiv' : (2,14), 'Alri' : (0, 1.5), 'Malaria' : (15,35), 'Tb' : (0,12), 'AntenatalCare' : (0,1.8), 'DeliveryCare' : (0,0.225), 'Diarrhoea' : (0,2.2), 'Depression' : (0,0.5), 'Measles': (0,0.8)}
        leading_causes = ['Hiv*', 'Malaria*', 'Measles*', 'Rti*', 'Tb*', 'Depression*', 'Alri*', 'AntenatalCare*', 'PostnatalCare*', 'DeliveryCare*','Diarrhoea*']

        unique_draws = HSI_total_by_year.columns.get_level_values('draw').unique()

        # Create a custom legend
        legend_elements = [Line2D([0], [0], linestyle='-', color='black', label='Requested'),
                            Line2D([0], [0], linestyle='--', color='black', label='Delivered')]

        for cause in leading_causes:
            plt.figure(figsize=(8, 6))
            cause_df = HSI_total_by_year.xs(cause, level='cause')
            cause_df_ran = HSI_ran_by_year.xs(cause, level='cause')
            for hs in ["status quo"]:
                for draw in unique_draws:
                    if hs in draw:
                        draw_data = cause_df[draw]
                        x_values = draw_data.index.get_level_values('date')
                        y_values = draw_data['mean']
                        lower_bounds = draw_data['lower']
                        upper_bounds = draw_data['upper']
                        draw_label = draw.replace(" " + hs, "")
                        plt.plot(x_values, y_values, label=f'{draw_label}', color = policy_colours[draw_label])
                        plt.fill_between(x_values, lower_bounds, upper_bounds, alpha=0.2, color = policy_colours[draw_label])
                        
                        # Include ran ones on same plot
                        draw_data = cause_df_ran[draw]
                        x_values = draw_data.index.get_level_values('date')
                        y_values = draw_data['mean']
                        lower_bounds = draw_data['lower']
                        upper_bounds = draw_data['upper']
                        plt.plot(x_values, y_values, label=f'{draw_label}', color = policy_colours[draw_label], linestyle="--")
                        plt.fill_between(x_values, lower_bounds, upper_bounds, alpha=0.2, color = policy_colours[draw_label])
                
                cause_label = cause.replace('/', '_or_')
                cause_label = cause.replace('*', '')
                if cause_label == 'Hiv':
                    cause_label = 'HIV'

                plt.title(f'{cause_label}')
                plt.xlabel("Year")
                plt.ylabel("Number of treatments (millions) per year")
                start_year = 2022
                end_year = 2042
                tick_interval = 2
                xtick_positions = np.arange(start_year, end_year+1, tick_interval)
                xtick_labels = [str(year) for year in xtick_positions]            # plt.legend()
                plt.xticks(xtick_positions, xtick_labels)  # Rotate labels for better visibility
                plt.xlim(min_year,end_year)
                plt.grid(True)
                name_fig = 'plots/HSI_time_evolution_'+cause_label+ "_" + str(min_year) + '-' + str(max_year) + '.png'
                name_fig = name_fig.replace(" ", "_")
                # Add legend to the plot
                if cause_label == 'HIV':
                    plt.legend(handles=legend_elements, loc='upper right')
                else:
                    plt.legend(handles=legend_elements, loc='upper left')
                plt.tight_layout()
                plt.savefig(name_fig, dpi=500)
                plt.close()

    #================================================================================
    # HSIs delivered and requested over entire period
    counts_of_hsi_by_treatment_id = summarize(
        extract_results(
            results_folder,
            module='tlo.methods.healthsystem.summary',
            key='HSI_Event',
            custom_generate_series=get_counts_of_hsi_by_treatment_id,
            do_scaling=True
        ),
        only_mean=True,
        collapse_columns=True,
    ).pipe(set_param_names_as_column_index_level_0)

    counts_of_hsi_by_treatment_id_short = summarize(
        extract_results(
            results_folder,
            module='tlo.methods.healthsystem.summary',
            key='HSI_Event',
            custom_generate_series=get_counts_of_hsi_by_short_treatment_id,
            do_scaling=True
        ),
        only_mean=True,
        collapse_columns=True,
    ).pipe(set_param_names_as_column_index_level_0)

    counts_of_never_ran_hsi_by_treatment_id = summarize(
        extract_results(
            results_folder,
            module='tlo.methods.healthsystem.summary',
            key='Never_ran_HSI_Event',
            custom_generate_series=get_counts_of_hsi_by_treatment_id,
            do_scaling=True
        ),
        only_mean=True,
        collapse_columns=True,
    ).pipe(set_param_names_as_column_index_level_0)

    counts_of_never_ran_hsi_by_treatment_id_short = summarize(
        extract_results(
            results_folder,
            module='tlo.methods.healthsystem.summary',
            key='Never_ran_HSI_Event',
            custom_generate_series=get_counts_of_hsi_by_short_treatment_id,
            do_scaling=True
        ),
        only_mean=True,
        collapse_columns=True,
    ).pipe(set_param_names_as_column_index_level_0)

    df1 = clean_df(counts_of_hsi_by_treatment_id.fillna(0))
    df2 = clean_df(counts_of_never_ran_hsi_by_treatment_id.fillna(0))
    df = df1.add(df2, fill_value=0)
    
    df1_short = clean_df(counts_of_hsi_by_treatment_id_short.fillna(0))
    df2_short = clean_df(counts_of_never_ran_hsi_by_treatment_id_short.fillna(0))
    df_short = df1_short.add(df2_short, fill_value=0)
    
    for level in ['Level_1a', 'Level_2']:
        plot_high_priority_treatments(level, df1, "Delivered", 2)
        plot_high_priority_treatments(level, df2, "Never delivered", 2)
        plot_high_priority_treatments(level, df, "Requested", 2)

    
    treatments_id_short = tuple(counts_of_hsi_by_treatment_id_short.index)

    def make_plot_for_each_short_id(df1, description):
    
        colours_keys = {"No Healthcare System": "#808080",
                        "No Policy ideal case": "#0051ff",
                        "HSSP-III EHP ideal case": "#FF0000","LCOA EHP ideal case": "#FF6666",
                        "RMNCH ideal case": "#660066",
                        "Clinically Vulnerable ideal case": "#CC99FF",
                        "Vertical Programmes ideal case": "#009900",
                        "CVD ideal case": "#994C00",
                        "No Policy status quo": "#0051ff",
                        "HSSP-III EHP status quo": "#FF0000","LCOA EHP status quo": "#FF6666",
                        "RMNCH status quo": "#660066",
                        "Clinically Vulnerable status quo": "#CC99FF",
                        "Vertical Programmes status quo": "#009900",
                        "CVD status quo": "#994C00"
                        }
                                
        #print(df1.columns.get_level_values('draw').unique())
        for policy in df1.columns.get_level_values('draw').unique():
            if "ideal" in policy:
                del df1[policy]

        for i in treatments_id_short:
            i = i.rstrip("*")
            short = i
            if short == 'Hiv':
                short = 'HIV'

            df_sub = df1[df1.index.str.contains(i+"_")]
            df_sub_mean = ((df_sub.loc[:,df_sub.columns.get_level_values('stat') == 'mean']).droplevel(level='stat', axis=1)).T
            df_sub_lower = ((df_sub.loc[:,df_sub.columns.get_level_values('stat') == 'lower']).droplevel(level='stat', axis=1)).T
            df_sub_upper = ((df_sub.loc[:,df_sub.columns.get_level_values('stat') == 'upper']).droplevel(level='stat', axis=1)).T
            if i == 'Hiv':
                print(description,df_sub_mean)
            width = 0.1

            # Create an array for the x-axis values
            x = np.arange(len(df_sub_mean.columns))

            # Create the bar plot
            fig, ax = plt.subplots(figsize=(6, 6))

            for i, (label, row) in enumerate(df_sub_mean.iterrows()):
                x_adjusted = [pos + i * width for pos in x]
                draw_label = label.replace(" status quo", "")
                ax.bar(x_adjusted, row, width, label=label, color=policy_colours[draw_label])

            include_errorbars = False
            if include_errorbars:
                # Create error bars using lower and upper bounds
                for i, (label, row) in enumerate(df_sub_mean.iterrows()):
                    x_adjusted = [pos + i * width for pos in x]
                    yerr = [row - df_sub_lower.loc[label], df_sub_upper.loc[label] - row]
                    #yerr[yerr < 1e-12] = 1e-12
                    ax.errorbar(
                        x_adjusted,
                        row,
                        yerr=yerr,
                        fmt='none',
                        ecolor='black'
                    )

            x_tick_labels = [label.replace(short+"_", "") for label in df_sub_mean.columns]
            
            if description == "Ran":
                ax.set_ylabel("Number of treatments delivered")
            elif description == "Never ran":
                ax.set_ylabel("Mean number of treatments never delivered")
            elif description == "Total":
                ax.set_ylabel("Total treatments requested")

            ax.set_title(short)
            ax.grid(axis="y")

            ax.set_xticks([pos + 3*width for pos in x])
            ax.set_xticklabels(x_tick_labels, rotation=90)
            ax.set_yscale('log')
            plt.tight_layout()
            
            plt.savefig(str(output_folder) + '/HSIs_' + description +'_Breakdown_'+ short + '_' + target_period() + '.png')

    make_plot_for_each_short_id(df1, "Ran")
    make_plot_for_each_short_id(df, "Total")

    # CREATE NETWORK OF HSIs COMPETING FOR SAME OFFICERS TIME
    
    hsi_events = pd.read_csv("hsi_events_hot_read.csv")
    appt_time = pd.read_csv("resources/healthsystem/human_resources/definitions/ResourceFile_Appt_Time_Table.csv")
    officer_types = tuple(set(appt_time['Officer_Category']))
    competing_resources = ("Treatment_ID", "Appt_Footprint", "Level") + officer_types 
    Table = pd.DataFrame(columns = competing_resources)

    #Replace nans with empty strings
    hsi_events.fillna("", inplace=True)
    pd.set_option('display.max_columns', None)

    # Match appt footprints with time request from each officer, at each level
    for index, row in hsi_events.iterrows():
        appt_foot = row['Appointment footprint']
        facility_level = row['Facility level']
        if ',' in appt_foot:
            appt_foot_list = appt_foot.split(',')
            for appt_foot_i in appt_foot_list:
                appt_foot_i = appt_foot_i.strip()
                new_row = {"Treatment_ID": row['Treatment'], "Appt_Footprint": appt_foot_i, "Level": facility_level}
                df_at = appt_time[(appt_time['Appt_Type_Code'] == appt_foot_i) & (appt_time['Facility_Level'] == facility_level)]
                for index2,row2 in df_at.iterrows():
                    new_row[row2['Officer_Category']] = row2['Time_Taken_Mins']
                Table = pd.concat([Table, pd.DataFrame([new_row])], ignore_index=True)
        else:
            new_row = {"Treatment_ID": row['Treatment'], "Appt_Footprint": appt_foot, "Level": facility_level}
            df_at = appt_time[(appt_time['Appt_Type_Code'] == appt_foot) & (appt_time['Facility_Level'] == facility_level)]
            for index2,row2 in df_at.iterrows():
                new_row[row2['Officer_Category']] = row2['Time_Taken_Mins']
            Table = pd.concat([Table, pd.DataFrame([new_row])], ignore_index=True)

    Table = Table.fillna(0)

    new_index = 'CMDs_Prevent._CommunityTestForHypert.'
    old_index = 'CardioMetabolicDisorders_Prevention_CommunityTestingForHypertension'
    
    Table['Treatment_ID_and_Level'] = Table['Treatment_ID'] + "_Level_" + Table['Level']
    pd.set_option('display.max_columns', None)

    # Use this if want to see treatments delivered at multiple levels
    duplicate_names = Table[Table.duplicated(subset='Treatment_ID', keep=False)]

    levels = ("0", "1a", "1b", "2", "3")    
    
    dict_competing_hsis = {}
    for do_level in levels:
        df_main = Table.loc[Table["Level"]==do_level]
        for officer in officer_types: 
            df_ot = df_main.sort_values(by=[officer],ascending=False)
            df_ot = df_ot.drop(df_ot[df_ot[officer] == 0.].index)
            s = (df_ot["Treatment_ID_and_Level"].drop_duplicates()).sort_values()
            myset = set(s) 
            if len(myset) > 0:
                dict_competing_hsis["Level_" + do_level + "_" +  officer] = SortTuple(tuple(myset))

    officers_with_levels = dict_competing_hsis.keys()
    treatments_with_levels = Table['Treatment_ID_and_Level'].drop_duplicates()
    
    # Make nodes for both officers and treatments
    HSIs_and_Officers_Network = nx.Graph()
    for officer in officers_with_levels:
        HSIs_and_Officers_Network.add_node(officer, type='Officer')
    for treatment in treatments_with_levels:
        HSIs_and_Officers_Network.add_node(treatment, type='Treatment')

    # Add edges connecting treatments and officers
    for officer in dict_competing_hsis.keys():
        for treatment in dict_competing_hsis[officer]:
            HSIs_and_Officers_Network.add_edge(officer, treatment)
    
    colours_nodes = {'Pharmacy' : '#ffa500', 'Clinical' : '#CC99FF', 'Nursing_and_Midwifery': '#99CCFF','DCSA' : '#66B2FF'}
   
    def make_plot_of_competing_HSIs(origin_node, level):
  
        try:
            # Create a copy of the original network but for
            # this level only, and clear labels such that
            # level is not included in treatment ID
            
            G = nx.Graph()
            for node in HSIs_and_Officers_Network.nodes:
                if level in node:
                    G.add_node(node)
                    
            for edge in HSIs_and_Officers_Network.edges:
                if any(level in node for node in edge):
                    G.add_edge(*edge)
                    
            # Remove 'level from the name of every node
            updated_nodes = {node: node.replace('_' + level, '') for node in G.nodes}

            # Update node names in the graph
            nx.relabel_nodes(G, updated_nodes, copy=False)
            
            # Remove 'this' from the name of every node
            updated_nodes = {node: node.replace(level + '_', '') for node in G.nodes}

            # Update node names in the graph
            nx.relabel_nodes(G, updated_nodes, copy=False)

            # Step 1: Get neighbors of the origin node
            officers_accessed_by_origin_node = list(G.neighbors(origin_node))

            # Step 2: Find nodes connected to neighbors of the origin node
            competing_HSIs = set()
            for neighbor in officers_accessed_by_origin_node:
                competing_HSIs.update(G.neighbors(neighbor))
                        
            subgraph_nodes = list(officers_accessed_by_origin_node) + list(competing_HSIs)
            subgraph = G.subgraph(subgraph_nodes)

            fig, ax = plt.subplots(figsize=(8, 8))
         
            # Draw the subgraph with custom styling
            pos = nx.fruchterman_reingold_layout(subgraph)
            
            nx.draw(subgraph, pos, with_labels=True, node_color='skyblue', node_size=100, font_size=13, edge_color='gray', linewidths=1, font_color='black', edgecolors='none')

            # Highlight the origin node in class A
            nx.draw_networkx_nodes(subgraph, pos, nodelist=[origin_node], node_color='red', node_size=100)

            for officer in list(officers_accessed_by_origin_node):
                nx.draw_networkx_nodes(subgraph, pos, nodelist=[officer], node_color=colours_nodes[officer], node_size=200, edgecolors='none')
                neighbors = list(G.neighbors(officer))
                # Highlight edges connecting origin node neighbors to class B
                highlighted_edges = [(officer, neighbor) for neighbor in neighbors]
                # Highlight edges connected to the origin node in red
                nx.draw_networkx_edges(subgraph, pos, edgelist=highlighted_edges, edge_color=colours_nodes[officer], width=2)

            # Show the plot
            plt.margins(x=0.9, y=0.9)

            plt.title(level, fontsize=18)
            plt.tight_layout()
            plt.savefig('plots/diagram_competition_' + origin_node + '_' + level + '.png', bbox_inches='tight', dpi=500)
            
        except:
            print("Error with HSI at this level")

       
    def get_competing_HSIs(origin_node, level):
        
        try:
            G = HSIs_and_Officers_Network
            origin_node = origin_node + "_" + level
            # Step 1: Get neighbors of the origin node in class A
            officers_accessed_by_origin_node = list(G.neighbors(origin_node))
            # Step 2: Find nodes in class B connected to neighbors of the origin node
            competing_HSIs = set()
            for neighbor in officers_accessed_by_origin_node:
                competing_HSIs.update(G.neighbors(neighbor))
            return list(competing_HSIs)
        except:
            print("Error with HSI at this level")
    
    # Index is too long, replace
    new_index = 'CMDs_Prevent._CommunityTestForHypert.'
    old_index = 'CardioMetabolicDisorders_Prevention_CommunityTestingForHypertension'
                      
    colours_keys = {
                  'No Policy':'#1f77b4',
                  'Naive':'#1f77b4',
                  'RMNCH':'#e377c2',
                  'Clinically Vulnerable':'#9467bd',
                  'Vertical Programmes':'#2ca02c',
                  'CVD':'#8c564b',
                  'HSSP-III EHP':'#d62728',
                  'LCOA EHP':'#ff7f0e'}


    for policy in df.columns.get_level_values('draw').unique():
        if 'ideal' in policy:
            df = df.drop([policy], axis=1)
    for policy in df1.columns.get_level_values('draw').unique():
        if 'ideal' in policy:
            df1 = df1.drop([policy], axis=1)

    target_treatment = "Malaria_Test"

    for what in ['delivered','requests']:

        for level in ["Level_0"]:
            if level == "Level_0":
                if what == 'requests':
                    fig, ax = plt.subplots(figsize=(6, 6))
                else:
                    fig, ax = plt.subplots(figsize=(6, 3))
            else:
                if what == 'requests':
                    fig, ax = plt.subplots(figsize=(20, 6))
                else:
                    fig, ax = plt.subplots(figsize=(20, 3))
        
            comp_HSIs = get_competing_HSIs(target_treatment, level)

            comp_HSIs_no_level = comp_HSIs
            condition = df.index.isin(comp_HSIs_no_level)

            condition_del = df1.index.isin(comp_HSIs_no_level)
            total_request = df.loc[condition]
            total_delivered = df1.loc[condition_del]

            fraction_request = total_request
            fraction_delivered = total_delivered
            
            fraction_delivered = total_delivered/total_request
            fraction_request = total_request

            reorder = fraction_request['No Policy status quo'].sort_values(by='mean', ascending=False)
            index_order = reorder.index
      
            fraction_request = fraction_request.reindex(index_order)
            fraction_delivered = fraction_delivered.reindex(index_order)

            # Create the bar plot with bars next to each other
            bar_width = 0.1
            index = np.arange(len(fraction_request.index))

            if what == 'requests':
                df_use = fraction_request
            else:
                df_use = fraction_delivered
                
                
            df_use.index = df_use.index.astype(str).str.replace("_"+level, '')
            df_use = df_use.rename(index={old_index:new_index})

            for i, column in enumerate(df_use.columns.get_level_values('draw').unique()):
                if 'status quo' in column:
                    column_label = column.replace(' status quo','')
                    column_label_legend = column_label
                    if column_label == "No Policy":
                        column_label_legend = "NP"
                    if column_label == "Vertical Programmes":
                        column_label_legend = "VP"
                    if column_label == "Clinically Vulnerable":
                        column_label_legend = "CV"
                    if column_label == "CVD":
                        column_label_legend = "CMD"
                    if column_label == "LCOA EHP":
                        column_label_legend = "LCOA"
                    ax.bar(index + i * bar_width, df_use[column, 'mean'], width=bar_width, label=column_label_legend,color=colours_keys[column_label])
                    if what == 'requests':
                        plt.legend(loc='upper right')
            # Set x-axis labels to be the DataFrame index
            ax.set_xticks(index + ((len(df_use.columns.get_level_values('draw').unique()) - 1) * bar_width) / 2)

            if what == 'requests':
                plt.yscale('log')

            # Add labels and a legend
            if what == 'requests':
                plt.ylabel('Number of treatments requested')
                ax.set_xticklabels(df_use.index, rotation=90)
            else:
                plt.ylabel('Fraction of treatments delivered')
                ax.set_xticklabels('')

            plt.tight_layout()
            plt.title(level)
            plt.grid(axis='y', linestyle='-', alpha=0.7)
            plt.tight_layout()
            if what == 'requests':
                plt.savefig('plots/' + target_treatment + "_" + level + "_fraction_of_appts_requested.png", dpi=500)
            else:
                plt.savefig('plots/' + target_treatment + "_" + level + "_fraction_of_appts_delivered.png", dpi=500)
                
    idx = pd.IndexSlice

    # Only keep mean values for stacked bars
    def make_summary_for_area(df1_short, what):
        df1_short_skim = df1_short.drop('FirstAttendance*')


        for policy in df1_short_skim.columns.get_level_values('draw').unique():
            if "ideal case" in policy:
                del df1_short_skim[policy]
                
        df1_short_skim = df1_short_skim.sort_values(by=('No Policy status quo','mean'), ascending=False)

        mean_data = df1_short_skim.loc[idx[:], idx[:,'mean']]
        mean_data = mean_data.droplevel(level=1, axis=1)

        mean_data.columns = [col.replace(' status quo', '') for col in mean_data.columns]
        mean_data = mean_data/1e6

        ax = mean_data.T.plot(kind='bar', stacked=True, color=color_palette, figsize=(16, 8))
        plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
        plt.xlabel('')
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.grid(axis='y', which='both')

        plt.ylabel('Total treatments ' +  what + ' (millions)', fontsize=14)
        plt.tight_layout()
        plt.savefig('plots/stacked_HSIs_' + what + '_by_area.png',dpi=500)
        plt.close()
        df = mean_data

        for index, row in df.iterrows():
            plt.figure()
            plt.figure(figsize=(4,6))
            index_label = index.replace('*', '')
            row.plot(kind='bar', title=f'{index_label}', color=[policy_colours[col] for col in df.columns], width=0.9)
            plt.ylabel('Total treatments ' + what + ' (millions)')
            plt.tight_layout()
            plt.savefig('plots/Total_HSIs_' + what + '_for_' + index_label + '.png')
            plt.close()
            
        mean_data = mean_data.drop(mean_data.index[:7])
        ax = mean_data.T.plot(kind='bar', stacked=True, figsize=(16, 8))
        plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
        plt.xlabel('')
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.grid(axis='y', which='both')

        plt.ylabel('Total treatments delivered (millions)', fontsize=14)
        plt.tight_layout()
        plt.savefig('plots/stacked_HSIs_delivered_by_area_focus.png',dpi=500)
        
    make_summary_for_area(df1_short, 'delivered')
    make_summary_for_area(df_short, 'requested')


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
