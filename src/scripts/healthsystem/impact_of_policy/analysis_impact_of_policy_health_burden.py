"""Produce plots to show the health impact when running under different policies (scenario_impact_of_policy.py)"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from tlo import Date
from tlo.analysis.utils import extract_results, summarize

# Range of years considered
min_year = 2023
max_year = 2042

hs_dict = {"short":"No Persistence",
           "long": "10-days Persistence"
}

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
                  
color_palette = {
    'AIDS':  '#1F449C',
    'Lower respiratory infections':'#F05039',
    'Neonatal Disorders': '#E57A77',
    'Depression / Self-harm': '#3D65A5',
    'Malaria': '#EEBAB4',
    'Transport Injuries': '#A8B6CC',
    'TB (non-AIDS)': '#FFD700',  # Gold
    'Measles': '#FF69B4',  # Pink
    'Childhood Diarrhoea': '#00FF7F',  # Spring Green
    'Maternal Disorders': '#9370DB',  # Medium Purple
    'Cancer (Other)': '#FF6347',  # Tomato
    'Congenital birth defects': '#1E90FF',  # Dodger Blue
    'Schistosomiasis': '#ADFF2F',  # Green Yellow
    'Heart Disease': '#8A2BE2',  # Blue Violet
    'Kidney Disease': '#20B2AA',  # Light Sea Green
    'Diabetes': '#FF8C00',  # Dark Orange
    'Stroke': '#7B68EE',  # Medium Slate Blue
    'Cancer (Bladder)': '#8B0000',  # Dark Red
    'Cancer (Breast)': '#FFC0CB',  # Pink
    'Cancer (Oesophagus)': '#9932CC',  # Dark Orchid
    'Cancer (Prostate)': '#FFD700',  # Gold
    'Epilepsy': '#00FFFF',  # Cyan
    'COPD': '#008080',  # Teal
    'Lower Back Pain': '#F08080',  # Light Coral
    'Other': '#D3D3D3',  # Light Gray
    }
    
def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None, ):
    """Produce standard set of plots describing the effect of each TREATMENT_ID.
    - We estimate the epidemiological impact as the EXTRA deaths that would occur if that treatment did not occur.
    - We estimate the draw on healthcare system resources as the FEWER appointments when that treatment does not occur.
    """

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

    def do_bar_plot_with_ci(_df, annotations=None):
        """Make a vertical bar plot for each row of _df, using the columns to identify the height of the bar and the
         extent of the error bar."""
        yerr = np.array([
            (_df['mean'] - _df['lower']).values,
            (_df['upper'] - _df['mean']).values,
        ])

        colours = ('#1f77b4','#d62728','#ff7f0e','#e377c2','#2ca02c','#9467bd','#8c564b')

        xticks = {(i + 0.5): k for i, k in enumerate(_df.index)}

        fig, ax = plt.subplots()

        ax.bar(
            xticks.keys(),
            _df['mean'].values,
            yerr=yerr,
            ecolor='black',
            capsize=10,
            color=colours # Need to fix this to adopt dictionary
        )

        if annotations:
            for xpos, ypos, text in zip(xticks.keys(), _df['mean'].values, annotations):
                ax.text(xpos, ypos, text, horizontalalignment='center')
        ax.set_xticks(list(xticks.keys()))
        ax.set_xticklabels(list(xticks.values()), rotation=90)
        ax.grid(axis="y", linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()

        return fig, ax
        
        
    def clean_dataframe(df):

        del df["No Healthcare System"]
        df = df.rename(columns={'Naive status quo': 'NP status quo'})
        df = df.rename(columns={'EHP III status quo': 'HSSP-III HBP status quo'})
        df = df.rename(columns={'LCOA EHP status quo': 'LCOA status quo'})
        df = df.rename(columns={'Vertical Programmes status quo': 'VP status quo'})
        df = df.rename(columns={'Clinically Vulnerable status quo': 'CV status quo'})
        df = df.rename(columns={'CVD status quo': 'CMD status quo'})
        
        new_order = ['NP status quo', 'HSSP-III HBP status quo', 'LCOA status quo', 'VP status quo', 'RMNCH status quo', 'CV status quo', 'CMD status quo']
      
        # Reorder policies
        df = df[new_order]

        return df
                    
    def clean_df_index(df, status_quo):
    
        # Remove policies we're not considering
        if status_quo:
            for policy in df.index:
                if "ideal case" in policy:
                    df = df.drop(index=policy)
        else:
            for policy in df.index:
                if "status quo" in policy:
                    df = df.drop(index=policy)

        if status_quo:
            substring_to_remove = ' status quo'
        else:
            substring_to_remove = ' ideal case'
            
        df.index = [index.replace(substring_to_remove, '') if substring_to_remove in index else index for index in df.index]

        # Rename Naive as No Policy
        df = df.rename(index={'Naive': 'NP'})
        df = df.rename(index={'EHP III': 'HSSP-III HBP'})
        df = df.rename(index={'LCOA EHP': 'LCOA'})
        df = df.rename(index={'CVD': 'CMD'})
        df = df.rename(index={'Clinically Vulnerable': 'CV'})
        df = df.rename(index={'Vertical Programmes': 'VP'})
                
        # Reorder policies
        df = df.reindex(['No Healthcare System',
                         'NP',
                         'HSSP-III HBP', 'LCOA',
                         'RMNCH', 'VP', 'CV', 'CMD']) #, level='draw')
                        
        # Not including No Health System case
        df = df.drop(index="No Healthcare System")
        
        return df
        
        
    def clean_df_columns(df):

        del df['No Healthcare System']
        df = df.rename(columns={'Naive status quo': 'NP status quo'})
        df = df.rename(columns={'EHP III status quo': 'HSSP-III HBP status quo'})
        df = df.rename(columns={'LCOA EHP status quo': 'LCOA status quo'})
        df = df.rename(columns={'Vertical Programmes status quo': 'VP status quo'})
        df = df.rename(columns={'Clinically Vulnerable status quo': 'CV status quo'})
        df = df.rename(columns={'CVD status quo': 'CMD status quo'})

        new_order = ['NP status quo', 'HSSP-III HBP status quo', 'LCOA status quo', 'VP status quo', 'RMNCH status quo', 'CV status quo', 'CMD status quo']
  
        # Reorder policies
        df = df[new_order]

        idx = pd.IndexSlice

        for policy in df.columns.get_level_values('draw').unique():
            if "ideal case" in policy:
                del df[policy]
                
        df = df.rename(columns={'Naive status quo': 'No Policy status quo'})

        # Rank by mean value of No policy status quo
        df_sorted = df.sort_values(by=('NP status quo','mean'), ascending=False)
        
        # Move 'Other' to the end of stacked DALYs
        row_to_move = df_sorted.loc[('Other')]
        df_sorted = pd.concat([df_sorted.drop(('Other')), row_to_move.to_frame().transpose()])

        # Remove the substring 'this' from all column names
        df_sorted = df_sorted.rename(columns=lambda x: x.replace(' status quo', ''))
        
        return df_sorted
        
    def clean_df_for_stacked_DALYs(df):

        del df['No Healthcare System']
        df = df.rename(columns={'Naive status quo': 'NP status quo'})
        df = df.rename(columns={'EHP III status quo': 'HSSP-III HBP status quo'})
        df = df.rename(columns={'LCOA EHP status quo': 'LCOA status quo'})
        df = df.rename(columns={'Vertical Programmes status quo': 'VP status quo'})
        df = df.rename(columns={'Clinically Vulnerable status quo': 'CV status quo'})
        df = df.rename(columns={'CVD status quo': 'CMD status quo'})

        new_order = ['NP status quo', 'HSSP-III HBP status quo', 'LCOA status quo', 'VP status quo', 'RMNCH status quo', 'CV status quo', 'CMD status quo']
  
        # Reorder policies
        df = df[new_order]

        idx = pd.IndexSlice

        for policy in df.columns.get_level_values('draw').unique():
            if "ideal case" in policy:
                del df[policy]
                
        df = df.rename(columns={'Naive status quo': 'No Policy status quo'})

        df_sorted = df.sort_values(by=('NP status quo','mean'), ascending=False)
        
        # Move 'Other' cause to bottom of stacked DALYs
        row_to_move = df_sorted.loc[('Other')]
        df_sorted = pd.concat([df_sorted.drop(('Other')), row_to_move.to_frame().transpose()])
        
        # Only keep mean values for stacked bars
        mean_data = df_sorted.loc[idx[:], idx[:,'mean']]
        mean_data = mean_data.droplevel(level=1, axis=1)

        mean_data.columns = [col.replace(' status quo', '') for col in mean_data.columns]
        print(mean_data)
        return mean_data/1e6
        
    # Obtain parameter names for this scenario file
    param_names = get_parameter_names_from_scenario_file()
    print('Param names are', param_names)

    # TOTAL DALYs
    # %% Quantify the health losses associated with all interventions combined.
    num_dalys = extract_results(
        results_folder,
        module='tlo.methods.healthburden',
        key='dalys_stacked',
        custom_generate_series=get_num_dalys,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0)

    num_dalys_raw = num_dalys.copy()

    # Obtain statistical summary
    num_dalys_summarized = summarize(num_dalys).loc[0].unstack()

    # Chose which class of scenarios we want to consider, and clean accordingly
    status_quo = True
    num_dalys_summarized = clean_df_index(num_dalys_summarized, status_quo)
    print(num_dalys_summarized)

    # Bar plots for DALYs incurred under each HealthCare Configuration Scenario
    # TOTAL DALYs
    name_of_plot = f'Total DALYs incurred under each policy, {target_period()}'

    fig, ax = do_bar_plot_with_ci(num_dalys_summarized / 1e6)
    
    box_x_range = (0, 7)
    box_y_range = (246, 283)

    # Create a transparent box with red line edges
    box = Rectangle((box_x_range[0], box_y_range[0]),
                    box_x_range[1] - box_x_range[0], box_y_range[1] - box_y_range[0],
                    edgecolor='red', facecolor='none', linewidth=2, alpha=0.5)

    # Add the box to the main plot
    ax.add_patch(box)
    zoomed = False
    if zoomed == False:
        ax.set_title(name_of_plot)
    ax.set_ylabel('DALYs (Millions)')
    colours = ('#1f77b4','#d62728','#ff7f0e','#e377c2','#2ca02c','#9467bd','#8c564b')

    # Extend range for No policy (NP) policy
    ax.fill_between(np.arange(len(num_dalys_summarized)+1), num_dalys_summarized.loc['NP','lower']/1e6, num_dalys_summarized.loc['NP','upper']/1e6, color=policy_colours["NP"], alpha=0.3, linewidth=0, zorder=0)
    fig.tight_layout()
    if max_year == 2037:
        plt.ylim(0,200)
    elif max_year == 2027:
        plt.ylim(0,70)
    elif max_year == 2025:
        plt.ylim(0,40)
    elif max_year == 2042:
        if zoomed == True:
            plt.ylim(245,285)
        else:
            plt.ylim(0,300)
    
    if zoomed == False:
        fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')), dpi=400)
    else:
        fig.savefig('plots/Total_DALYS_incurred_under_each_policy_zoomed_2023-2042.png', dpi=400)
    fig.show()
    plt.close(fig)
        
    # DALYs AVERTED
    # Plot DALYs averted compared to the ``No Policy'' policy
    if status_quo:
        comparison='Naive status quo'
    else:
        comparison='Naive ideal case'

    num_dalys_averted = summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison(
                num_dalys.loc[0],
                comparison=comparison)
        ).T
    ).iloc[0].unstack()

    num_dalys_averted_raw = -1.0 *pd.DataFrame(
            find_difference_relative_to_comparison(
                num_dalys.loc[0],
                comparison=comparison)
        ).T

    comp_policy = "LCOA EHP"
 
    diff_comp_policy_NP = num_dalys_raw[comp_policy +' status quo'] - num_dalys_raw['Naive status quo']

    num_dalys_averted = clean_df_index(num_dalys_averted, status_quo)
    num_dalys_averted = num_dalys_averted.drop(index = "NP")

    colours = ('#1f77b4','#d62728','#ff7f0e','#e377c2','#2ca02c','#9467bd','#8c564b')

    name_of_plot = f'Additional DALYs Averted vs No Policy, {target_period()}'
    fig, ax = do_bar_plot_with_ci(
        (num_dalys_averted / 1e6)
    )
    ax.set_title(name_of_plot)
    ax.set_ylabel('Additional DALYs Averted (Millions)')
    ax.set_ylim(-7.5,20)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)
    
    
    # TIME EVOLUTION OF TOTAL DALYs
    
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
    for year in range(min_year-1, max_year+1):
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
    
    df_total = dalys_by_year
    
    df_total = summarize(df_total)
    df_total = clean_dataframe(df_total)
    unique_draws = df_total.columns.get_level_values('draw').unique()

    plt.figure(figsize=(6, 8))
    for hs in ["status quo"]:
        fig, ax = plt.subplots()
        for draw in unique_draws:
            if hs in draw:
                draw_data = df_total[draw]/1e6
                x_values = draw_data.index.get_level_values('date')
                y_values = draw_data['mean']

                # Plot the data
                lower_bounds = draw_data['lower']
                upper_bounds = draw_data['upper']
                draw_label = draw.replace(" " + hs, "")
                ax.plot(x_values, y_values, label=f'{draw_label}', color = policy_colours[draw_label], linewidth=2)
                ax.fill_between(x_values, lower_bounds, upper_bounds, alpha=0.2, color = policy_colours[draw_label])

    ax.set_ylim(6,18)
    start_year = 2022
    end_year = 2042
    tick_interval = 2
    xtick_positions = np.arange(start_year, end_year+1, tick_interval)
    xtick_labels = [str(year) for year in xtick_positions]            # plt.legend()
    plt.xticks(xtick_positions, xtick_labels, rotation=90)  # Rotate labels for better visibility
    plt.xlim(min_year,end_year)
    plt.xlabel("Year")
    plt.ylabel("DALYs (millions) per year")
    ax.legend(loc='lower right')

    plt.grid(True)
    name_fig = 'plots/Yearly_DALYs_' + str(min_year) + '-' + str(max_year) + '.png'
    name_fig = name_fig.replace(" ", "_")
    plt.tight_layout()
    plt.savefig(name_fig, dpi=400)
    plt.close()
    
    # DALYs BROKEN DOWN BY CAUSES
    # TOTAL DALYs by cause
    num_dalys_by_cause = extract_results(
        results_folder,
        module='tlo.methods.healthburden',
        key='dalys_stacked',
        custom_generate_series=get_num_dalys_by_cause,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0)
    
    df_num_dalys_by_cause = num_dalys_by_cause
    df = summarize(df_num_dalys_by_cause)
    
    df = clean_df_columns(df)/1e6
    

    df.index.name = 'cause'
    # Iterate over unique 'cause' values and create a bar plot for each
    for cause in df.index.get_level_values('cause').unique():
        cause_label = cause.replace('/', '_or_')
        cause_label = cause_label.replace(' ', '_')
        plt.figure(figsize=(4, 6))
        cause_df = df.loc[df.index==cause]
        bars = cause_df.columns.get_level_values('draw').unique()
        means = cause_df.xs('mean', level='stat', axis=1).values.flatten()
        lower_bounds = cause_df.xs('lower', level='stat', axis=1).values.flatten()
        upper_bounds = cause_df.xs('upper', level='stat', axis=1).values.flatten()

        # Plotting with error bars and color mapping
        for bar, mean, lower, upper in zip(bars, means, lower_bounds, upper_bounds):
            color = policy_colours.get(bar, 'black')  # Default to black if draw name not found in the mapping
            plt.bar(bar, mean, label=cause, color=color, width=0.8)
            plt.errorbar(bar, mean, yerr=[[mean - lower], [upper - mean]], fmt='none', capsize=5, color='black')
        plt.ylabel('DALYs (million)')
        plt.title(cause + ' (' + str(min_year) + ' - ' + str(max_year) + ')')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('plots/DALYs_compare_policies_for_' + cause_label + '.png', dpi=400)
        plt.close()

    num_dalys_by_cause = summarize(num_dalys_by_cause)
    mean_data = clean_df_for_stacked_DALYs(num_dalys_by_cause)
    
    ax = mean_data.T.plot(kind='bar', stacked=True, color=color_palette, figsize=(16, 8))
    # Customize the plot (optional)
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
    plt.xlabel('')
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(axis='y', which='both')
    ax.set_ylim(0,290)

    plt.ylabel('DALYs (millions)', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/stacked_DALYs_by_cause_' + str(min_year) + '-' + str(max_year) + '.png',dpi=500)
    plt.close()
    
    for index, row in mean_data.iterrows():
        index_label = index.replace('/', '_or_')
        index_label = index_label.replace(' ', '_')
        
        row.plot(kind='bar', title=f'{index_label}', color=[policy_colours[col] for col in mean_data.columns], width=0.9)
        plt.ylabel('DALYs (millions)')
        plt.ylim(0,50)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig('plots/Total_DALYs_for_' + index_label + '.png')
        plt.close()
    

    # DALYs BROKEN DOWN BY CAUSES AND YEAR
    # DALYs by cause per year
    
    def clean_dataframe_year_and_cause(df):

        del df["No Healthcare System"]
        df = df.rename(columns={'Naive status quo': 'NP status quo'})
        df = df.rename(columns={'EHP III status quo': 'HSSP-III HBP status quo'})
        df = df.rename(columns={'LCOA EHP status quo': 'LCOA status quo'})
        df = df.rename(columns={'Vertical Programmes status quo': 'VP status quo'})
        df = df.rename(columns={'Clinically Vulnerable status quo': 'CV status quo'})
        df = df.rename(columns={'CVD status quo': 'CMD status quo'})

        new_order = ['NP status quo', 'HSSP-III HBP status quo', 'LCOA status quo', 'VP status quo', 'RMNCH status quo', 'CV status quo', 'CMD status quo']

        df = df[new_order]
        
        return df
    
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

    for year in range(min_year, max_year+1):
        year_target = year
        num_dalys_by_year = extract_results(
            results_folder,
            module='tlo.methods.healthburden',
            key='dalys_stacked',
            custom_generate_series=get_num_dalys_by_year_and_cause,
            do_scaling=True
        ).pipe(set_param_names_as_column_index_level_0)
        ALL[year_target] = summarize(num_dalys_by_year)

    # Concatenate the DataFrames into a single DataFrame
    concatenated_df = pd.concat(ALL.values(), keys=ALL.keys())
    print(concatenated_df)
    concatenated_df.index = concatenated_df.index.set_names(['date', 'cause'])
    
    df_total = clean_dataframe_year_and_cause(concatenated_df)/1e6

    leading_causes = ['Maternal Disorders', 'Neonatal Disorders',
       'Lower respiratory infections', 'Childhood Diarrhoea', 'AIDS',
       'Malaria', 'Measles', 'TB (non-AIDS)', 'Depression / Self-harm',
       'Transport Injuries']
    
    plt.figure(figsize=(10, 6))

    for cause in leading_causes:
        cause_df = df_total.xs(cause, level='cause')
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

            plt.title(f'{cause}')
            plt.xlabel("Year")
            plt.legend()
            plt.ylabel("DALYs (millions) per year")
            start_year = 2022
            end_year = 2042
            tick_interval = 2

            xtick_positions = np.arange(start_year, end_year+1, tick_interval)
            xtick_labels = [str(year) for year in xtick_positions]            # plt.legend()
            plt.xticks(xtick_positions, xtick_labels)  # Rotate labels for better visibility
            plt.xlim(min_year,end_year)
            plt.ylim(bottom=0)
            plt.grid(True)
            cause_label = cause.replace('/', '_or_')
            name_fig = 'plots/DALYs_'+cause_label+ "_" + str(min_year) + '-' + str(max_year) + '.png'
            name_fig = name_fig.replace(" ", "_")
            plt.savefig(name_fig, dpi=300)
            plt.close()
    
    unique_causes = df_total.index.get_level_values('cause').unique()
    
    # Rank by mean value of No policy status quo
    df_total = df_total[df_total.index.get_level_values('cause') != 'Other']

    df_cumulative = df_total.groupby(level='cause').sum()
    df_sum_over_causes = df_total.groupby(level='date').sum()
    df_cumulative = df_cumulative.sort_values(by=('NP status quo','mean'), ascending=False)
    first_10_causes = df_cumulative.index[:10]
    df_10_causes = df_total[df_total.index.get_level_values('cause').isin(leading_causes)]
    df_10_causes = df_10_causes.groupby(level='date').sum()
    
    color_palette_causes = {
    'Maternal Disorders': '#1F77B4',       # Blue
    'Neonatal Disorders': '#FF7F0E',       # Orange
    'Congenital birth defects': '#2CA02C', # Green
    'Lower respiratory infections': '#D62728', # Red
    'Childhood Diarrhoea': '#9467BD',    # Purple
    'AIDS': '#8C564B',                   # Brown
    'Malaria': '#E377C2',                # Pink
    'Measles': '#7F7F7F',                # Gray
    'TB (non-AIDS)': '#BCBD22',          # Olive
    'Depression / Self-harm': '#17BECF', # Cyan
    'Transport Injuries': '#1FB4B5',     # Teal
    'Cancer (Other)': '#7BA05B'         # Olive Green
    }

    plt.figure(figsize=(25, 6))
    for policy in unique_draws:
        if  "status quo" in policy:
            draw = policy
            fig, ax = plt.subplots()
            
            draw_data = df_sum_over_causes[draw]
            x_values = draw_data.index.get_level_values('date')
            y_values = draw_data['mean']
            lower_bounds = draw_data['lower']
            upper_bounds = draw_data['upper']
            
            ax.plot(x_values, y_values, label="Total", color="black")
            ax.fill_between(x_values, lower_bounds, upper_bounds, alpha=0.2, color="black")
            
            draw_data = df_10_causes[draw]
            x_values = draw_data.index.get_level_values('date')
            y_values = draw_data['mean']
            lower_bounds = draw_data['lower']
            upper_bounds = draw_data['upper']
            
            ax.plot(x_values, y_values, label="Top 10 causes", color="lightskyblue")
            ax.fill_between(x_values, lower_bounds, upper_bounds, alpha=0.4, color="lightskyblue")
            
            for cause in first_10_causes:
                cause_df = df_total.xs(cause, level='cause')
                draw_data = cause_df[draw]
                x_values = draw_data.index.get_level_values('date')
                y_values = draw_data['mean']
                lower_bounds = draw_data['lower']
                upper_bounds = draw_data['upper']
        
                ax.plot(x_values, y_values, label=cause, color=color_palette_causes[cause])
                ax.fill_between(x_values, lower_bounds, upper_bounds, alpha=0.2, color=color_palette[cause])

            ax.set_yscale('log')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax.set_ylim(1e-1,20)
            ax.set_xlim(2023,2042)
            ax.grid(axis='y', which='both')
            plt.xlabel("Year")
            plt.ylabel("DALYs (millions) per year")
            plt.grid(True)
            name_fig = 'plots/Compare_all_causes_policy_' + str(policy) + str(min_year) + '-' + str(max_year) + '.png'
            name_fig = name_fig.replace(" ", "_")
            plt.tight_layout()
            plt.savefig(name_fig,dpi=300)
            plt.close()
            
            
    #================================================================================
    # DALYs normalised by population size each year
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
    # Concatenate the DataFrames into a single DataFrame
    concatenated_df = pd.concat(ALL.values(), keys=ALL.keys())
    concatenated_df.index = concatenated_df.index.set_names(['date', 'index_original'])
    concatenated_df = concatenated_df.reset_index(level='index_original',drop=True)
    dalys_by_year = concatenated_df
    
    # Load and format model results (with year as integer):
    pop_model = extract_results(results_folder,
                                module="tlo.methods.demography",
                                key="population",
                                column="total",
                                index="date",
                                do_scaling=True
                                ).pipe(set_param_names_as_column_index_level_0)
    
    pop_model.index = pop_model.index.year
    pop_model = pop_model[(pop_model.index >= min_year) & (pop_model.index <= max_year)]

    assert dalys_by_year.index.equals(pop_model.index)

    normalised_dalys = (dalys_by_year/pop_model)/len(dalys_by_year)

    total_normalised_dalys = pd.DataFrame(normalised_dalys.sum()).T
    
    num_dalys_summarized = summarize(total_normalised_dalys).loc[0].unstack()
    num_dalys_summarized = clean_df_index(num_dalys_summarized, status_quo)
    
    # Bar plots for DALYs incurred under each HealthCare Configuration Scenario
    # TOTAL DALYs
    name_of_plot = f'Average DALYs incurred per population size, {target_period()}'
    colours = ('#d62728','#ff7f0e','#e377c2','#2ca02c','#9467bd','#8c564b')

    fig, ax = do_bar_plot_with_ci(num_dalys_summarized)
    ax.set_title(name_of_plot)
    ax.set_ylabel('Average DALYs incurred/population size')
    ax.set_ylim(0,0.6)
    ax.fill_between(np.arange(len(num_dalys_summarized)+1), num_dalys_summarized.loc['NP','lower'], num_dalys_summarized.loc['NP','upper'], color=policy_colours["NP"], alpha=0.3, linewidth=0, zorder=0)
    fig.tight_layout()
        
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')), dpi=400)
    plt.close(fig)

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
