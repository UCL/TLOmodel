"""Produce plots to show the health impact (deaths, dalys) each the healthcare system (overall health impact) when
running under different MODES and POLICIES (scenario_impact_of_capabilities_expansion_scaling.py)"""

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
import warnings
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import datetime

from tlo import Date
from tlo.analysis.utils import extract_results, summarize
from tlo.analysis.life_expectancy import get_life_expectancy_estimates

index_HTM_plot = ['AIDS', 'TB (non-AIDS)', 'Malaria']
index_NeoChild_plot = ['Lower respiratory infections','Childhood Diarrhoea', 'Maternal Disorders', 'Measles', 'Neonatal Disorders', 'Schistosomiasis'] #'Congenital birth defects',
index_NCDs_plot = ['COPD', 'Cancer (Bladder)', 'Cancer (Breast)', 'Cancer (Oesophagus)', 'Cancer (Other)', 'Cancer (Prostate)','Depression / Self-harm', 'Diabetes', 'Epilepsy', 'Heart Disease', 'Kidney Disease', 'Stroke', 'Transport Injuries']

min_NCDs = 0
max_NCDs = 1.6
ymin_range = {
'AIDS' : 0.1,
'TB (non-AIDS)' : 0.1,
'Malaria' : 0.1,
'Lower respiratory infections' : 0.0,
'Childhood Diarrhoea': 0.0,
'Maternal Disorders': 0.0,
'Measles': 0.0,
'Neonatal Disorders': 0.0,
'Schistosomiasis': 0.0,
'COPD' : min_NCDs, 'Cancer (Bladder)' : min_NCDs, 'Cancer (Breast)' : min_NCDs, 'Cancer (Oesophagus)' : min_NCDs, 'Cancer (Other)' : min_NCDs, 'Cancer (Prostate)' : min_NCDs,'Depression / Self-harm'  : min_NCDs, 'Diabetes' : min_NCDs, 'Epilepsy' : min_NCDs, 'Heart Disease' : min_NCDs, 'Kidney Disease' : min_NCDs, 'Stroke' : min_NCDs, 'Transport Injuries' : min_NCDs
}
max_RMNCH = 3
ymax_range = {
'AIDS' : 1.5,
'TB (non-AIDS)' : 1.5,
'Malaria' : 1.5,
'Lower respiratory infections' : max_RMNCH,
'Childhood Diarrhoea': max_RMNCH,
'Maternal Disorders': max_RMNCH, 'Measles': max_RMNCH,
'Neonatal Disorders': max_RMNCH,
'Schistosomiasis': max_RMNCH,
'COPD' : max_NCDs, 'Cancer (Bladder)' : max_NCDs, 'Cancer (Breast)' : max_NCDs, 'Cancer (Oesophagus)' : max_NCDs, 'Cancer (Other)' : max_NCDs, 'Cancer (Prostate)' : max_NCDs,'Depression / Self-harm'  : max_NCDs, 'Diabetes' : max_NCDs, 'Epilepsy' : max_NCDs, 'Heart Disease' : max_NCDs, 'Kidney Disease' : max_NCDs, 'Stroke' : max_NCDs, 'Transport Injuries' : max_NCDs
}

# Range of years considered
min_year = 2018
max_year = 2040
inc_IHME_2024 = False
scenarios_properties = {
    0: {'sf_name': 'No growth status quo', 'g_GDP': 0, 'g_fHE': 0, 'k': '-', 't_c': '-', 'THE': 22.00, 'int_name': 'No growth', 'color': '#441F54'},
    1: {'sf_name': 'GDP growth status quo', 'g_GDP': 4.2, 'g_fHE': 0, 'k': '-', 't_c': '-', 'THE': 36.53, 'int_name': 'GDP growth', 'color': '#257D8F'},
    2: {'sf_name': 'GDP growth fHE growth case 1 status quo', 'g_GDP': 4.2, 'g_fHE': -3.0, 'k': '-', 't_c': '-', 'THE': 24.93, 'int_name': '<< GDP growth', 'color': '#482071'},
    3: {'sf_name': 'GDP growth fHE growth case 3 status quo', 'g_GDP': 4.2, 'g_fHE': -1.5, 'k': '-', 't_c': '-', 'THE': 30.08, 'int_name': '< GDP growth', 'color': '#3B518B'},
    4: {'sf_name': 'GDP growth fHE growth case 4 status quo', 'g_GDP': 4.2, 'g_fHE': 1.5, 'k': '-', 't_c': '-', 'THE': 44.60, 'int_name': '> GDP growth', 'color': '#3EBC73'},
    5: {'sf_name': 'GDP growth fHE growth case 6 status quo', 'g_GDP': 4.2, 'g_fHE': 3.0, 'k': '-', 't_c': '-', 'THE': 54.75, 'int_name': '>> GDP growth', 'color': '#FDE624'},
    6: {'sf_name': 'No growth perfect consumables', 'g_GDP': 0, 'g_fHE': 0, 'k': '-', 't_c': '-', 'THE': 22.00, 'int_name': 'No growth', 'color': '#441F54'},
    7: {'sf_name': 'GDP growth perfect consumables', 'g_GDP': 4.2, 'g_fHE': 0, 'k': '-', 't_c': '-', 'THE': 36.53, 'int_name': 'GDP growth', 'color': '#257D8F'},
    8: {'sf_name': 'GDP growth fHE growth case 1 perfect consumables', 'g_GDP': 4.2, 'g_fHE': -3.0, 'k': '-', 't_c': '-', 'THE': 24.93, 'int_name': '<< GDP growth', 'color': '#482071'},
    9: {'sf_name': 'GDP growth fHE growth case 3 perfect consumables', 'g_GDP': 4.2, 'g_fHE': -1.5, 'k': '-', 't_c': '-', 'THE': 30.08, 'int_name': '< GDP growth', 'color': '#3B518B'},
    10: {'sf_name': 'GDP growth fHE growth case 4 perfect consumables', 'g_GDP': 4.2, 'g_fHE': 1.5, 'k': '-', 't_c': '-', 'THE': 44.60, 'int_name': '> GDP growth', 'color': '#3EBC73'},
    11: {'sf_name': 'GDP growth fHE growth case 6 perfect consumables', 'g_GDP': 4.2, 'g_fHE': 3.0, 'k': '-', 't_c': '-', 'THE': 54.75, 'int_name': '>> GDP growth', 'color': '#FDE624'},
    12: {'sf_name': 'No growth perfect healthsystem', 'g_GDP': 0, 'g_fHE': 0, 'k': '-', 't_c': '-', 'THE': 22.00, 'int_name': 'No growth', 'color': '#441F54'},
    13: {'sf_name': 'GDP growth perfect healthsystem', 'g_GDP': 4.2, 'g_fHE': 0, 'k': '-', 't_c': '-', 'THE': 36.53, 'int_name': 'GDP growth', 'color': '#257D8F'},
    14: {'sf_name': 'GDP growth fHE growth case 1 perfect healthsystem', 'g_GDP': 4.2, 'g_fHE': -3.0, 'k': '-', 't_c': '-', 'THE': 24.93, 'int_name': '<< GDP growth', 'color': '#482071'},
    15: {'sf_name': 'GDP growth fHE growth case 3 perfect healthsystem', 'g_GDP': 4.2, 'g_fHE': -1.5, 'k': '-', 't_c': '-', 'THE': 30.08, 'int_name': '< GDP growth', 'color': '#3B518B'},
    16: {'sf_name': 'GDP growth fHE growth case 4 perfect healthsystem', 'g_GDP': 4.2, 'g_fHE': 1.5, 'k': '-', 't_c': '-', 'THE': 44.60, 'int_name': '> GDP growth', 'color': '#3EBC73'},
    17: {'sf_name': 'GDP growth fHE growth case 6 perfect healthsystem', 'g_GDP': 4.2, 'g_fHE': 3.0, 'k': '-', 't_c': '-', 'THE': 54.75, 'int_name': '>> GDP growth', 'color': '#FDE624'},
}

sets = {
    'first_set' : ['No growth', 'GDP growth fHE growth case 1', 'GDP growth fHE growth case 3', 'GDP growth', 'GDP growth fHE growth case 4', 'GDP growth fHE growth case 6'],
    'second_set' : ['GDP growth fHE growth case 6', 'GDP growth',  'GDP growth FL case 2 const i', 'GDP growth FL case 1 const i', 'GDP growth FL case 1 vary i','GDP growth FL case 2 vary i'],
    'third_set' : ['No growth rescaled', 'GDP growth fHE growth case 1 rescaled', 'GDP growth rescaled', 'GDP growth fHE growth case 6 rescaled']
    }

# Combine the dataframe with the draw values
sf_THE_dict = {entry['sf_name']: entry['THE'] for entry in scenarios_properties.values()}
sf_comb_rate= {entry['sf_name']: (((1+0.01*float(entry['g_GDP']))*(1+0.01*float(entry['g_fHE'])))-1)*100 for entry in scenarios_properties.values()}
sf_color_dict = {entry['sf_name']: entry['color'] for entry in scenarios_properties.values()}
sf_int_name_dict = {entry['sf_name']: entry['int_name'] for entry in scenarios_properties.values()}
sf_scenario_name_dict = {value['sf_name']: value['int_name'] for key, value in scenarios_properties.items()}
sf_scenario_numbers_dict = {value['sf_name']: value['THE'] for key, value in scenarios_properties.items()}

status_quo_scenarios = ['No growth status quo', 'GDP growth fHE growth case 1 status quo', 'GDP growth fHE growth case 3 status quo', 'GDP growth status quo', 'GDP growth fHE growth case 4 status quo',  'GDP growth fHE growth case 6 status quo']

PC_scenarios = ['No growth perfect consumables', 'GDP growth fHE growth case 1 perfect consumables', 'GDP growth fHE growth case 3 perfect consumables', 'GDP growth perfect consumables', 'GDP growth fHE growth case 4 perfect consumables',  'GDP growth fHE growth case 6 perfect consumables']

scenarios_map = {
  #  'Status quo' : {'scenarios' : status_quo_scenarios, 'colour' : '#CC0066', 'comparison' : 'GDP growth status quo', 'title': 'Present-day Cons. Avail.'},
    'Perfect Consumables' : {'scenarios' : PC_scenarios, 'colour' : '#0080FF', 'comparison' : 'GDP growth perfect consumables', 'title': 'Perfect Cons. Avail.'},
}

color_palette = {
    'AIDS':  '#d62728',
    'Lower respiratory infections':'#bcbd22',
    'Neonatal Disorders': '#9467bd',
    'Depression / Self-harm': '#3D65A5',
    'Malaria': 'orange',
    'Transport Injuries': '#FF00FF',
    'TB (non-AIDS)': '#1f77b4',  # Gold
    'Measles': '#FF69B4',  # Pink
    'Childhood Diarrhoea': '#17becf',  # Spring Green
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
    'Combined': 'black',
    'NCDs': 'blue'
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
        from scripts.healthsystem.impact_of_const_capabilities_expansion.scenario_impact_of_capabilities_expansion_scaling import (
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
        # Clinic vul, ehp iii, lcoa, naive, rmnch, random, vertical programmes

        # No Healthcare System, Naive, EHP III, LCOA EHP, RMNCH, Vertical Programmes, Clinic, CVD
        #colours = ("#808080", "#0051ff", "#FF0000", "#FF6666","#660066", "#009900", "#CC99FF", "#994C00")
        #colours = ("#0051ff", "#FF0000", "#FF6666","#660066", "#009900", "#CC99FF", "#994C00")
        #colours = ("#FF0000", "#FF6666","#660066", "#009900", "#CC99FF", "#994C00")
        #colours = ('#d62728', '#ff7f0e','#e377c2', '#2ca02c','#9467bd', '#8c564b')
        colours = ('#1f77b4','#d62728','#ff7f0e','#e377c2','#2ca02c','#9467bd','#8c564b')

        #colours = ('#d62728','#ff7f0e','#e377c2','#2ca02c','#9467bd','#8c564b')

        xticks = {(i + 0.5): k for i, k in enumerate(_df.index)}

        fig, ax = plt.subplots()

        #plt.axhline(y = _df['No Policy', 'mean']/1e6, color = 'black', linestyle = '--')
        ax.bar(
            xticks.keys(),
            _df['mean'].values,
            yerr=yerr,
        #    alpha=0.5,
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

        
    def parabolaFunc(x, constant, linear, quadratic):
        y = constant + linear * x + quadratic * (x**2)
        return y
        
    def LinFunc(x, constant_lin, linear_lin):
        y = constant_lin + linear_lin * x
        return y
        

    def adjust(df_progress_causes_total):
        data = df_progress_causes_total
        s = pd.Series(data)

        # Extract draw information from index
        draw_index = [index[0] for index in s.index]

        # Create DataFrame with multi-index and one row
        df = pd.DataFrame([s.values], columns=pd.MultiIndex.from_tuples(s.index))
        # Set the first level of columns to 'draw'
        df.columns = pd.MultiIndex.from_tuples([(draw, col[1]) for draw, col in zip(draw_index, df.columns)])
        df.columns.names = ['draw', 'run']
        return df
      
    def fit(x):
        return 278.46631062 + x*(-13.04663168) + x*x*0.76443852
        
    # Obtain parameter names for this scenario file
    param_names = get_parameter_names_from_scenario_file()
    print('Param names are', param_names)


    # ================================================================================================
    # Collect and store data

    ALL = {}

    for year in range(2019, max_year+1, 2):
        year_target = year
        life_expectancy = get_life_expectancy_estimates(results_folder,target_period=(datetime.date(year_target, 1, 1), datetime.date(year_target+1, 12, 31)), summary=False).pipe(set_param_names_as_column_index_level_0)
        average_values = life_expectancy.mean().to_frame().T
       # print(summarize(life_expectancy))
       # print(summarize(average_values))
        ALL[year_target] = summarize(average_values)

    # Concatenate the DataFrames into a single DataFrame
    concatenated_df = pd.concat(ALL.values(), keys=ALL.keys())
    concatenated_df.index = concatenated_df.index.set_names(['date', 'index_original'])
    concatenated_df = concatenated_df.reset_index(level='index_original',drop=True)
    LifeExpectancy = concatenated_df.copy()

    num_dalys = extract_results(
        results_folder,
        module='tlo.methods.healthburden',
        key='dalys_stacked',
        custom_generate_series=get_num_dalys,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0)
    
    num_deaths = extract_results(
        results_folder,
        module='tlo.methods.demography',
        key='death',
        custom_generate_series=get_num_deaths,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0)
    
    # Save this
    Total_DALYs = num_dalys.copy()
    Total_deaths = num_deaths.copy()
    
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
    DALYs_with_time = concatenated_df.copy()
    
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
    for year in range(min_year, max_year+1):
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
    DALYs_with_time_and_cause = concatenated_df.copy()
    
    
    
    
    pop_model = extract_results(results_folder,
                                module="tlo.methods.demography",
                                key="population",
                                column="total",
                                index="date",
                                do_scaling=True
                                ).pipe(set_param_names_as_column_index_level_0)
    
    pop_model.index = pop_model.index.year
    pop_model = pop_model[(pop_model.index >= min_year) & (pop_model.index <= max_year)]
    Population_with_time = pop_model.copy()
    assert DALYs_with_time.index.equals(Population_with_time.index)
    assert all(DALYs_with_time.columns == Population_with_time.columns)
    DALYs_with_time_popnorm = (DALYs_with_time/Population_with_time).copy()
    DALYs_with_time_and_cause_popnorm = (DALYs_with_time_and_cause/Population_with_time).copy()

    DALYs_with_time_from_2019 = DALYs_with_time[(DALYs_with_time.index >= 2019)]

    filtered_df =DALYs_with_time_from_2019
    # Average rows two by two
    # Using iloc to split the DataFrame into chunks of two rows and compute their mean
    averaged_rows = [filtered_df.iloc[i:i+2].mean() for i in range(0, len(filtered_df), 2)]

    # Create a new DataFrame from the averaged rows
    df_averaged = pd.DataFrame(averaged_rows)

    # Set the correct index for the new DataFrame
    # We use the original index values for the new DataFrame
    new_index = [filtered_df.index[i] for i in range(0, len(filtered_df), 2)]
    df_averaged.index = new_index
    df_averaged.index.name = 'date'

    DALYs_with_time_from_2019_summarised = summarize(df_averaged)

    Total_DALYs_popnorm = (DALYs_with_time_popnorm.mean(axis=0))

    data = Total_DALYs_popnorm
    s = pd.Series(data)

    # Extract draw information from index
    draw_index = [index[0] for index in s.index]

    # Create DataFrame with multi-index and one row
    df = pd.DataFrame([s.values], columns=pd.MultiIndex.from_tuples(s.index))
    # Set the first level of columns to 'draw'
    df.columns = pd.MultiIndex.from_tuples([(draw, col[1]) for draw, col in zip(draw_index, df.columns)])
    df.columns.names = ['draw', 'run']
    Total_DALYs_popnorm = df

    Total_DALYs_summarised = summarize(Total_DALYs).loc[0].unstack()
    Total_deaths_summarised = summarize(Total_deaths).loc[0].unstack()
    Total_DALYs_popnorm_summarised = summarize(Total_DALYs_popnorm).loc[0].unstack()
    DALYs_with_time_summarised = summarize(DALYs_with_time)
    DALYs_with_time_popnorm_summarised = summarize(DALYs_with_time_popnorm)
    DALYs_with_time_and_cause_summarised = summarize(DALYs_with_time_and_cause)

    DALYs_with_time_and_cause_popnorm_summarised = summarize(DALYs_with_time_and_cause_popnorm)
    Population_with_time_summarised = summarize(Population_with_time)
    # print(DALYs_with_time_and_cause.index.get_level_values('cause').unique())
  
    Total_DALYs_in_period = adjust(((DALYs_with_time_and_cause[DALYs_with_time_and_cause.index.get_level_values('date') >= 2019]).groupby('date').sum()).sum())
    Total_DALYs_in_period_summarised = summarize(Total_DALYs_in_period).loc[0].unstack()

    unique_causes = DALYs_with_time_and_cause.index.get_level_values('cause').unique()
    print(unique_causes)

    df_total = DALYs_with_time_and_cause
    df_total = df_total[df_total.index.get_level_values('date') >= 2019]
    df_total_2018 = DALYs_with_time_and_cause[DALYs_with_time_and_cause.index.get_level_values('date') == 2018]

    df_total_sum = (df_total.groupby("date").sum()).sum()
    data = df_total_sum
    s = pd.Series(data)
    # Extract draw information from index
    draw_index = [index[0] for index in s.index]
    # Create DataFrame with multi-index and one row
    df = pd.DataFrame([s.values], columns=pd.MultiIndex.from_tuples(s.index))
    # Set the first level of columns to 'draw'
    df.columns = pd.MultiIndex.from_tuples([(draw, col[1]) for draw, col in zip(draw_index, df.columns)])
    df.columns.names = ['draw', 'run']
    df_total_sum = df
    
    
    # ================================================================================================
    # PLOT 1: time evolution of total DALYs and life expectancy
    def plot_X_with_time(df, ylabel, plot_name, title=''):

        for case in scenarios_map.keys():
            scenarios_spec = scenarios_map[case]['scenarios']

            i = 1
            fig, ax = plt.subplots(figsize=(6.5, 8))
            for draw in scenarios_spec:
                draw_data = df[draw]
                x_values = draw_data.index.get_level_values('date')
                y_values = draw_data['mean']
                linewidth = 2
                if 'const' in draw:
                    linewidth = 4
                # Plot the data
                #ax.plot(x, y)
                lower_bounds = draw_data['lower']
                upper_bounds = draw_data['upper']
                draw_label = sf_scenario_name_dict[draw]

                color = sf_color_dict[draw]
                ax.plot(x_values, y_values, label=f'{draw_label}', linewidth=linewidth, color=color)
                ax.fill_between(x_values, lower_bounds, upper_bounds, alpha=0.2, color=color)
                i += 1
                   # plt.errorbar(x_values, y_values, yerr=[y_values - lower_bounds, upper_bounds - y_values], label=f'{draw}')
            # Add a vertical dashed line at the specified x value
            #ax.set_ylim(6,18)
            start_year = min_year
            end_year = max_year
            tick_interval = 2
            if plot_name == 'LifeExpectancy' or plot_name == 'Yearly_DALYs_Averaged':
                start_year = 2019
                
            xtick_positions = np.arange(start_year, end_year, tick_interval)
            xtick_labels = [(str(year)) for year in xtick_positions]            # plt.legend()

            if plot_name == 'LifeExpectancy' or plot_name == 'Yearly_DALYs_Averaged':
                xtick_labels = [(str(year) + " - " + str(year+1)) for year in xtick_positions]            # plt.legend()

            plt.xticks(xtick_positions, xtick_labels, rotation=90)  # Rotate labels for better visibility
            plt.xlim(start_year,2039)

            plt.xlabel("Year")
            plt.grid(True)
            #plt.title(str(case))
            subfolder = ''
            if title != '':
                if title in index_HTM_plot:
                    subfolder = "HTM_plots/"
                elif title in index_NCDs_plot:
                    subfolder = "NCDs_and_RTIs_plots/"
                elif title in index_NeoChild_plot:
                    subfolder = "RMNCH_plots/"
                    
                if title in ymin_range.keys():
                    ax.set_ylim(ymin_range[title],ymax_range[title])
                if 'quo' in case:
                    title_comp = title #+ " (pres.-day cons. avail.)"
                else:
                    title_comp = title #+ " (perf. cons. avail.)"
                plt.title(title_comp)
            name_fig = 'plots/' + subfolder + plot_name + '_' + str(case) + '.png'
            plt.ylabel(ylabel)

            name_fig = name_fig.replace(" ", "_")
            if title == 'AIDS':
                plt.legend(loc='lower left')
            elif title == 'Childhood Diarrhoea' or title == 'Cancer (Other)':
                plt.legend(loc='upper right')
            plt.tight_layout()

            plt.savefig(name_fig, dpi=400)
            plt.close()
    
    df = LifeExpectancy.copy()
    plot_X_with_time(df, "Life Expectancy (years)", "LifeExpectancy")
    df = DALYs_with_time_from_2019_summarised.copy()/1e6
    plot_X_with_time(df, "Yearly DALYs (millions)", "Yearly_DALYs_Averaged")
    
    for cause in DALYs_with_time_and_cause_summarised.index.get_level_values('cause').unique():
        df_filtered = DALYs_with_time_and_cause_summarised.xs(cause, level='cause')/1e6
        use_cause = cause.replace("/", "_and_")
        plot_X_with_time(df_filtered, "Yearly DALYs (millions)", "Yearly_DALYs_" + use_cause, cause)

    # ================================================================================================
    # Plot 2: DALYs incurred as a fnc of growth and total health expenditure
    
    def format_ticks(x, pos):
        return f"{x:.2f}"  # Format x to two decimal places

    def plot_total_burden_vs_THE_alt_axis(df, ylabel, plot_name):
    
        # Initialize the plot
        plt.rcParams.update({'font.size': 9})

        x_values_dict = sf_THE_dict
        x_values_THE = [x_values_dict[stat] for stat in df.index]
        df['THE'] = x_values_THE
        x_values_dict = sf_comb_rate
        x_values_ExR = [x_values_dict[stat] for stat in df.index]
        df['YearlyExpRate'] = x_values_ExR
        df['unc'] = (df['upper'] - df['lower'])/2
        df['err_uc_lower'] = np.abs(1.0 - (df['lower']/(df['mean'] - df['unc'])))
        df['err_uc_upper'] =np.abs(1.0 - (df['upper']/(df['mean'] + df['unc'])))
        
        fig, ax = plt.subplots(figsize=(9, 7))

        # Create ax2 as a twin of ax1

        # Loop over scenarios
        for case, scenario_info in scenarios_map.items():
            spec_cases = pd.DataFrame(columns=df.columns)

            if len(scenarios_map)> 1:
                chosen_colour = scenario_info['colour']
                chosen_colour_shaded = scenario_info['colour']
                label_case=scenario_info['title']
                chosen_colour_text = 'black'
            else:
                chosen_colour = 'black'
                chosen_colour_shaded = scenario_info['colour']
                label_case=None
                chosen_colour_text = chosen_colour_shaded


            print("I'm at case ", case)
            scenario = scenario_info['scenarios']
            filtered_df = df[df.index.isin(scenario)].copy()
            first_two_rows = filtered_df.head(3)
            popt,pcov = curve_fit(LinFunc, first_two_rows['YearlyExpRate'], first_two_rows['mean'], sigma=first_two_rows['unc'], absolute_sigma=True)
            constant_lin = popt[0]
            linear_lin = popt[1]
            yfit2 = constant_lin + (linear_lin * filtered_df['YearlyExpRate'])
            
            
            # Plot fit data
            if 'Perfect' in case:
                label = "Linear fit (first three scenarios)"
            else:
                label = ""
            ax.plot(filtered_df['YearlyExpRate'], yfit2,
                    color="Grey",
                    alpha=0.6,
                    linestyle="-", label=label)
                    
            # Plot on ax1 (bottom x-axis)
            for s in scenario:
                row = df.loc[s]
                if 'case 1' in s or 'case 3' in s:
                    spec_cases = pd.concat([spec_cases, pd.DataFrame(row.to_frame().T)], ignore_index=True)
                label = ''
                if 'No growth' in s and len(scenarios_map)>1:
                    label = scenario_info['title']
                ax.errorbar(row['YearlyExpRate'], row['mean'], yerr=[[row['mean'] - row['lower']], [row['upper'] - row['mean']]],
                             fmt='o', color=chosen_colour, linestyle='None', capsize=5,label=label)

            #plt.plot(filtered_df['YearlyExpRate'], filtered_df['mean'], marker='o', color=scenario_info['colour'], label=scenario_info['title'])
            #plt.fill_between(filtered_df['YearlyExpRate'], filtered_df['lower'], filtered_df['upper'], alpha=0.2, color=scenario_info['colour'])

            popt, pcov = curve_fit(parabolaFunc, filtered_df['YearlyExpRate'], filtered_df['mean'], sigma=filtered_df['unc'], absolute_sigma=True)

            # Extract out the fitted parameters and standard errors
            constant = popt[0]
            linear = popt[1]
            quadratic = popt[2]
            perr = np.sqrt(np.diag(pcov))
           # print(popt)
          #  print(perr)
            constant_err = np.sqrt(pcov[0][0])
            linear_err = np.sqrt(pcov[1][1])
            quadratic_err = np.sqrt(pcov[2][2])
            yfit = constant + (linear * filtered_df['YearlyExpRate']) + (quadratic * (filtered_df['YearlyExpRate']**2))

            # Plot fit data
            if 'Perfect' in case:
                label = "Best-fit"
            else:
                label = ""

            ax.plot(filtered_df['YearlyExpRate'], yfit,
                    color="black",
                    label=label,
                    linestyle="dashed")
            """
            print(f"constant = {constant:.7f}")
            print(f"constant std. error = {constant_err:.7f} DALYs")
            print(f"linear = {linear:.2E}")
            print(f"linear std. error = {linear_err:.2E} DALYs")
            print(f"quadratic = {quadratic:.4E}")
            print(f"quadratic std. error = {quadratic_err:.2E} DALYs")
            """
            constant_str = f"{constant:.2f}"
            constant_err_str = f"{constant_err:.2f}"
            linear_str = f"{linear:.2f}"
            linear_err_str = f"{linear_err:.2f}"
            quadratic_str = f"{quadratic:.2f}"
            quadratic_err_str = f"{quadratic_err:.2f}"
            variables = ['Quadratic', 'Linear', 'Constant']
            values = [f"{quadratic_str} (std error: {quadratic_err_str})",
                      f"{linear_str} (std error: {linear_err_str})",
                      f"{constant_str} (std error: {constant_err_str})"]

            # Prepare the data in the required format
            if len(scenarios_map)==1:
                table_data = list(zip(variables, values))
                the_table = ax.table(cellText=table_data, loc='left', bbox=[0.6, 0.8, 0.35, 0.15], colWidths=[0.18, 0.42])#[0.5, 0.7, 0.7, 0.3])
            
            x1 = spec_cases.iloc[0, spec_cases.columns.get_loc('YearlyExpRate')]
            y1 = spec_cases.iloc[0, spec_cases.columns.get_loc('mean')]
            x2 = spec_cases.iloc[1, spec_cases.columns.get_loc('YearlyExpRate')]
            y2 = spec_cases.iloc[1, spec_cases.columns.get_loc('mean')]
            
            vertices = [(x1,100), (x2,100), (x2,y2), (x1,y1)]

            # Extract x and y values separately
            x_vals = [v[0] for v in vertices]
            y_vals = [v[1] for v in vertices]
            plt.fill_between(x_vals, y_vals, alpha=0.3, color=chosen_colour_shaded, edgecolor='none')
            
            vertices = [(-10,y1), (x1,y1), (x2,y2), (-10,y2)]

            # Extract x and y values separately
            x_vals = [v[0] for v in vertices]
            y_vals = [v[1] for v in vertices]
            plt.fill_between(x_vals, y_vals, alpha=0.3, color=chosen_colour_shaded, edgecolor='none')
            plt.text((x1+x2)/2, 200, 'IHME \n forecast', fontsize=12,ha='center', color=chosen_colour_text)

        ax.set_xlim(-0.5,7.5)
        if len(scenarios_map)==1:
            ax.set_ylim(180,250)
        else:
            ax.set_ylim(180,282)

        # print(spec_cases)

        # Set labels and legends
        ax.set_xlabel('Yearly expenditure growth (%)')
        ax.set_ylabel('Total DALYs incurred between 2019 - 2040 (millions)')
        ax.legend(loc='lower left')
        ax.grid(axis='y')

        case_label = {
            22.0: 'No growth',
            24.93: '<< GDP growth',
            30.08: '< GDP growth',
            36.53: 'GDP growth',
            44.6: '> GDP growth',
            54.75: '>> GDP growth',
            }
        thelbts_zs = np.array(x_values_ExR)
        thelbts = np.array(x_values_THE)
        p1b = ax.twiny()
        p1b.set_xlim(-0.5,7.5)
        p1b.set_xticks(thelbts_zs, minor=False)
        p1b.set_xticks([0], minor=True)
        combined_labels = [f'{case_label[thelbts[i]]}\n{thelbts[i]}' for i in range(len(thelbts))]
        p1b.set_xticklabels(combined_labels)
        p1b.set_xlabel(r'NHE')
        p1b.tick_params(direction="in")

        

        # Save or show the plot
        plt.savefig(str(output_path) + '/' + plot_name + '_n_scenarios_' + str(len(scenarios_map)) + '.png', dpi=400)
        plt.close()

        
    def plot_comparison_alt_axis(df_original, ylabel, plot_name, use_THE):
    
        plt.rcParams.update({'font.size': 9})
        fig, ax = plt.subplots(figsize=(9, 7))

        for scaled in [True]:
            print("For scaled", scaled)

            for case in scenarios_map.keys():
                #print(case)
                
                if len(scenarios_map)> 1:
                    chosen_colour = scenarios_map[case]['colour']
                    chosen_colour_shaded = scenarios_map[case]['colour']
                    label_case=scenarios_map[case]['title']
                    chosen_colour_text = 'black'
                else:
                    chosen_colour = 'black'
                    chosen_colour_shaded = scenarios_map[case]['colour']
                    label_case=None
                    chosen_colour_text = chosen_colour_shaded
                
                scenarios_spec = scenarios_map[case]['scenarios']
                comparison = scenarios_map[case]['comparison']
                colour = scenarios_map[case]['colour']
                
                num_dalys_averted = summarize(
                    -1.0 *
                    pd.DataFrame(
                        find_difference_relative_to_comparison(
                            df_original.loc[0],
                            comparison=comparison,
                            scaled=scaled)
                    ).T
                ).iloc[0].unstack()

                # Plots....
                if scaled:
                    num_dalys_averted *= 100.0
                else:
                    num_dalys_averted *= (1/1e6)
                
                df = num_dalys_averted
                df = df.sort_values(by='mean', ascending=False)
                x_values_dict = sf_THE_dict
                x_values_THE = [x_values_dict[stat] for stat in df.index]
                df['THE'] = x_values_THE
                x_values_dict = sf_comb_rate
                x_values_ExR = [x_values_dict[stat] for stat in df.index]
                df['YearlyExpRate'] = x_values_ExR
                df['unc'] = (df['upper'] - df['lower'])/2
                df['err_uc_lower'] = np.abs(1.0 - (df['lower']/(df['mean'] - df['unc'])))
                df['err_uc_upper'] =np.abs(1.0 - (df['upper']/(df['mean'] + df['unc'])))
              
                spec_cases = pd.DataFrame(columns=df.columns)

                for scenario in scenarios_spec:
                    if scenario != comparison:
                        row = df.loc[scenario]
                        label = ''
                        color = colour
                        #print(row)
                        ax.errorbar(row['YearlyExpRate'], row['mean'], yerr=[[row['mean'] - row['lower']], [row['upper'] - row['mean']]],
                                     fmt='o', color=chosen_colour, linestyle='None', capsize=5)
                        if 'case 1' in scenario or 'case 3' in scenario:
                            spec_cases = pd.concat([spec_cases, pd.DataFrame(row.to_frame().T)], ignore_index=True)
                                     
                ax.plot((((1+scenarios_properties[1]['g_GDP']/100)*(1+scenarios_properties[1]['g_fHE']/100))-1)*100, 0, 'ko')
                
                x1 = spec_cases.iloc[0, spec_cases.columns.get_loc('YearlyExpRate')]
                y1 = spec_cases.iloc[0, spec_cases.columns.get_loc('mean')]
                x2 = spec_cases.iloc[1, spec_cases.columns.get_loc('YearlyExpRate')]
                y2 = spec_cases.iloc[1, spec_cases.columns.get_loc('mean')]
                
                vertices = [(x1,-100), (x2,-100), (x2,y2), (x1,y1)]

                # Extract x and y values separately
                x_vals = [v[0] for v in vertices]
                y_vals = [v[1] for v in vertices]
                plt.fill_between(x_vals, y_vals, alpha=0.3, color=chosen_colour_shaded, edgecolor='none')
                
                vertices = [(-10,y1), (x1,y1), (x2,y2), (-10,y2)]

                # Extract x and y values separately
                x_vals = [v[0] for v in vertices]
                y_vals = [v[1] for v in vertices]
                plt.fill_between(x_vals, y_vals, alpha=0.3, color=chosen_colour_shaded, edgecolor='none')
                plt.text((x1+x2)/2, -15, 'IHME \n forecast', fontsize=12,ha='center', color=chosen_colour_text)

                #if scaled:
                #    print(df)
                    
            ax.set_xlim(-0.5,7.5)
            if len(scenarios_map)>1:
                ax.set_ylim(-30,10)
            else:
                ax.set_ylim(-30,10)

            case_label = {
                22.0: 'No growth',
                24.93: '<< GDP growth',
                30.08: '< GDP growth',
                36.53: 'GDP growth',
                44.6: '> GDP growth',
                54.75: '>> GDP growth',
                }
            thelbts_zs = np.array(x_values_ExR)
            thelbts = np.array(x_values_THE)
            p1b = ax.twiny()
            p1b.set_xlim(-0.5,7.5)
            p1b.set_xticks(thelbts_zs, minor=False)
            p1b.set_xticks([0], minor=True)
            combined_labels = [f'{case_label[thelbts[i]]}\n{thelbts[i]}' for i in range(len(thelbts))]
            p1b.set_xticklabels(combined_labels)
            p1b.set_xlabel(r'NHE')
            p1b.tick_params(direction="in")
            ax.set_xlabel('Yearly expenditure growth (%)')
            ax.grid(axis='y')

            if scaled:
                ax.set_ylabel('% ' + ylabel + ' compared to GDP growth scenario')
                plt.savefig(str(output_path) + '/' + plot_name + '_frac_diff_with_GDP_growth_vs_THE_n_scenarios_' + str(len(scenarios_map)) + '.png', dpi=400)
            else:
                ax.set_ylabel(ylabel + ' compared GDP growth scenario (millions)')
                plt.savefig(str(output_path) + '/' + plot_name + '_diff_with_GDP_growth_vs_THE_n_scenarios_' + str(len(scenarios_map)) + '.png', dpi=400)

            plt.close()
            
    df = (Total_DALYs_in_period_summarised/1e6).sort_values(by='mean', ascending=False)
    plot_total_burden_vs_THE_alt_axis(df, 'Total DALYs (millions)', 'DALYs_vs_THE_with_alt_axis')
    df = Total_DALYs_in_period
    plot_comparison_alt_axis(df,"total DALYs averted between 2019 - 2040", "DALYs_averted", False)
    

    # ================================================================================================
    # Plot 3: DALYs by area of health
    
    def check_if_cause_is_missing(index):

        unique_causes = df_total.index.get_level_values('cause').unique()

        # Check for causes that are not found in the DataFrame
        missing_causes = [cause for cause in index if cause not in unique_causes]
        if missing_causes:
            warnings.warn(f"The following causes were not found in the DataFrame: {missing_causes}")

      
    index_HTM = ['AIDS', 'TB (non-AIDS)', 'Malaria']
    index_NeoChild = ['Lower respiratory infections','Childhood Diarrhoea', 'Maternal Disorders', 'Measles', 'Neonatal Disorders', 'Schistosomiasis'] #'Congenital birth defects',
    index_NCDs = ['COPD', 'Cancer (Bladder)', 'Cancer (Breast)', 'Cancer (Oesophagus)', 'Cancer (Other)', 'Cancer (Prostate)','Depression / Self-harm', 'Diabetes', 'Epilepsy', 'Heart Disease', 'Kidney Disease', 'Stroke']
    check_if_cause_is_missing(index_HTM)
    check_if_cause_is_missing(index_NeoChild)
    check_if_cause_is_missing(index_NCDs)
    
    df_total_time_average = df_total.groupby('cause').mean()
    df_total_time_average_2018 = df_total_2018.groupby('cause').mean()

    def sum_over_indices_and_add(df, index_list):

        # Compute the sum of the selected rows
        total_row = df.loc[index_list].sum()

        # Create a new MultiIndex row while preserving the structure
        combined_index = pd.Index(['Combined'], name=df.index.names)

        # Convert summed row into DataFrame with correct index and columns
        total_row_df = pd.DataFrame([total_row], index=combined_index, columns=df.columns)

        # Concatenate to maintain structure
        df = pd.concat([df, total_row_df])
        df.index.names = ['cause']  # Explicitly set the index name to 'cause'
        return df
        
    def sum_over_indices_and_add_NCDs_and_RTI(df, index_list):

        # Compute the sum of the selected rows
        total_row = df.loc[index_list].sum()

        # Create a new MultiIndex row while preserving the structure
        combined_index = pd.Index(['NCDs'], name=df.index.names)

        # Convert summed row into DataFrame with correct index and columns
        total_row_df = pd.DataFrame([total_row], index=combined_index, columns=df.columns)

        # Concatenate to maintain structure
        df = pd.concat([df, total_row_df])
        df.index.names = ['cause']  # Explicitly set the index name to 'cause'
        index_add_RTIs = index_list + ['Transport Injuries']
        total_row = df.loc[index_add_RTIs].sum()
        combined_index = pd.Index(['Combined'], name=df.index.names)

        # Convert summed row into DataFrame with correct index and columns
        total_row_df = pd.DataFrame([total_row], index=combined_index, columns=df.columns)

        # Concatenate to maintain structure
        df = pd.concat([df, total_row_df])
        df.index.names = ['cause']  # Explicitly set the index name to 'cause'
        
        return df

    def plot_burden_by_area_vs_THE(df_total_time_average, df_total_time_average_2018, index, ylabel, plot_name, use_THE, title, inc_breakdown):

        df = df_total_time_average.copy()
        if 'NCD' in title:
            df = sum_over_indices_and_add_NCDs_and_RTI(df, index)
        else:
            df = sum_over_indices_and_add(df, index)
        df_total_time_average_temp = df.copy()

        df = df_total_time_average_2018.copy()
        if 'NCD' in title:
            df = sum_over_indices_and_add_NCDs_and_RTI(df, index)
        else:
            df = sum_over_indices_and_add(df, index)
        df_total_time_average_2018_temp = df.copy()
        
        """
        # Compute the sum of the selected rows
        #total_row = df.loc[index].sum()

        # Create a new MultiIndex row while preserving the structure
       # combined_index = pd.Index(['Combined'], name=df.index.names)

        # Convert summed row into DataFrame with correct index and columns
       # total_row_df = pd.DataFrame([total_row], index=combined_index, columns=df.columns)

        # Concatenate to maintain structure
        #df = pd.concat([df, total_row_df])
       # df.index.names = ['cause']  # Explicitly set the index name to 'cause'
        
        
        #print(df.index.names)
        #print(df)
       # exit(-1)
        #df_total_time_average_temp = df.copy()
        
        df = df_total_time_average_2018.copy()

        # Compute the sum of the selected rows
        total_row = df.loc[index].sum()

        # Create a new MultiIndex row while preserving the structure
        combined_index = pd.Index(['Combined'], name=df.index.names)

        # Convert summed row into DataFrame with correct index and columns
        total_row_df = pd.DataFrame([total_row], index=combined_index, columns=df.columns)

        # Concatenate to maintain structure
        df = pd.concat([df, total_row_df])
        df.index.names = ['cause']  # Explicitly set the index name to 'cause'
        print(df.index.names)
        print(df)
       # exit(-1)
        df_total_time_average_2018_temp = df.copy()
        """
        if 'NCD' in title:
            index_with_combined = index + ['NCDs', 'Transport Injuries','Combined']
        else:
            index_with_combined = index + ['Combined']
            
        df = summarize(df_total_time_average_temp[df_total_time_average_temp.index.get_level_values('cause').isin(index_with_combined)])

        df_2018 = summarize(df_total_time_average_2018_temp[df_total_time_average_2018_temp.index.get_level_values('cause').isin(index_with_combined)])

        
        df = df.sort_values(by=('No growth status quo', 'mean'), ascending=False)
        if 'NCD' in title:
            df = df.sort_index(key=lambda x: x != 'NCDs')
            df_2018 = df_2018.sort_index(key=lambda x: x != 'NCDs')
        df = df.sort_index(key=lambda x: x != 'Combined')
        df_2018 = df_2018.sort_index(key=lambda x: x != 'Combined')
        
         #df_total_time_average.loc['Total'] = df_total_time_average.sum()

        plt.figure(figsize=(7, 12))
        plt.rcParams.update({'font.size': 16})
                
 
        for case in scenarios_map.keys():
            scenario = scenarios_map[case]['scenarios']
            filtered_df = df.loc[:, scenario]
            #print(filtered_df)
            if use_THE:
                x_values_dict = sf_THE_dict
            else:
                x_values_dict = sf_comb_rate
            
            #filtered_df = df.loc[scenario].copy()
            
            if 'quo' in case:
                linestyle = '-'
            else:
                linestyle = '--'
 
            if 'NCD' in title:
                causes_to_include = ['NCDs', 'Transport Injuries', 'Combined']
            else:
                if inc_breakdown:
                    causes_to_include = filtered_df.index
                else:
                    causes_to_include = ['Combined']
 
            for cause in causes_to_include:
                print("Including", cause)
                # Extract the mean, lower, and upper values for the current cause
                mean_values = filtered_df.loc[cause].xs('mean', level='stat')
                lower_values = filtered_df.loc[cause].xs('lower', level='stat')
                upper_values = filtered_df.loc[cause].xs('upper', level='stat')

                lower_error = [mean - lower for mean, lower in zip(mean_values, lower_values)]
                upper_error = [upper - mean for upper, mean in zip(upper_values, mean_values)]
                yerr = [lower_error, upper_error]

                # Extract corresponding x_values using the draw columns
                x_values = [x_values_dict[draw] for draw in mean_values.index]

                if 'quo' in case:
                    marker_type = 'o'
                    if cause == 'Stroke':
                        markersize=14
                    else:
                        markersize=9
                    plt.errorbar(
                        x_values, mean_values, yerr=yerr, marker='o', markerfacecolor='none', markeredgecolor=color_palette[cause],
                        markeredgewidth=2, color=color_palette[cause], linewidth=2, markersize=markersize, capsize=5, linestyle='None'
                    )
                else:
                    marker_type = 'x'

                    plt.errorbar(
                        x_values, mean_values, yerr=yerr, fmt='x',
                        color=color_palette[cause], linewidth=2, markersize=12, mew=2,capsize=5,
                    )



                # Plot the mean values against the x_values
               # if 'quo' in case:
                #    plt.plot(x_values, mean_values.values, marker='o', label=cause, linestyle=linestyle, color = color_palette[cause], linewidth=2,markersize=8)
                #else:
                #    plt.plot(x_values, mean_values.values, marker='x', linestyle=linestyle, color = color_palette[cause], linewidth=2,markersize=10, mew=2)

                # Plot the shaded region for the lower and upper bounds
                #plt.fill_between(x_values, lower_values.values, upper_values.values, alpha=0.2, color = color_palette[cause])

                y = df_2018.loc[cause, ('No growth status quo', 'mean')]
                yerr_lower = y - df_2018.loc[cause, ('No growth status quo', 'lower')]
                yerr_upper = df_2018.loc[cause, ('No growth status quo', 'upper')] - y

                y = df_2018.loc[cause, ('No growth perfect consumables', 'mean')]
                yerr_lower = y - df_2018.loc[cause, ('No growth perfect consumables', 'lower')]
                yerr_upper = df_2018.loc[cause, ('No growth perfect consumables', 'upper')] - y
                plt.errorbar(-2, y, yerr=[[yerr_lower], [yerr_upper]], marker='x', linestyle=linestyle, color=color_palette[cause], linewidth=2)
                if 'Perfect' in case:
                    if cause == 'Combined':
                        if 'HTM' in title:
                            label = 'HTM combined'
                        elif 'NCD' in title:
                            label = 'NCDs and RTIs combined'
                        else:
                            if cause == 'Transport Injuries':
                                label = 'RTIs'
                            else:
                                label = cause
                    else:
                        label = cause
                    plt.axhline(y=df_2018.loc[cause, ('No growth status quo', 'mean')], color=color_palette[cause], linestyle='-', linewidth=2,label=label)
                else:
                    plt.axhline(y=df_2018.loc[cause, ('No growth status quo', 'mean')], color=color_palette[cause], linestyle='-', linewidth=2)


        if 'AIDS' in index:
            if len(scenarios_map)==1:
                plt.plot(-1.8, 2,  marker=marker_type, markerfacecolor='none', markeredgecolor='black', markeredgewidth=2, color='black', linewidth=2,markersize=8, label="2019 - 2040 average")
            else:
                 plt.plot(-1.8, 2,  marker='o', markerfacecolor='none', markeredgecolor='black', markeredgewidth=2, color='black', linewidth=2,markersize=8, label="2019 - 2040 av. (pres.-day cons. avail.)")
                 plt.plot(-1.8, 2,  marker='x', markerfacecolor='none', markeredgecolor='black', markeredgewidth=2, color='black', linewidth=2,markersize=8, label="2019 - 2040 av. (perf. cons. avail.)")
            plt.plot(-1.8, 2, linestyle='-',  color='black', linewidth=2,markersize=10, mew=2, label="2018 value")
            plt.legend()
        if 'NCD' in title:
           # plt.plot(-1.8, 2, linestyle='-',  color=color_palette['NCDs'], linewidth=2,markersize=10, mew=2, label="NCDs")
           # plt.plot(-1.8, 2, linestyle='-',  color=color_palette['Transport Injuries'], linewidth=2,markersize=10, mew=2, label="RTIs")
           # plt.plot(-1.8, 2, linestyle='-',  color=color_palette['Combined'], linewidth=2,markersize=10, mew=2, label="Combined")
            plt.legend()
            
        plt.axvspan(1.1, 2.6, facecolor=scenarios_map[case]['colour'], alpha=0.3,edgecolor='none')
        plt.text((1.1+2.6)/2, 3.5, 'IHME \n forecast', fontsize=12,ha='center', color=scenarios_map[case]['colour'])
        if inc_IHME_2024:
            plt.axvspan(2, 3.3, facecolor='orange', alpha=0.3,edgecolor='none')
            plt.text((2+3.3)/2, 4, 'IHME \n forecast \n 2024', fontsize=12,ha='center', color='red')
        
        if use_THE:
            plt.xlabel('NHE')
        else:
            plt.xlabel('Yearly expenditure growth (%)')
        plt.ylabel(ylabel)
        plt.ylim(0,5.5)
       # plt.grid(True)
       

        tick_interval = 1

        xtick_positions = np.arange(0, 9, tick_interval)
        xtick_labels = [str(year) for year in xtick_positions]            # plt.legend()
        plt.xticks(xtick_positions, xtick_labels)  # Rotate labels for better visibility
        plt.xlim(-0.1,7.5)
        plt.title(title)
        plt.savefig(str(output_path) + '/' + plot_name + '_n_scenarios_' + str(len(scenarios_map)) + '.png', dpi=400)
        plt.close()

    plot_burden_by_area_vs_THE(df_total_time_average/1e6,df_total_time_average_2018/1e6, index_HTM, "DALYs per year (millions)", "HTM_evolution", False, "HTM", True)
    plot_burden_by_area_vs_THE(df_total_time_average/1e6,df_total_time_average_2018/1e6, index_NeoChild, "", "RMNCH_evolution", False, "RMNCH", False)
    plot_burden_by_area_vs_THE(df_total_time_average/1e6,df_total_time_average_2018/1e6, index_NCDs, "", "NCDs_evolution", False, "NCDs and RTIs", True)
  

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
            "src/scripts/healthsystem/impact_of_const_capabilities_expansion/scenario_impact_of_capabilities_expansion_scaling.py "
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
