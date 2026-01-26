#!/usr/bin/env python
# coding: utf-8

# In[187]:


from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tlo import Date
from tlo.analysis.utils import (
    extract_results,
    summarize,
)
import geopandas as gpd


# In[188]:


results_folder = Path('/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/climate_scenario_runs_baseline-2025-12-04T163755Z')

output_folder = Path('/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/climate_scenario_runs_baseline-2025-12-04T163755Z')


# In[189]:


climate_sensitivity_analysis = False
parameter_sensitivity_analysis = False
main_text = True
mode_2 = False

scenario_names_all = ["baseline", "ssp126_highest", "ssp126_lowest", "ssp126_mean", "ssp245_highest", "ssp245_lowest", "ssp245_mean",  "ssp585_highest", "ssp585_lowest", "ssp585_mean"]
climate_sensitivity_analysis = True
parameter_sensitivity_analysis = False
main_text = True
mode_2 = False
if climate_sensitivity_analysis:

    suffix = "climate_SA"
    scenarios_of_interest = range(len(scenario_names_all) -1)
    scenario_names = scenario_names_all[1:]
if parameter_sensitivity_analysis:
    scenario_names = range(0, 9, 1)
    scenarios_of_interest = scenario_names

    suffix = "parameter_SA"
if main_text:
    scenario_names = [
        "Baseline",
        "SSP 2.45 Mean",
    ]
    suffix = "main_text"
    scenarios_of_interest = [0, 1]

if mode_2:
    scenario_names = [
        "Baseline",
        "SSP 5.85 Mean",
    ]
    suffix = "mode_2"
    scenarios_of_interest = [0, 1]


# In[190]:


min_year = 2025
max_year = 2041
spacing_of_years = 1
PREFIX_ON_FILENAME = '1'

scenario_names_all = ["baseline", "ssp126_highest", "ssp126_lowest", "ssp126_mean", "ssp245_highest", "ssp245_lowest", "ssp245_mean",  "ssp585_highest", "ssp585_lowest", "ssp585_mean"]

scenario_colours = ['#0081a7', '#00afb9', '#FEB95F', '#fed9b7', '#f07167']*4


vmin = 0.5e+06
vmax = 2.4e+06


# Helper functions

# In[196]:


target_year_sequence = range(min_year, max_year, spacing_of_years)

# Define the extraction functions (add these near your other helper functions)
def get_num_treatments_total_delayed(_df):
    """Count total number of delayed HSI events"""
    _df["date"] = pd.to_datetime(_df["date"])
    _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]
    print(_df)
    return pd.Series(len(_df), name="total")

def get_num_treatments_total_cancelled(_df):
    """Count total number of cancelled HSI events"""
    _df["date"] = pd.to_datetime(_df["date"])
    _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]
    return pd.Series(len(_df), name="total")

def get_num_treatments_total(_df):
    """Sum all treatment counts"""
    _df["date"] = pd.to_datetime(_df["date"])
    _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]
    total = {}
    for d in _df["hsi_event_key_to_counts"]:
        for k, v in d.items():
            total[k] = total.get(k, 0) + v
    return pd.Series(sum(total.values()), name="total")



# In[ ]:


def get_num_dalys_by_month(_df):
    """Sum all DALYs across all causes by month for the target year"""
    _df["date"] = pd.to_datetime(_df["date"])
    _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]

    # Sum across all disease columns (excluding non-disease columns)
    disease_columns = [col for col in _df.columns 
                      if col not in ['age_range', 'month', 'sex', 'year', 'date']]

    # Group by month and sum
    monthly_dalys = _df.groupby('month')[disease_columns].sum().sum(axis=1)

    return monthly_dalys


# In[197]:


import matplotlib.dates as mdates

# Storage dictionaries
all_scenarios_appointment_delayed_mean = {}
all_scenarios_appointment_cancelled_mean = {}
all_scenarios_dalys_mean = {}

# Define the extraction functions (add these near your other helper functions)
def get_num_treatments_total_delayed(_df):
    """Count total number of delayed HSI events"""
    _df["date"] = pd.to_datetime(_df["date"])
    _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]
    return pd.Series(len(_df), name="total")

def get_num_treatments_total_cancelled(_df):
    """Count total number of cancelled HSI events"""
    _df["date"] = pd.to_datetime(_df["date"])
    _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]
    return pd.Series(len(_df), name="total")

def get_num_treatments_total(_df):
    """Sum all treatment counts"""
    _df["date"] = pd.to_datetime(_df["date"])
    _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]
    total = {}
    for d in _df["hsi_event_key_to_counts"]:
        for k, v in d.items():
            total[k] = total.get(k, 0) + v
    return pd.Series(sum(total.values()), name="total")

# Main loop
for draw in range(len(scenario_names)):
    all_years_data_delayed_mean = {}
    all_years_data_cancelled_mean = {}
    all_years_dalys_mean = {}

    for target_year in target_year_sequence:
        TARGET_PERIOD = (Date(target_year, 1, 1), Date(target_year, 12, 31))

        # Get DALYs
        num_dalys = summarize(extract_results(
            results_folder,
            module='tlo.methods.healthburden',
            key='dalys_stacked',
            custom_generate_series=get_num_dalys_by_cause_label,
            do_scaling=True
        ), only_mean=False, collapse_columns=True)[draw]

        total_population = summarize(
            extract_results(
                results_folder,
                module="tlo.methods.demography",
                key="population",
                custom_generate_series=get_population_for_year,
                do_scaling=True,
            ),
            only_mean=True,
            collapse_columns=True,
        )[draw]

        all_years_dalys_mean[target_year] = pd.Series(
            [num_dalys['mean'].sum() / total_population['mean']], 
            name='mean'
        )

        if scenario_names[draw] == 'Baseline':
            # Baseline has no weather disruptions
            all_years_data_delayed_mean[target_year] = pd.Series([0], name='mean')
            all_years_data_cancelled_mean[target_year] = pd.Series([0], name='mean')
        elif main_text:
            # For main_text mode: no [draw] indexing
            num_delayed = summarize(extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='Weather_delayed_HSI_Event_full_info',
                custom_generate_series=get_num_treatments_total_delayed,
                do_scaling=True
            ), only_mean=False, collapse_columns=True)

            num_cancelled = summarize(extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='Weather_cancelled_HSI_Event_full_info',
                custom_generate_series=get_num_treatments_total_cancelled,
                do_scaling=True
            ), only_mean=False, collapse_columns=True)

            num_total = summarize(extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='hsi_event_counts',
                custom_generate_series=get_num_treatments_total,
                do_scaling=True
            ), only_mean=False, collapse_columns=True)[draw]

            # Extract values from Series
            delayed_val = num_delayed['mean'].values[0] if len(num_delayed['mean'].values) > 0 else 0
            cancelled_val = num_cancelled['mean'].values[0] if len(num_cancelled['mean'].values) > 0 else 0
            total_val = num_total['mean'].values[0] if len(num_total['mean'].values) > 0 else 1

            all_years_data_delayed_mean[target_year] = pd.Series(
                [delayed_val / total_val], 
                name='mean'
            )
            all_years_data_cancelled_mean[target_year] = pd.Series(
                [cancelled_val / total_val], 
                name='mean'
            )
        else:
            # For other modes: with [draw] indexing
            num_delayed = summarize(extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='Weather_delayed_HSI_Event_full_info',
                custom_generate_series=get_num_treatments_total_delayed,
                do_scaling=True
            ), only_mean=False, collapse_columns=True)[draw]

            num_cancelled = summarize(extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='Weather_cancelled_HSI_Event_full_info',
                custom_generate_series=get_num_treatments_total_cancelled,
                do_scaling=True
            ), only_mean=False, collapse_columns=True)[draw]

            num_total = summarize(extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='hsi_event_counts',
                custom_generate_series=get_num_treatments_total,
                do_scaling=True
            ), only_mean=False, collapse_columns=True)[draw]

            # Extract values from Series
            delayed_val = num_delayed['mean'].values[0] if len(num_delayed['mean'].values) > 0 else 0
            cancelled_val = num_cancelled['mean'].values[0] if len(num_cancelled['mean'].values) > 0 else 0
            total_val = num_total['mean'].values[0] if len(num_total['mean'].values) > 0 else 1

            all_years_data_delayed_mean[target_year] = pd.Series(
                [delayed_val / total_val], 
                name='mean'
            )
            all_years_data_cancelled_mean[target_year] = pd.Series(
                [cancelled_val / total_val], 
                name='mean'
            )

    all_scenarios_appointment_delayed_mean[scenario_names[draw]] = all_years_data_delayed_mean
    all_scenarios_appointment_cancelled_mean[scenario_names[draw]] = all_years_data_cancelled_mean
    all_scenarios_dalys_mean[scenario_names[draw]] = all_years_dalys_mean


# Plots

# In[193]:


# Now create the plots with dual y-axes
appointment_delayed_scenarios = {}
appointment_cancelled_scenarios = {}
daly_scenarios = {}

for scenario_name in scenario_names:

    if scenario_name == "Baseline":
        baseline_length = len(target_year_sequence)
        appointment_delayed_scenarios[scenario_name] = [0.0] * baseline_length
        appointment_cancelled_scenarios[scenario_name] = [0.0] * baseline_length
        # Get baseline DALYs from the appropriate data structure
        daly_by_scenario = all_scenarios_dalys_mean.get(scenario_name, {})
        dalys_all_years = [
            value for year in daly_by_scenario.keys()
            for value in daly_by_scenario[year].values.tolist()
        ]
        daly_scenarios[scenario_name] = dalys_all_years
        continue

    delayed_by_scenario = all_scenarios_appointment_delayed_mean[scenario_name]
    cancelled_by_scenario = all_scenarios_appointment_cancelled_mean[scenario_name]
    daly_by_scenario = all_scenarios_dalys_mean[scenario_name]

    delayed_all_years = [
        value for year in delayed_by_scenario.keys()
        for value in delayed_by_scenario[year].values.tolist()
    ]
    cancelled_all_years = [
        value for year in cancelled_by_scenario.keys()
        for value in cancelled_by_scenario[year].values.tolist()
    ]

    dalys_all_years = [
        value for year in daly_by_scenario.keys()
        for value in daly_by_scenario[year].values.tolist()
    ]

    appointment_delayed_scenarios[scenario_name] = delayed_all_years
    appointment_cancelled_scenarios[scenario_name] = cancelled_all_years
    daly_scenarios[scenario_name] = dalys_all_years

scenario_names_filtered = list(appointment_delayed_scenarios.keys())

n_scenarios = len(scenario_names_filtered)
n_cols = min(3, n_scenarios)
n_rows = (n_scenarios + n_cols - 1) // n_cols  

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
if n_scenarios == 1:
    axes = [axes]
else:
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

for i, scenario_name in enumerate(scenario_names_filtered):
    ax1 = axes[i]

    delayed_data = np.array(appointment_delayed_scenarios[scenario_name], dtype=float) * 100
    cancelled_data = np.array(appointment_cancelled_scenarios[scenario_name], dtype=float) * 100
    total_data = delayed_data + cancelled_data
    daly_data = np.array(daly_scenarios[scenario_name], dtype=float)

    n_months = len(delayed_data)
    start_date = pd.date_range(start='2025', periods=n_months, freq='Y')

    # Plot appointment disruptions on primary y-axis
    line1 = ax1.plot(start_date, delayed_data, label="Delayed", linewidth=2, color='#FF8C00')
    line2 = ax1.plot(start_date, cancelled_data, label="Cancelled", linewidth=2, color='#DC143C')
    line3 = ax1.plot(start_date, total_data, label="Total Disrupted", linewidth=2, color='#4169E1', linestyle='--')

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax1.set_xlabel("Time Period", fontsize=12)
    ax1.set_ylabel("% Disrupted", fontsize=12, color='#4169E1')
    ax1.tick_params(axis='y', labelcolor='#4169E1')
    ax1.set_title(f"{scenario_name}", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle=':')

    # Create secondary y-axis for DALYs
    ax2 = ax1.twinx()
    line4 = ax2.plot(start_date, daly_data, label="DALYs", linewidth=2, color='#2E8B57', linestyle='-.')
    ax2.set_ylabel("DALYs", fontsize=12, color='#2E8B57')
    ax2.tick_params(axis='y', labelcolor='#2E8B57')

    # Combine legends from both axes
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best', fontsize=10)

# Hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig(output_folder / f"{PREFIX_ON_FILENAME}_delayed_cancelled_dalys.png", dpi=300, bbox_inches='tight')
plt.show()


# Now only mean for main text 

# In[194]:


ssps = ["ssp245"]
index_name = ["SSP 2.45 Mean"]
model_type = "mean"  # only the middle panel
service = "ANC"

for ssp_scenario in ssps:
    fig, ax = plt.subplots(figsize=(12, 6))  # single panel

    # Load weather data
    weather_data_prediction_monthly = pd.read_csv(
        f"/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL/{ssp_scenario}/{model_type}_monthly_prediction_weather_by_facility_{service}.csv",
        dtype={'column_name': 'float64'}
    )

    # Subset relevant months
    mask = (weather_data_prediction_monthly.index > 11) & \
           (weather_data_prediction_monthly.index < (17*12))
    weather_data_prediction_monthly = weather_data_prediction_monthly.loc[mask].reset_index(drop=True)

    # Average across facilities
    weather_data_avg = weather_data_prediction_monthly.iloc[:, 1:].mean(axis=1)

    # Yearly cumulative precipitation
    yearly_precip = weather_data_avg.groupby(weather_data_avg.index // 12).sum()

    # Load appointment disruption data
    scenario_name = "SSP 2.45 Mean"    
    delayed_data = np.array(appointment_delayed_scenarios[scenario_name], dtype=float) * 100
    cancelled_data = np.array(appointment_cancelled_scenarios[scenario_name], dtype=float) * 100
    total_data = delayed_data + cancelled_data

    n_years = len(delayed_data)
    start_date = pd.date_range(start='2025-01', periods=n_years, freq='Y')

    # Plot precipitation on primary y-axis
    color_precip = '#1C6E8C'
    ax.plot(start_date, yearly_precip, label='Precipitation', color=color_precip,
            linewidth=2, linestyle='--')
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative Precipitation (mm)", color=color_precip)
    ax.tick_params(axis='y', labelcolor=color_precip)
    ax.grid(False)

    # Secondary y-axis for appointment disruptions
    ax2 = ax.twinx()
    ax2.plot(start_date, delayed_data, label="Delayed", linewidth=2, color='#FEB95F')
    ax2.plot(start_date, cancelled_data, label="Cancelled", linewidth=2, color='#f07167')
    ax2.plot(start_date, total_data, label="Total Disrupted", linewidth=2, color='#5A716A')
    ax2.set_ylabel("Appointment Disruption (%)", rotation = -90, labelpad=25)
    ax2.tick_params(axis='y')

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=False)
    ax.text(-0.0, 1.05, '(C)', transform=ax.transAxes,
                   fontsize=14, va='top', ha='right')
    plt.tight_layout()
    plt.show()


# Add in months

# In[208]:


import matplotlib.dates as mdates
from datetime import datetime

# Storage dictionaries
all_scenarios_appointment_delayed_mean = {}
all_scenarios_appointment_cancelled_mean = {}
all_scenarios_dalys_mean = {}

# Define the extraction functions (add these near your other helper functions)
def get_num_treatments_total_delayed(_df):
    """Count total number of delayed HSI events"""
    _df["date"] = pd.to_datetime(_df["date"])
    _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]
    return pd.Series(len(_df), name="total")

def get_num_treatments_total_cancelled(_df):
    """Count total number of cancelled HSI events"""
    _df["date"] = pd.to_datetime(_df["date"])
    _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]
    return pd.Series(len(_df), name="total")

def get_num_treatments_total(_df):
    """Sum all treatment counts"""
    _df["date"] = pd.to_datetime(_df["date"])
    _df = _df.loc[_df["date"].between(*TARGET_PERIOD)]
    total = {}
    for d in _df["hsi_event_key_to_counts"]:
        for k, v in d.items():
            total[k] = total.get(k, 0) + v
    return pd.Series(sum(total.values()), name="total")

# Define month sequence (example for 2025)
# Adjust start_year, end_year, and months as needed
start_year = 2025
end_year = 2040
months = range(1, 13)  # January to December

# Create list of (year, month) tuples
month_sequence = [(year, month) for year in range(start_year, end_year + 1) for month in months]

# Main loop
for draw in range(len(scenario_names)):
    all_months_data_delayed_mean = {}
    all_months_data_cancelled_mean = {}
    all_months_dalys_mean = {}

    for year, month in month_sequence:
        # Determine the start and end dates for the month
        from calendar import monthrange
        last_day = monthrange(year, month)[1]
        TARGET_PERIOD = (Date(year, month, 1), Date(year, month, last_day))

        # Create a key for this month (e.g., "2025-01" for January 2025)
        month_key = f"{year}-{month:02d}"

        # Get DALYs
        num_dalys = summarize(extract_results(
            results_folder,
            module='tlo.methods.healthburden',
            key='dalys_stacked',
            custom_generate_series=get_num_dalys_by_cause_label,
            do_scaling=True
        ), only_mean=False, collapse_columns=True)[draw]

        total_population = summarize(
            extract_results(
                results_folder,
                module="tlo.methods.demography",
                key="population",
                custom_generate_series=get_population_for_year,
                do_scaling=True,
            ),
            only_mean=True,
            collapse_columns=True,
        )[draw]

        all_months_dalys_mean[month_key] = pd.Series(
            [num_dalys['mean'].sum() / total_population['mean']], 
            name='mean'
        )

        if scenario_names[draw] == 'Baseline':
            # Baseline has no weather disruptions
            all_months_data_delayed_mean[month_key] = pd.Series([0], name='mean')
            all_months_data_cancelled_mean[month_key] = pd.Series([0], name='mean')
        elif main_text:
            # For main_text mode: no [draw] indexing
            num_delayed = summarize(extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='Weather_delayed_HSI_Event_full_info',
                custom_generate_series=get_num_treatments_total_delayed,
                do_scaling=True
            ), only_mean=False, collapse_columns=True)

            num_cancelled = summarize(extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='Weather_cancelled_HSI_Event_full_info',
                custom_generate_series=get_num_treatments_total_cancelled,
                do_scaling=True
            ), only_mean=False, collapse_columns=True)

            num_total = summarize(extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='hsi_event_counts',
                custom_generate_series=get_num_treatments_total,
                do_scaling=True
            ), only_mean=False, collapse_columns=True)[draw]

            # Extract values from Series
            delayed_val = num_delayed['mean'].values[0] if len(num_delayed['mean'].values) > 0 else 0
            cancelled_val = num_cancelled['mean'].values[0] if len(num_cancelled['mean'].values) > 0 else 0
            total_val = num_total['mean'].values[0] if len(num_total['mean'].values) > 0 else 1

            all_months_data_delayed_mean[month_key] = pd.Series(
                [delayed_val / total_val], 
                name='mean'
            )
            all_months_data_cancelled_mean[month_key] = pd.Series(
                [cancelled_val / total_val], 
                name='mean'
            )
        else:
            # For other modes: with [draw] indexing
            num_delayed = summarize(extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='Weather_delayed_HSI_Event_full_info',
                custom_generate_series=get_num_treatments_total_delayed,
                do_scaling=True
            ), only_mean=False, collapse_columns=True)[draw]

            num_cancelled = summarize(extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='Weather_cancelled_HSI_Event_full_info',
                custom_generate_series=get_num_treatments_total_cancelled,
                do_scaling=True
            ), only_mean=False, collapse_columns=True)[draw]

            num_total = summarize(extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='hsi_event_counts',
                custom_generate_series=get_num_treatments_total,
                do_scaling=True
            ), only_mean=False, collapse_columns=True)[draw]

            # Extract values from Series
            delayed_val = num_delayed['mean'].values[0] if len(num_delayed['mean'].values) > 0 else 0
            cancelled_val = num_cancelled['mean'].values[0] if len(num_cancelled['mean'].values) > 0 else 0
            total_val = num_total['mean'].values[0] if len(num_total['mean'].values) > 0 else 1

            all_months_data_delayed_mean[month_key] = pd.Series(
                [delayed_val / total_val], 
                name='mean'
            )
            all_months_data_cancelled_mean[month_key] = pd.Series(
                [cancelled_val / total_val], 
                name='mean'
            )

    all_scenarios_appointment_delayed_mean[scenario_names[draw]] = all_months_data_delayed_mean
    all_scenarios_appointment_cancelled_mean[scenario_names[draw]] = all_months_data_cancelled_mean
    all_scenarios_dalys_mean[scenario_names[draw]] = all_months_dalys_mean


# In[211]:


start_date


# In[222]:


ssps = ["ssp245"]
index_name = ["SSP 2.45 Mean"]
model_type = "mean"  # only the middle panel
service = "ANC"

for ssp_scenario in ssps:
    fig, ax = plt.subplots(figsize=(12, 6))  # single panel

    # Load weather data
    weather_data_prediction_monthly = pd.read_csv(
        f"/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL/{ssp_scenario}/{model_type}_monthly_prediction_weather_by_facility_{service}.csv",
        dtype={'column_name': 'float64'}
    )

    # Subset relevant months (12 months of 2024 through first months of 2029, or adjust as needed)
    mask = (weather_data_prediction_monthly.index > 11) & \
           (weather_data_prediction_monthly.index < (17*12))
    weather_data_prediction_monthly = weather_data_prediction_monthly.loc[mask].reset_index(drop=True)

    # Average across facilities (monthly)
    weather_data_avg = weather_data_prediction_monthly.iloc[:, 1:].mean(axis=1)

    # Load appointment disruption data
    scenario_name = "SSP 2.45 Mean"    

    # Get the monthly data dictionaries
    delayed_monthly = all_scenarios_appointment_delayed_mean[scenario_name]
    cancelled_monthly = all_scenarios_appointment_cancelled_mean[scenario_name]

    # Convert to arrays and multiply by 100 for percentage
    # Sort by month key to ensure correct order
    month_keys = sorted(delayed_monthly.keys())
    delayed_data = np.array([delayed_monthly[key].values[0] for key in month_keys]) * 100
    cancelled_data = np.array([cancelled_monthly[key].values[0] for key in month_keys]) * 100
    total_data = delayed_data + cancelled_data

    # Create date range for x-axis
    n_months = len(delayed_data)
    start_date = pd.date_range(start='2025-01', periods=n_months, freq='M')

    # Ensure weather data matches the length of appointment data
    if len(weather_data_avg) != n_months:
        print(f"Warning: Weather data length ({len(weather_data_avg)}) doesn't match appointment data ({n_months})")
        # Truncate or pad as needed
        min_len = min(len(weather_data_avg), n_months)
        weather_data_avg = weather_data_avg[:min_len]
        delayed_data = delayed_data[:min_len]
        cancelled_data = cancelled_data[:min_len]
        total_data = total_data[:min_len]
        start_date = start_date[:min_len]

    # Plot precipitation on primary y-axis
    color_precip = '#9AC4F8'
    ax.plot(start_date, weather_data_avg, label='Monthly Precipitation', color=color_precip,
            linewidth=2, linestyle='--')
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly Precipitation (mm)", color=color_precip)
    ax.tick_params(axis='y', labelcolor=color_precip)
    ax.grid(False)

    ax.set_xlim(left=datetime(2025, 1, 1))

    # Or if you want to set both start and end:
    # ax.set_xlim(datetime(2025, 1, 1), datetime(2030, 12, 31))

    # Format the x-axis date labels
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    # Secondary y-axis for appointment disruptions
    ax2 = ax.twinx()
    ax2.plot(start_date, delayed_data, label="Delayed", linewidth=2, color='#FEB95F')
    ax2.plot(start_date, cancelled_data, label="Cancelled", linewidth=2, color='#f07167')
    ax2.plot(start_date, total_data, label="Total Disrupted", linewidth=2, color='#5A716A')
    ax2.set_ylabel("Appointment Disruption (%)", rotation=-90, labelpad=25)
    ax2.tick_params(axis='y')

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=False)
    ax.text(-0.0, 1.05, '(C)', transform=ax.transAxes,
            fontsize=14, va='top', ha='right')
    plt.tight_layout()
    plt.show()


# Get predicted disruptions from linear model

# In[223]:


climate_ssps = ["ssp126", "ssp245", "ssp585"]
climate_model_ensemble_models = ["lowest", "mean", "highest"]

climate_all_scenarios = {}
climate_summary_stats = {}

for ssp in climate_ssps:
    for model in climate_model_ensemble_models:
        scenario_key = f"{ssp}_{model}"

        df = pd.read_csv(
            f'/Users/rem76/PycharmProjects/TLOmodel/resources/climate_change_impacts/'
            f'ResourceFile_Precipitation_Disruptions_{ssp}_{model}.csv'
        )
        df = df[(df["year"] >= 2025) & (df["year"] <= 2041)]

        # average per year-month
        avg_df = df.groupby(["year", "month"], as_index=False)["mean_all_service"].mean()
        values = avg_df["mean_all_service"].values.tolist()

        # store full time series
        climate_all_scenarios[scenario_key] = values

        # compute summary statistics
        climate_summary_stats[scenario_key] = {
            "min": float(avg_df["mean_all_service"].min()) ,
            "max": float(avg_df["mean_all_service"].max()) ,
            "mean": float(avg_df["mean_all_service"].mean() )
        }



# Plot predicted vs modelled disruptions

# In[224]:


appointment_all_scenarios = {}
for scenario_name in scenario_names:
    if scenario_name == "Baseline":
        continue
    appointment_by_scenario = all_scenarios_appointment_difference_mean[scenario_name]
    for year in appointment_by_scenario.keys():
            appointment_all_years = [
        value
        for year in appointment_by_scenario.keys()
        for value in appointment_by_scenario[year].values.tolist()
    ]

    appointment_all_scenarios[scenario_name] = appointment_all_years


scenario_names = [name for name in appointment_all_scenarios.keys() if name != "baseline"]

fig, axes = plt.subplots(3, 3, figsize=(20, 12))
axes = axes.flatten()  # Flatten to easily index

for i, scenario_name in enumerate(scenario_names):
    ax = axes[i]
    ax.plot(np.array(appointment_all_scenarios[scenario_name], dtype=float) * 0.6, label="TLO")
    ax.plot(climate_all_scenarios[scenario_name], label="Predicted")

    ax.set_xlabel("Year")
    ax.set_ylabel("Value")
    ax.set_title(f"{scenario_name}")
    ax.grid(True)
    ax.legend()


# In[186]:


import matplotlib.dates as mdates
appointment_all_scenarios = {}
for scenario_name in scenario_names:
    if scenario_name == "Baseline":
        continue
    # Filter for only SSP 2.45 scenarios
    if "ssp245" not in scenario_name:
        continue

    appointment_by_scenario = all_scenarios_appointment_difference_mean[scenario_name]
    # Collect all values across all years
    appointment_all_years = [
        value
        for year in appointment_by_scenario.keys()
        for value in appointment_by_scenario[year].values.tolist()
    ]

    appointment_all_scenarios[scenario_name] = appointment_all_years

# Filter scenario names for SSP 2.45
scenario_names_filtered = [name for name in appointment_all_scenarios.keys() if "ssp245" in name]

# Adjust subplot grid based on number of SSP 2.45 scenarios
n_scenarios = len(scenario_names_filtered)
n_cols = min(3, n_scenarios)
n_rows = (n_scenarios + n_cols - 1) // n_cols  

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
if n_scenarios == 1:
    axes = [axes]
else:
    axes = axes.flatten() if n_scenarios > 1 else [axes]

for i, scenario_name in enumerate(scenario_names_filtered):
    ax = axes[i]
    # Convert to % by multiplying by 100
    data_points = np.array(appointment_all_scenarios[scenario_name], dtype=float) * 100

    # Create month labels starting from Jan 2012
    n_months = len(data_points)
    start_date = pd.date_range(start='2025-01', periods=n_months, freq='MS')

    ax.plot(start_date, data_points, label="TLO % Delayed")

    # Format x-axis to show dates nicely
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax.set_xlabel("Time Period")
    ax.set_ylabel("% Disrupted")
    ax.set_title(f"{scenario_name}")
    ax.grid(False)

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()


# In[ ]:


 ssps = ["ssp126", "ssp245", "ssp585"]
models = ["lowest", "mean", "highest"]
service = "ANC"

for ssp_scenario in ssps:
    plt.figure(figsize=(12, 6))

    for model_type in models:
        weather_data_prediction_monthly_original = pd.read_csv(
            f"/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL/{ssp_scenario}/{model_type}_monthly_prediction_weather_by_facility_{service}.csv",
            dtype={'column_name': 'float64'}
        )

        mask = (weather_data_prediction_monthly_original.index > 23) & \
               (weather_data_prediction_monthly_original.index < (12*18))
        weather_data_prediction_monthly_original = weather_data_prediction_monthly_original.loc[mask].reset_index(drop=True)

        weather_data_prediction_monthly_average_facilities = weather_data_prediction_monthly_original.iloc[:, 1:].mean(axis=1)
        plt.plot(weather_data_prediction_monthly_average_facilities, label=model_type)

    plt.xticks(ticks=range(0,int(len(weather_data_prediction_monthly_original)), 12), labels=range(2026, 2026 + int(len(weather_data_prediction_monthly_original)/12), 1))
    plt.title(f"Predicted Monthly Weather by Facility – {ssp_scenario}")
    plt.xlabel("Month")
    plt.ylabel("Predicted Weather")
    plt.legend()
    plt.show()

