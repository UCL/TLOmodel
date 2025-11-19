
"""
plot the calibration figures
"""

from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import pandas as pd
# import lacroix
import matplotlib.colors as colors
import numpy as np
import statsmodels.api as sm
import seaborn as sns
from collections import defaultdict
import textwrap
from typing import Tuple, Union

from scipy.stats import norm

from tlo import Date, Simulation, logging
from tlo.analysis.utils import (
    format_gbd,
    make_age_grp_types,
    parse_log_file,
    compare_number_of_deaths,
    extract_params,
    compute_summary_statistics,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    compute_summary_statistics,
    make_age_grp_lookup,
    make_age_grp_types,
    unflatten_flattened_multi_index_in_logging,
)



resourcefilepath = Path("./resources")

output_folder = Path("./outputs/t.mangal@imperial.ac.uk")
results_folder = get_scenario_outputs("schisto_scenarios-2025.py", output_folder)[-1]

TARGET_PERIOD = (Date(2024, 1, 1), Date(2050, 12, 31))


def target_period() -> str:
    """Returns the target period as a string of the form YYYY-YYYY"""
    return "-".join(str(t.year) for t in TARGET_PERIOD)



# ------------------------------------------------
# 2022 Prevalence Model vs Data
# ------------------------------------------------

# file_path = results_folder / f"prev_haem_HML_All_district_summary {target_period()}.xlsx"
# model_haem = pd.read_excel(file_path)
#
# file_path = results_folder / f"prev_mansoni_HML_All_district_summary {target_period()}.xlsx"
# model_mansoni = pd.read_excel(file_path)


file_path = results_folder / f"prev_mansoni_HML_by_age_district_summary {target_period()}.xlsx"
model_haem = pd.read_excel(file_path)

file_path = results_folder / f"prev_mansoni_HML_by_age_district_summary {target_period()}.xlsx"
model_mansoni = pd.read_excel(file_path)



data_haem = pd.read_csv(resourcefilepath / "ResourceFile_Schisto" / "LatestData_haematobium.csv")
data_mansoni = pd.read_csv(resourcefilepath / "ResourceFile_Schisto" / "LatestData_mansoni.csv")



# select 2019
# select Continue WASH, MDA SAC as baseline - should all be the same at 2019
model_mansoni = model_mansoni.loc[model_mansoni.draw.isin(["Continue WASH, MDA SAC"])].copy()
model_haem = model_haem.loc[model_haem.draw.isin(["Continue WASH, MDA SAC"])].copy()

model_mansoni = model_mansoni.loc[model_mansoni.age_years.isin(["SAC"])].copy()
model_haem = model_haem.loc[model_haem.age_years.isin(["SAC"])].copy()

model_mansoni_2019 = model_mansoni.loc[model_mansoni.year == 2019].copy()
model_haem_2019 = model_haem.loc[model_haem.year == 2019].copy()

model_mansoni_2022 = model_mansoni.loc[model_mansoni.year == 2022].copy()
model_haem_2022 = model_haem.loc[model_haem.year == 2022].copy()



def plot_species_panel(ax,
                       df_data,
                       df_model,
                       title,
                       district_col_data='District',
                       district_col_model='district'):
    """
    df_data columns:
        [district_col_data, 'min_prevalence', 'max_prevalence', 'Endemicity2022']
        min/max already on 0–100 scale.

    df_model columns:
        [district_col_model, 'mean', 'lower_ci', 'upper_ci']
        (model values on 0–1 scale)
    """

    # Standardise naming
    df_d = (df_data[[district_col_data, 'min_prevalence', 'max_prevalence', 'Endemicity2022']]
            .rename(columns={district_col_data: 'District'}))

    df_m = (df_model[[district_col_model, 'mean', 'lower_ci', 'upper_ci']]
            .rename(columns={district_col_model: 'District'}))

    df = df_d.merge(df_m, on='District', how='inner')

    # Endemicity bands
    band_map = {
        'Low prevalence (0%-9%)': (0, 100),
        'Moderate prevalence (10%-49%)': (0, 100),
        'High prevalence (50% and above)': (0, 100),
    }

    numeric_colour = 'lightblue'
    band_colour = 'lightgrey'

    districts = df['District'].tolist()
    x = range(len(districts))

    # Draw bars
    for i, row in df.iterrows():

        # Case 1 — both 0: check endemicity band
        if (row['min_prevalence'] == 0) and (row['max_prevalence'] == 0):

            band = row['Endemicity2022']

            # If endemicity is literally 0 → skip entirely
            if band == 0:
                continue

            # Use endemicity band range
            low, high = band_map.get(band, (0, 0))
            bar_colour = band_colour

        else:
            # Use numeric min/max
            low = row['min_prevalence']
            high = row['max_prevalence']
            bar_colour = numeric_colour

        height = max(high - low, 0)

        if height > 0:
            ax.bar(i,
                   height=height,
                   bottom=low,
                   width=0.8,
                   color=bar_colour,
                   alpha=0.5,
                   edgecolor='none')

    # Model estimates (convert to %)
    mean_pct = df['mean'].values * 100
    lower_pct = df['lower_ci'].values * 100
    upper_pct = df['upper_ci'].values * 100

    yerr = [mean_pct - lower_pct, upper_pct - mean_pct]

    err_plot = ax.errorbar(
        x,
        mean_pct,
        yerr=yerr,
        fmt='x',
        color='red',
        capsize=3,
        label='Model prevalence'
    )

    # Axis formatting
    ax.set_ylim(0, 100)
    ax.set_ylabel('Prevalence (%)')
    ax.set_title(title)

    ax.set_xticks(list(x))
    ax.set_xticklabels(districts, rotation=90, ha='right')

    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.xaxis.grid(False)

    # Legend
    patch_numeric = Patch(facecolor=numeric_colour, alpha=0.5,
                          label='Survey min–max range')
    patch_band = Patch(facecolor=band_colour, alpha=0.5,
                       label='Survey endemicity band')

    ax.legend(handles=[err_plot[0], patch_numeric, patch_band],
              loc='upper left')



# PLOT
fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

plot_species_panel(ax_top, data_haem, model_haem_2022,
                   'Haematobium - District-wise Prevalence Comparison (2022)',
                   district_col_data='District',
                   district_col_model='district')

plot_species_panel(ax_bottom, data_mansoni, model_mansoni_2022,
                   'Mansoni - District-wise Prevalence Comparison (2022)',
                   district_col_data='District',
                   district_col_model='district')

fig.tight_layout()
plt.show()







############################################################################################




plt.rcParams['figure.dpi'] = 120

def prepare_calibration_df(
    df_data,
    df_model,
    district_col_data='District',
    min_col='min_prevalence',
    max_col='max_prevalence',
    endemic_col='Endemicity2022',
    district_col_model='district',
    model_col='mean',
    model_is_proportion=True,
):
    """
    df_data: at least [district_col_data, min_col, max_col, endemic_col]
    df_model: at least [district_col_model, model_col]

    Returns df with:
      ['district','data_min','data_max','data_mid','model',
       'inside_range','distance_to_range','data_status']
    where data_status ∈ {'range', 'zero', 'missing'}.
    """

    d = df_data[[district_col_data, min_col, max_col, endemic_col]].copy()
    d = d.rename(columns={
        district_col_data: 'district',
        min_col: 'data_min',
        max_col: 'data_max',
        endemic_col: 'Endemicity2022',
    })

    m = df_model[[district_col_model, model_col]].copy()
    m = m.rename(columns={
        district_col_model: 'district',
        model_col: 'model',
    })

    if model_is_proportion:
        m['model'] = m['model'] * 100.0

    df = d.merge(m, on='district', how='inner')

    # --- classification of data status ---

    # base masks
    zero_range = (df['data_min'] == 0) & (df['data_max'] == 0)
    zero_label = df['Endemicity2022'] == 'Zero'   # exactly the string 'Zero'

    df['data_status'] = 'range'   # default

    # true survey zeros
    df.loc[zero_range & zero_label, 'data_status'] = 'zero'

    # missing numeric data but endemicity info present/other (incl. NaN)
    df.loc[zero_range & ~zero_label, 'data_status'] = 'missing'

    # for missing, wipe numeric range so it does not count as calibration data
    df.loc[df['data_status'] == 'missing', ['data_min', 'data_max']] = np.nan

    # midpoint (only meaningful for 'range')
    df['data_mid'] = (df['data_min'] + df['data_max']) / 2.0

    # inside/outside only defined where we have a numeric range or zero
    df['inside_range'] = (df['model'] >= df['data_min']) & (df['model'] <= df['data_max'])

    # distance to nearest boundary (NaN if missing)
    below = df['data_min'] - df['model']
    above = df['model'] - df['data_max']
    df['distance_to_range'] = np.where(
        df['model'] < df['data_min'], below,
        np.where(df['model'] > df['data_max'], above, 0.0)
    )
    df.loc[df['data_status'] == 'missing', 'distance_to_range'] = np.nan

    df = df.sort_values(['data_status', 'data_mid'], na_position='last').reset_index(drop=True)

    return df





def calibration_summary(df, label=''):
    """
    Summarise fit **only** for districts with real numerical data
    (status 'range' or 'zero'). Districts with data_status=='missing'
    are reported but excluded from the metrics.
    """

    df_valid = df[df['data_status'] != 'missing'].copy()
    n_all = len(df)
    n_valid = len(df_valid)
    n_missing = n_all - n_valid

    within = df_valid['inside_range'].sum()
    above = (df_valid['distance_to_range'] > 0).sum()
    below = (df_valid['distance_to_range'] < 0).sum()
    dist_abs = df_valid['distance_to_range'].abs()

    print(f"\nCalibration summary {label}".strip())
    print("-" * 50)
    print(f"Total districts:                       {n_all}")
    print(f"  with numerical data:                 {n_valid}")
    print(f"  with missing data (endemicity only): {n_missing}")
    if n_valid > 0:
        print(f"Districts within empirical min–max:    {within}/{n_valid} ({100*within/n_valid:.1f}%)")
        print(f"Above empirical max:                   {above}/{n_valid} ({100*above/n_valid:.1f}%)")
        print(f"Below empirical min:                   {below}/{n_valid} ({100*below/n_valid:.1f}%)")
        print(f"Median distance to range:              {dist_abs.median():.2f} pp")
        print(f"90th percentile distance:              {dist_abs.quantile(0.9):.2f} pp")
        print(f"Maximum distance:                      {dist_abs.max():.2f} pp")




def plot_coverage(df, title):
    """
    Coverage plot that distinguishes:
      - 'range'   : vertical min–max bar + model point
      - 'zero'    : horizontal tick at 0 + model point
      - 'missing' : no empirical bar; model shown in grey and
                    excluded from fit statistics
    """

    fig, ax = plt.subplots(figsize=(14, 4))
    x = np.arange(len(df))

    # Convert data_status to numpy masks to avoid index/broadcast quirks
    status = df['data_status'].to_numpy()
    mask_range = status == 'range'
    mask_zero = status == 'zero'
    mask_missing = status == 'missing'

    # 1. empirical ranges (only if any)
    if mask_range.any():
        ax.vlines(
            x[mask_range],
            df.loc[mask_range, 'data_min'],
            df.loc[mask_range, 'data_max'],
            color='lightgrey',
            linewidth=5,
            label='Data min–max range'
        )

    # 2. true zeros – short horizontal ticks at 0 (only if any)
    if mask_zero.any():
        x_zero = x[mask_zero]
        ax.hlines(
            0,
            x_zero - 0.3,
            x_zero + 0.3,
            color='black',
            linewidth=2,
            label='Observed zero prevalence'
        )

    # 3. model points
    # start all as grey (missing / endemicity only)
    colours = np.full(len(df), 'grey', dtype=object)

    # for districts with numerical data (range or zero), colour by inside/outside
    mask_valid = ~mask_missing
    inside = df['inside_range'].fillna(False).to_numpy()

    colours[mask_valid & inside] = 'tab:green'
    colours[mask_valid & ~inside] = 'tab:red'

    ax.scatter(x, df['model'], c=colours, zorder=3, label='Model')

    ax.set_ylim(0, 100)

    # axes formatting
    ax.set_xticks(x)
    ax.set_xticklabels(df['district'], rotation=90)
    ax.set_ylabel('Prevalence (%)')
    ax.set_title(title)

    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.xaxis.grid(False)

    # legend
    legend_elements = []

    if mask_range.any():
        legend_elements.append(
            Line2D([0], [0], color='lightgrey', lw=5, label='Data min–max range')
        )
    if mask_zero.any():
        legend_elements.append(
            Line2D([0], [0], color='black', lw=2, label='Observed zero prevalence')
        )

    legend_elements.extend([
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:green',
               markersize=6, label='Model inside range'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:red',
               markersize=6, label='Model outside range'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
               markersize=6, label='Model (no survey data)'),
    ])

    ax.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    plt.show()
    return fig, ax



def plot_ratio_to_range(df, title):
    """
    For each district:
      ratio_min = model / data_min
      ratio_max = model / data_max

    If model lies inside the range and data_min>0, ratio_min >= 1 and ratio_max <= 1.
    """

    df = df.copy()
    df['ratio_min'] = df['model'] / df['data_min'].replace(0, np.nan)
    df['ratio_max'] = df['model'] / df['data_max'].replace(0, np.nan)

    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.axhline(1.0, color='black', linestyle='--', alpha=0.7,
               label='Perfect agreement (ratio = 1)')

    ax.plot(x, df['ratio_min'], marker='o', linestyle='-', label='Model / data_min')
    ax.plot(x, df['ratio_max'], marker='o', linestyle='-', label='Model / data_max')

    ax.set_xticks(x)
    ax.set_xticklabels(df['district'], rotation=90)
    ax.set_ylabel('Ratio')
    ax.set_title(title + ' – Ratios to boundaries')

    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.xaxis.grid(False)

    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    return fig, ax



df_haem_calib = prepare_calibration_df(
    df_data=data_haem,
    df_model=model_haem_2019,
    district_col_data='District',
    min_col='min_prevalence',
    max_col='max_prevalence',
    endemic_col='Endemicity2022',
    district_col_model='district',
    model_col='mean',
    model_is_proportion=True,
)

calibration_summary(df_haem_calib, label='(Haematobium)')
plot_coverage(df_haem_calib, 'Prevalence S. Haematobium')


df_mansoni_calib = prepare_calibration_df(
    df_data=data_mansoni,
    df_model=model_mansoni_2019,
    district_col_data='District',
    min_col='min_prevalence',
    max_col='max_prevalence',
    endemic_col='Endemicity2022',
    district_col_model='district',
    model_col='mean',
    model_is_proportion=True,
)

calibration_summary(df_mansoni_calib, label='(Mansoni)')
plot_coverage(df_mansoni_calib, 'Prevalence - S. Mansoni')








