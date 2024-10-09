"""
* Check key outputs for reporting in the calibration table of the write-up
* Produce representative plots for the default parameters

NB. To see larger effects
* Increase incidence of cancer (see tests)
* Increase symptom onset
* Increase progression rates (see tests)
"""

import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import math
from tlo import Simulation, logging, Date

from tlo.analysis.utils import make_age_grp_types, parse_log_file
from tlo.methods import (
    cervical_cancer,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
    epi,
    tb,
    hiv
)

seed = 100

log_config = {
    "filename": "cervical_cancer_analysis",   # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.cervical_cancer": logging.INFO,
        "tlo.methods.healthsystem": logging.INFO,
    }
}


start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
pop_size = 17

# This creates the Simulation instance for this run. Because we've passed the `seed` and
# `log_config` arguments, these will override the default behaviour.
sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

# Path to the resource files used by the disease and intervention methods
# resources = "./resources"
resourcefilepath = Path('./resources')

# Used to configure health system behaviour
service_availability = ["*"]

# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             cervical_cancer.CervicalCancer(resourcefilepath=resourcefilepath),
#                cc_test.CervicalCancer(resourcefilepath=resourcefilepath),
             simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
             enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
             healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                       disable=False,
                                       cons_availability='all'),
             symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
             healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
             healthburden.HealthBurden(resourcefilepath=resourcefilepath),
             epi.Epi(resourcefilepath=resourcefilepath),
             tb.Tb(resourcefilepath=resourcefilepath, run_with_checks=False),
             hiv.Hiv(resourcefilepath=resourcefilepath, run_with_checks=False)
             )

# create and run the simulation
sim.make_initial_population(n=pop_size)
sim.simulate(end_date=end_date)


# parse the simulation logfile to get the output dataframes
log_df = parse_log_file(sim.log_filepath)

start_year=2011
scale_factor = 1000


# Function to plot data
def plot_data(log_df, year_col, columns, prefix = '',scale_factor=1000, start_year=2011, title="", xlabel="Year", ylabel="", ylim=None, proportion_plot=False):
    # Filter by year and ensure only valid values
    log_df_plot = log_df["tlo.methods.cervical_cancer"]["all"]
    log_df_plot = log_df_plot[[year_col] + columns].dropna()
    log_df_plot = log_df_plot[log_df_plot[year_col] >= start_year]


    # If proportion plot is True, calculate proportions
    if proportion_plot:
        total_col = log_df_plot[columns].sum(axis=1)  # Sum across the columns to get the total for each row
        for col in columns:
            new_col_name = col.replace(prefix, '')  # Remove the prefix
            log_df_plot[f'proportion_{new_col_name}'] = log_df_plot[col] / total_col  # Calculate proportion

            # Update columns to use proportion columns and remove those containing 'none'
        columns = [f'proportion_{col.replace(prefix, "")}' for col in columns if 'none' not in col]

    # Scale values
    if not proportion_plot:
        for col in columns:
            log_df_plot[col] = log_df_plot[col] * scale_factor

    # Plotting logic
    plt.figure(figsize=(10, 6))

    if proportion_plot:
        bottom = 0
        for col in columns:
            plt.fill_between(log_df_plot[year_col], bottom, bottom + log_df_plot[col], label=col, alpha=0.7)
            bottom += log_df_plot[col]
        plt.legend(loc='upper right')
    else:
        plt.plot(log_df_plot[year_col], log_df_plot[columns[0]], marker='o')

    # Plot
    plt.style.use("ggplot")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    # Set y-axis limits if provided
    if ylim:
        plt.ylim(ylim)

    plt.show()

# Execute functions

# 1. Total deaths by Year
plot_data(log_df, year_col='rounded_decimal_year', columns=['n_deaths_past_year'], scale_factor=scale_factor, title='Total deaths by Year', ylabel='Total deaths past year', ylim=(0, 10000))

# 2. Total deaths cervical cancer in HIV negative by Year
plot_data(log_df, year_col='rounded_decimal_year', columns=['n_deaths_cc_hivneg_past_year'], scale_factor=scale_factor, title='Total deaths cervical cancer in HIV negative by Year', ylabel='Total deaths in HIV negative past year', ylim=(0, 10000))

# 3. Total deaths cervical cancer in HIV positive by Year
plot_data(log_df, year_col='rounded_decimal_year', columns=['n_deaths_cc_hivpos_past_year'], scale_factor=scale_factor, title='Total deaths cervical cancer in HIV positive by Year', ylabel='Total deaths in HIV positive past year', ylim=(0, 10000))

# 4. Total diagnosed per Year
plot_data(log_df, year_col='rounded_decimal_year', columns=['n_diagnosed_past_year'], scale_factor=scale_factor, title='Total diagnosed per Year', ylabel='Total diagnosed per year', ylim=(0, 10000))

# 5. Total treated per Year
plot_data(log_df, year_col='rounded_decimal_year', columns=['n_treated_past_year'], scale_factor=scale_factor, title='Total treated per Year', ylabel='Total treated per year', ylim=(0, 10000))

# 6. Total cured per Year
plot_data(log_df, year_col='rounded_decimal_year', columns=['n_cured_past_year'], scale_factor=scale_factor, title='Total cured per Year', ylabel='Total cured per year', ylim=(0, 10000))

# 7. Proportion of women aged 15+ with HPV, CIN, cervical cancer
plot_data(log_df, year_col='rounded_decimal_year', columns=['total_none', 'total_hpv', 'total_cin1', 'total_cin2', 'total_cin3', 'total_stage1',
                        'total_stage2a', 'total_stage2b', 'total_stage3', 'total_stage4'], prefix = 'total_',scale_factor=scale_factor, title='Proportion of women aged 15+ with HPV, CIN, cervical cancer', ylabel='Proportion', ylim=(0, 0.30), proportion_plot=True)

# 8. Proportion of people with cervical cancer who are HIV positive
plot_data(log_df, year_col='rounded_decimal_year', columns=['prop_cc_hiv'], title='Proportion of people with cervical cancer who are HIV positive', ylabel='Proportion', ylim=(0, 1))

# 9. Number of women living with unsuppressed HIV
plot_data(log_df, year_col='rounded_decimal_year', columns=['n_women_hiv_unsuppressed'], scale_factor=scale_factor, title='Number of women living with unsuppressed HIV', ylabel='n_women_hiv_unsuppressed', ylim=(0, 300000))

# 10. Proportion of HIV negative women aged 15+ with HPV, CIN, cervical cancer
plot_data(log_df, year_col='rounded_decimal_year', columns=['total_hivneg_none', 'total_hivneg_hpv', 'total_hivneg_cin1', 'total_hivneg_cin2', 'total_hivneg_cin3',
                        'total_hivneg_stage1','total_hivneg_stage2a', 'total_hivneg_stage2b', 'total_hivneg_stage3', 'total_hivneg_stage4'], prefix = 'total_',title='Proportion of HIV negative women aged 15+ with HPV, CIN, cervical cancer', ylabel='Proportion', ylim=(0, 0.30), proportion_plot=True)

# 11. Proportion of HIV positive women aged 15+ with HPV, CIN, cervical cancer
plot_data(log_df, year_col='rounded_decimal_year', columns=['total_hivpos_none', 'total_hivpos_hpv', 'total_hivpos_cin1', 'total_hivpos_cin2', 'total_hivpos_cin3',
                        'total_hivpos_stage1','total_hivpos_stage2a', 'total_hivpos_stage2b', 'total_hivpos_stage3', 'total_hivpos_stage4'], prefix = 'total_', title='Proportion of HIV positive women aged 15+ with HPV, CIN, cervical cancer', ylabel='Proportion', ylim=(0, 0.30), proportion_plot=True)

# 12. Number of HIV positive women in Stage 4
plot_data(log_df, year_col='rounded_decimal_year', columns=['total_hivpos_stage4'], scale_factor=scale_factor, title='Number of HIV positive women in Stage 4', ylabel='total_hivpos_stage4', ylim=(0, 100))

# 13. Number of HIV negative women in Stage 4
plot_data(log_df, year_col='rounded_decimal_year', columns=['total_hivneg_stage4'], scale_factor=scale_factor, title='Number of HIV negative women in Stage 4', ylabel='total_hivneg_stage4', ylim=(0, 100))
