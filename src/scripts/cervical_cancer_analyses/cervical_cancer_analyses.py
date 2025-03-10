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
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    cervical_cancer,
    demography,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    simplified_births,
    symptommanager,
    tb,
)

seed = 7

# Date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

log_config = {
    "filename": "cervical_cancer_analysis",   # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.cervical_cancer": logging.INFO,
        "tlo.methods.healthsystem": logging.INFO,
    }
}

# Set parameters for the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2030, 1, 1)
malawi_country_pop = 17000000
popsize = 1700

def run_sim(service_availability):
    # Establish the simulation object and set the seed
    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)
#     sim = Simulation(start_date=start_date, log_config={"filename": "logfile"})

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

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    return sim

# Create df from simulation
sim  = run_sim(service_availability=['*'])
log_df = parse_log_file(sim.log_filepath)
log_df_plot = log_df["tlo.methods.cervical_cancer"]["all"]

# Create output csv file to support plot generation from csv file
output_csv_file = Path("outputs/output7_data.csv")
if output_csv_file.exists():
    output_csv_file.unlink()
else:
    output_csv_file.touch()
log_df_plot.to_csv(output_csv_file)

# Scale factor
scale_factor = malawi_country_pop / popsize

# ---------------------------------------------------------------------------------------------------------
#   PLOTTING FOR CALIBRATION AND RESULTS
# ---------------------------------------------------------------------------------------------------------
# plot number of cervical cancer deaths in past year
out_df = pd.read_csv(output_csv_file)
out_df = out_df[['n_deaths_past_year', 'rounded_decimal_year']].dropna()
out_df = out_df[out_df['rounded_decimal_year'] >= 2011]
out_df['n_deaths_past_year'] = out_df['n_deaths_past_year'] * scale_factor
print(out_df)
plt.figure(figsize=(10, 6))
plt.plot(out_df['rounded_decimal_year'], out_df['n_deaths_past_year'], marker='o')
plt.title('Total deaths by Year')
plt.xlabel('Year')
plt.ylabel('Total deaths past year')
plt.grid(True)
plt.ylim(0, 10000)
plt.show()


# plot number of cervical cancer deaths in hivneg in past year
out_df_6 = pd.read_csv(output_csv_file)
out_df_6 = out_df_6[['n_deaths_cc_hivneg_past_year', 'rounded_decimal_year']].dropna()
out_df_6 = out_df_6[out_df_6['rounded_decimal_year'] >= 2011]
out_df_6['n_deaths_cc_hivneg_past_year'] = out_df_6['n_deaths_cc_hivneg_past_year'] * scale_factor
print(out_df_6)
plt.figure(figsize=(10, 6))
plt.plot(out_df_6['rounded_decimal_year'], out_df_6['n_deaths_cc_hivneg_past_year'], marker='o')
plt.title('Total deaths cervical cancer in hivneg by Year')
plt.xlabel('Year')
plt.ylabel('Total deaths cervical cancer in hivneg past year')
plt.grid(True)
plt.ylim(0, 10000)
plt.show()


# plot number of cervical cancer deaths in hivpos in past year
out_df_9 = pd.read_csv(output_csv_file)
out_df_9 = out_df_9[['n_deaths_cc_hivpos_past_year', 'rounded_decimal_year']].dropna()
out_df_9 = out_df_9[out_df_9['rounded_decimal_year'] >= 2011]
out_df_9['n_deaths_cc_hivpos_past_year'] = out_df_9['n_deaths_cc_hivpos_past_year'] * scale_factor
print(out_df_9)
plt.figure(figsize=(10, 6))
plt.plot(out_df_9['rounded_decimal_year'], out_df_9['n_deaths_cc_hivpos_past_year'], marker='o')
plt.title('Total deaths cervical cancer in hivpos by Year')
plt.xlabel('Year')
plt.ylabel('Total deaths cervical cancer in hivpos past year')
plt.grid(True)
plt.ylim(0, 10000)
plt.show()


# plot number of cc diagnoses in past year
out_df_4 = pd.read_csv(output_csv_file)
out_df_4 = out_df_4[['n_diagnosed_past_year', 'rounded_decimal_year']].dropna()
out_df_4 = out_df_4[out_df_4['rounded_decimal_year'] >= 2011]
out_df_4['n_diagnosed_past_year'] = out_df_4['n_diagnosed_past_year'] * scale_factor
print(out_df_4)
plt.figure(figsize=(10, 6))
plt.plot(out_df_4['rounded_decimal_year'], out_df_4['n_diagnosed_past_year'], marker='o')
plt.title('Total diagnosed per Year')
plt.xlabel('Year')
plt.ylabel('Total diagnosed per year')
plt.grid(True)
plt.ylim(0,10000)
plt.show()




# plot number cc treated in past year
out_df_13 = pd.read_csv(output_csv_file)
out_df_13 = out_df_13[['n_treated_past_year', 'rounded_decimal_year']].dropna()
out_df_13 = out_df_13[out_df_13['rounded_decimal_year'] >= 2011]
out_df_13['n_treated_past_year'] = out_df_13['n_treated_past_year'] * scale_factor
print(out_df_13)
plt.figure(figsize=(10, 6))
plt.plot(out_df_13['rounded_decimal_year'], out_df_13['n_treated_past_year'], marker='o')
plt.title('Total treated per Year')
plt.xlabel('Year')
plt.ylabel('Total treated per year')
plt.grid(True)
plt.ylim(0,10000)
plt.show()




# plot number cc cured in past year
out_df_14 = pd.read_csv(output_csv_file)
out_df_14 = out_df_14[['n_cured_past_year', 'rounded_decimal_year']].dropna()
out_df_14 = out_df_14[out_df_14['rounded_decimal_year'] >= 2011]
out_df_14['n_cured_past_year'] = out_df_14['n_cured_past_year'] * scale_factor
print(out_df_14)
plt.figure(figsize=(10, 6))
plt.plot(out_df_14['rounded_decimal_year'], out_df_14['n_cured_past_year'], marker='o')
plt.title('Total cured per Year')
plt.xlabel('Year')
plt.ylabel('Total cured per year')
plt.grid(True)
plt.ylim(0,10000)
plt.show()




# plot prevalence of each ce stage
out_df_2 = pd.read_csv(output_csv_file)
columns_to_calculate = ['total_none', 'total_hpv', 'total_cin1', 'total_cin2', 'total_cin3', 'total_stage1',
                        'total_stage2a', 'total_stage2b', 'total_stage3', 'total_stage4']
for column in columns_to_calculate:
    new_column_name = column.replace('total_', '')
    out_df_2[f'proportion_{new_column_name}'] = out_df_2[column] / out_df_2[columns_to_calculate].sum(axis=1)
print(out_df_2)
columns_to_plot = ['proportion_hpv', 'proportion_cin1', 'proportion_cin2', 'proportion_cin3',
                   'proportion_stage1', 'proportion_stage2a', 'proportion_stage2b', 'proportion_stage3',
                   'proportion_stage4']
plt.figure(figsize=(10, 6))
# Initialize the bottom of the stack
bottom = 0
for column in columns_to_plot:
    plt.fill_between(out_df_2['rounded_decimal_year'],
                     bottom,
                     bottom + out_df_2[column],
                     label=column,
                     alpha=0.7)
    bottom += out_df_2[column]
# plt.plot(out_df_2['rounded_decimal_year'], out_df_2['proportion_cin1'], marker='o')
plt.title('Proportion of women aged 15+ with HPV, CIN, cervical cancer')
plt.xlabel('Year')
plt.ylabel('Proportion')
plt.grid(True)
plt.legend(loc='upper right')
plt.ylim(0, 0.30)
plt.show()



# Proportion of people with cervical cancer who are HIV positive
out_df_3 = pd.read_csv(output_csv_file)
out_df_3 = out_df_3[['prop_cc_hiv', 'rounded_decimal_year']].dropna()
plt.figure(figsize=(10, 6))
plt.plot(out_df_3['rounded_decimal_year'], out_df_3['prop_cc_hiv'], marker='o')
plt.title('Proportion of people with cervical cancer who are HIV positive')
plt.xlabel('Year')
plt.ylabel('Proportion')
plt.grid(True)
plt.ylim(0, 1)
plt.show()

# log_config = {
#     "filename": "cervical_cancer_analysis",   # The name of the output file (a timestamp will be appended).
#     "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
#     "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
#         "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
#         "tlo.methods.cervical_cancer": logging.INFO,
#         "tlo.methods.healthsystem": logging.INFO,
#     }
# }



# plot number of women living with unsuppressed HIV
out_df_4 = pd.read_csv(output_csv_file)
out_df_4 = out_df_4[['n_women_hiv_unsuppressed', 'rounded_decimal_year']].dropna()
out_df_4 = out_df_4[out_df_4['rounded_decimal_year'] >= 2011]
out_df_4['n_women_hiv_unsuppressed'] = out_df_4['n_women_hiv_unsuppressed'] * scale_factor
print(out_df_4)
plt.figure(figsize=(10, 6))
plt.plot(out_df_4['rounded_decimal_year'], out_df_4['n_women_hiv_unsuppressed'], marker='o')
plt.title('n_women_hiv_unsuppressed')
plt.xlabel('Year')
plt.ylabel('n_women_hiv_unsuppressed')
plt.grid(True)
plt.ylim(0, 300000)
plt.show()



# plot prevalence of each ce stage for hivneg
out_df_5 = pd.read_csv(output_csv_file)
columns_to_calculate = ['total_hivneg_none', 'total_hivneg_hpv', 'total_hivneg_cin1', 'total_hivneg_cin2', 'total_hivneg_cin3',
                        'total_hivneg_stage1','total_hivneg_stage2a', 'total_hivneg_stage2b', 'total_hivneg_stage3', 'total_hivneg_stage4']
for column in columns_to_calculate:
    new_column_name = column.replace('total_hivneg_', '')
    out_df_5[f'proportion_hivneg_{new_column_name}'] = out_df_5[column] / out_df_5[columns_to_calculate].sum(axis=1)
print(out_df_5)
columns_to_plot = ['proportion_hivneg_hpv', 'proportion_hivneg_cin1', 'proportion_hivneg_cin2', 'proportion_hivneg_cin3',
                   'proportion_hivneg_stage1', 'proportion_hivneg_stage2a', 'proportion_hivneg_stage2b', 'proportion_hivneg_stage3',
                   'proportion_hivneg_stage4']
plt.figure(figsize=(10, 6))
# Initialize the bottom of the stack
bottom = 0
for column in columns_to_plot:
    plt.fill_between(out_df_5['rounded_decimal_year'],
                     bottom,
                     bottom + out_df_5[column],
                     label=column,
                     alpha=0.7)
    bottom += out_df_5[column]
plt.title('Proportion of hivneg women aged 15+ with HPV, CIN, cervical cancer')
plt.xlabel('Year')
plt.ylabel('Proportion')
plt.grid(True)
plt.legend(loc='upper right')
plt.ylim(0, 0.30)
plt.show()



# plot prevalence of each ce stage for hivpos
out_df_8 = pd.read_csv(output_csv_file)
columns_to_calculate = ['total_hivpos_none', 'total_hivpos_hpv', 'total_hivpos_cin1', 'total_hivpos_cin2', 'total_hivpos_cin3',
                        'total_hivpos_stage1','total_hivpos_stage2a', 'total_hivpos_stage2b', 'total_hivpos_stage3', 'total_hivpos_stage4']
for column in columns_to_calculate:
    new_column_name = column.replace('total_hivpos_', '')
    out_df_8[f'proportion_hivpos_{new_column_name}'] = out_df_8[column] / out_df_8[columns_to_calculate].sum(axis=1)
print(out_df_8)
columns_to_plot = ['proportion_hivpos_hpv', 'proportion_hivpos_cin1', 'proportion_hivpos_cin2', 'proportion_hivpos_cin3',
                   'proportion_hivpos_stage1', 'proportion_hivpos_stage2a', 'proportion_hivpos_stage2b', 'proportion_hivpos_stage3',
                   'proportion_hivpos_stage4']
plt.figure(figsize=(10, 6))
# Initialize the bottom of the stack
bottom = 0
for column in columns_to_plot:
    plt.fill_between(out_df_8['rounded_decimal_year'],
                     bottom,
                     bottom + out_df_8[column],
                     label=column,
                     alpha=0.7)
    bottom += out_df_8[column]
plt.title('Proportion of hivpos women aged 15+ with HPV, CIN, cervical cancer')
plt.xlabel('Year')
plt.ylabel('Proportion')
plt.grid(True)
plt.legend(loc='upper right')
plt.ylim(0, 0.30)
plt.show()


# plot number of hivpos in stage 4
out_df_11 = pd.read_csv(output_csv_file)
out_df_11 = out_df_11[['total_hivpos_stage4', 'rounded_decimal_year']].dropna()
# out_df_11 = out_df_11[out_df_11['rounded_decimal_year'] >= 2011]
# out_df_11['total_hivpos_stage4'] = out_df_11['total_hivpos_stage4'] * scale_factor
print(out_df_11)
plt.figure(figsize=(10, 6))
plt.plot(out_df_11['rounded_decimal_year'], out_df_11['total_hivpos_stage4'], marker='o')
plt.title('total_hivpos_stage4')
plt.xlabel('Year')
plt.ylabel('total_hivpos_stage4')
plt.grid(True)
plt.ylim(0,100)
plt.show()


# plot number of hivneg in stage 4
out_df_7 = pd.read_csv(output_csv_file)
out_df_7 = out_df_7[['total_hivneg_stage4', 'rounded_decimal_year']].dropna()
# out_df_7 = out_df_7[out_df_7['rounded_decimal_year'] >= 2011]
# out_df_7['total_hivneg_stage4'] = out_df_7['total_hivneg_stage4'] * scale_factor
print(out_df_7)
plt.figure(figsize=(10, 6))
plt.plot(out_df_7['rounded_decimal_year'], out_df_7['total_hivneg_stage4'], marker='o')
plt.title('total_hivneg_stage4')
plt.xlabel('Year')
plt.ylabel('total_hivneg_stage4')
plt.grid(True)
plt.ylim(0,100)
plt.show()


# plot number of hivneg in stage 4
out_df_13 = pd.read_csv(output_csv_file)
out_df_13 = out_df_13[['total_hivneg_stage4', 'rounded_decimal_year']].dropna()
out_df_13 = out_df_13[out_df_13['rounded_decimal_year'] >= 2011]
out_df_13['total_hivneg_stage4'] = out_df_13['total_hivneg_stage4'] * scale_factor
print(out_df_13)
plt.figure(figsize=(10, 6))
plt.plot(out_df_13['rounded_decimal_year'], out_df_13['total_hivneg_stage4'], marker='o')
plt.title('total_hivneg_stage4')
plt.xlabel('Year')
plt.ylabel('total_hivneg_stage4')
plt.grid(True)
plt.ylim(0,10000)
plt.show()

# LOG PLOTTING with function ---------------------------------------------------------------------------
#
# start_year=2011
# scale_factor = 10000
#
#
# # Function to plot data
# def plot_data(log_df, year_col, columns, prefix = '',scale_factor=1000, start_year=2011, title="", xlabel="Year", ylabel="", ylim=None, proportion_plot=False):
#     # Filter by year and ensure only valid values
#     log_df_plot = log_df["tlo.methods.cervical_cancer"]["all"]
#     log_df_plot = log_df_plot[[year_col] + columns].dropna()
#     log_df_plot = log_df_plot[log_df_plot[year_col] >= start_year]
#
#
#     # If proportion plot is True, calculate proportions
#     if proportion_plot:
#         total_col = log_df_plot[columns].sum(axis=1)  # Sum across the columns to get the total for each row
#         for col in columns:
#             new_col_name = col.replace(prefix, '')  # Remove the prefix
#             log_df_plot[f'proportion_{new_col_name}'] = log_df_plot[col] / total_col  # Calculate proportion
#
#             # Update columns to use proportion columns and remove those containing 'none'
#         columns = [f'proportion_{col.replace(prefix, "")}' for col in columns if 'none' not in col]
#
#     # Scale values
#     if not proportion_plot:
#         for col in columns:
#             log_df_plot[col] = log_df_plot[col] * scale_factor
#
#     # Plotting logic
#     plt.figure(figsize=(10, 6))
#
#     if proportion_plot:
#         bottom = 0
#         for col in columns:
#             plt.fill_between(log_df_plot[year_col], bottom, bottom + log_df_plot[col], label=col, alpha=0.7)
#             bottom += log_df_plot[col]
#         plt.legend(loc='upper right')
#     else:
#         plt.plot(log_df_plot[year_col], log_df_plot[columns[0]], marker='o')
#
#     # Plot
#     plt.style.use("seaborn-v0_8-white")
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.grid(True)
#
#     # Set y-axis limits if provided
#     if ylim:
#         plt.ylim(ylim)
#
#     plt.show()
#
# # Execute functions
#
# # 1. Total deaths by Year
# plot_data(log_df, year_col='rounded_decimal_year', columns=['n_deaths_past_year'], scale_factor=scale_factor, title='Total deaths by Year', ylabel='Total deaths past year', ylim=(0, 10000))
#
# # 2. Total deaths cervical cancer in HIV negative by Year
# plot_data(log_df, year_col='rounded_decimal_year', columns=['n_deaths_cc_hivneg_past_year'], scale_factor=scale_factor, title='Total deaths cervical cancer in HIV negative by Year', ylabel='Total deaths in HIV negative past year', ylim=(0, 10000))
#
# # 3. Total deaths cervical cancer in HIV positive by Year
# plot_data(log_df, year_col='rounded_decimal_year', columns=['n_deaths_cc_hivpos_past_year'], scale_factor=scale_factor, title='Total deaths cervical cancer in HIV positive by Year', ylabel='Total deaths in HIV positive past year', ylim=(0, 10000))
#
# # 4. Total diagnosed per Year
# plot_data(log_df, year_col='rounded_decimal_year', columns=['n_diagnosed_past_year'], scale_factor=scale_factor, title='Total diagnosed per Year', ylabel='Total diagnosed per year', ylim=(0, 10000))
#
# # 5. Total treated per Year
# plot_data(log_df, year_col='rounded_decimal_year', columns=['n_treated_past_year'], scale_factor=scale_factor, title='Total treated per Year', ylabel='Total treated per year', ylim=(0, 10000))
#
# # 6. Total cured per Year
# plot_data(log_df, year_col='rounded_decimal_year', columns=['n_cured_past_year'], scale_factor=scale_factor, title='Total cured per Year', ylabel='Total cured per year', ylim=(0, 10000))
#
# # 7. Proportion of women aged 15+ with HPV, CIN, cervical cancer
# plot_data(log_df, year_col='rounded_decimal_year', columns=['total_none', 'total_hpv', 'total_cin1', 'total_cin2', 'total_cin3', 'total_stage1',
#                         'total_stage2a', 'total_stage2b', 'total_stage3', 'total_stage4'], prefix = 'total_',scale_factor=scale_factor, title='Proportion of women aged 15+ with HPV, CIN, cervical cancer', ylabel='Proportion', ylim=(0, 0.30), proportion_plot=True)
#
# # 8. Proportion of people with cervical cancer who are HIV positive
# plot_data(log_df, year_col='rounded_decimal_year', columns=['prop_cc_hiv'], title='Proportion of people with cervical cancer who are HIV positive', ylabel='Proportion', ylim=(0, 1))
#
# # 9. Number of women living with unsuppressed HIV
# plot_data(log_df, year_col='rounded_decimal_year', columns=['n_women_hiv_unsuppressed'], scale_factor=scale_factor, title='Number of women living with unsuppressed HIV', ylabel='n_women_hiv_unsuppressed', ylim=(0, 300000))
#
# # 10. Proportion of HIV negative women aged 15+ with HPV, CIN, cervical cancer
# plot_data(log_df, year_col='rounded_decimal_year', columns=['total_hivneg_none', 'total_hivneg_hpv', 'total_hivneg_cin1', 'total_hivneg_cin2', 'total_hivneg_cin3',
#                         'total_hivneg_stage1','total_hivneg_stage2a', 'total_hivneg_stage2b', 'total_hivneg_stage3', 'total_hivneg_stage4'], prefix = 'total_',title='Proportion of HIV negative women aged 15+ with HPV, CIN, cervical cancer', ylabel='Proportion', ylim=(0, 0.30), proportion_plot=True)
#
# # 11. Proportion of HIV positive women aged 15+ with HPV, CIN, cervical cancer
# plot_data(log_df, year_col='rounded_decimal_year', columns=['total_hivpos_none', 'total_hivpos_hpv', 'total_hivpos_cin1', 'total_hivpos_cin2', 'total_hivpos_cin3',
#                         'total_hivpos_stage1','total_hivpos_stage2a', 'total_hivpos_stage2b', 'total_hivpos_stage3', 'total_hivpos_stage4'], prefix = 'total_', title='Proportion of HIV positive women aged 15+ with HPV, CIN, cervical cancer', ylabel='Proportion', ylim=(0, 0.30), proportion_plot=True)
#
# # 12. Number of HIV positive women in Stage 4
# plot_data(log_df, year_col='rounded_decimal_year', columns=['total_hivpos_stage4'], scale_factor=scale_factor, title='Number of HIV positive women in Stage 4', ylabel='total_hivpos_stage4', ylim=(0, 100))
#
# # 13. Number of HIV negative women in Stage 4
# plot_data(log_df, year_col='rounded_decimal_year', columns=['total_hivneg_stage4'], scale_factor=scale_factor, title='Number of HIV negative women in Stage 4', ylabel='total_hivneg_stage4', ylim=(0, 100))
