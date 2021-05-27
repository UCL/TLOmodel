from matplotlib import pyplot as plt

import math
import numpy as np
import os
import pandas as pd
import sys
# create helper function to get the summary statistics


def get_summary_stats(df):
    # get the columns of interest: all the rti columns and the cause of death column
    rti_columns_filter = df.filter(like='rt_').columns
    df = df[rti_columns_filter]
    sample_size = len(df)
    # isolate the injured population
    injured_pop = df.loc[df.rt_road_traffic_inc]
    # calculate the proportion of the population injured in road traffic injuries
    proportion_in_rti = df.rt_road_traffic_inc.mean()
    sample_sd_prop_in_rti = df.rt_road_traffic_inc.std() / sample_size ** (1/2)
    # calculate the proportion of injuries that are mild
    mild_injuries = np.where(injured_pop['rt_inj_severity'] == 'mild', True, False)
    percent_mild_injuries = mild_injuries.mean()
    std_mild_injuries = mild_injuries.std() / sample_size ** (1/2)
    # calculate the average daly weight of the injured population
    average_daly_wt = injured_pop['rt_disability'].mean()
    std_daly_wt = injured_pop['rt_disability'].std() / sample_size ** (1/2)
    # get the number of injured people who died from RTI
    died_from_rti = (injured_pop.rt_imm_death | injured_pop.rt_post_med_death | injured_pop.rt_no_med_death |
                     injured_pop.rt_unavailable_med_death | injured_pop.rt_death_from_shock)
    # calculate the percent of injuries that are fatal
    percent_fatal = died_from_rti.mean()
    std_percent_fatal = died_from_rti.std() / sample_size ** (1/2)
    # calculate the mean ISS score
    mean_iss_score = injured_pop['rt_ISS_score'].mean()
    std_iss_score = injured_pop['rt_ISS_score'].std() / sample_size ** (1/2)
    # calculate the percentage of injured people who sought treatment
    percent_sought_treatment = injured_pop.rt_diagnosed.mean()
    std_percent_sought_treatment = injured_pop.rt_diagnosed.std() / sample_size ** (1/2)
    # calculate the percentage of people who are recieving treatment
    percent_recieving_med = injured_pop.rt_med_int.mean()
    std_percent_recieving_med = injured_pop.rt_med_int.std() / sample_size ** (1/2)
    # calculate the percentage of people in the population who are in the icu
    percent_patients_in_icu = injured_pop.rt_in_icu_or_hdu.mean()
    std_patients_in_icu = injured_pop.rt_in_icu_or_hdu.std() / sample_size ** (1/2)
    # calculate the percentage of people in the population who are permanently disabled
    percent_perm_disabled = df.rt_perm_disability.mean()
    std_percent_perm_disabled = df.rt_perm_disability.std() / sample_size ** (1/2)

    dict_to_output = {'sample_size': sample_size,
                      'prop_injured': proportion_in_rti,
                      'std_prop_injured': sample_sd_prop_in_rti,
                      'prop_mild_inj': percent_mild_injuries,
                      'std_prop_mild_inj': std_mild_injuries,
                      'mean_health_burden': average_daly_wt,
                      'std_mean_health_burden': std_daly_wt,
                      'percent_fatal': percent_fatal,
                      'std_percent_fatal': std_percent_fatal,
                      'mean_iss_score': mean_iss_score,
                      'std_mean_iss_score': std_iss_score,
                      'percent_sought_treatment': percent_sought_treatment,
                      'std_percent_sought_treatment': std_percent_sought_treatment,
                      'percent_recieving_medical_care': percent_recieving_med,
                      'std_percent_recieving_medical_care': std_percent_recieving_med,
                      'percent_patients_in_icu': percent_patients_in_icu,
                      'std_percent_patients_in_icu': std_patients_in_icu,
                      'percent_perm_disabled': percent_perm_disabled,
                      'std_percent_perm_disabled': std_percent_perm_disabled
                      }
    return dict_to_output


def tidy(x, n):
    """Return 'x' rounded to 'n' significant digits."""
    y = abs(x)
    if y <= sys.float_info.min:
        return 0.0
    return round(x, int(n - math.ceil(math.log10(y))))


# create a sample run which samples the dataframe for various size n, see when changes to result stop
# create sample sizes
sample_size_list = np.linspace(50000, 1000000, 100).tolist()
sample_size_list = [int(sample) for sample in sample_size_list]
save_path = "C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/MinimumViablePopulation/SingleSamplePerN/"
summary_df = pd.DataFrame()
# get dataframe from the end of the model run
last_df = os.listdir("C:/Users/Robbie Manning Smith/Documents/Dataframe_dump")[-1]
file_path = "C:/Users/Robbie Manning Smith/Documents/Dataframe_dump/" + last_df

data = pd.read_csv(file_path)
for sample_size in sample_size_list:
    print(sample_size)
    # load the data
    # sample from the data
    sample_data = data.sample(n=sample_size)
    # calculate summary statistics
    summary_results = get_summary_stats(sample_data)
    # create a dataframe to store the results in
    results_df = pd.DataFrame(summary_results, index=[sample_size])
    # store the data frame in summary_df to work on later
    summary_df = summary_df.append(results_df)

sig_fig = 10
min_pop_sizes_per_estimate = []
column_names = []
for column in summary_df.columns[1:]:
    if column == 'sample_size' or column.startswith('std_'):
        pass
    else:
        plt.plot(summary_df['sample_size'], summary_df[column])
        upper_ci = summary_df[column] + 1.96 * summary_df['std_' + column]
        lower_ci = summary_df[column] - 1.96 * summary_df['std_' + column]
        plt.fill_between(summary_df['sample_size'], upper_ci, lower_ci, alpha=0.5)
        plt.xlabel('Sample size')
        plt.ylabel(column)
        plt.title('The effect of sample size on ' + column)
        plt.savefig(save_path + 'sample_size_on' + column)
        plt.clf()

        margin_of_error = tidy(summary_df[column].to_list()[-1], sig_fig) / 100
        min_viable_pop_size_for_estimate = (1.96 * summary_df['std_' + column].to_list()[-1] / margin_of_error) ** 2
        column_names.append(column)
        min_pop_sizes_per_estimate.append(min_viable_pop_size_for_estimate)
# plot minimum sample size
plt.bar(np.arange(len(column_names)), min_viable_pop_size_for_estimate)
plt.xticks(np.arange(len(column_names)), column_names, rotation=90)
plt.ylabel('Minimum viable population')
plt.title("The minimum viable population size for each estimate, based on \n"
          "a 90% certainty that the sample mean represents the population mean")
plt.savefig(save_path + 'Min_viab_pop_size_each_estimate.png')
plt.clf()
# create sample sizes
sample_size_list = np.linspace(50000, 1000000, 10).tolist()
sample_size_list = [int(sample) for sample in sample_size_list]
# Create a dataframe to store the summary information from the samples
summary_df = pd.DataFrame()
# Choose the number of samples per sample size
number_of_samples_per_sample_size = 10
# loop over the number of samples
for i in range(0, number_of_samples_per_sample_size):
    # loop over the sample sizes
    for sample_size in sample_size_list:
        # loop over the monthly slices of the model stored as csv files
        for csv_file in os.listdir("C:/Users/Robbie Manning Smith/Documents/Dataframe_dump"):
            # load the data
            data = pd.read_csv("C:/Users/Robbie Manning Smith/Documents/Dataframe_dump/" + csv_file)
            # sample from the data
            sample_data = data.sample(n=sample_size)
            # calculate summary statistics
            summary_results = get_summary_stats(sample_data)
            # create a dataframe to store the results in
            results_df = pd.DataFrame(summary_results, index=[sample_size])
            # store the data frame in summary_df to work on later
            summary_df = summary_df.append(results_df)

# group summary statistics by sample size, calculating the mean
group_by_sample_size = summary_df.groupby('sample_size').mean()
# calculate the standard deviation in the sample sizes
group_by_sample_size_std = summary_df.groupby('sample_size').std()
# calculate the upper bound for the 95% C.I.
group_by_sample_size_upper = group_by_sample_size + group_by_sample_size_std
# calculate the lower bound for the 95% C.I.
group_by_sample_size_lower = group_by_sample_size - group_by_sample_size_std
# Calculate the true values of the summary statistics in the model population
population_mean = group_by_sample_size.iloc[-1]
# Calculate the 95% C.I. of the summary statistics
population_upper = group_by_sample_size_upper.iloc[-1]
population_lower = group_by_sample_size_lower.iloc[-1]
# label the data from the population
ybar_coords = {'Proportion of population injured': population_mean['prop_injured'],
               'Proportion of injuries that are mild': population_mean['prop_mild_inj'],
               'Mean DALY weight in injured population': population_mean['mean_health_burden'],
               'Percent of injuries that were fatal': population_mean['percent_fatal'],
               'Mean ISS score of injured persons': population_mean['mean_iss_score'],
               'Percent of people who sought healthcare': population_mean['percent_sought_treatment'],
               'Percent of people currently recieving care': population_mean['percent_recieving_medical_care'],
               'Percent of RTI patients in the ICU': population_mean['percent_patients_in_icu'],
               'Percent of population left permanently disabled': population_mean['percent_perm_disabled']
               }
# label the upper bounds for the 95% C.I. of the summary stats of the population
ybar_upper_coords = {'Proportion of population injured': population_upper['prop_injured'],
                     'Proportion of injuries that are mild': population_upper['prop_mild_inj'],
                     'Mean DALY weight in injured population': population_upper['mean_health_burden'],
                     'Percent of injuries that were fatal': population_upper['percent_fatal'],
                     'Mean ISS score of injured persons': population_upper['mean_iss_score'],
                     'Percent of people who sought healthcare': population_upper['percent_sought_treatment'],
                     'Percent of people currently recieving care': population_upper['percent_recieving_medical_care'],
                     'Percent of RTI patients in the ICU': population_upper['percent_patients_in_icu'],
                     'Percent of population left permanently disabled': population_upper['percent_perm_disabled']
                     }
# label the lower bounds for the 95% C.I. of the summary stats of the population
ybar_lower_coords = {'Proportion of population injured': population_lower['prop_injured'],
                     'Proportion of injuries that are mild': population_lower['prop_mild_inj'],
                     'Mean DALY weight in injured population': population_lower['mean_health_burden'],
                     'Percent of injuries that were fatal': population_lower['percent_fatal'],
                     'Mean ISS score of injured persons': population_lower['mean_iss_score'],
                     'Percent of people who sought healthcare': population_lower['percent_sought_treatment'],
                     'Percent of people currently recieving care': population_lower['percent_recieving_medical_care'],
                     'Percent of RTI patients in the ICU': population_lower['percent_patients_in_icu'],
                     'Percent of population left permanently disabled': population_lower['percent_perm_disabled']
                     }
# Get the x coordinates (the sample sizes)
x_coords = group_by_sample_size.index.to_list()
# label the data from the samples
y_coords = {'Proportion of population injured': group_by_sample_size['prop_injured'].to_list(),
            'Proportion of injuries that are mild': group_by_sample_size['prop_mild_inj'].to_list(),
            'Mean DALY weight in injured population': group_by_sample_size['mean_health_burden'].to_list(),
            'Percent of injuries that were fatal': group_by_sample_size['percent_fatal'].to_list(),
            'Mean ISS score of injured persons': group_by_sample_size['mean_iss_score'].to_list(),
            'Percent of people who sought healthcare': group_by_sample_size['percent_sought_treatment'].to_list(),
            'Percent of people currently recieving care': group_by_sample_size['percent_recieving_medical_'
                                                                               'care'].to_list(),
            'Percent of RTI patients in the ICU': group_by_sample_size['percent_patients_in_icu'].to_list(),
            'Percent of population left permanently disabled': group_by_sample_size['percent_perm_disabled'].to_list()
            }
# label the upper bounds for the 95% C.I. of the summary stats of the samples
y_upper_bounds = {'Proportion of population injured': group_by_sample_size_upper['prop_injured'].to_list(),
                  'Proportion of injuries that are mild': group_by_sample_size_upper['prop_mild_inj'].to_list(),
                  'Mean DALY weight in injured population': group_by_sample_size_upper['mean_health_burden'].to_list(),
                  'Percent of injuries that were fatal': group_by_sample_size_upper['percent_fatal'].to_list(),
                  'Mean ISS score of injured persons': group_by_sample_size_upper['mean_iss_score'].to_list(),
                  'Percent of people who sought healthcare': group_by_sample_size_upper['percent_sought_'
                                                                                        'treatment'].to_list(),
                  'Percent of people currently recieving care': group_by_sample_size_upper['percent_recieving_medical_'
                                                                                           'care'].to_list(),
                  'Percent of RTI patients in the ICU': group_by_sample_size_upper['percent_patients_in_icu'].to_list(),
                  'Percent of population left permanently disabled': group_by_sample_size_upper['percent_perm_'
                                                                                                'disabled'].to_list()
                  }
# label the lower bounds for the 95% C.I. of the summary stats of the samples
y_lower_bounds = {'Proportion of population injured': group_by_sample_size_lower['prop_injured'].to_list(),
                  'Proportion of injuries that are mild': group_by_sample_size_lower['prop_mild_inj'].to_list(),
                  'Mean DALY weight in injured population': group_by_sample_size_lower['mean_health_burden'].to_list(),
                  'Percent of injuries that were fatal': group_by_sample_size_lower['percent_fatal'].to_list(),
                  'Mean ISS score of injured persons': group_by_sample_size_lower['mean_iss_score'].to_list(),
                  'Percent of people who sought healthcare': group_by_sample_size_lower['percent_sought_'
                                                                                        'treatment'].to_list(),
                  'Percent of people currently recieving care': group_by_sample_size_lower['percent_recieving_medical_'
                                                                                           'care'].to_list(),
                  'Percent of RTI patients in the ICU': group_by_sample_size_lower['percent_patients_in_icu'].to_list(),
                  'Percent of population left permanently disabled': group_by_sample_size_lower['percent_perm_'
                                                                                                'disabled'].to_list()
                  }
# iterate over each summary statistic
for result in y_coords.keys():
    # plot the effect of population size on the summary statistic
    plt.plot(x_coords, y_coords[result], label='Sample mean', color='lightsteelblue')
    # plot the standard deviation in the sample for each sample size
    plt.fill_between(x_coords, y_upper_bounds[result], y_lower_bounds[result], alpha=0.5, label='95% C.I.',
                     color='lightsteelblue')
    # plot the true value of the summary statistic in the population
    plt.hlines(y=ybar_coords[result], xmin=sample_size_list[0], xmax=sample_size_list[-1],
               label='Population mean', colors='lightsalmon')
    # plot the standard deviation of the summary statistic in the population
    plt.fill_between(x_coords, ybar_upper_coords[result], ybar_lower_coords[result], alpha=0.5, label='95% C.I.',
                     color='lightsalmon')
    plt.xlabel('Population sample size')
    plt.ylabel(result)
    plt.legend
    plt.title('The effect of population size on the \n' + result + f"\n using {number_of_samples_per_sample_size} "
                                                                   f"samples per population size")
    plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/MinimumViablePopulation/" + result + ".png",
                bbox_inches='tight')
    plt.clf()
