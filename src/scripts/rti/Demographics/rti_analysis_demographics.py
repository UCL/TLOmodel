from pathlib import Path

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    symptommanager,
    rti,
)
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy import stats

# # ======================================= Create outputs for GBD DATA ===============================================
# gbd_gender_data = pd.read_csv('resources/ResourceFile_RTI_GBD_gender_data.csv')
# global_males = gbd_gender_data.loc[(gbd_gender_data['location'] == 'Global') & (gbd_gender_data['sex'] == 'Male')]
# global_males_in_rti = global_males['val'].sum()
# global_females = gbd_gender_data.loc[(gbd_gender_data['location'] == 'Global') & (gbd_gender_data['sex'] == 'Female')]
# global_females_in_rti = global_females['val'].sum()
# global_gender_percentages = [global_males_in_rti / (global_females_in_rti + global_males_in_rti),
#                              global_females_in_rti / (global_females_in_rti + global_males_in_rti)]
# gbd_age_data = pd.read_csv('resources/ResourceFile_RTI_GBD_age_data.csv')
# age_1_to_4 = gbd_age_data.loc[gbd_age_data['age'] == '1 to 4']
# age_5_to_9 = gbd_age_data.loc[gbd_age_data['age'] == '5 to 9']
# age_10_to_14 = gbd_age_data.loc[gbd_age_data['age'] == '10 to 14']
# age_15_to_19 = gbd_age_data.loc[gbd_age_data['age'] == '15 to 19']
# age_20_to_24 = gbd_age_data.loc[gbd_age_data['age'] == '20 to 24']
# age_25_to_29 = gbd_age_data.loc[gbd_age_data['age'] == '25 to 29']
# age_30_to_34 = gbd_age_data.loc[gbd_age_data['age'] == '30 to 34']
# age_35_to_39 = gbd_age_data.loc[gbd_age_data['age'] == '35 to 39']
# age_40_to_44 = gbd_age_data.loc[gbd_age_data['age'] == '40 to 44']
# age_45_to_49 = gbd_age_data.loc[gbd_age_data['age'] == '45 to 49']
# age_50_to_54 = gbd_age_data.loc[gbd_age_data['age'] == '50 to 54']
# age_55_to_59 = gbd_age_data.loc[gbd_age_data['age'] == '55 to 59']
# age_60_to_64 = gbd_age_data.loc[gbd_age_data['age'] == '60 to 64']
# age_65_to_69 = gbd_age_data.loc[gbd_age_data['age'] == '65 to 69']
# age_70_to_74 = gbd_age_data.loc[gbd_age_data['age'] == '70 to 74']
# age_75_to_79 = gbd_age_data.loc[gbd_age_data['age'] == '75 to 79']
# age_80_to_84 = gbd_age_data.loc[gbd_age_data['age'] == '80 to 84']
# age_85_to_89 = gbd_age_data.loc[gbd_age_data['age'] == '85 to 89']
# global_total = age_1_to_4.loc[age_1_to_4['location'] == 'Global']['val'].sum() + \
#                age_5_to_9.loc[age_5_to_9['location'] == 'Global']['val'].sum() + \
#                age_10_to_14.loc[age_10_to_14['location'] == 'Global']['val'].sum() + \
#                age_15_to_19.loc[age_15_to_19['location'] == 'Global']['val'].sum() + \
#                age_20_to_24.loc[age_20_to_24['location'] == 'Global']['val'].sum() + \
#                age_25_to_29.loc[age_25_to_29['location'] == 'Global']['val'].sum() + \
#                age_30_to_34.loc[age_30_to_34['location'] == 'Global']['val'].sum() + \
#                age_35_to_39.loc[age_35_to_39['location'] == 'Global']['val'].sum() + \
#                age_40_to_44.loc[age_40_to_44['location'] == 'Global']['val'].sum() + \
#                age_45_to_49.loc[age_45_to_49['location'] == 'Global']['val'].sum() + \
#                age_50_to_54.loc[age_50_to_54['location'] == 'Global']['val'].sum() + \
#                age_55_to_59.loc[age_55_to_59['location'] == 'Global']['val'].sum() + \
#                age_60_to_64.loc[age_60_to_64['location'] == 'Global']['val'].sum() + \
#                age_65_to_69.loc[age_65_to_69['location'] == 'Global']['val'].sum() + \
#                age_70_to_74.loc[age_70_to_74['location'] == 'Global']['val'].sum() + \
#                age_75_to_79.loc[age_75_to_79['location'] == 'Global']['val'].sum() + \
#                age_80_to_84.loc[age_80_to_84['location'] == 'Global']['val'].sum() + \
#                age_85_to_89.loc[age_85_to_89['location'] == 'Global']['val'].sum()
# global_age_range = [age_1_to_4.loc[age_1_to_4['location'] == 'Global']['val'].sum(),
#                age_5_to_9.loc[age_5_to_9['location'] == 'Global']['val'].sum(),
#                age_10_to_14.loc[age_10_to_14['location'] == 'Global']['val'].sum(),
#                age_15_to_19.loc[age_15_to_19['location'] == 'Global']['val'].sum(),
#                age_20_to_24.loc[age_20_to_24['location'] == 'Global']['val'].sum(),
#                age_25_to_29.loc[age_25_to_29['location'] == 'Global']['val'].sum(),
#                age_30_to_34.loc[age_30_to_34['location'] == 'Global']['val'].sum(),
#                age_35_to_39.loc[age_35_to_39['location'] == 'Global']['val'].sum(),
#                age_40_to_44.loc[age_40_to_44['location'] == 'Global']['val'].sum(),
#                age_45_to_49.loc[age_45_to_49['location'] == 'Global']['val'].sum(),
#                age_50_to_54.loc[age_50_to_54['location'] == 'Global']['val'].sum(),
#                age_55_to_59.loc[age_55_to_59['location'] == 'Global']['val'].sum(),
#                age_60_to_64.loc[age_60_to_64['location'] == 'Global']['val'].sum(),
#                age_65_to_69.loc[age_65_to_69['location'] == 'Global']['val'].sum(),
#                age_70_to_74.loc[age_70_to_74['location'] == 'Global']['val'].sum(),
#                age_75_to_79.loc[age_75_to_79['location'] == 'Global']['val'].sum(),
#                age_80_to_84.loc[age_80_to_84['location'] == 'Global']['val'].sum(),
#                age_85_to_89.loc[age_85_to_89['location'] == 'Global']['val'].sum()
#                     ]
# global_age_distribution = np.divide(global_age_range, global_total)
# malawi_males = gbd_gender_data.loc[(gbd_gender_data['location'] == 'Malawi') & (gbd_gender_data['sex'] == 'Male')]
# malawi_males_in_rti = malawi_males['val'].sum()
# malawi_females = gbd_gender_data.loc[(gbd_gender_data['location'] == 'Malawi') & (gbd_gender_data['sex'] == 'Female')]
# malawi_females_in_rti = malawi_females['val'].sum()
# malawi_gender_percentages = [malawi_males_in_rti / (malawi_females_in_rti + malawi_males_in_rti),
#                              malawi_females_in_rti / (malawi_females_in_rti + malawi_males_in_rti)]
# malawi_total = age_1_to_4.loc[age_1_to_4['location'] == 'Malawi']['val'].sum() + \
#                age_5_to_9.loc[age_5_to_9['location'] == 'Malawi']['val'].sum() + \
#                age_10_to_14.loc[age_10_to_14['location'] == 'Malawi']['val'].sum() + \
#                age_15_to_19.loc[age_15_to_19['location'] == 'Malawi']['val'].sum() + \
#                age_20_to_24.loc[age_20_to_24['location'] == 'Malawi']['val'].sum() + \
#                age_25_to_29.loc[age_25_to_29['location'] == 'Malawi']['val'].sum() + \
#                age_30_to_34.loc[age_30_to_34['location'] == 'Malawi']['val'].sum() + \
#                age_35_to_39.loc[age_35_to_39['location'] == 'Malawi']['val'].sum() + \
#                age_40_to_44.loc[age_40_to_44['location'] == 'Malawi']['val'].sum() + \
#                age_45_to_49.loc[age_45_to_49['location'] == 'Malawi']['val'].sum() + \
#                age_50_to_54.loc[age_50_to_54['location'] == 'Malawi']['val'].sum() + \
#                age_55_to_59.loc[age_55_to_59['location'] == 'Malawi']['val'].sum() + \
#                age_60_to_64.loc[age_60_to_64['location'] == 'Malawi']['val'].sum() + \
#                age_65_to_69.loc[age_65_to_69['location'] == 'Malawi']['val'].sum() + \
#                age_70_to_74.loc[age_70_to_74['location'] == 'Malawi']['val'].sum() + \
#                age_75_to_79.loc[age_75_to_79['location'] == 'Malawi']['val'].sum() + \
#                age_80_to_84.loc[age_80_to_84['location'] == 'Malawi']['val'].sum() + \
#                age_85_to_89.loc[age_85_to_89['location'] == 'Malawi']['val'].sum()
# malawi_age_range = [age_1_to_4.loc[age_1_to_4['location'] == 'Malawi']['val'].sum(),
#                age_5_to_9.loc[age_5_to_9['location'] == 'Malawi']['val'].sum(),
#                age_10_to_14.loc[age_10_to_14['location'] == 'Malawi']['val'].sum(),
#                age_15_to_19.loc[age_15_to_19['location'] == 'Malawi']['val'].sum(),
#                age_20_to_24.loc[age_20_to_24['location'] == 'Malawi']['val'].sum(),
#                age_25_to_29.loc[age_25_to_29['location'] == 'Malawi']['val'].sum(),
#                age_30_to_34.loc[age_30_to_34['location'] == 'Malawi']['val'].sum(),
#                age_35_to_39.loc[age_35_to_39['location'] == 'Malawi']['val'].sum(),
#                age_40_to_44.loc[age_40_to_44['location'] == 'Malawi']['val'].sum(),
#                age_45_to_49.loc[age_45_to_49['location'] == 'Malawi']['val'].sum(),
#                age_50_to_54.loc[age_50_to_54['location'] == 'Malawi']['val'].sum(),
#                age_55_to_59.loc[age_55_to_59['location'] == 'Malawi']['val'].sum(),
#                age_60_to_64.loc[age_60_to_64['location'] == 'Malawi']['val'].sum(),
#                age_65_to_69.loc[age_65_to_69['location'] == 'Malawi']['val'].sum(),
#                age_70_to_74.loc[age_70_to_74['location'] == 'Malawi']['val'].sum(),
#                age_75_to_79.loc[age_75_to_79['location'] == 'Malawi']['val'].sum(),
#                age_80_to_84.loc[age_80_to_84['location'] == 'Malawi']['val'].sum(),
#                age_85_to_89.loc[age_85_to_89['location'] == 'Malawi']['val'].sum()
#                     ]
# malawi_age_distribution = np.divide(malawi_age_range, malawi_total)
#
# age_labels = ['1-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59',
#               '60-64', '65-69', '70-74', '75-79', '80-84', '85-89']
# plt.subplots(figsize=(18, 10))
#
# plt.subplot(2, 2, 1)
# colours = ['lightsteelblue', 'lightsalmon']
# plt.pie(global_gender_percentages, labels=['Males', 'Females'], autopct='%1.1f%%', startangle=90, colors=colours)
# plt.title('GBD global gender distribution'
#           '\n'
#           ' of RTI incidence, all years')
# plt.subplot(2, 2, 2)
# plt.bar(np.arange(len(global_age_distribution)), global_age_distribution, color='lightsteelblue')
# plt.xticks(np.arange(len(global_age_distribution)), age_labels, rotation=90)
# plt.ylabel('Percent')
# plt.title('GBD global age distribution '
#           '\n'
#           'of RTI incidence, all years')
# plt.subplot(2, 2, 3)
# plt.pie(malawi_gender_percentages, labels=['Males', 'Females'], autopct='%1.1f%%', startangle=90, colors=colours)
# plt.title('GBD Malawi gender distribution '
#           '\n'
#           'of RTI incidence, all years')
# plt.subplot(2, 2, 4)
# plt.bar(np.arange(len(malawi_age_distribution)), malawi_age_distribution, color='lightsteelblue')
# plt.xticks(np.arange(len(malawi_age_distribution)), age_labels, rotation=90)
# plt.ylabel('Percent')
# plt.title('GBD Malawi age distribution '
#           '\n'
#           'of RTI incidence, all years')
# plt.tight_layout()
# plt.savefig('outputs/Demographics_of_RTI/GBD_RTI_demography.png')
# plt.clf()
#
# gbd_gender_data = pd.read_csv('resources/ResourceFile_RTI_GBD_age_gender_data.csv')
# number_of_injuries = gbd_gender_data.loc[gbd_gender_data['metric'] == 'Number']
# number_males_injured = number_of_injuries.loc[number_of_injuries['sex'] == 'Male']
# number_females_injured = number_of_injuries.loc[number_of_injuries['sex'] == 'Female']
# rate_of_injuries = gbd_gender_data.loc[gbd_gender_data['metric'] == 'Rate']
# rate_males_injured = rate_of_injuries.loc[rate_of_injuries['sex'] == 'Male']
# rate_females_injured = rate_of_injuries.loc[rate_of_injuries['sex'] == 'Female']
# female_df = number_females_injured.groupby(['age']).sum()
# female_df['rank'] = [1, 3, 4, 5, 6, 7, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# female_df = female_df.sort_values(by=['rank'])
# ages = female_df.index.tolist()
# ages.reverse()
# female_number = female_df['val'].tolist()
# female_number.reverse()
# male_df = number_males_injured.groupby(['age']).sum()
# male_df['rank'] = [1, 3, 4, 5, 6, 7, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# male_df = male_df.sort_values(by=['rank'])
# male_number = male_df['val'].tolist()
# male_number.reverse()
# plt.close()
# plt.clf()
# plt.plot(number_males_injured.groupby(['year']).sum().index, number_males_injured.groupby(['year']).sum()['val'],
#          label='Males', color='lightsteelblue')
# plt.plot(number_females_injured.groupby(['year']).sum().index, number_females_injured.groupby(['year']).sum()['val'],
#          label='Females', color='lightsalmon')
# plt.plot(number_males_injured.groupby(['year']).sum().index,
#          number_males_injured.groupby(['year']).sum()['val'] + number_females_injured.groupby(['year']).sum()['val'],
#          label='Total', color='black')
# plt.xlabel('Year')
# plt.ylabel('Number of road traffic injuries')
# plt.title('Number of road traffic injuries in Malawi per year, GBD data')
# plt.legend(loc='center right')
# plt.savefig('outputs/Demographics_of_RTI/Malawi_Number_of_injuries.png')
# plt.clf()
# plt.barh(ages, male_number, alpha=0.5, label='Males', color='lightsteelblue')
# plt.barh(ages, np.multiply(female_number, -1), alpha=0.5, label='Females', color='lightsalmon')
# locs, labels = plt.xticks()
# plt.xticks(locs, np.sqrt(locs**2), fontsize=8)
# plt.title('Sum total of number of road traffic injuries in Malawi'
#           '\n'
#           'by age and sex over all years, GBD data')
# plt.xlabel('Number')
# plt.yticks(fontsize=7)
# plt.legend()
# plt.tight_layout()
# plt.savefig('outputs/Demographics_of_RTI/Malawi_Injury_Demographics.png')
# plt.clf()
# plt.barh(ages, np.divide(male_number, sum(male_number)), alpha=0.5, label='Males', color='lightsteelblue')
# plt.barh(ages, np.multiply(np.divide(female_number, sum(female_number)), -1), alpha=0.5,
#          label='Females', color='lightsalmon')
# locs, labels = plt.xticks()
# plt.xticks(locs, np.sqrt(locs**2), fontsize=8)
# plt.title('Distribution of number of road traffic injuries in Malawi'
#           '\n'
#           'by age and sex over all years, GBD data')
# plt.xlabel('Number')
# plt.yticks(fontsize=7)
# plt.legend()
# plt.tight_layout()
# plt.savefig('outputs/Demographics_of_RTI/Malawi_Injury_Demographics_percentage.png')
# plt.clf()
#
# gbd_cat_2017 = [24026.90542, 1082.276734, 7941.462531, 7578.726195, 7578.726195, 1825.22282, 106.8162861, 1004.93119,
#                 559.5158363, 10931.61332, 1712.892472]
# gbd_cat_2017_labels = ['Fracture', 'Dislocation', 'TBI', 'Soft Tissue Inj.', 'Int. Organ Inj.',
#                        'Int. Bleeding', 'SCI', 'Amputation', 'Eye injury', 'Laceration', 'Burn']
# plt.bar(np.arange(len(gbd_cat_2017)), np.divide(gbd_cat_2017, sum(gbd_cat_2017)))
# plt.xticks(np.arange(len(gbd_cat_2017)), gbd_cat_2017_labels, rotation=90)
# plt.title('GBD Injury categories Malawi 2017')
# plt.savefig('outputs/Demographics_of_RTI/GBD_injury_category_distribution.png', bbox_inches='tight')
# plt.clf()

# # Plot data on vehicle ownership vs death incidence in Africa
# df = pd.read_csv('resources/ResourceFile_RTI_Vehicle_Ownersip_Death_Data.csv', skipinitialspace=True)
# # Preprocessing
# df = df.dropna()
# df['n_vehicles'] = pd.to_numeric(df['n_vehicles'])
# df['adjusted_n_deaths'] = pd.to_numeric(df['adjusted_n_deaths'])
#
# def group_by_gdp(row):
#     if row['gdp_usd_per_capita'] < 1005:
#         val = 'Low income'
#     elif row['gdp_usd_per_capita'] < 3955:
#         val = 'Lower middle income'
#     elif row['gdp_usd_per_capita'] < 12235:
#         val = 'Upper middle income'
#     else:
#         val = 'High income'
#     return val
#
#
# df['income_index'] = df.apply(group_by_gdp, axis=1)
# # drop outliers
# df = df.drop(df.n_vehicles.nlargest(3).index)
# low_income_slope, low_income_intercept, r_value, low_income_p_value, std_err = \
#     stats.linregress(df.loc[df['income_index'] == 'Low income', 'n_vehicles'],
#                      df.loc[df['income_index'] == 'Low income', 'mortality_rate_2016'])
# low_middle_income_slope, low_middle_income_intercept, r_value, low_middle_income_p_value, std_err = \
#     stats.linregress(df.loc[df['income_index'] == 'Lower middle income', 'n_vehicles'],
#                      df.loc[df['income_index'] == 'Lower middle income', 'mortality_rate_2016'])
# upper_middle_income_slope, upper_middle_income_intercept, r_value, upper_middle_income_p_value, std_err = \
#     stats.linregress(df.loc[df['income_index'] == 'Upper middle income', 'n_vehicles'],
#                      df.loc[df['income_index'] == 'Upper middle income', 'mortality_rate_2016'])
# high_income_slope, high_income_intercept, r_value, high_income_p_value, std_err = \
#     stats.linregress(df.loc[df['income_index'] == 'High income', 'n_vehicles'],
#                      df.loc[df['income_index'] == 'High income', 'mortality_rate_2016'])
# groups = df.groupby('income_index')
# for name, group in groups:
#     plt.plot(group.n_vehicles, group.mortality_rate_2016, marker='o', linestyle='', markersize=12, label=name)
# plt.xlabel('Number of vehicles')
# plt.ylabel('Mortality rate per 100,000 people per year')
# plt.legend()
# plt.title('The number of vehicles vs the mortality rate due to RTI per 100,000, grouped by GDP')
# plt.savefig('outputs/Demographics_of_RTI/N_vehicles_vs_incidence_scatter.png', bbox_inches='tight')
# plt.clf()
# plt.subplot(2, 2, 1)
# plt.scatter(df.loc[df['income_index'] == 'Low income', 'n_vehicles'],
#             df.loc[df['income_index'] == 'Low income', 'mortality_rate_2016'], c='blue')
# plt.plot(df.loc[df['income_index'] == 'Low income', 'n_vehicles'],
#          low_income_intercept + low_income_slope * df.loc[df['income_index'] == 'Low income', 'n_vehicles'],
#          color='blue')
# plt.xlabel('Number of vehicles')
# plt.ylabel('Deaths per 100,000'
#            '\n'
#            ' population in 2016')
# plt.title(f"Low income, p = {np.round(low_income_p_value, 2)}")
# plt.subplot(2, 2, 2)
# plt.scatter(df.loc[df['income_index'] == 'Lower middle income', 'n_vehicles'],
#             df.loc[df['income_index'] == 'Lower middle income', 'mortality_rate_2016'], c='red')
# plt.plot(df.loc[df['income_index'] == 'Lower middle income', 'n_vehicles'],
#          low_middle_income_intercept +
#          low_middle_income_slope * df.loc[df['income_index'] == 'Lower middle income', 'n_vehicles'], color='red')
# plt.xlabel('Number of vehicles')
# plt.ylabel('Deaths per 100,000'
#            '\n'
#            ' population in 2016')
# plt.title(f"Lower middle income, p = {np.round(low_middle_income_p_value, 2)}")
# plt.subplot(2, 2, 3)
# plt.scatter(df.loc[df['income_index'] == 'Upper middle income', 'n_vehicles'],
#             df.loc[df['income_index'] == 'Upper middle income', 'mortality_rate_2016'], c='green')
# plt.plot(df.loc[df['income_index'] == 'Upper middle income', 'n_vehicles'],
#          upper_middle_income_intercept +
#          upper_middle_income_slope * df.loc[df['income_index'] == 'Upper middle income', 'n_vehicles'], color='green')
# plt.xlabel('Number of vehicles')
# plt.ylabel('Deaths per 100,000'
#            '\n'
#            ' population in 2016')
# plt.title(f"Upper middle income, p = {np.round(upper_middle_income_p_value, 2)}")
# plt.subplot(2, 2, 4)
# plt.scatter(df.loc[df['income_index'] == 'High income', 'n_vehicles'],
#             df.loc[df['income_index'] == 'High income', 'mortality_rate_2016'], c='yellow')
# plt.plot(df.loc[df['income_index'] == 'High income', 'n_vehicles'],
#          high_income_intercept + high_income_slope * df.loc[df['income_index'] == 'High income', 'n_vehicles'],
#          color='yellow')
# plt.title(f"High income, p = {np.round(high_income_p_value, 2)}")
# plt.xlabel('Number of vehicles')
# plt.ylabel('Deaths per 100,000'
#            '\n'
#            ' population in 2016')
# plt.tight_layout()
# plt.savefig('outputs/Demographics_of_RTI/Insignificant_relationship_between_n_vehicles_deaths.png')
# plt.clf()
# ============================================== Model run ============================================================
log_config = {
    "filename": "rti_health_system_comparison",  # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.rti": logging.INFO,
        "tlo.methods.healthsystem": logging.DEBUG
    }
}
# The Resource files [NB. Working directory must be set to the root of TLO: TLOmodel]
resourcefilepath = Path('./resources')
# Establish the simulation object
yearsrun = 10
start_date = Date(year=2010, month=1, day=1)
end_date = Date(year=(2010 + yearsrun), month=1, day=1)
pop_size = 50000
nsim = 20
output_for_different_incidence = dict()
service_availability = ['*']
sim_age_range = []
females = []
males = []
incidences_of_rti = []
incidences_of_death = []
incidences_of_injuries = []
ps_of_imm_death = []
ps_of_death_post_med = []
ps_of_death_without_med = []
percent_of_fatal_crashes = []
perc_mild = []
perc_severe = []
iss_scores = []
number_of_injured_body_locations = []
inj_loc_data = []
inj_cat_data = []
rti_model_flow_summary = []
number_of_injuries_in_sim = []
inc_amputations = []
inc_burns = []
inc_fractures = []
inc_tbi = []
inc_sci = []
inc_minor = []
inc_other = []
tot_inc_injuries = []
for i in range(0, nsim):
    age_range = []
    sim = Simulation(start_date=start_date)
    # We register all modules in a single call to the register method, calling once with multiple
    # objects. This is preferred to registering each module in multiple calls because we will be
    # able to handle dependencies if modules are registered together
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=service_availability),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        rti.RTI(resourcefilepath=resourcefilepath)
    )
    logfile = sim.configure_logging(filename="LogFile")
    # create and run the simulation
    sim.make_initial_population(n=pop_size)
    params = sim.modules['RTI'].parameters
    params['allowed_interventions'] = []
    params['base_rate_injrti'] = params['base_rate_injrti'] * 12.5
    params['head_prob_skin_wound'] = 0.25909 + (params['head_prob_fracture'] - 0.04) / 2
    params['head_prob_TBI'] = 0.69086 + (params['head_prob_fracture'] - 0.04) / 2
    params['head_prob_fracture'] = 0.04
    orig = params['rr_injrti_male']
    params['imm_death_proportion_rti'] = 0.1
    params['rr_injrti_male'] = 1.1
    params['rr_injrti_age018'] = 1.2
    params['rr_injrti_age1829'] = 1.4
    params['rr_injrti_age3039'] = 1.4
    params['rr_injrti_age4049'] = 1.15
    params['rr_injrti_age5059'] = 1.15
    params['rr_injrti_age6069'] = 1.15
    params['rr_injrti_age7079'] = 1.15

    sim.simulate(end_date=end_date)
    log_df = parse_log_file(logfile)
    demog = log_df['tlo.methods.rti']['rti_demography']
    males.append(sum(demog['males_in_rti']))
    females.append(sum(demog['females_in_rti']))
    this_sim_ages = demog['age'].tolist()
    incidences_of_rti.append(log_df['tlo.methods.rti']['summary_1m']['incidence of rti per 100,000'].tolist())
    number_of_injuries_in_sim.append(sum(log_df['tlo.methods.rti']['summary_1m']['total injuries']))
    incidences_of_death.append(log_df['tlo.methods.rti']['summary_1m']['incidence of rti death per 100,000'].tolist())
    incidences_of_injuries.append(log_df['tlo.methods.rti']['summary_1m']['injury incidence per 100,000'].tolist())
    deaths_df = log_df['tlo.methods.demography']['death']
    rti_deaths = len(deaths_df.loc[deaths_df['cause'] != 'Other'])
    try:
        ps_of_imm_death.append(len(deaths_df.loc[deaths_df['cause'] == 'RTI_imm_death']) / rti_deaths)
        ps_of_death_post_med.append(len(deaths_df[deaths_df['cause'] == 'RTI_death_with_med']) / rti_deaths)
    except ZeroDivisionError:
        ps_of_imm_death.append(0)
        ps_of_death_post_med.append(0)
    number_of_crashes = sum(log_df['tlo.methods.rti']['summary_1m']['number involved in a rti'])
    percent_of_fatal_crashes.append(rti_deaths / number_of_crashes)
    for elem in this_sim_ages:
        for item in elem:
            sim_age_range.append(item)
    mild_injuries_in_run = log_df['tlo.methods.rti']['injury_severity']['total_mild_injuries'].iloc[-1]
    severe_injuries_in_run = log_df['tlo.methods.rti']['injury_severity']['total_severe_injuries'].iloc[-1]
    perc_mild.append(mild_injuries_in_run / (mild_injuries_in_run + severe_injuries_in_run))
    perc_severe.append(severe_injuries_in_run / (mild_injuries_in_run + severe_injuries_in_run))
    severity_distibution = log_df['tlo.methods.rti']['injury_severity']['ISS_score'].iloc[-1]
    for score in severity_distibution:
        iss_scores.append(score)
    injury_number_distribution = log_df['tlo.methods.rti']['number_of_injuries'].drop('date', axis=1).iloc[-1].tolist()
    number_of_injured_body_locations.append(injury_number_distribution)
    injury_location_this_sim = log_df['tlo.methods.rti']['injury_location_data'].drop('date', axis=1).iloc[-1].tolist()
    inj_loc_data.append(injury_location_this_sim)
    injury_category_data = log_df['tlo.methods.rti']['injury_characteristics'].drop('date', axis=1).iloc[-1].tolist()
    inj_cat_data.append(injury_category_data)
    rti_model_flow_summary.append(log_df['tlo.methods.rti']['model_progression'].drop('date', axis=1).iloc[-1].tolist())
    injury_category_incidence = log_df['tlo.methods.rti']['Inj_category_incidence']
    inc_amputations.append(injury_category_incidence['inc_amputations'].tolist())
    inc_burns.append(injury_category_incidence['inc_burns'].tolist())
    inc_fractures.append(injury_category_incidence['inc_fractures'].tolist())
    inc_tbi.append(injury_category_incidence['inc_tbi'].tolist())
    inc_sci.append(injury_category_incidence['inc_sci'].tolist())
    inc_minor.append(injury_category_incidence['inc_minor'].tolist())
    inc_other.append(injury_category_incidence['inc_other'].tolist())
    tot_inc_injuries.append(injury_category_incidence['tot_inc_injuries'].tolist())

zero_to_five = len([i for i in sim_age_range if i < 6])
six_to_ten = len([i for i in sim_age_range if 6 <= i < 11])
eleven_to_fifteen = len([i for i in sim_age_range if 11 <= i < 16])
sixteen_to_twenty = len([i for i in sim_age_range if 16 <= i < 21])
twenty1_to_twenty5 = len([i for i in sim_age_range if 21 <= i < 26])
twenty6_to_thirty = len([i for i in sim_age_range if 26 <= i < 31])
thirty1_to_thirty5 = len([i for i in sim_age_range if 31 <= i < 36])
thirty6_to_forty = len([i for i in sim_age_range if 36 <= i < 41])
forty1_to_forty5 = len([i for i in sim_age_range if 41 <= i < 46])
forty6_to_fifty = len([i for i in sim_age_range if 46 <= i < 51])
fifty1_to_fifty5 = len([i for i in sim_age_range if 51 <= i < 56])
fifty6_to_sixty = len([i for i in sim_age_range if 56 <= i < 61])
sixty1_to_sixty5 = len([i for i in sim_age_range if 61 <= i < 66])
sixty6_to_seventy = len([i for i in sim_age_range if 66 <= i < 71])
seventy1_to_seventy5 = len([i for i in sim_age_range if 71 <= i < 76])
seventy6_to_eighty = len([i for i in sim_age_range if 76 <= i < 81])
eighty1_to_eighty5 = len([i for i in sim_age_range if 81 <= i < 86])
eighty6_to_ninety = len([i for i in sim_age_range if 86 <= i < 91])
ninety_plus = len([i for i in sim_age_range if 90 < i])
height_for_bar_plot = [zero_to_five, six_to_ten, eleven_to_fifteen, sixteen_to_twenty, twenty1_to_twenty5,
                       twenty6_to_thirty, thirty1_to_thirty5, thirty6_to_forty, forty1_to_forty5, forty6_to_fifty,
                       fifty1_to_fifty5, fifty6_to_sixty, sixty1_to_sixty5, sixty6_to_seventy, seventy1_to_seventy5,
                       seventy6_to_eighty, eighty1_to_eighty5, eighty6_to_ninety, ninety_plus]
height_for_bar_plot = np.divide(height_for_bar_plot, sum(height_for_bar_plot))
print(height_for_bar_plot, sum(height_for_bar_plot))
labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40',
          '41-45', '46-50', '51-55', '56-60', '61-65', '66-70', '71-75', '76-80',
          '81-85', '86-90', '90+']
plt.bar(np.arange(len(height_for_bar_plot)), height_for_bar_plot, color='lightsteelblue')
plt.xticks(np.arange(len(height_for_bar_plot)), labels, rotation=45)
plt.ylabel('Percentage')
plt.xlabel('Age')
plt.title(f"Age demographics of those with RTIs"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/Demographics_of_RTI/Age_demographics.png', bbox_inches='tight')
plt.clf()
plt.close()

total_injuries = [i + j for i, j in zip(males, females)]
male_perc = np.divide(males, total_injuries)
femal_perc = np.divide(females, total_injuries)
n = np.arange(2)
data = [np.mean(male_perc), np.mean(femal_perc)]
plt.bar(np.arange(2), data, yerr=[np.std(male_perc), np.std(femal_perc)], color='lightsteelblue')
for i in range(len(data)):
    plt.annotate(str(data[i]), xy=(n[i], data[i]), ha='center', va='bottom')
plt.xticks(np.arange(2), ['Males', 'Females'])
plt.ylabel('Percentage')
plt.xlabel('Gender')
plt.title(f"Gender demographics of those with RTIs"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/Demographics_of_RTI/Gender_demographics.png', bbox_inches='tight')
plt.clf()
average_incidence = [float(sum(col)) / len(col) for col in zip(*incidences_of_rti)]
std_incidence = [np.std(i) for i in zip(*incidences_of_rti)]
inc_upper = [inc + (1.96 * std) / nsim for inc, std in zip(average_incidence, std_incidence)]
inc_lower = [inc - (1.96 * std) / nsim for inc, std in zip(average_incidence, std_incidence)]
average_deaths = [float(sum(col)) / len(col) for col in zip(*incidences_of_death)]
std_deaths = [np.std(j) for j in zip(*incidences_of_death)]
death_upper = [inc + (1.96 * std) / nsim for inc, std in zip(average_deaths, std_deaths)]
death_lower = [inc - (1.96 * std) / nsim for inc, std in zip(average_deaths, std_deaths)]

average_injury_incidence = [float(sum(col)) / len(col) for col in zip(*incidences_of_injuries)]
overall_av_inc_sim = np.mean(average_incidence)
overall_av_death_inc_sim = np.mean(average_deaths)
overall_av_inc_injuries = np.mean(average_injury_incidence)
time = log_df['tlo.methods.rti']['summary_1m']['date']
plt.plot(time, average_incidence, color='blue', label='Incidence of RTI', zorder=2)
plt.fill_between(time.tolist(), inc_upper, inc_lower, alpha=0.5, color='blue', label='95% C.I., RTI inc.', zorder=1)
plt.plot(time, average_deaths, color='red', label='Incidence of death '
                                                  '\n'
                                                  'due to RTI', zorder=2)
plt.fill_between(time.tolist(), death_upper, death_lower, alpha=0.5, color='red', label='95% C.I. inc death', zorder=1)
# plt.plot(time, average_injury_incidence, color='green', label='Incidence of RTI injury')
plt.hlines(overall_av_inc_sim, time.iloc[0], time.iloc[-1], label=f"Average incidence of "
                                                                  f"\n"
                                                                  f"RTI = {np.round(overall_av_inc_sim, 2)}",
           color='blue', linestyles='--')
plt.hlines(overall_av_death_inc_sim, time.iloc[0], time.iloc[-1], label=f"Average incidence of "
                                                                        f"\n"
                                                                        f"death = "
                                                                        f"{np.round(overall_av_death_inc_sim, 2)}",
           color='red', linestyles='--')
plt.xlabel('Simulation time')
plt.ylabel('Incidence per 100,000')
plt.legend(loc='upper center', bbox_to_anchor=(1.1, 0.8), shadow=True, ncol=1)
plt.title(f"Average incidence of RTIs and deaths due to RTI"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/Demographics_of_RTI/Incidence_and_deaths.png', bbox_inches='tight')
plt.clf()
print(percent_of_fatal_crashes)
mean_fatal_crashes_of_all_sim = np.mean(percent_of_fatal_crashes)
std_fatal_crashes = np.std(percent_of_fatal_crashes)
non_fatal_crashes_of_all_sim = [i - j for i, j in zip(np.ones(len(percent_of_fatal_crashes)), percent_of_fatal_crashes)]
mean_non_fatal = np.mean(non_fatal_crashes_of_all_sim)
std_non_fatal_crashes = np.std(non_fatal_crashes_of_all_sim)
data = [mean_fatal_crashes_of_all_sim, mean_non_fatal]
n = np.arange(2)
plt.bar(n, data, yerr=[std_fatal_crashes, std_non_fatal_crashes], color='lightsteelblue')
for i in range(len(data)):
    plt.annotate(str(data[i]), xy=(n[i], data[i]), ha='center', va='bottom')
plt.xticks(np.arange(2), ['fatal', 'non-fatal'])
plt.title(f"Average percentage of those with RTI who perished"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/Demographics_of_RTI/Percentage_of_deaths.png', bbox_inches='tight')

plt.clf()
colours = ['lightsteelblue', 'lightsalmon']
plt.pie([np.mean(ps_of_imm_death), 1 - np.mean(ps_of_imm_death)], labels=['Death on scene', 'Death post med'],
        autopct='%1.1f%%', startangle=90, colors=colours)
plt.title(f"Average cause of death breakdown in RTI"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.axis('equal')

plt.savefig('outputs/Demographics_of_RTI/Percentage_cause_of_deaths.png', bbox_inches='tight')
plt.clf()
n = np.arange(2)
data = [np.mean(perc_mild), np.mean(perc_severe)]
plt.bar(n, data, yerr=[np.std(perc_mild), np.std(perc_severe)], color='lightsteelblue')
for i in range(len(data)):
    plt.annotate(str(data[i]), xy=(n[i], data[i]), ha='center', va='bottom')
plt.xticks(np.arange(2), labels=['Mild injuries', 'Severe injuries'])
plt.title(f"Average road traffic injury severity distribution"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")

plt.savefig('outputs/Demographics_of_RTI/Percentage_mild_severe_injuries.png', bbox_inches='tight')
plt.clf()
# Plot the distribution of the ISS scores
scores, counts = np.unique(iss_scores, return_counts=True)
plt.bar(scores, counts / sum(counts), width=0.8, color='lightsteelblue')
plt.xlabel('ISS scores')
plt.ylabel('Percentage')
plt.title(f"Average road traffic injury ISS score distribution"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.xlim([0, 75])
plt.savefig('outputs/Demographics_of_RTI/Average_ISS_scores.png', bbox_inches='tight')
plt.clf()
# Plot the distribution of the number of injured body regions
average_number_of_body_regions_injured = [float(sum(col)) / len(col) for col in zip(*number_of_injured_body_locations)]
plt.bar(np.arange(8), np.divide(average_number_of_body_regions_injured, sum(average_number_of_body_regions_injured)),
        color='lightsteelblue')
plt.xticks(np.arange(8), ['1', '2', '3', '4', '5', '6', '7', '8'])
plt.xlabel('Number of injured AIS body regions')
plt.ylabel('Percentage')
plt.title(f"Average injured body region distribution"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/Demographics_of_RTI/Average_injured_body_region_distribution.png', bbox_inches='tight')
plt.clf()
# plot the injury location data
average_inj_loc = [float(sum(col)) / len(col) for col in zip(*inj_loc_data)]
plt.bar(np.arange(8), np.divide(average_inj_loc, sum(average_inj_loc)), color='lightsteelblue')
plt.xticks(np.arange(8), ['Head', 'Face', 'Neck', 'Thorax', 'Abdomen', 'Spine', 'UpperX', 'LowerX'], rotation=45)
plt.xlabel('AIS body regions')
plt.ylabel('Percentage')
plt.title(f"Average injury location distribution"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/Demographics_of_RTI/Average_injury_location_distribution.png', bbox_inches='tight')
plt.clf()
# Plot the injury category data
average_inj_cat = [float(sum(col)) / len(col) for col in zip(*inj_cat_data)]
plt.bar(np.arange(len(average_inj_cat)), np.divide(average_inj_cat, sum(average_inj_cat)), color='lightsteelblue')
plt.xticks(np.arange(len(average_inj_cat)), ['Fracture', 'Dislocation', 'TBI', 'Soft Tissue Inj.', 'Int. Organ Inj.',
                                             'Int. Bleeding', 'SCI', 'Amputation', 'Eye injury', 'Laceration', 'Burn'],
           rotation=90)
plt.ylabel('Percentage')
plt.title(f"Average injury category distribution"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/Demographics_of_RTI/Average_injury_category_distribution.png', bbox_inches='tight')
plt.clf()
# Plot the incidence of injuries as per the GBD definitions
average_inc_amputations = [float(sum(col)) / len(col) for col in zip(*inc_amputations)]
mean_inc_amp = np.mean(average_inc_amputations)
std_amp = np.std(average_inc_amputations)
gbd_inc_amp = 5.85
average_inc_burns = [float(sum(col)) / len(col) for col in zip(*inc_burns)]
mean_inc_burns = np.mean(average_inc_burns)
std_burns = np.std(average_inc_burns)
gbd_inc_burns = 5.88
average_inc_fractures = [float(sum(col)) / len(col) for col in zip(*inc_fractures)]
mean_inc_fractures = np.mean(average_inc_fractures)
std_fractures = np.std(average_inc_fractures)
gbd_inc_fractures = 139.76
average_inc_tbi = [float(sum(col)) / len(col) for col in zip(*inc_tbi)]
mean_inc_tbi = np.mean(average_inc_tbi)
std_tbi = np.std(average_inc_tbi)
gbd_inc_tbi = 46.19
average_inc_sci = [float(sum(col)) / len(col) for col in zip(*inc_sci)]
mean_inc_sci = np.mean(average_inc_sci)
std_sci = np.std(average_inc_sci)
gbd_inc_sci = 0.62
average_inc_minor = [float(sum(col)) / len(col) for col in zip(*inc_minor)]
mean_inc_minor = np.mean(average_inc_minor)
std_minor = np.std(average_inc_minor)
gbd_inc_minor = 126.35
average_inc_other = [float(sum(col)) / len(col) for col in zip(*inc_other)]
mean_inc_other = np.mean(average_inc_other)
std_other = np.std(average_inc_other)
gbd_inc_other = 52.93
average_inc_total = [float(sum(col)) / len(col) for col in zip(*tot_inc_injuries)]
mean_inc_total = np.mean(average_inc_total)
std_total = np.std(average_inc_total)
gbd_total = gbd_inc_amp + gbd_inc_burns + gbd_inc_fractures + gbd_inc_minor + gbd_inc_other + gbd_inc_sci + gbd_inc_tbi
model_category_incidences = [mean_inc_amp, mean_inc_burns, mean_inc_fractures, mean_inc_tbi, mean_inc_sci,
                             mean_inc_minor, mean_inc_other, mean_inc_total]
model_inc_errors = [std_amp, std_burns, std_fractures, std_tbi, std_sci, std_minor, std_other, std_total]
gbd_category_incidences = [gbd_inc_amp, gbd_inc_burns, gbd_inc_fractures, gbd_inc_tbi, gbd_inc_sci, gbd_inc_minor,
                           gbd_inc_other, gbd_total]
width = 0.35
plt.bar(np.arange(len(model_category_incidences)), model_category_incidences, width, color='lightsteelblue',
        yerr=model_inc_errors, label='Model output')
plt.bar(np.arange(len(model_category_incidences)) + width, gbd_category_incidences, width, color='lightsalmon',
        label='GBD 2017 data')
plt.title(f"Average injury incidence compared to GBD data"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.ylabel('Incidence per 100,000 per year')
labels = ['Amputations', 'Burns', 'Fractures', 'TBI', 'SCI', 'Minor', 'Other', 'Total']
plt.xticks(np.arange(len(model_category_incidences)) + width / 2, labels, rotation=45)
plt.legend()
plt.savefig('outputs/Demographics_of_RTI/Average_injury_incidence_per_100000_bar.png', bbox_inches='tight')
plt.clf()
# plt.plot(injury_category_incidence['date'], average_inc_amputations, color='red', label='Amputations')
plt.hlines(mean_inc_amp, injury_category_incidence['date'].iloc[0], injury_category_incidence['date'].iloc[-1],
           color='red', label=f"Average incidence amputation = {mean_inc_amp}")
# plt.plot(injury_category_incidence['date'], average_inc_burns, color='blue', label='Burns')
plt.hlines(mean_inc_burns, injury_category_incidence['date'].iloc[0], injury_category_incidence['date'].iloc[-1],
           color='blue', label=f"Average incidence burns = {mean_inc_burns}")
# plt.plot(injury_category_incidence['date'], average_inc_fractures, color='yellow', label='Fractures')
plt.hlines(mean_inc_fractures, injury_category_incidence['date'].iloc[0], injury_category_incidence['date'].iloc[-1],
           color='yellow', label=f"Average incidence fractures = {mean_inc_fractures}")
# plt.plot(injury_category_incidence['date'], average_inc_tbi, color='green', label='TBI')
plt.hlines(mean_inc_tbi, injury_category_incidence['date'].iloc[0], injury_category_incidence['date'].iloc[-1],
           color='green', label=f"Average incidence TBI = {mean_inc_tbi}")
# plt.plot(injury_category_incidence['date'], average_inc_sci, color='pink', label='SCI')
plt.hlines(mean_inc_sci, injury_category_incidence['date'].iloc[0], injury_category_incidence['date'].iloc[-1],
           color='pink', label=f"Average incidence SCI = {mean_inc_sci}")
# plt.plot(injury_category_incidence['date'], average_inc_minor, color='darkseagreen', label='Minor')
plt.hlines(mean_inc_minor, injury_category_incidence['date'].iloc[0], injury_category_incidence['date'].iloc[-1],
           color='darkseagreen', label=f"Average incidence Minor = {mean_inc_minor}")
# plt.plot(injury_category_incidence['date'], average_inc_other, color='gold', label='Other')
plt.hlines(mean_inc_other, injury_category_incidence['date'].iloc[0], injury_category_incidence['date'].iloc[-1],
           color='gold', label=f"Average incidence other = {mean_inc_other}")
# plt.plot(injury_category_incidence['date'], average_inc_total, color='black', label='Total')
plt.hlines(mean_inc_total, injury_category_incidence['date'].iloc[0], injury_category_incidence['date'].iloc[-1],
           color='black', label=f"Average incidence total = {mean_inc_total}")
plt.xlabel('Time')
plt.ylabel('Incidence per 100,000')
plt.title(f"Average injury incidence by GBD categories"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.legend()
plt.savefig('outputs/Demographics_of_RTI/Average_injury_incidence_per_100000.png', bbox_inches='tight')
plt.clf()

params_dict = dict()
params_dict.update({'base_rate_injrti': params['base_rate_injrti'],
                    'rr_injrti_male': params['rr_injrti_male'],
                    'rr_injrti_age018': params['rr_injrti_age018'],
                    'rr_injrti_age1829': params['rr_injrti_age1829'],
                    'rr_injrti_age3039': params['rr_injrti_age3039'],
                    'rr_injrti_age4049': params['rr_injrti_age4049'],
                    'rr_injrti_age5059': params['rr_injrti_age5059'],
                    'rr_injrti_age6069': params['rr_injrti_age6069'],
                    'rr_injrti_age7079': params['rr_injrti_age7079']})
print(params_dict)
