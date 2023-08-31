"""
This file produces histograms of the healthsystem capabilities \
in terms of staff allocation and daily capabilities in minutes per cadre per facility level.
"""

from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Get the path of the folder that stores the data - three scenarios: actual, funded, funded_plus
workingpath = Path('./resources/healthsystem/human_resources')
wp_actual = workingpath / 'actual'
wp_funded_plus = workingpath / 'funded_plus'

# Define the path of output histograms - three scenarios: actual, funded, funded_plus
outputpath = Path('./outputs/healthsystem/human_resources/actual')
op_actual = outputpath / 'actual'
op_funded_plus = outputpath / 'funded_plus'

# Read actual data
data = pd.read_csv(wp_actual / 'ResourceFile_Daily_Capabilities.csv')

# Read funded_plus data
data_funded_plus = pd.read_csv(wp_funded_plus / 'ResourceFile_Daily_Capabilities.csv')


# ***for actual scenario***
# MINUTES PER HEALTH OFFICER CATEGORY BY DISTRICT
data_districts = data.dropna(inplace=False)
dat = pd.DataFrame(data_districts.groupby(['District', 'Officer_Category'], as_index=False)['Total_Mins_Per_Day'].sum())
dat['Total_Mins_Per_Day'] = dat['Total_Mins_Per_Day'] / 100000
tab = dat.pivot(index='District', columns='Officer_Category', values='Total_Mins_Per_Day')
ax = tab.plot.bar(stacked=True, fontsize='medium')
plt.ylabel('Average Total Minutes per Day in 1e5', fontsize='large')
plt.xlabel('District', fontsize='large')

ax.legend(ncol=3, bbox_to_anchor=(0, 1),
          loc='lower left', fontsize='medium')

plt.savefig(outputpath / 'health_officer_minutes_per_district.pdf', bbox_inches='tight')

# STAFF COUNTS PER HEALTH OFFICER CATEGORY BY DISTRICT
data_districts = data.dropna(inplace=False)
dat = pd.DataFrame(data_districts.groupby(['District', 'Officer_Category'], as_index=False)['Staff_Count'].sum())
dat['Staff_Count'] = dat['Staff_Count'] / 1000
tab = dat.pivot(index='District', columns='Officer_Category', values='Staff_Count')
ax = tab.plot.bar(stacked=True, fontsize='medium')
plt.ylabel('Staff counts in 1e3', fontsize='large')
plt.xlabel('District', fontsize='large')

ax.legend(ncol=3, bbox_to_anchor=(0, 1),
          loc='lower left', fontsize='medium')

plt.savefig(outputpath / 'staff_allocation_per_district.pdf', bbox_inches='tight')


# MINUTES PER HEALTH OFFICER CATEGORY BY LEVEL
dat = pd.DataFrame(data.groupby(['Facility_Level', 'Officer_Category'], as_index=False)['Total_Mins_Per_Day'].sum())
dat['Total_Mins_Per_Day'] = dat['Total_Mins_Per_Day'] / 100000
tab = dat.pivot(index='Facility_Level', columns='Officer_Category', values='Total_Mins_Per_Day')
ax = tab.plot.bar(stacked=True, fontsize='medium')
# ax = tab.plot.bar(stacked=True, log=True)
plt.ylabel('Average Total Minutes per Day in 1e5', fontsize='large')
plt.xlabel('Facility level', fontsize='large')

ax.tick_params(axis='x', rotation=0)

formatter = ScalarFormatter()
formatter.set_scientific(False)
ax.yaxis.set_major_formatter(formatter)

ax.legend(ncol=3, bbox_to_anchor=(0, 1),
          loc='lower left', fontsize='medium')

plt.savefig(outputpath / 'health_officer_minutes_per_level.pdf', bbox_inches='tight')

# STAFF COUNTS PER HEALTH OFFICER CATEGORY BY LEVEL
dat = pd.DataFrame(data.groupby(['Facility_Level', 'Officer_Category'], as_index=False)['Staff_Count'].sum())
dat['Staff_Count'] = dat['Staff_Count'] / 1000
tab = dat.pivot(index='Facility_Level', columns='Officer_Category', values='Staff_Count')
ax = tab.plot.bar(stacked=True, fontsize='medium')
# ax = tab.plot.bar(stacked=True, log=True)
plt.ylabel('Staff counts in 1e3', fontsize='large')
plt.xlabel('Facility level', fontsize='large')

ax.tick_params(axis='x', rotation=0)

ax.legend(ncol=3, bbox_to_anchor=(0, 1),
          loc='lower left', fontsize='medium')

plt.savefig(outputpath / 'staff_allocation_per_level.pdf', bbox_inches='tight')


# MINUTES PER HEALTH OFFICER CATEGORY BY LEVEL

# Level 0
data_level = data.loc[data['Facility_Level'] == '0', :]
tab = data_level.pivot(index='District', columns='Officer_Category', values='Total_Mins_Per_Day')
ax = tab.plot.bar(stacked=True, fontsize='medium')
plt.ylabel('Average Total Minutes per Day at Level 0', fontsize='large')
plt.xlabel('District', fontsize='large')

ax.legend(ncol=3, bbox_to_anchor=(0, 1),
          loc='lower left', fontsize='medium')

plt.savefig(outputpath / 'health_officer_minutes_per_district_level_0.pdf', bbox_inches='tight')

# Level 1a
data_level = data.loc[data['Facility_Level'] == '1a', :]
tab = data_level.pivot(index='District', columns='Officer_Category', values='Total_Mins_Per_Day')
ax = tab.plot.bar(stacked=True, fontsize='medium')
plt.ylabel('Average Total Minutes per Day at Level 1a', fontsize='large')
plt.xlabel('District', fontsize='large')

ax.legend(ncol=3, bbox_to_anchor=(0, 1),
          loc='lower left', fontsize='medium')

plt.savefig(outputpath / 'health_officer_minutes_per_district_level_1a.pdf', bbox_inches='tight')

# Level 1b
data_level = data.loc[data['Facility_Level'] == '1b', :]
tab = data_level.pivot(index='District', columns='Officer_Category', values='Total_Mins_Per_Day')
ax = tab.plot.bar(stacked=True, fontsize='medium')
plt.ylabel('Average Total Minutes per Day at Level 1b', fontsize='large')
plt.xlabel('District', fontsize='large')

ax.legend(ncol=3, bbox_to_anchor=(0, 1),
          loc='lower left', fontsize='medium')

plt.savefig(outputpath / 'health_officer_minutes_per_district_level_1b.pdf', bbox_inches='tight')

# Level 2
data_level = data.loc[data['Facility_Level'] == '2', :]
tab = data_level.pivot(index='District', columns='Officer_Category', values='Total_Mins_Per_Day')
ax = tab.plot.bar(stacked=True, fontsize='medium')
plt.ylabel('Average Total Minutes per Day at Level 2', fontsize='large')
plt.xlabel('District', fontsize='large')

ax.legend(ncol=3, bbox_to_anchor=(0, 1),
          loc='lower left', fontsize='medium')

plt.savefig(outputpath / 'health_officer_minutes_per_district_level_2.pdf', bbox_inches='tight')

# Level 3
data_level = data.loc[data['Facility_Level'] == '3', :]
tab = data_level.pivot(index='Region', columns='Officer_Category', values='Total_Mins_Per_Day')
ax = tab.plot.bar(stacked=True, fontsize='medium')
plt.ylabel('Average Total Minutes per Day at Level 3', fontsize='large')
plt.xlabel('Regional Referral Hospital', fontsize='large')
ax.tick_params(axis='x', rotation=0)

ax.legend(ncol=3, bbox_to_anchor=(0, 1),
          loc='lower left', fontsize='medium')

plt.savefig(outputpath / 'health_officer_minutes_per_district_level_3.pdf', bbox_inches='tight')

# Level 4
data_level = data.loc[data['Facility_Level'] == '4', :]
tab = data_level.pivot(index='Facility_Name', columns='Officer_Category', values='Total_Mins_Per_Day')
ax = tab.plot.bar(stacked=True, width=0.1, fontsize='medium')
plt.ylabel('Average Total Minutes per Day at Level 4', fontsize='large')
plt.xlabel('National resource hospital', fontsize='large')
ax.tick_params(axis='x', rotation=0)

ax.legend(ncol=3, bbox_to_anchor=(0, 1),
          loc='lower left', fontsize='medium')

plt.savefig(outputpath / 'health_officer_minutes_per_district_level_4.pdf', bbox_inches='tight')

# ***end of actual scenario***

# ***compare actual and funded_plus scenarios***
total_actual = data.drop_duplicates().groupby(['Officer_Category']).agg(
    {'Total_Mins_Per_Day': 'sum', 'Staff_Count': 'sum'}).reset_index()
total_actual['Total_Mins_Per_Year'] = total_actual['Total_Mins_Per_Day'] * 365.25
total_actual['Scenario'] = 'Actual'
total_actual[['Abs_Change_Staff_Count', 'Rel_Change_Staff_Count', 'Abs_Change_Total_Mins', 'Rel_Change_Total_Mins']] = 0

total_funded_plus = data_funded_plus.drop_duplicates().groupby(['Officer_Category']).agg(
    {'Total_Mins_Per_Day': 'sum', 'Staff_Count': 'sum'}).reset_index()
total_funded_plus['Total_Mins_Per_Year'] = total_funded_plus['Total_Mins_Per_Day'] * 365.25
total_funded_plus['Scenario'] = 'Establishment'

assert (total_actual.Officer_Category == total_funded_plus.Officer_Category).all()
total_funded_plus['Abs_Change_Staff_Count'] = total_funded_plus['Staff_Count'] - total_actual['Staff_Count']
total_funded_plus['Rel_Change_Staff_Count'] = (total_funded_plus['Staff_Count'] - total_actual['Staff_Count']
                                               ) / total_actual['Staff_Count']
total_funded_plus['Abs_Change_Total_Mins'] = (total_funded_plus['Total_Mins_Per_Year'] -
                                              total_actual['Total_Mins_Per_Year'])
total_funded_plus['Rel_Change_Total_Mins'] = (total_funded_plus['Total_Mins_Per_Year'] -
                                              total_actual['Total_Mins_Per_Year']
                                              ) / total_actual['Total_Mins_Per_Year']

total = pd.concat([total_actual, total_funded_plus]).reset_index(drop=True)
