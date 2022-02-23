"""
This file produces histograms of the healthsystem capabilities \
in terms of staff allocation and daily capabilities in minutes per cadre per facility level.
"""

from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Get the path of the folder that stores the data - three scenarios: actual, funded, funded_plus
workingpath = Path('./resources/healthsystem/human_resources/actual')

# Define the path of output histograms - three scenarios: actual, funded, funded_plus
outputpath = Path('./outputs/healthsystem/human_resources/actual')

# Read data
data = pd.read_csv(workingpath / 'ResourceFile_Daily_Capabilities.csv')


# MINUTES PER HEALTH OFFICER CATEGORY BY DISTRICT
data_districts = data.dropna(inplace=False)
dat = pd.DataFrame(data_districts.groupby(['District', 'Officer_Category'], as_index=False)['Total_Mins_Per_Day'].sum())
tab = dat.pivot(index='District', columns='Officer_Category', values='Total_Mins_Per_Day')
ax = tab.plot.bar(stacked=True, fontsize='medium')
plt.ylabel('Average Total Minutes per Day', fontsize='large')
plt.xlabel('District', fontsize='large')

ax.legend(ncol=3, bbox_to_anchor=(0, 1),
          loc='lower left', fontsize='medium')

plt.savefig(outputpath / 'health_officer_minutes_per_district.pdf', bbox_inches='tight')

# STAFF COUNTS PER HEALTH OFFICER CATEGORY BY DISTRICT
data_districts = data.dropna(inplace=False)
dat = pd.DataFrame(data_districts.groupby(['District', 'Officer_Category'], as_index=False)['Staff_Count'].sum())
tab = dat.pivot(index='District', columns='Officer_Category', values='Staff_Count')
ax = tab.plot.bar(stacked=True, fontsize='medium')
plt.ylabel('Staff counts', fontsize='large')
plt.xlabel('District', fontsize='large')

ax.legend(ncol=3, bbox_to_anchor=(0, 1),
          loc='lower left', fontsize='medium')

plt.savefig(outputpath / 'staff_allocation_per_district.pdf', bbox_inches='tight')


# MINUTES PER HEALTH OFFICER CATEGORY BY LEVEL
dat = pd.DataFrame(data.groupby(['Facility_Level', 'Officer_Category'], as_index=False)['Total_Mins_Per_Day'].sum())
tab = dat.pivot(index='Facility_Level', columns='Officer_Category', values='Total_Mins_Per_Day')
ax = tab.plot.bar(stacked=True, fontsize='medium')
# ax = tab.plot.bar(stacked=True, log=True)
plt.ylabel('Average Total Minutes per Day', fontsize='large')
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
tab = dat.pivot(index='Facility_Level', columns='Officer_Category', values='Staff_Count')
ax = tab.plot.bar(stacked=True, fontsize='medium')
# ax = tab.plot.bar(stacked=True, log=True)
plt.ylabel('Staff counts', fontsize='large')
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
