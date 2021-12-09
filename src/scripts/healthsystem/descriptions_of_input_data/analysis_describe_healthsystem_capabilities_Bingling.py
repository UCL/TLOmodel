"""
This file produces histograms of the healthsystem capabilities \
in terms of staff allocation and daily capabilities in minutes per cadre per facility level.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Get the path of the folder that stores the data - three scenarios: actual, funded, funded_plus
workingpath = Path('./resources/healthsystem/human_resources/funded_plus')

# Define the path of output histograms - three scenarios: actual, funded, funded_plus
outputpath = Path('./outputs/healthsystem/human_resources/funded_plus')

# Read data
data = pd.read_csv(workingpath / 'ResourceFile_Daily_Capabilities.csv')


# MINUTES PER HEALTH OFFICER CATEGORY BY DISTRICT
data_districts = data.dropna(inplace=False)
dat = pd.DataFrame(data_districts.groupby(['District', 'Officer_Category'], as_index=False)['Total_Mins_Per_Day'].sum())
tab = dat.pivot(index='District', columns='Officer_Category', values='Total_Mins_Per_Day')
ax = tab.plot.bar(stacked=True)
plt.ylabel('Minutes per day')
plt.xlabel('District')

ax.legend(ncol=3, bbox_to_anchor=(0, 1),
          loc='lower left', fontsize='small')

plt.savefig(outputpath / 'health_officer_minutes_per_district.pdf', bbox_inches='tight')

# STAFF COUNTS PER HEALTH OFFICER CATEGORY BY DISTRICT
data_districts = data.dropna(inplace=False)
dat = pd.DataFrame(data_districts.groupby(['District', 'Officer_Category'], as_index=False)['Staff_Count'].sum())
tab = dat.pivot(index='District', columns='Officer_Category', values='Staff_Count')
ax = tab.plot.bar(stacked=True)
plt.ylabel('Staff counts')
plt.xlabel('District')

ax.legend(ncol=3, bbox_to_anchor=(0, 1),
          loc='lower left', fontsize='small')

plt.savefig(outputpath / 'staff_allocation_per_district.pdf', bbox_inches='tight')


# MINUTES PER HEALTH OFFICER CATEGORY BY LEVEL
dat = pd.DataFrame(data.groupby(['Facility_Level', 'Officer_Category'], as_index=False)['Total_Mins_Per_Day'].sum())
tab = dat.pivot(index='Facility_Level', columns='Officer_Category', values='Total_Mins_Per_Day')
ax = tab.plot.bar(stacked=True)
# ax = tab.plot.bar(stacked=True, log=True)
plt.ylabel('Minutes per day')
plt.xlabel('Facility level')

ax.tick_params(axis='x', rotation=0)

formatter = ScalarFormatter()
formatter.set_scientific(False)
ax.yaxis.set_major_formatter(formatter)

ax.legend(ncol=3, bbox_to_anchor=(0, 1),
          loc='lower left', fontsize='small')

plt.savefig(outputpath / 'health_officer_minutes_per_level.pdf', bbox_inches='tight')

# STAFF COUNTS PER HEALTH OFFICER CATEGORY BY LEVEL
dat = pd.DataFrame(data.groupby(['Facility_Level', 'Officer_Category'], as_index=False)['Staff_Count'].sum())
tab = dat.pivot(index='Facility_Level', columns='Officer_Category', values='Staff_Count')
ax = tab.plot.bar(stacked=True)
# ax = tab.plot.bar(stacked=True, log=True)
plt.ylabel('Staff counts')
plt.xlabel('Facility level')

ax.tick_params(axis='x', rotation=0)

ax.legend(ncol=3, bbox_to_anchor=(0, 1),
          loc='lower left', fontsize='small')

plt.savefig(outputpath / 'staff_allocation_per_level.pdf', bbox_inches='tight')


# MINUTES PER HEALTH OFFICER CATEGORY BY LEVEL

# Level 0
data_level = data.loc[data['Facility_Level'] == '0',:]
tab = data_level.pivot(index='District', columns='Officer_Category', values='Total_Mins_Per_Day')
ax = tab.plot.bar(stacked=True)
plt.ylabel('Minutes per day at level 0')
plt.xlabel('District')

ax.legend(ncol=3, bbox_to_anchor=(0, 1),
          loc='lower left', fontsize='small')

plt.savefig(outputpath / 'health_officer_minutes_per_district_level_0.pdf', bbox_inches='tight')

# Level 1a
data_level = data.loc[data['Facility_Level'] == '1a',:]
tab = data_level.pivot(index='District', columns='Officer_Category', values='Total_Mins_Per_Day')
ax = tab.plot.bar(stacked=True)
plt.ylabel('Minutes per day at level 1a')
plt.xlabel('District')

ax.legend(ncol=3, bbox_to_anchor=(0, 1),
          loc='lower left', fontsize='small')

plt.savefig(outputpath / 'health_officer_minutes_per_district_level_1a.pdf', bbox_inches='tight')

# Level 1b
data_level = data.loc[data['Facility_Level'] == '1b',:]
tab = data_level.pivot(index='District', columns='Officer_Category', values='Total_Mins_Per_Day')
ax = tab.plot.bar(stacked=True)
plt.ylabel('Minutes per day at level 1b')
plt.xlabel('District')

ax.legend(ncol=3, bbox_to_anchor=(0, 1),
          loc='lower left', fontsize='small')

plt.savefig(outputpath / 'health_officer_minutes_per_district_level_1b.pdf', bbox_inches='tight')

# Level 2
data_level = data.loc[data['Facility_Level'] == '2',:]
tab = data_level.pivot(index='District', columns='Officer_Category', values='Total_Mins_Per_Day')
ax = tab.plot.bar(stacked=True)
plt.ylabel('Minutes per day at level 2')
plt.xlabel('District')

ax.legend(ncol=3, bbox_to_anchor=(0, 1),
          loc='lower left', fontsize='small')

plt.savefig(outputpath / 'health_officer_minutes_per_district_level_2.pdf', bbox_inches='tight')

# Level 3
data_level = data.loc[data['Facility_Level'] == '3',:]
tab = data_level.pivot(index='Region', columns='Officer_Category', values='Total_Mins_Per_Day')
ax = tab.plot.bar(stacked=True)
plt.ylabel('Minutes per day at level 3')
plt.xlabel('Regional Referral Hospital')
ax.tick_params(axis='x', rotation=0)

ax.legend(ncol=3, bbox_to_anchor=(0, 1),
          loc='lower left', fontsize='small')

plt.savefig(outputpath / 'health_officer_minutes_per_district_level_3.pdf', bbox_inches='tight')

# Level 4
data_level = data.loc[data['Facility_Level'] == '4',:]
tab = data_level.pivot(index='Facility_Name', columns='Officer_Category', values='Total_Mins_Per_Day')
ax = tab.plot.bar(stacked=True, width=0.1)
plt.ylabel('Minutes per day at level 4')
plt.xlabel('National resource hospital')
ax.tick_params(axis='x', rotation=0)

ax.legend(ncol=3, bbox_to_anchor=(0, 1),
          loc='lower left', fontsize='small')

plt.savefig(outputpath / 'health_officer_minutes_per_district_level_4.pdf', bbox_inches='tight')
