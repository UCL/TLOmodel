"""
This file produces a nice plot of the capabilities of the healthsystem in terms of the hours available for
different cadres of healthcare workers.
"""

# %%

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

resourcefilepath = Path("./resources")

# %%


data = pd.read_csv(
    Path(resourcefilepath) / "ResourceFile_Daily_Capabilities.csv"
)

# [['Total_Minutes_Per_Day','Officer_Type','District']]

data = data.dropna()
# data['District'] = data['District'].fillna('National')

# do some re-grouping to make a more manageable number of health cadres:
data['Officer_Type'] = data['Officer_Type'].replace('DCSA', 'CHW')
data['Officer_Type'] = data['Officer_Type'].replace(['Lab Officer', 'Lab Technician', 'Lab Assistant'], 'Lab Support')
data['Officer_Type'] = data['Officer_Type'].replace(['Radiographer', 'Radiography Technician'], 'Radiography')
data['Officer_Type'] = data['Officer_Type'].replace(['Nurse Officer', 'Nutrition Staff', 'Med. Assistant'], 'Nurse')
data['Officer_Type'] = data['Officer_Type'].replace('Nurse Midwife Technician', 'MidWife')
data['Officer_Type'] = data['Officer_Type'].replace(['Pharmacist', 'Pharm Technician', 'Pharm Assistant'], 'Pharmacy')
data['Officer_Type'] = data['Officer_Type'].replace(['Medical Officer / Specialist', 'Clinical Officer / Technician'],
                                                    'Clinician')
data['Officer_Type'] = data['Officer_Type'].replace(['Dental Therapist'], 'Dentist')

# MINUTES PER HEALTH OFFICER TYPE BY DISTRICT:
dat = pd.DataFrame(data.groupby(['District', 'Officer_Type'], as_index=False)['Total_Minutes_Per_Day'].sum())
tab = dat.pivot(index='District', columns='Officer_Type', values='Total_Minutes_Per_Day')
ax = tab.plot.bar(stacked=True)
plt.ylabel('Minutes per day')
plt.xlabel('District')

ax.legend(ncol=3, bbox_to_anchor=(0, 1),
          loc='lower left', fontsize='small')

plt.savefig('health_officer_minutes_per_district.pdf', bbox_inches='tight')
plt.show()

# %%
