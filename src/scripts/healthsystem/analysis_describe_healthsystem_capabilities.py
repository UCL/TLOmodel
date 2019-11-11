"""
This file produces a nice plot of the capabilities of the healthsystem in terms of the hours available for
different cadres of healthcare workers.
"""

#%%

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


resourcefilepath = Path("./resources")

data = pd.read_csv(
    Path(resourcefilepath) / "ResourceFile_Daily_Capabilities.csv"
)[['Total_Minutes_Per_Day','Officer_Type','District']]

data = data.dropna()
# data['District'] = data['District'].fillna('National')

# do some re-grouping to make a more manageable number of health cadres:
data['Officer_Type'] = data['Officer_Type'].replace('DCSA','CHW')
data['Officer_Type'] = data['Officer_Type'].replace(['Lab Officer','Lab Technician', 'Lab Assistant'],'Lab Support')
data['Officer_Type'] = data['Officer_Type'].replace([],'Radiography')
data['Officer_Type'] = data['Officer_Type'].replace(['Nurse Officer', 'Nutrition Staff', 'Med. Assistant'],'Nurse')
data['Officer_Type'] = data['Officer_Type'].replace('Nurse Midwife Technician','MidWife')


tab = data.pivot(index='District',columns='Officer_Type',values='Total_Minutes_Per_Day')
tab.plot.bar(stacked=True, legend=None)
plt.ylabel('Minutes per day')
plt.xlabel('District')
plt.show()

#%%
N = 5
labels = ['G1', 'G2', 'G3', 'G4', 'G5']
men_means = [20, 34, 30, 35, 27]
women_means = [25, 32, 34, 20, 25]

ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence


x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width, label='Men')
rects2 = ax.bar(x + width/2, women_means, width, label='Women', bottom=men_means)



plt.ylabel('Scores')
plt.title('Scores by group and gender')
plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0]), ('Men', 'Women'))

plt.show()

