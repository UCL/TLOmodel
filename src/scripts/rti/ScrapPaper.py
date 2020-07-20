import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# yearsrun = 1234
# popsize = 5678
# data = pd.read_csv('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/RTIInjuryDemographics.csv')
#
# childrenCounts = len(data.loc[data['age_years'].between(0, 17, inclusive=True)])
# youngAdultCounts = len(data.loc[data['age_years'].between(18, 29, inclusive=True)])
# thirtiesCounts = len(data.loc[data['age_years'].between(30, 39, inclusive=True)])
# fourtiesCounts = len(data.loc[data['age_years'].between(40, 49, inclusive=True)])
# fiftiesAndSixtiesCounts = len(data.loc[data['age_years'].between(50, 69, inclusive=True)])
# seventiesPlus = len(data.loc[data['age_years'] >= 70])
# counts = [childrenCounts, youngAdultCounts, thirtiesCounts, fourtiesCounts, fiftiesAndSixtiesCounts, seventiesPlus]
# percentages = np.divide(counts, sum(counts))
# labels = ['0-17', '18-29', '30-39', '40-49', '50-69', '70+']
# fig, ax = plt.subplots()
#
# ax.bar(labels, percentages, color='lightsteelblue')
# plt.xlabel('Age')
# plt.ylabel('Percentage of those with RTIs')
# plt.title(f'{yearsrun} year model run, N={popsize}: Age demographic distribution of RTIs')
# plt.show()
itemcodes = [1, 2, 3]
available = [True, True, True]
zip_iterator = zip(itemcodes, available)
boolDict = dict(zip_iterator)
if all(value == 1 for value in boolDict.values()):
    print("Everything is available")
