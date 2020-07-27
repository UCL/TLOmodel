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
# # plt.show()
# itemcodes1 = ['1', '2', '3']
# quantity1 = [1, 2, 3]
# itemcodes2 = ['4', '5', '6']
# quantity2 = [4, 5, 6]
# zip_iterator1 = zip(itemcodes1, quantity1)
# zip_iterator2 = zip(itemcodes2, quantity2)
# boolDict1 = dict(zip_iterator1)
# boolDict2 = dict(zip_iterator2)
# print(boolDict1, boolDict2)
# boolDict1 = {**boolDict1, **boolDict2}
# print(boolDict1)
# zip_iterator = zip(itemcodes, available)
# boolDict = dict(zip_iterator)
# print(boolDict)
# if all(value == 1 for value in boolDict.values()):
#     print("Everything is available")
#
# consumabledict1 = {dict(), {'item1': 1}}
# df = pd.read_csv('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/population_props.csv')
#
#
# # found = df[df['Column'].str.contains('Text_to_search')]
# def rti_make_injuries_permanent(df, codes):
#     # df = self.sim.population.props
#     columns = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
#                'rt_injury_7', 'rt_injury_8']
#     persons_injuries = df.loc[[91], columns]
#     injury_numbers = range(1, 9)
#     for code in codes:
#         new_code = "P" + code
#         for injury_number in injury_numbers:
#             found = df[df[f"rt_injury_{injury_number}"].str.contains(code)]
#             if len(found) > 0:
#                 df.loc[[91], f"rt_injury_{injury_number}"] = new_code
#
#
# print(rti_make_injuries_permanent(df, ['813', '456']))
# columns = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
#                'rt_injury_7', 'rt_injury_8']
# persons_injuries = df.loc[[91], columns]
# print(persons_injuries)
# a = range(1,9)
# for num in a:
#     print(num)

df = pd.DataFrame({'rt_injury_1': ['8101', '712'],
                   'rt_injury_2': ['8111', '552'],
                   'rt_med_int': [["HSI_RTI_Major_Surgeries", "HSI_RTI_MedicalIntervention"],
                                  ["HSI_RTI_Fracture_Cast", "HSI_R TI_Suture"]]}, index=[1, 2])
injuries = df[['rt_injury_1', 'rt_injury_2']]
treatments = df['rt_med_int']

# .apply(lambda row: row.astype(str).str.contains('HSI_RTI_Suture').any(), axis=1)
# print("Injuries")
# print(injuries)
print("Treatments")
print(df.rt_med_int)
mask = df.rt_med_int.apply(lambda x: 'HSI_RTI_Suture' in x)
print(sum(mask))

# treatmentOccurs = treatments.apply(lambda row: row.astype(list).str.contains('HSI_RTI_Suture').any(), axis=1)
# injuryOccurred = injuries.apply(lambda row: row.astype(str).str.contains('8111').any(), axis=1)
# print(treatmentOccurs)
# print(injuryOccurred)
