import pandas as pd
import numpy as np

import matplotlib

matplotlib.use('TkAgg')
from matplotlib.sankey import Sankey
from matplotlib import pyplot as plt
data = pd.read_csv("C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/RTIDeathData.csv")
data = data.drop('date', axis=1)
data = data.drop(data.columns[0], axis=1)
inj1 = data['First injury'].value_counts()
inj2 = data['Second injury'].value_counts()
inj3 = data['Third injury'].value_counts()
inj4 = data['Fourth injury'].value_counts()
inj5 = data['Fifth injury'].value_counts()
inj6 = data['Sixth injury'].value_counts()
inj7 = data['Seventh injury'].value_counts()
inj8 = data['Eigth injury'].value_counts()
injuries = inj1.append([inj2, inj3, inj4, inj5, inj6, inj7, inj8])
injdict = injuries.to_dict()
injdict.pop('none')
plt.bar(range(len(injdict)), list(injdict.values()), align='center')
plt.xticks(range(len(injdict)), list(injdict.keys()))
plt.xticks(rotation=45)
plt.show()
#
# # first flow into the diagram, the first value is the total quantity introduced into the
# # flow, the remaining, the subsequent terms remove a certain quantity from the first flow.
#
# # In this example we will plot the percent health care budget consumed by various conditions, 50% is spent on HIV, 30%
# # on road traffic injuries and 20% on epilepsy, we store this information in two arrays, flows1 which houses the
# # data and labels1 which gives each percentage a label.
#
# # I want the total budget to go in a straight line from left to right, the hiv budget to go up from the total budget,
# # the road traffic budget to carry on straight and the epilepsy budget to go down, I will store these directions in an
# # array orientations1, the first entry is 0 as we don't want to change the orientations from the default direction,
# # the second entry is 1 as we want the HIV budget to go up, the third entry is 0 as we want the entry to go on straight,
# # the fourth entry is -1 as we want the epilepsy budget to go down.
# flow1 = [100, -30, -50, -20]
# labels1 = ['Total expenditure on health', '% spent on HIV', "% spent on road "
#                                                             "\n"
#                                                             "traffic injuries",
#            '% spent on epilepsy']
# orientations1 = [0, 1, 0, -1]
# # The second flow breaks down the what the road traffic injuries budget was spent on, 10% of
# # the total budget was spent on bandages, 15% on plaster of paris and 5% on surgery, we store data in flows 2 and the
# # labelling info in labels2, leaving the first entry blank as this is where the '% spent on road traffic injuries'
# # flow links to the breakdown flow.
# # In the orientations, the first entry is zero as we want this flow to carry on in the same direction, the second entry
# # for bandage expenditure is 1 as we want this to head up from the flow, the second entry is zero as we want the
# # plaster of paris expenditure to go on straight and finally the fourth entry is -1 to make the surgery expenditure go
# # down
# flow2 = [50, -10, -15, -25]
# labels2 = ['', 'bandages', 'plaster of paris', 'surgery']
# orientations2 = [0, 1, 0, -1]
#
# # Now we have created the flows and set the labels and directions the arrows go in, we can create the sankey diagram.
#
# # The sankey object needs to be scaled (controls how much space the diagram takes up), but if it's not scaled properly
# # it can look pretty terrible, I found that a fairly reasonable scale to use is to use 1/a where a is the first entry of
# # the first flow (in this example 100).
# # The offset zooms into and out from the diagram
#
# fig = plt.figure(figsize=(20, 10))  # create figure
# ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
# ax.axis('off')  # Turn off the box for the plot
# plt.title('Budget example')  # create title
# sankey = Sankey(ax=ax,
#                 scale=1 / flow1[0],
#                 offset=0.2,
#                 unit='')  # create sankey object
# sankey.add(flows=flow1,  # add the first flow
#            labels=labels1,
#            orientations=orientations1,
#            pathlengths=[0.1, 0.1, 0.1, 0.1],
#            trunklength=0.605,  # how long this flow is
#            edgecolor='#027368',  # choose colour
#            facecolor='#027368')
# sankey.add(flows=flow2,
#            labels=labels2,
#            trunklength=0.5,
#            pathlengths=[0.25, 0.25, 0.25, 0.25],
#            orientations=[0, 1, 0, -1],
#            prior=0,  # which sankey are you connecting to (0-indexed)
#            connect=(2, 0),  # flow number to connect: this is the index of road traffic injury portion of the budget in
#            # the first flow (third entry, python index 2) which connects to the first entry in the second flow (python
#            # index 0).
#            edgecolor='#58A4B0',  # choose colour
#            facecolor='#58A4B0')
# diagrams = sankey.finish()
#
# plt.show()
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

# df = pd.DataFrame({'rt_injury_1': ['8101', '712'],
#                    'rt_injury_2': ['8111', '552'],
#                    'rt_med_int': [["HSI_RTI_Major_Surgeries", "HSI_RTI_MedicalIntervention"],
#                                   ["HSI_RTI_Fracture_Cast", "HSI_R TI_Suture"]]}, index=[1, 2])
# injuries = df[['rt_injury_1', 'rt_injury_2']]
# treatments = df['rt_med_int']
#
# # .apply(lambda row: row.astype(str).str.contains('HSI_RTI_Suture').any(), axis=1)
# # print("Injuries")
# # print(injuries)
# print("Treatments")
# print(df.rt_med_int)
# mask = df.rt_med_int.apply(lambda x: 'HSI_RTI_Suture' in x)
# print(sum(mask))

# treatmentOccurs = treatments.apply(lambda row: row.astype(list).str.contains('HSI_RTI_Suture').any(), axis=1)
# injuryOccurred = injuries.apply(lambda row: row.astype(str).str.contains('8111').any(), axis=1)
# print(treatmentOccurs)
# print(injuryOccurred)
