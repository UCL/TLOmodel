import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np

# df = pd.read_csv('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/poppropsdf.csv')
#
# columns = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
#            'rt_injury_7', 'rt_injury_8']
# persons_injuries = df[columns]
# injured = persons_injuries.loc[df['rt_injury_1'] != "none"]
# multiple_injured = injured.loc[injured['rt_injury_2'] != "none"]
# print(len(injured), len(multiple_injured))
# selected_for_treated_injuries = injured.sample(n=3)
#
# broken_leg = multiple_injured.apply(lambda row: row.astype(str).str.contains('812').any(), axis=1)
# injury_columns = multiple_injured.columns[(multiple_injured.values == '812').any(0)].tolist()
# injury_rows = multiple_injured.apply(lambda row: row.astype(str).str.contains('812').any(), axis=1)



# for col in injured.columns:
#     subdf = injured[[col]]
#     broked_leg = subdf.apply(lambda row: row.astype(str).str.contains('812').any(), axis=1)

# print(df.loc[11, ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4']])
#
#
# def treated_injuries(dataframe, person_id, tloinjcodes):
#     cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
#             'rt_injury_7', 'rt_injury_8']
#     person_injuries = dataframe.loc[[person_id], cols]
#
#     for code in tloinjcodes:
#         injury_cols = person_injuries.columns[(person_injuries.values == code).any(0)].tolist()
#         dataframe.loc[person_id, injury_cols] = "none"
#
#
# codes = ['812', '7101']
# treated_injuries(df, 11, codes)
# print(df.loc[11, ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4']])

# simeans = [0.3, 0.8, 1.2, 1.6]
# TBImeans = [24.9, 28.4, 36.2, 43]
# nonTBImeans = [13.3, 15.8, 23.4, 30.6]
# TBI = plt.scatter(TBImeans, simeans)
# nonTBI = plt.scatter(nonTBImeans, simeans)
# plt.legend((TBI, nonTBI), ('TBI', 'Non-TBI'))
# plt.xlabel('Mean ISS score')
# plt.ylabel('Mean SI score')
# plt.show()
ISS_score = np.linspace(1, 75, 75)
plt.plot(ISS_score, 0.247 * ISS_score)
plt.xlabel('ISS score')
plt.ylabel('Shock Index')
plt.show()
