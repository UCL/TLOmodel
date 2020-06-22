import pandas as pd
import numpy as np
from matplotlib.sankey import Sankey
from floweaver import *
import matplotlib

# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

df = pd.read_csv('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/outputdf.csv')
rt_df = pd.read_csv('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/rti_summary_df.csv')
newdf = pd.read_csv('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/deathsdf.csv')
data = np.genfromtxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/RTIflow.txt')
print('Number involved in a rti')
NumberInRTI = rt_df['number involved in a rti'].sum()
print(NumberInRTI)
diedaftermed = len(newdf.loc[newdf['cause'] == 'RTI_death_with_med'])
print('Number who died immediately')
diedimm = len(newdf.loc[newdf['cause'] == 'RTI_imm_death'])
print(diedimm)
diedwithoutmed = newdf.loc[newdf['cause'] == 'RTI_death_without_med']
numdiedwithoutmed = diedwithoutmed['person_id'].nunique()
print('Number who sought A and E')
FirstAandE = df.loc[df['TREATMENT_ID'] == 'GenericEmergencyFirstApptAtFacilityLevel1']
FirstAandEran = FirstAandE.did_run
print(len(FirstAandE))
print('Number seen without delay')
print(len(FirstAandEran))
FirstAandEDelay = FirstAandE.loc[FirstAandE['did_run'] == False]
print('Number delayed in their A&E appointment')
print(len(FirstAandEDelay))
PostAandEFlow = [len(FirstAandEran)]
allhealthappointment = df.loc[(df['TREATMENT_ID'] == 'RTI_MedicalIntervention')]
print('Treatment recieved without delay')
healthappointmentran = allhealthappointment.loc[allhealthappointment['did_run'] == True]
print(len(healthappointmentran))
print('Number of people with delayed treatment')
healthappointmentdelay = allhealthappointment.loc[allhealthappointment['did_run'] == False]
numberofpersonsdelayed = len(healthappointmentdelay.Person_ID.unique())
print('First flow')
PreAandEFlow = [NumberInRTI, -len(FirstAandE), -diedimm]
print(PreAandEFlow)

PreAandELabels = ["Number "
                  "\n"
                  "involved"
                  "\n"
                  "in road traffic"
                  "\n"
                  " accident", "Sought "
                               "\n"
                               "health care", 'Died on scene']
PreAandEOrientation = [0, 0, 1]
PreAandEPathLength = [0.25, 0.25, 0.1]
print('Second flow')
AandEFlow = [len(FirstAandE),  -len(FirstAandEran), -len(FirstAandEDelay)]
print(AandEFlow)
AandELabels = ['', "Received "
                   "\n"
                   "A&E appointment "
                   "\n"
                   "without delay", 'Delayed in initial appointment']
AandEOrientations = [0, 0, 1]
AandEPathLength = [0.25, 0.25, 0.1]
print('Third flow')
HealthSystemFlows = [len(FirstAandEran), - len(healthappointmentran), -numberofpersonsdelayed]
print(HealthSystemFlows)
HealthSystemLabels = ['', "Received "
                          "\n"
                          "treatment "
                          "\n"
                          "without delay",
                      "Delay in "
                      "\n"
                      "receiving treatment"]
HealthSystemPathLength = [0.25, 0.25, 0.1]
HealthSystemOrientations = [0, 0, 1]
print('Fourth flow')
DeathWithoutMedFlows = [numberofpersonsdelayed, -len(diedwithoutmed), -(numberofpersonsdelayed - len(diedwithoutmed))]
print(DeathWithoutMedFlows)
DeathWithoutMedLabels = ['', "Died without "
                             "\n"
                             "receiving care", 'Survived with delay']
DeathWithoutMedPathLength = [0.1, 0.1, 0.1]
DeathWithoutMedOrientations = [0, -1, -1]
print('Fifth flow')
OutcomesFlows = [len(healthappointmentran), -(len(healthappointmentran)-diedaftermed -
                                              rt_df['number permanently disabled'].max()), -diedaftermed,
                 -rt_df['number permanently disabled'].max()]
print(OutcomesFlows)
OutcomesLabels = ['', 'Survived', 'Died after medical intervention', 'Permanently disabled']
OutcomesPathLength = [0.25, 0.25, 0.1, 0.1]
OutcomesOrientations = [0, 0, 1, -1]


fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],
                     title=f"{2} year model run, N={10000}: model flow")
# ax.set_facecolor('silver')
sankey = Sankey(ax=ax,
                scale=PreAandEFlow[0] / (PreAandEFlow[0] * PreAandEFlow[0]),
                offset=0.2,
                format='%d')

sankey.add(flows=PreAandEFlow,
           labels=PreAandELabels,
           orientations=PreAandEOrientation,  # arrow directions
           pathlengths=PreAandEPathLength,
           trunklength=0.5,
           edgecolor='red',
           facecolor='red')
sankey.add(flows=AandEFlow,
           labels=AandELabels,
           prior=0,
           connect=(1, 0),
           orientations=AandEOrientations,
           pathlengths=AandEPathLength,
           trunklength=0.5,
           edgecolor='gold',
           facecolor='gold')
sankey.add(flows=HealthSystemFlows,
           labels=HealthSystemLabels,
           prior=1,
           connect=(1, 0),
           orientations=HealthSystemOrientations,
           pathlengths=HealthSystemPathLength,
           trunklength=0.5,
           edgecolor='darkgreen',
           facecolor='darkgreen'
           )
sankey.add(flows=DeathWithoutMedFlows,
           labels=DeathWithoutMedLabels,
           prior=2,
           connect=(2, 0),
           orientations=DeathWithoutMedOrientations,
           pathlengths=DeathWithoutMedPathLength,
           trunklength=0.5,
           edgecolor='darkseagreen',
           facecolor='darkseagreen'
           )
sankey.add(flows=OutcomesFlows,
           labels=OutcomesLabels,
           prior=2,
           connect=(1, 0),
           orientations=OutcomesOrientations,
           pathlengths=OutcomesPathLength,
           trunklength=0.5,
           edgecolor='black',
           facecolor='black',
)
# sankey.finish()
diagrams = sankey.finish()
diagrams[1].texts[1].set_color('white')
diagrams[2].texts[1].set_color('white')

plt.savefig('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/RTIModelFlow.png')

# plt.show()

# df = pd.read_csv('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/outputdf.csv')
# df = df.loc[df['TREATMENT_ID'] == 'RTI_MedicalIntervention']
# df_did_run = df.loc[df.did_run]
# df_did_not_run = df.loc[~df.did_run]
# No_treatment_available = healthappointment.loc[~healthappointment.did_run]
# soughthealthcare = len(df_did_run)
# persons_delayed_care = df_did_not_run['Person_ID'].nunique()
# diedwithouthealthcare = len(newdf.loc[newdf['cause'] == 'RTI_death_without_med'])
# print([data[0], - persons_delayed_care, -diedimm, -soughthealthcare],
#       sum([data[0], - persons_delayed_care, -diedimm, -soughthealthcare]))
# print([soughthealthcare, -diedaftermed, -data[4], -(soughthealthcare - diedaftermed -
#                                                                                       data[4])],
#       sum([soughthealthcare, -diedaftermed, -data[4], -(soughthealthcare - diedaftermed -
#                                                                                       data[4])]))
# fig = plt.figure(figsize=(20, 10))
# ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],
#                      title=f"{2} year model run, N={5000}: model flow")
# sankey = Sankey(ax=ax,
#                 scale=data[0] / (data[0] * data[0]),
#                 offset=0.2,
#                 format='%d')
#
# sankey.add(flows=[int(data[0]), int(-soughthealthcare), int(- persons_delayed_care), int(-diedimm)],
#            labels=['Number of injured persons', 'Sought health care without delay', 'Delay in care', 'Died on scene'],
#            orientations=[0, 0, 1, -1],  # arrow directions
#            pathlengths=[0.4, 0.2, 0.1, 0.1],
#            trunklength=0.5,
#            edgecolor='blue',
#            facecolor='blue')
# testflow = [int(soughthealthcare)]
# realflow = [int(soughthealthcare), int(-diedaftermed), int(-data[4]),
#                   int(-(soughthealthcare - diedaftermed - data[4]))]
# testlabel = ['']
# reallabel = ['', 'Died after treatment', 'Treated but still disabled', 'Recovered']
# testorientation = [0]
# realorientation = [0, 1, -1, 0]
# testpathlength = [0.4]
# realpathlength = [0.4, 0.2, 0.2, 0.1]
# sankey.add(flows=realflow,
#            labels=reallabel,
#            prior=0,
#            connect=(1, 0),
#            orientations=realorientation,
#            pathlengths=realpathlength,
#            trunklength=0.5,
#            edgecolor='red',
#            facecolor='red')
# # sankey.add(flows=[persons_delayed_care, - numdiedwithoutmed, -(persons_delayed_care-numdiedwithoutmed)],
# #            labels=['', 'Died without access to care', 'Sought care with delay'],
# #            prior=0,
# #            connect=(1, 0),
# #            orientations=[-1, 1, -1],
# #            pathlengths=[0.2, 0.4, 0.2],
# #            trunklength=0.2,
# #            edgecolor='#028368',
# #            facecolor='#028368'
# #            )
# # sankey.add(flows=[No_treatment_available, -diedwithouthealthcare, -(No_treatment_available - diedwithouthealthcare)],
# #            labels=['', 'Died after not receiving treatment', 'recovered without treatment'],
# #            prior=1,
# #            connect=(2, 1),
# #            orientations=[0, 1, 0],
# #            pathlengths=[0.4, 0.2, 0.2],
# #            trunklength=0.5,
# #            edgecolor='#022368',
# #            facecolor='#022368'
# #            )
# #
# sankey.finish()
# # plt.show()
# # plt.savefig('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/RTIModelFlow.png')
# # plt.clf()
