import pandas as pd
import numpy as np
from matplotlib.sankey import Sankey
from floweaver import *
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

df = pd.read_csv('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/outputdf.csv')
rt_df = pd.read_csv('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/rti_summary_df.csv')
newdf = pd.read_csv('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/deathsdf.csv')
data = np.genfromtxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/RTIflow.txt')
diedaftermed = len(newdf.loc[newdf['cause'] == 'RTI_death_with_med'])
diedimm = len(newdf.loc[newdf['cause'] == 'RTI_imm_death'])
diedwithoutmed = newdf.loc[newdf['cause'] == 'RTI_death_without_med']
numdiedwithoutmed = diedwithoutmed['person_id'].nunique()
healthappointment = df.loc[(df['TREATMENT_ID'] == 'RTI_MedicalIntervention')]
healthappointment = healthappointment.loc[healthappointment.did_run]
rti_health_appointment = len(healthappointment)
df = pd.read_csv('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/outputdf.csv')
df = df.loc[df['TREATMENT_ID'] == 'RTI_MedicalIntervention']
df_did_run = df.loc[df.did_run]
df_did_not_run = df.loc[~df.did_run]
No_treatment_available = healthappointment.loc[~healthappointment.did_run]
soughthealthcare = len(df_did_run)
persons_delayed_care = df_did_not_run['Person_ID'].nunique()
diedwithouthealthcare = len(newdf.loc[newdf['cause'] == 'RTI_death_without_med'])
ordering = [['start'], ['end']]
nodes = {
    'start': ProcessGroup(['Number of injured persons']),
    'end': ProcessGroup(list(['Died on scene', 'Sought health care']))
}
bundles = [Bundle('start', 'end')]
nodes['start'].partition = Partition.Simple('source', ['Number of injured persons'])
nodes['end'].partition = Partition.Simple('target', list(['Died on scene', 'Sought health care']))
sdd = SankeyDefinition(nodes, bundles, ordering)
# fig = plt.figure(figsize=(20, 10))
# ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],
#                      title=f"{2} year model run, N={5000}: model flow")
# sankey = Sankey(ax=ax,
#                 scale=data[0] / (data[0] * data[0]),
#                 offset=0.2,
#                 format='%d')
#
# sankey.add(flows=[data[0], - persons_delayed_care, -diedimm, -soughthealthcare],
#            labels=['Number of injured persons', 'Delay in care', 'Died on scene', 'Sought health care without delay'],
#            orientations=[0, 1, -1, 0],  # arrow directions
#            pathlengths=[0.4, 0.2, 0.1, 0.1],
#            trunklength=0.5,
#            edgecolor='#027368',
#            facecolor='#027368')
# sankey.add(flows=[soughthealthcare, -diedaftermed, -data[4], -(soughthealthcare - diedaftermed -
#                                                                                       data[4])],
#            labels=['', 'Died after treatment', 'Treated but still disabled', 'Recovered'],
#            prior=0,
#            connect=(1, 0),
#            orientations=[0, 1, -1, 0],
#            pathlengths=[0.4, 0.2, 0.2, 0.1],
#            trunklength=0.5,
#            edgecolor='#58A4B0',
#            facecolor='#58A4B0')
# sankey.add(flows=[persons_delayed_care, - numdiedwithoutmed, -(persons_delayed_care-numdiedwithoutmed)],
#            labels=['', 'Died without access to care', 'Sought care with delay'],
#            prior=0,
#            connect=(1, 0),
#            orientations=[-1, 1, -1],
#            pathlengths=[0.2, 0.4, 0.2],
#            trunklength=0.2,
#            edgecolor='#028368',
#            facecolor='#028368'
#            )
# sankey.add(flows=[No_treatment_available, -diedwithouthealthcare, -(No_treatment_available - diedwithouthealthcare)],
#            labels=['', 'Died after not receiving treatment', 'recovered without treatment'],
#            prior=1,
#            connect=(2, 1),
#            orientations=[0, 1, 0],
#            pathlengths=[0.4, 0.2, 0.2],
#            trunklength=0.5,
#            edgecolor='#022368',
#            facecolor='#022368'
#            )
#
# sankey.finish()
# plt.show()
# plt.savefig('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/RTIModelFlow.png')
# plt.clf()
