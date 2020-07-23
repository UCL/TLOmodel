from pathlib import Path
import numpy as np
from matplotlib.sankey import Sankey
import seaborn as sns
from matplotlib import cm
from floweaver import *
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pandas as pd
# import os
from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    dx_algorithm_child,
    enhanced_lifestyle,
    labour,
    newborn_outcomes,
    pregnancy_supervisor,
    contraception,
    male_circumcision,
    hiv,
    hiv_behaviour_change,
    tb,
    tb_hs_engagement,
    antenatal_care,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    symptommanager,
    rti,
    epilepsy,
    oesophageal_cancer,

)


from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

# The Resource files [NB. Working directory must be set to the root of TLO: TLOmodel]
resourcefilepath = Path('./resources')
# Establish the simulation object
yearsrun = 2
start_date = Date(year=2010, month=1, day=1)
end_date = Date(year=(2010 + yearsrun), month=1, day=1)
popsize = 10000

sim = Simulation(start_date=start_date)
logfile = sim.configure_logging(filename="LogFile")
# if os.path.exists(logfile):
#     os.remove(logfile)
# Make all services available:
service_availability = ['*']
logging.getLogger('tlo.methods.RTI').setLevel(logging.DEBUG)

# Register the appropriate 'core' modules, can register y/n. Runs without issue? Y,N
sim.register(demography.Demography(resourcefilepath=resourcefilepath),  # y, Y
             # contraception.Contraception(resourcefilepath=resourcefilepath),  # y, Y
             enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),  # y, Y
             healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                       service_availability=service_availability,
                                       mode_appt_constraints=2,
                                       capabilities_coefficient=1.0,
                                       ignore_cons_constraints=False,
                                       disable=False),  # y, Y
             symptommanager.SymptomManager(resourcefilepath=resourcefilepath),  # y, Y
             healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),  # y, Y
             dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),  # y, Y
             healthburden.HealthBurden(resourcefilepath=resourcefilepath),  # y, Y
             epilepsy.Epilepsy(resourcefilepath=resourcefilepath),  # y, Y
             oesophageal_cancer.Oesophageal_Cancer(resourcefilepath=resourcefilepath),  # y, Y
             # labour.Labour(resourcefilepath=resourcefilepath),  # y, Y
             # newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),  # y, N
             # pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),  # y, Y
             # antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),  # y, Y
             # hiv.hiv(resourcefilepath=resourcefilepath),  # y, N (Issue with requesting consumables)
             # hiv_behaviour_change.BehaviourChange,  # n
             # male_circumcision.male_circumcision(resourcefilepath=resourcefilepath),  # y, Y (Issue with requesting
             # consumables)
             # tb.tb(resourcefilepath=resourcefilepath),  # y, N (probably because of HIV not working without issue)
             # tb_hs_engagement.health_system_tb,  # n
             rti.RTI(resourcefilepath=resourcefilepath)  # y, Y
             )

# custom_levels = {
#     # '*': logging.CRITICAL,  # disable logging for all modules
#     'tlo.methods.RTI': logging.INFO,  # enable logging at INFO level
#     'tlo.methods.RTI': logging.DEBUG
#                   }

# Run the simulation
sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# Read the results
output = parse_log_file(logfile)
output['tlo.methods.healthsystem']['HSI_Event'].to_csv(
    'C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/outputdf.csv')
rt_df = output['tlo.methods.rti']['summary_1m']
requested_pain_management = output['tlo.methods.rti']['Requested_Pain_Management']
successful_pain_management = output['tlo.methods.rti']['Successful_Pain_Management']
requested_pain_management. \
    to_csv('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/requested_pain.csv')
successful_pain_management. \
    to_csv('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/succesful_pain.csv')
deaths_df = output['tlo.methods.demography']['death']
deaths_df['date'] = pd.to_datetime(deaths_df['date'])
newdf = deaths_df[['person_id', 'date', 'cause']]
newdf.to_csv('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/deathsdf.csv')
deaths_df['year'] = deaths_df['date'].dt.year
death_by_cause = deaths_df.groupby(['year', 'cause'])['person_id'].size()
rt_df.to_csv('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/rti_summary_df.csv')

# death_with_medical = death_by_cause.get_group('death_with_med')
# imm_death = death_by_cause.get_group('RTI_imm_death')
# print(len(death_with_medical), len(imm_death))

data = np.genfromtxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/Injcategories.txt')

fig, ax = plt.subplots()
theme = plt.get_cmap('Pastel2')

ax.set_prop_cycle("color", [theme(1. * i / len(data))
                            for i in range(len(data))])
labels = ['Fractures', 'Dislocations', 'TBI', 'Soft tiss', 'Int. org', 'Int. bleed',
          'SCI', 'Amputation', 'Eye injury', 'Skin wounds', 'Burns']
explode = [0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0]
ax.pie(data, explode=explode, labels=labels, autopct='%1.1f%%',
       shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title(f"{yearsrun} year model run, N={popsize}: injury categories")
plt.savefig('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/InjuryCategoriesPie.png')
plt.clf()

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],
                     title=f"{yearsrun} year model run, N={popsize}: injury characteristics")
# injcategories = [self.totfracnumber, self.totdisnumber, self.tottbi, self.totsoft, self.totintorg,
#                          self.totintbled, self.totsci, self.totamp, self.toteye, self.totextlac]
sankey = Sankey(ax=ax,
                scale=data[0] / (data[0] * data[0]),
                offset=0.2,
                format='%d')

sankey.add(flows=[sum(data), -data[0], -data[1], -data[2], -data[3], -data[4],
                  - data[5], -data[6], -data[7], -data[8], -data[9], - data[10]],
           labels=['Number of injuries', 'Fractures', 'Dislocations', 'TBI', 'Soft tiss', 'Int. org', 'Int. bleed',
                   'SCI', 'Amputation', 'Eye injury', 'Skin wounds', 'Burns'],
           orientations=[0, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],  # arrow directions
           pathlengths=[0.1, 0.5, 1.2, 0.5, 0.5, 0.8, 0.8, 0.5, 0.4, 0.8, 0.5, 1.2],
           trunklength=2,
           edgecolor='lightblue',
           facecolor='lightblue')
sankey.finish()
plt.savefig('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/InjuryCharacteristics.png')
plt.clf()

data = np.genfromtxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/RTIflow.txt')
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],
                     title=f"{yearsrun} year model run, N={popsize} RTI summary")

fig, ax = plt.subplots()
data = np.genfromtxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/Injlocs.txt')
theme = plt.get_cmap('Pastel2')

ax.set_prop_cycle("color", [theme(1. * i / len(data))
                            for i in range(len(data))])
labels = ['head', 'face', 'neck', 'thorax', 'abdomen', 'spine', 'upper x', 'lower x']
explode = [0, 0, 0, 0, 0, 0.2, 0, 0]
ax.pie(data, explode=explode, labels=labels, autopct='%1.1f%%',
       shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title(f"{yearsrun} year model run, N={popsize} AIS body regions")
plt.savefig('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/AISregions.png')
plt.clf()

data = np.genfromtxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/Injnumber.txt')
plt.bar(['1', '2', '3', '4', '5', '6', '7', '8'], data / sum(data), color='lightsteelblue')
plt.xlabel('Number of injured body regions')
plt.ylabel('Frequency')
plt.title(f'{yearsrun} year model run, N={popsize}:'
          '\n'
          r'Distribution of number of injured body regions')
plt.savefig('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/NumberofInjuries.png')

plt.clf()
fig, ax = plt.subplots()
data = np.genfromtxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/Injsev.txt')
theme = plt.get_cmap('Pastel2')

ax.set_prop_cycle("color", [theme(1. * i / len(data))
                            for i in range(len(data))])
labels = ['mild', 'severe']
explode = [0, 0]
ax.pie(data, explode=explode, labels=labels, autopct='%1.1f%%',
       shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title(f'{yearsrun} year model run, N={popsize} Distribution of injury severity')
plt.savefig('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/InjurySeverity.png')
plt.clf()
data = np.genfromtxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/ISSscores.txt')
scores, counts = np.unique(data, return_counts=True)
fig, ax = plt.subplots()

ax.bar(scores, counts / sum(counts), color='lightsteelblue')
plt.xlabel('ISS scores')
plt.ylabel('Frequency')
plt.title(f'{yearsrun} year model run, N={popsize}: Distribution of ISS scores')
plt.savefig('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/ISSscoreDistribution.png')

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
AandEFlow = [len(FirstAandE), -len(FirstAandEran), -len(FirstAandEDelay)]
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
OutcomesFlows = [len(healthappointmentran), -(len(healthappointmentran) - diedaftermed -
                                              rt_df['number permanently disabled'].max()), -diedaftermed,
                 -rt_df['number permanently disabled'].max()]
print(OutcomesFlows)
OutcomesLabels = ['', 'Survived', 'Died after medical intervention', 'Permanently disabled']
OutcomesPathLength = [0.25, 0.25, 0.1, 0.1]
OutcomesOrientations = [0, 0, 1, -1]

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],
                     title=f"{yearsrun} year model run, N={popsize}: model flow")
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

data = \
    pd.read_csv('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/GBD2017InjuryCategories.csv')
labels = data.columns.tolist()
weights = data.iloc[0, :].tolist()

fig, ax = plt.subplots()
theme = plt.get_cmap('Pastel2')

ax.set_prop_cycle("color", [theme(1. * i / len(weights))
                            for i in range(len(weights))])

explode = [0, 0, 0, 0.2, 0, 0, 0, 0]
ax.pie(weights, explode=explode, labels=labels, autopct='%1.1f%%',
       shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("GBD 2017 Malawi road traffic injury categories")
plt.savefig('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/GBDCategoriesPie.png')
plt.clf()

# ================================= Visualise pain management ==========================================================
req = np.genfromtxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/AllPainReliefRequests.txt')
suc = pd.read_csv('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/succesful_pain.csv')

mild = req[0]
sucmild = suc.loc[suc['pain level'] == "mild"]
mod = req[1]
sucmod = suc.loc[suc['pain level'] == "moderate"]
sev = req[2]
sucsev = suc.loc[suc['pain level'] == "severe"]
flows1 = [sum(req), -mild, - mod, - sev]
labels1 = ["Total requests"
           "\n"
           "for pain management",
           "Requests for"
           "\n"
           "mild pain relief",
           "Requests for"
           "\n"
           "moderate pain relief",
           "Requests for"
           "\n"
           "severe pain relief"
           ]
orientations1 = [0, 1, 0, -1]
PathLengths1 = [0.25, 0.25, 0.25, 0.25]
print(len(sucmild))
mildflow = [mild, -len(sucmild), - (mild - len(sucmild))]
mildlabels = ['', 'Received pain relief', 'No pain relief available']
mildorientations = [-1, 0, 1]
mildlengths = [0.25, 0.25, 0.25]
modflow = [mod, -len(sucmod), - (mod - len(sucmod))]
modlabels = ['', 'Received pain relief', 'No pain relief available']
modorientations = [0, 0, 1]
modlengths = [0.25, 0.25, 0.25]
sevflow = [sev, -len(sucsev), - (sev - len(sucsev))]
sevlabels = ['', 'Received pain relief', 'No pain relief available']
sevorientations = [1, 0, 1]
sevlengths = [0.25, 0.25, 0.25]
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],
                     title=f"{yearsrun} year model run, N={popsize}: pain management flow")
sankey = Sankey(ax=ax,
                scale=flows1[0] / (flows1[0] * flows1[0]),
                offset=0.2,
                format='%d')
sankey.add(flows=flows1,
           labels=labels1,
           orientations=orientations1,  # arrow directions
           pathlengths=PathLengths1,
           trunklength=1,
           edgecolor='red',
           facecolor='red')
sankey.add(flows=mildflow,
           labels=mildlabels,
           orientations=mildorientations,  # arrow directions
           pathlengths=mildlengths,
           prior=0,
           connect=(1, 0),
           trunklength=0.5,
           edgecolor='gold',
           facecolor='gold')
sankey.add(flows=modflow,
           labels=modlabels,
           orientations=modorientations,  # arrow directions
           pathlengths=modlengths,
           prior=0,
           connect=(2, 0),
           trunklength=0.5,
           edgecolor='darkgreen',
           facecolor='darkgreen')
sankey.add(flows=sevflow,
           labels=sevlabels,
           orientations=sevorientations,  # arrow directions
           pathlengths=sevlengths,
           prior=0,
           connect=(3, 0),
           trunklength=0.5,
           edgecolor='slategray',
           facecolor='slategray')
sankey.finish()
plt.savefig('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/RTIPainManagementFlow.png')
plt.clf()

# =========================== Plot where injuries occured on body ======================================================

data = np.genfromtxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/Injlocs.txt')

def main():
    try:
        img = Image.open('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/bodies-cropped.jpg')
        # img = img.filter(ImageFilter.SHARPEN)
        thresh = 230
        fn = lambda x: 255 if x > thresh else 0
        img = img.convert('L').point(fn, mode='1')
        # img = img.convert('1')
        #

        font_path = "C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/Fonts/BOD_R.TTF"
        fnt = ImageFont.truetype(font_path, 15)
        font_path = "C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/Fonts/BOD_B.TTF"
        titlefnt = ImageFont.truetype(font_path, 20)
        d = ImageDraw.Draw(img)
        d.text((0, 10), f"The distribution of injury location: {yearsrun} year model run,"
                        "\n"
                        f"population size = {popsize}", font=titlefnt, fill='black')
        d.text((120, 80), "Head:"
                          "\n"
                          f"{round(data[0] / sum(data), 2) * 100} %", font=fnt, fill='black')
        d.line(xy=[170, 90, 200, 90], fill='black', width=1)
        d.text((300, 100), "Face:"
                           "\n"
                           f"{round(data[1] / sum(data), 2) * 100} %", font=fnt, fill='black')
        d.line(xy=[230, 110, 270, 110], fill='black', width=1)

        d.text((120, 120), "Neck:"
                           "\n"
                           f"{round(data[2] / sum(data), 2) * 100} %", font=fnt, fill='black')
        d.line(xy=[170, 140, 210, 150], fill='black', width=1)

        d.text((200, 180), "Thorax:"
                           "\n"
                           f"{round(data[3] / sum(data), 2) * 100} %", font=fnt, fill='black')
        d.text((205, 250), "Spine:"
                           "\n"
                           f"{round(data[5] / sum(data), 2) * 100} %", font=fnt, fill='black')
        d.text((205, 300), "Abdomen"
                           "\n"
                           f"{round(data[4] / sum(data), 2) * 100} %", font=fnt, fill='black')
        d.text((350, 220), "Upper"
                           "\n"
                           "extremity:"
                           "\n"
                           f"{round(data[6] / sum(data), 2) * 100} %", font=fnt, fill='black')
        d.line(xy=[340, 240, 160, 220], fill='black', width=1)
        d.line(xy=[340, 240, 300, 260], fill='black', width=1)

        d.text((300, 420), "Lower"
                           "\n"
                           "extremity:"
                           "\n"
                           f"{round(data[7] / sum(data), 2) * 100} %", font=fnt, fill='black')
        d.line(xy=[290, 440, 200, 440], fill='black', width=1)
        d.line(xy=[290, 440, 260, 540], fill='black', width=1)
        img.save('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/ModelInjuryLocationOnBody.jpg')

    except IOError:
        pass


if __name__ == "__main__":
    main()

data = np.genfromtxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/OpenWoundDistribution.txt')


def main():
    try:
        img = Image.open('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/bodies-cropped.jpg')
        # img = img.filter(ImageFilter.SHARPEN)
        thresh = 230
        fn = lambda x: 255 if x > thresh else 0
        img = img.convert('L').point(fn, mode='1')
        # img = img.convert('1')
        #

        font_path = "C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/Fonts/BOD_R.TTF"
        fnt = ImageFont.truetype(font_path, 15)
        font_path = "C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/Fonts/BOD_B.TTF"
        titlefnt = ImageFont.truetype(font_path, 20)
        d = ImageDraw.Draw(img)
        d.text((0, 10), f"The distribution of open wound location: {yearsrun} year model run,"
                        "\n"
                        f"population size = {popsize}", font=titlefnt, fill='black')
        d.text((120, 80), "Head:"
                          "\n"
                          f"{round(data[0] / sum(data), 2) * 100} %", font=fnt, fill='black')
        d.line(xy=[170, 90, 200, 90], fill='black', width=1)
        d.text((300, 100), "Face:"
                           "\n"
                           f"{round(data[1] / sum(data), 2) * 100} %", font=fnt, fill='black')
        d.line(xy=[230, 110, 270, 110], fill='black', width=1)

        d.text((120, 120), "Neck:"
                           "\n"
                           f"{round(data[2] / sum(data), 2) * 100} %", font=fnt, fill='black')
        d.line(xy=[170, 140, 210, 150], fill='black', width=1)

        d.text((200, 180), "Thorax:"
                           "\n"
                           f"{round(data[3] / sum(data), 2) * 100} %", font=fnt, fill='black')
        d.text((205, 250), "Spine:"
                           "\n"
                           f"{round(data[5] / sum(data), 2) * 100} %", font=fnt, fill='black')
        d.text((205, 300), "Abdomen"
                           "\n"
                           f"{round(data[4] / sum(data), 2) * 100} %", font=fnt, fill='black')
        d.text((350, 220), "Upper"
                           "\n"
                           "extremity:"
                           "\n"
                           f"{round(data[6] / sum(data), 2) * 100} %", font=fnt, fill='black')
        d.line(xy=[340, 240, 160, 220], fill='black', width=1)
        d.line(xy=[340, 240, 300, 260], fill='black', width=1)

        d.text((300, 420), "Lower"
                           "\n"
                           "extremity:"
                           "\n"
                           f"{round(data[7] / sum(data), 2) * 100} %", font=fnt, fill='black')
        d.line(xy=[290, 440, 200, 440], fill='black', width=1)
        d.line(xy=[290, 440, 260, 540], fill='black', width=1)
        img.save('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/OpenWoundLocationOnBody.jpg')

    except IOError:
        pass


if __name__ == "__main__":
    main()

data = np.genfromtxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/FractureDistribution.txt')


def main():
    try:
        img = Image.open('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/bodies-cropped.jpg')
        # img = img.filter(ImageFilter.SHARPEN)
        thresh = 230
        fn = lambda x: 255 if x > thresh else 0
        img = img.convert('L').point(fn, mode='1')
        # img = img.convert('1')
        #

        font_path = "C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/Fonts/BOD_R.TTF"
        fnt = ImageFont.truetype(font_path, 15)
        font_path = "C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/Fonts/BOD_B.TTF"
        titlefnt = ImageFont.truetype(font_path, 20)
        d = ImageDraw.Draw(img)
        d.text((0, 10), f"The distribution of fracture location: {yearsrun} year model run,"
                        "\n"
                        f"population size = {popsize}", font=titlefnt, fill='black')
        d.text((120, 80), "Head:"
                          "\n"
                          f"{round(data[0] / sum(data), 2) * 100} %", font=fnt, fill='black')
        d.line(xy=[170, 90, 200, 90], fill='black', width=1)
        d.text((300, 100), "Face:"
                           "\n"
                           f"{round(data[1] / sum(data), 2) * 100} %", font=fnt, fill='black')
        d.line(xy=[230, 110, 270, 110], fill='black', width=1)

        d.text((120, 120), "Neck:"
                           "\n"
                           f"{round(data[2] / sum(data), 2) * 100} %", font=fnt, fill='black')
        d.line(xy=[170, 140, 210, 150], fill='black', width=1)

        d.text((200, 180), "Thorax:"
                           "\n"
                           f"{round(data[3] / sum(data), 2) * 100} %", font=fnt, fill='black')
        d.text((205, 250), "Spine:"
                           "\n"
                           f"{round(data[5] / sum(data), 2) * 100} %", font=fnt, fill='black')
        d.text((205, 300), "Abdomen"
                           "\n"
                           f"{round(data[4] / sum(data), 2) * 100} %", font=fnt, fill='black')
        d.text((350, 220), "Upper"
                           "\n"
                           "extremity:"
                           "\n"
                           f"{round(data[6] / sum(data), 2) * 100} %", font=fnt, fill='black')
        d.line(xy=[340, 240, 160, 220], fill='black', width=1)
        d.line(xy=[340, 240, 300, 260], fill='black', width=1)

        d.text((300, 420), "Lower"
                           "\n"
                           "extremity:"
                           "\n"
                           f"{round(data[7] / sum(data), 2) * 100} %", font=fnt, fill='black')
        d.line(xy=[290, 440, 200, 440], fill='black', width=1)
        d.line(xy=[290, 440, 260, 540], fill='black', width=1)

        img.save('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/FractureLocationOnBody.jpg')

    except IOError:
        pass


if __name__ == "__main__":
    main()

data = np.genfromtxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/BurnDistribution.txt')


def main():
    try:
        img = Image.open('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/bodies-cropped.jpg')
        # img = img.filter(ImageFilter.SHARPEN)
        thresh = 230
        fn = lambda x: 255 if x > thresh else 0
        img = img.convert('L').point(fn, mode='1')
        # img = img.convert('1')
        #

        font_path = "C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/Fonts/BOD_R.TTF"
        fnt = ImageFont.truetype(font_path, 15)
        font_path = "C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/Fonts/BOD_B.TTF"
        titlefnt = ImageFont.truetype(font_path, 20)
        d = ImageDraw.Draw(img)
        d.text((0, 10), f"The distribution of burn location: {yearsrun} year model run,"
                        "\n"
                        f"population size = {popsize}", font=titlefnt, fill='black')
        d.text((120, 80), "Head:"
                          "\n"
                          f"{round(data[0] / sum(data), 2) * 100} %", font=fnt, fill='black')
        d.line(xy=[170, 90, 200, 90], fill='black', width=1)
        d.text((300, 100), "Face:"
                           "\n"
                           f"{round(data[1] / sum(data), 2) * 100} %", font=fnt, fill='black')
        d.line(xy=[230, 110, 270, 110], fill='black', width=1)

        d.text((120, 120), "Neck:"
                           "\n"
                           f"{round(data[2] / sum(data), 2) * 100} %", font=fnt, fill='black')
        d.line(xy=[170, 140, 210, 150], fill='black', width=1)

        d.text((200, 180), "Thorax:"
                           "\n"
                           f"{round(data[3] / sum(data), 2) * 100} %", font=fnt, fill='black')
        d.text((205, 250), "Spine:"
                           "\n"
                           f"{round(data[5] / sum(data), 2) * 100} %", font=fnt, fill='black')
        d.text((205, 300), "Abdomen"
                           "\n"
                           f"{round(data[4] / sum(data), 2) * 100} %", font=fnt, fill='black')
        d.text((350, 220), "Upper"
                           "\n"
                           "extremity:"
                           "\n"
                           f"{round(data[6] / sum(data), 2) * 100} %", font=fnt, fill='black')
        d.line(xy=[340, 240, 160, 220], fill='black', width=1)
        d.line(xy=[340, 240, 300, 260], fill='black', width=1)

        d.text((300, 420), "Lower"
                           "\n"
                           "extremity:"
                           "\n"
                           f"{round(data[7] / sum(data), 2) * 100} %", font=fnt, fill='black')
        d.line(xy=[290, 440, 200, 440], fill='black', width=1)
        d.line(xy=[290, 440, 260, 540], fill='black', width=1)

        img.save('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/BurnLocationOnBody.jpg')

    except IOError:
        pass


if __name__ == "__main__":
    main()

# ======================================= Plot Demographics ===========================================================
data = pd.read_csv('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/RTIInjuryDemographics.csv')
genderData = data['sex']
values, counts = np.unique(genderData, return_counts=True)

fig, ax = plt.subplots()

ax.bar(values, counts / sum(counts), color='lightsteelblue')
plt.xlabel('Gender')
plt.ylabel('Percentage of those with RTIs')
plt.title(f'{yearsrun} year model run, N={popsize}: Gender demographic distribution of RTIs')
plt.savefig('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/InRTIGenderDistribution.png')
plt.clf()

childrenCounts = len(data.loc[data['age_years'].between(0, 17, inclusive=True)])
youngAdultCounts = len(data.loc[data['age_years'].between(18, 29, inclusive=True)])
thirtiesCounts = len(data.loc[data['age_years'].between(30, 39, inclusive=True)])
fourtiesCounts = len(data.loc[data['age_years'].between(40, 49, inclusive=True)])
fiftiesAndSixtiesCounts = len(data.loc[data['age_years'].between(50, 69, inclusive=True)])
seventiesPlus = len(data.loc[data['age_years'] >= 70])
counts = [childrenCounts, youngAdultCounts, thirtiesCounts, fourtiesCounts, fiftiesAndSixtiesCounts, seventiesPlus]
percentages = np.divide(counts, sum(counts))
labels = ['0-17', '18-29', '30-39', '40-49', '50-69', '70+']
fig, ax = plt.subplots()
ax.bar(labels, percentages, color='lightsteelblue')
plt.xlabel('Age')
plt.ylabel('Percentage of those with RTIs')
plt.title(f'{yearsrun} year model run, N={popsize}: Age demographic distribution of RTIs')
plt.savefig('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/InRTIAgeDistribution.png')
plt.clf()
