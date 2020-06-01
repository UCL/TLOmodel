import numpy as np
import matplotlib
import pandas as pd
from matplotlib.sankey import Sankey

# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt



data = np.genfromtxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/Injcategories.txt')
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],
                     title="Two year model run, N=10,000: injury characteristics")
# injcategories = [self.totfracnumber, self.totdisnumber, self.tottbi, self.totsoft, self.totintorg,
#                          self.totintbled, self.totsci, self.totamp, self.toteye, self.totextlac]
sankey = Sankey(ax=ax,
                scale=0.0009,
                offset=0.2,
                format='%d')

sankey.add(flows=[sum(data), -data[0], -data[1], -data[2], -data[3], -data[4],
                  - data[5], -data[6], -data[7], -data[8], -data[9]],
           labels=['Number of injuries', 'Fractures', 'Dislocations', 'TBI', 'Soft tiss', 'Int. org', 'Int. bleed',
                   'SCI', 'Amputation', 'Eye injury', 'Skin wounds'],
           orientations=[0, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],  # arrow directions
           pathlengths=[0.1, 0.5, 0.8, 0.5, 0.5, 0.8, 0.8, 0.5, 0.4, 0.8, 0.5],
           trunklength=2,
           edgecolor='#027368',
           facecolor='#027368')
sankey.finish()
plt.savefig('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/InjuryCharacteristics.png')
plt.clf()

data = np.genfromtxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/RTIflow.txt')
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],
                     title="Two year model run, N=10,000: RTI summary")

fig, ax = plt.subplots()
data = np.genfromtxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/Injlocs.txt')
labels = ['head', 'face', 'neck', 'thorax', 'abdomen', 'spine', 'upper x', 'lower x']
explode = [0, 0, 0, 0, 0, 0.2, 0, 0]
ax.pie(data, explode=explode, labels=labels, autopct='%1.1f%%',
       shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Two year model run, N=10,000: AIS body regions')
plt.savefig('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/AISregions.png')
plt.clf()

data = np.genfromtxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/Injnumber.txt')
plt.bar(['1', '2', '3', '4', '5', '6', '7', '8'], data/sum(data))
plt.xlabel('Number of injured body regions')
plt.ylabel('Frequency')
plt.title(r'Two year model run, N=10,000:'
          '\n'
          r'Distribution of number of injured body regions')
plt.savefig('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/NumberofInjuries.png')

plt.clf()
fig, ax = plt.subplots()
data = np.genfromtxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/Injsev.txt')
labels = ['mild', 'severe']
explode = [0, 0]
ax.pie(data, explode=explode, labels=labels, autopct='%1.1f%%',
       shadow=True, startangle=90)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Two year model run, N=10,000: Distribution of injury severity')
plt.savefig('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/InjurySeverity.png')
plt.clf()
data = np.genfromtxt('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/ISSscores.txt')
scores, counts = np.unique(data, return_counts=True)
fig, ax = plt.subplots()

ax.bar(scores, counts/sum(counts))
plt.xlabel('ISS scores')
plt.ylabel('Frequency')
plt.title('Two year model run, N=10,000: Distribution of ISS scores')
plt.savefig('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/ISSscoreDistribution.png')

