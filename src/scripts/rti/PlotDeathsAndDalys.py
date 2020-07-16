import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
sns.set()
data = pd.read_csv('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/GBDDeathsPercentage.csv')

plt.plot(data['year'], data['percdeath'], color='black', label='Estimate')
plt.fill_between(data['year'], data['low'], data['up'], color='lightskyblue', label='Error')
plt.xlabel('Year')
plt.ylabel('Percent')
plt.legend()
plt.title('Percentage of all deaths in Malawi due to road traffic injuries, GBD data')
plt.savefig('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/DeathsRTIGBD.png')
plt.clf()

data = pd.read_csv('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/src/scripts/rti/GBDDALYPercentage.csv')
plt.plot(data['year'], data['dalyperc'], color='black', label='Estimate')
plt.fill_between(data['year'], data['low'], data['up'], color='lightskyblue', label='Error')
plt.xlabel('Year')
plt.ylabel('Percent')
plt.legend()
plt.title('Percentage of all DALYs in Malawi due to road traffic injuries, GBD data')
plt.savefig('C:/Users/Robbie Manning Smith/PycharmProjects/TLOmodel/outputs/DALYRTIGBD.png')
