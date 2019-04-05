"""
This file outputs the current staffing
"""

import pandas as pd

workingfile='/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Module-healthsystem/chai ehp resource use data/Formatting for ResourceFile.xlsx'

outputfile='/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Module-healthsystem/chai ehp resource use data/ResourceFile_CurrentStaff.csv'


wb =pd.read_excel(workingfile,sheet_name='RF_CurrentStaff')


# replace all null values with zero values

wb.fillna(0, inplace=True)

# check that the district will match up:
chai_districts = pd.unique( wb['District'] )

# get the distrcts from the resource file:
pop = pd.read_csv('/Users/tbh03/PycharmProjects/TLOmodel/resources/ResourceFile_PopBreakdownByVillage.csv')

pop_districts = pd.unique(pop['District'])


# check that every district in the resource file is represented in the CHAI list:
for d in pop_districts:
    print('Resource File district, ', d , ' in chai tables: ', (d in chai_districts))
# Likoma not in the Chai data (which may be classified as Nkhata Bay)


# check that every district in the chai table is represented in the resoure file list:
for d in chai_districts:
    print('Chai file district, ', d , ' in resource file: ', (d in pop_districts))
# HQ or missing', 'KCH', 'MCH', 'QECH' not in the pop data for districts


# Remedy the mismatches:

wb.loc[wb['District']=='KCH','District'] = 'Lilongwe'

wb.loc[wb['District']=='HQ or missing','District'] = 'Lilongwe'

wb.loc[wb['District']=='MCH','District'] = 'Mzimba'

wb.loc[wb['District']=='QECH','District'] = 'Blantyre'

wb.loc[wb['District']=='ZCH','District'] = 'Zomba'

# Add in Likoma (with zero health workers)
record = wb.loc[wb['District']==chai_districts[0]].copy()
record['District']='Likoma'
cols=record.columns
cols=cols[2:]
for c in cols:
    record.iloc[record.index >= 0, record.columns == c]=0

wb=wb.append(record)

# now check that merging:
pop_districts = pd.unique(pop['District'])
chai_districts = pd.unique( wb['District'] )

# check that every district in the resource file is represented in the CHAI list:
for d in pop_districts:
    print('Resource File district, ', d , ' in chai tables: ', (d in chai_districts))


# check that every district in the chai table is represented in the resoure file list:
for d in chai_districts:
    print('Chai file district, ', d , ' in resource file: ', (d in pop_districts))


# Decide which type of count we want to use:
wb.loc[wb['Count']=='CURRENT TOTAL STAFF (NON-VACANT POSITIONS)','Nurse Officer'].sum()
wb.loc[wb['Count']=='Funded TOTAL STAFF (NON-VACANT POSITIONS)','Nurse Officer'].sum()

wb.to_csv(outputfile)
