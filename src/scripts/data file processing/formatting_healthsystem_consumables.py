"""

This file sets up the data regarding the consumables that may be used in the course of an appointment.

The file is provided by Mathias Arnold and derives from the EHP work.

"""

# EHP Consumables list
workingfile = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Module-healthsystem/From Matthias Arnold/ORIGINAL_Intervention input.xlsx'

# OUTPUT RESOURCE_FILES TO:
resourcefilepath = '/Users/tbh03/PycharmProjects/TLOmodel/resources/'


# ----------
# ----------
# ----------

import pandas as pd
import numpy as np

# ----------

wb_import = pd.read_excel(workingfile, sheet_name='Intervention input costs', header=None)


# Drop the top row and first column as these were blank

wb = wb_import.drop(columns=0)
wb = wb.iloc[1:,:]

# Fill forward the first column to reflect that these are susbuming multiple of the rows
wb.loc[:,1]= wb.loc[:,1].fillna(method='bfill')

# Remove any entry that says "Total Cost"
wb=wb.drop(wb.index[wb.loc[:,2]=='Total cost'])

# get the header rows and forward fill these so that they relate to each row
hr = wb.index[wb.loc[:,2].isna()]
wb.loc[:,'Cat']=None
wb.loc[hr,'Cat']=wb.loc[hr,1]
wb.loc[:,'Cat']=wb.loc[:,'Cat'].fillna(method='ffill')
wb=wb.drop(hr)

# Make the top row into columns names and delete repeats of this header
wb.columns=wb.iloc[0]
wb = wb.drop(wb.index[0])
wb=wb.reset_index(drop=True)

# rationalise the columns and the names
wb.loc[:,'Expected_Units_Per_Case']= (wb.loc[:,'Proportion of patients receiving this input']/100) * \
                                        wb.loc[:,'Number of units'] * \
                                        wb.loc[:, 'Times per day'] * \
                                        wb.loc[:,'Days per case']
wb.loc[:,'Unit_Cost'] = wb.loc[:,'Unit cost (MWK) (2010)']

wb.loc[:,'Expected_Cost_Per_Case'] = wb.loc[:,'Expected_Units_Per_Case'] * wb.loc[:,'Unit_Cost']



wb= wb.rename(columns={'Drug/Supply Inputs':'Items',
               'Intervention':'Intervention_Pkg',
               'Maternal/Newborn and Reproductive Health':'Intervention_Cat'})

wb = wb.loc[:,['Intervention_Cat',
                'Intervention_Pkg',
                'Items',
                'Expected_Units_Per_Case',
                'Unit_Cost'
               ]]


# Assign a unique package code
unique_intvs = pd.unique(wb['Intervention_Pkg'])
intv_codes=pd.DataFrame({'Intervention_Pkg':unique_intvs,
              'Intervention_Pkg_Code':np.arange(0,len(unique_intvs ))})

wb=wb.merge(intv_codes,on='Intervention_Pkg',how='left',indicator=True)
assert (wb['_merge']=='both').all()
wb=wb.drop(columns='_merge')


# Assign a unique code for each item
unique_items = pd.unique(wb['Items'])
item_codes= pd.DataFrame({'Items':unique_items,
              'Item_Code':np.arange(0,len(unique_items ))})
wb=wb.merge(item_codes,on='Items',how='left',indicator=True)
assert (wb['_merge']=='both').all()
wb=wb.drop(columns='_merge')

# Reorder columns to be nice:
wb=wb[['Intervention_Cat',
                'Intervention_Pkg',
                'Intervention_Pkg_Code',
                'Items',
                'Item_Code',
                'Expected_Units_Per_Case',
                'Unit_Cost']]

wb.to_csv(resourcefilepath + 'ResourceFile_Consumables.csv')

