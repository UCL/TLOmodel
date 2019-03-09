# This is a scratch file by Tim Hallett that will get the UNCIEF Master Facility Level data in the right format for importing


import pandas as pd
import numpy as np

workingfile='/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Health System Resources/ORIGINAL_Master List Free Service Health Facilities in Malawi_no contact details.xlsx'
outputfile='/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Health System Resources/ResourceFile_MasterFacilitiesList.csv'

wb=pd.read_excel(workingfile,sheet_name='All HF Malawi')

wb=wb.drop(columns='Date')

wb['Facility_ID']=np.arange(len(wb))

wb.to_csv(outputfile)


# Add that every village is also attaching to a Referral Hospital based on regio

# Add that every village is also attching to all the hospitals in the district


# Add it that there is a central hospital at the top of the hierarchy for all villages

# experiment with the data:
# reduce it to one villge


myvillage='Iyela'

df=wb.loc[wb['Village']==myvillage]
