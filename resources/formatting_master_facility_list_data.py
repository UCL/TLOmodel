# This is a scratch file by Tim Hallett that will get the UNCIEF Master Facility Level data in the right format for importing


import pandas as pd
import numpy as np

workingfile='/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Health System Resources/ORIGINAL_Master List Free Service Health Facilities in Malawi_no contact details.xlsx'
outputfile='/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Health System Resources/ResourceFile_MasterFacilitiesList.csv'

wb=pd.read_excel(workingfile,sheet_name='All HF Malawi')

wb=wb.drop(columns='Date')

wb['Facility_ID']=np.arange(len(wb))

wb.to_csv(outputfile)



