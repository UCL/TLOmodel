"""
This is a scratch file by Tim Hallett
Purpose is to create a nice file for import that gives the population breakdown by region/district (not village!)
* The region and district breakdown is taken from the preliminary report (Dec 2018) of new census

NB. There are some issues with the merge, but this will be resolved new data given population sizes by village
"""

import pandas as pd

workingfile='/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Module-demography/village-district-breakdown/Census Data and Health System Data.xlsx'

resourcefilepath='/Users/tbh03/PycharmProjects/TLOmodel/resources/'

# Load census data on pop sizes in each distrct
district_wb=pd.read_excel(workingfile,sheet_name='PopBreakdownByDistrict_Census')

# need to tidy up what's come in (take of the "xao"-tails from the entries and ensure numbers are numbers!
district_wb.District=district_wb.District.str.strip()
district_wb.Region=district_wb.Region.str.strip()
district_wb.Total=district_wb.Total.str.strip()
district_wb.Men=district_wb.Men.str.strip()
district_wb.Women=district_wb.Women.str.strip()
district_wb.Total=district_wb.Total.str.replace(',','')
district_wb.Men=district_wb.Men.str.replace(',','')
district_wb.Women=district_wb.Women.str.replace(',','')
district_wb.District = district_wb.District.astype(str)
district_wb.Region = district_wb.Region.astype(str)
district_wb.Total = district_wb.Total.astype(float)
district_wb.Men = district_wb.Men.astype(float)
district_wb.Women = district_wb.Women.astype(float)

assert (~pd.isnull(district_wb).any()).all() # check for any null values

# Trim down the sheet
district_wb_trimmed=district_wb.drop(['Region','Men','Women'],axis=1)
district_wb_trimmed=district_wb_trimmed.rename(columns={'Total':'District Total'})

# This is the definitive listing for the Districts
district_names = district_wb_trimmed['District']

district_wb_trimmed.to_csv(resourcefilepath+ 'ResourceFile_DistrictPopulationData.csv')


