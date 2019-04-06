#------ Now look at the capabilities for offering appointments (of various types) that can come from that staffing pattern

import pandas as pd
import numpy as np

workingfile='/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Module-healthsystem/chai ehp resource use data/Formatting for ResourceFile.xlsx'

outputfile='/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Module-healthsystem/chai ehp resource use data/ResourceFile_HealthSystem_ApptTimes.csv'



sheet= pd.read_excel(workingfile,sheet_name='Time_Base',header=None)

# get rid of the junky rows
X=sheet.loc[[7,8,9,11,12,14,15,17,18,20,21,23,24,26,27],]

# get ride of junky columns


# turn the % and the time into an expectation
# The % give the percetnage of patients who require that time. Multiply these together to give an expectation

types_of_location = ['Central_Hospital', 'District_Hospital', 'Community_Hospital', 'Urban_HealthCentre', 'Rural_HealthCentre', 'HealthPost', 'Dispensary' ]


Central_Hospital_ExpecTime = X.loc[8,2:].values.astype(float) * X.loc[9,2:].values.astype(float)
District_Hospital_ExpecTime = X.loc[11,2:].values.astype(float) * X.loc[12,2:].values.astype(float)
Community_Hospital_ExpecTime = X.loc[14,2:].values.astype(float) * X.loc[15,2:].values.astype(float)
Urban_HealthCentre_ExpecTime = X.loc[17,2:].values.astype(float) * X.loc[18,2:].values.astype(float)
Rural_HealthCentre_ExpecTime = X.loc[20,2:].values.astype(float) * X.loc[21,2:].values.astype(float)
HealthPost_ExpecTime = X.loc[23,2:].values.astype(float) * X.loc[24,2:].values.astype(float)
Dispensary_ExpecTime = X.loc[26,2:].values.astype(float) * X.loc[27,2:].values.astype(float)

# compile into dict
ExpectTime = dict({
    'Central_Hospital' : Central_Hospital_ExpecTime,
    'District_Hospital' : District_Hospital_ExpecTime,
    'Community_Hospital' : Community_Hospital_ExpecTime,
    'Urban_HealthCentre' : Urban_HealthCentre_ExpecTime,
    'Rural_HealthCentre' : Rural_HealthCentre_ExpecTime,
    'HealthPost' : HealthPost_ExpecTime,
    'Dispensary' : Dispensary_ExpecTime
})

composite_code=X.loc[7,2:]

# break apart composite to give the appt_type and the officer_type
df=composite_code.str.split(pat='_',expand=True)
df=df.rename(columns={0: "Officer_Code", 1: "ApptType_Code"})
df= df.reset_index(drop=True)

# Repeat the dataframe for the type of facility at which the type of appt occures
df.loc[:,'Facility_Type']=None


# TODO: some kind of error due to nan in the expectime (RuntimeWarning: invalid value encountered in reduce return umr_minimum(a, axis, None, out, keepdims))

Table = pd.DataFrame(columns=df.columns)

for l in types_of_location:
    df_for_loc = df
    df_for_loc['Facility_Type'] = l

    df_for_loc['Time']=ExpectTime[l]

    Table=Table.append(df_for_loc)


# Merge back in the proper titles of the officers

officer_name= sheet.loc[3:4,2:22].transpose()
officer_name=officer_name.rename(columns={3: "Officer", 4:"Officer_Code"})

Table_merged=Table.merge(officer_name,how='left')

Table_merged.to_csv(outputfile)
