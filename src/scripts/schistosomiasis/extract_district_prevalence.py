""" load the data from ESPEN and extract key data """

import datetime
import pickle
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo.analysis.utils import compare_number_of_deaths

outputpath = Path("./outputs")  # folder for convenience of storing outputs
resourcefilepath = Path("./resources")

df = pd.read_excel(outputpath / 'ESPEN_district_data.xlsx', sheet_name='Sheet1')

# some district names in survey data need to be edited
df['ADMIN2_NAME'] = df['ADMIN2_NAME'].replace('Kasumgu', 'Kasungu')
df['ADMIN2_NAME'] = df['ADMIN2_NAME'].replace('Likoma Islands', 'Likoma')
df['ADMIN2_NAME'] = df['ADMIN2_NAME'].replace('Mzimba North', 'Mzimba')
df['ADMIN2_NAME'] = df['ADMIN2_NAME'].replace('Mzimba South', 'Mzimba')
df['ADMIN2_NAME'] = df['ADMIN2_NAME'].replace('Nkata-Bay', 'Nkhata Bay')
df['ADMIN2_NAME'] = df['ADMIN2_NAME'].replace('Nkhota kota', 'Nkhotakota')


# Step 1: Filter by SurveyYear < 2015, pre-MDA and age-group
df_filtered = df[df['SurveyYear'] < 2015]
df_filtered = df_filtered[df_filtered['Age_end'] < 16]

# Step 2: Group by ADMIN2_NAME and SCH_spp, and calculate the mean Prevalence
grouped = df_filtered.groupby(['ADMIN2_NAME', 'SCH_spp'])['Prevalence'].agg(['mean', 'min', 'max']).reset_index()

# Step 3: Define a function to classify the mean prevalence
def classify_prevalence(prevalence):
    if prevalence == 0:
        return 'zero'
    elif 0 < prevalence < 10:
        return 'low'
    elif 10 <= prevalence < 50:
        return 'moderate'
    else:
        return 'high'

# Apply the classification to the mean prevalence
grouped['classification'] = grouped['mean'].apply(classify_prevalence)

summary_df = grouped.rename(columns={
    'mean': 'mean_prevalence',
    'min': 'min_prevalence',
    'max': 'max_prevalence'
})

summary_df.to_csv(outputpath / 'summary_schisto_prevalence_data2010_2015.csv')


# %%  extract prevalence from last year of reported data
mda_data = pd.read_excel(resourcefilepath / 'ResourceFile_Schisto.xlsx', sheet_name='ESPEN_MDA')

# 2019 is the last year of prevalence data for around 6 districts, some end earlier

df_filtered = df[df['Age_end'] < 16]

# Identify the last available year for each ADMIN2_NAME and SCH_spp
last_years = df_filtered.groupby(['ADMIN2_NAME', 'SCH_spp'])['SurveyYear'].max().reset_index()

# Merge to get only the rows with the last available year
df_last_year = pd.merge(df_filtered, last_years, on=['ADMIN2_NAME', 'SCH_spp', 'SurveyYear'], how='inner')

# Group by ADMIN2_NAME, SCH_spp, and SurveyYear, and calculate the mean, min, max Prevalence
grouped = df_last_year.groupby(['ADMIN2_NAME', 'SCH_spp', 'SurveyYear'])['Prevalence'].agg(['mean', 'min', 'max']).reset_index()

# add in reported prevalence band from 2022 using ESPEN MDA schedule
# Filter mda_data for Year 2022
mda_data_2022 = mda_data[mda_data['Year'] == 2022]

# Merge mda_data_2022 with mda_prevalence (grouped) on ADMIN2_NAME and District
# Assuming that 'District' in mda_data corresponds to 'ADMIN2_NAME' in mda_prevalence
mda_prevalence = pd.merge(grouped, mda_data_2022[['District', 'Endemicity']],
                          left_on='ADMIN2_NAME', right_on='District', how='left')

# Drop the 'District' column from mda_prevalence (optional, if not needed)
mda_prevalence = mda_prevalence.drop(columns=['District'])

# List of new districts to create and the corresponding existing districts
new_districts = {'Lilongwe': 'Lilongwe City', 'Blantyre': 'Blantyre City', 'Zomba': 'Zomba City'}

# Create and append new rows for each SCH_spp
new_rows = []
for existing, new in new_districts.items():
    new_rows.append(mda_prevalence[mda_prevalence['ADMIN2_NAME'] == existing].assign(ADMIN2_NAME=new))

# Concatenate all new rows with the original DataFrame
mda_prevalence = pd.concat([mda_prevalence] + new_rows, ignore_index=True)

# Create two new rows for "Mzuzu City" with NaN values for each SCH_spp
unique_sch_spp = mda_prevalence['SCH_spp'].unique()
mzuzu_rows = pd.DataFrame({
    'ADMIN2_NAME': ['Mzuzu City'] * len(unique_sch_spp),
    'SCH_spp': unique_sch_spp,
}).reindex(columns=mda_prevalence.columns).fillna(0)

# Concatenate the Mzuzu City rows as well
mda_prevalence = pd.concat([mda_prevalence, mzuzu_rows], ignore_index=True)

# Apply the classification to the mean prevalence
mda_prevalence['classification2019'] = mda_prevalence['mean'].apply(classify_prevalence)
mda_prevalence = mda_prevalence.rename(columns={'Endemicity': 'Endemicity2022'})

prevalence_to_output = mda_prevalence.rename(columns={
    'mean': 'mean_prevalence',
    'min': 'min_prevalence',
    'max': 'max_prevalence'
})

# write out each species to a new sheet in resourcefile
df_haem = prevalence_to_output[prevalence_to_output['SCH_spp'] == 'S.haematobium']

# Filter rows where SCH_spp is 'mansoni'
df_mansoni = prevalence_to_output[prevalence_to_output['SCH_spp'] == 'S.mansoni']

#
with pd.ExcelWriter(outputpath / 'latest_schisto_data.xlsx') as writer:
    # Write the haem data to the first sheet
    df_haem.to_excel(writer, sheet_name='LatestData_haematobium', index=False)

    # Write the mansoni data to the second sheet
    df_mansoni.to_excel(writer, sheet_name='LatestData_mansoni', index=False)
