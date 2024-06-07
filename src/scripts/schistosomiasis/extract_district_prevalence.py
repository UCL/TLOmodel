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

df = pd.read_excel(outputpath / 'ESPEN_district_data.xlsx', sheet_name='Sheet1')

# Step 1: Filter by SurveyYear < 2015, pre-MDA
df_filtered = df[df['SurveyYear'] < 2015]

# Step 2: Group by ADMIN2_NAME and SCH_spp, and calculate the mean Prevalence
# grouped = df_filtered.groupby(['ADMIN2_NAME', 'SCH_spp'])['Prevalence'].mean().reset_index()
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

summary_df.to_csv(outputpath / 'summary_schisto_prevalence_data.csv')
