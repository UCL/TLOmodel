

import datetime
import pandas as pd
from fuzzywuzzy import process, fuzz
from pathlib import Path

#install fuzzywuzzy via pip install fuzzywuzzy
# Define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# Print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

# Define pathways to the data folders (note: adjust these paths)
outputfilepath = Path("./outputs")
resourcefilepath = Path("./resources")
datafilepath = Path("./../../My Drive/CHE-Work/Thanzi/Consumables")

# Load the CSV file using the complete path
filepath = datafilepath / 'CMST_catalogue.csv'
df_cons_catalogue = pd.read_csv(filepath, low_memory=False)

# Load the workbook
workbook = pd.ExcelFile(datafilepath / 'CMST_catalogue.csv')

# Load the specific worksheets
cmst_live_sheet = pd.read_excel(workbook, sheet_name='CMST_live')
master_tlo_sheet = pd.read_excel(workbook, sheet_name='Master_tlo')

# Extract the columns for fuzzy matching
cmst_live_items = cmst_live_sheet['Item Description'].dropna().tolist()
master_tlo_items = master_tlo_sheet['Item Description'].dropna().tolist()

# Perform fuzzy matching based on partial ratio
matches = process.extractBests(
    cmst_live_items, master_tlo_items,
    scorer=fuzz.partial_ratio, score_cutoff=90
)
# Create a DataFrame to store the matches
matches_df3 = pd.DataFrame(matches, columns=['CMST_live_Item', 'Master_tlo_Item', 'Score'])

# Print the matched items
print(matches_df3)
matches_df3.to_excel(filepath, index=False)

# Perform fuzzy matching based on simple ratio
matches = process.extractBests(
    cmst_live_items, master_tlo_items,
    scorer=fuzz.ratio, score_cutoff=90
)
# Create a DataFrame to store the matches
matches_df2 = pd.DataFrame(matches, columns=['CMST_live_Item', 'Master_tlo_Item', 'Score'])

# Print the matched items
print(matches_df2)
matches_df2.to_excel(filepath, index=False)

# Perform fuzzy matching based on token sort ratio
matches = process.extractBests(
    cmst_live_items, master_tlo_items,
    scorer=fuzz.token_sort_ratio, score_cutoff=90
)
# Create a DataFrame to store the matches
matches_df2 = pd.DataFrame(matches, columns=['CMST_live_Item', 'Master_tlo_Item', 'Score'])

# Print the matched items
print(matches_df2)
matches_df2.to_excel(filepath, index=False)

