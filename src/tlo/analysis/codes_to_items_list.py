"""
(1) Can be used for a list of items without item codes yet saved in a csv file named 'csv_file_to_update_name'.

This script will assign unique code to each unique item name which has no code assigned yet. The codes are
assigned in order from the sequence 0, 1, 2, ....

Duplicated items are allowed, the same code will be assigned to the same items.

(2) Can be used when new items are added later without item codes but some items with codes are already in the list.

This script will keep the existing codes for items with already assigned code and for items without existing
code will assign new code (continue in sequence, i.e. if the highest code is 5, it assigns new codes from the continuing
sequence 6, 7, 8, ...).

------
NB. Make sure the 'csv_file_to_update_name' is the file you want to update. The output will be named
'csv_file_to_update_name' + '_new.csv' to avoid unintentionally losing the previous version.
------
"""

import pandas as pd
from pathlib import Path


# ## CHANGE THIS IF YOU WANT TO USE DIFFERENT FILE AS INPUT
csv_file_to_update_name = 'ResourceFile_Equipment_withoutEquipmentCodes'

# Get the path of the current script file
script_path = Path(__file__)
print(script_path)

# Specify the file path to RF csv file
file_path = script_path.parent.parent.parent.parent / 'resources/healthsystem/infrastructure_and_equipment'

# Load the CSV RF into a DataFrame
df = pd.read_csv(Path(file_path) / str(csv_file_to_update_name + '.csv'))

# Find unique values in Equipment that have no code and are not None or empty
unique_values =\
    df.loc[df['Equip_Code'].isna() & df['Equip_Item'].notna() & (df['Equip_Item'] != ''), 'Equip_Item'].unique()

# Create a mapping of unique values to codes
value_to_code = {}
# Initialize the starting code value
if not df['Equip_Code'].isna().all():
    next_code = int(df['Equip_Code'].max()) + 1
else:
    next_code = 0

# Iterate through unique values
for value in unique_values:
    # Check if there is at least one existing code for this value
    matching_rows = df.loc[df['Equip_Item'] == value, 'Equip_Code'].dropna()
    if not matching_rows.empty:
        # Use the existing code for this value
        existing_code = int(matching_rows.iloc[0])
        # TODO: verify all the codes are the same
    else:
        # If no existing codes, start with the next available code
        existing_code = next_code
        next_code += 1
    value_to_code[value] = existing_code
    # Update the 'Equip_Code' column for matching rows
    df.loc[df['Equip_Item'] == value, 'Equip_Code'] = existing_code

# Convert 'Equip_Code' column to integers
df['Equip_Code'] = df['Equip_Code'].astype('Int64')  # Convert to nullable integer type

# Save CSV with equipment codes
df.to_csv(Path(file_path) / str(csv_file_to_update_name + '_new.csv'), index=False)
