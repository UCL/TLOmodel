""" convert long-list data on prevalence and MDA coverage by district
to wide
"""
import pandas as pd

# Load the data from the file
df = pd.read_excel('/Users/tmangal/Documents/Thanzi_docs/Schisto/data_MW_SCH_iu_MDA.xlsx',
                   header=0)

# Pivot the table to wide format using 'TargetPop' as the key
wide_df = df.pivot_table(
    index=['ADMIN1', 'ADMIN2', 'Year', 'Endemicity', 'MDA_scheme'],
    columns='TargetPop',
    values=['PopReq', 'PopTrg', 'PopTreat', 'Cov', 'EpiCov'],
    aggfunc='first'
).reset_index()

# Flatten the multi-level columns
wide_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in wide_df.columns.values]

# Reorder columns
# Define the desired order of columns
desired_columns = ['ADMIN1', 'ADMIN2', 'Year', 'Endemicity', 'MDA_scheme',
                   'PopReq_SAC', 'PopTrg_SAC', 'PopTreat_SAC', 'Cov_SAC', 'EpiCov_SAC',
                   'PopReq_Adults', 'PopTrg_Adults', 'PopTreat_Adults', 'Cov_Adults', 'EpiCov_Adults',
                   'PopReq_Total', 'PopTrg_Total', 'PopTreat_Total', 'Cov_Total', 'EpiCov_Total']

# Reorder columns in the DataFrame
ordered_df = wide_df[desired_columns]

# correct names to match existing resourcefiles
# Define replacements for ADMIN2 names
replacements = {
    'Nkhota kota': 'Nkhotakota',
    'Likoma Islands': 'Likoma',
    'Nkhatabay': 'Nkhata Bay',
    'Kasumgu': 'Kasungu'
}

# Replace values in ADMIN2 column
ordered_df['ADMIN2'] = ordered_df['ADMIN2'].replace(replacements)

# combine mzimba north and south -> mzimba
# Filter and combine 'Mzimba North' and 'Mzimba South' rows into 'Mzimba'
mzimba_north = ordered_df[ordered_df['ADMIN2'] == 'Mzimba North']
mzimba_south = ordered_df[ordered_df['ADMIN2'] == 'Mzimba South']

# Aggregate values for 'Mzimba'
mzimba_combined = pd.concat([mzimba_north, mzimba_south], ignore_index=True)
mzimba_combined['ADMIN2'] = 'Mzimba'
mzimba_combined = mzimba_combined.groupby(['ADMIN1', 'ADMIN2', 'Year', 'Endemicity', 'MDA_scheme']).agg({
    'PopReq_SAC': 'sum',
    'PopTrg_SAC': 'sum',
    'PopTreat_SAC': 'sum',
    'Cov_SAC': 'mean',
    'EpiCov_SAC': 'mean',
    'PopReq_Adults': 'sum',
    'PopTrg_Adults': 'sum',
    'PopTreat_Adults': 'sum',
    'Cov_Adults': 'mean',
    'EpiCov_Adults': 'mean',
    'PopReq_Total': 'sum',
    'PopTrg_Total': 'sum',
    'PopTreat_Total': 'sum',
    'Cov_Total': 'mean',
    'EpiCov_Total': 'mean'
}).reset_index()

# Remove 'Mzimba North' and 'Mzimba South' rows from original DataFrame
ordered_df = ordered_df[~ordered_df['ADMIN2'].isin(['Mzimba North', 'Mzimba South'])]

# Append the aggregated 'Mzimba' rows to the DataFrame
ordered_df = pd.concat([ordered_df, mzimba_combined], ignore_index=True)

# add Lilonge City, Blantyre City, Zomba City, Mzuzu City
# Filter rows where ADMIN2 is 'Lilongwe'
lilongwe_rows = ordered_df[ordered_df['ADMIN2'] == 'Lilongwe']
lilongwe_city_rows = lilongwe_rows.copy()
lilongwe_city_rows['ADMIN2'] = 'Lilongwe City'

blantyre_rows = ordered_df[ordered_df['ADMIN2'] == 'Blantyre']
blantyre_city_rows = blantyre_rows.copy()
blantyre_city_rows['ADMIN2'] = 'Blantyre City'

mzuzu_rows = ordered_df[ordered_df['ADMIN2'] == 'Mzimba']
mzuzu_city_rows = mzuzu_rows.copy()
mzuzu_city_rows['ADMIN2'] = 'Mzuzu City'

zomba_rows = ordered_df[ordered_df['ADMIN2'] == 'Zomba']
zomba_city_rows = zomba_rows.copy()
zomba_city_rows['ADMIN2'] = 'Zomba City'

ordered_df = pd.concat([ordered_df, lilongwe_city_rows], ignore_index=True)
ordered_df = pd.concat([ordered_df, blantyre_city_rows], ignore_index=True)
ordered_df = pd.concat([ordered_df, mzuzu_city_rows], ignore_index=True)
ordered_df = pd.concat([ordered_df, zomba_city_rows], ignore_index=True)

# check 32 districts in total
pop_file = pd.read_csv(
    '/Users/tmangal/PycharmProjects/TLOmodel/resources/demography/ResourceFile_Population_2010.csv')

# Lookup districts in population file and make sure all are contained in MDA data file
districts = pop_file['District'].drop_duplicates().to_list()
for district in districts:
    assert district in ordered_df['ADMIN2'].values, f"District '{district}' not found in dataframe."

ordered_df.rename(columns={'ADMIN2': 'District'}, inplace=True)

# add column coverage PSAC
ordered_df['Cov_PSAC'] = 0
ordered_df['EpiCov_PSAC'] = 0

# divide all coverage columns by 100 to get proportion covered
columns_to_divide = ordered_df.filter(regex='^(EpiCov|Cov)').columns
ordered_df[columns_to_divide] = ordered_df[columns_to_divide] / 100

# Save the transformed data to a new CSV file
output_file_path = '/Users/tmangal/Documents/Thanzi_docs/Schisto/data_MW_SCH_iu_MDA_wide.csv'
ordered_df.to_csv(output_file_path, index=False)

