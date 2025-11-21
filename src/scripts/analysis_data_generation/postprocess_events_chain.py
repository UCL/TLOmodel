import pandas as pd
from dateutil.relativedelta import relativedelta

# Remove from every individual's event chain all events that were fired after death
def cut_off_events_after_death(df):

    events_chain = df.groupby('person_ID')
    
    filtered_data = pd.DataFrame()

    for name, group in events_chain:

        # Find the first non-NaN 'date_of_death' and its index
        first_non_nan_index = group['date_of_death'].first_valid_index()
        
        if first_non_nan_index is not None:
            # Filter out all rows after the first non-NaN index
            filtered_group = group.loc[:first_non_nan_index]  # Keep rows up to and including the first valid index
            filtered_data = pd.concat([filtered_data, filtered_group])
        else:
            # If there are no non-NaN values, keep the original group
            filtered_data = pd.concat([filtered_data, group])

    return filtered_data

# Load into DataFrame
def load_csv_to_dataframe(file_path):
    try:
        # Load raw chains into df
        df = pd.read_csv(file_path)
        print("Raw event chains loaded successfully!")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

file_path = 'output.csv'  # Replace with the path to your CSV file

output = load_csv_to_dataframe(file_path)

# Some of the dates appeared not to be in datetime format. Correct here.
output['date_of_death'] = pd.to_datetime(output['date_of_death'], errors='coerce')
output['date_of_birth'] = pd.to_datetime(output['date_of_birth'], errors='coerce')
if 'hv_date_inf' in output.columns:
    output['hv_date_inf'] = pd.to_datetime(output['hv_date_inf'], errors='coerce')


date_start = pd.to_datetime('2010-01-01')
if 'Other' in output['cause_of_death'].values:
    print("ERROR: 'Other' was included in sim as possible cause of death")
    exit(-1)

# Choose which columns in individual properties to visualise
columns_to_print =['event','is_alive','hv_inf', 'hv_art','tb_inf', 'tb_date_active', 'event_date', 'when']
#columns_to_print =['person_ID', 'date_of_birth', 'date_of_death', 'cause_of_death','hv_date_inf', 'hv_art','tb_inf', 'tb_date_active', 'event date', 'event']

# When checking which individuals led to *any* changes in individual properties, exclude these columns from comparison
columns_to_exclude_in_comparison = ['when', 'event', 'event_date', 'age_exact_years', 'age_years', 'age_days', 'age_range', 'level', 'appt_footprint']

# If considering epidemiology consistent with sim, add check here.
check_ages_of_those_HIV_inf = False
if check_ages_of_those_HIV_inf:
    for index, row in output.iterrows():
        if pd.isna(row['hv_date_inf']):
            continue  # Skip this iteration
        diff = relativedelta(output.loc[index, 'hv_date_inf'],output.loc[index, 'date_of_birth'])
        if diff.years > 1 and diff.years<15:
            print("Person contracted HIV infection at age younger than 15", diff)

# Remove events after death
filtered_data = cut_off_events_after_death(output)

print_raw_events = True # Print raw chain of events for each individual
print_selected_changes = False
print_all_changes = True
person_ID_of_interest = 494

pd.set_option('display.max_rows', None)

for name, group in filtered_data.groupby('person_ID'):
    list_of_dob = group['date_of_birth']
    
    # Select individuals based on when they were born
    if list_of_dob.iloc[0].year<2010:

        # Check that immutable properties are fixed for this individual, i.e. that events were collated properly:
        all_identical_dob = group['date_of_birth'].nunique() == 1
        all_identical_sex = group['sex'].nunique() == 1
        if all_identical_dob is False or all_identical_sex is False:
            print("Immutable properties are changing! This is not chain for single individual")
            print(group)
            exit(-1)
            
        print("----------------------------------------------------------------------")
        print("person_ID ", group['person_ID'].iloc[0], "d.o.b ", group['date_of_birth'].iloc[0])
        print("Number of events for this individual ", group['person_ID'].iloc[0], "is :", len(group)/2) # Divide by 2 before printing Before/After for each event
        number_of_events =len(group)/2
        number_of_changes=0
        if print_raw_events:
            print(group)
        
        if print_all_changes:
            # Check each row
            comparison = group.drop(columns=columns_to_exclude_in_comparison).fillna(-99999).ne(group.drop(columns=columns_to_exclude_in_comparison).shift().fillna(-99999))

            # Iterate over rows where any column has changed
            for idx, row_changed in comparison.iloc[1:].iterrows():
                if row_changed.any():  # Check if any column changed in this row
                    number_of_changes+=1
                    changed_columns = row_changed[row_changed].index.tolist()  # Get the columns where changes occurred
                    print(f"Row {idx} - Changes detected in columns: {changed_columns}")
                    columns_output = ['event', 'event_date', 'appt_footprint', 'level'] + changed_columns
                    print(group.loc[idx, columns_output])  # Print only the changed columns
                    if group.loc[idx, 'when'] == 'Before':
                        print('-----> THIS CHANGE OCCURRED BEFORE EVENT!')
                    #print(group.loc[idx,columns_to_print])
                    print()  # For better readability
            print("Number of changes is ", number_of_changes, "out of ", number_of_events, " events")
        
        if print_selected_changes:
            tb_inf_condition = (
                ((group['tb_inf'].shift(1) == 'uninfected') & (group['tb_inf'] == 'active')) |
                ((group['tb_inf'].shift(1) == 'latent') & (group['tb_inf'] == 'active')) |
                ((group['tb_inf'].shift(1) == 'active') & (group['tb_inf'] == 'latent')) |
                ((group['hv_inf'].shift(1) is False) & (group['hv_inf'] is True)) |
                ((group['hv_art'].shift(1) == 'not') & (group['hv_art'] == 'on_not_VL_suppressed')) |
                ((group['hv_art'].shift(1) == 'not') & (group['hv_art'] == 'on_VL_suppressed')) |
                ((group['hv_art'].shift(1) == 'on_VL_suppressed') & (group['hv_art'] == 'on_not_VL_suppressed')) |
                ((group['hv_art'].shift(1) == 'on_VL_suppressed') & (group['hv_art'] == 'not')) |
                ((group['hv_art'].shift(1) == 'on_not_VL_suppressed') & (group['hv_art'] == 'on_VL_suppressed')) |
                ((group['hv_art'].shift(1) == 'on_not_VL_suppressed') & (group['hv_art'] == 'not'))
            )

            alive_condition = (
                (group['is_alive'].shift(1) is True) & (group['is_alive'] is False)
            )
            # Combine conditions for rows of interest
            transition_condition = tb_inf_condition | alive_condition

            if list_of_dob.iloc[0].year >= 2010:
                print("DETECTED OF INTEREST")
                print(group[group['event'] == 'Birth'][columns_to_print])

            # Filter the DataFrame based on the condition
            filtered_transitions = group[transition_condition]
            if not filtered_transitions.empty:
                if list_of_dob.iloc[0].year < 2010:
                    print("DETECTED OF INTEREST")
                print(filtered_transitions[columns_to_print])
    
    
print("Number of individuals simulated ", filtered_data.groupby('person_ID').ngroups)



