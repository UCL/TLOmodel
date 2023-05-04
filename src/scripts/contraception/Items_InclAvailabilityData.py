"""
This script checks ..
(A) if what_to_check = 'lines':
.. which item codes from the 'RF_of_interest_filename' on lines x-y (incl) (ie 'lines_of_interest' = [x, y])
are (not) in the availability data 'avail_data_filename'
& prints Number of unavailable items (including repetitions) + Unavailable item codes.

XOR

(B) if what_to_check = 'item_name'
.. which items from the 'RF_of_interest_filename' containing 'name_containing' in item name (Items)
are (not) in the availability data 'avail_data_filename'
& prints Codes of all items containing 'name_containing' + Available item codes.
"""

from pathlib import Path

import pandas as pd

# ### TO SET ##########
RF_of_interest_filename = 'ResourceFile_Consumables_Items_and_Packages.csv'
# lines_of_interest = [2, 40]
# this can be set easy by numbers as above, or if I expect it to change soon, and last line can be clearly defined, then
# similar to the below can be used
# (the below prepared for the first line defined by the nmb of line in csv & last line defined by Intervention_Pkg name)
first_line_nmb_csv = 2
last_line_interv_pkg_name = 'Female Condom'
name_containing = 'sutur'
what_to_check = 'lines'  # 'lines' or 'item_name'
avail_data_filename = 'ResourceFile_Consumables_availability_small.csv'
#####

avail_data = pd.read_csv(Path(
    'resources/healthsystem/consumables/' + avail_data_filename
))

items_pkgs = pd.read_csv(Path(
    'resources/healthsystem/consumables/' + RF_of_interest_filename
))
avail_codes = avail_data['item_code'].unique()
items_pkgs_codes = sorted(items_pkgs['Item_Code'].unique())
# print(items_pkgs_codes)


def var_exists(var):
    return var in locals() or var in globals()


if var_exists('lines_of_interest') and var_exists('last_line_interv_pkg_name'):
    raise ValueError(
        "Only one of the parameters should be defined, 'lines_of_interest' or 'last_line_interv_pkg_name', but both "
        "were given."
    )
elif not var_exists('lines_of_interest') and not var_exists('last_line_interv_pkg_name'):
    raise ValueError(
        "Exactly one of the parameters should be defined, 'lines_of_interest' or 'last_line_interv_pkg_name', but none "
        "was given."
    )
elif var_exists('last_line_interv_pkg_name'):
    last_line_nmb_data = items_pkgs.loc[items_pkgs['Intervention_Pkg'] == last_line_interv_pkg_name].index[0]
    lines_of_interest = [first_line_nmb_csv, last_line_nmb_data + 2]

if what_to_check == 'lines':
    unavail_codes = set()
    nmb_unavail_items = 0
    for item_code_to_check in items_pkgs['Item_Code'][lines_of_interest[0]-2:lines_of_interest[1]-1]:
        if item_code_to_check not in avail_codes:
            unavail_codes.add(item_code_to_check)
            nmb_unavail_items += 1
    print('Number of unavailable items (including repetitions): ' + str(nmb_unavail_items) + ' of ' +
          str(lines_of_interest[1] - lines_of_interest[0] + 1))
    print('Unavailable item codes: ' + str(unavail_codes))
elif what_to_check == 'item_name':
    avail_items_name_containing_codes = set()
    avail_items_name_containing_names = set()
    name_containing_items = set()
    line = 0
    for item_name in items_pkgs['Items']:
        if name_containing in str.lower(item_name):
            name_containing_items.add(items_pkgs['Item_Code'][line])
            if items_pkgs['Item_Code'][line] in avail_codes:
                avail_items_name_containing_codes.add(items_pkgs['Item_Code'][line])
                avail_items_name_containing_names.add(items_pkgs['Items'][line])
        line += 1
    print('All ' + name_containing + ' items: ' + str(name_containing_items))
    print('Available item codes: ' + str(avail_items_name_containing_codes))
    print('Available item names: ' + str(avail_items_name_containing_names))
