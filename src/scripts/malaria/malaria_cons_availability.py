import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

resourcefilepath = Path("./resources")
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

outputspath = Path("./outputs/t.mangal@imperial.ac.uk")

# %%: get consumables availability

# get consumables spreadsheet
cons_availability = pd.read_csv(
    resourcefilepath / "healthsystem/consumables/ResourceFile_Consumables_availability_small.csv")
items_list = pd.read_csv(
    resourcefilepath / "healthsystem/consumables/ResourceFile_Consumables_Items_and_Packages.csv")

# import master facilities list to get facility levels mapped to facility ID
master_fac = pd.read_csv(
    resourcefilepath / "healthsystem/organisation/ResourceFile_Master_Facilities_List.csv")

# map facility levels to facility ID in consumables spreadsheet
cons_full = pd.merge(cons_availability, master_fac,
                     left_on=["Facility_ID"], right_on=["Facility_ID"], how='left')

# groupby item code & facility level -> average availability by facility level for all items
average_cons_availability = cons_full.groupby(["item_code", "Facility_Level"])["available_prop"].mean().reset_index()


def get_item_codes_from_package_name(lookup_df: pd.DataFrame, package: str) -> dict:
    return int(pd.unique(lookup_df.loc[lookup_df["Intervention_Pkg"] == package, "Item_Code"]))


def get_item_code_from_item_name(lookup_df: pd.DataFrame, item: str) -> int:
    """Helper function to provide the item_code (an int) when provided with the name of the item"""
    return int(pd.unique(lookup_df.loc[lookup_df["Items"] == item, "Item_Code"])[0])


# malaria consumables

item_codes_dict = dict()

# diagnostics
item_codes_dict["rdt"] = get_item_code_from_item_name(items_list, "Malaria test kit (RDT)")

# treatment
item_codes_dict["Artemether_lumefantrine"] = get_item_code_from_item_name(
    items_list,
    "Lumefantrine 120mg/Artemether 20mg,  30x18_540_CMST")
item_codes_dict["paracetamol_syrup"] = get_item_code_from_item_name(
    items_list, "Paracetamol syrup 120mg/5ml_0.0119047619047619_CMST")
item_codes_dict["paracetamol"] = get_item_code_from_item_name(items_list, "Paracetamol 500mg_1000_CMST")
item_codes_dict["Injectable_artesunate"] = get_item_code_from_item_name(items_list, "Injectable artesunate")
item_codes_dict["cannula"] = get_item_code_from_item_name(items_list,
                                                          "Cannula iv  (winged with injection pot) 18_each_CMST")
item_codes_dict["gloves"] = get_item_code_from_item_name(items_list,
                                                         "Disposables gloves, powder free, 100 pieces per box")
item_codes_dict["gauze"] = get_item_code_from_item_name(items_list, "Gauze, absorbent 90cm x 40m_each_CMST")
item_codes_dict["water_for_injection"] = get_item_code_from_item_name(items_list, "Water for injection, 10ml_Each_CMST")

# select item codes from item_codes_dict
selected_cons_availability = average_cons_availability[
    average_cons_availability["item_code"].isin(item_codes_dict.values())]
# remove level 0
selected_cons_availability = selected_cons_availability.loc[selected_cons_availability.Facility_Level != "0"]
# remove level 4
selected_cons_availability = selected_cons_availability.loc[selected_cons_availability.Facility_Level != "4"]

# replace item code with item name
selected_cons_availability.loc[selected_cons_availability.item_code == 163, "item_code"] = "rdt"
selected_cons_availability.loc[selected_cons_availability.item_code == 164, "item_code"] = "Artemether_lumefantrine"
selected_cons_availability.loc[selected_cons_availability.item_code == 71, "item_code"] = "paracetamol_syrup"
selected_cons_availability.loc[selected_cons_availability.item_code == 113, "item_code"] = "paracetamol"
selected_cons_availability.loc[selected_cons_availability.item_code == 170, "item_code"] = "Injectable_artesunate"
selected_cons_availability.loc[selected_cons_availability.item_code == 171, "item_code"] = "cannula"
selected_cons_availability.loc[selected_cons_availability.item_code == 135, "item_code"] = "gloves"
selected_cons_availability.loc[selected_cons_availability.item_code == 101, "item_code"] = "gauze"
selected_cons_availability.loc[selected_cons_availability.item_code == 98, "item_code"] = "water_for_injection"

df_heatmap = selected_cons_availability.pivot_table(
    values='available_prop',
    index='item_code',
    columns='Facility_Level',
    aggfunc=np.mean)

ax = sns.heatmap(df_heatmap, annot=True)
plt.tight_layout()

plt.xlabel('Facility level')
plt.ylabel('')
# plt.savefig(outputspath / "cons_availability.png", bbox_inches='tight')
plt.show()
