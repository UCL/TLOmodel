

import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import rgb_to_hsv


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


def get_item_codes_from_package_name(lookup_df: pd.DataFrame, package: str) -> int:
    return int(pd.unique(lookup_df.loc[lookup_df["Intervention_Pkg"] == package, "Item_Code"])[0])
    # return int(pd.unique(lookup_df.loc[lookup_df["Intervention_Pkg"] == package, "Item_Code"]))


def get_item_code_from_item_name(lookup_df: pd.DataFrame, item: str) -> int:
    """Helper function to provide the item_code (an int) when provided with the name of the item"""
    return int(pd.unique(lookup_df.loc[lookup_df["Items"] == item, "Item_Code"])[0])


## Malaria consumables

item_codes_dict = dict()

# diagnostics
item_codes_dict["RDT"] = get_item_code_from_item_name(items_list, "Malaria test kit (RDT)")

# treatment
item_codes_dict["malaria_uncomplicated"] = get_item_code_from_item_name(
    items_list, "Lumefantrine 120mg/Artemether 20mg,  30x18_540_CMST")
item_codes_dict["malaria_complicated"] = get_item_code_from_item_name(
    items_list, "Injectable artesunate")
item_codes_dict["malaria_iptp"] = get_item_code_from_item_name(
    items_list, "Sulfamethoxazole + trimethropin, tablet 400 mg + 80 mg")
item_codes_dict["bednet"] = get_item_code_from_item_name(
    items_list, "Insecticide-treated net")

# select item codes from item_codes_dict
selected_cons_availability = average_cons_availability[average_cons_availability["item_code"].isin(item_codes_dict.values())]
# remove level 0
selected_cons_availability = selected_cons_availability.loc[selected_cons_availability.Facility_Level != "0"]
# remove level 4
selected_cons_availability = selected_cons_availability.loc[selected_cons_availability.Facility_Level != "4"]


# Change the dtype of 'item_code' to object to allow mixed types
selected_cons_availability['item_code'] = selected_cons_availability['item_code'].astype(object)


# replace item code with item name
selected_cons_availability.loc[selected_cons_availability.item_code == 163, "item_code"] = "Malaria RDT"
selected_cons_availability.loc[selected_cons_availability.item_code == 164, "item_code"] = "Lumefantrine/ \nArtemether"
selected_cons_availability.loc[selected_cons_availability.item_code == 170, "item_code"] = "Injectable \nartesunate"
selected_cons_availability.loc[selected_cons_availability.item_code == 162, "item_code"] = "Sulfamethoxazole/ \ntrimethropin"
selected_cons_availability.loc[selected_cons_availability.item_code == 160, "item_code"] = "Insecticide-treated \nnet"


# Create the pivot table for the heatmap
df_heatmap = selected_cons_availability.pivot_table(
    values='available_prop',
    index='item_code',
    columns='Facility_Level',
    aggfunc='mean'
)


# Function to format annotation text

def annotate_heatmap(ax, data, valfmt="{x:.2f}", **textkw):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            text_color = 'white' if rgb_to_hsv(ax.collections[0].get_facecolors()[i * data.shape[1] + j][:3])[2] < 0.5 else 'black'
            ax.text(j + 0.5, i + 0.5, valfmt.format(x=value), ha='center', va='center', color=text_color, **textkw)

ax = sns.heatmap(df_heatmap, annot=False, cbar_kws={'label': ''}, vmin=0, vmax=1, fmt=".2f")
annotate_heatmap(ax, df_heatmap.values)

plt.tight_layout()
plt.xlabel('Facility Level')
plt.ylabel('')
plt.title('Malaria consumables')
# plt.savefig(outputspath / "cons_availability.png", bbox_inches='tight')
plt.savefig(outputspath / "malaria_cons_availability.pdf", bbox_inches='tight')
plt.show()



