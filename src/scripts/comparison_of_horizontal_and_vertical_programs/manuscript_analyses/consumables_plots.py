

import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import rgb_to_hsv


resourcefilepath = Path("./resources")
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

results_folder=Path("/Users/tmangal/PycharmProjects/TLOmodel/outputs/t.mangal@imperial.ac.uk/htm_and_hss_runs-2025-10-14T084418Z")
output_folder=Path("/Users/tmangal/PycharmProjects/TLOmodel/outputs/t.mangal@imperial.ac.uk/htm_and_hss_runs-2025-10-14T084418Z")


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





## HIV consumables

item_codes_dict = dict()

# diagnostics
item_codes_dict["HIV test"] = get_item_code_from_item_name(items_list, "Test, HIV EIA Elisa")
item_codes_dict["Viral load"] = get_item_codes_from_package_name(items_list, "Viral Load")
item_codes_dict["VMMC"] = get_item_code_from_item_name(items_list, "male circumcision kit, consumables (10 procedures)_1_IDA")

# treatment
item_codes_dict["Adult PrEP"] = get_item_code_from_item_name(items_list, "Tenofovir (TDF)/Emtricitabine (FTC), tablet, 300/200 mg")
item_codes_dict["Infant PrEP"] = get_item_code_from_item_name(items_list, "Nevirapine, oral solution, 10 mg/ml")
item_codes_dict['First-line ART regimen: adult'] = get_item_code_from_item_name(items_list, "First-line ART regimen: adult")
item_codes_dict['First-line ART regimen: adult: cotrimoxazole'] = get_item_code_from_item_name(items_list, "Cotrimoxizole, 960mg pppy")

# ART for older children aged ("ART_age_cutoff_younger_child" < age <= "ART_age_cutoff_older_child"):
item_codes_dict['First line ART regimen: older child'] = get_item_code_from_item_name(items_list, "First line ART regimen: older child")
item_codes_dict['First line ART regimen: older child: cotrimoxazole'] = get_item_code_from_item_name(items_list, "Sulfamethoxazole + trimethropin, tablet 400 mg + 80 mg")

# ART for younger children aged (age < "ART_age_cutoff_younger_child"):
item_codes_dict['First line ART regimen: young child'] = get_item_code_from_item_name(items_list, "First line ART regimen: young child")
item_codes_dict['First line ART regimen: young child: cotrimoxazole'] = get_item_code_from_item_name(items_list, "Sulfamethoxazole + trimethropin, oral suspension, 240 mg, 100 ml")

# select item codes from item_codes_dict
selected_cons_availability = average_cons_availability[average_cons_availability["item_code"].isin(item_codes_dict.values())]
# remove level 0
selected_cons_availability = selected_cons_availability.loc[selected_cons_availability.Facility_Level != "0"]
# remove level 4
selected_cons_availability = selected_cons_availability.loc[selected_cons_availability.Facility_Level != "4"]

# replace item code with item name
selected_cons_availability.loc[selected_cons_availability.item_code == 196, "item_code"] = "HIV test"
selected_cons_availability.loc[selected_cons_availability.item_code == 190, "item_code"] = "Viral load"
selected_cons_availability.loc[selected_cons_availability.item_code == 197, "item_code"] = "VMMC"
selected_cons_availability.loc[selected_cons_availability.item_code == 1191, "item_code"] = "Adult PrEP"
selected_cons_availability.loc[selected_cons_availability.item_code == 198, "item_code"] = "Infant PrEP"
selected_cons_availability.loc[selected_cons_availability.item_code == 2671, "item_code"] = "Adult ART"
selected_cons_availability.loc[selected_cons_availability.item_code == 204, "item_code"] = "Adult cotrimoxazole"
selected_cons_availability.loc[selected_cons_availability.item_code == 2672, "item_code"] = "Child ART"
selected_cons_availability.loc[selected_cons_availability.item_code == 162, "item_code"] = "Child cotrimoxazole"
selected_cons_availability.loc[selected_cons_availability.item_code == 2673, "item_code"] = "Infant ART"
selected_cons_availability.loc[selected_cons_availability.item_code == 202, "item_code"] = "Infant cotrimoxazole"


# Create the pivot table for the heatmap
df_heatmap = selected_cons_availability.pivot_table(
    values='available_prop',
    index='item_code',
    columns='Facility_Level',
    aggfunc='mean',
)


# Function to format annotation text
def annotate_heatmap(ax, data, valfmt="{x:.2f}", **textkw):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            text_color = 'white' if rgb_to_hsv(ax.collections[0].get_facecolors()[i * data.shape[1] + j][:3])[2] < 0.5 else 'black'
            ax.text(j + 0.5, i + 0.5, valfmt.format(x=value), ha='center', va='center', color=text_color, **textkw)

ax = sns.heatmap(
    df_heatmap,
    annot=False,
    cmap='RdYlGn_r',       # reversed red-yellow-green scale
    vmin=0,
    vmax=1,
    cbar_kws={'label': ''},
    fmt=".2f"
)
annotate_heatmap(ax, df_heatmap.values)

plt.tight_layout()
plt.xlabel('Facility Level')
plt.ylabel('')
plt.title('')
plt.savefig(output_folder / "hiv_cons_availability.png", bbox_inches='tight')
# plt.savefig(output_folder / "hiv_cons_availability.pdf", bbox_inches='tight')
plt.show()




## TB consumables

item_codes_dict = dict()

# diagnostics
item_codes_dict["sputum_test"] = get_item_codes_from_package_name(items_list, "Microscopy Test")
item_codes_dict["xpert_test"] = get_item_codes_from_package_name(items_list, "Xpert test")
item_codes_dict["chest_xray"] = get_item_code_from_item_name(items_list, "X-ray")

# treatment
item_codes_dict["tb_tx_adult"] = get_item_code_from_item_name(items_list, "Cat. I & III Patient Kit A")
item_codes_dict["tb_tx_child"] = get_item_code_from_item_name(items_list, "Cat. I & III Patient Kit B")
item_codes_dict["tb_retx_adult"] = get_item_code_from_item_name(items_list, "Cat. II Patient Kit A1")
item_codes_dict["tb_retx_child"] = get_item_code_from_item_name(items_list, "Cat. II Patient Kit A2")
item_codes_dict["tb_mdrtx"] = get_item_code_from_item_name(items_list, "Category IV")
item_codes_dict["tb_ipt"] = get_item_code_from_item_name(items_list, "Isoniazid/Pyridoxine, tablet 300 mg")

# select item codes from item_codes_dict
selected_cons_availability = average_cons_availability[average_cons_availability["item_code"].isin(item_codes_dict.values())]
# remove level 0
selected_cons_availability = selected_cons_availability.loc[selected_cons_availability.Facility_Level != "0"]
# remove level 4
selected_cons_availability = selected_cons_availability.loc[selected_cons_availability.Facility_Level != "4"]


# Change the dtype of 'item_code' to object to allow mixed types
selected_cons_availability['item_code'] = selected_cons_availability['item_code'].astype(object)


# replace item code with item name
selected_cons_availability.loc[selected_cons_availability.item_code == 184, "item_code"] = "Sputum test"
selected_cons_availability.loc[selected_cons_availability.item_code == 187, "item_code"] = "GeneXpert test"
selected_cons_availability.loc[selected_cons_availability.item_code == 175, "item_code"] = "Chest X-ray"
selected_cons_availability.loc[selected_cons_availability.item_code == 176, "item_code"] = "Adult TB treatment"
selected_cons_availability.loc[selected_cons_availability.item_code == 178, "item_code"] = "Child TB treatment"
selected_cons_availability.loc[selected_cons_availability.item_code == 177, "item_code"] = "Adult TB retreatment"
selected_cons_availability.loc[selected_cons_availability.item_code == 179, "item_code"] = "Child TB retreatment"
selected_cons_availability.loc[selected_cons_availability.item_code == 180, "item_code"] = "MDR-TB treatment"
selected_cons_availability.loc[selected_cons_availability.item_code == 192, "item_code"] = "TPT"


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

ax = sns.heatmap(
    df_heatmap,
    annot=False,
    cmap='RdYlGn_r',       # reversed red-yellow-green scale
    vmin=0,
    vmax=1,
    cbar_kws={'label': ''},
    fmt=".2f"
)
annotate_heatmap(ax, df_heatmap.values)

plt.tight_layout()
plt.xlabel('Facility Level')
plt.ylabel('')
plt.title('')
plt.savefig(output_folder / "tb_cons_availability.png", bbox_inches='tight')
# plt.savefig(output_folder / "tb_cons_availability.pdf", bbox_inches='tight')
plt.show()





## Malaria consumables

item_codes_dict = dict()

# diagnostics
item_codes_dict["RDT test"] = get_item_code_from_item_name(items_list, "Malaria test kit (RDT)")


# treatment
item_codes_dict["malaria_uncomplicated"] = get_item_code_from_item_name(items_list, "Lumefantrine 120mg/Artemether 20mg,  30x18_540_CMST")
item_codes_dict['malaria_complicated_artesunate'] = get_item_code_from_item_name(items_list, "Injectable artesunate")
item_codes_dict['malaria_iptp'] = get_item_code_from_item_name(items_list, "Fansidar (sulphadoxine / pyrimethamine tab)")
item_codes_dict['malaria_paracetamol'] = get_item_code_from_item_name(items_list, "Paracetamol syrup 120mg/5ml_0.0119047619047619_CMST")

# select item codes from item_codes_dict
selected_cons_availability = average_cons_availability[average_cons_availability["item_code"].isin(item_codes_dict.values())]
# remove level 0
selected_cons_availability = selected_cons_availability.loc[selected_cons_availability.Facility_Level != "0"]
# remove level 4
selected_cons_availability = selected_cons_availability.loc[selected_cons_availability.Facility_Level != "4"]

# replace item code with item name
selected_cons_availability.loc[selected_cons_availability.item_code == 163, "item_code"] = "Malaria RDT"
selected_cons_availability.loc[selected_cons_availability.item_code == 164, "item_code"] = "Lumefantrine / Artemether"
selected_cons_availability.loc[selected_cons_availability.item_code == 170, "item_code"] = "Injectable artesunate"
selected_cons_availability.loc[selected_cons_availability.item_code == 1124, "item_code"] = "IPTp"
selected_cons_availability.loc[selected_cons_availability.item_code == 71, "item_code"] = "Paracetamol"


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

ax = sns.heatmap(
    df_heatmap,
    annot=False,
    cmap='RdYlGn_r',       # reversed red-yellow-green scale
    vmin=0,
    vmax=1,
    cbar_kws={'label': 'Proportion of days consumables are available'},
    fmt=".2f"
)
annotate_heatmap(ax, df_heatmap.values)

plt.tight_layout()
plt.xlabel('Facility Level')
plt.ylabel('')
plt.title('')
plt.savefig(output_folder / "malaria_cons_availability.png", bbox_inches='tight')
# plt.savefig(output_folder / "malaria_cons_availability.pdf", bbox_inches='tight')
plt.show()



