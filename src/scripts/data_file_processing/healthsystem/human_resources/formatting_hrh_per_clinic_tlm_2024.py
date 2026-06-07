from pathlib import Path

import numpy as np
import pandas as pd

resourcefilepath = Path("./resources")

path_to_tlm_tool_six = (
    resourcefilepath
    / "healthsystem"
    / "human_resources"
    / "TLM_2024"
    / "TLM_Tool_6_Facility_Level_TMS_v1_cleaned_v4.xlsx"
)

path_to_tlm_tool_six_cadre_name_cleaning = (
    resourcefilepath
    / "healthsystem"
    / "human_resources"
    / "TLM_2024"
    / "tool_6_cadre_name_clean.csv"
)

hrh_per_clinic = pd.read_excel(path_to_tlm_tool_six, sheet_name="Facility Level TMS")
cadre_name = pd.read_csv(path_to_tlm_tool_six_cadre_name_cleaning)

# issue: inconsistent cadre names
# transfer text to dict
def parse_staff_dict(x):
    if pd.isna(x):
        return {}
    x = str(x).strip()

    if x.startswith("{") and x.endswith("}"):
        x = x[1:-1]

    out = {}

    for item in x.split(","):
        item = item.strip()

        if not item:
            continue

        key, value = item.rsplit(":", 1)

        key = key.strip()
        value = float(value.strip())

        out[key] = out.get(key, 0) + value

    return out

cadre_num = hrh_per_clinic["formatted_cadre_and_number"].apply(parse_staff_dict).apply(pd.Series).fillna(0)

# correct original cadre name and group up same cadre names
cadre_mapping = dict(
    zip(cadre_name["original"],
        cadre_name["formatted"])
)
cadre_num = cadre_num.rename(columns=cadre_mapping)
cadre_num_formatted = cadre_num.T.groupby(level=0, sort=False).sum().T
assert (cadre_num_formatted.index == cadre_num.index).all()
assert set(cadre_num_formatted.columns).issubset(
    set(cadre_name["formatted"].drop_duplicates())
)

cadre_cat = cadre_name["main_hcw_seeing_patient"].drop_duplicates().reset_index(drop=True)
cadre_num_formatted_with_categories = cadre_num_formatted.copy()

cadre_num_formatted_with_categories["num_of_staff_all"] = cadre_num_formatted_with_categories.sum(axis=1)

for cat in cadre_cat:
    cadre_list = (
        cadre_name
        .loc[cadre_name["main_hcw_seeing_patient"] == cat, "formatted"]
        .dropna()
        .unique()
    )

    existing_cols = [
        col for col in cadre_list
        if col in cadre_num_formatted_with_categories.columns
    ]

    if existing_cols:
        cadre_num_formatted_with_categories[cat] = (
            cadre_num_formatted_with_categories[existing_cols]
            .fillna(0)
            .sum(axis=1)
        )
    else:
        cadre_num_formatted_with_categories[cat] = 0

cadre_num_formatted_with_categories["num_of_staff_a^"] = (
    cadre_num_formatted_with_categories[cadre_cat]
    .sum(axis=1)
)
assert (cadre_num_formatted_with_categories["num_of_staff_a^"] ==
        cadre_num_formatted_with_categories["num_of_staff_all"]
        ).all()

cadre_num_formatted_with_categories.drop(columns=["num_of_staff_a^"], inplace=True)

cadre_num_formatted_with_categories["num_of_staff_main_and_support"] =(
    cadre_num_formatted_with_categories[["Main", "Supporting staff"]].sum(axis=1)
)

cadre_num_formatted_with_categories = cadre_num_formatted_with_categories.rename(
    columns={"Main": "num_of_staff_main"}
)

id_vars = [
    "Facility ID",
    "Clinic/ward/department",
    "Opening date and time",
    "Closing date and time",
    "Total number of patients verified by records",
    "Number of staff",
    "Cadres of staff and number of staff per cadre",
    "Facility Type",
    "District"
]
tool_six_main = pd.concat([hrh_per_clinic[id_vars], cadre_num_formatted_with_categories], axis=1)

tool_six_main = tool_six_main.rename(
    columns={
        "Facility ID": "facility_id",
        "Clinic/ward/department": "clinic",
        "Opening date and time": "opening_date",
        "Closing date and time": "closing_date",
        "Total number of patients verified by records": "num_of_patients",
        "Number of staff": "num_of_staff_raw",
        "Cadres of staff and number of staff per cadre": "cadre_and_count_raw",
        "Facility Type": "facility_type",
        "District": "district"
    }
)

# calculate patient load per hcw
tool_six_main["pat_load_per_hcw_raw"] = tool_six_main["num_of_patients"] / tool_six_main["num_of_staff_raw"]
tool_six_main["pat_load_per_hcw_main"] = tool_six_main["num_of_patients"] / tool_six_main["num_of_staff_main"]
tool_six_main["pat_load_per_hcw_all"] = tool_six_main["num_of_patients"] / tool_six_main["num_of_staff_all"]
tool_six_main["pat_load_per_hcw_main_and_support"] = (tool_six_main["num_of_patients"] /
                                                      tool_six_main["num_of_staff_main_and_support"])

tool_six_main["same_obs_day"] = (
    tool_six_main["opening_date"].dt.normalize()
    == tool_six_main["closing_date"].dt.normalize()
)

tool_six_main["opening_hours"] = (
    tool_six_main["closing_date"] - tool_six_main["opening_date"]
).dt.total_seconds() / 3600

tool_six_main = tool_six_main.loc[(tool_six_main["same_obs_day"]) &  # do not consider night shift, 24-hour shift, etc.
                                  (tool_six_main["opening_hours"] > 0) &  # drop incorrect info
                                  np.isfinite(tool_six_main["pat_load_per_hcw_main"])
                                  # few inf rows is due to only supporting staff working that day
                                 ].copy()

tool_six_main.to_csv(
    resourcefilepath / "healthsystem" / "human_resources" / "TLM_2024"
    / "TLM_Tool_6_Facility_Level_TMS_v1_cleaned_v5.csv")


# # check the different between num_of_staff_all and num_of_staff_raw
# tool_six_main["num_of_staff_diff"] = tool_six_main["num_of_staff_raw"] - tool_six_main["num_of_staff_all"]
#
# # checked that 10 rows are involved; some have different number of patients/staff; no clear why duplicated;
# # so keep them for now
# duplicated_rows = tool_six_main[
#     tool_six_main.duplicated(subset=["facility_id", "clinic", "opening_date", "closing_date"], keep=False)
# ]
