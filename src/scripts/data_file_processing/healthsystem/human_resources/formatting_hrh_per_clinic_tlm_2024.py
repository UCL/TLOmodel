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

hrh_per_clinic = pd.read_excel(path_to_tlm_tool_six, sheet_name="Facility Level TMS")

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

id_vars = [
    "Facility ID",
    "Clinic/ward/department",
    "Opening date and time",
    "Closing date and time",
    "Total number of patients verified by records",
    "Number of staff"
]
df_formatted = pd.concat([hrh_per_clinic[id_vars], cadre_num], axis=1)
# todo: calculate number of HCWs who actually see patients

# issue: some duplicate rows


duplicated_rows = hrh_per_clinic[
    hrh_per_clinic.duplicated(subset=id_vars, keep=False)
]
