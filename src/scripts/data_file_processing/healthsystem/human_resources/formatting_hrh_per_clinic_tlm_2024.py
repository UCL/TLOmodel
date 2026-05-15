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

id_vars = [
    "Facility ID",
    "Clinic/ward/department",
    "Opening date and time",
    "Closing date and time",
]

duplicated_rows = hrh_per_clinic[
    hrh_per_clinic.duplicated(subset=id_vars, keep=False)
]
