"""
generate_expanded_facility_info.py

Generates expanded_facility_info_by_all_facilities_lm.csv from
facilities_with_lat_long_region.csv.

No weather/NetCDF data needed — purely facility metadata.
Output format matches the original expanded_facility_info files:
    rows = variables, cols = facility Fname
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

DATA_PATH = "/Users/rem76/Desktop/Climate_Change_Health/Data/"

# ── Load facilities ───────────────────────────────────────────────────────────
facilities = pd.read_csv(
    f"{DATA_PATH}facilities_with_lat_long_region.csv",
    low_memory=False
)
facilities = facilities.drop_duplicates(subset="Fname", keep="first").reset_index(drop=True)
facilities = facilities.dropna(subset=["A109__Latitude", "A109__Longitude"]).reset_index(drop=True)

print(f"Facilities with coordinates: {len(facilities)}")

# ── Extract relevant columns ──────────────────────────────────────────────────
cols = ["Fname", "Zonename", "Resid", "Dist", "A105",
        "A109__Altitude", "Ftype", "A109__Latitude", "A109__Longitude"]

expanded = facilities[cols].copy()

# Fix known district name typos (matches original script)
expanded["Dist"] = expanded["Dist"].replace(
    {"Blanytyre": "Blantyre", "Nkhatabay": "Nkhata Bay"}
)

expanded = expanded.set_index("Fname")

# ── Compute minimum_distance (Euclidean on raw lat/lon — matches original) ────
coords = expanded[["A109__Latitude", "A109__Longitude"]].values
distances = cdist(coords, coords, metric="euclidean")
np.fill_diagonal(distances, np.inf)
expanded["minimum_distance"] = np.nanmin(distances, axis=1)

# ── Transpose to match original format: rows = variables, cols = Fname ────────
expanded = expanded.T

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = f"{DATA_PATH}expanded_facility_info_by_all_facilities_lm.csv"
expanded.to_csv(out_path)
print(f"Saved to: {out_path}")
print(f"Shape: {expanded.shape}  (variables × facilities)")
