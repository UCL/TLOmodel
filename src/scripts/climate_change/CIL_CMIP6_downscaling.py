import planetary_computer
import pystac_client

import xarray as xr
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from tqdm.auto import tqdm

import os
import re
import glob
import shutil
import zipfile
from pathlib import Path

import difflib
from scipy.spatial import KDTree

import matplotlib.pyplot as plt
import geopandas as gpd
import regionmask
import cartopy.crs as ccrs

from netCDF4 import Dataset

from carbonplan import styles  # noqa: F401
import intake
import cmip6_downscaling
from dask.distributed import Client
from planetary_computer import sign_inplace
from pystac_client import Client

# Open the catalog
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1/",
    modifier=sign_inplace,
)

# Get the collections
scenarios = ["ssp245"]  # Change as needed
variable_id = "pr"  # Precipitation variable

for scenario in scenarios:
    search = catalog.search(
        collections=["cil-gdpcir-cc0", "cil-gdpcir-cc-by"],
        query={"cmip6:experiment_id": {"eq": scenario}},
    )
    ensemble = search.item_collection()
    print(f"Number of items found: {len(ensemble)}")

    # Read and process each dataset
    datasets_by_model = []
    for item in tqdm(ensemble):
        asset = item.assets[variable_id]
        datasets_by_model.append(
            xr.open_dataset(asset.href, **asset.extra_fields["xarray:open_kwargs"])
        )

    # Combine datasets by model
    all_datasets = xr.concat(
        datasets_by_model,
        dim=pd.Index([ds.attrs["source_id"] for ds in datasets_by_model], name="model"),
        combine_attrs="drop_conflicts",
    )

    # Define the spatial and temporal bounds
    lon_bounds = slice(32.67161823, 35.91841716)
    lat_bounds = slice(-17.12627881, -9.36366167)
    time_range = pd.date_range("2065-01-01", "2071-01-01", freq="Y")

    # Process each year
    output_dir = "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL/"
    yearly_files = []
    for year in time_range.year:
        yearly_subset = all_datasets.pr.sel(
            lon=lon_bounds,
            lat=lat_bounds,
            time=slice(f"{year}-01-01", f"{year}-12-31"),
        )
        yearly_file = f"{output_dir}/CIL_subset_{scenario}_{year}.nc"
        yearly_subset.to_netcdf(yearly_file)
        yearly_files.append(yearly_file)
        print(f"Saved yearly data for {year} to {yearly_file}")

    # Combine all yearly files into one NetCDF file
    combined_output = f"{output_dir}/CIL_subsetted_all_model_{scenario}.nc"
    combined_dataset = xr.open_mfdataset(yearly_files, combine="by_coords")
    combined_dataset.to_netcdf(combined_output)
    print(f"Saved combined dataset to {combined_output}")
