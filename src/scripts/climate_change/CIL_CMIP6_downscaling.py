#!/usr/bin/env python
# coding: utf-8

# From https://planetarycomputer.microsoft.com/dataset/cil-gdpcir-cc0#Ensemble-example

import collections

import numpy as np
# optional imports used in this notebook
import pandas as pd
import planetary_computer
import pystac_client
# required to load a zarr array using xarray
import xarray as xr
from dask.diagnostics import ProgressBar
from tqdm.auto import tqdm

# Load and organise data


catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1/",
    modifier=planetary_computer.sign_inplace,
)
collection_cc0 = catalog.get_collection("cil-gdpcir-cc0")

collection_cc0.summaries.to_dict()
print(collection_cc0)
search = catalog.search(
    collections=["cil-gdpcir-cc0", "cil-gdpcir-cc-by"], # both creative licenses
    query={"cmip6:experiment_id": {"eq": "ssp585"}},
)
ensemble = search.item_collection()
print(len(ensemble))

collections.Counter(x.collection_id for x in ensemble)

variable_id = "pr"

datasets_by_model = []

for item in tqdm(ensemble):
    asset = item.assets[variable_id]
    datasets_by_model.append(
        xr.open_dataset(asset.href, **asset.extra_fields["xarray:open_kwargs"])
    )

all_datasets = xr.concat(
    datasets_by_model,
    dim=pd.Index([ds.attrs["source_id"] for ds in datasets_by_model], name="model"),
    combine_attrs="drop_conflicts",
)


# Subset for Malawi and 2025-2100
subset = all_datasets.pr.sel(
    lon=slice(32.67161823,35.91841716),
    lat=slice(-17.12627881, -9.36366167),
    time=slice("2025-01-01", "2100-01-01"),
)

subset.to_netcdf("/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL/CIL_subsetted_all_model_ssp585.nc")
