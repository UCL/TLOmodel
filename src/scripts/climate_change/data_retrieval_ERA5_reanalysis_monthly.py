import os

import cdsapi

base_dir = "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical/monthly_data"
os.chdir(base_dir)
dataset = "reanalysis-era5-single-levels-monthly-means"
request = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": ["total_precipitation"],
    "year": [
        "2011", "2012", "2013",
        "2014", "2015", "2016",
        "2017", "2018", "2019",
        "2020", "2021", "2022",
        "2023", "2024"
    ],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [-9.36366167, 32.67161823, -17.12627881, 35.91841716]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
