import os

import cdsapi

years = [str(year) for year in range(2011, 2024)]

base_dir = "/Users/rem76/Desktop/Climate_change_health/Data/Temperature_data/2m_dewpoint/Historical/hourly/"

for year in years:
    year_dir = os.path.join(base_dir, year)
    if not os.path.exists(year_dir):
        os.makedirs(year_dir)
    os.chdir(year_dir)
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
        "2m_dewpoint_temperature",
    ],
        "year": year,
        "month": ["1", "2", "3", "4", "5", "6", "7", "8", "9",
            "10", "11", "12"
        ],
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "time": ["00:00", "01:00", "02:00",
                 "03:00", "04:00", "05:00",
                 "06:00", "07:00", "08:00",
                 "09:00", "10:00", "11:00",
                 "12:00", "13:00", "14:00",
                 "15:00", "16:00", "17:00",
                 "18:00", "19:00", "20:00",
                 "21:00", "22:00", "23:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [-9.36366167, 32.67161823, -17.12627881, 35.91841716]
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request).download()
