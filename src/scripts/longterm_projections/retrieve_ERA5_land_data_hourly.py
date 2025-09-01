import cdsapi
import os

years = [str(year) for year in range(2019, 2020)]

base_dir = "/Users/rem76/Desktop/Climate_change_health/Data/Temperature_data/Historical/Hourly"

dataset = "reanalysis-era5-land"
for year in years:
    year_dir = os.path.join(base_dir, year)
    request = {
        "variable": [
            "2m_dewpoint_temperature",
            "2m_temperature"
        ],
        "year": year,
        "month": [str(month) for month in range(1, 6)],
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
        "time": [
            "05:00",
            "06:00", "07:00", "08:00",
            "09:00", "10:00", "11:00",
            "12:00", "13:00", "14:00",
            "15:00", "16:00", "17:00",
            "18:00", "19:00",
        ],
        "data_format": "netcdf",
        "download_format": "unarchived"
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request).download()
