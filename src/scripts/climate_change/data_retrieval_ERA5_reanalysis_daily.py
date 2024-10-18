import os

import cdsapi
years = ["2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021",
         "2022", "2023", "2024"]
base_dir = "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/Historical/daily_maximum"

for year in years:
        year_dir = os.path.join(base_dir, year)
        if not os.path.exists(year_dir):
            os.makedirs(year_dir)
        os.chdir(year_dir)
        dataset = "reanalysis-era5-single-levels"
        request = {
            "product_type": ["reanalysis"],
            "variable": ["total_precipitation"],
            "year": year,
            "month": [
                "01", "02", "03",
                "04", "05", "06",
                "07", "08", "09",
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
            "daily_statistic": "daily_maximum",
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            "area": [-9.36366167, 32.67161823, -17.12627881, 35.91841716]
        }

        client = cdsapi.Client()
        client.retrieve(dataset, request).download()
