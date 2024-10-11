import os

import cdsapi

models = ["cams_csm1_0", "ipsl_cm6a_lr", "miroc6","miroc_es2l", "mri_esm2_0", "canesm5", "cnrm_esm2_1", "ec_earth3", "ec_earth3_veg_lr", "fgoals_g3", "gfdl_esm4", "ukesm1_0_ll"]
scenarios = ["ssp1_1_9","ssp1_2_6", "ssp4_3_4", "ssp5_3_4OS", "ssp2_4_5", "ssp4_6_0", "ssp3_7_0", "ssp5_8_5"]
base_dir = "/Users/rem76/Desktop/Climate_change_health/Data/Precipitation_data/"
os.chdir(base_dir)
for scenario in scenarios:
    scenario_dir = os.path.join(base_dir, scenario)
    if not os.path.exists(scenario_dir):
        os.makedirs(scenario_dir)
    os.chdir(scenario_dir)
    for model in models:
        dataset = "projections-cmip6"
        request = {
            "temporal_resolution": "daily",
            "experiment": scenario,
            "variable": "precipitation",
            "model": model,
            "year": [
                "2050", "2051", "2052",
                "2053", "2054", "2055",
                "2056", "2057", "2058",
                "2059", "2060", "2061",
                "2062", "2063", "2064",
                "2065", "2066", "2067",
                "2068", "2069", "2070",
                "2071", "2072", "2073",
                "2074", "2075", "2076",
                "2077", "2078", "2079",
                "2080", "2081", "2082",
                "2083", "2084", "2085",
                "2086", "2087", "2088",
                "2089", "2090", "2091",
                "2092", "2093", "2094",
                "2095", "2096", "2097",
                "2098", "2099", "2015",
                "2016", "2017", "2018",
                "2019", "2020", "2021",
                "2022", "2023", "2024",
                "2025", "2026", "2027",
                "2028", "2029", "2030",
                "2031", "2032", "2033",
                "2034", "2035", "2036",
                "2037", "2038", "2039",
                "2040", "2041", "2042",
                "2043", "2044", "2045",
                "2046", "2047", "2048",
                "2049"
            ],
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


            'area': [-9.36366167,  35.91841716,  -17.12627881, 32.67161823, ] # boundaries for all of Malawi
        }

        client = cdsapi.Client()
        client.retrieve(dataset, request).download()

