import os
import cdsapi

years = [str(year) for year in range(2010, 2025)]

variables = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_solar_radiation_downwards",
    "total_sky_direct_solar_radiation_at_surface",
    "surface_pressure",
]

base_dir = "Users/rachelmurray/Documents/Heat_data/ERA5"

client = cdsapi.Client()

for year in years:
    for variable in variables:
        # one subdirectory per variable, mirroring your NEX-GDDP layout
        var_dir = os.path.join(base_dir, variable, year)
        os.makedirs(var_dir, exist_ok=True)
        os.chdir(var_dir)

        out_file = f"{variable}_{year}.nc"
        if os.path.exists(out_file):
            print(f"Skipping {variable} {year} — already downloaded")
            continue

        dataset = "reanalysis-era5-single-levels"
        request = {
            "product_type": ["reanalysis"],
            "variable": [variable],
            "year": year,
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [-9.36366167, 32.67161823, -17.12627881, 35.91841716],
        }

        print(f"Requesting {variable} for {year}...")
        client.retrieve(dataset, request).download(out_file)
