from pathlib import Path

import xarray as xr

BASE_DIR = Path(
    "/Users/rachelmurray-watson/Documents/Heat_data/NASA_GDDP-CMIP6"
)

OUT_DIR = Path(
    "/Users/rachelmurray-watson/Documents/Heat_data/NASA_GDDP-CMIP6/Combined"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = ["ssp245"]
VARIABLES = ["hurs", "huss", "rlds", "rsds", "sfcWind", "tas", "tasmax", "tasmin"]

YEAR_START = 2025
YEAR_END = 2040


for model_dir in BASE_DIR.iterdir():
    if not model_dir.is_dir():
        continue

    model = model_dir.name

    # Don't recurse into the Combined output directory itself
    if model == "Combined":
        continue

    for scenario in SCENARIOS:
        for variable in VARIABLES:

            var_dir = model_dir / scenario / variable
            if not var_dir.exists():
                continue

            files = sorted(
                var_dir.glob(
                    f"{variable}_day_{model}_{scenario}_*_gn_*_malawi.nc"
                )
            )

            if not files:
                print(f"No files for {model} {scenario} {variable}")
                continue

            print(f"→ Combining {model} {scenario} {variable}")
            print(f"  {len(files)} files")

            ds = xr.open_mfdataset(
                files,
                combine="by_coords",
                decode_times=True,
                parallel=False
            )

            out_file = (
                OUT_DIR
                / f"{variable}_day_{model}_{scenario}_malawi_{YEAR_START}_{YEAR_END}.nc"
            )

            ds.to_netcdf(
                out_file,
                format="NETCDF4",
                encoding={
                    variable: {
                        "zlib": True,
                        "complevel": 4
                    }
                }
            )

            ds.close()
            print(f"Saved {out_file.name}")

print("\nDone combining all files.")
