import xarray as xr
base_dir = "/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT/ERA5_reanalysis"
WBGT_DIRECTORY = "/Users/rachelmurray-watson/Documents/Heat_data/Thermofeel_WBGT"

ds = xr.open_mfdataset(f"{base_dir}/total_precipitation/*/*.nc")
tp = ds["tp"]                      # confirm short name in your file; check the time dim name too

TIME = "valid_time"               # newer CDS netcdf uses valid_time; older files use "time"
daily = (tp * 1000.0).resample({TIME: "1D"}).sum()          # mm/day

precip_month = daily.resample({TIME: "1MS"}).sum().rename("precip_month")
r5           = daily.rolling({TIME: 5}, min_periods=5).sum()
precip_5day  = r5.resample({TIME: "1MS"}).max().rename("precip_5day")

out = xr.Dataset({"precip_month": precip_month, "precip_5day": precip_5day})
out = out.rename({TIME: "time"})  # match WBGT_TIME_COORD
out.to_netcdf(f"{WBGT_DIRECTORY}/precip_monthly_ERA5_historical.nc")
