from pathlib import Path
import pandas as pd
import numpy as np

base = Path("/Users/rem76/Desktop/Climate_Change_Health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL")

ssps = ["ssp126", "ssp245", "ssp585"]
models = ["lowest", "mean", "highest"]
WET_SEASON_MONTHS = {11, 12, 1, 2, 3, 4}

rows = []
for ssp in ssps:
    for model in models:
        path = base / ssp / f"{model}_monthly_prediction_weather_by_facility_ANC.csv"
        df = pd.read_csv(path)

        df["year"] = df["Unnamed: 0"].str.split("-").str[0].astype(int)
        df["month"] = df["Unnamed: 0"].str.split("-").str[1].astype(int)
        df = df[df["year"].between(2025, 2040)]

        facility_cols = [c for c in df.columns if c not in ["Unnamed: 0", "year", "month"]]

        # All months
        all_vals = df[facility_cols].values.flatten()

        # Wet season only
        wet = df[df["month"].isin(WET_SEASON_MONTHS)]
        wet_vals = wet[facility_cols].values.flatten()

        # Top decile of wet season months (extreme events)
        top_decile = wet_vals[wet_vals >= np.percentile(wet_vals, 90)]

        rows.append({
            "ssp": ssp,
            "model": model,
            "mean_all_months": all_vals.mean(),
            "mean_wet_season": wet_vals.mean(),
            "p95_wet_season": np.percentile(wet_vals, 95),
            "mean_top_decile": top_decile.mean(),
        })

results = pd.DataFrame(rows).sort_values("mean_wet_season", ascending=False)
print(results.to_string(index=False))
