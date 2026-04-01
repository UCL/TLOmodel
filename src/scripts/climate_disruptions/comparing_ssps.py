from pathlib import Path
import pandas as pd

base = Path("/Users/rem76/Desktop/Climate_Change_Health/Data/Precipitation_data/Downscaled_CMIP6_data_CIL")

ssps = ["ssp126", "ssp245", "ssp585"]
models = ["lowest", "mean", "highest"]

rows = []
for ssp in ssps:
    for model in models:
        path = base / ssp / f"{model}_monthly_prediction_weather_by_facility_ANC.csv"
        df = pd.read_csv(path)

        # Filter to 2025-2040 — adjust column name if needed
        df = df[df["year"].between(2025, 2040)]

        rows.append({
            "ssp": ssp,
            "model": model,
            "mean_precip_mm": df["precipitation"].mean(),  # adjust column name if needed
            "p95_precip_mm": df["precipitation"].quantile(0.95),
        })

results = pd.DataFrame(rows).sort_values("mean_precip_mm", ascending=False)
print(results.to_string(index=False))
