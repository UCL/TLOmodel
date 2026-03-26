from pathlib import Path

import pandas as pd

from tlo import logging
from tlo.analysis.utils import parse_log_file


def get_disruptions_by_facility(log_dir: str | Path) -> pd.DataFrame:
    """Parse healthsystem summary log and return total disruptions by RealFacility_ID."""
    log_dir = Path(log_dir)

    # Find the log file - either a .log file or a directory path
    if log_dir.is_dir():
        log_files = list(log_dir.glob("*.log"))
        if not log_files:
            raise FileNotFoundError(f"No .log files found in {log_dir}")
        log_filepath = log_files[0]
    else:
        log_filepath = log_dir

    print(f"Parsing log file: {log_filepath}")
    log = parse_log_file(log_filepath, level=logging.DEBUG)

    hs_summary = log.get("tlo.methods.healthsystem.summary", {})

    if "WeatherDisruptions" not in hs_summary:
        print("No WeatherDisruptions data found in log.")
        return pd.DataFrame()

    disruptions_df = hs_summary["WeatherDisruptions"]
    print(f"Columns available: {disruptions_df.columns.tolist()}")

    # Aggregate total disruptions by RealFacility_ID
    disruptions_by_facility = (
        disruptions_df
        .groupby("RealFacility_ID")
        .agg(
            total_disrupted=("n_disrupted", "sum"),
            n_months=("date", "count"),
        )
        .reset_index()
        .sort_values("total_disrupted", ascending=False)
    )

    return disruptions_by_facility


# --- Run ---
log_dir = "/Users/rem76/PycharmProjects/TLOmodel/outputs/climate_disruption_scenario-2026-03-06T155802Z/0/0"
disruptions = get_disruptions_by_facility(log_dir)
print(disruptions.head(20))
