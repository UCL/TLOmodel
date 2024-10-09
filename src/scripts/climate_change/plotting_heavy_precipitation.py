import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#
# def apply(results_folder: Path, output_folder: Path):
output_folder = Path("/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/Climate_health")

all_data_max_5_day_precip = pd.ExcelFile(
    '/Users/rem76/Documents/Climate_change_data/Max_5_day_precip/ssp126/cmip6-x0.25_timeseries_rx5day_timeseries_seasonal_2015-2100_p90_ssp126_ensemble_all.xlsx')
median_max_5_day_precip = pd.read_excel(all_data_max_5_day_precip, 'median')
median_max_5_day_precip.drop("code", inplace=True, axis=1)
p10_max_5_day_precip = pd.read_excel(all_data_max_5_day_precip, 'p10')
p10_max_5_day_precip.drop("code", inplace=True, axis=1)
p90_max_5_day_precip = pd.read_excel(all_data_max_5_day_precip, 'p90')
p90_max_5_day_precip.drop("code", inplace=True, axis=1)

median_values = pd.to_numeric(
    median_max_5_day_precip[median_max_5_day_precip['name'] == 'Central Region'].iloc[0, 1:].values, errors='coerce')
p10_values = pd.to_numeric(p10_max_5_day_precip[median_max_5_day_precip['name'] == 'Central Region'].iloc[0, 1:].values,
                           errors='coerce')
p90_values = pd.to_numeric(p90_max_5_day_precip[median_max_5_day_precip['name'] == 'Central Region'].iloc[0, 1:].values,
                           errors='coerce')
median_values = np.nan_to_num(median_values, nan=0.0, posinf=0.0, neginf=0.0)
p10_values = np.nan_to_num(p10_values, nan=0.0, posinf=0.0, neginf=0.0)
p90_values = np.nan_to_num(p90_values, nan=0.0, posinf=0.0, neginf=0.0)
name_of_plot = f'Max 5-day precipitation (mm) under ssp126'
x_labels = [p10_max_5_day_precip.columns[i] if i % 4 == 0 and i >= 2 else "" for i in range(len(median_values))]

make_graph_file_name = lambda stub: output_folder / f"Precipitation_by_region_{stub}.png"  # noqa: E731

fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(range(len(median_values)), median_values, label='Central Region', color='red', marker='o')
ax.fill_between(range(len(median_values)), p10_values, p90_values, color="grey", alpha=0.3)
ax.set_title(name_of_plot, {'size': 12, 'color': 'black'})
ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(x_labels, rotation=45, ha='right')
fig.savefig(make_graph_file_name(name_of_plot))
ax.set_title(name_of_plot, {'size': 12, 'color': 'black'})
print(make_graph_file_name(name_of_plot))
fig.savefig(make_graph_file_name(name_of_plot))
plt.close(fig)
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("results_folder", type=Path)
#     args = parser.parse_args()
#     apply(
#         results_folder=args.results_folder,
#         output_folder=args.results_folder)
