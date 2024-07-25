import datetime
from pathlib import Path
import os

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
import seaborn as sns
import lacroix
import math

from tlo.analysis.utils import (
    compare_number_of_deaths,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)
from tlo import Date

resourcefilepath = Path("./resources")
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

outputspath = Path("./outputs/t.mangal@imperial.ac.uk")

# Example data
data = {
    'item_code': ['Chest X-ray', 'Chest X-ray', 'Chest X-ray', 'Chest X-ray',
                  'Adult treatment', 'Adult treatment', 'Adult treatment', 'Adult treatment',
                  'Adult retreatment', 'Adult retreatment', 'Adult retreatment', 'Adult retreatment',
                  'Child treatment', 'Child treatment', 'Child treatment', 'Child treatment',
                  'Child retreatment', 'Child retreatment', 'Child retreatment', 'Child retreatment',
                  'Sputum test', 'Sputum test', 'Sputum test', 'Sputum test',
                  'GeneXpert test', 'GeneXpert test', 'GeneXpert test', 'GeneXpert test',
                  'IPT', 'IPT', 'IPT', 'IPT'],
    'Facility_Level': ['1a', '1b', '2', '3',
                       '1a', '1b', '2', '3',
                       '1a', '1b', '2', '3',
                       '1a', '1b', '2', '3',
                       '1a', '1b', '2', '3',
                       '1a', '1b', '2', '3',
                       '1a', '1b', '2', '3',
                       '1a', '1b', '2', '3'],
    'available_prop': [0.000000, 0.489697, 0.513891, 0.580488,
                       0.638823, 0.622227, 0.873183, 0.857766,
                       0.529122, 0.493359, 0.646302, 0.713457,
                       0.557301, 0.571757, 0.715505, 0.784691,
                       0.529122, 0.493359, 0.646302, 0.713457,
                       0.850000, 0.900000, 0.950000, 0.990000,
                       0.000000, 0.000000, 0.310136, 0.263678,
                       0.393363, 0.525564, 0.687006, 0.737243]
}

# Create a DataFrame
selected_cons_availability = pd.DataFrame(data)

# Ensure data types are correct
selected_cons_availability['item_code'] = selected_cons_availability['item_code'].astype(str)
selected_cons_availability['Facility_Level'] = selected_cons_availability['Facility_Level'].astype(str)
selected_cons_availability['available_prop'] = selected_cons_availability['available_prop'].astype(float)

# Create the pivot table for the heatmap
df_heatmap = selected_cons_availability.pivot_table(
    values='available_prop',
    index='item_code',
    columns='Facility_Level',
    aggfunc='mean'
)

# Create the heatmap
plt.figure(figsize=(12, 10))

# Function to format annotation text
def annotate_heatmap(ax, data, valfmt="{x:.2f}", **textkw):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j + 0.5, i + 0.5, valfmt.format(x=data[i, j]), ha='center', va='center', **textkw)

ax = sns.heatmap(df_heatmap, annot=False, cbar_kws={'label': 'Availability Proportion'}, fmt=".2f")
annotate_heatmap(ax, df_heatmap.values)

plt.tight_layout()
plt.xlabel('Facility Level')
plt.ylabel('Item Code')
plt.title('')
# Uncomment the following lines to save the figure
# plt.savefig(outputspath / "cons_availability.png", bbox_inches='tight')
# plt.savefig(outputspath / "cons_availability.pdf", bbox_inches='tight')
plt.show()
