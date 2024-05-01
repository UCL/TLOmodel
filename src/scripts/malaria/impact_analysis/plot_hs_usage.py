"""This file uses the results of the scenario runs to generate plots

* bar plot showing the numbers of opd, inpat and emergency appts across the draws

"""

import datetime
from pathlib import Path

# import lacroix
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns

from tlo import Date

from tlo.analysis.utils import (
    compare_number_of_deaths,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
    make_age_grp_lookup,
    make_age_grp_types,
)

# outputspath = Path("./outputs/t.mangal@imperial.ac.uk")

outputspath = Path("./outputs")

# Find results_folder associated with a given batch_file (and get most recent [-1])
# results_folder = get_scenario_outputs("exclude_HTM_services.py", outputspath)[-1]
results_folder = Path("./outputs/exclude_HTM_services_Apr2024")

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"  # noqa: E731

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
scenario_info = get_scenario_info(results_folder)

# Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)


# -------------------- summary appt numbers ------------------------------ #

# get numbers of appt types from outputs (extract_tx_numbers) and edit in spreadsheet
# outpatient, inpatient, emergency
# mean numbers of appts across the 5 runs, values have been scaled to full pop size
# todo not updated
sc0 = [1014736527.00706, 118017802.698474, 8371732.386892000]
sc1 = [1001762514.02496, 112815850.469672, 8271380.005574000]
sc2 = [1016739067.35463, 117661524.119538, 8334772.700814]
sc3 = [1017297708.21162, 115089176.495258, 9084900.208342]
sc4 = [1011410591.44831, 119425614.879508, 9106157.116700000]

data = {'Column1': sc0, 'Column2': sc1, 'Column3': sc2, 'Column4': sc3, 'Column5': sc4}
df = pd.DataFrame(data)


# Sample data
categories = ['Status Quo', 'HIV services excl', 'TB service excluded', 'Malaria services excl', 'Full impact']

bar_positions = list(range(len(categories)))
bar_width = 0.35

# Define y-axis label array
y_labels = ['1.00', '1.02', '1.04', '1.06', '1.08', '1.10', '1.12', '1.14']

plt.figure(figsize=(7, 5))
plt.subplots_adjust(left=0.15, right=0.75, top=0.85, bottom=0.25)

# Create a stacked bar plot
plt.bar(categories, df.iloc[0], width=bar_width,
        label='OPD')
plt.bar(categories, df.iloc[1], width=bar_width,
        bottom=df.iloc[0],
        label='Inpatient')
plt.bar(categories, df.iloc[2], width=bar_width,
        bottom=[d1 + d2 for d1, d2 in zip(df.iloc[0], df.iloc[1])],
        label='Emergency')

# Add labels and legend
plt.yscale('log')
plt.xlabel('')
plt.xticks(bar_positions, categories, rotation=45, ha='right')

plt.ylim(1.0e+9, 1.15e+9)
# Set the y-axis labels
plt.yticks([1.0e+9, 1.02e+9, 1.04e+9, 1.06e+9, 1.08e+9, 1.10e+9, 1.12e+9, 1.14e+9],
           y_labels)

# Add "x10^9" to the axis at the top
plt.text(-0.7, 1.155e+9, 'x10e9', fontsize=10, verticalalignment='center')

plt.ylabel('Log number of appts')
plt.title('')

# Add legend to the right of the plot
plt.legend(loc='center left', bbox_to_anchor=(1, 1))

plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')

# Show the plot
# plt.savefig(outputspath / "appt_type_by_scenario.png")

plt.show()
# plt.close()









