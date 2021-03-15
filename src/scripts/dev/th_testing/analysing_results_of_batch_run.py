"""This file uses the results of the batch file to make some summary statistics.
The results of the bachrun were put into the 'outputs' folder
"""

from pathlib import Path


# import the class used to generate the batch
from scripts.dev.th_testing.mockitis_batch import Mockitis_Batch
bd = Mockitis_Batch()

# Declare output folder:
folder = Path('outputs/mockitis_batch-2021-03-15T125516Z')

# %% Unpack results to produce a dataframe that summaries one series from the log, with column multi-index for draw/run

# Define the log-element to extract:
log_element = {
    'component': 'tlo.method.mockitis',    # <--- the dataframe that is output
    'series': ''                     # <--- series in the dateframe to be extracted
}


for draw in range(bd.number_of_draws):
    for run in range(bd.runs_per_draw):

        log_component_file = folder / str(draw) / str(run) / str(log_element['component'] + '.pickle')



# get number of draws
ndrwas = bd.number_of_draws

# get number of runs


# Make summmaries of the log-component across the runs:
