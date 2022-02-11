
# need scipy version 1.7 so have to update package (in requirements v1.6 used)
from pathlib import Path

import pandas as pd
from scipy.stats import qmc

number_of_draws = 20

# set up LHC sampler
sampler = qmc.LatinHypercube(d=2)
sample = sampler.random(n=number_of_draws)

l_bounds = [0.08, 1.5]
u_bounds = [0.18, 2.00]
sampled_params = pd.DataFrame(qmc.scale(sample, l_bounds, u_bounds))

# write to csv
outputpath = Path("./outputs")  # folder for convenience of storing outputs

writer = pd.ExcelWriter(outputpath / ("LHC_Samples" + ".xlsx"))
sampled_params.to_excel(writer, sheet_name="LHC_samples")
writer.save()
