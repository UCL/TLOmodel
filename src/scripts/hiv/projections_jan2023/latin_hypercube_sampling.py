
# need scipy version 1.7 so have to update package (in requirements v1.6 used)
from pathlib import Path

import pandas as pd
import scipy.stats as sc

number_of_draws = 20

# set up LHC sampler
sampler = sc.qmc.LatinHypercube(d=2)
sample = sampler.random(n=number_of_draws)

l_bounds = [0.10, 0.17]  # hiv then tb
u_bounds = [0.15, 0.21]
sampled_params = pd.DataFrame(sc.qmc.scale(sample, l_bounds, u_bounds))

# write to excel
outputpath = Path("./outputs")  # folder for convenience of storing outputs

with pd.ExcelWriter(outputpath / ("LHC_Samples_Dec2022" + ".xlsx"), engine='openpyxl') as writer:
    sampled_params.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.save()
