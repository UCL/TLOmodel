
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import squarify

from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)

outputspath = Path('./outputs/rmjlra2@ucl.ac.uk/')

results_folder = get_scenario_outputs('rti_recalibrate_mortality.py', outputspath)[- 1]
# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)
# get main paper results, incidence of RTI, incidence of death and DALYs
extracted_incidence_of_death = extract_results(results_folder,
                                               module="tlo.methods.rti",
                                               key="summary_1m",
                                               column="incidence of rti death per 100,000",
                                               index="date"
                                               )
extracted_incidence_of_RTI = extract_results(results_folder,
                                             module="tlo.methods.rti",
                                             key="summary_1m",
                                             column="incidence of rti per 100,000",
                                             index="date"
                                             )
mean_incidence_of_death = summarize(extracted_incidence_of_death, only_mean=True).mean()
mean_incidence_of_RTI = summarize(extracted_incidence_of_RTI, only_mean=True).mean()
scale_for_inc = np.divide(954.2, mean_incidence_of_RTI)
scaled_inc_death = np.multiply(mean_incidence_of_death, scale_for_inc)
plt.bar(np.arange(len(params)), scaled_inc_death, color='lightsteelblue', label='Model estimates')
plt.axhline(35, color='r', label='WHO estimate')
plt.axhline(12.1, color='c', label='GBD estimate')
plt.ylabel('Incidence of death per 100,000')
plt.legend()
plt.xticks(np.arange(len(params)), params.value)
plt.xlabel('unavailable_treatment_mortality_iss_cutoff')
plt.show()
