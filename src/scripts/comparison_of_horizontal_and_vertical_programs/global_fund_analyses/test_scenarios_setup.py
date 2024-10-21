import datetime
from pathlib import Path
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from tlo.analysis.utils import (
    compare_number_of_deaths,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)

results_folder=Path('/Users/tmangal/PycharmProjects/TLOmodel/outputs/hss_elements-2024-10-21T084526Z')

log = load_pickled_dataframes(results_folder, draw=1, run=0)
info = get_scenario_info(results_folder)
info

params = extract_params(results_folder)
params

consAVAIL = extract_results(
    results_folder,
    module='tlo.methods.healthsystem.summary',
    key='Consumables',
    column='Item_Available',
    do_scaling=False
)
