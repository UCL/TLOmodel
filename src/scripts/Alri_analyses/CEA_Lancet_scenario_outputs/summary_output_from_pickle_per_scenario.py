""" This script will run the summary statistics per scenario (before bootstrap)  """

import random
from pathlib import Path
import os
from typing import List
import datetime
from math import e
from openpyxl import Workbook
from openpyxl import load_workbook
import scipy.stats as stats
import pickle

# from tlo.util import random_date, sample_outcome
import numpy.random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
from tlo.lm import LinearModel, LinearModelType, Predictor


# Store all the scenarios results
scenarios_summary = []  # Initialize empty list to store dataframes

scenarios = ['baseline_ant',
             # 'baseline_ant_with_po_level2', 'baseline_ant_with_po_level1b',
             # 'baseline_ant_with_po_level1a', 'baseline_ant_with_po_level0',
             # 'existing_psa', 'existing_psa_with_po_level2', 'existing_psa_with_po_level1b',
             # 'existing_psa_with_po_level1a', 'existing_psa_with_po_level0',
             # 'planned_psa', 'planned_psa_with_po_level2', 'planned_psa_with_po_level1b',
             # 'planned_psa_with_po_level1a',
             # 'planned_psa_with_po_level0'
             ]

dx_accuracy = 'imperfect'
# sa_name = ''
# sa_name = 'reduced_hw_dx'
# sa_name = 'perfect_hw_dx'
# sa_name = 'reduce_referral'
# sa_name = 'reduce_ox_effect'
# sa_name = 'reduce_incidence'
# sa_name = 'reduce_mortality'
# sa_name = 'planned_psa_70'
# sa_name = 'remove_death_adjustment'
# sa_name = 'oxygen_cost_50'
# sa_name = 'po_cost_double'
sa_name = 'out_inpatient_cost_double'



# Date for saving the image for log-file
datestamp = datetime.date.today().strftime("__%Y_%m_%d")


# Open all scenario outputs
for scenario in scenarios:

    with open(f'debug_output_{scenario}_{dx_accuracy}_{sa_name}.pkl', 'rb') as f:
        scenario_output = pickle.load(f)
        table = scenario_output

        # Get the total costs for the cohort
        all_variables = list()

        # # DALYs discounted
        # YLL_discounted_person = ((table['mortality_outcome'].astype(int)*(1-np.exp(-0.03*(
        #     54.7 - table['age_exact_years']))))/0.03)
        # DALYs_discounted = YLL_discounted_person.sum()

        low_oxygen = (table["oxygen_saturation"])
        seek_facility = (table["seek_level"])
        classification_by_seek_level = table[f'classification_in_{scenario}_{dx_accuracy}_hw_dx']
        final_facility = table[f'final_facility_scenario_{scenario}_{dx_accuracy}_hw_dx']
        final_facility_follow_up = table[f'final_facility_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx']

        all_variables.append({
            # Total deaths
            f'total_deaths': table['mortality_outcome'].sum(),
            # CFR
            f'CFR': table['mortality_outcome'].sum() / len(table),
            # DALYs discounted
            f'DALYs': ((table['mortality_outcome'].astype(int)*(1-np.exp(-0.03*(
                54.7 - table['age_exact_years']))))/0.03).sum(),
            # Total PO cost
            f'sum_po_cost': table['all_po_cost'].sum(),
            # Total oxygen cost
            f'sum_oxygen_cost': table['all_oxygen_cost'].sum(),
            # Total outpatient consultation cost
            f'sum_consultation_cost': table['all_outpatient_consultation_cost'].sum(),
            # Total oral antibiotic cost
            f'sum_oral_ant_cost': table['all_oral_amox_cost'].sum(),
            # Total IV antibiotic cost
            f'sum_iv_ant_cost': table['all_iv_antibiotics_cost'].sum(),
            # Total hospitalisation cost
            f'sum_hospitalisation_cost': table['all_inpatient_bed_cost'].sum(),
            f'total_costs': table['total_costs'].sum(),
            f'oxygen_need_detected': table.groupby(
                by=[classification_by_seek_level, low_oxygen, final_facility]).size().sum(level=[0, 1]).reindex(
                pd.MultiIndex.from_tuples([('danger_signs_pneumonia', '<90%')]))[0],
            f'oxygen_provided':
                table.loc[((final_facility == '2') | (final_facility == '1b')),
                          f'oxygen_provided_scenario_{scenario}_{dx_accuracy}_hw_dx'].sum() +
                table.loc[((final_facility_follow_up == '2') | (final_facility_follow_up == '1b')),
                          f'oxygen_provided_follow_up_scenario_{scenario}_{dx_accuracy}_hw_dx'].sum(),
            f'oxygen_liters_provided': table['all_total_oxygen_liters_used'].sum(),
        })

        totals_df = pd.DataFrame(all_variables)
        # Add scenario name as a column (important for identification)
        totals_df['scenario'] = scenario
        totals_df.insert(0, 'scenario', totals_df.pop('scenario'))
        scenarios_summary.append(totals_df)


# Concatenate all dataframes at once (more efficient)
final_df = pd.concat(scenarios_summary, ignore_index=True)
check = 0

