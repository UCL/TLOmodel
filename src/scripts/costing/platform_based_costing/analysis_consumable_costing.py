"""Produce outputs for PLatform-based costing of consumables.
This analysis was performed for MOH's 2025 Platform-based Costing Exercise
"""

import datetime
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict
from ast import literal_eval

from scripts.costing.cost_estimation import load_unit_cost_assumptions
from tlo import Date
from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    compute_summary_statistics
)

# Define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# Print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

# Create folders to store results
resourcefilepath = Path("./resources")
outputfilepath = Path('./outputs/')
figurespath = Path('./outputs/platform_based_costing/')
path_for_consumable_resourcefiles = resourcefilepath / "healthsystem/consumables"
if not os.path.exists(figurespath):
    os.makedirs(figurespath)

from tlo.analysis.utils import create_pickles_locally

# Load result files
# ------------------------------------------------------------------------------------------------------------------
results_folder = get_scenario_outputs('consumables_costing-2025-12-04T200931Z.py', outputfilepath)[0] # consumables_costing-2025-12-02T185908Z

#log = pd.read_csv(results_folder1 / "0" / "0" / 'impact_of_consumables_availability__2025-11-25T120830.log', sep="\t")
#create_pickles_locally(scenario_output_dir = "outputs/consumables_costing-2025-12-04T200931Z")

# Check can read results from draw=0, run=0
log = load_pickled_dataframes(results_folder, 0, 0)  # look at one log (so can decide what to extract)
params = extract_params(results_folder)
info = get_scenario_info(results_folder)

# Declare default parameters for cost analysis
# ------------------------------------------------------------------------------------------------------------------
# Period relevant for costing
TARGET_PERIOD = (Date(2010, 1, 1), Date(2030, 12, 31))  # This is the period that is costed
relevant_period_for_costing = [i.year for i in TARGET_PERIOD]
list_of_relevant_years_for_costing = list(range(relevant_period_for_costing[0], relevant_period_for_costing[1] + 1))
list_of_years_for_plot = list(range(2010, 2031))
number_of_years_costed = relevant_period_for_costing[1] - 2023 + 1

# Scenarios
cost_scenarios = {0: "Actual", 1: "Perfect consumable availability"}

# Costing parameters
discount_rate = 0

# Extract consumables dispensed data
def drop_outside_period(_df):
    """Return a dataframe which only includes for which the date is within the limits defined by TARGET_PERIOD"""
    return _df.drop(index=_df.index[~_df['date'].between(*TARGET_PERIOD)])

def get_quantity_of_consumables_dispensed(results_folder):
    def convert_str_to_dict(row_entry):
        if isinstance(row_entry, str):
            row_entry_dict = literal_eval(row_entry)
        else:
            row_entry_dict = row_entry
        return row_entry_dict

    def get_counts_of_items_requested(_df):
        _df = drop_outside_period(_df)
        # Dict with date and Treatment_ID
        counts_used = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        counts_notavail = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        for _, row in _df.iterrows():
            date = row['date']
            treatment = row['TREATMENT_ID']

            # Used Items
            for item, num in convert_str_to_dict(row['Item_Used']).items():
                counts_used[date][treatment][item] += num

            # Not Available Items
            for item, num in convert_str_to_dict(row['Item_NotAvailable']).items():
                counts_notavail[date][treatment][item] += num

        # Convert nested dicts → MultiIndex Series

        # Used
        used_records = []
        for date, tdict in counts_used.items():
            for treatment, idict in tdict.items():
                for item, num in idict.items():
                    used_records.append((date, treatment, item, 'Used', num))

        used_df = pd.DataFrame(used_records,
                               columns=['date', 'TREATMENT_ID', 'item', 'status', 'value'])

        # Not Available
        na_records = []
        for date, tdict in counts_notavail.items():
            for treatment, idict in tdict.items():
                for item, num in idict.items():
                    na_records.append((date, treatment, item, 'NotAvailable', num))

        notavail_df = pd.DataFrame(na_records,
                                   columns=['date', 'TREATMENT_ID', 'item', 'status', 'value'])

        # Combine
        combined = pd.concat([used_df, notavail_df], axis=0)

        # Convert to series with MultiIndex
        combined_series = combined.set_index(['date', 'TREATMENT_ID', 'item', 'status'])['value']

        return combined_series

    # Extract results using your existing pipeline
    cons_req = extract_results(
        results_folder,
        module='tlo.methods.healthsystem',
        key='Consumables',
        custom_generate_series=get_counts_of_items_requested,
        do_scaling=True
    )

    # Only keep 'Used'
    cons_dispensed = cons_req.xs("Used", level='status')

    return cons_dispensed

idx = pd.IndexSlice

consumables_dispensed = get_quantity_of_consumables_dispensed(results_folder)

consumables_dispensed_summary = compute_summary_statistics(consumables_dispensed, central_measure = 'median')

consumables_dispensed_summary = consumables_dispensed_summary.reset_index()
consumables_dispensed_summary[idx['year']] = pd.to_datetime(
    consumables_dispensed_summary[idx['date']]).dt.year

# Add consumable name and unit cost
consumables_dict = \
    pd.read_csv(path_for_consumable_resourcefiles / 'ResourceFile_Consumables_Items_and_Packages.csv', low_memory=False,
                encoding="ISO-8859-1")[['Items', 'Item_Code']]
consumables_dict = dict(zip(consumables_dict['Item_Code'], consumables_dict['Items']))
unit_costs = load_unit_cost_assumptions(resourcefilepath)
consumables_costs_by_item_code = unit_costs["consumables"]
consumables_costs_by_item_code = dict(zip(consumables_costs_by_item_code['Item_Code'], consumables_costs_by_item_code['Price_per_unit']))
consumables_dispensed_summary[idx['item_name']] = consumables_dispensed_summary[idx['item']].map(consumables_dict)
consumables_dispensed_summary[idx['unit_cost']] = consumables_dispensed_summary[idx['item']].map(consumables_costs_by_item_code)

consumables_dispensed_summary.to_csv(figurespath / 'sample_output_v1.csv')

# Extract supplementary data
# 1. Count of HSIs
def get_hsi_summary(results_folder, key, var, do_scaling = True):
    def flatten_nested_dict(d, parent_key=()):
        items = {}
        for k, v in d.items():
            new_key = parent_key + (k,)
            if isinstance(v, dict):
                items.update(flatten_nested_dict(v, new_key))
            else:
                items[new_key] = v
        return items

    def get_counts_of_hsi_events(_df: pd.Series):
        """Summarise the parsed logged-key results for one draw (as dataframe) into a pd.Series."""
        _df = _df.set_axis(_df['date'].dt.year).drop(columns=['date'])
        flat_series = _df[(var)].apply(flatten_nested_dict)

        return flat_series.apply(pd.Series).stack().stack()


    count = compute_summary_statistics(extract_results(
        Path(results_folder),
        module='tlo.methods.healthsystem.summary',
        key=key,
        custom_generate_series=get_counts_of_hsi_events,
        do_scaling=do_scaling,
    ), central_measure='mean')

    return count

count_by_treatment_id = get_hsi_summary(results_folder, key = 'HSI_Event_non_blank_appt_footprint',
                                        var = "TREATMENT_ID", do_scaling = True)
count_by_treatment_id = count_by_treatment_id.droplevel(2)
count_by_appointment = get_hsi_summary(results_folder, key = 'HSI_Event_non_blank_appt_footprint',
                                        var = "Number_By_Appt_Type_Code_And_Level", do_scaling = True)

count_by_treatment_id.to_csv(figurespath / 'sample_hsi_count_by_treatment.csv')
count_by_appointment.to_csv(figurespath / 'sample_hsi_count_by_appointment.csv')

# Disease specific information
# TODO update code to extract relevant results alongside prevalance
log['tlo.methods.tb'].keys()
# tb_incidence - num_new_active_tb, prop_active_tb_in_plhiv; tb_prevalence -> tbPrevActive, tbPrevActiveAdult, tbPrevActiveChild; tb_mdr - tbPropActiveCasesMdr; tb_treatment - tbPropDiagnosed, tbTreatmentCoverage, tbIptCoverage
log['tlo.methods.malaria'].keys()
# prevalence; tx_coverage - 'number_diagnosed', 'number_treated', 'proportion_diagnosed', 'treatment_coverage'
log['tlo.methods.epi'].keys()
# ep_vaccine_coverage - 'epBcgCoverage', 'epDtp3Coverage', 'epHep3Coverage',
#        'epHib3Coverage', 'epHpvCoverage', 'epMeasles2Coverage',
#        'epMeaslesCoverage', 'epNumInfantsUnder1', 'epOpv3Coverage',
#        'epPneumo3Coverage', 'epRota2Coverage', 'epRubellaCoverage'
log['tlo.methods.cardio_metabolic_disorders'].keys()
# diabetes_prevalence, 'diabetes_diagnosis_prevalence', 'diabetes_medication_prevalence'
#  'hypertension_prevalence', 'hypertension_diagnosis_prevalence', 'hypertension_medication_prevalence'
# 'chronic_kidney_disease_prevalence', 'chronic_kidney_disease_diagnosis_prevalence'
# 'chronic_ischemic_hd_prevalence', 'chronic_ischemic_hd_diagnosis_prevalence', 'chronic_ischemic_hd_medication_prevalence'
# 'ever_stroke_prevalence'
# 'ever_heart_attack_prevalence'
log['tlo.methods.wasting'].keys()
# 'wasting_prevalence_props' - 'total_mod_under5_prop', 'total_sev_under5_prop'

#TODO add bednets and IRS

