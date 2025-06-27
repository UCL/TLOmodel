from pathlib import Path

from collections import Counter, defaultdict

import os
import scipy.stats as st
from scipy.stats import t, norm, shapiro

import pandas as pd
import tableone
from tableone import TableOne

import matplotlib.pyplot as plt
import numpy as np

from typing import Callable, Dict, Iterable, List, Literal, Optional, TextIO, Tuple, Union

from tlo import Date
from tlo.analysis.utils import (bin_hsi_event_details, extract_results, get_scenario_outputs, compute_summary_statistics,
make_age_grp_types, get_scenario_info, make_calendar_period_lookup, make_calendar_period_type, parse_log_file)

outputspath = './outputs/sejjj49@ucl.ac.uk/'

scenario = 'integration_scenario_max_test_2462999'
results_folder= get_scenario_outputs(scenario, outputspath)[-1]
# create_pickles_locally(results_folder, compressed_file_name_prefix='service_integration_scenario')


# int_names = ['status_quo',
#              'chronic_care_clinic',
#              'screening_htn',
#              'screening_dm',
#              'screening_hiv',
#              'screening_tb',
#              'screening_fp',
#              'screening_mal',
#              'screening_all',
#              'mch_clinic_pnc',
#              'mch_clinic_fp',
#              'mch_clinic_all',
#              'all_integration']

int_names = ['status_quo',
             'htn',
            'htn_max',
            'dm',
            'dm_max',
            'hiv',
            'hiv_max',
            'tb',
            'tb_max',
            'mal',
            'mal_max',
            'fp_scr',
            'fp_scr_max',
            'pnc',
            'pnc_max',
            'fp_pn',
            'fp_pn_max',
            'chronic_care',
            'chronic_care_max',
            'all_screening',
            'all_screening_max',
             'all_mch',
             'all_mch_max',
            'all_int',
            'all_int_max']

# Create a folder to store graphs (if it hasn't already been created when ran previously)
g_path = f'{outputspath}graphs_{scenario}'

info = get_scenario_info(results_folder)
draws = [x for x in range(info['number_of_draws'])]

if not os.path.isdir(g_path):
        os.makedirs(f'{outputspath}graphs_{scenario}')


TARGET_PERIOD = (Date(2011, 1, 1), Date(2015, 12, 31))

def get_num_dalys(_df):
    """Return total number of DALYS (Stacked) by label (total within the TARGET_PERIOD).
    Throw error if not a record for every year in the TARGET PERIOD (to guard against inadvertently using
    results from runs that crashed mid-way through the simulation.
    """
    # TO DO: this isnt outputting all dalys (missing 2013 onwards)
    years_needed = [i.year for i in TARGET_PERIOD]
    assert set(_df.year.unique()).issuperset(years_needed), "Some years are not recorded."
    return pd.Series(
        data=_df
        .loc[_df.year.between(*years_needed)]
        .drop(columns=['date', 'sex', 'age_range', 'year'])
        .sum().sum()
    )

num_dalys = extract_results(
        results_folder,
        module='tlo.methods.healthburden',
        key='dalys_stacked',
        custom_generate_series=get_num_dalys,
        do_scaling=False
    )

idx = pd.IndexSlice
total_dalys_dfs = {k: num_dalys.loc[:, idx[d, :]] for k, d in zip (int_names, draws)}

def get_diff_multi_index(df, int_name, draw):
    diff = df[int_name][draw] - df['status_quo'][0]
    diff.columns=df[int_name].columns
    return diff

total_dalys_diff_dfs = {k: get_diff_multi_index(total_dalys_dfs, k, d) for k, d in zip(int_names, draws)}

total_dalys_summ = {k:compute_summary_statistics(total_dalys_dfs[k]) for k in int_names}
total_dalys_diff_summ = {k:compute_summary_statistics(total_dalys_diff_dfs[k]) for k in int_names}

all_dalys_dfs = extract_results(
            results_folder,
            module="tlo.methods.healthburden",
            key="dalys_stacked",
            custom_generate_series=(
                lambda df: df.drop(
                    columns=['date', 'sex', 'age_range']).groupby(['year']).sum().stack()),
            do_scaling=False)
all_dalys_dfs.index.names = ['year', 'cause']
years_to_sum = list(range(2011, 2016))

# Filter the DataFrame to include only those years
df_subset = all_dalys_dfs.loc[all_dalys_dfs.index.get_level_values('year').isin(years_to_sum)]

# Group by 'cause' and sum
cause_totals = df_subset.groupby('cause').sum()
total_cause_dfs = {k: cause_totals.loc[:, idx[d, :]] for k, d in zip (int_names, draws)}
total_cause_summ = {k:compute_summary_statistics(total_cause_dfs[k]) for k in int_names}

total_cause_diff_dfs = {k: get_diff_multi_index(total_cause_dfs, k, d) for k, d in zip(int_names, draws)}
total_cause_summ_diff = {k:compute_summary_statistics(total_cause_diff_dfs[k]) for k in int_names}


# GRAPHS AND CSV FILES

for k in total_cause_diff_dfs:
    total_cause_diff_dfs[k].to_csv(f'{g_path}/{k}_diffs.csv')

for k, d in zip(total_cause_diff_dfs, draws):
    labels = total_cause_summ_diff[k].index
    median = total_cause_summ_diff[k][d]['central'].values
    lower_errors = total_cause_summ_diff[k][d]['lower'].values
    upper_errors = total_cause_summ_diff[k][d]['upper'].values

    # lower_errors = [data[k].loc[0, 'lower'] for k in labels]
    # upper_errors = [data[k].loc[0, 'upper'] for k in labels]

    # lower_errors = [data[k][d].loc[0, 'lower'] - data[k][d].loc[0, 'central']for k, d in zip(labels, draws)]
    # upper_errors = [data[k][d].loc[0, 'upper'] - data[k][d].loc[0, 'lower'] for k, d in zip(labels, draws)]
    # errors = [lower_errors, upper_errors]

    # Compute distances from mean to bounds (must be non-negative)
    yerr_lower = [mean - low for mean, low in zip(median, lower_errors)]
    yerr_upper = [up - mean for mean, up in zip(median, upper_errors)]

    # Create bar chart with error bars
    fig, ax = plt.subplots()
    ax.bar(labels, median, yerr=[yerr_lower, yerr_upper], capsize=5, alpha=0.7, ecolor='black')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_ylabel('Difference in DALYs from SQ')
    ax.set_title(f'{k} Vs status_quo: Difference in DALYs by cause')

    # Adjust label size
    plt.xticks(fontsize=8, rotation=90)
    plt.tight_layout()
    plt.savefig(f'{g_path}/{k}_diff_dalys_cause.png', bbox_inches='tight')
    plt.show()


def barcharts(data, y_label, title):

    # Extract means and errors
    labels = data.keys()
    median = [data[k][d].loc[0, 'central'] for k, d in zip(labels, draws)]
    lower_errors = [data[k][d].loc[0, 'lower'] for k, d in zip(labels, draws)]
    upper_errors = [data[k][d].loc[0, 'upper'] for k, d in zip(labels, draws)]

    # lower_errors = [data[k].loc[0, 'lower'] for k in labels]
    # upper_errors = [data[k].loc[0, 'upper'] for k in labels]

    # lower_errors = [data[k][d].loc[0, 'lower'] - data[k][d].loc[0, 'central']for k, d in zip(labels, draws)]
    # upper_errors = [data[k][d].loc[0, 'upper'] - data[k][d].loc[0, 'lower'] for k, d in zip(labels, draws)]
    # errors = [lower_errors, upper_errors]

    # Compute distances from mean to bounds (must be non-negative)
    yerr_lower = [mean - low for mean, low in zip(median, lower_errors)]
    yerr_upper = [up - mean for mean, up in zip(median, upper_errors)]

    # Create bar chart with error bars
    fig, ax = plt.subplots()
    ax.bar(labels, median, yerr=[yerr_lower, yerr_upper], capsize=5, alpha=0.7, ecolor='black')
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Adjust label size
    plt.xticks(fontsize=8, rotation=90)
    plt.tight_layout()
    plt.savefig(f'{g_path}/{title}.png', bbox_inches='tight')
    plt.show()

barcharts(total_dalys_diff_summ, 'Difference in DALYs', 'Total Difference in Total DALYs from Status Quo by '
                                                   'Scenario')

barcharts(total_dalys_summ, 'DALYs', ' Total DALYs from Status Quo by Scenario')


keys = list(total_cause_summ.keys())
baseline_key = keys[0]
baseline_df = total_cause_summ[baseline_key]

categories = baseline_df.index
x = np.arange(len(categories))
width = 0.35  # width of each bar

for key, draw in zip(keys[1:], draws[1:]):
    comp_df = total_cause_summ[key]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract data and compute asymmetric error bars
    # Baseline
    baseline_central = baseline_df[0]['central']
    baseline_err_lower = baseline_central - baseline_df[0]['lower']
    baseline_err_upper = baseline_df[0]['upper'] - baseline_central

    # Comparison
    comp_central = comp_df[draw]['central']
    comp_err_lower = comp_central - comp_df[draw]['lower']
    comp_err_upper = comp_df[draw]['upper'] - comp_central

    # Plot bars with asymmetric error bars
    ax.bar(x - width/2, baseline_central, width,
           yerr=[baseline_err_lower, baseline_err_upper],
           capsize=5, label=baseline_key, alpha=0.8)

    ax.bar(x + width/2, comp_central, width,
           yerr=[comp_err_lower, comp_err_upper],
           capsize=5, label=key, alpha=0.8)

    ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    ax.set_title(f"Comparison: {key} vs {baseline_key}")
    ax.set_ylabel("DALYs")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f'{g_path}/{key}_dalys_cause.png', bbox_inches='tight')
    plt.show()




def get_dalys_by_period_sex_agegrp_label(df):
    """Sum the dalys by period, sex, age-group and label"""
    df['age_grp'] = df['age_range'].astype(make_age_grp_types())
    df = df.drop(columns=['date', 'age_range', 'sex'])
    df = df.groupby(by=["year", "age_grp"]).sum().stack()
    df.index = df.index.set_names('label', level=2)
    return df

dalys = extract_results(
                results_folder,
                module="tlo.methods.healthburden",
                key="dalys_stacked_by_age_and_time",  # <-- for DALYS stacked by age and time
                custom_generate_series=get_dalys_by_period_sex_agegrp_label,
                do_scaling=False
            )
dalys.index = dalys.index.set_names('age_group', level=1)

def get_pop_by_agegrp_label(df):
    """Sum the dalys by period, sex, age-group and label"""
    df['year'] = df['date'].dt.year
    df_melted = df.melt(id_vars=['year'], value_vars=[col for col in df.columns if col not in ['date', 'year']],
                        var_name='age_group', value_name='count')
    series_multi = df_melted.set_index(['year', 'age_group'])['count'].sort_index()

    return series_multi


pop_f = extract_results(
                results_folder,
                module="tlo.methods.demography",
                key="age_range_f",  # <-- for DALYS stacked by age and time
                custom_generate_series=get_pop_by_agegrp_label,
                do_scaling=False
            )

pop_m = extract_results(
                results_folder,
                module="tlo.methods.demography",
                key="age_range_m",  # <-- for DALYS stacked by age and time
                custom_generate_series=get_pop_by_agegrp_label,
                do_scaling=False
            )

pop = pop_f + pop_m

pop_summ = compute_summary_statistics(pop)
dalys_summ = compute_summary_statistics(dalys)


# TODO OTHER OUTPUTS
# Notes from epi meeting with Tim C/Andrew
# - consider splitting appointments by those which could be done by staff member with sufficient training (e.g. refills) and
# those that could only be done by a specialist (e.g. initial HIV care)
# - start modelling at 2025
# - present % of total DALYs attributable to each scenario
# - can we look at TB dalys in people with HIV
# - for contraception, present met need instead of DALYs? (although could present maternal dalys only)
# they were unsure about age standardizaton

# =============================================== CONSUMABLES =========================================================
def drop_outside_period(_df):
    """Return a dataframe which only includes for which the date is within the limits defined by TARGET_PERIOD"""
    return _df.drop(index=_df.index[~_df['date'].between(*TARGET_PERIOD)])

def get_quantity_of_consumables_dispensed(results_folder):
    def get_counts_of_items_requested(_df):
        _df = drop_outside_period(_df)
        counts_of_used = defaultdict(lambda: defaultdict(int))
        counts_of_not_available = defaultdict(lambda: defaultdict(int))

        for _, row in _df.iterrows():
            date = row['date']
            for item, num in row['Item_Used'].items():
                counts_of_used[date][item] += num
            for item, num in row['Item_NotAvailable'].items():
                counts_of_not_available[date][item] += num
        used_df = pd.DataFrame(counts_of_used).fillna(0).astype(int).stack().rename('Used')
        not_available_df = pd.DataFrame(counts_of_not_available).fillna(0).astype(int).stack().rename('Not_Available')

        # Combine the two dataframes into one series with MultiIndex (date, item, availability_status)
        combined_df = pd.concat([used_df, not_available_df], axis=1).fillna(0).astype(int)

        # Convert to a pd.Series, as expected by the custom_generate_series function
        return combined_df.stack()

    cons_req = extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='Consumables',
        custom_generate_series=get_counts_of_items_requested,
        do_scaling=False)

    cons_dispensed = cons_req.xs("Used", level=2)  # only keep actual dispensed amount, i.e. when available
    return cons_dispensed

consumables_dispensed = get_quantity_of_consumables_dispensed(results_folder)
consumables_dispensed = consumables_dispensed.groupby(level=0).sum()

base = consumables_dispensed.loc[:, 0]

# Calculate percentage difference
percent_diff = consumables_dispensed.copy()
for col in consumables_dispensed.columns:
    if col[0] != 0:
        # Get corresponding (0, col[1]) for comparison
        base_col = (0, col[1])
        percent_diff[col] = (consumables_dispensed[col] - consumables_dispensed[base_col]) / consumables_dispensed[base_col] * 100
    else:
        percent_diff[col] = 0  # or np.nan if you prefer

pdiff_sum = compute_summary_statistics(percent_diff)


# todo: what about converting this back to a normal dose/size/measure?

ic = {'htn': {'Hydralazine (oral)': 221},

      'dm': {'Metformin (oral)': 233,
             'Blood glucose test': 216,},

      'hiv': {'HIV test': 196,
              'First-line ART regimen (adult)': 2671,
              'Cotrimoxizole, 960mg': 204,
              'First line ART regimen (older child)': 2672,
              'Cotrimoxazole 120mg' : 203,
              'First line ART regimen (young child)': 2673},


      'tb': {'ZN Stain': 186,
             'Xpert': 187,
             'X-ray': 175,
             'MGIT960 Culture and DST': 188,
             'Cat. I & III Patient Kit A': 176,
             'Cat. I & III Patient Kit B': 178,
             'Cat. II Patient Kit A1': 177,
             'Cat. II Patient Kit A2': 179,
             'Treatment (second-line drugs)': 181,
             'Isoniazid/Pyridoxine, tablet 300 mg': 192,
             'Isoniazid/Rifapentine': 2678},

      'mal': {'Supplementary spread, sachet': 1221,
              'Complementary feeding': 1171},

      'fp': {'Levonorgestrel': 1,
             'Condom, male': 2,
             'IUD, Copper': 7,
             'Depot': 3,
             'Jadelle (implant)': 12,
             'Pregnancy test kit': 2019},

      'pnc': {'Hydralazine' : 60,
              'Methyldopa' : 222,
              'Magnesium sulfate' : 61,
              'Benzylpenicillin' : 99,
              'Gentamycin' : 28,
              'Oxytocin, injection' : 56,
              'Blood, one unit' : 141,
              'Haemoglobin test (HB)' : 50,
              'Ferrous Salt + Folic Acid' : 140},

      'chronic_care': {'Phenobarbital': 278,
                        'Carbamazepine': 276,
                        'Phenytoin sodium': 279,
                        'Amitriptyline': 267}}

draw_numbs = {'htn': [1, 2],
              'dm': [3, 4],
              }


def get_data_as_list_for_bc(draw_numbs):
    nc = draw_numbs[0]
    mc = draw_numbs[1]

    def get_med_and_error(draw):
        med = [pdiff_sum.at[f'{ic}', (draw, 'central')] for ic in item_codes]
        lq = [pdiff_sum.at[f'{ic}', (draw, 'lower')] for ic in item_codes]
        uq = [pdiff_sum.at[f'{ic}', (draw, 'upper')] for ic in item_codes]
        int_err_lower = [a - b for a, b in zip(med, lq)]
        int_err_upper = [a - b for a, b in zip(uq, med)]

        return [med, int_err_lower, int_err_upper]

    return [get_med_and_error(nc), get_med_and_error(mc)]

filtered_int_name = [s for s in int_names if not s.endswith('_max')]
filtered_int_name.remove('status_quo')

for scen in filtered_int_name:
    if scen.startswith('htn'):
        item_codes = list(ic['htn'].values())
        labels = list(ic['htn'].keys())
        data = get_data_as_list_for_bc(draw_numbs['htn'])
        title = scen

    elif scen.startswith('dm'):
        item_codes = list(ic['dm'].values())
        labels = list(ic['dm'].keys())
        data = get_data_as_list_for_bc(draw_numbs['dm'])
        title = scen

    else:
        pass

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, data[0][0], width,
           yerr=[data[0][1], data[0][2]],
           capsize=5, label='Normal Cons.', alpha=0.8)

    ax.bar(x + width / 2, data[1][0], width,
           yerr=[data[1][1], data[1][2]],
           capsize=5, label='Max Con.s', alpha=0.8)

    ax.set_title(f"Percentage Difference in Consumable Use from SQ - {title}")
    ax.set_ylabel("Percentage Difference")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{g_path}/{scenario}_cons_pdiff.png', bbox_inches='tight')
    plt.show()



# ========================================= APPOINTMENTS/HCW TIME =====================================================
# # NUMBER OF APPOINTMENTS

def compute_service_statistics(counters_by_draw_and_run):
    grouped_data = defaultdict(lambda: defaultdict(list))

    # Step 1: Group counts by first key and service name
    for (group_idx, _), counter in counters_by_draw_and_run.items():
        for service_name, count in counter.items():
            grouped_data[group_idx][service_name].append(count)

    # Step 2: Compute statistics
    result = defaultdict(dict)
    width_of_range = 0.95
    lower_quantile = (1. - width_of_range) / 2.
    for group_idx, service_dict in grouped_data.items():
        for service_name, counts in service_dict.items():
            arr = np.array(counts)
            result[group_idx][service_name] = {
                "median": float(np.median(arr)),
                "lower_quartile": float(np.quantile(arr, lower_quantile)),
                "upper_quartile": float(np.quantile(arr, 1 - lower_quantile))
            }

    return result

counts_by_treatment_id = bin_hsi_event_details(
            results_folder,
            lambda event_details, count: sum(
                [
                    Counter({
                        (
                            event_details["treatment_id"]
                        ):
                        count * appt_number
                    })
                    for appt_type, appt_number in event_details["appt_footprint"]
                ],
                Counter()
            ),
            *TARGET_PERIOD,
            True
        )

# TODO - what about other HSIs that might be impacted (anc, pnc etc), should we do this more generally

hsi_results = compute_service_statistics(counts_by_treatment_id)
hsi_results = {k:v for k, v in zip(int_names, hsi_results.values())}

hsi_by_scen = {
             'htn':['CardioMetabolicDisorders_Prevention_CommunityTestingForHypertension',
                                        'CardioMetabolicDisorders_Investigation',
                                        'CardioMetabolicDisorders_Prevention_WeightLoss'],

            'htn_max':['CardioMetabolicDisorders_Prevention_CommunityTestingForHypertension',
                                        'CardioMetabolicDisorders_Investigation',
                                        'CardioMetabolicDisorders_Prevention_WeightLoss'],

            'dm':[  'CardioMetabolicDisorders_Investigation',
                                        'CardioMetabolicDisorders_Prevention_WeightLoss'],

            'dm_max':['CardioMetabolicDisorders_Investigation',
                      'CardioMetabolicDisorders_Prevention_WeightLoss'],

            'hiv': ['Hiv_Test', 'Hiv_Treatment'],

            'hiv_max': ['Hiv_Test', 'Hiv_Treatment'],

            'tb': ['Tb_Test_Screening',
                    'Tb_Test_Clinical',
                    'Tb_Test_Xray',
                    'Tb_Treatment'],

            'tb_max': ['Tb_Test_Screening',
                       'Tb_Test_Clinical',
                        'Tb_Test_Xray',
                        'Tb_Treatment'],

            'mal': ['Undernutrition_Feeding'],

            'mal_max': ['Undernutrition_Feeding'],

            'fp_scr': ['Contraception_Routine'],

            'fp_scr_max': ['Contraception_Routine'],

            'pnc': ['PostnatalCare_Neonatal',
                      'PostnatalCare_Maternal'],

            'pnc_max': ['PostnatalCare_Neonatal',
                      'PostnatalCare_Maternal'],

            'fp_pn': ['Contraception_Routine'],

            'fp_pn_max': ['Contraception_Routine'],

            'chronic_care': ['CardioMetabolicDisorders_Investigation',
                                        'CardioMetabolicDisorders_Prevention_WeightLoss',
                                        'Hiv_Test',
                                        'Hiv_Treatment',
                                        'Tb_Test_Screening',
                                        'Tb_Test_Clinical',
                                        'Tb_Test_Xray',
                                        'Tb_Treatment',
                                        'Depression_TalkingTherapy',
                                        'Depression_Treatment',
                                        'Epilepsy_Treatment_Start',
                                        'Epilepsy_Treatment_Followup'],

            'chronic_care_max': ['CardioMetabolicDisorders_Investigation',
                                        'CardioMetabolicDisorders_Prevention_WeightLoss',
                                        'Hiv_Test',
                                        'Hiv_Treatment',
                                        'Tb_Test_Screening',
                                        'Tb_Test_Clinical',
                                        'Tb_Test_Xray',
                                        'Tb_Treatment',
                                        'Depression_TalkingTherapy',
                                        'Depression_Treatment',
                                        'Epilepsy_Treatment_Start',
                                        'Epilepsy_Treatment_Followup'],

            'all_screening': ['CardioMetabolicDisorders_Prevention_CommunityTestingForHypertension',
                                        'CardioMetabolicDisorders_Investigation',
                                        'CardioMetabolicDisorders_Prevention_WeightLoss',
                                        'Contraception_Routine',
                                        'Undernutrition_Feeding',
                                        'Hiv_Test',
                                        'Hiv_Treatment',
                                        'Tb_Test_Screening',
                                        'Tb_Test_Clinical',
                                        'Tb_Test_Xray',
                                        'Tb_Treatment'],

            'all_screening_max': ['CardioMetabolicDisorders_Prevention_CommunityTestingForHypertension',
                                        'CardioMetabolicDisorders_Investigation',
                                        'CardioMetabolicDisorders_Prevention_WeightLoss',
                                        'Contraception_Routine',
                                        'Undernutrition_Feeding',
                                        'Hiv_Test',
                                        'Hiv_Treatment',
                                        'Tb_Test_Screening',
                                        'Tb_Test_Clinical',
                                        'Tb_Test_Xray',
                                        'Tb_Treatment'],

            'all_mch': ['Undernutrition_Feeding',
                                        'PostnatalCare_Neonatal',
                                        'PostnatalCare_Maternal',
                                        'Contraception_Routine'],

             'all_mch_max': ['Undernutrition_Feeding',
                                        'PostnatalCare_Neonatal',
                                        'PostnatalCare_Maternal',
                                        'Contraception_Routine'],

            'all_int': ['CardioMetabolicDisorders_Prevention_CommunityTestingForHypertension',
                                        'CardioMetabolicDisorders_Investigation',
                                        'CardioMetabolicDisorders_Prevention_WeightLoss',
                                        'Contraception_Routine',
                                        'Undernutrition_Feeding',
                                        'Hiv_Test',
                                        'Hiv_Treatment',
                                        'Tb_Test_Screening',
                                        'Tb_Test_Clinical',
                                        'Tb_Test_Xray',
                                        'Tb_Treatment',
                                        'PostnatalCare_Neonatal',
                                        'PostnatalCare_Maternal',
                                        'Depression_TalkingTherapy',
                                        'Depression_Treatment',
                                        'Epilepsy_Treatment_Start',
                                        'Epilepsy_Treatment_Followup'],

            'all_int_max': ['CardioMetabolicDisorders_Prevention_CommunityTestingForHypertension',
                                        'CardioMetabolicDisorders_Investigation',
                                        'CardioMetabolicDisorders_Prevention_WeightLoss',
                                        'Contraception_Routine',
                                        'Undernutrition_Feeding',
                                        'Hiv_Test',
                                        'Hiv_Treatment',
                                        'Tb_Test_Screening',
                                        'Tb_Test_Clinical',
                                        'Tb_Test_Xray',
                                        'Tb_Treatment',
                                        'PostnatalCare_Neonatal',
                                        'PostnatalCare_Maternal',
                                        'Depression_TalkingTherapy',
                                        'Depression_Treatment',
                                        'Epilepsy_Treatment_Start',
                                        'Epilepsy_Treatment_Followup']}


def get_sq_hsi_data(data, labels):
    med = [data[v]['median'] if v in data else 0 for v in labels]
    lq = [data[v]['lower_quartile'] if v in data else 0 for v in labels]
    uq = [data[v]['upper_quartile'] if v in data else 0 for v in labels]

    sq_err_lower = [a - b for a, b in zip(med, lq)]
    sq_err_upper = [a - b for a, b in zip(uq, med)]

    return [med, sq_err_lower, sq_err_upper]

for scenario in int_names:
    if scenario == 'status_quo':
        pass
    else:
        labels = hsi_by_scen[scenario]
        int_data = hsi_results[scenario]
        baseline_data = get_sq_hsi_data(hsi_results['status_quo'], labels)

        median = [int_data[v]['median'] if v in int_data else 0 for v in labels]
        lq = [int_data[v]['lower_quartile'] if v in int_data else 0 for v in labels]
        uq = [int_data[v]['upper_quartile'] if v in int_data else 0 for v in labels]

        fig, ax = plt.subplots(figsize=(10, 6))

        int_err_lower = [a - b for a, b in zip(median, lq)]
        int_err_upper = [a - b for a, b in zip(uq, median)]

        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width / 2, baseline_data[0], width,
               yerr=[baseline_data[1], baseline_data[2]],
               capsize=5, label='Status Quo', alpha=0.8)

        ax.bar(x + width / 2, median, width,
               yerr=[int_err_lower, int_err_upper],
               capsize=5, label=scenario, alpha=0.8)

        # ax.axhline(0, color='gray', linestyle='--', linewidth=1)

        ax.set_title(f"Comparison: {scenario} vs Status Quo")
        ax.set_ylabel("Number of HSIs")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{g_path}/{scenario}_hsi_counts.png', bbox_inches='tight')
        plt.show()





# def get_mean_pop_by_age_for_sex_and_year(sex):
#     years_needed = [i.year for i in TARGET_PERIOD]
#
#     if sex == 'F':
#         key = "age_range_f"
#     else:
#         key = "age_range_m"
#
#     num_by_age = compute_summary_statistics(
#         extract_results(results_folder,
#                         module="tlo.methods.demography",
#                         key=key,
#                         custom_generate_series=(
#
#                             lambda df_: df_.drop(
#                                 columns=['date']
#                             ).melt(
#                                 var_name='age_grp'
#                             ).set_index('age_grp')['value']
#                         ),
#                         do_scaling=False
#                         ),
#         collapse_columns=True,
#     )
#     print(num_by_age.index[num_by_age.index.duplicated()])
#     # num_by_age = num_by_age.reindex(make_age_grp_types().categories)
#     return num_by_age
#
# model_m = get_mean_pop_by_age_for_sex_and_year('M')
# model_f = get_mean_pop_by_age_for_sex_and_year('F')
