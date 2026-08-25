"""
Script to calculate and export cost per scenario as CSV.
Stripped-down version of dcea_analysis_cost_vertical_programs_with_and_without_hss.py
"""
from pathlib import Path
import pandas as pd


from scripts.costing.cost_estimation import estimate_input_cost_of_scenarios, summarize_cost_data
from tlo import Date
from tlo.analysis.utils import (
    extract_params,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
)
# Function to get incremental values
def find_difference_relative_to_comparison(_ser: pd.Series,
                                           comparison: str,
                                           scaled: bool = False,
                                           drop_comparison: bool = True,
                                           ):
    """Find the difference in the values in a pd.Series with a multi-index, between the draws (level 0)
    within the runs (level 1), relative to where draw = `comparison`.
    The comparison is `X - COMPARISON`."""
    return _ser \
        .unstack(level=0) \
        .apply(lambda x: (x - x[comparison]) / (x[comparison] if scaled else 1.0), axis=1) \
        .drop(columns=([comparison] if drop_comparison else [])) \
        .stack()

# Load result files
# results_folder = get_scenario_outputs('htm_with_and_without_hss-2026-08-07T092059Z.py',
#                                       Path('./outputs/n.fuller@ic.ac.uk'))[0]
outputfilepath = Path('./outputs/t.mangal@imperial.ac.uk')
results_folder = get_scenario_outputs('htm_and_hss_runs-2025-10-14T084418Z.py', outputfilepath)[0]

# Check can read results from draw=0, run=0
log = load_pickled_dataframes(results_folder, 0, 0)
params = extract_params(results_folder)
info = get_scenario_info(results_folder)

# Period relevant for costing
TARGET_PERIOD = (Date(2025, 1, 1), Date(2035, 12, 31))  # This is the period that is costed

relevant_period_for_costing = [i.year for i in TARGET_PERIOD]
list_of_relevant_years_for_costing = list(range(relevant_period_for_costing[0], relevant_period_for_costing[1] + 1))

# Scenarios
# all_manuscript_scenarios = {0: "Baseline",
#                             1: "FULL.HSS.PACKAGE",
#                             2: "FULL.HSS.PACKAGE.Norm.HCS",
#                             3: "HIV.Programs.Scale.up.WITHOUT.HSS.PACKAGE",
#                             4: "HIV.Programs.Scale.up.WITH.HSS.PACKAGE",
#                             5: "HIV.Programs.Scale.up.WITH.HSS.PACKAGE.Norm.HCS",
#                             6: "TB.Programs.Scale.up.WITHOUT.HSS.PACKAGE",
#                             7: "TB.Programs.Scale.up.WITH.HSS.PACKAGE",
#                             8: "TB.Programs.Scale.up.WITH.HSS.PACKAGE.Norm.HCS",
#                             9: "Malaria.Programs.Scale.up.WITHOUT.HSS.PACKAGE",
#                             10: "Malaria.Programs.Scale.up.WITH.HSS.PACKAGE",
#                             11: "Malaria.Programs.Scale.up.WITH.HSS.PACKAGE.Norm.HCS",
#                             12: "HIV.Tb.Malaria.Programs.Scale.up.WITHOUT.HSS.PACKAGE",
#                             13: "HIV.Tb.Malaria.Programs.Scale.up.WITH.HSS.PACKAGE",
#                             14: "HIV.Tb.Malaria.Programs.Scale.up.WITH.HSS.PACKAGE.Norm.HCS"}

# Scenarios
# Full list of scenarios used in the manuscript
main_manuscript_scenarios = {0: "Baseline",
                            1: "Pessimistic HRH Scale-up", 2: "Historical HRH Scale-up",
                            3: "Optimistic HRH Scale-up",
                            4: "Consumables Increased to 75th Percentile",
                            5: "Consumables Increased to HIV levels", 6: "Consumables Increased to EPI Levels",
                            7: "HSS Expansion Package",
                            8: "HIV Program Scale-up Without HSS Expansion",
                            15: "HIV Program Scale-up With HSS Expansion Package",
                            16: "TB Program Scale-up Without HSS Expansion",
                            23: "TB Program Scale-up With HSS Expansion Package",
                            24: "Malaria Program Scale-up Without HSS Expansion",
                            31: "Malaria Program Scale-up With HSS Expansion Package",
                            32: "HTM Programs Scale-up Without HSS Expansion",
                            39: "HTM Programs Scale-up With HSS Expansion Package"}

frontier_scenarios = {9:"HIV Program Scale-up With Pessimistic HRH Scale-up",
                      10:"HIV Program Scale-up With Historical HRH Scale-up",
                      11:"HIV Program Scale-up With Optimistic HRH Scale-up",
                      12:"HIV Program Scale-up With Consumables Increased to 75th Percentile",
                      13:"HIV Program Scale-up With Consumables Increased to HIV levels",
                      14:"HIV Program Scale-up With Consumables Increased to EPI Levels",
                      17:"TB Program Scale-up With Pessimistic HRH Scale-up",
                      18:"TB Program Scale-up With Historical HRH Scale-up",
                      19:"TB Program Scale-up With Optimistic HRH Scale-up",
                      20:"TB Program Scale-up With Consumables Increased to 75th Percentile",
                      21:"TB Program Scale-up With Consumables Increased to HIV levels",
                      22:"TB Program Scale-up With Consumables Increased to EPI Levels",
                      25:"Malaria Program Scale-up With Pessimistic HRH Scale-up",
                      26:"Malaria Program Scale-up With Historical HRH Scale-up",
                      27:"Malaria Program Scale-up With Optimistic HRH Scale-up",
                      28:"Malaria Program Scale-up With Consumables Increased to 75th Percentile",
                      29:"Malaria Program Scale-up With Consumables Increased to HIV levels",
                      30:"Malaria Program Scale-up With Consumables Increased to EPI Levels",
                      33:"HTM Scale-up With Pessimistic HRH Scale-up",
                      34:"HTM Scale-up With Historical HRH Scale-up",
                      35:"HTM Scale-up With Optimistic HRH Scale-up",
                      36:"HTM Scale-up With Consumables Increased to 75th Percentile",
                      37:"HTM Scale-up With Consumables Increased to HIV levels",
                      38:"HTM Scale-up With Consumables Increased to EPI Levels",
                      40:"HIV + TB Scale-up Without HSS Expansion",
                      41:"HIV + TB Scale-up With Historical HRH Scale-up",
                      42:"HIV + TB Scale-up With Consumables Increased to 75th Percentile",
                      43:"HIV + TB Scale-up With HSS Expansion",
                      44:"HIV + Malaria Scale-up Without HSS Expansion",
                      45:"HIV + Malaria Scale-up With Historical HRH Scale-up",
                      46:"HIV + Malaria Scale-up With Consumables Increased to 75th Percentile",
                      47:"HIV + Malaria Scale-up With HSS Expansion",
                      48:"TB + Malaria Scale-up Without HSS Expansion",
                      49:"TB + Malaria Scale-up With Historical HRH Scale-up",
                      50:"TB + Malaria Scale-up With Consumables Increased to 75th Percentile",
                      51:"TB + Malaria Scale-up With HSS Expansion"}

all_manuscript_scenarios = {**main_manuscript_scenarios, **frontier_scenarios}

all_manuscript_scenarios_reverse = {v: k for k, v in all_manuscript_scenarios.items()}



# Discount rate for costs
discount_rate_cost = 0.03

# Estimate input costs per scenario
resourcefilepath = Path("./resources")
input_costs = estimate_input_cost_of_scenarios(
    results_folder,
    resourcefilepath,
    _years=list_of_relevant_years_for_costing,
    cost_only_used_staff=True,
    _discount_rate=discount_rate_cost,
    _metric='median',
    _draws=list(all_manuscript_scenarios.keys())
)

# Calculate total cost per scenario
total_input_cost = input_costs.groupby(['draw', 'run'])['cost'].sum()

incremental_scenario_cost = (pd.DataFrame(
    find_difference_relative_to_comparison(
        total_input_cost,
        comparison=0)  # sets the comparator to draw 0 which is the Actual scenario
).T.iloc[0].unstack()).T

incremental_scenario_cost_summarized = summarize_cost_data(incremental_scenario_cost, _metric='median')
incremental_scenario_cost_df = incremental_scenario_cost_summarized.reset_index()
incremental_scenario_cost_df['scenario'] = incremental_scenario_cost_df['draw'].map(all_manuscript_scenarios)

# Save to CSV
output_path = Path('./outputs/dcea_costs')
output_path.mkdir(parents=True, exist_ok=True)
incremental_scenario_cost_df.to_csv(output_path / 'incremental_cost_per_scenario.csv', index=False)

print(f"Cost per scenario saved to {output_path / 'incremental_cost_per_scenario.csv'}")
print(incremental_scenario_cost_df.head())

chosen_cet = 191.4304166  # This is based on the estimate from Lomas et al (2023)- $160.595987085533 in 2019 USD coverted to 2023 USD
# based on Ochalek et al (2018) - the paper provided the value $61 in 2016 USD terms, this value is $77.4 in 2023 USD terms
