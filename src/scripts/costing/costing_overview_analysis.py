"""Produce outputs for cost overview paper.
The draft version of the paper uses outputs from scenario_impact_of_healthsystem.py, used to model HSS scenarios for
FCDO and Global Fund.

with reduced consumables logging
/Users/tmangal/PycharmProjects/TLOmodel/outputs/t.mangal@imperial.ac.uk/hss_elements-2024-11-12T172311Z
"""

import datetime
import os
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts.costing.cost_estimation import (
    create_summary_treemap_by_cost_subgroup,
    do_line_plot_of_cost,
    do_stacked_bar_plot_of_cost_by_category,
    estimate_input_cost_of_scenarios,
    clean_consumable_name,
    clean_equipment_name,
)
from tlo import Date
from tlo.analysis.utils import (
    extract_params,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    compute_summary_statistics,
    extract_results
)

# Define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# Print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

# Create folders to store results
resourcefilepath = Path("./resources")
outputfilepath = Path('./outputs/sakshi.mohan@york.ac.uk')
figurespath = Path('./outputs/costing_dec25/overview/')
if not os.path.exists(figurespath):
    os.makedirs(figurespath)
path_for_consumable_resourcefiles = resourcefilepath / "healthsystem/consumables"

# Load result files
# ------------------------------------------------------------------------------------------------------------------
results_folder = get_scenario_outputs('full_system_costing-2025-12-15T162956Z.py', outputfilepath)[0] # Dec 2025 runs

# Check can read results from draw=0, run=0
log = load_pickled_dataframes(results_folder, 0, 0)  # look at one log (so can decide what to extract)
params = extract_params(results_folder)
info = get_scenario_info(results_folder)

# Declare default parameters for cost analysis
# ------------------------------------------------------------------------------------------------------------------
# Period relevant for costing
TARGET_PERIOD = (Date(2023, 1, 1), Date(2030, 12, 31))  # This is the period that is costed
relevant_period_for_costing = [i.year for i in TARGET_PERIOD]
list_of_relevant_years_for_costing = list(range(relevant_period_for_costing[0], relevant_period_for_costing[1] + 1))
list_of_years_for_plot = list(range(2023, 2031))
number_of_years_costed = relevant_period_for_costing[1] - 2023 + 1

# Scenarios
cost_scenarios = {0: "Actual", 1: "Expanded HRH", 2: "Improved consumable availability",
                  3: "Expanded HRH + Improved consumable availability"}

# Costing parameters
discount_rate = 0.03
discount_rate_lomas = {2023: 0.0036, 2024: 0.0040, 2025: 0.0039, 2026: 0.0042, 2027: 0.0042, 2028: 0.0041,
                       2029: 0.0041, 2030: 0.0040}# get the list of discount rates from 2023 until 2030

# Function to extract number of HSIs
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

# Estimate standard input costs of scenario
# -----------------------------------------------------------------------------------------------------------------------
# Standard 3% discount rate
input_costs = estimate_input_cost_of_scenarios(results_folder, resourcefilepath, _draws = list(cost_scenarios.keys()),
                                               _years=list_of_relevant_years_for_costing, cost_only_used_staff=True,
                                               _discount_rate = discount_rate, summarize = True)

# Undiscounted costs
input_costs_undiscounted = estimate_input_cost_of_scenarios(results_folder, resourcefilepath, _draws = list(cost_scenarios.keys()),
                                               _years=list_of_relevant_years_for_costing, cost_only_used_staff=True,
                                               _discount_rate = 0, summarize = True)

# Cost with variable discount rate based on Lomas et al (2021)
input_costs_variable_discounting = estimate_input_cost_of_scenarios(results_folder, resourcefilepath, _draws = list(cost_scenarios.keys()),
                                               _years=list_of_relevant_years_for_costing, cost_only_used_staff=True,
                                               _discount_rate = discount_rate_lomas, summarize = True)

def scale_consumable_cost(cost_df, item_name, scaling_factor):
    cost_df.loc[cost_df.cost_subgroup == item_name, 'cost'] = scaling_factor * cost_df.loc[cost_df.cost_subgroup == item_name, 'cost']
    return cost_df

def clean_consumables_cost_for_manuscript(df, drop_consumables):
    # 1. Temporary fix to the cost of F-75 Therapeutic milk
    # # The amount currently listed as 102.5 g should, in fact, be 102.5 ml. A sachet of 102.5 g is used to prepare 500 ml of
    # # F-75 milk. So it should have been: 102.5÷500×102.5 = 21 g per feed (every 3 hours for 3 days)
    df = scale_consumable_cost(
        cost_df=df,
        item_name='F-75 therapeutic milk, 102.5 g',
        scaling_factor=102.5 / 500
    )

    # 2. Clean consumable names
    df['cost_subgroup'] = (
        df['cost_subgroup']
        .astype(str)
        .apply(clean_consumable_name)
    )

    # 3. Drop non-consumables
    drop_consumables = drop_consumables

    df = df[
        ~df['cost_subgroup'].isin(drop_consumables)
    ]

    return df

def clean_equipment_cost_for_manuscript(df, drop_equipment):
    # 2. Clean consumable names
    df['cost_subgroup'] = (
        df['cost_subgroup']
        .astype(str)
        .apply(clean_equipment_name)
    )

    # 3. Drop non-medical equipment
    drop_mask = (
        df['cost_subgroup'].isin(drop_equipment)
        & (df['cost'] == 0)
    )

    df = df[~(drop_mask)]

    return df

drop_consumables = ['Endoscope',
        'Cystoscope',
        'Complementary feeding--education only drugs/supplies to service a client']

# Furniture, fittings, tools, containers
drop_equipment = [
    'Wheelbarrows',
    'Shovels',
    'Basin',
    'Bucket Stand',
    'Bucket without tap 20 LITRES',
    'Bench',
    'Chair',
    'Desk',
    'Table',
    'Shelves',
    'Cabinet, filing',
    'Cupboard, for medicine, lockable',
    'Linen',
    'Pillow cases',
    'Pillows of various shapes',
    'Mixing bowls',
    'Measuring Cup',
    'Measuring gauges',
    'Clock, wall type',
    'Mirror',
    'Pedal bin',
    'Waste Bin',
    'Otto Bins',
    'Water container, 20 litres with tap, on stand with soap',
    'Racks, storage, set',
    'Sample Rack',
    'Containers for Decontamination, Cleaning, Instruments and Linen',
]
# IT / software / office-adjacent
drop_equipment += [
    'Analytical software',
    'Medical software',
    'Office software',
    'Computer set',
    'Laptop',
    'Printer',
    'Router machine',
    'UPS (uninterruptible power supply)',
    'Receiver',
]
# Cleaning / garments
drop_equipment += [
    'Cleaning utensils, set',
    'Backsplit cotton gown',
]

input_costs = clean_consumables_cost_for_manuscript(input_costs, drop_consumables)
input_costs_undiscounted = clean_consumables_cost_for_manuscript(input_costs_undiscounted, drop_consumables)
input_costs_variable_discounting = clean_consumables_cost_for_manuscript(input_costs_variable_discounting, drop_consumables)
input_costs = clean_equipment_cost_for_manuscript(input_costs, drop_equipment)
input_costs_undiscounted = clean_equipment_cost_for_manuscript(input_costs_undiscounted, drop_equipment)
input_costs_variable_discounting = clean_equipment_cost_for_manuscript(input_costs_variable_discounting, drop_equipment)


# Get per capita estimates:
# Get population size for per capita estimates
def get_total_population_by_year(_df):
    years_needed = [i.year for i in TARGET_PERIOD]  # Malaria scale-up period years
    _df['year'] = pd.to_datetime(_df['date']).dt.year

    # Validate that all necessary years are in the DataFrame
    if not set(years_needed).issubset(_df['year'].unique()):
        raise ValueError("Some years are not recorded in the dataset.")

    # Filter for relevant years and return the total population as a Series
    return \
        _df.loc[_df['year'].between(min(years_needed), max(years_needed)), ['year', 'total']].set_index('year')[
            'total']


# Get total population by year
total_population_by_year = extract_results(
    results_folder,
    module='tlo.methods.demography',
    key='population',
    custom_generate_series=get_total_population_by_year,
    do_scaling=True
)
total_population_by_year = compute_summary_statistics(total_population_by_year, central_measure = 'mean')
total_population_by_year = (total_population_by_year
        .stack(level=["draw", "stat"])   # move draw & stat into index
        .reset_index()                   # turn all index levels into columns
        .rename(columns={0: "population"})    # name the value column
).set_index(["draw", "stat",'year'])
total_population_by_year = total_population_by_year[total_population_by_year.index.get_level_values('draw').isin(cost_scenarios.keys())]['population']
total_population_by_year = total_population_by_year.rename(
    index={"central": "mean"},
    level="stat"
)
cost_by_draw_and_year = input_costs_undiscounted.groupby(['draw', 'stat', 'year'])['cost'].sum()
per_capita_cost_by_draw_and_year = cost_by_draw_and_year/total_population_by_year
per_capita_cost_by_draw_and_year = per_capita_cost_by_draw_and_year.reset_index()
per_capita_cost_by_draw = per_capita_cost_by_draw_and_year.groupby(['draw', 'stat'])[0].mean()

# Get overall estimates for main text
# -----------------------------------------------------------------------------------------------------------------------
cost_by_draw = input_costs.groupby(['draw', 'stat'])['cost'].sum()
undiscounted_cost_by_draw = input_costs_undiscounted.groupby(['draw', 'stat'])['cost'].sum()

# Abstract
consumable_cost_by_draw = input_costs[(input_costs.cost_category == 'medical consumables') & (input_costs.stat == 'mean')].groupby(['draw'])['cost'].sum()
print(f"Under current system capacity, total healthcare delivery costs for 2023–2030 were estimated at "
      f"\{cost_by_draw[0,'mean']/1e9:,.2f} billion [95\% confidence interval (CI), \${cost_by_draw[0,'lower']/1e9:,.2f}b - \${cost_by_draw[0,'upper']/1e9:,.2f}b], averaging "
      f"{undiscounted_cost_by_draw[0,'mean']/1e6/number_of_years_costed:,.2f} million [\${undiscounted_cost_by_draw[0,'lower']/1e6/number_of_years_costed:,.2f}m - \${undiscounted_cost_by_draw[0,'upper']/1e6/number_of_years_costed:,.2f}m] annually. Scenario analysis highlighted strong interdependencies "
      f"within the health system. Improving consumable availability alone increased consumables costs by "
      f"{(consumable_cost_by_draw[2]/consumable_cost_by_draw[0] - 1) * 100:.2f}\%, while expanding human resources for health (HRH) alone increased them by "
      f"{(consumable_cost_by_draw[1]/consumable_cost_by_draw[0] - 1) * 100:.2f}\%. "
      f"When both HRH and consumable availability were expanded together, consumable costs rose by "
      f"{(consumable_cost_by_draw[3]/consumable_cost_by_draw[0] - 1) * 100:.2f}\%"
      f",a combined effect larger than either change alone, illustrating how bottlenecks in one component constrain the impact of improvements in another.")
# Results 1
print(f"The total cost of healthcare delivery in Malawi between 2023 and 2030 was estimated to be "
      f"\${cost_by_draw[0,'mean']/1e9:,.2f} billion [95\% confidence interval (CI), \${cost_by_draw[0,'lower']/1e9:,.2f}b - \${cost_by_draw[0,'upper']/1e9:,.2f}b], under the actual scenario, and increased to "
      f"\${cost_by_draw[2,'mean']/1e9:,.2f} billion [\${cost_by_draw[2,'lower']/1e9:,.2f}b - \${cost_by_draw[2,'upper']/1e9:,.2f}b] under the improved consumable availability scenario, "
      f"followed by \${cost_by_draw[1,'mean']/1e9:,.2f} billion [\${cost_by_draw[1,'lower']/1e9:,.2f}b - \${cost_by_draw[1,'upper']/1e9:,.2f}b] under the expanded HRH scenario and finally "
      f"\${cost_by_draw[3,'mean']/1e9:,.2f} billion [\${cost_by_draw[3,'lower']/1e9:,.2f}b - \${cost_by_draw[3,'upper']/1e9:,.2f}b] under the expanded HRH + improved consumable availability scenario.")
# Results 2
print(f"This translates to an average annual cost of "
      f"\${undiscounted_cost_by_draw[0,'mean']/1e6/number_of_years_costed:,.2f} million [\${undiscounted_cost_by_draw[0,'lower']/1e6/number_of_years_costed:,.2f}m - \${undiscounted_cost_by_draw[0,'upper']/1e6/number_of_years_costed:,.2f}m], under the actual scenario, "
      f"\${undiscounted_cost_by_draw[2,'mean']/1e6/number_of_years_costed:,.2f} million [\${undiscounted_cost_by_draw[2,'lower']/1e6/number_of_years_costed:,.2f}m - \${undiscounted_cost_by_draw[2,'upper']/1e6/number_of_years_costed:,.2f}m] under the improved consumable availability scenario, followed by "
      f"\${undiscounted_cost_by_draw[1,'mean']/1e6/number_of_years_costed:,.2f} million [\${undiscounted_cost_by_draw[1,'lower']/1e6/number_of_years_costed:,.2f}m - \${undiscounted_cost_by_draw[1,'upper']/1e6/number_of_years_costed:,.2f}m] under the expanded HRH scenario and finally "
      f"\${undiscounted_cost_by_draw[3,'mean']/1e6/number_of_years_costed:,.2f} million [\${undiscounted_cost_by_draw[3,'lower']/1e6/number_of_years_costed:,.2f}m - \${undiscounted_cost_by_draw[3,'upper']/1e6/number_of_years_costed:,.2f}m] under the expanded HRH + improved consumable availability scenario.")
# Results 3
print(f"In per capita terms, this is "
      f"\${per_capita_cost_by_draw[0,'mean']:,.2f} [\${per_capita_cost_by_draw[0,'lower']:,.2f} - \${per_capita_cost_by_draw[0,'upper']:,.2f}], "
      f"\${per_capita_cost_by_draw[2,'mean']:,.2f} [\${per_capita_cost_by_draw[2,'lower']:,.2f} - \${per_capita_cost_by_draw[2,'upper']:,.2f}], "
      f"\${per_capita_cost_by_draw[1,'mean']:,.2f} [\${per_capita_cost_by_draw[1,'lower']:,.2f} - \${per_capita_cost_by_draw[1,'upper']:,.2f}], and "
      f"\${per_capita_cost_by_draw[3,'mean']:,.2f} [\${per_capita_cost_by_draw[3,'lower']:,.2f} - \${per_capita_cost_by_draw[3,'upper']:,.2f}] respectively under the four scenarios.")

# Results 4
print(f"Notably, improving consumable availability alone increases the cost of medical consumables by just "
      f"{(consumable_cost_by_draw[2]/consumable_cost_by_draw[0] - 1) * 100:.2f}\% "
      f"because the limited health workforce (HRH) restricts the number of feasible appointments and, consequently, the quantity of consumables dispensed. "
      f"In contrast, expanding HRH alone raises consumable costs by "
      f"{(consumable_cost_by_draw[1]/consumable_cost_by_draw[0] - 1) * 100:.2f}\%"
      f". When both HRH and consumable availability are expanded together, consumable costs increase by "
      f"{(consumable_cost_by_draw[3]/consumable_cost_by_draw[0] - 1) * 100:.2f}\% "
      f"compared to the actual scenario.")
# Results 5
cost_of_hiv_testing =  input_costs[(input_costs.cost_subgroup.str.contains('EIA Elisa')) & (input_costs.stat == 'mean')].groupby(['draw'])['cost'].sum()
cost_of_hiv_treatment =  input_costs[(input_costs.cost_subgroup == 'First-line ART regimen: adult') & (input_costs.stat == 'mean')].groupby(['draw'])['cost'].sum()
cost_of_jadelle =  input_costs[(input_costs.cost_subgroup.str.contains('Jadelle')) & (input_costs.stat == 'mean')].groupby(['draw'])['cost'].sum()
# Get availability estimates
tlo_availability_df = pd.read_csv(path_for_consumable_resourcefiles / "ResourceFile_Consumables_availability_small.csv")
tlo_availability_df = tlo_availability_df[['Facility_ID', 'month','item_code', 'available_prop', 'available_prop_scenario6']]
program_item_mapping = pd.read_csv(path_for_consumable_resourcefiles  / 'ResourceFile_Consumables_Item_Designations.csv')[['Item_Code', 'item_category']]
program_item_mapping = program_item_mapping.rename(columns ={'Item_Code': 'item_code'})[program_item_mapping.item_category.notna()]
mfl = pd.read_csv(resourcefilepath / "healthsystem" / "organisation" / "ResourceFile_Master_Facilities_List.csv")
districts = set(pd.read_csv(resourcefilepath / 'demography' / 'ResourceFile_Population_2010.csv')['District'])
fac_levels = {'0', '1a', '1b', '2', '3', '4'}
tlo_availability_df = tlo_availability_df.merge(mfl[['District', 'Facility_Level', 'Facility_ID']],
                    on = ['Facility_ID'], how='left')
tlo_availability_df = tlo_availability_df.merge(program_item_mapping,
                    on = ['item_code'], how='left')
# Jadelle
jadelle_actual = tlo_availability_df[(tlo_availability_df.item_code == 12) &
                    (tlo_availability_df.Facility_Level.isin(['1a','1b']))]['available_prop'].mean()
jadelle_new = tlo_availability_df[(tlo_availability_df.item_code == 12) &
                    (tlo_availability_df.Facility_Level.isin(['1a','1b']))]['available_prop_scenario6'].mean()
# ART
art_actual = tlo_availability_df[(tlo_availability_df.item_code == 2671) &
                    (tlo_availability_df.Facility_Level.isin(['1a','1b']))]['available_prop'].mean()
art_new = tlo_availability_df[(tlo_availability_df.item_code == 2671) &
                    (tlo_availability_df.Facility_Level.isin(['1a','1b']))]['available_prop_scenario6'].mean()

# Count of HIV testing service delivered
count_by_treatment_id = get_hsi_summary(results_folder, key = 'HSI_Event_non_blank_appt_footprint',
                                        var = "TREATMENT_ID", do_scaling = True)
hiv_test_services = count_by_treatment_id[count_by_treatment_id.index.get_level_values(1) == 'Hiv_Test']
idx = pd.IndexSlice
hiv_test_services = hiv_test_services[hiv_test_services.index.get_level_values(0).isin(list(range(2023,2031)))].loc[:, idx[:, 'central']].sum()

# Browse HIV logs
def get_hiv_summary(results_folder, key, var, summarise_func = 'sum', do_scaling = True):
    def get_count(_df:pd.Series):
        """Summarise the parsed logged-key results for one draw (as dataframe) into a pd.Series."""
        _df = _df.set_axis(_df['date'].dt.year).drop(columns=['date'])

        _df = _df[var]
        _df.index.name = 'year'
        return _df

    hiv_stat = compute_summary_statistics(extract_results(
        Path(results_folder),
        module='tlo.methods.hiv',
        key=key,
        custom_generate_series=get_count,
        do_scaling=do_scaling,
    ), central_measure = 'mean')

    hiv_stat_summmary = hiv_stat.loc[hiv_stat.index.isin(list_of_years_for_plot),[(0,'central'),(1,'central'),(2,'central'),(3,'central')]].agg(summarise_func)

    return hiv_stat_summmary

art_coverage_adult = get_hiv_summary(results_folder, key = 'hiv_program_coverage', var = 'art_coverage_adult', summarise_func = 'mean', do_scaling = False)
art_coverage_adult_VL_suppression = get_hiv_summary(results_folder, key = 'hiv_program_coverage', var = 'art_coverage_adult_VL_suppression', summarise_func = 'mean', do_scaling = False)
prop_tested_adult = get_hiv_summary(results_folder, key = 'hiv_program_coverage', var = 'prop_tested_adult', summarise_func = 'mean', do_scaling = False)
plhiv = get_hiv_summary(results_folder, key = 'summary_inc_and_prev_for_adults_and_children_and_fsw', var = 'total_plhiv', summarise_func = 'sum', do_scaling = True)
testing_yield = get_hiv_summary(results_folder, key = 'hiv_program_coverage', var = 'testing_yield', summarise_func = 'mean', do_scaling = False)
prop_adults_exposed_to_behav_intv = get_hiv_summary(results_folder, key = 'hiv_program_coverage', var = 'prop_adults_exposed_to_behav_intv', summarise_func = 'mean', do_scaling = False)
per_capita_testing_rate = get_hiv_summary(results_folder, key = 'hiv_program_coverage', var = 'per_capita_testing_rate', summarise_func = 'mean', do_scaling = False)
hiv_prev_adult_15plus = get_hiv_summary(results_folder, key = 'summary_inc_and_prev_for_adults_and_children_and_fsw', var = 'hiv_prev_adult_15plus', summarise_func = 'mean', do_scaling = False)

print(f"First, the changes are driven by which constraint is binding. For example, the pattern for the Jadelle"
      f" contraceptive implant differs markedly from that of adult antiretroviral therapy (ART). "
      f"Expanding HRH alone increases the cost of Jadelle by just "
      f"{(cost_of_jadelle[1]/cost_of_jadelle[0] - 1)*100:.2f}\%, whereas improved consumable availability results in a "
      f"{(cost_of_jadelle[2]/cost_of_jadelle[0] - 1)*100:.2f}\% increase, rising to "
      f"{(cost_of_jadelle[3]/cost_of_jadelle[0] - 1)*100:.2f}\% when both HRH and consumables expand. "
      f"In contrast, ART costs rise more modestly across the same scenarios "
      f"({(cost_of_hiv_treatment[1]/cost_of_hiv_treatment[0] - 1)*100:.2f}\%, "
      f"{(cost_of_hiv_treatment[2]/cost_of_hiv_treatment[0] - 1)*100:.2f}\%, and "
      f"{(cost_of_hiv_treatment[3]/cost_of_hiv_treatment[0] - 1)*100:.2f}\%). "
      f"This reflects differences in baseline availability: ART already had very high availability in level 1a/1b facilities "
      f"({art_actual*100:.2f}\%), "
      f"and the improved supply chain scenario increased this by only around "
      f"{(art_new - art_actual)*100:.0f} percentage points. "
      f"For Jadelle, availability increased from "
      f"{(jadelle_actual)*100:.2f}\% by "
      f"{(jadelle_new - jadelle_actual)*100:.0f} percentage points, "
      f"meaning consumable availability was a stronger binding constraint for this service.")

print(f"Second, cost changes do not scale linearly with changes in system capacity. The dynamic nature of the model "
      f"aggregates multiple interacting processes—including population growth, prevention coverage, "
      f" prevalence, testing rates and yields, and competition between appointments (ADD REFERENCE). "
      f"Consequently, while ART costs increase progressively across scenarios "
      f"({(cost_of_hiv_treatment[1]/cost_of_hiv_treatment[0] - 1)*100:.2f}\%, "
      f"{(cost_of_hiv_treatment[2]/cost_of_hiv_treatment[0] - 1)*100:.2f}\%, "
      f"{(cost_of_hiv_treatment[3]/cost_of_hiv_treatment[0] - 1)*100:.2f}\%), "
      f"the cost of HIV testing shows a different pattern: a small increase under expanded HRH "
      f"({(cost_of_hiv_testing[1]/cost_of_hiv_testing[0] - 1)*100:.2f}\%) "
      f"but reductions of "
      f"{(cost_of_hiv_testing[2]/cost_of_hiv_testing[0] - 1)*-100:.2f}\% and "
      f"{(cost_of_hiv_testing[3]/cost_of_hiv_testing[0] - 1)*-100:.2f}\% under the improved consumables and combined scenarios, respectively."
      f" These reductions arise because increased consumable availability intensifies competition with other services, "
      f"leading to fewer HIV testing appointments being delivered.")

# Get figures for overview paper
# -----------------------------------------------------------------------------------------------------------------------
# Figure 2: Estimated costs by cost category
do_stacked_bar_plot_of_cost_by_category(_df = input_costs, _cost_category = 'all', _disaggregate_by_subgroup = False,
                                        _year = list_of_relevant_years_for_costing,show_title = False,
                                        _outputfilepath = figurespath, _scenario_dict = cost_scenarios)

revised_consumable_subcategories = {'cost_of_separately_managed_medical_supplies_dispensed':'cost_of_consumables_dispensed', 'cost_of_excess_separately_managed_medical_supplies_stocked': 'cost_of_excess_consumables_stocked', 'supply_chain':'supply_chain'}
input_costs_new = input_costs.copy()
input_costs_new['cost_subcategory'] = input_costs_new['cost_subcategory'].map(revised_consumable_subcategories).fillna(input_costs_new['cost_subcategory'])

# Figure 3: Estimated costs by cost sub-category
do_stacked_bar_plot_of_cost_by_category(_df = input_costs_new, _cost_category = 'medical consumables', _disaggregate_by_subgroup = False,
                                        _year = list_of_years_for_plot, show_title = False,
                                        _outputfilepath = figurespath, _scenario_dict = cost_scenarios)
do_stacked_bar_plot_of_cost_by_category(_df = input_costs, _cost_category = 'human resources for health', _disaggregate_by_subgroup = False,
                                        _year = list_of_years_for_plot, show_title = False,
                                        _outputfilepath = figurespath, _scenario_dict = cost_scenarios)
do_stacked_bar_plot_of_cost_by_category(_df = input_costs, _cost_category = 'medical equipment', _disaggregate_by_subgroup = False,
                                        _year = list_of_years_for_plot, show_title = False,
                                        _outputfilepath = figurespath, _scenario_dict = cost_scenarios)
do_stacked_bar_plot_of_cost_by_category(_df = input_costs, _cost_category = 'facility operating cost', _disaggregate_by_subgroup = False,
                                        _year = list_of_years_for_plot, show_title = False,
                                        _outputfilepath = figurespath, _scenario_dict = cost_scenarios)


# Figure 4: Estimated costs by year
do_line_plot_of_cost(_df = input_costs_undiscounted, _cost_category='all',
                         _year=list_of_years_for_plot, _draws= [0],
                         disaggregate_by= 'cost_category',
                         _y_lim = 400,
                         show_title = False,
                         _outputfilepath = figurespath)
do_line_plot_of_cost(_df = input_costs_undiscounted, _cost_category='all',
                         _year=list_of_years_for_plot, _draws= [1],
                         disaggregate_by= 'cost_category',
                         _y_lim = 400,
                         show_title = False,
                         _outputfilepath = figurespath)
do_line_plot_of_cost(_df = input_costs_undiscounted, _cost_category='all',
                         _year=list_of_years_for_plot, _draws= [2],
                         disaggregate_by= 'cost_category',
                         _y_lim = 400,
                         show_title = False,
                         _outputfilepath = figurespath)
do_line_plot_of_cost(_df = input_costs_undiscounted, _cost_category='all',
                         _year=list_of_years_for_plot, _draws= [3],
                         disaggregate_by= 'cost_category',
                         _y_lim = 400,
                         show_title = False,
                         _outputfilepath = figurespath)

# Figure D1: Total cost by scenario assuming 0% discount rate
do_stacked_bar_plot_of_cost_by_category(_df = input_costs_undiscounted,
                                        _cost_category = 'all',
                                        _year=list_of_years_for_plot,
                                        _disaggregate_by_subgroup = False,
                                        _outputfilepath = figurespath,
                                        _scenario_dict = cost_scenarios,
                                        _add_figname_suffix = '_UNDISCOUNTED')

# Figure D2: Total cost by scenario assuming variable discount rates
do_stacked_bar_plot_of_cost_by_category(_df = input_costs_variable_discounting,
                                        _cost_category = 'all',
                                        _year=list_of_years_for_plot,
                                        _disaggregate_by_subgroup = False,
                                        _outputfilepath = figurespath,
                                        _scenario_dict = cost_scenarios,
                                        _add_figname_suffix = '_VARIABLE_DISCOUNTING')


# Figure F1-F4: Cost by cost sub-group
cost_categories = ['human resources for health', 'medical consumables',
       'medical equipment', 'facility operating cost']
draws = input_costs.draw.unique().tolist()
colourmap_for_consumables = {
    'First-line ART regimen: adult': '#1f77b4',
    'Test, HIV EIA Elisa': '#ff7f0e',
    'Viral load test': '#2ca02c',
    'Depot-medroxyprogesterone acetate 150 mg - 3 monthly': '#d62728',
    'Oxygen, 1000 liters, primarily with oxygen cylinders': '#9467bd',
    'Phenobarbital, 100 mg': '#8c564b',
    'Rotavirus vaccine': '#e377c2',
    'Carbamazepine 200mg': '#7f7f7f',
    'Infant resuscitator, clear plastic + mask + bag': '#bcbd22',
    'Multiple micronutrient powder (MNP) supplement': '#17becf',
    'Tenofovir (TDF)/Emtricitabine (FTC), tablet, 300/200 mg': '#2b8cbe',
    'Ready-to-use therapeutic food (RUTF)': '#fdae61',
    'Corn Soya Blend (or Supercereal - CSB++)': '#d73027',
    'Male circumcision kit, consumables (10 procedures)': '#756bb1',
    'Jadelle (implant), box of 2': '#ffdd44',
    'Urine analysis': '#66c2a5'
}

for _cat in cost_categories:
    for _d in draws:
        if _cat == 'medical consumables':
            create_summary_treemap_by_cost_subgroup(_df = input_costs, _year = list_of_years_for_plot,
                                               _cost_category = _cat, _draw = _d, _color_map=colourmap_for_consumables,
                                                show_title= False, _label_fontsize= 8, _outputfilepath=figurespath)
        else:
            create_summary_treemap_by_cost_subgroup(_df=input_costs, _year=list_of_years_for_plot,
                                                    _cost_category=_cat, _draw=_d, show_title= False,
                                                    _label_fontsize= 8.5, _outputfilepath=figurespath)


# Get tables for overview paper
# -----------------------------------------------------------------------------------------------------------------------
# Group data and aggregate cost for each draw and stat
def generate_detail_cost_table(_groupby_var, _groupby_var_name, _longtable = False):
    edited_input_costs = input_costs.copy()
    edited_input_costs[_groupby_var] = edited_input_costs[_groupby_var].replace('_', ' ', regex=True)
    edited_input_costs[_groupby_var] = edited_input_costs[_groupby_var].replace('%', '\%', regex=True)
    edited_input_costs[_groupby_var] = edited_input_costs[_groupby_var].replace('&', '\&', regex=True)

    grouped_costs = edited_input_costs.groupby(['cost_category', _groupby_var, 'draw', 'stat'])['cost'].sum()
    # Format the 'cost' values before creating the LaTeX table
    grouped_costs = grouped_costs.apply(lambda x: f"{float(x):,.0f}")
    # Remove underscores from all column values

    # Create a pivot table to restructure the data for LaTeX output
    pivot_data = {}
    for draw in list(cost_scenarios.keys()):
        draw_data = grouped_costs.xs(draw, level='draw').unstack(fill_value=0)  # Unstack to get 'stat' as columns
        # Concatenate 'mean' with 'lower-upper' in the required format
        pivot_data[draw] = draw_data['mean'].astype(str) + ' [' + \
                           draw_data['lower'].astype(str) + '-' + \
                           draw_data['upper'].astype(str) + ']'

    # Combine draw data into a single DataFrame
    table_data = pd.concat([pivot_data[0], pivot_data[1], pivot_data[2], pivot_data[3]], axis=1, keys=['draw=0', 'draw=1', 'draw=2', 'draw=3']).reset_index()

    # Rename columns for clarity
    table_data.columns = ['Cost Category', _groupby_var_name, 'Actual', 'Expanded HRH', 'Improved consumable availability', 'Expanded HRH +\n Improved consumable availability']

    # Replace '\n' with '\\' for LaTeX line breaks
    #table_data['Real World'] = table_data['Real World'].apply(lambda x: x.replace("\n", "\\\\"))
    #table_data['Perfect Health System'] = table_data['Perfect Health System'].apply(lambda x: x.replace("\n", "\\\\"))

    # Convert to LaTeX format with horizontal lines after every row
    latex_table = table_data.to_latex(
        longtable=_longtable,  # Use the longtable environment for large tables
        column_format='|R{3cm}|R{3cm}|R{2.2cm}|R{2.2cm}|R{2.2cm}|R{2.2cm}|',
        caption=f"Summarized Costs by Category and {_groupby_var_name}",
        label=f"tab:cost_by_{_groupby_var}",
        position="h",
        index=False,
        escape=False,  # Prevent escaping special characters like \n
        header=True
    )

    # Add \hline after the header and after every row for horizontal lines
    latex_table = latex_table.replace("\\\\", "\\\\ \\hline")  # Add \hline after each row
    #latex_table = latex_table.replace("_", " ")  # Add \hline after each row

    # Specify the file path to save
    latex_file_path = figurespath / f'cost_by_{_groupby_var}.tex'

    # Write to a file
    with open(latex_file_path, 'w') as latex_file:
        latex_file.write(latex_table)

    # Print latex for reference
    print(latex_table)

# Table F1: Cost by cost subcategory
generate_detail_cost_table(_groupby_var = 'cost_subcategory', _groupby_var_name = 'Cost Subcategory', _longtable = True)
# Table F2: Cost by cost subgroup
generate_detail_cost_table(_groupby_var = 'cost_subgroup', _groupby_var_name = 'Category Subgroup', _longtable = True)

# Figure E1: Consumable inflow to outflow ratio figure
# -----------------------------------------------------------------------------------------------------------------------
inflow_to_outflow_ratio = pd.read_csv(resourcefilepath / "costing/ResourceFile_Consumables_Inflow_Outflow_Ratio.csv")

# Clean category names for plot
clean_category_names = {'cancer': 'Cancer', 'cardiometabolicdisorders': 'Cardiometabolic Disorders',
                        'contraception': 'Contraception', 'general': 'General', 'hiv': 'HIV', 'malaria': 'Malaria',
                        'ncds': 'Non-communicable Diseases', 'neonatal_health': 'Neonatal Health',
                        'other_childhood_illnesses': 'Other Childhood Illnesses', 'reproductive_health': 'Reproductive Health',
                        'road_traffic_injuries': 'Road Traffic Injuries', 'tb': 'Tuberculosis',
                        'undernutrition': 'Undernutrition'}
inflow_to_outflow_ratio['category'] = inflow_to_outflow_ratio['item_category'].map(clean_category_names)


def plot_inflow_to_outflow_ratio(_df, groupby_var, _outputfilepath):
    # Plot the bar plot with gray bars
    plt.figure(figsize=(10, 6))
    sns.barplot(data=_df, x=groupby_var, y='inflow_to_outflow_ratio', errorbar=None, color="gray")

    # Add points representing the distribution of individual values
    sns.stripplot(data=_df, x=groupby_var, y='inflow_to_outflow_ratio', color='black', size=5, alpha=0.2)

    # Wrap x-axis labels ONLY if they are strings and longer than 15 characters
    labels = []
    for label in _df[groupby_var].unique():
        if isinstance(label, str) and len(label) > 15:
            labels.append(textwrap.fill(label, width=15))
        else:
            labels.append(label)
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=90, ha='center')

    # Set labels and title
    plt.xlabel(groupby_var)
    plt.ylabel('Inflow to Outflow Ratio')

    # Show and save plot
    plt.tight_layout()
    plt.savefig(_outputfilepath / f'inflow_to_outflow_ratio_by_{groupby_var}.png')
    plt.close()

plot_inflow_to_outflow_ratio(inflow_to_outflow_ratio, 'fac_type_tlo', _outputfilepath = figurespath)
plot_inflow_to_outflow_ratio(inflow_to_outflow_ratio, 'district', _outputfilepath = figurespath)
plot_inflow_to_outflow_ratio(inflow_to_outflow_ratio, 'item_code', _outputfilepath = figurespath)
plot_inflow_to_outflow_ratio(inflow_to_outflow_ratio, 'category', _outputfilepath = figurespath)

# Generate consumable availability plot
#---------------------------------------
# Load availability data
path_for_cons_resourcefiles = resourcefilepath / "healthsystem/consumables"
full_df_with_scenario = pd.read_csv(path_for_cons_resourcefiles / "ResourceFile_Consumables_availability_small.csv")

# Import item_category
program_item_mapping = pd.read_csv(path_for_cons_resourcefiles / 'ResourceFile_Consumables_Item_Designations.csv')[
    ['Item_Code', 'item_category']]
program_item_mapping = program_item_mapping.rename(columns={'Item_Code': 'item_code'})[
    program_item_mapping.item_category.notna()]

# Get TLO Facility_ID for each district and facility level
mfl = pd.read_csv(resourcefilepath / "healthsystem" / "organisation" / "ResourceFile_Master_Facilities_List.csv")
districts = set(pd.read_csv(resourcefilepath / 'demography' / 'ResourceFile_Population_2010.csv')['District'])
fac_levels = {'0', '1a', '1b', '2', '3', '4'}

df_for_plots = full_df_with_scenario.merge(mfl[['Facility_ID', 'Facility_Level']], on='Facility_ID', how='left',
                                           validate="m:1")
df_for_plots = df_for_plots.merge(program_item_mapping, on='item_code', how='left', validate="m:1")

# Choose scenarios to plot
scenario_list = [6]
chosen_availability_columns = ['available_prop'] + [f'available_prop_scenario{i}' for i in
                                                    scenario_list]
scenario_names_dict = {'available_prop': 'Actual',
                       'available_prop_scenario6': 'Improved consumable availability'}
# recreate the chosen columns list based on the mapping above
chosen_availability_columns = [scenario_names_dict[col] for col in chosen_availability_columns]
df_for_plots = df_for_plots.rename(columns=scenario_names_dict)

# Create heatmap of average availability by Facility_Level and Program for the chosen scenario
for figure_column in ['Actual', 'Improved consumable availability']:
    # Pivot the DataFrame
    aggregated_df = df_for_plots.groupby(['item_category', 'Facility_Level'])[figure_column].mean().reset_index()
    heatmap_df = aggregated_df.pivot(
        index='item_category',
        columns='Facility_Level',
        values=figure_column
    )
    heatmap_df.columns = heatmap_df.columns

    # Calculate the aggregate row and column
    aggregate_col = aggregated_df.groupby('item_category')[figure_column].mean()
    overall_aggregate = aggregated_df[figure_column].mean()
    aggregate_row = aggregated_df.groupby('Facility_Level')[figure_column].mean()

    # Add aggregate row and column
    heatmap_df['Average'] = aggregate_col
    heatmap_df.loc['Average'] = aggregate_row.tolist() + [overall_aggregate]

    # Generate the heatmap
    sns.set(font_scale=1.2)
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_df, annot=True, cmap='RdYlGn',
                cbar_kws={'label': 'Proportion of days on which consumable is available'})

    # Customize the plot
    plt.title(f'')
    plt.xlabel('Facility Level')
    plt.ylabel(f'Disease/Public Health Programme')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    plt.savefig(figurespath / f'heatmap_program_and_level_75perc_{figure_column}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


