from pathlib import Path
from collections import defaultdict

import datetime
import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo.analysis.utils import (
    extract_results,
    get_scenario_outputs,
)
from tlo.methods.healthsystem import get_item_code_from_item_name
from scripts.costing.cost_estimation import (estimate_input_cost_of_scenarios,
                                             do_stacked_bar_plot_of_cost_by_category,
                                             load_unit_cost_assumptions)

# Define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# Print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

# Establish common paths
resourcefilepath = Path("./resources")

# Steps: 1. Create a mapping of data labels in model_costing and relevant calibration data, 2. Create a dataframe with model_costs and calibration costs;
# Load costing resourcefile
unit_costs = load_unit_cost_assumptions(resourcefilepath)
calibration_data = unit_costs["actual_expenditure_data"]

# Make sure values are numeric
budget_columns = ['BUDGETS (USD) (Jul 2019 - Jun 2020)', 'BUDGETS (USD) (Jul 2020 - Jun 2021)',
       'BUDGETS (USD) (Jul 2021 - Jun 2022)']
expenditure_columns = ['EXPENDITURE (USD) (Jul 2018 - Jun 2019)']
calibration_data[budget_columns + expenditure_columns] = calibration_data[budget_columns + expenditure_columns].apply(lambda x: pd.to_numeric(x, errors='coerce'))
# For calibration to budget figures, we take the maximum value across the three years in the RM to provide an
# the maximum of the budget between 2020 and 2022 provides the upper limit to calibrate to (expenditure providing the lower limit)
calibration_data['max_annual_budget_2020-22'] = calibration_data[budget_columns].max(axis=1, skipna = True)
calibration_data = calibration_data.rename(columns = {'EXPENDITURE (USD) (Jul 2018 - Jun 2019)': 'actual_expenditure_2019',
                                                      'Calibration_category': 'calibration_category'})
calibration_data = calibration_data[['calibration_category','actual_expenditure_2019', 'max_annual_budget_2020-22']]
calibration_data = calibration_data.groupby('calibration_category')[['actual_expenditure_2019', 'max_annual_budget_2020-22']].sum().reset_index()
# Repeat this dataframe three times to map to the lower, upper and mean stats in the cost data
calibration_data1 = calibration_data.copy()
calibration_data1['stat'] = 'lower'
calibration_data2 = calibration_data.copy()
calibration_data2['stat'] = 'mean'
calibration_data3 = calibration_data.copy()
calibration_data3['stat'] = 'upper'
calibration_data = pd.concat([calibration_data1, calibration_data2, calibration_data3], axis = 0)
calibration_data = calibration_data.set_index(['calibration_category', 'stat'])

# %%
# Estimate cost for validation
#-----------------------------
# Load result files
resourcefilepath = Path("./resources")
outputfilepath = Path('./outputs/t.mangal@imperial.ac.uk')
#results_folder = get_scenario_outputs('hss_elements-2024-11-12T172311Z.py', outputfilepath)[0] # November 2024 runs
results_folder = get_scenario_outputs('htm_and_hss_runs-2025-01-16T135243Z.py', outputfilepath)[0] # January 2025 runs

# Estimate costs for 2018
input_costs = estimate_input_cost_of_scenarios(results_folder, resourcefilepath, _years = [2018], _draws = [0], summarize = True, cost_only_used_staff=False)

# Manually create a dataframe of model costs and relevant calibration values
def assign_item_codes_to_consumables(_df):
    path_for_consumable_resourcefiles = resourcefilepath / "healthsystem/consumables"
    # Retain only consumable costs
    _df = _df[_df['cost_category'] == 'medical consumables']

    # Create dictionary mapping item_codes to consumables names
    consumables_df = pd.read_csv(resourcefilepath / "costing" / "ResourceFile_Costing_Consumables.csv")[['Item_Code', 'Items']]
    consumables_df = consumables_df[consumables_df['Item_Code'].notna()]
    consumables_dict = dict(zip(consumables_df['Items'], consumables_df['Item_Code']))

    # Replace consumable_name_tlo with item_code
    _df = _df.copy()
    _df['cost_subgroup'] = _df['cost_subgroup'].map(consumables_dict)

    return _df

def get_calibration_relevant_subset_of_costs(_df, # simulated consumable costs by item_code
                                             _col, # column od _df on the basis of which relevant consumable
                                             _col_value, # names of consumables in _df included in calibration category
                                             _calibration_category, # Calibration category to match with
                                             ):
    if (len(_col_value) == 1):
        _df = _df[_df[_col] == _col_value[0]]
    else:
        _df = _df[_df[_col].isin(_col_value)]
    _df['calibration_category'] = _calibration_category
    return _df.groupby(['calibration_category' ,'stat'])['cost'].sum()

# Consumables
#-----------------------------------------------------------------------------------------------------------------------
calibration_data['model_cost'] = np.nan
consumables_costs_by_item_code = assign_item_codes_to_consumables(input_costs)
consumable_list = pd.read_csv(resourcefilepath / 'healthsystem' / 'consumables' / 'ResourceFile_Consumables_Items_and_Packages.csv')
def get_item_code(item):
    return get_item_code_from_item_name(consumable_list, item)

# Malaria consumables
irs = [get_item_code('Indoor residual spraying drugs/supplies to service a client')]
bednets = [get_item_code('Insecticide-treated net')]
antimalarials = [get_item_code('Lumefantrine 120mg/Artemether 20mg,  30x18_540_CMST'),
                 get_item_code('Injectable artesunate'),
                 get_item_code('Fansidar (sulphadoxine / pyrimethamine tab)')]
malaria_rdts = [get_item_code('Malaria test kit (RDT)')]

# HIV consumables
hiv_screening = [get_item_code('Test, HIV EIA Elisa'), get_item_code('VL Test'), get_item_code('CD4 test')]

art = [get_item_code("First-line ART regimen: adult"), get_item_code("Cotrimoxizole, 960mg pppy"), # adult
        get_item_code("First line ART regimen: older child"), get_item_code("Cotrimoxazole 120mg_1000_CMST"), # Older children
        get_item_code("First line ART regimen: young child"), # younger children (also get cotrimoxazole 120mg
        get_item_code('Sulfamethoxazole + trimethropin, tablet 400 mg + 80 mg'),
        get_item_code("Tenofovir (TDF)/Emtricitabine (FTC), tablet, 300/200 mg"), # Adult prep
        get_item_code("Nevirapine, oral solution, 10 mg/ml")] # infant prep

circumcision = [get_item_code('male circumcision kit, consumables (10 procedures)_1_IDA')]

# Tuberculosis consumables
tb_tests = [get_item_code("ZN Stain"), get_item_code("Sputum container"), get_item_code("Microscope slides, lime-soda-glass, pack of 50"),
            get_item_code("Xpert"), get_item_code("Lead rubber x-ray protective aprons up to 150kVp 0.50mm_each_CMST"),
            get_item_code("X-ray"), get_item_code("MGIT960 Culture and DST"),
            get_item_code("Solid culture and DST")]
# consider removing X-ray
tb_treatment = [get_item_code("Cat. I & III Patient Kit A"), # adult primary
                get_item_code("Cat. I & III Patient Kit B"), # child primary
                get_item_code("Cat. II Patient Kit A1"), # adult secondary
                get_item_code("Cat. II Patient Kit A2"), # child secondary
                get_item_code("Treatment: second-line drugs"), # MDR
                get_item_code("Isoniazid/Pyridoxine, tablet 300 mg"), # IPT
                get_item_code("Isoniazid/Rifapentine")] # 3 HP
# Family planning consumables
other_family_planning = [get_item_code("Levonorgestrel 0.15 mg + Ethinyl estradiol 30 mcg (Microgynon), cycle"), # pill
                        get_item_code("IUD, Copper T-380A"), # IUD
                         get_item_code("Depot-Medroxyprogesterone Acetate 150 mg - 3 monthly"), # injection
                         get_item_code("Jadelle (implant), box of 2_CMST"), # implant
                         get_item_code('Implanon (Etonogestrel 68 mg)'), # implant - not currently in use in the model
                         get_item_code("Atropine sulphate  600 micrograms/ml, 1ml_each_CMST")] # female sterilization
condoms = [get_item_code("Condom, male"),
           get_item_code("Female Condom_Each_CMST")]
# Undernutrition
undernutrition = [get_item_code('Supplementary spread, sachet 92g/CAR-150'),
                  get_item_code('Complementary feeding--education only drugs/supplies to service a client'),
                  get_item_code('SAM theraputic foods'),
                  get_item_code('SAM medicines'),
                  get_item_code('Therapeutic spread, sachet 92g/CAR-150'),
                  get_item_code('F-100 therapeutic diet, sach., 114g/CAR-90')]
# Cervical cancer
cervical_cancer = [get_item_code('Specimen container'),
                   get_item_code('Biopsy needle'),
                   get_item_code('Cyclophosphamide, 1 g')]
# Vaccines
vaccines = [get_item_code("Syringe, autodisposable, BCG, 0.1 ml, with needle"),
            get_item_code("Polio vaccine"),
            get_item_code("Pentavalent vaccine (DPT, Hep B, Hib)"),
            get_item_code("Rotavirus vaccine"),
            get_item_code("Measles vaccine"),
            get_item_code("Pneumococcal vaccine"),
            get_item_code("HPV vaccine"),
            get_item_code("Tetanus toxoid, injection")] # not sure if this should be included

other_drugs = set(consumables_costs_by_item_code['cost_subgroup'].unique()) - set(irs) - set(bednets) - set(undernutrition) - set(other_family_planning) - set(vaccines) \
              - set(art) - set(tb_treatment) - set(antimalarials) - set(malaria_rdts) - set(hiv_screening)\
              - set(condoms) - set(tb_tests) # - set(cervical_cancer)

def fill_calibration_costs_from_cost_groups(calibration_df: pd.DataFrame,
                                            mappings: list[tuple[pd.DataFrame, str, str, list[str], float]]) -> pd.DataFrame:
    """
    Fill missing model_costs in a calibration DataFrame based on cost group matches.

    Parameters
    ----------
    calibration_df : pd.DataFrame
        The main calibration DataFrame containing 'calibration_category', 'stat', and 'model_cost'.

    cost_df : pd.DataFrame
        The cost breakdown DataFrame, e.g. `consumables_costs_by_item_code`.

    mappings : list of tuples
        Each tuple defines one mapping:
        (column name to match, calibration category, values to match in that column, optional multiplier)

    Returns
    -------
    pd.DataFrame
        Updated calibration_df with 'model_cost' filled where applicable.
    """
    calibration_df = calibration_df.copy()

    for cost_df, col, category, values, multiplier in mappings:
        print(category)
        grouped = get_calibration_relevant_subset_of_costs(
            _df=cost_df,
            _col=col,
            _col_value=values,
            _calibration_category=category
        )

        if multiplier != 1:
            grouped *= multiplier

        grouped = grouped.reset_index().rename(columns={'cost': 'filled_cost'})

        calibration_df = calibration_df.merge(grouped, on=['calibration_category', 'stat'], how='left')
        calibration_df.loc[calibration_df['model_cost'].isna(), 'model_cost'] = calibration_df.loc[
            calibration_df['model_cost'].isna(), 'filled_cost']
        calibration_df.drop(columns='filled_cost', inplace=True)

    return calibration_df.set_index(['calibration_category', 'stat'])

mappings = [
    # Consumables
    (consumables_costs_by_item_code, 'cost_subgroup', 'Antiretrovirals', art, 82 / (0.103 * 365)),  # Note that the main ARV  regimen in 2018 was tenofovir/lamivudine/efavirenz as opposed to Tenofovir/Lamivudine/Dolutegravir as used in the RF_Costing. The price of this
# was $82 per year (80/(0.103*365)) times what's estimated by the model so let's update this
    (consumables_costs_by_item_code, 'cost_subgroup', 'Indoor Residual Spray', irs, 1),
    (consumables_costs_by_item_code, 'cost_subgroup', 'Bednets', bednets, 1),
    (consumables_costs_by_item_code, 'cost_subgroup', 'Undernutrition commodities', undernutrition, 1),
    (consumables_costs_by_item_code, 'cost_subgroup', 'Other family planning commodities', other_family_planning, 1),
    (consumables_costs_by_item_code, 'cost_subgroup', 'Vaccines', vaccines, 1),
    (consumables_costs_by_item_code, 'cost_subgroup', 'TB Treatment', tb_treatment, 1),
    (consumables_costs_by_item_code, 'cost_subgroup', 'Antimalarials', antimalarials, 1),
    (consumables_costs_by_item_code, 'cost_subgroup', 'Malaria RDTs', malaria_rdts, 1),
    (consumables_costs_by_item_code, 'cost_subgroup', 'HIV Screening/Diagnostic Tests', hiv_screening, 1),
    (consumables_costs_by_item_code, 'cost_subgroup', 'Condoms and Lubricants', condoms, 1),
    (consumables_costs_by_item_code, 'cost_subgroup', 'TB Tests (including RDTs)', tb_tests, 1),
    (consumables_costs_by_item_code, 'cost_subgroup', 'Other Drugs, medical supplies, and commodities', other_drugs, 1),
    (consumables_costs_by_item_code, 'cost_subgroup', 'Voluntary Male Medical Circumcision', circumcision, 1),
    (input_costs, 'cost_subcategory', 'Supply Chain', ['supply_chain'], 1),

    # HR
    (input_costs[input_costs['cost_category'] == 'human resources for health'], 'cost_subcategory', 'Health Worker Salaries', ['salary_for_all_staff'], 1),
    (input_costs[input_costs['cost_category'] == 'human resources for health'], 'cost_subcategory', 'Health Worker Training - Pre-Service', ['preservice_training_and_recruitment_cost_for_attrited_workers'], 1),
    (input_costs[input_costs['cost_category'] == 'human resources for health'], 'cost_subcategory', 'Health Worker Training - In-Service', ['inservice_training_cost_for_all_staff'], 1),
    (input_costs[input_costs['cost_category'] == 'human resources for health'], 'cost_subcategory', 'Mentorships & Supportive Supervision', ['mentorship_and_supportive_cost_for_all_staff'], 1),

    # Equipment
    (input_costs[input_costs['cost_category'] == 'medical equipment'], 'cost_subcategory', 'Medical Equipment - Purchase', ['replacement_cost_annual_total'], 1),
    (input_costs[input_costs['cost_category'] == 'medical equipment'], 'cost_subcategory', 'Medical Equipment - Maintenance', ['service_fee_annual_total', 'spare_parts_annual_total', 'major_corrective_maintenance_cost_annual_total'], 1),

    # Facility operation costs
    (input_costs, 'cost_subgroup', 'Facility utility bills', ['Electricity', 'Water', 'Cleaning', 'Security', 'Food for inpatient cases', 'Facility management'], 1),
    (input_costs, 'cost_subgroup', 'Infrastructure - Rehabilitation', ['Building maintenance'], 1),
    (input_costs, 'cost_subgroup', 'Vehicles - Fuel and Maintenance', ['Vehicle maintenance', 'Ambulance fuel'], 1)]

calibration_data = fill_calibration_costs_from_cost_groups(
    calibration_df=calibration_data,
    mappings=mappings
)


# %%
# 3. Create calibration plot
list_of_consumables_costs_for_calibration_only_hiv = ['Voluntary Male Medical Circumcision', 'HIV Screening/Diagnostic Tests', 'Antiretrovirals']
list_of_consumables_costs_for_calibration_without_hiv =['Indoor Residual Spray', 'Bednets', 'Malaria RDTs', 'Antimalarials', 'TB Tests (including RDTs)', 'TB Treatment', 'Vaccines',
                                                        'Condoms and Lubricants', 'Other family planning commodities',
                                                        'Undernutrition commodities', 'Other Drugs, medical supplies, and commodities']
list_of_hr_costs_for_calibration = ['Health Worker Salaries', 'Health Worker Training - In-Service', 'Health Worker Training - Pre-Service', 'Mentorships & Supportive Supervision']
list_of_equipment_costs_for_calibration = ['Medical Equipment - Purchase', 'Medical Equipment - Maintenance']
list_of_operating_costs_for_calibration = ['Facility utility bills', 'Infrastructure - Rehabilitation', 'Vehicles - Maintenance','Vehicles - Fuel and Maintenance']

# Create folders to store results
costing_outputs_folder = Path('./outputs/costing')
if not os.path.exists(costing_outputs_folder):
    os.makedirs(costing_outputs_folder)
figurespath = costing_outputs_folder / "figures_post_jan2025fix"
if not os.path.exists(figurespath):
    os.makedirs(figurespath)
calibration_outputs_folder = Path(figurespath / 'calibration')
if not os.path.exists(calibration_outputs_folder):
    os.makedirs(calibration_outputs_folder)

def do_cost_calibration_plot(_df, _costs_included, _xtick_fontsize = 10):
    # Filter the dataframe
    _df = _df[(_df.model_cost.notna()) & (_df.index.get_level_values(0).isin(_costs_included))]

    # Reorder the first level of the index based on _costs_included while keeping the second level intact
    _df.index = pd.MultiIndex.from_arrays([
        pd.CategoricalIndex(_df.index.get_level_values(0), categories=_costs_included, ordered=True),
        _df.index.get_level_values(1)
    ])
    _df = _df.sort_index()  # Apply the custom order by sorting the DataFrame

    # For df_mean
    df_mean = _df.loc[_df.index.get_level_values('stat') == 'mean'].reset_index(level='stat', drop=True)/1e6
    total_mean = pd.DataFrame(df_mean.sum()).T  # Calculate the total and convert it to a DataFrame
    total_mean.index = ['Total']  # Name the index of the total row as 'Total'
    df_mean = pd.concat([df_mean, total_mean], axis=0)  # Concatenate the total row

    # For df_lower
    df_lower = _df.loc[_df.index.get_level_values('stat') == 'lower'].reset_index(level='stat', drop=True)/1e6
    total_lower = pd.DataFrame(df_lower.sum()).T  # Calculate the total and convert it to a DataFrame
    total_lower.index = ['Total']  # Name the index of the total row as 'Total'
    df_lower = pd.concat([df_lower, total_lower], axis=0)  # Concatenate the total row

    # For df_upper
    df_upper = _df.loc[_df.index.get_level_values('stat') == 'upper'].reset_index(level='stat', drop=True)/1e6
    total_upper = pd.DataFrame(df_upper.sum()).T  # Calculate the total and convert it to a DataFrame
    total_upper.index = ['Total']  # Name the index of the total row as 'Total'
    df_upper = pd.concat([df_upper, total_upper], axis=0)  # Concatenate the total row

    # Create the dot plot
    plt.figure(figsize=(12, 8))

    # Plot model_cost as dots with confidence interval error bars
    yerr_lower = (df_mean['model_cost'] - df_lower['model_cost']).clip(lower = 0)
    yerr_upper = (df_upper['model_cost'] - df_mean['model_cost']).clip(lower = 0)
    plt.errorbar(df_mean.index, df_mean['model_cost'],
                 yerr=[yerr_lower, yerr_upper],
                 fmt='o', label='Model Cost', ecolor='gray', capsize=5, color='saddlebrown')

    # Plot annual_expenditure_2019 and max_annual_budget_2020-22 as dots
    plt.plot(df_mean.index, df_mean['actual_expenditure_2019'], 'bo', label='Actual Expenditure 2019', markersize=8)
    plt.plot(df_mean.index, df_mean['max_annual_budget_2020-22'], 'go', label='Max Annual Budget 2020-22', markersize=8)

    # Draw a blue line between annual_expenditure_2019 and max_annual_budget_2020-22
    plt.vlines(df_mean.index, df_mean['actual_expenditure_2019'], df_mean['max_annual_budget_2020-22'], color='blue',
               label='Expenditure-Budget Range')

    # Add labels to the model_cost dots (yellow color, slightly shifted right)
    for i, (x, y) in enumerate(zip(df_mean.index, df_mean['model_cost'])):
        plt.text(i + 0.05, y, f'{y:.2f}', ha='left', va='bottom', fontsize=9,
                 color='saddlebrown')  # label model_cost values

    # Add labels and title
    cost_subcategory = [name for name in globals() if globals()[name] is _costs_included][0]
    cost_subcategory = cost_subcategory.replace('list_of_', '').replace('_for_calibration', '')
    plt.xlabel('Cost Sub-Category')
    plt.ylabel('Costs (USD), millions')
    plt.title(f'Model Cost vs Annual Expenditure 2019 and Max(Annual Budget 2020-22)\n {cost_subcategory}')

    # Set a white background and black border
    plt.grid(False)
    ax = plt.gca()  # Get current axes
    ax.set_facecolor('white')  # Set the background color to white
    for spine in ax.spines.values():  # Iterate over all borders (spines)
        spine.set_edgecolor('black')  # Set the border color to black
        spine.set_linewidth(1.5)  # Adjust the border width if desired

    # Customize x-axis labels for readability
    max_label_length = 15  # Define a maximum label length for wrapping
    wrapped_labels = [textwrap.fill(str(label), max_label_length) for label in df_mean.index]
    plt.xticks(ticks=range(len(wrapped_labels)), labels=wrapped_labels, rotation=45, ha='right', fontsize=_xtick_fontsize)

    # Adding a legend
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)

    # Tight layout and save the figure
    plt.tight_layout()
    plt.savefig(calibration_outputs_folder / f'calibration_dot_plot_{cost_subcategory}.png', dpi=100,
                bbox_inches='tight')
    plt.close()

# Call the function for each variable and cost list
all_consumable_costs = list_of_consumables_costs_for_calibration_only_hiv + list_of_consumables_costs_for_calibration_without_hiv + ['Supply Chain']
all_calibration_costs = all_consumable_costs + list_of_hr_costs_for_calibration + list_of_equipment_costs_for_calibration + list_of_operating_costs_for_calibration

do_cost_calibration_plot(calibration_data,list_of_consumables_costs_for_calibration_without_hiv)
do_cost_calibration_plot(calibration_data,list_of_consumables_costs_for_calibration_only_hiv)
do_cost_calibration_plot(calibration_data,all_consumable_costs)
do_cost_calibration_plot(calibration_data, list_of_hr_costs_for_calibration)
do_cost_calibration_plot(calibration_data, list_of_equipment_costs_for_calibration)
do_cost_calibration_plot(calibration_data, list_of_operating_costs_for_calibration)
do_cost_calibration_plot(calibration_data,all_calibration_costs, _xtick_fontsize = 7)

# Extract calibration data table for manuscript appendix
calibration_data_extract = calibration_data[calibration_data.index.get_level_values(1) == 'mean']
calibration_data_extract = calibration_data_extract.droplevel(level=1).reset_index()
# Create a higher level cost category in the calibration data
calibration_categories_dict = {'Other Drugs, medical supplies, and commodities': 'medical consumables',
'Program Management & Administration': 'Not represented in TLO model',
'Non-EHP consumables': 'Not represented in TLO model',
'Voluntary Male Medical Circumcision': 'medical consumables',
'Indoor Residual Spray': 'medical consumables',
'Bednets': 'medical consumables',
'Antimalarials': 'medical consumables',
'Undernutrition commodities': 'medical consumables',
'Cervical Cancer': 'medical consumables',
'Condoms and Lubricants': 'medical consumables',
'Other family planning commodities': 'medical consumables',
'TB Tests (including RDTs)': 'medical consumables',
'TB Treatment': 'medical consumables',
'Vaccines': 'medical consumables',
'Malaria RDTs': 'medical consumables',
'HIV Screening/Diagnostic Tests': 'medical consumables',
'Antiretrovirals': 'medical consumables',
'Health Worker Salaries': 'human resources for health',
'Health Worker Training - In-Service': 'human resources for health',
'Health Worker Training - Pre-Service': 'human resources for health',
'Mentorships & Supportive Supervision': 'human resources for health',
'Facility utility bills': 'facility operating cost',
'Infrastructure - New Builds': 'Not represented in TLO model',
'Infrastructure - Rehabilitation': 'facility operating cost',
'Infrastructure - Upgrades': 'Not represented in TLO model',
'Medical Equipment - Maintenance': 'medical equipment',
'Medical Equipment - Purchase': 'medical equipment',
'Vehicles - Fuel and Maintenance': 'facility operating cost',
'Vehicles - Purchase': 'Not represented in TLO model',
'Vehicles - Fuel and Maintenance (Beyond Government and CHAM)': 'Not represented in TLO model',
'Supply Chain': 'medical consumables',
'Supply Chain - non-EHP consumables': 'Not represented in TLO model',
'Unclassified': 'Not represented in TLO model'}
calibration_data_extract['cost_category'] = calibration_data_extract['calibration_category'].map(calibration_categories_dict)

# Obtain the magnitude of deviation of model estimate from expenditure or budget estimate, whichever is closer
# Step 1: Calculate signed deviations
calibration_data_extract['dev_from_expenditure'] = (
    (calibration_data_extract['model_cost'] - calibration_data_extract['actual_expenditure_2019'])
    / calibration_data_extract['actual_expenditure_2019']
)

calibration_data_extract['dev_from_budget'] = (
    (calibration_data_extract['model_cost'] - calibration_data_extract['max_annual_budget_2020-22'])
    / calibration_data_extract['max_annual_budget_2020-22']
)

# Step 2: For each row, pick the value (signed) with the minimum absolute deviation
calibration_data_extract['Deviation of estimated cost from nearest benchmark (%)'] = calibration_data_extract.apply(
    lambda row: min(
        [row['dev_from_expenditure'], row['dev_from_budget']],
        key=lambda x: abs(x) if pd.notna(x) else float('inf')
    ),
    axis=1
)

# Format the deviation as a percentage with 2 decimal points
calibration_data_extract['Deviation of estimated cost from nearest benchmark (%)'] = (
    calibration_data_extract['Deviation of estimated cost from nearest benchmark (%)']
    .map(lambda x: f"{x * 100:.2f}%")
)
calibration_data_extract.loc[calibration_data_extract['Deviation of estimated cost from nearest benchmark (%)'] == 'nan%', 'Deviation of estimated cost from nearest benchmark (%)'] = 'NA'
# Replace if calibration is fine
calibration_condition_met = ((calibration_data_extract['model_cost'] > calibration_data_extract[['actual_expenditure_2019', 'max_annual_budget_2020-22']].min(axis=1)) &
    (calibration_data_extract['model_cost'] < calibration_data_extract[['actual_expenditure_2019', 'max_annual_budget_2020-22']].max(axis=1)))

calibration_data_extract.loc[calibration_condition_met,
    'Deviation of estimated cost from nearest benchmark (%)'
] = 'Within expenditure-budget range'

calibration_data_extract.loc[calibration_data_extract['model_cost'].isna(), 'model_cost'] = 'NA'

calibration_data_extract = calibration_data_extract.sort_values(by=['cost_category', 'calibration_category'])
calibration_data_extract = calibration_data_extract[['cost_category', 'calibration_category', 'actual_expenditure_2019', 'max_annual_budget_2020-22', 'model_cost', 'Deviation of estimated cost from nearest benchmark (%)']]
calibration_data_extract = calibration_data_extract.rename(columns = {'cost_category': 'Cost Category',
                                                            'calibration_category': 'Relevant RM group',
                                                            'actual_expenditure_2019': 'Recorded Expenditure (FY 2018/19)',
                                                            'max_annual_budget_2020-22': 'Maximum Recorded Annual Budget (FY 2019/20 - 2021/22)',
                                                            'model_cost': 'Estimated cost (TLO Model, 2018)'
    })

calibration_data_extract.to_csv(figurespath / 'calibration/calibration.csv')
def convert_df_to_latex(_df, _longtable = False, numeric_columns = []):
    _df['Relevant RM group'] = _df['Relevant RM group'].str.replace('&', r'\&', regex=False)
    # Format numbers to the XX,XX,XXX.XX format for all numeric columns
    _df[numeric_columns] = _df[numeric_columns].applymap(lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x)

    # Convert to LaTeX format with horizontal lines after every row
    latex_table = _df.to_latex(
        longtable=_longtable,  # Use the longtable environment for large tables
        column_format='|R{3.5cm}|R{3.5cm}|R{2.1cm}|R{2.1cm}|R{2.1cm}|R{2.1cm}|',
        caption=f"Comparison of Model Estimates with Resource Mapping data",
        label=f"tab:calibration_breakdown",
        position="h",
        index=False,
        escape=False,  # Prevent escaping special characters like \n
        header=True
    )

    # Add \hline after the header and after every row for horizontal lines
    latex_table = latex_table.replace("\\\\", "\\\\ \\hline")  # Add \hline after each row
    latex_table = latex_table.replace("%", "\%")  # Add \hline after each row
    latex_table = latex_table.replace("Program Management & Administration", "Program Management \& Administration")  # Add \hline after each row
    latex_table = latex_table.replace("Mentorships & Supportive Supervision", "Mentorships \& Supportive Supervision")  # Add \hline after each row

    # latex_table = latex_table.replace("_", " ")  # Add \hline after each row

    # Specify the file path to save
    latex_file_path = calibration_outputs_folder / f'calibration_breakdown.tex'

    # Write to a file
    with open(latex_file_path, 'w') as latex_file:
        latex_file.write(latex_table)

    # Print latex for reference
    print(latex_table)

convert_df_to_latex(calibration_data_extract, _longtable = True, numeric_columns = ['Recorded Expenditure (FY 2018/19)',
                                                                                     'Maximum Recorded Annual Budget (FY 2019/20 - 2021/22)',
                                                                                     'Estimated cost (TLO Model, 2018)'])

# Stacked bar charts to represent all cost sub-groups
do_stacked_bar_plot_of_cost_by_category(_df = input_costs, _cost_category = 'all', _disaggregate_by_subgroup = False,
                                        _outputfilepath = calibration_outputs_folder)

# Extract for manuscript
# Convert column to float (removing commas first)
cols_to_convert = ['Recorded Expenditure (FY 2018/19)', 'Estimated cost (TLO Model, 2018)']
calibration_data_extract[cols_to_convert] = (
    calibration_data_extract[cols_to_convert]
    .replace({'NA': None})  # Convert 'NA' to None (optional, depends on dataset)
    .apply(lambda x: x.str.replace(',', '', regex=True))
    .astype(float)
)
# Sum only the relevant rows
total_expenditure = calibration_data_extract[calibration_data_extract['Cost Category'] != 'Not represented in TLO model']['Recorded Expenditure (FY 2018/19)'].sum()
total_cost_estimate = calibration_data_extract[calibration_data_extract['Cost Category'] != 'Not represented in TLO model']['Estimated cost (TLO Model, 2018)'].sum()

# Extract
print(f"Based on the TLO model, we estimate the total healthcare cost to be "
      f"\${total_cost_estimate/1e6:,.2f} million "
      f"({(1 - total_cost_estimate/total_expenditure)*100:,.2f}\% "
      f"lower than the RM expenditure estimate).")

# Extracts on consumable calibration for Appendix C
# first obtain consumables dispensed estimate
years = [2018]

def drop_outside_period(_df, _years):
    """Return a DataFrame filtered to only include rows within the specified _years"""
    # Define year range
    start_year = min(_years)
    end_year = max(_years)

    # Filter rows by year
    return _df[_df['date'].dt.year.between(start_year, end_year)]

def get_quantity_of_consumables_dispensed(results_folder, _years):
    def get_counts_of_items_requested(_df):
        _df = drop_outside_period(_df, _years)
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
        do_scaling=True)

    cons_dispensed = cons_req.xs("Used", level=2)  # only keep actual dispensed amount, i.e. when available
    return cons_dispensed
idx = pd.IndexSlice
consumables_dispensed = get_quantity_of_consumables_dispensed(results_folder, _years = years)
consumables_dispensed = consumables_dispensed.reset_index().rename(columns={'level_0': 'Item_Code', 'level_1': 'year'})
consumables_dispensed[idx['year']] = pd.to_datetime(
    consumables_dispensed[idx['year']]).dt.year  # Extract only year from date
# Keep only baseline
consumables_dispensed_filtered = consumables_dispensed.loc[:, consumables_dispensed.columns.get_level_values(0) == 0]
consumables_dispensed_summary = pd.concat([
    consumables_dispensed_filtered.mean(axis=1).rename(('mean',)),
    consumables_dispensed_filtered.quantile(0.025, axis=1).rename(('lower',)),
    consumables_dispensed_filtered.quantile(0.975, axis=1).rename(('upper',))
], axis=1)

consumables_dispensed_summary = pd.concat([consumables_dispensed_summary, consumables_dispensed[[idx['Item_Code'], idx['year']]]], axis=1)
consumables_dispensed_summary.columns = ['mean', 'lower', 'upper', 'Item_Code', 'year']
consumables_dispensed_dict = dict(zip(consumables_dispensed_summary['Item_Code'], consumables_dispensed_summary['mean']))

# Antimalarials
la = get_item_code('Lumefantrine 120mg/Artemether 20mg,  30x18_540_CMST')
artesunate = get_item_code('Injectable artesunate')
sp = get_item_code('Fansidar (sulphadoxine / pyrimethamine tab)')
print(f"{consumables_dispensed_dict[str(la)]:,.0f} tablets of Lumefantrine/Arthemeter, "
      f"{consumables_dispensed_dict[str(artesunate)]:,.0f} ampoules of Injectable artesunate, "
      f"and {consumables_dispensed_dict[str(sp)]:,.0f} tablets of Sulphadoxine / pyrimethamine were dispensed as per the model."
      f"The units of dispensation in the Open LMIS are not clear so these could not be compared directly. ")

# Malaria testing
malaria_rdt = get_item_code('Malaria test kit (RDT)')
print(f"There is good correspondence between quantity of Malaria test kits (RDT) logged by the TLO model and LMIS data -  "
      f"14,295,107 units dispensed as per OpenLMIS, "
      f"{consumables_dispensed_dict[str(malaria_rdt)]:,.0f} units dispensed as per modelled estimates")

# Bednets
bednets = get_item_code('Insecticide-treated net')
print(f"792,101  units dispensed as per OpenLMIS, "
      f"{consumables_dispensed_dict[str(bednets)]:,.0f} units dispensed as per modelled estimates")

# TB treatment
adult_primary = get_item_code("Cat. I & III Patient Kit A") # adult primary
child_primary = get_item_code("Cat. I & III Patient Kit B") # child primary
adult_second = get_item_code("Cat. II Patient Kit A1") # adult secondary
child_second = get_item_code("Cat. II Patient Kit A2") # child secondary
mdr = get_item_code("Treatment: second-line drugs") # MDR
ipt = get_item_code("Isoniazid/Pyridoxine, tablet 300 mg") # IPT
iso_rifa = get_item_code("Isoniazid/Rifapentine")
print(f"\item {consumables_dispensed_dict[str(adult_primary)]:,.0f} units of primary treatment kits for adults "
      f"\item {consumables_dispensed_dict[str(child_primary)]:,.0f} units of primary treatment kids for children "
      f"\item {consumables_dispensed_dict[str(adult_second)]:,.0f} units of secondary treatment kits for adults "
      f"\item {consumables_dispensed_dict[str(child_second)]:,.0f} units of secondary treatment kits for children "
      f"\item {consumables_dispensed_dict[str(mdr)]:,.0f} kits for Multi-drug resistant treatment "
      f"\item {consumables_dispensed_dict[str(ipt)]:,.0f} tablets of preventive Isoniazid/Pyridoxine, and "
      f"\item {consumables_dispensed_dict[str(iso_rifa)]:,.0f} tablets of preventive Isoniazid/Rifapentine")

# TB testing
zn_stain = get_item_code("ZN Stain")
sputum_container = get_item_code("Sputum container")
slides = get_item_code("Microscope slides, lime-soda-glass, pack of 50")
xpert = get_item_code("Xpert")
xray_aprons = get_item_code("Lead rubber x-ray protective aprons up to 150kVp 0.50mm_each_CMST")
film = get_item_code("X-ray")
culture = get_item_code("MGIT960 Culture and DST")
solid_culture = get_item_code("Solid culture and DST")

print(f"\item `ZN Stain' - No record in OpenLMIS; {consumables_dispensed_dict[str(zn_stain)]:,.0f} units dispensed as per modelled estimates"
      f"\item `Sputum container' - No record in OpenLMIS; {consumables_dispensed_dict[str(sputum_container)]:,.0f} units dispensed as per modelled estimates"
      f"\item `Microscope slides, lime-soda-glass, pack of 50' - No record in OpenLMIS; {consumables_dispensed_dict[str(slides)]:,.0f} units dispensed as per modelled estimates"
      f"\item `Xpert cartridge' - 25,205 cartridges recorded in OpenLMIS; {consumables_dispensed_dict[str(xpert)]:,.0f} units dispensed as per modelled estimates. "
      f"\item `Lead rubber x-ray protective aprons up to 150kVp 0.50mm' - No record in OpenLMIS; {consumables_dispensed_dict[str(xray_aprons)]:,.0f} units dispensed as per modelled estimates. "
      f"\item `X-Ray film' - No record in OpenLMIS; {consumables_dispensed_dict[str(film)]:,.0f} units dispensed as per modelled estimates. ")
# Culture not included as this these have been replaced by ZN stain - there was no record in OpenLMIS


# HIV testing
hiv_test = get_item_code('Test, HIV EIA Elisa')
vl_test = get_item_code('VL Test')
print(f"{consumables_dispensed_dict[str(hiv_test)]:,.0f} units of 'Test, HIV EIA Elisa', and "
      f"{consumables_dispensed_dict[str(vl_test)]:,.0f} units of 'VL Test' were dispensed in 2018 as per the model."
      f" OpenLMIS recorded 9,382,640 units of 'Test, HIV EIA Elisa' and there was no record of VL tests. We suspect that this discrepancy arises "
      f"because some channels of HIV testing might not be recorded in the model.")

# Family Planning commodities
jadelle = get_item_code("Jadelle (implant), box of 2_CMST")
iud = get_item_code("IUD, Copper T-380A")
levonorgestrel = get_item_code("Levonorgestrel 0.15 mg + Ethinyl estradiol 30 mcg (Microgynon), cycle")
depot = get_item_code("Depot-Medroxyprogesterone Acetate 150 mg - 3 monthly")

print(f"\item `Jadelle (implant), box of 2\_CMST' -  53,585 units dispensed as per OpenLMIS, {consumables_dispensed_dict[str(jadelle)]:,.0f} units dispensed as per modelled estimates"
      f"\item `IUD, Copper T-380A' -  4,079 units dispensed as per OpenLMIS, {consumables_dispensed_dict[str(iud)]:,.0f} units dispensed as per modelled estimates"
      f"\item `Depot-Medroxyprogesterone Acetate 150 mg - 3 monthly' -   2,807,681 units dispensed as per OpenLMIS, {consumables_dispensed_dict[str(depot)]:,.0f} dispensed as per modelled estimates"
      f"\item `Levonorgestrel 0.15 mg + Ethinyl estradiol 30 mcg (Microgynon), cycle' -  1,795,325 units (37,701,825 tablets) dispensed as per OpenLMIS, {consumables_dispensed_dict[str(levonorgestrel)]:,.0f} tablets dispensed as per modelled estimates")
