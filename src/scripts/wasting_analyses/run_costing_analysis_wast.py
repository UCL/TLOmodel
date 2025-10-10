"""
Produces cost analysis outputs for wasting paper
TODO: add more details
"""

# %% Import statements
import time
from pathlib import Path

import pandas as pd

from src.scripts.costing.cost_estimation import (
    do_stacked_bar_plot_of_cost_by_category,
    estimate_input_cost_of_scenarios,
)
from tlo import Date
from tlo.analysis.utils import (
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
)


def run_costing_analysis_wast(cost_outcome_folderpath: Path, SQ_timestamp: str, scen_timestamps_suffix: str,
                              force_calculation: list):
    # `start time of the analysis
    total_time_start = time.time()

    # Save resource, output, outcome data, and figure output paths
    resourcefilepath = Path("./resources")  # resources (parameters etc)
    outputfilepath = Path('./outputs/sejjej5@ucl.ac.uk/wasting/scenarios/costing_outputs')  # simulated data
    figurespath = Path('./outputs/sejjej5@ucl.ac.uk/wasting/scenarios/_outcomes')  # figures

    # Load result files
    # ------------------------------------------------------------------------------------------------------------------
    results_folder = get_scenario_outputs(f'costing-{SQ_timestamp}.py', outputfilepath)[0]

    # Check can read results from draw=0, run=0
    load_pickled_dataframes(results_folder, 0, 0)  # look at one log (so can decide what to extract)
    # params = extract_params(results_folder)
    get_scenario_info(results_folder)

    # Declare default parameters for cost analysis
    # ------------------------------------------------------------------------------------------------------------------
    # Period relevant for costing
    TARGET_PERIOD = (Date(2026, 1, 1), Date(2030, 12, 31))
    relevant_period_for_costing = [i.year for i in TARGET_PERIOD]
    list_of_relevant_years_for_costing = list(range(relevant_period_for_costing[0], relevant_period_for_costing[1] + 1))
    list_of_years_for_plot = list(range(2026, 2031))
    # number_of_years_costed = relevant_period_for_costing[1] - relevant_period_for_costing[0] + 1

    # Scenarios
    cost_scenarios = {0: "SQ", 1: "GM", 2: "CS", 3: "FS", 4:"GM_FS", 5:"CS_FS", 6:"GM_CS_FS", 7:"GM_CS"}

    # Costing parameters
    discount_rate = 0.03
    # discount_rate_lomas = {2023: 0.0036, 2024: 0.0040, 2025: 0.0039, 2026: 0.0042, 2027: 0.0042, 2028: 0.0041,
    #                        2029: 0.0041, 2030: 0.0040}# get the list of discount rates from 2023 until 2030

    # Estimate standard input costs of scenario
    # -----------------------------------------------------------------------------------------------------------------------
    cost_scenarios_draw_nmbs = list(cost_scenarios.keys())

    input_costs_file_path = cost_outcome_folderpath / f"input_cost_outcomes_{SQ_timestamp}.pkl"
    if input_costs_file_path.exists() and not force_calculation[4]:
        print("\nloading input cost outcomes from file ...")
        input_costs = pd.read_pickle(input_costs_file_path)
    else:
        print("\ninput cost outcomes calculation ...")
        # Standard 3% discount rate
        input_costs = estimate_input_cost_of_scenarios(
            results_folder, resourcefilepath, _draws=cost_scenarios_draw_nmbs,
            _years=list_of_relevant_years_for_costing, cost_only_used_staff=True,
            _discount_rate=discount_rate, summarize=True
        )
        print("saving input cost outcomes to file ...")
        input_costs.to_pickle(input_costs_file_path)

    # pd.set_option('display.max_columns', None)  # Show all columns
    # pd.set_option('display.max_rows', None)  # Show all rows
    # pd.set_option('display.max_colwidth', None)  # Show full content of each row
    # print(f"\ninput_costs:\n{input_costs}")
    # print(f"\ninput_costs index:\n{input_costs.index},"
    #       f"\ninput_costs columns:\n{input_costs.columns}")
    # print(f"\nUnique cost_category values:\n{input_costs['cost_category'].unique()}")
    #
    # print(f"\ninput_costs (medical consumables only):\n{input_costs[input_costs['cost_category'] == 'medical consumables']}")



    # # Undiscounted costs
    # input_costs_undiscounted = estimate_input_cost_of_scenarios(results_folder, resourcefilepath, _draws = cost_scenarios_draw_nmbs,
    #                                                             _years=list_of_relevant_years_for_costing, cost_only_used_staff=True,
    #                                                             _discount_rate = 0, summarize = True)
    #
    # # Cost with variable discount rate based on Lomas et al (2021)
    # input_costs_variable_discounting = estimate_input_cost_of_scenarios(results_folder, resourcefilepath, _draws = cost_scenarios_draw_nmbs,
    #                                                                     _years=list_of_relevant_years_for_costing, cost_only_used_staff=True,
    #                                                                     _discount_rate = discount_rate_lomas, summarize = True)

    # Get overall estimates for main text
    # -----------------------------------------------------------------------------------------------------------------------
    # cost_by_draw = input_costs.groupby(['draw', 'stat'])['cost'].sum()
    # undiscounted_cost_by_draw = input_costs_undiscounted.groupby(['draw', 'stat'])['cost'].sum()

    # Abstract
    # consumable_cost_by_draw = input_costs[(input_costs.cost_category == 'medical consumables') & (input_costs.stat == 'mean')].groupby(['draw'])['cost'].sum()
    # print(f"Under current system capacity, total healthcare delivery costs for 2023–2030 were estimated at \$"
    #       f"{cost_by_draw[0,'mean']/1e9:,.2f} billion [95\% confidence interval (CI), \${cost_by_draw[0,'lower']/1e9:,.2f}b - \${cost_by_draw[0,'upper']/1e9:,.2f}b], averaging \$"
    #       f"{undiscounted_cost_by_draw[0,'mean']/1e6/number_of_years_costed:,.2f} million [\${undiscounted_cost_by_draw[0,'lower']/1e6/number_of_years_costed:,.2f}m - \${undiscounted_cost_by_draw[0,'upper']/1e6/number_of_years_costed:,.2f}m] annually."
    #       f" Scenario analysis revealed the importance of health system interdependencies: improving consumable availability alone led to a modest "
    #       f"{(consumable_cost_by_draw[5]/consumable_cost_by_draw[0] - 1) * 100:.2f}\%"
    #       f" increase in consumables cost due to constraints in the health workforce. In contrast, expanding human resources for health (HRH) increased consumables costs by "
    #       f"{(consumable_cost_by_draw[3]/consumable_cost_by_draw[0] - 1) * 100:.2f}\%"
    #       f", while jointly expanding HRH and consumable availability raised consumables costs by "
    #       f"{(consumable_cost_by_draw[8]/consumable_cost_by_draw[0] - 1) * 100:.2f}\%, "
    #       f"illustrating how bottlenecks in one component limit the effect of changes in another.")
    # Results 1
    # print(f"The total cost of healthcare delivery in Malawi between 2023 and 2030 was estimated to be "
    #       f"\${cost_by_draw[0,'mean']/1e9:,.2f} billion [95\% confidence interval (CI), \${cost_by_draw[0,'lower']/1e9:,.2f}b - \${cost_by_draw[0,'upper']/1e9:,.2f}b], under the actual scenario, and increased to "
    #       f"\${cost_by_draw[5,'mean']/1e9:,.2f} billion [\${cost_by_draw[5,'lower']/1e9:,.2f}b - \${cost_by_draw[5,'upper']/1e9:,.2f}b] under the improved consumable availability scenario, "
    #       f"followed by \${cost_by_draw[3,'mean']/1e9:,.2f} billion [\${cost_by_draw[3,'lower']/1e9:,.2f}b - \${cost_by_draw[3,'upper']/1e9:,.2f}b] under the expanded HRH scenario and finally "
    #       f"\${cost_by_draw[8,'mean']/1e9:,.2f} billion [\${cost_by_draw[8,'lower']/1e9:,.2f}b - \${cost_by_draw[8,'upper']/1e9:,.2f}b] under the expanded HRH + improved consumable availability scenario.")
    # # Results 2
    # print(f"This translates to an average annual cost of "
    #       f"\${undiscounted_cost_by_draw[0,'mean']/1e6/number_of_years_costed:,.2f} million [\${undiscounted_cost_by_draw[0,'lower']/1e6/number_of_years_costed:,.2f}m - \${undiscounted_cost_by_draw[0,'upper']/1e6/number_of_years_costed:,.2f}m], under the actual scenario, "
    #       f"\${undiscounted_cost_by_draw[5,'mean']/1e6/number_of_years_costed:,.2f} million [\${undiscounted_cost_by_draw[5,'lower']/1e6/number_of_years_costed:,.2f}m - \${undiscounted_cost_by_draw[5,'upper']/1e6/number_of_years_costed:,.2f}m] under the improved consumable availability scenario, followed by "
    #       f"\${undiscounted_cost_by_draw[3,'mean']/1e6/number_of_years_costed:,.2f} million [\${undiscounted_cost_by_draw[3,'lower']/1e6/number_of_years_costed:,.2f}m - \${undiscounted_cost_by_draw[3,'upper']/1e6/number_of_years_costed:,.2f}m] under the expanded HRH scenario and finally "
    #       f"\${undiscounted_cost_by_draw[8,'mean']/1e6/number_of_years_costed:,.2f} million [\${undiscounted_cost_by_draw[8,'lower']/1e6/number_of_years_costed:,.2f}m - \${undiscounted_cost_by_draw[8,'upper']/1e6/number_of_years_costed:,.2f}m] under the expanded HRH + improved consumable availability scenario.")
    # # Results 3
    # print(f"Notably, improving consumable availability alone increases the cost of medical consumables by just "
    #       f"{(consumable_cost_by_draw[5]/consumable_cost_by_draw[0] - 1) * 100:.2f}\% "
    #       f"because the limited health workforce (HRH) restricts the number of feasible appointments and, consequently, the quantity of consumables dispensed. "
    #       f"In contrast, expanding HRH alone raises consumable costs by "
    #       f"{(consumable_cost_by_draw[3]/consumable_cost_by_draw[0] - 1) * 100:.2f}\%"
    #       f". When both HRH and consumable availability are expanded together, consumable costs increase by "
    #       f"{(consumable_cost_by_draw[8]/consumable_cost_by_draw[0] - 1) * 100:.2f}\% "
    #       f"compared to the actual scenario.")
    # # Results 4
    # cost_of_hiv_testing =  input_costs[(input_costs.cost_subgroup == 'Test, HIV EIA Elisa') & (input_costs.stat == 'mean')].groupby(['draw'])['cost'].sum()
    # print(f"For instance, the cost of HIV testing consumables increases by {(cost_of_hiv_testing[3]/cost_of_hiv_testing[0] - 1)*100:.2f}\% under the expanded HRH scenario and by "
    #       f"{(cost_of_hiv_testing[8]/cost_of_hiv_testing[0] - 1)*100:.2f}\% under the combined expanded HRH and improved consumable availability scenario, "
    #       f"while showing almost no change under the scenario with improved consumable availability alone")

    # Get figures for overview paper
    # -----------------------------------------------------------------------------------------------------------------------
    # Figure 2: Estimated costs by cost category
    # do_stacked_bar_plot_of_cost_by_category(_df = input_costs, _cost_category = 'all', _disaggregate_by_subgroup = False,
    #                                         _year = list_of_relevant_years_for_costing,show_title = False,
    #                                         _outputfilepath = figurespath, _scenario_dict = cost_scenarios)

    revised_consumable_subcategories = {'cost_of_separately_managed_medical_supplies_dispensed':'cost_of_consumables_dispensed', 'cost_of_excess_separately_managed_medical_supplies_stocked': 'cost_of_excess_consumables_stocked', 'supply_chain':'supply_chain'}
    input_costs_new = input_costs.copy()
    input_costs_new['cost_subcategory'] = input_costs_new['cost_subcategory'].map(revised_consumable_subcategories).fillna(input_costs_new['cost_subcategory'])

    # Figure 3: Estimated costs by cost sub-category
    output_costs_medical = do_stacked_bar_plot_of_cost_by_category(_df = input_costs_new, _cost_category = 'medical consumables', _disaggregate_by_subgroup = False,
                                            _year = list_of_years_for_plot, show_title = False,
                                            _outputfilepath = figurespath, _scenario_dict = cost_scenarios, _add_figname_suffix=scen_timestamps_suffix)
    output_costs_medical_file_path = cost_outcome_folderpath / f"output_costs_medical_outcomes_{SQ_timestamp}.pkl"
    if not output_costs_medical_file_path.exists():
        print("saving output cost medical outcomes to file ...")
        col_names = ['total', 'lower_bound', 'upper_bound']
        output_costs_medical_df = pd.DataFrame({name: t for name, t in zip(col_names, output_costs_medical)})
        output_costs_medical_df = output_costs_medical_df * 10 ** 6
        output_costs_medical_df['interv'] = output_costs_medical_df.index.map(cost_scenarios)
        output_costs_medical_df = output_costs_medical_df.set_index('interv')
        output_costs_medical_df.to_pickle(output_costs_medical_file_path)
    # do_stacked_bar_plot_of_cost_by_category(_df = input_costs, _cost_category = 'human resources for health', _disaggregate_by_subgroup = False,
    #                                         _year = list_of_years_for_plot, show_title = False,
    #                                         _outputfilepath = figurespath, _scenario_dict = cost_scenarios)
    # do_stacked_bar_plot_of_cost_by_category(_df = input_costs, _cost_category = 'medical equipment', _disaggregate_by_subgroup = False,
    #                                         _year = list_of_years_for_plot, show_title = False,
    #                                         _outputfilepath = figurespath, _scenario_dict = cost_scenarios)
    # do_stacked_bar_plot_of_cost_by_category(_df = input_costs, _cost_category = 'facility operating cost', _disaggregate_by_subgroup = False,
    #                                         _year = list_of_years_for_plot, show_title = False,
    #                                         _outputfilepath = figurespath, _scenario_dict = cost_scenarios)


    # # Figure 4: Estimated costs by year
    # do_line_plot_of_cost(_df = input_costs_undiscounted, _cost_category='all',
    #                          _year=list_of_years_for_plot, _draws= [0],
    #                          disaggregate_by= 'cost_category',
    #                          _y_lim = 400,
    #                          show_title = False,
    #                          _outputfilepath = figurespath)
    # do_line_plot_of_cost(_df = input_costs_undiscounted, _cost_category='all',
    #                          _year=list_of_years_for_plot, _draws= [3],
    #                          disaggregate_by= 'cost_category',
    #                          _y_lim = 400,
    #                          show_title = False,
    #                          _outputfilepath = figurespath)
    # do_line_plot_of_cost(_df = input_costs_undiscounted, _cost_category='all',
    #                          _year=list_of_years_for_plot, _draws= [5],
    #                          disaggregate_by= 'cost_category',
    #                          _y_lim = 400,
    #                          show_title = False,
    #                          _outputfilepath = figurespath)
    # do_line_plot_of_cost(_df = input_costs_undiscounted, _cost_category='all',
    #                          _year=list_of_years_for_plot, _draws= [8],
    #                          disaggregate_by= 'cost_category',
    #                          _y_lim = 400,
    #                          show_title = False,
    #                          _outputfilepath = figurespath)

    # # Figure D1: Total cost by scenario assuming 0% discount rate
    # do_stacked_bar_plot_of_cost_by_category(_df = input_costs_undiscounted,
    #                                         _cost_category = 'all',
    #                                         _year=list_of_years_for_plot,
    #                                         _disaggregate_by_subgroup = False,
    #                                         _outputfilepath = figurespath,
    #                                         _scenario_dict = cost_scenarios,
    #                                         _add_figname_suffix = '_UNDISCOUNTED')
    #
    # # Figure D2: Total cost by scenario assuming variable discount rates
    # do_stacked_bar_plot_of_cost_by_category(_df = input_costs_variable_discounting,
    #                                         _cost_category = 'all',
    #                                         _year=list_of_years_for_plot,
    #                                         _disaggregate_by_subgroup = False,
    #                                         _outputfilepath = figurespath,
    #                                         _scenario_dict = cost_scenarios,
    #                                         _add_figname_suffix = '_VARIABLE_DISCOUNTING')


    # Figure F1-F4: Cost by cost sub-group
    #TODO: this might be useful

    # cost_categories = ['human resources for health', 'medical consumables',
    #        'medical equipment', 'facility operating cost']
    # draws = input_costs.draw.unique().tolist()
    # colourmap_for_consumables = {'First-line ART regimen: adult':'#1f77b4',
    #                              'Test, HIV EIA Elisa': '#ff7f0e',
    #                              'VL Test': '#2ca02c',
    #                              'Depot-Medroxyprogesterone Acetate 150 mg - 3 monthly': '#d62728',
    #                              'Oxygen, 1000 liters, primarily with oxygen cylinders': '#9467bd',
    #                              'Phenobarbital, 100 mg': '#8c564b',
    #                              'Rotavirus vaccine': '#e377c2',
    #                              'Carbamazepine 200mg_1000_CMST': '#7f7f7f',
    #                              'Infant resuscitator, clear plastic + mask + bag_each_CMST': '#bcbd22',
    #                              'Dietary supplements (country-specific)': '#17becf',
    #                              'Tenofovir (TDF)/Emtricitabine (FTC), tablet, 300/200 mg': '#2b8cbe',
    #                              'Pneumococcal vaccine': '#fdae61',
    #                              'Pentavalent vaccine (DPT, Hep B, Hib)': '#d73027',
    #                              'male circumcision kit, consumables (10 procedures)_1_IDA': '#756bb1',
    #                              'Jadelle (implant), box of 2_CMST': '#ffdd44',
    #                              'Urine analysis': '#66c2a5'}

    # for _cat in cost_categories:
    #     for _d in draws:
    #         if _cat == 'medical consumables':
    #             create_summary_treemap_by_cost_subgroup(_df = input_costs, _year = list_of_years_for_plot,
    #                                                _cost_category = _cat, _draw = _d, _color_map=colourmap_for_consumables,
    #                                                 show_title= False, _label_fontsize= 8, _outputfilepath=figurespath)
    #         else:
    #             create_summary_treemap_by_cost_subgroup(_df=input_costs, _year=list_of_years_for_plot,
    #                                                     _cost_category=_cat, _draw=_d, show_title= False,
    #                                                     _label_fontsize= 8.5, _outputfilepath=figurespath)


    # # Get tables for overview paper
    # # -----------------------------------------------------------------------------------------------------------------------
    # # Group data and aggregate cost for each draw and stat
    # def generate_detail_cost_table(_groupby_var, _groupby_var_name, _longtable = False):
    #     edited_input_costs = input_costs.copy()
    #     edited_input_costs[_groupby_var] = edited_input_costs[_groupby_var].replace('_', ' ', regex=True)
    #     edited_input_costs[_groupby_var] = edited_input_costs[_groupby_var].replace('%', '\%', regex=True)
    #     edited_input_costs[_groupby_var] = edited_input_costs[_groupby_var].replace('&', '\&', regex=True)
    #
    #     grouped_costs = edited_input_costs.groupby(['cost_category', _groupby_var, 'draw', 'stat'])['cost'].sum()
    #     # Format the 'cost' values before creating the LaTeX table
    #     grouped_costs = grouped_costs.apply(lambda x: f"{float(x):,.0f}")
    #     # Remove underscores from all column values
    #
    #     # Create a pivot table to restructure the data for LaTeX output
    #     pivot_data = {}
    #     for draw in cost_scenarios_draw_nmbs:
    #         draw_data = grouped_costs.xs(draw, level='draw').unstack(fill_value=0)  # Unstack to get 'stat' as columns
    #         # Concatenate 'mean' with 'lower-upper' in the required format
    #         pivot_data[draw] = draw_data['mean'].astype(str) + ' [' + \
    #                            draw_data['lower'].astype(str) + '-' + \
    #                            draw_data['upper'].astype(str) + ']'
    #
    #     # Combine draw data into a single DataFrame
    #     table_data = pd.concat([pivot_data[0], pivot_data[3], pivot_data[5], pivot_data[8]], axis=1, keys=['draw=0', 'draw=3', 'draw=5', 'draw=8']).reset_index()
    #
    #     # Rename columns for clarity
    #     table_data.columns = ['Cost Category', _groupby_var_name, 'Actual', 'Expanded HRH', 'Improved consumable availability', 'Expanded HRH +\n Improved consumable availability']
    #
    #     # Replace '\n' with '\\' for LaTeX line breaks
    #     #table_data['Real World'] = table_data['Real World'].apply(lambda x: x.replace("\n", "\\\\"))
    #     #table_data['Perfect Health System'] = table_data['Perfect Health System'].apply(lambda x: x.replace("\n", "\\\\"))
    #
    #     # Convert to LaTeX format with horizontal lines after every row
    #     latex_table = table_data.to_latex(
    #         longtable=_longtable,  # Use the longtable environment for large tables
    #         column_format='|R{3cm}|R{3cm}|R{2.2cm}|R{2.2cm}|R{2.2cm}|R{2.2cm}|',
    #         caption=f"Summarized Costs by Category and {_groupby_var_name}",
    #         label=f"tab:cost_by_{_groupby_var}",
    #         position="h",
    #         index=False,
    #         escape=False,  # Prevent escaping special characters like \n
    #         header=True
    #     )
    #
    #     # Add \hline after the header and after every row for horizontal lines
    #     latex_table = latex_table.replace("\\\\", "\\\\ \\hline")  # Add \hline after each row
    #     #latex_table = latex_table.replace("_", " ")  # Add \hline after each row
    #
    #     # Specify the file path to save
    #     latex_file_path = figurespath / f'cost_by_{_groupby_var}.tex'
    #
    #     # Write to a file
    #     with open(latex_file_path, 'w') as latex_file:
    #         latex_file.write(latex_table)
    #
    #     # Print latex for reference
    #     print(latex_table)

    # # Table F1: Cost by cost subcategory
    # generate_detail_cost_table(_groupby_var = 'cost_subcategory', _groupby_var_name = 'Cost Subcategory', _longtable = True)
    # # Table F2: Cost by cost subgroup
    # generate_detail_cost_table(_groupby_var = 'cost_subgroup', _groupby_var_name = 'Category Subgroup', _longtable = True)

    # # Figure E1: Consumable inflow to outflow ratio figure
    # # -----------------------------------------------------------------------------------------------------------------------
    # inflow_to_outflow_ratio = pd.read_csv(resourcefilepath / "costing/ResourceFile_Consumables_Inflow_Outflow_Ratio.csv")
    #
    # # Clean category names for plot
    # clean_category_names = {'cancer': 'Cancer', 'cardiometabolicdisorders': 'Cardiometabolic Disorders',
    #                         'contraception': 'Contraception', 'general': 'General', 'hiv': 'HIV', 'malaria': 'Malaria',
    #                         'ncds': 'Non-communicable Diseases', 'neonatal_health': 'Neonatal Health',
    #                         'other_childhood_illnesses': 'Other Childhood Illnesses', 'reproductive_health': 'Reproductive Health',
    #                         'road_traffic_injuries': 'Road Traffic Injuries', 'tb': 'Tuberculosis',
    #                         'undernutrition': 'Undernutrition'}
    # inflow_to_outflow_ratio['category'] = inflow_to_outflow_ratio['item_category'].map(clean_category_names)
    #
    #
    # def plot_inflow_to_outflow_ratio(_df, groupby_var, _outputfilepath):
    #     # Plot the bar plot with gray bars
    #     plt.figure(figsize=(10, 6))
    #     sns.barplot(data=_df, x=groupby_var, y='inflow_to_outflow_ratio', errorbar=None, color="gray")
    #
    #     # Add points representing the distribution of individual values
    #     sns.stripplot(data=_df, x=groupby_var, y='inflow_to_outflow_ratio', color='black', size=5, alpha=0.2)
    #
    #     # Wrap x-axis labels ONLY if they are strings and longer than 15 characters
    #     labels = []
    #     for label in _df[groupby_var].unique():
    #         if isinstance(label, str) and len(label) > 15:
    #             labels.append(textwrap.fill(label, width=15))
    #         else:
    #             labels.append(label)
    #     plt.xticks(ticks=range(len(labels)), labels=labels, rotation=90, ha='center')
    #
    #     # Set labels and title
    #     plt.xlabel(groupby_var)
    #     plt.ylabel('Inflow to Outflow Ratio')
    #
    #     # Show and save plot
    #     plt.tight_layout()
    #     plt.savefig(_outputfilepath / f'inflow_to_outflow_ratio_by_{groupby_var}.png')
    #     plt.close()
    #
    # plot_inflow_to_outflow_ratio(inflow_to_outflow_ratio, 'fac_type_tlo', _outputfilepath = figurespath)
    # plot_inflow_to_outflow_ratio(inflow_to_outflow_ratio, 'district', _outputfilepath = figurespath)
    # plot_inflow_to_outflow_ratio(inflow_to_outflow_ratio, 'item_code', _outputfilepath = figurespath)
    # plot_inflow_to_outflow_ratio(inflow_to_outflow_ratio, 'category', _outputfilepath = figurespath)

    total_time_end = time.time()
    print(f"\ntotal running time (s) of cost calculations: {(total_time_end - total_time_start)}")
