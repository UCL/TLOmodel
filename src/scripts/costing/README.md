# Cost Estimation Utility Functions (`cost_estimation.py`)

## Overview

This script provides utlity functions to estimate the **cost of healthcare delivery**. using simulation outputs from the **Thanzi La Onse (TLO)** model. It supports analysis of cost drivers, scenario comparisons, and ROI estimation.
Full description of the method is available in the preprint - [Method for costing a health system using a Health Systems Model](https://doi.org/10.1101/2025.01.22.25320881)

Cost components include:
- 🧑‍⚕️ Human Resources for Health
- 💊 Medical Consumables
- 🏥 Medical Equipment
- 🧾 Facility Operating Costs

It also includes utilities for:
- 💸 Discounting and summarizing costs
- 📊 Generating plots (stacked bars, line charts, treemaps)
- 💹 ROI estimation following the method specified in [Reference Case Guidelines for Benefit-Cost Analysis in Global Health and Development](https://doi.org/10.2139/ssrn.4015886)
- 📉 Projecting health spending based on  [Dieleman et al.(2019)](https://pubmed.ncbi.nlm.nih.gov/29678341/)
---

## Key Functions

| Function | Description                                                                                                                                 |
|----------|---------------------------------------------------------------------------------------------------------------------------------------------|
| `estimate_input_cost_of_scenarios()` | Main function to estimate and optionally summarize costs by draw, year, and cost component                                                  |
| `estimate_projected_health_spending()` | Projects total health spending based on per capita spending projections from Dieleman et al (2019) and population estimates from simulation |
| `summarize_cost_data()` | Aggregates cost data across runs - mean/median and confidence intervals                                                                     |
| `apply_discounting_to_cost_data()` | Applies a fixed or time-varying discount rate to cost columns (also integrated within _estimate_input_cost_of_scenarios()_)                 |
| `do_stacked_bar_plot_of_cost_by_category()` | Generates stacked bar charts of cost composition                                                                                            |
| `do_line_plot_of_cost()` | Plots cost trends over time, optionally disaggregated                                                                                       |
| `create_summary_treemap_by_cost_subgroup()` | Creates a treemap showing cost subgroups                                                                                                    |
| `generate_multiple_scenarios_roi_plot()` | Compares ROI curves across multiple scenarios over different hypothetical above service level costs                                         |

---

## Required Resource Files

To run the model, the following resource files must be placed in the correct folder structure under `resourcefilepath`:

### `/costing/`

| File | Purpose                                                                                                        |
|------|----------------------------------------------------------------------------------------------------------------|
| `ResourceFile_Costing_HR.csv` | Salary, training, and supervision cost parameters for health worker cadres                                     |
| `ResourceFile_Costing_Consumables.csv` | Unit costs for consumables used in the model                                                                   |
| `ResourceFile_Resource_Mapping.csv` | 'Actual' budget (2019/20 - 2020/21) and expenditure (2018/19) recorded as per Resource Mapping Round 5         |
| `ResourceFile_Consumables_Inflow_Outflow_Ratio.csv` | Provides the ratio of consumable inflows to outflows to help account for wastage; disaggregated by `Item_Code` |
| `ResourceFile_Costing_Facility_Operations.csv` | Facility operating costs (utilities, food, cleaning, etc.)                                                     |
| `ResourceFile_Health_Spending_Projections.csv` | Per capita spending projections from   [Dieleman et al.(2019)](https://pubmed.ncbi.nlm.nih.gov/29678341/)      |
| `ResourceFile_Resource_Mapping.csv` | 'Actual' budget (2019/20 - 2020/21) and expenditure (2018/19) recorded as per Resource Mapping Round 5         |

### `/healthsystem/consumables/`
| File | Purpose                                                                                                        |
|------|----------------------------------------------------------------------------------------------------------------|
| `ResourceFile_Consumables_Items_and_Packages.csv` | Mapping of `Item_Code` for consumables to consumable names to facilitate interpretation of cost estimate                                |

### `/healthsystem/organisation/`

| File | Purpose                                         |
|------|-------------------------------------------------|
| `ResourceFile_Master_Facilities_List.csv` | Mapping of facility IDs to levels and districts |

### `/demography/`

| File | Purpose                            |
|------|------------------------------------|
| `ResourceFile_Population_2010.csv` | Actual 2010 population by district |

---

## Adapting to Other Country Settings

To apply this module in a new country:

1. 🔁 Replace all cost files in `/costing/` with local unit costs and assumptions.
2. 🏥 Update the Master Facilities List to reflect your country’s health facility types and levels.
3. 🗺 Update population/district files to match local administrative units.
4. 🛠 Modify functions such as `get_staff_count_by_facid_and_officer_type()`, `update_itemuse_for_level1b_using_level2_data()` functions if model assumptions (e.g., facility level) differ.
5. 🛠 Fix other hard coded assumptions -
   1. The function `update_itemuse_for_level1b_using_level2_data()` is used to deal with the idiosyncrasy of the current model framework which treats levels 1b and 2 as identical.
   2. The line `available_staff_count_by_level_and_officer_type = available_staff_count_by_level_and_officer_type.drop(available_staff_count_by_level_and_officer_type[available_staff_count_by_level_and_officer_type['Facility_Level'] == '5'].index)` drops headquarters from the calculation of HR costs
   3. For equipment costing, the formulae for estimating `service_fee_annual`, `spare_parts_annual` and `major_corrective_maintenance_cost_annual` are based on Malawi HSSP-III assumptions

---

## Outputs

- 📑 A DataFrame of costs by year, draw, and cost category (further disaggregated into category-specific components such as cadre for HR)
- 📈 Plots : stacked bars, line plots, treemaps
- 💹 ROI charts for scenario comparisons

---

## Other Scripts / Example Usage

### `scripts/costing/`
| Script                                                                                             | Purpose                                                                                                                                                                             |
|----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `cost_overview_analysis.py`   | Generates cost estimates and plots for the manuscript [Method for costing a health system using a Health Systems Model](https://doi.org/10.1101/2025.01.22.25320881)                |
| `costing_validation.py`  | Compares cost estimates in the manuscript with Resource Mapping Data [Method for costing a health system using a Health Systems Model](https://doi.org/10.1101/2025.01.22.25320881) |

### `scripts/comparison_of_horizontal_and_vertical_programs/`
| Script                                                                                             | Purpose                                                                                                                                                                             |
|----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `roi_analysis_horizontal_vs_vertical.py` | Generates cost, ICERs and ROI estimates and plots for the manuscript [System-Wide Investments Enhance HIV, TB and Malaria Control in Malawi and Deliver Greater Health Impact](https://doi.org/10.1101/2025.04.29.25326667)    |

## Potential Improvements
See [GitHub Issue 1635](https://github.com/UCL/TLOmodel/issues/1635)

## Contact

For questions, reach out to Sakshi Mohan
