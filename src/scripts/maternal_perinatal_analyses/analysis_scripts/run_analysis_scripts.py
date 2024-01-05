import maternal_newborn_health_thesis_analysis

# create dict of some scenario 'title' and the filename of the associated title
scenario_dict1 = {'Status Quo': 'baseline_scenario',
                  'Intervention 1': 'increased_anc_scenario',
                  'Intervention 2': 'anc_scenario_plus_cons_and_qual',
                  'Sensitivity (min)': 'min_anc_sensitivity_scenario',
                  'Sensitivity (max)': 'max_anc_sensitivity_scenario'}

scenario_dict2 = {'Status Quo': 'baseline_scenario',
                  'Intervention 1': 'bemonc',
                  'Intervention 2': 'cemonc',
                  'Sensitivity (min)': 'min_sba_sensitivity_analysis',
                  'Sensitivity (max)': 'max_sba_sensitivity_analysis'}

scenario_dict3 = {'Status Quo': 'baseline_scenario',
                  'Intervention 1': 'increased_pnc_scenario',
                  'Intervention 2': 'pnc_scenario_plus_cons',
                  'Sensitivity (min)': 'min_pnc_sensitivity_analysis',
                  'Sensitivity (max)': 'max_pnc_sensitivity_analysis'}


# define key variables used within the analysis scripts
intervention_years = list(range(2023, 2031))
sim_years = list(range(2012, 2031))
output_path = './outputs/sejjj49@ucl.ac.uk/'


for scenario_dict, service, colours in zip([scenario_dict1, scenario_dict2, scenario_dict3],
                                           ['anc', 'sba', 'pnc'],
                                           [['#a6611a', '#dfc27d', '#f5f5f5', '#80cdc1', '#018571'],
                                            ['#7b3294', '#c2a5cf', '#f7f7f7', '#a6dba0', '#008837'],
                                            ['#d7191c', '#fdae61', '#ffffbf', '#abd9e9', '#2c7bb6']]):

    service_of_interest = service
    scen_colours = colours

    maternal_newborn_health_thesis_analysis.run_maternal_newborn_health_thesis_analysis(
         scenario_file_dict=scenario_dict,
         outputspath=output_path,
         sim_years=sim_years,
         intervention_years=intervention_years,
         service_of_interest=service_of_interest,
         scen_colours=scen_colours)
