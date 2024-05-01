import maternal_newborn_health_thesis_analysis

# create dict of some scenario 'title' and the filename of the associated title
scenario_dict1 = {'Status Quo': 'baseline_scenario',
                  'AN coverage.': 'increased_anc_scenario',
                  'AN coverage\nand qual.': 'anc_scenario_plus_cons_and_qual',
                  'AN min.': 'min_anc_sensitivity_scenario',
                  'AN max.': 'max_anc_sensitivity_scenario'}

scenario_dict2 = {'Status Quo': 'baseline_scenario',
                  'IP BEmONC.': 'bemonc',
                  'IP CEmONC.': 'cemonc',
                  'IP min.': 'min_sba_sensitivity_analysis',
                  'IP max.': 'max_sba_sensitivity_analysis'}

scenario_dict3 = {'Status Quo': 'baseline_scenario',
                  'PN coverage.': 'increased_pnc_scenario',
                  'PN coverage\nand qual.': 'pnc_scenario_plus_cons',
                  'PN min.': 'min_pnc_sensitivity_analysis',
                  'PN max.': 'max_pnc_sensitivity_analysis'}

scenario_dict4 = {'Status Quo': 'baseline_scenario',
                  'All services coverage\nand qual': 'uhc_cov_qual',
                  'All services max.': 'max_uhc'}

# define key variables used within the analysis scripts
intervention_years = list(range(2023, 2031))
sim_years = list(range(2012, 2031))
output_path = './outputs/sejjj49@ucl.ac.uk/'


# for scenario_dict, service, colours in zip([scenario_dict1, scenario_dict2, scenario_dict3],
#                                            ['anc', 'sba', 'pnc'],
#                                            [['#8c510a', '#d73027', '#8073ac', '#01665e', '#c51b7d'],
#                                             ['#8c510a', '#de77ae', '#5aae61', '#e08214',  '#4575b4'],
#                                             ['#8c510a', '#dfc27d', '#1b7837', '#f46d43', '#80cdc1']]):
for scenario_dict, service, colours in zip([scenario_dict4],
                                            ['uhc'],
                                            [['#8c510a', '#d73027', '#8073ac']]):
    service_of_interest = service
    scen_colours = colours

    maternal_newborn_health_thesis_analysis.run_maternal_newborn_health_thesis_analysis(
         scenario_file_dict=scenario_dict,
         outputspath=output_path,
         sim_years=sim_years,
         intervention_years=intervention_years,
         service_of_interest=service_of_interest,
         scen_colours=scen_colours)
