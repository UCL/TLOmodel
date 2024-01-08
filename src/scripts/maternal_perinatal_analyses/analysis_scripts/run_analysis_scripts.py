import maternal_newborn_health_thesis_analysis

# create dict of some scenario 'title' and the filename of the associated title
scenario_dict1 = {'Status Quo': 'baseline_scenario',
                  'Intervention 1': 'increased_anc_scenario',
                  'Intervention 2': 'anc_scenario_plus_cons_and_qual',
                  'Sensitivity 1': 'min_anc_sensitivity_scenario',
                  'Sensitivity 2': 'max_anc_sensitivity_scenario'}

scenario_dict2 = {'Status Quo': 'baseline_scenario',
                  'Intervention 1': 'bemonc',
                  'Intervention 2': 'cemonc',
                  'Sensitivity 1': 'min_sba_sensitivity_analysis',
                  'Sensitivity 2': 'max_sba_sensitivity_analysis'}

scenario_dict3 = {'Status Quo': 'baseline_scenario',
                  'Intervention 1': 'increased_pnc_scenario',
                  'Intervention 2': 'pnc_scenario_plus_cons',
                  'Sensitivity 1': 'min_pnc_sensitivity_analysis',
                  'Sensitivity 2': 'max_pnc_sensitivity_analysis'}


# define key variables used within the analysis scripts
intervention_years = list(range(2023, 2031))
sim_years = list(range(2012, 2031))
output_path = './outputs/sejjj49@ucl.ac.uk/'


for scenario_dict, service, colours in zip([scenario_dict1, scenario_dict2, scenario_dict3],
                                           ['anc', 'sba', 'pnc'],
                                           [['#8c510a', '#d73027', '#8073ac', '#01665e', '#c51b7d'],
                                            ['#8c510a', '#de77ae', '#5aae61', '#e08214',  '#4575b4'],
                                            ['#8c510a', '#dfc27dtg', '#1b7837', '#f46d43', '#80cdc1']]):

    service_of_interest = service
    scen_colours = colours

    maternal_newborn_health_thesis_analysis.run_maternal_newborn_health_thesis_analysis(
         scenario_file_dict=scenario_dict,
         outputspath=output_path,
         sim_years=sim_years,
         intervention_years=intervention_years,
         service_of_interest=service_of_interest,
         scen_colours=scen_colours)
