import compare_incidence_rates_between_scenarios_v2
import maternal_newborn_health_analysis_v2
import met_need

# create dict of some scenario 'title' and the filename of the associated title
scenario_dict1 = {'Status Quo': 'baseline_scenario',
                  'UHC (90%)': 'uhc_maternity_services',

                 }

scenario_dict2 = {'Status Quo': 'baseline_scenario',
                  'Intervention 1': 'bemonc',
                  'Intervention 2': 'cemonc',
                  'Sensitivity (min)': 'min_sba_sensitivity_analysis',
                  'Sensitivity (max)': 'max_sba_sensitivity_analysis',
                 }

scenario_dict3 = {'Status Quo': 'baseline_scenario',
                  'Intervention 1': 'increased_pnc_scenario',
                  'Intervention 2': 'pnc_scenario_plus_cons',
                  'Sensitivity (min)': 'min_pnc_sensitivity_analysis',
                  'Sensitivity (max)': 'max_pnc_sensitivity_analysis',
                 }


# define key variables used within the analysis scripts
intervention_years = list(range(2023, 2031))
sim_years = list(range(2010, 2031))
output_path = './outputs/sejjj49@ucl.ac.uk/'

for scenario_dict, service, colours in zip([scenario_dict1],
                                           ['UHC'],
                                           [ ['plum', 'purple', 'deeppink', 'crimson', 'maroon']]):

    service_of_interest = service
    scen_colours = colours

    maternal_newborn_health_analysis_v2.run_maternal_newborn_health_analysis(
         scenario_file_dict=scenario_dict,
         outputspath=output_path,
         sim_years=sim_years,
         intervention_years=intervention_years,
         service_of_interest=service_of_interest,
         show_all_results=True,
         scen_colours=scen_colours)


for scenario_dict, service, colours in zip([scenario_dict1],
                                           ['anc', 'sba', 'pnc'],
                                           [['lightcoral', 'firebrick', 'red', 'chocolate', 'darkorange'],
                                            ['cadetblue', 'deepskyblue', 'midnightblue', 'green', 'palegreen'],
                                            ['plum', 'purple', 'deeppink', 'crimson', 'maroon']]):

    service_of_interest = service
    scen_colours = colours

    maternal_newborn_health_analysis_v2.run_maternal_newborn_health_analysis(
         scenario_file_dict=scenario_dict,
         outputspath=output_path,
         sim_years=sim_years,
         intervention_years=intervention_years,
         service_of_interest=service_of_interest,
         show_all_results=True,
         scen_colours=scen_colours)

    compare_incidence_rates_between_scenarios_v2.compare_key_rates_between_multiple_scenarios(
        scenario_file_dict=scenario_dict,
        outputspath=output_path,
        sim_years=sim_years,
        intervention_years=intervention_years,
        service_of_interest=service_of_interest,
        scen_colours=scen_colours)

    met_need.met_need_and_contributing_factors_for_deaths(scenario_dict, output_path, intervention_years,
                                                          service_of_interest)
