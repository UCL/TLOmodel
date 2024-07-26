# Check that all registered disease modules have the report_daly_values() function
import argparse
import os
from tlo.analysis.utils import parse_log_file
from pathlib import Path

### from healthburden model
#for module_name in self.recognised_modules_names:
#    assert getattr(self.sim.modules[module_name], 'report_daly_values', None) and \
#           callable(self.sim.modules[module_name].report_daly_values), 'A module that declares use of ' \
#                                                                       'HealthBurden module must have a ' \
#
#                                                                       'callable function "report_daly_values"'

module_prevalence_key = {'stunting':{'prevalence':["(0, 'HAZ<-3')", "(0, '-3<=HAZ<-2')", "(0, 'HAZ>=-2')",
                                                   "(1, 'HAZ<-3')", "(1, '-3<=HAZ<-2')", "(1, 'HAZ>=-2')",
                                                   "(2, 'HAZ<-3')", "(2, '-3<=HAZ<-2')", "(2, 'HAZ>=-2')",
                                                   "(3, 'HAZ<-3')", "(3, '-3<=HAZ<-2')", "(3, 'HAZ>=-2')",
                                                   "(4, 'HAZ<-3')", "(4, '-3<=HAZ<-2')", "(4, 'HAZ>=-2')"]}, # prevalence
                         'measels':['incidence'],
                         'hiv':{'infections_by_2age_groups_and_sex':['total_prev']},
                         'malaria':{'prev_district': list(map(str, range(32)))}, # district prevalences
                         'rti':{'Inj_category_incidence':['number_of_injuries']}, #number
                         'prostate_cancer':{'summary_stats': ['total_none', 'total_prostate_confined', 'total_local_ln', 'total_metastatic']}, # numbers
                         'other_adult_cancers':{'summary_stats': ['total_none', 'total_metastatic', 'total_site_confined', 'total_local_ln']}, #numebers
                         'tb': {'tb_prevalence':['tbPrevActive','tbPrevActiveAdult', 'tbPrevActiveChild', 'tbPrevLatent', 'tbPrevLatentAdult', 'tbPrevLatentChild']}
            }

# Define the function to record prevalence of diseases
#def record_prevalence_diseases(self):
#    for module in self.recognised_module_names:
        # Access the nested dictionary
#        module_dict = module_prevalence_key[module]
#        for key, values in module_dict.items():
 #           for value in values:
 #               print()
 # Define the function to record prevalence of diseases

def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    def list_disease_logs(results_folder: Path):
        registered_diseases = set()
        disease_files = []
        print(os.listdir(results_folder))

        for file in os.listdir(results_folder):
            if file.startswith('tlo.methods') and file.endswith('.log'):
                suffix = file[len('tlo.methods'):]
                print(suffix)
                if suffix in module_prevalence_key.keys():
                    print(suffix)
                    registered_diseases.add(suffix)
                    disease_files.append((suffix, file))

        return {disease: [file for ds, file in disease_files if ds == disease] for disease in registered_diseases}

    registered_disease_logs = list_disease_logs(results_folder)
    print(registered_disease_logs)

    possible_log_names = [f"tlo.methods{key}.log" for key in module_prevalence_key.keys()]

    for disease in registered_disease_logs.keys():
        # Access the nested dictionary
        module_dict = module_prevalence_key[disease]

        for key, values in module_dict.items():
            for value in values:
                results = extract_results(
                    results_folder,
                    module=registered_disease_logs[disease],
                    key=value,
                    do_scaling=True
                )
                print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources')
    )
