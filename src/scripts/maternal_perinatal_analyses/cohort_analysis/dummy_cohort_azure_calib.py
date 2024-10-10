import os

import pandas as pd

from tlo.analysis.utils import extract_results, get_scenario_outputs, summarize

outputspath = './outputs/sejjj49@ucl.ac.uk/'
scenario_filename = 'cohort_test-2024-10-09T130546Z'

results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]

def get_data_frames(key):
    def sort_df(_df):
        _x = _df.drop(columns=['date'], inplace=False)
        return _x.iloc[0]

    results_df = summarize (extract_results(
                results_folder,
                module="tlo.methods.pregnancy_supervisor",
                key=key,
                custom_generate_series=sort_df,
                do_scaling=False
            ))

    return results_df

results = {k:get_data_frames(k) for k in ['mat_comp_incidence', 'nb_comp_incidence', 'deaths_and_stillbirths',
                                          'service_coverage']}
