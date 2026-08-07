from pathlib import Path

import os
import pandas as pd

from tlo.analysis.utils import get_scenario_outputs, extract_results
from scipy.stats import t

outputspath = './outputs/sejjj49@ucl.ac.uk/'
resourcefilepath = Path("./resources")

# create_pickles_locally(results_folder, compressed_file_name_prefix='block_intervention_big_run')

#  ======================================= DEFINE SCENARIO INFORMATION  ===============================================
scenario = 'testing_scenario_747943'
results_folder= get_scenario_outputs(scenario, outputspath)[-1]

g_path = f'{outputspath}calibration_{scenario}'

if not os.path.isdir(g_path):
        os.makedirs(f'{outputspath}calibration_{scenario}')

def summarize_confidence_intervals(results: pd.DataFrame) -> pd.DataFrame:
    """Utility function to compute summary statistics

    Finds mean value and 95% interval across the runs for each draw.
    """

    # Calculate summary statistics
    grouped = results.groupby(axis=1, by='draw', sort=False)
    mean = grouped.mean()
    sem = grouped.sem()  # Standard error of the mean

    # Calculate the critical value for a 95% confidence level
    n = grouped.size().max()  # Assuming the largest group size determines the degrees of freedom
    critical_value = t.ppf(0.975, df=n - 1)  # Two-tailed critical value

    # Compute the margin of error
    margin_of_error = critical_value * sem

    # Compute confidence intervals
    lower = mean - margin_of_error
    upper = mean + margin_of_error

    # Combine into a single DataFrame
    summary = pd.concat({'mean': mean, 'lower': lower, 'upper': upper}, axis=1)

    # Format the DataFrame as in the original code
    summary.columns = summary.columns.swaplevel(1, 0)
    summary.columns.names = ['draw', 'stat']
    summary = summary.sort_index(axis=1)

    return summary


def get_ps_data_frames(key, results_folder):
    def sort_df(_df):
        _x = _df.drop(columns=['date'], inplace=False)
        return _x.iloc[0]

    results_df = extract_results(
                results_folder,
                module="tlo.methods.pregnancy_supervisor",
                key=key,
                custom_generate_series=sort_df,
                do_scaling=False
            )

    results_df_summ = summarize_confidence_intervals(results_df)

    return {'crude':results_df, 'summarised':results_df_summ}

#  ========================================== EXTRACT CORE DATA  =====================================================
results = {k:get_ps_data_frames(k, results_folder) for k in
           ['mat_comp_incidence', 'nb_comp_incidence', 'deaths_and_stillbirths','service_coverage', 'met_need',
            'yearly_mnh_counter_dict', 'intervention_coverage']}
