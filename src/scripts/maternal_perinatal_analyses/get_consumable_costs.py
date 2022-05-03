from pathlib import Path
import pandas as pd
import time
from tlo.analysis.utils import load_pickled_dataframes, get_scenario_outputs, extract_results

start_time = time.time()

outputspath = './outputs/sejjj49@ucl.ac.uk/'

bl_scenario_filename = 'baseline_sba_scenario'
int_scenario_filename = 'bemonc'

baseline_results_folder = get_scenario_outputs(bl_scenario_filename, outputspath)[-1]
intervention_results_folder = get_scenario_outputs(int_scenario_filename, outputspath)[-1]

intervention_years = list(range(2020, 2031))
runs = list(range(0, 4))

resourcefilepath = Path("./resources/healthsystem/consumables/")
consumables_df = pd.read_csv(Path(resourcefilepath) / 'ResourceFile_Consumables_Items_and_Packages.csv')


def get_cons_cost_per_year(results_folder, hsi_of_interest):
    # Create df storing to store total cost per year per run
    total_cost_per_draw_per_year = pd.DataFrame(columns=[runs], index=[intervention_years])

    # Loop over each run
    for run in runs:
        # Load run log
        run_df = load_pickled_dataframes(results_folder, draw=0, run=run)

        # Select the appropriate rows relating to HSIs of interest from main log
        cons = run_df['tlo.methods.healthsystem']['Consumables']
        cons['year'] = cons['date'].dt.year
        anc_cons = cons.loc[(cons.TREATMENT_ID.str.contains(hsi_of_interest)) &
                            (cons.year >= intervention_years[0])]

        # Create df that will store cost associated with each requested consumable for this run
        cons_df_for_this_draw = pd.DataFrame(index=[intervention_years])

        # Loop over each year
        for year in intervention_years:
            # Select the year of interest and extract dicts
            year_anc_cons_eval = anc_cons.loc[anc_cons.year == year]['Item_Available'].apply(lambda x: eval(x))

            # For each row (hsi) in that year we unpack the dictionary
            for row in year_anc_cons_eval.index:

                # Use the key of each dict (the item code) to create a column in a dataframe and total the number of
                # times that consumable was successfully requested
                for k in year_anc_cons_eval.at[row]:
                    if k in cons_df_for_this_draw.columns:
                        cons_df_for_this_draw.at[year, k] += year_anc_cons_eval.at[row][k]
                    elif k not in cons_df_for_this_draw.columns:
                        cons_df_for_this_draw[k] = year_anc_cons_eval.at[row][k]

        # Then with the final dataframe for the run we multiply the total number of each consumable requested by the
        # cost (in kwacha) and convert to USD
        for row in cons_df_for_this_draw.index:
            for column in cons_df_for_this_draw.columns:

                # TODO: some consumables do not have a unit cost- raise with sakshi?
                if consumables_df[consumables_df.Item_Code == column]['Unit_Cost'].iloc[0] == 0:
                    print(f'item code {column} does not have an associated cost stored in the consumables dataframe')

                cons_df_for_this_draw.at[row, column] = \
                    (cons_df_for_this_draw.at[row, column] *
                     (consumables_df[consumables_df.Item_Code == column]['Unit_Cost'].iloc[0]))
                cons_df_for_this_draw.at[row, column] = cons_df_for_this_draw.at[row, column] * 0.0012

        for index in total_cost_per_draw_per_year.index:
            total_cost_per_draw_per_year.at[index, run] = cons_df_for_this_draw.loc[index].sum()

    # return the df to be used for outputs
    return total_cost_per_draw_per_year


baseline_cost_data = get_cons_cost_per_year(baseline_results_folder, 'AntenatalCare')
intervention_cost_data = get_cons_cost_per_year(intervention_results_folder, 'AntenatalCare')

end_time = time.time()
print("The time of execution of above program is :", end_time-start_time)
