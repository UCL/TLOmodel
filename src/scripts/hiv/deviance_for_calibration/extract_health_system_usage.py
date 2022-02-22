
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Define paths and filenames
rfp = Path("./resources")
outputpath = Path("./outputs")
results_filename = outputpath / 'default_run.pickle'
make_file_name = lambda stub: outputpath / f"{datetime.today().strftime('%Y_%m_%d''')}_{stub}.png"

with open(results_filename, 'rb') as f:
    output = pickle.load(f)

# %% Scaling Factor
scaling_factor = output["tlo.methods.demography"]["scaling_factor"].scaling_factor.values[0]

# HEALTH CARE WORKER TIME
cap = output['tlo.methods.healthsystem']['Capacity'].copy()
cap["date"] = pd.to_datetime(cap["date"])
cap = cap.set_index('date')

frac_time_used = cap['Frac_Time_Used_Overall']
cap = cap.drop(columns=['Frac_Time_Used_Overall'])

# Plot Fraction of total time of health-care-workers being used
frac_time_used.plot()
plt.title("Fraction of total health-care worker time being used")
plt.xlabel("Date")
#plt.savefig(make_file_name('HSI_Frac_time_used'))
plt.show()


# %% Breakdowns by HSI:
hsi = output['tlo.methods.healthsystem']['HSI_Event'].copy()
hsi["date"] = pd.to_datetime(hsi["date"])
hsi["month"] = hsi["date"].dt.month
# Reduce TREATMENT_ID to the originating module
hsi["Module"] = hsi["TREATMENT_ID"].str.split('_').apply(lambda x: x[0])

# Plot the HSI that are taking place, by month, in a a particular year
year = 2012
evs = hsi.loc[hsi.date.dt.year == year]\
    .groupby(by=['month', 'Module'])\
    .size().reset_index().rename(columns={0: 'count'})\
    .pivot_table(index='month', columns='Module', values='count', fill_value=0)
evs *= scaling_factor

evs.plot.bar(stacked=True)
plt.title(f"HSI by Module, per Month (year {year})")
plt.ylabel('Total per month')
#plt.savefig(make_file_name('HSI_per_module_per_month'))
plt.show()

# Plot the breakdown of all HSI, over all the years
evs = hsi.groupby(by=['Module'])\
    .size().rename(columns={0: 'count'}) * scaling_factor
evs.plot.pie()
plt.title(f"HSI by Module")
#plt.savefig(make_file_name('HSI_per_module'))
plt.show()


# =================================================== CONSUMABLE COST =================================================
def get_mean_and_quants(df, sim_years):
    year_means = list()
    lower_quantiles = list()
    upper_quantiles = list()

    for year in sim_years:
        if year in df.index:
            year_means.append(df.loc[year].mean(axis=1).iloc[0])
            lower_quantiles.append(df.loc[year].iloc[0].quantile(0.025))
            upper_quantiles.append(df.loc[year].iloc[0].quantile(0.925))
        else:
            year_means.append(0)
            lower_quantiles.append(0)
            lower_quantiles.append(0)

    return [year_means, lower_quantiles, upper_quantiles]


draws = [0, 1, 2, 3, 4]
resourcefilepath = Path("./resources/healthsystem/consumables/")
consumables_df = pd.read_csv(Path(resourcefilepath) / 'ResourceFile_Consumables.csv')


def get_cons_cost_per_year(results_folder):
    # Create df that replicates the 'extracted' df
    total_cost_per_draw_per_year = pd.DataFrame(columns=[draws], index=[intervention_years])

    # Loop over each draw
    for draw in draws:
        # Load df, add year column and select only ANC interventions
        draw_df = load_pickled_dataframes(results_folder, draw=draw)

        cons = draw_df['tlo.methods.healthsystem']['Consumables']
        cons['year'] = cons['date'].dt.year
        total_anc_cons = cons.loc[cons.TREATMENT_ID.str.contains('AntenatalCare')]
        anc_cons = total_anc_cons.loc[total_anc_cons.year >= intervention_years[0]]

        cons_df_for_this_draw = pd.DataFrame(index=[intervention_years])

        # Loop over each year
        for year in intervention_years:
            # Select the year of interest
            year_df = anc_cons.loc[anc_cons.year == year]

            # For each row (hsi) in that year we unpack the dictionary
            for row in year_df.index:
                cons_dict = year_df.at[row, 'Item_Available']
                cons_dict = eval(cons_dict)

                # todo: check this works where there are muliple dicts
                # For each dictionary
                for k, v in cons_dict.items():
                    if k in cons_df_for_this_draw.columns:
                        cons_df_for_this_draw.at[year, k] += v
                    elif k not in cons_df_for_this_draw.columns:
                        cons_df_for_this_draw[k] = v

        for row in cons_df_for_this_draw.index:
            for column in cons_df_for_this_draw.columns:
                cons_df_for_this_draw.at[row, column] =\
                    (cons_df_for_this_draw.at[row, column] *
                     (consumables_df[consumables_df.Item_Code == 0]['Unit_Cost'].iloc[0]))
                cons_df_for_this_draw.at[row, column] = cons_df_for_this_draw.at[row, column] * 0.0014
                # todo: this is usd conversion
                # todo: account for inflation, and use 2010 rate

        for index in total_cost_per_draw_per_year.index:
            total_cost_per_draw_per_year.at[index, draw] = cons_df_for_this_draw.loc[index].sum()

    final_cost_data = get_mean_and_quants(total_cost_per_draw_per_year, intervention_years)
    return final_cost_data


baseline_cost_data = get_cons_cost_per_year(baseline_results_folder)
intervention_cost_data = get_cons_cost_per_year(intervention_results_folder)

fig, ax = plt.subplots()
ax.plot(intervention_years, baseline_cost_data[0], label="Baseline (mean)", color='deepskyblue')
ax.fill_between(intervention_years, baseline_cost_data[1], baseline_cost_data[2], color='b', alpha=.1)
ax.plot(intervention_years, intervention_cost_data[0], label="Intervention (mean)", color='olivedrab')
ax.fill_between(intervention_years, intervention_cost_data[1], intervention_cost_data[2], color='g', alpha=.1)
plt.xlabel('Year')
plt.ylabel("Total Cost (USD)")
plt.title('Total Cost Attributable To Antenatal Care Per Year (in USD) (unscaled)')
plt.legend()
# plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/COST.png')
plt.show()
