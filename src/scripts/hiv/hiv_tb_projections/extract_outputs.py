
from pathlib import Path

from tlo.analysis.utils import (
    extract_results,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)

outputspath = Path("./outputs/t.mangal@imperial.ac.uk")

# download all files (and get most recent [-1])
# results_folder = get_scenario_outputs("scenario1.py", outputspath)[-1]
results_folder = get_scenario_outputs("baseline_runs.py", outputspath)[-1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# check log format for consumables


# group each consumable by year and type
# then can use "summarize" for means
# output time-series numbers of consumables by year for each cons type
extracted = extract_results(
    results_folder,
    module="tlo.methods.healthsystem",
    key="Consumables",
    column="deviance_measure",
    custom_generate_series=(
        lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'type'])['person'].count()),
    do_scaling=False
)

# custom_generate_series = (
# lambda df: df.assign(year=pd.to_datetime(df['date']).dt.year).groupby(['year'])[
#                                  'year'].count()


# fraction of HCW time
# output fraction of time by year
capacity = extract_results(
        results_folder,
        module="tlo.methods.healthsystem",
        key="Capacity",
        column="XXXXX",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['Frac_Time_Used_Overall'].mean()
        ),
    )

model_tb_inc = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_incidence",
        column="num_new_active_tb",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)

# numbers of HSIs by type or total
# output numbers of each type of appt for years 2022-2035 with uncertainty
hsi = extract_results(
        results_folder,
        module="tlo.methods.healthsystem",
        key="HSI_Event",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['TREATMENT_ID'].count()
        ),
    )

# hsi = output['tlo.methods.healthsystem']['HSI_Event'].copy()
# hsi["date"] = pd.to_datetime(hsi["date"])
# hsi["month"] = hsi["date"].dt.month
# # Reduce TREATMENT_ID to the originating module
# hsi["Module"] = hsi["TREATMENT_ID"].str.split('_').apply(lambda x: x[0])

# # Plot the breakdown of all HSI, over all the years
# evs = hsi.groupby(by=['Module'])\
#     .size().rename(columns={0: 'count'}) * scaling_factor
# evs.plot.pie()
# plt.title(f"HSI by Module")
# #plt.savefig(make_file_name('HSI_per_module'))
# plt.show()


# ---------------------------------- PLOTS ---------------------------------------

# total numbers of HSI by year for each scenario (line plot)


# maybe leave this one
# numbers of HSI by type over 2022-2035 for each scenario (clustered bar plot)


# numbers of each appt type by year 2010-2035 (3 x heatmap)
# separate plot for each scenario


# fraction total HCW time used by year for each scenario (line plot)


# numbers of consumables used for each program by year for each scenario (2 x line plots)
