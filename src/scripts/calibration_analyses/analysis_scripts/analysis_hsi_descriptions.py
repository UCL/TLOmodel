# todo - describe the HSI that are actually being done (desreiptions and frequency)
#
# """Produce plots to show the usage of the healthcare system when 'Everything' is in service_availability.
# This uses the file that is created by: run_healthsystem_analysis_and_pickle_log
# """
#
# import pickle
# from datetime import datetime
# from pathlib import Path
#
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
#
# from tlo.methods.demography import get_scaling_factor
#
# # Define paths and filenames
# rfp = Path("./resources")
# outputpath = Path("./outputs")  # folder for convenience of storing outputs
# results_filename = outputpath / '2020_11_23_health_system_systematic_run.pickle'
# make_file_name = lambda stub: outputpath / f"{datetime.today().strftime('%Y_%m_%d''')}_{stub}.png"
#
# with open(results_filename, 'rb') as f:
#     output = pickle.load(f)['results']['Everything']
#
# # %% Scaling Factor
# scaling_factor = get_scaling_factor(output, rfp)
#
# # %% Show overall usage of the healthsystem:
#
# cap = output['tlo.methods.healthsystem']['Capacity'].copy()
# cap["date"] = pd.to_datetime(cap["date"])
# cap = cap.set_index('date')
#
# frac_time_used = cap['Frac_Time_Used_Overall']
# cap = cap.drop(columns=['Frac_Time_Used_Overall'])
#
# # Plot Fraction of total time of health-care-workers being used
# frac_time_used.plot()
# plt.title("Fraction of total health-care worker time being used")
# plt.xlabel("Date")
# plt.savefig(make_file_name('HSI_Frac_time_used'))
# plt.show()
#
# # %% Breakdowns by HSI:
# hsi = output['tlo.methods.healthsystem']['HSI_Event'].copy()
# hsi["date"] = pd.to_datetime(hsi["date"])
# hsi["month"] = hsi["date"].dt.month
# # Reduce TREATMENT_ID to the originating module
# hsi["Module"] = hsi["TREATMENT_ID"].str.split('_').apply(lambda x: x[0])
#
# # Plot the HSI that are taking place, by month, in a a particular year
# year = 2012
# evs = hsi.loc[hsi.date.dt.year == year]\
#     .groupby(by=['month', 'Module'])\
#     .size().reset_index().rename(columns={0: 'count'})\
#     .pivot_table(index='month', columns='Module', values='count', fill_value=0)
# evs *= scaling_factor
#
# evs.plot.bar(stacked=True)
# plt.title(f"HSI by Module, per Month (year {year})")
# plt.ylabel('Total per month')
# plt.savefig(make_file_name('HSI_per_module_per_month'))
# plt.show()
#
# # Plot the breakdown of all HSI, over all the years
# evs = hsi.groupby(by=['Module'])\
#     .size().rename(columns={0: 'count'}) * scaling_factor
# evs.plot.pie()
# plt.title(f"HSI by Module")
# plt.savefig(make_file_name('HSI_per_module'))
# plt.show()
