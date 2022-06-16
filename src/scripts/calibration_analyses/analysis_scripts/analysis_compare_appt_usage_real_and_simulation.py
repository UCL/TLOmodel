"""
Compare appointment usage from model output with real appointment usage.

The real appointment usage is collected from DHIS2 system and HIV Dept.

N.B. This script uses the package `squarify`: so run, `pip install squarify` first.
"""

from pathlib import Path

from tlo.analysis.utils import get_scenario_outputs

# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import calendar

sns.set_theme(style="darkgrid")

# path of resource files: real appt usage and mfl (facility id, level, district)
rfp = Path('./resources/healthsystem')

# real usage data
real_usage = pd.read_csv(rfp / 'real_appt_usage_data' / 'real_monthly_usage_of_appt_type.csv')
real_usage_TB = pd.read_csv(rfp / 'real_appt_usage_data' / 'real_yearly_usage_of_TBNotifiedAll.csv')
# for TB usage, drop years outside of 2017-2019 according to data consistency and pandemic
real_usage_TB = real_usage_TB[real_usage_TB['Year'].isin([2017, 2018, 2019])].copy()

# TLO simulation usage path
# the name of the file that specified the scenarios used in this run.
scenario_filename = 'long_run_all_diseases.py'
# path of model output
model_output_path = Path('./outputs/bshe@ic.ac.uk')
# the results folder for the most recent run generated using that scenario_filename
results_folder = get_scenario_outputs(scenario_filename, model_output_path)[-1]

# the simulation data
simulation_usage = pd.read_csv(results_folder / 'Simulated appt usage between 2015 and 2019.csv')
# rename some appts to be compared with real usage
appt_dict = {'Under5OPD': 'OPD',
             'Over5OPD': 'OPD',
             'AntenatalFirst': 'AntenatalTotal',
             'ANCSubsequent': 'AntenatalTotal',
             'NormalDelivery': 'Delivery',
             'CompDelivery': 'Delivery',
             'EstMedCom': 'EstAdult',
             'EstNonCom': 'EstAdult',
             'DentAccidEmerg': 'DentalAll',
             'DentSurg': 'DentalAll',
             'DentU5': 'DentalAll',
             'DentO5': 'DentalAll',
             'MentOPD': 'MentalAll',
             'MentClinic': 'MentalAll'
             }
simulation_usage['Appt_Type'] = simulation_usage['Appt_Type'].replace(appt_dict)
simulation_usage = pd.DataFrame(simulation_usage.groupby(
    by=['Year', 'Month', 'Facility_ID', 'Appt_Type']).sum().reset_index())
simulation_usage.rename(columns={'mean': 'Usage'}, inplace=True)

# Output path
output_path = Path(results_folder)

# add facility level and district columns to both real and simulation usage
mfl = pd.read_csv(rfp / 'organisation' / 'ResourceFile_Master_Facilities_List.csv')
real_usage = real_usage.merge(mfl[['Facility_ID', 'Facility_Level', 'District']],
                              on='Facility_ID', how='left')
real_usage_TB = real_usage_TB.merge(mfl[['Facility_ID', 'Facility_Level', 'District']],
                                    on='Facility_ID', how='left')
simulation_usage = simulation_usage.merge(mfl[['Facility_ID', 'Facility_Level', 'District']],
                                          on='Facility_ID', how='left')

# comparison and plots
# the appts to be compared
appts_real = list(pd.unique(real_usage['Appt_Type'])) + list(pd.unique(real_usage_TB['Appt_Type']))
appts_model = list(pd.unique(simulation_usage['Appt_Type']))
appts_to_compare = ['InpatientDays', 'IPAdmission', 'OPD',
                    'U5Malnutr',
                    'Delivery', 'Csection',
                    'FamPlan',
                    'AntenatalTotal',
                    'EPI',
                    'AccidentsandEmerg',
                    'MentalAll',
                    'NewAdult', 'EstAdult', 'Peds', 'VCTNegative', 'VCTPositive', 'MaleCirc',
                    'TBNew']
appts_not_compare = ['AntenatalFirst',  # real data only for 2013-2016
                     'MajorSurg', 'MinorSurg',  # no real data
                     'TBFollowUp',  # no real data
                     'PMTCT', 'STI', 'DentalAll',  # no model data
                     'LAB', 'RADIO',  # categories that have no real data
                     'ConWithDCSA'  # no real data
                     ]


# calculations and plots
# Average annual usage per appt type
def avg_yearly_usage_by_nation(usage_df):
    usage_df = pd.DataFrame(usage_df.groupby(
        by=['Year', 'Appt_Type'], dropna=False).agg({'Usage': 'sum'}).reset_index())

    usage_df = pd.DataFrame(usage_df.groupby(
        by=['Appt_Type'], dropna=False).agg({'Usage': 'mean'}).reset_index())

    return usage_df


real_usage_year_nation = pd.concat([avg_yearly_usage_by_nation(real_usage),
                                    avg_yearly_usage_by_nation(real_usage_TB)],
                                   ignore_index=True)

simulation_usage_year_nation = avg_yearly_usage_by_nation(simulation_usage)

usage_year_nation = real_usage_year_nation.merge(
    simulation_usage_year_nation, how='outer', on='Appt_Type').rename(
    columns={'Usage_x': 'Real_Usage', 'Usage_y': 'Simulation_Usage'}).dropna().reset_index(drop=True)

usage_year_nation['Relative_Difference'] = (
    (usage_year_nation['Simulation_Usage'] - usage_year_nation['Real_Usage']) /
    usage_year_nation['Real_Usage']
)

usage_year_nation = usage_year_nation[usage_year_nation['Relative_Difference'] <= 1].reset_index(drop=True)

usage_year_nation.plot(kind='scatter', x='Appt_Type', y='Relative_Difference',
                       title='Relative difference of model and real average annual usage by appt type')
plt.xticks(rotation=90)
plt.hlines(y=0, xmin=0, xmax=len(usage_year_nation)-1, colors='green', linewidth=2)
for i in usage_year_nation.index:
    plt.annotate(usage_year_nation.loc[i, 'Relative_Difference'].round(2),
                 xy=(i-0.5, usage_year_nation.loc[i, 'Relative_Difference'] + 200 * (i % 2 + 1)))
plt.tight_layout()
plt.show()
