import calendar
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo.analysis.utils import get_root_path

# Numbers employed in HRH (As provided by Dominic Nkhoma: Email 13/8/14)
# The numbers below are as quoted in the email by Dominic and found in the spreadsheet at the path given below.
# Path(<<TLO SHARED FOLDER>>/07 - Data/Historical_Changes_In_HR/03/Malawi MOH Yearly_Employees_Data_Updated.xlsx

year_by_year = pd.Series({
    2017: 24863,
    2018: 24156,
    2019: 25994,
    2020: 24763,
    2021: 28737,
    2022: 29570,
    2023: 31304,
    2024: 34486,
})


fig, ax = plt.subplots()
year_by_year.plot(ax=ax, legend=False, marker='o')
ax.set_title('Trend in Healthcare Workers', fontweight='bold', fontsize=10)
ax.set_ylabel('Number of HCW')
ax.set_ylim(0, 40_000)
ax.set_xlim(2016, 2025)
fig.tight_layout()
fig.show()


# difference vs 2017
diff_since_2017 = year_by_year - year_by_year.at[2017]


# %% Curve-fitting to the scale-up

from scipy.optimize import curve_fit

def func(y, beta, ystart):
    return np.exp(beta * (y - ystart - 2017).clip(0.0))

popt, pcov = curve_fit(func,
                       year_by_year.index.to_numpy(),
                       year_by_year.to_numpy() / year_by_year[2017],
                       )

plt.figure()
plt.plot(year_by_year.index.to_numpy(), year_by_year.to_numpy() / year_by_year[2017], marker='o', label='Historical data')
plt.plot(year_by_year.index.to_numpy(), func(year_by_year.index.to_numpy(), *popt), label='fit')
plt.show()


#%% Plot to explain setup of Scenario

to_plot = pd.DataFrame(index=pd.Index(range(2017, 2030), name='year'))

to_plot['Data'] = year_by_year / year_by_year[2017]  # data is the year-on-year trend, normalised to 2017

# Assign the date of mid-year to the data points
to_plot['mid-year_date'] = pd.to_datetime(dict(year=to_plot.index, month=7, day=1)).dt.date.values

# Define scale-up pattern: fitted line
to_plot['Scale-up'] = pd.Series(index=year_by_year.index.to_numpy(), data=func(year_by_year.index.to_numpy(), *popt))

# Define counterfactual scenario
to_plot['No Scale-up'] = 1.0

# Actual and Counterfactual are held to the last level achieved in the data when we go forward
to_plot['Scale-up'] = to_plot['Scale-up'].ffill()
to_plot['No Scale-up'] = to_plot['No Scale-up'].ffill()

# For plotting the scenarios, we'll show that  the changes happen at the start of the year.
step_dates = [datetime.date(y, 1, 1) for y in to_plot.index] + [datetime.date(to_plot.index.max() + 1, 1, 1)]

for xlim in (datetime.date(2025, 1, 1), datetime.date(2031, 1, 1)):
    fig, ax = plt.subplots()
    ax.stairs(                                      # line for the actual scenario
        values=to_plot['Scale-up'],
        edges=step_dates, baseline=None,
        label='Scale-up Actual Scenario',
        color='r',
        zorder=2,
        linewidth=3)
    ax.stairs(                                      # the shading between the actual and counterfactual scenarios
        values=to_plot['Scale-up'],
        edges=step_dates,
        baseline=1.0,
        label=None,
        color='r',
        zorder=2,
        fill=True,
        alpha=0.3)
    ax.stairs(                                      # line for the counterfactual scenario
        values=to_plot['No Scale-up'],
        edges=step_dates, baseline=None,
        label='No Scale-up Counterfactual',
        color='g',
        zorder=3,
        linewidth=3)
    ax.plot(                                        # the data
        to_plot['mid-year_date'],
        to_plot['Data'],
        marker='o',
        linestyle='--',
        label='Data')
    ax.set_title('Change in the Number of Healthcare Workers')
    ax.set_ylabel('Number of Staff\n(Normalised to 2017)')
    ax.legend(loc='upper left')
    ax.grid()
    ax.set_ylim(0.95, 1.6)
    ax.set_xlabel('Date')
    ax.set_xlim(datetime.date(2017, 7, 1), xlim)
    fig.show()


#%% Save this as a scale-up scenario

# Work-out the annual multipliers that will give the desired scale-up pattern
scale_up_multipliers = dict()
scale_up_multipliers[2010] = 1.0
for idx, val in to_plot['Scale-up'].sort_index().items():
    if idx-1 > to_plot['Scale-up'].index[0]:
        scale_up_multipliers[idx] = val / to_plot.loc[idx-1, 'Scale-up']


scale_up_scenario = pd.DataFrame({'dynamic_HR_scaling_factor': pd.Series(scale_up_multipliers)})
scale_up_scenario['scale_HR_by_popsize'] = ["FALSE"] * len(scale_up_scenario)
scale_up_scenario = scale_up_scenario.reset_index()
scale_up_scenario = scale_up_scenario.rename(columns={'index': 'year'})
scale_up_scenario['year'] = scale_up_scenario['year'].astype(int)
scale_up_scenario.sort_values('year', inplace=True)

# Add (or over-write) a sheet called 'historical_scaling' with the scale-up pattern to the relevant ResourceFile
target_file = get_root_path() / 'resources' / 'healthsystem' / 'human_resources' / 'scaling_capabilities' / 'ResourceFile_dynamic_HR_scaling.xlsx'

with pd.ExcelWriter(target_file, engine='openpyxl', mode='a', if_sheet_exists="replace") as writer:
    scale_up_scenario.to_excel(writer, sheet_name='historical_scaling', index=False)
