import calendar
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set local Dropbox source
path_to_dropbox = Path(  # <-- point to the TLO dropbox locally
    '/Users/tbh03/Library/CloudStorage/OneDrive-SharedLibraries-ImperialCollegeLondon/TLOModel - WP - Documents/'
)

# File provided by Dominic Nkhoma (29th July 2024)
df = pd.read_excel(
    path_to_dropbox / '07 - Data' / 'Historical_Changes_in_HR' / 'Yearly Employees Data Consolidated (Vote 006).xlsx',
    sheet_name='Sheet1')

# Censor everything after February 2022 (Dominic advises that this dataset is unreliable after that date)
months = {month: index for index, month in enumerate(calendar.month_name) if month}
dates = pd.to_datetime(dict(year=df.Year, month=df['Month'].str.strip().map(months).astype(int), day=1)).dt.date
df = df.loc[dates < datetime.date(2022, 2, 1)]

# Assemble into multi-index series
num_employees = df.set_index(['District', 'Month', 'Year'])['Emp_Totals']

# Summarise trend over years (taking average over months, and then summing over District)
year_by_year = num_employees.groupby(by=['Year', 'District']).mean().groupby(by='Year').sum()

# Summarise trend over years alt. (taking sum over the districts, and then averaging over months)
year_by_year2 = num_employees.groupby(by=['Year', 'Month']).sum().groupby(by='Year').mean()

fig, ax = plt.subplots()
year_by_year.plot(ax=ax, legend=False, marker='o')
# year_by_year2.plot(ax=ax, legend=False)
ax.set_title('Trend in Healthcare Workers', fontweight='bold', fontsize=10)
ax.set_ylabel('Number of HCW')
ax.set_ylim(20_000, 30_000)
ax.set_xlim(2016, 2023)
fig.tight_layout()
fig.show()


# Summarise trend over years, in each district
fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
num_employees.groupby(by=['Year', 'District']).mean().unstack().plot(ax=ax, legend=False, marker='.')
ax.set_title('Trend in Healthcare Workers by District', fontweight='bold', fontsize=10)
ax.set_ylabel('Number of HCW')
ax.set_ylim([0, 5_000])
fig.legend(loc="outside lower center", ncols=5, fontsize='small')
fig.show()

# difference vs 2017
diff_since_2017 = year_by_year - year_by_year.at[2017]


#%% Plot to explain setup of Scenario

to_plot = pd.DataFrame(index=pd.Index(range(2017, 2024), name='year'))

to_plot['Data'] = year_by_year / year_by_year[2017]  # data is the year-on-year trend, normalised to 2017

# Assign the date of mid-year to the data points
to_plot['mid-year_date'] = pd.to_datetime(dict(year=to_plot.index, month=7, day=1)).dt.date.values

def extrapolate(x_new, x_known, y_known):
    """1-D Extrapolation"""
    return np.poly1d(np.polyfit(x_known, y_known, deg=1))(x_new)

# Define scale-up pattern: equal to the data, but trimming out small decrease and extrapolation in 2023 using 2021-2022
to_plot['Scale-up'] = to_plot['Data'].clip(lower=1.0)
yrs_to_extrpolate_from = [2021, 2022]
to_plot.loc[2023, 'Scale-up'] = extrapolate(2023, yrs_to_extrpolate_from, to_plot.loc[yrs_to_extrpolate_from, 'Data'].values)

# Define counterfactual scenario
to_plot['No Scale-up'] = 1.0

# For plotting the scenarios, we'll show that  the changes happen at the start of the year.
step_dates = [datetime.date(y, 1, 1) for y in to_plot.index] + [datetime.date(to_plot.index.max() + 1, 1, 1)]

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
ax.set_title('Change in the Number of Healthcare Workers, 2017-2022')
ax.set_ylabel('Number of Staff\n(Normalised to 2017)')
ax.legend(loc='upper left')
ax.grid()
ax.set_ylim(0.95, 1.3)
ax.set_xlabel('Date')
fig.show()
