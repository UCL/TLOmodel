import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd

# Set local Dropbox source
path_to_dropbox = Path(  # <-- point to the TLO dropbox locally
    '/Users/tbh03/Library/CloudStorage/OneDrive-SharedLibraries-ImperialCollegeLondon/TLOModel - WP - Documents/'
)

# File provided by Dominic Nkhoma (29th July 2024)
df = pd.read_excel(
    path_to_dropbox / '07 - Data' / 'Historical_Changes_in_HR' / 'Yearly Employees Data Consolidated (Vote 006).xlsx',
    sheet_name='Sheet1')

num_employees = df.set_index(['District', 'Month', 'Year'])['Emp_Totals']


# Summarise trend over years (taking average by month and summing over District)
year_by_year = num_employees.groupby(by=['Year', 'District']).mean().groupby(by='Year').sum()

# Summarise trend over years alt. (taking sum over the districts, and then averaging over months)
year_by_year2 = num_employees.groupby(by=['Year', 'Month']).sum().groupby(by='Year').mean()


fig, ax = plt.subplots()
year_by_year.plot(ax=ax, legend=False)
# year_by_year2.plot(ax=ax, legend=False)
ax.set_title('Trend in Healthcare Workers, 2017-2024', fontweight='bold', fontsize=10)
ax.set_ylabel('Number of HCW')
ax.set_ylim(0, 30_000)
fig.tight_layout()
fig.show()


# Summarise trend over years, in each district
fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
num_employees.groupby(by=['Year', 'District']).mean().unstack().plot(ax=ax, legend=False)
ax.set_title('Trend in Healthcare Workers by District, 2017-2024', fontweight='bold', fontsize=10)
ax.set_ylabel('Number of HCW')
ax.set_ylim([0, 5_000])
fig.legend(loc="outside lower center", ncols=5, fontsize='small')
fig.show()

# difference vs 2017
diff_since_2017 = year_by_year - year_by_year.at[2017]


#%% Plot to explain setup of Scenario
year_on_year_trend_normalised = year_by_year / year_by_year[2017]
year_on_year_trend_normalised.plot(label='Data', marker='o')

# Fit a regression line from 2017 to 2022
snippet_to_fit_to = year_on_year_trend_normalised.loc[slice(2017, 2022)]
x = np.array(snippet_to_fit_to.index)
y = snippet_to_fit_to.values

# Fit to X and Y forcing the intercept to be zero
from statistics import linear_regression
gradient, _ = linear_regression(x - 2017, y - 1.0, proportional=True)  # The parameter proportional is set to True, to specify that
# x and y are assumed to be directly proportional (and the data to be fit to a line passing through the origin).

increase_str = f"Average increase: {round(100 * gradient, 0)}% per year"

plt.title('Change in the Number of Healthcare Workers, 2017-2022')
plt.ylabel('Number of Staff\n(Normalised to 2017)')
plt.plot(x, 1.0 + (x-2017) * gradient, '--rx', label='Best Fit, 2017-2022\n'+increase_str)
plt.plot(x, [1.0] * len(x), '-gx', label='No Scale-up Counterfactual')
plt.legend()
plt.grid()
# plt.axhline(y=1.0, color='k')
plt.ylim(0.95, 1.3)
plt.show()
