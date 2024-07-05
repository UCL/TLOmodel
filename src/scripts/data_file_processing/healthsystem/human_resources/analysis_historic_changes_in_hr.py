import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd

# Set local Dropbox source
path_to_dropbox = Path(  # <-- point to the TLO dropbox locally
    '/Users/tbh03/SPH Imperial College Dropbox/Tim Hallett/Thanzi la Onse Theme 1 SHARE'
)


# File provided by Dominic Nkhome
f = path_to_dropbox / '07 - Data' / 'Historical_Changes_in_HR' / 'Monthly Staff Totals Consolidated.xlsx'
wb = pd.read_excel(f, sheet_name=None)

def process_sheet(sheet_name):
    df = wb[sheet_name].set_index('District').stack()
    df.index = df.index.set_names(['District', 'Month'])
    df.name = int(sheet_name)
    return df

df = pd.concat([process_sheet(sheet) for sheet in wb.keys() if sheet in (str(y) for y in (2017, 2018, 2019, 2020, 2021, 2022, 2023))], axis=1).stack()
# NB. Not importing 2024 as the data are incomplete
df.index = df.index.set_names(['District', 'Month', 'Year'])


# Summarise trend over years
df.groupby('Year').sum().plot()
plt.show()

# Summarise trend over years, in each district
df.groupby(['Year', 'District']).sum().unstack().plot()
plt.title('Change in the Number of Healthcare Workers by District, 2017-2024')
plt.ylabel('Number of Staff')
plt.tight_layout()
plt.show()


# Normalise to 2018 numbers
year_on_year_trend = df.groupby('Year').sum()
year_on_year_trend_normalised = year_on_year_trend / year_on_year_trend[2018]
year_on_year_trend_normalised.plot(label='Data', marker='o')

# Fit a regression line from 2017 to 2021
snippet_to_fit_to = year_on_year_trend_normalised.loc[slice(2017, 2021)]
x = np.array(snippet_to_fit_to.index)
y = snippet_to_fit_to.values

coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
# poly1d_fn is now a function which takes in x and returns an estimate for y
plt.title('Change in the Number of Healthcare Workers, 2017-2023')
plt.ylabel('Number of Staff\n(Normalised to 2018)')
plt.plot(x, poly1d_fn(x), '--r', label='Best Fit, 2017-2021')
plt.legend()
plt.axhline(y=1.0, color='k')
plt.show()





