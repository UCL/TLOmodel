from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tlo import Date
from tlo.analysis.utils import extract_results, load_pickled_dataframes, summarize

folder = Path("/Users/rem76/PycharmProjects/TLOmodel/outputs/rm916@ic.ac.uk/longterm_trends_all_diseases-2024-09-25T110820Z")
nd = extract_results(
    folder,
    module="tlo.methods.demography.detail",
    key="properties_of_deceased_persons",
    custom_generate_series=(
        lambda df: df.loc[(df['age_days'] < 364) & (df['age_years'] == 0)].assign(
            year=df['date'].dt.year).groupby(['year'])['year'].count()),
    do_scaling=True)
neo_deaths = nd.fillna(0)


nd_first_half = extract_results(
    folder,
    module="tlo.methods.demography.detail",
    key="properties_of_deceased_persons",
    custom_generate_series=(
        lambda df: df.loc[(df['age_days'] < 182) & (df['age_years'] == 0)].assign(
            year=df['date'].dt.year).groupby(['year'])['year'].count()),
    do_scaling=True)
nd_first_half_deaths = nd_first_half.fillna(0)
nd_first_half_deaths.to_csv("nd_first_half_deaths.csv")
#print(nd_first_half_deaths/neo_deaths)

print(summarize(nd_first_half_deaths/neo_deaths))


## number of deaths by cause
output = load_pickled_dataframes(folder)  # parse output file

demography_details_death = output['tlo.methods.demography.detail']['properties_of_deceased_persons']
demog_below_six_months = demography_details_death.loc[(demography_details_death.age_days < 182) & (demography_details_death['age_years'] == 0)]
demog_above_six_months = demography_details_death.loc[(demography_details_death.age_days > 182)& (demography_details_death['age_years'] == 0)]

below_six_months_counts = demog_below_six_months['cause_of_death'].value_counts()
above_six_months_counts= demog_above_six_months['cause_of_death'].value_counts()


combined_counts = pd.DataFrame({
    'Above Six Months': above_six_months_counts,
    'Below Six Months': below_six_months_counts
}).fillna(0)  # Fill NaN with 0 for causes not present in both groups

print(combined_counts)
# Plotting
combined_counts.plot(kind='bar', color=['#1C6E8C', '#9AC4F8'], figsize=(10, 6))
plt.title('Cause of Death for Individuals Above and Below Six Months')
plt.xlabel('Cause of Death')
plt.ylabel('Count')
plt.legend(title='Age Group')
plt.tight_layout()
plt.show()

### All age days, all years?
below_one_year = demography_details_death[demography_details_death["age_years"] < 1]
below_one_year['age_days'].plot(kind='hist', bins=30)  # Adjust bins as needed
plt.xlabel("Days made through year - below age one")
plt.ylabel("Frequency")
plt.title("Histogram of Age in Days")
plt.show()


### All age days, above 5 years
above_five_years = demography_details_death[demography_details_death["age_years"] > 5]
how_far_in_year = above_five_years['age_days'] % 365
how_far_in_year.plot(kind='hist', bins=30)  # Adjust bins as needed
plt.xlabel("Days made through year - above age 5")
plt.ylabel("Frequency")
plt.title("Histogram of Age in Days")
plt.show()


### How far into 5 year age groups
above_five_years = demography_details_death[demography_details_death["age_years"] > 5]
age_5_year_groups = (above_five_years['age_years'] // 5) * 5  # Group ages in 5-year intervals
days_in_5_year_group = (above_five_years['age_days'] % (5 * 365))
days_in_5_year_group.plot(kind='hist', bins=30)
plt.xlabel("Days into 5-year interval")
plt.ylabel("Frequency")
plt.title("Histogram of Days Lived into Each 5-Year Age Group")
plt.show()


### How far into 1-4 year age groups
below_five_years = demography_details_death[(demography_details_death["age_years"] > 1) & (demography_details_death["age_years"] < 5)]
days_in_3_year_group = (below_five_years['age_days'] % (3 * 365))
days_in_3_year_group.plot(kind='hist', bins=30)
plt.xlabel("Days into 4-year interval")
plt.ylabel("Frequency")
plt.title("Histogram of Days Lived into 1-4 Year Age Group")
plt.show()

