"""Produce comparisons between model and GBD of deaths by cause in a particular year."""

# todo - update to newest data
# todo - deaths, to include 0 year-olds.
# todo - look to see why diarrhoea causes so many deaths

import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tlo.methods import demography

# Define the particular year for the focus of this analysis
year = 2010

# Resource file path
rfp = Path("./resources")


# Where will outputs be found
outputpath = Path("./outputs")  # folder for convenience of storing outputs
results_filename = outputpath / 'long_run.pickle'

with open(results_filename, 'rb') as f:
    output = pickle.load(f)['output']

datestamp = datetime.today().strftime("__%Y_%m_%d")

# %% do the scaling so that the population size matches that of the real population of Malawi
def get_scaling_factor(parsed_output):
    """Find the factor that the model results should be multiplied by to be comparable to data"""
    # Get information about the real population size (Malawi Census in 2018)
    cens_tot = pd.read_csv(rfp / "ResourceFile_PopulationSize_2018Census.csv")['Count'].sum()
    cens_yr = 2018

    # Get information about the model population size in 2018 (and fail if no 2018)
    model_res = parsed_output['tlo.methods.demography']['population']
    model_yr = pd.to_datetime(model_res.date).dt.year

    if cens_yr in model_yr.values:
        model_tot = model_res.loc[model_yr == cens_yr, 'total'].values[0]
    else:
        print("WARNING: Model results do not contain the year of the census, so cannot scale accurately")
        model_tot = model_res.at[abs(model_yr - cens_yr).idxmin(), 'total']

    # Calculate ratio for scaling
    return cens_tot / model_tot

scaling_factor = get_scaling_factor(output)

# %% load gbd data
gbd = pd.read_csv(rfp / "ResourceFile_Deaths_And_Causes_DeathRates_GBD.csv")

# %% Declare mapping between GBD and TLO causes
def get_causes_mappers():
    # Make a dict that gives a mapping for each cause, from the GBD string and the strings put out from the TLO model
    # todo - automate declaration and check that all tlo causes accounted for
    gbd_causes = pd.Series(list(set(gbd['cause_name']))).sort_values()
    tlo_causes = pd.Series(list(set(pd.unique(output['tlo.methods.demography']['death']['cause'])))).sort_values()

    causes = dict()
    causes['AIDS'] = {
        'gbd_strings': ['HIV/AIDS'],
        'tlo_strings': ['AIDS']
    }
    causes['Malaria'] = {
        'gbd_strings': ['Malaria'],
        'tlo_strings': ['severe_malaria']
    }
    causes['Childhood Diarrhoea'] = {
        'gbd_strings': ['Diarrheal diseases'],
        'tlo_strings': ['Diarrhoea_rotavirus',
                        'Diarrhoea_shigella',
                        'Diarrhoea_astrovirus',
                        'Diarrhoea_campylobacter',
                        'Diarrhoea_cryptosporidium',
                        'Diarrhoea_sapovirus',
                        'Diarrhoea_tEPEC',
                        'Diarrhoea_adenovirus',
                        'Diarrhoea_norovirus',
                        'Diarrhoea_ST-ETEC']
    }
    causes['Oesophageal Cancer'] = {
        'gbd_strings': ['Esophageal cancer'],
        'tlo_strings': ['OesophagealCancer']
    }
    causes['Epilepsy'] = {
        'gbd_strings': ['Epilepsy'],
        'tlo_strings': ['Epilepsy']
    }
    causes['Self-harm'] = {
        'gbd_strings': ['Self-harm'],
        'tlo_strings': ['Suicide']
    }
    causes['Complications in Labour'] = {
        'gbd_strings': ['Maternal disorders', 'Neonatal disorders', 'Congenital birth defects'],
        'tlo_strings': ['postpartum labour', 'labour']
    }

    # Catch-all groups for Others:
    #  - map all the un-assigned gbd strings to Other
    all_gbd_strings_mapped = []
    for v in causes.values():
        all_gbd_strings_mapped.extend(v['gbd_strings'])

    gbd_strings_not_assigned = list(set(pd.unique(gbd.cause_name)) - set(all_gbd_strings_mapped))

    causes['Other'] = {
        'gbd_strings': gbd_strings_not_assigned,
        'tlo_strings': ['Other']
    }

    # make the mappers:
    causes_df = pd.DataFrame.from_dict(causes, orient='index')

    #  - from tlo_strings (key=tlo_string, value=unified_name)
    mapper_from_tlo_strings = dict((v, k) for k,v in (
        causes_df.tlo_strings.apply(pd.Series).stack().reset_index(level=1, drop=True)
    ).iteritems())
    # fn = lambda x:  mapper_from_tlo_strings[x] if x in mapper_from_tlo_strings else 'Other'

    #  - from gbd_strings (key=gbd_string, value=unified_name)
    mapper_from_gbd_strings = dict((v, k) for k,v in (
        causes_df.gbd_strings.apply(pd.Series).stack().reset_index(level=1, drop=True)
    ).iteritems())

    # check that the mappers are exhaustive for all causes in both gbd and tlo
    assert all([c in mapper_from_tlo_strings for c in pd.unique(output['tlo.methods.demography']['death']['cause'])])
    assert all([c in mapper_from_gbd_strings for c in pd.unique(gbd['cause_name'])])

    return mapper_from_tlo_strings, mapper_from_gbd_strings

mapper_from_tlo_strings, mapper_from_gbd_strings = get_causes_mappers()

dem = demography.Demography()
AGE_RANGE_CATEGORIES = dem.AGE_RANGE_CATEGORIES.copy()
AGE_RANGE_LOOKUP = dem.AGE_RANGE_LOOKUP.copy()


# %% Define age categories
def age_cats(ages_in_years):

    # edit to that top-end is 95+ to match with how GBD death are reported
    for age in range(95, 100):
        AGE_RANGE_LOOKUP[age] = '95+'

    if '95-99' in AGE_RANGE_CATEGORIES:
        AGE_RANGE_CATEGORIES.remove('95-99')
    if '100+' in AGE_RANGE_CATEGORIES:
        AGE_RANGE_CATEGORIES.remove('100+')
    if '95+' not in AGE_RANGE_CATEGORIES:
        AGE_RANGE_CATEGORIES.append('95+')

    age_cats = pd.Series(
        pd.Categorical(ages_in_years.map(AGE_RANGE_LOOKUP),
                       categories=AGE_RANGE_CATEGORIES, ordered=True)
    )
    return age_cats


# %% Get deaths from Model into a pivot-table (index=sex/age, columns=unified_cause) (in a particular year)

deaths_df = output["tlo.methods.demography"]["death"]
deaths_df["date"] = pd.to_datetime(deaths_df["date"])
deaths_df["year"] = deaths_df["date"].dt.year
deaths_df['age_range'] = age_cats(deaths_df.age)

deaths_df["unified_cause"] = deaths_df["cause"].map(mapper_from_tlo_strings)
assert not pd.isnull(deaths_df["cause"]).any()

df = deaths_df.loc[(deaths_df.year == year)].groupby(
    ['sex', 'age_range', 'unified_cause']).size().reset_index()
df = df.rename(columns={0: 'count'})
df['count'] *= scaling_factor

model_pt = df.pivot_table(index=['sex', 'age_range'], columns='unified_cause', values='count', fill_value=0)

# todo - guaranteee that all values of age_range are represented

# %% Get deaths from GBD into a pivot-table (index=sex/age, columns=unified_cause) (in a particular year)

# limit to relevant year
gbd = gbd.loc[gbd.year == year]

# map onto unified causes of death, standardised age-groups and collapse into age/cause count of death:
gbd["unified_cause"] = gbd["cause_name"].map(mapper_from_gbd_strings)

gbd['Age_Grp'] = gbd['Age_Grp'].replace({'1-4': '0-4'})  # todo - GBD data should pertain to 0 year-olds
gbd['age_range'] = pd.Categorical(gbd['Age_Grp'], categories=AGE_RANGE_CATEGORIES, ordered=True)
gbd['sex'] = gbd['sex_name'].map({'Male': 'M', 'Female': 'F'})

gbd_pt = gbd.pivot_table(index=['sex', 'age_range'], columns='unified_cause', values='val', fill_value=0)


# %% Make figures

# Overall Summary of deaths by cause
model_pt.sum().plot.bar()
plt.xlabel('Cause')
plt.ylabel(f"Total deaths in {year}")
plt.savefig(outputpath / f"Deaths by cause {datestamp}.pdf")
plt.show()

def plot_stacked_bar_chart_of_deaths_by_cause(ax, pt, title):
    pt.plot.bar(stacked=True, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Number of deaths")
    ax.set_label('Age Group')
    ax.get_legend().remove()

sex_as_string = lambda x: 'Females' if x == 'F' else 'Males'

fig, axs = plt.subplots(2, 2)
for i, sex in enumerate(['F', 'M']):
     plot_stacked_bar_chart_of_deaths_by_cause(axs[i, 0], model_pt.loc[(sex,), ], title=f"Model: {sex_as_string(sex)}")
     plot_stacked_bar_chart_of_deaths_by_cause(axs[i, 1], gbd_pt.loc[(sex, ), ], title=f"Data: {sex_as_string(sex)}")
plt.show()
plt.savefig(outputpath / 'Deaths_Calibration_StackedBars.png')

# %% Plots comparing between model and actual deaths across all ages and sex:
xs = gbd_pt.sum()
ys = model_pt.sum()
line_x = [min(xs), max(xs)]
line_y = [min(ys), max(ys)]
plt.plot(xs, ys, 'bo')
show_labels = ['AIDS', 'Childhood Diarrhoea', 'Other']
for i, (x, y) in enumerate(zip(xs, ys)):
    label = xs.index[i]
    if label in show_labels:
        plt.annotate(label,
                     (x, y),
                     textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                     ha='center'
                     )
plt.plot(line_x, line_y, 'r')
plt.xlabel('Number of deaths in GBD')
plt.ylabel('Number of deaths in Model')
plt.title(f'Total Numbers of Deaths in {year}')
plt.savefig(outputpath / 'Deaths_Calibration_ScatterPlot.png')
plt.show()

# %% Plots of DALYS

# Get Model DALYS:
dalys = output['tlo.methods.healthburden']['dalys'].copy()

# drop date because this is just the date of logging and not the date to which the results pertain.
dalys = dalys.drop(columns=['date'])

# limit to the year (so the output wil be new year's day of the next year)
dalys = dalys.loc[dalys.year == (year + 1)]
dalys = dalys.drop(columns=['year'])

# format age-groups and set index to be sex/age_range: collapse the
dalys['age_range'] = pd.Categorical(dalys['age_range'].replace({
    '95-99': '95+',
    '100+': '95+'}
), categories=AGE_RANGE_CATEGORIES, ordered=True)

dalys = pd.DataFrame(dalys.groupby(['sex', 'age_range']).sum())

# Get GBD Dalys
# todo!


fig, axs = plt.subplots(2, 1)
for i, sex in enumerate(['F', 'M']):
    dalys.loc[(sex, )].plot.bar(stacked=True, legend=False, ax=axs[i])
    axs[i].set_title(f"DALYS in Model for {sex_as_string(sex)}")

plt.savefig(outputpath / 'DALYS_stacked_bars.png')
plt.show()

