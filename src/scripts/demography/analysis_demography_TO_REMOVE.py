# %% Import Statements
import datetime
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import demography
from tlo.methods.demography import make_age_range_lookup

# Where will output go - by default, wherever this script is run
outputpath = ""

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource file for demography module
# assume Python console is started in the top-leve TLOModel directory
resourcefilepath = Path("./resources")


# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2031, 1, 1)  # so last full year of simulation will be 2030
popsize = 1000

# add file handler for the purpose of logging
sim = Simulation(start_date=start_date)

# this block of code is to capture the output to file
logfile = outputpath + "LogFile" + datestamp + ".log"

if os.path.exists(logfile):
    os.remove(logfile)
fh = logging.FileHandler(logfile)
fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
fh.setFormatter(fr)
logging.getLogger().addHandler(fh)

# run the simulation
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.seed_rngs(1)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# this will make sure that the logging file is complete
fh.flush()

# %% read the results
output = parse_log_file(logfile)

# %% Plot Population Size Over time:

# Load Model Results
pop_df = output["tlo.methods.demography"]["population"]
Model_Years = pd.to_datetime(pop_df.date)
Model_Pop = pop_df.total
Model_Pop_Normalised = (
    100 * np.asarray(Model_Pop) / np.asarray(Model_Pop[Model_Years == "2010-01-01"])
)

# Load Data
Data = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_DemographicData.xlsx",
    sheet_name="Interpolated Pop Structure",
)
Data_Pop = Data.groupby(by="year")["value"].sum()
Data_Years = Data.groupby(by="year")["year"].mean()
Data_Years = pd.to_datetime(Data_Years, format="%Y")
Data_Pop_Normalised = (
    100 * Data_Pop / np.asarray(Data_Pop[(Data_Years == Date(2010, 1, 1))])
)

plt.plot(np.asarray(Model_Years), Model_Pop_Normalised)
plt.plot(Data_Years, Data_Pop_Normalised)
plt.title("Population Size")
plt.xlabel("Year")
plt.ylabel("Population Size (Normalised to 2010)")
plt.gca().set_xlim(Date(2010, 1, 1), Date(2050, 1, 1))
plt.legend(["Model", "Data"])
plt.savefig(outputpath + "PopSize" + datestamp + ".pdf")

plt.show()

# %% Population Pyramid in 2015

# Make Dateframe of the relevant output:

# get the dataframe for men and women:
pop_f_df = output["tlo.methods.demography"]["age_range_f"]
pop_m_df = output["tlo.methods.demography"]["age_range_m"]

# create mask for the 2015 and 2030 results:
m2015 = pd.to_datetime(pop_f_df["date"]).dt.year == 2015
m2030 = pd.to_datetime(pop_f_df["date"]).dt.year == 2030

# Extract the results for men and women for these two years and combine
model = pd.concat(
    [
        pd.concat(
            [
                pd.melt(pop_f_df.loc[m2015, pop_f_df.columns[1:]]),
                pd.DataFrame(index=range(21), data={"Year": 2015, "Sex": "F"}),
            ],
            axis=1,
        ),  # Female, 2015
        pd.concat(
            [
                pd.melt(pop_f_df.loc[m2030, pop_f_df.columns[1:]]),
                pd.DataFrame(index=range(21), data={"Year": 2030, "Sex": "F"}),
            ],
            axis=1,
        ),  # Female, 2030
        pd.concat(
            [
                pd.melt(pop_m_df.loc[m2015, pop_m_df.columns[1:]]),
                pd.DataFrame(index=range(21), data={"Year": 2015, "Sex": "M"}),
            ],
            axis=1,
        ),  # Male, 2015
        pd.concat(
            [
                pd.melt(pop_m_df.loc[m2030, pop_m_df.columns[1:]]),
                pd.DataFrame(index=range(21), data={"Year": 2030, "Sex": "M"}),
            ],
            axis=1,
        ),  # Male, 2030
    ],
    axis=0,
)

model_pop_pyramind = model.rename(
    columns={"variable": "agegrp", "value": "pop", "Year": "year", "Sex": "sex"}
)


# Load population data to compare population pyramid: 2015
Data = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_DemographicData.xlsx",
    sheet_name="Interpolated Pop Structure",
)
df = Data[(Data["year"] == 2015) | (Data["year"] == 2030)].copy()

# get look-up to group ages by same groups
(__tmp__, age_grp_lookup) = make_age_range_lookup()
df["agegrp"] = df["age_from"].map(age_grp_lookup)
df = df.rename(columns={"gender": "sex", "value": "pop"})
df["sex"] = df["sex"].map({"female": "F", "male": "M"})
data_pop_pyramid = df.groupby(["year", "agegrp", "sex"])["pop"].sum().reset_index()


# Join the model and the data together for ease of plotting
combo = data_pop_pyramid.merge(
    model_pop_pyramind,
    on=["agegrp", "year", "sex"],
    how="inner",
    suffixes=("_data", "_model"),
)

# give the right order of the agegrp cateogorical variable
combo["agegrp"] = combo["agegrp"].astype("category")
combo["agegrp"].cat.reorder_categories(
    pd.unique(list(age_grp_lookup.values())), inplace=True
)
combo = combo.sort_values(by=["year", "sex", "agegrp"]).reset_index(drop=True)

# Add in a scaled population size (within a particular year) so that its percentage in that age/sex grp
combo["pop_data_sc"] = combo["pop_data"]
combo.loc[combo["year"] == 2015, "pop_data_sc"] /= combo.loc[
    combo["year"] == 2015, "pop_data"
].sum()
combo.loc[combo["year"] == 2030, "pop_data_sc"] /= combo.loc[
    combo["year"] == 2030, "pop_data"
].sum()

combo["pop_model_sc"] = combo["pop_model"]
combo.loc[combo["year"] == 2015, "pop_model_sc"] /= combo.loc[
    combo["year"] == 2015, "pop_model"
].sum()
combo.loc[combo["year"] == 2030, "pop_model_sc"] /= combo.loc[
    combo["year"] == 2030, "pop_model"
].sum()


# Model Population Pyramid
# TODO: new plots


# Traditional Populaiton Pyramid: Model Only
fig, axes = plt.subplots(ncols=2, nrows=2, sharey=True)
age_grp_labels = pd.unique(list(age_grp_lookup.values()))
combo.loc[(combo["year"] == 2015) & (combo["sex"] == "F"), ["pop_model_sc"]].plot.barh(
    ax=axes[0][0], align="center", color="red", zorder=10
)
axes[0][0].set(title="Women, 2015")


combo.loc[(combo["year"] == 2015) & (combo["sex"] == "M"), ["pop_model_sc"]].plot.barh(
    ax=axes[0][1], align="center", color="blue", zorder=10
)
axes[0][1].set(title="Men, 2015")


combo.loc[(combo["year"] == 2030) & (combo["sex"] == "F"), ["pop_model_sc"]].plot.barh(
    ax=axes[1][0], align="center", color="red", zorder=10
)
axes[1][0].set(title="Women, 2030")


combo.loc[(combo["year"] == 2030) & (combo["sex"] == "M"), ["pop_model_sc"]].plot.barh(
    ax=axes[1][1], align="center", color="blue", zorder=10
)
axes[1][1].set(title="Men, 2030")


axes[0][0].invert_xaxis()
axes[0][0].set(
    yticks=np.arange(len(age_grp_labels))[::2], yticklabels=age_grp_labels[::2]
)
axes[0][0].yaxis.tick_right()
axes[1][0].invert_xaxis()
axes[1][0].set(
    yticks=np.arange(len(age_grp_labels))[::2], yticklabels=age_grp_labels[::2]
)
axes[1][0].yaxis.tick_right()

for ax in axes.flat:
    ax.margins(0.03)
    ax.grid(True)

fig.tight_layout()
fig.subplots_adjust(wspace=0.09)
plt.savefig(outputpath + "PopPyramidModelOnly" + datestamp + ".pdf")
plt.show()

# Model Vs Data Pop Pyramid
fig, axes = plt.subplots(ncols=2, nrows=2, sharey=True)
xticks = [0, 5, 10, 15, 20]
xticklabels = age_grp_labels[xticks]

combo.loc[
    (combo["year"] == 2015) & (combo["sex"] == "F"), ["pop_model_sc", "pop_data_sc"]
].reset_index(drop=True).plot(ax=axes[0][0])
axes[0][0].set(title="Women, 2015")
axes[0][0].set(xticks=xticks, xticklabels=xticklabels)

combo.loc[
    (combo["year"] == 2015) & (combo["sex"] == "M"), ["pop_model_sc", "pop_data_sc"]
].reset_index(drop=True).plot(ax=axes[0][1])
axes[0][1].set(title="Men, 2015")
axes[0][1].set(xticks=xticks, xticklabels=xticklabels)

combo.loc[
    (combo["year"] == 2030) & (combo["sex"] == "F"), ["pop_model_sc", "pop_data_sc"]
].reset_index(drop=True).plot(ax=axes[1][0])
axes[1][0].set(title="Women, 2015")
axes[1][0].set(xticks=xticks, xticklabels=xticklabels)

combo.loc[
    (combo["year"] == 2030) & (combo["sex"] == "M"), ["pop_model_sc", "pop_data_sc"]
].reset_index(drop=True).plot(ax=axes[1][1])
axes[1][1].set(title="Men, 2030")
axes[1][1].set(xticks=xticks, xticklabels=xticklabels)

for ax in axes.flat:
    ax.margins(0.03)
    ax.grid(True)

fig.tight_layout()
fig.subplots_adjust(wspace=0.09)
plt.savefig(outputpath + "PopPyramid_ModelVsData" + datestamp + ".pdf")
plt.show()


# %% Plots births ....

births_df = output["tlo.methods.demography"]["on_birth"]

plt.plot_date(births_df["date"], births_df["mother_age"])
plt.xlabel("Year")
plt.ylabel("Age of Mother")
plt.savefig(outputpath + "Births" + datestamp + ".pdf")
plt.show()

# %% Plots deaths ...

deaths_df = output["tlo.methods.demography"]["death"]

plt.plot_date(deaths_df["date"], deaths_df["age"])
plt.xlabel("Year")
plt.ylabel("Age at Death")
plt.savefig(outputpath + "Deaths" + datestamp + ".pdf")
plt.show()

# %% Plot deaths by cause:

deaths_df = output["tlo.methods.demography"]["death"]
deaths_df["date"] = pd.to_datetime(deaths_df["date"])
deaths_df["year"] = deaths_df["date"].dt.year
death_by_cause = deaths_df.groupby(["year", "cause"])["person_id"].size()

death_by_cause = death_by_cause.reset_index()
death_by_cause.index = death_by_cause["year"]
death_by_cause.drop(columns="year", inplace=True)
death_by_cause = death_by_cause.rename(columns={"person_id": "num_deaths"})

death_by_cause.plot.bar(stacked=True)
plt.show()
