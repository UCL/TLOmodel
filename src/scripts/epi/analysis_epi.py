import datetime
import os
import time
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
)

start_time = time.time()

# Where will output go
outputpath = Path("./outputs/epi")

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

start_date = Date(2010, 1, 1)
end_date = Date(2025, 12, 31)
popsize = 2000

log_config = {
    'filename': 'Epi_LogFile',
    'custom_levels': {"*": logging.WARNING,
                      "tlo.methods.epi": logging.INFO,
                      "tlo.methods.healthsystem.summary": logging.INFO}
}

# Establish the simulation object
sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

# ----- Control over the types of intervention that can occur -----
# Make a list that contains the treatment_id that will be allowed. Empty list means nothing allowed.
# '*' means everything. It will allow any treatment_id that begins with a stub (e.g. Mockitis*)
service_availability = ["*"]

# Register the appropriate modules
sim.register(
    demography.Demography(resourcefilepath=resourcefilepath),
    simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
    healthsystem.HealthSystem(
        resourcefilepath=resourcefilepath,
        service_availability=["*"],  # all treatment allowed
        mode_appt_constraints=1,  # mode of constraints to do with officer numbers and time
        cons_availability="default",  # mode for consumable constraints (if ignored, all consumables available)
        ignore_priority=False,  # do not use the priority information in HSI event to schedule
        capabilities_coefficient=1.0,  # multiplier for the capabilities of health officers
        use_funded_or_actual_staffing="funded_plus",  # actual: use numbers/distribution of staff available currently
        disable=False,  # disables the healthsystem (no constraints and no logging) and every HSI runs
        disable_and_reject_all=False,  # disable healthsystem and no HSI runs
    ),
    symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
    healthburden.HealthBurden(resourcefilepath=resourcefilepath),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
    epi.Epi(resourcefilepath=resourcefilepath))

sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

print("--- %s seconds ---" % (time.time() - start_time))

# %% read the results

# ------------------------------------- MODEL OUTPUTS  ------------------------------------- #

output = parse_log_file(sim.log_filepath)
model_vax_coverage = output["tlo.methods.epi"]["ep_vaccine_coverage"]
model_date = pd.to_datetime(model_vax_coverage.date)
model_date = model_date.apply(lambda x: x.year)

# ------------------------------------- DATA  ------------------------------------- #
# import vaccine coverage data
workbook = pd.read_excel(os.path.join(resourcefilepath,
                                      'ResourceFile_EPI_WHO_estimates.xlsx'), sheet_name=None)

coverage_data = workbook["WHO_estimates"]

# select years included in simulation
# end_date +1 to get the final value
coverage_data2010 = coverage_data.loc[
    (coverage_data.Year >= 2010) & (coverage_data.Year < (end_date.year + 1))
]
coverage_data2010_years = pd.to_datetime(coverage_data2010.Year, format="%Y")
# coverage_data2010_years = coverage_data2010_years.values

coverage_data2010_years = coverage_data2010_years.apply(lambda x: x.year)

# ------------------------------------- PLOTS  ------------------------------------- #

plt.style.use("ggplot")
fontsize = 9


# BCG coverage
plt.subplot(331)  # numrows, numcols, fignum
plt.plot(coverage_data2010_years, coverage_data2010.BCG*100)
plt.plot(model_date, model_vax_coverage.epBcgCoverage)
plt.title("BCG", fontsize=fontsize)
plt.ylabel("Coverage", fontsize=fontsize)
plt.gca().set_xlim(2010, 2025)
plt.gca().set_ylim(0, 110)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

# DTP3 coverage
plt.subplot(332)  # numrows, numcols, fignum
plt.plot(coverage_data2010_years, coverage_data2010.DTP3*100)
plt.plot(model_date, model_vax_coverage.epDtp3Coverage)
plt.title("DTP3", fontsize=fontsize)
plt.gca().set_xlim(2010, 2025)
plt.gca().set_ylim(0, 110)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

# OPV3 coverage
plt.subplot(333)  # numrows, numcols, fignum
plt.plot(coverage_data2010_years, coverage_data2010.Pol3*100)
plt.plot(model_date, model_vax_coverage.epOpv3Coverage)
plt.title("OPV3", fontsize=fontsize)
plt.gca().set_xlim(2010, 2025)
plt.gca().set_ylim(0, 110)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.legend(["WHO", "Model"])

# Hib3 coverage
plt.subplot(334)  # numrows, numcols, fignum
plt.plot(coverage_data2010_years, coverage_data2010.Hib3*100)
plt.plot(model_date, model_vax_coverage.epHib3Coverage)
plt.title("Hib3", fontsize=fontsize)
plt.ylabel("Coverage", fontsize=fontsize)
plt.gca().set_xlim(2010, 2025)
plt.gca().set_ylim(0, 110)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

# Hep3 coverage
plt.subplot(335)  # numrows, numcols, fignum
plt.plot(coverage_data2010_years, coverage_data2010.HepB3*100)
plt.plot(model_date, model_vax_coverage.epHep3Coverage)
plt.title("Hep3", fontsize=fontsize)
plt.gca().set_xlim(2010, 2025)
plt.gca().set_ylim(0, 110)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

# Pneumo3 coverage
plt.subplot(336)  # numrows, numcols, fignum
plt.plot(coverage_data2010_years, coverage_data2010.PCV3*100)
plt.plot(model_date, model_vax_coverage.epPneumo3Coverage)
plt.title("Pneumo3", fontsize=fontsize)
plt.gca().set_xlim(2010, 2025)
plt.gca().set_ylim(0, 110)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

# Rotavirus2 coverage
plt.subplot(337)  # numrows, numcols, fignum
plt.plot(coverage_data2010_years, coverage_data2010.Rota*100)
plt.plot(model_date, model_vax_coverage.epRota2Coverage)
plt.title("Rotavirus2", fontsize=fontsize)
plt.ylabel("Coverage", fontsize=fontsize)
plt.gca().set_xlim(2010, 2025)
plt.gca().set_ylim(0, 110)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

# Measles coverage (1 dose recommended up to 2018)
plt.subplot(338)  # numrows, numcols, fignum
plt.plot(coverage_data2010_years, coverage_data2010.MCV1*100)
plt.plot(model_date, model_vax_coverage.epMeaslesCoverage)
plt.title("Measles/Rubella", fontsize=fontsize)
plt.gca().set_xlim(2010, 2025)
plt.gca().set_ylim(0, 110)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

# Measles coverage (2 doses recommended from 2018)
plt.subplot(339)  # numrows, numcols, fignum
plt.plot(coverage_data2010_years, coverage_data2010.MCV2*100)
plt.plot(model_date, model_vax_coverage.epMeasles2Coverage)
plt.title("Measles/Rubella 2 dose", fontsize=fontsize)
plt.gca().set_xlim(2010, 2025)
plt.gca().set_ylim(0, 110)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

# set the spacing between subplots
plt.subplots_adjust(wspace=0.4,
                    hspace=0.6)

plt.show()


# # ---------------------- plot vaccine delivery across facility levels -------------------
facilities = output["tlo.methods.healthsystem.summary"]["HSI_Event"]

tmp = facilities.Number_By_Appt_Type_Code_And_Level

t1 = pd.DataFrame(tmp.values.tolist())
t2 = t1.set_axis(["level0", "level1a", "level1b", "level2", "level3", "level4"], axis=1)

epi = pd.DataFrame(columns=["level0", "level1a", "level1b", "level2", "level3", "level4"])

for i in range(len(t2.index)):
    out = [d.get('EPI') for d in t2.iloc[i]]
    epi.loc[i] = out

print(epi)

total_epi_by_facility_level = epi.sum()
total_epi = total_epi_by_facility_level.sum()

colours = ['#B7C3F3', '#DD7596', '#8EB897', '#FFF68F']

plt.rcParams["axes.titlesize"] = 9

# calculate proportion of childhood vaccines delivered by facility level
level0 = total_epi_by_facility_level['level0'] / total_epi
level1a = total_epi_by_facility_level['level1a'] / total_epi
level1b = total_epi_by_facility_level['level1b'] / total_epi
level2 = total_epi_by_facility_level['level2'] / total_epi

ax = plt.subplot(111)  # numrows, numcols, fignum
plt.pie([level0, level1a, level1b, level2], labels=['level 0', 'level 1a', 'level 1b', 'level 2'],
        wedgeprops={'linewidth': 3, 'edgecolor': 'white'},
        autopct='%.1f%%',
        colors=colours)
plt.title("Facility level giving childhood vaccines")
plt.show()
