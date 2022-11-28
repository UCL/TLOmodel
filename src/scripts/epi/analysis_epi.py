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
end_date = Date(2014, 12, 31)
popsize = 500

log_config = {
    'filename': 'Epi_LogFile',
    'custom_levels': {"*": logging.WARNING, "tlo.methods.epi": logging.INFO}
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
        service_availability=service_availability,
        mode_appt_constraints=0,
        ignore_cons_constraints=True,
        ignore_priority=True,
        capabilities_coefficient=1.0,
        disable=True,
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

# BCG coverage
plt.subplot(221)  # numrows, numcols, fignum
plt.plot(coverage_data2010_years, coverage_data2010.BCG*100)
plt.plot(model_date, model_vax_coverage.epBcgCoverage)
plt.title("BCG vaccine coverage", fontsize=11)
# plt.xlabel("Year")
plt.ylabel("Coverage")
# plt.xticks(rotation=90)
plt.gca().set_xlim(2010, 2025)
plt.gca().set_ylim(0, 110)
# plt.legend(["WHO", "Model"], bbox_to_anchor=(1.04, 1), loc="upper left")

# DTP3 coverage
plt.subplot(222)  # numrows, numcols, fignum
plt.plot(coverage_data2010_years, coverage_data2010.DTP3*100)
plt.plot(model_date, model_vax_coverage.epDtp3Coverage)
plt.title("DTP3 vaccine coverage", fontsize=11)
# plt.xlabel("Year")
# plt.ylabel("Coverage")
# plt.xticks(rotation=90)
plt.gca().set_xlim(2010, 2025)
plt.gca().set_ylim(0, 110)
plt.legend(["WHO", "Model"], bbox_to_anchor=(1.04, 1), loc="upper right")

# OPV3 coverage
plt.subplot(223)  # numrows, numcols, fignum
plt.plot(coverage_data2010_years, coverage_data2010.Pol3*100)
plt.plot(model_date, model_vax_coverage.epOpv3Coverage)
plt.title("OPV3 vaccine coverage", fontsize=11)
# plt.xlabel("Year")
plt.ylabel("Coverage")
# plt.xticks(rotation=90)
plt.gca().set_xlim(2010, 2025)
plt.gca().set_ylim(0, 110)
# plt.legend(["WHO", "Model"], bbox_to_anchor=(1.04, 1), loc="upper left")

# Hib3 coverage
plt.subplot(224)  # numrows, numcols, fignum
plt.plot(coverage_data2010_years, coverage_data2010.Hib3*100)
plt.plot(model_date, model_vax_coverage.epHib3Coverage)
plt.title("Hib3 vaccine coverage", fontsize=11)
# plt.xlabel("Year")
# plt.ylabel("Coverage")
# plt.xticks(rotation=90)
plt.gca().set_xlim(2010, 2025)
plt.gca().set_ylim(0, 110)
# plt.legend(["WHO", "Model"], bbox_to_anchor=(1.04, 1), loc="upper left")
plt.tight_layout()
plt.savefig(outputpath / ("EPI_1" + datestamp + ".pdf"), format='pdf', bbox_inches='tight')
plt.show()

# Hep3 coverage
plt.subplot(221)  # numrows, numcols, fignum
plt.plot(coverage_data2010_years, coverage_data2010.HepB3*100)
plt.plot(model_date, model_vax_coverage.epHep3Coverage)
plt.title("Hep3 vaccine coverage", fontsize=11)
# plt.xlabel("Year")
plt.ylabel("Coverage")
# plt.xticks(rotation=90)
plt.gca().set_xlim(2010, 2025)
plt.gca().set_ylim(0, 110)
# plt.legend(["WHO", "Model"], bbox_to_anchor=(1.04, 1), loc="upper left")

# Pneumo3 coverage
plt.subplot(222)  # numrows, numcols, fignum
plt.plot(coverage_data2010_years, coverage_data2010.PCV3*100)
plt.plot(model_date, model_vax_coverage.epPneumo3Coverage)
plt.title("Pneumo3 vaccine coverage", fontsize=11)
# plt.xlabel("Year")
# plt.ylabel("Coverage")
# plt.xticks(rotation=90)
plt.gca().set_xlim(2010, 2025)
plt.gca().set_ylim(0, 110)
plt.legend(["WHO", "Model"], bbox_to_anchor=(1.04, 1), loc="upper left")

# Rotavirus2 coverage
plt.subplot(223)  # numrows, numcols, fignum
plt.plot(coverage_data2010_years, coverage_data2010.Rota*100)
plt.plot(model_date, model_vax_coverage.epRota2Coverage)
plt.title("Rotavirus2 vaccine coverage", fontsize=11)
# plt.xlabel("Year")
plt.ylabel("Coverage")
# plt.xticks(rotation=90)
plt.gca().set_xlim(2010, 2025)
plt.gca().set_ylim(0, 110)
# plt.legend(["WHO", "Model"], bbox_to_anchor=(1.04, 1), loc="upper left")

# Measles coverage (1 dose)
plt.subplot(224)  # numrows, numcols, fignum
plt.plot(coverage_data2010_years, coverage_data2010.MCV1*100)
plt.plot(model_date, model_vax_coverage.epMeaslesCoverage)
plt.title("Measles vaccine coverage", fontsize=11)
# plt.xlabel("Year")
# plt.ylabel("Coverage")
# plt.xticks(rotation=90)
plt.gca().set_xlim(2010, 2025)
plt.gca().set_ylim(0, 110)
plt.legend(["WHO", "Model"], bbox_to_anchor=(1.04, 1), loc="upper left")
plt.tight_layout()

plt.show()
