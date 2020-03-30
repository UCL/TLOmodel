from pathlib import Path
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    contraception,
    healthburden,
    healthsystem,
    enhanced_lifestyle,
    epi,
)

start_time = time.time()

# Where will output go
outputpath = Path("./outputs/epi")

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

start_date = Date(2010, 1, 1)
end_date = Date(2020, 12, 31)
popsize = 1000

# Establish the simulation object
sim = Simulation(start_date=start_date)

# ----- Control over the types of intervention that can occur -----
# Make a list that contains the treatment_id that will be allowed. Empty list means nothing allowed.
# '*' means everything. It will allow any treatment_id that begins with a stub (e.g. Mockitis*)
service_availability = ["*"]

# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(
    healthsystem.HealthSystem(
        resourcefilepath=resourcefilepath,
        service_availability=service_availability,
        mode_appt_constraints=2,  # no constraints by officer type/time
        ignore_cons_constraints=True,
        ignore_priority=True,
        capabilities_coefficient=1.0,
        disable=True,
    )
)  # disables the health system constraints so all HSI events run
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
sim.register(epi.Epi(resourcefilepath=resourcefilepath))

# Sets all modules to WARNING threshold, then alter epi to INFO
custom_levels = {"*": logging.WARNING, "tlo.methods.epi": logging.INFO}

# configure_logging automatically appends datetime
logfile = sim.configure_logging(filename="Epi_LogFile", custom_levels=custom_levels)

# Run the simulation and flush the logger
sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

print("--- %s seconds ---" % (time.time() - start_time))

# %% read the results

# ------------------------------------- MODEL OUTPUTS  ------------------------------------- #

output = parse_log_file(logfile)
model_vax_coverage = output["tlo.methods.epi"]["ep_vaccine_coverage"]
model_date = pd.to_datetime(model_vax_coverage.date)
model_date = model_date.apply(lambda x: x.year)

# ------------------------------------- DATA  ------------------------------------- #
# import vaccine coverage data
coverage_data = pd.read_excel(
    Path(resourcefilepath) / "ResourceFile_EPI.xlsx", sheet_name="WHO_Estimates",
)

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
plt.plot(coverage_data2010_years, coverage_data2010.BCG)
plt.plot(model_date, model_vax_coverage.epBcgCoverage)
plt.title("BCG vaccine coverage")
plt.xlabel("Year")
plt.ylabel("Coverage")
plt.xticks(rotation=90)
plt.gca().set_xlim(2010, 2025)
plt.gca().set_ylim(0, 110)
plt.legend(["WHO", "Model"], bbox_to_anchor=(1.04, 1), loc="upper left")

# DTP3 coverage
plt.subplot(222)  # numrows, numcols, fignum
plt.plot(coverage_data2010_years, coverage_data2010.DTP3)
plt.plot(model_date, model_vax_coverage.epDtp3Coverage)
plt.title("DTP3 vaccine coverage")
plt.xlabel("Year")
plt.ylabel("Coverage")
plt.xticks(rotation=90)
plt.gca().set_xlim(2010, 2025)
plt.gca().set_ylim(0, 110)
plt.legend(["WHO", "Model"], bbox_to_anchor=(1.04, 1), loc="upper left")

# OPV3 coverage
plt.subplot(223)  # numrows, numcols, fignum
plt.plot(coverage_data2010_years, coverage_data2010.Pol3)
plt.plot(model_date, model_vax_coverage.epOpv3Coverage)
plt.title("OPV3 vaccine coverage")
plt.xlabel("Year")
plt.ylabel("Coverage")
plt.xticks(rotation=90)
plt.gca().set_xlim(2010, 2025)
plt.gca().set_ylim(0, 110)
plt.legend(["WHO", "Model"], bbox_to_anchor=(1.04, 1), loc="upper left")

# Hib3 coverage
plt.subplot(224)  # numrows, numcols, fignum
plt.plot(coverage_data2010_years, coverage_data2010.Hib3)
plt.plot(model_date, model_vax_coverage.epHib3Coverage)
plt.title("Hib3 vaccine coverage")
plt.xlabel("Year")
plt.ylabel("Coverage")
plt.xticks(rotation=90)
plt.gca().set_xlim(2010, 2025)
plt.gca().set_ylim(0, 110)
plt.legend(["WHO", "Model"], bbox_to_anchor=(1.04, 1), loc="upper left")
plt.show()

# Hep3 coverage
plt.subplot(221)  # numrows, numcols, fignum
plt.plot(coverage_data2010_years, coverage_data2010.HepB3)
plt.plot(model_date, model_vax_coverage.epHep3Coverage)
plt.title("Hep3 vaccine coverage")
plt.xlabel("Year")
plt.ylabel("Coverage")
plt.xticks(rotation=90)
plt.gca().set_xlim(2010, 2025)
plt.gca().set_ylim(0, 110)
plt.legend(["WHO", "Model"], bbox_to_anchor=(1.04, 1), loc="upper left")

# Pneumo3 coverage
plt.subplot(222)  # numrows, numcols, fignum
plt.plot(coverage_data2010_years, coverage_data2010.PCV3)
plt.plot(model_date, model_vax_coverage.epPneumo3Coverage)
plt.title("Pneumo3 vaccine coverage")
plt.xlabel("Year")
plt.ylabel("Coverage")
plt.xticks(rotation=90)
plt.gca().set_xlim(2010, 2025)
plt.gca().set_ylim(0, 110)
plt.legend(["WHO", "Model"], bbox_to_anchor=(1.04, 1), loc="upper left")

# Rotavirus2 coverage
plt.subplot(223)  # numrows, numcols, fignum
plt.plot(coverage_data2010_years, coverage_data2010.Rota)
plt.plot(model_date, model_vax_coverage.epRota2Coverage)
plt.title("Rotavirus2 vaccine coverage")
plt.xlabel("Year")
plt.ylabel("Coverage")
plt.xticks(rotation=90)
plt.gca().set_xlim(2010, 2025)
plt.gca().set_ylim(0, 110)
plt.legend(["WHO", "Model"], bbox_to_anchor=(1.04, 1), loc="upper left")

# Measles coverage (1 dose)
plt.subplot(224)  # numrows, numcols, fignum
plt.plot(coverage_data2010_years, coverage_data2010.MCV1)
plt.plot(model_date, model_vax_coverage.epMeaslesCoverage)
plt.title("Measles vaccine coverage")
plt.xlabel("Year")
plt.ylabel("Coverage")
plt.xticks(rotation=90)
plt.gca().set_xlim(2010, 2025)
plt.gca().set_ylim(0, 110)
plt.legend(["WHO", "Model"], bbox_to_anchor=(1.04, 1), loc="upper left")

plt.show()
