import datetime
import logging
import os
import time
from pathlib import Path

from tlo import Date, Simulation
from tlo.methods import (
    demography,
    contraception,
    healthburden,
    healthsystem,
    enhanced_lifestyle,
    malaria,
    dx_algorithm_child,
    dx_algorithm_adult,
    healthseekingbehaviour,
    symptommanager,
)

t0 = time.time()

# Where will output go
outputpath = './src/scripts/outputs/'

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

start_date = Date(2010, 1, 1)
end_date = Date(2025, 12, 31)
popsize = 50000

# Establish the simulation object
sim = Simulation(start_date=start_date)

# TODO change the seed + filepath + itn for each simulation
sim.seed_rngs(72)
logfile = outputpath + 'Malaria3_ITN0.7' + datestamp + '.log'

if os.path.exists(logfile):
    os.remove(logfile)
fh = logging.FileHandler(logfile)
fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
fh.setFormatter(fr)
logging.getLogger().addHandler(fh)

# ----- Control over the types of intervention that can occur -----
# Make a list that contains the treatment_id that will be allowed. Empty list means nothing allowed.
# '*' means everything. It will allow any treatment_id that begins with a stub (e.g. Mockitis*)
service_availability = ['*']
malaria_strat = 1  # levels: 0 = national; 1 = district
malaria_testing = 0.35  # adjust this to match rdt/tx levels
itn = 0.7

# Register the appropriate modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath))
sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                       service_availability=service_availability,
                                       mode_appt_constraints=0,
                                       ignore_cons_constraints=True,
                                       ignore_priority=True,
                                       capabilities_coefficient=1.0,
                                       disable=True))  # disables the health system constraints so all HSI events run
sim.register(symptommanager.SymptomManager(resourcefilepath=resourcefilepath))
sim.register(healthseekingbehaviour.HealthSeekingBehaviour())
sim.register(dx_algorithm_child.DxAlgorithmChild())
sim.register(dx_algorithm_adult.DxAlgorithmAdult())
sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
sim.register(enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath))
sim.register(malaria.Malaria(resourcefilepath=resourcefilepath,
                             level=malaria_strat, testing=malaria_testing, itn=itn))

for name in logging.root.manager.loggerDict:
    if name.startswith("tlo"):
        logging.getLogger(name).setLevel(logging.WARNING)

logging.getLogger('tlo.methods.malaria').setLevel(logging.INFO)

# Run the simulation and flush the logger
# sim.seed_rngs(0)
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)
fh.flush()
# fh.close()

t1 = time.time()
print('Time taken', t1 - t0)

# ---------------------------------------- PLOTS ---------------------------------------- #
# %% read the results
# from tlo.analysis.utils import parse_log_file
# import datetime
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
#
# datestamp = datetime.date.today().strftime("__%Y_%m_%d")
#
# # --------------------------------------- import files --------------------------------------- #
# outputpath = './src/scripts/outputs/'
# baseline_file1 = outputpath + 'Malaria_Baseline1__2020_01_28' + '.log'
# baseline_file2 = outputpath + 'Malaria_Baseline2__2020_01_28' + '.log'
# baseline_file3 = outputpath + 'Malaria_Baseline3__2020_01_28' + '.log'
#
# itn07file1 = outputpath + 'Malaria1_ITN0.7__2020_01_28' + '.log'
# itn07file2 = outputpath + 'Malaria2_ITN0.7__2020_01_28' + '.log'
# itn07file3 = outputpath + 'Malaria3_ITN0.7__2020_01_28' + '.log'
#
# itn08file1 = outputpath + 'Malaria1_ITN0.8__2020_01_28' + '.log'
# itn08file2 = outputpath + 'Malaria2_ITN0.8__2020_01_28' + '.log'
# itn08file3 = outputpath + 'Malaria3_ITN0.8__2020_01_28' + '.log'
#
# itn09file1 = outputpath + 'Malaria1_ITN0.9__2020_01_28' + '.log'
# itn09file2 = outputpath + 'Malaria2_ITN0.9__2020_01_28' + '.log'
# itn09file3 = outputpath + 'Malaria3_ITN0.9__2020_01_28' + '.log'
#
# baseline1 = parse_log_file(baseline_file1)
# baseline2 = parse_log_file(baseline_file2)
# baseline3 = parse_log_file(baseline_file3)
#
# itn071 = parse_log_file(itn07file1)
# itn072 = parse_log_file(itn07file2)
# itn073 = parse_log_file(itn07file3)
#
# itn081 = parse_log_file(itn08file1)
# itn082 = parse_log_file(itn08file2)
# itn083 = parse_log_file(itn08file3)
#
# itn091 = parse_log_file(itn09file1)
# itn092 = parse_log_file(itn09file2)
# itn093 = parse_log_file(itn09file3)
#
# # plot variables
# # INCIDENCE
# inc_baseline1 = baseline1['tlo.methods.malaria']['incidence']
# inc_baseline2 = baseline2['tlo.methods.malaria']['incidence']
# inc_baseline3 = baseline3['tlo.methods.malaria']['incidence']
#
# inc_itn071 = itn071['tlo.methods.malaria']['incidence']
# inc_itn072 = itn072['tlo.methods.malaria']['incidence']
# inc_itn073 = itn073['tlo.methods.malaria']['incidence']
#
# inc_itn081 = itn081['tlo.methods.malaria']['incidence']
# inc_itn082 = itn082['tlo.methods.malaria']['incidence']
# inc_itn083 = itn083['tlo.methods.malaria']['incidence']
#
# inc_itn091 = itn091['tlo.methods.malaria']['incidence']
# inc_itn092 = itn092['tlo.methods.malaria']['incidence']
# inc_itn093 = itn093['tlo.methods.malaria']['incidence']
#
# # PFPR
# pfpr_baseline1 = baseline1['tlo.methods.malaria']['prevalence']
# pfpr_baseline2 = baseline2['tlo.methods.malaria']['prevalence']
# pfpr_baseline3 = baseline3['tlo.methods.malaria']['prevalence']
#
# pfpr_itn071 = itn071['tlo.methods.malaria']['prevalence']
# pfpr_itn072 = itn072['tlo.methods.malaria']['prevalence']
# pfpr_itn073 = itn073['tlo.methods.malaria']['prevalence']
#
# pfpr_itn081 = itn081['tlo.methods.malaria']['prevalence']
# pfpr_itn082 = itn082['tlo.methods.malaria']['prevalence']
# pfpr_itn083 = itn083['tlo.methods.malaria']['prevalence']
#
# pfpr_itn091 = itn091['tlo.methods.malaria']['prevalence']
# pfpr_itn092 = itn092['tlo.methods.malaria']['prevalence']
# pfpr_itn093 = itn093['tlo.methods.malaria']['prevalence']
#
# # MORTALITY
# mort_baseline1 = baseline1['tlo.methods.malaria']['ma_mortality']
# mort_baseline2 = baseline2['tlo.methods.malaria']['ma_mortality']
# mort_baseline3 = baseline3['tlo.methods.malaria']['ma_mortality']
#
# mort_itn071 = itn071['tlo.methods.malaria']['ma_mortality']
# mort_itn072 = itn072['tlo.methods.malaria']['ma_mortality']
# mort_itn073 = itn073['tlo.methods.malaria']['ma_mortality']
#
# mort_itn081 = itn081['tlo.methods.malaria']['ma_mortality']
# mort_itn082 = itn082['tlo.methods.malaria']['ma_mortality']
# mort_itn083 = itn083['tlo.methods.malaria']['ma_mortality']
#
# mort_itn091 = itn091['tlo.methods.malaria']['ma_mortality']
# mort_itn092 = itn092['tlo.methods.malaria']['ma_mortality']
# mort_itn093 = itn093['tlo.methods.malaria']['ma_mortality']
#
# # START / END DATES
# model_years = pd.to_datetime(inc_baseline1.date)
# model_years = model_years.dt.year
# start_date = 2010
# end_date = 2025
#
# # --------------------------------------- get averages --------------------------------------- #
# # INCIDENCE
# inc_baseline = np.mean(
#     [inc_baseline1.inc_clin_counter,
#      inc_baseline2.inc_clin_counter,
#      inc_baseline3.inc_clin_counter],
#     axis=0)
#
# inc_itn07 = np.mean(
#     [inc_itn071.inc_clin_counter,
#      inc_itn072.inc_clin_counter,
#      inc_itn073.inc_clin_counter],
#     axis=0)
#
# inc_itn08 = np.mean(
#     [inc_itn081.inc_clin_counter,
#      inc_itn082.inc_clin_counter,
#      inc_itn083.inc_clin_counter],
#     axis=0)
#
# inc_itn09 = np.mean(
#     [inc_itn091.inc_clin_counter,
#      inc_itn092.inc_clin_counter,
#      inc_itn093.inc_clin_counter],
#     axis=0)
#
# # PFPR
# pfpr_baseline = np.mean(
#     [pfpr_baseline1.child2_10_prev,
#      pfpr_baseline2.child2_10_prev,
#      pfpr_baseline3.child2_10_prev],
#     axis=0)
#
# pfpr_itn07 = np.mean(
#     [pfpr_itn071.child2_10_prev,
#      pfpr_itn072.child2_10_prev,
#      pfpr_itn073.child2_10_prev],
#     axis=0)
#
# pfpr_itn08 = np.mean(
#     [pfpr_itn081.child2_10_prev,
#      pfpr_itn082.child2_10_prev,
#      pfpr_itn083.child2_10_prev],
#     axis=0)
#
# pfpr_itn09 = np.mean(
#     [pfpr_itn091.child2_10_prev,
#      pfpr_itn092.child2_10_prev,
#      pfpr_itn093.child2_10_prev],
#     axis=0)
#
# # MORTALITY
# mort_baseline = np.mean(
#     [mort_baseline1.mort_rate,
#      mort_baseline2.mort_rate,
#      mort_baseline3.mort_rate],
#     axis=0)
#
# mort_itn07 = np.mean(
#     [mort_itn071.mort_rate,
#      mort_itn072.mort_rate,
#      mort_itn073.mort_rate],
#     axis=0)
#
# mort_itn08 = np.mean(
#     [mort_itn081.mort_rate,
#      mort_itn082.mort_rate,
#      mort_itn083.mort_rate],
#     axis=0)
#
# mort_itn09 = np.mean(
#     [mort_itn091.mort_rate,
#      mort_itn092.mort_rate,
#      mort_itn093.mort_rate],
#     axis=0)
# # --------------------------------------- create plots --------------------------------------- #
# ## FIGURES
# plt.figure(1, figsize=(8, 6))
#
# # Malaria incidence per 1000py - all ages
# ax = plt.subplot(311)  # numrows, numcols, fignum
# plt.plot(model_years, inc_baseline)  # baseline
# plt.plot(model_years, inc_itn07)  # itn 0.7
# plt.plot(model_years, inc_itn08)  # itn 0.8
# plt.plot(model_years, inc_itn09)  # itn 0.9
# plt.axvline(x=2020, color='grey', linestyle='--')
# plt.title("Malaria Inc / 1000py")
# plt.xlabel("Year")
# plt.ylabel("Incidence (/1000py)")
# plt.xticks(rotation=90)
# plt.gca().set_xlim(start_date, end_date)
# plt.legend(["Baseline", "ITN 0.7", "ITN 0.8", "ITN 0.9"],
#            bbox_to_anchor=(1.04, 1), loc="upper left")
# plt.tight_layout()
#
# # Malaria parasite prevalence rate - 2-10 year olds
# ax2 = plt.subplot(312)  # numrows, numcols, fignum
# plt.plot(model_years, pfpr_baseline)  # baseline
# plt.plot(model_years, pfpr_itn07)  # itn 0.7
# plt.plot(model_years, pfpr_itn08)  # itn 0.8
# plt.plot(model_years, pfpr_itn09)  # itn 0.9
# plt.axvline(x=2020, color='grey', linestyle='--')
# plt.title("Malaria PfPR 2-10 yrs")
# plt.xlabel("Year")
# plt.ylabel("PfPR (%)")
# plt.xticks(rotation=90)
# plt.gca().set_xlim(start_date, end_date)
# plt.legend(["Baseline", "ITN 0.7", "ITN 0.8", "ITN 0.9"],
#            bbox_to_anchor=(1.04, 1), loc="upper left")
# plt.tight_layout()
#
# # Malaria mortality rate - all ages
# ax3 = plt.subplot(313)  # numrows, numcols, fignum
# plt.plot(model_years, mort_baseline)  # baseline
# plt.plot(model_years, mort_itn07)  # itn 0.7
# plt.plot(model_years, mort_itn08)  # itn 0.8
# plt.plot(model_years, mort_itn09)  # itn 0.9
# plt.axvline(x=2020, color='grey', linestyle='--')
# plt.title("Malaria Mortality Rate")
# plt.xlabel("Year")
# plt.ylabel("Mortality rate")
# plt.xticks(rotation=90)
# plt.gca().set_xlim(start_date, end_date)
# plt.legend(["Baseline", "ITN 0.7", "ITN 0.8", "ITN 0.9"],
#            bbox_to_anchor=(1.04, 1), loc="upper left")
# plt.tight_layout()
#
# out_path = '//fi--san02/homes/tmangal/Thanzi la Onse/Malaria/model_outputs/ITN_projections_28Jan2010/'
# figpath = out_path + "ITN_projections_averages" + datestamp + ".png"
# plt.savefig(figpath, bbox_inches='tight')
#
# plt.show()
# plt.close()
