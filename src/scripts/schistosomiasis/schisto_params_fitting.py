import datetime
import logging
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.dates import DateFormatter
from pathlib import Path

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    healthburden,
    healthsystem,
    contraception,
    schisto
)
import json
import sys

INFECTION_TYPE = 'mansoni'
OUTPUT_PATH = 'C:/Users/ieh19/Desktop/Project 1/model_outputs/params_fitting_mansoni/'
def run_simulation(infection_type, alpha, r0):
    outputpath = Path("./outputs")  # folder for convenience of storing outputs
    datestamp = datetime.datetime.now().strftime("__%Y_%m_%d_%H_%M")

    # The resource files
    resourcefilepath = Path("./resources")
    start_date = Date(2010, 1, 1)
    end_date = Date(2020, 2, 1)
    popsize = 10000

    # Establish the simulation object
    sim = Simulation(start_date=start_date)

    # Establish the logger
    logfile = outputpath / ('LogFile' + datestamp + '.log')

    if os.path.exists(logfile):
        os.remove(logfile)
    fh = logging.FileHandler(logfile)
    fr = logging.Formatter("%(levelname)s|%(name)s|%(message)s")
    fh.setFormatter(fr)
    logging.getLogger().addHandler(fh)


    logging.getLogger("tlo.methods.demography").setLevel(logging.WARNING)
    logging.getLogger("tlo.methods.contraception").setLevel(logging.WARNING)
    logging.getLogger("tlo.methods.healthburden").setLevel(logging.WARNING)
    logging.getLogger("tlo.methods.healthsystem").setLevel(logging.WARNING)
    logging.getLogger("tlo.methods.schisto").setLevel(logging.INFO)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath))
    sim.register(healthsystem.HealthSystem(resourcefilepath=resourcefilepath))
    sim.register(healthburden.HealthBurden(resourcefilepath=resourcefilepath))
    sim.register(contraception.Contraception(resourcefilepath=resourcefilepath))
    sim.register(schisto.Schisto(resourcefilepath=resourcefilepath))
    if infection_type == 'haematobium':
        sim.register(schisto.Schisto_Haematobium(resourcefilepath=resourcefilepath, alpha=alpha, r0=r0))
    if infection_type == 'mansoni':
        sim.register(schisto.Schisto_Mansoni(resourcefilepath=resourcefilepath, alpha=alpha, r0=r0))

    # Run the simulation and flush the logger
    sim.seed_rngs(0)
    sim.make_initial_population(n=popsize)

    # start the simulation
    sim.simulate(end_date=end_date)
    fh.flush()
    output = parse_log_file(logfile)

    return sim, output

def run_sims_and_save(infection_type, a, ro, ii, jj):
    sim, output = run_simulation(infection_type, a, ro)
    global districts

    district_prevalence = get_prev_per_districts(infection_type, output, districts)
    districts_mwb = get_mwb_per_districts(infection_type, output, districts)

    output_path = OUTPUT_PATH
    save_str_prev = output_path + 'prev_r0=' + str(ro) + '_alpha=' + str(a) + '.json'
    save_str_mwb = output_path + 'mwb_r0=' + str(ro) + '_alpha=' + str(a) + '.json'
    json_prev = json.dumps(district_prevalence)
    f = open(save_str_prev, "w")
    f.write(json_prev)
    f.close()
    json_mwb = json.dumps(districts_mwb)
    f = open(save_str_mwb, "w")
    f.write(json_mwb)
    f.close()

    for d in districts:
        global all_prev_district
        global all_mwb_district
        d_prevs = all_prev_district[d]
        d_prevs[ii, jj] = district_prevalence[d]
        all_prev_district.update({d: d_prevs})
        d_mwb = all_mwb_district[d]
        d_mwb[ii, jj] = districts_mwb[d]
        all_mwb_district.update({d: d_mwb})

def get_prev_per_districts(infection_type, output, districts):
    districts_prevalence = {}
    for distr in districts:
        prev = output['tlo.methods.schisto'][distr + '_' + infection_type.capitalize()].Prevalence.values[-1]
        districts_prevalence.update({distr: prev})
    return districts_prevalence

def get_mwb_per_districts(infection_type, output, districts):
    districts_mwb = {}
    for distr in districts:
        mwb = output['tlo.methods.schisto'][distr + '_' + infection_type.capitalize()].MeanWormBurden.values[-1]
        districts_mwb.update({distr: mwb})
    return districts_mwb

r0 = np.arange(0.25, 1.5, 0.25)
alpha = np.arange(0.01, 0.07, 0.01)
districts = ['Mangochi', 'Lilongwe City', 'Balaka', 'Lilongwe', 'Kasungu', 'Mzimba', 'Chitipa', 'Mulanje',
             'Zomba', 'Dowa', 'Blantyre City', 'Ntcheu', 'Mzuzu City', 'Nsanje', 'Phalombe', 'Nkhata Bay',
             'Chiradzulu', 'Thyolo', 'Blantyre', 'Chikwawa', 'Salima', 'Dedza', 'Nkhotakota', 'Neno',
             'Karonga', 'Zomba City', 'Mchinji', 'Machinga', 'Ntchisi', 'Rumphi', 'Mwanza', 'Likoma']
all_prev_district = {}
all_mwb_district = {}

for d in districts:
    all_prev_district.update({d: np.zeros((len(r0), len(alpha)))})
    all_mwb_district.update({d: np.zeros((len(r0), len(alpha)))})

# run simulations - this takes a lot of time!
old_stdout = sys.stdout
for ii in range(len(r0)):
    for jj in range(len(alpha)):
        ro = round(r0[ii], 2)
        a = round(alpha[jj], 2)  # for some reason it was extending the floats
        sys.stdout = old_stdout
        print('SIMULATION STARTS', ro, a)
        f = open('nul', 'w')
        sys.stdout = f

        run_sims_and_save(INFECTION_TYPE, a, ro, ii, jj)

# save final outputs in an excel file
writer = pd.ExcelWriter(OUTPUT_PATH + 'ParameterFitting.xlsx')
for distr in districts:
    index = r0
    columns = alpha
    pd.DataFrame(columns=columns, index=index, data=all_prev_district[distr]).to_excel(writer, sheet_name=distr+'_prev')
    pd.DataFrame(columns=columns, index=index, data=all_mwb_district[distr]).to_excel(writer, sheet_name=distr+'_mwb')
writer.save()

# analyse the outputs
# read in the baseline prevalence
sys.stdout = old_stdout
resourcefilepath = Path("./resources/ResourceFile_Schisto.xlsx")
baseline_prev = pd.read_excel(resourcefilepath, sheet_name='District_Params_' + INFECTION_TYPE)
baseline_prev.set_index("District", inplace=True)
baseline_prev = baseline_prev.loc[:, 'Prevalence']
baseline_prev = baseline_prev.to_dict()

# read in the simulations outputs
# simulated_results_lower = pd.read_excel(output_path + 'ParameterFitting.xlsx', sheet_name=None)
simulated_results = pd.read_excel(OUTPUT_PATH + 'ParameterFitting.xlsx', sheet_name=None)
# for k in simulated_results.keys():
#     simulated_results[k] = simulated_results[k].join(simulated_results_lower[k])
#     simulated_results[k] = simulated_results[k].reindex(sorted(simulated_results[k].columns), axis=1)
diff_in_prev_all_districts = {}
writer = pd.ExcelWriter(OUTPUT_PATH + 'ParamsFittingPrevDifferenceAllAlphas.xlsx')

# calculate the difference between baseline and simulated prevalence and save in a file
# save in a dictionary the best pairs of (r0, alpha) that gives the lowest absolute difference
fitted_params = pd.DataFrame(columns=['district', 'R0', 'alpha'])
for d in districts:
    base_prev_expected = baseline_prev[d]
    simulated_prev = simulated_results[(d + '_prev')]
    diff_in_prev = abs(simulated_prev.loc[:, :] - base_prev_expected)
    r0_min, alpha_min = diff_in_prev.stack().idxmin()
    print(d, 'alpha=', alpha_min, 'r0=', r0_min)
    new_row = pd.DataFrame({'district': [d],
                            'R0': [r0_min],
                            'alpha': [alpha_min]})
    fitted_params = fitted_params.append(new_row)
    diff_in_prev_all_districts.update({d: diff_in_prev})
    diff_in_prev.to_excel(writer, sheet_name=d)
writer.save()
fitted_params.to_csv(OUTPUT_PATH + 'FittedParams.csv', index=False)

