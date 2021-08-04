import ast
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    dx_algorithm_adult,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    rti,
    simplified_births,
    symptommanager,
)

# =============================== Analysis description ========================================================
# This analysis file will eventually become what I use to produce the introduction to RTI paper. Here I run the model
# initally only allowing one injury per person, capturing the incidence of RTI and incidence of RTI death, calibrating
# this to the GBD estimates. I then run the model with multiple injuries and compare the outputs, the question being
# asked here is what happens to road traffic injury deaths if we allow multiple injuries to occur

# ============================================== Model run ============================================================
log_config = {
    "filename": "rti_health_system_comparison",  # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.rti": logging.INFO,
        "tlo.methods.healthsystem": logging.DEBUG,
        "tlo.methods.labour": logging.disable(logging.DEBUG)
    }
}
# The Resource files [NB. Working directory must be set to the root of TLO: TLOmodel]
resourcefilepath = Path('./resources')
save_file_path = "C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/"
# Establish the simulation object
yearsrun = 10
start_date = Date(year=2010, month=1, day=1)
end_date = Date(year=(2010 + yearsrun), month=1, day=1)
service_availability = ['*']
pop_size = 10000
nsim = 2
# Iterate over the number of simulations nsim
log_file_location = './outputs/single_injury_model_vs_multiple_injury/'
# store relevent model outputs
sing_inj_incidences_of_rti = []
sing_inj_incidences_of_death = []
sing_inj_incidences_of_injuries = []
sing_inj_cause_of_death_in_sim = []
sing_number_of_injuries = []
sing_dalys = []
sing_number_of_deaths = []
sing_inpatient_days = []
sing_percent_sought_healthcare = []
sing_number_of_surg = []
sing_number_of_consumables = []
sing_percent_perm_disability = []
sing_fraction_of_healthsystem_usage = []
sing_inj_icu_usage = []
sing_list_extrapolated_deaths = []
sing_list_extrapolated_dalys = []
sing_list_extrapolated_yld = []
sing_list_extrapolated_yll = []
for i in range(0, nsim):
    # Create the simulation object
    sim = Simulation(start_date=start_date)
    # Register the modules
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath),
        dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        rti.RTI(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
    )
    # Get the log file
    logfile = sim.configure_logging(filename="LogFile_single_injury",
                                    directory=log_file_location + "single_injury")
    # create and run the simulation
    sim.make_initial_population(n=pop_size)
    # alter the number of injuries given out
    # Injury vibes number of GBD injury category distribution:
    number_inj_data = [1, 0, 0, 0, 0, 0, 0, 0]
    sim.modules['RTI'].parameters['number_of_injured_body_regions_distribution'] = [
        [1, 2, 3, 4, 5, 6, 7, 8], number_inj_data
    ]
    imm_death = sim.modules['RTI'].parameters['imm_death_proportion_rti']
    sim.modules['RTI'].parameters['base_rate_injrti'] = sim.modules['RTI'].parameters['base_rate_injrti'] * 0.9872
    # sim.modules['RTI'].parameters['rt_emergency_care_ISS_score_cut_off'] = 1
    # Run the simulation
    sim.simulate(end_date=end_date)
    # Parse the logfile of this simulation
    log_df = parse_log_file(logfile)
    # Store the incidence of RTI per 100,000 person years in this sim
    sing_inj_incidences_of_rti.append(log_df['tlo.methods.rti']['summary_1m']['incidence of rti per 100,000'].tolist())
    # Store the incidence of death due to RTI per 100,000 person years and the sub categories in this sim
    sing_inj_incidences_of_death.append(
        log_df['tlo.methods.rti']['summary_1m']['incidence of rti death per 100,000'].tolist())
    # one injury per person implies above are equivalent
    sing_inj_incidences_of_injuries.append(
        log_df['tlo.methods.rti']['summary_1m']['incidence of rti per 100,000'].tolist())
    deaths_in_sim = log_df['tlo.methods.demography']['death']
    sing_number_of_injuries.append(
        log_df['tlo.methods.rti']['Inj_category_incidence']['number_of_injuries'].tolist())
    rti_deaths = deaths_in_sim.loc[deaths_in_sim['cause'] != 'Other']
    sing_inj_cause_of_death_in_sim.append(rti_deaths['cause'].to_list())
    dalys_df = log_df['tlo.methods.healthburden']['dalys']['Transport Injuries']
    DALYs = dalys_df.sum()
    sing_dalys.append(DALYs)
    sing_number_of_deaths.append(log_df['tlo.methods.rti']['summary_1m']['number rti deaths'].sum())
    inpatient_day_df = log_df['tlo.methods.healthsystem']['HSI_Event'].loc[
        log_df['tlo.methods.healthsystem']['HSI_Event']['TREATMENT_ID'] == 'RTI_MedicalIntervention']
    # iterate over the people in inpatient_day_df
    for person in inpatient_day_df.index:
        # Get the number of inpatient days per person, if there is a key error when trying to access inpatient days it
        # means that this patient didn't require any so append (0)
        try:
            sing_inpatient_days.append(inpatient_day_df.loc[person, 'Number_By_Appt_Type_Code']['InpatientDays'])
        except KeyError:
            sing_inpatient_days.append(0)
    hsb_log = log_df['tlo.methods.rti']['summary_1m']['percent sought healthcare'].tolist()
    percent_sought_healthcare = [i for i in hsb_log if i != 'none_injured']
    ave_percent_sought_care = np.mean(percent_sought_healthcare)
    sing_percent_sought_healthcare.append(ave_percent_sought_care)
    health_system_events = log_df['tlo.methods.healthsystem']['HSI_Event']
    rti_events = ['RTI_MedicalIntervention', 'RTI_Shock_Treatment', 'RTI_Fracture_Cast', 'RTI_Open_Fracture_Treatment',
                  'RTI_Suture', 'RTI_Burn_Management', 'RTI_Tetanus_Vaccine', 'RTI_Acute_Pain_Management',
                  'RTI_Major_Surgeries', 'RTI_Minor_Surgeries']
    rti_treatments = health_system_events.loc[health_system_events['TREATMENT_ID'].isin(rti_events)]
    list_of_appt_footprints = rti_treatments['Number_By_Appt_Type_Code'].to_list()
    num_surg = 0
    for dictionary in list_of_appt_footprints:
        if 'MajorSurg' in dictionary.keys():
            num_surg += 1
        if 'MinorSurg' in dictionary.keys():
            num_surg += 1
    sing_number_of_surg.append(num_surg)
    # get the consumables used in each simulation
    consumables_list = log_df['tlo.methods.healthsystem']['Consumables']['Item_Available'].tolist()
    # Create empty list to store the consumables used in the simulation
    consumables_list_to_dict = []
    for string in consumables_list:
        consumables_list_to_dict.append(ast.literal_eval(string))
    # Begin counting the number of consumables used in the simulation starting at 0
    number_of_consumables_in_sim = 0
    for dictionary in consumables_list_to_dict:
        number_of_consumables_in_sim += sum(dictionary.values())
    sing_number_of_consumables.append(number_of_consumables_in_sim)
    rti_demog = log_df['tlo.methods.rti']['rti_demography']
    number_in_crashes = rti_demog['males_in_rti'] + rti_demog['females_in_rti']
    number_perm_disabled = log_df['tlo.methods.rti']['summary_1m']['number permanently disabled'].iloc[-1]
    percent_perm_disabled = number_perm_disabled / number_in_crashes
    sing_percent_perm_disability.append(number_perm_disabled)
    sing_fraction_of_healthsystem_usage.append(log_df['tlo.methods.healthsystem']['Capacity']['Frac_Time_Used_Overall'])
    sing_inj_icu_usage.append(np.mean(
        [i for i in log_df['tlo.methods.rti']['summary_1m']['percent admitted to ICU or HDU'].tolist() if i !=
         'none_injured'])
    )
    # create an extrapolated estimate for the number of deaths occuring in the population
    data = pd.read_csv("resources/demography/ResourceFile_Pop_Annual_WPP.csv")
    sim_start_year = sim.start_date.year
    sim_end_year = sim.date.year
    sim_year_range = pd.Index(np.arange(sim_start_year, sim_end_year + 1))
    Data_Pop = data.groupby(by="Year")["Count"].sum()
    Data_Pop = Data_Pop.loc[sim_year_range]
    model_pop_size = log_df['tlo.methods.demography']['population']['total'].tolist()
    model_pop_size.append(len(sim.population.props.loc[sim.population.props.is_alive]))
    scaling_df = pd.DataFrame({'total': model_pop_size})
    scaling_df['pred_pop_size'] = Data_Pop.to_list()
    scaling_df['scale_for_each_year'] = scaling_df['pred_pop_size'] / scaling_df['total']
    scaling_df.index = sim_year_range
    rti_deaths = log_df['tlo.methods.demography']['death']
    # calculate the total number of rti related deaths
    # find deaths caused by RTI
    rti_deaths = rti_deaths.loc[rti_deaths['label'] == 'Transport Injuries']
    # create a column to show the year deaths occurred in
    rti_deaths['year'] = rti_deaths['date'].dt.year.to_list()
    # group by the year and count how many deaths ocurred
    rti_deaths = rti_deaths.groupby('year').count()
    # calculate extrapolated number of deaths
    rti_deaths['estimated_n_deaths'] = rti_deaths['cause'] * scaling_df.loc[rti_deaths.index, 'scale_for_each_year']
    # store the extrapolated number of deaths over the course of the sim
    sing_list_extrapolated_deaths.append(rti_deaths['estimated_n_deaths'].sum())
    dalys_df = log_df['tlo.methods.healthburden']['dalys']
    dalys_df = dalys_df.groupby('year').sum()
    dalys_df['extrapolated_dalys'] = dalys_df['Transport Injuries'] * scaling_df['scale_for_each_year']
    dalys_df = dalys_df.loc[~pd.isnull(dalys_df['extrapolated_dalys'])]
    sing_list_extrapolated_dalys.append(dalys_df['extrapolated_dalys'].sum())
    yld_df = log_df['tlo.methods.healthburden']['yld_by_causes_of_disability'].groupby('year').sum()
    yld_df['scaled_yld'] = yld_df['RTI'] * scaling_df['scale_for_each_year']
    yld_df = yld_df.dropna()
    sing_list_extrapolated_yld.append(yld_df['scaled_yld'].sum())
    yll_df = log_df['tlo.methods.healthburden']['yll_by_causes_of_death'].groupby('year').sum()
    rti_columns = [col for col in yll_df.columns if 'RTI' in col]
    yll_df['scaled_yll'] = [0.0] * len(yll_df)
    for col in rti_columns:
        yll_df['scaled_yll'] += yll_df[col] * scaling_df['scale_for_each_year']
    sing_list_extrapolated_yll.append(yll_df['scaled_yll'].sum())


mult_inj_incidences_of_rti = []
mult_inj_incidences_of_death = []
mult_inj_incidences_of_injuries = []
mult_inj_cause_of_death_in_sim = []
mult_number_of_injuries = []
mult_dalys = []
mult_number_of_deaths = []
mult_inpatient_days = []
mult_percent_sought_care = []
mult_number_of_surg = []
mult_number_of_consumables = []
mult_percent_perm_disability = []
mult_fraction_of_healthsystem_usage = []
mult_inj_icu_usage = []
mult_list_extrapolated_deaths = []
mult_list_extrapolated_dalys = []
mult_list_extrapolated_yld = []
mult_list_extrapolated_yll = []
for i in range(0, nsim):
    # Create the simulation object
    sim = Simulation(start_date=start_date)
    # Register the modules
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath),
        dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        rti.RTI(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
    )
    # Get the log file
    logfile = sim.configure_logging(filename="LogFile_multiple_injury",
                                    directory=log_file_location + "multiple_injury")
    # create and run the simulation
    sim.make_initial_population(n=pop_size)
    # sim.modules['RTI'].parameters['rt_emergency_care_ISS_score_cut_off'] = 1
    # Run the simulation
    sim.simulate(end_date=end_date)
    # Parse the logfile of this simulation
    log_df = parse_log_file(logfile)
    # Store the incidence of RTI per 100,000 person years in this sim
    mult_inj_incidences_of_rti.append(log_df['tlo.methods.rti']['summary_1m']['incidence of rti per 100,000'].tolist())
    # Store the incidence of death due to RTI per 100,000 person years and the sub categories in this sim
    mult_inj_incidences_of_death.append(
        log_df['tlo.methods.rti']['summary_1m']['incidence of rti death per 100,000'].tolist())
    mult_inj_incidences_of_injuries.append(
        log_df['tlo.methods.rti']['Inj_category_incidence']['number_of_injuries'].tolist())
    deaths_in_sim = log_df['tlo.methods.demography']['death']
    rti_deaths = deaths_in_sim.loc[deaths_in_sim['cause'] != 'Other']
    mult_inj_cause_of_death_in_sim.append(rti_deaths['cause'].to_list())
    mult_number_of_injuries.append(
        log_df['tlo.methods.rti']['Inj_category_incidence']['number_of_injuries'].tolist())
    dalys_df = log_df['tlo.methods.healthburden']['dalys']['Transport Injuries']
    DALYs = dalys_df.sum()
    mult_dalys.append(DALYs)
    mult_number_of_deaths.append(log_df['tlo.methods.rti']['summary_1m']['number rti deaths'].sum())
    inpatient_day_df = log_df['tlo.methods.healthsystem']['HSI_Event'].loc[
        log_df['tlo.methods.healthsystem']['HSI_Event']['TREATMENT_ID'] == 'RTI_MedicalIntervention']
    # iterate over the people in inpatient_day_df
    for person in inpatient_day_df.index:
        # Get the number of inpatient days per person, if there is a key error when trying to access inpatient days it
        # means that this patient didn't require any so append (0)
        try:
            mult_inpatient_days.append(inpatient_day_df.loc[person, 'Number_By_Appt_Type_Code']['InpatientDays'])
        except KeyError:
            mult_inpatient_days.append(0)
    hsb_log = log_df['tlo.methods.rti']['summary_1m']['percent sought healthcare'].tolist()
    percent_sought_healthcare = [i for i in hsb_log if i != 'none_injured']
    ave_percent_sought_care = np.mean(percent_sought_healthcare)
    mult_percent_sought_care.append(ave_percent_sought_care)
    health_system_events = log_df['tlo.methods.healthsystem']['HSI_Event']
    rti_events = ['RTI_MedicalIntervention', 'RTI_Shock_Treatment', 'RTI_Fracture_Cast', 'RTI_Open_Fracture_Treatment',
                  'RTI_Suture', 'RTI_Burn_Management', 'RTI_Tetanus_Vaccine', 'RTI_Acute_Pain_Management',
                  'RTI_Major_Surgeries', 'RTI_Minor_Surgeries']
    rti_treatments = health_system_events.loc[health_system_events['TREATMENT_ID'].isin(rti_events)]
    list_of_appt_footprints = rti_treatments['Number_By_Appt_Type_Code'].to_list()
    num_surg = 0
    for dictionary in list_of_appt_footprints:
        if 'MajorSurg' in dictionary.keys():
            num_surg += 1
        if 'MinorSurg' in dictionary.keys():
            num_surg += 1
    mult_number_of_surg.append(num_surg)
    # get the consumables used in each simulation
    consumables_list = log_df['tlo.methods.healthsystem']['Consumables']['Item_Available'].tolist()
    # Create empty list to store the consumables used in the simulation
    consumables_list_to_dict = []
    for string in consumables_list:
        consumables_list_to_dict.append(ast.literal_eval(string))
    # Begin counting the number of consumables used in the simulation starting at 0
    number_of_consumables_in_sim = 0
    for dictionary in consumables_list_to_dict:
        number_of_consumables_in_sim += sum(dictionary.values())
    mult_number_of_consumables.append(number_of_consumables_in_sim)
    rti_demog = log_df['tlo.methods.rti']['rti_demography']
    number_in_crashes = rti_demog['males_in_rti'] + rti_demog['females_in_rti']
    number_perm_disabled = log_df['tlo.methods.rti']['summary_1m']['number permanently disabled'].iloc[-1]
    percent_perm_disabled = number_perm_disabled / number_in_crashes
    mult_percent_perm_disability.append(number_perm_disabled)
    mult_fraction_of_healthsystem_usage.append(log_df['tlo.methods.healthsystem']['Capacity']['Frac_Time_Used_Overall'])
    mult_inj_icu_usage.append(np.mean(
        [i for i in log_df['tlo.methods.rti']['summary_1m']['percent admitted to ICU or HDU'].tolist() if i !=
         'none_injured']
    ))
    data = pd.read_csv("resources/demography/ResourceFile_Pop_Annual_WPP.csv")
    sim_start_year = sim.start_date.year
    sim_end_year = sim.date.year
    sim_year_range = pd.Index(np.arange(sim_start_year, sim_end_year))
    Data_Pop = data.groupby(by="Year")["Count"].sum()
    Data_Pop = Data_Pop.loc[sim_year_range]
    model_pop_size = log_df['tlo.methods.demography']['population']['total']
    scaling_df = pd.DataFrame(model_pop_size)
    scaling_df['pred_pop_size'] = Data_Pop.to_list()
    scaling_df['scale_for_each_year'] = scaling_df['pred_pop_size'] / scaling_df['total']
    scaling_df.index = sim_year_range
    rti_deaths = log_df['tlo.methods.demography']['death']
    # calculate the total number of rti related deaths
    # find deaths caused by RTI
    rti_deaths = rti_deaths.loc[rti_deaths['label'] == 'Transport Injuries']
    # create a column to show the year deaths occurred in
    rti_deaths['year'] = rti_deaths['date'].dt.year.to_list()
    # group by the year and count how many deaths ocurred
    rti_deaths = rti_deaths.groupby('year').count()
    # calculate extrapolated number of deaths
    rti_deaths['estimated_n_deaths'] = rti_deaths['cause'] * scaling_df.loc[rti_deaths.index, 'scale_for_each_year']
    # store the extrapolated number of deaths over the course of the sim
    mult_list_extrapolated_deaths.append(rti_deaths['estimated_n_deaths'].sum())
    dalys_df = log_df['tlo.methods.healthburden']['dalys']
    dalys_df = dalys_df.groupby('year').sum()
    dalys_df['extrapolated_dalys'] = dalys_df['Transport Injuries'] * scaling_df['scale_for_each_year']
    dalys_df = dalys_df.loc[~pd.isnull(dalys_df['extrapolated_dalys'])]
    mult_list_extrapolated_dalys.append(dalys_df['extrapolated_dalys'].sum())
    yld_df = log_df['tlo.methods.healthburden']['yld_by_causes_of_disability'].groupby('year').sum()
    yld_df['scaled_yld'] = yld_df['RTI'] * scaling_df['scale_for_each_year']
    yld_df = yld_df.dropna()
    mult_list_extrapolated_yld.append(yld_df['scaled_yld'].sum())
    yll_df = log_df['tlo.methods.healthburden']['yll_by_causes_of_death'].groupby('year').sum()
    rti_columns = [col for col in yll_df.columns if 'RTI' in col]
    yll_df['scaled_yll'] = [0.0] * len(yll_df)
    for col in rti_columns:
        yll_df['scaled_yll'] += yll_df[col] * scaling_df['scale_for_each_year']
    mult_list_extrapolated_yll.append(yll_df['scaled_yll'].sum())
# Create a results dictionary to save results in
single_injury_results = {}
multiple_injury_results = {}
# plot extrapolated deaths compared to GBD estimate
gbd_data = pd.read_csv('resources/gbd/ResourceFile_Deaths_And_DALYS_GBD2019.csv')
gbd_deaths = gbd_data.loc[gbd_data['measure_name'] == 'Deaths']
gbd_deaths = gbd_deaths.loc[gbd_deaths['cause_name'] == 'Road injuries']
gbd_deaths = gbd_deaths.groupby('Year').sum()
gbd_deaths = gbd_deaths.loc[sim_year_range, 'GBD_Est'].sum()
sing_mean_est_deaths = np.mean(sing_list_extrapolated_deaths)
mult_mean_est_deaths = np.mean(mult_list_extrapolated_deaths)
data = [gbd_deaths, sing_mean_est_deaths, mult_mean_est_deaths]
single_injury_results['deaths'] = data[1]
multiple_injury_results['deaths'] = data[2]
plt.bar(np.arange(len(data)), data, color='lightsteelblue')
plt.xticks(np.arange(len(data)), ['GBD', 'Single\ninjury', 'Multiple\ninjury'])
plt.ylabel('Number of RTI deaths')
plt.title(f"The number of deaths predicted from {sim_start_year} to {sim_end_year} by the GBD model and\n"
          f"the scaled output from the single and the multiple injury models.\n"
          f"Number of simulations: {nsim}, population size: {pop_size}, "
          f"years ran: {yearsrun}")
plt.savefig(save_file_path + f"Scaled number of deaths {imm_death}.png", bbox_inches='tight')
plt.clf()
data = [gbd_deaths, mult_mean_est_deaths]
plt.bar(np.arange(len(data)), data, color='lightsteelblue')
plt.xticks(np.arange(len(data)), ['GBD', 'Model'])
plt.ylabel('Number of RTI deaths')
plt.title(f"The number of deaths predicted from {sim_start_year} to {sim_end_year}\n"
          f" by the GBD model and the scaled output from our model.\n"
          f"Number of simulations: {nsim}, population size: {pop_size}, "
          f"years ran: {yearsrun}")
plt.savefig(save_file_path + f"Scaled number of deaths gbd mult {imm_death}.png", bbox_inches='tight')
plt.clf()
gbd_dalys = gbd_data.loc[gbd_data['measure_name'] == 'DALYs (Disability-Adjusted Life Years)']
gbd_dalys = gbd_dalys.loc[gbd_dalys['cause_name'] == 'Road injuries']
gbd_dalys = gbd_dalys.groupby('Year').sum()
gbd_dalys = gbd_dalys.loc[sim_year_range, 'GBD_Est'].sum()
sing_mean_est_dalys = np.mean(sing_list_extrapolated_dalys)
mult_mean_est_dalys = np.mean(mult_list_extrapolated_dalys)
data = [gbd_dalys, sing_mean_est_dalys, mult_mean_est_dalys]
single_injury_results['dalys'] = data[1]
multiple_injury_results['dalys'] = data[2]
plt.bar(np.arange(len(data)), data, color='lightsteelblue')
plt.xticks(np.arange(len(data)), ['GBD', 'Single\ninjury', 'Multiple\ninjury'])
plt.ylabel('Number of RTI DALYs')
plt.title(f"The number of DALYs predicted from {sim_start_year} to {sim_end_year} by the GBD model and\n"
          f"the scaled output from the single and the multiple injury models.\n"
          f"Number of simulations: {nsim}, population size: {pop_size}, "
          f"years ran: {yearsrun}")
plt.savefig(save_file_path + f"Scaled number of DALYs {imm_death}.png", bbox_inches='tight')
plt.clf()
data = [gbd_dalys, mult_mean_est_dalys]
plt.bar(np.arange(len(data)), data, color='lightsteelblue')
plt.xticks(np.arange(len(data)), ['GBD', 'Model'])
plt.ylabel('Number of RTI DALYs')
plt.title(f"The number of DALYs predicted from {sim_start_year} to {sim_end_year}\n"
          f"by the GBD model and the scaled output from our model.\n"
          f"Number of simulations: {nsim}, population size: {pop_size}, "
          f"years ran: {yearsrun}")
plt.savefig(save_file_path + f"Scaled number of DALYs gbd mult {imm_death}.png", bbox_inches='tight')
plt.clf()
# breakdown and compare the model's yll yld and compare to GBD
# recreate GBD data here
gbd_yll_yld_data = {'year': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
                    'yll': [103892.35, 107353.63, 107015.04, 106125.14, 105933.16, 106551.59, 106424.49, 105551.97,
                            108052.59, 109301.18],
                    'yld': [16689.13, 17201.73, 17780.11, 18429.77, 19100.62, 19805.86, 20462.97, 21169.19, 22055.06,
                            23081.26]}
gbd_yll_yld_data = pd.DataFrame(gbd_yll_yld_data, index=gbd_yll_yld_data['year'])
gbd_yll_yld_data = gbd_yll_yld_data.loc[gbd_yll_yld_data['year'] <= sim_year_range.max()]
gbd_yll = gbd_yll_yld_data['yll'].sum()
gbd_yld = gbd_yll_yld_data['yld'].sum()
sing_model_yll = sum(sing_list_extrapolated_yll)
sing_model_yld = sum(sing_list_extrapolated_yld)
mult_model_yll = sum(mult_list_extrapolated_yll)
mult_model_yld = sum(mult_list_extrapolated_yld)
# plot the yll predicted between models
data = [gbd_yll, sing_model_yll, mult_model_yll]
single_injury_results['yll'] = data[1]
multiple_injury_results['yll'] = data[2]
plt.bar(np.arange(len(data)), data, color='lightsteelblue')
plt.xticks(np.arange(len(data)), ['GBD\nmodel', 'Single\ninjury', 'Multiple\ninjury'])
plt.ylabel('YLL')
plt.title(f"The scaled number of YLL predicted by the GBD model \n"
          f"and the single and multiple injury forms of the model.\n"
          f"Number of simulations: {nsim}, population size: {pop_size}, years ran: {yearsrun}")
plt.savefig(save_file_path + f" YLL, imm death {imm_death}.png", bbox_inches='tight')
plt.clf()
data = [gbd_yll, mult_model_yll]
plt.bar(np.arange(len(data)), data, color='lightsteelblue')
plt.xticks(np.arange(len(data)), ['GBD\nmodel', 'Multiple\ninjury'])
plt.ylabel('YLL')
plt.title(f"The scaled number of YLL predicted by the GBD model \n"
          f"and the multiple injury form of the model.\n"
          f"Number of simulations: {nsim}, population size: {pop_size}, years ran: {yearsrun}")
plt.savefig(save_file_path + f" YLL, gbd mult imm death {imm_death}.png", bbox_inches='tight')
plt.clf()
# plot the yll predicted between models
data = [gbd_yld, sing_model_yld, mult_model_yld]
single_injury_results['yld'] = data[1]
multiple_injury_results['yld'] = data[2]
plt.bar(np.arange(len(data)), data, color='lightsteelblue')
plt.xticks(np.arange(len(data)), ['GBD\nmodel', 'Single\ninjury', 'Multiple\ninjury'])
plt.ylabel('YLD')
plt.title(f"The scaled number of YLD predicted by the GBD model \n"
          f"and the single and multiple injury forms of the model.\n"
          f"Number of simulations: {nsim}, population size: {pop_size}, years ran: {yearsrun}")
plt.savefig(save_file_path + f" YLD, imm death {imm_death}.png", bbox_inches='tight')
plt.clf()
data = [gbd_yld, mult_model_yld]
plt.bar(np.arange(len(data)), data, color='lightsteelblue')
plt.xticks(np.arange(len(data)), ['GBD\nmodel', 'Multiple\ninjury'])
plt.ylabel('YLD')
plt.title(f"The scaled number of YLD predicted by the GBD model \n"
          f"and the multiple injury form of the model.\n"
          f"Number of simulations: {nsim}, population size: {pop_size}, years ran: {yearsrun}")
plt.savefig(save_file_path + f" YLD, gbd mult imm death {imm_death}.png", bbox_inches='tight')
plt.clf()
# Plot and compare the breakdown of DALYs in the model
plt.bar([0], gbd_dalys, color='lightsteelblue', label='DALYs')
plt.bar([1], gbd_yld, color='lightsalmon', label='YLD')
plt.bar([1], gbd_yll, color='lemonchiffon', bottom=gbd_yld, label='YLL')
plt.legend()
plt.xticks([0, 1], ['DALYs', 'YLD & YLL'])
plt.ylabel('Number')
plt.title("The breakdown of the GBD estimated DALYs into \nyears living with disease and years of life lost.")
plt.savefig(save_file_path + "GBD dalys breakdown", bbox_inches='tight')
plt.clf()
# breakdown DALYs into yll yld components using the proportion of dalys being yll to guide, model outputted dalys
# != YLD + YLL for some reason
sing_prop_yll = sing_model_yll / (sing_model_yll + sing_model_yld)
plt.bar([0], sing_mean_est_dalys, color='lightsteelblue', label='DALYs')
plt.bar([1], sing_mean_est_dalys * (1 - sing_prop_yll), color='gold', label='YLD')
plt.bar([1], sing_mean_est_dalys * sing_prop_yll, color='lemonchiffon',
        bottom=sing_mean_est_dalys * (1 - sing_prop_yll), label='YLL')
plt.legend()
plt.xticks([0, 1], ['DALYs', 'YLD & YLL'])
plt.ylabel('Number')
plt.title(f"The breakdown of DALYs estimated in the single injury model into"
          f"\nyears living with disease and years of life lost."
          f"\nNumber of simulations: {nsim}, population size: {pop_size}, "
          f"years ran: {yearsrun}")
plt.savefig(save_file_path + "sing dalys breakdown", bbox_inches='tight')
plt.clf()
mult_prop_yll = mult_model_yll / (mult_model_yll + mult_model_yld)
plt.bar([0], mult_mean_est_dalys, color='lightsteelblue', label='DALYs')
plt.bar([1], mult_mean_est_dalys * (1 - mult_prop_yll), color='gold', label='YLD')
plt.bar([1], mult_mean_est_dalys * mult_prop_yll, color='lemonchiffon',
        bottom=mult_mean_est_dalys * (1 - mult_prop_yll), label='YLL')
plt.legend()
plt.xticks([0, 1], ['DALYs', 'YLD & YLL'])
plt.ylabel('Number')
plt.title(f"The breakdown of DALYs estimated in the multiple injury model into"
          f"\nyears living with disease and years of life lost."
          f"\nNumber of simulations: {nsim}, population size: {pop_size}, "
          f"years ran: {yearsrun}")
plt.savefig(save_file_path + "mult dalys breakdown", bbox_inches='tight')
plt.clf()
plt.bar([0], gbd_yld, color='forestgreen', label='GBD YLD')
plt.bar([0], gbd_yll, color='limegreen', bottom=gbd_yld, label='GBD YLL')
plt.bar([1], sing_mean_est_dalys * (1 - sing_prop_yll), color='sandybrown', label='Sing. YLD')
plt.bar([1], sing_mean_est_dalys * sing_prop_yll, color='peru',
        bottom=sing_mean_est_dalys * (1 - sing_prop_yll), label='Sing. YLL')
plt.bar([2], mult_mean_est_dalys * (1 - mult_prop_yll), color='tan', label='Mult. YLD')
plt.bar([2], mult_mean_est_dalys * mult_prop_yll, color='navajowhite',
        bottom=mult_mean_est_dalys * (1 - mult_prop_yll), label='Mult. YLL')
plt.legend()
plt.xticks([0, 1, 2], ['GBD \nmodel', 'Single \ninjury', 'Multiple \ninjury'])
plt.ylabel('Number')
plt.title(f"A comparison of the breakdown of the estimated number of DALYs \n"
          f"between the GBD model and our models."
          f"increase in RTI ICU patients.\nNumber of simulations: {nsim}, population size: {pop_size}, "
          f"years ran: {yearsrun}")
plt.savefig(save_file_path + f" DALYs breakdown comparison {imm_death}.png", bbox_inches='tight')
plt.clf()
plt.bar([0], gbd_yld, color='forestgreen', label='GBD YLD')
plt.bar([0], gbd_yll, color='limegreen', bottom=gbd_yld, label='GBD YLL')
plt.bar([1], mult_mean_est_dalys * (1 - mult_prop_yll), color='tan', label='Mult. YLD')
plt.bar([1], mult_mean_est_dalys * mult_prop_yll, color='navajowhite',
        bottom=mult_mean_est_dalys * (1 - mult_prop_yll), label='Mult. YLL')
plt.legend()
plt.xticks([0, 1], ['GBD \nmodel', 'Multiple \ninjury'])
plt.ylabel('Number')
plt.title(f"A comparison of the breakdown of the estimated number of DALYs \n"
          f"between the GBD model our model."
          f"increase in RTI ICU patients.\nNumber of simulations: {nsim}, population size: {pop_size}, "
          f"years ran: {yearsrun}")
plt.savefig(save_file_path + f" DALYs breakdown comparison gbd mult {imm_death}.png", bbox_inches='tight')
plt.clf()
# Plot the mean percentage of patients admitted to the ICU for RTI
data = [np.mean(sing_inj_icu_usage), np.mean(mult_inj_icu_usage)]
single_injury_results['icu_use'] = data[0]
multiple_injury_results['icu_use'] = data[1]
percent_increase_in_icu_use = (data[1] / data[0]) * 100 - 100
yerr = [np.std(sing_inj_icu_usage), np.std(mult_inj_icu_usage)]
plt.bar(np.arange(2), data, yerr=yerr, color='lightsteelblue')
plt.xticks(np.arange(len(data)), ['Single\ninjury', 'Multiple\ninjury'])
plt.ylabel('Percent sought admitted to ICU')
plt.title(f"The average percent of patients admitted to the ICU in the \nsingle injury and multiple injury form of the "
          f"model.\nMultiple injuries resulted in a {np.round(percent_increase_in_icu_use, 2)}% "
          f"increase in RTI ICU patients.\nNumber of simulations: {nsim}, population size: {pop_size}, "
          f"years ran: {yearsrun}")
plt.savefig(save_file_path + f" ICU use, imm death {imm_death}.png", bbox_inches='tight')
plt.clf()
# Plot the mean percentage of health system time used to treat RTI patients
data = [np.mean(sing_fraction_of_healthsystem_usage), np.mean(mult_fraction_of_healthsystem_usage)]
single_injury_results['health_sys_use'] = data[0]
multiple_injury_results['health_sys_use'] = data[1]
percent_increase_in_health_sys_use = (data[1] / data[0]) * 100 - 100
yerr = [np.std(sing_fraction_of_healthsystem_usage), np.std(mult_fraction_of_healthsystem_usage)]
plt.ticklabel_format(useOffset=False)
plt.bar(np.arange(2), data, yerr=yerr, color='lightsteelblue')
plt.xticks(np.arange(len(data)), ['Single\ninjury', 'Multiple\ninjury'])
plt.ylabel('Percent healthsystem time usage')
plt.title(f"The average percent of health system time used in the \nsingle injury and multiple injury form of the "
          f"model.\nMultiple injuries resulted in a {np.round(percent_increase_in_health_sys_use, 2)}% "
          f"increase in time spent treating RTI patients.\nNumber of simulations: {nsim}, population size: {pop_size}, "
          f"years ran: {yearsrun}")
plt.savefig(save_file_path + f" Time usage, imm death {imm_death}.png", bbox_inches='tight')
plt.clf()
# Plot the percentage of people who sought care for their injuries
data = [np.mean(sing_percent_sought_healthcare), np.mean(mult_percent_sought_care)]
single_injury_results['HSB'] = data[0]
multiple_injury_results['HSB'] = data[1]
percent_increase_in_health_seeking_behaviour = (data[1] / data[0]) * 100 - 100
yerr = [np.std(sing_percent_sought_healthcare), np.std(mult_percent_sought_care)]
plt.bar(np.arange(2), data, yerr=yerr, color='lightsteelblue')
plt.xticks(np.arange(len(data)), ['Single\ninjury', 'Multiple\ninjury'])
plt.ylabel('Percent sought healthcare')
plt.title(f"The average percent of health seeking behaviour in the \nsingle injury and multiple injury form of the "
          f"model.\nMultiple injuries resulted in a {np.round(percent_increase_in_health_seeking_behaviour, 2)}% "
          f"increase in health seeking behaviour.\nNumber of simulations: {nsim}, population size: {pop_size}, "
          f"years ran: {yearsrun}")
plt.savefig(save_file_path + f" HSB, imm death {imm_death}.png", bbox_inches='tight')
plt.clf()

# plot the average percentage of crashes that result in permanent disability for the single and multiple injury forms
# of the model
data = [np.mean(sing_percent_perm_disability), np.mean(mult_percent_perm_disability)]
single_injury_results['perc_perm_dis'] = data[0]
multiple_injury_results['perc_perm_dis'] = data[1]
percent_increase_in_permanent_disability = (data[1] / data[0]) * 100 - 100
yerr = [np.std(sing_percent_perm_disability), np.std(mult_percent_perm_disability)]
plt.bar(np.arange(2), data, yerr=yerr, color='lightsteelblue')
plt.xticks(np.arange(len(data)), ['Single\ninjury', 'Multiple\ninjury'])
plt.ylabel('Percent permanent disability')
plt.title(f"The average percent of permanent disabiltiy in the \nsingle injury and multiple injury form of the model.\n"
          f"Multiple injuries resulted in a {np.round(percent_increase_in_permanent_disability, 2)}% "
          f"increase in permanent disability.\nNumber of simulations: {nsim}, population size: {pop_size}, "
          f"years ran: {yearsrun}")
plt.savefig(save_file_path + f" permanent disability, imm death {imm_death}.png", bbox_inches='tight')
plt.clf()
# Plot differences in number of inpatient days used
mean_sing_inpatient_days = np.mean(sing_inpatient_days)
std_sing_inpatient_days = np.std(sing_inpatient_days)
mean_mult_inpatient_days = np.mean(mult_inpatient_days)
std_mult_inpatient_days = np.std(mult_inpatient_days)
data = [mean_sing_inpatient_days, mean_mult_inpatient_days]
single_injury_results['inpatient_days'] = data[0]
multiple_injury_results['inpatient_days'] = data[1]
percentage_increase_in_inpatient_day_uses = (data[1] / data[0]) * 100 - 100
yerr = [std_sing_inpatient_days, std_mult_inpatient_days]
plt.bar(np.arange(2), data, yerr=yerr, color='lightsteelblue')
plt.xticks(np.arange(2), ['Single\ninjury', 'Multiple\ninjury'])
plt.ylabel('Average inpatient day usage')
plt.title(f"The average inpatient day usage for the \nsingle injury and multiple injury form of the model.\n"
          f"Multiple injuries resulted in a {np.round(percentage_increase_in_inpatient_day_uses, 2)}% "
          f"increase in inpatient day usage."
          f"\nNumber of simulations: {nsim}, population size: {pop_size}, years ran: {yearsrun}")
plt.savefig(save_file_path + f" inpatient_day, imm death {imm_death}.png", bbox_inches='tight')
plt.clf()
# plot the differences in number of consumables used
sing_mean_consumables = np.mean(sing_number_of_consumables)
sing_std_consumables = np.std(sing_number_of_consumables)
mult_mean_consumables = np.mean(mult_number_of_consumables)
mult_std_consumables = np.std(mult_number_of_consumables)
data = [sing_mean_consumables, mult_mean_consumables]
single_injury_results['consumables'] = data[0]
multiple_injury_results['consumables'] = data[1]
percentage_increase_in_consumable_usage = (data[1] / data[0]) * 100 - 100
yerr = [sing_std_consumables, mult_std_consumables]
plt.bar(np.arange(2), data, yerr=yerr, color='lightsteelblue')
plt.xticks(np.arange(2), ['Single\ninjury', 'Multiple\ninjury'])
plt.title(f"The average consumable usage for the \nsingle injury and multiple injury form of the model.\n"
          f"Multiple injuries resulted in a {np.round(percentage_increase_in_consumable_usage, 2)}% "
          f"increase in consumable usage."
          f"\nNumber of simulations: {nsim}, population size: {pop_size}, years ran: {yearsrun}")
plt.savefig(save_file_path + f" consumable, imm death {imm_death}.png", bbox_inches='tight')
plt.clf()
# plot the number of surgeries performed in each model run
sing_mean_surgeries = np.mean(sing_number_of_surg)
sing_std_surgeries = np.std(sing_number_of_surg)
mult_mean_surgeries = np.mean(mult_number_of_surg)
mult_std_surgeries = np.std(mult_number_of_surg)
data = [sing_mean_surgeries, mult_mean_surgeries]
single_injury_results['surgeries'] = data[0]
multiple_injury_results['surgeries'] = data[1]
percent_increase_in_surgeries = (data[1] / data[0]) * 100 - 100
yerr = [sing_std_surgeries, mult_std_surgeries]
plt.bar(np.arange(2), data, yerr=yerr, color='lightsteelblue')
plt.xticks(np.arange(2), ['Single\ninjury', 'Multiple\ninjury'])
plt.title(f"The average number of surgeries for the \nsingle injury and multiple injury form of the model. \n"
          f"Multiple injuries resulted in a {np.round(percent_increase_in_surgeries, 2)}% "
          f"increase in the number of surgeries."
          f"\nNumber of simulations: {nsim}, population size: {pop_size}, years ran: {yearsrun}")
plt.savefig(save_file_path + f" surgeries, imm death {imm_death}.png", bbox_inches='tight')
plt.clf()
# flatten number of injury lists
sing_n_inj_list_of_lists = []
for result in sing_number_of_injuries:
    sing_n_inj_list_of_lists.append([item for item_list in result for item in item_list])
sing_n_inj_list_of_lists = [item for item_list in sing_n_inj_list_of_lists for item in item_list]
num, counts = np.unique(sing_n_inj_list_of_lists, return_counts=True)
single_inj_dist = [num, counts]
single_inj_dist[1] = np.divide(single_inj_dist[1], sum(single_inj_dist[1]))
true_inc_rti = np.mean([incidence for incidence_list in sing_inj_incidences_of_rti for incidence in incidence_list])
true_inc_injury_single_run = 0
for number_of_injuries in single_inj_dist[0]:
    idx = np.where(single_inj_dist[0] == number_of_injuries)
    true_inc_injury_single_run += true_inc_rti * number_of_injuries * float(single_inj_dist[1][idx])
mult_n_inj_list_of_lists = []
for result in mult_number_of_injuries:
    mult_n_inj_list_of_lists.append([item for item_list in result for item in item_list])
mult_n_inj_list_of_lists = [item for item_list in mult_n_inj_list_of_lists for item in item_list]
num, counts = np.unique(mult_n_inj_list_of_lists, return_counts=True)
mult_inj_dist = [num, counts]
mult_inj_dist[1] = np.divide(mult_inj_dist[1], sum(mult_inj_dist[1]))
true_inc_rti = np.mean([incidence for incidence_list in mult_inj_incidences_of_rti for incidence in incidence_list])
true_inc_injury_mult_run = 0
for number_of_injuries in mult_inj_dist[0]:
    idx = np.where(mult_inj_dist[0] == number_of_injuries)
    true_inc_injury_mult_run += true_inc_rti * number_of_injuries * float(mult_inj_dist[1][idx])
# flatten cause of death lists
sing_data = []
flattened_cause_of_death_sing = [cause for deaths in sing_inj_cause_of_death_in_sim for cause in deaths]
causes_of_death_in_sing = list(set(flattened_cause_of_death_sing))
for cause in causes_of_death_in_sing:
    sing_data.append(flattened_cause_of_death_sing.count(cause))
mult_data = []
flattened_cause_of_death_mult = [cause for deaths in mult_inj_cause_of_death_in_sim for cause in deaths]
causes_of_death_in_mult = list(set(flattened_cause_of_death_mult))
for cause in causes_of_death_in_mult:
    mult_data.append(flattened_cause_of_death_mult.count(cause))
number_of_injuries_data = [sum(sing_n_inj_list_of_lists), sum(mult_n_inj_list_of_lists)]
cause_of_death_dict = {
    'Single injury model': [sing_data, causes_of_death_in_sing],
    'Multiple injury model': [mult_data, causes_of_death_in_mult],
}
for result in cause_of_death_dict.keys():
    cause_of_death_as_percent = [i / sum(cause_of_death_dict[result][0]) for i in cause_of_death_dict[result][0]]
    plt.bar(np.arange(len(cause_of_death_dict[result][0])), cause_of_death_as_percent, color='lightsteelblue')
    plt.xticks(np.arange(len(cause_of_death_dict[result][0])), cause_of_death_dict[result][1])
    plt.ylabel('Percent')
    plt.title("The percentage cause of death from\n road traffic injuries in the "
              + result + f" run.\n Number of simulations: {nsim}, population size: {pop_size}, years ran: {yearsrun}")
    plt.savefig(save_file_path + result + f" cause of death by percentage, imm death {imm_death}.png",
                bbox_inches='tight')
    plt.clf()
# Get GBD data to compare model to
data = pd.read_csv('resources/ResourceFile_RTI_GBD_Number_And_Incidence_Data.csv')
data = data.loc[data['metric'] == 'Rate']
data = data.loc[data['year'] > 2009]
death_data = data.loc[data['measure'] == 'Deaths']
in_rti_data = data.loc[data['measure'] == 'Incidence']
gbd_ten_year_average_inc = in_rti_data['val'].mean()
gbd_ten_year_average_inc_upper = in_rti_data['upper'].mean()
gbd_ten_year_average_inc_lower = in_rti_data['lower'].mean()
gbd_inc_yerr = gbd_ten_year_average_inc_upper - gbd_ten_year_average_inc
gbd_ten_year_average_inc_death = death_data['val'].mean()
gbd_ten_year_average_inc_death_upper = death_data['upper'].mean()
gbd_ten_year_average_inc_death_lower = death_data['lower'].mean()
gbd_inc_death_yerr = gbd_ten_year_average_inc_death_upper - gbd_ten_year_average_inc_death
gbd_yerr = [gbd_inc_yerr, gbd_inc_death_yerr, gbd_inc_yerr]
# Single injury run summary stats
single_injury_mean_incidence_rti = np.mean(sing_inj_incidences_of_rti)
single_injury_std_incidence_rti = np.std(sing_inj_incidences_of_rti)
single_injury_mean_incidence_rti_death = np.mean(sing_inj_incidences_of_death)
single_injury_std_incidence_rti_death = np.std(sing_inj_incidences_of_death)
single_injury_mean_incidence_injuries = true_inc_injury_single_run
single_injury_std_incidence_injuries = np.std(sing_inj_incidences_of_rti)
sing_yerr = [single_injury_std_incidence_rti, single_injury_std_incidence_rti_death,
             single_injury_std_incidence_injuries]
# model run with vibes number of injury distribution
multiple_injury_mean_incidence_rti = np.mean(mult_inj_incidences_of_rti)
multiple_injury_std_incidence_rti = np.std(mult_inj_incidences_of_rti)
multiple_injury_mean_incidence_rti_death = np.mean(mult_inj_incidences_of_death)
multiple_injury_std_incidence_rti_death = np.std(mult_inj_incidences_of_death)
multiple_injury_mean_incidence_injuries = true_inc_injury_mult_run
multiple_injury_std_incidence_injuries = np.std(mult_inj_incidences_of_rti)

mult_yerr = [multiple_injury_std_incidence_rti, multiple_injury_std_incidence_rti_death,
             multiple_injury_std_incidence_injuries]

results_single = [single_injury_mean_incidence_rti, single_injury_mean_incidence_rti_death,
                  single_injury_mean_incidence_injuries]
results_mult = [multiple_injury_mean_incidence_rti, multiple_injury_mean_incidence_rti_death,
                multiple_injury_mean_incidence_injuries]
single_injury_results['incidence_of_rti'] = results_single[0]
multiple_injury_results['incidence_of_rti'] = results_mult[0]
single_injury_results['incidence_of_death'] = results_single[1]
multiple_injury_results['incidence_of_death'] = results_mult[1]
single_injury_results['incidence_of_injuries'] = results_single[2]
multiple_injury_results['incidence_of_injuries'] = results_mult[2]
results_gbd = [gbd_ten_year_average_inc, gbd_ten_year_average_inc_death, gbd_ten_year_average_inc]
n = np.arange(3)
plt.bar(n, results_gbd, yerr=gbd_yerr, width=0.3, color='lightsalmon', label='GBD estimates')
plt.bar(n + 0.3, results_single, yerr=sing_yerr, width=0.3, color='lightsteelblue', label='Single injury model run')
plt.bar(n + 0.6, results_mult, yerr=mult_yerr, width=0.3, color='burlywood', label='multiple injury model run')
plt.xticks(n + 0.3, ['Incidence of \nRTI', 'Incidence of \nRTI death', 'Incidence of \ninjuries'])
for i in range(len(results_gbd)):
    plt.annotate(str(np.round(results_gbd[i], 1)), xy=(n[i], results_gbd[i]), ha='center', va='bottom', rotation=60)
for i in range(len(results_single)):
    plt.annotate(str(np.round(results_single[i], 1)), xy=(n[i] + 0.3, results_single[i]), ha='center', va='bottom',
                 rotation=60)
for i in range(len(results_mult)):
    plt.annotate(str(np.round(results_mult[i], 1)), xy=(n[i] + 0.6, results_mult[i]), ha='center', va='bottom',
                 rotation=60)

plt.ylabel('Incidence per 100,000')
plt.legend()
plt.title(f"The effect of allowing multiple injuries\n in the model on population health outcomes.\n"
          f"Number of simulations: {nsim}, population size: {pop_size}, years run: {yearsrun}")
plt.savefig(save_file_path + f"Single_vs_multiple_injuries_full_comp_imm_death_{imm_death}.png",
            bbox_inches='tight')
plt.ylim([0, max(results_mult) + 1000])
plt.clf()
plt.bar(n, results_single, width=0.3, color='lightsalmon', label='single injury model')
plt.bar(n + 0.3, results_mult, width=0.3, color='burlywood', label='multiple injury model run')
plt.xticks(n + 0.15, ['Incidence of \nRTI', 'Incidence of \nRTI death', 'Incidence of \ninjuries'])
for i in range(len(results_single)):
    plt.annotate(str(np.round(results_single[i], 1)), xy=(n[i], results_single[i]), ha='center', va='bottom',
                 rotation=60)
for i in range(len(results_mult)):
    plt.annotate(str(np.round(results_mult[i], 1)), xy=(n[i] + 0.3, results_mult[i]), ha='center', va='bottom',
                 rotation=60)
plt.ylabel('Incidence per 100,000')
plt.ylim([0, max(results_mult) + 1000])
plt.legend()
plt.title(f"The effect of allowing multiple injuries\n in the model on population health outcomes.\n"
          f"Number of simulations: {nsim}, population size: {pop_size}, years run: {yearsrun}")
plt.savefig(save_file_path + f"Single_vs_multiple_injurie_model_comp_imm_death_{imm_death}.png", bbox_inches='tight')
plt.clf()
plt.bar([1, 2], number_of_injuries_data, color='lightsteelblue')
plt.xticks([1, 2], ['Single injury \nmodel', 'Multiple injury \nmodel'])
plt.ylabel('Number of injuries')
plt.title(f"The number of injuries assigned in the single and mutliple injury forms of the model.\n"
          f"Number of simulations: {nsim}, population size: {pop_size}, years run: {yearsrun}")
plt.savefig(save_file_path + f"Single_vs_multiple_injurie_model_comp_n_injuries_{imm_death}.png", bbox_inches='tight')
plt.clf()
plt.bar(n, results_gbd, width=0.3, color='lightsalmon', label='GBD estimates')
plt.bar(n + 0.3, results_mult, width=0.3, color='burlywood', label='multiple injury model run')
plt.xticks(n + 0.15, ['Incidence of \nRTI', 'Incidence of \nRTI death', 'Incidence of \ninjuries'])
for i in range(len(results_single)):
    plt.annotate(str(np.round(results_gbd[i], 1)), xy=(n[i], results_gbd[i]), ha='center', va='bottom',
                 rotation=60)
for i in range(len(results_mult)):
    plt.annotate(str(np.round(results_mult[i], 1)), xy=(n[i] + 0.3, results_mult[i]), ha='center', va='bottom',
                 rotation=60)
plt.ylabel('Incidence per 100,000')
plt.ylim([0, max(results_mult) + 500])
plt.legend()
plt.title(f"The effect of allowing multiple injuries\n in the model on population health outcomes.\n"
          f"Number of simulations: {nsim}, population size: {pop_size}, years run: {yearsrun}")
plt.savefig(save_file_path + f"GBD_vs_multiple_injurie_model_comp_imm_death_{imm_death}.png", bbox_inches='tight')
plt.clf()

dalys_data = [np.mean(sing_dalys), np.mean(mult_dalys)]
single_injury_results['dalys'] = dalys_data[0]
multiple_injury_results['dalys'] = dalys_data[1]
percent_increase_in_disability_burden = (dalys_data[1] / dalys_data[0]) * 100 - 100
plt.bar([1, 2], dalys_data)
plt.xticks([1, 2], ['Single injury \nmodel', 'Multiple injury \nmodel'])
plt.ylabel('DALYs')
plt.title(f"The effect of allowing multiple injuries\n in the model on population health burden.\n"
          f"Multiple injuries resulted in a {np.round(percent_increase_in_disability_burden, 2)}% increase in DALYs.\n"
          f"Number of simulations: {nsim}, population size: {pop_size}, years run: {yearsrun}")
plt.savefig(save_file_path + f"Single_vs_multiple_injury_model_comp_DALYs_imm_death_{imm_death}.png",
            bbox_inches='tight')
plt.clf()

# plot the number of dalys predicted by the model extrapolated to match the level expected at population level

deaths_data = [np.mean(sing_number_of_deaths), np.mean(mult_number_of_deaths)]
single_injury_results['deaths'] = deaths_data[0]
multiple_injury_results['deaths'] = deaths_data[1]
perecnt_increase_in_deaths = (deaths_data[1] / deaths_data[0]) * 100 - 100
plt.bar([1, 2], deaths_data)
plt.xticks([1, 2], ['Single injury \nmodel', 'Multiple injury \nmodel'])
plt.ylabel('Deaths')
plt.title(f"The effect of allowing multiple injuries\n in the model on number of deaths.\n"
          f"Multiple injuries resulted in a {np.round(perecnt_increase_in_deaths, 2)}% increase in deaths.\n"
          f"Number of simulations: {nsim}, population size: {pop_size}, years run: {yearsrun}")
plt.savefig(save_file_path + f"Single_vs_multiple_injury_model_comp_n_deaths_imm_death_{imm_death}.png",
            bbox_inches='tight')
plt.clf()
single_results_df = pd.DataFrame(data=single_injury_results.values(), index=single_injury_results.keys())
single_results_df.to_csv(save_file_path + "single_results.csv")
multiple_results_df = pd.DataFrame(data=multiple_injury_results.values(), index=multiple_injury_results.keys())
multiple_results_df.to_csv(save_file_path + "multiple_results.csv")
# Plot single vs multiple injury results in a subplot figure
data = [
    [np.mean(sing_percent_sought_healthcare), np.mean(mult_percent_sought_care)],
    [mean_sing_inpatient_days, mean_mult_inpatient_days],
    [np.mean(sing_inj_icu_usage), np.mean(mult_inj_icu_usage)],
    [np.mean(sing_fraction_of_healthsystem_usage), np.mean(mult_fraction_of_healthsystem_usage)],
    [sing_mean_consumables, mult_mean_consumables],
    [sing_mean_surgeries, mult_mean_surgeries]
]
yerr = [
    [np.std(sing_percent_sought_healthcare), np.std(mult_percent_sought_care)],
    [std_sing_inpatient_days, std_mult_inpatient_days],
    [np.std(sing_inj_icu_usage), np.std(mult_inj_icu_usage)],
    [np.std(sing_fraction_of_healthsystem_usage), np.std(mult_fraction_of_healthsystem_usage)],
    [sing_std_consumables, mult_std_consumables],
    [sing_std_surgeries, mult_std_surgeries]
]
titles = ['HSB', 'Inpatient day\nusage', 'ICU usage', 'Time usage', 'Consumables used', 'Surgeries\nperformed']

for i in range(0, len(data)):
    percent_increase = np.round((data[i][1] / data[i][0]) * 100 - 100, 2)
    plt.subplot(3, 2, i + 1, aspect='equal')
    plt.bar(np.arange(len(data[i])), data[i], yerr=yerr[i], color='lightsteelblue')
    if i + 1 > 4:
        plt.xticks(np.arange(len(data[i])), ['Single\ninjury', 'Multiple\ninjury'])
    plt.title(titles[i] + ", " + str(percent_increase) + "%", fontdict={'fontsize': 10})
plt.tight_layout()
plt.savefig(save_file_path + f"Single_vs_multiple_injury_model_comp_all_{imm_death}.png",
            bbox_inches='tight')

for i in range(0, len(data)):
    percent_increase = np.round((data[i][1] / data[i][0]) * 100 - 100, 2)
    plt.subplot(2, 3, i + 1, aspect='equal')
    plt.bar(np.arange(len(data[i])), data[i], yerr=yerr[i], color='lightsteelblue')
    if i + 1 > 4:
        plt.xticks(np.arange(len(data[i])), ['Single\ninjury', 'Multiple\ninjury'])
    plt.title(titles[i] + ", " + str(percent_increase) + "%", fontdict={'fontsize': 10})
plt.tight_layout()
plt.savefig(save_file_path + f"Single_vs_multiple_injury_model_comp_all_alt_{imm_death}.png",
            bbox_inches='tight')
