from pathlib import Path

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    symptommanager,
    rti,
    dx_algorithm_adult,
    dx_algorithm_child,
    antenatal_care,
    contraception,
    labour,
    newborn_outcomes,
    pregnancy_supervisor,
)
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import ast




# =============================== Analysis description ========================================================
# This analysis file has essentially become the model fitting analysis, seeing what happens when we run the model
# and whether an ordinary model run will behave how we would expect it to, hitting the right demographics, producing
# the right injuries, measuring the percent of crashes involving alcohol

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
# Establish the simulation object
yearsrun = 10
start_date = Date(year=2010, month=1, day=1)
end_date = Date(year=(2010 + yearsrun), month=1, day=1)
service_availability = ['*']
pop_size = 10000
nsim = 2

# Create a variable whether to save figures or not
save_figures = True
# Create lists to store information from each simulation in
# Age demographics
sim_age_range = []
sim_male_age_range = []
sim_female_age_range = []
# Gender demographics
females = []
males = []
# Alcohol consumption demographics
percents_attributable_to_alcohol = []
# Health outcomes
list_number_of_crashes = []
list_number_of_disabilities = []
number_of_deaths_pre_hospital = []
number_of_deaths_in_hospital = []
number_of_deaths_no_med = []
number_of_deaths_unavailable_med = []
# Incidences
incidences_of_rti = []
incidences_of_rti_in_children = []
incidences_of_death = []
incidences_of_death_pre_hospital = []
incidences_of_death_post_med = []
incidences_of_death_no_med = []
incidences_of_death_unavailable_med = []
incidences_of_rti_yearly_average = []
incidences_of_death_yearly_average = []
incidences_of_injuries = []
inc_amputations = []
inc_burns = []
inc_fractures = []
inc_tbi = []
inc_sci = []
inc_minor = []
inc_other = []
tot_inc_injuries = []
# Percentages of the context of deaths
ps_of_imm_death = []
ps_of_death_post_med = []
ps_of_death_without_med = []
ps_of_death_unavailable_med = []
percent_died_after_med = []
# Overall percentage of fatal crashes
percent_of_fatal_crashes = []
# Injury severity
perc_mild = []
perc_severe = []
iss_scores = []
# Injury level data
number_of_injured_body_locations = []
inj_loc_data = []
inj_cat_data = []
per_injury_fatal = []
number_of_injuries_per_sim = []
# Flows between model states
rti_model_flow_summary = []
# Health seeking behaviour
percent_sought_healthcare = []
# Admitted to ICU or HDU
percent_admitted_to_icu_or_hdu = []
# Inpatient days
all_sim_inpatient_days = []
# Per sim inpatient days
per_sim_inpatient_days = []
# Number of consumables used
list_consumables_dict = []
# Overall healthsystem time usage
health_system_time_usage = []
# number of major surgeries
per_sim_major_surg = []
# number of minor surgeries
per_sim_minor_surg = []
# number of fractures cast
per_sim_frac_cast = []
# number of lacerations stitched
per_sim_laceration = []
# number of burns managed
per_sim_burn_treated = []
# number of tetanus vaccine administered
per_sim_tetanus = []
# number of pain medicine requested
per_sim_pain_med = []
# Deaths in 2010
deaths_2010 = []
# Injuries in 2010
injuries_in_2010 = []
# Number of injuries per year
injuries_per_year = []
# Number of prehospital deaths in 2010
number_of_prehospital_deaths_2010 = []
# ICU injury characteristics
ICU_frac = []
ICU_dis = []
ICU_tbi = []
ICU_soft = []
ICU_int_o = []
ICU_int_b = []
ICU_sci = []
ICU_amp = []
ICU_eye = []
ICU_lac = []
ICU_burn = []

for i in range(0, nsim):
    sim = Simulation(start_date=start_date)
    # We register all modules in a single call to the register method, calling once with multiple
    # objects. This is preferred to registering each module in multiple calls because we will be
    # able to handle dependencies if modules are registered together
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=service_availability),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath),
        dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        rti.RTI(resourcefilepath=resourcefilepath),
        contraception.Contraception(resourcefilepath=resourcefilepath),
        labour.Labour(resourcefilepath=resourcefilepath),
        newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
        pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
        antenatal_care.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
    )
    logfile = sim.configure_logging(filename="LogFile")
    # create and run the simulation
    sim.make_initial_population(n=pop_size)
    params = sim.modules['RTI'].parameters
    params['allowed_interventions'] = []
    sim.simulate(end_date=end_date)
    log_df = parse_log_file(logfile)
    # Get the relevant information from the rti_demography logging
    demog = log_df['tlo.methods.rti']['rti_demography']
    # get total number of males and females with RTI in this sim
    males.append(sum(demog['males_in_rti']))
    females.append(sum(demog['females_in_rti']))
    # Get the total number of injured persons
    list_number_of_crashes.append(sum(demog['males_in_rti']) + sum(demog['females_in_rti']))
    # Get the total age demographics of those with RTI in the sim
    this_sim_ages = demog['age'].tolist()
    # Get the male and female age demographics of those with RTI in the sim
    this_sim_male_ages = demog['male_age'].tolist()
    this_sim_female_ages = demog['female_age'].tolist()
    for elem in this_sim_ages:
        for item in elem:
            sim_age_range.append(item)
    for elem in this_sim_male_ages:
        for item in elem:
            sim_male_age_range.append(item)
    for elem in this_sim_female_ages:
        for item in elem:
            sim_female_age_range.append(item)
    # Get the percent of crashes attributable to alcohol
    percents_attributable_to_alcohol.append(demog['percent_related_to_alcohol'].tolist())
    # Get the total number of people disabled after RTIs
    list_number_of_disabilities.append(log_df['tlo.methods.rti']['summary_1m']
                                       ['number permanently disabled'].iloc[-1])
    # Get the number of deaths in the sim by cause
    number_of_deaths_pre_hospital.append(
        log_df['tlo.methods.rti']['summary_1m']['number immediate deaths'].sum())
    number_of_deaths_in_hospital.append(
        log_df['tlo.methods.rti']['summary_1m']['number deaths post med'].sum())
    number_of_deaths_no_med.append(
        log_df['tlo.methods.rti']['summary_1m']['number deaths without med'].sum())
    number_of_deaths_unavailable_med.append(
        log_df['tlo.methods.rti']['summary_1m']['number deaths unavailable med'].sum())
    # Get the number of prehospital deaths in 2010
    log_df['tlo.methods.rti']['summary_1m']['year'] = log_df['tlo.methods.rti']['summary_1m']['date'].dt.year
    grouped_by_year = log_df['tlo.methods.rti']['summary_1m'].groupby('year')
    number_of_prehospital_deaths_2010.append(grouped_by_year.get_group(2010)['number immediate deaths'].sum())

    # Get the percentage of those who sought health care, have made the logger output 'none_injured' if no one
    # was injured that month, hence the conditional append statement below
    percent_sought_healthcare.append(
        [i for i in log_df['tlo.methods.rti']['summary_1m']['percent sought healthcare'].tolist() if i !=
         'none_injured']
    )
    # Get the percentage of patients admitted to ICU or HDU
    percent_admitted_to_icu_or_hdu.append(
        [i for i in log_df['tlo.methods.rti']['summary_1m']['percent admitted to ICU or HDU'].tolist() if i !=
         'none_injured']
    )
    icu_df = \
        log_df['tlo.methods.rti']['ICU_patients']
    icu_df = icu_df.drop('date', axis=1)
    road_traffic_injuries = sim.modules['RTI']
    frac_codes = ['112', '113', '211', '212', '412', '414', '612', '712', '712a', '712b', '712c',
                  '811', '812', '813', '813a', '813b', '813c']
    idx, frac_counts = road_traffic_injuries.rti_find_and_count_injuries(icu_df, frac_codes)
    perc_frac = (len(idx) / len(icu_df)) * 100
    dislocationcodes = ['322', '323', '722', '822', '822a', '822b']
    idx, dis_counts = road_traffic_injuries.rti_find_and_count_injuries(icu_df, dislocationcodes)
    perc_dis = (len(idx) / len(icu_df)) * 100
    tbi_codes = ['133', '133a', '133b', '133c', '133d', '134', '134a', '134b', '135']
    idx, tbi_counts = road_traffic_injuries.rti_find_and_count_injuries(icu_df, tbi_codes)
    perc_tbi = (len(idx) / len(icu_df)) * 100
    softtissueinjcodes = ['241', '342', '343', '441', '442', '443']
    idx, soft_counts = road_traffic_injuries.rti_find_and_count_injuries(icu_df, softtissueinjcodes)
    perc_soft = (len(idx) / len(icu_df)) * 100
    organinjurycodes = ['453', '453a', '453b', '552', '553', '554']
    idx, int_o_counts = road_traffic_injuries.rti_find_and_count_injuries(icu_df, organinjurycodes)
    perc_int_o = (len(idx) / len(icu_df)) * 100
    internalbleedingcodes = ['361', '363', '461', '463']
    idx, int_b_counts = road_traffic_injuries.rti_find_and_count_injuries(icu_df, internalbleedingcodes)
    perc_int_b = (len(idx) / len(icu_df)) * 100
    spinalcordinjurycodes = ['673', '673a', '673b', '674', '674a', '674b', '675', '675a', '675b', '676']
    idx, sci_counts = road_traffic_injuries.rti_find_and_count_injuries(icu_df, spinalcordinjurycodes)
    perc_sci = (len(idx) / len(icu_df)) * 100
    amputationcodes = ['782', '782a', '782b', '783', '882', '883', '884']
    idx, amp_counts = road_traffic_injuries.rti_find_and_count_injuries(icu_df, amputationcodes)
    perc_amp = (len(idx) / len(icu_df)) * 100
    eyecodes = ['291']
    idx, eyecounts = road_traffic_injuries.rti_find_and_count_injuries(icu_df, eyecodes)
    perc_eye = (len(idx) / len(icu_df)) * 100
    externallacerationcodes = ['1101', '2101', '3101', '4101', '5101', '7101', '8101']
    idx, externallacerationcounts = road_traffic_injuries.rti_find_and_count_injuries(icu_df,
                                                                                      externallacerationcodes)
    perc_lac = (len(idx) / len(icu_df)) * 100
    burncodes = ['1114', '2114', '3113', '4113', '5113', '7113', '8113']
    idx, burncounts = road_traffic_injuries.rti_find_and_count_injuries(icu_df, burncodes)
    perc_burn = (len(idx) / len(icu_df)) * 100
    if len(icu_df) > 0:
        ICU_frac.append(perc_frac)
        ICU_dis.append(perc_dis)
        ICU_tbi.append(perc_tbi)
        ICU_soft.append(perc_soft)
        ICU_int_o.append(perc_int_o)
        ICU_int_b.append(perc_int_b)
        ICU_sci.append(perc_sci)
        ICU_amp.append(perc_amp)
        ICU_eye.append(perc_eye)
        ICU_lac.append(perc_lac)
        ICU_burn.append(perc_burn)
    # Get the percentage of people who died after seeking healthcare
    percent_died_after_med.append(
        log_df['tlo.methods.rti']['summary_1m']['number deaths post med'].sum() /
        log_df['tlo.methods.rti']['model_progression']['total_sought_medical_care'].iloc[-1]
    )
    # Get the incidence of RTI per 100,000 person years
    incidences_of_rti.append(log_df['tlo.methods.rti']['summary_1m']['incidence of rti per 100,000'].tolist())
    # Get the incidence of death due to RTI per 100,000 person years and the sub categories
    incidences_of_death.append(log_df['tlo.methods.rti']['summary_1m']['incidence of rti death per 100,000'].tolist())
    incidences_of_death_pre_hospital.append(
        log_df['tlo.methods.rti']['summary_1m']['incidence of prehospital death per 100,000'].tolist()
    )
    incidences_of_death_post_med.append(
        log_df['tlo.methods.rti']['summary_1m']['incidence of death post med per 100,000'].tolist()
    )
    incidences_of_death_no_med.append(
        log_df['tlo.methods.rti']['summary_1m']['incidence of death without med per 100,000'].tolist()
    )
    incidences_of_death_unavailable_med.append(
        log_df['tlo.methods.rti']['summary_1m']
        ['incidence of death due to unavailable med per 100,000'].tolist()
    )
    # Get incidences of death average per year
    log_df['tlo.methods.rti']['summary_1m']['year'] = log_df['tlo.methods.rti']['summary_1m']['date'].dt.year
    incidences_of_death_yearly_average.append(
        log_df['tlo.methods.rti']['summary_1m'].groupby('year').mean()['incidence of rti death per 100,000'].tolist())
    # Get the incidence of rtis average per year
    incidences_of_rti_yearly_average.append(
        log_df['tlo.methods.rti']['summary_1m'].groupby('year').mean()['incidence of rti per 100,000'].tolist())
    # Get the incidence of rtis in children per year
    incidences_of_rti_in_children.append(
        log_df['tlo.methods.rti']['summary_1m']['incidence of rti per 100,000 in children'].tolist())
    # Get the incidence of injuries per 100,000
    incidences_of_injuries.append(log_df['tlo.methods.rti']['summary_1m']['injury incidence per 100,000'].tolist())
    # Get information on the deaths that occurred in the sim
    deaths_df = log_df['tlo.methods.demography']['death']
    rti_death_causes = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death']
    rti_deaths = len(deaths_df.loc[deaths_df['cause'].isin(rti_death_causes)])
    # Get the number of pre-hospitl deaths to extrapolate the total number of injuries

    # Get the number of deaths in 2010
    first_year_deaths = deaths_df.loc[deaths_df['date'] < pd.datetime(2011, 1, 1)]
    first_year_rti_deaths = len(first_year_deaths.loc[first_year_deaths['cause'].isin(rti_death_causes)])
    deaths_2010.append(first_year_rti_deaths)
    try:
        # Get the breakdown of road traffic injuries deaths by context by percentage
        ps_of_imm_death.append(len(deaths_df.loc[deaths_df['cause'] == 'RTI_imm_death']) / rti_deaths)
        ps_of_death_post_med.append(len(deaths_df[deaths_df['cause'] == 'RTI_death_with_med']) / rti_deaths)
        ps_of_death_without_med.append(len(deaths_df[deaths_df['cause'] == 'RTI_death_without_med']) / rti_deaths)
        ps_of_death_unavailable_med.append(len(deaths_df[deaths_df['cause'] == 'RTI_unavailable_med']) / rti_deaths)
    except ZeroDivisionError:
        ps_of_imm_death.append(0)
        ps_of_death_post_med.append(0)
        ps_of_death_without_med.append(0)
        ps_of_death_unavailable_med.append(0)
    # Get a rough estimate for the percentage road traffic injury deaths for those involved in RTI
    number_of_crashes = sum(log_df['tlo.methods.rti']['summary_1m']['number involved in a rti'])
    percent_of_fatal_crashes.append(rti_deaths / number_of_crashes)
    injury_info = log_df['tlo.methods.rti']['Injury_information']
    # Get information on injury severity
    mild_inj = [1 for sublist in injury_info['Per_person_severity_category'].tolist() for item in sublist if
                'mild' in item]
    severe_inj = [1 for sublist in injury_info['Per_person_severity_category'].tolist() for item in
                  sublist if 'severe' in item]

    perc_mild.append(sum(mild_inj) / (sum(mild_inj) + sum(severe_inj)))
    perc_severe.append(sum(severe_inj) / (sum(mild_inj) + sum(severe_inj)))
    # Get information on the distribution of ISS scores in the simulation
    severity_distibution = injury_info['Per_person_injury_severity'].tolist()
    # severity_distibution = log_df['tlo.methods.rti']['injury_severity']['ISS_score'].iloc[-1]
    for score in severity_distibution:
        iss_scores.append(score)
    # Get information on the number of injuries each person was given
    ninj_list = injury_info['Number_of_injuries'].tolist()
    ninj_list = [int(item) for sublist in ninj_list for item in sublist]
    ninj_data = {'date': injury_info['date'],
                 'ninj': [sum(list) for list in injury_info['Number_of_injuries'].tolist()]}
    ninj_df = pd.DataFrame(data=ninj_data)
    ninj_df['year'] = pd.DatetimeIndex(ninj_df['date']).year
    number_of_injuries_per_sim.append(ninj_df['ninj'].sum())
    injuries_in_2010.append(ninj_df.loc[ninj_df['year'] == 2010]['ninj'].sum())
    injuries_per_year.append(ninj_df.groupby('year').sum()['ninj'].tolist())
    injury_number_distribution = log_df['tlo.methods.rti']['number_of_injuries'].drop('date', axis=1).iloc[-1].tolist()
    ninj_list_sorted = [ninj_list.count(i) for i in [1, 2, 3, 4, 5, 6, 7, 8]]
    number_of_injured_body_locations.append(ninj_list_sorted)
    # Get the per injury fatality ratio
    per_injury_fatal.append(
        len(log_df['tlo.methods.demography']['death'].loc[
                log_df['tlo.methods.demography']['death']['cause'].isin(['RTI_death_without_med', 'RTI_death_with_med',
                                                                         'RTI_unavailable_med'])]) /
        np.multiply(ninj_list_sorted,
                    [1, 2, 3, 4, 5, 6, 7, 8]).sum())
    # Get information on where these injuries occured on each person
    injury_loc_list = injury_info['Location_of_injuries'].tolist()
    injury_loc_list = [int(item) for sublist in injury_loc_list for item in sublist]
    binned_loc_dist = []
    for loc in [1, 2, 3, 4, 5, 6, 7, 8]:
        binned_loc_dist.append(injury_loc_list.count(loc))
    inj_loc_data.append(binned_loc_dist)
    # Get information on the injury category distribution this run
    inj_cat_list = injury_info['Injury_category'].tolist()
    inj_cat_list = [int(item) for sublist in inj_cat_list for item in sublist]
    binned_cat_dist = []
    for cat in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
        binned_cat_dist.append(inj_cat_list.count(cat))
    assert len(inj_cat_list) == len(injury_loc_list)
    inj_cat_data.append(binned_cat_dist)
    # Get information on the flows between each model state this run
    rti_model_flow_summary.append(log_df['tlo.methods.rti']['model_progression'].drop('date', axis=1).iloc[-1].tolist())
    # Get information on the total incidence of injuries and the breakdown of injury by type
    injury_category_incidence = log_df['tlo.methods.rti']['Inj_category_incidence']
    inc_amputations.append(injury_category_incidence['inc_amputations'].tolist())
    inc_burns.append(injury_category_incidence['inc_burns'].tolist())
    inc_fractures.append(injury_category_incidence['inc_fractures'].tolist())
    inc_tbi.append(injury_category_incidence['inc_tbi'].tolist())
    inc_sci.append(injury_category_incidence['inc_sci'].tolist())
    inc_minor.append(injury_category_incidence['inc_minor'].tolist())
    inc_other.append(injury_category_incidence['inc_other'].tolist())
    tot_inc_injuries.append(injury_category_incidence['tot_inc_injuries'].tolist())
    # Get the inpatient days usage. I take the overall inpatient day usage from all the simulations and the per-sim
    # inpatient day info
    this_sim_inpatient_days = []
    inpatient_day_df = log_df['tlo.methods.healthsystem']['HSI_Event'].loc[
        log_df['tlo.methods.healthsystem']['HSI_Event']['TREATMENT_ID'] == 'RTI_MedicalIntervention']
    for person in inpatient_day_df.index:
        # Get the number of inpatient days per person, if there is a key error when trying to access inpatient days it
        # means that this patient didn't require any so append (0)
        try:
            all_sim_inpatient_days.append(inpatient_day_df.loc[person, 'Number_By_Appt_Type_Code']['InpatientDays'])
            this_sim_inpatient_days.append(inpatient_day_df.loc[person, 'Number_By_Appt_Type_Code']['InpatientDays'])
        except KeyError:
            all_sim_inpatient_days.append(0)
            this_sim_inpatient_days.append(0)
    # get the inpatient days used in this sim
    per_sim_inpatient_days.append(this_sim_inpatient_days)
    # get the consumables used in each simulation
    consumables_list = log_df['tlo.methods.healthsystem']['Consumables']['Item_Available'].tolist()
    consumables_list_to_dict = []
    for string in consumables_list:
        consumables_list_to_dict.append(ast.literal_eval(string))
    number_of_consumables_in_sim = 0
    for dictionary in consumables_list_to_dict:
        number_of_consumables_in_sim += sum(dictionary.values())
    list_consumables_dict.append(number_of_consumables_in_sim)
    # get the overall health system time used
    health_system_time_usage.append(np.mean(log_df['tlo.methods.healthsystem']['Capacity']['Frac_Time_Used_Overall']))
    # get the number of treating hsi events by type
    appointments = log_df['tlo.methods.healthsystem']['HSI_Event']
    appointments = appointments.loc[appointments['did_run'] == True]
    per_sim_burn_treated.append(len(appointments.loc[appointments['TREATMENT_ID'] == 'RTI_Burn_Management']))
    per_sim_frac_cast.append(len(appointments.loc[appointments['TREATMENT_ID'] == 'RTI_Fracture_Cast']))
    per_sim_laceration.append(len(appointments.loc[appointments['TREATMENT_ID'] == 'RTI_Suture']))
    per_sim_major_surg.append(len(appointments.loc[appointments['TREATMENT_ID'] == 'RTI_Major_Surgeries']))
    per_sim_minor_surg.append(len(appointments.loc[appointments['TREATMENT_ID'] == 'RTI_Minor_Surgeries']))
    per_sim_tetanus.append(len(appointments.loc[appointments['TREATMENT_ID'] == 'RTI_Tetanus_Vaccine']))
    per_sim_pain_med.append(len(appointments.loc[appointments['TREATMENT_ID'] == 'RTI_Acute_Pain_Management']))
    # todo: plot the urban vs rural injury severity currently produced by the model (injury_severity)
    print(i)

def age_breakdown(age_array):
    """
    A function which breaks down an array of ages into specific age ranges
    :param age_array:
    :return:
    """
    # Breakdown the age data into boundaries 0-5, 6-10, 11-15, 16-20 etc...
    zero_to_five = len([i for i in age_array if i < 6])
    six_to_ten = len([i for i in age_array if 6 <= i < 11])
    eleven_to_fifteen = len([i for i in age_array if 11 <= i < 16])
    sixteen_to_twenty = len([i for i in age_array if 16 <= i < 21])
    twenty1_to_twenty5 = len([i for i in age_array if 21 <= i < 26])
    twenty6_to_thirty = len([i for i in age_array if 26 <= i < 31])
    thirty1_to_thirty5 = len([i for i in age_array if 31 <= i < 36])
    thirty6_to_forty = len([i for i in age_array if 36 <= i < 41])
    forty1_to_forty5 = len([i for i in age_array if 41 <= i < 46])
    forty6_to_fifty = len([i for i in age_array if 46 <= i < 51])
    fifty1_to_fifty5 = len([i for i in age_array if 51 <= i < 56])
    fifty6_to_sixty = len([i for i in age_array if 56 <= i < 61])
    sixty1_to_sixty5 = len([i for i in age_array if 61 <= i < 66])
    sixty6_to_seventy = len([i for i in age_array if 66 <= i < 71])
    seventy1_to_seventy5 = len([i for i in age_array if 71 <= i < 76])
    seventy6_to_eighty = len([i for i in age_array if 76 <= i < 81])
    eighty1_to_eighty5 = len([i for i in age_array if 81 <= i < 86])
    eighty6_to_ninety = len([i for i in age_array if 86 <= i < 91])
    ninety_plus = len([i for i in age_array if 90 < i])
    return [zero_to_five, six_to_ten, eleven_to_fifteen, sixteen_to_twenty, twenty1_to_twenty5, twenty6_to_thirty, \
            thirty1_to_thirty5, thirty6_to_forty, forty1_to_forty5, forty6_to_fifty, fifty1_to_fifty5, fifty6_to_sixty, \
            sixty1_to_sixty5, sixty6_to_seventy, seventy1_to_seventy5, seventy6_to_eighty, eighty1_to_eighty5, \
            eighty6_to_ninety, ninety_plus]

# Plot the per injury fatality produced by the model and compare to the gbd data
data = pd.read_csv('resources/ResourceFile_RTI_GBD_Number_And_Incidence_Data.csv')
gbd_death_data = data.loc[data['measure'] == 'Deaths']
gbd_in_rti_data = data.loc[data['measure'] == 'Incidence']
gbd_death_data = gbd_death_data.sort_values(by='year')
gbd_in_rti_data = gbd_in_rti_data.sort_values(by='year')
gbd_death_number = gbd_death_data.loc[gbd_death_data['metric'] == 'Number']
gbd_rti_number = gbd_in_rti_data.loc[gbd_in_rti_data['metric'] == 'Number']
gbd_percent_fatal_ratio = gbd_death_number['val'].sum() / gbd_rti_number['val'].sum()
model_percent_fatal_ratio = np.mean(per_injury_fatal)
plt.bar(np.arange(2), [model_percent_fatal_ratio, gbd_percent_fatal_ratio],
        color=['lightsteelblue', 'lightsalmon'])
plt.xticks(np.arange(2), ['Model per injury \n death percentage', 'GBD per injury \n death percentage'])
plt.ylabel('Percent')
plt.title(f"Per injury fatality ratio, model compared to GBD"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Per_injury_fatality.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()

# plot the percentage of those who sought health care
per_sim_average_health_seeking = [np.mean(i) for i in percent_sought_healthcare]
overall_average_health_seeking_behaviour = np.mean(per_sim_average_health_seeking)
plt.pie([overall_average_health_seeking_behaviour, 1 - overall_average_health_seeking_behaviour],
        explode=None, labels=['Sought care', "Didn't seek care"], colors=['lightsteelblue', 'lightsalmon'],
        autopct='%1.1f%%')
plt.title(f"Average percentage of those with road traffic injuries who sought health care"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Percent_Sought_Healthcare.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# Plot the percentage of people admitted to ICU and HDU
per_sim_icu_or_hdu_average = np.mean([np.mean(i) for i in percent_admitted_to_icu_or_hdu])

data = np.multiply([per_sim_icu_or_hdu_average], 100)
plt.bar(np.arange(1), data, color='lightsteelblue', label='Model', width=0.4)
plt.bar(np.arange(1) + 0.4, [2.7 + 3.3], color='lightsalmon', label='KCH', width=0.4)
plt.xticks(np.arange(1) + 0.2, ['Percent admitted ICU or HDU'])
plt.ylabel('Percentage')
plt.legend()
plt.title(f"Average percentage admitted to ICU/HDU"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Percent_admitted_icu_hdu_bar.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()

# Plot the percentage of death post med
overall_average_post_med_death = np.mean(percent_died_after_med)
plt.pie([overall_average_post_med_death, 1 - overall_average_post_med_death],
        explode=None, labels=['Fatal', "Non-fatal"], colors=['lightsteelblue', 'lightsalmon'],
        autopct='%1.1f%%')
plt.title(f"Average percent survival outcome of those with road traffic injuries who sought health care"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Percent_Survival_Healthcare.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# Plot the percent of death post med compared to Kamuzu central see https://doi.org/10.1016/j.jsurg.2014.09.010
percent_mortality_kamuzu = (182 + 38) / (3840 + 1227 + 182 + 38)
plt.bar(np.arange(2), [percent_mortality_kamuzu, 1 - percent_mortality_kamuzu], width=0.3,
        color='lightsalmon', label='In-hospital mortality, \nKamuzu central hospital')
plt.bar(np.arange(2) + 0.5, [overall_average_post_med_death, 1 - overall_average_post_med_death], width=0.3,
        color='lightsteelblue', label='Model in-hospital mortality')
plt.xticks(np.arange(2) + 0.25, ['Fatal', 'Non-fatal'])
plt.legend()
plt.title(f"In-hospital fatality due to injury percentage \n "
          f"model prediction: {np.round(overall_average_post_med_death, 2)} \n"
          f"Kamuzu central hospital: {np.round(percent_mortality_kamuzu, 2)} \n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/Demographics_of_RTI/Percent_Survival_Healthcare_compare_Kamuzu.png', bbox_inches='tight')
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Percent_Survival_Healthcare_compare_Kamuzu.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# Plot health outcomes for road traffic injuries
total_n_crashes = sum(list_number_of_crashes)
total_n_hospital_deaths = sum(number_of_deaths_in_hospital)
total_n_prehospital_deaths = sum(number_of_deaths_pre_hospital)
total_n_no_hospital_deaths = sum(number_of_deaths_no_med)
total_n_unavailable_med_deaths = sum(number_of_deaths_unavailable_med)
total_n_perm_disability = sum(list_number_of_disabilities)
total_survived = total_n_crashes - total_n_hospital_deaths - total_n_prehospital_deaths - total_n_no_hospital_deaths - \
                 total_n_unavailable_med_deaths
plt.pie([total_survived, total_n_perm_disability, total_n_prehospital_deaths, total_n_hospital_deaths,
         total_n_no_hospital_deaths, total_n_unavailable_med_deaths], explode=None,
        labels=['Non-fatal', 'Permanently disabled', 'Pre-hospital mortality', 'In-hospital mortality',
                'No-hospital mortality', 'Unavailable care mortality'],
        colors=['lightsteelblue', 'lightsalmon', 'wheat', 'darkcyan', 'orchid'], autopct='%1.1f%%')
plt.title(f"Outcomes for road traffic injuries in the model"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Outcome_Of_Crashes.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# Plot average age of those in RTI compared to other sources

mean_age = np.mean(sim_age_range)
std_age = np.std(sim_age_range)
police_ave_age = 32
police_std_age = 12
queen_elizabeth_central_hospital_mean = 32
queen_elizabeth_central_hospital_std = 12
plt.bar(np.arange(3), [mean_age, police_ave_age, queen_elizabeth_central_hospital_mean],
        yerr = [std_age, police_std_age, queen_elizabeth_central_hospital_std], color='wheat')
plt.xticks(np.arange(3), ['Model', 'Police data', 'Queen Elizabeth Central \n Hospital data'])
plt.ylabel('Age')
plt.title(f"Average age with RTIs compared to police and hospital data"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Age_average_comp_to_pol_hos.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# Plot age demographics of those in RTI by percentage
height_for_bar_plot = age_breakdown(sim_age_range)
height_for_bar_plot = np.divide(height_for_bar_plot, sum(height_for_bar_plot))
labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40',
          '41-45', '46-50', '51-55', '56-60', '61-65', '66-70', '71-75', '76-80',
          '81-85', '86-90', '90+']
plt.bar(np.arange(len(height_for_bar_plot)), height_for_bar_plot, color='lightsteelblue')
plt.xticks(np.arange(len(height_for_bar_plot)), labels, rotation=45)
plt.ylabel('Percentage')
plt.xlabel('Age')
plt.title(f"Age demographics of those with RTIs"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Age_demographics_percentage.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# Plot age demographics of those in rti by number
height_for_bar_plot = age_breakdown(sim_age_range)
labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40',
          '41-45', '46-50', '51-55', '56-60', '61-65', '66-70', '71-75', '76-80',
          '81-85', '86-90', '90+']
plt.bar(np.arange(len(height_for_bar_plot)), height_for_bar_plot, color='lightsteelblue')
plt.xticks(np.arange(len(height_for_bar_plot)), labels, rotation=45)
plt.ylabel('Number')
plt.xlabel('Age')
plt.title(f"Age demographics of those with RTIs"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Age_demographics_number.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# Plot age/gender demographics of those in rti by number
height_for_males_bar_plot = age_breakdown(sim_male_age_range)
height_for_female_bar_plot = age_breakdown(sim_female_age_range)
labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40',
          '41-45', '46-50', '51-55', '56-60', '61-65', '66-70', '71-75', '76-80',
          '81-85', '86-90', '90+']
labels.reverse()
height_for_males_bar_plot.reverse()
height_for_female_bar_plot.reverse()
plt.barh(labels, height_for_males_bar_plot, alpha=0.5, label='Males', color='lightsteelblue')
plt.barh(labels, np.multiply(height_for_female_bar_plot, -1), alpha=0.5, label='Females', color='lightsalmon')
locs, labels = plt.xticks()
plt.xticks(locs, np.sqrt(locs ** 2), fontsize=8)
plt.title(f"Sum total of number of road traffic injuries"
          f"\n"
          f"by age and sex."
          f"\n"
          f"Population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}"
          )
plt.xlabel('Number')
plt.yticks(fontsize=7)
plt.legend()
plt.tight_layout()
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Age_Gender_Demographics_number.png')
    plt.clf()
else:
    plt.clf()
labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40',
          '41-45', '46-50', '51-55', '56-60', '61-65', '66-70', '71-75', '76-80',
          '81-85', '86-90', '90+']
labels.reverse()
# Plot age/gender demographics of those in rti by percent
height_for_males_bar_plot = np.divide(height_for_males_bar_plot, sum(height_for_males_bar_plot))
if sum(height_for_female_bar_plot) > 0:
    height_for_female_bar_plot = np.divide(height_for_female_bar_plot, sum(height_for_female_bar_plot))
else:
    height_for_female_bar_plot = np.zeros(len(height_for_female_bar_plot))
plt.barh(labels, height_for_males_bar_plot, alpha=0.5, label='Males', color='lightsteelblue')
plt.barh(labels, np.multiply(height_for_female_bar_plot, -1), alpha=0.5, label='Females', color='lightsalmon')
locs, labels = plt.xticks()
plt.xticks(locs, np.round(np.sqrt(locs ** 2), 2), fontsize=8)
plt.title(f"Percentage of road traffic injuries"
          f"\n"
          f"by age and sex in all simulations."
          f"\n"
          f"Population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}"
          )
plt.xlabel('Number')
plt.yticks(fontsize=7)
plt.legend()
plt.tight_layout()
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Age_Gender_Demographics_percentage.png')
    plt.clf()
else:
    plt.clf()

total_injuries = [i + j for i, j in zip(males, females)]
male_perc = np.divide(males, total_injuries)
femal_perc = np.divide(females, total_injuries)
n = np.arange(2)
data = [np.round(np.mean(male_perc), 3), np.round(np.mean(femal_perc), 3)]
plt.bar(np.arange(2), data, yerr=[np.std(male_perc), np.std(femal_perc)], color='lightsteelblue')
for i in range(len(data)):
    plt.annotate(str(data[i]), xy=(n[i], data[i]), ha='center', va='bottom')
plt.xticks(np.arange(2), ['Males', 'Females'])
plt.ylabel('Percentage')
plt.xlabel('Gender')
plt.title(f"Gender demographics of those with RTIs"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Gender_demographics.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
plt.pie(data, explode=None, labels=['Males', 'Females'], colors=['lightsteelblue', 'lightsalmon'], autopct='%1.1f%%',
        startangle=90)
plt.title(f"Gender demographics of those with RTIs"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Gender_demographics_pie.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# Plot the percentage of crashes attributable to alcohol
means_of_sim = [np.mean(i) for i in percents_attributable_to_alcohol]
means_non_alocohol = np.subtract(np.ones(len(means_of_sim)), means_of_sim)
mean_of_means = np.mean(means_of_sim)
mean_of_means_non_alcohol = np.mean(means_non_alocohol)
std_of_means = np.std(means_of_sim)
std_of_means_non_alcohol = np.std(means_non_alocohol)
plt.pie([mean_of_means, 1 - mean_of_means], explode=None, labels=['Alcohol related', 'Non-alcohol related'],
        colors=['lightsteelblue', 'lightsalmon'], autopct='%1.1f%%')
plt.title(f"Average percentage of RTIs attributable to Alcohol"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Alcohol_demographics.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
plt.bar(np.arange(2), [mean_of_means, mean_of_means_non_alcohol], yerr=[std_of_means, std_of_means_non_alcohol])
plt.xticks(np.arange(2), ['Attributable to alcohol', 'Not attributable to alcohol'])
plt.title(f"Average percentage of RTIs attributable to Alcohol"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Alcohol_demographics_bar.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# plot the incidence of RTI and death from RTI
average_incidence = [float(sum(col)) / len(col) for col in zip(*incidences_of_rti)]
std_incidence = [np.std(i) for i in zip(*incidences_of_rti)]
inc_upper = [inc + (1.96 * std) / nsim for inc, std in zip(average_incidence, std_incidence)]
inc_lower = [inc - (1.96 * std) / nsim for inc, std in zip(average_incidence, std_incidence)]
average_deaths = [float(sum(col)) / len(col) for col in zip(*incidences_of_death)]
std_deaths = [np.std(j) for j in zip(*incidences_of_death)]
death_upper = [inc + (1.96 * std) / nsim for inc, std in zip(average_deaths, std_deaths)]
death_lower = [inc - (1.96 * std) / nsim for inc, std in zip(average_deaths, std_deaths)]

average_injury_incidence = [float(sum(col)) / len(col) for col in zip(*incidences_of_injuries)]
overall_av_inc_sim = np.mean(average_incidence)
overall_av_death_inc_sim = np.mean(average_deaths)
overall_av_inc_injuries = np.mean(average_injury_incidence)
time = log_df['tlo.methods.rti']['summary_1m']['date']
plt.plot(time, average_incidence, color='lightsteelblue', label='Incidence of RTI', zorder=2)
plt.fill_between(time.tolist(), inc_upper, inc_lower, alpha=0.5, color='lightsteelblue', label='95% C.I., RTI inc.',
                 zorder=1)
plt.plot(time, average_deaths, color='lightsalmon', label='Incidence of death '
                                                          '\n'
                                                          'due to RTI', zorder=2)
plt.fill_between(time.tolist(), death_upper, death_lower, alpha=0.5, color='lightsalmon',
                 label='95% C.I. inc death', zorder=1)
# plt.plot(time, average_injury_incidence, color='green', label='Incidence of RTI injury')
plt.hlines(overall_av_inc_sim, time.iloc[0], time.iloc[-1], label=f"Average incidence of "
                                                                  f"\n"
                                                                  f"RTI = {np.round(overall_av_inc_sim, 2)}",
           color='lightsteelblue', linestyles='--')
plt.hlines(overall_av_death_inc_sim, time.iloc[0], time.iloc[-1], label=f"Average incidence of "
                                                                        f"\n"
                                                                        f"death = "
                                                                        f"{np.round(overall_av_death_inc_sim, 2)}",
           color='lightsalmon', linestyles='--')
plt.xlabel('Simulation time')
plt.ylabel('Incidence per 100,000')
plt.legend(loc='upper center', bbox_to_anchor=(1.1, 0.8), shadow=True, ncol=1)
plt.title(f"Average incidence of RTIs and deaths due to RTI"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Incidence_and_deaths.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# Plot the incidence predicted by the model compared to the GBD data
data = pd.read_csv('resources/ResourceFile_RTI_GBD_Number_And_Incidence_Data.csv')
data = data.loc[data['metric'] == 'Rate']
data = data.loc[data['measure'] == 'Deaths']
data = data.loc[data['year'] > 2009]
gbd_time = ['2010-01-01', '2011-01-01', '2012-01-01', '2013-01-01',
            '2014-01-01', '2015-01-01', '2016-01-01', '2017-01-01',
            '2018-01-01', '2019-01-01']
yearly_average_deaths = [float(sum(col)) / len(col) for col in zip(*incidences_of_death_yearly_average)]
std_yearly_death_incidence = [np.std(i) for i in zip(*incidences_of_death_yearly_average)]
yearly_death_inc_upper = [inc + (1.96 * std) / nsim for inc, std in zip(yearly_average_deaths,
                                                                        std_yearly_death_incidence)]
yearly_death_inc_lower = [inc - (1.96 * std) / nsim for inc, std in zip(yearly_average_deaths,
                                                                        std_yearly_death_incidence)]
log_df['tlo.methods.rti']['summary_1m']['year'] = log_df['tlo.methods.rti']['summary_1m']['date'].dt.year
time = pd.to_datetime(gbd_time[:yearsrun])
plt.plot(time, yearly_average_deaths, color='lightsalmon', label='Model', zorder=2)
plt.fill_between(time.tolist(), yearly_death_inc_upper, yearly_death_inc_lower, alpha=0.5, color='lightsalmon',
                 label='95% C.I. model', zorder=1)
plt.plot(pd.to_datetime(gbd_time), data.val, color='mediumaquamarine', label='GBD', zorder=2)
plt.fill_between(pd.to_datetime(gbd_time), data.upper, data.lower, alpha=0.5, color='mediumaquamarine',
                 label='95% C.I. GBD', zorder=1)
plt.xlabel('Year')
plt.ylabel('Incidence of Death')
plt.legend()
plt.title(f"Incidence of death over time, model compared to GBD estimate"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Incidence_model_over_time.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()

# Plot the incidences of death showing the specific causes:
average_deaths_no_med = [float(sum(col)) / len(col) for col in zip(*incidences_of_death_no_med)]
std_deaths_no_med = [np.std(j) for j in zip(*incidences_of_death_no_med)]
death_no_med_upper = [inc + (1.96 * std) / nsim for inc, std in zip(average_deaths_no_med, std_deaths_no_med)]
death_no_med_lower = [inc - (1.96 * std) / nsim for inc, std in zip(average_deaths_no_med, std_deaths_no_med)]
overall_av_death_no_med = np.mean(average_deaths_no_med)
average_deaths_unavailable_med = \
    [float(sum(col)) / len(col) for col in zip(*incidences_of_death_unavailable_med)]
std_deaths_unavailable_med = [np.std(j) for j in zip(*incidences_of_death_unavailable_med)]
death_un_med_upper = \
    [inc + (1.96 * std) / nsim for inc, std in zip(average_deaths_unavailable_med, std_deaths_unavailable_med)]
death_un_med_lower = \
    [inc - (1.96 * std) / nsim for inc, std in zip(average_deaths_unavailable_med, std_deaths_unavailable_med)]
overall_av_death_un_med = np.mean(average_deaths_unavailable_med)
average_deaths_with_med = [float(sum(col)) / len(col) for col in zip(*incidences_of_death_post_med)]
std_deaths_with_med = [np.std(j) for j in zip(*incidences_of_death_post_med)]
death_upper_with_med = [inc + (1.96 * std) / nsim for inc, std in zip(average_deaths_with_med, std_deaths_with_med)]
death_lower_with_med = [inc - (1.96 * std) / nsim for inc, std in zip(average_deaths_with_med, std_deaths_with_med)]
overall_av_death_with_med = np.mean(average_deaths_with_med)
average_deaths_pre_hospital = [float(sum(col)) / len(col) for col in zip(*incidences_of_death_pre_hospital)]
std_deaths_with_pre_hospital = [np.std(j) for j in zip(*incidences_of_death_pre_hospital)]
death_upper_with_pre_hospital = \
    [inc + (1.96 * std) / nsim for inc, std in zip(average_deaths_pre_hospital, std_deaths_with_pre_hospital)]
death_lower_with_pre_hospital = \
    [inc - (1.96 * std) / nsim for inc, std in zip(average_deaths_pre_hospital, std_deaths_with_pre_hospital)]
overall_av_death_pre_hospital = np.mean(average_deaths_pre_hospital)
plt.bar(np.arange(5), [overall_av_death_inc_sim, overall_av_death_pre_hospital, overall_av_death_with_med,
                       overall_av_death_no_med, overall_av_death_un_med], color='lightsteelblue')
plt.xticks(np.arange(5), ['Overall \n incidence \n of \n deaths',
                          'Incidence \n of \n pre-hospital \n deaths',
                          'Incidence \n of \n deaths \n with \n treatment',
                          'Incidence \n of \n deaths\n without \n treatment',
                          'Incidence \n of \n deaths\n unavailable \n treatment'])
plt.ylabel('Incidence per 100,000')
plt.title(f"Average incidence of deaths due to RTI"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Incidence_of_rti_deaths.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# Plot the incidence of pre-hospital mortality from the model compared to police record data
# Results from Schlottmann et al. DOI:10.4314/mmj.v29i4.4
incidence_of_on_scene_mortality_police = 6
plt.bar(np.arange(2), [incidence_of_on_scene_mortality_police, overall_av_death_pre_hospital],
        color='mediumaquamarine')
plt.xticks(np.arange(2), ['Estimated incidence from'
                          '\n'
                          'police data',
                          'Model incidence'])
plt.ylabel('Incidence per 100,000 person years')
plt.title(f"Average incidence of deaths due to RTI"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Incidence_of_on_scene_mortality_comp_to_police.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# Plot the incidences of deaths from in hospital and prehospital deaths, i.e. deaths that would be picked up by
# the hospital of the police and compare the model output to that of Samuel et al. 2012, who gave probably the
# best estimate from Malawi specific data for the incidence of death
samuel_incidence_of_rti_death = 20.9

plt.bar(np.arange(3),
        [samuel_incidence_of_rti_death,  # Estimated incidence of death from hospital/police data
         overall_av_death_pre_hospital + overall_av_death_with_med + overall_av_death_un_med,  # Equivalent model output
         overall_av_death_no_med],  # remaining deaths
        color=['lightsalmon', 'lightsteelblue', 'lightsteelblue'])
plt.xticks(np.arange(3),
           ['Estimated incidence'
            '\n'
            'of hospital and police'
            '\n'
            'recorded deaths'
            '\n'
            'from Samuel et al. 2012',
            'Estimated incidence'
            '\n'
            'from matching categories'
            '\n'
            'in models',
            'Unaccounted for mortality',
            ])
plt.ylabel('Incidence per 100,000')
plt.title(f"The model's predicted incidence of RTI related death "
          f"\n"
          f"compared to the estimate of Samuel et al. 2012"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Incidence_of_rti_deaths_compare_Samuel.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# Plot the incidence of deaths produced by the model compared to the estimates from the Malawian hospital registry data,
# Samuel et al. estimated incidence of death, and the WHO estimated incidence of death
hospital_registry_inc = 5.1
data = pd.read_csv('resources/ResourceFile_RTI_GBD_Number_And_Incidence_Data.csv')
data = data.loc[data['measure'] == 'Deaths']
data = data.loc[data['metric'] == 'Rate']
data = data.loc[data['year'] > 2009]
gbd_average_2010_to_2019 = data['val'].mean()
police_estimated_inc = 7.5
who_est = 35
plt.bar(np.arange(6),
        [hospital_registry_inc,  # Estimated incidence of death from hospital data
         police_estimated_inc,  # Estimated incidence of death from police data
         gbd_average_2010_to_2019,  # Estimated incidence of death from GBD Malawi data 2010 - 2019
         samuel_incidence_of_rti_death,  # Estimated incidence of death from capture recapture method
         who_est,  # Estimated incidence of death from the WHO
         overall_av_death_inc_sim,  # Models predicted incidence of death
         ],  # remaining deaths
        color=['lightsalmon', 'lightsteelblue', 'mediumaquamarine', 'wheat', 'olive', 'blue'])
plt.xticks(np.arange(6),
           ['Estimated'
            '\n'
            'incidence'
            '\n'
            'of death'
            '\n'
            'from'
            '\n'
            'hospital '
            '\n'
            'records',
            'Estimated'
            '\n'
            'incidence'
            '\n'
            'of death from'
            '\n'
            'police '
            '\n'
            'records',
            'Estimated'
            '\n'
            'incidence'
            '\n'
            'of death'
            '\n'
            'from GBD'
            '\n'
            '2010-2019',
            'Estimated'
            '\n'
            'incidence '
            '\n'
            'of death '
            '\n'
            'from '
            '\n'
            'Samuel '
            '\n'
            'et al. 2012',
            'Estimated'
            '\n'
            ' incidence'
            '\n'
            'of death'
            '\n'
            'from WHO',
            'Estimated'
            '\n'
            ' incidence'
            '\n'
            'of death'
            '\n'
            'from model'])
plt.ylabel('Incidence per 100,000')
plt.title(f"The model's predicted incidence of RTI related death "
          f"\n"
          f"compared to the various estimates for Malawi"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Incidence_of_rti_deaths_compare_hospital_police_samuel_who.png',
                bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# Plot the number of deaths and number of injuries predicted by the model compared to the GBD data
GBD_death_data = pd.read_csv('resources/ResourceFile_Deaths_And_Causes_DeathRates_GBD.csv')
road_data = GBD_death_data.loc[GBD_death_data['cause_name'] == 'Road injuries']
road_data_2010 = road_data.loc[road_data['year'] == 2010]
Malawi_Pop_2010 = pd.read_csv('resources/ResourceFile_Population_2010.csv')
Malawi_pop_size_2010 = sum(Malawi_Pop_2010['Count'])
scaler_to_pop_size = Malawi_pop_size_2010 / pop_size
scaled_2010_deaths = np.mean(deaths_2010) * scaler_to_pop_size
injury_number_data = pd.read_csv('resources/ResourceFile_RTI_GBD_Injury_Categories.csv')
injury_number_data = injury_number_data.loc[injury_number_data['metric'] == 'Number']
injury_number_in_2010 = injury_number_data.loc[injury_number_data['year'] == 2010]
gbd_number_of_injuries_2010 = injury_number_in_2010['val'].sum()
model_injury_number_in_2010 = np.mean(injuries_in_2010)
scaled_model_injury_number_in_2010 = model_injury_number_in_2010 * scaler_to_pop_size
# plot the number of deaths
plt.bar(np.arange(2), [scaled_2010_deaths, sum(road_data_2010['val'])], color='lightsteelblue')
plt.ylabel('Number of deaths')
plt.xticks(np.arange(2), ['Scaled model deaths', 'GBD estimated'
                                                 '\n'
                                                 'number of deaths'
                                                 '\n'
                                                 'for 2010'])
plt.title(f"The model's predicted number of RTI related death "
          f"\n"
          f"compared to the GBD 2010 estimate"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Number_of_deaths_comp_to_GBD_2010.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# plot the model number of deaths and number of injuries compared to GBD estimates
injury_output = [scaled_model_injury_number_in_2010, gbd_number_of_injuries_2010]
plt.bar(np.arange(2), injury_output, color='lightsalmon', label='number of injuries')
plt.xticks(np.arange(2), ['Model output', 'GBD estimate'])
plt.ylabel('Number')
plt.title(f"The model's predicted number of road traffic injuries"
          f"\n"
          f"compared to the GBD 2010 estimates"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Number_of_RTI_comp_to_GBD_2010.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# Extrapolate the number of injuries of those who were dead on arival to include to the estimate
average_number_of_injuries_of_those_DOA = 2.9 * np.mean(number_of_prehospital_deaths_2010)
average_number_of_injuries_of_those_DOA_scaled = scaler_to_pop_size * average_number_of_injuries_of_those_DOA
extrapolated_injury_output = [scaled_model_injury_number_in_2010,
                              scaled_model_injury_number_in_2010 + average_number_of_injuries_of_those_DOA_scaled,
                              gbd_number_of_injuries_2010]
plt.bar(np.arange(3), extrapolated_injury_output, color='lightsalmon', label='number of injuries')
plt.xticks(np.arange(3), ['Model output', 'Model output with extrapolation', 'GBD estimate'])
plt.ylabel('Number')
plt.title(f"The model's predicted number of road traffic injuries"
          f"\n"
          f"compared to the GBD 2010 estimates"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Extrapolated_number_of_RTI_comp_to_GBD_2010.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# Compare model deaths over popsize for model and Malawi GBD data
deaths_over_pop_size = [np.mean(deaths_2010) / pop_size,
                        sum(road_data_2010['val']) / Malawi_pop_size_2010]
plt.bar(np.arange(2), deaths_over_pop_size, color='lightsalmon')
plt.ylabel('Deaths/Population')
plt.xticks(np.arange(2), ['Model deaths'
                          '\n'
                          'divided by \n'
                          'simulation population',
                          'GBD estimated'
                          '\n'
                          'number of deaths'
                          '\n'
                          'divided by population'
                          '\n'
                          '2010'])
plt.title(f"The model's predicted Number of RTI related death divided by population size"
          f"\n"
          f"compared to the GBD 2010 estimate"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")  # Plot the overall percent fatality of those involved in road traffic injuries
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Number_of_deaths_over_pop_comp_to_GBD_2010.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# compare incidence of death in the model and the GBD 2010 estimate
data = pd.read_csv('resources/ResourceFile_RTI_GBD_Number_And_Incidence_Data.csv')
data = data.loc[data['metric'] == 'Rate']
data = data.loc[data['measure'] == 'Deaths']
data = data.loc[data['year'] > 2009]
gbd_mean_inc_death_2010_to_2019 = data.val.mean()
plt.bar(np.arange(2), [np.mean(average_deaths), gbd_mean_inc_death_2010_to_2019], color='wheat')
plt.ylabel('Incidence of deaths')
plt.xticks(np.arange(2), ['Model incidence of death', 'Average incidence of death'
                                                      '\n'
                                                      'of GBD estimates'
                                                      '\n'
                                                      '2010 to 2019'])
plt.title(f"The model's predicted incidence of RTI related death "
          f"\n"
          f"compared to the average incidence of death from GBD study 2010-2019"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Incidence_of_deaths_comp_to_GBD_2010.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
mean_fatal_crashes_of_all_sim = np.mean(percent_of_fatal_crashes)
std_fatal_crashes = np.std(percent_of_fatal_crashes)
non_fatal_crashes_of_all_sim = [i - j for i, j in zip(np.ones(len(percent_of_fatal_crashes)), percent_of_fatal_crashes)]
mean_non_fatal = np.mean(non_fatal_crashes_of_all_sim)
std_non_fatal_crashes = np.std(non_fatal_crashes_of_all_sim)
data = [np.round(mean_fatal_crashes_of_all_sim, 3), np.round(mean_non_fatal, 3)]
n = np.arange(2)
plt.bar(n, data, yerr=[std_fatal_crashes, std_non_fatal_crashes], color='lightsteelblue')
for i in range(len(data)):
    plt.annotate(str(data[i]), xy=(n[i], data[i]), ha='center', va='bottom')
plt.xticks(np.arange(2), ['fatal', 'non-fatal'])
plt.title(f"Average percentage of those with RTI who perished"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.savefig('outputs/Demographics_of_RTI/Percentage_of_deaths.png', bbox_inches='tight')
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Percentage_of_deaths.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# plot the percentage of the context of those who died in a road traffic accident
colours = ['lightsteelblue', 'lightsalmon', 'gold', 'lemonchiffon']
plt.pie([np.mean(ps_of_imm_death), np.mean(ps_of_death_post_med), np.mean(ps_of_death_without_med),
         np.mean(ps_of_death_unavailable_med)],
        labels=['Death on scene', 'Death post med', 'Death without med', 'Death due to unavailable med'],
        autopct='%1.1f%%', startangle=90, colors=colours)
plt.title(f"Average cause of death breakdown in RTI"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.axis('equal')
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Percentage_cause_of_deaths.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# Plot the distribution of injury severity
n = np.arange(2)
data = [np.round(np.mean(perc_mild), 3), np.round(np.mean(perc_severe), 3)]
plt.bar(n, data, yerr=[np.std(perc_mild), np.std(perc_severe)], color='lightsteelblue')
for i in range(len(data)):
    plt.annotate(str(data[i]), xy=(n[i], data[i]), ha='center', va='bottom')
plt.xticks(np.arange(2), labels=['Mild injuries', 'Severe injuries'])
plt.title(f"Average road traffic injury severity distribution"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Percentage_mild_severe_injuries.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# Plot the percentage of mild and severe injuries to other sources
# Police data from Schlottmann et al doi: 10.4314/mmj.v29i4.4
plt.bar(np.arange(2), data, color='lightsteelblue', width=0.4, label='Model data')
plt.bar(np.arange(2)+0.4, [0.64, 0.36], color='lightsalmon', width=0.4, label='Police data')
plt.xticks(np.arange(2) + 0.2, ['Mild', 'Severe'])
plt.ylabel('Percent')
plt.legend()
plt.title(f"Average road traffic injury severity distribution compared to police data"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Percentage_mild_severe_injuries_comparison.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# Plot the distribution of the ISS scores
flattened_scores = [score for sublist in iss_scores for score in sublist]
scores, counts = np.unique(flattened_scores, return_counts=True)
distribution_of_scores = counts / sum(counts)
plt.bar(scores, distribution_of_scores, width=0.8, color='lightsteelblue')
plt.xlabel('ISS scores')
plt.ylabel('Percentage')
plt.title(f"Average road traffic injury ISS score distribution"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.xlim([0, 75])
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Average_ISS_scores.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# Show the cumulative distribution of ISS scores
injury_severity_cumulative_sum = np.cumsum(distribution_of_scores)
icu_cut_off_index = list(filter(lambda k: k > 1 - 0.027, injury_severity_cumulative_sum))[0]
hdu_cut_off_index = list(filter(lambda k: k > 1 - (0.027 + 0.033), injury_severity_cumulative_sum))[0]
index_for_ICU = list(injury_severity_cumulative_sum).index(icu_cut_off_index)
index_for_HCU = list(injury_severity_cumulative_sum).index(hdu_cut_off_index)
score_for_ICU = scores[index_for_ICU]
score_for_HCU = scores[index_for_HCU]
plt.scatter(scores, injury_severity_cumulative_sum, color='lightsteelblue')
plt.hlines(hdu_cut_off_index, 0, score_for_HCU, colors='r', linestyles='dashed')
plt.vlines(score_for_HCU, 0, hdu_cut_off_index, colors='r', linestyles='dashed', label=f"Cut off score for HDU:\n"
                                                                                       f"{score_for_HCU}")
plt.hlines(icu_cut_off_index, 0, score_for_ICU, colors='g', linestyles='dashed')
plt.vlines(score_for_ICU, 0, icu_cut_off_index, colors='g', linestyles='dashed', label=f"Cut off score for ICU:\n"
                                                                                       f"{score_for_ICU}")
plt.xlabel('ISS scores')
plt.ylabel('Cumulative sum')
plt.legend()
plt.title(f"Cumulative summation of ISS scores and scores for HDU and ICU admission"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.xlim([0, 75])
plt.ylim([0, 1])
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/ISS_scores_ICU_HDU_admission.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# Plot the distribution of the number of injured body regions
average_number_of_body_regions_injured = [float(sum(col)) / len(col) for col in zip(*number_of_injured_body_locations)]
plt.bar(np.arange(8), np.divide(average_number_of_body_regions_injured, sum(average_number_of_body_regions_injured)),
        color='lightsteelblue')
plt.xticks(np.arange(8), ['1', '2', '3', '4', '5', '6', '7', '8'])
plt.xlabel('Number of injured AIS body regions')
plt.ylabel('Percentage')
plt.title(f"Average injured body region distribution"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Average_injured_body_region_distribution.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# plot the injury location data

average_inj_loc = [float(sum(col)) / len(col) for col in zip(*inj_loc_data)]
plt.bar(np.arange(8), np.divide(average_inj_loc, sum(average_inj_loc)), color='lightsteelblue')
plt.xticks(np.arange(8), ['Head', 'Face', 'Neck', 'Thorax', 'Abdomen', 'Spine', 'UpperX', 'LowerX'], rotation=45)
plt.xlabel('AIS body regions')
plt.ylabel('Percentage')
plt.title(f"Average injury location distribution"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Average_injury_location_distribution.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# Plot the injury category data
average_inj_cat = [float(sum(col)) / len(col) for col in zip(*inj_cat_data)]
plt.bar(np.arange(len(average_inj_cat)), np.divide(average_inj_cat, sum(average_inj_cat)), color='lightsteelblue')
plt.xticks(np.arange(len(average_inj_cat)), ['Fracture', 'Dislocation', 'TBI', 'Soft Tissue Inj.', 'Int. Organ Inj.',
                                             'Int. Bleeding', 'SCI', 'Amputation', 'Eye injury', 'Laceration', 'Burn'],
           rotation=90)
plt.ylabel('Percentage')
plt.title(f"Average injury category distribution"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Average_injury_category_distribution.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# Plot the incidence of injuries as per the GBD definitions
average_inc_amputations = [float(sum(col)) / len(col) for col in zip(*inc_amputations)]
mean_inc_amp = np.mean(average_inc_amputations)
std_amp = np.std(average_inc_amputations)
gbd_inc_amp = 5.85
gbd_perc_amp_2010 = 1.41
average_inc_burns = [float(sum(col)) / len(col) for col in zip(*inc_burns)]
mean_inc_burns = np.mean(average_inc_burns)
std_burns = np.std(average_inc_burns)
gbd_inc_burns = 5.88
average_inc_fractures = [float(sum(col)) / len(col) for col in zip(*inc_fractures)]
mean_inc_fractures = np.mean(average_inc_fractures)
std_fractures = np.std(average_inc_fractures)
gbd_inc_fractures = 139.76
average_inc_tbi = [float(sum(col)) / len(col) for col in zip(*inc_tbi)]
mean_inc_tbi = np.mean(average_inc_tbi)
std_tbi = np.std(average_inc_tbi)
gbd_inc_tbi = 46.19
average_inc_sci = [float(sum(col)) / len(col) for col in zip(*inc_sci)]
mean_inc_sci = np.mean(average_inc_sci)
std_sci = np.std(average_inc_sci)
gbd_inc_sci = 0.62
average_inc_minor = [float(sum(col)) / len(col) for col in zip(*inc_minor)]
mean_inc_minor = np.mean(average_inc_minor)
std_minor = np.std(average_inc_minor)
gbd_inc_minor = 126.35
average_inc_other = [float(sum(col)) / len(col) for col in zip(*inc_other)]
mean_inc_other = np.mean(average_inc_other)
std_other = np.std(average_inc_other)
gbd_inc_other = 52.93
average_inc_total = [float(sum(col)) / len(col) for col in zip(*tot_inc_injuries)]
mean_inc_total = np.mean(average_inc_total)
std_total = np.std(average_inc_total)
gbd_total = gbd_inc_amp + gbd_inc_burns + gbd_inc_fractures + gbd_inc_minor + gbd_inc_other + gbd_inc_sci + gbd_inc_tbi
incidence_dict = {'amputations': [gbd_inc_amp, mean_inc_amp],
                  'burns': [gbd_inc_burns, mean_inc_burns],
                  'fractures': [gbd_inc_fractures, mean_inc_fractures],
                  'tbi': [gbd_inc_tbi, mean_inc_tbi],
                  'sci': [gbd_inc_sci, mean_inc_sci],
                  'minor': [gbd_inc_minor, mean_inc_minor],
                  'other': [gbd_inc_other, mean_inc_other],
                  'total': [gbd_total, mean_inc_total]}
incidence_dict_perc = {'amputations': [gbd_inc_amp / gbd_total, mean_inc_amp / mean_inc_total],
                       'burns': [gbd_inc_burns / gbd_total, mean_inc_burns / mean_inc_total],
                       'fractures': [gbd_inc_fractures / gbd_total, mean_inc_fractures / mean_inc_total],
                       'tbi': [gbd_inc_tbi / gbd_total, mean_inc_tbi / mean_inc_total],
                       'sci': [gbd_inc_sci / gbd_total, mean_inc_sci / mean_inc_total],
                       'minor': [gbd_inc_minor / gbd_total, mean_inc_minor / mean_inc_total],
                       'other': [gbd_inc_other / gbd_total, mean_inc_other / mean_inc_total],
                       'total': [gbd_total / gbd_total, mean_inc_total / mean_inc_total]}
print('incidence of categories, GBD then Model:')

print(incidence_dict)
print(incidence_dict_perc)
model_category_incidences = [mean_inc_amp, mean_inc_burns, mean_inc_fractures, mean_inc_tbi, mean_inc_sci,
                             mean_inc_minor, mean_inc_other, mean_inc_total]
model_inc_errors = [std_amp, std_burns, std_fractures, std_tbi, std_sci, std_minor, std_other, std_total]
gbd_category_incidences = [gbd_inc_amp, gbd_inc_burns, gbd_inc_fractures, gbd_inc_tbi, gbd_inc_sci, gbd_inc_minor,
                           gbd_inc_other, gbd_total]
width = 0.35
plt.bar(np.arange(len(model_category_incidences)), model_category_incidences, width, color='lightsteelblue',
        yerr=model_inc_errors, label='Model output')
plt.bar(np.arange(len(model_category_incidences)) + width, gbd_category_incidences, width, color='lightsalmon',
        label='GBD 2017 data')
plt.title(f"Average injury incidence compared to GBD data"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.ylabel('Incidence per 100,000 person years')
labels = ['Amputations', 'Burns', 'Fractures', 'TBI', 'SCI', 'Minor', 'Other', 'Total']
plt.xticks(np.arange(len(model_category_incidences)) + width / 2, labels, rotation=45)
plt.legend()
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Average_injury_incidence_per_100000_bar.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# Plotting the average incidence of various injury categories
# plt.plot(injury_category_incidence['date'], average_inc_amputations, color='red', label='Amputations')
plt.hlines(mean_inc_amp, injury_category_incidence['date'].iloc[0], injury_category_incidence['date'].iloc[-1],
           color='red', label=f"Average incidence amputation = {mean_inc_amp}")
# plt.plot(injury_category_incidence['date'], average_inc_burns, color='blue', label='Burns')
plt.hlines(mean_inc_burns, injury_category_incidence['date'].iloc[0], injury_category_incidence['date'].iloc[-1],
           color='blue', label=f"Average incidence burns = {mean_inc_burns}")
# plt.plot(injury_category_incidence['date'], average_inc_fractures, color='yellow', label='Fractures')
plt.hlines(mean_inc_fractures, injury_category_incidence['date'].iloc[0], injury_category_incidence['date'].iloc[-1],
           color='yellow', label=f"Average incidence fractures = {mean_inc_fractures}")
# plt.plot(injury_category_incidence['date'], average_inc_tbi, color='green', label='TBI')
plt.hlines(mean_inc_tbi, injury_category_incidence['date'].iloc[0], injury_category_incidence['date'].iloc[-1],
           color='green', label=f"Average incidence TBI = {mean_inc_tbi}")
# plt.plot(injury_category_incidence['date'], average_inc_sci, color='pink', label='SCI')
plt.hlines(mean_inc_sci, injury_category_incidence['date'].iloc[0], injury_category_incidence['date'].iloc[-1],
           color='pink', label=f"Average incidence SCI = {mean_inc_sci}")
# plt.plot(injury_category_incidence['date'], average_inc_minor, color='darkseagreen', label='Minor')
plt.hlines(mean_inc_minor, injury_category_incidence['date'].iloc[0], injury_category_incidence['date'].iloc[-1],
           color='darkseagreen', label=f"Average incidence Minor = {mean_inc_minor}")
# plt.plot(injury_category_incidence['date'], average_inc_other, color='gold', label='Other')
plt.hlines(mean_inc_other, injury_category_incidence['date'].iloc[0], injury_category_incidence['date'].iloc[-1],
           color='gold', label=f"Average incidence other = {mean_inc_other}")
# plt.plot(injury_category_incidence['date'], average_inc_total, color='black', label='Total')
plt.hlines(mean_inc_total, injury_category_incidence['date'].iloc[0], injury_category_incidence['date'].iloc[-1],
           color='black', label=f"Average incidence total = {mean_inc_total}")
plt.xlabel('Time')
plt.ylabel('Incidence per 100,000')
plt.title(f"Average injury incidence by GBD categories"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.legend()
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Average_injury_incidence_per_100000.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()

# Plot percentage of injury by type in model compared to the GBD estimates
model_injury_types = [mean_inc_amp, mean_inc_burns, mean_inc_fractures, mean_inc_tbi,
                      mean_inc_minor, mean_inc_other, mean_inc_sci]
model_injury_type_percentages = np.divide(model_injury_types, sum(model_injury_types))
model_injury_type_percentages = np.multiply(model_injury_type_percentages, 100)
gbd_inj_cat_data = pd.read_csv('resources/ResourceFile_RTI_GBD_Injury_Categories.csv')
gbd_inj_cat_data = gbd_inj_cat_data.loc[gbd_inj_cat_data['metric'] == 'Number']
gbd_inj_cat_data = gbd_inj_cat_data.groupby('rei').sum()
gbd_inj_cat_data['percentage'] = gbd_inj_cat_data['val'] / sum(gbd_inj_cat_data['val'])
gbd_percentages = np.multiply(gbd_inj_cat_data['percentage'].tolist(), 100)
labels = gbd_inj_cat_data['percentage'].index
plt.bar(np.arange(len(model_injury_type_percentages)), model_injury_type_percentages, width=0.4, color='lightsteelblue',
        label='Model %\nof injury\n incidence\n by type')
plt.bar(np.arange(len(gbd_percentages)) + 0.4, gbd_percentages, width=0.4, color='lightsalmon',
        label='GBD 2010 %\n of injury\n incidence\n by type')
plt.xticks(np.arange(len(model_injury_type_percentages)) + 0.2, labels, rotation=45)
plt.ylabel('Percentage')
plt.title(f"Model injury type predictions compared to GBD average over all years"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
plt.legend()
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Average_injury_type_distribution.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# ================================== Plot inpatient day distribution ==================================================
# Malawi injury inpatient days data from https://doi.org/10.1016/j.jsurg.2014.09.010
labels = ['< 1', '1', '2', '3', '4-7', '8-14', '15-30', '> 30']
inpatient_days_Tyson_et_al = [107 + 40, 854 + 56, 531 + 19, 365 + 22, 924 + 40, 705 + 23, 840 + 8, 555 + 11]
inpatient_days_Tyson_et_al_dist = np.divide(inpatient_days_Tyson_et_al, sum(inpatient_days_Tyson_et_al))
# Sort model data to fit the above boundaries
zero_days = [1 if inpatient_day == 0 else 0 for inpatient_day in all_sim_inpatient_days]
one_day = [1 if inpatient_day == 1 else 0 for inpatient_day in all_sim_inpatient_days]
two_days = [1 if inpatient_day == 2 else 0 for inpatient_day in all_sim_inpatient_days]
three_days = [1 if inpatient_day == 3 else 0 for inpatient_day in all_sim_inpatient_days]
four_to_seven_days = [1 if 4 <= inpatient_day < 7 else 0 for inpatient_day in all_sim_inpatient_days]
eight_to_fourteen = [1 if 8 <= inpatient_day < 14 else 0 for inpatient_day in all_sim_inpatient_days]
fifteen_to_thirty = [1 if 15 <= inpatient_day < 30 else 0 for inpatient_day in all_sim_inpatient_days]
thiry_plus = [1 if 30 <= inpatient_day else 0 for inpatient_day in all_sim_inpatient_days]
model_inpatient_days = [sum(zero_days), sum(one_day), sum(two_days), sum(three_days), sum(four_to_seven_days),
                        sum(eight_to_fourteen), sum(fifteen_to_thirty), sum(thiry_plus)]
model_inpatient_days_dist = np.divide(model_inpatient_days, sum(model_inpatient_days))
plt.bar(np.arange(len(inpatient_days_Tyson_et_al_dist)), inpatient_days_Tyson_et_al_dist, width=0.3,
        color='lightsalmon', label='Inpatient day data\nfrom Kamuza central hospital')
plt.bar(np.arange(len(model_inpatient_days_dist)) + 0.5, model_inpatient_days_dist, width=0.3,
        color='lightsteelblue', label='Model inpatient days')
plt.xticks(np.arange(len(model_inpatient_days_dist)) + 0.25, labels)
plt.xlabel('Inpatient days')
plt.ylabel('Percentage')
plt.legend()
plt.title(f"Model injury inpatient days compared to Kamuzu central hospital data"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")

if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Inpatient_day_distribution_comp_to_kamuzu.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# Compare inpatient admissions from the model to Kamuzu central hospital data
# Based on Purcel et al. DOI: 10.1007/s00268-020-05853-z
percent_admitted_kch = 89.8
percent_admitted_in_model = ((len(all_sim_inpatient_days) - sum(zero_days)) / len(all_sim_inpatient_days)) * 100
plt.bar(np.arange(2), [np.round(percent_admitted_in_model, 2), percent_admitted_kch], color='lightsteelblue')
plt.xticks(np.arange(2), ['Percent admitted in model', 'Percent admitted in KCH'])
plt.ylabel('Percentage')
plt.title(f"Model percentage inpatient admission compared to Kamuzu central hospital data"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")

if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Inpatient_admission_comp_to_kamuzu.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# Plot the distribution of inpatient days due to RTI
days, counts = np.unique(all_sim_inpatient_days, return_counts=True)
plt.bar(days, counts / sum(counts), width=0.8, color='lightsteelblue')
plt.xlabel('Inpatient days')
plt.ylabel('Percentage of patients')
plt.title(f"Distribution of inpatient days produced by the model for RTIs"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Inpatient_day_distribution.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()

# Plot overall health system usage
mean_consumables_used_per_sim = np.mean(list_consumables_dict)
mean_inpatient_days = np.mean([np.mean(sim_days) for sim_days in per_sim_inpatient_days])
mean_fraction_health_system_time_used = np.mean(health_system_time_usage)
average_number_of_burns_treated = np.mean(per_sim_burn_treated)
average_number_of_fractures_treated = np.mean(per_sim_frac_cast)
average_number_of_lacerations_treated = np.mean(per_sim_laceration)
average_number_of_major_surgeries_performed = np.mean(per_sim_major_surg)
average_number_of_minor_surgeries_performed = np.mean(per_sim_minor_surg)
average_number_of_tetanus_jabs = np.mean(per_sim_tetanus)
average_number_of_pain_meds = np.mean(per_sim_pain_med)
# plot the number of consumables used
plt.bar(np.arange(1), mean_consumables_used_per_sim, color='lightsteelblue')
plt.xticks(np.arange(1), ['Consumables'])
plt.ylabel('Average consumables used')
plt.title(f"Average number of consumables used per sim"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Average_consumables_used.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# plot the average total number of inpatient days taken
plt.bar(np.arange(1), mean_inpatient_days, color='lightsalmon')
plt.xticks(np.arange(1), ['Inpatient days'])
plt.ylabel('Average total inpatient days used')
plt.title(f"Average number of inpatient days used per sim"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Average_inpatient_days_used.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# plot the average health system time usage
plt.bar(np.arange(1), mean_fraction_health_system_time_used, color='wheat')
plt.xticks(np.arange(1), ['Fraction of time'])
plt.ylabel('Average health system time usage')
plt.title(f"Average fraction of total health system time usage per sim"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Average_health_sys_time_used.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# plot the average number of treatments provided
data = [average_number_of_burns_treated,
        average_number_of_fractures_treated,
        average_number_of_lacerations_treated,
        average_number_of_major_surgeries_performed,
        average_number_of_minor_surgeries_performed,
        average_number_of_tetanus_jabs,
        average_number_of_pain_meds]
labels = ['Burn\nmanagement', 'Fracture\ncast', 'Suture', 'Major\nsurgery', 'Minor\nsurgery', 'Tetanus\nvaccine',
          'Pain\nmanagement']

plt.bar(np.arange(len(data)), data, color='cornflowerblue')
plt.xticks(np.arange(len(data)), labels)
plt.ylabel('Average number of appointments')
plt.title(f"Average number of HSI events performed per sim"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Average_HSI_appointments_per_sim.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()

# === Plot incidence of road traffic injuries in children===
data = pd.read_csv('resources/ResourceFile_RTI_Incidence_of_rti_per_100000_children.csv')
data = data.dropna(subset=['Incidence per 100,000'])
data.sort_values(by=['Incidence per 100,000'], inplace=True)
average_inc_in_children = np.mean([float(sum(col)) / len(col) for col in zip(*incidences_of_rti_in_children)])
weighted_average_from_Hyder_et_al = 110.81
plt.bar(np.arange(len(data)), data['Incidence per 100,000'], color='lightsalmon')
plt.xticks(np.arange(len(data)), data['Source'], rotation=90)
plt.ylabel('Incidence per 100,000')
plt.title('Reported incidences of RTIs in children aged 0-18 in SSA')
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/RTI_incidence_in_children.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()

plt.bar(np.arange(2), [weighted_average_from_Hyder_et_al, average_inc_in_children])
plt.xticks(np.arange(2), ['Weighted average', 'Model'])
plt.ylabel('Incidence per 100,000')
plt.title(f"Weighted average of RTI incidence in children from Hyder et al\n"
          f"compared to the model\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}"
          )

if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Incidence_of_RTI_in_children_model_comparison.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# ============== Plot number of injuries compared to Sundet et al =========================================
# DOI: 10.1177/0049475518808969
Sundet_n_patients = 4776
Sundet_n_injuries = 7057
Sundet_injuries_per_patient = Sundet_n_injuries / Sundet_n_patients
total_number_of_injuries = sum(number_of_injuries_per_sim)
Model_injuries_per_patient = total_number_of_injuries / total_n_crashes
plt.bar(np.arange(2), [Sundet_injuries_per_patient, Model_injuries_per_patient], color='lightsteelblue')
plt.xticks(np.arange(2), ['Injuries per person\nSundet et al. 2018', 'Injuries per person\n model'])
plt.title(f"Injuries per person, Model compared to Sundet et al. 2018"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/Injuries_per_person_model_Sundet_comp.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()

# Plot the ICU patient characteristics
mean_perc_frac = np.mean(ICU_frac)
mean_perc_dis = np.mean(ICU_dis)
mean_perc_tbi = np.mean(ICU_tbi)
mean_perc_soft = np.mean(ICU_soft)
mean_perc_int_o = np.mean(ICU_int_o)
mean_perc_int_b = np.mean(ICU_int_b)
mean_perc_sci = np.mean(ICU_sci)
mean_perc_amp = np.mean(ICU_amp)
mean_perc_eye = np.mean(ICU_eye)
mean_perc_lac = np.mean(ICU_lac)
mean_perc_burn = np.mean(ICU_burn)
data = [mean_perc_frac, mean_perc_dis, mean_perc_tbi, mean_perc_soft, mean_perc_int_o, mean_perc_int_b, mean_perc_sci,
        mean_perc_amp, mean_perc_eye, mean_perc_lac, mean_perc_burn]
labels = ['Fractures', 'Dislocations', 'TBI', 'Soft tissue injury', 'Internal organ injury', 'Internal bleeding',
          'SCI', 'Amputation', 'Eye injury', 'Laceration', 'Burn']
plt.bar(np.arange(len(labels)), data, color='lightsteelblue')
plt.ylabel('Percent injury type in ICU patients')
plt.title(f"Injuries in ICU patients by type"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/ICU_injury_categories.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# Plot the ICU injury characteristics compared to a Tanzanian ICU unit
lacerations_percent = 97.8
fractures_percent = 32.4
tbi_percent = 21.5
visceral_injury_percent = 13.1
burns_percent = 2.9
other_injuries = 3.8
tanzanian_data = [lacerations_percent, fractures_percent, tbi_percent, visceral_injury_percent, burns_percent,
                  other_injuries]
model_equiv_data = [mean_perc_lac, mean_perc_frac, mean_perc_tbi, mean_perc_int_o + mean_perc_int_b, mean_perc_burn,
                    mean_perc_soft + mean_perc_amp + mean_perc_eye + mean_perc_dis + mean_perc_sci]
labels = ['Lacerations', 'Fractures', 'TBI', 'Visceral injury', 'Burns', 'Other']
plt.bar(np.arange(len(tanzanian_data)), tanzanian_data, label='ICU data', color='lightsalmon', width=0.4)
plt.bar(np.arange(len(model_equiv_data)) + 0.4, model_equiv_data, label='Model output', color='lightsteelblue',
        width=0.4)
plt.xticks(np.arange(len(model_equiv_data)) + 0.2, labels, rotation=45)
plt.legend()
plt.title(f"Injuries in ICU patients by type compared to Chalya et al. 2011"
          f"\n"
          f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
if save_figures is True:
    plt.savefig('outputs/Demographics_of_RTI/ICU_injury_categories_comp_chalya.png', bbox_inches='tight')
    plt.clf()
else:
    plt.clf()
# ======================================= Create outputs for GBD DATA ===============================================
# The following code is commented out as it relates to GBD data and not the simulation data
# data = pd.read_csv('resources/ResourceFile_RTI_GBD_Number_And_Incidence_Data.csv')
# gbd_death_data = data.loc[data['measure'] == 'Deaths']
# gbd_in_rti_data = data.loc[data['measure'] == 'Incidence']
# gbd_death_data = gbd_death_data.sort_values(by='year')
# gbd_in_rti_data = gbd_in_rti_data.sort_values(by='year')
# gbd_death_number = gbd_death_data.loc[gbd_death_data['metric'] == 'Number']
# gbd_rti_number = gbd_in_rti_data.loc[gbd_in_rti_data['metric'] == 'Number']
# plt.subplot(2, 1, 1)
# plt.plot(gbd_rti_number['year'], gbd_rti_number['val'], 'lightsteelblue', label='Number of RTIs')
# plt.fill_between(gbd_rti_number['year'], gbd_rti_number['upper'], gbd_rti_number['lower'], color='lightsteelblue',
#                  alpha=0.5, label='95% C.I.')
# plt.legend()
# plt.xlabel('Years')
# plt.ylabel('Number')
# plt.title('Number of RTIs in Malawi, GBD estimates')
# plt.subplot(2, 1, 2)
# plt.plot(gbd_death_number['year'], gbd_death_number['val'], 'lightsalmon', label='Deaths')
#
# plt.fill_between(gbd_death_number['year'], gbd_death_number['upper'], gbd_death_number['lower'], color='lightsalmon',
#                  alpha=0.5, label='95% C.I.')
# plt.legend()
# plt.xlabel('Years')
# plt.ylabel('Number')
# plt.title('Number of RTI deaths in Malawi, GBD estimates')
# left  = 0.125  # the left side of the subplots of the figure
# right = 0.9    # the right side of the subplots of the figure
# bottom = 0.1   # the bottom of the subplots of the figure
# top = 0.9      # the top of the subplots of the figure
# wspace = 0.2   # the amount of width reserved for blank space between subplots
# hspace = 0.45   # the amount of height reserved for white space between subplots
# plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=0.2, hspace=hspace)
# plt.savefig('outputs/Demographics_of_RTI/GBD_RTI_number_of_rtis_and_deaths.png', bbox_inches='tight')
# plt.clf()
# per_injury_fatal_ratio = np.divide(gbd_death_number['val'].tolist(), gbd_rti_number['val'].tolist())
# plt.plot(gbd_death_number['year'], per_injury_fatal_ratio, 'lightsteelblue')
# plt.xlabel('Year')
# plt.ylabel('Percent')
# plt.title('Number of deaths per-injury in Malawi, GBD estimates')
# plt.savefig('outputs/Demographics_of_RTI/GBD_percent_fatal_injuries.png', bbox_inches='tight')
# plt.clf()
# gbd_gender_data = pd.read_csv('resources/ResourceFile_RTI_GBD_gender_data.csv')
# global_males = gbd_gender_data.loc[(gbd_gender_data['location'] == 'Global') & (gbd_gender_data['sex'] == 'Male')]
# global_males_in_rti = global_males['val'].sum()
# global_females = gbd_gender_data.loc[(gbd_gender_data['location'] == 'Global') & (gbd_gender_data['sex'] == 'Female')]
# global_females_in_rti = global_females['val'].sum()
# global_gender_percentages = [global_males_in_rti / (global_females_in_rti + global_males_in_rti),
#                              global_females_in_rti / (global_females_in_rti + global_males_in_rti)]
# gbd_age_data = pd.read_csv('resources/ResourceFile_RTI_GBD_age_data.csv')
# age_1_to_4 = gbd_age_data.loc[gbd_age_data['age'] == '1 to 4']
# age_5_to_9 = gbd_age_data.loc[gbd_age_data['age'] == '5 to 9']
# age_10_to_14 = gbd_age_data.loc[gbd_age_data['age'] == '10 to 14']
# age_15_to_19 = gbd_age_data.loc[gbd_age_data['age'] == '15 to 19']
# age_20_to_24 = gbd_age_data.loc[gbd_age_data['age'] == '20 to 24']
# age_25_to_29 = gbd_age_data.loc[gbd_age_data['age'] == '25 to 29']
# age_30_to_34 = gbd_age_data.loc[gbd_age_data['age'] == '30 to 34']
# age_35_to_39 = gbd_age_data.loc[gbd_age_data['age'] == '35 to 39']
# age_40_to_44 = gbd_age_data.loc[gbd_age_data['age'] == '40 to 44']
# age_45_to_49 = gbd_age_data.loc[gbd_age_data['age'] == '45 to 49']
# age_50_to_54 = gbd_age_data.loc[gbd_age_data['age'] == '50 to 54']
# age_55_to_59 = gbd_age_data.loc[gbd_age_data['age'] == '55 to 59']
# age_60_to_64 = gbd_age_data.loc[gbd_age_data['age'] == '60 to 64']
# age_65_to_69 = gbd_age_data.loc[gbd_age_data['age'] == '65 to 69']
# age_70_to_74 = gbd_age_data.loc[gbd_age_data['age'] == '70 to 74']
# age_75_to_79 = gbd_age_data.loc[gbd_age_data['age'] == '75 to 79']
# age_80_to_84 = gbd_age_data.loc[gbd_age_data['age'] == '80 to 84']
# age_85_to_89 = gbd_age_data.loc[gbd_age_data['age'] == '85 to 89']
# global_total = age_1_to_4.loc[age_1_to_4['location'] == 'Global']['val'].sum() + \
#                age_5_to_9.loc[age_5_to_9['location'] == 'Global']['val'].sum() + \
#                age_10_to_14.loc[age_10_to_14['location'] == 'Global']['val'].sum() + \
#                age_15_to_19.loc[age_15_to_19['location'] == 'Global']['val'].sum() + \
#                age_20_to_24.loc[age_20_to_24['location'] == 'Global']['val'].sum() + \
#                age_25_to_29.loc[age_25_to_29['location'] == 'Global']['val'].sum() + \
#                age_30_to_34.loc[age_30_to_34['location'] == 'Global']['val'].sum() + \
#                age_35_to_39.loc[age_35_to_39['location'] == 'Global']['val'].sum() + \
#                age_40_to_44.loc[age_40_to_44['location'] == 'Global']['val'].sum() + \
#                age_45_to_49.loc[age_45_to_49['location'] == 'Global']['val'].sum() + \
#                age_50_to_54.loc[age_50_to_54['location'] == 'Global']['val'].sum() + \
#                age_55_to_59.loc[age_55_to_59['location'] == 'Global']['val'].sum() + \
#                age_60_to_64.loc[age_60_to_64['location'] == 'Global']['val'].sum() + \
#                age_65_to_69.loc[age_65_to_69['location'] == 'Global']['val'].sum() + \
#                age_70_to_74.loc[age_70_to_74['location'] == 'Global']['val'].sum() + \
#                age_75_to_79.loc[age_75_to_79['location'] == 'Global']['val'].sum() + \
#                age_80_to_84.loc[age_80_to_84['location'] == 'Global']['val'].sum() + \
#                age_85_to_89.loc[age_85_to_89['location'] == 'Global']['val'].sum()
# global_age_range = [age_1_to_4.loc[age_1_to_4['location'] == 'Global']['val'].sum(),
#                age_5_to_9.loc[age_5_to_9['location'] == 'Global']['val'].sum(),
#                age_10_to_14.loc[age_10_to_14['location'] == 'Global']['val'].sum(),
#                age_15_to_19.loc[age_15_to_19['location'] == 'Global']['val'].sum(),
#                age_20_to_24.loc[age_20_to_24['location'] == 'Global']['val'].sum(),
#                age_25_to_29.loc[age_25_to_29['location'] == 'Global']['val'].sum(),
#                age_30_to_34.loc[age_30_to_34['location'] == 'Global']['val'].sum(),
#                age_35_to_39.loc[age_35_to_39['location'] == 'Global']['val'].sum(),
#                age_40_to_44.loc[age_40_to_44['location'] == 'Global']['val'].sum(),
#                age_45_to_49.loc[age_45_to_49['location'] == 'Global']['val'].sum(),
#                age_50_to_54.loc[age_50_to_54['location'] == 'Global']['val'].sum(),
#                age_55_to_59.loc[age_55_to_59['location'] == 'Global']['val'].sum(),
#                age_60_to_64.loc[age_60_to_64['location'] == 'Global']['val'].sum(),
#                age_65_to_69.loc[age_65_to_69['location'] == 'Global']['val'].sum(),
#                age_70_to_74.loc[age_70_to_74['location'] == 'Global']['val'].sum(),
#                age_75_to_79.loc[age_75_to_79['location'] == 'Global']['val'].sum(),
#                age_80_to_84.loc[age_80_to_84['location'] == 'Global']['val'].sum(),
#                age_85_to_89.loc[age_85_to_89['location'] == 'Global']['val'].sum()
#                     ]
# global_age_distribution = np.divide(global_age_range, global_total)
# malawi_males = gbd_gender_data.loc[(gbd_gender_data['location'] == 'Malawi') & (gbd_gender_data['sex'] == 'Male')]
# malawi_males_in_rti = malawi_males['val'].sum()
# malawi_females = gbd_gender_data.loc[(gbd_gender_data['location'] == 'Malawi') & (gbd_gender_data['sex'] == 'Female')]
# malawi_females_in_rti = malawi_females['val'].sum()
# malawi_gender_percentages = [malawi_males_in_rti / (malawi_females_in_rti + malawi_males_in_rti),
#                              malawi_females_in_rti / (malawi_females_in_rti + malawi_males_in_rti)]
# malawi_total = age_1_to_4.loc[age_1_to_4['location'] == 'Malawi']['val'].sum() + \
#                age_5_to_9.loc[age_5_to_9['location'] == 'Malawi']['val'].sum() + \
#                age_10_to_14.loc[age_10_to_14['location'] == 'Malawi']['val'].sum() + \
#                age_15_to_19.loc[age_15_to_19['location'] == 'Malawi']['val'].sum() + \
#                age_20_to_24.loc[age_20_to_24['location'] == 'Malawi']['val'].sum() + \
#                age_25_to_29.loc[age_25_to_29['location'] == 'Malawi']['val'].sum() + \
#                age_30_to_34.loc[age_30_to_34['location'] == 'Malawi']['val'].sum() + \
#                age_35_to_39.loc[age_35_to_39['location'] == 'Malawi']['val'].sum() + \
#                age_40_to_44.loc[age_40_to_44['location'] == 'Malawi']['val'].sum() + \
#                age_45_to_49.loc[age_45_to_49['location'] == 'Malawi']['val'].sum() + \
#                age_50_to_54.loc[age_50_to_54['location'] == 'Malawi']['val'].sum() + \
#                age_55_to_59.loc[age_55_to_59['location'] == 'Malawi']['val'].sum() + \
#                age_60_to_64.loc[age_60_to_64['location'] == 'Malawi']['val'].sum() + \
#                age_65_to_69.loc[age_65_to_69['location'] == 'Malawi']['val'].sum() + \
#                age_70_to_74.loc[age_70_to_74['location'] == 'Malawi']['val'].sum() + \
#                age_75_to_79.loc[age_75_to_79['location'] == 'Malawi']['val'].sum() + \
#                age_80_to_84.loc[age_80_to_84['location'] == 'Malawi']['val'].sum() + \
#                age_85_to_89.loc[age_85_to_89['location'] == 'Malawi']['val'].sum()
# malawi_age_range = [age_1_to_4.loc[age_1_to_4['location'] == 'Malawi']['val'].sum(),
#                age_5_to_9.loc[age_5_to_9['location'] == 'Malawi']['val'].sum(),
#                age_10_to_14.loc[age_10_to_14['location'] == 'Malawi']['val'].sum(),
#                age_15_to_19.loc[age_15_to_19['location'] == 'Malawi']['val'].sum(),
#                age_20_to_24.loc[age_20_to_24['location'] == 'Malawi']['val'].sum(),
#                age_25_to_29.loc[age_25_to_29['location'] == 'Malawi']['val'].sum(),
#                age_30_to_34.loc[age_30_to_34['location'] == 'Malawi']['val'].sum(),
#                age_35_to_39.loc[age_35_to_39['location'] == 'Malawi']['val'].sum(),
#                age_40_to_44.loc[age_40_to_44['location'] == 'Malawi']['val'].sum(),
#                age_45_to_49.loc[age_45_to_49['location'] == 'Malawi']['val'].sum(),
#                age_50_to_54.loc[age_50_to_54['location'] == 'Malawi']['val'].sum(),
#                age_55_to_59.loc[age_55_to_59['location'] == 'Malawi']['val'].sum(),
#                age_60_to_64.loc[age_60_to_64['location'] == 'Malawi']['val'].sum(),
#                age_65_to_69.loc[age_65_to_69['location'] == 'Malawi']['val'].sum(),
#                age_70_to_74.loc[age_70_to_74['location'] == 'Malawi']['val'].sum(),
#                age_75_to_79.loc[age_75_to_79['location'] == 'Malawi']['val'].sum(),
#                age_80_to_84.loc[age_80_to_84['location'] == 'Malawi']['val'].sum(),
#                age_85_to_89.loc[age_85_to_89['location'] == 'Malawi']['val'].sum()
#                     ]
# malawi_age_distribution = np.divide(malawi_age_range, malawi_total)
#
# age_labels = ['1-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59',
#               '60-64', '65-69', '70-74', '75-79', '80-84', '85-89']
# plt.subplots(figsize=(18, 10))
#
# plt.subplot(2, 2, 1)
# colours = ['lightsteelblue', 'lightsalmon']
# plt.pie(global_gender_percentages, labels=['Males', 'Females'], autopct='%1.1f%%', startangle=90, colors=colours)
# plt.title('GBD global gender distribution'
#           '\n'
#           ' of RTI incidence, all years')
# plt.subplot(2, 2, 2)
# plt.bar(np.arange(len(global_age_distribution)), global_age_distribution, color='lightsteelblue')
# plt.xticks(np.arange(len(global_age_distribution)), age_labels, rotation=90)
# plt.ylabel('Percent')
# plt.title('GBD global age distribution '
#           '\n'
#           'of RTI incidence, all years')
# plt.subplot(2, 2, 3)
# plt.pie(malawi_gender_percentages, labels=['Males', 'Females'], autopct='%1.1f%%', startangle=90, colors=colours)
# plt.title('GBD Malawi gender distribution '
#           '\n'
#           'of RTI incidence, all years')
# plt.subplot(2, 2, 4)
# plt.bar(np.arange(len(malawi_age_distribution)), malawi_age_distribution, color='lightsteelblue')
# plt.xticks(np.arange(len(malawi_age_distribution)), age_labels, rotation=90)
# plt.ylabel('Percent')
# plt.title('GBD Malawi age distribution '
#           '\n'
#           'of RTI incidence, all years')
# plt.tight_layout()
# plt.savefig('outputs/Demographics_of_RTI/GBD_RTI_demography.png')
# plt.clf()
#
# gbd_gender_data = pd.read_csv('resources/ResourceFile_RTI_GBD_age_gender_data.csv')
# number_of_injuries = gbd_gender_data.loc[gbd_gender_data['metric'] == 'Number']
# number_males_injured = number_of_injuries.loc[number_of_injuries['sex'] == 'Male']
# number_females_injured = number_of_injuries.loc[number_of_injuries['sex'] == 'Female']
# rate_of_injuries = gbd_gender_data.loc[gbd_gender_data['metric'] == 'Rate']
# rate_males_injured = rate_of_injuries.loc[rate_of_injuries['sex'] == 'Male']
# rate_females_injured = rate_of_injuries.loc[rate_of_injuries['sex'] == 'Female']
# female_df = number_females_injured.groupby(['age']).sum()
# female_df['rank'] = [1, 3, 4, 5, 6, 7, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# female_df = female_df.sort_values(by=['rank'])
# ages = female_df.index.tolist()
# ages.reverse()
# female_number = female_df['val'].tolist()
# female_number.reverse()
# male_df = number_males_injured.groupby(['age']).sum()
# male_df['rank'] = [1, 3, 4, 5, 6, 7, 8, 9, 10, 2, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# male_df = male_df.sort_values(by=['rank'])
# male_number = male_df['val'].tolist()
# male_number.reverse()
# plt.close()
# plt.clf()
# plt.plot(number_males_injured.groupby(['year']).sum().index, number_males_injured.groupby(['year']).sum()['val'],
#          label='Males', color='lightsteelblue')
# plt.plot(number_females_injured.groupby(['year']).sum().index, number_females_injured.groupby(['year']).sum()['val'],
#          label='Females', color='lightsalmon')
# plt.plot(number_males_injured.groupby(['year']).sum().index,
#          number_males_injured.groupby(['year']).sum()['val'] + number_females_injured.groupby(['year']).sum()['val'],
#          label='Total', color='black')
# plt.xlabel('Year')
# plt.ylabel('Number of road traffic injuries')
# plt.title('Number of road traffic injuries in Malawi per year, GBD data')
# plt.legend(loc='center right')
# plt.savefig('outputs/Demographics_of_RTI/Malawi_Number_of_injuries.png')
# plt.clf()
# plt.barh(ages, male_number, alpha=0.5, label='Males', color='lightsteelblue')
# plt.barh(ages, np.multiply(female_number, -1), alpha=0.5, label='Females', color='lightsalmon')
# locs, labels = plt.xticks()
# plt.xticks(locs, np.sqrt(locs**2), fontsize=8)
# plt.title('Sum total of number of road traffic injuries in Malawi'
#           '\n'
#           'by age and sex over all years, GBD data')
# plt.xlabel('Number')
# plt.yticks(fontsize=7)
# plt.legend()
# plt.tight_layout()
# plt.savefig('outputs/Demographics_of_RTI/Malawi_Injury_Demographics.png')
# plt.clf()
# plt.barh(ages, np.divide(male_number, sum(male_number)), alpha=0.5, label='Males', color='lightsteelblue')
# plt.barh(ages, np.multiply(np.divide(female_number, sum(female_number)), -1), alpha=0.5,
#          label='Females', color='lightsalmon')
# locs, labels = plt.xticks()
# plt.xticks(locs, np.sqrt(locs**2), fontsize=8)
# plt.title('Distribution of number of road traffic injuries in Malawi'
#           '\n'
#           'by age and sex over all years, GBD data')
# plt.xlabel('Number')
# plt.yticks(fontsize=7)
# plt.legend()
# plt.tight_layout()
# plt.savefig('outputs/Demographics_of_RTI/Malawi_Injury_Demographics_percentage.png')
# plt.clf()
#
# gbd_cat_2017 = [24026.90542, 1082.276734, 7941.462531, 7578.726195, 7578.726195, 1825.22282, 106.8162861, 1004.93119,
#                 559.5158363, 10931.61332, 1712.892472]
# gbd_cat_2017_labels = ['Fracture', 'Dislocation', 'TBI', 'Soft Tissue Inj.', 'Int. Organ Inj.',
#                        'Int. Bleeding', 'SCI', 'Amputation', 'Eye injury', 'Laceration', 'Burn']
# plt.bar(np.arange(len(gbd_cat_2017)), np.divide(gbd_cat_2017, sum(gbd_cat_2017)))
# plt.xticks(np.arange(len(gbd_cat_2017)), gbd_cat_2017_labels, rotation=90)
# plt.title('GBD Injury categories Malawi 2017')
# plt.savefig('outputs/Demographics_of_RTI/GBD_injury_category_distribution.png', bbox_inches='tight')
# plt.clf()

# # Plot data on vehicle ownership vs death incidence in Africa
# df = pd.read_csv('resources/ResourceFile_RTI_Vehicle_Ownersip_Death_Data.csv', skipinitialspace=True)
# # Preprocessing
# df = df.dropna()
# df['n_vehicles'] = pd.to_numeric(df['n_vehicles'])
# df['adjusted_n_deaths'] = pd.to_numeric(df['adjusted_n_deaths'])
#
# def group_by_gdp(row):
#     if row['gdp_usd_per_capita'] < 1005:
#         val = 'Low income'
#     elif row['gdp_usd_per_capita'] < 3955:
#         val = 'Lower middle income'
#     elif row['gdp_usd_per_capita'] < 12235:
#         val = 'Upper middle income'
#     else:
#         val = 'High income'
#     return val
#
#
# df['income_index'] = df.apply(group_by_gdp, axis=1)
# # drop outliers
# df = df.drop(df.n_vehicles.nlargest(3).index)
# low_income_slope, low_income_intercept, r_value, low_income_p_value, std_err = \
#     stats.linregress(df.loc[df['income_index'] == 'Low income', 'n_vehicles'],
#                      df.loc[df['income_index'] == 'Low income', 'mortality_rate_2016'])
# low_middle_income_slope, low_middle_income_intercept, r_value, low_middle_income_p_value, std_err = \
#     stats.linregress(df.loc[df['income_index'] == 'Lower middle income', 'n_vehicles'],
#                      df.loc[df['income_index'] == 'Lower middle income', 'mortality_rate_2016'])
# upper_middle_income_slope, upper_middle_income_intercept, r_value, upper_middle_income_p_value, std_err = \
#     stats.linregress(df.loc[df['income_index'] == 'Upper middle income', 'n_vehicles'],
#                      df.loc[df['income_index'] == 'Upper middle income', 'mortality_rate_2016'])
# high_income_slope, high_income_intercept, r_value, high_income_p_value, std_err = \
#     stats.linregress(df.loc[df['income_index'] == 'High income', 'n_vehicles'],
#                      df.loc[df['income_index'] == 'High income', 'mortality_rate_2016'])
# groups = df.groupby('income_index')
# for name, group in groups:
#     plt.plot(group.n_vehicles, group.mortality_rate_2016, marker='o', linestyle='', markersize=12, label=name)
# plt.xlabel('Number of vehicles')
# plt.ylabel('Mortality rate per 100,000 people per year')
# plt.legend()
# plt.title('The number of vehicles vs the mortality rate due to RTI per 100,000, grouped by GDP')
# plt.savefig('outputs/Demographics_of_RTI/N_vehicles_vs_incidence_scatter.png', bbox_inches='tight')
# plt.clf()
# plt.subplot(2, 2, 1)
# plt.scatter(df.loc[df['income_index'] == 'Low income', 'n_vehicles'],
#             df.loc[df['income_index'] == 'Low income', 'mortality_rate_2016'], c='blue')
# plt.plot(df.loc[df['income_index'] == 'Low income', 'n_vehicles'],
#          low_income_intercept + low_income_slope * df.loc[df['income_index'] == 'Low income', 'n_vehicles'],
#          color='blue')
# plt.xlabel('Number of vehicles')
# plt.ylabel('Deaths per 100,000'
#            '\n'
#            ' population in 2016')
# plt.title(f"Low income, p = {np.round(low_income_p_value, 2)}")
# plt.subplot(2, 2, 2)
# plt.scatter(df.loc[df['income_index'] == 'Lower middle income', 'n_vehicles'],
#             df.loc[df['income_index'] == 'Lower middle income', 'mortality_rate_2016'], c='red')
# plt.plot(df.loc[df['income_index'] == 'Lower middle income', 'n_vehicles'],
#          low_middle_income_intercept +
#          low_middle_income_slope * df.loc[df['income_index'] == 'Lower middle income', 'n_vehicles'], color='red')
# plt.xlabel('Number of vehicles')
# plt.ylabel('Deaths per 100,000'
#            '\n'
#            ' population in 2016')
# plt.title(f"Lower middle income, p = {np.round(low_middle_income_p_value, 2)}")
# plt.subplot(2, 2, 3)
# plt.scatter(df.loc[df['income_index'] == 'Upper middle income', 'n_vehicles'],
#             df.loc[df['income_index'] == 'Upper middle income', 'mortality_rate_2016'], c='green')
# plt.plot(df.loc[df['income_index'] == 'Upper middle income', 'n_vehicles'],
#          upper_middle_income_intercept +
#          upper_middle_income_slope * df.loc[df['income_index'] == 'Upper middle income', 'n_vehicles'], color='green')
# plt.xlabel('Number of vehicles')
# plt.ylabel('Deaths per 100,000'
#            '\n'
#            ' population in 2016')
# plt.title(f"Upper middle income, p = {np.round(upper_middle_income_p_value, 2)}")
# plt.subplot(2, 2, 4)
# plt.scatter(df.loc[df['income_index'] == 'High income', 'n_vehicles'],
#             df.loc[df['income_index'] == 'High income', 'mortality_rate_2016'], c='yellow')
# plt.plot(df.loc[df['income_index'] == 'High income', 'n_vehicles'],
#          high_income_intercept + high_income_slope * df.loc[df['income_index'] == 'High income', 'n_vehicles'],
#          color='yellow')
# plt.title(f"High income, p = {np.round(high_income_p_value, 2)}")
# plt.xlabel('Number of vehicles')
# plt.ylabel('Deaths per 100,000'
#            '\n'
#            ' population in 2016')
# plt.tight_layout()
# plt.savefig('outputs/Demographics_of_RTI/Insignificant_relationship_between_n_vehicles_deaths.png')
# plt.clf()
