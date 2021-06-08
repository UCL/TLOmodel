import ast
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo.analysis.utils import parse_log_file
from tlo.methods.rti import RTI


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
    return [zero_to_five, six_to_ten, eleven_to_fifteen, sixteen_to_twenty, twenty1_to_twenty5, twenty6_to_thirty,
            thirty1_to_thirty5, thirty6_to_forty, forty1_to_forty5, forty6_to_fifty, fifty1_to_fifty5,
            fifty6_to_sixty,
            sixty1_to_sixty5, sixty6_to_seventy, seventy1_to_seventy5, seventy6_to_eighty, eighty1_to_eighty5,
            eighty6_to_ninety, ninety_plus]


def create_rti_data(logfile):
    # create a series of lists used to store data from the log files
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
    # number of open fracture appointments used
    per_sim_open_frac = []
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
    # injury severity of rural vs urban injuries
    per_sim_rural_severe = []
    per_sim_urban_severe = []
    # proportion of lower extremity fractures that are open
    per_sim_average_percentage_lx_open = []

    parsed_log = parse_log_file(logfile)
    # get
    demog = parsed_log['tlo.methods.rti']['rti_demography']

    # Get the total age demographics of those with RTI in the sim
    this_sim_ages = demog['age'].tolist()
    # Get the male and female age demographics of those with RTI in the sim
    this_sim_male_ages = demog['male_age'].tolist()
    this_sim_female_ages = demog['female_age'].tolist()
    # Store the ages of those involved in RTI in this sim
    for elem in this_sim_ages:
        for item in elem:
            sim_age_range.append(item)
    # Store the ages of the males involved in RTI in this sim
    for elem in this_sim_male_ages:
        for item in elem:
            sim_male_age_range.append(item)
    # Store the ages of the females involved in RTI in this sim
    for elem in this_sim_female_ages:
        for item in elem:
            sim_female_age_range.append(item)
    # Get the number of males and females in a RTI this sim
    males = sum(demog['males_in_rti'])
    females = sum(demog['females_in_rti'])
    # total number of people in RTI in this sim
    total = males + females
    # Get the percentage of RTI crashes attributable to RTI
    percent_attributable_to_alcohol = np.mean(demog['percent_related_to_alcohol'].tolist())
    # Number permanently disabled
    number_perm_disabled = parsed_log['tlo.methods.rti']['summary_1m']['number permanently disabled'].iloc[-1]
    # Deaths in different contexts
    number_of_deaths_pre_hospital = parsed_log['tlo.methods.rti']['summary_1m']['number immediate deaths'].sum()
    number_of_deaths_in_hospital = parsed_log['tlo.methods.rti']['summary_1m']['number deaths post med'].sum()
    number_of_deaths_no_med = parsed_log['tlo.methods.rti']['summary_1m']['number deaths without med'].sum()
    number_of_deaths_unavailable_med = parsed_log['tlo.methods.rti']['summary_1m']['number deaths unavailable med'] \
        .sum()
    # Store the number of prehospital deaths in 2010
    # Create and extra column in log_df['tlo.methods.rti']['summary_1m'] which stores the year information
    parsed_log['tlo.methods.rti']['summary_1m']['year'] = parsed_log['tlo.methods.rti']['summary_1m'][
        'date'].dt.year
    # group log_df['tlo.methods.rti']['summary_1m'] by simulated year
    grouped_by_year = parsed_log['tlo.methods.rti']['summary_1m'].groupby('year')
    # store the number of prehospital deaths in 2010
    number_of_prehospital_deaths_2010 = grouped_by_year.get_group(2010)['number immediate deaths'].sum()
    # Store the percentage of those who sought health care. I made the logging output 'none_injured' if no one
    # was injured that month, so need to filter out those instances
    percent_sought_healthcare = \
        [i for i in parsed_log['tlo.methods.rti']['summary_1m']['percent sought healthcare'].tolist() if i !=
         'none_injured']

    # Store the percentage of patients admitted to ICU or HDU
    percent_admitted_to_icu_or_hdu = \
        [i for i in parsed_log['tlo.methods.rti']['summary_1m']['percent admitted to ICU or HDU'].tolist() if i !=
         'none_injured']
    percent_admitted_to_icu_or_hdu = np.mean(percent_admitted_to_icu_or_hdu)
    # Create a dataframe which handles the information on ICU patients to create a picture of which injuries the
    # model is predicting will need a stay in ICU
    try:
        icu_df = parsed_log['tlo.methods.rti']['ICU_patients']
        # Drop the date of the logging
        icu_df = icu_df.drop('date', axis=1)
        # Find all the fracture injuries in ICU patients
        frac_codes = ['112', '113', '211', '212', '412', '414', '612', '712', '712a', '712b', '712c',
                      '811', '812', '813', '813a', '813b', '813c', '813bo', '813co', '813do', '813eo']
        idx, frac_counts = RTI.rti_find_and_count_injuries(icu_df, frac_codes)
        # Find the percentage of ICU patients with fractures
        perc_frac = (len(idx) / len(icu_df)) * 100
        # Find all the dislocation injuries in ICU patients
        dislocationcodes = ['322', '323', '722', '822', '822a', '822b']
        idx, dis_counts = RTI.rti_find_and_count_injuries(icu_df, dislocationcodes)
        # Find the percentage of ICU patients with dislocations
        perc_dis = (len(idx) / len(icu_df)) * 100
        # Find all the traumatic brain injuries in ICU patients
        tbi_codes = ['133', '133a', '133b', '133c', '133d', '134', '134a', '134b', '135']
        idx, tbi_counts = RTI.rti_find_and_count_injuries(icu_df, tbi_codes)
        # Find the percentage of ICU patients with TBI
        perc_tbi = (len(idx) / len(icu_df)) * 100
        # Find all the ICU patients with soft tissue injuries
        softtissueinjcodes = ['241', '342', '343', '441', '442', '443']
        idx, soft_counts = RTI.rti_find_and_count_injuries(icu_df, softtissueinjcodes)
        # Find the percentage of ICU patients with soft tissue injury
        perc_soft = (len(idx) / len(icu_df)) * 100
        # Find all the ICU patients with internal organ injuries
        organinjurycodes = ['453', '453a', '453b', '552', '553', '554']
        idx, int_o_counts = RTI.rti_find_and_count_injuries(icu_df, organinjurycodes)
        # Find the percentage of ICU patients with internal organ injury
        perc_int_o = (len(idx) / len(icu_df)) * 100
        # Find all the ICU patients with internal bleeding
        internalbleedingcodes = ['361', '363', '461', '463']
        idx, int_b_counts = RTI.rti_find_and_count_injuries(icu_df, internalbleedingcodes)
        # Find the percentage of ICU patients with internal bleeding
        perc_int_b = (len(idx) / len(icu_df)) * 100
        # Find all the ICU patients with spinal cord injuries
        spinalcordinjurycodes = ['673', '673a', '673b', '674', '674a', '674b', '675', '675a', '675b', '676']
        idx, sci_counts = RTI.rti_find_and_count_injuries(icu_df, spinalcordinjurycodes)
        # Find the percentage of ICU patients with spinal cord injuries
        perc_sci = (len(idx) / len(icu_df)) * 100
        # Find all the ICU patients with amputations
        amputationcodes = ['782', '782a', '782b', '783', '882', '883', '884']
        idx, amp_counts = RTI.rti_find_and_count_injuries(icu_df, amputationcodes)
        # Find the percentage of ICU patients with amputations
        perc_amp = (len(idx) / len(icu_df)) * 100
        # Find all the ICU patients with eye injuries
        eyecodes = ['291']
        idx, eyecounts = RTI.rti_find_and_count_injuries(icu_df, eyecodes)
        # Find the percentage of ICU patients with eye injuries
        perc_eye = (len(idx) / len(icu_df)) * 100
        # Find all the ICU patients with laterations
        externallacerationcodes = ['1101', '2101', '3101', '4101', '5101', '7101', '8101']
        idx, externallacerationcounts = RTI.rti_find_and_count_injuries(icu_df,
                                                                        externallacerationcodes)
        # Find the percentage of ICU patients with lacerations
        perc_lac = (len(idx) / len(icu_df)) * 100
        # Find all the  ICU patients with burns
        burncodes = ['1114', '2114', '3113', '4113', '5113', '7113', '8113']
        idx, burncounts = RTI.rti_find_and_count_injuries(icu_df, burncodes)
        # Find the percentage of ICU patients with burns
        perc_burn = (len(idx) / len(icu_df)) * 100
        # check if anyone was admitted to ICU in this sim
        if len(icu_df) > 0:
            # Store injury information
            ICU_frac = perc_frac
            ICU_dis = perc_dis
            ICU_tbi = perc_tbi
            ICU_soft = perc_soft
            ICU_int_o = perc_int_o
            ICU_int_b = perc_int_b
            ICU_sci = perc_sci
            ICU_amp = perc_amp
            ICU_eye = perc_eye
            ICU_lac = perc_lac
            ICU_burn = perc_burn
    except KeyError:
        # Store injury information
        ICU_frac = 0
        ICU_dis = 0
        ICU_tbi = 0
        ICU_soft = 0
        ICU_int_o = 0
        ICU_int_b = 0
        ICU_sci = 0
        ICU_amp = 0
        ICU_eye = 0
        ICU_lac = 0
        ICU_burn = 0
    # Store the percentage of people who died after seeking healthcare in this sim
    percent_died_after_med = parsed_log['tlo.methods.rti']['summary_1m']['number deaths post med'].sum() / \
                             parsed_log['tlo.methods.rti']['model_progression']['total_sought_medical_care'].iloc[
                                 -1]
    # Store the incidence of RTI per 100,000 person years in this sim
    incidences_of_rti = parsed_log['tlo.methods.rti']['summary_1m']['incidence of rti per 100,000'].tolist()
    # Store the incidence of death due to RTI per 100,000 person years and the sub categories in this sim
    incidences_of_death = parsed_log['tlo.methods.rti']['summary_1m']['incidence of rti death per 100,000'].tolist()
    incidences_of_death_pre_hospital = \
        parsed_log['tlo.methods.rti']['summary_1m']['incidence of prehospital death per 100,000'].tolist()

    incidences_of_death_post_med = \
        parsed_log['tlo.methods.rti']['summary_1m']['incidence of death post med per 100,000'].tolist()

    incidences_of_death_no_med = \
        parsed_log['tlo.methods.rti']['summary_1m']['incidence of death without med per 100,000'].tolist()

    incidences_of_death_unavailable_med = \
        parsed_log['tlo.methods.rti']['summary_1m'][
            'incidence of death due to unavailable med per 100,000'].tolist()
    # Store incidences of death average per year in this sim
    parsed_log['tlo.methods.rti']['summary_1m']['year'] = parsed_log['tlo.methods.rti']['summary_1m'][
        'date'].dt.year
    incidences_of_death_yearly_average = \
        parsed_log['tlo.methods.rti']['summary_1m'].groupby('year').mean()['incidence of rti death per 100,000'] \
            .tolist()
    # Store the incidence of rtis average per year in this sim
    incidences_of_rti_yearly_average = \
        parsed_log['tlo.methods.rti']['summary_1m'].groupby('year').mean()['incidence of rti per 100,000'].tolist()
    # Store the incidence of rtis in children per year in this sim
    incidences_of_rti_in_children = \
        parsed_log['tlo.methods.rti']['summary_1m']['incidence of rti per 100,000 in children'].tolist()
    # Store the incidence of injuries per 100,000 in this sim
    incidences_of_injuries = parsed_log['tlo.methods.rti']['summary_1m']['injury incidence per 100,000'].tolist()
    # Get information on the deaths that occurred in the sim
    deaths_df = parsed_log['tlo.methods.demography']['death']
    # Create list of RTI specific deaths
    rti_death_causes = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death',
                        'RTI_death_shock']
    # Filter the deaths information to only show RTI related deaths
    rti_deaths = len(deaths_df.loc[deaths_df['cause'].isin(rti_death_causes)])
    # Get the number of deaths in 2010
    first_year_deaths = deaths_df.loc[deaths_df['date'] < pd.datetime(2011, 1, 1)]
    first_year_rti_deaths = len(first_year_deaths.loc[first_year_deaths['cause'].isin(rti_death_causes)])
    # Store the number of deaths in 2010 in this sim
    deaths_2010 = first_year_rti_deaths
    # Create information on the percentage of deaths caused by road traffic injuries, use try statement to stop
    # ZeroDivisionError from occuring when no one died due to RTI in this sim
    try:
        # Get the breakdown of road traffic injuries deaths by context by percentage
        ps_of_imm_death = len(deaths_df.loc[deaths_df['cause'] == 'RTI_imm_death']) / rti_deaths
        ps_of_death_post_med = len(deaths_df[deaths_df['cause'] == 'RTI_death_with_med']) / rti_deaths
        ps_of_death_without_med = len(deaths_df[deaths_df['cause'] == 'RTI_death_without_med']) / rti_deaths
        ps_of_death_unavailable_med = len(deaths_df[deaths_df['cause'] == 'RTI_unavailable_med']) / rti_deaths
        ps_of_death_shock = len(deaths_df[deaths_df['cause'] == 'RTI_death_shock']) / rti_deaths
    except ZeroDivisionError:
        ps_of_imm_death = 0
        ps_of_death_post_med = 0
        ps_of_death_without_med = 0
        ps_of_death_unavailable_med = 0
        ps_of_death_shock = 0
    # Get a rough estimate for the percentage road traffic injury deaths for those involved in RTI
    # Get the number of people with RTIs
    number_of_crashes = sum(parsed_log['tlo.methods.rti']['summary_1m']['number involved in a rti'])
    # Store the number of RTI deaths divided by the number of RTIs
    percent_of_fatal_crashes = rti_deaths / number_of_crashes
    # Get qualitative description of RTI injuries, stored in Injury_information
    injury_info = parsed_log['tlo.methods.rti']['Injury_information']
    # Get information on injury severity
    mild_inj = [1 for sublist in injury_info['Per_person_severity_category'].tolist() for item in sublist if
                'mild' in item]
    severe_inj = [1 for sublist in injury_info['Per_person_severity_category'].tolist() for item in
                  sublist if 'severe' in item]
    # Store the percentage of injuries that are mild
    perc_mild = sum(mild_inj) / (sum(mild_inj) + sum(severe_inj))
    # Store the percentage of injuries that are severe
    perc_severe = (sum(severe_inj) / (sum(mild_inj) + sum(severe_inj)))
    # Get information on the distribution of ISS scores in the simulation
    severity_distibution = injury_info['Per_person_injury_severity'].tolist()
    iss_scores = [score for score_list in severity_distibution for score in score_list]
    # Get information on the number of injuries each person was given
    ninj_list = injury_info['Number_of_injuries'].tolist()
    # Count the number of people with i injuries for i in ...
    ninj_list_sorted = [ninj_list.count(i) for i in [1, 2, 3, 4, 5, 6, 7, 8]]
    # Create a dataframe with the date of injuries and the total number of injuries given out in this sim
    ninj_data = {'date': injury_info['date'],
                 'ninj': sum(ninj_list)}
    ninj_df = pd.DataFrame(data=ninj_data)
    # Log the total number of injuries that occured this sim
    number_of_injuries_per_sim = ninj_df['ninj'].sum()
    # Create a column showing which year each log happened in
    ninj_df['year'] = pd.DatetimeIndex(ninj_df['date']).year
    # Store the number of injuries that occured in 2010
    injuries_in_2010 = ninj_df.loc[ninj_df['year'] == 2010]['ninj'].sum()
    # Store the number of injuries that occurred each year
    injuries_per_year = ninj_df.groupby('year').sum()['ninj'].tolist()
    # Store the per injury fatality ratio
    diedfromrticond = parsed_log['tlo.methods.demography']['death']['cause'].isin(rti_death_causes)
    # Following calculation in simpler terms is the number of RTI deaths divided by the total number of RTIs
    per_injury_fatal = \
        len(parsed_log['tlo.methods.demography']['death'].loc[diedfromrticond]) / \
        np.multiply(ninj_list_sorted, [1, 2, 3, 4, 5, 6, 7, 8]).sum()
    # Get information on where these injuries occured on each person
    injury_loc_list = injury_info['Location_of_injuries'].tolist()
    # Flatted the injury location informaiton
    injury_loc_list = [int(item) for sublist in injury_loc_list for item in sublist]
    # Create empty list to store the information
    binned_loc_dist = []
    # Iterate over the injury locations and store the number of times each injury location appears
    for loc in [1, 2, 3, 4, 5, 6, 7, 8]:
        binned_loc_dist.append(injury_loc_list.count(loc))
    # Store the injury location data in this sim
    inj_loc_data = binned_loc_dist
    # Get information on the injury category distribution this run
    inj_cat_list = injury_info['Injury_category'].tolist()
    # Flatten the injury category list
    inj_cat_list = [int(item) for sublist in inj_cat_list for item in sublist]
    # Create empty list to store the information
    binned_cat_dist = []
    # Iterate over the injury categories and store the number of times each injury category appears
    for cat in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
        binned_cat_dist.append(inj_cat_list.count(cat))
    # Store the injury category information
    inj_cat_data = binned_cat_dist
    # Get information on the total incidence of injuries and the breakdown of injury by type
    injury_category_incidence = parsed_log['tlo.methods.rti']['Inj_category_incidence']
    inc_amputations = injury_category_incidence['inc_amputations'].tolist()
    inc_burns = injury_category_incidence['inc_burns'].tolist()
    inc_fractures = injury_category_incidence['inc_fractures'].tolist()
    inc_tbi = injury_category_incidence['inc_tbi'].tolist()
    inc_sci = injury_category_incidence['inc_sci'].tolist()
    inc_minor = injury_category_incidence['inc_minor'].tolist()
    inc_other = injury_category_incidence['inc_other'].tolist()
    tot_inc_injuries = injury_category_incidence['tot_inc_injuries'].tolist()
    # Get the inpatient days usage. I take the overall inpatient day usage from all the simulations and the per-sim
    # inpatient day info
    # Create empty list to store inpatient day information in
    this_sim_inpatient_days = []
    # Create a dataframe containing all instances of the RTI_MedicalIntervention event

    inpatient_day_df = parsed_log['tlo.methods.healthsystem']['HSI_Event'].loc[
        parsed_log['tlo.methods.healthsystem']['HSI_Event']['TREATMENT_ID'] == 'RTI_MedicalIntervention']
    # iterate over the people in inpatient_day_df
    for person in inpatient_day_df.index:
        # Get the number of inpatient days per person, if there is a key error when trying to access inpatient days it
        # means that this patient didn't require any so append (0)
        try:
            this_sim_inpatient_days.append(inpatient_day_df.loc[person, 'Number_By_Appt_Type_Code']['InpatientDays'])
        except KeyError:
            this_sim_inpatient_days.append(0)

    # get the consumables used in each simulation
    consumables_list = parsed_log['tlo.methods.healthsystem']['Consumables']['Item_Available'].tolist()
    # Create empty list to store the consumables used in the simulation
    consumables_list_to_dict = []
    for string in consumables_list:
        consumables_list_to_dict.append(ast.literal_eval(string))
    # Begin counting the number of consumables used in the simulation starting at 0
    number_of_consumables_in_sim = 0
    for dictionary in consumables_list_to_dict:
        number_of_consumables_in_sim += sum(dictionary.values())
    health_system_time_usage = np.mean(parsed_log['tlo.methods.healthsystem']['Capacity']['Frac_Time_Used_Overall'])
    # get the number of rti-hsi interaction events by type
    # get the dataframe of the health system events
    appointments = parsed_log['tlo.methods.healthsystem']['HSI_Event']
    # isolate appointments than ran
    appointments = appointments.loc[appointments['did_run']]
    # isolate appointments by type
    per_sim_burn_treated = len(appointments.loc[appointments['TREATMENT_ID'] == 'RTI_Burn_Management'])
    per_sim_frac_cast = len(appointments.loc[appointments['TREATMENT_ID'] == 'RTI_Fracture_Cast'])
    per_sim_laceration = len(appointments.loc[appointments['TREATMENT_ID'] == 'RTI_Suture'])
    per_sim_major_surg = len(appointments.loc[appointments['TREATMENT_ID'] == 'RTI_Major_Surgeries'])
    per_sim_minor_surg = len(appointments.loc[appointments['TREATMENT_ID'] == 'RTI_Minor_Surgeries'])
    per_sim_tetanus = len(appointments.loc[appointments['TREATMENT_ID'] == 'RTI_Tetanus_Vaccine'])
    per_sim_pain_med = len(appointments.loc[appointments['TREATMENT_ID'] == 'RTI_Acute_Pain_Management'])
    per_sim_open_frac = len(appointments.loc[appointments['TREATMENT_ID'] == 'RTI_Open_Fracture_Treatment'])
    per_sim_shock = len(appointments.loc[appointments['TREATMENT_ID'] == 'RTI_Shock_Treatment'])
    # store the relating to the percentage of injury severity in both rural and urban settings
    per_sim_rural_severe = parsed_log['tlo.methods.rti']['injury_severity']['Percent_severe_rural'].tolist()
    per_sim_urban_severe = parsed_log['tlo.methods.rti']['injury_severity']['Percent_severe_urban'].tolist()
    # Store the proportion of lower extremity fractures that are open in this sim
    proportions_of_open_lx_fractures_in_sim = \
        [i for i in parsed_log['tlo.methods.rti']['Open_fracture_information']['Proportion_lx_fracture_open'].values
         if i != 'no_lx_fractures']
    per_sim_average_percentage_lx_open = np.mean(proportions_of_open_lx_fractures_in_sim)

    # Get simulaion data
    pop_size = parsed_log['tlo.methods.demography']['population']['total'].iloc[0]
    sim_start_date = parsed_log['tlo.methods.demography']['population']['date'].iloc[0]
    sim_end_date = parsed_log['tlo.methods.healthburden']['dalys']['date'].iloc[0]
    years_run = sim_end_date.year - sim_start_date.year
    time = parsed_log['tlo.methods.rti']['summary_1m']['date'].tolist()
    results_dict = {'age_range': sim_age_range,
                    'male_age_range': sim_male_age_range,
                    'female_age_range': sim_female_age_range,
                    'females_in_rti': females,
                    'males_in_rti': males,
                    'total_in_rti': total,
                    'percent_attributable_to_alcohol': percent_attributable_to_alcohol,
                    'number_perm_disabled': number_perm_disabled,
                    'number_of_deaths_pre_hospital': number_of_deaths_pre_hospital,
                    'number_of_deaths_in_hospital': number_of_deaths_in_hospital,
                    'number_of_deaths_no_med': number_of_deaths_no_med,
                    'number_of_deaths_unavailable_med': number_of_deaths_unavailable_med,
                    'number_of_prehospital_deaths_2010': number_of_prehospital_deaths_2010,
                    'percent_sought_healthcare': percent_sought_healthcare,
                    'percent_admitted_to_icu_or_hdu': percent_admitted_to_icu_or_hdu,
                    'ICU_frac': ICU_frac,
                    'ICU_dis': ICU_dis,
                    'ICU_tbi': ICU_tbi,
                    'ICU_soft': ICU_soft,
                    'ICU_int_o': ICU_int_o,
                    'ICU_int_b': ICU_int_b,
                    'ICU_sci': ICU_sci,
                    'ICU_amp': ICU_amp,
                    'ICU_eye': ICU_eye,
                    'ICU_lac': ICU_lac,
                    'ICU_burn': ICU_burn,
                    'percent_died_after_med': percent_died_after_med,
                    'incidences_of_rti': incidences_of_rti,
                    'incidences_of_death': incidences_of_death,
                    'incidences_of_death_pre_hospital': incidences_of_death_pre_hospital,
                    'incidences_of_death_post_med': incidences_of_death_post_med,
                    'incidences_of_death_no_med': incidences_of_death_no_med,
                    'incidences_of_death_unavailable_med': incidences_of_death_unavailable_med,
                    'incidences_of_death_yearly_average': incidences_of_death_yearly_average,
                    'incidences_of_rti_yearly_average': incidences_of_rti_yearly_average,
                    'incidences_of_rti_in_children': incidences_of_rti_in_children,
                    'incidences_of_injuries': incidences_of_injuries,
                    'deaths_2010': deaths_2010,
                    'ps_of_imm_death': ps_of_imm_death,
                    'ps_of_death_post_med': ps_of_death_post_med,
                    'ps_of_death_without_med': ps_of_death_without_med,
                    'ps_of_death_unavailable_med': ps_of_death_unavailable_med,
                    'ps_of_death_shock': ps_of_death_shock,
                    'percent_of_fatal_crashes': percent_of_fatal_crashes,
                    'perc_mild': perc_mild,
                    'perc_severe': perc_severe,
                    'iss_scores': iss_scores,
                    'ninj_list_sorted': ninj_list_sorted,
                    'number_of_injuries_per_sim': number_of_injuries_per_sim,
                    'injuries_in_2010': injuries_in_2010,
                    'injuries_per_year': injuries_per_year,
                    'per_injury_fatal': per_injury_fatal,
                    'inj_loc_data': inj_loc_data,
                    'inj_cat_list': inj_cat_list,
                    'inj_cat_data': inj_cat_data,
                    'inc_amputations': inc_amputations,
                    'inc_burns': inc_burns,
                    'inc_fractures': inc_fractures,
                    'inc_tbi': inc_tbi,
                    'inc_sci': inc_sci,
                    'inc_minor': inc_minor,
                    'inc_other': inc_other,
                    'tot_inc_injuries': tot_inc_injuries,
                    'this_sim_inpatient_days': this_sim_inpatient_days,
                    'number_of_consumables_in_sim': number_of_consumables_in_sim,
                    'health_system_time_usage': health_system_time_usage,
                    'per_sim_burn_treated': per_sim_burn_treated,
                    'per_sim_frac_cast': per_sim_frac_cast,
                    'per_sim_laceration': per_sim_laceration,
                    'per_sim_major_surg': per_sim_major_surg,
                    'per_sim_minor_surg': per_sim_minor_surg,
                    'per_sim_tetanus': per_sim_tetanus,
                    'per_sim_pain_med': per_sim_pain_med,
                    'per_sim_open_frac': per_sim_open_frac,
                    'per_sim_shock': per_sim_shock,
                    'per_sim_rural_severe': per_sim_rural_severe,
                    'per_sim_urban_severe': per_sim_urban_severe,
                    'per_sim_average_percentage_lx_open': per_sim_average_percentage_lx_open,
                    'years_run': years_run,
                    'pop_size': pop_size,
                    'time': time
                    }
    return results_dict

# create a function to plot general simulation outputs
def create_rti_graphs(logfile_directory, save_directory, filename_description):
    # determine whether graphs are being created for a single logfile or averaging over multiple logs
    if len(os.listdir(logfile_directory)) > 1:
        nsim = len(os.listdir(logfile_directory))
        results_all_runs = pd.DataFrame()
        for log in os.listdir(logfile_directory):
            data = create_rti_data(logfile_directory + "/" + log)
            results_all_runs = results_all_runs.append(data, ignore_index=True)

    else:
        nsim = 1
        results_all_runs = pd.DataFrame()
        data = create_rti_data(logfile_directory)
        results_all_runs = results_all_runs.append(data, ignore_index=True)

    # Create graphs and save them in the specified save file path
    yearsrun = int(results_all_runs['years_run'].mean())
    pop_size = int(results_all_runs['pop_size'].mean())
    icu_characteristics = results_all_runs[
        ['ICU_amp', 'ICU_burn', 'ICU_dis', 'ICU_eye', 'ICU_frac', 'ICU_int_b',
         'ICU_int_o', 'ICU_lac', 'ICU_sci', 'ICU_soft', 'ICU_tbi']]
    plt.bar(np.arange(len(icu_characteristics.mean().index)), icu_characteristics.mean().values,
            color='lightsalmon')
    plt.xticks(np.arange(len(icu_characteristics.mean().index)), icu_characteristics.mean().index, rotation=90)
    plt.ylabel('Percent injury type in ICU patients')
    plt.title(f"Injuries in ICU patients by type"
              f"\n"
              f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")

    plt.savefig(save_directory + "/" + filename_description + "_" +
                f"ICU_injury_categories_pop_{pop_size}_years_{yearsrun}_runs_{nsim}.png",
                bbox_inches='tight')
    plt.clf()

    # ======================== Plot average age of those in RTI compared to other sources =================
    # Calculate the mean age and standard deviation in age of those involved in RTIs
    mean_age_per_run = [np.mean(age_list) for age_list in results_all_runs['age_range'].tolist()]
    std_age_per_run = [np.std(age_list) for age_list in results_all_runs['age_range'].tolist()]
    results_all_runs['mean_age'] = mean_age_per_run
    results_all_runs['std_age'] = std_age_per_run
    mean_age = results_all_runs['mean_age'].mean()
    std_age = results_all_runs['std_age'].mean()
    # Use data from: https://doi.org/10.1007/s00268-020-05853-z
    police_ave_age = 32
    police_std_age = 12
    queen_elizabeth_central_hospital_mean = 32
    queen_elizabeth_central_hospital_std = 12
    # plot the model data compared to the police and QECH data
    plt.bar(np.arange(3), [mean_age, police_ave_age, queen_elizabeth_central_hospital_mean],
            yerr=[std_age, police_std_age, queen_elizabeth_central_hospital_std], color='wheat')
    plt.xticks(np.arange(3), ['Model', 'Police data', 'Queen Elizabeth Central \n Hospital data'])
    plt.ylabel('Age')
    plt.title(f"Average age with RTIs compared to police and hospital data"
              f"\n"
              f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")

    plt.savefig(
        save_directory + "/" + filename_description + "_" +
        f"Age_average_comp_to_pol_hos_pop_{pop_size}_years_{yearsrun}_runs_{nsim}.png",
        bbox_inches='tight')
    plt.clf()

    # ===================== Plot age demographics of those in RTI by percentage ================================
    # Bin the ages of those involved in road traffic ages
    sim_age_range = [age for age_list in results_all_runs['age_range'].tolist() for age in age_list]
    height_for_bar_plot = age_breakdown(sim_age_range)
    # Calculate the percentage of RTIs occuring in each age group
    height_for_bar_plot = np.divide(height_for_bar_plot, sum(height_for_bar_plot))
    labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40',
              '41-45', '46-50', '51-55', '56-60', '61-65', '66-70', '71-75', '76-80',
              '81-85', '86-90', '90+']
    # Plot the RTI age distribution in a bar chart
    plt.bar(np.arange(len(height_for_bar_plot)), height_for_bar_plot, color='lightsteelblue')
    plt.xticks(np.arange(len(height_for_bar_plot)), labels, rotation=45)
    plt.ylabel('Percentage')
    plt.xlabel('Age')
    plt.title(f"Age demographics of those with RTIs"
              f"\n"
              f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")

    plt.savefig(
        save_directory + "/" + filename_description + "_" +
        f"Age_demographics_percentage_pop_{pop_size}_years_{yearsrun}_runs_{nsim}.png",
        bbox_inches='tight')
    plt.clf()

    # =================== Plot age demographics of those in rti by number =================================
    # Calculate the number of RTI's that occur in each age group
    height_for_bar_plot = age_breakdown(sim_age_range)
    labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40',
              '41-45', '46-50', '51-55', '56-60', '61-65', '66-70', '71-75', '76-80',
              '81-85', '86-90', '90+']
    # plot the data in a bar chart
    plt.bar(np.arange(len(height_for_bar_plot)), height_for_bar_plot, color='lightsteelblue')
    plt.xticks(np.arange(len(height_for_bar_plot)), labels, rotation=45)
    plt.ylabel('Number')
    plt.xlabel('Age')
    plt.title(f"Age demographics of those with RTIs"
              f"\n"
              f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
    plt.savefig(save_directory + "/" + filename_description + "_" +
                f"Age_demographics_number_pop_{pop_size}_years_{yearsrun}_runs_{nsim}.png",
                bbox_inches='tight')
    plt.clf()
    # Injury information graphs
    # ================ Plot the distribution of the ISS scores ====================================
    # Flatten the list of ISS scores predicted in the simulation
    iss_scores = results_all_runs['iss_scores'].to_list()
    flattened_scores = [score for sublist in iss_scores for score in sublist]
    # use np.unique to count the various scores and create labels for the graph
    scores, counts = np.unique(flattened_scores, return_counts=True)
    # calculate the distribution of ISS scores
    distribution_of_scores = counts / sum(counts)
    # Plot data in a bar chart
    plt.bar(scores, distribution_of_scores, width=0.8, color='lightsteelblue')
    plt.xlabel('ISS scores')
    plt.ylabel('Percentage')
    plt.title(f"Average road traffic injury ISS score distribution"
              f"\n"
              f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
    plt.xlim([0, 75])
    plt.savefig(save_directory + "/" + filename_description + "_" +
                f"Average_ISS_scores_pop_{pop_size}_years_{yearsrun}_runs_{nsim}.png",
                bbox_inches='tight')
    plt.clf()
    # plot the percentage of injuries that are mild
    injury_severity_percentages = results_all_runs[['perc_mild', 'perc_severe']]
    n = np.arange(2)
    # data is the rounded percentage of injuries that were mild and the percentage that were severe
    data = [np.round(injury_severity_percentages['perc_mild'].mean(), 3),
            np.round(injury_severity_percentages['perc_severe'].mean(), 3)]
    # plot data in a bar chart
    plt.bar(n, data,
            yerr=[np.std(injury_severity_percentages['perc_mild']),
                  np.std(injury_severity_percentages['perc_severe'])],
            color='lightsteelblue')
    # Annotate the graph with the numerical values of the percentage
    for i in range(len(data)):
        plt.annotate(str(data[i]), xy=(n[i], data[i]), ha='center', va='bottom')
    plt.xticks(np.arange(2), labels=['Mild injuries', 'Severe injuries'])
    plt.title(f"Average road traffic injury severity distribution"
              f"\n"
              f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")

    plt.savefig(
        save_directory + "/" + filename_description + "_" +
        f"Percentage_mild_severe_injuries_pop_{pop_size}_years_{yearsrun}_runs_{nsim}.png",
        bbox_inches='tight')
    plt.clf()
    # ================= Plot the percentage of mild and severe injuries to other sources =====================
    # Police data from Schlottmann et al doi: 10.4314/mmj.v29i4.4
    police_data = [0.64, 0.36]
    # plot the model data (as it was above) and compare it do the police data
    plt.bar(np.arange(2), data, color='lightsteelblue', width=0.4, label='Model data')
    plt.bar(np.arange(2) + 0.4, police_data, color='lightsalmon', width=0.4, label='Police data')
    plt.xticks(np.arange(2) + 0.2, ['Mild', 'Severe'])
    plt.ylabel('Percent')
    plt.legend()
    plt.title(f"Average road traffic injury severity distribution compared to police data"
              f"\n"
              f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")

    plt.savefig(save_directory + "/" + filename_description + "_" +
                f"Percentage_mild_severe_injuries_comparison_pop_{pop_size}_years_{yearsrun}"
                f"_runs_{nsim}.png", bbox_inches='tight')
    plt.clf()

    # ======================= Plot the distribution of the number of injured body regions =========================
    # Calculate the average number of injured body regions
    number_of_injured_body_locations = results_all_runs['ninj_list_sorted']
    average_number_of_body_regions_injured = [float(sum(col)) / len(col) for col in
                                              zip(*number_of_injured_body_locations)]
    # Calculate the average distribution of number of injured body regions
    data = np.divide(average_number_of_body_regions_injured, sum(average_number_of_body_regions_injured))
    plt.bar(np.arange(8), data, color='lightsteelblue')
    plt.xticks(np.arange(8), ['1', '2', '3', '4', '5', '6', '7', '8'])
    plt.xlabel('Number of injured AIS body regions')
    plt.ylabel('Percentage')
    plt.title(f"Average injured body region distribution"
              f"\n"
              f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")

    plt.savefig(save_directory + "/" + filename_description + "_" +
                f"Average_injured_body_region_distribution_pop_{pop_size}_years_{yearsrun}"
                f"_runs_{nsim}.png", bbox_inches='tight')
    plt.clf()

    # ========================= plot the injury location data =====================================================
    # Calculate the average number of injuries occuring in each body region
    inj_loc_data = results_all_runs['inj_loc_data']
    average_inj_loc = [float(sum(col)) / len(col) for col in zip(*inj_loc_data)]
    # Calculate the distribution of average number of injuries occuring in each body region
    data = np.divide(average_inj_loc, sum(average_inj_loc))
    plt.bar(np.arange(8), data, color='lightsteelblue')
    plt.xticks(np.arange(8), ['Head', 'Face', 'Neck', 'Thorax', 'Abdomen', 'Spine', 'UpperX', 'LowerX'],
               rotation=45)
    plt.xlabel('AIS body regions')
    plt.ylabel('Percentage')
    plt.title(f"Average injury location distribution"
              f"\n"
              f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
    plt.savefig(save_directory + "/" + filename_description + "_" +
                f"Average_injury_location_distribution_pop_{pop_size}_years_"
                f"{yearsrun}_runs_{nsim}.png", bbox_inches='tight')
    plt.clf()
    # ============================== Plot the injury category data =================================================
    # Calculate the average number of injuries which fall into each injury category in the simulation
    inj_cat_data = results_all_runs['inj_cat_data']
    average_inj_cat = [float(sum(col)) / len(col) for col in zip(*inj_cat_data)]
    # Calculate the distribution of average number of injuries which fall into each injury category in the
    # simulation
    data = np.divide(average_inj_cat, sum(average_inj_cat))
    plt.bar(np.arange(len(average_inj_cat)), data, color='lightsteelblue')
    plt.xticks(np.arange(len(average_inj_cat)),
               ['Fracture', 'Dislocation', 'TBI', 'Soft Tissue Inj.', 'Int. Organ Inj.',
                'Int. Bleeding', 'SCI', 'Amputation', 'Eye injury', 'Laceration', 'Burn'],
               rotation=90)
    plt.ylabel('Percentage')
    plt.title(f"Average injury category distribution"
              f"\n"
              f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
    plt.savefig(save_directory + "/" + filename_description + "_" +
                f"Average_injury_category_distribution_pop_{pop_size}_years_{yearsrun}_runs_"
                f"{nsim}.png", bbox_inches='tight')
    plt.clf()

    # =============== Plot the per injury fatality produced by the model and compare to the gbd data ===============
    # Get GBD data
    data = pd.read_csv('resources/ResourceFile_RTI_GBD_Number_And_Incidence_Data.csv')
    # Isolate death date
    gbd_death_data = data.loc[data['measure'] == 'Deaths']
    # Isolate incidence data
    gbd_in_rti_data = data.loc[data['measure'] == 'Incidence']
    # Sort death and rti incidence data by year
    gbd_death_data = gbd_death_data.sort_values(by='year')
    gbd_in_rti_data = gbd_in_rti_data.sort_values(by='year')
    # isolate the number of predicted death and predicted rti incidence in the GBD data
    gbd_death_number = gbd_death_data.loc[gbd_death_data['metric'] == 'Number']
    gbd_rti_number = gbd_in_rti_data.loc[gbd_in_rti_data['metric'] == 'Number']
    # Calculate the GBD's predicted injury fatality ratio
    gbd_percent_fatal_ratio = gbd_death_number['val'].sum() / gbd_rti_number['val'].sum()
    # Calculate the average percent fatal ratio from the model simulations

    model_percent_fatal_ratio = results_all_runs['per_injury_fatal'].mean()
    # Plot these together in a bar chart
    plt.bar(np.arange(2), [model_percent_fatal_ratio, gbd_percent_fatal_ratio],
            color=['lightsteelblue', 'lightsalmon'])
    plt.xticks(np.arange(2), ['Model per injury \n death percentage', 'GBD per injury \n death percentage'])
    plt.ylabel('Percent')
    plt.title(f"Per injury fatality ratio, model compared to GBD"
              f"\n"
              f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
    plt.savefig(save_directory + "/" + filename_description + "_" +
                f"Per_injury_fatality_pop_{pop_size}_years_{yearsrun}_runs_{nsim}.png",
                bbox_inches='tight')
    plt.clf()
    # ========== plot the percentage of lower extremity fractures that are open in the model's simulations =========
    # state the data we mean to plot
    data = [results_all_runs['per_sim_average_percentage_lx_open'].mean(),
            1 - results_all_runs['per_sim_average_percentage_lx_open'].mean()]
    # Plot the data in a pie chart
    plt.pie(data,
            explode=None, labels=['Open lx fracture', "Closed lx fracture"],
            colors=['lightsteelblue', 'lightsalmon'],
            autopct='%1.1f%%')
    plt.title(f"Average percentage of lower extremity fractures that are open"
              f"\n"
              f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
    plt.savefig(save_directory + "/" + filename_description + "_" +
                f"Percent_lx_fracture_open_pop_{pop_size}_years_{yearsrun}_runs_{nsim}.png",
                bbox_inches='tight')
    plt.clf()

    # ================= plot the percentage of those who sought health care ========================================
    # calculate the mean percentage of people with RTI who sought health care in each sim
    per_sim_average_health_seeking = [np.mean(i) for i in results_all_runs['percent_sought_healthcare'].tolist()]
    # Calculate the overall average of people with RTI who seek healthcare for each sim (mean of mean above)
    overall_average_health_seeking_behaviour = np.mean(per_sim_average_health_seeking)
    # plot in a pie chart
    plt.pie([overall_average_health_seeking_behaviour, 1 - overall_average_health_seeking_behaviour],
            explode=None, labels=['Sought care', "Didn't seek care"], colors=['lightsteelblue', 'lightsalmon'],
            autopct='%1.1f%%')
    plt.title(f"Average percentage of those with road traffic injuries who sought health care"
              f"\n"
              f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
    plt.savefig(save_directory + "/" + filename_description + "_" +
                f"Percent_Sought_Healthcare_pop_{pop_size}_years_{yearsrun}_runs_{nsim}.png",
                bbox_inches='tight')
    plt.clf()
    # ========================= Plot the percentage of people admitted to ICU and HDU ============================
    # Calculate the average percentage of RTI patients who are admitted to the ICU/HDU
    per_sim_icu_or_hdu_average = results_all_runs['percent_admitted_to_icu_or_hdu'].tolist()

    # set the comparative values from Kamuzu Central Hospital, see: https://doi.org/10.1007/s00268-020-05853-z
    kch_data = [2.7 + 3.3]
    # Change data format to a percentage
    data = np.multiply([np.mean(per_sim_icu_or_hdu_average)], 100)
    # Plot the model data
    plt.bar(np.arange(1), data, color='lightsteelblue', label='Model', width=0.4)
    # Plot the KCH data
    plt.bar(np.arange(1) + 0.4, kch_data, color='lightsalmon', label='KCH', width=0.4)
    plt.xticks(np.arange(1) + 0.2, ['Percent admitted ICU or HDU'])
    plt.ylabel('Percentage')
    plt.legend()
    plt.title(f"Average percentage admitted to ICU/HDU"
              f"\n"
              f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")

    plt.savefig(save_directory + "/" + filename_description + "_" +
                f"Percent_admitted_icu_hdu_bar_pop_{pop_size}_years_{yearsrun}_runs_{nsim}.png",
                bbox_inches='tight')
    plt.clf()

    # ======================== Plot the percentage of death post med ============================================
    # Calculate the overall percentage of death post medical intervention for RTI
    overall_average_post_med_death = results_all_runs['percent_died_after_med'].mean()
    # Plot this data in a pie chart
    plt.pie([overall_average_post_med_death, 1 - overall_average_post_med_death],
            explode=None, labels=['Fatal', "Non-fatal"], colors=['lightsteelblue', 'lightsalmon'],
            autopct='%1.1f%%')
    plt.title(f"Average percent survival outcome of those with road traffic injuries who sought health care"
              f"\n"
              f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
    plt.savefig(save_directory + "/" + filename_description + "_" +
                f"Percent_Survival_Healthcare_pop_{pop_size}_years_{yearsrun}_runs_{nsim}.png",
                bbox_inches='tight')
    plt.clf()

    # ================== Plot the percent of death post med compared to Kamuzu central ============================
    # Take KCH values from: https://doi.org/10.1016/j.jsurg.2014.09.010
    percent_mortality_kamuzu = (182 + 38) / (3840 + 1227 + 182 + 38)
    # Plot the KCH data as a bar chart
    plt.bar(np.arange(2), [percent_mortality_kamuzu, 1 - percent_mortality_kamuzu], width=0.3,
            color='lightsalmon', label='In-hospital mortality, \nKamuzu central hospital')
    # Plot the model data next to the KCH data
    plt.bar(np.arange(2) + 0.5, [overall_average_post_med_death, 1 - overall_average_post_med_death], width=0.3,
            color='lightsteelblue', label='Model in-hospital mortality')
    plt.xticks(np.arange(2) + 0.25, ['Fatal', 'Non-fatal'])
    plt.legend()
    plt.title(f"In-hospital fatality due to injury percentage \n "
              f"model prediction: {np.round(overall_average_post_med_death, 2)} \n"
              f"Kamuzu central hospital: {np.round(percent_mortality_kamuzu, 2)} \n"
              f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
    plt.savefig(
        save_directory + "/" + filename_description + "_" +
        f"Percent_Survival_Healthcare_compare_Kamuzu_pop_{pop_size}_years_{yearsrun}_runs_"
        f"{nsim}.png", bbox_inches='tight')
    plt.clf()
    # ============================== Plot health outcomes for road traffic injuries ===============================
    # Calculate the total number of people involved in RTI
    total_n_crashes = results_all_runs['total_in_rti'].mean()
    # Calculate the number of deaths in hospital
    total_n_hospital_deaths = results_all_runs['number_of_deaths_in_hospital'].mean()
    # Calculate the number of deaths pre hospital
    total_n_prehospital_deaths = results_all_runs['number_of_deaths_no_med'].mean()
    # Calculate the number of deaths occuring in people not seeking medical care
    total_n_no_hospital_deaths = results_all_runs['number_of_deaths_no_med'].mean()
    # Calculate the number of deaths due to medical care being unavailable
    total_n_unavailable_med_deaths = results_all_runs['number_of_deaths_unavailable_med'].mean()
    # Calculate the number of permanent disabilities due to RTI
    total_n_perm_disability = results_all_runs['number_perm_disabled'].mean()
    # Calculate the total number of people who survived their injuries
    total_survived = total_n_crashes - total_n_hospital_deaths - total_n_prehospital_deaths - \
                     total_n_no_hospital_deaths - total_n_unavailable_med_deaths
    # Plot a pie chart showing the health outcomes of those with RTI
    plt.pie([total_survived, total_n_perm_disability, total_n_prehospital_deaths, total_n_hospital_deaths,
             total_n_no_hospital_deaths, total_n_unavailable_med_deaths], explode=None,
            labels=['Non-fatal', 'Permanently disabled', 'Pre-hospital mortality', 'In-hospital mortality',
                    'No-hospital mortality', 'Unavailable care mortality'],
            colors=['lightsteelblue', 'lightsalmon', 'wheat', 'darkcyan', 'orchid'], autopct='%1.1f%%')
    plt.title(f"Outcomes for road traffic injuries in the model"
              f"\n"
              f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
    plt.savefig(save_directory + "/" + filename_description + "_" +
                f"Outcome_Of_Crashes_pop_{pop_size}_years_{yearsrun}_runs_{nsim}.png", bbox_inches='tight')
    plt.clf()
    # ===================== Plot the gender demographics of those involved in RTIs =================================
    # Calculate the total number of people involved in RTIs

    total_injuries = [i + j for i, j in zip(results_all_runs['males_in_rti'], results_all_runs['females_in_rti'])]
    # Calculate the percentage of all RTIs that occur in males and females
    male_perc = np.divide(results_all_runs['males_in_rti'], total_injuries)
    femal_perc = np.divide(results_all_runs['females_in_rti'], total_injuries)
    n = np.arange(2)
    # Round off the data
    data = [np.round(np.mean(male_perc), 3), np.round(np.mean(femal_perc), 3)]
    # plot the data as a bar chart
    plt.bar(n, data, yerr=[np.std(male_perc), np.std(femal_perc)], color='lightsteelblue')
    # Annotate the graph with the values of th percentages
    for i in range(len(data)):
        plt.annotate(str(data[i]), xy=(n[i], data[i]), ha='center', va='bottom')
    plt.xticks(n, ['Males', 'Females'])
    plt.ylabel('Percentage')
    plt.xlabel('Gender')
    plt.title(f"Gender demographics of those with RTIs"
              f"\n"
              f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
    plt.savefig(save_directory + "/" + filename_description + "_" +
                f"Gender_demographics_pop_{pop_size}_years_{yearsrun}_runs_{nsim}.png",
                bbox_inches='tight')
    plt.clf()

    # Plot the gender demographics of those in RTI as a pie chart, using data as above
    plt.pie(data, explode=None, labels=['Males', 'Females'], colors=['lightsteelblue', 'lightsalmon'],
            autopct='%1.1f%%',
            startangle=90)
    plt.title(f"Gender demographics of those with RTIs"
              f"\n"
              f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
    plt.savefig(save_directory + "/" + filename_description + "_" +
                f"Gender_demographics_pie_pop_{pop_size}_years_{yearsrun}_runs_{nsim}.png",
                bbox_inches='tight')
    plt.clf()

    # ================= Plot the percentage of crashes attributable to alcohol ====================================
    # Get the mean percentage of chrases related to alcohol in each sim
    means_of_sim = results_all_runs['percent_attributable_to_alcohol'].mean()
    # Get the mean percentage of chrashed occuring without alcohol
    means_non_alocohol = 1 - means_of_sim
    # plot the data in a pie chart
    plt.pie([means_of_sim, means_non_alocohol], explode=None, labels=['Alcohol related', 'Non-alcohol related'],
            colors=['lightsteelblue', 'lightsalmon'], autopct='%1.1f%%')
    plt.title(f"Average percentage of RTIs attributable to Alcohol"
              f"\n"
              f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
    plt.savefig(save_directory + "/" + filename_description + "_" +
                f"Alcohol_demographics_pop_{pop_size}_years_{yearsrun}_runs_{nsim}.png",
                bbox_inches='tight')
    plt.clf()
    # ========================= plot the incidence of RTI and death from RTI =================================
    # Calculate the average incidence of RTI at each logging event for the simulations
    average_incidence = [float(sum(col)) / len(col) for col in zip(*results_all_runs['incidences_of_rti'])]
    # Calculate the standard deviation of the incidence of RTIs at each logging event
    std_incidence = [np.std(i) for i in zip(*results_all_runs['incidences_of_rti'])]
    # Calculate the upper and lowere limits for the confidence intervals of RTI incidence
    inc_upper = [inc + (1.96 * std) / nsim for inc, std in zip(average_incidence, std_incidence)]
    inc_lower = [inc - (1.96 * std) / nsim for inc, std in zip(average_incidence, std_incidence)]
    # Calculate the average incidence of death at each logging event for the simulations
    average_deaths = [float(sum(col)) / len(col) for col in zip(*results_all_runs['incidences_of_death'])]
    # Calculate the standard deviation of the incidence of RTI death at each logging event
    std_deaths = [np.std(j) for j in zip(*results_all_runs['incidences_of_death'])]
    # Calculate the upper and lowere limits for the confidence intervals of RTI death
    death_upper = [inc + (1.96 * std) / nsim for inc, std in zip(average_deaths, std_deaths)]
    death_lower = [inc - (1.96 * std) / nsim for inc, std in zip(average_deaths, std_deaths)]
    # Calculate the average incidence of injuries occuring at each logging event
    average_injury_incidence = [float(sum(col)) / len(col) for col in
                                zip(*results_all_runs['incidences_of_injuries'])]
    # calculate the overall average incidence in the simulations
    overall_av_inc_sim = np.mean(average_incidence)
    # Calculate the overall average deaths incidence in the simulations
    overall_av_death_inc_sim = np.mean(average_deaths)
    # Calculate the average injury incidence in the simulations
    overall_av_inc_injuries = np.mean(average_injury_incidence)
    # Get the time stamps of the logging events to use as our x axis intervals
    time = results_all_runs['time'].iloc[-1]
    # plot the average incidence of rtis
    plt.plot(time, average_incidence, color='lightsteelblue', label='Incidence of RTI', zorder=2)
    # Plot the 95% c.i.
    plt.fill_between(time, inc_upper, inc_lower, alpha=0.5, color='lightsteelblue', label='95% C.I., RTI inc.',
                     zorder=1)
    # plot the average incidence of rti deaths
    plt.plot(time, average_deaths, color='lightsalmon', label='Incidence of death '
                                                              '\n'
                                                              'due to RTI', zorder=2)
    # Plot the 95% c.i.
    plt.fill_between(time, death_upper, death_lower, alpha=0.5, color='lightsalmon',
                     label='95% C.I. inc death', zorder=1)
    # plot the average incidence of RTIs in the simulations
    plt.hlines(overall_av_inc_sim, time[0], time[-1], label=f"Average incidence of "
                                                            f"\n"
                                                            f"RTI = {np.round(overall_av_inc_sim, 2)}",
               color='lightsteelblue', linestyles='--')
    # Plot the average incidence of RTI deaths in the simulations
    plt.hlines(overall_av_death_inc_sim, time[0], time[-1], label=f"Average incidence of "
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
    plt.savefig(save_directory + "/" + filename_description + "_" +
                f"Incidence_and_deaths_pop_{pop_size}_years_{yearsrun}_runs_{nsim}.png",
                bbox_inches='tight')
    plt.clf()

    # ================== Plot the incidence predicted by the model compared to the GBD data =======================
    data = pd.read_csv('resources/ResourceFile_RTI_GBD_Number_And_Incidence_Data.csv')
    data = data.loc[data['metric'] == 'Rate']
    data = data.loc[data['measure'] == 'Deaths']
    data = data.loc[data['year'] > 2009]
    gbd_time = ['2010-01-01', '2011-01-01', '2012-01-01', '2013-01-01',
                '2014-01-01', '2015-01-01', '2016-01-01', '2017-01-01',
                '2018-01-01', '2019-01-01']
    yearly_average_deaths = \
        [float(sum(col)) / len(col) for col in zip(*results_all_runs['incidences_of_death_yearly_average'])]
    std_yearly_death_incidence = [np.std(i) for i in zip(*results_all_runs['incidences_of_death_yearly_average'])]
    yearly_death_inc_upper = [inc + (1.96 * std) / nsim for inc, std in zip(yearly_average_deaths,
                                                                            std_yearly_death_incidence)]
    yearly_death_inc_lower = [inc - (1.96 * std) / nsim for inc, std in zip(yearly_average_deaths,
                                                                            std_yearly_death_incidence)]

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
    plt.savefig(save_directory + "/" + filename_description + "_" +
                f"Incidence_model_over_time_pop_{pop_size}_years_{yearsrun}_runs_{nsim}.png",
                bbox_inches='tight')
    plt.clf()
    # ===== Plot the number of deaths and number of injuries predicted by the model compared to the GBD data =====
    # Get the GBD death data
    GBD_death_data = pd.read_csv('resources/ResourceFile_Deaths_And_Causes_DeathRates_GBD.csv')
    # Isolate RTI deaths
    road_data = GBD_death_data.loc[GBD_death_data['cause_name'] == 'Road injuries']
    # Isolate RTI deaths in 2010
    road_data_2010 = road_data.loc[road_data['year'] == 2010]
    # Get the Malawian population values for 2010
    Malawi_Pop_2010 = pd.read_csv('resources/ResourceFile_Population_2010.csv')
    # Get an estimate of Malawis population size in 2010
    Malawi_pop_size_2010 = sum(Malawi_Pop_2010['Count'])
    # Calculate how much to scale the model's population size by to match Malawi's population
    scaler_to_pop_size = Malawi_pop_size_2010 / pop_size
    # Scale up the model's predicted deaths to match the estimate from Malawi
    scaled_2010_deaths = results_all_runs['deaths_2010'].mean() * scaler_to_pop_size
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
    plt.savefig(
        save_directory + "/" + filename_description + "_" +
        f"Number_of_deaths_comp_to_GBD_2010_pop_{pop_size}_years_{yearsrun}_runs_{nsim}.png",
        bbox_inches='tight')
    plt.clf()
    # ========== plot the model number of deaths and number of injuries compared to GBD estimates ===============
    # Get GBD data of the number of injuries
    injury_number_data = pd.read_csv('resources/ResourceFile_RTI_GBD_Injury_Categories.csv')
    # Isolate information on the number of injuries
    injury_number_data = injury_number_data.loc[injury_number_data['metric'] == 'Number']
    # Isolate number of injuries in 2010
    injury_number_in_2010 = injury_number_data.loc[injury_number_data['year'] == 2010]
    # Calculate number of injuries in 2010
    gbd_number_of_injuries_2010 = injury_number_in_2010['val'].sum()
    # Get the model's predicted number of injuries in 2010
    model_injury_number_in_2010 = results_all_runs['injuries_in_2010'].mean()
    # Scale this up to with respect to population size
    scaled_model_injury_number_in_2010 = model_injury_number_in_2010 * scaler_to_pop_size

    injury_output = [scaled_model_injury_number_in_2010, gbd_number_of_injuries_2010]
    # plot data in bar chart
    plt.bar(np.arange(2), injury_output, color='lightsalmon', label='number of injuries')
    plt.xticks(np.arange(2), ['Model output', 'GBD estimate'])
    plt.ylabel('Number')
    plt.title(f"The model's predicted number of road traffic injuries"
              f"\n"
              f"compared to the GBD 2010 estimates"
              f"\n"
              f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
    plt.savefig(save_directory + "/" + filename_description + "_" +
                f"Number_of_RTI_comp_to_GBD_2010_pop_{pop_size}_years_{yearsrun}_runs_{nsim}.png",
                bbox_inches='tight')
    plt.clf()
    # =================== Compare model deaths over popsize for model and Malawi GBD data ====================
    # Calculate deaths divided by populaton size for the model and the GBD data
    deaths_over_pop_size = [results_all_runs['deaths_2010'].mean() / pop_size,
                            sum(road_data_2010['val']) / Malawi_pop_size_2010]
    # Plot data in a bar chart
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
              f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
    # Plot the overall percent fatality of those involved in road traffic injuries
    plt.savefig(save_directory + "/" + filename_description + "_" +
                f"Number_of_deaths_over_pop_comp_to_GBD_2010_pop_{pop_size}_years_{yearsrun}_runs"
                f"_{nsim}.png", bbox_inches='tight')
    plt.clf()
    # ============ compare incidence of death in the model and the GBD 2010 estimate ===========================
    data = pd.read_csv('resources/ResourceFile_RTI_GBD_Number_And_Incidence_Data.csv')
    data = data.loc[data['metric'] == 'Rate']
    data = data.loc[data['year'] > 2009]
    death_data = data.loc[data['measure'] == 'Deaths']
    in_rti_data = data.loc[data['measure'] == 'Incidence']
    gbd_ten_year_average_inc = in_rti_data['val'].mean()
    gbd_ten_year_average_inc_death = death_data['val'].mean()
    plt.bar(np.arange(2), [np.mean(average_deaths), gbd_ten_year_average_inc_death], color='lightsalmon')
    plt.xticks(np.arange(2), ['Model incidence \nof death', 'GBD incidence \nof death'])
    plt.ylabel('Incidence of death')
    plt.title(f"The model's 10 year average predicted incidence of RTI related death "
              f"\n"
              f"compared to the average incidence of death from GBD study 2010-2019"
              f"\n"
              f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
    plt.savefig(save_directory + "/" + filename_description + "_" +
                f"Incidence_of_deaths_comp_to_GBD_2010_pop_{pop_size}_years_{yearsrun}_runs_"
                f"{nsim}.png", bbox_inches='tight')
    plt.clf()

    # plot ten year average of incidence compared to rti
    plt.bar(np.arange(2), [np.mean(average_incidence), gbd_ten_year_average_inc], color='lightsteelblue')
    plt.xticks(np.arange(2), ['Model incidence \nof RTI', 'GBD incidence \nof RTI'])
    plt.ylabel('Incidence of RTI')
    plt.title(f"The model's 10 year average predicted incidence of RTI"
              f"\n"
              f"compared to the average incidence of death from GBD study 2010-2019"
              f"\n"
              f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
    plt.savefig(save_directory + "/" + filename_description + "_" +
                f"Incidence_of_RTI_comp_to_GBD_2010_pop_{pop_size}_years_{yearsrun}_runs_"
                f"{nsim}.png", bbox_inches='tight')
    plt.clf()
    # =================== Plot the percentage of crashes that are fatal =========================================
    # Calculate the mean percentage of fatal crashes in simulation and standard deviation
    mean_fatal_crashes_of_all_sim = results_all_runs['percent_of_fatal_crashes'].mean()
    std_fatal_crashes = results_all_runs['percent_of_fatal_crashes'].std()
    # calculate mean percenatage of non fatal chrashes in all simulations
    mean_non_fatal = 1 - results_all_runs['percent_of_fatal_crashes'].mean()
    # calculate standard deviation of non-fatal chrashes in all simulations
    std_non_fatal_crashes = std_fatal_crashes
    # round data of to 3 decimal places
    data = [np.round(mean_fatal_crashes_of_all_sim, 3), np.round(mean_non_fatal, 3)]
    n = np.arange(2)
    # plot data in bar chart
    plt.bar(n, data, yerr=[std_fatal_crashes, std_non_fatal_crashes], color='lightsteelblue')
    # annotate graph with values for percentage of fatal and non fatal crashes
    for i in range(len(data)):
        plt.annotate(str(data[i]), xy=(n[i], data[i]), ha='center', va='bottom')
    plt.xticks(np.arange(2), ['fatal', 'non-fatal'])
    plt.title(f"Average percentage of those with RTI who perished"
              f"\n"
              f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
    plt.savefig(save_directory + "/" + filename_description + "_" +
                f"Percentage_of_deaths_pop_{pop_size}_years_{yearsrun}_runs_{nsim}.png",
                bbox_inches='tight')
    plt.clf()

    # Plot mean incidence of RTIs, mean number of rti deaths and the predicted incidence of RTI death
    gbd_data = [954.24, 12.13, 954.24]
    n = np.arange(len(gbd_data))
    tot_inc_injuries = results_all_runs['tot_inc_injuries'].tolist()
    tot_inc_injuries = [int(item) for sublist in tot_inc_injuries for item in sublist]
    mean_inc_total = np.mean(tot_inc_injuries)
    model_data = [np.mean(average_incidence), np.mean(average_deaths), mean_inc_total]
    plt.bar(n, gbd_data, width=0.4, color='lightsalmon', label='GBD estimates')
    plt.bar(n + 0.4, model_data, width=0.4, color='lightsteelblue', label='Model estimates')
    plt.xticks(n + 0.2, ['Incidence of people with RTIs', 'Incidence of death', 'Incidence of injuries'])
    for i in range(len(gbd_data)):
        plt.annotate(str(np.round(gbd_data[i], 2)), xy=(n[i], gbd_data[i]), ha='center', va='bottom')
    for i in range(len(model_data)):
        plt.annotate(str(np.round(model_data[i], 2)), xy=(n[i] + 0.4, model_data[i]), ha='center', va='bottom')
    plt.legend()
    plt.savefig(save_directory + "/" + filename_description + "_" +
                f"GBD_comparison_pop_{pop_size}_years_{yearsrun}_runs_{nsim}.png",
                bbox_inches='tight')
    plt.clf()
    # ============ Plot overall health system usage =============================================
    # Calculate average number of consumables used per sim
    mean_consumables_used_per_sim = results_all_runs['number_of_consumables_in_sim'].mean()
    # Calculate average number of inpatient days
    sim_inpatient_days = results_all_runs['this_sim_inpatient_days'].tolist()
    mean_inpatient_days = np.mean([np.mean(sim_days) for sim_days in sim_inpatient_days])
    # Calculate the standard deviation in each simulation, then calculate the average
    sd_inpatient_days = np.mean([np.std(sim_days) for sim_days in sim_inpatient_days])
    # Calculate average fraction of time used in health system
    mean_fraction_health_system_time_used = results_all_runs['health_system_time_usage'].mean()
    # Calculate average number of burn treatments issued
    average_number_of_burns_treated = results_all_runs['per_sim_burn_treated'].mean()
    # Calculate average number of fracture treatments issued
    average_number_of_fractures_treated = results_all_runs['per_sim_frac_cast'].mean()
    # Calculate average number of laceration treatments issued
    average_number_of_lacerations_treated = results_all_runs['per_sim_laceration'].mean()
    # Calculate average number of major surgeries performed
    average_number_of_major_surgeries_performed = results_all_runs['per_sim_major_surg'].mean()
    # Calculate average number of minor surgeries performed
    average_number_of_minor_surgeries_performed = results_all_runs['per_sim_minor_surg'].mean()
    # Calculate average number of tetanus vaccines issued
    average_number_of_tetanus_jabs = results_all_runs['per_sim_tetanus'].mean()
    # Calculate average number of pain management treatments provided
    average_number_of_pain_meds = results_all_runs['per_sim_pain_med'].mean()
    # Calculate average number of open fracture treatments provided
    average_number_of_open_fracs = results_all_runs['per_sim_open_frac'].mean()
    # plot the number of consumables used on a bar chart
    plt.bar(np.arange(1), mean_consumables_used_per_sim, color='lightsteelblue')
    plt.xticks(np.arange(1), ['Consumables'])
    plt.ylabel('Average consumables used')
    plt.title(f"Average number of consumables used per sim"
              f"\n"
              f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
    plt.savefig(save_directory + "/" + filename_description + "_" + "Average_consumables_used.png", bbox_inches='tight')
    plt.clf()

    # plot the average total number of inpatient days taken on a bar chart and compare it to reported
    # inpatient day usage in KCH, see https://doi.org/10.1016/j.ijso.2017.11.004
    kch_mean_inpatient_day_rti = 13.7
    kcd_sd_inpatient_day_rti = 19.6
    plt.bar(np.arange(2), [mean_inpatient_days, kch_mean_inpatient_day_rti],
            yerr=[sd_inpatient_days, kcd_sd_inpatient_day_rti],
            color=['lightsteelblue', 'lightsalmon'])
    plt.xticks(np.arange(2), ['Model', 'KCH'])
    plt.ylabel('Average total inpatient days used')
    plt.title(f"Average number of inpatient days used per sim in the model compared to inpatient day usage"
              f"\n"
              f"in Kamuzu Central Hospital"
              f"\n"
              f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
    plt.savefig(save_directory + "/" + filename_description + "_" + "Average_inpatient_days_"
                "used.png", bbox_inches='tight')
    plt.clf()
    # ================================== Plot inpatient day distribution ==================================================
    # Malawi injury inpatient days data from https://doi.org/10.1016/j.jsurg.2014.09.010
    flatten_inpatient_days = [day for day_list in sim_inpatient_days for day in day_list]
    # Create labels
    labels = ['< 1', '1', '2', '3', '4-7', '8-14', '15-30', '> 30']
    # get data from https://doi.org/10.1016/j.jsurg.2014.09.010
    inpatient_days_Tyson_et_al = [107 + 40, 854 + 56, 531 + 19, 365 + 22, 924 + 40, 705 + 23, 840 + 8, 555 + 11]
    # Calculate inpatient days distribution
    inpatient_days_Tyson_et_al_dist = np.divide(inpatient_days_Tyson_et_al, sum(inpatient_days_Tyson_et_al))
    # Sort model data to fit the above boundaries
    zero_days = [1 if inpatient_day == 0 else 0 for inpatient_day in flatten_inpatient_days]
    one_day = [1 if inpatient_day == 1 else 0 for inpatient_day in flatten_inpatient_days]
    two_days = [1 if inpatient_day == 2 else 0 for inpatient_day in flatten_inpatient_days]
    three_days = [1 if inpatient_day == 3 else 0 for inpatient_day in flatten_inpatient_days]
    four_to_seven_days = [1 if 4 <= inpatient_day < 7 else 0 for inpatient_day in flatten_inpatient_days]
    eight_to_fourteen = [1 if 8 <= inpatient_day < 14 else 0 for inpatient_day in flatten_inpatient_days]
    fifteen_to_thirty = [1 if 15 <= inpatient_day < 30 else 0 for inpatient_day in flatten_inpatient_days]
    thiry_plus = [1 if 30 <= inpatient_day else 0 for inpatient_day in flatten_inpatient_days]
    # Calculate the number of patients who needed x number of impatient days
    model_inpatient_days = [sum(zero_days), sum(one_day), sum(two_days), sum(three_days), sum(four_to_seven_days),
                            sum(eight_to_fourteen), sum(fifteen_to_thirty), sum(thiry_plus)]
    # Calculate distribution of inpatient days in model
    model_inpatient_days_dist = np.divide(model_inpatient_days, sum(model_inpatient_days))
    # plot data in a bar chart
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

    plt.savefig(save_directory + "/" + filename_description + "_" + "Inpatient_day_distribution_comp_"
                "to_kamuzu.png", bbox_inches='tight')
    plt.clf()

    # ====================== Compare inpatient admissions from the model to Kamuzu central hospital data =========
    # Based on Purcel et al. DOI: 10.1007/s00268-020-05853-z
    # Load data from Purcel et al
    percent_admitted_kch = 89.8
    # Calculate percentage of people admitted in the model
    percent_admitted_in_model = ((len(flatten_inpatient_days) - sum(zero_days)) / len(flattened_scores)) * 100
    # plot the comparison in a bar chart
    plt.bar(np.arange(2), [np.round(percent_admitted_in_model, 2), percent_admitted_kch], color='lightsteelblue')
    plt.xticks(np.arange(2), ['Percent admitted in model', 'Percent admitted in KCH'])
    plt.ylabel('Percentage')
    plt.title(f"Model percentage inpatient admission compared to Kamuzu central hospital data"
              f"\n"
              f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")

    plt.savefig(save_directory + "/" + filename_description + "_" + "Inpatient_admission_comp_to_kamuzu.png",
                bbox_inches='tight')
    plt.clf()

    # ============= Plot the distribution of inpatient days due to RTI ================================
    # use np.unique to calculate the distribution of inpatient days and to create labels
    days, counts = np.unique(flatten_inpatient_days, return_counts=True)
    # plot data in a bar chart
    plt.bar(days, counts / sum(counts), width=0.8, color='lightsteelblue')
    plt.xlabel('Inpatient days')
    plt.ylabel('Percentage of patients')
    plt.title(f"Distribution of inpatient days produced by the model for RTIs"
              f"\n"
              f"population size: {pop_size}, years modelled: {yearsrun}, number of runs: {nsim}")
    plt.savefig(save_directory + "/" + filename_description + "_" + "Inpatient_day_distribution.png",
                bbox_inches='tight')
    plt.clf()

