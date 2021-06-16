
from tlo import Module, Property, Types, logging
from tlo.methods import Metadata

from matplotlib import pyplot as plt
import numpy as np
from tlo import Date
import statistics as st
import pandas as pd


def get_incidence(logs_dict, module, complication, dictionary):
    incidence = dict()
    for file in logs_dict:
        if f'tlo.methods.{module}' in logs_dict[file]:
            if 'maternal_complication' in logs_dict[file][f'tlo.methods.{module}']:
                comps = logs_dict[file][f'tlo.methods.{module}']['maternal_complication']
                data = {file: len(comps.loc[(comps['type'] == f'{complication}')])}

                incidence.update(data)

    dictionary[complication] = incidence


def get_prop_unintended_preg(logs_dict, dict):
    for file in logs_dict:
        if 'fail_contraception' in logs_dict[file]['tlo.methods.contraception']:
            comps = logs_dict[file]['tlo.methods.contraception']['fail_contraception']
            new_row = {file: len(comps)}
            dict.update(new_row)


def get_total_births(logs_dict):
    births = 0
    for file in logs_dict:
        if 'on_birth' in logs_dict[file]['tlo.methods.demography']:
            births_df = logs_dict[file]['tlo.methods.demography']['on_birth']
            births += len(births_df)
    return births


def get_death_from_a_comp(logs_dict, cause):
    deaths = 0
    for file in logs_dict:
        if 'death' in logs_dict[file]['tlo.methods.demography']:
            deaths_df = logs_dict[file]['tlo.methods.demography']['death']
            deaths += len(deaths_df.loc[deaths_df['cause'] == f'{cause}'])
    return deaths


def get_an_stillbirths(logs_dict):
    stillbirths = 0
    for file in logs_dict:
        if 'antenatal_stillbirth' in logs_dict[file]['tlo.methods.pregnancy_supervisor']:
            sb_df = logs_dict[file]['tlo.methods.pregnancy_supervisor']['antenatal_stillbirth']
            stillbirths += len(sb_df)
    return stillbirths


def get_ip_stillbirths(logs_dict):
    ip_stillbirths = 0
    for file in logs_dict:
        if 'tlo.methods.labour' in logs_dict[file]:
            if 'intrapartum_stillbirth' in logs_dict[file]['tlo.methods.labour']:
                sb_df = logs_dict[file]['tlo.methods.labour']['intrapartum_stillbirth']
                ip_stillbirths += len(sb_df)
            if 'intrapartum_stillbirth' in logs_dict[file]['tlo.methods.newborn_outcomes']:
                sb_df = logs_dict[file]['tlo.methods.newborn_outcomes']['intrapartum_stillbirth']
                ip_stillbirths += len(sb_df)

    return ip_stillbirths


def get_htn_disorders_graph(master_dict_an, master_dict_la):

    gh_rate = (sum(master_dict_an['mild_gest_htn'].values()) / 10000) * 1000
    sgh_rate = ((sum(master_dict_an['severe_gest_htn'].values()) +
                 sum(master_dict_la['severe_gest_htn'].values())) / 10000) * 1000
    mpe_rate = (sum(master_dict_an['mild_pre_eclamp'].values()) / 10000) * 1000
    spe_rate = ((sum(master_dict_an['severe_pre_eclamp'].values()) +
                 sum(master_dict_la['severe_pre_eclamp'].values())) / 10000) * 1000
    ec_rate = ((sum(master_dict_an['eclampsia'].values()) +
                sum(master_dict_la['eclampsia'].values())) / 10000) * 1000

    N = 5
    model_rates = (gh_rate, sgh_rate, mpe_rate, spe_rate, ec_rate)
    target_rates = (25.7, 5.67, 30.8, 22, 10)
    ind = np.arange(N)
    width = 0.35
    plt.bar(ind, model_rates, width, label='Model', color='seagreen')
    plt.bar(ind + width, target_rates, width,
            label='Target Rate', color='mediumseagreen')
    plt.ylabel('Rate per 1000 pregnancies')
    plt.title('Rates of hypertensive disorders (antenatal + intrapartum)')
    plt.xticks(ind + width / 2, ('GH', 'SGH', 'MPE', 'SPE', 'EC'))
    plt.legend(loc='best')
    plt.show()

def get_generic_incidence_graph(master_dict, complication, target):


    rate = (sum(master_dict[f'{complication}'].values()) / 10000) * 1000
    objects = ('Total FDR', 'Hospital DR', 'Health Centre DR', 'Calibration')
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, [fd_rate_2010, hpd_rate_2010, hcd_rate_2010, 73], align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Facility Deliveries/ Total births')
    plt.title('Facility Delivery Rate 2010')
    plt.show()


def get_anc_coverage_graph(logs_dict_file, year):

    anc1 = logs_dict_file['tlo.methods.care_of_women_during_pregnancy']['anc1']
    anc1_at_birth = logs_dict_file['tlo.methods.care_of_women_during_pregnancy']['anc_ga_first_visit']
    total_anc = logs_dict_file['tlo.methods.care_of_women_during_pregnancy']['anc_count_on_birth']
    total_anc_num = len(total_anc)
    total_women_anc1 = len(anc1_at_birth)

    # of all women who delivered in this simulation...how many anc visits did they attenc
    total_anc1 = len(total_anc.loc[total_anc['total_anc'] > 0])
    total_anc4 = len(total_anc.loc[total_anc['total_anc'] > 3])
    total_anc8 = len(total_anc.loc[total_anc['total_anc'] > 7])

    # of the women who delivered in this simulation AND attendend at least one ANC visits, when did they attend
    anc1_3 = len(anc1_at_birth.loc[anc1_at_birth['ga_anc_one'] <= 13])
    anc1_4_5 = len(anc1_at_birth.loc[(anc1_at_birth['ga_anc_one'] > 13) & (anc1_at_birth['ga_anc_one'] <= 22)])
    anc1_months_6_7 = len(anc1_at_birth.loc[(anc1_at_birth['ga_anc_one'] > 22) & (anc1_at_birth['ga_anc_one'] < 32)])
    anc1_months_8 = len(anc1_at_birth.loc[anc1_at_birth['ga_anc_one'] > 31])

    medians = list()
    medians.append(anc1_at_birth['ga_anc_one'].median())

    anc1_rate = (total_anc1 / total_anc_num) * 100
    anc4_rate = (total_anc4 / total_anc_num) * 100
    anc8_rate = (total_anc8 / total_anc_num) * 100

    # todo: is this the right denominator, otherwise we need to catch gestation of  anc 1 at birth
    month_less_than_4 = (anc1_3 / total_women_anc1) * 100
    month_5_5 = (anc1_4_5 / total_women_anc1) * 100
    month_6_7 = (anc1_months_6_7 / total_women_anc1) * 100
    month_8_plus = (anc1_months_8 / total_women_anc1) * 100

    median_week = sum(medians) / len(medians)  # todo: this might not be right
    median_month = median_week / 4.5  # todo: replace with a better check

    N = 8
    model_rates = (anc1_rate, anc4_rate, anc8_rate, median_month, month_less_than_4, month_5_5, month_6_7, month_8_plus)
    if year == 2010:
        target_rates = (94.7, 46, 2, 5.6, 12.4, 48.2, 35.6, 2)
        colours = ['lightcoral', 'firebrick']
    else:
        target_rates = (95, 51, 2, 4.8, 24, 51.2, 21.4, 1.5)
        colours = ['slategrey', 'lightsteelblue']

    ind = np.arange(N)
    width = 0.35
    plt.bar(ind, model_rates, width, label='Model', color=colours[0])
    plt.bar(ind + width, target_rates, width, label='Target Rate', color=colours[1])
    plt.ylabel('Proportion of women who attended ANC during pregnancy')
    plt.title(f'Coverage indicators of ANC in {year}')
    plt.xticks(ind + width / 2, ('ANC1', 'ANC4+', 'ANC8+', 'Med.M A1', '<M4', 'M4/5', 'M6/7', 'M8+'))
    plt.legend(loc='best')
    plt.show()


def get_coverage_of_anc_interventions(logs_dict):

    interventions = ['dipstick', 'bp_measurement', 'admission', 'depression_screen', 'iron_folic_acid', 'b_e_p',
                     'LLITN', 'tt', 'calcium', 'hb_screen', 'albendazole', 'hep_b', 'syphilis_test', 'syphilis_treat',
                     'hiv_screen', 'iptp', 'gdm_screen']
    coverage = dict()
    for intervention in interventions:
        row = {intervention: 0}
        coverage.update(row)

    total_women_anc = 0

    # todo: this might not be the right denominator, as women arent able to receive all inteventions (wouldnt be that
    #  hard to do it by visit number i.e. of women who attended 3 visits did they have all the visit three info)

    for file in logs_dict:
        if 'anc1' in logs_dict[file]['tlo.methods.care_of_women_during_pregnancy']:
            total_anc = logs_dict[file][f'tlo.methods.care_of_women_during_pregnancy']['anc1']

            date_corrected = total_anc.loc[total_anc['date'] < Date(2010, 10, 1)]
            total_women_anc += len(date_corrected)

    for file in logs_dict:
        if 'anc_interventions' in logs_dict[file]['tlo.methods.care_of_women_during_pregnancy']:
            ints_orig = logs_dict[file][f'tlo.methods.care_of_women_during_pregnancy']['anc_interventions']
            ints = ints_orig.loc[ints_orig['date'] < Date(2010, 10, 1)]
            ints.drop(['date'], axis=1)
            duplicates = ints.duplicated(subset=None, keep='first')
            ints.drop(index=duplicates.index)

            for intervention in coverage:
                total_intervention = len(ints.loc[(ints['intervention'] == intervention)])
                cover = (total_intervention / total_women_anc) * 100
                coverage[intervention] += cover

    for intervention in coverage:
        coverage[intervention] = coverage[intervention] / len(logs_dict.keys())

    labels = ['DS', 'BP', 'ADM.', 'DSc', 'IFA', 'BEP', 'ITN', 'TT', 'CA', 'Hb', 'AL', 'HEP', 'S.Te', 'S.Tr',
              'HIV', 'IPTP', 'GDM']
    N = 17
    model_rates = coverage.values()
    ind = np.arange(N)
    width = 0.35
    plt.bar(ind, model_rates, width, label='Model', color='lightcoral')
    plt.ylabel('Proportion of women who attended ANC during pregnancy')
    plt.title('Coverage indicators of ANC')
    plt.xticks(ind + width / 2, (labels))
    plt.legend(loc='best')
    plt.show()


def get_facility_delivery_graph(logs_dict_file, total_births, year):
    hospital_deliveries = 0
    health_centre_deliveries = 0

    if 'delivery_setting' in logs_dict_file['tlo.methods.labour']:
        facility_deliveries = logs_dict_file[f'tlo.methods.labour']['delivery_setting']
        hospital_deliveries += len(facility_deliveries.loc[facility_deliveries['facility_type'] == 'hospital'])
        health_centre_deliveries += len(facility_deliveries.loc[facility_deliveries['facility_type'] ==
                                                                'health_centre'])

    hpd_rate = (hospital_deliveries / total_births) * 100
    hcd_rate = (health_centre_deliveries / total_births) * 100
    fd_rate = ((hospital_deliveries + health_centre_deliveries) / total_births) * 100
    hb_rate = 100 - fd_rate
    if hb_rate < 0:
        hb_rate = 0

    print(f'{year}, FD {fd_rate}, HP {hpd_rate}, HC {hcd_rate}, HB {hb_rate}')

    N = 4
    model_rates = (fd_rate, hpd_rate, hcd_rate, hb_rate)
    if year == 2010:
        target_rates = (73, 32, 41, 27)
        colours = ['midnightblue', 'lavender']
    else:
        target_rates = (91, 40, 52, 8)
        colours = ['goldenrod', 'cornsilk']

    ind = np.arange(N)
    width = 0.35
    plt.bar(ind, model_rates, width, label='Model', color=colours[0])
    plt.bar(ind + width, target_rates, width, label='Target Rate', color=colours[1])
    plt.ylabel('Proportion of total births by delivery locations')
    plt.title(f'Facility and Home Delivery Rates in {year}')
    plt.xticks(ind + width / 2, ('FDR', 'Ho.FDR', 'Hc.FDR', 'HBR'))
    plt.legend(loc='best')
    plt.show()
