import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def get_incidence(logs_dict_file, module, complication, dictionary, specific_year, year, age_group):
    if specific_year:
        if f'tlo.methods.{module}' in logs_dict_file:
            if f'{age_group}_complication' in logs_dict_file[f'tlo.methods.{module}']:
                comps = logs_dict_file[f'tlo.methods.{module}'][f'{age_group}_complication']
                comps['date'] = pd.to_datetime(comps['date'])
                comps['year'] = comps['date'].dt.year
                dictionary[complication] = len(comps.loc[(comps['type'] == f'{complication}') &
                                                         (comps['year'] == year)])

    else:
        if f'tlo.methods.{module}' in logs_dict_file:
            if f'{age_group}_complication' in logs_dict_file[f'tlo.methods.{module}']:
                comps = logs_dict_file[f'tlo.methods.{module}'][f'{age_group}_complication']
                dictionary[complication] = len(comps.loc[(comps['type'] == f'{complication}')])


def get_prop_unintended_preg(logs_dict, dict):
    for file in logs_dict:
        if 'fail_contraception' in logs_dict[file]['tlo.methods.contraception']:
            comps = logs_dict[file]['tlo.methods.contraception']['fail_contraception']
            new_row = {file: len(comps)}
            dict.update(new_row)


def get_total_births(logs_dict_file):
    if 'on_birth' in logs_dict_file['tlo.methods.demography']:
        births_df = logs_dict_file['tlo.methods.demography']['on_birth']
        return len(births_df)


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


def get_pregnancies_in_a_year(logs_dict_file, year):
    preg_poll = logs_dict_file['tlo.methods.contraception']['pregnant_at_age']
    preg_poll['date'] = pd.to_datetime(preg_poll['date'])
    preg_poll['year'] = preg_poll['date'].dt.year
    pp_pregs = len(preg_poll.loc[preg_poll['year'] == year])

    if 'failed_contraception' in logs_dict_file['tlo.methods.contraception']:
        failed_contraception = logs_dict_file['tlo.methods.contraception']['fail_contraception']
        failed_contraception['date'] = pd.to_datetime(failed_contraception['date'])
        failed_contraception['year'] = failed_contraception['date'].dt.year
        fc_pregs = len(failed_contraception.loc[failed_contraception['year'] == year])
    else:
        fc_pregs = 0

    total_pregnancies = pp_pregs + fc_pregs

    return total_pregnancies


def get_pregnancies_from_dummy_contraception(logs_dict_file, year):
    preg_poll = logs_dict_file['tlo.methods.dummy_contraception']['pregnancy']
    preg_poll['date'] = pd.to_datetime(preg_poll['date'])
    preg_poll['year'] = preg_poll['date'].dt.year

    return len(preg_poll.loc[preg_poll['year'] == year])


def get_completed_pregnancies_in_a_year(logs_dict_file, master_dict):
    sum_ended_pregs_from_dict = master_dict['spontaneous_abortion'] + master_dict['induced_abortion'] + \
                                master_dict['ectopic_unruptured']

    births = logs_dict_file['tlo.methods.demography']['on_birth']
    all_births = len(births)

    an_stillbirths = logs_dict_file['tlo.methods.pregnancy_supervisor']['antenatal_stillbirth']
    all_stillbirths = len(an_stillbirths)

    total_ended_pregnancies = sum_ended_pregs_from_dict + all_births + all_stillbirths

    return total_ended_pregnancies


def get_parity_graphs(log_file):
    proportions_15_19 = dict()
    proportions_20_24 = dict()
    proportions_25_29 = dict()
    proportions_30_34 = dict()
    proportions_35_39 = dict()
    proportions_40_44 = dict()
    proportions_45_49 = dict()
    total = dict()
    all_ages = dict()

    def get_proportions(low_limit, high_limit, proportions, parity):
        parity_df = log_file['tlo.methods.labour']['parity']

        denom = len(parity_df.loc[(parity_df['age'] > low_limit) & (parity_df['age'] < high_limit)])

        if parity != 10:
            prop = (len(parity_df.loc[(parity_df['age'] > low_limit) & (parity_df['age'] < high_limit) &
                                      (parity_df['parity'] == parity)]) / denom) * 100
        else:
            prop = (len(parity_df.loc[(parity_df['age'] > low_limit) & (parity_df['age'] < high_limit) &
                                      (parity_df['parity'] >= parity)]) / denom) * 100
        new_row = {parity: prop}
        proportions.update(new_row)

    for parity in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        get_proportions(14, 20, proportions_15_19, parity)
        get_proportions(19, 25, proportions_20_24, parity)
        get_proportions(24, 31, proportions_25_29, parity)
        get_proportions(30, 35, proportions_30_34, parity)
        get_proportions(34, 40, proportions_35_39, parity)
        get_proportions(39, 45, proportions_40_44, parity)
        get_proportions(44, 50, proportions_45_49, parity)
        get_proportions(14, 50, total, parity)
        get_proportions(14, 100, all_ages, parity)

    def make_parity_graphs(dict, target_rates, age_group):
        N = 11
        model_rates = (dict[0], dict[1], dict[2], dict[3], dict[4], dict[5], dict[6], dict[7], dict[8], dict[9],
                       dict[10])
        target_rates = (target_rates)
        ind = np.arange(N)
        width = 0.35
        plt.bar(ind, model_rates, width, label='Model', color='seagreen')
        plt.bar(ind + width, target_rates, width,
                label='Target Rate', color='mediumseagreen')
        plt.ylabel('Proportion of total women')
        plt.title(f'Female parity at baseline for age group {age_group}')
        plt.xticks(ind + width / 2, ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
        plt.legend(loc='best')
        plt.show()

    make_parity_graphs(proportions_15_19, [79.9, 17.6, 2.3, 0.2, 0, 0, 0, 0, 0, 0, 0], '15_19')
    make_parity_graphs(proportions_20_24, [15.4, 31.7, 34.1, 14.5, 3.6, 0.6, 0.1, 0, 0, 0, 0], '20_24')
    make_parity_graphs(proportions_25_29, [3.8, 8.8, 22.6, 30.5, 23.2, 8, 2.6, 0.4, 0.1, 0, 0], '25_29')
    make_parity_graphs(proportions_30_34, [1.9, 4.3, 7.4, 18, 22.7, 22.7, 13.2, 6.1, 1.9, 0.4, 0.3], '30_34')
    make_parity_graphs(proportions_35_39, [2.1, 2.3, 4.3, 7.4, 13.3, 20.3, 19.8, 15.6, 8.5, 4.2, 2.3], '35_39')
    make_parity_graphs(proportions_40_44, [1.3, 2.4, 4.0, 6.1, 9.1, 12.8, 13.9, 21.6, 12.1, 8.3, 8.4], '40_44')
    make_parity_graphs(proportions_45_49, [1.6, 2.9, 3.6, 4.7, 6.6, 9.2, 12.4, 13.6, 13.8, 14, 17.5], '45_49')
    make_parity_graphs(total, [21.8, 13, 13.8, 12.9, 10.9, 8.7, 6.4, 5.2, 3.1, 2.1, 2.1], '15_49')
    make_parity_graphs(all_ages, [21.8, 13, 13.8, 12.9, 10.9, 8.7, 6.4, 5.2, 3.1, 2.1, 2.1], '15_100')


def get_htn_disorders_graph(master_dict_an, master_dict_la, master_dict_pn, denominator, year):

    gh_rate = ((master_dict_an['mild_gest_htn'] + master_dict_pn['mild_gest_htn']) / denominator) * 1000
    print(f'total gh rate {year} =', gh_rate)

    sgh_rate = ((master_dict_an['severe_gest_htn'] +
                 master_dict_la['severe_gest_htn'] +
                 master_dict_pn['severe_gest_htn']) / denominator) * 1000
    print(f'total sgh rate {year} =', sgh_rate)

    mpe_rate = ((master_dict_an['mild_pre_eclamp'] +
                master_dict_pn['mild_pre_eclamp']) / denominator) * 1000
    print(f'total mpe rate {year} =', mpe_rate)

    spe_rate = ((master_dict_an['severe_pre_eclamp'] +
                 master_dict_la['severe_pre_eclamp'] +
                 master_dict_pn['severe_pre_eclamp']) / denominator) * 1000
    print(f'total spe rate {year} =', spe_rate)

    ec_rate = ((master_dict_an['eclampsia'] + master_dict_la['eclampsia'] + master_dict_pn['eclampsia']) /
               denominator) * 1000
    print(f'total ec rate {year} =', ec_rate)

    N = 5
    model_rates = (gh_rate, sgh_rate, mpe_rate, spe_rate, ec_rate)
    target_rates = (36.8, 8.1, 44, 22, 10)
    ind = np.arange(N)
    width = 0.35
    plt.bar(ind, model_rates, width, label='Model', color='seagreen')
    plt.bar(ind + width, target_rates, width,
            label='Target Rate', color='mediumseagreen')
    plt.ylabel('Rate per 1000 pregnancies')
    plt.title(f'Rates of hypertensive disorders in {year}')
    plt.xticks(ind + width / 2, ('GH', 'SGH', 'MPE', 'SPE', 'EC'))
    plt.legend(loc='best')
    plt.show()

    """
    an_mpe_rate = (master_dict_an['mild_pre_eclamp'] / denominator) * 1000
    pn_mpe_rate = (master_dict_pn['mild_pre_eclamp'] / denominator) * 1000

    an_spe_rate = (master_dict_an['severe_pre_eclamp'] / denominator) * 1000
    la_spe_rate = (master_dict_la['severe_pre_eclamp'] / denominator) * 1000
    pn_spe_rate = (master_dict_pn['severe_pre_eclamp'] / denominator) * 1000

    an_ec_rate = (master_dict_an['eclampsia'] / denominator) * 1000
    la_ec_rate = (master_dict_la['eclampsia'] / denominator) * 1000
    pn_ec_rate = (master_dict_pn['eclampsia'] / denominator) * 1000

    an_gh_rate = (master_dict_an['mild_gest_htn'] / denominator) * 1000
    pn_gh_rate = (master_dict_pn['mild_gest_htn'] / denominator) * 1000

    an_sgh_rate = (master_dict_an['severe_gest_htn'] / denominator) * 1000
    la_sgh_rate = (master_dict_la['severe_gest_htn'] / denominator) * 1000
    pn_sgh_rate = (master_dict_pn['severe_gest_htn'] / denominator) * 1000

    labels = ['PE', 'SPE', 'EC', 'GH', 'SGH']
    an_rates = [an_mpe_rate, an_spe_rate, an_ec_rate, an_gh_rate, an_sgh_rate]
    la_rates =[0, la_spe_rate, la_ec_rate, 0, la_sgh_rate]
    pn_rates =[pn_mpe_rate, pn_spe_rate, pn_ec_rate, pn_gh_rate, pn_sgh_rate]

    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(labels, an_rates, width, label='Antenatal Rate', color='lightcoral')
    ax.bar(labels, la_rates, width, label='Intrapartum Rate', color='mistyrose')
    ax.bar(labels, pn_rates, width, bottom=pn_rates, label='Postnatal Rate', color='firebrick')

    ax.set_ylabel('Rate per 1000 pregnancies')
    ax.set_title('Hypertensive disorders of pregnancy')
    ax.legend()
    plt.show() """


def get_single_year_twins_graph(log_dict, total_births, target, colours):

    if 'twin_birth' in log_dict['tlo.methods.newborn_outcomes']:
        twin_births = len(log_dict['tlo.methods.newborn_outcomes']['twin_birth'])
    total_births_corr = total_births - twin_births

    rate = (twin_births / total_births_corr) * 100

    print('twins rate', rate, 'per 100 births')
    model_rates = [rate]
    target_rates = [target]

    barWidth = 0.35
    r1 = np.arange(len(model_rates))
    r2 = [x + barWidth for x in r1]

    plt.bar(r1, model_rates, width=barWidth, color=colours[0], capsize=7, label='model')
    plt.bar(r2, target_rates, width=barWidth, color=colours[1], capsize=7, label='target')

    plt.title('Modelled incidence of twin birth')
    plt.xticks([r + barWidth for r in range(len(model_rates))], ['twins'])
    plt.ylabel('proportion of all births')
    plt.legend()
    plt.show()


def get_single_year_generic_incidence_graph(complication, dict, denominator, target, colours, multiplier):
    rate = (dict[complication] / denominator) * multiplier
    print(f'{complication} rate', rate)
    model_rates = [rate]
    target_rates = [target]
    plt.style.use('ggplot')

    barWidth = 0.35
    r1 = np.arange(len(model_rates))
    r2 = [x + barWidth for x in r1]

    plt.bar(r1, model_rates, width=barWidth, color=colours[0], capsize=7, label='model')
    plt.bar(r2, target_rates, width=barWidth, color=colours[1], capsize=7, label='target')

    plt.title(f'Modelled incidence of {complication}')
    plt.xticks([r + barWidth for r in range(len(model_rates))], [f'{complication}'])
    plt.ylabel('rate per 1000')
    plt.legend()
    plt.show()


def get_generic_incidence_graph(complication, dict_2010, dict_2015, denominator_2010, denominator_2015, target_2010,
                                target_2015, colours):

    rate_2010 = (dict_2010[complication] / denominator_2010) * 1000
    print(f'{complication} rate 2010 {rate_2010}')
    rate_2015 = (dict_2015[complication] / denominator_2015) * 1000
    print(f'{complication} rate 2015 {rate_2015}')

    N = 2
    model_rates = (rate_2010, rate_2015)
    target_rates = (target_2010, target_2015)

    ind = np.arange(N)
    width = 0.35
    plt.bar(ind, model_rates, width, label='Model', color=colours[0])
    plt.bar(ind + width, target_rates, width, label='Target Rate', color=colours[1])
    plt.ylabel('Rate per 000 pregnancies')
    plt.title(f'Modelled incidence of {complication}')
    plt.xticks(ind + width / 2, ('2010', '2015'))
    plt.legend(loc='best')
    plt.show()


def get_total_anaemia_graph(logs_2010, logs_2015, colours):

    def get_anaemia_prevalence(log):
        anaemia_df = log['tlo.methods.pregnancy_supervisor']['anaemia_on_birth']
        total_prevalence = (len(anaemia_df.loc[anaemia_df['anaemia_status'] != 'none']) / len(anaemia_df)) * 100
        mild_prevalence = (len(anaemia_df.loc[anaemia_df['anaemia_status'] == 'mild']) / len(anaemia_df)) * 100
        moderate_prevalence = (len(anaemia_df.loc[anaemia_df['anaemia_status'] == 'moderate']) / len(anaemia_df)) * 100
        severe_prevalence = (len(anaemia_df.loc[anaemia_df['anaemia_status'] == 'severe']) / len(anaemia_df)) * 100

        return [total_prevalence, mild_prevalence, moderate_prevalence, severe_prevalence]

    prev_2010 = get_anaemia_prevalence(logs_2010)[0]
    print(f'anaemia prevalence 2010 {prev_2010}')
    prev_2015 = get_anaemia_prevalence(logs_2015)[0]
    print(f'anaemia prevalence 2015 {prev_2015}')

    mild_prev_10 = get_anaemia_prevalence(logs_2010)[1]
    mild_prev_15 = get_anaemia_prevalence(logs_2015)[1]
    mod_prev_10 = get_anaemia_prevalence(logs_2010)[2]
    mod_prev_15 = get_anaemia_prevalence(logs_2015)[2]
    sev_prev_10 = get_anaemia_prevalence(logs_2010)[3]
    sev_prev_15 = get_anaemia_prevalence(logs_2015)[3]

    N = 8
    model_rates = (prev_2010, prev_2015, mild_prev_10, mild_prev_15, mod_prev_10, mod_prev_15, sev_prev_10, sev_prev_15)
    target_rates = (34.5, 45.1, 19.5, 22.7, 17.8, 20.8, 0.2, 1.6)

    ind = np.arange(N)
    width = 0.35
    plt.bar(ind, model_rates, width, label='Model', color=colours[0])
    plt.bar(ind + width, target_rates, width, label='Target Rate', color=colours[1])
    plt.ylabel('Prevalence')
    plt.title('Prevalence of Maternal anaemia at birth')
    plt.xticks(ind + width / 2, ('2010', '2015', 'm2010', 'm2015', 'mo2010', 'mo2015', 's2010', 's2015'))
    plt.legend(loc='best')
    plt.show()


def get_incidence_graph_from_an_and_la(complication, dict_an_2010, dict_la_2010, dict_an_2015, dict_la_2015,
                                       denom_10, denom15, target_2010, target_2015, colours):

    total_comps_10 = dict_an_2010[complication] + dict_la_2010[complication]
    total_comps_15 = dict_an_2015[complication] + dict_la_2015[complication]

    rate_2010 = (total_comps_10 / denom_10) * 1000
    print(f'{complication} rate 2010 {rate_2010}')
    rate_2015 = (total_comps_15 / denom15) * 1000
    print(f'{complication} rate 2015 {rate_2015}')

    N = 2
    model_rates = (rate_2010, rate_2015)
    target_rates = (target_2010, target_2015)

    ind = np.arange(N)
    width = 0.35
    plt.bar(ind, model_rates, width, label='Model', color=colours[0])
    plt.bar(ind + width, target_rates, width, label='Target Rate', color=colours[1])
    plt.ylabel('Rate per 000 facility deliveries')
    plt.title(f'Modelled incidence of {complication}')
    plt.xticks(ind + width / 2, ('2010', '2015'))
    plt.legend(loc='best')
    plt.show()


def get_abortion_complication_graphs(dict_2010, dict_2015, denominator_2010, denominator_2015, colours):

    complicated_rate_sa_10 = (dict_2010['complicated_spontaneous_abortion'] / denominator_2010) * 1000
    complicated_proportion_sa_10 = (dict_2010['complicated_spontaneous_abortion'] /
                                    dict_2010['spontaneous_abortion']) * 100
    complicated_rate_sa_15 = (dict_2015['complicated_spontaneous_abortion'] / denominator_2015) * 1000
    complicated_proportion_sa_15 = (dict_2015['complicated_spontaneous_abortion'] /
                                    dict_2015['spontaneous_abortion']) * 100

    complicated_rate_ia_10 = (dict_2010['complicated_induced_abortion'] / denominator_2010) * 1000
    complicated_proportion_ia_10 = (dict_2010['complicated_induced_abortion'] /
                                    dict_2010['induced_abortion']) * 100
    complicated_rate_ia_15 = (dict_2015['complicated_induced_abortion'] / denominator_2015) * 1000
    complicated_proportion_ia_15 = (dict_2015['complicated_induced_abortion'] /
                                    dict_2015['induced_abortion']) * 100

    N = 4
    model_rates = (complicated_rate_sa_10, complicated_rate_sa_15, complicated_rate_ia_10, complicated_rate_ia_15)
    target_rates = (26, 22, 24, 59)

    ind = np.arange(N)
    width = 0.35
    plt.bar(ind, model_rates, width, label='Model', color=colours[0])
    plt.bar(ind + width, target_rates, width, label='Target Rate', color=colours[1])
    plt.ylabel('Rate per 000 pregnancies')
    plt.title('Rate of complicated spontaneous and induced abortion')
    plt.xticks(ind + width / 2, ('SA10', 'SA15', 'IA10', 'IA15'))
    plt.legend(loc='best')
    plt.show()

    N = 4
    model_rates = (complicated_proportion_sa_10, complicated_proportion_sa_15,
                   complicated_proportion_ia_10, complicated_proportion_ia_15)

    target_rates = (14, 12, 28, 37)
    ind = np.arange(N)
    width = 0.35
    plt.bar(ind, model_rates, width, label='Model', color=colours[0])
    plt.bar(ind + width, target_rates, width, label='Target Rate', color=colours[1])
    plt.ylabel('Proportion of total abortion by type')
    plt.title('Proportion of abortions leading to complications')
    plt.xticks(ind + width / 2, ('SA10', 'SA15', 'IA10', 'IA15'))
    plt.legend(loc='best')
    plt.show()

    sepsis_sa_10 = (dict_2010['spontaneous_abortion_sepsis'] / dict_2010['complicated_spontaneous_abortion']) * 100
    sepsis_sa_15 = (dict_2015['spontaneous_abortion_sepsis'] / dict_2015['complicated_spontaneous_abortion']) * 100

    haem_sa_10 = (dict_2010['spontaneous_abortion_haemorrhage'] / dict_2010['complicated_spontaneous_abortion']) * 100
    haem_sa_15 = (dict_2015['spontaneous_abortion_haemorrhage'] / dict_2015['complicated_spontaneous_abortion']) * 100

    other_sa_10 = (dict_2010['spontaneous_abortion_other_comp'] / dict_2010['complicated_spontaneous_abortion']) * 100
    other_sa_15 = (dict_2015['spontaneous_abortion_other_comp'] / dict_2015['complicated_spontaneous_abortion']) * 100

    sepsis_ia_10 = (dict_2010['induced_abortion_sepsis'] / dict_2010['complicated_induced_abortion']) * 100
    sepsis_ia_15 = (dict_2015['induced_abortion_sepsis'] / dict_2015['complicated_induced_abortion']) * 100

    haem_ia_10 = (dict_2010['induced_abortion_haemorrhage'] / dict_2010['complicated_induced_abortion']) * 100
    haem_ia_15 = (dict_2015['induced_abortion_haemorrhage'] / dict_2015['complicated_induced_abortion']) * 100

    inj_ia_10 = (dict_2010['induced_abortion_injury'] / dict_2010['complicated_induced_abortion']) * 100
    inj_ia_15 = (dict_2015['induced_abortion_injury'] / dict_2015['complicated_induced_abortion']) * 100

    other_ia_10 = (dict_2010['induced_abortion_other_comp'] / dict_2010['complicated_induced_abortion']) * 100
    other_ia_15 = (dict_2015['induced_abortion_other_comp'] / dict_2015['complicated_induced_abortion']) * 100

    N = 14
    model_rates = (sepsis_sa_10, sepsis_sa_15, haem_sa_10, haem_sa_15,  other_sa_10, other_sa_15,  sepsis_ia_10,
                   sepsis_ia_15, haem_ia_10, haem_ia_15, inj_ia_10, inj_ia_15, other_ia_10, other_ia_15)

    target_rates = (23, 23, 13.5, 13.5, 0, 0, 23, 23, 13.5, 13.5, 5, 5, 0, 0)
    ind = np.arange(N)
    width = 0.35
    plt.bar(ind, model_rates, width, label='Model', color=colours[0])
    plt.bar(ind + width, target_rates, width, label='Target Rate', color=colours[1])
    plt.ylabel('% total complicated abortions')
    plt.title('Proportion of complicated abortions by complication')
    plt.xticks(ind + width / 2, ('S.SA10', 'S.SA15', 'H.SA10', 'H.SA15', 'O.SA10', 'O.SA15', 'S.IA10', 'S.IA15',
                                 'H.AA10', 'H.IA15', 'I.IA10', 'I.IA15', 'O.IA10', 'O.IA15'), rotation=45)
    plt.legend(loc='best')
    plt.show()


def get_preterm_birth_graph(dict_2010, dict_2015, total_births_2010, total_births_2015, colours):

    early_rate_2010 = (dict_2010['early_preterm_labour'] / total_births_2010) * 100
    print(f'eptl rate 2010 {early_rate_2010}')
    late_rate_2010 = (dict_2010['late_preterm_labour'] / total_births_2010) * 100
    print(f'lptl rate 2010 {late_rate_2010}')
    total_preterm_rate_2010 = early_rate_2010 + late_rate_2010
    print(f'total ptl rate 2010 {total_preterm_rate_2010}')

    early_rate_2015 = (dict_2015['early_preterm_labour'] / total_births_2015) * 100
    print(f'eptl rate 2015 {early_rate_2015}')
    late_rate_2015 = (dict_2015['late_preterm_labour'] / total_births_2015) * 100
    print(f'lptl rate 2015 {late_rate_2015}')
    total_preterm_rate_2015 = early_rate_2015 + late_rate_2015
    print(f'total ptl rate 2015 {total_preterm_rate_2015}')

    N = 6
    model_rates = (total_preterm_rate_2010, late_rate_2010, early_rate_2010, total_preterm_rate_2015, early_rate_2015,
                   late_rate_2015)
    target_rates = (19.3, 14.5, 4.7, 19.3, 14.5, 4.7)

    ind = np.arange(N)
    width = 0.35
    plt.bar(ind, model_rates, width, label='Model', color=colours[0])
    plt.bar(ind + width, target_rates, width, label='Target Rate', color=colours[1])
    plt.ylabel('Rate per 00 births')
    plt.title('Modelled incidence of Preterm Birth')
    plt.xticks(ind + width / 2, ('Rate 10', 'Late 10', 'Early 10', 'Total 15', 'Late 15', 'Early 15', ))
    plt.legend(loc='best')
    plt.show()


def get_anc_coverage_graph(logs_dict_file, year):

    # anc1 = logs_dict_file['tlo.methods.care_of_women_during_pregnancy']['anc1']
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
    print(f'anc1_rate {year}', anc1_rate)
    anc4_rate = (total_anc4 / total_anc_num) * 100
    print(f'anc4_rate {year}', anc4_rate)
    anc8_rate = (total_anc8 / total_anc_num) * 100
    print(f'anc8_rate {year}', anc8_rate)

    # todo: is this the right denominator, otherwise we need to catch gestation of  anc 1 at birth
    month_less_than_4 = (anc1_3 / total_women_anc1) * 100
    print(f'anc1 < 4 months {year}', month_less_than_4)
    month_5_5 = (anc1_4_5 / total_women_anc1) * 100
    print(f'anc1 4-5 months {year}', month_5_5)
    month_6_7 = (anc1_months_6_7 / total_women_anc1) * 100
    print(f'anc1 6-7 months {year}', month_6_7)
    month_8_plus = (anc1_months_8 / total_women_anc1) * 100
    print(f'anc1 8+ months {year}', month_8_plus)

    median_week = sum(medians) / len(medians)  # todo: this might not be right
    median_month = median_week / 4.5  # todo: replace with a better check
    print(f'median month first visit {year}', median_month)

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


def get_coverage_of_anc_interventions(logs_dict_file):

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

    if 'anc1' in logs_dict_file['tlo.methods.care_of_women_during_pregnancy']:
        total_anc = logs_dict_file['tlo.methods.care_of_women_during_pregnancy']['anc1']
        total_women_anc = len(total_anc)

    if 'anc_interventions' in logs_dict_file['tlo.methods.care_of_women_during_pregnancy']:
        ints_orig = logs_dict_file['tlo.methods.care_of_women_during_pregnancy']['anc_interventions']
        ints_no_date = ints_orig.drop(['date'], axis=1)
        duplicates = ints_no_date.duplicated(subset=None, keep='first')
        final_ints = ints_no_date.drop(index=duplicates.loc[duplicates].index)

        for intervention in coverage:
            total_intervention = len(final_ints.loc[(final_ints['intervention'] == intervention)])
            coverage[intervention] = (total_intervention / total_women_anc) * 100

    labels = ['DS', 'BP', 'ADM.', 'DSc', 'IFA', 'BEP', 'ITN', 'TT', 'CA', 'Hb', 'AL', 'HEP', 'S.Te', 'S.Tr',
              'HIV', 'IPTP', 'GDM']
    N = 17
    model_rates = coverage.values()
    ind = np.arange(N)
    width = 0.35
    plt.bar(ind, model_rates, width, label='Model', color='lightcoral')
    plt.ylabel('Proportion of women who attended ANC during pregnancy')
    plt.title('Coverage indicators of ANC')
    plt.xticks(ind + width / 2, labels)
    plt.legend(loc='best')
    plt.show()


def get_total_facility_deliveries(logs_dict_file):
    if 'delivery_setting' in logs_dict_file['tlo.methods.labour']:
        fd_df = logs_dict_file['tlo.methods.labour']['delivery_setting']
        facility_deliveries = \
            len(fd_df.loc[fd_df['facility_type'] == 'hospital']) + len(fd_df.loc[fd_df['facility_type'] == 'health_'
                                                                                                           'centre'])

        return facility_deliveries


def get_facility_delivery_graph(logs_dict_file, total_births, year):
    hospital_deliveries = 0
    health_centre_deliveries = 0

    if 'delivery_setting' in logs_dict_file['tlo.methods.labour']:
        facility_deliveries = logs_dict_file['tlo.methods.labour']['delivery_setting']
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


def get_pnc_coverage(logs_dict_file, total_births, year):

    if 'postnatal_check' in logs_dict_file['tlo.methods.labour']:
        maternal_pnc = logs_dict_file['tlo.methods.labour']['postnatal_check']
        maternal_pnc.drop(['date'], axis=1)
        duplicates = maternal_pnc.duplicated(subset=None, keep='first')
        maternal_pnc.drop(index=duplicates.index)
        early_pnc_mother = len(maternal_pnc.loc[maternal_pnc['timing'] == 'early'])

    if 'postnatal_check' in logs_dict_file['tlo.methods.newborn_outcomes']:
        neonatal_pnc = logs_dict_file['tlo.methods.newborn_outcomes']['postnatal_check']
        neonatal_pnc.drop(['date'], axis=1)
        duplicates = neonatal_pnc.duplicated(subset=None, keep='first')
        neonatal_pnc.drop(index=duplicates.index)
        early_neonatal_pnc = len(neonatal_pnc.loc[neonatal_pnc['timing'] == 'early'])

    if 'total_mat_pnc_visits' in logs_dict_file['tlo.methods.postnatal_supervisor']:
        total_visits = logs_dict_file['tlo.methods.postnatal_supervisor']['total_mat_pnc_visits']
        more_than_zero = len(total_visits.loc[total_visits['visits'] > 0])
        one_visit = len(total_visits.loc[total_visits['visits'] == 1])
        two_visit = len(total_visits.loc[total_visits['visits'] == 2])
        two_plus_visit = len(total_visits.loc[total_visits['visits'] > 2])

    if 'total_neo_pnc_visits' in logs_dict_file['tlo.methods.postnatal_supervisor']:
        total_visits_n = logs_dict_file['tlo.methods.postnatal_supervisor']['total_neo_pnc_visits']
        more_than_zero_n = len(total_visits_n.loc[total_visits_n['visits'] > 0])
        one_visit_n = len(total_visits_n.loc[total_visits_n['visits'] == 1])
        two_visit_n = len(total_visits_n.loc[total_visits_n['visits'] == 2])
        two_plus_visit_n = len(total_visits_n.loc[total_visits_n['visits'] > 2])

    maternal_pnc_coverage = (len(maternal_pnc) / total_births) * 100
    print(f'maternal pnc coverage {year}', maternal_pnc_coverage)
    neonatal_pnc_coverage = (len(neonatal_pnc) / total_births) * 100
    print(f'neonatal pnc coverage {year}', neonatal_pnc_coverage)

    early_pnc_mother = (early_pnc_mother / total_births) * 100
    print(f'early maternal pnc coverage {year}', early_pnc_mother)
    early_neonatal_pnc = (early_neonatal_pnc / total_births) * 100
    print(f'early neonatal  pnc coverage {year}', early_neonatal_pnc)

    N = 4
    model_rates = (maternal_pnc_coverage, early_pnc_mother, neonatal_pnc_coverage, early_neonatal_pnc)
    if year == 2010:
        target_rates = (50, 43, 60, 60)
        colours = ['midnightblue', 'lavender']
    else:
        target_rates = (48, 42, 60, 60)
        colours = ['goldenrod', 'cornsilk']

    ind = np.arange(N)
    width = 0.35
    plt.bar(ind, model_rates, width, label='Model', color=colours[0])
    plt.bar(ind + width, target_rates, width, label='Target Rate', color=colours[1])
    plt.ylabel('Proportion of women and newborns recently born attending PNC')
    plt.title(f'Postnatal care coverage rates in {year}')
    plt.xticks(ind + width / 2, ('mPNC', 'emPNC', 'nPNC', 'enPNC'))
    plt.legend(loc='best')
    plt.show()

    one_visit_rate = (one_visit / more_than_zero) * 100
    two_visit_rate = (two_visit / more_than_zero) * 100
    two_plus_visit_rate = (two_plus_visit / more_than_zero) * 100

    objects = ('PNC1', 'PNC2', 'PNC2+')
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, [one_visit_rate, two_visit_rate, two_plus_visit_rate], align='center', alpha=0.5, color='grey')
    plt.xticks(y_pos, objects)
    plt.ylabel('PNC visits/Total women with 1 or more visit')
    plt.title('Distribution of maternal PNC visits')
    plt.show()

    one_visit_rate_n = (one_visit_n / more_than_zero_n) * 100
    two_visit_rate_n = (two_visit_n / more_than_zero_n) * 100
    two_plus_visit_rate_n = (two_plus_visit_n / more_than_zero_n) * 100

    objects = ('PNC1', 'PNC2', 'PNC2+')
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, [one_visit_rate_n, two_visit_rate_n, two_plus_visit_rate_n], align='center', alpha=0.5, color='pink')
    plt.xticks(y_pos, objects)
    plt.ylabel('PNC visits/Total neonates with 1 or more visit')
    plt.title('Distribution of neonatal PNC visits')
    plt.show()


def get_mmr(logs_dict_file, list, live_births, year):
    total_direct_death = 0
    total_indirect_death = 0

    if 'death' in logs_dict_file['tlo.methods.demography']:
        total_deaths = logs_dict_file['tlo.methods.demography']['death']
        for cause in list:
            number_of_deaths = len(total_deaths.loc[(total_deaths['cause'] == f'{cause}')])

            total_direct_death += number_of_deaths

        deaths = total_deaths.loc[
            total_deaths['pregnancy'] & (total_deaths['cause'].str.contains('AIDS|severe_malaria|Suicide|diabetes|'
                                                                            'chronic_kidney_disease|'
                                                                            'chronic_ischemic_hd'))]
        indirect_deaths_preg_2011 = len(deaths)

        indirect_deaths_postnatal_2011 = len(
            total_deaths.loc[total_deaths['postnatal'] &
                             (total_deaths['cause'].str.contains('AIDS|severe_malaria|Suicide|diabetes|'
                                                                 'chronic_kidney_disease|chronic_ischemic_hd'))])

        total_indirect_death += indirect_deaths_preg_2011
        total_indirect_death += indirect_deaths_postnatal_2011

    # maternal_deaths = total_direct_death + total_indirect_death
    # total_mmr = (maternal_deaths / live_births) * 100000

    # prop_indirect_deaths = (total_indirect_death / maternal_deaths) * 100
    indirect_mmr = (total_indirect_death / live_births) * 100000
    direct_mmr = (total_direct_death / live_births) * 100000

    # PLOT ...
    labels = [f'{year}', 'Calib. Target']

    if year == 2010:
        direct_deaths = [direct_mmr, 540]
        indirect_deaths = [indirect_mmr, 135]
        print('2010 direct model mmr', direct_mmr, 'direct target mmr', 540)
        print('2010 indirect model mmr', indirect_mmr, 'direct target mmr', 135)

    else:
        direct_deaths = [direct_mmr, 307]
        indirect_deaths = [indirect_mmr, 132]
        print('2015 direct model mmr', direct_mmr, 'direct target mmr', 307)
        print('2015 indirect model mmr', indirect_mmr, 'direct target mmr', 132)

    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(labels, direct_deaths, width, label='Direct Deaths')
    ax.bar(labels, indirect_deaths, width, bottom=direct_deaths,
           label='Indirect Deaths')

    ax.set_ylabel('Maternal Deaths per 100,000 live births')
    ax.set_title(f'Maternal Mortality Ratio Calibration {year}')
    ax.legend()
    plt.show()


def output_distribution_of_ga_at_birth_for_logfile_year(logs_dict):
    ga_births = logs_dict['tlo.methods.labour']['live_birth']['ga'].sort_values(ascending=True).astype(int)
    counts = ga_births.value_counts()

    total = sum(counts)
    for week in counts.index:
        counts.at[week] = (counts.at[week]/total) * 100

    x = counts.index
    ga = counts.values

    x_pos = [i for i, _ in enumerate(x)]

    plt.style.use('ggplot')
    plt.bar(x_pos, ga, color='green')
    plt.xlabel("Weeks")
    plt.ylabel("Proportion of total births")
    plt.title("Gestational Age in Weeks at Birth ")

    plt.xticks(x_pos, x)

    plt.show()
