from pathlib import Path
import os
import pandas as pd
from matplotlib import pyplot as plt

from tlo.analysis.utils import (
    extract_results,
    get_scenario_outputs,create_pickles_locally
)

from src.scripts.maternal_perinatal_analyses import analysis_utility_functions
plt.style.use('seaborn')

def output_key_outcomes_from_scenario_file(scenario_filename, pop_size, outputspath, sim_years, show_and_store_graphs):
    """
    :param scenario_filename:
    :param pop_size:
    :param outputspath:
    :param sim_years:
    :param show_and_store_graphs:
    :return:
    """

    # Find results folder (most recent run generated using that scenario_filename)
    results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]

    # Create folder to store graphs (if it hasnt already been created when ran previously)
    path = f'{outputspath}/calibration_output_graphs_{pop_size}_{results_folder.name}'
    if not os.path.isdir(path):
        os.makedirs(f'{outputspath}/calibration_output_graphs_{pop_size}_{results_folder.name}')

    graph_location = path

    # create_pickles_locally(results_folder)

    # ============================================HELPER FUNCTIONS... =================================================
    def get_modules_maternal_complication_dataframes(module):
        complications_df = extract_results(
            results_folder,
            module=f"tlo.methods.{module}",
            key="maternal_complication",
            custom_generate_series=(
                lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'type'])['person'].count()),
            do_scaling=False
        )

        return complications_df

    #  COMPLICATION DATA FRAMES....
    an_comps = get_modules_maternal_complication_dataframes('pregnancy_supervisor')
    la_comps = get_modules_maternal_complication_dataframes('labour')
    pn_comps = get_modules_maternal_complication_dataframes('postnatal_supervisor')

    # ============================================  DENOMINATORS... ===================================================
    # ---------------------------------------------Total_pregnancies...------------------------------------------------
    pregnancy_poll_results = extract_results(
        results_folder,
        module="tlo.methods.contraception",
        key="pregnancy",
        custom_generate_series=(
            lambda df: df.assign(year=pd.to_datetime(df['date']).dt.year).groupby(['year'])['year'].count()
        ))

    preg_data = analysis_utility_functions.get_mean_and_quants(pregnancy_poll_results, sim_years)

    analysis_utility_functions.simple_line_chart_with_ci(
        sim_years, preg_data, 'Pregnancies (mean)', 'Mean number of pregnancies', 'pregnancies', graph_location)

    # -----------------------------------------------------Total births...---------------------------------------------
    births_results = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="on_birth",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()
        ),
     )

    births_results_exc_2010 = extract_results(
                results_folder,
                module="tlo.methods.demography",
                key="on_birth",
                custom_generate_series=(
                    lambda df:
                    df.loc[(df['mother'] != -1)].assign(year=df['date'].dt.year).groupby(['year'])['year'].count()))

    birth_data = analysis_utility_functions.get_mean_and_quants(births_results, sim_years)

    analysis_utility_functions.simple_line_chart_with_ci(
        sim_years, birth_data, 'Births (mean)', 'Mean number of Births per Year', 'births', graph_location)

    # todo: some testing looking at live births vs total births...

    # -------------------------------------------------Completed pregnancies...----------------------------------------
    ectopic_mean_numbers_per_year = analysis_utility_functions.get_mean_and_quants_from_str_df(
        an_comps, 'ectopic_unruptured', sim_years)[0]

    ia_mean_numbers_per_year = analysis_utility_functions.get_mean_and_quants_from_str_df(
        an_comps, 'induced_abortion', sim_years)[0]

    sa_mean_numbers_per_year = analysis_utility_functions.get_mean_and_quants_from_str_df(
        an_comps, 'spontaneous_abortion', sim_years)[0]

    an_stillbirth_results = extract_results(
        results_folder,
        module="tlo.methods.pregnancy_supervisor",
        key="antenatal_stillbirth",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()
        ),
    )

    ip_stillbirth_results = extract_results(
        results_folder,
        module="tlo.methods.labour",
        key="intrapartum_stillbirth",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()
        ),
    )

    an_still_birth_data = analysis_utility_functions.get_mean_and_quants(an_stillbirth_results, sim_years)
    ip_still_birth_data = analysis_utility_functions.get_mean_and_quants(ip_stillbirth_results, sim_years)

    total_completed_pregnancies_per_year = [a + b + c + d + e for a, b, c, d, e in zip(birth_data[0],
                                                                                       ectopic_mean_numbers_per_year,
                                                                                       ia_mean_numbers_per_year,
                                                                                       sa_mean_numbers_per_year,
                                                                                       an_still_birth_data[0])]

    # ================================== PROPORTION OF PREGNANCIES ENDING IN BIRTH... ================================
    total_pregnancy_losses = [a + b + c + d + e for a, b, c, d, e in zip(ectopic_mean_numbers_per_year,
                                                                         ia_mean_numbers_per_year,
                                                                         sa_mean_numbers_per_year,
                                                                         an_still_birth_data[0],
                                                                         ip_still_birth_data[0])]

    prop_lost_pregnancies = [(x/y) * 100 for x, y in zip(total_pregnancy_losses, preg_data[0])]

    analysis_utility_functions.simple_bar_chart(
        prop_lost_pregnancies, 'Year', '% of Total Pregnancies',
        'Proportion of total pregnancies ending in pregnancy loss', 'preg_loss_proportion', sim_years,
        graph_location)

    # todo IP stillbirths need to be added to total births and shouldnt count as a pregnancy loss

    # ========================================== INTERVENTION COVERAGE... =============================================
    # 1.) Antenatal Care... # TODO: THIS COULD CERTAINLY BE SIMPLIFIED
    # Mean proportion of women (across draws) who have given birth that have attended ANC1, ANC4+ and ANC8+ per year...
    results = extract_results(
        results_folder,
        module="tlo.methods.care_of_women_during_pregnancy",
        key="anc_count_on_birth",
        custom_generate_series=(
            lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'total_anc'])['person_id'].count()),
        do_scaling=False
    )

    anc_count_df = pd.DataFrame(columns=sim_years, index=[0, 1, 2, 3, 4, 5, 6, 7, 8])

    # get yearly outputs
    for year in sim_years:
        for row in anc_count_df.index:
            if row in results.loc[year].index:
                x = results.loc[year, row]
                mean = x.mean()
                lq = x.quantile(0.025)
                uq = x.quantile(0.925)
                anc_count_df.at[row, year] = [mean, lq, uq]
            else:
                anc_count_df.at[row, year] = [0, 0, 0]

    yearly_anc1_rates = list()
    anc1_lqs = list()
    anc1_uqs = list()
    yearly_anc4_rates = list()
    anc4_lqs = list()
    anc4_uqs = list()
    yearly_anc8_rates = list()

    for year in sim_years:
        anc_total = 0
        four_or_more_visits = 0
        eight_or_more_visits = 0

        for row in anc_count_df[year]:
            anc_total += row[0]

        yearly_anc1_rates.append(100 - ((anc_count_df.at[0, year][0] / anc_total) * 100))
        anc1_lqs.append(100 - ((anc_count_df.at[0, year][1] / anc_total) * 100))
        anc1_uqs.append(100 - ((anc_count_df.at[0, year][2] / anc_total) * 100))

        four_or_more_visits_slice = anc_count_df.loc[anc_count_df.index > 3]
        f_lqs = 0
        f_uqs = 0
        for row in four_or_more_visits_slice[year]:
            four_or_more_visits += row[0]
            f_lqs += row[1]
            f_uqs += row[2]

        yearly_anc4_rates.append((four_or_more_visits / anc_total) * 100)
        anc4_lqs.append((f_lqs / anc_total) * 100)
        anc4_uqs.append((f_uqs / anc_total) * 100)

        eight_or_more_visits_slice = anc_count_df.loc[anc_count_df.index > 7]
        for row in eight_or_more_visits_slice[year]:
            eight_or_more_visits += row[0]

        yearly_anc8_rates.append((eight_or_more_visits / anc_total) * 100)

    target_anc1_dict = {'double': True,
                        'first': {'year': 2010, 'value': 94, 'label': 'DHS 2010', 'ci': 0},
                        'second': {'year': 2015, 'value': 95, 'label': 'DHS 2015', 'ci': 0}}
    target_anc4_dict = {'double': True,
                        'first': {'year': 2010, 'value': 45.5, 'label': 'DHS 2010', 'ci': 0},
                        'second': {'year': 2015, 'value': 51, 'label': 'DHS 2015', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, yearly_anc1_rates, anc1_lqs, anc1_uqs, target_anc1_dict, '% of total births',
        'Proportion of women attending >= 1 ANC contact per year', graph_location, 'anc_prop_anc1')

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, yearly_anc4_rates, anc4_lqs, anc4_uqs, target_anc4_dict, '% total births',
        'Proportion of women attending >= 4 ANC contact per year', graph_location, 'anc_prop_anc4')

    plt.plot(sim_years, yearly_anc1_rates,  label="anc1", color='palevioletred')
    plt.plot(sim_years, yearly_anc4_rates,   label="anc4+", color='crimson')
    plt.plot(sim_years, yearly_anc8_rates,  label="anc8+", color='pink')
    plt.xlabel('Year')
    plt.ylabel('% total births')
    plt.title('Proportion of women attending ANC1, AN4, ANC8 ')
    plt.legend()
    plt.savefig(f'{graph_location}/anc_coverage.png')
    plt.show()

    anc_count_df = anc_count_df.drop([0])
    for year in sim_years:
        total_per_year = 0
        for row in anc_count_df[year]:
            total_per_year += row[0]

        if total_per_year != 0:
            for index in anc_count_df.index:
                anc_count_df.at[index, year] = (anc_count_df.at[index, year][0]/total_per_year) * 100

    labels = sim_years
    width = 0.35       # the width of the bars: can also be len(x) sequence
    fig, ax = plt.subplots()
    ax.bar(labels, anc_count_df.loc[8], width, label=8, bottom=anc_count_df.loc[1] + anc_count_df.loc[2] +
           anc_count_df.loc[3] + anc_count_df.loc[4] + anc_count_df.loc[5] + anc_count_df.loc[6] + anc_count_df.loc[7])
    ax.bar(labels, anc_count_df.loc[7], width, label=7, bottom=anc_count_df.loc[1] + anc_count_df.loc[2] +
           anc_count_df.loc[3] + anc_count_df.loc[4] + anc_count_df.loc[5] + anc_count_df.loc[6])
    ax.bar(labels, anc_count_df.loc[6], width, label=6, bottom=anc_count_df.loc[1] + anc_count_df.loc[2] +
           anc_count_df.loc[3] + anc_count_df.loc[4] + anc_count_df.loc[5])
    ax.bar(labels, anc_count_df.loc[5], width, label=5, bottom=anc_count_df.loc[1] + anc_count_df.loc[2] +
           anc_count_df.loc[3] + anc_count_df.loc[4])
    ax.bar(labels, anc_count_df.loc[4], width, label=4, bottom=anc_count_df.loc[1] + anc_count_df.loc[2] +
           anc_count_df.loc[3])
    ax.bar(labels, anc_count_df.loc[3], width, label=3, bottom=anc_count_df.loc[1] + anc_count_df.loc[2])
    ax.bar(labels, anc_count_df.loc[2], width, label=2, bottom=anc_count_df.loc[1])
    ax.bar(labels, anc_count_df.loc[1], width, label=1,)
    ax.set_ylabel('% of total yearly visits')
    ax.set_title('Number of ANC visits at birth per year')
    ax.legend()
    plt.savefig(f'{graph_location}/anc_total_visits.png')
    plt.show()

    # Mean proportion of women who attended at least one ANC visit that attended at < 4, 4-5, 6-7 and > 8 months
    # gestation...
    anc_ga_first_visit = extract_results(
        results_folder,
        module="tlo.methods.care_of_women_during_pregnancy",
        key="anc_count_on_birth",
        custom_generate_series=(
            lambda df_: df_.assign(year=df_['date'].dt.year) .groupby(
                ['year', 'total_anc', 'ga_anc_one'])['person_id'].count()),
        do_scaling=False
    )

    anc_before_four_months = list()
    early_anc_4_list = list()
    late_anc_4_list = list()
    anc_before_four_five_months = list()
    anc_before_six_seven_months = list()
    anc_before_eight_plus_months = list()

    def get_means(df):
        mean_list = 0
        for index in df.index:
            mean_list += df.loc[index].mean()
        return mean_list

    for year in sim_years:
        year_df = anc_ga_first_visit.loc[year]
        total_women_that_year = 0
        for index in year_df.index:
            total_women_that_year += year_df.loc[index].mean()

        anc1 = get_means(year_df.loc[0, 0:len(year_df.columns)])

        total_women_anc = total_women_that_year - anc1

        early_anc = year_df.loc[(slice(1, 8), slice(0, 13)), 0:len(year_df.columns)]
        early_anc4 = year_df.loc[(slice(4, 8), slice(0, 17)), 0:len(year_df.columns)]
        late_anc4 = year_df.loc[(slice(4, 8), slice(18, 50)), 0:len(year_df.columns)]
        four_to_five = year_df.loc[(slice(1, 8), slice(14, 22)), 0:len(year_df.columns)]
        six_to_seven = year_df.loc[(slice(1, 8), slice(23, 31)), 0:len(year_df.columns)]
        eight_plus = year_df.loc[(slice(1, 8), slice(32, 50)), 0:len(year_df.columns)]

        sum_means_early = get_means(early_anc)
        sum_means_early_anc4 = get_means(early_anc4)
        sum_means_late_anc4 = get_means(late_anc4)
        sum_four = get_means(four_to_five)
        sum_six = get_means(six_to_seven)
        sum_eight = get_means(eight_plus)

        early_anc_4_list.append((sum_means_early_anc4 / total_women_anc) * 100)
        late_anc_4_list.append((sum_means_late_anc4 / total_women_anc) * 100)
        anc_before_four_months.append((sum_means_early/total_women_anc) * 100)
        anc_before_four_five_months.append((sum_four/total_women_anc) * 100)
        anc_before_six_seven_months.append((sum_six/total_women_anc) * 100)
        anc_before_eight_plus_months.append((sum_eight/total_women_anc) * 100)

    labels = sim_years
    width = 0.35       # the width of the bars: can also be len(x) sequence
    fig, ax = plt.subplots()
    ax.bar(labels, anc_before_eight_plus_months, width, label='>8m',
           bottom=[x + y + z for x, y, z in zip(anc_before_four_months, anc_before_four_five_months,
                                                anc_before_six_seven_months)])
    ax.bar(labels, anc_before_six_seven_months, width, label='6-7m',
           bottom=[x + y for x, y in zip(anc_before_four_months, anc_before_four_five_months)])
    ax.bar(labels, anc_before_four_five_months, width, label='4-5m',
           bottom=anc_before_four_months)
    ax.bar(labels, anc_before_four_months, width, label='<4m')
    ax.set_ylabel('% of ANC1 visits by gestational age')
    ax.set_title('Gestational age at first ANC visit by Year')
    ax.legend()
    plt.savefig(f'{graph_location}/anc_ga_first_visit_update.png')
    plt.show()

    target_rate_eanc4 = list()
    for year in sim_years:
        if year < 2015:
            target_rate_eanc4.append(24.5)
        else:
            target_rate_eanc4.append(36.7)

    analysis_utility_functions.simple_line_chart_with_target(
        sim_years, early_anc_4_list, target_rate_eanc4, '% total deliveries',
        'Proportion of women attending attending ANC4+ with first visit early', 'anc_prop_early_anc4', graph_location)

    total_anc4 = [x + y for x, y in zip(late_anc_4_list, early_anc_4_list)]
    prop_early = [(x / y) * 100 for x, y in zip(early_anc_4_list, total_anc4)]
    prop_late = [(x / y) * 100 for x, y in zip(late_anc_4_list, total_anc4)]

    labels = sim_years
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(labels, prop_early, width, label='Early ANC4+', bottom=prop_late)
    ax.bar(labels, prop_late, width, label='Late ANC4+')
    ax.set_ylabel('% of women attending ANC4+')
    ax.set_title('Early vs Late initation of ANC4+')
    ax.legend()
    plt.savefig(f'{graph_location}/early_late_ANC4+.png')
    plt.show()

    # TODO: quartiles, median month ANC1
    # todo: target rates

    # 2.) Facility delivery
    # Total FDR per year (denominator - total births)

    # WE EXCLUDE 2010 ADDED BIRTHS HERE TODO: ADD DOCUMENTATION
    birth_data_ex2010 = analysis_utility_functions.get_mean_and_quants(births_results_exc_2010, sim_years)

    deliver_setting_results = extract_results(
            results_folder,
            module="tlo.methods.labour",
            key="delivery_setting_and_mode",
            custom_generate_series=(
                lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'facility_type'])['mother'].count()),
            do_scaling=False
        )

    hb_data = analysis_utility_functions.get_mean_and_quants_from_str_df(
        deliver_setting_results, 'home_birth', sim_years)

    hp_data = analysis_utility_functions.get_mean_and_quants_from_str_df(
        deliver_setting_results, 'hospital', sim_years)

    hc_data = analysis_utility_functions.get_mean_and_quants_from_str_df(
        deliver_setting_results, 'health_centre', sim_years)

    all_deliveries = [x + y + z for x, y, z in zip(hb_data[0], hc_data[0], hp_data[0])]

    home_birth_rate = [(x / y) * 100 for x, y in zip(hb_data[0], all_deliveries)]

    health_centre_rate = [(x / y) * 100 for x, y in zip(hc_data[0], all_deliveries)]
    health_centre_lq = [(x / y) * 100 for x, y in zip(hc_data[1], all_deliveries)]
    health_centre_uq = [(x / y) * 100 for x, y in zip(hc_data[2], all_deliveries)]

    hospital_rate = [(x / y) * 100 for x, y in zip(hp_data[0], all_deliveries)]
    hospital_lq = [(x / y) * 100 for x, y in zip(hp_data[1], all_deliveries)]
    hospital_uq = [(x / y) * 100 for x, y in zip(hp_data[2], all_deliveries)]

    total_fd_rate = [x + y for x, y in zip(health_centre_rate, hospital_rate)]
    fd_lqs = [x + y for x, y in zip(health_centre_lq, hospital_lq)]
    fd_uqs = [x + y for x, y in zip(health_centre_uq, hospital_uq)]

    target_fd_dict = {'double': True,
                      'first': {'year': 2010, 'value': 73, 'label': 'DHS 2010', 'ci': 0},
                      'second': {'year': 2015, 'value': 91, 'label': 'DHS 2015', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, total_fd_rate, fd_lqs, fd_uqs, target_fd_dict, '% of total births',
        'Proportion of Women Delivering in a Health Facility per Year', graph_location, 'sba_prop_facility_deliv')

    labels = sim_years
    width = 0.35       # the width of the bars: can also be len(x) sequence
    fig, ax = plt.subplots()

    ax.bar(labels, hospital_rate, width, label='Hospital Birth',
           bottom=[x + y for x, y in zip(home_birth_rate, health_centre_rate)])
    ax.bar(labels, health_centre_rate, width, label='Health Centre Birth',
           bottom=home_birth_rate)
    ax.bar(labels, home_birth_rate, width, label='Home Birth')
    ax.set_ylabel('% of Births by Location')
    ax.set_title('Proportion of Total Births by Location of Delivery')
    ax.legend()
    plt.savefig(f'{graph_location}/sba_delivery_location.png')
    plt.show()

    # 3.) Postnatal Care
    pnc_results_maternal = extract_results(
        results_folder,
        module="tlo.methods.postnatal_supervisor",
        key="total_mat_pnc_visits",
        custom_generate_series=(
            lambda df: df.loc[df['visits'] > 0].assign(year=df['date'].dt.year).groupby(['year'])['mother'].count()),
        do_scaling=False
    )
    pnc_results_newborn = extract_results(
        results_folder,
        module="tlo.methods.postnatal_supervisor",
        key="total_neo_pnc_visits",
        custom_generate_series=(
            lambda df: df.loc[df['visits'] > 0].assign(year=df['date'].dt.year).groupby(['year'])['child'].count()),
        do_scaling=False
    )
    pn_mat_data = analysis_utility_functions.get_mean_and_quants(pnc_results_maternal, sim_years)
    pn_neo_data = analysis_utility_functions.get_mean_and_quants(pnc_results_newborn, sim_years)

    pnc_1_plus_rate_mat = [(x / y) * 100 for x, y in zip(pn_mat_data[0], birth_data_ex2010[0])]
    pnc_mat_lqs = [(x / y) * 100 for x, y in zip(pn_mat_data[1], birth_data_ex2010[0])]
    pnc_mat_uqs = [(x / y) * 100 for x, y in zip(pn_mat_data[2], birth_data_ex2010[0])]

    pnc1_plus_rate_neo = [(x / y) * 100 for x, y in zip(pn_neo_data[0], birth_data_ex2010[0])]
    pnc_neo_lqs = [(x / y) * 100 for x, y in zip(pn_neo_data[1], birth_data_ex2010[0])]
    pnc_neo_uqs = [(x / y) * 100 for x, y in zip(pn_neo_data[2], birth_data_ex2010[0])]

    target_mpnc_dict = {'double': True,
                        'first': {'year': 2010, 'value': 50, 'label': 'DHS 2010', 'ci': 0},
                        'second': {'year': 2015, 'value': 48, 'label': 'DHS 2015', 'ci': 0}}

    target_npnc_dict = {'double': False,
                        'first': {'year': 2015, 'value': 60, 'label': 'DHS 2015', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, pnc_1_plus_rate_mat, pnc_mat_lqs, pnc_mat_uqs, target_mpnc_dict, '% of total births',
        'Proportion of Women post-delivery attending PNC per year', graph_location, 'pnc_mat')

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, pnc1_plus_rate_neo, pnc_neo_lqs, pnc_neo_uqs, target_npnc_dict, '% of total births',
        'Proportion of Neonates per year attending PNC', graph_location, 'pnc_neo')

    def get_early_late_pnc_split(module, target, file_name):
        pnc = extract_results(
            results_folder,
            module=f"tlo.methods.{module}",
            key="postnatal_check",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'timing',
                                                                       'visit_number'])['person_id'].count()
            ),
        )
        early_rate = list()
        late_rate = list()

        for year in sim_years:
            total = pnc.loc[year, 'early', 0].mean() + pnc.loc[year, 'late', 0].mean() + pnc.loc[year, 'none', 0].mean()
            early = (pnc.loc[year, 'early', 0].mean() / total) * 100
            late = ((pnc.loc[year, 'late', 0].mean() + pnc.loc[year, 'none', 0].mean()) / total) * 100
            early_rate.append(early)
            late_rate.append(late)

        labels = sim_years
        width = 0.35       # the width of the bars: can also be len(x) sequence
        fig, ax = plt.subplots()
        ax.bar(labels, early_rate, width, label='< 48hrs')
        ax.bar(labels, late_rate, width, bottom=early_rate, label='>48hrs')
        ax.set_ylabel('% of PNC 1 visits')
        ax.set_title(f'Proportion of {target} PNC1 Visits occurring pre/post 48hrs Postnatal')
        ax.legend()
        plt.savefig(f'{graph_location}/{file_name}.png')
        plt.show()

    get_early_late_pnc_split('labour', 'Maternal', 'pnc_maternal_early')
    get_early_late_pnc_split('newborn_outcomes', 'Neonatal', 'pnc_neonatal_early')

    # ========================================== COMPLICATION/DISEASE RATES.... =======================================
    # ---------------------------------------- Twinning Rate... -------------------------------------------------------
    # % Twin births/Total Births per year
    twins_results = extract_results(
        results_folder,
        module="tlo.methods.newborn_outcomes",
        key="twin_birth",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()
        ),
    )

    twin_data = analysis_utility_functions.get_mean_and_quants(twins_results, sim_years)

    total_deliveries = [x - y for x, y in zip(birth_data_ex2010[0], twin_data[0])]

    final_twining_rate = [(x / y) * 100 for x, y in zip(twin_data[0], total_deliveries)]
    lq_rate = [(x / y) * 100 for x, y in zip(twin_data[1], total_deliveries)]
    uq_rate = [(x / y) * 100 for x, y in zip(twin_data[2], total_deliveries)]

    target_twin_dict = {'double': False,
                        'first': {'year': 2010, 'value': 3.9, 'label': 'DHS 2010', 'ci': 0}}

    # todo: ADD additional HIV report data?
    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, final_twining_rate, lq_rate, uq_rate, target_twin_dict, 'Rate per 100 pregnancies',
        'Yearly trends for Twin Births', graph_location, 'twin_rate')

    # ---------------------------------------- Early Pregnancy Loss... ------------------------------------------------
    # Ectopic pregnancies/Total pregnancies
    ectopic_data = analysis_utility_functions.get_comp_mean_and_rate(
        'ectopic_unruptured', preg_data[0], an_comps, 1000, sim_years)

    target_ect_dict = {'double': True,
                       'first': {'year': 2010, 'value': 4.9, 'label': 'GBD 2010', 'ci': 0},
                       'second': {'year': 2015, 'value': 3.6, 'label': 'GBD 2015', 'ci': 0}}

    # todo: if were using GBD data why cant we have this rate yearly?

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, ectopic_data[0], ectopic_data[1], ectopic_data[2], target_ect_dict, 'Rate per 100 pregnancies',
        'Yearly trends for Ectopic Pregnancy', graph_location, 'ectopic_rate')

    # Ruptured ectopic pregnancies / Total Pregnancies
    mean_unrup_ectopics = analysis_utility_functions.get_mean_and_quants_from_str_df(
        an_comps, 'ectopic_unruptured', sim_years)[0]

    proportion_of_ectopics_that_rupture_per_year = analysis_utility_functions.get_comp_mean_and_rate(
        'ectopic_ruptured', mean_unrup_ectopics, an_comps, 100, sim_years)[0]

    proportion_of_ectopics_that_rupture_per_year = [100 if i > 100 else i for i in
                                                    proportion_of_ectopics_that_rupture_per_year]

    target_rate_rup = list()
    for year in sim_years:
        target_rate_rup.append(92)

    analysis_utility_functions.simple_line_chart_with_target(
        sim_years, proportion_of_ectopics_that_rupture_per_year, target_rate_rup, 'Proportion of all Ectopic Cases',
        'Proportion of Ectopic Pregnancies Leading to Rupture per year', 'ectopic_rupture_prop', graph_location)

    # Spontaneous Abortions....
    spotaneous_abortion_data = analysis_utility_functions.get_comp_mean_and_rate(
        'spontaneous_abortion', total_completed_pregnancies_per_year, an_comps, 1000, sim_years)

    target_sa_dict = {'double': False,
                      'first': {'year': 2016, 'value': 130, 'label': 'Dellicour et al.', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, spotaneous_abortion_data[0], spotaneous_abortion_data[1], spotaneous_abortion_data[2],
        target_sa_dict, 'Rate per 1000 completed pregnancies', 'Yearly rate of Miscarriage', graph_location,
        'miscarriage_rate' )

    # Complicated SA / Total SA
    mean_complicated_sa = analysis_utility_functions.get_mean_and_quants_from_str_df(
        an_comps, 'spontaneous_abortion', sim_years)[0]

    proportion_of_complicated_sa_per_year = analysis_utility_functions.get_comp_mean_and_rate(
        'complicated_spontaneous_abortion', mean_complicated_sa, an_comps, 100, sim_years)[0]

    analysis_utility_functions.simple_bar_chart(
        proportion_of_complicated_sa_per_year, 'Year', '% of Total Miscarriages',
        'Proportion of miscarriages leading to complications', 'miscarriage_prop_complicated', sim_years,
        graph_location)

    # Induced Abortions...
    id_data = analysis_utility_functions.get_comp_mean_and_rate(
        'induced_abortion', total_completed_pregnancies_per_year, an_comps, 1000, sim_years)

    target_ia_dict = {'double': True,
                      'first': {'year': 2010, 'value': 86, 'label': 'Levandowski et al.', 'ci': 0},
                      'second': {'year': 2015, 'value': 159, 'label': 'Polis et al.', 'ci': 0},
                      }

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, id_data[0], id_data[1], id_data[2], target_ia_dict, 'Rate per 1000 completed pregnancies',
        'Yearly rate of Induced Abortion',  graph_location, 'abortion_rate')

    # Complicated IA / Total IA
    mean_complicated_ia = analysis_utility_functions.get_mean_and_quants_from_str_df(
        an_comps, 'induced_abortion', sim_years)[0]

    proportion_of_complicated_ia_per_year = analysis_utility_functions.get_comp_mean_and_rate(
        'complicated_induced_abortion', mean_complicated_ia, an_comps, 100, sim_years)[0]

    analysis_utility_functions.simple_bar_chart(
        proportion_of_complicated_ia_per_year, 'Year', '% of Total Abortions',
        'Proportion of Abortions leading to complications', 'abortion_prop_complicated', sim_years, graph_location)

    # --------------------------------------------------- Syphilis Rate... --------------------------------------------
    syphilis_data = analysis_utility_functions.get_comp_mean_and_rate(
        'syphilis', total_completed_pregnancies_per_year, an_comps, 1000, sim_years)

    target_syph_dict = {'double': False,
                        'first': {'year': 2018, 'value': 20, 'label': 'HIV rpt data', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, syphilis_data[0], syphilis_data[1],  syphilis_data[2], target_syph_dict,
        'Rate per 1000 completed pregnancies', 'Yearly rate of Syphilis', graph_location, 'syphilis_rate')

    # ------------------------------------------------ Gestational Diabetes... ----------------------------------------
    gdm_data = analysis_utility_functions.get_comp_mean_and_rate(
        'gest_diab', total_completed_pregnancies_per_year, an_comps, 1000, sim_years)

    target_gdm_dict = {'double': False,
                       'first': {'year': 2019, 'value': 16, 'label': 'Phiri et al.', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, gdm_data[0], gdm_data[1], gdm_data[2], target_gdm_dict,'Rate per 1000 completed pregnancies',
        'Yearly rate of Gestational Diabetes', graph_location, 'gest_diab_rate', )

    # ------------------------------------------------ PROM... --------------------------------------------------------
    prom_data = analysis_utility_functions.get_comp_mean_and_rate('PROM', birth_data_ex2010[0], an_comps, 1000,
                                                                  sim_years)

    target_prm_dict = {'double': False,
                       'first': {'year': 2020, 'value': 27, 'label': 'Onwughara et al.', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, prom_data[0], prom_data[1], prom_data[2], target_prm_dict, 'Rate per 1000 births',
        'Yearly rate of PROM', graph_location, 'prom_rate')

    # ---------------------------------------------- Anaemia... -------------------------------------------------------
    # Total prevalence of Anaemia at birth (total cases of anaemia at birth/ total births per year) and by severity
    anaemia_results = extract_results(
        results_folder,
        module="tlo.methods.pregnancy_supervisor",
        key="anaemia_on_birth",
        custom_generate_series=(
            lambda df: df.loc[df['anaemia_status'] != 'none'].assign(year=df['date'].dt.year).groupby(['year'])[
                'year'].count()))

    pnc_anaemia = extract_results(
        results_folder,
        module="tlo.methods.postnatal_supervisor",
        key="total_mat_pnc_visits",
        custom_generate_series=(
            lambda df: df.loc[df['anaemia'] != 'none'].assign(year=df['date'].dt.year).groupby(['year'])[
                'year'].count()))

    preg_an_severity = extract_results(
        results_folder,
        module="tlo.methods.pregnancy_supervisor",
        key="anaemia_on_birth",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'anaemia_status'])['year'].count()
        ),
    )

    pn_pn_severity = extract_results(
        results_folder,
        module="tlo.methods.postnatal_supervisor",
        key="total_mat_pnc_visits",
        custom_generate_series=(
            lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'anaemia'])['mother'].count()),
        do_scaling=False
    )

    an_data = analysis_utility_functions.get_mean_and_quants(anaemia_results, sim_years)
    pn_data = analysis_utility_functions.get_mean_and_quants(pnc_anaemia, sim_years)

    def get_anaemia_graphs(data, timing, severity_df):
        target_an_dict = {'double': True,
                          'first': {'year': 2010, 'value': 37.5, 'label': 'DHS 2010', 'ci': 0},
                          'second': {'year': 2015, 'value': 45.1, 'label': 'DHS 2015', 'ci': 0},
                          }

        prev = [(x / y) * 100 for x, y in zip(data[0], birth_data_ex2010[0])]
        lq = [(x / y) * 100 for x, y in zip(data[1], birth_data_ex2010[0])]
        uq = [(x / y) * 100 for x, y in zip(data[2], birth_data_ex2010[0])]

        analysis_utility_functions.line_graph_with_ci_and_target_rate(
            sim_years, prev, lq, uq, target_an_dict,
            'Prevalence at birth', f'Yearly prevalence of Anaemia (all severity) at {timing}', graph_location,
            f'anaemia_prev_{timing}')

        # todo: should maybe be total postnatal women still alive as opposed to births as will inflate

        mild_anaemia_at_birth = analysis_utility_functions.get_mean_and_quants_from_str_df(
            severity_df, 'mild', sim_years)[0]
        moderate_anaemia_at_birth = analysis_utility_functions.get_mean_and_quants_from_str_df(
            severity_df, 'moderate', sim_years)[0]
        severe_anaemia_at_birth = analysis_utility_functions.get_mean_and_quants_from_str_df(
            severity_df, 'severe', sim_years)[0]

        prevalence_of_mild_anaemia_per_year = [(x/y) * 100 for x, y in zip(
            mild_anaemia_at_birth, birth_data_ex2010[0])]
        prevalence_of_mod_anaemia_per_year = [(x/y) * 100 for x, y in zip(
            moderate_anaemia_at_birth, birth_data_ex2010[0])]
        prevalence_of_sev_anaemia_per_year = [(x/y) * 100 for x, y in zip(
            severe_anaemia_at_birth, birth_data_ex2010[0])]

        plt.plot(sim_years, prevalence_of_mild_anaemia_per_year, label="mild")
        plt.plot(sim_years, prevalence_of_mod_anaemia_per_year, label="moderate")
        plt.plot(sim_years, prevalence_of_sev_anaemia_per_year, label="severe")
        plt.xlabel('Year')
        plt.ylabel(f'Prevalence at {timing}')
        plt.title(f'Yearly trends for prevalence of anaemia by severity at {timing}')
        plt.legend()
        plt.savefig(f'{graph_location}/anaemia_by_severity_{timing}.png')
        plt.show()

    get_anaemia_graphs(an_data, 'delivery', preg_an_severity)
    get_anaemia_graphs(pn_data, 'postnatal', pn_pn_severity)

    # ------------------------------------------- Hypertensive disorders -----------------------------------------------
    gh_data = analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
        'mild_gest_htn', birth_data_ex2010[0], 1000, [an_comps, pn_comps], sim_years)
    sgh_data = analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
        'severe_gest_htn', birth_data_ex2010[0], 1000, [an_comps, la_comps, pn_comps], sim_years)
    mpe_data = analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
        'mild_pre_eclamp', birth_data_ex2010[0], 1000, [an_comps, pn_comps], sim_years)
    spe_data = analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
        'severe_pre_eclamp', birth_data_ex2010[0], 1000, [an_comps, la_comps, pn_comps], sim_years)
    ec_data = analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
        'eclampsia', birth_data_ex2010[0], 1000, [an_comps, la_comps, pn_comps], sim_years)

    target_gh_dict = {'double': False,
                      'first': {'year': 2019, 'value': 36.8, 'label': 'Noubiap et al.', 'ci': 0},
                      }

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, gh_data[0], gh_data[1], gh_data[2], target_gh_dict, 'Rate per 1000 births',
        'Rate of Gestational Hypertension per Year',  graph_location, 'gest_htn_rate',)

    target_sgh_dict = {'double': False,
                       'first': {'year': 2019, 'value': 8.1, 'label': 'Noubiap et al.', 'ci': 0},
                       }

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, sgh_data[0], sgh_data[1], sgh_data[2], target_sgh_dict, 'Rate per 1000 births',
        'Rate of Severe Gestational Hypertension per Year', graph_location, 'severe_gest_htn_rate', )

    target_mpe_dict = {'double': False,
                       'first': {'year': 2019, 'value': 44, 'label': 'Noubiap et al', 'ci': 0},
                       }

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, mpe_data[0], mpe_data[1], mpe_data[2], target_mpe_dict, 'Rate per 1000 births',
        'Rate of Mild pre-eclampsia per Year', graph_location, 'mild_pre_eclampsia_rate')

    target_spe_dict = {'double': False,
                       'first': {'year': 2019, 'value': 22, 'label': 'Noubiap et al', 'ci': 0},
                       }

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, spe_data[0], spe_data[1], spe_data[2], target_spe_dict, 'Rate per 1000 births',
        'Rate of Severe pre-eclampsia per Year', graph_location,  'severe_pre_eclampsia_rate')

    target_ec_dict = {'double': False,
                      'first': {'year': 2019, 'value': 10, 'label': 'Vousden et al.', 'ci': 0},
                      }
    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, ec_data[0], ec_data[1], ec_data[2], target_ec_dict, 'Rate per 1000 births',
        'Rate of Eclampsia per Year', graph_location, 'eclampsia_rate')

    #  ---------------------------------------------Placenta praevia... ----------------------------------------------
    pp_data = analysis_utility_functions.get_comp_mean_and_rate(
        'placenta_praevia', preg_data[0], an_comps, 1000, sim_years)

    target_pp_dict = {'double': False,
                      'first': {'year': 2017, 'value': 5.67, 'label': 'Senkoro et al.', 'ci': 0},
                      }
    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, pp_data[0], pp_data[1], pp_data[2], target_pp_dict, 'Rate per 1000 pregnancies',
        'Rate of Placenta Praevia per Year', graph_location, 'praevia_rate')

    #  ---------------------------------------------Placental abruption... --------------------------------------------
    pa_data = analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
        'placental_abruption', birth_data_ex2010[0], 1000, [an_comps, la_comps], sim_years)

    target_pa_dict = {'double': False,
                      'first': {'year': 2015, 'value': 3, 'label': 'Macheku et al.', 'ci': 0},
                      }

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, pa_data[0], pa_data[1], pa_data[2], target_pa_dict, 'Rate per 1000 births',
        'Rate of Placental Abruption per Year', graph_location, 'abruption_rate')

    # --------------------------------------------- Antepartum Haemorrhage... -----------------------------------------
    # Rate of APH/total births (antenatal and labour)
    mm_aph_data = analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
        'mild_mod_antepartum_haemorrhage', birth_data_ex2010[0], 1000, [an_comps, la_comps], sim_years)

    s_aph_data = analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
        'severe_antepartum_haemorrhage', birth_data_ex2010[0], 1000, [an_comps, la_comps], sim_years)

    total_aph_rates = [x + y for x, y in zip(mm_aph_data[0], s_aph_data[0])]
    aph_lqs = [x + y for x, y in zip(mm_aph_data[1], s_aph_data[1])]
    aph_uqs = [x + y for x, y in zip(mm_aph_data[2], s_aph_data[2])]

    target_aph_dict = {'double': False,
                       'first': {'year': 2015, 'value': 6.4, 'label': 'BEmONC.', 'ci': 0},
                       }

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, total_aph_rates, aph_lqs, aph_uqs, target_aph_dict, 'Rate per 1000 births',
        'Rate of Antepartum Haemorrhage per Year', graph_location, 'aph_rate')

    # --------------------------------------------- Preterm birth ... ------------------------------------------------
    early_ptl_data = analysis_utility_functions.get_comp_mean_and_rate(
        'early_preterm_labour', birth_data_ex2010[0], la_comps, 100, sim_years)
    late_ptl_data = analysis_utility_functions.get_comp_mean_and_rate(
        'late_preterm_labour', birth_data_ex2010[0], la_comps, 100, sim_years)

    target_ptl_dict = {'double': True,
                       'first': {'year': 2012, 'value': 19.8, 'label': 'Antony et al.', 'ci': 0},
                       'second': {'year': 2014, 'value': 10, 'label': 'Chawanpaiboon et al.', 'ci': (14.3-7.4)/2},
                       }

    total_ptl_rates = [x + y for x, y in zip(early_ptl_data[0], late_ptl_data[0])]
    ptl_lqs = [x + y for x, y in zip(early_ptl_data[1], late_ptl_data[1])]
    ltl_uqs = [x + y for x, y in zip(early_ptl_data[2], late_ptl_data[2])]

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, total_ptl_rates, ptl_lqs, ltl_uqs, target_ptl_dict, 'Proportion of total births',
        'Preterm birth rate', graph_location, 'ptb_rate')

    prop_early = [(x / y) * 100 for x, y in zip(early_ptl_data[0], total_ptl_rates)]
    prop_late = [(x / y) * 100 for x, y in zip(late_ptl_data[0], total_ptl_rates)]

    labels = sim_years
    width = 0.35       # the width of the bars: can also be len(x) sequence
    fig, ax = plt.subplots()

    ax.bar(labels, prop_early, width, label='Early Preterm',
           bottom=prop_late)
    ax.bar(labels, prop_late, width, label='Late Preterm')
    ax.set_ylabel('% of total Preterm Births')
    ax.set_title('Early vs Late Preterm Births')
    ax.legend()
    plt.savefig(f'{graph_location}/early_late_preterm.png')
    plt.show()

    # todo plot early and late seperated

    # --------------------------------------------- Post term birth ... -----------------------------------------------
    potl_data = analysis_utility_functions.get_comp_mean_and_rate(
        'post_term_labour', birth_data_ex2010[0], la_comps, 100, sim_years)

    target_potl_dict = {'double': False,
                        'first': {'year': 2014, 'value': 3.2, 'label': 'van den Broek et al.', 'ci': 0},
                        }

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, potl_data[0], potl_data[1], potl_data[2], target_potl_dict, 'Proportion of total births',
        'Post term birth rate', graph_location, 'potl_rate')

    # ------------------------------------------- Antenatal Stillbirth ... -------------------------------------------
    an_sbr_per_year = [(x/y) * 1000 for x, y in zip(an_still_birth_data[0], birth_data_ex2010[0])]
    an_sbr_lqs = [(x/y) * 1000 for x, y in zip(an_still_birth_data[1], birth_data_ex2010[0])]
    an_sbr_uqs = [(x/y) * 1000 for x, y in zip(an_still_birth_data[2], birth_data_ex2010[0])]

    target_ansbr_dict = {'double': True,
                         'first': {'year': 2010, 'value': 10, 'label': 'UN est.', 'ci': 0},
                         'second': {'year': 2015, 'value': 8.15, 'label': 'UN est.', 'ci': 0},
                         }

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, an_sbr_per_year, an_sbr_lqs, an_sbr_uqs, target_ansbr_dict, 'Rate per 1000 births',
        'Antenatal Stillbirth Rate per Year', graph_location, 'sbr_an')

    # ------------------------------------------------- Birth weight... -----------------------------------------------
    nb_outcomes_df = extract_results(
            results_folder,
            module="tlo.methods.newborn_outcomes",
            key="newborn_complication",
            custom_generate_series=(
                lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'type'])['newborn'].count()),
            do_scaling=False
        )

    nb_outcomes_pn_df = extract_results(
            results_folder,
            module="tlo.methods.postnatal_supervisor",
            key="newborn_complication",
            custom_generate_series=(
                lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'type'])['newborn'].count()),
            do_scaling=False
        )

    lbw_data = analysis_utility_functions.get_comp_mean_and_rate(
        'low_birth_weight', birth_data_ex2010[0], nb_outcomes_df, 100, sim_years)

    macro_data = analysis_utility_functions.get_comp_mean_and_rate(
        'macrosomia', birth_data_ex2010[0], nb_outcomes_df, 100, sim_years)

    sga_data = analysis_utility_functions.get_comp_mean_and_rate(
        'small_for_gestational_age', birth_data_ex2010[0], nb_outcomes_df, 100, sim_years)

    target_lbw_dict = {'double': True,
                       'first': {'year': 2010, 'value': 12, 'label': 'DHS 2010', 'ci': 0},
                       'second': {'year': 2015, 'value': 12, 'label': 'DHS 2015', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, lbw_data[0], lbw_data[1], lbw_data[2], target_lbw_dict, 'Proportion of total births',
        'Yearly Prevalence of Low Birth Weight', graph_location, 'neo_lbw_prev')

    target_mac_dict = {'double': False,
                       'first': {'year': 2019, 'value': 5.13, 'label': 'Ngwira et al.', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, macro_data[0], macro_data[1], macro_data[2], target_mac_dict, 'Proportion of total births',
        'Yearly Prevalence of Macrosomia', graph_location, 'neo_macrosomia_prev')

    target_sga_dict = {'double': False,
                       'first': {'year': 2010, 'value': 23.2, 'label': 'Lee et al.', 'ci': (27-19.1)/2}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, sga_data[0], sga_data[1], sga_data[2], target_sga_dict, 'Proportion of total births',
        'Yearly Prevalence of Small for Gestational Age', graph_location, 'neo_sga_prev')

    # todo: check with Ines r.e. SGA and the impact on her modules

    # --------------------------------------------- Obstructed Labour... ----------------------------------------------
    ol_data = analysis_utility_functions.get_comp_mean_and_rate(
        'obstructed_labour', birth_data_ex2010[0], la_comps, 1000, sim_years)

    target_rate_ol = list()
    for year in sim_years:
        if year < 2015:
            target_rate_ol.append(17)
        else:
            target_rate_ol.append(31)

    target_ol_dict = {'double': True,
                      'first': {'year': 2010, 'value': 17, 'label': 'BEmONC 2010', 'ci': 0},
                      'second': {'year': 2015, 'value': 31, 'label': 'BEmONC 2015', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, ol_data[0], ol_data[1], ol_data[2], target_ol_dict,'Rate per 1000 births',
        'Obstructed Labour Rate per Year', graph_location, 'ol_rate')

    # --------------------------------------------- Uterine rupture... -----------------------------------------------
    ur_data = analysis_utility_functions.get_comp_mean_and_rate(
        'uterine_rupture', birth_data_ex2010[0], la_comps, 1000, sim_years)

    target_ur_dict = {'double': True,
                      'first': {'year': 2010, 'value': 1.2, 'label': 'BEmONC 2010', 'ci': 0},
                      'second': {'year': 2015, 'value': 0.8, 'label': 'BEmONC 2015', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, ur_data[0], ur_data[1], ur_data[2], target_ur_dict, 'Rate per 1000 births',
        'Rate of Uterine Rupture per Year', graph_location, 'ur_rate')

    # ---------------------------Caesarean Section Rate & Assisted Vaginal Delivery Rate... --------------------------
    delivery_mode = extract_results(
        results_folder,
        module="tlo.methods.labour",
        key="delivery_setting_and_mode",
        custom_generate_series=(
            lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'mode'])['mother'].count()),
        do_scaling=False
    )

    cs_results = extract_results(
            results_folder,
            module="tlo.methods.labour",
            key="cs_indications",
            custom_generate_series=(
                lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'indication'])['id'].count()),
            do_scaling=False
        )

    cs_data = analysis_utility_functions.get_comp_mean_and_rate(
        'caesarean_section', birth_data_ex2010[0], delivery_mode, 100, sim_years)

    avd_data = analysis_utility_functions.get_comp_mean_and_rate(
        'instrumental', birth_data_ex2010[0], delivery_mode, 100, sim_years)

    target_cs_dict = {'double': True,
                      'first': {'year': 2010, 'value': 4.6, 'label': 'DHS 2010', 'ci': 0},
                      'second': {'year': 2015, 'value': 6, 'label': 'DHS 2015', 'ci': 0}}
    # todo: add bemonc estimates as well?

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, cs_data[0], cs_data[1], cs_data[2], target_cs_dict, 'Proportion of total births',
        'Caesarean Section Rate per Year', graph_location, 'caesarean_section_rate')

    target_avd_dict = {'double': False,
                       'first': {'year': 2017, 'value': 1, 'label': 'HIV reports.', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, avd_data[0], avd_data[1], avd_data[2], target_avd_dict, 'Proportion of total births',
        'Assisted Vaginal Delivery Rate per Year', graph_location, 'avd_rate')

    proportions_dict_cs = dict()
    total_cs_per_year = list()

    for year in sim_years:
        yearly_mean_number = list()
        causes = dict()

        for indication in ['twins', 'an_aph_pa', 'an_aph_pp', 'la_aph', 'ol', 'ol_failed_avd', 'ur', 'spe_ec', 'other',
                           'previous_scar']:
            if indication in cs_results.loc[year].index:
                mean = cs_results.loc[year, indication].mean()
                yearly_mean_number.append(mean)
                causes.update({f'{indication}': mean})
            else:
                yearly_mean_number.append(0)

        total_cs_this_year = sum(yearly_mean_number)
        total_cs_per_year.append(total_cs_this_year)

        for indication in ['twins','an_aph_pa', 'an_aph_pp', 'la_aph', 'ol', 'ol_failed_avd', 'ur', 'spe_ec', 'other',
                           'none']:
            if indication in cs_results.loc[year].index:
                causes[indication] = (causes[indication] / total_cs_this_year) * 100
            else:
                causes[indication] = 0

        new_dict = {year: causes}
        proportions_dict_cs.update(new_dict)

    props_df = pd.DataFrame(data=proportions_dict_cs)
    props_df = props_df.fillna(0)

    labels = list()
    values = list()
    for index in props_df.index:
        value = round(props_df.loc[index].mean(), 2)
        values.append(value)
        labels.append(f'{index} ({value}%)')
    sizes = values

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.legend(labels, loc="best")
    plt.title('Proportion of total CS deliveries by indication')
    plt.savefig(f'{graph_location}/cs_by_indication.png')
    plt.show()

    # ------------------------------------------ Maternal Sepsis Rate... ----------------------------------------------
    an_sep_data = analysis_utility_functions.get_comp_mean_and_rate(
        'clinical_chorioamnionitis', birth_data_ex2010[0], an_comps, 1000, sim_years)

    la_sep_data = analysis_utility_functions.get_comp_mean_and_rate(
        'sepsis', birth_data[0], la_comps, 1000, sim_years)

    pn_la_sep_data = analysis_utility_functions.get_comp_mean_and_rate(
        'sepsis_postnatal', birth_data_ex2010[0], la_comps, 1000, sim_years)

    pn_sep_data = analysis_utility_functions.get_comp_mean_and_rate(
        'sepsis', birth_data_ex2010[0], pn_comps, 1000, sim_years)

    complete_pn_sep_data = [x + y for x, y in zip(pn_la_sep_data[0], pn_sep_data[0])]
    complete_pn_sep_lq = [x + y for x, y in zip(pn_la_sep_data[1], pn_sep_data[1])]
    complete_pn_sep_up = [x + y for x, y in zip(pn_la_sep_data[2], pn_sep_data[2])]

    total_sep_rates = [x + y + z for x, y, z in zip(an_sep_data[0], la_sep_data[0], complete_pn_sep_data)]
    sep_lq = [x + y + z for x, y, z in zip(an_sep_data[1], la_sep_data[1], complete_pn_sep_lq)]
    sep_uq = [x + y + z for x, y, z in zip(an_sep_data[2], la_sep_data[2], complete_pn_sep_up)]

    # todo: note, we would expect our rate to be higher than this
    target_sep_dict = {'double': True,
                       'first': {'year': 2010, 'value': 4.7, 'label': 'BEmONC 2010', 'ci': 0},
                       'second': {'year': 2015, 'value': 1.89, 'label': 'BEmONC 2015', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, total_sep_rates, sep_lq, sep_uq, target_sep_dict, 'Rate per 1000 births',
        'Rate of Maternal Sepsis per Year', graph_location, 'sepsis_rate')

    # ----------------------------------------- Postpartum Haemorrhage... ---------------------------------------------
    la_pph_data = analysis_utility_functions.get_comp_mean_and_rate(
        'primary_postpartum_haemorrhage', birth_data_ex2010[0], la_comps, 1000, sim_years)

    pn_pph_data = analysis_utility_functions.get_comp_mean_and_rate(
        'secondary_postpartum_haemorrhage', birth_data_ex2010[0], pn_comps, 1000, sim_years)

    total_pph_rates = [x + y for x, y in zip(la_pph_data[0], pn_pph_data[0])]
    pph_lq = [x + y for x, y in zip(la_pph_data[1], pn_pph_data[1])]
    pph_uq = [x + y for x, y in zip(la_pph_data[2], pn_pph_data[2])]

    target_pph_dict = {'double': True,
                       'first': {'year': 2010, 'value': 16, 'label': 'BEmONC 2010', 'ci': 0},
                       'second': {'year': 2015, 'value': 14.6, 'label': 'BEmONC 2015', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, total_pph_rates, pph_lq, pph_uq, target_pph_dict, 'Rate per 1000 births',
        'Rate of Postpartum Haemorrhage per Year', graph_location, 'pph_rate')

    # ------------------------------------------- Intrapartum Stillbirth ... ------------------------------------------
    ip_still_birth_data = analysis_utility_functions.get_mean_and_quants(ip_stillbirth_results, sim_years)
    ip_sbr_per_year = [(x/y) * 1000 for x, y in zip(ip_still_birth_data[0], birth_data_ex2010[0])]
    ip_sbr_lqs = [(x/y) * 1000 for x, y in zip(ip_still_birth_data[1], birth_data_ex2010[0])]
    ip_sbr_uqs = [(x/y) * 1000 for x, y in zip(ip_still_birth_data[2], birth_data_ex2010[0])]

    target_ipsbr_dict = {'double': True,
                         'first': {'year': 2010, 'value': 10, 'label': 'UN est.', 'ci': 0},
                         'second': {'year': 2015, 'value': 8.15, 'label': 'UN est.', 'ci': 0},
                         }

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, ip_sbr_per_year, ip_sbr_lqs, ip_sbr_uqs, target_ipsbr_dict, 'Rate per 1000 births',
        'Intrapartum Stillbirth Rate per Year', graph_location, 'sbr_ip')

    total_sbr = [x + y for x, y in zip(an_sbr_per_year, ip_sbr_per_year)]
    total_lqs = [x + y for x, y in zip(an_sbr_lqs, ip_sbr_lqs)]
    total_uqs = [x + y for x, y in zip(an_sbr_uqs, ip_sbr_uqs)]

    fig, ax = plt.subplots()
    ax.plot(sim_years, total_sbr, label="Model (mean)", color='deepskyblue')
    ax.fill_between(sim_years, total_lqs, total_uqs, color='b', alpha=.1)
    plt.errorbar(2010, 20, yerr=(23-17)/2, label='UN sb report', fmt='o', color='green', ecolor='mediumseagreen',
                 elinewidth=3, capsize=0)
    plt.errorbar(2015, 16.3, yerr=(18.1-14.7)/2, label='UN sb report', fmt='o', color='green', ecolor='mediumseagreen',
                 elinewidth=3, capsize=0)
    ax.plot([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019], [20, 19, 19, 18, 18, 17, 17, 17, 17, 16],
            label="UN IGCME", color='red')
    ax.fill_between([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
                    [17, 17, 17, 17, 17, 16, 16, 16, 15, 15],
                    [23, 22, 21, 20, 19, 18, 18, 18, 18, 18], color='pink', alpha=.1)

    plt.xlabel('Year')
    plt.ylabel("Stillbirths per 1000 live births")
    plt.title('Stillbirth Rate per Year')
    plt.legend()
    plt.savefig(f'{graph_location}/sbr.png')
    plt.show()

    # ----------------------------------------- Fistula... -------------------------------------------------
    vv_fis_data = analysis_utility_functions.get_comp_mean_and_rate(
        'vesicovaginal_fistula', birth_data_ex2010[0], pn_comps, 1000, sim_years)

    rv_fis_data = analysis_utility_functions.get_comp_mean_and_rate(
        'rectovaginal_fistula', birth_data_ex2010[0], pn_comps, 1000, sim_years)

    total_fistula_rates = [x + y for x, y in zip(vv_fis_data[0], rv_fis_data[0])]
    fis_lqs = [x + y for x, y in zip(vv_fis_data[1], rv_fis_data[1])]
    fis_uqs = [x + y for x, y in zip(vv_fis_data[2], rv_fis_data[2])]

    target_fistula_dict = {'double': False,
                           'first': {'year': 2015, 'value': 6, 'label': 'DHS 2015', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, total_fistula_rates, fis_lqs, fis_uqs, target_fistula_dict, 'Rate per 1000 births',
        'Rate of Obstetric Fistula per Year', graph_location, 'fistula_rate')

    # ==================================================== NEWBORN OUTCOMES ==========================================
    #  ------------------------------------------- Neonatal sepsis (labour & postnatal) ------------------------------
    early_ns_data = analysis_utility_functions.get_comp_mean_and_rate(
        'early_onset_sepsis', birth_data_ex2010[0], nb_outcomes_df, 1000, sim_years)

    early_ns_pn = analysis_utility_functions.get_comp_mean_and_rate(
        'early_onset_sepsis', birth_data_ex2010[0], nb_outcomes_pn_df, 1000, sim_years)

    late_ns_data = analysis_utility_functions.get_comp_mean_and_rate(
        'late_onset_sepsis', birth_data_ex2010[0], nb_outcomes_pn_df, 1000, sim_years)

    target_nsep_dict = {'double': False,
                        'first': {'year': 2020, 'value': 39.3, 'label': 'Fleischmann et al.', 'ci': 0},
                        }

    total_ns_rates = [x + y + z for x, y, z in zip(early_ns_data[0], early_ns_pn[0], late_ns_data[0])]
    ns_lqs = [x + y + z for x, y, z in zip(early_ns_data[1], early_ns_pn[1], late_ns_data[1])]
    ns_uqs = [x + y + z for x, y, z in zip(early_ns_data[2], early_ns_pn[2], late_ns_data[2])]

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, total_ns_rates, ns_lqs, ns_uqs, target_nsep_dict, 'Rate per 1000 births',
        'Rate of Neonatal Sepsis per year', graph_location, 'neo_sepsis_rate')

    # TODO: more analysis on rate by timing?

    #  ------------------------------------------- Neonatal encephalopathy ------------------------------------------
    mild_data = analysis_utility_functions.get_comp_mean_and_rate(
        'mild_enceph', birth_data_ex2010[0], nb_outcomes_df, 1000, sim_years)

    mod_data = analysis_utility_functions.get_comp_mean_and_rate(
        'moderate_enceph', birth_data_ex2010[0], nb_outcomes_df, 1000, sim_years)

    sev_data = analysis_utility_functions.get_comp_mean_and_rate(
        'severe_enceph', birth_data_ex2010[0], nb_outcomes_df, 1000, sim_years)

    total_enceph_rates = [x + y + z for x, y, z in zip(mild_data[0], mod_data[0], sev_data[0])]
    enceph_lq = [x + y + z for x, y, z in zip(mild_data[1], mod_data[1], sev_data[1])]
    enceph_uq = [x + y + z for x, y, z in zip(mild_data[2], mod_data[2], sev_data[2])]

    target_rate_enceph = list()  # todo: replace
    for year in sim_years:
        target_rate_enceph.append(19)

    target_enceph_dict = {'double': True,
                          'first': {'year': 2010, 'value': 19.42, 'label': 'GBD 2010', 'ci': 0},
                          'second': {'year': 2015, 'value': 18.59, 'label': 'GBD 2015', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, total_enceph_rates, enceph_lq, enceph_uq, target_enceph_dict, 'Rate per 1000 births',
        'Rate of Neonatal Encephalopathy per year', graph_location, 'neo_enceph_rate')

    # ----------------------------------------- Respiratory Depression ------------------------------------------------
    rd_data = analysis_utility_functions.get_comp_mean_and_rate(
        'not_breathing_at_birth', birth_data_ex2010[0], nb_outcomes_df, 1000, sim_years)

    dummy_dict = {'double': False,
                  'first': {'year': 2010, 'value': 0, 'label': 'UNK.', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, rd_data[0], rd_data[1], rd_data[2], dummy_dict,'Rate per 1000 births',
        'Rate of Neonatal Respiratory Depression per year', graph_location, 'neo_resp_depression_rate')

    # ----------------------------------------- Respiratory Distress Syndrome ------------------------------------------
    ept = analysis_utility_functions.get_mean_and_quants_from_str_df(la_comps, 'early_preterm_labour', sim_years)[0]
    # todo: should be live births
    lpt = analysis_utility_functions.get_mean_and_quants_from_str_df(la_comps, 'late_preterm_labour', sim_years)[0]
    total_ptbs = [x + y for x, y in zip(ept, lpt)]

    rds_data = analysis_utility_functions.get_comp_mean_and_rate(
        'respiratory_distress_syndrome', total_ptbs, nb_outcomes_df, 1000, sim_years)

    target_rds_dict = {'double': False,
                       'first': {'year': 2019, 'value': 350, 'label': 'Muhe et al.', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, rds_data[0], rds_data[1], rds_data[2], target_rds_dict, 'Rate per 1000 preterm births',
        'Rate of Preterm Respiratory Distress Syndrome per year', graph_location, 'neo_rds_rate')

    # - TOTAL NOT BREATHING NEWBORNS-
    rds_data_over_births = analysis_utility_functions.get_comp_mean_and_rate(
        'respiratory_distress_syndrome', birth_data_ex2010[0], nb_outcomes_df, 1000, sim_years)

    rate = [x + y + z for x, y, z in zip(rds_data_over_births[0], total_enceph_rates, rd_data[0])]

    analysis_utility_functions.simple_bar_chart(
        rate, 'Year', 'Rate per 1000 births', 'Rate of Not Breathing Newborns per year', 'neo_total_not_breathing',
        sim_years, graph_location)

    # TODO: add calibration target for 'apnea' which should be approx 5.7% total births

    # ----------------------------------------- Congenital Anomalies -------------------------------------------------
    rate_of_ca = analysis_utility_functions.get_comp_mean_and_rate(
        'congenital_heart_anomaly', birth_data_ex2010[0], nb_outcomes_df, 1000, sim_years)[0]

    rate_of_laa = analysis_utility_functions.get_comp_mean_and_rate(
        'limb_or_musculoskeletal_anomaly', birth_data_ex2010[0], nb_outcomes_df, 1000, sim_years)[0]

    rate_of_ua = analysis_utility_functions.get_comp_mean_and_rate(
        'urogenital_anomaly', birth_data_ex2010[0], nb_outcomes_df, 1000, sim_years)[0]

    rate_of_da = analysis_utility_functions.get_comp_mean_and_rate(
        'digestive_anomaly', birth_data_ex2010[0], nb_outcomes_df, 1000, sim_years)[0]

    rate_of_oa = analysis_utility_functions.get_comp_mean_and_rate(
        'other_anomaly', birth_data_ex2010[0], nb_outcomes_df, 1000, sim_years)[0]

    plt.plot(sim_years, rate_of_ca, label="heart")
    plt.plot(sim_years, rate_of_laa, label="limb/musc.")
    plt.plot(sim_years, rate_of_ua, label="urogenital")
    plt.plot(sim_years, rate_of_da, label="digestive")
    plt.plot(sim_years, rate_of_oa, label="other")

    plt.xlabel('Year')
    plt.ylabel('Rate per 1000 births')
    plt.title('Yearly trends for Congenital Birth Anomalies')
    plt.legend()
    plt.savefig(f'{graph_location}/neo_rate_of_cong_anom.png')
    plt.show()

    # Breastfeeding
    # todo

    # GET MEAN/MEDIAN SQUEEZE
    #for hsi in ['SkilledBirthAttendance', 'AntenatalWardInpatientCare', 'CareOfTheNewbornBySkilledAttendantAtBirth',
    #            'Labour_ReceivesPostnatalCheck', 'NewbornOutcomes_ReceivesPostnatalCheck',
    #            'ReceivesComprehensiveEmergencyObstetricCare']:
    #    analysis_utility_functions.return_median_and_mean_squeeze_factor_for_hsi(results_folder, hsi, sim_years,
    #                                                                             graph_location)
