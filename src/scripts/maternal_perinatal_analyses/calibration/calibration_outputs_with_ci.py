import os
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as st
from tlo.analysis.utils import extract_results, get_scenario_outputs

from src.scripts.maternal_perinatal_analyses.analysis_scripts import analysis_utility_functions

plt.style.use('seaborn')


def output_incidence_for_calibration(scenario_filename, pop_size, outputspath, sim_years):
    """
    This function extracts incidence rates (and more) from the model and generates key calibration plots
    :param scenario_filename: file name of the scenario
    :param pop_size: population size this scenario was run on
    :param outputspath:directory for graphs to be saved
    :param sim_years: years the scenario was ran for
    """

    # Find results folder (most recent run generated using that scenario_filename)
    results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]

    # Create folder to store graphs (if it hasnt already been created when ran previously)
    path = f'{outputspath}/calibration_output_graphs_{pop_size}_{results_folder.name}'
    if not os.path.isdir(path):
        os.makedirs(f'{outputspath}/calibration_output_graphs_{pop_size}_{results_folder.name}')

    graph_location = path

    # ============================================HELPER FUNCTIONS... =================================================
    def get_modules_maternal_complication_dataframes(module):
        cd_df = extract_results(
            results_folder,
            module=f"tlo.methods.{module}",
            key="maternal_complication",
            custom_generate_series=(
                lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'type'])['person'].count()),
            do_scaling=False
        )
        complications_df = cd_df.fillna(0)

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

    preg_data = analysis_utility_functions.return_95_CI_across_runs(pregnancy_poll_results, sim_years)
    analysis_utility_functions.simple_line_chart_with_ci(
        sim_years, preg_data, 'Pregnancies (mean)', 'Mean number of pregnancies', 'pregnancies', graph_location)

    # ---------------------------------------------Total births and stillbirths...-------------------------------------
    # Live births...
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

    # birth_data = analysis_utility_functions.return_95_CI_across_runs(births_results, sim_years)

    # Stillbirths...
    an_sb = extract_results(
        results_folder,
        module="tlo.methods.pregnancy_supervisor",
        key="antenatal_stillbirth",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()
        ),
    )
    an_stillbirth_results = an_sb.fillna(0)

    ip_sb = extract_results(
        results_folder,
        module="tlo.methods.labour",
        key="intrapartum_stillbirth",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()
        ),
    )
    ip_stillbirth_results = ip_sb.fillna(0)

    lb_data = analysis_utility_functions.return_95_CI_across_runs(births_results_exc_2010, sim_years)
    analysis_utility_functions.simple_line_chart_with_ci(
        sim_years, lb_data, 'Births (mean)', 'Mean number of Live Births per Year', 'live_births',
        graph_location)

    all_births_df = an_stillbirth_results + ip_stillbirth_results + births_results_exc_2010
    all_births_data = analysis_utility_functions.return_95_CI_across_runs(all_births_df, sim_years)
    analysis_utility_functions.simple_line_chart_with_ci(
        sim_years, all_births_data, 'Births (mean)', 'Mean number of Total Births per Year', 'births', graph_location)

    # Stillbirth rates...
    an_sb_df = (an_stillbirth_results/all_births_df) * 1000
    an_sbr_per_year = analysis_utility_functions.return_95_CI_across_runs(an_sb_df, sim_years)
    ip_sb_df = (ip_stillbirth_results/all_births_df) * 1000
    ip_sbr_per_year = analysis_utility_functions.return_95_CI_across_runs(ip_sb_df, sim_years)
    total_sb_df = ((an_stillbirth_results + ip_stillbirth_results)/all_births_df) * 1000
    total_sbr = analysis_utility_functions.return_95_CI_across_runs(total_sb_df, sim_years)

    un_igcme_data = [[20, 19, 19, 18, 18, 17, 17, 17, 17, 16, 16, 16],
                     [17, 17, 17, 17, 17, 16, 16, 16, 15, 15, 14, 13],
                     [23, 22, 21, 20, 19, 18, 18, 18, 18, 18, 19, 19]]

    un_igcme_data_adj = [[(x / 2) for x in un_igcme_data[0]],
                         [(x / 2) for x in un_igcme_data[1]],
                         [(x / 2) for x in un_igcme_data[2]]]

    def get_stillbirth_graphs(rate_data, calib_data, group):
        fig, ax = plt.subplots()
        ax.plot(sim_years, rate_data[0], label="Model", color='deepskyblue')
        ax.fill_between(sim_years, rate_data[1], rate_data[2], label='95% CI', color='b', alpha=.1)
        # plt.errorbar(2010, 20, yerr=(23 - 17) / 2, label='UN (2010)', fmt='o', color='green', ecolor='mediumseagreen',
        #              elinewidth=3, capsize=0)
        # plt.errorbar(2019, 16.3, yerr=(18.1 - 14.7) / 2, label='UN (2019)', fmt='o', color='green',
        #              ecolor='mediumseagreen',
        #              elinewidth=3, capsize=0)
        ax.plot([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021], calib_data[0],
                label="UN IGCME 2020 (Uncertainty interval)", color='green')
        ax.fill_between([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021], calib_data[2],
                        calib_data[1], color='mediumseagreen', alpha=.1)
        if group == 'Total':
            ax.set(ylim=(0, 26))
        else:
            ax.set(ylim=(0, 15))

        plt.xlabel('Year')
        plt.ylabel("Stillbirths per 1000 total births")
        plt.title(f'{group} stillbirth rate per year')
        plt.legend()
        plt.savefig(f'{graph_location}/{group}_sbr.png')
        plt.show()

    get_stillbirth_graphs(an_sbr_per_year, un_igcme_data_adj, 'Antenatal')
    get_stillbirth_graphs(ip_sbr_per_year, un_igcme_data_adj, 'Intrapartum')
    get_stillbirth_graphs(total_sbr, un_igcme_data, 'Total')

    # -------------------------------------------------Completed pregnancies...----------------------------------------
    total_completed_pregnancies_df = an_comps.loc[(slice(None), 'ectopic_unruptured'), slice(None)].droplevel(1) + \
                                     an_comps.loc[(slice(None), 'induced_abortion'), slice(None)].droplevel(1) + \
                                     an_comps.loc[(slice(None), 'spontaneous_abortion'), slice(None)].droplevel(1) + \
                                     an_stillbirth_results + \
                                     ip_stillbirth_results + \
                                     births_results_exc_2010

    # ================================== PROPORTION OF PREGNANCIES ENDING IN BIRTH... ================================
    total_pregnancy_losses_df =  an_comps.loc[(slice(None), 'ectopic_unruptured'), slice(None)].droplevel(1) + \
                                     an_comps.loc[(slice(None), 'induced_abortion'), slice(None)].droplevel(1) + \
                                     an_comps.loc[(slice(None), 'spontaneous_abortion'), slice(None)].droplevel(1) + \
                                     an_stillbirth_results + \
                                     ip_stillbirth_results

    prop_lost_pregnancies_df = (total_pregnancy_losses_df / pregnancy_poll_results) * 100
    prop_lost_pregnancies = analysis_utility_functions.return_95_CI_across_runs(prop_lost_pregnancies_df, sim_years)

    analysis_utility_functions.simple_bar_chart(
        prop_lost_pregnancies[0], 'Year', '% of Total Pregnancies',
        'Proportion of total pregnancies ending in pregnancy loss', 'preg_loss_proportion', sim_years,
        graph_location)

    # todo IP stillbirths need to be added to total births and shouldnt count as a pregnancy loss

    # ========================================== INTERVENTION COVERAGE... =============================================
    # 1.) Antenatal Care... # TODO: THIS COULD CERTAINLY BE SIMPLIFIED
    # Mean proportion of women (across draws) who have given birth that have attended ANC1, ANC4+ and ANC8+ per year...
    anc_coverage = extract_results(
        results_folder,
        module="tlo.methods.care_of_women_during_pregnancy",
        key="anc_count_on_birth",
        custom_generate_series=(
            lambda df: df.assign(
                year=df['date'].dt.year).groupby(['year'])['person_id'].count()),
        do_scaling=True
    )

    # Next get a version of that DF with women who attended >= 4/8 visits by birth
    def get_anc_df(visit_number):
        an = extract_results(
            results_folder,
            module="tlo.methods.care_of_women_during_pregnancy",
            key="anc_count_on_birth",
            custom_generate_series=(
                lambda df: df.loc[df['total_anc'] >= visit_number].assign(
                    year=df['date'].dt.year).groupby(['year'])['person_id'].count()),
            do_scaling=True
        )
        anc_cov_of_interest = an.fillna(0)
        return anc_cov_of_interest

    cov_4 = (get_anc_df(4) / anc_coverage) * 100
    cov_4_df = cov_4.fillna(0)

    cov_1 = (get_anc_df(1) / anc_coverage) * 100
    cov_1_df = cov_1.fillna(0)

    anc1_coverage = analysis_utility_functions.return_95_CI_across_runs(cov_1_df, sim_years)
    anc4_coverage = analysis_utility_functions.return_95_CI_across_runs(cov_4_df, sim_years)

    target_anc1_dict = {'double': True,
                        'first': {'year': 2010, 'value': 94, 'label': 'DHS (2010)', 'ci': 0},
                        'second': {'year': 2015, 'value': 95, 'label': 'DHS (2015)', 'ci': 0}}
    target_anc4_dict = {'double': True,
                        'first': {'year': 2010, 'value': 45.5, 'label': 'DHS (2010)', 'ci': 0},
                        'second': {'year': 2015, 'value': 51, 'label': 'DHS (2015)', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, anc1_coverage, target_anc1_dict, '% of women who gave birth',
        'Proportion of women who gave birth that attended one or more ANC contacts per year', graph_location,
        'anc_prop_anc1')

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, anc4_coverage, target_anc4_dict, '% of women who gave birth',
        'Proportion of women who gave birth that attended four or more ANC contacts per year', graph_location,
        'anc_prop_anc4')

    # TOTAL ANC COUNTS PER WOMAN
    r = extract_results(
        results_folder,
        module="tlo.methods.care_of_women_during_pregnancy",
        key="anc_count_on_birth",
        custom_generate_series=(
            lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'total_anc'])['person_id'].count()),
        do_scaling=False
    )
    results = r.fillna(0)
    # get yearly outputs
    anc_count_df = pd.DataFrame(columns=sim_years, index=[0, 1, 2, 3, 4, 5, 6, 7, 8])

    for year in sim_years:
        for row in anc_count_df.index:
            if row in results.loc[year].index:
                anc_count_df.at[row, year] = results.loc[year, row].mean()
            else:
                anc_count_df.at[row, year] = 0

    anc_count_df = anc_count_df.drop([0])
    for year in sim_years:
        total_per_year = 0
        for row in anc_count_df[year]:
            total_per_year += row

        if total_per_year != 0:
            for index in anc_count_df.index:
                anc_count_df.at[index, year] = (anc_count_df.at[index, year]/total_per_year) * 100

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
    ax.set_ylabel('% of total women attending one or more ANC contact')
    ax.set_title('Number of ANC contacts attended by women attending at least one contact per year')
    ax.legend(bbox_to_anchor=(1.2, 1.1))
    plt.savefig(f'{graph_location}/anc_total_visits.png', bbox_inches="tight")
    plt.show()

    # Mean proportion of women who attended at least one ANC visit that attended at < 4, 4-5, 6-7 and > 8 months
    # gestation...
    anc_ga = extract_results(
        results_folder,
        module="tlo.methods.care_of_women_during_pregnancy",
        key="anc_count_on_birth",
        custom_generate_series=(
            lambda df_: df_.assign(year=df_['date'].dt.year) .groupby(
                ['year', 'total_anc', 'ga_anc_one'])['person_id'].count()),
        do_scaling=False
    )
    anc_ga_first_visit = anc_ga.fillna(0)

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
        # Add calibration data (ive added the 'unknown' from DHS data to the 8+)
        if year == 2010:
            anc_before_four_months.append(12.4)
            anc_before_four_five_months.append(48.2)
            anc_before_six_seven_months.append(35.6)
            anc_before_eight_plus_months.append(3.8)
        if year == 2015:
            anc_before_four_months.append(24)
            anc_before_four_five_months.append(51.2)
            anc_before_six_seven_months.append(21.4)
            anc_before_eight_plus_months.append(3.4)

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

    labels = ['DHS (2010)', '2010', '2011', '2012', '2013', '2014', 'DHS (2015)', '2015', '2016', '2017', '2018',
              '2019', '2020', '2021', '2022']
    width = 0.35  # the width of the bars: can also be len(x) sequence
    x = np.arange(len(labels))

    fig, ax = plt.subplots()
    ax.bar(labels, anc_before_eight_plus_months, width, label='>8m',
           bottom=[x + y + z for x, y, z in zip(anc_before_four_months, anc_before_four_five_months,
                                                anc_before_six_seven_months)])
    ax.bar(labels, anc_before_six_seven_months, width, label='6-7m',
           bottom=[x + y for x, y in zip(anc_before_four_months, anc_before_four_five_months)])
    ax.bar(labels, anc_before_four_five_months, width, label='4-5m',
           bottom=anc_before_four_months)
    ax.bar(labels, anc_before_four_months, width, label='<4m')
    plt.xticks(rotation=90, ha='right')
    # Put a legend to the right of the current axis
    ax.legend(bbox_to_anchor=(1.2, 1.1))
    ax.set_ylabel('% of first ANC contacts')
    ax.set_title('Maternal gestational age at first ANC contact by year')
    plt.savefig(f'{graph_location}/anc_ga_first_visit_update.png', bbox_inches="tight")
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

    # 2.) Facility delivery
    # Total FDR per year (denominator - total births)
    all_deliveries = extract_results(
        results_folder,
        module="tlo.methods.labour",
        key="delivery_setting_and_mode",
        custom_generate_series=(
            lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year'])[
                'mother'].count()),
        do_scaling=True
    )

    def get_delivery_rate(facility_type):
        if facility_type == 'facility':
            deliver_setting_results = extract_results(
                results_folder,
                module="tlo.methods.labour",
                key="delivery_setting_and_mode",
                custom_generate_series=(
                    lambda df: df.loc[df['facility_type'] != 'home_birth'].assign(
                        year=df['date'].dt.year).groupby(['year'])[
                        'mother'].count()))
        else:
            deliver_setting_results = extract_results(
                results_folder,
                module="tlo.methods.labour",
                key="delivery_setting_and_mode",
                custom_generate_series=(
                    lambda df: df.loc[df['facility_type'] == facility_type].assign(
                        year=df['date'].dt.year).groupby(['year'])[
                        'mother'].count()))

        rate = (deliver_setting_results/births_results_exc_2010) * 100
        data = analysis_utility_functions.return_95_CI_across_runs(rate, sim_years)
        return data

    fd_data = get_delivery_rate('facility')
    hb_data = get_delivery_rate('home_birth')
    hp_data = get_delivery_rate('hospital')
    hc_data = get_delivery_rate('health_centre')

    target_fd_dict = {'double': True,
                      'first': {'year': 2010, 'value': 73, 'label': 'DHS (2010)', 'ci': 0},
                      'second': {'year': 2015, 'value': 91, 'label': 'DHS (2015)', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, fd_data, target_fd_dict, '% of total births',
        'Proportion of births occurring in a health facility per year ', graph_location, 'sba_prop_facility_deliv')

    labels = sim_years
    width = 0.35       # the width of the bars: can also be len(x) sequence
    fig, ax = plt.subplots()

    ax.bar(labels, hp_data[0], width, label='Hospital',
           bottom=[x + y for x, y in zip(hb_data[0], hc_data[0])])
    ax.bar(labels, hc_data[0], width, label='Health Centre',
           bottom=hb_data[0])
    ax.bar(labels, hb_data[0], width, label='Home')
    ax.set_ylabel('% of total births')
    ax.set_title('Proportion of total births by location of delivery')
    ax.legend(bbox_to_anchor=(1.3, 1.))
    plt.savefig(f'{graph_location}/sba_delivery_location.png', bbox_inches="tight")
    plt.show()

    # 3.) Postnatal Care
    # --- PNC ---
    all_surviving_mothers = extract_results(
        results_folder,
        module="tlo.methods.postnatal_supervisor",
        key="total_mat_pnc_visits",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['mother'].count()))

    # Extract data on all women with 1+ PNC visits
    pnc_results_maternal = extract_results(
        results_folder,
        module="tlo.methods.postnatal_supervisor",
        key="total_mat_pnc_visits",
        custom_generate_series=(
            lambda df: df.loc[df['visits'] > 0].assign(year=df['date'].dt.year).groupby(['year'])[
                'mother'].count()))

    # Followed by newborns...
    all_surviving_newborns = extract_results(
        results_folder,
        module="tlo.methods.postnatal_supervisor",
        key="total_neo_pnc_visits",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])[
                'child'].count()))

    pnc_results_newborn = extract_results(
        results_folder,
        module="tlo.methods.postnatal_supervisor",
        key="total_neo_pnc_visits",
        custom_generate_series=(
            lambda df: df.loc[df['visits'] > 0].assign(year=df['date'].dt.year).groupby(['year'])[
                'child'].count()),

    )

    # TODO: survival denominator

    cov_mat_birth_df = (pnc_results_maternal / births_results_exc_2010) * 100
    mat_birth_pnc_data = analysis_utility_functions.return_95_CI_across_runs(cov_mat_birth_df, sim_years)
    cov_mat_surv_df = (pnc_results_maternal/all_surviving_mothers) * 100
    mat_surv_pnc_data = analysis_utility_functions.return_95_CI_across_runs(cov_mat_surv_df, sim_years)

    cov_neo_birth_df = (pnc_results_newborn / births_results_exc_2010) * 100
    neo_birth_pnc_data = analysis_utility_functions.return_95_CI_across_runs(cov_neo_birth_df, sim_years)
    cov_neo_surv_df = (pnc_results_newborn / all_surviving_newborns) * 100
    neo_surv_pnc_data = analysis_utility_functions.return_95_CI_across_runs(cov_neo_surv_df, sim_years)

    target_mpnc_dict = {'double': True,
                        'first': {'year': 2010, 'value': 50, 'label': 'DHS (2010)', 'ci': 0},
                        'second': {'year': 2015, 'value': 48, 'label': 'DHS (2015)', 'ci': 0}}

    target_npnc_dict = {'double': False,
                        'first': {'year': 2015, 'value': 60, 'label': 'DHS (2015)', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, mat_birth_pnc_data, target_mpnc_dict, '% of total births',
        'Proportion of total births after which the mother received any PNC per year', graph_location, 'pnc_mat')

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, neo_birth_pnc_data, target_npnc_dict, '% of total births',
        'Proportion of total births after which the neonate received any PNC per year', graph_location, 'pnc_neo')

    def get_early_late_pnc_split(module, target, file_name):
        p = extract_results(
            results_folder,
            module=f"tlo.methods.{module}",
            key="postnatal_check",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'timing',
                                                                       'visit_number'])['person_id'].count()
            ),
        )
        pnc = p.fillna(0)
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
    def return_rate(num_df, denom_df, denom_val):
        rate_df = (num_df / denom_df) * denom_val
        data = analysis_utility_functions.return_95_CI_across_runs(rate_df, sim_years)
        return data

    # ---------------------------------------- Twinning Rate... -------------------------------------------------------
    # % Twin births/Total Births per year
    tr = extract_results(
        results_folder,
        module="tlo.methods.newborn_outcomes",
        key="twin_birth",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()
        ),
    )
    twins_results = tr.fillna(0)
    total_deliv = births_results_exc_2010 - twins_results
    final_twining_rate = return_rate(twins_results, total_deliv, 100)

    target_twin_dict = {'double': False,
                        'first': {'year': 2010, 'value': 3.9, 'label': 'DHS 2010', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, final_twining_rate, target_twin_dict, 'Rate per 100 pregnancies',
        'Yearly trends for Twin Births', graph_location, 'twin_rate')

    # ---------------------------------------- Early Pregnancy Loss... ------------------------------------------------
    # Ectopic pregnancies/Total pregnancies
    ep = an_comps.loc[(slice(None), 'ectopic_unruptured'), slice(None)].droplevel(1)
    ectopic_data = return_rate(ep, pregnancy_poll_results, 1000)
    target_ect_dict = {'double': False,
                       'first': {'year': 2015, 'value': 10, 'label': 'Est.', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, ectopic_data, target_ect_dict, 'Rate per 1000 pregnancies',
        'Yearly trends for Ectopic Pregnancy', graph_location, 'ectopic_rate')

    # Ruptured ectopic pregnancies / Total Pregnancies
    ep_r = an_comps.loc[(slice(None), 'ectopic_ruptured'), slice(None)].droplevel(1)
    proportion_of_ectopics_that_rupture_per_year = return_rate(ep, ep_r, 1000)

    target_rate_rup = {'double': False,
                       'first': {'year': 2015, 'value': 92, 'label': 'Est.', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, proportion_of_ectopics_that_rupture_per_year, target_rate_rup, 'Proportion of ectopic pregnancies',
        'Proportion of Ectopic Pregnancies ending in rupture', graph_location, 'prop_rupture')

    # Spontaneous Abortions....
    sa = an_comps.loc[(slice(None), 'spontaneous_abortion'), slice(None)].droplevel(1)
    spotaneous_abortion_data = return_rate(sa, total_completed_pregnancies_df, 1000)

    target_sa_dict = {'double': False,
                      'first': {'year': 2016, 'value': 130, 'label': 'Dellicour et al.', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, spotaneous_abortion_data,
        target_sa_dict, 'Rate per 1000 completed pregnancies', 'Yearly rate of Miscarriage', graph_location,
        'miscarriage_rate')

    # Complicated SA / Total SA
    c_sa = an_comps.loc[(slice(None), 'complicated_spontaneous_abortion'), slice(None)].droplevel(1)
    proportion_of_complicated_sa_per_year = return_rate(c_sa, sa, 100)

    # TODO: COULD ADD 95% CI now
    analysis_utility_functions.simple_bar_chart(
        proportion_of_complicated_sa_per_year[0], 'Year', '% of Total Miscarriages',
        'Proportion of miscarriages leading to complications', 'miscarriage_prop_complicated', sim_years,
        graph_location)

    # Induced Abortions...
    ia = an_comps.loc[(slice(None), 'induced_abortion'), slice(None)].droplevel(1)
    ia_data = return_rate(ia, total_completed_pregnancies_df, 1000)

    target_ia_dict = {'double': True,
                      'first': {'year': 2010, 'value': 86, 'label': 'Levandowski et al.', 'ci': 0},
                      'second': {'year': 2015, 'value': 159, 'label': 'Polis et al.', 'ci': 0},
                      }

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, ia_data, target_ia_dict, 'Rate per 1000 completed pregnancies',
        'Yearly rate of Induced Abortion',  graph_location, 'abortion_rate')

    # Complicated IA / Total IA
    c_ia = an_comps.loc[(slice(None), 'complicated_induced_abortion'), slice(None)].droplevel(1)
    proportion_of_complicated_ia_per_year = return_rate(c_ia, ia, 100)

    analysis_utility_functions.simple_bar_chart(
        proportion_of_complicated_ia_per_year[0], 'Year', '% of Total Abortions',
        'Proportion of Abortions leading to complications', 'abortion_prop_complicated', sim_years, graph_location)

    # --------------------------------------------------- Syphilis Rate... --------------------------------------------
    syphilis_data = return_rate(an_comps.loc[(slice(None), 'syphilis'), slice(None)].droplevel(1),
                                total_completed_pregnancies_df, 1000)

    target_syph_dict = {'double': False,
                        'first': {'year': 2018, 'value': 20, 'label': 'HIV rpt data', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, syphilis_data, target_syph_dict,
        'Rate per 1000 completed pregnancies', 'Yearly rate of Syphilis', graph_location, 'syphilis_rate')

    # ------------------------------------------------ Gestational Diabetes... ----------------------------------------
    gdm_data = return_rate(an_comps.loc[(slice(None), 'gest_diab'), slice(None)].droplevel(1),
                           total_completed_pregnancies_df, 1000)

    target_gdm_dict = {'double': False,
                       'first': {'year': 2019, 'value': 16, 'label': 'Phiri et al.', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, gdm_data, target_gdm_dict, 'Rate per 1000 completed pregnancies',
        'Yearly rate of Gestational Diabetes', graph_location, 'gest_diab_rate', )

    # ------------------------------------------------ PROM... --------------------------------------------------------
    prom_data = return_rate(an_comps.loc[(slice(None), 'PROM'), slice(None)].droplevel(1), births_results_exc_2010,
                            1000)

    target_prm_dict = {'double': False,
                       'first': {'year': 2020, 'value': 27, 'label': 'Onwughara et al.', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, prom_data, target_prm_dict, 'Rate per 1000 births', 'Yearly rate of PROM', graph_location,
        'prom_rate')

    # ---------------------------------------------- Anaemia... -------------------------------------------------------
    # Total prevalence of Anaemia at birth (total cases of anaemia at birth/ total births per year) and by severity
    anaemia_results = extract_results(
        results_folder,
        module="tlo.methods.pregnancy_supervisor",
        key="conditions_on_birth",
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
        key="conditions_on_birth",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'anaemia_status'])['year'].count()
        ),
    )

    pn_pn_severity = extract_results(
        results_folder,
        module="tlo.methods.postnatal_supervisor",
        key="total_mat_pnc_visits",
        custom_generate_series=(
            lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'anaemia'])['mother'].count()))

    # an_data = analysis_utility_functions.get_mean_and_quants(anaemia_results, sim_years)
    # pn_data = analysis_utility_functions.get_mean_and_quants(pnc_anaemia, sim_years)

    def get_anaemia_graphs(df, timing, severity_df):
        target_an_dict = {'double': True,
                          'first': {'year': 2010, 'value': 37.5, 'label': 'DHS 2010', 'ci': 0},
                          'second': {'year': 2015, 'value': 45.1, 'label': 'DHS 2015', 'ci': 0},
                          }
        prev = return_rate(df, births_results_exc_2010, 100)

        analysis_utility_functions.line_graph_with_ci_and_target_rate(
            sim_years, prev, target_an_dict,
            'Prevalence at birth', f'Yearly prevalence of Anaemia (all severity) at {timing}', graph_location,
            f'anaemia_prev_{timing}')

        # todo: should maybe be total postnatal women still alive as opposed to births as will inflate

        prevalence_of_mild_anaemia_per_year = return_rate(
            severity_df.loc[(slice(None), 'mild'), slice(None)].droplevel(1), births_results_exc_2010, 100)
        prevalence_of_mod_anaemia_per_year = return_rate(
            severity_df.loc[(slice(None), 'moderate'), slice(None)].droplevel(1),births_results_exc_2010, 100)
        prevalence_of_sev_anaemia_per_year = return_rate(
            severity_df.loc[(slice(None), 'severe'), slice(None)].droplevel(1), births_results_exc_2010, 100)

        plt.plot(sim_years, prevalence_of_mild_anaemia_per_year[0], label="mild")
        plt.plot(sim_years, prevalence_of_mod_anaemia_per_year[0], label="moderate")
        plt.plot(sim_years, prevalence_of_sev_anaemia_per_year[0], label="severe")
        plt.xlabel('Year')
        plt.ylabel(f'Prevalence at {timing}')
        plt.title(f'Yearly trends for prevalence of anaemia by severity at {timing}')
        plt.legend()
        plt.savefig(f'{graph_location}/anaemia_by_severity_{timing}.png')
        plt.show()

    get_anaemia_graphs(anaemia_results, 'delivery', preg_an_severity)
    get_anaemia_graphs(pnc_anaemia, 'postnatal', pn_pn_severity)

    # ------------------------------------------- Hypertensive disorders -----------------------------------------------
    gh_df = an_comps.loc[(slice(None), 'mild_gest_htn'), slice(None)].droplevel(1) + \
            pn_comps.loc[(slice(None), 'mild_gest_htn'), slice(None)].droplevel(1)
    gh_data = return_rate(gh_df, births_results_exc_2010, 1000)

    sgh_df = an_comps.loc[(slice(None), 'severe_gest_htn'), slice(None)].droplevel(1) + \
             la_comps.loc[(slice(None), 'severe_gest_htn'), slice(None)].droplevel(1) + \
             pn_comps.loc[(slice(None), 'severe_gest_htn'), slice(None)].droplevel(1)
    sgh_data = return_rate(sgh_df, births_results_exc_2010, 1000)

    mpe_df = an_comps.loc[(slice(None), 'mild_pre_eclamp'), slice(None)].droplevel(1) + \
            pn_comps.loc[(slice(None), 'mild_pre_eclamp'), slice(None)].droplevel(1)
    mpe_data = return_rate(mpe_df, births_results_exc_2010, 1000)

    spe_df = an_comps.loc[(slice(None), 'severe_pre_eclamp'), slice(None)].droplevel(1) + \
             la_comps.loc[(slice(None), 'severe_pre_eclamp'), slice(None)].droplevel(1) + \
             pn_comps.loc[(slice(None), 'severe_pre_eclamp'), slice(None)].droplevel(1)
    spe_data = return_rate(spe_df, births_results_exc_2010, 1000)

    ec_df = an_comps.loc[(slice(None), 'eclampsia'), slice(None)].droplevel(1) + \
             la_comps.loc[(slice(None), 'eclampsia'), slice(None)].droplevel(1) + \
             pn_comps.loc[(slice(None), 'eclampsia'), slice(None)].droplevel(1)
    ec_data = return_rate(ec_df, births_results_exc_2010, 1000)

    target_gh_dict = {'double': False,
                      'first': {'year': 2019, 'value': 43.8, 'label': 'Noubiap et al.', 'ci': 0},
                      }
    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, gh_data, target_gh_dict, 'Rate per 1000 births',
        'Rate of Gestational Hypertension per Year',  graph_location, 'gest_htn_rate',)

    target_sgh_dict = {'double': False,
                       'first': {'year': 2019, 'value': 5.98, 'label': 'Noubiap et al.', 'ci': 0},
                       }

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, sgh_data, target_sgh_dict, 'Rate per 1000 births',
        'Rate of Severe Gestational Hypertension per Year', graph_location, 'severe_gest_htn_rate', )

    target_mpe_dict = {'double': False,
                       'first': {'year': 2019, 'value': 44, 'label': 'Noubiap et al', 'ci': 0},
                       }

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, mpe_data, target_mpe_dict, 'Rate per 1000 births',
        'Rate of Mild pre-eclampsia per Year', graph_location, 'mild_pre_eclampsia_rate')

    target_spe_dict = {'double': False,
                       'first': {'year': 2019, 'value': 22, 'label': 'Noubiap et al', 'ci': 0},
                       }

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, spe_data, target_spe_dict, 'Rate per 1000 births',
        'Rate of Severe pre-eclampsia per Year', graph_location,  'severe_pre_eclampsia_rate')

    target_ec_dict = {'double': False,
                      'first': {'year': 2019, 'value': 10, 'label': 'Vousden et al.', 'ci': 0},
                      }
    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, ec_data, target_ec_dict, 'Rate per 1000 births',
        'Rate of Eclampsia per Year', graph_location, 'eclampsia_rate')

    #  ---------------------------------------------Placenta praevia... ----------------------------------------------
    pp_data = return_rate(an_comps.loc[(slice(None), 'placenta_praevia'), slice(None)].droplevel(1), pregnancy_poll_results,
                            1000)

    target_pp_dict = {'double': False,
                      'first': {'year': 2017, 'value': 5.67, 'label': 'Senkoro et al.', 'ci': 0},
                      }
    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, pp_data, target_pp_dict, 'Rate per 1000 pregnancies',
        'Rate of Placenta Praevia per Year', graph_location, 'praevia_rate')

    #  ---------------------------------------------Placental abruption... --------------------------------------------
    pa_df = an_comps.loc[(slice(None), 'placental_abruption'), slice(None)].droplevel(1) + \
             la_comps.loc[(slice(None), 'placental_abruption'), slice(None)].droplevel(1)
    pa_data = return_rate(pa_df, births_results_exc_2010, 1000)

    target_pa_dict = {'double': False,
                      'first': {'year': 2015, 'value': 3, 'label': 'Macheku et al.', 'ci': 0},
                      }

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, pa_data, target_pa_dict, 'Rate per 1000 births',
        'Rate of Placental Abruption per Year', graph_location, 'abruption_rate')

    # --------------------------------------------- Antepartum Haemorrhage... -----------------------------------------
    # Rate of APH/total births (antenatal and labour)
    aph_df = an_comps.loc[(slice(None), 'mild_mod_antepartum_haemorrhage'), slice(None)].droplevel(1) + \
             an_comps.loc[(slice(None), 'severe_antepartum_haemorrhage'), slice(None)].droplevel(1) + \
             la_comps.loc[(slice(None), 'mild_mod_antepartum_haemorrhage'), slice(None)].droplevel(1) + \
             la_comps.loc[(slice(None), 'severe_antepartum_haemorrhage'), slice(None)].droplevel(1)

    aph_data = return_rate(aph_df, births_results_exc_2010, 1000)

    target_aph_dict = {'double': False,
                       'first': {'year': 2015, 'value': 4.6, 'label': 'BEmONC.', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, aph_data, target_aph_dict, 'Rate per 1000 births',
        'Rate of Antepartum Haemorrhage per Year', graph_location, 'aph_rate')

    # --------------------------------------------- Preterm birth ... ------------------------------------------------
    ptl_df = la_comps.loc[(slice(None), 'early_preterm_labour'), slice(None)].droplevel(1) + \
             la_comps.loc[(slice(None), 'late_preterm_labour'), slice(None)].droplevel(1)
    ptl_data = return_rate(ptl_df, births_results_exc_2010, 100)

    target_ptl_dict = {'double': True,
                       'first': {'year': 2012, 'value': 19.8, 'label': 'Antony et al.', 'ci': 0},
                       'second': {'year': 2014, 'value': 10, 'label': 'Chawanpaiboon et al.', 'ci': (14.3-7.4)/2},
                       }

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, ptl_data, target_ptl_dict, 'Proportion of total births',
        'Preterm birth rate', graph_location, 'ptb_rate')

    prop_early = return_rate(la_comps.loc[(slice(None), 'early_preterm_labour'), slice(None)].droplevel(1), ptl_df, 100)
    prop_late = return_rate(la_comps.loc[(slice(None), 'late_preterm_labour'), slice(None)].droplevel(1), ptl_df, 100)

    labels = sim_years
    width = 0.35       # the width of the bars: can also be len(x) sequence
    fig, ax = plt.subplots()

    ax.bar(labels, prop_early[0], width, label='Early Preterm',
           bottom=prop_late[0])
    ax.bar(labels, prop_late[0], width, label='Late Preterm')
    ax.set_ylabel('% of total Preterm Births')
    ax.set_title('Early vs Late Preterm Births')
    ax.legend()
    plt.savefig(f'{graph_location}/early_late_preterm.png')
    plt.show()

    # todo plot early and late seperated

    # --------------------------------------------- Post term birth ... -----------------------------------------------
    potl_data = return_rate(la_comps.loc[(slice(None), 'post_term_labour'), slice(None)].droplevel(1),
                            births_results_exc_2010, 100)

    target_potl_dict = {'double': False,
                        'first': {'year': 2014, 'value': 3.2, 'label': 'van den Broek et al.', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, potl_data, target_potl_dict, 'Proportion of total births',
        'Post term birth rate', graph_location, 'potl_rate')

    # ------------------------------------------------- Birth weight... -----------------------------------------------
    nb_oc_df = extract_results(
            results_folder,
            module="tlo.methods.newborn_outcomes",
            key="newborn_complication",
            custom_generate_series=(
                lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'type'])['newborn'].count()),
            do_scaling=False
        )
    nb_outcomes_df = nb_oc_df.fillna(0)

    nb_oc_pn_df = extract_results(
            results_folder,
            module="tlo.methods.postnatal_supervisor",
            key="newborn_complication",
            custom_generate_series=(
                lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'type'])['newborn'].count()),
            do_scaling=False
        )
    nb_outcomes_pn_df = nb_oc_pn_df.fillna(0)

    lbw_data = return_rate(nb_outcomes_df.loc[(slice(None), 'low_birth_weight'), slice(None)].droplevel(1),
                  births_results_exc_2010, 100)

    macro_data = return_rate(nb_outcomes_df.loc[(slice(None), 'macrosomia'), slice(None)].droplevel(1),
                  births_results_exc_2010, 100)

    sga_data = return_rate(nb_outcomes_df.loc[(slice(None), 'small_for_gestational_age'), slice(None)].droplevel(1),
                  births_results_exc_2010, 100)

    target_lbw_dict = {'double': True,
                       'first': {'year': 2010, 'value': 12, 'label': 'DHS 2010', 'ci': 0},
                       'second': {'year': 2015, 'value': 12, 'label': 'DHS 2015', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, lbw_data, target_lbw_dict, 'Proportion of total births',
        'Yearly Prevalence of Low Birth Weight', graph_location, 'neo_lbw_prev')

    target_mac_dict = {'double': False,
                       'first': {'year': 2019, 'value': 5.13, 'label': 'Ngwira et al.', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, macro_data, target_mac_dict, 'Proportion of total births',
        'Yearly Prevalence of Macrosomia', graph_location, 'neo_macrosomia_prev')

    target_sga_dict = {'double': False,
                       'first': {'year': 2010, 'value': 23.2, 'label': 'Lee et al.', 'ci': (27-19.1)/2}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, sga_data, target_sga_dict, 'Proportion of total births',
        'Yearly Prevalence of Small for Gestational Age', graph_location, 'neo_sga_prev')

    # todo: check with Ines r.e. SGA and the impact on her modules

    # --------------------------------------------- Obstructed Labour... ----------------------------------------------
    ol_data = return_rate(la_comps.loc[(slice(None), 'obstructed_labour'), slice(None)].droplevel(1),
                  births_results_exc_2010, 1000)

    target_ol_dict = {'double': True,
                      'first': {'year': 2010, 'value': 18.3, 'label': 'BEmONC 2010', 'ci': 0},
                      'second': {'year': 2015, 'value': 33.7, 'label': 'BEmONC 2015', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, ol_data, target_ol_dict, 'Rate per 1000 births',
        'Obstructed Labour Rate per Year', graph_location, 'ol_rate')

    # --------------------------------------------- Uterine rupture... -----------------------------------------------
    ur_data = return_rate(la_comps.loc[(slice(None), 'uterine_rupture'), slice(None)].droplevel(1),
                  births_results_exc_2010, 1000)

    target_ur_dict = {'double': True,
                      'first': {'year': 2010, 'value': 1.2, 'label': 'BEmONC 2010', 'ci': 0},
                      'second': {'year': 2015, 'value': 0.8, 'label': 'BEmONC 2015', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, ur_data,  target_ur_dict, 'Rate per 1000 births',
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

    cs_data = return_rate(delivery_mode.loc[(slice(None), 'caesarean_section'), slice(None)].droplevel(1),
                  births_results_exc_2010, 100)

    avd_data = return_rate(delivery_mode.loc[(slice(None), 'instrumental'), slice(None)].droplevel(1),
                  births_results_exc_2010, 100)

    target_cs_dict = {'double': True,
                      'first': {'year': 2010, 'value': 3.7, 'label': 'EmONC Surv. (2010)', 'ci': 0},
                      'second': {'year': 2015, 'value': 4, 'label': 'EMONC Surv. (2015)', 'ci': 0}}
    # todo: add bemonc estimates as well?

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, cs_data, target_cs_dict, '% of total births',
        'Proportion of total births delivered via caesarean section', graph_location, 'caesarean_section_rate')

    target_avd_dict = {'double': False,
                       'first': {'year': 2017, 'value': 1, 'label': 'HIV reports.', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, avd_data, target_avd_dict, 'Proportion of total births',
        'Assisted Vaginal Delivery Rate per Year', graph_location, 'avd_rate')

    proportions_dict_cs = dict()
    total_cs_per_year = list()

    for year in sim_years:
        yearly_mean_number = list()
        causes = dict()

        for indication in ['an_aph_pa', 'an_aph_pp', 'la_aph', 'ol', 'ur', 'spe_ec', 'other', 'previous_scar']:
            if indication in cs_results.loc[year].index:
                mean = cs_results.loc[year, indication].mean()
                yearly_mean_number.append(mean)
                causes.update({f'{indication}': mean})
            else:
                yearly_mean_number.append(0)

        total_cs_this_year = sum(yearly_mean_number)
        total_cs_per_year.append(total_cs_this_year)

        for indication in ['an_aph_pa', 'an_aph_pp', 'la_aph', 'ol', 'ur', 'spe_ec', 'other', 'previous_scar']:
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
    sepsis_df = an_comps.loc[(slice(None), 'clinical_chorioamnionitis'), slice(None)].droplevel(1) + \
                la_comps.loc[(slice(None), 'sepsis'), slice(None)].droplevel(1) + \
                pn_comps.loc[(slice(None), 'sepsis_postnatal'), slice(None)].droplevel(1) + \
                pn_comps.loc[(slice(None), 'sepsis'), slice(None)].droplevel(1)

    total_sep_rates = return_rate(sepsis_df, births_results_exc_2010, 1000)

    # todo: note, we would expect our rate to be higher than this
    target_sep_dict = {'double': True,
                       'first': {'year': 2010, 'value': 2.34, 'label': 'BEmONC 2010', 'ci': 0},
                       'second': {'year': 2015, 'value': 1.5, 'label': 'BEmONC 2015', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, total_sep_rates, target_sep_dict, 'Rate per 1000 births',
        'Rate of Maternal Sepsis per Year', graph_location, 'sepsis_rate')

    # ----------------------------------------- Postpartum Haemorrhage... ---------------------------------------------
    pph_data = pn_comps.loc[(slice(None), 'primary_postpartum_haemorrhage'), slice(None)].droplevel(1) + \
                pn_comps.loc[(slice(None), 'secondary_postpartum_haemorrhage'), slice(None)].droplevel(1)

    total_pph_rates = return_rate(pph_data, births_results_exc_2010, 1000)

    target_pph_dict = {'double': True,
                       'first': {'year': 2010, 'value': 7.95, 'label': 'BEmONC 2010', 'ci': 0},
                       'second': {'year': 2015, 'value': 12.8, 'label': 'BEmONC 2015', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, total_pph_rates, target_pph_dict, 'Rate per 1000 births',
        'Rate of Postpartum Haemorrhage per Year', graph_location, 'pph_rate')

    # ----------------------------------------- Fistula... -------------------------------------------------
    of_data = pn_comps.loc[(slice(None), 'vesicovaginal_fistula'), slice(None)].droplevel(1) + \
               pn_comps.loc[(slice(None), 'rectovaginal_fistula'), slice(None)].droplevel(1)


    total_fistula_rates = return_rate(of_data, births_results_exc_2010, 1000)

    target_fistula_dict = {'double': False,
                           'first': {'year': 2015, 'value': 6, 'label': 'DHS 2015', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, total_fistula_rates, target_fistula_dict, 'Rate per 1000 births',
        'Rate of Obstetric Fistula per Year', graph_location, 'fistula_rate')

    # ==================================================== NEWBORN OUTCOMES ==========================================
    #  ------------------------------------------- Neonatal sepsis (labour & postnatal) ------------------------------
    ns_df = nb_outcomes_df.loc[(slice(None), 'early_onset_sepsis'), slice(None)].droplevel(1) + \
                nb_outcomes_pn_df.loc[(slice(None), 'early_onset_sepsis'), slice(None)].droplevel(1) + \
                nb_outcomes_pn_df.loc[(slice(None), 'late_onset_sepsis'), slice(None)].droplevel(1)

    target_nsep_dict = {'double': False,
                        'first': {'year': 2020, 'value': 39.3, 'label': 'Fleischmann et al.', 'ci': 0}}

    total_ns_rates = return_rate(ns_df, births_results_exc_2010, 1000)

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, total_ns_rates, target_nsep_dict, 'Rate per 1000 births',
        'Rate of Neonatal Sepsis per year', graph_location, 'neo_sepsis_rate')

    #  ------------------------------------------- Neonatal encephalopathy ------------------------------------------
    ne_df = nb_outcomes_df.loc[(slice(None), 'mild_enceph'), slice(None)].droplevel(1) + \
            nb_outcomes_df.loc[(slice(None), 'moderate_enceph'), slice(None)].droplevel(1) + \
            nb_outcomes_df.loc[(slice(None), 'severe_enceph'), slice(None)].droplevel(1)

    total_enceph_rates = return_rate(ne_df, births_results_exc_2010, 1000)

    target_enceph_dict = {'double': True,
                          'first': {'year': 2010, 'value': 19.42, 'label': 'GBD 2010', 'ci': 0},
                          'second': {'year': 2015, 'value': 18.59, 'label': 'GBD 2015', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, total_enceph_rates, target_enceph_dict, 'Rate per 1000 births',
        'Rate of Neonatal Encephalopathy per year', graph_location, 'neo_enceph_rate')

    # ----------------------------------------- Respiratory Depression ------------------------------------------------
    rd_data = return_rate(nb_outcomes_df.loc[(slice(None), 'not_breathing_at_birth'), slice(None)].droplevel(1),
                  births_results_exc_2010, 1000)

    dummy_dict = {'double': False,
                  'first': {'year': 2010, 'value': 0, 'label': 'UNK.', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, rd_data, dummy_dict, 'Rate per 1000 births',
        'Rate of Neonatal Respiratory Depression per year', graph_location, 'neo_resp_depression_rate')

    # ----------------------------------------- Respiratory Distress Syndrome ------------------------------------------
    # ept = analysis_utility_functions.get_mean_and_quants_from_str_df(la_comps, 'early_preterm_labour', sim_years)[0]
    # # todo: should be live births
    # lpt = analysis_utility_functions.get_mean_and_quants_from_str_df(la_comps, 'late_preterm_labour', sim_years)[0]
    # total_ptbs = [x + y for x, y in zip(ept, lpt)]

    rds_data = return_rate(nb_outcomes_df.loc[(slice(None), 'respiratory_distress_syndrome'), slice(None)].droplevel(1),
                  births_results_exc_2010, 1000)

    target_rds_dict = {'double': False,
                       'first': {'year': 2019, 'value': 350, 'label': 'Muhe et al.', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, rds_data, target_rds_dict, 'Rate per 1000  births',
        'Rate of Preterm Respiratory Distress Syndrome per year', graph_location, 'neo_rds_rate')

    # - TOTAL NOT BREATHING NEWBORNS-
    total_not_breathing_df = \
        nb_outcomes_df.loc[(slice(None), 'respiratory_distress_syndrome'),slice(None)].droplevel(1) + \
        nb_outcomes_df.loc[(slice(None), 'not_breathing_at_birth'),slice(None)].droplevel(1) + ne_df

    nb_rate = return_rate(total_not_breathing_df, births_results_exc_2010, 1000)

    target_nb_dict = {'double': False,
                     'first': {'year': 2019, 'value': 100, 'label': 'Muhe et al.', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, nb_rate, target_nb_dict, 'Rate per 1000  births',
        'Rate of Preterm Respiratory Distress Syndrome per year', graph_location, 'neo_total_not_breathing')

    # TODO: add calibration target for 'apnea' which should be approx 5.7% total births

    # ----------------------------------------- Congenital Anomalies -------------------------------------------------
    rate_of_ca = return_rate(nb_outcomes_df.loc[(slice(None), 'congenital_heart_anomaly'), slice(None)].droplevel(1),
                  births_results_exc_2010, 1000)

    rate_of_laa = return_rate(
        nb_outcomes_df.loc[(slice(None), 'limb_or_musculoskeletal_anomaly'), slice(None)].droplevel(1),
        births_results_exc_2010, 1000)

    rate_of_ua = return_rate(
        nb_outcomes_df.loc[(slice(None), 'urogenital_anomaly'), slice(None)].droplevel(1),
        births_results_exc_2010, 1000)

    rate_of_da = return_rate(
        nb_outcomes_df.loc[(slice(None), 'digestive_anomaly'), slice(None)].droplevel(1),
        births_results_exc_2010, 1000)

    rate_of_oa = return_rate(
        nb_outcomes_df.loc[(slice(None), 'other_anomaly'), slice(None)].droplevel(1),
        births_results_exc_2010, 1000)

    plt.plot(sim_years, rate_of_ca[0], label="heart")
    plt.plot(sim_years, rate_of_laa[0], label="limb/musc.")
    plt.plot(sim_years, rate_of_ua[0], label="urogenital")
    plt.plot(sim_years, rate_of_da[0], label="digestive")
    plt.plot(sim_years, rate_of_oa[0], label="other")

    plt.xlabel('Year')
    plt.ylabel('Rate per 1000 births')
    plt.title('Yearly trends for Congenital Birth Anomalies')
    plt.legend()
    plt.savefig(f'{graph_location}/neo_rate_of_cong_anom.png')
    plt.show()

    # Breastfeeding
    # todo

    # ==================================== DEATH CALIBRATIONS =========================================================
    # Create folder to store graphs (if it hasnt already been created when ran previously)
    path = f'{outputspath}/calibration_output_graphs_{pop_size}_{results_folder.name}/death'
    if not os.path.isdir(path):
        os.makedirs(f'{outputspath}/calibration_output_graphs_{pop_size}_{results_folder.name}/death')

    graph_location = path

    # read in daly data
    dalys_data = pd.read_csv(Path('./resources/gbd') / 'ResourceFile_Deaths_and_DALYS_GBD2019.CSV')
    gbd_years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]

    # =========================================  Direct maternal causes of death... ===================================
    direct_causes = ['ectopic_pregnancy', 'spontaneous_abortion', 'induced_abortion',
                     'severe_gestational_hypertension', 'severe_pre_eclampsia', 'eclampsia', 'antenatal_sepsis',
                     'uterine_rupture', 'intrapartum_sepsis', 'postpartum_sepsis', 'postpartum_haemorrhage',
                     'secondary_postpartum_haemorrhage', 'antepartum_haemorrhage']

    # ==============================================  YEARLY MMR... ==================================================
    # Output direct deaths...
    def get_maternal_death_dfs(scaled):
        direct_deaths = extract_results(
            results_folder,
            module="tlo.methods.demography",
            key="death",
            custom_generate_series=(
                lambda df: df.loc[(df['label'] == 'Maternal Disorders')].assign(
                    year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=scaled)
        direct_deaths_final = direct_deaths.fillna(0)

        indirect_deaths_non_hiv = extract_results(
            results_folder,
            module="tlo.methods.demography.detail",
            key="properties_of_deceased_persons",
            custom_generate_series=(
                lambda df: df.loc[(df['is_pregnant'] | df['la_is_postpartum']) &
                                  (df['cause_of_death'].str.contains('Malaria|Suicide|ever_stroke|diabetes|'
                                                                     'chronic_ischemic_hd|ever_heart_attack|'
                                                                     'chronic_kidney_disease') |
                                   (df['cause_of_death'] == 'TB'))].assign(
                    year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=scaled)
        indirect_deaths_non_hiv_final = indirect_deaths_non_hiv.fillna(0)

        hiv_pd = extract_results(
            results_folder,
            module="tlo.methods.demography.detail",
            key="properties_of_deceased_persons",
            custom_generate_series=(
                lambda df: df.loc[(df['is_pregnant'] | df['la_is_postpartum']) &
                                  (df['cause_of_death'].str.contains('AIDS_non_TB|AIDS_TB'))].assign(
                    year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=scaled)

        hiv_pd_fill = hiv_pd.fillna(0)
        hiv_indirect = hiv_pd_fill * 0.3
        hiv_indirect_maternal_deaths = hiv_indirect.round(0)

        indirect_deaths_final = indirect_deaths_non_hiv_final + hiv_indirect_maternal_deaths

        total_deaths = direct_deaths_final + indirect_deaths_final

        return {'direct_deaths_final':direct_deaths_final,
                'indirect_deaths_final': indirect_deaths_final,
                'total_deaths':total_deaths}

    mat_d_unscaled = get_maternal_death_dfs(False)

    direct_mmr_by_year = return_rate(mat_d_unscaled['direct_deaths_final'], births_results_exc_2010, 100_000)
    indirect_mmr_by_year = return_rate(mat_d_unscaled['indirect_deaths_final'], births_results_exc_2010, 100_000)
    total_mmr_by_year = return_rate(mat_d_unscaled['total_deaths'], births_results_exc_2010, 100_000)

    unmmeig = [513, 496, 502, 473, 442, 445, 430, 375, 392, 370, 381]
    unmmeig_lower = [408, 394, 397, 374, 349, 348, 330, 283, 291, 268, 269]
    unmmeig_upper = [659, 642, 654, 620, 583, 592, 577, 508, 537, 515, 543]
    unmmeig_yrs = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
    dhs_10 = [675, 570, 780]
    dhs_15 = [439, 348, 531]
    gbd = [242, 235, 229, 223, 219, 219, 217, 214, 209]
    gbd_l = [168, 165, 158, 151, 150, 146, 141, 141, 134]
    gbd_u = [324, 313, 310, 307, 304, 307, 304, 300, 294]
    gbd_years_mmr = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]

    for data, title in zip([direct_mmr_by_year, indirect_mmr_by_year, total_mmr_by_year],
                           ['Direct', 'Indirect', 'Total']):

        if title == 'Direct':
            mp = 0.7
        elif title == 'Indirect':
            mp = 0.3
        else:
            mp = 1

        unmmeig_adj = [x * mp for x in unmmeig]
        unmmeig_l_adj = [x * mp for x in unmmeig_lower]
        unmmeig_u_adj = [x * mp for x in unmmeig_upper]
        dhs_10_adj = [x * mp for x in dhs_10]
        dhs_15_adj = [x * mp for x in dhs_15]
        gbd_adj = [x * mp for x in gbd]
        gbd_l_adj = [x * mp for x in gbd_l]
        gbd_u_adj = [x * mp for x in gbd_u]

        fig, ax = plt.subplots()
        ax.plot(sim_years, data[0], label="Model (95% CI)", color='deepskyblue')
        ax.fill_between(sim_years, data[1], data[2], color='b', alpha=.1)
        plt.errorbar(2010, dhs_10_adj[0], yerr=(dhs_10_adj[2]-dhs_10_adj[1])/2, label='DHS 2010 (95% CI)', fmt='o',
                     color='green',
                     ecolor='mediumseagreen',
                     elinewidth=3, capsize=0)
        plt.errorbar(2015, dhs_15_adj[0], yerr=(dhs_15_adj[2]-dhs_15_adj[1])/2, label='DHS 2015 (95% CI)', fmt='o',
                     color='green',
                     ecolor='mediumseagreen',
                     elinewidth=3, capsize=0)
        ax.plot(unmmeig_yrs, unmmeig_adj, label="UN MMEIG 2020 (Uncertainty interval)", color='cadetblue')
        ax.fill_between(unmmeig_yrs, unmmeig_l_adj, unmmeig_u_adj, color='cadetblue', alpha=.1)
        ax.plot(gbd_years_mmr, gbd_adj, label="GBD 2019 (lower & upper bounds)", color='darkslateblue')
        ax.fill_between(gbd_years_mmr, gbd_l_adj, gbd_u_adj, color='slateblue',alpha=.1)

        if title == 'Total':
            ax.set(ylim=(0, 900))
        else:
            ax.set(ylim=(0, 400))
        plt.xlabel('Year')
        plt.ylabel("Deaths per 100 000 live births")
        plt.title(f'{title} Maternal Mortality Ratio per Year')
        plt.legend()
        plt.savefig(f'{graph_location}/{title}_mmr.png')
        plt.show()

    labels = sim_years
    width = 0.35  # the width of the bars: can also be len(x) sequence
    fig, ax = plt.subplots()
    ax.bar(labels, direct_mmr_by_year[0], width, label='Direct', color='brown')
    ax.bar(labels, indirect_mmr_by_year[0], width, bottom=direct_mmr_by_year[0], label='Indirect', color='lightsalmon')
    ax.set_ylabel('Maternal Deaths per 100,000 live births')
    ax.set_title('Total Maternal Mortality Ratio per Year')
    ax.legend()
    plt.savefig(f'{graph_location}/total_mmr_bar.png')
    plt.show()

    # ---------------------------------------- PROPORTION OF INDIRECT DEATHS BY CAUSE --------------------------------
    indirect_deaths_by_cause = extract_results(
        results_folder,
        module="tlo.methods.demography.detail",
        key="properties_of_deceased_persons",
        custom_generate_series=(
            lambda df: df.loc[(df['is_pregnant'] | df['la_is_postpartum']) &
                              (df['cause_of_death'].str.contains(
                                  'AIDS_non_TB|AIDS_TB|Malaria|Suicide|ever_stroke|diabetes|chronic_ischemic_hd|'
                                  'ever_heart_attack|chronic_kidney_disease') | (df['cause_of_death'] == 'TB'))].assign(
                year=df['date'].dt.year).groupby(['year', 'cause_of_death'])['year'].count()))
    id_by_cause_df = indirect_deaths_by_cause.fillna(0)

    indirect_causes = ['AIDS_non_TB', 'AIDS_TB', 'TB', 'Malaria', 'Suicide', 'ever_stroke', 'diabetes',
                       'chronic_ischemic_hd', 'ever_heart_attack', 'chronic_kidney_disease']

    indirect_deaths_means = {}

    for complication in indirect_causes:
        indirect_deaths_means.update({complication: []})

        for year in sim_years:
            if complication in id_by_cause_df.loc[year].index:
                births = births_results_exc_2010.loc[year].mean()
                deaths = id_by_cause_df.loc[year, complication].mean()
                if 'AIDS' in complication:
                    deaths = deaths * 0.3
                indirect_deaths_means[complication].append((deaths/births) * 100000)
            else:
                indirect_deaths_means[complication].append(0)

    labels = sim_years
    width = 0.35  # the width of the bars: can also be len(x) sequence
    fig, ax = plt.subplots()

    ax.bar(labels, indirect_deaths_means['AIDS_TB'], width, label='AIDS_TB',
           bottom=[a + b + c + d + e + f + g + h + i for a, b, c, d, e, f, g, h, i in zip(
               indirect_deaths_means['AIDS_non_TB'],
               indirect_deaths_means['Malaria'],
               indirect_deaths_means['TB'],
               indirect_deaths_means['Suicide'],
               indirect_deaths_means['ever_stroke'],
               indirect_deaths_means['diabetes'],
               indirect_deaths_means['chronic_ischemic_hd'],
               indirect_deaths_means['ever_heart_attack'],
               indirect_deaths_means['chronic_kidney_disease'])],
           color='yellow')

    ax.bar(labels, indirect_deaths_means['chronic_kidney_disease'], width, label='CKD',
           bottom=[a + b + c + d + e + f + g + h for a, b, c, d, e, f, g, h in zip(
               indirect_deaths_means['AIDS_non_TB'],
               indirect_deaths_means['Malaria'],
               indirect_deaths_means['TB'],
               indirect_deaths_means['Suicide'],
               indirect_deaths_means['ever_stroke'],
               indirect_deaths_means['diabetes'],
               indirect_deaths_means['chronic_ischemic_hd'],
               indirect_deaths_means['ever_heart_attack'], )],
           color='pink')

    ax.bar(labels, indirect_deaths_means['ever_heart_attack'], width, label='MI',
           bottom=[a+b+c+d+e+f+g for a, b, c, d, e, f, g in zip(indirect_deaths_means['AIDS_non_TB'],
                                                                indirect_deaths_means['Malaria'],
                                                                indirect_deaths_means['TB'],
                                                                indirect_deaths_means['Suicide'],
                                                                indirect_deaths_means['ever_stroke'],
                                                                indirect_deaths_means['diabetes'],
                                                                indirect_deaths_means['chronic_ischemic_hd'])],
           color='darkred')

    ax.bar(labels, indirect_deaths_means['chronic_ischemic_hd'], width, label='Chronic HD',
           bottom=[a+b+c+d+e+f for a, b, c, d, e, f in zip(indirect_deaths_means['AIDS_non_TB'],
                                                           indirect_deaths_means['Malaria'],
                                                           indirect_deaths_means['TB'],
                                                           indirect_deaths_means['Suicide'],
                                                           indirect_deaths_means['ever_stroke'],
                                                           indirect_deaths_means['diabetes'])], color='grey')

    ax.bar(labels, indirect_deaths_means['diabetes'], width, label='Diabetes',
           bottom=[a+b+c+d+e for a, b, c, d, e in zip(indirect_deaths_means['AIDS_non_TB'],
                                                      indirect_deaths_means['Malaria'],
                                                      indirect_deaths_means['TB'], indirect_deaths_means['Suicide'],
                                                      indirect_deaths_means['ever_stroke'])], color='darkorange')

    ax.bar(labels, indirect_deaths_means['ever_stroke'], width, label='Stoke',
           bottom=[a + b + c + d for a, b, c, d in zip(indirect_deaths_means['AIDS_non_TB'],
                                                       indirect_deaths_means['Malaria'],
                                                       indirect_deaths_means['TB'],
                                                       indirect_deaths_means['Suicide'])], color='yellowgreen')
    ax.bar(labels, indirect_deaths_means['Suicide'], width, label='Suicide',
           bottom=[a + b + c for a, b, c in zip(indirect_deaths_means['AIDS_non_TB'],
                                                indirect_deaths_means['Malaria'],
                                                indirect_deaths_means['TB'])], color='cornflowerblue')

    ax.bar(labels, indirect_deaths_means['TB'], width, label='TB',
           bottom=[a + b for a, b in zip(indirect_deaths_means['AIDS_non_TB'],
                                         indirect_deaths_means['Malaria'])],
           color='darkmagenta')

    ax.bar(labels, indirect_deaths_means['Malaria'], width, label='Malaria',
           bottom=indirect_deaths_means['AIDS_non_TB'],
           color='slategrey')
    ax.bar(labels, indirect_deaths_means['AIDS_non_TB'], width, label='AIDS_non_TB', color='hotpink')

    ax.set(ylim=(0, 350))
    ax.set_ylabel('Deaths per 100,000 live births')
    ax.set_xlabel('Year')
    ax.set_title('Indirect causes of maternal death within the TLO model')
    ax.legend()
    plt.savefig(f'{graph_location}/indirect_death_mmr_cause.png')
    plt.show()

    # ==============================================  DEATHS... ======================================================
    mat_d_scaled = get_maternal_death_dfs(True)

    m_deaths = analysis_utility_functions.return_95_CI_across_runs(mat_d_scaled['total_deaths'], gbd_years)
    # m_deaths = analysis_utility_functions.return_95_CI_across_runs(
    #     scaled_deaths.loc[(slice(None), 'Maternal Disorders'), slice(None)].droplevel(1), gbd_years)
    def extract_deaths_gbd_data(group):
        dalys_df = dalys_data.loc[(dalys_data['measure_name'] == 'Deaths') &
                                  (dalys_data['cause_name'] == group) & (dalys_data['Year'] > 2009)]
        gbd_deaths = list()
        gbd_deaths_lq = list()
        gbd_deaths_uq = list()

        for year in gbd_years:
            gbd_deaths.append(dalys_df.loc[(dalys_df['Year'] == year)].sum()['GBD_Est'])
            gbd_deaths_lq.append(dalys_df.loc[(dalys_df['Year'] == year)].sum()['GBD_Lower'])
            gbd_deaths_uq.append(dalys_df.loc[(dalys_df['Year'] == year)].sum()['GBD_Upper'])

        return [gbd_deaths, gbd_deaths_lq, gbd_deaths_uq]

    gbd_deaths_2010_2019_data = extract_deaths_gbd_data('Maternal disorders')

    model_ci = [(x - y) / 2 for x, y in zip(m_deaths[2], m_deaths[1])]
    gbd_ci = [(x - y) / 2 for x, y in zip(gbd_deaths_2010_2019_data[2], gbd_deaths_2010_2019_data[1])]

    N = len(m_deaths[0])
    ind = np.arange(N)
    width = 0.35
    plt.bar(ind, m_deaths[0], width, label='Model (95% CI)', yerr=model_ci, color='teal')
    plt.bar(ind + width, gbd_deaths_2010_2019_data[0], width, label='GBD (upper & lower bounds)', yerr=gbd_ci,
            color='olivedrab')
    plt.ylabel('Total Deaths Maternal Deaths (scaled)')
    plt.title('Yearly Modelled Maternal Deaths Compared to GBD')
    plt.xticks(ind + width / 2, gbd_years)
    plt.legend(loc='best')
    plt.savefig(f'{graph_location}/deaths_gbd_comparison.png')
    plt.show()

    # =================================== COMPLICATION LEVEL MMR ======================================================
    d_r = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'cause'])['year'].count()
        ),
    )
    death_results = d_r.fillna(0)
    simplified_causes = ['ectopic_pregnancy', 'abortion', 'severe_pre_eclampsia', 'sepsis', 'uterine_rupture',
                         'postpartum_haemorrhage',  'antepartum_haemorrhage']

    ec_tr = {'double': True, 'first': {'year': 2010, 'value': 18.9, 'label': 'UNK.', 'ci': 0},
                             'second': {'year': 2015, 'value': 3.51, 'label': 'UNK.', 'ci': 0}}
    ab_tr = {'double': True, 'first': {'year': 2010, 'value': 51.3, 'label': 'UNK.', 'ci': 0},
             'second': {'year': 2015, 'value': 29.9, 'label': 'UNK.', 'ci': 0}}
    spe_ec_tr = {'double': True, 'first': {'year': 2010, 'value': 64.8, 'label': 'UNK.', 'ci': 0},
                                 'second': {'year': 2015, 'value': 69.8, 'label': 'UNK.', 'ci': 0}}
    sep_tr = {'double': True, 'first': {'year': 2010, 'value': 74.3, 'label': 'UNK.', 'ci': 0},
             'second': {'year': 2015, 'value': 55.3, 'label': 'UNK.', 'ci': 0}}
    ur_tr = {'double': True, 'first': {'year': 2010, 'value': 18.9, 'label': 'UNK.', 'ci': 0},
             'second': {'year': 2015, 'value': 3.51, 'label': 'UNK.', 'ci': 0}}
    pph_tr = {'double': True, 'first': {'year': 2010, 'value': 229.5, 'label': 'UNK.', 'ci': 0},
             'second': {'year': 2015, 'value': 116.8, 'label': 'UNK.', 'ci': 0}}
    aph_tr = {'double': True, 'first': {'year': 2010, 'value': 47.3, 'label': 'UNK.', 'ci': 0},
             'second': {'year': 2015, 'value': 23.3, 'label': 'UNK.', 'ci': 0}}

    trs = [ec_tr, ab_tr, spe_ec_tr, sep_tr, ur_tr, pph_tr, aph_tr]

    for cause, tr in zip(simplified_causes, trs):
        if (cause == 'ectopic_pregnancy') or (cause == 'antepartum_haemorrhage') or (cause == 'uterine_rupture'):
            # deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(death_results, cause, sim_years)[0]
            deaths = death_results.loc[(slice(None), cause), slice(None)].droplevel(1)

        elif cause == 'abortion':
            ia_deaths = death_results.loc[(slice(None), 'induced_abortion'), slice(None)].droplevel(1)
            sa_deaths = death_results.loc[(slice(None), 'spontaneous_abortion'), slice(None)].droplevel(1)
            deaths = ia_deaths + sa_deaths

        elif cause == 'severe_pre_eclampsia':
            spe_deaths = death_results.loc[(slice(None), 'severe_pre_eclampsia'), slice(None)].droplevel(1)
            ec_deaths = death_results.loc[(slice(None), 'eclampsia'), slice(None)].droplevel(1)
            # TODO:- FIX SGH ISSUE
            # sgh_deaths = death_results.loc[(slice(None), 'severe_gestational_hypertension'), slice(None)].droplevel(1)
            deaths = spe_deaths + ec_deaths

        elif cause == 'postpartum_haemorrhage':
            p_deaths = death_results.loc[(slice(None), 'postpartum_haemorrhage'), slice(None)].droplevel(1)
            s_deaths = death_results.loc[(slice(None), 'secondary_postpartum_haemorrhage'), slice(None)].droplevel(1)
            deaths = p_deaths + s_deaths

        elif cause == 'sepsis':
            a_deaths = death_results.loc[(slice(None), 'antenatal_sepsis'), slice(None)].droplevel(1)
            i_deaths = death_results.loc[(slice(None), 'intrapartum_sepsis'), slice(None)].droplevel(1)
            p_deaths = death_results.loc[(slice(None), 'postpartum_sepsis'), slice(None)].droplevel(1)
            deaths = a_deaths + i_deaths + p_deaths

        mmr = return_rate(deaths, births_results_exc_2010, 100_000)
        analysis_utility_functions.line_graph_with_ci_and_target_rate(
            sim_years, mmr, tr, 'Rate per 100,000 births',
            f'Maternal Mortality Ratio per Year for {cause}', graph_location, f'mmr_{cause}')

    # =================================== DEATH PROPORTIONS... ========================================================
    proportions_dicts = dict()
    total_deaths_per_year = list()

    for year in sim_years:
        yearly_mean_number = list()
        causes = dict()

        for complication in direct_causes:
            if complication in death_results.loc[year].index:
                mean = death_results.loc[year, complication].mean()
                yearly_mean_number.append(mean)
                causes.update({f'{complication}': mean})
            else:
                yearly_mean_number.append(0)

        total_deaths_this_year = sum(yearly_mean_number)
        total_deaths_per_year.append(total_deaths_this_year)

        for complication in causes:
            causes[complication] = (causes[complication] / total_deaths_this_year) * 100
        new_dict = {year: causes}
        proportions_dicts.update(new_dict)
    #
    # def pie_prop_cause_of_death(values, years, labels, title):
    #     sizes = values
    #     fig1, ax1 = plt.subplots()
    #     ax1.pie(sizes, shadow=True, startangle=90)
    #     ax1.axis('equal')
    #     box = ax.get_position()
    #     ax.set_position([box.x0, box.y0, box.width * 0.5, box.height])
    #     plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5))
    #     # Equal aspect ratio ensures that pie is drawn as a circle.
    #     plt.title(f'Proportion of total maternal deaths by cause ({title}) {years}')
    #     plt.savefig(f'{graph_location}/mat_death_by_cause_{title}_{years}.png',
    #                 bbox_inches="tight")
    #     plt.show()

    props_df = pd.DataFrame(data=proportions_dicts)
    props_df = props_df.fillna(0)

    simplified_df = props_df.transpose()

    simplified_df['Abortion'] = simplified_df['induced_abortion'] + simplified_df['spontaneous_abortion']
    simplified_df['Severe PE/Eclampsia'] = simplified_df['severe_pre_eclampsia'] + simplified_df['eclampsia']
    simplified_df['PPH'] = simplified_df['postpartum_haemorrhage'] + simplified_df['secondary_postpartum_haemorrhage']

    simplified_df['Sepsis'] = pd.Series(0, index=sim_years)
    if 'postpartum_sepsis' in simplified_df.columns:
        simplified_df['Sepsis'] = simplified_df['Sepsis'] + simplified_df['postpartum_sepsis']
    if 'intrapartum_sepsis' in simplified_df.columns:
        simplified_df['Sepsis'] = simplified_df['Sepsis'] + simplified_df['intrapartum_sepsis']
    if 'antenatal_sepsis' in simplified_df.columns:
        simplified_df['Sepsis'] = simplified_df['Sepsis'] + simplified_df['antenatal_sepsis']

    for column in ['postpartum_haemorrhage', 'secondary_postpartum_haemorrhage', 'severe_pre_eclampsia', 'eclampsia',
                   'severe_gestational_hypertension',
                   'induced_abortion', 'spontaneous_abortion', 'intrapartum_sepsis', 'postpartum_sepsis',
                   'antenatal_sepsis']:
        if column in simplified_df.columns:
            simplified_df = simplified_df.drop(columns=[column])

    all_labels = list()
    all_values = list()
    av_l_ci = list()
    av_u_ci = list()
    for column in simplified_df.columns:
        all_values.append(simplified_df[column].mean())
        ci = st.t.interval(0.95, len(simplified_df[column]) - 1, loc=np.mean(simplified_df[column]),
                           scale=st.sem(simplified_df[column]))
        av_l_ci.append(ci[0])
        av_u_ci.append(ci[1])
        all_labels.append(f'{column} ({round(simplified_df[column].mean(), 2)} %)')

    # pie_prop_cause_of_death(all_values, '2010-2020', all_labels, 'total')

    labels = ['EP', 'UR', 'APH', 'Abr.', 'SPE/E.', 'PPH', 'Sep.']
    model = all_values
    bemonc_data = [0.8, 12.6, 5.3, 6.8, 15.9, 26, 18.9]  # order = ectopic,
    ui = [(x - y) / 2 for x, y in zip(av_u_ci, av_l_ci)]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, model, width, yerr=ui, label='Model (95% CI)')
    ax.bar(x + width / 2, bemonc_data, width, label='BEmONC Survey 2015')
    # ax.bar_label(labels)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('% of total direct deaths')
    ax.set_xlabel('Cause of death')
    ax.set_title('Proportion of total direct maternal deaths by cause (2010-2022)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper left')
    fig.tight_layout()
    plt.savefig(f'{graph_location}/proportions_cause_of_death.png')
    plt.show()

    # =========================================== CASE FATALITY PER COMPLICATION ======================================
    # tr = list()  # todo:update?
    # dummy_denom = list()
    # for years in sim_years:
    #     tr.append(0)
    #     dummy_denom.append(1)
    #
    # mean_ep = analysis_utility_functions.get_mean_and_quants_from_str_df(
    #     an_comps, 'ectopic_unruptured', sim_years)[0]
    # mean_sa = analysis_utility_functions.get_mean_and_quants_from_str_df(
    #     an_comps, 'complicated_spontaneous_abortion', sim_years)[0]
    # mean_ia = analysis_utility_functions.get_mean_and_quants_from_str_df(
    #     an_comps, 'complicated_induced_abortion', sim_years)[0]
    # mean_ur = analysis_utility_functions.get_mean_and_quants_from_str_df(
    #     la_comps, 'uterine_rupture', sim_years)[0]
    # mean_lsep = analysis_utility_functions.get_mean_and_quants_from_str_df(
    #     la_comps, 'sepsis', sim_years)[0]
    # mean_psep = analysis_utility_functions.get_mean_and_quants_from_str_df(
    #     pn_comps, 'sepsis', sim_years)[0]
    # mean_asep = analysis_utility_functions.get_mean_and_quants_from_str_df(
    #     an_comps, 'clinical_chorioamnionitis', sim_years)[0]
    #
    # mean_ppph = analysis_utility_functions.get_mean_and_quants_from_str_df(
    #     la_comps, 'primary_postpartum_haemorrhage', sim_years)[0]
    # mean_spph = analysis_utility_functions.get_mean_and_quants_from_str_df(
    #     pn_comps, 'secondary_postpartum_haemorrhage', sim_years)[0]
    #
    # mean_spe = analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
    #     'severe_pre_eclamp', dummy_denom, 1, [an_comps, la_comps, pn_comps], sim_years)[0]
    #
    # mean_ec = analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
    #     'eclampsia', dummy_denom, 1, [an_comps, la_comps, pn_comps], sim_years)[0]
    #
    # mean_sgh = analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
    #     'severe_gest_htn', dummy_denom, 1, [an_comps, la_comps, pn_comps], sim_years)[0]
    #
    # mm_aph_mean = analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
    #     'mild_mod_antepartum_haemorrhage', dummy_denom, 1, [an_comps, la_comps], sim_years)[0]
    #
    # s_aph_mean = analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
    #     'severe_antepartum_haemorrhage', dummy_denom, 1, [an_comps, la_comps], sim_years)[0]
    #
    # mean_aph = [x + y for x, y in zip(mm_aph_mean, s_aph_mean)]
    #
    # for inc_list in [mean_ep, mean_sa, mean_ia, mean_ur, mean_lsep,
    #                  mean_psep, mean_asep, mean_ppph, mean_spph, mean_spe, mean_ec, mean_sgh, mean_aph]:
    #
    #     for index, item in enumerate(inc_list):
    #         if item == 0:
    #             inc_list[index] = 0.1
    #
    #     print(inc_list)
    #
    # for inc_list, complication in \
    #     zip([mean_ep, mean_sa, mean_ia, mean_ur, mean_psep, mean_ppph, mean_spph, mean_spe, mean_ec,
    #          mean_sgh, mean_aph],
    #         ['ectopic_pregnancy', 'spontaneous_abortion', 'induced_abortion', 'uterine_rupture',
    #          'postpartum_sepsis', 'postpartum_haemorrhage', 'secondary_postpartum_haemorrhage',
    #          'severe_pre_eclampsia', 'eclampsia', 'severe_gestational_hypertension', 'antepartum_haemorrhage']):
    #
    #     cfr = analysis_utility_functions.get_comp_mean_and_rate(
    #         complication, inc_list, death_results, 100, sim_years)[0]
    #     print(complication, cfr)
    #     analysis_utility_functions.simple_line_chart_with_target(
    #         sim_years, cfr, tr, 'Total CFR', f'Yearly CFR for {complication}', f'{complication}_cfr_per_year',
    #         graph_location)
    #
    # mean_lsep = analysis_utility_functions.get_mean_and_quants_from_str_df(la_comps, 'sepsis', sim_years)[0]
    # mean_asep = analysis_utility_functions.get_mean_and_quants_from_str_df(
    #     an_comps, 'clinical_chorioamnionitis', sim_years)[0]
    # total_an_cases = [x + y for x, y in zip(mean_asep, mean_lsep)]
    #
    # a_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
    #     death_results, 'antenatal_sepsis', sim_years)[0]
    # i_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
    #     death_results, 'intrapartum_sepsis', sim_years)[0]
    #
    # total_an_sepsis_deaths = [x + y for x, y in zip(a_deaths, i_deaths)]
    # an_sep_cfr = [(x/y) * 100 for x, y in zip(total_an_sepsis_deaths, total_an_cases)]
    # analysis_utility_functions.simple_line_chart_with_target(
    #     sim_years, an_sep_cfr, tr, 'Total CFR', 'Yearly CFR for antenatal/intrapartum sepsis',
    #     'an_ip_sepsis_cfr_per_year', graph_location)
    #
    # # todo: issue with incidenec and logging of sepsis
    # total_sepsis_cases = [x + y for x, y in zip(total_an_cases, mean_psep)]
    # p_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(death_results, 'postpartum_sepsis',
    #                                                                       sim_years)[0]
    # total_sepsis_deaths = [x + y for x, y in zip(p_deaths, total_an_sepsis_deaths)]
    # sep_cfr = [(x/y) * 100 for x, y in zip(total_sepsis_deaths, total_sepsis_cases)]
    # analysis_utility_functions.simple_line_chart_with_target(
    #     sim_years, sep_cfr, tr, 'Total CFR', 'Yearly CFR for Sepsis (combined)', 'combined_sepsis_cfr_per_year',
    #     graph_location)
    #
    # total_pph_cases = [x + y for x, y in zip(mean_ppph, mean_spph)]
    # p_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
    #     death_results, 'postpartum_haemorrhage', sim_years)[0]
    # s_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
    #     death_results, 'secondary_postpartum_haemorrhage', sim_years)[0]
    # total_pph_deaths = [x + y for x, y in zip(p_deaths, s_deaths)]
    # cfr = [(x/y) * 100 for x, y in zip(total_pph_deaths, total_pph_cases)]
    # analysis_utility_functions.simple_line_chart_with_target(
    #     sim_years, cfr, tr, 'Total CFR', 'Yearly CFR for PPH (combined)', 'combined_pph_cfr_per_year', graph_location)
    #
    # total_ab_cases = [x + y for x, y in zip(mean_ia, mean_sa)]
    # ia_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
    #     death_results, 'induced_abortion', sim_years)[0]
    # sa_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
    #     death_results, 'spontaneous_abortion', sim_years)[0]
    # total_ab_deaths = [x + y for x, y in zip(ia_deaths, sa_deaths)]
    # cfr = [(x/y) * 100 for x, y in zip(total_ab_deaths, total_ab_cases)]
    # analysis_utility_functions.simple_line_chart_with_target(
    #     sim_years, cfr, tr, 'Total CFR', 'Yearly CFR for Abortion (combined)', 'combined_abortion_cfr_per_year',
    #     graph_location)
    #
    # total_spec_cases = [x + y + z for x, y, z in zip(mean_spe, mean_ec, mean_sgh)]
    # spe_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
    #     death_results, 'severe_pre_eclampsia', sim_years)[0]
    # ec_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
    #     death_results, 'eclampsia', sim_years)[0]
    # total_spec_deaths = [x + y + z for x, y, z in zip(spe_deaths, ec_deaths, sgh_deaths)]
    # cfr = [(x/y) * 100 for x, y in zip(total_spec_deaths, total_spec_cases)]
    # analysis_utility_functions.simple_line_chart_with_target(
    #     sim_years, cfr, tr, 'Total CFR', 'Yearly CFR for Severe Pre-eclampsia/Eclampsia',
    #     'combined_spe_ec_cfr_per_year', graph_location)

    # ================================================================================================================
    # =================================================== Neonatal Death ==============================================
    # ================================================================================================================

    # ----------------------------------------------- NEONATAL MORTALITY RATE ----------------------------------------
    # NMR due to neonatal disorders...
    # NEONATAL DISORDERS NMR - ROUGHLY EQUATES TO FIRST WEEK NMR
    death_results_labels = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'label'])['year'].count()))
    nd_nmr = return_rate(death_results_labels.loc[(slice(None), 'Neonatal Disorders'), slice(None)].droplevel(1),
                         births_results_exc_2010, 1000)

    # Total NMR...(FROM ALL CAUSES UP TO 28 DAYS)
    tnd = extract_results(
        results_folder,
        module="tlo.methods.demography.detail",
        key="properties_of_deceased_persons",
        custom_generate_series=(
            lambda df: df.loc[(df['age_days'] <= 28)].assign(
                year=df['date'].dt.year).groupby(['year'])['year'].count()))
    total_neonatal_deaths = tnd.fillna(0)
    tnmr = return_rate(total_neonatal_deaths, births_results_exc_2010, 1000)

    def get_nmr_graphs(data, colours, title, save_name):
        fig, ax = plt.subplots()
        ax.plot(sim_years, data[0], label="Model (95% CI)", color=colours[0])
        ax.fill_between(sim_years, data[1], data[2], color=colours[1], alpha=.1)
        un_yrs = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]

        data = {'dhs': {'mean': [31, 27], 'lq': [26, 22], 'uq': [34, 34]},
                     # 'hug': {'mean': 22.7, 'lq': 15.6, 'uq': 28.8},
                     # 'gbd': {'mean': 25, 'lq': 21.4, 'uq': 26.6},
                     'un': {'mean': [28, 27, 26, 25, 24, 23, 22, 22, 20, 20, 20, 19],
                            'lq': [25, 24, 23, 22, 20, 18, 16, 15, 14, 13, 13, 12],
                            'uq': [31, 31, 30, 29, 29, 28, 28, 29, 29, 30, 30, 31]}}

        plt.errorbar(2010, data['dhs']['mean'][0], label='DHS 2010 (95% CI)',
                     yerr=(data['dhs']['uq'][0] - data['dhs']['lq'][0]) / 2, fmt='o', color='mediumseagreen',
                     ecolor='green',
                     elinewidth=3, capsize=0)

        plt.errorbar(2015, data['dhs']['mean'][1], label='DHS 2015 (95% CI)',
                     yerr=(data['dhs']['uq'][1] - data['dhs']['lq'][1]) / 2, fmt='o', color='mediumseagreen',
                     ecolor='green',
                     elinewidth=3, capsize=0)

        # plt.errorbar(2017, data['hug']['mean'], label='Hug (2017)',
        #              yerr=(data['hug']['uq'] - data['hug']['lq']) / 2,
        #              fmt='o', color='purple', ecolor='purple', elinewidth=3, capsize=0)
        #
        # plt.errorbar(2019, data['gbd']['mean'], label='GBD (2019)',
        #              yerr=(data['gbd']['uq'] - data['gbd']['lq']) / 2,
        #              fmt='o', color='pink', ecolor='pink', elinewidth=3, capsize=0)

        ax.plot(un_yrs, data['un']['mean'], label="UN IGCME (Uncertainty interval)", color='grey')
        ax.fill_between(un_yrs, data['un']['lq'], data['un']['uq'], color='grey', alpha=.1)
        ax.set(ylim=(0, 35))
        plt.xlabel('Year')
        plt.ylabel("Neonatal deaths per 1000 live births")
        plt.title(title)
        plt.legend(loc='lower left')
        plt.savefig(f'{graph_location}/{save_name}.png')
        plt.show()

    get_nmr_graphs(tnmr, ['deepskyblue', 'b'], 'Yearly Total Neonatal Mortality Rate', 'total_nmr')
    get_nmr_graphs(nd_nmr, ['deepskyblue', 'b'], 'Yearly NMR due to GBD "Neonatal Disorders"', 'neonatal_disorders_nmr')

    fig, ax = plt.subplots()
    ax.plot(sim_years, tnmr[0], label="Model-Total NMR(95% CI)", color='deepskyblue')
    ax.fill_between(sim_years, tnmr[1], tnmr[2], color='b', alpha=.1)
    ax.plot(sim_years, nd_nmr[0], label="Model-ND NMR (95% CI)", color='salmon')
    ax.fill_between(sim_years, nd_nmr[1], nd_nmr[2], color='r', alpha=.1)

    un_yrs = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]

    data = {'dhs': {'mean': [31, 27], 'lq': [26, 22], 'uq': [34, 34]},
            'un': {'mean': [28, 27, 26, 25, 24, 23, 22, 22, 20, 20, 20, 19],
                   'lq': [25, 24, 23, 22, 20, 18, 16, 15, 14, 13, 13, 12],
                   'uq': [31, 31, 30, 29, 29, 28, 28, 29, 29, 30, 30, 31]}}

    plt.errorbar(2010, data['dhs']['mean'][0], label='DHS 2010 (95% CI)',
                 yerr=(data['dhs']['uq'][0] - data['dhs']['lq'][0]) / 2, fmt='o', color='mediumseagreen',
                 ecolor='green',
                 elinewidth=3, capsize=0)

    plt.errorbar(2015, data['dhs']['mean'][1], label='DHS 2015 (95% CI)',
                 yerr=(data['dhs']['uq'][1] - data['dhs']['lq'][1]) / 2, fmt='o', color='mediumseagreen',
                 ecolor='green',
                 elinewidth=3, capsize=0)

    ax.plot(un_yrs, data['un']['mean'], label="UN IGCME (Uncertainty interval)", color='grey')
    ax.fill_between(un_yrs, data['un']['lq'], data['un']['uq'], color='grey', alpha=.1)
    ax.set(ylim=(0, 35))
    plt.xlabel('Year')
    plt.ylabel("Neonatal deaths per 1000 live births")
    plt.title('Yearly neonatal mortality ratio (NMR)')
    plt.legend(loc='lower left')
    plt.savefig(f'{graph_location}/all_nmr_one_graph.png')
    plt.show()

    # ------------------------------------------ CRUDE DEATHS PER YEAR ------------------------------------------------
    # Neonatal Disorders...
    scaled_neo_deaths = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=(
            lambda df: df.assign(
                year=df['date'].dt.year).groupby(['year', 'label'])['year'].count()),
        do_scaling=True)

    neo_deaths = analysis_utility_functions.return_95_CI_across_runs(
        scaled_neo_deaths.loc[(slice(None), 'Neonatal Disorders'), slice(None)].droplevel(1), gbd_years)

    # Congenital Anomalies...
    ca_deaths = analysis_utility_functions.return_95_CI_across_runs(
        scaled_neo_deaths.loc[(slice(None), 'Congenital birth defects'), slice(None)].droplevel(1), gbd_years)

    # GBD data...
    gbd_neo_deaths = extract_deaths_gbd_data('Neonatal disorders')
    gbd_cba_deaths = extract_deaths_gbd_data('Congenital birth defects')

    def gbd_bar_chart(data, gbd_data, title, save_name):
        model_ci_neo = [(x - y) / 2 for x, y in zip(data[2], data[1])]
        gbd_ci_neo = [(x - y) / 2 for x, y in zip(gbd_data[2], gbd_data[1])]

        N = len(data[0])
        ind = np.arange(N)
        width = 0.35
        plt.bar(ind, data[0], width, label='Model', yerr=model_ci_neo, color='teal')
        plt.bar(ind + width, gbd_data[0], width, label='GBD', yerr=gbd_ci_neo, color='olivedrab')
        plt.ylabel('Crude Deaths (scaled)')
        plt.title(title)
        plt.xticks(ind + width / 2, gbd_years)
        plt.legend(loc='best')
        plt.savefig(f'{graph_location}/{save_name}.png')
        plt.show()

    gbd_bar_chart(neo_deaths, gbd_neo_deaths, 'Total Deaths Attributable to "Neonatal Disorders" per Year',
                  'crude_deaths_nd')
    gbd_bar_chart(ca_deaths, gbd_cba_deaths, 'Total Deaths Attributable to "Congenital Birth Defects" per Year',
                  'crude_deaths_cba')

    # --------------------------- PROPORTION OF 'NEONATAL DISORDER' DEATHS BY CAUSE -----------------------------------
    # TODO: could output other causes also
    neo_calib_targets_fottrell = [27.3, 25.8, 8.95]  # prematuirty, birth asphyxia, early onset sepsis
    neo_calib_targets_bemonc = [26, 52, 6] # prematuirty, birth asphyxia, early onset sepsis

    direct_neonatal_causes = ['early_onset_sepsis', 'late_onset_sepsis', 'encephalopathy', 'preterm_other',
                              'respiratory_distress_syndrome', 'neonatal_respiratory_depression']

    causes_prop = dict()
    for cause in direct_neonatal_causes:
        causes_prop.update({cause: [0, 0, 0]})

    for year in sim_years:
        total_deaths_py = tnd.loc[year].mean()

        for complication in direct_neonatal_causes:
            if complication in death_results.loc[year].index:
                prop = (death_results.loc[year, complication] / total_deaths_py) * 100

                causes_prop[complication][0] += prop.mean()
                ci = st.t.interval(0.95, len(prop) - 1, loc=np.mean(prop), scale=st.sem(prop))
                causes_prop[complication][1] += ci[0]
                causes_prop[complication][2] += ci[1]

    for c in causes_prop:
        causes_prop[c] = [x / len(sim_years) for x in causes_prop[c]]

    results = [(causes_prop['respiratory_distress_syndrome'][0] + causes_prop['preterm_other'][0]),
               (causes_prop['encephalopathy'][0] + causes_prop['neonatal_respiratory_depression'][0]),
               causes_prop['early_onset_sepsis'][0] + causes_prop['late_onset_sepsis'][0]]

    results_uq = [(causes_prop['respiratory_distress_syndrome'][2] + causes_prop['preterm_other'][2]),
               (causes_prop['encephalopathy'][2] + causes_prop['neonatal_respiratory_depression'][2]),
               causes_prop['early_onset_sepsis'][2] + causes_prop['late_onset_sepsis'][2]]

    results_lq = [(causes_prop['respiratory_distress_syndrome'][1] + causes_prop['preterm_other'][1]),
               (causes_prop['encephalopathy'][1] + causes_prop['neonatal_respiratory_depression'][1]),
               causes_prop['early_onset_sepsis'][1] + causes_prop['late_onset_sepsis'][1]]

    ui = [(x - y) / 2 for x, y in zip(results_uq, results_lq)]
    # create the base axis
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # set the labels
    # and the x positions
    label = ['Prematurity', '"Birth Asphyxia"', 'Neonatal sepsis']
    x = np.arange(len(label))
    width = 0.2
    rect1 = ax.bar(x - width, results, width=width, yerr=ui, label='Model')
    rect2 = ax.bar(x, neo_calib_targets_fottrell, width=width, label='Fottrell (2015)',)
    rects2 = ax.bar(x + width, neo_calib_targets_bemonc, width=width, label='EMoNC (2015)',)
    ax.set_ylabel("% of total deaths")
    ax.set_xlabel("Cause of death")
    ax.set_title("Proportion of total neonatal deaths by leading causes (2010-2020)")
    ax.set_xticks(x)
    ax.set_xticklabels(label)
    ax.legend(loc='upper right')
    ax.tick_params(axis="x",  which="both")
    ax.tick_params(axis="y",  which="both")
    plt.savefig(f'{graph_location}/proportions_cause_of_death_neo.png')
    plt.show()

    # # ------------------------------------------- CASE FATALITY PER COMPLICATION ------------------------------------
    # nb_oc_df = extract_results(
    #         results_folder,
    #         module="tlo.methods.newborn_outcomes",
    #         key="newborn_complication",
    #         custom_generate_series=(
    #             lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'type'])['newborn'].count()),
    #         do_scaling=False
    #     )
    # nb_outcomes_df = nb_oc_df.fillna(0)
    #
    # nb_oc_pn_df = extract_results(
    #         results_folder,
    #         module="tlo.methods.postnatal_supervisor",
    #         key="newborn_complication",
    #         custom_generate_series=(
    #             lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'type'])['newborn'].count()),
    #         do_scaling=False
    #     )
    # nb_outcomes_pn_df = nb_oc_pn_df.fillna(0)
    #
    # tr = list()
    # dummy_denom = list()
    # for years in sim_years:
    #     tr.append(0)
    #     dummy_denom.append(1)
    #
    # early_ns = analysis_utility_functions.get_mean_and_quants_from_str_df(
    #     nb_outcomes_df, 'early_onset_sepsis', sim_years)[0]
    # early_ns_pn = analysis_utility_functions.get_mean_and_quants_from_str_df(
    #     nb_outcomes_pn_df, 'early_onset_sepsis', sim_years)[0]
    #
    # total_ens = [x + y for x, y in zip(early_ns, early_ns_pn)]
    #
    # late_ns_nb = analysis_utility_functions.get_mean_and_quants_from_str_df(
    #     nb_outcomes_df, 'late_onset_sepsis', sim_years)[0]
    # late_ns_pn = analysis_utility_functions.get_mean_and_quants_from_str_df(
    #     nb_outcomes_pn_df, 'late_onset_sepsis', sim_years)[0]
    #
    # late_ns = [x + y for x, y in zip(late_ns_nb, late_ns_pn)]
    #
    # mild_en = analysis_utility_functions.get_mean_and_quants_from_str_df(nb_outcomes_df, 'mild_enceph', sim_years)[0]
    # mod_en = analysis_utility_functions.get_mean_and_quants_from_str_df(nb_outcomes_df, 'moderate_enceph', sim_years)[0]
    # sev_en = analysis_utility_functions.get_mean_and_quants_from_str_df(nb_outcomes_df, 'severe_enceph', sim_years)[0]
    # total_encp = [x + y + z for x, y, z in zip(mild_en, mod_en, sev_en)]
    #
    # early_ptl_data = analysis_utility_functions.get_mean_and_quants_from_str_df(
    #     la_comps, 'early_preterm_labour', sim_years)[0]
    # late_ptl_data = analysis_utility_functions.get_mean_and_quants_from_str_df(
    #     la_comps, 'late_preterm_labour', sim_years)[0]
    # total_ptl_rates = [x + y for x, y in zip(early_ptl_data, late_ptl_data)]
    #
    # rd = analysis_utility_functions.get_mean_and_quants_from_str_df(
    #     nb_outcomes_df, 'not_breathing_at_birth', sim_years)[0]
    #
    # rds_data = analysis_utility_functions.get_mean_and_quants_from_str_df(
    #     nb_outcomes_df, 'respiratory_distress_syndrome', sim_years)[0]
    #
    # rate_of_ca = analysis_utility_functions.get_mean_and_quants_from_str_df(
    #     nb_outcomes_df, 'congenital_heart_anomaly', sim_years)[0]
    # rate_of_laa = analysis_utility_functions.get_mean_and_quants_from_str_df(
    #     nb_outcomes_df, 'limb_or_musculoskeletal_anomaly', sim_years)[0]
    # rate_of_ua = analysis_utility_functions.get_mean_and_quants_from_str_df(
    #     nb_outcomes_df, 'urogenital_anomaly', sim_years)[0]
    # rate_of_da = analysis_utility_functions.get_mean_and_quants_from_str_df(
    #     nb_outcomes_df, 'digestive_anomaly', sim_years)[0]
    # rate_of_oa = analysis_utility_functions.get_mean_and_quants_from_str_df(
    #     nb_outcomes_df, 'other_anomaly', sim_years)[0]
    #
    # for inc_list in [total_ens, late_ns, total_encp, total_ptl_rates, rds_data, rd, rate_of_ca, rate_of_laa, rate_of_ua,
    #                  rate_of_da, rate_of_oa]:
    #
    #     for index, item in enumerate(inc_list):
    #         if item == 0:
    #             inc_list[index] = 0.1
    #
    # for inc_list, complication in \
    #     zip([total_ens, late_ns, total_encp, total_ptl_rates, rds_data, rd, rate_of_ca, rate_of_laa, rate_of_ua,
    #          rate_of_da, rate_of_oa],
    #         ['early_onset_sepsis', 'late_onset_sepsis', 'encephalopathy', 'preterm_other',
    #          'respiratory_distress_syndrome', 'neonatal_respiratory_depression',
    #          'congenital_heart_anomaly', 'limb_or_musculoskeletal_anomaly', 'urogenital_anomaly',
    #          'digestive_anomaly', 'other_anomaly']):
    #
    #     cfr = analysis_utility_functions.get_comp_mean_and_rate(
    #         complication, inc_list, death_results, 100, sim_years)[0]
    #
    #     analysis_utility_functions.simple_line_chart_with_target(
    #         sim_years, cfr, tr, 'Total CFR', f'Yearly CFR for {complication}', f'{complication}_neo_cfr_per_year',
    #         graph_location)

    # PROPORTION OF NMR
    simplified_causes = ['prematurity', 'encephalopathy', 'neonatal_sepsis', 'neonatal_respiratory_depression',
                         'congenital_anomalies']

    ptb_tr = list()
    enceph_tr = list()
    sep = list()
    rd_tr = list()
    ca_tr = list()

    for year in sim_years:
        if year < 2015:
            ptb_tr.append(25*0.27)
            enceph_tr.append(25*0.25)
            sep.append(25*0.08)
            rd_tr.append(0)
            ca_tr.append(0)
        else:
            ptb_tr.append(22*0.27)
            enceph_tr.append(22*0.25)
            sep.append(22*0.08)
            rd_tr.append(0)
            ca_tr.append(0)

    ptb_tr = {'double': True, 'first': {'year': 2010, 'value': 6.75, 'label': 'UNK.', 'ci': 0},
             'second': {'year': 2015, 'value': 5.94, 'label': 'UNK.', 'ci': 0}}
    enceph_tr = {'double': True, 'first': {'year': 2010, 'value': 6.25, 'label': 'UNK.', 'ci': 0},
             'second': {'year': 2015, 'value': 5.5, 'label': 'UNK.', 'ci': 0}}
    sep = {'double': True, 'first': {'year': 2010, 'value': 2, 'label': 'UNK.', 'ci': 0},
                 'second': {'year': 2015, 'value': 1.76, 'label': 'UNK.', 'ci': 0}}
    rd_tr = {'double': True, 'first': {'year': 2010, 'value': 0, 'label': 'UNK.', 'ci': 0},
              'second': {'year': 2015, 'value': 0, 'label': 'UNK.', 'ci': 0}}
    ca_tr = {'double': True, 'first': {'year': 2010, 'value': 0, 'label': 'UNK.', 'ci': 0},
             'second': {'year': 2015, 'value': 0, 'label': 'UNK.', 'ci': 0}}

    trs = [ptb_tr, enceph_tr, sep, rd_tr, ca_tr]

    for cause, tr in zip(simplified_causes, trs):
        if (cause == 'encephalopathy') or (cause == 'neonatal_respiratory_depression'):
            deaths = death_results.loc[(slice(None), cause), slice(None)].droplevel(1)

        elif cause == 'neonatal_sepsis':
            early = death_results.loc[(slice(None), 'early_onset_sepsis'), slice(None)].droplevel(1)
            late = death_results.loc[(slice(None), 'late_onset_sepsis'), slice(None)].droplevel(1)
            deaths = early + late

        elif cause == 'prematurity':
            rds_deaths = death_results.loc[(slice(None), 'respiratory_distress_syndrome'), slice(None)].droplevel(1)
            other_deaths = death_results.loc[(slice(None), 'preterm_other'), slice(None)].droplevel(1)
            deaths = rds_deaths + other_deaths

        elif cause == 'congenital_anomalies':
            ca_deaths = death_results.loc[(slice(None), 'congenital_heart_anomaly'), slice(None)].droplevel(1)
            la_deaths = death_results.loc[(slice(None), 'limb_or_musculoskeletal_anomaly'), slice(None)].droplevel(1)
            ua_deaths = death_results.loc[(slice(None), 'urogenital_anomaly'), slice(None)].droplevel(1)
            da_deaths = death_results.loc[(slice(None), 'digestive_anomaly'), slice(None)].droplevel(1)
            oa_deaths = death_results.loc[(slice(None), 'other_anomaly'), slice(None)].droplevel(1)

            deaths = ca_deaths + la_deaths + ua_deaths + da_deaths + oa_deaths
        nmr = return_rate(deaths, births_results_exc_2010, 1000)
        analysis_utility_functions.line_graph_with_ci_and_target_rate(
            sim_years, nmr, tr, 'Rate per 1000 live births',
            f'Neonatal Mortality Rate per Year for {cause}', graph_location, f'nmr_{cause}')

    # proportion causes for preterm birth
    # -------------------------------------------------------- DALYS -------------------------------------------------

    def extract_dalys_gbd_data(group):
        dalys_df = dalys_data.loc[(dalys_data['measure_name'] == 'DALYs (Disability-Adjusted Life Years)') &
                                  (dalys_data['cause_name'] == f'{group} disorders') & (dalys_data['Year'] > 2009)]

        gbd_dalys = list()
        gbd_dalys_lq = list()
        gbd_dalys_uq = list()

        for year in gbd_years:
            gbd_dalys.append(dalys_df.loc[(dalys_df['Year'] == year)].sum()['GBD_Est'])
            gbd_dalys_lq.append(dalys_df.loc[(dalys_df['Year'] == year)].sum()['GBD_Lower'])
            gbd_dalys_uq.append(dalys_df.loc[(dalys_df['Year'] == year)].sum()['GBD_Upper'])

        if group == 'Maternal':
            # Rates/total DALYs are adjusted to remove 'indirect maternal' and HIV aggravated deaths
            mat_dalys_not_modelled = [[9735, 9795, 9673, 9484, 9305, 9153, 9220, 9177, 9115, 9128],
                                      [13667, 13705, 13588, 13441, 13359, 13230, 13555, 13558, 13506, 13553],
                                      [6308, 6304, 6225, 6148, 5878, 5764, 5763, 5664, 5611, 5548]]

            maternal_gbd_dalys_adj = [[x - y for x, y in zip(gbd_dalys, mat_dalys_not_modelled[0])],
                                      [x - y for x, y in zip(gbd_dalys_uq, mat_dalys_not_modelled[1])],
                                      [x - y for x, y in zip(gbd_dalys_lq, mat_dalys_not_modelled[2])]]

            gbd_dalys_rate = [[591, 561, 529, 501, 475, 452, 439, 423, 411, 404],
                              [763, 731, 695, 660, 631, 611, 599, 577, 560, 555],
                              [422, 403, 382, 358, 337, 320, 306, 288, 278, 275]]

            gbd_dalys_rate_adj = [[523, 494, 465, 440, 417, 396, 385, 370, 360, 354],
                                  [668, 638, 606, 574, 547, 531, 520, 499, 484, 482],
                                  [378, 360, 341, 319, 300, 285, 272, 256, 247, 245]]

            return {'total': [gbd_dalys, gbd_dalys_lq, gbd_dalys_uq],
                    'total_adj': maternal_gbd_dalys_adj,
                    'rate': gbd_dalys_rate,
                    'rate_adj': gbd_dalys_rate_adj}
        else:

            rate = [[8059, 7754, 7388, 7039, 6667, 6323, 6051, 5637, 5402, 5199],
                    [9576, 9191, 8784, 8402, 8010, 7798, 7458, 7091, 6901, 6756],
                    [6785, 6460, 6115, 5773, 5449, 5167, 4847, 4486, 4246, 4044]]

            return {'total': [gbd_dalys, gbd_dalys_lq, gbd_dalys_uq],
                    'rate': rate}

    maternal_gbd_dalys = extract_dalys_gbd_data('Maternal')
    neonatal_gbd_dalys = extract_dalys_gbd_data('Neonatal')

    dalys_stacked = extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked",
        custom_generate_series=(
            lambda df: df.drop(
                columns='date').groupby(['year']).sum().stack()),
        do_scaling=True)

    person_years_total = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="person_years",
        custom_generate_series=(
            lambda df: df.assign(total=(df['M'].apply(lambda x: sum(x.values()))) + df['F'].apply(
                lambda x: sum(x.values()))).assign(
                year=df['date'].dt.year).groupby(['year'])['total'].sum()),
        do_scaling=True)

    mat_model_dalys_data = dict()
    neo_model_dalys_data = dict()

    mat_model_dalys_data.update({'rate': return_rate(
        dalys_stacked.loc[(slice(None), 'Maternal Disorders'), slice(None)].droplevel(1), person_years_total, 100_000)})
    neo_model_dalys_data.update(
        {'rate': return_rate(
        dalys_stacked.loc[(slice(None), 'Neonatal Disorders'), slice(None)].droplevel(1), person_years_total, 100_000)})
    mat_model_dalys_data.update({'total': analysis_utility_functions.return_95_CI_across_runs(
        dalys_stacked.loc[(slice(None), 'Maternal Disorders'), slice(None)].droplevel(1), sim_years)})
    neo_model_dalys_data.update({'total': analysis_utility_functions.return_95_CI_across_runs(
        dalys_stacked.loc[(slice(None), 'Neonatal Disorders'), slice(None)].droplevel(1), sim_years)})

    def get_daly_graphs(group, model_data, gbd_data):

        # Total
        fig, ax = plt.subplots()
        ax.plot(sim_years, model_data['total'][0], label=f"Model (95% CI)", color='deepskyblue')
        ax.fill_between(sim_years, model_data['total'][1], model_data['total'][2], color='b', alpha=.1)

        ax.plot(gbd_years, gbd_data['total'][0], label="GBD (Lower & Upper bounds)", color='olivedrab')
        ax.fill_between(gbd_years, gbd_data['total'][1], gbd_data['total'][2], color='g', alpha=.1)

        if group == 'Maternal':
            # ax.plot(gbd_years, gbd_data['total_adj'][0], label="GBD DALY Adj.", color='darkslateblue')
            # ax.fill_between(gbd_years, gbd_data['total_adj'][1], gbd_data['total_adj'][2], color='slateblue'
            #                 , alpha=.1)
            ax.set(ylim=(0, 160_000))
        else:
            ax.set(ylim=0)

        plt.xlabel('Year')
        plt.ylabel("DALYs (stacked)")
        plt.title(f'Total DALYs per year attributable to {group} disorders')
        plt.legend()
        plt.savefig(f'{graph_location}/{group}_dalys_stacked.png')
        plt.show()

        # Rate
        fig, ax = plt.subplots()
        ax.plot(sim_years, model_data['rate'][0], label=f"Model (95% CI)", color='deepskyblue')
        ax.fill_between(sim_years, model_data['rate'][1], model_data['rate'][2], color='b', alpha=.1)

        ax.plot(gbd_years, gbd_data['rate'][0], label="GBD (Lower & Upper bounds)", color='olivedrab')
        ax.fill_between(gbd_years, gbd_data['rate'][1], gbd_data['rate'][2], color='g', alpha=.1)

        if group == 'Maternal':
            # ax.plot(gbd_years, gbd_data['rate_adj'][0], label="GBD DALY Adj. rate", color='darkslateblue')
            # ax.fill_between(gbd_years, gbd_data['rate_adj'][1], gbd_data['rate_adj'][2], color='slateblue'
            #                 , alpha=.1)
            ax.set(ylim=(0, 850))
        else:
            ax.set(ylim=0)

        plt.xlabel('Year')
        plt.ylabel("DALYs per 100k person years")
        plt.title(f'Total DALYs per 100,000 person years per year attributable to {group} Disorders')
        plt.legend()
        plt.savefig(f'{graph_location}/{group}_dalys_stacked_rate.png')
        plt.show()

    get_daly_graphs('Maternal', mat_model_dalys_data, maternal_gbd_dalys)
    get_daly_graphs('Neonatal', neo_model_dalys_data, neonatal_gbd_dalys)

