import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
from matplotlib import pyplot as plt

from src.scripts.maternal_perinatal_analyses.analysis_scripts import analysis_utility_functions
from tlo.analysis.utils import extract_results, get_scenario_outputs

plt.style.use('seaborn')


def output_incidence_for_calibration(scenario_filename, pop_size, outputspath, sim_years):
    """
    This function extracts and plots all calibration outputs for the Maternal and Perinatal Health Model.
    :param scenario_filename: file name of the scenario
    :param pop_size: population size this scenario was run on (used to name file)
    :param outputspath:directory for graphs to be saved
    :param sim_years: years the scenario was ran for
    """

    # Store location of scenario file for which calibration is being assessed
    results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]

    # Create folder to store graphs (if it hasn't already been created when ran previously)
    path = f'{outputspath}/calibration_output_graphs_{pop_size}_{results_folder.name}'
    if not os.path.isdir(path):
        os.makedirs(f'{outputspath}/calibration_output_graphs_{pop_size}_{results_folder.name}')

    graph_location = path

    # Data is presented between 2011-2022 to allow for stabilisation of outputs in 2010 when the simulation starts
    alt_years = sim_years[1:13]

    def get_modules_maternal_complication_dataframes(module):
        """Returns a dataframe which has the total number of maternal complications per year across scenario runs
        for a specific module"""
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

    #  Store the complication dataframes for each of the modules
    an_comps = get_modules_maternal_complication_dataframes('pregnancy_supervisor')
    la_comps = get_modules_maternal_complication_dataframes('labour')
    pn_comps = get_modules_maternal_complication_dataframes('postnatal_supervisor')

    # ============================================  DENOMINATORS... ===================================================
    # Here the relevant denominators are extracted from the scenario file which are used throughout to calculate
    # the rate of key outcomes
    # ---------------------------------------------Total_pregnancies...------------------------------------------------
    pregnancy_poll_results = extract_results(
        results_folder,
        module="tlo.methods.contraception",
        key="pregnancy",
        custom_generate_series=(
            lambda df: df.assign(year=pd.to_datetime(df['date']).dt.year).groupby(['year'])['year'].count()
        ))

    # As well as extracting the data for future calculations the total number of, in this case, pregnancies per year
    # of the simulation is plotted along with the 95% CI - whilst this is not a calibration outcome it is used as a
    # sense check
    preg_data = analysis_utility_functions.return_95_CI_across_runs(pregnancy_poll_results, sim_years)
    analysis_utility_functions.simple_line_chart_with_ci(
        sim_years, preg_data, 'Pregnancies (mean)', 'Mean number of pregnancies', 'pregnancies', graph_location)

    # ----------------------------------------------- Total births --------------------------------------------------
    # This is process is repeated for births (artificial births added in by the demography module to aid calibration
    # are excluded)
    births_results_exc_2010 = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="on_birth",
        custom_generate_series=(
            lambda df:
            df.loc[(df['mother'] >= 0)].assign(year=df['date'].dt.year).groupby(['year'])['year'].count()))

    lb_data = analysis_utility_functions.return_95_CI_across_runs(births_results_exc_2010, sim_years)
    analysis_utility_functions.simple_line_chart_with_ci(
        sim_years, lb_data, 'Births (mean)', 'Mean number of Live Births per Year', 'live_births',
        graph_location)

    # To get the total number of births (live + stillbirth) stillbirths are extracted here and added to data extracted
    # above
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

    #  Total births are then also plotted
    all_births_df = an_stillbirth_results + ip_stillbirth_results + births_results_exc_2010
    all_births_data = analysis_utility_functions.return_95_CI_across_runs(all_births_df, sim_years)
    analysis_utility_functions.simple_line_chart_with_ci(
        sim_years, all_births_data, 'Births (mean)', 'Mean number of Total Births per Year', 'births', graph_location)

    # -------------------------------------------Completed pregnancies...-------------------------------------------
    # The number of pregnancies which are 'completed' each year is then extracted as the total cases of pregnancy loss
    # (by all causes) and live births
    total_completed_pregnancies_df = \
        an_comps.loc[(slice(None), 'ectopic_unruptured'), slice(None)].droplevel(1) + \
        an_comps.loc[(slice(None), 'induced_abortion'), slice(None)].droplevel(1) + \
        an_comps.loc[(slice(None), 'spontaneous_abortion'), slice(None)].droplevel(1) + \
        an_stillbirth_results + ip_stillbirth_results + births_results_exc_2010

    # -----------------------------------Percentage of pregnancies ending in birth ------------------------------------
    # Finally the total number of pregnancies which end in live birth is extracted, again as a sense check not as a
    # calibration target
    total_pregnancy_losses_df = \
        an_comps.loc[(slice(None), 'ectopic_unruptured'), slice(None)].droplevel(1) + \
        an_comps.loc[(slice(None), 'induced_abortion'), slice(None)].droplevel(1) + \
        an_comps.loc[(slice(None), 'spontaneous_abortion'), slice(None)].droplevel(1) + \
        an_stillbirth_results + ip_stillbirth_results

    percent_pregs_end_live_birth_df = 100 - ((total_pregnancy_losses_df / pregnancy_poll_results) * 100)
    percent_live_birth = analysis_utility_functions.return_95_CI_across_runs(percent_pregs_end_live_birth_df, alt_years)

    analysis_utility_functions.simple_bar_chart(
        percent_live_birth[0], 'Year', '% of Total Pregnancies',
        'Percentage of Total Pregnancies Ending in a Live Birth per Year', 'preg_loss_proportion', alt_years,
        graph_location)

    # ======================================== HEALTH SERVICE CALIBRATION =============================================
    # Here, calibration plots relating to maternity service coverage are generated

    def cov_bar_chart(mdata, cdata, title, save_name):
        """Generates a plot comparing coverage of a health service with a calibration data point from the Malawi
         DHS as a barchart"""

        model_ci = [(x - y) / 2 for x, y in zip(mdata[2], mdata[1])]
        N = len(mdata[0])
        ind = np.arange(N)
        width = 0.28
        plt.bar(ind, mdata[0], width, label='Model (95% CI)', yerr=model_ci, color='cornflowerblue')
        plt.bar(ind + width, cdata, width, label='DHS', color='forestgreen')
        plt.ylabel('Percentage of Total Births')
        plt.xlabel('Year')
        plt.ylim(0, 100)
        plt.title(title)
        plt.xticks(ind + width / 2, ['2010', '2015'])
        plt.legend(bbox_to_anchor=(1.4, 1.1))
        plt.savefig(f'{graph_location}/{save_name}.png', bbox_inches="tight")
        plt.show()

    # --------------------------------------------- Antenatal Care (ANC) ---------------------------------------------
    # Mean proportion of women (across draws) who have given birth that have attended ANC1, ANC4+ and ANC8+ per year...

    # First the total number of women who have given birth, and therefore had their ANC data recorded, are counted and
    # that data is extracted
    anc_coverage = extract_results(
        results_folder,
        module="tlo.methods.care_of_women_during_pregnancy",
        key="anc_count_on_birth",
        custom_generate_series=(
            lambda df: df.assign(
                year=df['date'].dt.year).groupby(['year'])['person_id'].count()),
        do_scaling=True
    )

    # This function is then used to extract a version of the data counting women who have attended >1 or >4 visits
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

    # A dataframe containing the yearly coverage of ANC1 and ANC4 is generated
    cov_4 = (get_anc_df(4) / anc_coverage) * 100
    cov_4_df = cov_4.fillna(0)

    cov_1 = (get_anc_df(1) / anc_coverage) * 100
    cov_1_df = cov_1.fillna(0)

    # And the average coverage across the runs for each year is calculated
    anc1_coverage = analysis_utility_functions.return_95_CI_across_runs(cov_1_df, sim_years)
    anc4_coverage = analysis_utility_functions.return_95_CI_across_runs(cov_4_df, sim_years)

    # This is then plotted as a line graph allowing for comparison to calibration data points
    target_anc1_dict = {'double': True,
                        'first': {'year': 2010, 'value': 94, 'label': 'DHS (2010)', 'ci': 0},
                        'second': {'year': 2015, 'value': 95, 'label': 'DHS (2015)', 'ci': 0}}
    target_anc4_dict = {'double': True,
                        'first': {'year': 2010, 'value': 45.5, 'label': 'DHS (2010)', 'ci': 0},
                        'second': {'year': 2015, 'value': 51, 'label': 'DHS (2015)', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, anc1_coverage, target_anc1_dict, 100, '% of women who gave birth',
        'Percentage of women who gave birth that attended one or more ANC contacts per year', graph_location,
        'anc_prop_anc1')

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, anc4_coverage, target_anc4_dict, 100, '% of women who gave birth',
        'Percentage of women who gave birth that attended four or more ANC contacts per year', graph_location,
        'anc_prop_anc4')

    # Next, the mean (95% CI) coverage for 2010 and 2015 is extracted from the above data
    anc1_for_bc = [[anc1_coverage[0][0], anc1_coverage[0][5]],
                   [anc1_coverage[1][0], anc1_coverage[1][5]],
                   [anc1_coverage[2][0], anc1_coverage[2][5]]]
    anc4_for_bc = [[anc4_coverage[0][0], anc4_coverage[0][5]],
                   [anc4_coverage[1][0], anc4_coverage[1][5]],
                   [anc4_coverage[2][0], anc4_coverage[2][5]]]

    # And is displayed next to the DHS calibration points as a bar graph
    for model_data, calib_data, title, save_name in zip([anc1_for_bc, anc4_for_bc],
                                                        [[94, 95], [45.5, 51]],
                                                        ['Percentage of women who gave birth and received ANC1+',
                                                         'Percentage of women who gave birth and received ANC4+'],
                                                        ['anc1+_cov_bar', 'anc4+_cov_bar'],
                                                        ):
        cov_bar_chart(model_data, calib_data, title, save_name)

    # Then the total ANC visits per woman is calculated...
    # A dataframe totalling the number of women by the total number of ANC contacts is generated
    r = extract_results(
        results_folder,
        module="tlo.methods.care_of_women_during_pregnancy",
        key="anc_count_on_birth",
        custom_generate_series=(
            lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'total_anc'])['person_id'].count()),
        do_scaling=False
    )
    results = r.fillna(0)

    # A new dataframe is defined to store data after reformatting below
    columns = ['DHS 2010', 2010, 2011, 2012, 2013, 2014, 'DHS 2015', 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
    anc_count_df = pd.DataFrame(columns=columns, index=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    anc_count_df['DHS 2010'] = pd.Series([0, 2.8, 13.5, 38, 25.9, 12, 4.9, 1.3, 1.6], index=anc_count_df.index)
    anc_count_df['DHS 2015'] = pd.Series([0, 2, 10, 36, 30, 14, 5, 1.2, 1.8], index=anc_count_df.index)

    # The mean number of women who have attended each ANC contact number is calculated for each year of the simulation.
    # From this the percentage of women who have attended 0, 1, 2, 3 etc. ANC visits on birth is calculated
    # (this is extracted for every year as opposed to just the calibration years as a sense check that the calibration
    # stays stable over time - I currently only output calibration years)
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
                anc_count_df.at[index, year] = (anc_count_df.at[index, year] / total_per_year) * 100

    # These results are then outputted as a stacked barchart for the years 2010 and 2015
    new_anc_count_df = anc_count_df.drop([2011, 2012, 2013, 2014, 2016, 2017, 2018, 2019, 2020, 2021, 2022], axis=1)
    labels = ['DHS 2010', '2010 (Model)', 'DHS 2015', '2015 (Model)']
    width = 0.35  # the width of the bars: can also be len(x) sequence
    fig, ax = plt.subplots()
    ax.bar(labels, new_anc_count_df.loc[4] + new_anc_count_df.loc[5] + new_anc_count_df.loc[6] +
           new_anc_count_df.loc[7] + new_anc_count_df.loc[8], width, label='4+',
           bottom=new_anc_count_df.loc[1] + new_anc_count_df.loc[2] + new_anc_count_df.loc[3])
    ax.bar(labels, new_anc_count_df.loc[3], width, label=3, bottom=new_anc_count_df.loc[1] + new_anc_count_df.loc[2])
    ax.bar(labels, new_anc_count_df.loc[2], width, label=2, bottom=new_anc_count_df.loc[1])
    ax.bar(labels, new_anc_count_df.loc[1], width, label=1, )
    ax.set_ylabel('Percentage of women attending one or more ANC contact')
    ax.set_xlabel('Year')
    ax.set_title('Number of ANC Contacts Attended by Women Attending at Least One Contact (2010 & 2015)')
    ax.legend(bbox_to_anchor=(1.2, 1.1))
    plt.savefig(f'{graph_location}/anc_total_visits_2010_2015.png', bbox_inches="tight")
    plt.show()

    # Next, the mean proportion of women who attended at least one ANC visit that attended at < 4, 4-5, 6-7 and
    # > 8 months gestational age is outputted and compared to the DHS data set
    # ( as with above, this is extracted for every year as opposed to just the calibration years as a sense check that
    # the calibration stays stable over time - I currently only output calibration years)

    anc_ga = extract_results(
        results_folder,
        module="tlo.methods.care_of_women_during_pregnancy",
        key="anc_count_on_birth",
        custom_generate_series=(
            lambda df_: df_.assign(year=df_['date'].dt.year).groupby(
                ['year', 'total_anc', 'ga_anc_one'])['person_id'].count()),
        do_scaling=False
    )
    anc_ga_first_visit = anc_ga.fillna(0)

    anc_before_four_months = list()
    early_anc_4_list = list()
    anc_before_four_five_months = list()
    anc_before_six_seven_months = list()
    anc_before_eight_plus_months = list()

    def get_means(df):
        mean_list = 0
        for index in df.index:
            mean_list += df.loc[index].mean()
        return mean_list

    # For each year the total women attending ANC1 by each of the gestational age groupings is calculated, first as a
    # crude number then as a percentage of the total women who attended ANC1 that year
    for year in sim_years:
        # Add calibration data
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
        year_df = year_df.sort_index()
        total_women_that_year = 0
        for index in year_df.index:
            total_women_that_year += year_df.loc[index].mean()

        anc1 = get_means(year_df.loc[0, 0:len(year_df.columns)])

        total_women_anc = total_women_that_year - anc1

        early_anc = year_df.loc[(slice(1, 8), slice(0, 13)), 0:len(year_df.columns)]
        early_anc4 = year_df.loc[(slice(4, 8), slice(0, 17)), 0:len(year_df.columns)]
        four_to_five = year_df.loc[(slice(1, 8), slice(14, 22)), 0:len(year_df.columns)]
        six_to_seven = year_df.loc[(slice(1, 8), slice(23, 31)), 0:len(year_df.columns)]
        eight_plus = year_df.loc[(slice(1, 8), slice(32, 50)), 0:len(year_df.columns)]

        sum_means_early = get_means(early_anc)
        sum_means_early_anc4 = get_means(early_anc4)
        sum_four = get_means(four_to_five)
        sum_six = get_means(six_to_seven)
        sum_eight = get_means(eight_plus)

        # Also output the proportion of women per year who attend 4 visits with there first visit before month 5
        early_anc_4_list.append((sum_means_early_anc4 / total_women_anc) * 100)

        anc_before_four_months.append((sum_means_early / total_women_anc) * 100)
        anc_before_four_five_months.append((sum_four / total_women_anc) * 100)
        anc_before_six_seven_months.append((sum_six / total_women_anc) * 100)
        anc_before_eight_plus_months.append((sum_eight / total_women_anc) * 100)

    # Finally plot these results as a stacked bar graph using model data from 2010/2015 compared to the
    # calibration data
    a_4 = [anc_before_four_months[0], anc_before_four_months[1], anc_before_four_months[6], anc_before_four_months[8]]
    a_5 = [anc_before_four_five_months[0], anc_before_four_five_months[1], anc_before_four_five_months[6],
           anc_before_four_five_months[8]]
    a_6 = [anc_before_six_seven_months[0], anc_before_six_seven_months[1], anc_before_six_seven_months[6],
           anc_before_six_seven_months[8]]
    a_8 = [anc_before_eight_plus_months[0], anc_before_eight_plus_months[1], anc_before_eight_plus_months[6],
           anc_before_eight_plus_months[8]]

    labels = ['2010 (DHS)', '2010 (Model)', '2015 (DHS)', '2016 (Model)']
    fig, ax = plt.subplots()
    ax.bar(labels, a_8, width, label='>8m',
           bottom=[x + y + z for x, y, z in zip(a_4, a_5, a_6)])
    ax.bar(labels, a_6, width, label='6-7m',
           bottom=[x + y for x, y in zip(a_4, a_5)])
    ax.bar(labels, a_5, width, label='4-5m',
           bottom=a_4)
    ax.bar(labels, a_4, width, label='<4m')
    ax.legend(bbox_to_anchor=(1.2, 1.1))
    ax.set_ylabel('% of first ANC contacts')
    ax.set_title('Maternal gestational age at first ANC contact by year')
    plt.savefig(f'{graph_location}/anc_ga_first_visit_update_2010_2015.png', bbox_inches="tight")
    plt.show()

    # --------------------------------------------- Facility delivery -------------------------------------------------
    # Here the yearly health centre, hospital, home and facility delivery rates are outputted against calibration
    # targets

    def get_delivery_rate(facility_type):
        """Returns the years average coverage of rate of delivery for a given setting per year of the simulation"""
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

        rate = (deliver_setting_results / births_results_exc_2010) * 100
        data = analysis_utility_functions.return_95_CI_across_runs(rate, sim_years)
        return data

    # Calculate delivery rates
    fd_data = get_delivery_rate('facility')
    hb_data = get_delivery_rate('home_birth')
    hp_data = get_delivery_rate('hospital')
    hc_data = get_delivery_rate('health_centre')

    # Plot the facility delivery rate over time compared to the DHS data. This dictionary (and those like it throughout)
    # contain calibration target data and the relevant source which is plotted alongside model data)
    target_fd_dict = {'double': True,
                      'first': {'year': 2010, 'value': 73, 'label': 'DHS (2010)', 'ci': 0},
                      'second': {'year': 2015, 'value': 91, 'label': 'DHS (2015)', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, fd_data, target_fd_dict, 100, '% of total births',
        'Percentage of births occurring in a health facility per year ', graph_location, 'sba_prop_facility_deliv')

    # Next for the years 2010 and 2015 the percentage of total births occurring at home, hospital or health centre are
    # outputted as a stacked bar chart along with estimates from the DHS
    hb_data = [27, hb_data[0][0], 8, hb_data[0][5]]
    hc_data = [41, hc_data[0][0], 52, hc_data[0][5]]
    hp_data = [32, hp_data[0][0], 40, hp_data[0][5]]
    labels = ['2010 (DHS)', '2010 (Model)', '2015 (DHS)', '2015 (model)']
    width = 0.35  # the width of the bars: can also be len(x) sequence
    fig, ax = plt.subplots()

    ax.bar(labels, hp_data, width, label='Hospital',
           bottom=[x + y for x, y in zip(hb_data, hc_data)])
    ax.bar(labels, hc_data, width, label='Health Centre',
           bottom=hb_data)
    ax.bar(labels, hb_data, width, label='Home')
    ax.set_ylabel('% of total births')
    ax.set_title('Percentage of total births (2010 & 2015) by location of delivery')
    ax.legend(bbox_to_anchor=(1.3, 1.))
    plt.savefig(f'{graph_location}/sba_delivery_location.png', bbox_inches="tight")
    plt.show()

    # Finally, a bar chart is plotted comparing the total coverage of facility delivery in 2010 and 2015 with the DHS
    fd_for_bc = [[fd_data[0][0], fd_data[0][5]],
                 [fd_data[1][0], fd_data[1][5]],
                 [fd_data[2][0], fd_data[2][5]]]

    cov_bar_chart(fd_for_bc, [73, 91], 'Percentage of Total Births Occurring in a Health Facility', 'fd_cov_bar')

    # ----------------------------------------- Postnatal care (PNC) -------------------------------------------------
    # Data on postnatal care visits for mothers and newborns is extracted from the scenario
    pnc_results_maternal = extract_results(
        results_folder,
        module="tlo.methods.postnatal_supervisor",
        key="total_mat_pnc_visits",
        custom_generate_series=(
            lambda df: df.loc[df['visits'] > 0].assign(year=df['date'].dt.year).groupby(['year'])[
                'mother'].count()))

    pnc_results_newborn = extract_results(
        results_folder,
        module="tlo.methods.postnatal_supervisor",
        key="total_neo_pnc_visits",
        custom_generate_series=(
            lambda df: df.loc[df['visits'] > 0].assign(year=df['date'].dt.year).groupby(['year'])[
                'child'].count()))

    # Using the data above, the average rate is calculated for each year of the simulation
    cov_mat_birth_df = (pnc_results_maternal / births_results_exc_2010) * 100
    mat_birth_pnc_data = analysis_utility_functions.return_95_CI_across_runs(cov_mat_birth_df, sim_years)

    cov_neo_birth_df = (pnc_results_newborn / births_results_exc_2010) * 100
    neo_birth_pnc_data = analysis_utility_functions.return_95_CI_across_runs(cov_neo_birth_df, sim_years)

    # And outputted as a line graph, compared to estimates from the DHS
    target_mpnc_dict = {'double': True,
                        'first': {'year': 2010, 'value': 50, 'label': 'DHS (2010)', 'ci': 0},
                        'second': {'year': 2015, 'value': 48, 'label': 'DHS (2015)', 'ci': 0}}

    target_npnc_dict = {'double': False,
                        'first': {'year': 2015, 'value': 60, 'label': 'DHS (2015)', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, mat_birth_pnc_data, target_mpnc_dict, 100, '% of total births',
        'Percentage of total births after which the mother received any PNC per year', graph_location, 'pnc_mat')

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, neo_birth_pnc_data, target_npnc_dict, 100, '% of total births',
        'Percentage of total births after which the neonate received any PNC per year', graph_location, 'pnc_neo')

    # Then a bar chart is outputted comparing coverage of PNC in the model for 2010 and 2015 with the DHS for both
    # mothers and newborns
    m_pnc_for_bc = [[mat_birth_pnc_data[0][1], mat_birth_pnc_data[0][5]],
                    [mat_birth_pnc_data[1][1], mat_birth_pnc_data[1][5]],
                    [mat_birth_pnc_data[2][1], mat_birth_pnc_data[2][5]]]

    cov_bar_chart(m_pnc_for_bc, [50, 48], 'Percentage of Women Receiving Any Postnatal Care Following Birth',
                  'mat_pnc_cov_bar')

    n_pnc_for_bc = [neo_birth_pnc_data[0][5], neo_birth_pnc_data[1][5], neo_birth_pnc_data[2][5]]
    model_ci = (n_pnc_for_bc[2] - n_pnc_for_bc[1]) / 2
    N = 1
    ind = np.arange(N)
    width = 0.3
    plt.bar(ind, n_pnc_for_bc[0], width, label='Model (95% CI)', yerr=model_ci, color='cornflowerblue')
    plt.bar(ind + 0.5, 60, width, label='DHS', color='forestgreen')
    plt.ylabel('Percentage of Total Births')
    plt.xlabel('Year')
    plt.ylim(0, 100)
    plt.title('Percentage of Neonates Receiving Any Postnatal Care Following Birth')
    plt.xticks(ind + 0.5 / 2, ['2015'])
    plt.legend()
    plt.savefig(f'{graph_location}/pnc_neo_2015'
                f'.png', bbox_inches="tight")
    plt.show()

    # Finally, the percentage of PNC visits by mothers and newborns that occur within 48hrs and after 48hrs of
    # delivery is outputted for each year of the simulation
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
        width = 0.35
        fig, ax = plt.subplots()
        ax.bar(labels, early_rate, width, label='< 48hrs')
        ax.bar(labels, late_rate, width, bottom=early_rate, label='>48hrs')
        ax.set_ylabel('% of PNC 1 visits')
        ax.set_title(f'Percentage of {target} PNC1 Visits occurring pre/post 48hrs Postnatal')
        ax.legend()
        plt.savefig(f'{graph_location}/{file_name}.png')
        plt.show()

    get_early_late_pnc_split('labour', 'Maternal', 'pnc_maternal_early')
    get_early_late_pnc_split('newborn_outcomes', 'Neonatal', 'pnc_neonatal_early')

    # ====================================== RATE OF COMPLICATIONS CALIBRATION =======================================
    # ========================================== MATERNAL COMPLICATIONS ==============================================
    # Here the yearly rate/prevalence of each of the modelled complications is outputted and compared to some
    # predetermined calibration target
    def return_rate(num_df, denom_df, denom_val, years):
        """Returns the average rate of a complication per year across runs of the simulation"""
        rate_df = (num_df / denom_df) * denom_val
        data = analysis_utility_functions.return_95_CI_across_runs(rate_df, years)
        return data

    # ------------------------------------------ Twinning Rate -------------------------------------------------------
    # Twin births are captured in a separate log, which is extracted here
    tr = extract_results(
        results_folder,
        module="tlo.methods.newborn_outcomes",
        key="twin_birth",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()
        ),
    )
    twins_results = tr.fillna(0)

    # The denominator here is equal to the number of mothers who have given birth (equal to the total number of
    # children born minus total recorded twin births to avoid double counting)
    total_deliv = births_results_exc_2010 - twins_results
    final_twining_rate = return_rate(twins_results, total_deliv, 100, alt_years)

    target_twin_dict = {'double': False,
                        'first': {'year': 2011, 'value': 3.9, 'label': 'Monden & Smits', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, final_twining_rate, target_twin_dict, 5, 'Rate per 100 births',
        'Twin Births per 100 Births per Year', graph_location, 'twin_rate')

    # For most of the remaining complications, the number of a times a given complication has occurred is extracted from
    # the complication dataframes defined earlier, the rate is calculated using a predetermined denominator and is
    # outputted in addition to a calibration target (these data are stored in dictionaries shown before the function
    # which outputs plots)

    # ---------------------------------------- Early Pregnancy Loss... ------------------------------------------------
    # Ectopic pregnancies
    ep = an_comps.loc[(slice(None), 'ectopic_unruptured'), slice(None)].droplevel(1)
    ectopic_data = return_rate(ep, pregnancy_poll_results, 1000, alt_years)
    target_ect_dict = {'double': False,
                       'first': {'year': 2015, 'value': 10, 'label': 'Panelli et al.,', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, ectopic_data, target_ect_dict, 12, 'Rate per 1000 pregnancies',
        'Ectopic Pregnancies per 1000 Pregnancies per Year', graph_location, 'ectopic_rate')

    # Ruptured ectopic pregnancies
    ep_r = an_comps.loc[(slice(None), 'ectopic_ruptured'), slice(None)].droplevel(1)
    proportion_of_ectopics_that_rupture_per_year = return_rate(ep_r, ep, 100, alt_years)

    target_rate_rup = {'double': False,
                       'first': {'year': 2015, 'value': 92, 'label': 'Est.', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, proportion_of_ectopics_that_rupture_per_year, target_rate_rup, 100,
        'Proportion of ectopic pregnancies',
        'Proportion of Ectopic Pregnancies ending in rupture', graph_location, 'prop_rupture')

    # Spontaneous Abortions
    sa = an_comps.loc[(slice(None), 'spontaneous_abortion'), slice(None)].droplevel(1)
    spotaneous_abortion_data = return_rate(sa, total_completed_pregnancies_df, 1000, alt_years)

    target_sa_dict = {'double': False,
                      'first': {'year': 2015, 'value': 153, 'label': 'Polis et al.', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, spotaneous_abortion_data,
        target_sa_dict, 165, 'Rate per 1000 completed pregnancies',
        'Spontaneous Abortions per 1000 Completed Pregnancies per Year', graph_location,
        'miscarriage_rate')

    # Induced Abortions
    ia = an_comps.loc[(slice(None), 'induced_abortion'), slice(None)].droplevel(1)
    ia_data = return_rate(ia, total_completed_pregnancies_df, 1000, alt_years)

    target_ia_dict = {'double': True,
                      'first': {'year': 2011, 'value': 86, 'label': 'Levandowski et al.', 'ci': 0},
                      'second': {'year': 2015, 'value': 159, 'label': 'Polis et al.', 'ci': 0},
                      }

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, ia_data, target_ia_dict, 170, 'Rate per 1000 completed pregnancies',
        'Induced Abortions per 1000 Completed Pregnancies per Year', graph_location, 'abortion_rate')

    # (In addition to the rate of induced/spontaneous abortion, the proportion of I/SA which lead to complications is
    # also outputted for each year as a sense check)

    # Complicated spontaneous abortion
    c_sa = an_comps.loc[(slice(None), 'complicated_spontaneous_abortion'), slice(None)].droplevel(1)
    proportion_of_complicated_sa_per_year = return_rate(c_sa, sa, 100, alt_years)

    analysis_utility_functions.simple_bar_chart(
        proportion_of_complicated_sa_per_year[0], 'Year', '% of Spontaneous Abortions',
        'Proportion of Spontaneous Abortions Leading to Complications', 'miscarriage_prop_complicated', alt_years,
        graph_location)

    # Complicated IA / Total IA
    c_ia = an_comps.loc[(slice(None), 'complicated_induced_abortion'), slice(None)].droplevel(1)
    proportion_of_complicated_ia_per_year = return_rate(c_ia, ia, 100, alt_years)

    analysis_utility_functions.simple_bar_chart(
        proportion_of_complicated_ia_per_year[0], 'Year', '% of Induced Abortions',
        'Proportion of Induced Abortions Leading to Complications', 'abortion_prop_complicated', alt_years,
        graph_location)

    # --------------------------------------------------- Syphilis Rate... --------------------------------------------
    syphilis_data = return_rate(an_comps.loc[(slice(None), 'syphilis'), slice(None)].droplevel(1),
                                total_completed_pregnancies_df, 1000, alt_years)

    target_syph_dict = {'double': False,
                        'first': {'year': 2019, 'value': 20, 'label': 'Integrated HIV program report', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, syphilis_data, target_syph_dict, 30,
        'Rate per 1000 completed pregnancies', 'Syphilis Cases in Pregnancy per 1000 Completed Pregnancies',
        graph_location, 'syphilis_rate')

    # ------------------------------------------------ Gestational Diabetes ------------------------------------------
    gdm_data = return_rate(an_comps.loc[(slice(None), 'gest_diab'), slice(None)].droplevel(1),
                           total_completed_pregnancies_df, 1000, alt_years)

    target_gdm_dict = {'double': False,
                       'first': {'year': 2019, 'value': 16, 'label': 'Phiri et al.', 'ci': (4 - 0.3) / 2}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, gdm_data, target_gdm_dict, 20, 'Rate per 1000 completed pregnancies',
        'Gestational Diabetes Cases per 1000 Completed Pregnancies', graph_location, 'gest_diab_rate', )

    # --------------------------------- Premature Rupture of Membranes (PROM)-----------------------------------------
    prom_data = return_rate(an_comps.loc[(slice(None), 'PROM'), slice(None)].droplevel(1), births_results_exc_2010,
                            1000, alt_years)

    target_prm_dict = {'double': False,
                       'first': {'year': 2020, 'value': 27, 'label': 'Onwughara et al.', 'ci': (34 - 19) / 2}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, prom_data, target_prm_dict, 35, 'Rate per 1000 births',
        'Premature Rupture of Membranes (PROM) per 1000 Births', graph_location,
        'prom_rate')

    # ---------------------------------------------- Anaemia... -------------------------------------------------------
    # As anaemia is logged separately from other complications (on birth and at the end of the postnatal period) this
    # is extracted here so that the yearly prevalence can be calculated
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
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'anaemia_status'])['year'].count()))

    pn_pn_severity = extract_results(
        results_folder,
        module="tlo.methods.postnatal_supervisor",
        key="total_mat_pnc_visits",
        custom_generate_series=(
            lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'anaemia'])['mother'].count()))

    def get_anaemia_graphs(df, timing, severity_df):
        """Calculates the yearly prevalence of anaemia on birth/at the end of the postnatal period and outputs
        this as a line graph"""
        target_an_dict = {'double': True,
                          'first': {'year': 2011, 'value': 37.5, 'label': 'DHS 2010', 'ci': 0},
                          'second': {'year': 2015, 'value': 45.1, 'label': 'DHS 2015', 'ci': 0},
                          }
        prev = return_rate(df, births_results_exc_2010, 100, alt_years)

        if timing == 'delivery':
            title = 'Prevalence of Maternal Anaemia at Birth per Year'
            y = 'delivery'
        else:
            title = 'Prevalence of Maternal Anaemia at Six Weeks Post-Birth per Year'
            y = 'six weeks postnatal'

        analysis_utility_functions.line_graph_with_ci_and_target_rate(
            alt_years, prev, target_an_dict, 100,
            f'Prevalence at {y} (%)', title, graph_location,
            f'anaemia_prev_{timing}')

        prevalence_of_mild_anaemia_per_year = return_rate(
            severity_df.loc[(slice(None), 'mild'), slice(None)].droplevel(1), births_results_exc_2010, 100, alt_years)

        prevalence_of_mod_anaemia_per_year = return_rate(
            severity_df.loc[(slice(None), 'moderate'), slice(None)].droplevel(1), births_results_exc_2010, 100,
            alt_years)

        prevalence_of_sev_anaemia_per_year = return_rate(
            severity_df.loc[(slice(None), 'severe'), slice(None)].droplevel(1), births_results_exc_2010, 100, alt_years)

        plt.plot(alt_years, prevalence_of_mild_anaemia_per_year[0], label="mild")
        plt.plot(alt_years, prevalence_of_mod_anaemia_per_year[0], label="moderate")
        plt.plot(alt_years, prevalence_of_sev_anaemia_per_year[0], label="severe")
        plt.xlabel('Year')
        plt.ylabel(f'Prevalence at {timing}')
        plt.title(f'Yearly trends for prevalence of anaemia by severity at {timing}')
        plt.legend()
        plt.savefig(f'{graph_location}/anaemia_by_severity_{timing}.png')
        plt.show()

    get_anaemia_graphs(anaemia_results, 'delivery', preg_an_severity)
    get_anaemia_graphs(pnc_anaemia, 'postnatal', pn_pn_severity)

    # ------------------------------------------- Hypertensive disorders -----------------------------------------------
    # Total cases of each hypertensive disorder across the antenatal, intrapartum and postpartum periods are summed to
    # calculate the final yearly rate
    gh_df = an_comps.loc[(slice(None), 'mild_gest_htn'), slice(None)].droplevel(1) + pn_comps.loc[
        (slice(None), 'mild_gest_htn'), slice(None)].droplevel(1)
    gh_data = return_rate(gh_df, births_results_exc_2010, 1000, alt_years)

    sgh_df = an_comps.loc[(slice(None), 'severe_gest_htn'), slice(None)].droplevel(1) + la_comps.loc[
        (slice(None), 'severe_gest_htn'), slice(None)].droplevel(1) + pn_comps.loc[
        (slice(None), 'severe_gest_htn'), slice(None)].droplevel(1)
    sgh_data = return_rate(sgh_df, births_results_exc_2010, 1000, alt_years)

    mpe_df = an_comps.loc[(slice(None), 'mild_pre_eclamp'), slice(None)].droplevel(1) + pn_comps.loc[
        (slice(None), 'mild_pre_eclamp'), slice(None)].droplevel(1)
    mpe_data = return_rate(mpe_df, births_results_exc_2010, 1000, alt_years)

    spe_df = an_comps.loc[(slice(None), 'severe_pre_eclamp'), slice(None)].droplevel(1) + la_comps.loc[
        (slice(None), 'severe_pre_eclamp'), slice(None)].droplevel(1) + pn_comps.loc[
        (slice(None), 'severe_pre_eclamp'), slice(None)].droplevel(1)
    spe_data = return_rate(spe_df, births_results_exc_2010, 1000, alt_years)

    ec_df = an_comps.loc[(slice(None), 'eclampsia'), slice(None)].droplevel(1) + la_comps.loc[
        (slice(None), 'eclampsia'), slice(None)].droplevel(1) + pn_comps.loc[
        (slice(None), 'eclampsia'), slice(None)].droplevel(1)
    ec_data = return_rate(ec_df, births_results_exc_2010, 1000, alt_years)

    target_gh_dict = {'double': False,
                      'first': {'year': 2019, 'value': 43.8, 'label': 'Noubiap et al.', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, gh_data, target_gh_dict, 55, 'Rate per 1000 births',
        'Mild Gestational Hypertension Cases per 1000 Births per Year', graph_location, 'gest_htn_rate', )

    target_sgh_dict = {'double': False,
                       'first': {'year': 2019, 'value': 5.98, 'label': 'Noubiap et al.', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, sgh_data, target_sgh_dict, 8, 'Rate per 1000 births',
        'Severe Gestational Hypertension Cases per 1000 Births per Year', graph_location, 'severe_gest_htn_rate', )

    target_mpe_dict = {'double': False,
                       'first': {'year': 2019, 'value': 44, 'label': 'Noubiap et al', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, mpe_data, target_mpe_dict, 55, 'Rate per 1000 births',
        'Mild Pre-eclampsia Cases per 1000 Births per Year', graph_location, 'mild_pre_eclampsia_rate')

    target_spe_dict = {'double': False,
                       'first': {'year': 2019, 'value': 22, 'label': 'Noubiap et al', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, spe_data, target_spe_dict, 35, 'Rate per 1000 births',
        'Severe Pre-eclampsia Cases per 1000 Births per Year', graph_location, 'severe_pre_eclampsia_rate')

    target_ec_dict = {'double': False,
                      'first': {'year': 2019, 'value': 10, 'label': 'Vousden et al.', 'ci': 0}}
    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, ec_data, target_ec_dict, 15, 'Rate per 1000 births',
        'Eclampsia Cases per 1000 Births per Year', graph_location, 'eclampsia_rate')

    #  ---------------------------------------------Placenta praevia... ----------------------------------------------
    pp_data = return_rate(an_comps.loc[(slice(None), 'placenta_praevia'), slice(None)].droplevel(1),
                          pregnancy_poll_results, 1000, alt_years)

    target_pp_dict = {'double': False,
                      'first': {'year': 2017, 'value': 5.67, 'label': 'Senkoro et al.', 'ci': 0},
                      }
    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, pp_data, target_pp_dict, 8, 'Rate per 1000 pregnancies',
        'Cases of Placenta Praevia per 1000 Pregnancies per Year', graph_location, 'praevia_rate')

    #  ---------------------------------------------Placental abruption... --------------------------------------------
    pa_df = an_comps.loc[(slice(None), 'placental_abruption'), slice(None)].droplevel(1) + la_comps.loc[
        (slice(None), 'placental_abruption'), slice(None)].droplevel(1)
    pa_data = return_rate(pa_df, births_results_exc_2010, 1000, alt_years)

    target_pa_dict = {'double': False,
                      'first': {'year': 2015, 'value': 3, 'label': 'Macheku et al.', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, pa_data, target_pa_dict, 5, 'Rate per 1000 births',
        'Cases of Placental Abruption per 1000 Births Per Year', graph_location, 'abruption_rate')

    # --------------------------------------------- Antepartum Haemorrhage... -----------------------------------------
    # Both antenatal and intrapartum bleeding are counted here to give the total rate of APH
    aph_df = an_comps.loc[(slice(None), 'mild_mod_antepartum_haemorrhage'), slice(None)].droplevel(1) + an_comps.loc[
        (slice(None), 'severe_antepartum_haemorrhage'), slice(None)].droplevel(1) + la_comps.loc[
        (slice(None), 'mild_mod_antepartum_haemorrhage'), slice(None)].droplevel(1) + la_comps.loc[
        (slice(None), 'severe_antepartum_haemorrhage'), slice(None)].droplevel(1)

    aph_data = return_rate(aph_df, births_results_exc_2010, 1000, alt_years)

    target_aph_dict = {'double': False,
                       'first': {'year': 2015, 'value': 4.6, 'label': 'Malawi EmONC Needs Assessment', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, aph_data, target_aph_dict, 7, 'Rate per 1000 births',
        'Cases of Antepartum & Intrapartum Haemorrhages per 1000 Births per Year', graph_location, 'aph_rate')

    # --------------------------------------------- Preterm birth ... ------------------------------------------------
    ptl_df = la_comps.loc[(slice(None), 'early_preterm_labour'), slice(None)].droplevel(1) + la_comps.loc[
        (slice(None), 'late_preterm_labour'), slice(None)].droplevel(1)
    ptl_data = return_rate(ptl_df, births_results_exc_2010, 100, alt_years)

    target_ptl_dict = {'double': False,
                       'first': {'year': 2014, 'value': 10, 'label': 'Chawanpaiboon et al.', 'ci': (14.3 - 7.4) / 2},
                       }

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, ptl_data, target_ptl_dict, 15, 'Rate per 100 births',
        'Preterm Births per 100 Births per Year', graph_location, 'ptb_rate')

    # In addition to the yearly rate of preterm birth, here the percentage of all preterm births which occur early and
    # late per year are outputted to check the assumed distribution is correct. This is shown in a stacked bar chart
    prop_early = return_rate(la_comps.loc[(slice(None), 'early_preterm_labour'), slice(None)].droplevel(1), ptl_df,
                             100, alt_years)
    prop_late = return_rate(la_comps.loc[(slice(None), 'late_preterm_labour'), slice(None)].droplevel(1), ptl_df,
                            100, alt_years)
    labels = alt_years
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(labels, prop_early[0], width, label='Early Preterm',
           bottom=prop_late[0])
    ax.bar(labels, prop_late[0], width, label='Late Preterm')
    ax.set_ylabel('% of total Preterm Births')
    ax.set_title('Early vs Late Preterm Births')
    ax.legend()
    plt.savefig(f'{graph_location}/early_late_preterm.png')
    plt.show()

    # --------------------------------------------- Post term birth ... -----------------------------------------------
    potl_data = return_rate(la_comps.loc[(slice(None), 'post_term_labour'), slice(None)].droplevel(1),
                            births_results_exc_2010, 100, alt_years)

    target_potl_dict = {'double': False,
                        'first': {'year': 2014, 'value': 3.2, 'label': 'van den Broek et al.', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, potl_data, target_potl_dict, 5, 'Rate per 100 Births',
        'Post term Births per 100 Births per Year', graph_location, 'potl_rate')

    # --------------------------------------------- Obstructed Labour... ----------------------------------------------
    ol_data = return_rate(la_comps.loc[(slice(None), 'obstructed_labour'), slice(None)].droplevel(1),
                          births_results_exc_2010, 1000, alt_years)

    target_ol_dict = {'double': True,
                      'first': {'year': 2011, 'value': 18.3, 'label': 'Malawi EmONC Needs Assessment', 'ci': 0},
                      'second': {'year': 2015, 'value': 33.7, 'label': 'Malawi EmONC Needs Assessment', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, ol_data, target_ol_dict, 45, 'Rate per 1000 births',
        'Cases of Obstructed Labour per 1000 Births per Year', graph_location, 'ol_rate')

    # --------------------------------------------- Uterine rupture... -----------------------------------------------
    ur_data = return_rate(la_comps.loc[(slice(None), 'uterine_rupture'), slice(None)].droplevel(1),
                          births_results_exc_2010, 1000, alt_years)

    target_ur_dict = {'double': True,
                      'first': {'year': 2011, 'value': 1.2, 'label': 'Malawi EmONC Needs Assessment', 'ci': 0},
                      'second': {'year': 2015, 'value': 0.8, 'label': 'Malawi EmONC Needs Assessment', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, ur_data, target_ur_dict, 3, 'Rate per 1000 births',
        'Cases of Uterine Rupture per 1000 Births per Year', graph_location, 'ur_rate')

    # ------------------------------------------ Maternal sepsis  --------------------------------------------------
    sepsis_df = an_comps.loc[(slice(None), 'clinical_chorioamnionitis'), slice(None)].droplevel(1) + la_comps.loc[
        (slice(None), 'sepsis'), slice(None)].droplevel(1) + pn_comps.loc[
        (slice(None), 'sepsis_postnatal'), slice(None)].droplevel(1) + pn_comps.loc[
        (slice(None), 'sepsis'), slice(None)].droplevel(1)

    total_sep_rates = return_rate(sepsis_df, births_results_exc_2010, 1000, alt_years)

    target_sep_dict = {'double': True,
                       'first': {'year': 2011, 'value': 2.34, 'label': 'Malawi EmONC Needs Assessment', 'ci': 0},
                       'second': {'year': 2015, 'value': 1.5, 'label': 'Malawi EmONC Needs Assessment', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, total_sep_rates, target_sep_dict, 3.5, 'Rate per 1000 births',
        'Cases of Maternal Sepsis per 1000 Births per Year', graph_location, 'sepsis_rate')

    # ----------------------------------------- Postpartum Haemorrhage... ---------------------------------------------
    pph_data = pn_comps.loc[(slice(None), 'primary_postpartum_haemorrhage'), slice(None)].droplevel(1) + pn_comps.loc[
        (slice(None), 'secondary_postpartum_haemorrhage'), slice(None)].droplevel(1)

    total_pph_rates = return_rate(pph_data, births_results_exc_2010, 1000, alt_years)

    target_pph_dict = {'double': True,
                       'first': {'year': 2011, 'value': 7.95, 'label': 'Malawi EmONC Needs Assessment', 'ci': 0},
                       'second': {'year': 2015, 'value': 12.8, 'label': 'Malawi EmONC Needs Assessment', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, total_pph_rates, target_pph_dict, 15, 'Rate per 1000 births',
        'Cases of Postpartum Haemorrhage per 1000 births Year', graph_location, 'pph_rate')

    # ----------------------------------------- Fistula... -------------------------------------------------
    of_data = pn_comps.loc[(slice(None), 'vesicovaginal_fistula'), slice(None)].droplevel(1) + pn_comps.loc[
        (slice(None), 'rectovaginal_fistula'), slice(None)].droplevel(1)

    total_fistula_rates = return_rate(of_data, births_results_exc_2010, 1000, alt_years)

    target_fistula_dict = {'double': False,
                           'first': {'year': 2015, 'value': 6, 'label': 'DHS 2015', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, total_fistula_rates, target_fistula_dict, 8, 'Rate per 1000 births',
        'Cases of Obstetric Fistula per 1000 Births per Year', graph_location, 'fistula_rate')

    # ---------------------------Caesarean Section Rate & Assisted Vaginal Delivery Rate... --------------------------
    # Finally, the yearly rate of caesarean section and assisted vaginal delivery is outputted

    # Firstly the total number of women for which delivery data was captured is calculated (i.e. total births) which
    # is used as the denominator
    delivery_mode = extract_results(
        results_folder,
        module="tlo.methods.labour",
        key="delivery_setting_and_mode",
        custom_generate_series=(
            lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'mode'])['mother'].count()),
        do_scaling=False)

    # Using this data the total rate of delivery by each mode is calculated and plotted as a line graph compared to
    # calibration targets
    cs_data = return_rate(delivery_mode.loc[(slice(None), 'caesarean_section'), slice(None)].droplevel(1),
                          births_results_exc_2010, 100, alt_years)

    avd_data = return_rate(delivery_mode.loc[(slice(None), 'instrumental'), slice(None)].droplevel(1),
                           births_results_exc_2010, 100, alt_years)

    target_cs_dict = {'double': True,
                      'first': {'year': 2011, 'value': 3.7, 'label': 'Malawi EmONC Needs Assessment', 'ci': 0},
                      'second': {'year': 2015, 'value': 4, 'label': 'Malawi EmONC Needs Assessment', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, cs_data, target_cs_dict, 5, '% of total births',
        'Percentage of total births delivered via caesarean section', graph_location, 'caesarean_section_rate')

    target_avd_dict = {'double': False,
                       'first': {'year': 2015, 'value': 1, 'label': 'Malawi Intergrated HIV report', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, avd_data, target_avd_dict, 4, '% of total births',
        'Percentage of Total Births Delivered via Assisted Vaginal Delivery ', graph_location, 'avd_rate')

    # Then the CS rate for 2010 and 2015 is compared to the calibration data from the EmONC survey via a barchart
    mdata = [[cs_data[0][0], cs_data[0][5]],
             [cs_data[1][0], cs_data[1][5]],
             [cs_data[2][0], cs_data[2][5]]]
    cdata = [3.7, 4]

    model_ci = [(x - y) / 2 for x, y in zip(mdata[2], mdata[1])]
    N = len(mdata[0])
    ind = np.arange(N)
    width = 0.28
    plt.bar(ind, mdata[0], width, label='Model (95% CI)', yerr=model_ci, color='cornflowerblue')
    plt.bar(ind + width, cdata, width, label='BEmONC Survey', color='forestgreen')
    plt.ylabel('Percentage of Total Births')
    plt.xlabel('Year')
    plt.ylim(0, 10)
    plt.title('Percentage of total births delivered via caesarean section (2010 & 2015)')
    plt.xticks(ind + width / 2, ['2010', '2015'])
    plt.legend()
    plt.savefig(f'{graph_location}/cs_rate_2010_2015.png', bbox_inches="tight")
    plt.show()

    # ================================================ STILLBIRTH ====================================================
    # The antenatal, intrapartum and total stillbirth rates are calculated using the total births
    # (live births + stillbirths) as the denominator

    an_sb_df = (an_stillbirth_results / all_births_df) * 1000
    an_sbr_per_year = analysis_utility_functions.return_95_CI_across_runs(an_sb_df, sim_years)

    ip_sb_df = (ip_stillbirth_results / all_births_df) * 1000
    ip_sbr_per_year = analysis_utility_functions.return_95_CI_across_runs(ip_sb_df, sim_years)

    total_sb_df = ((an_stillbirth_results + ip_stillbirth_results) / all_births_df) * 1000
    total_sbr = analysis_utility_functions.return_95_CI_across_runs(total_sb_df, sim_years)

    # UN IGCME calibration data is inputted here
    un_igcme_data = [[20, 19, 19, 18, 18, 17, 17, 17, 17, 16, 16, 16],
                     [17, 17, 17, 17, 17, 16, 16, 16, 15, 15, 14, 13],
                     [23, 22, 21, 20, 19, 18, 18, 18, 18, 18, 19, 19]]

    # which is adjusted for comparison to the antenatal/intrapartum rates (assuming 50% of stillbirths occur in the
    # respective time periods)
    un_igcme_data_adj = [[(x / 2) for x in un_igcme_data[0]],
                         [(x / 2) for x in un_igcme_data[1]],
                         [(x / 2) for x in un_igcme_data[2]]]

    def get_stillbirth_graphs(rate_data, calib_data, group):
        """ Plots the SBR over time compared to calibration data"""
        fig, ax = plt.subplots()
        ax.plot(sim_years, rate_data[0], label="Model (95% CI)", color='deepskyblue')
        ax.fill_between(sim_years, rate_data[1], rate_data[2], color='b', alpha=.1)
        ax.plot([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021], calib_data[0],
                label="UN IGCME 2020 (Uncertainty interval)", color='green')
        ax.fill_between([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021], calib_data[2],
                        calib_data[1], color='mediumseagreen', alpha=.1)
        if group == 'Total':
            ax.set(ylim=(0, 26))
        else:
            ax.set(ylim=(0, 15))

        plt.xlabel('Year')
        plt.ylabel("Stillbirths per 1000 Total Births")
        if group == 'Total':
            plt.title('Stillbirth Rate per Year')
        else:
            plt.title(f'{group} Stillbirth Rate per Year')
        plt.legend()
        plt.savefig(f'{graph_location}/{group}_sbr.png')
        plt.show()

    get_stillbirth_graphs(an_sbr_per_year, un_igcme_data_adj, 'Antenatal')
    get_stillbirth_graphs(ip_sbr_per_year, un_igcme_data_adj, 'Intrapartum')
    get_stillbirth_graphs(total_sbr, un_igcme_data, 'Total')

    # ==================================================== NEWBORN COMPLICATIONS ======================================
    # Dataframes containing the counts of newborn complications across the newborn and postnatal module are extracted
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

    # ------------------------------------------------- Birth weight... -----------------------------------------------
    lbw_data = return_rate(nb_outcomes_df.loc[(slice(None), 'low_birth_weight'), slice(None)].droplevel(1),
                           births_results_exc_2010, 100, alt_years)

    macro_data = return_rate(nb_outcomes_df.loc[(slice(None), 'macrosomia'), slice(None)].droplevel(1),
                             births_results_exc_2010, 100, alt_years)

    sga_data = return_rate(nb_outcomes_df.loc[(slice(None), 'small_for_gestational_age'), slice(None)].droplevel(1),
                           births_results_exc_2010, 100, alt_years)

    target_lbw_dict = {'double': True,
                       'first': {'year': 2011, 'value': 12, 'label': 'DHS 2010', 'ci': 0},
                       'second': {'year': 2015, 'value': 12, 'label': 'DHS 2015', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, lbw_data, target_lbw_dict, 20, 'Rate per 100 births',
        'Cases of Low Birth Weight per 100 Births per Year', graph_location, 'neo_lbw_prev')

    target_mac_dict = {'double': False,
                       'first': {'year': 2019, 'value': 5.13, 'label': 'Ngwira et al.', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, macro_data, target_mac_dict, 7, 'Rate per 100 births',
        'Cases of Macrosomia per 100 Births per Year', graph_location, 'neo_macrosomia_prev')

    target_sga_dict = {'double': False,
                       'first': {'year': 2011, 'value': 23.2, 'label': 'Lee et al.', 'ci': (27 - 19.1) / 2}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, sga_data, target_sga_dict, 30, 'Rate per 100 births',
        'Cases of Small for Gestational Age per 100 Births Per Year', graph_location, 'neo_sga_prev')

    #  ------------------------------------------- Neonatal sepsis (labour & postnatal) ------------------------------
    ns_df = nb_outcomes_df.loc[(slice(None), 'early_onset_sepsis'), slice(None)].droplevel(1) + nb_outcomes_pn_df.loc[
        (slice(None), 'early_onset_sepsis'), slice(None)].droplevel(1) + nb_outcomes_pn_df.loc[
        (slice(None), 'late_onset_sepsis'), slice(None)].droplevel(1)

    target_nsep_dict = {'double': False,
                        'first': {'year': 2020, 'value': 39.3, 'label': 'Fleischmann et al.', 'ci': (78.1 - 19.4) / 2}}

    total_ns_rates = return_rate(ns_df, births_results_exc_2010, 1000, alt_years)

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, total_ns_rates, target_nsep_dict, 85, 'Rate per 1000 births',
        'Cases of Neonatal Sepsis per 1000 Births per Year', graph_location, 'neo_sepsis_rate')

    #  ------------------------------------------- Neonatal encephalopathy ------------------------------------------
    ne_df = nb_outcomes_df.loc[(slice(None), 'mild_enceph'), slice(None)].droplevel(1) + nb_outcomes_df.loc[
        (slice(None), 'moderate_enceph'), slice(None)].droplevel(1) + nb_outcomes_df.loc[
        (slice(None), 'severe_enceph'), slice(None)].droplevel(1)

    total_enceph_rates = return_rate(ne_df, births_results_exc_2010, 1000, alt_years)

    target_enceph_dict = {'double': False,
                          'first': {'year': 2015, 'value': 18.59, 'label': 'GBD 2015', 'ci': (24.9 - 14.3) / 2}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, total_enceph_rates, target_enceph_dict, 25, 'Rate per 1000 births',
        'Cases of Neonatal Encephalopathy per 1000 Births per year', graph_location, 'neo_enceph_rate')

    # ----------------------------------------- Respiratory Depression ------------------------------------------------
    rd_data = return_rate(nb_outcomes_df.loc[(slice(None), 'not_breathing_at_birth'), slice(None)].droplevel(1),
                          births_results_exc_2010, 1000, alt_years)

    dummy_dict = {'double': False,
                  'first': {'year': 2011, 'value': 0, 'label': 'UNK.', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, rd_data, dummy_dict, 15, 'Rate per 1000 births',
        'Cases of Neonatal Respiratory Depression per 1000 Births per Year', graph_location, 'neo_resp_depression_rate')

    # ----------------------------------------Preterm Respiratory Distress Syndrome -----------------------------------
    rds_data_ptb = return_rate(
        nb_outcomes_df.loc[(slice(None), 'respiratory_distress_syndrome'), slice(None)].droplevel(1),
        ptl_df, 1000, alt_years)

    target_rds_dict = {'double': False,
                       'first': {'year': 2019, 'value': 180, 'label': 'Estimated (see text)', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, rds_data_ptb, target_rds_dict, 190, 'Rate per 1000 preterm births',
        'Rate of Preterm Respiratory Distress Syndrome per 1000 Preterm Births per Year', graph_location,
        'neo_rds_rate_ptb')

    # ---------------------------------All respiratory distress conditions (apnoea at birth) --------------------------
    # Here the total rate of newborns with any respiratory distress is calculated by summing the cases of preterm RDS,
    # neonatal encephalopathy and respiratory depression
    total_not_breathing_df = \
        nb_outcomes_df.loc[(slice(None), 'respiratory_distress_syndrome'), slice(None)].droplevel(1) + \
        nb_outcomes_df.loc[(slice(None), 'not_breathing_at_birth'), slice(None)].droplevel(1) +\
        ne_df

    nb_rate = return_rate(total_not_breathing_df, births_results_exc_2010, 100, alt_years)

    target_nb_dict = {'double': False,
                      'first': {'year': 2014, 'value': 5.7, 'label': 'Vossius et al.', 'ci': 0}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, nb_rate, target_nb_dict, 7, 'Rate per 100 births',
        'Total Cases of All Respiratory Complications per 100 Births Per Year', graph_location,
        'neo_total_not_breathing')

    # ----------------------------------------- Congenital Anomalies -------------------------------------------------
    cba_df = nb_outcomes_df.loc[
                 (slice(None), 'congenital_heart_anomaly'), slice(None)].droplevel(1) + nb_outcomes_df.loc[
        (slice(None), 'limb_or_musculoskeletal_anomaly'), slice(None)].droplevel(1) + nb_outcomes_df.loc[
        (slice(None), 'urogenital_anomaly'), slice(None)].droplevel(1) + nb_outcomes_df.loc[
        (slice(None), 'digestive_anomaly'), slice(None)].droplevel(1) + nb_outcomes_df.loc[
        (slice(None), 'other_anomaly'), slice(None)].droplevel(1)

    rate_of_cba = return_rate(cba_df, births_results_exc_2010, 1000, alt_years)

    target_cba_dict = {'double': False,
                       'first': {'year': 2020, 'value': 20.4, 'label': 'Adane et al.', 'ci': (23.8 - 17) / 2}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        alt_years, rate_of_cba, target_cba_dict, 30, 'Rate per 1000 births',
        'Total Cases of Congenital Birth Anomalies per 1000 Births Per Year', graph_location,
        'cba_total_rate')

    # ====================================== MORTALITY AND DALYS CALIBRATION ==========================================
    # Create folder to store graphs (if it hasnt already been created when ran previously)
    path = f'{outputspath}/calibration_output_graphs_{pop_size}_{results_folder.name}/death'
    if not os.path.isdir(path):
        os.makedirs(f'{outputspath}/calibration_output_graphs_{pop_size}_{results_folder.name}/death')

    graph_location = path

    # Read in data containing total DALYs by cause in Malawi from resource file
    dalys_data = pd.read_csv(Path('./resources/gbd') / 'ResourceFile_Deaths_and_DALYS_GBD2019.CSV')
    gbd_years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]

    # ------------------------------------------ Maternal Mortality Ratio... -------------------------------------------

    # First output direct deaths...
    def get_maternal_death_dfs(scaled):
        """Extracts DFs containing direct and indirect maternal deaths. Function allows for deaths to be returned
        scaled or not scaled"""
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

        # Total HIV deaths are adjusted in line with UN MMEIG approach
        hiv_pd_fill = hiv_pd.fillna(0)
        hiv_indirect = hiv_pd_fill * 0.3
        hiv_indirect_maternal_deaths = hiv_indirect.round(0)

        indirect_deaths_final = indirect_deaths_non_hiv_final + hiv_indirect_maternal_deaths

        total_deaths = direct_deaths_final + indirect_deaths_final

        return {'direct_deaths_final': direct_deaths_final,
                'indirect_deaths_final': indirect_deaths_final,
                'total_deaths': total_deaths}

    # Total, direct and indirect MMR is calculated using this data and total live births
    mat_d_unscaled = get_maternal_death_dfs(False)

    direct_mmr_by_year = return_rate(mat_d_unscaled['direct_deaths_final'], births_results_exc_2010, 100_000,
                                     sim_years)
    indirect_mmr_by_year = return_rate(mat_d_unscaled['indirect_deaths_final'], births_results_exc_2010, 100_000,
                                       sim_years)
    total_mmr_by_year = return_rate(mat_d_unscaled['total_deaths'], births_results_exc_2010, 100_000,
                                    sim_years)

    # Calibration data is defined to be plotted agains model outputs
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

    # Total, direct and indirect MMR are then plotted over time along side calibration data
    for data, title in zip([direct_mmr_by_year, indirect_mmr_by_year, total_mmr_by_year],
                           ['Direct', 'Indirect', 'Total']):

        # Calibration data is adjusted assuming 70% of total maternal deaths are direct and the rest are indirect
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
        plt.errorbar(2010, dhs_10_adj[0], yerr=(dhs_10_adj[2] - dhs_10_adj[1]) / 2, label='DHS 2010 (95% CI)', fmt='o',
                     color='green',
                     ecolor='mediumseagreen',
                     elinewidth=3, capsize=0)
        plt.errorbar(2015, dhs_15_adj[0], yerr=(dhs_15_adj[2] - dhs_15_adj[1]) / 2, label='DHS 2015 (95% CI)', fmt='o',
                     color='green',
                     ecolor='mediumseagreen',
                     elinewidth=3, capsize=0)
        ax.plot(unmmeig_yrs, unmmeig_adj, label="UN MMEIG 2020 (Uncertainty interval)", color='cadetblue')
        ax.fill_between(unmmeig_yrs, unmmeig_l_adj, unmmeig_u_adj, color='cadetblue', alpha=.1)
        ax.plot(gbd_years_mmr, gbd_adj, label="GBD 2019 (lower & upper bounds)", color='darkslateblue')
        ax.fill_between(gbd_years_mmr, gbd_l_adj, gbd_u_adj, color='slateblue', alpha=.1)

        if title == 'Total':
            ax.set(ylim=(0, 900))
        elif title == 'Direct':
            ax.set(ylim=(0, 700))
        else:
            ax.set(ylim=(0, 350))

        plt.xlabel('Year')
        plt.ylabel("Deaths per 100 000 live births")
        if title == 'Total':
            plt.title('Maternal Mortality Ratio per Year')
        else:
            plt.title(f'{title} Maternal Mortality Ratio per Year')

        plt.legend()
        plt.savefig(f'{graph_location}/{title}_mmr.png')
        plt.show()

    # In addition, the percentage of total deaths due to direct and indirect causes per year is outputted as a stacked
    # bar chart
    labels = sim_years
    width = 0.35  # the width of the bars: can also be len(x) sequence
    fig, ax = plt.subplots()
    ax.bar(labels, direct_mmr_by_year[0], width, label='Direct', color='brown')
    ax.bar(labels, indirect_mmr_by_year[0], width, bottom=direct_mmr_by_year[0], label='Indirect', color='lightsalmon')
    ax.set_ylabel('Maternal Deaths per 100,000 live births')
    ax.set_title('Maternal Mortality Ratio per Year')
    ax.legend()
    plt.savefig(f'{graph_location}/total_mmr_bar.png')
    plt.show()

    # ------------------------------------------- Crude Maternal Deaths ----------------------------------------------
    # Next the crude number of total maternal deaths per year of the simulation is compared to estimates from the
    # GBD

    # The mean and 95%CI deaths for each year are calculated from the data extracted above
    mat_d_scaled = get_maternal_death_dfs(True)
    m_deaths = analysis_utility_functions.return_95_CI_across_runs(mat_d_scaled['total_deaths'], gbd_years)

    def extract_deaths_gbd_data(group):
        """Extracts the total deaths per year in Malawi estimated by the GBD for a given cause (either
        Maternal/Neonatal Disorders)"""
        dalys_df = dalys_data.loc[(dalys_data['measure_name'] == 'Deaths') &
                                  (dalys_data['cause_name'] == group) & (dalys_data['Year'] > 2010)]
        gbd_deaths = list()
        gbd_deaths_lq = list()
        gbd_deaths_uq = list()

        for year in gbd_years:
            gbd_deaths.append(dalys_df.loc[(dalys_df['Year'] == year)].sum()['GBD_Est'])
            gbd_deaths_lq.append(dalys_df.loc[(dalys_df['Year'] == year)].sum()['GBD_Lower'])
            gbd_deaths_uq.append(dalys_df.loc[(dalys_df['Year'] == year)].sum()['GBD_Upper'])

        return [gbd_deaths, gbd_deaths_lq, gbd_deaths_uq]

    # Once the model and GBD deaths are extracted they are plotted next to each other for each year of the simulation
    # in a bar chart
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

    # ------------------------------- Cause-specific MMRs (direct complications) ------------------------------------
    # Extract deaths by cause data
    d_r = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'cause'])['year'].count()
        ),
    )
    death_results = d_r.fillna(0)

    def update_dfs_to_replace_missing_causes(df, causes):
        """Checks DF that a predetermined list of outcomes are present in the DF index. If not
        (the outcome did not occur for a given year) a row is added containing zero"""
        t = []
        for year in sim_years:
            for cause in causes:
                if cause not in df.loc[year].index:
                    index = pd.MultiIndex.from_tuples([(year, cause)], names=["year", "cause_of_death"])
                    new_row = pd.DataFrame(columns=df.columns, index=index)
                    f_df = new_row.fillna(0.0)
                    t.append(f_df)
        if t:
            causes_df = pd.concat(t)
            updated_df = pd.concat([df, causes_df])
            return updated_df
        else:
            return df

    death_results = update_dfs_to_replace_missing_causes(death_results, ['antenatal_sepsis'])

    # List each cause of interest
    simplified_causes = ['ectopic_pregnancy', 'abortion', 'severe_pre_eclampsia', 'sepsis', 'uterine_rupture',
                         'postpartum_haemorrhage', 'antepartum_haemorrhage']

    # Define the calibration target data for cause-specific MMR for each complication - calculated from the Malawi
    # EmONC needs assessment data which is plotted against model outputs
    ec_tr = {'double': True, 'first': {'year': 2011, 'value': 1.4, 'label': 'UNK.', 'ci': 0},
             'second': {'year': 2015, 'value': 3, 'label': 'UNK.', 'ci': 0}}
    ab_tr = {'double': True, 'first': {'year': 2011, 'value': 36.8, 'label': 'UNK.', 'ci': 0},
             'second': {'year': 2015, 'value': 20.9, 'label': 'UNK.', 'ci': 0}}
    spe_ec_tr = {'double': True, 'first': {'year': 2011, 'value': 50.6, 'label': 'UNK.', 'ci': 0},
                 'second': {'year': 2015, 'value': 55.3, 'label': 'UNK.', 'ci': 0}}
    sep_tr = {'double': True, 'first': {'year': 2011, 'value': 91, 'label': 'UNK.', 'ci': 0},
              'second': {'year': 2015, 'value': 67.6, 'label': 'UNK.', 'ci': 0}}
    ur_tr = {'double': True, 'first': {'year': 2011, 'value': 55, 'label': 'UNK.', 'ci': 0},
             'second': {'year': 2015, 'value': 43, 'label': 'UNK.', 'ci': 0}}
    pph_tr = {'double': True, 'first': {'year': 2011, 'value': 174.8, 'label': 'UNK.', 'ci': 0},
              'second': {'year': 2015, 'value': 95.4, 'label': 'UNK.', 'ci': 0}}
    aph_tr = {'double': True, 'first': {'year': 2011, 'value': 36.8, 'label': 'UNK.', 'ci': 0},
              'second': {'year': 2015, 'value': 16.9, 'label': 'UNK.', 'ci': 0}}

    trs = [ec_tr, ab_tr, spe_ec_tr, sep_tr, ur_tr, pph_tr, aph_tr]

    # For each of the defined causes the total number of deaths due to that cause is extracted. Where multiple modelled
    # conditions contribute to the same overall cause these have been combined
    for cause, tr in zip(simplified_causes, trs):
        if (cause == 'ectopic_pregnancy') or (cause == 'antepartum_haemorrhage') or (cause == 'uterine_rupture'):
            deaths = death_results.loc[(slice(None), cause), slice(None)].droplevel(1)

        elif cause == 'abortion':
            ia_deaths = death_results.loc[(slice(None), 'induced_abortion'), slice(None)].droplevel(1)
            sa_deaths = death_results.loc[(slice(None), 'spontaneous_abortion'), slice(None)].droplevel(1)
            deaths = ia_deaths + sa_deaths

        elif cause == 'severe_pre_eclampsia':
            spe_deaths = death_results.loc[(slice(None), 'severe_pre_eclampsia'), slice(None)].droplevel(1)
            ec_deaths = death_results.loc[(slice(None), 'eclampsia'), slice(None)].droplevel(1)
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

        mmr = return_rate(deaths, births_results_exc_2010, 100_000, sim_years)
        ylim = mmr[0][0] + 20
        if pd.isna(ylim):
            ylim = 0


        # Plot each yearly cause-specific MMR against calibration plots
        analysis_utility_functions.line_graph_with_ci_and_target_rate(
            sim_years, mmr, tr, ylim, 'Rate per 100,000 births',
            f'Maternal Mortality Ratio per Year for {cause}', graph_location, f'mmr_{cause}')

    # ----------------------- Percentage of total direct deaths by cause -------------------------------------------
    # The proportion of total direct deaths by the causes defined below is extracted and compared to calibration points
    # from the EmONC needs assessment data. As above - where appropriate multiple causes have been simplified under one
    # heading to reasonably match the definitions used in the data source.

    simp_causes = ['ectopic_pregnancy', 'uterine_rupture', 'antepartum_haemorrhage', 'abortion', 'severe_pe_ec', 'pph',
                   'sepsis']
    t = []
    for year in sim_years:
        for cause in simp_causes:
            index = pd.MultiIndex.from_tuples([(year, cause)], names=["year", "cause_of_death"])
            new_df = pd.DataFrame(columns=death_results.columns, index=index)

            if cause == 'abortion':
                new_df.loc[year, cause] = death_results.loc[year, 'induced_abortion'] + \
                                          death_results.loc[year, 'spontaneous_abortion']
                new_df.loc[year, cause] = \
                    (new_df.loc[year, cause] / mat_d_unscaled['direct_deaths_final'].loc[year]) * 100
                t.append(new_df)
            elif cause == 'severe_pe_ec':
                new_df.loc[year, cause] = death_results.loc[year, 'severe_pre_eclampsia'] + \
                                          death_results.loc[year, 'eclampsia']
                new_df.loc[year, cause] = \
                    (new_df.loc[year, cause] / mat_d_unscaled['direct_deaths_final'].loc[year]) * 100
                t.append(new_df)

            elif cause == 'pph':
                new_df.loc[year, cause] = death_results.loc[year, 'postpartum_haemorrhage'] + \
                                          death_results.loc[year, 'secondary_postpartum_haemorrhage']
                new_df.loc[year, cause] = \
                    (new_df.loc[year, cause] / mat_d_unscaled['direct_deaths_final'].loc[year]) * 100
                t.append(new_df)

            elif cause == 'sepsis':
                new_df.loc[year, cause] = death_results.loc[year, 'postpartum_sepsis'] + \
                                          death_results.loc[year, 'intrapartum_sepsis'] + \
                                          death_results.loc[year, 'antenatal_sepsis']
                new_df.loc[year, cause] = \
                    (new_df.loc[year, cause] / mat_d_unscaled['direct_deaths_final'].loc[year]) * 100
                t.append(new_df)

            elif cause == 'aph':
                new_df.loc[year, cause] = death_results.loc[year, 'antepartum'] + \
                                          death_results.loc[year, 'secondary_postpartum_haemorrhage']
                new_df.loc[year, cause] = \
                    (new_df.loc[year, cause] / mat_d_unscaled['direct_deaths_final'].loc[year]) * 100
                t.append(new_df)

            else:
                new_df.loc[year, cause] = death_results.loc[year, cause]
                new_df.loc[year, cause] = \
                    (new_df.loc[year, cause] / mat_d_unscaled['direct_deaths_final'].loc[year]) * 100
                t.append(new_df)

    direct_d_by_cause_df = pd.concat(t)
    all_values_2015 = []
    av_u_ci = []
    av_l_ci = []

    # From the extracted dataframes the mean and CI for 2015 proportions are calculated here
    for cause in simp_causes:
        row = direct_d_by_cause_df.loc[2015, cause]
        all_values_2015.append(row.mean())
        ci = st.t.interval(0.95, len(row) - 1, loc=np.mean(row), scale=st.sem(row))
        av_l_ci.append(ci[0])
        av_u_ci.append(ci[1])

    # Data is plotted against calibration targets from 2015, as such only 2015 model outputs are plotted
    labels = ['Ectopic\nPreg.', 'Uterine\nRupture', 'Antepartum\nHaem.', 'Abortion.', 'Severe Pre-eclamp.\n/Eclampsia.',
              'Postpartum\nHaem', 'Sepsis']
    model = all_values_2015
    bemonc_data = [0.1, 14, 5.3, 6.8, 18, 31, 22]  # order = ectopic,

    ui = [(x - y) / 2 for x, y in zip(av_u_ci, av_l_ci)]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, model, width, yerr=ui, label='Model 2015 (95% CI)', color='cornflowerblue')
    ax.bar(x + width / 2, bemonc_data, width, label='BEmONC Survey 2015', color='forestgreen')
    ax.set_ylabel('% of Total Direct Deaths in 2015')
    ax.set_xlabel('Cause of Death')
    ax.set_title('Percentage of Total Direct Maternal Deaths by Cause in 2015')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper left')
    fig.tight_layout()
    plt.xticks(fontsize=9)
    plt.savefig(f'{graph_location}/proportions_cause_of_death_2015.png', bbox_inches="tight")
    plt.show()

    # --------------------------------------------- Indirect deaths by cause --------------------------------------
    # Finally for maternal mortality outcomes, indirect maternal mortality is extracted

    # Here the number of deaths due to causes defined in the lambda function below are sourced from demography logging
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

    # Using the listed causes the mean number of deaths due to each cause per year is calculated from the extracted data
    indirect_causes = ['AIDS_non_TB', 'AIDS_TB', 'TB', 'Malaria', 'Suicide', 'ever_stroke', 'diabetes',
                       'chronic_ischemic_hd', 'ever_heart_attack', 'chronic_kidney_disease']

    indirect_deaths_means = {}

    for complication in indirect_causes:
        indirect_deaths_means.update({complication: []})

        # As stated previously, HIV deaths are adjusted in line with UN MMEIG methods
        for year in alt_years:
            if complication in id_by_cause_df.loc[year].index:
                births = births_results_exc_2010.loc[year].mean()
                deaths = id_by_cause_df.loc[year, complication].mean()
                if 'AIDS' in complication:
                    deaths = deaths * 0.3
                indirect_deaths_means[complication].append((deaths / births) * 100000)
            else:
                indirect_deaths_means[complication].append(0)

    # The total indirect deaths by cause are then plotted as a stacked barchart
    labels = alt_years
    width = 0.35
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
           bottom=[a + b + c + d + e + f + g for a, b, c, d, e, f, g in zip(indirect_deaths_means['AIDS_non_TB'],
                                                                            indirect_deaths_means['Malaria'],
                                                                            indirect_deaths_means['TB'],
                                                                            indirect_deaths_means['Suicide'],
                                                                            indirect_deaths_means['ever_stroke'],
                                                                            indirect_deaths_means['diabetes'],
                                                                            indirect_deaths_means[
                                                                                'chronic_ischemic_hd'])],
           color='darkred')

    ax.bar(labels, indirect_deaths_means['chronic_ischemic_hd'], width, label='Chronic HD',
           bottom=[a + b + c + d + e + f for a, b, c, d, e, f in zip(indirect_deaths_means['AIDS_non_TB'],
                                                                     indirect_deaths_means['Malaria'],
                                                                     indirect_deaths_means['TB'],
                                                                     indirect_deaths_means['Suicide'],
                                                                     indirect_deaths_means['ever_stroke'],
                                                                     indirect_deaths_means['diabetes'])], color='grey')

    ax.bar(labels, indirect_deaths_means['diabetes'], width, label='Diabetes',
           bottom=[a + b + c + d + e for a, b, c, d, e in zip(indirect_deaths_means['AIDS_non_TB'],
                                                              indirect_deaths_means['Malaria'],
                                                              indirect_deaths_means['TB'],
                                                              indirect_deaths_means['Suicide'],
                                                              indirect_deaths_means['ever_stroke'])],
           color='darkorange')

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

    ax.set(ylim=(0, 80))
    ax.set_ylabel('Deaths per 100,000 live births')
    ax.set_xlabel('Year')
    ax.set_title('Yearly Indirect Maternal Deaths By Cause')
    ax.legend()
    plt.xticks(alt_years, labels=alt_years, rotation=45, fontsize=8)
    plt.savefig(f'{graph_location}/indirect_death_mmr_cause.png')
    plt.show()

    # ============================================== NEONATAL MORTALITY ==============================================
    # ------------------------------------------- Neonatal mortality rate --------------------------------------------
    # First the NMR secondary to 'neonatal disorders' (those modelled in the MPHM) is outputted
    death_results_labels = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'label'])['year'].count()))
    nd_nmr = return_rate(death_results_labels.loc[(slice(None), 'Neonatal Disorders'), slice(None)].droplevel(1),
                         births_results_exc_2010, 1000, sim_years)

    # Followed by the total NMR which includes all deaths in the first 28 days of life
    tnd = extract_results(
        results_folder,
        module="tlo.methods.demography.detail",
        key="properties_of_deceased_persons",
        custom_generate_series=(
            lambda df: df.loc[(df['age_days'] <= 28)].assign(
                year=df['date'].dt.year).groupby(['year'])['year'].count()))
    total_neonatal_deaths = tnd.fillna(0)
    tnmr = return_rate(total_neonatal_deaths, births_results_exc_2010, 1000, sim_years)

    def get_nmr_graphs(data, colours, title, save_name):
        """Generate calibration plots for the yearly mean (95%CI) NMR (total and neonatal disorders)"""
        fig, ax = plt.subplots()
        ax.plot(sim_years, data[0], label="Model (95% CI)", color=colours[0])
        ax.fill_between(sim_years, data[1], data[2], color=colours[1], alpha=.1)
        un_yrs = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]

        data = {'dhs': {'mean': [31, 27], 'lq': [26, 22], 'uq': [34, 34]},
                'un': {'mean': [28, 27, 26, 25, 24, 23, 22, 22, 20, 20, 20, 19],
                       'lq': [25, 24, 23, 22, 20, 18, 16, 15, 14, 13, 13, 12],
                       'uq': [31, 31, 30, 29, 29, 28, 28, 29, 29, 30, 30, 31]}}

        plt.errorbar(2011, data['dhs']['mean'][0], label='DHS 2010 (95% CI)',
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
        plt.title(title)
        plt.legend(loc='lower left')
        plt.savefig(f'{graph_location}/{save_name}.png')
        plt.show()

    # Output relevant plots
    get_nmr_graphs(tnmr, ['deepskyblue', 'b'], 'Yearly Total Neonatal Mortality Rate', 'total_nmr')
    get_nmr_graphs(nd_nmr, ['deepskyblue', 'b'], 'Yearly NMR due to GBD "Neonatal Disorders"', 'neonatal_disorders_nmr')

    # Here an additional plot is outputted where both the 'total' NMR and the 'neonatal disorders' NMR are plotted
    # on the same graph
    fig, ax = plt.subplots()
    ax.plot(sim_years, tnmr[0], label="Model: Total NMR(95% CI)", color='deepskyblue')
    ax.fill_between(sim_years, tnmr[1], tnmr[2], color='b', alpha=.1)
    #ax.plot(sim_years, nd_nmr[0], label="Model: 'Neonatal Disorders' NMR (95% CI)", color='salmon')
    #ax.fill_between(sim_years, nd_nmr[1], nd_nmr[2], color='r', alpha=.1)

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
    ax.set(ylim=(0, 45))
    plt.xlabel('Year')
    plt.ylabel("Neonatal Deaths per 1000 Live Births")
    plt.title('Neonatal Mortality Rate per Year')
    plt.legend(loc='lower left')
    plt.savefig(f'{graph_location}/all_nmr_one_graph.png')
    plt.show()

    # ---------------------------------- Total Crude Neonatal Deaths (Neonatal Disorders)------------------------------
    # The yearly total neonatal deaths due to 'Neonatal Disorders' are outputted and compared to the GBD estimates

    # Scaled deaths are extracted from the demography module
    scaled_neo_deaths = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=(
            lambda df: df.assign(
                year=df['date'].dt.year).groupby(['year', 'label'])['year'].count()),
        do_scaling=True)

    # Calculate the average total deaths per year across the runs
    neo_deaths = analysis_utility_functions.return_95_CI_across_runs(
        scaled_neo_deaths.loc[(slice(None), 'Neonatal Disorders'), slice(None)].droplevel(1), gbd_years)

    # And repeat this for congenital birth defects deaths (which are not counted as neonatal disorders)
    ca_deaths = analysis_utility_functions.return_95_CI_across_runs(
        scaled_neo_deaths.loc[(slice(None), 'Congenital birth defects'), slice(None)].droplevel(1), gbd_years)

    # Deaths estimated by the GBD group for Malawi are extracted
    gbd_neo_deaths = extract_deaths_gbd_data('Neonatal disorders')
    gbd_cba_deaths = extract_deaths_gbd_data('Congenital birth defects')

    def gbd_bar_chart(data, gbd_data, title, save_name):
        """Outputs a barchart which compares crude deaths outputted by the model with estimates from the GBD
        alongside uncertainty"""
        model_ci_neo = [(x - y) / 2 for x, y in zip(data[2], data[1])]
        gbd_ci_neo = [(x - y) / 2 for x, y in zip(gbd_data[2], gbd_data[1])]
        N = len(data[0])
        ind = np.arange(N)
        width = 0.35
        plt.bar(ind, data[0], width, label='Model (95% CI)', yerr=model_ci_neo, color='cornflowerblue')
        plt.bar(ind + width, gbd_data[0], width, label='GBD (Lower & Upper bounds)', yerr=gbd_ci_neo,
                color='forestgreen')
        plt.ylabel('Crude Deaths (scaled)')
        plt.title(title)
        plt.xticks(ind + width / 2, gbd_years)
        plt.xlabel('Year')
        plt.legend(loc='best')
        plt.savefig(f'{graph_location}/{save_name}.png')
        plt.show()

    # Plot crude deaths compared to GBD estimates
    gbd_bar_chart(neo_deaths, gbd_neo_deaths, 'Total Deaths Attributable to "Neonatal Disorders" per Year',
                  'crude_deaths_nd')
    gbd_bar_chart(ca_deaths, gbd_cba_deaths, 'Total Deaths Attributable to "Congenital Birth Defects" per Year',
                  'crude_deaths_cba')

    # ---------------------------------- Percentage of Neonatal Deaths by Cause ---------------------------------------
    # Here, the percentage of total neonatal deaths occuring in 2015 is compared to calibration data

    # Calibration data is defined
    neo_calib_targets_fottrell = [27.3, 25.8, 8.95]  # prematuirty, birth asphyxia, early onset sepsis
    neo_calib_targets_bemonc = [26, 52, 6]  # prematuirty, birth asphyxia, early onset sepsis

    # For the three leading causes of neonatal death, the percentage of deaths due to each cause is extracted from
    # model data for each year
    simp_causes_neo = ['Prematurity', 'Birth Asphyxia', 'Neonatal Sepsis']
    t = []
    for year in sim_years:
        for cause in simp_causes_neo:
            index = pd.MultiIndex.from_tuples([(year, cause)], names=["year", "cause_of_death"])
            new_df = pd.DataFrame(columns=death_results.columns, index=index)

            if cause == 'Prematurity':
                new_df.loc[year, cause] = death_results.loc[year, 'respiratory_distress_syndrome'] + \
                                          death_results.loc[year, 'preterm_other']
                new_df.loc[year, cause] = (new_df.loc[year, cause] / total_neonatal_deaths.loc[year]) * 100
                t.append(new_df)
            elif cause == 'Birth Asphyxia':
                new_df.loc[year, cause] = death_results.loc[year, 'encephalopathy'] + \
                                          death_results.loc[year, 'neonatal_respiratory_depression']
                new_df.loc[year, cause] = (new_df.loc[year, cause] / total_neonatal_deaths.loc[year]) * 100
                t.append(new_df)

            elif cause == 'Neonatal Sepsis':
                new_df.loc[year, cause] = death_results.loc[year, 'early_onset_sepsis'] + \
                                          death_results.loc[year, 'late_onset_sepsis']
                new_df.loc[year, cause] = (new_df.loc[year, cause] / total_neonatal_deaths.loc[year]) * 100
                t.append(new_df)

    direct_nd_by_cause_df = pd.concat(t)
    all_values_2015_neo = []
    av_u_ci_neo = []
    av_l_ci_neo = []

    # The values for 2015 are extracted in a format to allow for the plotting of a stacked barchart
    for cause in simp_causes_neo:
        row = direct_nd_by_cause_df.loc[2015, cause]
        all_values_2015_neo.append(row.mean())
        ci = st.t.interval(0.95, len(row) - 1, loc=np.mean(row), scale=st.sem(row))
        av_l_ci_neo.append(ci[0])
        av_u_ci_neo.append(ci[1])

    # Then the model data is plotted compared to the calibration estimates defined above
    ui = [(x - y) / 2 for x, y in zip(av_u_ci_neo, av_l_ci_neo)]
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    x = np.arange(len(simp_causes_neo))
    width = 0.2
    ax.bar(x - width, all_values_2015_neo, width=width, yerr=ui, label='Model', color='cornflowerblue')
    ax.bar(x, neo_calib_targets_fottrell, width=width, label='Fottrell (2015)', color='lightsteelblue')
    ax.bar(x + width, neo_calib_targets_bemonc, width=width, label='EMoNC (2015)', color='forestgreen')
    ax.set_ylabel("% of Total Neonatal Deaths in 2015")
    ax.set_xlabel("Cause of Death")
    ax.set_title("Percentage of Total Neonatal Deaths by Leading Causes in 2015")
    ax.set_xticks(x)
    ax.set_xticklabels(simp_causes_neo)
    ax.legend(loc='upper right')
    ax.tick_params(axis="x", which="both")
    ax.tick_params(axis="y", which="both")
    plt.savefig(f'{graph_location}/proportions_cause_of_death_neo.png')
    plt.show()

    #  ------------------------------------------- Cause-specific NMR ------------------------------------
    # Finally, for neonatal mortality, I output the yearly cause specific NMR for the modelled complications listed
    # below
    simplified_causes = ['prematurity', 'encephalopathy', 'neonatal_sepsis', 'neonatal_respiratory_depression']

    # The estimated calibration targets are defined below for 2010 and 2015, for comparison
    ptb_tr = list()
    enceph_tr = list()
    sep = list()
    rd_tr = list()

    for year in sim_years:
        if year < 2015:
            ptb_tr.append(25 * 0.27)
            enceph_tr.append(25 * 0.25)
            sep.append(25 * 0.08)
            rd_tr.append(0)
        else:
            ptb_tr.append(22 * 0.27)
            enceph_tr.append(22 * 0.25)
            sep.append(22 * 0.08)
            rd_tr.append(0)

    ptb_tr = {'double': True, 'first': {'year': 2011, 'value': 6.75, 'label': 'UNK.', 'ci': 0},
              'second': {'year': 2015, 'value': 5.94, 'label': 'UNK.', 'ci': 0}}
    enceph_tr = {'double': True, 'first': {'year': 2011, 'value': 6.25, 'label': 'UNK.', 'ci': 0},
                 'second': {'year': 2015, 'value': 5.5, 'label': 'UNK.', 'ci': 0}}
    sep = {'double': True, 'first': {'year': 2011, 'value': 2, 'label': 'UNK.', 'ci': 0},
           'second': {'year': 2015, 'value': 1.76, 'label': 'UNK.', 'ci': 0}}
    rd_tr = {'double': True, 'first': {'year': 2011, 'value': 0, 'label': 'UNK.', 'ci': 0},
             'second': {'year': 2015, 'value': 0, 'label': 'UNK.', 'ci': 0}}

    trs = [ptb_tr, enceph_tr, sep, rd_tr]

    # Then the NMR for each complication is calculated from the cause-specific death data frame defined above
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

        nmr = return_rate(deaths, births_results_exc_2010, 1000, sim_years)
        ylim = nmr[0][0] + 5
        print(cause)

        # Each NMR is plotted over time compared to the predefined calibration target
        analysis_utility_functions.line_graph_with_ci_and_target_rate(
            sim_years, nmr, tr, ylim, 'Rate per 1000 live births',
            f'Neonatal Mortality Rate per Year for {cause}', graph_location, f'nmr_{cause}')

    # ==================================== Disability Adjusted Life Years (DALYS) ===================================
    # Finally, DALYs attributable to maternal and neonatal disorders are extracted from the model and compared to
    # estimates from the GBD study

    def extract_dalys_gbd_data(group):
        """Extracts maternal and neonatal DALYs for Malawi as estimated in the 2019 GBD study"""
        dalys_df = dalys_data.loc[(dalys_data['measure_name'] == 'DALYs (Disability-Adjusted Life Years)') &
                                  (dalys_data['cause_name'] == f'{group} disorders') & (dalys_data['Year'] > 2010)]

        gbd_dalys = list()
        gbd_dalys_lq = list()
        gbd_dalys_uq = list()

        # Total dalys are extracted from previously defined DF containing data from Malawi
        for year in gbd_years:
            gbd_dalys.append(dalys_df.loc[(dalys_df['Year'] == year)].sum()['GBD_Est'])
            gbd_dalys_lq.append(dalys_df.loc[(dalys_df['Year'] == year)].sum()['GBD_Lower'])
            gbd_dalys_uq.append(dalys_df.loc[(dalys_df['Year'] == year)].sum()['GBD_Upper'])

        # GBD 'Maternal disorders' DALYs are adjusted to remove 'indirect maternal' and HIV aggravated deaths which
        # do not contribute to maternal DALYs in the model
        if group == 'Maternal':
            mat_dalys_not_modelled = [[9795, 9673, 9484, 9305, 9153, 9220, 9177, 9115, 9128],
                                      [13705, 13588, 13441, 13359, 13230, 13555, 13558, 13506, 13553],
                                      [6304, 6225, 6148, 5878, 5764, 5763, 5664, 5611, 5548]]

            maternal_gbd_dalys_adj = [[x - y for x, y in zip(gbd_dalys, mat_dalys_not_modelled[0])],
                                      [x - y for x, y in zip(gbd_dalys_uq, mat_dalys_not_modelled[1])],
                                      [x - y for x, y in zip(gbd_dalys_lq, mat_dalys_not_modelled[2])]]

            gbd_dalys_rate = [[561, 529, 501, 475, 452, 439, 423, 411, 404],
                              [731, 695, 660, 631, 611, 599, 577, 560, 555],
                              [403, 382, 358, 337, 320, 306, 288, 278, 275]]

            gbd_dalys_rate_adj = [[494, 465, 440, 417, 396, 385, 370, 360, 354],
                                  [638, 606, 574, 547, 531, 520, 499, 484, 482],
                                  [360, 341, 319, 300, 285, 272, 256, 247, 245]]

            return {'total': [gbd_dalys, gbd_dalys_lq, gbd_dalys_uq],
                    'total_adj': maternal_gbd_dalys_adj,
                    'rate': gbd_dalys_rate,
                    'rate_adj': gbd_dalys_rate_adj}
        else:
            # GBD data relating to DALY rates, YLL and YLD estimates are manually inputted

            rate = [[7754, 7388, 7039, 6667, 6323, 6051, 5637, 5402, 5199],
                    [9191, 8784, 8402, 8010, 7798, 7458, 7091, 6901, 6756],
                    [6460, 6115, 5773, 5449, 5167, 4847, 4486, 4246, 4044]]

            yll = [[1116648.79, 1090423.39, 1063931.80, 1031135.75, 1001028.50,
                    979690.75, 932464.88, 917047.17, 907909.52],
                   [926580.96, 898355.11, 868688.00, 838840.05, 806310.38, 806310.38, 730865.10,
                    707276.07, 693996.45],
                   [1325484.32, 1303762.52, 1278501.75, 1249389.10, 1243491.38, 1216195.29, 1187856.17,
                    1186101.78, 1190897.86]]

            yld = [[22980.05, 27396.62, 32943.15, 38722.71, 43760.64, 48920.63, 52681.07, 52985.03, 50919.88],
                   [17003.09, 20257.22, 24621.60, 28670.33, 32352.21, 36966.02, 39836.19, 39901.68,
                    37275.59],
                   [31058.88, 36616.19, 43246.68, 51007.54, 57403.37, 61380.25, 66633.24, 66586.96, 67266.20]]

            return {'total': [gbd_dalys, gbd_dalys_lq, gbd_dalys_uq],
                    'rate': rate,
                    'yll': yll,
                    'yld': yld}

    # Extract GBD DALYs data
    maternal_gbd_dalys = extract_dalys_gbd_data('Maternal')
    neonatal_gbd_dalys = extract_dalys_gbd_data('Neonatal')

    # Extract DALY data from the model including estimates for YLL/YLD
    dalys_stacked = extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked",
        custom_generate_series=(
            lambda df: df.drop(
                columns=['date', 'sex', 'age_range']).groupby(['year']).sum().stack()),
        do_scaling=True)

    yll_stacked = extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="yll_by_causes_of_death_stacked",
        custom_generate_series=(
            lambda df: df.drop(
                columns=['date', 'sex', 'age_range']).groupby(['year']).sum().stack()),
        do_scaling=True)
    yll_stacked_final = yll_stacked.fillna(0)

    yld = extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="yld_by_causes_of_disability",
        custom_generate_series=(
            lambda df: df.drop(
                columns=['date', 'sex', 'age_range']).groupby(['year']).sum().stack()),
        do_scaling=True)
    yld_final = yld.fillna(0)

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

    # Calculate the average rate of maternal DALYs per year using person years as a denominator
    mat_model_dalys_data.update({'rate': return_rate(
        dalys_stacked.loc[(slice(None), 'Maternal Disorders'), slice(None)].droplevel(1), person_years_total, 100_000,
        alt_years)})

    # Repeat this process for neonates
    neo_model_dalys_data.update(
        {'rate': return_rate(
            dalys_stacked.loc[(slice(None), 'Neonatal Disorders'), slice(None)].droplevel(1), person_years_total,
            100_000,
            alt_years)})

    # Now calculate the average number of crude DALYs attributable to maternal and neonatal disorders
    mat_model_dalys_data.update({'total': analysis_utility_functions.return_95_CI_across_runs(
        dalys_stacked.loc[(slice(None), 'Maternal Disorders'), slice(None)].droplevel(1), alt_years)})
    neo_model_dalys_data.update({'total': analysis_utility_functions.return_95_CI_across_runs(
        dalys_stacked.loc[(slice(None), 'Neonatal Disorders'), slice(None)].droplevel(1), alt_years)})

    # Define the causes of neonatal death and disability which contribute to YLL/YLDs within the model
    neo_causes_death = ['early_onset_sepsis', 'late_onset_sepsis', 'encephalopathy', 'preterm_other',
                        'respiratory_distress_syndrome', 'neonatal_respiratory_depression']

    neo_causes_disab = ['Retinopathy of Prematurity', 'Neonatal Encephalopathy',
                        'Neonatal Sepsis Long term Disability', 'Preterm Birth Disability']

    def get_total_dfs(df, causes):
        """Generates a new DF from an existing DF based on a list of index values"""
        dfs = []
        for k in causes:
            scen_df = df.loc[(slice(None), k), slice(None)].droplevel(1)
            dfs.append(scen_df)

        final_df = sum(dfs)
        return final_df

    # Using this function, a DF containing the total yearly YLL and YLD attributable to the above causes is extracted
    # from model data
    neo_yll_s_df = get_total_dfs(yll_stacked_final, neo_causes_death)
    neo_yld_df = get_total_dfs(yld_final, neo_causes_disab)

    # This allows for calculation of the average number of YLL/YLD due to these conditions per year across the runs
    neo_model_dalys_data.update({'yll': analysis_utility_functions.return_95_CI_across_runs(
        neo_yll_s_df, alt_years)})

    neo_model_dalys_data.update({'yld': analysis_utility_functions.return_95_CI_across_runs(
        neo_yld_df, alt_years)})

    def get_daly_graphs(group, model_data, gbd_data):
        """Generates calibration plots for total DALYs, YLL, YLD attributable to maternal and neonatal disorders within
        the model compared to estimates from the 2019 GBD study"""

        # Plot average total DALYs per year
        fig, ax = plt.subplots()
        ax.plot(alt_years, model_data['total'][0], label="Model (95% CI)", color='deepskyblue')
        ax.fill_between(alt_years, model_data['total'][1], model_data['total'][2], color='b', alpha=.1)
        ax.plot(gbd_years, gbd_data['total'][0], label="GBD (Lower & Upper bounds)", color='olivedrab')
        ax.fill_between(gbd_years, gbd_data['total'][1], gbd_data['total'][2], color='g', alpha=.1)
        if group == 'Maternal':
            ax.set(ylim=(0, 160_000))
        else:
            ax.set(ylim=0)
        plt.xlabel('Year')
        plt.ylabel("DALYs (stacked)")
        plt.title(f'Total DALYs per Year Attributable to {group} disorders')
        plt.legend()
        plt.savefig(f'{graph_location}/{group}_dalys_stacked.png')
        plt.show()

        # Plot average DALY rate per year
        fig, ax = plt.subplots()
        ax.plot(alt_years, model_data['rate'][0], label="Model (95% CI)", color='deepskyblue')
        ax.fill_between(alt_years, model_data['rate'][1], model_data['rate'][2], color='b', alpha=.1)
        ax.plot(gbd_years, gbd_data['rate'][0], label="GBD (Lower & Upper bounds)", color='olivedrab')
        ax.fill_between(gbd_years, gbd_data['rate'][1], gbd_data['rate'][2], color='g', alpha=.1)
        if group == 'Maternal':
            ax.set(ylim=(0, 950))
        else:
            ax.set(ylim=0)
        plt.xlabel('Year')
        plt.ylabel("DALYs per 100k Person Years")
        plt.title(f'Total DALYs per 100,000 Person Years per Year Attributable to {group} Disorders')
        plt.legend()
        plt.savefig(f'{graph_location}/{group}_dalys_stacked_rate.png')
        plt.show()

        if group == 'Neonatal':
            # Plot average YLL per year
            fig, ax = plt.subplots()
            ax.plot(alt_years, model_data['yll'][0], label="Model (95% CI)", color='deepskyblue')
            ax.fill_between(alt_years, model_data['yll'][1], model_data['yll'][2], color='b', alpha=.1)
            ax.plot(gbd_years, gbd_data['yll'][0], label="GBD (Lower & Upper bounds)", color='olivedrab')
            ax.fill_between(gbd_years, gbd_data['yll'][1], gbd_data['yll'][2], color='g', alpha=.1)
            plt.xlabel('Year')
            plt.ylabel("Years of Life Lost")
            plt.title(f'Total Years of Life Lost (YLL) per Year Attributable to {group} Disorders')
            plt.legend()
            plt.savefig(f'{graph_location}/{group}_yll_stacked.png')
            plt.show()

            # Plot average YLD per year
            fig, ax = plt.subplots()
            ax.plot(alt_years, model_data['yld'][0], label="Model (95% CI)", color='deepskyblue')
            ax.fill_between(alt_years, model_data['yld'][1], model_data['yld'][2], color='b', alpha=.1)
            ax.plot(gbd_years, gbd_data['yld'][0], label="GBD (Lower & Upper bounds)", color='olivedrab')
            ax.fill_between(gbd_years, gbd_data['yld'][1], gbd_data['yld'][2], color='g', alpha=.1)
            plt.xlabel('Year')
            plt.ylabel("Years Live with Disability")
            plt.title(f'Total Years Lived with Disability (YLD) per Year Attributable to {group} Disorders')
            plt.legend()
            plt.savefig(f'{graph_location}/{group}_yld.png')
            plt.show()

    # Output DALY calibration plots
    get_daly_graphs('Maternal', mat_model_dalys_data, maternal_gbd_dalys)
    get_daly_graphs('Neonatal', neo_model_dalys_data, neonatal_gbd_dalys)
