import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo.analysis.utils import extract_results, get_scenario_outputs

from .. import analysis_utility_functions

plt.style.use('seaborn')


def output_all_death_calibration_per_year(scenario_filename, outputspath, pop_size, sim_years, daly_years):
    results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]

    # Create folder to store graphs (if it hasnt already been created when ran previously)
    path = f'{outputspath}/calibration_output_graphs_{pop_size}_{results_folder.name}/death'
    if not os.path.isdir(path):
        os.makedirs(f'{outputspath}/calibration_output_graphs_{pop_size}_{results_folder.name}/death')

    graph_location = path

    # read in daly data
    dalys_data = pd.read_csv(Path('./resources/gbd') / 'ResourceFile_Deaths_and_DALYS_GBD2019.CSV')
    gbd_years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]

    # ============================================HELPER FUNCTIONS... =================================================
    def simple_line_chart_two_targets(model_rate, target_rate_one, target_rate_two, x_title, y_title, title, file_name):
        plt.plot(sim_years, model_rate, 'o-g', label="Model", color='deepskyblue')
        plt.plot(sim_years, target_rate_one, 'o-g', label="Target", color='darkseagreen')
        plt.plot(sim_years, target_rate_two, 'o-g', label="Target (adj.)", color='powderblue')
        plt.xlabel(x_title)
        plt.ylabel(y_title)
        plt.title(title)
        plt.legend()
        plt.savefig(f'{graph_location}/{file_name}.png')
        plt.show()

    def get_target_rate(first_rate, second_rate):
        target_rate = list()
        target_rate_adjusted = list()

        for year in sim_years:
            if year < 2015:
                target_rate.append(first_rate)
                target_rate_adjusted.append(first_rate * 0.64)
            else:
                target_rate.append(second_rate)
                target_rate_adjusted.append(second_rate * 0.70)

        return [target_rate, target_rate_adjusted]

    def get_modules_maternal_complication_dataframes(module):
        c_df = extract_results(
            results_folder,
            module=f"tlo.methods.{module}",
            key="maternal_complication",
            custom_generate_series=(
                lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'type'])['person'].count()),
            do_scaling=False
        )
        complications_df= c_df.fillna(0)

        return complications_df

    #  COMPLICATION DATA FRAMES....
    an_comps = get_modules_maternal_complication_dataframes('pregnancy_supervisor')
    la_comps = get_modules_maternal_complication_dataframes('labour')
    pn_comps = get_modules_maternal_complication_dataframes('postnatal_supervisor')

    # ============================================  Total births... ===================================================
    # births_results = extract_results(
    #     results_folder,
    #    module="tlo.methods.demography",
    #    key="on_birth",
    #    custom_generate_series=(
    #        lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()
    #    ),
    # )

    births_results_exc_2010 = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="on_birth",
        custom_generate_series=(
            lambda df:
            df.loc[(df['mother'] != -1)].assign(year=df['date'].dt.year).groupby(['year'])['year'].count()))

    # birth_data = analysis_utility_functions.get_mean_and_quants(births_results, sim_years)
    # total_births_per_year = birth_data[0]
    total_births_per_year_ex2010_data = analysis_utility_functions.get_mean_and_quants(births_results_exc_2010,
                                                                                       sim_years)
    total_births_per_year_ex2010 = total_births_per_year_ex2010_data[0]
    # =========================================  Direct maternal causes of death... ===================================
    direct_causes = ['ectopic_pregnancy', 'spontaneous_abortion', 'induced_abortion',
                     'severe_gestational_hypertension', 'severe_pre_eclampsia', 'eclampsia', 'antenatal_sepsis',
                     'uterine_rupture', 'intrapartum_sepsis', 'postpartum_sepsis', 'postpartum_haemorrhage',
                     'secondary_postpartum_haemorrhage', 'antepartum_haemorrhage']

    # ==============================================  YEARLY MMR... ==================================================
    # Output direct deaths...
    death_results = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'label'])['year'].count().fillna(0)
        ),
    )
    death_results_labels = death_results.fillna(0)
    mm = analysis_utility_functions.get_comp_mean_and_rate('Maternal Disorders', total_births_per_year_ex2010,
                                                           death_results_labels, 100000, sim_years)

    # Output indirect deaths...
    i_d = extract_results(
        results_folder,
        module="tlo.methods.demography.detail",
        key="properties_of_deceased_persons",
        custom_generate_series=(
            lambda df: df.loc[(df['is_pregnant'] | df['la_is_postpartum']) &
                              df['cause_of_death'].str.contains(
                                  'AIDS_non_TB|AIDS_TB|TB|Malaria|Suicide|ever_stroke|diabetes|chronic_ischemic_hd|'
                                  'ever_heart_attack|chronic_kidney_disease')].assign(
                year=df['date'].dt.year).groupby(['year'])['year'].count()))
    indirect_deaths = i_d.fillna(0)

    id_preg_data = analysis_utility_functions.get_mean_and_quants(indirect_deaths, sim_years)

    id_mmr_data = [[(x / y) * 100000 for x, y in zip(id_preg_data[0], total_births_per_year_ex2010)],
                   [(x / y) * 100000 for x, y in zip(id_preg_data[1], total_births_per_year_ex2010)],
                   [(x / y) * 100000 for x, y in zip(id_preg_data[2], total_births_per_year_ex2010)]]

    total_mmr_data = [[x + y for x, y in zip(id_mmr_data[0], mm[0])],
                      [x + y for x, y in zip(id_mmr_data[1], mm[1])],
                      [x + y for x, y in zip(id_mmr_data[2], mm[2])]]

    for data, title, l_colour, f_colour in zip([mm, id_mmr_data, total_mmr_data],
                                               ['Direct', 'Indirect', 'Total'],
                                               ['deepskyblue', 'mediumpurple', 'coral'],
                                               ['b', 'mediumslateblue', 'lightcoral']):

        if title == 'Direct':
            mp = 0.7
        elif title == 'Indirect':
            mp = 0.3
        else:
            mp = 1

        fig, ax = plt.subplots()
        ax.plot(sim_years, data[0], label="Model (mean)", color=l_colour)
        ax.fill_between(sim_years, data[1], data[2], color=f_colour, alpha=.1)
        plt.errorbar(2010, (675*mp), yerr=((780*mp)-(570*mp))/2, label='DHS 2010', fmt='o', color='green',
                     ecolor='mediumseagreen',
                     elinewidth=3, capsize=0)
        plt.errorbar(2015, (439*mp), yerr=((531*mp)-(348*mp))/2, label='DHS 2015', fmt='o', color='green',
                     ecolor='mediumseagreen',
                     elinewidth=3, capsize=0)
        ax.plot([2011, 2015, 2017], [(444*mp), (370*mp), (349*mp)], label="WHO MMEIG", color='red')
        ax.fill_between([2011, 2015, 2017], [(347*mp), (269*mp), (244*mp)], [(569*mp), (517*mp), (507*mp)],
                        color='pink', alpha=.1)
        ax.plot([2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
                [242*mp, 235*mp, 229*mp, 223*mp, 219*mp, 219*mp, 217*mp, 214*mp, 209*mp],
                label="GBD (2019)", color='black')
        ax.fill_between([2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
                        [168*mp, 165*mp, 158*mp, 151*mp, 150*mp, 146*mp, 141*mp, 141*mp, 134*mp],
                        [324*mp, 313*mp, 310*mp, 307*mp, 304*mp, 307*mp, 304*mp, 300*mp, 294*mp], color='grey',
                        alpha=.1)
        if title == 'Direct':
            ax.set(ylim=(0, 750))
        else:
            ax.set(ylim=(0, 2200))
        plt.xlabel('Year')
        plt.ylabel("Deaths per 100,000 live births")
        plt.title(f'{title} Maternal Mortality Ratio per Year')
        plt.legend()
        plt.savefig(f'{graph_location}/{title}_mmr.png')
        plt.show()

    target_indirect_mmr_dict = {
        'double': True,
        'first': {'year': 2010, 'value': 675*0.3, 'label': 'DHS 2010', 'ci': ((780*0.3)-(570*0.3))/2},
        'second': {'year': 2015, 'value': 439*0.3, 'label': 'DHS 2015', 'ci': ((531*0.3)-(348*0.3))/2}}

    analysis_utility_functions.line_graph_with_ci_and_target_rate(
        sim_years, id_mmr_data[0], id_mmr_data[1], id_mmr_data[2], target_indirect_mmr_dict, '% of total births',
        'Indirect Maternal Mortality Ratio per Year (w/v Target)', graph_location, 'indirect_mmr_w_target')

    labels = sim_years
    width = 0.35  # the width of the bars: can also be len(x) sequence
    fig, ax = plt.subplots()
    ax.bar(labels, mm[0], width, label='Direct', color='brown')
    ax.bar(labels, id_mmr_data[0], width, bottom=mm[0], label='Indirect', color='lightsalmon')
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
                              df['cause_of_death'].str.contains(
                                  'AIDS_non_TB|AIDS_TB|TB|Malaria|Suicide|ever_stroke|diabetes|chronic_ischemic_hd|'
                                  'ever_heart_attack|chronic_kidney_disease')].assign(
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

    ax.set(ylim=(0, 1200))
    ax.set_ylabel('Deaths per 100,000 live births')
    ax.set_ylabel('Year')
    ax.set_title('Indirect Causes of Maternal Death During Pregnancy')
    ax.legend()
    plt.savefig(f'{graph_location}/indirect_death_mmr_cause.png')
    plt.show()

    # ==============================================  DEATHS... ======================================================
    s_d = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'label'])['year'].count()),
        do_scaling=True
    )
    scaled_deaths = s_d.fillna(0)

    m_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(scaled_deaths, 'Maternal Disorders',
                                                                          gbd_years)

    def extract_deaths_gbd_data(group):
        dalys_df = dalys_data.loc[(dalys_data['measure_name'] == 'Deaths') &
                                  (dalys_data['cause_name'] == group) & (dalys_data['Year'] > 2009)]
        gbd_deaths = list()
        gbd_deaths_lq = list()
        gbd_deaths_uq = list()

        for year in daly_years:
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
    plt.bar(ind, m_deaths[0], width, label='Model', yerr=model_ci, color='teal')
    plt.bar(ind + width, gbd_deaths_2010_2019_data[0], width, label='GBD', yerr=gbd_ci, color='olivedrab')
    plt.ylabel('Total Deaths Maternal Deaths (scaled)')
    plt.title('Yearly Modelled Maternal Deaths Compared to GBD')
    plt.xticks(ind + width / 2, gbd_years)
    plt.legend(loc='best')
    plt.savefig(f'{graph_location}/deaths_gbd_comparison.png')
    plt.show()

    # do WHO estiamte also

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

    ec_tr = get_target_rate(18.9, 3.51)
    ab_tr = get_target_rate(51.3, 29.9)
    spe_ec_tr = get_target_rate(64.8, 69.8)
    sep_tr = get_target_rate(120.2, 83)
    ur_tr = get_target_rate(74.3, 55.3)
    pph_tr = get_target_rate(229.5, 116.8)
    aph_tr = get_target_rate(47.3, 23.3)

    trs = [ec_tr, ab_tr, spe_ec_tr, sep_tr, ur_tr, pph_tr, aph_tr]

    for cause, tr in zip(simplified_causes, trs):
        if (cause == 'ectopic_pregnancy') or (cause == 'antepartum_haemorrhage') or (cause == 'uterine_rupture'):
            deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(death_results, cause, sim_years)[0]

        elif cause == 'abortion':
            ia_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                death_results, 'induced_abortion',  sim_years)[0]
            sa_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                death_results, 'spontaneous_abortion', sim_years)[0]
            deaths = [x + y for x, y in zip(ia_deaths, sa_deaths)]

        elif cause == 'severe_pre_eclampsia':
            spe_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                death_results, 'severe_pre_eclampsia', sim_years)[0]
            ec_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                death_results, 'eclampsia', sim_years)[0]
            # we are choosing to include SGH deaths in SPE
            sgh_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                death_results, 'severe_gestational_hypertension', sim_years)[0]
            deaths = [x + y + z for x, y, z in zip(spe_deaths, ec_deaths, sgh_deaths)]

        elif cause == 'postpartum_haemorrhage':
            p_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                death_results, 'postpartum_haemorrhage', sim_years)[0]
            s_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                death_results, 'secondary_postpartum_haemorrhage', sim_years)[0]
            deaths = [x + y for x, y in zip(p_deaths, s_deaths)]

        elif cause == 'sepsis':
            a_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                death_results, 'antenatal_sepsis', sim_years)[0]
            i_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                death_results, 'intrapartum_sepsis', sim_years)[0]
            p_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                death_results, 'postpartum_sepsis', sim_years)[0]

            deaths = [x + y + z for x, y, z in zip(a_deaths, i_deaths, p_deaths)]

        mmr = [(x / y) * 100000 for x, y in zip(deaths, total_births_per_year_ex2010)]
        simple_line_chart_two_targets(mmr, tr[0], tr[1], 'Year', 'Rate per 100,000 births',
                                      f'Maternal Mortality Ratio per Year for {cause}', f'mmr_{cause}')

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

    def pie_prop_cause_of_death(values, years, labels, title):
        sizes = values
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, shadow=True, startangle=90)
        ax1.axis('equal')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.5, box.height])
        plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5))
        # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title(f'Proportion of total maternal deaths by cause ({title}) {years}')
        plt.savefig(f'{graph_location}/mat_death_by_cause_{title}_{years}.png',
                    bbox_inches="tight")
        plt.show()

    props_df = pd.DataFrame(data=proportions_dicts)
    props_df = props_df.fillna(0)

    # labels = list(props_df.index)
    labels_10 = list()
    labels_15 = list()
    values_10 = list()
    values_15 = list()

    for index in props_df.index:
        values_10.append(props_df.loc[index, slice(2010, 2014)].mean())
        labels_10.append(f'{index} ({round(props_df.loc[index, slice(2010, 2014)].mean(), 2)} %)')
        values_15.append(props_df.loc[index, slice(2015, 2020)].mean())
        labels_15.append(f'{index} ({round(props_df.loc[index, slice(2015, 2020)].mean(), 2)} %)')

    pie_prop_cause_of_death(values_10, '2010_2014', labels_10, 'all')
    pie_prop_cause_of_death(values_15, '2015-2020', labels_15, 'all')

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
    labels_10 = list()
    labels_15 = list()
    all_values = list()
    values_10 = list()
    values_15 = list()
    for column in simplified_df.columns:
        all_values.append(simplified_df[column].mean())
        all_labels.append(f'{column} ({round(simplified_df[column].mean(), 2)} %)')
        values_10.append(simplified_df.loc[slice(2010, 2014), column].mean())
        labels_10.append(f'{column} ({round(simplified_df.loc[slice(2010, 2014), column].mean(), 2)} %)')
        values_15.append(simplified_df.loc[slice(2015, 2020), column].mean())
        labels_15.append(f'{column} ({round(simplified_df.loc[slice(2015, 2020), column].mean(), 2)} %)')

    pie_prop_cause_of_death(values_10, '2010_2014', labels_10, 'combined')
    pie_prop_cause_of_death(values_15, '2015-2020', labels_15, 'combined')
    pie_prop_cause_of_death(all_values, '2010-2020', all_labels, 'total')

    # =========================================== CASE FATALITY PER COMPLICATION ======================================
    tr = list()  # todo:update?
    dummy_denom = list()
    for years in sim_years:
        tr.append(0)
        dummy_denom.append(1)

    mean_ep = analysis_utility_functions.get_mean_and_quants_from_str_df(
        an_comps, 'ectopic_unruptured', sim_years)[0]
    mean_sa = analysis_utility_functions.get_mean_and_quants_from_str_df(
        an_comps, 'complicated_spontaneous_abortion', sim_years)[0]
    mean_ia = analysis_utility_functions.get_mean_and_quants_from_str_df(
        an_comps, 'complicated_induced_abortion', sim_years)[0]
    mean_ur = analysis_utility_functions.get_mean_and_quants_from_str_df(
        la_comps, 'uterine_rupture', sim_years)[0]
    mean_lsep = analysis_utility_functions.get_mean_and_quants_from_str_df(
        la_comps, 'sepsis', sim_years)[0]
    mean_psep = analysis_utility_functions.get_mean_and_quants_from_str_df(
        pn_comps, 'sepsis', sim_years)[0]
    mean_asep = analysis_utility_functions.get_mean_and_quants_from_str_df(
        an_comps, 'clinical_chorioamnionitis', sim_years)[0]

    mean_ppph = analysis_utility_functions.get_mean_and_quants_from_str_df(
        la_comps, 'primary_postpartum_haemorrhage', sim_years)[0]
    mean_spph = analysis_utility_functions.get_mean_and_quants_from_str_df(
        pn_comps, 'secondary_postpartum_haemorrhage', sim_years)[0]

    mean_spe = analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
        'severe_pre_eclamp', dummy_denom, 1, [an_comps, la_comps, pn_comps], sim_years)[0]

    mean_ec = analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
        'eclampsia', dummy_denom, 1, [an_comps, la_comps, pn_comps], sim_years)[0]

    mean_sgh = analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
        'severe_gest_htn', dummy_denom, 1, [an_comps, la_comps, pn_comps], sim_years)[0]

    mm_aph_mean = analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
        'mild_mod_antepartum_haemorrhage', dummy_denom, 1, [an_comps, la_comps], sim_years)[0]

    s_aph_mean = analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
        'severe_antepartum_haemorrhage', dummy_denom, 1, [an_comps, la_comps], sim_years)[0]

    mean_aph = [x + y for x, y in zip(mm_aph_mean, s_aph_mean)]

    for inc_list in [mean_ep, mean_sa, mean_ia, mean_ur, mean_lsep,
                     mean_psep, mean_asep, mean_ppph, mean_spph, mean_spe, mean_ec, mean_sgh, mean_aph]:

        for index, item in enumerate(inc_list):
            if item == 0:
                inc_list[index] = 0.1

        print(inc_list)

    for inc_list, complication in \
        zip([mean_ep, mean_sa, mean_ia, mean_ur, mean_psep, mean_ppph, mean_spph, mean_spe, mean_ec,
             mean_sgh, mean_aph],
            ['ectopic_pregnancy', 'spontaneous_abortion', 'induced_abortion', 'uterine_rupture',
             'postpartum_sepsis', 'postpartum_haemorrhage', 'secondary_postpartum_haemorrhage',
             'severe_pre_eclampsia', 'eclampsia', 'severe_gestational_hypertension', 'antepartum_haemorrhage']):

        cfr = analysis_utility_functions.get_comp_mean_and_rate(
            complication, inc_list, death_results, 100, sim_years)[0]
        print(complication, cfr)
        analysis_utility_functions.simple_line_chart_with_target(
            sim_years, cfr, tr, 'Total CFR', f'Yearly CFR for {complication}', f'{complication}_cfr_per_year',
            graph_location)

    mean_lsep = analysis_utility_functions.get_mean_and_quants_from_str_df(la_comps, 'sepsis', sim_years)[0]
    mean_asep = analysis_utility_functions.get_mean_and_quants_from_str_df(
        an_comps, 'clinical_chorioamnionitis', sim_years)[0]
    total_an_cases = [x + y for x, y in zip(mean_asep, mean_lsep)]

    a_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
        death_results, 'antenatal_sepsis', sim_years)[0]
    i_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
        death_results, 'intrapartum_sepsis', sim_years)[0]

    total_an_sepsis_deaths = [x + y for x, y in zip(a_deaths, i_deaths)]
    an_sep_cfr = [(x/y) * 100 for x, y in zip(total_an_sepsis_deaths, total_an_cases)]
    analysis_utility_functions.simple_line_chart_with_target(
        sim_years, an_sep_cfr, tr, 'Total CFR', 'Yearly CFR for antenatal/intrapartum sepsis',
        'an_ip_sepsis_cfr_per_year', graph_location)

    # todo: issue with incidenec and logging of sepsis
    total_sepsis_cases = [x + y for x, y in zip(total_an_cases, mean_psep)]
    p_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(death_results, 'postpartum_sepsis',
                                                                          sim_years)[0]
    total_sepsis_deaths = [x + y for x, y in zip(p_deaths, total_an_sepsis_deaths)]
    sep_cfr = [(x/y) * 100 for x, y in zip(total_sepsis_deaths, total_sepsis_cases)]
    analysis_utility_functions.simple_line_chart_with_target(
        sim_years, sep_cfr, tr, 'Total CFR', 'Yearly CFR for Sepsis (combined)', 'combined_sepsis_cfr_per_year',
        graph_location)

    total_pph_cases = [x + y for x, y in zip(mean_ppph, mean_spph)]
    p_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
        death_results, 'postpartum_haemorrhage', sim_years)[0]
    s_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
        death_results, 'secondary_postpartum_haemorrhage', sim_years)[0]
    total_pph_deaths = [x + y for x, y in zip(p_deaths, s_deaths)]
    cfr = [(x/y) * 100 for x, y in zip(total_pph_deaths, total_pph_cases)]
    analysis_utility_functions.simple_line_chart_with_target(
        sim_years, cfr, tr, 'Total CFR', 'Yearly CFR for PPH (combined)', 'combined_pph_cfr_per_year', graph_location)

    total_ab_cases = [x + y for x, y in zip(mean_ia, mean_sa)]
    ia_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
        death_results, 'induced_abortion', sim_years)[0]
    sa_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
        death_results, 'spontaneous_abortion', sim_years)[0]
    total_ab_deaths = [x + y for x, y in zip(ia_deaths, sa_deaths)]
    cfr = [(x/y) * 100 for x, y in zip(total_ab_deaths, total_ab_cases)]
    analysis_utility_functions.simple_line_chart_with_target(
        sim_years, cfr, tr, 'Total CFR', 'Yearly CFR for Abortion (combined)', 'combined_abortion_cfr_per_year',
        graph_location)

    total_spec_cases = [x + y + z for x, y, z in zip(mean_spe, mean_ec, mean_sgh)]
    spe_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
        death_results, 'severe_pre_eclampsia', sim_years)[0]
    ec_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
        death_results, 'eclampsia', sim_years)[0]
    total_spec_deaths = [x + y + z for x, y, z in zip(spe_deaths, ec_deaths, sgh_deaths)]
    cfr = [(x/y) * 100 for x, y in zip(total_spec_deaths, total_spec_cases)]
    analysis_utility_functions.simple_line_chart_with_target(
        sim_years, cfr, tr, 'Total CFR', 'Yearly CFR for Severe Pre-eclampsia/Eclampsia',
        'combined_spe_ec_cfr_per_year', graph_location)

    # ================================================================================================================
    # =================================================== Neonatal Death ==============================================
    # ================================================================================================================

    # ----------------------------------------------- NEONATAL MORTALITY RATE ----------------------------------------
    # NMR due to neonatal disorders...
    # NEONATAL DISORDERS NMR - ROUGHLY EQUATES TO FIRST WEEK NMR
    nd_nmr = analysis_utility_functions.get_comp_mean_and_rate(
        'Neonatal Disorders', total_births_per_year_ex2010, death_results_labels, 1000, sim_years)

    # Total NMR...(FROM ALL CAUSES UP TO 28 DAYS)
    tnd = extract_results(
        results_folder,
        module="tlo.methods.demography.detail",
        key="properties_of_deceased_persons",
        custom_generate_series=(
            lambda df: df.loc[(df['age_days'] < 28) & (df['cause_of_death'] != 'Other')].assign(
                year=df['date'].dt.year).groupby(['year'])['year'].count()))
    total_neonatal_deaths = tnd.fillna(0)

    t_nm = analysis_utility_functions.get_mean_and_quants(total_neonatal_deaths, sim_years)
    tnmr = [x / y * 1000 for x, y in zip(t_nm[0], total_births_per_year_ex2010_data[0])]
    tnmr_lq = [x / y * 1000 for x, y in zip(t_nm[1], total_births_per_year_ex2010_data[1])]
    tnmr_uq = [x / y * 1000 for x, y in zip(t_nm[2], total_births_per_year_ex2010_data[2])]

    id_nd = extract_results(
        results_folder,
        module="tlo.methods.demography.detail",
        key="properties_of_deceased_persons",
        custom_generate_series=(
            lambda df: df.loc[(df['age_days'] < 28) &
                              df['cause_of_death'].str.contains(
                                  'AIDS_non_TB|AIDS_TB|TB|ALRI|Diarrhoea|Malaria|anomaly')].assign(
                year=df['date'].dt.year).groupby(['year'])['year'].count()))
    indirect_neonatal_deaths = id_nd.fillna(0)

    i_nm = analysis_utility_functions.get_mean_and_quants(indirect_neonatal_deaths, sim_years)
    i_nmr = [x / y * 1000 for x, y in zip(i_nm[0], total_births_per_year_ex2010_data[0])]
    i_tnmr_lq = [x / y * 1000 for x, y in zip(i_nm[1], total_births_per_year_ex2010_data[1])]
    i_tnmr_uq = [x / y * 1000 for x, y in zip(i_nm[2], total_births_per_year_ex2010_data[2])]

    def get_nmr_graphs(data, colours, title, save_name):
        fig, ax = plt.subplots()
        ax.plot(sim_years, data[0], label="Model (mean)", color=colours[0])
        ax.fill_between(sim_years, data[1], data[2], color=colours[1], alpha=.1)

        if save_name != 'neonatal_disorders_nmr':
            data = {'dhs': {'mean': [31, 27], 'lq': [26, 22], 'uq': [34, 34]},
                    'hug': {'mean': 22.7, 'lq': 15.6, 'uq': 28.8},
                    'gbd': {'mean': 25, 'lq': 21.4, 'uq': 26.6},
                    'un': {'mean': [28, 27, 26, 25, 24, 23, 22, 22, 20, 20],
                           'lq': [25, 24, 23, 22, 20, 18, 16, 15, 14, 13],
                           'uq': [31, 31, 30, 29, 29, 28, 28, 29, 29, 30]}}
        else:
            data = {'dhs': {'mean': [25, 22], 'lq': [22, 18], 'uq': [28, 28]},
                    'hug': {'mean': 18.6, 'lq': 13, 'uq': 24},
                    'gbd': {'mean': 20, 'lq': 17.12, 'uq': 21.3},
                    'un': {'mean': [22.4, 21.6, 20.8, 20, 19.2, 18.4, 17.6, 17.6, 16, 16],
                           'lq': [20, 19.2, 18.4, 17.6, 16, 14.4, 12.8, 12, 11.2, 10.4],
                           'uq': [25.6, 24.8, 24, 23.2, 23.2, 22.4, 22.4, 23.2, 23.2, 24]}}

        plt.errorbar(2010, data['dhs']['mean'][0], label='DHS 2010',
                     yerr=(data['dhs']['uq'][0] - data['dhs']['lq'][0]) / 2, fmt='o', color='green',
                     ecolor='mediumseagreen',
                     elinewidth=3, capsize=0)

        plt.errorbar(2015, data['dhs']['mean'][1], label='DHS 2015',
                     yerr=(data['dhs']['uq'][1] - data['dhs']['lq'][1]) / 2, fmt='o', color='black',
                     ecolor='grey',
                     elinewidth=3, capsize=0)

        plt.errorbar(2017, data['hug']['mean'], label='Hug 2017',
                     yerr=(data['hug']['uq'] - data['hug']['lq']) / 2,
                     fmt='o', color='purple', ecolor='pink', elinewidth=3, capsize=0)

        plt.errorbar(2019, data['gbd']['mean'], label='Paulson (GBD) 2019',
                     yerr=(data['gbd']['uq'] - data['gbd']['lq']) / 2,
                     fmt='o', color='purple', ecolor='pink', elinewidth=3, capsize=0)

        ax.plot([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
                data['un']['mean'], label="UN IGCME (unadj.)", color='purple')
        ax.fill_between([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
                        data['un']['lq'], data['un']['uq'], color='grey', alpha=.1)
        ax.set(ylim=(0, 45))
        plt.xlabel('Year')
        plt.ylabel("Rate per 1000 births")
        plt.title(title)
        plt.legend()
        plt.savefig(f'{graph_location}/{save_name}.png')
        plt.show()

    get_nmr_graphs(nd_nmr, ['deepskyblue', 'b'], 'Yearly NMR due to GBD "Neonatal Disorders"', 'neonatal_disorders_nmr')
    get_nmr_graphs([tnmr, tnmr_lq, tnmr_uq], ['coral', 'lightcoral'], 'Yearly Total NMR', 'total_nmr')
    get_nmr_graphs([i_nmr, i_tnmr_lq, i_tnmr_uq], ['seagreen', 'mediumseagreen'],
                   'Yearly NMR due to causes other than "Neonatal Disorders"', 'other_nmr')

    # ------------------------------------------ CRUDE DEATHS PER YEAR ------------------------------------------------
    # Neonatal Disorders...
    neo_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(scaled_deaths, 'Neonatal Disorders',
                                                                            daly_years)
    # Congenital Anomalies...
    ca_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
        scaled_deaths, 'Congenital birth defects', daly_years)

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
        plt.xticks(ind + width / 2, daly_years)
        plt.legend(loc='best')
        plt.savefig(f'{graph_location}/{save_name}.png')
        plt.show()

    gbd_bar_chart(neo_deaths, gbd_neo_deaths, 'Total Deaths Attributable to "Neonatal Disorders" per Year',
                  'crude_deaths_nd')
    gbd_bar_chart(ca_deaths, gbd_cba_deaths, 'Total Deaths Attributable to "Congenital Birth Defects" per Year',
                  'crude_deaths_cba')

    # --------------------------- PROPORTION OF 'NEONATAL DISORDER' DEATHS BY CAUSE -----------------------------------
    direct_neonatal_causes = ['early_onset_sepsis', 'late_onset_sepsis', 'encephalopathy', 'preterm_other',
                              'respiratory_distress_syndrome', 'neonatal_respiratory_depression']

    list_of_proportions_dicts_nb = list()
    total_deaths_per_year_nb = list()

    for year in sim_years:
        yearly_mean_number = list()
        causes = dict()

        for complication in direct_neonatal_causes:
            if complication in death_results.loc[year].index:
                mean = death_results.loc[year, complication].mean()
                yearly_mean_number.append(mean)
                causes.update({f'{complication}_{year}': mean})
            else:
                yearly_mean_number.append(0)

        total_deaths_this_year = sum(yearly_mean_number)
        total_deaths_per_year_nb.append(total_deaths_this_year)

        for complication in causes:
            causes[complication] = (causes[complication] / total_deaths_this_year) * 100

        list_of_proportions_dicts_nb.append(causes)

    # todo: force colours for each complication in each year to be the same
    for year, dictionary in zip(sim_years, list_of_proportions_dicts_nb):
        labels = list()
        sizes = list(dictionary.values())

        for key, value in zip(dictionary.keys(), sizes):
            labels.append(key.replace(f"_{year}", f" ({round(value, 2)} %)"))

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, shadow=True, startangle=90)
        ax1.axis('equal')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.5, box.height])
        plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(f'Proportion of total neonatal deaths by cause in {year}')
        plt.savefig(f'{graph_location}/neo_death_by_cause_{year}.png', bbox_inches="tight")
        plt.show()

    # ------------------------------------------- CASE FATALITY PER COMPLICATION ------------------------------------
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

    tr = list()
    dummy_denom = list()
    for years in sim_years:
        tr.append(0)
        dummy_denom.append(1)

    early_ns = analysis_utility_functions.get_mean_and_quants_from_str_df(
        nb_outcomes_df, 'early_onset_sepsis', sim_years)[0]
    early_ns_pn = analysis_utility_functions.get_mean_and_quants_from_str_df(
        nb_outcomes_pn_df, 'early_onset_sepsis', sim_years)[0]

    total_ens = [x + y for x, y in zip(early_ns, early_ns_pn)]

    late_ns_nb = analysis_utility_functions.get_mean_and_quants_from_str_df(
        nb_outcomes_df, 'late_onset_sepsis', sim_years)[0]
    late_ns_pn = analysis_utility_functions.get_mean_and_quants_from_str_df(
        nb_outcomes_pn_df, 'late_onset_sepsis', sim_years)[0]

    late_ns = [x + y for x, y in zip(late_ns_nb, late_ns_pn)]

    mild_en = analysis_utility_functions.get_mean_and_quants_from_str_df(nb_outcomes_df, 'mild_enceph', sim_years)[0]
    mod_en = analysis_utility_functions.get_mean_and_quants_from_str_df(nb_outcomes_df, 'moderate_enceph', sim_years)[0]
    sev_en = analysis_utility_functions.get_mean_and_quants_from_str_df(nb_outcomes_df, 'severe_enceph', sim_years)[0]
    total_encp = [x + y + z for x, y, z in zip(mild_en, mod_en, sev_en)]

    early_ptl_data = analysis_utility_functions.get_mean_and_quants_from_str_df(
        la_comps, 'early_preterm_labour', sim_years)[0]
    late_ptl_data = analysis_utility_functions.get_mean_and_quants_from_str_df(
        la_comps, 'late_preterm_labour', sim_years)[0]
    total_ptl_rates = [x + y for x, y in zip(early_ptl_data, late_ptl_data)]

    rd = analysis_utility_functions.get_mean_and_quants_from_str_df(
        nb_outcomes_df, 'not_breathing_at_birth', sim_years)[0]

    rds_data = analysis_utility_functions.get_mean_and_quants_from_str_df(
        nb_outcomes_df, 'respiratory_distress_syndrome', sim_years)[0]

    rate_of_ca = analysis_utility_functions.get_mean_and_quants_from_str_df(
        nb_outcomes_df, 'congenital_heart_anomaly', sim_years)[0]
    rate_of_laa = analysis_utility_functions.get_mean_and_quants_from_str_df(
        nb_outcomes_df, 'limb_or_musculoskeletal_anomaly', sim_years)[0]
    rate_of_ua = analysis_utility_functions.get_mean_and_quants_from_str_df(
        nb_outcomes_df, 'urogenital_anomaly', sim_years)[0]
    rate_of_da = analysis_utility_functions.get_mean_and_quants_from_str_df(
        nb_outcomes_df, 'digestive_anomaly', sim_years)[0]
    rate_of_oa = analysis_utility_functions.get_mean_and_quants_from_str_df(
        nb_outcomes_df, 'other_anomaly', sim_years)[0]

    for inc_list in [total_ens, late_ns, total_encp, total_ptl_rates, rds_data, rd, rate_of_ca, rate_of_laa, rate_of_ua,
                     rate_of_da, rate_of_oa]:

        for index, item in enumerate(inc_list):
            if item == 0:
                inc_list[index] = 0.1

    for inc_list, complication in \
        zip([total_ens, late_ns, total_encp, total_ptl_rates, rds_data, rd, rate_of_ca, rate_of_laa, rate_of_ua,
             rate_of_da, rate_of_oa],
            ['early_onset_sepsis', 'late_onset_sepsis', 'encephalopathy', 'preterm_other',
             'respiratory_distress_syndrome', 'neonatal_respiratory_depression',
             'congenital_heart_anomaly', 'limb_or_musculoskeletal_anomaly', 'urogenital_anomaly',
             'digestive_anomaly', 'other_anomaly']):

        cfr = analysis_utility_functions.get_comp_mean_and_rate(
            complication, inc_list, death_results, 100, sim_years)[0]

        analysis_utility_functions.simple_line_chart_with_target(
            sim_years, cfr, tr, 'Total CFR', f'Yearly CFR for {complication}', f'{complication}_neo_cfr_per_year',
            graph_location)

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

    trs = [ptb_tr, enceph_tr, sep, rd_tr, ca_tr]

    for cause, tr in zip(simplified_causes, trs):
        if (cause == 'encephalopathy') or (cause == 'neonatal_respiratory_depression'):
            deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(death_results, cause, sim_years)[0]

        elif cause == 'neonatal_sepsis':
            early = analysis_utility_functions.get_mean_and_quants_from_str_df(
                death_results, 'early_onset_sepsis', sim_years)[0]
            late = analysis_utility_functions.get_mean_and_quants_from_str_df(
                death_results, 'late_onset_sepsis', sim_years)[0]
            deaths = [x + y for x, y in zip(early, late)]

        elif cause == 'prematurity':
            rds_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                death_results, 'respiratory_distress_syndrome', sim_years)[0]
            other_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                death_results, 'preterm_other', sim_years)[0]
            deaths = [x + y for x, y in zip(rds_deaths, other_deaths)]

        elif cause == 'congenital_anomalies':
            ca_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                death_results, 'congenital_heart_anomaly', sim_years)[0]
            la_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                death_results, 'limb_or_musculoskeletal_anomaly', sim_years)[0]
            ua_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                death_results, 'urogenital_anomaly', sim_years)[0]
            da_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                death_results, 'digestive_anomaly', sim_years)[0]
            oa_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                death_results, 'other_anomaly', sim_years)[0]

            deaths = [a + b + c + d + e for a, b, c, d, e in zip(
                ca_deaths, la_deaths, ua_deaths, da_deaths, oa_deaths)]

        nmr = [(x / y) * 1000 for x, y in zip(deaths, total_births_per_year_ex2010)]
        analysis_utility_functions.simple_line_chart_with_target(
            sim_years, nmr, tr, 'Rate per 1000 births', f'Neonatal Mortality Ratio per Year for {cause}',
            f'nmr_{cause}', graph_location)

    # proportion causes for preterm birth
    # -------------------------------------------------------- DALYS -------------------------------------------------
    # todo: i think we just want stacked...

    # add in GBD?

    def extract_dalys_gbd_data(group):
        dalys_df = dalys_data.loc[(dalys_data['measure_name'] == 'DALYs (Disability-Adjusted Life Years)') &
                                  (dalys_data['cause_name'] == f'{group} disorders') & (dalys_data['Year'] > 2009)]

        gbd_dalys = list()
        gbd_dalys_lq = list()
        gbd_dalys_uq = list()

        for year in daly_years:
            gbd_dalys.append(dalys_df.loc[(dalys_df['Year'] == year)].sum()['GBD_Est'])
            gbd_dalys_lq.append(dalys_df.loc[(dalys_df['Year'] == year)].sum()['GBD_Lower'])
            gbd_dalys_uq.append(dalys_df.loc[(dalys_df['Year'] == year)].sum()['GBD_Upper'])

        return [gbd_dalys, gbd_dalys_lq, gbd_dalys_uq]

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

    def extract_dalys_tlo_model(group):
        stacked_dalys = list()
        stacked_dalys_lq = list()
        stacked_dalys_uq = list()

        for year in sim_years:
            if year in dalys_stacked.index:
                stacked_dalys.append(dalys_stacked.loc[year, f'{group} Disorders'].mean())
                stacked_dalys_lq.append(dalys_stacked.loc[year, f'{group} Disorders'].quantile(0.025))
                stacked_dalys_uq.append(dalys_stacked.loc[year, f'{group} Disorders'].quantile(0.925))

        return [stacked_dalys, stacked_dalys_lq, stacked_dalys_uq]

    maternal_dalys = extract_dalys_tlo_model('Maternal')
    neonatal_dalys = extract_dalys_tlo_model('Neonatal')

    def get_daly_graphs(group, dalys, gbd_estimate):
        fig, ax = plt.subplots()
        ax.plot(sim_years, dalys[0], label=f"{group} DALYs", color='deepskyblue')
        ax.fill_between(sim_years, dalys[1], dalys[2], color='b', alpha=.1)

        ax.plot(daly_years, gbd_estimate[0], label="GBD DALY Est.", color='olivedrab')
        ax.fill_between(daly_years, gbd_estimate[1], gbd_estimate[2], color='g', alpha=.1)
        plt.xlabel('Year')
        plt.ylabel("Disability Adjusted Life Years (stacked)")
        plt.title(f'Total DALYs per Year Attributable to {group} disorders')
        plt.legend()
        plt.savefig(f'{graph_location}/{group}_dalys_stacked.png')
        plt.show()

    get_daly_graphs('Maternal', maternal_dalys, maternal_gbd_dalys)
    get_daly_graphs('Neonatal', neonatal_dalys, neonatal_gbd_dalys)

    # todo: move to scenrio files
    # 1.) define HSIs of interest
