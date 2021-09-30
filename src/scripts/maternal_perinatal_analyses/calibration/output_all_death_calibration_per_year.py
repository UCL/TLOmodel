from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo.analysis.utils import (
    extract_results, extract_str_results,
    get_scenario_outputs, create_pickles_locally, summarize
)

# %% Declare the name of the file that specified the scenarios used in this run.
scenario_filename = 'calibration_run_all_modules.py'  # <-- update this to look at other results

# %% Declare usual paths:
outputspath = Path('./outputs/sejjj49@ucl.ac.uk/')
graph_location = 'output_graphs_30k_normal_pop_calibration_run_all_modules-2021-09-29T162954Z/death'
rfp = Path('./resources')

# Find results folder (most recent run generated using that scenario_filename)
results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]
#create_pickles_locally(results_folder)  # if not created via batch

# Enter the years the simulation has ran for here?
sim_years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
# todo: replace with something more clever at some point

# ============================================HELPER FUNCTIONS... =====================================================

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

def get_mean_and_quants(df):
    year_means = list()
    lower_quantiles = list()
    upper_quantiles = list()

    for year in sim_years:
        if year in df.index:
            year_means.append(df.loc[year].mean())
            lower_quantiles.append(df.loc[year].quantile(0.025))
            upper_quantiles.append(df.loc[year].quantile(0.925))
        else:
            year_means.append(0)
            lower_quantiles.append(0)
            lower_quantiles.append(0)

    return [year_means, lower_quantiles, upper_quantiles]


def get_mean_and_quants_from_str_df(df, complication):
    yearly_mean_number = list()
    yearly_lq = list()
    yearly_uq = list()
    for year in sim_years:
        if complication in df.loc[year].index:
            yearly_mean_number.append(df.loc[year, complication].mean())
            yearly_lq.append(df.loc[year, complication].quantile(0.025))
            yearly_uq.append(df.loc[year, complication].quantile(0.925))
        else:
            yearly_mean_number.append(0)
            yearly_lq.append(0)
            yearly_uq.append(0)

    return [yearly_mean_number, yearly_lq, yearly_uq]


def get_comp_mean_and_rate(complication, denominator_list, df, rate):
    yearly_means = get_mean_and_quants_from_str_df(df, complication)[0]
    yearly_lq = get_mean_and_quants_from_str_df(df, complication)[1]
    yearly_uq = get_mean_and_quants_from_str_df(df, complication)[2]

    yearly_mean_rate = [(x / y) * rate for x, y in zip(yearly_means, denominator_list)]
    yearly_lq_rate = [(x / y) * rate for x, y in zip(yearly_lq, denominator_list)]
    yearly_uq_rate = [(x / y) * rate for x, y in zip(yearly_uq, denominator_list)]

    return [yearly_mean_rate, yearly_lq_rate, yearly_uq_rate]


def get_comp_mean_and_rate_across_multiple_dataframes(complication, denominators, rate, dataframes):

    def get_list_of_rates_and_quants(df):
        rates_per_year = list()
        lq_per_year = list()
        uq_per_year = list()
        for year, denominator in zip(sim_years, denominators):
            if year in df.index:
                if complication in df.loc[year].index:
                    rates = (df.loc[year, complication].mean() / denominator) * rate
                    lq = (df.loc[year, complication].quantile(0.025) / denominator) * rate
                    uq = (df.loc[year, complication].quantile(0.925) / denominator) * rate
                    rates_per_year.append(rates)
                    lq_per_year.append(lq)
                    uq_per_year.append(uq)

                else:
                    rates_per_year.append(0)
                    lq_per_year.append(0)
                    uq_per_year.append(0)
            else:
                rates_per_year.append(0)
                lq_per_year.append(0)
                uq_per_year.append(0)

        return [rates_per_year, lq_per_year, uq_per_year]

    if len(dataframes) == 2:
        df_1_data = get_list_of_rates_and_quants(dataframes[0])
        df_2_data = get_list_of_rates_and_quants(dataframes[1])

        total_rates = [x + y for x, y in zip(df_1_data[0], df_2_data[0])]
        total_lq = [x + y for x, y in zip(df_1_data[1], df_2_data[1])]
        total_uq = [x + y for x, y in zip(df_1_data[2], df_2_data[2])]

    else:
        df_1_data = get_list_of_rates_and_quants(dataframes[0])
        df_2_data = get_list_of_rates_and_quants(dataframes[1])
        df_3_data = get_list_of_rates_and_quants(dataframes[2])

        total_rates = [x + y + z for x, y, z in zip(df_1_data[0], df_2_data[0], df_3_data[0])]
        total_lq = [x + y + z for x, y, z in zip(df_1_data[1], df_2_data[1], df_3_data[1])]
        total_uq = [x + y + z for x, y, z in zip(df_1_data[2], df_2_data[2], df_3_data[2])]

    return [total_rates, total_lq, total_uq]


def simple_line_chart(model_rate, target_rate, x_title, y_title, title, file_name):
    plt.plot(sim_years, model_rate, 'o-g', label="Model rate", color='steelblue')
    plt.plot(sim_years, target_rate, 'o-g', label="Target rate", color='darkseagreen')
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    plt.legend()
    plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/{file_name}.png')
    plt.show()


def simple_line_chart_two_targets(model_rate, target_rate_one, target_rate_two, x_title, y_title, title, file_name):
    plt.plot(sim_years, model_rate, 'o-g', label="Model rate", color='steelblue')
    plt.plot(sim_years, target_rate_one, 'o-g', label="Target rate", color='darkseagreen')
    plt.plot(sim_years, target_rate_two, 'o-g', label="Target rate (adj.)", color='powderblue')
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    plt.legend()
    plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/{file_name}.png')
    plt.show()


def simple_bar_chart(model_rates, x_title, y_title, title, file_name):
    bars = sim_years
    x_pos = np.arange(len(bars))
    plt.bar(x_pos, model_rates)
    plt.xticks(x_pos, bars)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/{file_name}.png')
    plt.show()


def line_graph_with_ci_and_target_rate(mean_list, lq_list, uq_list, target_rate, x_label, y_label, title, file_name):
    fig, ax = plt.subplots()
    ax.plot(sim_years, mean_list)
    ax.fill_between(sim_years, lq_list, uq_list, color='b', alpha=.1)
    plt.plot(sim_years, target_rate, 'o-g', label="Target rate", color='darkseagreen')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/{file_name}.png')
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

# ============================================  Total births... ======================================================
births_results = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="on_birth",
    custom_generate_series=(
        lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()
    ),
)
total_births_per_year = get_mean_and_quants(births_results)[0]

# =========================================  Direct maternal causes of death... =======================================
direct_causes = ['ectopic_pregnancy', 'spontaneous_abortion', 'induced_abortion',
                 'severe_gestational_hypertension', 'severe_pre_eclampsia', 'eclampsia', 'antenatal_sepsis',
                 'uterine_rupture', 'intrapartum_sepsis', 'postpartum_sepsis', 'postpartum_haemorrhage',
                 'secondary_postpartum_haemorrhage', 'antepartum_haemorrhage']


# ==============================================  YEARLY MMR... ======================================================
death_results_labels = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="death",
    custom_generate_series=(
        lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'label'])['year'].count()
    ),
)
mm = get_comp_mean_and_rate('Maternal Disorders', total_births_per_year, death_results_labels, 100000)

mmr_rates = get_target_rate(675, 439)

fig, ax = plt.subplots()
ax.plot(sim_years, mm[0])
ax.fill_between(sim_years, mm[1], mm[2], label="Model Output", color='b', alpha=.1)
plt.plot(sim_years, mmr_rates[0], 'o-g', label="Target rate", color='darkseagreen')
plt.plot(sim_years, mmr_rates[1], 'o-g', label="Target rate (Adj.)", color='darkslategrey')
plt.xlabel('Year')
plt.ylabel("Rate per 100,000 births")
plt.title('Maternal Mortality Rate per Year')
plt.legend()
plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/mmr.png')
plt.show()

# =================================== COMPLICATION LEVEL MMR ==========================================================
death_results = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="death",
    custom_generate_series=(
        lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'cause'])['year'].count()
    ),
)

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
        deaths = get_mean_and_quants_from_str_df(death_results, cause)[0]

    elif cause == 'abortion':
        ia_deaths = get_mean_and_quants_from_str_df(death_results, 'induced_abortion')[0]
        sa_deaths = get_mean_and_quants_from_str_df(death_results, 'spontaneous_abortion')[0]
        deaths = [x + y for x, y in zip(ia_deaths, sa_deaths)]

    elif cause == 'severe_pre_eclampsia':
        spe_deaths = get_mean_and_quants_from_str_df(death_results, 'severe_pre_eclampsia')[0]
        ec_deaths = get_mean_and_quants_from_str_df(death_results, 'eclampsia')[0]
        deaths = [x + y for x, y in zip(spe_deaths, ec_deaths)]

    elif cause == 'postpartum_haemorrhage':
        p_deaths = get_mean_and_quants_from_str_df(death_results, 'primary_postpartum_haemorrhage')[0]
        s_deaths = get_mean_and_quants_from_str_df(death_results, 'secondary_postpartum_haemorrhage')[0]
        deaths = [x + y for x, y in zip(p_deaths, s_deaths)]

    elif cause == 'sepsis':
        a_deaths = get_mean_and_quants_from_str_df(death_results, 'antenatal_sepsis')[0]
        i_deaths = get_mean_and_quants_from_str_df(death_results, 'intrapartum_sepsis')[0]
        p_deaths = get_mean_and_quants_from_str_df(death_results, 'postpartum_sepsis')[0]

        deaths = [x + y + z for x, y, z in zip(a_deaths, i_deaths, p_deaths)]

    mmr = [(x / y) * 100000 for x, y in zip(deaths, total_births_per_year)]
    simple_line_chart_two_targets(mmr, tr[0], tr[1], 'Year', 'Rate per 100,000 births',
                                  f'Maternal Mortality Ratio per Year for {cause}', f'mmr_{cause}')


# =================================== DEATH PROPORTIONS... ============================================================

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
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(f'Proportion of total maternal deaths by cause ({title}) {years}')
    plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/mat_death_by_cause_{title}_{years}.png')
    plt.show()

props_df = pd.DataFrame(data=proportions_dicts)
props_df = props_df.fillna(0)

labels = list(props_df.index)
values_10 = list()
values_15 = list()

for index in props_df.index:
    values_10.append(props_df.loc[index, slice(2010, 2014)].mean())
    values_15.append(props_df.loc[index, slice(2015, 2020)].mean())

pie_prop_cause_of_death(values_10, '2010_2014', list(props_df.index), 'all')
pie_prop_cause_of_death(values_15, '2015-2020', list(props_df.index), 'all')

simplified_df = props_df.transpose()

simplified_df['Abortion'] = simplified_df['induced_abortion'] + simplified_df['spontaneous_abortion']
simplified_df['Severe PE/Eclampsia'] = simplified_df['severe_pre_eclampsia'] + simplified_df['eclampsia']
simplified_df['PPH'] = simplified_df['postpartum_haemorrhage'] + simplified_df['secondary_postpartum_haemorrhage']
simplified_df['Sepsis'] = simplified_df['postpartum_sepsis'] + simplified_df['intrapartum_sepsis'] + \
                          simplified_df['antenatal_sepsis']

for column in ['postpartum_haemorrhage', 'secondary_postpartum_haemorrhage', 'severe_pre_eclampsia', 'eclampsia',
               'induced_abortion', 'spontaneous_abortion', 'intrapartum_sepsis', 'postpartum_sepsis',
               'antenatal_sepsis']:
    simplified_df = simplified_df.drop(columns=[column])

all_values = list()
values_10 = list()
values_15 = list()
for column in simplified_df.columns:
    all_values.append(simplified_df[column].mean())
    values_10.append(simplified_df.loc[slice(2010, 2014), column].mean())
    values_15.append(simplified_df.loc[slice(2015, 2020), column].mean())

pie_prop_cause_of_death(values_10, '2010_2014', list(simplified_df.columns), 'combined')
pie_prop_cause_of_death(values_15, '2015-2020', list(simplified_df.columns), 'combined')
pie_prop_cause_of_death(all_values, '2010-2020', list(simplified_df.columns), 'total')


# =========================================== CASE FATALITY PER COMPLICATION ==========================================
tr = list()  # todo:update?
dummy_denom = list()
for years in sim_years:
    tr.append(0)
    dummy_denom.append(1)

mean_ep = get_mean_and_quants_from_str_df(an_comps, 'ectopic_unruptured')[0]
mean_sa = get_mean_and_quants_from_str_df(an_comps, 'complicated_spontaneous_abortion')[0]
mean_ia = get_mean_and_quants_from_str_df(an_comps, 'complicated_induced_abortion')[0]
mean_ur = get_mean_and_quants_from_str_df(la_comps, 'uterine_rupture')[0]
mean_lsep = get_mean_and_quants_from_str_df(la_comps, 'sepsis')[0]
mean_psep = get_mean_and_quants_from_str_df(pn_comps, 'sepsis')[0]
mean_asep = get_mean_and_quants_from_str_df(an_comps, 'clinical_chorioamnionitis')[0]
mean_ppph = get_mean_and_quants_from_str_df(la_comps, 'primary_postpartum_haemorrhage')[0]
mean_spph = get_mean_and_quants_from_str_df(pn_comps, 'secondary_postpartum_haemorrhage')[0]


mean_spe = get_comp_mean_and_rate_across_multiple_dataframes('severe_pre_eclamp', dummy_denom, 1,
                                                             [an_comps, la_comps, pn_comps])[0]
mean_ec = get_comp_mean_and_rate_across_multiple_dataframes('eclampsia', dummy_denom, 1,
                                                             [an_comps, la_comps, pn_comps])[0]
mean_sgh = get_comp_mean_and_rate_across_multiple_dataframes('severe_gest_htn', dummy_denom, 1,
                                                             [an_comps, la_comps, pn_comps])[0]

mm_aph_mean = get_comp_mean_and_rate_across_multiple_dataframes('mild_mod_antepartum_haemorrhage', dummy_denom, 1,
                                                                [an_comps, la_comps])[0]
s_aph_mean = get_comp_mean_and_rate_across_multiple_dataframes('severe_antepartum_haemorrhage',
                                                               dummy_denom, 1, [an_comps, la_comps])[0]
mean_aph = [x + y for x, y in zip(mm_aph_mean, s_aph_mean)]

for inc_list in [mean_ep, mean_sa, mean_ia, mean_ur, mean_lsep,
                mean_psep, mean_asep, mean_ppph, mean_spph, mean_spe, mean_ec, mean_sgh, mean_aph]:

    for index, item in enumerate(inc_list):
        if item == 0:
            inc_list[index] = 0.1

for inc_list, complication in \
    zip([mean_ep, mean_sa, mean_ia, mean_ur, mean_lsep, mean_psep, mean_asep, mean_ppph, mean_spph, mean_spe, mean_ec,
         mean_sgh, mean_aph],
        ['ectopic_pregnancy', 'spontaneous_abortion', 'induced_abortion', 'uterine_rupture', 'intrapartum_sepsis',
         'postpartum_sepsis', 'antenatal_sepsis', 'postpartum_haemorrhage', 'secondary_postpartum_haemorrhage',
         'severe_pre_eclampsia', 'eclampsia', 'severe_gestational_hypertension', 'antepartum_haemorrhage']):

    cfr = get_comp_mean_and_rate(complication, inc_list, death_results, 100)[0]
    simple_line_chart(cfr, tr, 'Year', 'Total CFR', f'Yearly CFR for {complication}',
                      f'{complication}_cfr_per_year')

an = get_comp_mean_and_rate('antenatal_sepsis', mean_asep, death_results, 100)[0]
ip = get_comp_mean_and_rate('intrapartum_sepsis', mean_lsep, death_results, 100)[0]
pp = get_comp_mean_and_rate('postpartum_sepsis', mean_psep, death_results, 100)[0]
mean_cfr = [(x + y + z) / 3 for x, y, z in zip(an, ip, pp)]
simple_line_chart(mean_cfr, tr, 'Year', 'Total CFR', 'Yearly CFR for Sepsis (combined)', 'combined_sepsis_cfr_per_year')

ip = get_comp_mean_and_rate('postpartum_haemorrhage', mean_ppph, death_results, 100)[0]
pp = get_comp_mean_and_rate('secondary_postpartum_haemorrhage', mean_spph, death_results, 100)[0]
mean_cfr = [(x + y) / 2 for x, y in zip(ip, pp)]
simple_line_chart(mean_cfr, tr, 'Year', 'Total CFR', 'Yearly CFR for PPH (combined)', 'combined_pph_cfr_per_year')

ia = get_comp_mean_and_rate('induced_abortion', mean_ia, death_results, 100)[0]
sa = get_comp_mean_and_rate('spontaneous_abortion', mean_sa, death_results, 100)[0]
mean_cfr = [(x + y) / 2 for x, y in zip(ia, sa)]
simple_line_chart(mean_cfr, tr, 'Year', 'Total CFR', 'Yearly CFR for Abortion (combined)',
                  'combined_abortion_cfr_per_year')

spe = get_comp_mean_and_rate('severe_pre_eclampsia', mean_spe, death_results, 100)[0]
ec = get_comp_mean_and_rate('eclampsia', mean_ec, death_results, 100)[0]
mean_cfr = [(x + y) / 2 for x, y in zip(spe, ec)]
simple_line_chart(mean_cfr, tr, 'Year', 'Total CFR', 'Yearly CFR for Severe Pre-eclampsia/Eclampsia (combined)',
                  'combined_spe_ec_cfr_per_year')


# =================================================== Neonatal Death ==================================================

direct_neonatal_causes = ['early_onset_neonatal_sepsis', 'late_onset_sepsis', 'encephalopathy', 'preterm_other',
                          'respiratory_distress_syndrome', 'neonatal_respiratory_depression',
                          'congenital_heart_anomaly', 'limb_or_musculoskeletal_anomaly', 'urogenital_anomaly',
                          'digestive_anomaly', 'other_anomaly']

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


direct_nmr_per_year = [(x/y) * 1000 for x, y in zip(total_deaths_per_year_nb, total_births_per_year)]
target_rate_nmr = list()
for year in sim_years:
    if year < 2015:
        target_rate_nmr.append(25)
    else:
        target_rate_nmr.append(22)

nm = get_comp_mean_and_rate('Neonatal Disorders', total_births_per_year, death_results_labels, 1000)

line_graph_with_ci_and_target_rate(nm[0], nm[1], nm[2], target_rate_nmr, 'Year',
                                   'Rate per 1000 births', 'Neonatal Mortality Rate per Year (using labels)', 'nmr')

# todo: force colours for each complication in each year to be the same
for year, dictionary in zip(sim_years, list_of_proportions_dicts_nb):
    labels = list(dictionary.keys())
    sizes = list(dictionary.values())
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(f'Proportion of total neonatal deaths by cause in {year} ')
    plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/neo_death_by_cause_{year}.png')
    plt.show()






