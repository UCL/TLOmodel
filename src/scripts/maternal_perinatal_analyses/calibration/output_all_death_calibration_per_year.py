from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo.analysis.utils import extract_results, get_scenario_outputs

# %% Declare the name of the file that specified the scenarios used in this run.
scenario_filename = 'standard_mph_calibration.py'  # <-- update this to look at other results

# %% Declare usual paths:
outputspath = Path('./outputs/sejjj49@ucl.ac.uk/')
graph_location = 'ouput_graphs_10k_standard_mph_calibration-2022-01-18T142306Z/death'
rfp = Path('./resources')

# Find results folder (most recent run generated using that scenario_filename)
results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]
# create_pickles_locally(results_folder)  # if not created via batch

# Enter the years the simulation has ran for here?
sim_years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
daly_years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
# todo: replace with something more clever at some point

# read in daly data
dalys_data = pd.read_csv(Path('./resources/gbd') / 'ResourceFile_Deaths_and_DALYS_GBD2019.CSV')


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
    plt.plot(sim_years, model_rate, 'o-g', label="Model", color='deepskyblue')
    plt.plot(sim_years, target_rate, 'o-g', label="Target", color='darkseagreen')
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    plt.legend()
    plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/{file_name}.png')
    plt.show()


def simple_line_chart_two_targets(model_rate, target_rate_one, target_rate_two, x_title, y_title, title, file_name):
    plt.plot(sim_years, model_rate, 'o-g', label="Model", color='deepskyblue')
    plt.plot(sim_years, target_rate_one, 'o-g', label="Target", color='darkseagreen')
    plt.plot(sim_years, target_rate_two, 'o-g', label="Target (adj.)", color='powderblue')
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    plt.legend()
    plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/{file_name}.png')
    plt.show()


def simple_bar_chart(model_rates, x_title, y_title, title, file_name):
    bars = sim_years
    x_pos = np.arange(len(bars))
    plt.bar(x_pos, model_rates, color='thistle')
    plt.xticks(x_pos, bars)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    plt.legend()
    plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/{file_name}.png')
    plt.show()


def line_graph_with_ci_and_target_rate(mean_list, lq_list, uq_list, target_rate, x_label, y_label, title, file_name):
    fig, ax = plt.subplots()
    ax.plot(sim_years, mean_list, 'o-g', label="Model", color='deepskyblue')
    ax.fill_between(sim_years, lq_list, uq_list, color='b', alpha=.1, label="UI (2.5-92.5)")
    plt.plot(sim_years, target_rate, 'o-g', label="Target", color='darkseagreen')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
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

# mmr_rates = get_target_rate(675, 439)

fig, ax = plt.subplots()
ax.plot(sim_years, mm[0], label="Model (mean)", color='deepskyblue')
ax.fill_between(sim_years, mm[1], mm[2], color='b', alpha=.1)
plt.errorbar(2010, 675, yerr=(780-570)/2, label='DHS 2010', fmt='o', color='green', ecolor='mediumseagreen',
             elinewidth=3, capsize=0)
plt.errorbar(2015, 439, yerr=(531-348)/2, label='DHS 2015', fmt='o', color='green', ecolor='mediumseagreen',
             elinewidth=3, capsize=0)
ax.plot([2011, 2015, 2017], [444, 370, 349], label="WHO MMEIG", color='red')
ax.fill_between([2011, 2015, 2017], [347, 269, 244], [569, 517, 507], color='pink', alpha=.1)
ax.plot([2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019], [242, 235, 229, 223, 219, 219, 217, 214, 209],
        label="GBD (2019)", color='black')
ax.fill_between([2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
                [168, 165, 158, 151, 150, 146, 141, 141, 134],
                [324, 313, 310, 307, 304, 307, 304, 300, 294], color='grey', alpha=.1)
ax.set(ylim=(0, 1000))
plt.xlabel('Year')
plt.ylabel("Deaths per 100,000 live births")
plt.title('Maternal Mortality Ratio per Year')
plt.legend()
plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/mmr.png')
plt.show()

# ==============================================  DEATHS... ======================================================
scaled_deaths = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="death",
    custom_generate_series=(
        lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'label'])['year'].count()),
    do_scaling=True
)

deaths = get_mean_and_quants_from_str_df(scaled_deaths, 'Maternal Disorders')


def extract_deaths_gbd_data(group):
    dalys_df = dalys_data.loc[(dalys_data['measure_name'] == 'Deaths') &
                              (dalys_data['cause_name'] == f'{group} disorders') & (dalys_data['Year'] > 2009)]
    gbd_deaths = list()
    gbd_deaths_lq = list()
    gbd_deaths_uq = list()

    for year in daly_years:
        gbd_deaths.append(dalys_df.loc[(dalys_df['Year'] == year)].sum()['GBD_Est'])
        gbd_deaths_lq.append(dalys_df.loc[(dalys_df['Year'] == year)].sum()['GBD_Lower'])
        gbd_deaths_uq.append(dalys_df.loc[(dalys_df['Year'] == year)].sum()['GBD_Upper'])

    return [gbd_deaths, gbd_deaths_lq, gbd_deaths_uq]


gbd_deaths_2010_2019_data = extract_deaths_gbd_data('Maternal')

mean_deaths = list()
death_lq = list()
death_uq = list()
years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
for year in years:
    mean_deaths.append(scaled_deaths.loc[year, 'Maternal Disorders'].mean())
    death_lq.append(scaled_deaths.loc[year, 'Maternal Disorders'].quantile(0.025))
    death_uq.append(scaled_deaths.loc[year, 'Maternal Disorders'].quantile(0.925))


model_ci = [(x - y) / 2 for x, y in zip(death_uq, death_lq)]
gbd_ci = [(x - y) / 2 for x, y in zip(gbd_deaths_2010_2019_data[2], gbd_deaths_2010_2019_data[1])]

N = len(mean_deaths)
ind = np.arange(N)
width = 0.35
plt.bar(ind, mean_deaths, width, label='Model', yerr=model_ci, color='teal')
plt.bar(ind + width, gbd_deaths_2010_2019_data[0], width, label='GBD', yerr=gbd_ci, color='olivedrab')
plt.ylabel('Total Deaths Maternal Deaths (scaled)')
plt.title('Yearly Modelled Maternal Deaths Compared to GBD')
plt.xticks(ind + width / 2, years)
plt.legend(loc='best')
plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/deaths_gbd_comparison.png')
plt.show()

# do WHO estiamte also

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
        # we are choosing to include SGH deaths in SPE
        sgh_deaths = get_mean_and_quants_from_str_df(death_results, 'severe_gestational_hypertension')[0]
        deaths = [x + y + z for x, y, z in zip(spe_deaths, ec_deaths, sgh_deaths)]

    elif cause == 'postpartum_haemorrhage':
        p_deaths = get_mean_and_quants_from_str_df(death_results, 'postpartum_haemorrhage')[0]
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
    ax1.pie(sizes, shadow=True, startangle=90)
    ax1.axis('equal')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.5, box.height])
    plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5))
    # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(f'Proportion of total maternal deaths by cause ({title}) {years}')
    plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/mat_death_by_cause_{title}_{years}.png',
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

    print(inc_list)

for inc_list, complication in \
    zip([mean_ep, mean_sa, mean_ia, mean_ur, mean_psep, mean_ppph, mean_spph, mean_spe, mean_ec,
         mean_sgh, mean_aph],
        ['ectopic_pregnancy', 'spontaneous_abortion', 'induced_abortion', 'uterine_rupture',
         'postpartum_sepsis', 'postpartum_haemorrhage', 'secondary_postpartum_haemorrhage',
         'severe_pre_eclampsia', 'eclampsia', 'severe_gestational_hypertension', 'antepartum_haemorrhage']):

    cfr = get_comp_mean_and_rate(complication, inc_list, death_results, 100)[0]
    print(complication, cfr)
    simple_line_chart(cfr, tr, 'Year', 'Total CFR', f'Yearly CFR for {complication}',
                      f'{complication}_cfr_per_year')

mean_lsep = get_mean_and_quants_from_str_df(la_comps, 'sepsis')[0]
mean_asep = get_mean_and_quants_from_str_df(an_comps, 'clinical_chorioamnionitis')[0]
total_an_cases = [x + y for x, y in zip(mean_asep, mean_lsep)]
a_deaths = get_mean_and_quants_from_str_df(death_results, 'antenatal_sepsis')[0]
i_deaths = get_mean_and_quants_from_str_df(death_results, 'intrapartum_sepsis')[0]
total_an_sepsis_deaths = [x + y for x, y in zip(a_deaths, i_deaths)]
an_sep_cfr = [(x/y) * 100 for x, y in zip(total_an_sepsis_deaths, total_an_cases)]
simple_line_chart(an_sep_cfr, tr, 'Year', 'Total CFR', 'Yearly CFR for antenatal/intrapartum sepsis',
                  'an_ip_sepsis_cfr_per_year')

# todo: issue with incidenec and logging of sepsis
total_sepsis_cases = [x + y for x, y in zip(total_an_cases, mean_psep)]
p_deaths = get_mean_and_quants_from_str_df(death_results, 'postpartum_sepsis')[0]
total_sepsis_deaths = [x + y for x, y in zip(p_deaths, total_an_sepsis_deaths)]
sep_cfr = [(x/y) * 100 for x, y in zip(total_sepsis_deaths, total_sepsis_cases)]
simple_line_chart(sep_cfr, tr, 'Year', 'Total CFR', 'Yearly CFR for Sepsis (combined)', 'combined_sepsis_cfr_per_year')

total_pph_cases = [x + y for x, y in zip(mean_ppph, mean_spph)]
p_deaths = get_mean_and_quants_from_str_df(death_results, 'postpartum_haemorrhage')[0]
s_deaths = get_mean_and_quants_from_str_df(death_results, 'secondary_postpartum_haemorrhage')[0]
total_pph_deaths = [x + y for x, y in zip(p_deaths, s_deaths)]
cfr = [(x/y) * 100 for x, y in zip(total_pph_deaths, total_pph_cases)]
simple_line_chart(cfr, tr, 'Year', 'Total CFR', 'Yearly CFR for PPH (combined)', 'combined_pph_cfr_per_year')

total_ab_cases = [x + y for x, y in zip(mean_ia, mean_sa)]
ia_deaths = get_mean_and_quants_from_str_df(death_results, 'induced_abortion')[0]
sa_deaths = get_mean_and_quants_from_str_df(death_results, 'spontaneous_abortion')[0]
total_ab_deaths = [x + y for x, y in zip(ia_deaths, sa_deaths)]
cfr = [(x/y) * 100 for x, y in zip(total_ab_deaths, total_ab_cases)]
simple_line_chart(cfr, tr, 'Year', 'Total CFR', 'Yearly CFR for Abortion (combined)', 'combined_abortion_cfr_per_year')

total_spec_cases = [x + y + z for x, y, z in zip(mean_spe, mean_ec, mean_sgh)]
spe_deaths = get_mean_and_quants_from_str_df(death_results, 'severe_pre_eclampsia')[0]
ec_deaths = get_mean_and_quants_from_str_df(death_results, 'eclampsia')[0]
total_spec_deaths = [x + y + z for x, y, z in zip(spe_deaths, ec_deaths, sgh_deaths)]
cfr = [(x/y) * 100 for x, y in zip(total_spec_deaths, total_spec_cases)]
simple_line_chart(cfr, tr, 'Year', 'Total CFR', 'Yearly CFR for Severe Pre-eclampsia/Eclampsia',
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

nm = get_comp_mean_and_rate('Neonatal Disorders', total_births_per_year, death_results_labels, 1000)

fig, ax = plt.subplots()
ax.plot(sim_years, nm[0], label="Model (mean)", color='deepskyblue')
ax.fill_between(sim_years, nm[1], nm[2], color='b', alpha=.1)

plt.errorbar(2010, 25, label='DHS 2010 (adj)', yerr=(28-22)/2,  fmt='o', color='green', ecolor='mediumseagreen',
             elinewidth=3, capsize=0)
plt.errorbar(2015, 22, label='DHS 2015 (adj)', yerr=(24-18)/2, fmt='o', color='black',  ecolor='grey',
             elinewidth=3, capsize=0)
plt.errorbar(2017, 18.6, label='Hug 2017 (adj.)', yerr=(24-13)/2, fmt='o', color='purple',  ecolor='pink',
             elinewidth=3, capsize=0)
ax.plot([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
        [28, 27, 26, 25, 24, 23, 22, 22, 20, 20],  label="UN IGCME (unadj.)", color='purple')
ax.fill_between([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
                [25, 24, 23, 22, 20, 18, 16, 15, 14, 13],
                [32, 31, 30, 29, 29, 28, 28, 29, 29, 30], color='grey', alpha=.1)
ax.set(ylim=(0, 45))
plt.xlabel('Year')
plt.ylabel("Rate per 1000 births")
plt.title('Neonatal Mortality Rate per Year')
plt.legend()
plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/nmr.png')
plt.show()

# TOTAL DEATHS
deaths = get_mean_and_quants_from_str_df(scaled_deaths, 'Neonatal Disorders')

gbd_deaths_2010_2019_data_neo = [
    [12179.90, 11997.06, 11721.38, 11112.63, 10796.11, 11454.39, 10097.89, 10589.62, 9927.72,
     9837.758],
    [10192.87, 9966.29, 9653.34, 9071.51, 8732.92, 9408.53, 7857.45, 8413.62, 7661.01, 7523.02],
    [14459.23, 14243.08, 14022.25, 13431.77, 13402.91, 13801.05, 12797.50, 13162.74, 12858.11, 12876.41]]

mean_deaths_n = list()
death_lq_n = list()
death_uq_n = list()
years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
for year in years:
    if 'Neonatal Disorders' in scaled_deaths.loc[year].index:
        mean_deaths_n.append(scaled_deaths.loc[year, 'Neonatal Disorders'].mean())
        death_lq_n.append(scaled_deaths.loc[year, 'Neonatal Disorders'].quantile(0.025))
        death_uq_n.append(scaled_deaths.loc[year, 'Neonatal Disorders'].quantile(0.925))
    else:
        mean_deaths_n.append(0)
        death_lq_n.append(0)
        death_uq_n.append(0)

model_ci_neo = [(x - y) / 2 for x, y in zip(death_uq_n, death_lq_n)]
gbd_ci_neo = [(x - y) / 2 for x, y in zip(gbd_deaths_2010_2019_data_neo[2], gbd_deaths_2010_2019_data_neo[1])]

N = len(mean_deaths_n)
ind = np.arange(N)
width = 0.35
plt.bar(ind, mean_deaths_n, width, label='Model', yerr=model_ci_neo, color='teal')
plt.bar(ind + width, gbd_deaths_2010_2019_data_neo[0], width, label='GBD', yerr=gbd_ci_neo, color='olivedrab')
plt.ylabel('Total Deaths Neonatal Deaths (scaled)')
plt.title('Yearly Modelled Neonatal Deaths Compared to GBD')
plt.xticks(ind + width / 2, years)
plt.legend(loc='best')
plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/deaths_gbd_comparison_neo.png')
plt.show()

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
    plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/neo_death_by_cause_{year}.png', bbox_inches="tight")
    plt.show()

# CASE FATALITY PER COMPLICATION
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

tr = list()
dummy_denom = list()
for years in sim_years:
    tr.append(0)
    dummy_denom.append(1)
mean_ep = get_mean_and_quants_from_str_df(an_comps, 'ectopic_unruptured')[0]

early_ns = get_mean_and_quants_from_str_df(nb_outcomes_df, 'early_onset_sepsis')[0]
early_ns_pn = get_mean_and_quants_from_str_df(nb_outcomes_pn_df, 'early_onset_sepsis')[0]

total_ens = [x + y for x, y in zip(early_ns, early_ns_pn)]

late_ns = get_mean_and_quants_from_str_df(nb_outcomes_pn_df, 'late_onset_sepsis')[0]

mild_en = get_mean_and_quants_from_str_df(nb_outcomes_df, 'mild_enceph')[0]
mod_en = get_mean_and_quants_from_str_df(nb_outcomes_df, 'moderate_enceph')[0]
sev_en = get_mean_and_quants_from_str_df(nb_outcomes_df, 'severe_enceph')[0]
total_encp = [x + y + z for x, y, z in zip(mild_en, mod_en, sev_en)]

early_ptl_data = get_mean_and_quants_from_str_df(la_comps, 'early_preterm_labour')[0]
late_ptl_data = get_mean_and_quants_from_str_df(la_comps, 'late_preterm_labour')[0]
total_ptl_rates = [x + y for x, y in zip(early_ptl_data, late_ptl_data)]

rd = get_mean_and_quants_from_str_df(nb_outcomes_df, 'not_breathing_at_birth')[0]

ept = get_mean_and_quants_from_str_df(la_comps, 'early_preterm_labour')[0]  # todo: should be live births
lpt = get_mean_and_quants_from_str_df(la_comps, 'late_preterm_labour')[0]
total_ptbs = [x + y for x, y in zip(ept, lpt)]

rds_data = get_mean_and_quants_from_str_df(nb_outcomes_df, 'respiratory_distress_syndrome')[0]

rate_of_ca = get_mean_and_quants_from_str_df(nb_outcomes_df, 'congenital_heart_anomaly')[0]
rate_of_laa = get_mean_and_quants_from_str_df(nb_outcomes_df, 'limb_or_musculoskeletal_anomaly')[0]
rate_of_ua = get_mean_and_quants_from_str_df(nb_outcomes_df, 'urogenital_anomaly')[0]
rate_of_da = get_mean_and_quants_from_str_df(nb_outcomes_df, 'digestive_anomaly')[0]
rate_of_oa = get_mean_and_quants_from_str_df(nb_outcomes_df, 'other_anomaly')[0]

for inc_list in [total_ens, late_ns, total_encp, total_ptl_rates, rds_data, rd, rate_of_ca, rate_of_laa, rate_of_ua,
                 rate_of_da, rate_of_oa]:

    for index, item in enumerate(inc_list):
        if item == 0:
            inc_list[index] = 0.1


for inc_list, complication in \
    zip([total_ens, late_ns, total_encp, total_ptl_rates, rds_data, rd, rate_of_ca, rate_of_laa, rate_of_ua, rate_of_da,
        rate_of_oa],
        ['early_onset_neonatal_sepsis', 'late_onset_sepsis', 'encephalopathy', 'preterm_other',
         'respiratory_distress_syndrome', 'neonatal_respiratory_depression',
         'congenital_heart_anomaly', 'limb_or_musculoskeletal_anomaly', 'urogenital_anomaly',
         'digestive_anomaly', 'other_anomaly']):

    cfr = get_comp_mean_and_rate(complication, inc_list, death_results, 100)[0]
    simple_line_chart(cfr, tr, 'Year', 'Total CFR', f'Yearly CFR for {complication}',
                      f'{complication}_neo_cfr_per_year')

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
        deaths = get_mean_and_quants_from_str_df(death_results, cause)[0]

    elif cause == 'neonatal_sepsis':  # TODO: ERROR HERE( also early_onset_sepsis and late_onset_neonatal_sepsis) labels
        early = get_mean_and_quants_from_str_df(death_results, 'early_onset_neonatal_sepsis')[0]
        late = get_mean_and_quants_from_str_df(death_results, 'late_onset_sepsis')[0]
        deaths = [x + y for x, y in zip(early, late)]

    elif cause == 'prematurity':
        rds_deaths = get_mean_and_quants_from_str_df(death_results, 'respiratory_distress_syndrome')[0]
        other_deaths = get_mean_and_quants_from_str_df(death_results, 'preterm_other')[0]
        deaths = [x + y for x, y in zip(rds_deaths, other_deaths)]

    elif cause == 'congenital_anomalies':
        ca_deaths = get_mean_and_quants_from_str_df(death_results, 'congenital_heart_anomaly')[0]
        la_deaths = get_mean_and_quants_from_str_df(death_results, 'limb_or_musculoskeletal_anomaly')[0]
        ua_deaths = get_mean_and_quants_from_str_df(death_results, 'urogenital_anomaly')[0]
        da_deaths = get_mean_and_quants_from_str_df(death_results, 'digestive_anomaly')[0]
        oa_deaths = get_mean_and_quants_from_str_df(death_results, 'other_anomaly')[0]

        deaths = [a + b + c + d + e for a, b, c, d, e in zip(
            ca_deaths, la_deaths, ua_deaths, da_deaths, oa_deaths)]

    nmr = [(x / y) * 1000 for x, y in zip(deaths, total_births_per_year)]
    simple_line_chart(nmr, tr, 'Year', 'Rate per 1000 births', f'Neonatal Mortality Ratio per Year for {cause}',
                      f'nmr_{cause}')


# proportion causes for preterm birth
# DALYS
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
    plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/{group}_dalys_stacked.png')
    plt.show()


get_daly_graphs('Maternal', maternal_dalys, maternal_gbd_dalys)
get_daly_graphs('Neonatal', neonatal_dalys, neonatal_gbd_dalys)


# todo: move to scenrio files
# 1.) define HSIs of interest
