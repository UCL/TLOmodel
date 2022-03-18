from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from scripts.maternal_perinatal_analyses import analysis_utility_functions
from tlo.analysis.utils import extract_results, get_scenario_outputs, load_pickled_dataframes

# from tlo.methods.demography import get_scaling_factor


# %% Declare the name of the file that specified the scenarios used in this run.
baseline_scenario_filename = 'baseline_anc_scenario.py'
intervention_scenario_filename = 'increased_anc_scenario.py'

# %% Declare usual paths:
outputspath = Path('./outputs/sejjj49@ucl.ac.uk/')
graph_location = 'analysis_test_baseline_vs_increased_anc_scenario_(10k)-2021-11-15T140735Z'
rfp = Path('./resources')

# Find results folder (most recent run generated using that scenario_filename)
baseline_results_folder = get_scenario_outputs(baseline_scenario_filename, outputspath)[-1]
intervention_results_folder = get_scenario_outputs(intervention_scenario_filename, outputspath)[-1]

sim_years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027,
             2028, 2029, 2030]
intervention_years = [2020, 2021, 2022, 2023, 2024, 2025]


# GET BIRTHS...
def get_total_births_per_year(folder):
    births_results = extract_results(
        folder,
        module="tlo.methods.demography",
        key="on_birth",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()
        ),
    )
    total_births_per_year = analysis_utility_functions.get_mean_and_quants(births_results, intervention_years)[0]
    return total_births_per_year


baseline_births = get_total_births_per_year(baseline_results_folder)
intervention_births = get_total_births_per_year(intervention_results_folder)


# ===============================================CHECK INTERVENTION ===================================================
def get_anc_4_coverage(folder):
    results = extract_results(
        folder,
        module="tlo.methods.care_of_women_during_pregnancy",
        key="anc_count_on_birth",
        custom_generate_series=(
            lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'total_anc'])['person_id'].count()),
        do_scaling=False
    )
    anc_count_df = pd.DataFrame(columns=intervention_years, index=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    # get yearly outputs
    for year in intervention_years:
        for row in anc_count_df.index:
            if row in results.loc[year].index:
                x = results.loc[year, row]
                mean = x.mean()
                lq = x.quantile(0.025)
                uq = x.quantile(0.925)
                anc_count_df.at[row, year] = [mean, lq, uq]
            else:
                anc_count_df.at[row, year] = [0, 0, 0]

    yearly_anc4_rates = list()
    anc4_lqs = list()
    anc4_uqs = list()

    for year in intervention_years:
        anc_total = 0
        four_or_more_visits = 0

        for row in anc_count_df[year]:
            anc_total += row[0]

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

    return[yearly_anc4_rates, anc4_lqs, anc4_uqs]


baseline_anc4_coverage = get_anc_4_coverage(baseline_results_folder)
intervention_anc4_coverage = get_anc_4_coverage(intervention_results_folder)

fig, ax = plt.subplots()
ax.plot(intervention_years, baseline_anc4_coverage[0], label="Baseline (mean)", color='deepskyblue')
ax.fill_between(intervention_years, baseline_anc4_coverage[1], baseline_anc4_coverage[2], color='b', alpha=.1,
                label="UI (2.5-92.5)")

ax.plot(intervention_years, intervention_anc4_coverage[0], label="Intervention (mean)", color='olivedrab')
ax.fill_between(intervention_years, intervention_anc4_coverage[1], intervention_anc4_coverage[2], color='g', alpha=.1,
                label="UI (2.5-92.5)")

plt.xlabel('Year')
plt.ylabel('Coverage of ANC4')
plt.title('ANC4 coverage in baseline and intervention scenarios from 2020')
plt.legend()
# plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/anc4_intervention_coverage.png')
plt.show()


#  --------------------------------------------- MATERNAL DEATH ------------------------------------------------------
def get_yearly_mmr_and_nnmr(folder, births):
    death_results_labels = extract_results(
        folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'label'])['year'].count()
        ),
    )
    mm = analysis_utility_functions.get_comp_mean_and_rate('Maternal Disorders', births, death_results_labels, 100000,
                                                           intervention_years)
    nm = analysis_utility_functions.get_comp_mean_and_rate('Neonatal Disorders', births, death_results_labels, 1000,
                                                           intervention_years)
    return [mm, nm]


baseline_mmr = get_yearly_mmr_and_nnmr(baseline_results_folder, baseline_births)[0]
intervention_mmr = get_yearly_mmr_and_nnmr(intervention_results_folder, intervention_births)[0]
baseline_nmr = get_yearly_mmr_and_nnmr(baseline_results_folder, baseline_births)[1]
intervention_nmr = get_yearly_mmr_and_nnmr(intervention_results_folder, intervention_births)[1]


def get_mmr_nmr_graphs(bdata, idata, group):
    fig, ax = plt.subplots()
    ax.plot(intervention_years, bdata[0], label="Baseline (mean)", color='deepskyblue')
    ax.fill_between(intervention_years, bdata[1], bdata[2], color='b', alpha=.1)
    ax.plot(intervention_years, idata[0], label="Intervention (mean)", color='olivedrab')
    ax.fill_between(intervention_years, idata[1], idata[2], color='g', alpha=.1)
    # ax.set(ylim=(0, 800))
    plt.xlabel('Year')
    if group == 'Neonatal':
        plt.ylabel("Deaths per 1000 live births")
    else:
        plt.ylabel("Deaths per 100,000 live births")
    plt.title(f'{group} Mortality Ratio per Year at Baseline and Under Intervention')
    plt.legend()
    # plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/{group}_mr_int.png')
    plt.show()


get_mmr_nmr_graphs(baseline_mmr, intervention_mmr, 'Maternal')
get_mmr_nmr_graphs(baseline_nmr, intervention_nmr, 'Neonatal')


# STILLBIRTH
def get_yearly_sbr_data(folder, births):
    an_stillbirth_results = extract_results(
        folder,
        module="tlo.methods.pregnancy_supervisor",
        key="antenatal_stillbirth",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()
        ),
    )
    ip_stillbirth_results = extract_results(
        folder,
        module="tlo.methods.labour",
        key="intrapartum_stillbirth",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()
        ),
    )

    an_still_birth_data = analysis_utility_functions.get_mean_and_quants(an_stillbirth_results, intervention_years)
    ip_still_birth_data = analysis_utility_functions.get_mean_and_quants(ip_stillbirth_results, intervention_years)

    total_sbr = [((x + y) / z) * 1000 for x, y, z in zip(an_still_birth_data[0], ip_still_birth_data[0], births)]
    total_lqs = [((x + y) / z) * 1000 for x, y, z in zip(an_still_birth_data[1], ip_still_birth_data[1], births)]
    total_uqs = [((x + y) / z) * 1000 for x, y, z in zip(an_still_birth_data[2], ip_still_birth_data[2], births)]

    return [total_sbr, total_lqs, total_uqs]


baseline_sbr = get_yearly_sbr_data(baseline_results_folder, baseline_births)
intervention_sbr = get_yearly_sbr_data(intervention_results_folder, intervention_births)

fig, ax = plt.subplots()
ax.plot(intervention_years, baseline_sbr[0], label="Baseline (mean)", color='deepskyblue')
ax.fill_between(intervention_years, baseline_sbr[1], baseline_sbr[2], color='b', alpha=.1)
ax.plot(intervention_years, intervention_sbr[0], label="Intervention (mean)", color='olivedrab')
ax.fill_between(intervention_years, intervention_sbr[1], intervention_sbr[2], color='g', alpha=.1)
plt.xlabel('Year')
plt.ylabel("Stillbirths per 1000 live births")
plt.title('Stillbirth Rate per Year at Baseline and Under Intervention')
plt.legend()
# plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/sbr_int.png')
plt.show()

# HEALTH CARE WORKER TIME
# todo: not sure if we can extract this yet from master....
capacity = extract_results(
        baseline_results_folder,
        module="tlo.methods.healthsystem",
        key="Capacity",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['Frac_Time_Used_Overall'].mean()
        ),
    )

# cap = output['tlo.methods.healthsystem']['Capacity'].copy()
# cap["date"] = pd.to_datetime(cap["date"])
# cap = cap.set_index('date')
#
# frac_time_used = cap['Frac_Time_Used_Overall']
# cap = cap.drop(columns=['Frac_Time_Used_Overall'])
#
# # Plot Fraction of total time of health-care-workers being used
# frac_time_used.plot()
# plt.title("Fraction of total health-care worker time being used")
# plt.xlabel("Date")
# #plt.savefig(make_file_name('HSI_Frac_time_used'))
# plt.show()
#
# # %% Breakdowns by HSI:
# hsi = output['tlo.methods.healthsystem']['HSI_Event'].copy()
# hsi["date"] = pd.to_datetime(hsi["date"])
# hsi["month"] = hsi["date"].dt.month
# # Reduce TREATMENT_ID to the originating module
# hsi["Module"] = hsi["TREATMENT_ID"].str.split('_').apply(lambda x: x[0])
#
# # Plot the HSI that are taking place, by month, in a a particular year
# year = 2012
# evs = hsi.loc[hsi.date.dt.year == year]\
#     .groupby(by=['month', 'Module'])\
#     .size().reset_index().rename(columns={0: 'count'})\
#     .pivot_table(index='month', columns='Module', values='count', fill_value=0)
# evs *= scaling_factor
#
# evs.plot.bar(stacked=True)
# plt.title(f"HSI by Module, per Month (year {year})")
# plt.ylabel('Total per month')
# #plt.savefig(make_file_name('HSI_per_module_per_month'))
# plt.show()
#
# # Plot the breakdown of all HSI, over all the years
# evs = hsi.groupby(by=['Module'])\
#     .size().rename(columns={0: 'count'}) * scaling_factor
# evs.plot.pie()
# plt.title(f"HSI by Module")
# #plt.savefig(make_file_name('HSI_per_module'))
# plt.show()


# =================================================== CONSUMABLE COST =================================================
def get_mean_and_quants(df, sim_years):
    year_means = list()
    lower_quantiles = list()
    upper_quantiles = list()

    for year in sim_years:
        if year in df.index:
            year_means.append(df.loc[year].mean(axis=1).iloc[0])
            lower_quantiles.append(df.loc[year].iloc[0].quantile(0.025))
            upper_quantiles.append(df.loc[year].iloc[0].quantile(0.925))
        else:
            year_means.append(0)
            lower_quantiles.append(0)
            lower_quantiles.append(0)

    return [year_means, lower_quantiles, upper_quantiles]


draws = [0, 1, 2, 3, 4]
resourcefilepath = Path("./resources/healthsystem/consumables/")
consumables_df = pd.read_csv(Path(resourcefilepath) / 'ResourceFile_Consumables.csv')

# TODO: this should be scaled to the correct population size?
# todo: also so slow...


def get_cons_cost_per_year(results_folder):
    # Create df that replicates the 'extracted' df
    total_cost_per_draw_per_year = pd.DataFrame(columns=[draws], index=[intervention_years])

    # Loop over each draw
    for draw in draws:
        # Load df, add year column and select only ANC interventions
        draw_df = load_pickled_dataframes(results_folder, draw=draw)

        cons = draw_df['tlo.methods.healthsystem']['Consumables']
        cons['year'] = cons['date'].dt.year
        total_anc_cons = cons.loc[cons.TREATMENT_ID.str.contains('AntenatalCare')]
        anc_cons = total_anc_cons.loc[total_anc_cons.year >= intervention_years[0]]

        cons_df_for_this_draw = pd.DataFrame(index=[intervention_years])

        # Loop over each year
        for year in intervention_years:
            # Select the year of interest
            year_df = anc_cons.loc[anc_cons.year == year]

            # For each row (hsi) in that year we unpack the dictionary
            for row in year_df.index:
                cons_dict = year_df.at[row, 'Item_Available']
                cons_dict = eval(cons_dict)

                # todo: check this works where there are muliple dicts
                # For each dictionary
                for k, v in cons_dict.items():
                    if k in cons_df_for_this_draw.columns:
                        cons_df_for_this_draw.at[year, k] += v
                    elif k not in cons_df_for_this_draw.columns:
                        cons_df_for_this_draw[k] = v

        for row in cons_df_for_this_draw.index:
            for column in cons_df_for_this_draw.columns:
                cons_df_for_this_draw.at[row, column] =\
                    (cons_df_for_this_draw.at[row, column] *
                     (consumables_df[consumables_df.Item_Code == 0]['Unit_Cost'].iloc[0]))
                cons_df_for_this_draw.at[row, column] = cons_df_for_this_draw.at[row, column] * 0.0014
                # todo: this is usd conversion
                # todo: account for inflation, and use 2010 rate

        for index in total_cost_per_draw_per_year.index:
            total_cost_per_draw_per_year.at[index, draw] = cons_df_for_this_draw.loc[index].sum()

    final_cost_data = get_mean_and_quants(total_cost_per_draw_per_year, intervention_years)
    return final_cost_data


baseline_cost_data = get_cons_cost_per_year(baseline_results_folder)
intervention_cost_data = get_cons_cost_per_year(intervention_results_folder)

fig, ax = plt.subplots()
ax.plot(intervention_years, baseline_cost_data[0], label="Baseline (mean)", color='deepskyblue')
ax.fill_between(intervention_years, baseline_cost_data[1], baseline_cost_data[2], color='b', alpha=.1)
ax.plot(intervention_years, intervention_cost_data[0], label="Intervention (mean)", color='olivedrab')
ax.fill_between(intervention_years, intervention_cost_data[1], intervention_cost_data[2], color='g', alpha=.1)
plt.xlabel('Year')
plt.ylabel("Total Cost (USD)")
plt.title('Total Cost Attributable To Antenatal Care Per Year (in USD) (unscaled)')
plt.legend()
# plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/COST.png')
plt.show()

# (DALYS)


def get_yearly_dalys(folder):
    dalys = extract_results(
                folder,
                module="tlo.methods.healthburden",
                key="dalys",
                custom_generate_series=(
                    lambda df_: df_.drop(
                        columns='date'
                    ).rename(
                        columns={'age_range': 'age_grp'}
                    ).groupby(['year']).sum().stack()
                ),
                do_scaling=True
            )
    yearly_mat_dalys = list()
    yearly_neo_dalys = list()

    for year in intervention_years:
        if year in dalys.index:
            yearly_mat_dalys.append(dalys.loc[year, 'Maternal Disorders'].mean())
            yearly_neo_dalys.append(dalys.loc[year, 'Neonatal Disorders'].mean())

    return [yearly_mat_dalys, yearly_neo_dalys]


baseline_dalys = get_yearly_dalys(baseline_results_folder)
intervention_dalys = get_yearly_dalys(baseline_results_folder)


def daly_graphs(condition, b_data, i_data):
    fig, ax = plt.subplots()
    ax.plot(intervention_years, b_data, label="Baseline (mean)", color='deepskyblue')
    ax.plot(intervention_years, i_data, label="Intervention (mean)", color='olivedrab')
    plt.xlabel('Year')
    plt.ylabel("Disability Adjusted Life Years")
    plt.title(f'Total DALYs per Year Attributable to {condition} disorders')
    plt.legend()
    # plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/{condition}_dalys.png')
    plt.show()


daly_graphs('Maternal', baseline_dalys[0], intervention_dalys[0])
daly_graphs('Neonatal', baseline_dalys[1], intervention_dalys[1])

# ======================================================= ANC 4 ======================================================

# SCENARIO ONE: BASELINE ANC 4 COVERAGE VERSES 90% (UHC) EPMM COVERAGE (CONSUMABLES CONSTRAINED, SQUEEZE CONSTRAINED)

# MATERNAL DEATH
# STILLBIRTH
# NEONATAL DEATH
# HEALTH CARE WORKER TIME
# CONSUMABLES
# (DALYS)

# SCENARIO TWO: BASELINE ANC 4 COVERAGE VERSES 90% (UHC) EPMM COVERAGE (NO CONSUMABLES, QUALITY OR SQUEEZE CONTRAINTS
# IN THE COMPARATOR)

# MATERNAL DEATH
# STILLBIRTH
# NEONATAL DEATH
# HEALTH CARE WORKER TIME
# CONSUMABLES
# (DALYS)

# ======================================================= ANC 8 ======================================================

# SCENARIO ONE: BASELINE ANC 8 COVERAGE VERSES 50% COVERAGE (CONSUMABLES CONSTRAINED, SQUEEZE CONSTRAINED)

# MATERNAL DEATH
# STILLBIRTH
# NEONATAL DEATH
# HEALTH CARE WORKER TIME
# CONSUMABLES
# (DALYS)

# SCENARIO TWO: BASELINE ANC 8 COVERAGE VERSES 50% COVERAGE (NO CONSUMABLES, QUALITY OR SQUEEZE CONTRAINTS
# IN THE COMPARATOR)

# MATERNAL DEATH
# STILLBIRTH
# NEONATAL DEATH
# HEALTH CARE WORKER TIME
# CONSUMABLES
# (DALYS)

# SCENARIO THREE: BASELINE ANC 8 COVERAGE VERSES 90% (UHC) EPMM COVERAGE( CONSUMABLES CONSTRAINED, SQUEEZE CONSTRAINED)

# MATERNAL DEATH
# STILLBIRTH
# NEONATAL DEATH
# HEALTH CARE WORKER TIME
# CONSUMABLES
# (DALYS)

# SCENARIO FOUR: BASELINE ANC 8 COVERAGE VERSES 90% (UHC) EPMM COVERAGE (NO CONSUMABLES, QUALITY OR SQUEEZE CONTRAINTS
# IN THE COMPARATOR)

# MATERNAL DEATH
# STILLBIRTH
# NEONATAL DEATH
# HEALTH CARE WORKER TIME
# CONSUMABLES
# (DALYS)
