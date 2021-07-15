from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo.analysis.utils import (
    extract_results, extract_str_results,
    get_scenario_outputs,
)

# %% Declare the name of the file that specified the scenarios used in this run.
scenario_filename = 'multi_run_calibration.py'  # <-- update this to look at other results

# %% Declare usual paths:
outputspath = Path('./outputs/sejjj49@ucl.ac.uk/')
rfp = Path('./resources')

# Find results folder (most recent run generated using that scenario_filename)
results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]
# create_pickles_locally(results_folder)  # if not created via batch

# HELPER FUNCTIONS

def get_df_without_null_values(extracted_df):
    new_df = pd.DataFrame(index=[0])
    for column in extracted_df:
        null_values = extracted_df[column].isnull()
        new_df[f"draw {column}"] = len(null_values.loc[~null_values].index)

    return new_df


def return_graph_with_credible_intervals(target_rates, model_rates, ci, colours, labels, title, ylabel):
    barWidth = 0.35
    y_r = [model_rates[i] - ci[i][1] for i in range(len(ci))]
    r1 = np.arange(len(model_rates))
    r2 = [x + barWidth for x in r1]
    plt.bar(r1, model_rates, width=barWidth, color=colours[0], yerr=y_r, capsize=7, label='model')
    plt.bar(r2, target_rates, width=barWidth, color=colours[1], capsize=7, label='target')
    plt.title(title)
    plt.xticks([r + barWidth for r in range(len(model_rates))], labels)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


# ============================================== ANTENATAL CARE =======================================================
extracted_anc = extract_results(results_folder,
                                module="tlo.methods.care_of_women_during_pregnancy",
                                key="anc_ga_first_visit",  # <-- the key used for the logging entry
                                column="ga_anc_one")

extracted_anc.columns = extracted_anc.columns.get_level_values(0)
anc_ga_birth = get_df_without_null_values(extracted_anc)

mean = anc_ga_birth.mean(axis=1)
lower_q = anc_ga_birth.quantile(0.025, axis=1)
upper_q = anc_ga_birth.quantile(0.975, axis=1)

anc_ga_groupings = pd.DataFrame(columns=['less_4', '4_5', '6_7', '8+'], index=[extracted_anc.columns])

for column in extracted_anc:
    anc_ga_groupings.at[column, 'less_4'] = len(extracted_anc.loc[extracted_anc[column] <= 13])
    anc_ga_groupings.at[column, '4_5'] = len(extracted_anc.loc[(extracted_anc[column] > 13) &
                                                               (extracted_anc[column] <= 22)])
    anc_ga_groupings.at[column, '6_7'] = len(extracted_anc.loc[(extracted_anc[column] > 22) &
                                                               (extracted_anc[column] <= 32)])
    anc_ga_groupings.at[column, '8+'] = len(extracted_anc.loc[extracted_anc[column] > 31])

less_4 = (anc_ga_groupings['less_4'].mean(axis=0)/mean) * 100
less_4_uq = (anc_ga_groupings['less_4'].quantile(0.925)/mean) * 100
less_4_lq = (anc_ga_groupings['less_4'].quantile(0.025) / mean) * 100
ga_4_5 = (anc_ga_groupings['4_5'].mean(axis=0)/mean) * 100
ga_4_5_uq = (anc_ga_groupings['4_5'].quantile(0.925) / mean) * 100
ga_4_5_lq = (anc_ga_groupings['4_5'].quantile(0.025) / mean) * 100
ga_6_7 = (anc_ga_groupings['6_7'].mean(axis=0) / mean) * 100
ga_6_7_uq = (anc_ga_groupings['6_7'].quantile(0.925) / mean) * 100
ga_6_7_lq = (anc_ga_groupings['6_7'].quantile(0.025) / mean) * 100
ga_8 = (anc_ga_groupings['8+'].mean(axis=0) / mean) * 100
ga_8_uq = (anc_ga_groupings['8+'].quantile(0.925) / mean) * 100
ga_8_lq = (anc_ga_groupings['8+'].quantile(0.025) / mean) * 100

extracted_total = extract_results(results_folder,
                                  module="tlo.methods.care_of_women_during_pregnancy",
                                  key="anc_count_on_birth",  # <-- the key used for the logging entry
                                  column="total_anc")

extracted_total.columns = extracted_total.columns.get_level_values(0)


anc_tot = get_df_without_null_values(extracted_total)
anc_visits = pd.DataFrame(columns=["anc1", "anc4", "anc8"], index=[extracted_total.columns])

mean_total_women = anc_tot.mean(axis=1)
lower_tot = anc_tot.quantile(0.025, axis=1)
upper_tot = anc_tot.quantile(0.975, axis=1)

for column in extracted_total:
    anc_visits.at[column, 'anc1'] = len(extracted_total.loc[extracted_total[column] > 0])
    anc_visits.at[column, 'anc4'] = len(extracted_total.loc[extracted_total[column] > 3])
    anc_visits.at[column, 'anc8'] = len(extracted_total.loc[extracted_total[column] > 7])

mean_anc_visits = anc_visits.mean(axis=0)

anc1_rate = (mean_anc_visits.loc['anc1']/mean_total_women) * 100
anc1_uq = (anc_visits['anc1'].quantile(0.925)/mean_total_women) * 100
anc1_lq = (anc_visits['anc1'].quantile(0.025) / mean_total_women) * 100
anc4_rate = (mean_anc_visits.loc['anc4']/mean_total_women) * 100
anc4_uq = (anc_visits['anc4'].quantile(0.925)/mean_total_women) * 100
anc4_lq = (anc_visits['anc4'].quantile(0.025) / mean_total_women) * 100
anc8_rate = (mean_anc_visits.loc['anc8']/mean_total_women) * 100
anc8_uq = (anc_visits['anc8'].quantile(0.925)/mean_total_women) * 100
anc8_lq = (anc_visits['anc8'].quantile(0.025) / mean_total_women) * 100

barWidth = 0.35
model_rates = [int(anc1_rate.values), int(anc4_rate.values), int(anc8_rate.values), int(less_4.values),
               int(ga_4_5.values), int(ga_6_7.values), int(ga_8.values)]
target_rates = [94.7, 46, 2, 12.4, 48.2, 35.6, 2]

ci = [(int(anc1_lq.values), int(anc1_uq.values)), (int(anc4_lq.values), int(anc4_uq.values)),
      (int(anc8_lq.values), int(anc8_uq.values)), (int(less_4_lq.values), int(less_4_uq.values)),
       (int(ga_4_5_lq.values), int(ga_4_5_uq.values)), (int(ga_6_7_lq.values), int(ga_6_7_uq.values)),
       (int(ga_8_lq.values), int(ga_8_uq.values))]

return_graph_with_credible_intervals(target_rates, model_rates, ci, ['mediumturquoise', 'cyan'],
                                     ['ANC1', 'ANC4+', 'ANC8+', '<4', 'M4/5', 'M6/7', 'M8+'],
                                     'Coverage indicators of ANC in 2010',
                                     '% of women delivered')

# todo: median GA at first visit

# ============================================== FACILITY DELIVERY ===================================================

extracted_births = extract_results(results_folder,
                                module="tlo.methods.demography",
                                key="on_birth",  # <-- the key used for the logging entry
                                column="child")

extracted_births.columns = extracted_births.columns.get_level_values(0)
births = pd.DataFrame(index=[0])

for column in extracted_births:
    null_values = extracted_births[column].isnull()
    births[f"draw {column}"] = len(null_values.loc[~null_values].index)

mean_births = births.mean(axis=1)
lower_q_births = births.quantile(0.025, axis=1)
upper_q_births = births.quantile(0.975, axis=1)

extracted_fds = extract_str_results(results_folder,
                                    module="tlo.methods.labour",
                                    key="delivery_setting",  # <-- the key used for the logging entry
                                    column="facility_type")
extracted_fds.columns = extracted_fds.columns.get_level_values(0)

fds = pd.DataFrame(index=[0])

for column in extracted_fds:
    fds.at[column, 'health_centre'] = len(extracted_fds.loc[extracted_fds[column] == 'health_centre'])
    fds.at[column, 'hospital'] = len(extracted_fds.loc[extracted_fds[column] == 'hospital'])
    fds.at[column, 'home_birth'] = len(extracted_fds.loc[extracted_fds[column] == 'home_birth'])
    fds.at[column, 'total_fd'] = ((len(extracted_fds.loc[extracted_fds[column] == 'health_centre'])) +
                                  (len(extracted_fds.loc[extracted_fds[column] == 'hospital'])))


mean_fds = fds.mean(axis=0)
hc_rate = (mean_fds.loc['health_centre']/mean_births) * 100
hc_uq = (fds['health_centre'].quantile(0.925)/mean_births) * 100
hc_lq = (fds['health_centre'].quantile(0.025) / mean_births) * 100
hp_rate = (mean_fds.loc['hospital']/mean_births) * 100
hp_uq = (fds['hospital'].quantile(0.925)/mean_births) * 100
hp_lq = (fds['hospital'].quantile(0.025) / mean_births) * 100
hb_rate = (mean_fds.loc['home_birth']/mean_births) * 100
hb_uq = (fds['home_birth'].quantile(0.925)/mean_births) * 100
hb_lq = (fds['home_birth'].quantile(0.025) / mean_births) * 100
fd_rate = (mean_fds.loc['total_fd']/mean_births) * 100
fd_uq = (fds['total_fd'].quantile(0.925)/mean_births) * 100
fd_lq = (fds['total_fd'].quantile(0.025) / mean_births) * 100


model_rates_fd = [int(fd_rate.values), int(hc_rate.values), int(hp_rate.values), int(hb_rate.values)]
target_rates_fd = [73, 32, 41, 27]

ci_fd = [(int(fd_lq.values), int(fd_uq.values)), (int(hc_lq.values), int(hc_uq.values)), (int(hp_lq.values),
                                                                                          int(hp_uq.values)),
         (int(hb_lq.values), int(hb_uq.values))]

return_graph_with_credible_intervals(target_rates_fd, model_rates_fd, ci_fd, ['palevioletred', 'lavenderblush'],
                                     ['FD', 'HC', 'HP', 'HB'],
                                     'Coverage indicators of Facility Delivery in 2010',
                                     '% of total births')

# ============================================== POSTNATAL CARE ===================================================
extracted_pnc_lab = extract_results(results_folder,
                                    module="tlo.methods.labour",
                                    key="postnatal_check",  # <-- the key used for the logging entry
                                    column="visit_number")

extracted_pnc_lab.columns = extracted_pnc_lab.columns.get_level_values(0)
mat_pnc = pd.DataFrame(index=[0])

for column in extracted_pnc_lab:
    not_first_visit = extracted_pnc_lab.loc[extracted_pnc_lab[column] > 0]
    null_values = extracted_pnc_lab[column].isnull()
    mat_pnc[f"draw {column}"] = len(null_values.loc[~null_values].index) - len(not_first_visit)

mean_mpnc = mat_pnc.mean(axis=1)
mpnc_rate = (mean_mpnc/mean_births) * 100
mpnc_uq = (mat_pnc.quantile(0.925, axis=1)/mean_births) * 100
mpnc_lq = (mat_pnc.quantile(0.025, axis=1) / mean_births) * 100


extracted_pnc_nb = extract_results(results_folder,
                                   module="tlo.methods.newborn_outcomes",
                                   key="postnatal_check",  # <-- the key used for the logging entry
                                   column="visit_number")

extracted_pnc_nb.columns = extracted_pnc_nb.columns.get_level_values(0)
nb_pnc = pd.DataFrame(index=[0])

for column in extracted_pnc_nb:
    not_first_visit = extracted_pnc_nb.loc[extracted_pnc_nb[column] > 0]
    null_values = extracted_pnc_nb[column].isnull()
    nb_pnc[f"draw {column}"] = len(null_values.loc[~null_values].index) - len(not_first_visit)

mean_npnc = nb_pnc.mean(axis=1)
npnc_rate = (mean_npnc/mean_births) * 100
npnc_uq = (nb_pnc.quantile(0.925, axis=1)/mean_births) * 100
npnc_lq = (nb_pnc.quantile(0.025, axis=1) / mean_births) * 100

# todo: early vs late

model_rates_pn = [int(mpnc_rate.values), int(npnc_rate.values)]
target_rates_pn = [50, 60]
ci_pn = [(int(mpnc_lq.values), int(mpnc_uq.values)), (int(npnc_lq.values), int(npnc_uq.values))]

return_graph_with_credible_intervals(target_rates_pn, model_rates_pn, ci_pn, ['darkseagreen', 'honeydew'],
                                     ['Mat.', 'Neo.'],
                                     'Coverage indicators of Postnatal Care in 2010',
                                     '% of total births')
