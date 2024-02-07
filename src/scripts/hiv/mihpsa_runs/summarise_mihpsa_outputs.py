"""This file uses the results of the batch file to make some summary statistics.
"""

import datetime
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from tlo.analysis.utils import (
    compare_number_of_deaths,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)

resourcefilepath = Path("./resources")
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

outputspath = Path("./outputs/t.mangal@imperial.ac.uk")
# %% Analyse results of runs

# 0) Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("mihpsa_runs.py", outputspath)[-1]
scaling_factor = load_pickled_dataframes(results_folder, 0, 0, 'tlo.methods.population'
                                         )['tlo.methods.population']['scaling_factor']['scaling_factor'].values[0]


# Collect results from each draw/run
def extract_outputs(results_folder: Path,
                    module: str,
                    key: str,
                    column: str) -> pd.DataFrame:
    # get number of draws and numbers of runs
    info = get_scenario_info(results_folder)

    df_pop = pd.DataFrame()
    df_prev = pd.DataFrame()
    df_inf = pd.DataFrame()
    df_dx = pd.DataFrame()
    df_tx = pd.DataFrame()
    df_vs = pd.DataFrame()
    df_out = pd.DataFrame()

    # 10 draws, 1 run
    run = 0
    for draw in range(info['number_of_draws']):
        # load the log file
        df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run, module)[module][key]
        # first column is mean

        tmp = pd.DataFrame(df[column].to_list(), columns=['pop', 'prev',
                                                          "inf", "dx",
                                                          "tx", "vs"])
        df_pop[draw] = tmp["pop"]
        df_prev[draw] = tmp["prev"]
        df_inf[draw] = tmp["inf"]
        df_dx[draw] = tmp["dx"]
        df_tx[draw] = tmp["tx"]
        df_vs[draw] = tmp["vs"]

    df_out["mean_pop"] = df_pop.mean(axis=1) * scaling_factor
    df_out["mean_prev"] = df_prev.mean(axis=1)
    df_out["mean_inf"] = df_inf.mean(axis=1) * scaling_factor
    df_out["mean_dx"] = df_dx.mean(axis=1)
    df_out["mean_tx"] = df_tx.mean(axis=1)
    df_out["mean_vs"] = df_vs.mean(axis=1)

    return df_out


# extract pop size for M, F and Total
baseline_m = extract_outputs(results_folder=results_folder,
                             module="tlo.methods.hiv",
                             key="hiv_baseline_outputs",
                             column="outputs_age15_64_M")

baseline_f = extract_outputs(results_folder=results_folder,
                             module="tlo.methods.hiv",
                             key="hiv_baseline_outputs",
                             column="outputs_age15_64_F")

baseline_all = extract_outputs(results_folder=results_folder,
                               module="tlo.methods.hiv",
                               key="hiv_baseline_outputs",
                               column="outputs_age15_64")


# Collect results from each draw/run
def extract_outputs_detailed(results_folder: Path,
                             column_m: str,
                             column_f: str) -> pd.DataFrame:
    module = "tlo.methods.hiv"
    key = "hiv_detailed_outputs"
    columns = [column_m, column_f]

    # get number of draws and numbers of runs
    info = get_scenario_info(results_folder)
    df_out = pd.DataFrame()

    for sex in ["M", "F"]:

        column = columns[0] if (sex == "M") else columns[1]

        df_inf = pd.DataFrame()
        df_no_dx = pd.DataFrame()
        df_dx_no_tx = pd.DataFrame()
        df_art_not_vs = pd.DataFrame()
        df_art_vs = pd.DataFrame()
        df_art_under_6mths_not_vs = pd.DataFrame()
        df_art_under_6mths_vs = pd.DataFrame()
        df_art_over_6mths_not_vs = pd.DataFrame()
        df_art_over_6mths_vs = pd.DataFrame()
        df_art_interruption = pd.DataFrame()

        # 10 draws, 1 run
        run = 0
        for draw in range(info['number_of_draws']):
            # load the log file
            df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run, module)[module][key]

            tmp = pd.DataFrame(df[column].to_list(),
                               columns=[
                                   "num_infected", "num_not_diagnosed",
                                   "num_dx_no_art",
                                   "num_art_not_vs", "num_art_vs",
                                   "num_art_under_6mths_not_vs", "num_art_under_6mths_vs",
                                   "num_art_over_6mths_not_vs", "num_art_over_6mths_vs",
                                   "num_any_interruption"])

            df_inf[draw] = tmp["num_infected"]
            df_no_dx[draw] = tmp["num_not_diagnosed"]
            df_dx_no_tx[draw] = tmp["num_dx_no_art"]
            df_art_not_vs[draw] = tmp["num_art_not_vs"]
            df_art_vs[draw] = tmp["num_art_vs"]
            df_art_under_6mths_not_vs[draw] = tmp["num_art_under_6mths_not_vs"]
            df_art_under_6mths_vs[draw] = tmp["num_art_under_6mths_vs"]
            df_art_over_6mths_not_vs[draw] = tmp["num_art_over_6mths_not_vs"]
            df_art_over_6mths_vs[draw] = tmp["num_art_over_6mths_vs"]
            df_art_interruption[draw] = tmp["num_any_interruption"]

        df_out["num_infected"] = df_inf.mean(axis=1) * scaling_factor
        df_out["num_not_diagnosed"] = df_no_dx.mean(axis=1) * scaling_factor
        df_out["num_dx_no_art"] = df_dx_no_tx.mean(axis=1) * scaling_factor
        df_out["num_art_not_vs"] = df_art_not_vs.mean(axis=1) * scaling_factor
        df_out["num_art_vs"] = df_art_vs.mean(axis=1) * scaling_factor
        df_out["num_art_under_6mths_not_vs"] = df_art_under_6mths_not_vs.mean(axis=1) * scaling_factor
        df_out["num_art_under_6mths_vs"] = df_art_under_6mths_vs.mean(axis=1) * scaling_factor
        df_out["num_art_over_6mths_not_vs"] = df_art_over_6mths_not_vs.mean(axis=1) * scaling_factor
        df_out["num_art_over_6mths_vs"] = df_art_over_6mths_vs.mean(axis=1) * scaling_factor
        df_out["num_any_interruption"] = df_art_interruption.mean(axis=1) * scaling_factor

        if sex == "M":
            df_out.columns += "_M"

    # reorder columns
    df_out = df_out[["num_infected_M", "num_infected",
                     "num_not_diagnosed_M", "num_not_diagnosed",
                     "num_dx_no_art_M", "num_dx_no_art",
                     "num_art_vs_M", "num_art_vs",
                     "num_art_not_vs_M", "num_art_not_vs",
                     "num_art_under_6mths_vs_M", "num_art_under_6mths_vs",
                     "num_art_under_6mths_not_vs_M", "num_art_under_6mths_not_vs",
                     "num_art_over_6mths_vs_M", "num_art_over_6mths_vs",
                     "num_art_over_6mths_not_vs_M", "num_art_over_6mths_not_vs",
                     "num_any_interruption_M", "num_any_interruption"]]

    return df_out


age15_19 = extract_outputs_detailed(results_folder=results_folder,
                                    column_m="outputs_age15_19_M",
                                    column_f="outputs_age15_19_F")

age20_24 = extract_outputs_detailed(results_folder=results_folder,
                                    column_m="outputs_age20_24_M",
                                    column_f="outputs_age20_24_F")

age25_29 = extract_outputs_detailed(results_folder=results_folder,
                                    column_m="outputs_age25_29_M",
                                    column_f="outputs_age25_29_F")

age30_34 = extract_outputs_detailed(results_folder=results_folder,
                                    column_m="outputs_age30_34_M",
                                    column_f="outputs_age30_34_F")

age35_39 = extract_outputs_detailed(results_folder=results_folder,
                                    column_m="outputs_age35_39_M",
                                    column_f="outputs_age35_39_F")

age40_44 = extract_outputs_detailed(results_folder=results_folder,
                                    column_m="outputs_age40_44_M",
                                    column_f="outputs_age40_44_F")

age45_49 = extract_outputs_detailed(results_folder=results_folder,
                                    column_m="outputs_age45_49_M",
                                    column_f="outputs_age45_49_F")

age50_54 = extract_outputs_detailed(results_folder=results_folder,
                                    column_m="outputs_age50_54_M",
                                    column_f="outputs_age50_54_F")

age55_59 = extract_outputs_detailed(results_folder=results_folder,
                                    column_m="outputs_age55_59_M",
                                    column_f="outputs_age55_59_F")

age60_64 = extract_outputs_detailed(results_folder=results_folder,
                                    column_m="outputs_age60_64_M",
                                    column_f="outputs_age60_64_F")

age65_69 = extract_outputs_detailed(results_folder=results_folder,
                                    column_m="outputs_age65_69_M",
                                    column_f="outputs_age65_69_F")

age70_74 = extract_outputs_detailed(results_folder=results_folder,
                                    column_m="outputs_age70_74_M",
                                    column_f="outputs_age70_74_F")

age75_79 = extract_outputs_detailed(results_folder=results_folder,
                                    column_m="outputs_age75_79_M",
                                    column_f="outputs_age75_79_F")

age80_84 = extract_outputs_detailed(results_folder=results_folder,
                                    column_m="outputs_age80_84_M",
                                    column_f="outputs_age80_84_F")


# baseline_outputs = log["tlo.methods.hiv"]["hiv_baseline_outputs"]
# detailed_outputs = log["tlo.methods.hiv"]["hiv_detailed_outputs"]
# deaths = log["tlo.methods.hiv"]["death"]
#

# extract deaths by age-group
def extract_deaths(results_folder: Path) -> pd.DataFrame:
    module = "tlo.methods.demography"
    key = "death"

    # get number of draws and numbers of runs
    info = get_scenario_info(results_folder)
    df_out = pd.DataFrame()

    # 10 draws, 1 run
    run = 0
    for draw in range(info['number_of_draws']):
        # load the log file
        df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run, module)[module][key]

        # categorise age into groups
        df["age_group"] = (df["age"] / 5).astype(int) * 5

        # sum AIDS deaths by year and age-group
        keep = (df.label == "AIDS")
        aids_deaths = df.loc[keep].copy()
        aids_deaths["year"] = aids_deaths["date"].dt.year  # count by year
        # remove children
        aids_deaths = aids_deaths.loc[aids_deaths.age_group >= 15]
        aids_deaths_summary = aids_deaths.groupby(by=["year", "sex", "age_group"]).size().to_frame(
            name='count').reset_index()

        # for first draw, create the dataframe
        # subsequent draws can be merged in
        if draw == 0:
            df_out = aids_deaths_summary
        else:
            df_out = pd.merge(df_out, aids_deaths_summary,
                              how="left",
                              on=["year", "sex", "age_group"])

    # take mean across all draws
    df_out['avg'] = df_out.iloc[:, -10:].mean(axis=1) * scaling_factor

    return df_out


deaths = extract_deaths(results_folder=results_folder)
deaths.to_csv(outputspath / ("MIHPSA_deaths_May2023" + ".csv"), index=None)


# write to excel
with pd.ExcelWriter(outputspath / ("MIHPSA_outputs2" + ".xlsx"), engine='openpyxl') as writer:
    baseline_outputs.to_excel(writer, sheet_name='Sheet1', index=False)
    detailed_outputs.to_excel(writer, sheet_name='Sheet2', index=False)
    deaths.to_excel(writer, sheet_name='Sheet3', index=False)
    writer.save()
