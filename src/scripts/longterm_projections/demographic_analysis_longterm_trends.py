import argparse
import datetime
from pathlib import Path

import imageio
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from tlo.analysis.life_expectancy import get_life_expectancy_estimates
from tlo.analysis.utils import (
    extract_results,
    format_gbd,
    make_age_grp_lookup,
    make_age_grp_types,
    make_calendar_period_lookup,
    make_calendar_period_type,
    summarize,
    unflatten_flattened_multi_index_in_logging,
)

PREFIX_ON_FILENAME = '1'
min_year = "2020"
max_year = "2040"
def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    # Declare path for output graphs from this script
    make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_{stub}.png"  # noqa: E731

    # Define colo(u)rs to use:
    colors = {
        'Model': 'royalblue',
        'Census': 'darkred',
        'WPP': 'forestgreen',
        'GBD': 'plum'
    }

    # Define how to call the sexes:
    sexname = lambda x: 'Females' if x == 'F' else 'Males'  # noqa: E731

    # Get helpers for age and calendar period aggregation
    agegrps, agegrplookup = make_age_grp_lookup()
    calperiods, calperiodlookup = make_calendar_period_lookup()
    adult_age_groups = ['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49']

    # %% Examine the results folder:

    # look at one log (so can decide what to extract)
    # from tlo.analysis.utils import load_pickled_dataframes
    # log = load_pickled_dataframes(results_folder)

    # get basic information about the results
    # from tlo.analysis.utils import get_scenario_info
    # info = get_scenario_info(results_folder)

    # 1) Extract the parameters that have varied over the set of simulations (will report that no parameters changed)
    # from tlo.analysis.utils import extract_params
    # params = extract_params(results_folder)

    # %% Population Size
    # Trend in Number Over Time

    # 1) Population Growth Over Time:

    # Load and format model results (with year as integer):
    pop_model = summarize(extract_results(results_folder,
                                          module="tlo.methods.demography",
                                          key="population",
                                          column="total",
                                          index="date",
                                          do_scaling=True
                                          ),
                          collapse_columns=True
                          )
    pop_model.index = pop_model.index.year
    # Load Data: WPP_Annual
    wpp_ann = pd.read_csv(Path(resourcefilepath) / "demography" / "ResourceFile_Pop_Annual_WPP.csv")
    wpp_ann['Age_Grp'] = wpp_ann['Age_Grp'].astype(make_age_grp_types())
    wpp_ann_total = wpp_ann.groupby(by=['Year'])['Count'].sum()

    # Load Data: Census
    cens = pd.read_csv(Path(resourcefilepath) / "demography" / "ResourceFile_PopulationSize_2018Census.csv")
    cens['Age_Grp'] = cens['Age_Grp'].astype(make_age_grp_types())
    cens_2018 = cens.groupby('Sex')['Count'].sum()

    # Plot population size over time
    fig, ax = plt.subplots()
    ax.plot(wpp_ann_total.index, wpp_ann_total / 1e6,
            label='WPP', color=colors['WPP'])
    ax.plot(2018.5, cens_2018.sum() / 1e6,
            marker='o', markersize=10, linestyle='none', label='Census', zorder=10, color=colors['Census'])
    ax.plot(pop_model.index, pop_model['mean'] / 1e6,
            label='Model (mean)', color=colors['Model'])
    ax.fill_between((pop_model.index).to_numpy(),
                    (pop_model['lower'] / 1e6).to_numpy(),
                    (pop_model['upper'] / 1e6).to_numpy(),
                    color=colors['Model'],
                    alpha=0.2,
                    zorder=5
                    )
    ax.set_title("Population Size 2010-2060")
    ax.set_xlabel("Year")
    ax.set_ylabel("Population Size (millions)")
    ax.set_xlim(2010, int(max_year))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.set_ylim(0, 40)
    ax.legend()
    fig.tight_layout()
    plt.savefig(make_graph_file_name("Pop_Over_Time"))
    plt.close(fig)

    # Make a gif
    for year in range(int(min_year), int(max_year),1):
        if year in pop_model.index:
            fig, ax = plt.subplots()
            # Get WPP data:
            wpp_ann_subset = wpp_ann_total.loc[wpp_ann_total.index <= year]
            pop_model_subset = pop_model.loc[pop_model.index <= year]
            ax.plot(pop_model_subset.index, pop_model_subset['mean'] / 1e6,
                    label=f'Model (mean)', color=colors['Model'])
            ax.plot(wpp_ann_subset.index, wpp_ann_subset / 1e6,
                    label=f'WPP', color=colors['WPP'])
            if year >= 2018:
                ax.plot(2018.5, cens_2018.sum() / 1e6,
                        marker='o', markersize=10, linestyle='none', label='Census', zorder=10, color=colors['Census'])
            ax.set_title(f"Population Size {min_year}-{max_year}")
            ax.set_xlabel("Year")
            ax.set_ylabel("Population Size (millions)")
            ax.set_xlim(2010, int(max_year))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            ax.set_ylim(0, 50)
            ax.legend()
            fig.tight_layout()
            plt.savefig(make_graph_file_name(f"Pop_Over_Time_line_{year}"))
            plt.close(fig)
    # first need to make plots
    frames = []
    for year in range(int(min_year), int(max_year)):
        image = imageio.v2.imread(make_graph_file_name(f"Pop_Over_Time_line_{year}"))
        frames.append(image)

    imageio.mimsave(output_folder / f"Pop_Line_{min_year}-{max_year}.gif",
                    frames,
                    fps=10)

    # 2) Population Size in 2018 (broken down by Male and Female)

    # Census vs WPP vs Model
    wpp_2018 = wpp_ann.groupby(['Year', 'Sex'])['Count'].sum()[2018]

    # Get Model totals for males and females in 2018 (with scaling factor)
    pop_model_male = summarize(extract_results(results_folder,
                                               module="tlo.methods.demography",
                                               key="population",
                                               column="male",
                                               index="date",
                                               do_scaling=True),
                               collapse_columns=True
                               )
    pop_model_male.index = pop_model_male.index.year

    pop_model_female = summarize(extract_results(results_folder,
                                                 module="tlo.methods.demography",
                                                 key="population",
                                                 column="female",
                                                 index="date",
                                                 do_scaling=True),
                                 collapse_columns=True
                                 )
    pop_model_female.index = pop_model_female.index.year

    pop_2018 = {
        'Census': cens_2018,
        'WPP': wpp_2018,
        'Model': pd.Series({
            'F': pop_model_female.loc[2018, 'mean'],
            'M': pop_model_male.loc[2018, 'mean']
        })
    }

    # Plot:
    labels = ['F', 'M']

    width = 0.2
    x = np.arange(len(labels))  # the label locations

    fig, ax = plt.subplots()
    for i, key in enumerate(pop_2018):
        ax.bar(x=x + (i - 1) * width * 1.05, height=[pop_2018[key][sex] / 1e6 for sex in labels],
               width=width,
               label=key,
               color=colors[key]
               )
    ax.set_xticks(x)
    ax.set_xticklabels([sexname(sex) for sex in labels])
    ax.set_ylabel('Sex')
    ax.set_ylabel('Population Size (millions)')
    ax.set_ylim(0, 10)
    ax.set_title('Population Size 2018')
    ax.legend()
    fig.tight_layout()
    plt.savefig(make_graph_file_name("Pop_Males_Females_2018"))
    plt.close(fig)

    # Population Pyramid at two time points
    def plot_population_pyramid(data, fig):
        """Plot a population pyramid on the specified figure. Data is of the form:
        {
           'F': {
                    'Model': pd.Series(index_age_groups),
                    'WPP': pd.Series(index=age_groups)
                 },
           'M': {
                    'Model': pd.Series(index_age_groups),
                    'WPP': pd.Series(index=age_groups)
                 },
        }
        """
        ax = fig.add_subplot(111)

        # reformat data to ensure axes align
        sources = data['M'].keys()
        correct_index_order = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34',
                               '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69',
                               '70-74', '75-79', '80-84', '85-89', '90-94', '95-99', '100+']
        for _sex in ['M', 'F']:
            for _source in sources:
                data[_sex][_source].index = pd.Categorical(data[_sex][_source].index,
                                                           categories=correct_index_order,
                                                           ordered=True)
                data[_sex][_source] = data[_sex][_source].sort_index()

        # Now concatenate the data
        dat = {
            _sex: pd.concat(
                {_source: data[_sex][_source] for _source in sources}, axis=1
            ) for _sex in ['M', 'F']
        }

        # Use horizontal bar chart functions to plot the pyramid
        ax.barh(dat['M'].index, dat['M']['Model'].values / 1e3, alpha=1.0, label='Model', color=colors['Model'])
        ax.barh(dat['F'].index, -dat['F']['Model'].values / 1e3, alpha=1.0, label='_', color='cornflowerblue')
        # use plot to overlay the comparison data sources (whatever is available from 'WPP' and/or 'Census')
        for _dat_source in sorted(set(sources).intersection(['WPP', 'Census'])):
            ax.plot(data['M'][_dat_source].values / 1e3, dat['M'].index, label=_dat_source, color=colors[_dat_source])
            ax.plot(-data['F'][_dat_source].values / 1e3, dat['F'].index, label='_', color=colors[_dat_source])

        ax.axvline(0.0, 0.0, color='black')

        # label the plot with titles and correct the x axis tick labels (replace negative values with positive)
        ax.legend()
        ax.set_ylabel('Age Groups')
        ax.set_xlabel('Population (1000s)')

        ax.text(x=1e3, y=10, s="Males", fontdict={'size': 15}, ha='right')
        ax.text(x=-1e3, y=10, s="Females", fontdict={'size': 15}, ha='left')

        # reverse order of legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='upper right')

        locs = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]) * 1e3
        ax.set_xticks(locs)
        ax.set_xticklabels(np.round(np.sqrt(locs ** 2)).astype(int))

        ax.set_axisbelow(True)
        # ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.grid()

        return ax

    # Get Age/Sex Breakdown of population (with scaling)
    calperiods, calperiodlookup = make_calendar_period_lookup()
    def get_mean_pop_by_age_for_sex_and_year(sex, year):
        if sex == 'F':
            key = "age_range_f"
        else:
            key = "age_range_m"

        num_by_age = summarize(
            extract_results(results_folder,
                            module="tlo.methods.demography",
                            key=key,
                            custom_generate_series=(
                                lambda df_: df_.loc[pd.to_datetime(df_.date).dt.year == year].drop(
                                    columns=['date']
                                ).melt(
                                    var_name='age_grp'
                                ).set_index('age_grp')['value']
                            ),
                            do_scaling=True
                            ),
            collapse_columns=True,
            only_mean=True
        )
        return num_by_age

    for year in range(int(min_year), int(max_year),1): #2049, 2059, 2069, 2079]:
        if year in pop_model.index:
            # Get WPP data:
            wpp_thisyr = wpp_ann.loc[wpp_ann['Year'] == year].groupby(['Sex', 'Age_Grp'])['Count'].sum()

            pops = dict()
            for sex in ['M', 'F']:
                # Import model results and scale:
                model = get_mean_pop_by_age_for_sex_and_year(sex, year)
                # Make into dataframes for plotting:
                pops[sex] = {
                    'Model': model,
                    'WPP': wpp_thisyr.loc[sex]
                }

                if year == 2018:
                    # Import and format Census data, and add to the comparison if the year is 2018 (year of census)
                    pops[sex]['Census'] = cens.loc[cens['Sex'] == sex].groupby(by='Age_Grp')['Count'].sum()

            # Simple plot of population pyramid
            fig = plt.figure()
            ax = plot_population_pyramid(data=pops, fig=fig)
            ax.set_title(f'Population Pyramid in {year}')
            fig.savefig(make_graph_file_name(f"Pop_Size_{year}"))
            plt.close(fig)

    # Make a gif
    frames = []
    for year in range(int(min_year), int(max_year)):
        image = imageio.v2.imread(make_graph_file_name(f"Pop_Size_{year}"))
        frames.append(image)

    imageio.mimsave(output_folder / f"Pop_Pyramids_{min_year}-{max_year}.gif",
                    frames,
                    fps=10)

    # %% Births: Number over time
    # Births over time (Model)
    births_results = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="on_birth",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()
        ),
        do_scaling=True
    )

    # Aggregate the model outputs into five year periods:
    calperiods, calperiodlookup = make_calendar_period_lookup()
    births_results.index = births_results.index.map(calperiodlookup).astype(make_calendar_period_type())
    births_results = births_results.groupby(by=births_results.index).sum()
    births_results = births_results.replace({0: np.nan})

    # Produce summary of results:
    births_model = summarize(births_results, collapse_columns=True)
    births_model.columns = ['Model_' + col for col in births_model.columns]

    # Births over time (WPP)
    wpp_births = pd.read_csv(Path(resourcefilepath) / "demography" / "ResourceFile_TotalBirths_WPP.csv")
    wpp_births = wpp_births.groupby(['Period', 'Variant'])['Total_Births'].sum().unstack()
    wpp_births.index = wpp_births.index.astype(make_calendar_period_type())
    wpp_births.columns = 'WPP_' + wpp_births.columns

    # Make the WPP line connect to the 'medium variant' to make the lines look like they join up
    wpp_births['WPP_continuous'] = wpp_births['WPP_Estimates'].combine_first(wpp_births['WPP_Medium variant'])

    # Births in 2018 Census
    cens_births = pd.read_csv(Path(resourcefilepath) / "demography" / "ResourceFile_Births_2018Census.csv")
    cens_births_per_5y_per = cens_births['Count'].sum() * 5

    # Merge in model results
    births = wpp_births.merge(births_model, right_index=True, left_index=True, how='left')
    births['Census'] = np.nan
    births.at[cens['Period'][0], 'Census'] = cens_births_per_5y_per

    # Create a variety of time periods for the plot
    time_period = {
        '1950-2099': births.index,
        '2010-2029': [(2010 <= int(x[0])) & (int(x[1]) < 2030) for x in births.index.str.split('-')],
        '2010-2040': [(2010 <= int(x[0])) & (int(x[1]) <= 2040) for x in births.index.str.split('-')],
        '2010-2060': [(2010 <= int(x[0])) & (int(x[1]) <= int(max_year)) for x in births.index.str.split('-')],
    }

    # Plot:
    for tp in time_period:
        births_loc = births.loc[time_period[tp]]
        fig, ax = plt.subplots()
        ax.plot()
        ax.plot(
            births_loc.index,
            births_loc['Census'] / 1e6,
            linestyle='none', marker='o', markersize=10, label='Census', zorder=10, color=colors['Census'])
        ax.plot(
            births_loc.index,
            births_loc['Model_mean'] / 1e6,
            label='Model',
            color=colors['Model']
        )
        ax.fill_between((births_loc.index).to_numpy(),
                        (births_loc['Model_lower'] / 1e6).to_numpy(),
                        (births_loc['Model_upper'] / 1e6).to_numpy(),
                        facecolor=colors['Model'], alpha=0.2)
        ax.plot(
            births_loc.index,
            births_loc['WPP_continuous'] / 1e6,
            color=colors['WPP'],
            label='WPP'
        )
        ax.fill_between((births_loc.index).to_numpy(),
                        (births_loc['WPP_Low variant'] / 1e6).to_numpy(),
                        (births_loc['WPP_High variant'] / 1e6).to_numpy(),
                        facecolor=colors['WPP'], alpha=0.2)
        ax.legend(loc='upper left')
        plt.xticks(rotation=90)
        ax.set_title(f"Number of Births {tp}")
        ax.set_xlabel('Calendar Period')
        ax.set_ylabel('Births per period (millions)')
        ax.set_ylim(0, 8.0)
        plt.xticks(np.arange(len(births_loc.index)), births_loc.index)
        plt.tight_layout()
        plt.savefig(make_graph_file_name(f"Births_Over_Time_{tp}"))
        plt.close(fig)


    # %% Age-specific fertility

    def get_births_by_year_and_age_range_of_mother_at_pregnancy(_df):
        _df = _df.drop(_df.index[_df.mother == -1])
        _df = _df.assign(year=_df['date'].dt.year)
        _df['mother_age_range'] = _df['mother_age_at_pregnancy'].map(agegrplookup)
        return _df.groupby(['year', 'mother_age_range'])['year'].count()

    births_by_mother_age_at_pregnancy = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="on_birth",
        custom_generate_series=get_births_by_year_and_age_range_of_mother_at_pregnancy,
        do_scaling=False
    )

    def get_num_adult_women_by_age_range(_df):
        _df = _df.assign(year=_df['date'].dt.year)
        _df = _df.set_index(_df['year'], drop=True)
        _df = _df.drop(columns='date')
        select_col = adult_age_groups
        ser = _df[select_col].stack()
        ser.index.names = ['year', 'mother_age_range']
        return ser

    num_adult_women = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="age_range_f",
        custom_generate_series=get_num_adult_women_by_age_range,
        do_scaling=False
    )

    # Compute age-specific fertility rates
    asfr = summarize(births_by_mother_age_at_pregnancy.div(num_adult_women)).sort_index()

    # Get the age-specific fertility rates of the WPP source
    wpp = pd.read_csv(resourcefilepath / 'demography' / 'ResourceFile_ASFR_WPP.csv')

    def expand_by_year(periods, vals, years=range(2010, int(max_year))):
        _ser = dict()
        for y in years:
            _ser[y] = vals.loc[(periods == calperiodlookup[y])].values[0]
        return _ser.keys(), _ser.values()

    fig, ax = plt.subplots(2, 4, sharex=True, sharey=True)
    ax = ax.reshape(-1)
    years = range(2010, int(max_year))
    for i, _agegrp in enumerate(adult_age_groups):
        model = asfr.loc[(slice(2011, years[-1]), _agegrp), :].unstack()
        data = wpp.loc[
            (wpp.Age_Grp == _agegrp) & wpp.Variant.isin(['WPP_Estimates', 'WPP_Medium variant']), ['Period', 'asfr']
        ]
        data_year, data_asfr = expand_by_year(data.Period, data.asfr, years)

        l1 = ax[i].plot(data_year, data_asfr, 'k-', label='WPP')
        l2 = ax[i].plot(model.index, model[(0, 'mean', _agegrp)], 'r-', label='Model')
        ax[i].fill_between((model.index).to_numpy(),
                           (model[(0, 'lower', _agegrp)]).to_numpy(),
                           (model[(0, 'upper', _agegrp)]).to_numpy(),
                           color='r',
                           alpha=0.2)
        ax[i].set_ylim(0, 0.3)
        ax[i].set_title(f'Age at Conception: {_agegrp}y', fontsize=6)
        ax[i].set_xlabel('Year')
        ax[i].set_ylabel('Live births per woman')

    ax[-1].set_axis_off()
    fig.legend(handles=(l1[0], l2[0]), labels=('WPP', 'Model'), loc='lower right')
    fig.tight_layout()
    fig.savefig(make_graph_file_name("asfr_model_vs_data"))
    plt.close(fig)

    # Plot with respect to age, averaged in the five-year periods:
    model_asfr = asfr.unstack() \
                     .droplevel(axis=1, level=0) \
                     .groupby(by=asfr.index.levels[0].map(calperiodlookup)) \
                     .mean() \
                     .stack()

    data_asfr = wpp.loc[
        wpp.Variant.isin(['WPP_Estimates', 'WPP_Medium variant']), ['Age_Grp', 'Period', 'asfr']
    ].groupby(by=['Age_Grp', 'Period'])['asfr'].mean().unstack().T

    fig, ax = plt.subplots()
    _period = '2015-2019'
    to_plot = pd.concat([model_asfr.loc[_period], data_asfr.loc[_period]], axis=1)
    ax.plot(to_plot.index, to_plot[_period], label='WPP', color=colors['WPP'])
    ax.plot(to_plot.index, to_plot['mean'], label='Model', color=colors['Model'])
    ax.fill_between((to_plot.index).to_numpy(),
                    (to_plot['lower']).to_numpy(),
                    (to_plot['upper']).to_numpy(),
                    color=colors['Model'],
                    alpha=0.2)
    ax.set_xlabel('Age at Conception')
    ax.set_ylabel('Live births per woman-year')
    ax.set_title(f'{_period}')
    ax.legend()
    fig.suptitle('Live Births By Age of Mother At Conception')
    fig.tight_layout()
    fig.savefig(make_graph_file_name("asfr_model_vs_data_average_by_age_2015-2019"))
    plt.close(fig)

    # %% All-Cause Deaths
    #  Get Model output (aggregating by period before doing the summarize)

    # Aggregate the model outputs into five-year periods for age and time:
    def get_counts_of_death_by_period_sex_agegrp(df):
        df['year'] = df['date'].dt.year
        df['Age_Grp'] = df['age'].map(agegrplookup).astype(make_age_grp_types())
        df['Period'] = df['year'].map(calperiodlookup).astype(make_calendar_period_type())
        df['Sex'] = df['sex']
        return df.groupby(by=['Period', 'Sex', 'Age_Grp'])['person_id'].count()

    results_deaths = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=get_counts_of_death_by_period_sex_agegrp,
        do_scaling=True
    )

    # Load WPP data
    wpp_deaths = pd.read_csv(Path(resourcefilepath) / "demography" / "ResourceFile_TotalDeaths_WPP.csv")
    wpp_deaths['Period'] = wpp_deaths['Period'].astype(make_calendar_period_type())
    wpp_deaths['Age_Grp'] = wpp_deaths['Age_Grp'].astype(make_age_grp_types())

    # Load GBD
    gbd_deaths = format_gbd(pd.read_csv(resourcefilepath / "gbd" / "ResourceFile_TotalDeaths_GBD2019.csv"))

    # For GBD, compute sums by period
    gbd_deaths = pd.DataFrame(
        gbd_deaths.drop(columns=['Year']).groupby(by=['Period', 'Sex', 'Age_Grp', 'Variant']).sum()).reset_index()

    # 1) Plot deaths over time (all ages)

    # Summarize model results (for all ages) and process into desired format:
    deaths_model_by_period = summarize(results_deaths.groupby(level=0).sum(), collapse_columns=True).reset_index()
    deaths_model_by_period = deaths_model_by_period.melt(
        id_vars=['Period'], value_vars=['mean', 'lower', 'upper'], var_name='Variant', value_name='Count')
    deaths_model_by_period['Variant'] = 'Model_' + deaths_model_by_period['Variant']

    # Sum WPP and GBD to give total deaths by period
    wpp_deaths_byperiod = wpp_deaths.groupby(by=['Variant', 'Period'])['Count'].sum().reset_index()
    gbd_deaths_by_period = gbd_deaths.groupby(by=['Variant', 'Period'])['Count'].sum().reset_index()

    # Combine into one large dataframe
    deaths_by_period = pd.concat(
        [deaths_model_by_period,
         wpp_deaths_byperiod,
         gbd_deaths_by_period
         ],
        ignore_index=True, sort=False
    )

    deaths_by_period['Period'] = deaths_by_period['Period'].astype(make_calendar_period_type())

    # Total of deaths over time (summing age/sex)
    deaths_by_period = pd.DataFrame(deaths_by_period.groupby(by=['Period', 'Variant']).sum()).unstack()
    deaths_by_period.columns = pd.Index([label[1] for label in deaths_by_period.columns.tolist()])
    deaths_by_period = deaths_by_period.replace(0, np.nan)

    # Make the WPP line connect to the 'medium variant' to make the lines look like they join up
    deaths_by_period['WPP_continuous'] = deaths_by_period['WPP_Estimates'].combine_first(
        deaths_by_period['WPP_Medium variant'])

    # Plot:
    fig, ax = plt.subplots()
    ax.plot(
        deaths_by_period.index,
        deaths_by_period['WPP_continuous'] / 1e6,
        label='WPP',
        color=colors['WPP'])
    ax.fill_between(
        (deaths_by_period.index).to_numpy(),
        (deaths_by_period['WPP_Low variant'] / 1e6).to_numpy(),
        (deaths_by_period['WPP_High variant'] / 1e6).to_numpy(),
        facecolor=colors['WPP'], alpha=0.2)
    ax.plot(
        deaths_by_period.index,
        deaths_by_period['GBD_Est'] / 1e6,
        label='GBD',
        color=colors['GBD']
    )
    ax.fill_between(
        (deaths_by_period.index).to_numpy(),
        (deaths_by_period['GBD_Lower'] / 1e6).to_numpy(),
        (deaths_by_period['GBD_Upper'] / 1e6).to_numpy(),
        facecolor=colors['GBD'], alpha=0.2)
    ax.plot(
        deaths_by_period.index,
        deaths_by_period['Model_mean'] / 1e6,
        label='Model',
        color=colors['Model']
    )
    ax.fill_between(
        (deaths_by_period.index).to_numpy(),
        (deaths_by_period['Model_lower'] / 1e6).to_numpy(),
        (deaths_by_period['Model_upper'] / 1e6).to_numpy(),
        facecolor=colors['Model'], alpha=0.2)

    def find_index_with_string(target_list, target_string=max_year):

        for index, string in enumerate(target_list):
            if target_string in string:
                return index
        return -1
    # Function to extract the starting year from the index string

    max_index = find_index_with_string(deaths_by_period.index)
    ax.set_title('Number of Deaths')
    ax.legend(loc='upper left')
    ax.set_xlabel('Calendar Period')
    ax.set_ylabel('Number per period (millions)')
    plt.xticks(np.arange(len(deaths_by_period.index)), deaths_by_period.index, rotation=90)
    ax.set_xlim(right = max_index)
    fig.tight_layout()
    plt.savefig(make_graph_file_name("Deaths_OverTime"))
    plt.close(fig)


    # 2) Plots by age-group for selected period:

    # Summarize model results (with breakdown by age/sex/period) and process into desired format:
    deaths_model_by_ageperiod = summarize(results_deaths, collapse_columns=True).reset_index()
    deaths_model_by_ageperiod = deaths_model_by_ageperiod.melt(
        id_vars=['Period',  'Age_Grp'], value_vars=['mean', 'lower', 'upper'], var_name='Variant',
        value_name='Count')
    deaths_model_by_ageperiod['Variant'] = 'Model_' + deaths_model_by_ageperiod['Variant']

    # Combine into one large dataframe
    deaths_by_ageperiod = pd.concat(
        [deaths_model_by_ageperiod,
         wpp_deaths,
         gbd_deaths
         ],
        ignore_index=True, sort=False
    )

    # Deaths by age, during in selection periods
    calperiods_selected = list()
    for cal in calperiods:
        if cal != '2100+':
            if (2010 <= int(cal.split('-')[0])) and (int(cal.split('-')[1]) < int(max_year)):
                calperiods_selected.append(cal)

    for period in calperiods_selected:

            fig, ax = plt.subplots()
            tot_deaths_byage = pd.DataFrame(
                deaths_by_ageperiod.loc[
                    (deaths_by_ageperiod['Period'] == period)].groupby(
                    by=['Variant', 'Age_Grp'])['Count'].sum()).unstack()
            tot_deaths_byage.columns = pd.Index([label[1] for label in tot_deaths_byage.columns.tolist()])
            tot_deaths_byage = tot_deaths_byage.transpose()

            if 'WPP_Medium variant' in tot_deaths_byage.columns:
                ax.plot(
                    tot_deaths_byage.index,
                    tot_deaths_byage['WPP_Medium variant'] / 1e3,
                    label='WPP',
                    color=colors['WPP'])
                ax.fill_between(
                    (tot_deaths_byage.index).to_numpy(),
                    (tot_deaths_byage['WPP_Low variant'] / 1e3).to_numpy(),
                    (tot_deaths_byage['WPP_High variant'] / 1e3).to_numpy(),
                    facecolor=colors['WPP'], alpha=0.2)
            else:
                ax.plot(
                    tot_deaths_byage.index,
                    tot_deaths_byage['WPP_Estimates'] / 1e3,
                    label='WPP',
                    color=colors['WPP'])

            if 'GBD_Est' in tot_deaths_byage.columns:
                ax.plot(
                    tot_deaths_byage.index,
                    tot_deaths_byage['GBD_Est'] / 1e3,
                    label='GBD',
                    color=colors['GBD'])
                ax.fill_between(
                    (tot_deaths_byage.index).to_numpy(),
                    (tot_deaths_byage['GBD_Lower'] / 1e3).to_numpy(),
                    (tot_deaths_byage['GBD_Upper'] / 1e3).to_numpy(),
                    facecolor=colors['GBD'], alpha=0.2)

            ax.plot(
                tot_deaths_byage.index,
                tot_deaths_byage['Model_mean'] / 1e3,
                label='Model',
                color=colors['Model'])
            ax.fill_between(
                (tot_deaths_byage.index).to_numpy(),
                (tot_deaths_byage['Model_lower'] / 1e3).to_numpy(),
                (tot_deaths_byage['Model_upper'] / 1e3).to_numpy(),
                facecolor=colors['Model'], alpha=0.2)

            ax.set_xticks(np.arange(len(tot_deaths_byage.index)))
            ax.set_xticklabels(tot_deaths_byage.index, rotation=90)
            ax.set_title(f"Number of Deaths {period}")
            ax.legend(loc='upper right')
            ax.set_xlabel('Age Group')
            ax.set_ylabel('Deaths per period (thousands)')
            ax.set_ylim(0, 250)

            fig.tight_layout()
            fig.savefig(make_graph_file_name(f"Deaths_By_Age_{period}"))
            plt.close(fig)


    # 2b) All on one graph
    # Create a figure and axis for plotting

    fig, axs = plt.subplots(int(len(calperiods_selected) / 3), 3, figsize=(int(len(calperiods_selected) / 3) * 7.5,  3 * 3.5))

    for idx, period in enumerate(calperiods_selected):
        row = idx // 3
        col = idx % 3
        ax = axs[row, col]

        tot_deaths_byage = pd.DataFrame(
            deaths_by_ageperiod.loc[
                (deaths_by_ageperiod['Period'] == period)].groupby(
                by=['Variant', 'Age_Grp'])['Count'].sum()).unstack()

        tot_deaths_byage.columns = pd.Index([label[1] for label in tot_deaths_byage.columns.tolist()])
        tot_deaths_byage = tot_deaths_byage.transpose()

        if 'WPP_Medium variant' in tot_deaths_byage.columns:
            ax.plot(
                tot_deaths_byage.index,
                tot_deaths_byage['WPP_Medium variant'] / 1e3,
                label='WPP',
                color=colors['WPP'])
            ax.fill_between(
                (tot_deaths_byage.index).to_numpy(),
                (tot_deaths_byage['WPP_Low variant'] / 1e3).to_numpy(),
                (tot_deaths_byage['WPP_High variant'] / 1e3).to_numpy(),
                facecolor=colors['WPP'], alpha=0.2)
        else:
            ax.plot(
                tot_deaths_byage.index,
                tot_deaths_byage['WPP_Estimates'] / 1e3,
                label='WPP',
                color=colors['WPP'])

        if 'GBD_Est' in tot_deaths_byage.columns:
            ax.plot(
                tot_deaths_byage.index,
                tot_deaths_byage['GBD_Est'] / 1e3,
                label='GBD',
                color=colors['GBD'])
            ax.fill_between(
                (tot_deaths_byage.index).to_numpy(),
                (tot_deaths_byage['GBD_Lower'] / 1e3).to_numpy(),
                (tot_deaths_byage['GBD_Upper'] / 1e3).to_numpy(),
                facecolor=colors['GBD'], alpha=0.2)

        ax.plot(
            tot_deaths_byage.index,
            tot_deaths_byage['Model_mean'] / 1e3,
            label='Model',
            color=colors['Model'])
        ax.fill_between(
            (tot_deaths_byage.index).to_numpy(),
            (tot_deaths_byage['Model_lower'] / 1e3).to_numpy(),
            (tot_deaths_byage['Model_upper'] / 1e3).to_numpy(),
            facecolor=colors['Model'], alpha=0.2)

        ax.set_xticks(np.arange(len(tot_deaths_byage.index)))
        ax.set_xticklabels(tot_deaths_byage.index, rotation=90)
        ax.set_title(f"Number of Deaths {period}")
        if idx == 0:
            ax.legend(loc='upper right')
        ax.set_xlabel('Age Group')
        ax.set_ylabel('Deaths per period (thousands)')
        ax.set_ylim(0, 250)

    fig.tight_layout()
    fig.savefig(make_graph_file_name("Deaths_By_Age_All_Periods"))
    # 3) Plots by sex and age-group for selected period:

    # Summarize model results (with breakdown by age/sex/period) and process into desired format:
    deaths_model_by_agesexperiod = summarize(results_deaths, collapse_columns=True).reset_index()
    deaths_model_by_agesexperiod = deaths_model_by_agesexperiod.melt(
        id_vars=['Period', 'Sex', 'Age_Grp'], value_vars=['mean', 'lower', 'upper'], var_name='Variant',
        value_name='Count')
    deaths_model_by_agesexperiod['Variant'] = 'Model_' + deaths_model_by_agesexperiod['Variant']

    # Combine into one large dataframe
    deaths_by_agesexperiod = pd.concat(
        [deaths_model_by_agesexperiod,
         wpp_deaths,
         gbd_deaths
         ],
        ignore_index=True, sort=False
    )

    # Deaths by age, during in selection periods
    calperiods_selected = list()
    for cal in calperiods:
        if cal != '2100+':
            if (2010 <= int(cal.split('-')[0])) and (int(cal.split('-')[1]) < int(max_year)):
                calperiods_selected.append(cal)

    for period in calperiods_selected:

        for i, sex in enumerate(['F', 'M']):

            fig, ax = plt.subplots()
            tot_deaths_byage = pd.DataFrame(
                deaths_by_agesexperiod.loc[
                    (deaths_by_agesexperiod['Period'] == period) & (deaths_by_agesexperiod['Sex'] == sex)].groupby(
                    by=['Variant', 'Age_Grp'])['Count'].sum()).unstack()
            tot_deaths_byage.columns = pd.Index([label[1] for label in tot_deaths_byage.columns.tolist()])
            tot_deaths_byage = tot_deaths_byage.transpose()

            if 'WPP_Medium variant' in tot_deaths_byage.columns:
                ax.plot(
                    tot_deaths_byage.index,
                    tot_deaths_byage['WPP_Medium variant'] / 1e3,
                    label='WPP',
                    color=colors['WPP'])
                ax.fill_between(
                    (tot_deaths_byage.index).to_numpy(),
                    (tot_deaths_byage['WPP_Low variant'] / 1e3).to_numpy(),
                    (tot_deaths_byage['WPP_High variant'] / 1e3).to_numpy(),
                    facecolor=colors['WPP'], alpha=0.2)
            else:
                ax.plot(
                    tot_deaths_byage.index,
                    tot_deaths_byage['WPP_Estimates'] / 1e3,
                    label='WPP',
                    color=colors['WPP'])

            if 'GBD_Est' in tot_deaths_byage.columns:
                ax.plot(
                    tot_deaths_byage.index,
                    tot_deaths_byage['GBD_Est'] / 1e3,
                    label='GBD',
                    color=colors['GBD'])
                ax.fill_between(
                    (tot_deaths_byage.index).to_numpy(),
                    (tot_deaths_byage['GBD_Lower'] / 1e3).to_numpy(),
                    (tot_deaths_byage['GBD_Upper'] / 1e3).to_numpy(),
                    facecolor=colors['GBD'], alpha=0.2)

            ax.plot(
                tot_deaths_byage.index,
                tot_deaths_byage['Model_mean'] / 1e3,
                label='Model',
                color=colors['Model'])
            ax.fill_between(
                (tot_deaths_byage.index).to_numpy(),
                (tot_deaths_byage['Model_lower'] / 1e3).to_numpy(),
                (tot_deaths_byage['Model_upper'] / 1e3).to_numpy(),
                facecolor=colors['Model'], alpha=0.2)

            ax.set_xticks(np.arange(len(tot_deaths_byage.index)))
            ax.set_xticklabels(tot_deaths_byage.index, rotation=90)
            ax.set_title(f"Number of Deaths {period}: {sexname(sex)}")
            ax.legend(loc='upper right')
            ax.set_xlabel('Age Group')
            ax.set_ylabel('Deaths per period (thousands)')
            ax.set_ylim(0, 200)

            fig.tight_layout()
            fig.savefig(make_graph_file_name(f"Deaths_By_Age_{sex}_{period}"))
            plt.close(fig)

    # 4) Life expectancy
    fig.tight_layout()

    dataframes = []
    wpp_le = pd.read_csv("/Users/rem76/PycharmProjects/TLOmodel/src/scripts/longterm_projections/Life_Expectancy_WPP_2010_2014.csv")
    for year in range(2010, int(max_year) + 1):
        df = get_life_expectancy_estimates(
            results_folder=args.results_folder,
            target_period=(datetime.date(year, 1, 1), datetime.date(year, 12, 31)),
            summary=False,
        )
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = summarize(results=df, only_mean=False, collapse_columns=False)#
        df['Year'] = year  # Add a new column for the year
        dataframes.append(df)
    # Concatenate all dataframes
    le_all_years = pd.concat(dataframes, ignore_index=True)
    le_all_years.set_index('Year', inplace=True)
    le_all_years.to_csv(args.results_folder / 'life_expectancy_estimates.csv', index=True)

    ax.plot(wpp_le['Time'], wpp_le['Value'], marker='o', color=colors['WPP'], label="WPP")
    le_all_years.columns = le_all_years.columns.get_level_values('stat')



    # 5) Deaths and Life Expectancy


    fig, ax = plt.subplots(1, 2, figsize=(15, 7.5))

    ax[0].plot(
        deaths_by_period.index,
        deaths_by_period['WPP_continuous'] / 1e6,
        label='WPP',
        color=colors['WPP'])
    ax[0].fill_between(
        (deaths_by_period.index).to_numpy(),
        (deaths_by_period['WPP_Low variant'] / 1e6).to_numpy(),
        (deaths_by_period['WPP_High variant'] / 1e6).to_numpy(),
        facecolor=colors['WPP'], alpha=0.2)
    ax[0].plot(
        deaths_by_period.index,
        deaths_by_period['GBD_Est'] / 1e6,
        label='GBD',
        color=colors['GBD']
    )
    ax[0].fill_between(
        (deaths_by_period.index).to_numpy(),
        (deaths_by_period['GBD_Lower'] / 1e6).to_numpy(),
        (deaths_by_period['GBD_Upper'] / 1e6).to_numpy(),
        facecolor=colors['GBD'], alpha=0.2)
    ax[0].plot(
        deaths_by_period.index,
        deaths_by_period['Model_mean'] / 1e6,
        label='Model',
        color=colors['Model']
    )
    ax[0].fill_between(
        (deaths_by_period.index).to_numpy(),
        (deaths_by_period['Model_lower'] / 1e6).to_numpy(),
        (deaths_by_period['Model_upper'] / 1e6).to_numpy(),
        facecolor=colors['Model'], alpha=0.2)

    max_index = find_index_with_string(deaths_by_period.index)
    min_index = find_index_with_string(deaths_by_period.index, '2000')
    period_labels = deaths_by_period.index[min_index:max_index].astype(str)
    ax[0].set_title('Panel A: Number of Deaths')
    ax[0].legend(loc='upper left')
    ax[0].set_xlabel('Calendar Period')
    ax[0].set_ylabel('Number per period (millions)')
    ax[0].set_xlim(left = min_index, right = max_index - 1)
    fig.tight_layout()

    # Panel B - Life expectancy
    ax[1].plot(
        le_all_years.index[1::2],
        le_all_years.iloc[1::2]['mean'],
        marker='o',
        color='#9DC7C8',
        label="F"
    )
    ax[1].fill_between(
        le_all_years.index[1::2],
        le_all_years.iloc[1::2]['lower'],
        le_all_years.iloc[1::2]['upper'],
        color='#9DC7C8',
        alpha=0.3
    )

    ax[1].plot(
        le_all_years.index[0::2],
        le_all_years.iloc[0::2]['mean'],
        marker='o',
        color='#4E598C',
        label="M"
    )

    ax[1].fill_between(
        le_all_years.index[0::2],
        le_all_years.iloc[0::2]['lower'],
        le_all_years.iloc[0::2]['upper'],
        color='#4E598C',
        alpha=0.3
    )
    ax[1].plot(wpp_le['Time'], wpp_le['Value'], marker='o', color=colors['WPP'], label="WPP")

    ax[1].legend(loc='lower right')
    ax[1].set_xlabel('Year')
    ax[1].set_ylim(0, 80)
    ax[1].set_ylabel('Life Expectancy (Years)')
    ax[1].set_title('Panel B: Life Expectancy')
    fig.tight_layout()
    plt.savefig(make_graph_file_name("Death_Life_expectancy_over_years"))
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources')
    )

