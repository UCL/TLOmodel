from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from tlo import Date
from tlo.analysis.utils import extract_results, get_scenario_outputs, summarize

PREFIX_ON_FILENAME = '4'

# Declare period for which the results will be generated (defined inclusively)
TARGET_PERIOD = (Date(2015, 1, 1), Date(2019, 12, 31))

# appointment dict to match model and data
appt_dict = {'Under5OPD': 'OPD',
             'Over5OPD': 'OPD',
             'AntenatalFirst': 'AntenatalTotal',
             'ANCSubsequent': 'AntenatalTotal',
             'NormalDelivery': 'Delivery',
             'CompDelivery': 'Delivery',
             'EstMedCom': 'EstAdult',
             'EstNonCom': 'EstAdult',
             'VCTPositive': 'VCTTests',
             'VCTNegative': 'VCTTests',
             'DentAccidEmerg': 'DentalAll',
             'DentSurg': 'DentalAll',
             'DentU5': 'DentalAll',
             'DentO5': 'DentalAll',
             'MentOPD': 'MentalAll',
             'MentClinic': 'MentalAll'
             }


def get_annual_num_appts_by_level(results_folder: Path) -> pd.DataFrame:
    """Return pd.DataFrame gives the (mean) simulated annual number of appointments of each type at each level."""

    def get_counts_of_appts(_df):
        """Get the mean number of appointments of each type being used each year at each level.
        Need to rename appts to match standardized categories from the DHIS2 data."""

        def unpack_nested_dict_in_series(_raw: pd.Series):
            return pd.concat(
                {
                  idx: pd.DataFrame.from_dict(mydict) for idx, mydict in _raw.items()
                 }
             ).unstack().fillna(0.0).astype(int)

        return _df \
            .loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD), 'Number_By_Appt_Type_Code_And_Level'] \
            .pipe(unpack_nested_dict_in_series) \
            .rename(columns=appt_dict, level=1) \
            .groupby(level=[0, 1], axis=1).sum() \
            .mean(axis=0)  # mean over each year (row)

    return summarize(
        extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='HSI_Event',
                custom_generate_series=get_counts_of_appts,
                do_scaling=True
            ),
        only_mean=True,
        collapse_columns=True,
        ).unstack().astype(int)


def get_annual_num_appts_by_level_with_confidence_interval(results_folder: Path) -> pd.DataFrame:
    """Return pd.DataFrame gives the (mean) simulated annual number of appointments of each type at each level,
    with 95% confidence interval."""

    def get_counts_of_appts(_df):
        """Get the mean number of appointments of each type being used each year at each level.
        Need to rename appts to match standardized categories from the DHIS2 data."""

        def unpack_nested_dict_in_series(_raw: pd.Series):
            return pd.concat(
                {
                  idx: pd.DataFrame.from_dict(mydict) for idx, mydict in _raw.iteritems()
                 }
             ).unstack().fillna(0.0).astype(int)

        return _df \
            .loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD), 'Number_By_Appt_Type_Code_And_Level'] \
            .pipe(unpack_nested_dict_in_series) \
            .rename(columns=appt_dict, level=1) \
            .groupby(level=[0, 1], axis=1).sum() \
            .mean(axis=0)  # mean over each year (row)

    return summarize(
        extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='HSI_Event',
                custom_generate_series=get_counts_of_appts,
                do_scaling=True
            ),
        only_mean=False,
        collapse_columns=True,
        ).unstack().astype(int)


def get_annual_num_appts_with_confidence_interval(results_folder: Path) -> pd.DataFrame:
    """Return pd.DataFrame gives the (mean) simulated annual number of appointments of each type at all levels,
    with 95% confidence interval."""

    def get_counts_of_appts(_df) -> pd.Series:
        """Get the mean number of appointments of each type being used each year at all levels.
        Need to rename appts to match standardized categories from the DHIS2 data."""

        return _df \
            .loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD), 'Number_By_Appt_Type_Code'] \
            .apply(pd.Series) \
            .rename(columns=appt_dict) \
            .groupby(level=0, axis=1).sum() \
            .mean(axis=0)  # mean over each year (row)

    return summarize(
        extract_results(
                results_folder,
                module='tlo.methods.healthsystem.summary',
                key='HSI_Event',
                custom_generate_series=get_counts_of_appts,
                do_scaling=True
            ),
        only_mean=False,
        collapse_columns=True,
        ).unstack().astype(int)


def get_simulation_usage_by_level(results_folder: Path) -> pd.DataFrame:
    """Returns the simulated MEAN USAGE PER YEAR DURING THE TIME_PERIOD, by appointment type and level.
    """

    # Get model outputs
    model_output = get_annual_num_appts_by_level(results_folder=results_folder)

    return model_output


def get_simulation_usage_by_level_with_confidence_interval(results_folder: Path) -> pd.DataFrame:
    """Returns the simulated MEAN USAGE PER YEAR DURING THE TIME_PERIOD with 95% confidence interval,
    by appointment type and level.
    """

    # Get model outputs
    model_output = get_annual_num_appts_by_level_with_confidence_interval(results_folder=results_folder)

    # Reformat
    model_output.columns = [' '.join(col).strip() for col in model_output.columns.values]
    model_output = model_output.melt(var_name='name', value_name='value', ignore_index=False)
    model_output['name'] = model_output['name'].str.split(' ')
    model_output['value_type'] = model_output['name'].str[0]
    model_output['appt_type'] = model_output['name'].str[1]
    model_output.drop(columns='name', inplace=True)
    model_output.reset_index(drop=False, inplace=True)
    model_output.rename(columns={'index': 'facility_level'}, inplace=True)

    return model_output


def get_simulation_usage_with_confidence_interval(results_folder: Path) -> pd.DataFrame:
    """Returns the simulated MEAN USAGE PER YEAR DURING THE TIME_PERIOD with 95% confidence interval,
    by appointment type.
    """

    # Get model outputs
    model_output = get_annual_num_appts_with_confidence_interval(results_folder=results_folder)

    # Reformat
    model_output = pd.DataFrame(model_output).T
    model_output.columns = [' '.join(col).strip() for col in model_output.columns.values]
    model_output = model_output.melt(var_name='name', value_name='value', ignore_index=False)
    model_output['name'] = model_output['name'].str.split(' ')
    model_output['value_type'] = model_output['name'].str[0]
    model_output['appt_type'] = model_output['name'].str[1]
    model_output.drop(columns='name', inplace=True)
    model_output.reset_index(drop=True, inplace=True)

    return model_output


def adjust_real_usage_on_mentalall(real_usage_df) -> pd.DataFrame:
    """This is to adjust the annual MentalAll usage in real usage dataframe.
    The MentalAll usage was not adjusted in the preprocessing stage considering individual facilities and very low
    reporting rates.
    We now directly adjust its annual usage by facility level using the aggregated annual reporting rates by
    facility level. The latter is calculated based on DHIS2 Mental Health Report reporting rates."""
    # the annual reporting rates for Mental Health Report by facility level (%), 2015-2019
    # could turn the reporting rates data into a ResourceFile if necessary
    rr = pd.DataFrame(index=['1a', '1b', '2', '3'], columns=list(range(2015, 2020)),
                      data=[[44.00, 39.33, 79.00, 97.33, 95.00],
                            [10.42, 12.50, 25.00, 40.00, 68.33],
                            [36.67, 39.44, 37.22, 63.89, 56.67],
                            [50.00, 45.83, 45.83, 50.00, 45.83]])
    # make the adjustment assuming 100% reporting rates for each year
    for l in ['1a', '1b', '2', '3']:
        for y in range(2015, 2020):
            real_usage_df.loc[(real_usage_df.Facility_Level == l)
                              & (real_usage_df.Year == y)
                              & (real_usage_df.Appt_Type == 'MentalAll'), 'Usage'] = (
                real_usage_df.loc[(real_usage_df.Facility_Level == l)
                                  & (real_usage_df.Year == y)
                                  & (real_usage_df.Appt_Type == 'MentalAll'), 'Usage'] * 100 / rr.loc[l, y]
            )

    return real_usage_df


def get_real_usage(resourcefilepath, adjusted=True) -> pd.DataFrame:
    """
    Returns the adjusted (default) or unadjusted real data on the (MEAN) USAGE PER YEAR DURING THE TIME_PERIOD
    for each appointment at each level and all levels.
    """

    # add facility level and district columns to both real and simulation usage
    mfl = pd.read_csv(
        resourcefilepath / 'healthsystem' / 'organisation' / 'ResourceFile_Master_Facilities_List.csv')

    # Get real usage data
    if adjusted:
        real_usage = pd.read_csv(
            resourcefilepath / 'healthsystem' / 'real_appt_usage_data' /
            'real_monthly_usage_of_appt_type.csv')
    else:
        real_usage = pd.read_csv(
            resourcefilepath / 'healthsystem' / 'real_appt_usage_data' /
            'unadjusted_real_monthly_usage_of_appt_type.csv')

    # add Csection usage to Delivery, as Delivery has excluded Csection in real data file (to avoid overlap)
    # whereas Delivery in tlo output has included Csection
    real_delivery = real_usage.loc[(real_usage.Appt_Type == 'Delivery') | (real_usage.Appt_Type == 'Csection')
                                   ].groupby(['Year', 'Month', 'Facility_ID']).agg({'Usage': 'sum'}).reset_index()
    real_delivery['Appt_Type'] = 'Delivery'
    real_usage = pd.concat([real_usage.drop(real_usage[real_usage.Appt_Type == 'Delivery'].index),
                            real_delivery])

    # get facility_level for each record
    real_usage = real_usage.merge(mfl[['Facility_ID', 'Facility_Level']], left_on='Facility_ID', right_on='Facility_ID')

    # adjust annual MentalAll usage using annual reporting rates
    if adjusted:
        real_usage = adjust_real_usage_on_mentalall(real_usage)

    # assign date to each record
    real_usage['date'] = pd.to_datetime({'year': real_usage['Year'], 'month': real_usage['Month'], 'day': 1})

    # Produce table of the AVERAGE NUMBER PER YEAR DURING THE TIME_PERIOD of appointment type by level
    # limit to date
    totals_by_year = real_usage \
        .loc[real_usage['date'].between(*TARGET_PERIOD)] \
        .groupby(['Year', 'Appt_Type', 'Facility_Level'])['Usage'].sum()

    # Combine the TB data [which is yearly] (after dropping period outside 2017-2019 according to data consistency
    # and pandemic) with the rest of the data.
    # Note that TB data is not adjusted considering comparability with NTP reports.
    real_usage_TB = pd.read_csv(
        resourcefilepath / 'healthsystem' / 'real_appt_usage_data' / 'real_yearly_usage_of_TBNotifiedAll.csv')
    real_usage_TB = real_usage_TB.loc[real_usage_TB['Year'].isin([2017, 2018, 2019])]
    real_usage_TB = real_usage_TB.merge(mfl[['Facility_ID', 'Facility_Level']],
                                        left_on='Facility_ID', right_on='Facility_ID')
    totals_by_year_TB = real_usage_TB.groupby(['Year', 'Appt_Type', 'Facility_Level'])['Usage'].sum()

    annual_usage_by_level = pd.concat([totals_by_year.reset_index(), totals_by_year_TB.reset_index()], axis=0)

    # prepare annual usage by level with mean, 75% percentile, and 25% percentile
    annual_usage_by_level_with_ci = annual_usage_by_level.drop(columns='Year').groupby(
        ['Appt_Type', 'Facility_Level']
    ).describe(percentiles=[0.025, 0.975]
               ).stack(level=[0])[['mean', '2.5%', '97.5%']].reset_index().drop(columns='level_2')

    average_annual_by_level = annual_usage_by_level_with_ci[['Appt_Type', 'Facility_Level', 'mean']].set_index(
        ['Appt_Type', 'Facility_Level']).unstack()
    average_annual_by_level.columns = average_annual_by_level.columns.get_level_values(1)
    average_annual_by_level = average_annual_by_level.T

    annual_usage_by_level_with_ci = pd.melt(annual_usage_by_level_with_ci,
                                            id_vars=['Appt_Type', 'Facility_Level'], var_name='value_type')
    annual_usage_by_level_with_ci.value_type = annual_usage_by_level_with_ci.value_type.replace({'2.5%': 'lower',
                                                                                                 '97.5%': 'upper'})

    # prepare annual usage at all levels with mean, 75% percentile, and 25% percentile
    annual_usage_with_ci = annual_usage_by_level.groupby(
        ['Year', 'Appt_Type'])['Usage'].sum().reset_index().drop(columns='Year').groupby(
        'Appt_Type').describe(percentiles=[0.025, 0.975]
                              ).stack(level=[0])[['mean', '2.5%', '97.5%']].reset_index().drop(columns='level_1')
    annual_usage_with_ci = pd.melt(annual_usage_with_ci,
                                   id_vars='Appt_Type', var_name='value_type')
    annual_usage_with_ci.value_type = annual_usage_with_ci.value_type.replace({'2.5%': 'lower', '97.5%': 'upper'})

    return average_annual_by_level, annual_usage_by_level_with_ci, annual_usage_with_ci


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    """Compare appointment usage from model output with real appointment usage.
    The real appointment usage is collected from DHIS2 system and HIV Dept."""

    def format_and_save(_fig, _ax, _name_of_plot):
        _ax.set_title(_name_of_plot)
        _ax.set_yscale('log')
        _ax.set_ylim(1 / 20, 20)
        _ax.set_yticks([1 / 10, 1.0, 10])
        _ax.set_yticklabels(("<= 1/10", "1.0", ">= 10"))
        _ax.set_ylabel('Model / Data')
        _ax.set_xlabel('Appointment Type')
        _ax.tick_params(axis='x', labelrotation=90)
        _ax.xaxis.grid(True, which='major', linestyle='--')
        _ax.yaxis.grid(True, which='both', linestyle='--')
        _ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        _fig.tight_layout()
        _fig.savefig(make_graph_file_name(_name_of_plot.replace('\n', '_').replace(' ', '_')))
        _fig.show()
        plt.close(_fig)

    make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_{stub}.png"  # noqa: E731

    # get average annual usage by level for Simulation and Real
    simulation_usage = get_simulation_usage_by_level(results_folder)
    simulation_usage = simulation_usage.reset_index().replace({'index': {'1b': '2'}}).groupby('index').sum()

    real_usage = get_real_usage(resourcefilepath)[0]
    real_usage = real_usage.reset_index().replace({'Facility_Level': {'1b': '2'}}).groupby('Facility_Level').sum()

    # Plot Simulation vs Real usage (Across all levels) (trimmed to 0.1 and 10)
    rel_diff_all_levels = (
        simulation_usage.sum(axis=0) / real_usage.sum(axis=0)
    ).clip(lower=0.1, upper=10.0)

    name_of_plot = 'Model vs Real usage per appt type at all facility levels' \
                   '\n[Model average annual, Real average annual]'
    fig, ax = plt.subplots()
    ax.stem(rel_diff_all_levels.index, rel_diff_all_levels.values, bottom=1.0, label='All levels')
    for idx in rel_diff_all_levels.index:
        if not pd.isna(rel_diff_all_levels[idx]):
            ax.text(idx, rel_diff_all_levels[idx]*(1+0.2), round(rel_diff_all_levels[idx], 1),
                    ha='left', fontsize=8)
    format_and_save(fig, ax, name_of_plot)

    # Plot Simulation vs Real usage (At each level) (trimmed to 0.1 and 10)
    rel_diff_by_levels = (
        simulation_usage / real_usage
    ).clip(upper=10, lower=0.1).dropna(how='all', axis=0)

    name_of_plot = 'Model vs Real usage per appt type per facility level\n[Model average annual, Real average annual]'
    fig, ax = plt.subplots()
    marker_dict = {'0': 0,
                   '1a': 4,
                   '1b': 5,
                   '2': 6,
                   '3': 7,
                   '4': 1}  # Note that level 0/3/4 has very limited data
    for _level, _results in rel_diff_by_levels.iterrows():
        ax.plot(_results.index, _results.values, label=_level, linestyle='none', marker=marker_dict[_level])
    ax.axhline(1.0, color='r')
    format_and_save(fig, ax, name_of_plot)

    # Plot Simulation vs Real usage by level by appt type and show fraction of usage at each level
    # format data
    real_usage_all_levels = real_usage.sum(axis=0).reset_index().rename(
        columns={0: 'real_usage_all_levels', 'Appt_Type': 'appt_type'})
    real_usage_reformat = pd.melt(real_usage.reset_index(), id_vars='Facility_Level',
                                  var_name='appt_type', value_name='real_usage'
                                  ).rename(columns={'Facility_Level': 'facility_level'})
    simulation_usage_diff = pd.melt(simulation_usage.reset_index(), id_vars='index',
                                    var_name='appt_type', value_name='simulation_usage'
                                    ).rename(columns={'index': 'facility_level'})
    simulation_usage_diff = simulation_usage_diff.merge(
        real_usage_all_levels, on='appt_type', how='outer').merge(
        real_usage_reformat, on=['facility_level', 'appt_type'], how='outer')
    simulation_usage_diff['ratio'] = (simulation_usage_diff['simulation_usage'] /
                                      simulation_usage_diff['real_usage_all_levels'])
    simulation_usage_diff['real_usage_proportion'] = (simulation_usage_diff['real_usage'] /
                                                      simulation_usage_diff['real_usage_all_levels'])
    simulation_usage_plot = pd.pivot(simulation_usage_diff[['appt_type', 'facility_level', 'ratio']],
                                     index='appt_type', columns='facility_level', values='ratio'
                                     ).dropna(axis=1, how='all')
    real_usage_plot = pd.pivot(simulation_usage_diff[['appt_type', 'facility_level', 'real_usage_proportion']],
                               index='appt_type', columns='facility_level', values='real_usage_proportion'
                               ).dropna(axis=1, how='all')
    # plot
    name_of_plot = 'Model vs Real usage per appointment type, with fraction per level' \
                   '\n[Model average annual, Real average annual]'
    fig, ax = plt.subplots(figsize=(8, 5))
    cmp_paired = plt.get_cmap('Paired')
    cmp_paried_0 = matplotlib.colors.ListedColormap(tuple(cmp_paired.colors[i] for i in range(0, 10, 2)))
    cmp_paried_1 = matplotlib.colors.ListedColormap(tuple(cmp_paired.colors[i] for i in range(1, 11, 2)))
    simulation_usage_plot.plot(kind='bar', stacked=True, width=0.4,
                               cmap=cmp_paried_1, hatch='',
                               ax=ax, position=0)
    real_usage_plot.plot(kind='bar', stacked=True, width=0.4,
                         edgecolor='black', cmap=cmp_paried_1, hatch='',
                         ax=ax, position=1)
    ax.set_xlim(right=len(simulation_usage_plot) - 0.5)
    ax.yaxis.grid(True, which='major', linestyle='--')
    ax.set_ylabel('Usage per level / Real usage all levels')
    ax.set_xlabel('Appointment Type')
    ax.set_title(name_of_plot)
    legend_1 = plt.legend(simulation_usage_plot.columns, loc='upper left', bbox_to_anchor=(1.0, 0.5),
                          title='Facility Level')
    patch_simulation = matplotlib.patches.Patch(facecolor='beige', hatch='', label='Model')
    patch_real = matplotlib.patches.Patch(facecolor='beige', hatch='', edgecolor="black", label='Real')
    legend_2 = plt.legend(handles=[patch_simulation, patch_real], loc='lower left', bbox_to_anchor=(1.0, 0.6))
    fig.add_artist(legend_1)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace('\n', '_').replace(' ', '_')))
    plt.show()

    # Plot Simulation vs Adjusted & Real usage by level by appt type and show fraction of usage at each level
    # format data
    def format_real_usage(adjusted=True):
        _real_usage = get_real_usage(resourcefilepath, adjusted)[0]
        _real_usage_all_levels = _real_usage.sum(axis=0).reset_index().rename(
            columns={0: 'real_usage_all_levels' if adjusted else 'unadjusted_real_usage_all_levels',
                     'Appt_Type': 'appt_type'})
        return _real_usage_all_levels

    real_usage_all_levels = format_real_usage(adjusted=True)
    unadjusted_real_usage_all_levels = format_real_usage(adjusted=False)

    simulation_usage_diff = pd.melt(simulation_usage.reset_index(), id_vars='index',
                                    var_name='appt_type', value_name='simulation_usage'
                                    ).rename(columns={'index': 'facility_level'})
    simulation_usage_diff = simulation_usage_diff.merge(
        real_usage_all_levels, on='appt_type', how='outer').merge(
        unadjusted_real_usage_all_levels, on='appt_type', how='outer')
    simulation_usage_diff['ratio_to_adjusted_real'] = (simulation_usage_diff['simulation_usage'] /
                                                       simulation_usage_diff['real_usage_all_levels'])
    simulation_usage_diff['ratio_to_unadjusted_real'] = (simulation_usage_diff['simulation_usage'] /
                                                         simulation_usage_diff['unadjusted_real_usage_all_levels'])
    simulation_usage_plot_0 = pd.pivot(simulation_usage_diff[['appt_type', 'facility_level', 'ratio_to_adjusted_real']],
                                       index='appt_type', columns='facility_level', values='ratio_to_adjusted_real'
                                       ).dropna(axis=1, how='all')
    simulation_usage_plot_1 = pd.pivot(simulation_usage_diff[['appt_type', 'facility_level', 'ratio_to_unadjusted_real']],
                                       index='appt_type', columns='facility_level', values='ratio_to_unadjusted_real'
                                       ).dropna(axis=1, how='all')
    # plot
    name_of_plot = 'Model vs Real usage per appointment type, with fraction per level' \
                   '\n[Model average annual, Adjusted & Unadjusted Real average annual]'
    fig, ax = plt.subplots(figsize=(8, 5))
    cmp_paired = plt.get_cmap('Paired')
    cmp_paried_0 = matplotlib.colors.ListedColormap(tuple(cmp_paired.colors[i] for i in range(0, 10, 2)))
    cmp_paried_1 = matplotlib.colors.ListedColormap(tuple(cmp_paired.colors[i] for i in range(1, 11, 2)))
    simulation_usage_plot_0.plot(kind='bar', stacked=True, width=0.4,
                                 cmap=cmp_paried_1, hatch='',
                                 ax=ax, position=0)
    simulation_usage_plot_1.plot(kind='bar', stacked=True, width=0.4,
                                 edgecolor='black', cmap=cmp_paried_1, hatch='///',
                                 ax=ax, position=1)
    ax.set_xlim(right=len(simulation_usage_plot) - 0.5)
    ax.yaxis.grid(True, which='major', linestyle='--')
    ax.set_ylabel('Simulation usage per level / Real usage all levels')
    ax.set_xlabel('Appointment Type')
    ax.set_title(name_of_plot)
    legend_0 = plt.legend(simulation_usage_plot_0.columns, loc='upper left', bbox_to_anchor=(1.0, 0.5),
                          title='Facility Level')
    patch_simulation_0 = matplotlib.patches.Patch(facecolor='beige', hatch='', label='Adjusted real')
    patch_simulation_1 = matplotlib.patches.Patch(facecolor='beige', hatch='///', edgecolor="black",
                                                  label='Unadjusted real')
    legend_1 = plt.legend(handles=[patch_simulation_0, patch_simulation_1], loc='lower left', bbox_to_anchor=(1.0, 0.6))
    fig.add_artist(legend_0)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace('\n', '_').replace(' ', '_')))
    plt.show()

    def plot_model_vs_data_usage_with_ci(_rel_diff_with_ci, _name_of_plot):
        """ Plot Simulation (with 95% CI) vs Real usage (with 95% CI) (Across all levels) (trimmed to 0.1 and 10). """

        # prepare data
        _rel_diff_with_ci = _rel_diff_with_ci.pivot(
            index='appt_type', columns='value_type', values='ratio')
        _rel_diff_with_ci['lower_error'] = (_rel_diff_with_ci['mean'] - _rel_diff_with_ci['lower'])
        _rel_diff_with_ci['upper_error'] = (_rel_diff_with_ci['upper'] - _rel_diff_with_ci['mean'])

        # plot
        _fig, _ax = plt.subplots()
        asymmetric_error = [_rel_diff_with_ci['lower_error'].values,
                            _rel_diff_with_ci['upper_error'].values]
        _ax.errorbar(_rel_diff_with_ci.index.values,
                     _rel_diff_with_ci['mean'].values,
                     asymmetric_error, fmt='.', capsize=3.0)
        for _idx in _rel_diff_with_ci.index:
            if not pd.isna(_rel_diff_with_ci.loc[_idx, 'mean']):
                _ax.text(_idx,
                         _rel_diff_with_ci.loc[_idx, 'mean'] * (1 + 0.2),
                         round(_rel_diff_with_ci.loc[_idx, 'mean'], 1),
                         ha='left', fontsize=8)
        _ax.axhline(1.0, color='r')
        format_and_save(_fig, _ax, _name_of_plot)

    # Plot Simulation with 95% CI vs Real
    # Get usage and ratio across all levels
    simulation_usage_with_ci = get_simulation_usage_with_confidence_interval(results_folder)
    real_usage_all_levels = real_usage.sum(axis=0).reset_index().rename(
        columns={0: 'real_usage', 'Appt_Type': 'appt_type'})
    rel_diff_with_ci_simulation = simulation_usage_with_ci.merge(real_usage_all_levels, on='appt_type', how='outer')
    rel_diff_with_ci_simulation['ratio'] = (rel_diff_with_ci_simulation['value'] /
                                            rel_diff_with_ci_simulation['real_usage']).clip(upper=10, lower=0.1)
    rel_diff_with_ci_simulation.drop(columns=['value', 'real_usage'], inplace=True)
    # plot
    name_of_plot_with_ci_simulation = 'Model vs Real usage per appt type at all facility levels' \
                                      '\n[Model average annual with 95% CI, Real average annual]'
    plot_model_vs_data_usage_with_ci(rel_diff_with_ci_simulation, name_of_plot_with_ci_simulation)

    # Plot Simulation vs Real with 95% CI
    # Get usage and ratio across all levels
    real_usage_with_ci = get_real_usage(resourcefilepath)[2].rename(columns={'Appt_Type': 'appt_type'})
    simulation_usage_all_levels = simulation_usage.sum(axis=0).reset_index().rename(
        columns={0: 'simulation_usage', 'index': 'appt_type'})
    rel_diff_with_ci_real = real_usage_with_ci.merge(simulation_usage_all_levels, on='appt_type', how='outer')
    rel_diff_with_ci_real['ratio'] = (rel_diff_with_ci_real['simulation_usage'] /
                                      rel_diff_with_ci_real['value']).clip(upper=10, lower=0.1)
    rel_diff_with_ci_real.drop(columns=['value', 'simulation_usage'], inplace=True)
    rel_diff_with_ci_real.value_type = rel_diff_with_ci_real.value_type.replace({'lower': 'upper', 'upper': 'lower'})
    # plot
    name_of_plot_with_ci_real = 'Model vs Real usage per appt type at all facility levels' \
                                '\n[Model average annual, Real average annual with 95% CI]'
    plot_model_vs_data_usage_with_ci(rel_diff_with_ci_real, name_of_plot_with_ci_real)


if __name__ == "__main__":
    outputspath = Path('./outputs/bshe@ic.ac.uk')
    rfp = Path('./resources')

    # Find results folder (most recent run generated using that scenario_filename)
    scenario_filename = 'long_run_all_diseases.py'
    results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]

    # Test dataset:
    # results_folder = Path('/Users/tbh03/GitHub/TLOmodel/outputs/tbh03@ic.ac.uk/long_run_all_diseases-small')

    # If needed -- in the case that pickles were not created remotely during batch
    # create_pickles_locally(results_folder)

    # Run all the calibrations
    apply(results_folder=results_folder, output_folder=results_folder, resourcefilepath=rfp)
