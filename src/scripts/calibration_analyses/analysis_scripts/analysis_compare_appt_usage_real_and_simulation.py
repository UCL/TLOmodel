import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo import Date
from tlo.analysis.utils import extract_results, summarize

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

# appt color for plots
appt_color_dict = {
    'OPD': 'lightpink',
    'IPAdmission': 'palevioletred',
    'InpatientDays': 'mediumvioletred',

    'U5Malnutr': 'orchid',

    'FamPlan': 'darkseagreen',
    'AntenatalTotal': 'green',
    'Delivery': 'limegreen',
    'Csection': 'springgreen',
    'EPI': 'paleturquoise',
    'STI': 'mediumaquamarine',

    'AccidentsandEmerg': 'orange',

    'TBNew': 'yellow',

    'VCTTests': 'lightsteelblue',
    'NewAdult': 'cornflowerblue',
    'EstAdult': 'royalblue',
    'Peds': 'lightskyblue',
    'PMTCT': 'deepskyblue',
    'MaleCirc': 'mediumslateblue',

    'MentalAll': 'darkgrey',

    'DentalAll': 'silver',
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
            .rename(columns={'1b': '2'}, level=0) \
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
            .rename(columns={'1b': '2'}, level=0) \
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
    for level in ['1a', '1b', '2', '3']:
        for y in range(2015, 2020):
            real_usage_df.loc[(real_usage_df.Facility_Level == level)
                              & (real_usage_df.Year == y)
                              & (real_usage_df.Appt_Type == 'MentalAll'), 'Usage'] = (
                real_usage_df.loc[(real_usage_df.Facility_Level == level)
                                  & (real_usage_df.Year == y)
                                  & (real_usage_df.Appt_Type == 'MentalAll'), 'Usage'] * 100 / rr.loc[level, y]
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
    # For the details of adjustment of real usage data, refer to Paper
    # "The Changes in Health Service Utilisation in Malawi during the COVID-19 Pandemic"
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

    # adjust annual MentalAll usage using annual reporting rates if needed,
    # for now do not adjust it considering very low reporting rates of Mental Health report
    # and better match with Model usage
    # if adjusted:
    #     real_usage = adjust_real_usage_on_mentalall(real_usage)

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

    # group levels 1b and 2 into 2
    annual_usage_by_level['Facility_Level'] = annual_usage_by_level['Facility_Level'].replace({'1b': '2'})
    annual_usage_by_level = annual_usage_by_level.groupby(
        ['Year', 'Appt_Type', 'Facility_Level'])['Usage'].sum().reset_index()

    # prepare annual usage by level with mean, 97.5% percentile, and 2.5% percentile
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

    # prepare annual usage at all levels with mean, 97.5% percentile, and 2.5% percentile
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

    make_graph_file_name = lambda stub: output_folder / f"{PREFIX_ON_FILENAME}_{stub}.png"  # noqa: E731

    # Plot Simulation vs Real usage (Across all levels and At each level) (trimmed to 0.1 and 10)
    # format plot
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
        _fig.savefig(make_graph_file_name(_name_of_plot.replace(',', '').replace('\n', '_').replace(' ', '_')))
        _fig.show()
        plt.close(_fig)

    # get average annual usage by level for Simulation and Real
    simulation_usage = get_simulation_usage_by_level(results_folder)

    real_usage = get_real_usage(resourcefilepath)[0]

    # find appts that are not included in both simulation and real usage dataframe
    appts_real_only = set(real_usage.columns.values) - set(simulation_usage.columns.values)
    appts_simulation_only = set(simulation_usage.columns.values) - set(real_usage.columns.values)

    # format data
    rel_diff_all_levels = (
        simulation_usage.sum(axis=0) / real_usage.sum(axis=0)
    ).clip(lower=0.1, upper=10.0)

    # plot for all levels
    name_of_plot = 'Model vs Data usage per appt type at all facility levels' \
                   '\n[Model average annual, Adjusted Data average annual]'
    fig, ax = plt.subplots()
    ax.stem(rel_diff_all_levels.index, rel_diff_all_levels.values, bottom=1.0, label='All levels')
    for idx in rel_diff_all_levels.index:
        if not pd.isna(rel_diff_all_levels[idx]):
            ax.text(idx, rel_diff_all_levels[idx]*(1+0.2), round(rel_diff_all_levels[idx], 1),
                    ha='left', fontsize=8)
    format_and_save(fig, ax, name_of_plot)

    # plot for each level
    rel_diff_by_levels = (
        simulation_usage / real_usage
    ).clip(upper=10, lower=0.1).dropna(how='all', axis=0)

    name_of_plot = 'Model vs Data usage per appt type per facility level' \
                   '\n[Model average annual, Adjusted Data average annual]'
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

    # Plot two stacked bars for Model and Data to compare the usage of overall and individual appts
    # format data
    real_usage_all = real_usage.sum(axis=0).reset_index().rename(columns={0: 'Data'})
    simulation_usage_all = simulation_usage.sum(axis=0).reset_index().rename(columns={'index': 'Appt_Type', 0: 'Model'})
    usage_all = simulation_usage_all.merge(real_usage_all, on='Appt_Type', how='inner').melt(
        id_vars='Appt_Type', value_vars=['Model', 'Data'], var_name='Type', value_name='Value').pivot(
        index='Type', columns='Appt_Type', values='Value')
    usage_all = usage_all / 1e6

    # plot
    name_of_plot = 'Model vs Data on average annual health service volume'
    fig, ax = plt.subplots()
    usage_all.plot(kind='bar', stacked=True, color=appt_color_dict, rot=0, ax=ax)
    ax.set_ylabel('Health service volume in millions')
    ax.set(xlabel=None)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title='Appointment type', fontsize=9)
    plt.title(name_of_plot)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(',', '').replace('\n', '_').replace(' ', '_')))
    plt.show()

    # Plot Simulation with 95% CI vs Adjusted Real usage by appt type, across all levels (trimmed to 0.1 and 10)
    # format data
    def format_rel_diff(adjusted=True):
        def format_real_usage():
            _real_usage = get_real_usage(resourcefilepath, adjusted)[0]
            _real_usage_all_levels = _real_usage.sum(axis=0).reset_index().rename(
                columns={0: 'real_usage_all_levels', 'Appt_Type': 'appt_type'})
            return _real_usage_all_levels

        simulation_usage_all_levels_with_ci = get_simulation_usage_with_confidence_interval(results_folder)
        _rel_diff = simulation_usage_all_levels_with_ci.merge(format_real_usage(), on='appt_type', how='outer')

        _rel_diff['ratio'] = (_rel_diff['value'] / _rel_diff['real_usage_all_levels'])

        _rel_diff = _rel_diff[['appt_type', 'value_type', 'ratio']].pivot(
            index='appt_type', columns='value_type', values='ratio').dropna(axis=1, how='all')

        _rel_diff['lower_error'] = (_rel_diff['mean'] - _rel_diff['lower'])
        _rel_diff['upper_error'] = (_rel_diff['upper'] - _rel_diff['mean'])
        _asymmetric_error = [_rel_diff['lower_error'].values, _rel_diff['upper_error'].values]

        _rel_diff = pd.DataFrame(_rel_diff['mean'])

        return _rel_diff, _asymmetric_error

    rel_diff_real, err_real = format_rel_diff(adjusted=True)

    # plot
    name_of_plot = 'Model vs Data usage per appointment type at all facility levels' \
                   '\n[Model average annual 95% CI, Adjusted Data average annual]'
    fig, ax = plt.subplots()
    ax.errorbar(rel_diff_real.index.values,
                rel_diff_real['mean'].values,
                err_real, fmt='.', capsize=3.0, label='All levels')
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    for idx in rel_diff_real.index:
        if not pd.isna(rel_diff_real.loc[idx, 'mean']):
            ax.text(idx,
                    rel_diff_real.loc[idx, 'mean'] * (1 + 0.2),
                    round(rel_diff_real.loc[idx, 'mean'], 1),
                    ha='left', fontsize=8)
    ax.axhline(1.0, color='r')
    format_and_save(fig, ax, name_of_plot)

    # Plot Simulation vs Real usage by appt type and show fraction of usage at each level
    # Model, Adjusted real and Unadjusted real average annual usage all normalised to 1
    # format data
    def format_simulation_usage_fraction():
        _usage = get_simulation_usage_by_level(results_folder)
        _usage_all_levels = _usage.sum(axis=0).reset_index().rename(
            columns={0: '_usage_all_levels', 'index': 'appt_type'})

        _usage = pd.melt(_usage.reset_index(), id_vars='index',
                         var_name='appt_type', value_name='_usage'
                         ).rename(columns={'index': 'facility_level'})

        _usage_fraction = _usage.merge(_usage_all_levels, on='appt_type', how='outer')
        _usage_fraction['ratio'] = (_usage_fraction['_usage'] /
                                    _usage_fraction['_usage_all_levels'])

        _usage_fraction = pd.pivot(_usage_fraction, index='appt_type', columns='facility_level', values='ratio')

        # add nan rows of appts_real_only
        nan_df = pd.DataFrame(index=appts_real_only, columns=_usage_fraction.columns)
        _usage_fraction = pd.concat([_usage_fraction, nan_df]).sort_index()

        # make row of appts_simulation_only nan
        _usage_fraction.loc[_usage_fraction.index.isin(appts_simulation_only), :] = np.NaN

        return _usage_fraction

    def format_real_usage_fraction(adjusted=True):
        _usage = get_real_usage(resourcefilepath, adjusted)[0]
        _usage_all_levels = _usage.sum(axis=0).reset_index().rename(
            columns={0: '_usage_all_levels', 'Appt_Type': 'appt_type'})

        _usage = pd.melt(_usage.reset_index(), id_vars='Facility_Level',
                         var_name='appt_type', value_name='_usage'
                         ).rename(columns={'Facility_Level': 'facility_level'})

        _usage_fraction = _usage.merge(_usage_all_levels, on='appt_type', how='outer')
        _usage_fraction['ratio'] = (_usage_fraction['_usage'] /
                                    _usage_fraction['_usage_all_levels'])

        _usage_fraction = pd.pivot(_usage_fraction, index='appt_type', columns='facility_level', values='ratio')

        # add nan rows of appts_simulation_only
        nan_df = pd.DataFrame(index=appts_simulation_only, columns=_usage_fraction.columns)
        _usage_fraction = pd.concat([_usage_fraction, nan_df]).sort_index()

        # make row of appts_real_only nan
        _usage_fraction.loc[_usage_fraction.index.isin(appts_real_only), :] = np.NaN

        return _usage_fraction

    simulation_usage_plot = format_simulation_usage_fraction()
    real_usage_plot = format_real_usage_fraction(adjusted=True)
    unadjusted_real_usage_plot = format_real_usage_fraction(adjusted=False)
    assert simulation_usage_plot.index.equals(real_usage_plot.index)
    assert simulation_usage_plot.index.equals(unadjusted_real_usage_plot.index)

    # plot
    name_of_plot = 'Model vs Data usage per appointment type on fraction per level' \
                   '\n[Model average annual, Adjusted & Unadjusted Data average annual]'
    fig, ax = plt.subplots(figsize=(12, 5))
    simulation_usage_plot.plot(kind='bar', stacked=True, width=0.3,
                               edgecolor='dimgrey', hatch='',
                               ax=ax, position=0)
    real_usage_plot.plot(kind='bar', stacked=True, width=0.25,
                         edgecolor='dimgrey', hatch='.',
                         ax=ax, position=1)
    unadjusted_real_usage_plot.plot(kind='bar', stacked=True, width=0.25,
                                    edgecolor='dimgrey', hatch='//',
                                    ax=ax, position=2)
    ax.set_xlim(right=len(simulation_usage_plot) - 0.45)
    ax.set_ylabel('Usage per level / Usage all levels')
    ax.set_xlabel('Appointment Type')
    ax.set_title(name_of_plot)
    legend_1 = plt.legend(simulation_usage_plot.columns, loc='upper left', bbox_to_anchor=(1.0, 0.5),
                          title='Facility Level')
    patch_simulation = matplotlib.patches.Patch(facecolor='lightgrey', hatch='', edgecolor="dimgrey", label='Model')
    patch_real = matplotlib.patches.Patch(facecolor='lightgrey', hatch='...', edgecolor="dimgrey",
                                          label='Adjusted Data')
    patch_unadjusted_real = matplotlib.patches.Patch(facecolor='lightgrey', hatch='///', edgecolor="dimgrey",
                                                     label='Unadjusted Data')

    plt.legend(handles=[patch_unadjusted_real, patch_real, patch_simulation],
               loc='lower left', bbox_to_anchor=(1.0, 0.6))
    fig.add_artist(legend_1)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(',', '').replace('\n', '_').replace(' ', '_')))
    plt.show()

    # appendix - plot Simulation with 95% CI vs Adjusted & Unadjusted real, across all levels
    def format_data_for_bar_plot(_usage):
        """reduce the model/data ratio by 1.0, for the bar plot that starts from y=1.0 instead of y=0.0."""
        _usage['mean'] = _usage['mean'] - 1.0
        return _usage

    rel_diff_unadjusted_real, err_unadjusted_real = format_rel_diff(adjusted=False)
    rel_diff_unadjusted_real = format_data_for_bar_plot(rel_diff_unadjusted_real)
    rel_diff_real = format_data_for_bar_plot(rel_diff_real)
    assert (rel_diff_unadjusted_real.index == rel_diff_real.index).all()

    name_of_plot = 'Model vs Data usage per appointment type at all facility levels' \
                   '\n[Model average annual 95% CI, Adjusted & Unadjusted Data average annual]'
    fig, ax = plt.subplots(figsize=(8, 5))
    rel_diff_unadjusted_real.plot(kind='bar', yerr=err_unadjusted_real, width=0.4,
                                  ax=ax, position=0, bottom=1.0,
                                  legend=False, color='salmon')
    rel_diff_real.plot(kind='bar', yerr=err_real, width=0.4,
                       ax=ax, position=1, bottom=1.0,
                       legend=False, color='yellowgreen')
    ax.axhline(1.0, color='r')
    ax.set_xlim(right=len(rel_diff_real) - 0.3)
    ax.set_yscale('log')
    ax.set_ylim(1 / 20, 20)
    ax.set_yticks([1 / 10, 1.0, 10])
    ax.set_yticklabels(("<= 1/10", "1.0", ">= 10"))
    ax.set_ylabel('Model / Data')
    ax.set_xlabel('Appointment Type')
    ax.xaxis.grid(True, which='major', linestyle='--')
    ax.yaxis.grid(True, which='both', linestyle='--')
    ax.set_title(name_of_plot)
    patch_real = matplotlib.patches.Patch(facecolor='yellowgreen', label='Adjusted Data')
    patch_unadjusted_real = matplotlib.patches.Patch(facecolor='salmon', label='Unadjusted Data')
    legend = plt.legend(handles=[patch_real, patch_unadjusted_real], loc='center left', bbox_to_anchor=(1.0, 0.5))
    fig.add_artist(legend)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(',', '').replace('\n', '_').replace(' ', '_')))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    apply(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources')
    )
