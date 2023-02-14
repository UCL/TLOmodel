from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tlo import Date
from tlo.analysis.utils import extract_results, get_scenario_outputs, summarize

PREFIX_ON_FILENAME = '4'

# Declare period for which the results will be generated (defined inclusively)
TARGET_PERIOD = (Date(2015, 1, 1), Date(2019, 12, 31))


def get_annual_num_appts_by_level(results_folder: Path) -> pd.DataFrame:
    """Return pd.DataFrame gives the (mean) simulated annual number of appointments of each type at each level."""

    def get_counts_of_appts(_df):
        """Get the mean number of appointments of each type being used each year at each level."""

        def unpack_nested_dict_in_series(_raw: pd.Series):
            return pd.concat(
                {
                  idx: pd.DataFrame.from_dict(mydict) for idx, mydict in _raw.iteritems()
                 }
             ).unstack().fillna(0.0).astype(int)

        return _df \
            .loc[pd.to_datetime(_df['date']).between(*TARGET_PERIOD), 'Number_By_Appt_Type_Code_And_Level'] \
            .pipe(unpack_nested_dict_in_series) \
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


def get_simulation_usage(results_folder: Path) -> pd.DataFrame:
    """Returns the simulated MEAN USAGE PER YEAR DURING THE TIME_PERIOD, by appointment type and level.
    With reformatting ...
      * to match standardized categories from the DHIS2 data,
      * to match the districts in the DHIS2 data.  (see commits in PR616)
    """

    # Get model outputs
    model_output = get_annual_num_appts_by_level(results_folder=results_folder)

    # Rename some appts to be compared with real usage
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
    model_output.columns = model_output.columns.map(lambda _name: appt_dict.get(_name, _name))
    return model_output.groupby(axis=1, level=0).sum()


def get_real_usage(resourcefilepath) -> pd.DataFrame:
    """Returns the real data on the MEAN USAGE PER YEAR DURING THE TIME_PERIOD for each appointment at each level."""

    # add facility level and district columns to both real and simulation usage
    mfl = pd.read_csv(
        resourcefilepath / 'healthsystem' / 'organisation' / 'ResourceFile_Master_Facilities_List.csv')

    # Get real usage data
    real_usage = pd.read_csv(
        resourcefilepath / 'healthsystem' / 'real_appt_usage_data' / 'real_monthly_usage_of_appt_type.csv')

    # get facility_level for each record
    real_usage = real_usage.merge(mfl[['Facility_ID', 'Facility_Level']], left_on='Facility_ID', right_on='Facility_ID')

    # assign date to each record
    real_usage['date'] = pd.to_datetime({'year': real_usage['Year'], 'month': real_usage['Month'], 'day': 1})

    # Produce table of the AVERAGE NUMBER PER YEAR DURING THE TIME_PERIOD of appointment type by level
    # limit to date
    totals_by_year = real_usage \
        .loc[real_usage['date'].between(*TARGET_PERIOD)] \
        .groupby(['Year', 'Appt_Type', 'Facility_Level'])['Usage'].sum()
    annual_mean = totals_by_year.groupby(level=[1, 2]).mean().unstack()

    # Combine the TB data [which is yearly] (after dropping period outside 2017-2019 according to data consistency
    # and pandemic) with the rest of the data.
    real_usage_TB = pd.read_csv(
        resourcefilepath / 'healthsystem' / 'real_appt_usage_data' / 'real_yearly_usage_of_TBNotifiedAll.csv')
    real_usage_TB = real_usage_TB.loc[real_usage_TB['Year'].isin([2017, 2018, 2019])]
    real_usage_TB = real_usage_TB.merge(mfl[['Facility_ID', 'Facility_Level']],
                                        left_on='Facility_ID', right_on='Facility_ID')
    totals_by_year_TB = real_usage_TB.groupby(['Year', 'Appt_Type', 'Facility_Level'])['Usage'].sum()
    annual_mean_TB = totals_by_year_TB.groupby(level=[1, 2]).mean().unstack()

    return pd.concat([annual_mean, annual_mean_TB], axis=0).T


def adjust_real_usage_on_mentalall(real_usage_df) -> pd.DataFrame:
    """This is to adjust the average annual MentalAll usage in real usage dataframe that is output by get_real_usage.
    The MentalAll usage was not adjusted in the preprocessing stage considering individual facilities and very low
    reporting rates.
    We now directly adjust its annual usage by facility level using the aggregated average annual reporting rates by
    facility level. The latter is calculated based on DHIS2 Mental Health Report reporting rates."""
    # the average annual reporting rates for Mental Health Report by facility level (%), 2015-2019
    # could turn the reporting rates data into a ResourceFile if necessary
    rr = {'0': None, '1a': 70.93, '1b': 31.25, '2': 46.78, '3': 47.50, '4': None}
    rr_df = pd.DataFrame.from_dict(rr, orient='index').rename(columns={0: 'avg_annual_rr'})
    # make the adjustment assuming 100% reporting rates
    real_usage_df.loc[['1a', '1b', '2', '3'], 'MentalAll'] = (
        real_usage_df.loc[['1a', '1b', '2', '3'], 'MentalAll'] * 100 /
        rr_df.loc[['1a', '1b', '2', '3'], 'avg_annual_rr'])

    return real_usage_df


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

    simulation_usage = get_simulation_usage(results_folder)

    real_usage = get_real_usage(resourcefilepath)
    real_usage = adjust_real_usage_on_mentalall(real_usage)

    # Plot Simulation vs Real usage (Across all levels) (trimmed to 0.1 and 10)
    rel_diff_all_levels = (
        simulation_usage.sum(axis=0) / real_usage.sum(axis=0)
    ).clip(lower=0.1, upper=10.0)

    name_of_plot = 'Model vs Real average annual usage by appt type\n[All Facility Levels]'
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

    name_of_plot = 'Model vs Real average annual usage by appt type\n[By Facility Level]'
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
