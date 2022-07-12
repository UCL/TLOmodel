from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tlo import Date
from tlo.analysis.utils import extract_results, summarize, get_scenario_outputs

# Declare period for which the results will be generated (defined inclusively)
TARGET_PERIOD = (Date(2023, 1, 1), Date(2035, 12, 31))


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
      * to match the districts in the DHIS2 data.
    """

    # Get model outputs
    model_output = get_annual_num_appts_by_level(results_folder=results_folder)

    # # Rename some appts to be compared with real usage
    # appt_dict = {'Under5OPD': 'OPD',
    #              'Over5OPD': 'OPD',
    #              'AntenatalFirst': 'AntenatalTotal',
    #              'ANCSubsequent': 'AntenatalTotal',
    #              'NormalDelivery': 'Delivery',
    #              'CompDelivery': 'Delivery',
    #              'EstMedCom': 'EstAdult',
    #              'EstNonCom': 'EstAdult',
    #              'VCTPositive': 'VCTTests',
    #              'VCTNegative': 'VCTTests',
    #              'DentAccidEmerg': 'DentalAll',
    #              'DentSurg': 'DentalAll',
    #              'DentU5': 'DentalAll',
    #              'DentO5': 'DentalAll',
    #              'MentOPD': 'MentalAll',
    #              'MentClinic': 'MentalAll'
    #              }
    # model_output.columns = model_output.columns.map(lambda _name: appt_dict.get(_name, _name))
    return model_output.groupby(axis=1, level=0).sum()



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

    make_graph_file_name = lambda stub: output_folder / f"{stub}.png"  # noqa: E731

    simulation_usage = get_simulation_usage(results_folder)

    # Plot Relative Difference Between Simulation and Real (Across all levels)
    rel_diff_all_levels = (
        simulation_usage.sum(axis=0) / real_usage.sum(axis=0)
    ).clip(lower=0.1, upper=10.0)

    name_of_plot = 'Model vs Real average annual usage by appt type\n[All Facility Levels]'
    fig, ax = plt.subplots()
    ax.stem(rel_diff_all_levels.index, rel_diff_all_levels.values, bottom=1.0, label='All levels')
    format_and_save(fig, ax, name_of_plot)

    # Plot Relative Difference Between Simulation and Real (At each level) (trimmed to 0.1 and 10)
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
    outputspath = Path('./outputs/t.mangal@imperial.ac.uk')
    rfp = Path('./resources')

    # Find results folder (most recent run generated using that scenario_filename)
    scenario_filename = 'scenario0.py'
    results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]

    # Run all the calibrations
    apply(results_folder=results_folder, output_folder=results_folder, resourcefilepath=rfp)
