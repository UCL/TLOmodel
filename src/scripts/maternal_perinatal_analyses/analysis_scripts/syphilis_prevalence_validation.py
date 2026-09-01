import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo.analysis.utils import get_scenario_outputs, parse_log_file


LOGGER_NAME = 'tlo.analysis.syphilis_prevalence_validation'
LOG_KEY = 'maternal_syphilis_prevalence'
SCENARIO_FILENAME = 'syphilis_baseline_validation_scenario.py'
STAGES = ['primary', 'secondary', 'early_latent', 'late_latent']


def _sort_key(path):
    return (0, int(path.name)) if path.name.isdigit() else (1, path.name)


def get_latest_results_folder(outputs_dir):
    scenario_outputs = get_scenario_outputs(SCENARIO_FILENAME, outputs_dir)
    if not scenario_outputs:
        raise FileNotFoundError(
            f'No outputs found for {SCENARIO_FILENAME} under {outputs_dir}. '
            'Run the validation scenario first.'
        )
    return scenario_outputs[-1]


def iter_run_folders(results_folder):
    if list(results_folder.glob('*.log')) or (results_folder / f'{LOGGER_NAME}.pickle').exists():
        yield 0, 0, results_folder
        return

    draw_dirs = sorted(
        [path for path in results_folder.iterdir() if path.is_dir()],
        key=_sort_key
    )

    for draw_dir in draw_dirs:
        run_dirs = sorted(
            [path for path in draw_dir.iterdir() if path.is_dir()],
            key=_sort_key
        )
        for run_dir in run_dirs:
            yield int(draw_dir.name), int(run_dir.name), run_dir


def load_prevalence_log_for_run(run_folder):
    pickle_path = run_folder / f'{LOGGER_NAME}.pickle'
    if pickle_path.exists():
        with pickle_path.open('rb') as file:
            parsed_module_log = pickle.load(file)
        return parsed_module_log[LOG_KEY].copy()

    log_files = list(run_folder.glob('*.log'))
    if len(log_files) != 1:
        raise FileNotFoundError(
            f'Expected exactly one .log file in {run_folder}, found {len(log_files)}.'
        )

    parsed_log = parse_log_file(log_files[0])
    try:
        return parsed_log[LOGGER_NAME][LOG_KEY].copy()
    except KeyError as error:
        raise KeyError(
            f'{LOGGER_NAME}/{LOG_KEY} was not found in {log_files[0]}. '
            'Check that the run used syphilis_baseline_validation_scenario.py.'
        ) from error


def load_prevalence_logs(results_folder):
    frames = []
    for draw, run, run_folder in iter_run_folders(results_folder):
        frame = load_prevalence_log_for_run(run_folder)
        if 'date' not in frame.columns:
            frame = frame.reset_index()
        frame['date'] = pd.to_datetime(frame['date'])
        frame['draw'] = draw
        frame['run'] = run
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f'No draw/run folders found under {results_folder}.')

    return pd.concat(frames, ignore_index=True)


def prepare_monthly_prevalence(prevalence_log, start_year):
    monthly = prevalence_log.loc[prevalence_log['date'].dt.year >= start_year].copy()
    if monthly.empty:
        raise ValueError(f'No maternal syphilis prevalence records found from {start_year} onward.')

    denominator = monthly['number_pregnant'].replace(0, np.nan)
    for stage in STAGES:
        monthly[f'{stage}_prevalence_percent'] = (monthly[stage] / denominator) * 100

    monthly['year'] = monthly['date'].dt.year
    monthly['month'] = monthly['date'].dt.month
    return monthly


def annualise_monthly_prevalence(monthly):
    value_columns = [
        'number_pregnant',
        'number_active_syphilis_pregnant',
        'prevalence_percent',
        'active_syphilis_treated',
        'treated_cured_pregnant',
        *[f'{stage}_prevalence_percent' for stage in STAGES],
    ]

    return (
        monthly
        .groupby(['year', 'draw', 'run'], as_index=False)[value_columns]
        .mean()
    )


def summarise_across_runs(annual_by_run):
    summary_rows = []
    prevalence_columns = [
        'prevalence_percent',
        *[f'{stage}_prevalence_percent' for stage in STAGES],
    ]

    for year, group in annual_by_run.groupby('year'):
        row = {'year': year}
        for column in prevalence_columns:
            values = group[column].dropna()
            if values.empty:
                row[f'{column}_mean'] = np.nan
                row[f'{column}_lower'] = np.nan
                row[f'{column}_upper'] = np.nan
            else:
                row[f'{column}_mean'] = values.mean()
                row[f'{column}_lower'] = values.quantile(0.025)
                row[f'{column}_upper'] = values.quantile(0.975)

        for column in [
            'number_pregnant',
            'number_active_syphilis_pregnant',
            'active_syphilis_treated',
            'treated_cured_pregnant',
        ]:
            row[f'{column}_mean'] = group[column].mean()

        summary_rows.append(row)

    return pd.DataFrame(summary_rows).sort_values('year')


def plot_total_prevalence(summary, output_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    years = summary['year'].to_numpy()
    mean = summary['prevalence_percent_mean'].to_numpy(dtype=float)
    lower = summary['prevalence_percent_lower'].to_numpy(dtype=float)
    upper = summary['prevalence_percent_upper'].to_numpy(dtype=float)

    ax.plot(
        years,
        mean,
        color='#2166ac',
        marker='o',
        label='Mean'
    )
    ax.fill_between(
        years,
        lower,
        upper,
        color='#67a9cf',
        alpha=0.25,
        label='2.5-97.5 percentile across runs'
    )
    ax.set_title('Maternal syphilis prevalence among pregnant women')
    ax.set_xlabel('Year')
    ax.set_ylabel('Prevalence (%)')
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'maternal_syphilis_prevalence_2020_onward.png', dpi=300)
    plt.close(fig)


def plot_stage_prevalence(summary, output_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    years = summary['year'].to_numpy()
    colours = {
        'primary': '#b2182b',
        'secondary': '#ef8a62',
        'early_latent': '#4d9221',
        'late_latent': '#276419',
    }

    for stage in STAGES:
        ax.plot(
            years,
            summary[f'{stage}_prevalence_percent_mean'].to_numpy(dtype=float),
            marker='o',
            color=colours[stage],
            label=stage.replace('_', ' ')
        )

    ax.set_title('Maternal syphilis prevalence by infection stage')
    ax.set_xlabel('Year')
    ax.set_ylabel('Prevalence among pregnant women (%)')
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'maternal_syphilis_stage_prevalence_2020_onward.png', dpi=300)
    plt.close(fig)


def run_analysis(results_folder=None, outputs_dir=Path('./outputs'), start_year=2020, output_dir=None):
    outputs_dir = Path(outputs_dir)
    results_folder = Path(results_folder) if results_folder else get_latest_results_folder(outputs_dir)
    output_dir = Path(output_dir) if output_dir else results_folder / 'syphilis_prevalence_validation_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    prevalence_log = load_prevalence_logs(results_folder)
    monthly = prepare_monthly_prevalence(prevalence_log, start_year=start_year)
    annual_by_run = annualise_monthly_prevalence(monthly)
    annual_summary = summarise_across_runs(annual_by_run)

    monthly.to_csv(output_dir / 'maternal_syphilis_prevalence_monthly.csv', index=False)
    annual_by_run.to_csv(output_dir / 'maternal_syphilis_prevalence_annual_by_run.csv', index=False)
    annual_summary.to_csv(output_dir / 'maternal_syphilis_prevalence_annual_summary.csv', index=False)

    plot_total_prevalence(annual_summary, output_dir)
    plot_stage_prevalence(annual_summary, output_dir)

    print(f'Results folder: {results_folder}')
    print(f'Analysis outputs: {output_dir}')
    print(annual_summary[['year', 'prevalence_percent_mean']])


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot maternal syphilis prevalence among pregnant women from scenario outputs.'
    )
    parser.add_argument(
        '--results-folder',
        type=Path,
        default=None,
        help='Specific scenario results folder. If omitted, the latest validation scenario output is used.'
    )
    parser.add_argument(
        '--outputs-dir',
        type=Path,
        default=Path('./outputs'),
        help='Directory containing TLO scenario output folders.'
    )
    parser.add_argument(
        '--start-year',
        type=int,
        default=2020,
        help='First calendar year to include in the prevalence analysis.'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Directory to write CSV and PNG outputs. Defaults to a subfolder of the results folder.'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_analysis(
        results_folder=args.results_folder,
        outputs_dir=args.outputs_dir,
        start_year=args.start_year,
        output_dir=args.output_dir,
    )
