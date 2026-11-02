"""Task 1: HPV time-step sensitivity / convergence (no recalibration).
Run from the TLO project root, where ./resources is available.
Keep this file separate from the HPV model and from general-purpose plotting.
"""
from pathlib import Path
import inspect
import json
import argparse
import pandas as pd
import numpy as np
from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography, enhanced_lifestyle, epi, healthburden,
    healthseekingbehaviour, healthsystem, measles,
    simplified_births, symptommanager, hpv_timestep, hiv, tb,
)

OUTPUT = Path('./outputs/time_step_convergence_b058_age_flat_5seeds')
OUTPUT.mkdir(parents=True, exist_ok=True)
RESOURCE = Path('./resources')
START = Date(2010, 1, 1)
END = Date(2024, 1, 1)
POPULATION = 50000
SEEDS = [12345, 23456, 34567, 45678, 56789]  # Five independent replicates
B_HPV = 0.58  # Fixed previously calibrated value; no recalibration in Task 1
# For replicated Monte Carlo uncertainty, add multiple seeds in a subsequent experiment.
STEPS = {'1_month': 0, '2_weeks': 14, '1_week': 7}
AGE_LABELS = ['15_19', '20_24', '25_34', '35_44', '45_54', '55plus']
# All six HPV genotype groups remain unchanged.
HPV_GROUPS = ['hr1', 'hr2', 'hr3', 'hr4', 'hr5', 'hr6']


def run_one(step_label: str, step_days: int, seed: int) -> pd.DataFrame:
    """Only HPV update frequency changes between runs; logging is monthly."""
    run_dir = OUTPUT / f'seed_{seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    sim = Simulation(
        start_date=START, seed=seed, resourcefilepath=RESOURCE,
        show_progress_bar=False,
        log_config={
            'filename': f'hpv_timestep_{step_label}',
            'directory': run_dir,
            'custom_levels': {'*': logging.WARNING, 'tlo.methods.hpv_timestep': logging.INFO},
        },
    )
    sim.register(
        demography.Demography(), simplified_births.SimplifiedBirths(),
        enhanced_lifestyle.Lifestyle(),
        healthsystem.HealthSystem(
            service_availability=['*'], mode_appt_constraints=1,
            cons_availability='default', ignore_priority=False,
            capabilities_coefficient=1.0,
            use_funded_or_actual_staffing='actual',
            disable=False, disable_and_reject_all=False,
        ),
        symptommanager.SymptomManager(),
        healthseekingbehaviour.HealthSeekingBehaviour(),
        healthburden.HealthBurden(), epi.Epi(), hiv.Hiv(),
        measles.Measles(), tb.Tb(), hpv_timestep.HPV(),
    )
    # Fixed beta and no recalibration: only HPV event frequency changes between scenarios.
    hpv_pars = sim.modules['HPV'].parameters
    hpv_pars['b_hpv'] = B_HPV
    # [NEW] Disable the DIRECT age-acquisition modifier; retain age mixing.
    hpv_pars['hpv_age_rise'] = 0.0
    hpv_pars['hpv_age_decline'] = 0.0
    hpv_pars['hpv_event_frequency_days'] = step_days  # 0=one calendar month; 14/7=days
    hpv_pars['hpv_event_frequency_months'] = 1  # legacy one-calendar-month baseline
    hpv_pars['hpv_logging_frequency_months'] = 1
    # [DIAGNOSTIC] Fail fast if the wrong HPV implementation is imported.
    hpv_instance = sim.modules['HPV']
    loaded_path = Path(inspect.getfile(type(hpv_instance))).resolve()
    print(f'\n[HPV CHECK] seed={seed}, scenario={step_label}, module_file={loaded_path}')
    print(f'[HPV CHECK] b_hpv={hpv_pars["b_hpv"]}, frequency_days={hpv_pars["hpv_event_frequency_days"]}, frequency_months={hpv_pars["hpv_event_frequency_months"]}')
    if loaded_path.stem != 'hpv_timestep':
        raise RuntimeError(f'Wrong HPV module loaded: {loaded_path}; expected hpv_timestep.py')
    assert hpv_pars['hpv_event_frequency_days'] == step_days
    assert np.isclose(float(hpv_pars['b_hpv']), B_HPV)
    # Record the full HPV parameters to verify that only cadence differs.
    parameters_snapshot = {str(k): str(v) for k, v in sorted(hpv_pars.items())}
    with (run_dir / f'parameters_{step_label}.json').open('w', encoding='utf-8') as f:
        json.dump(parameters_snapshot, f, indent=2, ensure_ascii=False)
    # Only event cadence differs; keep all other transmission/clearance parameters unchanged.
    sim.make_initial_population(n=POPULATION)
    sim.simulate(end_date=END)
    log = parse_log_file(sim.log_filepath)
    df = pd.DataFrame(log['tlo.methods.hpv_timestep']['summary']).copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').drop_duplicates('date', keep='last')
    df['step'] = step_label
    df['seed'] = seed
    df.to_csv(run_dir / f'hpv_summary_{step_label}.csv', index=False)
    return df


def prepare_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """A tidy metric table, with monthly new-infection counts and point prevalence."""
    metrics = {'TotalPrev': 'overall_any'}
    metrics.update({f'Any_All_{a}_Prev': f'age_{a}' for a in AGE_LABELS})
    metrics.update({f'Xpert_{p}_Prev': p for p in ['P1', 'P2', 'P3']})
    metrics.update({f'NewInf_{g}': f'newinf_{g}' for g in HPV_GROUPS})
    metrics['NewInf_Total'] = 'newinf_total'
    # NewInf_Total counts genotype-specific acquisition events, not unique people.
    # Counts are accumulated between monthly logs, not necessarily calendar-month totals.
    # The first log can precede any HPV update, so event-count columns may be absent.
    # Fill only missing NewInf columns with zero; prevalence columns must exist.
    for col in [c for c in metrics if c.startswith('NewInf_')]:
        if col not in df.columns:
            df[col] = 0
    missing = [c for c in metrics if c not in df.columns]
    if missing:
        raise KeyError(f'Missing HPV summary columns: {missing}')
    tidy = df[['date', 'step', 'seed', *metrics]].rename(columns=metrics).melt(
        id_vars=['date', 'step', 'seed'], var_name='metric', value_name='value'
    )
    tidy['value'] = pd.to_numeric(tidy['value'], errors='coerce')
    tidy['metric_kind'] = np.where(tidy['metric'].str.startswith('newinf_'), 'count', 'prevalence')
    return tidy


def make_comparison(tidy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match identical logging dates and compare to the finest (7-day) version."""
    wide = tidy.pivot_table(index=['date', 'metric', 'metric_kind', 'seed'], columns='step',
                            values='value', aggfunc='first').reset_index()
    required = list(STEPS)
    if any(c not in wide for c in required):
        raise ValueError('Not all three run results were found.')
    common = wide.dropna(subset=required).copy()
    # Absolute pp for prevalence; raw number differences for event counts.
    for coarse in ['1_month', '2_weeks']:
        common[f'{coarse}_minus_1_week'] = common[coarse] - common['1_week']
        common[f'{coarse}_abs_diff'] = common[f'{coarse}_minus_1_week'].abs()
        common[f'{coarse}_relative_diff_pct'] = np.where(
            common['1_week'].abs() > 0,
            100 * common[f'{coarse}_abs_diff'] / common['1_week'].abs(),
            np.nan,
        )
    rows = []
    for metric, grp in common.groupby('metric', sort=False):
        grp = grp.sort_values('date')
        kind = grp['metric_kind'].iloc[0]
        for coarse in ['1_month', '2_weeks']:
            diffs = grp[f'{coarse}_abs_diff']
            rows.append({
                'metric': metric, 'kind': kind,
                'comparison': f'{coarse} vs 1_week',
                'months_compared': len(grp),
                'MAE_pp_or_count': diffs.mean() * (100 if kind == 'prevalence' else 1),
                'max_abs_diff_pp_or_count': diffs.max() * (100 if kind == 'prevalence' else 1),
                'final_1_month_or_2_weeks': grp[coarse].iloc[-1],
                'final_1_week': grp['1_week'].iloc[-1],
                'final_diff_pp_or_count': (grp[coarse].iloc[-1] - grp['1_week'].iloc[-1])
                                         * (100 if kind == 'prevalence' else 1),
            })
    return common, pd.DataFrame(rows)


def run_task(task_id: int):
    """One SLURM array task = one independent seed x time-step simulation."""
    if not 0 <= task_id < len(SEEDS) * len(STEPS):
        raise ValueError(f'Invalid task_id: {task_id}')
    seed_index, step_index = divmod(task_id, len(STEPS))
    seed = SEEDS[seed_index]
    step_label = list(STEPS)[step_index]
    step_days = STEPS[step_label]
    print(f'[TASK {task_id}] seed={seed}, step={step_label}', flush=True)
    frame = run_one(step_label, step_days, seed)
    run_dir = OUTPUT / f'seed_{seed}'
    prepare_metrics(frame).to_csv(run_dir / f'timestep_series_{step_label}.csv', index=False)
    print(f'[DONE] seed={seed}, step={step_label}', flush=True)


def aggregate():
    """Combine results only after all 15 HPC array tasks finish successfully."""
    frames = []
    per_seed_summary = []
    ignored = {'hpv_event_frequency_days', 'hpv_event_frequency_months'}
    for seed in SEEDS:
        run_dir = OUTPUT / f'seed_{seed}'
        snapshots = []
        seed_frames = []
        for label in STEPS:
            series_path = run_dir / f'timestep_series_{label}.csv'
            snapshot_path = run_dir / f'parameters_{label}.json'
            if not series_path.is_file() or not snapshot_path.is_file():
                raise FileNotFoundError(f'Missing result for seed={seed}, step={label}')
            piece = pd.read_csv(series_path, parse_dates=['date'])
            if not ((piece['seed'] == seed) & (piece['step'] == label)).all():
                raise ValueError(f'Incorrect seed/step identifiers: {series_path}')
            seed_frames.append(piece)
            with snapshot_path.open(encoding='utf-8') as f:
                snapshots.append({k: v for k, v in json.load(f).items() if k not in ignored})
        if any(snapshot != snapshots[0] for snapshot in snapshots[1:]):
            raise RuntimeError(f'HPV parameter mismatch within seed={seed}')
        if not all(float(snapshot['hpv_age_rise']) == float(snapshot['hpv_age_decline']) == 0
                   for snapshot in snapshots):
            raise RuntimeError(f'Age modifier was not disabled in seed={seed}')
        seed_tidy = pd.concat(seed_frames, ignore_index=True)
        seed_tidy.to_csv(run_dir / 'timestep_series_long.csv', index=False)
        matched, summary = make_comparison(seed_tidy)
        matched.to_csv(run_dir / 'timestep_matched_dates.csv', index=False)
        summary.insert(0, 'seed', seed)
        summary.to_csv(run_dir / 'timestep_convergence_table.csv', index=False)
        frames.append(seed_tidy)
        per_seed_summary.append(summary)

    all_series = pd.concat(frames, ignore_index=True)
    if all_series.duplicated(['seed', 'step', 'date', 'metric']).any():
        raise ValueError('Duplicate seed/step/date/metric rows')
    all_series.to_csv(OUTPUT / 'timestep_series_long_all_seeds.csv', index=False)
    pd.concat(per_seed_summary, ignore_index=True).to_csv(
        OUTPUT / 'timestep_convergence_per_seed.csv', index=False)
    stats = (all_series.groupby(['date', 'metric', 'metric_kind', 'step'], as_index=False)
             .agg(mean=('value', 'mean'), sd=('value', 'std'), n=('value', 'count')))
    if (stats['n'] != len(SEEDS)).any():
        raise ValueError('Missing replicate at a date/metric/step')
    stats.to_csv(OUTPUT / 'timestep_series_mean_sd.csv', index=False)
    mean_tidy = stats.rename(columns={'mean': 'value'})[
        ['date', 'metric', 'metric_kind', 'step', 'value']]
    mean_wide = mean_tidy.pivot(index=['date', 'metric', 'metric_kind'],
                                columns='step', values='value').reset_index()
    mean_wide.dropna(subset=list(STEPS)).to_csv(
        OUTPUT / 'timestep_matched_mean.csv', index=False)
    _, mean_summary = make_comparison(mean_tidy.assign(seed=-1))
    mean_summary.to_csv(OUTPUT / 'timestep_convergence_mean.csv', index=False)
    print('[DONE] 15 runs aggregated:', OUTPUT.resolve())


def main():
    parser = argparse.ArgumentParser()
    choose = parser.add_mutually_exclusive_group(required=True)
    choose.add_argument('--task-id', type=int, choices=range(len(SEEDS) * len(STEPS)),
                        metavar='0..14', help='SLURM array task ID (0 through 14)')
    choose.add_argument('--aggregate', action='store_true',
                        help='Combine all outputs after all 15 jobs complete')
    args = parser.parse_args()
    if args.aggregate:
        aggregate()
    else:
        run_task(args.task_id)


if __name__ == '__main__':
    main()
