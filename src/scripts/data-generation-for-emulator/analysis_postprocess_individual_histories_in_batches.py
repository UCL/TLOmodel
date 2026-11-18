"""Combine the postprocessed individual histories of several batch submissions of
scenario_data_generation_for_emulator.py into one dataset, and optionally log it as a single wandb artifact.

Each batch is a separate TLO job with its own results folder. Within a job TLO numbers the draws
0 .. BATCH_SIZE-1 (local draw number); the scenario records each draw's position in the full Sobol design
as its draw name, 'global_draw_<N>'. This script:
  1. postprocesses each batch with the standard postprocessing (analysis_postprocess_individual_histories.py),
  2. replaces the local draw number in the 'draw' column with the global draw number,
  3. checks that the batches fit together (no draw twice, same varied parameters, same fixed parameters),
  4. concatenates them and (optionally) logs one wandb artifact whose metadata lists every batch.

Usage:
    1. List the job IDs to combine in JOB_IDS below (and set RESULTS_ROOT to the folder containing them).
       The batch number is read from each job ID, e.g. 'track_individual_histories_batch3-2026-11-18T120000Z'.
    2. python analysis_postprocess_individual_histories_in_batches.py [--output-path DIR] [--log-to-wandb]
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import wandb

# Re-use the standard postprocessing so the two scripts stay in sync
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_postprocess_individual_histories as base  # noqa: E402

# ======================================================================================================
# EDIT HERE: folder containing the batch results, and the job IDs (folder names) of the batches to combine.
# Order does not matter; the batch number is read from each job ID.
RESULTS_ROOT = Path('outputs')
JOB_IDS = [
    # 'track_individual_histories_batch0-2026-11-18T100000Z',
    # 'track_individual_histories_batch1-2026-11-18T110000Z',
]
# ======================================================================================================

GLOBAL_DRAW_NAME = re.compile(r'^global_draw_(\d+)$')
BATCH_IN_JOB_ID = re.compile(r'_batch(\d+)-')


def get_batch_number(job_ID):
    """Batch number from a job ID such as 'track_individual_histories_batch3-<timestamp>'."""
    match = BATCH_IN_JOB_ID.search(job_ID)
    if match is None:
        print(f"ERROR: cannot find a batch number ('_batch<N>-') in job ID '{job_ID}'")
        sys.exit(1)
    return int(match.group(1))


def get_batches_to_process():
    """(batch_number, results_folder) for every job in JOB_IDS, sorted by batch number."""
    if len(JOB_IDS) == 0:
        print("ERROR: JOB_IDS is empty - list the job IDs of the batches to combine at the top of this file")
        sys.exit(1)

    batches = {}
    for job_ID in JOB_IDS:
        batch_number = get_batch_number(job_ID)
        if batch_number in batches:
            print(f"ERROR: batch {batch_number} is listed twice: '{batches[batch_number].name}' and '{job_ID}'")
            sys.exit(1)
        results_folder = RESULTS_ROOT / job_ID
        if not results_folder.is_dir():
            print(f"ERROR: results folder not found: '{results_folder}'")
            sys.exit(1)
        batches[batch_number] = results_folder

    print(f"Batches to combine: {sorted(batches)}")
    return sorted(batches.items())


def get_global_draw_numbers(draws, results_folder):
    """Global draw number of each draw in a batch, read from its draw_name ('global_draw_<N>')."""
    global_draws = []
    for d in draws:
        match = GLOBAL_DRAW_NAME.match(str(d.get('draw_name', '')))
        if match is None:
            print(f"ERROR: in '{results_folder}', draw {d.get('draw_number')} has draw_name "
                  f"{d.get('draw_name')!r}, expected 'global_draw_<N>'. "
                  f"Was this job submitted with the batched scenario file?")
            sys.exit(1)
        global_draws.append(int(match.group(1)))
    return global_draws


def as_comparable(obj):
    """String form of a (nested) parameter dict, for checking equality across batches."""
    return json.dumps(obj, sort_keys=True, default=str)


def process_batch(batch_number: int, results_folder: Path):
    """Postprocess one batch; return its dataset (with global draw numbers) and its metadata."""
    print(f"\n=== Processing batch {batch_number} in {results_folder}")

    scenario_json_data = base.retrieve_scenario_json_file(results_folder)
    draws_parameters = scenario_json_data['draws']
    global_draws = get_global_draw_numbers(draws_parameters, results_folder)
    job_ID = results_folder.resolve().name

    # Check the batch number in the job ID matches the draws the job actually ran
    # (batch b of size n runs global draws b*n .. b*n + n - 1)
    n = len(draws_parameters)
    expected = list(range(batch_number * n, (batch_number + 1) * n))
    if global_draws != expected:
        print(f"ERROR: job ID '{job_ID}' says batch {batch_number}, which should contain global draws "
              f"{expected[0]}-{expected[-1]}, but its draws file has {min(global_draws)}-{max(global_draws)}")
        sys.exit(1)

    # Standard postprocessing. NB: it numbers draws by their position in this job (local draw number),
    # and it modifies draws_parameters in place so that only parameters common to all draws remain.
    individual_histories = base.extract_individual_histories(results_folder)
    df = base.postprocess_individual_histories(individual_histories, draws_parameters)

    if df.empty:
        print(f"WARNING: no completed episodes found in batch {job_ID}")
    else:
        local_to_global = dict(enumerate(global_draws))
        df['local_draw'] = df['draw']
        df['draw'] = df['draw'].map(local_to_global)
        df['batch_job_ID'] = job_ID

    batch_metadata = {
        'job_ID': job_ID,
        'batch_number': batch_number,
        'scenario_script_path': scenario_json_data['scenario_script_path'],
        'scenario_script_commit_hash': scenario_json_data.get('commit'),
        'scenario_seed': scenario_json_data.get('scenario_seed'),
        # The batched scenario offsets the seed by the runs in earlier batches; undo that to recover the
        # top-level seed, which should be the same for every batch
        'base_seed': scenario_json_data['scenario_seed'] - min(global_draws) * scenario_json_data['runs_per_draw'],
        'global_draws': global_draws,
    }
    # After postprocessing, every draw holds only the parameters common to all draws in this batch
    common_parameters = draws_parameters[0]['parameters']
    return df, batch_metadata, common_parameters


def check_batches_fit_together(datasets, batches_metadata, common_parameters_per_batch):
    """Stop if batches overlap or were generated with different fixed/varied parameters; warn on softer issues."""
    # 1. No global draw may appear in two batches (e.g. the same batch passed twice, or resubmitted)
    seen = {}
    for b in batches_metadata:
        for g in b['global_draws']:
            if g in seen:
                print(f"ERROR: global draw {g} appears in both {seen[g]} and {b['job_ID']}")
                sys.exit(1)
            seen[g] = b['job_ID']

    # 2. Parameters held fixed must be identical in every batch (otherwise a parameter that differs
    #    between batches, but not within one, would be silently missing from the dataset's columns)
    reference = as_comparable(common_parameters_per_batch[0])
    for b, params in zip(batches_metadata, common_parameters_per_batch):
        if as_comparable(params) != reference:
            print(f"ERROR: the parameters held fixed in {b['job_ID']} differ from those in "
                  f"{batches_metadata[0]['job_ID']}. These batches cannot be combined.")
            sys.exit(1)

    # 3. The same parameters must be varied in every batch
    draw_param_cols = [frozenset(c for c in df.columns if c.startswith('draw_parameter_'))
                       for df in datasets if not df.empty]
    if len(set(draw_param_cols)) > 1:
        print("ERROR: batches vary different sets of parameters:")
        for b, cols in zip(batches_metadata, draw_param_cols):
            print(f"  {b['job_ID']}: {sorted(cols)}")
        sys.exit(1)

    # 4. Softer checks: same scenario file, commit and top-level seed (warn only - you may have committed
    #    unrelated changes between submissions)
    for key in ('scenario_script_path', 'scenario_script_commit_hash', 'base_seed'):
        values = {b[key] for b in batches_metadata}
        if len(values) > 1:
            print(f"WARNING: batches differ in {key}: {sorted(map(str, values))}. "
                  f"Check that model code relevant to the emulator did not change between submissions.")

    # 5. Report which global draws are missing (e.g. batches not yet run)
    all_draws = sorted(seen)
    missing = sorted(set(range(all_draws[-1] + 1)) - set(all_draws))
    if missing:
        print(f"NOTE: {len(missing)} global draws below {all_draws[-1]} are not included: {missing}")
    return all_draws, missing


def apply(output_folder: Path, log_to_wandb):

    batches = get_batches_to_process()

    metadata = {}

    # 1. Check that the analysis scripts have been committed, and store path + commit
    file = Path(__file__).resolve()
    if log_to_wandb:
        for f in (file, Path(base.__file__).resolve()):
            base.check_repo_not_dirty(f)
        print("Repo is clean and can proceed")

    git_root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    metadata['analysis_script_path'] = str(file.relative_to(git_root))
    metadata['base_analysis_script_path'] = str(Path(base.__file__).resolve().relative_to(git_root))
    metadata['analysis_script_commit_hash'] = base.retrieve_analysis_script_commit_hash()

    # 2. Postprocess each batch
    datasets, batches_metadata, common_parameters_per_batch = [], [], []
    for batch_number, results_folder in batches:
        df, batch_metadata, common_parameters = process_batch(batch_number, results_folder)
        datasets.append(df)
        batches_metadata.append(batch_metadata)
        common_parameters_per_batch.append(common_parameters)

    # 3. Check the batches are compatible, then combine
    all_draws, missing = check_batches_fit_together(datasets, batches_metadata, common_parameters_per_batch)

    dataset = pd.concat([df for df in datasets if not df.empty], ignore_index=True, sort=False)
    dataset = dataset.sort_values(['draw', 'person_ID_in_draw'], kind='stable').reset_index(drop=True)

    # Keep the draw parameters at the end, as in the standard postprocessing
    draw_cols = [c for c in dataset.columns if c.startswith('draw_parameter')]
    dataset = dataset[[c for c in dataset.columns if c not in draw_cols] + draw_cols]

    print(f"\nCombined {len(batches_metadata)} batches: {len(all_draws)} draws, {len(dataset)} episodes")

    # 4. Metadata for the combined dataset
    metadata['scenario_script_path'] = batches_metadata[0]['scenario_script_path']
    metadata['batches'] = batches_metadata
    metadata['global_draws_included'] = all_draws
    metadata['global_draws_missing'] = missing
    metadata['parameters'] = common_parameters_per_batch[0]  # identical across batches (checked above)

    output_folder.mkdir(parents=True, exist_ok=True)
    dataset_path = output_folder / 'dataset.csv'
    dataset.to_csv(dataset_path, mode='w')
    with open(output_folder / 'dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Wrote {dataset_path} and dataset_metadata.json")

    # 5. Store in wandb as a single dataset artifact
    if log_to_wandb:
        if wandb.run is not None:
            wandb.finish()

        wandb.init(project="dataset-demo", job_type="create and store dataset", name="test-run2-batches")

        artifact = wandb.Artifact(
            "test_dataset",
            type="dataset",
            description="Test dataset artifact used to train TLO emulators (combined from batch submissions)",
            metadata=metadata,
        )
        artifact.add_file(str(dataset_path))
        wandb.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine postprocessed individual histories from several batch submissions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path('.'),
        help="Directory to write the combined dataset and its metadata to",
    )
    parser.add_argument(
        "--log-to-wandb",
        action="store_true",
        help="Log the combined dataset as a wandb artifact",
    )
    args = parser.parse_args()

    apply(
        output_folder=args.output_path,
        log_to_wandb=args.log_to_wandb,
    )
