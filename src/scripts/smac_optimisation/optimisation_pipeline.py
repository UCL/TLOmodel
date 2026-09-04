"""
Ask-and-tell integration between SMAC3 and TLOmodel's Azure Batch system
(src/tlo/cli.py), with multiple concurrent jobs in flight.

Conceptually:

    smac.ask()  -> SMAC hands you a config to evaluate. It does NOT run
                   anything itself and does not care how or where you run it.
    (you run it)-> entirely your own code: submit to Azure Batch by
                   calling tlo.cli's reusable functions directly, poll
                   job state, download + parse outputs.
    smac.tell() -> you hand the result back. This is the ONLY point at
                   which SMAC's internal runhistory / surrogate model /
                   incumbent tracking are updated.

TLOmodel-SPECIFIC DESIGN NOTE
------------------------------
`tlo batch-submit` (the CLI command) requires the scenario file to be
committed and pushed, and is built to be invoked from a terminal - it
loads the scenario from a file path and parses CLI-style scenario_args.
Since this loop is calling from Python already, submit_azure_job()
below skips that CLI layer entirely: it builds the TloOptimisationScenario
object directly, sets SMAC's config values on it as real Python
attributes, and calls the same underlying functions (get_batch_client,
create_job, add_tasks, etc.) that batch_submit's Click command uses
internally - no string serialisation, no argparse, no subprocess.

The git-clean/commit check is still required (TLOmodel's reproducibility
model ties every batch job to a specific commit) but only needs to run
once per optimisation run, not once per SMAC trial, since the scenario
file's *code* never changes between trials - only the attribute values
set on it before each submission.

HYPERPARAMETERS: every tunable knob in this file is marked inline with a
"HYPERPARAMETER" comment - grep for that tag across all files
(constrained_ei.py, smac_scenario.py, convergence_monitoring.py,
optimisation_pipeline.py) to find the complete list in one pass.
"""

from __future__ import annotations

import datetime
import os
import time
import uuid
from string import Template
from dataclasses import dataclass
from pathlib import Path
import logging
logging.getLogger("azure").setLevel(logging.WARNING)
import numpy as np
from git import Repo
from azure.batch import models as batch_models

from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer, Categorical
from smac import HyperparameterOptimizationFacade, Scenario
from smac.runhistory.dataclasses import TrialInfo, TrialValue
from smac.runhistory.enumerations import StatusType

from tlo.cli import (
    is_file_clean, load_config, get_batch_client,
    create_file_share, create_directory, upload_local_file,
    create_job, add_tasks,
)
from smac_scenario import TloOptimisationScenario  # imported directly - it's a
                                                       # plain Python class now,
                                                       # not loaded from a file path
from constrained_ei import ConstrainedEI  # the module built earlier
from postprocess_output import postprocess_run
from convergence_monitoring import (
    append_history_to_file, config_key, get_best_feasible_dalys, check_convergence,
)
import json
JOB_LOG_FILE = Path("submitted_jobs.jsonl")

def _log_submitted_job(job_id: str, config: Configuration, seed: int, commit_hexsha: str) -> None:
    """
    Appends one line per submission, immediately after a job is
    successfully created on Azure. JSON Lines format specifically
    because each line is a complete, independent record - if the
    process crashes mid-write, only the last (incomplete) line is
    affected, every prior submission's record stays intact and readable.

    commit_hexsha records exactly which commit of smac_scenario.py this
    job was submitted under - relevant if the scenario file changes
    between pipeline runs, so a recovered result can always be traced
    back to the code that actually produced it.
    """
    record = {"job_id": job_id, "config": dict(config), "seed": seed, "commit": commit_hexsha}
    with open(JOB_LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

# --------------------------------------------------------------------------
# 1. TLO/Azure Batch interaction, built directly on tlo.cli's reusable
#    functions rather than going through the `tlo` command-line tool at
#    all - submission is entirely in-process Python from here on.
# --------------------------------------------------------------------------

SCENARIO_FILE = "src/scripts/smac_optimisation/smac_scenario.py"  # committed once;
                                                                     # only used here
                                                                     # for the git-clean
                                                                     # check, not for loading
CONFIG_FILE = "tlo.conf"

# --- Suspend/resume (https://github.com/UCL/TLOmodel/wiki/Suspend-and-resume-simulations) ---
# When SUSPENDED_JOB_ID is set (not None), every submission resumes from
# the SAME pre-recorded suspended simulation - i.e. every new SMAC trial
# shares identical pre-resume history, and only parameters affecting
# POST-resume behaviour can meaningfully differ between trials (confirmed
# with the person building this pipeline: all their tunable parameters
# are post-resume only, so this is a safe fit for their use case - see
# the caveat in submit_azure_job()'s docstring for what would go wrong
# if that weren't true).
#
# SUSPENDED_JOB_ID/DRAW/RUN together identify exactly ONE suspended
# pickle - <job_id>/<draw>/<run>/suspended_simulation.pickle on the file
# share - to resume from every time. Draw/run default to 0, matching
# this pipeline's own convention (number_of_draws=1, runs_per_draw=1),
# but are left overridable in case the suspended job being resumed from
# was created outside this pipeline (e.g. a manually-run multi-draw
# `tlo batch-submit ... --suspend-date ...`).
SUSPENDED_JOB_ID: str | None = None   # e.g. "long_run_all_diseases-2025-08-12T133044Z"
SUSPENDED_JOB_DRAW = 0
SUSPENDED_JOB_RUN = 0

_config = None  # lazily loaded, see _get_config()
_commit_hexsha = None  # resolved once per process, see _get_commit()
_batch_client = None  # lazily built, see _get_batch_client()


def _get_config():
    global _config
    if _config is None:
        _config = load_config(CONFIG_FILE)
    return _config


def _get_batch_client():
    """
    Cached so repeated polling (azure_job_is_finished / azure_task_succeeded
    get called once per pending job, every POLL_INTERVAL_SECONDS) doesn't
    re-authenticate against Key Vault on every single check.
    """
    global _batch_client
    if _batch_client is None:
        tlo_config = _get_config()
        _batch_client = get_batch_client(
            tlo_config["BATCH"]["CLIENT_ID"], tlo_config["BATCH"]["SECRET"],
            tlo_config["AZURE"]["TENANT_ID"], tlo_config["BATCH"]["URL"],
        )
    return _batch_client


def _get_commit() -> str:
    """
    Confirms the repo is committed & pushed (the same reproducibility
    guarantee `tlo batch-submit` enforces via is_file_clean), and resolves
    the commit hash once. Since the scenario file never changes between
    SMAC trials, this only needs to succeed once per optimisation run,
    not once per config.
    """
    global _commit_hexsha
    if _commit_hexsha is not None:
        return _commit_hexsha

    try:
        current_branch = is_file_clean(SCENARIO_FILE)
    except Exception as e:
        raise RuntimeError(
            "Scenario file's branch has not been pushed to remote yet - "
            "there's no origin/<branch> to compare against. Run "
            "'git push -u origin <branch>' once, then retry."
        ) from e

    if current_branch is False:
        raise RuntimeError(
            "Scenario file has uncommitted changes, or local commits that "
            "haven't been pushed yet. Commit and push before starting the "
            "SMAC run - this only needs doing once, not per trial."
        )

    repo = Repo(".")
    _commit_hexsha = next(repo.iter_commits(max_count=1)).hexsha
    return _commit_hexsha


@dataclass
class AzureJobHandle:
    job_id: str
    submitted_at: float
    commit_hexsha: str  # which commit this job was submitted under


def submit_azure_job(config: Configuration, seed: int) -> AzureJobHandle:
    """
    Builds a TloOptimisationScenario in-process (config values set as
    real Python attributes - no serialization to CLI strings and back),
    then reproduces the job-creation portion of `batch_submit` using the
    same reusable functions cli.py exports. This is everything
    batch_submit's Click command does internally, minus the CLI-parsing
    and scenario-file-loading steps we don't need since we're already
    holding the class in Python.

    `seed` is SMAC's own info.seed, and IS passed straight through to
    scenario.seed - this is the point of setting deterministic=False on
    the SMAC Scenario: it lets SMAC's own intensifier decide when a
    config is promising enough to justify evaluating it again under a
    different seed, rather than paying a fixed per-config averaging cost
    regardless of merit. See the SEEDING note at the top of
    smac_scenario.py for the full reasoning, and note runs_per_draw=1
    below - TLO's own multi-seed averaging is intentionally not used
    here, since SMAC is now the one deciding how many realisations a
    given config gets.
    """
    commit_hexsha = _get_commit()
    tlo_config = _get_config()

    # --- build the scenario as a plain Python object ---
    tlo_scenario = TloOptimisationScenario()
    for key, value in dict(config).items():
        if isinstance(value, np.bool_):
            value = bool(value)
        setattr(tlo_scenario, key, value)  # e.g. scenario.intervention_coverage = 0.73
    tlo_scenario.seed = seed  # SMAC's seed drives this run's stochasticity directly
    tlo_scenario.number_of_draws = 1
    tlo_scenario.runs_per_draw = 1  # one physical realisation per submission

    tlo_scenario.scenario_path = Path(SCENARIO_FILE)   # <-- add this line

    run_json = tlo_scenario.save_draws(commit=commit_hexsha)

    # --- from here down mirrors batch_submit's body in cli.py ---
    file_share_mount_point = "mnt"
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")
    job_id = tlo_scenario.get_log_config()["filename"] + "-" + timestamp + "-" + uuid.uuid4().hex[:8]
    azure_directory = f"{tlo_config['DEFAULT']['USERNAME']}/{job_id}"

    batch_client = _get_batch_client()
    create_file_share(tlo_config["STORAGE"]["CONNECTION_STRING"], tlo_config["STORAGE"]["FILESHARE"])
    for idx in range(len(os.path.split(azure_directory))):
        create_directory(
            tlo_config["STORAGE"]["CONNECTION_STRING"], tlo_config["STORAGE"]["FILESHARE"],
            "/".join(os.path.split(azure_directory)[: idx + 1]),
        )
    upload_local_file(
        tlo_config["STORAGE"]["CONNECTION_STRING"], run_json,
        tlo_config["STORAGE"]["FILESHARE"], azure_directory + "/" + os.path.basename(run_json),
    )

    pool_node_count = tlo_scenario.number_of_draws * tlo_scenario.runs_per_draw
    auto_user = batch_models.AutoUserSpecification(
        elevation_level=batch_models.ElevationLevel.admin, scope=batch_models.AutoUserScope.task,
    )
    user_identity = batch_models.UserIdentity(auto_user=auto_user)
    azure_file_url = "https://{}.file.core.windows.net/{}".format(
        tlo_config["STORAGE"]["NAME"], tlo_config["STORAGE"]["FILESHARE"],
    )
    container_registry = batch_models.ContainerRegistry(
        registry_server=tlo_config["REGISTRY"]["SERVER"],
        user_name=tlo_config["REGISTRY"]["NAME"], password=tlo_config["REGISTRY"]["KEY"],
    )
    image_name = f"{tlo_config['REGISTRY']['SERVER']}/{tlo_config['REGISTRY']['IMAGE']}:{tlo_config['REGISTRY']['DEFAULT_TAG']}"
    container_conf = batch_models.ContainerConfiguration(
        type="dockerCompatible", container_image_names=[image_name], container_registries=[container_registry],
    )
    azure_file_share_configuration = batch_models.AzureFileShareConfiguration(
        account_name=tlo_config["STORAGE"]["NAME"], azure_file_url=azure_file_url,
        account_key=tlo_config["STORAGE"]["KEY"], relative_mount_path=file_share_mount_point,
        mount_options="-o rw",
    )
    mount_configuration = batch_models.MountConfiguration(
        azure_file_share_configuration=azure_file_share_configuration,
    )

    remote_azure_directory = "${{AZ_BATCH_NODE_MOUNTS_DIR}}/" + f"{file_share_mount_point}/{azure_directory}"
    azure_run_json = f"{remote_azure_directory}/{os.path.basename(run_json)}"
    working_dir = "${{AZ_BATCH_TASK_WORKING_DIR}}"
    task_dir = "${{AZ_BATCH_TASK_DIR}}"

    # If SUSPENDED_JOB_ID is set, every task resumes from the SAME
    # suspended pickle rather than starting fresh. Path mirrors the
    # structure documented on the wiki (<job_id>/<draw>/<run>/
    # suspended_simulation.pickle), referenced via the SAME file-share
    # mount every other path in this function already uses - since
    # every job this pipeline submits copies its full working directory
    # (including any suspended_simulation.pickle) back to the file
    # share via the final `cp -r` below, an earlier suspended job's
    # pickle is already sitting there to be referenced.
    #
    # ASSUMPTION, not yet independently verified: this assumes
    # `tlo batch-run` itself accepts --resume-simulation <path> directly
    # (mirroring `tlo scenario-run`'s documented syntax), since this
    # pipeline builds its own remote command rather than going through
    # `tlo batch-submit`'s CLI layer (which is what the wiki's own
    # examples use, and which may do its own path resolution/translation
    # before invoking batch-run remotely). Worth confirming with a
    # cheap manual test (or `tlo batch-run --help`) before relying on
    # this for a real, costly run.
    resume_arg = ""
    if SUSPENDED_JOB_ID is not None:
        suspended_pickle_path = (
            "${{AZ_BATCH_NODE_MOUNTS_DIR}}/"
            + f"{file_share_mount_point}/{tlo_config['DEFAULT']['USERNAME']}/"
              f"{SUSPENDED_JOB_ID}/{SUSPENDED_JOB_DRAW}/{SUSPENDED_JOB_RUN}/suspended_simulation.pickle"
        )
        resume_arg = f"--resume-simulation {suspended_pickle_path}"

    command_template = Template("""
    git fetch origin $commit_hexsha
    git checkout $commit_hexsha
    pip install -r requirements/base.txt
    PYTHONOPTIMIZE=1 tlo --config-file tlo.example.conf batch-run $azure_run_json $working_dir {draw_number} {run_number} $resume_arg
    tlo --config-file tlo.example.conf parse-log $working_dir/{draw_number}/{run_number}
    cp $task_dir/std*.txt $working_dir/{draw_number}/{run_number}/.
    gzip $working_dir/{draw_number}/{run_number}/*.{{txt,log}}
    cp -r $working_dir/* $remote_azure_directory/.
    """)
    command = command_template.substitute(
        commit_hexsha=commit_hexsha,
        azure_run_json=azure_run_json,
        working_dir=working_dir,
        task_dir=task_dir,
        remote_azure_directory=remote_azure_directory,
        resume_arg=resume_arg,
    )
    command = f"/bin/bash -c '{command}'"

    create_job(
        batch_client, tlo_config["BATCH"]["POOL_VM_SIZE"], pool_node_count, job_id,
        container_conf, [mount_configuration], False, tlo_config["BATCH"]["SUBNET_ID"],
    )
    add_tasks(batch_client, user_identity, job_id, image_name, "--rm --workdir /TLOmodel", tlo_scenario, command)
    _log_submitted_job(job_id, config, seed, commit_hexsha)   # <-- add this line

    print(f"[submitted] job_id={job_id}")

    return AzureJobHandle(job_id=job_id, submitted_at=time.time(), commit_hexsha=commit_hexsha)


def azure_job_is_finished(job: AzureJobHandle) -> bool:
    """
    True once the task has reached a terminal state (completed OR
    failed) - i.e. it's no longer running and safe to stop polling.
    Does NOT imply success: Batch's "completed" state means the task
    finished RUNNING, not that it finished successfully. A crashed TLO
    run (non-zero exit code) also reaches "completed" - see
    azure_task_succeeded() to distinguish a genuine result from that.
    """
    batch_client = _get_batch_client()
    tasks = list(batch_client.task.list(job_id=job.job_id))
    if not tasks:
        return False  # task not yet visible to the API
    return tasks[0].state == "completed"


def azure_task_succeeded(job: AzureJobHandle) -> bool:
    """
    Only meaningful once azure_job_is_finished(job) is True. Checks the
    task's actual exit code - a completed-but-crashed TLO run (e.g. an
    exception partway through the simulation) still reaches "completed"
    task state, so exit code is what actually separates a real result
    from a failure. exit_code == 0 is success; anything else, or a
    missing execution_info entirely, is treated as failed.
    """
    batch_client = _get_batch_client()
    tasks = list(batch_client.task.list(job_id=job.job_id))
    if not tasks or tasks[0].execution_info is None:
        return False
    return tasks[0].execution_info.exit_code == 0


def download_run_outputs(job: AzureJobHandle) -> Path:
    """
    Downloads job outputs by walking the file share directly (the same
    traversal `tlo batch-download` performs). Returns the local
    directory containing draw 0's runs - download only, no analysis.
    """
    from azure.storage.fileshare import ShareClient

    tlo_config = _get_config()
    username = tlo_config["DEFAULT"]["USERNAME"]
    share_client = ShareClient.from_connection_string(
        tlo_config["STORAGE"]["CONNECTION_STRING"], tlo_config["STORAGE"]["FILESHARE"],
    )

    remote_root = f"{username}/{job.job_id}"
    local_root = Path("outputs", remote_root)

    def walk(dir_name: str):
        local_root_dir = Path("outputs", dir_name)
        os.makedirs(local_root_dir, exist_ok=True)
        for item in share_client.list_directories_and_files(dir_name):
            if item["is_directory"]:
                walk(f"{dir_name}/{item['name']}")
            else:
                file_client = share_client.get_file_client(f"{dir_name}/{item['name']}")
                with open(local_root_dir / item["name"], "wb") as f:
                    f.write(file_client.download_file().readall())

    walk(remote_root)
    return local_root / "0"  # draw 0 - only draw we ever submit


def aggregate_postprocessed_results(draw_dir: Path) -> dict:
    """Shared by fetch_azure_result (fresh download) and crash-recovery
    (already-downloaded outputs from a prior process)."""
    per_run_results = [postprocess_run(run_dir) for run_dir in sorted(draw_dir.iterdir())]
    return {
        "dalys": float(np.mean([r["dalys"] for r in per_run_results])),
        "cost": float(np.mean([r["cost"] for r in per_run_results])),
        "hr_used": float(np.mean([r["hr_used"] for r in per_run_results])),
        "stock_used": float(np.mean([r["stock_used"] for r in per_run_results])),
    }

def fetch_azure_result(job: AzureJobHandle) -> dict:
    draw_dir = download_run_outputs(job)
    return aggregate_postprocessed_results(draw_dir)



# --------------------------------------------------------------------------
# 2. Your real config space, constraint limits, and shared history log
#    NOTE: every hyperparameter name here must match an attribute
#    TloOptimisationScenario expects in smac_scenario.py, since
#    submit_azure_job() does setattr(scenario, key, value) for each one.
# --------------------------------------------------------------------------

configspace = ConfigurationSpace()
configspace.add(Float("config_annual_testing_rate_adults",(0.,1.)))
configspace.add(Float("annual_rate_selftest",(0.,1.)))
configspace.add(Float("prob_hiv_test_at_anc_or_delivery",(0.,1.)))
configspace.add(Float("prob_hiv_test_for_newborn_infant",(0.,1.)))
configspace.add(Float("prob_prep_for_fsw_after_hiv_test",(0.,1.)))
configspace.add(Float("prob_prep_for_agyw",(0.,1.)))
configspace.add(Float("prob_injectable_prep_vs_oral",(0.,1.)))
configspace.add(Float("prob_circ_after_hiv_test",(0.,1.)))
configspace.add(Float("linked_to_care_after_selftest",(0.,1.)))
configspace.add(Float("prob_receive_viral_load_test_result",(0.,1.)))
configspace.add(Float("config_coverage_plhiv",(0.,1.)))
configspace.add(Categorical("switch_vl_test_to_tdf",[True,False]))

# Problem-defined constraints, not algorithm hyperparameters - but kept
# here as a reminder these are duplicated in constrained_ei.py's
# standalone example_usage() too, and could drift out of sync if only
# one copy gets updated.
COST_LIMIT = 2_000_000
HR_LIMIT = 500
STOCK_LIMIT = 10_000

history: list[dict] = []  # raw, disaggregated results - the source of truth


def record_result(
    config: Configuration, seed: int, result: dict,
    job_id: str | None = None, commit: str | None = None,
) -> TrialValue:
    """
    Turns a raw Azure result into (a) a history entry with full detail,
    and (b) the single scalar TrialValue SMAC's own bookkeeping needs.
    This is the ask-tell equivalent of what target_function used to do
    in the synchronous version - same logic, just called manually here
    instead of by SMAC itself. Shared by both the live polling loop and
    seed_history_with_prior_runs() below, so there's one place computing
    violations/penalty rather than two copies that could drift apart.

    `seed` is logged into history (not just used for TrialInfo) because,
    with deterministic=False and SMAC's intensifier possibly requesting
    multiple seeds for the same competitive config, history can now
    contain several distinct noisy realisations of the same config -
    seed lets you trace and, at final-selection time, group them back
    together (see the feasible/best selection logic at the bottom of
    this file).

    `job_id`/`commit` are provenance only, for monitoring - they're
    merged into the on-disk history_log.jsonl record below, but
    deliberately kept OUT of `history` itself, so ConstrainedEI's
    surrogates never see them as if they were real features.
    """
    dalys, cost = result["dalys"], result["cost"]
    hr_used, stock_used = result["hr_used"], result["stock_used"]

    cost_violation = max(0.0, cost / COST_LIMIT - 1)
    hr_violation = max(0.0, hr_used / HR_LIMIT - 1)
    stock_violation = max(0.0, stock_used / STOCK_LIMIT - 1)

    history.append({
        "config_object": config,
        "seed": seed,
        "dalys": dalys,
        "cost_violation": cost_violation,
        "hr_violation": hr_violation,
        "stock_violation": stock_violation,
    })

    append_history_to_file({**history[-1], "job_id": job_id, "commit": commit})

    K = 3 * dalys  # HYPERPARAMETER: penalty coefficient - rough, not load-bearing
    # for search quality now that ConstrainedEI does the real steering,
    # but keeps smac.incumbent / logging / terminate_cost_threshold sane.
    penalty = K * (cost_violation + hr_violation + stock_violation)
    return TrialValue(cost=dalys + penalty)


def recover_from_job_log() -> list[dict]:
    """
    Checks every previously-submitted job against the local outputs
    directory. If a job's output was already downloaded (i.e. a prior
    process got far enough to fetch it, but crashed before calling
    smac.tell()), re-postprocess it and return it in PRIOR_RUNS shape.

    Only jobs submitted under the CURRENT commit are recovered - a run
    submitted under a different commit may have used a different
    smac_scenario.py (different draw_parameters mapping, different
    modules, etc.), so silently folding its DALYs/cost into history
    would risk mixing results that aren't actually comparable. Such
    jobs are skipped, with a warning, rather than loaded.

    SCOPE: only recovers jobs whose outputs are already downloaded
    locally. A job that finished on Azure but was never downloaded
    before the crash is NOT recovered here - it's neither re-checked
    against Azure nor re-downloaded.
    """
    if not JOB_LOG_FILE.exists():
        return []

    tlo_config = _get_config()
    username = tlo_config["DEFAULT"]["USERNAME"]
    current_commit = _get_commit()

    recovered = []
    with open(JOB_LOG_FILE) as f:
        for line in f:
            record = json.loads(line)

            if record.get("commit") != current_commit:
                print(
                    f"[warning] skipping recovered job {record['job_id']} - "
                    f"submitted under commit {record.get('commit', 'unknown')[:12]}, "
                    f"current commit is {current_commit[:12]}"
                )
                continue

            draw_dir = Path("outputs", username, record["job_id"], "0")
            if not draw_dir.exists() or not any(draw_dir.iterdir()):
                continue  # not downloaded - out of scope, see docstring

            result = aggregate_postprocessed_results(draw_dir)
            recovered.append({
                "config": record["config"], "seed": record["seed"],
                "job_id": record["job_id"], "commit": record["commit"],
                **result,
            })
    return recovered


def seed_history_with_prior_runs(prior_runs: list[dict], smac) -> int:
    """
    Warm-starts BOTH `history` (which ConstrainedEI trains its random
    forests on) and SMAC's own runhistory/incumbent tracking, using
    already-completed runs - so the search doesn't start from scratch.

    Each entry in prior_runs should have the same shape as a
    postprocessed+aggregated fetch_azure_result() output, plus the
    raw config values that produced it:

        {
            "config": {"intervention_coverage": 0.6, "consumable_stock_target": 0.9, ...},
            "dalys": 41.2, "cost": 1_750_000.0, "hr_used": 470.0, "stock_used": 9100.0,
        }

    Confirmed supported by SMAC3: previously-evaluated configs can be
    added via tell() even though they were never produced by SMAC's own
    ask() - you construct the TrialInfo yourself instead. Call this
    BEFORE the main ask-tell loop starts.

    Returns the number of runs seeded (use this to offset n_completed's
    starting value in the main loop, so scenario.n_trials counts total
    trials including these, not only newly-submitted ones).
    """
    n_seeded = 0
    for i, run in enumerate(prior_runs):
        config = Configuration(configspace, values=run["config"])
        # use the real seed if this prior run's was recorded (honest
        # provenance in SMAC's runhistory); fall back to the enumeration
        # index only when it's genuinely unknown.
        seed = run.get("seed", i)
        value = record_result(config, seed, run, job_id=run.get("job_id"), commit=run.get("commit"))
        info = TrialInfo(config=config, seed=seed)
        smac.tell(info, value)
        n_seeded += 1
    return n_seeded


# --------------------------------------------------------------------------
# 3. Build SMAC in ask-tell mode (n_trials still needed for its budget
#    bookkeeping, even though you're driving the loop yourself)
# --------------------------------------------------------------------------

scenario = Scenario(configspace, n_trials=100, deterministic=False)  # HYPERPARAMETER (n_trials):
                                                                        # total trial budget - remember
                                                                        # max_config_calls means this
                                                                        # isn't the same as "number of
                                                                        # distinct configs explored"

# HYPERPARAMETER: max_config_calls caps how many seeds the intensifier
# will use to confirm any single config (default is 3). Set to 5 to
# match the "5 seeds for true convergence" threshold - a config can
# still be discarded early on fewer seeds if it's clearly uncompetitive,
# but nothing gets promoted to incumbent-quality trust on fewer than up
# to 5 confirmations. Combined with the noisy-EI correction in
# ConstrainedEI, this makes the adaptive sampling trustworthy rather
# than just efficient.
intensifier = HyperparameterOptimizationFacade.get_intensifier(scenario, max_config_calls=5)

acquisition_function = ConstrainedEI(
    configspace=configspace,
    objective_name="dalys",
    constraint_names=["cost_violation", "hr_violation", "stock_violation"],
    history_provider=lambda: history,
    retrain_every=5,  # HYPERPARAMETER: refit every 5 new results
)

smac = HyperparameterOptimizationFacade(
    scenario,
    target_function=lambda config, seed=0: 0.0,  # never actually called in
                                                    # ask-tell mode, but the
                                                    # facade still requires
                                                    # something with the right
                                                    # signature at construction
    acquisition_function=acquisition_function,
    intensifier=intensifier,
    overwrite=True,
)


# --------------------------------------------------------------------------
# 4. Warm-start with runs you've already completed, THEN start the loop.
#    Replace this with however you're currently loading your existing
#    completed runs (CSV, dataframe, pickle, whatever they're sitting in).
# --------------------------------------------------------------------------

PRIOR_RUNS = [
    #{
    #    "config": {"year_mode_switch": 2019, "tclose_days_offset_overwrite": 5},
    #    "dalys": 45.3, "cost": 1_680_000.0, "hr_used": 455.0, "stock_used": 8_900.0,
    #},
    # ... your other already-completed runs ...
]

recovered_runs = recover_from_job_log()
all_prior_runs = PRIOR_RUNS + recovered_runs
n_seeded = seed_history_with_prior_runs(all_prior_runs, smac)
print(f"Warm-started with {n_seeded} run(s): {len(PRIOR_RUNS)} manual, {len(recovered_runs)} recovered from job log")


# --------------------------------------------------------------------------
# 5. The ask-tell loop itself: keep N_CONCURRENT Azure jobs in flight,
#    ask() for a replacement each time one completes and is told back.
# --------------------------------------------------------------------------

N_CONCURRENT = 6           # HYPERPARAMETER (operational): concurrent Azure jobs in flight -
                            # interacts with retrain_every; several jobs finishing in the
                            # same polling pass can mean refitting more often than intended
POLL_INTERVAL_SECONDS = 10  # HYPERPARAMETER (operational): low-stakes, trades API call
                              # frequency against latency between job completion and SMAC seeing it

pending: list[tuple[TrialInfo, AzureJobHandle]] = []

# prime the pipeline
for _ in range(N_CONCURRENT):
    info = smac.ask()
    pending.append((info, submit_azure_job(info.config, info.seed)))

n_completed = n_seeded  # start counting from the warm-started runs, not zero
best_dalys_over_time: list[float] = []
converged = False


def _top_up_pending(still_pending: list) -> None:
    """
    Ask SMAC for a fresh config and submit it, keeping N_CONCURRENT
    slots full - unless the budget or convergence say otherwise. Shared
    by the success path and both failure paths below, so a failed Azure
    task or a postprocessing crash doesn't permanently shrink how many
    jobs are in flight.
    """
    if n_completed < scenario.n_trials and not converged:
        new_info = smac.ask()
        still_pending.append((new_info, submit_azure_job(new_info.config, new_info.seed)))


while pending or (n_completed < scenario.n_trials and not converged):
    made_progress = False
    still_pending = []

    for info, job in pending:
        if not azure_job_is_finished(job):
            still_pending.append((info, job))
            continue

        # --- the Azure task itself failed (non-zero exit code) - this
        # trial produced no usable result. Told to SMAC as CRASHED (not
        # skipped entirely) so its runhistory/intensifier have an honest
        # record that this (config, seed) was attempted and failed -
        # SMAC's own runhistory encoder already knows to exclude crashed
        # trials from surrogate training via considered_states, so this
        # doesn't pollute the model, but it does stop SMAC from being
        # blind to the fact that this specific combination was tried.
        # Doesn't count toward n_trials or touch history/ConstrainedEI -
        # a replacement config is still submitted so the failure doesn't
        # shrink concurrency.
        if not azure_task_succeeded(job):
            print(
                f"[failed] job_id={job.job_id} - Azure task exited non-zero. "
                f"Skipping this trial."
            )
            smac.tell(info, TrialValue(cost=np.inf, status=StatusType.CRASHED))
            made_progress = True
            _top_up_pending(still_pending)
            continue

        # --- the Azure task succeeded, but postprocessing can still
        # fail (missing/malformed output files, the year-completeness
        # assertion in postprocess_output.py, etc). Catching this
        # separately from the exit-code check above means one bad run
        # can't propagate up and kill the whole optimisation process -
        # in particular, it can't wipe out SMAC's accumulated in-memory
        # surrogate/incumbent state built up over potentially hours.
        # Same CRASHED-status reasoning as above applies here too.
        try:
            result = fetch_azure_result(job)
            value = record_result(info.config, info.seed, result, job_id=job.job_id, commit=job.commit_hexsha)
            smac.tell(info, value)          # <-- the actual "notify SMAC" step
        except Exception as e:
            print(f"[postprocessing failed] job_id={job.job_id} - {e!r}. Skipping this trial.")
            smac.tell(info, TrialValue(cost=np.inf, status=StatusType.CRASHED))
            made_progress = True
            _top_up_pending(still_pending)
            continue

        n_completed += 1
        made_progress = True

        # --- convergence check (logic lives in convergence_monitoring.py) ---
        current_best = get_best_feasible_dalys(history)
        if current_best is not None:
            best_dalys_over_time.append(current_best)

        if not converged and check_convergence(best_dalys_over_time):
            converged = True
            print(f"Stopping new submissions; draining {len(pending) - 1} pending job(s).")

        _top_up_pending(still_pending)

    pending = still_pending
    if not made_progress:
        time.sleep(POLL_INTERVAL_SECONDS)


# --------------------------------------------------------------------------
# 6. Final answer: group history by config and average across whatever
#    seeds SMAC ended up requesting for it, THEN filter+select - never
#    trust smac.incumbent directly, and never trust a single noisy
#    realisation's DALYs either, now that history holds individual
#    (config, seed) results rather than pre-averaged bundles.
# --------------------------------------------------------------------------

grouped: dict[tuple, list[dict]] = {}
for h in history:
    grouped.setdefault(config_key(h["config_object"]), []).append(h)

aggregated = []
for entries in grouped.values():
    aggregated.append({
        "config_object": entries[0]["config_object"],
        "n_seeds_evaluated": len(entries),
        "dalys": float(np.mean([e["dalys"] for e in entries])),
        "cost_violation": float(np.mean([e["cost_violation"] for e in entries])),
        "hr_violation": float(np.mean([e["hr_violation"] for e in entries])),
        "stock_violation": float(np.mean([e["stock_violation"] for e in entries])),
    })

feasible = [
    a for a in aggregated
    if a["cost_violation"] == 0 and a["hr_violation"] == 0 and a["stock_violation"] == 0
]
best = min(feasible, key=lambda a: a["dalys"])
print(
    "Best feasible config:", dict(best["config_object"]),
    "DALYs:", best["dalys"], f"(averaged over {best['n_seeds_evaluated']} seed(s))",
)
