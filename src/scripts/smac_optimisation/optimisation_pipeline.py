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
import pandas as pd
from git import Repo
from azure.batch import models as batch_models

from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer, Categorical
from ConfigSpace.hyperparameters import CategoricalHyperparameter  # NOT the same as
    # Categorical above - that's a convenience FUNCTION (confirmed directly from
    # ConfigSpace's own docs: "Categorical is actually a function, please use the
    # corresponding return types if doing an isinstance(param, type) check"), not a
    # class - isinstance(hp, Categorical) raises TypeError ("arg 2 must be a type"),
    # confirmed directly from a real traceback. CategoricalHyperparameter is the
    # actual underlying class Categorical(...) constructs and returns - use THIS
    # for any isinstance check (see sample_near_baseline_configs()).
from smac import HyperparameterOptimizationFacade, Scenario
from smac.runhistory.dataclasses import TrialInfo, TrialValue
from smac.runhistory.enumerations import StatusType

from tlo import Date
from tlo.cli import (
    is_file_clean, load_config, get_batch_client,
    create_file_share, create_directory, upload_local_file,
    create_job, add_tasks,
)
from smac_scenario import TloOptimisationScenario  # imported directly - it's a
                                                       # plain Python class now,
                                                       # not loaded from a file path
from smac_scenario_suspend import TloCheckpointScenario  # used ONLY for checkpoint
    # generation, never real trials - see generate_checkpoint_job(). Genuinely
    # different class name from smac_scenario.py's own TloOptimisationScenario now,
    # so no import alias is needed to avoid a collision.
from smac_scenario_baseline import TloCheckpointScenario as TloBaselineScenario  # ALIASED:
    # smac_scenario_baseline.py's class is ALSO literally named
    # TloCheckpointScenario (same name as smac_scenario_suspend.py's own class) -
    # used ONLY for the one-off baseline run (see submit_baseline_job()), never
    # for checkpoints or real trials. runs_per_draw=10 is hardcoded in that
    # file's own __init__ - this scenario always produces 10 differently-seeded
    # full (non-suspend/resume) runs per submission. BASELINE_CONFIG_VALUES
    # (the baseline's own genuine values for the 13 configspace parameters)
    # is imported from initialise.py further down instead, NOT from here -
    # it must NOT be imported at module top-level: initialise.py's own
    # module-level code reads COST_LIMITS_FILE at import time, which the
    # baseline-run block below is what actually WRITES, so importing
    # initialise.py (for ANY name) any earlier than that block already
    # runs would reintroduce the exact ordering bug this project's
    # baseline-run submission was specifically restructured to avoid.
from constrained_ei import ConstrainedEI  # the module built earlier
from postprocess_output import postprocess_run, compute_and_save_baseline_budgets
from checkpoint_seeds import checkpoint_job_id, CHECKPOINT_SEEDS
from convergence_monitoring import (
    append_history_to_file, config_key, get_best_feasible_dalys, check_convergence,
    json_safe_config,
)
from optimisation_parameters import (
    N_TRIALS, MAX_CONFIG_CALLS, RETRAIN_EVERY, EI_XI, MIN_SAMPLES_LEAF, PENALTY_COEFFICIENT_MULTIPLIER,
    INFEASIBILITY_FLOOR_MULTIPLIER, MERIT_VIOLATION_THRESHOLD, MERIT_PENALTY_ALPHA,
    N_CONCURRENT, POLL_INTERVAL_SECONDS, USE_SUSPEND_RESUME, SUBMIT_SUSPEND_PART,
    VALID_CHECKPOINT_COMMITS, VALID_PRIOR_RUN_COMMITS, CONFIG_YEAR_START_DATE,
    SUBMIT_BASELINE_RUN, START_FIRST_BOUNDARY, END_THIRD_BOUNDARY,
    SUBMIT_INITIAL_DESIGN, N_INIT, INITIAL_DESIGN_NEAR_BASELINE_FRACTION, INITIAL_DESIGN_PERTURBATION_STD,
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
    record = {"job_id": job_id, "config": json_safe_config(config), "seed": seed, "commit": commit_hexsha}
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
CHECKPOINT_SCENARIO_FILE = "src/scripts/smac_optimisation/smac_scenario_suspend.py"  # the
    # scenario used ONLY for checkpoint generation (see
    # generate_checkpoint_job()) - PATH ASSUMED to match SCENARIO_FILE's own
    # convention (same directory); confirm this is actually where the file
    # lives/will be committed.
BASELINE_SCENARIO_FILE = "src/scripts/smac_optimisation/smac_scenario_baseline.py"  # the
    # scenario used ONLY for the one-off baseline run (see
    # submit_baseline_job()) - NOT the same file as CHECKPOINT_SCENARIO_FILE,
    # despite both files' classes being literally named TloCheckpointScenario
    # (see the aliased import above) - this one hardcodes runs_per_draw=10
    # and never gets suspended, and its own module docstring currently
    # says "Committed once at .../smac_scenario_suspend.py" (a leftover
    # copy-paste from when this file was cloned from that one) - worth
    # fixing in smac_scenario_baseline.py itself, though harmless here
    # since we never actually read that docstring.
CONFIG_FILE = "tlo.conf"

# --- Suspend/resume (https://github.com/UCL/TLOmodel/wiki/Suspend-and-resume-simulations) ---
# USE_SUSPEND_RESUME / SUBMIT_SUSPEND_PART now live in
# optimisation_parameters.py (imported above), alongside every other
# pipeline hyperparameter, rather than being defined here.
#
# When USE_SUSPEND_RESUME is True, every submission resumes from a
# pre-resume checkpoint - found via find_checkpoint_commit_for_seed()
# (which commit's checkpoint to trust for this seed) and
# checkpoint_seeds.checkpoint_job_id() (that commit's actual job id).
# The checkpoint is referenced by JOB ID directly, through the SAME
# shared file-share mount every task already has - confirmed against
# TLOmodel's own molaro/optimise_hiv_program_w_smac branch and the
# wiki's documented examples (<job_id>/<draw>, draw not run). NO local
# download or re-upload is involved (an earlier version of this
# pipeline did both, based on an incorrect understanding of the
# mechanism) - ensure_checkpoints_ready() only confirms readiness.
#
# There is no longer a single shared "SUSPENDED_JOB_ID": each of the
# MAX_CONFIG_CALLS checkpoints is its OWN separate Azure job, now also
# scoped by commit (see VALID_CHECKPOINT_COMMITS in
# optimisation_parameters.py). Each checkpoint must be submitted as a
# single-run job (sample_number=0) with scenario.seed set DIRECTLY to
# its target seed, since TLO computes
# simulation_seed = low_bias_32(scenario_seed + sample_number), and a
# real trial resuming later always has sample_number=0 too
# (runs_per_draw=1). Batching multiple checkpoints under one shared
# scenario.seed would instead vary sample_number, producing a
# completely different (and wrong) low_bias_32 output that no real
# trial would ever independently reproduce - see checkpoint_seeds.py's
# module docstring for the full reasoning.

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

    IMPORTANT, when USE_SUSPEND_RESUME is True: scenario.seed set here
    is still serialised into this trial's own JSON exactly as usual, but
    it has NO EFFECT on a resumed trial's actual randomness. Confirmed
    directly from TLO's own docs - tlo.core: every module carries its
    own numpy.random.RandomState, with its own internal state; tlo.
    simulation: save_to_pickle()/load_from_pickle() serialise/restore
    the ENTIRE Simulation object via dill, which necessarily captures
    every module's RNG state exactly as it stood at suspend time.
    Simulation.load_from_pickle() also takes no seed argument at all.
    So a resumed run's randomness continues from whatever RNG state was
    already baked into the checkpoint at generation time (see
    generate_checkpoint_job(), which sets scenario.seed to
    CHECKPOINT_SEEDS[i] - THAT assignment is the one that actually
    matters) - this trial's own `seed` here only ever gets used LOCALLY,
    to determine which checkpoint job (via find_checkpoint_commit_for_seed())
    this trial's --resume-simulation reference points at, never to seed
    anything on the remote node.

    RESUME MECHANISM, confirmed directly against TLO's current master
    cli.py: `tlo batch-run` (what actually executes remotely) takes a
    FIXED four positional arguments and nothing else - no
    --resume-simulation, no catch-all for extra scenario args. Passing
    --resume-simulation on the remote command line (an earlier version
    of this function did) fails outright. The REAL mechanism, mirrored
    from batch_submit's own body in cli.py: scenario.parse_arguments()
    gets called LOCALLY, BEFORE scenario.save_draws() - so whatever
    --resume-simulation sets gets baked directly into the run_json file
    itself, and the remote `tlo batch-run` command never needs to see
    any extra flags at all.

    --resume-simulation's VALUE is a JOB ID reference (<job_id>/<draw>,
    draw not run), NOT a local file path - confirmed directly against
    TLOmodel's own molaro/optimise_hiv_program_w_smac branch (see
    batch_submit's own --resume-simulation rewriting) and the wiki's
    documented examples. The checkpoint is referenced through the SAME
    shared file-share mount every task already has - no local download,
    no re-upload into this trial's own directory (an earlier version of
    this function did both, based on an incorrect understanding of the
    mechanism - see ensure_checkpoints_ready() for the much simpler
    corresponding readiness check, which only confirms the checkpoint
    job finished successfully on Azure).
    """
    commit_hexsha = _get_commit()
    tlo_config = _get_config()

    # --- build the scenario as a plain Python object ---
    tlo_scenario = TloOptimisationScenario()
    for key, value in json_safe_config(config).items():
        setattr(tlo_scenario, key, value)  # e.g. scenario.intervention_coverage = 0.73
    tlo_scenario.seed = seed  # drives this run's stochasticity ONLY when
                               # USE_SUSPEND_RESUME is False (a fresh
                               # Simulation genuinely gets seeded from this
                               # value via low_bias_32). When resuming, this
                               # is set/serialised for consistency but is
                               # otherwise INERT - see this function's
                               # docstring for why, confirmed from TLO's docs.
    tlo_scenario.number_of_draws = 1
    tlo_scenario.runs_per_draw = 1  # one physical realisation per submission

    tlo_scenario.scenario_path = Path(SCENARIO_FILE)   # <-- add this line

    # --- job identity / remote paths ---
    file_share_mount_point = "mnt"

    # If USE_SUSPEND_RESUME is True, this task resumes from a checkpoint
    # via --resume-simulation, referenced by JOB ID directly (through
    # the SAME shared file-share mount every task already has) - NOT by
    # downloading anything locally or re-uploading a copy. Confirmed
    # directly against TLOmodel's own molaro/optimise_hiv_program_w_smac
    # branch (batch_submit's own --resume-simulation rewriting) - value
    # prefixed with ${AZ_BATCH_NODE_MOUNTS_DIR}/<mount>/<username>/,
    # mirroring path_to_job's own construction exactly (verified
    # empirically: a genuine f-string with escaped double braces
    # collapses to single braces immediately, matching path_to_job's own
    # behaviour byte-for-byte).
    #
    # VALUE IS THE BARE JOB ID - confirmed from a real traceback:
    # run_sample_by_number() itself appends /{draw}/{sample}/
    # suspended_simulation.pickle to whatever --resume-simulation value
    # it's given, REGARDLESS of what we pass - an earlier version of
    # this line added a trailing /0 (misreading the wiki's <job_id>
    # [/draw] syntax as something WE needed to supply), which produced
    # a doubled draw component (job_id/0/0/0/... instead of
    # job_id/0/0/...) and a second FileNotFoundError. Bare job_id + TLO's
    # own appended /0/0 (our checkpoints are always single-draw,
    # single-run) is exactly correct.
    #
    # Earlier versions of this function also downloaded the checkpoint
    # locally and re-uploaded it into each trial's own directory, based
    # on an incorrect understanding of how --resume-simulation is meant
    # to be used - removed entirely; see ensure_checkpoints_ready() for
    # the (much simpler) corresponding checkpoint-readiness check.
    if USE_SUSPEND_RESUME:
        commit_for_checkpoint = find_checkpoint_commit_for_seed(seed)
        if commit_for_checkpoint is None:
            raise RuntimeError(
                f"No checkpoint job found for seed {seed} under the current commit "
                f"or any commit in VALID_CHECKPOINT_COMMITS (optimisation_parameters.py)."
            )
        checkpoint_job = checkpoint_job_id(seed, commit_for_checkpoint)
        resume_reference = (f"${{AZ_BATCH_NODE_MOUNTS_DIR}}/"
                             f"{file_share_mount_point}/"
                             f"{tlo_config['DEFAULT']['USERNAME']}/"
                             f"{checkpoint_job}")
        tlo_scenario.parse_arguments(["--resume-simulation", resume_reference])

    run_json = tlo_scenario.save_draws(commit=commit_hexsha)

    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")
    job_id = tlo_scenario.get_log_config()["filename"] + "-" + timestamp + "-" + uuid.uuid4().hex[:8]
    azure_directory = f"{tlo_config['DEFAULT']['USERNAME']}/{job_id}"
    remote_azure_directory = "${{AZ_BATCH_NODE_MOUNTS_DIR}}/" + f"{file_share_mount_point}/{azure_directory}"

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

    azure_run_json = f"{remote_azure_directory}/{os.path.basename(run_json)}"
    working_dir = "${{AZ_BATCH_TASK_WORKING_DIR}}"
    task_dir = "${{AZ_BATCH_TASK_DIR}}"

    # NOTE: no --resume-simulation (or anything else) appended to the
    # remote command - `tlo batch-run` has a fixed 4-positional-argument
    # signature in TLO's current master cli.py, with no mechanism to
    # accept extra flags at all. Resume behaviour is baked into run_json
    # via parse_arguments() above.
    #
    # resume_reference (built above) is DELIBERATELY constructed to
    # match batch_submit's own path_to_job exactly, byte for byte -
    # confirmed correct on this project's own molaro/
    # optimise_hiv_program_w_smac branch, where `tlo batch-submit ...
    # --resume-simulation` is confirmed working on Azure. No local
    # patching (an earlier version of this function added a sed step
    # here, based on being unable to independently verify from source
    # how the bash-variable-reference text gets resolved) - trusting the
    # confirmed-working mechanism directly, rather than second-guessing
    # it: this pipeline's only actual difference from batch_submit is
    # WHERE the job-id-to-resume-from comes from (looked up here via
    # find_checkpoint_commit_for_seed()/checkpoint_job_id(), matched to
    # this trial's own seed, rather than taken from a user-supplied CLI
    # argument) - the resulting value and how it's used are identical.
    # sed step is REQUIRED here, not optional - CONFIRMED from Microsoft's
    # own Azure Batch documentation (Task runtime environment variables):
    # "The command lines executed by tasks on compute nodes don't run
    # under a shell... To use [environment variable expansion] you must
    # invoke the shell in the command line." This bash script IS an
    # explicitly-invoked shell (command = f"/bin/bash -c '{command}'"
    # below) - which is exactly why $azure_run_json/$working_dir/
    # $task_dir, USED WITHIN THIS SCRIPT, correctly expand. But
    # --resume-simulation's value never appears on any command line at
    # all - it's content INSIDE run_json, read directly by Python's
    # Path()/open() (tlo/scenario.py's run_sample_by_number), with no
    # shell ever invoked to process it. Confirmed also directly against
    # tlo/scenario.py's own parse_arguments (type=str, no expansion
    # anywhere) and run_sample_by_number (plain Path() construction).
    # This is NOT a deviation from batch_submit's own cli.py logic (that
    # file is confirmed byte-for-byte identical between master and this
    # project's branch) - it's compensating for whatever differs in
    # tlo/scenario.py between the two, since suspend/resume is confirmed
    # working on master but not on this branch.
    patch_json_line = (
        'sed -i "s|\\${{AZ_BATCH_NODE_MOUNTS_DIR}}|$AZ_BATCH_NODE_MOUNTS_DIR|g" ' + azure_run_json
        if USE_SUSPEND_RESUME else ""
    )

    # DIAGNOSTIC LINE (env | grep "^AZ_" ...) copied VERBATIM from
    # batch_submit's own command construction in cli.py - prints every
    # AZ_-prefixed environment variable actually present on this
    # specific remote task, directly to stdout.
    command_template = Template("""
    git fetch origin $commit_hexsha
    git checkout $commit_hexsha
    pip install -r requirements/base.txt
    env | grep "^AZ_" | while read line; do echo "$$line"; done
    $patch_json_line
    PYTHONOPTIMIZE=1 tlo --config-file tlo.example.conf batch-run $azure_run_json $working_dir {draw_number} {run_number}
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
        patch_json_line=patch_json_line,
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


# --------------------------------------------------------------------------
# Pre-resume checkpoint generation - the "first part" of suspend/resume.
#
# One standalone job per seed in CHECKPOINT_SEEDS, each running only up
# to SUSPEND_DATE (before any config-dependent behaviour starts) and
# saving a suspended_simulation.pickle - reused later by every real
# trial that happens to get that same seed (see checkpoint_seeds.py's
# module docstring for why each must be its OWN single-run submission,
# not batched, for TLO's low_bias_32 formula to line up correctly).
# --------------------------------------------------------------------------

SUSPEND_DATE = Date(CONFIG_YEAR_START_DATE - 1, 12, 31)  # the LAST DAY BEFORE
    # config-dependent behaviour starts, NOT CONFIG_YEAR_START_DATE itself.
    # TLO's resume mechanism has no separate "resume date" argument -
    # Simulation.load_from_pickle() just continues the simulation forward
    # from wherever its internal clock/event queue stopped (confirmed from
    # its signature - no date parameter at all). So suspending on
    # (CONFIG_YEAR_START_DATE-1)-12-31 means the very next simulated day,
    # once resumed, is naturally CONFIG_YEAR_START_DATE-01-01 - exactly
    # the intended resume point - with no second constant needed.
    # Suspending ON CONFIG_YEAR_START_DATE-01-01 itself (the previous,
    # incorrect version of this line) would instead checkpoint one day
    # too late - AFTER that first day had already been simulated under
    # whatever fixed, no-scale-up behaviour the checkpoint scenario uses,
    # rather than handing that day over to the real, resumed config.
    #
    # Derived directly from CONFIG_YEAR_START_DATE (optimisation_parameters.py),
    # the SAME constant smac_scenario_suspend.py's config_start_year and
    # initialise.py's PERIOD_BOUNDARIES both derive from.


def generate_checkpoint_job(seed: int) -> AzureJobHandle:
    """
    Submits ONE standalone job that runs only up to SUSPEND_DATE and
    saves a suspended_simulation.pickle - closely mirrors
    submit_azure_job()'s own mechanics, with three deliberate
    differences:

    1. job_id is checkpoint_job_id(seed, commit) - deterministic, not
       the usual filename+timestamp+uuid scheme - so a later resuming
       trial can compute this job's location directly from its own seed
       (and whichever commit it's decided to trust - see
       find_checkpoint_commit_for_seed()) with no lookup table involved
       (see checkpoint_seeds.py).
    2. scenario.seed is set DIRECTLY to `seed` (not SMAC's info.seed via
       the normal ask-tell loop - this function is called ahead of the
       main loop, not from within it), with number_of_draws=1,
       runs_per_draw=1 (draw=0, run=0), so TLO's
       low_bias_32(scenario_seed + sample_number=0) here matches exactly
       what a real trial resuming with info.seed=seed will independently
       compute for itself. THIS is the assignment that actually
       determines a resumed trial's randomness - confirmed from TLO's
       own docs that save_to_pickle()/load_from_pickle() preserve every
       module's RandomState exactly as it stood at suspend time, and
       load_from_pickle() takes no seed argument at all. The seed set on
       a RESUMING trial's own scenario (submit_azure_job(), when
       USE_SUSPEND_RESUME is True) is genuinely inert by comparison -
       see that function's docstring.
    3. The remote command runs a scenario that's already been configured,
       via parse_arguments(), to suspend at SUSPEND_DATE - not by
       passing --suspend-date to the remote `tlo batch-run` command
       (an earlier version of this function did - confirmed against
       TLO's current master cli.py that `batch_run` has a FIXED four-
       positional-argument signature with no mechanism to accept extra
       flags at all, so that always failed). Instead, mirroring
       batch_submit's own body in cli.py: parse_arguments() is called
       LOCALLY, BEFORE save_draws(), so the suspend-date gets baked
       directly into run_json itself - the remote command becomes a
       plain, unmodified `tlo batch-run`.

    ASSUMPTION, not yet independently verified: that scenario.
    parse_arguments(["--suspend-date", "<date>"]) is genuinely how
    TLO's own tlo/scenario.py expects this to be invoked, mirroring
    --resume-simulation's confirmed handling in cli.py's batch_submit -
    --suspend-date itself doesn't appear anywhere in cli.py, so its
    exact parsing lives entirely in tlo/scenario.py, which hasn't been
    directly inspected. Worth confirming with a cheap manual test
    before relying on this for a real, costly checkpoint-generation run.
    """
    commit_hexsha = _get_commit()
    tlo_config = _get_config()

    tlo_scenario = TloCheckpointScenario()  # smac_scenario_suspend.py's class -
        # NOT smac_scenario.py's - see the module-level import comment
    tlo_scenario.seed = seed
    # number_of_draws/runs_per_draw already correctly 1/1 in
    # TloCheckpointScenario's own __init__ (single hardcoded scenario) -
    # set again here explicitly anyway, for defensiveness against that
    # class's own defaults ever changing.
    tlo_scenario.number_of_draws = 1
    tlo_scenario.runs_per_draw = 1
    tlo_scenario.scenario_path = Path(CHECKPOINT_SCENARIO_FILE)

    # Baked into run_json via parse_arguments(), BEFORE save_draws() -
    # see docstring point 3 for why this replaced passing --suspend-date
    # as a remote CLI argument.
    tlo_scenario.parse_arguments(["--suspend-date", SUSPEND_DATE.strftime("%Y-%m-%d")])

    run_json = tlo_scenario.save_draws(commit=commit_hexsha)

    file_share_mount_point = "mnt"
    job_id = checkpoint_job_id(seed, commit_hexsha)  # deterministic, commit-scoped - see docstring point 1
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

    # NOTE: no --suspend-date (or anything else) appended to the remote
    # command any more - see docstring point 3. Plain, unmodified
    # `tlo batch-run` - the suspend behaviour is already baked into
    # run_json via parse_arguments() above.
    #
    # DIAGNOSTIC LINE copied VERBATIM from batch_submit's own command
    # construction in cli.py - see submit_azure_job()'s identical
    # addition for why.
    command_template = Template("""
    git fetch origin $commit_hexsha
    git checkout $commit_hexsha
    pip install -r requirements/base.txt
    env | grep "^AZ_" | while read line; do echo "$$line"; done
    PYTHONOPTIMIZE=1 tlo --config-file tlo.example.conf batch-run $azure_run_json $working_dir {draw_number} {run_number}
    cp -r $working_dir/* $remote_azure_directory/.
    """)
    command = command_template.substitute(
        commit_hexsha=commit_hexsha,
        azure_run_json=azure_run_json,
        working_dir=working_dir,
        remote_azure_directory=remote_azure_directory,
    )
    command = f"/bin/bash -c '{command}'"

    create_job(
        batch_client, tlo_config["BATCH"]["POOL_VM_SIZE"], pool_node_count, job_id,
        container_conf, [mount_configuration], False, tlo_config["BATCH"]["SUBNET_ID"],
    )
    add_tasks(batch_client, user_identity, job_id, image_name, "--rm --workdir /TLOmodel", tlo_scenario, command)

    print(f"[checkpoint submitted] job_id={job_id} seed={seed}")
    return AzureJobHandle(job_id=job_id, submitted_at=time.time(), commit_hexsha=commit_hexsha)


def generate_all_checkpoints() -> None:
    """
    Submits one checkpoint job for EVERY seed in CHECKPOINT_SEEDS,
    unconditionally - controlled purely by the SUBMIT_SUSPEND_PART
    hyperparameter (optimisation_parameters.py) at the call site, NOT by
    checking Azure for what already exists. An earlier version of this
    function checked whether a job already existed first and skipped
    anything already present - deliberately removed: whether to
    (re)generate checkpoints this run is now entirely a user decision
    (set SUBMIT_SUSPEND_PART), not something inferred from Azure state.
    If you don't want to regenerate existing checkpoints, set
    SUBMIT_SUSPEND_PART = False.

    Call this ONCE, before the main ask-tell loop, whenever
    SUBMIT_SUSPEND_PART is True - and wait for all of them to finish
    (e.g. via azure_job_is_finished polling) before letting the main
    loop submit any real trial that might reference one of these
    checkpoints.
    """
    for seed in CHECKPOINT_SEEDS:
        generate_checkpoint_job(seed)


def submit_baseline_job() -> AzureJobHandle:
    """
    TODO / KNOWN AWKWARDNESS, worth consolidating later: this is one of
    TWO genuinely separate "baseline" mechanisms in this file, which
    currently don't share any code or data with each other, despite
    both representing the same underlying status-quo scenario:
      1. THIS function - smac_scenario_baseline.py's own scenario
         ('type_of_scaleup': 'none'), 10 non-suspend/resume runs, used
         ONLY to derive the budget (compute_and_save_baseline_budgets()).
      2. submit_initial_design_jobs() below - BASELINE_CONFIG_VALUES
         (initialise.py) submitted through the STANDARD smac_scenario.py
         path instead (suspend/resume, single seed), as one of the
         initial-design points SMAC's own search gets warm-started with.
    Kept deliberately separate for now (simpler to reason about while
    this is still being iterated on) rather than unifying them into one
    submission path - a reasonable thing to revisit once both have
    proven out, not before.

    Submits ONE standard (non-suspend/resume) Azure job running
    smac_scenario_baseline.py's TloBaselineScenario, which hardcodes
    runs_per_draw=10 - i.e. this single submission produces 10 FULL,
    INDEPENDENTLY-SEEDED, complete simulation runs (draw_0-run_0
    through draw_0-run_9), each a genuine, unmodified end-to-end run -
    no --suspend-date, no --resume-simulation, nothing baked into
    run_json beyond the scenario's own fixed, no-scale-up parameters.

    Deliberately does NOT use CHECKPOINT_SEEDS or any seed-matching
    logic at all - unlike a real trial or a checkpoint, this run's own
    seeds don't need to match anything else in the pipeline. TLO's own
    scenario-execution machinery assigns each of the 10 runs its own
    seed internally (via low_bias_32(scenario_seed + sample_number),
    sample_number = 0..9) - we never need to know or control those
    individual values, only that they're genuinely different from each
    other, which runs_per_draw=10 alone guarantees.

    Closely mirrors generate_checkpoint_job()'s general submission
    mechanics (build scenario, save_draws, upload, create_job,
    add_tasks) - see that function and submit_azure_job() for the fully
    commented version of each step; comments here focus only on what's
    different for the baseline case.
    """
    commit_hexsha = _get_commit()
    tlo_config = _get_config()

    tlo_scenario = TloBaselineScenario()  # smac_scenario_baseline.py's class -
        # aliased on import to avoid colliding with smac_scenario_suspend.py's
        # own, differently-behaved TloCheckpointScenario. number_of_draws=1/
        # runs_per_draw=10 already correctly set in its own __init__ - NOT
        # overridden here, unlike generate_checkpoint_job()'s defensive
        # re-assignment, since 10 runs is the entire point of this submission.
    tlo_scenario.scenario_path = Path(BASELINE_SCENARIO_FILE)

    # No parse_arguments() call at all - this is a full, ordinary run,
    # not suspend/resume, so there's nothing to bake into run_json beyond
    # what TloBaselineScenario's own draw_parameters() already provides.
    run_json = tlo_scenario.save_draws(commit=commit_hexsha)

    file_share_mount_point = "mnt"
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H%M%SZ")
    job_id = tlo_scenario.get_log_config()["filename"] + "-baseline-" + timestamp
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

    pool_node_count = tlo_scenario.number_of_draws * tlo_scenario.runs_per_draw  # 1 * 10 = 10
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

    # Standard command - no sed patch needed (nothing suspend/resume-related
    # baked into run_json for this submission), diagnostic env line kept
    # for consistency with the other two submission functions.
    command_template = Template("""
    git fetch origin $commit_hexsha
    git checkout $commit_hexsha
    pip install -r requirements/base.txt
    env | grep "^AZ_" | while read line; do echo "$$line"; done
    PYTHONOPTIMIZE=1 tlo --config-file tlo.example.conf batch-run $azure_run_json $working_dir {draw_number} {run_number}
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
    )
    command = f"/bin/bash -c '{command}'"

    create_job(
        batch_client, tlo_config["BATCH"]["POOL_VM_SIZE"], pool_node_count, job_id,
        container_conf, [mount_configuration], False, tlo_config["BATCH"]["SUBNET_ID"],
    )
    add_tasks(batch_client, user_identity, job_id, image_name, "--rm --workdir /TLOmodel", tlo_scenario, command)

    print(f"[baseline submitted] job_id={job_id} (10 runs)")
    return AzureJobHandle(job_id=job_id, submitted_at=time.time(), commit_hexsha=commit_hexsha)


def sample_near_baseline_configs(n: int) -> list[Configuration]:
    """
    Generates n Configuration objects as LOCAL PERTURBATIONS of the
    baseline's own values (BASELINE_CONFIG_VALUES) - see
    optimisation_parameters.INITIAL_DESIGN_NEAR_BASELINE_FRACTION's own
    docstring for the full motivation (uniform sampling across a
    13-dimensional space rarely lands near any single reference point by
    chance; the baseline is presumably already a reasonable, near-feasible
    operating point, and the whole point of the search is to find
    something that BEATS it).

    Each FLOAT parameter is perturbed via Normal(baseline_value,
    INITIAL_DESIGN_PERTURBATION_STD * (upper - lower)), then clipped back
    into [lower, upper] - scaled by each parameter's OWN range (not an
    absolute value) so the same std setting means the same relative
    "closeness" regardless of a parameter's own scale. Each CATEGORICAL
    parameter (tdf_test_replace_vl_test, targeted_adherence_monitoring)
    keeps the baseline's own value UNCHANGED, every time - no perturbation
    at all, since there's no natural notion of "slightly different" for a
    binary switch (see that same docstring for the reasoning, and where
    to change this if you'd rather they still vary with some probability).

    Bounds are read directly from each hyperparameter's own object in
    configspace (hp.lower/hp.upper) rather than hardcoded, so this stays
    correct if configspace's own (0., 1.) ranges are ever changed.
    """
    configs = []
    for _ in range(n):
        values = {}
        for name, baseline_value in BASELINE_CONFIG_VALUES.items():
            hp = configspace[name]
            if isinstance(hp, CategoricalHyperparameter):
                values[name] = baseline_value
            else:
                perturbed = np.random.normal(
                    baseline_value, INITIAL_DESIGN_PERTURBATION_STD * (hp.upper - hp.lower)
                )
                values[name] = float(np.clip(perturbed, hp.lower, hp.upper))
        configs.append(Configuration(configspace, values=values))
    return configs


def submit_initial_design_jobs() -> None:
    """
    TODO / KNOWN AWKWARDNESS, worth consolidating later - see
    submit_baseline_job()'s own docstring above: this function's own
    use of BASELINE_CONFIG_VALUES (submitted through the standard
    smac_scenario.py path, suspend/resume, single seed) is a SEPARATE
    mechanism from submit_baseline_job()'s own 10-run, non-suspend/
    resume smac_scenario_baseline.py submission - both represent the
    same underlying status-quo scenario, but currently share no code or
    data. Kept deliberately separate for now.

    Submits N_INIT jobs (optimisation_parameters.py) BEFORE the main
    ask-tell loop starts - N_INIT-1 configs, plus the baseline's own
    config (BASELINE_CONFIG_VALUES, imported from initialise.py alongside
    PRIOR_RUNS) - each submitted with exactly ONE seed (no intensification
    at this stage; that's SMAC's own intensifier's job, later, once the
    main loop is running).

    Of the N_INIT-1 configs, a fraction (INITIAL_DESIGN_NEAR_BASELINE_
    FRACTION) are sampled as LOCAL PERTURBATIONS of the baseline itself
    (sample_near_baseline_configs(), above) rather than uniformly across
    the whole search space (configspace.sample_configuration()) - see
    that function's own docstring, and INITIAL_DESIGN_NEAR_BASELINE_
    FRACTION's own comment in optimisation_parameters.py, for the full
    motivation: uniform sampling rarely lands near a specific reference
    point by chance in a 13-dimensional space, but the baseline is
    presumably already reasonable/near-feasible, and the search's whole
    goal is to find something that BEATS it - so giving the initial
    design a real chance at feasible, competitive neighbours of the
    baseline, not just wildly different global samples, is a more
    targeted use of these first few trials. The rest are still sampled
    globally, uniformly, so broad exploration isn't lost either.

    Waits here for every one of these jobs to actually finish (blocking,
    polling all of them concurrently - same pattern as
    ensure_checkpoints_ready()) - but does NOT itself load anything into
    history/SMAC, and does NOT call record_result()/smac.tell() at all.
    Once these jobs are genuinely finished, they get picked up entirely
    by the EXISTING recover_from_job_log() mechanism (called right after
    this function, in this file's own module-level flow) - since
    submit_azure_job() already logs every submission to JOB_LOG_FILE
    internally (via _log_submitted_job()), under the SAME commit every
    other submission this run uses, these jobs need no separate loading
    path at all: the same commit-matching, the same restart-
    deduplication (append_history_to_file()'s own job_id-keyed check)
    that already apply to every other recovered job apply here too,
    automatically.

    ALL jobs here - every random config AND the baseline - share the
    SAME SINGLE seed: CHECKPOINT_SEEDS[0]. An earlier version of this
    function gave each job its OWN, distinct seed, on the assumption
    that seed diversity was needed for these jobs to be meaningfully
    different from each other - WRONG: this genuinely replicates how
    the live pipeline's OWN early behaviour actually works. Confirmed
    directly from a real history_log.jsonl earlier in this project:
    SMAC's own intensifier submits its FIRST several challengers - each
    a genuinely different, newly-proposed config - all under the SAME
    seed, only drawing a DIFFERENT seed later, when it decides to
    INTENSIFY (confirm) one specific, already-promising config with an
    additional seed. The diversity across these N_INIT jobs is meant to
    come entirely from the CONFIGS themselves (genuinely different
    parameter values), not from seed variation - matching that same
    convention keeps these jobs directly, honestly comparable to
    whatever SMAC's own intensifier does with CHECKPOINT_SEEDS[0] once
    the main loop actually starts, and removes any need to worry about
    N_INIT exceeding CHECKPOINT_SEEDS' own pool size (an earlier
    version of this function raised ValueError over exactly that,
    unnecessarily, given every job now shares the one seed regardless
    of how large N_INIT is).

    A failed initial-design job is WARNED about, not raised on - unlike
    the baseline's own submission (which the real budget derivation
    depends on), these are exploratory warm-start jobs the pipeline can
    perfectly well proceed without; a failure here just means one fewer
    initial-design point gets picked up by recover_from_job_log() below.
    """
    n_random = N_INIT - 1
    shared_seed = CHECKPOINT_SEEDS[0]

    configs_and_seeds: list[tuple[Configuration, int]] = []

    n_near = round(n_random * INITIAL_DESIGN_NEAR_BASELINE_FRACTION) if n_random > 0 else 0
    n_global = n_random - n_near

    if n_random > 0:
        near_configs = sample_near_baseline_configs(n_near) if n_near > 0 else []

        global_configs = []
        if n_global > 0:
            sampled = configspace.sample_configuration(size=n_global)
            # ConfigSpace's own API quirk: sample_configuration(size=1) can
            # return a bare Configuration rather than a length-1 list,
            # depending on version - normalise defensively either way.
            if not isinstance(sampled, list):
                sampled = [sampled]
            global_configs = sampled

        for config in near_configs + global_configs:
            configs_and_seeds.append((config, shared_seed))

    baseline_config = Configuration(configspace, values=BASELINE_CONFIG_VALUES)
    configs_and_seeds.append((baseline_config, shared_seed))

    jobs: list[AzureJobHandle] = []
    for config, seed in configs_and_seeds:
        jobs.append(submit_azure_job(config, seed))
    print(
        f"[initial design] submitted {len(jobs)} job(s) ({n_near} near-baseline + "
        f"{n_global} global random + 1 baseline), all under seed {shared_seed}."
    )

    still_waiting = set(range(len(jobs)))
    while still_waiting:
        made_progress = False
        for i in list(still_waiting):
            job = jobs[i]
            if not azure_job_is_finished(job):
                continue
            made_progress = True
            still_waiting.discard(i)
            if azure_task_succeeded(job):
                print(f"[initial design] {job.job_id} confirmed ready.")
            else:
                print(f"[initial design] WARNING: {job.job_id} finished but FAILED - skipping it.")
        if still_waiting and not made_progress:
            time.sleep(POLL_INTERVAL_SECONDS)


def azure_job_exists(job_id: str) -> bool:
    """
    True if a job with this id already exists on Azure Batch, regardless
    of its current state. Used by find_checkpoint_commit_for_seed() to
    search across candidate commits for an existing, reusable checkpoint
    - a genuine LOOKUP, not (per the earlier design decision) something
    used to silently skip a submission the user asked for.
    generate_all_checkpoints()/generate_checkpoint_job() still submit
    unconditionally, controlled purely by SUBMIT_SUSPEND_PART - see
    that function's own docstring - this function is never called from
    there.
    """
    batch_client = _get_batch_client()
    try:
        batch_client.job.get(job_id=job_id)
        return True
    except Exception:
        return False


def find_checkpoint_commit_for_seed(seed: int) -> str | None:
    """
    Searches, in order, the CURRENT commit first, then every commit
    listed in VALID_CHECKPOINT_COMMITS (optimisation_parameters.py), for
    one that has an EXISTING checkpoint job for `seed` - lets you reuse
    a checkpoint generated under an older, manually-vetted commit (e.g.
    a later commit changed something that doesn't affect the pre-resume
    portion of the simulation at all) rather than being forced to
    regenerate one every time the commit changes.

    Returns the first matching commit, or None if none of the searched
    commits has a checkpoint job for this seed at all.
    """
    current_commit = _get_commit()
    candidate_commits = [current_commit] + [c for c in VALID_CHECKPOINT_COMMITS if c != current_commit]
    for commit in candidate_commits:
        if azure_job_exists(checkpoint_job_id(seed, commit)):
            return commit
    return None


def ensure_checkpoints_ready() -> None:
    """
    Called ONCE, before the main ask-tell loop, whenever
    USE_SUSPEND_RESUME is True. Confirms every checkpoint this run will
    need has ACTUALLY FINISHED on Azure, and succeeded - does NOT
    download anything locally.

    An earlier version of this function also downloaded each checkpoint
    locally (via download_run_outputs) so it could be re-uploaded into
    each real trial's own directory - based on an INCORRECT
    understanding of how --resume-simulation actually works. Confirmed
    directly against TLOmodel's own molaro/optimise_hiv_program_w_smac
    branch and the wiki's documented examples: a resuming trial
    references its checkpoint by JOB ID directly (<job_id>/<draw>),
    through the SAME shared file-share mount every task already has -
    there is nothing to download here, only readiness to confirm. See
    submit_azure_job() for where the actual --resume-simulation
    reference gets constructed.

    ALL seeds are polled CONCURRENTLY each pass (same pattern as the
    main ask-tell loop's pending/still_pending) - since
    generate_all_checkpoints() submits every checkpoint job at once,
    they're all running concurrently on Azure, and waiting on them
    strictly sequentially would mean only ever checking on one job at a
    time while others might already be sitting finished, unnoticed.

    Raises loudly (RuntimeError) if a needed seed has no checkpoint job
    at all under the current commit or any commit in
    VALID_CHECKPOINT_COMMITS, or if one exists but failed - never
    silently proceeds in either case.
    """
    jobs_by_seed: dict[int, AzureJobHandle] = {}
    for seed in CHECKPOINT_SEEDS:
        commit = find_checkpoint_commit_for_seed(seed)
        if commit is None:
            raise RuntimeError(
                f"No checkpoint job found for seed {seed} under the current commit "
                f"or any commit in VALID_CHECKPOINT_COMMITS (optimisation_parameters.py). "
                f"Run with SUBMIT_SUSPEND_PART=True first, or add an appropriate "
                f"commit to VALID_CHECKPOINT_COMMITS."
            )
        jobs_by_seed[seed] = AzureJobHandle(job_id=checkpoint_job_id(seed, commit), submitted_at=0.0, commit_hexsha=commit)
        print(f"[checkpoint] waiting for {jobs_by_seed[seed].job_id} (seed={seed}) to finish...")

    still_waiting = set(CHECKPOINT_SEEDS)
    while still_waiting:
        made_progress = False

        for seed in list(still_waiting):
            job = jobs_by_seed[seed]
            if not azure_job_is_finished(job):
                continue

            made_progress = True
            still_waiting.discard(seed)

            if not azure_task_succeeded(job):
                raise RuntimeError(
                    f"Checkpoint job {job.job_id} (seed={seed}) finished but FAILED - "
                    f"cannot proceed with USE_SUSPEND_RESUME=True until this is resolved "
                    f"(regenerate it, or remove this commit from VALID_CHECKPOINT_COMMITS "
                    f"if it's no longer trusted)."
                )

            print(f"[checkpoint] {job.job_id} confirmed ready.")

        if still_waiting and not made_progress:
            time.sleep(POLL_INTERVAL_SECONDS)


def azure_job_is_finished(job: AzureJobHandle) -> bool:
    """
    True once EVERY task in the job has reached a terminal state
    (completed OR failed) - i.e. none are still running, and it's safe
    to stop polling.

    For jobs with a single task (real trials, checkpoints -
    pool_node_count is always 1 there) this is equivalent to checking
    that one task. For MULTI-TASK jobs (the baseline run -
    pool_node_count = number_of_draws * runs_per_draw = 10 there), an
    earlier version of this function only ever checked tasks[0],
    silently ignoring the other 9 - meaning the wait loop could return
    True, and the caller would proceed straight to downloading/
    postprocessing, the moment task 0 finished, even while several of
    the other 9 were still running or had genuinely failed. Confirmed
    as the actual cause of a real baseline run reporting only 8 of its
    expected 10 runs.

    Does NOT imply success: Batch's "completed" state means a task
    finished RUNNING, not that it finished successfully. A crashed TLO
    run (non-zero exit code) also reaches "completed" - see
    azure_task_succeeded() to distinguish a genuine result from that.
    """
    batch_client = _get_batch_client()
    tasks = list(batch_client.task.list(job_id=job.job_id))
    if not tasks:
        return False  # no tasks visible to the API yet
    return all(t.state == "completed" for t in tasks)


def azure_task_succeeded(job: AzureJobHandle) -> bool:
    """
    Only meaningful once azure_job_is_finished(job) is True. Checks
    EVERY task's actual exit code, not just the first (see
    azure_job_is_finished()'s own docstring for why this matters
    specifically for multi-task jobs - the baseline run) - a
    completed-but-crashed TLO run (e.g. an exception partway through
    the simulation) still reaches "completed" task state, so exit code
    is what actually separates a real result from a failure. Every
    task's exit_code == 0 is required for success; any task with a
    different exit code, or missing execution_info entirely, makes
    this False.
    """
    batch_client = _get_batch_client()
    tasks = list(batch_client.task.list(job_id=job.job_id))
    if not tasks:
        return False
    return all(t.execution_info is not None and t.execution_info.exit_code == 0 for t in tasks)


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
    """
    Shared by fetch_azure_result (fresh download) and crash-recovery
    (already-downloaded outputs from a prior process). Takes the MEDIAN
    of both by-year cost dicts (hiv_hrh_cost_by_year,
    hiv_consumable_cost_by_year) and of dalys, across whatever runs
    exist per draw (currently always exactly 1, given runs_per_draw=1 -
    see smac_scenario.py - but implemented generally in case that ever
    changes), year by year rather than as flat totals. MEDIAN, not
    mean, for consistency with the rest of the pipeline's own DALYs/cost
    aggregation (see the final config-selection grouping and
    convergence_monitoring.get_best_feasible_dalys) - this is
    specifically the "single runs" side of that convention: this
    function's OWN output is what gets compared against the baseline's
    upper-95%-CI budget in bucket_cumulative_violation(), which is a
    genuinely different (and unchanged) computation.
    """
    per_run_results = [postprocess_run(run_dir) for run_dir in sorted(draw_dir.iterdir())]

    def median_yearly(key: str) -> dict:
        all_years = set()
        for r in per_run_results:
            all_years.update(r[key].keys())
        return {
            year: float(np.median([r[key].get(year, 0.0) for r in per_run_results]))
            for year in all_years
        }

    return {
        "dalys": float(np.median([r["dalys"] for r in per_run_results])),
        "hiv_hrh_cost_by_year": median_yearly("hiv_hrh_cost_by_year"),
        "hiv_consumable_cost_by_year": median_yearly("hiv_consumable_cost_by_year"),
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
configspace.add(Float("config_annual_testing_rate_adults", (0., 1.)))
configspace.add(Float("annual_rate_selftest", (0., 1.)))
configspace.add(Float("prob_hiv_test_at_anc_or_delivery", (0., 1.)))
configspace.add(Float("prob_hiv_test_for_newborn_infant", (0., 1.)))
configspace.add(Float("prob_prep_for_fsw_after_hiv_test", (0., 1.)))
configspace.add(Float("prob_prep_for_agyw", (0., 1.)))
configspace.add(Float("prob_injectable_prep_vs_oral", (0., 1.)))
configspace.add(Float("prob_circ_after_hiv_test", (0., 1.)))
configspace.add(Float("linked_to_care_after_selftest", (0., 1.)))
configspace.add(Float("prob_receive_viral_load_test_result", (0., 1.)))
configspace.add(Float("config_coverage_plhiv", (0., 1.)))
configspace.add(Categorical("tdf_test_replace_vl_test", [True, False]))
configspace.add(Categorical("targeted_adherence_monitoring", [True, False]))
configspace.add(Float("config_target_IPT", (0., 1.)))

# --------------------------------------------------------------------------
# Period-bucketed HIV-HRH and HIV-consumable cost constraints
#
# Rather than a year-by-year constraint (prohibitive across a 25-year
# horizon - would need ~25 separate RF surrogates per cost type, with
# the multiplied-probability problem discussed at length getting worse
# with every added constraint), each cost type is bucketed into a small,
# USER-DEFINABLE number of periods. Within each period, the CUMULATIVE
# (mean-across-years, not max) violation is computed:
#
#     period_violation = mean( max(0, cost_year/limit_year - 1) for year in period )
#
# Mean-across-years (not max-across-years, and NOT an average restricted
# to only the years that violate) so that an ADDITIONAL bad year can
# only raise or maintain a period's score, never lower it - averaging
# only the positive values would let a config with MORE violating years
# score BETTER than one with fewer, which is the wrong direction.
#
# PERIOD_BOUNDARIES is user-definable (absolute calendar years,
# inclusive on both ends) and MUST be the same list for HIV-HRH and
# HIV-consumable costs - both are bucketed identically, not with
# independently-chosen periods per cost type.
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# 2b. Checkpoint-generation AND baseline-run SUBMISSION, kicked off
#     together here (both fire-and-forget submissions, running
#     concurrently on Azure) - BEFORE initialise.py gets imported below,
#     since that import reads COST_LIMITS_FILE at MODULE IMPORT TIME, and
#     the baseline run's whole purpose is to (re)derive that file's
#     contents before the real constraint setup ever happens. Checkpoint
#     generation doesn't strictly need to move this early too - it has
#     no COST_LIMITS_FILE dependency at all - but is moved here anyway so
#     both submissions genuinely overlap on Azure ("alongside"), rather
#     than needlessly serialising checkpoint generation after the
#     baseline run's own wait below.
#
#     SUBMIT_SUSPEND_PART / SUBMIT_BASELINE_RUN are both plain user
#     toggles (optimisation_parameters.py) - NOT derived from checking
#     what's already on Azure or already in COST_LIMITS_FILE. Checkpoint
#     generation is NOT waited on here (see "2b (continued)" below,
#     right before "3. Build SMAC" - unchanged from before) - only the
#     baseline run is, since ITS result is needed before initialise.py's
#     import, a few lines down, can safely proceed.
# --------------------------------------------------------------------------

# Called once, here, before ANYTHING submits a real trial - blocks
# (polls) until every needed checkpoint has actually finished on Azure
# and succeeded. No local download involved - see ensure_checkpoints_
# ready()'s own docstring for why, and submit_azure_job() for where the
# actual --resume-simulation job-id reference gets constructed.
#
# MUST happen here, right after generate_all_checkpoints() itself, and
# BEFORE submit_initial_design_jobs() below - an earlier version of
# this file left this call much later (see the "2b." section comment
# further down, which still explains the reasoning for WHY it can
# safely live outside the COST_LIMITS_FILE/initialise.py ordering
# constraint) on the assumption that nothing would submit a real trial
# before that later point - true until submit_initial_design_jobs()
# was added, which DOES submit real, suspend/resume-using trials, via
# the exact same submit_azure_job() every real trial uses. If the
# checkpoint for CHECKPOINT_SEEDS[0] isn't actually finished and
# confirmed ready yet, those jobs have nothing to resume from.
if SUBMIT_SUSPEND_PART:
    generate_all_checkpoints()
if USE_SUSPEND_RESUME:
    ensure_checkpoints_ready()

if SUBMIT_BASELINE_RUN:
    baseline_job = submit_baseline_job()
    print(f"[baseline] waiting for {baseline_job.job_id} to finish (10 runs)...")
    while not azure_job_is_finished(baseline_job):
        time.sleep(POLL_INTERVAL_SECONDS)
    if not azure_task_succeeded(baseline_job):
        raise RuntimeError(
            f"Baseline job {baseline_job.job_id} finished but FAILED - cannot "
            f"derive budget constraints from it. Investigate and resubmit "
            f"(SUBMIT_BASELINE_RUN=True) before proceeding."
        )
    print(f"[baseline] {baseline_job.job_id} confirmed ready - computing budget CIs.")
    baseline_draw_dir = download_run_outputs(baseline_job)
    compute_and_save_baseline_budgets(baseline_draw_dir, START_FIRST_BOUNDARY, END_THIRD_BOUNDARY)

from initialise import (
    PERIOD_BOUNDARIES, HIV_HRH_BUDGET_BY_YEAR, HIV_CONSUMABLE_BUDGET_BY_YEAR,
    bucket_cumulative_violation, HIV_HRH_CONSTRAINT_NAMES,
    HIV_CONSUMABLE_CONSTRAINT_NAMES, CONSTRAINT_NAMES, PRIOR_RUNS, BASELINE_CONFIG_VALUES,
)

# Initial design: N_INIT-1 random configs + the baseline, ONE seed each,
# submitted and waited on BEFORE recover_from_job_log() runs below - so
# that mechanism picks them up naturally, the same way it already
# handles any other previously-submitted job. See
# submit_initial_design_jobs()'s own docstring for the full picture -
# no separate loading step is needed here; recover_from_job_log() (a
# few lines down) already does that, for these jobs exactly as it
# already does for every other one.
if SUBMIT_INITIAL_DESIGN:
    submit_initial_design_jobs()

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

    PERIOD_BOUNDARIES/budgets/bucket_cumulative_violation/constraint
    names all live in initialise.py - see that file for the actual
    constraint definitions and the cost-vs-budget terminology note.
    """
    dalys = result["dalys"]

    hrh_period_violations = bucket_cumulative_violation(
        result["hiv_hrh_cost_by_year"], HIV_HRH_BUDGET_BY_YEAR
    )
    consumable_period_violations = bucket_cumulative_violation(
        result["hiv_consumable_cost_by_year"], HIV_CONSUMABLE_BUDGET_BY_YEAR
    )

    entry = {"config_object": config, "seed": seed, "dalys": dalys}
    for name, v in zip(HIV_HRH_CONSTRAINT_NAMES, hrh_period_violations):
        entry[name] = v
    for name, v in zip(HIV_CONSUMABLE_CONSTRAINT_NAMES, consumable_period_violations):
        entry[name] = v
    history.append(entry)

    append_history_to_file({**history[-1], "job_id": job_id, "commit": commit})

    K = PENALTY_COEFFICIENT_MULTIPLIER * dalys  # penalty coefficient - rough, not load-bearing
    # for search quality now that ConstrainedEI does the real steering,
    # but keeps smac.incumbent / logging / terminate_cost_threshold sane.
    total_violation = sum(entry[name] for name in CONSTRAINT_NAMES)
    penalty = K * total_violation
    if total_violation > 0:
        # See optimisation_parameters.INFEASIBILITY_FLOOR_MULTIPLIER's own
        # comment for why this floor is necessary - without it, a trial
        # with a small enough total_violation barely moves cost above its
        # own raw dalys, letting it look "better" than a genuinely
        # feasible config with higher dalys to SMAC's OWN intensifier
        # (which, unlike ConstrainedEI, has no separate awareness of
        # feasibility at all) - confirmed as the actual cause of a real
        # observed case where an infeasible config got intensified.
        penalty += INFEASIBILITY_FLOOR_MULTIPLIER * dalys
    return TrialValue(cost=dalys + penalty)


def recover_from_job_log() -> list[dict]:
    """
    Checks every previously-submitted job against the local outputs
    directory. Two recovery paths, both re-postprocessing into
    PRIOR_RUNS shape:

    1. Already downloaded locally (a prior process got far enough to
       fetch it, but crashed before calling smac.tell()) - just
       re-postprocess what's already on disk.
    2. NOT yet downloaded, but genuinely finished on Azure - checked via
       azure_job_is_finished()/azure_task_succeeded() (the same
       functions the live polling loop uses), downloaded now via
       fetch_azure_result(), then treated the same as path 1. This
       closes the gap flagged in earlier versions of this function:
       previously, a job that completed remotely but was never
       downloaded before a crash was silently unrecoverable - genuine
       paid-for compute, lost. A failed/still-running job is correctly
       left unrecovered either way.

    Jobs are recovered if they were submitted under EITHER the CURRENT
    commit, or a commit listed in VALID_PRIOR_RUN_COMMITS
    (optimisation_parameters.py) - mirroring exactly how
    find_checkpoint_commit_for_seed() treats VALID_CHECKPOINT_COMMITS for
    checkpoints. A run submitted under some OTHER, unlisted commit may
    have used a different smac_scenario.py (different draw_parameters
    mapping, different modules, etc.), so silently folding its
    DALYs/cost into history would risk mixing results that aren't
    actually comparable - such jobs are skipped, with a warning, rather
    than loaded. An earlier version of this function only ever accepted
    the current commit, with no way to explicitly vet and reuse results
    from an older, still-trusted commit.

    DELIBERATELY recovers EVERY previously-submitted, still-valid-commit
    job on EVERY call, with NO check against what's already in
    HISTORY_LOG_FILE - even jobs that were already successfully
    recorded in a PREVIOUS run of this process, before some restart.
    This looks like it should cause duplication, but doesn't: `history`
    (in-memory) and SMAC's own runhistory are both PROCESS-LOCAL state
    with no persistence of their own - recover_from_job_log() ->
    seed_history_with_prior_runs() -> smac.tell() is the ONLY way a
    previously-completed trial ever reaches a FRESH process's SMAC
    instance again after a restart, so skipping already-recorded jobs
    HERE would leave a freshly-restarted SMAC with no memory of them at
    all, free to re-propose the same or similar configs it had already
    tried before the restart - genuine wasted compute, not just a
    cosmetic issue. An earlier version of this function DID skip
    already-recorded jobs here, based on a real observation (duplicate
    entries in a real history_log.jsonl) - but that fix was applied at
    the wrong layer: it stopped the FILE duplicating, at the cost of
    also stopping SMAC from being told about those trials again. The
    correct fix lives in convergence_monitoring.append_history_to_file()
    instead - deduplicating the FILE WRITE specifically, which has no
    effect on whether smac.tell() gets called, letting this function
    freely re-process everything on every restart exactly as before
    that history_log.jsonl-duplication bug was ever found.
    """
    if not JOB_LOG_FILE.exists():
        return []

    tlo_config = _get_config()
    username = tlo_config["DEFAULT"]["USERNAME"]
    current_commit = _get_commit()
    candidate_commits = {current_commit, *VALID_PRIOR_RUN_COMMITS}

    recovered = []
    with open(JOB_LOG_FILE) as f:
        for line in f:
            record = json.loads(line)

            job_commit = record.get("commit")

            if job_commit not in candidate_commits:
                print(
                    f"[warning] skipping recovered job {record['job_id']} - "
                    f"submitted under commit {(job_commit or 'unknown')[:12]}, which is "
                    f"neither the current commit ({current_commit[:12]}) nor listed in "
                    f"VALID_PRIOR_RUN_COMMITS"
                )
                continue

            draw_dir = Path("outputs", username, record["job_id"], "0")

            if not draw_dir.exists() or not any(draw_dir.iterdir()):
                # not downloaded yet - check whether it's actually
                # finished on Azure before giving up on it
                job = AzureJobHandle(
                    job_id=record["job_id"], submitted_at=0.0, commit_hexsha=job_commit,
                    # commit_hexsha is this JOB's OWN recorded commit, not
                    # necessarily current_commit (an earlier version of
                    # this line hardcoded current_commit here regardless
                    # - harmless in practice, since nothing downstream of
                    # this AzureJobHandle actually reads commit_hexsha
                    # for API calls, only job_id, but factually wrong for
                    # a job recovered under a VALID_PRIOR_RUN_COMMITS entry).
                )
                if not azure_job_is_finished(job):
                    continue  # still running - genuinely not recoverable yet
                if not azure_task_succeeded(job):
                    print(f"[warning] recovered job {record['job_id']} failed on Azure - skipping.")
                    continue
                print(f"[recovering] job {record['job_id']} finished on Azure but was never downloaded - fetching now.")
                draw_dir = download_run_outputs(job)

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
            "config": {"config_annual_testing_rate_adults": 0.8, "annual_rate_selftest": 0.3, ...},
            "dalys": 41.2,
            "hiv_hrh_cost_by_year": {2025: 12000.0, 2026: 12500.0, ...},
            "hiv_consumable_cost_by_year": {2025: 8000.0, 2026: 8100.0, ...},
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
# 2b. Checkpoint-generation and baseline-run SUBMISSION now happen much
#     earlier (see right before the initialise.py import above) - baseline
#     results must be available to update COST_LIMITS_FILE BEFORE
#     initialise.py reads it at import time. Checkpoint WAITING
#     (ensure_checkpoints_ready()) has moved up there too now, right after
#     generate_all_checkpoints() itself - it has no COST_LIMITS_FILE
#     ordering constraint of its own (it never touches that file), but DOES
#     need to happen before submit_initial_design_jobs() (also up there),
#     which submits real, suspend/resume-using trials of its own.
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# 3. Build SMAC in ask-tell mode (n_trials still needed for its budget
#    bookkeeping, even though you're driving the loop yourself)
# --------------------------------------------------------------------------

scenario = Scenario(configspace, n_trials=N_TRIALS, deterministic=False)  # total trial budget -
                                                                        # remember max_config_calls
                                                                        # means this isn't the same
                                                                        # as "number of distinct
                                                                        # configs explored"

# max_config_calls caps how many seeds the intensifier will use to
# confirm any single config (default is 3). Set (via
# optimisation_parameters.MAX_CONFIG_CALLS) to match the "N seeds for
# true convergence" threshold - a config can still be discarded early on
# fewer seeds if it's clearly uncompetitive, but nothing gets promoted
# to incumbent-quality trust on fewer than MAX_CONFIG_CALLS
# confirmations. Combined with the noisy-EI correction in ConstrainedEI,
# this makes the adaptive sampling trustworthy rather than just
# efficient. SAME value checkpoint_seeds.py uses for CHECKPOINT_SEEDS -
# see optimisation_parameters.py's own comment on why these must match.
intensifier = HyperparameterOptimizationFacade.get_intensifier(scenario, max_config_calls=MAX_CONFIG_CALLS)

acquisition_function = ConstrainedEI(
    configspace=configspace,
    objective_name="dalys",
    constraint_names=CONSTRAINT_NAMES,
    history_provider=lambda: history,
    xi=EI_XI,
    retrain_every=RETRAIN_EVERY,  # refit on every new result by default -
                       # refitting is cheap relative to simulation cost,
                       # so there's no reason to tolerate staleness
                       # (see earlier discussion)
    min_samples_leaf=MIN_SAMPLES_LEAF,  # an earlier version of this
                       # call never passed this at all, silently relying
                       # on MultiSurrogateModel's own hardcoded default
                       # deep inside constrained_ei.py, with no way to
                       # actually tune it from here
    alpha=MERIT_PENALTY_ALPHA,      # scales the additive merit term
                       # (sum_j pi_j(x)) into the objective's own DALYs
                       # units - see optimisation_parameters.py's own
                       # description of MERIT_PENALTY_ALPHA
    tau=MERIT_VIOLATION_THRESHOLD,  # tolerance on each constraint's
                       # predicted violation probability - see
                       # optimisation_parameters.py's own description of
                       # MERIT_VIOLATION_THRESHOLD
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
#    PRIOR_RUNS itself lives in initialise.py (imported at the top of
#    this file), alongside the constraint setup - edit it there.
# --------------------------------------------------------------------------

recovered_runs = recover_from_job_log()
all_prior_runs = PRIOR_RUNS + recovered_runs
n_seeded = seed_history_with_prior_runs(all_prior_runs, smac)
print(f"Warm-started with {n_seeded} run(s): {len(PRIOR_RUNS)} manual, {len(recovered_runs)} recovered from job log")


# --------------------------------------------------------------------------
# 5. The ask-tell loop itself: keep N_CONCURRENT Azure jobs in flight,
#    ask() for a replacement each time one completes and is told back.
#    N_CONCURRENT / POLL_INTERVAL_SECONDS now live in
#    optimisation_parameters.py (imported at the top of this file).
# --------------------------------------------------------------------------

def _safe_ask(max_retries: int = 3, retry_delay_seconds: float = 5.0) -> TrialInfo:
    """
    Wraps smac.ask() so a failure INSIDE SMAC's own internals doesn't
    take down the whole process the way a bare smac.ask() call does.

    BACKGROUND: a real, repeated crash chain was observed where a
    postprocessing failure (missing 'tlo.methods.healthburden' logger,
    or a stale-FileHandler dill UnpicklingError - see
    postprocess_output.py) is correctly caught and told to SMAC as
    TrialValue(cost=np.inf, status=StatusType.CRASHED) - but the VERY
    NEXT smac.ask() call then crashes with
    "ValueError: Input y contains NaN" deep inside SMAC's own default
    surrogate model (built internally by HyperparameterOptimizationFacade
    since no explicit model=/runhistory_encoder= override is passed
    here - separate from, and unrelated to, ConstrainedEI's own RF
    surrogate, which is never involved: record_result(), the only thing
    that appends to `history`/feeds ConstrainedEI, is never called on
    the CRASHED path). Confirmed by direct tracing that this is NOT a
    ConstrainedEI/history problem - the exact internal SMAC mechanism
    (most likely objective-bound normalization degenerating when too
    few SUCCESS-status trials exist in runhistory at that moment) has
    NOT been independently verified against the installed SMAC version,
    so this is a defensive wrapper, not a real fix for the root cause.

    Strategy: retry a few times with a short delay first (in case the
    failure is transient - e.g. another in-flight tell() lands and
    gives the encoder enough SUCCESS data to recover) - then, if it's
    still failing, fall back to a MANUALLY sampled config (uniform from
    configspace, exactly like configspace.sample_configuration()) with
    a hand-built TrialInfo, bypassing SMAC's ask() entirely for this one
    trial. This keeps the pipeline (and any jobs already in flight)
    alive rather than losing the whole run to one bad trial - the
    manually-sampled trial is still submitted, postprocessed, and
    told back to SMAC exactly like any other, so SMAC's own runhistory
    stays complete; it just wasn't SMAC's own acquisition function that
    chose it this one time. Uses CHECKPOINT_SEEDS[0] as the fallback
    seed - the same convention submit_initial_design_jobs() uses - so a
    fallback trial can still genuinely use suspend/resume and stays
    comparable to anything else sharing that seed.
    """
    last_exception: Exception | None = None
    for attempt in range(max_retries):
        try:
            return smac.ask()
        except Exception as e:
            last_exception = e
            print(
                f"[ask failed] smac.ask() raised {e!r} (attempt {attempt + 1}/{max_retries}) - "
                f"retrying in {retry_delay_seconds}s."
            )
            time.sleep(retry_delay_seconds)
    print(
        f"[ask failed] smac.ask() still failing after {max_retries} attempts "
        f"({last_exception!r}) - falling back to a manually-sampled config for "
        f"this one trial rather than crashing the whole pipeline."
    )
    fallback_config = configspace.sample_configuration()
    return TrialInfo(config=fallback_config, seed=CHECKPOINT_SEEDS[0])


pending: list[tuple[TrialInfo, AzureJobHandle]] = []

# prime the pipeline
for _ in range(N_CONCURRENT):
    info = _safe_ask()
    pending.append((info, submit_azure_job(info.config, info.seed)))

n_completed = 0  # N_TRIALS is a budget for NEW trials only - warm-started
    # runs (recovered, PRIOR_RUNS, N_INIT) don't count against it,
    # regardless of n_seeded.
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
        new_info = _safe_ask()
        still_pending.append((new_info, submit_azure_job(new_info.config, new_info.seed)))


def _drain_pending(pending: list) -> None:
    """
    Waits for every job still in `pending` to finish on Azure and
    records each (record_result()/smac.tell(), or CRASHED on failure) -
    same per-job handling as the main loop, but proposes NO new jobs.
    Called from the loop's own except clause below, so an uncaught
    exception or Ctrl+C still drains outstanding jobs into history_log
    before the process actually exits, rather than abandoning them.
    A failure resolving any individual job here is logged and skipped
    (not retried indefinitely) - it remains recoverable on a future
    restart via recover_from_job_log().
    """
    print(f"[shutdown] draining {len(pending)} still-pending job(s) before exiting...")
    still_waiting = list(pending)
    while still_waiting:
        next_round = []
        for info, job in still_waiting:
            try:
                if not azure_job_is_finished(job):
                    next_round.append((info, job))
                    continue
                if not azure_task_succeeded(job):
                    print(f"[shutdown] {job.job_id} failed - recording as CRASHED.")
                    smac.tell(info, TrialValue(cost=np.inf, status=StatusType.CRASHED))
                    continue
                result = fetch_azure_result(job)
                value = record_result(info.config, info.seed, result, job_id=job.job_id, commit=job.commit_hexsha)
                smac.tell(info, value)
                print(f"[shutdown] {job.job_id} recorded.")
            except Exception as e:
                print(
                    f"[shutdown] WARNING: could not resolve {job.job_id} ({e!r}) - "
                    f"leaving it for a future restart's recover_from_job_log() to pick up."
                )
        if next_round:
            still_waiting = next_round
            time.sleep(POLL_INTERVAL_SECONDS)
        else:
            still_waiting = []
    print("[shutdown] all pending jobs resolved.")


try:
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
except (Exception, KeyboardInterrupt) as e:
    # Covers BOTH an uncaught exception from a call this loop doesn't
    # already wrap (azure_job_is_finished/azure_task_succeeded/smac.ask/
    # smac.tell itself - none of these are retried, unlike the
    # postprocessing try/except above) AND a manual Ctrl+C - either way,
    # `pending` may still hold jobs genuinely running on Azure right now.
    # Without this, the process would exit immediately, leaving them
    # untracked by THIS run (still recoverable later, but only via a
    # future restart - not what was actually asked for: this run should
    # itself wait for them).
    print(f"[shutdown] main loop stopped ({e!r}) - {len(pending)} job(s) still pending.")
    _drain_pending(pending)
    raise



# --------------------------------------------------------------------------
# 6. Final answer: group history by config and take the MEDIAN across
#    whatever seeds SMAC ended up requesting for it, THEN filter+select -
#    never trust smac.incumbent directly, and never trust a single noisy
#    realisation's DALYs either, now that history holds individual
#    (config, seed) results rather than pre-averaged bundles. MEDIAN, not
#    mean, for consistency across this pipeline's own DALYs/cost
#    aggregation - see aggregate_postprocessed_results() and
#    convergence_monitoring.get_best_feasible_dalys() for the same
#    convention applied elsewhere.
# --------------------------------------------------------------------------

grouped: dict[tuple, list[dict]] = {}
for h in history:
    grouped.setdefault(config_key(h["config_object"]), []).append(h)

aggregated = []
for entries in grouped.values():
    agg = {
        "config_object": entries[0]["config_object"],
        "n_seeds_evaluated": len(entries),
        "dalys": float(np.median([e["dalys"] for e in entries])),
    }
    for name in CONSTRAINT_NAMES:
        agg[name] = float(np.median([e[name] for e in entries]))
    aggregated.append(agg)

feasible = [a for a in aggregated if all(a[name] == 0 for name in CONSTRAINT_NAMES)]
best = min(feasible, key=lambda a: a["dalys"])
print(
    "Best feasible config:", dict(best["config_object"]),
    "DALYs:", best["dalys"], f"(averaged over {best['n_seeds_evaluated']} seed(s))",
)
