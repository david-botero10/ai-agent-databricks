"""Runs the agent quality gate on Databricks and exits non-zero when it does not pass.

Uploads the quality_gate notebook (plus what its %run cells and imports need) to a per-run
workspace folder, submits a serverless one-time run with the prompt and thresholds as
parameters, waits for it, prints the notebook output or error, and exits 1 unless SUCCESS.

Environment:
  DATABRICKS_HOST, DATABRICKS_TOKEN  required (read by databricks-sdk)
  MIN_CORRECTNESS, MIN_SAFETY        optional thresholds (default 0.7 and 1.0)
  GITHUB_SHA                         commit under test (default 'local')
  CI_KEEP_WORKSPACE=1                keep the uploaded workspace folder after the run
"""

import base64
import os
import sys
import time
from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import compute, jobs
from databricks.sdk.service.workspace import ImportFormat, Language

# Serverless job environment for the gate notebook (same packages as the %pip cell of 02.1).
DEPENDENCIES = [
    "mlflow>=3.10.1",
    "databricks-langchain",
    "databricks-agents",
    "unitycatalog-langchain[databricks]",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_DIR = REPO_ROOT / "ai-agent"
PROMPT_FILE = PROJECT_DIR / "prompts" / "system_prompt.txt"

# (local path relative to ai-agent/, imported as notebook?) -> same relative path in the workspace.
# Notebooks are imported without the .py extension; agent.py must stay a real file to be importable.
UPLOADS = [
    ("02-agent-eval/quality_gate.py", True),
    ("02-agent-eval/agent.py", False),
    ("config.py", True),                        # %run ../config
    ("_resources/00-global-setup-v2.py", True),  # %run ../_resources/00-global-setup-v2
]
GATE_NOTEBOOK = "02-agent-eval/quality_gate"

RUN_TIMEOUT_S = 50 * 60  # below the 60 min timeout of the GitHub job
POLL_INTERVAL_S = 15
TERMINAL_STATES = {
    jobs.RunLifeCycleState.TERMINATED,
    jobs.RunLifeCycleState.SKIPPED,
    jobs.RunLifeCycleState.INTERNAL_ERROR,
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def upload(w, base):
    """Imports every file of UPLOADS under `base`, preserving relative paths."""
    for rel, as_notebook in UPLOADS:
        src = PROJECT_DIR / rel
        if not src.is_file():
            raise FileNotFoundError(f"Missing file to upload: {src}")
        target = f"{base}/{rel[:-3]}" if as_notebook else f"{base}/{rel}"
        w.workspace.mkdirs(target.rsplit("/", 1)[0])
        content = base64.b64encode(src.read_bytes()).decode()
        if as_notebook:
            w.workspace.import_(target, content=content, format=ImportFormat.SOURCE,
                                language=Language.PYTHON, overwrite=True)
        else:
            w.workspace.import_(target, content=content, format=ImportFormat.AUTO, overwrite=True)
        log(f"uploaded {rel} -> {target}")


def wait_for_run(w, run_id):
    """Polls the run until it reaches a terminal state; cancels it on timeout."""
    deadline = time.time() + RUN_TIMEOUT_S
    last_state = None
    while True:
        run = w.jobs.get_run(run_id)
        state = run.state.life_cycle_state if run.state else None
        if state != last_state:
            log(f"run state: {state.value if state else 'UNKNOWN'}")
            last_state = state
        if state in TERMINAL_STATES:
            return run
        if time.time() > deadline:
            log(f"timeout after {RUN_TIMEOUT_S}s, cancelling run {run_id}")
            w.jobs.cancel_run(run_id)
            return w.jobs.get_run(run_id)
        time.sleep(POLL_INTERVAL_S)


def print_output(w, run):
    """Prints the notebook exit value, the error and the tail of the logs of the gate task."""
    if not run.tasks:
        return
    out = w.jobs.get_run_output(run.tasks[0].run_id)
    if out.notebook_output and out.notebook_output.result:
        print("--- notebook output ---")
        print(out.notebook_output.result)
    if out.error:
        print("--- error ---")
        print(out.error)
    if out.error_trace:
        print("--- error trace (tail) ---")
        print(out.error_trace[-6000:])
    if out.logs:
        print("--- logs (tail) ---")
        print(out.logs[-6000:])


def main():
    missing = [v for v in ("DATABRICKS_HOST", "DATABRICKS_TOKEN") if not os.environ.get(v)]
    if missing:
        print(f"ERROR: missing environment variables: {', '.join(missing)}", file=sys.stderr)
        return 1

    system_prompt = PROMPT_FILE.read_text(encoding="utf-8").strip()
    if not system_prompt:
        print(f"ERROR: {PROMPT_FILE} is empty", file=sys.stderr)
        return 1
    min_correctness = float(os.environ.get("MIN_CORRECTNESS") or "0.7")
    min_safety = float(os.environ.get("MIN_SAFETY") or "1.0")
    git_sha = os.environ.get("GITHUB_SHA") or "local"

    w = WorkspaceClient()
    user = w.current_user.me().user_name
    ci_root = f"/Users/{user}/ci"
    base = f"{ci_root}/{git_sha}"
    log(f"workspace folder: {base}")

    try:
        w.workspace.delete(base, recursive=True)  # clean leftovers from a previous attempt on this commit
    except Exception:
        pass
    upload(w, base)

    run_id = None
    try:
        submitted = w.jobs.submit(
            run_name=f"quality_gate-{git_sha[:12]}",
            tasks=[
                jobs.SubmitTask(
                    task_key="quality_gate",
                    notebook_task=jobs.NotebookTask(
                        notebook_path=f"{base}/{GATE_NOTEBOOK}",
                        base_parameters={
                            "system_prompt": system_prompt,
                            "min_correctness": str(min_correctness),
                            "min_safety": str(min_safety),
                            "experiment_path": f"{ci_root}/quality_gate_experiment",
                            "git_sha": git_sha,
                        },
                    ),
                    environment_key="gate",
                    timeout_seconds=RUN_TIMEOUT_S,
                )
            ],
            environments=[
                jobs.JobEnvironment(
                    environment_key="gate",
                    spec=compute.Environment(client="2", dependencies=DEPENDENCIES),
                )
            ],
        )
        run_id = submitted.run_id
        log(f"submitted run_id={run_id} | thresholds correctness>={min_correctness} safety>={min_safety}")

        run = wait_for_run(w, run_id)
        if run.run_page_url:
            log(f"run page: {run.run_page_url}")
        print_output(w, run)

        result = run.state.result_state if run.state else None
        log(f"result: {result.value if result else 'UNKNOWN'}")
        return 0 if result == jobs.RunResultState.SUCCESS else 1
    except KeyboardInterrupt:
        # GitHub cancels the job with SIGINT: do not leave the evaluation running on Databricks.
        if run_id is not None:
            log(f"interrupted, cancelling run {run_id}")
            w.jobs.cancel_run(run_id)
        return 1
    finally:
        if os.environ.get("CI_KEEP_WORKSPACE") != "1":
            try:
                w.workspace.delete(base, recursive=True)
            except Exception as e:
                log(f"could not delete {base}: {e}")


if __name__ == "__main__":
    sys.exit(main())
