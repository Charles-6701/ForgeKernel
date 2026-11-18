"""GPU jobs: one detached runner per job, its state in plain files.

``submit`` allocates a job id, detaches a runner process and prints ``job=<id>``; the
runner runs one gpu_entry task (forgekernel.submit.direct: this machine, or one ssh host)
and writes the job's log and, on every exit path, its verdict. ``status`` / ``logs`` /
``stop`` read and act on those files, so the loop can poll a job without a server, and a
job outlives the process that submitted it (a restarted loop adopts it).

State lives in ``operators/<op>/.jobs/<id>/``:

* ``log.txt``     -- what gpu_entry printed (closed before the verdict is written, so
  the log of a finished job is whole);
* ``result.json`` -- ``{"id": <id>, "status": "Succeeded" | "Failed"}``;
* ``runner.pid``  -- the runner's pid (a session leader: stopping it stops its group);
* ``runner.log``  -- the runner process's own output (a crash before the log opens).

A job id is the submit time in epoch seconds, bumped past collisions.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from forgekernel.submit.config import SubmitConfig

JOBS_DIRNAME = ".jobs"
LOCK_NAME = "lock"

STATUS_RUNNING = "Running"
STATUS_SUCCEEDED = "Succeeded"
STATUS_FAILED = "Failed"
# The runner is gone and left no verdict (a reboot, an OOM kill): nothing says the job
# ran, so the loop resubmits it rather than charging the kernel.
STATUS_LOST = "Lost"

# The first line a job's command prints on the GPU machine: everything before it is
# transport (the copy, the ssh session), everything after it is the job.
BEGIN_MARKER = "FK_JOB_BEGIN"
# The job never reached the GPU machine: the copy or the ssh session failed every
# attempt. Not the kernel's failure -- the loop reports it as a transport problem.
TRANSPORT_ERROR_MARKER = "[jobs] TRANSPORT ERROR (the job could not be delivered):"
# A leaderboard card's gpu_entry.sh refuses to time when the board's clock lock does not
# hold on the GPU. Matched at the START of a line only: a probe that runs a nested
# gpu_entry prints it indented.
CLOCK_LOCK_REFUSED_MARKER = "[gpu_entry.sh] REFUSED: this card times under the board's clock lock"

# How long past the task's own timeout the runner waits before killing it.
RUN_HEADROOM_S = 180


def clock_lock_refusal(log: str | None) -> str | None:
    """The outer gpu_entry's clock-lock refusal line in ``log`` (stripped), else None."""
    for line in (log or "").splitlines():
        if line.startswith(CLOCK_LOCK_REFUSED_MARKER):
            return line.strip()
    return None


def jobs_dir(op_dir: Path) -> Path:
    return Path(op_dir) / JOBS_DIRNAME


def job_dir(op_dir: Path, job_id: str) -> Path:
    return jobs_dir(op_dir) / str(job_id)


def result_path(op_dir: Path, job_id: str) -> Path:
    return job_dir(op_dir, job_id) / "result.json"


def log_path(op_dir: Path, job_id: str) -> Path:
    return job_dir(op_dir, job_id) / "log.txt"


def pid_path(op_dir: Path, job_id: str) -> Path:
    return job_dir(op_dir, job_id) / "runner.pid"


def is_job(op_dir: Path, job_id: str) -> bool:
    return str(job_id).isdigit() and job_dir(op_dir, job_id).is_dir()


def alloc_job_id(op_dir: Path, now: float | None = None) -> str:
    """Epoch-seconds job id, bumped past collisions."""
    jid = int(now if now is not None else time.time())
    base = jobs_dir(op_dir)
    base.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            (base / str(jid)).mkdir()
            return str(jid)
        except FileExistsError:
            jid += 1


@contextlib.contextmanager
def locked(op_dir: Path):
    """One exclusive lock per operator tree: its jobs run one at a time (they share the
    tree's GPU, and the extracted tree)."""
    d = jobs_dir(op_dir)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / LOCK_NAME, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def job_status(op_dir: Path, job_id: str) -> dict:
    """``{"id", "status"}`` for a job: its verdict once written, Running while its runner
    lives, Lost when the runner is gone without one."""
    res = result_path(op_dir, job_id)
    if res.is_file():
        try:
            return json.loads(res.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"id": int(job_id), "status": STATUS_FAILED}
    try:
        pid = int(pid_path(op_dir, job_id).read_text().strip())
    except (OSError, ValueError):
        pid = 0
    if pid and _pid_alive(pid):
        return {"id": int(job_id), "status": STATUS_RUNNING}
    return {"id": int(job_id), "status": STATUS_LOST}


def wait_job(op_dir: Path, job_id: str, *, timeout_s: float,
             check_s: float = 1.0, heartbeat_s: float = 0.0,
             on_heartbeat=None, monotonic=time.monotonic, sleep=time.sleep) -> dict:
    """Block locally until a detached job writes its terminal verdict.

    The durable completion signal is ``result.json``.  Waiting here keeps the
    submitter/model out of a status-query loop: the only repeated work is a cheap
    local file/process check inside this process.  ``heartbeat_s`` is optional and
    deliberately independent of ``check_s`` so a long JIT can remain observable
    without emitting every check.  Injected clock/sleep make the deadline testable.

    ``Lost`` is terminal too: the runner disappeared without producing a verdict and
    the caller decides whether to resubmit it.
    """
    if timeout_s < 0:
        raise ValueError("timeout_s must be non-negative")
    if check_s <= 0:
        raise ValueError("check_s must be positive")
    started = monotonic()
    deadline = started + timeout_s
    next_heartbeat = (started + heartbeat_s) if heartbeat_s > 0 else None
    while True:
        status = job_status(op_dir, job_id)
        if status.get("status") != STATUS_RUNNING:
            return status
        now = monotonic()
        if now >= deadline:
            raise TimeoutError(
                f"job {job_id} did not reach a terminal state within {timeout_s:.0f}s "
                f"(last status={status.get('status')!r})")
        if next_heartbeat is not None and now >= next_heartbeat:
            if on_heartbeat is not None:
                on_heartbeat(status, now - started)
            # Skip missed intervals after a delayed wake instead of printing a burst.
            next_heartbeat = now + heartbeat_s
        sleep(min(max(0.0, check_s), max(0.0, deadline - now)))


def job_log(op_dir: Path, job_id: str) -> str:
    try:
        return log_path(op_dir, job_id).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def stop_job(op_dir: Path, job_id: str) -> bool:
    """SIGTERM a running job's runner group (the runner records it as Failed). True when a
    signal was sent."""
    if job_status(op_dir, job_id).get("status") != STATUS_RUNNING:
        return False
    try:
        pid = int(pid_path(op_dir, job_id).read_text().strip())
        # the runner is a session leader, so its pid is the group: the ssh session and
        # gpu_entry go with it
        os.killpg(pid, signal.SIGTERM)
    except (OSError, ValueError):
        return False
    return True


def spawn_runner(cfg: SubmitConfig, mode: str, branch: str) -> str:
    """Detach a runner for one mode of ``branch``; returns the new job id.

    The runner re-execs ``forgekernel.submit <op> _run`` in its own session so it
    survives the submitting process (the loop submits and then polls)."""
    op_dir = Path(cfg.op_dir)
    jid = alloc_job_id(op_dir)
    with open(job_dir(op_dir, jid) / "runner.log", "w") as runner_log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "forgekernel.submit", op_dir.name, "_run", jid,
             "--mode", mode, "--branch", branch, "--base-dir", str(op_dir.parents[1])],
            stdout=runner_log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True)
    pid_path(op_dir, jid).write_text(str(proc.pid), encoding="utf-8")
    return jid


def runner_main(cfg: SubmitConfig, job_id: str, mode: str, branch: str) -> int:
    """The detached runner: one mode, its verdict to result.json on every exit path."""
    from forgekernel.submit.direct import DirectRunner

    op_dir = Path(cfg.op_dir)
    status = STATUS_FAILED
    log_path(op_dir, job_id).parent.mkdir(parents=True, exist_ok=True)
    # SIGTERM (`submit stop`) must become SystemExit so the finally below still writes a
    # verdict: a stopped job with none would read back as Lost and be resubmitted.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        with open(log_path(op_dir, job_id), "a", encoding="utf-8") as lf:
            try:
                status = DirectRunner(cfg).run_mode(branch, mode, lf)
            except SystemExit:
                lf.write("[jobs] runner stopped (SIGTERM) -- recorded as Failed so nothing "
                         "resubmits it\n")
                raise
            except Exception as e:  # noqa: BLE001 -- the verdict must still be written
                lf.write(f"[jobs] runner crashed: {e!r}\n")
                status = STATUS_FAILED
    finally:
        res = result_path(op_dir, job_id)
        tmp = res.with_suffix(".tmp")
        tmp.write_text(json.dumps({"id": int(job_id), "status": status}), encoding="utf-8")
        tmp.replace(res)
    return 0 if status == STATUS_SUCCEEDED else 1
