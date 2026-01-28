"""Shared utilities for the bilevel pipeline."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


def log(msg: str) -> None:
    print(f"[bilevel] {msg}")


def run_bash(command: str, cwd: Optional[Path] = None) -> None:
    printable = command.replace("\n", " ").strip()
    log(f"run: {printable}")
    result = subprocess.run(
        ["bash", "-lc", command],
        cwd=str(cwd) if cwd else None,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {printable}")


# ---------------------------------------------------------------------------
# Config accessors
# ---------------------------------------------------------------------------
def detector_cfg(config: dict) -> dict:
    label = config["outer_loop"]["detector_label"]
    return config["detectors"][label]


def runs_dir(config: dict) -> Path:
    d = Path(config["runtime"]["runs_dir"]).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def output_dir(config: dict) -> Path:
    d = runs_dir(config) / "sim_outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir(config: dict) -> Path:
    d = runs_dir(config) / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def manifest_path(config: dict) -> Path:
    """Path to the manifest JSON written by the outer loop."""
    return output_dir(config) / "manifest.json"


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
def env_setup_cmd(config: dict) -> str:
    """Return the shell command that sets up the runtime environment."""
    return config["key4hep"]["setup_script"]


def setup_env(config: dict) -> None:
    """Validate that the runtime environment can be loaded."""
    if not config["key4hep"]["run_setup"]:
        return
    run_bash(f"{env_setup_cmd(config)} && echo 'env OK'")


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------
def submit_jobs(rd: Path) -> None:
    """Submit all .slurm files found in rd/slurm/ via sbatch."""
    slurm_files = sorted((rd / "slurm").glob("*.slurm"))
    if not slurm_files:
        log("No slurm files to submit.")
        return
    for slurm_file in slurm_files:
        run_bash(f"sbatch {slurm_file}", cwd=slurm_file.parent)


def run_local(script: Path) -> None:
    """Run the generated run_all.sh locally."""
    log(f"Running locally: {script}")
    run_bash(f"bash {script}", cwd=script.parent)
