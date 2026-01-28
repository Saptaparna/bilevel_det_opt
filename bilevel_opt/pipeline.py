"""
One-click bilevel detector optimization pipeline.

Usage:
    python pipeline.py --config config.yaml [options]

Stages:
    outer  - Source key4hep, generate geometry-scan runs, submit SLURM jobs.
    inner  - Run inner-loop optimization on ROOT outputs from the geometry scan.
    all    - Run both stages sequentially (default).

The outer loop produces ROOT files. Set run_mode in config (or --run-mode on CLI):
    local     - run ddsim jobs sequentially in the current shell.
    slurm     - submit SLURM batch jobs.
    generate  - only write run scripts (default).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:
    sys.exit(f"PyYAML is required. Install with `pip install pyyaml` ({exc})")

from bilevel_opt.outerloop import generate_runs
from bilevel_opt.util import manifest_path, results_dir, run_local, runs_dir, setup_env, submit_jobs
from bilevel_opt.innerloop import run_inner_loop


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Bilevel detector optimization: one-click pipeline."
    )
    p.add_argument("--config", required=True, help="YAML config path.")
    p.add_argument("--run", default=None, help="Run only the named run (default: all runs).")
    p.add_argument(
        "--stage",
        choices=["outer", "inner", "all"],
        default="all",
        help="Which stage(s) to run (default: all).",
    )

    outer = p.add_argument_group("outer loop")
    outer.add_argument("--run-mode", choices=["local", "slurm", "generate"], default=None,
                       help="Override config run_mode (local | slurm | generate).")

    inner = p.add_argument_group("inner loop")
    inner.add_argument("--manifest", default=None,
                       help="Override path to manifest.json (default: from outer loop).")
    inner.add_argument("--max-events", type=int, default=None,
                       help="Override max_events from config.")
    inner.add_argument("--outdir", default=None,
                       help="Override inner-loop output directory.")

    return p


def merge_run_config(global_config: dict, run_block: dict) -> dict:
    """Merge a run block's outer_loop/inner_loop into global config.

    Paths are scoped: runs/<name>/sim_outputs and runs/<name>/results.
    """
    name = run_block["name"]
    merged = {**global_config}
    merged["outer_loop"] = run_block["outer_loop"]
    merged["inner_loop"] = run_block["inner_loop"]
    merged["runtime"] = {
        **global_config["runtime"],
        "runs_dir": str(Path(global_config["runtime"]["runs_dir"]) / name),
    }
    return merged


def run_outer(config: dict, args: argparse.Namespace) -> str:
    """Run all outer-loop steps. Returns the manifest path for the inner loop."""
    mode = args.run_mode or config["run_mode"]
    generate_runs(config, run_mode=mode)
    if mode == "local":
        run_local(runs_dir(config) / "run_all.sh")
    elif mode == "slurm":
        submit_jobs(runs_dir(config))
    return str(manifest_path(config))


def run_inner(config: dict, args: argparse.Namespace, mf: str) -> dict:
    """Run the inner loop on ROOT files listed in the manifest."""
    il_cfg = dict(config["inner_loop"])
    if args.max_events is not None:
        il_cfg["max_events"] = args.max_events
    outdir = args.outdir or str(results_dir(config))
    return run_inner_loop(manifest_file=mf, il_cfg=il_cfg, outdir=outdir)


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser()
    if not cfg_path.exists():
        sys.exit(f"Config file not found: {cfg_path}")
    global_config = yaml.safe_load(cfg_path.read_text())

    setup_env(global_config)

    run_blocks = global_config["runs"]
    if args.run:
        run_blocks = [r for r in run_blocks if r["name"] == args.run]
        if not run_blocks:
            names = [r["name"] for r in global_config["runs"]]
            sys.exit(f"Run '{args.run}' not found. Available: {names}")

    for run_block in run_blocks:
        run_name = run_block["name"]
        config = merge_run_config(global_config, run_block)

        print()
        print("#" * 60)
        print(f"  RUN: {run_name}")
        print("#" * 60)

        mf = args.manifest

        if args.stage in ("outer", "all"):
            print("=" * 60)
            print(f"  OUTER LOOP: geometry sweep generation [{run_name}]")
            print("=" * 60)
            mf_from_outer = run_outer(config, args)
            if mf is None:
                mf = mf_from_outer

        if args.stage in ("inner", "all"):
            if mf is None:
                mf = str(manifest_path(config))
            print()
            print("=" * 60)
            print(f"  INNER LOOP: algorithm parameter optimization [{run_name}]")
            print("=" * 60)
            summary = run_inner(config, args, mf)

            best = summary["global_best"]
            gv = ", ".join(f"{k}={v}" for k, v in best["geom_values"].items())
            bp = ", ".join(f"{k}={v:.4g}" for k, v in best["best_params"].items())
            print()
            print("=" * 60)
            print(f"  RESULT [{run_name}]: {gv}, {bp}, "
                  f"score={best['best_score']:.6g}")
            print("=" * 60)


if __name__ == "__main__":
    main()
