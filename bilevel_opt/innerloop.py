#!/usr/bin/env python3
"""
Inner loop: for each geometry variation (ROOT file), optimize reconstruction
algorithm parameters and produce the bilevel loss landscape.

Supports arbitrary numbers of algorithm parameters via N-D optimization.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import (
    minimize, minimize_scalar, differential_evolution,
    dual_annealing, shgo, brute,
)

import ROOT
from ROOT import TFile

from bilevel_opt.algorithms import ALGORITHMS, SCORES
from bilevel_opt.edm import HIT_TYPES


@dataclass
class GeometryResult:
    geom_values: dict       # {geom_param: float, ...}
    root_path: str
    best_params: dict       # {algo_param: float, ...}
    best_score: float


# ---------------------------------------------------------------------------
# ROOT I/O
# ---------------------------------------------------------------------------
def load_manifest(manifest_path: str) -> List[dict]:
    """Read the manifest JSON written by the outer loop."""
    with open(manifest_path) as f:
        entries = json.load(f)
    entries.sort(key=lambda e: tuple(sorted(e["geom_values"].items())))
    return entries


def load_events(
    path: str,
    tree_name: str,
    branches: List[dict],
    max_events: Optional[int],
) -> List[dict]:
    """Load events from a ROOT file, reading multiple branches.

    branches is a list of {"name": str, "hit_type": str}.
    Returns one dict per tree entry, mapping branch name to its hit collection.
    """
    branch_info = [(b["name"], *HIT_TYPES[b["hit_type"]]) for b in branches]
    f = TFile.Open(path)
    if not f or f.IsZombie():
        raise RuntimeError(f"Failed to open ROOT file: {path}")
    t = f.Get(tree_name)
    if not t:
        f.Close()
        raise RuntimeError(f"Tree '{tree_name}' not found in {path}")

    n = min(int(t.GetEntries()), max_events or int(t.GetEntries()))
    out = []
    for i in range(n):
        t.GetEntry(i)
        event = {}
        for bname, HitCls, CollCls in branch_info:
            data = getattr(t, bname)
            event[bname] = CollCls([HitCls(h) for h in data])
        out.append(event)
    f.Close()
    return out


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------
# Available methods and their descriptions (for config reference):
#   bounded          — scipy minimize_scalar (1-D only), fast local search
#   L-BFGS-B         — quasi-Newton with bounds, fast local search (gradient-based)
#   differential_evolution — evolutionary global optimizer, gradient-free, reliable
#   dual_annealing   — simulated annealing + local search, good for rugged landscapes
#   shgo             — simplicial homology, deterministic, finds all local minima
#   brute            — exhaustive grid search + local polish

OPTIMIZE_METHODS = {
    "bounded", "L-BFGS-B",
    "differential_evolution", "dual_annealing", "shgo", "brute",
}


@dataclass
class OptResult:
    best_params: dict       # {param_name: float}
    best_fun: float
    history_x: np.ndarray   # (n_evals, n_params) — every point evaluated
    history_f: np.ndarray   # (n_evals,) — objective value at each point


def _optimize(objective, param_names, param_bounds, method: str,
              brute_grid_sizes=None) -> OptResult:
    """Optimize objective over N parameters, recording every evaluation.

    brute_grid_sizes is only used when method='brute'.
    """
    if method not in OPTIMIZE_METHODS:
        raise ValueError(f"Unknown optimize method '{method}'. "
                         f"Choose from: {sorted(OPTIMIZE_METHODS)}")

    n = len(param_names)
    bounds_list = [tuple(b) for b in param_bounds]

    # Wrap objective to record all evaluations
    history_x = []
    history_f = []

    def obj_vec(vec):
        val = objective({name: float(v) for name, v in zip(param_names, vec)})
        history_x.append([float(v) for v in vec])
        history_f.append(float(val))
        return val

    def make_result(best_vec, best_fun):
        return OptResult(
            best_params={name: float(v) for name, v in zip(param_names, best_vec)},
            best_fun=float(best_fun),
            history_x=np.array(history_x),
            history_f=np.array(history_f),
        )

    # 1-D: use minimize_scalar for bounded, wrap others
    if n == 1:
        if method == "bounded":
            def obj_scalar(v):
                return obj_vec([v])
            res = minimize_scalar(
                obj_scalar,
                bounds=bounds_list[0], method="bounded",
                options={"xatol": 0.5},
            )
            return make_result([res.x], res.fun)
        # For global methods, fall through to N-D path below

    if method == "L-BFGS-B":
        x0 = [(lo + hi) / 2 for lo, hi in bounds_list]
        res = minimize(obj_vec, x0, bounds=bounds_list, method="L-BFGS-B")
        return make_result(res.x, res.fun)

    if method == "differential_evolution":
        res = differential_evolution(obj_vec, bounds_list)
        return make_result(res.x, res.fun)

    if method == "dual_annealing":
        res = dual_annealing(obj_vec, bounds_list)
        return make_result(res.x, res.fun)

    if method == "shgo":
        res = shgo(obj_vec, bounds_list)
        return make_result(res.x, res.fun)

    if method == "brute":
        if not brute_grid_sizes:
            raise ValueError("brute_grid_sizes is required when optimize_method is 'brute'")
        ranges = [slice(lo, hi, complex(0, gs))
                  for (lo, hi), gs in zip(bounds_list, brute_grid_sizes)]
        result = brute(obj_vec, ranges, finish=minimize)
        if np.ndim(result) == 0:
            result = [float(result)]
        best_vec = [float(v) for v in result]
        best_fun = obj_vec(best_vec)
        return make_result(best_vec, best_fun)

    raise ValueError(f"Unhandled method: {method}")


def _eval_grid(objective, param_names, plot_grids):
    """Evaluate objective on the full grid. Returns an N-D score array."""
    mesh = np.meshgrid(*plot_grids, indexing="ij")
    flat = [m.ravel() for m in mesh]
    scores = np.empty(len(flat[0]), dtype=np.float64)
    for i in range(len(flat[0])):
        params = {name: float(flat[j][i]) for j, name in enumerate(param_names)}
        scores[i] = -objective(params)
    shape = tuple(len(g) for g in plot_grids)
    return scores.reshape(shape)


# ---------------------------------------------------------------------------
# Main inner-loop
# ---------------------------------------------------------------------------
def run_inner_loop(manifest_file: str, il_cfg: dict, outdir: str) -> dict:
    """
    Run the inner loop over ROOT files listed in the manifest.

    il_cfg["algorithm"]["optimize_params"] is a list of:
        {name, label, bounds: [lo, hi], grid_points}
    """
    os.makedirs(outdir, exist_ok=True)
    ROOT.gROOT.SetBatch(True)

    # --- resolve config ---
    algo_cfg = il_cfg["algorithm"]
    score_cfg = il_cfg["score"]
    max_events = il_cfg["max_events"]

    opt_method = algo_cfg["optimize_method"]
    brute_grid_sizes = algo_cfg.get("brute_grid_sizes")
    opt_params = algo_cfg["optimize_params"]
    param_names = [p["name"] for p in opt_params]
    param_labels = [p["label"] for p in opt_params]
    param_bounds = [p["bounds"] for p in opt_params]
    plot_grid_sizes = [p["plot_grid_points"] for p in opt_params]
    plot_grids = [np.linspace(b[0], b[1], gs)
                  for b, gs in zip(param_bounds, plot_grid_sizes)]
    n_algo_params = len(opt_params)

    # --- instantiate algo + score from registry ---
    algo_cls = ALGORITHMS[algo_cfg["name"]]
    algo = algo_cls(**algo_cfg.get("params", {}))  # params optional for no-arg algos

    score_cls = SCORES[score_cfg["name"]]
    scorer = score_cls(**score_cfg["params"])

    # --- load manifest ---
    items = load_manifest(manifest_file)
    if not items:
        raise RuntimeError(f"Manifest is empty: {manifest_file}")

    # --- sweep ---
    # Grid landscape: auto for 1-2 params, configurable via force_grid_landscape
    force_grid = algo_cfg.get("force_grid_landscape", False)
    compute_grid = n_algo_params <= 2 or force_grid
    if force_grid and n_algo_params > 2:
        total_pts = 1
        for gs in plot_grid_sizes:
            total_pts *= gs
        print(f"[inner loop] Warning: force_grid_landscape with {n_algo_params} params = "
              f"{total_pts} evaluations per geometry point")
    landscapes: List[np.ndarray] = []
    results: List[GeometryResult] = []

    for idx, entry in enumerate(items):
        gv = entry["geom_values"]
        path = entry["path"]
        gv_str = ", ".join(f"{k}={v}" for k, v in gv.items())
        print(f"[inner loop] [{idx+1}/{len(items)}] {gv_str}  {path}")
        events = load_events(path, il_cfg["tree_name"], il_cfg["branches"],
                              max_events)

        def obj(params_dict, _events=events):
            reco = algo.reconstruct(_events, params_dict)
            return -scorer(_events, reco)

        opt = _optimize(obj, param_names, param_bounds, opt_method, brute_grid_sizes)
        best_score = -opt.best_fun

        # Grid landscape
        if compute_grid:
            score_grid = _eval_grid(obj, param_names, plot_grids)
            landscapes.append(score_grid)
        else:
            score_grid = None

        results.append(GeometryResult(gv, path, opt.best_params, best_score))

        # Per-geometry NPZ - includes optimizer evaluation history
        save_kw = {
            "best_score": best_score,
            **{f"best_{k}": v for k, v in opt.best_params.items()},
            **{f"geom_{k}": v for k, v in gv.items()},
            "opt_history_x": opt.history_x,       # (n_evals, n_params)
            "opt_history_f": -opt.history_f,       # (n_evals,) scores (positive)
            "opt_param_names": param_names,
        }
        if score_grid is not None:
            save_kw["score"] = score_grid
            for i, (name, grid) in enumerate(zip(param_names, plot_grids)):
                save_kw[f"grid_{name}"] = grid
        np.savez(os.path.join(outdir, f"landscape_{idx:04d}.npz"), **save_kw)

    # --- aggregate ---
    best_idx = int(np.argmax([r.best_score for r in results]))
    global_best = results[best_idx]

    summary = {
        "meta": {
            "manifest": manifest_file,
            "algo": algo.name,
            "score": scorer.name,
            "optimize_params": [p["name"] for p in opt_params],
            "optimize_method": opt_method,
        },
        "per_geometry": [
            {"geom_values": r.geom_values, "root_path": r.root_path,
             "best_params": r.best_params, "best_score": r.best_score}
            for r in results
        ],
        "global_best": {
            "geom_values": global_best.geom_values,
            "best_params": global_best.best_params,
            "best_score": global_best.best_score,
        },
    }

    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # CSV: flatten geom_values and best_params into columns
    geom_keys = sorted(results[0].geom_values.keys()) if results else []
    fieldnames = geom_keys + [f"best_{n}" for n in param_names] + ["best_score", "root_path"]
    with open(os.path.join(outdir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = {**r.geom_values,
                   **{f"best_{n}": r.best_params[n] for n in param_names},
                   "best_score": r.best_score, "root_path": r.root_path}
            w.writerow(row)

    # Save landscape tensor
    if landscapes:
        tensor = np.stack(landscapes)  # shape: (n_geom, *grid_shape)
        save_kw = {"score": tensor}
        for name, grid in zip(param_names, plot_grids):
            save_kw[f"grid_{name}"] = grid
        np.savez(os.path.join(outdir, "loss_landscape_tensor.npz"), **save_kw)

    _make_plots(outdir, results, global_best, il_cfg,
                param_names, param_labels, plot_grids,
                landscapes if landscapes else None)

    print(f"\n[inner loop] Done. Outputs in {outdir}/")
    gv_str = ", ".join(f"{k}={v}" for k, v in global_best.geom_values.items())
    bp_str = ", ".join(f"{k}={v:.4g}" for k, v in global_best.best_params.items())
    print(f"  Global best: {gv_str}, {bp_str}, score={global_best.best_score:.6g}")

    return summary


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _make_plots(outdir, results, global_best, il_cfg,
                param_names, param_labels, plot_grids, landscapes):
    if not results:
        return

    plt.rcParams.update({"font.size": 14, "axes.titlesize": 16,
                         "axes.labelsize": 14, "xtick.labelsize": 12,
                         "ytick.labelsize": 12})

    geom_keys = sorted(results[0].geom_values.keys())
    geom_labels = il_cfg.get("geometry_labels", {})
    heatmap_cfg = il_cfg.get("heatmap", {})

    def glabel(param):
        return geom_labels.get(param, param)

    # --- Geometry-space heatmaps (best_score over geom params) ---
    axes = heatmap_cfg.get("axes", geom_keys[:2])
    fixed = heatmap_cfg.get("fixed", {})
    if fixed:
        plot_results = [r for r in results
                        if all(abs(r.geom_values.get(k, float("nan")) - v) < 1e-9
                               for k, v in fixed.items())]
    else:
        plot_results = results

    # 1 geom param x 1 algo param: full landscape heatmap
    if len(axes) == 1 and len(param_names) == 1 and landscapes and plot_results:
        plot_indices = [i for i, r in enumerate(results) if r in plot_results]
        plot_landscape = np.vstack([landscapes[i] for i in plot_indices])
        _plot_landscape_1d(outdir, plot_grids[0], plot_landscape, plot_results,
                           global_best, axes[0], glabel(axes[0]), param_labels[0])

    # 2 geom params: best_score heatmap
    if len(axes) == 2 and plot_results:
        _plot_heatmap_2d(outdir, plot_results, global_best, axes, glabel)

    # --- Per-geometry-parameter scatter plots ---
    # One point per geometry combo, showing full spread across other params.
    for gkey in geom_keys:
        xs = [r.geom_values[gkey] for r in results]

        plt.figure()
        plt.scatter(xs, [r.best_score for r in results], alpha=0.6)
        plt.xlabel(glabel(gkey))
        plt.ylabel("Best score (inner-loop optimized)")
        plt.title(f"Best score vs {glabel(gkey)}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"best_score_vs_{gkey}.png"), dpi=200)
        plt.close()

        for pname, plabel in zip(param_names, param_labels):
            plt.figure()
            plt.scatter(xs, [r.best_params[pname] for r in results], alpha=0.6)
            plt.xlabel(glabel(gkey))
            plt.ylabel(f"Optimized {plabel}")
            plt.title(f"Best {plabel} vs {glabel(gkey)}")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"best_{pname}_vs_{gkey}.png"), dpi=200)
            plt.close()


def _plot_landscape_1d(outdir, param_grid, landscape_arr, results,
                       global_best, geom_key, geom_label, param_label):
    """2D heatmap: single geometry param x single algo param -> score."""
    geom_arr = np.array([r.geom_values[geom_key] for r in results])
    plt.figure()
    extent = [float(param_grid[0]), float(param_grid[-1]),
              float(geom_arr[0]), float(geom_arr[-1])]
    plt.imshow(landscape_arr, aspect="auto", origin="lower",
               extent=extent, interpolation="nearest")
    plt.xlabel(param_label)
    plt.ylabel(geom_label)
    plt.colorbar(label="Score")
    plt.scatter([r.best_params[list(r.best_params.keys())[0]] for r in results],
                geom_arr, marker="x")
    if geom_key in global_best.geom_values:
        plt.scatter([global_best.best_params[list(global_best.best_params.keys())[0]]],
                    [global_best.geom_values[geom_key]], marker="*", s=200)
    plt.title("Bilevel landscape")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "landscape_heatmap.png"), dpi=200)
    plt.close()


def _plot_heatmap_2d(outdir, results, global_best, axes, glabel):
    """2D color heatmap + 3D surface: two geometry params -> best score."""
    key_x, key_y = axes

    ux = np.unique([r.geom_values[key_x] for r in results])
    uy = np.unique([r.geom_values[key_y] for r in results])

    score_grid = np.full((len(uy), len(ux)), np.nan)
    count_grid = np.zeros_like(score_grid)
    for r in results:
        ix = int(np.searchsorted(ux, r.geom_values[key_x]))
        iy = int(np.searchsorted(uy, r.geom_values[key_y]))
        if np.isnan(score_grid[iy, ix]):
            score_grid[iy, ix] = r.best_score
        else:
            score_grid[iy, ix] += r.best_score
        count_grid[iy, ix] += 1
    mask = count_grid > 0
    score_grid[mask] /= count_grid[mask]

    plt.figure()
    plt.pcolormesh(ux, uy, score_grid, shading="auto", cmap="viridis")
    plt.xlabel(glabel(key_x))
    plt.ylabel(glabel(key_y))
    plt.colorbar(label="Best score")
    if key_x in global_best.geom_values and key_y in global_best.geom_values:
        plt.scatter([global_best.geom_values[key_x]],
                    [global_best.geom_values[key_y]], marker="*", s=200,
                    c="red", zorder=5)
    plt.title("Best score landscape")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "landscape_heatmap.png"), dpi=200)
    plt.close()

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    X, Y = np.meshgrid(ux, uy)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, score_grid, cmap="viridis", alpha=0.8)
    ax.set_xlabel(glabel(key_x))
    ax.set_ylabel(glabel(key_y))
    ax.set_zlabel("Best score")
    ax.set_title("Best score landscape")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "landscape_3d.png"), dpi=200)
    plt.close()
