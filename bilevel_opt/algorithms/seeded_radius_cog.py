"""SeededRadiusCoG  algorithm and SNREnergyScore scorer.

Toy algorithm for demonstrating the bilevel optimization pipeline.

Parameters optimized by the inner loop:
    R_cluster_mm     — clustering radius around the seed hit [mm]
    E_threshold_GeV  — minimum hit energy to include in cluster [GeV]

Physics motivation: R_cluster_mm controls how much of the shower is captured.
Too small misses energy; too large picks up noise from neighboring showers.
E_threshold_GeV suppresses low-energy noise hits and shower tails that
degrade the energy-weighted position and energy resolution. The optimal
(R, E_threshold) pair depends on the detector granularity (geometry).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class ClusterReco:
    E_cluster: float
    x_cog: float
    y_cog: float
    n_hits: int


class SeededRadiusCoG:
    """
    Clustering: find highest-energy hit (seed), collect hits within radius R
    that pass an energy threshold, compute energy-weighted center-of-gravity.

    Constructor params (from config algorithm.params):
        branch (str) — which branch to cluster on

    Optimized params:
        R_cluster_mm (float)     — radius cut around seed
        E_threshold_GeV (float)  — minimum energy per hit (0 = no threshold)
    """
    name = "seeded_radius_cog"

    def __init__(self, branch: str):
        self.branch = branch

    def reconstruct(self, events, params: Dict[str, float]) -> List[ClusterReco]:
        R2 = float(params["R_cluster_mm"]) ** 2
        E_thr = float(params.get("E_threshold_GeV", 0.0))
        out: List[ClusterReco] = []
        for event in events:
            ev = event[self.branch]
            if ev.E.size == 0:
                out.append(ClusterReco(0.0, np.nan, np.nan, 0))
                continue
            seed = int(np.argmax(ev.E))
            dx = ev.x - ev.x[seed]
            dy = ev.y - ev.y[seed]
            sel = ((dx * dx + dy * dy) <= R2) & (ev.E >= E_thr)
            if not np.any(sel):
                out.append(ClusterReco(0.0, np.nan, np.nan, 0))
                continue
            Esel = ev.E[sel]
            Ecl = float(np.sum(Esel))
            nh = int(Esel.size)
            if Ecl <= 0.0:
                out.append(ClusterReco(Ecl, np.nan, np.nan, nh))
                continue
            xc = float(np.sum(Esel * ev.x[sel]) / Ecl)
            yc = float(np.sum(Esel * ev.y[sel]) / Ecl)
            out.append(ClusterReco(Ecl, xc, yc, nh))
        return out


class SNREnergyScore:
    """Score = mean( E_cluster / (sigma0 * sqrt(N_hits)) ) over events."""
    name = "snr_energy"

    def __init__(self, sigma0: float):
        self.sigma0 = float(sigma0)

    def __call__(self, events, reco) -> float:
        vals = [
            r.E_cluster / (self.sigma0 * math.sqrt(r.n_hits))
            for r in reco if r.n_hits > 0
        ]
        return float(np.mean(vals)) if vals else float("-inf")
