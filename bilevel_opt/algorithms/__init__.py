"""Algorithm and score plugins for the inner loop.

Each algorithm must have:
    name: str
    reconstruct(events, params: dict) -> reco (any type)

Each score must have:
    name: str
    __call__(events, reco) -> float
"""

from bilevel_opt.algorithms.registry import ALGORITHMS, SCORES

__all__ = ["ALGORITHMS", "SCORES"]
