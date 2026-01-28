"""Algorithm and score registries - looked up by name from config."""

from bilevel_opt.algorithms.seeded_radius_cog import SeededRadiusCoG, SNREnergyScore

ALGORITHMS = {
    "seeded_radius_cog": SeededRadiusCoG,
}

SCORES = {
    "snr_energy": SNREnergyScore,
}
