import numpy as np


EXPERIMENT_SMD_RECIPES = {
    "default": {
        # We will use shared delta(for decay filters) and gamma_size (for all filters)
        # You can overwrite them at "filter_params" section for a single filter.
        "delta": 0.99,
        "gamma_size": 200,
        # These are the parameters on the x axis of the Power-FDR curve for
        # fixed threshold and FDRC filters.
        "fixed_thresholds": list(np.logspace(-1, -5, 100)),
        "fdr_targets": list(np.linspace(0.99, 0, 100, endpoint=False)),
        # The experiment will run for the filters in this list.
        # You can overwrite this list to skip some filters.
        "filters": [
            "LORDFilter",
            "DecayLORDFilter",
            "SAFFRONFilter",
            "DecaySAFFRONFilter",
            "ADDISFilter",
        ],
        # Default parameters for filters (besides delta and gamma_size)
        "filter_params": {
            "LORDFilter": {
                "epsilon_w": 0.0,
                "epsilon_r": 0.0,
            },
            "DecayLORDFilter": {
                "eta": 1.0,
            },
            "SAFFRONFilter": {
                "candidacy_threshold": 0.5,
                "gamma_exponent": 1.6,
            },
            "DecaySAFFRONFilter": {
                "candidacy_threshold": 0.5,
                "gamma_exponent": 1.6,
                "eta": 1.0,
            },
            "ADDISFilter": {
                "candidacy_threshold": 0.25,
                "discarding_threshold": 0.5,
                "gamma_exponent": 1.6,
                "eta": 1.0,
            },
        },
    },
    "p10_small": {
        "metric": "p10",
        "fixed_thresholds": list(np.logspace(-1, -5, 4)),
        "fdr_targets": list(np.linspace(0.99, 0, 4, endpoint=False)),
    },
    "p10": {
        "metric": "p10",
    },
    "p10_long_gamma": {
        "metric": "p10",
        "gamma_size": 1000,
    },
    "p5": {
        "metric": "p5",
    },
    "p5_long_gamma": {
        "metric": "p5",
        "gamma_size": 1000,
    },
    "p1": {
        "metric": "p1",
    },
    "p1_long_gamma": {
        "metric": "p1",
        "gamma_size": 1000,
    },
    "p20": {
        "metric": "p20",
    },
    "p20_long_gamma": {
        "metric": "p20",
        "gamma_size": 1000,
    },
    "p50": {
        "metric": "p20",
    },
    "p50_long_gamma": {
        "metric": "p20",
        "gamma_size": 1000,
    },
    "chi": {
        "metric": "chi",
    },
    "chi_long_gamma": {
        "metric": "chi",
        "gamma_size": 1000,
    },

}
