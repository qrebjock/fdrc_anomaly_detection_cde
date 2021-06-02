import numpy as np


EXPERIMENT1_RECIPES = {
    "default": {
        # We will run for varying anomaly ratios (for repeats time)
        "anomaly_ratios": [float(r) for r in np.logspace(-4, -0.05, 16)],
        "repeats": [100] * 8 + [50] * 4 + [10] * 4,
        # This is the shared fdr_target which we keep fixed during this experiment.
        "fdr_target": 0.1,
        # We will use shared delta(for decay filters) and gamma_size (for all filters)
        # You can overwrite them at "filter_params" section for a single filter.
        "delta": 0.99,
        "gamma_size": 200,
        # By default we use Gaussian data model
        "data_model": "gaussian",
        "data_params": {
            "size": 10000,
            "signal_strength": 3.0,
            "data_noise": 1.0,
            "one_sided": True,
        },
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
            }
        },
    },
    "test_gaussian": {
        "anomaly_ratios": [float(r) for r in np.logspace(-3, -0.05, 10)],
        "repeats": [10] * 10,
        "data_model": "gaussian",
        "data_params": {
            "size": 1000,
            "signal_strength": 3.0,
            "data_noise": 1.0,
            "one_sided": True,
        },
    },
    "test_ar": {
        "anomaly_ratios": [float(r) for r in np.logspace(-3, -0.05, 10)],
        "repeats": [10] * 10,
        "data_model": "gaussian",
        "data_params": {
            "size": 1000,
            "signal_strength": 5.0,
            "c": 0.5,
            "phi": 0.9,
            "model": "sliding_window",
            "window_size": 10,
            "one_sided": True
        },
    },
    "submitted": {
        "anomaly_ratios": [float(r) for r in np.logspace(-4, -0.05, 16)],
        "repeats": [100] * 8 + [50] * 4 + [10] * 4,
        "delta": 0.99,
        "gamma_size": 20000,
        "fdr_target": 0.1,
        "data_model": "gaussian",
        "data_params": {
            "size": 20000,
            "signal_strength": 3.0,
            "data_noise": 1.0,
            "one_sided": True,
        },
    },
    "ar-model-1": {

    },
}
