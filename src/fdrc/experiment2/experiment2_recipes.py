import numpy as np


EXPERIMENT2_RECIPES = {
    "default": {
        "repeat": 100,
        # We will use shared delta(for decay filters) and gamma_size (for all filters)
        # You can overwrite them at "filter_params" section for a single filter.
        "delta": 0.99,
        "gamma_size": 200,
        # These are the parameters on the x axis of the Power-FDR curve for
        # fixed threshold and FDRC filters.
        "fixed_thresholds": list(np.logspace(-1, -5, 100)),
        "fdr_targets": list(np.linspace(0.99, 0, 100, endpoint=False)),
        # By default we use Gaussian data model
        "data_model": "gaussian",
        "data_params": {
            "size": 2000,
            "anomaly_ratio": 0.01,
            "signal_strength": 3.0,
            "data_noise": 1.0,
            "one_sided": True
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
            },
        },
    },
    "test_gaussian": {
        "repeat": 10,
        "data_model": "gaussian",
        "data_params": {
            "size": 500,
            "anomaly_ratio": 0.01,
            "signal_strength": 3.0,
            "data_noise": 1.0,
            "one_sided": True
        },
        "fixed_thresholds": list(np.logspace(-1, -5, 10)),
        "fdrc_targets": list(np.linspace(0.99, 0, 10, endpoint=False)),
    },
    "test_ar": {
        "repeat": 10,
        "data_model": "ar",
        "data_params": {
            "size": 500,
            "anomaly_ratio": 0.01,
            "signal_strength": 3.0,
            "c": 0.2,
            "phi": 0.9,
        },
        "fixed_thresholds": list(np.logspace(-1, -5, 10)),
        "fdrc_targets": list(np.linspace(0.99, 0, 10, endpoint=False)),
    },
    "gaussian_increasing_noise": {
        "data_model": "gaussian",
        "data_params": {
            "size": 2000,
            "anomaly_ratio": 0.01,
            "signal_strength": 5.0,
            "data_noise": list(np.linspace(1, 3, 2000)),
            "model_noise": 1.0,
            "one_sided": True,
        },
    },
    "gaussian_increasing_signal_strength": {
        "data_model": "gaussian",
        "data_params": {
            "size": 2000,
            "anomaly_ratio": 0.01,
            "signal_strength": list(np.linspace(2, 5, 2000)),
            "data_noise": 1.0,
            "model_noise": 1.0,
            "one_sided": True,
        },
    },
    "gaussian_decreasing_signal_strength": {
        "data_model": "gaussian",
        "data_params": {
            "size": 2000,
            "anomaly_ratio": 0.01,
            "signal_strength": list(np.linspace(5, 2, 2000)),
            "data_noise": 1.0,
            "model_noise": 1.0,
            "one_sided": True,
        },
    },
    "gaussian_increasing_anomalies": {
        "data_model": "gaussian",
        "data_params": {
            "size": 2000,
            "anomaly_ratio": list(np.linspace(0.001, 0.01, 2000)),
            "signal_strength": 3,
            "data_noise": 1.0,
            "model_noise": 1.0,
            "one_sided": True,
        },
    },
    "gaussian_decreasing_anomalies": {
        "data_model": "gaussian",
        "data_params": {
            "size": 2000,
            "anomaly_ratio": list(np.linspace(0.01, 0.001, 2000)),
            "signal_strength": 3,
            "data_noise": 1.0,
            "model_noise": 1.0,
            "one_sided": True,
        },
    },
    "gaussian_increasing_combined": {
        "data_model": "gaussian",
        "data_params": {
            "size": 2000,
            "anomaly_ratio": list(np.linspace(0.001, 0.01, 2000)),
            "signal_strength": list(np.linspace(2, 5, 2000)),
            "data_noise": 1.0,
            "model_noise": 1.0,
            "one_sided": True,
        },
    },
    "gaussian_decreasing_combined": {
        "data_model": "gaussian",
        "data_params": {
            "size": 2000,
            "anomaly_ratio": list(np.linspace(0.01, 0.001, 2000)),
            "signal_strength": list(np.linspace(5, 2, 2000)),
            "data_noise": 1.0,
            "model_noise": 1.0,
            "one_sided": True,
        },
    },
    "gaussian_increasing_noise_strength": {
        "data_model": "gaussian",
        "data_params": {
            "size": 2000,
            "anomaly_ratio": 0.01,
            "signal_strength": list(np.linspace(3, 5, 2000)),
            "data_noise": list(np.linspace(1, 3, 2000)),
            "model_noise": 1.0,
            "one_sided": True,
        },
    },
    "piecewise-1": {
        "delta": 0.99,
        "gamma_size": 10000,
        "data_model": "gaussian",
        "data_params": {
            "size": 14000,
            "anomaly_ratio": [0.01] * 2000 + [0.0] * 10000 + [0.01] * 2000,
            "signal_strength": 3.0,
            "data_noise": 1.0,
            "model_noise": 1.0,
            "one_sided": True,
        },
    },
    "piecewise-2": {
        "delta": 0.99,
        "gamma_size": 10000,
        "data_model": "gaussian",
        "data_params": {
            "size": 14000,
            "anomaly_ratio": [0.01] * 2000 + [0.0] * 10000 + [0.01] * 2000,
            "signal_strength": 4.0,
            "data_noise": 1.0,
            "model_noise": 1.0,
            "one_sided": True,
        },
    },
    "fixed-threshold-1": {
        "delta": 0.99,
        "repeat": 30,
        "gamma_size": 2000,
        "data_model": "gaussian",
        "data_params": {
            "size": 2000,
            "anomaly_ratio": 0.01,
            "signal_strength": list(np.linspace(3.0, 4.0, 2000)),
            "data_noise": 1.0,
            "model_noise": 1.0,
            "one_sided": True,
        },
    },
    "fixed-threshold-2": {
        "delta": 0.99,
        "repeat": 30,
        "gamma_size": 2000,
        "data_model": "ar",
        "data_params": {
            "anomaly_ratio": 0.01,
            "size": 2000,
            "signal_strength": 4.0,
            "noise": 1.0,
            "c": 0.0,
            "phi": 0.9,
            "one_sided": True,
            "model": "exact",
            "method": "filter",
            "window_size": 64
        }
    }
}
