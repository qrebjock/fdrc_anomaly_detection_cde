import numpy as np

from collections import deque
from scipy.stats import norm
from typing import Dict, List, Optional, Union


def make_hypothesis(model: str, params: Dict, n: int = 1):
    """
    Generate data using Gaussian Mixture or AR models

    Parameters
    ----------
    model: str
        "gaussian" or "ar"
    params: Dict
        Parameters for the hypothesis
    n: int
        Number of datasets to generate

    Returns
    -------
    Dict or List[Dict]
        One or more datasets generated
    """
    if model == "gaussian":
        generator = make_gaussian_hypotheses
    elif model == "ar":
        generator = make_ar_hypotheses
    else:
        raise ValueError(f"No such model: {model}")

    if n == 1:
        return generator(**params)

    return [generator(**params) for _ in range(n)]


def make_gaussian_hypotheses(
    size: int,
    anomaly_ratio: Union[float, List] = 0.1,
    signal_strength: float = 3.0,
    data_noise: Union[float, List] = 1.0,
    model_noise: Union[float, List] = None,
    one_sided: bool = True,
) -> Dict:
    """
    Generates a sequences of independent hypotheses.
    Each hypothesis is alternative with probability pi and is
    null with probability 1 - pi. For null hypotheses the outcome
    is a random value sampled from N(0, noise). For alternatives,
    it is sampled from N(signal_strength, noise).

    Parameters
    ----------
    size : int
        Number of hypotheses.
    anomaly_ratio : List or float
        Proportion of alternative hypotheses.
    signal_strength : List or float
        Mean for the alternative hypotheses.
    data_noise : float or list of floats
        True noise level for data
    model_noise : Optional
        Noise known to model. If not given, model is assumed to know the true noise.
    one_sided : bool
        Whether or not the p_values should be one sided.

    Returns
    -------
    Dict
        A Dictionary containing target, labels and p-values.
    """
    if model_noise is None:
        model_noise = data_noise

    assert type(anomaly_ratio) == float or len(anomaly_ratio) == size
    assert type(data_noise) == float or len(data_noise) == size
    assert type(model_noise) == float or len(model_noise) == size

    ground_truth = np.random.binomial(n=1, p=anomaly_ratio, size=size).astype(float)
    z = np.random.normal(ground_truth * signal_strength, data_noise, size=size)

    if one_sided:
        p_values = 1 - norm.cdf(z, loc=0, scale=model_noise)
    else:
        p_values = 2 * (1 - norm.cdf(np.abs(z), loc=0, scale=model_noise))

    return {
        "target": list(z),
        "labels": list(ground_truth),
        "p_values": list(p_values),
    }


def make_ar_hypotheses(
    size: int,
    anomaly_ratio: Union[float, List] = 0.1,
    signal_strength: float = 3.0,
    c: float = 0.0,
    phi: float = 0.9,
    noise: float = 1.0,
    model: str = "simple",
    window_size: int = 10,
    one_sided: bool = True,
):
    if model == "sliding_window":
        size += window_size

    # Make AR data for size steps
    target = np.random.normal(0, noise, size=size)
    for i in range(1, size):
        target[i] += c + phi * target[i-1]

    # Add anomalies
    labels = np.random.binomial(n=1, p=anomaly_ratio, size=size).astype(float)
    target += labels * signal_strength

    # Calculate p-values assuming the model knows true parameters
    if model == "exact":
        forecasts = [0] + list(c + phi * target[:-1])
        errors = [noise] * size
    elif model == "simple":
        forecasts = [np.mean(target)] * size
        errors = [np.std(target)] * size
    elif model == "sliding_window":
        forecasts, errors = [], []
        window = deque(target[:window_size], maxlen=window_size)
        for i in range(window_size, size):
            forecasts.append(np.mean(window))
            errors.append(np.std(window))
            window.append(target[i])
        target = target[window_size:]
        labels = labels[window_size:]
    else:
        raise ValueError(f"No such {model}.")

    if one_sided:
        p_values =  1 - norm.cdf(target, loc=forecasts, scale=errors)
    else:
        z = np.asarray(target) - forecasts
        p_values = 2 * (1 - norm.cdf(np.abs(z), loc=0, scale=errors))
    return {
        "target": list(target),
        "labels": list(labels),
        "p_values": list(p_values),
        "forecasts": list(forecasts),
        "errors": list(errors)
    }
