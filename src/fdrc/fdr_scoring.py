import numpy as np

from typing import List, Union


def fdp_score(
    labels: List[bool],
    ground_truth: List[bool],
    offset: float = 0,
    cumulative: bool = False,
    delta: float = 1,
    weights: List[float] = None,
) -> Union[float, List[float]]:
    """
    Computes the false discovery proportion (FDP, whose expectation is the FDR)
    of the subset of rejected hypotheses given the ground truth.
    Optionally gives weights to hypotheses and discounts past decisions over time,
    as shown in this paper https://arxiv.org/abs/1710.00499 (Section 6.1).

    Parameters
    ----------
    labels : List[bool]
        List of labels describing the hypotheses that were rejected.
    ground_truth : List[bool]
        List containing the ground truth for the labels.
    offset : float
        Offset in the denominator term of the modified FDP expression, defaults to 0.
    cumulative : bool
        If True, returns the FDP at each time step, otherwise only for the last one. Defaults to False.
    delta : float
        Discount factor. Defaults to 1 (no discount).
    weights : List[float]
        Sequence of weights given to each hypothesis. Defaults to None (weight 1 everywhere).

    Returns
    -------
    float or np.ndarray
        FDP scores over time or for the last step only.
    """
    assert (
        len(ground_truth) == len(labels)
    ), f"The labels and the ground truth must have the same length"
    assert (
        offset >= 0
    ), f"The modified FDP offset should be non-negative, found {offset}"
    assert (
        0 <= delta <= 1
    ), f"The discount factor should be between 0 and 1, found {delta}"

    labels_arr = np.asarray(labels, dtype=bool)
    ground_truth_arr = np.asarray(ground_truth, dtype=bool)

    false_discoveries = labels_arr & ~ground_truth_arr
    discoveries = labels_arr

    if weights is not None:
        assert (
            len(weights) == len(labels)
        ), f"The labels and the weights must have the same length"
        weights_arr = np.asarray(weights)
        false_discoveries *= weights_arr
        discoveries *= weights_arr

    deltas = np.power(delta, np.arange(len(labels), dtype=float))
    v_t = np.convolve(false_discoveries, deltas, mode="full")[:len(labels)]
    r_t = np.convolve(discoveries, deltas, mode="full")[:len(labels)]

    if not cumulative:
        v_t = v_t[-1]
        r_t = r_t[-1]

    # Clipping to handle the case were power = 0
    r_t = np.clip(r_t, a_min=1, a_max=None)

    return v_t / (r_t + offset)


def power_score(
    labels: List[bool], ground_truth: List[bool], cumulative: bool = False
) -> Union[float, List[float]]:
    """
    Computes the statistical power of the subset of rejected hypotheses given the
    ground truth.

    Parameters
    ----------
    labels : List[bool]
        List of labels describing the hypotheses that were rejected.
    ground_truth : List[bool]
        List containing the ground truth for the labels.
    cumulative : bool
        If True, returns the power for each time step, otherwise only for the last one.
        Defaults to False.

    Returns
    -------
    float or np.ndarray
        Power scores over time or for the last step only.
    """
    assert (
        len(labels) == len(ground_truth)
    ), f"The labels and the ground truth must have the same length"

    labels_arr = np.asarray(labels, dtype=bool)
    ground_truth_arr = np.asarray(ground_truth, dtype=bool)

    if cumulative:
        true_discoveries = np.cumsum(labels_arr & ground_truth_arr)
        true_hypotheses = np.cumsum(ground_truth_arr)
    else:
        true_discoveries = np.sum(labels_arr & ground_truth_arr)
        true_hypotheses = np.sum(ground_truth_arr)

    # Clipping to handle the case were power = 0
    true_hypotheses = np.clip(true_hypotheses, a_min=1, a_max=None)

    return true_discoveries / true_hypotheses
