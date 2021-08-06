import json
import numpy as np

from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Union


def timestamped_name(folder_name, recipe_name, suffix=None):
    time_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    name = str(folder_name) + "/" + f"{recipe_name}-{time_stamp}"
    if suffix:
        name += "." + suffix
    return name


def save_json(data, filename, indent=None):
    with open(str(filename), "w") as fp:
        json.dump(data, fp, sort_keys=True, indent=indent)


def load_json(filename):
    with open(str(filename), "r") as fp:
        return json.load(fp)


def load_experiment(folder: Union[Path, str]) -> Tuple:

    def read_if_exists(filename):
        if Path(filename).exists():
            return load_json(filename)
        return None

    folder = Path(folder)
    return (
        read_if_exists(folder / "recipe.json"),
        read_if_exists(folder / "data.json"),
        read_if_exists(folder / "results.json"),
    )


def calculate_precision_recall(labels: List, ground_truth: List):
    """
    Calculates precision / recall for given estimations

    Parameters
    ----------
    labels : List[bool]
        List of labels describing the hypotheses that were rejected.
    ground_truth : List[bool]
        List containing the ground truth for the labels.

    Returns
    -------
    float
        Precision. (None if estimations has no anomalies)
    float
        Recall. (None if ground truth  has no anomalies)
    """
    assert (
        len(ground_truth) == len(labels)
    ), f"The labels and the ground truth must have the same length"

    tp = np.sum(np.asarray(labels) * np.asarray(ground_truth))
    p = np.sum(labels)
    t = np.sum(ground_truth)
    precision = tp / p if p > 0 else 0
    recall = tp / t if t > 0 else 0
    return precision, recall


def benjamini_hochberg(p_values: List[float], fdr_target: float):
    m = len(p_values)
    p_values = np.asarray(p_values)

    sort_permutation = np.argsort(p_values)
    sorted_p_values = p_values[sort_permutation]
    index = np.arange(1, m + 1)
    candidates = np.where((fdr_target * index / m - sorted_p_values) >= 0)[0]

    if len(candidates) > 0:
        threshold = sorted_p_values[candidates[-1]]
    else:
        threshold = 0

    return p_values <= threshold


class TicTocTimer:
    def __init__(self):
        self.start = None
        self.end = None
        self.elapsed = None
        self.tic()

    def tic(self):
        self.start = datetime.now().replace(microsecond=0)

    def toc(self):
        self.end = datetime.now().replace(microsecond=0)
        self.elapsed = self.end - self.start
        return self.elapsed
