import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple, Optional

COLORS = {
    "target_color": "darkslategray",
    "p_values_color": "#1f77b4",
    "thresholds_color": "#ff7f0e",
    "default_anomaly_color": "#9467bd",
    "tp_anomaly_color": "green",
    "fp_anomaly_color": "red",
    "forecast_color": "m",
    "LORDFilter": '#1f77b4',
    "DecayLORDFilter": '#ff7f0e',
    "SAFFRONFilter": '#2ca02c',
    "DecaySAFFRONFilter": '#d62728',
    "ADDISFilter": '#9467bd',
    "FixedThresholdFilter": "black"
}

MARKERS = {
    "FixedThresholdFilter": "x",
    "LORDFilter": "o",
    "DecayLORDFilter": "s",
    "SAFFRONFilter": "d",
    "DecaySAFFRONFilter": "*",
    "ADDISFilter": "+"
}

PLOT_ORDER = [
    "FixedThresholdFilter",
    "ADDISFilter",
    "DecayLORDFilter",
    "DecaySAFFRONFilter",
    "LORDFilter",
    "SAFFRONFilter",
]

LABELS = {
    "LORDFilter": "Lord",
    "SAFFRONFilter": "Saffron",
    "DecaySAFFRONFilter": "DecaySaffron",
    "DecayLORDFilter": "DecayLord",
    "ADDISFilter": "Addis",
}

# To make plots that render well with LaTeX:
plt.rc("text", usetex=True)
plt.rc("font", family="serif")


def plot_dataset(
    dataset: Dict,
    thresholds: Optional[List[float]] = None,
    log_y_scale: bool = True
):
    """
    Plots dataset target and p-values. Also plot thresholds if given.
    """

    # Part 1: Target
    fig = plt.Figure(figsize=(20, 8))
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(dataset["target"], color=COLORS["target_color"])
    if "forecasts" in dataset:
        forecasts = np.asarray(dataset["forecasts"])
        ax.plot(forecasts, color=COLORS["forecast_color"])
        if "errors" in dataset:
            errors = np.asarray(dataset["errors"])
            y_min = forecasts - 3 * errors
            y_max = forecasts + 3 * errors
            x = range(0, len(y_max))
            ax.fill_between(x, y_max, y_min, color=COLORS["forecast_color"], alpha=0.3)

    _plot_markers(
        ax, dataset["target"], dataset["labels"], COLORS["default_anomaly_color"]
    )
    ax.set_title("Target", fontsize=16)
    ax.grid(True, color='gainsboro')

    # Part 2: P-values (cropped at 1-20 for visibility)
    ax2 = fig.add_subplot(2, 1, 2)
    _plot_results_to_ax(
        ax2, dataset["labels"], dataset["p_values"], thresholds, log_y_scale
    )
    return fig


def plot_result(
    dataset,
    thresholds: List[float],
    log_y_scale: bool = True
):
    """
    Plots p-values and thresholds, but not the target data.
    """
    fig = plt.Figure(figsize=(20, 5))
    ax = fig.gca()
    _plot_results_to_ax(
        ax, dataset["labels"], dataset["p_values"], thresholds, log_y_scale
    )
    return fig


def plot_curves(
    x: List,
    y: List,
    filter_names: List,
    y_errors: Optional[List] = None,
    fig_size=(8, 8),
    title: str = None,
    x_label: str = None,
    y_label: str = None,
    label_size: int = 16,
    legend_size: int = 12,
    legend_loc: str = "upper left",
    invert_xaxis: bool = False,
    log_xscale: bool = False,
    x_lim: Optional[Tuple] = None,
    y_lim: Optional[Tuple] = None,
    target: Optional[int] = None
):
    fig = plt.Figure(figsize=fig_size)
    ax = fig.gca()

    if target:
        ax.axhline(target, color="k", alpha=0.8, label="Target")

    for filter_name in PLOT_ORDER:
        if filter_name not in filter_names:
            continue
        idx = filter_names.index(filter_name)
        _plot_with_error(
            x=x[idx],
            y=y[idx],
            y_error=y_errors[idx] if y_errors is not None else None,
            label=LABELS[filter_name],
            color=COLORS[filter_name],
            ax=ax,
            base_kwargs={
                'linestyle': '-',
                'marker': MARKERS[filter_name],
                'alpha': 0.7
            }
        )
    if invert_xaxis:
        ax.invert_xaxis()
    if log_xscale:
        ax.set_xscale('log')
    if title:
        ax.set_title(title)
    ax.grid(linestyle='--', linewidth=1)
    ax.legend(loc=legend_loc, fontsize=legend_size)
    ax.set_xlabel(x_label, fontsize=label_size)
    ax.set_ylabel(y_label, fontsize=label_size)
    if y_lim:
        ax.set_ylim(y_lim)
    if x_lim:
        ax.set_ylim(x_lim)
    return fig

# -----------------------------------------
#            Helper Functions
# -----------------------------------------
# The functions above are helper functions.
# You probably need only the above functions.


def _plot_with_error(
    x, y, y_error, label=None, color=None, alpha_fill=0.3, ax=None, base_kwargs=None
):
    """
    Plot with transparent error.
    :param x: List of abscissas.
    :param y: List of ordinates.
    :param y_error: Amplitude of the error.
    :param label: Label of the curve (legend).
    :param color: Color of the plot.
    :param alpha_fill: Transparency of the error (default 0.3).
    :param ax: Axe where to plot the curve. If None (default), creates a new one.
    """
    if base_kwargs is None:
        base_kwargs = {}
    x, y = np.array(x), np.array(y)
    ax = ax if ax is not None else plt.gca()
    (base_line,) = ax.plot(x, y, label=label, color=color, **base_kwargs)
    if color is None:
        color = base_line.get_color()
    if y_error is not None:
        if np.isscalar(y_error) or len(y_error) == len(y):
            y_min = y - y_error
            y_max = y + y_error
        elif len(y_error) == 2:
            y_min, y_max = y_error
        else:
            raise ValueError(
                f"y_error must be either a scalar, a list of the same length as y, "
                f"or a tuple containing the min and the max errors. Found {y_error}"
            )
        ax.fill_between(x, y_max, y_min, color=color, alpha=alpha_fill)


def _plot_markers(ax, x, y, y_color):
    ax.plot(
        x,
        markevery=list(np.array(y, dtype=bool)),
        linestyle="none",
        marker="o",
        markersize=8,
        color=y_color,
    )


def _plot_results_to_ax(
    ax, labels, p_values, thresholds, log_y_scale=True
):
    # Crop p-values (cropped at 1-20 for visibility)
    p_values = np.clip(p_values, a_min=1e-20, a_max=1)
    ax.plot(p_values, color=COLORS["p_values_color"])

    if thresholds is None:
        _plot_markers(ax, p_values, labels, COLORS["default_anomaly_color"])
        ax.set_title("log10(p-values)", fontsize=16)
    else:
        thresholds = np.clip(thresholds, a_min=1e-22, a_max=1)
        ax.plot(thresholds, color=COLORS["thresholds_color"])

        # True Positive Markers
        est_labels = [p <= t for p,t in zip(p_values, thresholds)]
        tp_labels = np.asarray(labels) * np.asarray(est_labels)
        _plot_markers(ax, p_values, tp_labels, COLORS["tp_anomaly_color"])

        # False Positive Markers
        fp_labels = (1-np.asarray(labels)) * np.asarray(est_labels)
        _plot_markers(ax, p_values, fp_labels, COLORS["fp_anomaly_color"])

        # False Negative Markers
        fn_labels = np.asarray(labels) * (1-np.asarray(est_labels))
        _plot_markers(ax, p_values, fn_labels, COLORS["default_anomaly_color"])

        ax.set_title("log10(p-values and thresholds)", fontsize=16)

    ax.grid(True, color='gainsboro')
    if log_y_scale:
        ax.set_yscale("log")
