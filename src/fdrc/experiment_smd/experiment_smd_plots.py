import os
import click
import numpy as np

from pathlib import Path
from fdrc.utils import load_experiment
from fdrc.experiment2 import make_fdp_curve
from fdrc.plots import plot_curves

SMD_RESULTS_DIR = Path(os.path.abspath(__file__)).parents[3] / "results/experiment_smd"

EXCLUDED_FILTERS = ["FixedThresholdFilter", "ADDISFilter"]


def plot_smd(results, recipe, metric, y_label):
    filter_results = [
        result for result in results if result["name"] not in EXCLUDED_FILTERS
    ]

    y_errors = [
        np.asarray([x for _, x in result[metric]]) / np.sqrt(28)
        for result in filter_results
    ]

    return plot_curves(
        x=[recipe["fdr_targets"]] * len(filter_results),
        y=[[x for x, _ in result[metric]] for result in filter_results],
        y_errors=y_errors,
        filter_names=[result["name"] for result in filter_results],
        fig_size=(10, 10),
        label_size=20,
        legend_size=14,
        legend_loc="lower right",
        x_label="Target FDR (1-Precision)",
        y_label=y_label,
        x_lim=(0, 1),
        y_lim=(0, 1),
    )


def make_plot_for(experiment):
    exp_dir = SMD_RESULTS_DIR / experiment
    recipe, data, results = load_experiment(exp_dir)
    assert recipe is not None and data is not None and results is not None, \
        "Experiment not loaded!"

    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig1 = make_fdp_curve(
        [result for result in results if result["name"] != "ADDISFilter"]
    )
    fig1.savefig(str(plots_dir / "fdp_curve.pdf"))
    fig1.savefig(str(plots_dir / "fdp_curve.png"))

    fig2 = plot_smd(results, recipe, "fdr", y_label="Actual FDR (1-Precision)")
    fig2.savefig(str(plots_dir / "true_fdr.pdf"))
    fig2.savefig(str(plots_dir / "true_fdr.png"))

    fig3 = plot_smd(
        results, recipe, "decay_fdr", y_label="Actual FDR$_\delta$ (1-Precision)"
    )
    fig3.savefig(str(plots_dir / "decay_fdr.pdf"))
    fig3.savefig(str(plots_dir / "decay_fdr.png"))

    fig4 = plot_smd(results, recipe, "power", y_label="Power (Recall)")
    fig4.savefig(str(plots_dir / "power.pdf"))
    fig4.savefig(str(plots_dir / "power.png"))


def make_plots(experiment_name: str = None):
    """
    If experiment_name is None, plot for all experiments
    """
    experiments = [x.name for x in SMD_RESULTS_DIR.glob('*') if x.is_dir()]
    if experiment_name:
        assert experiment_name in experiments, f"No such experiment: {experiment_name}"
        experiments = [experiment_name]

    for i, experiment in enumerate(experiments, start=1):
        print(f"Plot {i} of {len(experiments)}: {experiment}")
        make_plot_for(experiment)


@click.command()
@click.option("--experiment-name", "-e", default=None)
def main(experiment_name):
    make_plots(experiment_name)


if __name__ == "__main__":
    main()
