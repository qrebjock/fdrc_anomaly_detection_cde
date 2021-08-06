import click
import numpy as np

from pathlib import Path
from joblib import Parallel, delayed
from typing import Dict, List, Optional, Tuple

from fdrc.artificial_data import make_hypothesis
from fdrc.utils import TicTocTimer, save_json, timestamped_name
from fdrc.runner import run_filter_for_datasets, get_recipe
from fdrc.plots import plot_curves
from fdrc.experiment1.experiment1_recipes import EXPERIMENT1_RECIPES


RESULTS_FOLDER = Path(__file__).resolve().parents[3] / "results"
EXPERIMENT_FOLDER = RESULTS_FOLDER / "experiment1"


def experiment1(
    recipe: Dict,
    output_dir: Optional[str] = None,
    make_figures: bool = False
) -> Tuple:

    timer = TicTocTimer()

    # -----------------------
    # Part 1: Generate Data
    # -----------------------
    print("Generating datasets.")
    datasets_list = []
    for anomaly_ratio, rep in zip(recipe["anomaly_ratios"], recipe["repeats"]):
        params = recipe["data_params"].copy()
        params["anomaly_ratio"] = anomaly_ratio
        datasets_list.append(
            make_hypothesis(recipe["data_model"], params, n=rep)
        )

    # --------------------------------
    # Part 2: Run given FDRC Filters
    # --------------------------------

    # Apply shared params:
    for filter_name, filter_param in recipe["filter_params"].items():
        if "fdr_target" in recipe:
            filter_param["fdr_target"] = recipe["fdr_target"]
        if "gamma_size" in recipe:
            filter_param["gamma_size"] = recipe["gamma_size"]
        if "delta" in recipe and filter_name.startswith("Decay"):
            filter_param["delta"] = recipe["delta"]

    n_jobs = len(recipe["filters"])
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(experiment1_main)(
            filter_name=filter_name,
            filter_params=recipe["filter_params"][filter_name],
            datasets_list=datasets_list,
            decay_fdr_delta=recipe["delta"]
        ) for filter_name in recipe["filters"]
    )

    # --------------------------------
    # Part 3: Save Results
    # --------------------------------
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True)

        save_json(recipe, output_dir / "recipe.json", indent=4)
        save_json(datasets_list, output_dir / "data.json")
        save_json(results, output_dir / "results.json")
        print(f"Results are saved to: {output_dir.relative_to(RESULTS_FOLDER)}")

        if make_figures:
            make_curves(recipe, results, output_dir)

    print(f"Total time: {timer.toc()}")
    return recipe, datasets_list, results


def experiment1_main(
    filter_name: str,
    filter_params: Dict,
    datasets_list: List,
    decay_fdr_delta: float
) -> Dict:

    def get_stats(x):
        return np.nanmean(x), np.nanstd(x)

    print(f"{filter_name} started.")
    timer = TicTocTimer()

    labels, power, fdr, decay_fdr = [], [], [], []
    for datasets in datasets_list:
        results = run_filter_for_datasets(
            datasets, filter_name, filter_params, decay_fdr_delta
        )
        labels.append(results["labels"])
        power.append(get_stats(results["power"]))
        fdr.append(get_stats(results["fdr"]))
        decay_fdr.append(get_stats(results["decay_fdr"]))

    print(f"{filter_name} finished. Elapsed time: {str(timer.toc())}")
    return {
        "name": filter_name,
        "labels": labels,
        "power": power,
        "fdr": fdr,
        "decay_fdr": decay_fdr
    }


def make_curves(recipe, results, output_dir):
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig = make_power_curve(recipe, results)
    fig.savefig(str(figure_dir / "power_curve.eps"), dpi=300)

    fig = make_fdr_curve(recipe, results)
    fig.savefig(str(figure_dir / "fdr_curve.eps"), dpi=300)

    fig = make_decay_fdr_curve(recipe, results)
    fig.savefig(str(figure_dir / "decay_fdr_curve.eps"), dpi=300)

    print(f"Figures are saved to: {figure_dir.relative_to(RESULTS_FOLDER)}")


def make_power_curve(recipe, results):
    anomaly_ratios = recipe["anomaly_ratios"]
    repeats = recipe["repeats"]
    y_errors = [
        np.asarray([x for _, x in result["power"]]) / np.sqrt(repeats)
        for result in results
    ]

    return plot_curves(
        x=[anomaly_ratios] * len(results),
        y=[[x for x, _ in result["power"]] for result in results],
        y_errors=y_errors,
        filter_names=[result["name"] for result in results],
        x_label="Proportion of Anomalies $\pi_1$",
        y_label="Power (Recall)",
        invert_xaxis=True,
        log_xscale=True,
        legend_loc="lower left",
        fig_size=(12, 4),
        label_size=20,
        legend_size=12
    )


def make_fdr_curve(recipe, results, target: Optional[int] = None):
    anomaly_ratios = recipe["anomaly_ratios"]
    repeats = recipe["repeats"]
    y_errors = [
        np.asarray([x for _, x in result["fdr"]]) / np.sqrt(repeats)
        for result in results
    ]
    return plot_curves(
        x=[anomaly_ratios] * len(results),
        y=[[x for x, _ in result["fdr"]] for result in results],
        y_errors=y_errors,
        filter_names=[result["name"] for result in results],
        x_label="Proportion of Anomalies $\pi_1$",
        y_label="FDR (1-Precision)",
        invert_xaxis=True,
        log_xscale=True,
        legend_loc="upper left",
        fig_size=(12, 4),
        label_size=20,
        legend_size=12,
        target=target,
    )


def make_decay_fdr_curve(recipe, results, target=None, y_lim=None):
    anomaly_ratios = recipe["anomaly_ratios"]
    repeats = recipe["repeats"]
    y_errors = [
        np.asarray([x for _, x in result["decay_fdr"]]) / np.sqrt(repeats)
        for result in results
    ]
    return plot_curves(
        x=[anomaly_ratios] * len(results),
        y=[[x for x, _ in result["decay_fdr"]] for result in results],
        y_errors=y_errors,
        filter_names=[result["name"] for result in results],
        x_label="Proportion of Anomalies $\pi_1$",
        y_label="FDR$_\delta$ (1-Precision)",
        invert_xaxis=True,
        log_xscale=True,
        legend_loc="upper left",
        fig_size=(12, 4),
        label_size=20,
        legend_size=12,
        target=target,
        y_lim=y_lim,
    )


@click.command()
@click.option("--recipe-name", "-r", required=True)
@click.option("--make-figures", "-f", is_flag=True)
def main(recipe_name, make_figures):
    return experiment1(
        recipe=get_recipe(EXPERIMENT1_RECIPES, recipe_name),
        output_dir=timestamped_name(EXPERIMENT_FOLDER, recipe_name),
        make_figures=make_figures
    )


if __name__ == "__main__":
    main()
