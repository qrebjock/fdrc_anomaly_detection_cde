import click
import json
import numpy as np

from pathlib import Path
from joblib import Parallel, delayed
from typing import Dict, List, Optional, Tuple

from fdrc.utils import TicTocTimer, save_json, timestamped_name
from fdrc.runner import run_filter_for_datasets, get_recipe
from fdrc.plots import plot_curves
from fdrc.experiment_smd.experiment_smd_recipes import EXPERIMENT_SMD_RECIPES


RESULTS_FOLDER = Path(__file__).resolve().parents[3] / "results"
EXPERIMENT_FOLDER = RESULTS_FOLDER / "experiment_smd"
SMD_P_VALUES_FOLDER = "/home/barisk/fdrc/results"
DATASETS = [
    "machine-1-1",
    "machine-1-2",
    "machine-1-3",
    "machine-1-4",
    "machine-1-5",
    "machine-1-6",
    "machine-1-7",
    "machine-1-8",
    "machine-2-1",
    "machine-2-2",
    "machine-2-3",
    "machine-2-4",
    "machine-2-5",
    "machine-2-6",
    "machine-2-7",
    "machine-2-8",
    "machine-2-9",
    "machine-3-1",
    "machine-3-2",
    "machine-3-3",
    "machine-3-4",
    "machine-3-5",
    "machine-3-6",
    "machine-3-7",
    "machine-3-8",
    "machine-3-9",
    "machine-3-10",
    "machine-3-11",
]


def load_smd_p_values(metric):
    def __load__(dataset_name):
        filename = Path(SMD_P_VALUES_FOLDER) / dataset_name / "p_values_labels.json"
        with open(filename, "r") as fp:
            d = json.load(fp)
            return {
                "labels": d["labels"],
                "p_values": d["p_values"][metric]
            }
    return [__load__(dataset_name) for dataset_name in DATASETS]


def experiment_smd(
    recipe: Dict,
    output_dir: Optional[str] = None,
    make_figures: bool = False
) -> Tuple:

    results = []
    timer = TicTocTimer()

    print(f"Running recipe:")
    print(recipe)
    print(f"----------------")

    # -------------------------------------------
    # Part 1: Read Datasets
    # -------------------------------------------
    metric = recipe["metric"]
    print(f"Loading Datasets with metric: {metric}")
    datasets = load_smd_p_values(metric)
    print(f"Loaded {len(datasets)} datasets.")

    # -------------------------------------------
    # Part 2: Run Fixed Threshold Scorer
    # -------------------------------------------
    if "fixed_thresholds" in recipe:
        results.append(
            experiment_smd_main(
                "FixedThresholdFilter",
                datasets,
                param_name="threshold",
                param_values=recipe["fixed_thresholds"],
                decay_fdr_delta=recipe["delta"],
            )
        )

    # -------------------------------------------
    # Part 3: Run given FDRC Filters
    # -------------------------------------------

    # Apply shared params:
    for filter_name, filter_param in recipe["filter_params"].items():
        if "gamma_size" in recipe:
            filter_param["gamma_size"] = recipe["gamma_size"]
        if "delta" in recipe and filter_name.startswith("Decay"):
            filter_param["delta"] = recipe["delta"]

    n_jobs = len(recipe["filters"])
    results += Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(experiment_smd_main)(
            filter_name,
            datasets,
            param_name="fdr_target",
            param_values=recipe["fdr_targets"],
            filter_params=recipe["filter_params"][filter_name],
            decay_fdr_delta=recipe["delta"],
        ) for filter_name in recipe["filters"]
    )

    # -------------------------------------------
    # Part 4: Save Results
    # -------------------------------------------
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True)

        save_json(recipe, output_dir / "recipe.json")
        save_json(datasets, output_dir / "data.json")
        save_json(results, output_dir / "results.json")

        if make_figures:
            figure_dir = output_dir / "figures"
            figure_dir.mkdir(parents=True)

            fig = make_fdp_curve(results)
            fig.savefig(str(figure_dir / "fdr_power_curve.png"))

        print(f"Results are saved to: {output_dir.relative_to(RESULTS_FOLDER)}")

    print(f"Total time: {timer.toc()}")
    return recipe, datasets, results


def experiment_smd_main(
    filter_name: str,
    datasets: List[Dict],
    param_name: str,
    param_values: List[float],
    filter_params: Optional[Dict] = None,
    decay_fdr_delta: Optional[float] = None,
) -> Dict:
    """
    Runs the filter on each dataset for all values in param_values and calculates
    mean/std FDP and Power scores. Decay FDR is also calculated for all decay and
    non-decay filters.
    """
    def get_stats(x):
        return np.nanmean(x), np.nanstd(x)

    filter_params = filter_params if filter_params is not None else {}

    decay_fdr_delta = None if filter_name.startswith("Decay") else decay_fdr_delta

    print(f"{filter_name} started.")
    timer = TicTocTimer()

    power, fdr, decay_fdr = [], [], []
    for i, param_value in enumerate(param_values):
        filter_params[param_name] = param_value
        results = run_filter_for_datasets(
            datasets, filter_name, filter_params, decay_fdr_delta
        )
        power.append(get_stats(results["power"]))
        fdr.append(get_stats(results["fdr"]))
        decay_fdr.append(get_stats(results["decay_fdr"]))

    print(f"{filter_name} finished. Elapsed time: {str(timer.toc())}")
    return {
        "name": filter_name,
        "power": power,
        "fdr": fdr,
        "decay_fdr": decay_fdr
    }


def make_fdp_curve(results):
    return plot_curves(
        x=[[x for x, _ in result["fdr"]] for result in results],
        y=[[x for x, _ in result["power"]] for result in results],
        y_errors=None,
        filter_names=[result["name"] for result in results],
        x_label="FDR (1-Precision)",
        y_label="Power (Recall)",
        legend_loc="lower right"
    )


@click.command()
@click.option("--recipe-name", "-r", default="default")
def main(recipe_name):
    return experiment_smd(
        recipe=get_recipe( EXPERIMENT_SMD_RECIPES, recipe_name),
        output_dir=timestamped_name(EXPERIMENT_FOLDER, recipe_name)
    )


if __name__ == "__main__":
    main()
