import click
import numpy as np

from pathlib import Path
from joblib import Parallel, delayed
from typing import Dict, List, Optional, Tuple

from fdrc.artificial_data import make_hypothesis
from fdrc.utils import TicTocTimer, save_json, timestamped_name
from fdrc.runner import run_filter_for_datasets, get_recipe
from fdrc.plots import plot_curves
from fdrc.experiment2.experiment2_recipes import EXPERIMENT2_RECIPES


RESULTS_FOLDER = Path(__file__).resolve().parents[3] / "results"
EXPERIMENT_FOLDER = RESULTS_FOLDER / "experiment2"


def experiment2(
    recipe: Dict,
    output_dir: Optional[str]
) -> Tuple:

    results, timer = [], TicTocTimer()

    # -----------------------
    # Part 1: Generate Data
    # -----------------------
    print("Generating datasets.")
    datasets = make_hypothesis(
        model=recipe["data_model"],
        params=recipe['data_params'],
        n=recipe["repeat"],
    )

    # -------------------------------------------
    # Part 2: Run Fixed Threshold Scorer
    # -------------------------------------------
    if "fixed_thresholds" in recipe:
        results.append(
            experiment2_main(
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
        delayed(experiment2_main)(
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

        figure_dir = output_dir / "figures"
        figure_dir.mkdir(parents=True)

        fig = make_fdp_curve(results)
        fig.savefig(str(figure_dir/"fdr_power_curve.png"))

        save_json(recipe, output_dir / "recipe.json", indent=4)
        save_json(datasets, output_dir / "data.json")
        save_json(results, output_dir / "results.json")

        print(f"Results are saved to: {output_dir.relative_to(RESULTS_FOLDER)}")

    print(f"Total time: {timer.toc()}")
    return recipe, datasets, results


def experiment2_main(
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
@click.option("--recipe-name", "-r", required=True)
def main(recipe_name):
    return experiment2(
        recipe=get_recipe(EXPERIMENT2_RECIPES, recipe_name),
        output_dir=timestamped_name(EXPERIMENT_FOLDER, recipe_name)
    )


if __name__ == "__main__":
    main()
