from typing import Dict, List, Optional, Tuple
from fdrc.filters import build_filter
from fdrc.fdr_scoring import power_score, fdp_score


def run_filter_for_datasets(
    datasets: List[Dict],
    filter_name: str,
    filter_params: Dict,
    decay_fdr_delta: float = None
) -> Tuple:
    """
    Runs given filter on datasets and returns raw data (labels) and scorings.

    We want to calculate decay fdr for non-decay models too:
        For decay models, decay_fdr_delta is read from the filter object.
        For non-decay models, decay_fdr_delta needs to be set manually.

    """
    power, fdr, decay_fdr, labels_list = [], [], [], []
    fdr_filter = build_filter(filter_name, filter_params)

    if decay_fdr_delta is None:
        decay_fdr_delta = getattr(fdr_filter, "delta", 1.0)

    for dataset in datasets:
        labels, _ = fdr_filter.step(dataset["p_values"])
        labels_list.append(labels)
        power.append(power_score(labels, dataset["labels"]))
        fdr.append(fdp_score(labels, dataset["labels"]))
        decay_fdr.append(fdp_score(labels, dataset["labels"], delta=decay_fdr_delta))

    return {
        "labels": labels_list,
        "power": power,
        "fdr": fdr,
        "decay_fdr": decay_fdr,
    }


def get_recipe(
    recipes: Dict,
    recipe_name: Optional[str] = None,
    overwrites: Optional[Dict] = None
) -> Dict:
    """
    First, loads the default recipe then overwrites it with the given recipe.
    You can overwrite single model parameter.
    """
    def overwrite(_recipe, _overwrites):
        for key, value in _overwrites.items():
            if key == "filter_params":
                for _filter_name, _filter_params in value.items():
                    _recipe["filter_params"][_filter_name].update(_filter_params)
            else:
                _recipe[key] = value

    recipe = recipes["default"].copy()

    # Overwrite params from recipe
    if recipe_name is not None and recipe_name != "default":
        overwrite(recipe, recipes[recipe_name])

    # Overwrite params from command line
    if overwrites:
        overwrite(recipe, overwrites)

    return recipe
