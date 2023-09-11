from adhs import ShrinkageClassifier, ShrinkageRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from typing import List, Callable, Dict
from _util import TreeBasedModel


def run_single_replication(
    X,
    y,
    base_estimator: TreeBasedModel,
    shrink_modes: List[str],
    lambdas: List[float],
    problem_type: str,
    score_fn: Callable,
) -> Dict[str, List[float]]:
    """
    Performs a single replication of the experiment for a given dataset.
    Parallelization will be over the replications, so this function does not
    use parallelization, as it corresponds to a single process.
    """
    # Split the data
    train_index, test_index = train_test_split(
        np.arange(X.shape[0]), test_size=0.2
    )

    # shrink_mode -> [n_lambdas]
    scores = {sm: [] for sm in shrink_modes}

    # Make the model
    if problem_type == "regression":
        model = ShrinkageRegressor(base_estimator=base_estimator)
    elif problem_type == "classification":
        model = ShrinkageClassifier(base_estimator=base_estimator)
    else:
        raise ValueError("Unknown problem type")
    # Fit the model
    model.fit(X[train_index], y[train_index])

    # Compute scores for all shrink modes and lambdas
    for shrink_mode in shrink_modes:
        for lmb in lambdas:
            model.reshrink(shrink_mode=shrink_mode, lmb=lmb)
            scores[shrink_mode].append(
                score_fn(y[test_index], model.predict(X[test_index]))
            )

    return scores

