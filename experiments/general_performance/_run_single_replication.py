from typing import Callable, Dict, List

import numpy as np
from _util import TreeBasedModel
from numpy import typing as npt
from sklearn.model_selection import train_test_split

from adhs import ShrinkageClassifier, ShrinkageRegressor


def run_single_replication(
    X,
    y,
    base_estimator: TreeBasedModel,
    shrink_modes: List[str],
    lambdas: List[float],
    problem_type: str,
    score_fn: Callable,
) -> Dict[str, npt.NDArray]:
    """
    Performs a single replication of the experiment for a given dataset.
    Parallelization will be over the replications, so this function does not
    use parallelization, as it corresponds to a single process.

    Returns a dictionary mapping shrink_mode to a NumPy array of shape
        (len(lambdas), num_trees)
    where num_trees is the number of trees in the base estimator.
    Element (i,j) corresponds to the score achieved by setting lambda to
    lambdas[i] and using only the first [j] trees.
    """
    # Split the data
    train_index, test_index = train_test_split(
        np.arange(X.shape[0]), test_size=0.2, stratify=y
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
            model.reshrink(
                shrink_mode=shrink_mode,
                lmb=lmb,
                X=X[train_index],
                y=y[train_index],
            )
            predict_fn = model.predict
            if isinstance(model, ShrinkageClassifier):
                predict_fn = model.predict_proba

            predictions = predict_fn(X[test_index], individual_trees=True)
            scores[shrink_mode].append(
                [
                    score_fn(y[test_index], np.average(predictions[:i]))
                    for i in range(1, len(predictions) + 1)
                ]
            )
    result = {}
    for shrink_mode in scores:
        result[shrink_mode] = np.array(scores[shrink_mode])
    return result
