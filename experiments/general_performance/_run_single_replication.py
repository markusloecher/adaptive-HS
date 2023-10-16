from typing import Callable, Dict, List, Tuple

import numpy as np
from _util import TreeBasedModel
from numpy import typing as npt
from sklearn.model_selection import train_test_split, StratifiedKFold

from adhs import ShrinkageClassifier, ShrinkageRegressor


def run_single_replication(
    X,
    y,
    base_estimator: TreeBasedModel,
    shrink_modes: List[str],
    lambdas: List[float],
    problem_type: str,
    score_fn: Callable,
) -> Tuple[npt.NDArray, str, float]:
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
        np.arange(X.shape[0]), test_size=0.33, stratify=y
    )
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print("Labels:", np.unique(y_train))

    # Make the model
    if problem_type == "regression":
        model = ShrinkageRegressor(base_estimator=base_estimator)
    elif problem_type == "classification":
        model = ShrinkageClassifier(base_estimator=base_estimator)
    else:
        raise ValueError("Unknown problem type")

    # Use predict_proba if model is classifier
    predict_fn = model.predict
    if isinstance(model, ShrinkageClassifier):
        predict_fn = model.predict_proba

    # Use 3-fold CV to tune hyperparameters
    skf = StratifiedKFold(n_splits=3)
    scores: List[Dict[Tuple, float]] = []  # (shrink_mode, lmb) -> score
    for cv_train_idx, cv_val_idx in skf.split(X_train, y_train):
        # Select samples
        X_train_cv, X_val = X_train[cv_train_idx], X_train[cv_val_idx]
        y_train_cv, y_val = y_train[cv_train_idx], y_train[cv_val_idx]

        # Fit the model
        model.fit(X_train_cv, y_train_cv)

        # Compute scores for all shrink modes and lambdas
        fold_scores: Dict[Tuple, float] = {}  # (shrink_mode, lmb) -> score
        for shrink_mode in shrink_modes:
            for lmb in lambdas:
                # Shrink without retraining
                model.reshrink(
                    shrink_mode=shrink_mode,
                    lmb=lmb,
                    X=X_train_cv,
                    y=y_train_cv,
                )

                # Compute scores
                predictions = predict_fn(X_val)
                fold_scores[(shrink_mode, lmb)] = score_fn(
                    y_val, predictions[:, 0]
                )
        scores.append(fold_scores)

    # Get average scores over the folds
    avg_scores = {}
    for shrink_mode in shrink_modes:
        for lmb in lambdas:
            avg_scores[(shrink_mode, lmb)] = np.mean(
                [score[(shrink_mode, lmb)] for score in scores]
            )

    # Get the best hyperparameters
    best_shrink_mode, best_lmb = max(avg_scores, key=lambda k: avg_scores[k])

    # Retrain using best hyperparameters
    model.shrink_mode = best_shrink_mode
    model.lmb = best_lmb
    model.fit(X_train, y_train)

    # Compute test scores using the best hyperparameters
    predictions = predict_fn(X_test, individual_trees=True)
    # TODO this does not yet work for single DT models
    result = np.array(
        [
            score_fn(y_test, np.average(predictions[:i], axis=1))
            for i in range(1, len(predictions) + 1)
        ]
    )
    return result, best_shrink_mode, best_lmb
