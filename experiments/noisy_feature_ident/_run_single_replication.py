from typing import Callable, Dict, List

import numpy as np
from _util import TreeBasedModel
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from adhs import ShrinkageClassifier, ShrinkageRegressor
import ipdb

def _prepare_data(X, y, problem_type):
    # Split the data
    train_index, test_index = train_test_split(
        np.arange(X.shape[0]),
        test_size=0.33,
        stratify=y if problem_type == "classification" else None,
    )
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # kfold doesn't shuffle by default so the same folds are used for all
    # shrinkage modes
    kfold = (
        KFold(n_splits=3)
        if problem_type == "regression"
        else StratifiedKFold(n_splits=3)
    )
    return X_train, X_test, y_train, y_test, kfold


def _get_model(
    base_estimator: TreeBasedModel,
    problem_type: str,
):
    if problem_type == "regression":
        model = ShrinkageRegressor(base_estimator=base_estimator)
        predict_fn = lambda x, individual_trees: model.predict(
            x, individual_trees=individual_trees
        )
    elif problem_type == "classification":
        model = ShrinkageClassifier(base_estimator=base_estimator)
        predict_fn = lambda x, individual_trees: model.predict_proba(
            x, individual_trees=individual_trees
        )[..., 1]
    else:
        raise ValueError("Unknown problem type")
    return model, predict_fn


def _get_best_lambda(
    model,
    predict_fn,
    shrink_mode,
    lambdas,
    X_train,
    y_train,
    kfold,
    score_fn,
):
    scores: List[Dict[float, float]] = []
    for cv_train_idx, cv_val_idx in kfold.split(X_train, y_train):
        # Select samples
        X_train_cv, X_val = X_train[cv_train_idx], X_train[cv_val_idx]
        y_train_cv, y_val = y_train[cv_train_idx], y_train[cv_val_idx]

        # Fit the model
        model.fit(X_train_cv, y_train_cv)

        # Compute scores for all lambdas
        fold_scores: Dict[float, float] = {}  # {lmb -> score}
        for lmb in lambdas:
            # Shrink without retraining
            model.reshrink(
                shrink_mode=shrink_mode,
                lmb=lmb,
                X=X_train_cv,
                y=y_train_cv,
            )

            # Compute scores
            predictions = predict_fn(X_val, individual_trees=False)
            fold_scores[lmb] = score_fn(y_val, predictions)
        scores.append(fold_scores)
    # Get average scores over the folds
    avg_scores: Dict[float, float] = {
        lmb: np.mean([fold_scores[lmb] for fold_scores in scores])
        for lmb in lambdas
    }
    # Get best lambda
    return max(avg_scores.keys(), key=lambda k: avg_scores[k])


def run_single_replication(
    X,
    y,
    base_estimator: TreeBasedModel,
    shrink_modes: List[str],
    lambdas: List[float],
    problem_type: str,
    score_fn: Callable,
    individual_trees=True
) -> List[Dict[str, float | int | str]]:
    """
    Performs a single replication of the experiment for a given dataset.
    Parallelization will be over the replications, so this function does not
    use parallelization, as it corresponds to a single process.

    Returns a list of records, where each record shows the performance of a
    given shrink_mode with cross-validated lambda (k=3), for a given number of
    trees.
    """
    X_train, X_test, y_train, y_test, kfold = _prepare_data(X, y, problem_type)
    model, predict_fn = _get_model(base_estimator, problem_type)
    #ipdb.set_trace()

    result = []
    for shrink_mode in shrink_modes:
        model.shrink_mode = shrink_mode
        best_lmb = (
            _get_best_lambda(
                model,
                predict_fn,
                shrink_mode,
                lambdas,
                X_train,
                y_train,
                kfold,
                score_fn,
            )
            if shrink_mode != "no_shrinkage"
            else 0.0
        )

        model.lmb = best_lmb
        model.fit(X_train, y_train)

        # Shape: [num_trees, num_samples]
        predictions = predict_fn(X_test, individual_trees=individual_trees)
        num_trees = predictions.shape[0]
        shrink_mode_scores = np.array(
            [
                score_fn(y_test, np.average(predictions[:i], axis=0))
                for i in range(1, len(predictions) + 1)
            ]
        )

        score_key = "R2" if problem_type == "regression" else "ROC AUC"
        result += [
            {
                "shrink_mode": shrink_mode,
                "lambda": best_lmb,
                "num_trees": i + 1,
                score_key: shrink_mode_scores[i],
            }
            for i in range(num_trees)
        ]
    return result
