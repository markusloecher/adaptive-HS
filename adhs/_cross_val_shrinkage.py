from copy import deepcopy
from tqdm import tqdm

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.metrics import (
    roc_auc_score,
)
from joblib import Parallel, delayed


def cross_val_shrinkage(
    shrinkage_estimator,
    X,
    y,
    param_grid,
    n_splits=5,
    score_fn=None,  # Default: roc_auc_score
    n_jobs=-1,
    verbose=0,
    return_param_values=True,
):
    """
    Parallelization happens over the grid points, not over the folds.
    """
    if score_fn is None:
        score_fn = roc_auc_score
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    shrink_modes = param_grid["shrink_mode"]
    lmbs = param_grid["lmb"]

    # Create lists of all combinations of parameters
    param_shrink_mode = []
    param_lmb = []
    for shrink_mode in shrink_modes:
        for lmb in lmbs:
            param_shrink_mode.append(shrink_mode)
            param_lmb.append(lmb)

    def _train_model(estimator, train_index):
        """
        Helper function to train the base model for a single train-test split.
        """
        X_train, y_train = X[train_index], y[train_index]
        # Set shrink mode to a value that will be used in the grid search
        # This prevents computing node values for shrink modes that will
        # not be used in the grid search
        estimator.shrink_mode = shrink_modes[0]
        estimator.fit(X_train, y_train)
        return estimator

    def _single_setting(shrink_mode, lmb, fold_models):
        """
        Helper function to evaluate the performance of a single setting of the
        shrinkage parameters, on all folds.
        """
        scores = []
        for i, (train_index, test_index) in enumerate(cv.split(X)):
            X_test = X[test_index]
            y_test = y[test_index]
            X_train = X[train_index]
            y_train = y[train_index]

            if n_jobs != 1:
                estimator = deepcopy(fold_models[i])
            else:
                estimator = fold_models[i]
            estimator.reshrink(
                shrink_mode=shrink_mode, lmb=lmb, X=X_train, y=y_train
            )
            scores.append(score_fn(y_test, estimator.predict(X_test)))
        return np.mean(scores)

    # Train a model on each fold in parallel
    if verbose != 0:
        print("Training base models...")
    fold_models = [clone(shrinkage_estimator) for _ in range(n_splits)]
    if n_jobs != 1:
        with Parallel(n_jobs=n_jobs, verbose=verbose) as parallel:
            parallel(
                delayed(_train_model)(fold_models[i], train_index)
                for i, (train_index, _) in enumerate(cv.split(X))
            )
    else:
        gen = (
            enumerate(tqdm(cv.split(X)))
            if verbose == 1
            else enumerate(cv.split(X))
        )
        for i, (train_index, _) in gen:
            fold_models[i] = _train_model(fold_models[i], train_index)
    if verbose != 0:
        print("Done.")

    # Evaluate all settings in parallel
    if verbose != 0:
        print("Evaluating settings...")
    if n_jobs != 1:
        with Parallel(n_jobs=n_jobs, verbose=verbose) as parallel:
            scores = np.array(
                parallel(
                    delayed(_single_setting)(
                        param_shrink_mode[i], param_lmb[i], fold_models
                    )
                    for i in range(len(param_shrink_mode))
                )
            )
    else:
        param_range = (
            range(len(param_shrink_mode))
            if verbose == 0
            else tqdm(range(len(param_shrink_mode)))
        )
        scores = np.array(
            [
                _single_setting(
                    param_shrink_mode[i], param_lmb[i], fold_models
                )
                for i in param_range
            ]
        )
    if verbose != 0:
        print("Done.")

    if return_param_values:
        return scores, param_shrink_mode, param_lmb
    return scores
