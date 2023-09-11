import pandas as pd
from numpy import typing as npt
from typing import List, Tuple
import numpy as np
from sklearn.utils import check_X_y
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import RegressorMixin
from copy import deepcopy


def check_fit_arguments(
    X: pd.DataFrame | npt.NDArray,
    y: pd.DataFrame | npt.NDArray,
    feature_names: List[str] | None = None,
) -> Tuple[npt.NDArray, npt.NDArray, List[str]]:
    """Check if the arguments to fit are valid:
    - X and y are valid
    - feature_names is valid, or can be inferred from X

    Parameters
    ----------
    X : pd.DataFrame | npt.NDArray
        Input data matrix.
    y : pd.DataFrame | npt.NDArray
        Target values.
    feature_names : List[str] | None
        Names of the features. If None, they will be inferred from X.

    Returns
    -------
    Tuple[npt.NDArray, npt.NDArray, List[str]]
        X, y, and feature_names, where feature_names is guaranteed to be a
        list of strings.
    """
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            X, y = check_X_y(X, y)
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    else:
        assert (
            len(feature_names) == X.shape[1]
        ), "Number of feature names must match number of features"

    X_arr = X if isinstance(X, np.ndarray) else X.values
    y_arr = y if isinstance(y, np.ndarray) else y.values
    X_arr, y_arr = check_X_y(X_arr, y_arr)
    return X_arr, y_arr, feature_names


def normalize_value(dt: DecisionTreeClassifier | DecisionTreeRegressor, node):
    # If regression, no normalization is necessary
    if isinstance(dt, RegressorMixin):
        return deepcopy(dt.tree_.value[node, :, :])
    # If classification, normalize count vector to probability vector
    return dt.tree_.value[node, :, :] / dt.tree_.n_node_samples[node]  # type: ignore
