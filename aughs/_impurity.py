import numpy as np
from ._criterion import _CRITERION_FNS


def impurity_reduction(criterion_fn, X, y, feature, threshold):
    # Compute impurity reduction of the split
    split_feature = X[:, feature]
    y_left = y[split_feature <= threshold]
    y_right = y[split_feature > threshold]

    return (
        len(y) * criterion_fn(y)
        - len(y_left) * criterion_fn(y_left)
        - len(y_right) * criterion_fn(y_right)
    )


def best_impurity_reduction(X, y, feature, criterion):
    # Compute impurity reduction of the split
    thresholds = np.unique(X[:, feature])
    best_impurity_reduction = -np.inf
    criterion_fn = _CRITERION_FNS[criterion]

    for threshold in thresholds[:-1]:
        cur_impurity_reduction = impurity_reduction(
            criterion_fn, X, y, feature, threshold
        )
        if cur_impurity_reduction > best_impurity_reduction:
            best_impurity_reduction = cur_impurity_reduction
    return best_impurity_reduction