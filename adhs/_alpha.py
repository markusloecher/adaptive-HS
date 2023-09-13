from ._criterion import _CRITERION_FNS
from ._impurity import impurity_reduction, best_impurity_reduction
import numpy as np
from numpy import typing as npt


def compute_alpha(X_cond, y_cond, feature, threshold, criterion) -> npt.NDArray:
    # Compute original impurity reduction
    criterion_fn = _CRITERION_FNS[criterion]
    orig_impurity_reduction = impurity_reduction(
        criterion_fn, X_cond, y_cond, feature, threshold
    )

    # Compute best impurity reduction for the shuffled labels
    y_shuffled = np.random.permutation(y_cond)
    best_shuffled_impurity_reduction = best_impurity_reduction(
        X_cond, y_shuffled, feature, criterion
    )

    # Compute alpha
    # Adding \epsilon to both terms to prevent extreme values of alpha
    best_shuffled_impurity_reduction = best_shuffled_impurity_reduction + 1e-4
    orig_impurity_reduction = orig_impurity_reduction + 1e-4
    alpha = 1 - best_shuffled_impurity_reduction / orig_impurity_reduction
    # Also adding \epsilon to alpha itself, since it will be used as a
    # denominator
    alpha = np.maximum(alpha, 0) + 1e-4
    return 1 / alpha


def compute_global_alpha(
    X_cond, X_cond_shuffled, y_cond, feature, threshold, criterion
) -> npt.NDArray:
    """
    This is basically the same function as _compute_alpha, but instead of
    shuffling the labels, we use a previously shuffled version of the columns.
    This ensures that the same permutation is used for all nodes.
    """

    # Compute original impurity reduction
    criterion_fn = _CRITERION_FNS[criterion]
    orig_impurity_reduction = impurity_reduction(
        criterion_fn, X_cond, y_cond, feature, threshold
    )

    # Compute best impurity reduction for the shuffled labels
    best_shuffled_impurity_reduction = best_impurity_reduction(
        X_cond_shuffled, y_cond, feature, criterion
    )

    # Compute alpha
    # Adding \epsilon to both terms to prevent extreme values of alpha
    best_shuffled_impurity_reduction = best_shuffled_impurity_reduction + 1e-4
    orig_impurity_reduction = orig_impurity_reduction + 1e-4
    alpha = 1 - best_shuffled_impurity_reduction / orig_impurity_reduction
    # Also adding \epsilon to alpha itself, since it will be used as a
    # denominator
    alpha = np.maximum(alpha, 0) + 1e-4
    return 1 / alpha
