# Scikit-Learn-compatible implementation of Adaptive Hierarchical Shrinkage
This directory contains an implementation of Adaptive Hierarchical Shrinkage that is compatible with Scikit-Learn. It exports 2 classes:
- `ShrinkageClassifier`
- `ShrinkageRegressor`

## Installation
### `adhs` Package
The `adhs` package, which contains the implementations of
Adaptive Hierarchical Shrinkage, can be installed using:
```
pip install .
```

### Experiments
To be able to run the scripts in the `experiments` directory, some extra
requirements are needed. These can be installed in a new conda
environment as follows:
```
conda create -n shrinkage python=3.10
conda activate shrinkage
pip install .[experiments]
```

## Basic API
This package exports 2 classes and 1 method:
- `ShrinkageClassifier`
- `ShrinkageRegressor`
- `cross_val_shrinkage`

### `ShrinkageClassifier` and `ShrinkageRegressor`
Both classes inherit from `ShrinkageEstimator`, which extends `sklearn.base.BaseEstimator`. Adaptive hierarchical shrinkage can be summarized as follows:
$$
\hat{f}(\mathbf{x}) = \mathbb{E}_{t_0}[y] + \sum_{l=1}^L\frac{\mathbb{E}_{t_l}[y] - \mathbb{E}_{t_{l-1}}[y]}{1 + \frac{g(t_{l-1})}{N(t_{l-1})}}
$$
where $g(t_{l-1})$ is some function of the node $t_{l-1}$. Classical hierarchical shrinkage (Agarwal et al. 2022) corresponds to $g(t_{l-1}) = \lambda$, where $\lambda$ is a chosen constant.

- `__init__()` parameters:
    - `base_estimator`: the estimator around which we "wrap" hierarchical shrinkage. This should be a tree-based estimator: `DecisionTreeClassifier`, `RandomForestClassifier`, ... (analogous for `Regressor`s)
    - `shrink_mode`: 6 options:
        - `"no_shrinkage"`: dummy value. This setting will not influence the `base_estimator` in any way, and is equivalent to just using the `base_estimator` by itself. Added for easy comparison between different modes of shrinkage and no shrinkage at all.
        - `"hs"`: classical Hierarchical Shrinkage (from Agarwal et al. 2022): $g(t_{l-1}) = \lambda$.
        - `"hs_entropy"`: Adaptive Hierarchical Shrinkage with added entropy term: $g(t_{l-1}) = \lambda H(t_{l-1})$.
        - `"hs_log_cardinality"`: Adaptive Hierarchical Shrinkage with log of cardinality term: $g(t_{l-1}) = \lambda \log C(t_{l-1})$ where $C(t)$ is the number of unique values in $t$.
        - `"hs_permutation"`: Adaptive Hierarchical Shrinkage with $g(t_{l-1}) = \frac{1}{\alpha(t_{l-1})}$, with $\alpha(t_{l-1}) = 1 - \frac{\Delta_\mathcal{I}(t_{l-1}, { }_\pi x(t_{l-1})) + \epsilon}{\Delta_\mathcal{I}(t_{l-1}, x(t_{l-1}))+ \epsilon}$
        - `"hs_global_permutation"`: Same as `"hs_permutation"`, but the data is permuted only once for the full dataset rather than once in each node.
    - `lmb`: $\lambda$ hyperparameter
    - `random_state`: random state for reproducibility
- `reshrink(shrink_mode, lmb, X)`: changes the shrinkage mode and/or lambda value in the shrinkage process. Calling `reshrink` with a given value of `shrink_mode` and/or `lmb` on an existing model is equivalent to fitting a new model with the same base estimator but the new, given values for `shrink_mode` and/or `lmb`. This method can avoid redundant computations in the shrinkage process, so can be more efficient than re-fitting a new `ShrinkageClassifier` or `ShrinkageRegressor`.
- Other functions: `fit(X, y)`, `predict(X)`, `predict_proba(X)`, `score(X, y)` work just like with any other `sklearn` estimator.

### `cross_val_shrinkage`
This method can be used to efficiently run cross-validation for the `shrink_mode` and/or `lmb` hyperparameters. As adaptive hierarchical shrinkage is a fully post-hoc procedure, cross-validation requires no retraining of the base model. This function exploits this property.

## Tutorials

- [General usage](notebooks/tutorial_general_usage.ipynb): Shows how to apply
hierarchical shrinkage on a simple dataset and access feature importances.
- [Cross-validating shrinkage parameters](notebooks/tutorial_shrinkage_cf.ipynb):
Hyperparameters for (augmented) hierarchical shrinkage (i.e. `shrink_mode` and
`lmb`) can be tuned using cross-validation, without having to retrain the
underlying model. This is because (augmented) hierarchical shrinkage is a
**fully post-hoc** procedure. As the `ShrinkageClassifier` and
`ShrinkageRegressor` are valid scikit-learn estimators, you could simply tune
these hyperparameters using [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) as you would do with any other scikit-learn
model. However, this **will** retrain the decision tree or random forest, which
leads to unnecessary performance loss. This notebook shows how you can use our
cross-validation function to cross-validate `shrink_mode` and `lmb` without
this performance loss.
