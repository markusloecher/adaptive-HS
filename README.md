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
Both classes inherit from `ShrinkageEstimator`, which extends `sklearn.base.BaseEstimator`.
Usage of these two classes is entirely analogous, and works just like any other `sklearn` estimator:
- `__init__()` parameters:
    - `base_estimator`: the estimator around which we "wrap" hierarchical shrinkage. This should be a tree-based estimator: `DecisionTreeClassifier`, `RandomForestClassifier`, ... (analogous for `Regressor`s)
    - `shrink_mode`: 4 options:
        - `"hs"`: classical Hierarchical Shrinkage (from Agarwal et al. 2022)
        $$
        \hat{f}(\mathbf{x}) = \mathbb{E}_{t_0}[y] + \sum_{l=1}^L\frac{\mathbb{E}_{t_l}[y] - \mathbb{E}_{t_{l-1}}[y]}{1 + \frac{\lambda}{N(t_{l-1})}}
        $$
        - `"hs_entropy"`: Augmented Hierarchical Shrinkage with added entropy term in the numerator of the fraction.
        $$
        \hat{f}(\mathbf{x}) = \mathbb{E}_{t_0}[y] + \sum_{l=1}^L\frac{\mathbb{E}_{t_l}[y] - \mathbb{E}_{t_{l-1}}[y]}{1 + \frac{\lambda H(t_{l-1})}{N(t_{l-1})}}
        $$
        - `"hs_log_cardinality"`: Augmented Hierarchical Shrinkage with log of cardinality term in numerator of the fraction.
        $$
        \hat{f}(\mathbf{x}) = \mathbb{E}_{t_0}[y] + \sum_{l=1}^L\frac{\mathbb{E}_{t_l}[y] - \mathbb{E}_{t_{l-1}}[y]}{1 + \frac{\lambda \log C(t_{l-1})}{N(t_{l-1})}}
        $$
        where $C(t)$ is the number of unique values in $t$. **Note that $\log C(t)$ is exactly the maximal entropy in node $t$.** i.e. this approach is a modification of `hs_entropy` where we assume that all feature values are uniformly distributed within the nodes.
    - `lmb`: lambda hyperparameter
    - `random_state`: random state for reproducibility
- Other functions: `fit(X, y)`, `predict(X)`, `predict_proba(X)`, `score(X, y)` work just like with any other `sklearn` estimator.

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
