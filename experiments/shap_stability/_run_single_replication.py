from typing import List
import pandas as pd
from numpy import typing as npt
from sklearn.model_selection import train_test_split
from adhs import cross_val_shrinkage, ShrinkageClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
from shap import TreeExplainer


def run_single_replication(
    feature_names: List[str],
    shrink_modes: List[str],
    lambdas: List[float],
    X: npt.NDArray,
    y: npt.NDArray,
    X_shap: npt.NDArray,
    y_shap: npt.NDArray,
):
    # Randomly sample 2/3 of X, y
    X_train, _, y_train, _ = train_test_split(X, y, test_size=1/3, stratify=y)

    hsc = ShrinkageClassifier(base_estimator=RandomForestClassifier())

    result_dict = {
        "shrink_mode": [],
        "sample_idx": [],
    }
    for feature_name in feature_names:
        result_dict[f"SHAP_{feature_name}"] = []
    
    for shrink_mode in shrink_modes:
        # Get best lambda using CV
        param_grid = {"shrink_mode": [shrink_mode], "lmb": lambdas}
        lmb_scores = cross_val_shrinkage(
            hsc,
            X_train,
            y_train,
            param_grid,
            n_splits=5,
            score_fn=roc_auc_score,
            n_jobs=1,
            return_param_values=False,
            verbose=0
        )
        best_idx = np.argmax(lmb_scores)
        best_lmb = lambdas[best_idx]

        # Train model with best lambda on full train set
        hsc.shrink_mode = shrink_mode
        hsc.lmb = best_lmb
        hsc.fit(X_train, y_train)

        # Compute SHAP values on X_shap, y_shap
        explainer = TreeExplainer(hsc.estimator_, X_train)
        # [50, n_features]
        shap_values = np.array(explainer.shap_values(X_shap, check_additivity=False))

        # Save SHAP values
        result_dict["shrink_mode"] += [shrink_mode] * 50
        result_dict["sample_idx"] += list(range(50))
        for i, feature_name in enumerate(feature_names):
            result_dict[f"SHAP_{feature_name}"] += shap_values[y_shap, np.arange(50), i].tolist()
    
    # Return results
    return pd.DataFrame.from_dict(result_dict)