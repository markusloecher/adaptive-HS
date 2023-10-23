from typing import Dict, List, Tuple

import numpy as np
from _simulate_data import simulate_data
from _util import TreeBasedModel
from shap import TreeExplainer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from adhs import ShrinkageClassifier


def run_single_replication(
    lmb: float,
    relevances: List[float],
    shrink_modes: List[str],
    base_estimator: TreeBasedModel,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    result_importances = []
    result_scores = []
    for relevance in relevances:
        relevance_str = "{:.2f}".format(relevance)[2:]
        X, y = simulate_data(1000, relevance)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y
        )

        for shrink_mode in shrink_modes:
            hsc = ShrinkageClassifier(
                base_estimator=base_estimator, lmb=lmb, shrink_mode=shrink_mode
            )

            hsc.fit(X_train, y_train)
            score = roc_auc_score(y_test, hsc.predict_proba(X_test)[:, 1])

            importances_record = {
                f"MDI_{i}": imp
                for i, imp in enumerate(hsc.estimator_.feature_importances_)
            }
            explainer = TreeExplainer(hsc.estimator_, X_train)
            # Shape: (n_outputs, n_samples, n_features)
            shap_values = np.array(
                explainer.shap_values(X_test, check_additivity=False)
            )
            for i in range(5):
                importances_record[f"SHAP_{i}"] = np.mean(
                    np.abs(shap_values[y_test, :, i])
                )

            importances_record["relevance"] = relevance_str
            importances_record["shrink_mode"] = shrink_mode
            result_importances.append(importances_record)

            result_scores.append(
                {
                    "relevance": relevance_str,
                    "shrink_mode": shrink_mode,
                    "ROC AUC": score,
                }
            )
    return result_importances, result_scores
