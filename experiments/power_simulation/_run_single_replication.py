import numpy as np
from typing import Tuple, List, Dict
from adhs import ShrinkageClassifier, cross_val_shrinkage
from sklearn.metrics import roc_auc_score
from _simulate_data import simulate_data
from _util import TreeBasedModel


def run_single_replication(
    lambdas: List[float],
    relevances: List[float],
    shrink_modes: List[str],
    base_estimator: TreeBasedModel,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    relevances_str = ["{:.2f}".format(rel)[2:] for rel in relevances]
    result_importances = []
    result_scores = []
    for i, relevance in enumerate(relevances):
        rel_str = relevances_str[i]
        X, y = simulate_data(1000, relevance)

        # Create base classifier
        hsc = ShrinkageClassifier(base_estimator=base_estimator)

        # Run cross-validation to get best lambda for each shrink mode
        scores, param_shrink_mode, param_lmb = cross_val_shrinkage(
            hsc,
            X,
            y,
            {"shrink_mode": shrink_modes, "lmb": lambdas},
            n_splits=5,
            score_fn=roc_auc_score,
            n_jobs=1,
            return_param_values=True,
            verbose=0,
        )
        # Save scores and get best score for each shrink mode
        best_scores = {sm: -np.inf for sm in shrink_modes}
        best_lambdas = {sm: None for sm in shrink_modes}
        for score, shrink_mode, lmb in zip(scores, param_shrink_mode, param_lmb):
            result_scores.append(
                {
                    "relevance": rel_str,
                    "shrink_mode": shrink_mode,
                    "lambda": lmb,
                    "ROC AUC": score,
                }
            )
            if score > best_scores[shrink_mode]:
                best_scores[shrink_mode] = score
                best_lambdas[shrink_mode] = lmb
        
        # Get feature importances for best lambda for each shrink mode
        for shrink_mode in shrink_modes:
            hsc.shrink_mode = shrink_mode
            hsc.lmb = best_lambdas[shrink_mode]
            hsc.fit(X, y)
            importances_record = {
                f"MDI_{i}": imp
                for i, imp in enumerate(hsc.estimator_.feature_importances_)
            }
            importances_record["relevance"] = rel_str
            importances_record["shrink_mode"] = shrink_mode
            importances_record["lambda"] = best_lambdas[shrink_mode]
            result_importances.append(importances_record)
    return result_importances, result_scores

