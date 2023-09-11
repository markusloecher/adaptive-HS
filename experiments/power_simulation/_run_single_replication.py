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

        # Handle each shrinkage mode separately
        for shrink_mode in shrink_modes:
            # Perform grid search for best value of lambda
            param_grid = {"shrink_mode": [shrink_mode], "lmb": lambdas}
            lmb_scores = cross_val_shrinkage(
                hsc,
                X,
                y,
                param_grid,
                n_splits=5,
                score_fn=roc_auc_score,
                n_jobs=1,
                return_param_values=False,
                verbose=0,
            )
            for lmb, score in zip(lambdas, lmb_scores):
                result_scores.append(
                    {
                        "relevance": rel_str,
                        "shrink_mode": shrink_mode,
                        "lambda": lmb,
                        "ROC AUC": score,
                    }
                )
            best_idx = np.argmax(lmb_scores)
            best_lmb = lambdas[best_idx]

            # Get feature importances for best value of lambda
            hsc.shrink_mode = shrink_mode
            hsc.lmb = best_lmb
            hsc.fit(X, y)
            importances_record = {
                f"MDI_{i}": imp
                for i, imp in enumerate(hsc.estimator_.feature_importances_)
            }
            importances_record["relevance"] = rel_str
            importances_record["shrink_mode"] = shrink_mode
            importances_record["lambda"] = best_lmb
            result_importances.append(importances_record)
    return result_importances, result_scores

