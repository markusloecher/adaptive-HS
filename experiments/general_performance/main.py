import argparse
import os
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np

from _util import CLF_DATASETS, REG_DATASETS, SHRINKAGE_TYPES, EXPERIMENTS
from _run_experiment import run_experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, required=True)
    parser.add_argument("--n-replications", type=int, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument(
        "--shrink-modes",
        type=str,
        nargs="+",
        choices=SHRINKAGE_TYPES,
        default=SHRINKAGE_TYPES,
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        choices=EXPERIMENTS,
        default=EXPERIMENTS,
    )
    args = parser.parse_args()

    lambdas = [0.0, 0.1, 1.0, 10.0, 25.0, 50.0, 100.0]
    np.seterr(all="raise")

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if "classification_dt" in args.experiments:
        run_experiment(
            CLF_DATASETS,
            DecisionTreeClassifier(),
            args.shrink_modes,
            lambdas,
            "classification",
            roc_auc_score,
            args.n_jobs,
            args.n_replications,
            args.out_dir,
            "classification_dt",
        )
    if "classification_rf" in args.experiments:
        run_experiment(
            CLF_DATASETS,
            RandomForestClassifier(n_estimators=10),
            args.shrink_modes,
            lambdas,
            "classification",
            roc_auc_score,
            args.n_jobs,
            args.n_replications,
            args.out_dir,
            "classification_rf",
        )
    if "regression" in args.experiments:
        run_experiment(
            REG_DATASETS,
            DecisionTreeRegressor(),
            args.shrink_modes,
            lambdas,
            "regression",
            r2_score,
            args.n_jobs,
            args.n_replications,
            args.out_dir,
            "regression",
        )
