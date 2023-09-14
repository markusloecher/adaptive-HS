import argparse
import pandas as pd
from _util import CLF_DATASETS, SHRINKAGE_TYPES
from tqdm import trange
from _run_single_replication import run_single_replication
import joblib
from imodels.util.data_util import get_clean_dataset
from sklearn.model_selection import train_test_split
from typing import List
import os


if __name__ == "__main__":
    dataset_names = [ds[0] for ds in CLF_DATASETS]
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int)
    parser.add_argument("--n-replications", type=int)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        choices=dataset_names,
        default=dataset_names,
    )
    parser.add_argument(
        "--shrink-modes",
        type=str,
        nargs="+",
        choices=SHRINKAGE_TYPES,
        default=SHRINKAGE_TYPES,
    )
    args = parser.parse_args()

    lambdas = [0.0, 0.1, 1.0, 10.0, 25.0, 50.0, 100.0]

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    datasets = [ds for ds in CLF_DATASETS if ds[0] in args.datasets]
    for ds_name, ds_id, ds_source in datasets:
        X, y, feature_names = get_clean_dataset(ds_id, ds_source)

        # Separate 50 samples for computing Shapley values
        X, X_shap, y, y_shap = train_test_split(
            X, y, test_size=50, random_state=0, stratify=y
        )

        results: List[pd.DataFrame] = []
        if args.n_jobs == 1:
            prog = trange(args.n_replications, desc=f"Running {ds_name}")
            for _ in prog:
                results.append(run_single_replication(
                    feature_names,
                    args.shrink_modes,
                    lambdas,
                    X,
                    y,
                    X_shap,
                    y_shap,
                ))
        else:
            results = joblib.Parallel(n_jobs=args.n_jobs, verbose=10)(
                joblib.delayed(run_single_replication)(
                    feature_names,
                    args.shrink_modes,
                    lambdas,
                    X,
                    y,
                    X_shap,
                    y_shap,
                )
                for _ in range(args.n_replications)
            )  # type: ignore
        
        # Add replication column to each dataframe
        for i, df in enumerate(results):
            df["replication"] = i
        
        all_results = pd.concat(results)
        all_results.to_csv(os.path.join(args.out_dir, f"{ds_name}.csv"), index=False)