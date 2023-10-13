import os
from typing import Callable, Dict, List, Tuple

import pandas as pd
from _run_single_replication import run_single_replication
from _util import TreeBasedModel
from imodels.util.data_util import get_clean_dataset
from joblib import Parallel, delayed
from numpy import typing as npt
from tqdm import tqdm


def run_experiment(
    datasets: List[Tuple[str, str, str]],
    base_estimator: TreeBasedModel,
    shrink_modes: List[str],
    lambdas: List[float],
    problem_type: str,
    score_fn: Callable,
    n_jobs: int,
    n_replications: int,
    out_dir: str,
    exp_name: str,
):
    prog = tqdm(datasets)
    prog.set_description(f"Running {exp_name}")

    out_path = os.path.join(out_dir, exp_name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for ds_name, ds_id, ds_source in prog:
        ds_path = os.path.join(out_path, ds_name)
        if not os.path.isdir(ds_path):
            os.makedirs(ds_path)
        prog.set_postfix({"dataset": ds_name})
        X, y, _ = get_clean_dataset(ds_id, ds_source)

        if ds_name == "red-wine":
            y = y.astype(int)

        # results is a list of results from single_rep:
        # [{shrink_mode -> NDArray(len(lambdas), num_trees)}]
        results: List[Dict[str, npt.NDArray]] = []
        if n_jobs == 1:
            for _ in range(n_replications):
                results.append(
                    run_single_replication(
                        X,
                        y,
                        base_estimator,
                        shrink_modes,
                        lambdas,
                        problem_type,
                        score_fn,
                    )
                )
        else:
            results = Parallel(
                n_jobs=n_jobs, verbose=0
            )(
                delayed(run_single_replication)(
                    X,
                    y,
                    base_estimator,
                    shrink_modes,
                    lambdas,
                    problem_type,
                    score_fn,
                )
                for _ in range(n_replications)
            )  # type: ignore

        # Save results to CSV:
        # replication, shrink_mode, lambda, score
        score_key = "R2" if problem_type == "regression" else "ROC AUC"
        for i, result in enumerate(results):
            for shrink_mode, sm_result in result.items():
                records = []
                for j, lmb in enumerate(lambdas):
                    for k in range(sm_result.shape[1]):
                        records.append(
                            {
                                "shrink_mode": shrink_mode,
                                "lambda": lmb,
                                "num_trees": k + 1,
                                score_key: sm_result[j,k],
                            }
                        )
                pd.DataFrame(records).to_csv(
                    os.path.join(ds_path, f"rep_{i}.csv"), index=False
                )

