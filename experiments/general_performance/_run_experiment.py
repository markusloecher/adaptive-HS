import pandas as pd
import os
from tqdm import tqdm
from imodels.util.data_util import get_clean_dataset
from joblib import Parallel, delayed
from typing import List, Tuple, Callable, Dict
from _run_single_replication import run_single_replication
from _util import TreeBasedModel


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
        prog.set_postfix({"dataset": ds_name})
        X, y, _ = get_clean_dataset(ds_id, ds_source)

        if ds_name == "red-wine":
            y = y.astype(int)

        # results is a list of results from single_rep:
        # [{shrink_mode -> n_lambdas}]
        results: List[Dict[str, List[float]]] = []
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
        records = []
        score_key = "MSE" if problem_type == "regression" else "ROC AUC"
        for i, result in enumerate(results):
            for shrink_mode in result:
                for j, lmb in enumerate(lambdas):
                    records.append(
                        {
                            "replication": i,
                            "shrink_mode": shrink_mode,
                            "lambda": lmb,
                            score_key: result[shrink_mode][j],
                        }
                    )
        pd.DataFrame(records).to_csv(
            os.path.join(out_path, f"{ds_name}.csv"), index=False
        )

