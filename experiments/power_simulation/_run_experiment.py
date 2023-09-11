from typing import Tuple, List, Dict
from tqdm import trange
import joblib
from _util import TreeBasedModel
from _run_single_replication import run_single_replication
import pandas as pd
import os


def run_experiment(
    lambdas: List[float],
    relevances: List[float],
    shrink_modes: List[str],
    base_estimator: TreeBasedModel,
    n_jobs: int,
    n_replications: int,
    out_dir: str,
    exp_name: str,
):
    out_path = os.path.join(out_dir, exp_name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    results: List[Tuple[List[Dict[str, float]], List[Dict[str, float]]]] = []
    if n_jobs != 1:
        results = joblib.Parallel(n_jobs=n_jobs, verbose=10)(
            joblib.delayed(run_single_replication)(
                lambdas, relevances, shrink_modes, base_estimator
            )
            for _ in range(n_replications)
        )  # type: ignore
    else:
        results = []
        for _ in trange(n_replications):
            results.append(
                run_single_replication(
                    lambdas,
                    relevances,
                    shrink_modes,
                    base_estimator,
                )
            )

    importances = pd.concat([pd.DataFrame.from_records(r[0]) for r in results])
    scores = pd.concat([pd.DataFrame.from_records(r[1]) for r in results])

    importances.to_csv(
        os.path.join(out_path, "importances.csv"),
        index=True,
        index_label="index",
    )
    scores.to_csv(
        os.path.join(out_path, "scores.csv"), index=True, index_label="index"
    )
