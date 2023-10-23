import argparse
from _run_experiment import run_experiment
from _util import SHRINKAGE_TYPES, EXPERIMENTS
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


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

    lmb = 100.0
    relevances = [0.0, 0.05, 0.1, 0.15, 0.2]

    if "strobl_rf" in args.experiments:
        run_experiment(
            lmb,
            relevances,
            args.shrink_modes,
            RandomForestClassifier(),
            args.n_jobs,
            args.n_replications,
            args.out_dir,
            "strobl_rf",
        )
    if "strobl_dt" in args.experiments:
        run_experiment(
            lmb,
            relevances,
            args.shrink_modes,
            DecisionTreeClassifier(),
            args.n_jobs,
            args.n_replications,
            args.out_dir,
            "strobl_dt",
        )
