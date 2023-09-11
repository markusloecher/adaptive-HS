import argparse
import os
from tqdm import tqdm
from aughs import ShrinkageClassifier, ShrinkageRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from imodels.util.data_util import get_clean_dataset
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


CLF_DATASETS = [
    ("heart", "heart", "imodels"),
    ("breast-cancer", "breast_cancer", "imodels"),
    ("haberman", "haberman", "imodels"),
    ("ionosphere", "ionosphere", "pmlb"),
    ("diabetes-clf", "diabetes", "pmlb"),
    ("german", "german", "pmlb"),
    ("juvenile", "juvenile_clean", "imodels"),
    ("recidivism", "compas_two_year_clean", "imodels"),
]

REG_DATASETS = [
    ("friedman1", "friedman1", "synthetic"),
    ("friedman3", "friedman3", "synthetic"),
    ("diabetes-reg", "diabetes", "sklearn"),
    ("abalone", "183", "openml"),
    ("satellite-image", "294_satellite_image", "pmlb"),
    #("california-housing", "california_housing", "sklearn"),
]


def plot_results(results, lambdas, ds_name, problem_type, shrink_modes):
    # Put results into a single dict of np arrays
    # shrink_mode -> [n_lambdas, n_replications]
    scores = {sm: [] for sm in shrink_modes}
    no_shrinkage_scores = []
    for result in results:
        for shrink_mode in shrink_modes:
            scores[shrink_mode].append(result[0][shrink_mode])
        no_shrinkage_scores.append(result[1])
    for sm in scores:
        scores[sm] = np.array(scores[sm])
    no_shrinkage_scores = np.array(no_shrinkage_scores)

    # Plot results
    plt.figure(figsize=(10, 6))
    for sm in scores:
        avg = np.mean(scores[sm], axis=0)
        std = np.std(scores[sm], axis=0)
        n = scores[sm].shape[0]
        conf = 1.96 * std / np.sqrt(n)
        plt.plot(lambdas, avg, label=sm)
        plt.fill_between(
            lambdas,
            avg - conf,
            avg + conf,
            alpha=0.2,
        )
    avg_no_shrinkage = np.mean(no_shrinkage_scores).repeat(len(lambdas))
    std_no_shrinkage = np.std(no_shrinkage_scores).repeat(len(lambdas))
    n_no_shrinkage = no_shrinkage_scores.shape[0]
    conf_no_shrinkage = 1.96 * std_no_shrinkage / np.sqrt(n_no_shrinkage)

    plt.plot(lambdas, avg_no_shrinkage, label="No shrinkage")
    plt.fill_between(
        lambdas,
        avg_no_shrinkage - conf_no_shrinkage,
        avg_no_shrinkage + conf_no_shrinkage,
        alpha=0.2,
    )

    plt.legend()
    plt.xscale("log")
    plt.xlabel("Lambda")
    if problem_type == "regression":
        plt.ylabel("MSE")
    elif problem_type == "classification":
        plt.ylabel("ROC AUC")
    plt.title(ds_name)
    plt.savefig(f"{args.out_dir}/{ds_name}.png")
    plt.close()


def single_rep(
    X, y, base_estimator, shrink_modes, lambdas, problem_type, score_fn
):
    """
    Performs a single replication of the experiment for a given dataset.
    Parallelization will be over the replications, so this function does not
    use parallelization, as it corresponds to a single process.
    """
    # shrink_mode -> [n_lambdas]
    scores = {sm: [] for sm in shrink_modes}
    for shrink_mode in shrink_modes:
        # Make the model
        if problem_type == "regression":
            model = ShrinkageRegressor(
                base_estimator=base_estimator, shrink_mode=shrink_mode
            )
        elif problem_type == "classification":
            model = ShrinkageClassifier(
                base_estimator=base_estimator, shrink_mode=shrink_mode
            )
        else:
            raise ValueError("Unknown problem type")

        # Split the data
        train_index, test_index = train_test_split(
            np.arange(X.shape[0]), test_size=0.2
        )

        # Fit the model
        model.fit(X[train_index], y[train_index])

        # Compute scores for all lambdas
        shrink_mode_scores = []
        for lmb in lambdas:
            model.reshrink(shrink_mode=shrink_mode, lmb=lmb)
            shrink_mode_scores.append(
                score_fn(y[test_index], model.predict(X[test_index]))
            )
        scores[shrink_mode] = shrink_mode_scores

    # Also get the score of a non-shrinkage model
    base_estimator.fit(X[train_index], y[train_index])
    no_shrinkage_score = score_fn(
        y[test_index], base_estimator.predict(X[test_index])
    )

    return scores, no_shrinkage_score


if __name__ == "__main__":
    all_shrinkage_types = [
        "hs",
        "hs_entropy",
        "hs_log_cardinality",
        "hs_permutation",
        "hs_global_permutation"
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type", type=str, default="dt", choices=["dt", "rf"]
    )
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--n-replications", type=int, default=8)
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument(
        "--shrink-modes",
        type=str,
        nargs="+",
        choices=all_shrinkage_types,
        default=all_shrinkage_types,
    )
    args = parser.parse_args()

    lambdas = [0.0, 0.1, 1.0, 10.0, 25.0, 50.0, 100.0]
    np.seterr(all="raise")

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    ##################
    # CLASSIFICATION #
    ##################
    print("Running classification experiments...")
    for ds_name, ds_id, ds_source in tqdm(CLF_DATASETS):
        if args.model_type == "dt":
            base_estimator = DecisionTreeClassifier()
        elif args.model_type == "rf":
            base_estimator = RandomForestClassifier()

        X, y, feature_names = get_clean_dataset(ds_id, ds_source)

        # results is a list of results from single_rep:
        # [({shrink_mode -> n_lambdas}, no_shrinkage_score)]
        results = Parallel(n_jobs=args.n_jobs, verbose=0)(
            delayed(single_rep)(
                X,
                y,
                base_estimator,
                args.shrink_modes,
                lambdas,
                "classification",
                roc_auc_score,
            )
            for _ in range(args.n_replications)
        )
        plot_results(
            results, lambdas, ds_name, "classification", args.shrink_modes
        )

    ##############
    # REGRESSION #
    ##############
    print("Running regression experiments...")
    for ds_name, ds_id, ds_source in tqdm(REG_DATASETS):
        if args.model_type == "dt":
            base_estimator = DecisionTreeRegressor()
        elif args.model_type == "rf":
            base_estimator = RandomForestRegressor()

        X, y, feature_names = get_clean_dataset(ds_id, ds_source)

        # results is a list of results from single_rep:
        # [({shrink_mode -> n_lambdas}, no_shrinkage_score)]
        results = Parallel(n_jobs=args.n_jobs, verbose=0)(
            delayed(single_rep)(
                X,
                y,
                base_estimator,
                args.shrink_modes,
                lambdas,
                "regression",
                mean_squared_error,
            )
            for _ in range(args.n_replications)
        )
        plot_results(results, lambdas, ds_name, "regression", args.shrink_modes)
