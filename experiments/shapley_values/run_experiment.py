import sys
sys.path.append("../../")

import argparse
import os
from shap import TreeExplainer, summary_plot
from imodels.util.data_util import get_clean_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import numpy as np
from aughs import ShrinkageClassifier
import matplotlib.pyplot as plt


def train_model(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    bal_acc = balanced_accuracy_score(y_test, model.predict(X_test))
    return model, bal_acc

def generate_summary_plot(X_train, X_test, feature_names, model):
    explainer = TreeExplainer(model, X_train)
    shap_values = np.array(explainer.shap_values(X_test))
    summary_plot(shap_values[0, ...], features=X_test, feature_names=feature_names, show=False)
    fig = plt.gcf()
    return fig



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="plot")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    X, y, feature_names = get_clean_dataset("breast_cancer", "imodels")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Generate summary plot for Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf, bal_acc = train_model(X_train, X_test, y_train, y_test, rf)
    fig = generate_summary_plot(X_train, X_test, feature_names, rf)
    fig.suptitle(f"Random Forest (Bal. Acc: {bal_acc:.2f})")
    fig.savefig(os.path.join(args.output_dir, "summary_plot_rf.png"))
    plt.clf()

    for shrink_mode in ["hs", "hs_entropy", "hs_log_cardinality"]:
        for lmb in [1, 10, 100, 1000]:
            # Generate summary plot for AugHS
            shrink = ShrinkageClassifier(shrink_mode=shrink_mode, lmb=lmb)
            shrink, bal_acc = train_model(X_train, X_test, y_train, y_test, shrink)
            fig = generate_summary_plot(X_train, X_test, feature_names, shrink.estimator_)
            fig.suptitle(f"Shrinkage {shrink_mode} ($\lambda$={lmb}) (Bal. Acc: {bal_acc:.2f}%)")
            fig.savefig(os.path.join(args.output_dir, f"summary_plot_shrink_{shrink_mode}_{lmb}.png"))
            plt.clf()