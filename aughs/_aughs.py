from abc import abstractmethod
from copy import deepcopy

import numpy as np
import scipy
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.estimator_checks import check_estimator

from ._alpha import compute_alpha, compute_global_alpha
from ._util import normalize_value, check_fit_arguments

class ShrinkageEstimator(BaseEstimator):
    def __init__(
        self,
        base_estimator: BaseEstimator | None = None,
        shrink_mode: str = "hs",
        lmb: float = 1,
        random_state=None,
        compute_all_values=True,
    ):
        self.base_estimator = base_estimator
        self.shrink_mode = shrink_mode
        self.lmb = lmb
        self.random_state = random_state
        self.compute_all_values = compute_all_values

    @abstractmethod
    def get_default_estimator(self):
        raise NotImplemented

    def _compute_node_values_rec(
        self,
        dt,
        X_cond,
        y_cond,
        node=0,
        entropies=None,
        log_cardinalities=None,
        alphas=None,
        global_alphas=None,
        X_shuffled=None,
        X_shuffled_cond=None,
    ):
        """
        Compute the entropy of each node in the tree.
        These values are used in the entropy-based shrinkage.
        Note that if shrink_mode is "hs_permutation", the returned values
        are not technically "entropies", but rather the $\alpha$ values
        used in the permutation-based shrinkage.

        Parameters
        ----------
        dt : DecisionTreeClassifier or DecisionTreeRegressor
            The decision tree to compute the node values for.
        X_cond : np.ndarray
            The training data conditioned on the current node.
        y_cond : np.ndarray
            The labels conditioned on the current node.
        node : int
            The current node.
        entropies : np.ndarray
            The entropies of the nodes in the tree.
        log_cardinalities : np.ndarray
            The log cardinalities of the nodes in the tree.
        alphas : np.ndarray
            The $\alpha$ values of the nodes in the tree.
        global_alphas : np.ndarray
            The $\alpha$ values of the nodes in the tree, computed globally
            (i.e. not conditioned on the current node).
        X_shuffled : np.ndarray
            Full training data with the columns shuffled.
            Columns are shuffled separately.
        X_shuffled_cond : np.ndarray
            Training data conditioned on the current node, with the columns
            shuffled.
        """
        left = dt.tree_.children_left[node]
        right = dt.tree_.children_right[node]
        feature = dt.tree_.feature[node]
        threshold = dt.tree_.threshold[node]
        criterion = dt.criterion

        # Initialize arrays if not provided
        if entropies is None:
            num_nodes = len(dt.tree_.n_node_samples)
            entropies = np.zeros(num_nodes)
            log_cardinalities = np.zeros(num_nodes)
            alphas = np.zeros(num_nodes)
            global_alphas = np.zeros(num_nodes)

            # Shuffle the columns of X separately
            X_shuffled = X_cond.copy()
            for i in range(X_shuffled.shape[1]):
                X_shuffled[:, i] = np.random.permutation(X_shuffled[:, i])
            X_shuffled_cond = X_shuffled.copy()

        assert log_cardinalities is not None
        assert alphas is not None
        assert global_alphas is not None
        assert X_shuffled is not None
        assert X_shuffled_cond is not None

        # If not leaf node, compute entropy, cardinality, and alpha of the node
        if not (left == -1 and right == -1):
            split_feature = X_cond[:, feature]
            _, counts = np.unique(split_feature, return_counts=True)

            entropies[node] = np.maximum(scipy.stats.entropy(counts), 1)
            log_cardinalities[node] = np.log(len(counts))
            alphas[node] = compute_alpha(
                X_cond, y_cond, feature, threshold, criterion
            )
            global_alphas[node] = compute_global_alpha(
                X_cond, X_shuffled_cond, y_cond, feature, threshold, criterion
            )

            left_rows = split_feature <= threshold
            X_train_left = X_cond[left_rows]
            X_train_right = X_cond[~left_rows]
            y_train_left = y_cond[left_rows]
            y_train_right = y_cond[~left_rows]
            X_shuffled_left = X_shuffled_cond[left_rows]
            X_shuffled_right = X_shuffled_cond[~left_rows]

            # Recursively compute entropy and cardinality of the children
            self._compute_node_values_rec(
                dt,
                X_train_left,
                y_train_left,
                left,
                entropies,
                log_cardinalities,
                alphas,
                global_alphas,
                X_shuffled,
                X_shuffled_left,
            )
            self._compute_node_values_rec(
                dt,
                X_train_right,
                y_train_right,
                right,
                entropies,
                log_cardinalities,
                alphas,
                global_alphas,
                X_shuffled,
                X_shuffled_right,
            )
        return entropies, log_cardinalities, alphas, global_alphas

    def _compute_node_values(self, X, y):
        self.entropies_ = []
        self.log_cardinalities_ = []
        self.alphas_ = []
        self.global_alphas_ = []
        if hasattr(self.estimator_, "estimators_"):  # Random Forest
            for estimator in self.estimator_.estimators_:
                (
                    entropies,
                    log_cardinalities,
                    alphas,
                    global_alphas,
                ) = self._compute_node_values_rec(estimator, X, y)
                self.entropies_.append(entropies)
                self.log_cardinalities_.append(log_cardinalities)
                self.alphas_.append(alphas)
        else:  # Single tree
            (
                entropies,
                log_cardinalities,
                alphas,
                global_alphas,
            ) = self._compute_node_values_rec(self.estimator_, X, y)
            self.entropies_.append(entropies)
            self.log_cardinalities_.append(log_cardinalities)
            self.alphas_.append(alphas)
            self.global_alphas_.append(global_alphas)

    def _shrink_tree_rec(
        self,
        dt,
        dt_idx,
        node=0,
        parent_node=None,
        parent_val=None,
        cum_sum=None,
    ):
        """
        Go through the tree and shrink contributions recursively
        """
        left = dt.tree_.children_left[node]
        right = dt.tree_.children_right[node]
        parent_num_samples = dt.tree_.n_node_samples[parent_node]
        value = normalize_value(dt, node)

        # cum_sum contains the value of the telescopic sum
        # If root: initialize cum_sum to the value of the root node
        if parent_node is None:
            cum_sum = value
        else:
            # If not root: update cum_sum based on the value of the current
            # node and the parent node
            reg = 1
            if self.shrink_mode == "hs":
                # Classic hierarchical shrinkage
                reg = 1 + (self.lmb / parent_num_samples)
            else:
                # Adaptive shrinkage
                if self.shrink_mode == "hs_entropy":
                    node_value = self.entropies_[dt_idx][parent_node]
                elif self.shrink_mode == "hs_log_cardinality":
                    node_value = self.log_cardinalities_[dt_idx][parent_node]
                elif self.shrink_mode == "hs_permutation":
                    node_value = 1 / self.alphas_[dt_idx][parent_node]
                elif self.shrink_mode == "hs_global_permutation":
                    node_value = 1 / self.global_alphas_[dt_idx][parent_node]
                else:
                    raise ValueError(
                        f"Unknown shrink mode: {self.shrink_mode}"
                    )
                reg = 1 + (self.lmb * node_value / parent_num_samples)
            cum_sum += (value - parent_val) / reg

        # Set the value of the node to the value of the telescopic sum
        assert not np.isnan(cum_sum).any(), "Cumulative sum is NaN"
        dt.tree_.value[node, :, :] = cum_sum
        # Update the impurity of the node
        dt.tree_.impurity[node] = 1 - np.sum(np.power(cum_sum, 2))
        assert not np.isnan(dt.tree_.impurity[node]), "Impurity is NaN"

        # If not leaf: recurse
        if not (left == -1 and right == -1):
            self._shrink_tree_rec(
                dt, dt_idx, left, node, value, cum_sum.copy()
            )
            self._shrink_tree_rec(
                dt, dt_idx, right, node, value, cum_sum.copy()
            )

    def fit(self, X, y, **kwargs):
        X, y = self._validate_arguments(
            X, y, kwargs.pop("feature_names", None)
        )

        if self.base_estimator is not None:
            self.estimator_ = clone(self.base_estimator)
        else:
            self.estimator_ = self.get_default_estimator()

        self.estimator_.set_params(random_state=self.random_state)
        self.estimator_.fit(X, y, **kwargs)

        # Save a copy of the original estimator
        self.orig_estimator_ = deepcopy(self.estimator_)

        # Compute node values (entropy, log cardinality, alpha)
        self._compute_node_values(X, y)

        # Apply hierarchical shrinkage
        self.shrink()
        return self

    def shrink(self):
        if hasattr(self.estimator_, "estimators_"):  # Random Forest
            for i, estimator in enumerate(self.estimator_.estimators_):
                self._shrink_tree_rec(estimator, i)
        else:  # Single tree
            self._shrink_tree_rec(self.estimator_, 0)

    def reshrink(self, shrink_mode=None, lmb=None):
        if shrink_mode is not None:
            self.shrink_mode = shrink_mode
        if lmb is not None:
            self.lmb = lmb

        # Reset the estimator to the original one
        self.estimator_ = deepcopy(self.orig_estimator_)

        # Apply hierarchical shrinkage
        self.shrink()

    def _validate_arguments(self, X, y, feature_names):
        if self.shrink_mode not in [
            "hs",
            "hs_entropy",
            "hs_log_cardinality",
            "hs_permutation",
            "hs_global_permutation",
        ]:
            raise ValueError("Invalid choice for shrink_mode")
        X, y, feature_names = check_fit_arguments(
            X, y, feature_names=feature_names
        )
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = feature_names
        return X, y

    def predict(self, X, *args, **kwargs):
        check_is_fitted(self)
        return self.estimator_.predict(X, *args, **kwargs)

    def score(self, X, y, *args, **kwargs):
        check_is_fitted(self)
        return self.estimator_.score(X, y, *args, **kwargs)


class ShrinkageClassifier(ShrinkageEstimator, ClassifierMixin):
    def get_default_estimator(self):
        return DecisionTreeClassifier()

    def fit(self, X, y, **kwargs):
        super().fit(X, y, **kwargs)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X, *args, **kwargs):
        check_is_fitted(self)
        return self.estimator_.predict_proba(X, *args, **kwargs)


class ShrinkageRegressor(ShrinkageEstimator, RegressorMixin):
    def get_default_estimator(self):
        return DecisionTreeRegressor()


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    print("Testing ShrinkageClassifier and ShrinkageRegressor...")
    check_estimator(ShrinkageClassifier(RandomForestClassifier()))
    check_estimator(ShrinkageRegressor(RandomForestRegressor()))
