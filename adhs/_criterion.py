import numpy as np
import scipy

def _gini(y):
    # Computes the Gini impurity of a vector of labels:
    # 1 - \sum_i (p_i)^2
    _, counts = np.unique(y, return_counts=True)
    return 1 - np.sum(np.power(counts / len(y), 2))


def _entropy(y):
    _, counts = np.unique(y, return_counts=True)
    return scipy.stats.entropy(counts)


_CRITERION_FNS = {
    "gini": _gini,
    "entropy": _entropy,
    "log_loss": _entropy,  # Log loss and entropy are the same
    "squared_error": lambda y: np.var(y),
    "absolute_error": lambda y: np.mean(np.abs(y - np.median(y))),
}