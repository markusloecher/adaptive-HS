import numpy as np


def simulate_data(n_samples: int, relevance: float):
    X = np.zeros((n_samples, 5))
    X[:, 0] = np.random.normal(0, 1, n_samples)
    n_categories = [2, 4, 10, 20]
    for i in range(1, 5):
        X[:, i] = np.random.choice(
            a=n_categories[i - 1],
            size=n_samples,
            p=np.ones(n_categories[i - 1]) / n_categories[i - 1],
        )
    y = np.zeros(n_samples)
    y[X[:, 1] == 0] = np.random.binomial(1, 0.5 - relevance, np.sum(X[:, 1] == 0))
    y[X[:, 1] == 1] = np.random.binomial(1, 0.5 + relevance, np.sum(X[:, 1] == 1))
    return X, y
