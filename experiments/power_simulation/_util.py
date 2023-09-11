from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

TreeBasedModel = (
    DecisionTreeClassifier
    | DecisionTreeRegressor
    | RandomForestClassifier
    | RandomForestRegressor
)

SHRINKAGE_TYPES = [
    "hs",
    "hs_entropy",
    "hs_log_cardinality",
    "hs_permutation",
    "hs_global_permutation",
]

EXPERIMENTS = ["strobl_rf", "strobl_dt"]