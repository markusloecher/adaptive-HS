from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from _simulate_data import simulate_data
from imodels.util.data_util import get_clean_dataset

TreeBasedModel = (
    DecisionTreeClassifier
    | DecisionTreeRegressor
    | RandomForestClassifier
    | RandomForestRegressor
)

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
    ("geographical-music", "4544", "openml"),
    ("red-wine", "40691", "openml"),
    ("abalone", "183", "openml"),
    ("satellite-image", "294_satellite_image", "pmlb"),
    ("california-housing", "california_housing", "sklearn"),
]

SHRINKAGE_TYPES = [
    "no_shrinkage",
    "hs",
    "hs_entropy",
    "hs_log_cardinality",
    "hs_permutation",
    "hs_global_permutation",
]

EXPERIMENTS = ["classification_dt", "classification_rf", "regression"]

def get_data_bench_sim(ds_id, ds_source, N, p):
    if ds_source == "sim":
        X, y, rlvFtrs = simulate_data(N, p)
        X = X.to_numpy()
        return X,y,rlvFtrs
    else:
        X, y, _ = get_clean_dataset(ds_id, ds_source)
        return X,y,_ 