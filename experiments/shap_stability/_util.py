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
    "hs",
    "hs_entropy",
    "hs_log_cardinality",
    "hs_permutation",
    "hs_global_permutation",
]