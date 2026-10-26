"""Model code shared by train.py, recommend.py and make_demo_model.py."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier

from candidates import LABEL_NAMES as CANDS
from features import FEATURE_COLS

SCORE_COLS = [f"{c}__score" for c in CANDS]


def fit_rankers(train, n_jobs=-1, n_estimators=200):
    """Train all rankers on a labelled dataframe (needs FEATURE_COLS, 'label' and the score columns)."""
    X, y, S = train[FEATURE_COLS], train["label"], train[SCORE_COLS]
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=10, min_samples_leaf=3,
                                 random_state=0, n_jobs=n_jobs).fit(X, y)
    reg = RandomForestRegressor(n_estimators=n_estimators, max_depth=12, min_samples_leaf=3,
                                random_state=0, n_jobs=n_jobs).fit(X, S)
    tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=0).fit(X, y)
    static = S.mean().to_numpy()
    return {"clf": clf, "reg": reg, "tree": tree, "static": static}


def class_proba_matrix(model, X):
    """Class probabilities aligned to CANDS (0 for candidates that never won in training)."""
    P = np.zeros((len(X), len(CANDS)))
    proba = model.predict_proba(X)
    for j, c in enumerate(model.classes_):
        P[:, CANDS.index(c)] = proba[:, j]
    return P
