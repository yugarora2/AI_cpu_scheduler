"""
Core logic behind the app (no user interface in here, so it can be tested and reused):

  validate()     check the process table the user typed in
  rank_table()   ask the trained model how promising each scheduler looks
  recommend()    keep the top-k, simulate only those on the real processes,
                 score them and pick the best
"""
import os
import numpy as np
import pandas as pd
import joblib

from config import WEIGHTS
from candidates import CANDIDATES, LABEL_NAMES, NAMES, CLAIRVOYANT
from features import extract_features, FEATURE_COLS
from modeling import class_proba_matrix

MODEL_PATHS = ["models/ranker.joblib", "ranker.joblib"]
TRAIN_N_RANGE = (10, 60)          # number of processes the model was trained on
TRAIN_MAX_BURST = 150
MAX_TOTAL_BURST = 200_000         # safety limit: the tick-by-tick schedulers get slow beyond this


# ----------------------------------------------------------------------------- model
def find_model_path():
    for p in MODEL_PATHS:
        if os.path.exists(p):
            return p
    return None


def list_models():
    """Every model file the app can use: models/*.joblib (newest first), plus ./ranker.joblib for old setups."""
    import glob
    paths = glob.glob("models/*.joblib")
    if os.path.exists("ranker.joblib"):
        paths.append("ranker.joblib")
    return sorted(paths, key=os.path.getmtime, reverse=True)


def model_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def load_bundle(src):
    """src = file path or file-like object. Only load model files you created yourself."""
    bundle = joblib.load(src)
    if list(bundle["cands"]) != list(LABEL_NAMES):
        raise ValueError("This model was trained with a different list of candidate schedulers than "
                         "candidates.py now defines. Retrain the model with the current code.")
    if list(bundle["features"]) != list(FEATURE_COLS):
        raise ValueError("This model was trained with different features than features.py now defines. "
                         "Retrain the model with the current code.")
    return bundle


def version_warnings(bundle):
    """Compare library versions used for training with the ones installed here."""
    import sklearn
    meta = bundle.get("meta", {})
    msgs = []
    if meta.get("sklearn") and meta["sklearn"] != sklearn.__version__:
        msgs.append(f"Model trained with scikit-learn {meta['sklearn']}, but {sklearn.__version__} is installed here. "
                    f"If predictions look wrong, install scikit-learn=={meta['sklearn']}.")
    return msgs


# ----------------------------------------------------------------------------- input
def validate(df):
    """df: columns pid, at, bt, priority (any extra/blank rows allowed).
    Returns (processes, errors, warnings)."""
    errors, warnings = [], []
    df = df.copy()
    df = df.dropna(how="all")
    df = df[~(df["pid"].isna() & df["at"].isna() & df["bt"].isna())]

    if len(df) == 0:
        return [], ["The table is empty. Add at least 3 processes."], []

    procs = []
    for i, r in df.reset_index(drop=True).iterrows():
        row = i + 1
        pid = "" if pd.isna(r["pid"]) else str(r["pid"]).strip()
        if not pid:
            errors.append(f"Row {row}: process name is empty.")
            continue
        vals = {}
        for key, label in (("at", "arrival time"), ("bt", "burst time"), ("priority", "priority")):
            v = r[key]
            if pd.isna(v):
                if key == "priority":
                    v = 0
                else:
                    errors.append(f"Row {row} ({pid}): {label} is missing.")
                    continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                errors.append(f"Row {row} ({pid}): {label} must be a number.")
                continue
            if f != int(f):
                errors.append(f"Row {row} ({pid}): {label} must be a whole number.")
                continue
            vals[key] = int(f)
        if len(vals) < 3:
            continue
        if vals["at"] < 0:
            errors.append(f"Row {row} ({pid}): arrival time cannot be negative.")
        if vals["bt"] < 1:
            errors.append(f"Row {row} ({pid}): burst time must be at least 1.")
        procs.append({"pid": pid, "at": vals["at"], "bt": vals["bt"], "priority": vals["priority"]})

    pids = [p["pid"] for p in procs]
    dup = sorted({x for x in pids if pids.count(x) > 1})
    if dup:
        errors.append(f"Process names must be unique. Repeated: {', '.join(dup)}.")
    if len(procs) < 3:
        errors.append("Enter at least 3 processes (the model's features need a few jobs to work with).")
    if sum(p["bt"] for p in procs) > MAX_TOTAL_BURST:
        errors.append(f"Total burst time is above {MAX_TOTAL_BURST:,}; that is too large for this tool.")
    if errors:
        return [], errors, warnings

    n = len(procs)
    lo, hi = TRAIN_N_RANGE
    if n < lo or n > hi:
        warnings.append(f"You entered {n} processes, but the model was trained on workloads of {lo} to {hi} processes. "
                        f"The shortlist is an extrapolation and may be less reliable. The simulation results "
                        f"themselves are exact.")
    if max(p["bt"] for p in procs) > TRAIN_MAX_BURST:
        warnings.append(f"A burst time is above {TRAIN_MAX_BURST}, the largest value seen in training.")
    return procs, errors, warnings


# ----------------------------------------------------------------------------- model ranking
def rank_table(bundle, procs):
    """One row per candidate: vote share (classifier) and predicted score (regressor)."""
    feats = extract_features(procs)
    X = pd.DataFrame([feats])[FEATURE_COLS]
    m = bundle["models"]
    table = pd.DataFrame({
        "scheduler": LABEL_NAMES,
        "vote_share": class_proba_matrix(m["clf"], X)[0],
        "predicted_score": m["reg"].predict(X)[0],
    })
    return feats, table


def order_table(table, method):
    """method: 'classifier' (highest vote share first) or 'regressor' (lowest predicted score first)."""
    if method == "classifier":
        t = table.sort_values(["vote_share", "predicted_score"], ascending=[False, True], kind="stable")
    else:
        t = table.sort_values(["predicted_score", "vote_share"], ascending=[True, False], kind="stable")
    t = t.reset_index(drop=True)
    t.insert(0, "rank", np.arange(1, len(t) + 1))
    return t


# ----------------------------------------------------------------------------- scoring
def score_set(results, weights=None):
    """Same scoring rule as training, scaled WITHIN the given set of schedulers (0 = best of them)."""
    weights = weights or WEIGHTS
    names = list(results)
    total = {n: 0.0 for n in names}
    for metric, w in weights.items():
        vals = np.array([results[n][metric] for n in names], float)
        lo, hi = vals.min(), vals.max()
        norm = (vals - lo) / (hi - lo) if hi > lo else np.zeros_like(vals)
        for n, v in zip(names, norm):
            total[n] += w * v
    return total


METRIC_COLS = ["avg_wt", "avg_tat", "avg_rt", "max_wt", "switches"]


def metrics_frame(results, scores):
    rows = []
    for n, r in results.items():
        rows.append({"scheduler": n, "avg_wt": r["avg_wt"], "avg_tat": r["avg_tat"], "avg_rt": r["avg_rt"],
                     "max_wt": r["max_wt"], "switches": r["switches"], "score": scores.get(n, np.nan)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main entry
def recommend(bundle, procs, k=3, method="classifier", verify=True):
    feats, table = rank_table(bundle, procs)
    ranking = order_table(table, method)
    k = max(1, min(int(k), len(ranking)))
    shortlist = list(ranking["scheduler"][:k])

    results = {n: CANDIDATES[n](procs) for n in shortlist}          # simulate ONLY the shortlist
    scores = score_set(results)
    best_score = min(scores.values())
    winner = next(n for n in shortlist if scores[n] <= best_score + 1e-12)   # ties: higher-ranked wins

    out = {"features": feats, "ranking": ranking, "shortlist": shortlist, "results": results,
           "scores": scores, "winner": winner, "k": k, "method": method, "verify": None}

    if verify:                                                       # demonstration only: run everything
        all_results = {n: CANDIDATES[n](procs) for n in NAMES}
        pool = {n: all_results[n] for n in LABEL_NAMES}
        pool_scores = score_set(pool)
        pool_best = min(pool_scores.values())
        best_names = [n for n in LABEL_NAMES if pool_scores[n] <= pool_best + 1e-12]
        out["verify"] = {
            "all_results": all_results, "pool_scores": pool_scores, "best_names": best_names,
            "shortlist_has_best": any(n in best_names for n in shortlist),
            "winner_regret": pool_scores[winner] - pool_best,
        }
    return out
